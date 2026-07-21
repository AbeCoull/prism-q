use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::SparseBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::backend::{Backend, max_statevector_qubits};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};

#[cfg(any(feature = "gpu", feature = "distributed"))]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

#[cfg(feature = "distributed")]
use crate::backend::distributed_statevector::DistributedStatevectorBackend;
#[cfg(feature = "distributed")]
use crate::distributed::DistributedContext;

use super::{RunOutcome, try_backend_probabilities};

pub(super) const AUTO_MPS_BOND_DIM: usize = 256;

pub(super) const MAX_AUTO_T_COUNT_EXACT: usize = 18;

pub(super) const MAX_AUTO_T_COUNT_APPROX: usize = 28;

pub(super) const MAX_AUTO_T_COUNT_SHOTS: usize = 40;

pub(super) const MAX_STABILIZER_RANK_QUBITS: usize = 25;

pub(super) const AUTO_APPROX_MAX_TERMS: usize = 8192;

pub(super) const MIN_QUBITS_FOR_SPD_AUTO: usize = 12;

pub(super) const AUTO_SPD_MAX_TERMS: usize = 65536;

pub(super) const MIN_FACTORED_STABILIZER_QUBITS: usize = 128;

pub(super) const MIN_BLOCK_FOR_FACTORED_STAB: usize = 16;

#[inline]
pub(super) fn stabilizer_rank_budget(num_qubits: usize) -> usize {
    let log2n = if num_qubits >= 2 {
        (num_qubits as f64).log2().ceil() as usize * 2
    } else {
        0
    };
    num_qubits.saturating_sub(log2n)
}

// GPU crossover threshold and its env override live in `crate::gpu` so users
// can introspect them without depending on internal dispatch plumbing. The
// dispatch layer calls `crate::gpu::min_qubits()` directly; there is no
// private duplicate.

/// Automatically select the optimal backend based on circuit analysis.
///
/// Decision tree:
/// 1. No entangling gates        → ProductState (O(n))
/// 2. All Clifford gates         → Stabilizer (O(n²))
/// 3. Clifford+T, t ≤ 12        → StabilizerRank (O(2^t · n²))
/// 4. Above memory limit:
///    a. Sparse-friendly         → Sparse (O(k) where k = non-zero amplitudes)
///    b. Otherwise               → MPS (bounded bond dimension)
/// 5. Otherwise                  → Statevector (exact, general-purpose)
#[derive(Debug, Clone)]
pub enum BackendKind {
    Auto,
    Statevector,
    Stabilizer,
    Sparse,
    Mps {
        max_bond_dim: usize,
    },
    ProductState,
    TensorNetwork,
    Factored,
    StabilizerRank,
    FactoredStabilizer,
    StochasticPauli {
        num_samples: usize,
    },
    DeterministicPauli {
        epsilon: f64,
        max_terms: usize,
    },
    /// Automatic backend selection with GPU acceleration opted in.
    ///
    /// Makes the same shape-based routing decisions as [`BackendKind::Auto`],
    /// but when the selected family (per sub-block, after subsystem
    /// decomposition) has a device capability row and the block clears the
    /// qubit-count crossover with VRAM to spare, that block runs on the
    /// supplied context. Every other choice, and every block that fails the
    /// crossover or VRAM check, runs on the identical CPU path Auto would
    /// take. Device paths resolve soft: an allocation that fails after the
    /// VRAM check degrades to the host, so a missing, unfit, or racing device
    /// stays on CPU rather than erroring.
    ///
    /// Acceleration reaches every entry point through one resolution
    /// mechanism: single runs, terminal shot and counts sampling, expectation
    /// values, temporal-Clifford tails, and non-Pauli noisy trajectories.
    ///
    /// The context is user-supplied and is never acquired implicitly.
    #[cfg(feature = "gpu")]
    AutoGpu {
        context: Arc<GpuContext>,
    },
    /// Statevector backed by a CUDA GPU execution context.
    ///
    /// Circuits (or decomposed sub-blocks) with fewer than
    /// [`crate::gpu::min_qubits()`] qubits (tunable via
    /// `PRISM_GPU_MIN_QUBITS`, default [`crate::gpu::MIN_QUBITS_DEFAULT`])
    /// transparently fall back to the host statevector path, since
    /// small states do not survive PCIe and launch-latency overhead.
    /// Larger circuits allocate a device-resident state and route gate
    /// application through GPU kernels.
    ///
    /// Compose with `simulate(...).backend(...).seed(...).run()` to get fusion
    /// plus independent-subsystem decomposition; each sub-block is evaluated
    /// against the crossover independently.
    #[cfg(feature = "gpu")]
    StatevectorGpu {
        context: Arc<GpuContext>,
    },
    /// Stabilizer backend backed by a CUDA GPU tableau.
    ///
    /// Circuits (or decomposed sub-blocks) with fewer than
    /// [`crate::gpu::stabilizer_min_qubits()`] qubits (tunable via
    /// `PRISM_STABILIZER_GPU_MIN_QUBITS`, default
    /// [`crate::gpu::STABILIZER_MIN_QUBITS_DEFAULT`]) fall back to the CPU
    /// stabilizer path. The GPU path routes gate application to device
    /// kernels. Measurement and reset stay on device, while probabilities and
    /// export-style helpers still read back to the CPU algorithms.
    ///
    /// Compose with `simulate(...).backend(...).seed(...).run()` to pick up
    /// independent-subsystem decomposition; non-Clifford circuits are rejected
    /// at dispatch time with the same error shape as [`BackendKind::Stabilizer`].
    #[cfg(feature = "gpu")]
    StabilizerGpu {
        context: Arc<GpuContext>,
    },
    /// Exact state vector distributed across `2^p` ranks via a
    /// [`DistributedContext`]. The low `n - p` qubits are simulated locally with
    /// the standard SIMD kernels; the top `p` qubits select the rank.
    ///
    /// Results are independent of the rank count. With a single rank the path is
    /// identical to [`BackendKind::Statevector`].
    #[cfg(feature = "distributed")]
    StatevectorDistributed {
        context: Arc<DistributedContext>,
    },
}

impl BackendKind {
    /// Whether this kind uses automatic shape-based routing. True for
    /// [`BackendKind::Auto`] and its GPU-accelerated sibling
    /// [`BackendKind::AutoGpu`], which make identical routing decisions and
    /// differ only in whether a cleared statevector or stabilizer block runs on
    /// the device.
    #[inline]
    pub(crate) fn is_auto(&self) -> bool {
        match self {
            BackendKind::Auto => true,
            #[cfg(feature = "gpu")]
            BackendKind::AutoGpu { .. } => true,
            _ => false,
        }
    }

    pub fn supports_noisy_per_shot(&self) -> bool {
        !matches!(
            self,
            BackendKind::StabilizerRank
                | BackendKind::StochasticPauli { .. }
                | BackendKind::DeterministicPauli { .. }
        )
    }

    pub fn supports_general_noise(&self) -> bool {
        match self {
            BackendKind::Auto
            | BackendKind::Statevector
            | BackendKind::Sparse
            | BackendKind::Mps { .. }
            | BackendKind::ProductState
            | BackendKind::Factored => true,
            #[cfg(feature = "gpu")]
            BackendKind::AutoGpu { .. } | BackendKind::StatevectorGpu { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn is_stabilizer_family(&self) -> bool {
        matches!(
            self,
            BackendKind::Stabilizer | BackendKind::FactoredStabilizer
        ) || {
            #[cfg(feature = "gpu")]
            {
                matches!(self, BackendKind::StabilizerGpu { .. })
            }
            #[cfg(not(feature = "gpu"))]
            {
                false
            }
        }
    }

    pub(crate) fn general_noise_backend_names() -> &'static str {
        #[cfg(feature = "gpu")]
        {
            "Auto, Statevector, StatevectorGpu, Sparse, Mps, ProductState, or Factored"
        }
        #[cfg(not(feature = "gpu"))]
        {
            "Auto, Statevector, Sparse, Mps, ProductState, or Factored"
        }
    }
}

pub(super) fn validate_explicit_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    if kind.is_stabilizer_family() && !circuit.is_clifford_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "stabilizer".into(),
            reason: "circuit contains non-Clifford gates".into(),
        });
    }
    match kind {
        BackendKind::ProductState if circuit.has_entangling_gates() => {
            return Err(PrismError::IncompatibleBackend {
                backend: "productstate".into(),
                reason: "circuit contains entangling gates".into(),
            });
        }
        BackendKind::StabilizerRank if !circuit.has_t_gates() => {
            return Err(PrismError::IncompatibleBackend {
                backend: "stabilizer_rank".into(),
                reason: "circuit has no T gates; use Stabilizer instead".into(),
            });
        }
        _ => {}
    }
    Ok(())
}

/// Simulator family, independent of how it was selected (auto routing or an
/// explicit kind) and of where it executes (host or device).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Family {
    ProductState,
    Stabilizer,
    Sparse,
    Mps,
    Factored,
    FactoredStabilizer,
    TensorNetwork,
    Statevector,
}

fn select_auto_backend_choice(circuit: &Circuit, has_partial_independence: bool) -> Family {
    if !circuit.has_entangling_gates() {
        Family::ProductState
    } else if circuit.is_clifford_only() {
        Family::Stabilizer
    } else if circuit.num_qubits > max_statevector_qubits() {
        if circuit.is_sparse_friendly() {
            Family::Sparse
        } else {
            Family::Mps
        }
    } else if has_partial_independence {
        Family::Factored
    } else {
        Family::Statevector
    }
}

pub(super) fn auto_selects_cpu_statevector(
    circuit: &Circuit,
    has_partial_independence: bool,
) -> bool {
    matches!(
        select_auto_backend_choice(circuit, has_partial_independence),
        Family::Statevector
    )
}

/// Execution target for a resolved family: host, or a device context with a
/// resolved failure mode. `soft` builds fall back to the host if device
/// allocation fails at `init`; hard builds surface the error.
#[derive(Clone)]
pub(super) enum Accel {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu {
        context: Arc<GpuContext>,
        soft: bool,
    },
}

/// Device eligibility for one family: the qubit crossover and the VRAM-fit
/// predicate, owned together. Families without device kernels are `None` rows
/// in [`gpu_capability`]; adding a device path for a family means adding its
/// row there, not touching dispatch.
#[cfg(feature = "gpu")]
struct GpuCapability {
    min_qubits: fn() -> usize,
    fits: fn(&Arc<GpuContext>, usize) -> bool,
}

/// Families with a `gpu_capability` row. Keep in sync with the match below;
/// a new device family needs an entry in both.
#[cfg(feature = "gpu")]
const GPU_CAPABLE_FAMILIES: [Family; 2] = [Family::Statevector, Family::Stabilizer];

#[cfg(feature = "gpu")]
fn gpu_capability(family: Family) -> Option<GpuCapability> {
    match family {
        Family::Statevector => Some(GpuCapability {
            min_qubits: crate::gpu::min_qubits,
            fits: |ctx, n| ctx.fits_statevector_with_scratch(n).unwrap_or(false),
        }),
        Family::Stabilizer => Some(GpuCapability {
            min_qubits: crate::gpu::stabilizer_min_qubits,
            fits: |ctx, n| ctx.fits_tableau(n).unwrap_or(false),
        }),
        Family::ProductState
        | Family::Sparse
        | Family::Mps
        | Family::Factored
        | Family::FactoredStabilizer
        | Family::TensorNetwork => None,
    }
}

/// The one decision point for CPU vs GPU execution of a family.
///
/// `AutoGpu` requires the family's capability row, the qubit crossover, and the
/// VRAM-fit gate, and resolves soft so an allocation race still degrades to the
/// host. An explicit GPU kind applies only the crossover for its own family and
/// resolves hard: the user asked for the device, so an unfit device errors
/// loudly. Every other kind, and every family without a capability row,
/// resolves to the host.
///
/// The verdict (including the VRAM query) is intended to be taken once per
/// user-level call and reused across shots; soft-mode init fallback covers
/// VRAM shrinking after the fact.
#[cfg(feature = "gpu")]
pub(super) fn accel_for(kind: &BackendKind, family: Family, num_qubits: usize) -> Accel {
    let Some((context, soft)) = gpu_request(kind, family) else {
        return Accel::Cpu;
    };
    let Some(cap) = gpu_capability(family) else {
        return Accel::Cpu;
    };
    if num_qubits < (cap.min_qubits)() {
        return Accel::Cpu;
    }
    if soft && !(cap.fits)(context, num_qubits) {
        return Accel::Cpu;
    }
    Accel::Gpu {
        context: context.clone(),
        soft,
    }
}

/// Which (kind, family) pairs request device execution, and whether the
/// request resolves soft. Shared by [`accel_for`] and [`may_resolve_to_gpu`]
/// so the kind-to-family matching lives in one place.
#[cfg(feature = "gpu")]
fn gpu_request(kind: &BackendKind, family: Family) -> Option<(&Arc<GpuContext>, bool)> {
    match (kind, family) {
        (BackendKind::AutoGpu { context }, _) => Some((context, true)),
        (BackendKind::StatevectorGpu { context }, Family::Statevector) => Some((context, false)),
        (BackendKind::StabilizerGpu { context }, Family::Stabilizer) => Some((context, false)),
        _ => None,
    }
}

/// Whether this kind could resolve any family to the device at the given
/// width. Over-approximates [`accel_for`]: it applies only the qubit
/// crossover and skips the VRAM-fit gate, so the verdict cannot flip between
/// a scheduling decision and the later per-block resolution. Multi-block
/// drivers consult it before spreading work across threads; GPU backends
/// share one CUDA stream per context and must not execute concurrently.
#[cfg(feature = "gpu")]
pub(super) fn may_resolve_to_gpu(kind: &BackendKind, num_qubits: usize) -> bool {
    GPU_CAPABLE_FAMILIES.into_iter().any(|family| {
        gpu_request(kind, family).is_some()
            && gpu_capability(family).is_some_and(|cap| num_qubits >= (cap.min_qubits)())
    })
}

#[cfg(not(feature = "gpu"))]
pub(super) fn accel_for(_kind: &BackendKind, _family: Family, _num_qubits: usize) -> Accel {
    Accel::Cpu
}

pub(super) fn build_statevector(accel: &Accel, seed: u64) -> StatevectorBackend {
    match accel {
        Accel::Cpu => StatevectorBackend::new(seed),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: true,
        } => StatevectorBackend::new(seed).with_gpu_auto(context.clone()),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: false,
        } => StatevectorBackend::new(seed).with_gpu(context.clone()),
    }
}

fn build_stabilizer(accel: &Accel, seed: u64) -> StabilizerBackend {
    match accel {
        Accel::Cpu => StabilizerBackend::new(seed),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: true,
        } => StabilizerBackend::new(seed).with_gpu_auto(context.clone()),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: false,
        } => StabilizerBackend::new(seed).with_gpu(context.clone()),
    }
}

/// Resolved family plus acceleration for one user-level call. `build` is
/// cheap enough to call once per shot or per trajectory: a constructor, a
/// `Box::new`, and at most one `Arc` clone. All circuit analysis and every
/// driver VRAM query happen earlier, in [`resolve`].
#[derive(Clone)]
pub(super) enum BackendPlan {
    ProductState,
    Sparse,
    TensorNetwork,
    Factored,
    FactoredStabilizer,
    Mps {
        max_bond_dim: usize,
    },
    Stabilizer {
        accel: Accel,
    },
    Statevector {
        accel: Accel,
    },
    #[cfg(feature = "distributed")]
    Distributed(Arc<DistributedContext>),
}

impl BackendPlan {
    pub(super) fn build(&self, seed: u64) -> Box<dyn Backend> {
        match self {
            BackendPlan::ProductState => Box::new(ProductStateBackend::new(seed)),
            BackendPlan::Sparse => Box::new(SparseBackend::new(seed)),
            BackendPlan::TensorNetwork => Box::new(TensorNetworkBackend::new(seed)),
            BackendPlan::Factored => Box::new(crate::backend::factored::FactoredBackend::new(seed)),
            BackendPlan::FactoredStabilizer => {
                Box::new(crate::backend::factored_stabilizer::FactoredStabilizerBackend::new(seed))
            }
            BackendPlan::Mps { max_bond_dim } => Box::new(MpsBackend::new(seed, *max_bond_dim)),
            BackendPlan::Stabilizer { accel } => Box::new(build_stabilizer(accel, seed)),
            BackendPlan::Statevector { accel } => Box::new(build_statevector(accel, seed)),
            #[cfg(feature = "distributed")]
            BackendPlan::Distributed(context) => {
                Box::new(DistributedStatevectorBackend::new(context.clone(), seed))
            }
        }
    }

    pub(super) fn is_gpu(&self) -> bool {
        match self {
            BackendPlan::Stabilizer { accel } | BackendPlan::Statevector { accel } => {
                !matches!(accel, Accel::Cpu)
            }
            _ => false,
        }
    }

    /// Whether the fusion pipeline should synthesize fused-matrix gates for
    /// this plan. Tableau-based families reject non-Clifford fused matrices;
    /// every dense or factored representation accepts them.
    pub(super) fn supports_fused(&self) -> bool {
        !matches!(
            self,
            BackendPlan::Stabilizer { .. } | BackendPlan::FactoredStabilizer
        )
    }

    #[cfg(test)]
    pub(super) fn family(&self) -> Family {
        match self {
            BackendPlan::ProductState => Family::ProductState,
            BackendPlan::Sparse => Family::Sparse,
            BackendPlan::TensorNetwork => Family::TensorNetwork,
            BackendPlan::Factored => Family::Factored,
            BackendPlan::FactoredStabilizer => Family::FactoredStabilizer,
            BackendPlan::Mps { .. } => Family::Mps,
            BackendPlan::Stabilizer { .. } => Family::Stabilizer,
            BackendPlan::Statevector { .. } => Family::Statevector,
            #[cfg(feature = "distributed")]
            BackendPlan::Distributed(_) => Family::Statevector,
        }
    }

    #[cfg(test)]
    pub(super) fn accel(&self) -> &Accel {
        match self {
            BackendPlan::Stabilizer { accel } | BackendPlan::Statevector { accel } => accel,
            _ => &Accel::Cpu,
        }
    }
}

/// Whole-call routing decision produced by [`resolve`]. Backend execution is
/// described by a buildable [`BackendPlan`]; the remaining variants are the
/// non-backend engines (stabilizer rank, Pauli propagation) that callers
/// handle outside the `Backend` world.
pub(super) enum ExecutionPlan {
    Backend(BackendPlan),
    StabilizerRank,
    StochasticPauli { num_samples: usize },
    DeterministicPauli { epsilon: f64, max_terms: usize },
}

pub(super) fn plan_for_family(
    kind: &BackendKind,
    family: Family,
    num_qubits: usize,
) -> BackendPlan {
    match family {
        Family::ProductState => BackendPlan::ProductState,
        Family::Sparse => BackendPlan::Sparse,
        Family::TensorNetwork => BackendPlan::TensorNetwork,
        Family::Factored => BackendPlan::Factored,
        Family::FactoredStabilizer => BackendPlan::FactoredStabilizer,
        Family::Mps => BackendPlan::Mps {
            max_bond_dim: AUTO_MPS_BOND_DIM,
        },
        Family::Stabilizer => BackendPlan::Stabilizer {
            accel: accel_for(kind, Family::Stabilizer, num_qubits),
        },
        Family::Statevector => BackendPlan::Statevector {
            accel: accel_for(kind, Family::Statevector, num_qubits),
        },
    }
}

/// Resolve `kind` against `circuit` into an [`ExecutionPlan`], exactly once
/// per user-level call. Auto kinds run the shape-based decision tree; explicit
/// kinds map 1:1 onto their family. The CPU-vs-GPU verdict, including the
/// VRAM-fit query, is taken here through [`accel_for`] and reused across
/// shots; soft-mode init fallback covers VRAM shrinking after resolution.
pub(super) fn resolve(
    kind: &BackendKind,
    circuit: &Circuit,
    has_partial_independence: bool,
) -> ExecutionPlan {
    let family = match kind {
        BackendKind::Auto => select_auto_backend_choice(circuit, has_partial_independence),
        #[cfg(feature = "gpu")]
        BackendKind::AutoGpu { .. } => {
            select_auto_backend_choice(circuit, has_partial_independence)
        }
        BackendKind::Statevector => Family::Statevector,
        BackendKind::Stabilizer => Family::Stabilizer,
        BackendKind::Sparse => Family::Sparse,
        BackendKind::Mps { max_bond_dim } => {
            return ExecutionPlan::Backend(BackendPlan::Mps {
                max_bond_dim: *max_bond_dim,
            });
        }
        BackendKind::ProductState => Family::ProductState,
        BackendKind::TensorNetwork => Family::TensorNetwork,
        BackendKind::Factored => Family::Factored,
        BackendKind::FactoredStabilizer => Family::FactoredStabilizer,
        BackendKind::StabilizerRank => return ExecutionPlan::StabilizerRank,
        BackendKind::StochasticPauli { num_samples } => {
            return ExecutionPlan::StochasticPauli {
                num_samples: *num_samples,
            };
        }
        BackendKind::DeterministicPauli { epsilon, max_terms } => {
            return ExecutionPlan::DeterministicPauli {
                epsilon: *epsilon,
                max_terms: *max_terms,
            };
        }
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => Family::Statevector,
        #[cfg(feature = "gpu")]
        BackendKind::StabilizerGpu { .. } => Family::Stabilizer,
        #[cfg(feature = "distributed")]
        BackendKind::StatevectorDistributed { context } => {
            return ExecutionPlan::Backend(BackendPlan::Distributed(context.clone()));
        }
    };
    ExecutionPlan::Backend(plan_for_family(kind, family, circuit.num_qubits))
}

pub(super) fn resolve_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    has_partial_independence: bool,
) -> BackendPlan {
    match resolve(kind, circuit, has_partial_independence) {
        ExecutionPlan::Backend(plan) => plan,
        _ => unreachable!("non-backend dispatch should be handled by caller"),
    }
}

#[inline]
pub(super) fn min_clifford_prefix_gates(num_qubits: usize) -> usize {
    (num_qubits * 2).max(16)
}

pub(super) fn has_temporal_clifford_opportunity(kind: &BackendKind, circuit: &Circuit) -> bool {
    if !kind.is_auto() {
        return false;
    }
    if circuit.num_qubits > max_statevector_qubits() {
        return false;
    }
    let min_gates = min_clifford_prefix_gates(circuit.num_qubits);
    let mut prefix_gates = 0;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, .. } => {
                if !gate.is_clifford() {
                    break;
                }
                prefix_gates += 1;
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. } => break,
            Instruction::Barrier { .. } => {}
        }
    }
    prefix_gates >= min_gates && prefix_gates < circuit.instructions.len()
}

/// Temporal-Clifford execution split into a seed-independent plan and a
/// per-seed run, so shot loops split and fuse the circuit once instead of
/// once per shot. The stabilizer prefix runs on the host tableau; the
/// crossover data in [`gpu_capability`] makes a device prefix unreachable
/// here (temporal-Clifford requires fitting the dense statevector, far below
/// the stabilizer crossover).
pub(super) struct TemporalCliffordPlan {
    prefix: Circuit,
    fused_tail: Circuit,
    tail_num_classical_bits: usize,
    tail_accel: Accel,
}

pub(super) fn plan_temporal_clifford(
    kind: &BackendKind,
    circuit: &Circuit,
) -> Option<TemporalCliffordPlan> {
    if !kind.is_auto() {
        return None;
    }
    if circuit.num_qubits > max_statevector_qubits() {
        return None;
    }
    let (prefix, tail) = circuit.clifford_prefix_split()?;
    if prefix.gate_count() < min_clifford_prefix_gates(circuit.num_qubits) {
        return None;
    }
    let tail_accel = accel_for(kind, Family::Statevector, circuit.num_qubits);
    let tail_num_classical_bits = tail.num_classical_bits;
    let fused_tail = match &tail_accel {
        Accel::Cpu => crate::circuit::fusion::fuse_circuit(&tail, true).into_owned(),
        #[cfg(feature = "gpu")]
        Accel::Gpu { .. } => {
            let expanded = crate::circuit::expand_qft_blocks(&tail);
            crate::circuit::fusion::fuse_circuit(&expanded, true).into_owned()
        }
    };
    Some(TemporalCliffordPlan {
        prefix,
        fused_tail,
        tail_num_classical_bits,
        tail_accel,
    })
}

pub(super) fn run_temporal_clifford(
    plan: &TemporalCliffordPlan,
    seed: u64,
    want_probabilities: bool,
) -> Result<RunOutcome> {
    let mut stab = StabilizerBackend::new(seed);
    stab.init(plan.prefix.num_qubits, plan.prefix.num_classical_bits)?;
    stab.enable_lazy_destab();
    for inst in &plan.prefix.instructions {
        stab.apply(inst)?;
    }

    let state = stab.export_statevector()?;

    let mut sv = build_statevector(&plan.tail_accel, seed);
    sv.init_from_state(state, plan.tail_num_classical_bits)?;
    for inst in &plan.fused_tail.instructions {
        sv.apply(inst)?;
    }

    let probabilities = if want_probabilities {
        try_backend_probabilities(&sv)?
    } else {
        None
    };

    Ok(RunOutcome {
        classical_bits: sv.classical_results().to_vec(),
        probabilities,
    })
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_crossover_tests {
    use super::*;
    use crate::gates::Gate;

    fn stub_kind() -> BackendKind {
        BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    fn run_query(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
        crate::sim::simulate(circuit).backend(kind).seed(seed).run()
    }

    /// The builder GPU shortcut must compose identically to constructing the
    /// variant manually. Uses the stub context at a small circuit so crossover
    /// fires and proves the composition is side-effect equivalent.
    #[test]
    fn builder_gpu_wraps_statevector_gpu_variant() {
        let ctx = GpuContext::stub_for_tests();
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);

        let direct = crate::sim::simulate(&circuit)
            .gpu(ctx.clone())
            .seed(42)
            .run()
            .expect("builder GPU shortcut must honor crossover and route to CPU");
        let manual = crate::sim::simulate(&circuit)
            .backend(stub_kind())
            .seed(42)
            .run()
            .expect("manual variant reference");

        let dp = direct.probabilities.expect("direct probs").to_vec();
        let mp = manual.probabilities.expect("manual probs").to_vec();
        assert_eq!(dp, mp);
    }

    /// A 4q circuit is far below the default 14q threshold. If the dispatch
    /// layer were to build a GPU backend anyway, `GpuState::new` on the stub
    /// context would return `BackendUnsupported`. Success proves the
    /// crossover in `select_dispatch` is routing small circuits to the host
    /// path.
    #[test]
    fn small_circuit_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);

        let result = run_query(stub_kind(), &circuit, 42)
            .expect("stub context must not be touched for a 4q circuit");
        let probs = result
            .probabilities
            .expect("probabilities missing")
            .to_vec();

        let mut expected = [0.0_f64; 16];
        expected[0b0000] = 0.25;
        expected[0b0011] = 0.25;
        expected[0b1100] = 0.25;
        expected[0b1111] = 0.25;
        for (i, (p, e)) in probs.iter().zip(&expected).enumerate() {
            assert!((p - e).abs() < 1e-10, "p[{i}] = {p}, expected {e}");
        }
    }

    /// `independent_bell_pairs(8)` spans 16 qubits but decomposes into 8
    /// independent 2q blocks. With `BackendKind::StatevectorGpu`, each
    /// sub-block is below the 14q threshold and must route to CPU. If
    /// decomposition failed to fire, the 16q monolithic path would attempt
    /// `GpuState::new` through the stub and return `BackendUnsupported`.
    /// Success here proves decomposition survives across the GPU dispatch.
    #[test]
    fn decomposable_16q_circuit_runs_per_block_on_cpu() {
        let circuit = crate::circuits::independent_bell_pairs(8);
        assert_eq!(circuit.num_qubits, 16);

        let cpu = run_query(BackendKind::Statevector, &circuit, 42).expect("cpu baseline");
        let gpu = run_query(stub_kind(), &circuit, 42).expect("stub must stay out of the way");

        let cpu_p = cpu.probabilities.expect("cpu probs").to_vec();
        let gpu_p = gpu.probabilities.expect("gpu probs").to_vec();
        assert_eq!(cpu_p.len(), gpu_p.len());
        for (i, (c, g)) in cpu_p.iter().zip(gpu_p.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-10,
                "prob[{i}] cpu={c}, gpu={g}, diff={}",
                (c - g).abs()
            );
        }
    }

    fn stabilizer_stub_kind() -> BackendKind {
        BackendKind::StabilizerGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    /// A 4q Clifford circuit is far below the stabilizer GPU threshold, so the
    /// stub context must never be touched. Produces the same measurement bits
    /// as a plain CPU stabilizer run.
    #[test]
    fn stabilizer_gpu_small_circuit_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 4);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        circuit.add_measure(2, 2);
        circuit.add_measure(3, 3);

        let cpu_run = run_query(BackendKind::Stabilizer, &circuit, 42).expect("cpu baseline");
        let gpu_run = run_query(stabilizer_stub_kind(), &circuit, 42)
            .expect("stub must stay out of the way for small circuits");
        assert_eq!(cpu_run.classical_bits, gpu_run.classical_bits);
    }

    /// Non-Clifford circuits are rejected at dispatch time with the same error
    /// shape as `BackendKind::Stabilizer`.
    #[test]
    fn stabilizer_gpu_rejects_non_clifford_at_dispatch() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_gate(Gate::T, &[0]);
        let err = run_query(stabilizer_stub_kind(), &circuit, 42).unwrap_err();
        assert!(matches!(err, PrismError::IncompatibleBackend { .. }));
    }

    fn auto_gpu_stub_kind() -> BackendKind {
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    fn assert_probs_match(a: &RunOutcome, b: &RunOutcome) {
        let ap = a.probabilities.as_ref().expect("probs a").to_vec();
        let bp = b.probabilities.as_ref().expect("probs b").to_vec();
        assert_eq!(ap.len(), bp.len());
        for (i, (x, y)) in ap.iter().zip(bp.iter()).enumerate() {
            assert!((x - y).abs() < 1e-10, "prob[{i}]: {x} vs {y}");
        }
    }

    /// A small Clifford circuit selects the stabilizer choice, which sits below
    /// the stabilizer GPU crossover, so `AutoGpu` builds a CPU stabilizer and
    /// never touches the stub. Results match the plain `Auto` path.
    #[test]
    fn auto_gpu_small_clifford_routes_to_cpu() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42)
            .expect("stub must stay out of the way below the stabilizer crossover");
        assert_probs_match(&cpu, &gpu);
    }

    /// `independent_bell_pairs(8)` spans 16 qubits but decomposes into 8
    /// independent 2q blocks, each below the statevector crossover. Every block
    /// stays on CPU under `AutoGpu`; if decomposition failed, the monolithic 16q
    /// path would still hit the VRAM gate and fall back rather than error.
    #[test]
    fn auto_gpu_decomposable_16q_runs_per_block_on_cpu() {
        let circuit = crate::circuits::independent_bell_pairs(8);
        assert_eq!(circuit.num_qubits, 16);

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42).expect("stub stays out of the way");
        assert_probs_match(&cpu, &gpu);
    }

    /// A 16q entangled non-Clifford circuit selects the statevector choice and
    /// clears the 14q crossover, so `AutoGpu` reaches the GPU decision. Because
    /// the stub cannot report VRAM, the fits check fails closed and the block
    /// runs on CPU: same results as `Auto`, no error. The explicit
    /// `StatevectorGpu` path has no VRAM gate, so the same circuit touches the
    /// stub and surfaces `BackendUnsupported`, isolating the added fallback.
    #[test]
    fn auto_gpu_large_block_falls_back_to_cpu_on_stub() {
        let mut circuit = Circuit::new(16, 0);
        for q in 0..16 {
            circuit.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..15 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }

        let cpu = run_query(BackendKind::Auto, &circuit, 42).expect("cpu auto baseline");
        let gpu = run_query(auto_gpu_stub_kind(), &circuit, 42)
            .expect("stub VRAM query fails closed, so AutoGpu must fall back to CPU without error");
        assert_probs_match(&cpu, &gpu);

        let explicit = BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        };
        assert!(matches!(
            run_query(explicit, &circuit, 42).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }
}

#[cfg(all(test, feature = "gpu"))]
mod accel_tests {
    use super::*;

    fn stub() -> Arc<GpuContext> {
        GpuContext::stub_for_tests()
    }

    fn is_cpu(accel: &Accel) -> bool {
        matches!(accel, Accel::Cpu)
    }

    #[test]
    fn auto_gpu_statevector_below_crossover_is_cpu() {
        let kind = BackendKind::AutoGpu { context: stub() };
        let n = crate::gpu::min_qubits() - 1;
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n)));
    }

    /// Above the crossover the stub cannot report VRAM, so the fit gate fails
    /// closed and the soft path resolves to the host.
    #[test]
    fn auto_gpu_statevector_fits_fails_closed_on_stub() {
        let kind = BackendKind::AutoGpu { context: stub() };
        let n = crate::gpu::min_qubits() + 2;
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n)));
    }

    #[test]
    fn auto_gpu_cpu_only_families_stay_cpu() {
        let kind = BackendKind::AutoGpu { context: stub() };
        for family in [
            Family::ProductState,
            Family::Sparse,
            Family::Mps,
            Family::Factored,
            Family::FactoredStabilizer,
            Family::TensorNetwork,
        ] {
            assert!(is_cpu(&accel_for(&kind, family, 1 << 10)), "{family:?}");
        }
    }

    #[test]
    fn explicit_statevector_gpu_is_hard_at_crossover_and_cpu_below() {
        let kind = BackendKind::StatevectorGpu { context: stub() };
        let n = crate::gpu::min_qubits();
        assert!(matches!(
            accel_for(&kind, Family::Statevector, n),
            Accel::Gpu { soft: false, .. }
        ));
        assert!(is_cpu(&accel_for(&kind, Family::Statevector, n - 1)));
    }

    #[test]
    fn explicit_stabilizer_gpu_is_hard_at_crossover_and_cpu_below() {
        let kind = BackendKind::StabilizerGpu { context: stub() };
        let n = crate::gpu::stabilizer_min_qubits();
        assert!(matches!(
            accel_for(&kind, Family::Stabilizer, n),
            Accel::Gpu { soft: false, .. }
        ));
        assert!(is_cpu(&accel_for(&kind, Family::Stabilizer, n - 1)));
    }

    /// An explicit GPU kind accelerates only its own family; any other family
    /// resolves to the host regardless of size.
    #[test]
    fn explicit_kind_other_family_is_cpu() {
        let sv = BackendKind::StatevectorGpu { context: stub() };
        assert!(is_cpu(&accel_for(&sv, Family::Stabilizer, 1 << 20)));
        let stab = BackendKind::StabilizerGpu { context: stub() };
        assert!(is_cpu(&accel_for(&stab, Family::Statevector, 1 << 20)));
    }

    #[test]
    fn cpu_kinds_are_cpu_everywhere() {
        for family in [Family::Statevector, Family::Stabilizer] {
            assert!(is_cpu(&accel_for(&BackendKind::Auto, family, 1 << 20)));
            assert!(is_cpu(&accel_for(
                &BackendKind::Statevector,
                family,
                1 << 20
            )));
        }
    }

    /// `init_from_state` on a soft GPU backend degrades to the host when the
    /// device upload fails, preserving the supplied amplitudes.
    #[test]
    fn init_from_state_soft_falls_back_to_host_on_stub() {
        use num_complex::Complex64;
        let mut sv = StatevectorBackend::new(42).with_gpu_auto(stub());
        let amp = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let state = vec![amp, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), amp];
        sv.init_from_state(state.clone(), 0).unwrap();
        let exported = sv.export_statevector().unwrap();
        for (e, s) in exported.iter().zip(&state) {
            assert!((e - s).norm() < 1e-12);
        }
    }

    #[test]
    fn init_from_state_hard_errors_on_stub() {
        use num_complex::Complex64;
        let mut sv = StatevectorBackend::new(42).with_gpu(stub());
        let err = sv
            .init_from_state(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)], 0)
            .unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }
}

/// Table-driven coverage of [`resolve`]: every backend kind against every
/// circuit shape class, asserting the resolved family and execution target.
/// This is the contract that all simulator families dispatch uniformly on
/// both targets; extend it when a family gains a capability row.
#[cfg(test)]
mod dispatch_matrix_tests {
    use super::*;
    use crate::gates::Gate;

    fn product(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        c
    }

    fn clifford(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        c.add_gate(Gate::H, &[0]);
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        c
    }

    fn dense(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        c
    }

    /// One qubit past the statevector memory cap, or `None` when memory
    /// detection failed and the cap is disabled (no width is oversize then).
    fn oversize_qubits() -> Option<usize> {
        let cap = max_statevector_qubits();
        if cap >= usize::BITS as usize {
            eprintln!(
                "SKIP: statevector qubit cap disabled on this host; skipping oversize checks"
            );
            return None;
        }
        Some(cap + 1)
    }

    fn oversize_sparse(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::T, &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        c
    }

    fn oversize_dense(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        c
    }

    fn resolved(kind: &BackendKind, circuit: &Circuit, hpi: bool) -> BackendPlan {
        match resolve(kind, circuit, hpi) {
            ExecutionPlan::Backend(plan) => plan,
            _ => panic!("expected a backend plan"),
        }
    }

    fn assert_cpu_family(kind: &BackendKind, circuit: &Circuit, hpi: bool, family: Family) {
        let plan = resolved(kind, circuit, hpi);
        assert_eq!(plan.family(), family, "kind {kind:?}");
        assert!(
            matches!(plan.accel(), Accel::Cpu),
            "kind {kind:?} family {family:?} must resolve to the host"
        );
    }

    fn auto_kinds() -> Vec<BackendKind> {
        #[cfg(feature = "gpu")]
        {
            vec![
                BackendKind::Auto,
                BackendKind::AutoGpu {
                    context: GpuContext::stub_for_tests(),
                },
            ]
        }
        #[cfg(not(feature = "gpu"))]
        {
            vec![BackendKind::Auto]
        }
    }

    /// Auto and AutoGpu make identical family choices across every circuit
    /// shape class; on the stub context every choice resolves to the host
    /// (small circuits by crossover, large ones by the fail-closed VRAM gate).
    #[test]
    fn auto_family_matrix() {
        let oversize = oversize_qubits();
        for kind in auto_kinds() {
            assert_cpu_family(&kind, &product(6), false, Family::ProductState);
            assert_cpu_family(&kind, &clifford(6), false, Family::Stabilizer);
            assert_cpu_family(&kind, &clifford(16), false, Family::Stabilizer);
            assert_cpu_family(&kind, &dense(8), false, Family::Statevector);
            assert_cpu_family(&kind, &dense(16), false, Family::Statevector);
            assert_cpu_family(&kind, &dense(8), true, Family::Factored);
            if let Some(n) = oversize {
                assert_cpu_family(&kind, &oversize_sparse(n), false, Family::Sparse);
                assert_cpu_family(&kind, &oversize_dense(n), false, Family::Mps);
            }
        }
    }

    #[test]
    fn auto_oversize_mps_uses_auto_bond_dim() {
        let Some(n) = oversize_qubits() else { return };
        for kind in auto_kinds() {
            let plan = resolved(&kind, &oversize_dense(n), false);
            assert!(matches!(
                plan,
                BackendPlan::Mps {
                    max_bond_dim: AUTO_MPS_BOND_DIM
                }
            ));
        }
    }

    /// Explicit CPU kinds map 1:1 onto their family regardless of circuit
    /// shape, always on the host.
    #[test]
    fn explicit_cpu_kind_matrix() {
        let circuit = dense(6);
        let cases = [
            (BackendKind::Statevector, Family::Statevector),
            (BackendKind::Stabilizer, Family::Stabilizer),
            (BackendKind::Sparse, Family::Sparse),
            (BackendKind::ProductState, Family::ProductState),
            (BackendKind::TensorNetwork, Family::TensorNetwork),
            (BackendKind::Factored, Family::Factored),
            (BackendKind::FactoredStabilizer, Family::FactoredStabilizer),
        ];
        for (kind, family) in cases {
            assert_cpu_family(&kind, &circuit, false, family);
        }

        let plan = resolved(&BackendKind::Mps { max_bond_dim: 77 }, &circuit, false);
        assert!(matches!(plan, BackendPlan::Mps { max_bond_dim: 77 }));
    }

    #[test]
    fn non_backend_kinds_resolve_to_their_engines() {
        let circuit = dense(6);
        assert!(matches!(
            resolve(&BackendKind::StabilizerRank, &circuit, false),
            ExecutionPlan::StabilizerRank
        ));
        assert!(matches!(
            resolve(
                &BackendKind::StochasticPauli { num_samples: 9 },
                &circuit,
                false
            ),
            ExecutionPlan::StochasticPauli { num_samples: 9 }
        ));
        assert!(matches!(
            resolve(
                &BackendKind::DeterministicPauli {
                    epsilon: 0.5,
                    max_terms: 3
                },
                &circuit,
                false
            ),
            ExecutionPlan::DeterministicPauli {
                epsilon: e,
                max_terms: 3
            } if e == 0.5
        ));
    }

    /// Explicit GPU kinds resolve hard device execution at their crossover
    /// (no VRAM gate) and host execution below it; the family choice never
    /// changes with the target.
    #[cfg(feature = "gpu")]
    #[test]
    fn explicit_gpu_kind_matrix() {
        let sv_kind = BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        };
        let large = dense(crate::gpu::min_qubits());
        let plan = resolved(&sv_kind, &large, false);
        assert_eq!(plan.family(), Family::Statevector);
        assert!(matches!(plan.accel(), Accel::Gpu { soft: false, .. }));
        assert_cpu_family(
            &sv_kind,
            &dense(crate::gpu::min_qubits() - 1),
            false,
            Family::Statevector,
        );

        let stab_kind = BackendKind::StabilizerGpu {
            context: GpuContext::stub_for_tests(),
        };
        let huge = Circuit::new(crate::gpu::stabilizer_min_qubits(), 0);
        let plan = resolved(&stab_kind, &huge, false);
        assert_eq!(plan.family(), Family::Stabilizer);
        assert!(matches!(plan.accel(), Accel::Gpu { soft: false, .. }));
        assert_cpu_family(&stab_kind, &clifford(8), false, Family::Stabilizer);
    }
}
