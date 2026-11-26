//! Backend selection and execution planning.
//!
//! Resolves a [`BackendKind`] against a circuit into a buildable plan,
//! including the CPU-vs-GPU accel verdict and the temporal-Clifford split.

use crate::backend::density_matrix::DensityMatrixBackend;
use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::{MAX_SPARSE_INDEX_QUBITS, SparseBackend};
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::backend::{Backend, check_state_allocation};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::limits::{LimitKnob, ResolvedLimits};
use crate::sim::unified_pauli::SpdTruncation;

#[cfg(any(feature = "gpu", feature = "distributed"))]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

#[cfg(feature = "distributed")]
use crate::backend::distributed_statevector::DistributedStatevectorBackend;
#[cfg(feature = "distributed")]
use crate::distributed::DistributedContext;

use super::StartState;
use super::metadata::ResolvedBackend;
use super::{RunOutcome, try_backend_probabilities};

/// Fixed cost of one gate at one cut in [`mps_apply_cost`] units, standing for
/// the per-gate work (tensor reshapes, gauge moves, allocation) a small bond does
/// not amortize. Fit on 85 circuits at 32 to 64 qubits: about 4 us per gate-cut
/// against about 15 ns per unit of bond cubed.
/// Bracketed by `auto/expectation_above_cap` (`hea_l2/64` against `su4_brick_d6/64`).
const MPS_GATE_CUT_OVERHEAD: f64 = 256.0;

/// Estimate the work an MPS capped at `max_bond` spends applying `circuit`, in
/// units of one bond dimension cubed.
///
/// Each gate on two or more qubits multiplies the bond at every cut it spans by
/// its operator Schmidt rank (2 for controlled and `Rzz` gates, 4 otherwise),
/// capped at `min(2^min(left, right), max_bond)`, and costs the cube of each
/// resulting bond, the SVD it triggers there, plus [`MPS_GATE_CUT_OVERHEAD`].
/// Gates on one pair with nothing between them on either qubit or any spanned
/// cut form one block, whose rank stays at most 4, as fusion delivers them
/// merged. On one thread an MPS run costs 50 to 70 ns per unit on CX and CZ
/// brickworks at 32 to 256 qubits and 20 to 45 ns on hardware-efficient ladders;
/// the bond is an upper bound, and on SU(4) brickworks, whose blocks reach it
/// late, the estimate runs 2 to 40 times high. Also returns the cuts at which a
/// block starts, each one SVD a merged run pays at whatever bond it reaches.
pub(super) fn mps_apply_cost(circuit: &Circuit, max_bond: usize) -> (f64, usize) {
    let n = circuit.num_qubits;
    if n < 2 {
        return (0.0, 0);
    }
    let limit: Vec<f64> = (0..n - 1)
        .map(|cut| {
            2f64.powi((cut + 1).min(n - cut - 1) as i32)
                .min(max_bond as f64)
        })
        .collect();
    let mut bond = vec![1.0f64; n - 1];
    let mut base = vec![1.0f64; n - 1];
    let mut block_rank = vec![1.0f64; n - 1];
    let mut qubit_block = vec![0usize; n];
    let mut cut_block = vec![0usize; n - 1];
    let mut next_block = 1usize;
    let mut cost = 0.0;
    let mut block_cuts = 0;
    for instruction in &circuit.instructions {
        let Instruction::Gate { gate, targets } = instruction else {
            continue;
        };
        if targets.len() < 2 {
            continue;
        }
        let lo = *targets.iter().min().expect("two or more targets");
        let hi = *targets.iter().max().expect("two or more targets");
        let rank = match gate {
            Gate::Cx | Gate::Cz | Gate::Rzz(_) | Gate::Cu(_) | Gate::Mcu(_) => 2.0,
            _ => 4.0,
        };
        let block = qubit_block[lo];
        let continues = targets.len() == 2
            && block != 0
            && qubit_block[hi] == block
            && cut_block[lo..hi].iter().all(|&b| b == block);
        let block = if continues {
            block
        } else {
            block_cuts += hi - lo;
            next_block += 1;
            next_block - 1
        };
        for cut in lo..hi {
            if continues {
                block_rank[cut] = (block_rank[cut] * rank).min(4.0);
            } else {
                base[cut] = bond[cut];
                block_rank[cut] = rank;
            }
            bond[cut] = (base[cut] * block_rank[cut]).min(limit[cut]);
            cut_block[cut] = block;
            cost += MPS_GATE_CUT_OVERHEAD + bond[cut].powi(3);
        }
        for &q in targets.iter() {
            qubit_block[q] = if targets.len() == 2 { block } else { 0 };
        }
    }
    (cost, block_cuts)
}

/// Most T gates, guarded ones included, at which `Auto` samples a circuit past the
/// statevector cap on the stabilizer-rank branches when its measurements stay in place.
/// Each T doubles the branches, and each measurement takes `B (B + 1)` MPS inner products
/// at `B` branches: 22 ms a shot at 6 T and 1.7 s at 8 T on 32 qubits.
/// Bracketed by `auto/clifford_t_price` (`rank_mid/mid_32q_t6_16shots` and `rank_mid/mid_32q_t8_4shots` on the branches, `auto_mid/guarded_s_32q_t{6,8}` on the route `Auto` takes).
pub(super) const MAX_AUTO_MID_CIRCUIT_BRANCH_T: usize = 6;

/// Bracketed by `auto/crossover` (`spd_10q_8t`, `spd_12q_8t` and `spd_16q_8t` rows).
pub(super) const MIN_QUBITS_FOR_SPD_AUTO: usize = 12;

/// Statevector amplitude updates one SPD term step is priced at when `Auto`
/// answers marginals. A term step measured 100 to 300 ns against about 1.4 ns
/// per amplitude update; 64 leans toward the Pauli route so a circuit it wins
/// stays on it.
/// Bracketed by `auto/crossover` (`spd_wide_*` rows).
pub(super) const AUTO_SPD_AMPLITUDES_PER_TERM: usize = 64;

/// Bracketed by `auto/crossover` (`fstab_*` rows).
#[cfg(not(prism_lowered_thresholds))]
pub(super) const MIN_FACTORED_STABILIZER_QUBITS: usize = 16;
#[cfg(prism_lowered_thresholds)]
pub(super) const MIN_FACTORED_STABILIZER_QUBITS: usize = 6;

/// No bench row brackets this yet.
pub(super) const MIN_BLOCK_FOR_FACTORED_STAB: usize = 2;

/// Entry bound under which `Auto` takes the sparse map over the MPS past the
/// statevector cap. A sparse gate costs about 8 ns per entry and an MPS gate at
/// bond 2 about 2 us, so on a circuit that barely entangles the two tie near 256.
/// Bracketed by `auto/sparse_support_shots` (`wide_k6/64` against `wide_k12/64`).
const AUTO_SPARSE_MAX_SUPPORT: usize = 128;

/// Amplitude updates of `instructions` gates applied to a dense `width`-qubit state,
/// saturating at `usize::MAX`.
#[inline]
pub(super) fn statevector_work(instructions: usize, width: usize) -> usize {
    instructions.saturating_mul(1usize.checked_shl(width as u32).unwrap_or(usize::MAX))
}

/// SPD term steps `Auto` spends on a marginals query before handing it to the
/// statevector: the dense run's [`statevector_work`], priced at
/// [`AUTO_SPD_AMPLITUDES_PER_TERM`] each.
#[inline]
pub(super) fn auto_spd_work_budget(circuit: &Circuit) -> usize {
    statevector_work(circuit.instructions.len(), circuit.num_qubits) / AUTO_SPD_AMPLITUDES_PER_TERM
}

/// Backend selection for a simulation run.
///
/// `Auto` resolves per call from circuit shape. Two routes run before the
/// family tree: circuits that decompose into independent blocks run per block
/// (Clifford-only circuits at 16 qubits and above with a block of 2 or more use
/// FactoredStabilizer), and shot paths sample Clifford+T circuits through the
/// StabilizerRank expansion: terminal shots when the Clifford register fits 20 qubits and
/// is priced below the dense run or the dense run passes the statevector cap, and shots
/// with a mid-circuit measurement, a guard or a reset on the same register once their
/// measurements move to the end, the dense side priced per outcome history. Past the
/// statevector cap, such a circuit whose measurements stay in place takes the weighted
/// branches up to 6 T gates. A probability
/// query never takes that expansion under `Auto`, since it is slower than the
/// statevector at every width both can hold. The pruned expansion is reachable
/// only through [`run_stabilizer_rank_approx`], never from `Auto`. Expectation and
/// marginal queries on unitary Clifford+T circuits read the register exactly as one
/// factor per set of qubits the T rotations link, when that costs less than the dense
/// run or the circuit passes the statevector cap. Marginal queries it declines at 12
/// qubits and above answer via Sparse Pauli Dynamics, which hands a circuit that fits
/// the statevector back to the tree once its Pauli sum costs more than the dense run
/// would. The remaining tree:
///
/// 1. No entangling gates        → ProductState (O(n))
/// 2. All Clifford gates         → Stabilizer (O(n²))
/// 3. Above the statevector memory cap:
///    a. Sparse-friendly         → Sparse (O(k) where k = non-zero amplitudes)
///    b. Otherwise               → MPS (bounded bond dimension)
/// 4. Partial independence       → Factored (per-group dense sub-states)
/// 5. Otherwise                  → Statevector (exact, general-purpose)
///
/// [`run_stabilizer_rank_approx`]: crate::run_stabilizer_rank_approx
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum BackendKind {
    Auto,
    Statevector,
    /// Tableau simulation for Clifford-only circuits.
    Stabilizer,
    /// Sparse state vector holding only nonzero amplitudes.
    Sparse,
    /// Matrix Product State simulation with a bounded bond dimension.
    Mps {
        max_bond_dim: usize,
    },
    /// Per-qubit product state for circuits without entangling gates.
    ProductState,
    /// Deferred-contraction tensor network for low-treewidth circuits.
    TensorNetwork,
    /// The tensor network with bond truncation at the peak cap.
    ///
    /// An intermediate over the tensor-network peak cap is factored and the
    /// new bond kept only as far as discarding at most `tolerance` of that
    /// cut's squared weight allows; index slicing covers whatever stays above
    /// the cap. `tolerance` must be finite and above zero, since zero is
    /// [`BackendKind::TensorNetwork`]. The run is approximate, so
    /// [`Simulate::require_exact`] rejects it.
    ///
    /// [`Simulate::require_exact`]: crate::sim::Simulate::require_exact
    TensorNetworkBounded {
        tolerance: f64,
    },
    /// Dynamic split-state simulation for sparse-entanglement circuits.
    Factored,
    /// Clifford+T decomposition into weighted stabilizer branches.
    StabilizerRank,
    /// Independent Clifford blocks, each on its own tableau.
    FactoredStabilizer,
    /// Exact density-matrix backend for mixed-state evolution.
    ///
    /// Explicit-dispatch only: [`BackendKind::Auto`] never selects it. Stores
    /// `4^n` `Complex64` amplitudes, so the qubit ceiling is roughly half the
    /// statevector cap (14 on a 16 GiB host); `PRISM_MAX_DM_QUBITS` moves it
    /// within that bound. Fused payloads are accepted, with every fusion floor
    /// gated on the `2n`-qubit buffer the backend sweeps rather than on the
    /// circuit width.
    ///
    /// Under an attached noise model this is the exact route: the mixture is
    /// evolved once and every terminal reads it, so shot counts carry sampling
    /// noise only and observables carry none.
    DensityMatrix,
    /// Stochastic Pauli propagation (SPP); serves marginal and observable
    /// queries only.
    StochasticPauli {
        num_samples: usize,
    },
    /// Deterministic sparse Pauli dynamics (SPD); serves marginal and
    /// observable queries only. [`SpdTruncation`] picks between the threshold
    /// policy and the budget one, which holds a fixed term count and so cannot
    /// die at the internal ceiling on growth the budget already caps.
    DeterministicPauli {
        truncation: SpdTruncation,
    },
    /// Heisenberg Pauli propagation through a noise model; serves expectation
    /// values and observable expectations only.
    ///
    /// Explicit-dispatch only: [`BackendKind::Auto`] never selects it. Noise
    /// enters as the channel's action on the Pauli basis, which shrinks
    /// coefficients while the circuit's rotations grow the term count, so the
    /// weighted sum stays small exactly when noise outpaces the branching
    /// rotations. Terms below `epsilon` are dropped once the sum exceeds
    /// `max_terms` (0 disables truncation and makes the run exact).
    ///
    /// A channel with no Pauli-basis form (custom Kraus, two-qubit Kraus,
    /// readout error) is rejected naming the density matrix.
    PauliPath {
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
    /// Applies to single runs, terminal shot and counts sampling, expectation
    /// values, temporal-Clifford tails, and non-Pauli noisy trajectories.
    #[cfg(feature = "gpu")]
    AutoGpu {
        context: Arc<GpuContext>,
    },
    /// Statevector backed by a CUDA GPU execution context.
    ///
    /// Circuits (or decomposed sub-blocks) with fewer than
    /// [`crate::gpu::min_qubits()`] qubits (tunable via
    /// `PRISM_GPU_MIN_QUBITS`, default [`crate::gpu::MIN_QUBITS_DEFAULT`])
    /// fall back to the host statevector path, since small states do not
    /// survive PCIe and launch-latency overhead.
    ///
    /// Compose with `simulate(...).backend(...).seed(...).run()` to get fusion
    /// plus independent-subsystem decomposition; each sub-block is evaluated
    /// against the crossover independently.
    #[cfg(feature = "gpu")]
    StatevectorGpu {
        context: Arc<GpuContext>,
    },
    /// Density matrix held in device memory on the supplied context.
    ///
    /// Explicit-dispatch only: neither [`BackendKind::Auto`] nor
    /// [`BackendKind::AutoGpu`] selects it. There is no crossover and no host
    /// fallback: every run allocates the `4^n` mixture on the device, after a
    /// budget check against the free VRAM that errors before allocating. An
    /// 11 GiB card holds 14 qubits (4 GiB at 14 plus a `2^n` diagonal scratch;
    /// 15 needs 16 GiB).
    /// Every channel, measurement, and readout sweep runs as a kernel; the
    /// noisy terminals answer from the exact mixture as
    /// [`BackendKind::DensityMatrix`] does.
    #[cfg(feature = "gpu")]
    DensityMatrixGpu {
        context: Arc<GpuContext>,
    },
    /// Stabilizer backend backed by a CUDA GPU tableau.
    ///
    /// Circuits (or decomposed sub-blocks) with fewer than
    /// [`crate::gpu::stabilizer_min_qubits()`] qubits (tunable via
    /// `PRISM_STABILIZER_GPU_MIN_QUBITS`, default
    /// [`crate::gpu::STABILIZER_MIN_QUBITS_DEFAULT`]) fall back to the CPU
    /// stabilizer path, as do circuits with a measurement group shorter than eight, a
    /// reset or a guarded region. Measurement and reset stay on device; probabilities
    /// and exports read back to the CPU algorithms.
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

    /// False for the engines without a per-shot pure state: stabilizer rank,
    /// Pauli propagation, density matrix. The density matrix still serves a
    /// noise model, by evolving the mixture once and sampling the exact
    /// distribution, so its noisy terminals route around the trajectory engine
    /// rather than through it.
    pub fn supports_noisy_per_shot(&self) -> bool {
        !matches!(
            self,
            BackendKind::StabilizerRank
                | BackendKind::StochasticPauli { .. }
                | BackendKind::DeterministicPauli { .. }
                | BackendKind::PauliPath { .. }
        ) && !self.is_density_matrix()
    }

    /// True for the two kinds that hold the exact mixture, host or device.
    pub(crate) fn is_density_matrix(&self) -> bool {
        match self {
            BackendKind::DensityMatrix => true,
            #[cfg(feature = "gpu")]
            BackendKind::DensityMatrixGpu { .. } => true,
            _ => false,
        }
    }

    /// True for kinds that can run non-Pauli channels (damping, thermal
    /// relaxation, custom Kraus): the trajectory engine everywhere except the
    /// density matrix, which applies the channel to the mixture instead.
    pub fn supports_general_noise(&self) -> bool {
        match self {
            BackendKind::Auto
            | BackendKind::Statevector
            | BackendKind::Sparse
            | BackendKind::Mps { .. }
            | BackendKind::ProductState
            | BackendKind::Factored
            | BackendKind::TensorNetwork
            | BackendKind::TensorNetworkBounded { .. }
            | BackendKind::DensityMatrix => true,
            #[cfg(feature = "gpu")]
            BackendKind::AutoGpu { .. }
            | BackendKind::StatevectorGpu { .. }
            | BackendKind::DensityMatrixGpu { .. } => true,
            #[cfg(feature = "distributed")]
            BackendKind::StatevectorDistributed { .. } => true,
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

    pub(crate) fn general_noise_backend_names() -> String {
        let mut names = vec!["Auto", "Statevector"];
        #[cfg(feature = "gpu")]
        names.push("StatevectorGpu");
        #[cfg(feature = "distributed")]
        names.push("StatevectorDistributed");
        names.extend([
            "Sparse",
            "Mps",
            "ProductState",
            "Factored",
            "TensorNetwork",
            "DensityMatrix",
        ]);
        #[cfg(feature = "gpu")]
        names.push("DensityMatrixGpu");
        let last = names.pop().expect("at least one name");
        format!("{}, or {last}", names.join(", "))
    }
}

/// Reject a circuit the explicitly selected `kind` cannot run; an `Auto` kind passes.
pub(super) fn validate_explicit_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<()> {
    if kind.is_auto() {
        return Ok(());
    }
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
        BackendKind::TensorNetworkBounded { tolerance }
            if !(tolerance.is_finite() && *tolerance > 0.0) =>
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "tensor-network tolerance must be a finite fraction above zero, got {tolerance}; \
                     TensorNetwork is the exact contraction"
                ),
            });
        }
        BackendKind::StabilizerRank if !circuit.has_t_gates() => {
            return Err(PrismError::IncompatibleBackend {
                backend: "stabilizer_rank".into(),
                reason: "circuit has no T gates; use Stabilizer instead".into(),
            });
        }
        BackendKind::DensityMatrix => {
            check_state_allocation(
                "density_matrix",
                circuit.num_qubits,
                limits.density_matrix_qubits,
                LimitKnob::DensityMatrixQubits,
            )?;
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
    DensityMatrix,
}

#[cfg(test)]
thread_local! {
    pub(super) static SPARSE_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn select_auto_backend_choice(
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> Family {
    select_auto_backend_choice_with_probe(circuit, has_partial_independence, limits).0
}

fn select_auto_backend_choice_with_probe(
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> (Family, Option<SparseBackend>) {
    if !circuit.has_entangling_gates() {
        (Family::ProductState, None)
    } else if circuit.is_clifford_only() {
        (Family::Stabilizer, None)
    } else if circuit.num_qubits > limits.statevector_qubits.value {
        if circuit.num_qubits <= MAX_SPARSE_INDEX_QUBITS {
            if circuit.is_sparse_friendly() {
                return (Family::Sparse, None);
            }
            if let Some(probe) = sparse_support_probe(circuit, AUTO_SPARSE_MAX_SUPPORT, limits) {
                return (Family::Sparse, Some(probe));
            }
        }
        (Family::Mps, None)
    } else if has_partial_independence {
        (Family::Factored, None)
    } else {
        (Family::Statevector, None)
    }
}

/// Run the circuit's gates on a sparse map and return the evolved backend when
/// every step stays within `cap` entries. Declines resets, saves, and anything
/// after a measurement, whose branch could reach a different support.
fn sparse_support_probe(
    circuit: &Circuit,
    cap: usize,
    limits: &ResolvedLimits,
) -> Option<SparseBackend> {
    if !circuit.has_terminal_measurements_only() {
        return None;
    }
    #[cfg(test)]
    SPARSE_PROBES.with(|c| c.set(c.get() + 1));
    let mut probe = SparseBackend::with_limits(0, limits);
    if probe.init(circuit.num_qubits, 0).is_err() {
        return None;
    }
    let lowered = super::expand_for_backend(&probe, circuit);
    let ok = lowered.instructions.iter().all(|inst| match inst {
        Instruction::Gate { .. } => probe.apply(inst).is_ok() && probe.entry_count() <= cap,
        Instruction::Measure { .. } | Instruction::Barrier { .. } => true,
        _ => false,
    });
    ok.then_some(probe)
}

#[cfg(test)]
fn sparse_support_fits(circuit: &Circuit, cap: usize, limits: &ResolvedLimits) -> bool {
    sparse_support_probe(circuit, cap, limits).is_some()
}

pub(super) fn auto_selects_cpu_statevector(
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> bool {
    if circuit.num_qubits > limits.statevector_qubits.value {
        return false;
    }
    matches!(
        select_auto_backend_choice(circuit, has_partial_independence, limits),
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
const GPU_CAPABLE_FAMILIES: [Family; 3] = [
    Family::Statevector,
    Family::Stabilizer,
    Family::DensityMatrix,
];

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
        // Explicit-only: no crossover, and the `Auto` tree never selects the
        // family, so the soft fit gate is never consulted. The hard path budgets
        // the `4^n` buffer at `init`.
        Family::DensityMatrix => Some(GpuCapability {
            min_qubits: || 0,
            fits: |ctx, n| ctx.fits_statevector_with_scratch(2 * n).unwrap_or(false),
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
        (BackendKind::DensityMatrixGpu { context }, Family::DensityMatrix) => {
            Some((context, false))
        }
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

pub(super) fn build_statevector(
    accel: &Accel,
    seed: u64,
    limits: &ResolvedLimits,
) -> StatevectorBackend {
    let sv = StatevectorBackend::with_limits(seed, limits);
    match accel {
        Accel::Cpu => sv,
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: true,
        } => sv.with_gpu_auto(context.clone()),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: false,
        } => sv.with_gpu(context.clone()),
    }
}

pub(super) fn build_density_matrix(
    accel: &Accel,
    seed: u64,
    limits: &ResolvedLimits,
) -> DensityMatrixBackend {
    let dm = DensityMatrixBackend::with_limits(seed, limits);
    match accel {
        Accel::Cpu => dm,
        #[cfg(feature = "gpu")]
        Accel::Gpu { context, .. } => dm.with_gpu(context.clone()),
    }
}

/// True when no instruction can force destabilizer materialization: gates,
/// satisfied conditionals, and barriers only. Measurements, resets, and
/// guarded regions (whose bodies can hold either) all disqualify.
fn stabilizer_can_run_lazy(circuit: &Circuit) -> bool {
    circuit.instructions.iter().all(|inst| {
        matches!(
            inst,
            Instruction::Gate { .. }
                | Instruction::Conditional { .. }
                | Instruction::Barrier { .. }
        )
    })
}

/// Whether every measurement and reset of `circuit` reaches the device tableau inside a
/// run it keeps on the device: consecutive measurements at least
/// `MIN_MEASURES_FOR_BATCH` long. A shorter group or a reset takes a host round trip
/// each, about ten times the host's own cost, and a guarded region reads classical bits
/// between its instructions.
#[cfg(feature = "gpu")]
fn stabilizer_measurements_stay_on_device(circuit: &Circuit) -> bool {
    use crate::backend::stabilizer::kernels::batch::MIN_MEASURES_FOR_BATCH;
    let mut run = 0usize;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Measure { .. } => run += 1,
            Instruction::Reset { .. } | Instruction::Region(_) => return false,
            _ => {
                if run > 0 && run < MIN_MEASURES_FOR_BATCH {
                    return false;
                }
                run = 0;
            }
        }
    }
    run == 0 || run >= MIN_MEASURES_FOR_BATCH
}

fn build_stabilizer(
    accel: &Accel,
    lazy: bool,
    seed: u64,
    limits: &ResolvedLimits,
) -> StabilizerBackend {
    // Lazy destabilizers: gates touch half the tableau. The plan sets `lazy`
    // only for circuits that never measure or reset, so the Gaussian
    // elimination that materialization costs is never paid; on a measuring
    // circuit the reconstruction can exceed the gate savings (the GHZ
    // measure-all shape loses outright, since its SGI gate cost is already
    // near zero). GPU init overrides the flag; the soft-fallback host path
    // keeps it.
    let stab = StabilizerBackend::with_limits(seed, limits);
    let stab = if lazy { stab.with_lazy_destab() } else { stab };
    match accel {
        Accel::Cpu => stab,
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: true,
        } => stab.with_gpu_auto(context.clone()),
        #[cfg(feature = "gpu")]
        Accel::Gpu {
            context,
            soft: false,
        } => stab.with_gpu(context.clone()),
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
    TensorNetwork {
        tolerance: Option<f64>,
    },
    Factored,
    FactoredStabilizer,
    Mps {
        max_bond_dim: usize,
    },
    Stabilizer {
        accel: Accel,
        lazy: bool,
    },
    Statevector {
        accel: Accel,
    },
    DensityMatrix {
        accel: Accel,
    },
    #[cfg(feature = "distributed")]
    Distributed(Arc<DistributedContext>),
}

/// True when two accelerator targets are the same device with the same failure
/// mode. A soft and a hard build of one context differ, because a device
/// allocation failure sends them to different places.
fn same_accel(a: &Accel, b: &Accel) -> bool {
    match (a, b) {
        (Accel::Cpu, Accel::Cpu) => true,
        #[cfg(feature = "gpu")]
        (
            Accel::Gpu {
                context: cx,
                soft: sx,
            },
            Accel::Gpu {
                context: cy,
                soft: sy,
            },
        ) => Arc::ptr_eq(cx, cy) && sx == sy,
        #[cfg(feature = "gpu")]
        _ => false,
    }
}

/// True when two plans build the same engine with the same settings, so one
/// backend can serve both.
///
/// Written out rather than derived because the distributed variant holds a
/// context that has no equality, and sharing a rank's backend across circuits
/// is not something this answers yes to.
pub(super) fn same_plan(a: &BackendPlan, b: &BackendPlan) -> bool {
    match (a, b) {
        (BackendPlan::ProductState, BackendPlan::ProductState)
        | (BackendPlan::Sparse, BackendPlan::Sparse)
        | (BackendPlan::Factored, BackendPlan::Factored)
        | (BackendPlan::FactoredStabilizer, BackendPlan::FactoredStabilizer) => true,
        (
            BackendPlan::TensorNetwork { tolerance: x },
            BackendPlan::TensorNetwork { tolerance: y },
        ) => x == y,
        (BackendPlan::Mps { max_bond_dim: x }, BackendPlan::Mps { max_bond_dim: y }) => x == y,
        (
            BackendPlan::Stabilizer {
                accel: ax,
                lazy: lx,
            },
            BackendPlan::Stabilizer {
                accel: ay,
                lazy: ly,
            },
        ) => same_accel(ax, ay) && lx == ly,
        (BackendPlan::Statevector { accel: x }, BackendPlan::Statevector { accel: y })
        | (BackendPlan::DensityMatrix { accel: x }, BackendPlan::DensityMatrix { accel: y }) => {
            same_accel(x, y)
        }
        _ => false,
    }
}

impl BackendPlan {
    pub(super) fn supports_native_sampling(&self) -> bool {
        match self {
            Self::ProductState
            | Self::Sparse
            | Self::TensorNetwork { .. }
            | Self::Factored
            | Self::Mps { .. } => true,
            #[cfg(feature = "distributed")]
            Self::Distributed(_) => true,
            _ => false,
        }
    }

    /// Engine this plan builds, for a shot request that runs the plan zero
    /// times and so has no backend to read provenance off.
    pub(super) fn resolved(&self) -> ResolvedBackend {
        match self {
            BackendPlan::ProductState => ResolvedBackend::ProductState,
            BackendPlan::Sparse => ResolvedBackend::Sparse,
            BackendPlan::TensorNetwork { .. } => ResolvedBackend::TensorNetwork,
            BackendPlan::Factored => ResolvedBackend::Factored,
            BackendPlan::FactoredStabilizer => ResolvedBackend::FactoredStabilizer,
            BackendPlan::Mps { .. } => ResolvedBackend::Mps,
            BackendPlan::Stabilizer { .. } => ResolvedBackend::Stabilizer,
            BackendPlan::Statevector { .. } => ResolvedBackend::Statevector,
            BackendPlan::DensityMatrix { .. } => ResolvedBackend::DensityMatrix,
            #[cfg(feature = "distributed")]
            BackendPlan::Distributed(_) => ResolvedBackend::Distributed,
        }
    }

    pub(super) fn build(&self, seed: u64, limits: &ResolvedLimits) -> Box<dyn Backend + Send> {
        match self {
            BackendPlan::ProductState => Box::new(ProductStateBackend::with_limits(seed, limits)),
            BackendPlan::Sparse => Box::new(SparseBackend::with_limits(seed, limits)),
            BackendPlan::TensorNetwork { tolerance } => {
                Box::new(TensorNetworkBackend::with_limits(seed, *tolerance, limits))
            }
            BackendPlan::Factored => Box::new(
                crate::backend::factored::FactoredBackend::with_limits(seed, limits),
            ),
            BackendPlan::FactoredStabilizer => Box::new(
                crate::backend::factored_stabilizer::FactoredStabilizerBackend::with_limits(
                    seed, limits,
                ),
            ),
            BackendPlan::Mps { max_bond_dim } => {
                Box::new(MpsBackend::with_limits(seed, *max_bond_dim, limits))
            }
            BackendPlan::Stabilizer { accel, lazy } => {
                Box::new(build_stabilizer(accel, *lazy, seed, limits))
            }
            BackendPlan::Statevector { accel } => Box::new(build_statevector(accel, seed, limits)),
            BackendPlan::DensityMatrix { accel } => {
                Box::new(build_density_matrix(accel, seed, limits))
            }
            #[cfg(feature = "distributed")]
            BackendPlan::Distributed(context) => Box::new(
                DistributedStatevectorBackend::with_limits(context.clone(), seed, limits),
            ),
        }
    }

    /// Ask the backend this plan builds, built without state, whether it runs a
    /// two-qubit Kraus branch.
    pub(super) fn supports_two_qubit_kraus(&self, limits: &ResolvedLimits) -> bool {
        self.build(0, limits).supports_two_qubit_kraus()
    }

    pub(super) fn is_host_statevector(&self) -> bool {
        matches!(self, BackendPlan::Statevector { accel: Accel::Cpu })
    }

    pub(super) fn is_gpu(&self) -> bool {
        match self {
            BackendPlan::Stabilizer { accel, .. }
            | BackendPlan::Statevector { accel }
            | BackendPlan::DensityMatrix { accel } => !matches!(accel, Accel::Cpu),
            _ => false,
        }
    }

    #[cfg(test)]
    pub(super) fn family(&self) -> Family {
        match self {
            BackendPlan::ProductState => Family::ProductState,
            BackendPlan::Sparse => Family::Sparse,
            BackendPlan::TensorNetwork { .. } => Family::TensorNetwork,
            BackendPlan::Factored => Family::Factored,
            BackendPlan::FactoredStabilizer => Family::FactoredStabilizer,
            BackendPlan::Mps { .. } => Family::Mps,
            BackendPlan::Stabilizer { .. } => Family::Stabilizer,
            BackendPlan::Statevector { .. } => Family::Statevector,
            BackendPlan::DensityMatrix { .. } => Family::DensityMatrix,
            #[cfg(feature = "distributed")]
            BackendPlan::Distributed(_) => Family::Statevector,
        }
    }

    #[cfg(test)]
    pub(super) fn accel(&self) -> &Accel {
        match self {
            BackendPlan::Stabilizer { accel, .. }
            | BackendPlan::Statevector { accel }
            | BackendPlan::DensityMatrix { accel } => accel,
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
    DeterministicPauli { truncation: SpdTruncation },
    PauliPath,
}

/// Name the engine `kind` resolves to when that engine can discard state
/// weight or estimate by sampling, `None` when the route is exact.
///
/// Exactness is a property of the route and the circuit together, not of the
/// route alone: an engine that can truncate still answers exactly on a circuit
/// whose state it holds without loss, and only a predicate that proves that for
/// every circuit of the shape may say so here.
///
/// Resolved from the circuit rather than read off a finished run, so
/// [`Simulate::require_exact`] rejects before paying for the state it would
/// throw away.
///
/// [`Simulate::require_exact`]: crate::sim::Simulate::require_exact
pub(super) fn approximate_route_name(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Option<&'static str> {
    // Diagonal and permutation gates keep the all-zero start on a single basis
    // state, so an MPS holds it at bond 1 and has nothing to truncate. A start
    // state other than |0...0> never reaches an MPS route at all, since
    // `initial_state_plan` rejects every representation derived from that start.
    let mps_is_exact = circuit.is_sparse_friendly();
    match kind {
        BackendKind::Mps { .. } => {
            if mps_is_exact {
                return None;
            }
            return Some("Mps");
        }
        BackendKind::TensorNetworkBounded { .. } => return Some("TensorNetworkBounded"),
        BackendKind::StochasticPauli { .. } => return Some("StochasticPauli"),
        BackendKind::DeterministicPauli { truncation } => {
            return truncation.can_discard().then_some("DeterministicPauli");
        }
        BackendKind::PauliPath { epsilon, max_terms } => {
            let truncation = SpdTruncation::Threshold {
                epsilon: *epsilon,
                max_terms: *max_terms,
            };
            return truncation.can_discard().then_some("PauliPath");
        }
        _ => {}
    }
    if !kind.is_auto() {
        return None;
    }
    let (_, has_partial_independence) = crate::sim::analyze_independence(circuit);
    match select_auto_backend_choice(circuit, has_partial_independence, limits) {
        Family::Mps if !mps_is_exact => Some("Mps"),
        _ => None,
    }
}

pub(super) fn plan_for_family(
    kind: &BackendKind,
    family: Family,
    num_qubits: usize,
    limits: &ResolvedLimits,
) -> BackendPlan {
    match family {
        Family::ProductState => BackendPlan::ProductState,
        Family::Sparse => BackendPlan::Sparse,
        Family::TensorNetwork => BackendPlan::TensorNetwork {
            tolerance: match kind {
                BackendKind::TensorNetworkBounded { tolerance } => Some(*tolerance),
                _ => None,
            },
        },
        Family::Factored => BackendPlan::Factored,
        Family::FactoredStabilizer => BackendPlan::FactoredStabilizer,
        Family::Mps => BackendPlan::Mps {
            max_bond_dim: limits.auto_mps_bond_dim.value,
        },
        Family::Stabilizer => BackendPlan::Stabilizer {
            accel: accel_for(kind, Family::Stabilizer, num_qubits),
            lazy: false,
        },
        Family::Statevector => BackendPlan::Statevector {
            accel: accel_for(kind, Family::Statevector, num_qubits),
        },
        Family::DensityMatrix => BackendPlan::DensityMatrix {
            accel: accel_for(kind, Family::DensityMatrix, num_qubits),
        },
    }
}

/// Resolve `kind` against `circuit` into an [`ExecutionPlan`]. Auto kinds run
/// the shape-based decision tree; explicit kinds map 1:1 onto their family. The
/// CPU-vs-GPU verdict, including the VRAM-fit query, is taken here through
/// [`accel_for`] and reused across shots; soft-mode init fallback covers VRAM
/// shrinking after resolution.
///
/// A shot loop resolves once and replays the plan; a decomposable circuit with a
/// temporal Clifford prefix plans each block once. Only the temporal-Clifford
/// fallback resolves per shot, for a save point inside the leading Clifford run,
/// and the public shot paths reject save points before reaching it.
pub(super) fn resolve(
    kind: &BackendKind,
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> ExecutionPlan {
    resolve_with_probe(kind, circuit, has_partial_independence, limits).0
}

pub(super) fn resolve_with_probe(
    kind: &BackendKind,
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> (ExecutionPlan, Option<SparseBackend>) {
    let mut probe = None;
    let family = match kind {
        BackendKind::Auto => {
            let (family, sparse) =
                select_auto_backend_choice_with_probe(circuit, has_partial_independence, limits);
            probe = sparse;
            family
        }
        #[cfg(feature = "gpu")]
        BackendKind::AutoGpu { .. } => {
            let (family, sparse) =
                select_auto_backend_choice_with_probe(circuit, has_partial_independence, limits);
            probe = sparse;
            family
        }
        BackendKind::Statevector => Family::Statevector,
        BackendKind::Stabilizer => Family::Stabilizer,
        BackendKind::Sparse => Family::Sparse,
        BackendKind::Mps { max_bond_dim } => {
            return (
                ExecutionPlan::Backend(BackendPlan::Mps {
                    max_bond_dim: *max_bond_dim,
                }),
                None,
            );
        }
        BackendKind::ProductState => Family::ProductState,
        BackendKind::TensorNetwork | BackendKind::TensorNetworkBounded { .. } => {
            Family::TensorNetwork
        }
        BackendKind::Factored => Family::Factored,
        BackendKind::FactoredStabilizer => Family::FactoredStabilizer,
        BackendKind::DensityMatrix => Family::DensityMatrix,
        BackendKind::StabilizerRank => return (ExecutionPlan::StabilizerRank, None),
        BackendKind::StochasticPauli { num_samples } => {
            return (
                ExecutionPlan::StochasticPauli {
                    num_samples: *num_samples,
                },
                None,
            );
        }
        BackendKind::PauliPath { .. } => return (ExecutionPlan::PauliPath, None),
        BackendKind::DeterministicPauli { truncation } => {
            return (
                ExecutionPlan::DeterministicPauli {
                    truncation: *truncation,
                },
                None,
            );
        }
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => Family::Statevector,
        #[cfg(feature = "gpu")]
        BackendKind::StabilizerGpu { .. } => Family::Stabilizer,
        #[cfg(feature = "gpu")]
        BackendKind::DensityMatrixGpu { .. } => Family::DensityMatrix,
        #[cfg(feature = "distributed")]
        BackendKind::StatevectorDistributed { context } => {
            return (
                ExecutionPlan::Backend(BackendPlan::Distributed(context.clone())),
                None,
            );
        }
    };
    let mut plan = plan_for_family(kind, family, circuit.num_qubits, limits);
    if let BackendPlan::Stabilizer { lazy, accel } = &mut plan {
        *lazy = stabilizer_can_run_lazy(circuit);
        #[cfg(feature = "gpu")]
        if matches!(accel, Accel::Gpu { .. }) && !stabilizer_measurements_stay_on_device(circuit) {
            *accel = Accel::Cpu;
        }
        #[cfg(not(feature = "gpu"))]
        let _ = accel;
    }
    (ExecutionPlan::Backend(plan), probe)
}

/// Backend plan for a run that starts from a caller-supplied state.
///
/// Routing is constrained here rather than consulted. Every shortcut the
/// shape-based tree can pick reads the circuit alone and is only valid from the
/// |0...0⟩ start: a Clifford circuit is a stabilizer state only when its input
/// is one, a product state and a subsystem split assume unentangled inputs, and
/// the Pauli engines propagate observables back to |0...0⟩. Selecting one of
/// them for an arbitrary start state is a wrong answer rather than an error, so
/// only the representations that can hold an arbitrary state are reachable: the
/// statevector, on the host, on a device, or sharded across ranks, and the
/// density matrix.
/// `Auto` lands on the statevector unconditionally: the caller already holds
/// `2^n` amplitudes, so the dense state is affordable by construction.
///
/// A start mixture narrows further, to the density matrix and its device
/// sibling: no other representation holds one, and `Auto` never selects the
/// density matrix, so it is declined by name like the rest.
pub(super) fn initial_state_plan(
    kind: &BackendKind,
    num_qubits: usize,
    start: StartState<'_>,
    limits: &ResolvedLimits,
) -> Result<BackendPlan> {
    if let StartState::DensityMatrix(_) = start {
        return match kind {
            BackendKind::DensityMatrix => Ok(plan_for_family(
                kind,
                Family::DensityMatrix,
                num_qubits,
                limits,
            )),
            #[cfg(feature = "gpu")]
            BackendKind::DensityMatrixGpu { .. } => Ok(plan_for_family(
                kind,
                Family::DensityMatrix,
                num_qubits,
                limits,
            )),
            other => Err(PrismError::IncompatibleBackend {
                backend: format!("{other:?}"),
                reason: "a start density matrix runs on the density matrix, which holds a \
                         mixture; every other representation holds a pure state, so select \
                         `BackendKind::DensityMatrix` or its device sibling"
                    .into(),
            }),
        };
    }
    match kind {
        BackendKind::Auto | BackendKind::Statevector => Ok(plan_for_family(
            kind,
            Family::Statevector,
            num_qubits,
            limits,
        )),
        #[cfg(feature = "gpu")]
        BackendKind::AutoGpu { .. } | BackendKind::StatevectorGpu { .. } => Ok(plan_for_family(
            kind,
            Family::Statevector,
            num_qubits,
            limits,
        )),
        BackendKind::DensityMatrix => Ok(plan_for_family(
            kind,
            Family::DensityMatrix,
            num_qubits,
            limits,
        )),
        #[cfg(feature = "gpu")]
        BackendKind::DensityMatrixGpu { .. } => Ok(plan_for_family(
            kind,
            Family::DensityMatrix,
            num_qubits,
            limits,
        )),
        #[cfg(feature = "distributed")]
        BackendKind::StatevectorDistributed { context } => {
            Ok(BackendPlan::Distributed(context.clone()))
        }
        other => Err(PrismError::IncompatibleBackend {
            backend: format!("{other:?}"),
            reason: "a start state other than |0...0> runs on the statevector or the density \
                     matrix; every other representation is derived from that start"
                .into(),
        }),
    }
}

pub(super) fn resolve_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> BackendPlan {
    match resolve(kind, circuit, has_partial_independence, limits) {
        ExecutionPlan::Backend(plan) => plan,
        _ => unreachable!("non-backend dispatch should be handled by caller"),
    }
}

#[inline]
pub(super) fn min_clifford_prefix_gates(num_qubits: usize) -> usize {
    (num_qubits * 2).max(16)
}

pub(super) fn has_temporal_clifford_opportunity(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> bool {
    if !kind.is_auto() {
        return false;
    }
    if circuit.num_qubits > limits.statevector_qubits.value {
        return false;
    }
    // The split pays for itself only when the tail needs a dense state. A
    // Clifford-only circuit has no such tail, and exporting the tableau to run
    // measurements or a guarded region densely is strictly a loss.
    if circuit.is_clifford_only() {
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
            | Instruction::Conditional { .. }
            | Instruction::Region(_) => break,
            Instruction::Barrier { .. } | Instruction::Save { .. } => {}
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
    limits: ResolvedLimits,
}

pub(super) fn plan_temporal_clifford(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Option<TemporalCliffordPlan> {
    if !kind.is_auto() {
        return None;
    }
    if circuit.num_qubits > limits.statevector_qubits.value {
        return None;
    }
    if circuit.is_clifford_only() {
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
            let expanded = crate::circuit::expand_pauli_rotations(&expanded).into_owned();
            crate::circuit::fusion::fuse_circuit(&expanded, true).into_owned()
        }
    };
    Some(TemporalCliffordPlan {
        prefix,
        fused_tail,
        tail_num_classical_bits,
        tail_accel,
        limits: *limits,
    })
}

pub(super) fn run_temporal_clifford(
    plan: &TemporalCliffordPlan,
    seed: u64,
    want_probabilities: bool,
) -> Result<RunOutcome> {
    let mut stab = StabilizerBackend::with_limits(seed, &plan.limits);
    stab.init(plan.prefix.num_qubits, plan.prefix.num_classical_bits)?;
    stab.enable_lazy_destab();
    for inst in &plan.prefix.instructions {
        stab.apply(inst)?;
    }

    let state = stab.export_statevector()?;

    let mut sv = build_statevector(&plan.tail_accel, seed, &plan.limits);
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
        metadata: crate::sim::backend_metadata(&sv),
        saves: Vec::new(),
    })
}

#[cfg(all(test, feature = "gpu"))]
#[path = "dispatch_gpu_tests.rs"]
mod dispatch_gpu_tests;

/// Table-driven coverage of [`resolve`]: every backend kind against every
/// circuit shape class, asserting the resolved family and execution target.
/// Extend it when a family gains a capability row.
#[cfg(test)]
#[path = "dispatch_matrix_tests.rs"]
mod dispatch_matrix_tests;
