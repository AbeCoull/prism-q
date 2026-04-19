use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::SparseBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};

#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

use super::{Probabilities, SimulationResult};

pub(super) enum DispatchAction {
    Backend(Box<dyn Backend>),
    StabilizerRank,
    StochasticPauli { num_samples: usize },
    DeterministicPauli { epsilon: f64, max_terms: usize },
}

fn env_override_usize(var: &str) -> Option<usize> {
    std::env::var(var).ok()?.parse().ok()
}

pub(super) fn max_statevector_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        env_override_usize("PRISM_MAX_SV_QUBITS").unwrap_or_else(|| {
            match detect_max_sv_qubits() {
                Some(n) => n,
                None => {
                    eprintln!(
                        "warning: could not detect system memory; statevector qubit cap is disabled. \
                         Large circuits may abort on allocation. Set PRISM_MAX_SV_QUBITS to suppress."
                    );
                    usize::MAX
                }
            }
        })
    })
}

#[cfg(windows)]
fn detect_max_sv_qubits() -> Option<usize> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }

    // SAFETY: zeroed MemoryStatusEx is valid (all-zero bit pattern is a valid repr(C) struct)
    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
    // SAFETY: status is a valid MemoryStatusEx with dw_length set; FFI call reads/writes only within the struct
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }

    let budget = status.ull_total_phys / 2;
    let max_elements = budget / 16;
    if max_elements == 0 {
        return None;
    }
    let n = 63 - max_elements.leading_zeros() as usize;
    Some(n.min(33))
}

#[cfg(unix)]
fn detect_max_sv_qubits() -> Option<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.trim().trim_end_matches(" kB").trim().parse().ok()?;
            let budget = (kb * 1024) / 2;
            let max_elements = budget / 16;
            if max_elements == 0 {
                return None;
            }
            let n = 63 - max_elements.leading_zeros() as usize;
            return Some(n.min(33));
        }
    }
    None
}

#[cfg(not(any(windows, unix)))]
fn detect_max_sv_qubits() -> Option<usize> {
    None
}

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

/// Minimum qubit count to route a sub-circuit to GPU when
/// [`BackendKind::StatevectorGpu`] is selected.
///
/// Below this threshold the dispatch layer builds a plain host
/// `StatevectorBackend` instead, keeping PCIe round-trips and kernel launch
/// latency off the critical path for circuits that fit comfortably in L3.
/// Empirically (GTX 1080 Ti, 2026-04-18), 10q random-depth-20 is 0.18x on
/// GPU vs CPU while 14q is 9.37x. 14 is the lowest known-win point; users
/// with faster GPUs or different workloads can override via the
/// `PRISM_GPU_MIN_QUBITS` environment variable.
#[cfg(feature = "gpu")]
pub(super) const GPU_MIN_QUBITS_DEFAULT: usize = 14;

#[cfg(feature = "gpu")]
pub(super) fn gpu_min_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        env_override_usize("PRISM_GPU_MIN_QUBITS").unwrap_or(GPU_MIN_QUBITS_DEFAULT)
    })
}

/// Whether this kind can build a GPU-backed backend for a circuit of the
/// given width. GPU backends share one CUDA stream per context and must not
/// execute concurrently, so multi-shot and multi-block drivers consult this
/// before spreading work across threads.
#[cfg(feature = "gpu")]
pub(super) fn may_route_to_gpu(kind: &BackendKind, num_qubits: usize) -> bool {
    matches!(kind, BackendKind::StatevectorGpu { .. }) && num_qubits >= gpu_min_qubits()
}

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
    FilteredStabilizer,
    FactoredStabilizer,
    StochasticPauli {
        num_samples: usize,
    },
    DeterministicPauli {
        epsilon: f64,
        max_terms: usize,
    },
    /// Statevector backed by a CUDA GPU execution context.
    ///
    /// Circuits (or decomposed sub-blocks) with fewer than 14 qubits
    /// (tunable via `PRISM_GPU_MIN_QUBITS`) transparently fall back to the
    /// host statevector path: small states do not survive PCIe and
    /// launch-latency overhead. Circuits too large for device VRAM also stay
    /// on the host path. Everything inside that window allocates a
    /// device-resident state and routes gate application through GPU kernels.
    ///
    /// Compose with [`crate::sim::run_with`] to get fusion plus independent-
    /// subsystem decomposition for free; each sub-block is evaluated against
    /// the crossover independently.
    #[cfg(feature = "gpu")]
    StatevectorGpu {
        context: Arc<GpuContext>,
    },
}

pub(super) fn validate_explicit_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    match kind {
        BackendKind::Stabilizer
        | BackendKind::FilteredStabilizer
        | BackendKind::FactoredStabilizer
            if !circuit.is_clifford_only() =>
        {
            return Err(PrismError::IncompatibleBackend {
                backend: "stabilizer".into(),
                reason: "circuit contains non-Clifford gates".into(),
            });
        }
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

pub(super) fn supports_fused_for_kind(kind: &BackendKind, circuit: &Circuit) -> bool {
    match kind {
        BackendKind::Stabilizer
        | BackendKind::FilteredStabilizer
        | BackendKind::FactoredStabilizer
        | BackendKind::StabilizerRank
        | BackendKind::StochasticPauli { .. }
        | BackendKind::DeterministicPauli { .. } => false,
        BackendKind::Auto => !(circuit.is_clifford_only() && circuit.has_entangling_gates()),
        _ => true,
    }
}

/// Build a `StatevectorBackend` configured for GPU execution when the
/// circuit falls inside the crossover window, otherwise a plain host
/// backend. Called from `select_dispatch` (which runs per sub-block after
/// decomposition) so small blocks transparently stay on CPU.
///
/// The upper bound keeps the state strictly below half of device VRAM: the
/// probability readback needs an extra `8 * 2^n` scratch buffer on top of the
/// `16 * 2^n` state, so a circuit at the whole-VRAM limit would fail mid-run.
/// Oversize circuits fall back to the host path, matching what explicit
/// `BackendKind::Statevector` would do. If the VRAM query fails (stub device),
/// the bound is skipped and any failure surfaces at allocation.
#[cfg(feature = "gpu")]
fn statevector_gpu_with_crossover(
    context: &Arc<GpuContext>,
    circuit: &Circuit,
    seed: u64,
) -> StatevectorBackend {
    let n = circuit.num_qubits;
    let fits_device = context
        .max_qubits_for_statevector()
        .map_or(true, |max| n < max);
    if n >= gpu_min_qubits() && fits_device {
        StatevectorBackend::new(seed).with_gpu(context.clone())
    } else {
        StatevectorBackend::new(seed)
    }
}

pub(super) fn select_dispatch(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    has_partial_independence: bool,
) -> DispatchAction {
    match kind {
        BackendKind::Auto => {
            if !circuit.has_entangling_gates() {
                DispatchAction::Backend(Box::new(ProductStateBackend::new(seed)))
            } else if circuit.is_clifford_only() {
                DispatchAction::Backend(Box::new(StabilizerBackend::new(seed)))
            } else if circuit.num_qubits > max_statevector_qubits() {
                if circuit.is_sparse_friendly() {
                    DispatchAction::Backend(Box::new(SparseBackend::new(seed)))
                } else {
                    DispatchAction::Backend(Box::new(MpsBackend::new(seed, AUTO_MPS_BOND_DIM)))
                }
            } else if has_partial_independence {
                DispatchAction::Backend(Box::new(crate::backend::factored::FactoredBackend::new(
                    seed,
                )))
            } else {
                DispatchAction::Backend(Box::new(StatevectorBackend::new(seed)))
            }
        }
        BackendKind::Statevector => {
            DispatchAction::Backend(Box::new(StatevectorBackend::new(seed)))
        }
        BackendKind::Stabilizer => DispatchAction::Backend(Box::new(StabilizerBackend::new(seed))),
        BackendKind::FilteredStabilizer => DispatchAction::Backend(Box::new(
            crate::backend::stabilizer::FilteredStabilizerBackend::new(seed),
        )),
        BackendKind::Sparse => DispatchAction::Backend(Box::new(SparseBackend::new(seed))),
        BackendKind::Mps { max_bond_dim } => {
            DispatchAction::Backend(Box::new(MpsBackend::new(seed, *max_bond_dim)))
        }
        BackendKind::ProductState => {
            DispatchAction::Backend(Box::new(ProductStateBackend::new(seed)))
        }
        BackendKind::TensorNetwork => {
            DispatchAction::Backend(Box::new(TensorNetworkBackend::new(seed)))
        }
        BackendKind::Factored => DispatchAction::Backend(Box::new(
            crate::backend::factored::FactoredBackend::new(seed),
        )),
        BackendKind::FactoredStabilizer => DispatchAction::Backend(Box::new(
            crate::backend::factored_stabilizer::FactoredStabilizerBackend::new(seed),
        )),
        BackendKind::StabilizerRank => DispatchAction::StabilizerRank,
        BackendKind::StochasticPauli { num_samples } => DispatchAction::StochasticPauli {
            num_samples: *num_samples,
        },
        BackendKind::DeterministicPauli { epsilon, max_terms } => {
            DispatchAction::DeterministicPauli {
                epsilon: *epsilon,
                max_terms: *max_terms,
            }
        }
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { context } => DispatchAction::Backend(Box::new(
            statevector_gpu_with_crossover(context, circuit, seed),
        )),
    }
}

pub(super) fn select_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    has_partial_independence: bool,
) -> Box<dyn Backend> {
    match select_dispatch(kind, circuit, seed, has_partial_independence) {
        DispatchAction::Backend(b) => b,
        _ => unreachable!("non-backend dispatch should be handled by caller"),
    }
}

#[inline]
pub(super) fn min_clifford_prefix_gates(num_qubits: usize) -> usize {
    (num_qubits * 2).max(16)
}

pub(super) fn has_temporal_clifford_opportunity(kind: &BackendKind, circuit: &Circuit) -> bool {
    if !matches!(kind, BackendKind::Auto) {
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

pub(super) fn try_temporal_clifford(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Option<Result<SimulationResult>> {
    if !matches!(kind, BackendKind::Auto) {
        return None;
    }
    if circuit.num_qubits > max_statevector_qubits() {
        return None;
    }
    let (prefix, tail) = circuit.clifford_prefix_split()?;
    if prefix.gate_count() < min_clifford_prefix_gates(circuit.num_qubits) {
        return None;
    }

    let mut stab = StabilizerBackend::new(seed);
    if let Err(e) = stab.init(prefix.num_qubits, prefix.num_classical_bits) {
        return Some(Err(e));
    }
    stab.enable_lazy_destab();
    for inst in &prefix.instructions {
        if let Err(e) = stab.apply(inst) {
            return Some(Err(e));
        }
    }

    let state = match stab.export_statevector() {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };

    let mut sv = StatevectorBackend::new(seed);
    if let Err(e) = sv.init_from_state(state, tail.num_classical_bits) {
        return Some(Err(e));
    }

    let fused_tail = crate::circuit::fusion::fuse_circuit(&tail, sv.supports_fused_gates());
    for inst in &fused_tail.instructions {
        if let Err(e) = sv.apply(inst) {
            return Some(Err(e));
        }
    }

    let probs = sv.probabilities().ok().map(Probabilities::Dense);

    Some(Ok(SimulationResult {
        classical_bits: sv.classical_results().to_vec(),
        probabilities: probs,
    }))
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_crossover_tests {
    use super::*;
    use crate::gates::Gate;
    use crate::sim::run_with;

    fn stub_kind() -> BackendKind {
        BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    fn assert_probs_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < 1e-10,
                "prob[{i}] = {a}, expected {e}, diff={}",
                (a - e).abs()
            );
        }
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

        let result = run_with(stub_kind(), &circuit, 42)
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
        assert_probs_close(&probs, &expected);
    }

    /// A 14q entangled chain sits exactly at the default crossover, so the
    /// dispatch layer must build a GPU backend. On the stub context
    /// `GpuState::new` fails with `BackendUnsupported`; seeing that error
    /// proves the GPU branch fired. Without this test, a regression that
    /// silently neuters GPU routing (dropping `with_gpu`, an off-by-one at
    /// the boundary) would leave the two negative tests green.
    #[test]
    fn crossover_circuit_routes_to_gpu() {
        let mut circuit = Circuit::new(14, 0);
        circuit.add_gate(Gate::H, &[0]);
        for q in 0..13 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }

        let err = run_with(stub_kind(), &circuit, 42)
            .expect_err("a 14q circuit must reach the stub GPU context");
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
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

        let cpu = run_with(BackendKind::Statevector, &circuit, 42).expect("cpu baseline");
        let gpu = run_with(stub_kind(), &circuit, 42).expect("stub must stay out of the way");

        let cpu_p = cpu.probabilities.expect("cpu probs").to_vec();
        let gpu_p = gpu.probabilities.expect("gpu probs").to_vec();
        assert_probs_close(&gpu_p, &cpu_p);
    }
}
