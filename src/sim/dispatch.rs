use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::SparseBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};

use super::{Probabilities, SimulationResult};

pub(super) enum DispatchAction {
    Backend(Box<dyn Backend>),
    StabilizerRank,
    StochasticPauli { num_samples: usize },
    DeterministicPauli { epsilon: f64, max_terms: usize },
}

pub(super) fn max_statevector_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_MAX_SV_QUBITS") {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
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
    Mps { max_bond_dim: usize },
    ProductState,
    TensorNetwork,
    Factored,
    StabilizerRank,
    FilteredStabilizer,
    FactoredStabilizer,
    StochasticPauli { num_samples: usize },
    DeterministicPauli { epsilon: f64, max_terms: usize },
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
