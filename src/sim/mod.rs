//! Simulation orchestration.
//!
//! Connects the circuit IR to a backend. This module is deliberately thin —
//! the complexity lives in the backends and the parser.

pub mod compiled;
pub mod homological;
pub mod noise;
pub mod stabilizer_rank;
pub mod unified_pauli;

use std::collections::HashMap;

use crate::backend::mps::MpsBackend;
use crate::backend::product::ProductStateBackend;
use crate::backend::sparse::SparseBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::tensornetwork::TensorNetworkBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};

/// Maximum qubit count for statevector simulation in Auto dispatch.
///
/// Dynamically computed from available system memory (50% budget).
/// Falls back to 28 (4 GB) when memory detection is unavailable.
/// Can be overridden via `PRISM_MAX_SV_QUBITS` environment variable.
fn max_statevector_qubits() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if let Ok(val) = std::env::var("PRISM_MAX_SV_QUBITS") {
            if let Ok(n) = val.parse::<usize>() {
                return n;
            }
        }
        detect_max_sv_qubits().unwrap_or(28)
    })
}

/// Detect system physical memory and compute the largest statevector qubit
/// count that fits within 50% of it. Each qubit doubles memory: 2^n × 16 bytes.
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

    let mut status: MemoryStatusEx = unsafe { std::mem::zeroed() };
    status.dw_length = std::mem::size_of::<MemoryStatusEx>() as u32;
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return None;
    }

    let budget = status.ull_total_phys / 2;
    let max_elements = budget / 16;
    if max_elements == 0 {
        return Some(28);
    }
    let n = 63 - max_elements.leading_zeros() as usize;
    Some(n.min(33)) // cap at 33 qubits (128 GB state)
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
                return Some(28);
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

/// Default bond dimension for MPS when auto-selected.
const AUTO_MPS_BOND_DIM: usize = 256;

/// Maximum T-gate count for automatic Clifford+T dispatch.
/// At t=12, stabilizer rank creates 2^12 = 4096 branches.
const MAX_AUTO_T_COUNT: usize = 12;

/// Maximum qubit count for stabilizer rank exact probability computation.
const MAX_STABILIZER_RANK_QUBITS: usize = 25;

/// Minimum qubit count for subsystem decomposition.
///
/// Below this threshold, statevector is tiny and the analysis overhead
/// exceeds any savings from decomposition.
const MIN_DECOMPOSITION_QUBITS: usize = 8;

/// Check whether decomposition is worthwhile based on block structure.
///
/// Decomposition only helps when it meaningfully reduces the largest
/// block size. If the largest block is nearly the full circuit, the
/// Kronecker merge cost (~3 × 2^N alloc) exceeds the savings.
fn should_decompose(components: &[Vec<usize>], total_qubits: usize) -> bool {
    let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
    // Require at least 3 qubits of savings to justify merge overhead.
    max_block + 3 <= total_qubits
}

/// Options controlling what `sim::run` computes.
///
/// Use [`Default`] for full output (probabilities included), or
/// [`SimOptions::classical_only`] to skip the probabilities pass.
#[derive(Debug, Clone)]
pub struct SimOptions {
    /// Whether to compute the full probability vector after simulation.
    ///
    /// Skipping this saves one complete statevector pass (~2 ms at 20 qubits).
    /// When `false`, `SimulationResult::probabilities` will be `None`.
    pub probabilities: bool,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            probabilities: true,
        }
    }
}

impl SimOptions {
    /// Options for measurement-only runs: skip probability extraction.
    pub fn classical_only() -> Self {
        Self {
            probabilities: false,
        }
    }
}

/// A single block in a factored probability distribution.
///
/// Each block represents the marginal probabilities for one independent
/// subsystem. The `mask` indicates which global qubit positions belong
/// to this block, and `probs` holds the 2^k marginal distribution.
#[derive(Debug, Clone)]
pub struct FactoredBlock {
    /// Marginal probability vector for this block (length 2^k).
    pub probs: Vec<f64>,
    /// Bitmask of global qubit positions belonging to this block.
    pub mask: u64,
}

/// Probability distribution over computational basis states.
///
/// For monolithic simulations this wraps a dense `Vec<f64>` of length 2^n.
/// For decomposed simulations with independent subsystems, this stores
/// per-block marginal distributions that are multiplied on demand —
/// avoiding the O(2^N) Kronecker product unless explicitly requested.
#[derive(Debug, Clone)]
pub enum Probabilities {
    /// Full probability vector of length 2^n.
    Dense(Vec<f64>),
    /// Lazy Kronecker product of independent block distributions.
    Factored {
        /// Per-block marginal probability vectors and bitmasks.
        blocks: Vec<FactoredBlock>,
        /// Total qubit count across all blocks.
        total_qubits: usize,
    },
}

impl Probabilities {
    /// Number of basis states (2^n).
    pub fn len(&self) -> usize {
        match self {
            Probabilities::Dense(v) => v.len(),
            Probabilities::Factored { total_qubits, .. } => 1 << total_qubits,
        }
    }

    /// Always false — a probability distribution has at least one state.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Probability of a single computational basis state. O(1) for dense,
    /// O(K) for factored where K is the number of independent blocks.
    ///
    /// # Panics
    /// Panics if `index >= self.len()`.
    pub fn get(&self, index: usize) -> f64 {
        match self {
            Probabilities::Dense(v) => v[index],
            Probabilities::Factored { blocks, .. } => {
                let mut p = 1.0;
                for block in blocks {
                    let local = extract_block_bits(index, block.mask);
                    p *= block.probs[local];
                }
                p
            }
        }
    }

    /// Iterate over all basis-state probabilities in order.
    ///
    /// For `Dense` this is a direct slice iteration. For `Factored` each
    /// probability is computed on the fly in O(K) per element.
    pub fn iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        match self {
            Probabilities::Dense(v) => Box::new(v.iter().copied()),
            Probabilities::Factored {
                blocks,
                total_qubits,
            } => {
                let n = 1usize << total_qubits;
                Box::new((0..n).map(move |i| {
                    let mut p = 1.0;
                    for block in blocks {
                        let local = extract_block_bits(i, block.mask);
                        p *= block.probs[local];
                    }
                    p
                }))
            }
        }
    }

    /// Materialize the full probability vector. O(1) clone for dense,
    /// O(K × 2^N) for factored. Prefer [`Probabilities::get`] for spot-checking.
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            Probabilities::Dense(v) => v.clone(),
            Probabilities::Factored {
                blocks,
                total_qubits,
            } => {
                let n = 1usize << total_qubits;
                let mut result = vec![0.0f64; n];
                #[cfg(feature = "parallel")]
                {
                    const MIN_PAR_STATES: usize = 1 << 14;
                    if n >= MIN_PAR_STATES {
                        use rayon::prelude::*;
                        crate::backend::init_thread_pool();
                        result.par_iter_mut().enumerate().for_each(|(i, slot)| {
                            let mut p = 1.0;
                            for block in blocks {
                                let local = extract_block_bits(i, block.mask);
                                p *= block.probs[local];
                            }
                            *slot = p;
                        });
                        return result;
                    }
                }
                for (i, slot) in result.iter_mut().enumerate() {
                    let mut p = 1.0;
                    for block in blocks {
                        let local = extract_block_bits(i, block.mask);
                        p *= block.probs[local];
                    }
                    *slot = p;
                }
                result
            }
        }
    }
}

impl std::ops::Index<usize> for Probabilities {
    type Output = f64;

    /// Index into a dense probability vector.
    ///
    /// Only works for `Dense`. Panics on `Factored` because `Index` must
    /// return `&f64` and factored values are computed, not stored.
    /// Use [`Probabilities::get`] or [`Probabilities::iter`] instead.
    fn index(&self, index: usize) -> &f64 {
        match self {
            Probabilities::Dense(v) => &v[index],
            Probabilities::Factored { .. } => {
                panic!("cannot index Factored probabilities; use .get(i) or .to_vec()")
            }
        }
    }
}

/// Extract the bits of `global_index` at positions set in `mask`,
/// packing them into contiguous low bits.
#[inline]
fn extract_block_bits(global_index: usize, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi2") {
            return unsafe { core::arch::x86_64::_pext_u64(global_index as u64, mask) as usize };
        }
    }
    let mut result = 0usize;
    let mut bit = 0;
    let mut m = mask;
    while m != 0 {
        let pos = m.trailing_zeros() as usize;
        if global_index & (1 << pos) != 0 {
            result |= 1 << bit;
        }
        bit += 1;
        m &= m.wrapping_sub(1);
    }
    result
}

/// Result of a simulation run.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Classical measurement outcomes, indexed by classical bit number.
    /// `true` = measured |1⟩.
    pub classical_bits: Vec<bool>,
    /// Probability of each computational basis state (length 2^n).
    /// `None` if the backend does not support probability extraction,
    /// or if [`SimOptions::probabilities`] was `false`.
    pub probabilities: Option<Probabilities>,
}

/// Backend selection for [`run_with`] and [`run_with_opts`].
///
/// Use `Auto` for automatic dispatch based on circuit properties, or specify
/// a backend explicitly.
#[derive(Debug, Clone)]
pub enum BackendKind {
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
    Auto,
    /// Full state-vector simulation (exact, exponential memory).
    Statevector,
    /// Clifford-only O(n^2) stabilizer tableau.
    Stabilizer,
    /// Sparse state-vector, O(k) in non-zero amplitudes.
    Sparse,
    /// Matrix Product State with bounded bond dimension.
    Mps {
        /// Maximum bond dimension (controls accuracy vs speed).
        max_bond_dim: usize,
    },
    /// Per-qubit product state (non-entangling circuits only).
    ProductState,
    /// Tensor network with deferred contraction.
    TensorNetwork,
    /// Dynamic split-state with on-demand merging.
    Factored,
    /// Stabilizer rank decomposition for Clifford+T circuits.
    ///
    /// Maintains a weighted sum of stabilizer states. Clifford gates are O(n²)
    /// per term. Each T gate doubles the term count via T = α·I + β·Z.
    /// Accumulates weighted amplitudes for exact probabilities. Limits: t ≤ 20, n ≤ 25.
    StabilizerRank,
    /// Filtered stabilizer: per-cluster tableaux with dynamic merging.
    ///
    /// Starts with one qubit per cluster. Gates within a cluster operate on
    /// a small tableau. Cross-cluster 2q gates merge tableaux. Independent
    /// subsystems never merge, giving O(block_size²) per gate vs O(n²).
    FilteredStabilizer,
    /// Stochastic Pauli Propagation for Clifford+T circuits.
    ///
    /// Backward-propagates measurement observables through the circuit as Pauli
    /// strings. Clifford gates conjugate in O(1). T gates branch stochastically.
    /// Per-path cost O(d×n/64), independent of T-gate count. Returns marginal
    /// probabilities via Monte Carlo estimation.
    StochasticPauli {
        /// Number of stochastic paths per measurement qubit.
        num_samples: usize,
    },
    /// Deterministic Sparse Pauli Dynamics for Clifford+T circuits.
    ///
    /// Backward-propagates measurement observables as a weighted sum of Pauli
    /// strings stored in a HashMap. T gates deterministically branch X/Y terms.
    /// Identical strings auto-merge. Optional ε-truncation for approximate mode.
    /// Exact for small T-counts, approximate with bounded error for larger ones.
    DeterministicPauli {
        /// Drop Pauli terms with coefficient magnitude below this threshold.
        /// Use 0.0 for exact mode.
        epsilon: f64,
        /// Maximum number of terms before forced truncation. Use 0 for unlimited.
        max_terms: usize,
    },
}

/// Validate that an explicitly-chosen backend is compatible with the circuit.
///
/// Auto dispatch never fails — it always selects a valid backend. This
/// validation only applies to explicit backend requests, catching mistakes
/// early with clear error messages instead of cryptic backend errors later.
fn validate_explicit_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    match kind {
        BackendKind::Stabilizer | BackendKind::FilteredStabilizer => {
            if !circuit.is_clifford_only() {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer".into(),
                    reason: "circuit contains non-Clifford gates".into(),
                });
            }
        }
        BackendKind::ProductState => {
            if circuit.has_entangling_gates() {
                return Err(PrismError::IncompatibleBackend {
                    backend: "productstate".into(),
                    reason: "circuit contains entangling gates".into(),
                });
            }
        }
        BackendKind::StabilizerRank => {
            if !circuit.has_t_gates() {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason: "circuit has no T gates; use Stabilizer instead".into(),
                });
            }
        }
        _ => {}
    }
    Ok(())
}

fn supports_fused_for_kind(kind: &BackendKind, circuit: &Circuit) -> bool {
    match kind {
        BackendKind::Stabilizer
        | BackendKind::FilteredStabilizer
        | BackendKind::StabilizerRank
        | BackendKind::StochasticPauli { .. }
        | BackendKind::DeterministicPauli { .. } => false,
        BackendKind::Auto => !(circuit.is_clifford_only() && circuit.has_entangling_gates()),
        _ => true,
    }
}

fn select_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    has_partial_independence: bool,
) -> Box<dyn Backend> {
    match kind {
        BackendKind::Auto => {
            if !circuit.has_entangling_gates() {
                Box::new(ProductStateBackend::new(seed))
            } else if circuit.is_clifford_only() {
                Box::new(StabilizerBackend::new(seed))
            } else if circuit.num_qubits > max_statevector_qubits() {
                if circuit.is_sparse_friendly() {
                    Box::new(SparseBackend::new(seed))
                } else {
                    Box::new(MpsBackend::new(seed, AUTO_MPS_BOND_DIM))
                }
            } else if has_partial_independence {
                Box::new(crate::backend::factored::FactoredBackend::new(seed))
            } else {
                Box::new(StatevectorBackend::new(seed))
            }
        }
        BackendKind::Statevector => Box::new(StatevectorBackend::new(seed)),
        BackendKind::Stabilizer => Box::new(StabilizerBackend::new(seed)),
        BackendKind::FilteredStabilizer => Box::new(
            crate::backend::stabilizer::FilteredStabilizerBackend::new(seed),
        ),
        BackendKind::Sparse => Box::new(SparseBackend::new(seed)),
        BackendKind::Mps { max_bond_dim } => Box::new(MpsBackend::new(seed, *max_bond_dim)),
        BackendKind::ProductState => Box::new(ProductStateBackend::new(seed)),
        BackendKind::TensorNetwork => Box::new(TensorNetworkBackend::new(seed)),
        BackendKind::Factored => Box::new(crate::backend::factored::FactoredBackend::new(seed)),
        BackendKind::StabilizerRank => {
            unreachable!("StabilizerRank is handled before select_backend")
        }
        BackendKind::StochasticPauli { .. } => {
            unreachable!("StochasticPauli is handled before select_backend")
        }
        BackendKind::DeterministicPauli { .. } => {
            unreachable!("DeterministicPauli is handled before select_backend")
        }
    }
}

/// Execute blocks, choosing between parallel and sequential dispatch.
///
/// When all blocks are small (< `MAX_BLOCK_QUBITS_FOR_PAR`), we run them
/// concurrently via Rayon — each block's internal gate kernels stay sequential,
/// and block-level parallelism uses the thread pool. When any block is large,
/// each block already uses Rayon internally for gate kernels, so we run
/// blocks sequentially to avoid thread oversubscription.
fn run_blocks_maybe_par(
    kind: &BackendKind,
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    _components: &[Vec<usize>],
    seed: u64,
    opts: &SimOptions,
    k: usize,
) -> Vec<Result<SimulationResult>> {
    #[cfg(feature = "parallel")]
    {
        let all_small = _components
            .iter()
            .all(|c| c.len() < MAX_BLOCK_QUBITS_FOR_PAR);
        if all_small && k >= 2 {
            use rayon::prelude::*;
            crate::backend::init_thread_pool();
            return (0..k)
                .into_par_iter()
                .map(|i| run_subcircuit(kind, &partitions[i].0, seed.wrapping_add(i as u64), opts))
                .collect();
        }
    }
    (0..k)
        .map(|i| run_subcircuit(kind, &partitions[i].0, seed.wrapping_add(i as u64), opts))
        .collect()
}

/// Execute a pre-extracted sub-circuit on an appropriate backend.
///
/// Delegates to `run_with_opts` so subcircuits benefit from the full Auto
/// dispatch tree (Clifford+T → StabilizerRank, temporal Clifford, etc.).
fn run_subcircuit(
    kind: &BackendKind,
    sub_circuit: &Circuit,
    block_seed: u64,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let block_kind = if matches!(kind, BackendKind::Auto) {
        BackendKind::Auto
    } else {
        kind.clone()
    };
    run_with_opts(block_kind, sub_circuit, block_seed, opts.clone())
}

#[cfg(feature = "parallel")]
const MAX_BLOCK_QUBITS_FOR_PAR: usize = 14;

/// Simulate independent subsystems separately, then merge results.
///
/// Uses `partition_subcircuits` for O(N) instruction routing (single pass)
/// instead of K calls to `extract_subcircuit` (K passes).
fn run_decomposed(
    kind: &BackendKind,
    components: &[Vec<usize>],
    circuit: &Circuit,
    seed: u64,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let partitions = circuit.partition_subcircuits(components);
    let k = partitions.len();

    let results: Vec<Result<SimulationResult>> =
        run_blocks_maybe_par(kind, &partitions, components, seed, opts, k);

    merge_decomposed_results(
        results,
        components,
        &partitions,
        circuit.num_classical_bits,
        circuit.num_qubits,
        opts,
    )
}

fn merge_decomposed_results(
    results: Vec<Result<SimulationResult>>,
    components: &[Vec<usize>],
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    num_classical_bits: usize,
    num_qubits: usize,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let mut factored_blocks: Vec<FactoredBlock> = Vec::new();
    let mut merged_classical = vec![false; num_classical_bits];

    for (i, result) in results.into_iter().enumerate() {
        let result = result?;
        let (_, ref qubit_map, ref classical_map) = partitions[i];

        for (local_idx, &global_idx) in classical_map.iter().enumerate() {
            merged_classical[global_idx] = result.classical_bits[local_idx];
        }

        if opts.probabilities && num_qubits <= 64 {
            if let Some(probs) = result.probabilities {
                let dense = probs.to_vec();
                let mut mask = 0u64;
                for &global_qubit in qubit_map {
                    mask |= 1u64 << global_qubit;
                }
                factored_blocks.push(FactoredBlock { probs: dense, mask });
            }
        }
    }

    let probabilities = if opts.probabilities && factored_blocks.len() == components.len() {
        Some(Probabilities::Factored {
            blocks: factored_blocks,
            total_qubits: num_qubits,
        })
    } else {
        None
    };

    Ok(SimulationResult {
        classical_bits: merged_classical,
        probabilities,
    })
}

fn run_decomposed_prefused(
    kind: &BackendKind,
    components: &[Vec<usize>],
    partitions: &[(Circuit, Vec<usize>, Vec<usize>)],
    fused_blocks: &[std::borrow::Cow<'_, Circuit>],
    seed: u64,
    opts: &SimOptions,
    original_circuit: &Circuit,
) -> Result<SimulationResult> {
    let k = partitions.len();
    let results: Vec<Result<SimulationResult>> = (0..k)
        .map(|i| {
            let block_seed = seed.wrapping_add(i as u64);
            let sub = &partitions[i].0;
            let block_kind = if matches!(kind, BackendKind::Auto) {
                BackendKind::Auto
            } else {
                kind.clone()
            };
            if !matches!(block_kind, BackendKind::Auto) {
                validate_explicit_backend(&block_kind, sub)?;
            }
            let mut backend = select_backend(&block_kind, sub, block_seed, false);
            execute_circuit(&mut *backend, &fused_blocks[i], opts)
        })
        .collect();
    merge_decomposed_results(
        results,
        components,
        partitions,
        original_circuit.num_classical_bits,
        original_circuit.num_qubits,
        opts,
    )
}

/// Combine per-block probability vectors via Kronecker product.
///
/// Two-pass O(2^N) algorithm:
/// 1. In-place Kronecker product — single allocation, reverse-iteration expansion
/// 2. Bit permutation to map natural (block-sequential) bit positions
///    to global qubit positions (parallelized at ≥2^14 states)
pub(crate) fn merge_probabilities(
    blocks: &[(Vec<f64>, Vec<usize>)],
    total_qubits: usize,
) -> Vec<f64> {
    let n_states = 1usize << total_qubits;

    // Pass 1: In-place Kronecker product — single 2^N allocation.
    // Expand in reverse order so writes never clobber unread source data.
    let mut result = vec![0.0f64; n_states];
    result[0] = 1.0;
    let mut cur_len = 1usize;

    for (probs, _) in blocks {
        let block_len = probs.len();
        let new_len = cur_len * block_len;
        for i in (0..cur_len).rev() {
            let r = result[i];
            for j in 0..block_len {
                result[i * block_len + j] = r * probs[j];
            }
        }
        cur_len = new_len;
    }
    debug_assert_eq!(cur_len, n_states);

    // Build natural-to-global bit position mapping.
    // Kronecker A ⊗ B puts A's bits in the MSBs and B's in the LSBs.
    // After iterating blocks in forward order, the LAST block occupies
    // the lowest bit positions in the result index.
    let mut natural_to_global = Vec::with_capacity(total_qubits);
    for (_probs, qubits) in blocks.iter().rev() {
        natural_to_global.extend_from_slice(qubits);
    }

    if natural_to_global.iter().enumerate().all(|(i, &g)| i == g) {
        return result;
    }

    // Pass 2: Bit permutation via perm table.
    let mut perm = vec![0usize; n_states];
    for (nat_bit, &global_bit) in natural_to_global.iter().enumerate() {
        let half = 1usize << nat_bit;
        for i in 0..half {
            perm[i | half] = perm[i] | (1 << global_bit);
        }
    }

    let mut permuted = vec![0.0f64; n_states];

    #[cfg(feature = "parallel")]
    {
        const MIN_PAR_PERM: usize = 1 << 14;
        if n_states >= MIN_PAR_PERM {
            use rayon::prelude::*;
            result
                .par_iter()
                .zip(perm.par_iter())
                .for_each(|(&prob, &global_idx)| {
                    // SAFETY: perm is a bijection so each global_idx is written exactly once.
                    // No two threads write to the same index.
                    unsafe {
                        let ptr = permuted.as_ptr() as *mut f64;
                        *ptr.add(global_idx) = prob;
                    }
                });
            return permuted;
        }
    }

    for (nat_idx, &prob) in result.iter().enumerate() {
        permuted[perm[nat_idx]] = prob;
    }
    permuted
}

/// Minimum Clifford prefix gate count for temporal decomposition to be worthwhile.
///
/// The stabilizer→statevector export costs O(2^n × n). Each prefix gate
/// saved is ~2^n ops. At small n the export dominates; at large n the prefix
/// savings dominate. Scale with qubit count: minimum 2*n gates, floor at 16.
#[inline]
fn min_clifford_prefix_gates(num_qubits: usize) -> usize {
    (num_qubits * 2).max(16)
}

fn has_temporal_clifford_opportunity(kind: &BackendKind, circuit: &Circuit) -> bool {
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
            Instruction::Measure { .. } | Instruction::Conditional { .. } => break,
            Instruction::Barrier { .. } => {}
        }
    }
    prefix_gates >= min_gates && prefix_gates < circuit.instructions.len()
}

/// Try temporal Clifford decomposition: run a Clifford prefix on Stabilizer,
/// convert to statevector, then continue the tail on Statevector.
///
/// Returns `None` if the circuit doesn't qualify (no prefix, too few prefix
/// gates, too many qubits for export, or not an Auto dispatch).
fn try_temporal_clifford(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: &SimOptions,
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

    // Phase 1: Run Clifford prefix on Stabilizer
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

    // Phase 2: Export stabilizer state as dense statevector
    let state = match stab.export_statevector() {
        Ok(s) => s,
        Err(e) => return Some(Err(e)),
    };

    // Phase 3: Initialize statevector backend from exported state
    let mut sv = StatevectorBackend::new(seed);
    if let Err(e) = sv.init_from_state(state, tail.num_classical_bits) {
        return Some(Err(e));
    }

    // Phase 4: Fuse and execute the tail
    let fused_tail = crate::circuit::fusion::fuse_circuit(&tail, sv.supports_fused_gates());
    for inst in &fused_tail.instructions {
        if let Err(e) = sv.apply(inst) {
            return Some(Err(e));
        }
    }

    let probs = if opts.probabilities {
        sv.probabilities().ok().map(Probabilities::Dense)
    } else {
        None
    };

    Some(Ok(SimulationResult {
        classical_bits: sv.classical_results().to_vec(),
        probabilities: probs,
    }))
}

/// Core execution: fuse, init, apply, extract.
fn execute(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    let fused = crate::circuit::fusion::fuse_circuit(circuit, backend.supports_fused_gates());
    execute_circuit(backend, &fused, opts)
}

/// Shared init → apply → extract logic.
fn execute_circuit(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    backend.apply_instructions(&circuit.instructions)?;

    let probs = if opts.probabilities {
        backend.probabilities().ok().map(Probabilities::Dense)
    } else {
        None
    };

    Ok(SimulationResult {
        classical_bits: backend.classical_results().to_vec(),
        probabilities: probs,
    })
}

/// Execute a circuit with automatic backend selection.
///
/// The simplest entry point. Uses [`BackendKind::Auto`] to select the
/// optimal backend based on circuit properties, then runs the circuit.
pub fn run(circuit: &Circuit, seed: u64) -> Result<SimulationResult> {
    run_with(BackendKind::Auto, circuit, seed)
}

/// Execute a circuit with explicit backend selection.
///
/// Constructs the backend internally based on [`BackendKind`], then runs
/// the circuit. For a pre-constructed backend instance, use [`run_on`].
pub fn run_with(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<SimulationResult> {
    run_with_opts(kind, circuit, seed, SimOptions::default())
}

/// Execute a circuit with explicit backend selection and output options.
///
/// Like [`run_with`], but accepts [`SimOptions`] to skip expensive outputs
/// (e.g., probability extraction) when only classical results are needed.
pub fn run_with_opts(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
) -> Result<SimulationResult> {
    if !matches!(kind, BackendKind::Auto) {
        validate_explicit_backend(&kind, circuit)?;
    }
    let mut has_partial_independence = false;
    if circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        let components = circuit.independent_subsystems();
        if components.len() > 1 {
            if should_decompose(&components, circuit.num_qubits) {
                return run_decomposed(&kind, &components, circuit, seed, &opts);
            }
            has_partial_independence = true;
        }
    }
    if matches!(kind, BackendKind::StabilizerRank) {
        let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
        return Ok(SimulationResult {
            probabilities: Some(Probabilities::Dense(sr.probabilities)),
            classical_bits: vec![],
        });
    }
    if let BackendKind::StochasticPauli { num_samples } = kind {
        let spp = unified_pauli::run_spp(circuit, num_samples, seed)?;
        let probs = unified_pauli::spp_to_probabilities(&spp);
        return Ok(SimulationResult {
            probabilities: Some(Probabilities::Dense(probs)),
            classical_bits: vec![],
        });
    }
    if let BackendKind::DeterministicPauli { epsilon, max_terms } = kind {
        let spd = unified_pauli::run_spd(circuit, epsilon, max_terms)?;
        let probs = unified_pauli::spd_to_probabilities(&spd);
        return Ok(SimulationResult {
            probabilities: Some(Probabilities::Dense(probs)),
            classical_bits: vec![],
        });
    }
    // Auto: Clifford+T with small T-count → stabilizer rank decomposition.
    // Each branch is O(n²) stabilizer instead of O(2^n) statevector, but we
    // need 2^t branches for exact probabilities, so limit t ≤ 12 (4096 branches).
    if matches!(kind, BackendKind::Auto)
        && circuit.is_clifford_plus_t()
        && circuit.has_t_gates()
        && circuit.t_count() <= MAX_AUTO_T_COUNT
        && circuit.num_qubits <= MAX_STABILIZER_RANK_QUBITS
    {
        let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
        return Ok(SimulationResult {
            probabilities: Some(Probabilities::Dense(sr.probabilities)),
            classical_bits: vec![],
        });
    }
    // Temporal Clifford decomposition: if the circuit has a substantial Clifford
    // prefix followed by non-Clifford gates, run the prefix on Stabilizer and
    // the tail on Statevector with the exported state.
    if let Some(result) = try_temporal_clifford(&kind, circuit, seed, &opts) {
        return result;
    }
    let mut backend = select_backend(&kind, circuit, seed, has_partial_independence);
    execute(&mut *backend, circuit, &opts)
}

/// Execute a circuit on a pre-constructed backend.
///
/// Use this when you need direct control over the backend instance
/// (e.g., testing a specific backend). For automatic dispatch, use [`run`].
pub fn run_on(backend: &mut dyn Backend, circuit: &Circuit) -> Result<SimulationResult> {
    run_on_opts(backend, circuit, &SimOptions::default())
}

/// Execute a circuit on a pre-constructed backend with output options.
///
/// Like [`run_on`], but accepts [`SimOptions`] to skip expensive outputs.
pub fn run_on_opts(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<SimulationResult> {
    execute(backend, circuit, opts)
}

/// Parse an OpenQASM string and execute with automatic backend selection.
pub fn run_qasm(qasm: &str, seed: u64) -> Result<SimulationResult> {
    let circuit = crate::circuit::openqasm::parse(qasm)?;
    run(&circuit, seed)
}

/// Result of a multi-shot simulation run.
#[derive(Debug, Clone)]
pub struct ShotsResult {
    /// Classical measurement outcomes for each shot.
    /// `shots[i][j]` is the j-th classical bit from the i-th shot.
    pub shots: Vec<Vec<bool>>,
    /// Probability distribution from the final shot (pre-measurement state
    /// is identical across shots, so one extraction suffices).
    pub probabilities: Option<Probabilities>,
}

impl ShotsResult {
    /// Build a frequency histogram of measurement outcomes.
    pub fn counts(&self) -> HashMap<Vec<bool>, usize> {
        let mut map = HashMap::new();
        for shot in &self.shots {
            *map.entry(shot.clone()).or_insert(0) += 1;
        }
        map
    }
}

impl std::fmt::Display for ShotsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = self.counts();
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        for (bits, count) in &entries {
            let bitstring: String = bits.iter().map(|&b| if b { '1' } else { '0' }).collect();
            writeln!(f, "{bitstring}: {count}")?;
        }
        Ok(())
    }
}

fn build_cdf(probs: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(probs.len());
    let mut acc = 0.0;
    for &p in probs {
        acc += p;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn sample_from_cdf(cdf: &[f64], r: f64) -> usize {
    match cdf.binary_search_by(|p| p.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i.min(cdf.len() - 1),
    }
}

fn sample_shots(
    probs: &Probabilities,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if meas_map.is_empty() {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    let mut shots = Vec::with_capacity(num_shots);

    match probs {
        Probabilities::Dense(v) => {
            let cdf = build_cdf(v);
            for _ in 0..num_shots {
                let r: f64 = rng.gen();
                let state_idx = sample_from_cdf(&cdf, r);
                let mut bits = vec![false; num_classical_bits];
                for &(qubit, cbit) in meas_map {
                    bits[cbit] = (state_idx >> qubit) & 1 == 1;
                }
                shots.push(bits);
            }
        }
        Probabilities::Factored { blocks, .. } => {
            let block_cdfs: Vec<Vec<f64>> = blocks.iter().map(|b| build_cdf(&b.probs)).collect();
            for _ in 0..num_shots {
                let mut global_idx = 0usize;
                for (block, cdf) in blocks.iter().zip(block_cdfs.iter()) {
                    let r: f64 = rng.gen();
                    let local_idx = sample_from_cdf(cdf, r);
                    let mut m = block.mask;
                    let mut bit = 0;
                    while m != 0 {
                        let pos = m.trailing_zeros() as usize;
                        if local_idx & (1 << bit) != 0 {
                            global_idx |= 1 << pos;
                        }
                        bit += 1;
                        m &= m.wrapping_sub(1);
                    }
                }
                let mut bits = vec![false; num_classical_bits];
                for &(qubit, cbit) in meas_map {
                    bits[cbit] = (global_idx >> qubit) & 1 == 1;
                }
                shots.push(bits);
            }
        }
    }

    shots
}

/// Execute a circuit multiple times, collecting measurement outcomes.
pub fn run_shots(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    run_shots_with(BackendKind::Auto, circuit, num_shots, seed)
}

/// Execute a circuit multiple times with a random seed.
///
/// Like [`run_shots`], but generates a random base seed using system
/// entropy. Results are non-deterministic across runs.
pub fn run_shots_random(circuit: &Circuit, num_shots: usize) -> Result<ShotsResult> {
    run_shots_with(BackendKind::Auto, circuit, num_shots, rand::random())
}

/// Execute a circuit multiple times with explicit backend selection.
pub fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    // Compiled sampler: O(n²·m) compile + O(r·m/64) per shot with LUT.
    // Always polynomial — avoids the O(2^k) probability computation path.
    if (matches!(
        kind,
        BackendKind::Auto | BackendKind::Stabilizer | BackendKind::FilteredStabilizer
    )) && circuit.is_clifford_only()
        && circuit
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Measure { .. }))
        && num_shots >= 2
    {
        return compiled::run_shots_compiled(circuit, num_shots, seed);
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let result = run_with_opts(kind.clone(), &stripped, seed, SimOptions::default())?;
        if let Some(probs) = result.probabilities {
            let meas_map = circuit.measurement_map();
            let shots = sample_shots(
                &probs,
                &meas_map,
                circuit.num_classical_bits,
                num_shots,
                seed,
            );
            return Ok(ShotsResult {
                shots,
                probabilities: Some(probs),
            });
        }
    }

    // Slow path: mid-circuit measurements require per-shot simulation.
    // Pre-compute seed-independent analysis to avoid redundant work.
    if !matches!(kind, BackendKind::Auto) {
        validate_explicit_backend(&kind, circuit)?;
    }

    let mut has_partial_independence = false;
    let decompose = if circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        let comps = circuit.independent_subsystems();
        if comps.len() > 1 {
            if should_decompose(&comps, circuit.num_qubits) {
                Some(comps)
            } else {
                has_partial_independence = true;
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    if matches!(kind, BackendKind::StabilizerRank) {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }
    // Auto: Clifford+T with small T-count → stabilizer rank shot sampling.
    // Each shot samples a random Clifford branch (O(n²) stabilizer), avoiding
    // exponential statevector/MPS cost. No qubit limit for shots.
    if matches!(kind, BackendKind::Auto)
        && circuit.is_clifford_plus_t()
        && circuit.has_t_gates()
        && circuit.t_count() <= MAX_AUTO_T_COUNT
    {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }

    if has_temporal_clifford_opportunity(&kind, circuit) {
        return run_shots_fallback(&kind, circuit, num_shots, seed);
    }

    let supports_fused = supports_fused_for_kind(&kind, circuit);
    let mut shots = Vec::with_capacity(num_shots);
    let mut probabilities = None;

    if let Some(ref comps) = decompose {
        let partitions = circuit.partition_subcircuits(comps);
        let fused_blocks: Vec<_> = partitions
            .iter()
            .map(|(sub, _, _)| {
                crate::circuit::fusion::fuse_circuit(sub, supports_fused_for_kind(&kind, sub))
            })
            .collect();

        for i in 0..num_shots {
            let shot_seed = seed.wrapping_add(i as u64);
            let opts = if i == num_shots - 1 {
                SimOptions::default()
            } else {
                SimOptions::classical_only()
            };
            let result = run_decomposed_prefused(
                &kind,
                comps,
                &partitions,
                &fused_blocks,
                shot_seed,
                &opts,
                circuit,
            )?;
            shots.push(result.classical_bits);
            if i == num_shots - 1 {
                probabilities = result.probabilities;
            }
        }
    } else {
        let fused = crate::circuit::fusion::fuse_circuit(circuit, supports_fused);

        for i in 0..num_shots {
            let shot_seed = seed.wrapping_add(i as u64);
            let opts = if i == num_shots - 1 {
                SimOptions::default()
            } else {
                SimOptions::classical_only()
            };
            let mut backend = select_backend(&kind, circuit, shot_seed, has_partial_independence);
            let result = execute_circuit(&mut *backend, &fused, &opts)?;
            shots.push(result.classical_bits);
            if i == num_shots - 1 {
                probabilities = result.probabilities;
            }
        }
    }

    Ok(ShotsResult {
        shots,
        probabilities,
    })
}

/// Execute a noisy circuit for multiple shots with explicit backend selection.
///
/// For Clifford circuits with Auto/Stabilizer/FilteredStabilizer backends,
/// uses the compiled noisy sampler (fast O(n²·m) compile + O(events·m/64) per shot).
/// For all other cases, falls back to per-shot simulation with noise injection.
pub fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let use_compiled = matches!(
        kind,
        BackendKind::Auto | BackendKind::Stabilizer | BackendKind::FilteredStabilizer
    ) && circuit.is_clifford_only();

    if use_compiled {
        return noise::run_shots_noisy(circuit, noise_model, num_shots, seed);
    }

    noise::run_shots_noisy_brute_with(
        |s| select_backend(&kind, circuit, s, false),
        circuit,
        noise_model,
        num_shots,
        seed,
    )
}

fn run_shots_fallback(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let mut shots = Vec::with_capacity(num_shots);
    let mut probabilities = None;
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let opts = if i == num_shots - 1 {
            SimOptions::default()
        } else {
            SimOptions::classical_only()
        };
        let result = run_with_opts(kind.clone(), circuit, shot_seed, opts)?;
        shots.push(result.classical_bits);
        if i == num_shots - 1 {
            probabilities = result.probabilities;
        }
    }
    Ok(ShotsResult {
        shots,
        probabilities,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::Gate;

    fn make_clifford_circuit() -> Circuit {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::S, &[0]);
        c
    }

    fn make_product_circuit() -> Circuit {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Rx(1.0), &[1]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::Y, &[3]);
        c
    }

    fn make_general_circuit() -> Circuit {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c
    }

    #[test]
    fn test_circuit_is_clifford_only() {
        assert!(make_clifford_circuit().is_clifford_only());
        assert!(!make_general_circuit().is_clifford_only());
        assert!(!make_product_circuit().is_clifford_only());
    }

    #[test]
    fn test_circuit_has_entangling_gates() {
        assert!(make_clifford_circuit().has_entangling_gates());
        assert!(make_general_circuit().has_entangling_gates());
        assert!(!make_product_circuit().has_entangling_gates());
    }

    #[test]
    fn test_auto_selects_product() {
        let circuit = make_product_circuit();
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "productstate");
    }

    #[test]
    fn test_auto_selects_stabilizer() {
        let circuit = make_clifford_circuit();
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "stabilizer");
    }

    #[test]
    fn test_auto_selects_statevector() {
        let circuit = make_general_circuit();
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "statevector");
    }

    #[test]
    fn test_run_with_auto_matches_explicit() {
        let circuit = make_general_circuit();
        let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_run_with_explicit_backends() {
        let circuit = make_clifford_circuit();
        assert!(run_with(BackendKind::Statevector, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Stabilizer, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Sparse, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).is_ok());
    }

    #[test]
    fn test_run_auto_clifford_probs_match_statevector() {
        let circuit = make_clifford_circuit();
        let auto_result = run(&circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &circuit).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_run_qasm() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];";
        let result = run_qasm(qasm, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_circuit_is_clifford_and_no_entangling() {
        let c = Circuit::new(2, 0);
        assert!(c.is_clifford_only());
        assert!(!c.has_entangling_gates());
    }

    #[test]
    fn test_validate_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit(); // has T gate
        let result = run_with(BackendKind::Stabilizer, &circuit, 42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_product_rejects_entangling() {
        let circuit = make_clifford_circuit(); // has CX
        let result = run_with(BackendKind::ProductState, &circuit, 42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_passes_for_compatible() {
        let clifford = make_clifford_circuit();
        assert!(run_with(BackendKind::Stabilizer, &clifford, 42).is_ok());

        let product = make_product_circuit();
        assert!(run_with(BackendKind::ProductState, &product, 42).is_ok());
    }

    #[test]
    fn test_auto_large_qubit_count() {
        // Use a qubit count guaranteed to exceed any reasonable memory threshold
        let mut circuit = Circuit::new(34, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]); // non-Clifford → not sparse-friendly
        circuit.add_gate(Gate::Cx, &[0, 1]); // entangling
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "mps");
    }

    #[test]
    fn test_auto_sparse_friendly_large() {
        // Sparse-friendly (no superposition gates) + above statevector threshold
        let mut circuit = Circuit::new(34, 0);
        circuit.add_gate(Gate::X, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]); // entangling, but sparse-preserving
        circuit.add_gate(Gate::T, &[2]); // diagonal, sparse-preserving
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "sparse");
    }

    #[test]
    fn test_auto_moderate_qubit_count_uses_statevector() {
        let mut circuit = Circuit::new(20, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, false);
        assert_eq!(backend.name(), "statevector");
    }

    #[test]
    fn test_auto_selects_factored_with_partial_independence() {
        let mut circuit = Circuit::new(10, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, true);
        assert_eq!(backend.name(), "factored");
    }

    #[test]
    fn test_auto_ignores_partial_independence_when_no_entangling() {
        let circuit = make_product_circuit();
        let backend = select_backend(&BackendKind::Auto, &circuit, 42, true);
        assert_eq!(backend.name(), "productstate");
    }

    #[test]
    fn test_classical_only_skips_probabilities() {
        let qasm =
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];";
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with_opts(
            BackendKind::Statevector,
            &circuit,
            42,
            SimOptions::classical_only(),
        )
        .unwrap();
        assert!(result.probabilities.is_none());
        assert_eq!(result.classical_bits.len(), 1);
    }

    #[test]
    fn test_default_options_include_probabilities() {
        let circuit = make_general_circuit();
        let result = run_with_opts(
            BackendKind::Statevector,
            &circuit,
            42,
            SimOptions::default(),
        )
        .unwrap();
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_run_opts_classical_only() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        let result = run_on_opts(&mut backend, &circuit, &SimOptions::classical_only()).unwrap();
        assert!(result.probabilities.is_none());
    }

    #[test]
    fn test_run_opts_with_probabilities() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        let result = run_on_opts(&mut backend, &circuit, &SimOptions::default()).unwrap();
        assert!(result.probabilities.is_some());
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_clifford_matches_statevector() {
        // Circuit with a long Clifford prefix followed by non-Clifford gates.
        // At 10q with 20+ prefix gates, temporal decomposition should trigger.
        let mut c = Circuit::new(10, 0);

        // Clifford prefix: 22 gates (above min_clifford_prefix_gates(10)=20)
        for i in 0..10 {
            c.add_gate(Gate::H, &[i]);
        }
        for i in 0..9 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Sdg, &[3]);
        c.add_gate(Gate::SX, &[7]);

        // Non-Clifford tail
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Rx(0.7), &[1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Rz(1.2), &[2]);

        let (prefix, _tail) = c.clifford_prefix_split().unwrap();
        assert!(prefix.gate_count() >= super::min_clifford_prefix_gates(c.num_qubits));

        let auto_result = run(&c, 42).unwrap();

        // Pure statevector reference (no temporal decomposition)
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        assert_eq!(auto_probs.len(), sv_probs.len());
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!(
                (a - s).abs() < 1e-10,
                "temporal decomp mismatch: auto={a}, sv={s}"
            );
        }
    }

    #[test]
    fn test_temporal_clifford_complex_circuit_matches_sv() {
        // 3q circuit: prefix too short for temporal (min_clifford_prefix_gates(3)=16)
        // but auto must still match statevector.
        let mut c = Circuit::new(3, 0);

        // Clifford prefix with i-phase generators
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Y, &[1]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::SXdg, &[2]);
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::Swap, &[0, 2]);
        c.add_gate(Gate::S, &[1]);

        // Non-Clifford tail
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Ry(0.3), &[1]);
        c.add_gate(Gate::Cx, &[1, 2]);

        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!(
                (a - s).abs() < 1e-10,
                "complex temporal mismatch: auto={a}, sv={s}"
            );
        }
    }

    #[test]
    fn test_temporal_clifford_skipped_when_prefix_too_short() {
        // Short Clifford prefix — should NOT use temporal decomposition,
        // but result must still be correct.
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]); // 2 Clifford gates < min_clifford_prefix_gates(2)=16

        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decomposed_random_blocks_matches_monolithic() {
        let circuit = crate::circuits::independent_random_blocks(10, 2, 5, 0xDEAD_BEEF);
        let decomposed = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let monolithic = run_on(&mut sv, &circuit).unwrap();
        let d_probs = decomposed.probabilities.unwrap().to_vec();
        let m_probs = monolithic.probabilities.unwrap().to_vec();
        assert_eq!(d_probs.len(), m_probs.len());
        for (d, m) in d_probs.iter().zip(m_probs.iter()) {
            assert!(
                (d - m).abs() < 1e-10,
                "mismatch: decomposed={d}, monolithic={m}"
            );
        }
    }

    #[test]
    fn test_per_block_clifford_dispatch() {
        // 6-qubit circuit: qubits 0-2 = Clifford block, qubits 3-5 = non-Clifford block.
        // The two blocks are independent (no gates bridge them).
        // Under Auto dispatch with decomposition, block 0-2 should use Stabilizer
        // and block 3-5 should use Statevector. We verify correctness by comparing
        // against a monolithic statevector run.
        let mut c = Circuit::new(6, 0);

        // Block A (Clifford): GHZ state on qubits 0,1,2
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::S, &[0]);

        // Block B (non-Clifford): qubits 3,4,5
        c.add_gate(Gate::H, &[3]);
        c.add_gate(Gate::T, &[3]);
        c.add_gate(Gate::Cx, &[3, 4]);
        c.add_gate(Gate::Rx(0.7), &[5]);
        c.add_gate(Gate::Cx, &[4, 5]);

        let components = c.independent_subsystems();
        assert_eq!(components.len(), 2);

        let (sub_a, _, _) = c.extract_subcircuit(&components[0]);
        assert!(sub_a.is_clifford_only());
        let backend_a = select_backend(&BackendKind::Auto, &sub_a, 42, false);
        assert_eq!(backend_a.name(), "stabilizer");

        let (sub_b, _, _) = c.extract_subcircuit(&components[1]);
        assert!(!sub_b.is_clifford_only());
        let backend_b = select_backend(&BackendKind::Auto, &sub_b, 43, false);
        assert_eq!(backend_b.name(), "statevector");

        // End-to-end: auto (decomposed) must match monolithic statevector
        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let mono_result = run_on(&mut sv, &c).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let mono_probs = mono_result.probabilities.unwrap().to_vec();
        assert_eq!(auto_probs.len(), mono_probs.len());
        for (a, m) in auto_probs.iter().zip(mono_probs.iter()) {
            assert!((a - m).abs() < 1e-10, "prob mismatch: auto={a}, mono={m}");
        }
    }

    #[test]
    fn test_decomposed_bell_pairs_matches_monolithic() {
        let circuit = crate::circuits::independent_bell_pairs(10);
        let decomposed = run(&circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let monolithic = run_on(&mut sv, &circuit).unwrap();
        let d_probs = decomposed.probabilities.unwrap().to_vec();
        let m_probs = monolithic.probabilities.unwrap().to_vec();
        assert_eq!(d_probs.len(), m_probs.len());
        for (d, m) in d_probs.iter().zip(m_probs.iter()) {
            assert!(
                (d - m).abs() < 1e-10,
                "mismatch: decomposed={d}, monolithic={m}"
            );
        }
    }

    #[test]
    fn test_measurement_normalization_statevector() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "statevector post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_measurement_normalization_mps() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "MPS post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_measurement_normalization_sparse() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Sparse, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "sparse post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_conditional_gate_execution() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[1] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        // q[0] measured as 1, so if-gate fires, q[1] flipped to |1⟩
        // Final state: |11⟩ = index 3
        let probs = result.probabilities.unwrap().to_vec();
        assert!(
            probs[3] > 0.99,
            "conditional gate should flip q[1]: probs={probs:?}"
        );
        assert!(result.classical_bits[0]);
    }

    fn make_bell_with_measure() -> Circuit {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        crate::circuit::openqasm::parse(qasm).unwrap()
    }

    #[test]
    fn test_shots_deterministic() {
        let circuit = make_bell_with_measure();
        let a = run_shots(&circuit, 10, 42).unwrap();
        let b = run_shots(&circuit, 10, 42).unwrap();
        assert_eq!(a.shots, b.shots);
    }

    #[test]
    fn test_shots_distribution_convergence() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 10000, 42).unwrap();
        let counts = result.counts();
        let n_00 = counts.get(&vec![false, false]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![true, true]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![false, true]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![true, false]).copied().unwrap_or(0);
        assert!(
            (4500..=5500).contains(&n_00),
            "|00> count {n_00} outside [4500, 5500]"
        );
        assert!(
            (4500..=5500).contains(&n_11),
            "|11> count {n_11} outside [4500, 5500]"
        );
        assert_eq!(n_01, 0, "|01> should never appear in Bell state");
        assert_eq!(n_10, 0, "|10> should never appear in Bell state");
    }

    #[test]
    fn test_shots_single_valid_outcome() {
        let circuit = make_bell_with_measure();
        let shots_result = run_shots(&circuit, 1, 42).unwrap();
        let shot = &shots_result.shots[0];
        assert_eq!(shot[0], shot[1], "Bell state: both bits must agree");
    }

    #[test]
    fn test_shots_all_zero() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[3] c;
            c[0] = measure q[0];
            c[1] = measure q[1];
            c[2] = measure q[2];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(
                shot.iter().all(|&b| !b),
                "shot {i} should be all-zero: {shot:?}"
            );
        }
    }

    #[test]
    fn test_shots_mid_circuit_measurement() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(shot[0], "shot {i}: q[0] should always be 1");
            assert!(shot[1], "shot {i}: q[1] should always be 1 (conditional)");
        }
    }

    #[test]
    fn test_shots_has_probabilities() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::Rx(std::f64::consts::FRAC_PI_4), &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let result = run_shots(&circuit, 5, 42).unwrap();
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_shots_counts_sum() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 500, 42).unwrap();
        let counts = result.counts();
        let total: usize = counts.values().sum();
        assert_eq!(total, 500);
    }

    fn assert_unit_norm(state: &[num_complex::Complex64], label: &str) {
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "{label}: norm = {norm}, expected 1.0"
        );
    }

    #[test]
    fn test_export_norm_statevector_bell() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/bell");
    }

    #[test]
    fn test_export_norm_statevector_parametric() {
        let circuit = crate::circuits::hardware_efficient_ansatz(6, 3, 42);
        let mut backend = StatevectorBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/hea_6q");
    }

    #[test]
    fn test_export_norm_stabilizer() {
        let circuit = make_clifford_circuit();
        let mut backend = StabilizerBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "stabilizer");
    }

    #[test]
    fn test_export_norm_sparse() {
        let circuit = make_general_circuit();
        let mut backend = SparseBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "sparse");
    }

    #[test]
    fn test_export_norm_mps() {
        let circuit = make_general_circuit();
        let mut backend = MpsBackend::new(64, 42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "mps");
    }

    #[test]
    fn test_export_norm_product_state() {
        let circuit = make_product_circuit();
        let mut backend = ProductStateBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "productstate");
    }

    #[test]
    fn test_export_norm_tensor_network() {
        let circuit = make_general_circuit();
        let mut backend = TensorNetworkBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "tensornetwork");
    }

    #[test]
    fn test_export_norm_after_measurement() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[1] c;
            h q[0];
            cx q[0], q[1];
            h q[2];
            c[0] = measure q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        for backend_kind in [
            BackendKind::Statevector,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
        ] {
            let label = format!("{backend_kind:?}/post-measure");
            let mut backend = select_backend(&backend_kind, &circuit, 42, false);
            run_on(backend.as_mut(), &circuit).unwrap();
            let state = backend.export_statevector().unwrap();
            assert_unit_norm(&state, &label);
        }
    }

    #[test]
    fn test_export_norm_qft() {
        let circuit = crate::circuits::qft_circuit(8);
        for (kind, label) in [
            (BackendKind::Statevector, "statevector/qft8"),
            (BackendKind::Sparse, "sparse/qft8"),
            (BackendKind::Mps { max_bond_dim: 128 }, "mps/qft8"),
            (BackendKind::TensorNetwork, "tn/qft8"),
        ] {
            let mut backend = select_backend(&kind, &circuit, 42, false);
            run_on(backend.as_mut(), &circuit).unwrap();
            let state = backend.export_statevector().unwrap();
            assert_unit_norm(&state, label);
        }
    }

    #[test]
    fn test_export_factored_unsupported() {
        let circuit = make_general_circuit();
        let mut backend = crate::backend::factored::FactoredBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert!(backend.export_statevector().is_err());
    }

    #[test]
    fn test_shots_random_convergence() {
        let circuit = make_bell_with_measure();
        let result = run_shots_random(&circuit, 10000).unwrap();
        let counts = result.counts();
        let n_00 = counts.get(&vec![false, false]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![true, true]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![false, true]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![true, false]).copied().unwrap_or(0);
        // p=0.5, n=10000 → σ=50. Bounds at ±10σ: failure prob < 10^-23.
        assert!(
            (4500..=5500).contains(&n_00),
            "|00> count {n_00} outside [4500, 5500]"
        );
        assert!(
            (4500..=5500).contains(&n_11),
            "|11> count {n_11} outside [4500, 5500]"
        );
        assert_eq!(n_01, 0, "|01> should never appear in Bell state");
        assert_eq!(n_10, 0, "|10> should never appear in Bell state");
    }

    #[test]
    fn test_has_terminal_measurements_only() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        assert!(c.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[1] c;
            c[0] = measure q[0];
            h q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[1] c;
            h q[0];
            c[0] = measure q[0];
            x q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());
    }

    #[test]
    fn test_measurement_map() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[3] c;
            c[2] = measure q[0];
            c[0] = measure q[2];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let map = circuit.measurement_map();
        assert_eq!(map, vec![(0, 2), (2, 0), (1, 1)]);
    }

    #[test]
    fn test_fast_path_deterministic_x() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[1] c;
            x q[0];
            c[0] = measure q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(circuit.has_terminal_measurements_only());
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(shot[0], "shot {i}: X|0> should always measure 1");
        }
    }

    #[test]
    fn test_fast_path_no_measurements() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        let result = run_shots(&c, 50, 42).unwrap();
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
            assert!(!shot[0] && !shot[1], "no measurements → all-false");
        }
    }

    #[test]
    fn test_shots_cached_fusion_matches_uncached() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let cached = run_shots_with(BackendKind::Statevector, &circuit, 20, 42).unwrap();
        for i in 0..20 {
            let seed_i = 42u64.wrapping_add(i as u64);
            let single = run_with_opts(
                BackendKind::Statevector,
                &circuit,
                seed_i,
                SimOptions::default(),
            )
            .unwrap();
            assert_eq!(cached.shots[i], single.classical_bits, "shot {i} mismatch");
        }
    }

    #[test]
    fn test_shots_decomposed_cached() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[8] q;
            bit[8] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            x q[1];
            c[1] = measure q[1];
            h q[4];
            cx q[4], q[5];
            c[4] = measure q[4];
            x q[5];
            c[5] = measure q[5];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());
        let comps = circuit.independent_subsystems();
        assert!(comps.len() > 1, "circuit should decompose");

        let result = run_shots_with(BackendKind::Statevector, &circuit, 10, 42).unwrap();
        assert_eq!(result.shots.len(), 10);
        for shot in &result.shots {
            assert_eq!(shot.len(), 8);
        }
    }

    #[test]
    fn test_shots_temporal_clifford_fallback() {
        let mut c = Circuit::new(4, 4);
        for i in 0..4 {
            c.add_gate(Gate::H, &[i]);
        }
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        c.add_gate(Gate::T, &[0]);
        c.add_measure(0, 0);
        c.add_gate(Gate::X, &[1]);
        c.add_measure(1, 1);

        let result = run_shots_with(BackendKind::Auto, &c, 10, 42).unwrap();
        assert_eq!(result.shots.len(), 10);
        for shot in &result.shots {
            assert_eq!(shot.len(), 4);
        }
    }

    #[test]
    fn test_stabilizer_rank_dispatch() {
        let circuit = make_general_circuit();
        let result = run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert_eq!(probs.len(), 8);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (sr, sv)) in probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr - sv).abs() < 1e-10,
                "prob[{i}]: stab_rank={sr}, statevector={sv}"
            );
        }
    }

    #[test]
    fn test_stabilizer_rank_rejects_no_t() {
        let circuit = make_clifford_circuit();
        let result = run_with(BackendKind::StabilizerRank, &circuit, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_clifford_plus_t_probabilities() {
        let circuit = make_general_circuit();
        assert!(circuit.is_clifford_plus_t());
        assert!(circuit.has_t_gates());

        let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-10,
                "prob[{i}]: auto={a}, statevector={s}"
            );
        }
    }

    #[test]
    fn test_auto_clifford_plus_t_shots() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        let result = run_shots_with(BackendKind::Auto, &c, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
        }
    }

    #[test]
    fn test_decomposed_mixed_clifford_and_t() {
        // Two independent subsystems: q0-q1 (Clifford+T), q2-q3 (Clifford-only)
        // Under decomposition, q2-q3 should route to Stabilizer, q0-q1 to StabilizerRank
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[2, 3]);

        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 2);

        let auto_result = run_with(BackendKind::Auto, &c, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &c, 42).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-10,
                "prob[{i}]: auto={a}, statevector={s}"
            );
        }
    }

    #[test]
    fn test_run_shots_with_noise_clifford_uses_compiled() {
        let n = 10;
        let mut circuit = crate::circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        assert!(result.shots[0].len() == n);
    }

    #[test]
    fn test_run_shots_with_noise_statevector_brute() {
        let mut circuit = Circuit::new(3, 3);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result =
            run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 50, 42).unwrap();
        assert_eq!(result.shots.len(), 50);
        assert_eq!(result.shots[0].len(), 3);
    }

    #[test]
    fn test_run_shots_with_noise_auto_non_clifford() {
        let mut circuit = Circuit::new(3, 3);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
    }
}
