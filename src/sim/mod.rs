//! Simulation orchestration.
//!
//! Connects the circuit IR to a backend. This module is deliberately thin —
//! the complexity lives in the backends and the parser.

pub mod compiled;
mod decomposed;
mod dispatch;
pub mod homological;
pub mod noise;
pub mod stabilizer_rank;
mod trajectory;
pub mod unified_pauli;

pub(crate) use decomposed::merge_probabilities;
use decomposed::{
    run_decomposed, run_decomposed_prefused, should_decompose, MIN_DECOMPOSITION_QUBITS,
};
pub use dispatch::BackendKind;
use dispatch::{
    has_temporal_clifford_opportunity, select_backend, select_dispatch, supports_fused_for_kind,
    try_temporal_clifford, validate_explicit_backend, DispatchAction, AUTO_APPROX_MAX_TERMS,
    AUTO_SPD_MAX_TERMS, MAX_AUTO_T_COUNT_APPROX, MAX_AUTO_T_COUNT_EXACT, MAX_AUTO_T_COUNT_SHOTS,
    MAX_STABILIZER_RANK_QUBITS, MIN_BLOCK_FOR_FACTORED_STAB, MIN_FACTORED_STABILIZER_QUBITS,
    MIN_QUBITS_FOR_SPD_AUTO,
};

use std::collections::HashMap;

use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;

#[derive(Debug, Clone, Copy)]
pub(crate) struct SimOptions {
    pub(crate) probabilities: bool,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            probabilities: true,
        }
    }
}

impl SimOptions {
    pub(crate) fn classical_only() -> Self {
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
            // SAFETY: BMI2 availability checked by is_x86_feature_detected above
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
    /// `None` if the backend does not support probability extraction.
    pub probabilities: Option<Probabilities>,
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
    run_with_internal(kind, circuit, seed, SimOptions::default())
}

fn run_with_internal(
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
                let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
                if matches!(kind, BackendKind::Auto)
                    && circuit.is_clifford_only()
                    && circuit.num_qubits >= MIN_FACTORED_STABILIZER_QUBITS
                    && max_block >= MIN_BLOCK_FOR_FACTORED_STAB
                {
                    let mut backend =
                        crate::backend::factored_stabilizer::FactoredStabilizerBackend::new(seed);
                    let fs_opts = if circuit.num_qubits > 64 {
                        SimOptions {
                            probabilities: false,
                        }
                    } else {
                        opts
                    };
                    return execute(&mut backend, circuit, &fs_opts);
                }
                return run_decomposed(&kind, &components, circuit, seed, &opts);
            }
            has_partial_independence = true;
        }
    }
    if matches!(kind, BackendKind::Auto)
        && circuit.is_clifford_plus_t()
        && circuit.has_t_gates()
        && circuit.num_qubits <= MAX_STABILIZER_RANK_QUBITS
    {
        let t = circuit.t_count();
        let n = circuit.num_qubits;
        let log2n = if n >= 2 {
            (n as f64).log2().ceil() as usize * 2
        } else {
            0
        };
        let sr_budget = n.saturating_sub(log2n);
        if t <= MAX_AUTO_T_COUNT_EXACT && t <= sr_budget {
            let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
            return Ok(SimulationResult {
                probabilities: Some(Probabilities::Dense(sr.probabilities)),
                classical_bits: vec![],
            });
        }
        if t <= MAX_AUTO_T_COUNT_APPROX && t <= sr_budget {
            let sr =
                stabilizer_rank::run_stabilizer_rank_approx(circuit, AUTO_APPROX_MAX_TERMS, seed)?;
            return Ok(SimulationResult {
                probabilities: Some(Probabilities::Dense(sr.probabilities)),
                classical_bits: vec![],
            });
        }
    }
    if let Some(result) = try_temporal_clifford(&kind, circuit, seed) {
        return result;
    }
    match select_dispatch(&kind, circuit, seed, has_partial_independence) {
        DispatchAction::Backend(mut backend) => execute(&mut *backend, circuit, &opts),
        DispatchAction::StabilizerRank => {
            let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
            Ok(SimulationResult {
                probabilities: Some(Probabilities::Dense(sr.probabilities)),
                classical_bits: vec![],
            })
        }
        DispatchAction::StochasticPauli { num_samples } => {
            let spp = unified_pauli::run_spp(circuit, num_samples, seed)?;
            let probs = unified_pauli::spp_to_probabilities(&spp);
            Ok(SimulationResult {
                probabilities: Some(Probabilities::Dense(probs)),
                classical_bits: vec![],
            })
        }
        DispatchAction::DeterministicPauli { epsilon, max_terms } => {
            let spd = unified_pauli::run_spd(circuit, epsilon, max_terms)?;
            let probs = unified_pauli::spd_to_probabilities(&spd);
            Ok(SimulationResult {
                probabilities: Some(Probabilities::Dense(probs)),
                classical_bits: vec![],
            })
        }
    }
}

/// Execute a circuit on a pre-constructed backend.
///
/// Use this when you need direct control over the backend instance
/// (e.g., testing a specific backend). For automatic dispatch, use [`run`].
pub fn run_on(backend: &mut dyn Backend, circuit: &Circuit) -> Result<SimulationResult> {
    execute(backend, circuit, &SimOptions::default())
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
    num_classical_bits: usize,
}

impl ShotsResult {
    /// Build a frequency histogram of measurement outcomes.
    ///
    /// Keys are packed `Vec<u64>` where bit `i` of word `i/64` corresponds
    /// to classical bit `i`. Use [`bitstring`] to format keys for display.
    pub fn counts(&self) -> HashMap<Vec<u64>, u64> {
        let m_words = self.num_classical_bits.div_ceil(64).max(1);
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        for shot in &self.shots {
            let mut key = vec![0u64; m_words];
            for (i, &b) in shot.iter().enumerate() {
                if b {
                    key[i / 64] |= 1u64 << (i % 64);
                }
            }
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    pub fn num_shots(&self) -> usize {
        self.shots.len()
    }

    pub fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }
}

impl std::fmt::Display for ShotsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = self.counts();
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        for (bits, count) in &entries {
            let bs = bitstring(bits, self.num_classical_bits);
            writeln!(f, "{bs}: {count}")?;
        }
        Ok(())
    }
}

/// Format a packed `Vec<u64>` key (from [`ShotsResult::counts`]) as a binary string.
///
/// Bit 0 of the first word corresponds to classical bit 0 (leftmost character).
pub fn bitstring(key: &[u64], num_bits: usize) -> String {
    let mut s = String::with_capacity(num_bits);
    for i in 0..num_bits {
        let word = i / 64;
        let bit = i % 64;
        if word < key.len() && (key[word] >> bit) & 1 == 1 {
            s.push('1');
        } else {
            s.push('0');
        }
    }
    s
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

    let mut indices = Vec::with_capacity(num_shots);

    match probs {
        Probabilities::Dense(v) => {
            let cdf = build_cdf(v);
            for _ in 0..num_shots {
                let r: f64 = rng.random();
                indices.push(sample_from_cdf(&cdf, r));
            }
        }
        Probabilities::Factored { blocks, .. } => {
            let block_cdfs: Vec<Vec<f64>> = blocks.iter().map(|b| build_cdf(&b.probs)).collect();
            for _ in 0..num_shots {
                let mut global_idx = 0usize;
                for (block, cdf) in blocks.iter().zip(block_cdfs.iter()) {
                    let r: f64 = rng.random();
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
                indices.push(global_idx);
            }
        }
    }

    let mut flat = vec![false; num_shots * num_classical_bits];
    for (s, &state_idx) in indices.iter().enumerate() {
        let base = s * num_classical_bits;
        for &(qubit, cbit) in meas_map {
            flat[base + cbit] = (state_idx >> qubit) & 1 == 1;
        }
    }

    let mut shots = Vec::with_capacity(num_shots);
    for chunk in flat.chunks_exact(num_classical_bits) {
        shots.push(chunk.to_vec());
    }
    shots
}

/// Execute a circuit multiple times, collecting measurement outcomes.
pub fn run_shots(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    run_shots_with(BackendKind::Auto, circuit, num_shots, seed)
}

/// Execute a circuit multiple times and return outcome counts directly.
///
/// For Clifford circuits with Auto/Stabilizer/FilteredStabilizer backends,
/// routes through the compiled sampler's optimized counting path (rank-space
/// counting for low-rank circuits, histogram accumulation for high-rank).
/// For other backends, falls back to per-shot simulation with counting.
pub fn run_counts(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<HashMap<Vec<u64>, u64>> {
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
        let mut sampler = compiled::compile_measurements(circuit, seed)?;
        return Ok(sampler.sample_counts(num_shots));
    }

    let result = run_shots_with(kind, circuit, num_shots, seed)?;
    let m_words = circuit.num_classical_bits.div_ceil(64).max(1);
    let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
    for shot in &result.shots {
        let mut key = vec![0u64; m_words];
        for (i, &b) in shot.iter().enumerate() {
            if b {
                key[i / 64] |= 1u64 << (i % 64);
            }
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    Ok(counts)
}

/// Compute per-qubit marginal probabilities: P(q_i = 0) and P(q_i = 1) for each qubit.
///
/// Returns a `Vec<(f64, f64)>` of length `num_qubits`, where each element is `(p0, p1)`.
///
/// For Clifford+T circuits, uses Sparse Pauli Dynamics (SPD) — Heisenberg-picture
/// backward propagation that scales with Pauli complexity, not 2^n. This is orders
/// of magnitude faster than statevector for structured Clifford+T circuits (25-400x
/// at 14-22 qubits).
///
/// For pure Clifford circuits, compiles and extracts marginals from the parity matrix.
///
/// For other circuits, falls back to statevector probabilities and extracts marginals.
pub fn run_marginals(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<Vec<(f64, f64)>> {
    let n = circuit.num_qubits;

    if (matches!(
        kind,
        BackendKind::Auto | BackendKind::DeterministicPauli { .. }
    )) && circuit.is_clifford_plus_t()
        && circuit.has_t_gates()
        && n >= MIN_QUBITS_FOR_SPD_AUTO
    {
        let spd = unified_pauli::run_spd(circuit, 0.0, AUTO_SPD_MAX_TERMS)?;
        return Ok(spd
            .expectations
            .iter()
            .map(|ez| {
                let p0 = ((1.0 + ez) / 2.0).clamp(0.0, 1.0);
                (p0, 1.0 - p0)
            })
            .collect());
    }

    let result = run_with(kind, circuit, seed)?;
    if let Some(probs) = &result.probabilities {
        let mut marginals = vec![(0.0f64, 0.0f64); n];
        let dense = probs.to_vec();
        for (idx, &p) in dense.iter().enumerate() {
            for (q, m) in marginals.iter_mut().enumerate() {
                if (idx >> q) & 1 == 0 {
                    m.0 += p;
                } else {
                    m.1 += p;
                }
            }
        }
        Ok(marginals)
    } else {
        Ok(vec![(0.5, 0.5); n])
    }
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
        let result = run_with_internal(kind.clone(), &stripped, seed, SimOptions::default())?;
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
                num_classical_bits: circuit.num_classical_bits,
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
    if matches!(
        kind,
        BackendKind::StochasticPauli { .. } | BackendKind::DeterministicPauli { .. }
    ) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli propagation backends do not support mid-circuit measurements".into(),
        });
    }
    if matches!(kind, BackendKind::Auto) && circuit.is_clifford_plus_t() && circuit.has_t_gates() {
        let t = circuit.t_count();
        let n = circuit.num_qubits;
        let log2n = if n >= 2 {
            (n as f64).log2().ceil() as usize * 2
        } else {
            0
        };
        let sr_budget = n.saturating_sub(log2n);
        if t <= MAX_AUTO_T_COUNT_SHOTS && t <= sr_budget {
            return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
        }
    }

    if has_temporal_clifford_opportunity(&kind, circuit) {
        return run_shots_fallback(&kind, circuit, num_shots, seed);
    }

    let supports_fused = supports_fused_for_kind(&kind, circuit);
    let mut shots = Vec::with_capacity(num_shots);
    let opts = SimOptions::classical_only();

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
        }
    } else {
        let fused = crate::circuit::fusion::fuse_circuit(circuit, supports_fused);

        for i in 0..num_shots {
            let shot_seed = seed.wrapping_add(i as u64);
            let mut backend = select_backend(&kind, circuit, shot_seed, has_partial_independence);
            let result = execute_circuit(&mut *backend, &fused, &opts)?;
            shots.push(result.classical_bits);
        }
    }

    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
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
    if matches!(
        kind,
        BackendKind::StabilizerRank
            | BackendKind::StochasticPauli { .. }
            | BackendKind::DeterministicPauli { .. }
    ) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "this backend does not support noisy per-shot simulation".into(),
        });
    }

    let is_stabilizer_kind = matches!(
        kind,
        BackendKind::Stabilizer | BackendKind::FilteredStabilizer | BackendKind::FactoredStabilizer
    );

    if is_stabilizer_kind && !noise_model.is_pauli_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "stabilizer backends only support Pauli/depolarizing noise; \
                     use Statevector or MPS for amplitude damping, phase damping, \
                     thermal relaxation, custom Kraus, or readout errors"
                .into(),
        });
    }

    if is_stabilizer_kind && !circuit.is_clifford_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "circuit contains non-Clifford gates".into(),
        });
    }

    if noise_model.is_pauli_only() {
        let use_compiled = matches!(
            kind,
            BackendKind::Auto | BackendKind::Stabilizer | BackendKind::FilteredStabilizer
        ) && circuit.is_clifford_only();

        if use_compiled {
            return noise::run_shots_noisy(circuit, noise_model, num_shots, seed);
        }
    }

    trajectory::run_trajectories(
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
    let opts = SimOptions::classical_only();
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let result = run_with_internal(kind.clone(), circuit, shot_seed, opts)?;
        shots.push(result.classical_bits);
    }
    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::dispatch::min_clifford_prefix_gates;
    use super::*;
    use crate::backend::mps::MpsBackend;
    use crate::backend::product::ProductStateBackend;
    use crate::backend::sparse::SparseBackend;
    use crate::backend::stabilizer::StabilizerBackend;
    use crate::backend::statevector::StatevectorBackend;
    use crate::backend::tensornetwork::TensorNetworkBackend;
    use crate::circuit::smallvec;
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
        let result = run_with_internal(
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
        let result = run_with_internal(
            BackendKind::Statevector,
            &circuit,
            42,
            SimOptions::default(),
        )
        .unwrap();
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_run_on_always_computes_probabilities() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        let result = run_on(&mut backend, &circuit).unwrap();
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
        assert!(prefix.gate_count() >= min_clifford_prefix_gates(c.num_qubits));

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
        let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
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
    fn test_shots_counts_sum() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 500, 42).unwrap();
        let counts = result.counts();
        let total: u64 = counts.values().sum();
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
        let result = run_shots(&circuit, 10000, rand::random()).unwrap();
        let counts = result.counts();
        let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
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
            let single = run_with_internal(
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

    #[test]
    fn test_run_marginals_bell_pair() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let m = run_marginals(BackendKind::Auto, &c, 42).unwrap();
        assert_eq!(m.len(), 2);
        assert!((m[0].0 - 0.5).abs() < 1e-10);
        assert!((m[0].1 - 0.5).abs() < 1e-10);
        assert!((m[1].0 - 0.5).abs() < 1e-10);
        assert!((m[1].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_run_marginals_x_gate() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[0]);
        let m = run_marginals(BackendKind::Auto, &c, 42).unwrap();
        assert!((m[0].0 - 0.0).abs() < 1e-10);
        assert!((m[0].1 - 1.0).abs() < 1e-10);
        assert!((m[1].0 - 1.0).abs() < 1e-10);
        assert!((m[1].1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_run_marginals_clifford_t_spd_path() {
        let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
        let m_spd = run_marginals(BackendKind::Auto, &c, 42).unwrap();
        assert_eq!(m_spd.len(), 14);
        for (p0, p1) in &m_spd {
            assert!(*p0 >= 0.0 && *p0 <= 1.0);
            assert!((p0 + p1 - 1.0).abs() < 1e-10);
        }

        let m_sv = run_marginals(BackendKind::Statevector, &c, 42).unwrap();
        for i in 0..14 {
            assert!(
                (m_spd[i].0 - m_sv[i].0).abs() < 1e-6,
                "qubit {i}: SPD p0={} vs SV p0={}",
                m_spd[i].0,
                m_sv[i].0
            );
        }
    }

    // ── Dispatch validation ───────────────────────────────────────────

    #[test]
    fn test_validate_filtered_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit();
        assert!(matches!(
            run_with(BackendKind::FilteredStabilizer, &circuit, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_factored_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit();
        assert!(matches!(
            run_with(BackendKind::FactoredStabilizer, &circuit, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_stabilizer_rank_rejects_no_t_gates() {
        let circuit = make_clifford_circuit();
        assert!(matches!(
            run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_filtered_stabilizer_accepts_clifford() {
        assert!(run_with(
            BackendKind::FilteredStabilizer,
            &make_clifford_circuit(),
            42
        )
        .is_ok());
    }

    #[test]
    fn test_validate_factored_stabilizer_accepts_clifford() {
        assert!(run_with(
            BackendKind::FactoredStabilizer,
            &make_clifford_circuit(),
            42
        )
        .is_ok());
    }

    // ── Pauli backend error paths ───────────────────────────────────────

    #[test]
    fn test_pauli_backends_reject_mid_circuit_measurements() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            c[0] = measure q[0];
            cx q[0], q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();

        assert!(matches!(
            run_shots_with(
                BackendKind::StochasticPauli { num_samples: 100 },
                &circuit,
                10,
                42
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            run_shots_with(
                BackendKind::DeterministicPauli {
                    epsilon: 1e-3,
                    max_terms: 1000
                },
                &circuit,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    // ── Noisy simulation error paths ────────────────────────────────────

    #[test]
    fn test_noise_rejects_stabilizer_rank() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(matches!(
            run_shots_with_noise(BackendKind::StabilizerRank, &circuit, &nm, 10, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_rejects_pauli_backends() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);

        assert!(matches!(
            run_shots_with_noise(
                BackendKind::StochasticPauli { num_samples: 100 },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            run_shots_with_noise(
                BackendKind::DeterministicPauli {
                    epsilon: 1e-3,
                    max_terms: 1000
                },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_stabilizer_rejects_non_pauli_noise() {
        let circuit = make_clifford_circuit();
        let nm = noise::NoiseModel {
            after_gate: {
                let mut ag = vec![Vec::new(); circuit.instructions.len()];
                ag[0].push(noise::NoiseEvent {
                    channel: noise::NoiseChannel::AmplitudeDamping { gamma: 0.1 },
                    qubits: smallvec![0],
                });
                ag
            },
            readout: vec![None; circuit.num_qubits],
        };
        assert!(matches!(
            run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(matches!(
            run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    // ── Backend smoke tests ─────────────────────────────────────────────

    fn assert_probs_match(kind: BackendKind, circuit: &Circuit, expected: &[f64], tol: f64) {
        let label = format!("{kind:?}");
        let result = run_with(kind, circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert_eq!(probs.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, b)) in probs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "{label}: prob[{i}] = {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_smoke_all_backends_clifford() {
        let circuit = make_clifford_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();

        for kind in [
            BackendKind::Stabilizer,
            BackendKind::FilteredStabilizer,
            BackendKind::FactoredStabilizer,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
            BackendKind::TensorNetwork,
            BackendKind::Factored,
        ] {
            assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
        }
    }

    #[test]
    fn test_smoke_all_backends_general() {
        let circuit = make_general_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();

        for kind in [
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
            BackendKind::TensorNetwork,
            BackendKind::Factored,
        ] {
            assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
        }
    }

    #[test]
    fn test_smoke_product_state() {
        let circuit = make_product_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();
        assert_probs_match(BackendKind::ProductState, &circuit, &sv_probs, 1e-8);
    }

    #[test]
    fn test_smoke_stabilizer_rank() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::T, &[2]);

        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();
        assert_probs_match(BackendKind::StabilizerRank, &circuit, &sv_probs, 1e-6);
    }
}
