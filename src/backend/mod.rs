//! Simulation backend trait and implementations.
//!
//! Each backend owns its state representation and applies instructions to it.
//!
//! # Backend contract
//!
//! 1. Call [`Backend::init`] before any [`Backend::apply`] calls.
//! 2. Call [`Backend::apply`] for each instruction in circuit order.
//! 3. Measurement is destructive, it collapses the state.
//! 4. Given the same circuit and RNG seed, results must be deterministic.
//!
//! # Performance requirements for implementors
//!
//! - Gate application must avoid heap allocation in the hot path.
//! - Prefer direct indexing over iterator chains for state access.
//! - Use `#[inline]` and `#[inline(always)]` on gate kernels.
//! - Document all `unsafe` blocks with safety invariants.
//!
//! # Optional methods and what declines them
//!
//! Several trait methods carry a default that reports the operation unsupported.
//! Which representations decline, and why they cannot answer:
//!
//! | Method | Declined by | Reason |
//! | --- | --- | --- |
//! | `apply_1q_matrix` | Stabilizer, FactoredStabilizer | A tableau stores a state by its stabilizer group, closed under Clifford conjugation. A general 2x2 has no image in that group, and a Kraus branch is not even unitary. |
//! | `reduced_density_matrix_1q` | Stabilizer, FactoredStabilizer | Derivable from a tableau, but the operator it feeds cannot be applied (row above), so the branch would be sampled and never used. |
//! | `reduced_density_matrix_1q` | DistributedStatevector | Trajectories run shots on Rayon workers whose order differs per rank, so per-shot noise would issue rank collectives out of lockstep. `run_shots_with_noise` rejects the backend for that reason, which closes the only path here. |
//! | `reduced_density_matrix_2q` | TensorNetwork, DistributedStatevector | Feeds the branch weights of a correlated two-qubit Kraus channel, which needs the joint state of the pair. The default reads it out of `Backend::reduced_density_matrix`, so the decline follows that one. Answering is necessary and not sufficient: the branch operator is a general 2x2 block on the pair, so `Backend::supports_two_qubit_kraus` also requires a `Gate::Fused2q` kernel, which is what holds the two tableau backends and the product state out. `run_shots_with_noise` rejects on that query before allocating state. |
//! | `export_statevector` | DensityMatrix | A mixture of pure states has no statevector. Read `DensityMatrixBackend::purity` or reduce the state instead. |
//! | `export_statevector` | FactoredStabilizer | Exports while one tableau covers every qubit; past that there is no joint tableau to expand. |
//! | `init_from_density_matrix` | Everything except DensityMatrix | The input is a dense `4^n` mixture, and every other representation holds a pure state. The density matrix takes the buffer as its own, uploading it when the mixture is device resident. |
//! | `init_from_amplitudes` | Everything except Statevector, DistributedStatevector, and DensityMatrix | The input is a dense `2^n` amplitude vector, and a tableau, a product state, or a factored register holds only the states its structure can express. MPS could decode one by sequential SVD, but the bond cap would truncate the state the caller supplied. The distributed statevector takes the full vector on every rank and keeps its own slice. |
//! | `reduced_density_matrix` | TensorNetwork, DistributedStatevector | Each holds the state in a form a partial trace has to be contracted out of, a doubled network or a slice exchange across rank qubits, and neither kernel exists yet. The chain sweeps its own environment for it, at a cost set by the span the named qubits occupy rather than by how many there are. |
//! | `schmidt_values` | Everything except Statevector, Mps, ProductState, Stabilizer, and FactoredStabilizer | A mixture has no Schmidt decomposition. Sparse, factored, tensor-network and distributed states could answer through a reduced density matrix but do not yet, and `entanglement_entropy` follows wherever its default reads the spectrum. A stabilizer cut's spectrum is flat, so the two tableau backends build it from a rank and decline only past the dense export cap, where the `2^r` equal values no longer fit while the rank behind them still does. |
//! | `overlap_sq` | DensityMatrix | The fidelity of two mixtures is not an inner product, and the dense route the default takes needs a statevector a mixture has none of. |
//!
//! [`Backend::reduced_density_matrix_1q`] and [`Backend::apply_1q_matrix`] are
//! two halves of one capability, sampling a non-Pauli branch and applying the
//! operator it selects, so their coverage is one set by construction. That set is
//! `BackendKind::supports_general_noise`.

pub mod density_matrix;
#[cfg(feature = "distributed")]
pub mod distributed_statevector;
pub mod factored;
pub mod factored_stabilizer;
pub(crate) mod memory;
pub mod mps;
pub(crate) mod overlap;
pub mod product;
pub(crate) mod reduced_density;
pub(crate) mod schmidt;
pub(crate) mod simd;
pub mod sparse;
pub mod stabilizer;
pub mod statevector;
pub mod tensornetwork;
pub(crate) mod word_ops;

use num_complex::Complex64;

use crate::circuit::Instruction;
use crate::error::Result;
use crate::sim::unified_pauli::PauliTerm;
use crate::sim::{BondReport, Exactness, Placement, ResolvedBackend};

/// Qubit count at which dense amplitude kernels switch to Rayon; the factored
/// backend applies it per sub-state, the density matrix backend to its
/// `2n`-qubit buffer.
///
/// Under miri the four parallelism floors here drop so the same kernels split
/// across real worker threads at sizes the interpreter can execute; a 14-qubit
/// pass with the race detector on runs minutes per gate. `tests/determinism.rs`
/// is the miri CI target that relies on this.
#[cfg(not(miri))]
pub(crate) const PARALLEL_THRESHOLD_QUBITS: usize = 14;
#[cfg(miri)]
pub(crate) const PARALLEL_THRESHOLD_QUBITS: usize = 8;

/// Minimum elements per Rayon task in amplitude loops.
#[cfg(all(feature = "parallel", not(miri)))]
pub(crate) const MIN_PAR_ELEMS: usize = 4096;
#[cfg(all(feature = "parallel", miri))]
pub(crate) const MIN_PAR_ELEMS: usize = 64;

/// `with_min_len` value giving each Rayon task at least [`MIN_PAR_ELEMS`]
/// elements when the parallel iterator yields chunks of `chunk_size`.
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn chunk_min_len(chunk_size: usize) -> usize {
    (MIN_PAR_ELEMS / chunk_size).max(1)
}

/// Minimum iterations per Rayon task for index-driven loops whose
/// per-iteration work is heavier than one element (MCU, batch phase).
#[cfg(all(feature = "parallel", not(miri)))]
pub(crate) const MIN_PAR_ITERS: usize = 2048;
#[cfg(all(feature = "parallel", miri))]
pub(crate) const MIN_PAR_ITERS: usize = 32;

/// Element count above which a full-buffer reduction is worth Rayon fan-out.
/// Higher than [`PARALLEL_THRESHOLD_QUBITS`] because a reduction is one
/// lightweight streaming pass, so the fan-out only pays past `2^16`.
#[cfg(all(feature = "parallel", not(miri)))]
pub(crate) const MIN_PAR_REDUCE_ELEMS: usize = 1 << 16;
#[cfg(all(feature = "parallel", miri))]
pub(crate) const MIN_PAR_REDUCE_ELEMS: usize = 1 << 8;

/// `sum |a|^2` over a dense amplitude buffer, SIMD per chunk and parallel above
/// [`MIN_PAR_REDUCE_ELEMS`].
pub(crate) fn state_norm_sqr(state: &[Complex64]) -> f64 {
    #[cfg(feature = "parallel")]
    if state.len() >= MIN_PAR_REDUCE_ELEMS {
        use rayon::prelude::*;
        return state
            .par_chunks(MIN_PAR_ELEMS)
            .map(simd::norm_sqr_sum)
            .sum();
    }
    simd::norm_sqr_sum(state)
}

#[cfg(test)]
mod norm_tests {
    use super::state_norm_sqr;
    use num_complex::Complex64;

    // Both sides of the reduction's parallel threshold against a scalar sum.
    #[test]
    fn state_norm_sqr_matches_scalar_sum_across_the_parallel_threshold() {
        for len in [1usize, 3, 4096, (1 << 16) - 1, 1 << 16, (1 << 17) + 5] {
            let state: Vec<Complex64> = (0..len)
                .map(|i| Complex64::new(0.001 * i as f64 - 0.5, 0.002 * i as f64 + 0.25))
                .collect();
            let scalar: f64 = state.iter().map(Complex64::norm_sqr).sum();
            let got = state_norm_sqr(&state);
            assert!(
                (got - scalar).abs() <= 1e-9 * scalar.max(1.0),
                "len {len}: expected {scalar}, got {got}"
            );
        }
    }
}

/// Tolerance on `| ||psi||^2 - 1 |` for a caller-supplied start state.
pub(crate) const INITIAL_STATE_NORM_EPS: f64 = 1e-9;

/// Reject a start state that is not a normalized amplitude vector over a whole
/// number of qubits.
///
/// An unnormalized vector is rejected rather than rescaled: the dense backends
/// carry a deferred normalization factor that a start state resets to 1, so
/// rescaling here would hide the caller's error inside that factor. Finiteness
/// is checked first, since a NaN amplitude makes the norm comparison itself
/// false.
pub(crate) fn validate_initial_amplitudes(amplitudes: &[Complex64]) -> Result<()> {
    let dim = amplitudes.len();
    if !dim.is_power_of_two() || dim < 2 {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!("start state length must be a power of 2 and >= 2, got {dim}"),
        });
    }
    if amplitudes
        .iter()
        .any(|a| !a.re.is_finite() || !a.im.is_finite())
    {
        return Err(crate::error::PrismError::InvalidParameter {
            message: "start state has a non-finite amplitude".to_string(),
        });
    }
    let norm_sqr = state_norm_sqr(amplitudes);
    if (norm_sqr - 1.0).abs() > INITIAL_STATE_NORM_EPS {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!(
                "start state must be normalized, squared norm is {norm_sqr}; scale the \
                 amplitudes by 1/sqrt of it"
            ),
        });
    }
    Ok(())
}

/// Tolerance on `|rho[i][j] - conj(rho[j][i])|` for a caller-supplied mixture,
/// scaled by the larger of the two magnitudes when that exceeds 1.
pub(crate) const INITIAL_MIXTURE_HERMITIAN_EPS: f64 = 1e-12;

/// Reject a start mixture that is not a Hermitian, unit-trace `4^n` buffer in
/// the density-matrix layout, returning `n` otherwise.
///
/// Positive semidefiniteness is not checked: it needs an eigendecomposition
/// the backend does not carry, and a mixture with a negative eigenvalue
/// evolves without complaint, so that check stays with the caller.
pub(crate) fn validate_initial_density_matrix(rho: &[Complex64]) -> Result<usize> {
    let len = rho.len();
    let exponent = len.trailing_zeros();
    if !len.is_power_of_two() || len < 4 || !exponent.is_multiple_of(2) {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!(
                "start density matrix length must be 4^n for n >= 1 qubits, got {len}"
            ),
        });
    }
    if rho.iter().any(|a| !a.re.is_finite() || !a.im.is_finite()) {
        return Err(crate::error::PrismError::InvalidParameter {
            message: "start density matrix has a non-finite entry".to_string(),
        });
    }
    let num_qubits = (exponent / 2) as usize;
    let dim = 1usize << num_qubits;
    for r in 0..dim {
        for c in r..dim {
            let upper = rho[r * dim + c];
            let lower = rho[c * dim + r];
            let scale = upper.norm().max(lower.norm()).max(1.0);
            if (upper - lower.conj()).norm() > INITIAL_MIXTURE_HERMITIAN_EPS * scale {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "start density matrix is not Hermitian: entry ({r}, {c}) is {upper} \
                         but entry ({c}, {r}) conjugates to {}",
                        lower.conj()
                    ),
                });
            }
        }
    }
    let trace: f64 = (0..dim).map(|r| rho[r * dim + r].re).sum();
    if (trace - 1.0).abs() > INITIAL_STATE_NORM_EPS {
        return Err(crate::error::PrismError::InvalidParameter {
            message: format!(
                "start density matrix must have unit trace, trace is {trace}; scale the \
                 entries by 1/trace"
            ),
        });
    }
    Ok(num_qubits)
}

/// Tableau size at which stabilizer row loops parallelize.
#[cfg(feature = "parallel")]
pub(crate) const MIN_QUBITS_FOR_PAR_GATES: usize = 128;

/// Minimum anticommuting rows before a measurement's rowmul pass parallelizes.
#[cfg(feature = "parallel")]
pub(crate) const MIN_ANTI_ROWS_FOR_PAR: usize = 4;

/// Floor on an outcome probability before `1/sqrt`, so an outcome that rounds
/// to zero does not divide by zero.
pub(crate) const NORM_CLAMP_MIN: f64 = 1e-30;

/// Tolerance for treating a phase as 1+0i in diagonal and controlled-phase skips.
/// Tighter than identity detection (1e-12) because phase errors accumulate
/// multiplicatively.
pub(crate) const PHASE_IS_ONE_EPS: f64 = 1e-15;

pub(crate) use memory::{
    DM_QUBIT_CAP_ENV, check_state_allocation, dense_probability_len, dense_statevector_len,
    max_dense_outcome_bits, max_density_matrix_qubits, max_factored_merge_qubits,
    max_sparse_entries, max_stabilizer_cluster_qubits, max_statevector_qubits,
    mps_workspace_cap_elements, reserve_dense_output, stabilizer_cluster_error,
    statevector_probability_len, tensor_peak_cap_elements, tensor_peak_error,
    tensor_probability_len, workspace_allocation_error,
};

/// Whether `phase` equals `1+0i` within [`PHASE_IS_ONE_EPS`].
#[inline(always)]
pub(crate) fn is_phase_one(phase: Complex64) -> bool {
    (phase.re - 1.0).abs() < PHASE_IS_ONE_EPS && phase.im.abs() < PHASE_IS_ONE_EPS
}

/// Renormalization factor `1/sqrt(P(outcome))` after measurement collapse,
/// with the probability clamped to [`NORM_CLAMP_MIN`].
#[inline(always)]
pub(crate) fn measurement_inv_norm(outcome: bool, prob_one: f64) -> f64 {
    let prob_outcome = if outcome { prob_one } else { 1.0 - prob_one };
    1.0 / prob_outcome.clamp(NORM_CLAMP_MIN, 1.0).sqrt()
}

#[inline(always)]
/// A save point reached a backend, which means the runner did not split the
/// instruction stream at it.
///
/// Saves are serviced between segments by `sim`, using the backend's own export
/// methods, so no backend implements one. A route that cannot split (a compiled
/// sampler, a Pauli-propagation engine) declines the circuit before it runs, and
/// this is the message if one ever slips past that check.
pub(crate) fn save_not_applied(backend: &str, label: &str) -> crate::error::PrismError {
    crate::error::PrismError::IncompatibleBackend {
        backend: backend.to_string(),
        reason: format!(
            "save point `{label}` reached the backend; saves are recorded between              instruction segments, so this route does not support them"
        ),
    }
}

pub(crate) fn init_classical_bits(bits: &mut Vec<bool>, num: usize) {
    if bits.len() == num {
        bits.fill(false);
    } else {
        *bits = vec![false; num];
    }
}

/// Size the global Rayon pool to all logical cores unless `RAYON_NUM_THREADS` is
/// set. Hyperthreads hide memory latency once the state leaves L3: logical cores
/// measured 17% faster at 24 qubits.
///
/// Does nothing when the caller already runs inside a Rayon pool, as under
/// [`ThreadPool::install`](crate::ThreadPool::install). The `Once` is not
/// consumed on that path, so a later call from outside a pool still installs
/// the global pool. Otherwise only the first call takes effect.
#[cfg(feature = "parallel")]
pub(crate) fn init_thread_pool() {
    if rayon::current_thread_index().is_some() {
        return;
    }

    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
        }
    });
}

#[cfg(all(test, feature = "parallel"))]
mod thread_pool_tests {
    // The global pool is process state, so this is the only case in the lib
    // test binary that may configure it. The scoped half lives in
    // `tests/thread_pool.rs`.
    #[test]
    fn init_thread_pool_sizes_the_global_pool() {
        let expected: usize = match std::env::var("RAYON_NUM_THREADS") {
            Ok(requested) => requested.parse().expect("RAYON_NUM_THREADS is a count"),
            Err(_) => std::thread::available_parallelism().map_or(1, |n| n.get()),
        };

        super::init_thread_pool();

        assert_eq!(rayon::current_num_threads(), expected);
    }
}

/// Buffer width for [`sorted_mcu_qubits`]. A dense state indexes amplitudes with a
/// `usize` shift, so it never holds more than 63 qubits and an `Mcu` on one never
/// names more than that many.
pub(crate) const MCU_QUBIT_BUF: usize = 64;

/// Write `controls` plus `target` into `buf` sorted ascending, returning the count.
#[inline(always)]
pub(crate) fn sorted_mcu_qubits(
    controls: &[usize],
    target: usize,
    buf: &mut [usize; MCU_QUBIT_BUF],
) -> usize {
    let n = controls.len() + 1;
    buf[..controls.len()].copy_from_slice(controls);
    buf[controls.len()] = target;
    buf[..n].sort_unstable();
    n
}

/// Packed measurement outcomes produced by [`Backend::sample_basis_states`].
///
/// Holds `num_qubits.div_ceil(64)` words per shot; bit `q % 64` of word
/// `q / 64` carries the outcome for qubit `q`. Packed rather than one index
/// per shot because MPS and the factored backend run past 64 qubits, which is
/// the regime the native samplers exist for.
#[derive(Debug, Clone)]
pub struct BasisSamples {
    /// Outcome words for all shots, `words_per_shot` consecutive words each.
    words: Vec<u64>,
    /// Words per shot, `num_qubits.div_ceil(64)` with a minimum of 1.
    words_per_shot: usize,
}

impl BasisSamples {
    pub(crate) fn new(num_shots: usize, num_qubits: usize) -> Self {
        let words_per_shot = num_qubits.div_ceil(64).max(1);
        Self {
            words: vec![0u64; num_shots * words_per_shot],
            words_per_shot,
        }
    }

    #[inline(always)]
    pub(crate) fn set(&mut self, shot: usize, qubit: usize) {
        self.words[shot * self.words_per_shot + qubit / 64] |= 1u64 << (qubit % 64);
    }

    /// Record a whole shot from a basis-state index, for backends that key
    /// their state by one. A `usize` index never spans more than one word.
    #[inline(always)]
    pub(crate) fn set_index(&mut self, shot: usize, index: usize) {
        self.words[shot * self.words_per_shot] = index as u64;
    }

    pub fn num_shots(&self) -> usize {
        self.words.len() / self.words_per_shot
    }

    /// Word storage, [`Self::words_per_shot`] consecutive words per shot, so a
    /// sampler filling shots in parallel can split it into disjoint chunks.
    pub(crate) fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    pub(crate) fn words_per_shot(&self) -> usize {
        self.words_per_shot
    }

    /// Measured outcome for `qubit` in `shot`, `true` for |1⟩.
    #[inline(always)]
    pub fn bit(&self, shot: usize, qubit: usize) -> bool {
        let word = self.words[shot * self.words_per_shot + qubit / 64];
        (word >> (qubit % 64)) & 1 == 1
    }

    /// Unpack qubits `0..num_bits` of every shot, one set bit at a time rather
    /// than one [`Self::bit`] call per qubit. Only correct when classical bit
    /// `i` carries qubit `i`; callers test that before choosing this path.
    pub(crate) fn to_shots(&self, num_bits: usize) -> Vec<Vec<bool>> {
        let mut shots = vec![vec![false; num_bits]; self.num_shots()];
        let live_words = num_bits.div_ceil(64).min(self.words_per_shot);
        // Only the word straddling `num_bits` is masked, which is not
        // necessarily the last live one: a register narrower than the classical
        // register runs out of words first, and every word it does hold is
        // wholly in range.
        let straddling_word = num_bits / 64;
        let tail = num_bits % 64;
        for (s, shot) in shots.iter_mut().enumerate() {
            let row = &self.words[s * self.words_per_shot..(s + 1) * self.words_per_shot];
            for (w, &word) in row[..live_words].iter().enumerate() {
                let mut bits = word;
                if tail != 0 && w == straddling_word {
                    bits &= (1u64 << tail) - 1;
                }
                let base = w * 64;
                while bits != 0 {
                    shot[base + bits.trailing_zeros() as usize] = true;
                    bits &= bits - 1;
                }
            }
        }
        shots
    }
}

/// Trait that all simulation backends must implement.
pub trait Backend {
    /// Backend name used in error messages and benchmark labels.
    fn name(&self) -> &'static str;

    /// Which engine this is, for the provenance attached to every result. The
    /// default names an out-of-tree backend by its [`Backend::name`].
    fn resolved(&self) -> ResolvedBackend {
        ResolvedBackend::Other(self.name())
    }

    /// Whether this representation can discard state weight, and how much it
    /// discarded on the run just executed. Called once per run, after the
    /// circuit has been applied; the default reports an exact representation.
    fn exactness(&self) -> Exactness {
        Exactness::Exact
    }

    /// Where the state lived during the run. Only the statevector has a device
    /// path, and only when the `gpu` feature is on.
    fn placement(&self) -> Placement {
        Placement::Host
    }

    /// Peak bond dimension the run kept against its cap, for a representation
    /// that has one. Called once per run, after the circuit has been applied;
    /// the default reports none.
    fn bond_report(&self) -> Option<BondReport> {
        None
    }

    /// Reset to |0...0⟩ with all classical bits cleared.
    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()>;

    /// Whether [`Backend::init_from_amplitudes`] can start this backend from a
    /// caller-supplied state.
    fn supports_initial_state(&self) -> bool {
        false
    }

    /// Initialize from a dense amplitude vector instead of |0...0⟩.
    ///
    /// `amplitudes` is indexed like [`Backend::export_statevector`] output, qubit
    /// 0 in the least significant bit, and its length sets the register width.
    /// Implementors validate it through `validate_initial_amplitudes`, so a
    /// length that is not a power of two, a non-finite amplitude, or a squared
    /// norm off unity returns `InvalidParameter` rather than starting a run on a
    /// state that is not a state. The default reports that the representation
    /// cannot hold an arbitrary one; see the module docs for what declines it.
    fn init_from_amplitudes(
        &mut self,
        amplitudes: Vec<Complex64>,
        num_classical_bits: usize,
    ) -> Result<()> {
        let _ = (amplitudes, num_classical_bits);
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "initialization from a caller-supplied state".to_string(),
        })
    }

    /// Whether [`Backend::init_from_density_matrix`] can start this backend
    /// from a caller-supplied mixture.
    fn supports_initial_density_matrix(&self) -> bool {
        false
    }

    /// Initialize from a dense density matrix instead of |0...0⟩.
    ///
    /// `rho` is laid out like
    /// [`DensityMatrixBackend::density_matrix`](density_matrix::DensityMatrixBackend::density_matrix)
    /// output, row-major `2^n x 2^n` with qubit 0 in the least significant bit
    /// of both indices, and its length sets the register width. Implementors
    /// validate it through `validate_initial_density_matrix`, so a length that
    /// is not `4^n`, a non-finite entry, an entry pair off Hermitian by more
    /// than 1e-12, or a trace off unity by more than 1e-9 returns
    /// `InvalidParameter` naming the check. Positive semidefiniteness is not
    /// checked. The default reports that the representation holds only pure
    /// states; see the module docs for what declines it.
    fn init_from_density_matrix(
        &mut self,
        rho: Vec<Complex64>,
        num_classical_bits: usize,
    ) -> Result<()> {
        let _ = (rho, num_classical_bits);
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "initialization from a caller-supplied density matrix".to_string(),
        })
    }

    /// Apply a single instruction to the current state.
    ///
    /// Instructions arrive in circuit order. Backends may assume:
    /// - Qubit indices are valid (checked during circuit construction).
    /// - Gate arity matches target count.
    fn apply(&mut self, instruction: &Instruction) -> Result<()>;

    /// Classical bits by index, `true` for a measured |1⟩.
    fn classical_results(&self) -> &[bool];

    /// Probability of each basis state, length `2^n`. A backend that cannot
    /// expand its state returns `BackendUnsupported`.
    fn probabilities(&self) -> Result<Vec<f64>>;

    /// Per-block probabilities for a backend holding a product of independent
    /// sub-states, skipping the `2^n` Kronecker expansion
    /// [`Backend::probabilities`] would materialize.
    ///
    /// `None` means the caller takes [`Backend::probabilities`] instead, which
    /// is the right answer for a single block or a monolithic state. Each
    /// block's `probs` is indexed by its own qubits in ascending global order,
    /// matching the `mask` convention of [`crate::sim::FactoredBlock`].
    fn block_probabilities(&self) -> Option<crate::sim::Probabilities> {
        None
    }

    fn num_qubits(&self) -> usize;

    /// Apply instructions in order. The default loops over
    /// [`apply`](Backend::apply); the stabilizer backend overrides it to group
    /// gates by target word.
    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        for instruction in instructions {
            self.apply(instruction)?;
        }
        Ok(())
    }

    /// Execute a guarded region's body iff its condition holds.
    ///
    /// Every backend routes [`Instruction::Region`] here, so the branch
    /// semantics live in one place. The body reaches
    /// [`apply_instructions`](Backend::apply_instructions), which keeps any
    /// batching a backend overrides it with.
    fn apply_region(&mut self, region: &crate::circuit::GuardedRegion) -> Result<()> {
        let taken = region.condition().evaluate(self.classical_results());
        if taken {
            self.apply_instructions(region.body())
        } else {
            Ok(())
        }
    }

    /// Whether this backend accepts `Gate::Fused`; `sim` skips fusion when it
    /// returns `false`, as a tableau cannot decode a fused matrix.
    ///
    /// Returning `true` also accepts `MultiFused` and `Multi2q`, whose payload
    /// gate lists must be applied in the order they are stored. Fusion emits
    /// each list within one cache tier so the tiled kernels preserve that
    /// order; a backend that rewrites the payload's qubit indices can push
    /// gates across a tier boundary and get them silently reordered.
    fn supports_fused_gates(&self) -> bool {
        true
    }

    /// Width in qubits of the buffer this backend sweeps for an `n`-qubit
    /// circuit.
    ///
    /// The fusion floors are calibrated against the cost of one statevector
    /// pass, so `sim` gates them on this rather than on the circuit width. The
    /// two coincide everywhere except the density matrix, which holds an
    /// `n`-qubit mixture as a `2n`-qubit statevector.
    fn fusion_state_qubits(&self, num_qubits: usize) -> usize {
        num_qubits
    }

    /// Whether this backend has a native `Gate::QftBlock` kernel. Only the CPU
    /// statevector does; others receive the textbook gates.
    fn supports_qft_block(&self) -> bool {
        false
    }

    /// Whether this backend has a native `Gate::PauliRot` kernel. Only the host
    /// CPU statevector does; others receive the CNOT-ladder lowering from
    /// `circuit::expand_pauli_rotations`.
    fn supports_pauli_rotation(&self) -> bool {
        false
    }

    /// Export the state as a dense `2^n` amplitude vector, used for backend
    /// handoffs such as the temporal Clifford split. See the module docs for
    /// what declines it.
    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "statevector export".to_string(),
        })
    }

    /// P(qubit = |1⟩) without collapsing the state, for state-dependent noise
    /// channels. The default reads [`Backend::reduced_density_matrix_1q`].
    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        let rho = self.reduced_density_matrix_1q(qubit)?;
        Ok(rho[1][1].re.clamp(0.0, 1.0))
    }

    /// One-qubit reduced density matrix `[[p0, r*], [r, p1]]` with
    /// `r = <1|rho|0>`, without collapsing the state. Feeds Kraus channels whose
    /// branch weights depend on coherence; see the module docs for what
    /// declines it.
    fn reduced_density_matrix_1q(&self, _qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "reduced_density_matrix_1q".to_string(),
        })
    }

    /// Whether this backend can run a [`NoiseChannel::Kraus2q`](crate::sim::noise::NoiseChannel)
    /// branch, which needs both [`Backend::reduced_density_matrix_2q`] and a
    /// `Gate::Fused2q` kernel. Checked before a shot starts, so an incapable
    /// backend is named at dispatch rather than part way through a trajectory.
    fn supports_two_qubit_kraus(&self) -> bool {
        false
    }

    /// Two-qubit reduced density matrix, without collapsing the state.
    ///
    /// Indexed `rho[t][t']` with `t = 2 * bit(q0) + bit(q1)`, the packing
    /// [`crate::gates::Gate::matrix_4x4`] uses, so `q0` is the high bit.
    ///
    /// # Panics
    ///
    /// Implementations index the two qubits as distinct bit positions, so
    /// `q0 == q1` panics rather than returning a block read from overlapping
    /// amplitudes.
    /// The default reads the pair out of [`Backend::reduced_density_matrix`],
    /// so a representation with a partial trace answers without writing a
    /// second kernel, and one without it declines here as it does there.
    fn reduced_density_matrix_2q(&mut self, q0: usize, q1: usize) -> Result<[[Complex64; 4]; 4]> {
        assert_ne!(q0, q1, "reduced_density_matrix_2q needs distinct qubits");
        let rho = self.reduced_density_matrix(&[q1, q0])?;
        let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
        for (t, row) in out.iter_mut().enumerate() {
            for (tp, entry) in row.iter_mut().enumerate() {
                *entry = rho[t * 4 + tp];
            }
        }
        Ok(out)
    }

    /// Reset a qubit to |0⟩.
    ///
    /// The contract is the reset channel `rho -> |0><0| (x) tr_q rho`: the
    /// qubit is traced out and replaced by |0⟩, leaving the rest of the
    /// register in the mixture the trace produces. Projecting onto |0⟩ and
    /// renormalizing is not equivalent. The two agree only when the qubit is
    /// unentangled; when it is entangled, projection also collapses its
    /// partners into the branch correlated with the |0⟩ outcome. On a Bell
    /// pair, resetting qubit 1 leaves ⟨Z0⟩ = 0 under the channel and
    /// ⟨Z0⟩ = 1 under projection.
    ///
    /// A backend holding a single pure state cannot represent the mixture, so
    /// it implements one trajectory of the channel: sample the measurement
    /// outcome, collapse onto it, and apply X when the outcome is 1. Averaged
    /// over shots that reproduces the channel. Backends holding a mixed state
    /// (density matrix) apply the channel directly.
    fn reset(&mut self, _qubit: usize) -> Result<()> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "reset".to_string(),
        })
    }

    /// Whether [`Backend::sample_basis_states`] draws from this backend's own
    /// representation.
    ///
    /// `false` routes shot and count queries through the dense probability
    /// vector, which caps them at the machine's dense-output budget. Backends
    /// holding a polynomial-size representation override both this and
    /// [`Backend::sample_basis_states`].
    fn supports_native_sampling(&self) -> bool {
        false
    }

    /// Draw `num_shots` computational-basis outcomes from the current state.
    ///
    /// Seeded from `seed` alone rather than from the backend's own RNG, so a
    /// shot request replays exactly. Does not collapse the state. The default
    /// reports that the backend has no native sampler.
    ///
    /// Takes `&mut self` because a backend may have to reorganize its own
    /// storage before it can draw: the distributed backend restores its qubit
    /// map, which is a collective. The represented state is unchanged either
    /// way.
    fn sample_basis_states(&mut self, _num_shots: usize, _seed: u64) -> Result<BasisSamples> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "native basis-state sampling".to_string(),
        })
    }

    /// Whether [`Backend::pauli_expectations`] evaluates observables on this
    /// backend's own representation.
    fn supports_pauli_expectation(&self) -> bool {
        false
    }

    /// Exact `⟨ψ|P_k|ψ⟩` for each joint Pauli observable `P_k`.
    ///
    /// Each observable lists one factor per non-identity qubit; omitted qubits
    /// carry identity. Normalization independent, so implementors divide by
    /// `⟨ψ|ψ⟩` rather than assuming a unit-norm state. Duplicate factors on one
    /// qubit are rejected.
    fn pauli_expectations(&self, _observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "Pauli expectation values".to_string(),
        })
    }

    /// Schmidt values of the state across the cut between `subsystem` and its
    /// complement: descending, numerically zero values dropped, squares summing
    /// to 1 whatever norm the representation carries.
    ///
    /// `subsystem` must be non-empty, leave its complement non-empty, and name
    /// no qubit twice (`schmidt::validate_subsystem`); its order is irrelevant.
    /// Takes `&mut self` because the MPS answer walks its orthogonality center
    /// to the cut. The represented state is unchanged. The one-SVD and
    /// statevector routes resolve values down to the `1e-14` floor; the
    /// reduced-density route, which an MPS takes for a subsystem that is not
    /// contiguous in chain order, works from the squared spectrum and resolves
    /// them down to about `1e-7` of the largest. The default reports that the
    /// representation offers no spectrum; see the module docs for what
    /// declines it.
    fn schmidt_values(&mut self, _subsystem: &[usize]) -> Result<Vec<f64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "Schmidt values".to_string(),
        })
    }

    /// Entanglement entropy of `subsystem` in nats: `-sum p ln p` over
    /// `p = s^2 / sum s^2` for the [`Backend::schmidt_values`] `s`, which is
    /// where the default reads it from.
    fn entanglement_entropy(&mut self, subsystem: &[usize]) -> Result<f64> {
        let values = self.schmidt_values(subsystem)?;
        Ok(schmidt::entropy_of_schmidt_values(&values))
    }

    /// Reduced density matrix of `subsystem`, row major with side `2^k` for
    /// `k` qubits: `rho[t * 2^k + t']` is `<t|rho|t'>`, where bit `i` of `t`
    /// is the state of `subsystem[i]`, so `subsystem[0]` is the lowest bit as
    /// `q[0]` is in a basis index. Trace one whatever norm the representation
    /// carries; Hermitian to rounding.
    ///
    /// `subsystem` must be non-empty and name no qubit twice
    /// (`schmidt::validate_qubit_set`); the whole register is allowed, and on a
    /// mixture returns the state itself. The `4^k` entries are priced as a
    /// `2k`-qubit statevector against the dense export cap
    /// (`reduced_density::reduced_density_side`). Takes `&mut self` as
    /// [`Backend::schmidt_values`] does. The default reports that the
    /// representation offers no partial trace; see the module docs for what
    /// declines it.
    fn reduced_density_matrix(&mut self, _subsystem: &[usize]) -> Result<Vec<Complex64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "reduced density matrix".to_string(),
        })
    }

    /// The concrete backend behind a `&dyn Backend`, so [`Backend::overlap_sq`]
    /// can recognize its own representation on the other side of the inner
    /// product. An implementor that wants the fast paths writes `Some(self)`;
    /// the default hides the concrete type, which costs only the dense route.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// `|<self|other>|^2` over the two normalized states: 1 for the same state
    /// up to phase, 0 for orthogonal ones.
    ///
    /// The modulus squared rather than the amplitude, since a tableau keeps no
    /// global phase and every MPS truncation moves one, so this is the only
    /// number all the representations agree on. Widths that disagree report
    /// `InvalidParameter`. Normalization divides by `<self|self><other|other>`
    /// on every route, so an unnormalized chain or a pruned sparse map answers
    /// the same as a unit-norm state.
    ///
    /// The default exports both states under the dense export cap, which is
    /// how far a pair of unlike representations is served; an override that
    /// finds its own kind through [`Backend::as_any`] answers at any width.
    /// See the module docs for what declines it.
    fn overlap_sq(&self, other: &dyn Backend) -> Result<f64> {
        overlap::export_overlap_sq(
            self.name(),
            self.num_qubits(),
            || self.export_statevector(),
            other,
        )
    }

    /// Apply a 2×2 Kraus branch to one qubit. The matrix need not be unitary: a
    /// jump branch is a projector scaled by `1/sqrt(p_jump)`, and the no-jump
    /// branch renormalizes.
    ///
    /// The default builds a `Gate::Fused` and dispatches via `apply`, which
    /// heap-allocates once per call inside a gate-application path, so every
    /// backend that can reach this method overrides it. See the module docs for
    /// what declines it.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        use crate::circuit::smallvec;
        self.apply(&crate::circuit::Instruction::Gate {
            gate: crate::gates::Gate::Fused(Box::new(*matrix)),
            targets: smallvec![qubit],
        })
    }
}
