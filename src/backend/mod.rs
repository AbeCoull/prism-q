//! Simulation backend trait and implementations.
//!
//! Backends are the core execution engines. Each backend owns its quantum state
//! representation and applies circuit instructions to evolve the state.
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
//! | `export_statevector` | DensityMatrix | A mixture of pure states has no statevector. Read `DensityMatrixBackend::purity` or reduce the state instead. |
//! | `export_statevector` | FactoredStabilizer | Exports while one tableau covers every qubit; past that there is no joint tableau to expand. |
//! | `init_from_amplitudes` | Everything except Statevector and DensityMatrix | The input is a dense `2^n` amplitude vector, and a tableau, a product state, or a factored register holds only the states its structure can express. MPS could decode one by sequential SVD, but the bond cap would truncate the state the caller supplied. |
//!
//! [`Backend::reduced_density_matrix_1q`] and [`Backend::apply_1q_matrix`] are
//! two halves of one capability, sampling a non-Pauli branch and applying the
//! operator it selects, so their coverage is one set by construction. That set is
//! `BackendKind::supports_general_noise`.
//!
//! # Adding a new backend
//!
//! Implement the [`Backend`] trait below, following the contract and
//! performance requirements above.

pub mod density_matrix;
#[cfg(feature = "distributed")]
pub mod distributed_statevector;
pub mod factored;
pub mod factored_stabilizer;
pub(crate) mod memory;
pub mod mps;
pub mod product;
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
use crate::sim::{Exactness, Placement, ResolvedBackend};

/// Qubit count at which dense amplitude kernels switch to Rayon; the factored
/// backend applies it per sub-state, the density matrix backend to its
/// `2n`-qubit buffer.
pub(crate) const PARALLEL_THRESHOLD_QUBITS: usize = 14;

/// Minimum elements per Rayon task in amplitude loops.
#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_ELEMS: usize = 4096;

/// `with_min_len` value giving each Rayon task at least [`MIN_PAR_ELEMS`]
/// elements when the parallel iterator yields chunks of `chunk_size`.
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn chunk_min_len(chunk_size: usize) -> usize {
    (MIN_PAR_ELEMS / chunk_size).max(1)
}

/// Minimum iterations per Rayon task for index-driven loops whose
/// per-iteration work is heavier than one element (MCU, batch phase).
#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_ITERS: usize = 2048;

/// Element count above which a full-buffer reduction is worth Rayon fan-out.
/// Higher than [`PARALLEL_THRESHOLD_QUBITS`] because a reduction is one
/// lightweight streaming pass, so the fan-out only pays past `2^16`.
#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_REDUCE_ELEMS: usize = 1 << 16;

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

/// Tableau size at which stabilizer row loops parallelize.
#[cfg(feature = "parallel")]
pub(crate) const MIN_QUBITS_FOR_PAR_GATES: usize = 128;

/// Minimum anticommuting rows before a measurement's rowmul pass parallelizes.
#[cfg(feature = "parallel")]
pub(crate) const MIN_ANTI_ROWS_FOR_PAR: usize = 4;

/// Minimum probability/norm value for measurement normalization.
///
/// Used as `prob.clamp(NORM_CLAMP_MIN, 1.0).sqrt()` to avoid division by zero
/// when a measurement outcome has near-zero probability due to floating point.
pub(crate) const NORM_CLAMP_MIN: f64 = 1e-30;

/// Tolerance for detecting whether a complex phase equals 1+0i.
///
/// Used in diagonal gate optimizations (`skip_lo`) and controlled-phase
/// identity checks. Tighter than identity detection (1e-12) because phase
/// errors accumulate multiplicatively.
pub(crate) const PHASE_IS_ONE_EPS: f64 = 1e-15;

pub(crate) use memory::{
    DM_QUBIT_CAP_ENV, check_state_allocation, dense_probability_len, dense_statevector_len,
    max_dense_outcome_bits, max_density_matrix_qubits, max_factored_merge_qubits,
    max_sparse_entries, max_statevector_qubits, mps_workspace_cap_elements, reserve_dense_output,
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
pub(crate) fn init_classical_bits(bits: &mut Vec<bool>, num: usize) {
    if bits.len() == num {
        bits.fill(false);
    } else {
        *bits = vec![false; num];
    }
}

/// Initialize the Rayon thread pool to use all logical cores.
///
/// At 24q+ where state exceeds L3 cache, hyperthreads help hide memory
/// latency. Benchmarks show 17% improvement with logical cores at 24q.
/// The user can override via `RAYON_NUM_THREADS`.
///
/// Safe to call multiple times. Only the first call takes effect.
#[cfg(feature = "parallel")]
pub(crate) fn init_thread_pool() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            let threads = num_cpus::get();
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
        }
    });
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

    /// Measured outcome for `qubit` in `shot`, `true` for |1⟩.
    #[inline(always)]
    pub fn bit(&self, shot: usize, qubit: usize) -> bool {
        let word = self.words[shot * self.words_per_shot + qubit / 64];
        (word >> (qubit % 64)) & 1 == 1
    }
}

/// Trait that all simulation backends must implement.
pub trait Backend {
    /// Human-readable backend name (for error messages, logging, and benchmarks).
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

    /// Initialize (or reset) state for a circuit with the given dimensions.
    ///
    /// After this call the backend is in the |0...0⟩ state.
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

    /// Apply a single instruction to the current state.
    ///
    /// Instructions arrive in circuit order. Backends may assume:
    /// - Qubit indices are valid (checked during circuit construction).
    /// - Gate arity matches target count.
    fn apply(&mut self, instruction: &Instruction) -> Result<()>;

    /// Read classical measurement results.
    ///
    /// Returns a slice indexed by classical bit number. `true` = measured |1⟩.
    fn classical_results(&self) -> &[bool];

    /// Compute the probability of each computational basis state.
    ///
    /// Returns a `Vec<f64>` of length 2^num_qubits. Not all backends can
    /// provide this efficiently, they may return `Err(BackendUnsupported)`.
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

    /// Number of qubits the backend is currently configured for.
    fn num_qubits(&self) -> usize;

    /// Apply a batch of instructions to the current state.
    ///
    /// The default implementation calls [`apply`](Backend::apply) in a loop.
    /// Backends may override to batch gate operations for better cache
    /// utilization (e.g., the stabilizer backend groups gates by target word).
    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        for instruction in instructions {
            self.apply(instruction)?;
        }
        Ok(())
    }

    /// Whether this backend can handle `Gate::Fused` variants.
    ///
    /// Backends that operate on symbolic gate representations (e.g. stabilizer
    /// tableau) cannot decode a fused matrix back to individual gates. The
    /// simulation engine skips the fusion pass when this returns `false`.
    fn supports_fused_gates(&self) -> bool {
        true
    }

    /// Whether this backend has a native kernel for `Gate::QftBlock`.
    ///
    /// Native support is currently limited to whole-state CPU statevector QFT.
    /// Other backends expand the block to textbook gates before dispatch.
    fn supports_qft_block(&self) -> bool {
        false
    }

    /// Export the current quantum state as a dense statevector.
    ///
    /// Returns a `Vec<Complex64>` of length 2^n containing the full amplitude
    /// vector. Enables backend transitions (e.g., Stabilizer → Statevector
    /// for temporal Clifford decomposition).
    ///
    /// Implemented by every backend that holds a pure state, including the
    /// factored one, whose blocks tensor back into a joint vector. See the module
    /// docs for what declines it.
    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "statevector export".to_string(),
        })
    }

    /// Compute P(qubit = |1⟩) without collapsing the state.
    ///
    /// Used by the trajectory engine for state-dependent noise channels
    /// (amplitude damping, phase damping). The default derives from
    /// [`Backend::reduced_density_matrix_1q`].
    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        let rho = self.reduced_density_matrix_1q(qubit)?;
        Ok(rho[1][1].re.clamp(0.0, 1.0))
    }

    /// Compute the one-qubit reduced density matrix without collapsing the state.
    ///
    /// Returned as `[[p0, r*], [r, p1]]`, where `r = <1|rho|0>`.
    /// Used by the trajectory engine for dense custom Kraus channels whose
    /// branch probabilities depend on coherence, not just populations. Backends
    /// override when their representation can expose this efficiently enough for
    /// noise sampling; see the module docs for what declines it and why.
    fn reduced_density_matrix_1q(&self, _qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "reduced_density_matrix_1q".to_string(),
        })
    }

    /// Reset a qubit to |0⟩, discarding any prior amplitude on that qubit.
    ///
    /// Destructive, non-unitary. Used by OpenQASM `reset` and as a primitive
    /// for thermal relaxation trajectory simulation. The default returns
    /// `BackendUnsupported`; backends override for their native representation.
    ///
    /// The contract is the reset channel `rho -> |0><0| (x) tr_q rho`: the
    /// qubit is traced out and replaced by |0⟩, leaving the rest of the
    /// register in the mixture the trace produces. Projecting onto |0⟩ and
    /// renormalizing is **not** equivalent. The two agree only when the qubit
    /// is unentangled; when it is entangled, projection also collapses its
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

    /// Apply a 2×2 matrix to a single qubit without allocating.
    ///
    /// Used by the trajectory engine to apply Kraus operators (amplitude
    /// damping, phase damping, thermal relaxation, generic normalized Kraus).
    /// The matrix need not be unitary: a jump branch is a projector scaled by
    /// `1/sqrt(p_jump)`, and the no-jump branch renormalizes.
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
