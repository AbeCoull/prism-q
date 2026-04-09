//! Simulation backend trait and implementations.
//!
//! Backends are the core execution engines. Each backend owns its quantum state
//! representation and applies circuit instructions to evolve the state.
//!
//! # Backend contract
//!
//! 1. Call [`Backend::init`] before any [`Backend::apply`] calls.
//! 2. Call [`Backend::apply`] for each instruction in circuit order.
//! 3. Measurement is destructive — it collapses the state.
//! 4. Given the same circuit and RNG seed, results must be deterministic.
//!
//! # Performance requirements for implementors
//!
//! - Gate application must avoid heap allocation in the hot path.
//! - Prefer direct indexing over iterator chains for state access.
//! - Use `#[inline]` and `#[inline(always)]` on gate kernels.
//! - Document all `unsafe` blocks with safety invariants.
//!
//! # Adding a new backend
//!
//! See `docs/architecture.md` § "Add a new backend" for the full playbook.

pub mod factored;
pub mod factored_stabilizer;
pub mod mps;
pub mod product;
pub(crate) mod simd;
pub mod sparse;
pub mod stabilizer;
pub mod statevector;
pub mod tensornetwork;

use num_complex::Complex64;

use crate::circuit::Instruction;
use crate::error::Result;

#[cfg(feature = "parallel")]
pub(crate) const PARALLEL_THRESHOLD_QUBITS: usize = 14;

#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_ELEMS: usize = 4096;

#[cfg(feature = "parallel")]
pub(crate) const MIN_PAR_ITERS: usize = 2048;

#[cfg(feature = "parallel")]
pub(crate) const MIN_QUBITS_FOR_PAR_GATES: usize = 128;

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

pub(crate) const MAX_PROB_QUBITS: usize = 25;

#[inline(always)]
pub(crate) fn is_phase_one(phase: Complex64) -> bool {
    (phase.re - 1.0).abs() < PHASE_IS_ONE_EPS && phase.im.abs() < PHASE_IS_ONE_EPS
}

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
/// Safe to call multiple times — only the first call takes effect.
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

#[inline(always)]
pub(crate) fn sorted_mcu_qubits(controls: &[usize], target: usize, buf: &mut [usize; 10]) -> usize {
    let n = controls.len() + 1;
    buf[..controls.len()].copy_from_slice(controls);
    buf[controls.len()] = target;
    buf[..n].sort_unstable();
    n
}

/// Trait that all simulation backends must implement.
pub trait Backend {
    /// Human-readable backend name (for error messages, logging, and benchmarks).
    fn name(&self) -> &'static str;

    /// Initialize (or reset) state for a circuit with the given dimensions.
    ///
    /// After this call the backend is in the |0...0⟩ state.
    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()>;

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
    /// provide this efficiently — they may return `Err(BackendUnsupported)`.
    fn probabilities(&self) -> Result<Vec<f64>>;

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

    /// Export the current quantum state as a dense statevector.
    ///
    /// Returns a `Vec<Complex64>` of length 2^n containing the full amplitude
    /// vector. Enables backend transitions (e.g., Stabilizer → Statevector
    /// for temporal Clifford decomposition).
    ///
    /// Not all backends support this efficiently. The default implementation
    /// returns `Err(BackendUnsupported)`.
    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "statevector export".to_string(),
        })
    }

    /// Compute P(qubit = |1⟩) without collapsing the state.
    ///
    /// Used by the trajectory engine for state-dependent noise channels
    /// (amplitude damping, phase damping). The default returns `BackendUnsupported`.
    fn qubit_probability(&self, _qubit: usize) -> Result<f64> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "qubit_probability".to_string(),
        })
    }

    /// Reset a qubit to |0⟩, discarding any prior amplitude on that qubit.
    ///
    /// Destructive, non-unitary. Used by OpenQASM `reset` and as a primitive
    /// for thermal relaxation trajectory simulation. The default returns
    /// `BackendUnsupported`; backends override for their native representation.
    fn reset(&mut self, _qubit: usize) -> Result<()> {
        Err(crate::error::PrismError::BackendUnsupported {
            backend: self.name().to_string(),
            operation: "reset".to_string(),
        })
    }

    /// Apply a 2×2 matrix to a single qubit without allocating.
    ///
    /// Used by the trajectory engine to apply Kraus operators (amplitude
    /// damping, phase damping, thermal relaxation) without boxing into a
    /// `Gate::Fused` on every call. The default falls back to building a
    /// `Gate::Fused` and dispatching via `apply`; backends may override for a
    /// zero-allocation fast path.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        use crate::circuit::smallvec;
        self.apply(&crate::circuit::Instruction::Gate {
            gate: crate::gates::Gate::Fused(Box::new(*matrix)),
            targets: smallvec![qubit],
        })
    }
}
