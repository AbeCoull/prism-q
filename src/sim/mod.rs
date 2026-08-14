//! Simulation orchestration.
//!
//! Connects the circuit IR to a backend. This module is deliberately thin,
//! the complexity lives in the backends and the parser. Entry points:
//! [`simulate`], [`run_qasm`], [`run_on`].

pub mod compiled;
mod decomposed;
mod dispatch;
pub mod gradient;
pub mod homological;
mod metadata;
pub mod noise;
mod observable;
mod probability;
pub(crate) mod shots;
pub mod stabilizer_rank;
mod terminal_sampling;
mod trajectory;
pub mod unified_pauli;

pub(crate) use decomposed::merge_probabilities;
use decomposed::{
    MIN_DECOMPOSITION_QUBITS, run_decomposed, run_decomposed_prefused, should_decompose,
};
pub use dispatch::BackendKind;
use dispatch::{
    AUTO_APPROX_MAX_TERMS, AUTO_SPD_MAX_TERMS, BackendPlan, ExecutionPlan, Family,
    MAX_AUTO_T_COUNT_APPROX, MAX_AUTO_T_COUNT_EXACT, MAX_AUTO_T_COUNT_SHOTS,
    MAX_STABILIZER_RANK_QUBITS, MIN_BLOCK_FOR_FACTORED_STAB, MIN_FACTORED_STABILIZER_QUBITS,
    MIN_QUBITS_FOR_SPD_AUTO, TemporalCliffordPlan, accel_for, approximate_route_name,
    auto_selects_cpu_statevector, build_statevector, has_temporal_clifford_opportunity,
    initial_state_plan, plan_for_family, plan_temporal_clifford, resolve, resolve_backend,
    run_temporal_clifford, stabilizer_rank_budget, validate_explicit_backend,
};
pub use metadata::{Exactness, ExpectationResult, Placement, ResolvedBackend, RunMetadata};
pub use observable::{ObservableExpectation, PauliObservable};
pub use probability::{FactoredBlock, Probabilities, ProbabilitiesIter};
pub use shots::{ShotsResult, bitstring};

use std::collections::HashMap;

use num_complex::Complex64;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use shots::{packed_shots_to_classical_bits, sample_shots, shots_from_basis_samples};
use terminal_sampling::{
    sample_counts_from_probs, sample_counts_from_state, sample_shots_from_probs,
    sample_shots_from_state,
};
use unified_pauli::{PauliAxis, PauliTerm};

type TerminalStatevector = (StatevectorBackend, Vec<(usize, usize)>);

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

/// Result of a generic simulation run.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// Classical measurement outcomes, indexed by classical bit number.
    /// `true` = measured |1⟩.
    pub classical_bits: Vec<bool>,
    /// Probability of each computational basis state (length 2^n).
    ///
    /// `None` means the selected backend cannot expose a dense probability
    /// distribution for this circuit. Other probability extraction failures
    /// are returned as errors by the query that produced this result.
    pub probabilities: Option<Probabilities>,
    /// Which engine ran, whether the answer is exact, and where the state lived.
    pub metadata: RunMetadata,
}

/// Frequency histogram returned by query-aware count sampling.
#[derive(Debug, Clone)]
pub struct CountsResult {
    /// Histogram keyed by packed classical bits; same key layout as
    /// [`ShotsResult::counts`], formattable with [`bitstring`].
    pub counts: HashMap<Vec<u64>, u64>,
    pub num_classical_bits: usize,
    pub metadata: RunMetadata,
}

impl CountsResult {
    pub fn into_counts(self) -> HashMap<Vec<u64>, u64> {
        self.counts
    }
}

/// Per-qubit marginal probabilities returned by query-aware marginal sampling.
#[derive(Debug, Clone)]
pub struct MarginalsResult {
    /// `(P(0), P(1))` per qubit, indexed by qubit number.
    pub marginals: Vec<(f64, f64)>,
    pub metadata: RunMetadata,
}

impl MarginalsResult {
    pub fn into_vec(self) -> Vec<(f64, f64)> {
        self.marginals
    }
}

/// Typestate marker: [`Simulate`] builder with no seed chosen yet.
#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

/// Typestate marker: [`Simulate`] builder with its RNG seed fixed.
#[derive(Debug, Clone, Copy)]
pub struct Seeded {
    seed: u64,
}

/// Builder for query-aware simulation requests.
pub struct Simulate<'c, SeedState> {
    circuit: &'c Circuit,
    kind: BackendKind,
    seed: SeedState,
    noise_model: Option<&'c noise::NoiseModel>,
    initial_state: Option<&'c [Complex64]>,
    require_exact: bool,
}

impl<'c, SeedState> Simulate<'c, SeedState> {
    /// Select an explicit backend kind instead of [`BackendKind::Auto`] routing.
    #[inline]
    pub fn backend(mut self, kind: BackendKind) -> Self {
        self.kind = kind;
        self
    }

    /// Reject a route that could return an approximate answer, rather than
    /// taking it and saying so in the result.
    ///
    /// [`BackendKind::Auto`] sends a circuit past the statevector cap to an MPS
    /// at a bounded bond dimension, which is the only route those circuits have;
    /// the result reports [`Exactness::Approximate`] either way. Call this when
    /// an approximate answer is worse than no answer, and the run returns
    /// `IncompatibleBackend` naming the engine it would have used.
    ///
    /// Routes that can be decided from the circuit are rejected before any state
    /// is allocated; sparse Pauli dynamics only learns that it truncated while
    /// propagating, so that one is caught on the finished result instead.
    #[inline]
    pub fn require_exact(mut self) -> Self {
        self.require_exact = true;
        self
    }

    /// Attach a noise model.
    ///
    /// [`Simulate::shots`] and [`Simulate::sample_counts`] accept one on any
    /// backend with a per-shot pure state, averaging trajectories.
    /// [`Simulate::run`], [`Simulate::marginals`], and
    /// [`Simulate::expectation_values`] answer from the exact mixture instead,
    /// which only [`BackendKind::DensityMatrix`] holds, so they require that
    /// backend. [`Simulate::expectation_gradient`] rejects a noise model on
    /// every backend.
    #[inline]
    pub fn noise(mut self, model: &'c noise::NoiseModel) -> Self {
        self.noise_model = Some(model);
        self
    }

    /// Start from `amplitudes` instead of |0...0⟩.
    ///
    /// Indexed with qubit 0 in the least significant bit, length `2^n` for the
    /// circuit's `n` qubits, and normalized. A vector failing any of those is
    /// rejected with `InvalidParameter` before the run.
    ///
    /// A start state also constrains the route, because shape-based dispatch
    /// reads the circuit alone and its shortcuts hold only from |0...0⟩:
    /// [`BackendKind::Auto`] resolves to the statevector, and every backend
    /// other than the statevector and [`BackendKind::DensityMatrix`] reports
    /// `IncompatibleBackend`. [`Simulate::expectation_gradient`] declines a start
    /// state, as do [`Simulate::shots`] and [`Simulate::sample_counts`] with a
    /// noise model attached, since trajectory replay has no start-state path.
    #[inline]
    pub fn initial_state(mut self, amplitudes: &'c [Complex64]) -> Self {
        self.initial_state = Some(amplitudes);
        self
    }

    /// Shortcut for [`Simulate::backend`] with [`BackendKind::StatevectorGpu`].
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu(self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.backend(BackendKind::StatevectorGpu { context })
    }

    /// Automatic backend selection with GPU acceleration opted in via `context`.
    ///
    /// Routes like [`BackendKind::Auto`], but a selected statevector or
    /// stabilizer block that clears the qubit crossover with VRAM to spare runs
    /// on the device. Unsupported cases fall back to the identical CPU path.
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_auto(self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.backend(BackendKind::AutoGpu { context })
    }

    /// Distribute the exact state vector across the ranks of `context`.
    ///
    /// With a single rank this behaves like [`Simulate::backend`] with
    /// [`BackendKind::Statevector`].
    #[cfg(feature = "distributed")]
    pub fn distributed(
        self,
        context: std::sync::Arc<crate::distributed::DistributedContext>,
    ) -> Self {
        self.backend(BackendKind::StatevectorDistributed { context })
    }
}

impl<'c> Simulate<'c, Unseeded> {
    /// Query methods exist only on the seeded builder.
    #[inline]
    pub fn seed(self, seed: u64) -> Simulate<'c, Seeded> {
        Simulate {
            circuit: self.circuit,
            kind: self.kind,
            seed: Seeded { seed },
            noise_model: self.noise_model,
            initial_state: self.initial_state,
            require_exact: self.require_exact,
        }
    }
}

impl<'c> Simulate<'c, Seeded> {
    #[inline]
    fn seed_value(&self) -> u64 {
        self.seed.seed
    }

    /// Trajectory replay reinitializes a pure state per shot and the compiled
    /// noisy samplers are tableau based, so neither carries a start state.
    fn require_no_initial_state_under_noise(&self, terminal: &str) -> Result<()> {
        if self.initial_state.is_some() {
            return Err(reject_initial_state(
                &self.kind,
                terminal,
                "noisy trajectory replay starts every shot from |0...0>; read the exact mixture \
                 with `run`, `marginals`, or `expectation_values` on the density-matrix backend",
            ));
        }
        Ok(())
    }

    /// Execute the circuit once.
    ///
    /// With a noise model attached the probabilities are the exact noisy
    /// distribution rather than one trajectory, so the run needs the
    /// density-matrix backend; the classical bits are one draw from that
    /// distribution, matching `shots(1)`.
    #[inline]
    pub fn run(self) -> Result<RunOutcome> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "a single run")?;
            let probabilities =
                exact_noisy_probabilities(self.circuit, noise_model, self.initial_state, seed)?;
            let classical_bits =
                sample_exact_noisy_shots(&probabilities, self.circuit, noise_model, 1, seed)
                    .swap_remove(0);
            return Ok(RunOutcome {
                classical_bits,
                probabilities: Some(probabilities),
                metadata: RunMetadata::exact(ResolvedBackend::DensityMatrix),
            });
        }
        if let Some(state) = self.initial_state {
            return run_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                seed,
                &SimOptions::default(),
            );
        }
        let outcome = run_with_internal(self.kind, self.circuit, seed, SimOptions::default())?;
        ensure_exact_result(self.require_exact, &outcome.metadata)?;
        Ok(outcome)
    }

    /// Execute `num_shots` times, collecting per-shot classical bits. Accepts
    /// an attached noise model.
    #[inline]
    pub fn shots(self, num_shots: usize) -> Result<ShotsResult> {
        let seed = self.seed_value();
        let require_exact = self.require_exact;
        if require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        let result = if let Some(noise_model) = self.noise_model {
            self.require_no_initial_state_under_noise("shot sampling")?;
            run_shots_with_noise(self.kind, self.circuit, noise_model, num_shots, seed)?
        } else if let Some(state) = self.initial_state {
            shots_from_initial_state(&self.kind, self.circuit, state, num_shots, seed)?
        } else {
            run_shots_with(self.kind, self.circuit, num_shots, seed)?
        };
        ensure_exact_result(require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Sample a frequency histogram over `num_shots` executions. Accepts an
    /// attached noise model.
    ///
    /// Counts may be sampled directly from the output distribution, so seeded
    /// counts can differ from [`Simulate::shots`] plus [`ShotsResult::counts`]
    /// while drawing from the identical distribution.
    #[inline]
    pub fn sample_counts(self, num_shots: usize) -> Result<CountsResult> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        let (counts, metadata) = if let Some(noise_model) = self.noise_model {
            self.require_no_initial_state_under_noise("count sampling")?;
            let shots =
                run_shots_with_noise(self.kind, self.circuit, noise_model, num_shots, seed)?;
            (shots.counts(), shots.metadata)
        } else if let Some(state) = self.initial_state {
            let shots = shots_from_initial_state(&self.kind, self.circuit, state, num_shots, seed)?;
            (shots.counts(), shots.metadata)
        } else {
            run_counts_with(self.kind, self.circuit, num_shots, seed)?
        };
        ensure_exact_result(self.require_exact, &metadata)?;
        Ok(CountsResult {
            counts,
            num_classical_bits: self.circuit.num_classical_bits,
            metadata,
        })
    }

    /// Per-qubit marginal probabilities as `(P(0), P(1))` pairs. Rejects
    /// backends without probability output, and with a noise model attached
    /// answers exactly from the mixture, which needs the density-matrix
    /// backend.
    #[inline]
    pub fn marginals(self) -> Result<MarginalsResult> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "marginals")?;
            let probs =
                exact_noisy_probabilities(self.circuit, noise_model, self.initial_state, seed)?;
            return Ok(MarginalsResult {
                marginals: probs.marginals(),
                metadata: RunMetadata::exact(ResolvedBackend::DensityMatrix),
            });
        }
        let result = if let Some(state) = self.initial_state {
            marginals_from_initial_state(&self.kind, self.circuit, state, seed)?
        } else {
            run_marginals_result_with(self.kind, self.circuit, seed)?
        };
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on the circuit's
    /// output state, honoring the selected backend.
    ///
    /// Each observable is a product of single-qubit Paulis (identity factors
    /// omitted). The circuit must be unitary. Clifford circuits propagate each
    /// observable exactly. Non-Clifford circuits use the state vector while they
    /// fit it; above that cap the selected backend evaluates the observable on
    /// its own representation, and a backend without one reports
    /// `BackendUnsupported` naming itself.
    ///
    /// With a noise model attached the value is the exact `Tr(rho P)` on the
    /// evolved mixture, which needs the density-matrix backend.
    #[inline]
    pub fn expectation_values(self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        self.expectation_values_reported(observables)
            .map(ExpectationResult::into_values)
    }

    /// [`Simulate::expectation_values`] with the provenance of the run and, for
    /// a route that estimates rather than evaluates, a standard error per value.
    pub fn expectation_values_reported(
        self,
        observables: &[Vec<PauliTerm>],
    ) -> Result<ExpectationResult> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "expectation values")?;
            require_unitary_circuit(&self.kind, self.circuit)?;
            let values = noise::dm_expectation_values(
                self.circuit,
                observables,
                Some(noise_model),
                self.initial_state,
                seed,
            )?;
            return Ok(analytic_expectations(
                values,
                RunMetadata::exact(ResolvedBackend::DensityMatrix),
            ));
        }
        if let Some(state) = self.initial_state {
            require_unitary_circuit(&self.kind, self.circuit)?;
            return expectation_values_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                observables,
                seed,
            );
        }
        let result = run_expectation_values_reported(self.kind, self.circuit, observables, seed)?;
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Compute `⟨H⟩` and its grouped-measurement variance for a weighted
    /// Pauli observable on the circuit's output state.
    ///
    /// The statevector family evaluates one traversal per qubit-wise-commuting
    /// group and reports the variance; see [`ObservableExpectation::variance`]
    /// for what the number means. Every other route, including runs with a
    /// noise model or start state attached, evaluates term by term through
    /// [`Simulate::expectation_values`] semantics and reports the weighted
    /// mean with no variance.
    pub fn observable_expectation(
        self,
        observable: &PauliObservable,
    ) -> Result<ObservableExpectation> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "expectation values")?;
            require_unitary_circuit(&self.kind, self.circuit)?;
            let values = noise::dm_expectation_values(
                self.circuit,
                &observable_vecs(observable),
                Some(noise_model),
                self.initial_state,
                seed,
            )?;
            return Ok(weighted_observable_result(
                observable,
                &values,
                None,
                RunMetadata::exact(ResolvedBackend::DensityMatrix),
            ));
        }
        if let Some(state) = self.initial_state {
            require_unitary_circuit(&self.kind, self.circuit)?;
            let result = expectation_values_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                &observable_vecs(observable),
                seed,
            )?;
            return Ok(weighted_observable_result(
                observable,
                &result.values,
                result.std_errors.as_deref(),
                result.metadata,
            ));
        }
        let result =
            run_observable_expectation_reported(self.kind, self.circuit, observable, seed)?;
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Compute `⟨H⟩` and its exact gradient with respect to the bound
    /// parameters using the adjoint method.
    ///
    /// `hamiltonian` is a weighted Pauli sum `Σ c_k P_k` with real
    /// coefficients. `params` declares which gate instructions carry parameters.
    /// Runs on the statevector backend; the selected backend must be `Auto` or
    /// `Statevector`. The circuit must be unitary. See
    /// [`gradient::run_expectation_gradient`].
    #[inline]
    pub fn expectation_gradient(
        self,
        hamiltonian: &[(f64, Vec<PauliTerm>)],
        params: &crate::circuit::Parameters,
    ) -> Result<gradient::ExpectationGradient> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if self.noise_model.is_some() {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: "the adjoint method backpropagates through a pure state, so no backend \
                         has a noisy gradient path; drop the noise model, or differentiate noisy \
                         `expectation_values` numerically"
                    .into(),
            });
        }
        if self.initial_state.is_some() {
            return Err(reject_initial_state(
                &self.kind,
                "the adjoint gradient",
                "the backward pass reconstructs the input register by inverting the circuit from \
                 |0...0>, so a start state would have to be inverted with it",
            ));
        }
        if !(self.kind.is_auto() || matches!(self.kind, BackendKind::Statevector)) {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason:
                    "adjoint gradients run on the statevector backend; select Auto or Statevector"
                        .into(),
            });
        }
        gradient::run_expectation_gradient(self.circuit, hamiltonian, params, seed)
    }

    /// Compute `⟨H⟩` and its gradient by the parameter-shift rule on the
    /// selected backend.
    ///
    /// Serves the cases [`Simulate::expectation_gradient`] declines: any
    /// backend with a native observable path, circuits containing `QftBlock`,
    /// and widths past the statevector cap. It differentiates the same gate set
    /// (`Rx`, `Ry`, `Rz`, `Rzz`, `P`) at `1 + 2 * links` circuit evaluations
    /// against the adjoint's one, so the adjoint stays the better choice where
    /// it applies. A backend with no native observable path reports
    /// `BackendUnsupported` naming itself. See
    /// [`gradient::run_expectation_gradient_shift`].
    #[inline]
    pub fn expectation_gradient_shift(
        self,
        hamiltonian: &[(f64, Vec<PauliTerm>)],
        params: &crate::circuit::Parameters,
    ) -> Result<gradient::ExpectationGradient> {
        let seed = self.seed_value();
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit)?;
        }
        if self.noise_model.is_some() {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: "noisy parameter-shift gradients are not wired; drop the noise model, or \
                         differentiate noisy `expectation_values` numerically"
                    .into(),
            });
        }
        gradient::shift_gradient(
            &self.kind,
            self.circuit,
            hamiltonian,
            params,
            self.initial_state,
            seed,
        )
    }
}

/// Start a query-aware simulation request for `circuit`.
///
/// The returned builder defaults to automatic backend selection; chain
/// [`Simulate::seed`] to unlock the query methods.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, simulate};
///
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
///
/// let result = simulate(&circuit).seed(42).run()?;
/// let probs = result.probabilities.expect("no probabilities").to_vec();
/// // Bell state: ~50% |00>, ~50% |11>
/// assert!((probs[0] - 0.5).abs() < 1e-10);
/// assert!((probs[3] - 0.5).abs() < 1e-10);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
#[inline]
pub fn simulate(circuit: &Circuit) -> Simulate<'_, Unseeded> {
    Simulate {
        circuit,
        kind: BackendKind::Auto,
        seed: Unseeded,
        noise_model: None,
        initial_state: None,
        require_exact: false,
    }
}

/// Gate for the terminals that answer a noise model from the exact mixture,
/// which only the density matrix holds.
fn require_exact_mixture(kind: &BackendKind, terminal: &str) -> Result<()> {
    if matches!(kind, BackendKind::DensityMatrix) {
        return Ok(());
    }
    Err(PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason: format!(
            "{terminal} under a noise model reads the exact mixed state, which only the \
             density-matrix backend holds; select it, or average trajectories through `shots` \
             or `sample_counts`"
        ),
    })
}

/// Gate for the terminals that cannot carry a start state through their own
/// machinery.
fn reject_initial_state(kind: &BackendKind, terminal: &str, instead: &str) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason: format!("{terminal} does not accept a start state; {instead}"),
    }
}

/// Reject a start state whose width disagrees with the circuit's.
///
/// The circuit's declared register wins: the amplitude vector sets the backend's
/// width, so a shorter or longer one would silently simulate a different
/// register than the one the instructions index. A register too wide to index
/// with a `usize` has no dense start state at all.
fn check_initial_state_len(state: &[Complex64], num_qubits: usize) -> Result<()> {
    let want = (num_qubits < usize::BITS as usize).then(|| 1usize << num_qubits);
    if want == Some(state.len()) {
        return Ok(());
    }
    let needs = match want {
        Some(count) => count.to_string(),
        None => format!("2^{num_qubits}"),
    };
    Err(PrismError::InvalidParameter {
        message: format!(
            "start state has {} amplitudes, but a {num_qubits}-qubit circuit needs {needs}",
            state.len()
        ),
    })
}

/// Build the constrained backend for a start state and load it.
fn backend_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: &[Complex64],
    seed: u64,
) -> Result<Box<dyn Backend>> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit)?;
    }
    check_initial_state_len(state, circuit.num_qubits)?;
    let mut backend = initial_state_plan(kind, circuit.num_qubits)?.build(seed);
    backend.init_from_amplitudes(state.to_vec(), circuit.num_classical_bits)?;
    Ok(backend)
}

/// Fuse `circuit` for `backend` and apply it, leaving initialization to the
/// caller. The start-state analogue of [`execute`], which owns the |0...0⟩ init.
fn apply_fused_circuit(backend: &mut dyn Backend, circuit: &Circuit) -> Result<()> {
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.apply_instructions(&fused.instructions)
}

fn run_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: &[Complex64],
    seed: u64,
    opts: &SimOptions,
) -> Result<RunOutcome> {
    let mut backend = backend_from_initial_state(kind, circuit, state, seed)?;
    apply_fused_circuit(&mut *backend, circuit)?;

    let probabilities = if opts.probabilities {
        try_backend_probabilities(&*backend)?
    } else {
        None
    };
    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
        metadata: backend_metadata(&*backend),
    })
}

/// Shots for a start state. Terminal measurements sample one evolved
/// distribution; anything else replays the start state per shot, which is what
/// mid-circuit collapse and classical feedback need.
fn shots_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: &[Complex64],
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let bits = circuit.num_classical_bits;
    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let outcome = run_from_initial_state(kind, &stripped, state, seed, &SimOptions::default())?;
        if let Some(probs) = outcome.probabilities {
            let meas_map = circuit.measurement_map();
            return Ok(ShotsResult::from_shots(
                sample_shots(&probs, &meas_map, bits, num_shots, seed),
                bits,
            )
            .with_metadata(outcome.metadata));
        }
    }

    // Plan, expansion, and fusion are shot independent, so they are hoisted out
    // of the replay loop, matching `run_shots_per_shot`.
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit)?;
    }
    check_initial_state_len(state, circuit.num_qubits)?;
    let plan = initial_state_plan(kind, circuit.num_qubits)?;
    let probe = plan.build(seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if probe.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, probe.supports_fused_gates());

    collect_shots(circuit, num_shots, seed, plan.resolved(), |shot_seed| {
        let mut backend = plan.build(shot_seed);
        backend.init_from_amplitudes(state.to_vec(), circuit.num_classical_bits)?;
        backend.apply_instructions(&fused.instructions)?;
        Ok((
            backend.classical_results().to_vec(),
            backend_metadata(&*backend),
        ))
    })
}

fn marginals_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: &[Complex64],
    seed: u64,
) -> Result<MarginalsResult> {
    let mut backend = backend_from_initial_state(kind, circuit, state, seed)?;
    apply_fused_circuit(&mut *backend, circuit)?;
    if backend.supports_pauli_expectation() {
        return marginals_from_pauli_expectations(&*backend, circuit.num_qubits);
    }
    Ok(MarginalsResult {
        marginals: Probabilities::Dense(backend.probabilities()?).marginals(),
        metadata: backend_metadata(&*backend),
    })
}

fn expectation_values_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: &[Complex64],
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<ExpectationResult> {
    // Before the run, matching the dense statevector path.
    let masks = observables
        .iter()
        .map(|obs| pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;

    let mut backend = backend_from_initial_state(kind, circuit, state, seed)?;
    apply_fused_circuit(&mut *backend, circuit)?;
    let metadata = backend_metadata(&*backend);
    if backend.supports_pauli_expectation() {
        let values = backend.pauli_expectations(observables)?;
        return Ok(analytic_expectations(values, metadata));
    }

    let evolved = backend.export_statevector()?;
    let norm = crate::backend::state_norm_sqr(&evolved);
    let values = pauli_expectations_from_masks(&evolved, &masks, norm);
    Ok(analytic_expectations(values, metadata))
}

/// Reject a result that turned out approximate, for a caller that opted out.
///
/// The backstop behind [`reject_approximate_route`], which decides from the
/// circuit alone. Sparse Pauli dynamics truncates on coefficient magnitudes it
/// only learns while propagating, so whether it stayed exact is not knowable
/// before the run.
fn ensure_exact_result(require_exact: bool, metadata: &RunMetadata) -> Result<()> {
    if require_exact && !metadata.is_exact() {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{:?}", metadata.backend),
            reason: "require_exact rejects an approximate result; this engine discarded state                      weight while running, which the route could not predict"
                .into(),
        });
    }
    Ok(())
}

/// Reject an approximate route for a caller that opted out of one. The engine
/// is named in the error, since a caller who asked for exactness wants to know
/// which one would have answered.
fn reject_approximate_route(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    match approximate_route_name(kind, circuit) {
        Some(engine) => Err(PrismError::IncompatibleBackend {
            backend: engine.into(),
            reason: "require_exact rejects a route that can discard state weight; drop the \
                     requirement to accept the approximation, which the result reports, or \
                     select a backend that represents this circuit exactly"
                .into(),
        }),
        None => Ok(()),
    }
}

fn require_unitary_circuit(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "expectation values require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }
    Ok(())
}

/// Exact output distribution of `circuit` under `noise_model`, evolved once on
/// the density matrix. The mixture carries every measurement branch at once,
/// so it answers for a whole shot only when the measurements are terminal.
fn exact_noisy_probabilities(
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    initial_state: Option<&[Complex64]>,
    seed: u64,
) -> Result<Probabilities> {
    Ok(Probabilities::Dense(noise::density_matrix_probabilities(
        circuit,
        noise_model,
        initial_state,
        seed,
    )?))
}

/// Draw classical bits from the exact noisy distribution, then apply readout
/// error. Readout acts on the outcome rather than on the state, so it is not
/// carried by the distribution and has to be applied per draw, on a stream of
/// its own so it does not track the state draws.
fn sample_exact_noisy_shots(
    probs: &Probabilities,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    let bits = circuit.num_classical_bits;
    let mut shots = sample_shots(probs, &circuit.measurement_map(), bits, num_shots, seed);
    if noise_model.readout.iter().any(Option::is_some) {
        let mut rng = trajectory::noise_rng(seed);
        for shot in &mut shots {
            trajectory::apply_readout_errors(shot, &noise_model.readout, &mut rng);
        }
    }
    shots
}

#[inline]
fn probs_only_result(probs: Vec<f64>, metadata: RunMetadata) -> RunOutcome {
    RunOutcome {
        probabilities: Some(Probabilities::Dense(probs)),
        classical_bits: vec![],
        metadata,
    }
}

fn try_backend_probabilities(backend: &dyn Backend) -> Result<Option<Probabilities>> {
    if let Some(factored) = backend.block_probabilities() {
        return Ok(Some(factored));
    }
    match backend.probabilities() {
        Ok(probs) => Ok(Some(Probabilities::Dense(probs))),
        Err(PrismError::BackendUnsupported { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Core execution: fuse, init, apply, extract.
fn execute(backend: &mut dyn Backend, circuit: &Circuit, opts: &SimOptions) -> Result<RunOutcome> {
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    execute_circuit(backend, &fused, opts)
}

/// The execution route a template settles on, for a caller that holds one
/// circuit across many bindings and fuses it itself.
///
/// Resolving the route from the template rather than from a bound circuit
/// matters twice over: it is the work being amortized, and a fused stream
/// misreports the circuit to dispatch (a fused Clifford circuit no longer
/// looks Clifford).
pub(crate) struct PreparedRoute {
    plan: BackendPlan,
    supports_fused: bool,
    /// Backend held across points. `init` reuses its state buffer when the
    /// width matches, so a sweep pays one `2^n` allocation rather than one per
    /// point. Rebuilt when the seed changes, since the seed feeds its RNG.
    held: Option<(u64, Box<dyn Backend>)>,
}

impl PreparedRoute {
    /// True when the chosen backend accepts fused gates, so the caller should
    /// hand `run` a fused stream rather than the bound template.
    pub(crate) fn supports_fused(&self) -> bool {
        self.supports_fused
    }

    /// Apply `circuit` verbatim, with no further fusion.
    pub(crate) fn run(&mut self, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
        if !matches!(&self.held, Some((s, _)) if *s == seed) {
            self.held = Some((seed, self.plan.build(seed)));
        }
        let (_, backend) = self.held.as_mut().expect("just built");
        execute_circuit(&mut **backend, circuit, &SimOptions::default())
    }
}

/// Settle the route for `template`, or `None` when it takes one that reshapes
/// execution (decomposition, stabilizer rank, temporal Clifford) or expands
/// `QftBlock`, where a caller-supplied stream has nowhere to go.
pub(crate) fn prepared_route(kind: &BackendKind, template: &Circuit) -> Option<PreparedRoute> {
    if !kind.is_auto() && validate_explicit_backend(kind, template).is_err() {
        return None;
    }
    let ProbabilityRoute::Direct {
        has_partial_independence,
    } = plan_probability_route(kind, template)
    else {
        return None;
    };
    let ExecutionPlan::Backend(plan) = resolve(kind, template, has_partial_independence) else {
        return None;
    };
    let probe = plan.build(0);
    let has_qft_block = template.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: crate::gates::Gate::QftBlock { .. },
                ..
            }
        )
    });
    if has_qft_block && !probe.supports_qft_block() {
        return None;
    }
    Some(PreparedRoute {
        supports_fused: probe.supports_fused_gates(),
        plan,
        held: None,
    })
}

/// Shared init → apply → extract logic.
fn execute_circuit(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<RunOutcome> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    backend.apply_instructions(&circuit.instructions)?;

    let probabilities = if opts.probabilities {
        try_backend_probabilities(backend)?
    } else {
        None
    };

    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
        metadata: backend_metadata(backend),
    })
}

/// Provenance read off the engine that ran, after it ran. Exactness and
/// placement are reports rather than predictions: the MPS bound reflects what
/// this run discarded, and the placement reflects where the amplitudes ended up
/// after any device fallback.
pub(crate) fn backend_metadata(backend: &dyn Backend) -> RunMetadata {
    RunMetadata::new(backend.resolved(), backend.exactness(), backend.placement())
}

#[cfg(test)]
fn run(circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    run_with(BackendKind::Auto, circuit, seed)
}

/// Constructs the backend internally based on [`BackendKind`], then runs
/// the circuit. For a pre-constructed backend instance, use [`run_on`].
pub(crate) fn run_with(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    run_with_internal(kind, circuit, seed, SimOptions::default())
}

fn run_with_internal(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
) -> Result<RunOutcome> {
    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }
    // The distributed backend runs the whole circuit across ranks in lockstep.
    // Subsystem decomposition, Clifford+T, and temporal-Clifford shortcuts all
    // reshape execution per sub-block, which would desynchronize the collective
    // calls every rank must issue in the same order. Dispatch directly.
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        let mut backend = resolve_backend(&kind, circuit, false).build(seed);
        if opts.probabilities {
            // The gather cap follows from the register width alone, so a
            // register past it is rejected here rather than after the run has
            // been paid for. Same check `probabilities()` applies, so the
            // message does not depend on where it surfaced.
            crate::backend::dense_probability_len(backend.name(), circuit.num_qubits)?;
        }
        return execute(&mut *backend, circuit, &opts);
    }
    match plan_probability_route(&kind, circuit) {
        ProbabilityRoute::FactoredStabilizer => {
            let mut backend =
                crate::backend::factored_stabilizer::FactoredStabilizerBackend::new(seed);
            let fs_opts = if circuit.num_qubits > 64 {
                SimOptions {
                    probabilities: false,
                }
            } else {
                opts
            };
            execute(&mut backend, circuit, &fs_opts)
        }
        ProbabilityRoute::Decomposed(components) => {
            run_decomposed(&kind, &components, circuit, seed, &opts)
        }
        ProbabilityRoute::StabilizerRank { t_count } => {
            let exact = t_count <= MAX_AUTO_T_COUNT_EXACT;
            let sr = if exact {
                stabilizer_rank::run_stabilizer_rank(circuit, seed)?
            } else {
                stabilizer_rank::run_stabilizer_rank_approx(circuit, AUTO_APPROX_MAX_TERMS, seed)?
            };
            let metadata = if exact {
                RunMetadata::exact(ResolvedBackend::StabilizerRank)
            } else {
                RunMetadata::approximate(ResolvedBackend::StabilizerRank)
            };
            Ok(probs_only_result(sr.probabilities, metadata))
        }
        ProbabilityRoute::TemporalClifford(tc) => {
            run_temporal_clifford(&tc, seed, opts.probabilities)
        }
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => match resolve(&kind, circuit, has_partial_independence) {
            ExecutionPlan::Backend(plan) => {
                let mut backend = plan.build(seed);
                execute(&mut *backend, circuit, &opts)
            }
            ExecutionPlan::StabilizerRank => {
                let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
                Ok(probs_only_result(
                    sr.probabilities,
                    RunMetadata::exact(ResolvedBackend::StabilizerRank),
                ))
            }
            ExecutionPlan::StochasticPauli { num_samples } => {
                Err(crate::error::PrismError::IncompatibleBackend {
                    backend: format!(
                        "{:?}",
                        BackendKind::StochasticPauli { num_samples }
                    ),
                    reason: "StochasticPauli produces marginal estimates only; use `simulate(...).marginals()`".into(),
                })
            }
            ExecutionPlan::DeterministicPauli { epsilon, max_terms } => {
                Err(crate::error::PrismError::IncompatibleBackend {
                    backend: format!(
                        "{:?}",
                        BackendKind::DeterministicPauli { epsilon, max_terms }
                    ),
                    reason: "DeterministicPauli produces marginals only; use `simulate(...).marginals()`".into(),
                })
            }
        },
    }
}

/// Execute a circuit on a pre-constructed backend. For automatic dispatch,
/// use [`simulate`].
pub fn run_on(backend: &mut dyn Backend, circuit: &Circuit) -> Result<RunOutcome> {
    execute(backend, circuit, &SimOptions::default())
}

/// Execute a circuit on a pre-constructed backend from a start state other than
/// |0...0⟩, the [`run_on`] sibling for a caller holding its own amplitudes.
///
/// The backend must accept one; see [`Backend::init_from_amplitudes`] for the
/// validation applied to `initial_state` and for which backends decline it.
pub fn run_on_state(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    initial_state: &[Complex64],
) -> Result<RunOutcome> {
    check_initial_state_len(initial_state, circuit.num_qubits)?;
    backend.init_from_amplitudes(initial_state.to_vec(), circuit.num_classical_bits)?;
    apply_fused_circuit(backend, circuit)?;
    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities: try_backend_probabilities(backend)?,
        metadata: backend_metadata(backend),
    })
}

/// Parse an OpenQASM string and execute with automatic backend selection.
pub fn run_qasm(qasm: &str, seed: u64) -> Result<RunOutcome> {
    let circuit = crate::circuit::openqasm::parse(qasm)?;
    simulate(&circuit).seed(seed).run()
}

#[cfg(test)]
fn run_shots(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    run_shots_with(BackendKind::Auto, circuit, num_shots, seed)
}

pub(crate) fn supports_compiled_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && !circuit.has_resets()
        && circuit.has_terminal_measurements_only()
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
}

fn supports_deferred_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && (circuit.has_resets() || !circuit.has_terminal_measurements_only())
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
        && !circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Conditional { .. }))
}

fn is_clifford_sampler_kind(kind: &BackendKind) -> bool {
    if kind.is_auto() {
        return true;
    }
    match kind {
        BackendKind::Stabilizer | BackendKind::FactoredStabilizer => true,
        #[cfg(feature = "gpu")]
        BackendKind::StabilizerGpu { .. } => true,
        _ => false,
    }
}

fn should_use_compiled_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 2
        && supports_compiled_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

fn should_use_deferred_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 2
        && supports_deferred_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

fn compile_measurements_for_kind(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<compiled::CompiledSampler> {
    #[cfg(not(feature = "gpu"))]
    let _ = kind;

    let sampler = compiled::compile_measurements(circuit, seed)?;

    #[cfg(feature = "gpu")]
    if let BackendKind::StabilizerGpu { context } = kind {
        return Ok(sampler.with_gpu(context.clone()));
    }

    Ok(sampler)
}

/// Independence analysis shared by the routing prelude in
/// `run_with_internal`, the shots slow path, and the terminal fast-path
/// candidacy. Returns the components to decompose with when full
/// decomposition should fire, plus the partial-independence flag otherwise.
fn analyze_independence(circuit: &Circuit) -> (Option<Vec<Vec<usize>>>, bool) {
    if circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        let components = circuit.independent_subsystems();
        if components.len() > 1 {
            if should_decompose(&components, circuit.num_qubits) {
                return (Some(components), false);
            }
            return (None, true);
        }
    }
    (None, false)
}

/// `(t_count, stabilizer_rank_budget)` when the auto Clifford+T family gate
/// passes. Callers apply their own per-entry-point T-count ceilings.
fn auto_clifford_t_budget(circuit: &Circuit) -> Option<(usize, usize)> {
    (circuit.is_clifford_plus_t() && circuit.has_t_gates()).then(|| {
        (
            circuit.t_count(),
            stabilizer_rank_budget(circuit.num_qubits),
        )
    })
}

/// Auto-dispatch gate for the Clifford+T stabilizer-rank shortcut: the T
/// count must fit both the caller's ceiling and the size-derived
/// stabilizer-rank budget. Returns the T count when the shortcut applies.
pub(super) fn auto_stabilizer_rank_t_count(circuit: &Circuit, max_t: usize) -> Option<usize> {
    let (t, sr_budget) = auto_clifford_t_budget(circuit)?;
    (t <= max_t && t <= sr_budget).then_some(t)
}

/// Routing precedence for the probability path: decomposition (with the
/// large sparse-Clifford factored-stabilizer override), then the Clifford+T
/// stabilizer-rank shortcut, then temporal Clifford, then direct family
/// resolution. `run_with_internal` executes this plan and
/// `auto_terminal_statevector_candidate` consults it, so the two cannot
/// drift apart.
enum ProbabilityRoute {
    FactoredStabilizer,
    Decomposed(Vec<Vec<usize>>),
    StabilizerRank { t_count: usize },
    TemporalClifford(TemporalCliffordPlan),
    Direct { has_partial_independence: bool },
}

fn plan_probability_route(kind: &BackendKind, circuit: &Circuit) -> ProbabilityRoute {
    let (decompose, has_partial_independence) = analyze_independence(circuit);
    if let Some(components) = decompose {
        let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
        if kind.is_auto()
            && circuit.is_clifford_only()
            && circuit.num_qubits >= MIN_FACTORED_STABILIZER_QUBITS
            && max_block >= MIN_BLOCK_FOR_FACTORED_STAB
        {
            return ProbabilityRoute::FactoredStabilizer;
        }
        return ProbabilityRoute::Decomposed(components);
    }
    if kind.is_auto()
        && circuit.num_qubits <= MAX_STABILIZER_RANK_QUBITS
        && !has_nonunitary_or_classical_ops(circuit)
    {
        if let Some(t_count) = auto_stabilizer_rank_t_count(circuit, MAX_AUTO_T_COUNT_APPROX) {
            return ProbabilityRoute::StabilizerRank { t_count };
        }
    }
    if let Some(tc) = plan_temporal_clifford(kind, circuit) {
        return ProbabilityRoute::TemporalClifford(tc);
    }
    ProbabilityRoute::Direct {
        has_partial_independence,
    }
}

/// True when the auto probability route falls through to direct family
/// resolution and that resolver picks the CPU statevector.
fn auto_terminal_statevector_candidate(circuit: &Circuit) -> bool {
    match plan_probability_route(&BackendKind::Auto, circuit) {
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => auto_selects_cpu_statevector(circuit, has_partial_independence),
        _ => false,
    }
}

fn terminal_statevector_candidate(kind: &BackendKind, circuit: &Circuit) -> bool {
    if kind.is_auto() {
        return auto_terminal_statevector_candidate(circuit);
    }
    match kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => false,
    }
}

fn try_terminal_statevector_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<Option<TerminalStatevector>> {
    if !circuit.has_terminal_measurements_only() {
        return Ok(None);
    }

    let meas_map = circuit.measurement_map();
    if meas_map.is_empty() {
        return Ok(None);
    }

    let stripped = circuit.without_measurements();
    if !terminal_statevector_candidate(kind, &stripped) {
        return Ok(None);
    }

    let accel = accel_for(kind, Family::Statevector, stripped.num_qubits);
    let mut backend = build_statevector(&accel, seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(&stripped)
    } else {
        crate::circuit::expand_qft_blocks(&stripped)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    backend.apply_instructions(&fused.instructions)?;

    Ok(Some((backend, meas_map)))
}

/// Build and run the backend for a terminal-measurement circuit when routing
/// lands on a single backend that samples from its own representation.
///
/// Returns `None` when the route is not a direct single backend, or when that
/// backend has no native sampler, leaving the dense probability path untouched.
/// The capability is probed before `init`, so a backend without one costs an
/// allocation and nothing else.
///
/// The product state is the one route taken past subsystem decomposition: it
/// already stores one factor per qubit, so splitting the circuit into
/// independent blocks pays a backend, a partition, and a merge per block to
/// rebuild what one native draw reads straight off the state, and above 64
/// qubits the merged block distribution does not exist at all. Every other
/// backend keeps the block split.
fn try_native_terminal_backend(
    kind: &BackendKind,
    stripped: &Circuit,
    seed: u64,
) -> Result<Option<Box<dyn Backend>>> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, stripped)?;
    }
    let (decomposed, has_partial_independence) = match plan_probability_route(kind, stripped) {
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => (false, has_partial_independence),
        ProbabilityRoute::Decomposed(_) => (true, false),
        _ => return Ok(None),
    };
    let ExecutionPlan::Backend(plan) = resolve(kind, stripped, has_partial_independence) else {
        return Ok(None);
    };
    if decomposed && !matches!(plan, BackendPlan::ProductState) {
        return Ok(None);
    }
    let mut backend = plan.build(seed);
    if !backend.supports_native_sampling() {
        return Ok(None);
    }
    execute(&mut *backend, stripped, &SimOptions::classical_only())?;
    Ok(Some(backend))
}

/// Build and run the backend for a marginal query when routing lands on a
/// single backend that evaluates observables on its own representation.
///
/// Returns `None` when the route is not a direct single backend, or when that
/// backend has no native observable path, leaving the dense probability route
/// untouched. The capability is probed before `init`, so a backend without one
/// costs an allocation and nothing else.
///
/// The product state is carried past subsystem decomposition for the reason
/// [`try_native_terminal_backend`] carries it: it already holds one factor per
/// qubit, and the decomposed route would build the `2^n` merged distribution to
/// read marginals a per-qubit expectation answers directly. Every other backend
/// keeps the block split, which has no single backend holding the joint state.
fn try_native_marginal_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<Option<Box<dyn Backend>>> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit)?;
    }
    let (decomposed, has_partial_independence) = match plan_probability_route(kind, circuit) {
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => (false, has_partial_independence),
        ProbabilityRoute::Decomposed(_) => (true, false),
        _ => return Ok(None),
    };
    let ExecutionPlan::Backend(plan) = resolve(kind, circuit, has_partial_independence) else {
        return Ok(None);
    };
    if decomposed && !matches!(plan, BackendPlan::ProductState) {
        return Ok(None);
    }
    let mut backend = plan.build(seed);
    if !backend.supports_pauli_expectation() {
        return Ok(None);
    }
    execute(&mut *backend, circuit, &SimOptions::classical_only())?;
    Ok(Some(backend))
}

/// A source that answers every measurement in a circuit from a single state,
/// in the precedence [`prepare_shot_source`] applies. Shots and counts both
/// consume it, so the two cannot disagree about which shortcut applies.
enum ShotSource {
    /// `deferred` marks a sampler built from the measure/reset-deferred
    /// rewrite, whose renumbered measurements only read through `meas_map`.
    Compiled {
        sampler: Box<compiled::CompiledSampler>,
        meas_map: Vec<(usize, usize)>,
        deferred: bool,
    },
    TerminalStatevector {
        backend: Box<StatevectorBackend>,
        meas_map: Vec<(usize, usize)>,
    },
    /// A backend that draws basis states from its own representation.
    Native {
        backend: Box<dyn Backend>,
        meas_map: Vec<(usize, usize)>,
    },
    /// Dense output distribution of the measurement-stripped circuit, with the
    /// provenance of the run that produced it.
    TerminalProbabilities {
        probs: Probabilities,
        meas_map: Vec<(usize, usize)>,
        metadata: RunMetadata,
    },
    StabilizerRank,
    PerShot,
}

impl ShotSource {
    /// Which engine will answer, decided by `prepare_shot_source` and read here
    /// so the shot and count entry points do not re-derive it. `None` for the
    /// per-shot route, which builds one backend per shot and stamps its own.
    fn metadata(&self) -> Option<RunMetadata> {
        match self {
            ShotSource::Compiled { .. } => {
                Some(RunMetadata::exact(ResolvedBackend::CompiledStabilizer))
            }
            ShotSource::TerminalStatevector { backend, .. } => Some(backend_metadata(&**backend)),
            ShotSource::Native { backend, .. } => Some(backend_metadata(&**backend)),
            ShotSource::TerminalProbabilities { metadata, .. } => Some(metadata.clone()),
            ShotSource::StabilizerRank => Some(RunMetadata::exact(ResolvedBackend::StabilizerRank)),
            ShotSource::PerShot => None,
        }
    }
}

/// Select and prepare the sampling source for `circuit`.
///
/// Preparation is real work: compiling a sampler, building and running a
/// backend, or executing the stripped circuit once. Call once per entry point
/// and match on the result rather than re-deriving the choice.
fn prepare_shot_source(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotSource> {
    if should_use_compiled_clifford_sampling(kind, circuit, num_shots) {
        return Ok(ShotSource::Compiled {
            sampler: Box::new(compile_measurements_for_kind(kind, circuit, seed)?),
            meas_map: circuit.measurement_map(),
            deferred: false,
        });
    }

    if should_use_deferred_clifford_sampling(kind, circuit, num_shots) {
        if let Ok(deferred) = compiled::defer_measure_reset_circuit(circuit) {
            return Ok(ShotSource::Compiled {
                sampler: Box::new(compile_measurements_for_kind(kind, &deferred, seed)?),
                meas_map: deferred.measurement_map(),
                deferred: true,
            });
        }
    }

    if let Some((backend, meas_map)) = try_terminal_statevector_backend(kind, circuit, seed)? {
        return Ok(ShotSource::TerminalStatevector {
            backend: Box::new(backend),
            meas_map,
        });
    }

    if matches!(kind, BackendKind::StabilizerRank) && circuit.has_t_gates() {
        return Ok(ShotSource::StabilizerRank);
    }
    if kind.is_auto()
        && circuit.has_terminal_measurements_only()
        && circuit.num_qubits > MAX_STABILIZER_RANK_QUBITS
        && auto_stabilizer_rank_t_count(circuit, MAX_AUTO_T_COUNT_SHOTS).is_some()
    {
        return Ok(ShotSource::StabilizerRank);
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        if let Some(backend) = try_native_terminal_backend(kind, &stripped, seed)? {
            return Ok(ShotSource::Native {
                backend,
                meas_map: circuit.measurement_map(),
            });
        }
        let result = run_with_internal(kind.clone(), &stripped, seed, SimOptions::default())?;
        if let Some(probs) = result.probabilities {
            return Ok(ShotSource::TerminalProbabilities {
                probs,
                meas_map: circuit.measurement_map(),
                metadata: result.metadata,
            });
        }
    }

    Ok(ShotSource::PerShot)
}

#[cfg(test)]
fn run_counts(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<HashMap<Vec<u64>, u64>> {
    run_counts_with(BackendKind::Auto, circuit, num_shots, seed).map(|(counts, _)| counts)
}

/// Execute a circuit multiple times with explicit backend selection and return counts.
///
/// For Clifford circuits with terminal measurements and no resets, Auto,
/// Stabilizer, FactoredStabilizer, and explicit `StabilizerGpu` route through
/// the compiled sampler's optimized counting path. Explicit `StabilizerGpu`
/// carries its GPU context into the compiled sampler so large shot runs avoid
/// the raw tableau measurement round-trips. Other circuits fall back to
/// per-shot simulation with counting.
///
/// Optimized terminal statevector paths sample counts directly from the output
/// distribution. The distribution is equivalent to materializing shots first,
/// but finite seeded counts may differ from `run_shots_with(...).counts()`.
pub(crate) fn run_counts_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<(HashMap<Vec<u64>, u64>, RunMetadata)> {
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        let shots = run_shots_with(kind, circuit, num_shots, seed)?;
        return Ok((shots.counts(), shots.metadata));
    }

    let bits = circuit.num_classical_bits;
    let source = prepare_shot_source(&kind, circuit, num_shots, seed)?;
    let Some(metadata) = source.metadata() else {
        let shots = run_shots_per_shot(kind, circuit, num_shots, seed)?;
        return Ok((shots.counts(), shots.metadata));
    };
    let counts = match source {
        ShotSource::Compiled {
            mut sampler,
            meas_map,
            deferred,
        } => {
            if deferred {
                let packed = sampler.try_sample_bulk_packed(num_shots)?;
                counts_of(
                    packed_shots_to_classical_bits(&packed, &meas_map, bits),
                    bits,
                )
            } else {
                sampler.try_sample_counts(num_shots)?
            }
        }
        ShotSource::TerminalStatevector { backend, meas_map } => {
            if backend.is_gpu_resident() {
                let probs = backend.probabilities()?;
                sample_counts_from_probs(&probs, &meas_map, bits, num_shots, seed)
            } else {
                sample_counts_from_state(
                    backend.state_vector(),
                    backend.probability_scale(),
                    &meas_map,
                    bits,
                    num_shots,
                    seed,
                )
            }
        }
        ShotSource::Native {
            mut backend,
            meas_map,
        } => {
            let samples = backend.sample_basis_states(num_shots, seed)?;
            counts_of(shots_from_basis_samples(&samples, &meas_map, bits), bits)
        }
        ShotSource::TerminalProbabilities {
            probs, meas_map, ..
        } => counts_of(sample_shots(&probs, &meas_map, bits, num_shots, seed), bits),
        ShotSource::StabilizerRank => {
            stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed)?.counts()
        }
        ShotSource::PerShot => unreachable!("handled above"),
    };
    Ok((counts, metadata.with_shots(num_shots)))
}

fn counts_of(shots: Vec<Vec<bool>>, num_classical_bits: usize) -> HashMap<Vec<u64>, u64> {
    ShotsResult::from_shots(shots, num_classical_bits).counts()
}

#[cfg(test)]
fn run_marginals(circuit: &Circuit, seed: u64) -> Result<Vec<(f64, f64)>> {
    run_marginals_result_with(BackendKind::Auto, circuit, seed).map(MarginalsResult::into_vec)
}

#[cfg(test)]
fn run_marginals_with(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<Vec<(f64, f64)>> {
    run_marginals_result_with(kind, circuit, seed).map(MarginalsResult::into_vec)
}

/// Per-qubit marginals from native single-qubit Z expectations, for a backend
/// whose own representation answers them without a dense probability vector.
fn marginals_from_pauli_expectations(
    backend: &dyn Backend,
    num_qubits: usize,
) -> Result<MarginalsResult> {
    let observables: Vec<Vec<PauliTerm>> = (0..num_qubits).map(|q| vec![PauliTerm::z(q)]).collect();
    let expectations = backend.pauli_expectations(&observables)?;
    Ok(MarginalsResult {
        marginals: expectations_to_marginals(&expectations),
        metadata: backend_metadata(backend),
    })
}

/// Sparse Pauli dynamics is exact when it truncated nothing. `total_discarded`
/// is a coefficient magnitude rather than a state overlap, so a truncated run
/// reports approximate with no fidelity bound.
fn spd_metadata(total_discarded: f64) -> RunMetadata {
    if total_discarded == 0.0 {
        RunMetadata::exact(ResolvedBackend::DeterministicPauli)
    } else {
        RunMetadata::approximate(ResolvedBackend::DeterministicPauli)
    }
}

pub(crate) fn expectations_to_marginals(expectations: &[f64]) -> Vec<(f64, f64)> {
    expectations
        .iter()
        .map(|ez| {
            let p0 = ((1.0 + ez) / 2.0).clamp(0.0, 1.0);
            (p0, 1.0 - p0)
        })
        .collect()
}

pub(super) fn has_nonunitary_or_classical_ops(circuit: &Circuit) -> bool {
    circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
        )
    })
}

fn supports_pauli_marginal_backend(circuit: &Circuit) -> bool {
    circuit.is_clifford_plus_t() && !has_nonunitary_or_classical_ops(circuit)
}

fn validate_pauli_marginal_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    if !circuit.is_clifford_plus_t() {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli marginal backends require Clifford+T gates".into(),
        });
    }
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli marginal backends require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }
    Ok(())
}

fn run_marginals_result_with(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<MarginalsResult> {
    let n = circuit.num_qubits;

    match &kind {
        BackendKind::StochasticPauli { num_samples } => {
            validate_pauli_marginal_backend(&kind, circuit)?;
            let spp = unified_pauli::run_spp(circuit, *num_samples, seed)?;
            return Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spp.expectations),
                metadata: RunMetadata::approximate(ResolvedBackend::StochasticPauli)
                    .with_shots(*num_samples),
            });
        }
        BackendKind::DeterministicPauli { epsilon, max_terms } => {
            validate_pauli_marginal_backend(&kind, circuit)?;
            let spd = unified_pauli::run_spd(circuit, *epsilon, *max_terms)?;
            return Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spd.expectations),
                metadata: spd_metadata(spd.total_discarded),
            });
        }
        _ => {}
    }

    if kind.is_auto()
        && supports_pauli_marginal_backend(circuit)
        && circuit.has_t_gates()
        && n >= MIN_QUBITS_FOR_SPD_AUTO
    {
        let spd = unified_pauli::run_spd(circuit, 0.0, AUTO_SPD_MAX_TERMS)?;
        return Ok(MarginalsResult {
            marginals: expectations_to_marginals(&spd.expectations),
            metadata: spd_metadata(spd.total_discarded),
        });
    }

    // The distributed backend answers a marginal from rank-local sums plus one
    // `Allreduce`, so it never needs the 2^n gather the fallback below takes.
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = &kind {
        let mut backend =
            crate::backend::distributed_statevector::DistributedStatevectorBackend::new(
                context.clone(),
                seed,
            );
        execute(&mut backend, circuit, &SimOptions::classical_only())?;
        return marginals_from_pauli_expectations(&backend, n);
    }

    if let Some(backend) = try_native_marginal_backend(&kind, circuit, seed)? {
        return marginals_from_pauli_expectations(&*backend, n);
    }

    let result = run_with(kind, circuit, seed)?;
    if let Some(probs) = &result.probabilities {
        Ok(MarginalsResult {
            marginals: probs.marginals(),
            metadata: result.metadata.clone(),
        })
    } else {
        Err(PrismError::BackendUnsupported {
            backend: "simulate".into(),
            operation: format!(
                "marginals for {} qubits without backend probability output",
                circuit.num_qubits
            ),
        })
    }
}

/// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on a unitary circuit's
/// output state, using automatic backend selection. See
/// [`Simulate::expectation_values`] for explicit backend control.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, PauliTerm, run_expectation_values};
///
/// let mut bell = Circuit::new(2, 0);
/// bell.add_gate(Gate::H, &[0]);
/// bell.add_gate(Gate::Cx, &[0, 1]);
///
/// let observables = vec![
///     vec![PauliTerm::z(0)],
///     vec![PauliTerm::z(0), PauliTerm::z(1)],
/// ];
/// let values = run_expectation_values(&bell, &observables, 42)?;
/// assert!(values[0].abs() < 1e-10); // <Z0> = 0
/// assert!((values[1] - 1.0).abs() < 1e-10); // <Z0 Z1> = 1
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn run_expectation_values(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<Vec<f64>> {
    run_expectation_values_with(BackendKind::Auto, circuit, observables, seed)
}

fn run_expectation_values_with(
    kind: BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<Vec<f64>> {
    run_expectation_values_reported(kind, circuit, observables, seed)
        .map(ExpectationResult::into_values)
}

fn run_expectation_values_reported(
    kind: BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<ExpectationResult> {
    require_unitary_circuit(&kind, circuit)?;

    match &kind {
        BackendKind::StochasticPauli { num_samples } => {
            let mut values = Vec::with_capacity(observables.len());
            let mut std_errors = Vec::with_capacity(observables.len());
            for (i, obs) in observables.iter().enumerate() {
                let r = unified_pauli::run_spp_observable(
                    circuit,
                    obs,
                    *num_samples,
                    seed.wrapping_add(i as u64),
                )?;
                values.push(r.mean);
                std_errors.push(r.std_error);
            }
            Ok(ExpectationResult {
                values,
                std_errors: Some(std_errors),
                metadata: RunMetadata::approximate(ResolvedBackend::StochasticPauli)
                    .with_shots(*num_samples),
            })
        }
        BackendKind::DeterministicPauli { epsilon, max_terms } => {
            let mut values = Vec::with_capacity(observables.len());
            let mut discarded = 0.0;
            for obs in observables {
                let r = unified_pauli::run_spd_observable(circuit, obs, *epsilon, *max_terms)?;
                values.push(r.mean);
                discarded += r.total_discarded;
            }
            Ok(analytic_expectations(values, spd_metadata(discarded)))
        }
        _ if kind.is_auto() || kind.is_stabilizer_family() => {
            if circuit.is_clifford_only() {
                let mut values = Vec::with_capacity(observables.len());
                let mut discarded = 0.0;
                for obs in observables {
                    let r = unified_pauli::run_spd_observable(circuit, obs, 0.0, 0)?;
                    values.push(r.mean);
                    discarded += r.total_discarded;
                }
                Ok(analytic_expectations(values, spd_metadata(discarded)))
            } else if kind.is_auto() {
                if circuit.num_qubits > max_statevector_qubits() {
                    return expectation_values_native(&kind, circuit, observables, seed);
                }
                expectation_values_statevector(&kind, circuit, observables, seed)
            } else {
                Err(PrismError::IncompatibleBackend {
                    backend: format!("{kind:?}"),
                    reason: "stabilizer backends require a Clifford-only circuit".into(),
                })
            }
        }
        BackendKind::Statevector => {
            expectation_values_statevector(&kind, circuit, observables, seed)
        }
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => {
            expectation_values_statevector(&kind, circuit, observables, seed)
        }
        other => expectation_values_native(other, circuit, observables, seed),
    }
}

/// Compute `⟨H⟩` and its grouped-measurement variance for a weighted Pauli
/// observable, using automatic backend selection. See
/// [`Simulate::observable_expectation`] for explicit backend control and
/// [`ObservableExpectation::variance`] for the variance contract.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, PauliObservable, PauliTerm, run_observable_expectation};
///
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::H, &[0]);
/// circuit.add_gate(Gate::Cx, &[0, 1]);
/// circuit.add_gate(Gate::T, &[0]);
///
/// let hamiltonian = PauliObservable::from_terms([
///     (1.0, vec![PauliTerm::z(0)]),
///     (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
/// ])?;
///
/// // The T phase moves nothing here: both terms are Z-only, one commuting
/// // group covers them, and the variance is exactly Var(H) with outcomes
/// // 2 and 0 at probability 1/2 each.
/// let result = run_observable_expectation(&circuit, &hamiltonian, 42)?;
/// assert!((result.mean - 1.0).abs() < 1e-10);
/// assert!((result.variance.unwrap() - 1.0).abs() < 1e-10);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn run_observable_expectation(
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
) -> Result<ObservableExpectation> {
    run_observable_expectation_reported(BackendKind::Auto, circuit, observable, seed)
}

fn run_observable_expectation_reported(
    kind: BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
) -> Result<ObservableExpectation> {
    require_unitary_circuit(&kind, circuit)?;

    let grouped_statevector = match &kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => {
            kind.is_auto()
                && !circuit.is_clifford_only()
                && circuit.num_qubits <= max_statevector_qubits()
        }
    };
    if grouped_statevector {
        return grouped_expectation_statevector(&kind, circuit, observable, seed);
    }

    let result =
        run_expectation_values_reported(kind, circuit, &observable_vecs(observable), seed)?;
    Ok(weighted_observable_result(
        observable,
        &result.values,
        result.std_errors.as_deref(),
        result.metadata,
    ))
}

fn observable_vecs(observable: &PauliObservable) -> Vec<Vec<PauliTerm>> {
    observable
        .terms()
        .iter()
        .map(|(_, factors)| factors.clone())
        .collect()
}

/// Fold per-term values into the weighted mean; no grouped traversal ran, so
/// there is no variance to report. Independent per-term estimates combine
/// into the weighted-sum standard error.
fn weighted_observable_result(
    observable: &PauliObservable,
    values: &[f64],
    std_errors: Option<&[f64]>,
    metadata: RunMetadata,
) -> ObservableExpectation {
    let coefficients = observable.terms().iter().map(|(c, _)| *c);
    let mean = coefficients.clone().zip(values).map(|(c, v)| c * v).sum();
    let std_error = std_errors.map(|errors| {
        coefficients
            .zip(errors)
            .map(|(c, e)| (c * e).powi(2))
            .sum::<f64>()
            .sqrt()
    });
    ObservableExpectation {
        mean,
        variance: None,
        group_variances: None,
        std_error,
        metadata,
    }
}

/// Evaluate a weighted observable on the statevector: the mean and most group
/// variances from one shared batched traversal, large groups from a dedicated
/// moments pass.
///
/// `Var(H_g) = <H_g^2> - <H_g>^2` per commuting group. For a small group the
/// square expands into pairwise product strings appended to the same
/// traversal that serves the term means; a group past the pair budget takes a
/// single-pass moment accumulation instead, on the state as run when the
/// group is Z-only and on a basis-rotated copy otherwise.
fn grouped_expectation_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
) -> Result<ObservableExpectation> {
    let terms = observable.terms();
    // Validate before the 2^n simulation so bad observables fail cheaply.
    let masks = terms
        .iter()
        .map(|(_, factors)| pauli_masks(factors, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;

    let accel = accel_for(kind, Family::Statevector, circuit.num_qubits);
    let mut backend = build_statevector(&accel, seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    backend.apply_instructions(&fused.instructions)?;

    let exported;
    let state: &[Complex64] = if backend.is_gpu_resident() {
        exported = backend.export_statevector()?;
        &exported
    } else {
        backend.state_vector()
    };
    let norm = crate::backend::state_norm_sqr(state);
    let metadata = backend_metadata(&backend);

    let grouping = observable.grouping();

    // Small groups get `<H_g^2>` from pairwise product strings appended to the
    // shared traversal: qubit-wise-commuting strings multiply phase-free
    // (shared qubits carry equal axes and cancel to identity), so each pair is
    // one more mask. Large groups fall back to a dedicated moments pass, which
    // costs a fixed number of state sweeps where the pair expansion grows
    // quadratically.
    let mut combined = masks.clone();
    let mut pair_blocks: Vec<(usize, usize, Vec<f64>)> = Vec::new();
    let mut deferred: Vec<usize> = Vec::new();
    for (gi, group) in grouping.groups.iter().enumerate() {
        let members = &group.term_indices;
        if members.len() * (members.len() - 1) / 2 > MAX_PAIR_MASKS_PER_GROUP {
            deferred.push(gi);
            continue;
        }
        let first_mask = combined.len();
        let mut pair_coefficients = Vec::with_capacity(members.len() * (members.len() - 1) / 2);
        for (pos, &i) in members.iter().enumerate() {
            for &j in &members[pos + 1..] {
                let product_x = masks[i].0 ^ masks[j].0;
                let product_z = masks[i].1 ^ masks[j].1;
                combined.push((product_x, product_z, (product_x & product_z).count_ones()));
                pair_coefficients.push(2.0 * terms[i].0 * terms[j].0);
            }
        }
        pair_blocks.push((gi, first_mask, pair_coefficients));
    }

    let values = pauli_expectations_from_masks(state, &combined, norm);

    let mean: f64 = terms.iter().zip(&values).map(|((c, _), v)| c * v).sum();
    let mut group_variances = vec![0.0; grouping.groups.len()];

    for (gi, first_mask, pair_coefficients) in &pair_blocks {
        let group = &grouping.groups[*gi];
        let m1: f64 = group
            .term_indices
            .iter()
            .map(|&i| terms[i].0 * values[i])
            .sum();
        let square_diag: f64 = group
            .term_indices
            .iter()
            .map(|&i| terms[i].0 * terms[i].0)
            .sum();
        let square_cross: f64 = pair_coefficients
            .iter()
            .zip(&values[*first_mask..])
            .map(|(c, v)| c * v)
            .sum();
        group_variances[*gi] = (square_diag + square_cross - m1 * m1).max(0.0);
    }

    let mut scratch: Option<StatevectorBackend> = None;
    for &gi in &deferred {
        let group = &grouping.groups[gi];
        let coefficients: Vec<f64> = group.term_indices.iter().map(|&i| terms[i].0).collect();
        let (m1, m2) = if group.is_z_only() {
            let zmasks: Vec<usize> = group.term_indices.iter().map(|&i| masks[i].1).collect();
            observable::weighted_group_moments(state, &zmasks, &coefficients, norm)
        } else {
            let zmasks: Vec<usize> = group
                .term_indices
                .iter()
                .map(|&i| masks[i].0 | masks[i].1)
                .collect();
            let rotation_circuit = group.basis_rotation_circuit(circuit.num_qubits);
            let rotation = crate::circuit::fusion::fuse_circuit(&rotation_circuit, true);
            let rotated = scratch.get_or_insert_with(|| StatevectorBackend::new(seed));
            rotated.init_from_amplitudes(state.to_vec(), 0)?;
            rotated.apply_instructions(&rotation.instructions)?;
            observable::weighted_group_moments(rotated.state_vector(), &zmasks, &coefficients, norm)
        };
        group_variances[gi] = (m2 - m1 * m1).max(0.0);
    }

    let variance = group_variances.iter().sum();
    Ok(ObservableExpectation {
        mean,
        variance: Some(variance),
        group_variances: Some(group_variances),
        std_error: None,
        metadata,
    })
}

/// Pair-expansion budget per commuting group. Measured on the 2000-string
/// Jordan-Wigner fixture at n=20: one extra general mask in the shared
/// traversal costs about 0.6 ms while a scratch-rotation moments pass costs
/// about 11 ms, so groups whose pair count stays under that ratio expand
/// inline and larger groups take the dedicated pass.
const MAX_PAIR_MASKS_PER_GROUP: usize = 20;

/// Values from a route that evaluates rather than samples, so there is no
/// interval to report.
fn analytic_expectations(values: Vec<f64>, metadata: RunMetadata) -> ExpectationResult {
    ExpectationResult {
        values,
        std_errors: None,
        metadata,
    }
}

/// Evaluate `observables` on the backend `kind` resolves to, using that
/// backend's own representation.
///
/// Backends without a native Pauli path report `BackendUnsupported` naming
/// themselves, so a request that cannot be served says which engine could not
/// serve it rather than blaming the route that picked it.
fn expectation_values_native(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<ExpectationResult> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit)?;
    }
    // Before the run, matching the statevector path, so a typo in an observable
    // does not cost a 40-qubit simulation first.
    for observable in observables {
        validate_observable(observable, circuit.num_qubits)?;
    }

    let (_, has_partial_independence) = analyze_independence(circuit);
    let ExecutionPlan::Backend(plan) = resolve(kind, circuit, has_partial_independence) else {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "expectation values need a backend that holds a state; the stabilizer-rank \
                     route returns probabilities only"
                .into(),
        });
    };

    let mut backend = plan.build(seed);
    if !backend.supports_pauli_expectation() {
        return Err(PrismError::BackendUnsupported {
            backend: backend.name().to_string(),
            operation: "Pauli expectation values".to_string(),
        });
    }
    execute(&mut *backend, circuit, &SimOptions::classical_only())?;
    let values = backend.pauli_expectations(observables)?;
    Ok(analytic_expectations(values, backend_metadata(&*backend)))
}

fn expectation_values_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<ExpectationResult> {
    // Validate before the 2^n simulation so bad observables fail cheaply.
    let masks = observables
        .iter()
        .map(|obs| pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;

    let accel = accel_for(kind, Family::Statevector, circuit.num_qubits);
    let mut backend = build_statevector(&accel, seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    backend.apply_instructions(&fused.instructions)?;

    let exported;
    let state: &[Complex64] = if backend.is_gpu_resident() {
        exported = backend.export_statevector()?;
        &exported
    } else {
        backend.state_vector()
    };
    let norm = crate::backend::state_norm_sqr(state);
    let values = pauli_expectations_from_masks(state, &masks, norm);
    let metadata = backend_metadata(&backend);
    Ok(analytic_expectations(values, metadata))
}

/// Reject out-of-range qubits and duplicate factors in a joint Pauli
/// observable.
///
/// Same checks [`pauli_masks`] makes, without its `1 << qubit` mask width, so
/// it also covers the backends that run past 64 qubits.
pub(crate) fn validate_observable(observable: &[PauliTerm], num_qubits: usize) -> Result<()> {
    let mut seen = vec![false; num_qubits];
    for term in observable {
        if term.qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
    }
    Ok(())
}

/// Validate a joint Pauli observable and reduce it to `(Xmask, Zmask, #Y)`,
/// where `Xmask` covers X and Y factors and `Zmask` covers Z and Y factors.
pub(crate) fn pauli_masks(
    observable: &[PauliTerm],
    num_qubits: usize,
) -> Result<(usize, usize, u32)> {
    let mut xmask = 0usize;
    let mut zmask = 0usize;
    let mut num_y = 0u32;
    let mut seen = vec![false; num_qubits];
    for term in observable {
        if term.qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
        let bit = 1usize << term.qubit;
        match term.axis {
            PauliAxis::X => xmask |= bit,
            PauliAxis::Z => zmask |= bit,
            PauliAxis::Y => {
                xmask |= bit;
                zmask |= bit;
                num_y += 1;
            }
        }
    }
    Ok((xmask, zmask, num_y))
}

/// Complex Pauli sandwich `⟨λ|P|φ⟩`, where `P` acts as
/// `P|j⟩ = i^{#Y}·(-1)^{popcount(j & Zmask)}·|j ⊕ Xmask⟩`. Returns the raw
/// (unnormalized) complex value. The adjoint gradient engine uses this with
/// distinct `λ` and `φ`; `pauli_expectation_from_masks` is the `λ = φ` case.
///
/// Inlined explicitly: this is the adjoint engine's inner reduction, called
/// once per parameter, and leaving the decision to LTO ties it to how many
/// other callers the function happens to have.
#[inline]
pub(crate) fn pauli_sandwich(
    lambda: &[Complex64],
    phi: &[Complex64],
    xmask: usize,
    zmask: usize,
    num_y: u32,
) -> Complex64 {
    let term = |j: usize, amp: Complex64| {
        let partner = lambda[j ^ xmask];
        let sign = if (j & zmask).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        partner.conj() * amp * sign
    };

    // Higher threshold than gate kernels: the sandwich is a single lightweight
    // O(N) reduction, so Rayon fan-out only pays off past 2^16 elements. Below
    // that (and for a multi-term Hamiltonian's many small reductions) the
    // sequential path is faster.
    #[cfg(feature = "parallel")]
    const SANDWICH_MIN_PAR_QUBITS: usize = 16;
    #[cfg(feature = "parallel")]
    let acc: Complex64 = if phi.len() >= (1 << SANDWICH_MIN_PAR_QUBITS) {
        use rayon::prelude::*;
        phi.par_iter()
            .enumerate()
            .map(|(j, &amp)| term(j, amp))
            .sum()
    } else {
        phi.iter().enumerate().map(|(j, &amp)| term(j, amp)).sum()
    };
    #[cfg(not(feature = "parallel"))]
    let acc: Complex64 = phi.iter().enumerate().map(|(j, &amp)| term(j, amp)).sum();

    acc * i_pow(num_y)
}

/// `i^{num_y}`, the phase a joint Pauli picks up from its Y factors.
#[inline]
pub(crate) fn i_pow(num_y: u32) -> Complex64 {
    match num_y % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// Exact `⟨ψ|P|ψ⟩` from the reduced observable masks. Normalization
/// independent, so raw backend amplitudes are fine.
pub(crate) fn pauli_expectation_from_masks(
    state: &[Complex64],
    xmask: usize,
    zmask: usize,
    num_y: u32,
    norm: f64,
) -> f64 {
    if norm == 0.0 {
        return 0.0;
    }
    pauli_sandwich(state, state, xmask, zmask, num_y).re / norm
}

/// Exact `⟨ψ|P_i|ψ⟩` for every mask triple in one traversal of `state`.
///
/// Same value as [`pauli_expectation_from_masks`] per entry, to within the
/// association of the sum. A Z-only observable has `xmask == 0` and therefore
/// no Y factor, so its contribution is `±|amp|^2` and needs neither the partner
/// load nor complex arithmetic; the two families are accumulated separately for
/// that reason.
pub(crate) fn pauli_expectations_from_masks(
    state: &[Complex64],
    masks: &[(usize, usize, u32)],
    norm: f64,
) -> Vec<f64> {
    if norm == 0.0 {
        return vec![0.0; masks.len()];
    }
    if masks.len() < 2 {
        return masks
            .iter()
            .map(|&(xmask, zmask, num_y)| {
                pauli_expectation_from_masks(state, xmask, zmask, num_y, norm)
            })
            .collect();
    }

    let z_only: Vec<usize> = masks
        .iter()
        .filter(|&&(xmask, _, _)| xmask == 0)
        .map(|&(_, zmask, _)| zmask)
        .collect();
    let general: Vec<(usize, usize)> = masks
        .iter()
        .filter(|&&(xmask, _, _)| xmask != 0)
        .map(|&(xmask, zmask, _)| (xmask, zmask))
        .collect();

    let accumulate = |z_acc: &mut [f64], g_acc: &mut [Complex64], base: usize, len: usize| {
        for j in base..base + len {
            let amp = state[j];
            let n2 = amp.norm_sqr();
            for (slot, &zmask) in z_acc.iter_mut().zip(z_only.iter()) {
                *slot += if (j & zmask).count_ones() & 1 == 1 {
                    -n2
                } else {
                    n2
                };
            }
            for (slot, &(xmask, zmask)) in g_acc.iter_mut().zip(general.iter()) {
                let partner = state[j ^ xmask];
                let sign = if (j & zmask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                *slot += partner.conj() * amp * sign;
            }
        }
    };

    let zeros = || {
        (
            vec![0.0f64; z_only.len()],
            vec![Complex64::new(0.0, 0.0); general.len()],
        )
    };
    let (mut z_sum, mut g_sum) = zeros();

    #[cfg(feature = "parallel")]
    if state.len() >= crate::backend::MIN_PAR_REDUCE_ELEMS {
        use rayon::prelude::*;
        let chunk = crate::backend::MIN_PAR_ELEMS;
        let (z, g) = state
            .par_chunks(chunk)
            .enumerate()
            .fold(zeros, |mut acc, (c, block)| {
                accumulate(&mut acc.0, &mut acc.1, c * chunk, block.len());
                acc
            })
            .reduce(zeros, |mut a, b| {
                for (slot, v) in a.0.iter_mut().zip(b.0) {
                    *slot += v;
                }
                for (slot, v) in a.1.iter_mut().zip(b.1) {
                    *slot += v;
                }
                a
            });
        return finish_expectations(masks, &z, &g, norm);
    }

    accumulate(&mut z_sum, &mut g_sum, 0, state.len());
    finish_expectations(masks, &z_sum, &g_sum, norm)
}

/// Interleave the two accumulator families back into observable order.
fn finish_expectations(
    masks: &[(usize, usize, u32)],
    z_sum: &[f64],
    g_sum: &[Complex64],
    norm: f64,
) -> Vec<f64> {
    let (mut zi, mut gi) = (0, 0);
    masks
        .iter()
        .map(|&(xmask, _, num_y)| {
            if xmask == 0 {
                zi += 1;
                z_sum[zi - 1] / norm
            } else {
                gi += 1;
                (g_sum[gi - 1] * i_pow(num_y)).re / norm
            }
        })
        .collect()
}

/// Multi-shot execution for the distributed statevector backend.
///
/// Every rank runs this function in lockstep. Circuits with only terminal
/// measurements run once and sample basis indices without gathering the dense
/// state on any rank. Circuits with mid-circuit measurements run once per shot,
/// prefused, with per-shot seeds matching the generic slow path.
#[cfg(feature = "distributed")]
fn run_shots_distributed(
    context: std::sync::Arc<crate::distributed::DistributedContext>,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    use crate::backend::distributed_statevector::DistributedStatevectorBackend;

    let meas_map = circuit.measurement_map();
    if meas_map.is_empty() {
        // No measurements means every shot is all false, but init must still
        // run so invalid rank counts and local qubit floor violations surface as
        // errors instead of fabricated output.
        let mut backend = DistributedStatevectorBackend::new(context, seed);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        return Ok(ShotsResult::from_shots(
            vec![vec![false; circuit.num_classical_bits]; num_shots],
            circuit.num_classical_bits,
        )
        .with_metadata(backend_metadata(&backend)));
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let mut backend = DistributedStatevectorBackend::new(context, seed);
        execute(&mut backend, &stripped, &SimOptions::classical_only())?;
        let samples = backend.sample_basis_states(num_shots, seed)?;
        return Ok(ShotsResult::from_shots(
            shots_from_basis_samples(&samples, &meas_map, circuit.num_classical_bits),
            circuit.num_classical_bits,
        )
        .with_metadata(backend_metadata(&backend)));
    }

    let probe = DistributedStatevectorBackend::new(context.clone(), seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if probe.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, probe.supports_fused_gates());
    let opts = SimOptions::classical_only();
    let mut shots = Vec::with_capacity(num_shots);
    let mut metadata = RunMetadata::exact(ResolvedBackend::Distributed);
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let mut backend = DistributedStatevectorBackend::new(context.clone(), shot_seed);
        let result = execute_circuit(&mut backend, &fused, &opts)?;
        metadata.weaken_with(&result.metadata);
        shots.push(result.classical_bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

/// Execute a circuit multiple times with explicit backend selection.
pub(crate) fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    // The distributed backend runs every rank in lockstep, so shot execution
    // must not route through shortcuts that reshape the collective call
    // sequence. Dispatch directly.
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = &kind {
        return run_shots_distributed(context.clone(), circuit, num_shots, seed);
    }

    let bits = circuit.num_classical_bits;
    let source = prepare_shot_source(&kind, circuit, num_shots, seed)?;
    let Some(metadata) = source.metadata() else {
        return run_shots_per_shot(kind, circuit, num_shots, seed);
    };
    let result = match source {
        ShotSource::Compiled {
            mut sampler,
            meas_map,
            ..
        } => {
            let packed = sampler.try_sample_bulk_packed(num_shots)?;
            ShotsResult::from_shots(
                packed_shots_to_classical_bits(&packed, &meas_map, bits),
                bits,
            )
        }
        ShotSource::TerminalStatevector { backend, meas_map } => {
            let shots = if backend.is_gpu_resident() {
                let probs = backend.probabilities()?;
                sample_shots_from_probs(&probs, &meas_map, bits, num_shots, seed)
            } else {
                sample_shots_from_state(
                    backend.state_vector(),
                    backend.probability_scale(),
                    &meas_map,
                    bits,
                    num_shots,
                    seed,
                )
            };
            ShotsResult::from_shots(shots, bits)
        }
        ShotSource::Native {
            mut backend,
            meas_map,
        } => {
            let samples = backend.sample_basis_states(num_shots, seed)?;
            ShotsResult::from_shots(shots_from_basis_samples(&samples, &meas_map, bits), bits)
        }
        ShotSource::TerminalProbabilities {
            probs, meas_map, ..
        } => ShotsResult::from_shots(sample_shots(&probs, &meas_map, bits, num_shots, seed), bits),
        ShotSource::StabilizerRank => {
            stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed)?
        }
        ShotSource::PerShot => unreachable!("handled above"),
    };
    Ok(result.with_metadata(metadata))
}

/// Run `circuit` once per shot, which mid-circuit measurements force.
fn run_shots_per_shot(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    // Pre-compute seed-independent analysis to avoid redundant work.
    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }

    let (decompose, has_partial_independence) = analyze_independence(circuit);

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
    if kind.is_auto() && auto_stabilizer_rank_t_count(circuit, MAX_AUTO_T_COUNT_SHOTS).is_some() {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }

    if has_temporal_clifford_opportunity(&kind, circuit) {
        if decompose.is_none() {
            if let Some(tc) = plan_temporal_clifford(&kind, circuit) {
                let route = ResolvedBackend::Statevector;
                return collect_shots(circuit, num_shots, seed, route, |shot_seed| {
                    let outcome = run_temporal_clifford(&tc, shot_seed, false)?;
                    Ok((outcome.classical_bits, outcome.metadata))
                });
            }
        }
        // Decomposable circuits with a temporal prefix keep the per-shot
        // full-pipeline route; the prefix spans blocks that decomposition
        // would otherwise split.
        let opts = SimOptions::classical_only();
        let route = resolve_backend(&kind, circuit, has_partial_independence).resolved();
        return collect_shots(circuit, num_shots, seed, route, |shot_seed| {
            let outcome = run_with_internal(kind.clone(), circuit, shot_seed, opts)?;
            Ok((outcome.classical_bits, outcome.metadata))
        });
    }

    let opts = SimOptions::classical_only();

    if let Some(ref comps) = decompose {
        let partitions = circuit.partition_subcircuits(comps);
        let block_plans: Vec<BackendPlan> = partitions
            .iter()
            .map(|(sub, _, _)| {
                if !kind.is_auto() {
                    validate_explicit_backend(&kind, sub)?;
                }
                Ok(resolve_backend(&kind, sub, false))
            })
            .collect::<Result<_>>()?;
        let fused_blocks: Vec<_> = partitions
            .iter()
            .zip(&block_plans)
            .map(|((sub, _, _), plan)| {
                crate::circuit::fusion::fuse_circuit(sub, plan.supports_fused())
            })
            .collect();

        collect_shots(
            circuit,
            num_shots,
            seed,
            ResolvedBackend::Decomposed,
            |shot_seed| {
                let result = run_decomposed_prefused(
                    &block_plans,
                    comps,
                    &partitions,
                    &fused_blocks,
                    shot_seed,
                    &opts,
                    circuit,
                )?;
                Ok((result.classical_bits, result.metadata))
            },
        )
    } else {
        let plan = resolve_backend(&kind, circuit, has_partial_independence);
        let fused = crate::circuit::fusion::fuse_circuit(circuit, plan.supports_fused());

        collect_shots(circuit, num_shots, seed, plan.resolved(), |shot_seed| {
            let mut backend = plan.build(shot_seed);
            let outcome = execute_circuit(&mut *backend, &fused, &opts)?;
            Ok((outcome.classical_bits, outcome.metadata))
        })
    }
}

/// Each shot evolves its own state, so `shot` returns the provenance of its own
/// run and the ensemble keeps the weakest claim across them. `route` names the
/// engine for a zero-shot request, which runs nothing to read provenance off.
fn collect_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    route: ResolvedBackend,
    mut shot: impl FnMut(u64) -> Result<(Vec<bool>, RunMetadata)>,
) -> Result<ShotsResult> {
    let mut shots = Vec::with_capacity(num_shots);
    let mut metadata = RunMetadata::exact(route);
    for i in 0..num_shots {
        let (bits, shot_metadata) = shot(seed.wrapping_add(i as u64))?;
        if i == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

/// Family choice for auto-routed non-Pauli noise trajectories. Restricted to
/// families whose trajectory operations (1q Kraus, qubit probability, reduced
/// density matrix, reset) are supported; the statevector leaf carries the
/// kind's acceleration.
fn general_noise_plan(kind: &BackendKind, circuit: &Circuit) -> BackendPlan {
    let family = if !circuit.has_entangling_gates() {
        Family::ProductState
    } else if circuit.num_qubits > max_statevector_qubits() {
        if circuit.is_sparse_friendly() {
            Family::Sparse
        } else {
            Family::Mps
        }
    } else {
        Family::Statevector
    };
    plan_for_family(kind, family, circuit.num_qubits)
}

/// Execute a noisy circuit for multiple shots with explicit backend selection.
///
/// For Clifford circuits with Auto/Stabilizer/FactoredStabilizer backends,
/// uses the compiled noisy sampler (fast O(n²·m) compile + O(events·m/64) per shot).
/// For all other cases, falls back to per-shot simulation with noise injection.
/// The compiled noisy path is limited to terminal measurements with no resets
/// or classical conditionals.
pub(crate) fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    noise_model.validate_for(circuit)?;

    // Trajectory execution runs shots on Rayon worker threads, whose
    // scheduling order differs per rank. Per-shot distributed backends would
    // issue collectives out of lockstep and deadlock or corrupt exchanges.
    // Reject until a lockstep noisy path exists.
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "noisy shot sampling is not supported on the distributed backend; \
                     trajectory execution cannot keep rank collectives in lockstep"
                .into(),
        });
    }

    if matches!(kind, BackendKind::DensityMatrix) {
        let probs = exact_noisy_probabilities(circuit, noise_model, None, seed)?;
        return Ok(ShotsResult::from_shots(
            sample_exact_noisy_shots(&probs, circuit, noise_model, num_shots, seed),
            circuit.num_classical_bits,
        )
        .with_metadata(RunMetadata::exact(ResolvedBackend::DensityMatrix)));
    }

    if !kind.supports_noisy_per_shot() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "this backend holds no per-shot pure state to inject noise into; select \
                     DensityMatrix for the exact mixed state, or a backend that evolves one \
                     state per trajectory"
                .into(),
        });
    }

    let is_stabilizer_kind = kind.is_stabilizer_family();

    if is_stabilizer_kind && !noise_model.is_pauli_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "stabilizer backends only support Pauli/depolarizing noise; use {} for amplitude damping, phase damping, thermal relaxation, custom Kraus, or readout errors",
                BackendKind::general_noise_backend_names()
            ),
        });
    }

    if !noise_model.is_pauli_only() && !kind.supports_general_noise() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "non-Pauli noise requires {}",
                BackendKind::general_noise_backend_names()
            ),
        });
    }

    if is_stabilizer_kind && !circuit.is_clifford_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "circuit contains non-Clifford gates".into(),
        });
    }

    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }

    if noise_model.is_pauli_only() {
        let use_compiled = (kind.is_auto()
            || matches!(
                kind,
                BackendKind::Stabilizer | BackendKind::FactoredStabilizer
            ))
            && supports_compiled_measurement_sampling(circuit)
            || {
                #[cfg(feature = "gpu")]
                {
                    matches!(kind, BackendKind::StabilizerGpu { .. })
                        && supports_compiled_measurement_sampling(circuit)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    false
                }
            };

        if use_compiled {
            #[cfg(feature = "gpu")]
            if let BackendKind::StabilizerGpu { context } = &kind {
                return noise::run_shots_noisy_with_gpu(
                    circuit,
                    noise_model,
                    num_shots,
                    seed,
                    context.clone(),
                );
            }
            return noise::run_shots_noisy(circuit, noise_model, num_shots, seed);
        }
    }

    let plan = if kind.is_auto() && !noise_model.is_pauli_only() {
        general_noise_plan(&kind, circuit)
    } else {
        resolve_backend(&kind, circuit, false)
    };
    let route = plan.resolved();
    trajectory::run_trajectories(
        |s| plan.build(s),
        circuit,
        noise_model,
        num_shots,
        seed,
        plan.is_gpu(),
        route,
    )
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "gpu"))]
mod terminal_gpu_stub_tests;

#[cfg(all(test, feature = "gpu"))]
mod expectation_gpu_stub_tests;

#[cfg(all(test, feature = "gpu"))]
mod noise_gpu_stub_tests;

#[cfg(test)]
mod terminal_candidate_matrix_tests;
