//! The [`Simulate`] request builder and its terminals.

use super::*;

/// Typestate marker: [`Simulate`] builder with no seed chosen yet.
#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

/// Typestate marker: [`Simulate`] builder with its RNG seed fixed.
#[derive(Debug, Clone, Copy)]
pub struct Seeded {
    pub(super) seed: u64,
}

/// Builder for query-aware simulation requests.
pub struct Simulate<'c, SeedState> {
    pub(super) circuit: &'c Circuit,
    pub(super) kind: BackendKind,
    pub(super) seed: SeedState,
    pub(super) noise_model: Option<&'c noise::NoiseModel>,
    pub(super) initial_state: Option<StartState<'c>>,
    pub(super) require_exact: bool,
    pub(super) limits: Option<Limits>,
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
    /// at a bounded bond dimension, which is the only route those circuits have.
    /// Its result reports [`Exactness::Exact`] when no cut discarded weight, but
    /// that is known only after the run, so this rejects the route beforehand.
    /// Call this when an approximate answer is worse than no answer, and the run
    /// returns `IncompatibleBackend` naming the engine it would have used.
    ///
    /// A circuit whose gates all preserve computational basis states is the
    /// exception: it stays on one basis state, so that route holds it at bond 1
    /// and answers exactly.
    ///
    /// Routes that can be decided from the circuit are rejected before any state
    /// is allocated; sparse Pauli dynamics only learns that it truncated while
    /// propagating, so that one is caught on the finished result instead.
    #[inline]
    pub fn require_exact(mut self) -> Self {
        self.require_exact = true;
        self
    }

    /// Run under `limits` instead of the process default ([`set_default_limits`]).
    ///
    /// Values `limits` leaves unset come from the process default, then the
    /// `PRISM_MAX_*` variables, then detection; see [`Limits`] for how a memory
    /// budget interacts with the per-cap values below it. A cap the run exceeds
    /// reports [`PrismError::ResourceLimit`] naming the field to raise.
    ///
    /// [`set_default_limits`]: crate::set_default_limits
    #[inline]
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = Some(limits);
        self
    }

    /// Attach a noise model.
    ///
    /// [`Simulate::shots`] and [`Simulate::sample_counts`] accept one on any
    /// backend with a per-shot pure state, averaging trajectories.
    /// [`Simulate::run`], [`Simulate::marginals`],
    /// [`Simulate::expectation_values`] and
    /// [`Simulate::reduced_density_matrix`] answer from the exact mixture
    /// instead, which only [`BackendKind::DensityMatrix`] and its device
    /// sibling hold, so they require one of those, as does
    /// [`Simulate::expectation_gradient_shift`] and, only to decline on it,
    /// [`Simulate::entanglement_entropy`]: a mixture has no Schmidt
    /// decomposition. [`Simulate::expectation_gradient`] rejects a noise model
    /// on every backend.
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
    /// other than the statevector (dense, device, or distributed) and
    /// [`BackendKind::DensityMatrix`] reports `IncompatibleBackend`.
    /// [`Simulate::expectation_gradient`] declines a start
    /// state, as do [`Simulate::shots`] and [`Simulate::sample_counts`] with a
    /// noise model attached, since trajectory replay has no start-state path.
    #[inline]
    pub fn initial_state(mut self, amplitudes: &'c [Complex64]) -> Self {
        self.initial_state = Some(StartState::Amplitudes(amplitudes));
        self
    }

    /// Start from the mixture `rho` instead of |0...0⟩.
    ///
    /// Row-major `2^n x 2^n` in the layout
    /// [`DensityMatrixBackend::density_matrix`] exports, qubit 0 in the least
    /// significant bit of both indices, so entry `(r << n) | c` is `<r|rho|c>`.
    /// The buffer must be Hermitian, each entry pair agreeing to 1e-12, and
    /// have unit trace to 1e-9; one failing either check, or of a length other
    /// than `4^n`, is rejected with `InvalidParameter` naming the check.
    /// Positive semidefiniteness is not checked, because the backend carries no
    /// eigendecomposition to check it with.
    ///
    /// Only [`BackendKind::DensityMatrix`] and its device sibling hold a
    /// mixture, so every other kind, [`BackendKind::Auto`] included, reports
    /// `IncompatibleBackend`. The terminals that decline a start state decline
    /// this one for the same reasons. Replaces an earlier
    /// [`Simulate::initial_state`], as that call replaces this one.
    ///
    /// [`DensityMatrixBackend::density_matrix`]: crate::backend::density_matrix::DensityMatrixBackend::density_matrix
    #[inline]
    pub fn initial_density_matrix(mut self, rho: &'c [Complex64]) -> Self {
        self.initial_state = Some(StartState::DensityMatrix(rho));
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
            limits: self.limits,
        }
    }
}

impl<'c> Simulate<'c, Seeded> {
    #[inline]
    fn seed_value(&self) -> u64 {
        self.seed.seed
    }

    pub(super) fn resolved_limits(&self) -> ResolvedLimits {
        ResolvedLimits::resolve(self.limits.as_ref())
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
    /// density-matrix backend; the classical bits are one draw, matching
    /// `shots(1)`. Readout error reaches the draw and not the state, so a
    /// model carrying it is rejected rather than answered with two fields from
    /// different distributions; `shots` and `sample_counts` apply it.
    #[inline]
    pub fn run(self) -> Result<RunOutcome> {
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if self.require_exact {
            self.reject_planned_approximation(Query::Run)?;
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "a single run")?;
            reject_readout_at(self.circuit, noise_model, "a single run")?;
            let probabilities = exact_noisy_probabilities(
                &self.kind,
                self.circuit,
                noise_model,
                self.initial_state,
                seed,
                limits,
            )?;
            let classical_bits =
                sample_exact_noisy_shots(&probabilities, self.circuit, noise_model, 1, seed)
                    .swap_remove(0);
            let outcome = RunOutcome {
                classical_bits,
                probabilities: Some(probabilities),
                metadata: exact_mixture_metadata(&self.kind),
                saves: Vec::new(),
            };
            ensure_saves_recorded(self.circuit, &outcome)?;
            return Ok(outcome);
        }
        if let Some(state) = self.initial_state {
            return run_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                seed,
                &SimOptions::default(),
                limits,
            );
        }
        let outcome =
            run_with_internal(self.kind, self.circuit, seed, SimOptions::default(), limits)?;
        ensure_exact_result(self.require_exact, &outcome.metadata)?;
        Ok(outcome)
    }

    /// Refuse a circuit carrying save points on a terminal that cannot return
    /// them. Only [`Simulate::run`] has a place to put the records.
    fn reject_saves(&self, terminal: &str) -> Result<()> {
        if self.circuit.save_count() == 0 {
            return Ok(());
        }
        Err(PrismError::IncompatibleBackend {
            backend: format!("{:?}", self.kind),
            reason: format!(
                "the circuit carries {} save point(s), which only `run` returns, so \
                 `{terminal}` declines it",
                self.circuit.save_count()
            ),
        })
    }

    fn reject_planned_approximation(&self, query: Query<'_>) -> Result<()> {
        let plan = plan::inspect(self, query, &self.resolved_limits())?;
        let backend = plan::approximation_name(&self.kind, plan.backend);
        reject_predicted_route(
            (plan.exactness != PlannedExactness::Exact).then_some(backend.as_str()),
        )
    }

    /// Execute `num_shots` times, collecting per-shot classical bits. Accepts
    /// an attached noise model.
    #[inline]
    pub fn shots(self, num_shots: usize) -> Result<ShotsResult> {
        self.reject_saves("shots")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        let require_exact = self.require_exact;
        if require_exact {
            self.reject_planned_approximation(Query::Shots(num_shots))?;
        }
        let result = if let Some(noise_model) = self.noise_model {
            self.require_no_initial_state_under_noise("shot sampling")?;
            run_shots_with_noise(
                self.kind,
                self.circuit,
                noise_model,
                num_shots,
                seed,
                limits,
            )?
        } else if let Some(state) = self.initial_state {
            shots_from_initial_state(&self.kind, self.circuit, state, num_shots, seed, limits)?
        } else {
            run_shots_with(self.kind, self.circuit, num_shots, seed, limits)?
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
        self.reject_saves("sample_counts")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if self.require_exact {
            self.reject_planned_approximation(Query::Counts(num_shots))?;
        }
        let (counts, metadata) = if let Some(noise_model) = self.noise_model {
            self.require_no_initial_state_under_noise("count sampling")?;
            let shots = run_shots_with_noise(
                self.kind,
                self.circuit,
                noise_model,
                num_shots,
                seed,
                limits,
            )?;
            (shots.counts(), shots.metadata)
        } else if let Some(state) = self.initial_state {
            let shots =
                shots_from_initial_state(&self.kind, self.circuit, state, num_shots, seed, limits)?;
            (shots.counts(), shots.metadata)
        } else {
            run_counts_with(self.kind, self.circuit, num_shots, seed, limits)?
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
    /// backend, and rejects a model carrying readout error, since
    /// `sample_counts` is the terminal that applies it.
    #[inline]
    pub fn marginals(self) -> Result<MarginalsResult> {
        self.reject_saves("marginals")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "marginals")?;
            reject_readout_at(self.circuit, noise_model, "marginals")?;
            let probs = exact_noisy_probabilities(
                &self.kind,
                self.circuit,
                noise_model,
                self.initial_state,
                seed,
                limits,
            )?;
            return Ok(MarginalsResult {
                marginals: probs.marginals(),
                metadata: exact_mixture_metadata(&self.kind),
            });
        }
        let result = if let Some(state) = self.initial_state {
            marginals_from_initial_state(&self.kind, self.circuit, state, seed, limits)?
        } else {
            let route = decide_marginals(&self.kind, self.circuit, limits)?;
            if self.require_exact {
                plan::require_exact_marginal(&route, &self.kind, self.circuit, limits)?;
            }
            execute_marginals(
                route,
                &self.kind,
                self.circuit,
                seed,
                limits,
                self.require_exact,
            )?
        };
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on the circuit's
    /// output state, honoring the selected backend.
    ///
    /// Each observable is a product of single-qubit Paulis (identity factors
    /// omitted). The circuit must be unitary, except that the density-matrix
    /// backends evolve a reset as a channel. Clifford circuits propagate each
    /// observable exactly. Non-Clifford circuits use the state vector while they
    /// fit it; above that cap the selected backend evaluates the observable on
    /// its own representation, and a backend without one reports
    /// `BackendUnsupported` naming itself.
    ///
    /// With a noise model attached the value is the exact `Tr(rho P)` on the
    /// evolved mixture, which needs the density-matrix backend. A model
    /// carrying readout error is rejected: readout acts on the measurement
    /// record, which no observable sees, and `shots` on the same model would
    /// disagree by the readout rate.
    #[inline]
    pub fn expectation_values(self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        self.reject_saves("expectation_values")?;
        self.expectation_values_reported(observables)
            .map(ExpectationResult::into_values)
    }

    /// [`Simulate::expectation_values`] with the provenance of the run and, for
    /// a route that estimates rather than evaluates, a standard error per value.
    pub fn expectation_values_reported(
        self,
        observables: &[Vec<PauliTerm>],
    ) -> Result<ExpectationResult> {
        self.reject_saves("expectation_values_reported")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if let Some(noise_model) = self.noise_model {
            reject_readout_at(self.circuit, noise_model, "expectation values")?;
        }
        if let BackendKind::PauliPath { epsilon, max_terms } = self.kind {
            if self.require_exact {
                reject_approximate_route(&self.kind, self.circuit, limits)?;
            }
            reject_pauli_path_initial_state(self.initial_state)?;
            return pauli_path_expectations(
                self.circuit,
                self.noise_model,
                observables,
                epsilon,
                max_terms,
            );
        }
        if evolves_resets_as_channels(&self.kind, self.circuit) {
            let result = reset_channel_expectations(
                &self.kind,
                self.circuit,
                observables,
                self.noise_model,
                self.initial_state,
                seed,
                limits,
            )?;
            ensure_exact_result(self.require_exact, &result.metadata)?;
            return Ok(result);
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "expectation values")?;
            require_unitary_expectation_circuit(&self.kind, self.circuit)?;
            let values = noise::dm_expectation_values(
                &self.kind,
                self.circuit,
                observables,
                Some(noise_model),
                self.initial_state,
                seed,
                limits,
            )?;
            return Ok(analytic_expectations(
                values,
                exact_mixture_metadata(&self.kind),
            ));
        }
        if let Some(state) = self.initial_state {
            require_unitary_expectation_circuit(&self.kind, self.circuit)?;
            return expectation_values_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                observables,
                seed,
                limits,
            );
        }
        let route = decide_expectations(&self.kind, self.circuit, observables, limits)?;
        if self.require_exact {
            plan::require_exact_expectation(&route, &self.kind, self.circuit, limits)?;
        }
        let result =
            execute_expectations(route, &self.kind, self.circuit, observables, seed, limits)?;
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// Compute `⟨H⟩` and its grouped-measurement variance for a weighted
    /// Pauli observable on the circuit's output state.
    ///
    /// The statevector family evaluates one traversal per qubit-wise-commuting
    /// group and reports the variance; see [`ObservableExpectation::variance`]
    /// for what the number means. The deterministic Pauli and Pauli path routes
    /// propagate the weighted sum once, so their truncation applies to the
    /// merged sum. Every other route, including runs with a noise model or
    /// start state attached, evaluates term by term through
    /// [`Simulate::expectation_values`] semantics. Neither reports a variance.
    /// A noise model carrying readout error is rejected for the same reason as
    /// there.
    pub fn observable_expectation(
        self,
        observable: &PauliObservable,
    ) -> Result<ObservableExpectation> {
        self.reject_saves("observable_expectation")?;
        self.observable_expectation_ref(observable)
    }

    /// [`Simulate::observable_expectation`] without consuming the builder, so
    /// [`Simulate::observable_variance`] can evaluate `H` and `H^2` on one
    /// request.
    fn observable_expectation_ref(
        &self,
        observable: &PauliObservable,
    ) -> Result<ObservableExpectation> {
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if let Some(noise_model) = self.noise_model {
            reject_readout_at(self.circuit, noise_model, "observable expectation")?;
        }
        if let BackendKind::PauliPath { epsilon, max_terms } = self.kind {
            if self.require_exact {
                reject_approximate_route(&self.kind, self.circuit, limits)?;
            }
            reject_pauli_path_initial_state(self.initial_state)?;
            return pauli_path_observable(
                self.circuit,
                self.noise_model,
                observable,
                epsilon,
                max_terms,
            );
        }
        if evolves_resets_as_channels(&self.kind, self.circuit) {
            let result = reset_channel_expectations(
                &self.kind,
                self.circuit,
                &observable_vecs(observable),
                self.noise_model,
                self.initial_state,
                seed,
                limits,
            )?;
            ensure_exact_result(self.require_exact, &result.metadata)?;
            return Ok(weighted_observable_result(
                observable,
                &result.values,
                None,
                result.metadata,
            ));
        }
        if let Some(noise_model) = self.noise_model {
            require_exact_mixture(&self.kind, "expectation values")?;
            require_unitary_expectation_circuit(&self.kind, self.circuit)?;
            let values = noise::dm_expectation_values(
                &self.kind,
                self.circuit,
                &observable_vecs(observable),
                Some(noise_model),
                self.initial_state,
                seed,
                limits,
            )?;
            return Ok(weighted_observable_result(
                observable,
                &values,
                None,
                exact_mixture_metadata(&self.kind),
            ));
        }
        if let Some(state) = self.initial_state {
            require_unitary_expectation_circuit(&self.kind, self.circuit)?;
            let result = expectation_values_from_initial_state(
                &self.kind,
                self.circuit,
                state,
                &observable_vecs(observable),
                seed,
                limits,
            )?;
            return Ok(weighted_observable_result(
                observable,
                &result.values,
                result.std_errors.as_deref(),
                result.metadata,
            ));
        }
        let result = if grouped_statevector(&self.kind, self.circuit, limits) {
            grouped_expectation_statevector(&self.kind, self.circuit, observable, seed, limits)?
        } else {
            let observables = observable_vecs(observable);
            let route = decide_expectations(&self.kind, self.circuit, &observables, limits)?;
            if self.require_exact {
                plan::require_exact_expectation(&route, &self.kind, self.circuit, limits)?;
            }
            execute_observable(
                route,
                &self.kind,
                self.circuit,
                observable,
                &observables,
                seed,
                limits,
            )?
        };
        ensure_exact_result(self.require_exact, &result.metadata)?;
        Ok(result)
    }

    /// `Var(H) = <H^2> - <H>^2` for a weighted Pauli observable on the
    /// circuit's output state.
    ///
    /// This is the spread of the operator itself, the number a shot-based
    /// estimate of `<H>` converges on dividing by the shot count. It is not
    /// [`ObservableExpectation::variance`], which sums per-group variances and
    /// so drops the covariance between measurement groups.
    ///
    /// Evaluates `H` and the square of its traceless part through
    /// [`Simulate::observable_expectation`], so backend routing, noise, and
    /// start states behave as they do there. The constant term is held out of
    /// the square rather than cancelled inside it; see
    /// [`PauliObservable::split_identity`]. The square carries up to `T^2`
    /// terms over `H`'s `T`; see [`PauliObservable::square`].
    pub fn observable_variance(self, observable: &PauliObservable) -> Result<ObservableVariance> {
        self.reject_saves("observable_variance")?;
        let limits = &self.resolved_limits();
        if self.noise_model.is_none()
            && self.initial_state.is_none()
            && !matches!(self.kind, BackendKind::PauliPath { .. })
            && !evolves_resets_as_channels(&self.kind, self.circuit)
            && grouped_statevector(&self.kind, self.circuit, limits)
        {
            return operator_variance_statevector(
                &self.kind,
                self.circuit,
                observable,
                self.seed_value(),
                limits,
            );
        }
        let (offset, traceless) = observable.split_identity();
        let mean = self.observable_expectation_ref(observable)?;
        let second = self.observable_expectation_ref(&traceless.square())?;
        let centered = mean.mean - offset;
        Ok(ObservableVariance {
            variance: second.mean - centered * centered,
            mean: mean.mean,
            metadata: mean.metadata,
        })
    }

    /// Joint probability distribution over `qubits`, `2^k` entries with
    /// `qubits[0]` in the lowest bit.
    ///
    /// The subset generalizes [`Simulate::marginals`], which reports each
    /// qubit on its own and so cannot show correlation: a Bell pair reads
    /// `(0.5, 0.5)` twice there and `[0.5, 0, 0, 0.5]` here. Routing follows
    /// [`Simulate::run`], including the exact mixture a noise model asks for
    /// and its rejection of readout error, which acts on the measurement
    /// record rather than on the state.
    ///
    /// A backend that exposes no distribution for the circuit reports
    /// `BackendUnsupported` naming itself.
    pub fn probabilities_of(self, qubits: &[usize]) -> Result<Vec<f64>> {
        self.reject_saves("probabilities_of")?;
        crate::backend::schmidt::validate_qubit_set(qubits, self.circuit.num_qubits)?;
        if self.kind.is_auto()
            && self.noise_model.is_none()
            && self.initial_state.is_none()
            && let Some(plan) =
                decide_register_subset(self.circuit, qubits, &self.resolved_limits())
        {
            return Ok(subset_probabilities(plan.values()));
        }
        let kind = format!("{:?}", self.kind);
        let outcome = self.run()?;
        let probabilities = outcome
            .probabilities
            .ok_or(PrismError::BackendUnsupported {
                backend: kind,
                operation: "a probability distribution to marginalize".into(),
            })?;
        Ok(probabilities.subset_marginal(qubits))
    }

    /// Full amplitude vector of the circuit's output state, honoring the
    /// selected backend.
    ///
    /// Indexed with qubit 0 in the least significant bit, so `x q[0]` puts the
    /// amplitude at index 1. The circuit must be unitary, for the reason
    /// [`Simulate::reduced_density_matrix`] gives.
    ///
    /// A noise model declines: a mixture has no single amplitude vector, and
    /// [`Simulate::reduced_density_matrix`] over the whole register is the
    /// terminal that answers there. The density-matrix backend declines for
    /// the same reason whether or not noise is attached.
    ///
    /// The vector holds `2^n` amplitudes, so a register past the dense export
    /// cap reports `IncompatibleBackend` before allocating rather than after.
    pub fn state_vector(self) -> Result<Vec<Complex64>> {
        self.reject_saves("state_vector")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        let diagnostic = Diagnostic::StateVector;
        require_unitary_circuit(&self.kind, self.circuit, "a statevector requires")?;
        if self.require_exact {
            self.reject_planned_approximation(Query::StateVector)?;
        }
        if self.noise_model.is_some() {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: format!(
                    "{} is a pure state; a noise model evolves a mixture, which \
                     `reduced_density_matrix` over the whole register reports",
                    diagnostic.terminal()
                ),
            });
        }
        let (backend, _) = diagnostic_backend(
            &self.kind,
            self.circuit,
            self.initial_state,
            seed,
            diagnostic,
            self.circuit.num_qubits,
            limits,
        )?;
        ensure_exact_result(self.require_exact, &backend_metadata(&*backend))?;
        backend.export_statevector()
    }

    /// Reduced density matrix of `qubits` on the circuit's output state,
    /// honoring the selected backend.
    ///
    /// Row major with side `2^k`; [`ReducedDensityMatrix::data`] states the
    /// index order. The subsystem is named once and may be the whole register.
    /// The circuit must be unitary: the answer is read off one state, and a
    /// measurement, reset or conditional leaves one seeded branch of several.
    ///
    /// An explicitly selected backend that holds no partial trace reports
    /// `BackendUnsupported` naming itself. Under [`BackendKind::Auto`] a route
    /// that cannot answer falls back to the statevector while the circuit fits
    /// its cap, so the diagnostic is served rather than declined by a choice
    /// the caller did not make. With a noise model attached the answer is the
    /// marginal of the exact mixture, which needs the density-matrix backend.
    pub fn reduced_density_matrix(self, qubits: &[usize]) -> Result<ReducedDensityMatrix> {
        self.reject_saves("reduced_density_matrix")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        let diagnostic = Diagnostic::ReducedDensityMatrix;
        let terminal = diagnostic.terminal();
        crate::backend::schmidt::validate_qubit_set(qubits, self.circuit.num_qubits)?;
        require_unitary_circuit(
            &self.kind,
            self.circuit,
            "a reduced density matrix requires",
        )?;
        if self.require_exact {
            self.reject_planned_approximation(Query::ReducedDensityMatrix(qubits))?;
        }
        if let Some(noise_model) = self.noise_model {
            reject_readout_at(self.circuit, noise_model, terminal)?;
            require_exact_mixture(&self.kind, terminal)?;
            let mut mixture = noise::evolve_density_matrix(
                &self.kind,
                self.circuit,
                Some(noise_model),
                self.initial_state,
                seed,
                limits,
            )?;
            return Ok(ReducedDensityMatrix {
                qubits: qubits.to_vec(),
                data: mixture.reduced_density_matrix(qubits)?,
                metadata: exact_mixture_metadata(&self.kind),
            });
        }
        let (mut backend, reason) = diagnostic_backend(
            &self.kind,
            self.circuit,
            self.initial_state,
            seed,
            diagnostic,
            qubits.len(),
            limits,
        )?;
        ensure_exact_result(
            self.require_exact,
            &backend_metadata(&*backend).with_reason(reason),
        )?;
        let data = backend.reduced_density_matrix(qubits)?;
        // A tensor network truncates while contracting the query itself, so the
        // bound is read after it.
        Ok(ReducedDensityMatrix {
            qubits: qubits.to_vec(),
            data,
            metadata: backend_metadata(&*backend).with_reason(reason),
        })
    }

    /// Entanglement entropy of `subsystem` across its cut with the rest of the
    /// register, in nats, honoring the selected backend.
    ///
    /// `subsystem` must leave both sides of the cut non-empty, and the circuit
    /// must be unitary, for the reason [`Simulate::reduced_density_matrix`]
    /// gives. The Schmidt values come back with the entropy, descending and
    /// normalized.
    ///
    /// A backend that holds the entropy without the spectrum behind it, a
    /// stabilizer or factored cut past the export cap, answers with
    /// [`EntropyResult::schmidt_values`] set to `None`.
    /// An explicitly selected backend that holds neither reports
    /// `BackendUnsupported` naming itself; under [`BackendKind::Auto`] such a
    /// route falls back to the statevector while the circuit fits its cap. A
    /// noise model declines outright: it sends the run to the density matrix,
    /// whose mixed state has no Schmidt decomposition.
    pub fn entanglement_entropy(self, subsystem: &[usize]) -> Result<EntropyResult> {
        self.reject_saves("entanglement_entropy")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        let diagnostic = Diagnostic::Entropy;
        let terminal = diagnostic.terminal();
        crate::backend::schmidt::validate_subsystem(subsystem, self.circuit.num_qubits)?;
        require_unitary_circuit(&self.kind, self.circuit, "entanglement entropy requires")?;
        if self.require_exact {
            self.reject_planned_approximation(Query::Entropy(subsystem))?;
        }
        if let Some(noise_model) = self.noise_model {
            reject_readout_at(self.circuit, noise_model, terminal)?;
            require_exact_mixture(&self.kind, terminal)?;
            let mut mixture = noise::evolve_density_matrix(
                &self.kind,
                self.circuit,
                Some(noise_model),
                self.initial_state,
                seed,
                limits,
            )?;
            return Ok(EntropyResult {
                subsystem: subsystem.to_vec(),
                entropy: mixture.entanglement_entropy(subsystem)?,
                schmidt_values: None,
                metadata: exact_mixture_metadata(&self.kind),
            });
        }
        let (mut backend, reason) = diagnostic_backend(
            &self.kind,
            self.circuit,
            self.initial_state,
            seed,
            diagnostic,
            subsystem.len(),
            limits,
        )?;
        let metadata = backend_metadata(&*backend).with_reason(reason);
        ensure_exact_result(self.require_exact, &metadata)?;
        // A tableau past the export cap holds the entropy as a rank while its
        // `2^r` equal values do not fit, so the spectrum declining is not the
        // terminal declining. A backend holding neither reports the spectrum's
        // own error, which is what its entropy raises too.
        let (entropy, schmidt_values) = match backend.schmidt_values(subsystem) {
            Ok(values) => (
                crate::backend::schmidt::entropy_of_schmidt_values(&values),
                Some(values),
            ),
            Err(declined) => match backend.entanglement_entropy(subsystem) {
                Ok(entropy) => (entropy, None),
                Err(_) => return Err(declined),
            },
        };
        Ok(EntropyResult {
            subsystem: subsystem.to_vec(),
            entropy,
            schmidt_values,
            metadata,
        })
    }

    /// `|<a|b>|^2` between this circuit's output state and `other`'s, honoring
    /// the backend each side selected.
    ///
    /// The two circuits must declare the same width, and both must be unitary
    /// for the reason [`Simulate::reduced_density_matrix`] gives. Each side
    /// carries its own backend, seed and start state, and each resolves to a
    /// single backend rather than the decomposed route, since two circuits
    /// need not split into the same independent blocks.
    ///
    /// A pair of unlike representations is served by a dense export of both
    /// states, so it reaches as far as the export cap does. A pair that shares
    /// one answers natively at any width: two chains in the same site order,
    /// two tableaux, two product states, or two sparse maps. A noise model on
    /// either side is rejected, since the fidelity of two mixtures is not an
    /// inner product.
    pub fn overlap(self, other: Simulate<'_, Seeded>) -> Result<OverlapResult> {
        self.reject_saves("overlap")?;
        let diagnostic = Diagnostic::Overlap;
        if self.circuit.num_qubits != other.circuit.num_qubits {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "{} needs two circuits of the same width; got {} and {} qubits",
                    diagnostic.terminal(),
                    self.circuit.num_qubits,
                    other.circuit.num_qubits
                ),
            });
        }
        let (left_backend, left_reason) = self.overlap_side(diagnostic)?;
        let (right_backend, right_reason) = other.overlap_side(diagnostic)?;
        let fidelity = left_backend.overlap_sq(&*right_backend)?;
        // Metadata is read after the work it describes. No export is known to
        // truncate today, but a bounded tensor network books its cuts there.
        Ok(OverlapResult {
            fidelity,
            left: backend_metadata(&*left_backend).with_reason(left_reason),
            right: backend_metadata(&*right_backend).with_reason(right_reason),
        })
    }

    /// One side of [`Simulate::overlap`]: the guards both sides answer to,
    /// then the backend this side's kind resolves to, run and handed back with
    /// the rule that chose it.
    fn overlap_side(self, diagnostic: Diagnostic) -> Result<(Box<dyn Backend>, RouteRule)> {
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        let terminal = diagnostic.terminal();
        require_unitary_circuit(&self.kind, self.circuit, "a state overlap requires")?;
        if self.require_exact {
            reject_approximate_route(&self.kind, self.circuit, limits)?;
        }
        if self.noise_model.is_some() {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: format!(
                    "{terminal} is an inner product of two pure states, and a noise model \
                     evolves a mixture, whose fidelity is a different computation; drop \
                     the model, or compare the mixtures through `reduced_density_matrix`"
                ),
            });
        }
        let (backend, reason) = diagnostic_backend(
            &self.kind,
            self.circuit,
            self.initial_state,
            seed,
            diagnostic,
            0,
            limits,
        )?;
        ensure_exact_result(
            self.require_exact,
            &backend_metadata(&*backend).with_reason(reason),
        )?;
        Ok((backend, reason))
    }

    /// Compute `⟨H⟩` and its exact gradient with respect to the bound
    /// parameters using the adjoint method.
    ///
    /// `params` declares which gate instructions carry parameters.
    /// Runs on the statevector backend; the selected backend must be `Auto` or
    /// `Statevector`. The circuit must be unitary. See
    /// [`gradient::run_expectation_gradient`].
    #[inline]
    pub fn expectation_gradient(
        self,
        hamiltonian: &PauliObservable,
        params: &crate::circuit::Parameters,
    ) -> Result<gradient::ExpectationGradient> {
        self.reject_saves("expectation_gradient")?;
        let seed = self.seed_value();
        let limits = &self.resolved_limits();
        if self.noise_model.is_some() {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: "the adjoint method backpropagates through a pure state, so no backend \
                         has a noisy adjoint path; drop the noise model, or take \
                         `expectation_gradient_shift` on the density-matrix backend"
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
        gradient::expectation_gradient_within(
            self.circuit,
            hamiltonian.terms(),
            params,
            seed,
            limits,
            None,
        )
    }

    /// Compute `⟨H⟩` and its gradient by the parameter-shift rule on the
    /// selected backend.
    ///
    /// Serves the cases [`Simulate::expectation_gradient`] declines: any
    /// backend with a native observable path, circuits containing `QftBlock`,
    /// widths past the statevector cap, and a noise model. It differentiates the
    /// same gate set (`Rx`, `Ry`, `Rz`, `Rzz`, `P`, `PauliRot`) at `1 + 2 * links`
    /// circuit evaluations against the adjoint's one, so the adjoint stays the
    /// better choice where it applies. A backend with no native observable path
    /// reports `BackendUnsupported` naming itself. Under a noise model the
    /// backend must be [`BackendKind::DensityMatrix`] or its device sibling,
    /// where every evaluation reads the exact mixture, or
    /// [`BackendKind::PauliPath`], which carries the channels through its
    /// truncated Pauli sum. The shift rule holds because the channels do not
    /// depend on the shifted angle. See
    /// [`gradient::run_expectation_gradient_shift`].
    #[inline]
    pub fn expectation_gradient_shift(
        self,
        hamiltonian: &PauliObservable,
        params: &crate::circuit::Parameters,
    ) -> Result<gradient::ExpectationGradient> {
        self.reject_saves("expectation_gradient_shift")?;
        let seed = self.seed_value();
        if self.noise_model.is_some()
            && !self.kind.is_density_matrix()
            && !matches!(self.kind, BackendKind::PauliPath { .. })
        {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: "a parameter-shift gradient under a noise model evaluates the exact \
                         mixed state, which only the density-matrix backend holds; select \
                         `BackendKind::DensityMatrix` or its device sibling, or drop the noise \
                         model"
                    .into(),
            });
        }
        gradient::shift_gradient(
            &self.kind,
            self.circuit,
            hamiltonian.terms(),
            params,
            self.noise_model,
            self.initial_state,
            seed,
            self.limits.as_ref(),
            self.require_exact,
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
        limits: None,
    }
}
