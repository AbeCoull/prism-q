//! Query descriptions and pre-execution route reports.

use super::*;

/// Terminal description for route inspection.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Query<'a> {
    Run,
    Shots(usize),
    Counts(usize),
    Marginals,
    Expectation(&'a [Vec<PauliTerm>]),
    Observable(&'a PauliObservable),
    StateVector,
    ReducedDensityMatrix(&'a [usize]),
    Entropy(&'a [usize]),
    Gradient {
        hamiltonian: &'a [(f64, Vec<PauliTerm>)],
        parameters: &'a crate::circuit::Parameters,
    },
    GradientShift {
        hamiltonian: &'a [(f64, Vec<PauliTerm>)],
        parameters: &'a crate::circuit::Parameters,
    },
}

/// Decision that selected a route, or a fallback actually taken during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RouteRule {
    ExplicitBackend,
    NoEntanglingGates,
    CliffordOnly,
    LargeCliffordBlocks,
    IndependentBlocks,
    PartialIndependence,
    OverStatevectorCapSparse,
    OverStatevectorCapMps,
    TemporalCliffordPrefix,
    CompiledCliffordSampler,
    CliffordRegister,
    TerminalStatevector,
    NativeSampling,
    PerShot,
    SpdExact,
    TensorExpectation,
    TensorOverMps,
    StartState,
    ExactMixture,
    DiagnosticReroute,
    SpdBudgetFallback,
    DeviceInitializationFallback,
    ParameterShift,
    PreparedBackend,
    Default,
}

/// Whether an estimate is a bound or leaves the allocation size unresolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryEstimateKind {
    Exact,
    UpperBound,
    Unknown,
}

/// Peak state and terminal buffer payload in bytes, excluding circuit/fusion storage,
/// allocator overhead and process memory. `Unknown` carries no byte count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub bytes: Option<u128>,
    pub kind: MemoryEstimateKind,
}

/// Accuracy contract known before any state has been evolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlannedExactness {
    Exact,
    Approximate { bound_reported: bool },
    Estimated { samples: usize },
}

/// Rejection from the same explicit-backend validation used by execution.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectReason {
    pub error: PrismError,
}

/// A possible runtime fallback, including the accuracy contract after it fires.
#[derive(Debug, Clone)]
pub struct PlanFallback {
    pub backend: ResolvedBackend,
    pub rule: RouteRule,
    pub exactness: PlannedExactness,
    pub placement: Placement,
    /// A failure only if execution takes this fallback.
    pub would_fail: Option<PrismError>,
}

/// Route inspection without execution; resource and capability failures remain inspectable.
#[derive(Debug, Clone)]
pub struct Plan {
    pub backend: ResolvedBackend,
    pub engine: Option<Engine>,
    pub placement: Placement,
    pub rule: RouteRule,
    pub peak_memory: MemoryEstimate,
    pub exactness: PlannedExactness,
    pub alternatives: Vec<(ResolvedBackend, RejectReason)>,
    pub fallbacks: Vec<PlanFallback>,
    /// Baseline, then plus/minus evaluations in parameter-link order for a shift gradient.
    pub evaluations: Vec<Plan>,
    /// A predictable execution failure; malformed query input is returned as `Err` instead.
    pub would_fail: Option<PrismError>,
}

impl Plan {
    pub(super) fn new(backend: ResolvedBackend, rule: RouteRule) -> Self {
        Self {
            backend,
            engine: None,
            placement: Placement::Host,
            rule,
            peak_memory: MemoryEstimate {
                bytes: None,
                kind: MemoryEstimateKind::Unknown,
            },
            exactness: PlannedExactness::Exact,
            alternatives: Vec::new(),
            fallbacks: Vec::new(),
            evaluations: Vec::new(),
            would_fail: None,
        }
    }
}

impl<SeedState> Simulate<'_, SeedState> {
    /// Inspect a terminal's route without executing the requested terminal.
    ///
    /// Runtime fallbacks are listed separately. Memory remains `Unknown` when a
    /// peak bound cannot be established without evolving state.
    ///
    /// # Examples
    /// ```
    /// use prism_q::{Circuit, Query, simulate};
    /// let circuit = Circuit::new(2, 0);
    /// let plan = simulate(&circuit).plan(Query::Run)?;
    /// assert!(plan.would_fail.is_none());
    /// # Ok::<(), prism_q::PrismError>(())
    /// ```
    pub fn plan(&self, query: Query<'_>) -> Result<Plan> {
        let limits = ResolvedLimits::resolve(self.limits.as_ref());
        match query {
            Query::Gradient {
                hamiltonian,
                parameters,
            }
            | Query::GradientShift {
                hamiltonian,
                parameters,
            } => {
                parameters.validate(self.circuit)?;
                for (coefficient, obs) in hamiltonian {
                    if !coefficient.is_finite() {
                        return Err(PrismError::InvalidParameter {
                            message: "Hamiltonian coefficients must be finite".into(),
                        });
                    }
                    validate_observable(obs, self.circuit.num_qubits)?;
                }
            }
            Query::Expectation(observables) => {
                for obs in observables {
                    validate_observable(obs, self.circuit.num_qubits)?;
                }
            }
            Query::Observable(observable) => {
                for (_, obs) in observable.terms() {
                    validate_observable(obs, self.circuit.num_qubits)?;
                }
            }
            Query::ReducedDensityMatrix(qubits) => {
                crate::backend::schmidt::validate_qubit_set(qubits, self.circuit.num_qubits)?
            }
            Query::Entropy(qubits) => {
                crate::backend::schmidt::validate_subsystem(qubits, self.circuit.num_qubits)?
            }
            _ => {}
        }
        if let Some(model) = self.noise_model {
            model.validate_for(self.circuit)?;
        }
        let decision = inspect(self, query, &limits);
        let mut plan = match decision {
            Ok(plan) => plan,
            Err(
                error @ (PrismError::InvalidParameter { .. }
                | PrismError::InvalidQubit { .. }
                | PrismError::InvalidClassicalBit { .. }),
            ) => return Err(error),
            Err(error) => {
                let mut plan = execution_plan(
                    &resolve(&self.kind, self.circuit, false, &limits),
                    &self.kind,
                    self.circuit,
                    &limits,
                );
                plan.would_fail = Some(error);
                plan
            }
        };
        if !matches!(query, Query::Run) && self.circuit.save_count() != 0 {
            plan.would_fail = Some(PrismError::IncompatibleBackend {
                backend: format!("{:?}", self.kind),
                reason: "save points are only returned by run".into(),
            });
        }
        if self.require_exact && (plan.exactness != PlannedExactness::Exact) {
            plan.would_fail = Some(PrismError::IncompatibleBackend {
                backend: approximation_name(&self.kind, plan.backend),
                reason: "require_exact declines a route that may approximate its result".into(),
            });
        }
        for (kind, backend) in [
            (BackendKind::ProductState, ResolvedBackend::ProductState),
            (BackendKind::Stabilizer, ResolvedBackend::Stabilizer),
            (BackendKind::DensityMatrix, ResolvedBackend::DensityMatrix),
            (BackendKind::Statevector, ResolvedBackend::Statevector),
            (BackendKind::Sparse, ResolvedBackend::Sparse),
            (BackendKind::Factored, ResolvedBackend::Factored),
            (
                BackendKind::FactoredStabilizer,
                ResolvedBackend::FactoredStabilizer,
            ),
            (
                BackendKind::Mps {
                    max_bond_dim: limits.auto_mps_bond_dim.value,
                },
                ResolvedBackend::Mps,
            ),
            (BackendKind::TensorNetwork, ResolvedBackend::TensorNetwork),
            (
                BackendKind::StochasticPauli { num_samples: 1000 },
                ResolvedBackend::StochasticPauli,
            ),
            (
                BackendKind::DeterministicPauli {
                    truncation: SpdTruncation::Threshold {
                        epsilon: 0.0,
                        max_terms: 0,
                    },
                },
                ResolvedBackend::DeterministicPauli,
            ),
            (
                BackendKind::PauliPath {
                    epsilon: 0.0,
                    max_terms: 0,
                },
                ResolvedBackend::PauliPath,
            ),
            (BackendKind::StabilizerRank, ResolvedBackend::StabilizerRank),
        ] {
            if let Err(error) = validate_explicit_backend(&kind, self.circuit, &limits) {
                plan.alternatives.push((backend, RejectReason { error }));
            }
        }
        if matches!(query, Query::Run) && plan.would_fail.is_none() {
            plan.would_fail = match plan.backend {
                ResolvedBackend::Statevector if plan.placement == Placement::Host => {
                    crate::backend::statevector_probability_len(
                        "statevector",
                        self.circuit.num_qubits,
                        &limits,
                    )
                    .err()
                }
                ResolvedBackend::TensorNetwork => crate::backend::tensor_probability_len(
                    "tensor_network",
                    self.circuit.num_qubits,
                    &limits,
                )
                .err(),
                #[cfg(feature = "distributed")]
                ResolvedBackend::Distributed => crate::backend::dense_probability_len(
                    "distributed_statevector",
                    self.circuit.num_qubits,
                    &limits,
                )
                .err(),
                _ => None,
            };
        }
        plan.peak_memory = estimate_buffers(&plan, self.circuit, query, self.noise_model.is_some());
        Ok(plan)
    }
}

fn estimate_buffers(
    plan: &Plan,
    circuit: &Circuit,
    query: Query<'_>,
    noisy: bool,
) -> MemoryEstimate {
    let unknown = MemoryEstimate {
        bytes: None,
        kind: MemoryEstimateKind::Unknown,
    };
    if noisy
        || plan.placement != Placement::Host
        || circuit.save_count() != 0
        || !plan.fallbacks.is_empty()
    {
        return unknown;
    }
    if !matches!(query, Query::Run | Query::StateVector) {
        return unknown;
    }
    let simple = circuit
        .instructions
        .iter()
        .all(|instruction| match instruction {
            Instruction::Gate { gate, .. } => matches!(
                gate,
                crate::gates::Gate::H
                    | crate::gates::Gate::X
                    | crate::gates::Gate::Y
                    | crate::gates::Gate::Z
                    | crate::gates::Gate::S
                    | crate::gates::Gate::Sdg
                    | crate::gates::Gate::T
                    | crate::gates::Gate::Tdg
                    | crate::gates::Gate::Rx(_)
                    | crate::gates::Gate::Ry(_)
                    | crate::gates::Gate::Rz(_)
                    | crate::gates::Gate::P(_)
                    | crate::gates::Gate::Cx
                    | crate::gates::Gate::Cz
                    | crate::gates::Gate::Swap
            ),
            Instruction::Barrier { .. } => true,
            _ => false,
        });
    if !simple {
        return unknown;
    }
    let Some(dim) = u32::try_from(circuit.num_qubits)
        .ok()
        .and_then(|n| 1u128.checked_shl(n))
    else {
        return unknown;
    };
    let classical = 2 * circuit.num_classical_bits as u128;
    let bytes = match plan.backend {
        ResolvedBackend::ProductState if matches!(query, Query::Run) => dim
            .checked_mul(8)
            .and_then(|output| output.checked_add(64 * circuit.num_qubits as u128 + classical)),
        ResolvedBackend::Statevector
            if circuit.num_qubits < crate::circuit::fusion::MIN_QUBITS_FOR_FUSION =>
        {
            dim.checked_mul(32)
                .and_then(|state_and_output| state_and_output.checked_add(classical))
        }
        _ => None,
    };
    match bytes {
        Some(bytes) => MemoryEstimate {
            bytes: Some(bytes),
            kind: MemoryEstimateKind::UpperBound,
        },
        None => unknown,
    }
}

pub(super) fn inspect<S>(
    request: &Simulate<'_, S>,
    query: Query<'_>,
    limits: &ResolvedLimits,
) -> Result<Plan> {
    let kind = &request.kind;
    let circuit = request.circuit;
    let n = circuit.num_qubits;
    if let Query::GradientShift {
        hamiltonian,
        parameters,
    } = query
    {
        return inspect_shift(request, hamiltonian, parameters, limits);
    }
    if matches!(
        query,
        Query::Expectation(_)
            | Query::Observable(_)
            | Query::StateVector
            | Query::ReducedDensityMatrix(_)
            | Query::Entropy(_)
            | Query::Gradient { .. }
    ) {
        require_unitary_circuit(kind, circuit, "the planned terminal requires")?;
    }
    if let BackendKind::PauliPath { epsilon, max_terms } = kind
        && matches!(query, Query::Expectation(_) | Query::Observable(_))
    {
        reject_pauli_path_initial_state(request.initial_state)?;
        if let Some(model) = request.noise_model {
            reject_readout_at(circuit, model, "expectation values")?;
        }
        let mut plan = Plan::new(ResolvedBackend::PauliPath, RouteRule::ExplicitBackend);
        if (SpdTruncation::Threshold {
            epsilon: *epsilon,
            max_terms: *max_terms,
        })
        .can_discard()
        {
            plan.exactness = PlannedExactness::Approximate {
                bound_reported: true,
            };
        }
        return Ok(plan);
    }
    if let Some(model) = request.noise_model {
        if matches!(query, Query::Shots(_) | Query::Counts(_)) {
            if request.initial_state.is_some() {
                return Err(reject_initial_state(
                    kind,
                    "shot sampling",
                    "noise trajectories start from the zero state",
                ));
            }
            return Ok(match decide_noisy(kind, circuit, model, limits)? {
                NoisyRoute::Mixture => backend_plan(
                    &plan_for_family(kind, Family::DensityMatrix, n, limits),
                    kind,
                    circuit,
                    limits,
                    RouteRule::ExactMixture,
                ),
                NoisyRoute::Compiled => {
                    let mut p = Plan::new(
                        ResolvedBackend::CompiledStabilizer,
                        RouteRule::CompiledCliffordSampler,
                    );
                    p.engine = Some(if noise::use_frame_sampler(circuit) {
                        Engine::FrameSampler
                    } else {
                        Engine::NoisyCompiledSampler
                    });
                    p
                }
                NoisyRoute::Trajectory(backend) => {
                    backend_plan(&backend, kind, circuit, limits, RouteRule::PerShot)
                }
            });
        }
        if matches!(query, Query::StateVector | Query::Gradient { .. }) {
            return Err(PrismError::IncompatibleBackend {
                backend: format!("{kind:?}"),
                reason: "this terminal requires a pure state without noise".into(),
            });
        }
        reject_readout_at(circuit, model, "the planned terminal")?;
        require_exact_mixture(kind, "the planned terminal")?;
        noise::validate_mixture_inputs(circuit, Some(model), request.initial_state)?;
        let backend = plan_for_family(kind, Family::DensityMatrix, n, limits);
        match query {
            Query::ReducedDensityMatrix(qubits) => check_diagnostic_width(
                "density_matrix",
                Diagnostic::ReducedDensityMatrix,
                qubits.len(),
                limits,
            )?,
            Query::Entropy(_) => {
                return Err(PrismError::BackendUnsupported {
                    backend: "density_matrix".into(),
                    operation: "entanglement entropy".into(),
                });
            }
            _ => {}
        }
        return Ok(backend_plan(
            &backend,
            kind,
            circuit,
            limits,
            RouteRule::ExactMixture,
        ));
    }
    if let Some(state) = request.initial_state {
        check_initial_state_len(state, n)?;
        match state {
            StartState::Amplitudes(a) => crate::backend::validate_initial_amplitudes(a)?,
            StartState::DensityMatrix(rho) => {
                crate::backend::validate_initial_density_matrix(rho)?;
            }
        }
        let backend = initial_state_plan(kind, n, state, limits)?;
        if matches!(query, Query::Gradient { .. }) {
            return Err(reject_initial_state(
                kind,
                "the adjoint gradient",
                "the backward pass reconstructs the zero state",
            ));
        }
        let mut plan = backend_plan(&backend, kind, circuit, limits, RouteRule::StartState);
        match query {
            Query::StateVector | Query::ReducedDensityMatrix(_) | Query::Entropy(_) => {
                require_unitary_circuit(kind, circuit, "state diagnostics require")?;
                let (diagnostic, width) = match query {
                    Query::StateVector => (Diagnostic::StateVector, n),
                    Query::ReducedDensityMatrix(q) => (Diagnostic::ReducedDensityMatrix, q.len()),
                    Query::Entropy(q) => (Diagnostic::Entropy, q.len()),
                    _ => unreachable!(),
                };
                check_diagnostic_width("plan", diagnostic, width, limits)?;
                if !plan_answers(&backend, diagnostic) {
                    plan.would_fail = Some(PrismError::BackendUnsupported {
                        backend: format!("{:?}", plan.backend),
                        operation: diagnostic.terminal().into(),
                    });
                }
            }
            _ => {}
        }
        return Ok(plan);
    }
    match query {
        Query::Run => inspect_probability(kind, circuit, limits),
        Query::Shots(count) | Query::Counts(count) => {
            #[cfg(feature = "distributed")]
            if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
                return Ok(backend_plan(
                    &resolve_backend(kind, circuit, false, limits),
                    kind,
                    circuit,
                    limits,
                    RouteRule::ExplicitBackend,
                ));
            }
            let folded = circuit.fold_static_guards();
            let circuit = folded.as_ref();
            match decide_shots(kind, circuit, count, limits)? {
                ShotRoute::Compiled(_) => {
                    let mut p = Plan::new(
                        ResolvedBackend::CompiledStabilizer,
                        RouteRule::CompiledCliffordSampler,
                    );
                    p.engine = Some(Engine::CompiledSampler);
                    Ok(p)
                }
                ShotRoute::StabilizerRank => Ok(Plan::new(
                    ResolvedBackend::StabilizerRank,
                    RouteRule::CliffordRegister,
                )),
                ShotRoute::TerminalStatevector => Ok(backend_plan(
                    &plan_for_family(kind, Family::Statevector, n, limits),
                    kind,
                    circuit,
                    limits,
                    RouteRule::TerminalStatevector,
                )),
                ShotRoute::Native(backend) => Ok(backend_plan(
                    &backend,
                    kind,
                    circuit,
                    limits,
                    RouteRule::NativeSampling,
                )),
                ShotRoute::TerminalProbabilities => {
                    let mut p = inspect_probability(kind, &circuit.without_measurements(), limits)?;
                    p.fallbacks.push(PlanFallback {
                        backend: p.backend,
                        rule: RouteRule::PerShot,
                        exactness: p.exactness,
                        placement: p.placement,
                        would_fail: None,
                    });
                    Ok(p)
                }
                ShotRoute::PerShot => {
                    let mut p = inspect_probability(kind, circuit, limits)?;
                    if kind.is_auto() && auto_samples_by_stabilizer_rank(circuit) {
                        p.backend = ResolvedBackend::StabilizerRank;
                        p.exactness = PlannedExactness::Exact;
                    }
                    p.rule = RouteRule::PerShot;
                    Ok(p)
                }
            }
        }
        Query::Marginals => inspect_marginal(
            decide_marginals(kind, circuit, limits)?,
            kind,
            circuit,
            limits,
        ),
        Query::Expectation(observables) => inspect_expectation(
            decide_expectations(kind, circuit, observables, limits)?,
            kind,
            circuit,
            limits,
        ),
        Query::Observable(observable) => {
            if grouped_statevector(kind, circuit, limits) {
                require_unitary_circuit(kind, circuit, "expectation values require")?;
                Ok(backend_plan(
                    &plan_for_family(kind, Family::Statevector, n, limits),
                    kind,
                    circuit,
                    limits,
                    if kind.is_auto() {
                        RouteRule::Default
                    } else {
                        RouteRule::ExplicitBackend
                    },
                ))
            } else {
                inspect_expectation(
                    decide_expectations(kind, circuit, &observable_vecs(observable), limits)?,
                    kind,
                    circuit,
                    limits,
                )
            }
        }
        Query::StateVector | Query::ReducedDensityMatrix(_) | Query::Entropy(_) => {
            require_unitary_circuit(kind, circuit, "state diagnostics require")?;
            let diagnostic = match query {
                Query::StateVector => Diagnostic::StateVector,
                Query::ReducedDensityMatrix(_) => Diagnostic::ReducedDensityMatrix,
                _ => Diagnostic::Entropy,
            };
            let (backend, rule) = decide_diagnostic(kind, circuit, None, diagnostic, limits)?;
            let mut p = backend_plan(&backend, kind, circuit, limits, rule);
            if !plan_answers(&backend, diagnostic) {
                p.would_fail = Some(PrismError::BackendUnsupported {
                    backend: format!("{:?}", p.backend),
                    operation: diagnostic.terminal().into(),
                });
            }
            match query {
                Query::StateVector if n > limits.export_cap() => {
                    p.would_fail = Some(crate::backend::schmidt::export_cap_exceeded(
                        "plan",
                        "a dense statevector".into(),
                        n,
                        limits,
                    ))
                }
                Query::ReducedDensityMatrix(q) => {
                    if let Err(error) = crate::backend::reduced_density::reduced_density_side(
                        "plan",
                        q.len(),
                        limits,
                    ) {
                        p.would_fail = Some(error);
                    }
                }
                _ => {}
            }
            Ok(p)
        }
        Query::Gradient { .. } => {
            require_unitary_circuit(kind, circuit, "adjoint gradients require")?;
            if !(kind.is_auto() || matches!(kind, BackendKind::Statevector)) {
                return Err(PrismError::IncompatibleBackend { backend: format!("{kind:?}"), reason: "adjoint gradients run on the statevector backend; select Auto or Statevector".into() });
            }
            if circuit.instructions.iter().any(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: crate::gates::Gate::QftBlock { .. },
                        ..
                    }
                )
            }) {
                return Err(PrismError::IncompatibleBackend { backend: "Statevector".into(), reason: "adjoint gradients do not support QftBlock; expand it to primitive gates first".into() });
            }
            Ok(backend_plan(
                &plan_for_family(&BackendKind::Statevector, Family::Statevector, n, limits),
                kind,
                circuit,
                limits,
                if kind.is_auto() {
                    RouteRule::Default
                } else {
                    RouteRule::ExplicitBackend
                },
            ))
        }
        Query::GradientShift { .. } => unreachable!("handled above"),
    }
}

fn inspect_probability(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<Plan> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        return Ok(execution_plan(
            &resolve(kind, circuit, false, limits),
            kind,
            circuit,
            limits,
        ));
    }
    Ok(match plan_probability_route(kind, circuit, limits) {
        ProbabilityRoute::FactoredStabilizer => Plan::new(
            ResolvedBackend::FactoredStabilizer,
            RouteRule::LargeCliffordBlocks,
        ),
        ProbabilityRoute::Decomposed(components) => {
            let mut plan = Plan::new(ResolvedBackend::Decomposed, RouteRule::IndependentBlocks);
            let mut all_device = true;
            for (block, _, _) in circuit.partition_subcircuits(&components) {
                let child = inspect_probability(kind, &block, limits)?;
                all_device &= child.placement == Placement::Device;
                if child.exactness != PlannedExactness::Exact {
                    plan.exactness = child.exactness;
                }
                if plan.would_fail.is_none() {
                    plan.would_fail = child.would_fail;
                }
                plan.fallbacks.extend(child.fallbacks);
            }
            if all_device {
                plan.placement = Placement::Device;
            }
            plan
        }
        ProbabilityRoute::TemporalClifford {
            has_partial_independence,
        } => {
            let mut p = execution_plan(
                &resolve(kind, circuit, has_partial_independence, limits),
                kind,
                circuit,
                limits,
            );
            if plan_temporal_clifford(kind, circuit, limits).is_some() {
                p = backend_plan(
                    &plan_for_family(kind, Family::Statevector, circuit.num_qubits, limits),
                    kind,
                    circuit,
                    limits,
                    RouteRule::TemporalCliffordPrefix,
                );
            }
            p
        }
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => execution_plan(
            &resolve(kind, circuit, has_partial_independence, limits),
            kind,
            circuit,
            limits,
        ),
    })
}

fn execution_plan(
    route: &ExecutionPlan,
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Plan {
    match route {
        ExecutionPlan::Backend(backend) => backend_plan(
            backend,
            kind,
            circuit,
            limits,
            backend_rule(kind, circuit, backend),
        ),
        ExecutionPlan::StabilizerRank => {
            Plan::new(ResolvedBackend::StabilizerRank, RouteRule::ExplicitBackend)
        }
        ExecutionPlan::StochasticPauli { num_samples } => {
            let mut p = Plan::new(ResolvedBackend::StochasticPauli, RouteRule::ExplicitBackend);
            p.exactness = PlannedExactness::Estimated {
                samples: *num_samples,
            };
            p.would_fail = Some(PrismError::IncompatibleBackend {
                backend: "StochasticPauli".into(),
                reason: "use marginals or expectation values".into(),
            });
            p
        }
        ExecutionPlan::DeterministicPauli { truncation } => {
            let mut p = pauli_plan(*truncation, RouteRule::ExplicitBackend);
            p.would_fail = Some(PrismError::IncompatibleBackend {
                backend: "DeterministicPauli".into(),
                reason: "use marginals or expectation values".into(),
            });
            p
        }
        ExecutionPlan::PauliPath => {
            let mut p = Plan::new(ResolvedBackend::PauliPath, RouteRule::ExplicitBackend);
            p.would_fail = Some(reject_pauli_path("a single run"));
            p
        }
    }
}

fn backend_plan(
    backend: &BackendPlan,
    _kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
    rule: RouteRule,
) -> Plan {
    let mut p = Plan::new(backend.resolved(), rule);
    if backend.is_gpu() {
        p.placement = Placement::Device;
    }
    match backend {
        BackendPlan::Mps { .. } if !circuit.is_sparse_friendly() => {
            p.exactness = PlannedExactness::Approximate {
                bound_reported: true,
            }
        }
        BackendPlan::TensorNetwork { tolerance: Some(_) } => {
            p.exactness = PlannedExactness::Approximate {
                bound_reported: true,
            }
        }
        _ => {}
    }
    use crate::limits::LimitKnob;
    let allocation = match backend {
        BackendPlan::Statevector { .. } if !backend.is_gpu() => {
            Some((limits.statevector_qubits, LimitKnob::StatevectorQubits))
        }
        BackendPlan::DensityMatrix { .. } if !backend.is_gpu() => {
            Some((limits.density_matrix_qubits, LimitKnob::DensityMatrixQubits))
        }
        _ => None,
    };
    if let Some((cap, knob)) = allocation {
        p.would_fail = crate::backend::check_state_allocation(
            &format!("{:?}", backend.resolved()),
            circuit.num_qubits,
            cap,
            knob,
        )
        .err();
    }
    if matches!(backend, BackendPlan::Sparse) {
        p.would_fail = crate::backend::sparse::check_index_width(circuit.num_qubits).err();
    }
    #[cfg(feature = "gpu")]
    if matches!(
        backend,
        BackendPlan::Statevector {
            accel: dispatch::Accel::Gpu { soft: true, .. }
        } | BackendPlan::Stabilizer {
            accel: dispatch::Accel::Gpu { soft: true, .. },
            ..
        } | BackendPlan::DensityMatrix {
            accel: dispatch::Accel::Gpu { soft: true, .. }
        }
    ) {
        p.fallbacks.push(PlanFallback {
            backend: p.backend,
            rule: RouteRule::DeviceInitializationFallback,
            exactness: p.exactness,
            placement: Placement::Host,
            would_fail: None,
        });
    }
    p
}

fn pauli_plan(truncation: SpdTruncation, rule: RouteRule) -> Plan {
    let mut p = Plan::new(ResolvedBackend::DeterministicPauli, rule);
    if truncation.can_discard() {
        p.exactness = PlannedExactness::Approximate {
            bound_reported: false,
        };
    }
    p
}

fn inspect_expectation(
    route: ExpectationRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<Plan> {
    let rule = expectation_rule(&route, kind);
    Ok(match route {
        ExpectationRoute::Stochastic(samples) => {
            let mut p = Plan::new(ResolvedBackend::StochasticPauli, rule);
            p.exactness = PlannedExactness::Estimated { samples };
            p
        }
        ExpectationRoute::Deterministic(truncation) => pauli_plan(truncation, rule),
        ExpectationRoute::Tensor(planned, _) => {
            let mut plan = Plan::new(ResolvedBackend::TensorNetwork, rule);
            plan.would_fail = planned.failure().cloned();
            plan
        }
        ExpectationRoute::Statevector => backend_plan(
            &plan_for_family(kind, Family::Statevector, circuit.num_qubits, limits),
            kind,
            circuit,
            limits,
            rule,
        ),
        ExpectationRoute::Native(backend) => backend_plan(&backend, kind, circuit, limits, rule),
    })
}

fn inspect_marginal(
    route: MarginalRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<Plan> {
    match route {
        MarginalRoute::Stochastic(samples) => {
            let mut p = Plan::new(ResolvedBackend::StochasticPauli, RouteRule::ExplicitBackend);
            p.exactness = PlannedExactness::Estimated { samples };
            Ok(p)
        }
        MarginalRoute::Deterministic(truncation) => {
            Ok(pauli_plan(truncation, RouteRule::ExplicitBackend))
        }
        MarginalRoute::Spd { budget } => {
            let mut p = Plan::new(ResolvedBackend::DeterministicPauli, RouteRule::SpdExact);
            if budget.is_some() {
                let fallback = inspect_marginal(
                    decide_marginals_after_spd(kind, circuit, limits)?,
                    kind,
                    circuit,
                    limits,
                )?;
                p.fallbacks.push(PlanFallback {
                    backend: fallback.backend,
                    rule: RouteRule::SpdBudgetFallback,
                    exactness: fallback.exactness,
                    placement: fallback.placement,
                    would_fail: fallback.would_fail,
                });
            }
            Ok(p)
        }
        MarginalRoute::Tensor(planned, rule) => {
            let mut plan = Plan::new(ResolvedBackend::TensorNetwork, rule);
            plan.would_fail = planned.failure().cloned();
            Ok(plan)
        }
        MarginalRoute::Native(backend) => Ok(backend_plan(
            &backend,
            kind,
            circuit,
            limits,
            backend_rule(kind, circuit, &backend),
        )),
        MarginalRoute::Probabilities => inspect_probability(kind, circuit, limits),
    }
}

pub(super) fn require_exact_expectation(
    route: &ExpectationRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<()> {
    let approximate = match route {
        ExpectationRoute::Stochastic(_) => true,
        ExpectationRoute::Deterministic(t) => t.can_discard(),
        ExpectationRoute::Native(b) => {
            backend_plan(b, kind, circuit, limits, RouteRule::Default).exactness
                != PlannedExactness::Exact
        }
        _ => false,
    };
    let backend = match route {
        ExpectationRoute::Stochastic(_) => "StochasticPauli".into(),
        ExpectationRoute::Deterministic(_) => "DeterministicPauli".into(),
        ExpectationRoute::Native(b) => approximation_name(kind, b.resolved()),
        _ => String::new(),
    };
    reject_predicted_route(approximate.then_some(backend.as_str()))
}

pub(super) fn require_exact_marginal(
    route: &MarginalRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<()> {
    let approximate = match route {
        MarginalRoute::Stochastic(_) => true,
        MarginalRoute::Deterministic(t) => t.can_discard(),
        MarginalRoute::Native(b) => {
            backend_plan(b, kind, circuit, limits, RouteRule::Default).exactness
                != PlannedExactness::Exact
        }
        MarginalRoute::Probabilities => {
            approximate_route_name(kind, circuit, false, limits).is_some()
        }

        _ => false,
    };
    let backend = match route {
        MarginalRoute::Stochastic(_) => "StochasticPauli".into(),
        MarginalRoute::Deterministic(_) => "DeterministicPauli".into(),
        MarginalRoute::Native(b) => approximation_name(kind, b.resolved()),
        MarginalRoute::Probabilities => approximate_route_name(kind, circuit, false, limits)
            .unwrap_or_default()
            .into(),
        _ => String::new(),
    };
    reject_predicted_route(approximate.then_some(backend.as_str()))
}

fn inspect_shift<S>(
    request: &Simulate<'_, S>,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    parameters: &crate::circuit::Parameters,
    limits: &ResolvedLimits,
) -> Result<Plan> {
    let observables: Vec<_> = hamiltonian.iter().map(|(_, terms)| terms.clone()).collect();
    let prepared = if request.initial_state.is_none()
        && request.noise_model.is_none()
        && request.circuit.num_qubits >= crate::circuit::fusion::MIN_QUBITS_FOR_FUSION
    {
        prepared_route(&request.kind, request.circuit, request.limits.as_ref())
    } else {
        None
    };
    let inspect_evaluation = |circuit: &Circuit| -> Result<Plan> {
        if let Some(prepared) = &prepared {
            match prepared.values {
                PreparedValues::Held => {
                    return Ok(backend_plan(
                        &prepared.plan,
                        &request.kind,
                        circuit,
                        limits,
                        RouteRule::PreparedBackend,
                    ));
                }
                PreparedValues::TensorFirst => {
                    if let Some(result) = decide_tensor_expectations(circuit, &observables, limits)
                    {
                        let planned = result?;
                        let mut plan =
                            Plan::new(ResolvedBackend::TensorNetwork, RouteRule::TensorExpectation);
                        plan.would_fail = planned.failure().cloned();
                        return Ok(plan);
                    }
                    return Ok(backend_plan(
                        &prepared.plan,
                        &request.kind,
                        circuit,
                        limits,
                        RouteRule::PreparedBackend,
                    ));
                }
                PreparedValues::Simulate => {}
            }
        }
        let mut evaluation = simulate(circuit).backend(request.kind.clone());
        if let Some(limits) = request.limits {
            evaluation = evaluation.limits(limits);
        }
        if let Some(noise) = request.noise_model {
            evaluation = evaluation.noise(noise);
        }
        if let Some(state) = request.initial_state {
            evaluation = match state {
                StartState::Amplitudes(a) => evaluation.initial_state(a),
                StartState::DensityMatrix(rho) => evaluation.initial_density_matrix(rho),
            };
        }
        inspect(&evaluation, Query::Expectation(&observables), limits)
    };
    let baseline = inspect_evaluation(request.circuit)?;
    let mut plan = baseline.clone();
    plan.rule = RouteRule::ParameterShift;
    plan.evaluations.push(baseline);
    for link in parameters.links() {
        for delta in [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
            let mut shifted = request.circuit.clone();
            *crate::circuit::parameter::angle_mut(&mut shifted.instructions[link.instruction]) +=
                delta;
            let evaluation = inspect_evaluation(&shifted)?;
            if evaluation.exactness != PlannedExactness::Exact {
                plan.exactness = evaluation.exactness;
            }
            if plan.would_fail.is_none() {
                plan.would_fail = evaluation.would_fail.clone();
            }
            plan.evaluations.push(evaluation);
        }
    }
    Ok(plan)
}

pub(super) fn approximation_name(kind: &BackendKind, backend: ResolvedBackend) -> String {
    if backend == ResolvedBackend::TensorNetwork
        && matches!(kind, BackendKind::TensorNetworkBounded { .. })
    {
        "TensorNetworkBounded".into()
    } else {
        format!("{backend:?}")
    }
}
