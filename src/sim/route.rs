//! Shared route decisions and prepared contraction plans.

use super::*;

pub(super) enum ProbabilityRoute {
    FactoredStabilizer,
    Decomposed(Vec<Vec<usize>>),
    /// The temporal-Clifford predicate holds; `run_route` builds the plan and
    /// falls back to direct resolution should the split come back empty.
    TemporalClifford {
        has_partial_independence: bool,
    },
    Direct {
        has_partial_independence: bool,
    },
}

pub(super) fn plan_probability_route(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> ProbabilityRoute {
    let (decompose, has_partial_independence) = analyze_independence(circuit);
    if let Some((components, clifford_only)) = decompose {
        let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
        let auto_takes_it = kind.is_auto()
            && circuit.num_qubits >= MIN_FACTORED_STABILIZER_QUBITS
            && max_block >= MIN_BLOCK_FOR_FACTORED_STAB;
        if clifford_only && (auto_takes_it || matches!(kind, BackendKind::FactoredStabilizer)) {
            return ProbabilityRoute::FactoredStabilizer;
        }
        return ProbabilityRoute::Decomposed(components);
    }
    if has_temporal_clifford_opportunity(kind, circuit, limits) {
        return ProbabilityRoute::TemporalClifford {
            has_partial_independence,
        };
    }
    ProbabilityRoute::Direct {
        has_partial_independence,
    }
}

/// True when the auto probability route falls through to direct family
/// resolution and that resolver picks the CPU statevector.
pub(super) fn auto_terminal_statevector_candidate(
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> bool {
    match plan_probability_route(&BackendKind::Auto, circuit, limits) {
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => auto_selects_cpu_statevector(circuit, has_partial_independence, limits),
        _ => false,
    }
}

pub(super) fn terminal_statevector_candidate(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> bool {
    if kind.is_auto() {
        return auto_terminal_statevector_candidate(circuit, limits);
    }
    match kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => false,
    }
}

/// Width from which an `Auto` expectation terminal tries the scalar tensor
/// path first. Below it the statevector run takes a few milliseconds and the
/// bounded dry run that decides the route would cost as much as the run.
/// Bracketed by `expectation/pauli_sum` (`16` against `18` and `20`).
#[cfg(not(prism_lowered_thresholds))]
pub(super) const TENSOR_ROUTE_MIN_QUBITS: usize = 18;
#[cfg(prism_lowered_thresholds)]
pub(super) const TENSOR_ROUTE_MIN_QUBITS: usize = 4;

/// The scalar tensor route for an `Auto` expectation terminal under the
/// statevector cap: `None` below [`TENSOR_ROUTE_MIN_QUBITS`], on a circuit the
/// scalar network cannot hold, or when any observable's bounded greedy plan
/// crosses the peak bound, so the caller takes its dense route. Observables
/// are validated first, so a bad one errors before any dry run.
pub(super) fn decide_tensor_expectations(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    limits: &ResolvedLimits,
) -> Option<Result<crate::backend::tensornetwork::BoundedExpectations>> {
    if circuit.num_qubits < TENSOR_ROUTE_MIN_QUBITS || observables.is_empty() {
        return None;
    }
    for observable in observables {
        if let Err(e) = validate_observable(observable, circuit.num_qubits) {
            return Some(Err(e));
        }
    }
    let planned = match crate::backend::tensornetwork::plan_bounded_expectations(
        circuit,
        observables,
        crate::backend::tensornetwork::AUTO_EXPECTATION_PEAK_BOUND,
    )? {
        Ok(planned) => planned,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(planned.with_limits(limits)))
}

/// MPS work per observable, in [`mps_apply_cost`] units, from which an `Auto`
/// expectation above the statevector cap plans the scalar tensor route before
/// running MPS. About 2 ms of MPS time, the most one observable's dry run took
/// across the calibration circuits (0.1 to 1.9 ms), so a declined or losing
/// plan costs at most about the MPS run it tried to replace.
/// Bracketed by `auto/expectation_above_cap` (`hea_l2/64` and `su4_brick_d4/64` against `su4_brick_d6/64`).
const TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE: f64 = 1.5e5;

/// [`mps_apply_cost`] units one planned tensor-network element costs: about
/// 40 ns per element contracted against 15 ns per unit.
/// No bench row brackets this yet.
const TENSOR_ELEMENT_MPS_COST: f64 = 3.0;

/// The scalar tensor route for an `Auto` expectation above the statevector cap,
/// where the alternative is the MPS `Auto` resolves to rather than a dense run.
/// `None` when `Auto` resolves to another backend, when the MPS estimate is
/// under [`TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE`] per observable, when a plan
/// crosses the peak bound, or when the planned contraction costs more than the
/// MPS estimate.
pub(super) fn decide_tensor_over_mps(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    limits: &ResolvedLimits,
) -> Option<Result<crate::backend::tensornetwork::BoundedExpectations>> {
    if observables.is_empty() {
        return None;
    }
    let mps_cost = mps_apply_cost(circuit, limits.auto_mps_bond_dim.value);
    if mps_cost < TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE * observables.len() as f64 {
        return None;
    }
    let (_, has_partial_independence) = analyze_independence(circuit);
    let ExecutionPlan::Backend(plan) = resolve(
        &BackendKind::Auto,
        circuit,
        has_partial_independence,
        limits,
    ) else {
        return None;
    };
    if !matches!(plan, BackendPlan::Mps { .. }) {
        return None;
    }
    for observable in observables {
        if let Err(e) = validate_observable(observable, circuit.num_qubits) {
            return Some(Err(e));
        }
    }
    let planned = match crate::backend::tensornetwork::plan_bounded_expectations(
        circuit,
        observables,
        crate::backend::tensornetwork::AUTO_EXPECTATION_PEAK_BOUND,
    )? {
        Ok(planned) => planned,
        Err(e) => return Some(Err(e)),
    };
    if planned.total_elements() as f64 * TENSOR_ELEMENT_MPS_COST >= mps_cost {
        return None;
    }
    Some(Ok(planned.with_limits(limits)))
}

pub(super) fn contract_tensor(
    planned: crate::backend::tensornetwork::BoundedExpectations,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    planned.contract(limits).map(|values| {
        analytic_expectations(values, RunMetadata::exact(ResolvedBackend::TensorNetwork))
    })
}

pub(super) enum ShotRoute {
    Compiled(Option<Circuit>),
    TerminalStatevector,
    Native(BackendPlan),
    TerminalProbabilities,
    StabilizerRank,
    PerShot,
}

pub(super) fn decide_native(
    kind: &BackendKind,
    circuit: &Circuit,
    sampling: bool,
    limits: &ResolvedLimits,
) -> Result<Option<BackendPlan>> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    let (decomposed, partial) = match plan_probability_route(kind, circuit, limits) {
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => (false, has_partial_independence),
        ProbabilityRoute::Decomposed(_) => (true, false),
        _ => return Ok(None),
    };
    let ExecutionPlan::Backend(plan) = resolve(kind, circuit, partial, limits) else {
        return Ok(None);
    };
    if decomposed && !matches!(plan, BackendPlan::ProductState) {
        return Ok(None);
    }
    let capable = if sampling {
        plan.supports_native_sampling()
    } else {
        plan.supports_pauli_expectation()
    };
    Ok(capable.then_some(plan))
}

pub(super) fn decide_shots(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    limits: &ResolvedLimits,
) -> Result<ShotRoute> {
    if should_use_compiled_clifford_sampling(kind, circuit, num_shots) {
        return Ok(ShotRoute::Compiled(None));
    }
    if should_use_deferred_clifford_sampling(kind, circuit, num_shots)
        && let Ok(deferred) = compiled::defer_measure_reset_circuit(circuit)
    {
        return Ok(ShotRoute::Compiled(Some(deferred)));
    }
    if kind.is_auto() && auto_prefers_clifford_register(circuit) {
        return Ok(ShotRoute::StabilizerRank);
    }
    if circuit.has_terminal_measurements_only()
        && !circuit.measurement_map().is_empty()
        && terminal_statevector_candidate(kind, &circuit.without_measurements(), limits)
    {
        return Ok(ShotRoute::TerminalStatevector);
    }
    if matches!(kind, BackendKind::StabilizerRank) && circuit.has_t_gates()
        || kind.is_auto() && auto_samples_wide_by_stabilizer_rank(circuit)
    {
        return Ok(ShotRoute::StabilizerRank);
    }
    if circuit.has_terminal_measurements_only()
        || kind.is_density_matrix() && circuit.has_terminal_measurements_under_reset_channel()
    {
        let stripped = circuit.without_measurements();
        return Ok(match decide_native(kind, &stripped, true, limits)? {
            Some(plan) => ShotRoute::Native(plan),
            None => ShotRoute::TerminalProbabilities,
        });
    }
    Ok(ShotRoute::PerShot)
}

pub(super) enum ExpectationRoute {
    Stochastic(usize),
    Deterministic(SpdTruncation),
    Tensor(
        crate::backend::tensornetwork::BoundedExpectations,
        RouteRule,
    ),
    Statevector,
    Native(BackendPlan),
}

pub(super) fn decide_expectations(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    limits: &ResolvedLimits,
) -> Result<ExpectationRoute> {
    require_unitary_circuit(kind, circuit, "expectation values require")?;
    for observable in observables {
        validate_observable(observable, circuit.num_qubits)?;
    }
    match kind {
        BackendKind::StochasticPauli { num_samples } => {
            Ok(ExpectationRoute::Stochastic(*num_samples))
        }
        BackendKind::DeterministicPauli { truncation } => {
            Ok(ExpectationRoute::Deterministic(*truncation))
        }
        _ if (kind.is_auto() || kind.is_stabilizer_family()) && circuit.is_clifford_only() => {
            Ok(ExpectationRoute::Deterministic(SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            }))
        }
        _ if kind.is_auto() => {
            let above_cap = circuit.num_qubits > limits.statevector_qubits.value;
            let tensor = if above_cap {
                decide_tensor_over_mps(circuit, observables, limits)
            } else {
                decide_tensor_expectations(circuit, observables, limits)
            };
            if let Some(planned) = tensor {
                return Ok(ExpectationRoute::Tensor(
                    planned?,
                    if above_cap {
                        RouteRule::TensorOverMps
                    } else {
                        RouteRule::TensorExpectation
                    },
                ));
            }
            if !above_cap {
                return Ok(ExpectationRoute::Statevector);
            }
            decide_native_expectations(kind, circuit, limits).map(ExpectationRoute::Native)
        }
        BackendKind::Statevector => Ok(ExpectationRoute::Statevector),
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => Ok(ExpectationRoute::Statevector),
        BackendKind::TensorNetwork => {
            validate_explicit_backend(kind, circuit, limits)?;
            Ok(ExpectationRoute::Tensor(
                crate::backend::tensornetwork::plan_expectations(circuit, observables, limits)?,
                RouteRule::ExplicitBackend,
            ))
        }
        _ => decide_native_expectations(kind, circuit, limits).map(ExpectationRoute::Native),
    }
}

pub(super) fn decide_native_expectations(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<BackendPlan> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    let (_, partial) = analyze_independence(circuit);
    let ExecutionPlan::Backend(plan) = resolve(kind, circuit, partial, limits) else {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "expectation values need a backend that holds a state; the stabilizer-rank route returns probabilities only".into(),
        });
    };
    if !plan.supports_pauli_expectation() {
        return Err(PrismError::BackendUnsupported {
            backend: format!("{:?}", plan.resolved()),
            operation: "Pauli expectation values".into(),
        });
    }
    Ok(plan)
}

pub(super) fn grouped_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> bool {
    match kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => {
            kind.is_auto()
                && !circuit.is_clifford_only()
                && circuit.num_qubits <= limits.statevector_qubits.value
        }
    }
}

pub(super) fn expectation_rule(route: &ExpectationRoute, kind: &BackendKind) -> RouteRule {
    match route {
        ExpectationRoute::Tensor(_, rule) => *rule,
        ExpectationRoute::Deterministic(truncation)
            if !truncation.can_discard() && kind.is_auto() =>
        {
            RouteRule::SpdExact
        }
        _ if !kind.is_auto() => RouteRule::ExplicitBackend,
        _ => RouteRule::Default,
    }
}

pub(super) enum MarginalRoute {
    Stochastic(usize),
    Deterministic(SpdTruncation),
    Spd {
        budget: Option<usize>,
    },
    Tensor(
        crate::backend::tensornetwork::BoundedExpectations,
        RouteRule,
    ),
    Native(BackendPlan),
    Probabilities,
}

pub(super) fn decide_marginals(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<MarginalRoute> {
    match kind {
        BackendKind::StochasticPauli { num_samples } => {
            validate_pauli_marginal_backend(kind, circuit)?;
            return Ok(MarginalRoute::Stochastic(*num_samples));
        }
        BackendKind::DeterministicPauli { truncation } => {
            validate_pauli_marginal_backend(kind, circuit)?;
            return Ok(MarginalRoute::Deterministic(*truncation));
        }
        _ => {}
    }
    if kind.is_auto()
        && supports_pauli_marginal_backend(circuit)
        && circuit.has_t_gates()
        && circuit.num_qubits >= MIN_QUBITS_FOR_SPD_AUTO
    {
        return Ok(MarginalRoute::Spd {
            budget: (circuit.num_qubits <= limits.statevector_qubits.value)
                .then(|| auto_spd_work_budget(circuit)),
        });
    }
    decide_marginals_after_spd(kind, circuit, limits)
}

pub(super) fn decide_marginals_after_spd(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<MarginalRoute> {
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        return Ok(MarginalRoute::Native(resolve_backend(
            kind, circuit, false, limits,
        )));
    }
    if kind.is_auto() && !has_nonunitary_or_classical_ops(circuit) {
        let observables = (0..circuit.num_qubits)
            .map(|q| vec![PauliTerm::z(q)])
            .collect::<Vec<_>>();
        let above = circuit.num_qubits > limits.statevector_qubits.value;
        let tensor = if above {
            decide_tensor_over_mps(circuit, &observables, limits)
        } else {
            decide_tensor_expectations(circuit, &observables, limits)
        };
        if let Some(planned) = tensor {
            return Ok(MarginalRoute::Tensor(
                planned?,
                if above {
                    RouteRule::TensorOverMps
                } else {
                    RouteRule::TensorExpectation
                },
            ));
        }
    }
    Ok(match decide_native(kind, circuit, false, limits)? {
        Some(plan) => MarginalRoute::Native(plan),
        None => MarginalRoute::Probabilities,
    })
}

pub(super) fn decide_diagnostic(
    kind: &BackendKind,
    circuit: &Circuit,
    initial_state: Option<StartState<'_>>,
    diagnostic: Diagnostic,
    limits: &ResolvedLimits,
) -> Result<(BackendPlan, RouteRule)> {
    if let Some(state) = initial_state {
        check_initial_state_len(state, circuit.num_qubits)?;
        return Ok((
            initial_state_plan(kind, circuit.num_qubits, state, limits)?,
            RouteRule::StartState,
        ));
    }
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    let (_, partial) = analyze_independence(circuit);
    let mut plan = match resolve(kind, circuit, partial, limits) {
        ExecutionPlan::Backend(plan) => plan,
        ExecutionPlan::StabilizerRank => {
            return Err(stateless_route(kind, diagnostic, "stabilizer-rank"));
        }
        ExecutionPlan::StochasticPauli { .. } => {
            return Err(stateless_route(kind, diagnostic, "stochastic Pauli"));
        }
        ExecutionPlan::DeterministicPauli { .. } => {
            return Err(stateless_route(kind, diagnostic, "deterministic Pauli"));
        }
        ExecutionPlan::PauliPath => return Err(stateless_route(kind, diagnostic, "Pauli path")),
    };
    let mut rule = backend_rule(kind, circuit, &plan);
    if kind.is_auto()
        && !plan_answers(&plan, diagnostic)
        && circuit.num_qubits <= limits.statevector_qubits.value
    {
        plan = plan_for_family(kind, Family::Statevector, circuit.num_qubits, limits);
        rule = RouteRule::DiagnosticReroute;
    }
    Ok((plan, rule))
}

pub(super) fn backend_rule(kind: &BackendKind, circuit: &Circuit, plan: &BackendPlan) -> RouteRule {
    if !kind.is_auto() {
        return RouteRule::ExplicitBackend;
    }
    match plan {
        BackendPlan::ProductState => RouteRule::NoEntanglingGates,
        BackendPlan::Stabilizer { .. } => RouteRule::CliffordOnly,
        BackendPlan::FactoredStabilizer => RouteRule::LargeCliffordBlocks,
        BackendPlan::Sparse => RouteRule::OverStatevectorCapSparse,
        BackendPlan::Mps { .. } => RouteRule::OverStatevectorCapMps,
        BackendPlan::Factored if circuit.has_entangling_gates() => RouteRule::PartialIndependence,
        _ => RouteRule::Default,
    }
}

pub(super) enum NoisyRoute {
    Mixture,
    Compiled,
    Trajectory(BackendPlan),
}

pub(super) fn decide_noisy(
    kind: &BackendKind,
    circuit: &Circuit,
    noise_model: &NoiseModel,
    limits: &ResolvedLimits,
) -> Result<NoisyRoute> {
    noise_model.validate_for(circuit)?;

    if kind.is_density_matrix() {
        noise::validate_mixture_inputs(circuit, Some(noise_model), None)?;
        return Ok(NoisyRoute::Mixture);
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

    if is_stabilizer_kind && !noise_model.has_only_pauli_channels() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "stabilizer backends only support Pauli/depolarizing noise; use {} for amplitude damping, phase damping, thermal relaxation, or custom Kraus",
                BackendKind::general_noise_backend_names()
            ),
        });
    }

    if !noise_model.has_only_pauli_channels() && !kind.supports_general_noise() {
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
        validate_explicit_backend(kind, circuit, limits)?;
    }

    if noise_model.has_only_pauli_channels() {
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
            return Ok(NoisyRoute::Compiled);
        }
    }

    let plan = if kind.is_auto() && !noise_model.has_only_pauli_channels() {
        general_noise_plan(kind, circuit, noise_model, limits)
    } else {
        resolve_backend(kind, circuit, false, limits)
    };
    // Noise events are indexed per instruction, so trajectories apply the
    // stream raw and the Pauli-rotation lowering pass cannot run. The
    // statevector applies the gate natively (its device path lowers inline);
    // any other backend without the kernel is rejected before a shot starts.
    let has_pauli_rot = crate::circuit::any_gate(&circuit.instructions, &mut |gate| {
        matches!(gate, crate::gates::Gate::PauliRot(_))
    });
    if has_pauli_rot && !matches!(plan, BackendPlan::Statevector { .. }) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{:?}", plan.resolved()),
            reason: "noisy trajectories apply the instruction stream raw so noise events \
                     stay aligned to it, which leaves no room for the Pauli-rotation \
                     lowering this backend needs; run on the statevector, or expand the \
                     rotations with circuit::expand_pauli_rotations and attach the noise \
                     model to the expanded circuit"
                .into(),
        });
    }
    // A correlated two-qubit Kraus branch reads a two-qubit reduced density
    // matrix. Without this the model clears every routing gate and fails part
    // way through the first shot, on a backend an Auto route may have picked
    // for the caller.
    let answers_pair = match plan {
        BackendPlan::Statevector { .. }
        | BackendPlan::Sparse
        | BackendPlan::Factored
        | BackendPlan::Mps { .. } => true,
        #[cfg(feature = "distributed")]
        BackendPlan::Distributed(_) => true,
        _ => false,
    };
    if noise_model.has_two_qubit_kraus() && !answers_pair {
        #[cfg(feature = "distributed")]
        const PAIR_ROUTES: &str = "a statevector, sparse, factored, MPS or distributed statevector";
        #[cfg(not(feature = "distributed"))]
        const PAIR_ROUTES: &str = "a statevector, sparse, factored or MPS";
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{:?}", plan.resolved()),
            reason: format!(
                "a two-qubit Kraus channel needs the pair's reduced density matrix its \
                 branch probabilities are drawn from, and a kernel for the branch operator, \
                 which is not unitary. A tableau and a product state have no such kernel, \
                 and a tensor network traces out no arbitrary pair; run on {PAIR_ROUTES} \
                 route, or evaluate the channel exactly on BackendKind::DensityMatrix"
            ),
        });
    }
    // A two-qubit Kraus branch on the distributed statevector reads the pair's block
    // under the export cap, gathering the whole state when the pair cannot sit in the
    // local slice.
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = kind
        && noise_model.has_two_qubit_kraus()
    {
        let name = crate::backend::distributed_statevector::BACKEND_NAME;
        crate::backend::reduced_density::reduced_density_side(name, 2, limits)?;
        let rank_bits = context.size().trailing_zeros() as usize;
        if circuit.num_qubits.saturating_sub(rank_bits) < 2 {
            crate::backend::dense_statevector_len(
                name,
                "statevector export",
                circuit.num_qubits,
                limits,
            )?;
        }
    }
    Ok(NoisyRoute::Trajectory(plan))
}
