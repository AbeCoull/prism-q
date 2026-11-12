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
        usize::MAX,
    )? {
        Ok(planned) => planned,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(planned.with_limits(limits)))
}

/// MPS work per observable, in [`mps_apply_cost`] units, from which an `Auto`
/// expectation above the statevector cap plans the scalar tensor route before
/// running MPS, about 5 ms of MPS time per observable on one thread. Eight local
/// observables on a four-layer brickwork stay under it at any width, and on an
/// eight-layer one up to 64 qubits.
/// Bracketed by `auto/expectation_above_cap` (`su4_brick_d4/64` and `brickwork_d8_local/64` against `brickwork_d8_local/128`).
const TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE: f64 = 8e4;

/// [`mps_apply_cost`] units one planned tensor-network element costs: 60 to
/// 140 ns per element contracted against 50 to 70 ns per unit on brickwork
/// circuits, both on one thread.
/// Bracketed by `auto/expectation_above_cap` (`brickwork_d8_local/128` against `brickwork_d10_zz/48`).
const TENSOR_ELEMENT_MPS_COST: f64 = 2.0;

/// The scalar tensor route for an `Auto` expectation above the statevector cap,
/// where the alternative is the MPS `Auto` resolves to rather than a dense run.
/// `None` when `Auto` resolves to another backend, when the MPS estimate is
/// under [`TENSOR_ROUTE_MPS_COST_PER_OBSERVABLE`] per observable, or when the
/// planned contraction would cost more than the MPS estimate or hold an
/// intermediate over the tensor peak cap. The dry run stops as soon as the planned
/// elements overspend, so a declined plan costs a fraction of the MPS run.
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
        limits.tn_peak_elements(),
        (mps_cost / TENSOR_ELEMENT_MPS_COST) as usize,
    )? {
        Ok(planned) => planned,
        Err(e) => return Some(Err(e)),
    };
    Some(Ok(planned.with_limits(limits)))
}

/// Statevector instruction-amplitudes that one register amplitude costs per pass over
/// the Clifford register: 1.2 to 1.9 ns per register amplitude against 0.056 to 0.069 ns
/// per dense instruction-amplitude on four cores. The register runs serially, so the
/// ratio rises with core count.
/// Bracketed by `auto/clifford_t_price` (`distinct_17q_t16` flips at 11, `scrambled_22q_t20` at 86).
/// Fitted on the same group (`register/scrambled_22q_t20` and `register/scrambled_32q_t20`
/// give 22 to 27); `distinct_17q_t16` is the one row on the statevector side.
#[cfg(not(prism_lowered_thresholds))]
pub(super) const REGISTER_PASS_STATEVECTOR_AMPLITUDES: f64 = 25.0;
#[cfg(prism_lowered_thresholds)]
pub(super) const REGISTER_PASS_STATEVECTOR_AMPLITUDES: f64 = 1.0;

#[cfg(test)]
thread_local! {
    pub(super) static REGISTER_DRY_RUNS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Register work in statevector instruction-amplitudes: `rotation_work` amplitudes swept
/// by the T rotations plus about `3 (min(width, measured) + 1)` passes over `2^width`
/// amplitudes to diagonalize the measured rows.
fn clifford_register_work(width: usize, rotation_work: f64, measured: usize) -> f64 {
    let passes = 3 * (width.min(measured) + 1);
    REGISTER_PASS_STATEVECTOR_AMPLITUDES * (rotation_work + (passes << width) as f64)
}

/// Whether `Auto` samples the terminal measurements of `circuit` from the Clifford
/// register: a unitary Clifford+T circuit from [`MIN_AUTO_CLIFFORD_REGISTER_QUBITS`]
/// qubits whose register fits [`clifford_register::MAX_REGISTER_QUBITS`]. Within the
/// statevector cap the register must also cost less than the dense run over each
/// independent block. When a block passes the cap no dense run is exact, so a register
/// that fits is taken whatever the cap.
///
/// A bound from the T count decides most circuits. The rest take a tableau dry run
/// capped at the widest register that could still win.
pub(super) fn decide_clifford_register(circuit: &Circuit, limits: &ResolvedLimits) -> bool {
    #[cfg(feature = "bench-internal")]
    if !crate::bench_variant_b() {
        return legacy_prefers_clifford_register(circuit);
    }
    let n = circuit.num_qubits;
    if n < MIN_AUTO_CLIFFORD_REGISTER_QUBITS
        || !circuit.has_terminal_measurements_only()
        || circuit.has_resets()
    {
        return false;
    }
    let Some(t) = auto_clifford_t_count(circuit) else {
        return false;
    };
    let blocks = analyze_independence(circuit).0.map(|(blocks, _)| blocks);
    let widest_block = blocks.as_ref().map_or(n, |blocks| {
        blocks.iter().map(|block| block.len()).max().unwrap_or(0)
    });
    let instructions = circuit.instructions.len();
    let dense = (widest_block <= limits.statevector_qubits.value).then(|| {
        let work = match &blocks {
            Some(blocks) => blocks.iter().fold(0usize, |work, block| {
                work.saturating_add(statevector_work(instructions, block.len()))
            }),
            None => statevector_work(instructions, n),
        };
        work as f64
    });
    register_wins(t, n, measured_qubits(circuit), dense, |max_width| {
        stabilizer_rank::terminal_register_price(circuit, max_width)
    })
}

fn measured_qubits(circuit: &Circuit) -> usize {
    let mut is_measured = vec![false; circuit.num_qubits];
    for (q, _) in circuit.measurement_map() {
        is_measured[q] = true;
    }
    is_measured.iter().filter(|&&m| m).count()
}

/// Whether a register on `qubits` qubits under `t` T gates, `measured` of them read, costs
/// less than `dense` statevector work, or fits [`clifford_register::MAX_REGISTER_QUBITS`]
/// when `dense` is `None` because no dense run is exact. `price` dry runs the tableau
/// capped at a width, returning the width and rotation work, `None` past the cap.
fn register_wins(
    t: usize,
    qubits: usize,
    measured: usize,
    dense: Option<f64>,
    price: impl FnOnce(usize) -> Option<(usize, u64)>,
) -> bool {
    let max_width = clifford_register::MAX_REGISTER_QUBITS;
    let bound = t.min(qubits);
    let Some(dense) = dense else {
        if bound <= max_width {
            return true;
        }
        #[cfg(test)]
        REGISTER_DRY_RUNS.with(|runs| runs.set(runs.get() + 1));
        return price(max_width).is_some();
    };
    if bound <= max_width
        && clifford_register_work(bound, ((t - bound + 2) << bound) as f64, measured) < dense
    {
        return true;
    }
    let Some(widest) = (0..=bound.min(max_width))
        .rev()
        .find(|&k| clifford_register_work(k, (2usize << k) as f64, measured) < dense)
    else {
        return false;
    };
    #[cfg(test)]
    REGISTER_DRY_RUNS.with(|runs| runs.set(runs.get() + 1));
    price(widest)
        .is_some_and(|(k, rotations)| clifford_register_work(k, rotations as f64, measured) < dense)
}

/// How `Auto` samples a Clifford+T circuit with a mid-circuit measurement, a guard or a
/// reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MidCircuitRoute {
    /// The Clifford register on the circuit with its measurements moved to the end.
    Register,
    /// The stabilizer-rank branches, walked once per shot.
    Branches,
    /// The tree's per-shot route, the statevector walk under the cap.
    Decline,
}

fn split_events(instructions: &[Instruction]) -> usize {
    instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Measure { .. } | Instruction::Reset { .. } => 1,
            Instruction::Region(region) => split_events(region.body()),
            _ => 0,
        })
        .sum()
}

/// Outcome histories the statevector walk can reach on `shots` shots: two per measurement
/// or reset ahead of the closing run of measurements, which it samples in one pass. An
/// upper bound, since a deterministic outcome does not split.
fn walk_histories(circuit: &Circuit, shots: usize) -> usize {
    if circuit.has_terminal_measurements_only() {
        return 1;
    }
    let closing = circuit
        .instructions
        .iter()
        .rev()
        .take_while(|inst| {
            matches!(
                inst,
                Instruction::Measure { .. } | Instruction::Barrier { .. }
            )
        })
        .filter(|inst| matches!(inst, Instruction::Measure { .. }))
        .count();
    let splits = split_events(&circuit.instructions) - closing;
    let reachable = 1usize.checked_shl(splits as u32).unwrap_or(usize::MAX);
    shots.max(1).min(reachable)
}

/// Route for the shots of `circuit` when it is Clifford+T with a T gate and holds a
/// mid-circuit measurement, a guard or a reset; [`MidCircuitRoute::Decline`] otherwise.
///
/// The register samples the circuit with its measurements deferred to the end. Past the
/// statevector cap it is taken whenever it fits; within the cap, from
/// [`MIN_AUTO_CLIFFORD_REGISTER_QUBITS`] qubits, when it costs less than one dense run per
/// outcome history the walk can reach. Past the cap a circuit the register does not take
/// goes to the branches up to [`MAX_AUTO_MID_CIRCUIT_BRANCH_T`] T gates, guarded ones
/// included. The shot count enters only under the cap, where every route is exact.
pub(super) fn decide_mid_circuit_clifford_t(
    circuit: &Circuit,
    shots: usize,
    limits: &ResolvedLimits,
) -> MidCircuitRoute {
    if circuit.has_terminal_measurements_only() && !circuit.has_resets() {
        return MidCircuitRoute::Decline;
    }
    let Some(t) = auto_clifford_t_count(circuit) else {
        return MidCircuitRoute::Decline;
    };
    let n = circuit.num_qubits;
    let over_cap = n > limits.statevector_qubits.value;
    if (over_cap || n >= MIN_AUTO_CLIFFORD_REGISTER_QUBITS)
        && let Some(deferred) = stabilizer_rank::deferred_register_circuit(circuit)
    {
        let dense = (!over_cap).then(|| {
            statevector_work(circuit.instructions.len(), n) as f64
                * walk_histories(circuit, shots) as f64
        });
        let price = |max_width| {
            clifford_register::register_price(&deferred, max_width)
                .ok()
                .flatten()
        };
        if register_wins(
            t,
            deferred.num_qubits,
            measured_qubits(&deferred),
            dense,
            price,
        ) {
            return MidCircuitRoute::Register;
        }
    }
    if over_cap && t <= MAX_AUTO_MID_CIRCUIT_BRANCH_T {
        MidCircuitRoute::Branches
    } else {
        MidCircuitRoute::Decline
    }
}

pub(super) fn contract_tensor(
    planned: crate::backend::tensornetwork::BoundedExpectations,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    planned.contract(limits).map(|values| {
        analytic_expectations(values, RunMetadata::exact(ResolvedBackend::TensorNetwork))
    })
}

pub(super) struct TerminalCircuit {
    pub(super) circuit: Circuit,
    pub(super) meas_map: Vec<(usize, usize)>,
}

pub(super) enum ShotRoute {
    Compiled(Option<Circuit>),
    /// The rewrite to terminal measurements with guarded Paulis that land after sampling.
    Guarded(Box<compiled::DeferredCircuit>),
    TerminalStatevector(TerminalCircuit),
    Native(BackendPlan, TerminalCircuit),
    TerminalProbabilities(TerminalCircuit),
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
        && let Ok(deferred) = compiled::defer_guarded_circuit(circuit)
    {
        return Ok(if deferred.guarded.is_empty() {
            ShotRoute::Compiled(Some(deferred.circuit))
        } else {
            ShotRoute::Guarded(Box::new(deferred))
        });
    }
    if kind.is_auto() && decide_clifford_register(circuit, limits) {
        return Ok(ShotRoute::StabilizerRank);
    }
    if kind.is_auto()
        && mid_circuit_route_live()
        && decide_mid_circuit_clifford_t(circuit, num_shots, limits) != MidCircuitRoute::Decline
    {
        return Ok(ShotRoute::StabilizerRank);
    }
    let terminal_only = circuit.has_terminal_measurements_only();
    let terminal = if terminal_only {
        let meas_map = circuit.measurement_map();
        (!meas_map.is_empty()).then(|| TerminalCircuit {
            circuit: circuit.without_measurements(),
            meas_map,
        })
    } else {
        None
    };
    let terminal = match terminal {
        Some(terminal) if terminal_statevector_candidate(kind, &terminal.circuit, limits) => {
            return Ok(ShotRoute::TerminalStatevector(terminal));
        }
        terminal => terminal,
    };
    if matches!(kind, BackendKind::StabilizerRank) && circuit.has_t_gates() {
        return Ok(ShotRoute::StabilizerRank);
    }
    #[cfg(feature = "bench-internal")]
    if kind.is_auto() && auto_samples_wide_by_stabilizer_rank(circuit) {
        return Ok(ShotRoute::StabilizerRank);
    }
    if terminal_only
        || kind.is_density_matrix() && circuit.has_terminal_measurements_under_reset_channel()
    {
        let terminal = terminal.unwrap_or_else(|| TerminalCircuit {
            circuit: circuit.without_measurements(),
            meas_map: circuit.measurement_map(),
        });
        return Ok(
            match decide_native(kind, &terminal.circuit, true, limits)? {
                Some(plan) => ShotRoute::Native(plan, terminal),
                None if terminal.circuit.num_qubits > 64
                    && matches!(
                        plan_probability_route(kind, &terminal.circuit, limits),
                        ProbabilityRoute::Decomposed(_)
                    ) =>
                {
                    ShotRoute::PerShot
                }
                None => ShotRoute::TerminalProbabilities(terminal),
            },
        );
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
    require_unitary_expectation_circuit(kind, circuit)?;
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
    /// The circuit samples as it stands, or as the rewrite of a dynamic one to
    /// terminal measurements.
    Compiled(Option<Box<compiled::DeferredCircuit>>),
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

    check_noisy_per_shot(kind, circuit, noise_model, limits)?;

    if noise_model.has_only_pauli_channels() && is_clifford_sampler_kind(kind) {
        if supports_compiled_measurement_sampling(circuit) {
            return Ok(NoisyRoute::Compiled(None));
        }
        if let Ok(deferred) = compiled::defer_noisy_circuit(circuit, noise_model) {
            return Ok(NoisyRoute::Compiled(Some(Box::new(deferred))));
        }
    }

    trajectory_plan(kind, circuit, noise_model, limits).map(NoisyRoute::Trajectory)
}

/// The trajectory plan for a dynamic program under noise: `summary` is the
/// program's summary circuit and `channels` every event its blocks and runtime
/// rotations carry.
pub(super) fn decide_noisy_program(
    kind: &BackendKind,
    summary: &Circuit,
    channels: &NoiseModel,
    limits: &ResolvedLimits,
) -> Result<BackendPlan> {
    check_noisy_per_shot(kind, summary, channels, limits)?;
    trajectory_plan(kind, summary, channels, limits)
}

/// Reject a kind that cannot run `noise_model`'s channels on `circuit` one pure
/// state per shot.
fn check_noisy_per_shot(
    kind: &BackendKind,
    circuit: &Circuit,
    noise_model: &NoiseModel,
    limits: &ResolvedLimits,
) -> Result<()> {
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
    Ok(())
}

fn trajectory_plan(
    kind: &BackendKind,
    circuit: &Circuit,
    noise_model: &NoiseModel,
    limits: &ResolvedLimits,
) -> Result<BackendPlan> {
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
    if noise_model.has_two_qubit_kraus() && !plan.supports_two_qubit_kraus(limits) {
        #[cfg(feature = "distributed")]
        const PAIR_ROUTES: &str =
            "a host statevector, sparse, factored, MPS or distributed statevector";
        #[cfg(not(feature = "distributed"))]
        const PAIR_ROUTES: &str = "a host statevector, sparse, factored or MPS";
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{:?}", plan.resolved()),
            reason: format!(
                "a two-qubit Kraus channel needs the pair's reduced density matrix its \
                 branch probabilities are drawn from, and a kernel for the branch operator, \
                 which is not unitary. A tableau and a product state have no such kernel, \
                 a tensor network traces out no arbitrary pair, and a device-resident \
                 statevector reads no pair; run on {PAIR_ROUTES} route, or evaluate the \
                 channel exactly on BackendKind::DensityMatrix"
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
    Ok(plan)
}
