//! Marginal, expectation value and weighted observable routes.

use super::*;

/// Rejection naming the two terminals the Pauli path engine serves.
pub(super) fn reject_pauli_path(terminal: &str) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "PauliPath".into(),
        reason: format!(
            "{terminal} is not served by Pauli path propagation, which answers \
             `expectation_values` and `observable_expectation` only"
        ),
    }
}

pub(super) fn reject_pauli_path_initial_state(state: Option<StartState<'_>>) -> Result<()> {
    match state {
        Some(_) => Err(reject_pauli_path("a start state")),
        None => Ok(()),
    }
}

/// One Pauli path evaluation per observable, under `noise` when one is
/// attached and on the bare circuit otherwise.
///
/// The route is exact only when nothing was truncated, so exactness is decided
/// by the discarded mass the run reports rather than by the parameters it was
/// given: a budget that never binds still returns an exact value.
pub(super) fn pauli_path_expectations(
    circuit: &Circuit,
    noise: Option<&NoiseModel>,
    observables: &[Vec<PauliTerm>],
    epsilon: f64,
    max_terms: usize,
) -> Result<ExpectationResult> {
    let empty;
    let noise = match noise {
        Some(model) => model,
        None => {
            empty = NoiseModel::noiseless_for(circuit);
            &empty
        }
    };
    let mut values = Vec::with_capacity(observables.len());
    let mut discarded = 0.0f64;
    for obs in observables {
        let result =
            unified_pauli::run_pauli_path_observable(circuit, noise, obs, epsilon, max_terms)?;
        discarded = discarded.max(result.total_discarded);
        values.push(result.mean);
    }
    Ok(analytic_expectations(
        values,
        pauli_path_metadata(discarded),
    ))
}

/// [`pauli_path_expectations`] for a weighted observable, propagated as one sum.
pub(super) fn pauli_path_observable(
    circuit: &Circuit,
    noise: Option<&NoiseModel>,
    observable: &PauliObservable,
    epsilon: f64,
    max_terms: usize,
) -> Result<ObservableExpectation> {
    let empty;
    let noise = match noise {
        Some(model) => model,
        None => {
            empty = NoiseModel::noiseless_for(circuit);
            &empty
        }
    };
    let result = unified_pauli::run_pauli_path_weighted_observable(
        circuit,
        noise,
        &weighted_terms(observable),
        epsilon,
        max_terms,
    )?;
    Ok(analytic_observable(
        result.mean,
        pauli_path_metadata(result.total_discarded),
    ))
}

/// Metadata for an explicit Pauli path run, approximate once any term was discarded.
pub(super) fn pauli_path_metadata(total_discarded: f64) -> RunMetadata {
    let metadata = if total_discarded > 0.0 {
        RunMetadata::approximate(ResolvedBackend::PauliPath)
    } else {
        RunMetadata::exact(ResolvedBackend::PauliPath)
    };
    metadata.with_reason(RouteRule::ExplicitBackend)
}

/// Expectation values on the evolved density matrix of a circuit with resets,
/// with the provenance the same circuit without its resets reports: the exact
/// mixture's with noise, otherwise the start state's or the explicit backend's.
pub(super) fn reset_channel_expectations(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    noise_model: Option<&noise::NoiseModel>,
    initial_state: Option<StartState<'_>>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    let masks = observables
        .iter()
        .map(|obs| pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;
    let rule = if initial_state.is_some() {
        RouteRule::StartState
    } else {
        RouteRule::ExplicitBackend
    };
    let dm = noise::evolve_density_matrix(kind, circuit, noise_model, initial_state, seed, limits)?;
    let metadata = match noise_model {
        Some(_) => exact_mixture_metadata(kind),
        None => {
            let expected_device = !matches!(
                accel_for(kind, Family::DensityMatrix, circuit.num_qubits),
                dispatch::Accel::Cpu
            );
            route_metadata(&dm, expected_device, rule)
        }
    };
    Ok(analytic_expectations(
        dm.expectations_pauli(&masks),
        metadata,
    ))
}

/// Per-qubit marginals from native single-qubit Z expectations, for a backend
/// whose own representation answers them without a dense probability vector.
pub(super) fn marginals_from_pauli_expectations(
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

/// Route-level exactness per the [`Exactness`] convention: a policy that can
/// discard marks the route approximate even on a run that discarded nothing.
/// Only a zero-threshold policy stays exact, since it overflows into an error
/// instead of an approximation; a budget always drops once the sum passes it.
/// The realized bound is a coefficient magnitude rather than a state overlap,
/// so it stays `total_discarded` on the engine result and never a fidelity
/// bound.
pub(super) fn spd_metadata(truncation: &SpdTruncation) -> RunMetadata {
    if truncation.can_discard() {
        RunMetadata::approximate(ResolvedBackend::DeterministicPauli)
    } else {
        RunMetadata::exact(ResolvedBackend::DeterministicPauli)
    }
}

/// Joint distribution from `<Z_S>` over every nonempty subset `S` in bitmask order from 1:
/// `P(o) = 2^-k sum_S (-1)^|o & S| <Z_S>`, rounding below zero clamped.
pub(super) fn subset_probabilities(subset_values: Vec<f64>) -> Vec<f64> {
    let mut table = Vec::with_capacity(subset_values.len() + 1);
    table.push(1.0);
    table.extend(subset_values);
    let mut half = 1;
    while half < table.len() {
        for block in table.chunks_exact_mut(2 * half) {
            let (lo, hi) = block.split_at_mut(half);
            for (a, b) in lo.iter_mut().zip(hi) {
                (*a, *b) = (*a + *b, *a - *b);
            }
        }
        half *= 2;
    }
    let scale = 1.0 / table.len() as f64;
    table.iter().map(|v| (v * scale).max(0.0)).collect()
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

pub(super) fn supports_pauli_marginal_backend(circuit: &Circuit) -> bool {
    circuit.is_clifford_plus_t() && !has_nonunitary_or_classical_ops(circuit)
}

/// Gate-set support is left to the engines, which accept Clifford gates and
/// Pauli rotations and report the offending gate by name.
pub(super) fn validate_pauli_marginal_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli marginal backends require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }
    Ok(())
}

pub(super) fn execute_marginals(
    route: MarginalRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    limits: &ResolvedLimits,
    require_exact: bool,
) -> Result<MarginalsResult> {
    match route {
        MarginalRoute::Stochastic(num_samples) => {
            let spp = unified_pauli::run_spp(circuit, num_samples, seed)?;
            Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spp.expectations),
                metadata: RunMetadata::approximate(ResolvedBackend::StochasticPauli)
                    .with_shots(num_samples)
                    .with_reason(RouteRule::ExplicitBackend),
            })
        }
        MarginalRoute::Deterministic(truncation) => {
            let spd = unified_pauli::run_spd_with(circuit, &truncation)?;
            Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spd.expectations),
                metadata: spd_metadata(&truncation).with_reason(RouteRule::ExplicitBackend),
            })
        }
        MarginalRoute::Spd { budget } => {
            let exact = SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            };
            let result = match budget {
                Some(budget) => unified_pauli::run_spd_exact_within(circuit, budget)?,
                None => Some(unified_pauli::run_spd_with(circuit, &exact)?),
            };
            if let Some(spd) = result {
                return Ok(MarginalsResult {
                    marginals: expectations_to_marginals(&spd.expectations),
                    metadata: spd_metadata(&exact).with_reason(RouteRule::SpdExact),
                });
            }
            let fallback = decide_marginals_after_spd(kind, circuit, limits)?;
            if require_exact {
                plan::require_exact_marginal(&fallback, kind, circuit, limits)?;
            }
            let mut result =
                execute_marginals(fallback, kind, circuit, seed, limits, require_exact)?;
            result.metadata.reason = RouteRule::SpdBudgetFallback;
            Ok(result)
        }
        MarginalRoute::Register(plan) => Ok(MarginalsResult {
            marginals: expectations_to_marginals(&plan.values()),
            metadata: RunMetadata::exact(ResolvedBackend::StabilizerRank)
                .with_reason(RouteRule::CliffordRegister),
        }),
        MarginalRoute::Tensor(planned, rule) => {
            let values = planned.contract(limits)?;
            Ok(MarginalsResult {
                marginals: expectations_to_marginals(&values),
                metadata: RunMetadata::exact(ResolvedBackend::TensorNetwork).with_reason(rule),
            })
        }
        MarginalRoute::Native(plan) => {
            let rule = backend_rule(kind, circuit, &plan);
            let mut backend = plan.build(seed, limits);
            execute(&mut *backend, circuit, &SimOptions::classical_only())?;
            let mut result = marginals_from_pauli_expectations(&*backend, circuit.num_qubits)?;
            result.metadata.reason = route_metadata(&*backend, plan.is_gpu(), rule).reason;
            Ok(result)
        }
        MarginalRoute::Probabilities => {
            let result = run_with(kind.clone(), circuit, seed, limits)?;
            match result.probabilities {
                Some(probs) => Ok(MarginalsResult {
                    marginals: probs.marginals(),
                    metadata: result.metadata,
                }),
                None => Err(PrismError::BackendUnsupported {
                    backend: "simulate".into(),
                    operation: format!(
                        "marginals for {} qubits without backend probability output",
                        circuit.num_qubits
                    ),
                }),
            }
        }
    }
}

/// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on a unitary circuit's
/// output state, using automatic backend selection. See
/// [`Simulate::expectation_values`] for explicit backend control.
///
/// # Examples
///
/// ```
/// use prism_q::sim::run_expectation_values;
/// use prism_q::{Circuit, Gate, PauliTerm};
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
    let limits = &ResolvedLimits::process_default();
    run_expectation_values_with(BackendKind::Auto, circuit, observables, seed, limits)
}

pub(super) fn run_expectation_values_with(
    kind: BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Vec<f64>> {
    run_expectation_values_reported(kind, circuit, observables, seed, limits)
        .map(ExpectationResult::into_values)
}

pub(super) fn run_expectation_values_reported(
    kind: BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    let route = decide_expectations(&kind, circuit, observables, limits)?;
    execute_expectations(route, &kind, circuit, observables, seed, limits)
}

pub(super) fn execute_expectations(
    route: ExpectationRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    let rule = expectation_rule(&route, kind);
    let mut result = match route {
        ExpectationRoute::Stochastic(num_samples) => {
            let mut values = Vec::with_capacity(observables.len());
            let mut std_errors = Vec::with_capacity(observables.len());
            for (i, obs) in observables.iter().enumerate() {
                let r = unified_pauli::run_spp_observable(
                    circuit,
                    obs,
                    num_samples,
                    mix_seed(seed, i),
                )?;
                values.push(r.mean);
                std_errors.push(r.std_error);
            }
            Ok(ExpectationResult {
                values,
                std_errors: Some(std_errors),
                metadata: RunMetadata::approximate(ResolvedBackend::StochasticPauli)
                    .with_shots(num_samples),
            })
        }
        ExpectationRoute::Deterministic(truncation) => {
            let mut values = Vec::with_capacity(observables.len());
            for obs in observables {
                let r = unified_pauli::run_spd_observable_with(circuit, obs, &truncation)?;
                values.push(r.mean);
            }
            Ok(analytic_expectations(values, spd_metadata(&truncation)))
        }
        ExpectationRoute::Register(plan) => Ok(analytic_expectations(
            plan.values(),
            RunMetadata::exact(ResolvedBackend::StabilizerRank),
        )),
        ExpectationRoute::Tensor(planned, _) => contract_tensor(planned, limits),
        ExpectationRoute::Statevector => {
            expectation_values_statevector(kind, circuit, observables, seed, limits)
        }
        ExpectationRoute::Native(plan) => {
            let mut backend = plan.build(seed, limits);
            execute(&mut *backend, circuit, &SimOptions::classical_only())?;
            let values = backend.pauli_expectations(observables)?;
            Ok(analytic_expectations(
                values,
                route_metadata(&*backend, plan.is_gpu(), rule),
            ))
        }
    }?;
    if result.metadata.reason != RouteRule::DeviceInitializationFallback {
        result.metadata.reason = rule;
    }
    Ok(result)
}

/// Compute `⟨H⟩` and its grouped-measurement variance for a weighted Pauli
/// observable, using automatic backend selection. See
/// [`Simulate::observable_expectation`] for explicit backend control and
/// [`ObservableExpectation::variance`] for the variance contract.
///
/// # Examples
///
/// ```
/// use prism_q::sim::run_observable_expectation;
/// use prism_q::{Circuit, Gate, PauliObservable, PauliTerm};
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
    let limits = &ResolvedLimits::process_default();
    run_observable_expectation_reported(BackendKind::Auto, circuit, observable, seed, limits)
}

pub(super) fn run_observable_expectation_reported(
    kind: BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ObservableExpectation> {
    require_unitary_expectation_circuit(&kind, circuit)?;

    let grouped_statevector = grouped_statevector(&kind, circuit, limits);
    if grouped_statevector {
        return grouped_expectation_statevector(&kind, circuit, observable, seed, limits);
    }

    let observables = observable_vecs(observable);
    let route = decide_expectations(&kind, circuit, &observables, limits)?;
    execute_observable(
        route,
        &kind,
        circuit,
        observable,
        &observables,
        seed,
        limits,
    )
}

/// Evaluate a weighted observable on its expectation route. The deterministic
/// Pauli route propagates the whole sum once; every other route evaluates each
/// term and folds the values.
pub(super) fn execute_observable(
    route: ExpectationRoute,
    kind: &BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ObservableExpectation> {
    if let ExpectationRoute::Deterministic(truncation) = &route {
        let result = unified_pauli::run_spd_weighted_observable_with(
            circuit,
            &weighted_terms(observable),
            truncation,
        )?;
        let metadata = spd_metadata(truncation).with_reason(expectation_rule(&route, kind));
        return Ok(analytic_observable(result.mean, metadata));
    }
    let result = execute_expectations(route, kind, circuit, observables, seed, limits)?;
    Ok(weighted_observable_result(
        observable,
        &result.values,
        result.std_errors.as_deref(),
        result.metadata,
    ))
}

pub(super) fn weighted_terms(observable: &PauliObservable) -> Vec<(f64, &[PauliTerm])> {
    observable
        .terms()
        .iter()
        .map(|(weight, factors)| (*weight, factors.as_slice()))
        .collect()
}

pub(super) fn analytic_observable(mean: f64, metadata: RunMetadata) -> ObservableExpectation {
    ObservableExpectation {
        mean,
        variance: None,
        group_variances: None,
        std_error: None,
        metadata,
    }
}

pub(super) fn observable_vecs(observable: &PauliObservable) -> Vec<Vec<PauliTerm>> {
    observable
        .terms()
        .iter()
        .map(|(_, factors)| factors.clone())
        .collect()
}

/// Fold per-term values into the weighted mean; no grouped traversal ran, so
/// there is no variance to report. Independent per-term estimates combine
/// into the weighted-sum standard error.
pub(super) fn weighted_observable_result(
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

/// Evaluate a weighted observable on the statevector: small groups from one
/// shared batched traversal, each large group's mean and variance from a
/// dedicated moments pass.
///
/// `Var(H_g) = <H_g^2> - <H_g>^2` per commuting group. For a small group the
/// square expands into pairwise product strings appended to the same
/// traversal that serves its term means; a group past the pair budget takes a
/// single-pass moment accumulation instead, on the state as run when the
/// group is Z-only and on a basis-rotated copy otherwise.
pub(super) fn grouped_expectation_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ObservableExpectation> {
    require_unitary_expectation_circuit(kind, circuit)?;
    let terms = observable.terms();
    // Validate before the 2^n simulation so bad observables fail cheaply. The
    // mask reduction shifts by qubit index, so it follows the `init` that
    // rejects a width no mask can address.
    for (_, factors) in terms {
        validate_observable(factors, circuit.num_qubits)?;
    }

    let (backend, expected_device, ()) =
        evolved_statevector(kind, circuit, seed, limits, || Ok(()))?;
    let mut result = grouped_expectation_on_state(&backend, observable)?;
    result.metadata.reason = statevector_route_reason(kind, &backend, expected_device);
    Ok(result)
}

/// The reason a statevector route reports: the default or explicit rule for `kind`, or
/// the device fallback when a requested device did not take the state.
fn statevector_route_reason(
    kind: &BackendKind,
    backend: &StatevectorBackend,
    expected_device: bool,
) -> RouteRule {
    let reason = if kind.is_auto() {
        RouteRule::Default
    } else {
        RouteRule::ExplicitBackend
    };
    route_metadata(backend, expected_device, reason).reason
}

/// `observable_variance` on the statevector route: the circuit runs once and `<T>` and
/// `<T^2>` of the traceless part come out of one pass over `(T psi)[k]`, in place of a
/// second run over the expanded square.
pub(super) fn operator_variance_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observable: &PauliObservable,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ObservableVariance> {
    require_unitary_expectation_circuit(kind, circuit)?;
    for (_, factors) in observable.terms() {
        validate_observable(factors, circuit.num_qubits)?;
    }
    let (offset, traceless) = observable.split_identity();
    let terms = traceless
        .terms()
        .iter()
        .map(|(c, factors)| Ok((*c, pauli_masks(factors, circuit.num_qubits)?)))
        .collect::<Result<Vec<_>>>()?;
    let table = observable::OperatorTable::new(&terms);

    let (backend, expected_device, ()) =
        evolved_statevector(kind, circuit, seed, limits, || Ok(()))?;
    let (first, second) = backend.operator_moments(&table)?;
    let mut metadata = backend_metadata(&backend);
    metadata.reason = statevector_route_reason(kind, &backend, expected_device);
    Ok(ObservableVariance {
        variance: second - first * first,
        mean: first + offset,
        metadata,
    })
}

/// The reduction half of [`grouped_expectation_statevector`], on a backend
/// that has already run the circuit.
pub(super) fn grouped_expectation_on_state(
    backend: &StatevectorBackend,
    observable: &PauliObservable,
) -> Result<ObservableExpectation> {
    let num_qubits = backend.num_qubits();
    let terms = observable.terms();
    let masks = terms
        .iter()
        .map(|(_, factors)| pauli_masks(factors, num_qubits))
        .collect::<Result<Vec<_>>>()?;
    let metadata = backend_metadata(backend);

    let grouping = observable.grouping();

    // Small groups get `<H_g^2>` from pairwise product strings appended to the
    // shared traversal: qubit-wise-commuting strings multiply phase-free
    // (shared qubits carry equal axes and cancel to identity), so each pair is
    // one more mask. Large groups fall back to a dedicated moments pass, which
    // costs a fixed number of state sweeps where the pair expansion grows
    // quadratically. Each group's mean comes out of whichever pass serves its
    // variance, so a large group's terms stay out of the shared traversal.
    let mut combined = Vec::with_capacity(masks.len());
    let mut pair_blocks: Vec<(usize, usize, usize, Vec<f64>)> = Vec::new();
    let mut deferred: Vec<usize> = Vec::new();
    for (gi, group) in grouping.groups.iter().enumerate() {
        let members = &group.term_indices;
        if members.len() * (members.len() - 1) / 2 > MAX_PAIR_MASKS_PER_GROUP {
            deferred.push(gi);
            continue;
        }
        let first_term = combined.len();
        combined.extend(members.iter().map(|&i| masks[i]));
        let first_pair = combined.len();
        let mut pair_coefficients = Vec::with_capacity(members.len() * (members.len() - 1) / 2);
        for (pos, &i) in members.iter().enumerate() {
            for &j in &members[pos + 1..] {
                let product_x = masks[i].0 ^ masks[j].0;
                let product_z = masks[i].1 ^ masks[j].1;
                combined.push((product_x, product_z, (product_x & product_z).count_ones()));
                pair_coefficients.push(2.0 * terms[i].0 * terms[j].0);
            }
        }
        pair_blocks.push((gi, first_term, first_pair, pair_coefficients));
    }

    // A device-resident state reduces every mask on the card; the host keeps
    // its state and norm for the moments pass below.
    let (values, host) = match backend.pauli_expectations_on_device(&combined) {
        Some(values) => (values?, None),
        None => {
            let state = backend.state_vector();
            let (values, norm) = pauli_expectations_from_masks(state, &combined);
            (values, Some((state, norm)))
        }
    };

    let mut mean: f64 = terms
        .iter()
        .filter(|(_, factors)| factors.is_empty())
        .map(|(c, _)| c)
        .sum();
    let mut group_variances = vec![0.0; grouping.groups.len()];

    for (gi, first_term, first_pair, pair_coefficients) in &pair_blocks {
        let group = &grouping.groups[*gi];
        let m1: f64 = group
            .term_indices
            .iter()
            .zip(&values[*first_term..])
            .map(|(&i, v)| terms[i].0 * v)
            .sum();
        let square_diag: f64 = group
            .term_indices
            .iter()
            .map(|&i| terms[i].0 * terms[i].0)
            .sum();
        let square_cross: f64 = pair_coefficients
            .iter()
            .zip(&values[*first_pair..])
            .map(|(c, v)| c * v)
            .sum();
        mean += m1;
        group_variances[*gi] = (square_diag + square_cross - m1 * m1).max(0.0);
    }

    // A Z-only group reads the state as run, any other group a copy rotated so
    // its terms read as Z strings (`x | z` per term). A device-resident state
    // takes both on the card and is exported only when the rotated copy does
    // not fit there.
    if !deferred.is_empty() {
        let requests: Vec<(Vec<u64>, Vec<f64>, u64, u64)> = deferred
            .iter()
            .map(|&gi| {
                let group = &grouping.groups[gi];
                let (x_bits, y_bits) = if group.is_z_only() {
                    (0, 0)
                } else {
                    group.rotation_masks()
                };
                (
                    group
                        .term_indices
                        .iter()
                        .map(|&i| (masks[i].0 | masks[i].1) as u64)
                        .collect(),
                    group.term_indices.iter().map(|&i| terms[i].0).collect(),
                    x_bits as u64,
                    y_bits as u64,
                )
            })
            .collect();
        let on_device = match host {
            Some(_) => None,
            None => backend.group_moments_on_device(&requests).transpose()?,
        };
        let moments = match on_device {
            Some(moments) => moments,
            None => {
                let exported;
                let (state, norm): (&[Complex64], f64) = match host {
                    Some(host) => host,
                    None => {
                        exported = backend.export_statevector()?;
                        (&exported, crate::backend::state_norm_sqr(&exported))
                    }
                };
                host_group_moments(state, norm, &requests)
            }
        };
        for (&gi, (m1, m2)) in deferred.iter().zip(moments) {
            mean += m1;
            group_variances[gi] = (m2 - m1 * m1).max(0.0);
        }
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

/// The host moments pass over each `(zmasks, coefficients, x_bits, y_bits)`
/// request of [`grouped_expectation_on_state`].
pub(crate) fn host_group_moments(
    state: &[Complex64],
    norm: f64,
    requests: &[(Vec<u64>, Vec<f64>, u64, u64)],
) -> Vec<(f64, f64)> {
    let mut rotated = Vec::new();
    requests
        .iter()
        .map(|(zmasks, coefficients, x_bits, y_bits)| {
            let zmasks: Vec<usize> = zmasks.iter().map(|&z| z as usize).collect();
            let rotation = x_bits | y_bits;
            if rotation == 0 {
                return observable::weighted_group_moments(state, &zmasks, coefficients, norm);
            }
            observable::rotate_to_z_basis(state, &mut rotated, *x_bits as usize, *y_bits as usize);
            let growth = 2f64.powi(rotation.count_ones() as i32);
            observable::weighted_group_moments(&rotated, &zmasks, coefficients, norm * growth)
        })
        .collect()
}

/// Values from a route that evaluates rather than samples, so there is no
/// interval to report.
pub(super) fn analytic_expectations(values: Vec<f64>, metadata: RunMetadata) -> ExpectationResult {
    ExpectationResult {
        values,
        std_errors: None,
        metadata,
    }
}

pub(super) fn expectation_values_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    // Validate before the 2^n simulation so bad observables fail cheaply. The
    // mask reduction shifts by qubit index, so it follows the `init` that
    // rejects a width no mask can address.
    for obs in observables {
        validate_observable(obs, circuit.num_qubits)?;
    }

    let (backend, expected_device, masks) =
        evolved_statevector(kind, circuit, seed, limits, || {
            observables
                .iter()
                .map(|obs| pauli_masks(obs, circuit.num_qubits))
                .collect::<Result<Vec<_>>>()
        })?;

    let values = match backend.pauli_expectations_on_device(&masks) {
        Some(values) => values?,
        None => pauli_expectations_from_masks(backend.state_vector(), &masks).0,
    };
    let mut metadata = backend_metadata(&backend);
    if expected_device && metadata.placement == Placement::Host {
        metadata.reason = RouteRule::DeviceInitializationFallback;
    }
    Ok(analytic_expectations(values, metadata))
}
