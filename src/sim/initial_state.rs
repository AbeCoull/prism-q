//! Terminals run from a caller-supplied start state instead of |0...0>.

use super::*;

/// Gate for the terminals that cannot carry a start state through their own
/// machinery.
pub(super) fn reject_initial_state(
    kind: &BackendKind,
    terminal: &str,
    instead: &str,
) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason: format!("{terminal} does not accept a start state; {instead}"),
    }
}

/// Reject a start state whose width disagrees with the circuit's.
///
/// The circuit's declared register wins: the buffer sets the backend's width,
/// so a shorter or longer one would silently simulate a different register
/// than the one the instructions index. A register too wide to index with a
/// `usize` has no dense start state at all.
pub(super) fn check_initial_state_len(state: StartState<'_>, num_qubits: usize) -> Result<()> {
    let (buffer, bits, entries) = match state {
        StartState::Amplitudes(amplitudes) => (amplitudes, num_qubits, "amplitudes"),
        StartState::DensityMatrix(rho) => (rho, 2 * num_qubits, "density matrix entries"),
    };
    let want = (bits < usize::BITS as usize).then(|| 1usize << bits);
    if want == Some(buffer.len()) {
        return Ok(());
    }
    let needs = match want {
        Some(count) => count.to_string(),
        None => format!("2^{bits}"),
    };
    Err(PrismError::InvalidParameter {
        message: format!(
            "start state has {} {entries}, but a {num_qubits}-qubit circuit needs {needs}",
            buffer.len()
        ),
    })
}

/// Build the constrained backend for a start state and load it.
pub(super) fn backend_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: StartState<'_>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<(Box<dyn Backend>, RouteRule)> {
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    check_initial_state_len(state, circuit.num_qubits)?;
    let plan = initial_state_plan(kind, circuit.num_qubits, state, limits)?;
    let mut backend = plan.build(seed, limits);
    state.load(&mut *backend, circuit.num_classical_bits)?;
    let reason = route_metadata(&*backend, plan.is_gpu(), RouteRule::StartState).reason;
    Ok((backend, reason))
}

pub(super) fn run_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: StartState<'_>,
    seed: u64,
    opts: &SimOptions,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    let (mut backend, reason) = backend_from_initial_state(kind, circuit, state, seed, limits)?;
    let saves = apply_fused_circuit(&mut *backend, circuit)?;

    let probabilities = if opts.probabilities {
        try_backend_probabilities(&*backend)?
    } else {
        None
    };
    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
        metadata: backend_metadata(&*backend).with_reason(reason),
        saves,
    })
}

/// Shots for a start state. Terminal measurements sample one evolved
/// distribution; anything else replays the start state per shot, which is what
/// mid-circuit collapse and classical feedback need.
pub(super) fn shots_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: StartState<'_>,
    num_shots: usize,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    let bits = circuit.num_classical_bits;
    if circuit.has_terminal_measurements_only() && !circuit.has_resets() {
        let stripped = circuit.without_measurements();
        let (mut backend, reason) =
            backend_from_initial_state(kind, &stripped, state, seed, limits)?;
        apply_fused_circuit(&mut *backend, &stripped)?;
        let meas_map = circuit.measurement_map();
        let shots = if backend.supports_native_sampling() {
            let samples = backend.sample_basis_states(num_shots, seed)?;
            Some(shots_from_basis_samples(&samples, &meas_map, bits))
        } else {
            try_backend_probabilities(&*backend)?
                .map(|probs| sample_shots(&probs, &meas_map, bits, num_shots, seed))
        };
        if let Some(shots) = shots {
            return Ok(ShotsResult::from_shots(shots, bits)
                .with_metadata(backend_metadata(&*backend).with_reason(reason)));
        }
    }

    // Plan, expansion, and fusion are shot independent, so they are hoisted out
    // of the replay loop, matching `run_shots_per_shot`.
    if !kind.is_auto() {
        validate_explicit_backend(kind, circuit, limits)?;
    }
    check_initial_state_len(state, circuit.num_qubits)?;
    let plan = initial_state_plan(kind, circuit.num_qubits, state, limits)?;
    let probe = plan.build(seed, limits);
    let expanded = expand_for_backend(&*probe, circuit);
    let fused = fuse_for_backend(&*probe, &expanded);

    let route = plan.resolved();
    let states = [(route, circuit.num_qubits)];
    collect_shots(
        circuit,
        num_shots,
        seed,
        route,
        kind,
        &states,
        |shot_seed| {
            let mut backend = plan.build(shot_seed, limits);
            state.load(&mut *backend, circuit.num_classical_bits)?;
            backend.apply_instructions(&fused.instructions)?;
            Ok((
                backend.classical_results().to_vec(),
                route_metadata(&*backend, plan.is_gpu(), RouteRule::StartState),
            ))
        },
    )
}

pub(super) fn marginals_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: StartState<'_>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<MarginalsResult> {
    let (mut backend, reason) = backend_from_initial_state(kind, circuit, state, seed, limits)?;
    apply_fused_without_saves(&mut *backend, circuit, "marginals")?;
    if backend.supports_pauli_expectation() {
        let mut result = marginals_from_pauli_expectations(&*backend, circuit.num_qubits)?;
        result.metadata.reason = reason;
        return Ok(result);
    }
    Ok(MarginalsResult {
        marginals: Probabilities::Dense(backend.probabilities()?).marginals(),
        metadata: backend_metadata(&*backend).with_reason(reason),
    })
}

pub(super) fn expectation_values_from_initial_state(
    kind: &BackendKind,
    circuit: &Circuit,
    state: StartState<'_>,
    observables: &[Vec<PauliTerm>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<ExpectationResult> {
    // Before the run, matching the dense statevector path.
    let masks = observables
        .iter()
        .map(|obs| pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;

    let (mut backend, reason) = backend_from_initial_state(kind, circuit, state, seed, limits)?;
    apply_fused_without_saves(&mut *backend, circuit, "expectation_values")?;
    let metadata = backend_metadata(&*backend).with_reason(reason);
    if backend.supports_pauli_expectation() {
        let values = backend.pauli_expectations(observables)?;
        return Ok(analytic_expectations(values, metadata));
    }

    let evolved = backend.export_statevector()?;
    let (values, _) = pauli_expectations_from_masks(&evolved, &masks);
    Ok(analytic_expectations(values, metadata))
}
