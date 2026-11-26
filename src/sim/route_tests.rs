use super::*;
use crate::gates::Gate;
use std::cell::Cell;

fn limits() -> ResolvedLimits {
    ResolvedLimits::resolve(None)
}

fn h_measure_chain(splits: usize) -> Circuit {
    let mut circuit = Circuit::new(1, 1);
    for _ in 0..=splits {
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_measure(0, 0);
    }
    circuit
}

#[test]
fn register_wins_without_a_dense_run_iff_the_register_fits() {
    let max = clifford_register::MAX_REGISTER_QUBITS;
    let calls = Cell::new(0);
    let split = || {
        calls.set(calls.get() + 1);
        Some(1.0)
    };
    assert!(register_wins(max, max + 5, max, None, split));
    assert_eq!(
        calls.get(),
        0,
        "a T bound within the register takes no dry run"
    );

    assert!(register_wins(max + 1, max + 5, max, None, split));
    assert_eq!(calls.get(), 1);
    assert!(!register_wins(max + 1, max + 5, max, None, || None));
}

#[test]
fn register_wins_on_the_bound_alone_when_it_undercuts_the_dense_run() {
    let never = || -> Option<f64> { panic!("no dry run expected") };
    assert!(register_wins(2, 4, 4, Some(f64::MAX), never));
    // Even a width-zero register costs something, so a free dense run beats a free split.
    assert!(!register_wins(2, 4, 4, Some(0.0), || Some(0.0)));
}

#[test]
fn register_split_wins_iff_its_passes_undercut_the_dense_run() {
    let max = clifford_register::MAX_REGISTER_QUBITS;
    let (t, qubits, measured) = (3 * max, 3 * max, 3 * max);
    let dense = 1.0e6;
    let even = dense / REGISTER_PASS_STATEVECTOR_AMPLITUDES;
    let wins = |split: Option<f64>| register_wins(t, qubits, measured, Some(dense), || split);
    assert!(wins(Some(0.9 * even)));
    assert!(!wins(Some(1.1 * even)));
    assert!(!wins(None));
}

#[test]
fn register_work_grows_with_width_and_caps_measured_rows_at_width() {
    for width in 0..8 {
        let here = clifford_register_work(width, 100.0, width);
        assert!(clifford_register_work(width + 1, 100.0, width + 1) > here);
        assert!(clifford_register_work(width, 200.0, width) > here);
        assert_eq!(clifford_register_work(width, 100.0, width + 7), here);
        if width > 0 {
            assert!(clifford_register_work(width, 100.0, width - 1) < here);
        }
    }
}

#[test]
fn walk_histories_counts_two_per_split_ahead_of_the_closing_measurements() {
    let mut terminal = Circuit::new(2, 2);
    terminal.add_gate(Gate::H, &[0]);
    terminal.add_measure(0, 0);
    terminal.add_measure(1, 1);
    assert_eq!(walk_histories(&terminal, 1000), 1);

    let chain = h_measure_chain(3);
    assert_eq!(walk_histories(&chain, 1000), 8);
    assert_eq!(walk_histories(&chain, 5), 5);
    assert_eq!(walk_histories(&chain, 0), 1);

    // A barrier inside the closing run does not end it.
    let mut closing = h_measure_chain(1);
    closing.add_barrier(&[0]);
    closing.add_measure(0, 0);
    assert_eq!(walk_histories(&closing, 1000), 2);

    // A reset of an entangled qubit splits; one on a product qubit would not.
    let mut reset = Circuit::new(2, 1);
    reset.add_gate(Gate::H, &[0]);
    reset.add_gate(Gate::Cx, &[0, 1]);
    reset.add_reset(0);
    reset.add_gate(Gate::H, &[0]);
    reset.add_measure(0, 0);
    assert_eq!(walk_histories(&reset, 1000), 2);

    assert_eq!(walk_histories(&h_measure_chain(70), 12345), 12345);
}

#[test]
fn tensor_expectations_start_at_their_width() {
    let observables = vec![vec![PauliTerm::z(0)]];
    let below = Circuit::new(TENSOR_ROUTE_MIN_QUBITS - 1, 0);
    assert!(decide_tensor_expectations(&below, &observables, &limits()).is_none());

    let at = Circuit::new(TENSOR_ROUTE_MIN_QUBITS, 0);
    assert!(decide_tensor_expectations(&at, &[], &limits()).is_none());
    // Observables are validated before any dry run.
    let bad = vec![vec![PauliTerm::z(TENSOR_ROUTE_MIN_QUBITS)]];
    assert!(matches!(
        decide_tensor_expectations(&at, &bad, &limits()),
        Some(Err(PrismError::InvalidQubit { .. }))
    ));
}

#[test]
fn tensor_over_mps_declines_without_observables() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    assert!(decide_tensor_over_mps(&circuit, &[], &limits()).is_none());
}

#[test]
fn register_subset_declines_past_its_width_and_on_measured_circuits() {
    let mut circuit = Circuit::new(MAX_REGISTER_SUBSET_QUBITS + 2, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    let wide: Vec<usize> = (0..=MAX_REGISTER_SUBSET_QUBITS).collect();
    assert!(decide_register_subset(&circuit, &wide, &limits()).is_none());

    circuit.add_measure(0, 0);
    assert!(decide_register_subset(&circuit, &[0], &limits()).is_none());
}

#[test]
fn terminal_statevector_candidate_names_only_statevector_kinds_when_explicit() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let limits = limits();
    assert!(terminal_statevector_candidate(
        &BackendKind::Statevector,
        &circuit,
        &limits
    ));
    for kind in [
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::TensorNetwork,
        BackendKind::DensityMatrix,
    ] {
        assert!(
            !terminal_statevector_candidate(&kind, &circuit, &limits),
            "{kind:?}"
        );
    }
}

#[test]
fn explicit_pauli_kinds_keep_their_expectation_route_after_validation() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    let observables = vec![vec![PauliTerm::z(0)]];
    let limits = limits();
    let stochastic = BackendKind::StochasticPauli { num_samples: 17 };
    assert!(matches!(
        decide_expectations(&stochastic, &circuit, &observables, &limits),
        Ok(ExpectationRoute::Stochastic(17))
    ));
    let bad = vec![vec![PauliTerm::z(2)]];
    assert!(matches!(
        decide_expectations(&stochastic, &circuit, &bad, &limits),
        Err(PrismError::InvalidQubit { .. })
    ));
}

#[test]
fn clifford_expectations_take_exact_pauli_propagation_under_auto() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let observables = vec![vec![PauliTerm::z(0), PauliTerm::z(1)]];
    let route = decide_expectations(&BackendKind::Auto, &circuit, &observables, &limits()).unwrap();
    let ExpectationRoute::Deterministic(truncation) = route else {
        panic!("a Clifford circuit takes exact Pauli propagation");
    };
    assert!(!truncation.can_discard());
    assert_eq!(
        expectation_rule(&route, &BackendKind::Auto),
        RouteRule::SpdExact
    );
}

#[test]
fn expectation_rule_reports_explicit_kinds_as_explicit() {
    let exact = ExpectationRoute::Deterministic(SpdTruncation::Threshold {
        epsilon: 0.0,
        max_terms: 0,
    });
    let truncating = ExpectationRoute::Deterministic(SpdTruncation::Threshold {
        epsilon: 1e-6,
        max_terms: 0,
    });
    let explicit = BackendKind::DeterministicPauli {
        truncation: SpdTruncation::Threshold {
            epsilon: 0.0,
            max_terms: 0,
        },
    };
    assert_eq!(
        expectation_rule(&exact, &explicit),
        RouteRule::ExplicitBackend
    );
    assert_eq!(
        expectation_rule(&truncating, &BackendKind::Auto),
        RouteRule::Default
    );
    assert_eq!(
        expectation_rule(&ExpectationRoute::Statevector, &BackendKind::Statevector),
        RouteRule::ExplicitBackend
    );
    assert_eq!(
        expectation_rule(&ExpectationRoute::Statevector, &BackendKind::Auto),
        RouteRule::Default
    );
}

#[test]
fn backend_rule_names_the_auto_family_and_explicit_otherwise() {
    let mut entangled = Circuit::new(2, 0);
    entangled.add_gate(Gate::Cx, &[0, 1]);
    let product = Circuit::new(2, 0);
    for (plan, circuit, rule) in [
        (
            BackendPlan::ProductState,
            &product,
            RouteRule::NoEntanglingGates,
        ),
        (
            BackendPlan::FactoredStabilizer,
            &entangled,
            RouteRule::LargeCliffordBlocks,
        ),
        (
            BackendPlan::Sparse,
            &entangled,
            RouteRule::OverStatevectorCapSparse,
        ),
        (
            BackendPlan::Factored,
            &entangled,
            RouteRule::PartialIndependence,
        ),
        (BackendPlan::Factored, &product, RouteRule::Default),
    ] {
        assert_eq!(backend_rule(&BackendKind::Auto, circuit, &plan), rule);
        assert_eq!(
            backend_rule(&BackendKind::Sparse, circuit, &plan),
            RouteRule::ExplicitBackend
        );
    }
}

#[test]
fn decide_tensor_over_mps_for_plan_declines_when_planning_overhead_exhausts_budget() {
    let mut circuit = Circuit::new(32, 0);
    for layer in 0..4 {
        for q in (layer % 2..31).step_by(2) {
            circuit.add_gate(Gate::Ry(0.4), &[q]);
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    let (mps_cost, _) = mps_apply_cost(&circuit, 256);
    let too_many_obs = (mps_cost / TENSOR_PLAN_MPS_COST_PER_OBSERVABLE) as usize + 1;
    let observables: Vec<Vec<PauliTerm>> = (0..too_many_obs)
        .map(|k| vec![PauliTerm::z(k % 32)])
        .collect();
    let plan = BackendPlan::Mps { max_bond_dim: 256 };
    assert!(
        decide_tensor_over_mps_for_plan(&circuit, &observables, &plan, &limits()).is_none(),
        "planning overhead exceeding mps_cost must decline before planning"
    );
}

#[test]
fn decide_expectations_above_cap_probes_sparse_at_most_once() {
    let limits = ResolvedLimits::resolve(Some(&crate::Limits::new().max_statevector_qubits(8)));
    let w_state = crate::circuits::w_state_circuit(32);
    let obs = vec![vec![PauliTerm::z(0)]];

    super::dispatch::SPARSE_PROBES.with(|c| c.set(0));
    let route = decide_expectations(&BackendKind::Auto, &w_state, &obs, &limits).unwrap();
    assert!(matches!(
        route,
        ExpectationRoute::Native(BackendPlan::Sparse)
    ));
    assert_eq!(super::dispatch::SPARSE_PROBES.with(|c| c.get()), 1);

    let hea = crate::circuits::hardware_efficient_ansatz(32, 2, 42);
    let hea_obs: Vec<Vec<PauliTerm>> = (0..8).map(|k| vec![PauliTerm::z(k * 4)]).collect();
    super::dispatch::SPARSE_PROBES.with(|c| c.set(0));
    let route = decide_expectations(&BackendKind::Auto, &hea, &hea_obs, &limits).unwrap();
    assert!(matches!(
        route,
        ExpectationRoute::Native(BackendPlan::Mps { .. })
    ));
    assert_eq!(super::dispatch::SPARSE_PROBES.with(|c| c.get()), 1);

    super::dispatch::SPARSE_PROBES.with(|c| c.set(0));
    let marginal = decide_marginals_after_spd(&BackendKind::Auto, &w_state, &limits).unwrap();
    assert!(matches!(
        marginal,
        MarginalRoute::Native(BackendPlan::Sparse)
    ));
    assert_eq!(super::dispatch::SPARSE_PROBES.with(|c| c.get()), 1);
}

#[test]
fn auto_sparse_shots_reuses_probe_state_without_reexecuting() {
    let limits = ResolvedLimits::resolve(Some(&crate::Limits::new().max_statevector_qubits(16)));
    let mut circuit = crate::circuits::w_state_circuit(32);
    circuit.num_classical_bits = 32;
    circuit.measure_all();

    super::dispatch::SPARSE_PROBES.with(|c| c.set(0));
    super::shot_routes::NATIVE_SHOT_SOURCE_EXECUTIONS.with(|c| c.set(0));
    let (auto_counts, meta) =
        super::shot_routes::run_counts_with(BackendKind::Auto, &circuit, 32, 42, &limits).unwrap();
    assert_eq!(meta.backend, ResolvedBackend::Sparse);
    assert_eq!(super::dispatch::SPARSE_PROBES.with(|c| c.get()), 1);
    assert_eq!(
        super::shot_routes::NATIVE_SHOT_SOURCE_EXECUTIONS.with(|c| c.get()),
        0
    );

    let (sparse_counts, _) =
        super::shot_routes::run_counts_with(BackendKind::Sparse, &circuit, 32, 42, &limits)
            .unwrap();
    assert_eq!(auto_counts, sparse_counts);

    let auto_shots =
        super::shot_routes::run_shots_with(BackendKind::Auto, &circuit, 16, 42, &limits).unwrap();
    let sparse_shots =
        super::shot_routes::run_shots_with(BackendKind::Sparse, &circuit, 16, 42, &limits).unwrap();
    assert_eq!(auto_shots.shots, sparse_shots.shots);
}
