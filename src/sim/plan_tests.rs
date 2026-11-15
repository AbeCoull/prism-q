use super::*;
use crate::circuit::SaveSpec;
use crate::circuit::fusion::MIN_QUBITS_FOR_FUSION;
use crate::gates::Gate;

fn ghz(n: usize, classical: usize) -> Circuit {
    let mut circuit = Circuit::new(n, classical);
    circuit.add_gate(Gate::H, &[0]);
    for q in 1..n {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    circuit
}

fn with_t(mut circuit: Circuit) -> Circuit {
    circuit.add_gate(Gate::T, &[0]);
    circuit
}

fn upper_bound(bytes: u128) -> MemoryEstimate {
    MemoryEstimate {
        bytes: Some(bytes),
        kind: MemoryEstimateKind::UpperBound,
    }
}

const UNKNOWN: MemoryEstimate = MemoryEstimate {
    bytes: None,
    kind: MemoryEstimateKind::Unknown,
};

#[test]
fn statevector_bound_is_state_plus_output_plus_classical_bits() {
    // 2^n amplitudes of 16 bytes, an output copy of the same size, and two bytes per
    // classical bit.
    let circuit = ghz(3, 2);
    let plan = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .plan(Query::Run)
        .unwrap();
    assert_eq!(plan.peak_memory, upper_bound(32 * 8 + 4));
    let plan = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .plan(Query::StateVector)
        .unwrap();
    assert_eq!(plan.peak_memory, upper_bound(32 * 8 + 4));
}

#[test]
fn product_state_bound_covers_the_probability_output() {
    // 8 bytes per output probability plus 64 bytes of state per qubit.
    let mut circuit = Circuit::new(3, 1);
    for q in 0..3 {
        circuit.add_gate(Gate::H, &[q]);
    }
    let plan = simulate(&circuit).plan(Query::Run).unwrap();
    assert_eq!(plan.backend, ResolvedBackend::ProductState);
    assert_eq!(plan.peak_memory, upper_bound(8 * 8 + 64 * 3 + 2));
    let plan = simulate(&circuit).plan(Query::StateVector).unwrap();
    assert_eq!(plan.peak_memory, UNKNOWN);
}

#[test]
fn statevector_bound_stops_where_fusion_starts() {
    let below = ghz(MIN_QUBITS_FOR_FUSION - 1, 0);
    let plan = simulate(&below)
        .backend(BackendKind::Statevector)
        .plan(Query::Run)
        .unwrap();
    assert_eq!(
        plan.peak_memory,
        upper_bound(32 << (MIN_QUBITS_FOR_FUSION - 1))
    );

    let at = ghz(MIN_QUBITS_FOR_FUSION, 0);
    let plan = simulate(&at)
        .backend(BackendKind::Statevector)
        .plan(Query::Run)
        .unwrap();
    assert_eq!(plan.peak_memory, UNKNOWN);
}

#[test]
fn memory_is_unknown_off_the_simple_run_shapes() {
    let circuit = ghz(3, 0);
    let sv = || simulate(&circuit).backend(BackendKind::Statevector);
    assert_eq!(sv().plan(Query::Shots(10)).unwrap().peak_memory, UNKNOWN);
    assert_eq!(sv().plan(Query::Marginals).unwrap().peak_memory, UNKNOWN);

    let mut other_gate = ghz(3, 0);
    other_gate.add_gate(Gate::SX, &[2]);
    let plan = simulate(&other_gate)
        .backend(BackendKind::Statevector)
        .plan(Query::Run)
        .unwrap();
    assert_eq!(plan.peak_memory, UNKNOWN);

    let mut saved = ghz(3, 0);
    saved.add_save(SaveSpec::Probabilities, "mid");
    let plan = simulate(&saved)
        .backend(BackendKind::Statevector)
        .plan(Query::Run)
        .unwrap();
    assert_eq!(plan.peak_memory, UNKNOWN);
}

#[test]
fn save_points_fail_every_terminal_but_run() {
    let mut circuit = ghz(2, 0);
    circuit.add_save(SaveSpec::Probabilities, "mid");
    assert!(
        simulate(&circuit)
            .plan(Query::Run)
            .unwrap()
            .would_fail
            .is_none()
    );
    for query in [Query::Marginals, Query::StateVector, Query::Shots(4)] {
        let plan = simulate(&circuit).plan(query).unwrap();
        let Some(PrismError::IncompatibleBackend { reason, .. }) = plan.would_fail else {
            panic!("{query:?}: a save point must fail the plan");
        };
        assert!(reason.contains("save points"), "{query:?}: {reason}");
    }
}

#[test]
fn require_exact_fails_an_approximate_route_only() {
    let circuit = with_t(ghz(2, 0));
    let kind = BackendKind::StochasticPauli { num_samples: 64 };
    let plan = simulate(&circuit)
        .backend(kind.clone())
        .plan(Query::Marginals)
        .unwrap();
    assert_eq!(plan.exactness, PlannedExactness::Estimated { samples: 64 });
    assert!(plan.would_fail.is_none());

    let plan = simulate(&circuit)
        .backend(kind)
        .require_exact()
        .plan(Query::Marginals)
        .unwrap();
    let Some(PrismError::IncompatibleBackend { reason, .. }) = plan.would_fail else {
        panic!("require_exact must fail an estimated route");
    };
    assert!(reason.contains("require_exact"), "{reason}");

    let plan = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .require_exact()
        .plan(Query::Marginals)
        .unwrap();
    assert!(plan.would_fail.is_none());
}

#[test]
fn malformed_queries_are_errors_and_capability_failures_are_plans() {
    let circuit = ghz(2, 0);
    let bad = vec![vec![PauliTerm::z(2)]];
    assert!(matches!(
        simulate(&circuit).plan(Query::Expectation(&bad)),
        Err(PrismError::InvalidQubit { .. })
    ));
    assert!(
        simulate(&circuit)
            .plan(Query::ReducedDensityMatrix(&[0, 0]))
            .is_err()
    );

    let plan = simulate(&circuit)
        .backend(BackendKind::StochasticPauli { num_samples: 8 })
        .plan(Query::Run)
        .unwrap();
    assert!(
        matches!(
            plan.would_fail,
            Some(PrismError::IncompatibleBackend { .. })
        ),
        "{:?}",
        plan.would_fail
    );
    assert_eq!(plan.exactness, PlannedExactness::Estimated { samples: 8 });
}

#[test]
fn alternatives_list_the_explicit_backends_that_would_refuse() {
    let rejected = |circuit: &Circuit| -> Vec<ResolvedBackend> {
        simulate(circuit)
            .plan(Query::Run)
            .unwrap()
            .alternatives
            .into_iter()
            .map(|(backend, _)| backend)
            .collect()
    };
    let clifford = rejected(&ghz(2, 0));
    assert!(
        !clifford.contains(&ResolvedBackend::Stabilizer),
        "{clifford:?}"
    );
    assert!(
        !clifford.contains(&ResolvedBackend::Statevector),
        "{clifford:?}"
    );

    let clifford_t = rejected(&with_t(ghz(2, 0)));
    assert!(
        clifford_t.contains(&ResolvedBackend::Stabilizer),
        "{clifford_t:?}"
    );
    assert!(
        !clifford_t.contains(&ResolvedBackend::Statevector),
        "{clifford_t:?}"
    );
}

#[test]
fn pauli_plan_is_approximate_iff_the_truncation_can_drop_terms() {
    let exact = SpdTruncation::Threshold {
        epsilon: 0.0,
        max_terms: 0,
    };
    assert_eq!(
        pauli_plan(exact, RouteRule::SpdExact).exactness,
        PlannedExactness::Exact
    );
    for truncation in [
        SpdTruncation::Threshold {
            epsilon: 1e-9,
            max_terms: 0,
        },
        SpdTruncation::Budget { max_terms: 10 },
    ] {
        let plan = pauli_plan(truncation, RouteRule::ExplicitBackend);
        assert_eq!(
            plan.exactness,
            PlannedExactness::Approximate {
                bound_reported: false
            }
        );
        assert_eq!(plan.backend, ResolvedBackend::DeterministicPauli);
    }
}

#[test]
fn approximation_name_distinguishes_the_bounded_tensor_kind() {
    let bounded = BackendKind::TensorNetworkBounded { tolerance: 1e-6 };
    assert_eq!(
        approximation_name(&bounded, ResolvedBackend::TensorNetwork),
        "TensorNetworkBounded"
    );
    assert_eq!(
        approximation_name(&BackendKind::TensorNetwork, ResolvedBackend::TensorNetwork),
        "TensorNetwork"
    );
    assert_eq!(approximation_name(&bounded, ResolvedBackend::Mps), "Mps");
}
