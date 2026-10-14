use prism_q::{
    BackendKind, Circuit, Gate, Limits, PauliTerm, PlannedExactness, Query, RouteRule, simulate,
};

fn circuits() -> Vec<Circuit> {
    let mut product = Circuit::new(3, 3);
    product.add_gate(Gate::Ry(0.3), &[0]);
    let mut bell = Circuit::new(3, 3);
    bell.add_gate(Gate::H, &[0]);
    bell.add_gate(Gate::Cx, &[0, 1]);
    let mut magic = bell.clone();
    magic.add_gate(Gate::Ry(0.4), &[1]);
    magic.add_gate(Gate::Cx, &[1, 2]);
    vec![product, bell, magic]
}

#[test]
fn plan_run_and_marginal_corpus_matches_execution() {
    for circuit in circuits() {
        for kind in [
            BackendKind::Auto,
            BackendKind::Statevector,
            BackendKind::ProductState,
            BackendKind::Stabilizer,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 8 },
            BackendKind::Factored,
            BackendKind::DensityMatrix,
            BackendKind::TensorNetwork,
        ] {
            let request = || simulate(&circuit).backend(kind.clone()).seed(42);
            let plan = request().plan(Query::Run).unwrap();
            match request().run() {
                Ok(result) => {
                    assert!(plan.would_fail.is_none(), "{kind:?}: {:?}", plan.would_fail);
                    assert_eq!(plan.backend, result.metadata.backend, "{kind:?}");
                    assert_eq!(plan.rule, result.metadata.reason, "{kind:?}");
                }
                Err(_) => assert!(plan.would_fail.is_some(), "{kind:?}"),
            }
            let plan = request().plan(Query::Marginals).unwrap();
            match request().marginals() {
                Ok(result) => {
                    assert!(plan.would_fail.is_none(), "{kind:?}");
                    assert_eq!(plan.backend, result.metadata.backend, "{kind:?}");
                    assert_eq!(plan.rule, result.metadata.reason, "{kind:?}");
                }
                Err(_) => assert!(plan.would_fail.is_some(), "{kind:?}"),
            }
        }
    }
}

#[test]
fn plan_expectation_corpus_matches_execution() {
    let obs = vec![vec![PauliTerm::z(0)], vec![PauliTerm::x(1)]];
    for circuit in circuits() {
        for kind in [
            BackendKind::Auto,
            BackendKind::Statevector,
            BackendKind::Stabilizer,
            BackendKind::Mps { max_bond_dim: 8 },
            BackendKind::TensorNetwork,
        ] {
            let request = || simulate(&circuit).backend(kind.clone()).seed(42);
            let plan = request().plan(Query::Expectation(&obs)).unwrap();
            match request().expectation_values_reported(&obs) {
                Ok(result) => {
                    assert!(plan.would_fail.is_none(), "{kind:?}");
                    assert_eq!(plan.backend, result.metadata.backend, "{kind:?}");
                    assert_eq!(plan.rule, result.metadata.reason, "{kind:?}");
                }
                Err(_) => assert!(plan.would_fail.is_some(), "{kind:?}"),
            }
        }
    }
}

#[test]
fn plan_shots_counts_and_diagnostics_match_execution() {
    for circuit in circuits() {
        let request = || simulate(&circuit).seed(42);
        let plan = request().plan(Query::ReducedDensityMatrix(&[0])).unwrap();
        let actual = request().reduced_density_matrix(&[0]).unwrap();
        assert_eq!(plan.backend, actual.metadata.backend);
        assert_eq!(plan.rule, actual.metadata.reason);
        let mut measured = circuit.clone();
        for q in 0..3 {
            measured.add_measure(q, q);
        }
        for kind in [
            BackendKind::Auto,
            BackendKind::Statevector,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 8 },
        ] {
            let request = || simulate(&measured).backend(kind.clone()).seed(42);
            let plan = request().plan(Query::Shots(100)).unwrap();
            let actual = request().shots(100).unwrap();
            assert_eq!(plan.backend, actual.metadata.backend, "{kind:?}");
            assert_eq!(plan.rule, actual.metadata.reason, "{kind:?}");
            let plan = request().plan(Query::Counts(100)).unwrap();
            let actual = request().sample_counts(100).unwrap();
            assert_eq!(plan.backend, actual.metadata.backend, "{kind:?}");
            assert_eq!(plan.rule, actual.metadata.reason, "{kind:?}");
        }
    }
}

#[test]
fn plan_preserves_request_and_distinguishes_malformed_queries() {
    let circuit = circuits().pop().unwrap();
    let request = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .limits(Limits::new().max_statevector_qubits(2));
    assert!(request.plan(Query::Run).unwrap().would_fail.is_some());
    assert!(request.plan(Query::ReducedDensityMatrix(&[9])).is_err());
    assert!(request.plan(Query::Run).unwrap().would_fail.is_some());
    let plan = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 2 })
        .plan(Query::Run)
        .unwrap();
    assert_eq!(
        plan.exactness,
        PlannedExactness::Approximate {
            bound_reported: true
        }
    );
    assert_eq!(plan.rule, RouteRule::ExplicitBackend);
}

#[test]
fn exact_expectations_follow_query_route_above_statevector_cap() {
    let mut circuit = Circuit::new(18, 0);
    for q in 0..18 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..17 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let obs = vec![vec![PauliTerm::z(0)]];
    let request = || {
        simulate(&circuit)
            .limits(Limits::new().max_statevector_qubits(2))
            .require_exact()
            .seed(42)
    };
    let plan = request().plan(Query::Expectation(&obs)).unwrap();
    assert_eq!(plan.rule, RouteRule::SpdExact);
    assert!(plan.would_fail.is_none());
    assert_eq!(
        request()
            .expectation_values_reported(&obs)
            .unwrap()
            .metadata
            .reason,
        plan.rule
    );
}

#[test]
fn gradient_plans_include_actual_inputs_and_shifted_evaluations() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::Ry(0.3), &[0]);
    let parameters = prism_q::Parameters::all_rotations(&circuit);
    let hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
    let request = simulate(&circuit);
    let plan = request
        .plan(Query::GradientShift {
            hamiltonian: &hamiltonian,
            parameters: &parameters,
        })
        .unwrap();
    assert_eq!(plan.rule, RouteRule::ParameterShift);
    assert_eq!(plan.evaluations.len(), 3);
    assert!(
        plan.evaluations
            .iter()
            .all(|evaluation| evaluation.would_fail.is_none())
    );
    let gradient = request
        .seed(42)
        .expectation_gradient_shift(&hamiltonian, &parameters)
        .unwrap();
    assert!((gradient.gradient[0] + 0.3f64.sin()).abs() < 1e-10);
}

#[test]
fn started_state_diagnostics_report_export_limits() {
    let circuit = Circuit::new(3, 0);
    let mut state = vec![num_complex::Complex64::new(0.0, 0.0); 8];
    state[0].re = 1.0;
    let plan = simulate(&circuit)
        .initial_state(&state)
        .limits(Limits::new().max_export_qubits(2))
        .plan(Query::StateVector)
        .unwrap();
    assert!(plan.would_fail.is_some());
}

#[test]
fn simple_dense_buffer_bound_includes_export_copy() {
    let circuit = circuits().pop().unwrap();
    let plan = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .plan(Query::StateVector)
        .unwrap();
    assert_eq!(
        plan.peak_memory.kind,
        prism_q::MemoryEstimateKind::UpperBound
    );
    assert!(plan.peak_memory.bytes.unwrap() >= 32 * 8);
}

#[test]
fn dense_probability_cap_is_a_predictable_run_failure() {
    let circuit = circuits().pop().unwrap();
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::Statevector)
            .limits(Limits::new().max_probability_qubits(2))
            .seed(42)
    };
    assert!(request().plan(Query::Run).unwrap().would_fail.is_some());
    assert!(request().run().is_err());
}

#[test]
fn noisy_sampler_and_mixture_plans_match_execution() {
    for circuit in circuits() {
        let mut measured = circuit.clone();
        for q in 0..3 {
            measured.add_measure(q, q);
        }
        let model = prism_q::NoiseModel::uniform_depolarizing(&measured, 0.01);
        for kind in [
            BackendKind::Auto,
            BackendKind::Statevector,
            BackendKind::DensityMatrix,
        ] {
            let request = || {
                simulate(&measured)
                    .backend(kind.clone())
                    .noise(&model)
                    .seed(42)
            };
            let plan = request().plan(Query::Shots(10)).unwrap();
            let actual = request().shots(10).unwrap();
            assert_eq!(plan.backend, actual.metadata.backend, "{kind:?}");
            assert_eq!(plan.rule, actual.metadata.reason, "{kind:?}");
            assert_eq!(plan.engine, actual.metadata.engine, "{kind:?}");
        }
    }
}

#[test]
fn start_state_reports_preserve_route_reason() {
    let circuit = circuits().pop().unwrap();
    let mut state = vec![num_complex::Complex64::new(0.0, 0.0); 8];
    state[0].re = 1.0;
    let request = || simulate(&circuit).initial_state(&state).seed(42);
    assert_eq!(
        request().plan(Query::Marginals).unwrap().rule,
        request().marginals().unwrap().metadata.reason
    );
    assert_eq!(
        request()
            .plan(Query::ReducedDensityMatrix(&[0]))
            .unwrap()
            .rule,
        request()
            .reduced_density_matrix(&[0])
            .unwrap()
            .metadata
            .reason
    );
}

#[test]
fn noisy_mixture_preconditions_match_execution() {
    let mut circuit = Circuit::new(2, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .seed(42)
    };
    assert!(request().plan(Query::Run).unwrap().would_fail.is_some());
    assert!(request().run().is_err());
    assert!(
        request()
            .plan(Query::Shots(10))
            .unwrap()
            .would_fail
            .is_some()
    );
    assert!(request().shots(10).is_err());

    let circuit = Circuit::new(2, 0);
    let noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let invalid_state = [num_complex::Complex64::new(0.0, 0.0); 4];
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .initial_state(&invalid_state)
            .seed(42)
    };
    assert!(request().plan(Query::Marginals).is_err());
    assert!(request().marginals().is_err());
}

#[test]
fn malformed_gradient_slots_are_rejected_by_plan_and_execution() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Ry(0.2), &[0]);
    let parameters = prism_q::Parameters::from_links(
        vec![prism_q::ParamLink {
            instruction: 0,
            slot: 1,
        }],
        0,
    );
    let hamiltonian = [(1.0, vec![PauliTerm::z(0)])];
    assert!(
        simulate(&circuit)
            .plan(Query::Gradient {
                hamiltonian: &hamiltonian,
                parameters: &parameters
            })
            .is_err()
    );
    assert!(
        simulate(&circuit)
            .seed(42)
            .expectation_gradient(&hamiltonian, &parameters)
            .is_err()
    );
    assert!(
        simulate(&circuit)
            .plan(Query::GradientShift {
                hamiltonian: &hamiltonian,
                parameters: &parameters
            })
            .is_err()
    );
}

#[test]
fn tensor_slice_budget_failure_is_reported_before_execution() {
    let circuit = circuits().pop().unwrap();
    let observables = [vec![PauliTerm::z(0), PauliTerm::z(2)]];
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::TensorNetwork)
            .limits(Limits::new().max_tn_peak_qubits(0).max_tn_slices(1))
            .seed(42)
    };
    let plan = request().plan(Query::Expectation(&observables)).unwrap();
    assert!(matches!(
        plan.would_fail,
        Some(prism_q::PrismError::ResourceLimit(_))
    ));
    assert!(matches!(
        request().expectation_values(&observables),
        Err(prism_q::PrismError::ResourceLimit(_))
    ));
}

#[test]
fn sparse_basis_index_limit_matches_execution() {
    let n = usize::BITS as usize + 1;
    let mut circuit = Circuit::new(n, 0);
    for q in 1..n {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    let request = || simulate(&circuit).backend(BackendKind::Sparse).seed(42);
    assert!(matches!(
        request().plan(Query::Run).unwrap().would_fail,
        Some(prism_q::PrismError::ResourceLimit(_))
    ));
    assert!(matches!(
        request().run(),
        Err(prism_q::PrismError::ResourceLimit(_))
    ));
}

#[test]
fn observable_and_entropy_plans_match_execution() {
    let observable = prism_q::PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    for circuit in circuits() {
        for kind in [
            BackendKind::Auto,
            BackendKind::Statevector,
            BackendKind::Mps { max_bond_dim: 8 },
            BackendKind::TensorNetwork,
        ] {
            let request = || simulate(&circuit).backend(kind.clone()).seed(42);
            let plan = request().plan(Query::Observable(&observable)).unwrap();
            let actual = request().observable_expectation(&observable).unwrap();
            assert_eq!(plan.backend, actual.metadata.backend, "{kind:?}");
            assert_eq!(plan.rule, actual.metadata.reason, "{kind:?}");
            let plan = request().plan(Query::Entropy(&[0])).unwrap();
            match request().entanglement_entropy(&[0]) {
                Ok(actual) => {
                    assert_eq!(plan.backend, actual.metadata.backend);
                    assert_eq!(plan.rule, actual.metadata.reason);
                }
                Err(_) => assert!(plan.would_fail.is_some()),
            }
        }
    }
    let mut measured = circuits().pop().unwrap();
    measured.add_measure(0, 0);
    assert!(
        simulate(&measured)
            .plan(Query::Observable(&observable))
            .unwrap()
            .would_fail
            .is_some()
    );
    assert!(
        simulate(&measured)
            .seed(42)
            .observable_expectation(&observable)
            .is_err()
    );
}

#[cfg(feature = "gpu")]
#[test]
fn device_plan_matches_execution_and_adjoint_keeps_host_limits() {
    let context = match prism_q::gpu::GpuContext::new(0) {
        Ok(context) => context,
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "GPU required: {error}"
            );
            eprintln!("SKIP: no usable GPU ({error})");
            return;
        }
    };
    let mut circuit = Circuit::new(prism_q::gpu::min_qubits().max(3), 0);
    circuit.add_gate(Gate::Ry(0.4), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let observable = prism_q::PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::StatevectorGpu {
                context: context.clone(),
            })
            .limits(Limits::new().max_statevector_qubits(2))
            .seed(42)
    };
    let plan = request().plan(Query::Observable(&observable)).unwrap();
    assert!(plan.would_fail.is_none(), "{:?}", plan.would_fail);
    assert_eq!(plan.placement, prism_q::Placement::Device);
    let actual = request().observable_expectation(&observable).unwrap();
    assert_eq!(plan.placement, actual.metadata.placement);
    assert_eq!(plan.rule, actual.metadata.reason);

    let mut circuit = Circuit::new(24, 0);
    circuit.add_gate(Gate::Ry(0.2), &[0]);
    let parameters = prism_q::Parameters::all_rotations(&circuit);
    let hamiltonian = [(1.0, vec![PauliTerm::z(0)])];
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::AutoGpu {
                context: context.clone(),
            })
            .limits(Limits::new().max_statevector_qubits(2))
            .seed(42)
    };
    let plan = request()
        .plan(Query::Gradient {
            hamiltonian: &hamiltonian,
            parameters: &parameters,
        })
        .unwrap();
    assert_eq!(plan.placement, prism_q::Placement::Host);
    assert!(matches!(
        plan.would_fail,
        Some(prism_q::PrismError::ResourceLimit(_))
    ));
    assert!(matches!(
        request().expectation_gradient(&hamiltonian, &parameters),
        Err(prism_q::PrismError::ResourceLimit(_))
    ));
}

#[test]
fn successful_spd_primary_is_not_rejected_for_a_failed_tensor_fallback() {
    let mut circuit = Circuit::new(18, 0);
    for q in 0..18 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 1..18 {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    circuit.add_gate(Gate::T, &[0]);
    let request = || {
        simulate(&circuit)
            .limits(
                Limits::new()
                    .max_statevector_qubits(20)
                    .max_tn_peak_qubits(0)
                    .max_tn_slices(1),
            )
            .require_exact()
            .seed(42)
    };
    let plan = request().plan(Query::Marginals).unwrap();
    assert_eq!(plan.rule, RouteRule::SpdExact);
    assert!(plan.would_fail.is_none());
    assert!(
        plan.fallbacks
            .iter()
            .any(|fallback| fallback.would_fail.is_some())
    );
    assert_eq!(
        request().marginals().unwrap().metadata.reason,
        RouteRule::SpdExact
    );
}
