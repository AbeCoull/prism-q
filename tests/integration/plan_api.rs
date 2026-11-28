use prism_q::{
    BackendKind, Circuit, Exactness, Gate, Limits, Parameters, PauliObservable, PauliTerm,
    Placement, Plan, PlannedExactness, PrismError, Query, ResolvedBackend, RouteRule, RunMetadata,
    SaveSpec, SpdTruncation, simulate,
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
    let hamiltonian = prism_q::PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
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
    let parameters = prism_q::Parameters::from_links(vec![prism_q::ParamLink::new(0, 1)], 0);
    let hamiltonian = prism_q::PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
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
    let hamiltonian = prism_q::PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
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
    // One register part spans all 22 qubits with 21 excess T rotations, so the Clifford
    // register declines and the Pauli sum, one term per marginal, is the primary.
    let n = 22;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 1..n {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    for i in 0..21 {
        let q = (7 * i + 3) % n;
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_gate(Gate::Cx, &[q, (q + 11) % n]);
    }
    let request = || {
        simulate(&circuit)
            .limits(
                Limits::new()
                    .max_statevector_qubits(24)
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

#[test]
fn plan_reduced_density_matrix_matches_execution() {
    for circuit in circuits() {
        for kind in [
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
            let plan = request()
                .plan(Query::ReducedDensityMatrix(&[1, 0]))
                .unwrap();
            match request().reduced_density_matrix(&[1, 0]) {
                Ok(result) => {
                    assert!(plan.would_fail.is_none(), "{kind:?}: {:?}", plan.would_fail);
                    assert_eq!(plan.backend, result.metadata.backend, "{kind:?}");
                    assert_eq!(plan.rule, result.metadata.reason, "{kind:?}");
                }
                Err(_) => assert!(plan.would_fail.is_some(), "{kind:?}"),
            }
        }
    }
}

#[test]
fn plan_reduced_density_matrix_on_the_tensor_network_answers() {
    let circuit = circuits().pop().unwrap();
    let request = || {
        simulate(&circuit)
            .backend(BackendKind::TensorNetwork)
            .seed(42)
    };
    let plan = request()
        .plan(Query::ReducedDensityMatrix(&[2, 0]))
        .unwrap();
    assert!(plan.would_fail.is_none(), "{:?}", plan.would_fail);
    let result = request().reduced_density_matrix(&[2, 0]).unwrap();
    assert_eq!(plan.backend, result.metadata.backend);
}

#[test]
fn wide_gates_over_the_statevector_cap_plan_and_run() {
    let uniform = prism_q::circuits::qft_circuit(6);
    let mut flat = Circuit::new(6, 0);
    for q in 0..6 {
        flat.add_gate(Gate::H, &[q]);
    }
    flat.instructions.extend(uniform.instructions.clone());
    let mut delta = vec![0.0; 64];
    delta[0] = 1.0;

    // exp(-i theta/2 X0 Y3 Z5)|0> = cos(theta/2)|0> + sin(theta/2)|q0 q3>.
    let theta = 0.7_f64;
    let mut rotated = Circuit::new(6, 0);
    rotated.add_pauli_rotation(theta, &[PauliTerm::x(0), PauliTerm::y(3), PauliTerm::z(5)]);
    let mut two_peaks = vec![0.0; 64];
    two_peaks[0] = (theta / 2.0).cos().powi(2);
    two_peaks[0b001001] = (theta / 2.0).sin().powi(2);

    for (name, circuit, expected) in [
        ("qft", uniform, vec![1.0 / 64.0; 64]),
        ("qft of the flat state", flat, delta),
        ("pauli rotation", rotated, two_peaks),
    ] {
        let request = || {
            simulate(&circuit)
                .limits(Limits::new().max_statevector_qubits(4))
                .seed(42)
        };
        let plan = request().plan(Query::Run).unwrap();
        assert!(plan.would_fail.is_none(), "{name}: {:?}", plan.would_fail);
        let result = request().run().unwrap();
        assert_eq!(plan.backend, result.metadata.backend, "{name}");
        assert_eq!(plan.rule, result.metadata.reason, "{name}");
        let probabilities = result.probabilities.unwrap().to_vec();
        for (index, (got, want)) in probabilities.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "{name}: P({index}) = {got}, want {want}"
            );
        }
    }
}

// A guard that fires with certainty over a gate only the statevector applies. The
// QFT case undoes a flat state and the rotation flips its three qubits, so every
// measured bit is certain and a guard that did not lower reads as a wrong record.
fn guarded_wide_gate_circuits() -> [(&'static str, Circuit, Vec<bool>); 2] {
    let build = |flat: bool, wide: Gate| {
        let mut circuit = Circuit::new(5, 5);
        circuit.add_gate(Gate::X, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        if flat {
            for q in 2..5 {
                circuit.add_gate(Gate::H, &[q]);
            }
        }
        circuit.add_measure(0, 0);
        circuit
            .instructions
            .push(prism_q::Instruction::Conditional {
                condition: prism_q::ClassicalCondition::BitIsOne(0),
                gate: wide,
                targets: [2, 3, 4].into_iter().collect(),
            });
        for q in 1..5 {
            circuit.add_measure(q, q);
        }
        circuit
    };
    let qft = Gate::QftBlock {
        start: 2,
        num: 3,
        inverse: false,
        swaps: true,
        big_endian: false,
    };
    let mut rotation = Circuit::new(5, 0);
    rotation.add_pauli_rotation(
        std::f64::consts::PI,
        &[PauliTerm::x(2), PauliTerm::x(3), PauliTerm::x(4)],
    );
    let prism_q::Instruction::Gate { gate: rotation, .. } = rotation.instructions.remove(0) else {
        unreachable!()
    };
    [
        (
            "qft",
            build(true, qft),
            vec![true, true, false, false, false],
        ),
        ("pauli rotation", build(false, rotation), vec![true; 5]),
    ]
}

#[test]
fn a_guarded_wide_gate_lowers_on_every_backend() {
    for (name, circuit, record) in guarded_wide_gate_circuits() {
        let key = vec![
            record
                .iter()
                .rev()
                .fold(0u64, |acc, &bit| acc << 1 | u64::from(bit)),
        ];
        let capped = Limits::new().max_statevector_qubits(4);
        for (kind, limits) in [
            (BackendKind::Statevector, Limits::new()),
            (BackendKind::Sparse, Limits::new()),
            (BackendKind::Mps { max_bond_dim: 8 }, Limits::new()),
            (BackendKind::Factored, Limits::new()),
            (BackendKind::TensorNetwork, Limits::new()),
            (BackendKind::Auto, capped),
        ] {
            let request = || {
                simulate(&circuit)
                    .backend(kind.clone())
                    .limits(limits)
                    .seed(42)
            };
            for query in [Query::Run, Query::Shots(8), Query::Counts(8)] {
                let plan = request().plan(query).unwrap();
                assert!(plan.would_fail.is_none(), "{name} {kind:?}");
            }
            let result = request().run().unwrap();
            assert_eq!(result.classical_bits, record, "{name} {kind:?}");
            let result = request().shots(8).unwrap();
            assert!(
                result.shots.iter().all(|shot| *shot == record),
                "{name} {kind:?}"
            );
            let result = request().sample_counts(8).unwrap();
            assert_eq!(result.counts.get(&key), Some(&8), "{name} {kind:?}");
        }
    }
}

enum MatrixOutcome {
    WithMetadata(RunMetadata),
    WithoutMetadata,
}

fn all_cpu_kinds() -> Vec<BackendKind> {
    vec![
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::ProductState,
        BackendKind::TensorNetwork,
        BackendKind::TensorNetworkBounded { tolerance: 1e-3 },
        BackendKind::Factored,
        BackendKind::StabilizerRank,
        BackendKind::FactoredStabilizer,
        BackendKind::DensityMatrix,
        BackendKind::StochasticPauli { num_samples: 64 },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: 1e-3,
                max_terms: 1024,
            },
        },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Budget { max_terms: 16 },
        },
        BackendKind::PauliPath {
            epsilon: 1e-3,
            max_terms: 1024,
        },
        BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        },
    ]
}

fn matrix_circuits() -> Vec<(&'static str, Circuit)> {
    let mut base = circuits();
    let magic = base.pop().unwrap();
    let bell = base.pop().unwrap();
    let product = base.pop().unwrap();

    let mut clifford_t = Circuit::new(3, 3);
    clifford_t.add_gate(Gate::H, &[0]);
    clifford_t.add_gate(Gate::Cx, &[0, 1]);
    clifford_t.add_gate(Gate::T, &[0]);
    clifford_t.add_gate(Gate::Cx, &[1, 2]);

    let mut measured_bell = bell.clone();
    measured_bell.measure_all();

    let mut measured_clifford_t = clifford_t.clone();
    measured_clifford_t.measure_all();

    vec![
        ("product", product),
        ("bell", bell),
        ("magic", magic),
        ("clifford_t", clifford_t),
        ("measured_bell", measured_bell),
        ("measured_clifford_t", measured_clifford_t),
    ]
}

fn assert_exactness_matches(
    planned: PlannedExactness,
    actual: Exactness,
    shots: Option<usize>,
    label: &str,
) {
    match planned {
        PlannedExactness::Exact => assert_eq!(actual, Exactness::Exact, "{label}"),
        PlannedExactness::Approximate {
            bound_reported: true,
        } => {
            assert!(
                matches!(
                    actual,
                    Exactness::Exact
                        | Exactness::Approximate {
                            fidelity_lower_bound: Some(_)
                        }
                ),
                "{label}: planned {planned:?}, got {actual:?}"
            );
        }
        PlannedExactness::Approximate {
            bound_reported: false,
        } => {
            assert!(
                matches!(
                    actual,
                    Exactness::Exact
                        | Exactness::Approximate {
                            fidelity_lower_bound: None
                        }
                ),
                "{label}: planned {planned:?}, got {actual:?}"
            );
        }
        PlannedExactness::Estimated { samples } => {
            assert!(
                matches!(
                    actual,
                    Exactness::Approximate {
                        fidelity_lower_bound: None
                    }
                ),
                "{label}: planned {planned:?}, got {actual:?}"
            );
            assert_eq!(shots, Some(samples), "{label}: estimated sample count");
        }
        _ => unreachable!("unexpected PlannedExactness variant: {planned:?}"),
    }
}

fn assert_plan_matches_metadata(plan: &Plan, meta: &RunMetadata, query: Query<'_>, label: &str) {
    if let Query::Shots(n) | Query::Counts(n) = query {
        assert_eq!(meta.shots, Some(n), "{label}: shot count");
    }

    if plan.rule == meta.reason {
        assert_eq!(plan.backend, meta.backend, "{label}");
        assert_eq!(plan.placement, meta.placement, "{label}");
        assert_eq!(plan.engine, meta.engine, "{label}");
        assert_exactness_matches(plan.exactness, meta.exactness, meta.shots, label);
    } else if plan.placement == Placement::Device
        && meta.placement == Placement::Host
        && meta.reason == RouteRule::DeviceInitializationFallback
        && plan
            .fallbacks
            .iter()
            .any(|f| f.rule == RouteRule::DeviceInitializationFallback)
    {
        assert_eq!(plan.backend, meta.backend, "{label}");
        assert_eq!(plan.engine, meta.engine, "{label}");
        assert_exactness_matches(plan.exactness, meta.exactness, meta.shots, label);
    } else {
        let fallback = plan
            .fallbacks
            .iter()
            .find(|fb| fb.rule == meta.reason && fb.placement == meta.placement);
        assert!(
            fallback.is_some(),
            "{label}: plan rule {:?} != meta reason {:?} (fallbacks: {:?})",
            plan.rule,
            meta.reason,
            plan.fallbacks
        );
        let fallback = fallback.unwrap();
        assert_eq!(fallback.backend, meta.backend, "{label}: fallback backend");
        assert_exactness_matches(fallback.exactness, meta.exactness, meta.shots, label);
    }
}

fn execute_query_for_matrix(
    circuit: &Circuit,
    kind: &BackendKind,
    limits: Option<Limits>,
    query: Query<'_>,
) -> prism_q::Result<MatrixOutcome> {
    let build = || {
        let mut req = simulate(circuit).backend(kind.clone()).seed(42);
        if let Some(lim) = limits {
            req = req.limits(lim);
        }
        req
    };
    match query {
        Query::Run => build()
            .run()
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Shots(n) => build()
            .shots(n)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Counts(n) => build()
            .sample_counts(n)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Marginals => build()
            .marginals()
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Expectation(obs) => build()
            .expectation_values_reported(obs)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Observable(obs) => build()
            .observable_expectation(obs)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::StateVector => build()
            .state_vector()
            .map(|_| MatrixOutcome::WithoutMetadata),
        Query::ReducedDensityMatrix(q) => build()
            .reduced_density_matrix(q)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Entropy(q) => build()
            .entanglement_entropy(q)
            .map(|r| MatrixOutcome::WithMetadata(r.metadata)),
        Query::Gradient {
            hamiltonian,
            parameters,
        } => build()
            .expectation_gradient(hamiltonian, parameters)
            .map(|_| MatrixOutcome::WithoutMetadata),
        Query::GradientShift {
            hamiltonian,
            parameters,
        } => build()
            .expectation_gradient_shift(hamiltonian, parameters)
            .map(|_| MatrixOutcome::WithoutMetadata),
        _ => unreachable!("unexpected Query variant"),
    }
}

fn assert_contract_for_query(
    circuit: &Circuit,
    kind: &BackendKind,
    limits: Option<Limits>,
    query: Query<'_>,
    label: &str,
) {
    let mut req = simulate(circuit).backend(kind.clone()).seed(42);
    if let Some(lim) = limits {
        req = req.limits(lim);
    }
    let plan_res = req.plan(query);
    let exec_res = execute_query_for_matrix(circuit, kind, limits, query);

    match plan_res {
        Err(plan_err) => {
            let exec_err = exec_res.err().unwrap_or_else(|| {
                panic!("{label}: plan returned Err({plan_err:?}) but execution succeeded")
            });
            assert_eq!(
                std::mem::discriminant(&plan_err),
                std::mem::discriminant(&exec_err),
                "{label}: plan_err={plan_err:?}, exec_err={exec_err:?}"
            );
        }
        Ok(plan) => match (plan.would_fail.as_ref(), exec_res) {
            (Some(pe), Err(ae)) => {
                assert_eq!(
                    std::mem::discriminant(pe),
                    std::mem::discriminant(&ae),
                    "{label}: would_fail={pe:?}, exec_err={ae:?}"
                );
            }
            (None, Ok(MatrixOutcome::WithMetadata(meta))) => {
                assert_plan_matches_metadata(&plan, &meta, query, label);
            }
            (None, Ok(MatrixOutcome::WithoutMetadata)) => {}
            (Some(pe), Ok(_)) => {
                panic!("{label}: plan predicted failure ({pe:?}) but execution succeeded");
            }
            (None, Err(ae)) => {
                panic!(
                    "{label}: plan predicted success (backend={:?}, rule={:?}) but execution failed ({ae:?})",
                    plan.backend, plan.rule
                );
            }
        },
    }
}

fn run_query_matrix_for_kind(
    circuit_name: &str,
    circuit: &Circuit,
    kind: &BackendKind,
    limits: Option<Limits>,
    skip_or_pin: impl Fn(&str, &BackendKind, &Query<'_>) -> bool,
) {
    let obs = vec![vec![PauliTerm::z(0)], vec![PauliTerm::x(1)]];
    let observable = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    let parameters = Parameters::all_rotations(circuit);
    let rdm_qubits: &[usize] = if circuit.num_qubits >= 2 {
        &[1, 0]
    } else {
        &[0]
    };
    let queries = [
        ("Run", Query::Run),
        ("Shots(8)", Query::Shots(8)),
        ("Counts(8)", Query::Counts(8)),
        ("Marginals", Query::Marginals),
        ("Expectation", Query::Expectation(&obs)),
        ("Observable", Query::Observable(&observable)),
        ("StateVector", Query::StateVector),
        (
            "ReducedDensityMatrix",
            Query::ReducedDensityMatrix(rdm_qubits),
        ),
        ("Entropy", Query::Entropy(&[0])),
        (
            "Gradient",
            Query::Gradient {
                hamiltonian: &observable,
                parameters: &parameters,
            },
        ),
        (
            "GradientShift",
            Query::GradientShift {
                hamiltonian: &observable,
                parameters: &parameters,
            },
        ),
    ];

    for (qname, query) in queries {
        if skip_or_pin(circuit_name, kind, &query) {
            continue;
        }
        let label = format!("{circuit_name} / {kind:?} / {qname}");
        assert_contract_for_query(circuit, kind, limits, query, &label);
    }
}

#[test]
fn plan_matrix_covers_every_cpu_backend_and_query() {
    let circuits = matrix_circuits();
    let kinds = all_cpu_kinds();

    for (cname, circuit) in &circuits {
        for kind in &kinds {
            run_query_matrix_for_kind(cname, circuit, kind, None, |_name, _k, _query| false);
        }
    }
}

#[cfg(feature = "distributed")]
#[derive(Debug)]
struct MockRankComm {
    size: usize,
}

#[cfg(feature = "distributed")]
impl prism_q::RankComm for MockRankComm {
    fn rank(&self) -> usize {
        0
    }

    fn size(&self) -> usize {
        self.size
    }

    fn allgather_c64(&self, local: &[num_complex::Complex64]) -> Vec<num_complex::Complex64> {
        std::iter::repeat_n(local, self.size)
            .flatten()
            .copied()
            .collect()
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        std::iter::repeat_n(local, self.size)
            .flatten()
            .copied()
            .collect()
    }

    fn allgatherv_u64(&self, local: &[u64], counts: &[usize]) -> Vec<u64> {
        counts.iter().flat_map(|_| local.iter().copied()).collect()
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        value * self.size as f64
    }

    fn sendrecv_c64(
        &self,
        _partner: usize,
        _send: &[num_complex::Complex64],
        _recv: &mut [num_complex::Complex64],
    ) {
        unreachable!("mock rank comm is used only for early rejection checks")
    }

    fn barrier(&self) {}
}

#[cfg(feature = "distributed")]
#[test]
fn plan_matrix_distributed_serial_and_rank_pins() {
    let serial_kind = BackendKind::StatevectorDistributed {
        context: prism_q::DistributedContext::serial(),
    };
    for (cname, circuit) in &matrix_circuits() {
        run_query_matrix_for_kind(cname, circuit, &serial_kind, None, |_name, _k, _query| {
            false
        });
    }

    let bell = matrix_circuits()
        .into_iter()
        .find(|(name, _)| *name == "bell")
        .unwrap()
        .1;

    // Non power of two rank count: plan and execution both report BackendUnsupported.
    let odd_ranks = BackendKind::StatevectorDistributed {
        context: prism_q::DistributedContext::from_comm(std::sync::Arc::new(MockRankComm {
            size: 3,
        })),
    };
    let obs = vec![vec![PauliTerm::z(0)]];
    let observable = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    let params = Parameters::all_rotations(&bell);
    for query in [
        Query::Run,
        Query::Shots(4),
        Query::Counts(4),
        Query::Marginals,
        Query::Expectation(&obs),
        Query::Observable(&observable),
        Query::StateVector,
        Query::GradientShift {
            hamiltonian: &observable,
            parameters: &params,
        },
    ] {
        assert_contract_for_query(&bell, &odd_ranks, None, query, &format!("3-rank {query:?}"));
    }

    // Eight ranks below rank_bits + min_local_qubits: plan and execution both report BackendUnsupported.
    let below_min_n = 3 + prism_q::distributed::min_local_qubits().saturating_sub(1);
    let mut small_dist = Circuit::new(below_min_n, below_min_n);
    small_dist.add_gate(Gate::H, &[0]);
    small_dist.add_gate(Gate::Cx, &[0, 1]);
    let eight_ranks = BackendKind::StatevectorDistributed {
        context: prism_q::DistributedContext::from_comm(std::sync::Arc::new(MockRankComm {
            size: 8,
        })),
    };
    for query in [Query::Run, Query::Shots(4), Query::Counts(4)] {
        assert_contract_for_query(
            &small_dist,
            &eight_ranks,
            None,
            query,
            &format!("8-rank {query:?}"),
        );
    }

    // Statevector limit per rank: plan and execution both report ResourceLimit.
    let capped = Limits::new().max_statevector_qubits(2);
    assert_contract_for_query(
        &bell,
        &serial_kind,
        Some(capped),
        Query::Run,
        "dist serial sv cap",
    );

    let n_14 = 14;
    let mut c14 = Circuit::new(n_14, 0);
    c14.add_gate(Gate::H, &[0]);
    c14.add_gate(Gate::Cx, &[0, 1]);
    let two_ranks = BackendKind::StatevectorDistributed {
        context: prism_q::DistributedContext::from_comm(std::sync::Arc::new(MockRankComm {
            size: 2,
        })),
    };
    let cap_10 = Limits::new().max_statevector_qubits(10);
    assert_contract_for_query(
        &c14,
        &two_ranks,
        Some(cap_10),
        Query::Run,
        "dist 2-rank sv cap",
    );
}

#[cfg(feature = "gpu")]
fn acquire_gpu_context_or_skip() -> Option<std::sync::Arc<prism_q::gpu::GpuContext>> {
    match prism_q::gpu::GpuContext::new(0) {
        Ok(context) => Some(context),
        Err(error) => {
            assert!(
                std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                "GPU required: {error}"
            );
            eprintln!("SKIP: no usable GPU ({error})");
            None
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn device_plan_matrix_matches_execution() {
    let Some(context) = acquire_gpu_context_or_skip() else {
        return;
    };
    let gpu_kinds = [
        BackendKind::AutoGpu {
            context: context.clone(),
        },
        BackendKind::StatevectorGpu {
            context: context.clone(),
        },
        BackendKind::DensityMatrixGpu {
            context: context.clone(),
        },
        BackendKind::StabilizerGpu {
            context: context.clone(),
        },
    ];

    for (cname, circuit) in &matrix_circuits() {
        for kind in &gpu_kinds {
            run_query_matrix_for_kind(cname, circuit, kind, None, |_name, _k, _query| false);
        }
    }

    let n = prism_q::gpu::min_qubits().max(3);
    let mut device_magic = Circuit::new(n, n);
    device_magic.add_gate(Gate::H, &[0]);
    device_magic.add_gate(Gate::Cx, &[0, 1]);
    device_magic.add_gate(Gate::Ry(0.4), &[1]);
    for q in 2..n {
        device_magic.add_gate(Gate::Cx, &[q - 1, q]);
    }
    let mut device_measured = device_magic.clone();
    device_measured.measure_all();

    for (cname, circuit) in [
        ("device_magic", &device_magic),
        ("device_measured", &device_measured),
    ] {
        for kind in [
            BackendKind::AutoGpu {
                context: context.clone(),
            },
            BackendKind::StatevectorGpu {
                context: context.clone(),
            },
            BackendKind::StabilizerGpu {
                context: context.clone(),
            },
        ] {
            run_query_matrix_for_kind(cname, circuit, &kind, None, |_name, _k, _query| false);
        }
    }

    // Device probability cap check matches execution when plan.placement is Device.
    let prob_cap = Limits::new().max_probability_qubits(2);
    for kind in [
        BackendKind::StatevectorGpu {
            context: context.clone(),
        },
        BackendKind::AutoGpu {
            context: context.clone(),
        },
    ] {
        let plan = simulate(&device_magic)
            .backend(kind.clone())
            .limits(prob_cap)
            .plan(Query::Run)
            .unwrap();
        assert_eq!(plan.placement, Placement::Device);
        assert!(matches!(
            plan.would_fail,
            Some(PrismError::ResourceLimit(_))
        ));
        let exec = simulate(&device_magic)
            .backend(kind)
            .limits(prob_cap)
            .seed(42)
            .run();
        assert!(matches!(exec, Err(PrismError::ResourceLimit(_))));
    }

    // Device RDM and Entropy readback export caps match execution in plan().
    let rdm_cap = Limits::new().max_export_qubits(2);
    let sv_gpu = BackendKind::StatevectorGpu {
        context: context.clone(),
    };
    assert_contract_for_query(
        &device_magic,
        &sv_gpu,
        Some(rdm_cap),
        Query::ReducedDensityMatrix(&[0]),
        "sv_gpu rdm cap",
    );

    let entropy_cap = Limits::new().max_export_qubits(n);
    assert_contract_for_query(
        &device_magic,
        &sv_gpu,
        Some(entropy_cap),
        Query::Entropy(&[0]),
        "sv_gpu entropy cap",
    );

    let small_3q = matrix_circuits().into_iter().next().unwrap().1;
    let dm_gpu = BackendKind::DensityMatrixGpu { context };
    let dm_cap = Limits::new().max_export_qubits(4);
    assert_contract_for_query(
        &small_3q,
        &dm_gpu,
        Some(dm_cap),
        Query::ReducedDensityMatrix(&[0]),
        "dm_gpu rdm cap",
    );

    // DensityMatrixGpu predicts the qubit-measured-twice mixture settle rejection.
    let mut twice_meas = Circuit::new(1, 2);
    twice_meas.add_gate(Gate::H, &[0]);
    twice_meas.add_measure(0, 0);
    twice_meas.add_measure(0, 1);
    let twice_noise = prism_q::NoiseBuilder::new()
        .before_measurements(prism_q::NoiseChannel::Depolarizing { p: 0.01 })
        .build(&twice_meas)
        .unwrap();
    let dm_plan = simulate(&twice_meas)
        .backend(dm_gpu.clone())
        .noise(&twice_noise)
        .plan(Query::Run)
        .unwrap();
    assert!(matches!(
        dm_plan.would_fail,
        Some(PrismError::IncompatibleBackend { .. })
    ));
    let dm_exec = simulate(&twice_meas)
        .backend(dm_gpu)
        .noise(&twice_noise)
        .seed(42)
        .run();
    assert!(matches!(
        dm_exec,
        Err(PrismError::IncompatibleBackend { .. })
    ));
}

#[cfg(feature = "gpu")]
#[test]
fn device_plan_stabilizer_gpu_1000q_ghz_matches_execution() {
    let Some(context) = acquire_gpu_context_or_skip() else {
        return;
    };
    let n = prism_q::gpu::stabilizer_min_qubits();
    let mut ghz_unitary = Circuit::new(n, n);
    ghz_unitary.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        ghz_unitary.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mut ghz_measured = ghz_unitary.clone();
    ghz_measured.measure_all();

    let kind = BackendKind::StabilizerGpu { context };
    for (cname, circuit) in [
        ("ghz_unitary", &ghz_unitary),
        ("ghz_measured", &ghz_measured),
    ] {
        let run_plan = simulate(circuit)
            .backend(kind.clone())
            .plan(Query::Run)
            .unwrap();
        assert_eq!(run_plan.placement, Placement::Device, "{cname}");
        assert_eq!(run_plan.rule, RouteRule::ExplicitBackend, "{cname}");
        let run_outcome = simulate(circuit)
            .backend(kind.clone())
            .seed(42)
            .run()
            .unwrap();
        assert_eq!(run_outcome.metadata.placement, Placement::Device, "{cname}");
        assert_eq!(
            run_outcome.metadata.reason,
            RouteRule::ExplicitBackend,
            "{cname}"
        );
        run_query_matrix_for_kind(cname, circuit, &kind, None, |_name, _k, _query| false);
    }

    let below_min = prism_q::gpu::stabilizer_min_qubits() - 1;
    let mut ghz_below = Circuit::new(below_min, below_min);
    ghz_below.add_gate(Gate::H, &[0]);
    for q in 0..below_min - 1 {
        ghz_below.add_gate(Gate::Cx, &[q, q + 1]);
    }
    ghz_below.measure_all();
    let below_plan = simulate(&ghz_below)
        .backend(kind.clone())
        .plan(Query::Run)
        .unwrap();
    assert_eq!(below_plan.placement, Placement::Host);
    assert_eq!(below_plan.rule, RouteRule::ExplicitBackend);
    let below_outcome = simulate(&ghz_below)
        .backend(kind.clone())
        .seed(42)
        .run()
        .unwrap();
    assert_eq!(below_outcome.metadata.placement, Placement::Host);
    assert_eq!(below_outcome.metadata.reason, RouteRule::ExplicitBackend);

    let shots_plan = simulate(&ghz_measured)
        .backend(kind.clone())
        .plan(Query::Shots(8))
        .unwrap();
    assert_eq!(shots_plan.placement, Placement::Host);
    let shots_actual = simulate(&ghz_measured)
        .backend(kind)
        .seed(42)
        .shots(8)
        .unwrap();
    assert_eq!(shots_actual.metadata.placement, Placement::Host);
    assert_eq!(shots_actual.metadata.backend, shots_plan.backend);
}

#[test]
fn plan_stabilizer_rank_run_and_cap_match_execution() {
    let mut c26 = Circuit::new(26, 0);
    c26.add_gate(Gate::H, &[0]);
    c26.add_gate(Gate::T, &[0]);
    for q in 1..26 {
        c26.add_gate(Gate::Cx, &[q - 1, q]);
    }

    let mut c_rot = Circuit::new(4, 0);
    c_rot.add_gate(Gate::H, &[0]);
    c_rot.add_gate(Gate::T, &[0]);
    c_rot.add_gate(Gate::Ry(0.4), &[1]);

    let mut c_meas = Circuit::new(4, 1);
    c_meas.add_gate(Gate::H, &[0]);
    c_meas.add_gate(Gate::T, &[0]);
    c_meas.add_measure(0, 0);

    for (name, circuit) in [
        ("26q", &c26),
        ("t_plus_ry", &c_rot),
        ("t_plus_meas", &c_meas),
    ] {
        for query in [Query::Run, Query::Marginals] {
            assert_contract_for_query(
                circuit,
                &BackendKind::StabilizerRank,
                None,
                query,
                &format!("stab_rank {name} {query:?}"),
            );
        }
    }

    assert_contract_for_query(
        &c_rot,
        &BackendKind::StabilizerRank,
        None,
        Query::Shots(4),
        "stab_rank t_plus_ry shots",
    );

    let meas_shots_plan = simulate(&c_meas)
        .backend(BackendKind::StabilizerRank)
        .plan(Query::Shots(4))
        .unwrap();
    assert_eq!(meas_shots_plan.rule, RouteRule::CliffordRegister);
    let meas_shots = simulate(&c_meas)
        .backend(BackendKind::StabilizerRank)
        .seed(42)
        .shots(4)
        .unwrap();
    assert_eq!(meas_shots.metadata.reason, RouteRule::CliffordRegister);

    // StabilizerRank enforces max_probability_qubits in both plan and execution.
    let mut c14 = Circuit::new(14, 0);
    c14.add_gate(Gate::H, &[0]);
    c14.add_gate(Gate::T, &[0]);
    for q in 1..14 {
        c14.add_gate(Gate::Cx, &[q - 1, q]);
    }
    let prob_cap = Limits::new().max_probability_qubits(2);
    assert_contract_for_query(
        &c14,
        &BackendKind::StabilizerRank,
        Some(prob_cap),
        Query::Run,
        "stab_rank prob cap",
    );
}

#[test]
fn plan_noisy_mixture_settle_matches_execution() {
    let mut circuit = Circuit::new(1, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_measure(0, 1);
    let noise = prism_q::NoiseBuilder::new()
        .before_measurements(prism_q::NoiseChannel::Depolarizing { p: 0.01 })
        .build(&circuit)
        .unwrap();

    for query in [
        Query::Run,
        Query::Marginals,
        Query::Shots(10),
        Query::Counts(10),
    ] {
        let req = || {
            simulate(&circuit)
                .backend(BackendKind::DensityMatrix)
                .noise(&noise)
                .seed(42)
        };
        let plan = req().plan(query).unwrap();
        assert!(
            matches!(
                plan.would_fail,
                Some(PrismError::IncompatibleBackend { .. })
            ),
            "{query:?}"
        );
        let exec = match query {
            Query::Run => req().run().map(|_| ()),
            Query::Marginals => req().marginals().map(|_| ()),
            Query::Shots(n) => req().shots(n).map(|_| ()),
            Query::Counts(n) => req().sample_counts(n).map(|_| ()),
            _ => unreachable!(),
        };
        assert!(
            matches!(exec, Err(PrismError::IncompatibleBackend { .. })),
            "{query:?} expected IncompatibleBackend from settle_measured_qubits"
        );
    }
}

#[test]
fn plan_noisy_shift_gradient_readout_matches_execution() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::Rx(0.3), &[0]);
    let mut noise = prism_q::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    noise.with_readout_error(0.1, 0.1);
    let parameters = Parameters::all_rotations(&circuit);
    let hamiltonian = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();

    for kind in [
        BackendKind::DensityMatrix,
        BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        },
    ] {
        let req = || {
            simulate(&circuit)
                .backend(kind.clone())
                .noise(&noise)
                .seed(42)
        };
        assert!(matches!(
            req().plan(Query::GradientShift {
                hamiltonian: &hamiltonian,
                parameters: &parameters,
            }),
            Err(PrismError::InvalidParameter { .. })
        ));
        assert!(matches!(
            req()
                .require_exact()
                .expectation_gradient_shift(&hamiltonian, &parameters),
            Err(PrismError::InvalidParameter { .. })
        ));
        assert!(matches!(
            req().expectation_gradient_shift(&hamiltonian, &parameters),
            Err(PrismError::InvalidParameter { .. })
        ));
    }
}

#[test]
fn plan_run_batch_shared_rule_matches_solo() {
    let items = matrix_circuits();
    let product = &items[0].1;
    let bell = &items[1].1;
    let magic = &items[2].1;

    for (name, circuit, kind, expected_rule) in [
        (
            "product",
            product,
            BackendKind::Auto,
            RouteRule::NoEntanglingGates,
        ),
        ("bell", bell, BackendKind::Auto, RouteRule::CliffordOnly),
        (
            "magic",
            magic,
            BackendKind::Statevector,
            RouteRule::ExplicitBackend,
        ),
    ] {
        let plan = simulate(circuit)
            .backend(kind.clone())
            .plan(Query::Run)
            .unwrap();
        let solo = simulate(circuit)
            .backend(kind.clone())
            .seed(42)
            .run()
            .unwrap();
        assert_eq!(plan.rule, expected_rule, "{name}");
        assert_eq!(solo.metadata.reason, expected_rule, "{name}");

        let batch = prism_q::sim::run_batch(&[circuit.clone(), circuit.clone()], kind, 42).unwrap();
        assert_eq!(batch[0].metadata.reason, solo.metadata.reason, "{name}");
        assert_eq!(batch[1].metadata.reason, solo.metadata.reason, "{name}");
    }
}

#[test]
fn plan_additional_matrix_cases_match_execution() {
    let bell = &matrix_circuits()[1].1;
    let ent_cap = Limits::new().max_export_qubits(bell.num_qubits);
    assert_contract_for_query(
        bell,
        &BackendKind::Statevector,
        Some(ent_cap),
        Query::Entropy(&[0]),
        "host sv entropy cap",
    );

    let mut reset_circuit = Circuit::new(2, 0);
    reset_circuit.add_gate(Gate::H, &[0]);
    reset_circuit.add_gate(Gate::Cx, &[0, 1]);
    reset_circuit.add_gate(Gate::Ry(0.3), &[0]);
    reset_circuit.add_reset(1);
    let reset_noise = prism_q::NoiseModel::uniform_depolarizing(&reset_circuit, 0.01);
    let reset_params = Parameters::all_rotations(&reset_circuit);
    let hamiltonian = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();
    let plan = simulate(&reset_circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&reset_noise)
        .plan(Query::GradientShift {
            hamiltonian: &hamiltonian,
            parameters: &reset_params,
        })
        .unwrap();
    assert!(matches!(
        plan.would_fail,
        Some(PrismError::IncompatibleBackend { .. })
    ));
    let exec = simulate(&reset_circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&reset_noise)
        .seed(42)
        .expectation_gradient_shift(&hamiltonian, &reset_params);
    assert!(matches!(exec, Err(PrismError::IncompatibleBackend { .. })));

    let mut qft_circuit = Circuit::new(2, 0);
    qft_circuit.add_gate(
        Gate::QftBlock {
            start: 0,
            num: 2,
            inverse: false,
            swaps: true,
            big_endian: false,
        },
        &[0, 1],
    );

    for kind in [
        BackendKind::StochasticPauli { num_samples: 16 },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold {
                epsilon: 0.0,
                max_terms: 0,
            },
        },
    ] {
        assert_contract_for_query(
            &qft_circuit,
            &kind,
            None,
            Query::Marginals,
            &format!("qft marginals {kind:?}"),
        );
    }

    let pauli_path = BackendKind::PauliPath {
        epsilon: 0.0,
        max_terms: 0,
    };
    let obs = vec![vec![PauliTerm::z(0)]];
    assert_contract_for_query(
        &qft_circuit,
        &pauli_path,
        None,
        Query::Expectation(&obs),
        "pauli_path qft expectation",
    );

    let mut leak_circuit = Circuit::new(1, 0);
    leak_circuit.add_gate(Gate::H, &[0]);
    let leak_noise = prism_q::NoiseBuilder::new()
        .after_gates(
            prism_q::GateFilter::all(),
            prism_q::NoiseChannel::Leakage { p: 0.05 },
        )
        .build(&leak_circuit)
        .unwrap();
    let leak_plan = simulate(&leak_circuit)
        .backend(pauli_path.clone())
        .noise(&leak_noise)
        .plan(Query::Observable(&hamiltonian))
        .unwrap();
    assert!(matches!(
        leak_plan.would_fail,
        Some(PrismError::BackendUnsupported { .. })
    ));
    assert!(matches!(
        simulate(&leak_circuit)
            .backend(pauli_path.clone())
            .noise(&leak_noise)
            .seed(42)
            .observable_expectation(&hamiltonian),
        Err(PrismError::BackendUnsupported { .. })
    ));

    let mut mid_meas = Circuit::new(1, 2);
    mid_meas.add_gate(Gate::H, &[0]);
    mid_meas.add_measure(0, 0);
    mid_meas.add_gate(Gate::H, &[0]);
    mid_meas.add_measure(0, 1);
    for query in [Query::Shots(4), Query::Counts(4)] {
        assert_contract_for_query(
            &mid_meas,
            &pauli_path,
            None,
            query,
            &format!("pauli_path mid_meas {query:?}"),
        );
    }
}

#[test]
fn plan_auto_marginals_spd_budget_fallback_matches_execution() {
    let n = 16;
    let mut circuit = Circuit::new(n, 0);
    for layer in 0..8 {
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
        }
        let start = layer % 2;
        for q in (start..n - 1).step_by(2) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    let limits = Limits::new().max_statevector_qubits(20);
    assert_contract_for_query(
        &circuit,
        &BackendKind::Auto,
        Some(limits),
        Query::Marginals,
        "auto_marginals_spd_budget_fallback",
    );

    let plan = simulate(&circuit)
        .backend(BackendKind::Auto)
        .limits(limits)
        .plan(Query::Marginals)
        .unwrap();
    assert_eq!(plan.backend, ResolvedBackend::DeterministicPauli);
    assert_eq!(plan.rule, RouteRule::SpdExact);
    assert!(plan.fallbacks.iter().any(|fb| {
        fb.rule == RouteRule::SpdBudgetFallback && fb.backend == ResolvedBackend::Statevector
    }));

    let exec = simulate(&circuit)
        .backend(BackendKind::Auto)
        .limits(limits)
        .seed(42)
        .marginals()
        .unwrap();
    assert_eq!(exec.metadata.backend, ResolvedBackend::Statevector);
    assert_eq!(exec.metadata.reason, RouteRule::SpdBudgetFallback);
}

#[test]
fn plan_boundary_cases_match_execution() {
    let obs = vec![vec![PauliTerm::z(0)]];
    let hamiltonian = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(0)])]).unwrap();

    let zero_budget = BackendKind::DeterministicPauli {
        truncation: SpdTruncation::Budget { max_terms: 0 },
    };
    for (cname, circuit) in matrix_circuits()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "bell" | "magic" | "clifford_t"))
    {
        let params = Parameters::all_rotations(&circuit);
        for query in [
            Query::Marginals,
            Query::Expectation(&obs),
            Query::Observable(&hamiltonian),
            Query::GradientShift {
                hamiltonian: &hamiltonian,
                parameters: &params,
            },
        ] {
            assert_contract_for_query(
                &circuit,
                &zero_budget,
                None,
                query,
                &format!("zero_budget {cname} {query:?}"),
            );
        }
    }

    let mps_zero = BackendKind::Mps { max_bond_dim: 0 };
    for (cname, circuit) in matrix_circuits()
        .into_iter()
        .filter(|(name, _)| matches!(*name, "bell" | "magic" | "clifford_t"))
    {
        for query in [Query::Shots(0), Query::Counts(0)] {
            assert_contract_for_query(
                &circuit,
                &mps_zero,
                None,
                query,
                &format!("mps_zero {cname} {query:?}"),
            );
        }
    }

    let mut mps_save_dm = Circuit::new(2, 0);
    mps_save_dm.add_gate(Gate::H, &[0]);
    mps_save_dm.add_gate(Gate::Cx, &[0, 1]);
    mps_save_dm.add_save(SaveSpec::DensityMatrix, "rho");
    assert_contract_for_query(
        &mps_save_dm,
        &mps_zero,
        None,
        Query::Run,
        "mps_zero save dm",
    );

    let empty_0q = Circuit::new(0, 0);
    for kind in [
        BackendKind::TensorNetwork,
        BackendKind::TensorNetworkBounded { tolerance: 1e-3 },
    ] {
        for query in [Query::Run, Query::StateVector] {
            assert_contract_for_query(
                &empty_0q,
                &kind,
                None,
                query,
                &format!("empty_0q {kind:?} {query:?}"),
            );
        }
    }

    let mut tn_save_probs = Circuit::new(2, 0);
    tn_save_probs.add_gate(Gate::H, &[0]);
    tn_save_probs.add_save(SaveSpec::Probabilities, "probs");
    let tn_zero_tol = BackendKind::TensorNetworkBounded { tolerance: 0.0 };
    assert_contract_for_query(
        &tn_save_probs,
        &tn_zero_tol,
        None,
        Query::Marginals,
        "tn_zero_tol save probs marginals",
    );

    // PerShot on decomposed blocks runs classical_only() and ignores max_probability_qubits.
    let mut decomp_mid = Circuit::new(8, 2);
    for blk in 0..2 {
        let base = blk * 4;
        for q in 0..4 {
            decomp_mid.add_gate(Gate::Rx(0.4), &[base + q]);
        }
        decomp_mid.add_gate(Gate::Cx, &[base, base + 1]);
        decomp_mid.add_gate(Gate::Cx, &[base + 1, base + 2]);
        decomp_mid.add_gate(Gate::Cx, &[base + 2, base + 3]);
    }
    decomp_mid.add_measure(0, 0);
    decomp_mid.add_gate(Gate::H, &[0]);
    decomp_mid.add_measure(4, 1);
    let tight_prob = Limits::new().max_probability_qubits(2);
    let plan = simulate(&decomp_mid)
        .limits(tight_prob)
        .plan(Query::Shots(4))
        .unwrap();
    assert!(
        plan.would_fail.is_none(),
        "PerShot ignores prob cap: {plan:?}"
    );
    assert!(
        simulate(&decomp_mid)
            .limits(tight_prob)
            .seed(42)
            .shots(4)
            .is_ok()
    );

    // PauliPath accepts inert ReadoutError, matching plan().
    let mut pp_c = Circuit::new(2, 1);
    pp_c.add_gate(Gate::H, &[0]);
    pp_c.add_gate(Gate::Cx, &[0, 1]);
    let mut inert_ro = prism_q::NoiseModel::uniform_depolarizing(&pp_c, 0.05);
    inert_ro.with_readout_error(0.0, 0.0);
    let pp_kind = BackendKind::PauliPath {
        epsilon: 1e-4,
        max_terms: 1024,
    };
    let obs = [vec![PauliTerm::z(0)]];
    let pp_plan = simulate(&pp_c)
        .backend(pp_kind.clone())
        .noise(&inert_ro)
        .plan(Query::Expectation(&obs))
        .unwrap();
    assert!(pp_plan.would_fail.is_none(), "{pp_plan:?}");
    assert!(
        simulate(&pp_c)
            .backend(pp_kind)
            .noise(&inert_ro)
            .seed(42)
            .expectation_values(&obs)
            .is_ok()
    );

    // initial_state + Query::Entropy honors max_export_qubits cap.
    let mut bell = Circuit::new(3, 0);
    bell.add_gate(Gate::H, &[0]);
    bell.add_gate(Gate::Cx, &[0, 1]);
    let mut amps = vec![num_complex::Complex64::new(0.0, 0.0); 8];
    amps[0] = num_complex::Complex64::new(1.0, 0.0);
    let tight_export = Limits::new().max_export_qubits(2);
    let ent_plan = simulate(&bell)
        .initial_state(&amps)
        .limits(tight_export)
        .plan(Query::Entropy(&[0]))
        .unwrap();
    assert!(
        matches!(ent_plan.would_fail, Some(PrismError::ResourceLimit { .. })),
        "initial_state entropy should report export cap: {ent_plan:?}"
    );
    assert!(matches!(
        simulate(&bell)
            .initial_state(&amps)
            .limits(tight_export)
            .seed(42)
            .entanglement_entropy(&[0]),
        Err(PrismError::ResourceLimit { .. })
    ));
}
