//! Request and process-default limits, end to end. The process default is global, so
//! every test that sets it lives in one test function, and the others hold only for
//! limits a request sets itself.

use prism_q::{
    BackendKind, Circuit, CircuitBuilder, Gate, LimitKnob, LimitSource, Limits, PrismError, QecOp,
    QecOptions, QecPauli, QecProgram, QecRecordRef, ResolvedBackend, ResourceLimit, circuits,
    default_limits, set_default_limits, simulate,
};

/// A noiseless `EXP_VAL` program on `n` qubits whose feed-forward the analytical ladder
/// cannot lower, so it reaches the reference runner only within its width limit.
fn feedforward_exp_val(n: usize) -> QecProgram {
    let mut program = QecProgram::with_options(
        n,
        QecOptions {
            shots: 4,
            seed: 42,
            ..QecOptions::default()
        },
    );
    program.push_gate(Gate::X, &[0]).unwrap();
    let record = QecRecordRef::absolute(program.measure_z(0).unwrap());
    let correction = vec![QecOp::Gate {
        gate: Gate::X,
        targets: vec![n - 1],
    }];
    program.feedforward(&[record], true, correction).unwrap();
    program
        .expectation_value(&[QecPauli::z(n - 1)], 1.0)
        .unwrap();
    program
}

/// An entangling chain of non-Clifford rotations, which auto routing sends to the
/// statevector while it fits.
fn rotation_chain(n: usize) -> Circuit {
    let mut builder = CircuitBuilder::new(n);
    for q in 0..n {
        builder.ry(0.3 + 0.1 * q as f64, q);
    }
    for q in 0..n - 1 {
        builder.cx(q, q + 1);
    }
    builder.build()
}

fn resource_limit(err: PrismError) -> ResourceLimit {
    match err {
        PrismError::ResourceLimit(data) => ResourceLimit::clone(&data),
        other => panic!("expected ResourceLimit, got {other:?}"),
    }
}

#[test]
fn a_request_cap_tighter_than_the_default_names_its_setter() {
    let circuit = circuits::ghz_circuit(6);
    let err = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .limits(Limits::new().max_statevector_qubits(4))
        .seed(42)
        .run()
        .unwrap_err();
    let message = err.to_string();
    let data = resource_limit(err);
    assert_eq!(data.knob, Some(LimitKnob::StatevectorQubits));
    assert_eq!(data.limit_source, LimitSource::Request);
    assert_eq!((data.required, data.limit), (6, 4));
    assert_eq!(
        data.env_var, None,
        "the environment cannot raise a request's cap"
    );
    assert!(
        message.contains("set by this request's limits"),
        "{message}"
    );
    assert!(
        message.contains("Limits::max_statevector_qubits"),
        "{message}"
    );
}

#[test]
fn a_request_density_matrix_cap_rejects_at_dispatch() {
    let circuit = circuits::ghz_circuit(3);
    let err = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .limits(Limits::new().max_density_matrix_qubits(2))
        .seed(42)
        .run()
        .unwrap_err();
    let data = resource_limit(err);
    assert_eq!(data.knob, Some(LimitKnob::DensityMatrixQubits));
    assert_eq!(data.limit_source, LimitSource::Request);
}

#[test]
fn a_request_budget_reroutes_auto_past_the_dense_cap() {
    let circuit = rotation_chain(10);
    let dense = simulate(&circuit).seed(42).run().unwrap();
    assert_eq!(dense.metadata.backend, ResolvedBackend::Statevector);
    let tenant = Limits::new().memory_budget_bytes(1 << 12);
    let routed = simulate(&circuit).limits(tenant).seed(42).run().unwrap();
    assert_ne!(
        routed.metadata.backend,
        ResolvedBackend::Statevector,
        "a 4 KiB budget holds an 8-qubit statevector at most"
    );
}

#[test]
fn a_request_export_cap_bounds_the_statevector_terminal() {
    let circuit = circuits::ghz_circuit(5);
    let err = simulate(&circuit)
        .limits(Limits::new().max_export_qubits(3))
        .seed(42)
        .state_vector()
        .unwrap_err();
    let data = resource_limit(err);
    assert_eq!(data.knob, Some(LimitKnob::ExportQubits));
    assert_eq!(data.limit_source, LimitSource::Request);
}

#[test]
fn the_process_default_applies_without_a_restart() {
    let circuit = circuits::ghz_circuit(6);
    let run = || {
        simulate(&circuit)
            .backend(BackendKind::Statevector)
            .seed(42)
            .run()
    };
    assert!(run().is_ok());

    set_default_limits(Limits::new().max_statevector_qubits(4));
    assert_eq!(default_limits().max_statevector_qubits, Some(4));
    let data = resource_limit(run().unwrap_err());
    assert_eq!(data.knob, Some(LimitKnob::StatevectorQubits));
    assert_eq!(data.limit_source, LimitSource::ProcessDefault);
    let qasm = prism_q::run_qasm(
        "OPENQASM 3.0; include \"stdgates.inc\"; qubit[6] q; ry(0.3) q[0]; ry(0.4) q[1]; \
         ry(0.5) q[2]; ry(0.6) q[3]; ry(0.7) q[4]; ry(0.8) q[5]; cx q[0], q[1]; \
         cx q[1], q[2]; cx q[2], q[3]; cx q[3], q[4]; cx q[4], q[5];",
        42,
    )
    .unwrap();
    assert_ne!(
        qasm.metadata.backend,
        ResolvedBackend::Statevector,
        "run_qasm resolves against the process default"
    );

    let widened = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .limits(Limits::new().max_statevector_qubits(8))
        .seed(42)
        .run();
    assert!(widened.is_ok(), "a request may raise the process default");

    let estimate = |n: usize| prism_q::run_qec_program(&feedforward_exp_val(n));
    set_default_limits(Limits::new().max_qec_reference_qubits(4));
    let answered = estimate(4).unwrap();
    assert_eq!(answered.expectation_values.unwrap()[0].mean, -1.0);
    let err = estimate(5).unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { reason, .. }
            if reason.contains("FEEDFORWARD")
                && reason.contains("5 qubits")
                && reason.contains("max_qec_reference_qubits (4)")),
        "{err}"
    );
    set_default_limits(Limits::new().max_qec_reference_qubits(5));
    assert!(
        estimate(5).is_ok(),
        "raising the limit reroutes the program"
    );
    set_default_limits(Limits::new().max_qec_reference_qubits(64));
    let data = resource_limit(estimate(40).unwrap_err());
    assert_eq!(data.knob, Some(LimitKnob::StatevectorQubits));
    assert!(data.operation.contains("FEEDFORWARD"), "{}", data.operation);

    set_default_limits(Limits::new());
    assert!(
        run().is_ok(),
        "clearing the default restores the detected caps"
    );

    let mut param_circuit = Circuit::new(1, 0);
    param_circuit.add_gate(Gate::Rx(0.0), &[0]);
    let params = prism_q::Parameters::from_links(vec![prism_q::ParamLink::new(0, 0)], 1);
    let hamiltonian =
        prism_q::PauliObservable::from_terms([(1.0, vec![prism_q::PauliTerm::z(0)])]).unwrap();
    let mut prepared_before =
        prism_q::PreparedCircuit::new(param_circuit.clone(), params.clone()).unwrap();
    let bound = params.bind(&param_circuit, &[0.3]).unwrap();
    let qec = feedforward_exp_val(4);
    let dm_circuit = circuits::ghz_circuit(2);
    let mut stab_circuit = Circuit::new(2, 2);
    stab_circuit.add_gate(Gate::H, &[0]);
    stab_circuit.add_gate(Gate::T, &[0]);
    stab_circuit.add_gate(Gate::Cx, &[0, 1]);
    stab_circuit.add_measure(0, 0);
    stab_circuit.add_measure(1, 1);
    let mut noisy_circuit = Circuit::new(2, 2);
    noisy_circuit.add_gate(Gate::H, &[0]);
    noisy_circuit.add_gate(Gate::Cx, &[0, 1]);
    noisy_circuit.add_measure(0, 0);
    noisy_circuit.add_measure(1, 1);
    let noise = prism_q::NoiseBuilder::new().build(&noisy_circuit).unwrap();

    for bad in [-1e-12, 1.0 + 1e-12, f64::NAN, f64::NEG_INFINITY] {
        let mut bad_limits = Limits::new();
        bad_limits.min_fidelity_estimate = Some(bad);
        set_default_limits(bad_limits);

        let grad_err =
            prism_q::sim::gradient::run_expectation_gradient(&bound, &hamiltonian, &params, 42)
                .unwrap_err();
        assert!(
            matches!(grad_err, PrismError::InvalidParameter { .. }),
            "run_expectation_gradient for {bad}: {grad_err:?}"
        );

        let prep_err = prepared_before
            .expectation_gradient(&[0.3], &hamiltonian, 42)
            .unwrap_err();
        assert!(
            matches!(prep_err, PrismError::InvalidParameter { .. }),
            "PreparedCircuit::expectation_gradient for {bad}: {prep_err:?}"
        );

        let qec_err = prism_q::run_qec_program(&qec).unwrap_err();
        assert!(
            matches!(qec_err, PrismError::InvalidParameter { .. }),
            "run_qec_program for {bad}: {qec_err:?}"
        );

        let dm_err = prism_q::density_matrix_expectation_values(
            &dm_circuit,
            &[vec![prism_q::PauliTerm::z(0)]],
            None,
            42,
        )
        .unwrap_err();
        assert!(
            matches!(dm_err, PrismError::InvalidParameter { .. }),
            "density_matrix_expectation_values for {bad}: {dm_err:?}"
        );

        let noisy_err = prism_q::run_shots_noisy(&noisy_circuit, &noise, 4, 42).unwrap_err();
        assert!(
            matches!(noisy_err, PrismError::InvalidParameter { .. }),
            "run_shots_noisy for {bad}: {noisy_err:?}"
        );

        let sr_err = prism_q::sim::stabilizer_rank::run_stabilizer_rank_shots(&stab_circuit, 4, 42)
            .unwrap_err();
        assert!(
            matches!(sr_err, PrismError::InvalidParameter { .. }),
            "run_stabilizer_rank_shots for {bad}: {sr_err:?}"
        );
    }

    set_default_limits(Limits::new());
}
