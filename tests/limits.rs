//! Request and process-default limits, end to end. The process default is global, so
//! every test that sets it lives in one test function, and the others hold only for
//! limits a request sets itself.

use prism_q::{
    BackendKind, Circuit, CircuitBuilder, LimitKnob, LimitSource, Limits, PrismError,
    ResolvedBackend, ResourceLimit, circuits, default_limits, set_default_limits, simulate,
};

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

    set_default_limits(Limits::new());
    assert!(
        run().is_ok(),
        "clearing the default restores the detected caps"
    );
}
