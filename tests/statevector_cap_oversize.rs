//! Queries that cross the statevector cap: expectation values, result
//! metadata, and parameter-shift gradients. One binary because each pins
//! `PRISM_MAX_SV_QUBITS` to 4, and the cap is cached per process.

mod common;

use common::{SEED, caps};
use prism_q::gates::Gate;
use prism_q::{
    BackendKind, Circuit, Parameters, PauliTerm, PrismError, ResolvedBackend,
    run_expectation_gradient, run_expectation_values, simulate,
};

const CAP: usize = 4;
const TOL: f64 = 1e-9;

fn small_sv_cap() {
    caps::set_once(&[("PRISM_MAX_SV_QUBITS", "4")]);
}

// Above the cap the route lands on a backend that evaluates the observable on
// its own representation, so the reference is closed form; a statevector run is
// unavailable under the same cap.
#[test]
fn auto_non_clifford_oversize_evaluates_on_the_selected_backend() {
    small_sv_cap();

    let theta = 0.3_f64;
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Rx(theta), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);

    // Rx(theta) on every qubit, then CX(0->1). Z0 commutes through the control,
    // so <Z0> = cos(theta). Z5 is untouched by the CX, so <Z5> = cos(theta) too.
    // Z0*Z1 is the CX-conjugated Z0*Z0*Z1 = Z1, giving <Z0 Z1> = cos(theta).
    let vals = run_expectation_values(
        &c,
        &[
            vec![PauliTerm::z(0)],
            vec![PauliTerm::z(5)],
            vec![PauliTerm::z(0), PauliTerm::z(1)],
            vec![],
        ],
        SEED,
    )
    .unwrap();

    let want = [theta.cos(), theta.cos(), theta.cos(), 1.0];
    for (i, (&got, &expected)) in vals.iter().zip(&want).enumerate() {
        assert!(
            (got - expected).abs() < TOL,
            "observable {i}: got {got}, want {expected}"
        );
    }
}

/// Not sparse-friendly, so Auto takes the MPS branch of the oversize split
/// rather than the sparse one.
fn oversize_dense(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..num_qubits - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..num_qubits {
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit
}

// The dispatch-level half of the approximation contract: the same query above
// and below the statevector cap must differ in the flag, asserted at the
// `Simulate` level rather than on the backend.
#[test]
fn auto_reports_approximate_above_the_statevector_cap() {
    small_sv_cap();
    let below = simulate(&oversize_dense(CAP - 2))
        .seed(SEED)
        .marginals()
        .unwrap();
    let above = simulate(&oversize_dense(CAP + 2))
        .seed(SEED)
        .marginals()
        .unwrap();

    assert!(below.metadata.is_exact(), "{:?}", below.metadata);
    assert!(!above.metadata.is_exact(), "{:?}", above.metadata);
    assert_eq!(above.metadata.backend, ResolvedBackend::Mps);
}

#[test]
fn require_exact_rejects_the_approximate_route() {
    small_sv_cap();
    let circuit = oversize_dense(CAP + 2);
    let err = simulate(&circuit)
        .seed(SEED)
        .require_exact()
        .marginals()
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "Mps"),
        "{err:?}"
    );

    // The same builder without the requirement answers.
    assert!(simulate(&circuit).seed(SEED).marginals().is_ok());
}

struct Fixture {
    circuit: Circuit,
    params: Parameters,
    hamiltonian: Vec<(f64, Vec<PauliTerm>)>,
    want: Vec<f64>,
}

// Rx(theta_q) on every qubit, then CX(0->1) and CX(2->3). Under the Heisenberg
// conjugation of CX(0->1), Z0 is unchanged and Z0*Z1 becomes Z1, so
// <H> = cos(theta_0) + 0.5 cos(theta_1) for H = Z0 + 0.5 Z0 Z1. Slots 0 and 1
// therefore carry -sin(theta_0) and -0.5 sin(theta_1); every other slot is
// exactly zero.
fn fixture(n: usize) -> Fixture {
    let mut circuit = Circuit::new(n, 0);
    let angles: Vec<f64> = (0..n).map(|q| 0.2 + 0.1 * q as f64).collect();
    for (q, &theta) in angles.iter().enumerate() {
        circuit.add_gate(Gate::Rx(theta), &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[2, 3]);

    let params = Parameters::all_rotations(&circuit);
    let hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0)]),
        (0.5, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    ];
    let mut want = vec![0.0; n];
    want[0] = -angles[0].sin();
    want[1] = -0.5 * angles[1].sin();
    Fixture {
        circuit,
        params,
        hamiltonian,
        want,
    }
}

fn assert_gradient(label: &str, gradient: &[f64], want: &[f64]) {
    assert_eq!(gradient.len(), want.len());
    for (slot, (&got, &expected)) in gradient.iter().zip(want).enumerate() {
        assert!(
            (got - expected).abs() < TOL,
            "{label} slot {slot}: got {got}, want {expected}"
        );
    }
}

// The adjoint cannot run at this width, so the reference is closed form rather
// than a statevector comparison.
#[test]
fn shift_gradient_above_the_cap_on_mps_and_factored() {
    small_sv_cap();

    let f = fixture(8);

    // The adjoint declines at this width: it holds two statevectors.
    assert!(run_expectation_gradient(&f.circuit, &f.hamiltonian, &f.params, SEED).is_err());

    for kind in [BackendKind::Mps { max_bond_dim: 16 }, BackendKind::Factored] {
        let label = format!("{kind:?}");
        let g = simulate(&f.circuit)
            .backend(kind)
            .seed(SEED)
            .expectation_gradient_shift(&f.hamiltonian, &f.params)
            .unwrap();
        assert_gradient(&label, &g.gradient, &f.want);
    }

    // Auto routes past the cap to a native observable path of its own.
    let g = simulate(&f.circuit)
        .seed(SEED)
        .expectation_gradient_shift(&f.hamiltonian, &f.params)
        .unwrap();
    assert_gradient("auto", &g.gradient, &f.want);
}
