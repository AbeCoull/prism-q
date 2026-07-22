//! Public API coverage for `run_expectation_values` and
//! `Simulate::expectation_values`.

mod common;

use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, PauliAxis, PauliTerm, run_expectation_values, simulate};

const TOL: f64 = 1e-10;

fn bell() -> Circuit {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    common::assert_probs_close(got, want, tol, "expectation");
}

#[test]
fn clifford_expectations_are_exact() {
    // Bell state: <ZZ>=<XX>=1, <YY>=-1, <Z0>=0, and the empty (identity) = 1.
    let vals = run_expectation_values(
        &bell(),
        &[
            vec![PauliTerm::z(0), PauliTerm::z(1)],
            vec![PauliTerm::x(0), PauliTerm::x(1)],
            vec![PauliTerm::y(0), PauliTerm::y(1)],
            vec![PauliTerm::z(0)],
            vec![],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[1.0, 1.0, -1.0, 0.0, 1.0], TOL);
}

#[test]
fn plus_state_single_qubit() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let vals = run_expectation_values(
        &c,
        &[
            vec![PauliTerm::x(0)],
            vec![PauliTerm::y(0)],
            vec![PauliTerm::z(0)],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[1.0, 0.0, 0.0], TOL);
}

#[test]
fn non_clifford_statevector_route_matches_analytic() {
    let theta = 0.7_f64;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    // Rx(theta)|0>: <X>=0, <Y>=-sin(theta), <Z>=cos(theta).
    let vals = run_expectation_values(
        &c,
        &[
            vec![PauliTerm::x(0)],
            vec![PauliTerm::y(0)],
            vec![PauliTerm::z(0)],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[0.0, -theta.sin(), theta.cos()], TOL);
}

#[test]
fn clifford_route_matches_statevector_route() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);
    let observables = [
        vec![PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)],
        vec![PauliTerm::z(0), PauliTerm::z(2)],
        vec![PauliTerm::new(1, PauliAxis::X)],
    ];
    let auto = run_expectation_values(&c, &observables, 42).unwrap();
    let sv = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    assert_close(&auto, &sv, TOL);
}

#[test]
fn clifford_t_deterministic_pauli_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[1]);
    let observables = [
        vec![PauliTerm::x(0), PauliTerm::x(1)],
        vec![PauliTerm::z(0)],
    ];
    let sv = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    let spd = simulate(&c)
        .backend(BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        })
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    assert_close(&spd, &sv, 1e-9);
}

#[test]
fn stochastic_pauli_seeds_each_observable_independently() {
    // H,T,H gives a stochastic estimate; two identical observables must draw
    // from distinct sample streams rather than sharing one seed.
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    let z0 = vec![PauliTerm::z(0)];
    let vals = simulate(&c)
        .backend(BackendKind::StochasticPauli { num_samples: 64 })
        .seed(42)
        .expectation_values(&[z0.clone(), z0])
        .unwrap();
    assert!(vals[0] != vals[1], "identical observables shared a seed");
}

#[test]
fn invalid_observable_is_rejected() {
    use prism_q::PrismError::{InvalidParameter, InvalidQubit};
    let c = bell();
    assert!(matches!(
        run_expectation_values(&c, &[vec![PauliTerm::z(5)]], 42).unwrap_err(),
        InvalidQubit { .. }
    ));
    assert!(matches!(
        run_expectation_values(&c, &[vec![PauliTerm::z(0), PauliTerm::x(0)]], 42).unwrap_err(),
        InvalidParameter { .. }
    ));
}

#[test]
fn non_unitary_circuit_is_rejected() {
    // Rejected for every backend kind, with empty and non-empty observables.
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];\nc[0] = measure q[0];";
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    let observables: [&[Vec<PauliTerm>]; 2] = [&[], &[vec![PauliTerm::z(0)]]];
    for backend in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        },
        BackendKind::StochasticPauli { num_samples: 16 },
    ] {
        for obs in observables {
            let err = simulate(&c)
                .backend(backend.clone())
                .seed(42)
                .expectation_values(obs)
                .unwrap_err();
            assert!(
                matches!(err, prism_q::PrismError::IncompatibleBackend { .. }),
                "{backend:?} accepted a non-unitary circuit"
            );
        }
    }
}
