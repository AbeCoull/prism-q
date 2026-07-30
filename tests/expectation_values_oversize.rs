//! Auto non-Clifford expectation values above the statevector cap. Isolated
//! in its own test binary: it overrides `PRISM_MAX_SV_QUBITS`, which is
//! cached per process, so it must not share a process with tests that expect
//! the real cap. Above the cap the route lands on a backend that evaluates
//! the observable on its own representation, so the reference is closed form;
//! a statevector run is unavailable under the same cap.

use prism_q::gates::Gate;
use prism_q::{Circuit, PauliTerm, run_expectation_values};

const TOL: f64 = 1e-9;

#[test]
fn auto_non_clifford_oversize_evaluates_on_the_selected_backend() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_SV_QUBITS", "4") };

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
        42,
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
