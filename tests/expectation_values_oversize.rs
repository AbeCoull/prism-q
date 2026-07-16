//! Oversize guard for the Auto non-Clifford expectation-value route.
//!
//! Isolated in its own test binary: it overrides `PRISM_MAX_SV_QUBITS`, which
//! `max_statevector_qubits()` caches per process, so it must not share a process
//! with tests that expect the real memory-derived cap.

use prism_q::gates::Gate;
use prism_q::{Circuit, PauliTerm, run_expectation_values};

#[test]
fn auto_non_clifford_oversize_returns_clean_error() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_SV_QUBITS", "4") };

    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Rx(0.3), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);

    let err = run_expectation_values(&c, &[vec![PauliTerm::z(0)]], 42).unwrap_err();
    assert!(
        matches!(err, prism_q::PrismError::IncompatibleBackend { .. }),
        "expected a clean statevector-cap error, got {err:?}"
    );
}
