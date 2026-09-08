//! Tensor-network resource caps: the dense probability ceiling and the planned
//! peak-intermediate cap. Isolated in its own test binary: it overrides
//! `PRISM_MAX_PROB_QUBITS` and `PRISM_MAX_TN_PEAK_QUBITS`, which the cap
//! helpers cache per process. Every rejection is decided before the backend
//! contracts anything, so no test allocates an oversize tensor.

use std::sync::Once;

use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, PauliTerm, PrismError, simulate};

const SEED: u64 = 42;
const PROB_CAP: usize = 4;
/// Peak cap of `2^8` elements: a 4-qubit dense readout peaks at 16 elements
/// and stays under it, while a single 6-qubit MCU tensor already holds 4096.
const PEAK_CAP: usize = 8;

fn small_caps() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries a cap while it is written.
        unsafe {
            std::env::set_var("PRISM_MAX_PROB_QUBITS", "4");
            std::env::set_var("PRISM_MAX_TN_PEAK_QUBITS", "8");
        }
    });
}

fn incompatible_reason(err: PrismError) -> String {
    match err {
        PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, "tensornetwork");
            reason
        }
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

// The dispatch layer reads `BackendUnsupported` from a probability query as
// "no dense terminal" and reports `None`; the ceiling must not travel that way.
#[test]
fn explicit_probabilities_one_qubit_above_the_ceiling_name_the_cap() {
    small_caps();
    let n = PROB_CAP + 1;
    let err = simulate(&circuits::ghz_circuit(n))
        .backend(BackendKind::TensorNetwork)
        .seed(SEED)
        .run()
        .unwrap_err();
    let reason = incompatible_reason(err);
    assert!(
        reason.contains(&format!("{n} qubits")) && reason.contains(&format!("cap of {PROB_CAP}")),
        "{reason}"
    );

    let at_cap = simulate(&circuits::ghz_circuit(PROB_CAP))
        .backend(BackendKind::TensorNetwork)
        .seed(SEED)
        .run()
        .unwrap();
    assert!(at_cap.probabilities.is_some());
}

// A 30-qubit register with one 6-qubit MCU: no path here builds a 2^30
// vector, and any contraction touching the 12-leg MCU tensor plans an
// intermediate well past the pinned peak cap.
#[test]
fn doubled_contraction_over_the_peak_cap_errors_before_allocating() {
    small_caps();
    let n = 30;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
    }
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    circuit.add_gate(Gate::mcu(x_mat, 5), &[0, 1, 2, 3, 4, 5]);

    let mut tn = TensorNetworkBackend::new(SEED);
    tn.init(n, 0).unwrap();
    tn.apply_instructions(&circuit.instructions).unwrap();

    let err = tn.pauli_expectations(&[vec![PauliTerm::z(0)]]).unwrap_err();
    let reason = incompatible_reason(err);
    assert!(
        reason.contains("pauli expectation") && reason.contains(&format!("2^{PEAK_CAP}")),
        "{reason}"
    );
    let peak: usize = reason
        .split("peak intermediate of ")
        .nth(1)
        .and_then(|rest| rest.split(' ').next())
        .and_then(|digits| digits.parse().ok())
        .expect("reason names the planned peak");
    assert!(peak > 1 << PEAK_CAP, "{reason}");

    let err = tn.reduced_density_matrix_1q(0).unwrap_err();
    assert!(
        incompatible_reason(err).contains("reduced density matrix"),
        "the marginal path shares the guard"
    );
}
