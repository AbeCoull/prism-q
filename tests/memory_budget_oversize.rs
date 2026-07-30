//! Oversize guards for the backend memory-budget contract. Isolated in its
//! own test binary: it overrides `PRISM_MAX_SV_QUBITS` and
//! `PRISM_MAX_DM_QUBITS`, which the cap helpers cache per process. Every
//! rejection is decided before the backend reserves anything, so no test
//! allocates an oversize state.

use std::sync::Once;

use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::gates::Gate;
use prism_q::{Circuit, PrismError, StatevectorBackend, run_on};

/// Caps keep the default relationship, where a density matrix of `n` qubits is
/// a `2n`-qubit statevector, so neither backend's guard depends on the other
/// being set inconsistently.
const SV_CAP: usize = 8;
const DM_CAP: usize = SV_CAP / 2;

fn small_caps() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries a cap while it is written.
        unsafe {
            std::env::set_var("PRISM_MAX_SV_QUBITS", "8");
            std::env::set_var("PRISM_MAX_DM_QUBITS", "4");
        }
    });
}

fn entangling_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 1..n {
        c.add_gate(Gate::Cx, &[q - 1, q]);
    }
    c
}

fn assert_cap_error(err: PrismError, backend: &str) {
    match err {
        PrismError::IncompatibleBackend {
            backend: named,
            reason,
        } => {
            assert_eq!(named, backend, "wrong backend named: {reason}");
            assert!(
                reason.contains("exceeding the cap"),
                "expected a cap rejection, got {reason}"
            );
        }
        other => panic!("expected a clean cap error, got {other:?}"),
    }
}

// The path a caller reaches by driving a backend directly instead of through
// dispatch, which is what the Python `state_vector()` terminal does. Before
// the cap moved into `init` this allocated unchecked and aborted the process.
#[test]
fn statevector_init_over_cap_returns_clean_error() {
    small_caps();
    let mut backend = StatevectorBackend::new(42);
    let err = run_on(&mut backend, &entangling_circuit(SV_CAP + 2)).unwrap_err();
    assert_cap_error(err, "statevector");
}

#[test]
fn density_matrix_init_over_cap_returns_clean_error() {
    small_caps();
    let mut backend = DensityMatrixBackend::new(42);
    let err = backend.init(DM_CAP + 2, 0).unwrap_err();
    assert_cap_error(err, "density_matrix");
}

#[test]
fn statevector_init_at_the_cap_still_runs() {
    small_caps();
    let mut backend = StatevectorBackend::new(42);
    run_on(&mut backend, &entangling_circuit(SV_CAP)).expect("a circuit at the cap must still run");
    let probs = backend.probabilities().unwrap();
    assert_eq!(probs.len(), 1 << SV_CAP);
    assert!(
        (probs[0] - 0.5).abs() < 1e-12 && (probs[(1 << SV_CAP) - 1] - 0.5).abs() < 1e-12,
        "GHZ at the cap should stay correct: {probs:?}"
    );
}

#[test]
fn density_matrix_init_at_the_cap_still_runs() {
    small_caps();
    let mut backend = DensityMatrixBackend::new(42);
    backend
        .init(DM_CAP, 0)
        .expect("a circuit at the cap must still run");
    assert_eq!(backend.num_qubits(), DM_CAP);
}

#[test]
fn unaddressable_qubit_count_is_rejected_before_the_shift() {
    small_caps();
    let mut backend = StatevectorBackend::new(42);
    let err = backend.init(usize::BITS as usize, 0).unwrap_err();
    match err {
        PrismError::IncompatibleBackend { reason, .. } => {
            assert!(
                reason.contains("addressable memory"),
                "expected an addressability rejection, got {reason}"
            );
        }
        other => panic!("expected a clean addressability error, got {other:?}"),
    }
}
