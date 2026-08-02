//! Oversize guards for the backend memory-budget contract. Isolated in its
//! own test binary: it overrides `PRISM_MAX_SV_QUBITS` and
//! `PRISM_MAX_DM_QUBITS`, which the cap helpers cache per process. Every
//! rejection is decided before the backend reserves anything, so no test
//! allocates an oversize state.

use std::sync::Once;

use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, PrismError, StatevectorBackend, run_on, simulate};

/// A density matrix of `n` qubits is a `2n`-qubit statevector, so the effective
/// density-matrix cap is half the statevector cap. `PRISM_MAX_DM_QUBITS` is set
/// deliberately above that bound here, which is the case where an override that
/// looks accepted has no room to be honored.
const SV_CAP: usize = 8;
const DM_CAP: usize = SV_CAP / 2;
const DM_OVERRIDE: usize = DM_CAP + 2;

fn small_caps() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries a cap while it is written.
        unsafe {
            std::env::set_var("PRISM_MAX_SV_QUBITS", "8");
            std::env::set_var("PRISM_MAX_DM_QUBITS", "6");
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

// Between the clamp and the override, dispatch used to accept a width that
// `init` then rejected, advising an override the caller had already set.
#[test]
fn density_matrix_override_above_the_clamp_is_rejected_identically_at_both_gates() {
    small_caps();
    for width in [DM_CAP + 1, DM_OVERRIDE + 1] {
        let dispatched = simulate(&entangling_circuit(width))
            .backend(BackendKind::DensityMatrix)
            .seed(42)
            .run()
            .expect_err("dispatch must reject a width the backend cannot allocate");
        let initialized = DensityMatrixBackend::new(42)
            .init(width, 0)
            .expect_err("init must reject the same width");

        for (gate, err) in [("dispatch", &dispatched), ("init", &initialized)] {
            match err {
                PrismError::IncompatibleBackend { backend, reason } => {
                    assert_eq!(backend, "density_matrix", "{gate} named the wrong backend");
                    assert!(
                        reason.contains(&format!("exceeding the cap of {DM_CAP}")),
                        "{width}q {gate} must report the clamped cap, got {reason}"
                    );
                    assert!(
                        reason.contains("PRISM_MAX_DM_QUBITS and PRISM_MAX_SV_QUBITS"),
                        "{width}q {gate} must name both variables that bind, got {reason}"
                    );
                }
                other => panic!("{width}q {gate}: expected a clean cap error, got {other:?}"),
            }
        }
        assert_eq!(
            format!("{dispatched:?}"),
            format!("{initialized:?}"),
            "{width}q: dispatch and init must report the same cap rejection"
        );
    }
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
