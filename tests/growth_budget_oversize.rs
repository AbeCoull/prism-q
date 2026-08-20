//! Oversize guards for the growth-path memory ceilings. Isolated in its own
//! test binary: it overrides the per-path cap variables, which the helpers
//! cache per process. Every rejection is decided before the growth allocation,
//! so no test allocates an oversize state.

use std::sync::Once;

use num_complex::Complex64;
use prism_q::gates::Gate;
use prism_q::{
    Circuit, FactoredBackend, FactoredStabilizerBackend, MpsBackend, PrismError, SparseBackend,
    run_on,
};

// Merge cap 8: a factored merge past 8 qubits rejects. MPS workspace cap
// 2^8 = 256 amplitudes. Sparse cap 6: the map may hold at most 64 entries.
// The statevector cap stays untouched: these ceilings must bind on their own.
const MERGE_CAP: usize = 8;
const SPARSE_ENTRY_CAP: usize = 1 << 6;

fn small_caps() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries a cap while it is written.
        unsafe {
            std::env::set_var("PRISM_MAX_FACTORED_MERGE_QUBITS", "8");
            std::env::set_var("PRISM_MAX_MPS_WORKSPACE_QUBITS", "8");
            std::env::set_var("PRISM_MAX_SPARSE_QUBITS", "6");
            std::env::set_var("PRISM_MAX_STABILIZER_CLUSTER_QUBITS", "8");
        }
    });
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

// Two independent blocks built by CX chains, then one bridging CX whose merge
// would need a dense block of `left + right` qubits.
fn bridged_blocks(left: usize, right: usize) -> Circuit {
    let n = left + right;
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 1..left {
        c.add_gate(Gate::Cx, &[q - 1, q]);
    }
    c.add_gate(Gate::H, &[left]);
    for q in left + 1..n {
        c.add_gate(Gate::Cx, &[q - 1, q]);
    }
    c.add_gate(Gate::Cx, &[left - 1, left]);
    c
}

// Brickwork of Ry layers and CX ladders: bond dimension doubles per brick
// until the cap, so the MPS workspace grows deterministically.
fn dense_entangler(n: usize, layers: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for layer in 0..layers {
        for q in 0..n {
            c.add_gate(Gate::Ry(0.3 + 0.1 * (layer * n + q) as f64), &[q]);
        }
        let offset = layer % 2;
        for q in (offset..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

fn h_wall(n: usize, hs: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..hs {
        c.add_gate(Gate::H, &[q]);
    }
    c
}

#[test]
fn factored_merge_over_the_cap_is_rejected() {
    small_caps();
    let circuit = bridged_blocks(5, 5);
    let mut backend = FactoredBackend::new(42);
    let err = run_on(&mut backend, &circuit).unwrap_err();
    assert_cap_error(err, "factored");
}

#[test]
fn factored_merge_at_the_cap_still_runs() {
    small_caps();
    let circuit = bridged_blocks(MERGE_CAP / 2, MERGE_CAP / 2);
    let mut backend = FactoredBackend::new(42);
    run_on(&mut backend, &circuit).expect("an at-cap merge must run");
}

#[test]
fn sparse_densifying_circuit_is_rejected() {
    small_caps();
    // Seven branching gates project past the 64-entry cap on the seventh.
    let circuit = h_wall(8, 7);
    let mut backend = SparseBackend::new(42);
    let err = run_on(&mut backend, &circuit).unwrap_err();
    assert_cap_error(err, "sparse");
}

#[test]
fn sparse_state_at_the_cap_still_runs() {
    small_caps();
    let mut circuit = h_wall(8, 6);
    for q in 1..8 {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    let mut backend = SparseBackend::new(42);
    let outcome = run_on(&mut backend, &circuit).expect("an at-cap state must run");
    let probs = outcome.probabilities.expect("probabilities");
    let nonzero = probs.iter().filter(|&p| p > 0.0).count();
    assert_eq!(nonzero, SPARSE_ENTRY_CAP);
}

#[test]
fn mps_workspace_over_the_cap_is_rejected() {
    small_caps();
    let circuit = dense_entangler(8, 6);
    let mut backend = MpsBackend::new(42, 1 << 20);
    let err = run_on(&mut backend, &circuit).unwrap_err();
    assert_cap_error(err, "mps");
}

// The N-site path holds the assembled 4^n gate matrix and its reordered copy
// at peak, so a 4-control MCU needs 512 amplitudes against the cap's 256.
fn wide_mcu(n: usize, num_controls: u8) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let targets: Vec<usize> = (0..=num_controls as usize).collect();
    c.add_gate(Gate::mcu(x, num_controls), &targets);
    c
}

#[test]
fn mps_mcu_gate_matrix_over_the_cap_is_rejected() {
    small_caps();
    let circuit = wide_mcu(8, 4);
    let mut backend = MpsBackend::new(42, 1 << 20);
    let err = run_on(&mut backend, &circuit).unwrap_err();
    assert_cap_error(err, "mps");
}

#[test]
fn mps_narrow_mcu_still_runs() {
    small_caps();
    let circuit = wide_mcu(8, 2);
    let mut backend = MpsBackend::new(42, 1 << 20);
    run_on(&mut backend, &circuit).expect("an at-cap MCU must run");
}

#[test]
fn stabilizer_cluster_merge_over_the_cap_is_rejected() {
    small_caps();
    let circuit = bridged_blocks(5, 5);
    let mut backend = FactoredStabilizerBackend::new(42);
    let err = run_on(&mut backend, &circuit).unwrap_err();
    assert_cap_error(err, "factored-stabilizer");
}

#[test]
fn stabilizer_cluster_merge_at_the_cap_still_runs() {
    small_caps();
    let circuit = bridged_blocks(MERGE_CAP / 2, MERGE_CAP / 2);
    let mut backend = FactoredStabilizerBackend::new(42);
    run_on(&mut backend, &circuit).expect("an at-cap cluster merge must run");
}

#[test]
fn mps_low_bond_circuit_still_runs() {
    small_caps();
    let circuit = bridged_blocks(2, 2);
    let mut backend = MpsBackend::new(42, 1 << 20);
    run_on(&mut backend, &circuit).expect("a low-bond circuit must run");
}
