//! Cross-backend correctness tests for `ProductStateBackend`.
//!
//! Product state stores per-qubit `[α, β]` and rejects entangling
//! gates. Tests cover separable circuits only: `single_qubit_rotation_circuit`
//! and hand-built 1q-only patterns. Each runs both fused vs unfused on
//! the product backend and SV cross at the same precision.

mod common;

use common::{
    assert_backend_matches_sv, assert_fused_matches_unfused, run_unfused_probs, PRODUCT_EPS, SEED,
};
use num_complex::Complex64;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::Backend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use prism_q::gates::{DiagEntry, DiagonalBatchData, Gate};
use prism_q::sim;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = ProductStateBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, PRODUCT_EPS, label);
}

fn check_fused_vs_unfused(label: &str, circuit: &Circuit) {
    assert_fused_matches_unfused(
        || ProductStateBackend::new(SEED),
        circuit,
        PRODUCT_EPS,
        label,
    );
}

fn has_gate<F: Fn(&Gate) -> bool>(circuit: &Circuit, matches_gate: F) -> bool {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate { gate, .. } if matches_gate(gate)
        )
    })
}

// ===== single_qubit_rotation_circuit =====

#[test]
fn product_single_qubit_rotations_4q_sv() {
    check_sv_cross(
        "single_qubit_rotation 4q d5 sv",
        &circuits::single_qubit_rotation_circuit(4, 5, SEED),
    );
}

#[test]
fn product_single_qubit_rotations_4q_fused() {
    check_fused_vs_unfused(
        "single_qubit_rotation 4q d5 fused",
        &circuits::single_qubit_rotation_circuit(4, 5, SEED),
    );
}

#[test]
fn product_single_qubit_rotations_8q_sv() {
    check_sv_cross(
        "single_qubit_rotation 8q d10 sv",
        &circuits::single_qubit_rotation_circuit(8, 10, SEED),
    );
}

#[test]
fn product_single_qubit_rotations_8q_fused() {
    check_fused_vs_unfused(
        "single_qubit_rotation 8q d10 fused",
        &circuits::single_qubit_rotation_circuit(8, 10, SEED),
    );
}

#[test]
fn product_single_qubit_rotations_12q_sv() {
    check_sv_cross(
        "single_qubit_rotation 12q d10 sv",
        &circuits::single_qubit_rotation_circuit(12, 10, SEED),
    );
}

#[test]
fn product_single_qubit_rotations_16q_sv() {
    check_sv_cross(
        "single_qubit_rotation 16q d5 sv",
        &circuits::single_qubit_rotation_circuit(16, 5, SEED),
    );
}

#[test]
fn product_single_qubit_diagonal_batch_direct() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData {
            entries: vec![
                DiagEntry::Phase1q {
                    qubit: 0,
                    d0: Complex64::new(1.0, 0.0),
                    d1: Complex64::from_polar(1.0, 0.37),
                },
                DiagEntry::Phase1q {
                    qubit: 1,
                    d0: Complex64::new(1.0, 0.0),
                    d1: Complex64::from_polar(1.0, -0.53),
                },
            ],
        })),
        &[0, 1],
    );
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    check_sv_cross("single_qubit_diagonal_batch direct sv", &c);
    check_fused_vs_unfused("single_qubit_diagonal_batch direct fused", &c);
}

// ===== Hand-built separable circuits =====

#[test]
fn product_mixed_clifford_t_4q() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::Y, &[3]);
    c.add_gate(Gate::Sdg, &[1]);
    c.add_gate(Gate::Tdg, &[2]);
    c.add_gate(Gate::H, &[3]);
    check_sv_cross("mixed_clifford_t 4q sv", &c);
    check_fused_vs_unfused("mixed_clifford_t 4q fused", &c);
}

#[test]
fn product_full_rotation_chain_8q() {
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.add_gate(Gate::Rx(0.1 * (q + 1) as f64), &[q]);
        c.add_gate(Gate::Ry(0.2 * (q + 1) as f64), &[q]);
        c.add_gate(Gate::Rz(0.3 * (q + 1) as f64), &[q]);
        c.add_gate(Gate::P(0.05 * (q + 1) as f64), &[q]);
    }
    check_sv_cross("full_rotation_chain 8q sv", &c);
    check_fused_vs_unfused("full_rotation_chain 8q fused", &c);
}

#[test]
fn product_sx_sxdg_8q() {
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.add_gate(Gate::SX, &[q]);
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::SXdg, &[q]);
    }
    check_sv_cross("sx_sxdg 8q sv", &c);
    check_fused_vs_unfused("sx_sxdg 8q fused", &c);
}

// ===== Applicability gate =====

#[test]
fn product_rejects_cx() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(result.is_err(), "ProductState must reject Cx");
}

#[test]
fn product_rejects_cz() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cz, &[0, 1]);
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(result.is_err(), "ProductState must reject Cz");
}

#[test]
fn product_rejects_swap() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Swap, &[0, 1]);
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(result.is_err(), "ProductState must reject Swap");
}

#[test]
fn product_rejects_rzz() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Rzz(0.37), &[0, 1]);
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(result.is_err(), "ProductState must reject Rzz");
}

#[test]
fn product_rejects_batch_rzz_fusion_16q() {
    let c = circuits::qaoa_circuit(16, 1, SEED);
    assert!(
        has_gate(&c, |gate| matches!(gate, Gate::BatchRzz(_))),
        "16q QAOA circuit should use BatchRzz fusion"
    );
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(result.is_err(), "ProductState must reject BatchRzz");
}

#[test]
fn product_rejects_entangling_diagonal_batch_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::Cz, &[2, 3]);
    assert!(
        has_gate(&c, |gate| matches!(gate, Gate::DiagonalBatch(_))),
        "16q CZ run should use DiagonalBatch fusion"
    );
    let mut backend = ProductStateBackend::new(SEED);
    let result = sim::run_on(&mut backend, &c);
    assert!(
        result.is_err(),
        "ProductState must reject multi-qubit DiagonalBatch"
    );
}

#[test]
fn product_unfused_apply_is_separable_only() {
    // The unfused path (init + apply loop) must also reject entangling gates,
    // proving the rejection is in the kernel, not just a fusion-pass guard.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut backend = ProductStateBackend::new(SEED);
    backend.init(c.num_qubits, c.num_classical_bits).unwrap();
    let mut had_err = false;
    for instr in &c.instructions {
        if backend.apply(instr).is_err() {
            had_err = true;
            break;
        }
    }
    assert!(had_err, "raw apply must reject Cx on ProductStateBackend");
}

#[test]
fn product_unfused_runs_separable_circuit() {
    // Sanity: the unfused helper itself must succeed on a separable circuit.
    let c = circuits::single_qubit_rotation_circuit(6, 5, SEED);
    let mut backend = ProductStateBackend::new(SEED);
    let _ = run_unfused_probs(&mut backend, &c);
}
