//! Fusion pipeline correctness across circuit types, sizes, and threshold
//! boundaries. Each test compares unfused execution (manual apply loop)
//! against fused execution (`sim::run_on`, which runs the full pipeline).

mod common;

use common::assert_probs_close as assert_probs_close_labeled;
use common::circuits as corpus;
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim;

const EPS: f64 = 1e-10;

/// Run a circuit without any fusion (manual init + apply loop).
///
/// The statevector backend supports `Gate::QftBlock` natively, so this
/// helper still tests "without fusion" rather than "without QftBlock".
/// Expand here so the unfused reference path uses raw textbook gates and
/// remains comparable with fused execution.
fn run_unfused(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let expanded = prism_q::circuit::expand_qft_blocks(circuit);
    for instr in &expanded.instructions {
        backend.apply(instr).unwrap();
    }
    backend.probabilities().unwrap()
}

fn run_fused(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn run_unfused_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    let expanded = prism_q::circuit::expand_qft_blocks(circuit);
    for instr in &expanded.instructions {
        backend.apply(instr).unwrap();
    }
    backend.export_statevector().unwrap()
}

fn run_fused_state(circuit: &Circuit) -> Vec<Complex64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.export_statevector().unwrap()
}

fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_probs_close_labeled(actual, expected, eps, "fusion");
}

fn assert_state_close(actual: &[Complex64], expected: &[Complex64], eps: f64) {
    assert_eq!(actual.len(), expected.len(), "state vector length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *e).norm() < eps,
            "amp[{i}]: expected {e}, got {a} (diff {})",
            (*a - *e).norm()
        );
    }
}

fn assert_fusion_preserves_correctness(circuit: &Circuit) {
    let unfused = run_unfused(circuit);
    let fused = run_fused(circuit);
    assert_probs_close(&fused, &unfused, EPS);
}

fn assert_fusion_preserves_state(circuit: &Circuit) {
    let unfused = run_unfused_state(circuit);
    let fused = run_fused_state(circuit);
    assert_state_close(&fused, &unfused, EPS);
}

#[test]
fn subrange_qft_block_matches_textbook_expansion() {
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[4]);
    c.add_gate(Gate::QftBlock { start: 1, num: 3 }, &[1, 2, 3]);
    c.add_gate(Gate::Ry(0.37), &[2]);

    assert_fusion_preserves_state(&c);
}

fn has_2q_fusion(circuit: &Circuit) -> bool {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::Fused2q(_) | Gate::Multi2q(_),
                ..
            }
        )
    })
}

// ===== QFT =====

#[test]
fn fusion_qft_4q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(4));
}

#[test]
fn fusion_qft_8q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(8));
}

#[test]
fn fusion_qft_10q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(10));
}

#[test]
fn fusion_qft_12q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(12));
}

#[test]
fn fusion_qft_14q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(14));
}

#[test]
fn fusion_qft_16q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(16));
}

// ===== QFT amplitude-level checks (locks the FP-3 fix) =====
//
// Probability-only goldens are sign-invariant. Until 2026-05-11, fusion's
// `fuse_controlled_phases` pass reordered `cphase(c, t)` past non-diagonal
// 1q gates on `t` (notably `H[t]` immediately after, in the textbook QFT
// pattern). The result was probability-correct but amplitude-wrong on
// QFT-style circuits at >= 16q. These tests compare the full state vector
// element-wise so any future regression on the same shape will fail.

#[test]
fn fusion_qft_state_amplitude_8q() {
    assert_fusion_preserves_state(&circuits::qft_circuit(8));
}

#[test]
fn fusion_qft_state_amplitude_12q() {
    assert_fusion_preserves_state(&circuits::qft_circuit(12));
}

#[test]
fn fusion_qft_state_amplitude_16q() {
    assert_fusion_preserves_state(&circuits::qft_circuit(16));
}

#[test]
fn fusion_qft_state_amplitude_18q() {
    assert_fusion_preserves_state(&circuits::qft_circuit(18));
}

#[test]
fn fusion_qft_state_amplitude_20q() {
    assert_fusion_preserves_state(&circuits::qft_circuit(20));
}

#[test]
fn fusion_phase_estimation_state_amplitude_12q() {
    assert_fusion_preserves_state(&circuits::phase_estimation_circuit(12));
}

#[test]
fn fusion_phase_estimation_state_amplitude_16q() {
    assert_fusion_preserves_state(&circuits::phase_estimation_circuit(16));
}

// ===== Hardware-Efficient Ansatz =====

#[test]
fn fusion_hea_4q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(4, 5, 42));
}

#[test]
fn fusion_hea_8q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(8, 5, 42));
}

#[test]
fn fusion_hea_10q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(10, 5, 42));
}

#[test]
fn fusion_hea_12q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(12, 5, 42));
}

#[test]
fn fusion_hea_14q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(14, 5, 42));
}

#[test]
fn fusion_hea_16q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(16, 5, 42));
}

// ===== Random circuits =====

#[test]
fn fusion_random_4q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(4, 10, 42));
}

#[test]
fn fusion_random_8q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(8, 10, 42));
}

#[test]
fn fusion_random_10q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(10, 10, 42));
}

#[test]
fn fusion_random_12q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(12, 10, 42));
}

#[test]
fn fusion_random_14q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(14, 10, 42));
}

#[test]
fn fusion_random_16q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(16, 10, 42));
}

// ===== Clifford-heavy circuits =====

#[test]
fn fusion_clifford_4q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(4, 10, 42));
}

#[test]
fn fusion_clifford_8q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(8, 10, 42));
}

#[test]
fn fusion_clifford_10q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(10, 10, 42));
}

#[test]
fn fusion_clifford_12q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(12, 10, 42));
}

#[test]
fn fusion_clifford_14q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(14, 10, 42));
}

#[test]
fn fusion_clifford_16q() {
    assert_fusion_preserves_correctness(&circuits::clifford_heavy_circuit(16, 10, 42));
}

// ===== Phase estimation =====

#[test]
fn fusion_phase_estimation_4q() {
    assert_fusion_preserves_correctness(&circuits::phase_estimation_circuit(4));
}

#[test]
fn fusion_phase_estimation_8q() {
    assert_fusion_preserves_correctness(&circuits::phase_estimation_circuit(8));
}

#[test]
fn fusion_phase_estimation_12q() {
    assert_fusion_preserves_correctness(&circuits::phase_estimation_circuit(12));
}

// ===== Edge cases =====

#[test]
fn fusion_empty_circuit() {
    let c = Circuit::new(2, 0);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_single_1q_gate() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_single_2q_gate() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_only_2q_gates() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[2, 3]);
    c.add_gate(Gate::Cx, &[3, 0]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_only_1q_gates() {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::S, &[q]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_bell_state_2q() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_ghz_4q() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 0..3 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_alternating_1q_2q_pattern() {
    let mut c = Circuit::new(4, 0);
    for _ in 0..5 {
        for q in 0..4 {
            c.add_gate(Gate::Ry(1.23), &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        for q in 0..4 {
            c.add_gate(Gate::Rz(0.45), &[q]);
        }
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Cx, &[3, 0]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_cu_gates() {
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[1, 2]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Cu(Box::new(h_mat)), &[2, 3]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_cz_cx_mixed() {
    let mut c = Circuit::new(4, 0);
    for _ in 0..3 {
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cz, &[2, 3]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_self_inverse_cx_cancellation() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_swap_cancellation() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_gate(Gate::Cx, &[0, 2]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_deep_single_qubit_chain() {
    let mut c = Circuit::new(2, 0);
    for _ in 0..20 {
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Rx(0.1), &[0]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_fusion_preserves_correctness(&c);
}

// ===== Parametric rotation sweep =====

#[test]
fn fusion_rotation_sweep() {
    for angle_idx in 0..8 {
        let theta = std::f64::consts::TAU * (angle_idx as f64) / 8.0;
        let mut c = Circuit::new(4, 0);
        for q in 0..4 {
            c.add_gate(Gate::Rx(theta), &[q]);
            c.add_gate(Gate::Ry(theta * 0.7), &[q]);
            c.add_gate(Gate::Rz(theta * 1.3), &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        for q in 0..4 {
            c.add_gate(Gate::P(theta * 0.5), &[q]);
        }
        assert_fusion_preserves_correctness(&c);
    }
}

// ===== Threshold boundary tests =====

#[test]
fn fusion_threshold_9q_no_fusion() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(9, 10, 42));
}

#[test]
fn fusion_threshold_10q_basic_fusion() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(10, 10, 42));
}

#[test]
fn fusion_threshold_13q_before_multi() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(13, 10, 42));
}

#[test]
fn fusion_threshold_14q_multi_fusion() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(14, 10, 42));
}

#[test]
fn fusion_threshold_15q_before_cphase() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(15));
}

#[test]
fn fusion_threshold_16q_cphase_fusion() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(16));
}

circuit_case_tests! {
    cases: corpus::fusion_threshold_cases(),
    runner: assert_fusion_preserves_correctness,
    tests: {
        fusion_threshold_boundary_9q => "fusion_threshold_9",
        fusion_threshold_boundary_10q => "fusion_threshold_10",
        fusion_threshold_boundary_11q => "fusion_threshold_11",
        fusion_threshold_boundary_12q => "fusion_threshold_12",
        fusion_threshold_boundary_13q => "fusion_threshold_13",
        fusion_threshold_boundary_14q => "fusion_threshold_14",
        fusion_threshold_boundary_15q => "fusion_threshold_15",
        fusion_threshold_boundary_16q => "fusion_threshold_16",
        fusion_threshold_boundary_17q => "fusion_threshold_17",
        fusion_threshold_boundary_18q => "fusion_threshold_18",
        fusion_threshold_boundary_19q => "fusion_threshold_19",
    }
}

// ===== Large circuits (higher qubit counts) =====

#[test]
fn fusion_hea_18q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(18, 3, 42));
}

#[test]
fn fusion_random_18q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(18, 5, 42));
}

// ===== 2q fusion specific tests =====
// The fuse_2q_gates pass activates at 12 qubits and above.

#[test]
fn fusion_2q_hea_pattern_12q() {
    let circuit = circuits::hardware_efficient_ansatz(12, 3, 42);
    assert!(has_2q_fusion(&circuit), "12q HEA should use 2q fusion");
    assert_fusion_preserves_correctness(&circuit);
}

#[test]
fn fusion_2q_cx_sandwich_20q() {
    // Every CX is sandwiched by Ry/Rz on both qubits
    let mut c = Circuit::new(20, 0);
    for _ in 0..3 {
        for q in 0..20 {
            c.add_gate(Gate::Ry(1.23 + q as f64 * 0.1), &[q]);
            c.add_gate(Gate::Rz(0.45 + q as f64 * 0.2), &[q]);
        }
        for q in (0..19).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..20 {
            c.add_gate(Gate::Rx(0.67 + q as f64 * 0.15), &[q]);
        }
        for q in (1..19).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_2q_cz_with_1q_gates_12q() {
    // CZ gates are absorbed by 2q fusion like CX.
    let mut c = Circuit::new(12, 0);
    for _ in 0..3 {
        for q in 0..12 {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::T, &[q]);
        }
        for q in (0..11).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
        for q in 0..12 {
            c.add_gate(Gate::S, &[q]);
        }
        for q in (1..11).step_by(2) {
            c.add_gate(Gate::Cz, &[q, q + 1]);
        }
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_2q_swap_with_1q_gates_12q() {
    // SWAP is NOT absorbed by 2q fusion (specialized SIMD kernel is faster).
    let mut c = Circuit::new(12, 0);
    for q in 0..12 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in (0..11).step_by(2) {
        c.add_gate(Gate::Swap, &[q, q + 1]);
    }
    for q in 0..12 {
        c.add_gate(Gate::Ry(0.5), &[q]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_2q_cu_with_1q_gates_12q() {
    // Cu is NOT absorbed by 2q fusion (preserves cphase batching downstream).
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(12, 0);
    for q in 0..12 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in (0..11).step_by(2) {
        c.add_gate(Gate::Cu(Box::new(h_mat)), &[q, q + 1]);
    }
    for q in 0..12 {
        c.add_gate(Gate::T, &[q]);
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_2q_threshold_11q_no_2q_fusion() {
    let circuit = circuits::hardware_efficient_ansatz(11, 3, 42);
    assert!(!has_2q_fusion(&circuit), "11q HEA should skip 2q fusion");
    assert_fusion_preserves_correctness(&circuit);
}

#[test]
fn fusion_2q_threshold_12q_2q_fusion_active() {
    let circuit = circuits::hardware_efficient_ansatz(12, 3, 42);
    assert!(has_2q_fusion(&circuit), "12q HEA should use 2q fusion");
    assert_fusion_preserves_correctness(&circuit);
}

#[test]
fn fusion_2q_mixed_2q_gates_20q() {
    // Mix of CX, CZ, SWAP, Cu. CX and CZ get 2q-fused at 12q and above.
    let h_mat = Gate::H.matrix_2x2();
    let mut c = Circuit::new(20, 0);
    for _ in 0..3 {
        for q in 0..20 {
            c.add_gate(Gate::Ry(q as f64 * 0.3), &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cz, &[2, 3]);
        c.add_gate(Gate::Swap, &[4, 5]);
        c.add_gate(Gate::Cu(Box::new(h_mat)), &[6, 7]);
        c.add_gate(Gate::Cx, &[8, 9]);
        c.add_gate(Gate::Cz, &[10, 11]);
        c.add_gate(Gate::Swap, &[12, 13]);
        c.add_gate(Gate::Cx, &[14, 15]);
        c.add_gate(Gate::Cx, &[16, 17]);
        c.add_gate(Gate::Cx, &[18, 19]);
        for q in 0..20 {
            c.add_gate(Gate::Rz(q as f64 * 0.2), &[q]);
        }
    }
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_same_pair_w_state_20q() {
    assert_fusion_preserves_correctness(&circuits::w_state_circuit(20));
}

#[test]
fn fusion_same_pair_qv_8q() {
    assert_fusion_preserves_correctness(&circuits::quantum_volume_circuit(8, 1, 42));
}

#[test]
fn fusion_same_pair_qv_12q() {
    assert_fusion_preserves_correctness(&circuits::quantum_volume_circuit(12, 1, 42));
}

#[test]
fn fusion_same_pair_qv_16q() {
    assert_fusion_preserves_correctness(&circuits::quantum_volume_circuit(16, 1, 42));
}

#[test]
fn fusion_same_pair_qv_20q() {
    assert_fusion_preserves_correctness(&circuits::quantum_volume_circuit(20, 1, 42));
}

#[test]
fn fusion_qv_20q_depth_4() {
    // Multi-layer QV at 20q exercises `reorder_disjoint_fused2q`: each layer
    // emits ~10 disjoint Fused2q gates with mixed L2/L3/Individual tiers,
    // and the reorder must commute them into tier-grouped order without
    // changing the final state.
    assert_fusion_preserves_state(&circuits::quantum_volume_circuit(20, 4, 42));
}

#[test]
fn fusion_same_pair_reversed_targets_20q() {
    let mut c = Circuit::new(20, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.31), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.7), &[0]);
    c.add_gate(Gate::Ry(-0.4), &[1]);
    c.add_gate(Gate::Cx, &[1, 0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.2), &[1]);
    assert_fusion_preserves_state(&c);
}

#[test]
fn fusion_same_pair_keeps_diagonal_batch_paths() {
    let qaoa = circuits::qaoa_circuit(20, 3, 42);
    let qaoa_fused = prism_q::circuit::fusion::fuse_circuit(&qaoa, true);
    let batch_rzz = qaoa_fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                prism_q::circuit::Instruction::Gate {
                    gate: Gate::BatchRzz(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(batch_rzz, 3, "qaoa should keep BatchRzz fusion");

    // QFT emits one QftBlock. Expanded QFT still exercises BatchPhase.
    let qft = circuits::qft_circuit(20);
    let qft_fused = prism_q::circuit::fusion::fuse_circuit(&qft, true);
    let qft_block = qft_fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                prism_q::circuit::Instruction::Gate {
                    gate: Gate::QftBlock { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(qft_block, 1, "qft should emit a single QftBlock");

    let expanded = prism_q::circuit::expand_qft_blocks(&qft);
    let expanded_fused = prism_q::circuit::fusion::fuse_circuit(&expanded, true);
    let batch_phase_after_expand = expanded_fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                prism_q::circuit::Instruction::Gate {
                    gate: Gate::BatchPhase(_),
                    ..
                }
            )
        })
        .count();
    assert!(
        batch_phase_after_expand > 0,
        "expanded qft should still hit BatchPhase fusion"
    );

    let mut diagonal = Circuit::new(20, 0);
    diagonal.add_gate(Gate::Rzz(0.1), &[0, 1]);
    diagonal.add_gate(Gate::Rz(0.2), &[0]);
    let diagonal_fused = prism_q::circuit::fusion::fuse_circuit(&diagonal, true);
    let diagonal_batch = diagonal_fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                prism_q::circuit::Instruction::Gate {
                    gate: Gate::DiagonalBatch(_),
                    ..
                }
            )
        })
        .count();
    assert!(
        diagonal_batch > 0,
        "diagonal runs should keep DiagonalBatch fusion"
    );
}

/// A non-diagonal 1q gate may only sink past a BatchRzz when its qubit is
/// absent from the whole batch. Deferring it on "not seen in the run so far"
/// let the H below move behind the Rzz gates on q12 that follow it, which
/// silently dropped the Rz contributions the Rx then acted on.
fn batch_rzz_barrier_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    for (a, b) in [(1, 5), (4, 6), (0, 9), (8, 9), (3, 11)] {
        c.add_gate(Gate::Rzz(0.7), &[a, b]);
    }
    c.add_gate(Gate::H, &[12]);
    for a in [1, 2, 6, 7, 8, 10, 11] {
        c.add_gate(Gate::Rzz(0.7), &[a, 12]);
    }
    c.add_gate(Gate::Rx(0.9), &[12]);
    c.add_gate(Gate::Rzz(0.7), &[0, 13]);
    c
}

#[test]
fn fusion_batch_rzz_respects_non_diagonal_barrier() {
    for n in 14..=18 {
        let circuit = batch_rzz_barrier_circuit(n);
        let unfused = run_unfused_state(&circuit);
        let fused = run_fused_state(&circuit);
        assert_eq!(fused.len(), unfused.len(), "{n}q state length mismatch");
        for (i, (a, e)) in fused.iter().zip(&unfused).enumerate() {
            assert!(
                (*a - *e).norm() < EPS,
                "{n}q amp[{i}]: expected {e}, got {a}"
            );
        }
    }
}

#[test]
fn fusion_batch_rzz_still_batches_around_a_barrier() {
    let circuit = batch_rzz_barrier_circuit(16);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let batches = fused
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                prism_q::circuit::Instruction::Gate {
                    gate: Gate::BatchRzz(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        batches, 2,
        "the barrier should split the run into two batches, not disable batching"
    );
}

/// `fuse_diagonal_batch` carries the same deferral rule as `fuse_batch_rzz`, over
/// a much wider gate set (Z, S, T, Rz, P, CZ, Rzz, CPhase). Deferring the first
/// SX below on "q14 is not in the run yet" put the Rzz on q14 ahead of it.
fn diagonal_batch_barrier_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    c.add_gate(Gate::Cz, &[9, 0]);
    c.add_gate(Gate::SX, &[14]);
    c.add_gate(Gate::Rzz(8.25), &[14, 10]);
    c.add_gate(Gate::SX, &[14]);
    c
}

/// The same shape reached through a controlled phase rather than an Rzz.
fn controlled_phase_barrier_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    c.add_gate(Gate::H, &[15]);
    c.add_gate(Gate::Rx(2.81), &[1]);
    c.add_gate(
        Gate::cu([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, 2.9)],
        ]),
        &[1, 9],
    );
    c.add_gate(
        Gate::cu([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, 1.28)],
        ]),
        &[1, 15],
    );
    c.add_gate(Gate::H, &[15]);
    c
}

#[test]
fn fusion_diagonal_batch_respects_non_diagonal_barrier() {
    for n in 16..=18 {
        assert_fusion_preserves_state(&diagonal_batch_barrier_circuit(n));
        assert_fusion_preserves_state(&controlled_phase_barrier_circuit(n));
    }
}

// Pins the fused form for the `statevector/diag_mixed_l6` bench rows: the rows
// measure the `DiagonalBatch` sweep, so the batch must actually appear. Sizes
// cross the parallel threshold, so the state comparison also exercises the
// parallel sweep against the sequential per-gate reference.
#[test]
fn fusion_diag_mixed_batches_and_matches_unfused() {
    for &n in &[16usize, 20] {
        let c = circuits::diagonal_mixed_circuit(n, 6, common::SEED);
        let fused = prism_q::circuit::fusion::fuse_circuit(&c, true);
        let batches = fused
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: Gate::DiagonalBatch(_),
                        ..
                    }
                )
            })
            .count();
        assert!(
            batches > 0,
            "expected a DiagonalBatch in the fused stream at {n}q"
        );
        assert_fusion_preserves_state(&c);
    }
}

// Seeded sweep over the gate mix that exposed both batching reorder bugs. The
// deterministic cases above pin the two known shapes; this covers the class,
// which stayed hidden because the generated bench circuits never produce it.
#[test]
fn fusion_seeded_sweep_matches_unfused() {
    let mut state = common::SEED;
    let mut next = |bound: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % bound
    };

    for _ in 0..150 {
        let n = 16 + next(2) as usize;
        let depth = 6 + next(20) as usize;
        let mut c = Circuit::new(n, 0);
        for _ in 0..depth {
            let q0 = next(n as u64) as usize;
            let q1 = next(n as u64) as usize;
            let theta = next(1000) as f64 / 100.0;
            match next(13) {
                0 => c.add_gate(Gate::H, &[q0]),
                1 => c.add_gate(Gate::X, &[q0]),
                2 => c.add_gate(Gate::Rx(theta), &[q0]),
                3 => c.add_gate(Gate::Ry(theta), &[q0]),
                4 => c.add_gate(Gate::Rz(theta), &[q0]),
                5 => c.add_gate(Gate::T, &[q0]),
                6 => c.add_gate(Gate::S, &[q0]),
                7 => c.add_gate(Gate::P(theta), &[q0]),
                8 => c.add_gate(Gate::SX, &[q0]),
                9 if q0 != q1 => c.add_gate(Gate::Cz, &[q0, q1]),
                10 if q0 != q1 => c.add_gate(Gate::Rzz(theta), &[q0, q1]),
                11 if q0 != q1 => c.add_gate(
                    Gate::cu([
                        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                        [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, theta)],
                    ]),
                    &[q0, q1],
                ),
                12 if q0 != q1 => c.add_gate(Gate::Cx, &[q0, q1]),
                _ => c.add_gate(Gate::H, &[q0]),
            }
        }
        assert_fusion_preserves_state(&c);
    }
}

#[test]
fn fusion_2q_sparse_backend_12q() {
    use prism_q::backend::sparse::SparseBackend;
    let c = circuits::hardware_efficient_ansatz(12, 3, 42);
    let unfused = run_unfused(&c);

    let mut backend = SparseBackend::new(42);
    sim::run_on(&mut backend, &c).unwrap();
    let sparse_probs = backend.probabilities().unwrap();
    assert_probs_close(&sparse_probs, &unfused, 1e-10);
}

#[test]
fn fusion_2q_mps_backend_12q() {
    use prism_q::backend::mps::MpsBackend;
    let c = circuits::hardware_efficient_ansatz(12, 3, 42);
    let unfused = run_unfused(&c);

    let mut backend = MpsBackend::new(42, 256);
    sim::run_on(&mut backend, &c).unwrap();
    let mps_probs = backend.probabilities().unwrap();
    // MPS has looser tolerance due to SVD truncation
    assert_probs_close(&mps_probs, &unfused, 1e-6);
}

// ===== Dynamic fusion threshold tests =====

#[test]
fn fusion_dynamic_threshold_shallow_10q() {
    // Fewer instructions than MIN_INSTRUCTIONS_FOR_FUSION = 20, so fusion is
    // skipped (Cow::Borrowed).
    let mut c = Circuit::new(10, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_dynamic_threshold_deep_10q() {
    // 10q circuit with many instructions; fusion should be active
    assert_fusion_preserves_correctness(&circuits::random_circuit(10, 20, 42));
}

// ===== batch_post_phase_1q pass (≥18q) =====
// The 7th fusion pass batches consecutive 1q gates into MultiFused after
// fuse_controlled_phases separates H gates from BatchPhase gates in QFT.

#[test]
fn fusion_batch_post_phase_qft_18q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(18));
}

#[test]
fn fusion_batch_post_phase_qft_20q() {
    assert_fusion_preserves_correctness(&circuits::qft_circuit(20));
}

#[test]
fn fusion_all_passes_random_20q() {
    assert_fusion_preserves_correctness(&circuits::random_circuit(20, 10, 42));
}

#[test]
fn fusion_all_passes_hea_20q() {
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(20, 5, 42));
}

// ===== Non-adjacent self-inverse cancellation =====
// cancel_self_inverse_pairs should cancel CX·CX even with non-conflicting gates between.

#[test]
fn fusion_non_adjacent_cx_cancel() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]); // unrelated qubit, doesn't block cancellation
    c.add_gate(Gate::T, &[3]); // unrelated qubit
    c.add_gate(Gate::Cx, &[0, 1]); // should cancel with first CX
    c.add_gate(Gate::H, &[0]);

    let unfused = run_unfused(&c);
    let fused = run_fused(&c);
    assert_probs_close(&fused, &unfused, EPS);
}

#[test]
fn fusion_non_adjacent_cz_cancel() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::Rx(0.5), &[2]); // unrelated
    c.add_gate(Gate::Cz, &[0, 1]); // should cancel
    let unfused = run_unfused(&c);
    let fused = run_fused(&c);
    assert_probs_close(&fused, &unfused, EPS);
}

#[test]
fn fusion_non_adjacent_swap_cancel() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_gate(Gate::T, &[2]); // unrelated
    c.add_gate(Gate::Swap, &[0, 1]); // should cancel
    let unfused = run_unfused(&c);
    let fused = run_fused(&c);
    assert_probs_close(&fused, &unfused, EPS);
}

#[test]
fn fusion_non_adjacent_cancel_blocked_by_conflict() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[0]); // touches q0, blocks cancellation of CX pair
    c.add_gate(Gate::Cx, &[0, 1]);
    let unfused = run_unfused(&c);
    let fused = run_fused(&c);
    assert_probs_close(&fused, &unfused, EPS);
}

// ===== Multi-2q fusion threshold tests =====

#[test]
fn fusion_multi_2q_threshold_12q_active() {
    let circuit = circuits::quantum_volume_circuit(12, 1, 42);
    assert!(has_2q_fusion(&circuit), "12q QV should use Multi2q fusion");
    assert_fusion_preserves_correctness(&circuit);
}

// ===== Fused-batch producer bounds =====

// A dense Rzz layer is unbounded in edge count while the kernel group tables
// are not, so the producer has to split the run.
fn dense_rzz_layer_circuit(num_qubits: usize, span: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        c.add_gate(Gate::H, &[q]);
    }
    for a in 0..span {
        for b in (a + 1)..span {
            c.add_gate(Gate::Rzz(0.13 * (a + b + 1) as f64), &[a, b]);
        }
    }
    for q in 0..num_qubits {
        c.add_gate(Gate::H, &[q]);
    }
    c
}

#[test]
fn fusion_batch_rzz_beyond_group_capacity() {
    let circuit = dense_rzz_layer_circuit(16, 10);
    let unfused = run_unfused(&circuit);
    let fused = run_fused(&circuit);
    assert_probs_close(&fused, &unfused, EPS);
}

#[test]
fn fusion_batch_rzz_splits_oversize_run() {
    let circuit = dense_rzz_layer_circuit(16, 10);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    let mut edges = 0usize;
    for inst in &fused.instructions {
        if let prism_q::circuit::Instruction::Gate {
            gate: Gate::BatchRzz(data),
            ..
        } = inst
        {
            assert!(
                data.edges.len() <= 32,
                "BatchRzz carries {} edges, past the kernel group capacity",
                data.edges.len()
            );
            edges += data.edges.len();
        }
    }
    assert_eq!(edges, 45, "every Rzz edge should survive the split");
}

// Two cphase gates on the same (control, target) pair index one bit of the
// PEXT mask, so the second phase is dropped unless the producer folds it in.
fn stacked_cphase_circuit(num_qubits: usize) -> Circuit {
    let mut c = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        c.add_gate(Gate::H, &[q]);
    }
    c.add_gate(Gate::cphase(0.4), &[0, 1]);
    c.add_gate(Gate::cphase(0.7), &[0, 1]);
    c.add_gate(Gate::cphase(1.1), &[0, 2]);
    for q in 0..num_qubits {
        c.add_gate(Gate::H, &[q]);
    }
    c
}

#[test]
fn fusion_batch_phase_duplicate_pair() {
    let circuit = stacked_cphase_circuit(16);
    let unfused = run_unfused(&circuit);
    let fused = run_fused(&circuit);
    assert_probs_close(&fused, &unfused, EPS);
}

#[test]
fn fusion_batch_phase_folds_duplicate_pair() {
    let circuit = stacked_cphase_circuit(16);
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);
    for inst in &fused.instructions {
        if let prism_q::circuit::Instruction::Gate {
            gate: Gate::BatchPhase(data),
            ..
        } = inst
        {
            let mut seen: Vec<usize> = data.phases.iter().map(|&(t, _)| t).collect();
            seen.sort_unstable();
            let len = seen.len();
            seen.dedup();
            assert_eq!(len, seen.len(), "BatchPhase repeats a target qubit");
        }
    }
}

// qaoa puts three targets past the L3 tier at 20q (17, 18, 19) and two at 19q,
// so both the k=3 and k=2 shapes run; 18q leaves one gate on the single-gate
// path. Compared at amplitude level because the tail carries phases.
#[test]
fn fusion_multi_1q_high_targets_match_unfused() {
    for n in [18usize, 19, 20] {
        let circuit = circuits::qaoa_circuit(n, 3, 42);
        assert_state_close(
            &run_fused_state(&circuit),
            &run_unfused_state(&circuit),
            EPS,
        );
    }
}

// A `Fused2q` the parser built (`ms`, `syc`, `sqrt_iswap`, and the rest of the
// hardware two-qubit bases) absorbs its neighbouring single-qubit gates the way
// `Cx` does. Before this it did not, and a circuit transpiled into one of those
// bases carried every conjugating gate as a separate pass.
#[test]
fn parser_built_fused_2q_absorbs_neighbouring_1q_gates() {
    let n = 12;
    let mut qasm = format!("OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[{n}] q;\n");
    for q in 0..n {
        qasm.push_str(&format!("h q[{q}];\n"));
    }
    for q in 0..n - 1 {
        qasm.push_str(&format!(
            "h q[{q}];\nh q[{}];\nms(0, 0, pi/4) q[{q}], q[{}];\ns q[{q}];\ns q[{}];\n",
            q + 1,
            q + 1,
            q + 1
        ));
    }
    let circuit = prism_q::circuit::openqasm::parse(&qasm).unwrap();
    let fused = prism_q::circuit::fusion::fuse_circuit(&circuit, true);

    let two_qubit = |c: &Circuit| {
        c.instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Gate { gate, .. } if gate.num_qubits() == 2))
            .count()
    };
    assert_eq!(
        two_qubit(&circuit),
        n - 1,
        "the fixture should carry one two-qubit gate per adjacent pair"
    );
    assert!(
        fused.instructions.len() < circuit.instructions.len() / 2,
        "fusion left {} of {} instructions, so the conjugating gates were not absorbed",
        fused.instructions.len(),
        circuit.instructions.len()
    );
    assert_probs_close_labeled(
        &run_unfused(&circuit),
        &run_fused(&circuit),
        EPS,
        "ms-basis chain",
    );
}
