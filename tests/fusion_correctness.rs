//! Fusion correctness tests.
//!
//! Verifies that the fusion pipeline preserves simulation correctness across
//! circuit types, sizes, and fusion threshold boundaries. Each test compares
//! **unfused** execution (manual backend.apply loop) against **fused** execution
//! (sim::run_on, which applies the full fusion pipeline internally).

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::sim;

const EPS: f64 = 1e-10;

/// Run a circuit without any fusion — manual init + apply loop.
fn run_unfused(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    for instr in &circuit.instructions {
        backend.apply(instr).unwrap();
    }
    backend.probabilities().unwrap()
}

/// Run a circuit with full fusion pipeline via sim::run_on.
fn run_fused(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e:.12}, got {a:.12} (diff {:.2e})",
            (a - e).abs()
        );
    }
}

fn assert_fusion_preserves_correctness(circuit: &Circuit) {
    let unfused = run_unfused(circuit);
    let fused = run_fused(circuit);
    assert_probs_close(&fused, &unfused, EPS);
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
// These specifically target the boundaries where fusion passes activate.

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
// These test circuits where 1q gates are absorbed into adjacent 2q gates.
// The fuse_2q_gates pass activates at ≥20 qubits.

#[test]
fn fusion_2q_hea_pattern_20q() {
    // HEA: Ry-Rz per qubit, then CX ladder — classic 2q fusion target
    // At 20q, 2q fusion is active (MIN_QUBITS_FOR_2Q_FUSION = 20)
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(20, 3, 42));
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
    // CZ gates are NOT absorbed by 2q fusion (specialized SIMD kernel is faster),
    // but correctness should still hold through other fusion passes.
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
fn fusion_2q_threshold_19q_no_2q_fusion() {
    // 19q: below MIN_QUBITS_FOR_2Q_FUSION (20), should still work
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(19, 3, 42));
}

#[test]
fn fusion_2q_threshold_20q_2q_fusion_active() {
    // 20q: at threshold, 2q fusion is active
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(20, 3, 42));
}

#[test]
fn fusion_2q_mixed_2q_gates_20q() {
    // Mix of CX, CZ, SWAP, Cu — only CX gets 2q-fused at ≥20q
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
fn fusion_2q_sparse_backend_12q() {
    // Verify Fused2q works on sparse backend too
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
    // 10q circuit with very few instructions (< MIN_INSTRUCTIONS_FOR_FUSION = 20)
    // Fusion should be skipped (Cow::Borrowed) but results should still be correct
    let mut c = Circuit::new(10, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[1]);
    assert_fusion_preserves_correctness(&c);
}

#[test]
fn fusion_dynamic_threshold_deep_10q() {
    // 10q circuit with many instructions — fusion should be active
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
    c.add_gate(Gate::H, &[2]); // unrelated qubit — doesn't block cancellation
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
    c.add_gate(Gate::H, &[0]); // touches q0 — blocks cancellation of CX pair
    c.add_gate(Gate::Cx, &[0, 1]);
    let unfused = run_unfused(&c);
    let fused = run_fused(&c);
    assert_probs_close(&fused, &unfused, EPS);
}

// ===== Multi-2q fusion threshold tests =====

#[test]
fn fusion_multi_2q_threshold_19q_not_active() {
    // 19q < MIN_QUBITS_FOR_2Q_FUSION(20) — 2q fusion should not fire
    assert_fusion_preserves_correctness(&circuits::hardware_efficient_ansatz(19, 3, 42));
}
