use crate::backend::factored::FactoredBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::Backend;
use crate::circuit::{smallvec, Circuit, ClassicalCondition, Instruction};
use crate::gates::Gate;
use crate::sim;

fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "probability vector length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e}, got {a} (diff {})",
            (a - e).abs()
        );
    }
}

fn compare_with_statevector(circuit: &Circuit, eps: f64) {
    let mut sv = StatevectorBackend::new(42);
    let sv_result = sim::run_on(&mut sv, circuit).unwrap();
    let sv_probs = sv_result.probabilities.unwrap().to_vec();

    let mut fac = FactoredBackend::new(42);
    let fac_result = sim::run_on(&mut fac, circuit).unwrap();
    let fac_probs = fac_result.probabilities.unwrap().to_vec();

    assert_probs_close(&fac_probs, &sv_probs, eps);
}

// ---------- Basic single-qubit gates ----------

#[test]
fn test_x_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::X, &[0]);
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.0, 1.0], 1e-12);
}

#[test]
fn test_h_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.5, 0.5], 1e-12);
}

#[test]
fn test_h_on_second_qubit() {
    // H on q[2] of a 4-qubit circuit. Other qubits are |0⟩.
    // Only states |0000⟩ (idx 0) and |0100⟩ (idx 4) have non-zero probability.
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[2]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_diagonal_gates() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Z, &[0]);
    compare_with_statevector(&c, 1e-10);
}

// ---------- Two-qubit gates (trigger merge) ----------

#[test]
fn test_bell_state() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.5, 0.0, 0.0, 0.5], 1e-12);
}

#[test]
fn test_swap() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Swap, &[0, 1]);
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert_probs_close(&b.probabilities().unwrap(), &[0.0, 1.0, 0.0, 0.0], 1e-12);
}

#[test]
fn test_cz() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cz, &[0, 1]);
    compare_with_statevector(&c, 1e-10);
}

// ---------- Independent groups (factored advantage) ----------

#[test]
fn test_independent_bell_pairs() {
    // Two independent Bell pairs: (0,1) and (2,3). Should stay as 2 sub-states.
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_independent_groups_stay_separate() {
    let mut c = Circuit::new(6, 0);
    // Group A: qubits 0,1
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    // Group B: qubits 2,3
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    // Group C: qubits 4,5
    c.add_gate(Gate::X, &[4]);
    c.add_gate(Gate::Cx, &[4, 5]);

    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();

    // Should have 3 active sub-states
    let active_count = b.substates.iter().filter(|s| s.is_some()).count();
    assert_eq!(active_count, 3);

    compare_with_statevector(&c, 1e-10);
}

// ---------- Non-adjacent qubit merges ----------

#[test]
fn test_cx_non_adjacent() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 3]); // Merge qubits 0 and 3, leave 1,2 separate
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_progressive_merge() {
    // Start with 4 separate qubits, merge progressively
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]); // merge 0,1
    c.add_gate(Gate::Cx, &[2, 3]); // merge 2,3
    c.add_gate(Gate::Cx, &[1, 2]); // merge all
    compare_with_statevector(&c, 1e-10);
}

// ---------- Controlled gates ----------

#[test]
fn test_cu_gate() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    let mat = Gate::H.matrix_2x2();
    c.add_gate(Gate::Cu(Box::new(mat)), &[0, 2]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_cphase() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
    c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_2), &[1, 2]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_mcu_toffoli() {
    use crate::gates::McuData;
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat: Gate::X.matrix_2x2(),
            num_controls: 2,
        })),
        &[0, 1, 2],
    );
    compare_with_statevector(&c, 1e-10);
}

// ---------- Measurement ----------

#[test]
fn test_measurement() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    assert!(b.classical_results()[0]); // q[0] was |1⟩
    assert!(!b.classical_results()[1]); // q[1] was |0⟩
}

#[test]
fn test_measurement_in_substate() {
    // Measure one qubit in a Bell pair
    let mut c = Circuit::new(4, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::X, &[2]); // independent qubit
    c.add_measure(0, 0);

    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    // After measurement, q[0] and q[1] should be correlated (Bell state collapse)
    // q[2] should still be |1⟩ independently
}

// ---------- Golden tests: factored vs statevector on circuit builders ----------

#[test]
fn test_golden_qft_8() {
    let circuit = crate::circuits::qft_circuit(8);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_qft_12() {
    let circuit = crate::circuits::qft_circuit(12);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_random_16() {
    let circuit = crate::circuits::random_circuit(16, 10, 42);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_hea_8() {
    let circuit = crate::circuits::hardware_efficient_ansatz(8, 3, 42);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_bell_pairs() {
    let circuit = crate::circuits::independent_bell_pairs(6);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_independent_blocks() {
    let circuit = crate::circuits::independent_random_blocks(4, 3, 5, 42);
    compare_with_statevector(&circuit, 1e-10);
}

#[test]
fn test_golden_qpe() {
    let circuit = crate::circuits::phase_estimation_circuit(6);
    compare_with_statevector(&circuit, 1e-10);
}

// ---------- Edge cases ----------

#[test]
fn test_single_qubit_circuit() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_no_entanglement() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::T, &[3]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_immediate_full_merge() {
    // First instruction merges q[0] and q[n-1]
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[0, 2]); // merges all
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_backend_kind_factored() {
    use crate::sim::BackendKind;
    let circuit = crate::circuits::independent_bell_pairs(4);
    let result = sim::run_with(BackendKind::Factored, &circuit, 42).unwrap();
    let probs = result.probabilities.unwrap().to_vec();
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

// ---------- Conditional gates ----------

#[test]
fn test_conditional_gate() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    // Conditional X on q[1] if classical bit 0 is set
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![1],
    });

    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    // q[0] measured as 1, so conditional fires, q[1] becomes |1⟩
    let probs = b.probabilities().unwrap();
    assert!((probs[3] - 1.0).abs() < 1e-10); // |11⟩
}

// ---- Factored backend parallel dispatch tests ----
//
// These tests exercise the factored backend's Rayon-parallelized code paths
// when sub-states grow ≥ PARALLEL_THRESHOLD_QUBITS (14).

#[test]
fn test_factored_parallel_multifused_16q() {
    let mut c = Circuit::new(16, 0);
    for q in 0..16 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..15 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_parallel_cx_large_substate_16q() {
    let mut c = Circuit::new(16, 0);
    for q in 0..15 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in (0..15).step_by(2) {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_parallel_measure_16q() {
    let mut c = Circuit::new(16, 1);
    for q in 0..15 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    let mut fac = FactoredBackend::new(42);
    sim::run_on(&mut fac, &c).unwrap();
    let bits = fac.classical_results();
    assert!(bits[0]);
}

// ---- merge_substates path coverage ----
//
// These tests exercise specific branches in `merge_substates`: the dst-low
// SIMD Kronecker fast path with both arg orders (dst-low and src-low), the
// interleaved-qubit scatter fallback, and the Rayon-parallel branch under
// both balanced and imbalanced sub-state sizes.

#[test]
fn test_merge_src_low_path() {
    // Build {2,3} (the substate that will become dst because targets[0]=q[2]),
    // then build {0,1} (the substate that will become src). CX[2,0] dispatches
    // merge_substates(dst={2,3}, src={0,1}); src.qubits all below dst.qubits
    // forces the src_low branch (kron_low_high with args swapped).
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cx, &[2, 0]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_merge_interleaved_qubits_path() {
    // Build {0,2} and {1,3} sub-states, then merge them. Neither side's
    // qubits are wholly below the other, so merge_substates falls through
    // to kron_scatter (per-element interleaved scatter).
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 2]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 3]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::S, &[3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_merge_interleaved_three_way() {
    // Three-way interleave: {0,4} and {1,2,3}. After merging, qubits 1-3 sit
    // between dst's two qubits, exercising scatter with multiple bit hops.
    let mut c = Circuit::new(5, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 4]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::T, &[4]);
    c.add_gate(Gate::Cx, &[0, 1]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_merge_parallel_balanced_14q() {
    // Build two 7-qubit sub-states then merge: 7+7=14q hits the parallel
    // branch in kron_low_high with high_dim=128 chunks of 128 elements each.
    let mut c = Circuit::new(14, 0);
    for q in 0..7 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..6 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 7..14 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 7..13 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::Cx, &[0, 7]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_merge_parallel_imbalanced_15q() {
    // 14q substate merged with a singleton to 15q: high_dim=2 chunks of
    // 16384 elements. Stresses kron_low_high under-parallel high dimension.
    let mut c = Circuit::new(15, 0);
    for q in 0..14 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..13 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::X, &[14]);
    c.add_gate(Gate::Cx, &[0, 14]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_merge_parallel_interleaved_14q() {
    // Build two interleaved 7-qubit sub-states ({even} and {odd}) then merge,
    // exercising kron_scatter on a state large enough that performance matters
    // even if the scatter path is scalar.
    let mut c = Circuit::new(14, 0);
    for q in (0..14).step_by(2) {
        c.add_gate(Gate::H, &[q]);
    }
    for q in (0..12).step_by(2) {
        c.add_gate(Gate::Cx, &[q, q + 2]);
    }
    for q in (1..14).step_by(2) {
        c.add_gate(Gate::H, &[q]);
    }
    for q in (1..12).step_by(2) {
        c.add_gate(Gate::Cx, &[q, q + 2]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    compare_with_statevector(&c, 1e-10);
}
