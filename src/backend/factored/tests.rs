use crate::backend::Backend;
use crate::backend::factored::FactoredBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Circuit, ClassicalCondition, Instruction, smallvec};
use crate::gates::Gate;
use crate::sim;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

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

// ---- Basic single-qubit gates ----

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

// ---- Two-qubit gates (trigger merge) ----

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

// ---- Independent groups (factored advantage) ----

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

// ---- Non-adjacent qubit merges ----

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

// ---- Controlled gates ----

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

// ---- Measurement ----

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

// ---- Golden tests: factored vs statevector on circuit builders ----

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

// ---- Edge cases ----

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

// ---- Conditional gates ----

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
// when sub-states grow ≥ PARALLEL_THRESHOLD_QUBITS (15).

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
fn test_merge_parallel_balanced_15q() {
    // Build a 7-qubit and an 8-qubit sub-state then merge: 7+8=15q hits the
    // parallel branch in kron_low_high.
    let mut c = Circuit::new(15, 0);
    for q in 0..7 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..6 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 7..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 7..14 {
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

#[test]
fn test_par_diagonal_z_gate_15q() {
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    // Tie all qubits into one sub-state via a chain of CX
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    // Diagonal 1q gates trigger par_apply_diagonal
    for q in 0..15 {
        c.add_gate(Gate::Z, &[q]);
    }
    // Non-diagonal 1q gate triggers par_apply_1q catch-all
    for q in 0..15 {
        c.add_gate(Gate::X, &[q]);
    }
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_par_swap_15q() {
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::Swap, &[0, 14]);
    c.add_gate(Gate::Cz, &[1, 13]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_substate_larger_than_smallvec_inline() {
    // SmallVec<[usize; 8]> inline capacity; force > 8 qubits in one sub-state.
    let mut c = Circuit::new(10, 0);
    for q in 0..10 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..9 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_reduced_density_matrix_returns_ok() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let rho = b.reduced_density_matrix_1q(0).unwrap();
    let diag = rho[0][0].re + rho[1][1].re;
    assert!((diag - 1.0).abs() < 1e-10);
}

#[test]
fn test_factored_reset_after_measure() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.instructions.push(Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    });
    c.instructions.push(Instruction::Reset { qubit: 0 });
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!(probs[0] > 0.99);
}

#[test]
fn test_factored_conditional_branches() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::X, &[0]);
    c.instructions.push(Instruction::Measure {
        qubit: 0,
        classical_bit: 0,
    });
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![1],
    });
    let mut b = FactoredBackend::new(42);
    sim::run_on(&mut b, &c).unwrap();
    let probs = b.probabilities().unwrap();
    assert!(probs[3] > 0.99);
}

#[test]
fn test_factored_seq_batch_rzz_direct() {
    use crate::gates::BatchRzzData;
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    let data = Box::new(BatchRzzData {
        edges: vec![(0, 1, 0.3), (1, 2, 0.5), (2, 3, 0.7)],
    });
    c.add_gate(Gate::BatchRzz(data), &[0, 1, 2, 3]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_seq_diagonal_batch_direct() {
    use crate::gates::{DiagEntry, DiagonalBatchData};
    use num_complex::Complex64;
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::H, &[3]);
    let phase = Complex64::from_polar(1.0, 0.4);
    let entries = vec![
        DiagEntry::Phase1q {
            qubit: 0,
            d0: Complex64::new(1.0, 0.0),
            d1: phase,
        },
        DiagEntry::Phase2q {
            q0: 1,
            q1: 2,
            phase,
        },
        DiagEntry::Parity2q {
            q0: 2,
            q1: 3,
            same: Complex64::from_polar(1.0, 0.2),
            diff: Complex64::from_polar(1.0, -0.2),
        },
    ];
    let data = Box::new(DiagonalBatchData { entries });
    c.add_gate(Gate::DiagonalBatch(data), &[0, 1, 2, 3]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_seq_mcu_phase_3controls() {
    use crate::gates::McuData;
    use num_complex::Complex64;
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::X, &[q]);
    }
    let phase = Complex64::from_polar(1.0, 0.3);
    let mat = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), phase],
    ];
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 3,
        })),
        &[0, 1, 2, 3],
    );
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_seq_cu_general_matrix() {
    use num_complex::Complex64;
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mat = [
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ],
        [
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ],
    ];
    c.add_gate(Gate::Cu(Box::new(mat)), &[0, 1]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_par_rzz_15q() {
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Rzz(0.1 * q as f64), &[q, q + 1]);
    }
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_par_cu_general_15q() {
    use num_complex::Complex64;
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let mat = [
        [Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)],
        [Complex64::new(0.8, 0.0), Complex64::new(-0.6, 0.0)],
    ];
    c.add_gate(Gate::Cu(Box::new(mat)), &[0, 14]);
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_par_mcu_3control_15q() {
    use crate::gates::McuData;
    use num_complex::Complex64;
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let phase = Complex64::from_polar(1.0, 0.5);
    let mat = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), phase],
    ];
    c.add_gate(
        Gate::Mcu(Box::new(McuData {
            mat,
            num_controls: 3,
        })),
        &[0, 1, 2, 14],
    );
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_par_diagonal_batch_15q() {
    use crate::gates::{DiagEntry, DiagonalBatchData};
    use num_complex::Complex64;
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let entries = vec![
        DiagEntry::Phase1q {
            qubit: 0,
            d0: Complex64::new(1.0, 0.0),
            d1: Complex64::from_polar(1.0, 0.2),
        },
        DiagEntry::Phase2q {
            q0: 1,
            q1: 2,
            phase: Complex64::from_polar(1.0, 0.3),
        },
        DiagEntry::Parity2q {
            q0: 3,
            q1: 5,
            same: Complex64::from_polar(1.0, 0.1),
            diff: Complex64::from_polar(1.0, -0.1),
        },
    ];
    c.add_gate(
        Gate::DiagonalBatch(Box::new(DiagonalBatchData { entries })),
        &[0, 1, 2, 3, 5],
    );
    compare_with_statevector(&c, 1e-10);
}

#[test]
fn test_factored_par_batch_rzz_15q() {
    use crate::gates::BatchRzzData;
    let mut c = Circuit::new(15, 0);
    for q in 0..15 {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..14 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    let edges: Vec<(usize, usize, f64)> =
        (0..14).map(|q| (q, q + 1, 0.05 * (q + 1) as f64)).collect();
    c.add_gate(
        Gate::BatchRzz(Box::new(BatchRzzData { edges })),
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    );
    compare_with_statevector(&c, 1e-10);
}

// ---- Splitting on measurement and reset ----

fn group_widths(b: &FactoredBackend) -> Vec<usize> {
    let mut widths: Vec<usize> = b
        .substates
        .iter()
        .flatten()
        .map(|sub| sub.qubits.len())
        .collect();
    widths.sort_unstable_by(|a, b| b.cmp(a));
    widths
}

fn assert_layout_consistent(b: &FactoredBackend) {
    let mut seen = vec![false; b.num_qubits];
    for (idx, sub) in b.substates.iter().enumerate() {
        let Some(sub) = sub else { continue };
        assert_eq!(sub.state.len(), 1 << sub.qubits.len());
        assert!(sub.qubits.windows(2).all(|w| w[0] < w[1]));
        for &q in &sub.qubits {
            assert!(!seen[q], "qubit {q} sits in two sub-states");
            seen[q] = true;
            assert_eq!(
                b.qubit_to_substate[q], idx,
                "qubit {q} maps to the wrong slot"
            );
        }
    }
    assert!(seen.iter().all(|&s| s), "a qubit belongs to no sub-state");
}

fn observables(n: usize) -> Vec<Vec<PauliTerm>> {
    vec![
        vec![PauliTerm::new(0, PauliAxis::Z)],
        vec![PauliTerm::new(1, PauliAxis::X)],
        vec![
            PauliTerm::new(0, PauliAxis::X),
            PauliTerm::new(n / 2, PauliAxis::Y),
        ],
        (0..n).map(|q| PauliTerm::new(q, PauliAxis::Z)).collect(),
        vec![
            PauliTerm::new(1, PauliAxis::Y),
            PauliTerm::new(n - 1, PauliAxis::X),
            PauliTerm::new(n / 3, PauliAxis::Z),
        ],
    ]
}

// One trajectory on each backend from the same seed draws the same outcomes,
// so the classical record, the amplitudes and the expectations must all agree.
fn assert_trajectory_matches(circuit: &Circuit, seed: u64) -> FactoredBackend {
    let mut sv = StatevectorBackend::new(seed);
    sim::run_on(&mut sv, circuit).unwrap();
    let mut fac = FactoredBackend::new(seed);
    sim::run_on(&mut fac, circuit).unwrap();
    assert_layout_consistent(&fac);

    assert_eq!(
        fac.classical_results(),
        sv.classical_results(),
        "seed {seed}: classical record differs"
    );
    let fac_state = fac.export_statevector().unwrap();
    let sv_state = sv.export_statevector().unwrap();
    for (i, (a, e)) in fac_state.iter().zip(&sv_state).enumerate() {
        assert!(
            (a - e).norm() < 1e-10,
            "seed {seed}: amplitude {i} is {a}, statevector has {e}"
        );
    }
    let obs = observables(circuit.num_qubits);
    let fac_exp = fac.pauli_expectations(&obs).unwrap();
    let sv_exp = sv.pauli_expectations(&obs).unwrap();
    for (k, (a, e)) in fac_exp.iter().zip(&sv_exp).enumerate() {
        assert!(
            (a - e).abs() < 1e-10,
            "seed {seed}: observable {k} is {a}, statevector has {e}"
        );
    }
    fac
}

fn entangled_chain(n: usize, num_bits: usize) -> Circuit {
    let mut c = Circuit::new(n, num_bits);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.4 + 0.3 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        c.add_gate(Gate::Rz(0.2 + 0.17 * q as f64), &[q]);
    }
    c
}

#[test]
fn measure_split_fixture_leaves_the_survivors_alone() {
    for seed in 42..47 {
        let circuit = crate::circuits::measure_split_circuit(16, 4, seed);
        let fac = assert_trajectory_matches(&circuit, seed);
        let mut expected = vec![7, 2];
        expected.extend([1; 7]);
        assert_eq!(group_widths(&fac), expected, "seed {seed}");
    }
}

#[test]
fn measuring_a_wide_group_splits_the_qubit_out() {
    for target in [0usize, 1, 2, 5, 9] {
        let mut c = entangled_chain(10, 1);
        c.add_measure(target, 0);
        for seed in 42..46 {
            let fac = assert_trajectory_matches(&c, seed);
            assert_eq!(group_widths(&fac), vec![9, 1], "target {target}");
        }
    }
}

#[test]
fn reset_splits_to_zero() {
    for target in [0usize, 3, 9] {
        let mut c = entangled_chain(10, 0);
        c.add_reset(target);
        for seed in 42..46 {
            let fac = assert_trajectory_matches(&c, seed);
            assert_eq!(group_widths(&fac), vec![9, 1]);
            let ss = fac.qubit_to_substate[target];
            let single = &fac.substates[ss].as_ref().unwrap().state;
            assert!((single[0].norm() - 1.0).abs() < 1e-12 && single[1].norm() < 1e-12);
        }
    }
}

#[test]
fn narrow_groups_collapse_in_place() {
    let width = super::MIN_SPLIT_QUBITS - 1;
    let mut c = entangled_chain(width, 1);
    c.add_measure(width / 2, 0);
    c.add_reset(0);
    for seed in 42..46 {
        let fac = assert_trajectory_matches(&c, seed);
        assert_eq!(group_widths(&fac), vec![width]);
    }
}

#[test]
fn split_qubits_merge_back_at_every_position() {
    for target in [0usize, 1, 2, 4, 9] {
        let partner = if target == 0 { 9 } else { 0 };
        let mut c = entangled_chain(10, 2);
        c.add_measure(target, 0);
        c.add_gate(Gate::H, &[target]);
        c.add_gate(Gate::Cx, &[target, partner]);
        c.add_gate(Gate::Ry(0.7), &[target]);
        c.add_reset(target);
        c.add_gate(Gate::Rx(1.1), &[target]);
        c.add_gate(Gate::Cz, &[partner, target]);
        c.add_measure(target, 1);
        c.add_gate(Gate::Cx, &[(target + 3) % 10, target]);
        for seed in 42..46 {
            let fac = assert_trajectory_matches(&c, seed);
            assert_eq!(group_widths(&fac), vec![10], "target {target}");
        }
    }
}

#[test]
fn split_and_merge_back_on_parallel_groups() {
    let n = 17;
    let mut c = entangled_chain(n, 4);
    for (bit, &target) in [16usize, 15, 0, 8].iter().enumerate() {
        c.add_measure(target, bit);
    }
    for &target in &[0usize, 8, 16] {
        c.add_gate(Gate::H, &[target]);
        c.add_gate(Gate::Cx, &[target, 7]);
    }
    c.add_reset(16);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.1 * q as f64), &[q]);
    }
    for seed in 42..45 {
        let fac = assert_trajectory_matches(&c, seed);
        assert_eq!(group_widths(&fac), vec![15, 1, 1]);
    }
}

#[test]
fn random_dynamic_circuits_match_statevector() {
    use rand::{RngExt, SeedableRng};
    let n = 11;
    let bits = 6;
    for seed in 42..62u64 {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let mut c = entangled_chain(n, bits);
        for _ in 0..60 {
            let q = rng.random_range(0..n);
            match rng.random_range(0..8) {
                0 => c.add_gate(Gate::Ry(rng.random::<f64>() * 6.0), &[q]),
                1 => c.add_gate(Gate::Rz(rng.random::<f64>() * 6.0), &[q]),
                2 => c.add_gate(Gate::H, &[q]),
                3 | 4 => {
                    let mut p = rng.random_range(0..n);
                    if p == q {
                        p = (q + 1) % n;
                    }
                    c.add_gate(Gate::Cx, &[q, p]);
                }
                5 => c.add_measure(q, rng.random_range(0..bits)),
                6 => c.add_reset(q),
                _ => c.instructions.push(Instruction::Conditional {
                    condition: ClassicalCondition::BitIsOne(rng.random_range(0..bits)),
                    gate: Gate::Ry(rng.random::<f64>() * 6.0),
                    targets: smallvec![q],
                }),
            }
        }
        assert_trajectory_matches(&c, seed);
    }
}

#[test]
fn dynamic_shots_match_statevector_per_shot() {
    let mut c = crate::circuits::measure_split_circuit(12, 3, 42);
    let first = c.num_classical_bits;
    c.num_classical_bits += 4;
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: smallvec![0],
    });
    for (i, q) in [0usize, 2, 4, 6].into_iter().enumerate() {
        c.add_measure(q, first + i);
    }
    for seed in 42..45 {
        let fac = sim::simulate(&c)
            .backend(crate::BackendKind::Factored)
            .seed(seed)
            .shots(64)
            .unwrap();
        let sv = sim::simulate(&c)
            .backend(crate::BackendKind::Statevector)
            .seed(seed)
            .shots(64)
            .unwrap();
        assert_eq!(fac.metadata.backend, crate::sim::ResolvedBackend::Factored);
        assert_eq!(fac.shots, sv.shots, "seed {seed}");
    }
}

#[test]
fn native_sampler_matches_the_dense_factored_sampler_shot_for_shot() {
    let n = 18;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.3 + 0.07 * q as f64), &[q]);
    }
    for block in [0..6, 6..13, 13..18] {
        for q in block.start..block.end - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
            circuit.add_gate(Gate::Rz(0.41 + 0.05 * q as f64), &[q + 1]);
        }
    }
    let mut fac = FactoredBackend::new(42);
    sim::run_on(&mut fac, &circuit).unwrap();

    let blocks = fac.block_probabilities().expect("factored blocks");
    let meas_map: Vec<(usize, usize)> = (0..n).map(|q| (q, q)).collect();
    for shots in [1_000, 300, 31] {
        let dense = crate::sim::shots::sample_shots(&blocks, &meas_map, n, shots, 42);
        let native = fac.sample_basis_states(shots, 42).unwrap();
        let native = crate::sim::shots::shots_from_basis_samples(&native, &meas_map, n);
        assert_eq!(native, dense, "{shots} shots");
    }
}
