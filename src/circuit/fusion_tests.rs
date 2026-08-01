use super::*;
use crate::backend::Backend;
const EPS: f64 = 1e-12;

fn assert_mat_close(actual: &[[Complex64; 2]; 2], expected: &[[Complex64; 2]; 2]) {
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (actual[i][j] - expected[i][j]).norm() < EPS,
                "[{i}][{j}]: expected {:?}, got {:?}",
                expected[i][j],
                actual[i][j]
            );
        }
    }
}

fn identity_mat() -> [[Complex64; 2]; 2] {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    [[one, zero], [zero, one]]
}

fn count_fused_2q(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::Fused2q(_),
                    ..
                }
            )
        })
        .count()
}

#[test]
fn test_mat_mul_identity() {
    let id = identity_mat();
    let h = Gate::H.matrix_2x2();
    assert_mat_close(&mat_mul_2x2(&id, &h), &h);
    assert_mat_close(&mat_mul_2x2(&h, &id), &h);
}

#[test]
fn test_mat_mul_h_h_is_identity() {
    let h = Gate::H.matrix_2x2();
    let result = mat_mul_2x2(&h, &h);
    assert_mat_close(&result, &identity_mat());
}

#[test]
fn test_mat_mul_associative() {
    let a = Gate::Rx(1.0).matrix_2x2();
    let b = Gate::Ry(0.7).matrix_2x2();
    let c = Gate::Rz(2.3).matrix_2x2();
    let ab_c = mat_mul_2x2(&mat_mul_2x2(&a, &b), &c);
    let a_bc = mat_mul_2x2(&a, &mat_mul_2x2(&b, &c));
    assert_mat_close(&ab_c, &a_bc);
}

fn count_fused(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Fused(_),
                    ..
                }
            )
        })
        .count()
}

fn extract_fused_matrix(inst: &Instruction) -> [[Complex64; 2]; 2] {
    match inst {
        Instruction::Gate {
            gate: Gate::Fused(m),
            ..
        } => **m,
        _ => panic!("expected Fused gate"),
    }
}

#[test]
fn test_empty_circuit() {
    let c = Circuit::new(2, 0);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 0);
}

#[test]
fn test_no_fusion_single_gate() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);
    assert_eq!(count_fused(&fused), 0);
    match &fused.instructions[0] {
        Instruction::Gate {
            gate: Gate::H,
            targets,
        } => assert_eq!(targets.as_slice(), &[0]),
        _ => panic!("expected H gate"),
    }
}

#[test]
fn test_no_fusion_different_qubits() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 2);
    assert_eq!(count_fused(&fused), 0);
}

#[test]
fn test_fuse_adjacent_same_qubit() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);
    assert_eq!(count_fused(&fused), 1);
    let expected = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
    let actual = extract_fused_matrix(&fused.instructions[0]);
    assert_mat_close(&actual, &expected);
}

#[test]
fn test_fuse_across_different_qubit() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 2);
    // H·T on q0 stays Fused (no named match), X on q1 recognized as Gate::X
    assert_eq!(count_fused(&fused), 1);

    let expected = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
    let fused_mat = extract_fused_matrix(&fused.instructions[0]);
    assert_mat_close(&fused_mat, &expected);
}

#[test]
fn test_two_qubit_breaks_run() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 3);
    assert_eq!(count_fused(&fused), 0);
}

#[test]
fn test_measure_breaks_run() {
    let mut c = Circuit::new(1, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 3);
    assert_eq!(count_fused(&fused), 0);
}

#[test]
fn test_barrier_breaks_run() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_barrier(&[0]);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 3);
    assert_eq!(count_fused(&fused), 0);
}

#[test]
fn test_multiple_qubits_independent() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Ry(1.0), &[0]);
    c.add_gate(Gate::Rz(2.0), &[0]);
    c.add_gate(Gate::Ry(1.0), &[1]);
    c.add_gate(Gate::Rz(2.0), &[1]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 2);
    assert_eq!(count_fused(&fused), 2);
}

#[test]
fn test_long_chain() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::X, &[0]);
    c.add_gate(Gate::Z, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);
    assert_eq!(count_fused(&fused), 1);
    let expected = mat_mul_2x2(
        &Gate::Z.matrix_2x2(),
        &mat_mul_2x2(
            &Gate::X.matrix_2x2(),
            &mat_mul_2x2(
                &Gate::T.matrix_2x2(),
                &mat_mul_2x2(&Gate::S.matrix_2x2(), &Gate::H.matrix_2x2()),
            ),
        ),
    );
    let actual = extract_fused_matrix(&fused.instructions[0]);
    assert_mat_close(&actual, &expected);
}

#[test]
fn test_gate_count_after_fusion() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::Ry(1.0), &[0]);
    c.add_gate(Gate::Rz(2.0), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[1]);
    c.add_measure(1, 0);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.gate_count(), 3);
    assert_eq!(fused.instructions.len(), 4);
}

#[test]
fn test_fused_probabilities_match_unfused() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Ry(1.23), &[1]);
    c.add_gate(Gate::Rz(0.45), &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.78), &[0]);
    c.add_gate(Gate::Rx(2.34), &[0]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::S, &[2]);
    let mut b1 = StatevectorBackend::new(42);
    b1.init(c.num_qubits, c.num_classical_bits).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let probs_unfused = b1.probabilities().unwrap();
    let fused = fuse_single_qubit_gates(&c);
    let mut b2 = StatevectorBackend::new(42);
    b2.init(fused.num_qubits, fused.num_classical_bits).unwrap();
    for inst in &fused.instructions {
        b2.apply(inst).unwrap();
    }
    let probs_fused = b2.probabilities().unwrap();

    assert_eq!(probs_unfused.len(), probs_fused.len());
    for (i, (a, b)) in probs_unfused.iter().zip(&probs_fused).enumerate() {
        assert!((a - b).abs() < 1e-10, "prob[{i}]: unfused={a}, fused={b}");
    }
}

#[test]
fn test_hea_style_fusion() {
    let mut c = Circuit::new(4, 0);
    for q in 0..4 {
        c.add_gate(Gate::Ry(0.5 + q as f64), &[q]);
        c.add_gate(Gate::Rz(1.0 + q as f64), &[q]);
    }
    for q in 0..3 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..4 {
        c.add_gate(Gate::Ry(2.0 + q as f64), &[q]);
        c.add_gate(Gate::Rz(3.0 + q as f64), &[q]);
    }

    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.gate_count(), 11);
    assert_eq!(count_fused(&fused), 8);
    assert_eq!(c.gate_count(), 19);
}

#[test]
fn test_fusion_matrix_order_matters() {
    let mut c_xh = Circuit::new(1, 0);
    c_xh.add_gate(Gate::X, &[0]);
    c_xh.add_gate(Gate::H, &[0]);

    let mut c_hx = Circuit::new(1, 0);
    c_hx.add_gate(Gate::H, &[0]);
    c_hx.add_gate(Gate::X, &[0]);

    let fused_xh = fuse_single_qubit_gates(&c_xh);
    let fused_hx = fuse_single_qubit_gates(&c_hx);

    let mat_xh = extract_fused_matrix(&fused_xh.instructions[0]);
    let mat_hx = extract_fused_matrix(&fused_hx.instructions[0]);
    let differs = (0..2).any(|i| (0..2).any(|j| (mat_xh[i][j] - mat_hx[i][j]).norm() > EPS));
    assert!(
        differs,
        "X·H and H·X should produce different fused matrices"
    );
    let expected_xh = mat_mul_2x2(&Gate::H.matrix_2x2(), &Gate::X.matrix_2x2());
    assert_mat_close(&mat_xh, &expected_xh);

    let expected_hx = mat_mul_2x2(&Gate::X.matrix_2x2(), &Gate::H.matrix_2x2());
    assert_mat_close(&mat_hx, &expected_hx);
}

#[test]
fn test_s_squared_is_z_via_fusion() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::S, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);
    // S·S recognized as Gate::Z
    assert!(matches!(
        &fused.instructions[0],
        Instruction::Gate { gate: Gate::Z, .. }
    ));
}

#[test]
fn test_t_tdg_cancel_via_fusion() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Tdg, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 0);
}

#[test]
fn test_identity_elision_h_h() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 0);
}

#[test]
fn test_identity_elision_partial() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[1]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);
    // H·H on q0 elided, lone T on q1 recognized as Gate::T
    assert!(matches!(
        &fused.instructions[0],
        Instruction::Gate { gate: Gate::T, .. }
    ));
}

#[test]
fn test_identity_elision_preserves_probabilities() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(0.7), &[1]);
    c.add_gate(Gate::Rz(1.3), &[1]);

    let mut b1 = StatevectorBackend::new(42);
    b1.init(2, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let p1 = b1.probabilities().unwrap();

    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 1);

    let mut b2 = StatevectorBackend::new(42);
    b2.init(2, 0).unwrap();
    for inst in &fused.instructions {
        b2.apply(inst).unwrap();
    }
    let p2 = b2.probabilities().unwrap();

    for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
        assert!((a - b).abs() < 1e-12, "prob[{i}]: unfused={a}, fused={b}");
    }
}

#[test]
fn test_cphase_breaks_fusion_run() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Rz(0.5), &[0]);
    c.add_gate(Gate::cphase(0.3), &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 3);
    assert_eq!(count_fused(&fused), 0);
}

#[test]
fn test_fusion_around_cphase() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::cphase(0.5), &[0, 1]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::X, &[0]);
    let fused = fuse_single_qubit_gates(&c);
    assert_eq!(fused.instructions.len(), 3);
    assert_eq!(count_fused(&fused), 2);
}

#[test]
fn test_cphase_fused_probabilities_match() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;
    use crate::sim;

    let n = 4;
    let mut c = Circuit::new(n, 0);
    for i in 0..n {
        c.add_gate(Gate::H, &[i]);
        for j in (i + 1)..n {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            c.add_gate(Gate::cphase(theta), &[i, j]);
        }
    }
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::S, &[0]);

    let fused_circuit = fuse_single_qubit_gates(&c);

    let mut b1 = StatevectorBackend::new(42);
    sim::run_on(&mut b1, &c).unwrap();
    let p1 = b1.probabilities().unwrap();

    let mut b2 = StatevectorBackend::new(42);
    b2.init(n, 0).unwrap();
    for inst in &fused_circuit.instructions {
        b2.apply(inst).unwrap();
    }
    let p2 = b2.probabilities().unwrap();

    for (i, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
        assert!((*a - *b).abs() < EPS, "prob[{i}]: unfused={a}, fused={b}");
    }
}

#[test]
fn test_cancel_adjacent_cx() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let result = cancel_self_inverse_pairs(&c);
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 0);
}

#[test]
fn test_cancel_cx_with_non_conflicting_between() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]); // doesn't touch q0 or q1
    c.add_gate(Gate::Cx, &[0, 1]);
    let result = cancel_self_inverse_pairs(&c);
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 1); // only H(2) remains
}

#[test]
fn test_no_cancel_cx_with_conflicting_between() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[0]); // touches q0, breaks the chain
    c.add_gate(Gate::Cx, &[0, 1]);
    let result = cancel_self_inverse_pairs(&c);
    // No cancellation, H on q0 intervenes
    assert_eq!(result.instructions.len(), 3);
}

#[test]
fn test_no_cancel_cx_reversed_targets() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 0]); // reversed, CX is NOT symmetric
    let result = cancel_self_inverse_pairs(&c);
    assert_eq!(result.instructions.len(), 2);
}

#[test]
fn test_cancel_cz_symmetric() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::Cz, &[1, 0]); // reversed, CZ IS symmetric
    let result = cancel_self_inverse_pairs(&c);
    assert_eq!(result.instructions.len(), 0);
}

#[test]
fn test_cancel_swap_symmetric() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Swap, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Swap, &[1, 0]); // reversed order
    let result = cancel_self_inverse_pairs(&c);
    assert_eq!(result.instructions.len(), 1); // only H(2)
}

#[test]
fn test_cancel_multiple_pairs() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let result = cancel_self_inverse_pairs(&c);
    assert_eq!(result.instructions.len(), 0);
}

#[test]
fn test_cancel_no_candidates() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[1]);
    let result = cancel_self_inverse_pairs(&c);
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_cancel_preserves_probabilities() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]); // cancels with the first CX
    c.add_gate(Gate::H, &[1]);

    let mut b1 = StatevectorBackend::new(42);
    b1.init(3, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let p1 = b1.probabilities().unwrap();

    let cancelled = cancel_self_inverse_pairs(&c);
    let mut b2 = StatevectorBackend::new(42);
    b2.init(3, 0).unwrap();
    for inst in &cancelled.instructions {
        b2.apply(inst).unwrap();
    }
    let p2 = b2.probabilities().unwrap();

    for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "prob[{i}]: original={a}, cancelled={b}"
        );
    }
}

#[test]
fn test_reorder_1q_basic() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Fused(Box::new(Gate::H.matrix_2x2())), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Fused(Box::new(Gate::T.matrix_2x2())), &[2]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 3);
    assert!(
        result.instructions[0..2].iter().all(|i| matches!(
            i,
            Instruction::Gate {
                gate: Gate::Fused(_),
                ..
            }
        )),
        "first two instructions should be 1q Fused gates"
    );
    let targets: Vec<usize> = result.instructions[0..2]
        .iter()
        .map(|i| match i {
            Instruction::Gate { targets, .. } => targets[0],
            _ => unreachable!(),
        })
        .collect();
    assert!(targets.contains(&0) && targets.contains(&2));
    assert!(matches!(
        &result.instructions[2],
        Instruction::Gate { gate: Gate::Cx, .. }
    ));
}

#[test]
fn test_reorder_no_opportunity() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Fused(Box::new(Gate::H.matrix_2x2())), &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Fused(Box::new(Gate::T.matrix_2x2())), &[1]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_reorder_hea_pattern() {
    let mut c = Circuit::new(4, 0);
    let fused = |angle: f64| Gate::Fused(Box::new(Gate::Ry(angle).matrix_2x2()));
    c.add_gate(fused(0.1), &[0]);
    c.add_gate(fused(0.2), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(fused(0.3), &[2]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(fused(0.4), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert_eq!(result.instructions.len(), 7);
    for i in 0..4 {
        assert!(
            matches!(
                &result.instructions[i],
                Instruction::Gate { gate, .. } if gate.num_qubits() == 1
            ),
            "instruction {i} should be 1q gate"
        );
    }
    for i in 4..7 {
        assert!(
            matches!(
                &result.instructions[i],
                Instruction::Gate { gate: Gate::Cx, .. }
            ),
            "instruction {i} should be CX"
        );
    }
}

#[test]
fn test_reorder_preserves_probabilities() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(1.2), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.8), &[2]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::T, &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::H, &[0]);

    let mut b1 = StatevectorBackend::new(42);
    b1.init(4, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let p1 = b1.probabilities().unwrap();

    let reordered = reorder_1q_gates(Cow::Borrowed(&c));
    let mut b2 = StatevectorBackend::new(42);
    b2.init(4, 0).unwrap();
    for inst in &reordered.instructions {
        b2.apply(inst).unwrap();
    }
    let p2 = b2.probabilities().unwrap();

    for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "prob[{i}]: original={a}, reordered={b}"
        );
    }
}

#[test]
fn test_reorder_respects_barrier() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_barrier(&[0, 1, 2]);
    c.add_gate(Gate::T, &[2]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_reorder_diagonal_commutes_through_cx_control() {
    // Rz(0) should move past CX(0,1) since diagonal commutes on control
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.5), &[0]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 2);
    assert!(matches!(
        &result.instructions[0],
        Instruction::Gate {
            gate: Gate::Rz(_),
            targets
        } if targets[0] == 0
    ));
    assert!(matches!(
        &result.instructions[1],
        Instruction::Gate { gate: Gate::Cx, .. }
    ));
}

#[test]
fn test_reorder_nondiagonal_blocked_by_cx_control() {
    // H(0) should NOT move past CX(0,1), H is not diagonal
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[0]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    // No reorder, H is blocked by CX on q0
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_reorder_diagonal_blocked_on_cx_target() {
    // Rz(1) should NOT move past CX(0,1), q1 is the target
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.5), &[1]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Borrowed(_)));
}

#[test]
fn test_reorder_diagonal_commutes_through_cz() {
    // T(0) and T(1) should both move past CZ(0,1), diagonal commutes
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::S, &[1]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 3);
    // Both diagonal gates should precede CZ
    assert!(
        result.instructions[0..2]
            .iter()
            .all(|i| matches!(i, Instruction::Gate { gate, .. } if gate.num_qubits() == 1))
    );
    assert!(matches!(
        &result.instructions[2],
        Instruction::Gate { gate: Gate::Cz, .. }
    ));
}

#[test]
fn test_reorder_commutation_preserves_probabilities() {
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;

    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rz(0.7), &[0]); // diagonal on CX control, commutes
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::T, &[1]); // diagonal, commutes through CZ
    c.add_gate(Gate::S, &[2]); // diagonal, commutes through CZ

    let mut b1 = StatevectorBackend::new(42);
    b1.init(3, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let p1 = b1.probabilities().unwrap();

    let reordered = reorder_1q_gates(Cow::Borrowed(&c));
    let mut b2 = StatevectorBackend::new(42);
    b2.init(3, 0).unwrap();
    for inst in &reordered.instructions {
        b2.apply(inst).unwrap();
    }
    let p2 = b2.probabilities().unwrap();

    for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "prob[{i}]: original={a}, reordered={b}"
        );
    }
}

#[test]
fn test_reorder_respects_measurement() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::T, &[1]);

    let result = reorder_1q_gates(Cow::Borrowed(&c));
    assert!(matches!(result, Cow::Owned(_)));
    assert_eq!(result.instructions.len(), 3);
    assert!(
        result.instructions[0..2]
            .iter()
            .all(|i| matches!(i, Instruction::Gate { .. })),
        "first two instructions should be 1q gates"
    );
    let targets: Vec<usize> = result.instructions[0..2]
        .iter()
        .map(|i| match i {
            Instruction::Gate { targets, .. } => targets[0],
            _ => unreachable!(),
        })
        .collect();
    assert!(targets.contains(&0) && targets.contains(&1));
    assert!(matches!(
        &result.instructions[2],
        Instruction::Measure { .. }
    ));
}

#[test]
fn test_smart_multi_fusion_across_cx() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::S, &[2]);
    c.add_gate(Gate::Rx(1.0), &[3]);

    let pass1 = fuse_single_qubit_gates(&c);
    let pass2 = fuse_multi_1q_gates(pass1);

    let multi_count = pass2
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::MultiFused(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        multi_count, 1,
        "q2 and q3 gates should accumulate across CX(q0,q1) into one MultiFused"
    );

    if let Instruction::Gate {
        gate: Gate::MultiFused(data),
        ..
    } = &pass2.instructions.last().unwrap()
    {
        let targets: Vec<usize> = data.gates.iter().map(|&(t, _)| t).collect();
        assert!(targets.contains(&2), "q2 should be in the MultiFused batch");
        assert!(targets.contains(&3), "q3 should be in the MultiFused batch");
    } else {
        panic!("last instruction should be MultiFused");
    }

    let mut b1 = crate::backend::statevector::StatevectorBackend::new(42);
    b1.init(c.num_qubits, 0).unwrap();
    for inst in &c.instructions {
        b1.apply(inst).unwrap();
    }
    let probs1 = b1.probabilities().unwrap();

    let mut b2 = crate::backend::statevector::StatevectorBackend::new(42);
    b2.init(pass2.num_qubits, 0).unwrap();
    for inst in &pass2.instructions {
        b2.apply(inst).unwrap();
    }
    let probs2 = b2.probabilities().unwrap();

    for (a, b) in probs1.iter().zip(probs2.iter()) {
        assert!(
            (*a - *b).abs() < 1e-12,
            "probabilities must match: {a} vs {b}"
        );
    }
}

#[test]
fn reorder_exposes_cx_cancellation() {
    // CX q0,q1; T q0; CX q0,q1. T is diagonal on CX control,
    // reorder moves it past CX, exposing CX·CX cancellation.
    let mut c = Circuit::new(10, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);

    let fused = fuse_circuit(&c, true);
    // CX pair cancelled; only T (as Fused) on q0 remains
    let gate_count = fused
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Gate { .. }))
        .count();
    assert_eq!(gate_count, 1, "CX pair should cancel after reorder");
}

#[test]
fn reorder_exposes_1q_fusion() {
    // H q0; CZ q0,q1; T q0. T is diagonal, commutes through CZ on either qubit.
    // After reorder: H q0; T q0; CZ q0,q1. H and T fuse.
    let mut c = Circuit::new(10, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::T, &[0]);

    let fused = fuse_circuit(&c, true);
    let gates: Vec<_> = fused
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::Gate { gate, targets } => Some((gate.clone(), targets.clone())),
            _ => None,
        })
        .collect();
    // Should be: Fused(H·T) on q0, CZ on q0,q1
    assert_eq!(gates.len(), 2, "H and T should fuse after reorder");
    assert!(
        matches!(&gates[0].0, Gate::Fused(_)),
        "first gate should be Fused(H·T)"
    );
    assert!(matches!(&gates[1].0, Gate::Cz), "second gate should be CZ");

    // Verify fused matrix = T · H (T applied after H)
    let t_mat = Gate::T.matrix_2x2();
    let h_mat = Gate::H.matrix_2x2();
    let expected = mat_mul_2x2(&t_mat, &h_mat);
    if let Gate::Fused(mat) = &gates[0].0 {
        assert_mat_close(mat, &expected);
    }
}

#[test]
fn reorder_exposes_cz_cancellation() {
    // CZ q0,q1; S q0; CZ q0,q1. S is diagonal, commutes through CZ.
    // After reorder: S q0; CZ q0,q1; CZ q0,q1 → CZ pair cancels.
    let mut c = Circuit::new(10, 0);
    c.add_gate(Gate::Cz, &[0, 1]);
    c.add_gate(Gate::S, &[0]);
    c.add_gate(Gate::Cz, &[0, 1]);

    let fused = fuse_circuit(&c, true);
    let gate_count = fused
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Gate { .. }))
        .count();
    assert_eq!(gate_count, 1, "CZ pair should cancel after reorder");
}

#[test]
fn same_pair_2q_block_fuses_w_state_pairs() {
    let circuit = crate::circuits::w_state_circuit(20);
    let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
    let before = count_fused_2q(&pass_2q);
    let fused = fuse_same_pair_2q_blocks(pass_2q);
    let after = count_fused_2q(&fused);

    assert!(before >= 2, "w-state should expose paired Fused2q gates");
    assert!(
        after < before,
        "same-pair block fusion should reduce Fused2q count"
    );
}

#[test]
fn same_pair_2q_block_fuses_qv_blocks() {
    let circuit = crate::circuits::quantum_volume_circuit(20, 1, 42);
    let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
    let before = count_fused_2q(&pass_2q);
    let fused = fuse_same_pair_2q_blocks(pass_2q);
    let after = count_fused_2q(&fused);

    assert!(before >= 2, "qv block should expose paired Fused2q gates");
    assert!(
        after < before,
        "same-pair block fusion should reduce Fused2q count"
    );
}

#[test]
fn same_pair_2q_block_accepts_reversed_targets() {
    let mut circuit = Circuit::new(20, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Ry(0.7), &[1]);
    circuit.add_gate(Gate::Cx, &[1, 0]);

    let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
    let before = count_fused_2q(&pass_2q);
    let fused = fuse_same_pair_2q_blocks(pass_2q);
    let after = count_fused_2q(&fused);

    assert_eq!(before, 2);
    assert_eq!(after, 1, "reversed pair order should still fuse");
    assert!(matches!(
        &fused.instructions[0],
        Instruction::Gate {
            gate: Gate::Fused2q(_),
            targets
        } if targets.as_slice() == [0, 1]
    ));
}

#[test]
fn same_pair_2q_block_leaves_diagonal_runs() {
    let mut circuit = Circuit::new(20, 0);
    circuit.add_gate(Gate::Rz(0.3), &[0]);
    circuit.add_gate(Gate::Cz, &[0, 1]);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Cz, &[1, 0]);

    let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
    let before = count_fused_2q(&pass_2q);
    let fused = fuse_same_pair_2q_blocks(pass_2q);
    let after = count_fused_2q(&fused);

    assert_eq!(before, 2);
    assert_eq!(after, 2, "all-diagonal Fused2q runs should stay split");
}

#[test]
fn fuse_1q_returns_borrowed_without_consecutive_pair() {
    let mut c = Circuit::new(12, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rx(0.3), &[0]);
    assert!(matches!(fuse_single_qubit_gates(&c), Cow::Borrowed(_)));
}

#[test]
fn fuse_2q_returns_borrowed_without_absorbable_1q() {
    let mut c = Circuit::new(12, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cz, &[1, 2]);
    assert!(matches!(fuse_2q_gates(Cow::Borrowed(&c)), Cow::Borrowed(_)));
}

#[test]
fn fuse_multi_2q_returns_borrowed_without_tileable_run() {
    let mut c = Circuit::new(20, 0);
    c.add_gate(Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())), &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Fused2q(Box::new(Gate::Cz.matrix_4x4())), &[2, 3]);
    assert!(matches!(
        fuse_multi_2q_gates(Cow::Borrowed(&c)),
        Cow::Borrowed(_)
    ));
}

#[test]
fn fuse_controlled_phases_returns_borrowed_without_batchable_chain() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let phase = Complex64::new(0.0, 1.0);
    let mut c = Circuit::new(20, 0);
    c.add_gate(Gate::cu([[one, zero], [zero, phase]]), &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    assert!(matches!(
        fuse_controlled_phases(Cow::Borrowed(&c)),
        Cow::Borrowed(_)
    ));
}

#[test]
fn qaoa_20q_fuses_to_6_instructions() {
    let circuit = crate::circuits::qaoa_circuit(20, 3, 0xDEAD_BEEF);
    let fused = fuse_circuit(&circuit, true);
    assert_eq!(fused.instructions.len(), 6);

    let batch_rzz_count = fused
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::BatchRzz(_),
                    ..
                }
            )
        })
        .count();
    let multi_fused_count = fused
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::MultiFused(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(batch_rzz_count, 3);
    assert_eq!(multi_fused_count, 3);
}

#[test]
fn qv_12_uses_multi_2q_fusion() {
    let circuit = crate::circuits::quantum_volume_circuit(12, 1, 42);
    let fused = fuse_circuit(&circuit, true);
    let multi_2q_count = fused
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Multi2q(_),
                    ..
                }
            )
        })
        .count();
    assert!(multi_2q_count > 0, "QV 12q should use Multi2q fusion");
}

#[test]
fn test_recognition_extends_clifford_prefix() {
    let mut c = Circuit::new(2, 0);
    // T, T on q0 then CX then Rx on q1
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rx(0.7), &[1]);

    // Without fusion: first gate is T (non-Clifford), so no prefix at all
    assert!(c.clifford_prefix_split().is_none());

    // After fusion: T·T recognized as S (Clifford), so prefix = [S, CX], tail = [Rx]
    let fused = fuse_single_qubit_gates(&c);
    assert!(matches!(
        &fused.instructions[0],
        Instruction::Gate { gate: Gate::S, .. }
    ));
    let split = fused.clifford_prefix_split();
    assert!(split.is_some());
    let (pre_f, tail_f) = split.unwrap();
    assert_eq!(pre_f.instructions.len(), 2); // S + CX
    assert_eq!(tail_f.instructions.len(), 1); // Rx(0.7)
}
