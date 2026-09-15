use super::*;

use crate::circuit::Circuit;
use crate::simulate;

const EPS: f64 = 1e-9;

/// The unitary `instrs` implement over `targets`, in gate-matrix order where
/// `targets[0]` is the most significant bit.
///
/// Each column is read by running the instructions from the basis state that
/// column names, so the answer comes from the simulator rather than from a
/// second copy of the synthesis rules.
fn realized(instrs: &[Instruction], num_qubits: usize, targets: &[usize]) -> Vec<Complex64> {
    let width = targets.len();
    let dim = 1usize << width;
    let place = |state: usize| -> usize {
        targets.iter().enumerate().fold(0usize, |index, (at, &q)| {
            index | (state >> (width - 1 - at) & 1) << q
        })
    };
    let circuit = Circuit {
        num_qubits,
        num_classical_bits: 0,
        instructions: instrs.to_vec(),
    };
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for column in 0..dim {
        let mut start = vec![Complex64::new(0.0, 0.0); 1usize << num_qubits];
        start[place(column)] = Complex64::new(1.0, 0.0);
        let state = simulate(&circuit)
            .seed(42)
            .initial_state(&start)
            .state_vector()
            .expect("a unitary circuit");
        for row in 0..dim {
            out[row * dim + column] = state[place(row)];
        }
        // Nothing may leak onto a qubit the gate does not name.
        let named = targets.iter().fold(0usize, |mask, &q| mask | 1 << q);
        for (index, amplitude) in state.iter().enumerate() {
            if index & !named != 0 {
                assert!(amplitude.norm() < EPS, "leaked onto index {index}");
            }
        }
    }
    out
}

fn assert_matrix(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: wrong size");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).norm() < EPS,
            "{label}: entry {index} is {got} against {want}"
        );
    }
}

/// A deterministic unitary of the given width, built as a product of rotations
/// so that every entry is populated and no two are equal.
fn sample_unitary(width: usize, seed: u64) -> Vec<Complex64> {
    let dim = 1usize << width;
    let mut matrix = vec![Complex64::new(0.0, 0.0); dim * dim];
    for index in 0..dim {
        matrix[index * dim + index] = Complex64::new(1.0, 0.0);
    }
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for upper in 0..dim {
        for lower in upper + 1..dim {
            let theta = next() * std::f64::consts::PI;
            let phi = next() * std::f64::consts::TAU;
            let (cos, sin) = (theta.cos(), theta.sin());
            let phase = Complex64::from_polar(1.0, phi);
            for column in 0..dim {
                let a = matrix[upper * dim + column];
                let b = matrix[lower * dim + column];
                matrix[upper * dim + column] = a * cos - b * sin * phase;
                matrix[lower * dim + column] = a * sin * phase.conj() + b * cos;
            }
        }
    }
    matrix
}

// A one or two qubit matrix keeps its own gate variant, so the check here is
// that the packing survives: `targets[0]` stays the high bit.
#[test]
fn a_narrow_matrix_becomes_one_gate() {
    for width in [1usize, 2] {
        let matrix = sample_unitary(width, 7 + width as u64);
        let targets: Vec<usize> = (0..width).collect();
        let instrs = dense_unitary(&matrix, &targets);
        assert_eq!(instrs.len(), 1, "width {width} should be a single gate");
        assert_matrix(
            &realized(&instrs, width, &targets),
            &matrix,
            &format!("dense width {width}"),
        );
    }
}

// A wider matrix has no variant to carry it and goes through the two-level
// reduction. The reduction has to reproduce the matrix entry for entry,
// global phase included, since a `state_vector` result reports it.
#[test]
fn a_wide_matrix_is_reduced_to_multi_controlled_gates() {
    for width in [3usize, 4] {
        let matrix = sample_unitary(width, 11 + width as u64);
        let targets: Vec<usize> = (0..width).collect();
        let instrs = dense_unitary(&matrix, &targets);
        assert!(
            instrs.len() > 1,
            "width {width} should need more than one gate"
        );
        assert_matrix(
            &realized(&instrs, width, &targets),
            &matrix,
            &format!("two level width {width}"),
        );
    }
}

// The target order is part of the packing, so naming the same qubits in
// another order has to give the permuted unitary rather than the same one.
#[test]
fn a_wide_matrix_follows_its_target_order() {
    let matrix = sample_unitary(3, 23);
    let forward = realized(&dense_unitary(&matrix, &[0, 1, 2]), 3, &[0, 1, 2]);
    let shuffled = realized(&dense_unitary(&matrix, &[2, 0, 1]), 3, &[2, 0, 1]);
    assert_matrix(&forward, &matrix, "forward");
    assert_matrix(&shuffled, &matrix, "shuffled");
}

// A diagonal is the degenerate case of the reduction: every two-level block is
// the identity and only the phases survive.
#[test]
fn a_wide_diagonal_reduces_to_phases_alone() {
    let dim = 8;
    let mut matrix = vec![Complex64::new(0.0, 0.0); dim * dim];
    for index in 0..dim {
        matrix[index * dim + index] = Complex64::from_polar(1.0, 0.3 * (index as f64 + 1.0));
    }
    let instrs = dense_unitary(&matrix, &[0, 1, 2]);
    assert!(
        instrs.iter().all(|instr| !matches!(
            instr,
            Instruction::Gate {
                gate: Gate::Fused2q(_),
                ..
            }
        )),
        "a diagonal needs no two-qubit block"
    );
    assert_matrix(&realized(&instrs, 3, &[0, 1, 2]), &matrix, "diagonal");
}

// `ctrl(U)` leaves the low half alone and applies `U` to the high half,
// whatever the width of `U` and however many controls there are.
#[test]
fn a_controlled_matrix_acts_only_under_its_controls() {
    for (width, controls) in [(1usize, 1usize), (2, 1), (2, 2), (3, 1)] {
        let matrix = sample_unitary(width, 31 + width as u64 + controls as u64);
        let control_wires: Vec<usize> = (0..controls).collect();
        let targets: Vec<usize> = (controls..controls + width).collect();
        let instrs = controlled_dense(&matrix, &control_wires, &targets);

        let total = controls + width;
        let mut wires = control_wires.clone();
        wires.extend_from_slice(&targets);
        let whole = realized(&instrs, total, &wires);

        let dim = 1usize << width;
        let side = 1usize << total;
        let fires = side - dim;
        for row in 0..side {
            for column in 0..side {
                let (row_on, column_on) = (row >= fires, column >= fires);
                let expected = if row_on && column_on {
                    matrix[(row - fires) * dim + (column - fires)]
                } else if row == column {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (whole[row * side + column] - expected).norm() < EPS,
                    "width {width}, {controls} control(s): ({row}, {column}) is {} against {expected}",
                    whole[row * side + column]
                );
            }
        }
    }
}
