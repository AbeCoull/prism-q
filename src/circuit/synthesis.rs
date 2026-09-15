//! Lowering a dense unitary onto the gate variants the backends carry.
//!
//! One and two qubit matrices have their own variants. A wider matrix, or one
//! under a control, is reduced to multi-controlled gates instead of growing the
//! [`Gate`] enum, whose size the instruction stream pays for on every circuit.

use num_complex::Complex64;
use smallvec::SmallVec;

use super::Instruction;
use crate::gates::Gate;
use crate::gates::spectral::unitary_eigen;

/// Below this an entry is read as zero, a phase as absent, or a two-level
/// block as the identity. Set at the rounding a product of a few dozen
/// rotations carries, well above `f64::EPSILON` and well below any angle a
/// program means.
const NEGLIGIBLE: f64 = 1e-12;

/// Instructions applying `matrix` to `targets`.
///
/// `matrix` is row major over `2^k` with `targets[0]` the most significant bit
/// of both indices, the packing [`Gate::matrix_4x4`] uses. One and two qubit
/// matrices become a single gate; wider ones go through
/// [`two_level_sequence`].
pub(crate) fn dense_unitary(matrix: &[Complex64], targets: &[usize]) -> Vec<Instruction> {
    match targets.len() {
        1 => vec![instruction(
            Gate::Fused(Box::new([[matrix[0], matrix[1]], [matrix[2], matrix[3]]])),
            targets,
        )],
        2 => {
            let mut block = [[Complex64::new(0.0, 0.0); 4]; 4];
            for (row, entries) in block.iter_mut().enumerate() {
                entries.copy_from_slice(&matrix[row * 4..row * 4 + 4]);
            }
            vec![instruction(Gate::Fused2q(Box::new(block)), targets)]
        }
        _ => two_level_sequence(matrix, targets),
    }
}

/// Instructions applying `matrix` to `targets` when every control is `|1>`.
///
/// Uses `ctrl(V D V*) = V ctrl(D) V*`, which holds because the conjugating
/// factors cancel wherever the control is low. That keeps the cost independent
/// of the control count, where controlling the whole matrix as one wider
/// unitary would square the work with each control added.
pub(crate) fn controlled_dense(
    matrix: &[Complex64],
    controls: &[usize],
    targets: &[usize],
) -> Vec<Instruction> {
    debug_assert!(!controls.is_empty(), "a controlled gate needs a control");
    let dim = 1usize << targets.len();
    let (values, vectors) = unitary_eigen(&transpose(matrix, dim), dim);
    let basis = from_columns(&vectors, dim);
    let mut out = dense_unitary(&adjoint(&basis, dim), targets);
    out.extend(controlled_diagonal(&values, controls, targets));
    out.extend(dense_unitary(&basis, targets));
    out
}

/// Widest span an expansion may cover before a matrix power over it is
/// declined. At four qubits the reduction emits a few hundred gates; each
/// qubit past that quadruples both the matrix and the gate count.
pub(crate) const MAX_COMPOSED_QUBITS: usize = 4;

/// The unitary `instrs` implement, over the qubits they name sorted ascending.
///
/// `None` when an instruction is not a gate, carries no dense matrix, or the
/// span is wider than [`MAX_COMPOSED_QUBITS`]. Row major with the first qubit
/// of the span as the most significant bit, the packing [`dense_unitary`]
/// reads back.
pub(crate) fn expansion_matrix(instrs: &[Instruction]) -> Option<(Vec<usize>, Vec<Complex64>)> {
    let mut span: Vec<usize> = Vec::new();
    for instr in instrs {
        let Instruction::Gate { targets, .. } = instr else {
            return None;
        };
        for &qubit in targets {
            if !span.contains(&qubit) {
                span.push(qubit);
            }
        }
    }
    span.sort_unstable();
    if span.is_empty() || span.len() > MAX_COMPOSED_QUBITS {
        return None;
    }

    let dim = 1usize << span.len();
    let mut composed = vec![Complex64::new(0.0, 0.0); dim * dim];
    for index in 0..dim {
        composed[index * dim + index] = Complex64::new(1.0, 0.0);
    }
    for instr in instrs {
        let Instruction::Gate { gate, targets } = instr else {
            return None;
        };
        let block = gate.dense_matrix()?;
        let lifted = embed(&block, targets, &span);
        composed = multiply(&lifted, &composed, dim);
    }
    Some((span, composed))
}

/// Lift a matrix over `targets` onto the wider `span`, the identity elsewhere.
fn embed(block: &[Complex64], targets: &[usize], span: &[usize]) -> Vec<Complex64> {
    let width = span.len();
    let dim = 1usize << width;
    let inner = targets.len();
    let positions: Vec<usize> = targets
        .iter()
        .map(|target| span.iter().position(|q| q == target).expect("in span"))
        .collect();
    let named = positions
        .iter()
        .fold(0usize, |mask, &at| mask | 1 << (width - 1 - at));
    let local = |index: usize| {
        positions
            .iter()
            .enumerate()
            .fold(0usize, |acc, (bit, &at)| {
                acc | (index >> (width - 1 - at) & 1) << (inner - 1 - bit)
            })
    };
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for column in 0..dim {
            if row & !named != column & !named {
                continue;
            }
            out[row * dim + column] = block[local(row) * (1 << inner) + local(column)];
        }
    }
    out
}

fn multiply(left: &[Complex64], right: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for inner in 0..dim {
            let scale = left[row * dim + inner];
            if scale.norm() <= NEGLIGIBLE {
                continue;
            }
            for column in 0..dim {
                out[row * dim + column] += scale * right[inner * dim + column];
            }
        }
    }
    out
}

/// A diagonal on `targets` fired only when every control is `|1>`.
///
/// Each basis state of `targets` contributes one multi-controlled phase, the
/// targets joining the control list with the ones reading zero conjugated by
/// `X` so that every control fires on `|1>`.
fn controlled_diagonal(
    values: &[Complex64],
    controls: &[usize],
    targets: &[usize],
) -> Vec<Instruction> {
    let width = targets.len();
    let mut out = Vec::new();
    for (state, value) in values.iter().enumerate() {
        let angle = value.arg();
        if (value - Complex64::new(1.0, 0.0)).norm() <= NEGLIGIBLE {
            continue;
        }
        let low: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter(|(index, _)| state >> (width - 1 - index) & 1 == 0)
            .map(|(_, &qubit)| qubit)
            .collect();
        for &qubit in &low {
            out.push(instruction(Gate::X, &[qubit]));
        }
        let mut wires: Vec<usize> = controls.to_vec();
        wires.extend_from_slice(&targets[..width - 1]);
        let num_controls = wires.len();
        wires.push(targets[width - 1]);
        out.push(instruction(
            Gate::mcu(Gate::P(angle).matrix_2x2(), num_controls as u8),
            &wires,
        ));
        for &qubit in &low {
            out.push(instruction(Gate::X, &[qubit]));
        }
    }
    out
}

/// Reduce `matrix` to a diagonal by two-level rotations between basis states
/// that differ in one bit, then emit the reverse.
///
/// Rows are visited in Gray-code order, where neighbours differ in exactly one
/// bit, so every rotation is a multi-controlled single-qubit gate rather than
/// a permutation needing its own ladder. Costs `O(4^k)` rotations, which suits
/// the pragma and observable widths that reach it rather than a hot path.
fn two_level_sequence(matrix: &[Complex64], targets: &[usize]) -> Vec<Instruction> {
    let width = targets.len();
    let dim = 1usize << width;
    let mut work = matrix.to_vec();
    let order: Vec<usize> = (0..dim).map(|index| index ^ (index >> 1)).collect();

    let mut rotations: Vec<(usize, usize, [[Complex64; 2]; 2])> = Vec::new();
    for column in 0..dim - 1 {
        let pivot = order[column];
        for position in (column + 1..dim).rev() {
            let (upper, lower) = (order[position - 1], order[position]);
            let above = work[upper * dim + pivot];
            let below = work[lower * dim + pivot];
            if below.norm() <= NEGLIGIBLE {
                continue;
            }
            let scale = (above.norm_sqr() + below.norm_sqr()).sqrt();
            let block = [
                [above.conj() / scale, below.conj() / scale],
                [-below / scale, above / scale],
            ];
            for index in 0..dim {
                let left = work[upper * dim + index];
                let right = work[lower * dim + index];
                work[upper * dim + index] = block[0][0] * left + block[0][1] * right;
                work[lower * dim + index] = block[1][0] * left + block[1][1] * right;
            }
            rotations.push((upper, lower, block));
        }
    }

    let diagonal: Vec<Complex64> = (0..dim).map(|index| work[index * dim + index]).collect();
    let mut out = diagonal_phases(&diagonal, targets);
    for (upper, lower, block) in rotations.into_iter().rev() {
        let adjoint = [
            [block[0][0].conj(), block[1][0].conj()],
            [block[0][1].conj(), block[1][1].conj()],
        ];
        out.extend(two_level_instruction(upper, lower, &adjoint, targets));
    }
    out
}

/// A rotation between two basis states differing in one bit, as a
/// multi-controlled single-qubit gate on that bit.
fn two_level_instruction(
    upper: usize,
    lower: usize,
    block: &[[Complex64; 2]; 2],
    targets: &[usize],
) -> Vec<Instruction> {
    let width = targets.len();
    let differing = upper ^ lower;
    debug_assert!(
        differing.count_ones() == 1,
        "gray-code neighbours differ in one bit"
    );
    let position = differing.trailing_zeros() as usize;
    let moved = targets[width - 1 - position];
    // `upper` is the state the block indexes first, so the matrix is written
    // in target-bit order only when `upper` reads zero on the moved bit.
    let oriented = if upper >> position & 1 == 0 {
        *block
    } else {
        [[block[1][1], block[1][0]], [block[0][1], block[0][0]]]
    };
    if (oriented[0][0] - Complex64::new(1.0, 0.0)).norm() <= NEGLIGIBLE
        && (oriented[1][1] - Complex64::new(1.0, 0.0)).norm() <= NEGLIGIBLE
        && oriented[0][1].norm() <= NEGLIGIBLE
        && oriented[1][0].norm() <= NEGLIGIBLE
    {
        return Vec::new();
    }

    let mut held = Vec::with_capacity(width - 1);
    let mut low = Vec::new();
    for (index, &qubit) in targets.iter().enumerate() {
        let bit = width - 1 - index;
        if bit == position {
            continue;
        }
        held.push(qubit);
        if upper >> bit & 1 == 0 {
            low.push(qubit);
        }
    }

    let mut out = Vec::new();
    for &qubit in &low {
        out.push(instruction(Gate::X, &[qubit]));
    }
    let mut wires = held.clone();
    wires.push(moved);
    out.push(instruction(Gate::mcu(oriented, held.len() as u8), &wires));
    for &qubit in &low {
        out.push(instruction(Gate::X, &[qubit]));
    }
    out
}

/// A diagonal on `targets`, as one multi-controlled phase per basis state.
fn diagonal_phases(values: &[Complex64], targets: &[usize]) -> Vec<Instruction> {
    let width = targets.len();
    let mut out = Vec::new();
    for (state, value) in values.iter().enumerate() {
        if (value - Complex64::new(1.0, 0.0)).norm() <= NEGLIGIBLE {
            continue;
        }
        let low: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter(|(index, _)| state >> (width - 1 - index) & 1 == 0)
            .map(|(_, &qubit)| qubit)
            .collect();
        for &qubit in &low {
            out.push(instruction(Gate::X, &[qubit]));
        }
        out.push(instruction(
            Gate::mcu(Gate::P(value.arg()).matrix_2x2(), (width - 1) as u8),
            targets,
        ));
        for &qubit in &low {
            out.push(instruction(Gate::X, &[qubit]));
        }
    }
    out
}

fn instruction(gate: Gate, targets: &[usize]) -> Instruction {
    Instruction::Gate {
        gate,
        targets: SmallVec::from_slice(targets),
    }
}

fn transpose(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for column in 0..dim {
            out[column * dim + row] = matrix[row * dim + column];
        }
    }
    out
}

/// Row-major matrix whose columns are the column-major `vectors`.
fn from_columns(vectors: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for column in 0..dim {
        for row in 0..dim {
            out[row * dim + column] = vectors[column * dim + row];
        }
    }
    out
}

fn adjoint(matrix: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        for column in 0..dim {
            out[row * dim + column] = matrix[column * dim + row].conj();
        }
    }
    out
}

#[cfg(test)]
#[path = "synthesis_tests.rs"]
mod tests;
