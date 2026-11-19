//! Dense gate matrices the MPS chain builds before a contraction: the phase and
//! swap forms of a two-qubit gate, and the full matrix of a multi-controlled gate.

use num_complex::Complex64;

use super::{ONE, ZERO};

/// Controlled phase followed by SWAP, symmetric in the two qubits since the
/// phase acts on |11⟩ only.
pub(super) fn cu_phase_swap_matrix(phase: Complex64) -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, phase]]
}

pub(super) fn swap_gate_qubits(g: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut out = [[ZERO; 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[j * 2 + i][l * 2 + k] = g[i * 2 + j][k * 2 + l];
                }
            }
        }
    }
    out
}

/// Build a 2^N × 2^N MCU gate matrix (row-major, flattened).
///
/// `qubit_order[i]` maps block-position `i` to the MCU role index:
/// roles 0..num_controls are controls, role num_controls is the target.
/// The matrix is identity except when all controls are |1⟩, where `mat` is
/// applied to the target qubit.
pub(super) fn mcu_matrix(
    num_controls: usize,
    mat: &[[Complex64; 2]; 2],
    qubit_order: &[usize],
) -> Vec<Complex64> {
    let n = num_controls + 1;
    let dim = 1usize << n;
    let mut gate = vec![ZERO; dim * dim];

    // Block position p → bit (n-1-p) in state index (MSB = leftmost).
    let mut ctrl_mask: usize = 0;
    let mut tgt_bit: usize = 0;
    for (pos, &role) in qubit_order.iter().enumerate() {
        let bit = n - 1 - pos;
        if role < num_controls {
            ctrl_mask |= 1 << bit;
        } else {
            tgt_bit = bit;
        }
    }

    for s_in in 0..dim {
        if (s_in & ctrl_mask) == ctrl_mask {
            let t_in = (s_in >> tgt_bit) & 1;
            for (t_out, row) in mat.iter().enumerate() {
                let s_out = (s_in & !(1 << tgt_bit)) | (t_out << tgt_bit);
                gate[s_out * dim + s_in] = row[t_in];
            }
        } else {
            gate[s_in * dim + s_in] = ONE;
        }
    }
    gate
}
