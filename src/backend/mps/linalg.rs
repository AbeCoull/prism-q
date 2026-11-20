//! Dense gate matrices the MPS chain builds before a contraction: the phase and
//! swap forms of a two-qubit gate.

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
