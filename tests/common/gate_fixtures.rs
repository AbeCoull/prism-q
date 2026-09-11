//! Gate-level fixtures for the per-variant golden corpora: an instruction
//! builder and the sample matrices those corpora feed to the matrix gates.

use num_complex::Complex64;
use prism_q::PauliTerm;
use prism_q::circuit::{Circuit, Instruction, SmallVec, smallvec};
use prism_q::gates::Gate;

pub fn g(gate: Gate, targets: &[usize]) -> Instruction {
    let mut tv: SmallVec<[usize; 4]> = smallvec![];
    tv.extend_from_slice(targets);
    Instruction::Gate { gate, targets: tv }
}

pub fn sample_2x2() -> [[Complex64; 2]; 2] {
    [
        [Complex64::new(0.6, -0.1), Complex64::new(-0.3, 0.2)],
        [Complex64::new(0.2, 0.4), Complex64::new(0.7, -0.2)],
    ]
}

pub fn sample_4x4() -> [[Complex64; 4]; 4] {
    let mut mat = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in mat.iter_mut().enumerate() {
        for (col, entry) in row.iter_mut().enumerate() {
            *entry = Complex64::new(0.1 * (r as f64 + 1.0), 0.07 * (col as f64 + 1.0));
        }
    }
    mat
}

pub fn pauli_rot_sample() -> Gate {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_pauli_rotation(0.53, &[PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)]);
    match &circuit.instructions[0] {
        Instruction::Gate { gate, .. } => gate.clone(),
        _ => unreachable!("add_pauli_rotation appends a gate"),
    }
}
