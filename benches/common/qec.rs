//! QEC memory fixtures shared by the `qec_decoder` and `bench_shots_perf` targets.

use prism_q::{QecNoise, QecOptions, QecPauli, QecProgram, QecRecordRef};

// Rotated surface-code plaquettes on a d x d data grid, qubit r*d+c. Corners
// (i, j) of the dual lattice carry a plaquette over the up-to-four adjacent
// data cells; interior corners alternate Z/X by parity, two-cell boundary
// plaquettes survive on the top/bottom rows (Z) and left/right columns (X).
fn rotated_surface_stabilizers(distance: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut z_stabs = Vec::new();
    let mut x_stabs = Vec::new();
    for i in 0..=distance {
        for j in 0..=distance {
            let mut cells = Vec::new();
            for (row, column) in [
                (i.wrapping_sub(1), j.wrapping_sub(1)),
                (i.wrapping_sub(1), j),
                (i, j.wrapping_sub(1)),
                (i, j),
            ] {
                if row < distance && column < distance {
                    cells.push(row * distance + column);
                }
            }
            if cells.len() < 2 {
                continue;
            }
            let z_type = (i + j) % 2 == 1;
            if cells.len() == 2 {
                let keep = if z_type {
                    i == 0 || i == distance
                } else {
                    j == 0 || j == distance
                };
                if !keep {
                    continue;
                }
            }
            if z_type {
                z_stabs.push(cells);
            } else {
                x_stabs.push(cells);
            }
        }
    }
    (z_stabs, x_stabs)
}

/// Distance-d rotated surface-code Z memory: MPP stabilizer rounds, each after
/// `Depolarize1(p)` on every data qubit when `noise` is set, with round-0
/// detectors on the Z checks only, compare detectors on later rounds, final Z
/// checks reconstructed from the data readout, and the left column as the
/// logical observable. Data qubit (r, c) is r * d + c.
pub fn rotated_surface_memory(
    distance: usize,
    rounds: usize,
    noise: Option<f64>,
    options: QecOptions,
) -> QecProgram {
    let num_qubits = distance * distance;
    let (z_stabs, x_stabs) = rotated_surface_stabilizers(distance);
    let data: Vec<usize> = (0..num_qubits).collect();
    let mut program = QecProgram::with_options(num_qubits, options);
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        if let Some(p) = noise {
            program.noise(QecNoise::Depolarize1(p), &data).unwrap();
        }
        let mut records = Vec::with_capacity(z_stabs.len() + x_stabs.len());
        for stab in &z_stabs {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::z(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        for stab in &x_stabs {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::x(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        match &prev {
            None => {
                for &record in &records[..z_stabs.len()] {
                    program.detector(&[QecRecordRef::absolute(record)]).unwrap();
                }
            }
            Some(previous) => {
                for (&record, &prior) in records.iter().zip(previous) {
                    program
                        .detector(&[
                            QecRecordRef::absolute(record),
                            QecRecordRef::absolute(prior),
                        ])
                        .unwrap();
                }
            }
        }
        prev = Some(records);
    }
    let previous = prev.unwrap();
    let readout: Vec<usize> = (0..num_qubits)
        .map(|qubit| program.measure_z(qubit).unwrap())
        .collect();
    for (stab, &prior) in z_stabs.iter().zip(&previous[..z_stabs.len()]) {
        let mut refs: Vec<QecRecordRef> = stab
            .iter()
            .map(|&q| QecRecordRef::absolute(readout[q]))
            .collect();
        refs.push(QecRecordRef::absolute(prior));
        program.detector(&refs).unwrap();
    }
    let logical: Vec<QecRecordRef> = (0..distance)
        .map(|row| QecRecordRef::absolute(readout[row * distance]))
        .collect();
    program.observable_include(0, &logical).unwrap();
    program
}
