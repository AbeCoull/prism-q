//! Bulk union-find decode over sampled detector batches: repetition and
//! rotated-surface memories at distances 3 and 5.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use prism_q::{
    QecNoise, QecOptions, QecPauli, QecProgram, QecRecordRef, UnionFindDecoder, run_qec_program,
};
use std::hint::black_box;

mod common;
use common::SEED;

const NOISE_RATE: f64 = 0.02;
const SHOT_COUNTS: [usize; 2] = [1_000, 20_000];

fn bench_options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(4096),
        keep_measurements: false,
    }
}

// Distance-d repetition-code Z memory matching the shape of the decoder test
// fixture: per-round data noise and MPP ZZ checks, compare detectors, final
// readout detectors, observable on data qubit 0.
fn repetition_memory(distance: usize, rounds: usize, shots: usize) -> QecProgram {
    let data: Vec<usize> = (0..distance).collect();
    let mut program = QecProgram::with_options(distance, bench_options(shots));
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program
            .noise(QecNoise::Depolarize1(NOISE_RATE), &data)
            .unwrap();
        let records: Vec<usize> = (0..distance - 1)
            .map(|check| {
                program
                    .measure_pauli_product(&[QecPauli::z(check), QecPauli::z(check + 1)])
                    .unwrap()
            })
            .collect();
        match &prev {
            None => {
                for &record in &records {
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
    let readout: Vec<usize> = (0..distance)
        .map(|qubit| program.measure_z(qubit).unwrap())
        .collect();
    for (check, &prior) in previous.iter().enumerate() {
        program
            .detector(&[
                QecRecordRef::absolute(readout[check]),
                QecRecordRef::absolute(readout[check + 1]),
                QecRecordRef::absolute(prior),
            ])
            .unwrap();
    }
    program
        .observable_include(0, &[QecRecordRef::absolute(readout[0])])
        .unwrap();
    program
}

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

// Distance-d rotated surface-code Z memory: MPP stabilizer rounds with
// round-0 detectors on the Z checks only, compare detectors on later rounds,
// final Z checks reconstructed from the data readout, observable on the left
// column logical Z.
fn rotated_surface_memory(distance: usize, rounds: usize, shots: usize) -> QecProgram {
    let num_qubits = distance * distance;
    let (z_stabs, x_stabs) = rotated_surface_stabilizers(distance);
    let data: Vec<usize> = (0..num_qubits).collect();
    let mut program = QecProgram::with_options(num_qubits, bench_options(shots));
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program
            .noise(QecNoise::Depolarize1(NOISE_RATE), &data)
            .unwrap();
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

fn bench_qec_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_decoder");
    common::configure_group(&mut group);
    for shots in SHOT_COUNTS {
        let fixtures = [
            ("rep_d3_r3", repetition_memory(3, 3, shots)),
            ("rep_d5_r5", repetition_memory(5, 5, shots)),
            ("surface_d3_r3", rotated_surface_memory(3, 3, shots)),
            ("surface_d5_r5", rotated_surface_memory(5, 5, shots)),
        ];
        for (label, program) in fixtures {
            let model = program
                .detector_error_model()
                .unwrap()
                .decompose_graphlike()
                .unwrap();
            let decoder = UnionFindDecoder::from_model(&model).unwrap();
            let detectors = run_qec_program(&program).unwrap().detectors;
            group.bench_with_input(BenchmarkId::new(label, shots), &detectors, |b, input| {
                b.iter(|| black_box(decoder.decode_packed(black_box(input)).unwrap()));
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_qec_decoder
}
criterion_main!(benches);
