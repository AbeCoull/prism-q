//! Bulk decode over sampled detector batches and detector error model derivation:
//! union-find on repetition and rotated-surface memories at distances 3 and 5,
//! matching against union-find on rotated-surface memories up to distance 13, and
//! BP+OSD on a hypergraph color-code memory.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prism_q::{
    BpOsdDecoder, MatchingDecoder, QecNoise, QecOptions, QecPauli, QecProgram, QecRecordRef,
    UnionFindDecoder, run_qec_program,
};
use std::hint::black_box;

mod common;
use common::SEED;

const NOISE_RATE: f64 = 0.02;
const SHOT_COUNTS: [usize; 2] = [1_000, 20_000];
const COMPARE_SHOTS: usize = 4096;

fn bench_options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(4096),
        keep_measurements: false,
        ..QecOptions::default()
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

fn noisy_surface_memory(distance: usize, rounds: usize, p: f64, shots: usize) -> QecProgram {
    common::qec::rotated_surface_memory(distance, rounds, Some(p), bench_options(shots))
}

fn bench_qec_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_decoder");
    common::configure_group(&mut group);
    for shots in SHOT_COUNTS {
        let fixtures = [
            ("rep_d3_r3", repetition_memory(3, 3, shots)),
            ("rep_d5_r5", repetition_memory(5, 5, shots)),
            (
                "surface_d3_r3",
                noisy_surface_memory(3, 3, NOISE_RATE, shots),
            ),
            (
                "surface_d5_r5",
                noisy_surface_memory(5, 5, NOISE_RATE, shots),
            ),
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

// Distance-3 triangular color-code Z memory (the Steane code): X and Z checks on
// each plaquette, observable on a weight-3 logical Z. A single X error on the
// center qubit flips three Z plaquettes, so the model is a hypergraph.
fn color_code_memory_d3(rounds: usize, p: f64, shots: usize) -> QecProgram {
    const PLAQUETTES: [&[usize]; 3] = [&[0, 2, 4, 6], &[1, 2, 5, 6], &[3, 4, 5, 6]];
    let data: Vec<usize> = (0..7).collect();
    let mut program = QecProgram::with_options(7, bench_options(shots));
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program.noise(QecNoise::Depolarize1(p), &data).unwrap();
        let mut records = Vec::with_capacity(2 * PLAQUETTES.len());
        for stab in PLAQUETTES {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::z(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        for stab in PLAQUETTES {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::x(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        match &prev {
            None => {
                for &record in &records[..PLAQUETTES.len()] {
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
    let readout: Vec<usize> = (0..7).map(|q| program.measure_z(q).unwrap()).collect();
    for (stab, &prior) in PLAQUETTES.iter().zip(&previous) {
        let mut refs: Vec<QecRecordRef> = stab
            .iter()
            .map(|&q| QecRecordRef::absolute(readout[q]))
            .collect();
        refs.push(QecRecordRef::absolute(prior));
        program.detector(&refs).unwrap();
    }
    let logical: Vec<QecRecordRef> = [0usize, 1, 2]
        .iter()
        .map(|&q| QecRecordRef::absolute(readout[q]))
        .collect();
    program.observable_include(0, &logical).unwrap();
    program
}

// Per-shot decode cost of exact matching against union-find on the same
// samples, d rounds at distance d; and BP+OSD on a hypergraph model.
fn bench_qec_decoder_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_decoder_compare");
    common::configure_group(&mut group);
    group.throughput(Throughput::Elements(COMPARE_SHOTS as u64));
    for distance in [5, 9, 13] {
        for (tag, p) in [("p1e-3", 0.001), ("p5e-3", 0.005)] {
            let program = noisy_surface_memory(distance, distance, p, COMPARE_SHOTS);
            let model = program
                .detector_error_model()
                .unwrap()
                .decompose_graphlike()
                .unwrap();
            let detectors = run_qec_program(&program).unwrap().detectors;
            let label = format!("surface_d{distance}_r{distance}_{tag}");
            let union_find = UnionFindDecoder::from_model(&model).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("uf/{label}"), COMPARE_SHOTS),
                &detectors,
                |b, input| {
                    b.iter(|| black_box(union_find.decode_packed(black_box(input)).unwrap()))
                },
            );
            let matching = MatchingDecoder::from_model(&model).unwrap();
            group.bench_with_input(
                BenchmarkId::new(format!("mwpm/{label}"), COMPARE_SHOTS),
                &detectors,
                |b, input| b.iter(|| black_box(matching.decode_packed(black_box(input)).unwrap())),
            );
        }
    }
    for (tag, p) in [("p1e-3", 0.001), ("p1e-2", 0.01)] {
        let program = color_code_memory_d3(3, p, COMPARE_SHOTS);
        let model = program.detector_error_model().unwrap();
        let detectors = run_qec_program(&program).unwrap().detectors;
        let bposd = BpOsdDecoder::from_model(&model).unwrap();
        group.bench_with_input(
            BenchmarkId::new(format!("bposd/color_d3_r3_{tag}"), COMPARE_SHOTS),
            &detectors,
            |b, input| b.iter(|| black_box(bposd.decode_packed(black_box(input)).unwrap())),
        );
    }
    group.finish();
}

// Model derivation alone: the walk over the deferred circuit and the symptom merge.
fn bench_qec_dem(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_dem");
    common::configure_group(&mut group);
    let fixtures = [
        ("rep_d5_r5", repetition_memory(5, 5, 1)),
        ("surface_d3_r3", noisy_surface_memory(3, 3, NOISE_RATE, 1)),
        ("surface_d5_r5", noisy_surface_memory(5, 5, NOISE_RATE, 1)),
        ("surface_d5_r25", noisy_surface_memory(5, 25, NOISE_RATE, 1)),
    ];
    for (label, program) in fixtures {
        group.bench_with_input(BenchmarkId::new(label, 0), &program, |b, program| {
            b.iter(|| black_box(program.detector_error_model().unwrap()));
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets = bench_qec_decoder, bench_qec_decoder_compare, bench_qec_dem
}
criterion_main!(benches);
