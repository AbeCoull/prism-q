//! Focused benchmark for shot-sampling and counting performance.
//!
//! Measures:
//! 1. `sample_shots` (Dense path): per-shot bit extraction
//! 2. `ShotsResult::counts()`: histogram building from Vec<Vec<bool>>
//! 3. `PackedShots::counts()`: histogram from packed u64 representation
//! 4. `run_shots_compiled` round-trip: compile + sample + to_shots

#[cfg(feature = "bench-internal")]
use criterion::BatchSize;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::sim::noise::NoiseModel;
use prism_q::BackendKind;
#[cfg(feature = "bench-internal")]
use prism_q::{compile_qec_profiled_sampler, parse_qec_program};
use prism_q::{
    run_qec_program, HomologicalSampler, QecNoise, QecOptions, QecPauli, QecProgram, QecRecordRef,
};
use std::collections::HashMap;
#[cfg(feature = "bench-internal")]
use std::hint::black_box;
use std::time::Duration;

const SEED: u64 = 0xDEAD_BEEF;
const API_QUERY_SHOTS: usize = 100_000;

fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> prism_q::Result<prism_q::ShotsResult> {
    sim::simulate(circuit)
        .backend(kind)
        .seed(seed)
        .shots(num_shots)
}

fn run_counts_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> prism_q::Result<HashMap<Vec<u64>, u64>> {
    Ok(sim::simulate(circuit)
        .backend(kind)
        .seed(seed)
        .sample_counts(num_shots)?
        .into_counts())
}

fn bell_circuit_with_measurements(n_qubits: usize) -> Circuit {
    let mut c = Circuit::new(n_qubits, n_qubits);
    for q in (0..n_qubits).step_by(2) {
        if q + 1 < n_qubits {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for q in 0..n_qubits {
        c.add_measure(q, q);
    }
    c
}

fn ghz_circuit_with_measurements(n_qubits: usize) -> Circuit {
    let mut c = Circuit::new(n_qubits, n_qubits);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n_qubits - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n_qubits {
        c.add_measure(q, q);
    }
    c
}

fn non_clifford_terminal_circuit(n_qubits: usize, measured_qubits: usize) -> Circuit {
    let mut c = Circuit::new(n_qubits, n_qubits);
    for q in 0..n_qubits {
        c.add_gate(Gate::Ry(0.17 + q as f64 * 0.013), &[q]);
        c.add_gate(Gate::Rz(0.29 + q as f64 * 0.017), &[q]);
    }
    for q in 0..n_qubits - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n_qubits {
        c.add_gate(Gate::Rx(0.11 + q as f64 * 0.019), &[q]);
    }
    for q in 0..measured_qubits.min(n_qubits) {
        c.add_measure(q, q);
    }
    c
}

fn api_redesign_circuit(n_qubits: usize, with_measurements: bool) -> Circuit {
    let mut c = Circuit::new(n_qubits, if with_measurements { n_qubits } else { 0 });
    for q in 0..n_qubits {
        c.add_gate(Gate::Ry(0.13 + q as f64 * 0.011), &[q]);
        c.add_gate(Gate::Rz(0.19 + q as f64 * 0.017), &[q]);
    }
    for q in 0..n_qubits - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n_qubits {
        c.add_gate(Gate::Rx(0.07 + q as f64 * 0.023), &[q]);
    }
    if with_measurements {
        for q in 0..n_qubits {
            c.add_measure(q, q);
        }
    }
    c
}

fn clifford_t_marginal_circuit(n_qubits: usize) -> Circuit {
    let mut c = Circuit::new(n_qubits, 0);
    for q in 0..n_qubits {
        c.add_gate(Gate::H, &[q]);
        if q % 3 == 0 {
            c.add_gate(Gate::T, &[q]);
        }
    }
    for q in 0..n_qubits - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in (1..n_qubits).step_by(4) {
        c.add_gate(Gate::Tdg, &[q]);
    }
    c
}

fn bench_api_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("api_queries");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let run_circuit = api_redesign_circuit(12, false);
    group.bench_function("run_auto_12q", |b| {
        b.iter(|| sim::simulate(&run_circuit).seed(SEED).run().unwrap());
    });

    let shots_circuit = api_redesign_circuit(12, true);
    group.bench_function("shots_terminal_12q_100000", |b| {
        b.iter(|| {
            sim::simulate(&shots_circuit)
                .seed(SEED)
                .shots(API_QUERY_SHOTS)
                .unwrap()
        });
    });

    group.bench_function("sample_counts_terminal_12q_100000", |b| {
        b.iter(|| {
            sim::simulate(&shots_circuit)
                .seed(SEED)
                .sample_counts(API_QUERY_SHOTS)
                .unwrap()
        });
    });

    let marginal_circuit = clifford_t_marginal_circuit(14);
    group.bench_function("marginals_auto_clifford_t_14q", |b| {
        b.iter(|| {
            sim::simulate(&marginal_circuit)
                .seed(SEED)
                .marginals()
                .unwrap()
        });
    });

    group.finish();
}

fn bench_run_shots(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_shots");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_shots in &[1_000, 10_000, 100_000, 1_000_000] {
        let circuit = bell_circuit_with_measurements(16);
        group.bench_with_input(
            BenchmarkId::new("bell_16q", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    for &n_shots in &[1_000, 10_000, 100_000] {
        let circuit = ghz_circuit_with_measurements(20);
        group.bench_with_input(
            BenchmarkId::new("ghz_20q", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    for &n_shots in &[1_000, 10_000, 100_000] {
        let circuit = non_clifford_terminal_circuit(18, 18);
        group.bench_with_input(
            BenchmarkId::new("nonclifford_full_18q", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    for &n_shots in &[1_000, 10_000, 100_000] {
        let circuit = non_clifford_terminal_circuit(20, 8);
        group.bench_with_input(
            BenchmarkId::new("nonclifford_subset_20q_8m", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("shots_counts");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_shots in &[10_000, 100_000, 1_000_000] {
        let circuit = bell_circuit_with_measurements(16);
        let result = run_shots_with(BackendKind::Auto, &circuit, n_shots, SEED).unwrap();

        group.bench_with_input(
            BenchmarkId::new("ShotsResult_counts_16q", n_shots),
            &result,
            |b, res| {
                b.iter(|| res.counts());
            },
        );
    }

    group.finish();
}

fn bench_run_counts_terminal(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_counts_terminal");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_shots in &[10_000, 100_000] {
        let circuit = non_clifford_terminal_circuit(18, 18);
        group.bench_with_input(
            BenchmarkId::new("nonclifford_full_18q", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_counts_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    for &n_shots in &[10_000, 100_000] {
        let circuit = non_clifford_terminal_circuit(20, 8);
        group.bench_with_input(
            BenchmarkId::new("nonclifford_subset_20q_8m", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| run_counts_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_compiled_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("compiled_counts");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_shots in &[10_000, 100_000, 1_000_000] {
        let circuit = ghz_circuit_with_measurements(20);
        let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
        let packed = sampler.sample_bulk_packed(n_shots);

        group.bench_with_input(
            BenchmarkId::new("PackedShots_counts_20q", n_shots),
            &packed,
            |b, ps| {
                b.iter(|| ps.counts());
            },
        );
    }

    group.finish();
}

fn bench_histogram_counts(c: &mut Criterion) {
    use prism_q::HistogramAccumulator;
    let mut group = c.benchmark_group("histogram_counts");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_qubits in &[20, 100, 1000] {
        let circuit = ghz_circuit_with_measurements(n_qubits);
        for &n_shots in &[10_000, 100_000, 1_000_000] {
            let label = format!("ghz_{n_qubits}q");
            group.bench_with_input(BenchmarkId::new(&label, n_shots), &n_shots, |b, &shots| {
                let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
                b.iter(|| {
                    let mut acc = HistogramAccumulator::new();
                    sampler.sample_chunked(shots, &mut acc);
                    acc.into_counts()
                });
            });
        }
    }

    for &n_qubits in &[20, 100] {
        let circuit = clifford_circuit_with_measurements(n_qubits, 10);
        for &n_shots in &[10_000, 100_000, 1_000_000] {
            let label = format!("clifford_d10_{n_qubits}q");
            group.bench_with_input(BenchmarkId::new(&label, n_shots), &n_shots, |b, &shots| {
                let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
                b.iter(|| {
                    let mut acc = HistogramAccumulator::new();
                    sampler.sample_chunked(shots, &mut acc);
                    acc.into_counts()
                });
            });
        }
    }

    group.finish();
}

fn bench_rank_space_counts(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_space_counts");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    for &n_qubits in &[20, 100, 1000] {
        let circuit = ghz_circuit_with_measurements(n_qubits);
        for &n_shots in &[10_000, 100_000, 1_000_000] {
            let label = format!("ghz_{n_qubits}q");
            group.bench_with_input(BenchmarkId::new(&label, n_shots), &n_shots, |b, &shots| {
                let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
                b.iter(|| sampler.sample_counts(shots));
            });
        }
    }

    for &n_qubits in &[20, 100] {
        let circuit = clifford_circuit_with_measurements(n_qubits, 10);
        for &n_shots in &[10_000, 100_000, 1_000_000] {
            let label = format!("clifford_d10_{n_qubits}q");
            group.bench_with_input(BenchmarkId::new(&label, n_shots), &n_shots, |b, &shots| {
                let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
                b.iter(|| sampler.sample_counts(shots));
            });
        }
    }

    group.finish();
}

fn clifford_circuit_with_measurements(n_qubits: usize, depth: usize) -> Circuit {
    let mut c = prism_q::circuits::clifford_heavy_circuit(n_qubits, depth, SEED);
    c.num_classical_bits = n_qubits;
    for q in 0..n_qubits {
        c.add_measure(q, q);
    }
    c
}

fn qec_repetition_program(
    n_data: usize,
    rounds: usize,
    shots: usize,
    noise_rate: Option<f64>,
) -> QecProgram {
    qec_repetition_program_with_postselection(n_data, rounds, shots, noise_rate, false)
}

fn qec_repetition_program_with_postselection(
    n_data: usize,
    rounds: usize,
    shots: usize,
    noise_rate: Option<f64>,
    include_postselection: bool,
) -> QecProgram {
    assert!(n_data > 0);
    let options = QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(10_000),
        keep_measurements: false,
    };
    let mut program = QecProgram::with_options(n_data, options);
    let checks = n_data.saturating_sub(1);
    let mut previous_round = Vec::with_capacity(checks);
    let mut first_check = None;

    for _ in 0..rounds {
        let mut current_round = Vec::with_capacity(checks);
        for q in 0..checks {
            if let Some(p) = noise_rate {
                program
                    .noise(QecNoise::Depolarize1(p), &[q, q + 1])
                    .unwrap();
            }
            let record = program
                .measure_pauli_product(&[QecPauli::z(q), QecPauli::z(q + 1)])
                .unwrap();
            first_check.get_or_insert(record);
            if let Some(previous) = previous_round.get(q) {
                program
                    .detector(&[
                        QecRecordRef::absolute(*previous),
                        QecRecordRef::absolute(record),
                    ])
                    .unwrap();
            }
            current_round.push(record);
        }
        previous_round = current_round;
    }

    let logical = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(logical)])
        .unwrap();
    if include_postselection {
        if let Some(first_check) = first_check {
            program
                .postselect(&[QecRecordRef::absolute(first_check)], false)
                .unwrap();
        }
    }
    program
}

#[cfg(feature = "bench-internal")]
fn qec_repetition_program_text(
    n_data: usize,
    rounds: usize,
    noise_rate: Option<f64>,
    include_postselection: bool,
) -> String {
    assert!(n_data > 0);
    let checks = n_data.saturating_sub(1);
    let mut text = String::new();
    let mut previous_round = Vec::with_capacity(checks);
    let mut first_check = None;
    let mut records = 0usize;

    for _ in 0..rounds {
        let mut current_round = Vec::with_capacity(checks);
        for q in 0..checks {
            if let Some(p) = noise_rate {
                text.push_str(&format!("DEPOLARIZE1({p}) {q} {}\n", q + 1));
            }
            text.push_str(&format!("MPP Z{q}*Z{}\n", q + 1));
            let record = records;
            records += 1;
            first_check.get_or_insert(record);
            if let Some(previous) = previous_round.get(q) {
                let previous_distance = records - previous;
                text.push_str(&format!("DETECTOR rec[-{previous_distance}] rec[-1]\n"));
            }
            current_round.push(record);
        }
        previous_round = current_round;
    }

    text.push_str("M 0\n");
    records += 1;
    text.push_str("OBSERVABLE_INCLUDE(0) rec[-1]\n");
    if include_postselection {
        if let Some(first_check) = first_check {
            let first_distance = records - first_check;
            text.push_str(&format!("POSTSELECT rec[-{first_distance}]\n"));
        }
    }
    text
}

fn bench_qec_clifford_runner(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_clifford_runner");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &(n_data, rounds, shots) in &[(25, 5, 10_000), (25, 5, 100_000), (100, 3, 100_000)] {
        let program = qec_repetition_program(n_data, rounds, shots, None);
        let label = format!("{n_data}q_r{rounds}");
        group.bench_with_input(BenchmarkId::new(&label, shots), &program, |b, program| {
            b.iter(|| run_qec_program(program).unwrap());
        });
    }

    let zero_noise_program = qec_repetition_program(25, 5, 100_000, Some(0.0));
    group.bench_with_input(
        BenchmarkId::new("25q_r5_p0", 100_000),
        &zero_noise_program,
        |b, program| {
            b.iter(|| run_qec_program(program).unwrap());
        },
    );

    group.finish();
}

fn bench_qec_noisy_runner(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_noisy_runner");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &(n_data, rounds, shots) in &[(25, 5, 10_000), (25, 5, 100_000)] {
        let program = qec_repetition_program(n_data, rounds, shots, Some(0.001));
        let label = format!("{n_data}q_r{rounds}_p001");
        group.bench_with_input(BenchmarkId::new(&label, shots), &program, |b, program| {
            b.iter(|| run_qec_program(program).unwrap());
        });
    }

    group.finish();
}

#[cfg(not(feature = "bench-internal"))]
fn bench_qec_noisy_runner_split(_c: &mut Criterion) {}

#[cfg(feature = "bench-internal")]
fn bench_qec_noisy_runner_split(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_noisy_runner_split");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    let n_data = 25;
    let rounds = 5;
    let shots = 100_000;
    let noise_rate = 0.001;
    let label = format!("{n_data}q_r{rounds}_p001_post");
    let text = qec_repetition_program_text(n_data, rounds, Some(noise_rate), true);
    let program =
        qec_repetition_program_with_postselection(n_data, rounds, shots, Some(noise_rate), true);

    group.bench_with_input(BenchmarkId::new("parse", &label), &text, |b, text| {
        b.iter(|| {
            let mut program = parse_qec_program(black_box(text)).unwrap();
            let mut options = program.options();
            options.shots = shots;
            options.seed = SEED;
            options.chunk_size = Some(10_000);
            options.keep_measurements = false;
            program.set_options(options);
            black_box(program.num_measurements());
        });
    });

    group.bench_with_input(
        BenchmarkId::new("compile", &label),
        &program,
        |b, program| {
            b.iter(|| compile_qec_profiled_sampler(black_box(program)).unwrap());
        },
    );

    let mut sample_sampler = compile_qec_profiled_sampler(&program).unwrap();
    group.bench_with_input(BenchmarkId::new("sample", &label), &shots, |b, &shots| {
        b.iter(|| {
            sample_sampler
                .sample_noiseless_measurements_packed(black_box(shots))
                .unwrap()
        });
    });

    let mut noise_sampler = compile_qec_profiled_sampler(&program).unwrap();
    let noiseless = noise_sampler
        .sample_noiseless_measurements_packed(shots)
        .unwrap();
    group.bench_with_input(
        BenchmarkId::new("noise_generation", &label),
        &noiseless,
        |b, measurements| {
            b.iter_batched(
                || measurements.clone(),
                |measurements| {
                    noise_sampler
                        .apply_noise_to_measurements(measurements)
                        .unwrap()
                },
                BatchSize::LargeInput,
            );
        },
    );

    let mut staged_sampler = compile_qec_profiled_sampler(&program).unwrap();
    let noisy_measurements = staged_sampler.sample_measurements_packed(shots).unwrap();
    let observables = staged_sampler
        .observable_records(&noisy_measurements)
        .unwrap();

    group.bench_with_input(
        BenchmarkId::new("detector_projection", &label),
        &noisy_measurements,
        |b, measurements| {
            b.iter(|| {
                staged_sampler
                    .detector_records(black_box(measurements))
                    .unwrap()
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("postselection_logical_count", &label),
        &noisy_measurements,
        |b, measurements| {
            b.iter(|| {
                staged_sampler
                    .postselection_and_logical_counts(black_box(measurements), &observables)
                    .unwrap()
            });
        },
    );

    group.bench_with_input(BenchmarkId::new("total", &label), &program, |b, program| {
        b.iter(|| run_qec_program(black_box(program)).unwrap());
    });

    group.finish();
}

fn bench_homological_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("homological_compile");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &n in &[6, 10, 16, 20] {
        let circuit = ghz_circuit_with_measurements(n);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        if HomologicalSampler::compile(&circuit, &noise, SEED).is_ok() {
            group.bench_with_input(BenchmarkId::new("ghz", n), &n, |b, _| {
                b.iter(|| HomologicalSampler::compile(&circuit, &noise, SEED).unwrap());
            });
        }
    }

    for &n in &[10, 20, 30] {
        let circuit = clifford_circuit_with_measurements(n, 5);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        if HomologicalSampler::compile(&circuit, &noise, SEED).is_ok() {
            group.bench_with_input(BenchmarkId::new("clifford_d5", n), &n, |b, _| {
                b.iter(|| HomologicalSampler::compile(&circuit, &noise, SEED).unwrap());
            });
        }
    }

    group.finish();
}

fn bench_homological_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("homological_sample");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &n_shots in &[1_000, 10_000, 100_000, 1_000_000] {
        let circuit = ghz_circuit_with_measurements(20);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

        {
            let mut sampler = HomologicalSampler::compile(&circuit, &noise, SEED).unwrap();
            group.bench_with_input(
                BenchmarkId::new("ghz_20q_unpacked", n_shots),
                &n_shots,
                |b, &shots| {
                    b.iter(|| sampler.sample_bulk(shots));
                },
            );
        }

        {
            let mut sampler = HomologicalSampler::compile(&circuit, &noise, SEED).unwrap();
            group.bench_with_input(
                BenchmarkId::new("ghz_20q_packed", n_shots),
                &n_shots,
                |b, &shots| {
                    b.iter(|| sampler.sample_packed(shots));
                },
            );
        }

        {
            let mut sampler = HomologicalSampler::compile(&circuit, &noise, SEED).unwrap();
            group.bench_with_input(
                BenchmarkId::new("ghz_20q_marginals", n_shots),
                &n_shots,
                |b, &shots| {
                    b.iter(|| sampler.sample_marginals(shots));
                },
            );
        }
    }

    for &n_shots in &[1_000, 10_000, 100_000, 1_000_000] {
        let circuit = bell_circuit_with_measurements(16);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        if let Ok(mut sampler) = HomologicalSampler::compile(&circuit, &noise, SEED) {
            group.bench_with_input(
                BenchmarkId::new("bell_16q_packed", n_shots),
                &n_shots,
                |b, &shots| {
                    b.iter(|| sampler.sample_packed(shots));
                },
            );
        }
    }

    group.finish();
}

fn sparse_active_circuit(n_qubits: usize, n_active: usize) -> Circuit {
    let mut active = prism_q::circuits::clifford_heavy_circuit(n_active, 10, SEED);
    active.num_qubits = n_qubits;
    active.num_classical_bits = n_qubits;
    for q in 0..n_qubits {
        active.add_measure(q, q);
    }
    active
}

fn bench_sparse_deterministic(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_deterministic");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &(n_qubits, n_active) in &[(100, 10), (200, 20), (500, 50), (1000, 100)] {
        let circuit = sparse_active_circuit(n_qubits, n_active);
        let det_pct = 100 * (n_qubits - n_active) / n_qubits;
        for &n_shots in &[10_000, 100_000, 1_000_000] {
            let label = format!("{n_qubits}q_{n_active}active_{det_pct}pct_det");
            group.bench_with_input(BenchmarkId::new(&label, n_shots), &n_shots, |b, &shots| {
                let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
                b.iter(|| sampler.sample_counts(shots));
            });
        }
    }

    group.finish();
}

fn bench_analytical_marginals(c: &mut Criterion) {
    let mut group = c.benchmark_group("analytical_marginals");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(3));

    for &n in &[20, 50, 100, 200, 500, 1000] {
        let circuit = ghz_circuit_with_measurements(n);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        group.bench_with_input(BenchmarkId::new("ghz", n), &n, |b, _| {
            b.iter(|| prism_q::noisy_marginals_analytical(&circuit, &noise, SEED).unwrap());
        });
    }

    for &n in &[20, 50, 100, 200, 500, 1000] {
        let circuit = clifford_circuit_with_measurements(n, 2);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        group.bench_with_input(BenchmarkId::new("clifford_d2", n), &n, |b, _| {
            b.iter(|| prism_q::noisy_marginals_analytical(&circuit, &noise, SEED).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_api_queries,
    bench_run_shots,
    bench_counts,
    bench_run_counts_terminal,
    bench_compiled_counts,
    bench_histogram_counts,
    bench_rank_space_counts,
    bench_sparse_deterministic,
    bench_qec_clifford_runner,
    bench_qec_noisy_runner,
    bench_qec_noisy_runner_split,
    bench_homological_compile,
    bench_homological_sample,
    bench_analytical_marginals,
);
criterion_main!(benches);
