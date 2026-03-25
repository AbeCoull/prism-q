//! Focused benchmark for shot-sampling and counting performance.
//!
//! Measures:
//! 1. `sample_shots` (Dense path) — per-shot bit extraction
//! 2. `ShotsResult::counts()` — histogram building from Vec<Vec<bool>>
//! 3. `PackedShots::counts()` — histogram from packed u64 representation
//! 4. `run_shots_compiled` round-trip — compile + sample + to_shots

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::BackendKind;
use std::time::Duration;

const SEED: u64 = 0xDEAD_BEEF;

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
                b.iter(|| sim::run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
            },
        );
    }

    for &n_shots in &[1_000, 10_000, 100_000] {
        let circuit = ghz_circuit_with_measurements(20);
        group.bench_with_input(
            BenchmarkId::new("ghz_20q", n_shots),
            &n_shots,
            |b, &shots| {
                b.iter(|| sim::run_shots_with(BackendKind::Auto, &circuit, shots, SEED).unwrap());
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
        let result = sim::run_shots_with(BackendKind::Auto, &circuit, n_shots, SEED).unwrap();

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

criterion_group!(
    benches,
    bench_run_shots,
    bench_counts,
    bench_compiled_counts
);
criterion_main!(benches);
