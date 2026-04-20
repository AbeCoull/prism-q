//! GPU dispatch benchmarks for `BackendKind::StatevectorGpu`.
//!
//! Measures the end-to-end dispatch path (fusion plus independent-subsystem
//! decomposition plus crossover plus kernel execution) against the same set
//! of circuit builders the CPU `circuits` bench uses. Covers qubit counts
//! both below and above `GPU_MIN_QUBITS_DEFAULT` so the crossover branch
//! gets coverage alongside the GPU-dispatched branch.
//!
//! Skipped (group prints a message and returns) when no usable GPU is
//! available, so CI on CPU-only runners stays green. Runs with
//! `cargo bench --bench bench_gpu --features "parallel gpu"`.
//!
//! The benchmark file reuses one shared `Arc<GpuContext>`, so NVRTC compile
//! and module-load cost are paid once per process. The dispatched groups
//! time end-to-end simulation, including per-run state allocation. The
//! direct-kernel group excludes backend init and scratch-buffer growth from
//! the timed region so it better tracks kernel and launcher cost.

#![cfg(feature = "gpu")]

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::gpu::GpuContext;
use prism_q::{sim, BackendKind, StatevectorBackend};

const SEED: u64 = 0xDEAD_BEEF;

fn is_fast() -> bool {
    cfg!(feature = "bench-fast")
}

fn configure_group(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    if is_fast() {
        group.sample_size(10);
        group.warm_up_time(Duration::from_millis(200));
        group.measurement_time(Duration::from_secs(1));
    } else {
        group.sample_size(10);
    }
}

fn shared_ctx() -> Option<Arc<GpuContext>> {
    static CTX: OnceLock<Option<Arc<GpuContext>>> = OnceLock::new();
    CTX.get_or_init(|| match GpuContext::new(0) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("SKIP: no usable GPU ({e})");
            None
        }
    })
    .clone()
}

fn gpu_kind(ctx: &Arc<GpuContext>) -> BackendKind {
    BackendKind::StatevectorGpu {
        context: ctx.clone(),
    }
}

fn sweep_sizes() -> &'static [usize] {
    if is_fast() {
        // Quick: one below crossover, one above. Keeps the fast-mode smoke run short.
        &[10, 16]
    } else {
        // 10 exercises the CPU-fallback branch, 14 is at threshold, 16/18/20/22 are
        // the GPU-dispatched regime. 22 is the largest comfortably within 1 GB of VRAM
        // at `Complex64` amplitudes.
        &[10, 14, 16, 18, 20, 22]
    }
}

fn bench_gpu_random(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/random_d10");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n in sweep_sizes() {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gpu_qft(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/qft_textbook");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n in sweep_sizes() {
        let circuit = circuits::qft_circuit(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gpu_hea(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/hea_l5");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n in sweep_sizes() {
        let circuit = circuits::hardware_efficient_ansatz(n, 5, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_gpu_qaoa(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/qaoa_l3");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n in sweep_sizes() {
        let circuit = circuits::qaoa_circuit(n, 3, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

/// Independent Bell pairs at 16 qubits. Exercises the decomposition-aware
/// dispatch path: eight independent 2q sub-blocks, each below the crossover
/// and each routed to CPU through `run_decomposed`. Protects against future
/// regressions in the decomposition recursion.
fn bench_gpu_decomposed(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/bell_pairs");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n_pairs in &[4usize, 8, 16] {
        let circuit = circuits::independent_bell_pairs(n_pairs);
        let n = circuit.num_qubits;
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

/// Direct-kernel path via `StatevectorBackend::new(seed).with_gpu(ctx)`.
/// Bypasses `sim::run_with`, so fusion, decomposition, and crossover do not
/// run. Excludes backend init and first-use scratch allocation from the timed
/// region so it tracks kernel and launcher cost more directly than the
/// dispatched groups above.
fn bench_gpu_direct_kernel(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/direct/random_d10");
    configure_group(&mut group);

    for &n in sweep_sizes() {
        let circuit = circuits::random_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut backend = StatevectorBackend::new(42).with_gpu(ctx.clone());
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    let _ = backend.probabilities().unwrap();
                    let start = Instant::now();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    let _ = backend.probabilities().unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

/// GHZ-seeded circuit with mid-circuit measure + reset cycles. Exercises
/// `measure_prob_one` and `apply_measure_gpu` repeatedly inside a single
/// simulation. Future measurement-path reductions in `measure_prob_one`
/// should show up here; the dispatched groups above only hit the
/// measurement path once via `probabilities` at the end.
fn mid_measure_circuit(n_qubits: usize, rounds: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, n_qubits);
    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits.saturating_sub(1) {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for round in 0..rounds {
        let parity = round % 2;
        for q in (parity..n_qubits).step_by(2) {
            circuit.add_measure(q, q);
            circuit.add_reset(q);
            circuit.add_gate(Gate::H, &[q]);
            if q + 1 < n_qubits {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

fn bench_gpu_measurement(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu/mid_measure");
    configure_group(&mut group);
    let kind = gpu_kind(&ctx);

    for &n in sweep_sizes() {
        let circuit = mid_measure_circuit(n, 4);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_random,
    bench_gpu_qft,
    bench_gpu_hea,
    bench_gpu_qaoa,
    bench_gpu_decomposed,
    bench_gpu_direct_kernel,
    bench_gpu_measurement,
);
criterion_main!(benches);
