//! GPU benchmarks for the statevector and stabilizer GPU paths.
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

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_q::backend::Backend;
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::gates::Gate;
use prism_q::gpu::GpuContext;
use prism_q::{sim, BackendKind, StabilizerBackend, StatevectorBackend};

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

// ============================================================================
// Stabilizer GPU
// ============================================================================

fn stab_sweep_sizes() -> &'static [usize] {
    if is_fast() {
        // One below threshold, one above, to show crossover flipping.
        &[256, 1024]
    } else if std::env::var("PRISM_STAB_HIGH_N").is_ok() {
        // Temporary sweep for crossover exploration at very large tableaus.
        // CPU tableau exceeds L3 by ~10x at n=10000 and ~80x at n=20000, so
        // this is where any bandwidth-bound GPU advantage would surface.
        &[10000, 20000, 30000, 50000]
    } else {
        // Covers the crossover zone (~256-1024) plus the at-scale regimes
        // where GPU rowmul should clearly dominate.
        &[100, 500, 1000, 2000, 5000]
    }
}

fn stabilizer_dispatch_enabled(max_qubits: usize) -> bool {
    let threshold = prism_q::gpu::stabilizer_min_qubits();
    if threshold <= max_qubits {
        return true;
    }
    eprintln!(
        "SKIP: explicit StabilizerGpu dispatch benches require PRISM_STABILIZER_GPU_MIN_QUBITS<={max_qubits}, got {threshold}"
    );
    false
}

fn bts_gpu_shot_sizes() -> Vec<usize> {
    let threshold = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT);
    if is_fast() {
        vec![threshold.max(64)]
    } else {
        let mut sizes = vec![threshold.max(64), 1_000_000];
        sizes.sort_unstable();
        sizes.dedup();
        sizes
    }
}

/// Stabilizer CPU dispatch baseline through `sim::run_with`.
///
/// Includes the default probability extraction at the end of the run, so this
/// is an API-level end-to-end measurement rather than the hot-path throughput
/// number used for crossover tuning.
fn bench_stab_cpu_clifford_d10(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_stab_cpu/clifford_d10");
    configure_group(&mut group);
    for &n in stab_sweep_sizes() {
        let circuit = prism_q::circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(BackendKind::Stabilizer, circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

/// Stabilizer GPU dispatch through `run_with_stabilizer_gpu`.
///
/// Includes the default probability extraction at the end of the run. Use the
/// direct backend groups below for hot-path throughput comparisons.
fn bench_stab_gpu_clifford_d10(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    if !stabilizer_dispatch_enabled(stab_sweep_sizes().iter().copied().max().unwrap_or(0)) {
        return;
    }
    let mut group = c.benchmark_group("gpu_stab/clifford_d10");
    configure_group(&mut group);
    let kind = BackendKind::StabilizerGpu {
        context: ctx.clone(),
    };
    for &n in stab_sweep_sizes() {
        let circuit = prism_q::circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

/// Direct backend CPU timing for the stabilizer hot path.
///
/// Excludes the `run_with` probability readback and performs one untimed warm
/// pass plus a re-init before each timed iteration so backend scratch growth is
/// not charged to the measured apply.
fn bench_stab_direct_cpu_clifford_d10(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_stab_direct_cpu/clifford_d10");
    configure_group(&mut group);
    for &n in stab_sweep_sizes() {
        let circuit = prism_q::circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut backend = StabilizerBackend::new(42);
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    let start = Instant::now();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

/// Direct backend GPU timing for the stabilizer hot path.
///
/// Uses `StabilizerBackend::with_gpu(ctx)` directly so the timed region tracks
/// queued Clifford application and measurement kernels without the end-of-run
/// probability readback.
fn bench_stab_direct_gpu_clifford_d10(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let mut group = c.benchmark_group("gpu_stab_direct/clifford_d10");
    configure_group(&mut group);
    for &n in stab_sweep_sizes() {
        let circuit = prism_q::circuits::clifford_heavy_circuit(n, 10, SEED);
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let mut backend = StabilizerBackend::new(42).with_gpu(ctx.clone());
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    let start = Instant::now();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    total += start.elapsed();
                }
                total
            });
        });
    }
    group.finish();
}

/// GHZ-measure stress test: one H + chained CX over n qubits, then measure
/// every qubit. The group skips unless the process was launched with a low
/// enough `PRISM_STABILIZER_GPU_MIN_QUBITS` value to reach the GPU path.
fn bench_stab_gpu_ghz_measure(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };
    let sizes: &[usize] = if is_fast() {
        &[512]
    } else {
        &[200, 1000, 2000]
    };
    if !stabilizer_dispatch_enabled(sizes.iter().copied().max().unwrap_or(0)) {
        return;
    }
    let mut group = c.benchmark_group("gpu_stab/ghz_measure");
    configure_group(&mut group);
    let kind = BackendKind::StabilizerGpu {
        context: ctx.clone(),
    };
    for &n in sizes {
        let mut circuit = Circuit::new(n, n);
        circuit.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                sim::run_with(kind.clone(), circ, 42).unwrap();
            });
        });
    }
    group.finish();
}

fn bench_stab_bts_cpu_vs_gpu(c: &mut Criterion) {
    use prism_q::{run_shots_compiled, run_shots_compiled_with_gpu};
    let Some(ctx) = shared_ctx() else { return };

    let qubit_sizes: &[usize] = if is_fast() { &[512] } else { &[512, 2048] };
    let shot_sizes = bts_gpu_shot_sizes();

    for &n in qubit_sizes {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
        }
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for q in 0..n {
            circuit.add_measure(q, q);
        }

        let mut cpu_group = c.benchmark_group(format!("gpu_bts_cpu/n{}", n));
        configure_group(&mut cpu_group);
        for &shots in &shot_sizes {
            cpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter(|| {
                    run_shots_compiled(circ, shots, SEED).unwrap();
                });
            });
        }
        cpu_group.finish();

        let mut gpu_group = c.benchmark_group(format!("gpu_bts/n{}", n));
        configure_group(&mut gpu_group);
        for &shots in &shot_sizes {
            let ctx_clone = ctx.clone();
            gpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter(|| {
                    run_shots_compiled_with_gpu(circ, shots, SEED, ctx_clone.clone()).unwrap();
                });
            });
        }
        gpu_group.finish();
    }
}

/// Packed-shot sampling through `CompiledSampler::sample_bulk_packed`.
/// Excludes compilation from the timed region so the group isolates the
/// CPU and GPU sampling kernels plus packed-result materialization.
fn bench_stab_bts_packed_cpu_vs_gpu(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };

    let qubit_sizes: &[usize] = if is_fast() { &[512] } else { &[512, 2048] };
    let shot_sizes = bts_gpu_shot_sizes();

    for &n in qubit_sizes {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
        }
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for q in 0..n {
            circuit.add_measure(q, q);
        }

        let mut cpu_group = c.benchmark_group(format!("gpu_bts_packed_cpu/n{}", n));
        configure_group(&mut cpu_group);
        for &shots in &shot_sizes {
            cpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED).unwrap();
                        let start = Instant::now();
                        black_box(sampler.sample_bulk_packed(shots));
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        cpu_group.finish();

        let mut gpu_group = c.benchmark_group(format!("gpu_bts_packed/n{}", n));
        configure_group(&mut gpu_group);
        for &shots in &shot_sizes {
            let ctx_clone = ctx.clone();
            gpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED)
                            .unwrap()
                            .with_gpu(ctx_clone.clone());
                        let start = Instant::now();
                        black_box(sampler.sample_bulk_packed(shots));
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        gpu_group.finish();
    }
}

fn bts_low_rank_wide_circuit(num_qubits: usize, rank: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits, num_qubits);
    for q in 0..rank {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..num_qubits {
        circuit.add_measure(q, q);
    }
    circuit
}

/// Marginal extraction through `CompiledSampler::sample_marginals`.
/// The GPU path now keeps packed shots on device and copies back only one
/// counter per measurement row.
fn bench_stab_bts_marginals_cpu_vs_gpu(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };

    let qubit_sizes: &[usize] = if is_fast() { &[512] } else { &[512, 2048] };
    let shot_sizes = bts_gpu_shot_sizes();

    for &n in qubit_sizes {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
        }
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for q in 0..n {
            circuit.add_measure(q, q);
        }

        let mut cpu_group = c.benchmark_group(format!("gpu_bts_marginals_cpu/n{}", n));
        configure_group(&mut cpu_group);
        for &shots in &shot_sizes {
            cpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED).unwrap();
                        let start = Instant::now();
                        black_box(sampler.sample_marginals(shots));
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        cpu_group.finish();

        let mut gpu_group = c.benchmark_group(format!("gpu_bts_marginals/n{}", n));
        configure_group(&mut gpu_group);
        for &shots in &shot_sizes {
            let ctx_clone = ctx.clone();
            gpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED)
                            .unwrap()
                            .with_gpu(ctx_clone.clone());
                        let start = Instant::now();
                        black_box(sampler.sample_marginals(shots));
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        gpu_group.finish();
    }
}

/// Explicit device counts over a low-rank, wide-measurement workload.
/// This exercises `sample_bulk_packed_device().counts()` where the compacted
/// `(bitstring, count)` output is far smaller than the full packed shot matrix.
fn bench_stab_bts_device_counts_explicit(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };

    let threshold = std::env::var("PRISM_GPU_BTS_MIN_SHOTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(prism_q::gpu::BTS_MIN_SHOTS_DEFAULT);
    let shot_sizes: Vec<usize> = if is_fast() {
        vec![threshold.max(64)]
    } else {
        vec![threshold.max(64), 1_000_000]
    };
    let qubit_sizes: &[(usize, usize)] = if is_fast() {
        &[(512, 12)]
    } else {
        &[(512, 12), (1024, 12)]
    };

    for &(n, rank) in qubit_sizes {
        let circuit = bts_low_rank_wide_circuit(n, rank);

        let mut cpu_group =
            c.benchmark_group(format!("gpu_bts_device_counts_cpu/n{}_r{}", n, rank));
        configure_group(&mut cpu_group);
        for &shots in &shot_sizes {
            cpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED).unwrap();
                        let start = Instant::now();
                        black_box(sampler.sample_bulk_packed(shots).counts());
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        cpu_group.finish();

        let mut gpu_group = c.benchmark_group(format!("gpu_bts_device_counts/n{}_r{}", n, rank));
        configure_group(&mut gpu_group);
        for &shots in &shot_sizes {
            let ctx_clone = ctx.clone();
            gpu_group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let mut sampler = prism_q::compile_measurements(circ, SEED)
                            .unwrap()
                            .with_gpu(ctx_clone.clone());
                        let start = Instant::now();
                        black_box(
                            sampler
                                .sample_bulk_packed_device(shots)
                                .unwrap()
                                .counts()
                                .unwrap(),
                        );
                        total += start.elapsed();
                    }
                    total
                });
            });
        }
        gpu_group.finish();
    }
}

/// Explicit `BackendKind::StabilizerGpu` shot sampling. This covers the
/// `run_shots_with` dispatch path that now forwards Clifford measurement
/// workloads into the compiled sampler with the attached GPU context.
fn bench_stab_gpu_shots_explicit(c: &mut Criterion) {
    let Some(ctx) = shared_ctx() else { return };

    let qubit_sizes: &[usize] = if is_fast() { &[512] } else { &[512, 2048] };
    let shot_sizes = bts_gpu_shot_sizes();

    for &n in qubit_sizes {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
        }
        for i in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for q in 0..n {
            circuit.add_measure(q, q);
        }

        let mut group = c.benchmark_group(format!("gpu_stab_shots/n{}", n));
        configure_group(&mut group);
        let kind = BackendKind::StabilizerGpu {
            context: ctx.clone(),
        };
        for &shots in &shot_sizes {
            group.bench_with_input(BenchmarkId::from_parameter(shots), &circuit, |b, circ| {
                b.iter(|| {
                    sim::run_shots_with(kind.clone(), circ, shots, SEED).unwrap();
                });
            });
        }
        group.finish();
    }
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
    bench_stab_cpu_clifford_d10,
    bench_stab_gpu_clifford_d10,
    bench_stab_direct_cpu_clifford_d10,
    bench_stab_direct_gpu_clifford_d10,
    bench_stab_gpu_ghz_measure,
    bench_stab_bts_cpu_vs_gpu,
    bench_stab_bts_packed_cpu_vs_gpu,
    bench_stab_bts_marginals_cpu_vs_gpu,
    bench_stab_bts_device_counts_explicit,
    bench_stab_gpu_shots_explicit,
);
criterion_main!(benches);
