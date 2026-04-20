//! Three-path timing for the GPU crossover work: CPU direct, GPU direct
//! (`.with_gpu(ctx)`), GPU dispatched (`run_with(BackendKind::StatevectorGpu)`).
//!
//! Runs each path three times per (circuit, qubits) pair and reports the minimum.
//! The "GPU direct" path bypasses fusion + decomposition + crossover; the "GPU
//! dispatched" path goes through them.
//!
//! Build + run (`cargo run --release --features "parallel gpu" --example bench_gpu_dispatch`).

#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_q::backend::Backend;
#[cfg(feature = "gpu")]
use prism_q::circuits::{independent_bell_pairs, random_circuit};
#[cfg(feature = "gpu")]
use prism_q::{sim, BackendKind, Circuit, StatevectorBackend};

#[cfg(feature = "gpu")]
use prism_q::gpu::GpuContext;

#[cfg(feature = "gpu")]
use prism_q::run_with_gpu;

#[cfg(feature = "gpu")]
const SEED: u64 = 0xDEAD_BEEF;
#[cfg(feature = "gpu")]
const REPS: usize = 3;

#[cfg(feature = "gpu")]
fn time_cpu_direct(circuit: &Circuit) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let mut b = StatevectorBackend::new(42);
        b.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        let start = Instant::now();
        for inst in &circuit.instructions {
            b.apply(inst).unwrap();
        }
        let _ = b.probabilities().unwrap();
        let t = start.elapsed().as_secs_f64();
        if t < best {
            best = t;
        }
    }
    best
}

#[cfg(feature = "gpu")]
fn time_cpu_dispatched(circuit: &Circuit) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let start = Instant::now();
        let _ = sim::run_with(BackendKind::Statevector, circuit, 42).unwrap();
        let t = start.elapsed().as_secs_f64();
        if t < best {
            best = t;
        }
    }
    best
}

#[cfg(feature = "gpu")]
fn time_gpu_direct(circuit: &Circuit, ctx: &std::sync::Arc<GpuContext>) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let mut b = StatevectorBackend::new(42).with_gpu(ctx.clone());
        b.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        let start = Instant::now();
        for inst in &circuit.instructions {
            b.apply(inst).unwrap();
        }
        let _ = b.probabilities().unwrap();
        let t = start.elapsed().as_secs_f64();
        if t < best {
            best = t;
        }
    }
    best
}

#[cfg(feature = "gpu")]
fn time_gpu_dispatched(circuit: &Circuit, ctx: &std::sync::Arc<GpuContext>) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let start = Instant::now();
        let _ = run_with_gpu(circuit, 42, ctx.clone()).unwrap();
        let t = start.elapsed().as_secs_f64();
        if t < best {
            best = t;
        }
    }
    best
}

#[cfg(feature = "gpu")]
fn row(label: &str, circuit: Circuit, ctx: &std::sync::Arc<GpuContext>) {
    let cpu_d = time_cpu_direct(&circuit) * 1e3;
    let cpu_r = time_cpu_dispatched(&circuit) * 1e3;
    let gpu_d = time_gpu_direct(&circuit, ctx) * 1e3;
    let gpu_r = time_gpu_dispatched(&circuit, ctx) * 1e3;
    println!(
        "{:<18} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.2}x",
        label,
        cpu_d,
        cpu_r,
        gpu_d,
        gpu_r,
        gpu_d / gpu_r,
    );
}

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("build with --features \"parallel gpu\"");
    }

    #[cfg(feature = "gpu")]
    {
        let ctx = match GpuContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no usable GPU: {e}");
                return;
            }
        };

        println!(
            "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "config", "cpu_d(ms)", "cpu_r(ms)", "gpu_d(ms)", "gpu_r(ms)", "gpu_d/gpu_r"
        );
        println!("{}", "-".repeat(82));

        // Non-decomposable random circuits across the crossover boundary.
        for n in &[10usize, 14, 16, 18, 20] {
            row(
                &format!("random {n}q d10"),
                random_circuit(*n, 10, SEED),
                &ctx,
            );
        }

        // Decomposable circuit: 8 independent 2q Bell pairs at 16q total.
        row("bell_pairs 16q", independent_bell_pairs(8), &ctx);
    }
}
