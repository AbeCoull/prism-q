//! Quick GPU-vs-CPU timing for statevector simulation across qubit counts.
//!
//! Runs each (backend, qubits) pair three times, reports the minimum. Fusion is disabled on
//! the CPU side by calling `apply` in a loop so the kernels compared are apples-to-apples
//! (no fusion passes to bias CPU timing).
//!
//! Build + run:
//!
//! ```text
//! CUDA_PATH=... LIB=... cargo run --release \
//!     --features "parallel gpu" --example bench_gpu_vs_cpu
//! ```

use std::time::Instant;

use prism_q::backend::Backend;
use prism_q::circuits::random_circuit;
use prism_q::StatevectorBackend;

#[cfg(feature = "gpu")]
use prism_q::gpu::GpuContext;

const SEED: u64 = 0xDEAD_BEEF;
const REPS: usize = 3;

fn time_cpu(num_qubits: usize, depth: usize) -> f64 {
    let circuit = random_circuit(num_qubits, depth, SEED);
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let mut b = StatevectorBackend::new(42);
        b.init(num_qubits, 0).unwrap();
        let start = Instant::now();
        for inst in &circuit.instructions {
            b.apply(inst).unwrap();
        }
        let _ = b.probabilities().unwrap();
        let secs = start.elapsed().as_secs_f64();
        if secs < best {
            best = secs;
        }
    }
    best
}

#[cfg(feature = "gpu")]
fn time_gpu(num_qubits: usize, depth: usize) -> Option<f64> {
    let ctx = match GpuContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("gpu ctx init failed: {e}");
            return None;
        }
    };
    let circuit = random_circuit(num_qubits, depth, SEED);
    let mut best = f64::INFINITY;
    for _ in 0..REPS {
        let mut b = StatevectorBackend::new(42).with_gpu(ctx.clone());
        b.init(num_qubits, 0).unwrap();
        let start = Instant::now();
        for inst in &circuit.instructions {
            b.apply(inst).unwrap();
        }
        let _ = b.probabilities().unwrap();
        let secs = start.elapsed().as_secs_f64();
        if secs < best {
            best = secs;
        }
    }
    Some(best)
}

#[cfg(not(feature = "gpu"))]
fn time_gpu(_num_qubits: usize, _depth: usize) -> Option<f64> {
    None
}

fn main() {
    println!(
        "{:>7} {:>7} {:>14} {:>14} {:>10}",
        "qubits", "depth", "cpu (ms)", "gpu (ms)", "speedup"
    );
    println!("{}", "-".repeat(60));

    let configs: &[(usize, usize)] = &[(10, 20), (14, 20), (16, 20), (18, 20), (20, 20), (22, 20)];

    for &(n, d) in configs {
        let cpu_sec = time_cpu(n, d);
        let gpu_sec = time_gpu(n, d);
        match gpu_sec {
            Some(g) => {
                println!(
                    "{:>7} {:>7} {:>14.3} {:>14.3} {:>9.2}x",
                    n,
                    d,
                    cpu_sec * 1e3,
                    g * 1e3,
                    cpu_sec / g
                );
            }
            None => {
                println!(
                    "{:>7} {:>7} {:>14.3} {:>14} {:>10}",
                    n,
                    d,
                    cpu_sec * 1e3,
                    "-",
                    "n/a"
                );
            }
        }
    }
}
