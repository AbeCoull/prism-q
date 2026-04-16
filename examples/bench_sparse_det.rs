use prism_q::circuit::Circuit;
use std::time::Instant;

const SEED: u64 = 0xDEAD_BEEF;

fn sparse_active_circuit(n_qubits: usize, n_active: usize) -> Circuit {
    let mut active = prism_q::circuits::clifford_heavy_circuit(n_active, 10, SEED);
    active.num_qubits = n_qubits;
    active.num_classical_bits = n_qubits;
    for q in 0..n_qubits {
        active.add_measure(q, q);
    }
    active
}

fn main() {
    let configs = vec![(100, 10), (200, 20), (500, 50), (1000, 100)];
    let shot_counts = vec![10_000, 100_000, 1_000_000];

    for &(n_qubits, n_active) in &configs {
        let circuit = sparse_active_circuit(n_qubits, n_active);
        let det_pct = 100 * (n_qubits - n_active) / n_qubits;

        for &n_shots in &shot_counts {
            let mut sampler = prism_q::compile_measurements(&circuit, SEED).unwrap();
            // warmup
            let _ = sampler.sample_bulk_packed(n_shots);

            let mut times = Vec::new();
            for _ in 0..10 {
                let start = Instant::now();
                let _ = sampler.sample_bulk_packed(n_shots);
                times.push(start.elapsed());
            }
            times.sort();
            let median = times[5];
            println!(
                "{}q/{}active/{}%det  shots={:<10}  median={:.3}ms",
                n_qubits,
                n_active,
                det_pct,
                n_shots,
                median.as_nanos() as f64 / 1_000_000.0
            );
        }
    }
}
