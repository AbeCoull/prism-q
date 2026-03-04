use prism_q::CircuitBuilder;

fn main() {
    let n = 8;

    // Build a QFT circuit using CircuitBuilder, prepend X on q[0]
    // so the output isn't trivial.
    let mut builder = CircuitBuilder::new(n);
    builder.x(0);
    for i in 0..n {
        builder.h(i);
        for j in (i + 1)..n {
            let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
            builder.cphase(theta, i, j);
        }
    }
    for i in 0..n / 2 {
        builder.swap(i, n - 1 - i);
    }

    let result = builder.run(42).expect("simulation failed");
    let probs = result.probabilities.expect("no probabilities");

    println!("QFT on {n} qubits (input state |1⟩):");
    println!("  Total basis states: {}", probs.len());

    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("  Top 10 states:");
    for (idx, p) in indexed.iter().take(10) {
        println!("    |{:0width$b}⟩ = {:.6}", idx, p, width = n);
    }

    // Alternative: use the pre-built qft_circuit() from circuits module
    // let circuit = prism_q::circuits::qft_circuit(n);
    // let result = prism_q::run(&circuit, 42).expect("simulation failed");
}
