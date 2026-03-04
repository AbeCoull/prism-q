use prism_q::CircuitBuilder;

fn main() {
    let result = CircuitBuilder::new(2)
        .h(0)
        .cx(0, 1)
        .run(42)
        .expect("simulation failed");
    let probs = result.probabilities.expect("no probabilities");

    println!("Bell state (|00⟩ + |11⟩) / sqrt(2):");
    for i in 0..probs.len() {
        let p = probs.get(i);
        if p > 1e-10 {
            println!("  |{:02b}> = {:.4}", i, p);
        }
    }
}
