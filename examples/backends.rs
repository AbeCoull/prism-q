use prism_q::circuits::clifford_heavy_circuit;
use prism_q::{run_with, BackendKind};

fn main() {
    // A Clifford-only 4-qubit circuit with entangling gates.
    let circuit = clifford_heavy_circuit(4, 6, 42);

    let backends: Vec<(&str, BackendKind)> = vec![
        ("Statevector", BackendKind::Statevector),
        ("Stabilizer", BackendKind::Stabilizer),
        ("Sparse", BackendKind::Sparse),
        ("MPS (bond=64)", BackendKind::Mps { max_bond_dim: 64 }),
        ("Tensor Network", BackendKind::TensorNetwork),
        ("Factored", BackendKind::Factored),
        ("Auto", BackendKind::Auto),
        // ProductState omitted: rejects entangling gates by design.
    ];

    for (name, kind) in &backends {
        let result = run_with(kind.clone(), &circuit, 42).expect("simulation failed");
        let probs = result.probabilities.expect("no probabilities").to_vec();

        let nonzero: Vec<(usize, f64)> = probs
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-10)
            .map(|(i, &p)| (i, p))
            .collect();

        println!("{name}:");
        for (idx, p) in &nonzero {
            println!("  |{:04b}⟩ = {:.4}", idx, p);
        }
        println!();
    }
}
