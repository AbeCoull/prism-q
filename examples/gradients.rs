//! Adjoint and parameter-shift gradients for a variational circuit, and a batched sweep.

use std::f64::consts::PI;

use prism_q::{CircuitBuilder, PauliObservable, PauliTerm, PreparedCircuit, simulate};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn main() {
    let n = 4;
    let mut builder = CircuitBuilder::new(n);
    for q in 0..n {
        builder.ry(0.1, q).param(q);
    }
    for q in 0..n - 1 {
        builder.cx(q, q + 1);
    }
    for q in 0..n {
        builder.ry(0.1, q).param(n + q);
    }
    let (circuit, params) = builder.build_parametric();

    // Transverse-field Ising chain: -sum Z_i Z_{i+1} - 0.5 sum X_i.
    let mut hamiltonian: Vec<(f64, Vec<PauliTerm>)> = (0..n - 1)
        .map(|q| (-1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
        .collect();
    hamiltonian.extend((0..n).map(|q| (-0.5, vec![PauliTerm::x(q)])));

    let adjoint = simulate(&circuit)
        .seed(42)
        .expectation_gradient(&hamiltonian, &params)
        .expect("adjoint gradient failed");
    let shifted = simulate(&circuit)
        .seed(42)
        .expectation_gradient_shift(&hamiltonian, &params)
        .expect("parameter-shift gradient failed");
    let max_gap = adjoint
        .gradient
        .iter()
        .zip(&shifted.gradient)
        .map(|(a, s)| (a - s).abs())
        .fold(0.0, f64::max);
    println!("energy at the template angles: {:.6}", adjoint.value);
    println!("largest adjoint and parameter-shift gap: {max_gap:.2e}");

    let mut theta = vec![0.1; params.num_slots()];
    let mut energy = adjoint.value;
    for _ in 0..100 {
        let bound = params.bind(&circuit, &theta).expect("bind failed");
        let step = simulate(&bound)
            .seed(42)
            .expectation_gradient(&hamiltonian, &params)
            .expect("adjoint gradient failed");
        energy = step.value;
        for (t, g) in theta.iter_mut().zip(&step.gradient) {
            *t -= 0.2 * g;
        }
    }
    println!("energy after 100 steps: {energy:.6}");

    let observable = PauliObservable::from_terms(hamiltonian).expect("invalid observable");
    let mut prepared = PreparedCircuit::new(circuit, params).expect("prepare failed");
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let points: Vec<Vec<f64>> = (0..256)
        .map(|_| (0..2 * n).map(|_| rng.random_range(-PI..PI)).collect())
        .collect();
    let lowest = prepared
        .observable_expectation_many(&points, &observable, 42)
        .expect("batch failed")
        .iter()
        .map(|e| e.mean)
        .fold(f64::INFINITY, f64::min);
    println!("lowest of {} random points: {lowest:.6}", points.len());
}
