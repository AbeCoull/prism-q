//! Bench circuits that `tests/integration/bench_fixture_routing.rs` also builds, so its
//! routing checks run on the circuits the rows time.

use num_complex::Complex64;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `depth` layers of H on every qubit, then CX on every fourth bond of the chain, the
/// first bond advancing one per layer: a quarter of the dense layer's CX, with the
/// whole chain connected from `depth >= 4` so the decomposed route cannot claim it.
pub fn sparse_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for layer in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        for q in (layer % 4..n_qubits.saturating_sub(1)).step_by(4) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    circuit
}

/// Layer count of the `statevector/dense_unitary` rows.
pub const DENSE_UNITARY_LAYERS: usize = 2;

/// `layers` of Haar-random `k`-qubit unitaries on blocks of `k` adjacent qubits, wrapping
/// at the register end, the tiling shifted `ceil(k / 2)` qubits per layer so the next
/// layer straddles every block boundary. Fusion leaves every block a `Gate::Unitary`.
pub fn dense_unitary_layers(n_qubits: usize, k: usize, layers: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);

    for layer in 0..layers {
        let offset = layer * k.div_ceil(2);
        for block in 0..n_qubits / k {
            let targets: Vec<usize> = (0..k)
                .map(|i| (offset + block * k + i) % n_qubits)
                .collect();
            circuit
                .add_unitary(haar_unitary(k, &mut rng), &targets)
                .expect("Gram-Schmidt output is unitary");
        }
    }

    circuit
}

/// Row-major `2^k x 2^k` unitary from Gram-Schmidt over complex Gaussian columns.
fn haar_unitary(k: usize, rng: &mut ChaCha8Rng) -> Vec<Complex64> {
    let dim = 1usize << k;
    let mut columns: Vec<Vec<Complex64>> = (0..dim)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let radius = (-(1.0 - rng.random::<f64>()).ln()).sqrt();
                    Complex64::from_polar(radius, std::f64::consts::TAU * rng.random::<f64>())
                })
                .collect()
        })
        .collect();

    for j in 0..dim {
        let (done, rest) = columns.split_at_mut(j);
        let column = &mut rest[0];
        for basis in done.iter() {
            let projection: Complex64 = basis
                .iter()
                .zip(column.iter())
                .map(|(b, c)| b.conj() * c)
                .sum();
            for (c, b) in column.iter_mut().zip(basis) {
                *c -= projection * b;
            }
        }
        let norm = column.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        for c in column.iter_mut() {
            *c /= norm;
        }
    }

    (0..dim * dim)
        .map(|index| columns[index % dim][index / dim])
        .collect()
}
