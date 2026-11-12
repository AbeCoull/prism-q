//! Bench circuits that `tests/integration/bench_fixture_routing.rs` also builds, so its
//! routing checks run on the circuits the rows time.

use num_complex::Complex64;
use prism_q::PauliTerm;
use prism_q::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
use prism_q::gates::Gate;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Build ten-qubit Ry/Rz brickwork blocks with terminal measurements.
pub fn terminal_disjoint_blocks(n: usize, strided: bool, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n, n);
    let blocks = n / 10;
    for block in 0..blocks {
        let qubit = |local| {
            if strided {
                block + local * blocks
            } else {
                block * 10 + local
            }
        };
        for layer in 0..10 {
            for q in 0..10 {
                circuit.add_gate(
                    Gate::Ry(rng.random::<f64>() * std::f64::consts::TAU),
                    &[qubit(q)],
                );
                circuit.add_gate(
                    Gate::Rz(rng.random::<f64>() * std::f64::consts::TAU),
                    &[qubit(q)],
                );
            }
            for q in (layer % 2..9).step_by(2) {
                circuit.add_gate(Gate::Cx, &[qubit(q), qubit(q + 1)]);
            }
        }
    }
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    circuit
}

/// The `product/expectation/pauli` observables: `Z` on every qubit, then `XX` on every
/// bond of the chain.
pub fn product_observables(n_qubits: usize) -> Vec<Vec<PauliTerm>> {
    let singles = (0..n_qubits).map(|q| vec![PauliTerm::z(q)]);
    let bonds = (1..n_qubits).map(|q| vec![PauliTerm::x(q - 1), PauliTerm::x(q)]);
    singles.chain(bonds).collect()
}

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

/// H on every qubit and a CX on each chain bond with probability one half, `t_count` T
/// gates on random qubits, then three random Clifford layers of the same shape.
pub fn random_t_insertion_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let cliffords = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..n_qubits - 1 {
        if rng.random_bool(0.5) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    for _ in 0..t_count {
        let q = rng.random_range(0..n_qubits);
        circuit.add_gate(Gate::T, &[q]);
    }

    for _ in 0..3 {
        for q in 0..n_qubits {
            let gate_idx = rng.random_range(0..cliffords.len());
            circuit.add_gate(cliffords[gate_idx].clone(), &[q]);
        }
        for q in 0..n_qubits - 1 {
            if rng.random_bool(0.5) {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
    }
    circuit
}

/// H on every qubit, then per T gate a block of `3n` random H, S, Sdg or CX followed by
/// one T or Tdg on a random qubit, then a random 1q Clifford layer. The Clifford blocks
/// keep the T rotations from commuting into fewer distinct terms.
pub fn scrambled_clifford_t_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut circuit = Circuit::new(n_qubits, 0);
    let ones = [Gate::H, Gate::S, Gate::Sdg];
    let finals = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];

    for q in 0..n_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for _ in 0..t_count {
        for _ in 0..3 * n_qubits {
            let choice = rng.random_range(0..4);
            if choice < ones.len() {
                circuit.add_gate(ones[choice].clone(), &[rng.random_range(0..n_qubits)]);
            } else {
                let a = rng.random_range(0..n_qubits);
                let b = (a + rng.random_range(1..n_qubits)) % n_qubits;
                circuit.add_gate(Gate::Cx, &[a, b]);
            }
        }
        let t = if rng.random_bool(0.5) {
            Gate::T
        } else {
            Gate::Tdg
        };
        circuit.add_gate(t, &[rng.random_range(0..n_qubits)]);
    }
    for q in 0..n_qubits {
        circuit.add_gate(finals[rng.random_range(0..finals.len())].clone(), &[q]);
    }
    circuit
}

/// [`random_t_insertion_circuit`] with the two-bit mid-circuit tail of
/// [`append_measure_guard_reset`].
pub fn stabilizer_rank_mid_circuit_shot_circuit(
    n_qubits: usize,
    t_count: usize,
    seed: u64,
) -> Circuit {
    let mut circuit = random_t_insertion_circuit(n_qubits, t_count, seed);
    append_measure_guard_reset(&mut circuit);
    circuit
}

/// Random Clifford layers with a full random CX pairing each, and `t_count` T gates at
/// random spots ahead of a layer, every qubit measured.
pub fn random_clifford_t_layers(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = clifford_t_layers(n, depth, t_count, seed);
    c.measure_all();
    c
}

/// [`random_clifford_t_layers`] before its measurements, with the two-bit mid-circuit tail
/// of [`append_measure_guard_reset`] in their place.
pub fn clifford_t_layers_mid_circuit(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = clifford_t_layers(n, depth, t_count, seed);
    append_measure_guard_reset(&mut c);
    c
}

fn clifford_t_layers(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let ones = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Z];
    let mut spots: Vec<usize> = (0..depth * n).collect();
    for i in (1..spots.len()).rev() {
        spots.swap(i, rng.random_range(0..=i));
    }
    spots.truncate(t_count);
    let mut c = Circuit::new(n, n);
    for layer in 0..depth {
        for q in (0..n).filter(|q| spots.contains(&(layer * n + q))) {
            c.add_gate(Gate::T, &[q]);
        }
        for q in 0..n {
            c.add_gate(ones[rng.random_range(0..ones.len())].clone(), &[q]);
        }
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            perm.swap(i, rng.random_range(0..=i));
        }
        for pair in perm.as_chunks::<2>().0 {
            c.add_gate(Gate::Cx, &[pair[0], pair[1]]);
        }
    }
    c
}

/// Measure qubit 0 into bit 0, X on qubit 1 when that bit is one, reset qubit 0, measure
/// qubit 1 into bit 1, leaving two classical bits.
fn append_measure_guard_reset(circuit: &mut Circuit) {
    circuit.num_classical_bits = 2;
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[1]),
    });
    circuit.add_reset(0);
    circuit.add_measure(1, 1);
}
