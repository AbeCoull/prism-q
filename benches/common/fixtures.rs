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

/// [`random_t_insertion_circuit`], then qubit 0 measured into bit 0, `S` on qubit 1 when
/// that bit is one, and qubit 1 measured into bit 1. A guarded `S` has no controlled
/// Pauli form, so the measurement cannot be deferred to the end.
pub fn guarded_s_mid_circuit(n_qubits: usize, t_count: usize, seed: u64) -> Circuit {
    let mut circuit = random_t_insertion_circuit(n_qubits, t_count, seed);
    circuit.num_classical_bits = 2;
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::S,
        targets: SmallVec::from_slice(&[1]),
    });
    circuit.add_measure(1, 1);
    circuit
}

/// Random Clifford layers with a full random CX pairing each, and `t_count` T gates at
/// random spots ahead of a layer, every qubit measured.
pub fn random_clifford_t_layers(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = clifford_t_layers(n, depth, t_count, seed, false);
    c.measure_all();
    c
}

/// [`random_clifford_t_layers`] with CX on nearest neighbours in place of a random
/// pairing: bonds `(2i, 2i + 1)` on even layers and `(2i + 1, 2i + 2)` on odd ones.
pub fn local_clifford_t_layers(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = clifford_t_layers(n, depth, t_count, seed, true);
    c.measure_all();
    c
}

/// H on every qubit and one T on each of qubits `0..t_count`, then `depth` Clifford layers
/// of [`random_clifford_t_layers`], every qubit measured: a register `t_count` wide.
pub fn distinct_t_layers(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..t_count {
        c.add_gate(Gate::T, &[q]);
    }
    c.instructions
        .extend(clifford_t_layers(n, depth, 0, seed, false).instructions);
    c.measure_all();
    c
}

/// [`random_clifford_t_layers`] before its measurements, with the two-bit mid-circuit tail
/// of [`append_measure_guard_reset`] in their place.
pub fn clifford_t_layers_mid_circuit(n: usize, depth: usize, t_count: usize, seed: u64) -> Circuit {
    let mut c = clifford_t_layers(n, depth, t_count, seed, false);
    append_measure_guard_reset(&mut c);
    c
}

fn clifford_t_layers(n: usize, depth: usize, t_count: usize, seed: u64, local: bool) -> Circuit {
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
        if local {
            for q in (layer % 2..n.saturating_sub(1)).step_by(2) {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
            continue;
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

/// Repetition memory on a GHZ data register: each round measures check `Z_q Z_{q+1}` onto
/// ancilla `n_data + q`, then `X` on the ancilla when it read 1 returns it to zero, and the
/// check reading 1 applies `X` on data qubits `0..=q`. Every guard reads one bit, and the
/// ancilla is reused after its measurement with no reset between.
pub fn active_reset_repetition_circuit(n_data: usize, rounds: usize) -> Circuit {
    repetition_memory(n_data, rounds, |circuit, first_bit| {
        for q in 0..n_data - 1 {
            circuit.instructions.push(Instruction::Conditional {
                condition: ClassicalCondition::BitIsOne(first_bit + q),
                gate: Gate::X,
                targets: SmallVec::from_slice(&[n_data + q]),
            });
        }
        for q in 0..n_data - 1 {
            let body = (0..=q).map(x_on).collect();
            circuit.instructions.extend(prism_q::circuit::guarded(
                ClassicalCondition::BitIsOne(first_bit + q),
                body,
            ));
        }
    })
}

/// Repetition memory on a GHZ data register decoded by lookup: each round measures check
/// `Z_q Z_{q+1}` onto an ancilla it then resets, and one guarded body per nonzero syndrome,
/// a comparison of the round's whole check register, applies `X` on the lower-weight data
/// error with that syndrome.
pub fn lookup_repetition_circuit(n_data: usize, rounds: usize) -> Circuit {
    let n_checks = n_data - 1;
    repetition_memory(n_data, rounds, |circuit, first_bit| {
        for q in 0..n_checks {
            circuit.add_reset(n_data + q);
        }
        for syndrome in 1..1u64 << n_checks {
            let mut error = vec![false; n_data];
            for q in 0..n_checks {
                error[q + 1] = error[q] ^ ((syndrome >> q) & 1 == 1);
            }
            let flip = error.iter().filter(|&&e| e).count() > n_data / 2;
            let body = (0..n_data)
                .filter(|&q| error[q] != flip)
                .map(x_on)
                .collect();
            circuit.instructions.extend(prism_q::circuit::guarded(
                ClassicalCondition::RegisterEquals {
                    offset: first_bit,
                    size: n_checks,
                    value: syndrome,
                },
                body,
            ));
        }
    })
}

fn x_on(target: usize) -> Instruction {
    Instruction::Gate {
        gate: Gate::X,
        targets: SmallVec::from_slice(&[target]),
    }
}

/// GHZ data register, `rounds` rounds of check measurements each followed by
/// `after_round(circuit, first_bit)`, then the data measured into the last `n_data` bits.
fn repetition_memory(
    n_data: usize,
    rounds: usize,
    mut after_round: impl FnMut(&mut Circuit, usize),
) -> Circuit {
    let n_checks = n_data - 1;
    let mut circuit = Circuit::new(n_data + n_checks, rounds * n_checks + n_data);
    circuit.add_gate(Gate::H, &[0]);
    for q in 1..n_data {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }
    for round in 0..rounds {
        let first_bit = round * n_checks;
        for q in 0..n_checks {
            let ancilla = n_data + q;
            circuit.add_gate(Gate::Cx, &[q, ancilla]);
            circuit.add_gate(Gate::Cx, &[q + 1, ancilla]);
            circuit.add_measure(ancilla, first_bit + q);
        }
        after_round(&mut circuit, first_bit);
    }
    for q in 0..n_data {
        circuit.add_measure(q, rounds * n_checks + q);
    }
    circuit
}
