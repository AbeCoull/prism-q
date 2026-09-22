//! Dense k-qubit unitaries: agreement with the gate sequence the matrix comes
//! from, the backends that decline the variant, and the fusion barrier.

mod common;

use common::{DM_EPS, FACTORED_EPS, MPS_EPS, SEED, SV_EPS, TN_EPS, circuit_unitary};
use num_complex::Complex64;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::sim::{BackendKind, ResolvedBackend};
use prism_q::{Circuit, CircuitBuilder, Gate, Instruction, SpdTruncation, sim, simulate};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const THREE_Q_TARGETS: [usize; 3] = [4, 1, 3];
const FOUR_Q_TARGETS: [usize; 4] = [5, 0, 3, 2];
const WIDTH: usize = 6;

/// A random `k`-qubit circuit of rotations and CX, the source of both the
/// dense matrix under test and the gate sequence it is checked against.
fn random_block(k: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut b = CircuitBuilder::new(k);
    for _ in 0..3 {
        for q in 0..k {
            b.rz(rng.random::<f64>() * std::f64::consts::TAU, q);
            b.ry(rng.random::<f64>() * std::f64::consts::TAU, q);
        }
        for q in 0..k {
            b.cx(q, (q + 1) % k);
        }
    }
    b.build()
}

/// A state with no two amplitudes equal, so a wrong index convention cannot
/// hide behind a symmetric distribution.
fn prep(num_qubits: usize) -> Circuit {
    let mut b = CircuitBuilder::new(num_qubits);
    for q in 0..num_qubits {
        b.ry(0.3 + 0.17 * q as f64, q);
    }
    for q in 0..num_qubits - 1 {
        b.cx(q, q + 1);
    }
    b.build()
}

fn append(dst: &mut Circuit, block: &Circuit, targets: &[usize]) {
    for inst in &block.instructions {
        let Instruction::Gate {
            gate,
            targets: block_targets,
        } = inst
        else {
            panic!("the block holds gates only");
        };
        let mapped: Vec<usize> = block_targets.iter().map(|&q| targets[q]).collect();
        dst.add_gate(gate.clone(), &mapped);
    }
}

/// The dense circuit and the gate sequence it must agree with: one random
/// `k`-qubit block placed on `targets`, once as a `Gate::Unitary` and once as
/// the rotations and CX it was built from.
fn pair(k: usize, targets: &[usize]) -> (Circuit, Circuit) {
    let block = random_block(k, SEED);
    let mat: Vec<Complex64> = circuit_unitary(&block).into_iter().flatten().collect();
    let gate = Gate::unitary(mat, k).expect("a product of rotations is unitary");
    assert!(matches!(gate, Gate::Unitary(_)), "{gate:?}");

    let mut dense = prep(WIDTH);
    dense.add_gate(gate, targets);

    let mut reference = prep(WIDTH);
    append(&mut reference, &block, targets);

    (dense, reference)
}

fn probs<B: Backend>(backend: &mut B, circuit: &Circuit) -> Vec<f64> {
    sim::run_on(backend, circuit).expect("run");
    backend.probabilities().expect("probabilities")
}

fn assert_close(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "{label}: prob[{i}] expected {e:.12}, got {a:.12}"
        );
    }
}

fn assert_matches_gate_sequence(k: usize, targets: &[usize]) {
    let (dense, reference) = pair(k, targets);
    let expected = probs(&mut StatevectorBackend::new(SEED), &reference);

    let mut sv = StatevectorBackend::new(SEED);
    assert_close(&probs(&mut sv, &dense), &expected, SV_EPS, "statevector");

    let mut factored = FactoredBackend::new(SEED);
    assert_close(
        &probs(&mut factored, &dense),
        &expected,
        FACTORED_EPS,
        "factored",
    );

    let mut mps = MpsBackend::new(SEED, 64);
    assert_close(&probs(&mut mps, &dense), &expected, MPS_EPS, "mps");

    let mut tn = TensorNetworkBackend::new(SEED);
    assert_close(&probs(&mut tn, &dense), &expected, TN_EPS, "tensor network");

    let mut dm = DensityMatrixBackend::new(SEED);
    assert_close(&probs(&mut dm, &dense), &expected, DM_EPS, "density matrix");
}

#[test]
fn three_qubit_unitary_matches_the_gate_sequence_on_every_accepting_backend() {
    assert_matches_gate_sequence(3, &THREE_Q_TARGETS);
}

#[test]
fn four_qubit_unitary_matches_the_gate_sequence_on_every_accepting_backend() {
    assert_matches_gate_sequence(4, &FOUR_Q_TARGETS);
}

#[test]
fn the_first_target_is_the_most_significant_matrix_index_bit() {
    // `I (x) I (x) X` on targets [2, 0, 1] is `X` on the last target, qubit 1,
    // which is the opposite of the `q[0]`-is-least-significant state order.
    let dim = 8;
    let mut mat = vec![Complex64::new(0.0, 0.0); dim * dim];
    for row in 0..dim {
        mat[row * dim + (row ^ 1)] = Complex64::new(1.0, 0.0);
    }
    let gate = Gate::unitary(mat, 3).expect("a permutation is unitary");
    assert!(matches!(gate, Gate::Unitary(_)), "{gate:?}");

    let mut dense = Circuit::new(3, 0);
    dense.add_gate(gate, &[2, 0, 1]);

    let mut reference = Circuit::new(3, 0);
    reference.add_gate(Gate::X, &[1]);

    let expected = probs(&mut StatevectorBackend::new(SEED), &reference);
    let actual = probs(&mut StatevectorBackend::new(SEED), &dense);
    assert_close(&actual, &expected, SV_EPS, "index convention");
}

#[test]
fn applying_the_inverse_returns_the_state() {
    let block = random_block(3, SEED);
    let mat: Vec<Complex64> = circuit_unitary(&block).into_iter().flatten().collect();
    let gate = Gate::unitary(mat, 3).expect("a product of rotations is unitary");

    let start = prep(WIDTH);
    let mut round = start.clone();
    round.add_gate(gate.clone(), &THREE_Q_TARGETS);
    round.add_gate(gate.inverse(), &THREE_Q_TARGETS);

    let expected = probs(&mut StatevectorBackend::new(SEED), &start);
    let actual = probs(&mut StatevectorBackend::new(SEED), &round);
    assert_close(&actual, &expected, 1e-12, "inverse round trip");
}

/// One-qubit gates the declining backends all accept, then the dense gate, so
/// the rejection under test is the one the backend reaches first.
fn decline_circuit(with_t: bool) -> Circuit {
    let block = random_block(3, SEED);
    let mat: Vec<Complex64> = circuit_unitary(&block).into_iter().flatten().collect();
    let gate = Gate::unitary(mat, 3).expect("a product of rotations is unitary");

    let mut circuit = Circuit::new(WIDTH, 0);
    for q in 0..WIDTH {
        circuit.add_gate(Gate::H, &[q]);
    }
    if with_t {
        circuit.add_gate(Gate::T, &[0]);
    }
    circuit.add_gate(gate, &THREE_Q_TARGETS);
    circuit
}

/// Every backend without a dense multi-qubit path declines by name rather than
/// dropping the gate or panicking on a two-qubit matrix it cannot produce.
#[test]
fn backends_without_a_dense_path_decline_by_name() {
    let circuit = decline_circuit(false);

    let cases: Vec<(&str, Box<dyn Backend>)> = vec![
        ("sparse", Box::new(SparseBackend::new(SEED))),
        ("productstate", Box::new(ProductStateBackend::new(SEED))),
        ("stabilizer", Box::new(StabilizerBackend::new(SEED))),
        (
            "factored-stabilizer",
            Box::new(FactoredStabilizerBackend::new(SEED)),
        ),
    ];
    for (name, mut backend) in cases {
        let err = sim::run_on(backend.as_mut(), &circuit)
            .expect_err("expected a decline")
            .to_string();
        assert!(err.contains(name), "{name}: {err}");
        assert!(err.contains("unitary"), "{name}: {err}");
    }
}

#[test]
fn the_clifford_t_engines_decline_by_name() {
    let circuit = decline_circuit(true);
    for kind in [
        BackendKind::StabilizerRank,
        BackendKind::StochasticPauli { num_samples: 64 },
        BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Budget { max_terms: 64 },
        },
    ] {
        let label = format!("{kind:?}");
        let err = simulate(&circuit)
            .backend(kind)
            .seed(SEED)
            .marginals()
            .expect_err("expected a decline")
            .to_string();
        assert!(err.contains("unitary"), "{label}: {err}");
    }
}

#[test]
fn a_dense_unitary_has_no_openqasm_spelling() {
    let err = prism_q::circuit::qasm_export::to_qasm3(&decline_circuit(false))
        .expect_err("expected a decline")
        .to_string();
    assert!(err.contains("unitary"), "{err}");
}

#[test]
fn a_lowered_one_qubit_matrix_keeps_a_circuit_clifford() {
    let mut circuit = Circuit::new(30, 0);
    let h: Vec<Complex64> = Gate::H
        .matrix_2x2()
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    for q in 0..30 {
        circuit.add_gate(Gate::unitary(h.clone(), 1).unwrap(), &[q]);
        circuit.add_gate(Gate::Cx, &[q, (q + 1) % 30]);
    }
    assert!(circuit.is_clifford_only());
    let route = simulate(&circuit)
        .seed(SEED)
        .marginals()
        .expect("a Clifford circuit runs at 30 qubits")
        .metadata
        .backend;
    assert!(
        matches!(
            route,
            ResolvedBackend::Stabilizer | ResolvedBackend::FactoredStabilizer
        ),
        "{route:?}"
    );
}

#[test]
fn a_two_qubit_diagonal_becomes_the_diagonal_batch() {
    let phases = [0.0, 0.4, -1.1, 2.3];
    let dim = 4;
    let mut mat = vec![Complex64::new(0.0, 0.0); dim * dim];
    for (index, phase) in phases.iter().enumerate() {
        mat[index * dim + index] = Complex64::from_polar(1.0, *phase);
    }

    let mut circuit = Circuit::new(3, 0);
    circuit.add_unitary(mat, &[2, 0]).unwrap();
    let Instruction::Gate { gate, .. } = &circuit.instructions[0] else {
        panic!("expected a gate");
    };
    assert!(matches!(gate, Gate::DiagonalBatch(_)), "{gate:?}");

    let mut reference = Circuit::new(3, 0);
    for (index, phase) in phases.iter().enumerate() {
        let mut mat = vec![Complex64::new(0.0, 0.0); dim * dim];
        for i in 0..dim {
            mat[i * dim + i] = Complex64::new(1.0, 0.0);
        }
        mat[index * dim + index] = Complex64::from_polar(1.0, *phase);
        reference.add_unitary(mat, &[2, 0]).unwrap();
    }

    let mut prepared = prep(3);
    prepared.instructions.extend(circuit.instructions.clone());
    let mut prepared_reference = prep(3);
    prepared_reference
        .instructions
        .extend(reference.instructions.clone());

    let actual = probs(&mut StatevectorBackend::new(SEED), &prepared);
    let expected = probs(&mut StatevectorBackend::new(SEED), &prepared_reference);
    assert_close(&actual, &expected, SV_EPS, "diagonal lowering");
}

#[test]
fn fusion_treats_a_dense_unitary_as_a_barrier() {
    let num_qubits = 16;
    let targets = [2usize, 7, 11];
    let block = random_block(3, SEED);
    let mat: Vec<Complex64> = circuit_unitary(&block).into_iter().flatten().collect();
    let gate = Gate::unitary(mat, 3).expect("a product of rotations is unitary");

    let mut circuit = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::Rz(0.11 + 0.03 * q as f64), &[q]);
        circuit.add_gate(Gate::Ry(0.23 + 0.05 * q as f64), &[q]);
    }
    circuit.add_gate(gate, &targets);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::Ry(0.31 + 0.07 * q as f64), &[q]);
        circuit.add_gate(Gate::Rz(0.43 + 0.09 * q as f64), &[q]);
    }

    let fused = fuse_circuit(&circuit, true);
    assert!(
        fused.instructions.len() < circuit.instructions.len(),
        "nothing fused: {} instructions",
        fused.instructions.len()
    );

    let barrier = fused
        .instructions
        .iter()
        .position(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::Unitary(_),
                    ..
                }
            )
        })
        .expect("the dense gate survives fusion");
    assert_eq!(
        fused
            .instructions
            .iter()
            .filter(|inst| matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::Unitary(_),
                    ..
                }
            ))
            .count(),
        1
    );

    let touches = |inst: &Instruction, qubit: usize| match inst {
        Instruction::Gate { gate, targets } => {
            targets.contains(&qubit)
                || match gate {
                    Gate::MultiFused(data) => data.gates.iter().any(|&(q, _)| q == qubit),
                    Gate::Multi2q(data) => {
                        data.gates.iter().any(|&(a, b, _)| a == qubit || b == qubit)
                    }
                    _ => false,
                }
        }
        _ => false,
    };
    for &q in &targets {
        assert!(
            fused.instructions[..barrier]
                .iter()
                .any(|inst| touches(inst, q)),
            "qubit {q} lost its gates before the barrier"
        );
        assert!(
            fused.instructions[barrier + 1..]
                .iter()
                .any(|inst| touches(inst, q)),
            "a gate on qubit {q} crossed the barrier"
        );
    }

    let plain = probs(&mut StatevectorBackend::new(SEED), &circuit);
    let mut sv = StatevectorBackend::new(SEED);
    sv.init(num_qubits, 0).unwrap();
    sv.apply_instructions(&fused.instructions).unwrap();
    assert_close(&sv.probabilities().unwrap(), &plain, SV_EPS, "fused stream");
}
