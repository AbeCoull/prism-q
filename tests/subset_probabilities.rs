//! The `probabilities_of` terminal: its index convention, the backends that
//! answer it, and what it rejects.

mod common;

use prism_q::sim::noise::{NoiseEvent, NoiseModel};
use prism_q::{BackendKind, Circuit, CircuitBuilder, PrismError, simulate};

use common::{SEED, SV_EPS};

fn bell() -> Circuit {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).cx(0, 1);
    builder.build()
}

/// Three entangled qubits and one independent of them, so a factored route has
/// a block boundary to cross and a subset can straddle it.
fn mixed(qubits: usize) -> Circuit {
    let mut builder = CircuitBuilder::new(qubits);
    builder.h(0).cx(0, 1).t(1).h(2).cx(2, 1);
    for q in 3..qubits {
        builder.ry(0.4 + q as f64 * 0.3, q);
    }
    builder.build()
}

fn subset_of(probabilities: &[f64], qubits: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; 1 << qubits.len()];
    for (index, p) in probabilities.iter().enumerate() {
        let packed = qubits
            .iter()
            .enumerate()
            .fold(0usize, |acc, (bit, &q)| acc | (index >> q & 1) << bit);
        out[packed] += p;
    }
    out
}

// The subset index follows the basis index: `qubits[0]` is the lowest bit, so
// naming the same pair the other way round transposes the distribution.
#[test]
fn the_first_named_qubit_is_the_lowest_bit() {
    let mut builder = CircuitBuilder::new(2);
    builder.x(0);
    let circuit = builder.build();
    let forward = simulate(&circuit)
        .seed(SEED)
        .probabilities_of(&[0, 1])
        .unwrap();
    let reversed = simulate(&circuit)
        .seed(SEED)
        .probabilities_of(&[1, 0])
        .unwrap();
    common::assert_probs_close(&forward, &[0.0, 1.0, 0.0, 0.0], SV_EPS, "q0 low");
    common::assert_probs_close(&reversed, &[0.0, 0.0, 1.0, 0.0], SV_EPS, "q1 low");
}

// A per-qubit marginal cannot distinguish a Bell pair from two independent
// coins. The joint distribution is the whole reason this terminal exists.
#[test]
fn a_joint_distribution_shows_what_marginals_cannot() {
    let circuit = bell();
    let marginals = simulate(&circuit)
        .seed(SEED)
        .marginals()
        .unwrap()
        .into_vec();
    assert!((marginals[0].0 - 0.5).abs() < SV_EPS);
    assert!((marginals[1].0 - 0.5).abs() < SV_EPS);

    let joint = simulate(&circuit)
        .seed(SEED)
        .probabilities_of(&[0, 1])
        .unwrap();
    common::assert_probs_close(&joint, &[0.5, 0.0, 0.0, 0.5], SV_EPS, "bell");
}

#[test]
fn every_subset_agrees_with_the_full_distribution() {
    let circuit = mixed(5);
    let full = simulate(&circuit)
        .seed(SEED)
        .run()
        .unwrap()
        .probabilities
        .expect("a distribution")
        .to_vec();
    for qubits in [
        vec![0],
        vec![4],
        vec![0, 1],
        vec![1, 0],
        vec![0, 2, 4],
        vec![3, 1],
        vec![4, 3, 2, 1, 0],
    ] {
        let actual = simulate(&circuit)
            .seed(SEED)
            .probabilities_of(&qubits)
            .unwrap();
        common::assert_probs_close(
            &actual,
            &subset_of(&full, &qubits),
            SV_EPS,
            &format!("{qubits:?}"),
        );
    }
}

// The factored route holds per-block distributions rather than a joint vector,
// so a subset straddling two blocks multiplies block marginals instead of
// summing a walk. It has to land on the same numbers.
#[test]
fn every_backend_that_answers_agrees() {
    let circuit = mixed(6);
    let reference = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::Statevector)
        .probabilities_of(&[0, 2, 5])
        .unwrap();
    for backend in [
        BackendKind::Auto,
        BackendKind::Sparse,
        BackendKind::Factored,
        BackendKind::TensorNetwork { tolerance: None },
        BackendKind::DensityMatrix,
        BackendKind::Mps { max_bond_dim: 32 },
    ] {
        let actual = simulate(&circuit)
            .seed(SEED)
            .backend(backend.clone())
            .probabilities_of(&[0, 2, 5])
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        common::assert_probs_close(&actual, &reference, SV_EPS, &format!("{backend:?}"));
    }
}

#[test]
fn the_stabilizer_answers_a_clifford_circuit() {
    let mut builder = CircuitBuilder::new(4);
    builder.h(0).cx(0, 1).cx(1, 2).s(2).h(3);
    let circuit = builder.build();
    let actual = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::Stabilizer)
        .probabilities_of(&[0, 1])
        .unwrap();
    common::assert_probs_close(&actual, &[0.5, 0.0, 0.0, 0.5], SV_EPS, "ghz head");
}

// With a noise model the answer is the marginal of the exact mixture, so the
// correlation a bit flip breaks shows up as weight on the disagreeing strings.
#[test]
fn a_noise_model_gives_the_exact_mixture() {
    let circuit = bell();
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    noise.after_gate[1].push(NoiseEvent::pauli(1, 0.25, 0.0, 0.0));
    let actual = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .probabilities_of(&[0, 1])
        .unwrap();
    common::assert_probs_close(
        &actual,
        &[0.375, 0.125, 0.125, 0.375],
        SV_EPS,
        "flipped bell",
    );
}

#[test]
fn a_malformed_subset_is_rejected() {
    let circuit = bell();
    assert!(matches!(
        simulate(&circuit).seed(SEED).probabilities_of(&[]),
        Err(PrismError::InvalidParameter { .. })
    ));
    assert!(matches!(
        simulate(&circuit).seed(SEED).probabilities_of(&[0, 0]),
        Err(PrismError::InvalidParameter { .. })
    ));
    assert!(matches!(
        simulate(&circuit).seed(SEED).probabilities_of(&[2]),
        Err(PrismError::InvalidQubit { .. })
    ));
}
