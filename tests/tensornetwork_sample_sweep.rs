//! Native tensor-network sampling on the public route past the dense
//! probability ceiling, where `sample_basis_states` takes the conditional
//! sweep and its per-call plan cache. Isolated in its own test binary: it
//! overrides `PRISM_MAX_PROB_QUBITS`, which the cap helper caches per process.

mod common;

use common::{SEED, caps};
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuits;

const PROB_CAP: usize = 4;

fn small_prob_cap() {
    caps::set_once(&[("PRISM_MAX_PROB_QUBITS", "4")]);
}

// A random 6-qubit circuit two qubits over the pinned cap, so the public
// route runs the sweep with the plan cache read on every shot after the
// first. Joint distribution against the exported statevector (the export cap
// is a separate variable) with per-outcome binomial bands, and the seed
// contract on the packed words.
#[test]
fn sweep_on_a_random_circuit_matches_statevector_distribution() {
    small_prob_cap();
    let n = PROB_CAP + 2;
    let circuit = circuits::random_circuit(n, 5, SEED);
    let mut sv = StatevectorBackend::new(SEED);
    sv.init(n, 0).unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let probs: Vec<f64> = sv
        .export_statevector()
        .unwrap()
        .iter()
        .map(|amp| amp.norm_sqr())
        .collect();

    let mut tn = TensorNetworkBackend::new(SEED);
    tn.init(n, 0).unwrap();
    tn.apply_instructions(&circuit.instructions).unwrap();
    assert!(
        tn.probabilities().is_err(),
        "the pinned cap must force the sweep"
    );

    let shots = 2000usize;
    let samples = tn.sample_basis_states(shots, SEED).unwrap();
    let mut counts = vec![0usize; 1 << n];
    for shot in 0..shots {
        let index = (0..n)
            .filter(|&q| samples.bit(shot, q))
            .fold(0usize, |acc, q| acc | 1 << q);
        counts[index] += 1;
    }
    for (index, (&count, &p)) in counts.iter().zip(&probs).enumerate() {
        let freq = count as f64 / shots as f64;
        let sigma = (p * (1.0 - p) / shots as f64).sqrt().max(1e-3);
        assert!(
            (freq - p).abs() < 6.0 * sigma,
            "outcome {index}: {freq} vs {p}"
        );
    }

    let again = tn.sample_basis_states(shots, SEED).unwrap();
    let bits = |s: &prism_q::backend::BasisSamples| -> Vec<bool> {
        (0..shots)
            .flat_map(|shot| (0..n).map(move |q| (shot, q)))
            .map(|(shot, q)| s.bit(shot, q))
            .collect()
    };
    assert_eq!(bits(&samples), bits(&again));
}
