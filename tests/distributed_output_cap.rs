//! Distributed queries either need the dense gather or they do not. Isolated in
//! its own test binary: it overrides `PRISM_MAX_PROB_QUBITS`, which the cap
//! helper caches per process.
#![cfg(feature = "distributed")]

use std::sync::Once;

use prism_q::distributed::DistributedContext;
use prism_q::gates::Gate;
use prism_q::{Circuit, PrismError, simulate};

const SEED: u64 = 42;
const PROB_CAP: usize = 4;
const N: usize = PROB_CAP + 2;

fn small_prob_cap() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries the cap while it is written.
        unsafe { std::env::set_var("PRISM_MAX_PROB_QUBITS", "4") };
    });
}

/// Qubit 0 held at |1>, the rest in an even superposition, so every marginal is
/// known without a reference run (the reference would hit the same cap).
fn known_marginals_circuit(num_classical_bits: usize) -> Circuit {
    let mut circuit = Circuit::new(N, num_classical_bits);
    circuit.add_gate(Gate::X, &[0]);
    for q in 1..N {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit
}

#[test]
fn run_rejects_a_register_past_the_dense_cap_before_executing() {
    small_prob_cap();
    let err = simulate(&known_marginals_circuit(0))
        .distributed(DistributedContext::serial())
        .seed(SEED)
        .run()
        .unwrap_err();
    match err {
        PrismError::BackendUnsupported { backend, operation } => {
            assert_eq!(backend, "distributed_statevector");
            assert!(
                operation.contains("probabilities")
                    && operation.contains(&format!("max {PROB_CAP}")),
                "expected the dense probability cap, got {operation}"
            );
        }
        other => panic!("expected a cap rejection, got {other:?}"),
    }
}

#[test]
fn shots_answer_past_the_dense_cap() {
    small_prob_cap();
    let mut circuit = known_marginals_circuit(N);
    for q in 0..N {
        circuit.add_measure(q, q);
    }
    let result = simulate(&circuit)
        .distributed(DistributedContext::serial())
        .seed(SEED)
        .shots(16)
        .expect("shots do not need the dense gather");
    assert_eq!(result.shots.len(), 16);
    for shot in &result.shots {
        assert!(shot[0], "qubit 0 is held at |1> in every shot");
    }
}

#[test]
fn marginals_answer_past_the_dense_cap() {
    small_prob_cap();
    let marginals = simulate(&known_marginals_circuit(0))
        .distributed(DistributedContext::serial())
        .seed(SEED)
        .marginals()
        .expect("marginals come from per-qubit expectations, not the dense vector")
        .into_vec();
    assert_eq!(marginals.len(), N);
    assert!(marginals[0].0.abs() < 1e-12 && (marginals[0].1 - 1.0).abs() < 1e-12);
    for (q, m) in marginals.iter().enumerate().skip(1) {
        assert!(
            (m.0 - 0.5).abs() < 1e-12 && (m.1 - 0.5).abs() < 1e-12,
            "qubit {q}: expected an even split, got {m:?}"
        );
    }
}
