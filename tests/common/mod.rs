//! Shared helpers for the cross-backend correctness tests.
//!
//! Each per-backend test file declares `mod common;` and imports the
//! helpers it needs. Tolerance constants are split by backend so the
//! same helper can be reused with the right precision.

#![allow(dead_code)]

use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::sim;

pub const SV_EPS: f64 = 1e-10;
pub const STAB_EPS: f64 = 1e-12;
pub const MPS_EPS: f64 = 1e-9;
pub const TN_EPS: f64 = 1e-10;

pub const SEED: u64 = 42;

pub fn sv_reference_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(SEED);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

pub fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: probability vector length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < eps,
            "{label} prob[{i}]: expected {e:.12}, got {a:.12} (diff {diff:.2e}, eps {eps:.0e})"
        );
    }
}

pub fn assert_backend_matches_sv<B: Backend>(
    backend: &mut B,
    circuit: &Circuit,
    eps: f64,
    label: &str,
) {
    sim::run_on(backend, circuit).unwrap();
    let actual = backend.probabilities().unwrap();
    let expected = sv_reference_probs(circuit);
    assert_probs_close(&actual, &expected, eps, label);
}

pub fn is_clifford(circuit: &Circuit) -> bool {
    for instr in &circuit.instructions {
        match instr {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                if !gate.is_clifford() {
                    return false;
                }
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Barrier { .. } => {}
        }
    }
    true
}
