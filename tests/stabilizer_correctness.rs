//! Cross-backend correctness tests for `StabilizerBackend`.
//!
//! Each test compares stabilizer probabilities against the
//! `StatevectorBackend` reference. Only Clifford-eligible circuit
//! families are exercised; non-Clifford gates (T, Tdg, parametric
//! rotations, controlled-phase) would fall outside the stabilizer
//! formalism and are caught by the `is_clifford` guard in `check`.

mod common;

use common::{assert_backend_matches_sv, is_clifford, STAB_EPS};
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::circuits;

const SEED: u64 = 42;

fn check(label: &str, circuit: &prism_q::circuit::Circuit) {
    assert!(
        is_clifford(circuit),
        "{label}: circuit contains non-Clifford gates; \
         this test must only be called on Clifford circuits"
    );
    let mut backend = StabilizerBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, STAB_EPS, label);
}

// ===== clifford_heavy =====

#[test]
fn stabilizer_clifford_heavy_4q() {
    check(
        "clifford_heavy 4q d10",
        &circuits::clifford_heavy_circuit(4, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_8q() {
    check(
        "clifford_heavy 8q d10",
        &circuits::clifford_heavy_circuit(8, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_12q() {
    check(
        "clifford_heavy 12q d10",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_16q() {
    check(
        "clifford_heavy 16q d10",
        &circuits::clifford_heavy_circuit(16, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_20q() {
    check(
        "clifford_heavy 20q d10",
        &circuits::clifford_heavy_circuit(20, 10, SEED),
    );
}

// ===== clifford_random_pairs =====

#[test]
fn stabilizer_clifford_random_pairs_8q() {
    check(
        "clifford_random_pairs 8q d10",
        &circuits::clifford_random_pairs(8, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_random_pairs_16q() {
    check(
        "clifford_random_pairs 16q d10",
        &circuits::clifford_random_pairs(16, 10, SEED),
    );
}

// ===== ghz =====

#[test]
fn stabilizer_ghz_4q() {
    check("ghz 4q", &circuits::ghz_circuit(4));
}

#[test]
fn stabilizer_ghz_12q() {
    check("ghz 12q", &circuits::ghz_circuit(12));
}

#[test]
fn stabilizer_ghz_20q() {
    check("ghz 20q", &circuits::ghz_circuit(20));
}

// ===== bell_pairs =====

#[test]
fn stabilizer_bell_pairs_4q() {
    check("bell_pairs 4q", &circuits::independent_bell_pairs(2));
}

#[test]
fn stabilizer_bell_pairs_12q() {
    check("bell_pairs 12q", &circuits::independent_bell_pairs(6));
}

#[test]
fn stabilizer_bell_pairs_20q() {
    check("bell_pairs 20q", &circuits::independent_bell_pairs(10));
}

// ===== local_clifford_blocks =====

#[test]
fn stabilizer_local_clifford_blocks_12q() {
    check(
        "local_clifford_blocks 3x4q d10",
        &circuits::local_clifford_blocks(3, 4, 10, SEED),
    );
}

#[test]
fn stabilizer_local_clifford_blocks_16q() {
    check(
        "local_clifford_blocks 4x4q d10",
        &circuits::local_clifford_blocks(4, 4, 10, SEED),
    );
}

#[test]
fn stabilizer_local_clifford_blocks_20q() {
    check(
        "local_clifford_blocks 5x4q d10",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

// ===== applicability gate sanity =====

#[test]
fn is_clifford_rejects_t_gate() {
    let qft = circuits::qft_circuit(4);
    assert!(
        !is_clifford(&qft),
        "QFT should be flagged non-Clifford (uses controlled-phase)"
    );
}

#[test]
fn is_clifford_accepts_clifford_heavy() {
    let c = circuits::clifford_heavy_circuit(4, 10, SEED);
    assert!(is_clifford(&c), "clifford_heavy must be all-Clifford");
}
