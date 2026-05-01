//! Cross-backend correctness tests for `TensorNetworkBackend`.
//!
//! Deferred contraction with greedy min-size heuristic. Tests cross-validate
//! against the Statevector reference at sizes ≤ 16q (the backend's
//! `MAX_PROB_QUBITS = 25` cap is well above the slowest contraction covered
//! by the unit-test suite).

mod common;

use common::{assert_backend_matches_sv, SEED, TN_EPS};
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = TensorNetworkBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, TN_EPS, label);
}

// ===== qft =====

#[test]
fn tn_qft_4q_sv() {
    check_sv_cross("qft 4q sv", &circuits::qft_circuit(4));
}

#[test]
fn tn_qft_8q_sv() {
    check_sv_cross("qft 8q sv", &circuits::qft_circuit(8));
}

#[test]
fn tn_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

// ===== random =====

#[test]
fn tn_random_4q_sv() {
    check_sv_cross("random 4q d10 sv", &circuits::random_circuit(4, 10, SEED));
}

#[test]
fn tn_random_8q_sv() {
    check_sv_cross("random 8q d10 sv", &circuits::random_circuit(8, 10, SEED));
}

#[test]
fn tn_random_12q_sv() {
    check_sv_cross("random 12q d5 sv", &circuits::random_circuit(12, 5, SEED));
}

// ===== hardware_efficient_ansatz =====

#[test]
fn tn_hea_4q_sv() {
    check_sv_cross(
        "hea 4q l3 sv",
        &circuits::hardware_efficient_ansatz(4, 3, SEED),
    );
}

#[test]
fn tn_hea_12q_sv() {
    check_sv_cross(
        "hea 12q l2 sv",
        &circuits::hardware_efficient_ansatz(12, 2, SEED),
    );
}

// ===== ghz =====

#[test]
fn tn_ghz_4q_sv() {
    check_sv_cross("ghz 4q sv", &circuits::ghz_circuit(4));
}

#[test]
fn tn_ghz_12q_sv() {
    check_sv_cross("ghz 12q sv", &circuits::ghz_circuit(12));
}

#[test]
fn tn_ghz_16q_sv() {
    check_sv_cross("ghz 16q sv", &circuits::ghz_circuit(16));
}

// ===== qaoa =====

#[test]
fn tn_qaoa_4q_sv() {
    check_sv_cross("qaoa 4q l2 sv", &circuits::qaoa_circuit(4, 2, SEED));
}

#[test]
fn tn_qaoa_8q_sv() {
    check_sv_cross("qaoa 8q l2 sv", &circuits::qaoa_circuit(8, 2, SEED));
}

// ===== clifford_heavy =====

#[test]
fn tn_clifford_8q_sv() {
    check_sv_cross(
        "clifford_heavy 8q d10 sv",
        &circuits::clifford_heavy_circuit(8, 10, SEED),
    );
}

#[test]
fn tn_clifford_12q_sv() {
    check_sv_cross(
        "clifford_heavy 12q d10 sv",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

// ===== phase_estimation =====

#[test]
fn tn_qpe_4q_sv() {
    check_sv_cross("qpe 4q sv", &circuits::phase_estimation_circuit(4));
}

#[test]
fn tn_qpe_8q_sv() {
    check_sv_cross("qpe 8q sv", &circuits::phase_estimation_circuit(8));
}

// ===== w_state =====

#[test]
fn tn_w_state_4q_sv() {
    check_sv_cross("w_state 4q sv", &circuits::w_state_circuit(4));
}

#[test]
fn tn_w_state_8q_sv() {
    check_sv_cross("w_state 8q sv", &circuits::w_state_circuit(8));
}

// ===== single_qubit_rotation =====

#[test]
fn tn_single_qubit_rotation_8q_sv() {
    check_sv_cross(
        "single_qubit_rotation 8q d5 sv",
        &circuits::single_qubit_rotation_circuit(8, 5, SEED),
    );
}

#[test]
fn tn_single_qubit_rotation_16q_sv() {
    check_sv_cross(
        "single_qubit_rotation 16q d3 sv",
        &circuits::single_qubit_rotation_circuit(16, 3, SEED),
    );
}

// ===== cz_chain =====

#[test]
fn tn_cz_chain_8q_sv() {
    check_sv_cross("cz_chain 8q d5 sv", &circuits::cz_chain_circuit(8, 5, SEED));
}

#[test]
fn tn_cz_chain_12q_sv() {
    check_sv_cross(
        "cz_chain 12q d3 sv",
        &circuits::cz_chain_circuit(12, 3, SEED),
    );
}

// ===== independent_bell_pairs =====

#[test]
fn tn_bell_pairs_4q_sv() {
    check_sv_cross("bell_pairs 4q sv", &circuits::independent_bell_pairs(2));
}

#[test]
fn tn_bell_pairs_12q_sv() {
    check_sv_cross("bell_pairs 12q sv", &circuits::independent_bell_pairs(6));
}
