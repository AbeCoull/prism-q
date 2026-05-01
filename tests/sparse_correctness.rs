//! Cross-backend correctness tests for `SparseBackend`.
//!
//! Sparse stores `HashMap<usize, Complex64>` and supports the full
//! gate set. Tests cross-validate against the Statevector reference
//! and against the unfused apply loop on the same backend, across
//! eight circuit families at sizes `≤ 12q`.

mod common;

use common::{assert_backend_matches_sv, assert_fused_matches_unfused, SEED, SPARSE_EPS};
use prism_q::backend::sparse::SparseBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = SparseBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, SPARSE_EPS, label);
}

fn check_fused_vs_unfused(label: &str, circuit: &Circuit) {
    assert_fused_matches_unfused(|| SparseBackend::new(SEED), circuit, SPARSE_EPS, label);
}

// ===== qft =====

#[test]
fn sparse_qft_4q_sv() {
    check_sv_cross("qft 4q sv", &circuits::qft_circuit(4));
}

#[test]
fn sparse_qft_4q_fused() {
    check_fused_vs_unfused("qft 4q fused", &circuits::qft_circuit(4));
}

#[test]
fn sparse_qft_8q_sv() {
    check_sv_cross("qft 8q sv", &circuits::qft_circuit(8));
}

#[test]
fn sparse_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

#[test]
fn sparse_qft_12q_fused() {
    check_fused_vs_unfused("qft 12q fused", &circuits::qft_circuit(12));
}

// ===== random =====

#[test]
fn sparse_random_4q_sv() {
    check_sv_cross("random 4q d10 sv", &circuits::random_circuit(4, 10, SEED));
}

#[test]
fn sparse_random_8q_sv() {
    check_sv_cross("random 8q d10 sv", &circuits::random_circuit(8, 10, SEED));
}

#[test]
fn sparse_random_12q_sv() {
    check_sv_cross("random 12q d10 sv", &circuits::random_circuit(12, 10, SEED));
}

#[test]
fn sparse_random_12q_fused() {
    check_fused_vs_unfused(
        "random 12q d10 fused",
        &circuits::random_circuit(12, 10, SEED),
    );
}

// ===== hardware_efficient_ansatz =====

#[test]
fn sparse_hea_4q_sv() {
    check_sv_cross(
        "hea 4q l3 sv",
        &circuits::hardware_efficient_ansatz(4, 3, SEED),
    );
}

#[test]
fn sparse_hea_12q_sv() {
    check_sv_cross(
        "hea 12q l3 sv",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

#[test]
fn sparse_hea_12q_fused() {
    check_fused_vs_unfused(
        "hea 12q l3 fused",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

// ===== ghz =====

#[test]
fn sparse_ghz_4q_sv() {
    check_sv_cross("ghz 4q sv", &circuits::ghz_circuit(4));
}

#[test]
fn sparse_ghz_8q_sv() {
    check_sv_cross("ghz 8q sv", &circuits::ghz_circuit(8));
}

#[test]
fn sparse_ghz_12q_sv() {
    check_sv_cross("ghz 12q sv", &circuits::ghz_circuit(12));
}

#[test]
fn sparse_ghz_12q_fused() {
    check_fused_vs_unfused("ghz 12q fused", &circuits::ghz_circuit(12));
}

// ===== qaoa =====

#[test]
fn sparse_qaoa_4q_sv() {
    check_sv_cross("qaoa 4q l3 sv", &circuits::qaoa_circuit(4, 3, SEED));
}

#[test]
fn sparse_qaoa_12q_sv() {
    check_sv_cross("qaoa 12q l3 sv", &circuits::qaoa_circuit(12, 3, SEED));
}

#[test]
fn sparse_qaoa_12q_fused() {
    check_fused_vs_unfused("qaoa 12q l3 fused", &circuits::qaoa_circuit(12, 3, SEED));
}

// ===== clifford_heavy =====

#[test]
fn sparse_clifford_8q_sv() {
    check_sv_cross(
        "clifford_heavy 8q d10 sv",
        &circuits::clifford_heavy_circuit(8, 10, SEED),
    );
}

#[test]
fn sparse_clifford_12q_fused() {
    check_fused_vs_unfused(
        "clifford_heavy 12q d10 fused",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

// ===== phase_estimation =====

#[test]
fn sparse_qpe_4q_sv() {
    check_sv_cross("qpe 4q sv", &circuits::phase_estimation_circuit(4));
}

#[test]
fn sparse_qpe_8q_sv() {
    check_sv_cross("qpe 8q sv", &circuits::phase_estimation_circuit(8));
}

#[test]
fn sparse_qpe_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}

// ===== cz_chain =====

#[test]
fn sparse_cz_chain_8q_sv() {
    check_sv_cross("cz_chain 8q d5 sv", &circuits::cz_chain_circuit(8, 5, SEED));
}

#[test]
fn sparse_cz_chain_12q_fused() {
    check_fused_vs_unfused(
        "cz_chain 12q d5 fused",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}
