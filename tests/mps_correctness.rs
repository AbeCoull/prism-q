//! Cross-backend correctness tests for `MpsBackend`.
//!
//! Bond dimension is sized to the circuit so MPS represents the full state
//! exactly: `bond_dim = max(2^(n/2), 64)`. Tests cross-validate against the
//! Statevector reference and against the unfused apply loop on the same
//! backend at sizes ≤ 16q.

mod common;

use common::{assert_backend_matches_sv, assert_fused_matches_unfused, MPS_EPS, SEED};
use prism_q::backend::mps::MpsBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits;

fn bond_dim_for(n: usize) -> usize {
    let exact = 1usize << (n / 2);
    exact.max(64)
}

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let bd = bond_dim_for(circuit.num_qubits);
    let mut backend = MpsBackend::new(SEED, bd);
    assert_backend_matches_sv(&mut backend, circuit, MPS_EPS, label);
}

fn check_fused_vs_unfused(label: &str, circuit: &Circuit) {
    let bd = bond_dim_for(circuit.num_qubits);
    assert_fused_matches_unfused(|| MpsBackend::new(SEED, bd), circuit, MPS_EPS, label);
}

// ===== qft =====

#[test]
fn mps_qft_4q_sv() {
    check_sv_cross("qft 4q sv", &circuits::qft_circuit(4));
}

#[test]
fn mps_qft_8q_sv() {
    check_sv_cross("qft 8q sv", &circuits::qft_circuit(8));
}

#[test]
fn mps_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

#[test]
fn mps_qft_12q_fused() {
    check_fused_vs_unfused("qft 12q fused", &circuits::qft_circuit(12));
}

// QFT at 16q+ is a known SVD-truncation stress case for MPS:
// the cphase chain accumulates ~7e-6 truncation error even at bond_dim
// 256, well above the 1e-9 cross-backend tolerance. Smaller QFT sizes
// validate the same code paths without crossing that wall. The fusion
// path uses bubble routing (apply_batch_phase_bubble), whose SVD chain
// also differs from the unfused per-phase path, so even a same-backend
// fused-vs-unfused check at 16q is not guaranteed to match at 1e-9.

// ===== random =====

#[test]
fn mps_random_4q_sv() {
    check_sv_cross("random 4q d10 sv", &circuits::random_circuit(4, 10, SEED));
}

#[test]
fn mps_random_8q_sv() {
    check_sv_cross("random 8q d10 sv", &circuits::random_circuit(8, 10, SEED));
}

#[test]
fn mps_random_12q_fused() {
    check_fused_vs_unfused(
        "random 12q d5 fused",
        &circuits::random_circuit(12, 5, SEED),
    );
}

#[test]
fn mps_random_16q_sv() {
    check_sv_cross("random 16q d5 sv", &circuits::random_circuit(16, 5, SEED));
}

// ===== hardware_efficient_ansatz =====

#[test]
fn mps_hea_4q_sv() {
    check_sv_cross(
        "hea 4q l3 sv",
        &circuits::hardware_efficient_ansatz(4, 3, SEED),
    );
}

#[test]
fn mps_hea_12q_fused() {
    check_fused_vs_unfused(
        "hea 12q l2 fused",
        &circuits::hardware_efficient_ansatz(12, 2, SEED),
    );
}

#[test]
fn mps_hea_16q_sv() {
    check_sv_cross(
        "hea 16q l2 sv",
        &circuits::hardware_efficient_ansatz(16, 2, SEED),
    );
}

// ===== ghz =====

#[test]
fn mps_ghz_4q_sv() {
    check_sv_cross("ghz 4q sv", &circuits::ghz_circuit(4));
}

#[test]
fn mps_ghz_12q_fused() {
    check_fused_vs_unfused("ghz 12q fused", &circuits::ghz_circuit(12));
}

#[test]
fn mps_ghz_16q_sv() {
    check_sv_cross("ghz 16q sv", &circuits::ghz_circuit(16));
}

// ===== qaoa =====

#[test]
fn mps_qaoa_4q_sv() {
    check_sv_cross("qaoa 4q l2 sv", &circuits::qaoa_circuit(4, 2, SEED));
}

#[test]
fn mps_qaoa_12q_fused() {
    check_fused_vs_unfused("qaoa 12q l2 fused", &circuits::qaoa_circuit(12, 2, SEED));
}

// ===== clifford_heavy =====

#[test]
fn mps_clifford_8q_sv() {
    check_sv_cross(
        "clifford_heavy 8q d10 sv",
        &circuits::clifford_heavy_circuit(8, 10, SEED),
    );
}

#[test]
fn mps_clifford_12q_fused() {
    check_fused_vs_unfused(
        "clifford_heavy 12q d10 fused",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

// ===== phase_estimation =====

#[test]
fn mps_qpe_4q_sv() {
    check_sv_cross("qpe 4q sv", &circuits::phase_estimation_circuit(4));
}

#[test]
fn mps_qpe_8q_sv() {
    check_sv_cross("qpe 8q sv", &circuits::phase_estimation_circuit(8));
}

#[test]
fn mps_qpe_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}

// ===== w_state =====

#[test]
fn mps_w_state_4q_sv() {
    check_sv_cross("w_state 4q sv", &circuits::w_state_circuit(4));
}

#[test]
fn mps_w_state_8q_fused() {
    check_fused_vs_unfused("w_state 8q fused", &circuits::w_state_circuit(8));
}

// ===== quantum_volume =====

#[test]
fn mps_qv_4q_sv() {
    check_sv_cross(
        "quantum_volume 4q d2 sv",
        &circuits::quantum_volume_circuit(4, 2, SEED),
    );
}

#[test]
fn mps_qv_8q_fused() {
    check_fused_vs_unfused(
        "quantum_volume 8q d2 fused",
        &circuits::quantum_volume_circuit(8, 2, SEED),
    );
}

// ===== single_qubit_rotation =====

#[test]
fn mps_single_qubit_rotation_12q_sv() {
    check_sv_cross(
        "single_qubit_rotation 12q d5 sv",
        &circuits::single_qubit_rotation_circuit(12, 5, SEED),
    );
}

// ===== cz_chain =====

#[test]
fn mps_cz_chain_12q_fused() {
    check_fused_vs_unfused(
        "cz_chain 12q d3 fused",
        &circuits::cz_chain_circuit(12, 3, SEED),
    );
}
