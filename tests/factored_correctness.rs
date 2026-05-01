//! Cross-backend correctness tests for `FactoredBackend`.
//!
//! Dynamic split-state backend: handles both monolithic 2^n state vectors
//! and independent subsystems split into blocks. Each of the 16 builders
//! in `prism_q::circuits` runs through both an SV cross check and a
//! fused-vs-unfused self-check on the factored backend at moderate
//! sizes (8-12q). A second tier exercises blocked builders at 20q where
//! the factored backend's split-state path is exercised, and monolithic
//! builders at 16q to cover the larger fusion pipeline.

mod common;

use common::{assert_backend_matches_sv, assert_fused_matches_unfused, FACTORED_EPS, SEED};
use prism_q::backend::factored::FactoredBackend;
use prism_q::circuit::{Circuit, Instruction};
use prism_q::circuits;
use prism_q::gates::Gate;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = FactoredBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, FACTORED_EPS, label);
}

fn check_fused_vs_unfused(label: &str, circuit: &Circuit) {
    assert_fused_matches_unfused(|| FactoredBackend::new(SEED), circuit, FACTORED_EPS, label);
}

fn has_batch_phase(circuit: &Circuit) -> bool {
    let fused = prism_q::circuit::fusion::fuse_circuit(circuit, true);
    fused.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::BatchPhase(_),
                ..
            }
        )
    })
}

// ===== qft =====

#[test]
fn factored_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

#[test]
fn factored_qft_12q_fused() {
    check_fused_vs_unfused("qft 12q fused", &circuits::qft_circuit(12));
}

#[test]
fn factored_qft_16q_sv() {
    check_sv_cross("qft 16q sv", &circuits::qft_circuit(16));
}

#[test]
fn factored_batch_phase_phase_sensitive_16q() {
    let mut c = Circuit::new(16, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::cphase(0.37), &[0, 1]);
    c.add_gate(Gate::cphase(-0.91), &[0, 2]);
    c.add_gate(Gate::H, &[0]);
    assert!(
        has_batch_phase(&c),
        "phase-sensitive 16q circuit should use BatchPhase fusion"
    );
    check_sv_cross("batch_phase phase-sensitive 16q sv", &c);
    check_fused_vs_unfused("batch_phase phase-sensitive 16q fused", &c);
}

// ===== random =====

#[test]
fn factored_random_12q_sv() {
    check_sv_cross("random 12q d10 sv", &circuits::random_circuit(12, 10, SEED));
}

#[test]
fn factored_random_12q_fused() {
    check_fused_vs_unfused(
        "random 12q d10 fused",
        &circuits::random_circuit(12, 10, SEED),
    );
}

// ===== hardware_efficient_ansatz =====

#[test]
fn factored_hea_12q_sv() {
    check_sv_cross(
        "hea 12q l3 sv",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

#[test]
fn factored_hea_12q_fused() {
    check_fused_vs_unfused(
        "hea 12q l3 fused",
        &circuits::hardware_efficient_ansatz(12, 3, SEED),
    );
}

#[test]
fn factored_hea_16q_sv() {
    check_sv_cross(
        "hea 16q l2 sv",
        &circuits::hardware_efficient_ansatz(16, 2, SEED),
    );
}

// ===== clifford_heavy =====

#[test]
fn factored_clifford_heavy_12q_sv() {
    check_sv_cross(
        "clifford_heavy 12q d10 sv",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

#[test]
fn factored_clifford_heavy_12q_fused() {
    check_fused_vs_unfused(
        "clifford_heavy 12q d10 fused",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

// ===== clifford_random_pairs =====

#[test]
fn factored_clifford_random_pairs_12q_sv() {
    check_sv_cross(
        "clifford_random_pairs 12q d10 sv",
        &circuits::clifford_random_pairs(12, 10, SEED),
    );
}

#[test]
fn factored_clifford_random_pairs_12q_fused() {
    check_fused_vs_unfused(
        "clifford_random_pairs 12q d10 fused",
        &circuits::clifford_random_pairs(12, 10, SEED),
    );
}

// ===== independent_bell_pairs =====

#[test]
fn factored_bell_pairs_12q_sv() {
    check_sv_cross("bell_pairs 6 sv", &circuits::independent_bell_pairs(6));
}

#[test]
fn factored_bell_pairs_12q_fused() {
    check_fused_vs_unfused("bell_pairs 6 fused", &circuits::independent_bell_pairs(6));
}

#[test]
fn factored_bell_pairs_20q_sv() {
    check_sv_cross("bell_pairs 10 sv", &circuits::independent_bell_pairs(10));
}

#[test]
fn factored_bell_pairs_20q_fused() {
    check_fused_vs_unfused("bell_pairs 10 fused", &circuits::independent_bell_pairs(10));
}

// ===== independent_random_blocks =====

#[test]
fn factored_random_blocks_12q_sv() {
    check_sv_cross(
        "random_blocks 4x3 d5 sv",
        &circuits::independent_random_blocks(4, 3, 5, SEED),
    );
}

#[test]
fn factored_random_blocks_12q_fused() {
    check_fused_vs_unfused(
        "random_blocks 4x3 d5 fused",
        &circuits::independent_random_blocks(4, 3, 5, SEED),
    );
}

#[test]
fn factored_random_blocks_20q_sv() {
    check_sv_cross(
        "random_blocks 5x4 d5 sv",
        &circuits::independent_random_blocks(5, 4, 5, SEED),
    );
}

#[test]
fn factored_random_blocks_20q_fused() {
    check_fused_vs_unfused(
        "random_blocks 5x4 d5 fused",
        &circuits::independent_random_blocks(5, 4, 5, SEED),
    );
}

// ===== ghz =====

#[test]
fn factored_ghz_12q_sv() {
    check_sv_cross("ghz 12q sv", &circuits::ghz_circuit(12));
}

#[test]
fn factored_ghz_12q_fused() {
    check_fused_vs_unfused("ghz 12q fused", &circuits::ghz_circuit(12));
}

// ===== qaoa =====

#[test]
fn factored_qaoa_12q_sv() {
    check_sv_cross("qaoa 12q l3 sv", &circuits::qaoa_circuit(12, 3, SEED));
}

#[test]
fn factored_qaoa_12q_fused() {
    check_fused_vs_unfused("qaoa 12q l3 fused", &circuits::qaoa_circuit(12, 3, SEED));
}

#[test]
fn factored_qaoa_16q_sv() {
    check_sv_cross("qaoa 16q l3 sv", &circuits::qaoa_circuit(16, 3, SEED));
}

// ===== single_qubit_rotation =====

#[test]
fn factored_single_qubit_rotation_12q_sv() {
    check_sv_cross(
        "single_qubit_rotation 12q d5 sv",
        &circuits::single_qubit_rotation_circuit(12, 5, SEED),
    );
}

#[test]
fn factored_single_qubit_rotation_12q_fused() {
    check_fused_vs_unfused(
        "single_qubit_rotation 12q d5 fused",
        &circuits::single_qubit_rotation_circuit(12, 5, SEED),
    );
}

// ===== clifford_t =====

#[test]
fn factored_clifford_t_12q_sv() {
    check_sv_cross(
        "clifford_t 12q d10 t=0.2 sv",
        &circuits::clifford_t_circuit(12, 10, 0.2, SEED),
    );
}

#[test]
fn factored_clifford_t_12q_fused() {
    check_fused_vs_unfused(
        "clifford_t 12q d10 t=0.2 fused",
        &circuits::clifford_t_circuit(12, 10, 0.2, SEED),
    );
}

// ===== w_state =====

#[test]
fn factored_w_state_8q_sv() {
    check_sv_cross("w_state 8q sv", &circuits::w_state_circuit(8));
}

#[test]
fn factored_w_state_8q_fused() {
    check_fused_vs_unfused("w_state 8q fused", &circuits::w_state_circuit(8));
}

// ===== quantum_volume =====

#[test]
fn factored_quantum_volume_8q_sv() {
    check_sv_cross(
        "quantum_volume 8q d2 sv",
        &circuits::quantum_volume_circuit(8, 2, SEED),
    );
}

#[test]
fn factored_quantum_volume_12q_fused() {
    check_fused_vs_unfused(
        "quantum_volume 12q d1 fused",
        &circuits::quantum_volume_circuit(12, 1, SEED),
    );
}

// ===== local_clifford_blocks =====

#[test]
fn factored_local_clifford_blocks_12q_sv() {
    check_sv_cross(
        "local_clifford_blocks 4x3 d10 sv",
        &circuits::local_clifford_blocks(4, 3, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_12q_fused() {
    check_fused_vs_unfused(
        "local_clifford_blocks 4x3 d10 fused",
        &circuits::local_clifford_blocks(4, 3, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_20q_sv() {
    check_sv_cross(
        "local_clifford_blocks 5x4 d10 sv",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

#[test]
fn factored_local_clifford_blocks_20q_fused() {
    check_fused_vs_unfused(
        "local_clifford_blocks 5x4 d10 fused",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

// ===== cz_chain =====

#[test]
fn factored_cz_chain_12q_sv() {
    check_sv_cross(
        "cz_chain 12q d5 sv",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}

#[test]
fn factored_cz_chain_12q_fused() {
    check_fused_vs_unfused(
        "cz_chain 12q d5 fused",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}

// ===== phase_estimation =====

#[test]
fn factored_phase_estimation_8q_sv() {
    check_sv_cross("qpe 8q sv", &circuits::phase_estimation_circuit(8));
}

#[test]
fn factored_phase_estimation_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}
