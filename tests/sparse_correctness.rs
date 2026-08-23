//! Cross-backend correctness for `SparseBackend` against the statevector
//! reference and the unfused apply loop, across nine circuit families at
//! sizes up to 12q.

mod common;

use common::{SEED, SPARSE_EPS, assert_backend_matches_sv, assert_fused_matches_unfused};
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
fn sparse_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

#[test]
fn sparse_qft_12q_fused() {
    check_fused_vs_unfused("qft 12q fused", &circuits::qft_circuit(12));
}

// ===== random =====

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
fn sparse_qpe_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}

// ===== cz_chain =====

#[test]
fn sparse_cz_chain_12q_fused() {
    check_fused_vs_unfused(
        "cz_chain 12q d5 fused",
        &circuits::cz_chain_circuit(12, 5, SEED),
    );
}

// ===== monomial fused 4x4 =====

// Interference layers after the fused block make the final probabilities
// sensitive to every monomial phase, not just the permutation.
fn monomial_rich_circuit(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(prism_q::gates::Gate::H, &[q]);
    }
    c.add_barrier(&(0..n).collect::<Vec<_>>());
    for pair in 0..(n / 2) {
        let a = 2 * pair;
        let b = a + 1;
        c.add_gate(prism_q::gates::Gate::S, &[a]);
        c.add_gate(prism_q::gates::Gate::Cx, &[a, b]);
        c.add_gate(prism_q::gates::Gate::Z, &[b]);
        c.add_gate(prism_q::gates::Gate::X, &[a]);
        c.add_gate(prism_q::gates::Gate::Cx, &[b, a]);
    }
    c.add_barrier(&(0..n).collect::<Vec<_>>());
    for q in 0..n {
        c.add_gate(prism_q::gates::Gate::H, &[q]);
    }
    c
}

#[test]
fn sparse_monomial_2q_12q_sv() {
    check_sv_cross("monomial 2q 12q sv", &monomial_rich_circuit(12));
}

#[test]
fn sparse_monomial_2q_12q_fused() {
    check_fused_vs_unfused("monomial 2q 12q fused", &monomial_rich_circuit(12));
}
