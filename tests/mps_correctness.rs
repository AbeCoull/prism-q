//! Cross-backend correctness for `MpsBackend` against the statevector
//! reference and the unfused apply loop at sizes up to 16q. Bond dimension is
//! `max(2^(n/2), 64)` so the MPS holds the full state exactly.

mod common;

use common::{MPS_EPS, SEED, assert_backend_matches_sv, assert_fused_matches_unfused};
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
fn mps_ghz_12q_fused() {
    check_fused_vs_unfused("ghz 12q fused", &circuits::ghz_circuit(12));
}

#[test]
fn mps_ghz_16q_sv() {
    check_sv_cross("ghz 16q sv", &circuits::ghz_circuit(16));
}

// ===== qaoa =====

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
fn mps_qpe_12q_fused() {
    check_fused_vs_unfused("qpe 12q fused", &circuits::phase_estimation_circuit(12));
}

// ===== w_state =====

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

// ===== dense expansion =====

// The dense expansion splits the top sites across Rayon tasks once the chain
// is deeper than the per-task leaf floor, so 6 qubits take the serial walk and
// 7 and up take the split. Both must agree with the statevector.
#[test]
fn mps_dense_expansion_across_the_parallel_split() {
    for n in [6usize, 7, 8, 13] {
        check_sv_cross(
            &format!("dense expansion {n}q"),
            &circuits::random_circuit(n, 4, SEED),
        );
    }
}

// SWAP routing permutes the site layout, so the basis bit a site decides is no
// longer the site's own index. Long-range CX force that permutation, and
// export_statevector orders amplitudes by logical qubit, not by site.
#[test]
fn mps_dense_expansion_survives_swap_routing() {
    use prism_q::backend::Backend;
    use prism_q::gates::Gate;

    let n = 8;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::Ry(0.3 + 0.2 * q as f64), &[q]);
    }
    for (control, target) in [(0usize, 7usize), (1, 6), (2, 5), (0, 4)] {
        circuit.add_gate(Gate::Cx, &[control, target]);
    }
    circuit.add_gate(Gate::T, &[3]);

    let mut backend = MpsBackend::new(SEED, bond_dim_for(n));
    assert_backend_matches_sv(&mut backend, &circuit, MPS_EPS, "swap-routed expansion");

    let amplitudes = backend.export_statevector().unwrap();
    let probs = backend.probabilities().unwrap();
    assert_eq!(amplitudes.len(), probs.len());
    for (basis, amp) in amplitudes.iter().enumerate() {
        assert!(
            (amp.norm_sqr() - probs[basis]).abs() < MPS_EPS,
            "basis {basis}: |amp|^2 {} vs probability {}",
            amp.norm_sqr(),
            probs[basis]
        );
    }
}
