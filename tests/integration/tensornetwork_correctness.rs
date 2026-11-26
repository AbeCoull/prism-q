//! Cross-backend correctness for `TensorNetworkBackend` against the
//! statevector reference at sizes up to 16q, below the backend's
//! memory-derived dense probability cap (`PRISM_MAX_PROB_QUBITS`).

use crate::common;

use common::{SEED, TN_EPS, assert_backend_matches_sv};
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits;

fn check_sv_cross(label: &str, circuit: &Circuit) {
    let mut backend = TensorNetworkBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, TN_EPS, label);
}

// ===== qft =====

#[test]
fn tn_qft_12q_sv() {
    check_sv_cross("qft 12q sv", &circuits::qft_circuit(12));
}

// ===== random =====

#[test]
fn tn_random_12q_sv() {
    check_sv_cross("random 12q d5 sv", &circuits::random_circuit(12, 5, SEED));
}

// ===== hardware_efficient_ansatz =====

#[test]
fn tn_hea_12q_sv() {
    check_sv_cross(
        "hea 12q l2 sv",
        &circuits::hardware_efficient_ansatz(12, 2, SEED),
    );
}

// ===== ghz =====

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

// ===== w_state =====

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

#[test]
fn tn_pauli_expectations_and_identity_match_sv() {
    use num_complex::Complex64;
    use prism_q::backend::Backend;
    use prism_q::backend::statevector::StatevectorBackend;
    use prism_q::gates::Gate;
    use prism_q::sim::unified_pauli::PauliTerm;

    let mut circuit = Circuit::new(8, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Ry(0.6), &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_gate(Gate::H, &[4]);
    circuit.add_gate(Gate::Cz, &[4, 5]);
    circuit.add_gate(Gate::Rx(1.1), &[6]);
    circuit.add_gate(Gate::Cx, &[6, 7]);

    let observables = vec![
        Vec::new(),
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(0), PauliTerm::x(1)],
        vec![PauliTerm::z(2), PauliTerm::y(5)],
        vec![PauliTerm::z(7)],
    ];

    let mut sv = StatevectorBackend::new(SEED);
    sv.init(8, 0).unwrap();
    for inst in &circuit.instructions {
        sv.apply(inst).unwrap();
    }
    let sv_exp = sv.pauli_expectations(&observables).unwrap();

    let mut tn = TensorNetworkBackend::new(SEED);
    tn.init(8, 0).unwrap();
    for inst in &circuit.instructions {
        tn.apply(inst).unwrap();
    }
    let tn_exp = tn.pauli_expectations(&observables).unwrap();

    assert!((tn_exp[0] - 1.0).abs() < TN_EPS);
    for (got, want) in tn_exp.iter().zip(&sv_exp) {
        assert!((got - want).abs() < TN_EPS, "{got} vs {want}");
    }

    // Non-unitary Kraus-style 2q gate and 1q shrink still match expected normalized state.
    let zero = Complex64::new(0.0, 0.0);
    let kraus_2q = [
        [Complex64::new(0.5, 0.0), zero, zero, zero],
        [zero, Complex64::new(0.5, 0.0), zero, zero],
        [zero, zero, zero, zero],
        [zero, zero, zero, zero],
    ];
    let mut kraus_tn = TensorNetworkBackend::new(SEED);
    kraus_tn.init(3, 0).unwrap();
    kraus_tn
        .apply(&prism_q::circuit::Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
    kraus_tn
        .apply(&prism_q::circuit::Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![1],
        })
        .unwrap();
    kraus_tn
        .apply(&prism_q::circuit::Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec::smallvec![1, 2],
        })
        .unwrap();
    kraus_tn
        .apply(&prism_q::circuit::Instruction::Gate {
            gate: Gate::Fused2q(Box::new(kraus_2q)),
            targets: smallvec::smallvec![0, 1],
        })
        .unwrap();
    let k_exp = kraus_tn
        .pauli_expectations(&[
            vec![PauliTerm::z(0)],
            vec![PauliTerm::z(1)],
            vec![PauliTerm::z(1), PauliTerm::z(2)],
        ])
        .unwrap();
    assert!((k_exp[0] - 1.0).abs() < TN_EPS);
    assert!(k_exp[1].abs() < TN_EPS);
    assert!((k_exp[2] - 1.0).abs() < TN_EPS);
}

#[test]
fn tn_mid_circuit_measure_reset_rdm_and_expectations_match_sv() {
    use prism_q::backend::Backend;
    use prism_q::backend::statevector::StatevectorBackend;
    use prism_q::gates::Gate;
    use prism_q::sim::unified_pauli::PauliTerm;

    let mut circuit = Circuit::new(6, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_reset(0);
    circuit.add_gate(Gate::H, &[2]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_measure(2, 0);
    circuit.add_gate(Gate::H, &[3]);
    circuit.add_gate(Gate::Ry(0.7), &[4]);
    circuit.add_gate(Gate::Cx, &[4, 5]);

    for seed in [SEED, 7] {
        let mut sv = StatevectorBackend::new(seed);
        sv.init(6, 1).unwrap();
        let mut tn = TensorNetworkBackend::new(seed);
        tn.init(6, 1).unwrap();
        for inst in &circuit.instructions {
            sv.apply(inst).unwrap();
            tn.apply(inst).unwrap();
        }
        assert_eq!(tn.classical_results(), sv.classical_results());

        let obs = vec![
            vec![PauliTerm::x(1)],
            vec![PauliTerm::z(2)],
            vec![PauliTerm::x(3)],
            vec![PauliTerm::z(5)],
        ];
        let sv_exp = sv.pauli_expectations(&obs).unwrap();
        let tn_exp = tn.pauli_expectations(&obs).unwrap();
        for (got, want) in tn_exp.iter().zip(&sv_exp) {
            assert!((got - want).abs() < TN_EPS, "seed {seed}: {got} vs {want}");
        }

        let sv_rdm = sv.reduced_density_matrix(&[1, 3]).unwrap();
        let tn_rdm = tn.reduced_density_matrix(&[1, 3]).unwrap();
        for (got, want) in tn_rdm.iter().zip(&sv_rdm) {
            assert!((got - want).norm() < TN_EPS, "seed {seed}: {got} vs {want}");
        }
    }
}
