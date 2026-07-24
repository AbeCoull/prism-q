//! Cross-backend correctness for the density-matrix backend.
//!
//! On noiseless (unitary) circuits the density-matrix backend must reproduce the
//! statevector backend's basis-state probabilities exactly, and a pure state must
//! stay pure (`Tr(rho^2) == 1`).

use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{circuits, sim};

const DM_EPS: f64 = 1e-12;

fn statevector_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend.probabilities().unwrap()
}

fn run_dm(circuit: &Circuit) -> DensityMatrixBackend {
    let mut backend = DensityMatrixBackend::new(42);
    sim::run_on(&mut backend, circuit).unwrap();
    backend
}

fn dm_probs(circuit: &Circuit) -> Vec<f64> {
    run_dm(circuit).probabilities().unwrap()
}

fn assert_probs_close(actual: &[f64], expected: &[f64], label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: probability vector length mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < DM_EPS,
            "{label}: probability[{i}] mismatch: dm={a}, statevector={e}"
        );
    }
}

fn assert_matches_statevector(circuit: &Circuit, label: &str) {
    assert_probs_close(&dm_probs(circuit), &statevector_probs(circuit), label);
}

#[test]
fn dm_all_single_gates_match_statevector() {
    let gates = [
        Gate::Id,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::T,
        Gate::Tdg,
        Gate::SX,
        Gate::SXdg,
        Gate::Rx(0.7),
        Gate::Ry(1.1),
        Gate::Rz(0.3),
        Gate::P(0.5),
    ];
    for gate in gates {
        // The Hadamard sandwich converts phase-only differences into population,
        // so the probability comparison is sensitive to a mishandled conjugation.
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(gate.clone(), &[0]);
        c.add_gate(Gate::H, &[0]);
        assert_matches_statevector(&c, &format!("single_gate {gate:?}"));
    }
}

#[test]
fn dm_two_qubit_gates_match_statevector() {
    for gate in [Gate::Cx, Gate::Cz, Gate::Swap, Gate::Rzz(0.9)] {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Ry(0.6), &[1]);
        c.add_gate(gate.clone(), &[0, 1]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        assert_matches_statevector(&c, &format!("two_qubit_gate {gate:?}"));
    }
}

#[test]
fn dm_bell_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    assert_matches_statevector(&c, "bell");
}

#[test]
fn dm_ghz4_matches_statevector() {
    let c = circuits::ghz_circuit(4);
    assert_matches_statevector(&c, "ghz4");
}

#[test]
fn dm_complex_circuit_matches_statevector() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Ry(1.3), &[2]);
    c.add_gate(Gate::Cz, &[1, 2]);
    c.add_gate(Gate::Swap, &[0, 2]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Rz(0.9), &[1]);
    c.add_gate(Gate::Cx, &[2, 0]);
    assert_matches_statevector(&c, "complex");
}

#[test]
fn dm_qft_matches_statevector() {
    let c = circuits::qft_circuit(5);
    assert_matches_statevector(&c, "qft5");
}

#[test]
fn dm_random_layers_match_statevector() {
    let c = circuits::random_circuit(6, 10, 0xDEAD_BEEF);
    assert_matches_statevector(&c, "random6");
}

#[test]
fn dm_pure_state_stays_pure() {
    let c = circuits::random_circuit(5, 8, 0xDEAD_BEEF);
    let backend = run_dm(&c);
    assert!(
        (backend.purity() - 1.0).abs() < 1e-12,
        "purity of a unitary-evolved state must be 1, got {}",
        backend.purity()
    );
}

#[test]
fn dm_probabilities_sum_to_one() {
    let c = circuits::random_circuit(5, 8, 0xDEAD_BEEF);
    let total: f64 = dm_probs(&c).iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-12,
        "probabilities must sum to 1, got {total}"
    );
}

#[test]
fn dm_reduced_density_matrix_matches_statevector() {
    let c = circuits::random_circuit(4, 8, 0xDEAD_BEEF);
    let dm = run_dm(&c);
    let mut sv = StatevectorBackend::new(42);
    sim::run_on(&mut sv, &c).unwrap();
    for q in 0..4 {
        let a = dm.reduced_density_matrix_1q(q).unwrap();
        let b = sv.reduced_density_matrix_1q(q).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[i][j] - b[i][j]).norm() < 1e-12,
                    "rdm[{q}][{i}][{j}] mismatch: dm={}, sv={}",
                    a[i][j],
                    b[i][j]
                );
            }
        }
    }
}
