use super::*;
use crate::gates::Gate;

fn rx_cx_chain(circuit: &mut Circuit, qubits: std::ops::Range<usize>) {
    for q in qubits.clone() {
        circuit.add_gate(Gate::Rx(0.3), &[q]);
    }
    for q in qubits.start..qubits.end - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
}

#[test]
fn dense_entangled_circuit_is_candidate() {
    let mut circuit = Circuit::new(8, 0);
    rx_cx_chain(&mut circuit, 0..8);
    assert!(auto_terminal_statevector_candidate(&circuit));
}

#[test]
fn decomposable_circuit_is_not_candidate() {
    let mut circuit = Circuit::new(12, 0);
    rx_cx_chain(&mut circuit, 0..6);
    rx_cx_chain(&mut circuit, 6..12);
    assert!(!auto_terminal_statevector_candidate(&circuit));
}

#[test]
fn partial_independent_circuit_is_not_candidate() {
    let mut circuit = Circuit::new(10, 0);
    rx_cx_chain(&mut circuit, 0..8);
    rx_cx_chain(&mut circuit, 8..10);
    assert!(!auto_terminal_statevector_candidate(&circuit));
}

// A Clifford+T circuit inside the stabilizer-rank budget routes to the
// rank engine; one T past the budget falls back to the statevector and
// becomes a candidate. Pins the shared budget helper at its boundary.
#[test]
fn clifford_t_budget_boundary_flips_candidacy() {
    let n = 10;
    let budget = stabilizer_rank_budget(n);
    assert_eq!(budget, 2);

    let mut within = Circuit::new(n, 0);
    within.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        within.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..budget {
        within.add_gate(Gate::T, &[q]);
    }
    assert!(!auto_terminal_statevector_candidate(&within));

    let mut beyond = Circuit::new(n, 0);
    beyond.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        beyond.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..budget + 1 {
        beyond.add_gate(Gate::T, &[q]);
    }
    assert!(auto_terminal_statevector_candidate(&beyond));
}

#[test]
fn temporal_prefix_circuit_is_not_candidate() {
    let n = 8;
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::H, &[0]);
    for _ in 0..3 {
        for q in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    for q in 0..n {
        circuit.add_gate(Gate::Rx(0.3), &[q]);
    }
    assert!(has_temporal_clifford_opportunity(
        &BackendKind::Auto,
        &circuit
    ));
    assert!(!auto_terminal_statevector_candidate(&circuit));
}
