use super::*;
use crate::circuit::Circuit;
use crate::gates::Gate;

#[test]
fn rejects_too_many_qubits() {
    let n = MAX_STATEVECTOR_QUBITS + 1;
    let c = Circuit::new(n, 0);
    assert!(run_stabilizer_rank(&c, 0).is_err());
    assert!(run_stabilizer_rank_approx(&c, 0, 16).is_err());
}

#[test]
fn approx_path_runs_with_pruning() {
    let mut c = Circuit::new(3, 0);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::T, &[q]);
    }
    let result = run_stabilizer_rank_approx(&c, 42, 4).unwrap();
    let total: f64 = result.probabilities.iter().sum();
    assert!(total.is_finite());
}

#[test]
fn approx_path_tdg_runs() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Tdg, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Tdg, &[1]);
    let result = run_stabilizer_rank_approx(&c, 7, 16).unwrap();
    assert!(result.probabilities.iter().all(|p| p.is_finite()));
}
