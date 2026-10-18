use super::*;
use crate::circuit::Instruction;
use crate::gates::Gate;

fn heisenberg(n: usize) -> PauliObservable {
    let mut terms = Vec::new();
    for q in 0..n - 1 {
        terms.push((1.0, vec![PauliTerm::x(q), PauliTerm::x(q + 1)]));
        terms.push((0.8, vec![PauliTerm::y(q), PauliTerm::y(q + 1)]));
        terms.push((0.6, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]));
    }
    for q in 0..n {
        terms.push((0.3, vec![PauliTerm::x(q)]));
    }
    PauliObservable::from_terms(terms).unwrap()
}

fn angles(circuit: &Circuit) -> Vec<f64> {
    circuit
        .instructions
        .iter()
        .map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::PauliRot(data),
                ..
            } => data.theta(),
            Instruction::Gate {
                gate: Gate::Rx(theta) | Gate::Ry(theta) | Gate::Rz(theta) | Gate::Rzz(theta),
                ..
            } => *theta,
            other => panic!("unexpected instruction {other:?}"),
        })
        .collect()
}

#[test]
fn first_order_emits_one_rotation_per_term_per_step() {
    let h = heisenberg(4);
    let mut circuit = Circuit::new(4, 0);
    circuit.add_trotter(&h, 0.5, 3, TrotterOrder::First);
    assert_eq!(circuit.instructions.len(), 3 * h.num_terms());
    let (coefficient, _) = &h.terms()[0];
    assert!((angles(&circuit)[0] - 2.0 * coefficient * 0.5 / 3.0).abs() < 1e-15);
}

#[test]
fn second_order_merges_the_middle_and_the_step_boundary() {
    let h = heisenberg(4);
    let m = h.num_terms();
    let mut one = Circuit::new(4, 0);
    one.add_trotter(&h, 0.4, 1, TrotterOrder::Second);
    assert_eq!(one.instructions.len(), 2 * m - 1);
    let mut three = Circuit::new(4, 0);
    three.add_trotter(&h, 0.4, 3, TrotterOrder::Second);
    assert_eq!(three.instructions.len(), 3 * (2 * m - 1) - 2);

    let dt = 0.4;
    let angles = angles(&one);
    let (first, _) = &h.terms()[0];
    let (last, _) = &h.terms()[m - 1];
    assert!(
        (angles[0] - first * dt).abs() < 1e-15,
        "half step at the edge"
    );
    assert!(
        (angles[m - 1] - 2.0 * last * dt).abs() < 1e-15,
        "merged middle"
    );
    assert!((angles[2 * m - 2] - first * dt).abs() < 1e-15);
}

#[test]
fn suzuki_one_is_second_order_and_suzuki_two_costs_five_substeps() {
    let h = heisenberg(3);
    let m = h.num_terms();
    let mut second = Circuit::new(3, 0);
    second.add_trotter(&h, 0.3, 2, TrotterOrder::Second);
    let mut suzuki = Circuit::new(3, 0);
    suzuki.add_trotter(&h, 0.3, 2, TrotterOrder::Suzuki(1));
    assert_eq!(angles(&second), angles(&suzuki));

    let mut fourth = Circuit::new(3, 0);
    fourth.add_trotter(&h, 0.3, 1, TrotterOrder::Suzuki(2));
    assert_eq!(fourth.instructions.len(), 5 * (2 * m - 1) - 4);
    let total: f64 = angles(&fourth).iter().sum();
    let total_coefficient: f64 = h.terms().iter().map(|(c, _)| c).sum();
    assert!(
        (total - 2.0 * total_coefficient * 0.3).abs() < 1e-12,
        "substep lengths sum to the step"
    );
}

#[test]
fn zero_and_identity_terms_append_nothing() {
    let h = PauliObservable::from_terms([
        (0.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
        (2.5, vec![]),
        (0.7, vec![PauliTerm::z(0)]),
    ])
    .unwrap();
    let mut circuit = Circuit::new(2, 0);
    circuit.add_trotter(&h, 1.0, 2, TrotterOrder::First);
    // The surviving term is the only term, so its two steps merge too.
    assert!(matches!(
        circuit.instructions.as_slice(),
        [Instruction::Gate {
            gate: Gate::Rz(theta),
            ..
        }] if (theta - 1.4).abs() < 1e-15
    ));
}

#[test]
#[should_panic(expected = "at least one step")]
fn zero_steps_panics() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_trotter(&heisenberg(2), 1.0, 0, TrotterOrder::First);
}

#[test]
#[should_panic(expected = "k >= 1")]
fn suzuki_zero_panics() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_trotter(&heisenberg(2), 1.0, 1, TrotterOrder::Suzuki(0));
}
