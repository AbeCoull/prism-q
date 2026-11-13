use super::*;
use crate::circuit::{GuardedRegion, SaveSpec};

fn assert_same(actual: &Circuit, expected: &Circuit) {
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn cx(c: &mut Circuit, control: usize, target: usize) {
    c.add_gate(Gate::Cx, &[control, target]);
}

fn active_reset(c: &mut Circuit, bit: usize, qubit: usize) {
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(bit),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[qubit]),
    });
}

/// Two data qubits rotated, then one check onto ancilla 2 before `tail` and a terminal
/// readout of the data on bits 1 and 2.
fn one_check(tail: impl FnOnce(&mut Circuit)) -> Circuit {
    let mut c = Circuit::new(3, 3);
    c.add_gate(Gate::Ry(0.7), &[0]);
    c.add_gate(Gate::Ry(1.9), &[1]);
    cx(&mut c, 0, 2);
    cx(&mut c, 1, 2);
    tail(&mut c);
    c.add_measure(0, 1);
    c.add_measure(1, 2);
    c
}

fn measured_and_reset(c: &mut Circuit) {
    c.add_measure(2, 0);
    c.add_reset(2);
}

#[test]
fn a_two_source_check_becomes_a_fan_around_a_data_measurement() {
    let rewritten = rewrite_parity_ancillas(&one_check(measured_and_reset)).unwrap();
    let mut expected = Circuit::new(2, 3);
    expected.add_gate(Gate::Ry(0.7), &[0]);
    expected.add_gate(Gate::Ry(1.9), &[1]);
    cx(&mut expected, 1, 0);
    expected.add_measure(0, 0);
    cx(&mut expected, 1, 0);
    expected.add_measure(0, 1);
    expected.add_measure(1, 2);
    assert_same(&rewritten, &expected);
}

#[test]
fn an_odd_flip_count_inverts_the_readout_and_a_single_source_measures_alone() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::Ry(0.7), &[0]);
    c.add_gate(Gate::X, &[1]);
    cx(&mut c, 0, 1);
    c.add_measure(1, 0);
    active_reset(&mut c, 0, 1);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::X, &[1]);
    cx(&mut c, 0, 1);
    c.add_measure(1, 1);

    let mut expected = Circuit::new(1, 2);
    expected.add_gate(Gate::Ry(0.7), &[0]);
    expected.add_gate(Gate::X, &[0]);
    expected.add_measure(0, 0);
    expected.add_gate(Gate::X, &[0]);
    expected.add_measure(0, 1);
    assert_same(&rewrite_parity_ancillas(&c).unwrap(), &expected);
}

#[test]
fn compaction_keeps_qubit_order_and_bits_and_strips_barriers() {
    let mut c = Circuit::new(4, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Ry(0.4), &[2]);
    c.instructions.push(Instruction::Barrier {
        qubits: SmallVec::from_slice(&[0, 1, 2]),
    });
    cx(&mut c, 2, 1);
    c.add_measure(1, 1);
    c.add_reset(1);
    c.instructions.push(Instruction::Barrier {
        qubits: SmallVec::from_slice(&[1]),
    });
    c.add_gate(Gate::Cx, &[0, 3]);
    c.add_measure(3, 0);

    let rewritten = rewrite_parity_ancillas(&c).unwrap();
    let mut expected = Circuit::new(2, 2);
    expected.add_gate(Gate::H, &[0]);
    expected.add_gate(Gate::Ry(0.4), &[1]);
    expected.instructions.push(Instruction::Barrier {
        qubits: SmallVec::from_slice(&[0, 1]),
    });
    expected.add_measure(1, 1);
    expected.add_measure(0, 0);
    assert_same(&rewritten, &expected);
}

#[test]
fn ancillas_reused_over_rounds_and_interleaved_on_a_shared_source_all_rewrite() {
    let mut c = Circuit::new(4, 8);
    for round in 0..4 {
        c.add_gate(Gate::Ry(0.3 + round as f64), &[0]);
        c.add_gate(Gate::Ry(0.5 + round as f64), &[1]);
        cx(&mut c, 0, 2);
        cx(&mut c, 0, 3);
        cx(&mut c, 1, 2);
        cx(&mut c, 1, 3);
        c.add_measure(2, 2 * round);
        c.add_measure(3, 2 * round + 1);
        c.add_reset(2);
        active_reset(&mut c, 2 * round + 1, 3);
    }
    let rewritten = rewrite_parity_ancillas(&c).unwrap();
    assert_eq!(rewritten.num_qubits, 2);
    let measures = rewritten
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Measure { qubit: 0, .. }))
        .count();
    assert_eq!(measures, 8);
}

#[test]
fn a_final_span_may_leave_the_ancilla_unreset() {
    let c = one_check(|c| c.add_measure(2, 0));
    assert_eq!(rewrite_parity_ancillas(&c).unwrap().num_qubits, 2);
}

#[test]
fn an_untouched_qubit_and_a_reset_one_are_not_ancillas() {
    let mut c = Circuit::new(3, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_reset(1);
    c.add_measure(0, 0);
    assert!(rewrite_parity_ancillas(&c).is_none());
}

fn assert_declines(label: &str, c: &Circuit) {
    assert!(rewrite_parity_ancillas(c).is_none(), "{label}");
}

#[test]
fn a_gate_on_the_ancilla_other_than_cx_target_or_x_declines() {
    for gate in [Gate::H, Gate::Z, Gate::S, Gate::T, Gate::Rz(0.3)] {
        let c = one_check(|c| {
            c.add_gate(gate.clone(), &[2]);
            measured_and_reset(c);
        });
        assert_declines(gate.name(), &c);
    }
    let c = one_check(|c| {
        measured_and_reset(c);
        cx(c, 2, 0);
    });
    assert_declines("ancilla as a control", &c);
    let c = one_check(|c| {
        c.add_gate(Gate::Cz, &[0, 2]);
        measured_and_reset(c);
    });
    assert_declines("cz", &c);
}

#[test]
fn a_gate_on_a_source_inside_its_span_declines() {
    for gate in [Gate::Rz(0.2), Gate::H] {
        let c = one_check(|c| {
            c.add_gate(gate.clone(), &[0]);
            measured_and_reset(c);
        });
        assert_declines(gate.name(), &c);
    }
    let c = one_check(|c| {
        cx(c, 1, 0);
        measured_and_reset(c);
    });
    assert_declines("source as a cx target", &c);
    let c = one_check(|c| {
        c.add_measure(0, 1);
        measured_and_reset(c);
    });
    assert_declines("source measured", &c);
}

#[test]
fn a_source_driving_another_cx_or_a_gate_after_the_measurement_still_rewrites() {
    let mut c = Circuit::new(4, 3);
    c.add_gate(Gate::Ry(0.7), &[0]);
    c.add_gate(Gate::Ry(1.1), &[3]);
    cx(&mut c, 0, 2);
    cx(&mut c, 0, 3);
    c.add_gate(Gate::Ry(0.5), &[1]);
    c.add_measure(2, 0);
    c.add_gate(Gate::Rz(0.9), &[0]);
    c.add_reset(2);
    c.add_measure(0, 1);
    c.add_measure(3, 2);
    assert_eq!(rewrite_parity_ancillas(&c).unwrap().num_qubits, 3);
}

#[test]
fn an_empty_parity_declines() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::Ry(0.7), &[0]);
    cx(&mut c, 0, 1);
    cx(&mut c, 0, 1);
    c.add_measure(1, 0);
    c.add_measure(0, 1);
    assert_declines("cancelled pickups", &c);

    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_measure(1, 0);
    c.add_measure(0, 1);
    assert_declines("flips alone", &c);
}

#[test]
fn a_guard_on_the_ancilla_other_than_its_own_active_reset_declines() {
    let c = one_check(|c| {
        c.add_measure(2, 0);
        c.add_measure(0, 1);
        active_reset(c, 1, 2);
    });
    assert_declines("guard on another bit", &c);
    let c = one_check(|c| {
        c.add_measure(2, 0);
        c.add_measure(0, 0);
        active_reset(c, 0, 2);
    });
    assert_declines("bit overwritten before the guard", &c);
    let c = one_check(|c| {
        c.add_measure(2, 0);
        c.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsZero(0),
            gate: Gate::X,
            targets: SmallVec::from_slice(&[2]),
        });
    });
    assert_declines("guard on bit zero", &c);
    let c = one_check(|c| {
        c.instructions.push(Instruction::Conditional {
            condition: ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: SmallVec::from_slice(&[2]),
        });
        measured_and_reset(c);
    });
    assert_declines("guarded flip inside the span", &c);
    let c = one_check(|c| {
        measured_and_reset(c);
        c.instructions
            .push(Instruction::Region(Box::new(GuardedRegion::new(
                ClassicalCondition::BitIsOne(0),
                vec![Instruction::Reset { qubit: 2 }],
            ))));
    });
    assert_declines("region on the ancilla", &c);
}

#[test]
fn a_region_on_a_frozen_source_declines() {
    let c = one_check(|c| {
        c.instructions
            .push(Instruction::Region(Box::new(GuardedRegion::new(
                ClassicalCondition::BitIsOne(1),
                vec![Instruction::Gate {
                    gate: Gate::X,
                    targets: SmallVec::from_slice(&[0]),
                }],
            ))));
        measured_and_reset(c);
    });
    assert_declines("region on a source", &c);
}

#[test]
fn reuse_without_a_reset_declines() {
    let c = one_check(|c| {
        c.add_measure(2, 0);
        c.add_measure(2, 0);
    });
    assert_declines("second measurement", &c);
    let c = one_check(|c| {
        c.add_measure(2, 0);
        cx(c, 0, 2);
        c.add_measure(2, 0);
    });
    assert_declines("pickup after the measurement", &c);
}

#[test]
fn a_reset_with_no_measurement_or_an_unmeasured_span_declines() {
    let c = one_check(|c| c.add_reset(2));
    assert_declines("reset with no measurement", &c);
    let c = one_check(|_| {});
    assert_declines("span never measured", &c);
}

#[test]
fn any_save_declines() {
    let mut c = one_check(measured_and_reset);
    c.add_save(SaveSpec::Probabilities, "p");
    assert_declines("save", &c);
}

#[test]
fn a_demoted_ancilla_leaves_the_others_rewritten() {
    let mut c = one_check(measured_and_reset);
    c.num_qubits = 4;
    c.num_classical_bits = 4;
    cx(&mut c, 0, 3);
    c.add_gate(Gate::H, &[3]);
    c.add_measure(3, 3);
    let rewritten = rewrite_parity_ancillas(&c).unwrap();
    assert_eq!(rewritten.num_qubits, 3);
}
