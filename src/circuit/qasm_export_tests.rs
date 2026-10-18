use super::*;
use crate::circuit::openqasm;
use crate::circuit::{SmallVec, smallvec};

fn reparse(circuit: &Circuit) -> Circuit {
    openqasm::parse(&to_qasm3(circuit).expect("export")).expect("reparse")
}

fn one_gate(gate: Gate, targets: &[usize]) -> Circuit {
    let mut c = Circuit::new(targets.iter().max().unwrap() + 1, 0);
    c.add_gate(gate, targets);
    c
}

#[test]
fn named_gates_keep_their_name() {
    for gate in [Gate::H, Gate::T, Gate::Sdg, Gate::SX, Gate::Id] {
        assert_eq!(gate_head(&gate).unwrap(), gate.name());
    }
}

#[test]
fn angles_survive_exactly() {
    let round = reparse(&one_gate(Gate::Rx(0.735_193_628_1), &[0]));
    let Instruction::Gate { gate, .. } = &round.instructions[0] else {
        panic!("expected a gate");
    };
    assert_eq!(*gate, Gate::Rx(0.735_193_628_1));
}

#[test]
fn u3_matrix_round_trips() {
    let mat = Parser::u_matrix(0.7, -1.3, 2.1);
    let round = reparse(&one_gate(Gate::Fused(Box::new(mat)), &[0]));
    let Instruction::Gate { gate, .. } = &round.instructions[0] else {
        panic!("expected a gate");
    };
    assert!(close_2x2(&gate.matrix_2x2(), &mat));
}

#[test]
fn global_phase_on_a_fused_matrix_is_rejected() {
    let phase = Complex64::from_polar(1.0, PI / 3.0);
    let h = Gate::H.matrix_2x2();
    let mat = [
        [phase * h[0][0], phase * h[0][1]],
        [phase * h[1][0], phase * h[1][1]],
    ];
    let circuit = one_gate(Gate::Fused(Box::new(mat)), &[0]);
    assert!(matches!(
        to_qasm3(&circuit),
        Err(PrismError::ExportUnsupported { index: 0, .. })
    ));
}

#[test]
fn sx_family_from_a_matrix_keeps_its_name() {
    assert_eq!(spell_1q(&Gate::SX.matrix_2x2()).unwrap(), "sx");
    assert_eq!(spell_1q(&Gate::SXdg.matrix_2x2()).unwrap(), "sxdg");
    let circuit = one_gate(Gate::Fused(Box::new(Gate::SXdg.matrix_2x2())), &[0]);
    assert!(matches!(
        reparse(&circuit).instructions[0],
        Instruction::Gate {
            gate: Gate::SXdg,
            ..
        }
    ));
}

#[test]
fn rz_carries_the_phase_u3_cannot() {
    assert_eq!(
        spell_1q(&Gate::Rz(0.9).matrix_2x2()).unwrap(),
        format!("rz({})", 0.9_f64)
    );
}

#[test]
fn controlled_unitary_keeps_its_global_phase() {
    let mat = Gate::Rz(1.1).matrix_2x2();
    let round = reparse(&one_gate(Gate::cu(mat), &[0, 1]));
    let Instruction::Gate { gate, .. } = &round.instructions[0] else {
        panic!("expected a gate");
    };
    assert!(close_4x4(&gate.matrix_4x4(), &Gate::cu(mat).matrix_4x4()));
}

#[test]
fn multi_controlled_x_uses_the_named_forms() {
    assert_eq!(
        gate_head(&Gate::mcu(Gate::X.matrix_2x2(), 2)).unwrap(),
        "ccx"
    );
    assert_eq!(
        gate_head(&Gate::mcu(Gate::X.matrix_2x2(), 4)).unwrap(),
        "mcx"
    );
}

#[test]
fn multi_controlled_unitary_chains_ctrl() {
    let head = gate_head(&Gate::mcu(Gate::H.matrix_2x2(), 3)).unwrap();
    assert!(head.starts_with("ctrl @ ctrl @ cu("), "{head}");
}

#[test]
fn fusion_payloads_are_rejected() {
    let mut circuit = Circuit::new(2, 0);
    circuit.instructions.push(Instruction::Gate {
        gate: Gate::Fused2q(Box::new(Gate::Cx.matrix_4x4())),
        targets: smallvec![0, 1],
    });
    assert!(matches!(
        to_qasm3(&circuit),
        Err(PrismError::ExportUnsupported { index: 0, .. })
    ));
}

#[test]
fn conditions_split_the_classical_register() {
    let mut circuit = Circuit::new(1, 4);
    circuit.add_measure(0, 0);
    circuit.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::RegisterEquals {
            offset: 2,
            size: 2,
            value: 3,
        },
        gate: Gate::X,
        targets: SmallVec::from_slice(&[0]),
    });
    let qasm = to_qasm3(&circuit).expect("export");
    assert!(qasm.contains("bit[2] c0;\nbit[2] c1;"), "{qasm}");
    assert!(qasm.contains("if (c1 == 3) x q[0];"), "{qasm}");
    assert_eq!(
        openqasm::parse(&qasm).expect("reparse").num_classical_bits,
        4
    );
}

#[test]
fn empty_barrier_has_no_spelling() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_barrier(&[]);
    assert!(to_qasm3(&circuit).is_err());
}
