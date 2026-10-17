use super::*;
use std::f64::consts::PI;

#[test]
fn builder_bell_state() {
    let c = CircuitBuilder::new(2).h(0).cx(0, 1).build();
    assert_eq!(c.instructions.len(), 2);
    assert_eq!(c.num_qubits, 2);
    assert_eq!(c.num_classical_bits, 0);
}

#[test]
fn builder_parametric() {
    let c = CircuitBuilder::new(2).rx(PI, 0).rz(PI / 2.0, 1).build();
    assert_eq!(c.instructions.len(), 2);
    match &c.instructions[0] {
        Instruction::Gate { gate, targets } => {
            assert!(matches!(gate, Gate::Rx(_)));
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected Gate instruction"),
    }
}

#[test]
fn builder_measure_all() {
    let c = CircuitBuilder::new(3).h(0).measure_all().build();
    assert_eq!(c.num_classical_bits, 3);
    let measures: Vec<_> = c
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. }))
        .collect();
    assert_eq!(measures.len(), 3);
}

#[test]
fn builder_reset_emits_instruction_and_chains() {
    let c = CircuitBuilder::new(2).x(0).reset(0).h(1).build();
    assert_eq!(c.instructions.len(), 3);
    assert!(matches!(
        c.instructions.as_slice(),
        [
            Instruction::Gate {
                gate: Gate::X,
                targets: x_targets,
            },
            Instruction::Reset { qubit: 0 },
            Instruction::Gate {
                gate: Gate::H,
                targets: h_targets,
            },
        ] if x_targets.as_slice() == [0] && h_targets.as_slice() == [1]
    ));
}

#[test]
fn measure_in_basis_rotates_and_does_not_restore() {
    let c = CircuitBuilder::new_with_classical(1, 1)
        .measure_in_basis(0, PauliAxis::Y, 0)
        .build();
    assert!(matches!(
        c.instructions.as_slice(),
        [
            Instruction::Gate {
                gate: Gate::Sdg,
                ..
            },
            Instruction::Gate { gate: Gate::H, .. },
            Instruction::Measure {
                qubit: 0,
                classical_bit: 0
            },
        ]
    ));
    assert!(c.has_terminal_measurements_only());
}

#[test]
fn measure_pauli_product_allocates_one_scratch_and_resets_it_after_the_first() {
    let c = CircuitBuilder::new_with_classical(2, 2)
        .h(0)
        .measure_pauli_product(&[PauliTerm::x(0), PauliTerm::z(1)], 0)
        .measure_pauli_product(&[PauliTerm::z(0), PauliTerm::z(1)], 1)
        .build();
    assert_eq!(c.num_qubits, 3);
    assert_eq!(c.measurement_map(), [(2, 0), (2, 1)]);
    let mut hand = Circuit::new(3, 2);
    hand.add_gate(Gate::H, &[0]);
    hand.add_gate(Gate::H, &[0]);
    hand.add_gate(Gate::Cx, &[0, 2]);
    hand.add_gate(Gate::Cx, &[1, 2]);
    hand.add_gate(Gate::H, &[0]);
    hand.add_measure(2, 0);
    hand.add_reset(2);
    hand.add_gate(Gate::Cx, &[0, 2]);
    hand.add_gate(Gate::Cx, &[1, 2]);
    hand.add_measure(2, 1);
    assert_eq!(
        format!("{:?}", c.instructions),
        format!("{:?}", hand.instructions)
    );
}

#[test]
#[should_panic(expected = "duplicate term")]
fn measure_pauli_product_rejects_a_repeated_qubit() {
    CircuitBuilder::new_with_classical(2, 1)
        .measure_pauli_product(&[PauliTerm::x(0), PauliTerm::z(0)], 0);
}

#[test]
fn builder_conditional() {
    let c = CircuitBuilder::new_with_classical(2, 1)
        .x(0)
        .measure(0, 0)
        .conditional(ClassicalCondition::BitIsOne(0), Gate::X, &[1])
        .build();
    assert_eq!(c.instructions.len(), 3);
    assert!(matches!(
        &c.instructions[2],
        Instruction::Conditional { .. }
    ));
}

#[test]
fn builder_run_matches_direct() {
    let builder_result = CircuitBuilder::new(2)
        .h(0)
        .cx(0, 1)
        .run(42)
        .expect("builder run failed");
    let bp = builder_result.probabilities.expect("no probs").to_vec();

    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    let direct_result = crate::sim::simulate(&c)
        .seed(42)
        .run()
        .expect("direct run failed");
    let dp = direct_result.probabilities.expect("no probs").to_vec();

    assert_eq!(bp.len(), dp.len());
    for (b, d) in bp.iter().zip(dp.iter()) {
        assert!((b - d).abs() < 1e-12);
    }
}

#[test]
fn builder_generic_gate() {
    let c = CircuitBuilder::new(2).gate(Gate::Swap, &[0, 1]).build();
    assert_eq!(c.instructions.len(), 1);
    match &c.instructions[0] {
        Instruction::Gate { gate, targets } => {
            assert!(matches!(gate, Gate::Swap));
            assert_eq!(targets.as_slice(), &[0, 1]);
        }
        _ => panic!("expected Gate instruction"),
    }
}

#[test]
fn standard_gates_match_the_parsed_lowering() {
    let built = CircuitBuilder::new(3)
        .ccx(0, 1, 2)
        .cy(0, 1)
        .ch(1, 2)
        .crx(0.3, 0, 2)
        .cry(-0.7, 2, 1)
        .crz(1.1, 1, 0)
        .u(0.4, -1.2, 2.5, 1)
        .rxx(0.6, 0, 2)
        .ryy(-0.9, 2, 1)
        .iswap(0, 1)
        .cswap(2, 0, 1)
        .build();
    let parsed = crate::circuit::openqasm::parse(
        "OPENQASM 3.0;
include \"stdgates.inc\";
qubit[3] q;
             ccx q[0], q[1], q[2];
cy q[0], q[1];
ch q[1], q[2];
             crx(0.3) q[0], q[2];
cry(-0.7) q[2], q[1];
crz(1.1) q[1], q[0];
             u(0.4, -1.2, 2.5) q[1];
rxx(0.6) q[0], q[2];
ryy(-0.9) q[2], q[1];
             iswap q[0], q[1];
cswap q[2], q[0], q[1];
",
    )
    .expect("parse");
    assert_eq!(built.num_qubits, parsed.num_qubits);
    assert_eq!(
        format!("{:?}", built.instructions),
        format!("{:?}", parsed.instructions)
    );
}

#[test]
fn builder_cphase() {
    let c = CircuitBuilder::new(2).cphase(PI / 4.0, 0, 1).build();
    assert_eq!(c.instructions.len(), 1);
    match &c.instructions[0] {
        Instruction::Gate { gate, targets } => {
            assert!(matches!(gate, Gate::Cu(_)));
            assert_eq!(targets.as_slice(), &[0, 1]);
        }
        _ => panic!("expected Gate instruction"),
    }
}

#[test]
fn builder_mcu() {
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let x_mat = [[zero, one], [one, zero]];
    let c = CircuitBuilder::new(3).mcu(x_mat, &[0, 1], 2).build();
    assert_eq!(c.instructions.len(), 1);
    match &c.instructions[0] {
        Instruction::Gate { gate, targets } => {
            assert!(matches!(gate, Gate::Mcu(_)));
            assert_eq!(targets.as_slice(), &[0, 1, 2]);
        }
        _ => panic!("expected Gate instruction"),
    }
}
