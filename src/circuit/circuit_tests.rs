use super::*;

// Every backend routes a conditional through ClassicalCondition::evaluate,
// so the variant semantics are pinned once here rather than per backend.
// Registers read bit `offset + i` as `1 << i`.
#[test]
fn classical_condition_evaluates_every_variant() {
    let bits = [true, false, true];

    assert!(ClassicalCondition::BitIsOne(0).evaluate(&bits));
    assert!(!ClassicalCondition::BitIsOne(1).evaluate(&bits));
    assert!(ClassicalCondition::BitIsZero(1).evaluate(&bits));
    assert!(!ClassicalCondition::BitIsZero(0).evaluate(&bits));

    let reg = |value| ClassicalCondition::RegisterEquals {
        offset: 0,
        size: 3,
        value,
    };
    assert!(reg(0b101).evaluate(&bits));
    assert!(!reg(0b100).evaluate(&bits));

    let reg_ne = |value| ClassicalCondition::RegisterNotEquals {
        offset: 0,
        size: 3,
        value,
    };
    assert!(reg_ne(0b100).evaluate(&bits));
    assert!(!reg_ne(0b101).evaluate(&bits));

    let parity = |list: &[usize], expected| ClassicalCondition::Parity {
        bits: list.to_vec().into(),
        expected,
    };
    assert!(parity(&[0, 1], true).evaluate(&bits), "one bit set is odd");
    assert!(!parity(&[0, 2], true).evaluate(&bits), "two set is even");
    assert!(parity(&[0, 2], false).evaluate(&bits));
    assert!(parity(&[0], true).evaluate(&bits), "one bit is a bit test");
    assert!(
        !parity(&[0, 0], true).evaluate(&bits),
        "a bit cancels itself"
    );
    assert!(!parity(&[], true).evaluate(&bits), "no bits is parity zero");
    assert!(parity(&[], false).evaluate(&bits));
}

#[test]
fn classical_condition_register_honors_offset() {
    let bits = [true, true, false, true];
    let at = |offset| ClassicalCondition::RegisterEquals {
        offset,
        size: 2,
        value: 0b01,
    };
    assert!(at(1).evaluate(&bits), "bits 1..3 are 1,0 so the value is 1");
    assert!(
        !at(0).evaluate(&bits),
        "bits 0..2 are 1,1 so the value is 3"
    );
}

#[test]
fn test_circuit_builder() {
    let mut c = Circuit::new(3, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_measure(0, 0);
    c.add_measure(1, 1);

    assert_eq!(c.num_qubits, 3);
    assert_eq!(c.num_classical_bits, 2);
    assert_eq!(c.gate_count(), 3);
    assert_eq!(c.instructions.len(), 5);
}

#[test]
fn test_depth_linear() {
    // H(0), CX(0,1), CX(1,2): depth 3 (serial chain)
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    assert_eq!(c.depth(), 3);
}

#[test]
fn test_depth_parallel() {
    // H(0), H(1), H(2): depth 1 (all parallel)
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::H, &[2]);
    assert_eq!(c.depth(), 1);
}

#[test]
fn test_empty_depth() {
    let c = Circuit::new(4, 0);
    assert_eq!(c.depth(), 0);
}

#[test]
fn test_clifford_prefix_split_basic() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Rx(0.5), &[1]);

    let (prefix, tail) = c.clifford_prefix_split().unwrap();
    assert_eq!(prefix.gate_count(), 2); // H, CX
    assert_eq!(tail.gate_count(), 2); // T, Rx
}

#[test]
fn test_clifford_prefix_split_none_when_all_clifford() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::S, &[0]);
    assert!(c.clifford_prefix_split().is_none());
}

#[test]
fn test_clifford_prefix_split_none_when_first_non_clifford() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    assert!(c.clifford_prefix_split().is_none());
}

#[test]
fn test_clifford_prefix_split_stops_at_measure() {
    let mut c = Circuit::new(2, 1);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(0, 0);
    c.add_gate(Gate::H, &[1]);

    let (prefix, tail) = c.clifford_prefix_split().unwrap();
    assert_eq!(prefix.gate_count(), 2); // H, CX
    assert_eq!(tail.instructions.len(), 2); // measure, H
}

#[test]
fn test_clifford_prefix_split_barrier_transparent() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_barrier(&[0, 1]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[0]);

    let (prefix, tail) = c.clifford_prefix_split().unwrap();
    assert_eq!(prefix.instructions.len(), 3); // H, barrier, CX
    assert_eq!(tail.gate_count(), 1); // T
}

#[test]
fn test_subsystems_fully_connected() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0], vec![0, 1, 2, 3]);
}

#[test]
fn test_subsystems_disjoint_pairs() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[4, 5]);
    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 3);
    assert_eq!(subs[0], vec![0, 1]);
    assert_eq!(subs[1], vec![2, 3]);
    assert_eq!(subs[2], vec![4, 5]);
}

#[test]
fn test_subsystems_no_entangling() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::X, &[1]);
    c.add_gate(Gate::Z, &[2]);
    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 3);
}

#[test]
fn test_subsystems_empty() {
    let c = Circuit::new(0, 0);
    assert!(c.independent_subsystems().is_empty());
}

#[test]
fn test_subsystems_classical_dependency_merges() {
    let mut c = Circuit::new(4, 2);
    // q0-q1 entangled, q2-q3 entangled, but q0 measured and conditional on q2
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[2]),
    });
    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 1);
    assert_eq!(subs[0], vec![0, 1, 2, 3]);
}

#[test]
fn test_subsystems_no_classical_dependency() {
    let mut c = Circuit::new(4, 2);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);
    c.add_measure(2, 1);
    let subs = c.independent_subsystems();
    assert_eq!(subs.len(), 2);
}

#[test]
fn test_subsystems_join_a_condition_to_the_last_writer_of_its_bit() {
    let mut c = Circuit::new(5, 1);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[4]),
    });
    c.add_measure(0, 0);
    c.add_measure(2, 0);
    assert_eq!(
        c.independent_subsystems(),
        vec![vec![0], vec![1], vec![2, 4], vec![3]]
    );
}

#[test]
fn test_subsystems_and_clifford_reads_every_gate_form() {
    let region = |gate: Gate| {
        Instruction::Region(Box::new(GuardedRegion::new(
            ClassicalCondition::BitIsOne(0),
            vec![Instruction::Gate {
                gate,
                targets: SmallVec::from_slice(&[2]),
            }],
        )))
    };
    let conditional = |gate: Gate| Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate,
        targets: SmallVec::from_slice(&[3]),
    };

    for (tail, clifford) in [
        (vec![], true),
        (vec![conditional(Gate::S), region(Gate::H)], true),
        (vec![conditional(Gate::T)], false),
        (vec![region(Gate::T)], false),
        (
            vec![Instruction::Gate {
                gate: Gate::Rz(0.3),
                targets: SmallVec::from_slice(&[1]),
            }],
            false,
        ),
    ] {
        let mut c = Circuit::new(4, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.instructions.extend(tail);
        assert_eq!(c.is_clifford_only(), clifford);
        assert_eq!(c.subsystems_and_clifford().1, clifford);
    }
    assert!(Circuit::new(0, 0).subsystems_and_clifford().1);
}

#[test]
fn test_partition_subcircuits_basic() {
    let mut c = Circuit::new(4, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_measure(0, 0);
    c.add_measure(2, 1);

    let parts = c.partition_subcircuits(&c.independent_subsystems());
    let (sub, q_map, c_map) = &parts[1];
    assert_eq!(sub.num_qubits, 2);
    assert_eq!(sub.num_classical_bits, 1);
    assert_eq!(sub.gate_count(), 2); // H(0), CX(0,1) remapped
    assert_eq!(sub.instructions.len(), 3); // 2 gates + 1 measure
    assert_eq!(q_map, &[2, 3]);
    assert_eq!(c_map, &[1]); // classical bit 1 maps to local 0
}

#[test]
fn test_partition_subcircuits_shared_classical_bit() {
    let mut c = Circuit::new(2, 2);
    c.add_measure(0, 1);
    c.add_measure(1, 0);
    c.add_measure(1, 1);

    let parts = c.partition_subcircuits(&c.independent_subsystems());
    assert_eq!(parts[0].2, vec![1]);
    assert_eq!(parts[1].2, vec![0, 1]);
    let sub = &parts[1].0;
    assert_eq!(sub.num_classical_bits, 2);
    assert!(matches!(
        sub.instructions[1],
        Instruction::Measure {
            qubit: 0,
            classical_bit: 1
        }
    ));
}

#[test]
fn test_partition_subcircuits_remaps_indices() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(Gate::Cx, &[2, 3]);

    let parts = c.partition_subcircuits(&c.independent_subsystems());
    let (sub, _, _) = &parts[2];
    if let Instruction::Gate { targets, .. } = &sub.instructions[0] {
        assert_eq!(targets.as_slice(), &[0, 1]);
    } else {
        panic!("expected gate instruction");
    }
}

#[test]
#[should_panic(expected = "qubit index 5 out of bounds (circuit has 3 qubits)")]
fn add_gate_panics_on_out_of_bounds_target_in_release() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[5]);
}

#[test]
#[should_panic(expected = "qubit index 4 out of bounds (circuit has 3 qubits)")]
fn add_gate_panics_on_second_target_out_of_bounds() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Cx, &[0, 4]);
}

#[test]
#[should_panic(expected = "expects 2 qubits, got 1")]
fn add_gate_panics_on_arity_mismatch() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Cx, &[0]);
}

#[test]
#[should_panic(expected = "qubit index 2 out of bounds (circuit has 2 qubits)")]
fn add_measure_panics_on_out_of_bounds_qubit() {
    let mut c = Circuit::new(2, 2);
    c.add_measure(2, 0);
}

#[test]
#[should_panic(expected = "classical bit index 5 out of bounds (circuit has 2 classical bits)")]
fn add_measure_panics_on_out_of_bounds_classical_bit() {
    let mut c = Circuit::new(2, 2);
    c.add_measure(0, 5);
}

#[test]
#[should_panic(expected = "qubit index 9 out of bounds (circuit has 2 qubits)")]
fn add_reset_panics_on_out_of_bounds() {
    let mut c = Circuit::new(2, 0);
    c.add_reset(9);
}

#[test]
#[should_panic(expected = "qubit index 7 out of bounds (circuit has 4 qubits)")]
fn add_barrier_panics_on_out_of_bounds() {
    let mut c = Circuit::new(4, 0);
    c.add_barrier(&[0, 7]);
}

#[test]
fn a_register_condition_wider_than_64_bits_compares_exactly() {
    for size in [65, 70] {
        let equals_one = ClassicalCondition::RegisterEquals {
            offset: 1,
            size,
            value: 1,
        };
        let mut bits = vec![false; size + 1];
        bits[1] = true;
        assert!(equals_one.evaluate(&bits));
        bits[size] = true;
        assert!(
            !equals_one.evaluate(&bits),
            "a set bit {size} above the value"
        );
        bits[1] = false;
        bits[size] = false;
        assert!(!equals_one.evaluate(&bits));
        assert!(
            ClassicalCondition::RegisterNotEquals {
                offset: 1,
                size,
                value: 1,
            }
            .evaluate(&bits)
        );
    }
}

#[test]
fn a_qasm_comparison_on_a_65_bit_register_runs() {
    let run = |high_bit: bool| {
        let source = format!(
            "OPENQASM 3.0;\nqubit[2] q;\nbit[65] c;\nx q[0];\nc[0] = measure q[0];\n{}\
             if (c == 1) x q[1];\nc[1] = measure q[1];",
            if high_bit {
                "c[64] = measure q[0];\n"
            } else {
                ""
            }
        );
        let circuit = crate::circuit::openqasm::parse(&source).unwrap();
        crate::sim::simulate(&circuit)
            .seed(42)
            .shots(4)
            .unwrap()
            .shots
    };
    assert!(run(false).iter().all(|shot| shot[1]));
    assert!(run(true).iter().all(|shot| !shot[1]));
}
