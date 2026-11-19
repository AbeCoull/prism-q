use super::*;
use crate::circuit::smallvec;
use crate::limits::ResolvedLimits;
use crate::sim::BackendKind;
use crate::sim::compiled::compile_measurements;

#[test]
fn defer_measure_reset_rewrites_reuse() {
    let mut c = Circuit::new(1, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_reset(0);
    c.add_measure(0, 1);

    let deferred = defer_measure_reset_circuit(&c).unwrap();
    assert_eq!(deferred.num_qubits, 2);
    assert!(!deferred.has_resets());
    assert!(deferred.has_terminal_measurements_only());
    assert_eq!(deferred.measurement_map(), vec![(0, 0), (1, 1)]);

    let mut sampler = compile_measurements(&deferred, 42).unwrap();
    for shot in sampler.sample_bulk(64) {
        assert!(shot[0]);
        assert!(!shot[1]);
    }
}

#[test]
fn defer_measure_reset_copies_a_reused_measured_qubit() {
    let mut c = Circuit::new(1, 2);
    c.add_gate(Gate::X, &[0]);
    c.add_measure(0, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 1);

    let deferred = defer_measure_reset_circuit(&c).unwrap();
    let gates: Vec<(Gate, Vec<usize>)> = deferred
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate { gate, targets } => Some((gate.clone(), targets.to_vec())),
            _ => None,
        })
        .collect();
    assert_eq!(
        gates,
        vec![
            (Gate::X, vec![0]),
            (Gate::Cx, vec![0, 1]),
            (Gate::H, vec![0]),
            (Gate::H, vec![0]),
        ]
    );
    assert_eq!(deferred.num_qubits, 2);
    assert_eq!(deferred.measurement_map(), vec![(1, 0), (0, 1)]);

    let mut sampler = compile_measurements(&deferred, 42).unwrap();
    for shot in sampler.sample_bulk(64) {
        assert!(shot[0] && shot[1]);
    }
}

#[test]
fn defer_measure_reset_controls_a_guarded_pauli_on_the_measured_alias() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_measure(0, 0);
    c.add_reset(0);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::Y,
        targets: smallvec![1],
    });
    c.add_measure(1, 1);

    let deferred = defer_measure_reset_circuit(&c).unwrap();
    let gates: Vec<(Gate, Vec<usize>)> = deferred
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate { gate, targets } => Some((gate.clone(), targets.to_vec())),
            _ => None,
        })
        .collect();
    assert_eq!(
        gates,
        vec![
            (Gate::H, vec![0]),
            (Gate::Sdg, vec![1]),
            (Gate::Cx, vec![0, 1]),
            (Gate::S, vec![1]),
        ]
    );
    assert!(deferred.has_terminal_measurements_only());

    let mut sampler = compile_measurements(&deferred, 42).unwrap();
    for shot in sampler.sample_bulk(256) {
        assert_eq!(shot[0], shot[1]);
    }
}

#[test]
fn defer_measure_reset_rejects_guards_no_controlled_pauli_expresses() {
    let mut hadamard = Circuit::new(2, 2);
    hadamard.add_measure(0, 0);
    hadamard.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(0),
        gate: Gate::H,
        targets: smallvec![1],
    });
    hadamard.add_measure(1, 1);

    let mut both_bits = Circuit::new(3, 3);
    both_bits.add_measure(0, 0);
    both_bits.add_measure(1, 1);
    both_bits.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 2,
            value: 0b11,
        },
        gate: Gate::X,
        targets: smallvec![2],
    });
    both_bits.add_measure(2, 2);

    for (circuit, reason) in [
        (&hadamard, "classical conditionals on `h`"),
        (&both_bits, "reads two measured bits"),
    ] {
        let err = defer_measure_reset_circuit(circuit)
            .unwrap_err()
            .to_string();
        assert!(err.contains(reason), "{err}");
    }
}

#[test]
fn defer_guarded_keeps_a_comparison_of_two_bits_as_an_entry() {
    let mut c = Circuit::new(3, 3);
    c.add_measure(0, 0);
    c.add_measure(1, 1);
    c.instructions.push(Instruction::Conditional {
        condition: ClassicalCondition::RegisterNotEquals {
            offset: 0,
            size: 2,
            value: 0b01,
        },
        gate: Gate::Z,
        targets: smallvec![2],
    });
    c.add_measure(2, 2);

    let deferred = defer_guarded_circuit(&c).unwrap();
    assert!(
        deferred
            .circuit
            .instructions
            .iter()
            .all(|inst| !matches!(inst, Instruction::Gate { .. }))
    );
    let [entry] = deferred.guarded.as_slice() else {
        panic!("{} entries", deferred.guarded.len());
    };
    assert_eq!(entry.qubit, 2);
    assert_eq!(entry.kind, GuardedKind::Gate(3));
    assert_eq!(
        entry.guard,
        Guard::All {
            terms: smallvec![(0, true), (1, false)],
            negate: true,
        }
    );
}

#[test]
fn run_shots_deferred_reset_preserves_bell_correlation() {
    let mut c = Circuit::new(2, 2);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_measure(1, 0);
    c.add_reset(1);
    c.add_measure(0, 1);

    let result = crate::sim::run_shots_with(
        BackendKind::Stabilizer,
        &c,
        1024,
        42,
        &ResolvedLimits::process_default(),
    )
    .unwrap();
    assert_eq!(result.num_classical_bits, 2);
    let mut saw_zero = false;
    let mut saw_one = false;
    for shot in &result.shots {
        assert_eq!(shot[0], shot[1]);
        saw_zero |= !shot[0];
        saw_one |= shot[0];
    }
    assert!(saw_zero);
    assert!(saw_one);
}
