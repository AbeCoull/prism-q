//! Guarded-region contract: parsing, execution, fusion, export, and the
//! routes that reject a region.

use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuit::{
    Circuit, ClassicalCondition, GuardedRegion, Instruction, MAX_REGION_DEPTH, SmallVec, guarded,
    openqasm, qasm_export,
};
use prism_q::error::PrismError;
use prism_q::gates::Gate;
use prism_q::sim::unified_pauli::{PauliAxis, PauliTerm};
use prism_q::{CircuitBuilder, simulate};

const SEED: u64 = 42;

fn region(inst: &Instruction) -> &GuardedRegion {
    match inst {
        Instruction::Region(region) => region,
        other => panic!("expected a region, got {other:?}"),
    }
}

fn probabilities(circuit: &Circuit) -> Vec<f64> {
    simulate(circuit)
        .seed(SEED)
        .run()
        .expect("simulation")
        .probabilities
        .expect("probabilities")
        .to_vec()
}

#[test]
fn instruction_stays_96_bytes() {
    // The region body is boxed so the enum does not grow; inline growth here
    // is the regression this pins.
    assert_eq!(std::mem::size_of::<Instruction>(), 96);
}

#[test]
fn single_gate_body_lowers_to_conditional() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0]) { x q[0]; }";
    let circuit = openqasm::parse(qasm).expect("parse");
    assert!(matches!(
        circuit.instructions[0],
        Instruction::Conditional { .. }
    ));
}

#[test]
fn braced_body_becomes_a_region_over_its_qubits() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nbit[2] c;\n\
                if (c == 2) {\n  x q[2];\n  measure q[2] -> c[1];\n  reset q[0];\n}";
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.instructions.len(), 1);
    let region = region(&circuit.instructions[0]);
    assert_eq!(region.body().len(), 3);
    assert_eq!(region.qubits(), &[0, 2]);
    assert_eq!(region.depth(), 1);
    assert!(matches!(
        region.condition(),
        ClassicalCondition::RegisterEquals { value: 2, .. }
    ));
}

#[test]
fn empty_body_appends_nothing() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0]) { }";
    assert!(
        openqasm::parse(qasm)
            .expect("parse")
            .instructions
            .is_empty()
    );
}

#[test]
fn nesting_reports_its_depth() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\n\
                if (c[0]) {\n  x q[0];\n  if (c[1]) {\n    x q[1];\n    z q[1];\n  }\n}";
    let circuit = openqasm::parse(qasm).expect("parse");
    let outer = region(&circuit.instructions[0]);
    assert_eq!(outer.depth(), 2);
    assert_eq!(outer.qubits(), &[0, 1]);
}

fn nested_qasm(levels: usize) -> String {
    let mut qasm = String::from("OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n");
    for _ in 0..levels {
        qasm.push_str("if (c[0]) {\n");
    }
    qasm.push_str("x q[0];\nz q[0];\n");
    for _ in 0..levels {
        qasm.push_str("}\n");
    }
    qasm
}

#[test]
fn nesting_to_the_bound_is_accepted() {
    let circuit = openqasm::parse(&nested_qasm(MAX_REGION_DEPTH)).expect("parse at the bound");
    assert_eq!(region(&circuit.instructions[0]).depth(), MAX_REGION_DEPTH);
}

#[test]
fn nesting_past_the_bound_is_rejected() {
    let mut qasm = String::from("OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n");
    for _ in 0..=MAX_REGION_DEPTH {
        qasm.push_str("if (c[0]) {\n");
    }
    qasm.push_str("x q[0];\n");
    for _ in 0..=MAX_REGION_DEPTH {
        qasm.push_str("}\n");
    }
    let err = openqasm::parse(&qasm).expect_err("depth bound");
    assert!(
        matches!(&err, PrismError::Parse { message, .. } if message.contains("nest deeper")),
        "unexpected error: {err:?}"
    );
}

// `else` lowers to a second guard carrying the negated condition, in every
// shape the source can spell it: same line, own line, and unbraced.
#[test]
fn else_lowers_to_a_negated_sibling_in_every_shape() {
    let sources = [
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { x q[0]; } else { z q[0]; }",
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) {
  x q[0];
}
else {
  z q[0];
}",
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) x q[0];
else z q[0];",
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) x q[0]; else z q[0];",
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { x q[0]; }
else
z q[0];",
    ];
    for qasm in sources {
        let circuit = openqasm::parse(qasm).expect("parse");
        assert_eq!(circuit.instructions.len(), 2, "in {qasm:?}");
        let arms: Vec<_> = circuit
            .instructions
            .iter()
            .map(|inst| match inst {
                Instruction::Conditional {
                    condition, gate, ..
                } => (format!("{condition:?}"), gate.name().to_string()),
                other => panic!("expected a conditional, got {other:?}"),
            })
            .collect();
        assert_eq!(arms[0].1, "x", "in {qasm:?}");
        assert_eq!(arms[1].1, "z", "in {qasm:?}");
        assert!(arms[0].0.contains("BitIsOne"), "in {qasm:?}");
        assert!(arms[1].0.contains("BitIsZero"), "in {qasm:?}");
    }
}

#[test]
fn exactly_one_else_arm_runs() {
    let qasm = "OPENQASM 3.0;
qubit[3] q;
bit[1] c;
x q[0];
measure q[0] -> c[0];
if (c[0]) { x q[1]; } else { x q[2]; }";
    let circuit = openqasm::parse(qasm).expect("parse");
    let probs = probabilities(&circuit);
    // q0 and q1 set, q2 clear: index 0b011.
    assert!(probs[0b011] > 0.999, "{probs:?}");
}

#[test]
fn else_if_chains_nest_under_the_negated_arm() {
    let qasm = "OPENQASM 3.0;
qubit[4] q;
bit[2] c;
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c[0]) { x q[2]; } else if (c[1]) { x q[3]; } else { x q[2]; x q[3]; }";
    let circuit = openqasm::parse(qasm).expect("parse");
    let probs = probabilities(&circuit);
    // c[0] is 0 and c[1] is 1, so only the middle arm runs: q1 and q3 set.
    assert!(probs[0b1010] > 0.999, "{probs:?}");
}

// The negated arm re-reads the bits after the first body ran, so a body that
// measures into its own guard could take both arms. That is rejected, not
// lowered.
#[test]
fn else_rejects_a_body_that_overwrites_its_own_guard() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { measure q[0] -> c[0]; } else { x q[0]; }";
    match openqasm::parse(qasm).expect_err("unsound lowering") {
        PrismError::Parse { line, message } => {
            assert_eq!(line, 4);
            assert!(message.contains("overwrite"), "{message}");
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

#[test]
fn switch_lowers_to_one_region_per_case_label() {
    let qasm = "OPENQASM 3.0;
qubit[4] q;
bit[2] c;
switch (c) {
  case 0 { x q[0]; cx q[0], q[1]; }
  case 1, 2 { x q[2]; }
  default { x q[3]; z q[3]; }
}";
    let circuit = openqasm::parse(qasm).expect("parse");
    // One guard per case label, so the two-label arm emits two, then the
    // default nested once per label to spell the conjunction of its negations.
    assert_eq!(circuit.instructions.len(), 4);
    assert_eq!(region(&circuit.instructions[0]).body().len(), 2);
    let default = region(&circuit.instructions[3]);
    assert_eq!(default.depth(), 3);
}

#[test]
fn switch_selects_the_matching_case() {
    let qasm = "OPENQASM 3.0;
qubit[5] q;
bit[2] c;
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
switch (c) {
  case 0 { x q[2]; }
  case 2 { x q[3]; }
  default { x q[4]; }
}";
    let circuit = openqasm::parse(qasm).expect("parse");
    let probs = probabilities(&circuit);
    // c reads 0b10, so the `case 2` arm runs and sets q3.
    assert!(probs[0b01010] > 0.999, "{probs:?}");
}

#[test]
fn switch_default_runs_when_no_label_matches() {
    let qasm = "OPENQASM 3.0;
qubit[4] q;
bit[2] c;
x q[0];
x q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
switch (c) {
  case 0 { x q[2]; }
  case 1 { x q[2]; }
  default { x q[3]; }
}";
    let circuit = openqasm::parse(qasm).expect("parse");
    let probs = probabilities(&circuit);
    assert!(probs[0b1011] > 0.999, "{probs:?}");
}

#[test]
fn switch_rejects_a_duplicate_label_and_a_body_writing_its_operand() {
    let duplicate = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) {
  case 1 { x q[0]; }
  case 1 { z q[0]; }
}";
    match openqasm::parse(duplicate).expect_err("duplicate label") {
        PrismError::Parse { message, .. } => assert!(message.contains("twice"), "{message}"),
        other => panic!("expected a parse error, got {other:?}"),
    }

    let writes = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) {
  case 1 { measure q[0] -> c[0]; }
  case 2 { x q[0]; }
}";
    match openqasm::parse(writes).expect_err("arm writes the operand") {
        PrismError::Parse { message, .. } => {
            assert!(message.contains("overwrites"), "{message}")
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

#[test]
fn while_and_break_still_reject_by_name() {
    for (source, construct, line) in [
        (
            "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nwhile (c[0]) { x q[0]; }",
            "while",
            4,
        ),
        ("OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nbreak;", "break", 4),
    ] {
        match openqasm::parse(source).expect_err("unsupported") {
            PrismError::UnsupportedConstruct {
                construct: got,
                line: at,
            } => {
                assert_eq!(got, construct);
                assert_eq!(at, line);
            }
            other => panic!("expected UnsupportedConstruct for {construct}, got {other:?}"),
        }
    }
}

#[test]
fn negation_round_trips_every_condition_variant() {
    let variants = [
        ClassicalCondition::BitIsOne(0),
        ClassicalCondition::BitIsZero(1),
        ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 2,
            value: 3,
        },
        ClassicalCondition::RegisterNotEquals {
            offset: 1,
            size: 2,
            value: 1,
        },
        ClassicalCondition::Parity {
            bits: vec![0, 2].into(),
            expected: true,
        },
    ];
    let bits = [true, false, true, false];
    for condition in variants {
        assert_ne!(
            condition.evaluate(&bits),
            condition.negate().evaluate(&bits),
            "{condition:?}"
        );
        assert_eq!(
            condition.evaluate(&bits),
            condition.negate().negate().evaluate(&bits),
            "{condition:?}"
        );
    }
}

#[test]
fn parity_conditions_round_trip_through_qasm() {
    for source in [
        "OPENQASM 3.0;\nqubit[1] q;\nbit[3] c;\nif (c[0] ^ c[2]) { x q[0]; }",
        "OPENQASM 3.0;\nqubit[1] q;\nbit[3] c;\nif ((c[0] ^ c[2]) == 0) { x q[0]; }",
    ] {
        let circuit = openqasm::parse(source).expect("parse");
        let exported = qasm_export::to_qasm3(&circuit).expect("export");
        let reparsed = openqasm::parse(&exported).expect("reparse");
        assert_eq!(
            format!("{:?}", circuit.instructions),
            format!("{:?}", reparsed.instructions),
            "exported: {exported}"
        );
    }
}

#[test]
fn parity_selects_on_the_measured_bits() {
    let qasm = "OPENQASM 3.0;
qubit[3] q;
bit[3] c;
x q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c[0] ^ c[1]) { x q[2]; }";
    let circuit = openqasm::parse(qasm).expect("parse");
    let probs = probabilities(&circuit);
    assert!(probs[0b101] > 0.999, "{probs:?}");
}

// A `def` body can expand to a non-gate instruction. Before regions the parser
// emitted that instruction unguarded; the whole expansion is now one region.
#[test]
fn multi_instruction_expansion_stays_guarded() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\n\
                gate pair a, b { x a; cx a, b; }\nif (c[0]) pair q[0], q[1];";
    let circuit = openqasm::parse(qasm).expect("parse");
    assert_eq!(circuit.instructions.len(), 1);
    assert_eq!(region(&circuit.instructions[0]).body().len(), 2);
}

#[test]
fn taken_and_untaken_bodies_run_and_skip() {
    // q0 measures 1, so the body runs and flips q1 and q2.
    let mut taken = CircuitBuilder::new_with_classical(3, 1);
    taken.x(0).measure(0, 0).guarded(
        ClassicalCondition::BitIsOne(0),
        |body: &mut CircuitBuilder| {
            body.x(1).x(2);
        },
    );
    let probs = probabilities(&taken.build());
    assert!((probs[0b111] - 1.0).abs() < 1e-12, "{probs:?}");

    let mut untaken = CircuitBuilder::new_with_classical(3, 1);
    untaken.measure(0, 0).guarded(
        ClassicalCondition::BitIsOne(0),
        |body: &mut CircuitBuilder| {
            body.x(1).x(2);
        },
    );
    let probs = probabilities(&untaken.build());
    assert!((probs[0] - 1.0).abs() < 1e-12, "{probs:?}");
}

// A measurement inside a taken body writes a bit a later region reads.
#[test]
fn body_measurement_feeds_a_later_region() {
    let mut circuit = CircuitBuilder::new_with_classical(3, 2);
    circuit
        .x(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.x(1).measure(1, 1);
        })
        .guarded(ClassicalCondition::BitIsOne(1), |body| {
            body.x(2);
        });
    let probs = probabilities(&circuit.build());
    assert!((probs[0b111] - 1.0).abs() < 1e-12, "{probs:?}");
}

#[test]
fn nested_region_runs_only_when_both_conditions_hold() {
    let inner_then_outer = |outer_bit: bool| {
        let mut circuit = CircuitBuilder::new_with_classical(2, 2);
        if outer_bit {
            circuit.x(0);
        }
        circuit
            .measure(0, 0)
            .guarded(ClassicalCondition::BitIsOne(0), |body| {
                body.guarded(ClassicalCondition::BitIsZero(1), |inner| {
                    inner.x(1);
                });
            });
        probabilities(&circuit.build())
    };
    // Outer taken, inner reads an unwritten bit that is 0, so q1 flips.
    assert!((inner_then_outer(true)[0b11] - 1.0).abs() < 1e-12);
    assert!((inner_then_outer(false)[0] - 1.0).abs() < 1e-12);
}

// A region is opaque to every other pass, so its body is fused on its own or
// not at all. Pin both halves: that the body really does fuse, and that fusing
// it changes nothing observable on either branch of the guard.
#[test]
fn a_fused_region_body_agrees_with_the_bare_layers_on_both_branches() {
    #[derive(Clone, Copy, PartialEq)]
    enum Body {
        Guarded,
        Bare,
        Absent,
    }

    fn layers() -> Vec<Instruction> {
        let mut body = Vec::new();
        for q in 1..16u32 {
            for gate in [Gate::H, Gate::T] {
                body.push(Instruction::Gate {
                    gate,
                    targets: SmallVec::from_slice(&[q as usize]),
                });
            }
        }
        for q in 1..15usize {
            body.push(Instruction::Gate {
                gate: Gate::Cx,
                targets: SmallVec::from_slice(&[q, q + 1]),
            });
        }
        body
    }

    // BitIsZero(0) holds after measuring |0> and fails after the flip, so the
    // same guard drives both branches without touching the body.
    fn build(flip: bool, body: Body) -> Circuit {
        let mut circuit = Circuit::new(16, 1);
        if flip {
            circuit.add_gate(Gate::X, &[0]);
        }
        circuit.add_measure(0, 0);
        match body {
            Body::Guarded => circuit.instructions.push(
                guarded(ClassicalCondition::BitIsZero(0), layers()).expect("body is not empty"),
            ),
            Body::Bare => circuit.instructions.extend(layers()),
            Body::Absent => {}
        }
        circuit
    }

    let guarded_circuit = build(false, Body::Guarded);
    let fused = fuse_circuit(&guarded_circuit, true);
    let body = fused
        .instructions
        .iter()
        .find_map(|inst| match inst {
            Instruction::Region(region) => Some(region.body()),
            _ => None,
        })
        .expect("the region must survive fusion");
    assert!(
        body.len() < layers().len(),
        "fusion must shorten the body, got {} of {}",
        body.len(),
        layers().len()
    );
    assert!(
        body.iter().any(|inst| matches!(
            inst,
            Instruction::Gate {
                gate: Gate::Fused(_) | Gate::Fused2q(_) | Gate::MultiFused(_) | Gate::Multi2q(_),
                ..
            }
        )),
        "a fused body must carry at least one fused gate"
    );

    let taken = probabilities(&guarded_circuit);
    let bare = probabilities(&build(false, Body::Bare));
    for (i, (a, b)) in taken.iter().zip(&bare).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "taken branch differs at {i}: {a} vs {b}"
        );
    }

    let skipped = probabilities(&build(true, Body::Guarded));
    let absent = probabilities(&build(true, Body::Absent));
    for (i, (a, b)) in skipped.iter().zip(&absent).enumerate() {
        assert!(
            (a - b).abs() < 1e-12,
            "skipped branch differs at {i}: {a} vs {b}"
        );
    }
}

#[test]
fn fusion_never_absorbs_a_region_and_flushes_only_its_qubits() {
    let mut circuit = Circuit::new(16, 1);
    circuit.add_measure(0, 0);
    for q in 1..16 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.instructions.push(
        guarded(
            ClassicalCondition::BitIsOne(0),
            vec![Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[3]),
            }],
        )
        .expect("body is not empty"),
    );
    // Two runs of single-qubit gates on every wire, split by the region on q3.
    for q in 1..16 {
        circuit.add_gate(Gate::T, &[q]);
        circuit.add_gate(Gate::H, &[q]);
    }

    let fused = fuse_circuit(&circuit, true);
    let regions = fused
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Conditional { .. }))
        .count();
    assert_eq!(regions, 1, "the guarded gate must survive fusion");

    let unfused = probabilities(&circuit);
    let after = probabilities(fused.as_ref());
    for (a, b) in unfused.iter().zip(&after) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn a_region_is_a_barrier_only_on_the_qubits_it_touches() {
    // 12 qubits clears MIN_QUBITS_FOR_FUSION without reaching the tiled
    // multi-qubit pass, so each wire's fused run is one instruction.
    let mut circuit = Circuit::new(12, 1);
    circuit.add_measure(0, 0);
    for q in 1..12 {
        circuit.add_gate(Gate::Rx(0.3), &[q]);
    }
    circuit.instructions.push(
        guarded(
            ClassicalCondition::BitIsOne(0),
            vec![
                Instruction::Gate {
                    gate: Gate::X,
                    targets: SmallVec::from_slice(&[3]),
                },
                Instruction::Reset { qubit: 4 },
            ],
        )
        .expect("body is not empty"),
    );
    for q in 1..12 {
        circuit.add_gate(Gate::Rx(0.4), &[q]);
    }

    let fused = fuse_circuit(&circuit, true);
    let fused_1q_on = |q: usize| {
        fused
            .instructions
            .iter()
            .filter(|inst| {
                matches!(inst, Instruction::Gate { gate, targets }
                    if gate.num_qubits() == 1 && targets.as_slice() == [q])
            })
            .count()
    };
    // q3 and q4 are inside the region, so their H runs stay split. Every other
    // wire fuses its pair of Hs into one instruction.
    assert_eq!(fused_1q_on(3), 2);
    assert_eq!(fused_1q_on(4), 2);
    assert_eq!(fused_1q_on(5), 1);
}

#[test]
fn a_static_circuit_is_untouched_by_the_region_machinery() {
    let mut circuit = Circuit::new(4, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[2]);
    assert_eq!(circuit.gate_count(), 3);
    assert_eq!(circuit.t_count(), 1);
    assert!(circuit.has_terminal_measurements_only());
    assert!(!circuit.is_clifford_only());
}

// Region-body gates count toward the predicates that pick a backend; missing
// them would route a non-Clifford circuit to the stabilizer tableau.
#[test]
fn predicates_see_through_a_region_body() {
    let mut circuit = CircuitBuilder::new_with_classical(2, 1);
    circuit
        .h(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.t(1).reset(1);
        });
    let circuit = circuit.build();
    assert!(!circuit.is_clifford_only());
    assert!(circuit.has_t_gates());
    assert_eq!(circuit.t_count(), 1);
    assert!(circuit.has_resets());
    assert!(!circuit.has_terminal_measurements_only());
    assert_eq!(circuit.gate_count(), 2);
}

// A region conditioned on a bit written in another component couples the two,
// so the factored path cannot split them apart.
#[test]
fn a_region_couples_the_subsystems_its_condition_spans() {
    let mut circuit = CircuitBuilder::new_with_classical(2, 1);
    circuit
        .h(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.x(1);
        });
    let circuit = circuit.build();
    assert_eq!(circuit.independent_subsystems(), vec![vec![0, 1]]);
}

// `guarded` never builds an empty region, so stripping must not leave one.
#[test]
fn stripping_measurements_drops_a_region_it_empties() {
    let mut circuit = CircuitBuilder::new_with_classical(2, 1);
    circuit
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.measure(0, 0).measure(1, 0);
        });
    assert!(
        circuit
            .build()
            .without_measurements()
            .instructions
            .is_empty()
    );
}

#[test]
fn export_round_trips_a_nested_region() {
    let qasm = "OPENQASM 3.0;\nqubit[3] q;\nbit[2] c;\n\
                h q[0];\nc[0] = measure q[0];\n\
                if (c[0]) {\n  x q[1];\n  if (!c[1]) {\n    cx q[1], q[2];\n  }\n  reset q[2];\n}";
    let circuit = openqasm::parse(qasm).expect("parse");
    let exported = qasm_export::to_qasm3(&circuit).expect("export");
    let reparsed = openqasm::parse(&exported).expect("reparse");

    let outer = region(&reparsed.instructions[2]);
    assert_eq!(outer.body().len(), 3);
    // The inner single-gate `if` lowers back to a conditional, so the region
    // nesting collapses to one level on the round trip.
    assert_eq!(outer.depth(), 1);
    assert!(matches!(outer.body()[1], Instruction::Conditional { .. }));
    assert_eq!(outer.qubits(), &[1, 2]);
    assert_eq!(probabilities(&circuit), probabilities(&reparsed));
}

#[test]
fn unitary_only_routes_reject_a_region() {
    let mut circuit = CircuitBuilder::new_with_classical(2, 1);
    circuit
        .h(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.x(1).reset(1);
        });
    let circuit = circuit.build();
    let observable = vec![PauliTerm {
        qubit: 0,
        axis: PauliAxis::Z,
    }];

    let err = simulate(&circuit)
        .seed(SEED)
        .expectation_values(&[observable])
        .expect_err("expectation values require a unitary circuit");
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "{err:?}"
    );
}

#[test]
fn the_compiled_sampler_rejects_a_region() {
    let mut circuit = CircuitBuilder::new_with_classical(2, 2);
    circuit
        .h(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.x(1);
        })
        .measure(1, 1);
    let circuit = circuit.build();
    // The coarse predicate routes it to per-shot replay rather than the
    // one-distribution sampler, which is the cliff the contract prices.
    assert!(!circuit.has_terminal_measurements_only());
    let shots = simulate(&circuit)
        .seed(SEED)
        .shots(64)
        .expect("per-shot replay");
    assert_eq!(shots.shots.len(), 64);
}

#[test]
fn a_noise_model_rejects_a_region() {
    use prism_q::sim::noise::NoiseModel;

    let mut circuit = CircuitBuilder::new_with_classical(2, 1);
    circuit
        .h(0)
        .measure(0, 0)
        .guarded(ClassicalCondition::BitIsOne(0), |body| {
            body.x(1).z(1);
        });
    let circuit = circuit.build();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let err = noise
        .validate_for(&circuit)
        .expect_err("noise slots are per instruction");
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { reason, .. } if reason.contains("noise slots")),
        "{err:?}"
    );
}

// ---- Static-guard folding (the sampling reachability refinement) ----

fn dead_guard_circuit(with_guard: bool) -> Circuit {
    let mut circuit = Circuit::new(3, 3);
    if with_guard {
        let mut body = Circuit::new(3, 3);
        body.add_gate(Gate::X, &[0]);
        body.add_gate(Gate::Cx, &[0, 1]);
        circuit
            .instructions
            .extend(guarded(ClassicalCondition::BitIsOne(2), body.instructions));
    }
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    for q in 0..3 {
        circuit.add_measure(q, q);
    }
    circuit
}

#[test]
fn folding_drops_a_dead_guard_and_inlines_a_taken_one() {
    let dead = dead_guard_circuit(true);
    assert!(!dead.has_terminal_measurements_only());
    let folded = dead.fold_static_guards();
    assert_eq!(
        folded.instructions.len(),
        dead_guard_circuit(false).instructions.len()
    );
    assert!(folded.has_terminal_measurements_only());

    let mut taken = Circuit::new(2, 2);
    let mut body = Circuit::new(2, 2);
    body.add_gate(Gate::X, &[0]);
    body.add_gate(Gate::Cx, &[0, 1]);
    taken
        .instructions
        .extend(guarded(ClassicalCondition::BitIsZero(0), body.instructions));
    let folded = taken.fold_static_guards();
    assert_eq!(folded.instructions.len(), 2);
    assert!(
        folded
            .instructions
            .iter()
            .all(|inst| matches!(inst, Instruction::Gate { .. }))
    );
}

// The obligation the refinement carries: fold only what is genuinely constant.
#[test]
fn folding_keeps_a_guard_a_measurement_can_reach() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    let mut body = Circuit::new(2, 2);
    body.add_gate(Gate::X, &[1]);
    body.add_gate(Gate::Z, &[1]);
    circuit
        .instructions
        .extend(guarded(ClassicalCondition::BitIsOne(0), body.instructions));
    let folded = circuit.fold_static_guards();
    assert!(matches!(
        folded.instructions.last(),
        Some(Instruction::Region(_))
    ));

    // A measurement inside a live region writes bits later guards read, so a
    // guard on those bits stays too.
    let mut chained = Circuit::new(2, 2);
    chained.add_gate(Gate::H, &[0]);
    chained.add_measure(0, 0);
    let mut body = Circuit::new(2, 2);
    body.add_gate(Gate::X, &[1]);
    body.add_measure(1, 1);
    chained
        .instructions
        .extend(guarded(ClassicalCondition::BitIsOne(0), body.instructions));
    chained.instructions.extend(guarded(
        ClassicalCondition::BitIsOne(1),
        vec![
            Instruction::Gate {
                gate: Gate::Z,
                targets: SmallVec::from_slice(&[0]),
            },
            Instruction::Gate {
                gate: Gate::Z,
                targets: SmallVec::from_slice(&[1]),
            },
        ],
    ));
    let folded = chained.fold_static_guards();
    assert_eq!(folded.instructions.len(), chained.instructions.len());
}

#[test]
fn a_static_circuit_borrows_through_the_fold() {
    let circuit = dead_guard_circuit(false);
    let folded = circuit.fold_static_guards();
    assert!(matches!(folded, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn a_dead_guard_samples_the_same_shots_as_its_absence() {
    let with_guard = dead_guard_circuit(true);
    let without = dead_guard_circuit(false);
    let a = simulate(&with_guard).seed(SEED).shots(512).expect("shots");
    let b = simulate(&without).seed(SEED).shots(512).expect("shots");
    assert_eq!(a.counts(), b.counts());
}

// Obligation (ii) on a distribution the fold cannot make trivially equal: the
// folded circuit samples through the compiled path, the unfolded one runs
// per shot on the statevector, and the two must agree.
#[test]
fn folding_preserves_a_nontrivial_distribution() {
    let mut circuit = Circuit::new(3, 3);
    // Statically taken: bit 2 is unwritten and BitIsZero holds on all-zero.
    circuit.instructions.extend(guarded(
        ClassicalCondition::BitIsZero(2),
        vec![
            Instruction::Gate {
                gate: Gate::H,
                targets: SmallVec::from_slice(&[0]),
            },
            Instruction::Gate {
                gate: Gate::Cx,
                targets: SmallVec::from_slice(&[0, 1]),
            },
        ],
    ));
    circuit.add_measure(0, 0);
    // Live: bit 0 was just written, so this one survives the fold.
    circuit.instructions.extend(guarded(
        ClassicalCondition::BitIsOne(0),
        vec![
            Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[2]),
            },
            Instruction::Gate {
                gate: Gate::Z,
                targets: SmallVec::from_slice(&[2]),
            },
        ],
    ));
    circuit.add_measure(1, 1);
    circuit.add_measure(2, 2);

    let folded = circuit.fold_static_guards();
    assert!(matches!(folded, std::borrow::Cow::Owned(_)));
    assert!(
        folded
            .instructions
            .iter()
            .filter(|inst| matches!(inst, Instruction::Region(_)))
            .count()
            == 1,
        "the live guard survives and the taken one inlined"
    );

    let shots = 4096;
    let counts = simulate(&circuit).seed(SEED).shots(shots).expect("shots");
    let reference = simulate(&folded)
        .seed(SEED + 1)
        .shots(shots)
        .expect("shots");
    let share = |result: &prism_q::ShotsResult, key: &[u64]| {
        *result.counts().get(key).unwrap_or(&0) as f64 / shots as f64
    };
    // q0 and q1 agree; q2 is set only on the 1 branch. So 000 and 111.
    for key in [vec![0u64], vec![0b111u64]] {
        let a = share(&counts, &key);
        let b = share(&reference, &key);
        assert!((a - b).abs() < 0.05, "{key:?}: {a} against {b}");
        assert!((a - 0.5).abs() < 0.05, "{key:?} should be about half: {a}");
    }
}

// `static_value` must decline a condition it cannot decide: a register
// straddling a written bit, and a bit past the classical register.
#[test]
fn folding_declines_a_partially_written_register_and_an_out_of_range_bit() {
    let mut straddle = Circuit::new(2, 2);
    straddle.add_gate(Gate::H, &[0]);
    straddle.add_measure(0, 0);
    straddle.instructions.extend(guarded(
        ClassicalCondition::RegisterEquals {
            offset: 0,
            size: 2,
            value: 0,
        },
        vec![
            Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[1]),
            },
            Instruction::Gate {
                gate: Gate::Z,
                targets: SmallVec::from_slice(&[1]),
            },
        ],
    ));
    assert!(matches!(
        straddle.fold_static_guards().instructions.last(),
        Some(Instruction::Region(_))
    ));

    let mut out_of_range = Circuit::new(1, 1);
    out_of_range.instructions.extend(guarded(
        ClassicalCondition::BitIsZero(7),
        vec![
            Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[0]),
            },
            Instruction::Gate {
                gate: Gate::Z,
                targets: SmallVec::from_slice(&[0]),
            },
        ],
    ));
    let folded = out_of_range.fold_static_guards();
    assert!(matches!(folded, std::borrow::Cow::Borrowed(_)));
}

#[test]
fn switch_default_nesting_respects_the_enclosing_depth() {
    let mut qasm = String::from("OPENQASM 3.0;\nqubit[1] q;\nbit[2] c;\n");
    for _ in 0..MAX_REGION_DEPTH - 1 {
        qasm.push_str("if (c[0]) {\n");
    }
    qasm.push_str("switch (c) { case 0 { x q[0]; } case 1 { z q[0]; } default { h q[0]; } }\n");
    for _ in 0..MAX_REGION_DEPTH - 1 {
        qasm.push_str("}\n");
    }
    let err = openqasm::parse(&qasm).expect_err("the default would nest past the bound");
    assert!(
        matches!(&err, PrismError::Parse { message, .. } if message.contains("depth bound")),
        "unexpected error: {err:?}"
    );
}

#[test]
fn a_single_bit_parity_round_trips_as_a_bit_test() {
    for (expected, spelling) in [(true, "c[1]"), (false, "!c[1]")] {
        let mut circuit = Circuit::new(1, 2);
        circuit.instructions.extend(guarded(
            ClassicalCondition::Parity {
                bits: vec![1].into(),
                expected,
            },
            vec![Instruction::Gate {
                gate: Gate::X,
                targets: SmallVec::from_slice(&[0]),
            }],
        ));
        let exported = qasm_export::to_qasm3(&circuit).expect("export");
        assert!(exported.contains(spelling), "{exported}");
        let reparsed = openqasm::parse(&exported).expect("reparse");
        let bits = [false, true];
        let condition = match &reparsed.instructions[0] {
            Instruction::Conditional { condition, .. } => condition,
            other => panic!("expected a conditional, got {other:?}"),
        };
        assert_eq!(condition.evaluate(&bits), expected);
    }
}

#[test]
fn export_rejects_a_parity_past_the_classical_register() {
    let mut circuit = Circuit::new(1, 1);
    circuit.instructions.extend(guarded(
        ClassicalCondition::Parity {
            bits: vec![0, 5].into(),
            expected: true,
        },
        vec![Instruction::Gate {
            gate: Gate::X,
            targets: SmallVec::from_slice(&[0]),
        }],
    ));
    assert!(matches!(
        qasm_export::to_qasm3(&circuit),
        Err(PrismError::ExportUnsupported { .. })
    ));
}

#[test]
fn a_parity_term_carrying_a_comparison_is_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[2] c;\nif (c[0] == 1 ^ c[1]) x q[0];";
    assert!(matches!(
        openqasm::parse(qasm),
        Err(PrismError::Parse { .. })
    ));
}

// An `else` whose arm never arrives keeps the block open rather than dropping
// the guard and letting a following statement run unconditionally.
#[test]
fn an_else_with_no_arm_is_rejected() {
    for qasm in [
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { x q[0]; } else",
        "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { x q[0]; }
else",
    ] {
        assert!(
            matches!(openqasm::parse(qasm), Err(PrismError::Parse { .. })),
            "dangling else accepted in {qasm:?}"
        );
    }
}

// Both arms are guarded in every spelling, so exactly one body runs and nothing
// leaks out unguarded.
#[test]
fn no_else_spelling_emits_an_unguarded_body() {
    let sources = [
        "if (c[0]) x q[1]; else x q[2];",
        "if (c[0]) { x q[1]; }
else
x q[2];",
        "if (c[0]) { x q[1]; }
else
{ x q[2]; }",
    ];
    for tail in sources {
        let qasm = format!(
            "OPENQASM 3.0;
qubit[3] q;
bit[1] c;
measure q[0] -> c[0];
{tail}"
        );
        let circuit = openqasm::parse(&qasm).expect("parse");
        assert!(
            circuit
                .instructions
                .iter()
                .all(|inst| !matches!(inst, Instruction::Gate { .. })),
            "an arm escaped its guard in {tail:?}: {:?}",
            circuit.instructions
        );
        let probs = probabilities(&circuit);
        // c[0] reads 0, so only the else arm runs and q2 is set.
        assert!(probs[0b100] > 0.999, "{tail:?}: {probs:?}");
    }
}
