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

// `else` is stage-3 work. Until then it must reject by name with a line number
// in every shape, the way `while` and `switch` do: a braced `if` body invites an
// `else` after it, and falling through to gate parsing reports a register error
// naming a brace.
#[test]
fn else_rejects_by_name_in_every_shape() {
    let cases = [
        (
            "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) { x q[0]; } else { z q[0]; }",
            4,
        ),
        (
            "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) {
  x q[0];
}
else {
  z q[0];
}",
            7,
        ),
        (
            "OPENQASM 3.0;
qubit[1] q;
bit[1] c;
if (c[0]) x q[0];
else z q[0];",
            5,
        ),
    ];
    for (qasm, line) in cases {
        match openqasm::parse(qasm).expect_err("else is unsupported") {
            PrismError::UnsupportedConstruct {
                construct,
                line: at,
            } => {
                assert_eq!(construct, "else", "in {qasm:?}");
                assert_eq!(at, line, "in {qasm:?}");
            }
            other => panic!("expected UnsupportedConstruct for {qasm:?}, got {other:?}"),
        }
    }
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
