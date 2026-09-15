//! `#pragma braket` parsing end to end: what a program declares beside its
//! circuit, and what the parser refuses.

mod common;

use num_complex::Complex64;
use prism_q::PrismError;
use prism_q::circuit::Instruction;
use prism_q::circuit::braket::{ObservableFactor, ResultSpec, Targets};
use prism_q::circuit::openqasm::{self, Dialect};
use prism_q::gates::Gate;
use prism_q::sim::unified_pauli::PauliAxis;

use common::{SEED, SV_EPS};

fn program(body: &str) -> String {
    format!("OPENQASM 3.0;\nqubit[2] q;\n{body}\n")
}

fn parse(body: &str) -> openqasm::BraketProgram {
    let source = program(body);
    openqasm::parse_braket(&source).unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn parse_err(body: &str) -> PrismError {
    let source = program(body);
    openqasm::parse_braket(&source)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not parse"))
}

#[test]
fn result_pragmas_reach_the_caller_in_declaration_order() {
    let parsed = parse(
        "h q[0];\n\
         cnot q[0], q[1];\n\
         #pragma braket result probability all\n\
         #pragma braket result expectation z(q[0]) @ z(q[1])\n\
         #pragma braket result state_vector",
    );
    assert_eq!(parsed.circuit.gate_count(), 2);
    assert_eq!(parsed.results.len(), 3);
    assert_eq!(parsed.results[0], ResultSpec::Probability(Targets::All));
    assert_eq!(parsed.results[2], ResultSpec::StateVector);
    let ResultSpec::Expectation(observable) = &parsed.results[1] else {
        panic!("expected an expectation");
    };
    assert_eq!(observable.factors.len(), 2);
    assert!(matches!(
        observable.factors[0],
        ObservableFactor::Pauli {
            axis: PauliAxis::Z,
            ..
        }
    ));
}

// A pragma carries no instruction, so it must leave the gate stream alone.
#[test]
fn result_pragmas_add_no_instructions() {
    let plain = parse("h q[0];\ncnot q[0], q[1];");
    let annotated = parse(
        "h q[0];\n\
         cnot q[0], q[1];\n\
         #pragma braket result state_vector",
    );
    assert_eq!(
        plain.circuit.instructions.len(),
        annotated.circuit.instructions.len()
    );
    assert!(plain.results.is_empty());
}

#[test]
fn noise_pragmas_attach_after_the_instruction_they_follow() {
    let parsed = parse(
        "h q[0];\n\
         #pragma braket noise bit_flip(0.1) q[0]\n\
         cnot q[0], q[1];\n\
         #pragma braket noise two_qubit_depolarizing(0.2) q[0], q[1]",
    );
    let noise = parsed.noise.expect("a noise model");
    assert_eq!(noise.after_gate.len(), parsed.circuit.instructions.len());
    assert_eq!(noise.after_gate[0].len(), 1, "after the h");
    assert_eq!(noise.after_gate[1].len(), 1, "after the cnot");
    assert_eq!(noise.after_gate[0][0].qubits.as_slice(), [0]);
    assert_eq!(noise.after_gate[1][0].qubits.as_slice(), [0, 1]);
}

#[test]
fn a_program_without_noise_pragmas_carries_no_model() {
    assert!(parse("h q[0];").noise.is_none());
}

// The model indexes the instruction a channel follows, so a pragma standing
// before every instruction has nothing to attach to.
#[test]
fn a_leading_noise_pragma_is_rejected() {
    assert!(matches!(
        parse_err("#pragma braket noise bit_flip(0.1) q[0]\nh q[0];"),
        PrismError::Parse { .. }
    ));
}

#[test]
fn unitary_pragma_builds_a_matrix_gate() {
    let parsed = parse("#pragma braket unitary([[0, -1im], [1im, 0]]) q[0]");
    assert_eq!(parsed.circuit.instructions.len(), 1);
    let Instruction::Gate {
        gate: Gate::Fused(matrix),
        targets,
    } = &parsed.circuit.instructions[0]
    else {
        panic!("expected a Fused gate");
    };
    assert_eq!(targets.as_slice(), [0]);
    // Pauli Y.
    assert!((matrix[0][1] - Complex64::new(0.0, -1.0)).norm() < SV_EPS);
    assert!((matrix[1][0] - Complex64::new(0.0, 1.0)).norm() < SV_EPS);

    // A CNOT with `q[0]` controlling, which is asymmetric in the matrix
    // index: a transposed or bit-reversed copy would spell the other one.
    let two = parse("#pragma braket unitary([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]) q[0], q[1]");
    let Instruction::Gate {
        gate: Gate::Fused2q(matrix),
        targets,
    } = &two.circuit.instructions[0]
    else {
        panic!("expected a Fused2q gate");
    };
    assert_eq!(targets.as_slice(), [0, 1]);
    let expected = Gate::Cx.matrix_4x4();
    for (row, entries) in matrix.iter().enumerate() {
        for (column, entry) in entries.iter().enumerate() {
            assert!(
                (entry - expected[row][column]).norm() < SV_EPS,
                "({row}, {column}) is {entry} against {}",
                expected[row][column]
            );
        }
    }
}

#[test]
fn unitary_pragma_rejects_a_non_unitary_matrix() {
    assert!(matches!(
        parse_err("#pragma braket unitary([[1, 1], [0, 1]]) q[0]"),
        PrismError::InvalidParameter { .. }
    ));
}

// Three or more targets has no matrix-gate variant, so the pragma is reduced
// to multi-controlled gates. A permutation makes that visible: it has to move
// the basis states it names and nothing else.
#[test]
fn a_wide_unitary_pragma_is_reduced_to_gates() {
    // Swap the outer two qubits of three, leaving the middle alone.
    let permutation = |row: usize, column: usize| {
        let swapped = (column & 1) << 2 | (column & 2) | (column >> 2 & 1);
        usize::from(row == swapped)
    };
    let body = format!(
        "[{}]",
        (0..8)
            .map(|r| format!(
                "[{}]",
                (0..8)
                    .map(|c| permutation(r, c).to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let source = format!(
        "OPENQASM 3.0;
qubit[3] q;
#pragma braket unitary({body}) q[0], q[1], q[2]
"
    );
    let parsed = openqasm::parse_braket(&source).expect("a wide unitary reduces");
    assert!(
        parsed.circuit.instructions.len() > 1,
        "a wide matrix needs more than one gate"
    );
    let direct = openqasm::parse(
        "OPENQASM 3.0;
qubit[3] q;
swap q[0], q[2];
",
    )
    .unwrap();
    common::assert_unitary_close(
        &common::circuit_unitary(&parsed.circuit),
        &common::circuit_unitary(&direct),
        SV_EPS,
        "wide unitary pragma",
    );
}

// Past the reduction's own span the pragma declines rather than emitting a
// number of gates nobody asked for.
#[test]
fn unitary_pragma_declines_past_the_reduction_span() {
    let side = 32usize;
    let body = format!(
        "[{}]",
        (0..side)
            .map(|r| format!(
                "[{}]",
                (0..side)
                    .map(|c| if r == c { "1" } else { "0" })
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let source = format!(
        "OPENQASM 3.0;
qubit[5] q;
#pragma braket unitary({body}) q[0], q[1], q[2], q[3], q[4]
"
    );
    assert!(matches!(
        openqasm::parse_braket(&source),
        Err(PrismError::UnsupportedConstruct { .. })
    ));
}

// Verbatim marks a region a device compiler must not rewrite. A simulator has
// nothing to honour, so the body runs as written.
#[test]
fn a_verbatim_box_runs_its_body() {
    let parsed = parse("#pragma braket verbatim\nbox {\n  h q[0];\n  cnot q[0], q[1];\n}");
    assert_eq!(parsed.circuit.gate_count(), 2);
    let probs = prism_q::simulate(&parsed.circuit)
        .seed(SEED)
        .run()
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();
    common::assert_probs_close(&probs, &[0.5, 0.0, 0.0, 0.5], SV_EPS, "verbatim bell");
}

#[test]
fn a_box_without_a_verbatim_pragma_is_rejected() {
    assert!(matches!(
        parse_err("box {\n  h q[0];\n}"),
        PrismError::UnsupportedConstruct { .. }
    ));
}

#[test]
fn a_verbatim_pragma_without_a_box_is_rejected() {
    assert!(matches!(
        parse_err("#pragma braket verbatim\nh q[0];"),
        PrismError::Parse { .. }
    ));
}

// Pragmas are Braket's extension, so reading one under the native dialect has
// to fail loudly: dropping a result request leaves a caller with nothing to
// report and no reason why. `parse_with` under the Braket dialect reads the
// pragma but has nowhere to hand back what it declared, so it keeps the
// circuit and drops the request; `parse_braket` is the entry point that
// returns both.
#[test]
fn pragmas_need_the_braket_dialect() {
    let source = program("h q[0];\n#pragma braket result state_vector");
    assert!(matches!(
        openqasm::parse_with(&source, Dialect::Native),
        Err(PrismError::UnsupportedConstruct { .. })
    ));

    let circuit = openqasm::parse_with(&source, Dialect::Braket).expect("the Braket dialect reads");
    assert_eq!(
        circuit.instructions.len(),
        1,
        "the pragma adds no instruction"
    );
    assert_eq!(
        openqasm::parse_braket(&source).unwrap().results.len(),
        1,
        "`parse_braket` is where the request survives"
    );
}

#[test]
fn an_unknown_pragma_is_named() {
    for body in [
        "h q[0];\n#pragma braket nonsense",
        "h q[0];\n#pragma qiskit something",
    ] {
        assert!(
            matches!(parse_err(body), PrismError::UnsupportedConstruct { .. }),
            "`{body}`"
        );
    }
}

// A malformed pragma names the pragma rather than falling through to gate
// parsing, where the line would read as an undefined register.
#[test]
fn a_malformed_result_pragma_names_the_pragma() {
    let err = parse_err("h q[0];\n#pragma braket result expectation nonsense(q[0])");
    let text = format!("{err}");
    assert!(
        text.contains("observable"),
        "the error should name the observable, got: {text}"
    );
}
