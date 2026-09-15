//! Classical declarations, `let` aliases, and register slicing.

mod common;

use prism_q::PrismError;
use prism_q::circuit::{Circuit, openqasm};

use common::SV_EPS;

fn parse(body: &str) -> Circuit {
    let text = format!("OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\n{body}\n");
    openqasm::parse(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn parse_err(body: &str) -> PrismError {
    let text = format!("OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\n{body}\n");
    openqasm::parse(&text)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not parse"))
}

/// The qubits a program's gates touch, in the order the instructions name them.
fn touched(circuit: &Circuit) -> Vec<Vec<usize>> {
    circuit
        .instructions
        .iter()
        .filter_map(|instr| match instr {
            prism_q::circuit::Instruction::Gate { targets, .. } => Some(targets.to_vec()),
            _ => None,
        })
        .collect()
}

fn measured(circuit: &Circuit) -> Vec<(usize, usize)> {
    circuit
        .instructions
        .iter()
        .filter_map(|instr| match instr {
            prism_q::circuit::Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((*qubit, *classical_bit)),
            _ => None,
        })
        .collect()
}

// A declared value has to reach every place a literal would: an index, a loop
// bound, a gate angle, and a condition.
#[test]
fn a_classical_value_reaches_every_position_a_literal_does() {
    for (body, expected) in [
        ("int i = 2;\nh q[i];", vec![vec![2usize]]),
        ("const int i = 1 + 2;\nh q[i];", vec![vec![3]]),
        ("uint i = 0;\nx q[i];", vec![vec![0]]),
        ("bool b = true;\nh q[b];", vec![vec![1]]),
        ("bool b = false;\nh q[b];", vec![vec![0]]),
        ("int[32] i = 3;\nh q[i];", vec![vec![3]]),
        (
            "int n = 2;\nfor int k in [0:n] { h q[k]; }",
            vec![vec![0], vec![1], vec![2]],
        ),
        ("int i = 1;\ni = i + 2;\nh q[i];", vec![vec![3]]),
        ("int i = 1;\ni += 2;\nh q[i];", vec![vec![3]]),
        ("int i = 6;\ni /= 2;\nh q[i];", vec![vec![3]]),
    ] {
        assert_eq!(touched(&parse(body)), expected, "`{body}`");
    }
}

#[test]
fn a_declared_angle_folds_into_the_gate() {
    let circuit = parse("float theta = pi / 4;\nrx(theta) q[0];\nrx(2 * theta) q[1];");
    let reference = parse("rx(pi / 4) q[0];\nrx(pi / 2) q[1];");
    let left = common::circuit_unitary(&circuit);
    let right = common::circuit_unitary(&reference);
    common::assert_unitary_close(&left, &right, SV_EPS, "declared angle");
}

#[test]
fn an_angle_declaration_is_read_as_a_float() {
    let circuit = parse("angle a = pi;\nrz(a) q[0];");
    let reference = parse("rz(pi) q[0];");
    common::assert_unitary_close(
        &common::circuit_unitary(&circuit),
        &common::circuit_unitary(&reference),
        SV_EPS,
        "angle declaration",
    );
}

#[test]
fn a_declared_value_carries_into_a_condition() {
    let circuit = parse("int v = 1;\nc[0] = measure q[0];\nif (c == v) x q[1];");
    let reference = parse("c[0] = measure q[0];\nif (c == 1) x q[1];");
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn a_loop_body_declaration_does_not_leak() {
    // `j` is re-declared on every pass, which only holds if the scope the
    // body opened is dropped at the end of each one.
    let circuit = parse("for int k in [0:1] { int j = k + 1; h q[j]; }");
    assert_eq!(touched(&circuit), vec![vec![1], vec![2]]);
    assert!(matches!(
        parse_err("for int k in [0:1] { int j = 0; }\nh q[j];"),
        PrismError::Parse { .. }
    ));
}

#[test]
fn declarations_that_cannot_stand_are_rejected() {
    for body in [
        "int i = 1;\nint i = 2;",
        "int q = 1;",
        "const int i;",
        "const int i = 1;\ni = 2;",
        "complex z = 1;",
        "int i = 1.5;",
        "int i = j;",
        "int 9x = 1;",
    ] {
        let error = parse_err(body);
        assert!(
            matches!(
                error,
                PrismError::Parse { .. } | PrismError::UnsupportedConstruct { .. }
            ),
            "`{body}` gave {error}"
        );
    }
}

#[test]
fn a_slice_names_the_qubits_it_spans() {
    for (body, expected) in [
        ("h q[0:2];", vec![vec![0usize], vec![1], vec![2]]),
        ("h q[1:1];", vec![vec![1]]),
        ("h q[0:2:3];", vec![vec![0], vec![2]]),
        ("h q[3:-1:1];", vec![vec![3], vec![2], vec![1]]),
        ("h q[:1];", vec![vec![0], vec![1]]),
        ("h q[2:];", vec![vec![2], vec![3]]),
        ("h q[{0, 3}];", vec![vec![0], vec![3]]),
        ("h q[{3, 0}];", vec![vec![3], vec![0]]),
        ("int n = 1;\nh q[0:n];", vec![vec![0], vec![1]]),
    ] {
        assert_eq!(touched(&parse(body)), expected, "`{body}`");
    }
}

// A slice on one argument of a two-qubit gate broadcasts against the other,
// which is the rule a whole register already follows.
#[test]
fn a_slice_broadcasts_like_a_register() {
    assert_eq!(
        touched(&parse("cx q[0:1], q[2:3];")),
        vec![vec![0, 2], vec![1, 3]]
    );
    assert_eq!(
        touched(&parse("cx q[0], q[1:2];")),
        vec![vec![0, 1], vec![0, 2]]
    );
}

#[test]
fn an_alias_names_qubits_in_the_order_it_was_written() {
    for (body, expected) in [
        ("let a = q[0:1];\nh a;", vec![vec![0usize], vec![1]]),
        ("let a = q[2] ++ q[0];\nh a;", vec![vec![2], vec![0]]),
        ("let a = q[2] ++ q[0];\nh a[0];", vec![vec![2]]),
        ("let a = q;\nh a[3];", vec![vec![3]]),
        (
            "let a = q[0:2];\nlet b = a[1:2];\nh b;",
            vec![vec![1], vec![2]],
        ),
        ("let a = q[{3, 1}];\nx a[1];", vec![vec![1]]),
    ] {
        assert_eq!(touched(&parse(body)), expected, "`{body}`");
    }
}

#[test]
fn an_alias_reaches_measurement_and_conditions() {
    let circuit = parse("let a = q[1] ++ q[0];\nlet d = c[2:3];\nd = measure a;");
    assert_eq!(measured(&circuit), vec![(1, 2), (0, 3)]);

    let guarded = parse("let d = c[1];\nc[1] = measure q[0];\nif (d[0]) x q[1];");
    let reference = parse("c[1] = measure q[0];\nif (c[1]) x q[1];");
    assert_eq!(
        format!("{:?}", guarded.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn an_alias_carries_into_a_gate_body() {
    let circuit = parse("let a = q[1] ++ q[2];\ngate flip x { h x; }\nflip a[1];");
    assert_eq!(touched(&circuit), vec![vec![2]]);
}

#[test]
fn aliases_and_slices_that_cannot_stand_are_rejected() {
    for body in [
        "let a = q[0:1];\nlet a = q[2];",
        "let q = q[0];",
        "let a = q[0] ++ c[0];",
        "let a = nothing[0];",
        "let a = q[0];\nh a[1];",
        "h q[0:9];",
        "h q[2:1];",
        "h q[0:0:2];",
        "h q[{}];",
        "let a = c[0:1];\nh a;",
    ] {
        let error = parse_err(body);
        assert!(
            matches!(
                error,
                PrismError::Parse { .. }
                    | PrismError::UnsupportedConstruct { .. }
                    | PrismError::UndefinedRegister { .. }
                    | PrismError::InvalidQubit { .. }
                    | PrismError::InvalidClassicalBit { .. }
            ),
            "`{body}` gave {error}"
        );
    }
}
