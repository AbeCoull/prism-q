//! Qubit operands a call cannot take: trailing operands on a `def` call, and one
//! qubit named twice in a single gate application.

use super::*;

fn every_entry_point(source: &str) -> [(&'static str, Result<()>); 4] {
    [
        ("parse", parse(source).map(drop)),
        ("parse_dynamic", parse_dynamic(source).map(drop)),
        ("parse_parametric", parse_parametric(source).map(drop)),
        ("braket", parse_with(source, Dialect::Braket).map(drop)),
    ]
}

fn assert_declined(source: &str, wanted: &str) {
    for (entry, result) in every_entry_point(source) {
        match result {
            Err(err @ PrismError::Parse { .. }) => assert!(
                err.to_string().contains(wanted),
                "{entry} on `{source}`: got {err}"
            ),
            Err(err) => panic!("{entry} on `{source}`: wanted a parse error, got {err:?}"),
            Ok(()) => panic!(
                "{entry} on `{source}` parsed to {:?}",
                parse(source).map(|circuit| circuit.instructions)
            ),
        }
    }
}

fn instructions_of(source: &str) -> String {
    let circuit = parse(source).unwrap_or_else(|e| panic!("`{source}`: {e}"));
    format!("{:?}", circuit.instructions)
}

fn gate(gate: Gate, targets: &[usize]) -> Instruction {
    Instruction::Gate {
        gate,
        targets: targets.iter().copied().collect(),
    }
}

#[test]
fn a_def_call_with_trailing_qubit_operands_declines() {
    for source in [
        "OPENQASM 3.0;\nqubit[2] q;\ndef f(qubit a) { x a; }\nf(q[0]) q[1];",
        "OPENQASM 3.0;\nqubit[2] q;\ndef g() { }\ng q;",
        "OPENQASM 3.0;\nqubit[2] q;\ndef g() { }\ng q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\ndef f(qubit a) { x a; }\nf q;",
        "OPENQASM 2.0;\nqreg q[2];\ndef f(qubit a) { x a; }\nf(q[0]) q[1];",
    ] {
        assert_declined(source, "is a function, not a gate");
    }
}

#[test]
fn a_def_call_with_its_qubits_as_arguments_still_inlines() {
    let source = "OPENQASM 3.0;\nqubit[2] q;\ndef f(qubit a) { x a; }\nf(q[1]);";
    assert_eq!(
        instructions_of(source),
        format!("{:?}", [gate(Gate::X, &[1])])
    );
}

#[test]
fn a_gate_naming_one_qubit_twice_declines() {
    let header = "OPENQASM 3.0;\nqubit[3] q;\nqubit[2] r;\ngate g a, b { cx a, b; }\n";
    for (body, repeated) in [
        ("cx q[2], q[2];", 2),
        ("cx r, r;", 3),
        ("cx q[0], q;", 0),
        ("swap q[0], q[0];", 0),
        ("ccx q[0], q[1], q[0];", 0),
        ("g q[1], q[1];", 1),
        ("inv @ cz q[1], q[1];", 1),
        ("pow(2) @ swap q[2], q[2];", 2),
        ("ctrl @ ctrl @ x q[0], q[0], q[1];", 0),
        ("ctrl @ ctrl @ gphase(0.5) q[1], q[1];", 1),
    ] {
        assert_declined(
            &format!("{header}{body}"),
            &format!("names qubit {repeated} more than once"),
        );
    }
    for (source, repeated) in [
        ("OPENQASM 2.0;\nqreg q[2];\ncx q[1], q[1];", 1),
        ("OPENQASM 2.0;\nqreg q[2];\ncx q, q;", 0),
    ] {
        assert_declined(source, &format!("names qubit {repeated} more than once"));
    }
}

#[test]
fn a_control_that_is_also_a_target_still_says_so() {
    assert_declined(
        "OPENQASM 3.0;\nqubit[2] q;\nctrl @ x q[0], q[0];",
        "qubit 0 is both a control and a target",
    );
}

#[test]
fn distinct_operands_still_parse_to_the_same_instructions() {
    let header = "OPENQASM 3.0;\nqubit[3] q;\nqubit[2] r;\ngate g a, b { cx a, b; }\n";
    for (body, expected) in [
        ("cx q[2], q[1];", vec![gate(Gate::Cx, &[2, 1])]),
        (
            "cx q[0], r;",
            vec![gate(Gate::Cx, &[0, 3]), gate(Gate::Cx, &[0, 4])],
        ),
        (
            "cx q[1:2], r;",
            vec![gate(Gate::Cx, &[1, 3]), gate(Gate::Cx, &[2, 4])],
        ),
        ("swap q[0], q[1];", vec![gate(Gate::Swap, &[0, 1])]),
        ("g q[1], q[0];", vec![gate(Gate::Cx, &[1, 0])]),
        ("ctrl @ x q[0], q[1];", vec![gate(Gate::Cx, &[0, 1])]),
    ] {
        assert_eq!(
            instructions_of(&format!("{header}{body}")),
            format!("{expected:?}"),
            "{body}"
        );
    }
    let ccx = instructions_of(&format!("{header}ccx q[0], q[1], q[2];"));
    assert!(ccx.contains("[0, 1, 2]"), "{ccx}");
}
