use super::*;
use crate::circuit::qasm::{
    expr as qasm_expr, lexer,
    stream::{MAX_NESTING, Stream},
};

mod classical;
mod control_flow;
mod defs;
mod dynamic;
mod gates;
mod legacy_gates;

fn eval_expr(source: &str, line: usize, vars: Option<&HashMap<&str, f64>>) -> Result<f64> {
    let tokens = lexer::tokenize(source)?;
    let mut stream = Stream::new(&tokens);
    let expr = qasm_expr::parse(&mut stream)?;
    if !stream.at_end() {
        return Err(stream.expected("the end of the expression"));
    }
    let vars = vars.map(|vars| {
        vars.iter()
            .map(|(name, value)| (*name, ClassicalValue::Float(*value)))
            .collect::<HashMap<_, _>>()
    });
    Ok(qasm_expr::eval(&expr, line, vars.as_ref())?.to_float())
}

fn angle_of(source: &str) -> f64 {
    let circuit = parse(source).unwrap_or_else(|e| panic!("`{source}`: {e}"));
    match &circuit.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => *theta,
        other => panic!("expected an rx, got {other:?}"),
    }
}

fn angle_err(source: &str) -> PrismError {
    parse(source)
        .err()
        .unwrap_or_else(|| panic!("`{source}` should not parse"))
}

fn rx(expression: &str) -> String {
    format!("OPENQASM 3.0;\nqubit[1] q;\nrx({expression}) q[0];")
}

fn nested_parens(depth: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx({}1{}) q[0];",
        "(".repeat(depth),
        ")".repeat(depth)
    )
}

fn nested_ifs(depth: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n{}x q[0];",
        "if (c[0]) ".repeat(depth)
    )
}

fn long_sum(terms: usize) -> String {
    format!(
        "OPENQASM 3.0;\nqubit[1] q;\nrx(0{}) q[0];",
        " + 0.001".repeat(terms - 1)
    )
}

fn assert_too_deep(result: Result<impl std::fmt::Debug>) {
    let err = result.unwrap_err().to_string();
    assert!(err.contains("nests deeper than"), "{err}");
}

fn parse_within(source: &str, budget: usize) -> Result<Circuit> {
    let parser = Parser::new_with(source, Dialect::Native);
    parser.expansion_left.set(budget);
    parser.parse().map(|(circuit, _)| circuit)
}

fn assert_expands_past(result: Result<Circuit>) {
    let err = result.unwrap_err().to_string();
    assert!(err.contains("expands past"), "{err}");
}

fn rx_angles(source: &str) -> Vec<f64> {
    parse(source)
        .unwrap()
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::Rx(theta),
                ..
            } => Some(*theta),
            _ => None,
        })
        .collect()
}

fn same_instructions(a: &str, b: &str) {
    let (left, right) = (parse(a).unwrap(), parse(b).unwrap());
    assert_eq!(
        format!("{:?}", left.instructions),
        format!("{:?}", right.instructions),
        "`{a}` against `{b}`"
    );
}

/// The angle of the `rx` that `body` applies, after a two-qubit, four-bit header.
fn rx_angle(body: &str) -> f64 {
    let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
    let circuit = parse(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    match circuit.instructions.as_slice() {
        [
            Instruction::Gate {
                gate: Gate::Rx(angle),
                ..
            },
        ] => *angle,
        other => panic!("{body}: expected one rx, got {other:?}"),
    }
}

/// The guard `body`'s one conditional gate carries, after the same header.
fn guard_of(body: &str) -> ClassicalCondition {
    let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
    let circuit = parse(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    match circuit.instructions.as_slice() {
        [Instruction::Conditional { condition, .. }] => condition.clone(),
        other => panic!("{body}: expected one conditional, got {other:?}"),
    }
}

fn parse_error_of(body: &str) -> PrismError {
    parse(&format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}"))
        .err()
        .unwrap_or_else(|| panic!("{body}: parsed"))
}

fn rz_angles(source: &str) -> Vec<f64> {
    parse(source)
        .unwrap_or_else(|e| panic!("`{source}`: {e}"))
        .instructions
        .iter()
        .filter_map(|instruction| match instruction {
            Instruction::Gate {
                gate: Gate::Rz(angle),
                ..
            } => Some(*angle),
            _ => None,
        })
        .collect()
}

fn complex_parts(declarations: &str, value: &str) -> (f64, f64) {
    let source = format!(
        "OPENQASM 3.0;\nqubit q;\n{declarations}\nrz(real({value})) q;\nrz(imag({value})) q;"
    );
    match rz_angles(&source).as_slice() {
        [re, im] => (*re, *im),
        other => panic!("`{source}` emitted {other:?}"),
    }
}

fn array_values(declarations: &str, array: &str, count: usize) -> Vec<f64> {
    let reads: String = (0..count)
        .map(|at| format!("rz({array}[{at}]) q;\n"))
        .collect();
    rz_angles(&format!("OPENQASM 3.0;\nqubit q;\n{declarations}\n{reads}"))
}
