//! Classical expressions, typed values, complex numbers and arrays.

use super::*;

#[test]
fn test_expr_literal_float() {
    assert!((eval_expr("1.234", 0, None).unwrap() - 1.234).abs() < 1e-12);
    assert!((eval_expr("0.25", 0, None).unwrap() - 0.25).abs() < 1e-12);
    assert!((eval_expr("-0.5", 0, None).unwrap() - (-0.5)).abs() < 1e-12);
}

#[test]
fn test_expr_pi_constant() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi", 0, None).unwrap() - pi).abs() < 1e-12);
    assert!((eval_expr("-pi", 0, None).unwrap() - (-pi)).abs() < 1e-12);
    assert!((eval_expr("π", 0, None).unwrap() - pi).abs() < 1e-12);
}

#[test]
fn test_expr_tau_constant() {
    assert!((eval_expr("tau", 0, None).unwrap() - std::f64::consts::TAU).abs() < 1e-12);
}

#[test]
fn test_expr_euler_constant() {
    assert!((eval_expr("euler", 0, None).unwrap() - std::f64::consts::E).abs() < 1e-12);
    assert!(
        eval_expr("e", 0, None).is_err(),
        "OpenQASM 3 spells it `euler`"
    );
}

#[test]
fn a_declared_variable_shadows_a_constant_name() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
float e = 0.5;
gate g(euler) a { ry(euler) a; }
                rx(e) q[0];
g(0.25) q[0];";
    let c = parse(qasm).unwrap();
    let angles: Vec<f64> = c
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Gate {
                gate: Gate::Rx(theta) | Gate::Ry(theta),
                ..
            } => Some(*theta),
            _ => None,
        })
        .collect();
    assert_eq!(angles, [0.5, 0.25]);
}

#[test]
fn test_expr_pi_division() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi/2", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
    assert!((eval_expr("pi/4", 0, None).unwrap() - pi / 4.0).abs() < 1e-12);
    assert!((eval_expr("-pi/2", 0, None).unwrap() - (-pi / 2.0)).abs() < 1e-12);
}

#[test]
fn test_expr_pi_multiplication() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("2*pi", 0, None).unwrap() - 2.0 * pi).abs() < 1e-12);
    assert!((eval_expr("0.5*pi", 0, None).unwrap() - 0.5 * pi).abs() < 1e-12);
}

#[test]
fn test_expr_arithmetic() {
    assert!((eval_expr("1 + 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
    assert!((eval_expr("5 - 3", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("2 * 3", 0, None).unwrap() - 6.0).abs() < 1e-12);
    assert!((eval_expr("6 / 2", 0, None).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn test_expr_operator_precedence() {
    assert!((eval_expr("2 + 3 * 4", 0, None).unwrap() - 14.0).abs() < 1e-12);
    assert!((eval_expr("2 * 3 + 4", 0, None).unwrap() - 10.0).abs() < 1e-12);
    assert!((eval_expr("10 - 2 * 3", 0, None).unwrap() - 4.0).abs() < 1e-12);
    assert!((eval_expr("10 / 2 + 3", 0, None).unwrap() - 8.0).abs() < 1e-12);
}

#[test]
fn test_expr_parentheses() {
    assert!((eval_expr("(2 + 3) * 4", 0, None).unwrap() - 20.0).abs() < 1e-12);
    assert!((eval_expr("2 * (3 + 4)", 0, None).unwrap() - 14.0).abs() < 1e-12);
    assert!((eval_expr("((1 + 2) * (3 + 4))", 0, None).unwrap() - 21.0).abs() < 1e-12);
}

#[test]
fn test_expr_complex_pi_expressions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("pi/2 + pi/4", 0, None).unwrap() - (pi / 2.0 + pi / 4.0)).abs() < 1e-12);
    assert!((eval_expr("2*pi/3", 0, None).unwrap() - (2.0 * pi / 3.0)).abs() < 1e-12);
    assert!((eval_expr("pi/2 + 0.1", 0, None).unwrap() - (pi / 2.0 + 0.1)).abs() < 1e-12);
    assert!((eval_expr("(pi + pi)/4", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
}

#[test]
fn test_expr_unary_minus() {
    assert!((eval_expr("-1", 0, None).unwrap() - (-1.0)).abs() < 1e-12);
    assert!((eval_expr("-(2 + 3)", 0, None).unwrap() - (-5.0)).abs() < 1e-12);
    assert!((eval_expr("-(-1)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("--1", 0, None).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn test_expr_unary_plus() {
    assert!((eval_expr("+1", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("+(2 + 3)", 0, None).unwrap() - 5.0).abs() < 1e-12);
}

#[test]
fn test_expr_math_functions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("sin(pi/2)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("cos(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("sqrt(4)", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("exp(0)", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("ln(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("abs(-5)", 0, None).unwrap() - 5.0).abs() < 1e-12);
    assert!((eval_expr("asin(1)", 0, None).unwrap() - pi / 2.0).abs() < 1e-12);
    assert!((eval_expr("acos(1)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("atan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("tan(0)", 0, None).unwrap() - 0.0).abs() < 1e-12);
    assert!((eval_expr("log2(8)", 0, None).unwrap() - 3.0).abs() < 1e-12);
    assert!((eval_expr("floor(2.7)", 0, None).unwrap() - 2.0).abs() < 1e-12);
    assert!((eval_expr("ceil(2.1)", 0, None).unwrap() - 3.0).abs() < 1e-12);
}

#[test]
fn test_expr_nested_functions() {
    let pi = std::f64::consts::PI;
    assert!((eval_expr("sin(pi/4) * sin(pi/4)", 0, None).unwrap() - 0.5).abs() < 1e-12);
    assert!((eval_expr("sqrt(sin(pi/2))", 0, None).unwrap() - 1.0).abs() < 1e-12);
    assert!((eval_expr("asin(sin(pi/6))", 0, None).unwrap() - pi / 6.0).abs() < 1e-10);
}

#[test]
fn test_expr_variables() {
    let mut vars = HashMap::new();
    vars.insert("theta", std::f64::consts::FRAC_PI_4);
    vars.insert("phi", std::f64::consts::FRAC_PI_2);
    let v = Some(&vars);
    assert!((eval_expr("theta", 0, v).unwrap() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    assert!(
        (eval_expr("theta + phi", 0, v).unwrap()
            - (std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
            .abs()
            < 1e-12
    );
    assert!(
        (eval_expr("theta/2", 0, v).unwrap() - std::f64::consts::FRAC_PI_4 / 2.0).abs() < 1e-12
    );
    assert!(
        (eval_expr("2*theta + phi", 0, v).unwrap()
            - (2.0 * std::f64::consts::FRAC_PI_4 + std::f64::consts::FRAC_PI_2))
            .abs()
            < 1e-12
    );
    assert!(
        (eval_expr("sin(theta)", 0, v).unwrap() - std::f64::consts::FRAC_PI_4.sin()).abs() < 1e-12
    );
}

#[test]
fn test_expr_division_by_zero() {
    assert!(eval_expr("1/0", 0, None).is_err());
    assert!(eval_expr("pi/0", 0, None).is_err());
}

#[test]
fn test_expr_unknown_function() {
    assert!(eval_expr("foobar(1)", 0, None).is_err());
}

#[test]
fn test_expr_unknown_variable() {
    assert!(eval_expr("xyz", 0, None).is_err());
}

#[test]
fn test_expr_unbalanced_parens() {
    assert!(eval_expr("(1 + 2", 0, None).is_err());
    assert!(eval_expr("sin(pi/2", 0, None).is_err());
}

#[test]
fn test_expr_empty() {
    assert!(eval_expr("", 0, None).is_err());
}

#[test]
fn test_expr_trailing_chars() {
    assert!(eval_expr("1 2", 0, None).is_err());
}

#[test]
fn test_expr_scientific_notation() {
    assert!((eval_expr("1e-3", 0, None).unwrap() - 0.001).abs() < 1e-15);
    assert!((eval_expr("2.5E2", 0, None).unwrap() - 250.0).abs() < 1e-12);
    assert!((eval_expr("1.5e+2", 0, None).unwrap() - 150.0).abs() < 1e-12);
}

#[test]
fn test_qasm_arithmetic_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(pi/2 + pi/4) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                let expected = std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_4;
                assert!((theta - expected).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_qasm_multiply_pi() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(2*pi/3) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - 2.0 * std::f64::consts::PI / 3.0).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn test_qasm_function_in_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sqrt(2)/2) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::P(theta) => {
                assert!((theta - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
            }
            _ => panic!("expected P"),
        }
    }
}

#[test]
fn test_qasm_nested_function_in_param() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\np(sin(pi/4)) q[0];";
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::P(theta) => {
                assert!((theta - (std::f64::consts::FRAC_PI_4).sin()).abs() < 1e-12);
            }
            _ => panic!("expected P"),
        }
    }
}

#[test]
fn rx_with_arithmetic_overflow_rejected_as_non_finite() {
    // Pure arithmetic overflow: each literal is finite, but the product
    // exceeds f64::MAX. The top-level eval_expr finiteness check catches this
    // path; the per-function and per-literal checks do not.
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(1e308 * 1e308) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("must be finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_sqrt_of_negative_rejected_as_non_finite() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(sqrt(-1)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_acos_out_of_domain_rejected_as_non_finite() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(acos(2)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected non-finite parameter rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("non-finite"),
        "expected finiteness error, got: {msg}"
    );
}

#[test]
fn rx_with_division_by_zero_rejected() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(1.0 / (1.0 - 1.0)) q[0];
    "#;
    let err = parse(qasm).expect_err("expected division-by-zero rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("division by zero"),
        "expected division-by-zero error, got: {msg}"
    );
}

#[test]
fn rx_with_finite_expression_still_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
}

#[test]
fn parametric_gate_accepts_hex_literal() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(0x2 * pi / 8) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => {
            assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn parametric_gate_accepts_underscore_in_hex() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrz(0xff_ff * 0) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rz(theta),
            ..
        } => {
            assert!(theta.abs() < 1e-12);
        }
        _ => panic!("expected Rz"),
    }
}

#[test]
fn parametric_gate_accepts_boolean_literal() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nrx(true * pi) q[0];";
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            ..
        } => {
            assert!((theta - std::f64::consts::PI).abs() < 1e-12);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn expression_operators_follow_openqasm_precedence() {
    let cases = [
        ("2 ** 3", 8.0),
        // `**` binds tighter than unary minus, so this is -(2^2).
        ("-2 ** 2", -4.0),
        // and it is right associative.
        ("2 ** 3 ** 2", 512.0),
        ("2 ** -1", 0.5),
        ("2 ** 3 * 4", 32.0),
        ("7 % 3", 1.0),
        ("-7 % 3", -1.0),
        ("mod(7.5, 2)", 1.5),
        ("1 + 2 * 3 % 4", 3.0),
    ];
    for (expression, expected) in cases {
        let actual = angle_of(&rx(expression));
        assert!(
            (actual - expected).abs() < 1e-12,
            "`{expression}`: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn builtin_functions_take_their_spec_spellings() {
    let cases = [
        ("arcsin(1.0)", std::f64::consts::FRAC_PI_2),
        ("arccos(1.0)", 0.0),
        ("arctan(1.0)", std::f64::consts::FRAC_PI_4),
        ("asin(1.0)", std::f64::consts::FRAC_PI_2),
        ("ceiling(0.2)", 1.0),
        ("ceil(0.2)", 1.0),
        ("floor(1.8)", 1.0),
        ("log(euler)", 1.0),
        ("ln(euler)", 1.0),
        ("abs(-2.5)", 2.5),
        ("mod(7, 3)", 1.0),
        ("pow(2, 10)", 1024.0),
        ("popcount(7)", 3.0),
        ("popcount(0)", 0.0),
        ("popcount(0xff)", 8.0),
    ];
    for (expression, expected) in cases {
        let actual = angle_of(&rx(expression));
        assert!(
            (actual - expected).abs() < 1e-12,
            "`{expression}`: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn malformed_expressions_name_what_is_wrong() {
    for (expression, needle) in [
        ("mod(7)", "argument"),
        ("pow(2, 3, 4)", "argument"),
        ("popcount(1.5)", "popcount"),
        ("popcount(-1)", "popcount"),
        ("1 % 0", "modulo by zero"),
        ("1 / 0", "division by zero"),
        ("sizeof(2)", "declared array"),
    ] {
        let err = format!("{}", angle_err(&rx(expression)));
        assert!(
            err.contains(needle),
            "`{expression}`: expected `{needle}` in `{err}`"
        );
    }
}

// A static integer stores what a runtime one of its type would: truncation
// toward zero, then wraparound to the declared width.
#[test]
fn integer_assignments_truncate_and_wrap_to_the_declared_width() {
    let angles = rx_angles(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         int n = 7;\nn /= 2;\nrx(n) q[0];\n\
         int m = -7;\nm /= 2;\nrx(m) q[0];\n\
         uint[8] u = 300;\nrx(u) q[0];\n\
         uint[8] v = 250;\nv += 10;\nrx(v) q[0];\n\
         int[8] w = 127;\nw += 1;\nrx(w) q[0];\n\
         uint[4] z = 3;\nz -= 4;\nrx(z) q[0];\n\
         bool b = false;\nb += 2;\nrx(b) q[0];",
    );
    assert_eq!(angles, [3.0, -3.0, 44.0, 4.0, -128.0, 15.0, 1.0]);
}

#[test]
fn an_unsigned_integer_wraps_below_zero_and_back() {
    let angles = rx_angles(
        "OPENQASM 3.0;\nqubit[1] q;\n\
         uint x = 0;\nx -= 1;\nx += 1;\nrx(x) q[0];\n\
         uint[64] y = 0;\ny -= 1;\ny += 1;\nrx(y) q[0];\n\
         uint[64] z = 0;\nz -= 2;\nz += 3;\nrx(z) q[0];\n\
         uint[64] w = 0;\nw -= 1;\nuint[64] v = w;\nv += 1;\nrx(v) q[0];",
    );
    assert_eq!(angles, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn built_in_names_cannot_be_redeclared_at_the_top_level() {
    for decl in [
        "int true = 5;",
        "bool false = true;",
        "float pi = 3;",
        "float \u{3c0} = 3;",
        "float tau = 1;",
        "const float euler = 1;",
        "uint[8] true;",
    ] {
        let err = parse(&format!("OPENQASM 3.0;\nqubit[1] q;\n{decl}\nrx(1) q[0];"))
            .expect_err(decl)
            .to_string();
        assert!(err.contains("built-in"), "{decl}: {err}");
    }
}

#[test]
fn a_negative_index_counts_from_the_end() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\n";
    same_instructions(&format!("{head}x q[-1];"), &format!("{head}x q[3];"));
    same_instructions(&format!("{head}h q[1:-1];"), &format!("{head}h q[1:3];"));
    same_instructions(
        &format!("{head}h q[{{-1, 0}}];"),
        &format!("{head}h q[3];\nh q[0];"),
    );
    same_instructions(
        &format!("{head}const array[int, 3] a = {{0, 1, 2}};\nx q[a[-1]];"),
        &format!("{head}x q[2];"),
    );
}

#[test]
fn a_bit_declared_from_a_measurement_measures_into_it() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[2] q;\nh q;\nbit b = measure q[0];\nbit[2] c = measure q;",
        "OPENQASM 3.0;\nqubit[2] q;\nh q;\nbit b;\nb = measure q[0];\nbit[2] c;\nc = measure q;",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\n\
         def mx(qubit a) -> bit { h a; bit r = measure a; return r; }\nc[0] = mx(q[0]);",
        "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];\nc[0] = measure q[0];",
    );
}

#[test]
fn a_bit_initializer_of_zeros_is_the_cleared_register() {
    for decl in [
        "bit b = 0;",
        "bit[4] b = \"0000\";",
        "bit[4] b = \"00_00\";",
        "bit[3] b = 0;",
    ] {
        let c = parse(&format!("OPENQASM 3.0;\n{decl}\nqubit[1] q;\nx q[0];"))
            .unwrap_or_else(|e| panic!("{decl}: {e}"));
        assert_eq!(c.instructions.len(), 1, "{decl}");
    }
}

#[test]
fn a_bit_initializer_with_a_set_bit_is_declined_by_name() {
    for decl in ["bit[2] f = \"11\";", "bit b = 1;", "bit[3] f = 4;"] {
        let err = parse(&format!("OPENQASM 3.0;\n{decl}")).unwrap_err();
        assert!(
            matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("classical write")),
            "{decl}: {err}"
        );
    }
}

#[test]
fn a_bit_string_must_match_its_register() {
    for (decl, wanted) in [
        ("bit[3] f = \"01\";", "has 2 bit(s)"),
        ("bit[2] f = \"0a\";", "neither 0 nor 1"),
    ] {
        let err = parse(&format!("OPENQASM 3.0;\n{decl}")).unwrap_err();
        assert!(
            matches!(err, PrismError::Parse { .. }) && err.to_string().contains(wanted),
            "{decl}: {err}"
        );
    }
}

#[test]
fn integer_division_truncates_toward_zero() {
    assert_eq!(rx_angle("rx(1 / 2) q[0];"), 0.0);
    assert_eq!(rx_angle("int a = 7; int b = -2; rx(a / b) q[0];"), -3.0);
    assert_eq!(rx_angle("rx(7 % -2) q[0];"), 1.0);
    assert_eq!(rx_angle("rx(1.0 / 2) q[0];"), 0.5);
    assert_eq!(rx_angle("rx(pi / 2) q[0];"), std::f64::consts::FRAC_PI_2);
}

#[test]
fn integer_arithmetic_stays_exact_past_double_precision() {
    let body = "int x = 9007199254740993; rx(x - 9007199254740992) q[0];";
    assert_eq!(rx_angle(body), 1.0);
    assert_eq!(rx_angle("rx(0x10 + 0b11 + 0o7 + 1_000) q[0];"), 1026.0);
}

#[test]
fn a_zero_divisor_is_a_parse_error() {
    for (body, wanted) in [
        ("rx(1 / 0) q[0];", "division by zero"),
        ("rx(1 % 0) q[0];", "modulo by zero"),
        ("int a = 3; a /= 0; rx(a) q[0];", "division by zero"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn fixed_width_runtime_angles_decline_without_losing_precision() {
    for width in [1, 2, 64] {
        for body in [
            format!("angle[{width}] a = pi * int(c); rx(a) q;"),
            format!("angle[{width}] a = 0; if (c) a = pi / 3; rx(a) q;"),
            format!("angle[{width}] a = 0; while (c) {{ a += pi / 3; c = 0; }} rx(a) q;"),
            format!("rx(angle[{width}](pi * int(c))) q;"),
            format!("float a = angle[{width}](pi * int(c)); rx(a) q;"),
            format!("def f(bit b) -> angle[{width}] {{ return pi * int(b); }} rx(f(c)) q;"),
        ] {
            let source = format!("qubit q; bit c; h q; c = measure q; {body}");
            assert!(
                matches!(parse_dynamic(&source),
                Err(PrismError::UnsupportedConstruct { construct, .. })
                if construct.contains("fixed-width") && construct.contains("angle")),
                "{body}"
            );
        }
    }
    for body in ["angle[65] a = int(c);", "rx(angle[65](int(c))) q;"] {
        let source = format!("qubit q; bit c; c = measure q; {body}");
        assert!(
            matches!(parse_dynamic(&source), Err(PrismError::Parse { .. })),
            "{body}"
        );
    }
}

#[test]
fn folded_angle_precision_and_widthless_runtime_angles_are_preserved() {
    for body in [
        "angle[1] a = 3 * pi / 2; rx(a) q;",
        "angle[2] a = 0; a = pi / 3; rx(a) q;",
        "angle[64] a = -pi / 2; rx(a) q;",
        "rx(angle[2](pi / 3)) q;",
    ] {
        let source = format!("qubit q; {body}");
        let expected = parse(&source).unwrap();
        let program = parse_dynamic(&source).unwrap();
        assert_eq!(
            format!("{:?}", program.static_circuit().unwrap().instructions),
            format!("{:?}", expected.instructions),
            "{body}"
        );
    }
    let program = parse_dynamic(
        "qubit q; bit[2] c; h q; c[0] = measure q;
         angle a = -pi * int(c[0]); rx(angle(a)) q; c[1] = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| !shot[1]));
    assert!(shots.shots.iter().any(|shot| shot[0]));
    assert!(shots.shots.iter().any(|shot| !shot[0]));
}

#[test]
fn casts_convert_to_their_type() {
    for (body, want) in [
        ("rx(int(2.7)) q[0];", 2.0),
        ("rx(int(-2.7)) q[0];", -2.0),
        ("rx(int[8](300)) q[0];", 44.0),
        ("rx(int[8](200)) q[0];", -56.0),
        ("rx(uint[4](-1)) q[0];", 15.0),
        ("rx(bit[2](7)) q[0];", 3.0),
        ("rx(bool(5)) q[0];", 1.0),
        ("rx(bool(0.0)) q[0];", 0.0),
        ("rx(float[32](0.1)) q[0];", f64::from(0.1f32)),
        ("rx(angle[3](pi / 3)) q[0];", std::f64::consts::FRAC_PI_4),
        ("array[int[8], 2] a = {3, 4}; rx(int(a[1])) q[0];", 4.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
}

#[test]
fn a_cast_width_is_checked() {
    for (body, wanted) in [
        ("rx(bool[2](1)) q[0];", "takes no width"),
        ("rx(int[0](1)) q[0];", "needs 1 to 64 bits"),
        ("rx(uint[65](1)) q[0];", "needs 1 to 64 bits"),
        ("rx(float[8](1)) q[0];", "needs 16, 32, 64 or 128 bits"),
        ("float[24] f = 1.0;", "needs 16, 32, 64 or 128 bits"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn floats_follow_ieee_754_and_the_language_casts() {
    for (body, want) in [
        ("float[16] f = 0.1; rx(f) q[0];", 0.0999755859375),
        ("rx(float[16](3.14159)) q[0];", 3.140625),
        ("rx(float[128](0.1)) q[0];", 0.1),
        ("rx(mod(7.5, 2)) q[0];", 1.5),
        ("rx(bit[2](uint(2.5))) q[0];", 2.0),
        ("rx(int[8](1e30)) q[0];", 0.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
    for (body, wanted) in [
        ("rx(7.5 % 2) q[0];", "`%` takes integers"),
        ("float f = 7.5; f %= 2; rx(f) q[0];", "`%` takes integers"),
        ("rx(bit[2](2.5)) q[0];", "cannot hold the float"),
    ] {
        let err = parse_error_of(body);
        assert!(err.to_string().contains(wanted), "{body}: {err}");
    }
}

#[test]
fn a_typed_variable_holds_what_its_type_can() {
    let tau = std::f64::consts::TAU;
    for (body, want) in [
        ("float[32] f = 0.1; rx(f) q[0];", f64::from(0.1f32)),
        ("angle[2] a = 1.0; rx(a) q[0];", tau / 4.0),
        ("angle a = -pi / 2; rx(a) q[0];", 3.0 * tau / 4.0),
        ("angle[1] a = 3 * pi / 2; rx(a) q[0];", 0.0),
        ("int[4] n = 7; n += 1; rx(n) q[0];", -8.0),
        ("angle[2] a = 0; a = angle(pi / 3); rx(a) q[0];", tau / 4.0),
        ("uint[3] n = 5; n *= 2; rx(n) q[0];", 2.0),
    ] {
        assert_eq!(rx_angle(body), want, "{body}");
    }
}

#[test]
fn every_compound_assignment_applies_its_operator() {
    for (update, want) in [
        ("**= 2", 36.0),
        ("&= 3", 2.0),
        ("|= 3", 7.0),
        ("^= 3", 5.0),
        ("<<= 2", 24.0),
        (">>= 2", 1.0),
    ] {
        let body = format!("int a = 6; a {update}; rx(a) q[0];");
        assert_eq!(rx_angle(&body), want, "{body}");
    }
}

#[test]
fn an_expression_statement_is_evaluated_and_discarded() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint a = 2;\na == 2;\n(a + 1) * 3;\n-a;\nx q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
    same_instructions(
        "qubit q; def f(qubit a) -> int { x a; return 3; } f(q) == 3; h q;",
        "qubit q; x q; h q;",
    );
    let err = parse(
        "OPENQASM 3.0;\nqubit[1] q;\ndef f(qubit a) -> bit { bit r = measure a; return r; }\n\
         f(q[0]) == 1;",
    )
    .unwrap_err();
    assert!(
        matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
            if construct.contains("def `f`")),
        "{err}"
    );
}

#[test]
fn complex_arithmetic_matches_the_specification_examples() {
    let declarations = "complex[float[64]] a = 10.0 + 5.0im;\n\
                        complex[float[64]] b = -2.0 - 7.0im;";
    for (value, expected) in [
        ("a + b", (8.0, -2.0)),
        ("a - b", (12.0, 12.0)),
        ("a * b", (15.0, -80.0)),
        ("a / b", (-55.0 / 53.0, 60.0 / 53.0)),
        ("a ** b", (0.10694695640729072, 0.17536481119721312)),
        ("-a", (-10.0, -5.0)),
        ("2 * a", (20.0, 10.0)),
    ] {
        let (re, im) = complex_parts(declarations, value);
        assert!(
            (re - expected.0).abs() < 1e-12 && (im - expected.1).abs() < 1e-12,
            "`{value}`: got {re}+{im}im, expected {expected:?}"
        );
    }
}

#[test]
fn imaginary_literals_take_every_spelling() {
    for (value, expected) in [
        ("1.5im", 1.5),
        ("2 im", 2.0),
        ("3\tim", 3.0),
        ("1e1im", 10.0),
        (".5im", 0.5),
    ] {
        assert_eq!(complex_parts("", value), (0.0, expected), "`{value}`");
    }
    let (re, im) = complex_parts("complex[float] d = 2.0+sin(pi/2) + (3.1 * 5.5 im);", "d");
    assert!((re - 3.0).abs() < 1e-12 && (im - 17.05).abs() < 1e-12);
    same_instructions(
        "OPENQASM 3.0;\nqubit im;\nh im;",
        "OPENQASM 3.0;\nqubit q;\nh q;",
    );
}

#[test]
fn complex_builtins_and_casts_fold() {
    for (expression, expected) in [
        ("abs(3.0 + 4.0im)", 5.0),
        ("arg(2im)", std::f64::consts::FRAC_PI_2),
        ("real(exp(pi * 1im))", -1.0),
        ("imag(sqrt(-4 + 0im))", 2.0),
        ("real(2.5)", 2.5),
        ("imag(2.5)", 0.0),
        ("arg(-1)", std::f64::consts::PI),
        ("real(complex[float[64]](3))", 3.0),
        ("imag(complex(1im) * 1im)", 0.0),
        ("real(1im ** 2)", -1.0),
    ] {
        let angles = rz_angles(&format!("OPENQASM 3.0;\nqubit q;\nrz({expression}) q;"));
        assert!(
            (angles[0] - expected).abs() < 1e-12,
            "`{expression}`: got {}, expected {expected}",
            angles[0]
        );
    }
}

#[test]
fn complex_components_round_to_the_declared_width() {
    let (re, im) = complex_parts("complex[float[32]] z = 0.1 + 0.2im;", "z");
    assert_eq!(re, f64::from(0.1f32));
    assert_eq!(im, f64::from(0.2f32));
}

#[test]
fn complex_assignment_and_compound_operators_update_the_value() {
    let declarations = "complex z;\nz = 1 + 1im;\nz *= 2im;\nz += 1;";
    assert_eq!(complex_parts(declarations, "z"), (-1.0, 2.0));
    same_instructions(
        "OPENQASM 3.0;\nqubit q;\ncomplex z = 1 + 2im;\n\
         if (z == 1 + 2im) x q;\nif (z != 1 + 2im) y q;",
        "OPENQASM 3.0;\nqubit q;\nx q;",
    );
}

#[test]
fn complex_values_cross_def_boundaries() {
    let source = "const complex[float[64]] base = 1im;\n\
                  def scaled(complex[float[64]] z, float k) -> complex[float[64]] {\n\
                      return z * k + base;\n\
                  }\n\
                  complex w = scaled(2 + 1im, 3);";
    assert_eq!(complex_parts(source, "w"), (6.0, 4.0));
}

#[test]
fn complex_values_in_a_real_context_reject_with_a_parse_error() {
    for (body, needle) in [
        ("rx(1im) q;", "complex value where a real one belongs"),
        (
            "complex z = 1im; float f = z;",
            "complex value where a real one belongs",
        ),
        ("complex z = 1im; int n = int(z);", "casts a complex value"),
        (
            "complex c1 = 1.0 + 2.0im; complex c2 = mod(c1, 2);",
            "complex value",
        ),
        ("complex[int] z;", "`float` components"),
        ("complex z = 1im / 0im;", "division by zero"),
        ("rx(sin(1im)) q;", "complex value"),
    ] {
        let source = format!("OPENQASM 3.0;\nqubit q;\n{body}");
        match parse(&source) {
            Err(PrismError::Parse { message, .. }) => {
                assert!(message.contains(needle), "`{body}`: `{message}`")
            }
            other => panic!("`{body}` gave {other:?}"),
        }
    }
}

#[test]
fn a_runtime_complex_value_declines_by_name() {
    let source = "OPENQASM 3.0;\nqubit q;\nbit c;\ncomplex z = 0;\n\
                  c = measure q;\nif (c) { z = 1im; }\nrz(real(z)) q;";
    match crate::circuit::openqasm::parse_dynamic(source) {
        Err(PrismError::UnsupportedConstruct { construct, .. }) => {
            assert!(construct.contains("complex"), "{construct}")
        }
        other => panic!("expected a decline, got {other:?}"),
    }
    let folded = "OPENQASM 3.0;\nqubit q;\ncomplex z = 2 + 1im;\nrz(abs(z)) q;";
    assert!(crate::circuit::openqasm::parse_dynamic(folded).is_ok());
}

#[test]
fn sizeof_reads_array_and_slice_dimensions() {
    let head = "OPENQASM 3.0;\nqubit q;\narray[int, 4, 3] a;\narray[float, 5] b;\n";
    for (expression, expected) in [
        ("sizeof(a)", 4.0),
        ("sizeof(a, 0)", 4.0),
        ("sizeof(a, 1)", 3.0),
        ("sizeof(a[0])", 3.0),
        ("sizeof(a[1:2], 0)", 2.0),
        ("sizeof(b[0:2:4])", 3.0),
        ("sizeof(b) + 1", 6.0),
    ] {
        let angles = rz_angles(&format!("{head}rz({expression}) q;"));
        assert_eq!(angles, vec![expected], "`{expression}`");
    }
    for (expression, needle) in [
        ("sizeof(a, 2)", "no dimension 2"),
        ("sizeof(q)", "declared array"),
        ("sizeof(a, 0, 1)", "optional dimension"),
    ] {
        match parse(&format!("{head}rz({expression}) q;")) {
            Err(PrismError::Parse { message, .. }) => {
                assert!(message.contains(needle), "`{expression}`: `{message}`")
            }
            other => panic!("`{expression}` gave {other:?}"),
        }
    }
}

#[test]
fn array_slices_copy_rows_ranges_and_joins() {
    let head = "array[int[8], 2] first = {0, 1};\narray[int[8], 3] second = {2, 3, 4};\n\
                array[int[8], 4] third = {5, 6, 7, 8};\n";
    assert_eq!(
        array_values(
            &format!("{head}array[int[8], 5] c = first ++ second;"),
            "c",
            5
        ),
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        array_values(&format!("{head}array[int[8], 2] s = second[1:2];"), "s", 2),
        vec![3.0, 4.0]
    );
    assert_eq!(
        array_values(&format!("{head}second[1:2] = first[0:1];"), "second", 3),
        vec![2.0, 0.0, 1.0]
    );
    assert_eq!(
        array_values(
            &format!(
                "{head}array[int[8], 4] j = first ++ first;\nj[0:3] = first[0:1] ++ third[1:2];"
            ),
            "j",
            4
        ),
        vec![0.0, 1.0, 6.0, 7.0]
    );
    let grid = "array[float[64], 3, 2] g = {{1, 2}, {3, 4}, {5, 6}};\n\
                array[float[64], 2] row = g[2];\n\
                array[float[64], 3, 2] h = g;\n\
                g[0] = row;\ng[1:2, 0] = h[0:1, 1];";
    assert_eq!(array_values(grid, "row", 2), vec![5.0, 6.0]);
    let flat = rz_angles(&format!(
        "OPENQASM 3.0;\nqubit q;\n{grid}\n\
         for int i in [0:2] {{ for int k in [0:1] {{ rz(g[i, k]) q; }} }}"
    ));
    assert_eq!(flat, vec![5.0, 6.0, 2.0, 4.0, 4.0, 6.0]);
}

#[test]
fn array_elements_wrap_to_the_declared_width() {
    assert_eq!(
        array_values("array[int[8], 2] a = {200, -129};", "a", 2),
        vec![-56.0, 127.0]
    );
    assert_eq!(
        array_values("array[uint[4], 1] a;\na[0] = 17;", "a", 1),
        vec![1.0]
    );
}

#[test]
fn array_shape_mismatches_reject() {
    let head = "OPENQASM 3.0;\narray[int[8], 3] aa;\narray[int[8], 4, 3] bb;\nint[8] scalar;\n";
    for body in [
        "bb[0] = 1;",
        "bb[0, 0] = aa;",
        "bb = aa;",
        "array[int[8], 2] x = aa;",
        "array[int[8], 2] x = aa ++ bb;",
        "aa[0:1] = aa;",
    ] {
        match parse(&format!("{head}{body}")) {
            Err(PrismError::Parse { .. }) => {}
            other => panic!("`{body}` gave {other:?}"),
        }
    }
    assert!(
        parse(&format!(
            "{head}bb[0] = aa;\nbb[0, 1] = aa[2];\nbb[1:2] = bb[2:3];"
        ))
        .is_ok()
    );
    match parse(&format!("{head}aa[0:1] += aa[1:2];")) {
        Err(PrismError::UnsupportedConstruct { construct, .. }) => {
            assert!(construct.contains("compound"), "{construct}")
        }
        other => panic!("compound slice assignment gave {other:?}"),
    }
}

#[test]
fn array_arguments_pass_by_reference() {
    let program = "OPENQASM 3.0;\nqubit[2] q;\n\
                   def fill(mutable array[int[8], #dim = 1] out, int v) {\n\
                       for int i in [0:sizeof(out) - 1] { out[i] = v + i; }\n\
                   }\n\
                   def total(readonly array[int[8], 2, 3] grid) -> int {\n\
                       int sum = 0;\n\
                       for int i in [0:sizeof(grid, 0) - 1] {\n\
                           for int k in [0:sizeof(grid, 1) - 1] { sum += grid[i][k]; }\n\
                       }\n\
                       return sum;\n\
                   }\n\
                   array[int[8], 5] aa;\n\
                   array[int[8], 2, 3] bb = {{1, 2, 3}, {4, 5, 6}};\n\
                   fill(aa[1:3], 10);\n\
                   fill(bb[1, 0:1], 7);\n\
                   rz(total(bb)) q[0];\n\
                   for int i in [0:4] { rz(aa[i]) q[1]; }";
    assert_eq!(
        rz_angles(program),
        vec![
            1.0 + 2.0 + 3.0 + 7.0 + 8.0 + 6.0,
            0.0,
            10.0,
            11.0,
            12.0,
            0.0
        ]
    );
    same_instructions(
        program,
        "OPENQASM 3.0;\nqubit[2] q;\nrz(27) q[0];\nrz(0) q[1];\nrz(10) q[1];\n\
         rz(11) q[1];\nrz(12) q[1];\nrz(0) q[1];",
    );
}

#[test]
fn array_arguments_check_mutability_shape_and_type() {
    let head = "OPENQASM 3.0;\narray[int[8], 4] aa;\nconst array[int[8], 2] cc = {1, 2};\n\
                array[float[64], 4] ff;\n\
                def reads(readonly array[int[8], 4] a) { a[0] = 1; }\n\
                def sized(readonly array[int[8], 4] a) { }\n\
                def ranked(readonly array[int[8], #dim = 2] a) { }\n\
                def writes(mutable array[int[8], #dim = 1] a) { a[0] = 1; }\n";
    for call in [
        "reads(aa);",
        "sized(aa[0:2]);",
        "ranked(aa);",
        "writes(cc);",
        "sized(ff);",
        "sized(aa[{0, 1, 2, 3}]);",
        "sized(3);",
    ] {
        match parse(&format!("{head}{call}")) {
            Err(PrismError::Parse { .. }) => {}
            other => panic!("`{call}` gave {other:?}"),
        }
    }
    assert!(parse(&format!("{head}sized(aa);\nwrites(aa[1:2]);")).is_ok());
}

#[test]
fn integer_bit_slices_read_as_unsigned_values() {
    let head = "OPENQASM 3.0;\nqubit q;\nuint[16] u = 10;\nint[32] m = 15;\n";
    for (expression, expected) in [
        ("u[0:3]", 10.0),
        ("u[1:3]", 5.0),
        ("m[0:2:31]", 3.0),
        ("m[3:-1:0]", 15.0),
    ] {
        let angles = rz_angles(&format!("{head}rz({expression}) q;"));
        assert_eq!(angles, vec![expected], "`{expression}`");
    }
}
