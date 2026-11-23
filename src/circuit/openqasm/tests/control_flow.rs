//! Branches, loops, switches and early exits.

use super::*;

#[test]
fn test_switch_lowers_to_a_guard_per_label() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) { case 0 { x q[0]; } }";
    let circuit = parse(qasm).expect("parse");
    assert_eq!(circuit.instructions.len(), 1);
    assert!(matches!(
        &circuit.instructions[0],
        Instruction::Conditional {
            condition: ClassicalCondition::RegisterEquals {
                value: 0,
                size: 2,
                ..
            },
            ..
        }
    ));
}

#[test]
fn test_switch_body_must_be_case_or_default() {
    let qasm = "OPENQASM 3.0;
qubit[1] q;
bit[2] c;
switch (c) { x q[0]; }";
    match parse(qasm).unwrap_err() {
        PrismError::Parse { message, line } => {
            assert!(message.contains("`case` or `default`"), "{message}");
            assert_eq!(line, 4);
        }
        other => panic!("expected a parse error, got {other:?}"),
    }
}

#[test]
fn test_if_oq2_register_equals() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        creg c[2];
        if(c==1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional {
        condition,
        gate,
        targets,
    } = &c.instructions[0]
    {
        assert_eq!(*gate, Gate::X);
        assert_eq!(targets.as_slice(), &[0]);
        assert!(matches!(
            condition,
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 2,
                value: 1
            }
        ));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn test_if_oq3_single_bit() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        if (c[0]) x q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional {
        condition,
        gate,
        targets,
    } = &c.instructions[0]
    {
        assert_eq!(*gate, Gate::X);
        assert_eq!(targets.as_slice(), &[1]);
        assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn test_if_with_parametric_gate() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nif (c[0]) rz(pi/4) q[0];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Conditional { gate, .. } = &c.instructions[0] {
        assert!(matches!(gate, Gate::Rz(_)));
    } else {
        panic!("expected Conditional");
    }
}

#[test]
fn for_loop_unrolls_inclusive_range() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [0:3] {
            h q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    for (idx, instr) in c.instructions.iter().enumerate() {
        match instr {
            Instruction::Gate {
                gate: Gate::H,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[idx]);
            }
            _ => panic!("expected H gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_with_step() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[6] q;
        for int i in [0:2:4] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    let expected = [0usize, 2, 4];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_set_form() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[5] q;
        for int i in {0, 2, 4} {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    let expected = [0usize, 2, 4];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_negative_step() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [3:-1:0] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    let expected = [3usize, 2, 1, 0];
    for (idx, &qi) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::X,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[qi]);
            }
            _ => panic!("expected X gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_uses_loop_variable_in_param_expression() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for int i in [0:2] {
            rx(i * pi / 4) q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    for (idx, instr) in c.instructions.iter().enumerate() {
        match instr {
            Instruction::Gate {
                gate: Gate::Rx(theta),
                targets,
            } => {
                let expected = idx as f64 * std::f64::consts::PI / 4.0;
                assert!((theta - expected).abs() < 1e-12);
                assert_eq!(targets.as_slice(), &[idx]);
            }
            _ => panic!("expected Rx gate at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_nested() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        for int i in [0:1] {
            for int j in [0:1] {
                cx q[i], q[j+2];
            }
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    let expected = [(0usize, 2usize), (0, 3), (1, 2), (1, 3)];
    for (idx, &(ci, ti)) in expected.iter().enumerate() {
        match &c.instructions[idx] {
            Instruction::Gate {
                gate: Gate::Cx,
                targets,
            } => {
                assert_eq!(targets.as_slice(), &[ci, ti]);
            }
            _ => panic!("expected CX at slot {idx}"),
        }
    }
}

#[test]
fn for_loop_uint_keyword_accepted() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        for uint i in [0:2] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
}

#[test]
fn for_loop_variable_holds_values_as_its_declaration_does() {
    let unrolled: String = (0..=5)
        .map(|v| format!("uint[2] j{v} = {v}; x q[j{v}];"))
        .collect();
    for (looped, declared) in [
        ("for uint[2] i in [0:5] { x q[i]; }", unrolled.as_str()),
        (
            "for uint[2] i in [3:3] { i = i + 1; x q[i]; }",
            "uint[2] j = 3; j = j + 1; x q[j];",
        ),
        (
            "for uint[2] i in [3:3] { x q[i + 1]; }",
            "uint[2] j = 3; x q[j + 1];",
        ),
        (
            "for int[8] i in {200} { if (i < 0) x q[0]; }",
            "int[8] j = 200; if (j < 0) x q[0];",
        ),
        (
            "for uint[8] i in [-1:-1] { rz(i) q[0]; }",
            "uint[8] j = -1; rz(j) q[0];",
        ),
    ] {
        let header = "OPENQASM 3.0;\nqubit[6] q;\n";
        let expected = parse(&format!("{header}{declared}")).unwrap();
        let source = format!("{header}{looped}");
        let circuit = parse(&source).unwrap();
        let program = parse_dynamic(&source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("static loop")] {
            assert_eq!(
                format!("{:?}", actual.instructions),
                format!("{:?}", expected.instructions),
                "`{looped}` against `{declared}`"
            );
        }
    }
}

#[test]
fn for_loop_body_may_be_one_statement() {
    for (unbraced, braced) in [
        (
            "for int i in [0:2] x q[i];",
            "for int i in [0:2] { x q[i]; }",
        ),
        (
            "for int i in [0:1] for int j in [0:1] cx q[i], q[j + 2];",
            "for int i in [0:1] { for int j in [0:1] { cx q[i], q[j + 2]; } }",
        ),
    ] {
        let header = "OPENQASM 3.0;\nqubit[4] q;\n";
        let expected = parse(&format!("{header}{braced}")).unwrap();
        let source = format!("{header}{unbraced} h q[0];");
        let circuit = parse(&source).unwrap();
        let program = parse_dynamic(&source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("static loop")] {
            let (body, tail) = actual.instructions.split_at(expected.instructions.len());
            assert_eq!(
                format!("{body:?}"),
                format!("{:?}", expected.instructions),
                "`{unbraced}` against `{braced}`"
            );
            assert!(matches!(tail, [Instruction::Gate { gate: Gate::H, .. }]));
        }
    }
}

#[test]
fn if_singleton_bit_selections_keep_guard_lowering() {
    for (condition, reference) in [
        ("c[0:0]", "c[0]"),
        ("c[{0}]", "c[0]"),
        ("!c[0:0]", "!c[0]"),
        ("!c[{0}]", "!c[0]"),
        ("c[0:0] == 1", "c[0] == 1"),
        ("c[{0}] != 0", "c[0] != 0"),
    ] {
        let source = format!("qubit q; bit[2] c; if ({condition}) x q;");
        let expected = parse(&format!("qubit q; bit[2] c; if ({reference}) x q;")).unwrap();
        let circuit = parse(&source).unwrap();
        let program = parse_dynamic(&source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("bit guard")] {
            assert_eq!(
                format!("{:?}", actual.instructions),
                format!("{:?}", expected.instructions),
                "{condition}"
            );
        }
    }
}

#[test]
fn if_bit_equals_one_form_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional {
            condition,
            gate: Gate::X,
            targets,
        } => {
            assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected conditional X gate"),
    }
}

#[test]
fn if_bit_equals_zero_form_parses() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] == 0) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_bit_not_equal_one_negates() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] != 1) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_bit_not_equal_zero_is_one() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (c[0] != 0) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsOne(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_negated_bit_truthy_check() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        if (!c[0]) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(condition, ClassicalCondition::BitIsZero(0)));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_not_equals() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[3] c;
        if (c != 5) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterNotEquals { value: 5, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_equals_hex_literal() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[8] c;
        if (c == 0xff) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 255, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn if_register_equals_binary_literal() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        if (c == 0b1010) x q[0];
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 10, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn for_loop_accepts_hex_bound() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[8] q;
        for int i in [0:0x3] {
            x q[i];
        }
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
}

#[test]
fn an_if_on_a_parse_time_value_keeps_the_arm_it_takes() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint n = 3;\nif (n > 2 && n != 5) x q[0]; else y q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nif (1 == 2) { x q[0]; } else { y q[0]; }",
        "OPENQASM 3.0;\nqubit[1] q;\ny q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nif (false) x q[0];\nz q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nz q[0];",
    );
}

#[test]
fn a_condition_over_classical_bits_lowers_to_its_guard() {
    for (body, want) in [
        ("if (!c[0]) x q[0];", ClassicalCondition::BitIsZero(0)),
        ("if (c[2] == 1) x q[0];", ClassicalCondition::BitIsOne(2)),
        (
            "if (bool(c)) x q[0];",
            ClassicalCondition::RegisterNotEquals {
                offset: 0,
                size: 4,
                value: 0,
            },
        ),
        (
            "if (c == 1 + 2) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 3,
            },
        ),
        (
            "if (uint[2](c) != 3) x q[0];",
            ClassicalCondition::RegisterNotEquals {
                offset: 0,
                size: 2,
                value: 3,
            },
        ),
        (
            "if (int[8](c) == 5) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 5,
            },
        ),
        (
            "if (int[4](c) == -1) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 15,
            },
        ),
        (
            "if (int[3](c) == -4) x q[0];",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 3,
                value: 4,
            },
        ),
    ] {
        assert_eq!(
            format!("{:?}", guard_of(body)),
            format!("{want:?}"),
            "{body}"
        );
    }
}

#[test]
fn a_condition_no_guard_spells_points_at_the_dynamic_parse() {
    for body in [
        "if (c[0] && c[1]) x q[0];",
        "if (int[4](c) == 8) x q[0];",
        "if (c + 1 == 2) x q[0];",
    ] {
        let err = parse_error_of(body);
        assert!(
            matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("parse_dynamic")),
            "{body}: {err}"
        );
        let source = format!("OPENQASM 3.0;\nqubit[2] q;\nbit[4] c;\n{body}");
        parse_dynamic(&source).unwrap_or_else(|e| panic!("{body}: {e}"));
    }
}

#[test]
fn an_integer_reads_one_bit_per_index() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[4] q;\nuint[4] a = 5;\n\
         for int i in [0:3] { if (bool(a[i])) x q[i]; }",
        "OPENQASM 3.0;\nqubit[4] q;\nx q[0];\nx q[2];",
    );
    assert_eq!(rx_angle("uint[4] a = 6; rx(a[1]) q[0];"), 1.0);
    assert_eq!(rx_angle("int[4] a = -2; rx(a[-1] + 2 * a[0]) q[0];"), 1.0);
    let err = parse_error_of("uint[4] a = 5; rx(a[4]) q[0];");
    assert!(err.to_string().contains("outside `a`"), "{err}");
}

#[test]
fn a_switch_takes_any_value_its_guards_can_compare() {
    for (body, want) in [
        (
            "switch (int(c)) { case 1 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 4,
                value: 1,
            },
        ),
        (
            "switch (uint[2](c)) { case 3 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 0,
                size: 2,
                value: 3,
            },
        ),
        (
            "switch (c[1]) { case 1 { x q[0]; } }",
            ClassicalCondition::RegisterEquals {
                offset: 1,
                size: 1,
                value: 1,
            },
        ),
    ] {
        assert_eq!(
            format!("{:?}", guard_of(body)),
            format!("{want:?}"),
            "{body}"
        );
    }
    let err = parse_error_of("switch (int[2](c)) { case 1 { x q[0]; } }");
    assert!(err.to_string().contains("parse_dynamic"), "{err}");
}

#[test]
fn a_switch_on_a_parse_time_value_keeps_one_arm() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nint i = 2;\n\
         switch (i * 2) { case 3 { x q[0]; } case 4, 5 { y q[0]; } }",
        "OPENQASM 3.0;\nqubit[1] q;\ny q[0];",
    );
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\nswitch (5) { case 3 { x q[0]; } default { z q[0]; } }",
        "OPENQASM 3.0;\nqubit[1] q;\nz q[0];",
    );
    let err =
        parse("OPENQASM 3.0;\nqubit[1] q;\nswitch (1) { case 1 { } case 1 { } }").unwrap_err();
    assert!(err.to_string().contains("appears twice"), "{err}");
}

#[test]
fn break_and_continue_stop_an_unrolled_for() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\n";
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i == 2) break; x q[i]; }} h q[3];"),
        &format!("{head}x q[0]; x q[1]; h q[3];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i == 1) {{ continue; }} h q[i]; }}"),
        &format!("{head}h q[0]; h q[2]; h q[3];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ if (i % 2 == 1) continue; c[i] = measure q[i]; }}"),
        &format!("{head}c[0] = measure q[0]; c[2] = measure q[2];"),
    );
    same_instructions(
        &format!(
            "{head}for int i in [0:1] {{ for int j in [0:3] {{ if (j == 2) break; cx q[i], q[j + 2]; }} z q[i]; }}"
        ),
        &format!(
            "{head}cx q[0], q[2]; cx q[0], q[3]; z q[0]; cx q[1], q[2]; cx q[1], q[3]; z q[1];"
        ),
    );
    // A `switch` is not a loop, so its `break` leaves the enclosing `for`.
    same_instructions(
        &format!(
            "{head}for int i in [0:3] {{ switch (i) {{ case 1 {{ break; }} default {{ x q[i]; }} }} }}"
        ),
        &format!("{head}x q[0];"),
    );
    same_instructions(
        &format!(
            "{head}int n = 0; for int i in [0:3] {{ n += 1; if (n == 2) break; }} rx(n) q[0];"
        ),
        &format!("{head}rx(2) q[0];"),
    );
}

#[test]
fn end_stops_the_program_from_any_scope() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\n";
    same_instructions(
        &format!("{head}x q[0];\nend;\nx q[1];"),
        &format!("{head}x q[0];"),
    );
    same_instructions(
        &format!("{head}for int i in [0:3] {{ x q[i]; if (i == 1) end; }} h q[0];"),
        &format!("{head}x q[0]; x q[1];"),
    );
    same_instructions(
        &format!("{head}def f(qubit a) {{ x a; end; }} f(q[2]); h q[0];"),
        &format!("{head}x q[2];"),
    );
}

#[test]
fn loop_exits_outside_a_loop_are_parse_errors() {
    let head = "OPENQASM 3.0;\nqubit[2] q;\n";
    for body in [
        "break;",
        "if (true) { continue; }",
        "def f() { continue; } for int i in [0:1] { f(); }",
        "gate g a { break; } g q[0];",
    ] {
        match parse(&format!("{head}{body}")) {
            Err(PrismError::Parse { message, .. }) => {
                assert!(message.contains("outside a loop"), "`{body}`: {message}")
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
    match parse(&format!("{head}gate g a {{ end; }} g q[0];")) {
        Err(PrismError::Parse { message, .. }) => {
            assert!(message.contains("gate `g`"), "{message}")
        }
        other => panic!("`end` in a gate should be a parse error, got {other:?}"),
    }
}

#[test]
fn a_measured_exit_under_parse_points_at_parse_dynamic() {
    let head = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nc[0] = measure q[0];\n";
    for body in [
        "for int i in [0:1] { if (c[0]) break; x q[1]; }",
        "for int i in [0:1] { if (c[0]) { continue; } x q[1]; }",
        "if (c[0]) end;",
        "for int i in [0:1] { if (c[0]) end; }",
    ] {
        match parse(&format!("{head}{body}")) {
            Err(PrismError::UnsupportedConstruct { construct, line }) => {
                assert!(construct.contains("parse_dynamic"), "`{body}`: {construct}");
                assert_eq!(line, 5, "`{body}`");
            }
            other => panic!("`{body}` should decline, got {other:?}"),
        }
    }
}

#[test]
fn end_is_still_a_gate_name_when_it_takes_operands() {
    same_instructions(
        "OPENQASM 3.0;\nqubit[1] q;\ngate end a { x a; }\nend q[0];",
        "OPENQASM 3.0;\nqubit[1] q;\nx q[0];",
    );
}

#[test]
fn end_in_a_def_call_stops_the_rest_of_its_statement() {
    let head = "OPENQASM 3.0;\nqubit[4] q;\nbit[4] c;\n\
                def f(qubit a) -> int { x a; end; return 1; }\n\
                def g(qubit a) -> int { h a; return 2; }\n\
                def k(int a, int b, qubit w) -> int { z w; return a; }\n";
    for body in [
        "if (f(q[0]) == 0) { c[1] = measure q[1]; }",
        "if (f(q[0]) == 5) { } else { y q[1]; }",
        "int n = f(q[0]) + g(q[3]);",
        "int n = 0; n = f(q[0]) + g(q[3]);",
        "bool b = f(q[0]) == 0 && g(q[3]) == 2;",
        "rx(f(q[0])) q[1];",
        "int m = k(f(q[0]), g(q[3]), q[2]);",
        "def r(qubit u, qubit v) -> int { return f(u) + g(v); } int n = r(q[0], q[3]);",
        "if (true) { int n = f(q[0]) + g(q[3]); } y q[1];",
    ] {
        same_instructions(&format!("{head}{body} y q[2];"), &format!("{head}x q[0];"));
    }
    same_instructions(
        "OPENQASM 3.0;\nqubit[4] q;\nbit[1] o;\ndef f() -> int { end; return 1; }\n\
         x q[3];\nif (f() == 0) { o[0] = measure q[3]; }",
        "OPENQASM 3.0;\nqubit[4] q;\nx q[3];",
    );
}
