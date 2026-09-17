use super::super::lexer::tokenize;
use super::*;

fn parse(source: &str) -> Block<'_> {
    let tokens = tokenize(source).unwrap_or_else(|e| panic!("`{source}`: {e}"));
    parse_program(&tokens).unwrap_or_else(|e| panic!("`{source}`: {e}"))
}

fn error(source: &str) -> PrismError {
    let tokens = match tokenize(source) {
        Ok(tokens) => tokens,
        Err(e) => return e,
    };
    parse_program(&tokens)
        .err()
        .unwrap_or_else(|| panic!("`{source}` should not parse"))
}

fn one(source: &str) -> StmtKind<'_> {
    let mut block = parse(source);
    assert_eq!(block.len(), 1, "`{source}` gave {} statements", block.len());
    block.pop().expect("one statement").kind
}

#[test]
fn declarations_keep_their_size_and_name() {
    assert!(matches!(
        one("qubit[4] q;"),
        StmtKind::RegisterDecl {
            kind: RegisterKind::Qubit,
            size: Some(_),
            ..
        }
    ));
    assert!(matches!(
        one("bit c;"),
        StmtKind::RegisterDecl {
            kind: RegisterKind::Classical,
            size: None,
            ..
        }
    ));
    // The legacy spelling writes the size after the name and means the same.
    assert!(matches!(
        one("qreg q[4];"),
        StmtKind::RegisterDecl {
            kind: RegisterKind::Qubit,
            size: Some(_),
            ..
        }
    ));
    assert!(matches!(
        one("input float[64] theta;"),
        StmtKind::InputDecl { .. }
    ));
    assert!(matches!(
        one("output bit[2] c;"),
        StmtKind::OutputDecl { .. }
    ));
    assert!(matches!(
        one("const int n = 3;"),
        StmtKind::ClassicalDecl {
            constant: true,
            value: Some(_),
            ..
        }
    ));
    assert!(matches!(
        one("float x;"),
        StmtKind::ClassicalDecl {
            constant: false,
            value: None,
            ..
        }
    ));
}

// A name followed by `=` is an assignment and a name followed by operands is a
// gate. Only what comes after the name separates them.
#[test]
fn a_name_is_read_by_what_follows_it() {
    assert!(matches!(one("n = 1;"), StmtKind::Assign { op: None, .. }));
    assert!(matches!(
        one("n += 1;"),
        StmtKind::Assign {
            op: Some(AssignOp::Add),
            ..
        }
    ));
    assert!(matches!(one("c[0] = measure q[0];"), StmtKind::Measure(_)));
    assert!(matches!(one("measure q -> c;"), StmtKind::Measure(_)));
    match one("h q[0];") {
        StmtKind::Call { name, operands, .. } => {
            assert_eq!(name, "h");
            assert_eq!(operands.len(), 1);
        }
        other => panic!("{other:?}"),
    }
    match one("rx(pi / 2) q[0];") {
        StmtKind::Call { params, .. } => {
            assert_eq!(params.len(), 1);
            assert!(matches!(params[0], Argument::Value(_)));
        }
        other => panic!("{other:?}"),
    }
    // A subscript settles an argument as a qubit; a bare name stays a value
    // until the declaration says otherwise.
    match one("sub(q[0], 0.5);") {
        StmtKind::Call {
            params, operands, ..
        } => {
            assert!(operands.is_empty());
            assert!(matches!(params[0], Argument::Operand(_)));
            assert!(matches!(params[1], Argument::Value(_)));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_modifier_chain_reads_in_order() {
    match one("ctrl @ negctrl @ inv @ pow(2) @ x q[0], q[1], q[2];") {
        StmtKind::Call {
            modifiers, name, ..
        } => {
            assert_eq!(name, "x");
            assert!(matches!(modifiers[0], Modifier::Ctrl { negated: false }));
            assert!(matches!(modifiers[1], Modifier::Ctrl { negated: true }));
            assert!(matches!(modifiers[2], Modifier::Inv));
            assert!(matches!(modifiers[3], Modifier::Pow(_)));
        }
        other => panic!("{other:?}"),
    }
    // No space is required around `@`, which used to read as part of the name.
    assert!(matches!(one("ctrl@x q[0], q[1];"), StmtKind::Call { .. }));
    assert!(matches!(
        error("frobnicate @ x q[0];"),
        PrismError::UnsupportedConstruct { .. }
    ));
}

#[test]
fn a_subscript_carries_every_shape_it_was_written_in() {
    let shapes = [
        ("h q[0];", "single"),
        ("h q[0:2];", "range"),
        ("h q[0:2:6];", "stepped"),
        ("h q[:2];", "open start"),
        ("h q[2:];", "open stop"),
        ("h q[{0, 3}];", "set"),
    ];
    for (source, label) in shapes {
        match one(source) {
            StmtKind::Call { operands, .. } => {
                let index = operands[0].index.as_ref().expect(label);
                let matched = match (label, index) {
                    ("single", Index::Single(_)) | ("set", Index::Set(_)) => true,
                    ("range", Index::Range(range)) => {
                        range.step.is_none() && range.start.is_some() && range.stop.is_some()
                    }
                    ("stepped", Index::Range(range)) => range.step.is_some(),
                    ("open start", Index::Range(range)) => range.start.is_none(),
                    ("open stop", Index::Range(range)) => range.stop.is_none(),
                    _ => false,
                };
                assert!(matched, "{label}: {index:?}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn blocks_nest_without_counting_braces() {
    match one("if (c[0]) { x q[0]; if (c[1]) { y q[1]; } } else { z q[2]; }") {
        StmtKind::If(conditional) => {
            assert_eq!(conditional.then_body.len(), 2);
            assert_eq!(conditional.else_body.expect("else arm").len(), 1);
        }
        other => panic!("{other:?}"),
    }
    // An unbraced arm is one statement, and an `else if` nests under it.
    match one("if (c[0]) x q[0]; else if (c[1]) y q[1];") {
        StmtKind::If(conditional) => assert!(conditional.else_body.is_some()),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_statement_may_span_lines_anywhere() {
    let block = parse("h\n  q[0]\n  ;\nrx(\n  pi\n  / 2\n) q[1];");
    assert_eq!(block.len(), 2);
    assert_eq!(block[0].line, 1);
    assert_eq!(block[1].line, 4);
}

#[test]
fn loops_and_switches_keep_their_parts() {
    match one("for int k in [0:2:6] { h q[k]; }") {
        StmtKind::For {
            variable,
            range: ForRange::Range { step: Some(_), .. },
            body,
        } => {
            assert_eq!(variable, "k");
            assert_eq!(body.len(), 1);
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        one("for k in {1, 3} { h q[k]; }"),
        StmtKind::For {
            range: ForRange::Set(_),
            ..
        }
    ));
    match one("switch (c) { case 0, 1 { x q[0]; } default { y q[0]; } }") {
        StmtKind::Switch { arms, .. } => {
            assert_eq!(arms.len(), 2);
            assert_eq!(arms[0].labels.as_ref().expect("labels").len(), 2);
            assert!(arms[1].labels.is_none());
        }
        other => panic!("{other:?}"),
    }
    assert!(format!("{}", error("switch (c) { nope { } }")).contains("`case` or `default`"));
}

#[test]
fn definitions_keep_their_signatures() {
    match one("gate rzx(t) a, b { h b; cx a, b; rz(t) b; cx a, b; h b; }") {
        StmtKind::GateDef {
            name,
            params,
            qubits,
            body,
        } => {
            assert_eq!(name, "rzx");
            assert_eq!(params, vec!["t".to_string()]);
            assert_eq!(qubits.len(), 2);
            assert_eq!(body.len(), 5);
        }
        other => panic!("{other:?}"),
    }
    match one("def sub(qubit a, float t) { rx(t) a; }") {
        StmtKind::DefDef { args, .. } => {
            assert!(matches!(args[0], DefParam::Qubit(_)));
            assert!(matches!(
                args[1],
                DefParam::Value {
                    integral: false,
                    ..
                }
            ));
        }
        other => panic!("{other:?}"),
    }
    assert!(matches!(
        error("def sub(qubit a) -> bit { h a; }"),
        PrismError::UnsupportedConstruct { .. }
    ));
    assert!(matches!(
        error("def sub(bit c) { }"),
        PrismError::UnsupportedConstruct { .. }
    ));
}

#[test]
fn unimplemented_keywords_are_rejected_by_name() {
    for keyword in ["defcal", "extern", "opaque", "while", "return", "break"] {
        let source = format!("{keyword} x;");
        match error(&source) {
            PrismError::UnsupportedConstruct { construct, .. } => {
                assert_eq!(construct, keyword);
            }
            other => panic!("`{source}` gave {other:?}"),
        }
    }
}

#[test]
fn timing_and_array_declines_name_the_construct() {
    for source in [
        "delay[10ns] q[0];",
        "delay q[0];",
        "duration d = 10ns;",
        "array[int[32], 2] a;",
    ] {
        match error(source) {
            PrismError::UnsupportedConstruct { construct, .. } => {
                assert!(
                    construct.contains("delay")
                        || construct.contains("duration")
                        || construct.contains("array"),
                    "`{source}` gave `{construct}`"
                );
            }
            other => panic!("`{source}` gave {other:?}"),
        }
    }
}

#[test]
fn an_unclosed_block_names_what_was_missing() {
    let message = format!("{}", error("gate g a { h a;"));
    assert!(message.contains('}'), "{message}");
}

#[test]
fn a_pragma_survives_as_its_own_statement() {
    match one("#pragma braket result state_vector") {
        StmtKind::Pragma(text) => assert_eq!(text, "#pragma braket result state_vector"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn conditions_take_every_shape_the_language_allows() {
    for source in [
        "if (c == 1) x q[0];",
        "if (c != 0) x q[0];",
        "if (c[0]) x q[0];",
        "if (!c[0]) x q[0];",
        "if (c[0] == 1) x q[0];",
        "if (c[0] ^ c[2]) x q[0];",
        "if ((c[0] ^ c[2]) == 0) x q[0];",
        "if ((c[0])) x q[0];",
    ] {
        assert!(matches!(one(source), StmtKind::If(_)), "`{source}`");
    }
    match one("if ((c[0] ^ c[2]) == 0) x q[0];") {
        StmtKind::If(conditional) => match conditional.condition {
            Condition::Parity {
                bits,
                compare: Some(_),
            } => assert_eq!(bits.len(), 2),
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    }
}
