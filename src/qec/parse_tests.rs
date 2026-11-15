use super::*;

fn parse_err(text: &str) -> (usize, String) {
    match parse_qec_program(text) {
        Err(PrismError::Parse { line, message }) => (line, message),
        Err(other) => panic!("{text:?}: expected a parse error, got {other}"),
        Ok(_) => panic!("{text:?}: expected a parse error"),
    }
}

fn stim_err(text: &str) -> String {
    match parse_stim_circuit(text) {
        Err(PrismError::Parse { message, .. }) => message,
        Err(other) => panic!("{text:?}: expected a parse error, got {other}"),
        Ok(_) => panic!("{text:?}: expected a parse error"),
    }
}

fn detector_records(program: &QecProgram) -> Vec<Vec<QecRecordRef>> {
    program
        .ops()
        .iter()
        .filter_map(|op| match op {
            QecOp::Detector { records, .. } => Some(records.clone()),
            _ => None,
        })
        .collect()
}

fn detector_coords(program: &QecProgram) -> Vec<Vec<f64>> {
    program
        .ops()
        .iter()
        .filter_map(|op| match op {
            QecOp::Detector { coords, .. } => Some(coords.clone()),
            _ => None,
        })
        .collect()
}

fn abs(indices: &[usize]) -> Vec<QecRecordRef> {
    indices.iter().map(|&i| QecRecordRef::Absolute(i)).collect()
}

#[test]
fn record_refs_resolve_against_the_records_before_the_op() {
    assert_eq!(
        parse_record_ref("rec[-1]", 3, 1).unwrap(),
        QecRecordRef::Absolute(2)
    );
    assert_eq!(
        parse_record_ref("rec[-3]", 3, 1).unwrap(),
        QecRecordRef::Absolute(0)
    );
    for (target, expected) in [
        ("rec[-4]", "out of bounds for 3 records"),
        ("rec[0]", "must be negative"),
        ("rec[1]", "must be negative"),
        ("rec[-x]", "invalid offset"),
        ("rec[-1", "expected measurement record target"),
        ("!rec[-1]", "inverted target"),
    ] {
        let err = parse_record_ref(target, 3, 7).unwrap_err();
        let PrismError::Parse { line, message } = err else {
            panic!("{target}: not a parse error");
        };
        assert_eq!(line, 7, "{target}");
        assert!(message.contains(expected), "{target}: {message}");
    }
}

#[test]
fn split_instruction_uppercases_the_name_and_keeps_args_and_targets() {
    let (name, args, targets) = split_instruction("x_error(0.1) 0 1", 1).unwrap();
    assert_eq!((name.as_str(), args.as_deref()), ("X_ERROR", Some("0.1")));
    assert_eq!(targets, ["0", "1"]);

    let (name, args, targets) = split_instruction("DETECTOR( 1, 2 ) rec[-1]", 1).unwrap();
    assert_eq!((name.as_str(), args.as_deref()), ("DETECTOR", Some("1, 2")));
    assert_eq!(targets, ["rec[-1]"]);

    // A parenthesis after the first space belongs to the targets, not the arguments.
    let (name, args, targets) = split_instruction("M 0 (1)", 1).unwrap();
    assert_eq!((name.as_str(), args), ("M", None));
    assert_eq!(targets, ["0", "(1)"]);

    let (name, args, targets) = split_instruction("TICK", 1).unwrap();
    assert_eq!((name.as_str(), args, targets.len()), ("TICK", None, 0));

    let err = split_instruction("M(0.1 0", 4).unwrap_err();
    assert!(matches!(err, PrismError::Parse { line: 4, .. }), "{err}");
}

#[test]
fn instruction_tags_are_dropped_from_the_name() {
    assert_eq!(
        strip_instruction_tag("REPEAT[rounds] 2 {", 1).unwrap(),
        "REPEAT 2 {"
    );
    assert_eq!(
        strip_instruction_tag("M[tag](0.1) 0", 1).unwrap(),
        "M(0.1) 0"
    );
    assert_eq!(
        strip_instruction_tag("DETECTOR rec[-1]", 1).unwrap(),
        "DETECTOR rec[-1]"
    );
    assert_eq!(strip_instruction_tag("[x] 0", 1).unwrap(), "[x] 0");
    let err = strip_instruction_tag("M[tag 0", 5).unwrap_err();
    assert!(
        err.to_string().contains("unterminated instruction tag"),
        "{err}"
    );
}

#[test]
fn instruction_lines_keep_source_line_numbers() {
    let lines = instruction_lines("\n# comment\nH 0 # trailing\n\n  M 0  \n").unwrap();
    assert_eq!(lines, vec![(3, "H 0".to_string()), (5, "M 0".to_string())]);
    assert_eq!(parse_err("H 0\n\n# c\nBOGUS 1").0, 4);
}

#[test]
fn repeat_structure_errors_name_their_line() {
    for (text, line, expected) in [
        ("M 0\n}", 2, "unmatched `}`"),
        ("M 0\nREPEAT 2 {\nM 0", 3, "unterminated `REPEAT` block"),
        ("M 0 {", 1, "unexpected `{`"),
        ("REPEAT 2\nM 0\n}", 1, "must have form"),
        ("REPEAT -1 {\nM 0\n}", 1, "unsigned integer"),
        ("REPEAT 2 x\nM 0\n}", 1, "must have form"),
    ] {
        let (got_line, message) = parse_err(text);
        assert_eq!(got_line, line, "{text:?}: {message}");
        assert!(message.contains(expected), "{text:?}: {message}");
    }
    assert!(is_repeat_header("repeat 2 {"));
    assert!(!is_repeat_header("REPEATX 2 {"));
}

#[test]
fn repeat_expansion_cap_is_inclusive() {
    let at_cap = format!("REPEAT {MAX_QEC_EXPANDED_LINES} {{\nTICK\n}}");
    check_repeats(&instruction_lines(&at_cap).unwrap()).unwrap();

    let over = format!("{at_cap}\nTICK");
    let err = check_repeats(&instruction_lines(&over).unwrap()).unwrap_err();
    assert!(err.to_string().contains("exceeds"), "{err}");

    let nested = format!(
        "REPEAT 2 {{\nREPEAT {} {{\nTICK\nTICK\n}}\n}}",
        MAX_QEC_EXPANDED_LINES / 4 + 1
    );
    let err = check_repeats(&instruction_lines(&nested).unwrap()).unwrap_err();
    assert!(err.to_string().contains("exceeds"), "{err}");
}

#[test]
fn repeat_copies_shift_record_refs_by_the_records_each_copy_adds() {
    let program =
        parse_qec_program("M 0\nREPEAT 3 {\nM 0 1\nDETECTOR rec[-1] rec[-3]\n}\nDETECTOR rec[-7]")
            .unwrap();
    assert_eq!(program.num_measurements(), 7);
    assert_eq!(
        detector_records(&program),
        vec![abs(&[2, 0]), abs(&[4, 2]), abs(&[6, 4]), abs(&[0])]
    );
}

#[test]
fn a_zero_count_repeat_parses_nothing_inside_it() {
    let program = parse_qec_program("REPEAT 0 {\nREPEAT 2 {\nBOGUS\n}\n}\nM 0").unwrap();
    assert_eq!(program.ops().len(), 1);
}

#[test]
fn shift_records_moves_absolute_refs_only() {
    let mut op = QecOp::Feedforward {
        records: vec![QecRecordRef::Absolute(1), QecRecordRef::Lookback(2)],
        expected: true,
        body: vec![QecOp::Detector {
            records: vec![QecRecordRef::Absolute(0)],
            coords: vec![],
        }],
    };
    shift_records(&mut op, 3);
    let QecOp::Feedforward { records, body, .. } = op else {
        unreachable!();
    };
    assert_eq!(
        records,
        vec![QecRecordRef::Absolute(4), QecRecordRef::Lookback(2)]
    );
    assert_eq!(
        body,
        vec![QecOp::Detector {
            records: vec![QecRecordRef::Absolute(3)],
            coords: vec![],
        }]
    );
}

#[test]
fn shift_coords_offsets_stim_detectors_and_is_ignored_in_native_text() {
    let text = "M 0\nREPEAT 3 {\nSHIFT_COORDS(1, 10)\nDETECTOR(0, 0, 5) rec[-1]\n}";
    let stim = parse_stim_circuit(text).unwrap();
    assert_eq!(
        detector_coords(&stim),
        vec![
            vec![1.0, 10.0, 5.0],
            vec![2.0, 20.0, 5.0],
            vec![3.0, 30.0, 5.0]
        ]
    );
    let native = parse_qec_program(text).unwrap();
    assert_eq!(detector_coords(&native), vec![vec![0.0, 0.0, 5.0]; 3]);
}

#[test]
fn measurement_basis_reads_the_last_letter() {
    for name in ["M", "MZ", "MR", "MRZ", "R", "RZ"] {
        assert_eq!(measurement_basis(name), QecBasis::Z, "{name}");
    }
    for name in ["MX", "MRX", "RX"] {
        assert_eq!(measurement_basis(name), QecBasis::X, "{name}");
    }
    for name in ["MY", "MRY", "RY"] {
        assert_eq!(measurement_basis(name), QecBasis::Y, "{name}");
    }
}

#[test]
fn native_measurement_error_is_a_flip_that_anticommutes_with_the_basis() {
    for (text, channel, basis) in [
        ("M(0.1) 0", QecNoise::XError(0.1), QecBasis::Z),
        ("MX(0.1) 0", QecNoise::ZError(0.1), QecBasis::X),
        ("MY(0.1) 0", QecNoise::ZError(0.1), QecBasis::Y),
    ] {
        let program = parse_qec_program(text).unwrap();
        assert_eq!(
            program.ops(),
            [
                QecOp::Noise {
                    channel,
                    targets: vec![0],
                },
                QecOp::Measure { basis, qubit: 0 },
            ],
            "{text}"
        );
    }
    let program = parse_qec_program("M(0) 0").unwrap();
    assert_eq!(
        program.ops(),
        [QecOp::Measure {
            basis: QecBasis::Z,
            qubit: 0
        }]
    );
}

#[test]
fn stim_measurement_error_flips_the_record_after_the_measurement() {
    let program = parse_stim_circuit("M(0.1) 0").unwrap();
    assert_eq!(
        program.ops(),
        [
            QecOp::Measure {
                basis: QecBasis::Z,
                qubit: 0,
            },
            QecOp::RecordFlip { probability: 0.1 },
        ]
    );
}

#[test]
fn stim_inverted_measurement_flips_before_and_restores_on_next_use() {
    let x = |qubit| QecOp::Gate {
        gate: Gate::X,
        targets: vec![qubit],
    };
    let mz = |qubit| QecOp::Measure {
        basis: QecBasis::Z,
        qubit,
    };
    assert_eq!(parse_stim_circuit("M !0").unwrap().ops(), [x(0), mz(0)]);
    assert_eq!(
        parse_stim_circuit("M !0\nM 0").unwrap().ops(),
        [x(0), mz(0), x(0), mz(0)]
    );
    // A reset discards the pending restore, and an X-basis result flips with Z.
    assert_eq!(
        parse_stim_circuit("MX !0\nR 0\nM 0").unwrap().ops(),
        [
            QecOp::Gate {
                gate: Gate::Z,
                targets: vec![0],
            },
            QecOp::Measure {
                basis: QecBasis::X,
                qubit: 0,
            },
            QecOp::Reset {
                basis: QecBasis::Z,
                qubit: 0,
            },
            mz(0),
        ]
    );
    let (_, message) = parse_err("M !0");
    assert!(message.contains("inverted target"), "{message}");
}

#[test]
fn probability_arguments_must_lie_in_the_unit_interval() {
    for text in [
        "X_ERROR(1.5) 0",
        "X_ERROR(-0.1) 0",
        "X_ERROR(nan) 0",
        "X_ERROR(inf) 0",
    ] {
        assert!(parse_err(text).1.contains("finite and in [0, 1]"), "{text}");
    }
    for text in ["X_ERROR 0", "X_ERROR(0.1, 0.2) 0"] {
        assert!(
            parse_err(text).1.contains("requires one numeric argument"),
            "{text}"
        );
    }
    assert!(
        parse_err("X_ERROR(abc) 0")
            .1
            .contains("expected numeric argument")
    );
    assert!(
        parse_err("PAULI_CHANNEL_1(0.1, 0.1) 0")
            .1
            .contains("requires 3 numeric")
    );
    assert!(
        parse_err("PAULI_CHANNEL_1(0.1, 2, 0.1) 0")
            .1
            .contains("finite and in [0, 1]")
    );
    parse_qec_program("X_ERROR(0) 0\nX_ERROR(1) 0").unwrap();
}

#[test]
fn postselect_expected_value_is_zero_or_one() {
    let expected = |text: &str| match parse_qec_program(text).unwrap().ops().last() {
        Some(QecOp::Postselect { expected, .. }) => *expected,
        other => panic!("{text}: {other:?}"),
    };
    assert!(!expected("M 0\nPOSTSELECT rec[-1]"));
    assert!(!expected("M 0\nPOSTSELECT(0) rec[-1]"));
    assert!(expected("M 0\nPOSTSELECT(1) rec[-1]"));
    assert!(
        parse_err("M 0\nPOSTSELECT(0.5) rec[-1]")
            .1
            .contains("must be 0 or 1")
    );
    assert!(
        parse_err("M 0\nPOSTSELECT(1, 0) rec[-1]")
            .1
            .contains("zero or one")
    );
}

#[test]
fn observable_index_is_one_unsigned_integer() {
    for (text, expected) in [
        (
            "M 0\nOBSERVABLE_INCLUDE rec[-1]",
            "requires one integer argument",
        ),
        (
            "M 0\nOBSERVABLE_INCLUDE(0, 1) rec[-1]",
            "requires one integer argument",
        ),
        ("M 0\nOBSERVABLE_INCLUDE(-1) rec[-1]", "must be an integer"),
        ("M 0\nOBSERVABLE_INCLUDE(1.5) rec[-1]", "must be an integer"),
    ] {
        assert!(parse_err(text).1.contains(expected), "{text}");
    }
}

#[test]
fn pauli_terms_accept_either_case_and_need_an_index() {
    assert_eq!(
        parse_pauli_product("x0*Y1*z12", 1).unwrap(),
        vec![QecPauli::x(0), QecPauli::y(1), QecPauli::z(12)]
    );
    for (target, expected) in [
        ("X", "missing a qubit index"),
        ("Xa", "invalid qubit index"),
        ("W0", "expected Pauli target"),
        ("X0**Z1", "expected Pauli target"),
        ("", "empty Pauli product"),
        ("!X0", "inverted target"),
    ] {
        let err = parse_pauli_product(target, 1).unwrap_err();
        assert!(err.to_string().contains(expected), "{target:?}: {err}");
    }
}

#[test]
fn pair_instructions_need_an_even_target_count() {
    assert!(parse_err("CX 0 1 2").1.contains("even number"));
    assert!(parse_err("MZZ 0").1.contains("even number"));
    assert!(parse_err("CX").1.contains("at least one target"));
}

#[test]
fn qubit_count_covers_every_noted_qubit() {
    assert_eq!(parse_qec_program("").unwrap().num_qubits(), 0);
    assert_eq!(parse_qec_program("H 2\nM 0").unwrap().num_qubits(), 3);
    assert_eq!(
        parse_qec_program("QUBIT_COORDS(0, 0) 7")
            .unwrap()
            .num_qubits(),
        8
    );
    assert_eq!(parse_qec_program("EXP_VAL X4").unwrap().num_qubits(), 5);
}

#[test]
fn stim_dialect_refuses_every_native_extension() {
    for name in NATIVE_ONLY_INSTRUCTIONS {
        let message = stim_err(&format!("{name} 0"));
        assert!(
            message.contains("native QEC extension"),
            "{name}: {message}"
        );
    }
    assert!(stim_err("FOO 0").contains("unsupported Stim instruction `FOO`"));
    assert!(
        parse_err("FOO 0")
            .1
            .contains("unsupported QEC instruction `FOO`")
    );
}
