//! Gate definitions and subroutines.

use super::*;

#[test]
fn test_gate_def_no_params() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate bell a, b {
            h a;
            cx a, b;
        }
        bell q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
    if let Instruction::Gate { gate, targets } = &c.instructions[0] {
        assert_eq!(*gate, Gate::H);
        assert_eq!(targets.as_slice(), &[0]);
    } else {
        panic!("expected H gate");
    }
    if let Instruction::Gate { gate, targets } = &c.instructions[1] {
        assert_eq!(*gate, Gate::Cx);
        assert_eq!(targets.as_slice(), &[0, 1]);
    } else {
        panic!("expected CX gate");
    }
}

#[test]
fn test_gate_def_with_params() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate myrz(theta) a {
            rz(theta) a;
        }
        myrz(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    } else {
        panic!("expected gate");
    }
}

#[test]
fn test_gate_def_single_line() {
    let qasm = "OPENQASM 3.0;\nqubit[2] q;\ngate mygate a, b { cx a, b; }\nmygate q[0], q[1];";
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        assert_eq!(*gate, Gate::Cx);
    } else {
        panic!("expected CX");
    }
}

#[test]
fn test_gate_def_nested() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate myh a { h a; }
        gate mybell a, b { myh a; cx a, b; }
        mybell q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
}

#[test]
fn test_qasm_gate_def_with_expression_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        gate myrxx(theta) a, b {
            rx(theta/2) a;
            cx a, b;
            rx(-theta/2) a;
            cx a, b;
        }
        myrxx(pi) q[0], q[1];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 4);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[2] {
        match gate {
            Gate::Rx(theta) => {
                assert!((theta - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-12);
            }
            _ => panic!("expected Rx, got {:?}", gate),
        }
    }
}

#[test]
fn test_qasm_gate_def_multi_param_expression() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate myu3(theta, phi, lambda) a {
            rz(lambda) a;
            ry(theta) a;
            rz(phi) a;
        }
        myu3(pi/2, pi/4, pi/8) q[0];
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 3);
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::PI / 8.0).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[1] {
        match gate {
            Gate::Ry(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
            }
            _ => panic!("expected Ry"),
        }
    }
    if let Instruction::Gate { gate, .. } = &c.instructions[2] {
        match gate {
            Gate::Rz(theta) => {
                assert!((theta - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn test_qasm_gate_def_expression_with_arithmetic() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        gate half_rot(theta) a {
            rz(theta + pi) a;
        }
        half_rot(pi/4) q[0];
    "#;
    let c = parse(qasm).unwrap();
    if let Instruction::Gate { gate, .. } = &c.instructions[0] {
        match gate {
            Gate::Rz(theta) => {
                let expected = std::f64::consts::FRAC_PI_4 + std::f64::consts::PI;
                assert!((theta - expected).abs() < 1e-12);
            }
            _ => panic!("expected Rz"),
        }
    }
}

#[test]
fn def_subroutine_basic_inlines() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[2] q;
        def bell(qubit a, qubit b) {
            h a;
            cx a, b;
        }
        bell(q[0], q[1]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 2);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::H,
            targets,
        } => {
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected H"),
    }
    match &c.instructions[1] {
        Instruction::Gate {
            gate: Gate::Cx,
            targets,
        } => {
            assert_eq!(targets.as_slice(), &[0, 1]);
        }
        _ => panic!("expected CX"),
    }
}

#[test]
fn runtime_def_returned_bits_start_fresh_when_the_target_is_an_argument() {
    for (body, expected) in [
        ("bit r; r ^= b; return r;", true),
        ("bit r = 0; r ^= b; return r;", true),
        ("return b;", true),
        ("return b ^ 1;", false),
        ("return -1;", true),
    ] {
        let source = format!("bit c = 1; def f(bit b) -> bit {{ {body} }} c = f(c);");
        let program = parse_dynamic(&source).unwrap();
        let shots = crate::simulate_program(&program)
            .seed(42)
            .shots(10)
            .unwrap();
        assert!(shots.shots.iter().all(|shot| shot == &[expected]), "{body}");
    }
    let program = parse_dynamic("bit c = 1; def f() -> bit { bit r; return r; } c = f();").unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false]));
}

#[test]
fn def_constants_keep_their_lexical_values_under_caller_shadowing() {
    for (source, reference) in [
        ("qubit q; const int n = 4;
          def f() -> int { return n; }
          def g(int n) -> int { return f(); }
          for int n in [8:8] { rx(g(16)) q; }", "qubit q; rx(4) q;"),
        ("qubit q; const duration d = 5ns;
          def wait() -> duration { return d; }
          def g(duration d) -> float { duration v = wait(); return v / 1ns; }
          rx(g(20ns)) q;", "qubit q; rx(5) q;"),
        ("qubit q; def f(int n) -> int { switch(n) { case 1 { return 3; } default { return 4; } } }
          rx(f(1)) q;", "qubit q; rx(3) q;"),
    ] {
        let expected = parse(reference).unwrap();
        let circuit = parse(source).unwrap();
        let program = parse_dynamic(source).unwrap();
        for actual in [&circuit, program.static_circuit().expect("static call")] {
            assert_eq!(format!("{:?}", actual.instructions), format!("{:?}", expected.instructions));
        }
    }
}

#[test]
fn def_scope_cannot_capture_mutable_caller_values_or_registers() {
    for source in [
        "bit c; int n = 2; def f(bit b) -> int { return n + int(b); } int v = f(c);",
        "qubit q; bit c; def f(bit b) -> int { x q; return int(b); } int v = f(c);",
        "bit c; def f(bit b) -> int { c = 1; return int(b); } int v = f(c);",
        "def f() { qubit q; } f();",
    ] {
        assert!(parse(source).is_err(), "{source}");
        assert!(parse_dynamic(source).is_err(), "{source}");
    }
}

#[test]
fn runtime_def_short_circuit_calls_keep_their_gate_effects_conditional() {
    let program = parse_dynamic(
        "qubit[2] q; bit[3] c;
         def flip(qubit a) -> bool { x a; return true; }
         h q[0]; c[0] = measure q[0];
         bool a = bool(c[0]) && flip(q[1]);
         bool b = bool(c[0]) || flip(q[1]);
         c[1] = measure q[1]; c[2] = a == bool(c[0]) && b;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1] && shot[2]));
}

#[test]
fn runtime_def_calls_repeat_in_while_conditions() {
    let program = parse_dynamic(
        "qubit q; bit[2] c = 1;
         def check(bit b, qubit a) -> bool { x a; return bool(b); }
         while (check(c[0], q)) { c[0] = 0; }
         c[1] = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, false]));
}

#[test]
fn runtime_def_return_leaves_nested_loops_and_keeps_scalar_locals() {
    let program = parse_dynamic(
        "qubit q; bit[2] c;
         def f(bit b) -> int {
             int n = 0;
             while (n < 3) { if (n == int(b)) return n; n += 1; }
             return -1;
         }
         h q; c[0] = measure q;
         int count = 0; bool correct = true;
         while (count < 2) {
             c[0] ^= 1;
             int n = f(c[0]); correct = correct && n == int(c[0]); count += 1;
         }
         c[1] = correct;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1]));
}

#[test]
fn runtime_def_bit_parameters_forward_slices_and_keep_writes_local() {
    let program = parse_dynamic(
        "bit[4] c = 2;
         def f(bit[2] b) -> int { b[0] = 1; b[1] ^= 1; return int(b); }
         def g(bit[2] b) -> int { int n; n = f(b[{1,0}]); return n; }
         int n; n = g(c[0:1]); c[2] = n == 3; c[3] = (int(c) & 3) == 2;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(
        shots
            .shots
            .iter()
            .all(|shot| shot == &[false, true, true, true])
    );
}

#[test]
fn runtime_def_results_coerce_and_copy_bit_arguments() {
    let program = parse_dynamic(
        "qubit q; bit[2] c;
         def f(bit b) -> uint[4] { if (b) return 31; return 19; }
         h q; c[0] = measure q;
         int n = f(c[0]); c[1] = n == 3 + 12 * int(c[0]);",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot[1]));
    assert!(shots.shots.iter().any(|shot| shot[0]));
    assert!(shots.shots.iter().any(|shot| !shot[0]));
}

#[test]
fn runtime_def_return_keeps_the_callers_loop_running() {
    let program = parse_dynamic(
        "qubit q; bit[3] c = 1;
         def f(bit b, qubit a) { if (b) return; x a; }
         int n = 0;
         while (n < 2) { n += 1; f(c[0], q); x q; }
         c[1] = measure q; c[2] = n == 2;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, false, true]));
}

#[test]
fn runtime_def_nested_results_keep_distinct_values() {
    let program = parse_dynamic(
        "bit[3] c = 2;
         def f(bit b) -> int { if (b) return 3; return 1; }
         def g(bit a, bit b) -> int { return f(a) + f(b); }
         int n = g(c[0], c[1]); c[2] = n == 4;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, true, true]));
}

#[test]
fn runtime_def_scalar_parameters_are_values_with_declared_widths() {
    let program = parse_dynamic(
        "bit[2] c = 1;
         def f(uint[4] n) -> uint[4] { if (n > 0) n += 3; return n; }
         uint[8] n = int(c[0]) * 18;
         int r = f(n); c[1] = n == 18 && r == 5;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true]));
    assert!(matches!(
        parse_dynamic(
            "bit c; def f(int n, bit b) -> int { return n + int(b); } int v = f(0.5, c);"
        ),
        Err(PrismError::Parse { .. })
    ));
}

#[test]
fn runtime_def_requires_a_result_on_every_reachable_path() {
    let error =
        parse_dynamic("bit c; def f(bit b) -> int { if (b) return 1; } int n = f(c);").unwrap_err();
    assert!(
        matches!(error, PrismError::Parse { message, .. } if message.contains("without a result"))
    );
}

#[test]
fn runtime_def_call_can_decide_a_branch() {
    let program = parse_dynamic(
        "qubit q; bit[2] c = 1;
         def f(bit b) -> bool { return bool(b); }
         if (f(c[0])) x q; c[1] = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true]));
}

#[test]
fn def_scalar_results_coerce_at_the_declared_width() {
    for (signature, value, expected) in [
        ("int[4]", "9", "-7"),
        ("uint[4]", "19", "3"),
        ("bool", "2", "1"),
        ("float[16]", "1.0001", "1.0"),
        ("float[32]", "16777217", "16777216.0"),
        ("angle[3]", "pi / 3", "pi / 4"),
        ("int", "9007199254740993 - 9007199254740992", "1"),
    ] {
        let source = format!(
            "qubit q; def f() -> {signature} {{ return {value}; }} \
             float v = f(); rx(v) q;"
        );
        let actual = parse(&source).unwrap_or_else(|error| panic!("{signature}: {error}"));
        let reference = parse(&format!("qubit q; rx({expected}) q;")).unwrap();
        assert_eq!(
            format!("{:?}", actual.instructions),
            format!("{:?}", reference.instructions)
        );
    }
}

#[test]
fn def_signature_widths_keep_the_declaration_scope() {
    for (source, reference) in [
        (
            "qubit q; def ratio(float n, float d) -> float { return n / d; } rx(ratio(3, 5)) q;",
            "qubit q; rx(0.6) q;",
        ),
        (
            "qubit q; const int n = 4;
             def f() -> uint[n] { return 31; }
             uint[8] r; for int n in [8:8] { r = f(); } rx(r) q;
             def caller(int n) -> uint[8] { return f(); }
             r = caller(8); rx(r) q;",
            "qubit q; rx(15) q; rx(15) q;",
        ),
        (
            "qubit[2] q; const int n = 2;
             def f(qubit[n] a) { x a[1]; }
             for int n in [1:1] { f(q); }",
            "qubit[2] q; x q[1];",
        ),
    ] {
        let expected = parse(reference).unwrap();
        let circuit = parse(source).unwrap();
        let program = parse_dynamic(source).unwrap();
        for actual in [
            &circuit,
            program.static_circuit().expect("static signature"),
        ] {
            assert_eq!(
                format!("{:?}", actual.instructions),
                format!("{:?}", expected.instructions)
            );
        }
    }
    let wrong_width = "qubit[2] q; const int n = 2;
        def f(qubit[n] a) { x a; }
        for int n in [1:1] { f(q[0]); }";
    assert!(matches!(parse(wrong_width), Err(PrismError::Parse { .. })));
    assert!(matches!(
        parse_dynamic(wrong_width),
        Err(PrismError::Parse { .. })
    ));
}

#[test]
fn def_early_return_stops_nested_blocks_and_loop_iterations() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) -> int {
             for int i in [0:5] {
                 if (i == 2) { box { return i; } }
                 x a;
             }
             z a;
             return 7;
         }
         int n = f(q);
         rx(n) q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; x q; rx(2) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_calls_in_expressions_preserve_order_and_exact_results() {
    let circuit = parse(
        "qubit q;
         def a(qubit t) -> int { x t; return 2; }
         def b(qubit t) -> int { z t; return 3; }
         def twice(int n) -> int { return n * 2; }
         int v = twice(a(q)) + b(q);
         v += twice(2);
         rx(v) q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; z q; rx(11) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_void_return_skips_the_remaining_gates_and_keeps_the_caller() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) { x a; return; z a; }
         f(q); h q;",
    )
    .unwrap();
    let reference = parse("qubit q; x q; h q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_static_short_circuit_does_not_execute_a_skipped_call() {
    let circuit = parse(
        "qubit q;
         def f(qubit a) -> bool { x a; return true; }
         bool a = false && f(q);
         bool b = true || f(q);
         rx(int(a) + int(b)) q;",
    )
    .unwrap();
    let reference = parse("qubit q; rx(1) q;").unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn def_duration_result_keeps_its_units() {
    parse(
        "qubit q; def wait() -> duration { return 5ns; }
         duration d = wait(); delay[d] q;",
    )
    .unwrap();
    assert!(parse("def f() -> duration { return 5; } f();").is_err());
}

#[test]
fn def_result_requires_a_return_on_the_taken_path() {
    let error = parse("def f(int n) -> int { if (n > 0) return n; } int v = f(0);").unwrap_err();
    assert!(matches!(error, PrismError::Parse { message, .. }
        if message.contains("reaches its end without a result")));
}

#[test]
fn def_scalar_calls_keep_the_static_dynamic_program_fast_path() {
    let source = "qubit q;
        def f(int n) -> int { if (n > 1) return n; return 0; }
        if (f(2)) { rx(f(3)) q; }";
    let reference = parse(source).unwrap();
    let dynamic = parse_dynamic(source).unwrap();
    let circuit = dynamic.static_circuit().expect("no runtime values");
    assert_eq!(
        format!("{:?}", reference.instructions),
        format!("{:?}", circuit.instructions)
    );
}

#[test]
fn def_call_gates_precede_runtime_actions_using_the_result() {
    let program = parse_dynamic(
        "qubit q; bit c; h q;
         def f(qubit a) -> float { h a; return pi; }
         float t = f(q) + int(c) * 0.0;
         rz(t) q; c = measure q;",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false]));
}

#[test]
fn def_arguments_read_the_caller_before_parameter_names_are_bound() {
    let source = "qubit q; int a = 3;
        def f(int a, int b) -> int { return b; }
        int v = f(1, a); rx(v) q;";
    same_instructions(source, "qubit q; rx(3) q;");
    same_instructions(
        "qubit[2] q; def f(qubit a) -> int { x a; return 2; }
         int v = f(q[1]); rx(v) q[0];",
        "qubit[2] q; x q[1]; rx(2) q[0];",
    );
}

#[test]
fn def_runtime_result_paths_need_dynamic_programs() {
    for source in [
        "qubit q; bit c; def f(bit b) -> int { if (b) return 1; return 0; }
         c = measure q; int n = f(c);",
        "qubit q; bit c; def f(bit b) -> int { return int(b); }
         c = measure q; int n = f(c);",
        "qubit q; bit c; def f() -> bool { return true; }
         if (c && f()) x q;",
    ] {
        assert!(
            matches!(parse(source), Err(PrismError::UnsupportedConstruct { .. })),
            "{source}"
        );
        parse_dynamic(source).unwrap();
    }
}

#[test]
fn def_qubit_register_parameters_preserve_sliced_order() {
    same_instructions(
        "qubit[4] q;
         def f(qubit[2] a, qubit b) { cx a[0], b; x a[1]; }
         f(q[1:2:3], q[0]);",
        "qubit[4] q; cx q[1], q[0]; x q[3];",
    );
    for source in [
        "qubit[3] q; def f(qubit[2] a) { h a; } f(q);",
        "qubit[2] q; def f(qubit[2] a, qubit b) {} f(q, q[0]);",
        "qubit[2] q; def f(qubit[2] a) {} f(q[{0,0}]);",
    ] {
        assert!(
            matches!(parse(source), Err(PrismError::Parse { .. })),
            "{source}"
        );
    }
}

#[test]
fn def_scalar_results_assign_to_arrays_and_classical_bits() {
    same_instructions(
        "qubit q; def f() -> int { return 3; }
         array[int, 2] a; a[1] = f(); a[0] += f() + 1; rx(a[0] + a[1]) q;",
        "qubit q; rx(7) q;",
    );
    same_instructions(
        "qubit q; def f() -> float { return pi / 2; }
         float v = 0; v = sin(f()); rx(v) q;",
        "qubit q; rx(1) q;",
    );
    let program =
        parse_dynamic("bit[3] c; def f() -> int { return 3; } c = f(); c[2] = f();").unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(10)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[true, true, true]));
}

#[test]
fn def_register_result_runs_inside_a_repeat_until_success_loop() {
    let program = parse_dynamic(
        "qubit[2] q; bit[2] c = 3;
         def check(qubit[2] a) -> bit[2] {
             bit[2] r; reset a; h a; measure a -> r; return r;
         }
         while (int(c) != 0) { c = check(q); }",
    )
    .unwrap();
    let shots = crate::simulate_program(&program)
        .seed(42)
        .shots(100)
        .unwrap();
    assert!(shots.shots.iter().all(|shot| shot == &[false, false]));
}

#[test]
fn def_subroutine_with_float_parameter() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        def my_rx(float theta, qubit a) {
            rx(theta) a;
        }
        my_rx(0.5, q[0]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            targets,
        } => {
            assert!((theta - 0.5).abs() < 1e-12);
            assert_eq!(targets.as_slice(), &[0]);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn def_subroutine_with_int_parameter_used_as_index() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        def at_idx(int i, qubit a) {
            rx(i * pi / 2) a;
        }
        at_idx(2, q[1]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 1);
    match &c.instructions[0] {
        Instruction::Gate {
            gate: Gate::Rx(theta),
            targets,
        } => {
            assert!((theta - std::f64::consts::PI).abs() < 1e-12);
            assert_eq!(targets.as_slice(), &[1]);
        }
        _ => panic!("expected Rx"),
    }
}

#[test]
fn def_subroutine_with_for_loop_inside_body() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[3] q;
        def all_h(qubit a, qubit b, qubit c) {
            for int i in [0:2] {
                rx(i * pi / 2) a;
            }
            h b;
            h c;
        }
        all_h(q[0], q[1], q[2]);
    "#;
    let c = parse(qasm).unwrap();
    assert_eq!(c.gate_count(), 5);
}

#[test]
fn def_subroutine_called_inside_for() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[4] q;
        def step(qubit a) {
            h a;
        }
        for int i in [0:3] {
            step(q[i]);
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
fn if_register_compare_with_int_var_in_def() {
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        bit[4] c;
        def cond(int n, bit[4] b, qubit a) {
            if (b == n) x a;
        }
        cond(3, c, q[0]);
    "#;
    let c = parse(qasm).unwrap();
    match &c.instructions[0] {
        Instruction::Conditional { condition, .. } => {
            assert!(matches!(
                condition,
                ClassicalCondition::RegisterEquals { value: 3, .. }
            ));
        }
        _ => panic!("expected conditional"),
    }
}

#[test]
fn an_empty_def_is_a_call_that_does_nothing() {
    let c =
        parse("OPENQASM 3.0;\nqubit[1] q;\ndef idle(qubit a) { }\nidle(q[0]);\nx q[0];").unwrap();
    assert_eq!(c.instructions.len(), 1);
    let err = parse("OPENQASM 3.0;\nqubit[1] q;\ndef f(qubit a) -> bit { }").unwrap_err();
    assert!(err.to_string().contains("body is empty"), "{err}");
}
