//! Timing constructs, `box`, subroutines with classical arguments and returns, and
//! classical arrays.

use prism_q::PrismError;
use prism_q::circuit::{Circuit, openqasm};

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

fn assert_same_stream(body: &str, reference: &str) {
    let circuit = parse(body);
    let expected = parse(reference);
    assert_eq!(
        circuit.num_classical_bits, expected.num_classical_bits,
        "`{body}`"
    );
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", expected.instructions),
        "`{body}`"
    );
}

#[test]
fn a_delay_is_the_identity_on_its_qubits() {
    assert_same_stream(
        "h q[0];\ndelay[100ns] q[0];\ncx q[0], q[1];\ndelay[2.5us] q;\ndelay[4dt];\n\
         delay[1ms] q[1], q[2];\ndelay[3\u{b5}s] q[3];\ndelay[3\u{3bc}s] q[3];\ndelay[0.5s] q[0];",
        "h q[0];\ncx q[0], q[1];",
    );
}

#[test]
fn duration_arithmetic_folds_statically() {
    assert_same_stream(
        "const duration d = 50ns;\n\
         duration e = 2 * d + 1us;\n\
         delay[e - d] q[0];\n\
         rx(e / 100ns) q[0];\n\
         duration f = d;\n\
         f += 25ns;\n\
         f *= 2;\n\
         rx(f / 25ns) q[1];\n\
         duration t = 8dt;\n\
         rx(t / 2dt) q[2];\n\
         int n = 1us / 250ns;\n\
         h q[n - 1];",
        "rx(11) q[0];\nrx(6) q[1];\nrx(4) q[2];\nh q[3];",
    );
}

#[test]
fn the_guide_timing_example_folds_its_ratio() {
    let circuit = openqasm::parse(
        "OPENQASM 3.0;\nqubit[2] q;\nconst duration pulse = 40ns;\n\
         duration settle = 2 * pulse + 1us;\nstretch slack;\nh q[0];\n\
         delay[settle] q[0];\ndelay[slack] q;\nrz(settle / pulse) q[1];\n\
         box[settle] {\n  cx q[0], q[1];\n}",
    )
    .expect("parse");
    let reference =
        openqasm::parse("OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nrz(27) q[1];\ncx q[0], q[1];")
            .unwrap();
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn a_stretch_reads_anywhere_a_delay_takes_a_length() {
    assert_same_stream(
        "stretch s;\ndelay[s] q[0];\ndelay[2 * s + 10ns] q;\nx q[1];",
        "x q[1];",
    );
}

#[test]
fn a_def_takes_a_duration_argument() {
    assert_same_stream(
        "def wait(duration d, qubit a) { delay[d] a; x a; }\nwait(20ns, q[0]);\nwait(2 * 3dt, q[1]);",
        "x q[0];\nx q[1];",
    );
}

// The body joins the stream exactly as if the box were absent, so fusion sees no
// boundary to stop at.
#[test]
fn a_box_runs_its_body_in_place() {
    assert_same_stream(
        "h q[0];\nbox { cx q[0], q[1]; rz(0.3) q[1]; }\nbox[1us] { x q[2]; }\n\
         duration d = 20ns;\nbox[2 * d] { box { h q[3]; } }",
        "h q[0];\ncx q[0], q[1];\nrz(0.3) q[1];\nx q[2];\nh q[3];",
    );
    assert_same_stream(
        "c[0] = measure q[0];\nif (c[0]) { box { x q[1]; } }",
        "c[0] = measure q[0];\nif (c[0]) { x q[1]; }",
    );
    assert_same_stream(
        "def layer(qubit a) { box[40ns] { h a; } }\nlayer(q[2]);",
        "h q[2];",
    );
}

#[test]
fn a_box_scopes_what_it_declares() {
    // An assignment to an outer name survives the box; a declaration inside it
    // does not, so the same name can be declared again after it.
    assert_same_stream(
        "int k = 0;\nbox { int j = 2; k = j + 1; }\nh q[k];\nint j = 1;\nx q[j];",
        "h q[3];\nx q[1];",
    );
    assert!(matches!(
        parse_err("box { int j = 2; }\nh q[j];"),
        PrismError::Parse { .. }
    ));
}

#[test]
fn a_box_with_an_unusable_length_is_rejected() {
    for (body, needle) in [
        ("box[10] { x q[0]; }", "where a duration belongs"),
        ("box[-1ns] { x q[0]; }", "non-negative"),
        ("box[10ns] x q[0];", "expected `{`"),
        ("box { x q[0];", "`}`"),
    ] {
        match parse_err(body) {
            PrismError::Parse { message, .. } => {
                assert!(message.contains(needle), "`{body}`: {message}");
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
}

#[test]
fn timing_that_cannot_be_evaluated_is_rejected_by_name() {
    let unsupported = [
        ("stretch s;\nrx(s / 1ns) q[0];", "stretch"),
        ("rx(10ns / 2dt) q[0];", "`dt`"),
        ("delay[durationof({ x q[0]; })] q[0];", "`durationof`"),
    ];
    for (body, needle) in unsupported {
        match parse_err(body) {
            PrismError::UnsupportedConstruct { construct, line } => {
                assert!(construct.contains(needle), "`{body}`: {construct}");
                assert!(line >= 4, "`{body}` reported line {line}");
            }
            other => panic!("`{body}` should decline by name, got {other:?}"),
        }
    }

    let malformed = [
        ("delay[10] q[0];", "where a duration belongs"),
        ("delay[-5ns] q[0];", "non-negative"),
        (
            "duration d = 10ns;\nrx(d) q[0];",
            "a duration where a number belongs",
        ),
        (
            "stretch s;\nrx(s) q[0];",
            "a stretch where a number belongs",
        ),
        (
            "delay[10ns * 2ns] q[0];",
            "between a duration and a duration",
        ),
        ("delay[10ns + 1] q[0];", "between a duration and a number"),
        ("stretch s;\ns = 10ns;", "stretch"),
        ("duration d = 10ns;\nd /= 2ns;", "holding the number"),
        ("rx(sin(10ns)) q[0];", "takes numbers"),
        ("rx(1ns / 0ns) q[0];", "division by zero"),
        ("const duration d;", "needs a value"),
        ("delay q[0];", "expected `[`"),
    ];
    for (body, needle) in malformed {
        match parse_err(body) {
            PrismError::Parse { message, .. } => {
                assert!(message.contains(needle), "`{body}`: {message}");
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }

    assert!(matches!(
        parse_err("delay[10ns] r[0];"),
        PrismError::UndefinedRegister { .. }
    ));
    assert!(matches!(
        parse_err("delay[10ns] q[9];"),
        PrismError::InvalidQubit { .. }
    ));
}

#[test]
fn a_def_returning_a_measurement_writes_its_target() {
    assert_same_stream(
        "def mx(qubit a) -> bit { h a; return measure a; }\nc[1] = mx(q[0]);",
        "h q[0];\nc[1] = measure q[0];",
    );
    assert_same_stream(
        "def mx(qubit a) -> bit { h a; return measure a; }\n\
         c[0] = measure q[2];\nif (c[0]) { c[1] = mx(q[0]); }",
        "c[0] = measure q[2];\nif (c[0]) { h q[0]; c[1] = measure q[0]; }",
    );

    let circuit = parse("def mz(qubit a) -> bit { return measure a; }\nx q[0];\nc[3] = mz(q[0]);");
    let outcome = prism_q::simulate(&circuit).seed(42).run().expect("run");
    assert_eq!(outcome.classical_bits, vec![false, false, false, true]);
}

// The bit a body declares and returns is the caller's target, so a branch on it
// inside the body reads the measurement that is also the result.
#[test]
fn a_def_returning_a_declared_bit_writes_through_it() {
    assert_same_stream(
        "def flip(qubit a, qubit b) -> bit { bit r; r = measure a; if (r) x b; return r; }\n\
         c[2] = flip(q[0], q[1]);",
        "c[2] = measure q[0];\nif (c[2]) x q[1];",
    );
    assert_same_stream(
        "def both(qubit a, qubit b) -> bit[2] {\n\
           bit[2] r;\n  r[0] = measure a;\n  measure b -> r[1];\n  return r;\n}\n\
         c[1:2] = both(q[0], q[3]);",
        "c[1] = measure q[0];\nc[2] = measure q[3];",
    );
}

#[test]
fn a_def_reads_a_bit_argument() {
    assert_same_stream(
        "def fix(bit b, qubit a) { if (b) x a; if (!b) z a; }\n\
         c[0] = measure q[0];\nfix(c[0], q[1]);",
        "c[0] = measure q[0];\nif (c[0]) x q[1];\nif (!c[0]) z q[1];",
    );
    let circuit = parse("def pick(bit[2] s, qubit a) { if (s == 2) x a; }\npick(c[2:3], q[0]);");
    assert!(
        format!("{:?}", circuit.instructions)
            .contains("RegisterEquals { offset: 2, size: 2, value: 2 }"),
        "{:?}",
        circuit.instructions
    );
}

#[test]
fn a_single_bit_register_reads_as_a_condition() {
    let text = "OPENQASM 3.0;\nqubit[2] q;\nbit flag;\nflag = measure q[0];\nif (flag) x q[1];";
    let reference =
        "OPENQASM 3.0;\nqubit[2] q;\nbit flag;\nflag = measure q[0];\nif (flag[0]) x q[1];";
    assert_eq!(
        format!("{:?}", openqasm::parse(text).unwrap().instructions),
        format!("{:?}", openqasm::parse(reference).unwrap().instructions)
    );
}

#[test]
fn a_builtin_call_still_assigns_a_classical_value() {
    assert_same_stream(
        "float t = 0.5;\nfloat y = 0;\ny = sin(t);\nrx(y) q[0];",
        "rx(sin(0.5)) q[0];",
    );
}

#[test]
fn a_def_needing_more_than_a_guarded_region_declines_by_name() {
    let unsupported = [
        (
            "def mx(qubit a) -> bit { return measure a; }\nmx(q[0]);",
            "drops its `bit` result",
        ),
        ("def f(qubit a) { return; h a; }", "before the end"),
        (
            "def f(qubit a, bit b) { if (b) { return; } h a; }",
            "before the end",
        ),
        (
            "def f(qubit a, bit b) { b = measure a; }",
            "passed by value",
        ),
        (
            "def f(qubit a) { c[0] = measure a; }",
            "only through its result",
        ),
        (
            "def f(qubit a) { bit m; m = measure a; if (m) x a; }",
            "does not return it",
        ),
        ("def f(bit b) -> bit { return b; }", "bit parameter"),
        ("def f(qubit a) -> bit { return 1; }", "return of `1`"),
        ("def f(qubit a) -> int { return 1; }", "returning `int`"),
        (
            "def f(bit b, qubit a) -> bit { bit r; r = measure a; if (b) x a; return r; }\n\
             c[0] = f(c[0], q[0]);",
            "passing a bit it also assigns",
        ),
        ("return;", "outside a def"),
    ];
    for (body, needle) in unsupported {
        match parse_err(body) {
            PrismError::UnsupportedConstruct { construct, .. } => {
                assert!(construct.contains(needle), "`{body}`: {construct}");
            }
            other => panic!("`{body}` should decline by name, got {other:?}"),
        }
    }

    let malformed = [
        (
            "def mx(qubit a) -> bit { return measure a; }\nc = mx(q[0]);",
            "returns 1 bit(s)",
        ),
        (
            "def f(qubit a) { h a; }\nc[0] = f(q[0]);",
            "declares no result",
        ),
        (
            "def f(qubit a) -> bit { h a; }",
            "does not end by returning",
        ),
        (
            "def f(qubit a) { return measure a; }",
            "declares no `-> bit`",
        ),
        (
            "def f(bit[2] b, qubit a) { if (b == 1) x a; }\nf(c[0], q[0]);",
            "takes 2 bit(s)",
        ),
        ("def f(bit b, qubit a) { h a; }\nf(1, q[0]);", "needs a bit"),
        (
            "def f(qubit a) -> bit { bit[2] r; r[0] = measure a; return r; }\nc[0] = f(q[0]);",
            "declares 2 bit(s)",
        ),
        ("float y = 0;\ny = sin(q[0]);", "where a value belongs"),
    ];
    for (body, needle) in malformed {
        match parse_err(body) {
            PrismError::Parse { message, .. } => {
                assert!(message.contains(needle), "`{body}`: {message}");
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
}

#[test]
fn the_guide_subroutine_example_inlines_to_its_expansion() {
    let guide = "OPENQASM 3.0;\nqubit[3] q;\nbit[3] c;\n\
                 def mx(qubit a) -> bit {\n  h a;\n  return measure a;\n}\n\
                 def herald(qubit a, qubit flag) -> bit {\n  bit r;\n  r = measure a;\n  \
                 if (r) x flag;\n  return r;\n}\n\
                 def fix(bit b, qubit a) {\n  if (b) z a;\n}\n\
                 c[0] = mx(q[0]);\nc[1] = herald(q[1], q[2]);\nfix(c[0], q[2]);";
    let expanded = "OPENQASM 3.0;\nqubit[3] q;\nbit[3] c;\n\
                    h q[0];\nc[0] = measure q[0];\n\
                    c[1] = measure q[1];\nif (c[1]) x q[2];\n\
                    if (c[0]) z q[2];";
    assert_eq!(
        format!("{:?}", openqasm::parse(guide).unwrap().instructions),
        format!("{:?}", openqasm::parse(expanded).unwrap().instructions)
    );
}

#[test]
fn an_array_element_reads_anywhere_a_value_does() {
    assert_same_stream(
        "array[int[32], 3] a = {1, 2, 3};\nh q[a[1]];\nrx(a[2] * pi / 4) q[0];\n\
         for int k in [0:2] { x q[a[k]]; }\ndelay[a[0] * 1ns] q[0];",
        "h q[2];\nrx(3 * pi / 4) q[0];\nx q[1];\nx q[2];\nx q[3];",
    );
    assert_same_stream(
        "const array[float[64], 2, 2] m = {{0.1, 0.2}, {0.3, 0.4}};\n\
         rx(m[1, 0]) q[0];\nry(m[0][1]) q[1];\nrz(m[1][1] * 2) q[2];",
        "rx(0.3) q[0];\nry(0.2) q[1];\nrz(0.4 * 2) q[2];",
    );
    assert_same_stream(
        "def r(float t, qubit a) { rx(t) a; }\narray[float, 2] th = {0.5, 0.7};\n\
         r(th[1], q[0]);\nrx(th[0]) q[1];\nrx(th[0] + 1) q[2];",
        "rx(0.7) q[0];\nrx(0.5) q[1];\nrx(1.5) q[2];",
    );
}

// Filling an array in a loop is the common write, so an element write outlives
// the pass that made it while a declaration does not.
#[test]
fn an_array_element_takes_assignments() {
    assert_same_stream(
        "array[int, 4] idx;\nfor int k in [0:3] { idx[k] = 3 - k; }\nidx[0] += 0;\n\
         h q[idx[0]];\nh q[idx[3]];\n\
         array[bool, 2, 2] f = {{true, false}, {false, true}};\nf[0, 1] = 1;\n\
         if (c == f[0, 1]) x q[f[1][1]];",
        "h q[3];\nh q[0];\nif (c == 1) x q[1];",
    );
    assert_same_stream(
        "box { array[int, 1] t = {2}; h q[t[0]]; }\narray[int, 1] t = {1};\nh q[t[0]];\n\
         for int k in [0:1] { array[int, 1] u = {k}; x q[u[0]]; }",
        "h q[2];\nh q[1];\nx q[0];\nx q[1];",
    );
}

#[test]
fn a_def_reads_only_constant_arrays() {
    assert_same_stream(
        "const array[int, 2] k = {1, 2};\ndef g(qubit a) { rx(k[1]) a; }\ng(q[0]);",
        "rx(2) q[0];",
    );
    match parse_err("array[int, 2] k = {1, 2};\ndef g(qubit a) { rx(2 * k[1]) a; }\ng(q[0]);") {
        PrismError::Parse { message, .. } => {
            assert!(message.contains("not a declared array"), "{message}")
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn an_array_that_cannot_stand_is_rejected() {
    let deep = format!("array[int, 1] a = {}1{};", "{".repeat(40), "}".repeat(40));
    let malformed = [
        ("array[int, 2] a = {1, 2, 3};", "lists 3 entries"),
        ("array[int, 2, 2] a = {1, 2};", "does not match"),
        ("array[int, 2] a;\nh q[a[2]];", "outside `a`"),
        ("array[int, 2] a;\nh q[a[0, 1]];", "indexed with 2"),
        ("const array[int, 2] a = {1, 2};\na[0] = 3;", "`const`"),
        ("const array[int, 2] a;", "needs a value"),
        ("array[int, 0] a;", "must be > 0"),
        ("array[int, 2048, 2048] a;", "more than"),
        ("array[int, 2] a;\nint a = 1;", "already an array"),
        (
            "int a = 1;\narray[int, 2] a;",
            "already a classical variable",
        ),
        ("rx(b[0] + 1) q[0];", "not a declared array"),
        ("c[0] = 1;", "not a name an assignment can write"),
        ("array[int, 2] a;\na[0] /= 0;", "division by zero"),
        (deep.as_str(), "nests deeper"),
    ];
    for (body, needle) in malformed {
        match parse_err(body) {
            PrismError::Parse { message, .. } => {
                assert!(message.contains(needle), "`{body}`: {message}");
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
    for (body, needle) in [
        ("array[bit, 2] a;", "`bit[n]` register"),
        ("array[duration, 2] a;", "`array[duration, ...]`"),
    ] {
        match parse_err(body) {
            PrismError::UnsupportedConstruct { construct, .. } => {
                assert!(construct.contains(needle), "`{body}`: {construct}");
            }
            other => panic!("`{body}` should decline by name, got {other:?}"),
        }
    }
}

#[test]
fn the_guide_array_example_folds_to_its_gate() {
    assert_same_stream(
        "array[float[64], 2, 2] angles = {{0.1, 0.2}, {0.3, 0.4}};\narray[int, 4] order;\n\
         for int k in [0:3] { order[k] = 3 - k; }\nrx(angles[1, 0]) q[order[0]];",
        "rx(0.3) q[3];",
    );
}

// A loop body's writes to names declared outside it are the loop's whole
// effect on them, so they have to outlive each pass.
#[test]
fn a_loop_keeps_writes_to_outer_names() {
    assert_same_stream(
        "int n = 0;\nfor int i in [0:2] { n += 1; }\nh q[n];",
        "h q[3];",
    );
    assert_same_stream(
        "int n = 0;\nfor int i in [0:1] { for int j in [0:0] { n += 1; } n += i; }\nh q[n];",
        "h q[3];",
    );
    assert_same_stream(
        "float t = 0;\nfor int i in [1:3] { t = t + i; }\nrx(t) q[0];",
        "rx(6) q[0];",
    );
    assert_same_stream(
        "duration d = 0ns;\nfor int i in [0:2] { d += 10ns; }\nrx(d / 10ns) q[0];",
        "rx(3) q[0];",
    );
}

#[test]
fn a_loop_body_declaration_and_variable_stay_inside() {
    assert_same_stream(
        "int i = 2;\nfor int i in [0:1] { h q[i]; }\nx q[i];",
        "h q[0];\nh q[1];\nx q[2];",
    );
    for body in [
        "for int i in [0:1] { int j = i; }\nh q[j];",
        "for int i in [0:1] { h q[i]; }\nh q[i];",
        "for int i in [0:1] { duration d = 1ns; }\ndelay[d] q[0];",
    ] {
        assert!(
            matches!(parse_err(body), PrismError::Parse { .. }),
            "`{body}`"
        );
    }
}

// A def sees no non-constant global it does not take as an argument, so a
// write to one inside the body cannot be inlined and is rejected rather than
// dropped.
#[test]
fn a_def_cannot_write_an_outer_name() {
    for body in [
        "int n = 0;\ndef f(qubit a) { n += 1; h a; }\nf(q[0]);\nh q[n];",
        "duration d = 0ns;\ndef f(qubit a) { d += 1ns; h a; }\nf(q[0]);",
    ] {
        match parse_err(body) {
            PrismError::Parse { message, .. } => {
                assert!(
                    message.contains("not a declared classical variable"),
                    "`{body}`: {message}"
                );
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
    assert_same_stream(
        "const int k = 2;\ndef f(qubit a) { int m = k; m += 1; rx(m) a; }\nf(q[0]);",
        "rx(3) q[0];",
    );
}

// A classical variable is folded at parse time, so a write under a guard that a
// measurement decides would take effect whatever the measurement read.
#[test]
fn a_write_under_a_runtime_guard_is_declined() {
    for body in [
        "int n = 0;\nc[0] = measure q[0];\nif (c[0]) { n = 1; }\nh q[n];",
        "int n = 0;\nif (c[0]) n += 1;\nh q[n];",
        "int n = 0;\nif (c[0]) { x q[0]; } else { n = 2; }",
        "int n = 0;\nif (c[0]) { x q[0]; } else if (c[1]) { n -= 1; }",
        "int n = 0;\nswitch (c) { case 1 { n = 1; } default { x q[0]; } }",
        "int n = 0;\nswitch (c) { case 1 { x q[0]; } default { n *= 2; } }",
        "int n = 0;\nif (c[0]) { if (c[1]) { n += 1; } }",
        "int n = 0;\nfor int i in [0:1] { if (c[0]) { n = i; } }",
        "int n = 0;\nif (c[0]) { for int i in [0:1] { n += i; } }",
        "if (c[0]) { int k = 0; if (c[1]) { k = 1; } h q[k]; }",
        "float t = 0;\nif (c[0]) { t = sin(0.5); }",
        "duration d = 0ns;\nif (c[0]) { d += 10ns; }",
        "array[int, 2] a;\nif (c[0]) { a[1] = 3; }",
        "def f(qubit a, bit b) { int k = 0; if (b) { k = 1; } rx(k) a; }\nf(q[0], c[0]);",
    ] {
        match parse_err(body) {
            PrismError::UnsupportedConstruct { construct, .. } => {
                assert!(
                    construct.contains("runtime `if` or `switch`"),
                    "`{body}`: {construct}"
                );
            }
            other => panic!("`{body}` should decline by name, got {other:?}"),
        }
    }
}

#[test]
fn a_runtime_guard_keeps_what_does_not_write_outside_it() {
    assert_same_stream(
        "if (c[0]) { int k = 1; k += 1; h q[k]; }",
        "if (c[0]) { h q[2]; }",
    );
    assert_same_stream(
        "const int k = 2;\nint m = 3;\nif (c[0]) { x q[k]; rx(m) q[m]; }",
        "if (c[0]) { x q[2]; rx(3) q[3]; }",
    );
    assert_same_stream(
        "if (c[0]) { x q[1]; c[1] = measure q[1]; for int i in [0:1] { h q[i]; } }",
        "if (c[0]) { x q[1]; c[1] = measure q[1]; h q[0]; h q[1]; }",
    );
    assert_same_stream(
        "switch (c) { case 1 { array[int, 1] a = {2}; a[0] += 1; h q[a[0]]; } }",
        "switch (c) { case 1 { h q[3]; } }",
    );
    // A name declared under the guard goes out of scope with it.
    assert!(matches!(
        parse_err("if (c[0]) { int k = 1; }\nh q[k];"),
        PrismError::Parse { .. }
    ));
}
