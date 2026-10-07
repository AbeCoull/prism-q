//! Timing constructs, `box`, and subroutines with classical arguments and returns.

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
