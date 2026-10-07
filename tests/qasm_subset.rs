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
         delay[settle] q[0];\ndelay[slack] q;\nrz(settle / pulse) q[1];",
    )
    .expect("parse");
    let reference = openqasm::parse("OPENQASM 3.0;\nqubit[2] q;\nh q[0];\nrz(27) q[1];").unwrap();
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
