//! Timing folds under a model where every gate takes no time: duration literals
//! and arithmetic, `durationof`, stretches a fixed-length `box` resolves, and
//! `delay` and `box` leaving the circuit as if they were absent.

use prism_q::circuit::{Circuit, openqasm};
use prism_q::{PrismError, simulate};

fn program(header: &str, body: &str) -> String {
    format!("OPENQASM 3.0;\n{header}\n{body}\n")
}

fn parse(body: &str) -> Circuit {
    let text = program("qubit[4] q;\nbit[4] c;", body);
    openqasm::parse(&text).unwrap_or_else(|e| panic!("`{body}`: {e}"))
}

fn parse_err(body: &str) -> PrismError {
    let text = program("qubit[4] q;\nbit[4] c;", body);
    openqasm::parse(&text)
        .err()
        .unwrap_or_else(|| panic!("`{body}` should not parse"))
}

fn assert_same_stream(body: &str, reference: &str) {
    let circuit = parse(body);
    let expected = parse(reference);
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", expected.instructions),
        "`{body}`"
    );
}

fn assert_declines(body: &str, needle: &str) {
    match parse_err(body) {
        PrismError::UnsupportedConstruct { construct, .. } => {
            assert!(construct.contains(needle), "`{body}`: {construct}");
        }
        other => panic!("`{body}` should decline by name, got {other:?}"),
    }
}

fn assert_parse_error(body: &str, needle: &str) {
    match parse_err(body) {
        PrismError::Parse { message, .. } => {
            assert!(message.contains(needle), "`{body}`: {message}");
        }
        other => panic!("`{body}` should be a parse error, got {other:?}"),
    }
}

#[test]
fn every_unit_folds_to_one_length() {
    assert_same_stream(
        "rx(1s / 1ms) q[0];\nrx(1ms / 1us) q[1];\nrx(1us / 1ns) q[2];\n\
         rx(3\u{b5}s / 3\u{3bc}s) q[3];\nrx(2us / 500ns) q[0];\nrx(8dt / 2dt) q[1];\n\
         rx(1000 ms / 1 s) q[2];\nrx(1.5e3ns / 1us) q[3];",
        "rx(1000) q[0];\nrx(1000) q[1];\nrx(1000) q[2];\nrx(1) q[3];\nrx(4) q[0];\n\
         rx(4) q[1];\nrx(1) q[2];\nrx(1.5) q[3];",
    );
}

#[test]
fn duration_arithmetic_and_comparisons_fold() {
    assert_same_stream(
        "duration a = 300ns;\nduration b = 800dt;\n\
         rx((a + 2 * a - a / 2) / 1ns) q[0];\nrx(-a / 100ns) q[1];\nrx((b - 200dt) / b) q[2];\n\
         if (a < 1us) x q[3];\nif (a >= 301ns) z q[3];\nif (a == 0.3us) h q[0];\n\
         bool longer = 2us > a;\nif (longer) y q[1];\nif (a != 300ns) s q[2];",
        "rx(750) q[0];\nrx(-3) q[1];\nrx(0.75) q[2];\nx q[3];\nh q[0];\ny q[1];",
    );
}

// `dt` has no physical length here, so a sum keeps both parts and a ratio or
// comparison folds only when its answer holds for any length of `dt`.
#[test]
fn dt_and_si_units_mix_only_where_the_sample_period_cancels() {
    assert_same_stream(
        "duration d = 10ns + 2dt;\nrx(d / (5ns + 1dt)) q[0];\nif (d > 5ns) x q[1];\n\
         if (d - 2dt == 10ns) x q[2];",
        "rx(2) q[0];\nx q[1];\nx q[2];",
    );
    assert_declines("rx(10ns / 2dt) q[0];", "`dt`");
    assert_declines("if (10ns > 2dt) x q[0];", "`dt`");
    assert_declines("rx((10ns + 1dt) / 10ns) q[0];", "`dt`");
}

#[test]
fn durationof_is_the_longest_qubit_timeline_with_zero_length_gates() {
    assert_same_stream(
        "rx(durationof({ x q[0]; cx q[0], q[1]; }) / 1ns + 1) q[0];\n\
         rx(durationof({ delay[100ns] q[0]; delay[50ns] q[1]; }) / 1ns) q[1];\n\
         rx(durationof({ delay[100ns] q[0]; cx q[0], q[1]; delay[50ns] q[1]; }) / 1ns) q[2];\n\
         rx(durationof({ delay[100ns] q[0]; delay[50ns] q[0], q[1]; }) / 1ns) q[3];\n\
         rx(durationof({ delay[100ns] q[0]; h q; delay[50ns] q[1]; }) / 1ns) q[0];\n\
         rx(durationof({ delay[100ns] q[0]; barrier q; delay[50ns] q[1]; }) / 1ns) q[1];\n\
         rx(durationof({ box[200ns] { x q[0]; } delay[10ns] q[0]; }) / 1ns) q[2];\n\
         rx(durationof({ c[0] = measure q[0]; reset q[1]; nop q[2]; }) / 1ns + 2) q[3];",
        "rx(1) q[0];\nrx(100) q[1];\nrx(150) q[2];\nrx(150) q[3];\nrx(100) q[0];\n\
         rx(150) q[1];\nrx(210) q[2];\nrx(2) q[3];",
    );
}

#[test]
fn durationof_reads_its_block_without_running_it() {
    assert_same_stream(
        "duration d = durationof({ x q[0]; delay[40ns] q[0]; });\ndelay[d] q[1];\n\
         box[d] { h q[1]; }\nrx(d / 20ns) q[2];",
        "h q[1];\nrx(2) q[2];",
    );
    assert_same_stream(
        "def wait(qubit a) { delay[durationof({ x a; delay[8dt] a; })] a; h a; }\nwait(q[3]);",
        "h q[3];",
    );
}

#[test]
fn the_specification_examples_parse() {
    let text = "OPENQASM 3.0;\n\
        duration a = 300ns;\nduration b = 800dt;\nduration c = durationof({x $3;});\n\
        stretch d = a + 2 * c;\nstretch e = -0.5 * b + c;\n\
        stretch g;\nduration start_stretch = g - .5 * durationof({x $0;});\n\
        duration middle_stretch = g - .5 * durationof({x $0;}) - .5 * durationof({y $0;});\n\
        duration end_stretch = g - .5 * durationof({y $0;});\n\
        delay[start_stretch] $0;\nx $0;\ndelay[middle_stretch] $0;\ny $0;\n\
        delay[middle_stretch] $0;\nx $0;\ndelay[middle_stretch] $0;\ny $0;\n\
        delay[end_stretch] $0;\n";
    let circuit = openqasm::parse(text).expect("parse");
    // `durationof` reads `$3` without running it, so the program spans `$0` alone.
    let reference = openqasm::parse("OPENQASM 3.0;\nx $0;\ny $0;\nx $0;\ny $0;").expect("parse");
    assert_eq!(circuit.num_qubits, 1);
    assert_eq!(
        format!("{:?}", circuit.instructions),
        format!("{:?}", reference.instructions)
    );
}

#[test]
fn a_fixed_length_box_resolves_the_stretches_its_alignment_fixes() {
    assert_same_stream(
        "stretch s;\nbox[90ns] { delay[s] q[0]; x q[0]; delay[2 * s] q[0]; }\nrx(s / 1ns) q[1];",
        "x q[0];\nrx(30) q[1];",
    );
    assert_same_stream(
        "stretch a;\nbox[300ns] { delay[100ns] q[0]; delay[a] q[1]; cx q[0], q[1]; }\n\
         rx(a / 1ns) q[2];",
        "cx q[0], q[1];\nrx(100) q[2];",
    );
    assert_same_stream(
        "stretch t = 40ns;\nduration twice = 2 * t;\nbox[100ns] { delay[t] q[0]; }\n\
         rx(t / 1ns) q[1];\nrx(twice / 1ns) q[2];",
        "rx(100) q[1];\nrx(200) q[2];",
    );
    assert_same_stream(
        "stretch s;\nbox[100ns] { box[60ns] { delay[s] q[0]; } delay[40ns] q[0]; }\n\
         rx(s / 1ns) q[3];",
        "rx(60) q[3];",
    );
}

#[test]
fn a_stretch_nothing_fixes_still_declines_where_read() {
    assert_declines("stretch s;\ndelay[s] q[0];\nrx(s / 1ns) q[0];", "stretch");
    assert_declines(
        "stretch a;\nstretch b;\nbox[100ns] { delay[a] q[0]; delay[b] q[0]; }\nrx(a / 1ns) q[1];",
        "stretch",
    );
    assert_declines(
        "stretch a;\nbox[100ns] { delay[a] q[0]; delay[50ns] q[1]; delay[a] q[1]; }\n\
         rx(a / 1ns) q[1];",
        "stretch",
    );
    assert_declines("stretch s;\nif (s > 1ns) x q[0];", "stretch");
}

#[test]
fn durationof_declines_what_it_cannot_schedule() {
    assert_declines("rx(durationof({ int i = 1; }) / 1ns) q[0];", "`durationof`");
    assert_declines(
        "def f(qubit a) { x a; }\nrx(durationof({ f(q[0]); }) / 1ns) q[0];",
        "`durationof`",
    );
    assert_declines(
        "rx(durationof({ delay[10ns] q[0]; delay[3dt] q[1]; }) / 1ns) q[0];",
        "`dt`",
    );
    assert_parse_error("rx(durationof(x) / 1ns) q[0];", "`{`");
    assert_parse_error(
        "delay[durationof({ x q[0]; }) * 2ns] q[0];",
        "between a duration",
    );
}

#[test]
fn a_box_shorter_than_its_delays_is_rejected() {
    assert_parse_error(
        "box[10ns] { delay[100ns] q[0]; }",
        "longer than its duration",
    );
    assert_same_stream("box[100ns] { delay[100ns] q[0]; x q[1]; }", "x q[1];");
}

// Timing is the identity: the same program without its delays, boxes and timing
// declarations gives the same instructions and the same state.
#[test]
fn delay_and_box_leave_the_circuit_and_state_unchanged() {
    let timed = "const duration pulse = 40ns;\nstretch slack;\nh q[0];\n\
        delay[pulse] q[0];\ncx q[0], q[1];\ndelay[slack] q;\n\
        box[durationof({ delay[pulse] q[2]; })] { ry(0.4) q[2]; delay[slack] q[2]; }\n\
        box { cx q[1], q[2]; rz(pulse / 10ns) q[3]; }\ndelay[2 * pulse] q[1], q[3];\n\
        box[3 * pulse] { h q[3]; delay[pulse] q[3]; }";
    let plain = "h q[0];\ncx q[0], q[1];\nry(0.4) q[2];\ncx q[1], q[2];\nrz(4) q[3];\nh q[3];";
    assert_same_stream(timed, plain);
    let with_timing = simulate(&parse(timed)).seed(42).state_vector().unwrap();
    let without = simulate(&parse(plain)).seed(42).state_vector().unwrap();
    assert_eq!(with_timing.len(), without.len());
    for (a, b) in with_timing.iter().zip(&without) {
        assert!((a - b).norm() < 1e-12, "{a} vs {b}");
    }
}

#[test]
fn timing_folds_under_parse_dynamic() {
    let text = program(
        "qubit[2] q;\nbit[2] c;",
        "c[0] = measure q[0];\nduration d = durationof({ delay[30ns] q[1]; });\n\
         if (c[0]) { delay[d] q[1]; x q[1]; }\nrx(d / 10ns) q[0];",
    );
    openqasm::parse_dynamic(&text).expect("parse_dynamic");
}
