//! Dynamic programs: `while` loops and runtime classical values, parsed from
//! OpenQASM or built directly, run once per shot.

use std::collections::HashMap;

use prism_q::circuit::dynamic::{
    BinaryOp, ClassicalExpr, ClassicalType, DynamicProgramBuilder, RotationKind, Terminator,
};
use prism_q::circuit::openqasm;
use prism_q::circuit::{ClassicalCondition, Instruction, SmallVec};
use prism_q::{BackendKind, Circuit, DynamicProgram, Gate, PrismError, simulate, simulate_program};

const SEED: u64 = 42;

fn parse(source: &str) -> DynamicProgram {
    openqasm::parse_dynamic(source).unwrap_or_else(|err| panic!("{err}\n{source}"))
}

fn shots(program: &DynamicProgram, num_shots: usize) -> Vec<Vec<bool>> {
    simulate_program(program)
        .seed(SEED)
        .shots(num_shots)
        .expect("dynamic run")
        .shots
}

/// The value bits `range` hold in each shot, low bit first.
fn values(shots: &[Vec<bool>], range: std::ops::Range<usize>) -> Vec<u64> {
    shots
        .iter()
        .map(|shot| {
            shot[range.clone()]
                .iter()
                .rev()
                .fold(0, |acc, &bit| (acc << 1) | u64::from(bit))
        })
        .collect()
}

fn histogram(values: &[u64]) -> HashMap<u64, usize> {
    let mut counts = HashMap::new();
    for &value in values {
        *counts.entry(value).or_default() += 1;
    }
    counts
}

/// Assert each observed frequency sits within five standard errors of its
/// expected probability, and nothing lands outside the support.
fn assert_distribution(counts: &HashMap<u64, usize>, expected: &[(u64, f64)], shots: usize) {
    for (value, count) in counts {
        assert!(
            expected.iter().any(|(v, _)| v == value),
            "value {value} seen {count} times outside the support {expected:?}"
        );
    }
    for &(value, p) in expected {
        let seen = *counts.get(&value).unwrap_or(&0) as f64 / shots as f64;
        let sigma = (p * (1.0 - p) / shots as f64).sqrt();
        assert!(
            (seen - p).abs() <= 5.0 * sigma + 1e-12,
            "value {value}: frequency {seen} against {p} (sigma {sigma})"
        );
    }
}

/// Write the low `width` bits of `name` onto qubits `first..first + width` and
/// measure them into the bits of the same index.
fn readout(name: &str, first: usize, width: usize) -> String {
    let mut out = String::new();
    for bit in 0..width {
        let q = first + bit;
        out.push_str(&format!(
            "if ((({name} >> {bit}) & 1) == 1) {{ x q[{q}]; }}\nc[{q}] = measure q[{q}];\n"
        ));
    }
    out
}

// Prepare |+> and measure until the outcome is 0: the trial count is geometric
// with p = 1/2, so its mean is 2 and its variance 2.
#[test]
fn repeat_until_success_has_the_geometric_trial_count() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[5] q;
        bit[5] c;
        uint[4] tries = 1;
        h q[0];
        c[0] = measure q[0];
        while (c[0]) {{
          reset q[0];
          h q[0];
          c[0] = measure q[0];
          tries += 1;
        }}
        {}",
        readout("tries", 1, 4)
    );
    let program = parse(&source);
    let num_shots = 8000;
    let runs = shots(&program, num_shots);
    assert!(runs.iter().all(|shot| !shot[0]), "every shot ends on 0");
    let tries = values(&runs, 1..5);
    let mean = tries.iter().sum::<u64>() as f64 / num_shots as f64;
    let standard_error = (2.0 / num_shots as f64).sqrt();
    assert!(
        (mean - 2.0).abs() < 5.0 * standard_error,
        "mean trial count {mean}, expected 2"
    );
    let geometric: Vec<(u64, f64)> = (1..16).map(|k| (k, 0.5f64.powi(k as i32))).collect();
    let counts = histogram(&tries);
    for &(k, p) in &geometric[..4] {
        let seen = *counts.get(&k).unwrap_or(&0) as f64 / num_shots as f64;
        let sigma = (p * (1.0 - p) / num_shots as f64).sqrt();
        assert!((seen - p).abs() < 5.0 * sigma, "P(tries = {k}) = {seen}");
    }
}

// Up to four fair trials, counting the ones, stopping at the second: `continue`
// skips the count on a 0 and `break` leaves at the second 1. The joint law of
// (trials run, ones counted) is
//   (2, 2): 1/4, (3, 2): 1/4, (4, 2): 3/16, (4, 1): 4/16, (4, 0): 1/16.
#[test]
fn a_counter_loop_with_break_and_continue_matches_its_law() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[6] q;
        bit[6] c;
        uint[4] n = 0;
        uint[4] ones = 0;
        while (n < 4) {{
          n += 1;
          reset q[0];
          h q[0];
          c[0] = measure q[0];
          if (!c[0]) continue;
          ones += 1;
          if (ones == 2) {{ break; }}
        }}
        {}{}",
        readout("n", 1, 3),
        readout("ones", 4, 2)
    );
    let program = parse(&source);
    let num_shots = 8000;
    let runs = shots(&program, num_shots);
    let joint: Vec<u64> = values(&runs, 1..4)
        .into_iter()
        .zip(values(&runs, 4..6))
        .map(|(n, ones)| n * 10 + ones)
        .collect();
    assert_distribution(
        &histogram(&joint),
        &[
            (22, 0.25),
            (32, 0.25),
            (42, 3.0 / 16.0),
            (41, 4.0 / 16.0),
            (40, 1.0 / 16.0),
        ],
        num_shots,
    );
}

#[test]
fn break_leaves_only_the_innermost_loop() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[4] q;
        bit[4] c;
        uint[4] outer = 0;
        uint[4] total = 0;
        while (outer < 3) {{
          outer += 1;
          uint[4] inner = 0;
          while (true) {{
            inner += 1;
            total += 1;
            if (inner == 2) break;
          }}
        }}
        {}",
        readout("total", 0, 4)
    );
    let runs = shots(&parse(&source), 4);
    assert!(values(&runs, 0..4).iter().all(|&total| total == 6));
}

#[test]
fn a_measured_break_leaves_an_unrolled_for() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        for int i in [0:2] {
          h q[i];
          c[i] = measure q[i];
          if (c[i]) break;
        }",
    );
    let num_shots = 8000;
    let runs = shots(&program, num_shots);
    assert_distribution(
        &histogram(&values(&runs, 0..3)),
        &[(1, 0.5), (2, 0.25), (4, 0.125), (0, 0.125)],
        num_shots,
    );
}

#[test]
fn a_measured_continue_skips_the_rest_of_its_pass() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[3] f;
        bit[3] c;
        x q[1];
        for int i in [0:2] {
          h q[0];
          f[i] = measure q[0];
          reset q[0];
          if (f[i]) { continue; }
          c[i] = measure q[1];
        }",
    );
    let runs = shots(&program, 400);
    let flags = values(&runs, 0..3);
    assert!(flags.iter().any(|&f| f != 0) && flags.iter().any(|&f| f != 7));
    for (f, c) in flags.into_iter().zip(values(&runs, 3..6)) {
        assert_eq!(c, !f & 7, "a pass that continued measured nothing");
    }
}

#[test]
fn a_measured_break_in_a_nested_for_leaves_only_the_inner_one() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[6] q;
        bit[6] c;
        uint[4] outer = 0;
        uint[4] passes = 0;
        for int i in [0:1] {{
          outer += 1;
          for int j in [0:2] {{
            passes += 1;
            reset q[0];
            h q[0];
            c[0] = measure q[0];
            if (c[0]) break;
          }}
        }}
        {}{}",
        readout("outer", 1, 2),
        readout("passes", 3, 3)
    );
    let num_shots = 8000;
    let runs = shots(&parse(&source), num_shots);
    assert!(values(&runs, 1..3).iter().all(|&outer| outer == 2));
    // Each inner loop runs one to three passes, with probabilities 1/2, 1/4, 1/4.
    let one = [(1, 0.5), (2, 0.25), (3, 0.25)];
    let mut expected: Vec<(u64, f64)> = (2..=6).map(|total| (total, 0.0)).collect();
    for (a, pa) in one {
        for (b, pb) in one {
            expected[(a + b - 2) as usize].1 += pa * pb;
        }
    }
    assert_distribution(&histogram(&values(&runs, 3..6)), &expected, num_shots);
}

#[test]
fn a_measured_end_stops_the_shot() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[4] q;
        bit[4] c;
        for int i in [0:2] {
          h q[i];
          c[i] = measure q[i];
          if (c[i]) end;
        }
        x q[3];
        c[3] = measure q[3];",
    );
    let runs = shots(&program, 400);
    for value in values(&runs, 0..4) {
        let reached_the_end = value & 7 == 0;
        assert_eq!(value & 8 != 0, reached_the_end, "{value:04b}");
    }
}

#[test]
fn a_parse_time_break_after_a_measured_continue_still_builds_the_next_pass() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        for int i in [0:1] {
          c[0] = measure q[0];
          if (c[0]) { x q[0]; continue; }
          x q[1];
          break;
        }
        c[1] = measure q[1];",
    );
    assert!(
        values(&shots(&program, 64), 0..2)
            .iter()
            .all(|&value| value == 2)
    );
}

#[test]
fn a_parse_time_return_after_a_measured_continue_still_builds_the_next_pass() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        def f(qubit b, bit r) {
          for int i in [0:1] {
            if (r && i == 0) { continue; }
            x b;
            return;
          }
        }
        h q[0];
        c[0] = measure q[0];
        f(q[1], c[0]);
        c[1] = measure q[1];",
    );
    let counts = histogram(&values(&shots(&program, 400), 0..2));
    assert!(counts.keys().all(|&value| value >= 2), "{counts:?}");
    assert_eq!(counts.len(), 2, "{counts:?}");
}

#[test]
fn end_in_a_def_called_under_a_measured_if_stops_the_shot() {
    let source = "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        def stop() { end; }
        h q[0];
        c[0] = measure q[0];
        if (c[0]) { stop(); }
        x q[1];
        c[1] = measure q[1];";
    let counts = histogram(&values(&shots(&parse(source), 400), 0..2));
    assert!(
        counts.keys().all(|&value| value == 1 || value == 2),
        "{counts:?}"
    );
    assert_eq!(counts.len(), 2, "{counts:?}");
    match openqasm::parse(source) {
        Err(PrismError::UnsupportedConstruct { construct, .. }) => {
            assert!(construct.contains("parse_dynamic"), "{construct}")
        }
        other => panic!("`parse` should decline, got {other:?}"),
    }
}

#[test]
fn a_switch_on_a_parse_time_target_takes_its_arm_like_an_if() {
    let program = |select: &str| {
        format!(
            "OPENQASM 3.0;
            qubit[2] q;
            bit[2] r;
            x q[1];
            h q[0];
            r[0] = measure q[0];
            for int i in [0:1] {{ {select} }}
            r[1] = measure q[1];"
        )
    };
    let switched = shots(
        &parse(&program(
            "switch (i) { case 0 { if (r[0]) { end; } break; } default { } }",
        )),
        400,
    );
    let branched = shots(
        &parse(&program("if (i == 0) { if (r[0]) { end; } break; }")),
        400,
    );
    assert_eq!(switched, branched);
    let counts = histogram(&values(&switched, 0..2));
    assert!(
        counts.keys().all(|&value| value == 1 || value == 2),
        "{counts:?}"
    );
    assert_eq!(counts.len(), 2, "{counts:?}");
}

#[test]
fn a_parse_time_return_after_a_measured_break_keeps_what_follows_the_loop() {
    let head = "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;";
    let tail = "h q[0];
        c[0] = measure q[0];
        f(q[1], c[0]);
        c[1] = measure q[1];";
    let one_deep = "def f(qubit a, bit b) {
          for int i in [0:2] { if (b) { break; } return; }
          x a;
        }";
    let two_deep = "def f(qubit a, bit b) {
          for int i in [0:1] { for int j in [0:1] { if (b) { break; } return; } }
          x a;
        }";
    let with_result = "def g(bit b) -> uint[8] {
          uint[8] k = 0;
          for int i in [0:2] { k += 1; if (b) { break; } return k + 10; }
          return k + 20;
        }
        def f(qubit a, bit b) {
          uint[8] r = g(b);
          if (r == 21) { x a; } else if (r != 11) { x a; x a; x a; }
        }";
    for def in [one_deep, two_deep, with_result] {
        let source = format!("{head}\n{def}\n{tail}");
        let counts = histogram(&values(&shots(&parse(&source), 400), 0..2));
        assert!(
            counts.keys().all(|&value| value == 0 || value == 3),
            "{counts:?}\n{source}"
        );
        assert_eq!(counts.len(), 2, "{counts:?}\n{source}");
        assert!(
            matches!(
                openqasm::parse(&source),
                Err(PrismError::UnsupportedConstruct { .. })
            ),
            "{source}"
        );
    }
}

#[test]
fn an_end_every_path_reaches_in_a_def_stops_the_parse() {
    let recursion = "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        def r(int[8] d, qubit a) { if (d == 0) { end; } x a; r(d - 1, a); }
        x q[0];
        c[0] = measure q[0];
        r(2, q[1]);
        c[1] = measure q[1];";
    assert_eq!(openqasm::parse(recursion).unwrap().instructions.len(), 4);
    assert!(
        values(&shots(&parse(recursion), 64), 0..2)
            .iter()
            .all(|&value| value == 1)
    );
    // Statements after the `end` are not examined, so a name the def cannot
    // see goes unreported on both paths.
    let after_end = "OPENQASM 3.0;
        qubit[2] q;
        def stop() { end; }
        def g() { stop(); x q[1]; }
        x q[0];
        g();";
    assert_eq!(openqasm::parse(after_end).unwrap().instructions.len(), 1);
    parse(after_end);
    // Paths a measured break sent past the call still run what follows it.
    let diverted = "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        def stop() { end; }
        h q[0];
        c[0] = measure q[0];
        for int i in [0:1] { if (c[0]) { break; } stop(); }
        x q[1];
        c[1] = measure q[1];";
    let counts = histogram(&values(&shots(&parse(diverted), 400), 0..2));
    assert!(
        counts.keys().all(|&value| value == 0 || value == 3),
        "{counts:?}"
    );
    assert_eq!(counts.len(), 2, "{counts:?}");
}

#[test]
fn end_in_a_gate_body_is_a_parse_error_where_the_gate_is_defined() {
    let head = "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        gate g a { x a; for int i in [0:1] { if (i == 1) { end; } } }";
    for tail in ["", "g q[1];", "c[0] = measure q[0]; if (c[0]) { g q[1]; }"] {
        let source = format!("{head}\n{tail}");
        for result in [
            openqasm::parse(&source).map(drop),
            openqasm::parse_dynamic(&source).map(drop),
        ] {
            match result {
                Err(PrismError::Parse { message, .. }) => {
                    assert!(message.contains("gate `g`"), "{message}")
                }
                other => panic!("expected a parse error, got {other:?}\n{source}"),
            }
        }
    }
}

#[test]
fn a_variable_a_measured_break_cuts_short_is_runtime() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        uint[2] n = 0;
        for int i in [0:2] {{
          reset q[0];
          h q[0];
          c[0] = measure q[0];
          if (c[0]) break;
          n += 1;
        }}
        {}",
        readout("n", 1, 2)
    );
    let num_shots = 8000;
    let runs = shots(&parse(&source), num_shots);
    assert_distribution(
        &histogram(&values(&runs, 1..3)),
        &[(0, 0.5), (1, 0.25), (2, 0.125), (3, 0.125)],
        num_shots,
    );
}

#[test]
fn a_parse_time_exit_keeps_one_block() {
    let program = parse(
        "OPENQASM 3.0; qubit[4] q; bit[4] c;
         for int i in [0:3] { if (i == 2) break; x q[i]; }
         c = measure q;
         end;
         h q[0];",
    );
    assert_eq!(program.blocks().len(), 1);
    let runs = shots(&program, 4);
    assert!(values(&runs, 0..4).iter().all(|&value| value == 3));
}

#[test]
fn a_loop_exit_outside_a_loop_is_a_parse_error() {
    for body in [
        "break;",
        "def f() { continue; } while (true) { f(); }",
        "bit b; b = measure q[0]; if (b) break;",
    ] {
        let source = format!("OPENQASM 3.0; qubit[1] q; {body}");
        match openqasm::parse_dynamic(&source) {
            Err(PrismError::Parse { message, .. }) => {
                assert!(message.contains("outside a loop"), "`{body}`: {message}")
            }
            other => panic!("`{body}` should be a parse error, got {other:?}"),
        }
    }
}

#[test]
fn a_runaway_loop_stops_at_the_step_bound_naming_the_loop() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[1] q;
        bit[1] c;
        h q[0];
        while (true) { x q[0]; }",
    );
    let err = simulate_program(&program)
        .max_steps(500)
        .seed(SEED)
        .shots(3)
        .unwrap_err();
    match &err {
        PrismError::StepLimit { region, max_steps } => {
            assert_eq!(region, "loop `while at line 5`");
            assert_eq!(*max_steps, 500);
        }
        other => panic!("expected a step limit, got {other:?}"),
    }
    assert!(err.to_string().contains("max_steps"), "{err}");

    let default_bound = simulate_program(&program).seed(SEED).shots(1);
    assert!(matches!(
        default_bound,
        Err(PrismError::StepLimit {
            max_steps: 1_000_000,
            ..
        })
    ));
}

#[test]
fn a_measured_write_under_an_if_is_evaluated_at_runtime() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        int n = 0;
        h q[0];
        c[0] = measure q[0];
        if (c[0]) { n = 1; }
        if (n == 1) { x q[1]; }
        c[1] = measure q[1];",
    );
    let runs = shots(&program, 400);
    assert!(runs.iter().all(|shot| shot[0] == shot[1]));
    assert!(runs.iter().any(|shot| shot[0]) && runs.iter().any(|shot| !shot[0]));
}

#[test]
fn a_runtime_angle_drives_the_rotation() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        float t = 0;
        x q[1];
        c[1] = measure q[1];
        if (c[1]) t = pi;
        rx(t) q[0];
        c[0] = measure q[0];",
    );
    assert!(shots(&program, 50).iter().all(|shot| shot[0]));
}

#[test]
fn integer_variables_wrap_at_their_declared_width() {
    let source = format!(
        "OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        uint[2] n = 0;
        int[3] m = 3;
        c[0] = measure q[0];
        while (c[0] == 0 && m != -4) {{
          n += 1;
          m += 1;
        }}
        {}",
        readout("n", 1, 2)
    );
    let runs = shots(&parse(&source), 4);
    assert!(values(&runs, 1..3).iter().all(|&n| n == 1));
}

#[test]
fn a_switch_over_a_runtime_value_branches() {
    let source = "OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        uint[2] k = 0;
        h q[0];
        c[0] = measure q[0];
        if (c[0]) k = 2;
        switch (k) {
          case 0 { x q[1]; }
          case 1, 2 { x q[2]; }
          default { }
        }
        c[1] = measure q[1];
        c[2] = measure q[2];";
    let runs = shots(&parse(source), 400);
    assert!(
        runs.iter()
            .all(|shot| shot[1] != shot[0] && shot[2] == shot[0])
    );
}

#[test]
fn a_runtime_integer_reads_one_bit_per_index() {
    let source = "OPENQASM 3.0;
        qubit[4] q;
        bit[4] c;
        uint[2] k = 0;
        h q[0];
        x q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        k = c;
        if (k[0] == 1) x q[2];
        if (bool(k[-1])) x q[3];
        c[2] = measure q[2];
        c[3] = measure q[3];";
    let runs = shots(&parse(source), 200);
    assert!(runs.iter().all(|shot| shot[2] == shot[0] && shot[3]));
    assert!(runs.iter().any(|shot| shot[0]) && runs.iter().any(|shot| !shot[0]));
}

#[test]
fn a_switch_over_a_signed_cast_compares_the_signed_value() {
    let source = "OPENQASM 3.0;
        qubit[3] q;
        bit[2] c;
        bit[1] d;
        h q[0];
        x q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
        switch (int[2](c)) {
          case -2 { x q[2]; }
          case -1 { }
          default { x q[2]; x q[2]; x q[2]; }
        }
        d[0] = measure q[2];";
    let runs = shots(&parse(source), 200);
    assert!(runs.iter().all(|shot| shot[2] != shot[0]));
}

#[test]
fn an_else_may_follow_an_if_that_measures_its_own_condition() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[2] q;
        bit[2] c;
        x q[0];
        c[0] = measure q[0];
        if (c[0]) { reset q[0]; c[0] = measure q[0]; } else { x q[1]; }
        c[1] = measure q[1];",
    );
    assert!(
        openqasm::parse(
            "OPENQASM 3.0; qubit[2] q; bit[2] c; x q[0]; c[0] = measure q[0];
             if (c[0]) { reset q[0]; c[0] = measure q[0]; } else { x q[1]; }"
        )
        .is_err()
    );
    assert!(shots(&program, 50).iter().all(|shot| !shot[0] && !shot[1]));
}

// A program `parse` accepts lowers to one block holding the circuit `parse`
// returns, and runs through the same route with the same seeded shots.
#[test]
fn a_static_program_parses_and_runs_as_its_circuit() {
    for source in [
        "OPENQASM 3.0; qubit[3] q; bit[3] c; h q[0]; cx q[0], q[1]; c = measure q;",
        "OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0];
         if (c[0]) { x q[1]; } else { h q[1]; } c[1] = measure q[1];",
        "OPENQASM 3.0; qubit[4] q; bit[4] c; int n = 0;
         for int i in [0:3] { n += i; rx(n * 0.1) q[i]; }
         if (c[0]) { int k = 1; k += 1; h q[k]; }
         c = measure q;",
        "OPENQASM 3.0; qubit[2] q; bit[2] c; h q[0]; c[0] = measure q[0];
         switch (c) { case 1 { x q[1]; } default { z q[1]; } } c[1] = measure q[1];",
    ] {
        let circuit = openqasm::parse(source).unwrap();
        let program = parse(source);
        let lowered = program.static_circuit().expect("no runtime control flow");
        assert_eq!(
            format!("{:?}", lowered.instructions),
            format!("{:?}", circuit.instructions)
        );
        assert_eq!(lowered.num_qubits, circuit.num_qubits);
        assert_eq!(lowered.num_classical_bits, circuit.num_classical_bits);
        let plain = simulate(&circuit).seed(SEED).shots(64).unwrap();
        let dynamic = simulate_program(&program).seed(SEED).shots(64).unwrap();
        assert_eq!(plain.shots, dynamic.shots, "{source}");
        assert_eq!(plain.metadata.backend, dynamic.metadata.backend);
    }
}

fn entangled_with_midcircuit_measurement() -> Circuit {
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Ry(0.9), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 2]);
    circuit.add_gate(Gate::Cx, &[1, 3]);
    circuit.add_gate(Gate::Cx, &[2, 3]);
    circuit.add_measure(2, 2);
    circuit.add_gate(Gate::Rx(0.4), &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_measure(1, 1);
    circuit.add_measure(3, 3);
    circuit
}

// Split across blocks by an action, the same instructions branch on the same
// outcome draws on the host statevector, so the shots match the circuit's bit
// for bit.
#[test]
fn a_program_split_into_blocks_reproduces_the_seeded_circuit_shots() {
    let circuit = entangled_with_midcircuit_measurement();
    let (head, tail) = circuit.instructions.split_at(6);
    let mut b = DynamicProgramBuilder::new(4, 4);
    let n = b.declare("n", ClassicalType::Int { width: 8 }, 0i64.into());
    for instruction in head {
        b.add_instruction(instruction.clone());
    }
    b.assign(n, ClassicalExpr::from(1i64));
    for instruction in tail {
        b.add_instruction(instruction.clone());
    }
    let program = b.build().unwrap();
    assert_eq!(program.blocks().len(), 2);
    assert!(program.static_circuit().is_none());
    let plain = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(300)
        .unwrap();
    let dynamic = simulate_program(&program)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(300)
        .unwrap();
    assert_eq!(plain.shots, dynamic.shots);
}

// Auto samples a terminal-measurement circuit from one evolved distribution,
// which draws its randomness differently from a per-shot walk, so the
// agreement is in distribution rather than shot by shot.
#[test]
fn a_split_terminal_program_matches_the_circuit_distribution() {
    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Ry(1.1), &[1]);
    circuit.add_gate(Gate::Cx, &[0, 2]);
    circuit.add_gate(Gate::T, &[2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[2]);
    let mut b = DynamicProgramBuilder::new(3, 3);
    let n = b.declare("n", ClassicalType::Bool, false.into());
    b.append(&circuit);
    b.assign(n, true.into());
    for q in 0..3 {
        b.add_measure(q, q);
    }
    let program = b.build().unwrap();
    let probabilities = simulate(&circuit)
        .seed(SEED)
        .run()
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();
    circuit.measure_all();
    let num_shots = 20_000;
    let expected: Vec<(u64, f64)> = (0..8u64)
        .map(|index| (index, probabilities[index as usize]))
        .filter(|&(_, p)| p > 1e-12)
        .collect();
    let runs = shots(&program, num_shots);
    assert_distribution(&histogram(&values(&runs, 0..3)), &expected, num_shots);
    let plain = simulate(&circuit).seed(SEED).shots(num_shots).unwrap();
    assert_distribution(
        &histogram(&values(&plain.shots, 0..3)),
        &expected,
        num_shots,
    );
}

fn rus_program(num_qubits: usize) -> DynamicProgram {
    let mut b = DynamicProgramBuilder::new(num_qubits, 1);
    for q in 0..num_qubits {
        b.add_gate(Gate::H, &[q]);
    }
    for q in 1..num_qubits {
        b.add_gate(Gate::Cx, &[q - 1, q]);
    }
    b.add_measure(0, 0);
    b.begin_while("retry", ClassicalExpr::Bit(0));
    b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.end().unwrap();
    b.build().unwrap()
}

#[test]
fn every_per_shot_backend_runs_a_loop() {
    let program = rus_program(3);
    for kind in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::Factored,
        BackendKind::FactoredStabilizer,
        BackendKind::TensorNetwork,
        BackendKind::DensityMatrix,
    ] {
        let label = format!("{kind:?}");
        let runs = simulate_program(&program)
            .backend(kind)
            .seed(SEED)
            .shots(40)
            .unwrap_or_else(|err| panic!("{label}: {err}"));
        assert!(runs.shots.iter().all(|shot| !shot[0]), "{label}");
    }
}

#[test]
fn every_per_shot_backend_writes_classical_bits() {
    let mut b = DynamicProgramBuilder::new(2, 5);
    b.add_gate(Gate::H, &[0]).add_measure(0, 0);
    b.write_bits(
        &[1, 2],
        ClassicalExpr::binary(
            BinaryOp::Add,
            ClassicalExpr::Bit(0),
            ClassicalExpr::from(2i64),
        ),
    );
    b.begin_if(ClassicalExpr::Bit(1));
    b.add_gate(Gate::X, &[1]);
    b.end().unwrap();
    b.add_instruction(Instruction::Conditional {
        condition: ClassicalCondition::BitIsOne(2),
        gate: Gate::X,
        targets: SmallVec::from_slice(&[0]),
    });
    b.add_measure(0, 3).add_measure(1, 4);
    let program = b.build().unwrap();
    for kind in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 8 },
        BackendKind::Factored,
        BackendKind::FactoredStabilizer,
        BackendKind::TensorNetwork,
        BackendKind::DensityMatrix,
        BackendKind::ProductState,
    ] {
        let label = format!("{kind:?}");
        let runs = simulate_program(&program)
            .backend(kind)
            .seed(SEED)
            .shots(40)
            .unwrap_or_else(|err| panic!("{label}: {err}"));
        assert!(
            runs.shots.iter().all(|shot| shot[1] == shot[0]
                && shot[2]
                && shot[3] != shot[0]
                && shot[4] == shot[0]),
            "{label}"
        );
        assert!(runs.shots.iter().any(|shot| shot[0]), "{label}");
    }
}

#[test]
fn a_write_of_a_float_is_rejected_when_the_program_is_built() {
    let mut b = DynamicProgramBuilder::new(1, 1);
    b.write_bits(&[0], ClassicalExpr::from(0.5));
    let err = b.build().unwrap_err();
    assert!(err.to_string().contains("integer or a bool"), "{err}");
    let mut b = DynamicProgramBuilder::new(1, 1);
    b.write_bits(&[1], ClassicalExpr::from(1i64));
    assert!(matches!(
        b.build().unwrap_err(),
        PrismError::InvalidClassicalBit { index: 1, .. }
    ));
}

#[test]
fn a_bit_initializer_writes_its_bits() {
    let source = "OPENQASM 3.0;
        qubit[1] q;
        bit[3] f = \"110\";
        bit[3] g = 5;
        bit[2] z = \"00\";
        bit[1] h;
        x q[0];
        h[0] = measure q[0];
        bit[2] r = h[0] + 2 * h[0];";
    let runs = shots(&parse(source), 4);
    for shot in &runs {
        assert_eq!(
            shot,
            &[
                false, true, true, true, false, true, false, false, true, true, true
            ]
        );
    }
}

#[test]
fn an_assignment_to_classical_bits_writes_them() {
    let source = "OPENQASM 3.0;
        qubit[2] q;
        bit[4] c;
        x q[0];
        c[0] = measure q[0];
        c <<= 1;
        c[3] = c[1] ^ c[0];
        c[2] ^= 1;";
    let runs = shots(&parse(source), 4);
    for shot in &runs {
        assert_eq!(shot, &[false, true, true, true]);
    }
}

#[test]
fn a_write_under_a_measured_if_branches() {
    let source = "OPENQASM 3.0;
        qubit[1] q;
        bit[3] c;
        h q[0];
        c[0] = measure q[0];
        if (c[0]) { c[1] = 1; } else { c[2] = true; }";
    let runs = shots(&parse(source), 200);
    assert!(
        runs.iter()
            .all(|shot| shot[1] == shot[0] && shot[2] != shot[0])
    );
    assert!(runs.iter().any(|shot| shot[0]) && runs.iter().any(|shot| !shot[0]));
}

#[test]
fn a_constant_write_wider_than_a_value_spans_several_writes() {
    let source = "OPENQASM 3.0;
        qubit[1] q;
        bit[70] w;
        w = -1;
        w[69] = 0;";
    let runs = shots(&parse(source), 2);
    for shot in &runs {
        assert!(shot[..69].iter().all(|&bit| bit) && !shot[69]);
    }
}

#[test]
fn a_classical_bit_write_needs_the_dynamic_parse() {
    for body in [
        "bit[2] f = \"01\";",
        "bit[2] c; c = 2;",
        "bit[2] c; c[0] = 1;",
    ] {
        let source = format!("OPENQASM 3.0;\nqubit[1] q;\n{body}");
        let err = openqasm::parse(&source).unwrap_err();
        assert!(
            matches!(&err, PrismError::UnsupportedConstruct { construct, .. }
                if construct.contains("parse_dynamic")),
            "{body}: {err}"
        );
    }
    let err = openqasm::parse_dynamic("OPENQASM 3.0;\nbit[2] c;\nc = 0.5;").unwrap_err();
    assert!(
        err.to_string().contains("takes an integer or a bool"),
        "{err}"
    );
}

#[test]
fn engines_without_a_per_shot_state_decline() {
    let program = rus_program(2);
    for kind in [
        BackendKind::StabilizerRank,
        BackendKind::StochasticPauli { num_samples: 10 },
        BackendKind::PauliPath {
            epsilon: 0.0,
            max_terms: 0,
        },
    ] {
        let label = format!("{kind:?}");
        let err = simulate_program(&program)
            .backend(kind)
            .seed(SEED)
            .shots(4)
            .expect_err(&label);
        assert!(
            matches!(err, PrismError::IncompatibleBackend { .. }),
            "{label}: {err:?}"
        );
    }
}

#[test]
fn a_runtime_rotation_keeps_auto_off_the_tableau() {
    let mut b = DynamicProgramBuilder::new(2, 2);
    let t = b.declare("t", ClassicalType::Float { width: 64 }, 0.3f64.into());
    b.add_gate(Gate::H, &[0]).add_gate(Gate::Cx, &[0, 1]);
    b.add_measure(0, 0);
    b.add_rotation(RotationKind::Ry, &[1], t.into());
    b.add_measure(1, 1);
    let program = b.build().unwrap();
    let runs = simulate_program(&program).seed(SEED).shots(10).unwrap();
    assert_ne!(runs.metadata.backend, prism_q::ResolvedBackend::Stabilizer);
    let explicit = simulate_program(&program)
        .backend(BackendKind::Stabilizer)
        .seed(SEED)
        .shots(10);
    assert!(matches!(
        explicit,
        Err(PrismError::IncompatibleBackend { .. })
    ));
}

#[test]
fn seeded_runs_repeat_and_counts_agree_with_shots() {
    let program = rus_program(4);
    let first = simulate_program(&program).seed(9).shots(500).unwrap();
    let second = simulate_program(&program).seed(9).shots(500).unwrap();
    assert_eq!(first.shots, second.shots);
    let counts = simulate_program(&program)
        .seed(9)
        .sample_counts(500)
        .unwrap();
    assert_eq!(counts.counts, first.counts());
}

#[test]
fn parse_still_declines_loops_and_points_at_parse_dynamic() {
    let err = openqasm::parse("OPENQASM 3.0;\nqubit[1] q;\nwhile (true) { x q[0]; }").unwrap_err();
    assert!(matches!(
        &err,
        PrismError::UnsupportedConstruct { construct, line: 3 } if construct == "while"
    ));
    assert!(err.to_string().contains("parse_dynamic"), "{err}");
    let guarded =
        openqasm::parse("OPENQASM 3.0; qubit[1] q; bit[1] c; int n = 0; if (c[0]) { n = 1; }")
            .unwrap_err();
    assert!(guarded.to_string().contains("parse_dynamic"), "{guarded}");
}

#[test]
fn parse_dynamic_declines_what_has_no_runtime_form() {
    let prologue = "OPENQASM 3.0;\nqubit[2] q;\nbit[2] c;\nint n = 0;\nc[0] = measure q[0];\nif (c[0]) n = 1;\n";
    for (body, needle) in [
        ("h q[n];", "parse-time constant"),
        ("float t = sin(n);", "builtin `sin`"),
        ("n = measure q[1];", "measurement into variable"),
        (
            "while (n < 2) { bit[1] d; n += 1; }",
            "inside a `while` loop",
        ),
        ("u3(n, 0, 0) q[1];", "runtime angle on `u3`"),
        ("array[int, 2] a; if (c[1]) { a[0] = 1; }", "array `a`"),
        ("const int k = n;", "parse-time constant"),
    ] {
        let source = format!("{prologue}{body}");
        match openqasm::parse_dynamic(&source) {
            Err(PrismError::UnsupportedConstruct { construct, .. }) => {
                assert!(construct.contains(needle), "`{body}`: {construct}")
            }
            other => panic!("`{body}` should decline, got {other:?}"),
        }
    }
    assert!(openqasm::parse_dynamic("OPENQASM 3.0; input float t; qubit q; rx(t) q;").is_err());
}

#[test]
fn a_while_loop_lowers_to_a_tested_header() {
    let program = parse(
        "OPENQASM 3.0; qubit q; bit c; h q; c = measure q;
         while (c) { reset q; h q; c = measure q; }",
    );
    let headers: Vec<_> = program
        .blocks()
        .iter()
        .filter(|block| matches!(block.terminator, Terminator::Branch { .. }))
        .collect();
    assert_eq!(headers.len(), 1);
    assert!(headers[0].circuit.instructions.is_empty());
    assert_eq!(headers[0].loop_name.as_deref(), Some("while at line 2"));
    assert_eq!(program.blocks().len(), 4);
}

#[test]
fn compound_and_logical_conditions_parse_with_their_precedence() {
    let program = parse(
        "OPENQASM 3.0;
        qubit[3] q;
        bit[3] c;
        x q[0];
        c[0] = measure q[0];
        c[1] = measure q[1];
        if (c[0] == 1 && c[1] == 0) { x q[2]; }
        c[2] = measure q[2];",
    );
    assert!(shots(&program, 8).iter().all(|shot| shot[2]));
    let mut b = DynamicProgramBuilder::new(1, 1);
    let n = b.declare("n", ClassicalType::Uint { width: 8 }, 3i64.into());
    let expr = b.expr("n + 1 << 2 == 16 || false").unwrap();
    assert!(matches!(
        expr,
        ClassicalExpr::Binary {
            op: BinaryOp::Or,
            ..
        }
    ));
    b.begin_if(expr);
    b.add_gate(Gate::X, &[0]);
    b.end().unwrap();
    b.add_measure(0, 0);
    let _ = n;
    let program = b.build().unwrap();
    assert!(shots(&program, 4).iter().all(|shot| shot[0]));
}
