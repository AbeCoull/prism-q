//! Dynamic programs: `while` loops and runtime classical values, parsed from
//! OpenQASM or built directly, run once per shot.

mod common;

use std::collections::HashMap;

use common::mix_seed;
use prism_q::circuit::dynamic::{
    BinaryOp, ClassicalExpr, ClassicalType, DynamicProgramBuilder, RotationKind, Terminator,
};
use prism_q::circuit::openqasm;
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

// Split across blocks by an action, the same instructions run on the same
// backend from the same per-shot seeds, so each shot matches a run of the
// circuit on its seed bit for bit.
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
    let plain: Vec<Vec<bool>> = (0..300)
        .map(|i| {
            simulate(&circuit)
                .backend(BackendKind::Statevector)
                .seed(mix_seed(SEED, i))
                .run()
                .unwrap()
                .classical_bits
        })
        .collect();
    let dynamic = simulate_program(&program)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(300)
        .unwrap();
    assert_eq!(plain, dynamic.shots);
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
    let t = b.declare("t", ClassicalType::Float, 0.3f64.into());
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
        ("for int i in [0:1] { break; }", "`for` loop"),
        ("break;", "outside a loop"),
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
