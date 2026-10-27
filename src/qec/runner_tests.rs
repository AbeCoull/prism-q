use super::*;

fn parse_dropping_records(text: &str) -> QecProgram {
    let mut program = QecProgram::from_text(text).unwrap();
    program.set_options(QecOptions {
        keep_measurements: false,
        ..QecOptions::default()
    });
    program
}

#[test]
fn records_stay_on_the_old_path_when_kept_or_postselected() {
    let mut program = parse_dropping_records("M 0\nDETECTOR rec[-1]");
    assert!(qec_records_unobserved(&program));

    program.set_options(QecOptions::default());
    assert!(!qec_records_unobserved(&program));

    let postselected = parse_dropping_records("M 0\nDETECTOR rec[-1]\nPOSTSELECT rec[-1]");
    assert!(!qec_records_unobserved(&postselected));
}

#[test]
fn constant_pattern_exists_only_for_fixed_parities() {
    for (text, expected) in [
        (
            "X 0\nH 1\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]",
            Some(vec![0b11]),
        ),
        (
            "H 0\nCX 0 1\nM 0 1\nDETECTOR rec[-2] rec[-1]",
            Some(vec![0]),
        ),
        (
            "H 0\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]",
            None,
        ),
    ] {
        let program = parse_dropping_records(text);
        let circuit = lower_qec_program_to_clifford_circuit(&program).unwrap();
        let sampler = compile_detector_sampler(
            &circuit,
            program.detector_rows().unwrap(),
            program.observable_rows().unwrap(),
            42,
        )
        .unwrap();
        let projection = QecParityProjection::new(
            sampler.num_measurements(),
            sampler.detector_rows(),
            sampler.observable_rows(),
        );
        let pattern = projection.constant_pattern(sampler.measurement_sampler());
        assert_eq!(pattern, expected, "{text}");
    }
}

#[test]
fn parity_compile_fixes_the_parities_the_record_sampler_fixes() {
    for text in [
        "X_ERROR(0.1) 0
X 0
H 1
M 0 1
DETECTOR rec[-2]
OBSERVABLE_INCLUDE(0) rec[-2]",
        "DEPOLARIZE1(0.1) 0
H 0
M 0 1
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]",
        "RX 0
Z_ERROR(0.1) 0
MX 0
DETECTOR rec[-1]",
        "RY 0
X_ERROR(0.1) 0
MY 0
DETECTOR rec[-1]",
        "H 0
CX 0 1
DEPOLARIZE2(0.1) 0 1
MPP X0*X1 Z0*Z1
MPP X0*X1
DETECTOR rec[-1] rec[-3]",
        "X_ERROR(0.1) 0
MPP X0
DETECTOR rec[-1]",
        "H 0
M 0
R 0
X_ERROR(0.1) 0
M 0
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(0) rec[-2]",
        "H 0
M 0
R 0
X_ERROR(0.1) 0
M 0
DETECTOR rec[-1]",
        "RX 1
MRX 1
DEPOLARIZE2(0.2) 0 1
X 0
MR 0 1
DETECTOR rec[-2] rec[-3]
DETECTOR rec[-1]",
    ] {
        let program = parse_dropping_records(text);
        let projection = QecParityProjection::new(
            program.num_measurements(),
            &program.detector_rows().unwrap(),
            &program.observable_rows().unwrap(),
        );
        let sampler = compile_qec_noisy_sampler(&program).unwrap();
        let expected = projection.constant_pattern(sampler.noiseless());
        let pattern = compile_qec_parity_noise(&program, &projection)
            .unwrap()
            .map(|(pattern, _)| pattern);
        assert_eq!(pattern, expected, "{text}");
    }
}
fn random_clifford_qec_text(rng: &mut ChaCha8Rng) -> String {
    const PAULIS: [&str; 3] = ["X", "Y", "Z"];
    const GATES: [&str; 8] = ["H", "S", "S_DAG", "X", "Y", "Z", "CX", "CZ"];
    const MEASURES: [&str; 6] = ["M", "MX", "MY", "MR", "MRX", "MRY"];
    const RESETS: [&str; 3] = ["R", "RX", "RY"];
    let num_qubits = 3;
    let mut lines = Vec::new();
    let mut records = 0usize;
    for _ in 0..rng.random_range(4..24) {
        let q = rng.random_range(0..num_qubits);
        match rng.random_range(0..10) {
            0..=3 => {
                let gate = GATES[rng.random_range(0..GATES.len())];
                if gate.starts_with('C') {
                    let other = (q + rng.random_range(1..num_qubits)) % num_qubits;
                    lines.push(format!("{gate} {q} {other}"));
                } else {
                    lines.push(format!("{gate} {q}"));
                }
            }
            4..=5 => {
                let measure = MEASURES[rng.random_range(0..MEASURES.len())];
                lines.push(format!("{measure} {q}"));
                if !measure.starts_with("MR") {
                    lines.push(format!("{} {q}", RESETS[rng.random_range(0..RESETS.len())]));
                }
                records += 1;
            }
            6 => lines.push(format!("{} {q}", RESETS[rng.random_range(0..RESETS.len())])),
            7 => lines.push(format!("DEPOLARIZE1(0.1) {q}")),
            _ => {
                let mut terms = Vec::new();
                for t in 0..num_qubits {
                    if rng.random_range(0..2) == 0 {
                        terms.push(format!("{}{t}", PAULIS[rng.random_range(0..3)]));
                    }
                }
                if !terms.is_empty() {
                    lines.push(format!("MPP {}", terms.join("*")));
                    records += 1;
                }
            }
        }
    }
    if records == 0 {
        return String::new();
    }
    for output in 0..rng.random_range(1..3) {
        let refs: Vec<String> = (0..rng.random_range(1..4))
            .map(|_| format!("rec[-{}]", 1 + rng.random_range(0..records.min(4))))
            .collect();
        if output == 0 {
            lines.push(format!("DETECTOR {}", refs.join(" ")));
        } else {
            lines.push(format!("OBSERVABLE_INCLUDE(0) {}", refs.join(" ")));
        }
    }
    lines.join(
        "
",
    )
}

#[test]
fn parity_compile_matches_the_record_sampler_on_random_clifford_programs() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut fixed = 0;
    let mut ones = 0;
    for _ in 0..3000 {
        let text = random_clifford_qec_text(&mut rng);
        if text.is_empty() {
            continue;
        }
        let program = parse_dropping_records(&text);
        let projection = QecParityProjection::new(
            program.num_measurements(),
            &program.detector_rows().unwrap(),
            &program.observable_rows().unwrap(),
        );
        let sampler = compile_qec_noisy_sampler(&program).unwrap();
        let expected = projection.constant_pattern(sampler.noiseless());
        let compiled = compile_qec_parity_noise(&program, &projection).unwrap();
        let pattern = compiled.as_ref().map(|(pattern, _)| pattern.clone());
        assert_eq!(pattern, expected, "{text}");
        for windows in [2, 5] {
            let split =
                compile_qec_parity_noise_windows(&program, &projection, Some(windows)).unwrap();
            assert_eq!(
                split, compiled,
                "{text}
{windows} windows"
            );
        }
        if let Some(pattern) = pattern {
            fixed += 1;
            ones += usize::from(pattern[0] != 0);
        }
    }
    assert!(fixed > 300 && ones > 20, "fixed {fixed}, nonzero {ones}");
}

fn repetition_memory_text(n_data: usize, rounds: usize) -> String {
    let checks = n_data - 1;
    let data: Vec<String> = (0..n_data).map(|q| q.to_string()).collect();
    let mut lines = Vec::new();
    for round in 0..rounds {
        lines.push(format!("DEPOLARIZE1(0.01) {}", data.join(" ")));
        for c in 0..checks {
            lines.push(format!("DEPOLARIZE2(0.01) {} {}", c, c + 1));
            lines.push(format!("MPP Z{}*Z{}", c, c + 1));
        }
        for c in 0..checks {
            if round == 0 {
                lines.push(format!("DETECTOR rec[-{}]", checks - c));
            } else {
                lines.push(format!(
                    "DETECTOR rec[-{}] rec[-{}]",
                    checks - c,
                    2 * checks - c
                ));
            }
        }
    }
    lines.push(format!("M {}", data.join(" ")));
    for c in 0..checks {
        lines.push(format!(
            "DETECTOR rec[-{}] rec[-{}] rec[-{}]",
            n_data - c,
            n_data - c - 1,
            n_data + checks - c
        ));
    }
    lines.push(format!("OBSERVABLE_INCLUDE(0) rec[-{n_data}]"));
    lines.join(
        "
",
    )
}

// Windows own disjoint detector sets, hand flips on earlier windows' events back,
// and take the observable and fingerprint state from the sequential pass; every
// count must compile the same events, branches and pattern as the one-window walk.
#[test]
fn parity_compile_is_the_same_over_any_window_count() {
    let program = parse_dropping_records(&repetition_memory_text(5, 40));
    let projection = QecParityProjection::new(
        program.num_measurements(),
        &program.detector_rows().unwrap(),
        &program.observable_rows().unwrap(),
    );
    let reference = compile_qec_parity_noise_windows(&program, &projection, Some(1))
        .unwrap()
        .expect("a repetition memory has a fixed pattern");
    assert!(reference.1.event_count() > 300);
    for windows in [2, 3, 5, 8, 40] {
        let split = compile_qec_parity_noise_windows(&program, &projection, Some(windows))
            .unwrap()
            .unwrap();
        assert_eq!(split, reference, "{windows} windows");
    }
}

#[test]
fn postselection_counts_match_a_per_shot_count_in_both_layouts() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let (shots, rows, observables) = (200, 70, 3);
    let rows_and_expected: Vec<(Vec<usize>, bool)> = (0..rows)
        .map(|row| (vec![row], rng.random_range(0..2) == 1))
        .collect();
    let mut post = vec![0u64; shots * rows.div_ceil(64)];
    let mut obs = vec![0u64; shots];
    let mut accepted = 0;
    let mut errors = vec![0u64; observables];
    for shot in 0..shots {
        let rejected_row = (rng.random_range(0..2) == 1).then(|| rng.random_range(0..rows));
        for (row, (_, expected)) in rows_and_expected.iter().enumerate() {
            if *expected != (rejected_row == Some(row)) {
                post[shot * rows.div_ceil(64) + row / 64] |= 1 << (row % 64);
            }
        }
        obs[shot] = rng.random_range(0..1u64 << observables);
        if rejected_row.is_none() {
            accepted += 1;
            for (observable, count) in errors.iter_mut().enumerate() {
                *count += (obs[shot] >> observable) & 1;
            }
        }
    }
    let post = PackedShots::from_shot_major(post, shots, rows);
    let obs = PackedShots::from_shot_major(obs, shots, observables);
    for post in [post.clone(), post.to_meas_major()] {
        for obs in [obs.clone(), obs.to_meas_major()] {
            let mask = qec_postselection_accept_mask(&post, &rows_and_expected);
            assert_eq!(qec_count_ones(&mask), accepted);
            let mut counts = vec![0u64; observables];
            add_qec_logical_error_counts(&obs, Some(&mask), &mut counts);
            assert_eq!(counts, errors);
        }
    }
}
