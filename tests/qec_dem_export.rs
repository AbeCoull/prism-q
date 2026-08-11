//! Detector error model export: symptoms and probabilities against hand
//! enumeration, text round-trip, model-predicted detector statistics against
//! sampling, and a lookup decode checked against the model's exact prediction.

mod qec_common;

use prism_q::{Gate, PackedShots, QecNoise, QecPauli, QecProgram, QecRecordRef, run_qec_program};

const STAT_SHOTS: usize = 20_000;

struct ParsedMechanism {
    probability: f64,
    detectors: Vec<usize>,
    observables: Vec<usize>,
}

struct ParsedModel {
    mechanisms: Vec<ParsedMechanism>,
    detector_lines: usize,
    observable_lines: usize,
}

// Test-local reader for the emitted text, so every statistical check consumes
// only the export artifact rather than the in-memory model.
fn parse_dem_text(text: &str) -> ParsedModel {
    let mut model = ParsedModel {
        mechanisms: Vec::new(),
        detector_lines: 0,
        observable_lines: 0,
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("error(") {
            let (probability, targets) = rest.split_once(')').expect("unterminated error line");
            let mut mechanism = ParsedMechanism {
                probability: probability.parse().unwrap(),
                detectors: Vec::new(),
                observables: Vec::new(),
            };
            for target in targets.split_whitespace() {
                if let Some(detector) = target.strip_prefix('D') {
                    mechanism.detectors.push(detector.parse().unwrap());
                } else if let Some(observable) = target.strip_prefix('L') {
                    mechanism.observables.push(observable.parse().unwrap());
                } else {
                    panic!("unknown error target {target}");
                }
            }
            model.mechanisms.push(mechanism);
        } else if line.starts_with("detector") {
            model.detector_lines += 1;
        } else if line.starts_with("logical_observable") {
            model.observable_lines += 1;
        } else {
            panic!("unknown DEM line {line}");
        }
    }
    model
}

fn bit_rate(shots: &PackedShots, column: usize) -> f64 {
    let mut count = 0usize;
    for shot in 0..shots.num_shots() {
        count += usize::from(shots.get_bit(shot, column));
    }
    count as f64 / shots.num_shots() as f64
}

// Distance-3 repetition-code Z memory: noise on the data qubits before each
// round of MPP ZZ checks, consecutive-round detectors, final data readout
// detectors, observable on the final M 0 record.
fn repetition_memory(rounds: usize, noise: QecNoise) -> QecProgram {
    let mut program = QecProgram::with_options(3, qec_common::qec_options(STAT_SHOTS, 4096, false));
    let mut prev: Option<(usize, usize)> = None;
    for _ in 0..rounds {
        program.noise(noise, &[0, 1, 2]).unwrap();
        let c0 = program
            .measure_pauli_product(&[QecPauli::z(0), QecPauli::z(1)])
            .unwrap();
        let c1 = program
            .measure_pauli_product(&[QecPauli::z(1), QecPauli::z(2)])
            .unwrap();
        match prev {
            None => {
                program.detector(&[QecRecordRef::absolute(c0)]).unwrap();
                program.detector(&[QecRecordRef::absolute(c1)]).unwrap();
            }
            Some((last0, last1)) => {
                program
                    .detector(&[QecRecordRef::absolute(c0), QecRecordRef::absolute(last0)])
                    .unwrap();
                program
                    .detector(&[QecRecordRef::absolute(c1), QecRecordRef::absolute(last1)])
                    .unwrap();
            }
        }
        prev = Some((c0, c1));
    }
    let (last0, last1) = prev.unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_z(1).unwrap();
    let m2 = program.measure_z(2).unwrap();
    program
        .detector(&[
            QecRecordRef::absolute(m0),
            QecRecordRef::absolute(m1),
            QecRecordRef::absolute(last0),
        ])
        .unwrap();
    program
        .detector(&[
            QecRecordRef::absolute(m1),
            QecRecordRef::absolute(m2),
            QecRecordRef::absolute(last1),
        ])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
}

const SURFACE_X_STABILIZERS: [&[usize]; 4] = [&[0, 1, 3, 4], &[4, 5, 7, 8], &[2, 5], &[3, 6]];
const SURFACE_Z_STABILIZERS: [&[usize]; 4] = [&[1, 2, 4, 5], &[3, 4, 6, 7], &[0, 1], &[7, 8]];

// Distance-3 rotated surface-code Z memory via MPP stabilizer measurements.
// Round-0 detectors on the Z checks only (deterministic on |0..0>), compare
// detectors on later rounds, final Z checks reconstructed from the data
// readout, observable on the left-column logical Z.
fn surface_memory_d3(rounds: usize, noise: QecNoise) -> QecProgram {
    let mut program = QecProgram::with_options(9, qec_common::qec_options(STAT_SHOTS, 4096, false));
    let data_qubits: Vec<usize> = (0..9).collect();
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program.noise(noise, &data_qubits).unwrap();
        let mut records = Vec::with_capacity(8);
        for stab in SURFACE_Z_STABILIZERS {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::z(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        for stab in SURFACE_X_STABILIZERS {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::x(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        match &prev {
            None => {
                for &record in &records[..4] {
                    program.detector(&[QecRecordRef::absolute(record)]).unwrap();
                }
            }
            Some(previous) => {
                for (&record, &prior) in records.iter().zip(previous) {
                    program
                        .detector(&[
                            QecRecordRef::absolute(record),
                            QecRecordRef::absolute(prior),
                        ])
                        .unwrap();
                }
            }
        }
        prev = Some(records);
    }
    let previous = prev.unwrap();
    let data: Vec<usize> = (0..9).map(|q| program.measure_z(q).unwrap()).collect();
    for (stab, &prior) in SURFACE_Z_STABILIZERS.iter().zip(&previous[..4]) {
        let mut refs: Vec<QecRecordRef> = stab
            .iter()
            .map(|&q| QecRecordRef::absolute(data[q]))
            .collect();
        refs.push(QecRecordRef::absolute(prior));
        program.detector(&refs).unwrap();
    }
    let logical: Vec<QecRecordRef> = [0usize, 3, 6]
        .iter()
        .map(|&q| QecRecordRef::absolute(data[q]))
        .collect();
    program.observable_include(0, &logical).unwrap();
    program
}

#[test]
fn dem_x_error_symptoms_match_hand_enumeration() {
    // One syndrome round over three data qubits: X on qubit 0 flips only
    // check 0, X on qubit 1 both checks, X on qubit 2 only check 1, each at
    // exactly p.
    let p = 0.05;
    let mut program = QecProgram::new(5);
    program.noise(QecNoise::XError(p), &[0, 1, 2]).unwrap();
    for pair in [[0, 3], [1, 3], [1, 4], [2, 4]] {
        program.push_gate(Gate::Cx, &pair).unwrap();
    }
    let m0 = program.measure_z(3).unwrap();
    let m1 = program.measure_z(4).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program.detector(&[QecRecordRef::absolute(m1)]).unwrap();

    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_detectors(), 2);
    assert_eq!(model.num_observables(), 0);
    let symptoms: Vec<(&[usize], &[usize], f64)> = model
        .mechanisms()
        .iter()
        .map(|m| (m.detectors(), m.observables(), m.probability()))
        .collect();
    assert_eq!(
        symptoms,
        vec![
            (&[0][..], &[][..], p),
            (&[0, 1][..], &[][..], p),
            (&[1][..], &[][..], p),
        ]
    );
    assert_eq!(
        model.to_text(),
        "error(0.05) D0\nerror(0.05) D0 D1\nerror(0.05) D1\ndetector D0\ndetector D1\n"
    );

    // EXP_VAL ops do not perturb the derivation.
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1), QecPauli::z(2)], 1.0)
        .unwrap();
    assert_eq!(program.detector_error_model().unwrap(), model);
}

#[test]
fn dem_depolarize1_merges_exclusive_branches() {
    // All records are Z products, so per data qubit the X and Y branches share
    // a symptom and must sum to exactly 2p/3 while the Z branch flips nothing.
    // Hand enumeration for one round: qubit 0 flips check 0 and the observable
    // (the final-readout detector cancels: its data record and its check
    // record both flip), qubit 1 flips checks 0 and 1, qubit 2 flips check 1.
    let p = 0.09;
    let program = repetition_memory(1, QecNoise::Depolarize1(p));
    let model = program.detector_error_model().unwrap();

    assert_eq!(model.num_detectors(), 4);
    assert_eq!(model.num_observables(), 1);
    assert_eq!(model.num_mechanisms(), 3);
    let expected: [(&[usize], &[usize]); 3] = [(&[0], &[0]), (&[0, 1], &[]), (&[1], &[])];
    for (mechanism, (detectors, observables)) in model.mechanisms().iter().zip(expected) {
        assert_eq!(mechanism.detectors(), detectors);
        assert_eq!(mechanism.observables(), observables);
        assert!(
            (mechanism.probability() - 2.0 * p / 3.0).abs() < 1e-12,
            "exclusive branches must sum, got {}",
            mechanism.probability()
        );
    }
}

#[test]
fn dem_independent_annotations_xor_compose() {
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    program.noise(QecNoise::XError(0.2), &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();

    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_mechanisms(), 1);
    let expected = 0.1 * (1.0 - 0.2) + 0.2 * (1.0 - 0.1);
    assert!((model.mechanisms()[0].probability() - expected).abs() < 1e-12);
}

#[test]
fn dem_noiseless_program_has_no_mechanisms() {
    let program = QecProgram::from_text(
        "R 0 1 2
         CX 0 1 2 1
         MR 1
         DETECTOR rec[-1]
         M 0 2
         OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]",
    )
    .unwrap();
    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_mechanisms(), 0);
    assert_eq!(model.num_detectors(), 1);
    assert_eq!(model.num_observables(), 1);
    assert_eq!(model.to_text(), "detector D0\nlogical_observable L0\n");
}

#[test]
fn dem_detector_coordinates_survive_export() {
    let mut program = QecProgram::new(1);
    let m0 = program.measure_z(0).unwrap();
    program
        .detector_with_coords(&[QecRecordRef::absolute(m0)], &[0.5, 0.0])
        .unwrap();
    let model = program.detector_error_model().unwrap();
    assert_eq!(model.detector_coords(), &[vec![0.5, 0.0]]);
    assert_eq!(model.to_text(), "detector(0.5, 0) D0\n");
}

#[test]
fn dem_rejects_non_clifford_gates() {
    let mut program = QecProgram::new(1);
    program.push_gate(Gate::T, &[0]).unwrap();
    program.noise(QecNoise::XError(0.1), &[0]).unwrap();
    program.measure_z(0).unwrap();
    assert!(program.detector_error_model().is_err());
}

#[test]
fn dem_surface_d3_round_trip_counts() {
    let program = surface_memory_d3(2, QecNoise::Depolarize1(0.02));
    let model = program.detector_error_model().unwrap();
    assert_eq!(model.num_detectors(), program.num_detectors());
    assert_eq!(model.num_detectors(), 16);
    assert_eq!(model.num_observables(), 1);
    assert!(model.num_mechanisms() > 0);

    let parsed = parse_dem_text(&model.to_text());
    assert_eq!(parsed.detector_lines, model.num_detectors());
    assert_eq!(parsed.observable_lines, model.num_observables());
    assert_eq!(parsed.mechanisms.len(), model.num_mechanisms());
    for (mechanism, parsed) in model.mechanisms().iter().zip(&parsed.mechanisms) {
        assert_eq!(mechanism.detectors(), parsed.detectors.as_slice());
        assert_eq!(mechanism.observables(), parsed.observables.as_slice());
        assert_eq!(mechanism.probability(), parsed.probability);
    }
}

// P(parity of the mechanisms in `hits` is odd) = (1 - prod(1 - 2p)) / 2.
fn predicted_odd_rate(parity_probabilities: &[f64]) -> f64 {
    let attenuation: f64 = parity_probabilities
        .iter()
        .map(|&p| 1.0 - 2.0 * p)
        .product();
    (1.0 - attenuation) / 2.0
}

fn assert_model_predicts_statistics(program: &QecProgram, label: &str) {
    let model = program.detector_error_model().unwrap();
    let parsed = parse_dem_text(&model.to_text());
    let result = run_qec_program(program).unwrap();
    let num_detectors = model.num_detectors();
    assert_eq!(result.detectors.num_measurements(), num_detectors);

    for detector in 0..num_detectors {
        let hits: Vec<f64> = parsed
            .mechanisms
            .iter()
            .filter(|m| m.detectors.contains(&detector))
            .map(|m| m.probability)
            .collect();
        let predicted = predicted_odd_rate(&hits);
        let sampled = bit_rate(&result.detectors, detector);
        assert!(
            (sampled - predicted).abs() < 0.025,
            "{label} detector {detector}: sampled {sampled:.4} vs predicted {predicted:.4}"
        );
    }

    for i in 0..num_detectors {
        for j in (i + 1)..num_detectors {
            let hits: Vec<f64> = parsed
                .mechanisms
                .iter()
                .filter(|m| m.detectors.contains(&i) != m.detectors.contains(&j))
                .map(|m| m.probability)
                .collect();
            let predicted = predicted_odd_rate(&hits);
            let mut count = 0usize;
            for shot in 0..result.detectors.num_shots() {
                count += usize::from(
                    result.detectors.get_bit(shot, i) != result.detectors.get_bit(shot, j),
                );
            }
            let sampled = count as f64 / result.detectors.num_shots() as f64;
            assert!(
                (sampled - predicted).abs() < 0.025,
                "{label} pair ({i},{j}): sampled {sampled:.4} vs predicted {predicted:.4}"
            );
        }
    }
}

#[test]
fn dem_model_predicts_detector_statistics() {
    assert_model_predicts_statistics(
        &repetition_memory(3, QecNoise::Depolarize1(0.05)),
        "repetition d3",
    );
    assert_model_predicts_statistics(
        &surface_memory_d3(2, QecNoise::Depolarize1(0.02)),
        "surface d3",
    );
}

#[test]
fn dem_lookup_decode_matches_model_prediction() {
    let p = 0.05;
    let program = repetition_memory(3, QecNoise::Depolarize1(p));
    let model = program.detector_error_model().unwrap();
    let parsed = parse_dem_text(&model.to_text());
    let num_detectors = model.num_detectors();
    assert_eq!(num_detectors, 8);

    // Exact joint distribution over (syndrome, observable) under the model:
    // XOR-convolve one two-point distribution per mechanism.
    let states = 1usize << (num_detectors + 1);
    let mut dist = vec![0.0f64; states];
    dist[0] = 1.0;
    for mechanism in &parsed.mechanisms {
        let mut flip = 0usize;
        for &detector in &mechanism.detectors {
            flip |= 1 << detector;
        }
        for &observable in &mechanism.observables {
            assert_eq!(observable, 0);
            flip |= 1 << num_detectors;
        }
        let q = mechanism.probability;
        let mut next = vec![0.0f64; states];
        for (state, &mass) in dist.iter().enumerate() {
            next[state] += mass * (1.0 - q);
            next[state ^ flip] += mass * q;
        }
        dist = next;
    }

    // Maximum-likelihood lookup decoder and its exact logical error rate under
    // the model.
    let syndromes = 1usize << num_detectors;
    let mut predict = vec![false; syndromes];
    let mut analytic_error = 0.0f64;
    for syndrome in 0..syndromes {
        let quiet = dist[syndrome];
        let flipped = dist[syndrome | 1 << num_detectors];
        predict[syndrome] = flipped > quiet;
        analytic_error += quiet.min(flipped);
    }
    assert!(analytic_error > 1e-4, "vacuous fixture: no decodable mass");

    let result = run_qec_program(&program).unwrap();
    let mut mismatches = 0usize;
    for shot in 0..result.total_shots {
        let mut syndrome = 0usize;
        for detector in 0..num_detectors {
            syndrome |= usize::from(result.detectors.get_bit(shot, detector)) << detector;
        }
        let observed = result.observables.get_bit(shot, 0);
        mismatches += usize::from(predict[syndrome] != observed);
    }
    let empirical = mismatches as f64 / result.total_shots as f64;

    // 5 sigma of the model's own prediction, plus slack for the second-order
    // gap between independent mechanisms and exclusive channel branches.
    let sigma = (analytic_error * (1.0 - analytic_error) / result.total_shots as f64).sqrt();
    let tolerance = 5.0 * sigma + 0.005;
    assert!(
        (empirical - analytic_error).abs() < tolerance,
        "decoded rate {empirical:.5} vs model prediction {analytic_error:.5} (tol {tolerance:.5})"
    );
    assert!(empirical < p, "decode must beat the physical error rate");
}
