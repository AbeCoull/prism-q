//! Detector and observable bits measured against the noiseless reference run, the raw
//! parity option, and compile-once sampling through `QecProgram::compile_sampler`.

use prism_q::{
    BpOsdDecoder, QecBasis, QecCircuitNoise, QecDetectorReference, QecOptions, QecProgram,
    QecSampleResult, parse_qec_program, run_qec_program, run_qec_program_reference,
};

const COLOR_CODE_MEMORY_XYZ_D5: &str = include_str!("../fixtures/qec/color_code_memory_xyz_d5.txt");

fn with_options(mut program: QecProgram, options: QecOptions) -> QecProgram {
    program.set_options(options);
    program
}

fn options(shots: usize, keep_measurements: bool, reference: QecDetectorReference) -> QecOptions {
    QecOptions {
        shots,
        seed: 42,
        chunk_size: None,
        keep_measurements,
        detector_reference: reference,
    }
}

fn any_bit_set(result: &QecSampleResult) -> bool {
    result
        .detectors
        .to_shots()
        .iter()
        .chain(&result.observables.to_shots())
        .flatten()
        .any(|&bit| bit)
}

#[test]
fn a_noiselessly_flipped_record_reports_no_detection_event() {
    let text = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    let program = with_options(
        parse_qec_program(text).unwrap(),
        QecOptions {
            shots: 1000,
            ..QecOptions::default()
        },
    );
    let result = run_qec_program(&program).unwrap();
    assert!(!any_bit_set(&result));
    assert_eq!(result.logical_error_rates(), vec![0.0]);

    let raw = with_options(
        program,
        QecOptions {
            detector_reference: QecDetectorReference::Raw,
            shots: 1000,
            ..QecOptions::default()
        },
    );
    let result = run_qec_program(&raw).unwrap();
    assert!(result.detectors.to_shots().iter().flatten().all(|&bit| bit));
    assert_eq!(result.logical_error_rates(), vec![1.0]);
}

#[test]
fn noiseless_color_code_xyz_memory_never_fires() {
    let noiseless = COLOR_CODE_MEMORY_XYZ_D5.replace("(0.001)", "(0)");
    for keep_measurements in [true, false] {
        let program = with_options(
            parse_qec_program(&noiseless).unwrap(),
            options(256, keep_measurements, QecDetectorReference::Noiseless),
        );
        assert!(!any_bit_set(&run_qec_program(&program).unwrap()));

        let raw = with_options(
            program,
            options(256, keep_measurements, QecDetectorReference::Raw),
        );
        assert!(
            any_bit_set(&run_qec_program(&raw).unwrap()),
            "some detectors of this code have noiseless parity one"
        );
    }
}

#[test]
fn raw_and_referenced_bits_differ_by_one_pattern_in_every_shot() {
    let program = parse_qec_program(COLOR_CODE_MEMORY_XYZ_D5).unwrap();
    let referenced = run_qec_program(&with_options(
        program.clone(),
        options(500, true, QecDetectorReference::Noiseless),
    ))
    .unwrap();
    let raw = run_qec_program(&with_options(
        program,
        options(500, true, QecDetectorReference::Raw),
    ))
    .unwrap();
    assert_eq!(
        referenced.measurements.to_shots(),
        raw.measurements.to_shots()
    );
    let difference = |a: &[Vec<bool>], b: &[Vec<bool>]| -> Vec<Vec<bool>> {
        a.iter()
            .zip(b)
            .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p ^ q).collect())
            .collect()
    };
    for (a, b) in [
        (&referenced.detectors, &raw.detectors),
        (&referenced.observables, &raw.observables),
    ] {
        let diff = difference(&a.to_shots(), &b.to_shots());
        assert!(diff.iter().all(|shot| shot == &diff[0]));
    }
}

#[test]
fn generated_memories_never_fire_without_noise() {
    let noise = QecCircuitNoise::default();
    for distance in [3, 5] {
        for program in [
            QecProgram::repetition_memory(distance, distance, &noise).unwrap(),
            QecProgram::surface_memory(distance, distance, QecBasis::X, &noise).unwrap(),
            QecProgram::surface_memory(distance, distance, QecBasis::Z, &noise).unwrap(),
            QecProgram::color_memory(distance, distance, QecBasis::Z, &noise).unwrap(),
        ] {
            for reference in [QecDetectorReference::Noiseless, QecDetectorReference::Raw] {
                let program = with_options(program.clone(), options(128, false, reference));
                assert!(!any_bit_set(&run_qec_program(&program).unwrap()));
            }
        }
    }
}

#[test]
fn the_reference_runner_measures_against_the_same_reference() {
    let text = "R 0 1 2\nH 0\nCX 0 1\nX 2\nX_ERROR(0.05) 1 2\nM 0 1 2\nDETECTOR rec[-2] rec[-3]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-3]";
    for reference in [QecDetectorReference::Noiseless, QecDetectorReference::Raw] {
        let program = with_options(
            parse_qec_program(text).unwrap(),
            options(4000, true, reference),
        );
        let compiled = run_qec_program(&program).unwrap();
        let oracle = run_qec_program_reference(&program).unwrap();
        for (a, b) in [
            (&compiled.detectors, &oracle.detectors),
            (&compiled.observables, &oracle.observables),
        ] {
            for column in 0..a.num_measurements() {
                let rate = |packed: &prism_q::PackedShots| {
                    (0..packed.num_shots())
                        .filter(|&shot| packed.get_bit(shot, column))
                        .count() as f64
                        / packed.num_shots() as f64
                };
                assert!(
                    (rate(a) - rate(b)).abs() < 0.03,
                    "{reference:?} column {column}: {} vs {}",
                    rate(a),
                    rate(b)
                );
            }
        }
    }
}

#[test]
fn decoders_read_referenced_samples() {
    let program = with_options(
        parse_qec_program(COLOR_CODE_MEMORY_XYZ_D5).unwrap(),
        options(2000, false, QecDetectorReference::Noiseless),
    );
    let model = program.detector_error_model().unwrap();
    let decoder = BpOsdDecoder::from_model(&model).unwrap();
    let result = run_qec_program(&program).unwrap();
    let rate = decoder
        .logical_error_rate(&result.detectors, &result.observables)
        .unwrap();
    assert!(rate < 0.1, "logical error rate {rate}");
}

#[test]
fn a_detector_error_model_needs_fixed_detectors() {
    let program = parse_qec_program("H 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]").unwrap();
    let err = program.detector_error_model().unwrap_err().to_string();
    assert!(err.contains("detector 0 is not deterministic"), "{err}");
    let program = parse_qec_program(
        "R 0 1\nH 1\nX_ERROR(0.1) 0\nM 0 1\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
    )
    .unwrap();
    let err = program.detector_error_model().unwrap_err().to_string();
    assert!(err.contains("observable 0 is not deterministic"), "{err}");
}

#[test]
fn a_compiled_sampler_draws_what_run_draws_at_each_seed() {
    let program = with_options(
        parse_qec_program(COLOR_CODE_MEMORY_XYZ_D5).unwrap(),
        options(777, false, QecDetectorReference::Noiseless),
    );
    let mut sampler = program.compile_sampler().unwrap();
    for seed in [42, 7, 1 << 40] {
        let expected = run_qec_program(&with_options(
            program.clone(),
            QecOptions {
                seed,
                ..program.options()
            },
        ))
        .unwrap();
        let sampled = sampler.sample(777, Some(seed)).unwrap();
        assert_eq!(sampled.detectors.to_shots(), expected.detectors.to_shots());
        assert_eq!(
            sampled.observables.to_shots(),
            expected.observables.to_shots()
        );
        assert_eq!(sampled.logical_errors, expected.logical_errors);
    }
    let small = sampler.sample(1, None).unwrap();
    assert_eq!(small.total_shots, 1);
    assert_eq!(small.detectors.num_measurements(), program.num_detectors());
}
