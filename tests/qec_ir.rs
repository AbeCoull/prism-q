//! The native QEC program IR: record-referencing construction, the parsed text
//! form, the sample-result accessors, and agreement between the compiled and
//! reference runners on Clifford programs.

use prism_q::circuit::openqasm;
use prism_q::{
    Gate, PackedShots, PrismError, QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram,
    QecRecordRef, QecSampleResult, ShotLayout, compile_qec_program_rows, parse_qec_program,
    run_qec_program, run_qec_program_reference,
};

mod qec_common;

fn assert_f64_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {actual} to be within {tolerance} of {expected}"
    );
}

fn assert_interval_close(actual: (f64, f64), expected: (f64, f64), tolerance: f64) {
    assert_f64_close(actual.0, expected.0, tolerance);
    assert_f64_close(actual.1, expected.1, tolerance);
}

#[test]
fn qec_program_builds_measurement_record_rows() {
    let mut program = QecProgram::new(2);
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_x(1).unwrap();
    let m2 = program
        .measure_pauli_product(&[QecPauli::x(0), QecPauli::z(1)])
        .unwrap();

    assert_eq!(m0, 0);
    assert_eq!(m1, 1);
    assert_eq!(m2, 2);

    program
        .detector_with_coords(
            &[
                QecRecordRef::absolute(m0),
                QecRecordRef::lookback(1).unwrap(),
            ],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m1)])
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::x(1)], -0.5)
        .unwrap();
    program
        .postselect(&[QecRecordRef::lookback(1).unwrap()], false)
        .unwrap();

    assert_eq!(program.num_qubits(), 2);
    assert_eq!(program.num_measurements(), 3);
    assert_eq!(program.num_detectors(), 1);
    assert_eq!(program.num_observables(), 1);
    assert_eq!(program.detector_rows().unwrap(), vec![vec![0, 2]]);
    assert_eq!(program.observable_rows().unwrap(), vec![vec![1]]);
    assert_eq!(
        program.postselection_rows().unwrap(),
        vec![(vec![2], false)]
    );

    let result = program.empty_result();
    assert_eq!(result.measurements.num_measurements(), 3);
    assert_eq!(result.detectors.num_measurements(), 1);
    assert_eq!(result.observables.num_measurements(), 1);
    assert_eq!(result.logical_errors, vec![0]);
}

#[test]
fn qec_program_from_ops_validates_records_and_qubits() {
    let bad_record = QecProgram::from_ops(
        1,
        QecOptions::default(),
        vec![prism_q::QecOp::Detector {
            records: vec![QecRecordRef::absolute(0)],
            coords: Vec::new(),
        }],
    );
    assert!(bad_record.is_err());

    let mut program = QecProgram::new(1);
    assert!(program.measure_z(1).is_err());
    assert!(
        program
            .measure_pauli_product(&[QecPauli::x(0), QecPauli::z(0)])
            .is_err()
    );
    assert!(program.noise(QecNoise::XError(1.5), &[0]).is_err());
    assert!(program.noise(QecNoise::Depolarize2(0.001), &[0]).is_err());
    assert!(
        program
            .noise(QecNoise::Depolarize2(0.001), &[0, 0])
            .is_err()
    );
}

#[test]
fn qec_options_are_configurable() {
    let options = QecOptions {
        shots: 4096,
        seed: 7,
        chunk_size: Some(512),
        keep_measurements: false,
    };
    let mut program = QecProgram::with_options(3, options);
    assert_eq!(program.options(), options);

    let updated = QecOptions {
        seed: 11,
        ..options
    };
    program.set_options(updated);
    assert_eq!(program.options(), updated);
}

#[test]
fn qec_sample_result_validates_packed_dimensions() {
    let measurements = PackedShots::from_shot_major(vec![0, 1], 2, 1);
    let detectors = PackedShots::from_shot_major(vec![1, 0], 2, 1);
    let observables = PackedShots::from_shot_major(vec![1, 1], 2, 1);
    let result = QecSampleResult::new(measurements, detectors, observables, 2, 0, vec![1]);
    let result = result.unwrap();
    assert_eq!(result.total_shots, 2);

    let measurements = PackedShots::from_shot_major(vec![0, 1], 2, 1);
    let detectors = PackedShots::from_shot_major(vec![1], 1, 1);
    let observables = PackedShots::from_shot_major(vec![1, 1], 2, 1);
    let result = QecSampleResult::new(measurements, detectors, observables, 2, 0, vec![1]);
    assert!(result.is_err());
}

#[test]
fn qec_sample_result_allows_omitted_raw_measurements() {
    let measurements = PackedShots::from_meas_major(Vec::new(), 0, 2);
    let detectors = PackedShots::from_meas_major(vec![0b01], 2, 1);
    let observables = PackedShots::from_meas_major(vec![0b10], 2, 1);

    let result = QecSampleResult::new_with_total_shots(
        2,
        measurements,
        detectors,
        observables,
        1,
        1,
        vec![1],
    )
    .unwrap();

    assert_eq!(result.total_shots, 2);
    assert_eq!(result.measurements.num_shots(), 0);
    assert_eq!(result.measurements.num_measurements(), 2);
}

#[test]
fn qec_sample_result_rejects_impossible_logical_error_counts() {
    let measurements = PackedShots::from_shot_major(vec![0, 1], 2, 1);
    let detectors = PackedShots::from_shot_major(vec![1, 0], 2, 1);
    let observables = PackedShots::from_shot_major(vec![1, 1], 2, 1);

    let result = QecSampleResult::new(measurements, detectors, observables, 1, 1, vec![2]);

    assert!(result.is_err());
}

#[test]
fn qec_sample_result_reports_rates_and_wilson_intervals() {
    let measurements = PackedShots::from_shot_major(vec![0; 100], 100, 1);
    let detectors = PackedShots::from_shot_major(vec![0; 100], 100, 1);
    let observables = PackedShots::from_shot_major(vec![0; 100], 100, 2);

    let result = QecSampleResult::new_with_total_shots(
        100,
        measurements,
        detectors,
        observables,
        40,
        60,
        vec![5, 20],
    )
    .unwrap();
    let z_95 = 1.959_963_984_540_054;

    assert_f64_close(result.survivor_rate(), 0.4, 1e-12);
    assert_eq!(result.logical_error_rates().len(), 2);
    assert_f64_close(result.logical_error_rates()[0], 0.125, 1e-12);
    assert_f64_close(result.logical_error_rates()[1], 0.5, 1e-12);
    assert_interval_close(
        result.survivor_rate_wilson_interval(z_95),
        (0.309_401_286_432_459, 0.497_997_413_208_938),
        1e-12,
    );
    assert_eq!(result.logical_error_rate_wilson_intervals(z_95).len(), 2);
    assert_interval_close(
        result.logical_error_rate_wilson_intervals(z_95)[0],
        (0.054_595_002_509_454, 0.261_121_198_388_511),
        1e-12,
    );
    assert_interval_close(
        result.logical_error_rate_wilson_intervals(z_95)[1],
        (0.351_995_269_334_654, 0.648_004_730_665_346),
        1e-12,
    );
}

#[test]
fn qec_sample_result_reports_zero_rates_for_empty_results() {
    let result = QecSampleResult::empty(2, 1, 2);

    assert_eq!(result.survivor_rate(), 0.0);
    assert_eq!(result.logical_error_rates(), vec![0.0, 0.0]);
    assert_eq!(result.survivor_rate_wilson_interval(1.96), (0.0, 0.0));
    assert_eq!(
        result.logical_error_rate_wilson_intervals(1.96),
        vec![(0.0, 0.0), (0.0, 0.0)]
    );
}

#[test]
fn qec_reference_runner_executes_gates_and_postselection() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::X, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], true)
        .unwrap();

    let result = run_qec_program_reference(&program).unwrap();
    assert_eq!(result.total_shots, 4);
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.discarded_shots, 0);
    assert_eq!(result.logical_errors, vec![4]);
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.observables.to_shots(), vec![vec![true]; 4]);
}

// A non-Z basis measurement leaves the qubit in the Z frame rather than in the
// basis it named, so reusing it without a reset would read a state the program
// never asked for. Both runners reject instead of quietly disagreeing, which is
// what keeps the convention unobservable rather than merely documented.
#[test]
fn qec_rejects_reuse_of_a_basis_measured_qubit() {
    let mut program = QecProgram::new(1);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.measure_x(0).unwrap();
    program.measure_z(0).unwrap();

    for err in [
        run_qec_program(&program).unwrap_err(),
        run_qec_program_reference(&program).unwrap_err(),
    ] {
        match err {
            PrismError::InvalidParameter { message } => assert!(
                message.contains("must be reset before it is used again"),
                "expected a reuse rejection, got {message}"
            ),
            other => panic!("expected a reuse rejection, got {other:?}"),
        }
    }
}

// The same shape with the reset the contract requires. Measuring X on |+> is
// deterministic, the reset returns the qubit to |0>, and the Z measurement that
// follows reads 0 on both runners.
#[test]
fn qec_basis_measurement_then_reset_agrees_across_runners() {
    let options = QecOptions {
        shots: 64,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.measure_x(0).unwrap();
    program.reset(QecBasis::Z, 0).unwrap();
    program.measure_z(0).unwrap();

    let compiled = run_qec_program(&program).unwrap().measurements.to_shots();
    let reference = run_qec_program_reference(&program)
        .unwrap()
        .measurements
        .to_shots();

    assert_eq!(compiled, reference, "runners disagree after a legal reuse");
    assert_eq!(compiled, vec![vec![false, false]; 64]);
}

#[test]
fn qec_compiled_runner_executes_clifford_programs_without_statevector_fallback() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::X, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], true)
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.total_shots, 4);
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.discarded_shots, 0);
    assert_eq!(result.logical_errors, vec![4]);
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.observables.to_shots(), vec![vec![true]; 4]);
}

#[test]
fn qec_compiled_runner_handles_mpp_and_reset_reuse() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(2, options);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    let m0 = program
        .measure_pauli_product(&[QecPauli::z(0), QecPauli::z(1)])
        .unwrap();
    program.reset(QecBasis::Z, 1).unwrap();
    let m1 = program.measure_z(1).unwrap();
    program
        .detector(&[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.measurements.to_shots(), vec![vec![false, false]; 4]);
    assert_eq!(result.detectors.to_shots(), vec![vec![false]; 4]);
    assert_eq!(result.logical_errors, vec![0]);
}

#[test]
fn qec_compiled_runner_can_omit_raw_measurements() {
    let options = QecOptions {
        shots: 3,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::X, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.total_shots, 3);
    assert_eq!(result.measurements.num_shots(), 0);
    assert_eq!(result.measurements.num_measurements(), 1);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 3]);
    assert_eq!(result.observables.to_shots(), vec![vec![true]; 3]);
    assert_eq!(result.logical_errors, vec![3]);
}

#[test]
fn qec_compiled_runner_honors_chunk_size() {
    let options = QecOptions {
        shots: 5,
        seed: 42,
        chunk_size: Some(2),
        keep_measurements: false,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::X, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.total_shots, 5);
    assert_eq!(result.measurements.num_shots(), 0);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 5]);
    assert_eq!(result.observables.to_shots(), vec![vec![true]; 5]);
    assert_eq!(result.accepted_shots, 5);
    assert_eq!(result.logical_errors, vec![5]);
}

#[test]
fn qec_compiled_runner_rejects_zero_chunk_size() {
    let options = QecOptions {
        shots: 1,
        seed: 42,
        chunk_size: Some(0),
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.measure_z(0).unwrap();

    let err = run_qec_program(&program).unwrap_err();
    assert!(format!("{err}").contains("chunk_size"));
}

#[test]
fn qec_compiled_runner_rejects_zero_chunk_size_without_measurements() {
    let options = QecOptions {
        shots: 1,
        seed: 42,
        chunk_size: Some(0),
        keep_measurements: true,
    };
    let program = QecProgram::with_options(1, options);

    let err = run_qec_program(&program).unwrap_err();
    assert!(format!("{err}").contains("chunk_size"));
}

#[test]
fn qec_compiled_runner_rejects_non_clifford_without_reference_fallback() {
    let mut program = QecProgram::new(1);
    program.push_gate(Gate::T, &[0]).unwrap();
    program.measure_z(0).unwrap();

    let err = run_qec_program(&program).unwrap_err();
    assert!(format!("{err}").contains("requires Clifford gates"));
}

#[test]
fn qec_compiled_runner_applies_deterministic_x_noise() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.noise(QecNoise::XError(1.0), &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], true)
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 4]);
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.logical_errors, vec![4]);
}

#[test]
fn qec_compiled_runner_applies_basis_sensitive_z_noise() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.noise(QecNoise::ZError(1.0), &[0]).unwrap();
    program.measure_x(0).unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 4]);
}

#[test]
fn qec_compiled_runner_accepts_depolarizing_noise_channels() {
    let options = QecOptions {
        shots: 100,
        seed: 42,
        chunk_size: Some(17),
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(2, options);
    program.noise(QecNoise::Depolarize1(0.0), &[0]).unwrap();
    program.noise(QecNoise::Depolarize2(1.0), &[0, 1]).unwrap();
    program.measure_z(0).unwrap();
    program.measure_z(1).unwrap();

    let result = run_qec_program(&program).unwrap();
    let shots = result.measurements.to_shots();
    assert_eq!(result.total_shots, 100);
    assert_eq!(shots.len(), 100);
    assert!(shots.iter().any(|shot| shot[0] || shot[1]));
}

#[test]
fn qec_compiled_runner_treats_zero_probability_noise_as_clean() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.noise(QecNoise::XError(0.0), &[0]).unwrap();
    program.measure_z(0).unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![false]; 4]);
}

#[test]
fn qec_row_compiler_treats_zero_probability_noise_as_noop() {
    let mut program = QecProgram::new(1);
    program.noise(QecNoise::XError(0.0), &[0]).unwrap();
    let record = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(record)]).unwrap();

    let compiled = compile_qec_program_rows(&program).unwrap();
    assert_eq!(compiled.num_measurements(), 1);
    assert_eq!(compiled.detector_rows(), [vec![0]].as_slice());
}

#[test]
fn qec_reference_runner_does_not_consume_rng_for_zero_noise() {
    let options = QecOptions {
        shots: 64,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut baseline = QecProgram::with_options(1, options);
    baseline.noise(QecNoise::XError(0.5), &[0]).unwrap();
    baseline.measure_z(0).unwrap();

    let mut with_zero = QecProgram::with_options(1, options);
    with_zero.noise(QecNoise::XError(0.0), &[0]).unwrap();
    with_zero.noise(QecNoise::XError(0.5), &[0]).unwrap();
    with_zero.measure_z(0).unwrap();

    let baseline_result = run_qec_program_reference(&baseline).unwrap();
    let with_zero_result = run_qec_program_reference(&with_zero).unwrap();
    assert_eq!(
        baseline_result.measurements.to_shots(),
        with_zero_result.measurements.to_shots()
    );
}

#[test]
fn qec_compiled_runner_ignores_single_qubit_noise_after_measurement() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.measure_z(0).unwrap();
    program.noise(QecNoise::XError(1.0), &[0]).unwrap();
    program.reset(QecBasis::Z, 0).unwrap();
    program.measure_z(0).unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![false, false]; 4]);
}

#[test]
fn qec_compiled_runner_evaluates_noiseless_exp_val_via_analytic_route() {
    let mut exp_val_program = QecProgram::new(1);
    exp_val_program
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let result = run_qec_program(&exp_val_program).unwrap();
    let estimates = result.expectation_values.expect("EXP_VAL estimates");
    assert_eq!(estimates.len(), 1);
    assert!((estimates[0].mean - 1.0).abs() < 1e-12);
    assert!(estimates[0].variance <= 1e-12);
}

#[test]
fn qec_compiled_runner_rejects_measurement_reuse_without_reset() {
    let mut program = QecProgram::new(1);
    program.measure_z(0).unwrap();
    program.push_gate(Gate::X, &[0]).unwrap();
    program.measure_z(0).unwrap();

    let err = run_qec_program(&program).unwrap_err();
    assert!(format!("{err}").contains("reset before reusing"));
}

#[test]
fn qec_reference_runner_measures_mpp_products() {
    let options = QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(2, options);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    let m0 = program
        .measure_pauli_product(&[QecPauli::z(0), QecPauli::z(1)])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let result = run_qec_program_reference(&program).unwrap();
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.measurements.to_shots(), vec![vec![false]; 4]);
    assert_eq!(result.logical_errors, vec![0]);
}

#[test]
fn qec_reference_runner_can_omit_raw_measurements() {
    let options = QecOptions {
        shots: 3,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    };
    let mut program = QecProgram::with_options(1, options);
    program.push_gate(Gate::X, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let result = run_qec_program_reference(&program).unwrap();
    assert_eq!(result.total_shots, 3);
    assert_eq!(result.measurements.num_shots(), 0);
    assert_eq!(result.measurements.num_measurements(), 1);
    assert_eq!(result.detectors.to_shots(), vec![vec![true]; 3]);
    assert_eq!(result.observables.to_shots(), vec![vec![true]; 3]);
    assert_eq!(result.logical_errors, vec![3]);
}

#[test]
fn qec_reference_runner_applies_pauli_noise() {
    let options = QecOptions {
        shots: 3,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.noise(QecNoise::XError(1.0), &[0]).unwrap();
    program.measure_z(0).unwrap();

    let result = run_qec_program_reference(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 3]);
}

#[test]
fn qec_reference_runner_evaluates_exp_val_on_final_state() {
    let mut program = QecProgram::new(1);
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();

    let result = run_qec_program_reference(&program).unwrap();
    let estimates = result.expectation_values.expect("EXP_VAL estimates");
    assert_eq!(estimates.len(), 1);
    assert!((estimates[0].mean - 1.0).abs() < 1e-12);
    assert_eq!(estimates[0].variance, 0.0);
    assert_eq!(estimates[0].num_shots, program.options().shots);
}

#[test]
fn qec_reference_runner_rejects_zero_chunk_size() {
    let options = QecOptions {
        shots: 1,
        seed: 42,
        chunk_size: Some(0),
        keep_measurements: true,
    };
    let mut program = QecProgram::with_options(1, options);
    program.measure_z(0).unwrap();

    let err = run_qec_program_reference(&program).unwrap_err();
    assert!(format!("{err}").contains("chunk_size"));
}

#[test]
fn qec_module_does_not_change_openqasm_parsing() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let circuit = openqasm::parse(qasm).unwrap();
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.num_classical_bits, 2);
    assert_eq!(circuit.gate_count(), 2);
}

#[test]
fn qec_reset_and_measurement_basis_are_stored() {
    let mut program = QecProgram::new(1);
    program.reset(QecBasis::X, 0).unwrap();
    program.measure(QecBasis::Y, 0).unwrap();

    assert_eq!(program.ops().len(), 2);
}

#[test]
fn qec_text_parser_ingests_detector_mpp_and_exp_val_program() {
    let text = r#"
        # Representative measurement-record program.
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        H 0
        CX 0 1
        M 0
        MX 1
        MPP X0*Z1
        DETECTOR(1, 2, 3) rec[-1] rec[-3]
        OBSERVABLE_INCLUDE(0) rec[-2]
        EXP_VAL(0.5) X0*Z1
        TICK
    "#;

    let program = parse_qec_program(text).unwrap();
    assert_eq!(program.num_qubits(), 2);
    assert_eq!(program.num_measurements(), 3);
    assert_eq!(program.num_detectors(), 1);
    assert_eq!(program.num_observables(), 1);
    assert_eq!(program.detector_rows().unwrap(), vec![vec![2, 0]]);
    assert_eq!(program.observable_rows().unwrap(), vec![vec![1]]);
    assert!(program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::ExpectationValue { coefficient, .. } if (*coefficient - 0.5).abs() < 1e-12)));
    assert!(program.ops().iter().any(|op| matches!(op, QecOp::Tick)));
}

#[test]
fn qec_text_parser_flattens_repeat_blocks() {
    let text = r#"
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
    "#;

    let program = QecProgram::from_text(text).unwrap();
    assert_eq!(program.num_measurements(), 2);
    assert_eq!(program.num_detectors(), 2);
    assert_eq!(program.detector_rows().unwrap(), vec![vec![0], vec![1]]);
}

#[test]
fn qec_text_parser_ingests_noise_and_measure_reset() {
    let text = r#"
        R 0
        MRX 1
        X_ERROR(0.001) 0
        Z_ERROR(0.002) 1
        DEPOLARIZE1(0.003) 0 1
        DEPOLARIZE2(0.004) 0 1
    "#;

    let program = parse_qec_program(text).unwrap();
    assert_eq!(program.num_qubits(), 2);
    assert_eq!(program.num_measurements(), 1);
    assert_eq!(program.num_detectors(), 0);
    assert!(program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::Noise { channel: QecNoise::Depolarize2(p), .. } if (*p - 0.004).abs() < 1e-12)));
}

#[test]
fn qec_text_parser_lowers_basis_measurement_error_args() {
    let mut program = parse_qec_program("M(1.0) 0").unwrap();
    program.set_options(QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: true,
    });

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.measurements.to_shots(), vec![vec![true]; 4]);

    let err = parse_qec_program("MPP(0.1) X0").unwrap_err();
    assert!(format!("{err}").contains("MPP"));
}

#[test]
fn qec_text_parser_ingests_postselection() {
    let mut accepted_program =
        parse_qec_program("X_ERROR(1) 0\nM 0\nPOSTSELECT(1) rec[-1]").unwrap();
    accepted_program.set_options(QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    });
    assert_eq!(accepted_program.postselection_rows().unwrap().len(), 1);
    let accepted = run_qec_program(&accepted_program).unwrap();
    assert_eq!(accepted.accepted_shots, 4);
    assert_eq!(accepted.discarded_shots, 0);

    let mut rejected_program = parse_qec_program("M 0\nPOSTSELECT(1) rec[-1]").unwrap();
    rejected_program.set_options(QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    });
    let rejected = run_qec_program(&rejected_program).unwrap();
    assert_eq!(rejected.accepted_shots, 0);
    assert_eq!(rejected.discarded_shots, 4);

    let err = parse_qec_program("M 0\nPOSTSELECT(2) rec[-1]").unwrap_err();
    assert!(format!("{err}").contains("POSTSELECT"));
}

#[test]
fn qec_text_parser_ingests_empty_and_multi_record_postselection() {
    let mut program =
        parse_qec_program("M 0\nM 1\nPOSTSELECT(0) rec[-1] rec[-2]\nPOSTSELECT").unwrap();
    program.set_options(QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    });
    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.accepted_shots, 4);
    assert_eq!(result.discarded_shots, 0);

    let mut rejected_program = parse_qec_program("M 0\nPOSTSELECT(1)").unwrap();
    rejected_program.set_options(QecOptions {
        shots: 4,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    });
    let rejected = run_qec_program(&rejected_program).unwrap();
    assert_eq!(rejected.accepted_shots, 0);
    assert_eq!(rejected.discarded_shots, 4);
}

#[test]
fn qec_text_parser_skips_zero_probability_noise_annotations() {
    let program = parse_qec_program(
        r#"
        X_ERROR(0) 0
        M(0) 0
        MR(0) 0
        "#,
    )
    .unwrap();
    assert!(
        !program
            .ops()
            .iter()
            .any(|op| matches!(op, QecOp::Noise { .. }))
    );
}

#[test]
fn qec_text_parser_rejects_invalid_noise_probability() {
    for text in ["X_ERROR(-0.1) 0", "X_ERROR(NaN) 0", "DEPOLARIZE1(1.1) 0"] {
        let err = parse_qec_program(text).unwrap_err();
        assert!(format!("{err}").contains("probability"));
    }
}

#[test]
fn qec_text_parser_rejects_out_of_scope_record_refs() {
    let err = parse_qec_program("DETECTOR rec[-1]").unwrap_err();
    assert!(format!("{err}").contains("out of bounds"));
}

#[test]
fn qec_text_parser_rejects_inverted_targets() {
    let err = parse_qec_program("M !0").unwrap_err();
    assert!(format!("{err}").contains("inverted target"));

    let err = parse_qec_program("MPP !X0").unwrap_err();
    assert!(format!("{err}").contains("inverted target"));

    let err = parse_qec_program(
        r#"
        M 0
        DETECTOR !rec[-1]
        "#,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("inverted target"));
}

#[test]
fn qec_text_parser_caps_repeat_expansion() {
    let err = parse_qec_program(
        r#"
        REPEAT 1000001 {
            M 0
        }
        "#,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("expansion exceeds"));
}

#[test]
fn qec_compiles_measurement_records_into_pauli_rows() {
    let mut program = QecProgram::new(3);
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure(QecBasis::Y, 1).unwrap();
    let m2 = program
        .measure_pauli_product(&[QecPauli::x(0), QecPauli::z(2)])
        .unwrap();
    program
        .detector(&[QecRecordRef::absolute(m0), QecRecordRef::absolute(m2)])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m1)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m2)], true)
        .unwrap();

    let compiled = compile_qec_program_rows(&program).unwrap();
    assert_eq!(compiled.num_qubits(), 3);
    assert_eq!(compiled.num_measurements(), 3);
    assert_eq!(compiled.num_detectors(), 1);
    assert_eq!(compiled.num_observables(), 1);
    assert_eq!(compiled.num_postselections(), 1);
    assert_eq!(compiled.packed_row_words(), 1);
    assert_eq!(compiled.measurement_mask_bytes(), 3 * 2 * 8);

    let rows = compiled.measurement_rows();
    assert_eq!(rows[0].terms(), vec![QecPauli::z(0)]);
    assert_eq!(rows[1].pauli_at(1), Some(QecBasis::Y));
    assert_eq!(rows[1].weight(), 1);
    assert_eq!(rows[2].terms(), vec![QecPauli::x(0), QecPauli::z(2)]);
    assert_eq!(rows[2].x_mask(), &[0b001]);
    assert_eq!(rows[2].z_mask(), &[0b100]);
    assert_eq!(compiled.detector_rows(), [vec![0, 2]].as_slice());
    assert_eq!(compiled.observable_rows(), [vec![1]].as_slice());
    assert_eq!(compiled.postselection_rows(), [vec![2]].as_slice());
    assert_eq!(compiled.postselection_expected(), &[true]);
    assert_eq!(
        compiled.postselection_predicates().collect::<Vec<_>>(),
        vec![(vec![2].as_slice(), true)]
    );

    let measurements = PackedShots::from_meas_major(vec![0b101, 0b010, 0b111], 3, 3);
    let detectors = compiled.detector_parities(&measurements).unwrap();
    let observables = compiled.observable_parities(&measurements).unwrap();
    let postselection = compiled.postselection_parities(&measurements).unwrap();
    assert_eq!(detectors.raw_data(), &[0b010]);
    assert_eq!(observables.raw_data(), &[0b010]);
    assert_eq!(postselection.raw_data(), &[0b111]);
}

#[test]
fn qec_row_compiler_rejects_later_stage_features() {
    let mut gate_program = QecProgram::new(1);
    gate_program.push_gate(Gate::H, &[0]).unwrap();
    gate_program.measure_z(0).unwrap();
    let err = compile_qec_program_rows(&gate_program).unwrap_err();
    assert!(format!("{err}").contains("does not lower gates yet"));

    let mut reset_program = QecProgram::new(1);
    reset_program.reset(QecBasis::Z, 0).unwrap();
    reset_program.measure_z(0).unwrap();
    let err = compile_qec_program_rows(&reset_program).unwrap_err();
    assert!(format!("{err}").contains("does not lower resets yet"));

    let mut noisy_program = QecProgram::new(1);
    noisy_program.noise(QecNoise::XError(0.001), &[0]).unwrap();
    noisy_program.measure_z(0).unwrap();
    let err = compile_qec_program_rows(&noisy_program).unwrap_err();
    assert!(format!("{err}").contains("active noise annotations"));

    let mut exp_val_program = QecProgram::new(1);
    exp_val_program
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let err = compile_qec_program_rows(&exp_val_program).unwrap_err();
    assert!(format!("{err}").contains("EXP_VAL"));
}

// Not a multiple of 64, so the last packed word of every row is partial.
const PARITY_SHOTS: usize = 2_017;

#[track_caller]
fn assert_dropped_records_match_kept(program: &QecProgram, chunk_size: Option<usize>, label: &str) {
    let options = QecOptions {
        shots: PARITY_SHOTS,
        seed: 42,
        chunk_size,
        keep_measurements: false,
    };
    let mut program = program.clone();
    program.set_options(options);
    let dropped = run_qec_program(&program).unwrap();
    program.set_options(QecOptions {
        keep_measurements: true,
        ..options
    });
    let kept = run_qec_program(&program).unwrap();

    assert_eq!(dropped.measurements.num_shots(), 0, "{label}");
    assert_eq!(kept.measurements.num_shots(), PARITY_SHOTS, "{label}");
    let detector_rows = program.detector_rows().unwrap();
    let observable_rows = program.observable_rows().unwrap();
    let projected_detectors = kept.measurements.parity_rows(&detector_rows).unwrap();
    let projected_observables = kept.measurements.parity_rows(&observable_rows).unwrap();
    assert_eq!(
        kept.detectors.to_shots(),
        projected_detectors.to_shots(),
        "{label}"
    );
    assert_eq!(
        kept.observables.to_shots(),
        projected_observables.to_shots(),
        "{label}"
    );

    assert_eq!(dropped.total_shots, kept.total_shots, "{label}");
    assert_eq!(
        dropped.detectors.to_shots(),
        kept.detectors.to_shots(),
        "{label}"
    );
    assert_eq!(
        dropped.observables.to_shots(),
        kept.observables.to_shots(),
        "{label}"
    );
    assert_eq!(dropped.accepted_shots, kept.accepted_shots, "{label}");
    assert_eq!(dropped.discarded_shots, kept.discarded_shots, "{label}");
    assert_eq!(dropped.logical_errors, kept.logical_errors, "{label}");
}

fn any_detector_fires(program: &QecProgram) -> bool {
    let mut program = program.clone();
    program.set_options(QecOptions {
        shots: PARITY_SHOTS,
        seed: 42,
        chunk_size: None,
        keep_measurements: false,
    });
    let result = run_qec_program(&program).unwrap();
    result.detectors.to_shots().iter().flatten().any(|&bit| bit)
}

#[track_caller]
fn assert_parities_meas_major(program: &QecProgram, label: &str) {
    let result = run_qec_program(program).unwrap();
    assert_eq!(result.detectors.layout(), ShotLayout::MeasMajor, "{label}");
    assert_eq!(
        result.observables.layout(),
        ShotLayout::MeasMajor,
        "{label}"
    );
}

#[test]
fn qec_dropped_records_match_kept_projection_on_repetition_memory() {
    for distance in [4usize, 6, 10] {
        for noise in [
            QecNoise::Depolarize1(0.0),
            QecNoise::XError(0.05),
            QecNoise::Depolarize1(0.01),
            QecNoise::Depolarize2(0.02),
        ] {
            let program = qec_common::repetition_memory(distance, distance, noise, PARITY_SHOTS);
            if noise.probability() > 0.0 {
                assert!(any_detector_fires(&program), "d{distance} {noise:?}");
            }
            assert_parities_meas_major(&program, &format!("d{distance} {noise:?}"));
            for chunk_size in [None, Some(500)] {
                assert_dropped_records_match_kept(
                    &program,
                    chunk_size,
                    &format!("d{distance} {noise:?} chunk {chunk_size:?}"),
                );
            }
        }
    }
}

#[test]
fn qec_dropped_records_match_kept_projection_on_surface_memory() {
    let data: Vec<usize> = (0..9).collect();
    for noise in [QecNoise::Depolarize1(0.0), QecNoise::Depolarize1(0.01)] {
        let program = qec_common::surface_memory_d3(3, noise, &data, PARITY_SHOTS);
        assert_parities_meas_major(&program, &format!("surface d3 {noise:?}"));
        for chunk_size in [None, Some(500)] {
            assert_dropped_records_match_kept(
                &program,
                chunk_size,
                &format!("surface d3 {noise:?} chunk {chunk_size:?}"),
            );
        }
    }
}

// Memories far deeper than their distance, noiseless: every detector and the observable
// stay fixed at zero, and the dropped-record parities still match the kept projection.
// The bench rows go four times deeper; these depths keep the test near a second.
#[test]
fn qec_deep_noiseless_memories_keep_fixed_parities() {
    let surface_data: Vec<usize> = (0..81).collect();
    let programs = [
        (
            "repetition d15 x 256",
            qec_common::repetition_memory(15, 256, QecNoise::Depolarize1(0.0), PARITY_SHOTS),
        ),
        (
            "rotated surface d9 x 90",
            qec_common::rotated_surface_memory(
                9,
                90,
                QecNoise::Depolarize1(0.0),
                &surface_data,
                PARITY_SHOTS,
            ),
        ),
    ];
    for (label, program) in &programs {
        let result = run_qec_program(program).unwrap();
        assert_eq!(result.detectors.num_shots(), PARITY_SHOTS, "{label}");
        assert!(
            result
                .detectors
                .to_shots()
                .iter()
                .flatten()
                .all(|&bit| !bit),
            "{label}: a detector fired without noise"
        );
        assert!(
            result
                .observables
                .to_shots()
                .iter()
                .flatten()
                .all(|&bit| !bit),
            "{label}: the observable flipped without noise"
        );
        assert_dropped_records_match_kept(program, None, label);
    }
}

// Rounds of X checks make the noiseless records random while every detector stays
// fixed. XError(0.6) takes the per-shot draw path at every chunk size, and chunk 448 is
// a multiple of 64 where 500 is not.
#[test]
fn qec_dropped_records_match_kept_projection_on_rotated_surface_memory() {
    for distance in [3usize, 5] {
        let data: Vec<usize> = (0..distance * distance).collect();
        let pairs = &data[..distance * distance - 1];
        for (noise, targets) in [
            (QecNoise::Depolarize1(0.0), &data[..]),
            (QecNoise::Depolarize1(0.001), &data[..]),
            (QecNoise::Depolarize1(0.01), &data[..]),
            (QecNoise::XError(0.6), &data[..]),
            (QecNoise::Depolarize2(0.02), pairs),
        ] {
            let program = qec_common::rotated_surface_memory(
                distance,
                distance,
                noise,
                targets,
                PARITY_SHOTS,
            );
            if noise.probability() > 0.0 {
                assert!(any_detector_fires(&program), "d{distance} {noise:?}");
            }
            assert_parities_meas_major(&program, &format!("surface d{distance} {noise:?}"));
            for chunk_size in [None, Some(500), Some(448), Some(31)] {
                assert_dropped_records_match_kept(
                    &program,
                    chunk_size,
                    &format!("surface d{distance} {noise:?} chunk {chunk_size:?}"),
                );
            }
        }
    }
}

#[test]
fn qec_dropped_records_match_kept_projection_when_detectors_vary() {
    for noise in ["", "X_ERROR(0.1) 1\n"] {
        let program = parse_qec_program(&format!(
            "H 0\n{noise}M 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n\
             OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]"
        ))
        .unwrap();
        assert!(any_detector_fires(&program), "{noise:?}");
        for chunk_size in [None, Some(500)] {
            assert_dropped_records_match_kept(
                &program,
                chunk_size,
                &format!("random detector {noise:?} chunk {chunk_size:?}"),
            );
        }
    }
}

#[test]
fn qec_dropped_records_match_kept_projection_with_postselection() {
    let program = parse_qec_program(
        "X_ERROR(0.2) 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n\
         POSTSELECT rec[-2]",
    )
    .unwrap();
    for chunk_size in [None, Some(500)] {
        assert_dropped_records_match_kept(
            &program,
            chunk_size,
            &format!("postselected chunk {chunk_size:?}"),
        );
    }
}

// Shot counts around the 8192-shot noise unit: one partial unit, one shot past a unit,
// a 17-shot last unit under the 32-shot per-shot cutoff, and a partial third unit.
const UNIT_EDGE_SHOTS: [usize; 4] = [2_017, 8_193, 16_401, 20_000];

fn run_with(
    program: &QecProgram,
    shots: usize,
    chunk_size: Option<usize>,
    keep_measurements: bool,
) -> QecSampleResult {
    let mut program = program.clone();
    program.set_options(QecOptions {
        shots,
        seed: 42,
        chunk_size,
        keep_measurements,
    });
    run_qec_program(&program).unwrap()
}

#[track_caller]
fn assert_same_parities(actual: &QecSampleResult, expected: &QecSampleResult, label: &str) {
    assert_eq!(
        actual.detectors.to_shots(),
        expected.detectors.to_shots(),
        "{label}"
    );
    assert_eq!(
        actual.observables.to_shots(),
        expected.observables.to_shots(),
        "{label}"
    );
    assert_eq!(actual.accepted_shots, expected.accepted_shots, "{label}");
    assert_eq!(actual.logical_errors, expected.logical_errors, "{label}");
}

fn noisy_unit_fixtures() -> Vec<(String, QecProgram)> {
    let data: Vec<usize> = (0..9).collect();
    vec![
        (
            "repetition d5 depolarize1".to_string(),
            qec_common::repetition_memory(5, 5, QecNoise::Depolarize1(0.01), 1),
        ),
        (
            "repetition d4 depolarize2".to_string(),
            qec_common::repetition_memory(4, 4, QecNoise::Depolarize2(0.02), 1),
        ),
        (
            "surface d3 depolarize1".to_string(),
            qec_common::rotated_surface_memory(3, 3, QecNoise::Depolarize1(0.01), &data, 1),
        ),
        (
            "surface d3 depolarize2".to_string(),
            qec_common::rotated_surface_memory(3, 3, QecNoise::Depolarize2(0.02), &data[..8], 1),
        ),
        (
            "surface d3 x_error 0.6".to_string(),
            qec_common::rotated_surface_memory(3, 3, QecNoise::XError(0.6), &data, 1),
        ),
        (
            "surface d3 twelve rounds".to_string(),
            qec_common::rotated_surface_memory(3, 12, QecNoise::Depolarize1(0.01), &data, 1),
        ),
        (
            "measure-reset repetition with X resets".to_string(),
            parse_qec_program(&measure_reset_repetition(12)).unwrap(),
        ),
    ]
}

// Three data qubits checked by two ancillas that are measured, hit by a depolarize-2
// while measured, and reset through the X basis each round.
fn measure_reset_repetition(rounds: usize) -> String {
    let mut text = String::new();
    for round in 0..rounds {
        text.push_str(
            "DEPOLARIZE1(0.02) 0 1 2
CX 0 3 1 3 1 4 2 4
M 3 4
DEPOLARIZE2(0.02) 3 0
             RX 3
H 3
R 4
Z_ERROR(0.02) 3
",
        );
        text.push_str(if round == 0 {
            "DETECTOR rec[-2]
DETECTOR rec[-1]
"
        } else {
            "DETECTOR rec[-2] rec[-4]
DETECTOR rec[-1] rec[-3]
"
        });
    }
    text.push_str(
        "M 0 1 2
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
         OBSERVABLE_INCLUDE(0) rec[-1]
",
    );
    text
}

#[test]
fn qec_noise_ignores_chunk_size_and_record_path() {
    for (label, program) in noisy_unit_fixtures() {
        for shots in UNIT_EDGE_SHOTS {
            let reference = run_with(&program, shots, None, false);
            assert!(
                reference.logical_errors[0] > 0
                    || reference.detectors.to_shots().iter().flatten().any(|&b| b),
                "{label} {shots}"
            );
            for chunk_size in [None, Some(500), Some(31), Some(8_192), Some(10_007)] {
                for keep in [false, true] {
                    let result = run_with(&program, shots, chunk_size, keep);
                    assert_same_parities(
                        &result,
                        &reference,
                        &format!("{label} shots {shots} chunk {chunk_size:?} keep {keep}"),
                    );
                }
            }
        }
    }
}

#[test]
fn qec_noisy_records_ignore_chunk_size_when_noiseless_records_are_fixed() {
    let program = qec_common::repetition_memory(5, 5, QecNoise::Depolarize1(0.01), 1);
    let reference = run_with(&program, 20_000, None, true);
    for chunk_size in [Some(500), Some(31), Some(10_007)] {
        let result = run_with(&program, 20_000, chunk_size, true);
        assert_eq!(
            result.measurements.to_shots(),
            reference.measurements.to_shots(),
            "chunk {chunk_size:?}"
        );
    }
}

#[test]
fn qec_postselected_noise_ignores_chunk_size() {
    let program = parse_qec_program(
        "X_ERROR(0.2) 0 1\nM 0 1\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n\
         POSTSELECT rec[-2]",
    )
    .unwrap();
    let reference = run_with(&program, 20_000, None, false);
    assert!(reference.discarded_shots > 0);
    for chunk_size in [Some(500), Some(31), Some(10_007)] {
        let result = run_with(&program, 20_000, chunk_size, false);
        assert_same_parities(&result, &reference, &format!("chunk {chunk_size:?}"));
    }
}

#[cfg(feature = "parallel")]
#[test]
fn qec_noise_ignores_thread_count() {
    use prism_q::ThreadPool;

    let data: Vec<usize> = (0..25).collect();
    let program = qec_common::rotated_surface_memory(5, 5, QecNoise::Depolarize1(0.05), &data, 1);
    for keep in [false, true] {
        let reference = run_with(&program, 200_003, None, keep);
        for threads in [1, 4] {
            let pool = ThreadPool::with_threads(threads).unwrap();
            let result = pool.install(|| run_with(&program, 200_003, None, keep));
            assert_same_parities(
                &result,
                &reference,
                &format!("{threads} threads keep {keep}"),
            );
        }
    }
}

// Each detector reads one qubit after one channel, so its rate is that channel's chance of
// flipping a Z measurement: p for X_ERROR, 2p/3 for DEPOLARIZE1, 8p/15 per qubit for
// DEPOLARIZE2. The observable reads both DEPOLARIZE2 qubits, which disagree with
// probability 8p/15. X_ERROR(0.6) takes the per-shot draw path.
#[test]
fn qec_noise_matches_analytic_flip_rates() {
    let shots = 200_003;
    let program = parse_qec_program(
        "X_ERROR(0.03) 0\nDEPOLARIZE1(0.06) 1\nDEPOLARIZE2(0.075) 2 3\nX_ERROR(0.6) 4\n\
         M 0 1 2 3 4\nDETECTOR rec[-5]\nDETECTOR rec[-4]\nDETECTOR rec[-3]\n\
         DETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-3] rec[-2]",
    )
    .unwrap();
    let detector_rates = [0.03, 0.04, 0.04, 0.04, 0.6];
    for keep in [false, true] {
        let result = run_with(&program, shots, Some(10_007), keep);
        let detectors = result.detectors.to_shots();
        for (detector, &p) in detector_rates.iter().enumerate() {
            let fired = detectors.iter().filter(|shot| shot[detector]).count();
            let sigma = (shots as f64 * p * (1.0 - p)).sqrt();
            let deviation = (fired as f64 - shots as f64 * p).abs() / sigma;
            assert!(
                deviation < 5.0,
                "detector {detector} keep {keep}: {fired} fired, {deviation:.2} sigma from {p}"
            );
        }
        let p = 0.04;
        let sigma = (shots as f64 * p * (1.0 - p)).sqrt();
        let deviation = (result.logical_errors[0] as f64 - shots as f64 * p).abs() / sigma;
        assert!(
            deviation < 5.0,
            "observable keep {keep}: {deviation:.2} sigma"
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn fingerprint(words: &[u64]) -> u64 {
    words.iter().fold(0xCBF2_9CE4_8422_2325, |hash, &word| {
        (hash ^ word).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

// Pinned from the sampler before noise moved to per-unit streams. The X checks make the
// chunked noiseless records random, so the measurement fingerprint covers that stream.
// That stream is not the same on every host (the macOS ARM64 runner draws another), so
// the pin holds on x86_64, where it was taken.
#[cfg(target_arch = "x86_64")]
#[test]
fn qec_noiseless_sampling_is_unchanged_by_noise_units() {
    let data: Vec<usize> = (0..9).collect();
    let program = qec_common::rotated_surface_memory(3, 3, QecNoise::Depolarize1(0.0), &data, 1);
    let result = run_with(&program, 20_000, Some(4_096), true);
    let fingerprints = [
        fingerprint(result.measurements.raw_data()),
        fingerprint(result.detectors.raw_data()),
        fingerprint(result.observables.raw_data()),
    ];
    assert_eq!(
        fingerprints,
        [
            6_594_110_323_581_659_221,
            1_070_141_396_434_947_493,
            1_070_141_396_434_947_493
        ]
    );
}
