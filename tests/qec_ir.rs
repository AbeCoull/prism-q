use prism_q::circuit::openqasm;
use prism_q::{
    compile_qec_program_rows, parse_qec_program, run_qec_program, run_qec_program_reference, Gate,
    PackedShots, QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram, QecRecordRef,
    QecSampleResult,
};

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
    assert!(program
        .measure_pauli_product(&[QecPauli::x(0), QecPauli::z(0)])
        .is_err());
    assert!(program.noise(QecNoise::XError(1.5), &[0]).is_err());
    assert!(program.noise(QecNoise::Depolarize2(0.001), &[0]).is_err());
    assert!(program
        .noise(QecNoise::Depolarize2(0.001), &[0, 0])
        .is_err());
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
fn qec_compiled_runner_rejects_exp_val_without_reference_fallback() {
    let mut exp_val_program = QecProgram::new(1);
    exp_val_program
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let err = run_qec_program(&exp_val_program).unwrap_err();
    assert!(format!("{err}").contains("EXP_VAL"));
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
fn qec_reference_runner_rejects_exp_val_until_result_schema_lands() {
    let mut program = QecProgram::new(1);
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();

    let err = run_qec_program_reference(&program).unwrap_err();
    assert!(format!("{err}").contains("does not evaluate EXP_VAL yet"));
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
    assert!(!program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::Noise { .. })));
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
