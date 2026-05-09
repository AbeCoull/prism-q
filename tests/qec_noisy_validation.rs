#[cfg(feature = "bench-internal")]
use prism_q::compile_qec_profiled_sampler;
use prism_q::{
    parse_qec_program, run_qec_program, run_qec_program_reference, Gate, PackedShots, QecNoise,
    QecOptions, QecProgram, QecRecordRef,
};

const SEED: u64 = 0xDEAD_BEEF;
const STAT_SHOTS: usize = 20_000;

fn qec_options(shots: usize, keep_measurements: bool) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(4096),
        keep_measurements,
    }
}

fn bit_rate(shots: &PackedShots, column: usize) -> f64 {
    let mut count = 0usize;
    for shot in 0..shots.num_shots() {
        count += usize::from(shots.get_bit(shot, column));
    }
    count as f64 / shots.num_shots() as f64
}

#[track_caller]
fn assert_rate_close(label: &str, actual: f64, expected: f64, tolerance: f64) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{label}: actual {actual:.4}, expected {expected:.4}, diff {diff:.4}, tolerance {tolerance:.4}"
    );
}

#[test]
fn qec_noisy_single_qubit_channels_match_expected_rates() {
    let p = 0.25;

    let mut x_program = QecProgram::with_options(1, qec_options(STAT_SHOTS, true));
    x_program.noise(QecNoise::XError(p), &[0]).unwrap();
    x_program.measure_z(0).unwrap();
    let x_result = run_qec_program(&x_program).unwrap();
    assert_rate_close(
        "X_ERROR before MZ",
        bit_rate(&x_result.measurements, 0),
        p,
        0.025,
    );

    let mut z_program = QecProgram::with_options(1, qec_options(STAT_SHOTS, true));
    z_program.push_gate(Gate::H, &[0]).unwrap();
    z_program.noise(QecNoise::ZError(p), &[0]).unwrap();
    z_program.measure_x(0).unwrap();
    let z_result = run_qec_program(&z_program).unwrap();
    assert_rate_close(
        "Z_ERROR before MX",
        bit_rate(&z_result.measurements, 0),
        p,
        0.025,
    );

    let mut depol_program = QecProgram::with_options(1, qec_options(STAT_SHOTS, true));
    depol_program.noise(QecNoise::Depolarize1(p), &[0]).unwrap();
    depol_program.measure_z(0).unwrap();
    let depol_result = run_qec_program(&depol_program).unwrap();
    assert_rate_close(
        "DEPOLARIZE1 before MZ",
        bit_rate(&depol_result.measurements, 0),
        2.0 * p / 3.0,
        0.025,
    );
}

#[test]
fn qec_noisy_depolarize2_matches_marginal_and_detector_rates() {
    let p = 0.30;
    let expected = p * 8.0 / 15.0;
    let mut program = QecProgram::with_options(2, qec_options(STAT_SHOTS, true));
    program.noise(QecNoise::Depolarize2(p), &[0, 1]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_z(1).unwrap();
    program
        .detector(&[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    assert_rate_close(
        "DEPOLARIZE2 first target MZ",
        bit_rate(&result.measurements, 0),
        expected,
        0.025,
    );
    assert_rate_close(
        "DEPOLARIZE2 second target MZ",
        bit_rate(&result.measurements, 1),
        expected,
        0.025,
    );
    assert_rate_close(
        "DEPOLARIZE2 detector parity",
        bit_rate(&result.detectors, 0),
        expected,
        0.025,
    );
}

#[test]
fn qec_noisy_postselection_counts_use_noisy_measurement_records() {
    let p = 0.25;
    let mut program = QecProgram::with_options(1, qec_options(STAT_SHOTS, false));
    program.noise(QecNoise::XError(p), &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], true)
        .unwrap();

    let result = run_qec_program(&program).unwrap();
    let accepted_rate = result.accepted_shots as f64 / result.total_shots as f64;
    assert_rate_close("noisy postselection acceptance", accepted_rate, p, 0.025);
    assert_eq!(
        result.discarded_shots,
        result.total_shots - result.accepted_shots
    );
    assert_eq!(result.logical_errors, vec![result.accepted_shots as u64]);
    assert_eq!(result.measurements.num_shots(), 0);
}

#[test]
fn qec_text_measurement_error_arguments_match_expected_rates() {
    let mut z_program = parse_qec_program("M(0.2) 0").unwrap();
    z_program.set_options(qec_options(STAT_SHOTS, true));
    let z_result = run_qec_program(&z_program).unwrap();
    assert_rate_close(
        "M measurement error",
        bit_rate(&z_result.measurements, 0),
        0.2,
        0.025,
    );

    let mut x_program = parse_qec_program("H 0\nMX(0.2) 0").unwrap();
    x_program.set_options(qec_options(STAT_SHOTS, true));
    let x_result = run_qec_program(&x_program).unwrap();
    assert_rate_close(
        "MX measurement error",
        bit_rate(&x_result.measurements, 0),
        0.2,
        0.025,
    );
}

#[test]
fn qec_noisy_compiled_runner_matches_reference_statistics() {
    let mut program = QecProgram::with_options(2, qec_options(8_000, true));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.noise(QecNoise::XError(0.17), &[0]).unwrap();
    program.noise(QecNoise::ZError(0.11), &[1]).unwrap();
    program.noise(QecNoise::Depolarize2(0.13), &[0, 1]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_z(1).unwrap();
    program
        .detector(&[QecRecordRef::absolute(m0), QecRecordRef::absolute(m1)])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();

    let compiled = run_qec_program(&program).unwrap();
    let reference = run_qec_program_reference(&program).unwrap();
    assert_rate_close(
        "compiled versus reference detector rate",
        bit_rate(&compiled.detectors, 0),
        bit_rate(&reference.detectors, 0),
        0.05,
    );
    assert_rate_close(
        "compiled versus reference observable rate",
        bit_rate(&compiled.observables, 0),
        bit_rate(&reference.observables, 0),
        0.05,
    );
}

#[cfg(feature = "bench-internal")]
#[test]
fn qec_profiled_sampler_stages_match_public_runner_result() {
    let mut program = QecProgram::with_options(1, qec_options(16, false));
    program.noise(QecNoise::XError(1.0), &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program
        .observable_include(0, &[QecRecordRef::absolute(m0)])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], true)
        .unwrap();

    let public_result = run_qec_program(&program).unwrap();
    let mut sampler = compile_qec_profiled_sampler(&program).unwrap();
    assert!(sampler.has_noise());
    assert_eq!(sampler.num_measurements(), 1);
    assert_eq!(sampler.num_detectors(), 1);
    assert_eq!(sampler.num_observables(), 1);
    assert_eq!(sampler.num_postselections(), 1);

    let measurements = sampler
        .sample_noiseless_measurements_packed(program.options().shots)
        .unwrap();
    assert_eq!(measurements.to_shots(), vec![vec![false]; 16]);
    let measurements = sampler.apply_noise_to_measurements(measurements).unwrap();
    let observables = sampler.observable_records(&measurements).unwrap();
    let counts = sampler
        .postselection_and_logical_counts(&measurements, &observables)
        .unwrap();
    let staged_result = sampler.result_from_measurements(measurements).unwrap();

    assert_eq!(counts.accepted_shots, public_result.accepted_shots);
    assert_eq!(counts.discarded_shots, public_result.discarded_shots);
    assert_eq!(counts.logical_errors, public_result.logical_errors);
    assert_eq!(staged_result.measurements.num_shots(), 0);
    assert_eq!(
        staged_result.detectors.to_shots(),
        public_result.detectors.to_shots()
    );
    assert_eq!(
        staged_result.observables.to_shots(),
        public_result.observables.to_shots()
    );
    assert_eq!(staged_result.logical_errors, public_result.logical_errors);
}
