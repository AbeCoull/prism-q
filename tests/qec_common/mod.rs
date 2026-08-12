//! Shared helpers for the QEC test targets.

#![allow(dead_code)]

use prism_q::{
    QecNoise, QecObservableEstimate, QecOptions, QecPauli, QecProgram, QecRecordRef,
    QecSampleResult, QecTStrategy, run_qec_program, run_qec_program_reference,
};

/// Fixed test seed. `0xDEAD_BEEF` is reserved for bench circuits.
pub const SEED: u64 = 42;

pub const ANALYTICAL_STRATEGIES: [QecTStrategy; 3] =
    [QecTStrategy::Auto, QecTStrategy::Spd, QecTStrategy::Camps];

pub fn qec_options(shots: usize, chunk_size: usize, keep_measurements: bool) -> QecOptions {
    QecOptions {
        shots,
        seed: SEED,
        chunk_size: Some(chunk_size),
        keep_measurements,
    }
}

/// Distance-d repetition-code Z memory: noise on the data qubits before each
/// round of MPP ZZ checks, consecutive-round compare detectors, final data
/// readout detectors, observable on the final `M 0` record.
pub fn repetition_memory(
    distance: usize,
    rounds: usize,
    noise: QecNoise,
    shots: usize,
) -> QecProgram {
    let data: Vec<usize> = (0..distance).collect();
    let mut program = QecProgram::with_options(distance, qec_options(shots, 4096, false));
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program.noise(noise, &data).unwrap();
        let records: Vec<usize> = (0..distance - 1)
            .map(|check| {
                program
                    .measure_pauli_product(&[QecPauli::z(check), QecPauli::z(check + 1)])
                    .unwrap()
            })
            .collect();
        match &prev {
            None => {
                for &record in &records {
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
    let readout: Vec<usize> = (0..distance)
        .map(|qubit| program.measure_z(qubit).unwrap())
        .collect();
    for (check, &prior) in previous.iter().enumerate() {
        program
            .detector(&[
                QecRecordRef::absolute(readout[check]),
                QecRecordRef::absolute(readout[check + 1]),
                QecRecordRef::absolute(prior),
            ])
            .unwrap();
    }
    program
        .observable_include(0, &[QecRecordRef::absolute(readout[0])])
        .unwrap();
    program
}

pub const SURFACE_D3_X_STABILIZERS: [&[usize]; 4] =
    [&[0, 1, 3, 4], &[4, 5, 7, 8], &[2, 5], &[3, 6]];
pub const SURFACE_D3_Z_STABILIZERS: [&[usize]; 4] =
    [&[1, 2, 4, 5], &[3, 4, 6, 7], &[0, 1], &[7, 8]];

/// Distance-3 rotated surface-code Z memory via MPP stabilizer measurements.
/// Round-0 detectors on the Z checks only (deterministic on |0..0>), compare
/// detectors on later rounds, final Z checks reconstructed from the data
/// readout, observable on the left-column logical Z.
pub fn surface_memory_d3(
    rounds: usize,
    noise: QecNoise,
    noise_targets: &[usize],
    shots: usize,
) -> QecProgram {
    let mut program = QecProgram::with_options(9, qec_options(shots, 4096, false));
    let mut prev: Option<Vec<usize>> = None;
    for _ in 0..rounds {
        program.noise(noise, noise_targets).unwrap();
        let mut records = Vec::with_capacity(8);
        for stab in SURFACE_D3_Z_STABILIZERS {
            let terms: Vec<QecPauli> = stab.iter().map(|&q| QecPauli::z(q)).collect();
            records.push(program.measure_pauli_product(&terms).unwrap());
        }
        for stab in SURFACE_D3_X_STABILIZERS {
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
    for (stab, &prior) in SURFACE_D3_Z_STABILIZERS.iter().zip(&previous[..4]) {
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

/// The `EXP_VAL` estimates of a result, which every estimator path attaches.
pub fn estimates(result: &QecSampleResult) -> &[QecObservableEstimate] {
    result
        .expectation_values
        .as_deref()
        .expect("EXP_VAL estimates")
}

pub fn means(result: &QecSampleResult) -> Vec<f64> {
    estimates(result).iter().map(|e| e.mean).collect()
}

/// Variance at or below this carries no sampling information: SPD encodes
/// truncation as squared discarded weight, and a reference run over identical
/// shots leaves only float residue. A genuinely sampled estimate lands orders
/// of magnitude above it.
pub const EXACT_VARIANCE_EPS: f64 = 1e-12;

/// Assert an exact estimator reproduces `expected` and reports no spread.
///
/// Exact paths (the analytical ladder, the density-matrix estimator, and a
/// reference run whose shots are all identical) carry the closed-form value
/// itself, so `tolerance` covers arithmetic only. Callers that require a
/// literal `0.0` variance assert that separately.
pub fn assert_exact_estimates(
    estimates: &[QecObservableEstimate],
    expected: &[f64],
    tolerance: f64,
    label: &str,
) {
    assert_eq!(
        estimates.len(),
        expected.len(),
        "{label}: one estimate per EXP_VAL op in op order"
    );
    for (slot, (estimate, expected)) in estimates.iter().zip(expected).enumerate() {
        assert!(
            (estimate.mean - expected).abs() < tolerance,
            "{label} slot {slot}: {:.14} vs closed form {expected:.14}",
            estimate.mean
        );
        assert!(
            estimate.variance <= EXACT_VARIANCE_EPS,
            "{label} slot {slot}: exact estimate reports spread {:.3e}",
            estimate.variance
        );
    }
}

/// Assert every estimate reports a literal zero variance, the property that
/// separates an exact estimator from a sampled one.
pub fn assert_zero_variance(estimates: &[QecObservableEstimate], label: &str) {
    for (slot, estimate) in estimates.iter().enumerate() {
        assert_eq!(
            estimate.variance, 0.0,
            "{label} slot {slot}: the estimate is exact, not statistical"
        );
    }
}

/// Assert `run_qec_program` reproduced the reference runner exactly, the
/// signature of a program none of the faster routes could absorb. Returns the
/// routed result so callers can check the records it carries.
pub fn assert_matches_reference_runner(program: &QecProgram, label: &str) -> QecSampleResult {
    let routed = run_qec_program(program).unwrap();
    let reference = run_qec_program_reference(program).unwrap();
    assert_eq!(
        estimates(&routed),
        estimates(&reference),
        "{label}: must route to the reference runner"
    );
    routed
}

/// Assert sampled estimates agree with `reference` within their own sampling
/// error, using a 5-sigma band widened by `slack` to absorb the closed form's
/// own rounding and the variance estimate's noise at small shot counts.
pub fn assert_within_sampling_error(
    sampled: &[QecObservableEstimate],
    reference: &[f64],
    slack: f64,
    label: &str,
) {
    assert_eq!(
        sampled.len(),
        reference.len(),
        "{label}: one estimate per EXP_VAL op in op order"
    );
    for (slot, (estimate, reference)) in sampled.iter().zip(reference).enumerate() {
        let sigma = (estimate.variance / estimate.num_shots.max(1) as f64).sqrt();
        let tolerance = 5.0 * sigma + slack;
        assert!(
            (estimate.mean - reference).abs() < tolerance,
            "{label} slot {slot}: sampled {:.6} vs reference {reference:.6} (tol {tolerance:.6})",
            estimate.mean
        );
    }
}
