//! Shared helpers for the QEC test targets.

#![allow(dead_code)]

use prism_q::{
    QecObservableEstimate, QecOptions, QecProgram, QecSampleResult, QecTStrategy, run_qec_program,
    run_qec_program_reference,
};

pub const SEED: u64 = 0xDEAD_BEEF;

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
