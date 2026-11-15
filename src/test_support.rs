//! Assertions shared by the unit tests.

use num_complex::Complex64;

/// Assert equal lengths and `|actual[i] - expected[i]| < eps` at every index.
#[track_caller]
pub(crate) fn assert_probs_close(actual: &[f64], expected: &[f64], eps: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "probability vector length mismatch"
    );
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() < eps,
            "prob[{i}]: expected {e}, got {a} (diff {})",
            (a - e).abs()
        );
    }
}

/// [`assert_probs_close`] for amplitudes, on the modulus of each difference.
#[track_caller]
pub(crate) fn assert_amps_close(actual: &[Complex64], expected: &[Complex64], eps: f64) {
    assert_eq!(actual.len(), expected.len(), "state vector length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() < eps,
            "amp[{i}]: expected {e}, got {a} (diff {})",
            (a - e).norm()
        );
    }
}
