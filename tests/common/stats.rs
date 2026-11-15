//! Binomial bounds for sampled rates and the FNV-1a digest that seeded pins hash with.

/// Assert `observed` is within five binomial standard errors of `p` over `n` shots.
/// The standard error is floored at 1e-3 so a rate at or near 0 or 1 keeps a band.
#[track_caller]
pub fn assert_rate(observed: f64, p: f64, n: usize, label: &str) {
    let tolerance = 5.0 * (p * (1.0 - p) / n as f64).sqrt().max(1e-3);
    assert!(
        (observed - p).abs() <= tolerance,
        "{label}: observed {observed}, expected {p} +- {tolerance}"
    );
}

/// [`assert_rate`] on every outcome of a sampled distribution against the exact one.
#[track_caller]
pub fn assert_rates(sampled: &[f64], exact: &[f64], n: usize, label: &str) {
    assert_eq!(
        sampled.len(),
        exact.len(),
        "{label}: distribution length mismatch"
    );
    for (outcome, (&observed, &p)) in sampled.iter().zip(exact).enumerate() {
        assert_rate(observed, p, n, &format!("{label}: outcome {outcome:#b}"));
    }
}

/// FNV-1a over `values`, one multiply per value whatever its width.
pub fn fnv1a(values: impl IntoIterator<Item = u64>) -> u64 {
    values
        .into_iter()
        .fold(0xCBF2_9CE4_8422_2325, |hash, value| {
            (hash ^ value).wrapping_mul(0x0000_0100_0000_01B3)
        })
}
