//! Result types for native QEC sampling: packed measurement, detector, and
//! observable records plus postselection counts and observable estimates.

use crate::error::{PrismError, Result};
use crate::sim::compiled::PackedShots;

/// Packed result of native QEC sampling.
///
/// `accepted_shots + discarded_shots == total_shots`. The struct is
/// `#[non_exhaustive]`: build it through [`Self::new`],
/// [`Self::new_with_total_shots`], or [`Self::empty`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QecSampleResult {
    /// Number of shots requested from the sampler.
    pub total_shots: usize,
    /// Raw measurement records, or a zero-shot buffer when
    /// [`super::QecOptions::keep_measurements`] is `false`.
    pub measurements: PackedShots,
    /// Detector records: one bit per detector per shot.
    pub detectors: PackedShots,
    /// Logical observable records: one bit per observable per shot.
    ///
    /// Analytical T strategies (SPD, CAMPS, tensor network) have no per-shot
    /// stream and synthesize these rows to match `logical_errors`: the one-bits
    /// sit in `[0, accepted_shots)` and the rest up to `total_shots` is padding.
    /// Do not align them shot-for-shot with detector rows on that path.
    pub observables: PackedShots,
    /// Number of shots accepted after postselection (or `total_shots` when no
    /// postselection predicate is present).
    pub accepted_shots: usize,
    /// Number of shots rejected by postselection.
    pub discarded_shots: usize,
    /// Per observable, the count of accepted shots with parity 1.
    pub logical_errors: Vec<u64>,
    /// Per-observable unbiased estimate of `⟨1 - 2·parity⟩ ∈ [-γ^t, +γ^t]` (raw
    /// signed-importance mean) from weighted or analytical T strategies, where the
    /// expectation is not a ratio of bit counts; `None` when only bit counts exist.
    pub observable_expectations: Option<Vec<QecObservableEstimate>>,
    /// Estimates for the program's `EXP_VAL` ops, one per op in op order,
    /// each scaled by the op's coefficient. `None` when the program has no
    /// `EXP_VAL` ops. The reference runner reports the per-shot sample mean
    /// and unbiased sample variance over accepted shots; analytical
    /// strategies report the exact expectation with `variance` carrying
    /// squared truncation weight (0.0 when exact). Zero accepted shots
    /// yield `{mean: 0.0, variance: 0.0, num_shots: 0}`.
    pub expectation_values: Option<Vec<QecObservableEstimate>>,
}

/// Unbiased expectation estimate for one observable under a weighted
/// shot strategy, or for one `EXP_VAL` op (see
/// [`QecSampleResult::expectation_values`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QecObservableEstimate {
    /// Empirical mean of `Re(weight · (1 - 2·parity))` over the shot
    /// stream. For Z-basis Pauli observables on a true probability
    /// distribution this lies in `[-1, +1]`; quasi-probability variance
    /// can push raw shot estimates outside that range.
    pub mean: f64,
    /// Empirical variance of the weighted per-shot contribution.
    pub variance: f64,
    /// Number of shots that contributed to the estimate (excluding
    /// postselection rejections).
    pub num_shots: usize,
}

impl QecSampleResult {
    pub fn empty(num_measurements: usize, num_detectors: usize, num_observables: usize) -> Self {
        Self {
            total_shots: 0,
            measurements: PackedShots::from_meas_major(Vec::new(), 0, num_measurements),
            detectors: PackedShots::from_meas_major(Vec::new(), 0, num_detectors),
            observables: PackedShots::from_meas_major(Vec::new(), 0, num_observables),
            accepted_shots: 0,
            discarded_shots: 0,
            logical_errors: vec![0; num_observables],
            observable_expectations: None,
            expectation_values: None,
        }
    }

    /// Validate and build a result, taking the total from the first buffer that holds shots.
    pub fn new(
        measurements: PackedShots,
        detectors: PackedShots,
        observables: PackedShots,
        accepted_shots: usize,
        discarded_shots: usize,
        logical_errors: Vec<u64>,
    ) -> Result<Self> {
        let total_shots = infer_qec_result_total_shots(&measurements, &detectors, &observables);
        Self::new_with_total_shots(
            total_shots,
            measurements,
            detectors,
            observables,
            accepted_shots,
            discarded_shots,
            logical_errors,
        )
    }

    /// Build a result with an explicit total. `measurements` may be a zero-shot buffer;
    /// the detector and observable buffers must hold `total_shots`.
    pub fn new_with_total_shots(
        total_shots: usize,
        measurements: PackedShots,
        detectors: PackedShots,
        observables: PackedShots,
        accepted_shots: usize,
        discarded_shots: usize,
        logical_errors: Vec<u64>,
    ) -> Result<Self> {
        validate_qec_result_shots("measurement", &measurements, total_shots, true)?;
        validate_qec_result_shots("detector", &detectors, total_shots, false)?;
        validate_qec_result_shots("observable", &observables, total_shots, false)?;

        let accounted = accepted_shots.checked_add(discarded_shots).ok_or_else(|| {
            PrismError::InvalidParameter {
                message: "accepted and discarded shot counts overflow".to_string(),
            }
        })?;
        if accounted != total_shots {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "accepted plus discarded shots must equal {total_shots}, got {accounted}"
                ),
            });
        }
        if logical_errors.len() != observables.num_measurements() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "logical error count length {} does not match {} observables",
                    logical_errors.len(),
                    observables.num_measurements()
                ),
            });
        }
        if let Some((observable, count)) = logical_errors
            .iter()
            .enumerate()
            .find(|(_, count)| **count > accepted_shots as u64)
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "logical error count {count} for observable {observable} exceeds {accepted_shots} accepted shots"
                ),
            });
        }
        Ok(Self {
            total_shots,
            measurements,
            detectors,
            observables,
            accepted_shots,
            discarded_shots,
            logical_errors,
            observable_expectations: None,
            expectation_values: None,
        })
    }

    /// Attach per-observable estimates from quasi-probability T strategies; the length
    /// must equal the observable count.
    pub fn with_observable_expectations(
        mut self,
        estimates: Vec<QecObservableEstimate>,
    ) -> Result<Self> {
        if estimates.len() != self.observables.num_measurements() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "observable expectation length {} does not match {} observables",
                    estimates.len(),
                    self.observables.num_measurements()
                ),
            });
        }
        self.observable_expectations = Some(estimates);
        Ok(self)
    }

    /// Attach `EXP_VAL` estimates, one per op in op order; the length is not checked.
    pub fn with_expectation_values(mut self, estimates: Vec<QecObservableEstimate>) -> Self {
        self.expectation_values = Some(estimates);
        self
    }

    /// Fraction of requested shots accepted after postselection.
    pub fn survivor_rate(&self) -> f64 {
        qec_binomial_rate(self.accepted_shots as u64, self.total_shots)
    }

    /// Per-observable logical-error rates among accepted shots.
    pub fn logical_error_rates(&self) -> Vec<f64> {
        self.logical_errors
            .iter()
            .map(|&count| qec_binomial_rate(count, self.accepted_shots))
            .collect()
    }

    /// Wilson score interval for the survivor rate.
    ///
    /// `z_score` sets the confidence level: `1.959963984540054` gives a two-sided 95
    /// percent interval.
    pub fn survivor_rate_wilson_interval(&self, z_score: f64) -> (f64, f64) {
        qec_wilson_interval(self.accepted_shots as u64, self.total_shots, z_score)
    }

    /// Wilson score intervals for logical-error rates among accepted shots, with `z_score`
    /// as in [`Self::survivor_rate_wilson_interval`].
    pub fn logical_error_rate_wilson_intervals(&self, z_score: f64) -> Vec<(f64, f64)> {
        self.logical_errors
            .iter()
            .map(|&count| qec_wilson_interval(count, self.accepted_shots, z_score))
            .collect()
    }
}

fn qec_binomial_rate(successes: u64, trials: usize) -> f64 {
    if trials == 0 {
        return 0.0;
    }
    successes as f64 / trials as f64
}

fn qec_wilson_interval(successes: u64, trials: usize, z_score: f64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 0.0);
    }

    let n = trials as f64;
    let p = successes as f64 / n;
    let z = z_score.abs();
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let spread = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt() / denom;
    ((center - spread).max(0.0), (center + spread).min(1.0))
}

fn infer_qec_result_total_shots(
    measurements: &PackedShots,
    detectors: &PackedShots,
    observables: &PackedShots,
) -> usize {
    [
        measurements.num_shots(),
        detectors.num_shots(),
        observables.num_shots(),
    ]
    .into_iter()
    .find(|shots| *shots != 0)
    .unwrap_or(0)
}

fn validate_qec_result_shots(
    label: &str,
    shots: &PackedShots,
    total_shots: usize,
    allow_omitted: bool,
) -> Result<()> {
    if shots.num_shots() == total_shots {
        return Ok(());
    }
    if allow_omitted && shots.num_shots() == 0 && shots.raw_data().is_empty() {
        return Ok(());
    }
    Err(PrismError::InvalidParameter {
        message: format!(
            "QEC {label} shot count {} does not match total shots {total_shots}",
            shots.num_shots()
        ),
    })
}
