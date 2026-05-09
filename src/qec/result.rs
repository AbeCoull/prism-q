use crate::error::{PrismError, Result};
use crate::sim::compiled::PackedShots;

/// Packed result shell for native QEC sampling.
///
/// `accepted_shots + discarded_shots == total_shots`. `logical_errors[i]` is
/// the number of accepted shots whose `i`-th observable parity is 1. When
/// [`super::QecOptions::keep_measurements`] is `false`, [`Self::measurements`]
/// is returned with zero shots; detector and observable shots are always
/// populated.
#[derive(Debug, Clone)]
pub struct QecSampleResult {
    /// Number of shots requested from the sampler.
    pub total_shots: usize,
    /// Raw measurement records, or a zero-shot buffer when
    /// [`super::QecOptions::keep_measurements`] is `false`.
    pub measurements: PackedShots,
    /// Detector records: one bit per detector per shot.
    pub detectors: PackedShots,
    /// Logical observable records: one bit per observable per shot.
    pub observables: PackedShots,
    /// Number of shots accepted after postselection (or `total_shots` when no
    /// postselection predicate is present).
    pub accepted_shots: usize,
    /// Number of shots rejected by postselection.
    pub discarded_shots: usize,
    /// For each observable, count of accepted shots where the observable
    /// parity equals 1. Length equals the number of observables.
    pub logical_errors: Vec<u64>,
}

impl QecSampleResult {
    /// Create an empty result with zero shots.
    pub fn empty(num_measurements: usize, num_detectors: usize, num_observables: usize) -> Self {
        Self {
            total_shots: 0,
            measurements: PackedShots::from_meas_major(Vec::new(), 0, num_measurements),
            detectors: PackedShots::from_meas_major(Vec::new(), 0, num_detectors),
            observables: PackedShots::from_meas_major(Vec::new(), 0, num_observables),
            accepted_shots: 0,
            discarded_shots: 0,
            logical_errors: vec![0; num_observables],
        }
    }

    /// Create a result and validate packed-shot dimensions.
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

    /// Create a result with an explicit total shot count.
    ///
    /// Raw measurement records may be a zero-shot shape buffer when
    /// `QecOptions::keep_measurements` is false. Detector and observable
    /// buffers must carry the explicit shot count.
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
        })
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
    /// `z_score` controls the confidence level. For example, use
    /// `1.959963984540054` for a two-sided 95 percent interval.
    pub fn survivor_rate_wilson_interval(&self, z_score: f64) -> (f64, f64) {
        qec_wilson_interval(self.accepted_shots as u64, self.total_shots, z_score)
    }

    /// Wilson score intervals for logical-error rates among accepted shots.
    ///
    /// `z_score` controls the confidence level. For example, use
    /// `1.959963984540054` for a two-sided 95 percent interval.
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
