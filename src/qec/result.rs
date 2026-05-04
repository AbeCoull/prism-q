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
