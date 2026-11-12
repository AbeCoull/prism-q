//! Gate-level unitaries and leakage rates compiled from a pulse schedule on a model of one
//! to three transmons, closed system only. Unstable: outside the crate's stability
//! promise, so any item here may change or go in a minor release.

mod compiled;
mod device;
mod integrate;
mod schedule;

pub use compiled::CompiledGate;
pub use device::{Device, Transmon};
pub use schedule::{Envelope, Schedule};

use num_complex::Complex64;

/// Largest product of per-site levels a device may have.
pub const MAX_DIMENSION: usize = 512;

/// Invalid device or schedule input to [`compile`] and [`evolve`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum PulseError {
    #[error("a device needs one to three sites, got {0}")]
    SiteCount(usize),

    #[error("site {site} keeps {levels} levels; a transmon needs at least 2")]
    TooFewLevels { site: usize, levels: usize },

    #[error("the device has dimension {0}, above the limit of {MAX_DIMENSION}")]
    DimensionTooLarge(usize),

    #[error("{what} is not finite: {value}")]
    NonFinite { what: &'static str, value: f64 },

    #[error("the sample period must be finite and positive, got {0} s")]
    SamplePeriod(f64),

    #[error("substeps must be at least 1")]
    ZeroSubsteps,

    #[error("the schedule names site {site}, but the device has {sites} sites")]
    MissingSite { site: usize, sites: usize },

    /// A play or frame update whose time is negative or not a whole number of sample
    /// periods.
    #[error("time {time} s on site {site} is negative or off the sample grid")]
    OffGrid { site: usize, time: f64 },

    #[error("two plays on site {site} overlap at {time} s")]
    Overlap { site: usize, time: f64 },

    #[error("the play at {start} s on site {site} is invalid: {reason}")]
    InvalidPlay {
        site: usize,
        start: f64,
        reason: String,
    },

    /// The computational block is too close to singular for a nearest unitary: some
    /// computational state leaks almost entirely.
    #[error("the computational block has smallest singular value {0:e}; it leaked")]
    Leaked(f64),
}

/// Integrate `schedule` on `device` with `substeps` segments per sample period and
/// project the propagator onto the computational subspace.
///
/// Each site's frame at the end is its drive line's frame, so a net
/// [`Schedule::shift_phase`] shows up in the result as a Z rotation, the virtual Z
/// convention.
pub fn compile(
    device: &Device,
    schedule: &Schedule,
    substeps: usize,
) -> Result<CompiledGate, PulseError> {
    let plan = integrate::Plan::new(device, schedule, substeps)?;
    let propagator = plan.propagator();
    CompiledGate::new(device, propagator)
}

/// Populations of the device basis after each sample period, starting from `initial`.
///
/// Row `k` holds the populations at time `(k + 1) * sample_period`; there is one row per
/// sample up to the end of the last play or frame update.
///
/// # Panics
///
/// Panics if `initial.len()` is not [`Device::dimension`].
pub fn evolve(
    device: &Device,
    schedule: &Schedule,
    initial: &[Complex64],
    substeps: usize,
) -> Result<Vec<Vec<f64>>, PulseError> {
    let plan = integrate::Plan::new(device, schedule, substeps)?;
    assert_eq!(
        initial.len(),
        plan.dimension(),
        "initial state has {} amplitudes, the device has dimension {}",
        initial.len(),
        plan.dimension()
    );
    Ok(plan.populations(initial))
}
