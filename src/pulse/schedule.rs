//! Drive envelopes and the per-line schedule of plays and frame updates.

use num_complex::Complex64;

use super::PulseError;

/// Waveform of one play, as the complex drive amplitude `Omega(t)` in Hz.
///
/// On a two-level site a constant real `amp` drives Rabi oscillations at `amp` cycles per
/// second. The Gaussian shapes are lifted so they start and end at zero, then rescaled so
/// their peak is `amp`.
#[derive(Debug, Clone, PartialEq)]
pub enum Envelope {
    Constant {
        amp: f64,
    },
    /// Centred on the play, standard deviation `sigma` in seconds.
    Gaussian {
        amp: f64,
        sigma: f64,
    },
    /// The [`Envelope::Gaussian`] plus `i beta` times its time derivative, `beta` in
    /// seconds; `beta = -1 / (2 pi anharmonicity)` cancels leakage to `|2>` to first
    /// order.
    Drag {
        amp: f64,
        sigma: f64,
        beta: f64,
    },
    /// A flat top of `width` seconds centred on the play, with Gaussian edges of
    /// standard deviation `sigma` filling the rest.
    GaussianSquare {
        amp: f64,
        sigma: f64,
        width: f64,
    },
    /// One amplitude per sample period, held constant across it; the length must equal
    /// the play's duration in samples.
    Samples(Vec<Complex64>),
}

impl Envelope {
    fn check(&self, samples: usize) -> Result<(), String> {
        let finite = |name: &str, value: f64| {
            if value.is_finite() {
                Ok(())
            } else {
                Err(format!("{name} is not finite: {value}"))
            }
        };
        let positive = |name: &str, value: f64| {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(format!("{name} must be finite and positive, got {value}"))
            }
        };
        match self {
            Envelope::Constant { amp } => finite("amp", *amp),
            Envelope::Gaussian { amp, sigma } => {
                finite("amp", *amp)?;
                positive("sigma", *sigma)
            }
            Envelope::Drag { amp, sigma, beta } => {
                finite("amp", *amp)?;
                positive("sigma", *sigma)?;
                finite("beta", *beta)
            }
            Envelope::GaussianSquare { amp, sigma, width } => {
                finite("amp", *amp)?;
                positive("sigma", *sigma)?;
                finite("width", *width)?;
                if *width < 0.0 {
                    return Err(format!("width must not be negative, got {width}"));
                }
                Ok(())
            }
            Envelope::Samples(values) => {
                if values.len() != samples {
                    return Err(format!(
                        "{} samples for a play of {samples} sample periods",
                        values.len()
                    ));
                }
                match values.iter().position(|value| !value.is_finite()) {
                    Some(index) => Err(format!("sample {index} is not finite")),
                    None => Ok(()),
                }
            }
        }
    }

    /// `Omega(t)` at `t` seconds into a play of `duration` seconds.
    pub(super) fn value(&self, t: f64, duration: f64, period: f64) -> Complex64 {
        match self {
            Envelope::Constant { amp } => Complex64::new(*amp, 0.0),
            Envelope::Gaussian { amp, sigma } => {
                let half = 0.5 * duration;
                Complex64::new(amp * lifted(t - half, *sigma, half).0, 0.0)
            }
            Envelope::Drag { amp, sigma, beta } => {
                let half = 0.5 * duration;
                let (shape, slope) = lifted(t - half, *sigma, half);
                Complex64::new(amp * shape, amp * beta * slope)
            }
            Envelope::GaussianSquare { amp, sigma, width } => {
                let edge = 0.5 * (duration - width.min(duration));
                let shape = if t < edge {
                    lifted(t - edge, *sigma, edge).0
                } else if t > edge + width {
                    lifted(t - edge - width, *sigma, edge).0
                } else {
                    1.0
                };
                Complex64::new(amp * shape, 0.0)
            }
            Envelope::Samples(values) => {
                let index = ((t / period) as usize).min(values.len() - 1);
                values[index]
            }
        }
    }
}

/// Gaussian of width `sigma` at offset `x`, lifted by its value at `reach` so it is zero
/// there and rescaled to peak at one, with its derivative in `x`.
fn lifted(x: f64, sigma: f64, reach: f64) -> (f64, f64) {
    let gaussian = (-0.5 * (x / sigma).powi(2)).exp();
    let floor = (-0.5 * (reach / sigma).powi(2)).exp();
    let scale = 1.0 / (1.0 - floor);
    (
        (gaussian - floor) * scale,
        -x / (sigma * sigma) * gaussian * scale,
    )
}

#[derive(Debug, Clone, PartialEq)]
struct Play {
    site: usize,
    start: f64,
    duration: f64,
    envelope: Envelope,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Update {
    Frequency(f64),
    Phase(f64),
}

#[derive(Debug, Clone, PartialEq)]
struct FrameUpdate {
    site: usize,
    at: f64,
    update: Update,
}

/// Plays and frame updates on the drive lines of a [`Device`](super::Device), one line per
/// site, on a grid of `sample_period` seconds.
///
/// Every time must be a whole number of sample periods, and plays on one line must not
/// overlap. Each line's frame starts at its site's frequency with phase zero. Nothing is
/// checked until [`compile`](super::compile) or [`evolve`](super::evolve) reads the
/// schedule against a device. The schedule ends with its last play or frame update.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    sample_period: f64,
    plays: Vec<Play>,
    updates: Vec<FrameUpdate>,
}

impl Schedule {
    pub fn new(sample_period: f64) -> Self {
        Schedule {
            sample_period,
            plays: Vec::new(),
            updates: Vec::new(),
        }
    }

    pub fn sample_period(&self) -> f64 {
        self.sample_period
    }

    /// Play `envelope` on `site`'s line from `start` for `duration` seconds.
    pub fn play(
        &mut self,
        site: usize,
        start: f64,
        duration: f64,
        envelope: Envelope,
    ) -> &mut Self {
        self.plays.push(Play {
            site,
            start,
            duration,
            envelope,
        });
        self
    }

    /// Move `site`'s carrier to `hz` from time `at`, keeping its phase continuous.
    pub fn set_frequency(&mut self, site: usize, at: f64, hz: f64) -> &mut Self {
        self.updates.push(FrameUpdate {
            site,
            at,
            update: Update::Frequency(hz),
        });
        self
    }

    /// Multiply the drive on `site`'s line by `e^(i radians)` from time `at`.
    pub fn shift_phase(&mut self, site: usize, at: f64, radians: f64) -> &mut Self {
        self.updates.push(FrameUpdate {
            site,
            at,
            update: Update::Phase(radians),
        });
        self
    }

    pub(super) fn resolve(&self, sites: usize) -> Result<Timeline<'_>, PulseError> {
        let period = self.sample_period;
        if !(period.is_finite() && period > 0.0) {
            return Err(PulseError::SamplePeriod(period));
        }
        let mut lines: Vec<Line<'_>> = (0..sites).map(|_| Line::default()).collect();
        let mut samples = 0usize;

        for play in &self.plays {
            let site = play.site;
            if site >= sites {
                return Err(PulseError::MissingSite { site, sites });
            }
            let start = to_sample(play.start, period).ok_or(PulseError::OffGrid {
                site,
                time: play.start,
            })?;
            let invalid = |reason: String| PulseError::InvalidPlay {
                site,
                start: play.start,
                reason,
            };
            if !(play.duration.is_finite() && play.duration > 0.0) {
                return Err(invalid(format!(
                    "duration must be finite and positive, got {} s",
                    play.duration
                )));
            }
            let len = to_sample(play.duration, period).ok_or_else(|| {
                invalid(format!(
                    "duration {} s is not a whole number of sample periods",
                    play.duration
                ))
            })?;
            if let Envelope::GaussianSquare { width, .. } = play.envelope
                && width > play.duration
            {
                return Err(invalid(format!(
                    "width {width} s exceeds the duration {} s",
                    play.duration
                )));
            }
            play.envelope.check(len).map_err(invalid)?;
            lines[site].plays.push(SlotPlay {
                start,
                len,
                duration: len as f64 * period,
                envelope: &play.envelope,
            });
            samples = samples.max(start + len);
        }

        for update in &self.updates {
            let site = update.site;
            if site >= sites {
                return Err(PulseError::MissingSite { site, sites });
            }
            let at = to_sample(update.at, period).ok_or(PulseError::OffGrid {
                site,
                time: update.at,
            })?;
            let (what, value) = match update.update {
                Update::Frequency(hz) => ("carrier frequency", hz),
                Update::Phase(radians) => ("phase shift", radians),
            };
            if !value.is_finite() {
                return Err(PulseError::NonFinite { what, value });
            }
            lines[site].updates.push((at, update.update));
            samples = samples.max(at);
        }

        for (site, line) in lines.iter_mut().enumerate() {
            line.plays.sort_by_key(|play| play.start);
            line.updates.sort_by_key(|&(at, _)| at);
            if let Some(pair) = line
                .plays
                .windows(2)
                .find(|pair| pair[0].start + pair[0].len > pair[1].start)
            {
                return Err(PulseError::Overlap {
                    site,
                    time: pair[1].start as f64 * period,
                });
            }
        }
        Ok(Timeline { samples, lines })
    }
}

/// `time / period` as a sample index, or `None` when negative, non-finite or off the
/// grid by more than one part in a million of a sample.
fn to_sample(time: f64, period: f64) -> Option<usize> {
    let ratio = time / period;
    let index = ratio.round();
    if !ratio.is_finite() || index < 0.0 || (ratio - index).abs() > 1e-6 {
        return None;
    }
    Some(index as usize)
}

/// A schedule checked against a device: plays sorted and disjoint per line, times in
/// samples.
pub(super) struct Timeline<'a> {
    pub(super) samples: usize,
    pub(super) lines: Vec<Line<'a>>,
}

#[derive(Default)]
pub(super) struct Line<'a> {
    pub(super) plays: Vec<SlotPlay<'a>>,
    /// Sorted by sample, insertion order kept among updates at one sample.
    pub(super) updates: Vec<(usize, Update)>,
}

pub(super) struct SlotPlay<'a> {
    pub(super) start: usize,
    pub(super) len: usize,
    pub(super) duration: f64,
    pub(super) envelope: &'a Envelope,
}
