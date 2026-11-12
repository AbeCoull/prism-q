//! Piecewise-constant propagation of a schedule through the rotating-frame Hamiltonian.

use std::f64::consts::{PI, TAU};

use num_complex::Complex64;

use super::schedule::{Timeline, Update};
use super::{Device, MAX_DIMENSION, PulseError, Schedule};
use crate::gates::spectral::hermitian_eigen;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// A device and schedule checked against each other, ready to step.
///
/// Matrices here are column major, the layout `hermitian_eigen` takes, and every
/// Hamiltonian is stored already multiplied by the segment length, so its entries are
/// dimensionless phases.
pub(super) struct Plan<'a> {
    levels: Vec<usize>,
    strides: Vec<usize>,
    dim: usize,
    timeline: Timeline<'a>,
    substeps: usize,
    period: f64,
    segment: f64,
    site_hz: Vec<f64>,
    frame_hz: f64,
    static_phase: Vec<Complex64>,
}

impl<'a> Plan<'a> {
    pub(super) fn new(
        device: &Device,
        schedule: &'a Schedule,
        substeps: usize,
    ) -> Result<Self, PulseError> {
        let sites = device.sites();
        if sites.is_empty() || sites.len() > 3 {
            return Err(PulseError::SiteCount(sites.len()));
        }
        for (site, transmon) in sites.iter().enumerate() {
            if transmon.levels < 2 {
                return Err(PulseError::TooFewLevels {
                    site,
                    levels: transmon.levels,
                });
            }
            if !transmon.frequency.is_finite() {
                return Err(PulseError::NonFinite {
                    what: "transmon frequency",
                    value: transmon.frequency,
                });
            }
            if !transmon.anharmonicity.is_finite() {
                return Err(PulseError::NonFinite {
                    what: "anharmonicity",
                    value: transmon.anharmonicity,
                });
            }
        }
        if let Some(&(_, _, g)) = device.couplings().iter().find(|c| !c.2.is_finite()) {
            return Err(PulseError::NonFinite {
                what: "coupling",
                value: g,
            });
        }
        let dim = device.dimension();
        if dim > MAX_DIMENSION {
            return Err(PulseError::DimensionTooLarge(dim));
        }
        if substeps == 0 {
            return Err(PulseError::ZeroSubsteps);
        }
        let timeline = schedule.resolve(sites.len())?;

        let levels: Vec<usize> = sites.iter().map(|site| site.levels).collect();
        let strides = device.strides();
        let site_hz: Vec<f64> = sites.iter().map(|site| site.frequency).collect();
        let period = schedule.sample_period();
        let segment = period / substeps as f64;
        let frame_hz = reference_frequency(&timeline, &site_hz);

        let mut static_phase = vec![ZERO; dim * dim];
        for index in 0..dim {
            let mut energy = 0.0;
            for (site, transmon) in sites.iter().enumerate() {
                let n = ((index / strides[site]) % levels[site]) as f64;
                energy += (transmon.frequency - frame_hz) * n
                    + 0.5 * transmon.anharmonicity * n * (n - 1.0);
            }
            static_phase[index * dim + index] = Complex64::new(TAU * energy * segment, 0.0);
        }
        for &(a, b, g) in device.couplings() {
            for index in 0..dim {
                let na = (index / strides[a]) % levels[a];
                let nb = (index / strides[b]) % levels[b];
                if nb == 0 || na + 1 == levels[a] {
                    continue;
                }
                let target = index + strides[a] - strides[b];
                let entry = TAU * g * (((na + 1) * nb) as f64).sqrt() * segment;
                static_phase[index * dim + target] += entry;
                static_phase[target * dim + index] += entry;
            }
        }

        Ok(Plan {
            levels,
            strides,
            dim,
            timeline,
            substeps,
            period,
            segment,
            site_hz,
            frame_hz,
            static_phase,
        })
    }

    pub(super) fn dimension(&self) -> usize {
        self.dim
    }

    /// Full propagator in the device basis, column major, in each site's line frame at
    /// the end of the schedule.
    pub(super) fn propagator(&self) -> Vec<Complex64> {
        let dim = self.dim;
        let mut total = vec![ZERO; dim * dim];
        for index in 0..dim {
            total[index * dim + index] = ONE;
        }
        let lags = self.walk(|step, _| total = multiply(step, &total, dim));
        for row in 0..dim {
            let angle: f64 = (0..self.levels.len())
                .map(|site| lags[site] * ((row / self.strides[site]) % self.levels[site]) as f64)
                .sum();
            let phase = Complex64::from_polar(1.0, angle);
            for column in 0..dim {
                total[column * dim + row] *= phase;
            }
        }
        total
    }

    pub(super) fn populations(&self, initial: &[Complex64]) -> Vec<Vec<f64>> {
        let dim = self.dim;
        let mut state = initial.to_vec();
        let mut rows = Vec::with_capacity(self.timeline.samples);
        self.walk(|step, sample_end| {
            state = (0..dim)
                .map(|row| (0..dim).map(|k| step[k * dim + row] * state[k]).sum())
                .collect();
            if sample_end {
                rows.push(state.iter().map(Complex64::norm_sqr).collect());
            }
        });
        rows
    }

    /// Hand every segment propagator to `step` in time order, flagging the last segment
    /// of each sample, and return each line's frame phase lag at the end, relative to
    /// the integration frame.
    ///
    /// The integration frame rotates every site at one reference frequency, so the
    /// static part of the Hamiltonian is time independent and exact; only the drive
    /// terms are sampled. A line's frame lags it by its accumulated carrier offset minus
    /// its phase shifts, which is the angle the output is rotated by per excitation.
    fn walk(&self, mut step: impl FnMut(&[Complex64], bool)) -> Vec<f64> {
        let dim = self.dim;
        let sites = self.levels.len();
        let mut carrier = self.site_hz.clone();
        let mut phase = vec![0.0f64; sites];
        let mut lag = vec![0.0f64; sites];
        let mut next_update = vec![0usize; sites];
        let mut next_play = vec![0usize; sites];

        let apply_updates =
            |sample: usize, carrier: &mut [f64], phase: &mut [f64], next_update: &mut [usize]| {
                for (site, line) in self.timeline.lines.iter().enumerate() {
                    while let Some(&(at, update)) = line.updates.get(next_update[site]) {
                        if at > sample {
                            break;
                        }
                        match update {
                            Update::Frequency(hz) => carrier[site] = hz,
                            Update::Phase(radians) => phase[site] += radians,
                        }
                        next_update[site] += 1;
                    }
                }
            };

        for sample in 0..self.timeline.samples {
            apply_updates(sample, &mut carrier, &mut phase, &mut next_update);
            for (site, line) in self.timeline.lines.iter().enumerate() {
                while line
                    .plays
                    .get(next_play[site])
                    .is_some_and(|play| play.start + play.len <= sample)
                {
                    next_play[site] += 1;
                }
            }
            for sub in 0..self.substeps {
                let offset = (sub as f64 + 0.5) * self.segment;
                let mut hamiltonian = self.static_phase.clone();
                for (site, line) in self.timeline.lines.iter().enumerate() {
                    let Some(play) = line
                        .plays
                        .get(next_play[site])
                        .filter(|play| play.start <= sample)
                    else {
                        continue;
                    };
                    let t = (sample - play.start) as f64 * self.period + offset;
                    let omega = play.envelope.value(t, play.duration, self.period);
                    let angle =
                        phase[site] - lag[site] - TAU * (carrier[site] - self.frame_hz) * offset;
                    let coefficient = omega * Complex64::from_polar(PI * self.segment, angle);
                    self.add_drive(&mut hamiltonian, site, coefficient);
                }
                step(&exponential(&hamiltonian, dim), sub + 1 == self.substeps);
            }
            for (lag, carrier) in lag.iter_mut().zip(&carrier) {
                *lag += TAU * (carrier - self.frame_hz) * self.period;
            }
        }
        apply_updates(
            self.timeline.samples,
            &mut carrier,
            &mut phase,
            &mut next_update,
        );
        lag.iter()
            .zip(&phase)
            .map(|(lag, phase)| lag - phase)
            .collect()
    }

    /// Add `coefficient a^dagger + conj(coefficient) a` on `site`.
    fn add_drive(&self, hamiltonian: &mut [Complex64], site: usize, coefficient: Complex64) {
        let dim = self.dim;
        let stride = self.strides[site];
        for index in 0..dim {
            let n = (index / stride) % self.levels[site];
            if n + 1 == self.levels[site] {
                continue;
            }
            let raised = coefficient * ((n + 1) as f64).sqrt();
            hamiltonian[index * dim + index + stride] += raised;
            hamiltonian[(index + stride) * dim + index] += raised.conj();
        }
    }
}

/// Carrier of the earliest play, so a schedule driving one carrier integrates with no
/// time dependence beyond its envelopes; site 0's frequency when nothing plays.
fn reference_frequency(timeline: &Timeline<'_>, site_hz: &[f64]) -> f64 {
    let first = timeline
        .lines
        .iter()
        .enumerate()
        .filter_map(|(site, line)| line.plays.first().map(|play| (play.start, site)))
        .min();
    let Some((start, site)) = first else {
        return site_hz[0];
    };
    timeline.lines[site]
        .updates
        .iter()
        .take_while(|&&(at, _)| at <= start)
        .fold(site_hz[site], |hz, &(_, update)| match update {
            Update::Frequency(next) => next,
            Update::Phase(_) => hz,
        })
}

/// `exp(-i H)` for a Hermitian `H` of dimensionless phases, column major.
fn exponential(hamiltonian: &[Complex64], dim: usize) -> Vec<Complex64> {
    let (values, vectors) = hermitian_eigen(hamiltonian, dim);
    let phases: Vec<Complex64> = values
        .iter()
        .map(|&value| Complex64::from_polar(1.0, -value))
        .collect();
    let mut out = vec![ZERO; dim * dim];
    for column in 0..dim {
        for row in 0..dim {
            out[column * dim + row] = (0..dim)
                .map(|k| vectors[k * dim + row] * phases[k] * vectors[k * dim + column].conj())
                .sum();
        }
    }
    out
}

/// `a * b` for column-major square matrices.
pub(super) fn multiply(a: &[Complex64], b: &[Complex64], dim: usize) -> Vec<Complex64> {
    let mut out = vec![ZERO; dim * dim];
    for column in 0..dim {
        for k in 0..dim {
            let scale = b[column * dim + k];
            for row in 0..dim {
                out[column * dim + row] += a[k * dim + row] * scale;
            }
        }
    }
    out
}
