//! Pauli-noise machinery for the compiled QEC runner: deferred-measurement
//! lowering, backward-propagated sensitivity rows XORed onto packed records,
//! and the density-matrix lowering for noisy `EXP_VAL` estimation.

use super::runner::QecParityProjection;
use super::{
    QecNoise, QecOp, QecPauli, QecProgram, append_basis_to_z_rotation, append_mpp_parity_rotations,
    append_z_to_basis_rotation, ensure_lowered_record_count, qec_lowered_num_qubits,
    qec_non_clifford_error,
};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
#[cfg(feature = "parallel")]
use crate::sim::compiled::SendPtrU64;
use crate::sim::compiled::{
    CompiledSampler, PackedShots, batch_propagate_backward, compile_measurements,
    rng::Xoshiro256PlusPlus, xor_words,
};
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, geometric_sample_xoshiro};
use crate::sim::splitmix64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Shots per noise unit, part of the seeded-output contract. Unit `k` covers shots
/// `[k * QEC_NOISE_UNIT_SHOTS, (k + 1) * QEC_NOISE_UNIT_SHOTS)` and draws from ChaCha
/// stream `k` of the noise seed, so seeded noise depends on neither chunking nor threads.
const QEC_NOISE_UNIT_SHOTS: usize = 8192;

#[cfg(feature = "parallel")]
const QEC_PARALLEL_MIN_UNITS: usize = 4;

/// Expected faults per run below which noise units run on the calling thread. Faults
/// carry the work inside a unit; output rows are allocated and filled outside it.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_MIN_FIRINGS: f64 = 1e4;

#[derive(Clone)]
pub(super) struct QecDeferredNoiseEvent {
    pub(super) channel: QecNoise,
    pub(super) targets: Vec<usize>,
    pub(super) position: usize,
}

pub(super) struct QecDeferredProgram {
    pub(super) circuit: Circuit,
    noise_events: Vec<QecDeferredNoiseEvent>,
    pub(super) measurement_qubits: Vec<usize>,
    /// Gate position at which each alias comes into use: 0 for the initial aliases, the
    /// reset or `MPP` position for a fresh one. An alias is idle in `|0>` before it.
    alias_positions: Vec<usize>,
    /// Final lowered-circuit alias of each program qubit. Resets reassign a
    /// program qubit to a fresh alias, so ops that reference program qubits
    /// at the end of the stream (`EXP_VAL`) must translate through this map.
    pub(super) final_qubit_aliases: Vec<usize>,
}

pub(super) struct QecCompiledNoiseSampler {
    noiseless: CompiledSampler,
    events: QecNoiseSensitivity,
    num_measurements: usize,
    seed: u64,
    split_unit: QecSplitUnit,
}

/// Record flips of the unit a chunk boundary splits, drawn once and sorted by shot so
/// each chunk applies its own slice.
#[derive(Default)]
struct QecSplitUnit {
    unit: Option<usize>,
    flips: Vec<QecRecordFlip>,
}

#[derive(Clone, Copy)]
struct QecRecordFlip {
    shot: u32,
    event: u32,
    branch: u8,
}

/// Compiled noise events whose flips land on measurement-major detector and observable
/// rows instead of measurement records. Branch `b` flips the output rows
/// `outputs[branch_offsets[b]..branch_offsets[b + 1]]`.
pub(super) struct QecParityNoise {
    events: Vec<QecParityNoiseEvent>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
    seed: u64,
}

/// Branches run from `first_branch`: X, Y, Z for a single-qubit event, the 15
/// depolarize-2 branches in `push_pair` order for a pair.
struct QecParityNoiseEvent {
    draw: QecNoiseDraw,
    first_branch: usize,
}

impl QecParityNoise {
    /// Apply noise for all `shots` shots to `rows` of `row_words` words each, drawing
    /// exactly as the record path does.
    pub(super) fn apply(&self, rows: &mut [u64], row_words: usize, shots: usize) {
        let units = shots.div_ceil(QEC_NOISE_UNIT_SHOTS);
        #[cfg(feature = "parallel")]
        if qec_noise_units_in_parallel(units, shots, self.events.iter().map(|event| &event.draw)) {
            let rows_len = rows.len();
            let rows = SendPtrU64(rows.as_mut_ptr());
            (0..units).into_par_iter().for_each(|unit| {
                self.draw_unit(unit, shots, |output, word, bit| {
                    let offset = output * row_words + word;
                    debug_assert!(offset < rows_len);
                    // SAFETY: `output` is below the row count and `word` below `row_words`
                    // (its shot is below `shots`), so `offset` lies in `rows`. Unit `unit`
                    // writes only words `unit * U / 64..(unit + 1) * U / 64` of each row,
                    // with `U = QEC_NOISE_UNIT_SHOTS` a multiple of 64, so no two units
                    // touch the same word.
                    unsafe { rows.xor_word(offset, bit) }
                });
            });
            return;
        }
        for unit in 0..units {
            self.draw_unit(unit, shots, |output, word, bit| {
                rows[output * row_words + word] ^= bit;
            });
        }
    }

    /// Draw unit `unit` of a `shots`-shot run, calling `flip(output, word, bit)` for every
    /// output bit a fault toggles.
    #[inline(always)]
    fn draw_unit(&self, unit: usize, shots: usize, mut flip: impl FnMut(usize, usize, u64)) {
        let first_shot = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (shots - first_shot).min(QEC_NOISE_UNIT_SHOTS);
        let mut rng = qec_noise_unit_rng(self.seed, unit);
        draw_qec_unit(
            &self.events,
            |event| &event.draw,
            &mut rng,
            unit_shots,
            |_, event, shot, branch| {
                let shot = first_shot + shot;
                let word = shot / 64;
                let bit = 1u64 << (shot % 64);
                let branch = event.first_branch + branch;
                let range =
                    self.branch_offsets[branch] as usize..self.branch_offsets[branch + 1] as usize;
                for &output in &self.outputs[range] {
                    flip(output as usize, word, bit);
                }
            },
        );
    }

    fn new(seed: u64) -> Self {
        Self {
            events: Vec::new(),
            branch_offsets: vec![0],
            outputs: Vec::new(),
            seed,
        }
    }

    fn first_branch(&self) -> usize {
        self.branch_offsets.len() - 1
    }

    /// Record one branch's flips from a row over the walk's live columns, as the
    /// outputs those columns hold. Words past the columns (the fingerprint) carry none.
    fn push_branch(&mut self, words: impl Iterator<Item = u64>, output_of_slot: &[u32]) {
        let start = self.outputs.len();
        for (word_idx, word) in words.take(output_of_slot.len() / 64).enumerate() {
            let mut bits = word;
            while bits != 0 {
                let output = output_of_slot[word_idx * 64 + bits.trailing_zeros() as usize];
                debug_assert_ne!(output, u32::MAX, "a set column holds an output");
                self.outputs.push(output);
                bits &= bits - 1;
            }
        }
        self.outputs[start..].sort_unstable();
        let end = u32::try_from(self.outputs.len()).expect("parity noise output count fits u32");
        self.branch_offsets.push(end);
    }
}

impl QecCompiledNoiseSampler {
    #[cfg(test)]
    pub(super) fn noiseless(&self) -> &CompiledSampler {
        &self.noiseless
    }

    /// Sample shots `first_shot..first_shot + num_shots` of a `total_shots`-shot run.
    pub(super) fn sample_measurements_packed(
        &mut self,
        first_shot: usize,
        num_shots: usize,
        total_shots: usize,
    ) -> Result<PackedShots> {
        let measurements = self.sample_noiseless_measurements_packed(num_shots)?;
        self.apply_noise_to_measurements(measurements, first_shot, total_shots)
    }

    pub(super) fn sample_noiseless_measurements_packed(
        &mut self,
        num_shots: usize,
    ) -> Result<PackedShots> {
        self.noiseless.try_sample_bulk_packed(num_shots)
    }

    /// Apply noise to `measurements`, which hold shots from `first_shot` of a
    /// `total_shots`-shot run.
    pub(super) fn apply_noise_to_measurements(
        &mut self,
        measurements: PackedShots,
        first_shot: usize,
        total_shots: usize,
    ) -> Result<PackedShots> {
        let num_shots = measurements.num_shots();
        if measurements.num_measurements() != self.num_measurements {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "QEC noisy sampler expected {} measurement records, got {}",
                    self.num_measurements,
                    measurements.num_measurements()
                ),
            });
        }
        if self.events.is_empty() || num_shots == 0 || self.num_measurements == 0 {
            return Ok(measurements);
        }

        debug_assert!(first_shot + num_shots <= total_shots);
        let m_words = self.num_measurements.div_ceil(64);
        let mut data = measurements.into_shot_major_data();
        self.apply_noise_window(&mut data, m_words, first_shot, total_shots);
        Ok(PackedShots::from_shot_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }

    /// XOR noise into shot-major `data` holding shots from `first_shot`. Units inside the
    /// window draw straight onto their shots; a unit the window splits draws once into
    /// [`QecSplitUnit`] and contributes the flips that land in the window.
    fn apply_noise_window(
        &mut self,
        data: &mut [u64],
        m_words: usize,
        first_shot: usize,
        total_shots: usize,
    ) {
        let end_shot = first_shot + data.len() / m_words;
        let unit_end = |unit: usize| ((unit + 1) * QEC_NOISE_UNIT_SHOTS).min(total_shots);
        let is_split =
            |unit: usize| unit * QEC_NOISE_UNIT_SHOTS < first_shot || unit_end(unit) > end_shot;
        let first_unit = first_shot / QEC_NOISE_UNIT_SHOTS;
        let last_unit = (end_shot - 1) / QEC_NOISE_UNIT_SHOTS;

        if is_split(first_unit) {
            self.apply_split_unit(data, m_words, first_shot, total_shots, first_unit);
        }
        let whole_start = first_unit + usize::from(is_split(first_unit));
        let whole_end = (last_unit + 1 - usize::from(is_split(last_unit))).max(whole_start);
        if whole_start < whole_end {
            let lo = whole_start * QEC_NOISE_UNIT_SHOTS - first_shot;
            let hi = unit_end(whole_end - 1) - first_shot;
            self.apply_whole_units(&mut data[lo * m_words..hi * m_words], m_words, whole_start);
        }
        if last_unit != first_unit && is_split(last_unit) {
            self.apply_split_unit(data, m_words, first_shot, total_shots, last_unit);
        }
    }

    /// Draw units from `first_unit` onto `data`, which holds exactly their shots.
    fn apply_whole_units(&self, data: &mut [u64], m_words: usize, first_unit: usize) {
        let unit_words = QEC_NOISE_UNIT_SHOTS * m_words;
        let events = &self.events.events;
        let draw_unit = |(index, unit_data): (usize, &mut [u64])| {
            let mut rng = qec_noise_unit_rng(self.seed, first_unit + index);
            draw_qec_unit(
                events,
                |event| &event.draw,
                &mut rng,
                unit_data.len() / m_words,
                |_, event, shot, branch| {
                    event.flip(&mut unit_data[shot * m_words..(shot + 1) * m_words], branch)
                },
            );
        };
        #[cfg(feature = "parallel")]
        if qec_noise_units_in_parallel(
            data.len().div_ceil(unit_words),
            data.len() / m_words,
            events.iter().map(|event| &event.draw),
        ) {
            data.par_chunks_mut(unit_words)
                .enumerate()
                .for_each(draw_unit);
            return;
        }
        data.chunks_mut(unit_words).enumerate().for_each(draw_unit);
    }

    /// Apply the flips of unit `unit` that land in the shots `data` holds from `first_shot`.
    fn apply_split_unit(
        &mut self,
        data: &mut [u64],
        m_words: usize,
        first_shot: usize,
        total_shots: usize,
        unit: usize,
    ) {
        let unit_start = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (total_shots - unit_start).min(QEC_NOISE_UNIT_SHOTS);
        let events = &self.events.events;
        let split = &mut self.split_unit;
        if split.unit != Some(unit) {
            split.flips.clear();
            let mut rng = qec_noise_unit_rng(self.seed, unit);
            draw_qec_unit(
                events,
                |event| &event.draw,
                &mut rng,
                unit_shots,
                |event, _, shot, branch| {
                    split.flips.push(QecRecordFlip {
                        shot: shot as u32,
                        event: event as u32,
                        branch: branch as u8,
                    })
                },
            );
            split.flips.sort_unstable_by_key(|flip| flip.shot);
            split.unit = Some(unit);
        }

        let end_shot = first_shot + data.len() / m_words;
        let lo = (first_shot.max(unit_start) - unit_start) as u32;
        let hi = (end_shot.min(unit_start + unit_shots) - unit_start) as u32;
        let start = split.flips.partition_point(|flip| flip.shot < lo);
        for flip in split.flips[start..]
            .iter()
            .take_while(|flip| flip.shot < hi)
        {
            let shot = unit_start + flip.shot as usize - first_shot;
            events[flip.event as usize].flip(
                &mut data[shot * m_words..(shot + 1) * m_words],
                flip.branch as usize,
            );
        }
    }
}

struct QecNoiseSensitivity {
    events: Vec<QecNoiseSensitivityEvent>,
}

impl QecNoiseSensitivity {
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Where the sensitivity walk sends each event: kept as record flips for the record
/// path, or projected onto parities for the direct path. Both see the same decisions.
trait QecNoiseSink {
    fn push_single(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64);
    fn push_pair(
        &mut self,
        q0_x_flip: &[u64],
        q0_z_flip: &[u64],
        q1_x_flip: &[u64],
        q1_z_flip: &[u64],
        p: f64,
    );
}

impl QecNoiseSink for QecNoiseSensitivity {
    fn push_single(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        if let Some((px, py, pz)) = qec_single_noise_rates(x_flip, z_flip, px, py, pz) {
            self.events.push(QecNoiseSensitivityEvent {
                draw: QecNoiseDraw::single(px, py, pz),
                flips: QecNoiseFlips::Single {
                    x_flip: x_flip.to_vec(),
                    z_flip: z_flip.to_vec(),
                },
            });
        }
    }

    fn push_pair(
        &mut self,
        q0_x_flip: &[u64],
        q0_z_flip: &[u64],
        q1_x_flip: &[u64],
        q1_z_flip: &[u64],
        p: f64,
    ) {
        let mut branch_flips = Vec::new();
        if qec_pair_branch_flips(
            q0_x_flip,
            q0_z_flip,
            q1_x_flip,
            q1_z_flip,
            p,
            &mut branch_flips,
        ) {
            self.events.push(QecNoiseSensitivityEvent {
                draw: QecNoiseDraw::pair(p),
                flips: QecNoiseFlips::Pair { branch_flips },
            });
        }
    }
}

/// Branch rates a single-qubit event keeps once branches with equal record flips merge,
/// or `None` when it can flip no record.
fn qec_single_noise_rates(
    x_flip: &[u64],
    z_flip: &[u64],
    px: f64,
    py: f64,
    pz: f64,
) -> Option<(f64, f64, f64)> {
    let x_is_zero = x_flip.iter().all(|&w| w == 0);
    let z_is_zero = z_flip.iter().all(|&w| w == 0);
    if px + py + pz == 0.0 || (x_is_zero && z_is_zero) {
        return None;
    }
    let (px, py, pz) = if x_is_zero {
        (px + py, 0.0, 0.0)
    } else if z_is_zero {
        (0.0, 0.0, py + pz)
    } else if x_flip == z_flip {
        (px + pz, 0.0, 0.0)
    } else {
        (px, py, pz)
    };
    (px + py + pz != 0.0).then_some((px, py, pz))
}

/// Fill `branch_flips` with the record flips of the 15 depolarize-2 branches, and
/// report whether the event fires at all and flips some record.
fn qec_pair_branch_flips(
    q0_x_flip: &[u64],
    q0_z_flip: &[u64],
    q1_x_flip: &[u64],
    q1_z_flip: &[u64],
    p: f64,
    branch_flips: &mut Vec<u64>,
) -> bool {
    if p == 0.0 {
        return false;
    }
    let m_words = q0_x_flip.len();
    branch_flips.clear();
    branch_flips.resize(15 * m_words, 0);
    let mut any = false;
    for (sample, branch) in (1..=15).zip(branch_flips.chunks_exact_mut(m_words.max(1))) {
        append_qec_pauli_noise_effect(branch, sample / 4, q0_x_flip, q0_z_flip);
        append_qec_pauli_noise_effect(branch, sample % 4, q1_x_flip, q1_z_flip);
        any |= branch.iter().any(|&w| w != 0);
    }
    any
}

/// [`QecParityNoise`] built event by event as the sensitivity walk reaches each one.
/// Pushes an event's flips into `noise` as the output indices of its branches,
/// translating the walk's live columns through `output_of_slot`.
struct QecParityNoiseSink<'a> {
    noise: &'a mut QecParityNoise,
    output_of_slot: &'a [u32],
    branch_flips: &'a mut Vec<u64>,
}

impl QecNoiseSink for QecParityNoiseSink<'_> {
    fn push_single(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        let Some((px, py, pz)) = qec_single_noise_rates(x_flip, z_flip, px, py, pz) else {
            return;
        };
        let first_branch = self.noise.first_branch();
        self.noise
            .push_branch(z_flip.iter().copied(), self.output_of_slot);
        self.noise.push_branch(
            x_flip.iter().zip(z_flip).map(|(x, z)| x ^ z),
            self.output_of_slot,
        );
        self.noise
            .push_branch(x_flip.iter().copied(), self.output_of_slot);
        self.noise.events.push(QecParityNoiseEvent {
            draw: QecNoiseDraw::single(px, py, pz),
            first_branch,
        });
    }

    fn push_pair(
        &mut self,
        q0_x_flip: &[u64],
        q0_z_flip: &[u64],
        q1_x_flip: &[u64],
        q1_z_flip: &[u64],
        p: f64,
    ) {
        if !qec_pair_branch_flips(
            q0_x_flip,
            q0_z_flip,
            q1_x_flip,
            q1_z_flip,
            p,
            self.branch_flips,
        ) {
            return;
        }
        let first_branch = self.noise.first_branch();
        for flips in self.branch_flips.chunks_exact(q0_x_flip.len().max(1)) {
            self.noise
                .push_branch(flips.iter().copied(), self.output_of_slot);
        }
        self.noise.events.push(QecParityNoiseEvent {
            draw: QecNoiseDraw::pair(p),
            first_branch,
        });
    }
}

struct QecNoiseSensitivityEvent {
    draw: QecNoiseDraw,
    flips: QecNoiseFlips,
}

enum QecNoiseFlips {
    Single { x_flip: Vec<u64>, z_flip: Vec<u64> },
    Pair { branch_flips: Vec<u64> },
}

impl QecNoiseSensitivityEvent {
    /// XOR fault `branch` into one shot's record words.
    #[inline(always)]
    fn flip(&self, shot_words: &mut [u64], branch: usize) {
        match &self.flips {
            QecNoiseFlips::Single { x_flip, z_flip } => {
                apply_qec_single_noise_branch(shot_words, x_flip, z_flip, branch)
            }
            QecNoiseFlips::Pair { branch_flips } => {
                let m_words = shot_words.len();
                xor_words(
                    shot_words,
                    &branch_flips[branch * m_words..(branch + 1) * m_words],
                );
            }
        }
    }
}

/// One event's draw constants, computed once so every unit restarts the event from them.
#[derive(Clone, Copy)]
enum QecNoiseDraw {
    Single {
        rates: QecSingleNoiseRates,
        /// Branch rates given that the event fires.
        conditional: QecSingleNoiseRates,
        ln_1mp: f64,
    },
    Pair {
        p: f64,
        ln_1mp: f64,
    },
}

impl QecNoiseDraw {
    fn single(px: f64, py: f64, pz: f64) -> Self {
        let p_event = px + py + pz;
        let px_frac = px / p_event;
        let pxy_frac = (px + py) / p_event;
        Self::Single {
            rates: QecSingleNoiseRates { px, py, p_event },
            conditional: QecSingleNoiseRates {
                px: px_frac,
                py: pxy_frac - px_frac,
                p_event: 1.0,
            },
            ln_1mp: (1.0 - p_event).ln(),
        }
    }

    fn pair(p: f64) -> Self {
        Self::Pair {
            p,
            ln_1mp: (1.0 - p).ln(),
        }
    }

    #[cfg(feature = "parallel")]
    fn rate(&self) -> f64 {
        match *self {
            Self::Single { rates, .. } => rates.p_event,
            Self::Pair { p, .. } => p,
        }
    }
}

/// Whether to spread `units` noise units of `shots` shots in total over Rayon. Scheduling
/// only: every unit draws from its own stream into its own shots, so the answer cannot
/// change output.
#[cfg(feature = "parallel")]
fn qec_noise_units_in_parallel<'a>(
    units: usize,
    shots: usize,
    draws: impl Iterator<Item = &'a QecNoiseDraw>,
) -> bool {
    units >= QEC_PARALLEL_MIN_UNITS
        && shots as f64 * draws.map(QecNoiseDraw::rate).sum::<f64>() >= QEC_PARALLEL_MIN_FIRINGS
}

fn qec_noise_unit_rng(seed: u64, unit: usize) -> Xoshiro256PlusPlus {
    let mut seed_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0x51A7_EC01));
    seed_rng.set_stream(unit as u64);
    Xoshiro256PlusPlus::from_chacha(&mut seed_rng)
}

/// Draw every event over one unit of `unit_shots` shots, event-major, calling
/// `flip(event_index, event, shot, branch)` per fault. The record and parity paths both
/// draw through here, so they consume each unit's stream identically.
#[inline(always)]
fn draw_qec_unit<E>(
    events: &[E],
    draw: impl Fn(&E) -> &QecNoiseDraw,
    rng: &mut Xoshiro256PlusPlus,
    unit_shots: usize,
    mut flip: impl FnMut(usize, &E, usize, usize),
) {
    for (index, event) in events.iter().enumerate() {
        match *draw(event) {
            QecNoiseDraw::Single {
                rates,
                conditional,
                ln_1mp,
            } => draw_qec_single_noise(unit_shots, rates, conditional, ln_1mp, rng, |shot, b| {
                flip(index, event, shot, b)
            }),
            QecNoiseDraw::Pair { p, ln_1mp } => {
                draw_qec_pair_noise(unit_shots, p, ln_1mp, rng, |shot, b| {
                    flip(index, event, shot, b)
                })
            }
        }
    }
}

#[derive(Clone, Copy)]
struct QecSingleNoiseRates {
    px: f64,
    py: f64,
    p_event: f64,
}

impl QecSingleNoiseRates {
    /// Fault branch for the uniform draw `r`: 0, 1, 2 for X, Y, Z, `None` for no fault.
    #[inline(always)]
    fn branch(self, r: f64) -> Option<usize> {
        if r < self.px {
            Some(0)
        } else if r < self.px + self.py {
            Some(1)
        } else if r < self.p_event {
            Some(2)
        } else {
            None
        }
    }
}

pub(super) fn compile_qec_noisy_sampler(program: &QecProgram) -> Result<QecCompiledNoiseSampler> {
    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    let events = compile_qec_noise_sensitivity(&deferred)?;
    let noiseless = compile_measurements(&deferred.circuit, program.options().seed)?;
    Ok(QecCompiledNoiseSampler {
        noiseless,
        events,
        num_measurements: deferred.measurement_qubits.len(),
        seed: program.options().seed,
        split_unit: QecSplitUnit::default(),
    })
}

/// Lower a measurement-free QEC program into a circuit plus the matching
/// density-matrix noise model.
///
/// Gates map one to one; a basis reset becomes `Reset` followed by the
/// Z-to-basis rotation. Pauli-noise annotations become [`NoiseEvent`]s on the
/// instruction they follow, which the density-matrix backend applies through
/// its exact one-qubit Kraus and two-qubit depolarizing channels. Noise that
/// precedes every instruction anchors on a barrier so each event has a host.
///
/// Measurements are rejected: they carry a per-shot record stream the mixed
/// state does not hold. Callers route those programs to the reference runner.
pub(super) fn lower_qec_program_to_density_matrix(
    program: &QecProgram,
) -> Result<(Circuit, NoiseModel)> {
    let mut circuit = Circuit::new(program.num_qubits(), 0);
    let mut after_gate: Vec<Vec<NoiseEvent>> = Vec::new();

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => circuit.add_gate(gate.clone(), targets),
            QecOp::Reset { basis, qubit } => {
                circuit.add_reset(*qubit);
                append_z_to_basis_rotation(&mut circuit, *basis, *qubit);
            }
            QecOp::Noise { channel, targets } if channel.probability() > 0.0 => {
                if circuit.instructions.is_empty() {
                    circuit.add_barrier(&[]);
                }
                let anchor = circuit.instructions.len() - 1;
                after_gate.resize_with(circuit.instructions.len(), Vec::new);
                push_density_matrix_noise_events(&mut after_gate[anchor], *channel, targets);
            }
            QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC density-matrix estimator".to_string(),
                    reason: "the density matrix holds no measurement records; \
                             `run_qec_program` routes measuring programs to the reference runner"
                        .to_string(),
                });
            }
            QecOp::Feedforward { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC density-matrix estimator".to_string(),
                    reason: "`FEEDFORWARD` reads a measurement record, and the density matrix \
                             holds none; call `run_qec_program_reference` for such programs"
                        .to_string(),
                });
            }
            QecOp::Noise { .. }
            | QecOp::ExpectationValue { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
        }
    }

    after_gate.resize_with(circuit.instructions.len(), Vec::new);
    let noise = NoiseModel {
        after_gate,
        readout: Vec::new(),
    };
    Ok((circuit, noise))
}

fn push_density_matrix_noise_events(
    events: &mut Vec<NoiseEvent>,
    channel: QecNoise,
    targets: &[usize],
) {
    match channel {
        QecNoise::XError(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent::pauli(q, p, 0.0, 0.0)));
        }
        QecNoise::ZError(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent::pauli(q, 0.0, 0.0, p)));
        }
        QecNoise::Depolarize1(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent {
                channel: NoiseChannel::Depolarizing { p },
                qubits: SmallVec::from_slice(&[q]),
            }));
        }
        QecNoise::Depolarize2(p) => {
            events.extend(targets.chunks_exact(2).map(|pair| NoiseEvent {
                channel: NoiseChannel::TwoQubitDepolarizing { p },
                qubits: SmallVec::from_slice(pair),
            }));
        }
    }
}

pub(super) fn lower_qec_program_to_deferred_circuit(
    program: &QecProgram,
) -> Result<QecDeferredProgram> {
    lower_qec_program_to_deferred_circuit_inner(program, false)
}

/// Variant of [`lower_qec_program_to_deferred_circuit`] that admits non-Clifford
/// gates, for the SPD and CAMPS adapters that evaluate observables analytically over the
/// unitary circuit.
pub(super) fn lower_qec_program_to_deferred_circuit_allowing_non_clifford(
    program: &QecProgram,
) -> Result<QecDeferredProgram> {
    lower_qec_program_to_deferred_circuit_inner(program, true)
}

fn lower_qec_program_to_deferred_circuit_inner(
    program: &QecProgram,
    allow_non_clifford: bool,
) -> Result<QecDeferredProgram> {
    let base_qubits = qec_lowered_num_qubits(program);
    let scratch_qubit = program.num_qubits();
    let mut circuit = Circuit::new(base_qubits, program.num_measurements());
    let mut aliases: Vec<usize> = (0..base_qubits).collect();
    let mut measured_aliases = vec![false; base_qubits];
    let mut alias_positions = vec![0; base_qubits];
    let mut next_qubit = base_qubits;
    let mut next_record = 0usize;
    let mut deferred_measurements = Vec::with_capacity(program.num_measurements());
    let mut noise_events = Vec::new();

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => {
                if !allow_non_clifford && !gate.is_clifford() {
                    return Err(qec_non_clifford_error(gate));
                }
                let mapped = map_qec_deferred_targets(targets, &aliases, &measured_aliases)?;
                circuit.add_gate(gate.clone(), mapped.as_slice());
            }
            QecOp::Measure { basis, qubit } => {
                let alias = qec_deferred_target(*qubit, &aliases, &measured_aliases)?;
                append_basis_to_z_rotation(&mut circuit, *basis, alias);
                deferred_measurements.push((alias, next_record));
                measured_aliases[alias] = true;
                next_record += 1;
            }
            QecOp::MeasurePauliProduct { terms } => {
                let scratch_alias = if measured_aliases[aliases[scratch_qubit]] {
                    qec_assign_fresh_alias(
                        &mut circuit,
                        &mut aliases,
                        &mut measured_aliases,
                        &mut alias_positions,
                        &mut next_qubit,
                        scratch_qubit,
                    )
                } else {
                    aliases[scratch_qubit]
                };

                let mut mapped_terms = Vec::with_capacity(terms.len());
                for term in terms {
                    let alias = qec_deferred_target(term.qubit, &aliases, &measured_aliases)?;
                    mapped_terms.push(QecPauli::new(term.basis, alias));
                }

                append_mpp_parity_rotations(&mut circuit, &mapped_terms, scratch_alias);

                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::Reset { basis, qubit } => {
                let alias = qec_assign_fresh_alias(
                    &mut circuit,
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut next_qubit,
                    *qubit,
                );
                append_z_to_basis_rotation(&mut circuit, *basis, alias);
            }
            QecOp::Noise { channel, targets } => {
                if channel.probability() > 0.0 {
                    push_qec_deferred_noise_events(
                        *channel,
                        targets,
                        &aliases,
                        &measured_aliases,
                        circuit.instructions.len(),
                        &mut noise_events,
                    )?;
                }
            }
            QecOp::Feedforward { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC deferred sampler".to_string(),
                    reason: "deferred measurement sampling evaluates a static map from random \
                             bits to outcomes, which `FEEDFORWARD` makes depend on the sample; \
                             `run_qec_program_reference` executes such programs"
                        .to_string(),
                });
            }
            QecOp::ExpectationValue { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
        }
    }

    ensure_lowered_record_count(program, next_record, "deferred")?;

    let mut measurement_qubits = Vec::with_capacity(deferred_measurements.len());
    for (qubit, classical_bit) in deferred_measurements {
        measurement_qubits.push(qubit);
        circuit.add_measure(qubit, classical_bit);
    }

    let final_qubit_aliases = aliases[..program.num_qubits()].to_vec();
    Ok(QecDeferredProgram {
        circuit,
        noise_events,
        measurement_qubits,
        alias_positions,
        final_qubit_aliases,
    })
}

fn qec_assign_fresh_alias(
    circuit: &mut Circuit,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    alias_positions: &mut Vec<usize>,
    next_qubit: &mut usize,
    logical_qubit: usize,
) -> usize {
    let alias = *next_qubit;
    aliases[logical_qubit] = alias;
    *next_qubit += 1;
    measured_aliases.push(false);
    alias_positions.push(circuit.instructions.len());
    circuit.num_qubits = *next_qubit;
    alias
}

fn map_qec_deferred_targets(
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<SmallVec<[usize; 4]>> {
    let mut mapped = SmallVec::<[usize; 4]>::with_capacity(targets.len());
    for &target in targets {
        mapped.push(qec_deferred_target(target, aliases, measured_aliases)?);
    }
    Ok(mapped)
}

fn qec_deferred_target(
    target: usize,
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<usize> {
    if target >= aliases.len() {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: aliases.len(),
        });
    }
    let alias = aliases[target];
    if measured_aliases[alias] {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC compiled runner".to_string(),
            reason: "compiled QEC runner requires reset before reusing a measured qubit"
                .to_string(),
        });
    }
    Ok(alias)
}

fn push_qec_deferred_noise_events(
    channel: QecNoise,
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
    position: usize,
    noise_events: &mut Vec<QecDeferredNoiseEvent>,
) -> Result<()> {
    match channel {
        QecNoise::XError(_) | QecNoise::ZError(_) | QecNoise::Depolarize1(_) => {
            let mut live_targets = Vec::with_capacity(targets.len());
            for &target in targets {
                if let Some(alias) = qec_deferred_noise_target(target, aliases, measured_aliases)? {
                    live_targets.push(alias);
                }
            }
            if !live_targets.is_empty() {
                noise_events.push(QecDeferredNoiseEvent {
                    channel,
                    targets: live_targets,
                    position,
                });
            }
        }
        QecNoise::Depolarize2(p) => {
            for pair in targets.chunks_exact(2) {
                let first = qec_deferred_noise_target(pair[0], aliases, measured_aliases)?;
                let second = qec_deferred_noise_target(pair[1], aliases, measured_aliases)?;
                match (first, second) {
                    (Some(q0), Some(q1)) => noise_events.push(QecDeferredNoiseEvent {
                        channel,
                        targets: vec![q0, q1],
                        position,
                    }),
                    (Some(q), None) | (None, Some(q)) => {
                        noise_events.push(QecDeferredNoiseEvent {
                            channel: QecNoise::Depolarize1(p * 0.8),
                            targets: vec![q],
                            position,
                        });
                    }
                    (None, None) => {}
                }
            }
        }
    }
    Ok(())
}

fn qec_deferred_noise_target(
    target: usize,
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<Option<usize>> {
    if target >= aliases.len() {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: aliases.len(),
        });
    }
    let alias = aliases[target];
    if measured_aliases[alias] {
        return Ok(None);
    }
    Ok(Some(alias))
}

fn compile_qec_noise_sensitivity(deferred: &QecDeferredProgram) -> Result<QecNoiseSensitivity> {
    let mut events = QecNoiseSensitivity::new();
    walk_qec_noise_sensitivity(deferred, |event, x_packed, z_packed| {
        push_qec_noise_sensitivity_event(event, x_packed, z_packed, &mut events);
    })?;
    Ok(events)
}

/// Compile a noisy program straight onto the detector and observable bits of
/// `projection`: the noiseless pattern and the faults that flip it. `None` when a
/// detector or observable is random in the noiseless circuit, which the record path
/// then samples.
///
/// Events, branch rates and draw order match [`QecCompiledNoiseSampler`], so sampling
/// stays bit-identical to projecting its record flips. Nothing here grows with records
/// times gates or records times events: the walk carries one Pauli row per output
/// still live at its position (see [`QecOutputRows`]), and an event stores the output
/// indices its branches flip.
pub(super) fn compile_qec_parity_noise(
    program: &QecProgram,
    projection: &QecParityProjection,
) -> Result<Option<(Vec<u64>, QecParityNoise)>> {
    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    let mut noise = QecParityNoise::new(program.options().seed);
    let mut branch_flips = Vec::new();
    let mut rows = QecOutputRows::new(&deferred, projection);
    walk_qec_deferred_circuit(&deferred, &mut rows, |event, rows| {
        let mut sink = QecParityNoiseSink {
            noise: &mut noise,
            output_of_slot: &rows.output_of_slot,
            branch_flips: &mut branch_flips,
        };
        push_qec_noise_sensitivity_event(event, &rows.x, &rows.z, &mut sink);
    })?;
    Ok(rows.finish().map(|pattern| (pattern, noise)))
}

/// Walk the deferred circuit backward and visit every noise event with the
/// record-sensitivity masks at its anchor. `x_packed[q]` / `z_packed[q]` hold
/// the X / Z support on target `q` of each record's back-propagated Pauli, one
/// bit per measurement record: a Z fault at the anchor flips the records in
/// `x_packed[q]`, an X fault those in `z_packed[q]`, a Y fault the XOR of the
/// two. Events are visited in reverse circuit order, with targets renumbered to
/// rows of the aliases live at the anchor.
pub(super) fn walk_qec_noise_sensitivity(
    deferred: &QecDeferredProgram,
    mut visit: impl FnMut(&QecDeferredNoiseEvent, &[Vec<u64>], &[Vec<u64>]),
) -> Result<()> {
    let m_words = deferred.measurement_qubits.len().div_ceil(64);
    let mut rows = QecAliasRows::new(deferred, m_words);
    walk_qec_deferred_circuit(deferred, &mut rows, |event, rows| {
        visit(event, &rows.x, &rows.z)
    })
}

/// Rows the backward walk propagates: an X and a Z bitset per live alias, over
/// columns the implementation chooses, plus the sign of each column's Pauli.
trait QecWalkRows {
    /// Row of `alias`, created at the alias's last use where the walk first meets it.
    fn slot(&mut self, alias: usize) -> usize;
    /// Give back the row of `alias` where the alias comes into use.
    fn retire(&mut self, alias: usize);
    fn propagate(&mut self, gate: &Gate, slots: &[usize]);
}

/// Walk the deferred circuit backward over `rows`, visiting each noise event at its
/// anchor with its targets renumbered to rows, and retiring each alias at its creation.
fn walk_qec_deferred_circuit<R: QecWalkRows>(
    deferred: &QecDeferredProgram,
    rows: &mut R,
    mut visit: impl FnMut(&QecDeferredNoiseEvent, &R),
) -> Result<()> {
    let gate_count = deferred
        .circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Gate { .. }))
        .count();
    let mut noise_by_position = vec![Vec::new(); gate_count + 1];
    for event in &deferred.noise_events {
        if event.position > gate_count {
            return Err(PrismError::InvalidParameter {
                message: "QEC noise event position exceeds deferred gate count".to_string(),
            });
        }
        noise_by_position[event.position].push(event);
    }
    let mut created_by_position = vec![Vec::new(); gate_count + 1];
    for (alias, &position) in deferred.alias_positions.iter().enumerate() {
        created_by_position[position].push(alias);
    }

    let mut live_event = QecDeferredNoiseEvent {
        channel: QecNoise::XError(0.0),
        targets: Vec::new(),
        position: 0,
    };
    for position in (0..=gate_count).rev() {
        for event in &noise_by_position[position] {
            live_event.channel = event.channel;
            live_event.position = event.position;
            live_event.targets.clear();
            for &target in &event.targets {
                live_event.targets.push(rows.slot(target));
            }
            visit(&live_event, rows);
        }
        for &alias in &created_by_position[position] {
            rows.retire(alias);
        }
        let Some(gate_position) = position.checked_sub(1) else {
            break;
        };
        let (gate, targets) = match &deferred.circuit.instructions[gate_position] {
            Instruction::Gate { gate, targets } => (gate, targets.as_slice()),
            _ => {
                return Err(PrismError::InvalidParameter {
                    message: "QEC deferred circuit expected gate before terminal measurements"
                        .to_string(),
                });
            }
        };
        let slots: SmallVec<[usize; 4]> = targets.iter().map(|&t| rows.slot(t)).collect();
        rows.propagate(gate, &slots);
    }

    Ok(())
}

/// Sensitivity rows in record space for the aliases the backward walk has reached and
/// not yet retired, one bit per measurement record. An alias gets a row at its last
/// use, seeded with its own record, and gives it back where it comes into use, so the
/// row count stays near the program's qubit count however many aliases resets and
/// `MPP` scratch create.
struct QecAliasRows {
    x: Vec<Vec<u64>>,
    z: Vec<Vec<u64>>,
    sign: Vec<u64>,
    slot_of: Vec<usize>,
    record_of: Vec<usize>,
    free: Vec<usize>,
    m_words: usize,
}

impl QecAliasRows {
    fn new(deferred: &QecDeferredProgram, m_words: usize) -> Self {
        let num_aliases = deferred.circuit.num_qubits;
        let mut record_of = vec![usize::MAX; num_aliases];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            record_of[alias] = record;
        }
        Self {
            x: Vec::new(),
            z: Vec::new(),
            sign: vec![0; m_words],
            slot_of: vec![usize::MAX; num_aliases],
            record_of,
            free: Vec::new(),
            m_words,
        }
    }
}

impl QecWalkRows for QecAliasRows {
    fn slot(&mut self, alias: usize) -> usize {
        if self.slot_of[alias] != usize::MAX {
            return self.slot_of[alias];
        }
        let slot = self.free.pop().unwrap_or_else(|| {
            self.x.push(vec![0; self.m_words]);
            self.z.push(vec![0; self.m_words]);
            self.x.len() - 1
        });
        let record = self.record_of[alias];
        if record != usize::MAX {
            self.z[slot][record / 64] |= 1u64 << (record % 64);
        }
        self.slot_of[alias] = slot;
        slot
    }

    fn retire(&mut self, alias: usize) {
        let slot = std::mem::replace(&mut self.slot_of[alias], usize::MAX);
        if slot == usize::MAX {
            return;
        }
        self.x[slot].fill(0);
        self.z[slot].fill(0);
        self.free.push(slot);
    }

    fn propagate(&mut self, gate: &Gate, slots: &[usize]) {
        batch_propagate_backward(
            &mut self.x,
            &mut self.z,
            &mut self.sign,
            gate,
            slots,
            self.m_words,
        );
    }
}

/// Pauli rows in output space for the aliases the backward walk has reached and not
/// yet retired: one column per detector or observable whose Pauli is live, instead of
/// one per record.
///
/// An output's Pauli is the product of its records' back-propagated measurement
/// operators. A record seeds `Z` on its alias at the alias's last use (an output listing
/// a record twice cancels it), and the product
/// returns to identity, with its sign settled, once the walk passes the gates that join
/// its records: a memory experiment's detector spans about two rounds, so the live
/// columns stay near a round or two of outputs however deep the program runs, and each
/// gate touches that many words. A column whose Pauli is identity is reclaimed, its
/// sign folded into `pattern` first; an output that gets a later seed takes a fresh
/// column, so its pattern bit is the XOR of its columns' signs. An output with `X`
/// support on an alias at the alias's creation, where it sits in `|0>`, is random.
///
/// The record path merges or drops an event's branches by which records they flip,
/// which output columns cannot see once flips cancel in every detector. So every row
/// ends in one more word, the XOR of a 64-bit tag per record it holds in record
/// space: it rides through the same kernel, and its zero and equality tests stand in
/// for the record rows' (a collision has odds of 2^-64 per test), so the compiled
/// events, and the seeded samples, match the record path.
struct QecOutputRows<'a> {
    projection: &'a QecParityProjection,
    x: Vec<Vec<u64>>,
    z: Vec<Vec<u64>>,
    sign: Vec<u64>,
    words: usize,
    alias_slot_of: Vec<usize>,
    record_of: Vec<usize>,
    free_alias_slots: Vec<usize>,
    output_slot_of: Vec<u32>,
    output_of_slot: Vec<u32>,
    free_output_slots: Vec<u32>,
    used: Vec<u64>,
    pattern: Vec<u64>,
    fixed: bool,
}

impl<'a> QecOutputRows<'a> {
    fn new(deferred: &QecDeferredProgram, projection: &'a QecParityProjection) -> Self {
        let num_aliases = deferred.circuit.num_qubits;
        let mut record_of = vec![usize::MAX; num_aliases];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            record_of[alias] = record;
        }
        let mut rows = Self {
            projection,
            x: Vec::new(),
            z: Vec::new(),
            sign: Vec::new(),
            words: 0,
            alias_slot_of: vec![usize::MAX; num_aliases],
            record_of,
            free_alias_slots: Vec::new(),
            output_slot_of: vec![u32::MAX; projection.num_outputs()],
            output_of_slot: Vec::new(),
            free_output_slots: Vec::new(),
            used: Vec::new(),
            pattern: vec![0; projection.words],
            fixed: true,
        };
        rows.sign.push(0);
        rows.widen();
        rows
    }

    /// Column of `output`, reclaiming identity columns and then widening the rows when
    /// none is free.
    fn output_slot(&mut self, output: usize) -> usize {
        let mapped = self.output_slot_of[output];
        if mapped != u32::MAX {
            return mapped as usize;
        }
        if self.free_output_slots.is_empty() {
            self.reclaim_output_slots();
        }
        let slot = self
            .free_output_slots
            .pop()
            .expect("reclaiming leaves a free column");
        self.output_slot_of[output] = slot;
        self.output_of_slot[slot as usize] = output as u32;
        slot as usize
    }

    /// Free every mapped column no row has support on, folding its sign into the
    /// pattern, and widen the rows by a word when that frees less than a word of them.
    fn reclaim_output_slots(&mut self) {
        self.used.clear();
        self.used.resize(self.words, 0);
        for row in self.x.iter().chain(&self.z) {
            for (used, &word) in self.used.iter_mut().zip(row) {
                *used |= word;
            }
        }
        let mut freed = 0;
        for word_idx in 0..self.words {
            let mut idle = !self.used[word_idx];
            while idle != 0 {
                let bit = idle.trailing_zeros() as usize;
                idle &= idle - 1;
                let slot = word_idx * 64 + bit;
                let output = self.output_of_slot[slot];
                if output == u32::MAX {
                    continue;
                }
                self.fold_sign(slot, output);
                self.output_slot_of[output as usize] = u32::MAX;
                self.output_of_slot[slot] = u32::MAX;
                self.free_output_slots.push(slot as u32);
                freed += 1;
            }
        }
        if freed < 64 {
            self.widen();
        }
    }

    /// Add an output word ahead of the fingerprint word of every row.
    fn widen(&mut self) {
        let first = self.words * 64;
        for row in self.x.iter_mut().chain(&mut self.z) {
            row.insert(self.words, 0);
        }
        self.sign.insert(self.words, 0);
        self.words += 1;
        self.output_of_slot.resize(first + 64, u32::MAX);
        self.free_output_slots
            .extend((first..first + 64).rev().map(|slot| slot as u32));
    }

    fn fold_sign(&mut self, slot: usize, output: u32) {
        let sign = (self.sign[slot / 64] >> (slot % 64)) & 1;
        self.sign[slot / 64] &= !(1u64 << (slot % 64));
        self.pattern[output as usize / 64] ^= sign << (output % 64);
    }

    /// The noiseless pattern over the outputs, or `None` when one of them is random.
    fn finish(mut self) -> Option<Vec<u64>> {
        if !self.fixed {
            return None;
        }
        debug_assert!(
            self.x
                .iter()
                .chain(&self.z)
                .flatten()
                .all(|&word| word == 0),
            "every alias retires at its creation"
        );
        for slot in 0..self.words * 64 {
            let output = self.output_of_slot[slot];
            if output != u32::MAX {
                self.fold_sign(slot, output);
            }
        }
        Some(self.pattern)
    }
}

impl QecWalkRows for QecOutputRows<'_> {
    fn slot(&mut self, alias: usize) -> usize {
        if self.alias_slot_of[alias] != usize::MAX {
            return self.alias_slot_of[alias];
        }
        let slot = self.free_alias_slots.pop().unwrap_or_else(|| {
            self.x.push(vec![0; self.words + 1]);
            self.z.push(vec![0; self.words + 1]);
            self.x.len() - 1
        });
        self.alias_slot_of[alias] = slot;
        let record = self.record_of[alias];
        if record != usize::MAX {
            let projection = self.projection;
            for &output in projection.outputs(record) {
                let column = self.output_slot(output);
                self.z[slot][column / 64] ^= 1u64 << (column % 64);
            }
            self.z[slot][self.words] ^= splitmix64(record as u64);
        }
        slot
    }

    fn retire(&mut self, alias: usize) {
        let slot = std::mem::replace(&mut self.alias_slot_of[alias], usize::MAX);
        if slot == usize::MAX {
            return;
        }
        if self.x[slot][..self.words].iter().any(|&word| word != 0) {
            self.fixed = false;
        }
        self.x[slot].fill(0);
        self.z[slot].fill(0);
        self.free_alias_slots.push(slot);
    }

    fn propagate(&mut self, gate: &Gate, slots: &[usize]) {
        batch_propagate_backward(
            &mut self.x,
            &mut self.z,
            &mut self.sign,
            gate,
            slots,
            self.words + 1,
        );
    }
}

fn push_qec_noise_sensitivity_event(
    event: &QecDeferredNoiseEvent,
    x_packed: &[Vec<u64>],
    z_packed: &[Vec<u64>],
    events: &mut impl QecNoiseSink,
) {
    match event.channel {
        QecNoise::XError(p) => {
            for &target in &event.targets {
                events.push_single(&x_packed[target], &z_packed[target], p, 0.0, 0.0);
            }
        }
        QecNoise::ZError(p) => {
            for &target in &event.targets {
                events.push_single(&x_packed[target], &z_packed[target], 0.0, 0.0, p);
            }
        }
        QecNoise::Depolarize1(p) => {
            let branch_p = p / 3.0;
            for &target in &event.targets {
                events.push_single(
                    &x_packed[target],
                    &z_packed[target],
                    branch_p,
                    branch_p,
                    branch_p,
                );
            }
        }
        QecNoise::Depolarize2(p) => {
            for pair in event.targets.chunks_exact(2) {
                events.push_pair(
                    &x_packed[pair[0]],
                    &z_packed[pair[0]],
                    &x_packed[pair[1]],
                    &z_packed[pair[1]],
                    p,
                );
            }
        }
    }
}

/// Draw one single-qubit Pauli event over `num_shots` shots, calling `flip(shot, branch)`
/// per fault with the branch of [`QecSingleNoiseRates::branch`].
#[inline(always)]
fn draw_qec_single_noise(
    num_shots: usize,
    rates: QecSingleNoiseRates,
    conditional: QecSingleNoiseRates,
    ln_1mp: f64,
    rng: &mut Xoshiro256PlusPlus,
    mut flip: impl FnMut(usize, usize),
) {
    if rates.p_event == 0.0 {
        return;
    }

    if rates.p_event >= 0.5 || num_shots < 32 {
        for shot in 0..num_shots {
            if let Some(branch) = rates.branch(rng.next_f64()) {
                flip(shot, branch);
            }
        }
        return;
    }

    let mut shot = geometric_sample_xoshiro(rng, ln_1mp);
    while shot < num_shots {
        if let Some(branch) = conditional.branch(rng.next_f64()) {
            flip(shot, branch);
        }
        shot += 1 + geometric_sample_xoshiro(rng, ln_1mp);
    }
}

fn apply_qec_single_noise_branch(
    shot_words: &mut [u64],
    x_flip: &[u64],
    z_flip: &[u64],
    branch: usize,
) {
    // x_flip / z_flip are the X / Z components of the propagated measurement
    // Pauli at this point in the circuit. A Pauli error flips a measurement
    // record iff it anti-commutes with the propagated Pauli on this qubit:
    // X anti-commutes with Z, Z anti-commutes with X, Y anti-commutes with both.
    match branch {
        0 => xor_words(shot_words, z_flip),
        1 => {
            xor_words(shot_words, x_flip);
            xor_words(shot_words, z_flip);
        }
        _ => xor_words(shot_words, x_flip),
    }
}

/// Draw one depolarize-2 event over `num_shots` shots, calling `flip(shot, branch)` per
/// fault with a uniform branch in `0..15` (see `push_pair` for the order).
#[inline(always)]
fn draw_qec_pair_noise(
    num_shots: usize,
    p: f64,
    ln_1mp: f64,
    rng: &mut Xoshiro256PlusPlus,
    mut flip: impl FnMut(usize, usize),
) {
    if p == 0.0 {
        return;
    }

    if p >= 0.5 || num_shots < 32 {
        for shot in 0..num_shots {
            if rng.next_f64() < p {
                flip(shot, qec_uniform_15(rng));
            }
        }
        return;
    }

    let mut shot = geometric_sample_xoshiro(rng, ln_1mp);
    while shot < num_shots {
        flip(shot, qec_uniform_15(rng));
        shot += 1 + geometric_sample_xoshiro(rng, ln_1mp);
    }
}

pub(super) fn append_qec_pauli_noise_effect(
    branch: &mut [u64],
    pauli: usize,
    x_flip: &[u64],
    z_flip: &[u64],
) {
    match pauli {
        0 => {}
        1 => xor_words(branch, z_flip),
        2 => {
            xor_words(branch, x_flip);
            xor_words(branch, z_flip);
        }
        _ => xor_words(branch, x_flip),
    }
}

#[inline(always)]
fn qec_uniform_15(rng: &mut Xoshiro256PlusPlus) -> usize {
    const BRANCHES: u64 = 15;
    const ZONE: u64 = u64::MAX - (u64::MAX % BRANCHES);
    loop {
        let value = rng.next_u64();
        if value < ZONE {
            return (value % BRANCHES) as usize;
        }
    }
}
