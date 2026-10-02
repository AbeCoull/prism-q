//! Pauli-noise machinery for the compiled QEC runner: deferred-measurement
//! lowering, backward-propagated sensitivity rows XORed onto packed records,
//! and the density-matrix lowering for noisy `EXP_VAL` estimation.

use super::{
    QecNoise, QecOp, QecPauli, QecProgram, append_basis_to_z_rotation, append_mpp_parity_rotations,
    append_z_to_basis_rotation, ensure_lowered_record_count, qec_lowered_num_qubits,
    qec_non_clifford_error,
};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
#[cfg(feature = "parallel")]
use crate::sim::compiled::SendPtrU64;
use crate::sim::compiled::{
    CompiledSampler, PackedShots, batch_propagate_backward, compile_measurements,
    rng::Xoshiro256PlusPlus, xor_words,
};
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, geometric_sample_xoshiro};
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

    fn push_branch(&mut self, words: impl Iterator<Item = u64>) {
        for (word_idx, word) in words.enumerate() {
            let mut bits = word;
            while bits != 0 {
                self.outputs
                    .push((word_idx * 64 + bits.trailing_zeros() as usize) as u32);
                bits &= bits - 1;
            }
        }
        let end = u32::try_from(self.outputs.len()).expect("parity noise output count fits u32");
        self.branch_offsets.push(end);
    }
}

impl QecCompiledNoiseSampler {
    pub(super) fn noiseless(&self) -> &CompiledSampler {
        &self.noiseless
    }

    /// Project every event's record flips through the linear `project` onto `out_words`
    /// words of detector and observable bits, keeping event order, probabilities, and the
    /// noise stream, and store each branch as the output bits it flips.
    pub(super) fn into_parity_noise(
        self,
        out_words: usize,
        project: impl Fn(&[u64], &mut [u64]),
    ) -> QecParityNoise {
        let mut parity = QecParityNoise {
            events: Vec::with_capacity(self.events.events.len()),
            branch_offsets: vec![0],
            outputs: Vec::new(),
            seed: self.seed,
        };
        let mut x = vec![0u64; out_words];
        let mut z = vec![0u64; out_words];
        for event in self.events.events {
            let first_branch = parity.branch_offsets.len() - 1;
            match event.flips {
                QecNoiseFlips::Single { x_flip, z_flip } => {
                    project(&x_flip, &mut x);
                    project(&z_flip, &mut z);
                    parity.push_branch(z.iter().copied());
                    parity.push_branch(x.iter().zip(&z).map(|(x, z)| x ^ z));
                    parity.push_branch(x.iter().copied());
                }
                QecNoiseFlips::Pair { branch_flips } => {
                    for flips in branch_flips.chunks_exact(branch_flips.len() / 15) {
                        project(flips, &mut x);
                        parity.push_branch(x.iter().copied());
                    }
                }
            }
            parity.events.push(QecParityNoiseEvent {
                draw: event.draw,
                first_branch,
            });
        }
        parity
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

    fn push_single(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        let x_is_zero = x_flip.iter().all(|&w| w == 0);
        let z_is_zero = z_flip.iter().all(|&w| w == 0);
        if px + py + pz == 0.0 || (x_is_zero && z_is_zero) {
            return;
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
        if px + py + pz == 0.0 {
            return;
        }

        self.events.push(QecNoiseSensitivityEvent {
            draw: QecNoiseDraw::single(px, py, pz),
            flips: QecNoiseFlips::Single {
                x_flip: x_flip.to_vec(),
                z_flip: z_flip.to_vec(),
            },
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
        if p == 0.0 {
            return;
        }

        let m_words = q0_x_flip.len();
        let mut branch_flips = Vec::with_capacity(15 * m_words);
        let mut branch = vec![0u64; m_words];
        let mut any = false;
        for sample in 1..=15 {
            let first = sample / 4;
            let second = sample % 4;
            branch.fill(0);
            append_qec_pauli_noise_effect(&mut branch, first, q0_x_flip, q0_z_flip);
            append_qec_pauli_noise_effect(&mut branch, second, q1_x_flip, q1_z_flip);
            any |= branch.iter().any(|&w| w != 0);
            branch_flips.extend_from_slice(&branch);
        }

        if any {
            self.events.push(QecNoiseSensitivityEvent {
                draw: QecNoiseDraw::pair(p),
                flips: QecNoiseFlips::Pair { branch_flips },
            });
        }
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
        final_qubit_aliases,
    })
}

fn qec_assign_fresh_alias(
    circuit: &mut Circuit,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    next_qubit: &mut usize,
    logical_qubit: usize,
) -> usize {
    let alias = *next_qubit;
    aliases[logical_qubit] = alias;
    *next_qubit += 1;
    measured_aliases.push(false);
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

/// Walk the deferred circuit backward and visit every noise event with the
/// record-sensitivity masks at its anchor. `x_packed[q]` / `z_packed[q]` hold
/// the X / Z support on qubit `q` of each record's back-propagated Pauli, one
/// bit per measurement record: a Z fault at the anchor flips the records in
/// `x_packed[q]`, an X fault those in `z_packed[q]`, a Y fault the XOR of the
/// two. Events are visited in reverse circuit order.
pub(super) fn walk_qec_noise_sensitivity(
    deferred: &QecDeferredProgram,
    mut visit: impl FnMut(&QecDeferredNoiseEvent, &[Vec<u64>], &[Vec<u64>]),
) -> Result<()> {
    let num_measurements = deferred.measurement_qubits.len();
    let m_words = num_measurements.div_ceil(64);
    let mut x_packed = vec![vec![0u64; m_words]; deferred.circuit.num_qubits];
    let mut z_packed = vec![vec![0u64; m_words]; deferred.circuit.num_qubits];
    let mut sign_packed = vec![0u64; m_words];

    for (record, &qubit) in deferred.measurement_qubits.iter().enumerate() {
        z_packed[qubit][record / 64] |= 1u64 << (record % 64);
    }

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
        noise_by_position[event.position].push(event.clone());
    }

    for gate_position in (0..gate_count).rev() {
        for event in &noise_by_position[gate_position + 1] {
            visit(event, &x_packed, &z_packed);
        }
        let (gate, targets) = match &deferred.circuit.instructions[gate_position] {
            Instruction::Gate { gate, targets } => (gate, targets.as_slice()),
            _ => {
                return Err(PrismError::InvalidParameter {
                    message: "QEC deferred circuit expected gate before terminal measurements"
                        .to_string(),
                });
            }
        };
        batch_propagate_backward(
            &mut x_packed,
            &mut z_packed,
            &mut sign_packed,
            gate,
            targets,
            m_words,
        );
    }

    for event in &noise_by_position[0] {
        visit(event, &x_packed, &z_packed);
    }

    Ok(())
}

fn push_qec_noise_sensitivity_event(
    event: &QecDeferredNoiseEvent,
    x_packed: &[Vec<u64>],
    z_packed: &[Vec<u64>],
    events: &mut QecNoiseSensitivity,
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
