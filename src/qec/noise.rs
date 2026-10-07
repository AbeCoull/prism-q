//! Pauli-noise machinery for the compiled QEC runner: deferred-measurement
//! lowering, the Pauli frame pass that flips sampled records, the shared noise draw,
//! and the density-matrix lowering for noisy `EXP_VAL` estimation.

#[cfg(test)]
use super::runner::QecParityProjection;
use super::{
    QecBasis, QecNoise, QecOp, QecPauli, QecProgram, append_basis_to_z_rotation,
    append_mpp_parity_rotations, append_z_to_basis_rotation, ensure_lowered_record_count,
    qec_lowered_num_qubits, qec_non_clifford_error,
};
use crate::circuit::{Circuit, GateSink, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
#[cfg(feature = "parallel")]
use crate::sim::compiled::SendPtrU64;
use crate::sim::compiled::{
    CompiledSampler, PackedShots, compile_measurements, pair_rows_mut, rng::Xoshiro256PlusPlus,
    xor_words,
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
const QEC_PARALLEL_MIN_UNITS: usize = 2;

/// Expected faults per run below which noise units run on the calling thread. Faults
/// carry the work inside a unit, including the first touch of each page they land on.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_MIN_FIRINGS: f64 = 1e4;

/// Output rows per bucket when a unit lands its flips: 256 rows of one unit's 128 words
/// are 256 KB, inside the L2 share of one core.
const QEC_SCATTER_BLOCK_ROWS: usize = 256;

/// Drawn flips a unit buffers before landing them, 8 MB of packed entries.
const QEC_SCATTER_FLUSH: usize = 1 << 20;

/// Buffered flips from which a unit lands its buckets on the pool instead of in place.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_LAND_FLIPS: usize = 1 << 15;

#[derive(Clone)]
pub(super) struct QecDeferredNoiseEvent {
    pub(super) channel: QecNoise,
    pub(super) targets: Vec<usize>,
    pub(super) position: usize,
}

/// One gate of the deferred circuit: `targets[1]` is `u32::MAX` for a one-qubit gate.
/// 24 bytes against the 96 of an [`Instruction`], which the lowering and every walk
/// over the gates pay for.
#[derive(Clone)]
pub(super) struct DeferredGate {
    pub(super) gate: Gate,
    targets: [u32; 2],
}

const _: () = assert!(size_of::<DeferredGate>() == 24);

impl DeferredGate {
    #[inline]
    pub(super) fn targets(&self) -> &[u32] {
        &self.targets[..1 + usize::from(self.targets[1] != u32::MAX)]
    }
}

impl GateSink for Vec<DeferredGate> {
    #[inline]
    fn gate(&mut self, gate: Gate, targets: &[usize]) {
        debug_assert!(matches!(targets.len(), 1 | 2));
        self.push(DeferredGate {
            gate,
            targets: [
                targets[0] as u32,
                targets.get(1).map_or(u32::MAX, |&t| t as u32),
            ],
        });
    }
}

/// A QEC program with its measurements deferred: `gates` in program order, then one
/// terminal Z measurement per record on `measurement_qubits`, record order.
pub(super) struct QecDeferredProgram {
    pub(super) gates: Vec<DeferredGate>,
    pub(super) num_aliases: usize,
    pub(super) noise_events: Vec<QecDeferredNoiseEvent>,
    pub(super) measurement_qubits: Vec<usize>,
    /// Gate position at which each alias comes into use: 0 for the initial aliases, the
    /// reset or `MPP` position for a fresh one. An alias is idle in `|0>` before it.
    pub(super) alias_positions: Vec<usize>,
    /// Position at or before which a backward walk first meets each alias (the position
    /// after its last gate, or of its last noise event), `None` for an alias nothing
    /// touches.
    pub(super) last_use: Vec<Option<usize>>,
    /// Final lowered-circuit alias of each program qubit. Resets reassign a
    /// program qubit to a fresh alias, so ops that reference program qubits
    /// at the end of the stream (`EXP_VAL`) must translate through this map.
    pub(super) final_qubit_aliases: Vec<usize>,
}

impl QecDeferredProgram {
    /// The gates followed by the terminal measurements, record `j` on classical bit `j`.
    pub(super) fn to_circuit(&self) -> Circuit {
        let mut circuit = Circuit::new(self.num_aliases, self.measurement_qubits.len());
        circuit
            .instructions
            .reserve_exact(self.gates.len() + self.measurement_qubits.len());
        for gate in &self.gates {
            let targets: SmallVec<[usize; 4]> =
                gate.targets().iter().map(|&t| t as usize).collect();
            circuit.instructions.push(Instruction::Gate {
                gate: gate.gate.clone(),
                targets,
            });
        }
        for (record, &qubit) in self.measurement_qubits.iter().enumerate() {
            circuit.add_measure(qubit, record);
        }
        circuit
    }
}

pub(super) struct QecCompiledNoiseSampler {
    noiseless: CompiledSampler,
    noise: Option<QecRecordNoise>,
    num_measurements: usize,
}

/// Kept events of a program whose records are sampled, applied per unit as a Pauli
/// frame pass over the deferred circuit: a fault lands on the frame of its alias at its
/// anchor, gates push the frame forward, and the X frame of an alias at its last use
/// is the flip of its record.
pub(super) struct QecRecordNoise {
    deferred: QecDeferredProgram,
    /// Kept events in draw order, positions descending.
    events: Vec<QecRecordEvent>,
    /// Aliases the circuit touches, ascending by [`QecDeferredProgram::last_use`].
    aliases_by_last_use: Vec<u32>,
    record_of_alias: Vec<u32>,
    gate_count: usize,
    seed: u64,
}

#[derive(Clone, Copy)]
pub(super) struct QecRecordEvent {
    pub(super) draw: QecNoiseDraw,
    pub(super) position: u32,
    /// Target aliases: one and `u32::MAX` for a single-qubit event, two for a pair.
    pub(super) targets: [u32; 2],
}

/// Compiled noise events whose flips land on measurement-major detector and observable
/// rows instead of measurement records. Branch `b` flips the output rows
/// `outputs[branch_offsets[b]..branch_offsets[b + 1]]`.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct QecParityNoise {
    events: Vec<QecParityNoiseEvent>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
    seed: u64,
}

/// Branches run from `first_branch`: Z, Y, X for a single-qubit event, the 15
/// depolarize-2 branches in `qec_pair_branch_flips` order for a pair.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct QecParityNoiseEvent {
    pub(super) draw: QecNoiseDraw,
    pub(super) first_branch: usize,
}

impl QecParityNoise {
    /// Zeroed rows of `row_words` words for `num_rows` outputs with the noise of all
    /// `shots` shots XORed in, drawn exactly as the record path draws.
    pub(super) fn sample_rows(&self, num_rows: usize, row_words: usize, shots: usize) -> Vec<u64> {
        let mut rows = vec![0u64; num_rows * row_words];
        let units = shots.div_ceil(QEC_NOISE_UNIT_SHOTS);
        if units == 0 || row_words == 0 {
            return rows;
        }
        #[cfg(feature = "parallel")]
        {
            let rows_len = rows.len();
            let rows = SendPtrU64(rows.as_mut_ptr());
            let flip = move |offset: usize, bit: u64| {
                debug_assert!(offset < rows_len);
                // SAFETY: `offset` is an output row's word for a shot of the unit being
                // landed, so it lies in `rows`. Unit `unit` writes only words
                // `unit * U / 64..(unit + 1) * U / 64` of each row, with
                // `U = QEC_NOISE_UNIT_SHOTS` a multiple of 64, so no two units touch the
                // same word, and within a unit each block of rows lands on one thread.
                unsafe { rows.xor_word(offset, bit) }
            };
            let clear = move |offset: usize| {
                debug_assert!(offset < rows_len);
                // SAFETY: `offset` is a word of the unit being landed, in `rows` and
                // written by one thread only, for the reasons given in `flip`.
                unsafe { rows.write_word(offset, 0) }
            };
            if qec_noise_units_in_parallel(
                units,
                shots,
                self.events.iter().map(|event| &event.draw),
            ) {
                (0..units).into_par_iter().for_each(|unit| {
                    self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear)
                });
            } else {
                for unit in 0..units {
                    self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            let cells = std::cell::Cell::from_mut(rows.as_mut_slice()).as_slice_of_cells();
            let flip = |offset: usize, bit: u64| cells[offset].set(cells[offset].get() ^ bit);
            let clear = |offset: usize| cells[offset].set(0);
            for unit in 0..units {
                self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear);
            }
        }
        rows
    }

    /// Draw unit `unit` into one bucket of flips per block of output rows, then land the
    /// buckets, so the words a bucket's flips touch stay in cache: the rows span far more
    /// memory than the cache and a fault's outputs are spread over all of it.
    fn scatter_unit(
        &self,
        unit: usize,
        shots: usize,
        row_words: usize,
        num_rows: usize,
        flip: &impl LandFlip,
        clear: &impl LandClear,
    ) {
        let first_word = unit * QEC_NOISE_UNIT_SHOTS / 64;
        let span = UnitSpan {
            first_word,
            words: (row_words - first_word).min(QEC_NOISE_UNIT_SHOTS / 64),
            row_words,
            num_rows,
        };
        let mut buckets: Vec<LandBucket> = (0..num_rows.div_ceil(QEC_SCATTER_BLOCK_ROWS))
            .map(|_| LandBucket::default())
            .collect();
        let mut buffered = 0usize;
        self.draw_unit(unit, shots, |output, word, bit| {
            buckets[output / QEC_SCATTER_BLOCK_ROWS].flips.push(
                ((output as u64) << 32)
                    | (((word - first_word) as u64) << 6)
                    | u64::from(bit.trailing_zeros()),
            );
            buffered += 1;
            if buffered == QEC_SCATTER_FLUSH {
                land_buckets(&mut buckets, span, flip, clear);
                buffered = 0;
            }
        });
        land_buckets(&mut buckets, span, flip, clear);
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

    #[cfg(test)]
    pub(super) fn event_count(&self) -> usize {
        self.events.len()
    }

    pub(super) fn from_parts(
        seed: u64,
        events: Vec<QecParityNoiseEvent>,
        branch_offsets: Vec<u32>,
        outputs: Vec<u32>,
    ) -> Self {
        Self {
            events,
            branch_offsets,
            outputs,
            seed,
        }
    }
}

/// `flip(offset, bit)` XORs `bit` into word `offset` of the rows, shared across the
/// pool's threads when there is one, so a unit's blocks land in parallel.
#[cfg(feature = "parallel")]
trait LandFlip: Fn(usize, u64) + Sync {}
#[cfg(feature = "parallel")]
impl<F: Fn(usize, u64) + Sync> LandFlip for F {}
#[cfg(not(feature = "parallel"))]
trait LandFlip: Fn(usize, u64) {}
#[cfg(not(feature = "parallel"))]
impl<F: Fn(usize, u64)> LandFlip for F {}

/// `clear(offset)` writes zero to word `offset` of the rows without reading it.
#[cfg(feature = "parallel")]
trait LandClear: Fn(usize) + Sync {}
#[cfg(feature = "parallel")]
impl<F: Fn(usize) + Sync> LandClear for F {}
#[cfg(not(feature = "parallel"))]
trait LandClear: Fn(usize) {}
#[cfg(not(feature = "parallel"))]
impl<F: Fn(usize)> LandClear for F {}

/// Words `first_word..first_word + words` of each of `num_rows` rows of `row_words`
/// words: the part of the rows one noise unit writes.
#[derive(Clone, Copy)]
struct UnitSpan {
    first_word: usize,
    words: usize,
    row_words: usize,
    num_rows: usize,
}

/// Flips drawn for one block of rows, packed `output << 32 | unit_word << 6 | bit`, and
/// whether the block's span has been cleared.
#[derive(Default)]
struct LandBucket {
    flips: Vec<u64>,
    cleared: bool,
}

/// Land every bucket's flips, a bucket per thread when there is a pool and enough of
/// them, then empty the buckets. Before its first flip lands, a block's span is cleared
/// in row order: the rows are freshly zeroed, so this only faults the pages in, and a
/// page faulted in by a plain write costs less than one faulted in by a flip's read.
fn land_buckets(
    buckets: &mut [LandBucket],
    span: UnitSpan,
    flip: &impl LandFlip,
    clear: &impl LandClear,
) {
    let land_bucket = |block: usize, bucket: &mut LandBucket| {
        if bucket.flips.is_empty() {
            return;
        }
        if !bucket.cleared {
            bucket.cleared = true;
            let first_row = block * QEC_SCATTER_BLOCK_ROWS;
            for row in first_row..(first_row + QEC_SCATTER_BLOCK_ROWS).min(span.num_rows) {
                let base = row * span.row_words + span.first_word;
                clear(base);
                clear(base + span.words - 1);
            }
        }
        for &packed in &bucket.flips {
            let output = (packed >> 32) as usize;
            let word = ((packed >> 6) & 0x03FF_FFFF) as usize;
            flip(
                output * span.row_words + span.first_word + word,
                1u64 << (packed & 63),
            );
        }
        bucket.flips.clear();
    };
    #[cfg(feature = "parallel")]
    if buckets
        .iter()
        .map(|bucket| bucket.flips.len())
        .sum::<usize>()
        >= QEC_PARALLEL_LAND_FLIPS
    {
        buckets
            .par_iter_mut()
            .enumerate()
            .for_each(|(block, bucket)| land_bucket(block, bucket));
        return;
    }
    for (block, bucket) in buckets.iter_mut().enumerate() {
        land_bucket(block, bucket);
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
        &self,
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
        let Some(noise) = &self.noise else {
            return Ok(measurements);
        };
        if num_shots == 0 || self.num_measurements == 0 {
            return Ok(measurements);
        }

        debug_assert!(first_shot + num_shots <= total_shots);
        let mut data = measurements.into_meas_major_data();
        noise.apply(&mut data, num_shots, first_shot, total_shots);
        Ok(PackedShots::from_meas_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }
}

impl QecRecordNoise {
    /// `None` when no event can flip a record.
    fn compile(deferred: QecDeferredProgram, seed: u64) -> Result<Option<Self>> {
        let events = super::parity_walk::compile_record_events(&deferred)?;
        if events.is_empty() {
            return Ok(None);
        }
        let gate_count = deferred.gates.len();
        let num_aliases = deferred.num_aliases;
        let mut aliases_by_last_use: Vec<u32> = (0..num_aliases as u32)
            .filter(|&alias| deferred.last_use[alias as usize].is_some())
            .collect();
        aliases_by_last_use.sort_by_key(|&alias| deferred.last_use[alias as usize]);
        let mut record_of_alias = vec![u32::MAX; num_aliases];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            record_of_alias[alias] = record as u32;
        }
        Ok(Some(Self {
            deferred,
            events,
            aliases_by_last_use,
            record_of_alias,
            gate_count,
            seed,
        }))
    }

    /// XOR the noise of shots `first_shot..first_shot + num_shots` of a `total_shots`-shot
    /// run into measurement-major `data`, `num_shots.div_ceil(64)` words per record.
    /// Units run on the pool when the window starts on a word boundary, so no two touch
    /// one word.
    fn apply(&self, data: &mut [u64], num_shots: usize, first_shot: usize, total_shots: usize) {
        let s_words = num_shots.div_ceil(64);
        let end_shot = first_shot + num_shots;
        let first_unit = first_shot / QEC_NOISE_UNIT_SHOTS;
        let last_unit = (end_shot - 1) / QEC_NOISE_UNIT_SHOTS;
        let land = |record: usize,
                    row: &[u64],
                    unit_first: usize,
                    unit_shots: usize,
                    flip: &dyn Fn(usize, u64)| {
            let lo = first_shot.max(unit_first);
            let hi = end_shot.min(unit_first + unit_shots);
            if lo < hi {
                xor_bits(
                    record * s_words,
                    lo - first_shot,
                    row,
                    lo - unit_first,
                    hi - lo,
                    flip,
                );
            }
        };
        #[cfg(feature = "parallel")]
        if last_unit > first_unit
            && first_shot.is_multiple_of(64)
            && qec_noise_units_in_parallel(
                last_unit + 1 - first_unit,
                num_shots,
                self.events.iter().map(|event| &event.draw),
            )
        {
            let data_len = data.len();
            let data = SendPtrU64(data.as_mut_ptr());
            let flip = move |offset: usize, value: u64| {
                debug_assert!(offset < data_len);
                // SAFETY: `offset` is a word of a record row holding shots of the unit
                // being landed, so it lies in `data`. `first_shot` is a multiple of 64 and
                // units span `QEC_NOISE_UNIT_SHOTS` shots, also a multiple of 64, so unit
                // boundaries fall on word boundaries and no two units touch one word.
                unsafe { data.xor_word(offset, value) }
            };
            (first_unit..=last_unit).into_par_iter().for_each(|unit| {
                self.apply_unit(
                    unit,
                    total_shots,
                    &mut |record, row, unit_first, unit_shots| {
                        land(record, row, unit_first, unit_shots, &flip)
                    },
                );
            });
            return;
        }
        let cells = std::cell::Cell::from_mut(data).as_slice_of_cells();
        let flip = |offset: usize, value: u64| cells[offset].set(cells[offset].get() ^ value);
        for unit in first_unit..=last_unit {
            self.apply_unit(
                unit,
                total_shots,
                &mut |record, row, unit_first, unit_shots| {
                    land(record, row, unit_first, unit_shots, &flip)
                },
            );
        }
    }

    /// Draw unit `unit` of a `total_shots`-shot run and push its faults through the
    /// circuit, calling `land(record, x_row, unit_first_shot, unit_shots)` with the X
    /// frame of each measured alias at its last use.
    fn apply_unit(
        &self,
        unit: usize,
        total_shots: usize,
        land: &mut impl FnMut(usize, &[u64], usize, usize),
    ) {
        let unit_first = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (total_shots - unit_first).min(QEC_NOISE_UNIT_SHOTS);
        let words = unit_shots.div_ceil(64);

        let mut rng = qec_noise_unit_rng(self.seed, unit);
        let mut firings: Vec<u64> = Vec::new();
        draw_qec_unit(
            &self.events,
            |event| &event.draw,
            &mut rng,
            unit_shots,
            |index, _, shot, branch| {
                firings.push(((index as u64) << 32) | ((shot as u64) << 8) | branch as u64);
            },
        );
        let mut starts = vec![0u32; self.events.len() + 1];
        for &firing in &firings {
            starts[(firing >> 32) as usize + 1] += 1;
        }
        for index in 0..self.events.len() {
            starts[index + 1] += starts[index];
        }
        let mut fill = starts.clone();
        let mut by_event = vec![0u32; firings.len()];
        for &firing in &firings {
            let index = (firing >> 32) as usize;
            by_event[fill[index] as usize] = firing as u32;
            fill[index] += 1;
        }
        drop(firings);

        let deferred = &self.deferred;
        let mut frame = Frame::new(words, deferred.num_aliases);
        let mut next_event = self.events.len();
        let mut next_alias = 0usize;
        let mut slots = [0usize; 2];
        for position in 0..=self.gate_count {
            while next_event > 0 && self.events[next_event - 1].position as usize == position {
                next_event -= 1;
                let event = &self.events[next_event];
                let fired = &by_event[starts[next_event] as usize..starts[next_event + 1] as usize];
                if fired.is_empty() {
                    continue;
                }
                match event.draw {
                    QecNoiseDraw::Single { .. } => {
                        let slot = frame.slot(event.targets[0] as usize);
                        for &firing in fired {
                            frame.flip(slot, (firing >> 8) as usize, (firing & 0xFF) as usize + 1);
                        }
                    }
                    QecNoiseDraw::Pair { .. } => {
                        let slot0 = frame.slot(event.targets[0] as usize);
                        let slot1 = frame.slot(event.targets[1] as usize);
                        for &firing in fired {
                            let shot = (firing >> 8) as usize;
                            let sample = (firing & 0xFF) as usize + 1;
                            frame.flip(slot0, shot, sample / 4);
                            frame.flip(slot1, shot, sample % 4);
                        }
                    }
                }
            }
            while next_alias < self.aliases_by_last_use.len() {
                let alias = self.aliases_by_last_use[next_alias] as usize;
                if deferred.last_use[alias] != Some(position) {
                    break;
                }
                next_alias += 1;
                if let Some(slot) = frame.release(alias) {
                    let record = self.record_of_alias[alias];
                    if record != u32::MAX {
                        land(record as usize, frame.x_row(slot), unit_first, unit_shots);
                    }
                    frame.clear(slot);
                }
            }
            if position < self.gate_count {
                let gate = &deferred.gates[position];
                let targets = gate.targets();
                for (slot, &alias) in slots.iter_mut().zip(targets) {
                    *slot = frame.slot(alias as usize);
                }
                frame.gate(&gate.gate, &slots[..targets.len()]);
            }
        }
    }
}

/// Pauli frame of one unit's shots: a packed X row and Z row per live alias, one bit
/// per shot.
struct Frame {
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    slot_of_alias: Vec<u32>,
    free: Vec<u32>,
}

impl Frame {
    fn new(words: usize, num_aliases: usize) -> Self {
        Self {
            words,
            x: Vec::new(),
            z: Vec::new(),
            slot_of_alias: vec![u32::MAX; num_aliases],
            free: Vec::new(),
        }
    }

    /// Row of `alias`, zero where the frame first meets it.
    #[inline(always)]
    fn slot(&mut self, alias: usize) -> usize {
        let mapped = self.slot_of_alias[alias];
        if mapped != u32::MAX {
            return mapped as usize;
        }
        let slot = match self.free.pop() {
            Some(slot) => slot as usize,
            None => {
                self.x.resize(self.x.len() + self.words, 0);
                self.z.resize(self.z.len() + self.words, 0);
                self.x.len() / self.words - 1
            }
        };
        self.slot_of_alias[alias] = slot as u32;
        slot
    }

    fn release(&mut self, alias: usize) -> Option<usize> {
        let slot = std::mem::replace(&mut self.slot_of_alias[alias], u32::MAX);
        (slot != u32::MAX).then_some(slot as usize)
    }

    fn clear(&mut self, slot: usize) {
        let row = slot * self.words..(slot + 1) * self.words;
        self.x[row.clone()].fill(0);
        self.z[row].fill(0);
        self.free.push(slot as u32);
    }

    fn x_row(&self, slot: usize) -> &[u64] {
        &self.x[slot * self.words..(slot + 1) * self.words]
    }

    /// XOR Pauli `letter` (1 X, 2 Y, 3 Z, else none) into `shot` of `slot`.
    #[inline(always)]
    fn flip(&mut self, slot: usize, shot: usize, letter: usize) {
        let at = slot * self.words + shot / 64;
        let bit = 1u64 << (shot % 64);
        match letter {
            1 => self.x[at] ^= bit,
            2 => {
                self.x[at] ^= bit;
                self.z[at] ^= bit;
            }
            3 => self.z[at] ^= bit,
            _ => {}
        }
    }

    /// Push the frame through `gate` on `slots`, `P <- U P U†` up to sign.
    #[inline(always)]
    fn gate(&mut self, gate: &Gate, slots: &[usize]) {
        let words = self.words;
        let row = |slot: usize| slot * words..(slot + 1) * words;
        match *slots {
            [q] => match gate {
                Gate::H => self.x[row(q)].swap_with_slice(&mut self.z[row(q)]),
                Gate::S | Gate::Sdg => {
                    for (z, &x) in self.z[row(q)].iter_mut().zip(&self.x[row(q)]) {
                        *z ^= x;
                    }
                }
                Gate::SX | Gate::SXdg => {
                    for (x, &z) in self.x[row(q)].iter_mut().zip(&self.z[row(q)]) {
                        *x ^= z;
                    }
                }
                Gate::X | Gate::Y | Gate::Z | Gate::Id => {}
                _ => unreachable!("the deferred circuit holds Clifford gates only"),
            },
            [a, b] => match gate {
                Gate::Cx => {
                    let (x_control, x_target) = pair_rows_mut(&mut self.x, a, b, words);
                    xor_words(x_target, x_control);
                    let (z_control, z_target) = pair_rows_mut(&mut self.z, a, b, words);
                    xor_words(z_control, z_target);
                }
                Gate::Cz => {
                    let (z_a, z_b) = pair_rows_mut(&mut self.z, a, b, words);
                    xor_words(z_a, &self.x[row(b)]);
                    xor_words(z_b, &self.x[row(a)]);
                }
                Gate::Swap => {
                    let (x_a, x_b) = pair_rows_mut(&mut self.x, a, b, words);
                    x_a.swap_with_slice(x_b);
                    let (z_a, z_b) = pair_rows_mut(&mut self.z, a, b, words);
                    z_a.swap_with_slice(z_b);
                }
                _ => unreachable!("the deferred circuit holds Clifford gates only"),
            },
            _ => unreachable!("the deferred circuit holds Clifford gates only"),
        }
    }
}

/// XOR `count` bits of `src` from `src_bit` into the words from `dst_base`, at bit
/// `dst_bit`, through `flip(offset, value)`.
fn xor_bits(
    dst_base: usize,
    dst_bit: usize,
    src: &[u64],
    src_bit: usize,
    count: usize,
    flip: &dyn Fn(usize, u64),
) {
    let mut done = 0;
    while done < count {
        let n = (count - done).min(64);
        let value = read_bits(src, src_bit + done, n);
        let at = dst_bit + done;
        let lo = at % 64;
        if value != 0 {
            flip(dst_base + at / 64, value << lo);
            if lo + n > 64 {
                flip(dst_base + at / 64 + 1, value >> (64 - lo));
            }
        }
        done += n;
    }
}

/// `n` bits (at most 64) of `src` from `bit`, in the low bits.
fn read_bits(src: &[u64], bit: usize, n: usize) -> u64 {
    let lo = bit % 64;
    let word = bit / 64;
    let mut value = src[word] >> lo;
    if lo + n > 64 {
        value |= src[word + 1] << (64 - lo);
    }
    if n < 64 {
        value &= (1u64 << n) - 1;
    }
    value
}

/// Branch rates a single-qubit event keeps once branches with equal record flips merge,
/// or `None` when it can flip no record.
pub(super) fn qec_single_noise_rates(
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
pub(super) fn qec_pair_branch_flips(
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

/// One event's draw constants, computed once so every unit restarts the event from them.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) enum QecNoiseDraw {
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
    pub(super) fn single(px: f64, py: f64, pz: f64) -> Self {
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

    pub(super) fn pair(p: f64) -> Self {
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
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct QecSingleNoiseRates {
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
    let noiseless = compile_measurements(&deferred.to_circuit(), program.options().seed)?;
    let num_measurements = deferred.measurement_qubits.len();
    let noise = QecRecordNoise::compile(deferred, program.options().seed)?;
    Ok(QecCompiledNoiseSampler {
        noiseless,
        noise,
        num_measurements,
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
    let mut gates: Vec<DeferredGate> = Vec::with_capacity(qec_deferred_gate_count(program));
    touch_spare_capacity(&mut gates);
    let mut aliases: Vec<usize> = (0..base_qubits).collect();
    let mut measured_aliases = vec![false; base_qubits];
    let mut alias_positions = vec![0; base_qubits];
    let mut last_use: Vec<Option<usize>> = vec![None; base_qubits];
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
                gates.gate(gate.clone(), mapped.as_slice());
                for &alias in &mapped {
                    last_use[alias] = Some(gates.len());
                }
            }
            QecOp::Measure { basis, qubit } => {
                let alias = qec_deferred_target(*qubit, &aliases, &measured_aliases)?;
                append_basis_to_z_rotation(&mut gates, *basis, alias);
                if *basis != QecBasis::Z {
                    last_use[alias] = Some(gates.len());
                }
                deferred_measurements.push((alias, next_record));
                measured_aliases[alias] = true;
                next_record += 1;
            }
            QecOp::MeasurePauliProduct { terms } => {
                let scratch_alias = if measured_aliases[aliases[scratch_qubit]] {
                    qec_assign_fresh_alias(
                        gates.len(),
                        &mut aliases,
                        &mut measured_aliases,
                        &mut alias_positions,
                        &mut last_use,
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
                append_mpp_parity_rotations(&mut gates, &mapped_terms, scratch_alias);
                if !mapped_terms.is_empty() {
                    let position = gates.len();
                    for term in &mapped_terms {
                        last_use[term.qubit] = Some(position);
                    }
                    last_use[scratch_alias] = Some(position);
                }

                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::Reset { basis, qubit } => {
                let alias = qec_assign_fresh_alias(
                    gates.len(),
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut last_use,
                    &mut next_qubit,
                    *qubit,
                );
                append_z_to_basis_rotation(&mut gates, *basis, alias);
                if *basis != QecBasis::Z {
                    last_use[alias] = Some(gates.len());
                }
            }
            QecOp::Noise { channel, targets } => {
                if channel.probability() > 0.0 {
                    let first = noise_events.len();
                    push_qec_deferred_noise_events(
                        *channel,
                        targets,
                        &aliases,
                        &measured_aliases,
                        gates.len(),
                        &mut noise_events,
                    )?;
                    for event in &noise_events[first..] {
                        for &alias in &event.targets {
                            last_use[alias] = Some(event.position);
                        }
                    }
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

    let measurement_qubits: Vec<usize> = deferred_measurements
        .iter()
        .map(|&(qubit, _)| qubit)
        .collect();
    debug_assert!(
        deferred_measurements
            .iter()
            .enumerate()
            .all(|(record, &(_, bit))| record == bit)
    );

    let final_qubit_aliases = aliases[..program.num_qubits()].to_vec();
    Ok(QecDeferredProgram {
        gates,
        num_aliases: next_qubit,
        noise_events,
        measurement_qubits,
        alias_positions,
        last_use,
        final_qubit_aliases,
    })
}

/// Write one zero into every page of `v`'s spare capacity, from the pool when there is
/// one, so the first touch of a deep program's instruction list is not paid serially by
/// the lowering loop: a 96-byte instruction per gate means tens of megabytes of fresh
/// pages per megaquop program, and the page faults cost more than filling them.
fn touch_spare_capacity<T: Send>(v: &mut Vec<T>) {
    let per_page = (4096 / size_of::<T>().max(1)).max(1);
    let spare = v.spare_capacity_mut();
    #[cfg(feature = "parallel")]
    if spare.len() >= 64 * per_page {
        spare.par_chunks_mut(per_page).for_each(|page| {
            page[0] = std::mem::MaybeUninit::zeroed();
        });
        return;
    }
    for page in spare.chunks_mut(per_page) {
        page[0] = std::mem::MaybeUninit::zeroed();
    }
}

/// Gates the deferred lowering emits for `program`, so the list is built in one
/// allocation: the doubling copies and page faults of a growing list cost more than the
/// lowering itself.
fn qec_deferred_gate_count(program: &QecProgram) -> usize {
    let rotation = |basis: QecBasis| match basis {
        QecBasis::X => 1,
        QecBasis::Y => 2,
        QecBasis::Z => 0,
    };
    let gates: usize = program
        .ops()
        .iter()
        .map(|op| match op {
            QecOp::Gate { .. } => 1,
            QecOp::Measure { basis, .. } | QecOp::Reset { basis, .. } => rotation(*basis),
            QecOp::MeasurePauliProduct { terms } => {
                terms.iter().map(|term| 2 * rotation(term.basis) + 1).sum()
            }
            _ => 0,
        })
        .sum();
    gates
}

fn qec_assign_fresh_alias(
    position: usize,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    alias_positions: &mut Vec<usize>,
    last_use: &mut Vec<Option<usize>>,
    next_qubit: &mut usize,
    logical_qubit: usize,
) -> usize {
    let alias = *next_qubit;
    aliases[logical_qubit] = alias;
    *next_qubit += 1;
    measured_aliases.push(false);
    alias_positions.push(position);
    last_use.push(None);
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

/// Compile a noisy program straight onto the detector and observable bits of
/// `projection`: the noiseless pattern and the faults that flip it. `None` when a
/// detector or observable is random in the noiseless circuit, which the record path
/// then samples.
///
/// Events, branch rates and draw order match [`QecCompiledNoiseSampler`], so sampling
/// stays bit-identical to projecting its record flips. Nothing here grows with records
/// times gates or records times events: the walk carries one Pauli row per output
/// still live at its position, and an event stores the output indices its branches
/// flip.
#[cfg(test)]
pub(super) fn compile_qec_parity_noise(
    program: &QecProgram,
    projection: &QecParityProjection,
) -> Result<Option<(Vec<u64>, QecParityNoise)>> {
    compile_qec_parity_noise_windows(program, projection, None)
}

/// [`compile_qec_parity_noise`] over a fixed number of walk windows; `None` picks it
/// from the gate count and the thread pool. The result is the same for every count.
#[cfg(test)]
pub(super) fn compile_qec_parity_noise_windows(
    program: &QecProgram,
    projection: &QecParityProjection,
    windows: Option<usize>,
) -> Result<Option<(Vec<u64>, QecParityNoise)>> {
    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    super::parity_walk::compile_parity_noise(&deferred, projection, program.options().seed, windows)
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

/// Draw one depolarize-2 event over `num_shots` shots, calling `flip(shot, branch)` per
/// fault with a uniform branch in `0..15` (see [`qec_pair_branch_flips`] for the order).
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
