//! The record sampler: the reference records spread by a Pauli frame pass that lands
//! birth coins and faults on the measured aliases.

use super::feedforward::QecFeedforwardFlips;
use super::lower::lower_qec_program_to_deferred_circuit_inner;
#[cfg(feature = "parallel")]
use super::qec_noise_units_in_parallel;
use super::{
    QEC_NOISE_UNIT_SHOTS, QecDeferredProgram, QecNoiseDraw, draw_qec_unit, qec_coin_unit_rng,
    qec_noise_unit_rng,
};
#[cfg(any(test, feature = "bench-internal"))]
use crate::error::PrismError;
use crate::error::Result;
use crate::gates::Gate;
use crate::qec::QecProgram;
#[cfg(feature = "parallel")]
use crate::sim::compiled::SendPtrU64;
#[cfg(test)]
use crate::sim::compiled::{CompiledSampler, compile_measurements};
use crate::sim::compiled::{PackedShots, pair_rows_mut, xor_words};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub(in crate::qec) struct QecCompiledNoiseSampler {
    /// Packed records of one noiseless run with every random outcome 0, where a records
    /// sample starts. The frame walk spreads it over the noiseless outcome space with a
    /// fair-coin Z at the birth of every alias whose Z can flip a record.
    reference: Vec<u64>,
    noise: Option<QecRecordNoise>,
    feedforward: Option<QecFeedforwardFlips>,
    num_measurements: usize,
}

/// Kept events of a program whose records are sampled, applied per unit as a Pauli
/// frame pass over the deferred circuit: a fault lands on the frame of its alias at its
/// anchor, gates push the frame forward, and the X frame of an alias at its last use
/// is the flip of its record.
pub(in crate::qec) struct QecRecordNoise {
    deferred: QecDeferredProgram,
    /// Kept events in draw order, positions descending.
    events: Vec<QecRecordEvent>,
    fire_rates: Vec<f64>,
    /// Aliases the circuit touches, ascending by [`QecDeferredProgram::last_use`].
    aliases_by_last_use: Vec<u32>,
    record_of_alias: Vec<u32>,
    /// `(position, alias)` of every alias that takes a fair-coin Z at its birth,
    /// ascending.
    births: Vec<(u32, u32)>,
    gate_count: usize,
    seed: u64,
    cached_unit: Option<QecUnitFlips>,
}

/// Which parts of the frame one walk lands: the birth coins that spread the reference
/// over the noiseless outcome space, the faults, or both.
#[derive(Clone, Copy, PartialEq, Eq)]
struct QecFrameParts {
    coins: bool,
    events: bool,
}

impl QecFrameParts {
    const ALL: Self = Self {
        coins: true,
        events: true,
    };
    #[cfg(any(test, feature = "bench-internal"))]
    const EVENTS: Self = Self {
        coins: false,
        events: true,
    };
    #[cfg(any(test, feature = "bench-internal"))]
    const COINS: Self = Self {
        coins: true,
        events: false,
    };
}

/// Record flips of one noise unit, `words` per record in record order, kept while a
/// chunked run's windows stop inside the unit so the next window lands the rest of it
/// without drawing the unit again.
struct QecUnitFlips {
    unit: usize,
    total_shots: usize,
    parts: QecFrameParts,
    rows: Vec<u64>,
}

#[derive(Clone, Copy)]
pub(in crate::qec) struct QecRecordEvent {
    pub(in crate::qec) draw: QecNoiseDraw,
    pub(in crate::qec) position: u32,
    /// Target aliases: one and `u32::MAX` for a single-qubit event, two for a pair. A
    /// product event holds its deferred event index and `u32::MAX`, since its targets
    /// live on that event.
    pub(in crate::qec) targets: [u32; 2],
}

impl QecCompiledNoiseSampler {
    /// Packed reference records the sampler starts from.
    pub(in crate::qec) fn reference_records(&self) -> &[u64] {
        &self.reference
    }

    /// Restart every stream as a sampler compiled with `seed` starts it.
    pub(in crate::qec) fn reseed(&mut self, seed: u64) {
        if let Some(noise) = &mut self.noise {
            noise.seed = seed;
            noise.cached_unit = None;
        }
    }

    /// Sample shots `first_shot..first_shot + num_shots` of a `total_shots`-shot run.
    pub(in crate::qec) fn sample_measurements_packed(
        &mut self,
        first_shot: usize,
        num_shots: usize,
        total_shots: usize,
    ) -> Result<PackedShots> {
        let mut data = broadcast_records(&self.reference, self.num_measurements, num_shots);
        if num_shots > 0 {
            if let Some(noise) = &mut self.noise {
                noise.apply(
                    &mut data,
                    num_shots,
                    first_shot,
                    total_shots,
                    QecFrameParts::ALL,
                );
            }
            if let Some(feedforward) = &self.feedforward {
                feedforward.apply(&mut data, num_shots);
            }
        }
        Ok(PackedShots::from_meas_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }

    /// Noiseless records of a `num_shots`-shot run: the reference with the birth coins
    /// landed.
    #[cfg(feature = "bench-internal")]
    pub(in crate::qec) fn sample_noiseless_measurements_packed(
        &mut self,
        num_shots: usize,
    ) -> Result<PackedShots> {
        let mut data = broadcast_records(&self.reference, self.num_measurements, num_shots);
        if let Some(noise) = &mut self.noise
            && num_shots > 0
        {
            noise.apply(&mut data, num_shots, 0, num_shots, QecFrameParts::COINS);
        }
        Ok(PackedShots::from_meas_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }

    /// Apply noise, then feed-forward corrections, to `measurements`, which hold shots
    /// from `first_shot` of a `total_shots`-shot run.
    #[cfg(any(test, feature = "bench-internal"))]
    pub(in crate::qec) fn apply_noise_to_measurements(
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
        if self.noise.is_none() && self.feedforward.is_none() {
            return Ok(measurements);
        }
        if num_shots == 0 || self.num_measurements == 0 {
            return Ok(measurements);
        }

        debug_assert!(first_shot + num_shots <= total_shots);
        let mut data = measurements.into_meas_major_data();
        if let Some(noise) = &mut self.noise {
            noise.apply(
                &mut data,
                num_shots,
                first_shot,
                total_shots,
                QecFrameParts::EVENTS,
            );
        }
        if let Some(feedforward) = &self.feedforward {
            feedforward.apply(&mut data, num_shots);
        }
        Ok(PackedShots::from_meas_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }
}

impl QecRecordNoise {
    /// `None` when no event and no birth coin can flip a record. `with_coins` gives
    /// every alias whose Z at its birth flips a record a fair coin there.
    fn compile(deferred: QecDeferredProgram, seed: u64, with_coins: bool) -> Result<Option<Self>> {
        let (events, flipping_births) =
            crate::qec::parity_walk::compile_record_events(&deferred, None)?;
        let mut births: Vec<(u32, u32)> = if with_coins {
            flipping_births
                .into_iter()
                .map(|alias| (deferred.alias_positions[alias as usize] as u32, alias))
                .collect()
        } else {
            Vec::new()
        };
        if events.is_empty() && births.is_empty() {
            return Ok(None);
        }
        births.sort_unstable();
        let (aliases_by_last_use, record_of_alias) = record_walk_order(&deferred);
        let gate_count = deferred.gates.len();
        let fire_rates = events.iter().map(|event| event.draw.rate()).collect();
        Ok(Some(Self {
            deferred,
            events,
            fire_rates,
            aliases_by_last_use,
            record_of_alias,
            births,
            gate_count,
            seed,
            cached_unit: None,
        }))
    }

    /// XOR the noise of shots `first_shot..first_shot + num_shots` of a `total_shots`-shot
    /// run into measurement-major `data`, `num_shots.div_ceil(64)` words per record.
    /// A unit the window covers only in part lands through [`Self::land_cached_unit`].
    /// Whole units run on the pool when the window starts on a word boundary, so no two
    /// touch one word.
    fn apply(
        &mut self,
        data: &mut [u64],
        num_shots: usize,
        first_shot: usize,
        total_shots: usize,
        parts: QecFrameParts,
    ) {
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
        let unit_end = |unit: usize| ((unit + 1) * QEC_NOISE_UNIT_SHOTS).min(total_shots);
        let head_partial =
            first_shot > first_unit * QEC_NOISE_UNIT_SHOTS || end_shot < unit_end(first_unit);
        let tail_partial = last_unit > first_unit && end_shot < unit_end(last_unit);
        if head_partial || tail_partial {
            let cells = std::cell::Cell::from_mut(&mut *data).as_slice_of_cells();
            let flip = |offset: usize, value: u64| cells[offset].set(cells[offset].get() ^ value);
            let mut land_partial = |record: usize, row: &[u64], unit_first: usize, shots: usize| {
                land(record, row, unit_first, shots, &flip)
            };
            if head_partial {
                self.land_cached_unit(first_unit, total_shots, end_shot, parts, &mut land_partial);
            }
            if tail_partial {
                self.land_cached_unit(last_unit, total_shots, end_shot, parts, &mut land_partial);
            }
        }
        let whole =
            first_unit + usize::from(head_partial)..last_unit + 1 - usize::from(tail_partial);
        if whole.is_empty() {
            return;
        }
        #[cfg(feature = "parallel")]
        if whole.len() > 1
            && first_shot.is_multiple_of(64)
            && qec_noise_units_in_parallel(
                whole.len(),
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
            whole.into_par_iter().for_each(|unit| {
                self.apply_unit(
                    unit,
                    total_shots,
                    parts,
                    &mut |record, row, unit_first, unit_shots| {
                        land(record, row, unit_first, unit_shots, &flip)
                    },
                );
            });
            return;
        }
        let cells = std::cell::Cell::from_mut(data).as_slice_of_cells();
        let flip = |offset: usize, value: u64| cells[offset].set(cells[offset].get() ^ value);
        for unit in whole {
            self.apply_unit(
                unit,
                total_shots,
                parts,
                &mut |record, row, unit_first, unit_shots| {
                    land(record, row, unit_first, unit_shots, &flip)
                },
            );
        }
    }

    /// Land unit `unit` of a `total_shots`-shot run from its cached record flips, drawing
    /// them first when the cache holds another unit, and keep them while the window
    /// ending at `end_shot` stops short of the unit's last shot. Windows that split a unit
    /// then draw it once between them, as an unsplit run does.
    fn land_cached_unit(
        &mut self,
        unit: usize,
        total_shots: usize,
        end_shot: usize,
        parts: QecFrameParts,
        land: &mut impl FnMut(usize, &[u64], usize, usize),
    ) {
        let unit_first = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (total_shots - unit_first).min(QEC_NOISE_UNIT_SHOTS);
        let words = unit_shots.div_ceil(64);
        let cached = match self.cached_unit.take() {
            Some(cached)
                if cached.unit == unit
                    && cached.total_shots == total_shots
                    && cached.parts == parts =>
            {
                cached
            }
            _ => {
                let mut rows = vec![0u64; self.deferred.measurement_qubits.len() * words];
                self.apply_unit(unit, total_shots, parts, &mut |record, row, _, _| {
                    xor_words(&mut rows[record * words..(record + 1) * words], row)
                });
                QecUnitFlips {
                    unit,
                    total_shots,
                    parts,
                    rows,
                }
            }
        };
        for (record, row) in cached.rows.chunks_exact(words).enumerate() {
            land(record, row, unit_first, unit_shots);
        }
        if end_shot < unit_first + unit_shots {
            self.cached_unit = Some(cached);
        }
    }

    /// Draw unit `unit` of a `total_shots`-shot run and push its `parts` through the
    /// circuit, calling `land(record, x_row, unit_first_shot, unit_shots)` with the X
    /// frame of each measured alias at its last use. Coins draw from their own stream,
    /// so faults fire on the same shots with or without them.
    fn apply_unit(
        &self,
        unit: usize,
        total_shots: usize,
        parts: QecFrameParts,
        land: &mut impl FnMut(usize, &[u64], usize, usize),
    ) {
        let unit_first = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (total_shots - unit_first).min(QEC_NOISE_UNIT_SHOTS);
        let words = unit_shots.div_ceil(64);

        let events: &[QecRecordEvent] = if parts.events { &self.events } else { &[] };
        let births: &[(u32, u32)] = if parts.coins { &self.births } else { &[] };
        let mut rng = qec_noise_unit_rng(self.seed, unit);
        let mut coin_rng = qec_coin_unit_rng(self.seed, unit);
        let mut firings: Vec<u64> = Vec::new();
        draw_qec_unit(
            events,
            |event| &event.draw,
            &self.fire_rates[..events.len()],
            &self.deferred.pair_tables,
            &self.deferred.product_tables,
            &mut rng,
            unit_shots,
            |index, _, shot, branch| {
                firings.push(((index as u64) << 32) | ((shot as u64) << 16) | branch as u64);
            },
        );
        let mut starts = vec![0u32; events.len() + 1];
        for &firing in &firings {
            starts[(firing >> 32) as usize + 1] += 1;
        }
        for index in 0..events.len() {
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
        let mut next_event = events.len();
        let mut next_birth = 0usize;
        let mut next_alias = 0usize;
        let mut slots = [0usize; 2];
        for position in 0..=self.gate_count {
            while next_birth < births.len() && births[next_birth].0 as usize == position {
                let slot = frame.slot(births[next_birth].1 as usize);
                for word in frame.z_row_mut(slot) {
                    *word ^= coin_rng.next_u64();
                }
                next_birth += 1;
            }
            while next_event > 0 && events[next_event - 1].position as usize == position {
                next_event -= 1;
                let event = &events[next_event];
                let fired = &by_event[starts[next_event] as usize..starts[next_event + 1] as usize];
                if fired.is_empty() {
                    continue;
                }
                match event.draw {
                    QecNoiseDraw::Single { .. } => {
                        let slot = frame.slot(event.targets[0] as usize);
                        for &firing in fired {
                            frame.flip(
                                slot,
                                (firing >> 16) as usize,
                                (firing & 0xFFFF) as usize + 1,
                            );
                        }
                    }
                    QecNoiseDraw::Pair { .. } | QecNoiseDraw::PairTable { .. } => {
                        let slot0 = frame.slot(event.targets[0] as usize);
                        let slot1 = frame.slot(event.targets[1] as usize);
                        for &firing in fired {
                            let shot = (firing >> 16) as usize;
                            let sample = (firing & 0xFFFF) as usize + 1;
                            frame.flip(slot0, shot, sample / 4);
                            frame.flip(slot1, shot, sample % 4);
                        }
                    }
                    QecNoiseDraw::Products { table, .. } => {
                        let table = &deferred.product_tables[table as usize];
                        let targets = &deferred.noise_events[event.targets[0] as usize].targets;
                        for &firing in fired {
                            let shot = (firing >> 16) as usize;
                            let letters = table.branch_letters((firing & 0xFFFF) as usize);
                            for (&alias, &letter) in targets.iter().zip(letters) {
                                if letter != 0 {
                                    let slot = frame.slot(alias);
                                    frame.flip(slot, shot, letter as usize);
                                }
                            }
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

/// Aliases a forward walk meets, ascending by [`QecDeferredProgram::last_use`], and the
/// record each alias holds (`u32::MAX` for none).
pub(super) fn record_walk_order(deferred: &QecDeferredProgram) -> (Vec<u32>, Vec<u32>) {
    let num_aliases = deferred.num_aliases;
    let mut aliases_by_last_use: Vec<u32> = (0..num_aliases as u32)
        .filter(|&alias| deferred.last_use[alias as usize].is_some())
        .collect();
    aliases_by_last_use.sort_by_key(|&alias| deferred.last_use[alias as usize]);
    let mut record_of_alias = vec![u32::MAX; num_aliases];
    for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
        record_of_alias[alias] = record as u32;
    }
    (aliases_by_last_use, record_of_alias)
}

/// Walk `frame` forward over the deferred gates from `start`. At each position,
/// `inject(position, frame)` lands that position's Paulis, every alias whose last use
/// it is leaves the frame (calling `land(record, x_row)` when it holds a record, whose
/// flip is that X row), and then the gate there pushes the frame on. A walk may start
/// past 0 only when nothing is injected before `start`, since the frame is then empty.
#[inline(always)]
pub(super) fn walk_record_frame(
    deferred: &QecDeferredProgram,
    aliases_by_last_use: &[u32],
    record_of_alias: &[u32],
    frame: &mut Frame,
    start: usize,
    mut inject: impl FnMut(usize, &mut Frame),
    mut land: impl FnMut(usize, &[u64]),
) {
    let gate_count = deferred.gates.len();
    let mut next_alias = aliases_by_last_use
        .partition_point(|&alias| deferred.last_use[alias as usize] < Some(start));
    let mut slots = [0usize; 2];
    for position in start..=gate_count {
        inject(position, frame);
        while next_alias < aliases_by_last_use.len() {
            let alias = aliases_by_last_use[next_alias] as usize;
            if deferred.last_use[alias] != Some(position) {
                break;
            }
            next_alias += 1;
            if let Some(slot) = frame.release(alias) {
                let record = record_of_alias[alias];
                if record != u32::MAX {
                    land(record as usize, frame.x_row(slot));
                }
                frame.clear(slot);
            }
        }
        if position < gate_count {
            let gate = &deferred.gates[position];
            let targets = gate.targets();
            for (slot, &alias) in slots.iter_mut().zip(targets) {
                *slot = frame.slot(alias as usize);
            }
            frame.gate(&gate.gate, &slots[..targets.len()]);
        }
    }
}

/// Pauli frame of one unit's shots: a packed X row and Z row per live alias, one bit
/// per shot.
pub(in crate::qec) struct Frame {
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    slot_of_alias: Vec<u32>,
    free: Vec<u32>,
}

impl Frame {
    pub(in crate::qec) fn new(words: usize, num_aliases: usize) -> Self {
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
    pub(in crate::qec) fn slot(&mut self, alias: usize) -> usize {
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

    pub(in crate::qec) fn x_row(&self, slot: usize) -> &[u64] {
        &self.x[slot * self.words..(slot + 1) * self.words]
    }

    fn z_row_mut(&mut self, slot: usize) -> &mut [u64] {
        &mut self.z[slot * self.words..(slot + 1) * self.words]
    }

    /// XOR `x_bits` and `z_bits` into word `word` of `slot` where `mask` is set,
    /// which for uniform random bits is a uniformly random Pauli on those shots.
    #[inline(always)]
    pub(in crate::qec) fn scramble(
        &mut self,
        slot: usize,
        word: usize,
        mask: u64,
        x_bits: u64,
        z_bits: u64,
    ) {
        let at = slot * self.words + word;
        self.x[at] ^= x_bits & mask;
        self.z[at] ^= z_bits & mask;
    }

    /// XOR Pauli `letter` (1 X, 2 Y, 3 Z, else none) into `shot` of `slot`.
    #[inline(always)]
    pub(in crate::qec) fn flip(&mut self, slot: usize, shot: usize, letter: usize) {
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
    pub(in crate::qec) fn gate(&mut self, gate: &Gate, slots: &[usize]) {
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

/// Measurement-major rows of `num_shots` shots, each record's row all ones where its
/// packed `reference` bit is set.
fn broadcast_records(reference: &[u64], num_measurements: usize, num_shots: usize) -> Vec<u64> {
    let s_words = num_shots.div_ceil(64);
    let mut data = vec![0u64; num_measurements * s_words];
    if s_words == 0 {
        return data;
    }
    let tail = match num_shots % 64 {
        0 => u64::MAX,
        bits => (1u64 << bits) - 1,
    };
    for (record, row) in data.chunks_exact_mut(s_words).enumerate() {
        if reference[record / 64] >> (record % 64) & 1 == 1 {
            row.fill(u64::MAX);
            row[s_words - 1] = tail;
        }
    }
    data
}

pub(in crate::qec) fn compile_qec_noisy_sampler(
    program: &QecProgram,
) -> Result<QecCompiledNoiseSampler> {
    if program.has_leakage() {
        return Err(crate::qec::leakage::leakage_rejection(
            "QEC record-noise sampler",
        ));
    }
    let (reference, records) = crate::qec::runner::qec_join(
        || crate::qec::reference::forced_tableau_records(program, false),
        || -> Result<_> {
            let deferred = lower_qec_program_to_deferred_circuit_inner(program, false, true)?;
            let num_measurements = deferred.measurement_qubits.len();
            let feedforward = QecFeedforwardFlips::compile(&deferred);
            let noise = QecRecordNoise::compile(deferred, program.options().seed, true)?;
            Ok((num_measurements, feedforward, noise))
        },
    );
    let (num_measurements, feedforward, noise) = records?;
    let reference = reference?
        .expect("the deferred lowering admits Clifford gates only, so the reference run exists");
    Ok(QecCompiledNoiseSampler {
        reference,
        noise,
        feedforward,
        num_measurements,
    })
}

/// Record sampler over a compiled sampler of the deferred circuit, whose draws already
/// cover the noiseless outcome space, so the frame walk carries faults alone: the base
/// before the reference run took its place, kept to compare it against in distribution.
#[cfg(test)]
pub(in crate::qec) struct QecCompiledBaseSampler {
    base: CompiledSampler,
    /// Noise and feed-forward over the base's draws; its reference is empty and unread.
    records: QecCompiledNoiseSampler,
}

#[cfg(test)]
impl QecCompiledBaseSampler {
    pub(in crate::qec) fn compiled_base(&self) -> &CompiledSampler {
        &self.base
    }

    pub(in crate::qec) fn feedforward(&self) -> Option<&QecFeedforwardFlips> {
        self.records.feedforward.as_ref()
    }

    pub(in crate::qec) fn reseed(&mut self, seed: u64) {
        self.base.reseed(seed);
        self.records.reseed(seed);
    }

    pub(in crate::qec) fn sample_measurements_packed(
        &mut self,
        first_shot: usize,
        num_shots: usize,
        total_shots: usize,
    ) -> Result<PackedShots> {
        let measurements = self.base.try_sample_bulk_packed(num_shots)?;
        self.records
            .apply_noise_to_measurements(measurements, first_shot, total_shots)
    }
}

/// [`compile_qec_noisy_sampler`] on a compiled sampler over the deferred circuit.
#[cfg(test)]
pub(in crate::qec) fn compile_qec_noisy_sampler_on_compiled_base(
    program: &QecProgram,
) -> Result<QecCompiledBaseSampler> {
    if program.has_leakage() {
        return Err(crate::qec::leakage::leakage_rejection(
            "QEC record-noise sampler",
        ));
    }
    let deferred = lower_qec_program_to_deferred_circuit_inner(program, false, true)?;
    let base = compile_measurements(&deferred.to_circuit(), program.options().seed)?;
    let num_measurements = deferred.measurement_qubits.len();
    let feedforward = QecFeedforwardFlips::compile(&deferred);
    let noise = QecRecordNoise::compile(deferred, program.options().seed, false)?;
    Ok(QecCompiledBaseSampler {
        base,
        records: QecCompiledNoiseSampler {
            reference: Vec::new(),
            noise,
            feedforward,
            num_measurements,
        },
    })
}

#[cfg(test)]
#[path = "record_tests.rs"]
mod tests;
