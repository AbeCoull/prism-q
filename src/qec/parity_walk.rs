//! Output-space backward walk behind the record-free QEC compile: one Pauli row per live
//! alias over the detector and observable columns live at the walk's position, split
//! into position windows that walk in parallel behind a sequential pass that carries
//! only the observable columns and the record fingerprint.

use super::QecNoise;
use super::noise::{
    QecDeferredNoiseEvent, QecDeferredProgram, QecNoiseDraw, QecParityNoise, QecParityNoiseEvent,
    QecRecordEvent, append_qec_pauli_noise_effect, qec_pair_branch_flips, qec_single_noise_rates,
};
use super::runner::QecParityProjection;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::batch_propagate_backward_flat;
use crate::sim::splitmix64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Gates per window below which the walk takes fewer windows, so a shallow program
/// pays no overlap.
#[cfg(feature = "parallel")]
const MIN_GATES_PER_WINDOW: usize = 1 << 14;

/// Positions between checks of whether a window's walk past its own range can stop.
const OVERLAP_CHECK_PERIOD: usize = 64;

/// Compile the deferred circuit onto `projection`'s outputs: the noiseless pattern, or
/// `None` when an output is random, and the faults that flip it.
///
/// The walk runs over `windows` position ranges. Every window owns the outputs whose
/// last record the walk meets inside it, seeds and carries only those detector columns,
/// and walks on past its range until they settle. A sequential pass ahead of the windows
/// carries the observable columns and the fingerprint word, and hands each window their
/// state at its upper boundary, so the branch decisions of events inside a window match
/// the record path exactly as the one-window walk's do. Flips a window records for
/// events of an earlier window are merged in afterwards. The result does not depend on
/// `windows`.
pub(super) fn compile_parity_noise(
    deferred: &QecDeferredProgram,
    projection: &QecParityProjection,
    seed: u64,
    windows: Option<usize>,
) -> Result<Option<(Vec<u64>, QecParityNoise)>> {
    let plan = WalkPlan::new(deferred, projection, windows)?;
    let walk = walk_outputs(&plan)?;
    if !walk.fixed {
        return Ok(None);
    }
    let events = walk
        .events
        .iter()
        .map(|event| QecParityNoiseEvent {
            draw: event.kept.draw,
            first_branch: event.first_branch,
        })
        .collect();
    Ok(Some((
        walk.pattern,
        QecParityNoise::from_parts(seed, events, walk.branch_offsets, walk.outputs),
    )))
}

/// Compile every fault site of the deferred circuit onto `projection`'s outputs for the
/// detector error model: the branches that fire and the outputs each flips, sites in
/// program order. A random output is no obstacle here: the model describes flips of
/// the noiseless outcome.
pub(super) fn compile_fault_sites(
    deferred: &QecDeferredProgram,
    projection: &QecParityProjection,
) -> Result<FaultSites> {
    let plan = WalkPlan::new(deferred, projection, None)?;
    let walk = walk_outputs(&plan)?;
    let mut position_of_candidate = vec![0u32; plan.candidates as usize];
    for (slot, &index) in plan.events_by_position.iter().enumerate() {
        let event = &deferred.noise_events[index as usize];
        let base = plan.candidate_base[slot] as usize;
        position_of_candidate[base..base + candidate_count(event)].fill(event.position as u32);
    }
    let position_of = |at: usize| position_of_candidate[walk.events[at].kept.candidate as usize];
    let mut order = Vec::with_capacity(walk.events.len());
    let mut end = walk.events.len();
    while end > 0 {
        let mut start = end;
        while start > 0 && position_of(start - 1) == position_of(end - 1) {
            start -= 1;
        }
        order.extend((start..end).map(|at| at as u32));
        end = start;
    }
    Ok(FaultSites { walk, order })
}

/// Compile the deferred circuit's kept events for the record sampler, in draw order,
/// each with its anchor position and target aliases. The walk carries no output, only
/// the record fingerprint, which is all the branch decisions need.
pub(super) fn compile_record_events(deferred: &QecDeferredProgram) -> Result<Vec<QecRecordEvent>> {
    let projection = QecParityProjection::new(deferred.measurement_qubits.len(), &[], &[]);
    let plan = WalkPlan::new(deferred, &projection, None)?;
    let walk = walk_outputs(&plan)?;
    let mut site_of_candidate = vec![(0u32, 0u32); plan.candidates as usize];
    for (slot, &index) in plan.events_by_position.iter().enumerate() {
        let event = &deferred.noise_events[index as usize];
        let base = plan.candidate_base[slot] as usize;
        for (offset, site) in site_of_candidate[base..base + candidate_count(event)]
            .iter_mut()
            .enumerate()
        {
            *site = (index, offset as u32);
        }
    }
    Ok(walk
        .events
        .iter()
        .map(|kept| {
            let (index, offset) = site_of_candidate[kept.kept.candidate as usize];
            let event = &deferred.noise_events[index as usize];
            let offset = offset as usize;
            let targets = match event.channel {
                QecNoise::Depolarize2(_) => [
                    event.targets[2 * offset] as u32,
                    event.targets[2 * offset + 1] as u32,
                ],
                _ => [event.targets[offset] as u32, u32::MAX],
            };
            QecRecordEvent {
                draw: kept.kept.draw,
                position: event.position as u32,
                targets,
            }
        })
        .collect())
}

/// Fault sites of one program in program order (positions ascending, program order
/// within a position), the order the detector error model lists mechanisms in.
pub(super) struct FaultSites {
    walk: ParityWalk,
    /// Indices into `walk.events` in program order.
    order: Vec<u32>,
}

impl FaultSites {
    /// Every site's firing branches, as `(probability, outputs ascending)`.
    pub(super) fn sites(&self) -> impl Iterator<Item = impl Iterator<Item = (f64, &[u32])>> {
        self.order
            .iter()
            .map(move |&at| self.walk.branches(&self.walk.events[at as usize]))
    }
}

/// What a walk over the outputs produced: the noiseless pattern, whether every output
/// is fixed, and the kept events in draw order (positions descending, program order
/// within a position), branch `b` flipping `outputs[branch_offsets[b]..branch_offsets[b + 1]]`.
struct ParityWalk {
    pattern: Vec<u64>,
    fixed: bool,
    events: Vec<WalkEvent>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
}

struct WalkEvent {
    kept: KeptEvent,
    first_branch: usize,
}

/// An event the walk keeps: its candidate id, its draw, and the branch rates behind the
/// draw (X, Y, Z for a single-qubit event; the event rate and zeros for a pair).
#[derive(Clone, Copy)]
struct KeptEvent {
    candidate: u32,
    draw: QecNoiseDraw,
    rates: [f64; 3],
}

impl ParityWalk {
    /// `(probability, outputs ascending)` of each branch of `event` with a nonzero rate.
    fn branches(&self, event: &WalkEvent) -> impl Iterator<Item = (f64, &[u32])> {
        let (count, pair) = match event.kept.draw {
            QecNoiseDraw::Single { .. } => (3, false),
            QecNoiseDraw::Pair { .. } => (15, true),
        };
        let rates = event.kept.rates;
        let first = event.first_branch;
        (0..count).filter_map(move |branch| {
            let probability = if pair { rates[0] / 15.0 } else { rates[branch] };
            let range = self.branch_offsets[first + branch] as usize
                ..self.branch_offsets[first + branch + 1] as usize;
            (probability != 0.0).then(|| (probability, &self.outputs[range]))
        })
    }
}

/// Walk `plan` over every window and merge the windows' flips.
fn walk_outputs(plan: &WalkPlan<'_>) -> Result<ParityWalk> {
    let projection = plan.projection;
    let window_count = plan.windows.len();

    let mut snapshots = Vec::new();
    let mut prepass_pattern = vec![0u64; projection.words];
    let mut fixed = true;
    if window_count > 1 {
        let mut rows = WindowRows::new(plan, u32::MAX, true);
        let mut visitor = PrepassVisitor {
            plan,
            snapshots: Vec::with_capacity(window_count - 1),
            next: window_count - 1,
        };
        walk(plan, &mut rows, plan.gate_count, &mut visitor)?;
        snapshots = visitor.snapshots;
        rows.fold_observable_signs();
        fixed &= rows.fixed;
        prepass_pattern = rows.pattern;
    }

    let run_window = |window: usize| -> Result<WindowResult> {
        let mut rows = WindowRows::new(plan, window as u32, window_count == 1);
        if window + 1 < window_count {
            rows.load_snapshot(&snapshots[window_count - 2 - window]);
        }
        let [lo, hi] = plan.windows[window];
        let mut visitor = WindowVisitor::new(lo, hi);
        walk(plan, &mut rows, hi, &mut visitor)?;
        if window_count == 1 {
            rows.fold_observable_signs();
        }
        rows.fold_detector_signs();
        let (own, spill) = visitor.finish();
        Ok(WindowResult {
            own,
            spill,
            pattern: rows.pattern,
            fixed: rows.fixed,
        })
    };
    #[cfg(feature = "parallel")]
    let results: Vec<WindowResult> = if window_count > 1 {
        (0..window_count)
            .into_par_iter()
            .map(run_window)
            .collect::<Result<_>>()?
    } else {
        vec![run_window(0)?]
    };
    #[cfg(not(feature = "parallel"))]
    let results: Vec<WindowResult> = (0..window_count).map(run_window).collect::<Result<_>>()?;

    let mut pattern = prepass_pattern;
    let mut owns = Vec::with_capacity(window_count);
    let mut spills = Vec::with_capacity(window_count);
    for result in results {
        fixed &= result.fixed;
        for (word, &bits) in pattern.iter_mut().zip(&result.pattern) {
            *word ^= bits;
        }
        owns.push(result.own);
        spills.push(result.spill);
    }

    let pieces: Vec<NoisePiece> = if spills.iter().all(|spill| spill.keys.is_empty()) {
        owns.into_iter().map(|own| merge_window(own, &[])).collect()
    } else {
        #[cfg(feature = "parallel")]
        {
            owns.into_par_iter()
                .enumerate()
                .map(|(window, own)| merge_window(own, &spills[window + 1..]))
                .collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            owns.into_iter()
                .enumerate()
                .map(|(window, own)| merge_window(own, &spills[window + 1..]))
                .collect()
        }
    };

    let mut events = Vec::with_capacity(pieces.iter().map(|piece| piece.events.len()).sum());
    let mut branch_offsets = Vec::with_capacity(
        1 + pieces
            .iter()
            .map(|piece| piece.branch_offsets.len() - 1)
            .sum::<usize>(),
    );
    branch_offsets.push(0u32);
    let mut outputs = Vec::with_capacity(pieces.iter().map(|piece| piece.outputs.len()).sum());
    for piece in pieces.iter().rev() {
        let first_branch = branch_offsets.len() - 1;
        let base = outputs.len() as u32;
        events.extend(piece.events.iter().map(|&(kept, branch)| WalkEvent {
            kept,
            first_branch: first_branch + branch,
        }));
        branch_offsets.extend(piece.branch_offsets[1..].iter().map(|&end| end + base));
        outputs.extend_from_slice(&piece.outputs);
    }
    Ok(ParityWalk {
        pattern,
        fixed,
        events,
        branch_offsets,
        outputs,
    })
}

/// The deferred circuit laid out for the walk: events and alias creations in position
/// order, candidate numbering, and which window owns each output.
struct WalkPlan<'a> {
    deferred: &'a QecDeferredProgram,
    projection: &'a QecParityProjection,
    gate_count: usize,
    /// Indices into `deferred.noise_events`, ascending by position.
    events_by_position: Vec<u32>,
    /// First candidate id of each entry of `events_by_position`: candidates (events the
    /// sink is offered) are numbered in visit order, positions descending and program
    /// order within a position.
    candidate_base: Vec<u32>,
    /// Candidates in total.
    candidates: u32,
    /// Aliases ascending by creation position.
    aliases_by_creation: Vec<u32>,
    record_of_alias: Vec<u32>,
    /// `[lo, hi]` position ranges, ascending.
    windows: Vec<[usize; 2]>,
    /// Window owning each output: the one whose range holds the position at or before
    /// which the walk meets the last of its records (`QecDeferredProgram::last_use`, an
    /// upper bound, so the owner is never an earlier window than the meeting one).
    /// `u32::MAX` for an output no walk seeds.
    owner_of_output: Vec<u32>,
    /// Record seeds each window's owned outputs take, so a window knows when it has met
    /// them all.
    owned_seeds: Vec<usize>,
}

impl<'a> WalkPlan<'a> {
    fn new(
        deferred: &'a QecDeferredProgram,
        projection: &'a QecParityProjection,
        windows: Option<usize>,
    ) -> Result<Self> {
        let gate_count = deferred.gates.len();
        let num_aliases = deferred.num_aliases;

        let mut events_by_position: Vec<u32> = (0..deferred.noise_events.len() as u32).collect();
        for event in &deferred.noise_events {
            if event.position > gate_count {
                return Err(PrismError::InvalidParameter {
                    message: "QEC noise event position exceeds deferred gate count".to_string(),
                });
            }
        }
        events_by_position.sort_by_key(|&e| deferred.noise_events[e as usize].position);
        let position_of =
            |slot: usize| deferred.noise_events[events_by_position[slot] as usize].position;
        let mut candidate_base = vec![0u32; events_by_position.len()];
        let mut candidates = 0u32;
        let mut end = events_by_position.len();
        while end > 0 {
            let mut start = end;
            while start > 0 && position_of(start - 1) == position_of(end - 1) {
                start -= 1;
            }
            for slot in start..end {
                candidate_base[slot] = candidates;
                let event = &deferred.noise_events[events_by_position[slot] as usize];
                candidates += candidate_count(event) as u32;
            }
            end = start;
        }

        let mut aliases_by_creation: Vec<u32> = (0..num_aliases as u32).collect();
        aliases_by_creation.sort_by_key(|&a| deferred.alias_positions[a as usize]);

        let mut record_of_alias = vec![u32::MAX; num_aliases];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            record_of_alias[alias] = record as u32;
        }

        let window_count = match windows {
            Some(count) => count.max(1),
            None => default_window_count(gate_count),
        }
        .min(gate_count + 1);
        let mut window_ranges = Vec::with_capacity(window_count);
        for window in 0..window_count {
            let lo = (gate_count + 1) * window / window_count;
            let hi = (gate_count + 1) * (window + 1) / window_count - 1;
            window_ranges.push([lo, hi]);
        }
        let window_of_position = |position: usize| -> u32 {
            window_ranges.partition_point(|&[_, hi]| hi < position) as u32
        };

        let mut last_meet = vec![usize::MAX; projection.num_outputs()];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            let Some(meet) = deferred.last_use[alias] else {
                continue;
            };
            for &output in projection.outputs(record) {
                last_meet[output] = if last_meet[output] == usize::MAX {
                    meet
                } else {
                    last_meet[output].max(meet)
                };
            }
        }
        let owner_of_output: Vec<u32> = last_meet
            .iter()
            .map(|&position| {
                if position == usize::MAX {
                    u32::MAX
                } else {
                    window_of_position(position)
                }
            })
            .collect();
        let mut owned_seeds = vec![0usize; window_count];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            if deferred.last_use[alias].is_none() {
                continue;
            }
            for &output in projection.outputs(record) {
                if output < projection.num_detectors() {
                    owned_seeds[owner_of_output[output] as usize] += 1;
                }
            }
        }

        Ok(Self {
            deferred,
            projection,
            gate_count,
            events_by_position,
            candidate_base,
            candidates,
            aliases_by_creation,
            record_of_alias,
            windows: window_ranges,
            owner_of_output,
            owned_seeds,
        })
    }
}

fn default_window_count(gate_count: usize) -> usize {
    #[cfg(feature = "parallel")]
    {
        (gate_count / MIN_GATES_PER_WINDOW).clamp(1, rayon::current_num_threads().max(1))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = gate_count;
        1
    }
}

/// Events the sink is offered for one deferred event: one per target, or per pair.
fn candidate_count(event: &QecDeferredNoiseEvent) -> usize {
    match event.channel {
        QecNoise::Depolarize2(_) => event.targets.len() / 2,
        _ => event.targets.len(),
    }
}

/// Pauli rows of one walk, packed end to end: per slot `det_words` words of detector
/// columns, `obs_words` of observable columns, then the fingerprint word. Detector
/// columns exist only for the outputs `window` owns and are recycled as they settle;
/// `u32::MAX` owns no detector and carries observables and fingerprint alone.
struct WindowRows<'a> {
    plan: &'a WalkPlan<'a>,
    window: u32,
    det_words: usize,
    obs_words: usize,
    words: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    sign: Vec<u64>,
    slot_of_alias: Vec<u32>,
    free_slots: Vec<u32>,
    /// Detector column of each owned output, `u32::MAX` when it has none.
    column_of_output: Vec<u32>,
    /// Output held by each column, detector and observable alike.
    output_of_column: Vec<u32>,
    free_columns: Vec<u32>,
    used: Vec<u64>,
    pattern: Vec<u64>,
    fixed: bool,
    pending_seeds: usize,
    /// Whether this walk answers for the observables' pattern and randomness: the
    /// sequential pass, or the only window.
    checks_observables: bool,
}

impl<'a> WindowRows<'a> {
    fn new(plan: &'a WalkPlan<'a>, window: u32, checks_observables: bool) -> Self {
        let projection = plan.projection;
        let num_detectors = projection.num_detectors();
        let num_observables = projection.num_outputs() - num_detectors;
        let obs_words = num_observables.div_ceil(64);
        let mut output_of_column = vec![u32::MAX; obs_words * 64];
        for (observable, output) in output_of_column[..num_observables].iter_mut().enumerate() {
            *output = (num_detectors + observable) as u32;
        }
        let pending_seeds = plan.owned_seeds.get(window as usize).copied().unwrap_or(0);
        Self {
            plan,
            window,
            det_words: 0,
            obs_words,
            words: obs_words + 1,
            x: Vec::new(),
            z: Vec::new(),
            sign: vec![0; obs_words + 1],
            slot_of_alias: vec![u32::MAX; plan.deferred.num_aliases],
            free_slots: Vec::new(),
            column_of_output: vec![u32::MAX; num_detectors],
            output_of_column,
            free_columns: Vec::new(),
            used: Vec::new(),
            pattern: vec![0; projection.words],
            fixed: true,
            pending_seeds,
            checks_observables,
        }
    }

    fn load_snapshot(&mut self, snapshot: &Snapshot) {
        debug_assert_eq!(self.det_words, 0);
        let words = self.words;
        for (index, &alias) in snapshot.aliases.iter().enumerate() {
            let slot = self.new_slot(alias as usize);
            self.x[slot * words..(slot + 1) * words]
                .copy_from_slice(&snapshot.x[index * words..(index + 1) * words]);
            self.z[slot * words..(slot + 1) * words]
                .copy_from_slice(&snapshot.z[index * words..(index + 1) * words]);
        }
        self.sign.copy_from_slice(&snapshot.sign);
    }

    fn snapshot(&self) -> Snapshot {
        debug_assert_eq!(self.det_words, 0);
        let words = self.words;
        let mut snapshot = Snapshot {
            aliases: Vec::new(),
            x: Vec::new(),
            z: Vec::new(),
            sign: self.sign.clone(),
        };
        for (alias, &slot) in self.slot_of_alias.iter().enumerate() {
            if slot == u32::MAX {
                continue;
            }
            let slot = slot as usize;
            snapshot.aliases.push(alias as u32);
            snapshot
                .x
                .extend_from_slice(&self.x[slot * words..(slot + 1) * words]);
            snapshot
                .z
                .extend_from_slice(&self.z[slot * words..(slot + 1) * words]);
        }
        snapshot
    }

    fn new_slot(&mut self, alias: usize) -> usize {
        let slot = match self.free_slots.pop() {
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

    /// Row of `alias`, created and seeded with its record where the walk first meets it.
    #[inline(always)]
    fn slot(&mut self, alias: usize) -> usize {
        let mapped = self.slot_of_alias[alias];
        if mapped != u32::MAX {
            return mapped as usize;
        }
        let slot = self.new_slot(alias);
        let record = self.plan.record_of_alias[alias];
        if record != u32::MAX {
            self.seed_record(slot, record as usize);
        }
        slot
    }

    fn seed_record(&mut self, slot: usize, record: usize) {
        let plan = self.plan;
        let num_detectors = plan.projection.num_detectors();
        for &output in plan.projection.outputs(record) {
            let column = if output >= num_detectors {
                self.det_words * 64 + (output - num_detectors)
            } else if plan.owner_of_output[output] == self.window {
                self.pending_seeds -= 1;
                self.detector_column(output)
            } else {
                continue;
            };
            self.z[slot * self.words + column / 64] ^= 1u64 << (column % 64);
        }
        self.z[slot * self.words + self.words - 1] ^= splitmix64(record as u64);
    }

    /// Column of the owned detector `output`, reclaiming settled columns and then
    /// widening the rows when none is free.
    fn detector_column(&mut self, output: usize) -> usize {
        let mapped = self.column_of_output[output];
        if mapped != u32::MAX {
            return mapped as usize;
        }
        if self.free_columns.is_empty() {
            self.reclaim_detector_columns();
        }
        let column = self
            .free_columns
            .pop()
            .expect("reclaiming leaves a free column");
        self.column_of_output[output] = column;
        self.output_of_column[column as usize] = output as u32;
        column as usize
    }

    /// Free every detector column no row has support on, folding its sign into the
    /// pattern, and widen the rows by a word when that frees less than a word of them.
    fn reclaim_detector_columns(&mut self) {
        self.used.clear();
        self.used.resize(self.det_words, 0);
        let words = self.words;
        for row in self.x.chunks_exact(words).chain(self.z.chunks_exact(words)) {
            for (used, &word) in self.used.iter_mut().zip(row) {
                *used |= word;
            }
        }
        let mut freed = 0;
        for word_idx in 0..self.det_words {
            let mut idle = !self.used[word_idx];
            while idle != 0 {
                let column = word_idx * 64 + idle.trailing_zeros() as usize;
                idle &= idle - 1;
                let output = self.output_of_column[column];
                if output == u32::MAX {
                    continue;
                }
                self.fold_sign(column, output);
                self.column_of_output[output as usize] = u32::MAX;
                self.output_of_column[column] = u32::MAX;
                self.free_columns.push(column as u32);
                freed += 1;
            }
        }
        if freed < 64 {
            self.widen();
        }
    }

    /// Add a detector word ahead of the observable and fingerprint words of every row.
    fn widen(&mut self) {
        let old = self.words;
        let at = self.det_words;
        let slots = self.x.len() / old;
        let mut x = Vec::with_capacity(slots * (old + 1));
        let mut z = Vec::with_capacity(slots * (old + 1));
        for slot in 0..slots {
            for (dst, src) in [(&mut x, &self.x), (&mut z, &self.z)] {
                dst.extend_from_slice(&src[slot * old..slot * old + at]);
                dst.push(0);
                dst.extend_from_slice(&src[slot * old + at..(slot + 1) * old]);
            }
        }
        self.x = x;
        self.z = z;
        self.sign.insert(at, 0);
        let first = at * 64;
        self.output_of_column
            .splice(first..first, std::iter::repeat_n(u32::MAX, 64));
        self.free_columns
            .extend((first..first + 64).rev().map(|column| column as u32));
        self.det_words += 1;
        self.words += 1;
    }

    fn fold_sign(&mut self, column: usize, output: u32) {
        let sign = (self.sign[column / 64] >> (column % 64)) & 1;
        self.sign[column / 64] &= !(1u64 << (column % 64));
        self.pattern[output as usize / 64] ^= sign << (output % 64);
    }

    fn fold_detector_signs(&mut self) {
        for column in 0..self.det_words * 64 {
            let output = self.output_of_column[column];
            if output != u32::MAX {
                self.fold_sign(column, output);
            }
        }
    }

    fn fold_observable_signs(&mut self) {
        for column in self.det_words * 64..(self.det_words + self.obs_words) * 64 {
            let output = self.output_of_column[column];
            if output != u32::MAX {
                self.fold_sign(column, output);
            }
        }
    }

    /// Give back the row of `alias` where the alias comes into use, in `|0>`: X support
    /// there on a column this walk owns makes that output random.
    fn retire(&mut self, alias: usize) {
        let slot = std::mem::replace(&mut self.slot_of_alias[alias], u32::MAX);
        if slot == u32::MAX {
            return;
        }
        let slot = slot as usize;
        let row = slot * self.words..(slot + 1) * self.words;
        let owned_words = if self.checks_observables {
            0..self.det_words + self.obs_words
        } else {
            0..self.det_words
        };
        if self.x[row.clone()][owned_words]
            .iter()
            .any(|&word| word != 0)
        {
            self.fixed = false;
        }
        self.x[row.clone()].fill(0);
        self.z[row].fill(0);
        self.free_slots.push(slot as u32);
    }

    #[inline(always)]
    fn propagate(&mut self, gate: &Gate, slots: &[usize]) {
        batch_propagate_backward_flat(
            &mut self.x,
            &mut self.z,
            &mut self.sign,
            self.words,
            gate,
            slots,
        );
    }

    fn row(&self, slot: usize) -> (&[u64], &[u64]) {
        let range = slot * self.words..(slot + 1) * self.words;
        (&self.x[range.clone()], &self.z[range])
    }

    /// Whether any owned detector column still has support somewhere.
    fn detectors_live(&self) -> bool {
        let words = self.words;
        let det = self.det_words;
        self.x
            .chunks_exact(words)
            .chain(self.z.chunks_exact(words))
            .any(|row| row[..det].iter().any(|&word| word != 0))
    }
}

/// Rows live at a window boundary in the sequential pass: observable and fingerprint
/// words per alias, and the observable signs.
struct Snapshot {
    aliases: Vec<u32>,
    x: Vec<u64>,
    z: Vec<u64>,
    sign: Vec<u64>,
}

trait WalkVisitor {
    /// Called before the walk handles `position`; `false` stops the walk there.
    fn at_position(&mut self, rows: &mut WindowRows<'_>, position: usize) -> bool;
    fn event(
        &mut self,
        rows: &WindowRows<'_>,
        event: &QecDeferredNoiseEvent,
        slots: &[usize],
        candidate: u32,
    );
}

/// Walk positions from `hi` down until the visitor stops the walk or position 0 is done:
/// visit the events at each position, retire the aliases created there, then propagate
/// the gate before it.
fn walk<V: WalkVisitor>(
    plan: &WalkPlan<'_>,
    rows: &mut WindowRows<'_>,
    hi: usize,
    visitor: &mut V,
) -> Result<()> {
    let events = &plan.deferred.noise_events;
    let order = &plan.events_by_position;
    let alias_positions = &plan.deferred.alias_positions;
    let created = &plan.aliases_by_creation;
    let gates = &plan.deferred.gates;

    let mut next_event = order.partition_point(|&e| events[e as usize].position <= hi);
    let mut next_alias = created.partition_point(|&a| alias_positions[a as usize] <= hi);
    let mut event_slots: Vec<usize> = Vec::new();
    let mut gate_slots = [0usize; 2];
    let mut position = hi;
    loop {
        if !visitor.at_position(rows, position) {
            break;
        }
        let event_end = next_event;
        while next_event > 0 && events[order[next_event - 1] as usize].position == position {
            next_event -= 1;
        }
        for slot in next_event..event_end {
            let event = &events[order[slot] as usize];
            event_slots.clear();
            for &target in &event.targets {
                event_slots.push(rows.slot(target));
            }
            visitor.event(rows, event, &event_slots, plan.candidate_base[slot]);
        }
        let alias_end = next_alias;
        while next_alias > 0 && alias_positions[created[next_alias - 1] as usize] == position {
            next_alias -= 1;
        }
        for &alias in &created[next_alias..alias_end] {
            rows.retire(alias as usize);
        }
        if position == 0 {
            break;
        }
        let gate = &gates[position - 1];
        let targets = gate.targets();
        for (slot, &target) in gate_slots.iter_mut().zip(targets) {
            *slot = rows.slot(target as usize);
        }
        rows.propagate(&gate.gate, &gate_slots[..targets.len()]);
        position -= 1;
    }
    Ok(())
}

struct PrepassVisitor<'a> {
    plan: &'a WalkPlan<'a>,
    snapshots: Vec<Snapshot>,
    /// Window whose upper boundary comes next.
    next: usize,
}

impl WalkVisitor for PrepassVisitor<'_> {
    fn at_position(&mut self, rows: &mut WindowRows<'_>, position: usize) -> bool {
        while self.next > 0 && self.plan.windows[self.next - 1][1] == position {
            self.next -= 1;
            self.snapshots.push(rows.snapshot());
        }
        true
    }

    fn event(&mut self, _: &WindowRows<'_>, _: &QecDeferredNoiseEvent, _: &[usize], _: u32) {}
}

/// What one window's walk produced: the events it owns with their kept branches'
/// outputs, the detector flips it saw on earlier windows' events, and its share of the
/// pattern.
struct WindowResult {
    own: OwnFlips,
    spill: SpillFlips,
    pattern: Vec<u64>,
    fixed: bool,
}

/// Kept events of one window, with each branch's outputs in
/// `outputs[branch_offsets[b]..branch_offsets[b + 1]]`.
struct OwnFlips {
    events: Vec<KeptEvent>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
}

/// Detector flips one window saw on earlier windows' events, keyed by
/// `(candidate, branch)` ascending, outputs in `outputs[offsets[i]..offsets[i + 1]]`.
struct SpillFlips {
    keys: Vec<(u32, u8)>,
    offsets: Vec<u32>,
    outputs: Vec<u32>,
}

struct WindowVisitor {
    lo: usize,
    hi: usize,
    position: usize,
    own: OwnFlips,
    spill: SpillFlips,
    branch_flips: Vec<u64>,
}

impl WindowVisitor {
    fn new(lo: usize, hi: usize) -> Self {
        Self {
            lo,
            hi,
            position: hi,
            own: OwnFlips {
                events: Vec::new(),
                branch_offsets: vec![0],
                outputs: Vec::new(),
            },
            spill: SpillFlips {
                keys: Vec::new(),
                offsets: vec![0],
                outputs: Vec::new(),
            },
            branch_flips: Vec::new(),
        }
    }

    fn finish(self) -> (OwnFlips, SpillFlips) {
        (self.own, self.spill)
    }

    /// Record one kept branch of an owned event: the outputs its flips land on.
    fn push_branch(&mut self, rows: &WindowRows<'_>, words: impl Iterator<Item = u64>) {
        let start = self.own.outputs.len();
        push_outputs(&mut self.own.outputs, words, &rows.output_of_column);
        self.own.outputs[start..].sort_unstable();
        self.own.branch_offsets.push(self.own.outputs.len() as u32);
    }

    /// Record the detector flips of one branch of another window's event.
    fn push_spill(
        &mut self,
        rows: &WindowRows<'_>,
        candidate: u32,
        branch: u8,
        words: impl Iterator<Item = u64>,
    ) {
        let start = self.spill.outputs.len();
        push_outputs(
            &mut self.spill.outputs,
            words.take(rows.det_words),
            &rows.output_of_column,
        );
        if self.spill.outputs.len() == start {
            return;
        }
        self.spill.keys.push((candidate, branch));
        self.spill.offsets.push(self.spill.outputs.len() as u32);
    }

    fn single(
        &mut self,
        rows: &WindowRows<'_>,
        slot: usize,
        candidate: u32,
        px: f64,
        py: f64,
        pz: f64,
    ) {
        let (x, z) = rows.row(slot);
        if self.position >= self.lo {
            let Some((px, py, pz)) = qec_single_noise_rates(x, z, px, py, pz) else {
                return;
            };
            self.own.events.push(KeptEvent {
                candidate,
                draw: QecNoiseDraw::single(px, py, pz),
                rates: [px, py, pz],
            });
            self.push_branch(rows, z.iter().copied());
            self.push_branch(rows, x.iter().zip(z).map(|(x, z)| x ^ z));
            self.push_branch(rows, x.iter().copied());
        } else {
            self.push_spill(rows, candidate, 0, z.iter().copied());
            self.push_spill(rows, candidate, 1, x.iter().zip(z).map(|(x, z)| x ^ z));
            self.push_spill(rows, candidate, 2, x.iter().copied());
        }
    }

    fn pair(&mut self, rows: &WindowRows<'_>, slots: [usize; 2], candidate: u32, p: f64) {
        let (x0, z0) = rows.row(slots[0]);
        let (x1, z1) = rows.row(slots[1]);
        let mut branch_flips = std::mem::take(&mut self.branch_flips);
        if self.position >= self.lo {
            if qec_pair_branch_flips(x0, z0, x1, z1, p, &mut branch_flips) {
                self.own.events.push(KeptEvent {
                    candidate,
                    draw: QecNoiseDraw::pair(p),
                    rates: [p, 0.0, 0.0],
                });
                for flips in branch_flips.chunks_exact(rows.words) {
                    self.push_branch(rows, flips.iter().copied());
                }
            }
        } else if rows.det_words > 0 {
            let det = rows.det_words;
            branch_flips.clear();
            branch_flips.resize(15 * det, 0);
            for (sample, branch) in (1..=15).zip(branch_flips.chunks_exact_mut(det)) {
                append_qec_pauli_noise_effect(branch, sample / 4, &x0[..det], &z0[..det]);
                append_qec_pauli_noise_effect(branch, sample % 4, &x1[..det], &z1[..det]);
            }
            for (branch, flips) in branch_flips.chunks_exact(det).enumerate() {
                self.push_spill(rows, candidate, branch as u8, flips.iter().copied());
            }
        }
        self.branch_flips = branch_flips;
    }
}

/// Append the outputs of the set columns in `words`, through `output_of_column`.
fn push_outputs(
    outputs: &mut Vec<u32>,
    words: impl Iterator<Item = u64>,
    output_of_column: &[u32],
) {
    for (word_idx, word) in words.take(output_of_column.len() / 64).enumerate() {
        let mut bits = word;
        while bits != 0 {
            let output = output_of_column[word_idx * 64 + bits.trailing_zeros() as usize];
            debug_assert_ne!(output, u32::MAX, "a set column holds an output");
            outputs.push(output);
            bits &= bits - 1;
        }
    }
}

impl WalkVisitor for WindowVisitor {
    fn at_position(&mut self, rows: &mut WindowRows<'_>, position: usize) -> bool {
        self.position = position;
        if position >= self.lo {
            return true;
        }
        if !(self.lo - position).is_multiple_of(OVERLAP_CHECK_PERIOD) {
            return true;
        }
        rows.pending_seeds > 0 || rows.detectors_live()
    }

    fn event(
        &mut self,
        rows: &WindowRows<'_>,
        event: &QecDeferredNoiseEvent,
        slots: &[usize],
        candidate: u32,
    ) {
        debug_assert!(self.position <= self.hi);
        match event.channel {
            QecNoise::XError(p) => {
                for (offset, &slot) in slots.iter().enumerate() {
                    self.single(rows, slot, candidate + offset as u32, p, 0.0, 0.0);
                }
            }
            QecNoise::ZError(p) => {
                for (offset, &slot) in slots.iter().enumerate() {
                    self.single(rows, slot, candidate + offset as u32, 0.0, 0.0, p);
                }
            }
            QecNoise::Depolarize1(p) => {
                let branch_p = p / 3.0;
                for (offset, &slot) in slots.iter().enumerate() {
                    self.single(
                        rows,
                        slot,
                        candidate + offset as u32,
                        branch_p,
                        branch_p,
                        branch_p,
                    );
                }
            }
            QecNoise::Depolarize2(p) => {
                for (offset, pair) in slots.chunks_exact(2).enumerate() {
                    self.pair(rows, [pair[0], pair[1]], candidate + offset as u32, p);
                }
            }
        }
    }
}

/// One window's kept events with every window's flips merged in, in walk order.
struct NoisePiece {
    /// Event and its branch index into `branch_offsets`.
    events: Vec<(KeptEvent, usize)>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
}

fn branch_count(draw: QecNoiseDraw) -> usize {
    match draw {
        QecNoiseDraw::Single { .. } => 3,
        QecNoiseDraw::Pair { .. } => 15,
    }
}

/// Merge the flips later windows saw on `own`'s events into them.
fn merge_window(own: OwnFlips, later: &[SpillFlips]) -> NoisePiece {
    let sources: Vec<&SpillFlips> = later
        .iter()
        .filter(|spill| !spill.keys.is_empty())
        .collect();
    if sources.is_empty() {
        let mut branch = 0usize;
        let events = own
            .events
            .iter()
            .map(|&kept| {
                let first = branch;
                branch += branch_count(kept.draw);
                (kept, first)
            })
            .collect();
        return NoisePiece {
            events,
            branch_offsets: own.branch_offsets,
            outputs: own.outputs,
        };
    }
    let mut piece = NoisePiece {
        events: Vec::with_capacity(own.events.len()),
        branch_offsets: vec![0],
        outputs: Vec::with_capacity(own.outputs.len()),
    };
    let mut cursors = vec![0usize; sources.len()];
    let mut branch = 0usize;
    for &kept in &own.events {
        let branches = branch_count(kept.draw);
        piece.events.push((kept, piece.branch_offsets.len() - 1));
        for b in 0..branches {
            let start = piece.outputs.len();
            let own_range =
                own.branch_offsets[branch] as usize..own.branch_offsets[branch + 1] as usize;
            piece.outputs.extend_from_slice(&own.outputs[own_range]);
            let key = (kept.candidate, b as u8);
            let mut spilled = false;
            for (source, cursor) in sources.iter().zip(cursors.iter_mut()) {
                while *cursor < source.keys.len() && source.keys[*cursor] < key {
                    *cursor += 1;
                }
                if *cursor < source.keys.len() && source.keys[*cursor] == key {
                    let range =
                        source.offsets[*cursor] as usize..source.offsets[*cursor + 1] as usize;
                    piece.outputs.extend_from_slice(&source.outputs[range]);
                    *cursor += 1;
                    spilled = true;
                }
            }
            if spilled {
                piece.outputs[start..].sort_unstable();
            }
            piece.branch_offsets.push(piece.outputs.len() as u32);
            branch += 1;
        }
    }
    piece
}
