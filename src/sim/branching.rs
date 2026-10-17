//! Noiseless mid-circuit shots and dynamic programs on the host statevector by
//! outcome branching: each measurement splits the shots binomially and evolves
//! every outcome once.

use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use super::compiled::rng::{Xoshiro256PlusPlus, binomial_sample};
use super::dynamic::Prepared;
use super::{
    ShotsResult, apply_recording_saves, backend_metadata, mix_seed, read_save,
    sample_counts_from_state,
};
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::dynamic::ClassicalValue;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::limits::ResolvedLimits;

/// Stream of the run seed whose generator draws the root's outcome splits.
const SPLIT_STREAM: u64 = u64::MAX;
/// Stream of the run seed whose generator shuffles the finished rows.
const SHUFFLE_STREAM: u64 = u64::MAX - 1;

/// Width from which the two branches of a split may run on separate workers.
/// Below it a branch evolves in about the time fork-join takes to dispatch it.
#[cfg(feature = "parallel")]
pub(super) const MIN_QUBITS_FOR_PARALLEL_SPLITS: usize = 10;

/// Splits that may hand their two branches to separate workers, counted from
/// the root. Past it the walk stays on the worker it reached.
#[cfg(feature = "parallel")]
const MAX_PARALLEL_SPLIT_DEPTH: usize = 12;

/// Outcome record and the number of shots that ended on it.
type Leaf = (Vec<bool>, usize);

/// How [`branch_shots`] may spend memory and threads.
pub(super) struct BranchLimits {
    /// States of the circuit's width the memory budget holds at once.
    pub(super) max_live_states: usize,
    /// Whether the two branches of a split may run on separate workers.
    #[cfg(feature = "parallel")]
    pub(super) parallel: bool,
    /// Caps every branch state is built under.
    caps: ResolvedLimits,
}

impl BranchLimits {
    pub(super) fn for_width(num_qubits: usize, parallel: bool, caps: &ResolvedLimits) -> Self {
        let headroom = caps.statevector_qubits.value.saturating_sub(num_qubits);
        let max_live_states = u32::try_from(headroom)
            .ok()
            .and_then(|shift| 1usize.checked_shl(shift))
            .unwrap_or(usize::MAX);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        Self {
            max_live_states,
            #[cfg(feature = "parallel")]
            parallel,
            caps: *caps,
        }
    }
}

/// Run `num_shots` shots of `fused` from `|0...0>` on the host statevector.
///
/// At a measurement, or at a reset that a later measurement can observe, the
/// shots still on a branch split by an exact Binomial(shots, P(1)) draw and
/// each non-empty outcome evolves once, so a circuit costs one evolution per
/// distinct outcome history rather than one per shot. Draws come from a
/// generator seeded by the run seed and the branch's outcome path, never by
/// the worker, so the shots do not depend on thread count.
///
/// The walk recurses into the branch with fewer shots and continues on the
/// other, cloning a state only at a split, so a path holds at most
/// `log2(num_shots) + 1` states plus one per parallel split above it. A split
/// that would pass
/// [`BranchLimits::max_live_states`] runs its shots one at a time from the
/// branch state instead. Instructions after the last one that writes a
/// classical bit or reads a save are never applied: nothing they do reaches a
/// result. Once only measurements remain, two or more of them, a branch draws
/// all of its shots from its state's distribution over the measured qubits in
/// one pass, with no further splits or clones.
///
/// Branching groups shots by outcome, so the rows are shuffled with a seeded
/// Fisher-Yates pass when `shuffle` is set; a caller that only counts them
/// skips it.
pub(super) fn branch_shots(
    fused: &Circuit,
    num_shots: usize,
    seed: u64,
    limits: &BranchLimits,
    shuffle: bool,
) -> Result<ShotsResult> {
    let mut cursor = Cursor::default();
    cursor.enter_last(&fused.instructions);
    let walk = Walk {
        limits,
        program: None,
    };
    walk.shots(
        fused.num_qubits,
        fused.num_classical_bits,
        cursor,
        num_shots,
        seed,
        shuffle,
    )
}

/// Run `num_shots` shots of `program` from `|0...0>` by the walk of
/// [`branch_shots`], each branch following the control-flow graph.
///
/// Every classical value, loop test, and branch condition a shot meets is a
/// function of its outcome history, so all shots on a branch take one path
/// through the graph. Each branch counts the blocks it enters against the
/// program's step bound, as a shot does. Only the block that ends the graph
/// with no actions is trimmed after its last measurement or sampled as a
/// terminal readout; anywhere else a later block may still read the state.
pub(super) fn branch_program_shots(
    program: &Prepared<'_>,
    num_shots: usize,
    seed: u64,
    limits: &BranchLimits,
) -> Result<ShotsResult> {
    let mut cursor = Cursor::default();
    if num_shots > 0 {
        program.count_step(0, &mut cursor.steps)?;
    }
    cursor.vars.extend_from_slice(program.initial_vars());
    cursor.enter_block(program, 0);
    let walk = Walk {
        limits,
        program: Some(program),
    };
    walk.shots(
        program.num_qubits(),
        program.num_classical_bits(),
        cursor,
        num_shots,
        seed,
        true,
    )
}

fn stream_rng(seed: u64, stream: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    rng
}

/// Whether `inst` needs the quantum state to produce what it does: a
/// measurement, a save, or a region holding either.
fn needs_state(inst: &Instruction) -> bool {
    match inst {
        Instruction::Measure { .. } | Instruction::Save { .. } => true,
        Instruction::Region(region) => region.body().iter().any(needs_state),
        _ => false,
    }
}

/// Start of the run of measurements and barriers that ends `body`.
fn terminal_tail_start(body: &[Instruction]) -> usize {
    body.iter()
        .rposition(|inst| {
            !matches!(
                inst,
                Instruction::Measure { .. } | Instruction::Barrier { .. }
            )
        })
        .map_or(0, |last| last + 1)
}

fn draw_ones(rng: &mut ChaCha8Rng, shots: usize, prob_one: f64) -> usize {
    let mut fast = Xoshiro256PlusPlus::from_chacha(rng);
    binomial_sample(&mut fast, shots, prob_one)
}

/// Position in the instruction stream: the top-level list, then the body of
/// each region entered and not yet finished. In a program the top-level list is
/// the current block's circuit, and the cursor carries the shot's classical
/// state with it.
#[derive(Clone, Default)]
struct Cursor<'a> {
    frames: SmallVec<[(&'a [Instruction], usize); 2]>,
    /// Top-level index from which nothing ahead needs the state.
    live_end: usize,
    /// Top-level index where the closing run of measurements starts,
    /// `usize::MAX` when the top-level list does not end the shot.
    tail_start: usize,
    block: usize,
    vars: Vec<ClassicalValue>,
    /// Blocks entered so far.
    steps: u64,
}

impl<'a> Cursor<'a> {
    fn enter(&mut self, body: &'a [Instruction]) {
        self.frames.push((body, 0));
    }

    /// Enter a top-level list that ends the shot, cut after its last
    /// instruction that needs the state: nothing past it reaches a result.
    fn enter_last(&mut self, body: &'a [Instruction]) {
        let live_end = body
            .iter()
            .rposition(needs_state)
            .map_or(0, |last| last + 1);
        let live = &body[..live_end];
        self.live_end = live_end;
        self.tail_start = terminal_tail_start(live);
        self.frames.push((live, 0));
    }

    fn enter_block(&mut self, program: &'a Prepared<'_>, at: usize) {
        self.block = at;
        match program.block_body(at) {
            (body, true) => self.enter_last(body),
            (body, false) => {
                self.live_end = usize::MAX;
                self.tail_start = usize::MAX;
                self.frames.push((body, 0));
            }
        }
    }

    fn next(&mut self) -> Option<&'a Instruction> {
        while let Some((body, idx)) = self.frames.last_mut() {
            let body: &'a [Instruction] = body;
            if let Some(inst) = body.get(*idx) {
                *idx += 1;
                return Some(inst);
            }
            self.frames.pop();
        }
        None
    }

    /// Step back over the instruction [`Self::next`] just returned.
    fn rewind(&mut self) {
        if let Some((_, idx)) = self.frames.last_mut() {
            *idx -= 1;
        }
    }

    /// Whether an instruction still ahead needs the state.
    fn state_ahead(&self) -> bool {
        self.frames.iter().enumerate().any(|(depth, &(body, idx))| {
            if depth == 0 {
                idx < self.live_end
            } else {
                body[idx..].iter().any(needs_state)
            }
        })
    }

    /// Whether the instruction [`Self::next`] just returned opens a stretch of
    /// measurements and barriers that ends the shot.
    fn in_terminal_tail(&self) -> bool {
        let last = self.frames.len() - 1;
        self.frames.iter().enumerate().all(|(depth, &(body, idx))| {
            let from = if depth == last { idx - 1 } else { idx };
            let start = if depth == 0 {
                self.tail_start
            } else {
                terminal_tail_start(body)
            };
            from >= start
        })
    }

    /// Whether the instruction [`Self::next`] just returned opens a terminal
    /// stretch holding at least one more measurement. A lone last measurement
    /// stays on the split path, where it costs one marginal pass and one
    /// binomial draw; the sampler would pay a keyed pass over the state and a
    /// draw per shot for it.
    fn opens_sampled_tail(&self) -> bool {
        self.in_terminal_tail()
            && self.frames.iter().any(|&(body, idx)| {
                body[idx..]
                    .iter()
                    .any(|inst| matches!(inst, Instruction::Measure { .. }))
            })
    }
}

/// One branch of the walk: its state, the shots that followed it, and the
/// generator its outcome splits draw from.
struct Branch {
    sv: StatevectorBackend,
    shots: usize,
    rng: ChaCha8Rng,
}

struct Walk<'l> {
    limits: &'l BranchLimits,
    program: Option<&'l Prepared<'l>>,
}

impl<'l> Walk<'l> {
    fn shots(
        &self,
        num_qubits: usize,
        num_classical_bits: usize,
        cursor: Cursor<'l>,
        num_shots: usize,
        seed: u64,
        shuffle: bool,
    ) -> Result<ShotsResult> {
        let mut root = StatevectorBackend::with_limits(seed, &self.limits.caps);
        let metadata = backend_metadata(&root);
        let bits = num_classical_bits;
        if num_shots == 0 {
            return Ok(ShotsResult::from_shots(Vec::new(), bits).with_metadata(metadata));
        }
        root.init(num_qubits, bits)?;

        let root = Branch {
            sv: root,
            shots: num_shots,
            rng: stream_rng(seed, SPLIT_STREAM),
        };
        let mut leaves = Vec::new();
        self.run(root, cursor, 1, 0, &mut leaves)?;

        // Traversal order depends on which branches ran in parallel, so the rows
        // are laid out in record order; equal records are interchangeable.
        leaves.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut shots = Vec::with_capacity(num_shots);
        for (record, count) in leaves {
            shots.extend(std::iter::repeat_n(record, count));
        }
        if shuffle {
            let mut rng = stream_rng(seed, SHUFFLE_STREAM);
            for i in (1..shots.len()).rev() {
                shots.swap(i, rng.random_range(0..=i));
            }
        }
        Ok(ShotsResult::from_shots(shots, bits).with_metadata(metadata))
    }

    /// The next instruction of the shot, running the actions and terminator of
    /// each program block the cursor finishes on the way.
    fn next(
        &self,
        cursor: &mut Cursor<'l>,
        sv: &mut StatevectorBackend,
    ) -> Result<Option<&'l Instruction>> {
        loop {
            if let Some(inst) = cursor.next() {
                return Ok(Some(inst));
            }
            let Some(program) = self.program else {
                return Ok(None);
            };
            let Some(at) = program.finish_block(cursor.block, sv, &mut cursor.vars)? else {
                return Ok(None);
            };
            program.count_step(at, &mut cursor.steps)?;
            cursor.enter_block(program, at);
        }
    }

    /// Carry `branch` from `cursor` to the end, pushing one leaf per outcome
    /// history. `live` counts the states held on this path, the branch's own
    /// included, and `depth` the splits above it.
    fn run(
        &self,
        mut branch: Branch,
        mut cursor: Cursor<'l>,
        live: usize,
        depth: usize,
        leaves: &mut Vec<Leaf>,
    ) -> Result<()> {
        #[cfg(not(feature = "parallel"))]
        let _ = depth;
        while let Some(inst) = self.next(&mut cursor, &mut branch.sv)? {
            if matches!(inst, Instruction::Measure { .. }) && cursor.opens_sampled_tail() {
                sample_tail(&mut branch, inst, &cursor, leaves);
                return Ok(());
            }
            let sv = &mut branch.sv;
            let (qubit, classical_bit) = match inst {
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => (*qubit, Some(*classical_bit)),
                Instruction::Reset { qubit } => {
                    if !cursor.state_ahead() {
                        continue;
                    }
                    (*qubit, None)
                }
                Instruction::Region(region) => {
                    if region.condition().evaluate(&sv.classical_bits) {
                        cursor.enter(region.body());
                    }
                    continue;
                }
                Instruction::Save { spec, label, .. } => {
                    read_save(sv, *spec, label)?;
                    continue;
                }
                _ => {
                    sv.apply(inst)?;
                    continue;
                }
            };

            let prob_one = sv.qubit_probability(qubit)?.clamp(0.0, 1.0);
            let shots = branch.shots;
            let ones = draw_ones(&mut branch.rng, shots, prob_one);
            let state_ahead = cursor.state_ahead();
            if ones == 0 || ones == shots {
                let outcome = ones > 0;
                if let Some(bit) = classical_bit {
                    sv.classical_bits[bit] = outcome;
                }
                if state_ahead {
                    project(sv, qubit, classical_bit, outcome, prob_one);
                }
                continue;
            }

            if let (false, Some(bit)) = (state_ahead, classical_bit) {
                let mut record = std::mem::take(&mut sv.classical_bits);
                record[bit] = false;
                leaves.push((record.clone(), shots - ones));
                record[bit] = true;
                leaves.push((record, ones));
                return Ok(());
            }

            if live.saturating_add(2) > self.limits.max_live_states {
                cursor.rewind();
                return self.per_shot(&mut branch, &cursor, leaves);
            }

            let mut sv_one = StatevectorBackend::with_limits(0, &self.limits.caps);
            sv_one.copy_state_from(sv);
            if let Some(bit) = classical_bit {
                sv.classical_bits[bit] = false;
                sv_one.classical_bits[bit] = true;
            }
            project(sv, qubit, classical_bit, false, prob_one);
            project(&mut sv_one, qubit, classical_bit, true, prob_one);

            let seed_zero = branch.rng.next_u64();
            let seed_one = branch.rng.next_u64();
            let zero = Branch {
                sv: branch.sv,
                shots: shots - ones,
                rng: ChaCha8Rng::seed_from_u64(seed_zero),
            };
            let one = Branch {
                sv: sv_one,
                shots: ones,
                rng: ChaCha8Rng::seed_from_u64(seed_one),
            };
            let (small, large) = if one.shots <= zero.shots {
                (one, zero)
            } else {
                (zero, one)
            };

            #[cfg(feature = "parallel")]
            if self.limits.parallel && depth < MAX_PARALLEL_SPLIT_DEPTH {
                let small_cursor = cursor.clone();
                let mut small_leaves = Vec::new();
                let (small_run, large_run) = rayon::join(
                    || self.run(small, small_cursor, live + 1, depth + 1, &mut small_leaves),
                    || self.run(large, cursor, live + 1, depth + 1, leaves),
                );
                small_run?;
                large_run?;
                leaves.append(&mut small_leaves);
                return Ok(());
            }

            self.run(small, cursor.clone(), live + 1, depth + 1, leaves)?;
            branch = large;
        }
        leaves.push((branch.sv.classical_bits, branch.shots));
        Ok(())
    }

    /// Run each of the branch's shots on its own from its state and `cursor`,
    /// one scratch state reused across them, for a split the memory budget
    /// cannot hold.
    fn per_shot(
        &self,
        branch: &mut Branch,
        cursor: &Cursor<'_>,
        leaves: &mut Vec<Leaf>,
    ) -> Result<()> {
        let base = branch.rng.next_u64();
        let mut scratch = StatevectorBackend::with_limits(base, &self.limits.caps);
        let mut vars = Vec::new();
        for i in 0..branch.shots {
            scratch.copy_state_from(&branch.sv);
            scratch.rng = ChaCha8Rng::seed_from_u64(mix_seed(base, i));
            for &(body, idx) in cursor.frames.iter().rev() {
                apply_recording_saves(&mut scratch, &body[idx..])?;
            }
            if let Some(program) = self.program {
                vars.clone_from(&cursor.vars);
                program.run_from(cursor.block, cursor.steps, &mut scratch, &mut vars)?;
            }
            leaves.push((scratch.classical_bits.clone(), 1));
        }
        Ok(())
    }
}

fn project(
    sv: &mut StatevectorBackend,
    qubit: usize,
    classical_bit: Option<usize>,
    outcome: bool,
    prob_one: f64,
) {
    match classical_bit {
        Some(_) => sv.collapse_qubit(qubit, outcome, prob_one),
        None => sv.reset_qubit_from(qubit, outcome, prob_one),
    }
}

/// Draw the branch's shots for `first` and the measurements left in `cursor`
/// from its state, one leaf per distinct record.
fn sample_tail(
    branch: &mut Branch,
    first: &Instruction,
    cursor: &Cursor<'_>,
    leaves: &mut Vec<Leaf>,
) {
    let ahead = cursor
        .frames
        .iter()
        .rev()
        .flat_map(|&(body, idx)| &body[idx..]);
    let meas_map: Vec<(usize, usize)> = std::iter::once(first)
        .chain(ahead)
        .filter_map(|inst| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((*qubit, *classical_bit)),
            _ => None,
        })
        .collect();
    let sv = &branch.sv;
    let counts = sample_counts_from_state(
        sv.state_vector(),
        sv.probability_scale(),
        &meas_map,
        sv.classical_bits.len(),
        branch.shots,
        branch.rng.next_u64(),
        &sv.limits,
    );
    for (packed, count) in counts {
        let mut record = sv.classical_bits.clone();
        for &(_, bit) in &meas_map {
            record[bit] = (packed[bit / 64] >> (bit % 64)) & 1 == 1;
        }
        leaves.push((record, count as usize));
    }
}

#[cfg(test)]
#[path = "branching_tests.rs"]
mod tests;
