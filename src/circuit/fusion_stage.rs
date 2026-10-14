//! Tile staging for states past the last-level cache: [`plan_stages`] schedules the
//! stream `fuse_same_pair_2q_blocks` leaves into stages that each take one pass of an
//! existing batch kernel. The pass pipeline lives in [`crate::circuit::fusion`].

use std::borrow::Cow;
use std::sync::OnceLock;

use num_complex::Complex64;

use super::fusion::{fuse_diagonal_batch, gate_1q_matrix, push_unique};
use super::fusion_phase::fuse_controlled_phases;
use super::fusion_rzz::fuse_batch_rzz;
use super::plan::{Place, Tracer, WHOLE};
use super::{Circuit, Instruction, SmallVec, smallvec};
use crate::gates::{
    BatchRzzData, Gate, Multi2qData, MultiFusedData, MultiPauliRotData, is_diagonal_2x2,
    is_diagonal_4x4, kron_2x2, mat_mul_2x2, mat_mul_4x4, multi_2q_high_budget_for,
    multi_2q_low_bits, pauli_rot_masks,
};
use crate::sim::unified_pauli::PauliAxis;

/// Stage floor when the OS reports no last-level cache: an 8 MB cache, the host the
/// batching passes were tuned on.
const DEFAULT_STAGE_FLOOR: usize = 20;
/// Lowest floor a small cache can derive, the floor of the diagonal batches.
const MIN_STAGE_FLOOR: usize = 16;

/// Narrowest state the planner schedules: the first width whose amplitudes outgrow the
/// last-level cache, 20 qubits on an 8 MB L3. Below it a pass costs a cache sweep, not a
/// memory one, and the batching passes keep the stream.
fn stage_floor() -> usize {
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::backend::cache::last_level_cache()
            .map_or(DEFAULT_STAGE_FLOOR, |llc| {
                (llc.bytes / size_of::<Complex64>()).max(1).ilog2() as usize + 1
            })
            .max(MIN_STAGE_FLOOR)
    })
}

/// Whether fusion stages a `circuit_qubits` stream for an `n`-qubit buffer. A density
/// matrix sweeps a buffer twice the circuit width and keeps the batching passes.
pub(super) fn stages_at(n: usize, circuit_qubits: usize) -> bool {
    n == circuit_qubits && n <= u64::BITS as usize && n >= stage_floor()
}

/// The tile one stage gathers: qubits in `high` cost one of `budget` slots, the
/// contiguous low run below them is free.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TileGeometry {
    high: u64,
    budget: u32,
}

impl TileGeometry {
    /// The `Multi2q` subcube on an `n`-qubit state, the tile every batch route gathers.
    pub(super) fn for_width(n: usize) -> Self {
        let low = multi_2q_low_bits().min(n);
        Self {
            high: mask_below(n) & !mask_below(low),
            budget: multi_2q_high_budget_for(n) as u32,
        }
    }

    #[cfg(test)]
    pub(super) fn new(n: usize, low: usize, budget: u32) -> Self {
        Self {
            high: mask_below(n) & !mask_below(low),
            budget,
        }
    }
}

fn mask_below(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

fn qubit_mask(qubits: &[usize]) -> u64 {
    qubits.iter().fold(0, |m, &q| m | 1 << q)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    /// A one-qubit gate with a 2x2 matrix.
    One,
    /// A two-qubit gate with a 4x4 matrix, or one edge of a `BatchRzz`.
    Two,
    /// A `PauliRot` on two or more qubits.
    Pauli,
}

/// One schedulable gate. `nondiag` holds the qubits it moves; two ops commute when
/// every qubit they share is in neither one's `nondiag`.
#[derive(Clone, Copy, Debug)]
struct Op {
    src: u32,
    entry: u32,
    class: Class,
    touched: u64,
    nondiag: u64,
    /// Diagonal and expressible as diagonal batch entries.
    batchable: bool,
    /// Longest dependency chain from this op to the end of its segment.
    height: u32,
    /// Diagonal, and no later op in its segment moves a qubit it touches.
    terminal: bool,
}

/// Score of a taken op that moves a qubit, against 1 for a diagonal one: a tile stage
/// earns its pass by its moving ops, and its diagonal ops only break ties.
const MOVING: usize = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// One `Multi2q`, or one `MultiFused` when the stage holds no 2q gate.
    Tile,
    /// One `MultiPauliRot`; Z letters outside the tile cost nothing.
    Pauli,
    /// Diagonal batches, which need no tile.
    Diag,
}

/// Schedule `input` into stages for a state of its own width.
pub(super) fn plan_stages<'a>(input: Cow<'a, Circuit>, t: &mut Tracer) -> Cow<'a, Circuit> {
    let geometry = TileGeometry::for_width(input.num_qubits);
    plan_with(input, geometry, t)
}

/// Schedule `input` into stages of `geometry`.
///
/// Measurements, resets, conditionals, regions, barriers, saves and every gate the
/// batch kernels cannot carry (a `QftBlock`, an `Mcu`, a wide `Unitary`) end a
/// segment and keep their place. Inside a segment each stage takes, in stream order,
/// the ops whose dependencies are met and whose qubits fit its tile. A tile or Pauli
/// stage grows its high qubits from nothing, each step adding the frontier op's
/// qubits that let the stage move the most ops per qubit added, ties to the op with
/// the longest chain behind it, then the earliest. Diagonal ops ride in a tile when a
/// later op needs them and otherwise wait for a diagonal batch, taken when nothing
/// else moves or when clearing them first lets the next stage move twice as much.
/// The schedule reads only the stream and `geometry`. Returns the input when nothing
/// moves or batches.
pub(super) fn plan_with<'a>(
    input: Cow<'a, Circuit>,
    geometry: TileGeometry,
    t: &mut Tracer,
) -> Cow<'a, Circuit> {
    let circuit = input.as_ref();
    let mut planner = Planner::new(&circuit.instructions, circuit.num_qubits, geometry);
    t.begin();
    for (i, inst) in circuit.instructions.iter().enumerate() {
        if !planner.classify(inst, i, t) {
            planner.plan_segment(t);
            planner.keep(i, t);
        }
    }
    planner.plan_segment(t);

    if planner.changed {
        t.commit();
        Cow::Owned(circuit.with_instructions(planner.out))
    } else {
        t.discard();
        input
    }
}

/// A payload source for the tracer: instruction, entry, placement.
type Source = (u32, u32, Place);

/// The 1q ops on one qubit a stage has folded so far, as their product.
struct Run1q {
    qubit: usize,
    mat: [[Complex64; 2]; 2],
    srcs: SmallVec<[u32; 2]>,
}

struct Planner<'a> {
    insts: &'a [Instruction],
    n: usize,
    geometry: TileGeometry,
    all: u64,
    skip_limit: usize,
    ops: Vec<Op>,
    rem: Vec<u32>,
    taken: Vec<u32>,
    front: Vec<(u64, u32)>,
    out: Vec<Instruction>,
    changed: bool,
    next_src: u32,
}

impl<'a> Planner<'a> {
    fn new(insts: &'a [Instruction], n: usize, geometry: TileGeometry) -> Self {
        Self {
            insts,
            n,
            geometry,
            all: mask_below(n),
            skip_limit: 4 * n,
            ops: Vec::new(),
            rem: Vec::new(),
            taken: Vec::new(),
            front: Vec::new(),
            out: Vec::with_capacity(insts.len()),
            changed: false,
            next_src: 0,
        }
    }

    /// Queue instruction `i` as one or more ops, or return false for an opaque one.
    fn classify(&mut self, inst: &Instruction, i: usize, t: &mut Tracer) -> bool {
        let Instruction::Gate { gate, targets } = inst else {
            return false;
        };
        let src = i as u32;
        let touched = qubit_mask(targets);
        let mut push = |class, nondiag: u64, batchable| {
            self.ops.push(Op {
                src,
                entry: WHOLE,
                class,
                touched,
                nondiag,
                batchable,
                height: 0,
                terminal: false,
            });
        };
        match gate {
            Gate::Id
            | Gate::X
            | Gate::Y
            | Gate::Z
            | Gate::H
            | Gate::S
            | Gate::Sdg
            | Gate::T
            | Gate::Tdg
            | Gate::SX
            | Gate::SXdg
            | Gate::Rx(_)
            | Gate::Ry(_)
            | Gate::Rz(_)
            | Gate::P(_)
            | Gate::Fused(_) => {
                let diagonal = gate.is_diagonal_1q();
                if matches!(gate, Gate::Fused(_)) {
                    t.guard_diag_2x2(i, diagonal);
                }
                push(Class::One, if diagonal { 0 } else { touched }, diagonal);
            }
            Gate::PauliRot(data) if data.axes.len() == 1 => {
                push(Class::One, touched, false);
            }
            Gate::PauliRot(data) => {
                let moved: SmallVec<[usize; 4]> = targets
                    .iter()
                    .zip(data.axes())
                    .filter(|(_, axis)| !matches!(axis, PauliAxis::Z))
                    .map(|(&q, _)| q)
                    .collect();
                push(Class::Pauli, qubit_mask(&moved), false);
            }
            Gate::Cz => push(Class::Two, 0, true),
            Gate::Rzz(_) => {
                t.require_gate_4x4(i, WHOLE);
                push(Class::Two, 0, true);
            }
            Gate::Cx => push(Class::Two, 1 << targets[1], false),
            Gate::Swap => push(Class::Two, touched, false),
            Gate::Cu(mat) => {
                let diagonal = is_diagonal_2x2(mat);
                push(
                    Class::Two,
                    if diagonal { 0 } else { 1 << targets[1] },
                    diagonal,
                );
            }
            Gate::Fused2q(mat) => {
                let diagonal = is_diagonal_4x4(mat);
                t.guard_diag_4x4(&[(i, Place::Plain)], diagonal);
                push(Class::Two, if diagonal { 0 } else { touched }, diagonal);
            }
            Gate::BatchRzz(data) => {
                for (entry, &(q0, q1, _)) in data.edges.iter().enumerate() {
                    t.require_gate_4x4(i, entry as u32);
                    self.ops.push(Op {
                        src,
                        entry: entry as u32,
                        class: Class::Two,
                        touched: 1 << q0 | 1 << q1,
                        nondiag: 0,
                        batchable: true,
                        height: 0,
                        terminal: false,
                    });
                }
            }
            _ => return false,
        }
        true
    }

    /// Emit opaque instruction `i` in place.
    fn keep(&mut self, i: usize, t: &mut Tracer) {
        self.note_order(i as u32, WHOLE);
        self.out.push(self.insts[i].clone());
        t.keep(i);
    }

    /// Track whether the output still reads as the input, op for op.
    fn note_order(&mut self, src: u32, entry: u32) {
        if entry != WHOLE || src < self.next_src {
            self.changed = true;
        }
        self.next_src = src + 1;
    }

    /// Fill each op's height and terminal flag. A dependency chain only ever passes
    /// through a later op that shares a qubit, moved by one of the two, so a running
    /// maximum per qubit over the ops after it gives the longest chain.
    fn fill_heights(&mut self) {
        let mut any = vec![0u32; self.n];
        let mut moved = vec![0u32; self.n];
        let mut moved_later = 0u64;
        for op in self.ops.iter_mut().rev() {
            op.terminal = op.nondiag == 0 && op.touched & moved_later == 0;
            moved_later |= op.nondiag;
            let mut next = 0;
            for q in bits(op.touched) {
                next = next.max(if op.nondiag >> q & 1 == 1 {
                    any[q]
                } else {
                    moved[q]
                });
            }
            op.height = next + 1;
            for q in bits(op.touched) {
                any[q] = any[q].max(op.height);
                if op.nondiag >> q & 1 == 1 {
                    moved[q] = moved[q].max(op.height);
                }
            }
        }
    }

    /// The qubits `op` still needs in a `kind` stage with high qubits `h`, 0 when it
    /// fits, `u64::MAX` when no tile of that kind takes it.
    #[inline(always)]
    fn missing(&self, op: &Op, kind: Kind, h: u64) -> u64 {
        match (kind, op.class) {
            (Kind::Tile, Class::One | Class::Two) => op.touched & self.geometry.high & !h,
            (Kind::Pauli, Class::Pauli) => op.nondiag & self.geometry.high & !h,
            (Kind::Diag, _) if op.batchable => 0,
            _ => u64::MAX,
        }
    }

    /// Ops a `kind` stage with high qubits `h` takes from `rem`, in order, into
    /// `taken`: each one that fits and conflicts with no op skipped before it. Fills
    /// `front`, with the qubits each would add, with the skipped ops nothing blocked.
    fn scan(
        &self,
        kind: Kind,
        h: u64,
        taken: &mut Vec<u32>,
        mut front: Option<&mut Vec<(u64, u32)>>,
    ) {
        taken.clear();
        let room = self.geometry.budget - h.count_ones();
        let (mut blocked_touched, mut blocked_moved) = (0u64, 0u64);
        let mut skipped = 0;
        for &k in &self.rem {
            let op = &self.ops[k as usize];
            let blocked = (op.touched & blocked_moved) | (op.nondiag & blocked_touched) != 0;
            if !blocked {
                let missing = self.missing(op, kind, h);
                if missing == 0 {
                    taken.push(k);
                    continue;
                }
                if let Some(front) = front.as_deref_mut()
                    && missing.count_ones() <= room
                {
                    front.push((missing, op.height));
                }
            }
            blocked_touched |= op.touched;
            blocked_moved |= op.nondiag;
            skipped += 1;
            if skipped > self.skip_limit || blocked_moved == self.all {
                break;
            }
        }
    }

    /// Score of a `kind` stage over `taken`: [`MOVING`] per op it keeps that moves a
    /// qubit and 1 per diagonal one, after a tile stage defers what it would not keep.
    fn score(&self, kind: Kind, taken: &[u32]) -> usize {
        let weight = |op: &Op| if op.nondiag != 0 { MOVING } else { 1 };
        let all = || taken.iter().map(|&k| weight(&self.ops[k as usize])).sum();
        if kind != Kind::Tile {
            return all();
        }
        let paired = self.paired(taken);
        let score: usize = taken
            .iter()
            .map(|&k| &self.ops[k as usize])
            .filter(|op| Self::keeps(op, paired))
            .map(weight)
            .sum();
        if score > 0 { score } else { all() }
    }

    /// Grow the high qubits of a `kind` stage from nothing: add the frontier op's
    /// missing qubits that raise the stage's score most per qubit added, while any
    /// raises it. Returns the high qubits and the score.
    fn grow(&mut self, kind: Kind) -> (u64, usize) {
        let mut h = 0u64;
        let mut front = std::mem::take(&mut self.front);
        let mut taken = Vec::new();
        front.clear();
        self.scan(kind, h, &mut taken, Some(&mut front));
        let mut base = self.score(kind, &taken) as i64;
        loop {
            let mut best: Option<(u64, i64, u32)> = None;
            let mut seen: SmallVec<[u64; 16]> = SmallVec::new();
            for &(missing, height) in &front {
                if (h | missing).count_ones() > self.geometry.budget || seen.contains(&missing) {
                    continue;
                }
                seen.push(missing);
                self.scan(kind, h | missing, &mut taken, None);
                let score = self.score(kind, &taken) as i64;
                if score <= base {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((b_missing, b_score, b_height)) => {
                        let gain = (score - base) * b_missing.count_ones() as i64;
                        let b_gain = (b_score - base) * missing.count_ones() as i64;
                        gain > b_gain || (gain == b_gain && height > b_height)
                    }
                };
                if better {
                    best = Some((missing, score, height));
                }
            }
            let Some((missing, score, _)) = best else {
                break;
            };
            h |= missing;
            base = score;
            front.clear();
            self.scan(kind, h, &mut taken, Some(&mut front));
        }
        self.front = front;
        (h, base as usize)
    }

    fn plan_segment(&mut self, t: &mut Tracer) {
        if self.ops.is_empty() {
            return;
        }
        self.fill_heights();
        self.rem.clear();
        self.rem.extend(0..self.ops.len() as u32);
        while !self.rem.is_empty() {
            let (mut unit, mut pauli, mut diag) = (false, false, false);
            for &k in &self.rem {
                let op = &self.ops[k as usize];
                unit |= op.class != Class::Pauli;
                pauli |= op.class == Class::Pauli;
                diag |= op.batchable;
            }
            let tiled = self.best_tiled(unit, pauli);
            let mut taken = std::mem::take(&mut self.taken);
            let kind = self.choose(tiled, (unit, pauli, diag), &mut taken);
            self.emit(kind, &taken, t);
            let mut next = taken.iter().peekable();
            self.rem.retain(|k| {
                if next.peek() == Some(&k) {
                    next.next();
                    false
                } else {
                    true
                }
            });
            self.taken = taken;
        }
        self.ops.clear();
    }

    /// The tile or Pauli stage that scores highest on `rem`, as its kind, high qubits
    /// and score.
    fn best_tiled(&mut self, unit: bool, pauli: bool) -> (Kind, u64, usize) {
        let mut best = (Kind::Tile, 0u64, 0usize);
        for (kind, present) in [(Kind::Tile, unit), (Kind::Pauli, pauli)] {
            if present {
                let (h, score) = self.grow(kind);
                if score > best.2 {
                    best = (kind, h, score);
                }
            }
        }
        best
    }

    /// Fill `taken` with the next stage and return its kind. A tile or Pauli stage
    /// runs while it moves a qubit, unless a diagonal batch first would let the next
    /// one move more; otherwise diagonal ops wait, so one batch collects as many as
    /// it can.
    fn choose(
        &mut self,
        tiled: (Kind, u64, usize),
        (unit, pauli, diag): (bool, bool, bool),
        taken: &mut Vec<u32>,
    ) -> Kind {
        let (kind, h, score) = tiled;
        let mut diagonal = Vec::new();
        if diag {
            self.scan(Kind::Diag, 0, &mut diagonal, None);
        }
        if score >= MOVING {
            self.scan(kind, h, taken, None);
            if kind == Kind::Tile {
                self.defer_from_tile(taken);
            }
            if taken.iter().any(|&k| self.ops[k as usize].nondiag != 0) {
                if diagonal.is_empty() || !self.diagonal_unlocks(&diagonal, score, unit, pauli) {
                    return kind;
                }
                taken.clone_from(&diagonal);
                return Kind::Diag;
            }
        }
        if !diagonal.is_empty() {
            taken.clone_from(&diagonal);
            return Kind::Diag;
        }
        if score > 0 {
            self.scan(kind, h, taken, None);
            if kind == Kind::Tile {
                self.defer_from_tile(taken);
            }
            return kind;
        }
        taken.clear();
        taken.push(self.rem[0]);
        kind
    }

    /// Whether taking `diagonal` first lets the best tile or Pauli stage after it move
    /// at least twice the ops one scoring `score` moves now, enough to pay for the
    /// diagonal pass.
    fn diagonal_unlocks(
        &mut self,
        diagonal: &[u32],
        score: usize,
        unit: bool,
        pauli: bool,
    ) -> bool {
        let saved = self.rem.clone();
        let mut next = diagonal.iter().peekable();
        self.rem.retain(|k| {
            if next.peek() == Some(&k) {
                next.next();
                false
            } else {
                true
            }
        });
        let after = self.best_tiled(unit, pauli).2;
        self.rem = saved;
        after / MOVING >= 2 * (score / MOVING)
    }

    /// The qubits a tile stage over `taken` keeps 2q gates on: every one but a
    /// terminal diagonal, which a diagonal batch takes at the end without a dense 4x4
    /// per tile.
    fn paired(&self, taken: &[u32]) -> u64 {
        taken
            .iter()
            .map(|&k| &self.ops[k as usize])
            .filter(|op| op.class == Class::Two && !op.terminal)
            .fold(0, |m, op| m | op.touched)
    }

    /// Whether a tile stage whose kept 2q gates touch `paired` keeps `op`. It leaves a
    /// terminal diagonal 2q gate and, in a stage holding 2q gates, a 1q gate on a
    /// qubit none of them touch, which would need a second pass; a later stage can
    /// still take it.
    fn keeps(op: &Op, paired: u64) -> bool {
        !(op.class == Class::Two && op.terminal)
            && (op.class != Class::One || paired == 0 || op.touched & paired != 0)
    }

    /// Drop from a tile stage what [`Self::keeps`] leaves out, unless that would empty
    /// it.
    fn defer_from_tile(&self, taken: &mut Vec<u32>) {
        let paired = self.paired(taken);
        if taken
            .iter()
            .any(|&k| Self::keeps(&self.ops[k as usize], paired))
        {
            taken.retain(|&k| Self::keeps(&self.ops[k as usize], paired));
        }
    }

    fn emit(&mut self, kind: Kind, taken: &[u32], t: &mut Tracer) {
        let ops: SmallVec<[Op; 32]> = taken.iter().map(|&k| self.ops[k as usize]).collect();
        for op in &ops {
            self.note_order(op.src, op.entry);
        }
        if ops.len() == 1 {
            self.emit_single(&ops[0], t);
            return;
        }
        self.changed = true;
        match kind {
            Kind::Tile if ops.iter().any(|op| op.class == Class::Two) => {
                self.emit_multi_2q(&ops, t)
            }
            Kind::Tile => self.emit_multi_1q(&ops, t),
            Kind::Pauli => self.emit_multi_pauli(&ops, t),
            Kind::Diag => self.emit_diagonal(&ops, t),
        }
    }

    fn emit_single(&mut self, op: &Op, t: &mut Tracer) {
        if op.entry == WHOLE {
            self.out.push(self.insts[op.src as usize].clone());
            t.keep(op.src as usize);
        } else {
            let (q0, q1, theta) = self.edge(op);
            self.out.push(Instruction::Gate {
                gate: Gate::Rzz(theta),
                targets: smallvec![q0, q1],
            });
            t.emit_entries(&[vec![(op.src, op.entry, Place::Plain)]]);
        }
    }

    fn gate(&self, op: &Op) -> (&'a Gate, &'a [usize]) {
        match &self.insts[op.src as usize] {
            Instruction::Gate { gate, targets } => (gate, targets),
            _ => unreachable!("ops name gate instructions"),
        }
    }

    fn edge(&self, op: &Op) -> (usize, usize, f64) {
        match self.gate(op) {
            (Gate::BatchRzz(data), _) => data.edges[op.entry as usize],
            _ => unreachable!("an entry op names a BatchRzz edge"),
        }
    }

    /// The op's pair, high bit first, and its 4x4.
    fn pair(&self, op: &Op) -> (usize, usize, [[Complex64; 4]; 4]) {
        if op.entry != WHOLE {
            let (q0, q1, theta) = self.edge(op);
            return (q0, q1, Gate::Rzz(theta).matrix_4x4());
        }
        let (gate, targets) = self.gate(op);
        (targets[0], targets[1], gate.matrix_4x4())
    }

    /// Fold 1q `op` into the run on its qubit in `runs`, starting one if none is open.
    fn fold_1q(&self, op: &Op, runs: &mut SmallVec<[Run1q; 8]>) {
        let (gate, targets) = self.gate(op);
        let mat = gate_1q_matrix(gate);
        match runs.iter_mut().find(|run| run.qubit == targets[0]) {
            Some(run) => {
                run.mat = mat_mul_2x2(&mat, &run.mat);
                run.srcs.push(op.src);
            }
            None => runs.push(Run1q {
                qubit: targets[0],
                mat,
                srcs: smallvec![op.src],
            }),
        }
    }

    /// One `Multi2q` entry per 2q op, with each 1q op folded into the next 2q op on
    /// its qubit, or into the last one when none follows.
    fn emit_multi_2q(&mut self, ops: &[Op], t: &mut Tracer) {
        let identity = Gate::Id.matrix_2x2();
        let mut last: SmallVec<[(usize, usize); 16]> = SmallVec::new();
        let mut gates: Vec<(usize, usize, [[Complex64; 4]; 4])> = Vec::new();
        let mut srcs: Vec<Vec<Source>> = Vec::new();
        let mut queued: SmallVec<[Run1q; 8]> = SmallVec::new();
        for op in ops {
            if op.class == Class::One {
                self.fold_1q(op, &mut queued);
                continue;
            }
            let (q0, q1, mut mat) = self.pair(op);
            let mut take = |q: usize| {
                queued
                    .iter()
                    .position(|run| run.qubit == q)
                    .map(|at| queued.swap_remove(at))
            };
            let (pre0, pre1) = (take(q0), take(q1));
            if pre0.is_some() || pre1.is_some() {
                let m0 = pre0.as_ref().map_or(identity, |run| run.mat);
                let m1 = pre1.as_ref().map_or(identity, |run| run.mat);
                mat = mat_mul_4x4(&mat, &kron_2x2(&m0, &m1));
            }
            let mut entry: Vec<Source> = Vec::new();
            if t.on {
                for (pre, place) in [(&pre0, Place::Low), (&pre1, Place::High)] {
                    if let Some(run) = pre {
                        entry.extend(run.srcs.iter().map(|&s| (s, WHOLE, place)));
                    }
                }
                entry.push((op.src, op.entry, Place::Plain));
            }
            for q in [q0, q1] {
                match last.iter_mut().find(|l| l.0 == q) {
                    Some(l) => l.1 = gates.len(),
                    None => last.push((q, gates.len())),
                }
            }
            gates.push((q0, q1, mat));
            srcs.push(entry);
        }
        for run in queued {
            let at = last
                .iter()
                .find(|l| l.0 == run.qubit)
                .map(|l| l.1)
                .expect("a lone 1q gate left the stage");
            let (q0, q1, ref mut acc) = gates[at];
            let (embedded, place) = if run.qubit == q0 {
                (kron_2x2(&run.mat, &identity), Place::Low)
            } else {
                debug_assert_eq!(run.qubit, q1);
                (kron_2x2(&identity, &run.mat), Place::High)
            };
            *acc = mat_mul_4x4(&embedded, acc);
            if t.on {
                srcs[at].extend(run.srcs.iter().map(|&s| (s, WHOLE, place)));
            }
        }
        if gates.len() == 1 {
            let (q0, q1, mat) = gates[0];
            self.out.push(Instruction::Gate {
                gate: Gate::Fused2q(Box::new(mat)),
                targets: smallvec![q0, q1],
            });
            t.emit_entries(&srcs);
            return;
        }
        let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
        for &(q0, q1, _) in &gates {
            push_unique(&mut targets, q0);
            push_unique(&mut targets, q1);
        }
        targets.sort_unstable();
        self.out.push(Instruction::Gate {
            gate: Gate::Multi2q(Box::new(Multi2qData { gates })),
            targets,
        });
        t.emit_entries(&srcs);
    }

    /// One `MultiFused` entry per qubit, the product of its ops in order, by
    /// ascending qubit; a single qubit becomes one `Fused`.
    fn emit_multi_1q(&mut self, ops: &[Op], t: &mut Tracer) {
        let mut runs: SmallVec<[Run1q; 8]> = SmallVec::new();
        for op in ops {
            self.fold_1q(op, &mut runs);
        }
        runs.sort_unstable_by_key(|run| run.qubit);
        let entries: Vec<Vec<Source>> = runs
            .iter()
            .map(|run| run.srcs.iter().map(|&s| (s, WHOLE, Place::Plain)).collect())
            .collect();
        if let [run] = runs.as_slice() {
            self.out.push(Instruction::Gate {
                gate: Gate::Fused(Box::new(run.mat)),
                targets: smallvec![run.qubit],
            });
            t.emit_entries(&entries);
            return;
        }
        let targets: SmallVec<[usize; 4]> = runs.iter().map(|run| run.qubit).collect();
        self.out.push(Instruction::Gate {
            gate: Gate::MultiFused(Box::new(MultiFusedData::new(
                runs.iter().map(|run| (run.qubit, run.mat)).collect(),
            ))),
            targets,
        });
        t.emit_entries(&entries);
    }

    fn emit_multi_pauli(&mut self, ops: &[Op], t: &mut Tracer) {
        let mut rotations = Vec::with_capacity(ops.len());
        for op in ops {
            let (Gate::PauliRot(data), targets) = self.gate(op) else {
                unreachable!("a Pauli stage holds Pauli rotations");
            };
            let (xmask, zmask, _) = pauli_rot_masks(targets, &data.axes);
            rotations.push((xmask, zmask, data.theta));
        }
        let data = MultiPauliRotData { rotations };
        let mut support = data.support();
        let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
        while support != 0 {
            targets.push(support.trailing_zeros() as usize);
            support &= support - 1;
        }
        self.out.push(Instruction::Gate {
            gate: Gate::MultiPauliRot(Box::new(data)),
            targets,
        });
        let entries: Vec<Vec<Source>> = ops
            .iter()
            .map(|op| vec![(op.src, WHOLE, Place::Plain)])
            .collect();
        t.emit_entries(&entries);
    }

    /// A run of `Rzz` edges becomes `BatchRzz` chunks; other mixes use the diagonal passes.
    fn emit_diagonal(&mut self, ops: &[Op], t: &mut Tracer) {
        let is_rzz = |op: &Op| op.entry != WHOLE || matches!(self.gate(op).0, Gate::Rzz(_));
        if ops.iter().all(is_rzz) {
            for chunk in ops.chunks(BatchRzzData::MAX_EDGES) {
                if let [single] = chunk {
                    self.emit_single(single, t);
                    continue;
                }
                let mut edges = Vec::with_capacity(chunk.len());
                let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
                for op in chunk {
                    let edge = if op.entry == WHOLE {
                        let (gate, targets) = self.gate(op);
                        let Gate::Rzz(theta) = gate else {
                            unreachable!("checked above")
                        };
                        (targets[0], targets[1], *theta)
                    } else {
                        self.edge(op)
                    };
                    push_unique(&mut targets, edge.0);
                    push_unique(&mut targets, edge.1);
                    edges.push(edge);
                }
                self.out.push(Instruction::Gate {
                    gate: Gate::BatchRzz(Box::new(BatchRzzData { edges })),
                    targets,
                });
                let entries: Vec<Vec<Source>> = chunk
                    .iter()
                    .map(|op| vec![(op.src, op.entry, Place::Plain)])
                    .collect();
                t.emit_entries(&entries);
            }
            return;
        }
        let sources: Vec<Source> = if t.on {
            ops.iter()
                .map(|op| (op.src, op.entry, Place::Plain))
                .collect()
        } else {
            Vec::new()
        };
        let mut trace = t.subpass(&sources);
        let mut run: Vec<Instruction> = Vec::with_capacity(ops.len());
        for op in ops {
            if op.entry == WHOLE {
                run.push(self.insts[op.src as usize].clone());
            } else {
                let (q0, q1, theta) = self.edge(op);
                run.push(Instruction::Gate {
                    gate: Gate::Rzz(theta),
                    targets: smallvec![q0, q1],
                });
            }
        }
        let mut circuit = Circuit::new(self.n, 0);
        circuit.instructions = run;
        let rzz = fuse_batch_rzz(&circuit, &mut trace).into_owned();
        let batched = fuse_diagonal_batch(
            fuse_controlled_phases(Cow::Owned(rzz), &mut trace),
            &mut trace,
        );
        self.out.extend(batched.into_owned().instructions);
        t.finish_subpass(trace);
    }
}

fn bits(mut mask: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let q = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            q
        })
    })
}

#[cfg(test)]
#[path = "fusion_stage_tests.rs"]
mod tests;
