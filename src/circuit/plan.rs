//! Fusion plan capture and replay. [`FusionPlan`] records the block structure fusion chose
//! from the gate sequence, with a recipe per fused payload, so a later binding rebuilds the
//! matrices without re-running fusion unless the new angles break a recorded decision.

use std::ops::Range;

use num_complex::Complex64;

use super::fusion::is_identity;
use super::parameter::angle_of;
use super::{Circuit, Instruction};
use crate::gates::{
    DiagEntry, Gate, is_diagonal_2x2, is_diagonal_4x4, kron_2x2, mat_mul_2x2, mat_mul_4x4,
};

/// One factor of a fused payload, naming a template instruction and how its
/// matrix enters the product.
///
/// Embedding and SWAP conjugation are both multiplicative, so a nested recipe
/// splices into an enclosing one by rewriting each step's placement flag rather
/// than by materializing the inner product.
#[derive(Clone, Debug)]
enum Step {
    /// The 2x2 matrix of the template gate.
    M1 { src: u32 },
    /// The 2x2 matrix widened to a qubit pair; `high` selects `kron(I, M)`.
    E1 { src: u32, high: bool },
    /// The 4x4 matrix of the template gate, conjugated by SWAP when `swapped`.
    M2 { src: u32, swapped: bool },
    /// The Rzz matrix with a template Rz's angle.
    Rzz { src: u32 },
    /// One diagonal factor of a complete pair product.
    Project {
        steps: Box<[Step]>,
        part: DiagPart,
        high: Option<bool>,
    },
}

#[derive(Clone, Copy, Debug)]
enum DiagPart {
    First,
    Second,
    Phase,
}

impl Step {
    fn is_constant(&self, sources: &[bool]) -> bool {
        match self {
            Self::M1 { src } | Self::E1 { src, .. } | Self::M2 { src, .. } | Self::Rzz { src } => {
                sources[*src as usize]
            }
            Self::Project { steps, .. } => steps.iter().all(|step| step.is_constant(sources)),
        }
    }

    /// Widen a 1q step onto a pair, as the high factor or the low one. A step
    /// already at pair width is unaffected.
    fn embed(self, high: bool) -> Step {
        match self {
            Step::M1 { src } => Step::E1 { src, high },
            Step::Project {
                steps,
                part,
                high: None,
            } if !matches!(part, DiagPart::Phase) => Step::Project {
                steps,
                part,
                high: Some(high),
            },
            other => other,
        }
    }

    /// Re-orient this step for an enclosing pair in the opposite target order.
    fn orient(self, swapped: bool) -> Step {
        if !swapped {
            return self;
        }
        match self {
            Step::M1 { .. } => self,
            Step::E1 { src, high } => Step::E1 { src, high: !high },
            Step::M2 { src, swapped } => Step::M2 {
                src,
                swapped: !swapped,
            },
            Step::Project {
                steps,
                part,
                high: Some(high),
            } => Step::Project {
                steps,
                part,
                high: Some(!high),
            },
            other => other,
        }
    }
}

/// How one payload inside the fused skeleton is rebuilt from the template.
#[derive(Clone, Debug)]
enum Recipe {
    /// Product of 2x2 factors, applied left to right onto the accumulator.
    /// The flag is the diagonality the captured block had, which downstream
    /// batching read.
    Mat2(Range<u32>, bool),
    /// Product of 4x4 factors, with the captured block's diagonality.
    Mat4(Range<u32>, bool),
    /// The angle of a template gate, carried through unfused.
    Angle(u32),
    Phase(Range<u32>),
    Parity(Range<u32>),
}

/// Where a rebuilt payload is written back into the skeleton.
#[derive(Clone, Debug)]
struct Site {
    instruction: u32,
    /// Index of the payload inside a batch gate; 0 for single-payload gates.
    entry: u32,
    recipe: Recipe,
}

/// A decision a pass made from a matrix rather than from the gate sequence.
///
/// Recipes rebuild payloads, but a pass that dropped or renamed a block chose
/// the shape of the stream itself. Each guard re-tests one such choice against
/// the new angles, and a binding that flips any of them re-runs the pipeline.
#[derive(Clone, Debug)]
enum Guard {
    /// A 1q run emitted as `Fused`: it must stay non-identity, unrecognizable
    /// as a named gate, and as diagonal as it was, the three properties fusion
    /// read off it. Absorption into a 4x4 downstream leaves it with no payload
    /// of its own, so no site can carry this check.
    Fuses1q(Range<u32>, bool),
    /// A 1q block whose diagonality decided how the stage planner ordered it.
    Diag1q(Range<u32>, bool),
    /// A 2q block whose diagonality decided whether a same-pair run collapsed.
    Diag4q(Range<u32>, bool),
    Phase2q(Range<u32>),
}

/// The fused block structure of a template, with a recipe per angle-derived
/// payload.
#[derive(Clone, Debug)]
pub(super) struct FusionPlan {
    steps: Vec<Step>,
    sites: Vec<Site>,
    guard_steps: Vec<Step>,
    guards: Vec<Guard>,
}

/// Provenance of one instruction: an ordered recipe per payload it carries.
type Prov = Vec<Vec<Step>>;

/// Entry index naming the single payload of an instruction rather than one entry of a
/// batch.
pub(super) const WHOLE: u32 = u32::MAX;

/// Where a source's matrix sits inside the payload being built.
#[derive(Clone, Copy, Debug)]
pub(super) enum Place {
    /// Same width as the source.
    Plain,
    /// Widened to a pair as the low factor.
    Low,
    /// Widened to a pair as the high factor.
    High,
    /// Same width, target order reversed.
    Swapped,
}

impl Place {
    fn apply(self, step: Step) -> Step {
        match self {
            Place::Plain => step,
            Place::Low => step.embed(false),
            Place::High => step.embed(true),
            Place::Swapped => step.orient(true),
        }
    }
}

/// Records how each pass builds its output stream out of its input stream.
///
/// Inactive by default so the ordinary fusion path skips every update. A pass
/// that meets a construct with no recipe calls [`bail`](Tracer::bail), and
/// capture falls back to running fusion per binding.
pub(crate) struct Tracer {
    pub(super) on: bool,
    bailed: bool,
    constant_sources: Vec<bool>,
    /// Provenance of the stream entering the current pass.
    input: Vec<Prov>,
    /// Provenance of the stream the current pass is building.
    output: Vec<Prov>,
    /// Flat pool of steps the recorded guards index into.
    guard_steps: Vec<Step>,
    guards: Vec<Guard>,
    /// Lengths and bail state at the start of the current pass, so a pass that
    /// returns its input unchanged leaves nothing behind.
    marks: (usize, usize, bool),
}

impl Tracer {
    pub(crate) fn off() -> Self {
        Self {
            on: false,
            bailed: false,
            constant_sources: Vec::new(),
            input: Vec::new(),
            output: Vec::new(),
            guard_steps: Vec::new(),
            guards: Vec::new(),
            marks: (0, 0, false),
        }
    }

    /// Start tracing a template whose instructions are their own sources.
    fn tracking(circuit: &Circuit) -> Self {
        let input = circuit
            .instructions
            .iter()
            .enumerate()
            .map(|(i, inst)| identity_prov(i as u32, inst))
            .collect();
        Self {
            on: true,
            bailed: false,
            constant_sources: circuit
                .instructions
                .iter()
                .map(|inst| matches!(inst, Instruction::Gate { gate, .. } if gate.pauli_generator().is_none()))
                .collect(),
            input,
            output: Vec::new(),
            guard_steps: Vec::new(),
            guards: Vec::new(),
            marks: (0, 0, false),
        }
    }

    pub(super) fn bail(&mut self) {
        if self.on {
            self.bailed = true;
        }
    }

    pub(super) fn sources_are_constant(&self, sources: &[(usize, Place)]) -> bool {
        self.on
            && !sources.is_empty()
            && sources.iter().all(|&(i, _)| {
                let [payload] = self.input[i].as_slice() else {
                    return false;
                };
                !payload.is_empty()
                    && payload
                        .iter()
                        .all(|step| step.is_constant(&self.constant_sources))
            })
    }

    /// Begin a pass. Output provenance accumulates until [`commit`](Self::commit).
    pub(super) fn begin(&mut self) {
        if self.on {
            self.output = Vec::with_capacity(self.input.len());
            self.marks = (self.guards.len(), self.guard_steps.len(), self.bailed);
        }
    }

    /// Carry input instruction `i` through unchanged.
    pub(super) fn keep(&mut self, i: usize) {
        if self.on {
            let p = self.input[i].clone();
            self.output.push(p);
        }
    }

    /// Ordered steps of input instruction `i`, placed by `place`.
    fn placed(&mut self, i: usize, place: Place, into: &mut Vec<Step>) {
        let prov = &self.input[i];
        if prov.len() != 1 {
            self.bailed = true;
            return;
        }
        into.extend(prov[0].iter().cloned().map(|s| place.apply(s)));
    }

    pub(super) fn rzz(&mut self, i: usize) {
        if !self.on {
            return;
        }
        let [payload] = self.input[i].as_slice() else {
            self.bail();
            return;
        };
        let [Step::M1 { src }] = payload.as_slice() else {
            self.bail();
            return;
        };
        self.output.push(vec![vec![Step::Rzz { src: *src }]]);
    }

    pub(super) fn diagonal_batch(&mut self, srcs: &[usize], source: &[Instruction]) {
        if !self.on {
            return;
        }
        let mut entries = Vec::new();
        for &src in srcs {
            let Instruction::Gate { gate, .. } = &source[src] else {
                unreachable!("diagonal source is a gate");
            };
            let prov = self.input[src].clone();
            match gate {
                Gate::BatchRzz(_) | Gate::BatchPhase(_) | Gate::MultiFused(_) => {
                    entries.extend(prov);
                }
                Gate::Fused2q(_) | Gate::Multi2q(_) => {
                    for payload in prov {
                        self.guard_payload_diag(&payload);
                        for part in [DiagPart::First, DiagPart::Second, DiagPart::Phase] {
                            entries.push(vec![Step::Project {
                                steps: payload.clone().into_boxed_slice(),
                                part,
                                high: None,
                            }]);
                        }
                    }
                }
                Gate::Cu(_) if gate.controlled_phase().is_none() => {
                    let [payload] = prov.as_slice() else {
                        self.bail();
                        continue;
                    };
                    self.guard_payload_diag(payload);
                    for part in [DiagPart::First, DiagPart::Phase] {
                        entries.push(vec![Step::Project {
                            steps: payload.clone().into_boxed_slice(),
                            part,
                            high: None,
                        }]);
                    }
                }
                _ => entries.extend(prov),
            }
        }
        self.output.push(entries);
    }

    fn guard_payload_diag(&mut self, payload: &[Step]) {
        let start = self.guard_steps.len() as u32;
        self.guard_steps.extend_from_slice(payload);
        self.guards
            .push(Guard::Diag4q(start..self.guard_steps.len() as u32, true));
    }

    pub(super) fn guard_phase(&mut self, src: usize) {
        if self.on
            && let Some(range) = self.resolve(&[(src, Place::Plain)])
        {
            self.guards.push(Guard::Phase2q(range));
        }
    }

    pub(super) fn subpass(&mut self, sources: &[(u32, u32, Place)]) -> Self {
        if !self.on {
            return Self::off();
        }
        let mut child = Self::off();
        child.on = true;
        child.constant_sources.clone_from(&self.constant_sources);
        for &(i, entry, place) in sources {
            let prov = &self.input[i as usize];
            if entry == WHOLE {
                child.input.push(prov.clone());
            } else if let Some(payload) = prov.get(entry as usize) {
                child.input.push(vec![
                    payload.iter().cloned().map(|s| place.apply(s)).collect(),
                ]);
            } else {
                child.bailed = true;
                child.input.push(Vec::new());
            }
        }
        child
    }

    pub(super) fn finish_subpass(&mut self, child: Self) {
        if !self.on {
            return;
        }
        self.bailed |= child.bailed;
        let offset = self.guard_steps.len() as u32;
        self.guard_steps.extend(child.guard_steps);
        self.guards
            .extend(child.guards.into_iter().map(|guard| match guard {
                Guard::Fuses1q(r, d) => Guard::Fuses1q(r.start + offset..r.end + offset, d),
                Guard::Diag1q(r, d) => Guard::Diag1q(r.start + offset..r.end + offset, d),
                Guard::Diag4q(r, d) => Guard::Diag4q(r.start + offset..r.end + offset, d),
                Guard::Phase2q(r) => Guard::Phase2q(r.start + offset..r.end + offset),
            }));
        self.output.extend(child.input);
    }

    /// Emit an instruction whose single payload is the ordered product of the
    /// payloads of `srcs`.
    pub(super) fn merge(&mut self, srcs: &[(usize, Place)]) {
        if !self.on {
            return;
        }
        let mut steps = Vec::new();
        for &(i, place) in srcs {
            self.placed(i, place, &mut steps);
        }
        self.output.push(vec![steps]);
    }

    /// Emit a batch instruction carrying one payload per entry, each entry the
    /// ordered product of its own source list.
    pub(super) fn batch(&mut self, entries: &[Vec<(usize, Place)>]) {
        if !self.on {
            return;
        }
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut steps = Vec::new();
            for &(i, place) in entry {
                self.placed(i, place, &mut steps);
            }
            out.push(steps);
        }
        self.output.push(out);
    }

    /// Fold the product of `srcs` into the single payload of output entry `at`,
    /// applied after the factors it already holds.
    pub(super) fn extend(&mut self, at: usize, srcs: &[(usize, Place)]) {
        if !self.on {
            return;
        }
        let mut steps = Vec::new();
        for &(i, place) in srcs {
            self.placed(i, place, &mut steps);
        }
        match self.output[at].as_mut_slice() {
            [payload] => payload.extend(steps),
            _ => self.bailed = true,
        }
    }

    /// Emit an instruction carrying one payload per entry, each the ordered product of
    /// its `(instruction, entry, place)` sources. An entry of [`WHOLE`] names the
    /// single payload of its instruction, any other index one entry of a batch.
    pub(super) fn emit_entries(&mut self, entries: &[Vec<(u32, u32, Place)>]) {
        if !self.on {
            return;
        }
        let mut out = Vec::with_capacity(entries.len());
        for entry in entries {
            let mut steps = Vec::new();
            for &(i, e, place) in entry {
                let prov = &self.input[i as usize];
                let payload = if e == WHOLE {
                    (prov.len() == 1).then(|| &prov[0])
                } else {
                    prov.get(e as usize)
                };
                match payload {
                    Some(payload) => steps.extend(payload.iter().cloned().map(|s| place.apply(s))),
                    None => self.bailed = true,
                }
            }
            out.push(steps);
        }
        self.output.push(out);
    }

    /// Bail unless the payload names a pair gate or a rewritten `Rzz` angle.
    pub(super) fn require_gate_4x4(&mut self, i: usize, entry: u32) {
        if !self.on {
            return;
        }
        let prov = &self.input[i];
        let payload = if entry == WHOLE {
            prov.first().filter(|_| prov.len() == 1)
        } else {
            prov.get(entry as usize)
        };
        if !matches!(
            payload.map(Vec::as_slice),
            Some([Step::M2 { .. } | Step::Rzz { .. }])
        ) {
            self.bailed = true;
        }
    }

    /// Record the diagonality a pass read off the 1q payload of input instruction `i`.
    pub(super) fn guard_diag_2x2(&mut self, i: usize, diagonal: bool) {
        if !self.on {
            return;
        }
        if let Some(range) = self.resolve(&[(i, Place::Plain)]) {
            self.guards.push(Guard::Diag1q(range, diagonal));
        }
    }

    /// Record the diagonality a pass read off the 2q payload of output entry `at`.
    pub(super) fn guard_output_diag_4x4(&mut self, at: usize, diagonal: bool) {
        if !self.on {
            return;
        }
        let [payload] = self.output[at].as_slice() else {
            self.bailed = true;
            return;
        };
        let start = self.guard_steps.len() as u32;
        self.guard_steps.extend_from_slice(payload);
        self.guards.push(Guard::Diag4q(
            start..self.guard_steps.len() as u32,
            diagonal,
        ));
    }

    /// Resolve `srcs` to template steps without emitting an instruction.
    fn resolve(&mut self, srcs: &[(usize, Place)]) -> Option<Range<u32>> {
        let start = self.guard_steps.len() as u32;
        for &(i, place) in srcs {
            let prov = &self.input[i];
            if prov.len() != 1 {
                self.bailed = true;
                return None;
            }
            let placed: Vec<Step> = prov[0].iter().cloned().map(|s| place.apply(s)).collect();
            self.guard_steps.extend(placed);
        }
        Some(start..self.guard_steps.len() as u32)
    }

    /// Record that the 1q product of `srcs` is what fusion chose to emit as a
    /// `Fused` block rather than elide or rename.
    pub(super) fn guard_1q(&mut self, srcs: &[(usize, Place)], diagonal: bool) {
        if !self.on {
            return;
        }
        if let Some(range) = self.resolve(srcs) {
            self.guards.push(Guard::Fuses1q(range, diagonal));
        }
    }

    /// Record the diagonality a pass read off the 2q product of `srcs`.
    pub(super) fn guard_diag_4x4(&mut self, srcs: &[(usize, Place)], diagonal: bool) {
        if !self.on {
            return;
        }
        if let Some(range) = self.resolve(srcs) {
            self.guards.push(Guard::Diag4q(range, diagonal));
        }
    }

    /// Record a source index while tracking; the untraced path allocates nothing.
    #[inline]
    pub(super) fn note(&self, v: &mut Vec<(usize, Place)>, i: usize, place: Place) {
        if self.on {
            v.push((i, place));
        }
    }

    /// Index vectors stay empty when not tracking, so a caller pairing one with
    /// a payload vector must index it rather than zip, which would yield nothing
    /// and drop the payloads.
    #[inline]
    pub(super) fn note_idx(&self, v: &mut Vec<usize>, i: usize) {
        if self.on {
            v.push(i);
        }
    }

    /// A fresh single-source list, empty and unallocated when not tracking.
    #[inline]
    pub(super) fn seed(&self, i: usize, place: Place) -> Vec<(usize, Place)> {
        if self.on {
            vec![(i, place)]
        } else {
            Vec::new()
        }
    }

    #[inline]
    pub(super) fn seed_idx(&self, i: usize) -> Vec<usize> {
        if self.on { vec![i] } else { Vec::new() }
    }

    /// Finish a pass whose output the caller kept.
    pub(super) fn commit(&mut self) {
        if self.on {
            self.input = std::mem::take(&mut self.output);
        }
    }

    /// Finish a pass that returned its input untouched. Nothing it decided
    /// reached the output, so its guards and any bail are rolled back.
    pub(super) fn discard(&mut self) {
        if self.on {
            self.output.clear();
            self.guards.truncate(self.marks.0);
            self.guard_steps.truncate(self.marks.1);
            self.bailed = self.marks.2;
        }
    }
}

/// Provenance of a template instruction: one payload naming itself, or none
/// when the gate carries no angle-derived data.
fn identity_prov(index: u32, inst: &Instruction) -> Prov {
    match inst {
        // A weight-2 Pauli rotation is a pair gate and folds into a 4x4 like
        // any other. Wider ones no pass folds, so they name themselves for the
        // angle recipe alone.
        Instruction::Gate {
            gate: gate @ Gate::PauliRot(_),
            ..
        } if gate.num_qubits() > 2 => vec![vec![Step::M1 { src: index }]],
        Instruction::Gate { gate, .. } => match gate.num_qubits() {
            1 => vec![vec![Step::M1 { src: index }]],
            2 => vec![vec![Step::M2 {
                src: index,
                swapped: false,
            }]],
            _ => Vec::new(),
        },
        _ => Vec::new(),
    }
}

impl FusionPlan {
    /// Settle the fused structure of `template` for a `state_qubits`-wide buffer
    /// and record how to rebuild the angle-derived payloads.
    ///
    /// Returns the fused skeleton, and the plan when every payload in it has a
    /// recipe. A `None` plan means a binding has to run fusion itself.
    pub(super) fn capture(
        template: &Circuit,
        state_qubits: usize,
        profile: super::fusion::FusionProfile,
    ) -> (Circuit, Option<FusionPlan>) {
        let mut tracer = Tracer::tracking(template);
        let fused =
            super::fusion::fuse_at_width(template, state_qubits, profile, &mut tracer).into_owned();
        if tracer.bailed || tracer.input.len() != fused.instructions.len() {
            return (fused, None);
        }

        let mut steps = Vec::new();
        let mut sites = Vec::new();
        for (i, (inst, prov)) in fused.instructions.iter().zip(&tracer.input).enumerate() {
            if !record_sites(
                i as u32,
                inst,
                prov,
                &tracer.constant_sources,
                &mut steps,
                &mut sites,
            ) {
                return (fused, None);
            }
        }
        (
            fused,
            Some(FusionPlan {
                steps,
                sites,
                guard_steps: std::mem::take(&mut tracer.guard_steps),
                guards: std::mem::take(&mut tracer.guards),
            }),
        )
    }

    /// Rebuild every payload of `skeleton` from `bound`.
    ///
    /// Returns false when a bound angle invalidates a decision the plan rests
    /// on: a fused block collapsing to the identity, becoming a named gate, or
    /// changing whether it is diagonal. Fusion would have emitted a different
    /// stream in each case, so the caller re-runs it.
    #[must_use]
    pub(super) fn replay(&self, bound: &Circuit, skeleton: &mut Circuit) -> bool {
        for guard in &self.guards {
            let holds = match guard {
                Guard::Fuses1q(range, diagonal) => {
                    let mat = product_2x2(&self.guard_steps, bound, range.clone());
                    !is_identity(&mat)
                        && is_diagonal_2x2(&mat) == *diagonal
                        && (!could_be_named(&mat) || Gate::recognize_matrix(&mat).is_none())
                }
                Guard::Diag1q(range, diagonal) => {
                    let mat = product_2x2(&self.guard_steps, bound, range.clone());
                    is_diagonal_2x2(&mat) == *diagonal
                }
                Guard::Diag4q(range, diagonal) => {
                    let mat = product_4x4(&self.guard_steps, bound, range.clone());
                    is_diagonal_4x4(&mat) == *diagonal
                }
                Guard::Phase2q(range) => {
                    let mat = product_4x4(&self.guard_steps, bound, range.clone());
                    is_phase_pair(&mat)
                }
            };
            if !holds {
                return false;
            }
        }
        for site in &self.sites {
            let inst = &mut skeleton.instructions[site.instruction as usize];
            match &site.recipe {
                Recipe::Mat2(range, diagonal) => {
                    let mat = product_2x2(&self.steps, bound, range.clone());
                    if !write_2x2(inst, site.entry as usize, mat, *diagonal) {
                        return false;
                    }
                }
                Recipe::Mat4(range, diagonal) => {
                    let mat = product_4x4(&self.steps, bound, range.clone());
                    if !write_4x4(inst, site.entry as usize, mat, *diagonal) {
                        return false;
                    }
                }
                Recipe::Phase(range) | Recipe::Parity(range) => {
                    let mat = product_4x4(&self.steps, bound, range.clone());
                    if !write_diagonal_pair(
                        inst,
                        site.entry as usize,
                        mat,
                        matches!(site.recipe, Recipe::Parity(_)),
                    ) {
                        return false;
                    }
                }
                Recipe::Angle(src) => {
                    let theta = angle_of(&bound.instructions[*src as usize]);
                    if !write_angle(inst, site.entry as usize, theta) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

fn product_2x2(pool: &[Step], bound: &Circuit, range: Range<u32>) -> [[Complex64; 2]; 2] {
    let factor = |step: &Step| match step {
        Step::M1 { src } => gate_2x2(bound, *src),
        Step::Project {
            steps,
            part,
            high: None,
        } => projected_1q(steps, *part, bound),
        _ => unreachable!("2x2 recipe holds 1q steps"),
    };
    let steps = &pool[range.start as usize..range.end as usize];
    let Some((first, rest)) = steps.split_first() else {
        return Gate::Id.matrix_2x2();
    };
    let mut acc = factor(first);
    for step in rest {
        acc = mat_mul_2x2(&factor(step), &acc);
    }
    acc
}

fn product_4x4(pool: &[Step], bound: &Circuit, range: Range<u32>) -> [[Complex64; 4]; 4] {
    let id = Gate::Id.matrix_2x2();
    let factor = |step: &Step| match step {
        Step::M1 { src } => kron_2x2(&gate_2x2(bound, *src), &id),
        Step::E1 { src, high } => {
            let m = gate_2x2(bound, *src);
            if *high {
                kron_2x2(&id, &m)
            } else {
                kron_2x2(&m, &id)
            }
        }
        Step::M2 { src, swapped } => {
            let m = gate_4x4(bound, *src);
            if *swapped { swap_order(&m) } else { m }
        }
        Step::Rzz { src } => Gate::Rzz(angle_of(&bound.instructions[*src as usize])).matrix_4x4(),
        Step::Project {
            steps,
            part: DiagPart::Phase,
            ..
        } => {
            let m = product_4x4(steps, bound, 0..steps.len() as u32);
            let phase = m[3][3] * m[0][0] / (m[2][2] * m[1][1]);
            let mut out = kron_2x2(&id, &id);
            out[3][3] = phase;
            out
        }
        Step::Project { steps, part, high } => {
            let m = projected_1q(steps, *part, bound);
            if high == &Some(true) {
                kron_2x2(&id, &m)
            } else {
                kron_2x2(&m, &id)
            }
        }
    };
    let steps = &pool[range.start as usize..range.end as usize];
    let Some((first, rest)) = steps.split_first() else {
        return kron_2x2(&id, &id);
    };
    let mut acc = factor(first);
    for step in rest {
        acc = mat_mul_4x4(&factor(step), &acc);
    }
    acc
}

fn projected_1q(steps: &[Step], part: DiagPart, bound: &Circuit) -> [[Complex64; 2]; 2] {
    let m = product_4x4(steps, bound, 0..steps.len() as u32);
    let zero = Complex64::new(0.0, 0.0);
    let (d0, d1) = match part {
        DiagPart::First => (m[0][0], m[2][2]),
        DiagPart::Second => (Complex64::new(1.0, 0.0), m[1][1] / m[0][0]),
        DiagPart::Phase => unreachable!("phase projection is two-qubit"),
    };
    [[d0, zero], [zero, d1]]
}

fn swap_order(mat: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let swap = Gate::Swap.matrix_4x4();
    mat_mul_4x4(&swap, &mat_mul_4x4(mat, &swap))
}

fn gate_2x2(circuit: &Circuit, src: u32) -> [[Complex64; 2]; 2] {
    match &circuit.instructions[src as usize] {
        Instruction::Gate { gate, .. } => gate.matrix_2x2(),
        _ => unreachable!("recipe step names a gate instruction"),
    }
}

fn gate_4x4(circuit: &Circuit, src: u32) -> [[Complex64; 4]; 4] {
    match &circuit.instructions[src as usize] {
        Instruction::Gate { gate, .. } => gate.matrix_4x4(),
        _ => unreachable!("recipe step names a gate instruction"),
    }
}

/// Build the sites for one fused instruction, or report that its payload shape
/// has no recipe.
fn record_sites(
    index: u32,
    inst: &Instruction,
    prov: &Prov,
    constant_sources: &[bool],
    steps: &mut Vec<Step>,
    sites: &mut Vec<Site>,
) -> bool {
    let Instruction::Gate { gate, .. } = inst else {
        return prov.is_empty();
    };

    let mut push_mat = |entry: usize, recipe_steps: &[Step], four: bool, diagonal: bool| {
        if recipe_steps
            .iter()
            .all(|step| step.is_constant(constant_sources))
        {
            return;
        }
        let start = steps.len() as u32;
        steps.extend_from_slice(recipe_steps);
        let range = start..steps.len() as u32;
        sites.push(Site {
            instruction: index,
            entry: entry as u32,
            recipe: if four {
                Recipe::Mat4(range, diagonal)
            } else {
                Recipe::Mat2(range, diagonal)
            },
        });
    };

    match gate {
        Gate::Fused(m) => {
            if prov.len() != 1 {
                return false;
            }
            push_mat(0, &prov[0], false, is_diagonal_2x2(m));
            true
        }
        Gate::Fused2q(m) => {
            if prov.len() != 1 {
                return false;
            }
            push_mat(0, &prov[0], true, is_diagonal_4x4(m));
            true
        }
        Gate::MultiFused(d) => {
            if prov.len() != d.gates.len() {
                return false;
            }
            for (entry, p) in prov.iter().enumerate() {
                push_mat(entry, p, false, is_diagonal_2x2(&d.gates[entry].1));
            }
            true
        }
        Gate::Multi2q(d) => {
            if prov.len() != d.gates.len() {
                return false;
            }
            for (entry, p) in prov.iter().enumerate() {
                push_mat(entry, p, true, is_diagonal_4x4(&d.gates[entry].2));
            }
            true
        }
        Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_) | Gate::P(_) | Gate::Rzz(_) | Gate::PauliRot(_) => {
            let [single] = prov.as_slice() else {
                return false;
            };
            let [Step::M1 { src } | Step::M2 { src, .. } | Step::Rzz { src }] = single.as_slice()
            else {
                return false;
            };
            sites.push(Site {
                instruction: index,
                entry: 0,
                recipe: Recipe::Angle(*src),
            });
            true
        }
        Gate::BatchRzz(d) => {
            if prov.len() != d.edges.len() {
                return false;
            }
            for (entry, p) in prov.iter().enumerate() {
                let [Step::M2 { src, .. } | Step::Rzz { src }] = p.as_slice() else {
                    return false;
                };
                sites.push(Site {
                    instruction: index,
                    entry: entry as u32,
                    recipe: Recipe::Angle(*src),
                });
            }
            true
        }
        Gate::MultiPauliRot(d) => {
            if prov.len() != d.rotations.len() {
                return false;
            }
            for (entry, p) in prov.iter().enumerate() {
                let [Step::M1 { src } | Step::M2 { src, .. } | Step::Rzz { src }] = p.as_slice()
                else {
                    return false;
                };
                sites.push(Site {
                    instruction: index,
                    entry: entry as u32,
                    recipe: Recipe::Angle(*src),
                });
            }
            true
        }
        Gate::DiagonalBatch(data) => {
            if prov.len() != data.entries.len() {
                return false;
            }
            for (entry, (p, diagonal)) in prov.iter().zip(&data.entries).enumerate() {
                let start = steps.len() as u32;
                steps.extend_from_slice(p);
                let range = start..steps.len() as u32;
                let recipe = match diagonal {
                    DiagEntry::Phase1q { .. } => Recipe::Mat2(range, true),
                    DiagEntry::Phase2q { .. } => Recipe::Phase(range),
                    DiagEntry::Parity2q { .. } => Recipe::Parity(range),
                };
                sites.push(Site {
                    instruction: index,
                    entry: entry as u32,
                    recipe,
                });
            }
            true
        }
        Gate::BatchPhase(data) => {
            if prov.len() != data.phases.len() {
                return false;
            }
            for (entry, p) in prov.iter().enumerate() {
                let start = steps.len() as u32;
                steps.extend_from_slice(p);
                sites.push(Site {
                    instruction: index,
                    entry: entry as u32,
                    recipe: Recipe::Phase(start..steps.len() as u32),
                });
            }
            true
        }
        Gate::Cu(_) if gate.controlled_phase().is_some() => {
            let [payload] = prov.as_slice() else {
                return false;
            };
            let start = steps.len() as u32;
            steps.extend_from_slice(payload);
            sites.push(Site {
                instruction: index,
                entry: 0,
                recipe: Recipe::Phase(start..steps.len() as u32),
            });
            true
        }
        // No angle, so a binding leaves these untouched. Listed rather than
        // wildcarded so a new angle-carrying variant fails to compile here.
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
        | Gate::Cx
        | Gate::Cz
        | Gate::Swap
        | Gate::Cu(_)
        | Gate::Mcu(_)
        | Gate::QftBlock { .. }
        | Gate::Unitary(_) => true,
    }
}

/// True when `mat` still fuses the way capture recorded: not the identity, not
/// a named gate, and diagonal exactly as the captured block was.
fn matrix_2x2_still_fuses(mat: &[[Complex64; 2]; 2], diagonal: bool) -> bool {
    !is_identity(mat)
        && is_diagonal_2x2(mat) == diagonal
        && (!could_be_named(mat) || Gate::recognize_matrix(mat).is_none())
}

/// Necessary condition for [`Gate::recognize_matrix`] to match: every candidate
/// it scans has `|m[0][0]|^2` in {0, 1/2, 1}. A generic binding fails this and
/// skips the scan.
#[inline]
fn could_be_named(mat: &[[Complex64; 2]; 2]) -> bool {
    const EPS: f64 = 1e-8;
    let a = mat[0][0].norm_sqr();
    a < EPS || (a - 0.5).abs() < EPS || (a - 1.0).abs() < EPS
}

fn write_2x2(
    inst: &mut Instruction,
    entry: usize,
    mat: [[Complex64; 2]; 2],
    diagonal: bool,
) -> bool {
    let Instruction::Gate { gate, .. } = inst else {
        return false;
    };
    if let Gate::DiagonalBatch(data) = gate {
        if !is_diagonal_2x2(&mat) {
            return false;
        }
        let DiagEntry::Phase1q { d0, d1, .. } = &mut data.entries[entry] else {
            return false;
        };
        *d0 = mat[0][0];
        *d1 = mat[1][1];
        return true;
    }
    if !matrix_2x2_still_fuses(&mat, diagonal) {
        return false;
    }
    match gate {
        Gate::Fused(b) => {
            **b = mat;
            true
        }
        Gate::MultiFused(d) => match d.gates.get_mut(entry) {
            Some(slot) => {
                slot.1 = mat;
                true
            }
            None => false,
        },
        _ => false,
    }
}

fn is_phase_pair(mat: &[[Complex64; 4]; 4]) -> bool {
    let one = Complex64::new(1.0, 0.0);
    is_diagonal_4x4(mat)
        && (0..3).all(|i| (mat[i][i] - one).norm() < crate::gates::IDENTITY_EPS)
        && (mat[3][3].norm() - 1.0).abs() < crate::gates::IDENTITY_EPS
}

fn write_diagonal_pair(
    inst: &mut Instruction,
    entry: usize,
    mat: [[Complex64; 4]; 4],
    parity: bool,
) -> bool {
    if !is_diagonal_4x4(&mat) {
        return false;
    }
    let equal = |a: Complex64, b: Complex64| (a - b).norm() < crate::gates::IDENTITY_EPS;
    if parity {
        if !equal(mat[0][0], mat[3][3]) || !equal(mat[1][1], mat[2][2]) {
            return false;
        }
    } else if !is_phase_pair(&mat) {
        return false;
    }
    let Instruction::Gate { gate, .. } = inst else {
        return false;
    };
    match gate {
        Gate::DiagonalBatch(data) => match &mut data.entries[entry] {
            DiagEntry::Phase2q { phase, .. } if !parity => *phase = mat[3][3],
            DiagEntry::Parity2q { same, diff, .. } if parity => {
                *same = mat[0][0];
                *diff = mat[1][1];
            }
            _ => return false,
        },
        Gate::BatchPhase(data) if !parity => data.phases[entry].1 = mat[3][3],
        Gate::Cu(m) if !parity => m[1][1] = mat[3][3],
        _ => return false,
    }
    true
}

fn write_4x4(
    inst: &mut Instruction,
    entry: usize,
    mat: [[Complex64; 4]; 4],
    diagonal: bool,
) -> bool {
    let Instruction::Gate { gate, .. } = inst else {
        return false;
    };
    if is_diagonal_4x4(&mat) != diagonal {
        return false;
    }
    match gate {
        Gate::Fused2q(b) => {
            **b = mat;
            true
        }
        Gate::Multi2q(d) => match d.gates.get_mut(entry) {
            Some(slot) => {
                slot.2 = mat;
                true
            }
            None => false,
        },
        _ => false,
    }
}

pub(super) fn write_angle(inst: &mut Instruction, entry: usize, theta: f64) -> bool {
    let Instruction::Gate { gate, .. } = inst else {
        return false;
    };
    match gate {
        Gate::Rx(t) | Gate::Ry(t) | Gate::Rz(t) | Gate::P(t) | Gate::Rzz(t) => {
            *t = theta;
            true
        }
        Gate::PauliRot(d) => {
            d.set_theta(theta);
            true
        }
        Gate::BatchRzz(d) => match d.edges.get_mut(entry) {
            Some(edge) => {
                edge.2 = theta;
                true
            }
            None => false,
        },
        Gate::MultiPauliRot(d) => match d.rotations.get_mut(entry) {
            Some(rotation) => {
                rotation.2 = theta;
                true
            }
            None => false,
        },
        _ => false,
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;
