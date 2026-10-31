//! Per-shot execution of a [`DynamicProgram`].

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::branching::{self, BranchLimits};
use super::dispatch::{BackendPlan, ExecutionPlan};
use super::noise::{NoiseBuilder, NoiseEvent, NoiseModel, ReadoutError};
use super::trajectory::{self, PreparedNoise};
use super::{
    BackendKind, CountsResult, RunMetadata, Seeded, ShotsResult, Unseeded, backend_metadata,
    decide_noisy_program, expand_for_backend, fuse_for_backend, mix_seed, resolve, run_counts_with,
    run_shots_with, run_shots_with_noise, validate_explicit_backend,
};
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::dynamic::{
    Action, BasicBlock, ClassicalValue, DynamicProgram, RotationKind, Terminator,
};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::limits::ResolvedLimits;

/// Blocks a shot may run before [`SimulateProgram::shots`] stops it, unless
/// [`SimulateProgram::max_steps`] sets another bound.
const DEFAULT_MAX_STEPS: u64 = 1_000_000;

/// Builder for running a [`DynamicProgram`], the counterpart of
/// [`Simulate`](super::Simulate) for programs with runtime control flow.
pub struct SimulateProgram<'p, SeedState> {
    program: &'p DynamicProgram,
    kind: BackendKind,
    seed: SeedState,
    max_steps: u64,
    noise: Option<&'p NoiseBuilder>,
}

/// Start a run of `program` on [`BackendKind::Auto`].
///
/// A program with no runtime control flow runs as its one circuit does, through
/// [`simulate`](super::simulate), so it keeps every sampling shortcut and the
/// same seeded shots. Any other program has each block fused for the backend
/// once. On the host statevector the shots then walk the graph together,
/// splitting binomially at each measurement as circuits with mid-circuit
/// measurements do, so the program evolves once per distinct outcome history.
/// On any other backend every shot walks the graph on a fresh state, seeded as
/// the per-shot circuit routes seed shot `i`.
///
/// Auto routing reads every block at once, so a program holding only Clifford
/// gates runs on the stabilizer tableau, and a runtime rotation counts as a
/// non-Clifford gate. Stabilizer rank, Pauli propagation, and the distributed
/// statevector have no per-shot state to walk a graph on and decline.
pub fn simulate_program(program: &DynamicProgram) -> SimulateProgram<'_, Unseeded> {
    SimulateProgram {
        program,
        kind: BackendKind::Auto,
        seed: Unseeded,
        max_steps: DEFAULT_MAX_STEPS,
        noise: None,
    }
}

impl<'p, SeedState> SimulateProgram<'p, SeedState> {
    pub fn backend(mut self, kind: BackendKind) -> Self {
        self.kind = kind;
        self
    }

    /// Blocks one shot may run before it stops with
    /// [`PrismError::StepLimit`], one million by default. Every block entered
    /// counts, the empty test at the head of a loop included.
    pub fn max_steps(mut self, max_steps: u64) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Sample one noisy trajectory per shot, `rules` compiled against each block's
    /// circuit and each runtime rotation. Readout error flips the returned record,
    /// not the bits a branch reads.
    pub fn noise(mut self, rules: &'p NoiseBuilder) -> Self {
        self.noise = Some(rules);
        self
    }
}

impl<'p> SimulateProgram<'p, Unseeded> {
    pub fn seed(self, seed: u64) -> SimulateProgram<'p, Seeded> {
        SimulateProgram {
            program: self.program,
            kind: self.kind,
            seed: Seeded { seed },
            max_steps: self.max_steps,
            noise: self.noise,
        }
    }
}

impl SimulateProgram<'_, Seeded> {
    /// Run `num_shots` shots and collect each one's classical bits.
    ///
    /// # Errors
    ///
    /// [`PrismError::StepLimit`] when a shot runs past the step bound,
    /// [`PrismError::IncompatibleBackend`] for a backend that cannot walk the
    /// graph or hold the program's gates, and anything a block's execution
    /// raises. Under [`SimulateProgram::noise`], also anything
    /// [`NoiseBuilder::build`] reports for a block, and
    /// [`PrismError::IncompatibleBackend`] for the density matrix, for a channel
    /// the backend cannot sample, for a leakage or quasi-static channel, and for
    /// an over-rotation rule matching a runtime rotation.
    pub fn shots(self, num_shots: usize) -> Result<ShotsResult> {
        let seed = self.seed.seed;
        let limits = &ResolvedLimits::process_default();
        if let Some(circuit) = self.program.static_circuit() {
            return match self.noise {
                Some(rules) => run_shots_with_noise(
                    self.kind,
                    circuit,
                    &rules.build(circuit)?,
                    num_shots,
                    seed,
                    limits,
                ),
                None => run_shots_with(self.kind, circuit, num_shots, seed, limits),
            };
        }
        let prepared = Prepared::new(
            self.program,
            &self.kind,
            seed,
            self.max_steps,
            limits,
            self.noise,
        )?;
        prepared.run(&self.kind, num_shots, seed)
    }

    /// Run `num_shots` shots and count each distinct outcome.
    ///
    /// # Errors
    ///
    /// Same conditions as [`SimulateProgram::shots`].
    pub fn sample_counts(self, num_shots: usize) -> Result<CountsResult> {
        let num_classical_bits = self.program.num_classical_bits();
        if self.noise.is_none()
            && let Some(circuit) = self.program.static_circuit()
        {
            let (counts, metadata) = run_counts_with(
                self.kind,
                circuit,
                num_shots,
                self.seed.seed,
                &ResolvedLimits::process_default(),
            )?;
            return Ok(CountsResult {
                counts,
                num_classical_bits,
                metadata,
            });
        }
        let shots = self.shots(num_shots)?;
        Ok(CountsResult {
            counts: shots.counts(),
            num_classical_bits,
            metadata: shots.metadata,
        })
    }
}

struct PreparedBlock<'p> {
    instructions: Vec<Instruction>,
    block: &'p BasicBlock,
}

/// A program fused for one backend plan, ready to run any number of shots.
pub(super) struct Prepared<'p> {
    program: &'p DynamicProgram,
    plan: BackendPlan,
    blocks: Vec<PreparedBlock<'p>>,
    initial: Vec<ClassicalValue>,
    max_steps: u64,
    pub(super) limits: ResolvedLimits,
    noise: Option<ProgramNoise>,
}

/// A program's noise rules compiled block by block for its trajectories.
struct ProgramNoise {
    /// Each block's events in the slot layout of its own circuit.
    blocks: Vec<PreparedNoise>,
    /// Each block's events with one slot per action, empty but for rotations.
    actions: Vec<PreparedNoise>,
    readout: Vec<Option<ReadoutError>>,
}

impl ProgramNoise {
    /// Compile `rules` for `program`, also returning every event in one model
    /// for the route to read.
    fn compile(
        program: &DynamicProgram,
        summary: &Circuit,
        rules: &NoiseBuilder,
    ) -> Result<(Self, NoiseModel)> {
        let mut blocks = Vec::with_capacity(program.blocks().len());
        let mut actions = Vec::with_capacity(program.blocks().len());
        let mut channels = Vec::new();
        let mut readout = Vec::new();
        for block in program.blocks() {
            let mut model = rules.build(&block.circuit)?;
            let mut rotations = Vec::with_capacity(block.actions.len());
            for action in &block.actions {
                rotations.push(match action {
                    Action::Rotation { kind, targets, .. } => {
                        rotation_events(program, rules, *kind, targets)?
                    }
                    Action::Assign { .. } => Vec::new(),
                });
            }
            let rotations = NoiseModel {
                after_gate: rotations,
                readout: Vec::new(),
            };
            blocks.push(PreparedNoise::new(&model));
            actions.push(PreparedNoise::new(&rotations));
            readout = std::mem::take(&mut model.readout);
            channels.extend(model.after_gate);
            channels.extend(rotations.after_gate);
        }
        let channels = NoiseModel {
            after_gate: channels,
            readout: Vec::new(),
        };
        let compiled = Self {
            blocks,
            actions,
            readout: trajectory::written_readout(summary, &readout),
        };
        Ok((compiled, channels))
    }
}

/// The events `rules` place after a runtime `kind` rotation on `targets`,
/// compiled as a one-gate circuit.
fn rotation_events(
    program: &DynamicProgram,
    rules: &NoiseBuilder,
    kind: RotationKind,
    targets: &[usize],
) -> Result<Vec<NoiseEvent>> {
    let events_at = |angle: f64| -> Result<Vec<NoiseEvent>> {
        let mut circuit = Circuit::new(program.num_qubits(), program.num_classical_bits());
        circuit.add_gate(kind.gate(angle), targets);
        Ok(rules.build(&circuit)?.after_gate.swap_remove(0))
    };
    let events = events_at(1.0)?;
    // A rule reading the angle emits different events at another one, and a
    // runtime angle exists only once the shot evaluates it.
    if events != events_at(2.0)? {
        return Err(PrismError::IncompatibleBackend {
            backend: "NoiseBuilder".into(),
            reason: format!(
                "an over-rotation rule scales with the angle of the gate it follows, and \
                 a runtime {kind:?} angle is known only once each shot evaluates it"
            ),
        });
    }
    Ok(events)
}

#[cfg(feature = "distributed")]
const DISTRIBUTED_LOCKSTEP: &str =
    "the distributed statevector runs its ranks in lockstep on one circuit";

impl<'p> Prepared<'p> {
    pub(super) fn new(
        program: &'p DynamicProgram,
        kind: &BackendKind,
        seed: u64,
        max_steps: u64,
        limits: &ResolvedLimits,
        rules: Option<&NoiseBuilder>,
    ) -> Result<Self> {
        let summary = program.summary_circuit();
        let (plan, noise) = match rules {
            None => (Self::plan(kind, &summary, limits)?, None),
            Some(rules) => {
                let (noise, plan) = Self::noisy_plan(program, kind, &summary, limits, rules)?;
                (plan, Some(noise))
            }
        };
        let blocks = if noise.is_some() {
            program
                .blocks()
                .iter()
                .map(|block| PreparedBlock {
                    instructions: block.circuit.instructions.clone(),
                    block,
                })
                .collect()
        } else {
            let probe = plan.build(seed, limits);
            program
                .blocks()
                .iter()
                .map(|block| {
                    let expanded = expand_for_backend(&*probe, &block.circuit);
                    let fused = fuse_for_backend(&*probe, &expanded);
                    PreparedBlock {
                        instructions: fused.into_owned().instructions,
                        block,
                    }
                })
                .collect()
        };
        Ok(Self {
            program,
            plan,
            blocks,
            initial: program
                .variables()
                .iter()
                .map(|variable| variable.initial)
                .collect(),
            max_steps,
            limits: *limits,
            noise,
        })
    }

    fn plan(kind: &BackendKind, summary: &Circuit, limits: &ResolvedLimits) -> Result<BackendPlan> {
        if !kind.is_auto() {
            validate_explicit_backend(kind, summary, limits)?;
        }
        match resolve(kind, summary, false, limits) {
            #[cfg(feature = "distributed")]
            ExecutionPlan::Backend(BackendPlan::Distributed(_)) => {
                Err(decline(kind, DISTRIBUTED_LOCKSTEP))
            }
            ExecutionPlan::Backend(plan) => Ok(plan),
            _ => Err(decline(
                kind,
                "this engine keeps no per-shot state to walk a control-flow graph on",
            )),
        }
    }

    /// Compile `rules` and pick the trajectory backend that runs them.
    fn noisy_plan(
        program: &DynamicProgram,
        kind: &BackendKind,
        summary: &Circuit,
        limits: &ResolvedLimits,
        rules: &NoiseBuilder,
    ) -> Result<(ProgramNoise, BackendPlan)> {
        let incompatible = |reason: &str| PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: reason.into(),
        };
        if kind.is_density_matrix() {
            return Err(incompatible(
                "noise on a dynamic program runs one trajectory per shot, and the density \
                 matrix holds the mixture instead of a trajectory",
            ));
        }
        let (noise, channels) = ProgramNoise::compile(program, summary, rules)?;
        if channels.has_leakage() || channels.quasi_static_sources() > 0 {
            return Err(incompatible(
                "leakage and quasi-static channels keep per-shot memory, which noisy \
                 trajectories do not carry from one block of a dynamic program to the next",
            ));
        }
        let plan = decide_noisy_program(kind, summary, &channels, limits)?;
        #[cfg(feature = "distributed")]
        if matches!(plan, BackendPlan::Distributed(_)) {
            return Err(decline(kind, DISTRIBUTED_LOCKSTEP));
        }
        Ok((noise, plan))
    }

    fn run(&self, kind: &BackendKind, num_shots: usize, seed: u64) -> Result<ShotsResult> {
        let num_qubits = self.program.num_qubits();
        if self.noise.is_none()
            && self.plan.is_host_statevector()
            && num_qubits < self.limits.statevector_qubits.value
        {
            #[cfg(feature = "parallel")]
            let parallel =
                super::shots_split_across_workers(kind, &[(self.plan.resolved(), num_qubits)])
                    && num_qubits >= branching::MIN_QUBITS_FOR_PARALLEL_SPLITS;
            #[cfg(not(feature = "parallel"))]
            let parallel = false;
            let branch_limits = BranchLimits::for_width(num_qubits, parallel, &self.limits);
            return branching::branch_program_shots(self, num_shots, seed, &branch_limits);
        }
        self.run_per_shot(kind, num_shots, seed)
    }

    fn run_per_shot(&self, kind: &BackendKind, num_shots: usize, seed: u64) -> Result<ShotsResult> {
        let reuse = self.plan.is_host_statevector();
        let shot = |state: &mut ShotState, i: usize| -> Result<(Vec<bool>, RunMetadata)> {
            let shot_seed = mix_seed(seed, i);
            if reuse {
                let backend = state.statevector.get_or_insert_with(|| {
                    StatevectorBackend::with_limits(shot_seed, &self.limits)
                });
                backend.rng = ChaCha8Rng::seed_from_u64(shot_seed);
                let bits = self.run_shot(backend, &mut state.vars, shot_seed)?;
                return Ok((bits, backend_metadata(backend)));
            }
            let mut backend = self.plan.build(shot_seed, &self.limits);
            let bits = self.run_shot(&mut *backend, &mut state.vars, shot_seed)?;
            Ok((bits, backend_metadata(&*backend)))
        };

        #[cfg(feature = "parallel")]
        if num_shots > 1
            && super::shots_split_across_workers(
                kind,
                &[(self.plan.resolved(), self.program.num_qubits())],
            )
        {
            use rayon::prelude::*;
            let runs: Vec<Result<(Vec<bool>, RunMetadata)>> = (0..num_shots)
                .into_par_iter()
                .map_init(ShotState::default, |state, i| shot(state, i))
                .collect();
            return self.fold(runs);
        }
        #[cfg(not(feature = "parallel"))]
        let _ = kind;
        let mut state = ShotState::default();
        self.fold((0..num_shots).map(|i| shot(&mut state, i)))
    }

    fn fold(
        &self,
        runs: impl IntoIterator<Item = Result<(Vec<bool>, RunMetadata)>>,
    ) -> Result<ShotsResult> {
        let runs = runs.into_iter();
        let mut shots = Vec::with_capacity(runs.size_hint().0);
        let mut metadata = RunMetadata::exact(self.plan.resolved());
        for (i, run) in runs.enumerate() {
            let (bits, shot_metadata) = run?;
            if i == 0 {
                metadata = shot_metadata;
            } else {
                metadata.weaken_with(&shot_metadata);
            }
            shots.push(bits);
        }
        Ok(
            ShotsResult::from_shots(shots, self.program.num_classical_bits())
                .with_metadata(metadata),
        )
    }

    /// Walk the graph once from a fresh state, returning the classical bits.
    /// A noisy program draws its noise from the stream of `shot_seed` a circuit
    /// trajectory uses.
    fn run_shot(
        &self,
        backend: &mut dyn Backend,
        vars: &mut Vec<ClassicalValue>,
        shot_seed: u64,
    ) -> Result<Vec<bool>> {
        backend.init(self.program.num_qubits(), self.program.num_classical_bits())?;
        vars.clear();
        vars.extend_from_slice(&self.initial);
        let mut steps = 0;
        self.count_step(0, &mut steps)?;
        if let Some(noise) = &self.noise {
            let mut rng = trajectory::noise_rng(shot_seed);
            return self.run_noisy_shot(noise, steps, backend, vars, &mut rng);
        }
        self.apply_block(0, backend)?;
        self.run_from(0, steps, backend, vars)?;
        Ok(backend.classical_results().to_vec())
    }

    /// [`Prepared::run_shot`] past the entry's step count, firing each block's
    /// events after their instructions and readout error on the finished record.
    fn run_noisy_shot(
        &self,
        noise: &ProgramNoise,
        mut steps: u64,
        backend: &mut dyn Backend,
        vars: &mut [ClassicalValue],
        rng: &mut ChaCha8Rng,
    ) -> Result<Vec<bool>> {
        let mut at = 0;
        loop {
            let circuit = &self.blocks[at].block.circuit;
            trajectory::evolve_noisy(backend, circuit, &noise.blocks[at], 0, rng)?;
            let rotations = Some((&noise.actions[at], &mut *rng));
            let Some(next) = self.finish_block_with(at, backend, vars, rotations)? else {
                break;
            };
            self.count_step(next, &mut steps)?;
            at = next;
        }
        let mut bits = backend.classical_results().to_vec();
        trajectory::apply_readout_errors(&mut bits, &noise.readout, rng);
        Ok(bits)
    }

    /// Carry a shot whose state has run block `at`'s circuit to the end of the
    /// graph, `steps` blocks entered so far.
    pub(super) fn run_from(
        &self,
        mut at: usize,
        mut steps: u64,
        backend: &mut dyn Backend,
        vars: &mut [ClassicalValue],
    ) -> Result<()> {
        while let Some(next) = self.finish_block(at, backend, vars)? {
            self.count_step(next, &mut steps)?;
            self.apply_block(next, backend)?;
            at = next;
        }
        Ok(())
    }

    fn apply_block(&self, at: usize, backend: &mut dyn Backend) -> Result<()> {
        let instructions = &self.blocks[at].instructions;
        if !instructions.is_empty() {
            backend.apply_instructions(instructions)?;
        }
        Ok(())
    }

    /// Count entry into block `at` against the step bound.
    pub(super) fn count_step(&self, at: usize, steps: &mut u64) -> Result<()> {
        if *steps == self.max_steps {
            return Err(self.step_limit(at));
        }
        *steps += 1;
        Ok(())
    }

    /// Run block `at`'s actions and terminator, returning the next block or
    /// `None` at the end of the graph.
    pub(super) fn finish_block(
        &self,
        at: usize,
        backend: &mut dyn Backend,
        vars: &mut [ClassicalValue],
    ) -> Result<Option<usize>> {
        self.finish_block_with(at, backend, vars, None)
    }

    /// [`Prepared::finish_block`], firing slot `i` of `rotation_noise` after
    /// action `i` when it is a rotation.
    fn finish_block_with(
        &self,
        at: usize,
        backend: &mut dyn Backend,
        vars: &mut [ClassicalValue],
        mut rotation_noise: Option<(&PreparedNoise, &mut ChaCha8Rng)>,
    ) -> Result<Option<usize>> {
        let block = self.blocks[at].block;
        let variables = self.program.variables();
        for (index, action) in block.actions.iter().enumerate() {
            match action {
                Action::Assign { var, value } => {
                    let evaluated = value.eval(backend.classical_results(), vars)?;
                    vars[var.index()] = variables[var.index()].ty.store(evaluated);
                }
                Action::Rotation {
                    kind,
                    targets,
                    angle,
                } => {
                    let angle = angle.eval(backend.classical_results(), vars)?.to_float();
                    if !angle.is_finite() {
                        return Err(PrismError::InvalidParameter {
                            message: format!(
                                "a runtime {kind:?} angle evaluated to {angle}, which no \
                                 gate can carry"
                            ),
                        });
                    }
                    backend.apply(&Instruction::Gate {
                        gate: kind.gate(angle),
                        targets: targets.clone(),
                    })?;
                    if let Some((noise, rng)) = rotation_noise.as_mut() {
                        trajectory::fire_slot(backend, noise, index, rng)?;
                    }
                }
            }
        }
        Ok(match &block.terminator {
            Terminator::Jump(target) => Some(target.index()),
            Terminator::Branch {
                condition,
                then,
                otherwise,
            } => {
                if condition.eval(backend.classical_results(), vars)?.truthy() {
                    Some(then.index())
                } else {
                    Some(otherwise.index())
                }
            }
            Terminator::End => None,
        })
    }

    /// Block `at`'s fused circuit, and whether the shot ends with it: no
    /// actions follow and its terminator ends the graph.
    pub(super) fn block_body(&self, at: usize) -> (&[Instruction], bool) {
        let prepared = &self.blocks[at];
        let ends =
            prepared.block.actions.is_empty() && prepared.block.terminator == Terminator::End;
        (&prepared.instructions, ends)
    }

    pub(super) fn initial_vars(&self) -> &[ClassicalValue] {
        &self.initial
    }

    pub(super) fn num_qubits(&self) -> usize {
        self.program.num_qubits()
    }

    pub(super) fn num_classical_bits(&self) -> usize {
        self.program.num_classical_bits()
    }

    fn step_limit(&self, at: usize) -> PrismError {
        PrismError::StepLimit {
            region: match &self.blocks[at].block.loop_name {
                Some(name) => format!("loop `{name}`"),
                None => format!("block {at}"),
            },
            max_steps: self.max_steps,
        }
    }
}

/// What one worker keeps across the shots it runs.
#[derive(Default)]
struct ShotState {
    statevector: Option<StatevectorBackend>,
    vars: Vec<ClassicalValue>,
}

fn decline(kind: &BackendKind, reason: &str) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason: format!("a dynamic program runs once per shot, and {reason}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::dynamic::{BinaryOp, ClassicalExpr, ClassicalType, DynamicProgramBuilder};
    use crate::gates::Gate;
    use crate::sim::ResolvedBackend;

    fn rus() -> DynamicProgram {
        let mut b = DynamicProgramBuilder::new(2, 1);
        b.add_gate(Gate::H, &[0]).add_gate(Gate::Cx, &[0, 1]);
        b.add_measure(0, 0);
        b.begin_while("retry", ClassicalExpr::Bit(0));
        b.add_reset(0).add_gate(Gate::H, &[0]).add_measure(0, 0);
        b.end().unwrap();
        b.build().unwrap()
    }

    #[test]
    fn repeat_until_success_always_ends_on_zero() {
        let shots = simulate_program(&rus()).seed(7).shots(200).unwrap();
        assert!(shots.shots.iter().all(|shot| !shot[0]));
        assert_eq!(shots.metadata.backend, ResolvedBackend::Stabilizer);
    }

    #[test]
    fn a_loop_past_the_step_bound_names_itself() {
        let mut b = DynamicProgramBuilder::new(1, 0);
        b.begin_while("forever", ClassicalExpr::from(true));
        b.add_gate(Gate::X, &[0]);
        b.end().unwrap();
        let program = b.build().unwrap();
        let err = simulate_program(&program)
            .max_steps(50)
            .seed(1)
            .shots(2)
            .unwrap_err();
        match err {
            PrismError::StepLimit { region, max_steps } => {
                assert_eq!(region, "loop `forever`");
                assert_eq!(max_steps, 50);
            }
            other => panic!("expected a step limit, got {other:?}"),
        }
    }

    #[test]
    fn a_reused_statevector_matches_a_fresh_one_per_shot() {
        let mut b = DynamicProgramBuilder::new(2, 2);
        let n = b.declare("n", ClassicalType::Uint { width: 4 }, 0i64.into());
        b.add_gate(Gate::Ry(0.7), &[0]).add_gate(Gate::Cx, &[0, 1]);
        b.add_measure(0, 0);
        b.begin_while(
            "count",
            ClassicalExpr::binary(BinaryOp::Lt, n.into(), 3i64.into()),
        );
        b.add_gate(Gate::Ry(0.4), &[1]).add_measure(1, 1);
        b.assign(
            n,
            ClassicalExpr::binary(BinaryOp::Add, n.into(), 1i64.into()),
        );
        b.end().unwrap();
        let program = b.build().unwrap();
        let prepared = Prepared::new(
            &program,
            &BackendKind::Statevector,
            3,
            100,
            &ResolvedLimits::process_default(),
            None,
        )
        .unwrap();
        let reused = prepared
            .run_per_shot(&BackendKind::Statevector, 64, 3)
            .unwrap();
        let fresh: Vec<Vec<bool>> = (0..64)
            .map(|i| {
                let shot_seed = mix_seed(3, i);
                let mut backend = StatevectorBackend::new(shot_seed);
                prepared
                    .run_shot(&mut backend, &mut Vec::new(), shot_seed)
                    .unwrap()
            })
            .collect();
        assert_eq!(reused.shots, fresh);
    }
}
