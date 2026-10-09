//! Per-shot execution of a [`DynamicProgram`].

use rand::SeedableRng;

use super::dispatch::{BackendPlan, ExecutionPlan};
use super::{
    BackendKind, CountsResult, RunMetadata, Seeded, ShotsResult, Unseeded, backend_metadata,
    expand_for_backend, fuse_for_backend, mix_seed, resolve, run_counts_with, run_shots_with,
    validate_explicit_backend,
};
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Instruction;
use crate::circuit::dynamic::{Action, BasicBlock, ClassicalValue, DynamicProgram, Terminator};
use crate::error::{PrismError, Result};

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
}

/// Start a run of `program` on [`BackendKind::Auto`].
///
/// A program with no runtime control flow runs as its one circuit does, through
/// [`simulate`](super::simulate), so it keeps every sampling shortcut and the
/// same seeded shots. Any other program runs once per shot: each block is
/// fused for the backend once, then every shot walks the graph on a fresh
/// state, seeded as the per-shot circuit routes seed shot `i`.
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
}

impl<'p> SimulateProgram<'p, Unseeded> {
    pub fn seed(self, seed: u64) -> SimulateProgram<'p, Seeded> {
        SimulateProgram {
            program: self.program,
            kind: self.kind,
            seed: Seeded { seed },
            max_steps: self.max_steps,
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
    /// raises.
    pub fn shots(self, num_shots: usize) -> Result<ShotsResult> {
        let seed = self.seed.seed;
        if let Some(circuit) = self.program.static_circuit() {
            return run_shots_with(self.kind, circuit, num_shots, seed);
        }
        let prepared = Prepared::new(self.program, &self.kind, seed, self.max_steps)?;
        prepared.run(&self.kind, num_shots, seed)
    }

    /// Run `num_shots` shots and count each distinct outcome.
    ///
    /// # Errors
    ///
    /// Same conditions as [`SimulateProgram::shots`].
    pub fn sample_counts(self, num_shots: usize) -> Result<CountsResult> {
        let num_classical_bits = self.program.num_classical_bits();
        if let Some(circuit) = self.program.static_circuit() {
            let (counts, metadata) =
                run_counts_with(self.kind, circuit, num_shots, self.seed.seed)?;
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
struct Prepared<'p> {
    program: &'p DynamicProgram,
    plan: BackendPlan,
    blocks: Vec<PreparedBlock<'p>>,
    initial: Vec<ClassicalValue>,
    max_steps: u64,
}

impl<'p> Prepared<'p> {
    fn new(
        program: &'p DynamicProgram,
        kind: &BackendKind,
        seed: u64,
        max_steps: u64,
    ) -> Result<Self> {
        let summary = program.summary_circuit();
        if !kind.is_auto() {
            validate_explicit_backend(kind, &summary)?;
        }
        let plan = match resolve(kind, &summary, false) {
            #[cfg(feature = "distributed")]
            ExecutionPlan::Backend(BackendPlan::Distributed(_)) => {
                return Err(decline(
                    kind,
                    "the distributed statevector runs its ranks in lockstep on one circuit",
                ));
            }
            ExecutionPlan::Backend(plan) => plan,
            _ => {
                return Err(decline(
                    kind,
                    "this engine keeps no per-shot state to walk a control-flow graph on",
                ));
            }
        };
        let probe = plan.build(seed);
        let blocks = program
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
            .collect();
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
        })
    }

    fn run(&self, kind: &BackendKind, num_shots: usize, seed: u64) -> Result<ShotsResult> {
        let reuse = self.plan.is_host_statevector();
        let shot = |state: &mut ShotState, i: usize| -> Result<(Vec<bool>, RunMetadata)> {
            let shot_seed = mix_seed(seed, i);
            if reuse {
                let backend = state
                    .statevector
                    .get_or_insert_with(|| StatevectorBackend::new(shot_seed));
                backend.rng = rand_chacha::ChaCha8Rng::seed_from_u64(shot_seed);
                let bits = self.run_shot(backend, &mut state.vars)?;
                return Ok((bits, backend_metadata(backend)));
            }
            let mut backend = self.plan.build(shot_seed);
            let bits = self.run_shot(&mut *backend, &mut state.vars)?;
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
    fn run_shot(
        &self,
        backend: &mut dyn Backend,
        vars: &mut Vec<ClassicalValue>,
    ) -> Result<Vec<bool>> {
        backend.init(self.program.num_qubits(), self.program.num_classical_bits())?;
        vars.clear();
        vars.extend_from_slice(&self.initial);
        let variables = self.program.variables();
        let mut at = 0usize;
        let mut steps = 0u64;
        loop {
            if steps == self.max_steps {
                return Err(self.step_limit(at));
            }
            steps += 1;
            let prepared = &self.blocks[at];
            if !prepared.instructions.is_empty() {
                backend.apply_instructions(&prepared.instructions)?;
            }
            for action in &prepared.block.actions {
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
                    }
                }
            }
            at = match &prepared.block.terminator {
                Terminator::Jump(target) => target.index(),
                Terminator::Branch {
                    condition,
                    then,
                    otherwise,
                } => {
                    if condition.eval(backend.classical_results(), vars)?.truthy() {
                        then.index()
                    } else {
                        otherwise.index()
                    }
                }
                Terminator::End => break,
            };
        }
        Ok(backend.classical_results().to_vec())
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
        let reused = simulate_program(&program)
            .backend(BackendKind::Statevector)
            .seed(3)
            .shots(64)
            .unwrap();
        let prepared = Prepared::new(&program, &BackendKind::Statevector, 3, 100).unwrap();
        let fresh: Vec<Vec<bool>> = (0..64)
            .map(|i| {
                let mut backend = StatevectorBackend::new(mix_seed(3, i));
                prepared.run_shot(&mut backend, &mut Vec::new()).unwrap()
            })
            .collect();
        assert_eq!(reused.shots, fresh);
    }
}
