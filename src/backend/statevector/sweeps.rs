//! Static count of the full-state passes the statevector backend makes over a fused
//! stream, and a lower bound on the passes any tile schedule of the stream needs.

use num_complex::Complex64;
use smallvec::SmallVec;

#[cfg(feature = "parallel")]
use super::PARALLEL_THRESHOLD_QUBITS;
use super::StatevectorBackend;
use super::kernels::{self, Multi1qRoute, TargetedGate1q};
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction, qft_textbook_steps};
use crate::gates::{
    Gate, IDENTITY_EPS, is_diagonal_2x2, multi_2q_high_budget_for, multi_2q_low_bits,
    multi_2q_tile_bits_for,
};
use crate::sim::unified_pauli::PauliAxis;

/// The instruction families a [`SweepProfile`] counts apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SweepKind {
    Gate1q,
    Diagonal1q,
    /// `Cx`, `Cz`, `Swap` and `Rzz`.
    Gate2q,
    Fused2q,
    /// `Cu` and `Mcu`, phase forms included.
    Controlled,
    Unitary,
    PauliRot,
    MultiFused,
    MultiFusedDiagonal,
    Multi2q,
    MultiPauliRot,
    BatchRzz,
    BatchPhase,
    DiagonalBatch,
    QftBlock,
    Measure,
    Reset,
}

impl SweepKind {
    pub const ALL: [SweepKind; 17] = [
        SweepKind::Gate1q,
        SweepKind::Diagonal1q,
        SweepKind::Gate2q,
        SweepKind::Fused2q,
        SweepKind::Controlled,
        SweepKind::Unitary,
        SweepKind::PauliRot,
        SweepKind::MultiFused,
        SweepKind::MultiFusedDiagonal,
        SweepKind::Multi2q,
        SweepKind::MultiPauliRot,
        SweepKind::BatchRzz,
        SweepKind::BatchPhase,
        SweepKind::DiagonalBatch,
        SweepKind::QftBlock,
        SweepKind::Measure,
        SweepKind::Reset,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SweepKind::Gate1q => "gate_1q",
            SweepKind::Diagonal1q => "diagonal_1q",
            SweepKind::Gate2q => "gate_2q",
            SweepKind::Fused2q => "fused_2q",
            SweepKind::Controlled => "controlled",
            SweepKind::Unitary => "unitary",
            SweepKind::PauliRot => "pauli_rot",
            SweepKind::MultiFused => "multi_fused",
            SweepKind::MultiFusedDiagonal => "multi_fused_diagonal",
            SweepKind::Multi2q => "multi_2q",
            SweepKind::MultiPauliRot => "multi_pauli_rot",
            SweepKind::BatchRzz => "batch_rzz",
            SweepKind::BatchPhase => "batch_phase",
            SweepKind::DiagonalBatch => "diagonal_batch",
            SweepKind::QftBlock => "qft_block",
            SweepKind::Measure => "measure",
            SweepKind::Reset => "reset",
        }
    }
}

/// Totals for one [`SweepKind`]. `tile_apps` counts the gates applied inside a cache
/// tile (an in-place tier, a gathered subcube, a shared traversal, a QFT stage in a
/// tile or gathered row); a gate that takes a full-state pass of its own counts only
/// as that pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SweepCount {
    pub instructions: usize,
    pub passes: usize,
    pub tile_apps: usize,
}

/// Full-state passes and in-tile gate applications per [`SweepKind`], as the
/// statevector kernels route them. Guarded regions and conditionals count as taken; the
/// initial zero fill is not a pass here.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SweepProfile {
    counts: [SweepCount; SweepKind::ALL.len()],
}

impl SweepProfile {
    pub fn count(&self, kind: SweepKind) -> SweepCount {
        self.counts[kind as usize]
    }

    pub fn passes(&self) -> usize {
        self.counts.iter().map(|c| c.passes).sum()
    }

    pub fn tile_apps(&self) -> usize {
        self.counts.iter().map(|c| c.tile_apps).sum()
    }

    fn record(&mut self, kind: SweepKind, passes: usize, tile_apps: usize) {
        let count = &mut self.counts[kind as usize];
        count.instructions += 1;
        count.passes += passes;
        count.tile_apps += tile_apps;
    }
}

/// Full-state passes a statevector run of `circuit` makes, on the stream it expands and
/// fuses to. `threads` stands in for the Rayon pool width the `MultiFused` L3 tier reads.
pub fn sweep_profile_for(circuit: &Circuit, threads: usize) -> SweepProfile {
    let backend = StatevectorBackend::new(0);
    let expanded = crate::sim::expand_for_backend(&backend, circuit);
    let fused = crate::sim::fuse_for_backend(&backend, &expanded);
    sweep_profile(&fused.instructions, circuit.num_qubits, threads)
}

/// Count passes with private CPU scheduling forced at any width; `density_matrix` doubles the stream.
///
/// # Panics
/// Panics when a density-matrix stream contains operations outside the private schedule.
#[cfg(any(test, feature = "bench-internal"))]
pub fn scheduled_sweep_count(circuit: &Circuit, threads: usize, density_matrix: bool) -> usize {
    use super::schedule::{self, Action, Schedule};
    let width = circuit.num_qubits * if density_matrix { 2 } else { 1 };
    let backend: Box<dyn Backend> = if density_matrix {
        Box::new(crate::backend::density_matrix::DensityMatrixBackend::new(0))
    } else {
        Box::new(StatevectorBackend::new(0))
    };
    let expanded = crate::sim::expand_for_backend(&*backend, circuit);
    let fused = crate::circuit::fusion::fuse_circuit_for_width(&expanded, true, width);
    let actions = schedule::actions(
        &fused.instructions,
        density_matrix.then_some(circuit.num_qubits),
    );
    assert!(
        !density_matrix
            || actions
                .iter()
                .all(|action| !matches!(action, Action::Opaque(_))),
        "density-matrix census requires a supported gate stream"
    );
    let mut plan = Schedule::default();
    plan.update(&actions, width);
    let mut profiler = Profiler::new(width, threads);
    plan.stages
        .iter()
        .map(|stage| {
            if let Action::Opaque(i) = actions[stage[0]] {
                return profiler.passes_of(&fused.instructions[i]);
            }
            if stage.iter().all(|&i| matches!(actions[i], Action::Two(..))) {
                let gates: Vec<_> = stage
                    .iter()
                    .map(|&i| match actions[i] {
                        Action::Two(a, b, m) => (a, b, m),
                        _ => unreachable!(),
                    })
                    .collect();
                return profiler.multi_2q(&gates).0;
            }
            if stage.iter().all(|&i| matches!(actions[i], Action::One(..))) {
                let gates = crate::backend::statevector::schedule::one_gates(&actions, stage);
                return profiler.multi_1q(&gates).0;
            }
            if stage
                .iter()
                .all(|&i| matches!(actions[i], Action::Pauli(..)))
            {
                let gates: Vec<_> = stage
                    .iter()
                    .map(|&i| match actions[i] {
                        Action::Pauli(x, z, t) => (x, z, t),
                        _ => unreachable!(),
                    })
                    .collect();
                return profiler.multi_pauli_rot(&gates).0;
            }
            1
        })
        .sum()
}

/// [`sweep_profile_for`] on the stream the batching passes build at every width, as
/// fusion ran before the stage planner.
#[cfg(feature = "bench-internal")]
pub fn sweep_profile_unstaged(circuit: &Circuit, threads: usize) -> SweepProfile {
    let backend = StatevectorBackend::new(0);
    let expanded = crate::sim::expand_for_backend(&backend, circuit);
    let n = backend.fusion_state_qubits(circuit.num_qubits);
    let fused = crate::circuit::fusion::fuse_unstaged(&expanded, n);
    sweep_profile(&fused.instructions, circuit.num_qubits, threads)
}

/// Fastest of `reps` runs of the stage planner over `circuit` as fused before
/// batching, whatever its width.
#[cfg(feature = "bench-internal")]
pub fn stage_planning_time(circuit: &Circuit, reps: usize) -> std::time::Duration {
    let backend = StatevectorBackend::new(0);
    let expanded = crate::sim::expand_for_backend(&backend, circuit);
    let n = backend.fusion_state_qubits(circuit.num_qubits);
    let stream = crate::circuit::fusion::fuse_before_batching(&expanded, n).into_owned();
    (0..reps.max(1))
        .map(|_| {
            let start = std::time::Instant::now();
            std::hint::black_box(crate::circuit::fusion::plan_stages_for(&stream));
            start.elapsed()
        })
        .min()
        .expect("at least one rep")
}

/// Fewest passes any `Multi2q`-geometry tile schedule of `circuit` (as fused before batching)
/// needs: per run of tileable gates, the larger of the high-qubit cover and the worst
/// dependency chain's greedy segmentation, plus opaque instructions at their passes.
pub fn sweep_lower_bound(circuit: &Circuit, threads: usize) -> usize {
    let backend = StatevectorBackend::new(0);
    let expanded = crate::sim::expand_for_backend(&backend, circuit);
    let n = backend.fusion_state_qubits(circuit.num_qubits);
    let stream = crate::circuit::fusion::fuse_before_batching(&expanded, n);
    lower_bound(&stream.instructions, circuit.num_qubits, threads)
}

/// The passes [`StatevectorBackend`] makes applying `insts` to an `n`-qubit state with
/// a `threads`-wide pool, route by route as `dispatch_gate` picks them.
pub(crate) fn sweep_profile(insts: &[Instruction], n: usize, threads: usize) -> SweepProfile {
    let mut profile = SweepProfile::default();
    let mut profiler = Profiler::new(n, threads);
    for inst in insts {
        profiler.instruction(inst, &mut profile);
    }
    profile
}

struct Profiler {
    n: usize,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    threads: usize,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    medium_tiled: Option<bool>,
}

impl Profiler {
    fn new(n: usize, threads: usize) -> Self {
        Self {
            n,
            threads,
            medium_tiled: None,
        }
    }

    fn instruction(&mut self, inst: &Instruction, profile: &mut SweepProfile) {
        match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => {
                let (kind, passes, tile_apps) = self.gate(gate);
                profile.record(kind, passes, tile_apps);
            }
            Instruction::Measure { .. } => profile.record(SweepKind::Measure, 2, 0),
            Instruction::Reset { .. } => profile.record(SweepKind::Reset, 2, 0),
            Instruction::Barrier { .. } | Instruction::Save { .. } => {}
            Instruction::Region(region) => {
                for inst in region.body() {
                    self.instruction(inst, profile);
                }
            }
        }
    }

    fn passes_of(&mut self, inst: &Instruction) -> usize {
        let mut profile = SweepProfile::default();
        self.instruction(inst, &mut profile);
        profile.passes()
    }

    fn gate(&mut self, gate: &Gate) -> (SweepKind, usize, usize) {
        match gate {
            Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap => (SweepKind::Gate2q, 1, 0),
            Gate::Cu(_) | Gate::Mcu(_) => (SweepKind::Controlled, 1, 0),
            Gate::Fused2q(_) => (SweepKind::Fused2q, 1, 0),
            Gate::Unitary(_) => (SweepKind::Unitary, 1, 0),
            Gate::PauliRot(_) => (SweepKind::PauliRot, 1, 0),
            Gate::BatchPhase(_) => (SweepKind::BatchPhase, 1, 0),
            Gate::BatchRzz(_) => (SweepKind::BatchRzz, 1, 0),
            Gate::DiagonalBatch(_) => (SweepKind::DiagonalBatch, 1, 0),
            &Gate::QftBlock {
                start,
                num,
                inverse,
                swaps,
                big_endian,
            } => {
                let (passes, apps) =
                    self.qft_block(start as usize, num as usize, inverse, swaps, big_endian);
                (SweepKind::QftBlock, passes, apps)
            }
            Gate::MultiFused(data) if data.all_diagonal => {
                let (passes, apps) = self.multi_1q_diagonal(&data.gates);
                (SweepKind::MultiFusedDiagonal, passes, apps)
            }
            Gate::MultiFused(data) => {
                let (passes, apps) = self.multi_1q(&data.gates);
                (SweepKind::MultiFused, passes, apps)
            }
            Gate::Multi2q(data) => {
                let (passes, apps) = self.multi_2q(&data.gates);
                (SweepKind::Multi2q, passes, apps)
            }
            Gate::MultiPauliRot(data) => {
                let (passes, apps) = self.multi_pauli_rot(&data.rotations);
                (SweepKind::MultiPauliRot, passes, apps)
            }
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
                if gate.is_diagonal_1q() {
                    (SweepKind::Diagonal1q, 1, 0)
                } else {
                    (SweepKind::Gate1q, 1, 0)
                }
            }
        }
    }

    /// The L2 tier's highest target, as the sequential or the parallel kernel cuts it.
    fn max_l2_target(&self) -> usize {
        #[cfg(feature = "parallel")]
        if self.n >= PARALLEL_THRESHOLD_QUBITS {
            return kernels::max_target_for_tile(kernels::multi_gate_l2_tile_par(1 << self.n));
        }
        kernels::multi_gate_max_l2_target()
    }

    /// Whether the parallel `MultiFused` pass keeps its L3 tier, read in a pool of the
    /// profiled width since [`kernels::keep_l3_tier`] asks Rayon for its worker count.
    #[cfg(feature = "parallel")]
    fn medium_tiled(&mut self) -> bool {
        if let Some(kept) = self.medium_tiled {
            return kept;
        }
        let state_len = 1usize << self.n;
        let kept = if rayon::current_num_threads() == self.threads {
            kernels::keep_l3_tier(state_len)
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .build()
                .expect("a Rayon pool of the profiled width")
                .install(|| kernels::keep_l3_tier(state_len))
        };
        self.medium_tiled = Some(kept);
        kept
    }

    fn multi_1q(&mut self, gates: &[TargetedGate1q]) -> (usize, usize) {
        if gates.len() <= 1 {
            return (gates.len(), 0);
        }
        let max_l2 = self.max_l2_target();
        let (medium_tiled, shared) = self.shared_tiers();
        let (route, passes) = kernels::multi_1q_route(gates, self.n, max_l2, medium_tiled, shared);
        let mut small = SmallVec::new();
        let mut medium = SmallVec::new();
        let mut large = SmallVec::new();
        kernels::split_multi_1q_tiers(
            gates,
            max_l2,
            medium_tiled && matches!(route, Multi1qRoute::Tiered),
            &mut small,
            &mut medium,
            &mut large,
        );
        let shared_apps = |run: &[TargetedGate1q]| -> usize {
            run.chunks(shared)
                .filter(|chunk| chunk.len() > 1)
                .map(<[_]>::len)
                .sum()
        };
        let apps = match &route {
            Multi1qRoute::Whole(_) => gates.len(),
            Multi1qRoute::Tiered => small.len() + medium.len() + shared_apps(&large),
            Multi1qRoute::Split(runs) => {
                let mut rest = &large[..];
                let mut apps = small.len();
                for (len, plan) in runs {
                    let (run, tail) = rest.split_at(*len);
                    rest = tail;
                    apps += if plan.is_some() {
                        run.len()
                    } else {
                        shared_apps(run)
                    };
                }
                apps
            }
        };
        (passes, apps)
    }

    /// The L3 tier switch and the shared traversal width the `MultiFused` route reads.
    fn shared_tiers(&mut self) -> (bool, usize) {
        #[cfg(feature = "parallel")]
        if self.n >= PARALLEL_THRESHOLD_QUBITS {
            return (self.medium_tiled(), kernels::MAX_SHARED_1Q_TARGETS);
        }
        (true, 1)
    }

    fn multi_1q_diagonal(&self, gates: &[TargetedGate1q]) -> (usize, usize) {
        if gates.len() <= 1 {
            return (gates.len(), 0);
        }
        let mut small = SmallVec::new();
        let mut medium = SmallVec::new();
        let mut large = SmallVec::new();
        kernels::split_multi_1q_diagonal_tiers(
            gates,
            self.max_l2_target(),
            &mut small,
            &mut medium,
            &mut large,
        );
        let passes = usize::from(!small.is_empty()) + usize::from(!medium.is_empty()) + large.len();
        (passes, small.len() + medium.len())
    }

    fn multi_2q(&self, gates: &[(usize, usize, [[Complex64; 4]; 4])]) -> (usize, usize) {
        if gates.len() <= 1 {
            return (gates.len(), 0);
        }
        if kernels::subcube_plan(gates, self.n).is_some() {
            return (1, gates.len());
        }
        let tile = kernels::multi_2q_small_tile(self.n);
        let mut small = SmallVec::new();
        let mut medium = SmallVec::new();
        kernels::split_multi_2q_tiers(
            gates,
            kernels::max_target_for_tile(tile),
            &mut small,
            &mut medium,
        );
        let tiled = small.len() + medium.len();
        let passes =
            usize::from(!small.is_empty()) + usize::from(!medium.is_empty()) + gates.len() - tiled;
        (passes, tiled)
    }

    fn multi_pauli_rot(&self, rotations: &[(usize, usize, f64)]) -> (usize, usize) {
        let xall = rotations.iter().fold(0usize, |acc, &(x, _, _)| acc | x);
        let xqubits = (0..self.n).filter(|&q| xall >> q & 1 == 1);
        if kernels::qubit_subcube_plan(xqubits, self.n).is_some() {
            return (1, rotations.len());
        }
        let tile_bits = multi_2q_tile_bits_for(self.n).min(self.n);
        if xall >> tile_bits != 0 {
            (rotations.len(), 0)
        } else {
            (1, rotations.len())
        }
    }

    fn qft_block(
        &self,
        start: usize,
        num: usize,
        inverse: bool,
        swaps: bool,
        big_endian: bool,
    ) -> (usize, usize) {
        if start != 0 {
            return (
                qft_textbook_steps(start, num, inverse, swaps, big_endian).count(),
                0,
            );
        }
        let reverse_input = big_endian != (inverse && !swaps);
        let reverse_output = big_endian != (swaps || inverse);
        let widest = kernels::qft_widest_tile_bits_for(
            num,
            kernels::qft_max_tile_bits(),
            kernels::qft_min_tiles_log2(num, self.n),
        );
        let (high_passes, apps) = if num >= kernels::QFT_GATHER_MIN_QUBITS
            && num - widest >= kernels::QFT_GATHER_MIN_STAGES
        {
            ((num - widest).div_ceil(kernels::qft_gather_max_bits()), num)
        } else {
            let tile_bits = kernels::qft_tile_bits(num, self.n);
            let above = num - tile_bits;
            (above % 2 + above / 2, tile_bits)
        };
        let passes = usize::from(reverse_input) + high_passes + 1 + usize::from(reverse_output);
        (passes, apps)
    }
}

/// The bound of [`sweep_lower_bound`] on `insts`. A gate fits a pass whose tile holds
/// every qubit it acts on non-diagonally; a diagonal action costs no budget.
pub(crate) fn lower_bound(insts: &[Instruction], n: usize, threads: usize) -> usize {
    let mut run = TileRun::new(n);
    let mut profiler = Profiler::new(n, threads);
    let mut bound = 0;
    for inst in insts {
        match inst {
            Instruction::Gate { gate, targets }
            | Instruction::Conditional { gate, targets, .. } => match gate_support(gate, targets) {
                Some((touched, nondiag)) => run.add(touched, nondiag),
                None => bound += run.close() + profiler.passes_of(inst),
            },
            Instruction::Barrier { .. } => {}
            Instruction::Save { .. } => bound += run.close(),
            Instruction::Measure { .. } | Instruction::Reset { .. } | Instruction::Region(_) => {
                bound += run.close() + profiler.passes_of(inst);
            }
        }
    }
    bound + run.close()
}

/// The qubits `gate` touches and those it acts on non-diagonally, as masks, or `None`
/// for a gate no tile schedule runs (`QftBlock`).
fn gate_support(gate: &Gate, targets: &[usize]) -> Option<(u64, u64)> {
    let mask = |qubits: &mut dyn Iterator<Item = usize>| qubits.fold(0u64, |m, q| m | 1 << q);
    let touched = mask(&mut targets.iter().copied());
    let nondiag = match gate {
        Gate::QftBlock { .. } => return None,
        Gate::Cz
        | Gate::Rzz(_)
        | Gate::BatchPhase(_)
        | Gate::BatchRzz(_)
        | Gate::DiagonalBatch(_) => 0,
        Gate::Mcu(data) => {
            if is_diagonal_2x2(&data.mat) {
                0
            } else {
                1 << targets[targets.len() - 1]
            }
        }
        Gate::PauliRot(data) => mask(
            &mut targets
                .iter()
                .zip(data.axes())
                .filter(|(_, axis)| !matches!(axis, PauliAxis::Z))
                .map(|(&q, _)| q),
        ),
        Gate::MultiPauliRot(data) => data.rotations.iter().fold(0, |m, &(x, _, _)| m | x as u64),
        Gate::MultiFused(data) => mask(
            &mut data
                .gates
                .iter()
                .filter(|(_, mat)| !is_diagonal_2x2(mat))
                .map(|&(q, _)| q),
        ),
        Gate::Multi2q(data) => data.gates.iter().fold(0, |m, &(q0, q1, ref mat)| {
            let flat: Vec<Complex64> = mat.iter().flatten().copied().collect();
            m | dense_nondiagonal(&flat, &[q0, q1])
        }),
        _ => dense_nondiagonal(&gate.dense_matrix()?, targets),
    };
    Some((touched | nondiag, nondiag))
}

/// The targets a row-major dense matrix moves, `targets[0]` the most significant bit.
fn dense_nondiagonal(mat: &[Complex64], targets: &[usize]) -> u64 {
    let k = targets.len();
    let dim = 1usize << k;
    let mut flips = 0usize;
    for (index, value) in mat.iter().enumerate() {
        if value.norm() >= IDENTITY_EPS {
            flips |= (index / dim) ^ (index % dim);
        }
    }
    targets
        .iter()
        .enumerate()
        .filter(|&(i, _)| flips >> (k - 1 - i) & 1 == 1)
        .fold(0, |m, (_, &q)| m | 1 << q)
}

/// The greedy segmentation of one dependency chain: segments so far, the open one
/// included, and the high qubits the open segment holds.
#[derive(Clone, Copy)]
struct Chain {
    segments: usize,
    high: u64,
}

fn longer(a: Option<Chain>, b: Option<Chain>) -> Option<Chain> {
    match (a, b) {
        (Some(x), Some(y)) => Some(
            if (y.segments, y.high.count_ones()) > (x.segments, x.high.count_ones()) {
                y
            } else {
                x
            },
        ),
        (x, None) => x,
        (None, y) => y,
    }
}

/// One run of tileable gates between opaque instructions. A gate depends on the last
/// gate to act non-diagonally on any qubit it touches and, on a qubit it moves itself,
/// on every diagonal gate since; gates diagonal on all shared qubits commute.
struct TileRun {
    budget: usize,
    high_mask: u64,
    cover: u64,
    longest: usize,
    writer: Vec<Option<Chain>>,
    readers: Vec<Option<Chain>>,
}

impl TileRun {
    fn new(n: usize) -> Self {
        let low = multi_2q_low_bits().min(n);
        Self {
            budget: multi_2q_high_budget_for(n),
            high_mask: (u64::MAX >> (64 - n)) & !((1u64 << low) - 1),
            cover: 0,
            longest: 0,
            writer: vec![None; n],
            readers: vec![None; n],
        }
    }

    fn add(&mut self, touched: u64, nondiag: u64) {
        let high = nondiag & self.high_mask;
        self.cover |= high;
        let mut pred = None;
        for q in bits(touched) {
            pred = longer(pred, self.writer[q]);
            if nondiag >> q & 1 == 1 {
                pred = longer(pred, self.readers[q]);
            }
        }
        let chain = match pred {
            Some(c) if ((c.high | high).count_ones() as usize) <= self.budget => Chain {
                segments: c.segments,
                high: c.high | high,
            },
            Some(c) => Chain {
                segments: c.segments + 1,
                high,
            },
            None => Chain { segments: 1, high },
        };
        for q in bits(touched) {
            if nondiag >> q & 1 == 1 {
                self.writer[q] = Some(chain);
                self.readers[q] = None;
            } else {
                self.readers[q] = longer(self.readers[q], Some(chain));
            }
        }
        self.longest = self.longest.max(chain.segments);
    }

    /// The run's bound, leaving the run empty.
    fn close(&mut self) -> usize {
        let cover = (self.cover.count_ones() as usize).div_ceil(self.budget);
        let bound = self.longest.max(cover);
        self.cover = 0;
        self.longest = 0;
        self.writer.fill(None);
        self.readers.fill(None);
        bound
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
thread_local! {
    static PASS_TICKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Count one full-state pass on this thread, at each kernel entry the profile models.
#[cfg(test)]
pub(crate) fn tick() {
    PASS_TICKS.with(|ticks| ticks.set(ticks.get() + 1));
}

#[cfg(test)]
pub(super) fn take_ticks() -> usize {
    PASS_TICKS.with(|ticks| ticks.replace(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuits;

    const SEED: u64 = 0xDEAD_BEEF;

    fn take_ticks() -> usize {
        PASS_TICKS.with(|ticks| ticks.replace(0))
    }

    fn trotter(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for (coefficient, factors) in &circuits::jordan_wigner_hamiltonian(n, 200, SEED) {
            if !factors.is_empty() {
                c.add_pauli_rotation(0.1 * coefficient, factors);
            }
        }
        c
    }

    fn measured(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 2);
        c.instructions = circuits::hardware_efficient_ansatz(n, 2, SEED).instructions;
        c.add_measure(3, 0);
        c.add_reset(n - 1);
        c.add_gate(Gate::H, &[n - 1]);
        c.add_measure(n - 1, 1);
        c
    }

    fn corpus(n: usize) -> Vec<(&'static str, Circuit)> {
        vec![
            ("hea_l5", circuits::hardware_efficient_ansatz(n, 5, SEED)),
            ("random_d10", circuits::random_circuit(n, 10, SEED)),
            ("qaoa_l3", circuits::qaoa_circuit(n, 3, SEED)),
            ("qv", circuits::quantum_volume_circuit(n, n, SEED)),
            ("qft_textbook", circuits::qft_circuit(n)),
            (
                "diag_mixed_l6",
                circuits::diagonal_mixed_circuit(n, 6, SEED),
            ),
            ("trotter", trotter(n)),
            ("measured", measured(n)),
        ]
    }

    fn pool_width() -> usize {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads()
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    }

    #[test]
    fn sweep_profile_matches_kernel_pass_ticks() {
        for n in [14, 16, 18, 20] {
            for (name, circuit) in corpus(n) {
                let mut backend = StatevectorBackend::new(42);
                let expanded = crate::sim::expand_for_backend(&backend, &circuit);
                let fused = crate::sim::fuse_for_backend(&backend, &expanded);
                backend.init(n, circuit.num_classical_bits).unwrap();
                let threads = pool_width();
                take_ticks();
                for (i, inst) in fused.instructions.iter().enumerate() {
                    backend.apply(inst).unwrap();
                    let ticked = take_ticks();
                    let modeled = sweep_profile(std::slice::from_ref(inst), n, threads);
                    let kind = SweepKind::ALL
                        .into_iter()
                        .find(|&kind| modeled.count(kind).instructions > 0)
                        .map_or("barrier", SweepKind::name);
                    assert_eq!(
                        ticked,
                        modeled.passes(),
                        "{name}/{n} instruction {i} ({kind})"
                    );
                }
                let profile = sweep_profile_for(&circuit, threads);
                assert_eq!(profile, sweep_profile(&fused.instructions, n, threads));
                let by_kind = SweepKind::ALL.map(|kind| profile.count(kind));
                assert_eq!(
                    by_kind.iter().map(|c| c.tile_apps).sum::<usize>(),
                    profile.tile_apps()
                );
            }
        }
    }

    #[test]
    fn sweep_lower_bound_counts_cover_and_chains() {
        let n = 40;
        let low = multi_2q_low_bits();
        let budget = multi_2q_high_budget_for(n);
        let qubits: Vec<usize> = (low..low + 2 * budget).collect();
        let gate = |gate: Gate, targets: &[usize]| Instruction::Gate {
            gate,
            targets: targets.iter().copied().collect(),
        };

        let independent: Vec<Instruction> = qubits.iter().map(|&q| gate(Gate::H, &[q])).collect();
        assert_eq!(lower_bound(&independent, n, 1), 2);

        let swaps: Vec<Instruction> = qubits
            .windows(2)
            .map(|pair| gate(Gate::Swap, pair))
            .collect();
        assert_eq!(lower_bound(&swaps, n, 1), 3);

        let mut diagonal: Vec<Instruction> =
            qubits.windows(2).map(|pair| gate(Gate::Cz, pair)).collect();
        diagonal.push(gate(Gate::Cx, &[qubits[0], low - 1]));
        assert_eq!(lower_bound(&diagonal, n, 1), 1);

        let mut split = independent[..2].to_vec();
        split.push(Instruction::Measure {
            qubit: qubits[0],
            classical_bit: 0,
        });
        split.extend(independent[2..4].iter().cloned());
        assert_eq!(lower_bound(&split, n, 1), 4);
    }

    #[test]
    fn sweep_lower_bound_stays_under_the_profiled_passes() {
        for (name, circuit) in corpus(20) {
            let bound = sweep_lower_bound(&circuit, pool_width());
            let passes = sweep_profile_for(&circuit, pool_width()).passes();
            assert!(
                bound >= 1 && bound <= passes,
                "{name}: bound {bound}, passes {passes}"
            );
        }
    }
}
