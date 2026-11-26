//! Monte Carlo trajectory execution for noisy circuits: one pure-state
//! simulation per shot, sampling a noise branch after each instruction, or one
//! per distinct error pattern when every shot's Pauli errors can be drawn first.

use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;
use crate::limits::ResolvedLimits;
use crate::sim::ShotsResult;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError, pair_branch};
use crate::sim::unified_pauli::PauliAxis;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Noise sampler for one shot, on a ChaCha stream of its own.
///
/// The backend is seeded from the same shot seed and is also `ChaCha8Rng`, so on
/// the default stream the two emit the same sequence and a branch draw lands on
/// the same value as the measurement draw after it. Amplitude damping at
/// `gamma = 0.15` on `H|0>` read `P(1) = 0.384` over 200k shots against an exact
/// 0.425. A second stream keeps one seed reproducible and the draws independent.
pub(crate) fn noise_rng(shot_seed: u64) -> ChaCha8Rng {
    let mut rng = ChaCha8Rng::seed_from_u64(shot_seed);
    rng.set_stream(1);
    rng
}

/// A [`NoiseModel`]'s events lowered once per run, slot for slot in the
/// [`NoiseModel::after_gate`] layout, so a shot loop reads each channel's
/// constants instead of deriving them per event.
///
/// Each constant is the expression its draw would otherwise evaluate, in the same
/// order, so a seeded run draws the same bits.
pub(crate) struct PreparedNoise {
    events: Vec<PreparedEvent>,
    /// Slot `s` holds `events[offsets[s]..offsets[s + 1]]`.
    offsets: Vec<usize>,
}

impl PreparedNoise {
    pub(crate) fn new(noise: &NoiseModel) -> Self {
        Self::from_model(noise, &[])
    }

    pub(crate) fn compact_sources<'a>(
        models: impl IntoIterator<Item = &'a NoiseModel>,
    ) -> Vec<usize> {
        let mut sources: Vec<usize> = models
            .into_iter()
            .flat_map(|noise| noise.after_gate.iter().flatten())
            .filter_map(|event| match &event.channel {
                NoiseChannel::QuasiStatic { weights, .. } => {
                    Some(weights.iter().map(|&(source, _)| source))
                }
                _ => None,
            })
            .flatten()
            .collect();
        sources.sort_unstable();
        sources.dedup();
        sources
    }

    pub(crate) fn with_memory(noise: &NoiseModel) -> (Self, Option<MemorySpec>) {
        let sources = Self::compact_sources([noise]);
        let prepared = Self::from_model(noise, &sources);
        let memory = MemorySpec::with_sources(noise.has_leakage(), sources.len());
        (prepared, memory)
    }

    pub(crate) fn from_model(noise: &NoiseModel, sources: &[usize]) -> Self {
        let mut events = Vec::with_capacity(noise.after_gate.iter().map(Vec::len).sum());
        let mut offsets = Vec::with_capacity(noise.after_gate.len() + 1);
        offsets.push(0);
        for slot in &noise.after_gate {
            events.extend(
                slot.iter()
                    .map(|event| PreparedEvent::with_sources(event, sources)),
            );
            offsets.push(events.len());
        }
        Self { events, offsets }
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub(crate) fn slot(&self, slot: usize) -> &[PreparedEvent] {
        &self.events[self.offsets[slot]..self.offsets[slot + 1]]
    }

    /// Slot `slot`, or no events past the slots the model lays out: a model of one
    /// slot per top-level instruction gives guarded bodies none.
    #[inline]
    pub(crate) fn laid_out_slot(&self, slot: usize) -> &[PreparedEvent] {
        match self.offsets.get(slot + 1) {
            Some(&end) => &self.events[self.offsets[slot]..end],
            None => &[],
        }
    }

    /// Slots `from..` in order.
    #[inline]
    pub(crate) fn slots_from(&self, from: usize) -> impl Iterator<Item = &[PreparedEvent]> {
        self.offsets[from..]
            .windows(2)
            .map(|bounds| &self.events[bounds[0]..bounds[1]])
    }

    /// Every event of slots `..slot`, in slot order.
    fn events_before(&self, slot: usize) -> &[PreparedEvent] {
        &self.events[..self.offsets[slot]]
    }
}

pub(crate) struct PreparedEvent {
    pub(crate) channel: PreparedChannel,
    /// Targets in the order the channel indexes them. A one-qubit event repeats
    /// its target, so a leak check over both entries reads that one qubit.
    pub(crate) qubits: [usize; 2],
}

impl PreparedEvent {
    fn with_sources(event: &NoiseEvent, sources: &[usize]) -> Self {
        let q0 = event.qubits[0];
        Self {
            channel: PreparedChannel::with_sources(&event.channel, sources),
            qubits: [q0, event.qubits.get(1).copied().unwrap_or(q0)],
        }
    }
}

/// A [`NoiseChannel`] with what its draws read precomputed.
pub(crate) enum PreparedChannel {
    /// Thermal relaxation over a nonpositive time or lifetime, which draws
    /// nothing.
    Identity,
    /// Pauli and depolarizing channels.
    Pauli(PauliThresholds),
    AmplitudeDamping(Damping),
    PhaseDamping(Damping),
    Thermal(Box<ThermalBranches>),
    /// Two-qubit depolarizing at total rate `p`, each of its 15 branches `p / 15`
    /// wide.
    PauliPair {
        p: f64,
        branch_width: f64,
    },
    Unitary1q(Box<[[Complex64; 2]; 2]>),
    Custom(Box<[Kraus1q]>),
    Kraus2q(Box<[Kraus2q]>),
    Leakage(f64),
    Seepage(f64),
    LeakageTransport(f64),
    QuasiStatic {
        axis: PauliAxis,
        weights: Box<[(usize, f64)]>,
    },
}

impl PreparedChannel {
    #[cfg(any(test, feature = "bench-internal"))]
    fn new(channel: &NoiseChannel) -> Self {
        Self::with_sources(channel, &[])
    }

    fn with_sources(channel: &NoiseChannel, sources: &[usize]) -> Self {
        match channel {
            NoiseChannel::Pauli { px, py, pz } => Self::Pauli(PauliThresholds::new(*px, *py, *pz)),
            NoiseChannel::Depolarizing { p } => {
                let pp = p / 3.0;
                Self::Pauli(PauliThresholds::new(pp, pp, pp))
            }
            NoiseChannel::AmplitudeDamping { gamma } => {
                Self::AmplitudeDamping(Damping::new(*gamma))
            }
            NoiseChannel::PhaseDamping { gamma } => Self::PhaseDamping(Damping::new(*gamma)),
            NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                gate_time,
                excited_population,
            } => {
                if *t1 <= 0.0 || *t2 <= 0.0 || *gate_time <= 0.0 {
                    return Self::Identity;
                }
                let (gad, gpd) = crate::sim::noise::thermal_relaxation_rates(*t1, *t2, *gate_time);
                Self::Thermal(Box::new(ThermalBranches::new(
                    gad,
                    gpd,
                    *excited_population,
                )))
            }
            NoiseChannel::TwoQubitDepolarizing { p } => Self::PauliPair {
                p: *p,
                branch_width: p / 15.0,
            },
            NoiseChannel::Custom { kraus } => {
                if kraus.len() == 1
                    && crate::sim::noise::validate_kraus_set("custom", kraus).is_ok()
                {
                    Self::Unitary1q(Box::new(kraus[0]))
                } else {
                    Self::Custom(kraus.iter().map(Kraus1q::new).collect())
                }
            }
            NoiseChannel::Kraus2q { kraus } => Self::Kraus2q(
                kraus
                    .iter()
                    .map(|op| Kraus2q {
                        op: *op,
                        effect: kdagger_k(op),
                    })
                    .collect(),
            ),
            NoiseChannel::Leakage { p } => Self::Leakage(*p),
            NoiseChannel::Seepage { p } => Self::Seepage(*p),
            NoiseChannel::LeakageTransport { p } => Self::LeakageTransport(*p),
            NoiseChannel::QuasiStatic { axis, weights } => {
                if weights.is_empty() {
                    Self::Identity
                } else {
                    let mapped: Box<[(usize, f64)]> = if sources.is_empty() {
                        weights.as_slice().into()
                    } else {
                        weights
                            .iter()
                            .map(|&(source, weight)| {
                                (
                                    sources
                                        .binary_search(&source)
                                        .expect("active quasi-static source was collected"),
                                    weight,
                                )
                            })
                            .collect()
                    };
                    Self::QuasiStatic {
                        axis: *axis,
                        weights: mapped,
                    }
                }
            }
        }
    }
}

/// Cumulative branch bounds of a single-qubit Pauli channel: `X` below `x`,
/// `Y` below `xy`, `Z` below `xyz`.
#[derive(Clone, Copy)]
pub(crate) struct PauliThresholds {
    x: f64,
    xy: f64,
    xyz: f64,
}

impl PauliThresholds {
    fn new(px: f64, py: f64, pz: f64) -> Self {
        Self {
            x: px,
            xy: px + py,
            xyz: px + py + pz,
        }
    }

    /// Draw one branch, `None` for the identity.
    #[inline]
    fn draw(self, rng: &mut ChaCha8Rng) -> Option<PauliOp> {
        let r: f64 = rand::RngExt::random(rng);
        if r < self.x {
            Some(PauliOp::X)
        } else if r < self.xy {
            Some(PauliOp::Y)
        } else if r < self.xyz {
            Some(PauliOp::Z)
        } else {
            None
        }
    }

    #[inline]
    pub(crate) fn apply(
        self,
        backend: &mut dyn Backend,
        qubit: usize,
        rng: &mut ChaCha8Rng,
    ) -> Result<()> {
        match self.draw(rng) {
            Some(op) => apply_pauli_op(backend, qubit, op),
            None => Ok(()),
        }
    }
}

/// Amplitude or phase damping at rate `gamma`, with the no-jump factor
/// `keep = sqrt(1 - gamma)`.
#[derive(Clone, Copy)]
pub(crate) struct Damping {
    gamma: f64,
    keep: f64,
}

impl Damping {
    fn new(gamma: f64) -> Self {
        Self {
            gamma,
            keep: (1.0 - gamma).sqrt(),
        }
    }
}

/// The state-independent factors of [`apply_thermal_relaxation`]'s five branch
/// weights and operators.
pub struct ThermalBranches {
    pub(crate) relax: f64,
    pub(crate) dephase: f64,
    pub(crate) excite: f64,
    pub(crate) excited: f64,
    pub(crate) decay_keep: f64,
    pub(crate) dephase_keep: f64,
    pub(crate) hot: [f64; 2],
    pub(crate) cold_scale: f64,
    pub(crate) cold_keep: f64,
}

impl ThermalBranches {
    fn new(gad: f64, gpd: f64, excited: f64) -> Self {
        let cold = 1.0 - excited;
        Self {
            relax: cold * gad,
            dephase: ((cold * (1.0 - gad)) + excited) * gpd,
            excite: excited * gad,
            excited,
            decay_keep: 1.0 - gad,
            dephase_keep: 1.0 - gpd,
            hot: [(1.0 - gad).sqrt(), (1.0 - gpd).sqrt()],
            cold_scale: cold.sqrt(),
            cold_keep: ((1.0 - gad) * (1.0 - gpd)).sqrt(),
        }
    }
}

/// A one-qubit Kraus operator with its effect `Kdagger K`.
pub struct Kraus1q {
    pub(crate) op: [[Complex64; 2]; 2],
    pub(crate) effect: [[Complex64; 2]; 2],
    /// Whether [`apply_custom_kraus`]'s normalized branch `K / sqrt(p_k)` meets the
    /// backend's diagonal test (both off-diagonal norms below `1e-14`) for every
    /// branch weight `p_k`: `Some(true)` when both off-diagonals are zero, `Some(false)`
    /// when one is at least `1e-12` of the Frobenius norm, `None` when `p_k` decides.
    #[cfg(feature = "gpu")]
    pub(crate) diagonal: Option<bool>,
}

impl Kraus1q {
    fn new(op: &[[Complex64; 2]; 2]) -> Self {
        Self {
            op: *op,
            effect: kdagger_k(op),
            #[cfg(feature = "gpu")]
            diagonal: branch_is_diagonal(op),
        }
    }
}

/// [`Kraus1q::diagonal`] for `op`.
#[cfg(feature = "gpu")]
fn branch_is_diagonal(op: &[[Complex64; 2]; 2]) -> Option<bool> {
    let zero = Complex64::new(0.0, 0.0);
    if op[0][1] == zero && op[1][0] == zero {
        return Some(true);
    }
    // p_k = Tr(E rho) <= Tr(E) = |K|_F^2 on a unit trace, so an off-diagonal at
    // 1e-12 |K|_F leaves the normalized one near 1e-12, far above the cut.
    let off = op[0][1].norm().max(op[1][0].norm());
    let frobenius = op
        .iter()
        .flatten()
        .map(|z| z.norm_sqr())
        .sum::<f64>()
        .sqrt();
    (off >= 1e-12 * frobenius).then_some(false)
}

/// A two-qubit Kraus operator, indexed as [`NoiseChannel::Kraus2q`] is, with its
/// effect `Kdagger K`.
pub(crate) struct Kraus2q {
    op: [[Complex64; 4]; 4],
    effect: [[Complex64; 4]; 4],
}

/// Minimum p_jump for applying the jump branch instead of the no-jump
/// branch. Below this, the jump is numerically indistinguishable from zero and
/// dividing by sqrt(p_jump) produces NaN/inf.
pub(crate) const JUMP_EPSILON: f64 = 1e-12;

/// Apply a single-qubit diagonal Kraus channel with 2 operators of the form:
///   K_0 = diag(1, c), no-jump
///   K_1 = [[0, s01_upper], [0, s11_lower]], jump (AD)
///       or diag(0, s), jump (PD, pass s01_upper=0)
///
/// where c = sqrt(1-gamma). The effective jump probability is gamma * p1, taken
/// when the uniform draw `r` falls below it.
///
/// No `p1` up to one lifts `gamma * p1` past an `r >= gamma`, so there a backend
/// offering [`damp_no_jump`](crate::backend::sealed::Sealed::damp_no_jump) applies
/// the no-jump branch in the pass that reads `p1`. Rounding can leave that `p1` just
/// past one, so the jump test reruns on it, and a jump it then picks undoes the
/// branch's scaling of the `|1>` half.
fn apply_diagonal_kraus_2op(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    jump_moves_population: bool,
    r: f64,
) -> Result<()> {
    let Damping { gamma, keep } = damping;
    if r >= gamma
        && let Some(p1) = backend.damp_no_jump(qubit, gamma, keep)
    {
        let p_jump = gamma * p1;
        if r < p_jump && p_jump > JUMP_EPSILON {
            let denom = 1.0 - p_jump;
            let applied = if denom > JUMP_EPSILON {
                keep / denom.sqrt()
            } else {
                1.0
            };
            let s = (gamma / p_jump).sqrt() / applied;
            return apply_damping_jump(backend, qubit, s, jump_moves_population);
        }
        return Ok(());
    }
    apply_damping_two_step(backend, qubit, damping, jump_moves_population, r)
}

/// [`apply_diagonal_kraus_2op`] as a `p1` read followed by a matrix pass, with the
/// branch chosen on the device where the backend can.
fn apply_damping_two_step(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    jump_moves_population: bool,
    r: f64,
) -> Result<()> {
    if let Some(applied) =
        backend.damp_on_device(qubit, damping.gamma, damping.keep, jump_moves_population, r)
    {
        return applied;
    }
    damping_host_decided(backend, qubit, damping, jump_moves_population, r)
}

fn damping_host_decided(
    backend: &mut dyn Backend,
    qubit: usize,
    Damping { gamma, keep }: Damping,
    jump_moves_population: bool,
    r: f64,
) -> Result<()> {
    let p1 = match backend.damping_probability(qubit) {
        Some(p1) => p1?,
        None => backend.qubit_probability(qubit)?,
    };
    let p_jump = gamma * p1;

    if r < p_jump && p_jump > JUMP_EPSILON {
        let s = (gamma / p_jump).sqrt();
        return apply_damping_jump(backend, qubit, s, jump_moves_population);
    }
    let denom = 1.0 - p_jump;
    if denom <= JUMP_EPSILON {
        return Ok(());
    }
    let inv = 1.0 / denom.sqrt();
    let zero = Complex64::new(0.0, 0.0);
    let mat = [
        [Complex64::new(inv, 0.0), zero],
        [zero, Complex64::new(keep * inv, 0.0)],
    ];
    backend.apply_1q_matrix(qubit, &mat)
}

/// The jump branch scaled by `s`: `|0><1|` when it moves population, `|1><1|` when it
/// dephases.
fn apply_damping_jump(
    backend: &mut dyn Backend,
    qubit: usize,
    s: f64,
    jump_moves_population: bool,
) -> Result<()> {
    let zero = Complex64::new(0.0, 0.0);
    let s = Complex64::new(s, 0.0);
    let mat = if jump_moves_population {
        [[zero, s], [zero, zero]]
    } else {
        [[zero, zero], [zero, s]]
    };
    backend.apply_1q_matrix(qubit, &mat)
}

fn apply_amplitude_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, damping, true, rand::RngExt::random(rng))
}

fn apply_phase_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, damping, false, rand::RngExt::random(rng))
}

/// One shot's pass over a circuit and its noise slots, driven by
/// [`walk_noise_slots`].
pub(crate) trait SlotWalk {
    /// The classical bits, current for a guard to read.
    fn classical(&mut self) -> Result<&[bool]>;
    /// Apply an instruction other than a region, whose body the walk descends
    /// into itself.
    fn apply(&mut self, instruction: &Instruction) -> Result<()>;
    /// Fire the events of [`NoiseModel::after_gate`] slot `slot`.
    fn fire(&mut self, slot: usize) -> Result<()>;
}

/// Run `circuit.instructions[from..]` in execution order under a model laying
/// out guarded slots, firing each slot after its instruction and a guarded
/// body's slots only when the body runs. See [`NoiseModel::after_gate`].
pub(crate) fn walk_noise_slots(
    circuit: &Circuit,
    from: usize,
    walk: &mut impl SlotWalk,
) -> Result<()> {
    let nested = circuit.instructions.len() + circuit.guarded_noise_slots_before(from);
    walk_slot_list(&circuit.instructions[from..], from, nested, walk)
}

fn walk_slot_list(
    list: &[Instruction],
    base: usize,
    mut nested: usize,
    walk: &mut impl SlotWalk,
) -> Result<()> {
    for (offset, instruction) in list.iter().enumerate() {
        match instruction {
            Instruction::Region(region) => {
                if region.condition().evaluate(walk.classical()?) {
                    walk_region_body(region, nested, walk)?;
                }
                nested += region.noise_slots();
            }
            Instruction::Conditional { condition, .. } => {
                let taken = condition.evaluate(walk.classical()?);
                walk.apply(instruction)?;
                if taken {
                    walk.fire(nested)?;
                }
                nested += 1;
            }
            _ => walk.apply(instruction)?,
        }
        walk.fire(base + offset)?;
    }
    Ok(())
}

#[inline(never)]
fn walk_region_body(
    region: &crate::circuit::GuardedRegion,
    base: usize,
    walk: &mut impl SlotWalk,
) -> Result<()> {
    walk_slot_list(region.body(), base, base + region.body().len(), walk)
}

/// A trajectory over a model with guarded slots, firing each event through `A`.
struct TrajectoryWalk<'a, A> {
    backend: &'a mut dyn Backend,
    noise: &'a PreparedNoise,
    readout: &'a [Option<ReadoutError>],
    rng: &'a mut ChaCha8Rng,
    events: PhantomData<A>,
}

impl<A: EventApplier> SlotWalk for TrajectoryWalk<'_, A> {
    fn classical(&mut self) -> Result<&[bool]> {
        self.backend.sync_deferred_measurements()?;
        Ok(self.backend.classical_results())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        self.backend.apply(instruction)?;
        flip_written_bit(self.backend, instruction, self.readout, self.rng)
    }

    fn fire(&mut self, slot: usize) -> Result<()> {
        for event in self.noise.laid_out_slot(slot) {
            A::apply(self.backend, event, self.rng)?;
        }
        Ok(())
    }
}

/// [`TrajectoryWalk`] on the per-shot memory path.
struct MemoryWalk<'a> {
    backend: &'a mut dyn Backend,
    noise: &'a PreparedNoise,
    memory: &'a mut ShotMemory,
    readout: &'a [Option<ReadoutError>],
    rng: &'a mut ChaCha8Rng,
}

impl SlotWalk for MemoryWalk<'_> {
    fn classical(&mut self) -> Result<&[bool]> {
        Ok(self.backend.classical_results())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        apply_memory_instruction(self.backend, instruction, self.memory, self.rng)?;
        flip_written_bit(self.backend, instruction, self.readout, self.rng)
    }

    fn fire(&mut self, slot: usize) -> Result<()> {
        for event in self.noise.laid_out_slot(slot) {
            apply_memory_event(self.backend, event, self.memory, self.rng)?;
        }
        Ok(())
    }
}

/// Unravel thermal relaxation as the amplitude-damping and phase-damping
/// composition [`crate::sim::noise::kraus_1q`] lowers it to, so the trajectory
/// average reproduces `exp(-t/t1)` populations and `exp(-t/t2)` coherences. A
/// mixture of reset and `Z` cannot: it needs a negative dephasing probability
/// whenever `t1 < t2 <= 2*t1`.
///
/// Every operator in the composition has a diagonal `Kdagger K`, so the branch
/// probabilities read off `P(1)` alone and need no reduced density matrix. The
/// two that decay are both proportional to `|0><1|` and leave the same
/// normalized state, as are the two that dephase, so each pair merges. That
/// leaves five branches behind one probability read and one matrix pass, and at
/// zero temperature the two that excite or hold the hot steady state carry no
/// weight, leaving the three-branch unraveling.
fn apply_thermal_relaxation(
    backend: &mut dyn Backend,
    qubit: usize,
    branches: &ThermalBranches,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let r: f64 = rand::RngExt::random(rng);
    if let Some(applied) = backend.thermal_on_device(qubit, branches, r) {
        return applied;
    }
    if branches.excited == 0.0 && branches.cold_keep > 0.0 {
        let gamma = branches.relax + branches.dephase;
        if r >= gamma
            && let Some(p1) = backend.damp_no_jump(qubit, gamma, branches.cold_keep)
        {
            let p_relax = branches.relax * p1;
            let p_dephase = branches.dephase * p1;
            if r < p_relax + p_dephase {
                let denom = 1.0 - gamma * p1;
                let applied = if denom > JUMP_EPSILON {
                    branches.cold_keep / denom.sqrt()
                } else {
                    1.0
                };
                let s = (1.0 / p1).sqrt() / applied;
                if r < p_relax && p_relax > JUMP_EPSILON {
                    return apply_damping_jump(backend, qubit, s, true);
                }
                if p_dephase > JUMP_EPSILON {
                    return apply_damping_jump(backend, qubit, s, false);
                }
            }
            return Ok(());
        }
    }
    thermal_host_decided(backend, qubit, branches, r)
}

fn thermal_host_decided(
    backend: &mut dyn Backend,
    qubit: usize,
    branches: &ThermalBranches,
    r: f64,
) -> Result<()> {
    let p1 = match backend.damping_probability(qubit) {
        Some(p1) => p1?,
        None => backend.qubit_probability(qubit)?,
    };
    let p0 = 1.0 - p1;

    let p_relax = branches.relax * p1;
    let p_dephase = branches.dephase * p1;
    let p_excite = branches.excite * p0;
    let p_hold_hot = branches.excited * (branches.decay_keep * p0 + branches.dephase_keep * p1);

    let zero = Complex64::new(0.0, 0.0);
    let mat = if r < p_relax && p_relax > JUMP_EPSILON {
        let s = (1.0 / p1).sqrt();
        [[zero, Complex64::new(s, 0.0)], [zero, zero]]
    } else if r < p_relax + p_dephase && p_dephase > JUMP_EPSILON {
        let s = (1.0 / p1).sqrt();
        [[zero, zero], [zero, Complex64::new(s, 0.0)]]
    } else if r < p_relax + p_dephase + p_excite && p_excite > JUMP_EPSILON {
        let s = (1.0 / p0).sqrt();
        [[zero, zero], [Complex64::new(s, 0.0), zero]]
    } else if r < p_relax + p_dephase + p_excite + p_hold_hot && p_hold_hot > JUMP_EPSILON {
        let inv = (branches.excited / p_hold_hot).sqrt();
        [
            [Complex64::new(branches.hot[0] * inv, 0.0), zero],
            [zero, Complex64::new(branches.hot[1] * inv, 0.0)],
        ]
    } else {
        let denom = 1.0 - p_relax - p_dephase - p_excite - p_hold_hot;
        if denom <= JUMP_EPSILON {
            return Ok(());
        }
        let inv = branches.cold_scale / denom.sqrt();
        let keep = branches.cold_keep * inv;
        [
            [Complex64::new(inv, 0.0), zero],
            [zero, Complex64::new(keep, 0.0)],
        ]
    };
    backend.apply_1q_matrix(qubit, &mat)
}

/// Draw one branch of symmetric two-qubit depolarizing as an index into
/// [`TWO_QUBIT_PAULIS`], `None` for `I (x) I`. Each of the 15 non-identity
/// products has probability `branch_width = p/15`.
#[inline]
fn draw_two_qubit_depolarizing(p: f64, branch_width: f64, rng: &mut ChaCha8Rng) -> Option<usize> {
    let r: f64 = rand::RngExt::random(rng);
    if r >= p {
        return None;
    }
    Some(((r / branch_width) as usize).min(14))
}

fn apply_two_qubit_depolarizing(
    backend: &mut dyn Backend,
    [q0, q1]: [usize; 2],
    p: f64,
    branch_width: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let Some(idx) = draw_two_qubit_depolarizing(p, branch_width, rng) else {
        return Ok(());
    };
    let (pauli0, pauli1) = TWO_QUBIT_PAULIS[idx];
    apply_pauli_op(backend, q0, pauli0)?;
    apply_pauli_op(backend, q1, pauli1)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PauliOp {
    I,
    X,
    Y,
    Z,
}

const TWO_QUBIT_PAULIS: [(PauliOp, PauliOp); 15] = [
    (PauliOp::I, PauliOp::X),
    (PauliOp::I, PauliOp::Y),
    (PauliOp::I, PauliOp::Z),
    (PauliOp::X, PauliOp::I),
    (PauliOp::X, PauliOp::X),
    (PauliOp::X, PauliOp::Y),
    (PauliOp::X, PauliOp::Z),
    (PauliOp::Y, PauliOp::I),
    (PauliOp::Y, PauliOp::X),
    (PauliOp::Y, PauliOp::Y),
    (PauliOp::Y, PauliOp::Z),
    (PauliOp::Z, PauliOp::I),
    (PauliOp::Z, PauliOp::X),
    (PauliOp::Z, PauliOp::Y),
    (PauliOp::Z, PauliOp::Z),
];

fn apply_pauli_op<B: Backend + ?Sized>(backend: &mut B, qubit: usize, op: PauliOp) -> Result<()> {
    let gate = match op {
        PauliOp::X => Gate::X,
        PauliOp::Y => Gate::Y,
        PauliOp::Z => Gate::Z,
        PauliOp::I => return Ok(()),
    };
    backend.apply(&Instruction::Gate {
        gate,
        targets: smallvec![qubit],
    })
}

/// `Kdagger K`, entry `[i][j] = sum_r conj(K[r][i]) K[r][j]`.
#[inline]
fn kdagger_k<const D: usize>(k: &[[Complex64; D]; D]) -> [[Complex64; D]; D] {
    let mut effect = [[Complex64::new(0.0, 0.0); D]; D];
    for (i, effect_row) in effect.iter_mut().enumerate() {
        for (j, entry) in effect_row.iter_mut().enumerate() {
            for row in k.iter() {
                *entry += row[i].conj() * row[j];
            }
        }
    }
    effect
}

/// `Tr(E rho) = sum_{i,j} E[i][j] rho[j][i]` for a branch effect `E`, clamped
/// at zero.
#[inline]
fn effect_probability<const D: usize>(
    effect: &[[Complex64; D]; D],
    rho: &[[Complex64; D]; D],
) -> f64 {
    let mut p = Complex64::new(0.0, 0.0);
    for i in 0..D {
        for j in 0..D {
            p += effect[i][j] * rho[j][i];
        }
    }
    p.re.max(0.0)
}

/// Draw a branch with probability proportional to its entry of `probs`, returning its
/// index and probability. `None` when the total or the drawn branch is at most
/// [`JUMP_EPSILON`], and a total that small makes no draw.
#[inline]
fn pick_branch(probs: impl Iterator<Item = f64>, rng: &mut ChaCha8Rng) -> Option<(usize, f64)> {
    let mut cumulative: smallvec::SmallVec<[f64; 16]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for pk in probs {
        total += pk;
        cumulative.push(total);
    }

    if total <= JUMP_EPSILON {
        return None;
    }

    let r: f64 = rand::RngExt::random::<f64>(rng) * total;
    let chosen = cumulative
        .iter()
        .position(|&c| r < c)
        .unwrap_or(cumulative.len() - 1);
    let pk = if chosen == 0 {
        cumulative[0]
    } else {
        cumulative[chosen] - cumulative[chosen - 1]
    };
    // Compared with `<=` so a NaN `pk` still applies the branch.
    if pk <= JUMP_EPSILON {
        None
    } else {
        Some((chosen, pk))
    }
}

/// Sample and apply one of a set of 1-qubit Kraus operators.
///
/// Branch probabilities use `p_k = Tr(Kdagger K rho_q)`, where `rho_q` is
/// the qubit's reduced density matrix. This handles dense Kraus operators
/// whose branch probabilities depend on coherence.
///
/// A backend deciding on the device draws the branch unconditionally, where the host
/// skips the draw at `total <= JUMP_EPSILON`. A trace preserving channel has
/// `total = Tr(rho) = 1`, so the streams part only on a zero state.
fn apply_custom_kraus(
    backend: &mut dyn Backend,
    qubit: usize,
    kraus: &[Kraus1q],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if let Some(applied) = backend.kraus_on_device(qubit, kraus, rng) {
        return applied;
    }
    kraus_host_decided(backend, qubit, kraus, rng)
}

fn kraus_host_decided(
    backend: &mut dyn Backend,
    qubit: usize,
    kraus: &[Kraus1q],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    backend.sync_deferred_measurements()?;
    let rho = backend.reduced_density_matrix_1q(qubit)?;

    let probs = kraus.iter().map(|k| effect_probability(&k.effect, &rho));
    let Some((chosen, pk)) = pick_branch(probs, rng) else {
        return Ok(());
    };
    let inv = 1.0 / pk.sqrt();
    let inv_c = Complex64::new(inv, 0.0);
    let k = kraus[chosen].op;
    let normalized = [
        [k[0][0] * inv_c, k[0][1] * inv_c],
        [k[1][0] * inv_c, k[1][1] * inv_c],
    ];

    backend.apply_1q_matrix(qubit, &normalized)
}

/// Sample and apply one of a set of two-qubit Kraus operators, indexed
/// `K[t][t']` with `t = 2 * bit(q0) + bit(q1)`.
///
/// Branch probabilities are `p_k = Tr(Kdagger K rho)` over the pair's reduced
/// density matrix, so a correlated channel sees the coherence between the two
/// qubits and not only their populations. Backends without
/// [`Backend::reduced_density_matrix_2q`] report that rather than sampling a
/// branch from an approximation.
fn apply_custom_kraus_2q(
    backend: &mut dyn Backend,
    [q0, q1]: [usize; 2],
    kraus: &[Kraus2q],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let rho = backend.reduced_density_matrix_2q(q0, q1)?;

    let probs = kraus.iter().map(|k| effect_probability(&k.effect, &rho));
    let Some((chosen, pk)) = pick_branch(probs, rng) else {
        return Ok(());
    };

    let inv = Complex64::new(1.0 / pk.sqrt(), 0.0);
    let mut normalized = kraus[chosen].op;
    for row in normalized.iter_mut() {
        for entry in row.iter_mut() {
            *entry *= inv;
        }
    }
    backend.apply(&Instruction::Gate {
        gate: Gate::Fused2q(Box::new(normalized)),
        targets: smallvec![q0, q1],
    })
}

fn apply_noise_event(
    backend: &mut dyn Backend,
    event: &PreparedEvent,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let qubit = event.qubits[0];
    match &event.channel {
        PreparedChannel::Identity => Ok(()),
        PreparedChannel::Pauli(thresholds) => thresholds.apply(backend, qubit, rng),
        PreparedChannel::AmplitudeDamping(damping) => {
            apply_amplitude_damping(backend, qubit, *damping, rng)
        }
        PreparedChannel::PhaseDamping(damping) => {
            apply_phase_damping(backend, qubit, *damping, rng)
        }
        PreparedChannel::Thermal(branches) => {
            apply_thermal_relaxation(backend, qubit, branches, rng)
        }
        PreparedChannel::PauliPair { p, branch_width } => {
            apply_two_qubit_depolarizing(backend, event.qubits, *p, *branch_width, rng)
        }
        PreparedChannel::Unitary1q(matrix) => backend.apply_1q_matrix(qubit, matrix),
        PreparedChannel::Custom(kraus) => apply_custom_kraus(backend, qubit, kraus, rng),
        PreparedChannel::Kraus2q(kraus) => apply_custom_kraus_2q(backend, event.qubits, kraus, rng),
        PreparedChannel::QuasiStatic { weights, .. } if weights.is_empty() => Ok(()),
        PreparedChannel::Leakage(_)
        | PreparedChannel::Seepage(_)
        | PreparedChannel::LeakageTransport(_)
        | PreparedChannel::QuasiStatic { .. } => {
            unreachable!("channels with per-shot memory run through apply_memory_event")
        }
    }
}

/// How a trajectory fires one noise event.
trait EventApplier {
    fn apply(backend: &mut dyn Backend, event: &PreparedEvent, rng: &mut ChaCha8Rng) -> Result<()>;
}

/// Every channel unraveled by [`apply_noise_event`].
struct NoiseEvents;

impl EventApplier for NoiseEvents {
    #[inline(always)]
    fn apply(backend: &mut dyn Backend, event: &PreparedEvent, rng: &mut ChaCha8Rng) -> Result<()> {
        apply_noise_event(backend, event, rng)
    }
}

/// Pauli events drawn by [`apply_frame_event`], on the frame sampler's branch stream
/// rather than the [`TWO_QUBIT_PAULIS`] draw.
struct FrameEvents;

impl EventApplier for FrameEvents {
    #[inline(always)]
    fn apply(backend: &mut dyn Backend, event: &PreparedEvent, rng: &mut ChaCha8Rng) -> Result<()> {
        apply_frame_event(backend, event, rng)
    }
}

/// Draw one Pauli event and apply the letters it fires.
#[inline]
fn apply_frame_event(
    backend: &mut dyn Backend,
    event: &PreparedEvent,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match &event.channel {
        PreparedChannel::PauliPair { p, .. } => {
            let r: f64 = rand::RngExt::random(rng);
            if r < *p {
                let (l0, l1) = pair_branch(r / p);
                apply_pauli_letter(backend, event.qubits[0], l0)?;
                apply_pauli_letter(backend, event.qubits[1], l1)?;
            }
            Ok(())
        }
        PreparedChannel::Pauli(thresholds) => thresholds.apply(backend, event.qubits[0], rng),
        _ => unreachable!("frame replay runs Pauli-frame channels only"),
    }
}

/// Apply one letter of a two-qubit branch, indexed as [`pair_branch`] returns
/// it. Identity applies nothing.
fn apply_pauli_letter(backend: &mut dyn Backend, qubit: usize, letter: usize) -> Result<()> {
    let gate = match letter {
        1 => Gate::X,
        2 => Gate::Y,
        3 => Gate::Z,
        _ => return Ok(()),
    };
    backend.apply(&Instruction::Gate {
        gate,
        targets: SmallVec::from_elem(qubit, 1),
    })
}

/// Apply each `(qubit, channel)` site in turn on `backend` as a trajectory does, drawing
/// from the noise stream of `seed`. `host_decided` reads every damping, thermal and
/// one-qubit Kraus branch back to the host as a backend without device decisions does,
/// so the two runs can be compared bit for bit.
#[cfg(feature = "bench-internal")]
#[doc(hidden)]
pub fn apply_noise_sites(
    backend: &mut dyn Backend,
    sites: &[(usize, NoiseChannel)],
    seed: u64,
    host_decided: bool,
) -> Result<()> {
    let mut rng = noise_rng(seed);
    for (qubit, channel) in sites {
        let event = PreparedEvent {
            channel: PreparedChannel::new(channel),
            qubits: [*qubit; 2],
        };
        if !host_decided {
            apply_noise_event(backend, &event, &mut rng)?;
            continue;
        }
        match &event.channel {
            PreparedChannel::AmplitudeDamping(damping) => damping_host_decided(
                backend,
                *qubit,
                *damping,
                true,
                rand::RngExt::random(&mut rng),
            )?,
            PreparedChannel::PhaseDamping(damping) => damping_host_decided(
                backend,
                *qubit,
                *damping,
                false,
                rand::RngExt::random(&mut rng),
            )?,
            PreparedChannel::Thermal(branches) => {
                let r: f64 = rand::RngExt::random(&mut rng);
                thermal_host_decided(backend, *qubit, branches, r)?
            }
            PreparedChannel::Custom(kraus) => kraus_host_decided(backend, *qubit, kraus, &mut rng)?,
            _ => apply_noise_event(backend, &event, &mut rng)?,
        }
    }
    Ok(())
}

/// Run one trajectory shot of `circuit` under `noise` on `backend` from shot seed `seed`
/// and return its record. `defer` keeps measurement outcomes on the device between the
/// reads that need them, as a trajectory does; without it each outcome reads back as its
/// measurement lands, so the two runs can be compared bit for bit.
#[cfg(feature = "bench-internal")]
#[doc(hidden)]
pub fn run_noisy_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
    defer: bool,
) -> Result<Vec<bool>> {
    let prepared = PreparedNoise::new(noise);
    let readout = ShotReadout::new(circuit, &noise.readout);
    let mut rng = noise_rng(seed);
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    if defer {
        evolve_noisy(backend, circuit, &prepared, 0, &readout.at_write, &mut rng)?;
    } else {
        evolve_noisy_eager::<NoiseEvents>(
            backend,
            circuit,
            &prepared,
            0,
            &readout.at_write,
            &mut rng,
        )?;
    }
    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, &readout.record, &mut rng);
    Ok(results)
}

/// What a model asks of each shot beyond its event stream: leak flags when it
/// carries a leakage channel, and `sources` quasi-static standard normal draws.
#[derive(Clone, Copy)]
pub(crate) struct MemorySpec {
    leakage: bool,
    sources: usize,
}

impl MemorySpec {
    /// `None` for a model whose events need no per-shot memory, which keeps the
    /// trajectory loop and its draws exactly as they were before these channels.
    pub(crate) fn with_sources(leakage: bool, sources: usize) -> Option<Self> {
        (leakage || sources > 0).then_some(Self { leakage, sources })
    }

    #[cfg(test)]
    pub(crate) fn source_count(self) -> usize {
        self.sources
    }
}

/// Leak flags and quasi-static draws of one shot.
///
/// A qubit is reset and flipped to `|1>` when it leaks, and nothing acts on it
/// again until it returns. The reset is one trajectory of the channel that
/// traces the qubit out (see [`Backend::reset`]), so averaged over shots every
/// other qubit holds the reduced state it would hold had the qubit left the
/// register. The leaked qubit is then an unentangled `|1>`: measuring it
/// reports 1 and collapses nothing else, and a return to a random basis state
/// takes one coin and at most one `X`.
pub(crate) struct ShotMemory {
    leakage: bool,
    leaked: Vec<bool>,
    ever_leaked: Vec<bool>,
    drift: Vec<f64>,
}

impl ShotMemory {
    /// Draw the shot's quasi-static normals from `rng`, before any of its events.
    pub(crate) fn new(num_qubits: usize, spec: MemorySpec, rng: &mut ChaCha8Rng) -> Self {
        let mut drift = vec![0.0; spec.sources];
        standard_normals(rng, &mut drift);
        let flags = if spec.leakage { num_qubits } else { 0 };
        Self {
            leakage: spec.leakage,
            leaked: vec![false; flags],
            ever_leaked: vec![false; flags],
            drift,
        }
    }

    /// Return which qubits leaked at any point of the shot, empty without leakage.
    pub(crate) fn into_ever_leaked(self) -> Vec<bool> {
        self.ever_leaked
    }

    #[inline]
    fn is_leaked(&self, qubit: usize) -> bool {
        self.leaked.get(qubit).copied().unwrap_or(false)
    }

    fn any_leaked(&self, qubits: &[usize]) -> bool {
        qubits.iter().any(|&q| self.is_leaked(q))
    }
}

/// Fill `out` with independent standard normal draws, two per Box-Muller pair
/// of uniforms.
pub(crate) fn standard_normals(rng: &mut ChaCha8Rng, out: &mut [f64]) {
    for pair in out.chunks_mut(2) {
        let u1: f64 = 1.0 - rand::RngExt::random::<f64>(rng);
        let u2: f64 = rand::RngExt::random(rng);
        let radius = (-2.0 * u1.ln()).sqrt();
        let (sin, cos) = (std::f64::consts::TAU * u2).sin_cos();
        pair[0] = radius * cos;
        if let Some(second) = pair.get_mut(1) {
            *second = radius * sin;
        }
    }
}

fn leak(backend: &mut dyn Backend, qubit: usize, memory: &mut ShotMemory) -> Result<()> {
    backend.apply(&Instruction::Reset { qubit })?;
    apply_pauli_op(backend, qubit, PauliOp::X)?;
    memory.leaked[qubit] = true;
    memory.ever_leaked[qubit] = true;
    Ok(())
}

/// Stand in for a gate a leaked target blocks: the gate does not act, and each
/// unleaked target of a multi-qubit gate takes a uniformly random Pauli.
fn apply_blocked_gate(
    backend: &mut dyn Backend,
    targets: &[usize],
    memory: &ShotMemory,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if targets.len() < 2 {
        return Ok(());
    }
    for &qubit in targets {
        if memory.is_leaked(qubit) {
            continue;
        }
        let letter = match rand::RngExt::random::<u64>(rng) >> 62 {
            0 => PauliOp::I,
            1 => PauliOp::X,
            2 => PauliOp::Y,
            _ => PauliOp::Z,
        };
        apply_pauli_op(backend, qubit, letter)?;
    }
    Ok(())
}

fn apply_leaky_instruction(
    backend: &mut dyn Backend,
    instr: &Instruction,
    memory: &mut ShotMemory,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match instr {
        Instruction::Gate { targets, .. } if memory.any_leaked(targets) => {
            apply_blocked_gate(backend, targets, memory, rng)
        }
        Instruction::Conditional {
            condition, targets, ..
        } if memory.any_leaked(targets) => {
            if condition.evaluate(backend.classical_results()) {
                apply_blocked_gate(backend, targets, memory, rng)
            } else {
                Ok(())
            }
        }
        Instruction::Reset { qubit } => {
            backend.apply(instr)?;
            memory.leaked[*qubit] = false;
            Ok(())
        }
        _ => backend.apply(instr),
    }
}

fn apply_memory_instruction(
    backend: &mut dyn Backend,
    instr: &Instruction,
    memory: &mut ShotMemory,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if memory.leakage {
        apply_leaky_instruction(backend, instr, memory, rng)
    } else {
        backend.apply(instr)
    }
}

/// Apply one event on the memory path. A channel without memory of its own
/// skips an event naming a leaked qubit.
fn apply_memory_event(
    backend: &mut dyn Backend,
    event: &PreparedEvent,
    memory: &mut ShotMemory,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let qubit = event.qubits[0];
    match &event.channel {
        PreparedChannel::Leakage(p) => {
            if !memory.is_leaked(qubit) && rand::RngExt::random::<f64>(rng) < *p {
                leak(backend, qubit, memory)?;
            }
        }
        PreparedChannel::Seepage(p) => {
            if memory.is_leaked(qubit) && rand::RngExt::random::<f64>(rng) < *p {
                memory.leaked[qubit] = false;
                if rand::RngExt::random::<bool>(rng) {
                    apply_pauli_op(backend, qubit, PauliOp::X)?;
                }
            }
        }
        PreparedChannel::LeakageTransport(p) => {
            let other = event.qubits[1];
            if memory.is_leaked(qubit) != memory.is_leaked(other)
                && rand::RngExt::random::<f64>(rng) < *p
            {
                let target = if memory.is_leaked(qubit) {
                    other
                } else {
                    qubit
                };
                leak(backend, target, memory)?;
            }
        }
        PreparedChannel::QuasiStatic { axis, weights } => {
            if !memory.is_leaked(qubit) {
                let angle: f64 = weights
                    .iter()
                    .map(|&(source, weight)| weight * memory.drift[source])
                    .sum();
                let gate = match axis {
                    PauliAxis::X => Gate::Rx(angle),
                    PauliAxis::Y => Gate::Ry(angle),
                    PauliAxis::Z => Gate::Rz(angle),
                };
                backend.apply(&Instruction::Gate {
                    gate,
                    targets: smallvec![qubit],
                })?;
            }
        }
        _ if memory.any_leaked(&event.qubits) => {}
        _ => apply_noise_event(backend, event, rng)?,
    }
    Ok(())
}

/// [`run_trajectory_shot`] for a model with per-shot memory, also returning
/// which qubits leaked at any point of the shot (empty without leakage).
fn run_memory_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    readout: &ShotReadout,
    spec: MemorySpec,
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<bool>, Vec<bool>)> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let mut memory = ShotMemory::new(circuit.num_qubits, spec, rng);
    evolve_with_memory(backend, circuit, noise, &mut memory, &readout.at_write, rng)?;
    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, &readout.record, rng);
    Ok((results, memory.ever_leaked))
}

/// [`evolve_noisy`] from the first instruction on the memory path, reading and
/// updating the leak flags and drift `memory` holds for the rest of the shot, and
/// flipping each measured bit at its `readout` rate as the measurement writes it.
pub(crate) fn evolve_with_memory(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    memory: &mut ShotMemory,
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if readout.is_empty() && noise.num_slots() == circuit.instructions.len() {
        for (instr, events) in circuit.instructions.iter().zip(noise.slots_from(0)) {
            apply_memory_instruction(backend, instr, memory, rng)?;
            for event in events {
                apply_memory_event(backend, event, memory, rng)?;
            }
        }
        return Ok(());
    }
    let mut walk = MemoryWalk {
        backend,
        noise,
        memory,
        readout,
        rng,
    };
    walk_noise_slots(circuit, 0, &mut walk)
}

/// One trajectory on the path `memory` selects: its record, and the qubits it
/// leaked when the model carries leakage.
fn run_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    readout: &ShotReadout,
    memory: Option<MemorySpec>,
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<bool>, Vec<bool>)> {
    match memory {
        None => Ok((
            run_trajectory_shot(backend, circuit, noise, readout, rng)?,
            Vec::new(),
        )),
        Some(spec) => run_memory_shot(backend, circuit, noise, readout, spec, rng),
    }
}

/// Readout error restricted to the classical bits a measurement in `circuit`
/// writes. A bit no measurement reaches holds no outcome, so it is outside the
/// readout channel and stays an unwritten zero rather than a noisy one.
pub(crate) fn written_readout(
    circuit: &Circuit,
    readout: &[Option<ReadoutError>],
) -> Vec<Option<ReadoutError>> {
    let mut masked = vec![None; readout.len()];
    for bit in circuit.classical_bit_order() {
        if let Some(err) = readout.get(bit) {
            masked[bit] = err.clone();
        }
    }
    masked
}

/// Readout error as a trajectory applies it. A measurement flips the bit it writes,
/// so a condition reads the noisy bit and each write draws its own flip. When no
/// condition reads a bit that can flip, only each bit's last write reaches the record,
/// so the finished record is flipped instead: the same distribution, on the draws
/// those circuits have always made. One of the two tables is empty.
pub(crate) struct ShotReadout {
    /// Flipped into the finished record, from [`written_readout`].
    pub(crate) record: Vec<Option<ReadoutError>>,
    /// Flipped as each measurement writes its bit, from [`flippable_readout`].
    pub(crate) at_write: Vec<Option<ReadoutError>>,
}

impl ShotReadout {
    pub(crate) fn new(circuit: &Circuit, readout: &[Option<ReadoutError>]) -> Self {
        let at_write = flippable_readout(readout);
        let read = circuit.condition_bits();
        if at_write
            .iter()
            .zip(&read)
            .any(|(err, &read)| read && err.is_some())
        {
            return Self {
                record: Vec::new(),
                at_write,
            };
        }
        Self {
            record: written_readout(circuit, readout),
            at_write: Vec::new(),
        }
    }
}

/// `readout` with each entry that cannot flip a bit cleared, empty when none can.
pub(crate) fn flippable_readout(readout: &[Option<ReadoutError>]) -> Vec<Option<ReadoutError>> {
    if readout.iter().flatten().all(ReadoutError::is_inert) {
        return Vec::new();
    }
    readout
        .iter()
        .map(|err| err.clone().filter(|err| !err.is_inert()))
        .collect()
}

/// Flip the bit `instruction` wrote at its `readout` rate when it is a measurement.
#[inline]
pub(crate) fn flip_written_bit(
    backend: &mut dyn Backend,
    instruction: &Instruction,
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if let Instruction::Measure { classical_bit, .. } = instruction
        && let Some(Some(err)) = readout.get(*classical_bit)
    {
        backend.sync_deferred_measurements()?;
        let bit = backend.classical_results()[*classical_bit];
        let r: f64 = rand::RngExt::random(rng);
        if r < if bit { err.p10 } else { err.p01 } {
            backend.set_classical_bit(*classical_bit, !bit)?;
        }
    }
    Ok(())
}

/// `readout` is the output of [`written_readout`], so an unmeasured bit is `None`.
pub(crate) fn apply_readout_errors(
    results: &mut [bool],
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) {
    for (bit, ro) in results.iter_mut().zip(readout.iter()) {
        if let Some(err) = ro {
            let r: f64 = rand::RngExt::random(rng);
            if *bit {
                // 1→0 with probability p10
                if r < err.p10 {
                    *bit = false;
                }
            } else {
                // 0→1 with probability p01
                if r < err.p01 {
                    *bit = true;
                }
            }
        }
    }
}

/// `readout` comes from [`ShotReadout::new`] for this `circuit`.
pub(crate) fn run_trajectory_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    readout: &ShotReadout,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<bool>> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    finish_trajectory(backend, circuit, noise, 0, readout, rng)
}

/// Run a trajectory from instruction `from` on a `backend` holding its state
/// before `from`, with `rng` advanced past the events before it, and return its
/// record.
fn finish_trajectory(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    from: usize,
    readout: &ShotReadout,
    rng: &mut ChaCha8Rng,
) -> Result<Vec<bool>> {
    evolve_noisy(backend, circuit, noise, from, &readout.at_write, rng)?;
    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, &readout.record, rng);
    Ok(results)
}

/// Apply `circuit.instructions[from..]` to the state `backend` holds, firing each
/// slot of `noise` after its instruction and flipping each measured bit at its
/// `readout` rate as the measurement writes it. `noise` holds no per-shot memory
/// channel.
///
/// Measurement outcomes stay on the device between the reads that need them: a
/// conditional or region guard, a readout flip, a branch decided on the host, and the
/// end of the run.
pub(crate) fn evolve_noisy(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    from: usize,
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    backend.begin_deferred_measurements(measurement_count(&circuit.instructions[from..]))?;
    let evolved = evolve_noisy_eager::<NoiseEvents>(backend, circuit, noise, from, readout, rng);
    evolved.and(backend.end_deferred_measurements())
}

/// Measurements and resets in `instructions`, guarded bodies included.
fn measurement_count(instructions: &[Instruction]) -> usize {
    instructions
        .iter()
        .map(|instruction| match instruction {
            Instruction::Measure { .. } | Instruction::Reset { .. } => 1,
            Instruction::Region(region) => measurement_count(region.body()),
            _ => 0,
        })
        .sum()
}

/// [`evolve_noisy`] with each outcome read back as its measurement lands.
fn evolve_noisy_eager<A: EventApplier>(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    from: usize,
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if readout.is_empty() && noise.num_slots() == circuit.instructions.len() {
        for (instr, events) in circuit.instructions[from..]
            .iter()
            .zip(noise.slots_from(from))
        {
            backend.apply(instr)?;
            for event in events {
                A::apply(backend, event, rng)?;
            }
        }
        return Ok(());
    }
    let mut walk = TrajectoryWalk::<A> {
        backend,
        noise,
        readout,
        rng,
        events: PhantomData,
    };
    walk_noise_slots(circuit, from, &mut walk)
}

/// Apply `instruction` to the state `backend` holds and fire the events of
/// `noise` slot `slot` after it, on the memory path when the shot keeps `memory`.
pub(crate) fn apply_and_fire(
    backend: &mut dyn Backend,
    instruction: &Instruction,
    noise: &PreparedNoise,
    slot: usize,
    memory: Option<&mut ShotMemory>,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match memory {
        Some(memory) => {
            apply_memory_instruction(backend, instruction, memory, rng)?;
            for event in noise.slot(slot) {
                apply_memory_event(backend, event, memory, rng)?;
            }
        }
        None => {
            backend.apply(instruction)?;
            for event in noise.slot(slot) {
                apply_noise_event(backend, event, rng)?;
            }
        }
    }
    Ok(())
}

/// Trajectories split across Rayon workers when `sim::state_splits_across_workers`
/// accepts `route` at the circuit's width. Otherwise they run serially, one live
/// backend at a time, and the backend's own `init` cap is the only limit.
///
/// `force_serial` keeps every trajectory on one thread. Device-resident
/// backends set it: parallel trajectories would allocate one device state per
/// Rayon thread against a single-state VRAM verdict, and concurrent launches
/// on one device serialize anyway.
pub(crate) fn run_trajectories(
    backend_factory: impl Fn(u64) -> Box<dyn Backend> + Sync,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    force_serial: bool,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let (prepared, memory) = PreparedNoise::with_memory(noise);
    #[cfg(not(feature = "parallel"))]
    let _ = force_serial;
    #[cfg(feature = "parallel")]
    {
        if !force_serial && num_shots >= 4 && trajectories_split(route, circuit.num_qubits) {
            return run_trajectories_par(
                &backend_factory,
                circuit,
                noise,
                (&prepared, memory),
                num_shots,
                seed,
                route,
            );
        }
    }

    let readout = ShotReadout::new(circuit, &noise.readout);
    let serial = || {
        run_serial(circuit, memory, num_shots, seed, route, |shot_seed, rng| {
            let mut backend = backend_factory(shot_seed);
            let (result, leaked) =
                run_shot(backend.as_mut(), circuit, &prepared, &readout, memory, rng)?;
            Ok((
                result,
                leaked,
                crate::sim::backend_metadata(backend.as_ref()),
            ))
        })
    };
    #[cfg(feature = "parallel")]
    return crate::backend::hybrid::on_narrow_pool(
        !force_serial && crate::backend::hybrid::route_wants_narrow_pool(route, circuit.num_qubits),
        serial,
    );
    #[cfg(not(feature = "parallel"))]
    serial()
}

/// Serial trajectories on one held `backend`, which `reseed` hands each shot's seed
/// before the shot's `init` returns it to |0...0>. Records match
/// [`run_trajectories`] with `force_serial` on a factory building the same backend
/// from that seed.
#[cfg(feature = "distributed")]
pub(crate) fn run_trajectories_held<B: Backend>(
    backend: &mut B,
    mut reseed: impl FnMut(&mut B, u64),
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let (prepared, memory) = PreparedNoise::with_memory(noise);
    let readout = ShotReadout::new(circuit, &noise.readout);
    run_serial(circuit, memory, num_shots, seed, route, |shot_seed, rng| {
        reseed(backend, shot_seed);
        let (result, leaked) = run_shot(backend, circuit, &prepared, &readout, memory, rng)?;
        Ok((result, leaked, crate::sim::backend_metadata(backend)))
    })
}

/// Replay a Pauli `noise` model shot by shot on the backend `backend_factory` builds
/// from each shot's seed, drawing pair events as the frame sampler does.
pub(crate) fn run_shots_noisy_brute_with(
    backend_factory: impl Fn(u64) -> Box<dyn Backend>,
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let readout = ShotReadout::new(circuit, &noise.readout);
    let prepared = PreparedNoise::new(noise);
    let route = crate::sim::ResolvedBackend::Stabilizer;
    run_serial(circuit, None, num_shots, seed, route, |shot_seed, rng| {
        let mut backend = backend_factory(shot_seed);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        evolve_noisy_eager::<FrameEvents>(
            backend.as_mut(),
            circuit,
            &prepared,
            0,
            &readout.at_write,
            rng,
        )?;
        let mut results = backend.classical_results().to_vec();
        apply_readout_errors(&mut results, &readout.record, rng);
        Ok((
            results,
            Vec::new(),
            crate::sim::backend_metadata(backend.as_ref()),
        ))
    })
}

/// Run shot `i` for every `i` in order, `shot` taking its seed and noise stream and
/// returning its record, leak flags and metadata.
fn run_serial(
    circuit: &Circuit,
    memory: Option<MemorySpec>,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
    mut shot: impl FnMut(
        u64,
        &mut ChaCha8Rng,
    ) -> Result<(Vec<bool>, Vec<bool>, crate::sim::RunMetadata)>,
) -> Result<ShotsResult> {
    collect_trajectory_shots(
        circuit.num_classical_bits,
        memory,
        route,
        (0..num_shots).map(|i| {
            let shot_seed = crate::sim::mix_seed(seed, i);
            let mut rng = noise_rng(shot_seed);
            shot(shot_seed, &mut rng)
        }),
    )
}

/// Gather each shot's record, leak flags and metadata, in shot order, into a result
/// over `num_bits` classical bits, returning the first shot error.
pub(super) fn collect_trajectory_shots(
    num_bits: usize,
    memory: Option<MemorySpec>,
    route: crate::sim::ResolvedBackend,
    runs: impl IntoIterator<Item = Result<(Vec<bool>, Vec<bool>, crate::sim::RunMetadata)>>,
) -> Result<ShotsResult> {
    let runs = runs.into_iter();
    let mut shots = Vec::with_capacity(runs.size_hint().0);
    let mut leaked = Vec::with_capacity(runs.size_hint().0);
    let mut metadata = crate::sim::RunMetadata::exact(route);
    for (i, run) in runs.enumerate() {
        let (result, shot_leaked, shot_metadata) = run?;
        metadata.fold(i, shot_metadata);
        shots.push(result);
        leaked.push(shot_leaked);
    }
    Ok(with_leak_record(
        ShotsResult::from_shots(shots, num_bits).with_metadata(metadata),
        memory,
        leaked,
    ))
}

fn with_leak_record(
    result: ShotsResult,
    memory: Option<MemorySpec>,
    leaked: Vec<Vec<bool>>,
) -> ShotsResult {
    match memory {
        Some(spec) if spec.leakage => result.with_leaked(leaked),
        _ => result,
    }
}

/// Width below which trajectory runs on the host statevector split across workers, each
/// shot or group with its kernels serial: on an 8P+16E host, 512 shots split one task per
/// worker took 17% of the unsplit time at 16 qubits, 78% at 18 and 159% at 20.
/// Bracketed by `noisy_sampling/trajectory_pauli/non_clifford_{16,18,20}q_512`.
#[cfg(all(feature = "parallel", not(miri)))]
const SPLIT_STATEVECTOR_QUBITS: usize = 19;
#[cfg(all(feature = "parallel", miri))]
const SPLIT_STATEVECTOR_QUBITS: usize = crate::backend::PARALLEL_THRESHOLD_QUBITS;

/// Whether trajectory runs of `width` qubits on `route` split across workers: below
/// [`SPLIT_STATEVECTOR_QUBITS`] on the host statevector, and by
/// [`crate::sim::state_splits_across_workers`] on every other route.
#[cfg(feature = "parallel")]
fn trajectories_split(route: crate::sim::ResolvedBackend, width: usize) -> bool {
    match route {
        crate::sim::ResolvedBackend::Statevector => width < SPLIT_STATEVECTOR_QUBITS,
        _ => crate::sim::state_splits_across_workers(route, width),
    }
}

#[cfg(feature = "parallel")]
fn run_trajectories_par(
    backend_factory: &(impl Fn(u64) -> Box<dyn Backend> + Sync),
    circuit: &Circuit,
    noise: &NoiseModel,
    (prepared, memory): (&PreparedNoise, Option<MemorySpec>),
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let readout = ShotReadout::new(circuit, &noise.readout);
    type ParShot = (Vec<bool>, Vec<bool>, crate::sim::RunMetadata);
    let results: Result<Vec<ParShot>> = (0..num_shots)
        .into_par_iter()
        .map(|i| {
            crate::backend::with_serial_kernels(|| {
                let shot_seed = crate::sim::mix_seed(seed, i);
                let mut rng = noise_rng(shot_seed);
                let mut backend = backend_factory(shot_seed);
                let (bits, leaked) = run_shot(
                    backend.as_mut(),
                    circuit,
                    prepared,
                    &readout,
                    memory,
                    &mut rng,
                )?;
                Ok((bits, leaked, crate::sim::backend_metadata(backend.as_ref())))
            })
        })
        .collect();
    collect_trajectory_shots(
        circuit.num_classical_bits,
        memory,
        route,
        results?.into_iter().map(Ok),
    )
}

/// One sampled Pauli error: the ordinal of the event that fired, counting the
/// events before the first measurement in instruction order, and the letter it
/// applies to each of the event's qubits (`I` in the second slot for a one-qubit
/// event).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Insertion {
    event: usize,
    letters: [PauliOp; 2],
}

/// Every shot's Pauli error pattern, drawn up front and grouped so each distinct
/// pattern is simulated once.
///
/// A Pauli channel's branch weights do not depend on the state, and on a
/// circuit whose measurements are all terminal no draw depends on an outcome,
/// so the whole pattern of a shot can be drawn before anything is simulated.
/// Conditioned on its pattern a shot ends in one pure state, and its record is
/// a Born-rule draw from that state followed by readout error. Drawing the
/// pattern first and the record second is therefore the distribution of the
/// per-shot trajectory, at one simulation per pattern rather than per shot.
pub(crate) struct PauliGroups {
    /// Instructions before the first measurement, where evolution stops.
    prefix: usize,
    patterns: Vec<Vec<Insertion>>,
    /// Instruction whose events hold each group's first insertion, `prefix` for
    /// the error-free group. Every instruction before it runs noiselessly.
    starts: Vec<usize>,
    /// Instruction whose events hold each group's last insertion, `prefix` for
    /// the error-free group. Every instruction after it runs noiselessly.
    lasts: Vec<usize>,
    /// Shot indices ordered by group, ascending within each group.
    members: Vec<usize>,
    /// The model the patterns were drawn from.
    prepared: PreparedNoise,
    /// Group `g` owns `members[offsets[g]..offsets[g + 1]]`.
    offsets: Vec<usize>,
}

/// Instructions before the first measurement when every shot's error pattern can
/// be drawn up front, `None` otherwise.
///
/// Requires Pauli-frame channels only, no reset, measurements that are all
/// terminal (which rules out conditionals and regions), at least one
/// measurement, and no live event after the first measurement. A reset or a
/// damping or Kraus channel draws against the state, and a later event would
/// sit between measurements whose records it can split.
fn pauli_group_prefix(circuit: &Circuit, noise: &NoiseModel) -> Option<usize> {
    if !noise.has_only_pauli_channels()
        || circuit.has_resets()
        || !circuit.has_terminal_measurements_only()
    {
        return None;
    }
    let prefix = circuit
        .instructions
        .iter()
        .position(|inst| matches!(inst, Instruction::Measure { .. }))?;
    noise.after_gate[prefix..]
        .iter()
        .flatten()
        .all(|event| event.channel.is_inert())
        .then_some(prefix)
}

fn draw_frame_branch(event: &PreparedEvent, rng: &mut ChaCha8Rng) -> Option<[PauliOp; 2]> {
    match &event.channel {
        PreparedChannel::PauliPair { p, branch_width } => {
            let (a, b) = TWO_QUBIT_PAULIS[draw_two_qubit_depolarizing(*p, *branch_width, rng)?];
            Some([a, b])
        }
        PreparedChannel::Pauli(thresholds) => thresholds.draw(rng).map(|op| [op, PauliOp::I]),
        _ => unreachable!("grouped patterns draw Pauli-frame channels only"),
    }
}

impl PauliGroups {
    /// Draw the error pattern of each of `num_shots` shots and group them.
    ///
    /// Shot `i` draws from `noise_rng(mix_seed(seed, i))`, one uniform per event
    /// in instruction order, which is the draw sequence the per-shot trajectory
    /// makes, so a shot fires the same errors on either path. `None` when the
    /// circuit or model fails [`pauli_group_prefix`], or once distinct patterns
    /// pass nine tenths of the shots, where grouping saves too few simulations
    /// to pay for the pattern table.
    pub(crate) fn sample(
        circuit: &Circuit,
        noise: &NoiseModel,
        num_shots: usize,
        seed: u64,
    ) -> Option<Self> {
        Self::group(pauli_group_prefix(circuit, noise)?, noise, num_shots, seed)
    }

    /// [`sample`](Self::sample) for a circuit it declines, grouping the gates
    /// before the first instruction that is not a gate or barrier, `None` when
    /// there is no such gate or `sample` would accept the circuit.
    ///
    /// Every channel must be a Pauli-frame channel, so a shot's draws before
    /// that point are the ones [`sample`](Self::sample) makes.
    pub(crate) fn sample_replayed(
        circuit: &Circuit,
        noise: &NoiseModel,
        num_shots: usize,
        seed: u64,
    ) -> Option<Self> {
        if !noise.has_only_pauli_channels() || pauli_group_prefix(circuit, noise).is_some() {
            return None;
        }
        let prefix = circuit
            .instructions
            .iter()
            .position(|inst| {
                !matches!(inst, Instruction::Gate { .. } | Instruction::Barrier { .. })
            })
            .filter(|&prefix| prefix > 0)?;
        Self::group(prefix, noise, num_shots, seed)
    }

    fn group(prefix: usize, noise: &NoiseModel, num_shots: usize, seed: u64) -> Option<Self> {
        let prepared = PreparedNoise::new(noise);
        let mut index: HashMap<Vec<Insertion>, usize> = HashMap::new();
        let mut starts = Vec::new();
        let mut lasts = Vec::new();
        let mut shot_group = Vec::with_capacity(num_shots);
        let mut pattern = Vec::new();
        for shot in 0..num_shots {
            let mut rng = noise_rng(crate::sim::mix_seed(seed, shot));
            pattern.clear();
            let mut start = prefix;
            let mut last = prefix;
            let mut event = 0usize;
            for (instruction, events) in prepared.slots_from(0).take(prefix).enumerate() {
                for noise_event in events {
                    if let Some(letters) = draw_frame_branch(noise_event, &mut rng) {
                        if pattern.is_empty() {
                            start = instruction;
                        }
                        last = instruction;
                        pattern.push(Insertion { event, letters });
                    }
                    event += 1;
                }
            }
            let group = match index.get(pattern.as_slice()) {
                Some(&group) => group,
                None => {
                    let group = index.len();
                    if (group + 1) * 10 > num_shots * 9 {
                        return None;
                    }
                    index.insert(pattern.clone(), group);
                    starts.push(start);
                    lasts.push(last);
                    group
                }
            };
            shot_group.push(group);
        }

        let mut patterns = vec![Vec::new(); index.len()];
        for (pattern, group) in index {
            patterns[group] = pattern;
        }
        let mut offsets = vec![0usize; patterns.len() + 1];
        for &group in &shot_group {
            offsets[group + 1] += 1;
        }
        for group in 0..patterns.len() {
            offsets[group + 1] += offsets[group];
        }
        let mut cursor = offsets.clone();
        let mut members = vec![0usize; num_shots];
        for (shot, &group) in shot_group.iter().enumerate() {
            members[cursor[group]] = shot;
            cursor[group] += 1;
        }
        Some(Self {
            prefix,
            patterns,
            starts,
            lasts,
            members,
            offsets,
            prepared,
        })
    }

    pub(crate) fn num_groups(&self) -> usize {
        self.patterns.len()
    }

    fn members(&self, group: usize) -> &[usize] {
        &self.members[self.offsets[group]..self.offsets[group + 1]]
    }
}

/// Builds the backend a group evolves on, from the group's first member seed.
pub(crate) type GroupBackendFactory<'a> = &'a (dyn Fn(u64) -> Box<dyn Backend + Send> + Sync);

/// One group's index, member records in member order, and backend metadata.
type GroupRun = (usize, Vec<Vec<bool>>, crate::sim::RunMetadata);

/// Draw the record of each member shot of `group` from the state's
/// distribution, `weights` in basis order.
///
/// Member shot `i` draws its outcome and then its readout flips from
/// `ChaCha8Rng::seed_from_u64(mix_seed(seed, i))`, the stream a per-shot
/// trajectory measures on, apart from the stream its errors came from. The
/// outcome is an inverse-CDF draw: the members' uniforms are sorted and matched
/// against one cumulative pass over the distribution.
fn group_records(
    groups: &PauliGroups,
    group: usize,
    circuit: &Circuit,
    meas_map: &[(usize, usize)],
    readout: &[Option<ReadoutError>],
    seed: u64,
    weights: impl IntoIterator<Item = f64>,
) -> Vec<Vec<bool>> {
    let members = groups.members(group);
    let mut draws: Vec<(f64, usize)> = members
        .iter()
        .enumerate()
        .map(|(k, &shot)| {
            let mut rng = ChaCha8Rng::seed_from_u64(crate::sim::mix_seed(seed, shot));
            (rand::RngExt::random::<f64>(&mut rng), k)
        })
        .collect();
    draws.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let outcomes = inverse_cdf(&draws, weights);

    let has_readout = readout.iter().any(Option::is_some);
    members
        .iter()
        .zip(&outcomes)
        .map(|(&shot, &basis)| {
            let mut bits = vec![false; circuit.num_classical_bits];
            for &(qubit, cbit) in meas_map {
                bits[cbit] = (basis >> qubit) & 1 == 1;
            }
            if has_readout {
                let mut rng = ChaCha8Rng::seed_from_u64(crate::sim::mix_seed(seed, shot));
                let _: f64 = rand::RngExt::random(&mut rng);
                apply_readout_errors(&mut bits, readout, &mut rng);
            }
            bits
        })
        .collect()
}

/// Evolve one pattern's state on the backend `build` makes from the group's
/// first member seed, then draw its members' records from `probabilities()`.
#[allow(clippy::too_many_arguments)]
fn run_built_group(
    groups: &PauliGroups,
    group: usize,
    circuit: &Circuit,
    noise: &NoiseModel,
    meas_map: &[(usize, usize)],
    readout: &[Option<ReadoutError>],
    seed: u64,
    build: GroupBackendFactory<'_>,
) -> Result<GroupRun> {
    let mut backend = build(crate::sim::mix_seed(seed, groups.members(group)[0]));
    if groups.patterns[group].is_empty() {
        evolve_error_free(backend.as_mut(), circuit, groups.prefix)?;
    } else {
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        evolve_pattern(
            backend.as_mut(),
            groups,
            group,
            circuit,
            noise,
            0,
            groups.prefix,
            0,
        )?;
    }
    let probabilities = backend.probabilities()?;
    let records = group_records(
        groups,
        group,
        circuit,
        meas_map,
        readout,
        seed,
        probabilities,
    );
    Ok((
        group,
        records,
        crate::sim::backend_metadata(backend.as_ref()),
    ))
}

/// Run `chunk`, groups in ascending start order, on the host statevector.
///
/// A noiseless checkpoint advances once through the raw prefix. Each group with
/// errors copies it at its start and continues raw from there, which is the
/// operation sequence it would apply from |0...0⟩, up to its last insertion.
/// From the fusion width on, the noiseless rest runs as a fused stream shared by
/// every group whose last insertion sits on the same instruction. The weights
/// pass reads the amplitudes, so no `2^n` table is built. At the statevector
/// cap, where a second state would not fit, every group starts from |0...0⟩ and
/// runs raw instead.
#[allow(clippy::too_many_arguments)]
fn run_statevector_chunk(
    groups: &PauliGroups,
    chunk: &[usize],
    circuit: &Circuit,
    noise: &NoiseModel,
    meas_map: &[(usize, usize)],
    readout: &[Option<ReadoutError>],
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Vec<GroupRun>> {
    let checkpointed = circuit.num_qubits < limits.statevector_qubits.value;
    let mut checkpoint = StatevectorBackend::with_limits(seed, limits);
    if checkpointed {
        checkpoint.init(circuit.num_qubits, circuit.num_classical_bits)?;
    }
    let fused_tails =
        checkpointed && circuit.num_qubits >= crate::circuit::fusion::MIN_QUBITS_FOR_FUSION;
    let mut tails: BTreeMap<usize, Circuit> = BTreeMap::new();
    let mut backend = StatevectorBackend::with_limits(seed, limits);
    let mut at = 0usize;
    let mut ordinal = 0usize;
    let mut runs = Vec::with_capacity(chunk.len());
    for &group in chunk {
        if groups.patterns[group].is_empty() {
            evolve_error_free(&mut backend, circuit, groups.prefix)?;
        } else if !checkpointed {
            backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
            evolve_pattern(
                &mut backend,
                groups,
                group,
                circuit,
                noise,
                0,
                groups.prefix,
                0,
            )?;
        } else {
            let start = groups.starts[group];
            for (instruction, events) in circuit.instructions[at..start]
                .iter()
                .zip(&noise.after_gate[at..start])
            {
                checkpoint.apply(instruction)?;
                ordinal += events.len();
            }
            at = start;
            backend.copy_state_from(&checkpoint);
            if fused_tails {
                let tail = groups.lasts[group] + 1;
                evolve_pattern(
                    &mut backend,
                    groups,
                    group,
                    circuit,
                    noise,
                    start,
                    tail,
                    ordinal,
                )?;
                while let Some(entry) = tails.first_entry() {
                    if *entry.key() > start {
                        break;
                    }
                    entry.remove();
                }
                let fused = tails
                    .entry(tail)
                    .or_insert_with(|| fused_noiseless(&backend, circuit, tail..groups.prefix));
                backend.apply_instructions(&fused.instructions)?;
            } else {
                evolve_pattern(
                    &mut backend,
                    groups,
                    group,
                    circuit,
                    noise,
                    start,
                    groups.prefix,
                    ordinal,
                )?;
            }
        }
        let scale = backend.probability_scale();
        let weights = backend
            .state_vector()
            .iter()
            .map(|amp| amp.norm_sqr() * scale);
        let records = group_records(groups, group, circuit, meas_map, readout, seed, weights);
        runs.push((group, records, crate::sim::backend_metadata(&backend)));
    }
    Ok(runs)
}

/// Initialize `backend` and run the circuit's first `prefix` instructions through
/// the fused plan a noiseless run takes, which no error insertion splits.
fn evolve_error_free(backend: &mut dyn Backend, circuit: &Circuit, prefix: usize) -> Result<()> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let fused = fused_noiseless(backend, circuit, 0..prefix);
    backend.apply_instructions(&fused.instructions)
}

/// Instructions `span` of `circuit` as the fused plan a noiseless run of them
/// takes on `backend`.
fn fused_noiseless(
    backend: &dyn Backend,
    circuit: &Circuit,
    span: std::ops::Range<usize>,
) -> Circuit {
    let noiseless = circuit.with_instructions(circuit.instructions[span].to_vec());
    let expanded = crate::sim::expand_for_backend(backend, &noiseless);
    crate::sim::fuse_for_backend(backend, &expanded).into_owned()
}

/// Run instructions `from..to` of the circuit prefix with `group`'s Pauli errors
/// inserted after the events that fired, on a `backend` holding the state
/// before `from` and with `ordinal` events counted before it.
#[allow(clippy::too_many_arguments)]
fn evolve_pattern<B: Backend + ?Sized>(
    backend: &mut B,
    groups: &PauliGroups,
    group: usize,
    circuit: &Circuit,
    noise: &NoiseModel,
    from: usize,
    to: usize,
    mut ordinal: usize,
) -> Result<()> {
    let mut insertions = groups.patterns[group].iter().peekable();
    for (instruction, events) in circuit.instructions[from..to]
        .iter()
        .zip(&noise.after_gate[from..])
    {
        backend.apply(instruction)?;
        for event in events {
            if let Some(insertion) = insertions.next_if(|ins| ins.event == ordinal) {
                for (&qubit, &letter) in event.qubits.iter().zip(&insertion.letters) {
                    apply_pauli_op(backend, qubit, letter)?;
                }
            }
            ordinal += 1;
        }
    }
    Ok(())
}

/// Basis index each draw lands on, indexed by the draw's member slot, for
/// `draws` sorted by uniform and `weights` in basis order.
///
/// Draws past the last cumulative weight, which roundoff can leave short of
/// one, land on the last basis state with nonzero weight.
fn inverse_cdf(draws: &[(f64, usize)], weights: impl IntoIterator<Item = f64>) -> Vec<usize> {
    let mut outcomes = vec![0usize; draws.len()];
    let mut cumulative = 0.0f64;
    let mut last_nonzero = 0usize;
    let mut next = 0usize;
    for (basis, weight) in weights.into_iter().enumerate() {
        if weight == 0.0 {
            continue;
        }
        cumulative += weight;
        last_nonzero = basis;
        while next < draws.len() && draws[next].0 < cumulative {
            outcomes[draws[next].1] = basis;
            next += 1;
        }
        if next == draws.len() {
            break;
        }
    }
    for &(_, k) in &draws[next..] {
        outcomes[k] = last_nonzero;
    }
    outcomes
}

/// Split `order` into at most `parts` contiguous runs of about equal work. A
/// group with errors costs the instructions it replays past its start, and the
/// error-free group a whole prefix.
#[cfg(feature = "parallel")]
fn work_chunks<'a>(groups: &PauliGroups, order: &'a [usize], parts: usize) -> Vec<&'a [usize]> {
    balanced_chunks(order, parts, |&group| {
        if groups.patterns[group].is_empty() {
            groups.prefix
        } else {
            groups.prefix - groups.starts[group]
        }
    })
}

/// Split `items` into at most `parts` contiguous runs of about equal total `cost`.
#[cfg(feature = "parallel")]
fn balanced_chunks<T>(items: &[T], parts: usize, cost: impl Fn(&T) -> usize) -> Vec<&[T]> {
    let total: usize = items.iter().map(&cost).sum();
    let mut chunks = Vec::with_capacity(parts);
    let mut begin = 0usize;
    let mut done = 0usize;
    for (index, item) in items.iter().enumerate() {
        done += cost(item);
        if done * parts >= total * (chunks.len() + 1) {
            chunks.push(&items[begin..=index]);
            begin = index + 1;
        }
    }
    chunks
}

/// Run every shot of `groups`, one evolution per distinct error pattern, and
/// return the shots in index order.
///
/// `build` is `None` for the host statevector, where groups run in ascending
/// start order off a shared noiseless checkpoint, and otherwise builds the
/// `route` backend each group evolves on from |0...0⟩. Groups split across Rayon
/// workers under the rule the per-shot trajectories use for `route`: on the
/// host statevector as contiguous runs of the start order balanced by work, each
/// worker holding its checkpoint and one group state, and otherwise one group
/// at a time. Every draw depends only on its shot's seed, so the result does
/// not depend on the thread count.
pub(crate) fn run_pauli_groups(
    groups: &PauliGroups,
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
    build: Option<GroupBackendFactory<'_>>,
    route: crate::sim::ResolvedBackend,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    let meas_map = circuit.measurement_map();
    let readout = written_readout(circuit, &noise.readout);
    #[cfg(feature = "parallel")]
    let split = groups.num_groups() > 1 && trajectories_split(route, circuit.num_qubits);

    let Some(build) = build else {
        let mut order: Vec<usize> = (0..groups.num_groups()).collect();
        order.sort_by_key(|&group| groups.starts[group]);
        let run = |chunk: &[usize]| {
            run_statevector_chunk(
                groups, chunk, circuit, noise, &meas_map, &readout, seed, limits,
            )
        };
        #[cfg(feature = "parallel")]
        if split {
            let chunks = work_chunks(groups, &order, rayon::current_num_threads());
            let runs: Result<Vec<Vec<GroupRun>>> = chunks
                .into_par_iter()
                .map(|chunk| crate::backend::with_serial_kernels(|| run(chunk)))
                .collect();
            return collect_groups(groups, circuit, route, runs?.into_iter().flatten());
        }
        return collect_groups(groups, circuit, route, run(&order)?);
    };

    let run = |group: usize| {
        run_built_group(
            groups, group, circuit, noise, &meas_map, &readout, seed, build,
        )
    };
    #[cfg(feature = "parallel")]
    if split {
        let runs: Result<Vec<GroupRun>> =
            (0..groups.num_groups()).into_par_iter().map(run).collect();
        return collect_groups(groups, circuit, route, runs?);
    }
    let runs: Result<Vec<GroupRun>> = (0..groups.num_groups()).map(run).collect();
    collect_groups(groups, circuit, route, runs?)
}

/// A run of one group's members, `members(group)[range]`.
type Segment = (usize, std::ops::Range<usize>);

/// One shot's index, record, and backend metadata.
type ShotRun = (usize, Vec<bool>, crate::sim::RunMetadata);

/// Run every shot of `groups`, sampled by [`PauliGroups::sample_replayed`], on
/// the host statevector, and return the shots in index order.
///
/// Each distinct error pattern evolves raw off a noiseless checkpoint to the end
/// of the prefix, the operation sequence a trajectory applies from |0...0⟩. Each
/// member shot then copies that state, reseeds the backend with its shot seed,
/// redraws its prefix errors to advance its noise stream, and finishes as its
/// per-shot trajectory would, so it keeps its per-shot bits. Shots split across
/// Rayon workers under the per-shot rule for `route`, as contiguous runs of the
/// start order balanced by work; a group whose members span two runs evolves
/// once in each.
pub(crate) fn run_replayed_groups(
    groups: &PauliGroups,
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
    route: crate::sim::ResolvedBackend,
    limits: &ResolvedLimits,
) -> Result<ShotsResult> {
    let readout = ShotReadout::new(circuit, &noise.readout);
    let mut order: Vec<usize> = (0..groups.num_groups()).collect();
    order.sort_by_key(|&group| groups.starts[group]);
    let run = |segments: Vec<Segment>| {
        run_replayed_chunk(groups, &segments, circuit, noise, &readout, seed, limits)
    };
    let num_shots = groups.members.len();
    #[cfg(feature = "parallel")]
    if num_shots >= 4 && trajectories_split(route, circuit.num_qubits) {
        let suffix = circuit.instructions.len() - groups.prefix;
        let chunks = replay_chunks(groups, &order, rayon::current_num_threads(), suffix);
        let runs: Result<Vec<Vec<ShotRun>>> = chunks
            .into_par_iter()
            .map(|chunk| crate::backend::with_serial_kernels(|| run(chunk)))
            .collect();
        return fold_replayed(circuit, route, num_shots, runs?.into_iter().flatten());
    }
    let whole = order
        .iter()
        .map(|&group| (group, 0..groups.members(group).len()))
        .collect();
    fold_replayed(circuit, route, num_shots, run(whole)?)
}

/// Split the member shots of `order` into at most `parts` contiguous runs of
/// about equal work: a group's evolution past its start, then `suffix`
/// instructions per member.
#[cfg(feature = "parallel")]
fn replay_chunks(
    groups: &PauliGroups,
    order: &[usize],
    parts: usize,
    suffix: usize,
) -> Vec<Vec<Segment>> {
    let total: usize = order
        .iter()
        .map(|&group| groups.prefix - groups.starts[group] + groups.members(group).len() * suffix)
        .sum();
    let mut chunks = Vec::with_capacity(parts);
    let mut current = Vec::new();
    let mut done = 0usize;
    for &group in order {
        done += groups.prefix - groups.starts[group];
        let count = groups.members(group).len();
        let mut first = 0usize;
        for member in 0..count {
            done += suffix;
            if done * parts >= total * (chunks.len() + 1) {
                current.push((group, first..member + 1));
                chunks.push(std::mem::take(&mut current));
                first = member + 1;
            }
        }
        if first < count {
            current.push((group, first..count));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Run `segments`, in ascending start order, off one noiseless checkpoint.
fn run_replayed_chunk(
    groups: &PauliGroups,
    segments: &[Segment],
    circuit: &Circuit,
    noise: &NoiseModel,
    readout: &ShotReadout,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Vec<ShotRun>> {
    let mut checkpoint = StatevectorBackend::with_limits(seed, limits);
    checkpoint.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let mut state = StatevectorBackend::with_limits(seed, limits);
    let mut backend = StatevectorBackend::with_limits(seed, limits);
    let mut at = 0usize;
    let mut ordinal = 0usize;
    let mut runs = Vec::new();
    for (group, range) in segments {
        let start = groups.starts[*group];
        for (instruction, events) in circuit.instructions[at..start]
            .iter()
            .zip(&noise.after_gate[at..start])
        {
            checkpoint.apply(instruction)?;
            ordinal += events.len();
        }
        at = start;
        state.copy_state_from(&checkpoint);
        evolve_pattern(
            &mut state,
            groups,
            *group,
            circuit,
            noise,
            start,
            groups.prefix,
            ordinal,
        )?;
        for &shot in &groups.members(*group)[range.clone()] {
            let shot_seed = crate::sim::mix_seed(seed, shot);
            backend.copy_state_from(&state);
            backend.rng = ChaCha8Rng::seed_from_u64(shot_seed);
            let mut rng = noise_rng(shot_seed);
            for event in groups.prepared.events_before(groups.prefix) {
                draw_frame_branch(event, &mut rng);
            }
            let bits = finish_trajectory(
                &mut backend,
                circuit,
                &groups.prepared,
                groups.prefix,
                readout,
                &mut rng,
            )?;
            runs.push((shot, bits, crate::sim::backend_metadata(&backend)));
        }
    }
    Ok(runs)
}

fn fold_replayed(
    circuit: &Circuit,
    route: crate::sim::ResolvedBackend,
    num_shots: usize,
    runs: impl IntoIterator<Item = ShotRun>,
) -> Result<ShotsResult> {
    let mut slots = vec![None; num_shots];
    for (shot, bits, metadata) in runs {
        slots[shot] = Some((bits, metadata));
    }
    crate::sim::fold_shots(
        circuit,
        route,
        slots
            .into_iter()
            .map(|slot| Ok(slot.expect("every shot runs in one segment"))),
    )
}

fn collect_groups(
    groups: &PauliGroups,
    circuit: &Circuit,
    route: crate::sim::ResolvedBackend,
    runs: impl IntoIterator<Item = GroupRun>,
) -> Result<ShotsResult> {
    let mut shots = vec![Vec::new(); groups.members.len()];
    let mut metadata = crate::sim::RunMetadata::exact(route);
    for (index, (group, records, group_metadata)) in runs.into_iter().enumerate() {
        metadata.fold(index, group_metadata);
        for (&shot, record) in groups.members(group).iter().zip(records) {
            shots[shot] = record;
        }
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

/// Headroom over one in the `P(1)` a no-jump pass reads, which rounding can lift past
/// one. A damping draw at or above `gamma * (1 + P1_HEADROOM)` takes the no-jump pass
/// for any such `P(1)`.
const P1_HEADROOM: f64 = 1e-6;

/// What every shot still on a [`SharedPrefix`] does at one of its events.
#[derive(Clone, Copy)]
enum SharedEvent {
    /// Draws nothing and leaves the state alone.
    Silent,
    /// A Pauli channel, the identity on a draw at or above `bound`.
    Pauli { bound: f64 },
    /// Damping on `qubit`, the no-jump pass on a draw at or above `bound`.
    Damping {
        qubit: usize,
        damping: Damping,
        bound: f64,
    },
}

impl SharedEvent {
    fn of(event: &PreparedEvent) -> Option<Self> {
        Some(match &event.channel {
            PreparedChannel::Identity => Self::Silent,
            PreparedChannel::Pauli(thresholds) => Self::Pauli {
                bound: thresholds.xyz,
            },
            PreparedChannel::PauliPair { p, .. } => Self::Pauli { bound: *p },
            PreparedChannel::AmplitudeDamping(damping) | PreparedChannel::PhaseDamping(damping) => {
                Self::Damping {
                    qubit: event.qubits[0],
                    damping: *damping,
                    bound: damping.gamma * (1.0 + P1_HEADROOM),
                }
            }
            PreparedChannel::Thermal(branches)
                if branches.excited == 0.0 && branches.cold_keep > 0.0 =>
            {
                let gamma = branches.relax + branches.dephase;
                Self::Damping {
                    qubit: event.qubits[0],
                    damping: Damping {
                        gamma,
                        keep: branches.cold_keep,
                    },
                    bound: gamma * (1.0 + P1_HEADROOM),
                }
            }
            _ => return None,
        })
    }

    /// Draw the event's uniform, the one its channel draws, and report whether the shot
    /// leaves the shared state on it.
    #[inline]
    fn leaves(self, rng: &mut ChaCha8Rng) -> bool {
        match self {
            Self::Silent => false,
            Self::Pauli { bound } | Self::Damping { bound, .. } => {
                rand::RngExt::random::<f64>(rng) < bound
            }
        }
    }
}

/// The gates a damping run's shots share on the host statevector until each draws its
/// first branch.
///
/// Up to the first instruction that is not a gate or barrier, or whose slot holds a
/// channel other than Pauli, depolarizing, damping, or cold thermal relaxation,
/// a shot whose draws stay at or above each event's bound applies what every other such
/// shot applies: a Pauli channel's identity and the no-jump damping pass. Which event a
/// shot leaves on reads its noise stream alone, and the walk checks each `P(1)` it reads
/// against [`P1_HEADROOM`].
struct SharedPrefix<'a> {
    circuit: &'a Circuit,
    noise: &'a PreparedNoise,
    readout: &'a ShotReadout,
    seed: u64,
    limits: &'a ResolvedLimits,
    /// Instructions shared.
    len: usize,
    /// Every event of slots `..len`, in firing order.
    events: Vec<SharedEvent>,
}

/// The state of a [`SharedPrefix`] with instructions `..at` applied and the shared
/// branch of every event before `ordinal` taken.
struct SharedWalk {
    state: StatevectorBackend,
    at: usize,
    ordinal: usize,
}

impl SharedPrefix<'_> {
    fn start(&self) -> Result<SharedWalk> {
        let mut state = StatevectorBackend::with_limits(self.seed, self.limits);
        state.init(self.circuit.num_qubits, self.circuit.num_classical_bits)?;
        Ok(SharedWalk {
            state,
            at: 0,
            ordinal: 0,
        })
    }

    /// Advance `walk` until event `target` fires next, or to the end of the prefix for
    /// `events.len()`. `false` once a no-jump pass reads `P(1)` past its headroom.
    fn advance(&self, walk: &mut SharedWalk, target: usize) -> Result<bool> {
        let offsets = &self.noise.offsets;
        while walk.at < self.len && offsets[walk.at] <= target {
            if !self.fire(walk, offsets[walk.at]) {
                return Ok(false);
            }
            walk.state.apply(&self.circuit.instructions[walk.at])?;
            walk.at += 1;
        }
        Ok(self.fire(walk, target))
    }

    fn fire(&self, walk: &mut SharedWalk, to: usize) -> bool {
        use crate::backend::sealed::Sealed;
        for event in &self.events[walk.ordinal..to] {
            if let SharedEvent::Damping {
                qubit,
                damping,
                bound,
            } = *event
            {
                match walk.state.damp_no_jump(qubit, damping.gamma, damping.keep) {
                    Some(p1) if damping.gamma * p1 <= bound => {}
                    _ => return false,
                }
            }
        }
        walk.ordinal = to;
        true
    }

    /// Run `shot` on `backend` from where `walk` stands: its state, the shot's own
    /// measurement seed, and its noise stream redrawn past the events before
    /// `walk.ordinal`, then the rest of the slot and the circuit as its trajectory runs
    /// them.
    fn resume(
        &self,
        walk: &SharedWalk,
        backend: &mut StatevectorBackend,
        shot: usize,
    ) -> Result<ShotRun> {
        let shot_seed = crate::sim::mix_seed(self.seed, shot);
        backend.copy_state_from(&walk.state);
        backend.rng = ChaCha8Rng::seed_from_u64(shot_seed);
        let mut rng = noise_rng(shot_seed);
        for event in &self.events[..walk.ordinal] {
            event.leaves(&mut rng);
        }
        for event in &self.noise.events[walk.ordinal..self.noise.offsets[walk.at]] {
            apply_noise_event(backend, event, &mut rng)?;
        }
        let bits = finish_trajectory(
            backend,
            self.circuit,
            self.noise,
            walk.at,
            self.readout,
            &mut rng,
        )?;
        Ok((shot, bits, crate::sim::backend_metadata(backend)))
    }

    /// Resume each of `forks`, `(event, shot)` pairs in event order, where it leaves.
    fn run_forks(
        &self,
        walk: &mut SharedWalk,
        backend: &mut StatevectorBackend,
        forks: &[(usize, usize)],
        runs: &mut Vec<ShotRun>,
    ) -> Result<bool> {
        for &(ordinal, shot) in forks {
            if !self.advance(walk, ordinal)? {
                return Ok(false);
            }
            runs.push(self.resume(walk, backend, shot)?);
        }
        Ok(true)
    }

    /// Walk to the end of the prefix and resume each of `terminal` there, in at most
    /// `parts` runs that each hold a copy of the walked state.
    #[cfg(feature = "parallel")]
    fn run_terminal_split(&self, terminal: &[usize], parts: usize) -> Result<Option<Vec<ShotRun>>> {
        if terminal.is_empty() {
            return Ok(Some(Vec::new()));
        }
        let mut walk = self.start()?;
        if !self.advance(&mut walk, self.events.len())? {
            return Ok(None);
        }
        let runs: Vec<&[usize]> = terminal.chunks(terminal.len().div_ceil(parts)).collect();
        let mut walks: Vec<SharedWalk> = (1..runs.len())
            .map(|_| {
                let mut state = StatevectorBackend::with_limits(self.seed, self.limits);
                state.copy_state_from(&walk.state);
                SharedWalk {
                    state,
                    at: walk.at,
                    ordinal: walk.ordinal,
                }
            })
            .collect();
        walks.push(walk);
        let resumed = walks
            .into_par_iter()
            .zip(runs)
            .map(|(walk, shots)| {
                let mut backend = StatevectorBackend::with_limits(self.seed, self.limits);
                shots
                    .iter()
                    .map(|&shot| self.resume(&walk, &mut backend, shot))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(resumed.into_iter().flatten().collect()))
    }

    /// Instructions a shot leaving on event `ordinal` runs on its own.
    #[cfg(feature = "parallel")]
    fn fork_cost(&self, ordinal: usize) -> usize {
        self.circuit.instructions.len() + 1
            - self
                .noise
                .offsets
                .partition_point(|&offset| offset <= ordinal)
    }
}

/// Run `num_shots` trajectories of a damping model on the host statevector, with the
/// records [`run_trajectories`] draws, by walking their shared no-jump state once.
/// `None` when no damping event falls in the [`SharedPrefix`], or once the walk reads a
/// `P(1)` past its headroom.
///
/// Each shot copies the walked state at the event it leaves on, or at the end of the
/// prefix, reseeds the backend with its shot seed, redraws its noise stream that far,
/// and finishes as its per-shot trajectory would, so it keeps its per-shot bits. Shots
/// split across Rayon workers under the per-shot rule for `route`: those that leave
/// inside the prefix as contiguous runs of their leave order balanced by work, each run
/// walking a state as far as its last shot, and those that reach the end off one walk.
pub(crate) fn run_shared_prefix(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
    limits: &ResolvedLimits,
) -> Result<Option<ShotsResult>> {
    if noise.has_leakage() || noise.has_quasi_static() {
        return Ok(None);
    }
    let prepared = PreparedNoise::new(noise);
    let len = circuit
        .instructions
        .iter()
        .zip(prepared.slots_from(0))
        .position(|(instruction, slot)| {
            !matches!(
                instruction,
                Instruction::Gate { .. } | Instruction::Barrier { .. }
            ) || slot.iter().any(|event| SharedEvent::of(event).is_none())
        })
        .unwrap_or(circuit.instructions.len());
    let events: Vec<SharedEvent> = prepared
        .events_before(len)
        .iter()
        .filter_map(SharedEvent::of)
        .collect();
    if !events
        .iter()
        .any(|event| matches!(event, SharedEvent::Damping { .. }))
    {
        return Ok(None);
    }

    #[cfg(feature = "parallel")]
    let split =
        num_shots >= 4 && crate::sim::state_splits_across_workers(route, circuit.num_qubits);
    let leaves_on = |shot: usize| {
        let mut rng = noise_rng(crate::sim::mix_seed(seed, shot));
        events.iter().position(|event| event.leaves(&mut rng))
    };
    #[cfg(feature = "parallel")]
    let left: Vec<Option<usize>> = if split {
        (0..num_shots).into_par_iter().map(leaves_on).collect()
    } else {
        (0..num_shots).map(leaves_on).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let left: Vec<Option<usize>> = (0..num_shots).map(leaves_on).collect();
    let mut forks = Vec::new();
    let mut terminal = Vec::new();
    for (shot, ordinal) in left.into_iter().enumerate() {
        match ordinal {
            Some(ordinal) => forks.push((ordinal, shot)),
            None => terminal.push(shot),
        }
    }
    forks.sort_by_key(|&(ordinal, _)| ordinal);

    let readout = ShotReadout::new(circuit, &noise.readout);
    let prefix = SharedPrefix {
        circuit,
        noise: &prepared,
        readout: &readout,
        seed,
        limits,
        len,
        events,
    };
    let end = prefix.events.len();
    let shot_backend = || StatevectorBackend::with_limits(seed, limits);

    #[cfg(feature = "parallel")]
    if split {
        let chunks = balanced_chunks(&forks, rayon::current_num_threads(), |&(ordinal, _)| {
            prefix.fork_cost(ordinal)
        });
        let (forked, finished) = rayon::join(
            || {
                chunks
                    .into_par_iter()
                    .map(|chunk| {
                        let mut walk = prefix.start()?;
                        let mut runs = Vec::with_capacity(chunk.len());
                        let shared =
                            prefix.run_forks(&mut walk, &mut shot_backend(), chunk, &mut runs)?;
                        Ok(shared.then_some(runs))
                    })
                    .collect::<Result<Vec<_>>>()
            },
            || prefix.run_terminal_split(&terminal, rayon::current_num_threads()),
        );
        let mut runs = Vec::with_capacity(num_shots);
        for chunk in forked? {
            let Some(chunk) = chunk else {
                return Ok(None);
            };
            runs.extend(chunk);
        }
        let Some(finished) = finished? else {
            return Ok(None);
        };
        runs.extend(finished);
        return fold_replayed(circuit, route, num_shots, runs).map(Some);
    }

    let mut walk = prefix.start()?;
    let mut backend = shot_backend();
    let mut runs = Vec::with_capacity(num_shots);
    if !prefix.run_forks(&mut walk, &mut backend, &forks, &mut runs)? {
        return Ok(None);
    }
    if !terminal.is_empty() && !prefix.advance(&mut walk, end)? {
        return Ok(None);
    }
    for &shot in &terminal {
        runs.push(prefix.resume(&walk, &mut backend, shot)?);
    }
    fold_replayed(circuit, route, num_shots, runs).map(Some)
}

#[cfg(test)]
#[path = "trajectory_tests.rs"]
mod tests;
