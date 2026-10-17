//! Monte Carlo trajectory execution for noisy circuits: one pure-state
//! simulation per shot, sampling a noise branch after each instruction, or one
//! per distinct error pattern when every shot's Pauli errors can be drawn first.

use std::collections::{BTreeMap, HashMap};

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::limits::ResolvedLimits;
use crate::sim::ShotsResult;
use crate::sim::noise::{
    NoiseChannel, NoiseEvent, NoiseModel, ReadoutError, SlotWalk, walk_noise_slots,
};
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
        let mut events = Vec::with_capacity(noise.after_gate.iter().map(Vec::len).sum());
        let mut offsets = Vec::with_capacity(noise.after_gate.len() + 1);
        offsets.push(0);
        for slot in &noise.after_gate {
            events.extend(slot.iter().map(PreparedEvent::new));
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
    fn new(event: &NoiseEvent) -> Self {
        let q0 = event.qubits[0];
        Self {
            channel: PreparedChannel::new(&event.channel),
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
    fn new(channel: &NoiseChannel) -> Self {
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
            NoiseChannel::Custom { kraus } => Self::Custom(
                kraus
                    .iter()
                    .map(|op| Kraus1q {
                        op: *op,
                        effect: kdagger_k(op),
                    })
                    .collect(),
            ),
            NoiseChannel::Kraus2q { kraus } => Self::Kraus2q(
                kraus
                    .iter()
                    .map(|op| Kraus2q {
                        op: *op,
                        effect: kdagger_k_2q(op),
                    })
                    .collect(),
            ),
            NoiseChannel::Leakage { p } => Self::Leakage(*p),
            NoiseChannel::Seepage { p } => Self::Seepage(*p),
            NoiseChannel::LeakageTransport { p } => Self::LeakageTransport(*p),
            NoiseChannel::QuasiStatic { axis, weights } => Self::QuasiStatic {
                axis: *axis,
                weights: weights.as_slice().into(),
            },
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
pub(crate) struct ThermalBranches {
    relax: f64,
    dephase: f64,
    excite: f64,
    excited: f64,
    decay_keep: f64,
    dephase_keep: f64,
    hot: [f64; 2],
    cold_scale: f64,
    cold_keep: f64,
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
pub(crate) struct Kraus1q {
    op: [[Complex64; 2]; 2],
    effect: [[Complex64; 2]; 2],
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
const JUMP_EPSILON: f64 = 1e-12;

/// Apply a single-qubit diagonal Kraus channel with 2 operators of the form:
///   K_0 = diag(1, c), no-jump
///   K_1 = [[0, s01_upper], [0, s11_lower]], jump (AD)
///       or diag(0, s), jump (PD, pass s01_upper=0)
///
/// where c = sqrt(1-gamma). The effective jump probability is gamma * p1.
fn apply_diagonal_kraus_2op(
    backend: &mut dyn Backend,
    qubit: usize,
    Damping { gamma, keep }: Damping,
    jump_moves_population: bool,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let p1 = backend.qubit_probability(qubit)?;
    let p_jump = gamma * p1;
    let r: f64 = rand::RngExt::random(rng);

    let zero = Complex64::new(0.0, 0.0);

    if r < p_jump && p_jump > JUMP_EPSILON {
        let s = (gamma / p_jump).sqrt();
        let mat = if jump_moves_population {
            [[zero, Complex64::new(s, 0.0)], [zero, zero]]
        } else {
            [[zero, zero], [zero, Complex64::new(s, 0.0)]]
        };
        backend.apply_1q_matrix(qubit, &mat)?;
    } else {
        let denom = 1.0 - p_jump;
        if denom <= JUMP_EPSILON {
            return Ok(());
        }
        let inv = 1.0 / denom.sqrt();
        let mat = [
            [Complex64::new(inv, 0.0), zero],
            [zero, Complex64::new(keep * inv, 0.0)],
        ];
        backend.apply_1q_matrix(qubit, &mat)?;
    }
    Ok(())
}

fn apply_amplitude_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, damping, true, rng)
}

fn apply_phase_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    damping: Damping,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, damping, false, rng)
}

/// A trajectory over a model with guarded slots.
struct TrajectoryWalk<'a> {
    backend: &'a mut dyn Backend,
    noise: &'a PreparedNoise,
    rng: &'a mut ChaCha8Rng,
}

impl SlotWalk for TrajectoryWalk<'_> {
    fn classical(&self) -> &[bool] {
        self.backend.classical_results()
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        self.backend.apply(instruction)
    }

    fn fire(&mut self, slot: usize) -> Result<()> {
        for event in self.noise.slot(slot) {
            apply_noise_event(self.backend, event, self.rng)?;
        }
        Ok(())
    }
}

/// [`TrajectoryWalk`] on the per-shot memory path.
struct MemoryWalk<'a> {
    backend: &'a mut dyn Backend,
    noise: &'a PreparedNoise,
    memory: &'a mut ShotMemory,
    leakage: bool,
    rng: &'a mut ChaCha8Rng,
}

impl SlotWalk for MemoryWalk<'_> {
    fn classical(&self) -> &[bool] {
        self.backend.classical_results()
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        if self.leakage {
            apply_leaky_instruction(self.backend, instruction, self.memory, self.rng)
        } else {
            self.backend.apply(instruction)
        }
    }

    fn fire(&mut self, slot: usize) -> Result<()> {
        for event in self.noise.slot(slot) {
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
    let p1 = backend.qubit_probability(qubit)?;
    let p0 = 1.0 - p1;

    let p_relax = branches.relax * p1;
    let p_dephase = branches.dephase * p1;
    let p_excite = branches.excite * p0;
    let p_hold_hot = branches.excited * (branches.decay_keep * p0 + branches.dephase_keep * p1);

    let zero = Complex64::new(0.0, 0.0);
    let r: f64 = rand::RngExt::random(rng);
    let mat = if r < p_relax {
        let s = (1.0 / p1).sqrt();
        [[zero, Complex64::new(s, 0.0)], [zero, zero]]
    } else if r < p_relax + p_dephase {
        let s = (1.0 / p1).sqrt();
        [[zero, zero], [zero, Complex64::new(s, 0.0)]]
    } else if r < p_relax + p_dephase + p_excite {
        let s = (1.0 / p0).sqrt();
        [[zero, zero], [Complex64::new(s, 0.0), zero]]
    } else if r < p_relax + p_dephase + p_excite + p_hold_hot {
        if p_hold_hot <= JUMP_EPSILON {
            return Ok(());
        }
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

/// Compute the branch effect `Kdagger K` for a single-qubit Kraus operator.
#[inline]
fn kdagger_k(k: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            k[0][0].conj() * k[0][0] + k[1][0].conj() * k[1][0],
            k[0][0].conj() * k[0][1] + k[1][0].conj() * k[1][1],
        ],
        [
            k[0][1].conj() * k[0][0] + k[1][1].conj() * k[1][0],
            k[0][1].conj() * k[0][1] + k[1][1].conj() * k[1][1],
        ],
    ]
}

#[inline]
fn effect_probability(effect: &[[Complex64; 2]; 2], rho: &[[Complex64; 2]; 2]) -> f64 {
    let p = effect[0][0] * rho[0][0]
        + effect[0][1] * rho[1][0]
        + effect[1][0] * rho[0][1]
        + effect[1][1] * rho[1][1];
    p.re.max(0.0)
}

/// Sample and apply one of a set of 1-qubit Kraus operators.
///
/// Branch probabilities use `p_k = Tr(Kdagger K rho_q)`, where `rho_q` is
/// the qubit's reduced density matrix. This handles dense Kraus operators
/// whose branch probabilities depend on coherence.
fn apply_custom_kraus(
    backend: &mut dyn Backend,
    qubit: usize,
    kraus: &[Kraus1q],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let rho = backend.reduced_density_matrix_1q(qubit)?;

    let mut cumulative: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        let pk = effect_probability(&k.effect, &rho);
        total += pk;
        cumulative.push(total);
    }

    if total <= JUMP_EPSILON {
        return Ok(());
    }

    let r: f64 = rand::RngExt::random::<f64>(rng) * total;
    let chosen = cumulative
        .iter()
        .position(|&c| r < c)
        .unwrap_or(kraus.len() - 1);
    let pk = if chosen == 0 {
        cumulative[0]
    } else {
        cumulative[chosen] - cumulative[chosen - 1]
    };
    if pk <= JUMP_EPSILON {
        return Ok(());
    }
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

    let mut cumulative: smallvec::SmallVec<[f64; 16]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        total += effect_probability_2q(&k.effect, &rho);
        cumulative.push(total);
    }

    if total <= JUMP_EPSILON {
        return Ok(());
    }

    let r: f64 = rand::RngExt::random::<f64>(rng) * total;
    let chosen = cumulative
        .iter()
        .position(|&c| r < c)
        .unwrap_or(kraus.len() - 1);
    let pk = if chosen == 0 {
        cumulative[0]
    } else {
        cumulative[chosen] - cumulative[chosen - 1]
    };
    if pk <= JUMP_EPSILON {
        return Ok(());
    }

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

/// `Kdagger K`, entry `[i][j] = sum_r conj(K[r][i]) K[r][j]`.
fn kdagger_k_2q(k: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut effect = [[Complex64::new(0.0, 0.0); 4]; 4];
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
fn effect_probability_2q(effect: &[[Complex64; 4]; 4], rho: &[[Complex64; 4]; 4]) -> f64 {
    let mut p = Complex64::new(0.0, 0.0);
    for i in 0..4 {
        for j in 0..4 {
            p += effect[i][j] * rho[j][i];
        }
    }
    p.re.max(0.0)
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
        PreparedChannel::Custom(kraus) => apply_custom_kraus(backend, qubit, kraus, rng),
        PreparedChannel::Kraus2q(kraus) => apply_custom_kraus_2q(backend, event.qubits, kraus, rng),
        PreparedChannel::Leakage(_)
        | PreparedChannel::Seepage(_)
        | PreparedChannel::LeakageTransport(_)
        | PreparedChannel::QuasiStatic { .. } => {
            unreachable!("channels with per-shot memory run through apply_memory_event")
        }
    }
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
    pub(crate) fn of(noise: &NoiseModel) -> Option<Self> {
        let spec = Self {
            leakage: noise.has_leakage(),
            sources: noise.quasi_static_sources(),
        };
        (spec.leakage || spec.sources > 0).then_some(spec)
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
struct ShotMemory {
    leaked: Vec<bool>,
    ever_leaked: Vec<bool>,
    drift: Vec<f64>,
}

impl ShotMemory {
    fn new(num_qubits: usize, spec: MemorySpec, rng: &mut ChaCha8Rng) -> Self {
        let mut drift = vec![0.0; spec.sources];
        standard_normals(rng, &mut drift);
        let flags = if spec.leakage { num_qubits } else { 0 };
        Self {
            leaked: vec![false; flags],
            ever_leaked: vec![false; flags],
            drift,
        }
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
    readout: &[Option<ReadoutError>],
    spec: MemorySpec,
    rng: &mut ChaCha8Rng,
) -> Result<(Vec<bool>, Vec<bool>)> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let mut memory = ShotMemory::new(circuit.num_qubits, spec, rng);
    if noise.num_slots() == circuit.instructions.len() {
        for (instr, events) in circuit.instructions.iter().zip(noise.slots_from(0)) {
            if spec.leakage {
                apply_leaky_instruction(backend, instr, &mut memory, rng)?;
            } else {
                backend.apply(instr)?;
            }
            for event in events {
                apply_memory_event(backend, event, &mut memory, rng)?;
            }
        }
    } else {
        let mut walk = MemoryWalk {
            backend: &mut *backend,
            noise,
            memory: &mut memory,
            leakage: spec.leakage,
            rng: &mut *rng,
        };
        walk_noise_slots(circuit, 0, &mut walk)?;
    }
    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, readout, rng);
    Ok((results, memory.ever_leaked))
}

/// One trajectory on the path `memory` selects: its record, and the qubits it
/// leaked when the model carries leakage.
fn run_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    readout: &[Option<ReadoutError>],
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

/// `readout` comes from [`written_readout`] for this `circuit`.
pub(crate) fn run_trajectory_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &PreparedNoise,
    readout: &[Option<ReadoutError>],
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
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<bool>> {
    if noise.num_slots() == circuit.instructions.len() {
        for (instr, events) in circuit.instructions[from..]
            .iter()
            .zip(noise.slots_from(from))
        {
            backend.apply(instr)?;
            for event in events {
                apply_noise_event(backend, event, rng)?;
            }
        }
    } else {
        let mut walk = TrajectoryWalk {
            backend: &mut *backend,
            noise,
            rng: &mut *rng,
        };
        walk_noise_slots(circuit, from, &mut walk)?;
    }

    let mut results = backend.classical_results().to_vec();
    apply_readout_errors(&mut results, readout, rng);
    Ok(results)
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
    let prepared = PreparedNoise::new(noise);
    #[cfg(not(feature = "parallel"))]
    let _ = force_serial;
    #[cfg(feature = "parallel")]
    {
        if !force_serial
            && num_shots >= 4
            && crate::sim::state_splits_across_workers(route, circuit.num_qubits)
        {
            return run_trajectories_par(
                &backend_factory,
                circuit,
                noise,
                &prepared,
                num_shots,
                seed,
                route,
            );
        }
    }

    let readout = written_readout(circuit, &noise.readout);
    let memory = MemorySpec::of(noise);
    let mut shots = Vec::with_capacity(num_shots);
    let mut leaked = Vec::new();
    let mut metadata = crate::sim::RunMetadata::exact(route);
    for i in 0..num_shots {
        let shot_seed = crate::sim::mix_seed(seed, i);
        let mut rng = noise_rng(shot_seed);
        let mut backend = backend_factory(shot_seed);
        let (result, shot_leaked) = run_shot(
            backend.as_mut(),
            circuit,
            &prepared,
            &readout,
            memory,
            &mut rng,
        )?;
        let shot_metadata = crate::sim::backend_metadata(backend.as_ref());
        if i == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(result);
        leaked.push(shot_leaked);
    }

    Ok(with_leak_record(
        ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata),
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

#[cfg(feature = "parallel")]
fn run_trajectories_par(
    backend_factory: &(impl Fn(u64) -> Box<dyn Backend> + Sync),
    circuit: &Circuit,
    noise: &NoiseModel,
    prepared: &PreparedNoise,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let readout = written_readout(circuit, &noise.readout);
    let memory = MemorySpec::of(noise);
    type ParShot = (Vec<bool>, Vec<bool>, crate::sim::RunMetadata);
    let results: Result<Vec<ParShot>> = (0..num_shots)
        .into_par_iter()
        .map(|i| {
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
        .collect();

    let mut metadata = crate::sim::RunMetadata::exact(route);
    let mut shots = Vec::with_capacity(num_shots);
    let mut leaked = Vec::with_capacity(num_shots);
    for (index, (bits, shot_leaked, shot_metadata)) in results?.into_iter().enumerate() {
        if index == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(bits);
        leaked.push(shot_leaked);
    }
    Ok(with_leak_record(
        ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata),
        memory,
        leaked,
    ))
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
    let cost = |group: usize| {
        if groups.patterns[group].is_empty() {
            groups.prefix
        } else {
            groups.prefix - groups.starts[group]
        }
    };
    let total: usize = order.iter().map(|&group| cost(group)).sum();
    let mut chunks = Vec::with_capacity(parts);
    let mut begin = 0usize;
    let mut done = 0usize;
    for (index, &group) in order.iter().enumerate() {
        done += cost(group);
        if done * parts >= total * (chunks.len() + 1) {
            chunks.push(&order[begin..=index]);
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
    let split = groups.num_groups() > 1
        && crate::sim::state_splits_across_workers(route, circuit.num_qubits);

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
            let runs: Result<Vec<Vec<GroupRun>>> = chunks.into_par_iter().map(run).collect();
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
    let readout = written_readout(circuit, &noise.readout);
    let mut order: Vec<usize> = (0..groups.num_groups()).collect();
    order.sort_by_key(|&group| groups.starts[group]);
    let run = |segments: Vec<Segment>| {
        run_replayed_chunk(groups, &segments, circuit, noise, &readout, seed, limits)
    };
    let num_shots = groups.members.len();
    #[cfg(feature = "parallel")]
    if num_shots >= 4 && crate::sim::state_splits_across_workers(route, circuit.num_qubits) {
        let suffix = circuit.instructions.len() - groups.prefix;
        let chunks = replay_chunks(groups, &order, rayon::current_num_threads(), suffix);
        let runs: Result<Vec<Vec<ShotRun>>> = chunks.into_par_iter().map(run).collect();
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
    readout: &[Option<ReadoutError>],
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
        if index == 0 {
            metadata = group_metadata;
        } else {
            metadata.weaken_with(&group_metadata);
        }
        for (&shot, record) in groups.members(group).iter().zip(records) {
            shots[shot] = record;
        }
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

#[cfg(test)]
#[path = "trajectory_tests.rs"]
mod tests;
