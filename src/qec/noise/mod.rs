//! Pauli-noise machinery for the compiled QEC runner: deferred-measurement
//! lowering, the Pauli frame pass that flips sampled records, the shared noise draw,
//! and the density-matrix lowering for noisy `EXP_VAL` estimation.

mod feedforward;
mod lower;
mod parity;
mod record;

pub(in crate::qec) use feedforward::QecFeedforwardFlips;
pub(in crate::qec) use lower::{
    feedforward_body_letter, feedforward_body_rejection, lower_qec_program_to_deferred_circuit,
    lower_qec_program_to_deferred_circuit_allowing_non_clifford,
    lower_qec_program_to_density_matrix,
};
pub(in crate::qec) use parity::{QecParityNoise, QecParityNoiseEvent};
#[cfg(test)]
pub(in crate::qec) use record::compile_qec_noisy_sampler_on_compiled_base;
pub(in crate::qec) use record::{
    Frame, QecCompiledNoiseSampler, QecRecordEvent, compile_qec_noisy_sampler,
};

use super::leakage::QecDeferredLeakEvent;
use crate::circuit::{Circuit, GateSink, Instruction, SmallVec};
use crate::gates::Gate;
use crate::sim::compiled::{rng::Xoshiro256PlusPlus, xor_words};
use crate::sim::noise::geometric_sample_xoshiro;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Shots per noise unit, part of the seeded-output contract. Unit `k` covers shots
/// `[k * QEC_NOISE_UNIT_SHOTS, (k + 1) * QEC_NOISE_UNIT_SHOTS)` and draws from ChaCha
/// stream `k` of the noise seed, so seeded noise depends on neither chunking nor threads.
const QEC_NOISE_UNIT_SHOTS: usize = 8192;

#[cfg(feature = "parallel")]
const QEC_PARALLEL_MIN_UNITS: usize = 2;

/// Expected faults per run below which noise units run on the calling thread. Faults
/// carry the work inside a unit, including the first touch of each page they land on.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_MIN_FIRINGS: f64 = 1e4;

#[derive(Clone)]
pub(super) struct QecDeferredNoiseEvent {
    pub(super) channel: QecDeferredChannel,
    pub(super) targets: SmallVec<[usize; 2]>,
    pub(super) position: usize,
}

/// Noise channel of a deferred event, with `PAULI_CHANNEL_2` rates held in
/// [`QecDeferredProgram::pair_tables`] and correlated branches in
/// [`QecDeferredProgram::product_tables`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum QecDeferredChannel {
    /// One-qubit channel with its X, Y, Z branch rates.
    Single([f64; 3]),
    Depolarize2(f64),
    /// Two-qubit Pauli channel of total rate `p` whose branches are `pair_tables[table]`.
    PairTable {
        p: f64,
        table: u32,
    },
    /// Mutually exclusive Pauli products over all of the event's targets, of total rate
    /// `p`, with branches in `product_tables[table]`.
    Products {
        p: f64,
        table: u32,
    },
}

/// Branches of a correlated event: `rates[b]` is the absolute rate of branch `b`, which
/// applies letter `letters[b * width + t]` (1 X, 2 Y, 3 Z, 0 none) to the event's
/// target `t`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct QecProductTable {
    pub(super) rates: Vec<f64>,
    letters: Vec<u8>,
    width: usize,
}

impl QecProductTable {
    #[inline]
    pub(super) fn branch_letters(&self, branch: usize) -> &[u8] {
        &self.letters[branch * self.width..(branch + 1) * self.width]
    }
}

/// One gate of the deferred circuit: `targets[1]` is `u32::MAX` for a one-qubit gate.
/// 24 bytes against the 96 of an [`Instruction`], which the lowering and every walk
/// over the gates pay for.
#[derive(Clone)]
pub(super) struct DeferredGate {
    pub(super) gate: Gate,
    targets: [u32; 2],
}

const _: () = assert!(size_of::<DeferredGate>() == 24);

impl DeferredGate {
    #[inline]
    pub(super) fn targets(&self) -> &[u32] {
        &self.targets[..1 + usize::from(self.targets[1] != u32::MAX)]
    }
}

impl GateSink for Vec<DeferredGate> {
    #[inline]
    fn gate(&mut self, gate: Gate, targets: &[usize]) {
        debug_assert!(matches!(targets.len(), 1 | 2));
        self.push(DeferredGate {
            gate,
            targets: [
                targets[0] as u32,
                targets.get(1).map_or(u32::MAX, |&t| t as u32),
            ],
        });
    }
}

/// A QEC program with its measurements deferred: `gates` in program order, then one
/// terminal Z measurement per record on `measurement_qubits`, record order.
pub(super) struct QecDeferredProgram {
    pub(super) gates: Vec<DeferredGate>,
    pub(super) num_aliases: usize,
    pub(super) noise_events: Vec<QecDeferredNoiseEvent>,
    pub(super) measurement_qubits: Vec<usize>,
    /// Gate position at which each alias comes into use: 0 for the initial aliases, the
    /// reset or `MPP` position for a fresh one. An alias is idle in `|0>` before it.
    pub(super) alias_positions: Vec<usize>,
    /// Position at or before which a backward walk first meets each alias (the position
    /// after its last gate, or of its last noise event), `None` for an alias nothing
    /// touches.
    pub(super) last_use: Vec<Option<usize>>,
    /// Final lowered-circuit alias of each program qubit. Resets reassign a
    /// program qubit to a fresh alias, so ops that reference program qubits
    /// at the end of the stream (`EXP_VAL`) must translate through this map.
    pub(super) final_qubit_aliases: Vec<usize>,
    /// Branch rates of each `PAULI_CHANNEL_2` op, in [`qec_pair_branch_flips`] order.
    pub(super) pair_tables: Vec<[f64; 15]>,
    /// Branches of each correlated event, in event order.
    pub(super) product_tables: Vec<QecProductTable>,
    /// Leakage annotations in program order, which only
    /// [`QecLeakageSampler`](super::leakage::QecLeakageSampler) reads.
    pub(super) leak_events: Vec<QecDeferredLeakEvent>,
    /// Pauli feed-forward corrections in program order, empty unless the lowering
    /// admitted them.
    pub(super) feedforwards: Vec<QecDeferredFeedforward>,
}

/// A feed-forward whose body is a Pauli product: the Pauli lands on the live aliases at
/// `position` in the shots where the parity over `condition` XOR `invert` is one.
pub(super) struct QecDeferredFeedforward {
    position: usize,
    condition: Vec<usize>,
    invert: bool,
    /// `(alias, letter)` with letter 1 X, 2 Y, 3 Z.
    paulis: Vec<(u32, u8)>,
}

impl QecDeferredProgram {
    /// The gates followed by the terminal measurements, record `j` on classical bit `j`.
    pub(super) fn to_circuit(&self) -> Circuit {
        let mut circuit = Circuit::new(self.num_aliases, self.measurement_qubits.len());
        circuit
            .instructions
            .reserve_exact(self.gates.len() + self.measurement_qubits.len());
        for gate in &self.gates {
            let targets: SmallVec<[usize; 4]> =
                gate.targets().iter().map(|&t| t as usize).collect();
            circuit.instructions.push(Instruction::Gate {
                gate: gate.gate.clone(),
                targets,
            });
        }
        for (record, &qubit) in self.measurement_qubits.iter().enumerate() {
            circuit.add_measure(qubit, record);
        }
        circuit
    }
}

/// Branch rates a single-qubit event keeps once branches with equal record flips merge,
/// or `None` when it can flip no record.
pub(super) fn qec_single_noise_rates(
    x_flip: &[u64],
    z_flip: &[u64],
    px: f64,
    py: f64,
    pz: f64,
) -> Option<(f64, f64, f64)> {
    let x_is_zero = x_flip.iter().all(|&w| w == 0);
    let z_is_zero = z_flip.iter().all(|&w| w == 0);
    if px + py + pz == 0.0 || (x_is_zero && z_is_zero) {
        return None;
    }
    let (px, py, pz) = if x_is_zero {
        (px + py, 0.0, 0.0)
    } else if z_is_zero {
        (0.0, 0.0, py + pz)
    } else if x_flip == z_flip {
        (px + pz, 0.0, 0.0)
    } else {
        (px, py, pz)
    };
    (px + py + pz != 0.0).then_some((px, py, pz))
}

/// Fill `branch_flips` with the record flips of the 15 depolarize-2 branches, and
/// report whether the event fires at all and flips some record.
pub(super) fn qec_pair_branch_flips(
    q0_x_flip: &[u64],
    q0_z_flip: &[u64],
    q1_x_flip: &[u64],
    q1_z_flip: &[u64],
    p: f64,
    branch_flips: &mut Vec<u64>,
) -> bool {
    if p == 0.0 {
        return false;
    }
    let m_words = q0_x_flip.len();
    branch_flips.clear();
    branch_flips.resize(15 * m_words, 0);
    let mut any = false;
    for (sample, branch) in (1..=15).zip(branch_flips.chunks_exact_mut(m_words.max(1))) {
        append_qec_pauli_noise_effect(branch, sample / 4, q0_x_flip, q0_z_flip);
        append_qec_pauli_noise_effect(branch, sample % 4, q1_x_flip, q1_z_flip);
        any |= branch.iter().any(|&w| w != 0);
    }
    any
}

/// One event's draw constants, computed once so every unit restarts the event from them.
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) enum QecNoiseDraw {
    Single {
        rates: QecSingleNoiseRates,
        /// Branch rates given that the event fires.
        conditional: QecSingleNoiseRates,
        ln_1mp: f64,
    },
    Pair {
        p: f64,
        ln_1mp: f64,
    },
    /// A pair event drawing its branch from `pair_tables[table]`, which sums to `p`.
    PairTable {
        p: f64,
        ln_1mp: f64,
        table: u32,
    },
    /// A product event drawing its branch from `product_tables[table]`, whose rates sum
    /// to `p`.
    Products {
        p: f64,
        ln_1mp: f64,
        table: u32,
    },
}

impl QecNoiseDraw {
    pub(super) fn single(px: f64, py: f64, pz: f64) -> Self {
        let p_event = px + py + pz;
        let px_frac = px / p_event;
        let pxy_frac = (px + py) / p_event;
        Self::Single {
            rates: QecSingleNoiseRates { px, py, p_event },
            conditional: QecSingleNoiseRates {
                px: px_frac,
                py: pxy_frac - px_frac,
                p_event: 1.0,
            },
            ln_1mp: (-p_event).ln_1p(),
        }
    }

    pub(super) fn pair(p: f64) -> Self {
        Self::Pair {
            p,
            ln_1mp: (-p).ln_1p(),
        }
    }

    pub(super) fn pair_table(p: f64, table: u32) -> Self {
        Self::PairTable {
            p,
            ln_1mp: (-p).ln_1p(),
            table,
        }
    }

    pub(super) fn products(p: f64, table: u32) -> Self {
        Self::Products {
            p,
            ln_1mp: (-p).ln_1p(),
            table,
        }
    }

    /// Event rate of a pair draw.
    pub(super) fn pair_rate(&self) -> f64 {
        match *self {
            Self::Pair { p, .. } | Self::PairTable { p, .. } => p,
            Self::Single { .. } | Self::Products { .. } => unreachable!("pair draws only"),
        }
    }

    fn rate(&self) -> f64 {
        match *self {
            Self::Single { rates, .. } => rates.p_event,
            Self::Pair { p, .. } | Self::PairTable { p, .. } | Self::Products { p, .. } => p,
        }
    }
}

/// Whether to spread `units` noise units of `shots` shots in total over Rayon. Scheduling
/// only: every unit draws from its own stream into its own shots, so the answer cannot
/// change output.
#[cfg(feature = "parallel")]
fn qec_noise_units_in_parallel<'a>(
    units: usize,
    shots: usize,
    draws: impl Iterator<Item = &'a QecNoiseDraw>,
) -> bool {
    units >= QEC_PARALLEL_MIN_UNITS
        && shots as f64 * draws.map(QecNoiseDraw::rate).sum::<f64>() >= QEC_PARALLEL_MIN_FIRINGS
}

fn qec_noise_unit_rng(seed: u64, unit: usize) -> Xoshiro256PlusPlus {
    let mut seed_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0x51A7_EC01));
    seed_rng.set_stream(unit as u64);
    Xoshiro256PlusPlus::from_chacha(&mut seed_rng)
}

/// Stream of the birth coins of one unit, apart from [`qec_noise_unit_rng`] so the
/// faults keep their draws.
fn qec_coin_unit_rng(seed: u64, unit: usize) -> Xoshiro256PlusPlus {
    let mut seed_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0xC014_B127));
    seed_rng.set_stream(unit as u64);
    Xoshiro256PlusPlus::from_chacha(&mut seed_rng)
}

/// Draw every event over one unit of `unit_shots` shots, event-major, calling
/// `flip(event_index, event, shot, branch)` per fault. The record and parity paths both
/// draw through here, so they consume each unit's stream identically. `fire_rates`
/// holds each event's [`QecNoiseDraw::rate`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn draw_qec_unit<E>(
    events: &[E],
    draw: impl Fn(&E) -> &QecNoiseDraw,
    fire_rates: &[f64],
    pair_tables: &[[f64; 15]],
    product_tables: &[QecProductTable],
    rng: &mut Xoshiro256PlusPlus,
    unit_shots: usize,
    mut flip: impl FnMut(usize, &E, usize, usize),
) {
    debug_assert_eq!(events.len(), fire_rates.len());
    if unit_shots < QEC_SKIP_MIN_SHOTS {
        // Each kind's per-shot loop takes one uniform per shot and fires below the
        // event's rate, so the rates alone decide each miss and an event is read only
        // when it fires. A one-shot unit otherwise streams the whole event table for a
        // few faults.
        for (index, (event, &p)) in events.iter().zip(fire_rates).enumerate() {
            if p == 0.0 {
                continue;
            }
            for shot in 0..unit_shots {
                let r = rng.next_f64();
                if r >= p {
                    continue;
                }
                let branch = match *draw(event) {
                    QecNoiseDraw::Single { rates, .. } => match rates.branch(r) {
                        Some(branch) => branch,
                        None => continue,
                    },
                    QecNoiseDraw::Pair { .. } => qec_uniform_15(rng),
                    QecNoiseDraw::PairTable { table, .. } => {
                        qec_table_branch(&pair_tables[table as usize], r)
                    }
                    QecNoiseDraw::Products { table, .. } => {
                        qec_table_branch(&product_tables[table as usize].rates, r)
                    }
                };
                flip(index, event, shot, branch);
            }
        }
        return;
    }
    let mut misses = QecMissThresholds::new(unit_shots);
    for (index, event) in events.iter().enumerate() {
        match *draw(event) {
            QecNoiseDraw::Single {
                rates,
                conditional,
                ln_1mp,
            } => {
                let skip = QecSkip::new(ln_1mp, misses.below(ln_1mp));
                draw_qec_single_noise(unit_shots, rates, conditional, skip, rng, |shot, b| {
                    flip(index, event, shot, b)
                })
            }
            QecNoiseDraw::Pair { p, ln_1mp } => {
                let skip = QecSkip::new(ln_1mp, misses.below(ln_1mp));
                draw_qec_pair_noise(unit_shots, p, skip, rng, |shot, b| {
                    flip(index, event, shot, b)
                })
            }
            QecNoiseDraw::PairTable { p, ln_1mp, table } => draw_qec_table_noise(
                unit_shots,
                p,
                QecSkip::new(ln_1mp, misses.below(ln_1mp)),
                &pair_tables[table as usize],
                rng,
                |shot, b| flip(index, event, shot, b),
            ),
            QecNoiseDraw::Products { p, ln_1mp, table } => draw_qec_table_noise(
                unit_shots,
                p,
                QecSkip::new(ln_1mp, misses.below(ln_1mp)),
                &product_tables[table as usize].rates,
                rng,
                |shot, b| flip(index, event, shot, b),
            ),
        }
    }
}

/// Units below this many shots draw each event once per shot instead of skipping
/// between faults.
const QEC_SKIP_MIN_SHOTS: usize = 32;

/// The geometric skip between faults of one event, with the uniform draw below which
/// the first skip lands past every shot.
#[derive(Clone, Copy)]
struct QecSkip {
    ln_1mp: f64,
    miss_below: f64,
}

impl QecSkip {
    #[inline(always)]
    fn new(ln_1mp: f64, miss_below: f64) -> Self {
        Self { ln_1mp, miss_below }
    }

    /// The first fault's shot, consuming the same draw as
    /// [`geometric_sample_xoshiro`], or `usize::MAX` without the logarithm when the
    /// draw falls below `miss_below`.
    #[inline(always)]
    fn first(self, rng: &mut Xoshiro256PlusPlus) -> usize {
        let u = 1.0 - rng.next_f64();
        if u < self.miss_below {
            return usize::MAX;
        }
        (u.ln() / self.ln_1mp) as usize
    }

    #[inline(always)]
    fn next(self, rng: &mut Xoshiro256PlusPlus) -> usize {
        geometric_sample_xoshiro(rng, self.ln_1mp)
    }
}

/// `(1 - p)^shots` per rate, shaded down by a relative 1e-9 so a draw below it skips
/// past every shot whatever the rounding of `ln`. Circuit noise uses a handful of
/// rates, so a few slots cache the `exp` across a unit's events.
struct QecMissThresholds {
    shots: f64,
    slots: [(f64, f64); 4],
    next: usize,
}

impl QecMissThresholds {
    fn new(shots: usize) -> Self {
        Self {
            shots: shots as f64,
            slots: [(f64::NAN, 0.0); 4],
            next: 0,
        }
    }

    #[inline(always)]
    fn below(&mut self, ln_1mp: f64) -> f64 {
        if let Some(&(_, threshold)) = self.slots.iter().find(|(key, _)| *key == ln_1mp) {
            return threshold;
        }
        let threshold = (self.shots * ln_1mp).exp() * (1.0 - 1e-9);
        self.slots[self.next % self.slots.len()] = (ln_1mp, threshold);
        self.next += 1;
        threshold
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(super) struct QecSingleNoiseRates {
    px: f64,
    py: f64,
    p_event: f64,
}

impl QecSingleNoiseRates {
    /// Fault branch for the uniform draw `r`: 0, 1, 2 for X, Y, Z, `None` for no fault.
    #[inline(always)]
    fn branch(self, r: f64) -> Option<usize> {
        if r < self.px {
            Some(0)
        } else if r < self.px + self.py {
            Some(1)
        } else if r < self.p_event {
            Some(2)
        } else {
            None
        }
    }
}

/// Draw one single-qubit Pauli event over `num_shots` shots, calling `flip(shot, branch)`
/// per fault with the branch of [`QecSingleNoiseRates::branch`].
#[inline(always)]
fn draw_qec_single_noise(
    num_shots: usize,
    rates: QecSingleNoiseRates,
    conditional: QecSingleNoiseRates,
    skip: QecSkip,
    rng: &mut Xoshiro256PlusPlus,
    mut flip: impl FnMut(usize, usize),
) {
    if rates.p_event == 0.0 {
        return;
    }

    if rates.p_event >= 0.5 {
        for shot in 0..num_shots {
            if let Some(branch) = rates.branch(rng.next_f64()) {
                flip(shot, branch);
            }
        }
        return;
    }

    let mut shot = skip.first(rng);
    while shot < num_shots {
        if let Some(branch) = conditional.branch(rng.next_f64()) {
            flip(shot, branch);
        }
        shot += 1 + skip.next(rng);
    }
}

/// Draw one depolarize-2 event over `num_shots` shots, calling `flip(shot, branch)` per
/// fault with a uniform branch in `0..15` (see [`qec_pair_branch_flips`] for the order).
#[inline(always)]
fn draw_qec_pair_noise(
    num_shots: usize,
    p: f64,
    skip: QecSkip,
    rng: &mut Xoshiro256PlusPlus,
    mut flip: impl FnMut(usize, usize),
) {
    if p == 0.0 {
        return;
    }

    if p >= 0.5 {
        for shot in 0..num_shots {
            if rng.next_f64() < p {
                flip(shot, qec_uniform_15(rng));
            }
        }
        return;
    }

    let mut shot = skip.first(rng);
    while shot < num_shots {
        flip(shot, qec_uniform_15(rng));
        shot += 1 + skip.next(rng);
    }
}

/// Draw one `PAULI_CHANNEL_2` or product event over `num_shots` shots, calling
/// `flip(shot, branch)` per fault with a branch drawn from `rates`, which sum to `p`.
#[inline(always)]
fn draw_qec_table_noise(
    num_shots: usize,
    p: f64,
    skip: QecSkip,
    rates: &[f64],
    rng: &mut Xoshiro256PlusPlus,
    mut flip: impl FnMut(usize, usize),
) {
    if p == 0.0 {
        return;
    }

    if p >= 0.5 {
        for shot in 0..num_shots {
            let r = rng.next_f64();
            if r < p {
                flip(shot, qec_table_branch(rates, r));
            }
        }
        return;
    }

    let mut shot = skip.first(rng);
    while shot < num_shots {
        flip(shot, qec_table_branch(rates, rng.next_f64() * p));
        shot += 1 + skip.next(rng);
    }
}

/// Branch whose cumulative rate interval holds `r`, for `r` in `[0, p)`.
#[inline(always)]
fn qec_table_branch(rates: &[f64], r: f64) -> usize {
    let mut cumulative = 0.0;
    let mut last = 0;
    for (branch, &rate) in rates.iter().enumerate() {
        if rate > 0.0 {
            cumulative += rate;
            last = branch;
            if r < cumulative {
                return branch;
            }
        }
    }
    last
}

pub(super) fn append_qec_pauli_noise_effect(
    branch: &mut [u64],
    pauli: usize,
    x_flip: &[u64],
    z_flip: &[u64],
) {
    match pauli {
        0 => {}
        1 => xor_words(branch, z_flip),
        2 => {
            xor_words(branch, x_flip);
            xor_words(branch, z_flip);
        }
        _ => xor_words(branch, x_flip),
    }
}

#[inline(always)]
fn qec_uniform_15(rng: &mut Xoshiro256PlusPlus) -> usize {
    const BRANCHES: u64 = 15;
    const ZONE: u64 = u64::MAX - (u64::MAX % BRANCHES);
    loop {
        let value = rng.next_u64();
        if value < ZONE {
            return (value % BRANCHES) as usize;
        }
    }
}
