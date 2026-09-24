//! Monte Carlo trajectory execution for noisy circuits: one pure-state
//! simulation per shot, sampling a noise branch after each instruction, or one
//! per distinct error pattern when every shot's Pauli errors can be drawn first.

use std::collections::HashMap;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::smallvec;

use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::ShotsResult;
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel, ReadoutError};

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

/// Draw one branch of a single-qubit Pauli channel, `None` for the identity.
fn draw_pauli(px: f64, py: f64, pz: f64, rng: &mut ChaCha8Rng) -> Option<PauliOp> {
    let r: f64 = rand::RngExt::random(rng);
    if r < px {
        Some(PauliOp::X)
    } else if r < px + py {
        Some(PauliOp::Y)
    } else if r < px + py + pz {
        Some(PauliOp::Z)
    } else {
        None
    }
}

fn apply_pauli(
    backend: &mut dyn Backend,
    qubit: usize,
    px: f64,
    py: f64,
    pz: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match draw_pauli(px, py, pz, rng) {
        Some(op) => apply_pauli_op(backend, qubit, op),
        None => Ok(()),
    }
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
    gamma: f64,
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
            [zero, Complex64::new((1.0 - gamma).sqrt() * inv, 0.0)],
        ];
        backend.apply_1q_matrix(qubit, &mat)?;
    }
    Ok(())
}

fn apply_amplitude_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    gamma: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, gamma, true, rng)
}

fn apply_phase_damping(
    backend: &mut dyn Backend,
    qubit: usize,
    gamma: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    apply_diagonal_kraus_2op(backend, qubit, gamma, false, rng)
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
    t1: f64,
    t2: f64,
    gate_time: f64,
    excited: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    if t1 <= 0.0 || t2 <= 0.0 || gate_time <= 0.0 {
        return Ok(());
    }
    let (gad, gpd) = crate::sim::noise::thermal_relaxation_rates(t1, t2, gate_time);
    let p1 = backend.qubit_probability(qubit)?;
    let p0 = 1.0 - p1;
    let cold = 1.0 - excited;

    let p_relax = cold * gad * p1;
    let p_dephase = ((cold * (1.0 - gad)) + excited) * gpd * p1;
    let p_excite = excited * gad * p0;
    let p_hold_hot = excited * ((1.0 - gad) * p0 + (1.0 - gpd) * p1);

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
        let inv = (excited / p_hold_hot).sqrt();
        [
            [Complex64::new((1.0 - gad).sqrt() * inv, 0.0), zero],
            [zero, Complex64::new((1.0 - gpd).sqrt() * inv, 0.0)],
        ]
    } else {
        let denom = 1.0 - p_relax - p_dephase - p_excite - p_hold_hot;
        if denom <= JUMP_EPSILON {
            return Ok(());
        }
        let inv = cold.sqrt() / denom.sqrt();
        let keep = ((1.0 - gad) * (1.0 - gpd)).sqrt() * inv;
        [
            [Complex64::new(inv, 0.0), zero],
            [zero, Complex64::new(keep, 0.0)],
        ]
    };
    backend.apply_1q_matrix(qubit, &mat)
}

/// Draw one branch of symmetric two-qubit depolarizing as an index into
/// [`TWO_QUBIT_PAULIS`], `None` for `I (x) I`. Each of the 15 non-identity
/// products has probability `p/15`.
fn draw_two_qubit_depolarizing(p: f64, rng: &mut ChaCha8Rng) -> Option<usize> {
    let r: f64 = rand::RngExt::random(rng);
    if r >= p {
        return None;
    }
    Some(((r / (p / 15.0)) as usize).min(14))
}

fn apply_two_qubit_depolarizing(
    backend: &mut dyn Backend,
    q0: usize,
    q1: usize,
    p: f64,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let Some(idx) = draw_two_qubit_depolarizing(p, rng) else {
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

fn apply_pauli_op(backend: &mut dyn Backend, qubit: usize, op: PauliOp) -> Result<()> {
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
fn kraus_probability(k: &[[Complex64; 2]; 2], rho: &[[Complex64; 2]; 2]) -> f64 {
    let effect = kdagger_k(k);
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
    kraus: &[[[Complex64; 2]; 2]],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let rho = backend.reduced_density_matrix_1q(qubit)?;

    let mut cumulative: smallvec::SmallVec<[f64; 8]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        let pk = kraus_probability(k, &rho);
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
    let k = kraus[chosen];
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
    q0: usize,
    q1: usize,
    kraus: &[[[Complex64; 4]; 4]],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let rho = backend.reduced_density_matrix_2q(q0, q1)?;

    let mut cumulative: smallvec::SmallVec<[f64; 16]> = smallvec::SmallVec::new();
    let mut total = 0.0;
    for k in kraus {
        total += kraus_probability_2q(k, &rho);
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
    let mut normalized = kraus[chosen];
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

/// `Tr(Kdagger K rho) = sum_{i,j,r} conj(K[r][i]) K[r][j] rho[j][i]`.
fn kraus_probability_2q(k: &[[Complex64; 4]; 4], rho: &[[Complex64; 4]; 4]) -> f64 {
    let mut p = Complex64::new(0.0, 0.0);
    for i in 0..4 {
        for j in 0..4 {
            let mut effect = Complex64::new(0.0, 0.0);
            for row in k.iter() {
                effect += row[i].conj() * row[j];
            }
            p += effect * rho[j][i];
        }
    }
    p.re.max(0.0)
}

fn apply_noise_event(
    backend: &mut dyn Backend,
    event: &NoiseEvent,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match &event.channel {
        NoiseChannel::Pauli { px, py, pz } => {
            apply_pauli(backend, event.qubits[0], *px, *py, *pz, rng)
        }
        NoiseChannel::Depolarizing { p } => {
            let pp = p / 3.0;
            apply_pauli(backend, event.qubits[0], pp, pp, pp, rng)
        }
        NoiseChannel::AmplitudeDamping { gamma } => {
            apply_amplitude_damping(backend, event.qubits[0], *gamma, rng)
        }
        NoiseChannel::PhaseDamping { gamma } => {
            apply_phase_damping(backend, event.qubits[0], *gamma, rng)
        }
        NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        } => apply_thermal_relaxation(
            backend,
            event.qubits[0],
            *t1,
            *t2,
            *gate_time,
            *excited_population,
            rng,
        ),
        NoiseChannel::TwoQubitDepolarizing { p } => {
            apply_two_qubit_depolarizing(backend, event.qubits[0], event.qubits[1], *p, rng)
        }
        NoiseChannel::Custom { kraus } => apply_custom_kraus(backend, event.qubits[0], kraus, rng),
        NoiseChannel::Kraus2q { kraus } => {
            apply_custom_kraus_2q(backend, event.qubits[0], event.qubits[1], kraus, rng)
        }
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

/// `readout` is the output of [`written_readout`] for `circuit` and `noise`,
/// computed once by the caller rather than per shot.
pub(crate) fn run_trajectory_shot(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    noise: &NoiseModel,
    readout: &[Option<ReadoutError>],
    rng: &mut ChaCha8Rng,
) -> Result<Vec<bool>> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;

    for (idx, instr) in circuit.instructions.iter().enumerate() {
        backend.apply(instr)?;
        for event in &noise.after_gate[idx] {
            apply_noise_event(backend, event, rng)?;
        }
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
    #[cfg(not(feature = "parallel"))]
    let _ = force_serial;
    #[cfg(feature = "parallel")]
    {
        if !force_serial
            && num_shots >= 4
            && crate::sim::state_splits_across_workers(route, circuit.num_qubits)
        {
            return run_trajectories_par(&backend_factory, circuit, noise, num_shots, seed, route);
        }
    }

    let readout = written_readout(circuit, &noise.readout);
    let mut shots = Vec::with_capacity(num_shots);
    let mut metadata = crate::sim::RunMetadata::exact(route);
    for i in 0..num_shots {
        let shot_seed = crate::sim::mix_seed(seed, i);
        let mut rng = noise_rng(shot_seed);
        let mut backend = backend_factory(shot_seed);
        let result = run_trajectory_shot(backend.as_mut(), circuit, noise, &readout, &mut rng)?;
        let shot_metadata = crate::sim::backend_metadata(backend.as_ref());
        if i == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(result);
    }

    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
}

#[cfg(feature = "parallel")]
fn run_trajectories_par(
    backend_factory: &(impl Fn(u64) -> Box<dyn Backend> + Sync),
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
    route: crate::sim::ResolvedBackend,
) -> Result<ShotsResult> {
    let readout = written_readout(circuit, &noise.readout);
    let results: Result<Vec<(Vec<bool>, crate::sim::RunMetadata)>> = (0..num_shots)
        .into_par_iter()
        .map(|i| {
            let shot_seed = crate::sim::mix_seed(seed, i);
            let mut rng = noise_rng(shot_seed);
            let mut backend = backend_factory(shot_seed);
            let bits = run_trajectory_shot(backend.as_mut(), circuit, noise, &readout, &mut rng)?;
            Ok((bits, crate::sim::backend_metadata(backend.as_ref())))
        })
        .collect();

    let mut metadata = crate::sim::RunMetadata::exact(route);
    let mut shots = Vec::with_capacity(num_shots);
    for (index, (bits, shot_metadata)) in results?.into_iter().enumerate() {
        if index == 0 {
            metadata = shot_metadata;
        } else {
            metadata.weaken_with(&shot_metadata);
        }
        shots.push(bits);
    }
    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(metadata))
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
    /// Shot indices ordered by group, ascending within each group.
    members: Vec<usize>,
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

fn draw_frame_branch(event: &NoiseEvent, rng: &mut ChaCha8Rng) -> Option<[PauliOp; 2]> {
    if let Some(p) = event.channel.pauli_pair_rate() {
        let (a, b) = TWO_QUBIT_PAULIS[draw_two_qubit_depolarizing(p, rng)?];
        return Some([a, b]);
    }
    let (px, py, pz) = event.pauli_probs();
    draw_pauli(px, py, pz, rng).map(|op| [op, PauliOp::I])
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
        let prefix = pauli_group_prefix(circuit, noise)?;
        let mut index: HashMap<Vec<Insertion>, usize> = HashMap::new();
        let mut shot_group = Vec::with_capacity(num_shots);
        let mut pattern = Vec::new();
        for shot in 0..num_shots {
            let mut rng = noise_rng(crate::sim::mix_seed(seed, shot));
            pattern.clear();
            let events = noise.after_gate[..prefix].iter().flatten();
            for (event, noise_event) in events.enumerate() {
                if let Some(letters) = draw_frame_branch(noise_event, &mut rng) {
                    pattern.push(Insertion { event, letters });
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
            members,
            offsets,
        })
    }

    pub(crate) fn num_groups(&self) -> usize {
        self.patterns.len()
    }

    fn members(&self, group: usize) -> &[usize] {
        &self.members[self.offsets[group]..self.offsets[group + 1]]
    }
}

/// Evolve one pattern's state, then draw the record of each member shot from it.
///
/// Member shot `i` draws its outcome and then its readout flips from
/// `ChaCha8Rng::seed_from_u64(mix_seed(seed, i))`, the stream a per-shot
/// trajectory measures on, apart from the stream its errors came from. The
/// outcome is an inverse-CDF draw: the members' uniforms are sorted and matched
/// against one cumulative pass over the amplitudes, so no `2^n` table is built.
fn run_pauli_group(
    groups: &PauliGroups,
    group: usize,
    circuit: &Circuit,
    noise: &NoiseModel,
    meas_map: &[(usize, usize)],
    readout: &[Option<ReadoutError>],
    seed: u64,
) -> Result<(Vec<Vec<bool>>, crate::sim::RunMetadata)> {
    let members = groups.members(group);
    let mut backend = StatevectorBackend::new(crate::sim::mix_seed(seed, members[0]));
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let mut insertions = groups.patterns[group].iter().peekable();
    let mut ordinal = 0usize;
    for (instruction, events) in circuit.instructions[..groups.prefix]
        .iter()
        .zip(&noise.after_gate)
    {
        backend.apply(instruction)?;
        for event in events {
            if let Some(insertion) = insertions.next_if(|ins| ins.event == ordinal) {
                for (&qubit, &letter) in event.qubits.iter().zip(&insertion.letters) {
                    apply_pauli_op(&mut backend, qubit, letter)?;
                }
            }
            ordinal += 1;
        }
    }

    let mut draws: Vec<(f64, usize)> = members
        .iter()
        .enumerate()
        .map(|(k, &shot)| {
            let mut rng = ChaCha8Rng::seed_from_u64(crate::sim::mix_seed(seed, shot));
            (rand::RngExt::random::<f64>(&mut rng), k)
        })
        .collect();
    draws.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));

    let scale = backend.probability_scale();
    let mut outcomes = vec![0usize; members.len()];
    let mut cumulative = 0.0f64;
    let mut last_nonzero = 0usize;
    let mut next = 0usize;
    for (basis, amp) in backend.state_vector().iter().enumerate() {
        let weight = amp.norm_sqr() * scale;
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

    let has_readout = readout.iter().any(Option::is_some);
    let shots = members
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
        .collect();
    Ok((shots, crate::sim::backend_metadata(&backend)))
}

/// Run every shot of `groups` on the host statevector, one evolution per
/// distinct error pattern, and return the shots in index order.
///
/// Groups split across Rayon workers under the rule the per-shot trajectories
/// use, each worker holding one state at a time. Every draw depends only on
/// its shot's seed, so the result does not depend on the thread count.
pub(crate) fn run_pauli_groups(
    groups: &PauliGroups,
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<ShotsResult> {
    let meas_map = circuit.measurement_map();
    let readout = written_readout(circuit, &noise.readout);
    let run =
        |group: usize| run_pauli_group(groups, group, circuit, noise, &meas_map, &readout, seed);

    #[cfg(feature = "parallel")]
    if groups.num_groups() > 1
        && crate::sim::state_splits_across_workers(
            crate::sim::ResolvedBackend::Statevector,
            circuit.num_qubits,
        )
    {
        let runs: Vec<_> = (0..groups.num_groups()).into_par_iter().map(run).collect();
        return collect_groups(groups, circuit, runs);
    }
    collect_groups(groups, circuit, (0..groups.num_groups()).map(run))
}

fn collect_groups(
    groups: &PauliGroups,
    circuit: &Circuit,
    runs: impl IntoIterator<Item = Result<(Vec<Vec<bool>>, crate::sim::RunMetadata)>>,
) -> Result<ShotsResult> {
    let mut shots = vec![Vec::new(); groups.members.len()];
    let mut metadata = crate::sim::RunMetadata::exact(crate::sim::ResolvedBackend::Statevector);
    for (group, run) in runs.into_iter().enumerate() {
        let (records, group_metadata) = run?;
        if group == 0 {
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
mod tests {
    use super::*;
    use crate::circuits;

    // The replica cap is only a memory bound if it stays at or below the cap
    // governing a single state. Splitting statevector trajectories past the
    // statevector cap would let the parallel path allocate one oversize state
    // per thread, which is the failure `run_trajectories` avoids by falling
    // back to serial execution.
    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_trajectory_replicas_stay_within_a_single_state_cap() {
        let cap = crate::backend::max_statevector_qubits();
        let statevector = crate::sim::ResolvedBackend::Statevector;
        let replica_cap = (1..)
            .find(|&n| !crate::sim::state_splits_across_workers(statevector, n))
            .unwrap();
        assert!(
            replica_cap <= cap,
            "parallel replica cap {replica_cap} exceeds the statevector cap {cap}"
        );
    }

    #[test]
    fn trajectory_pauli_matches_brute_force() {
        let n = 5;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.measure_all();

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.02);
        let num_shots = 5000;

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };
        let trajectory = run_trajectories(
            factory,
            &circuit,
            &noise,
            num_shots,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let brute = crate::sim::noise::run_shots_noisy_brute_with(
            |s| Box::new(crate::backend::stabilizer::StabilizerBackend::new(s)),
            &circuit,
            &noise,
            num_shots,
            42,
        )
        .unwrap();

        let traj_coh = trajectory.coherent_fraction();
        let brute_coh = brute.coherent_fraction();
        // 5 sigma for a pairwise comparison at 5000 shots.
        assert!(
            (traj_coh - brute_coh).abs() < 0.05,
            "coherent fraction: trajectory={traj_coh:.3}, brute={brute_coh:.3}"
        );
        assert!(traj_coh < 1.0, "noise should produce non-GHZ outcomes");

        for bit in 0..n {
            let t = trajectory.marginal(bit);
            let b = brute.marginal(bit);
            assert!(
                (t - b).abs() < 0.05,
                "bit {bit}: trajectory marginal {t:.3} vs brute {b:.3}"
            );
        }
    }

    #[test]
    fn amplitude_damping_decays_to_ground() {
        // |1⟩ with strong amplitude damping should mostly decay to |0⟩
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
        circuit.add_measure(0, 0);

        let noise = NoiseModel::with_amplitude_damping(&circuit, 0.9);
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let num_zero = result.shots.iter().filter(|s| !s[0]).count();
        // With gamma=0.9 on |1⟩, P(decay) ≈ 0.9
        assert!(
            num_zero > 700,
            "strong AD should decay most to |0⟩, got {} zeros",
            num_zero
        );
    }

    #[test]
    fn phase_damping_preserves_populations() {
        // Phase damping should not change |0⟩/|1⟩ populations
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]); // prepare |1⟩
        circuit.add_measure(0, 0);

        let mut pd_noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        for events in &mut pd_noise.after_gate {
            *events = events
                .iter()
                .map(|e| NoiseEvent {
                    channel: NoiseChannel::PhaseDamping { gamma: 0.9 },
                    qubits: e.qubits.clone(),
                })
                .collect();
        }

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(
            factory,
            &circuit,
            &pd_noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let num_one = result.shots.iter().filter(|s| s[0]).count();
        // Phase damping on |1⟩ should keep it as |1⟩ (only dephases superpositions)
        assert_eq!(num_one, 1000, "PD should not change |1⟩ population");
    }

    #[test]
    fn readout_error_flips_bits() {
        let mut circuit = Circuit::new(1, 1);
        circuit.add_measure(0, 0); // measure |0⟩

        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        noise.with_readout_error(0.3, 0.0); // 30% chance of 0→1

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let num_one = result.shots.iter().filter(|s| s[0]).count();
        // Should see ~30% readout flips
        assert!(
            num_one > 200 && num_one < 400,
            "readout error p01=0.3 should flip ~30% of |0⟩ outcomes, got {}",
            num_one
        );
    }

    #[test]
    fn two_qubit_depolarizing_produces_errors() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);

        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        noise.after_gate[0] = vec![NoiseEvent {
            channel: NoiseChannel::TwoQubitDepolarizing { p: 0.5 },
            qubits: smallvec![0, 1],
        }];

        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let result = run_trajectories(
            factory,
            &circuit,
            &noise,
            1000,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        // With 50% 2q depolarizing, varied outcomes should appear.
        let num_00 = result.shots.iter().filter(|s| !s[0] && !s[1]).count();
        assert!(
            num_00 < 900,
            "strong 2q depolarizing should produce errors, got {} |00⟩",
            num_00
        );
    }

    #[test]
    fn zero_noise_trajectory_deterministic() {
        let n = 3;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.measure_all();

        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::statevector::StatevectorBackend::new(s))
        };

        let r1 = run_trajectories(
            factory,
            &circuit,
            &noise,
            100,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        let r2 = run_trajectories(
            factory,
            &circuit,
            &noise,
            100,
            42,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap();
        assert_eq!(r1.shots, r2.shots, "same seed must produce same results");
    }

    fn grouped_circuit(n: usize) -> Circuit {
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
        }
        for q in 0..n - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            circuit.add_gate(Gate::Ry(0.3 + 0.1 * q as f64), &[q]);
            circuit.add_gate(Gate::Rx(0.2 + 0.15 * q as f64), &[q]);
        }
        circuit.measure_all();
        circuit
    }

    fn statevector_route(
        circuit: &Circuit,
        noise: &NoiseModel,
        num_shots: usize,
        seed: u64,
    ) -> ShotsResult {
        crate::sim::run_shots_with_noise(
            crate::sim::BackendKind::Statevector,
            circuit,
            noise,
            num_shots,
            seed,
        )
        .unwrap()
    }

    fn per_shot(circuit: &Circuit, noise: &NoiseModel, num_shots: usize, seed: u64) -> ShotsResult {
        let factory = |s: u64| -> Box<dyn Backend> { Box::new(StatevectorBackend::new(s)) };
        run_trajectories(
            factory,
            circuit,
            noise,
            num_shots,
            seed,
            false,
            crate::sim::ResolvedBackend::Statevector,
        )
        .unwrap()
    }

    // Every branch kind the grouped path draws: symmetric depolarizing, an
    // asymmetric Pauli weighted onto Y, the two-qubit pair channel, and
    // asymmetric readout. The reference is the density-matrix distribution with
    // the readout flips folded in bit by bit, checked per outcome and pooled as a
    // chi-square at the Wilson-Hilferty 5-sigma quantile. The trailing rotations
    // leave X, Y and Z errors after `T` distinguishable in the record; with `Ry`
    // alone a Y and a Z there read the same, and swapping their rates passed.
    #[test]
    fn pauli_groups_sample_the_density_matrix_distribution_within_5_sigma() {
        let n = 5;
        let circuit = grouped_circuit(n);
        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        for (index, instruction) in circuit.instructions.iter().enumerate() {
            if let Instruction::Gate { gate, targets } = instruction {
                if matches!(gate, Gate::Cx) {
                    noise.after_gate[index].push(NoiseEvent {
                        channel: NoiseChannel::TwoQubitDepolarizing { p: 0.1 },
                        qubits: smallvec![targets[0], targets[1]],
                    });
                }
                if matches!(gate, Gate::T) {
                    noise.after_gate[index].push(NoiseEvent::pauli(targets[0], 0.005, 0.05, 0.01));
                }
            }
        }
        let quantum_only = noise.clone();
        let (p01, p10) = (0.02, 0.05);
        noise.with_readout_error(p01, p10);

        let num_shots = 100_000;
        let seed = 0xDEAD_BEEF;
        let groups = PauliGroups::sample(&circuit, &noise, num_shots, seed).unwrap();
        assert!(
            groups.num_groups() < num_shots / 2,
            "{} groups at {num_shots} shots leaves too little grouping to test",
            groups.num_groups()
        );
        let result = statevector_route(&circuit, &noise, num_shots, seed);
        assert_eq!(
            result.shots,
            run_pauli_groups(&groups, &circuit, &noise, seed)
                .unwrap()
                .shots
        );

        let probs = crate::sim::noise::density_matrix_probabilities(
            &crate::sim::BackendKind::DensityMatrix,
            &circuit,
            &quantum_only,
            None,
            seed,
        )
        .unwrap();
        let dim = 1usize << n;
        let mut exact = vec![0.0f64; dim];
        for (x, &px) in probs.iter().enumerate() {
            for (y, slot) in exact.iter_mut().enumerate() {
                let mut weight = px;
                for bit in 0..n {
                    weight *= match ((x >> bit) & 1, (y >> bit) & 1) {
                        (0, 0) => 1.0 - p01,
                        (0, _) => p01,
                        (_, 0) => p10,
                        _ => 1.0 - p10,
                    };
                }
                *slot += weight;
            }
        }

        let mut counts = vec![0u64; dim];
        for shot in &result.shots {
            let index = shot
                .iter()
                .enumerate()
                .fold(0usize, |acc, (bit, &b)| acc | (usize::from(b) << bit));
            counts[index] += 1;
        }
        let total = num_shots as f64;
        for (index, (&count, &p)) in counts.iter().zip(&exact).enumerate() {
            let sigma = (p * (1.0 - p) / total).sqrt();
            let observed = count as f64 / total;
            assert!(
                (observed - p).abs() <= 5.0 * sigma,
                "outcome {index}: exact {p}, sampled {observed}, tolerance {}",
                5.0 * sigma
            );
        }

        let chi_square: f64 = counts
            .iter()
            .zip(&exact)
            .map(|(&count, &p)| {
                let expected = p * total;
                (count as f64 - expected).powi(2) / expected
            })
            .sum();
        let dof = (dim - 1) as f64;
        let spread = (2.0 / (9.0 * dof)).sqrt();
        let bound = dof * (1.0 - 2.0 / (9.0 * dof) + 5.0 * spread).powi(3);
        assert!(
            chi_square < bound,
            "chi-square {chi_square:.1} over {dof} degrees of freedom exceeds {bound:.1}"
        );
    }

    #[test]
    fn pauli_groups_decline_what_draws_against_the_state_or_an_outcome() {
        let n = 4;
        let seed = 42;
        let shots = 200;
        let eligible = grouped_circuit(n);
        let low = |c: &Circuit| NoiseModel::uniform_depolarizing(c, 1e-3);
        assert!(PauliGroups::sample(&eligible, &low(&eligible), shots, seed).is_some());

        let mut readout_only = NoiseModel::uniform_depolarizing(&eligible, 0.0);
        readout_only.with_readout_error(0.1, 0.1);
        let groups = PauliGroups::sample(&eligible, &readout_only, shots, seed).unwrap();
        assert_eq!(groups.num_groups(), 1);

        let mut mid_measured = Circuit::new(n, n + 1);
        mid_measured.add_gate(Gate::H, &[0]);
        mid_measured.add_measure(0, n);
        mid_measured
            .instructions
            .extend(eligible.instructions.iter().cloned());
        let mut reset = eligible.clone();
        reset
            .instructions
            .insert(n, Instruction::Reset { qubit: 1 });
        let mut conditional = eligible.clone();
        conditional.instructions.insert(
            n,
            Instruction::Conditional {
                condition: crate::circuit::ClassicalCondition::BitIsOne(0),
                gate: Gate::X,
                targets: smallvec![1],
            },
        );
        let unmeasured = eligible.without_measurements();
        for (label, circuit) in [
            ("mid-circuit measurement", &mid_measured),
            ("reset", &reset),
            ("conditional", &conditional),
            ("no measurement", &unmeasured),
        ] {
            assert!(
                PauliGroups::sample(circuit, &low(circuit), shots, seed).is_none(),
                "{label} must keep the per-shot path"
            );
        }

        let damping = NoiseModel::with_amplitude_damping(&eligible, 0.01);
        assert!(PauliGroups::sample(&eligible, &damping, shots, seed).is_none());

        let mut after_measure = low(&eligible);
        let first_measure = eligible
            .instructions
            .iter()
            .position(|i| matches!(i, Instruction::Measure { .. }))
            .unwrap();
        after_measure.after_gate[first_measure].push(NoiseEvent::pauli(1, 0.01, 0.0, 0.0));
        assert!(PauliGroups::sample(&eligible, &after_measure, shots, seed).is_none());
        after_measure.after_gate[first_measure][0] = NoiseEvent::pauli(1, 0.0, 0.0, 0.0);
        assert!(PauliGroups::sample(&eligible, &after_measure, shots, seed).is_some());

        let saturated = NoiseModel::uniform_depolarizing(&eligible, 0.5);
        assert!(
            PauliGroups::sample(&eligible, &saturated, shots, seed).is_none(),
            "near one pattern per shot must fall back"
        );
    }

    // The route check, end to end: a circuit the grouping declines, and an
    // eligible circuit on a route other than the host statevector, both return
    // the per-shot trajectories bit for bit.
    #[test]
    fn ineligible_noisy_shots_keep_the_per_shot_trajectories() {
        let n = 6;
        let seed = 42;
        let mut mid_measured = Circuit::new(n, n + 1);
        mid_measured.add_gate(Gate::H, &[0]);
        mid_measured.add_gate(Gate::T, &[0]);
        mid_measured.add_measure(0, n);
        mid_measured
            .instructions
            .extend(grouped_circuit(n).instructions);
        let noise = NoiseModel::uniform_depolarizing(&mid_measured, 1e-3);
        assert_eq!(
            statevector_route(&mid_measured, &noise, 300, seed).shots,
            per_shot(&mid_measured, &noise, 300, seed).shots
        );

        let mut product = Circuit::new(n, n);
        for q in 0..n {
            product.add_gate(Gate::H, &[q]);
            product.add_gate(Gate::T, &[q]);
        }
        product.measure_all();
        let noise = NoiseModel::uniform_depolarizing(&product, 1e-3);
        assert!(PauliGroups::sample(&product, &noise, 300, seed).is_some());
        let auto = crate::sim::run_shots_with_noise(
            crate::sim::BackendKind::Auto,
            &product,
            &noise,
            300,
            seed,
        )
        .unwrap();
        assert_eq!(
            auto.metadata.backend,
            crate::sim::ResolvedBackend::ProductState
        );
        let factory = |s: u64| -> Box<dyn Backend> {
            Box::new(crate::backend::product::ProductStateBackend::new(s))
        };
        let product_per_shot = run_trajectories(
            factory,
            &product,
            &noise,
            300,
            seed,
            false,
            crate::sim::ResolvedBackend::ProductState,
        )
        .unwrap();
        assert_eq!(auto.shots, product_per_shot.shots);
    }

    // Shot `i` of a run seeded 42 must not reappear as shot `i + k` of a run
    // seeded 43, which a seed of `seed + i` would give.
    #[test]
    fn adjacent_run_seeds_draw_unrelated_grouped_shots() {
        let n = 8;
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
            circuit.add_gate(Gate::T, &[q]);
        }
        circuit.measure_all();
        let noise = NoiseModel::uniform_depolarizing(&circuit, 1e-3);
        let shots = 200;
        let groups = PauliGroups::sample(&circuit, &noise, shots, 42).unwrap();
        assert!(groups.num_groups() < shots / 2);

        let a = statevector_route(&circuit, &noise, shots, 42);
        let b = statevector_route(&circuit, &noise, shots, 43);
        for shift in 0..=4 {
            let aligned = (0..shots - shift)
                .filter(|&i| a.shots[i + shift] == b.shots[i] || a.shots[i] == b.shots[i + shift])
                .count();
            assert!(
                aligned < shots / 10,
                "runs seeded 42 and 43 agree at {aligned} positions under shift {shift}"
            );
        }
    }

    // Grouping moves no draw of the error stream, so a shot fires the errors it
    // fired on the per-shot path. With the outcome fixed by the circuit, the
    // records then agree shot for shot.
    #[test]
    fn grouped_shots_fire_the_per_shot_errors() {
        let n = 3;
        let mut circuit = Circuit::new(n, n);
        for q in 0..n {
            circuit.add_gate(Gate::T, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.measure_all();
        let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
        noise.after_gate[0].push(NoiseEvent::pauli(0, 0.05, 0.0, 0.0));
        noise.after_gate[n].push(NoiseEvent {
            channel: NoiseChannel::TwoQubitDepolarizing { p: 0.1 },
            qubits: smallvec![1, 2],
        });
        let shots = 500;
        assert!(PauliGroups::sample(&circuit, &noise, shots, 7).is_some());
        let grouped = statevector_route(&circuit, &noise, shots, 7);
        let reference = per_shot(&circuit, &noise, shots, 7);
        assert_eq!(grouped.shots, reference.shots);
        assert!(grouped.shots.iter().any(|s| s.iter().any(|&b| b)));
    }
}
