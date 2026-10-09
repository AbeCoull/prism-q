//! Leakage on the compiled QEC path: per-shot leak flags carried through a forward
//! Pauli frame pass, with each leaked qubit sampled as a heralded erasure.
//!
//! A leaked qubit is erased: its frame is replaced by a uniformly random Pauli when
//! it leaks, and again with its partner's at every two-qubit gate it meets, which is
//! the completely depolarizing channel on both. Since `Tr_ab(U rho U^dag)` equals
//! `Tr_ab(rho)` for any `U` on the pair, the rest of the register then sees exactly
//! what it sees when the gate is skipped and the partner depolarized, so on a
//! Clifford program the records match per-shot trajectory leakage in distribution.
//! The erased qubit's own record is forced to 1.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::noise::{Frame, QecDeferredChannel, QecDeferredNoiseEvent, QecDeferredProgram};
use super::{QecNoise, QecProgram};
use crate::error::{PrismError, Result};
use crate::sim::compiled::rng::Xoshiro256PlusPlus;
use crate::sim::compiled::{CompiledSampler, PackedShots, compile_measurements};
use crate::sim::noise::geometric_sample_xoshiro;

/// Shots per noise unit. Unit `k` draws from ChaCha stream `k` of the seed, so seeded
/// output depends on neither the chunk size nor the thread count.
const LEAK_UNIT_SHOTS: usize = 8192;

/// A leakage annotation on the deferred circuit. A target is `(alias, live)`, where a
/// target measured since its last reset is not live: its flag is frozen at the value
/// its record was forced with, and the annotation only reads it for the herald.
pub(super) struct QecDeferredLeakEvent {
    pub(super) channel: LeakChannel,
    pub(super) targets: Vec<(usize, bool)>,
    pub(super) position: usize,
    /// Pauli annotations lowered before this one, so the two lists replay in program
    /// order where they share a position.
    pub(super) pauli_before: usize,
    /// Herald column of the first target of a `LEAK`.
    pub(super) first_herald: usize,
}

/// Rate of a deferred leakage annotation, by kind.
#[derive(Clone, Copy)]
pub(super) enum LeakChannel {
    Leak(f64),
    Seep(f64),
    Transport(f64),
}

/// Lower one leakage annotation onto the aliases its targets hold at `position`.
pub(super) fn push_deferred_leak_event(
    channel: &QecNoise,
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
    position: usize,
    pauli_before: usize,
    events: &mut Vec<QecDeferredLeakEvent>,
) -> Result<()> {
    let mut lowered = Vec::with_capacity(targets.len());
    for &target in targets {
        let Some(&alias) = aliases.get(target) else {
            return Err(PrismError::InvalidQubit {
                index: target,
                register_size: aliases.len(),
            });
        };
        lowered.push((alias, !measured_aliases[alias]));
    }
    let channel = match *channel {
        QecNoise::Leak(p) => LeakChannel::Leak(p),
        QecNoise::Seep(p) => LeakChannel::Seep(p),
        QecNoise::LeakTransport(p) => LeakChannel::Transport(p),
        _ => unreachable!("Pauli annotations lower through push_qec_deferred_noise_events"),
    };
    let first_herald = events
        .iter()
        .rev()
        .find(|event| matches!(event.channel, LeakChannel::Leak(_)))
        .map_or(0, |event| event.first_herald + event.targets.len());
    events.push(QecDeferredLeakEvent {
        channel,
        targets: lowered,
        position,
        pauli_before,
        first_herald,
    });
    Ok(())
}

/// Sampler for a Clifford program carrying leakage annotations.
pub(super) struct QecLeakageSampler {
    noiseless: CompiledSampler,
    deferred: QecDeferredProgram,
    num_heralds: usize,
    seed: u64,
}

/// One unit's output: X-frame flip, leak flag at measurement, and herald rows, each
/// `words` long.
struct UnitRecords {
    first_shot: usize,
    shots: usize,
    flips: Vec<u64>,
    forced: Vec<u64>,
    heralds: Vec<u64>,
}

impl QecLeakageSampler {
    pub(super) fn compile(program: &QecProgram) -> Result<Self> {
        let deferred = super::noise::lower_qec_program_to_deferred_circuit(program)?;
        let noiseless = compile_measurements(&deferred.to_circuit(), program.options().seed)?;
        let num_heralds = deferred
            .leak_events
            .iter()
            .filter(|event| matches!(event.channel, LeakChannel::Leak(_)))
            .map(|event| event.targets.len())
            .sum();
        Ok(Self {
            noiseless,
            deferred,
            num_heralds,
            seed: program.options().seed,
        })
    }

    pub(super) fn num_heralds(&self) -> usize {
        self.num_heralds
    }

    /// Sample shots `first_shot..first_shot + num_shots` of a `total_shots`-shot run:
    /// the measurement records and the herald columns, both measurement-major.
    pub(super) fn sample(
        &mut self,
        first_shot: usize,
        num_shots: usize,
        total_shots: usize,
    ) -> Result<(PackedShots, PackedShots)> {
        let num_measurements = self.deferred.measurement_qubits.len();
        let s_words = num_shots.div_ceil(64);
        let mut records = if num_measurements == 0 {
            Vec::new()
        } else {
            self.noiseless
                .try_sample_bulk_packed(num_shots)?
                .into_meas_major_data()
        };
        let mut heralds = vec![0u64; self.num_heralds * s_words];
        if num_shots > 0 {
            let first_unit = first_shot / LEAK_UNIT_SHOTS;
            let last_unit = (first_shot + num_shots - 1) / LEAK_UNIT_SHOTS;
            let run = |unit: usize| self.run_unit(unit, total_shots);
            #[cfg(feature = "parallel")]
            let units: Vec<UnitRecords> =
                (first_unit..=last_unit).into_par_iter().map(run).collect();
            #[cfg(not(feature = "parallel"))]
            let units: Vec<UnitRecords> = (first_unit..=last_unit).map(run).collect();

            for unit in &units {
                let lo = first_shot.max(unit.first_shot);
                let hi = (first_shot + num_shots).min(unit.first_shot + unit.shots);
                let words = unit.shots.div_ceil(64);
                let (src, dst, count) = (lo - unit.first_shot, lo - first_shot, hi - lo);
                for record in 0..num_measurements {
                    let row = &mut records[record * s_words..(record + 1) * s_words];
                    let unit_row = record * words..(record + 1) * words;
                    land_bits(
                        row,
                        dst,
                        &unit.flips[unit_row.clone()],
                        src,
                        count,
                        |a, b| a ^ b,
                    );
                    land_bits(row, dst, &unit.forced[unit_row], src, count, |a, b| a | b);
                }
                for herald in 0..self.num_heralds {
                    let row = &mut heralds[herald * s_words..(herald + 1) * s_words];
                    let unit_row = &unit.heralds[herald * words..(herald + 1) * words];
                    land_bits(row, dst, unit_row, src, count, |a, b| a | b);
                }
            }
        }
        Ok((
            PackedShots::from_meas_major(records, num_shots, num_measurements),
            PackedShots::from_meas_major(heralds, num_shots, self.num_heralds),
        ))
    }

    /// Push unit `unit` of a `total_shots`-shot run through the deferred circuit.
    fn run_unit(&self, unit: usize, total_shots: usize) -> UnitRecords {
        let first_shot = unit * LEAK_UNIT_SHOTS;
        let shots = (total_shots - first_shot).min(LEAK_UNIT_SHOTS);
        let words = shots.div_ceil(64);
        let deferred = &self.deferred;
        let num_aliases = deferred.num_aliases;

        let mut seed_rng = ChaCha8Rng::seed_from_u64(self.seed.wrapping_add(0x1EA4_0E1D));
        seed_rng.set_stream(unit as u64);
        let mut rng = Xoshiro256PlusPlus::from_chacha(&mut seed_rng);

        let mut frame = Frame::new(words, num_aliases);
        let mut leak = vec![0u64; num_aliases * words];
        let mut heralds = vec![0u64; self.num_heralds * words];

        let mut noise = deferred.noise_events.iter().enumerate().peekable();
        let mut leaks = deferred.leak_events.iter().peekable();
        let gate_count = deferred.gates.len();
        for position in 0..=gate_count {
            loop {
                let bound = leaks
                    .peek()
                    .filter(|event| event.position == position)
                    .map_or(usize::MAX, |event| event.pauli_before);
                while let Some((_, event)) =
                    noise.next_if(|&(index, event)| event.position == position && index < bound)
                {
                    apply_pauli_event(
                        event,
                        shots,
                        &deferred.pair_tables,
                        &mut rng,
                        |alias, shot| flag(&leak, words, alias, shot),
                        |alias, shot, letter| {
                            let s = frame.slot(alias);
                            frame.flip(s, shot, letter);
                        },
                    );
                }
                let Some(event) = leaks.next_if(|event| event.position == position) else {
                    break;
                };
                let mut erase = |alias: usize, shot: usize, rng: &mut Xoshiro256PlusPlus| {
                    let s = frame.slot(alias);
                    frame.flip(s, shot, (rng.next_u64() >> 62) as usize);
                };
                apply_leak_event(
                    event,
                    shots,
                    words,
                    &mut leak,
                    &mut heralds,
                    &mut rng,
                    &mut erase,
                );
            }
            if position == gate_count {
                break;
            }
            let gate = &deferred.gates[position];
            let targets = gate.targets();
            let mut slots = [0usize; 2];
            for (s, &alias) in slots.iter_mut().zip(targets) {
                *s = frame.slot(alias as usize);
            }
            frame.gate(&gate.gate, &slots[..targets.len()]);
            if let [a, b] = *targets {
                let (a, b) = (a as usize, b as usize);
                for word in 0..words {
                    let mask = leak[a * words + word] | leak[b * words + word];
                    if mask != 0 {
                        for s in [slots[0], slots[1]] {
                            frame.scramble(s, word, mask, rng.next_u64(), rng.next_u64());
                        }
                    }
                }
            }
        }

        let num_measurements = deferred.measurement_qubits.len();
        let mut flips = vec![0u64; num_measurements * words];
        let mut forced = vec![0u64; num_measurements * words];
        for (record, &alias) in deferred.measurement_qubits.iter().enumerate() {
            let s = frame.slot(alias);
            flips[record * words..(record + 1) * words].copy_from_slice(frame.x_row(s));
            forced[record * words..(record + 1) * words]
                .copy_from_slice(&leak[alias * words..(alias + 1) * words]);
        }
        UnitRecords {
            first_shot,
            shots,
            flips,
            forced,
            heralds,
        }
    }
}

/// Call `fire(shot)` for each of `shots` shots an event of rate `p` fires on.
#[inline(always)]
fn for_each_firing(
    p: f64,
    shots: usize,
    rng: &mut Xoshiro256PlusPlus,
    mut fire: impl FnMut(usize, &mut Xoshiro256PlusPlus),
) {
    if p <= 0.0 {
        return;
    }
    if p >= 0.5 || shots < 32 {
        for shot in 0..shots {
            if rng.next_f64() < p {
                fire(shot, rng);
            }
        }
        return;
    }
    let ln_1mp = (1.0 - p).ln();
    let mut shot = geometric_sample_xoshiro(rng, ln_1mp);
    while shot < shots {
        fire(shot, rng);
        shot += 1 + geometric_sample_xoshiro(rng, ln_1mp);
    }
}

/// Draw one Pauli annotation, calling `flip(alias, shot, letter)` with letter 1, 2, 3
/// for X, Y, Z. A firing that names a qubit `leaked` in that shot is dropped, as the
/// trajectory engines skip an event naming a leaked qubit.
fn apply_pauli_event(
    event: &QecDeferredNoiseEvent,
    shots: usize,
    pair_tables: &[[f64; 15]],
    rng: &mut Xoshiro256PlusPlus,
    leaked: impl Fn(usize, usize) -> bool,
    mut flip: impl FnMut(usize, usize, usize),
) {
    let mut flip_pair = |pair: &[usize], shot: usize, sample: usize| {
        if !leaked(pair[0], shot) && !leaked(pair[1], shot) {
            flip(pair[0], shot, sample / 4);
            flip(pair[1], shot, sample % 4);
        }
    };
    match event.channel {
        QecDeferredChannel::Depolarize2(p) => {
            for pair in event.targets.chunks_exact(2) {
                for_each_firing(p, shots, rng, |shot, rng| {
                    let sample = 1 + ((rng.next_f64() * 15.0) as usize).min(14);
                    flip_pair(pair, shot, sample);
                });
            }
        }
        QecDeferredChannel::PairTable { p, table } => {
            let rates = &pair_tables[table as usize];
            for pair in event.targets.chunks_exact(2) {
                for_each_firing(p, shots, rng, |shot, rng| {
                    let mut r = rng.next_f64() * p;
                    let branch = rates
                        .iter()
                        .position(|&rate| {
                            r -= rate;
                            r < 0.0
                        })
                        .unwrap_or(14);
                    flip_pair(pair, shot, branch + 1);
                });
            }
        }
        QecDeferredChannel::Single([px, py, pz]) => {
            let p = px + py + pz;
            for &alias in &event.targets {
                for_each_firing(p, shots, rng, |shot, rng| {
                    let r = rng.next_f64() * p;
                    let letter = if r < px {
                        1
                    } else if r < px + py {
                        2
                    } else {
                        3
                    };
                    if !leaked(alias, shot) {
                        flip(alias, shot, letter);
                    }
                });
            }
        }
    }
}

#[inline(always)]
fn flag(leak: &[u64], words: usize, alias: usize, shot: usize) -> bool {
    leak[alias * words + shot / 64] >> (shot % 64) & 1 == 1
}

#[inline(always)]
fn toggle(leak: &mut [u64], words: usize, alias: usize, shot: usize) {
    leak[alias * words + shot / 64] ^= 1 << (shot % 64);
}

/// Draw one leakage annotation against the flags in `leak`, calling
/// `erase(alias, shot, rng)` on each qubit that leaks or returns, and copy each `LEAK`
/// target's flag into its herald row.
fn apply_leak_event(
    event: &QecDeferredLeakEvent,
    shots: usize,
    words: usize,
    leak: &mut [u64],
    heralds: &mut [u64],
    rng: &mut Xoshiro256PlusPlus,
    erase: &mut impl FnMut(usize, usize, &mut Xoshiro256PlusPlus),
) {
    match event.channel {
        LeakChannel::Leak(p) => {
            for (offset, &(alias, live)) in event.targets.iter().enumerate() {
                if live {
                    for_each_firing(p, shots, rng, |shot, rng| {
                        if !flag(leak, words, alias, shot) {
                            toggle(leak, words, alias, shot);
                            erase(alias, shot, rng);
                        }
                    });
                }
                let herald = event.first_herald + offset;
                heralds[herald * words..(herald + 1) * words]
                    .copy_from_slice(&leak[alias * words..(alias + 1) * words]);
            }
        }
        LeakChannel::Seep(p) => {
            for &(alias, live) in &event.targets {
                if live {
                    for_each_firing(p, shots, rng, |shot, rng| {
                        if flag(leak, words, alias, shot) {
                            toggle(leak, words, alias, shot);
                            erase(alias, shot, rng);
                        }
                    });
                }
            }
        }
        LeakChannel::Transport(p) => {
            for pair in event.targets.chunks_exact(2) {
                let ((a, a_live), (b, b_live)) = (pair[0], pair[1]);
                if !(a_live && b_live) {
                    continue;
                }
                for_each_firing(p, shots, rng, |shot, rng| {
                    match (flag(leak, words, a, shot), flag(leak, words, b, shot)) {
                        (true, false) => {
                            toggle(leak, words, b, shot);
                            erase(b, shot, rng);
                        }
                        (false, true) => {
                            toggle(leak, words, a, shot);
                            erase(a, shot, rng);
                        }
                        _ => {}
                    }
                });
            }
        }
    }
}

/// Combine `count` bits of `src` from bit `src_bit` into `dst` from bit `dst_bit`
/// with `op`, a word at a time. `op(dst, 0)` must leave `dst` unchanged.
pub(super) fn land_bits(
    dst: &mut [u64],
    dst_bit: usize,
    src: &[u64],
    src_bit: usize,
    count: usize,
    op: impl Fn(u64, u64) -> u64,
) {
    let mut done = 0;
    while done < count {
        let n = (count - done).min(64);
        let bit = src_bit + done;
        let mut value = src[bit / 64] >> (bit % 64);
        if bit % 64 + n > 64 {
            value |= src[bit / 64 + 1] << (64 - bit % 64);
        }
        if n < 64 {
            value &= (1u64 << n) - 1;
        }
        let at = dst_bit + done;
        let lo = at % 64;
        dst[at / 64] = op(dst[at / 64], value << lo);
        if lo + n > 64 {
            dst[at / 64 + 1] = op(dst[at / 64 + 1], value >> (64 - lo));
        }
        done += n;
    }
}

/// Error for an engine that cannot run a program carrying leakage annotations.
pub(super) fn leakage_rejection(engine: &str) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: engine.to_string(),
        reason: "leakage annotations (`LEAK`, `SEEP`, `LEAK_TRANSPORT`) carry a per-shot leak \
                 flag that only `run_qec_program` samples"
            .to_string(),
    }
}
