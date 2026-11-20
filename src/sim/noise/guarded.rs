//! Noisy compiled sampling of a deferred circuit whose guarded entries land on the shots
//! their guards hold.

use rand_chacha::ChaCha8Rng;

use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::sim::ShotsResult;
use crate::sim::compiled::{
    DeferredCircuit, Guard, GuardedEvent, GuardedKind, PackedShots, batch_propagate_backward,
    shot_tail_mask, xor_words,
};

use super::compiled::{
    apply_readout_packed, compile_validated_noisy, for_each_event_shot, geometric_sample,
    unpack_and_remap,
};

/// Sample a deferred circuit with guarded entries. The compiled sampler draws every
/// shot's records under the every-shot events, then each guarded entry, in execution
/// order, lands on the shots whose guard holds, then readout. A guard reads records
/// measured before its entry, which only earlier guarded entries can still flip.
///
/// Entries land on shot rows, except when a guarded gate is among them: a gate lands on
/// every shot its guard holds, which measurement rows evaluate a word of 64 shots at a
/// time.
pub(crate) fn run_shots_noisy_guarded(
    deferred: &DeferredCircuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let circuit = &deferred.circuit;
    let mut sampler = compile_validated_noisy(circuit, &deferred.noise, seed)?;
    let readout = std::mem::take(&mut sampler.readout);
    let (flips, num_measurements) = guarded_flips(circuit, &deferred.guarded);

    let (mut accum, m_words) = sampler.sample_bulk_words_shot_major_cpu(num_shots);
    if flips
        .iter()
        .any(|flip| matches!(flip.kind, GuardedKind::Gate(_)))
    {
        let mut rows =
            PackedShots::from_shot_major(accum, num_shots, num_measurements).into_meas_major_data();
        let mut fire = vec![0u64; num_shots.div_ceil(64)];
        for flip in &flips {
            flip.land_rows(&mut rows, num_shots, &mut fire, &mut sampler.rng);
        }
        accum =
            PackedShots::from_meas_major(rows, num_shots, num_measurements).into_shot_major_data();
    } else {
        let mut shot_rows = ShotRows {
            words: &mut accum,
            m_words,
        };
        for flip in &flips {
            flip.land(&mut shot_rows, num_shots, &mut sampler.rng);
        }
    }
    apply_readout_packed(&mut accum, num_shots, m_words, &readout, &mut sampler.rng);

    let shots = unpack_and_remap(
        &accum,
        m_words,
        num_shots,
        &circuit.classical_bit_order(),
        circuit.num_classical_bits,
    );
    Ok(
        ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(
            crate::sim::RunMetadata::exact(crate::sim::ResolvedBackend::CompiledStabilizer)
                .with_engine(crate::sim::Engine::NoisyCompiledSampler)
                .with_reason(crate::sim::RouteRule::CompiledCliffordSampler),
        ),
    )
}

/// Packed records a guarded entry reads and flips one shot at a time.
trait RecordRows {
    fn bit(&self, shot: usize, record: usize) -> bool;
    /// XOR `flips`, packed one bit per record, into shot `shot`.
    fn xor_flips(&mut self, shot: usize, flips: &[u64]);
}

/// Shot-major records: `m_words` words per shot.
struct ShotRows<'a> {
    words: &'a mut [u64],
    m_words: usize,
}

impl RecordRows for ShotRows<'_> {
    #[inline(always)]
    fn bit(&self, shot: usize, record: usize) -> bool {
        (self.words[shot * self.m_words + record / 64] >> (record % 64)) & 1 == 1
    }

    #[inline(always)]
    fn xor_flips(&mut self, shot: usize, flips: &[u64]) {
        let start = shot * self.m_words;
        xor_words(&mut self.words[start..start + self.m_words], flips);
    }
}

/// Measurement-major records: `s_words` words per record.
struct MeasRows<'a> {
    words: &'a mut [u64],
    s_words: usize,
}

impl RecordRows for MeasRows<'_> {
    #[inline(always)]
    fn bit(&self, shot: usize, record: usize) -> bool {
        (self.words[record * self.s_words + shot / 64] >> (shot % 64)) & 1 == 1
    }

    #[inline(always)]
    fn xor_flips(&mut self, shot: usize, flips: &[u64]) {
        let bit = 1u64 << (shot % 64);
        for_each_set_bit(flips, |record| {
            self.words[record * self.s_words + shot / 64] ^= bit;
        });
    }
}

#[inline(always)]
fn for_each_set_bit(words: &[u64], mut f: impl FnMut(usize)) {
    for (w, &word) in words.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            f(w * 64 + rest.trailing_zeros() as usize);
            rest &= rest - 1;
        }
    }
}

/// A guarded entry's record flips: an X error flips `x_flips`, a Z error `z_flips`, a Y
/// both. A gate's flips are folded into `x_flips`, as are a readout flip's.
struct GuardedFlip {
    guard: Guard,
    kind: GuardedKind,
    x_flips: Vec<u64>,
    z_flips: Vec<u64>,
}

impl GuardedFlip {
    /// Land the entry on each of `num_shots` shots of `rows` where its guard holds,
    /// drawing a noise entry per shot.
    fn land(&self, rows: &mut impl RecordRows, num_shots: usize, rng: &mut ChaCha8Rng) {
        match self.kind {
            GuardedKind::Gate(_) => {
                for shot in 0..num_shots {
                    if self.guard.holds(|record| rows.bit(shot, record)) {
                        rows.xor_flips(shot, &self.x_flips);
                    }
                }
            }
            GuardedKind::Noise(probs) => self.land_noise(rows, num_shots, probs, rng),
            GuardedKind::Readout { record, p01, p10 } => {
                self.land_readout(rows, num_shots, record, p01, p10, rng)
            }
        }
    }

    /// Flip `record` and the records its alias reaches at the rate its value picks, with
    /// the draws of [`apply_readout_packed`].
    fn land_readout(
        &self,
        rows: &mut impl RecordRows,
        num_shots: usize,
        record: usize,
        p01: f64,
        p10: f64,
        rng: &mut ChaCha8Rng,
    ) {
        let p_max = p01.max(p10);
        if p_max >= 0.5 || num_shots < 32 {
            for shot in 0..num_shots {
                let p = if rows.bit(shot, record) { p10 } else { p01 };
                let r: f64 = rand::RngExt::random(rng);
                if r < p {
                    rows.xor_flips(shot, &self.x_flips);
                }
            }
            return;
        }
        let ln_1mp = (-p_max).ln_1p();
        let accept_set = p10 / p_max;
        let accept_clear = p01 / p_max;
        let mut shot = geometric_sample(rng, ln_1mp);
        while shot < num_shots {
            let accept = if rows.bit(shot, record) {
                accept_set
            } else {
                accept_clear
            };
            let r: f64 = rand::RngExt::random(rng);
            if r < accept {
                rows.xor_flips(shot, &self.x_flips);
            }
            shot += 1 + geometric_sample(rng, ln_1mp);
        }
    }

    fn land_noise(
        &self,
        rows: &mut impl RecordRows,
        num_shots: usize,
        (px, py, pz): (f64, f64, f64),
        rng: &mut ChaCha8Rng,
    ) {
        let p_event = px + py + pz;
        for_each_event_shot(rng, p_event, 0, num_shots, rows, |rows, shot, r, dense| {
            let branch = if dense {
                if r >= p_event {
                    return;
                }
                r
            } else {
                r * p_event
            };
            if !self.guard.holds(|record| rows.bit(shot, record)) {
                return;
            }
            if branch < px + py {
                rows.xor_flips(shot, &self.x_flips);
            }
            if branch >= px {
                rows.xor_flips(shot, &self.z_flips);
            }
        });
    }

    /// [`Self::land`] on measurement-major `rows`, a gate a word of shots at a time
    /// through `fire`, one word per 64 shots.
    fn land_rows(
        &self,
        rows: &mut [u64],
        num_shots: usize,
        fire: &mut [u64],
        rng: &mut ChaCha8Rng,
    ) {
        let s_words = fire.len();
        if !matches!(self.kind, GuardedKind::Gate(_)) {
            let mut meas_rows = MeasRows {
                words: rows,
                s_words,
            };
            return self.land(&mut meas_rows, num_shots, rng);
        }
        if s_words == 0 {
            return;
        }
        let row = |record: usize| record * s_words..(record + 1) * s_words;
        match &self.guard {
            Guard::Parity { records, invert } => {
                fire.fill(if *invert { u64::MAX } else { 0 });
                for &record in records {
                    xor_words(fire, &rows[row(record)]);
                }
            }
            Guard::All { terms, negate } => {
                fire.fill(u64::MAX);
                for &(record, want) in terms {
                    let flip = if want { 0 } else { u64::MAX };
                    for (word, &bits) in fire.iter_mut().zip(&rows[row(record)]) {
                        *word &= bits ^ flip;
                    }
                }
                if *negate {
                    for word in fire.iter_mut() {
                        *word = !*word;
                    }
                }
            }
        }
        fire[s_words - 1] &= shot_tail_mask(num_shots);
        for_each_set_bit(&self.x_flips, |record| {
            xor_words(&mut rows[row(record)], fire);
        });
    }
}

/// The record flips of each guarded entry, in order, from one backward sweep of
/// `circuit`, whose measurements are all terminal, and the number of measurements.
fn guarded_flips(circuit: &Circuit, guarded: &[GuardedEvent]) -> (Vec<GuardedFlip>, usize) {
    let num_measurements = circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Measure { .. }))
        .count();
    let m_words = num_measurements.div_ceil(64);
    let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; circuit.num_qubits];
    let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; circuit.num_qubits];
    let mut sign_packed: Vec<u64> = vec![0u64; m_words];

    let mut flips = Vec::with_capacity(guarded.len());
    let mut pending = guarded.iter().rev().peekable();
    let mut mi = num_measurements;
    for (instr_idx, inst) in circuit.instructions.iter().enumerate().rev() {
        while let Some(event) = pending.next_if(|event| event.anchor == instr_idx) {
            let mut x_flips = z_packed[event.qubit].clone();
            let mut z_flips = x_packed[event.qubit].clone();
            match event.kind {
                GuardedKind::Gate(1) | GuardedKind::Readout { .. } => z_flips.fill(0),
                GuardedKind::Gate(2) => xor_words(&mut x_flips, &z_flips),
                GuardedKind::Gate(_) => std::mem::swap(&mut x_flips, &mut z_flips),
                GuardedKind::Noise(_) => {}
            }
            flips.push(GuardedFlip {
                guard: event.guard.clone(),
                kind: event.kind,
                x_flips,
                z_flips,
            });
        }
        match inst {
            Instruction::Gate { gate, targets } => {
                batch_propagate_backward(
                    &mut x_packed,
                    &mut z_packed,
                    &mut sign_packed,
                    gate,
                    targets.as_slice(),
                    m_words,
                );
            }
            Instruction::Measure { qubit, .. } => {
                mi -= 1;
                z_packed[*qubit][mi / 64] |= 1u64 << (mi % 64);
            }
            _ => {}
        }
    }
    flips.reverse();
    (flips, num_measurements)
}
