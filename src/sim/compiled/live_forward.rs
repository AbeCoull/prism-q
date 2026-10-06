//! Forward stabilizer compile over live qubit slots.
//!
//! A qubit holds a tableau column from its first gate to its last measurement, so the
//! tableau is sized by the peak number of live qubits rather than by the register. Each
//! record is taken in record order as soon as every gate on its qubit has run, which
//! leaves the compiled rows identical to a compile that measures at the end: which
//! records come out random, and how the rest depend on them, is fixed by the record
//! order alone.

use super::propagation::{batch_propagate_backward_flat, forward_as_backward_gate};
use super::{CompiledSampler, finish_sampler};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;

const NONE: u32 = u32::MAX;

/// Gate order, measurement hoist points and peak liveness of a terminal-measurement
/// circuit.
pub(super) struct LiveSchedule<'a> {
    gates: Vec<(&'a Gate, &'a [usize])>,
    /// Qubit read by each record.
    measured: Vec<usize>,
    /// Gate index before which each record is taken; `gates.len()` for the tail.
    hoist: Vec<usize>,
    /// Whether the record is the last on a gated qubit, which frees its slot.
    retires: Vec<bool>,
    /// Most qubits live at once: gated, and not yet past their last record.
    pub(super) peak_live: usize,
}

impl<'a> LiveSchedule<'a> {
    pub(super) fn plan(circuit: &'a Circuit) -> Self {
        let n = circuit.num_qubits;
        let mut gates: Vec<(&Gate, &[usize])> = Vec::new();
        let mut first = vec![NONE; n];
        let mut last = vec![NONE; n];
        let mut measured = Vec::new();
        for inst in &circuit.instructions {
            match inst {
                Instruction::Gate { gate, targets }
                | Instruction::Conditional { gate, targets, .. } => {
                    let g = gates.len() as u32;
                    for &q in targets.iter() {
                        if first[q] == NONE {
                            first[q] = g;
                        }
                        last[q] = g;
                    }
                    gates.push((gate, targets.as_slice()));
                }
                Instruction::Measure { qubit, .. } => measured.push(*qubit),
                _ => {}
            }
        }

        let mut hoist = Vec::with_capacity(measured.len());
        let mut point = 0usize;
        for &q in &measured {
            if last[q] != NONE {
                point = point.max(last[q] as usize + 1);
            }
            hoist.push(point);
        }

        let mut last_record = vec![usize::MAX; n];
        for (j, &q) in measured.iter().enumerate() {
            last_record[q] = j;
        }
        let retires: Vec<bool> = measured
            .iter()
            .enumerate()
            .map(|(j, &q)| last_record[q] == j && first[q] != NONE)
            .collect();

        let mut created_at = vec![0usize; gates.len() + 1];
        for &f in &first {
            if f != NONE {
                created_at[f as usize] += 1;
            }
        }
        let mut retired_at = vec![0usize; gates.len() + 1];
        for (j, &h) in hoist.iter().enumerate() {
            if retires[j] {
                retired_at[h] += 1;
            }
        }
        let mut live = 0usize;
        let mut peak_live = 0usize;
        for g in 0..=gates.len() {
            live -= retired_at[g];
            live += created_at[g];
            peak_live = peak_live.max(live);
        }

        Self {
            gates,
            measured,
            hoist,
            retires,
            peak_live,
        }
    }
}

/// Column-major tableau over slots: column `c` holds words `c * words..`, destabilizer
/// row of pair `s` is bit `s` of the first `dw` words and its stabilizer row is bit `s`
/// of the last `dw`. `dep[k]` marks the stabilizer rows whose sign carries random bit `k`.
struct LiveTableau {
    dw: usize,
    words: usize,
    slots: usize,
    x: Vec<u64>,
    z: Vec<u64>,
    phase: Vec<u64>,
    dep: Vec<Vec<u64>>,
    col_of: Vec<u32>,
    free_cols: Vec<usize>,
    free_pairs: Vec<usize>,
    hits: Vec<u64>,
    rest: Vec<u64>,
    zs: Vec<u64>,
    rows: Vec<u64>,
    sum0: Vec<u64>,
    sum1: Vec<u64>,
    product_dep: Vec<u64>,
}

pub(super) fn compile_forward_live(
    circuit: &Circuit,
    seed: u64,
    schedule: &LiveSchedule,
) -> Result<CompiledSampler> {
    let m = schedule.measured.len();
    if m == 0 {
        return Ok(CompiledSampler::empty(seed));
    }
    let m_words = m.div_ceil(64);
    let slots = schedule.peak_live;
    let dw = slots.div_ceil(64).max(1);
    let words = 2 * dw;
    let mut t = LiveTableau {
        dw,
        words,
        slots,
        x: vec![0u64; slots * words],
        z: vec![0u64; slots * words],
        phase: vec![0u64; words],
        dep: Vec::new(),
        col_of: vec![NONE; circuit.num_qubits],
        free_cols: (0..slots).rev().collect(),
        free_pairs: (0..slots).rev().collect(),
        hits: vec![0u64; dw],
        rest: vec![0u64; dw],
        zs: vec![0u64; dw],
        rows: vec![0u64; words],
        sum0: vec![0u64; words],
        sum1: vec![0u64; words],
        product_dep: Vec::new(),
    };

    let mut flip_rows: Vec<Vec<u64>> = Vec::new();
    let mut ref_bits = vec![false; m];
    let mut rank = 0usize;
    let mut next = 0usize;
    let mut cols: SmallVec<[usize; 4]> = SmallVec::new();

    for (g, &(gate, targets)) in schedule.gates.iter().enumerate() {
        while next < m && schedule.hoist[next] <= g {
            t.measure(
                schedule,
                next,
                &mut flip_rows,
                &mut ref_bits,
                &mut rank,
                m_words,
            );
            next += 1;
        }
        let backward = forward_as_backward_gate(gate)?;
        cols.clear();
        for &q in targets {
            if t.col_of[q] == NONE {
                t.create(q);
            }
            cols.push(t.col_of[q] as usize);
        }
        batch_propagate_backward_flat(&mut t.x, &mut t.z, &mut t.phase, words, backward, &cols);
    }
    while next < m {
        t.measure(
            schedule,
            next,
            &mut flip_rows,
            &mut ref_bits,
            &mut rank,
            m_words,
        );
        next += 1;
    }

    Ok(finish_sampler(flip_rows, rank, m, &ref_bits, seed))
}

#[inline]
fn first_set(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .position(|&w| w != 0)
        .map(|w| w * 64 + words[w].trailing_zeros() as usize)
}

#[inline]
fn mask(bit: u64) -> u64 {
    0u64.wrapping_sub(bit)
}

impl LiveTableau {
    fn create(&mut self, qubit: usize) {
        let c = self
            .free_cols
            .pop()
            .expect("slot count covers peak liveness");
        let s = self
            .free_pairs
            .pop()
            .expect("slot count covers peak liveness");
        let base = c * self.words;
        self.x[base + s / 64] |= 1u64 << (s % 64);
        self.z[base + self.dw + s / 64] |= 1u64 << (s % 64);
        self.col_of[qubit] = c as u32;
    }

    fn measure(
        &mut self,
        schedule: &LiveSchedule,
        record: usize,
        flip_rows: &mut Vec<Vec<u64>>,
        ref_bits: &mut [bool],
        rank: &mut usize,
        m_words: usize,
    ) {
        let qubit = schedule.measured[record];
        let c = self.col_of[qubit];
        if c == NONE {
            return;
        }
        let c = c as usize;
        let retire = schedule.retires[record];
        let base = c * self.words;
        match first_set(&self.x[base + self.dw..base + self.words]) {
            Some(p) => {
                let k = *rank;
                *rank += 1;
                flip_rows.push(vec![0u64; m_words]);
                flip_rows[k][record / 64] |= 1u64 << (record % 64);
                self.measure_random(c, p, k, retire);
            }
            None => {
                let sign = self.measure_deterministic(c, *rank, retire);
                ref_bits[record] = sign;
                for (k, row) in flip_rows.iter_mut().enumerate() {
                    if (self.product_dep[k / 64] >> (k % 64)) & 1 != 0 {
                        row[record / 64] |= 1u64 << (record % 64);
                    }
                }
            }
        }
        if retire {
            self.x[base..base + self.words].fill(0);
            self.z[base..base + self.words].fill(0);
            self.col_of[qubit] = NONE;
            self.free_cols.push(c);
        }
    }

    /// Stabilizer row `p` anticommutes with `Z_c`: multiply it into every other row with
    /// `X_c`, make it the new destabilizer of its pair and leave `Z_c` as the stabilizer
    /// carrying random bit `k`. With `retire`, the pair is cleared instead and the bit
    /// folds into the rows that still hold `Z_c`.
    fn measure_random(&mut self, c: usize, p: usize, k: usize, retire: bool) {
        let (dw, words) = (self.dw, self.words);
        let base = c * words;
        let (pw, pb) = (p / 64, p % 64);
        let pbit = 1u64 << pb;

        self.rows.copy_from_slice(&self.x[base..base + words]);
        self.rows[dw + pw] &= !pbit;
        let p_phase = (self.phase[dw + pw] >> pb) & 1;

        for dep in &mut self.dep[..k] {
            if (dep[pw] >> pb) & 1 != 0 {
                for (d, &r) in dep.iter_mut().zip(&self.rows[dw..]) {
                    *d ^= r;
                }
            }
            dep[pw] &= !pbit;
        }

        self.sum0.fill(0);
        self.sum1.fill(0);
        for c2 in 0..self.slots {
            let b = c2 * words;
            let xp = (self.x[b + dw + pw] >> pb) & 1;
            let zp = (self.z[b + dw + pw] >> pb) & 1;
            if xp | zp != 0 {
                let (x1, z1) = (mask(xp), mask(zp));
                for w in 0..words {
                    let x2 = self.x[b + w];
                    let z2 = self.z[b + w];
                    let r = self.rows[w];
                    let nx = x2 ^ (x1 & r);
                    let nz = z2 ^ (z1 & r);
                    let nonzero = (nx | nz) & (x2 | z2) & r;
                    let pos =
                        ((x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2)) & r;
                    let minus = nonzero & !pos;
                    self.sum1[w] ^= minus ^ (self.sum0[w] & nonzero);
                    self.sum0[w] ^= nonzero;
                    self.x[b + w] = nx;
                    self.z[b + w] = nz;
                }
            }
            if retire {
                self.x[b + pw] &= !pbit;
                self.x[b + dw + pw] &= !pbit;
                self.z[b + pw] &= !pbit;
                self.z[b + dw + pw] &= !pbit;
            } else {
                self.x[b + pw] = (self.x[b + pw] & !pbit) | (xp << pb);
                self.z[b + pw] = (self.z[b + pw] & !pbit) | (zp << pb);
                self.x[b + dw + pw] &= !pbit;
                self.z[b + dw + pw] = (self.z[b + dw + pw] & !pbit) | (((c2 == c) as u64) << pb);
            }
        }

        let flip = mask(p_phase);
        for w in 0..words {
            self.phase[w] ^= (self.sum1[w] ^ flip) & self.rows[w];
        }

        if retire {
            self.phase[pw] &= !pbit;
            self.phase[dw + pw] &= !pbit;
            self.zs.copy_from_slice(&self.z[base + dw..base + words]);
            self.zs[pw] &= !pbit;
            self.dep.push(self.zs.clone());
            self.free_pairs.push(p);
        } else {
            self.phase[pw] = (self.phase[pw] & !pbit) | (p_phase << pb);
            self.phase[dw + pw] &= !pbit;
            let mut own = vec![0u64; dw];
            own[pw] = pbit;
            self.dep.push(own);
        }
    }

    /// `Z_c` is the product of the stabilizers whose destabilizer holds `X_c`. Returns the
    /// product's sign and leaves its random-bit dependence in `product_dep`. With
    /// `retire`, that product replaces the lowest of those stabilizers, folds into every
    /// other row holding `Z_c`, and its pair is cleared.
    fn measure_deterministic(&mut self, c: usize, rank: usize, retire: bool) -> bool {
        let (dw, words) = (self.dw, self.words);
        let base = c * words;
        self.hits.copy_from_slice(&self.x[base..base + dw]);
        let g0 = first_set(&self.hits).expect("a live qubit has a destabilizer with X");
        let (gw, gb) = (g0 / 64, g0 % 64);
        let gbit = 1u64 << gb;
        self.rest.copy_from_slice(&self.hits);
        self.rest[gw] &= !gbit;

        let mut phase_count = 0u32;
        for w in 0..dw {
            phase_count += (self.phase[dw + w] & self.hits[w]).count_ones();
        }
        let mut y_count = 0u32;
        let mut pair_parity = 0u32;
        for c2 in 0..self.slots {
            let b = c2 * words;
            let mut any = 0u64;
            for w in 0..dw {
                any |= (self.x[b + dw + w] | self.z[b + dw + w]) & self.hits[w];
            }
            if any != 0 {
                let mut below = 0u32;
                for w in 0..dw {
                    let xh = self.x[b + dw + w] & self.hits[w];
                    let zh = self.z[b + dw + w] & self.hits[w];
                    y_count += (xh & zh).count_ones();
                    let mut bits = xh;
                    while bits != 0 {
                        let i = bits.trailing_zeros();
                        pair_parity ^= (below + (zh & ((1u64 << i) - 1)).count_ones()) & 1;
                        bits &= bits - 1;
                    }
                    below += zh.count_ones();
                }
            }
            if retire {
                if (self.x[b + gw] >> gb) & 1 != 0 {
                    for w in 0..dw {
                        self.x[b + w] ^= self.rest[w];
                    }
                }
                if (self.z[b + gw] >> gb) & 1 != 0 {
                    for w in 0..dw {
                        self.z[b + w] ^= self.rest[w];
                    }
                }
                self.x[b + gw] &= !gbit;
                self.x[b + dw + gw] &= !gbit;
                self.z[b + gw] &= !gbit;
                self.z[b + dw + gw] &= !gbit;
            }
        }
        let sign = (phase_count + pair_parity + (y_count >> 1)) & 1 != 0;

        self.product_dep.clear();
        self.product_dep.resize(rank.div_ceil(64), 0);
        for (k, dep) in self.dep.iter().enumerate() {
            let mut parity = 0u64;
            for (d, h) in dep.iter().zip(&self.hits) {
                parity ^= d & h;
            }
            self.product_dep[k / 64] |= ((parity.count_ones() & 1) as u64) << (k % 64);
        }

        if retire {
            self.zs.copy_from_slice(&self.z[base + dw..base + words]);
            if sign {
                for w in 0..dw {
                    self.phase[dw + w] ^= self.zs[w];
                }
            }
            for (k, dep) in self.dep.iter_mut().enumerate() {
                if (self.product_dep[k / 64] >> (k % 64)) & 1 != 0 {
                    for (d, &z) in dep.iter_mut().zip(&self.zs) {
                        *d ^= z;
                    }
                }
                dep[gw] &= !gbit;
            }
            self.phase[gw] &= !gbit;
            self.phase[dw + gw] &= !gbit;
            self.free_pairs.push(g0);
        }
        sign
    }
}
