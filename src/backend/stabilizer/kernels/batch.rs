//! Word-group gate batching and measurement-run batching for large tableaus,
//! amortizing full-tableau row sweeps once rows outgrow cache.

use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::Gate;

#[cfg(feature = "parallel")]
use super::MIN_QUBITS_FOR_PAR_GATES;
use super::StabilizerBackend;

/// Minimum number of u64 words for word-group gate batching to be profitable.
///
/// Below this threshold, the tableau fits in L1/L2 cache and the per-gate
/// match overhead in the batched inner loop exceeds the cache-amortization
/// benefit. At nw=4 (n=193+), each row is 64 bytes (one cache line) and
/// word-group batching avoids repeated full-row iteration per gate.
pub(crate) const MIN_WORDS_FOR_BATCH: usize = 4;

/// Minimum run of consecutive measurements to route through
/// `batch_measure_ref_info`. Individual measurements pay two full-tableau
/// scans each (O(m * n) row touches, cache-hostile at large n); the batch
/// path builds its index in one pass. Below this length the index pass
/// costs more than the scans it replaces.
pub(crate) const MIN_MEASURES_FOR_BATCH: usize = 8;

/// A run of `CX(k, k + 1)` at least this long takes the chain kernel, one pass
/// over the rows with the prefix parity carried across words, instead of a
/// word-group flush per word it crosses.
pub(crate) const MIN_CX_CHAIN: usize = 16;

/// Length of the leading run of `CX(q, q + 1)`, `CX(q + 1, q + 2)`, ... in
/// `instructions`, zero when it does not start with one.
pub(crate) fn cx_chain_len(instructions: &[Instruction]) -> usize {
    let mut len = 0;
    let mut expect: Option<usize> = None;
    for instruction in instructions {
        let Instruction::Gate {
            gate: Gate::Cx,
            targets,
        } = instruction
        else {
            break;
        };
        let [c, t] = targets.as_slice() else { break };
        if *t != c + 1 || expect.is_some_and(|q| q != *c) {
            break;
        }
        expect = Some(*t);
        len += 1;
    }
    len
}

/// Compact gate representation for batched word-group execution.
///
/// All gates in a word group target the same u64 word. `a_bit` and `b_bit`
/// are bit positions (0..63) within that word.
#[derive(Clone, Copy)]
pub(crate) struct BatchGate {
    kind: u8,
    a_bit: u8,
    b_bit: u8,
}

impl BatchGate {
    const ID: u8 = 0;
    const H: u8 = 1;
    const S: u8 = 2;
    const SDG: u8 = 3;
    const X: u8 = 4;
    const Y: u8 = 5;
    const Z: u8 = 6;
    const SX: u8 = 7;
    const SXDG: u8 = 8;
    const CX: u8 = 9;
    const CZ: u8 = 10;
    const SWAP: u8 = 11;
}

/// Buffered cross-word 2q gate for deferred application.
#[derive(Clone, Copy)]
struct CrossWordGate {
    kind: u8,
    w0: u16,
    w1: u16,
    b0: u8,
    b1: u8,
}

/// Per-type bitmasks for wordwise 1q gate application.
///
/// Each mask has bits set for the target positions of that gate type.
/// All masks are mutually exclusive (no bit set in more than one mask),
/// guaranteeing independent application in any order.
#[derive(Clone, Copy)]
struct OneMasks {
    h: u64,
    s: u64,
    sdg: u64,
    x: u64,
    y: u64,
    z: u64,
    sx: u64,
    sxdg: u64,
}

impl Default for OneMasks {
    #[inline(always)]
    fn default() -> Self {
        Self {
            h: 0,
            s: 0,
            sdg: 0,
            x: 0,
            y: 0,
            z: 0,
            sx: 0,
            sxdg: 0,
        }
    }
}

/// Pre-processed operation: a batch of 1q masks, a single 2q gate, or a run of
/// `CX(k, k + 1)` for `k` in `first..last` applied as one word operation.
#[derive(Clone, Copy)]
enum PrepOp {
    Masks(OneMasks),
    Gate2q(BatchGate),
    CxChain { first: u8, last: u8 },
}

/// Build a prepared operation sequence from a batch of gates.
///
/// Groups consecutive 1q gates with non-overlapping targets into `OneMasks`
/// segments. 2q gates and bit-conflicting 1q gates trigger segment boundaries.
/// The resulting sequence preserves gate ordering.
fn prepare_word_ops(gates: &[BatchGate]) -> Vec<PrepOp> {
    let mut ops = Vec::with_capacity(gates.len() / 4 + 2);
    let mut masks = OneMasks::default();
    let mut used = 0u64;
    let mut has_masks = false;

    for g in gates {
        if g.kind >= BatchGate::CX {
            if has_masks {
                ops.push(PrepOp::Masks(masks));
                masks = OneMasks::default();
                used = 0;
                has_masks = false;
            }
            if g.kind == BatchGate::CX && g.b_bit == g.a_bit + 1 {
                match ops.last_mut() {
                    Some(PrepOp::CxChain { last, .. }) if *last == g.a_bit => {
                        *last = g.b_bit;
                        continue;
                    }
                    Some(PrepOp::Gate2q(prev))
                        if prev.kind == BatchGate::CX
                            && prev.b_bit == prev.a_bit + 1
                            && prev.b_bit == g.a_bit =>
                    {
                        let first = prev.a_bit;
                        *ops.last_mut().unwrap() = PrepOp::CxChain {
                            first,
                            last: g.b_bit,
                        };
                        continue;
                    }
                    _ => {}
                }
            }
            ops.push(PrepOp::Gate2q(*g));
        } else {
            if g.kind == BatchGate::ID {
                continue;
            }
            let bit = 1u64 << g.a_bit;
            if used & bit != 0 {
                ops.push(PrepOp::Masks(masks));
                masks = OneMasks::default();
                used = 0;
            }
            used |= bit;
            has_masks = true;
            match g.kind {
                BatchGate::H => masks.h |= bit,
                BatchGate::S => masks.s |= bit,
                BatchGate::SDG => masks.sdg |= bit,
                BatchGate::X => masks.x |= bit,
                BatchGate::Y => masks.y |= bit,
                BatchGate::Z => masks.z |= bit,
                BatchGate::SX => masks.sx |= bit,
                _ => masks.sxdg |= bit,
            }
        }
    }
    if has_masks {
        ops.push(PrepOp::Masks(masks));
    }
    ops
}

/// Apply wordwise 1q masks to a single (xw, zw, phase) tuple.
///
/// Each gate type operates only on its masked bits. Since all masks are
/// non-overlapping, the order of type processing is irrelevant.
#[inline(always)]
fn apply_1q_masks(xw: &mut u64, zw: &mut u64, p: &mut bool, m: &OneMasks) {
    if m.h != 0 {
        *p ^= (*xw & *zw & m.h).count_ones() & 1 != 0;
        let tmp = *xw & m.h;
        *xw = (*xw & !m.h) | (*zw & m.h);
        *zw = (*zw & !m.h) | tmp;
    }
    if m.s != 0 {
        *p ^= (*xw & *zw & m.s).count_ones() & 1 != 0;
        *zw ^= *xw & m.s;
    }
    if m.sdg != 0 {
        *zw ^= *xw & m.sdg;
        *p ^= (*xw & *zw & m.sdg).count_ones() & 1 != 0;
    }
    if m.x != 0 {
        *p ^= (*zw & m.x).count_ones() & 1 != 0;
    }
    if m.y != 0 {
        *p ^= ((*xw ^ *zw) & m.y).count_ones() & 1 != 0;
    }
    if m.z != 0 {
        *p ^= (*xw & m.z).count_ones() & 1 != 0;
    }
    if m.sx != 0 {
        *p ^= (*zw & !*xw & m.sx).count_ones() & 1 != 0;
        *xw ^= *zw & m.sx;
    }
    if m.sxdg != 0 {
        *p ^= (*xw & *zw & m.sxdg).count_ones() & 1 != 0;
        *xw ^= *zw & m.sxdg;
    }
}

/// `CX(k, k + 1)` for `k` in `first..last`, in that order, as word operations.
///
/// Gate `k` reads control `k` after the gates before it and target `k + 1`
/// before any gate touches it, so the control's X bit is the prefix parity of
/// the original X bits from `first`, the target's bits are the originals, and
/// the control's Z bit is still the original. That fixes every phase term from
/// the input words alone; the X range then becomes the prefix parity and each
/// control's Z bit takes the XOR with its target's.
#[inline(always)]
fn apply_cx_chain(xw: &mut u64, zw: &mut u64, p: &mut bool, first: u8, last: u8) {
    let range = (u64::MAX >> (63 - last)) & (u64::MAX << first);
    let controls = range & !(1u64 << last);
    let mut prefix = *xw & range;
    prefix ^= prefix << 1;
    prefix ^= prefix << 2;
    prefix ^= prefix << 4;
    prefix ^= prefix << 8;
    prefix ^= prefix << 16;
    prefix ^= prefix << 32;
    prefix &= range;
    let z = *zw;
    let x = *xw;
    let terms = prefix & controls & (z >> 1) & !((x >> 1) ^ z);
    *p ^= terms.count_ones() & 1 != 0;
    *xw = (x & !range) | prefix;
    *zw = (z & !controls) | ((z ^ (z >> 1)) & controls);
}

/// Apply a pre-computed operation sequence to a single (xw, zw, phase) tuple.
#[inline(always)]
fn apply_prepared_ops(xw: &mut u64, zw: &mut u64, p: &mut bool, ops: &[PrepOp]) {
    for op in ops {
        match op {
            PrepOp::Masks(m) => apply_1q_masks(xw, zw, p, m),
            PrepOp::CxChain { first, last } => apply_cx_chain(xw, zw, p, *first, *last),
            PrepOp::Gate2q(g) => {
                let mask_a = 1u64 << g.a_bit;
                match g.kind {
                    BatchGate::CX => {
                        let mask_b = 1u64 << g.b_bit;
                        let xa = (*xw >> g.a_bit) & 1;
                        let za = (*zw >> g.a_bit) & 1;
                        let xb = (*xw >> g.b_bit) & 1;
                        let zb = (*zw >> g.b_bit) & 1;
                        *p ^= (xa & zb & (xb ^ za ^ 1)) == 1;
                        if xa == 1 {
                            *xw ^= mask_b;
                        }
                        if zb == 1 {
                            *zw ^= mask_a;
                        }
                    }
                    BatchGate::CZ => {
                        let mask_b = 1u64 << g.b_bit;
                        let xa = (*xw >> g.a_bit) & 1;
                        let xb = (*xw >> g.b_bit) & 1;
                        let za = (*zw >> g.a_bit) & 1;
                        let zb = (*zw >> g.b_bit) & 1;
                        *p ^= (xa & xb & (za ^ zb)) == 1;
                        if xb == 1 {
                            *zw ^= mask_a;
                        }
                        if xa == 1 {
                            *zw ^= mask_b;
                        }
                    }
                    _ => {
                        let mask_b = 1u64 << g.b_bit;
                        let xa = (*xw >> g.a_bit) & 1;
                        let xb = (*xw >> g.b_bit) & 1;
                        if xa != xb {
                            *xw ^= mask_a | mask_b;
                        }
                        let za = (*zw >> g.a_bit) & 1;
                        let zb = (*zw >> g.b_bit) & 1;
                        if za != zb {
                            *zw ^= mask_a | mask_b;
                        }
                    }
                }
            }
        }
    }
}

/// The rows one batched gate pass sweeps, borrowed from whichever tableau owns
/// them. `row_start` is the first row a gate updates: zero for a full tableau,
/// `n` when destabilizers are deferred.
pub(crate) struct GateRows<'a> {
    pub(crate) xz: &'a mut [u64],
    pub(crate) phase: &'a mut [bool],
    pub(crate) n: usize,
    pub(crate) num_words: usize,
    pub(crate) row_start: usize,
}

impl GateRows<'_> {
    #[inline(always)]
    fn stride(&self) -> usize {
        2 * self.num_words
    }

    /// `CX(k, k + 1)` for `k` in `start..end`, in one pass over the rows.
    ///
    /// Each word takes [`apply_cx_chain`] with the prefix parity of the word
    /// before it carried into its low bit, and the gate across the boundary is
    /// applied between the two: its control is that carry, its target's bits
    /// are read from the next word before anything touches them, and its
    /// control's Z bit is still the original because the in-word chain never
    /// writes the last bit of the range. A word with nothing in it once the
    /// carry lands, and no Z bit waiting in the next word's low position, is
    /// skipped whole.
    fn apply_cx_chain_run(&mut self, start: usize, end: usize) {
        let stride = self.stride();
        let nw = self.num_words;
        let gs = self.row_start;
        let (ws, we) = (start / 64, end / 64);
        let (first, last) = ((start % 64) as u8, (end % 64) as u8);

        let process_row = |row: &mut [u64], p: &mut bool| {
            let mut carry = 0u64;
            for w in ws..=we {
                let lo = if w == ws { first } else { 0 };
                let hi = if w == we { last } else { 63 };
                let z_next = if w < we { row[nw + w + 1] & 1 } else { 0 };
                let mut xw = row[w] ^ carry;
                let mut zw = row[nw + w];
                if xw | zw == 0 && z_next == 0 {
                    if carry != 0 {
                        row[w] = 0;
                    }
                    carry = 0;
                    continue;
                }
                apply_cx_chain(&mut xw, &mut zw, p, lo, hi);
                if w < we {
                    let x_next = row[w + 1] & 1;
                    let control = xw >> 63;
                    let z_control = zw >> 63;
                    *p ^= (control & z_next & !(x_next ^ z_control)) & 1 != 0;
                    zw ^= z_next << 63;
                    carry = control;
                }
                row[w] = xw;
                row[nw + w] = zw;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz[gs * stride..]
                .par_chunks_mut(stride)
                .zip(self.phase[gs..].par_iter_mut())
                .for_each(|(row, p)| process_row(row, p));
            return;
        }

        for (row, p) in self.xz[gs * stride..]
            .chunks_mut(stride)
            .zip(self.phase[gs..].iter_mut())
        {
            process_row(row, p);
        }
    }

    /// Execute all gates in a word group against every tableau row.
    ///
    /// Loads each row's X-word and Z-word once, applies all gates in the group,
    /// then stores. This amortizes cache line loads across multiple gate ops.
    fn flush_word_group(&mut self, word: usize, gates: &[BatchGate]) {
        if gates.is_empty() {
            return;
        }
        let stride = self.stride();
        let nw = self.num_words;
        let gs = self.row_start;
        let ops = prepare_word_ops(gates);

        let process_row = |row: &mut [u64], p: &mut bool| {
            let mut xw = row[word];
            let mut zw = row[nw + word];
            apply_prepared_ops(&mut xw, &mut zw, p, &ops);
            row[word] = xw;
            row[nw + word] = zw;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz[gs * stride..]
                .par_chunks_mut(stride)
                .zip(self.phase[gs..].par_iter_mut())
                .for_each(|(row, p)| process_row(row, p));
            return;
        }

        for (row, p) in self.xz[gs * stride..]
            .chunks_mut(stride)
            .zip(self.phase[gs..].iter_mut())
        {
            process_row(row, p);
        }
    }

    /// Flush all non-empty word groups in a single pass over all rows.
    ///
    /// Fuses K word-group flushes into one row iteration instead of K separate
    /// passes. Reduces memory traffic by ~K× at large qubit counts.
    fn flush_all_word_groups(&mut self, word_groups: &mut [Vec<BatchGate>]) {
        let mut active_count = 0usize;
        let mut single_w = 0usize;
        for (w, group) in word_groups.iter().enumerate() {
            if !group.is_empty() {
                active_count += 1;
                single_w = w;
            }
        }

        if active_count == 0 {
            return;
        }

        if active_count == 1 {
            self.flush_word_group(single_w, &word_groups[single_w]);
            word_groups[single_w].clear();
            return;
        }

        let stride = self.stride();
        let nw = self.num_words;
        let gs = self.row_start;

        let prepared: Vec<(usize, Vec<PrepOp>)> = word_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(w, g)| (w, prepare_word_ops(g)))
            .collect();

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz[gs * stride..]
                .par_chunks_mut(stride)
                .zip(self.phase[gs..].par_iter_mut())
                .for_each(|(row, p)| {
                    for &(w, ref ops) in &prepared {
                        let mut xw = row[w];
                        let mut zw = row[nw + w];
                        apply_prepared_ops(&mut xw, &mut zw, p, ops);
                        row[w] = xw;
                        row[nw + w] = zw;
                    }
                });
            for group in word_groups.iter_mut() {
                group.clear();
            }
            return;
        }

        for (row, p) in self.xz[gs * stride..]
            .chunks_mut(stride)
            .zip(self.phase[gs..].iter_mut())
        {
            for &(w, ref ops) in &prepared {
                let mut xw = row[w];
                let mut zw = row[nw + w];
                apply_prepared_ops(&mut xw, &mut zw, p, ops);
                row[w] = xw;
                row[nw + w] = zw;
            }
        }
        for group in word_groups.iter_mut() {
            group.clear();
        }
    }

    fn pcc_apply_cross_word(&mut self, cross_word: &[CrossWordGate]) {
        let gs = self.row_start;
        let total_rows = 2 * self.n + 1;
        let active_rows = total_rows - gs;
        let col_words = active_rows.div_ceil(64);
        let nw = self.num_words;
        let stride = self.stride();

        let mut qubit_to_idx = vec![u32::MAX; self.n];
        let mut idx_to_qubit: Vec<usize> = Vec::new();
        for g in cross_word {
            let q0 = g.w0 as usize * 64 + g.b0 as usize;
            let q1 = g.w1 as usize * 64 + g.b1 as usize;
            if qubit_to_idx[q0] == u32::MAX {
                qubit_to_idx[q0] = idx_to_qubit.len() as u32;
                idx_to_qubit.push(q0);
            }
            if qubit_to_idx[q1] == u32::MAX {
                qubit_to_idx[q1] = idx_to_qubit.len() as u32;
                idx_to_qubit.push(q1);
            }
        }
        let num_cached = idx_to_qubit.len();
        let mut x_cols = vec![0u64; num_cached * col_words];
        let mut z_cols = vec![0u64; num_cached * col_words];

        let qubit_info: Vec<(usize, u64)> = idx_to_qubit
            .iter()
            .map(|&q| (q / 64, 1u64 << (q % 64)))
            .collect();

        for (ci, &(word, mask)) in qubit_info.iter().enumerate() {
            let x_off = ci * col_words;
            let z_off = ci * col_words;
            for (row_idx, row) in self.xz[gs * stride..].chunks(stride).enumerate() {
                let cw = row_idx / 64;
                let cb = row_idx % 64;
                if row[word] & mask != 0 {
                    x_cols[x_off + cw] |= 1u64 << cb;
                }
                if row[nw + word] & mask != 0 {
                    z_cols[z_off + cw] |= 1u64 << cb;
                }
            }
        }

        let mut phase_col = vec![0u64; col_words];
        for (i, p) in self.phase[gs..].iter().enumerate() {
            if *p {
                phase_col[i / 64] |= 1u64 << (i % 64);
            }
        }

        for g in cross_word {
            let q0 = g.w0 as usize * 64 + g.b0 as usize;
            let q1 = g.w1 as usize * 64 + g.b1 as usize;
            let i0 = qubit_to_idx[q0] as usize;
            let i1 = qubit_to_idx[q1] as usize;
            let off0 = i0 * col_words;
            let off1 = i1 * col_words;

            match g.kind {
                BatchGate::CX => {
                    for w in 0..col_words {
                        let xa = x_cols[off0 + w];
                        let za = z_cols[off0 + w];
                        let xb = x_cols[off1 + w];
                        let zb = z_cols[off1 + w];
                        phase_col[w] ^= xa & zb & !(xb ^ za);
                        x_cols[off1 + w] = xb ^ xa;
                        z_cols[off0 + w] = za ^ zb;
                    }
                }
                BatchGate::CZ => {
                    for w in 0..col_words {
                        let xa = x_cols[off0 + w];
                        let xb = x_cols[off1 + w];
                        let za = z_cols[off0 + w];
                        let zb = z_cols[off1 + w];
                        phase_col[w] ^= xa & xb & (za ^ zb);
                        z_cols[off0 + w] = za ^ xb;
                        z_cols[off1 + w] = zb ^ xa;
                    }
                }
                _ => {
                    for w in 0..col_words {
                        let xa = x_cols[off0 + w];
                        let xb = x_cols[off1 + w];
                        x_cols[off0 + w] = xb;
                        x_cols[off1 + w] = xa;
                        let za = z_cols[off0 + w];
                        let zb = z_cols[off1 + w];
                        z_cols[off0 + w] = zb;
                        z_cols[off1 + w] = za;
                    }
                }
            }
        }

        for (ci, &(word, mask)) in qubit_info.iter().enumerate() {
            let bit = mask.trailing_zeros() as usize;
            let x_off = ci * col_words;
            let z_off = ci * col_words;
            for (row_idx, row) in self.xz[gs * stride..].chunks_mut(stride).enumerate() {
                let cw = row_idx / 64;
                let cb = row_idx % 64;
                let xbit = (x_cols[x_off + cw] >> cb) & 1;
                row[word] = (row[word] & !mask) | (xbit << bit);
                let zbit = (z_cols[z_off + cw] >> cb) & 1;
                row[nw + word] = (row[nw + word] & !mask) | (zbit << bit);
            }
        }

        for (i, p) in self.phase[gs..].iter_mut().enumerate() {
            *p = (phase_col[i / 64] >> (i % 64)) & 1 == 1;
        }
    }

    /// Flush all word groups and apply all buffered cross-word 2q gates in a
    /// single row iteration.
    ///
    /// This eliminates the cascading flush pattern where each cross-word CX
    /// forces an immediate flush of its two word groups. Instead, all word-group
    /// ops are applied first, then all cross-word gates in one pass over rows.
    fn flush_all_with_cross_word(
        &mut self,
        word_groups: &mut [Vec<BatchGate>],
        cross_word: &mut Vec<CrossWordGate>,
    ) {
        let has_wg = word_groups.iter().any(|g| !g.is_empty());
        let has_cw = !cross_word.is_empty();

        if !has_wg && !has_cw {
            return;
        }

        if !has_cw {
            self.flush_all_word_groups(word_groups);
            return;
        }

        if self.n >= 256 && cross_word.len() >= 4 {
            self.flush_all_word_groups(word_groups);
            self.pcc_apply_cross_word(cross_word);
            cross_word.clear();
            return;
        }

        let stride = self.stride();
        let nw = self.num_words;
        let gs = self.row_start;

        let prepared: Vec<(usize, Vec<PrepOp>)> = word_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(w, g)| (w, prepare_word_ops(g)))
            .collect();

        let cw_ref: &[CrossWordGate] = &*cross_word;

        let row_op = |row: &mut [u64], p: &mut bool| {
            for &(w, ref ops) in &prepared {
                let mut xw = row[w];
                let mut zw = row[nw + w];
                apply_prepared_ops(&mut xw, &mut zw, p, ops);
                row[w] = xw;
                row[nw + w] = zw;
            }
            for cg in cw_ref {
                let w0 = cg.w0 as usize;
                let w1 = cg.w1 as usize;
                let b0 = cg.b0 as usize;
                let b1 = cg.b1 as usize;
                let m0 = 1u64 << b0;
                let m1 = 1u64 << b1;
                if cg.kind == BatchGate::CX {
                    let xa = (row[w0] >> b0) & 1;
                    let za = (row[nw + w0] >> b0) & 1;
                    let xb = (row[w1] >> b1) & 1;
                    let zb = (row[nw + w1] >> b1) & 1;
                    *p ^= (xa & zb & (xb ^ za ^ 1)) == 1;
                    if xa == 1 {
                        row[w1] ^= m1;
                    }
                    if zb == 1 {
                        row[nw + w0] ^= m0;
                    }
                } else if cg.kind == BatchGate::CZ {
                    let xa = (row[w0] >> b0) & 1;
                    let xb = (row[w1] >> b1) & 1;
                    let za = (row[nw + w0] >> b0) & 1;
                    let zb = (row[nw + w1] >> b1) & 1;
                    *p ^= (xa & xb & (za ^ zb)) == 1;
                    if xb == 1 {
                        row[nw + w0] ^= m0;
                    }
                    if xa == 1 {
                        row[nw + w1] ^= m1;
                    }
                } else {
                    let xa = (row[w0] >> b0) & 1;
                    let xb = (row[w1] >> b1) & 1;
                    if xa != xb {
                        row[w0] ^= m0;
                        row[w1] ^= m1;
                    }
                    let za = (row[nw + w0] >> b0) & 1;
                    let zb = (row[nw + w1] >> b1) & 1;
                    if za != zb {
                        row[nw + w0] ^= m0;
                        row[nw + w1] ^= m1;
                    }
                }
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz[gs * stride..]
                .par_chunks_mut(stride)
                .zip(self.phase[gs..].par_iter_mut())
                .for_each(|(row, p)| row_op(row, p));
            for group in word_groups.iter_mut() {
                group.clear();
            }
            cross_word.clear();
            return;
        }

        for (row, p) in self.xz[gs * stride..]
            .chunks_mut(stride)
            .zip(self.phase[gs..].iter_mut())
        {
            row_op(row, p);
        }
        for group in word_groups.iter_mut() {
            group.clear();
        }
        cross_word.clear();
    }
}

impl StabilizerBackend {
    fn gate_rows_view(&mut self) -> GateRows<'_> {
        GateRows {
            n: self.n,
            num_words: self.num_words,
            row_start: self.gate_row_start,
            xz: &mut self.xz,
            phase: &mut self.phase,
        }
    }

    /// Classify a gate into a BatchGate for word-group batching.
    ///
    /// Returns `Some((word, BatchGate))` for batchable gates (1q or same-word 2q).
    /// Returns `None` for cross-word 2q gates or non-Clifford gates.
    pub(super) fn classify_gate(gate: &Gate, targets: &[usize]) -> Option<(usize, BatchGate)> {
        match gate {
            Gate::Id => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::ID,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::H => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::H,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::S => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::S,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::Sdg => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::SDG,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::X => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::X,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::Y => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::Y,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::Z => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::Z,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::SX => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::SX,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::SXdg => Some((
                targets[0] / 64,
                BatchGate {
                    kind: BatchGate::SXDG,
                    a_bit: (targets[0] % 64) as u8,
                    b_bit: 0,
                },
            )),
            Gate::Cx | Gate::Cz | Gate::Swap => {
                let w0 = targets[0] / 64;
                let w1 = targets[1] / 64;
                if w0 != w1 {
                    return None;
                }
                let kind = match gate {
                    Gate::Cx => BatchGate::CX,
                    Gate::Cz => BatchGate::CZ,
                    _ => BatchGate::SWAP,
                };
                Some((
                    w0,
                    BatchGate {
                        kind,
                        a_bit: (targets[0] % 64) as u8,
                        b_bit: (targets[1] % 64) as u8,
                    },
                ))
            }
            _ => None,
        }
    }

    /// Length of the leading run of measurements safe to batch as one unit.
    /// Batching matches sequential semantics only while every qubit and
    /// classical bit in the run is distinct, so the run stops at the first
    /// repeat or non-measure instruction.
    fn measure_run_len(&self, instructions: &[Instruction]) -> usize {
        let qubit_words = self.num_words;
        let cbit_words = self.classical_bits.len().div_ceil(64).max(1);
        let mut seen_qubits = vec![0u64; qubit_words];
        let mut seen_cbits = vec![0u64; cbit_words];
        let mut len = 0;

        for instruction in instructions {
            let Instruction::Measure {
                qubit,
                classical_bit,
            } = instruction
            else {
                break;
            };
            let q_mask = 1u64 << (qubit % 64);
            let c_mask = 1u64 << (classical_bit % 64);
            if seen_qubits[qubit / 64] & q_mask != 0 || seen_cbits[classical_bit / 64] & c_mask != 0
            {
                break;
            }
            seen_qubits[qubit / 64] |= q_mask;
            seen_cbits[classical_bit / 64] |= c_mask;
            len += 1;
        }
        len
    }

    pub(in crate::backend::stabilizer) fn apply_instructions_word_batch(
        &mut self,
        instructions: &[Instruction],
    ) -> Result<()> {
        self.sgi_stale = true;
        let nw = self.num_words;
        let mut word_groups: Vec<Vec<BatchGate>> = vec![Vec::new(); nw];
        let mut cross_word: Vec<CrossWordGate> = Vec::new();
        let mut cross_word_qubits: Vec<u64> = vec![0u64; nw];

        let mut idx = 0;
        while idx < instructions.len() {
            let instruction = &instructions[idx];
            match instruction {
                Instruction::Gate { gate, targets } => {
                    let chain = cx_chain_len(&instructions[idx..]);
                    if chain >= MIN_CX_CHAIN {
                        self.gate_rows_view()
                            .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                        let start = targets[0];
                        self.gate_rows_view()
                            .apply_cx_chain_run(start, start + chain);
                        idx += chain;
                        continue;
                    }
                    if let Some((w, bg)) = Self::classify_gate(gate, targets) {
                        let mut bits = 1u64 << bg.a_bit;
                        if bg.kind >= BatchGate::CX {
                            bits |= 1u64 << bg.b_bit;
                        }
                        if cross_word_qubits[w] & bits != 0 {
                            self.gate_rows_view()
                                .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                            cross_word_qubits.fill(0);
                        }
                        word_groups[w].push(bg);
                    } else if let (Gate::Cx | Gate::Cz | Gate::Swap, &[t0, t1]) =
                        (gate, targets.as_slice())
                    {
                        let w0 = t0 / 64;
                        let w1 = t1 / 64;
                        let b0 = (t0 % 64) as u8;
                        let b1 = (t1 % 64) as u8;
                        let m0 = 1u64 << b0;
                        let m1 = 1u64 << b1;
                        if cross_word_qubits[w0] & m0 != 0 || cross_word_qubits[w1] & m1 != 0 {
                            self.gate_rows_view()
                                .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                            cross_word_qubits.fill(0);
                        }
                        let kind = match gate {
                            Gate::Cx => BatchGate::CX,
                            Gate::Cz => BatchGate::CZ,
                            _ => BatchGate::SWAP,
                        };
                        cross_word.push(CrossWordGate {
                            kind,
                            w0: w0 as u16,
                            w1: w1 as u16,
                            b0,
                            b1,
                        });
                        cross_word_qubits[w0] |= m0;
                        cross_word_qubits[w1] |= m1;
                    } else {
                        self.gate_rows_view()
                            .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                        self.dispatch_gate(gate, targets)?;
                    }
                }
                Instruction::Measure { .. } => {
                    self.gate_rows_view()
                        .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                    let run_possible = matches!(
                        instructions.get(idx + MIN_MEASURES_FOR_BATCH - 1),
                        Some(Instruction::Measure { .. })
                    );
                    let run = if run_possible {
                        self.measure_run_len(&instructions[idx..])
                    } else {
                        1
                    };
                    if run >= MIN_MEASURES_FOR_BATCH {
                        let measurements: Vec<(usize, usize)> = instructions[idx..idx + run]
                            .iter()
                            .map(|inst| match inst {
                                Instruction::Measure {
                                    qubit,
                                    classical_bit,
                                } => (*qubit, *classical_bit),
                                _ => unreachable!("measure_run_len returns only measure runs"),
                            })
                            .collect();
                        self.batch_measure_ref_info(&measurements);
                        idx += run;
                        continue;
                    }
                    self.apply(instruction)?;
                }
                _ => {
                    self.gate_rows_view()
                        .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                    self.apply(instruction)?;
                }
            }
            idx += 1;
        }

        self.gate_rows_view()
            .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
        Ok(())
    }

    pub(in crate::backend::stabilizer) fn apply_gates_only_word_batch(
        &mut self,
        instructions: &[Instruction],
    ) -> Result<()> {
        self.sgi_stale = true;
        apply_gates_word_batch(self, instructions)
    }
}

/// A tableau a batched gate run can drive.
pub(crate) trait BatchTarget {
    fn num_words(&self) -> usize;
    fn batch_rows(&mut self) -> GateRows<'_>;
    /// Apply one gate the word classifier declined, after a flush.
    fn apply_one(&mut self, gate: &Gate, targets: &[usize]) -> Result<()>;
}

impl BatchTarget for StabilizerBackend {
    fn num_words(&self) -> usize {
        self.num_words
    }

    fn batch_rows(&mut self) -> GateRows<'_> {
        self.gate_rows_view()
    }

    fn apply_one(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        self.dispatch_gate(gate, targets)
    }
}

/// Word-group batching over a run of gates, for any tableau that can hand over
/// its gate rows. One pass over the rows covers every gate buffered since the
/// last flush, where the per-instruction path costs one pass per gate.
///
/// A gate the word classifier declines goes to [`BatchTarget::apply_one`] after
/// a flush, so ordering holds whatever the target does with it.
pub(crate) fn apply_gates_word_batch<T: BatchTarget>(
    target: &mut T,
    instructions: &[Instruction],
) -> Result<()> {
    let nw = target.num_words();
    let mut word_groups: Vec<Vec<BatchGate>> = vec![Vec::new(); nw];
    let mut cross_word: Vec<CrossWordGate> = Vec::new();
    let mut cross_word_qubits: Vec<u64> = vec![0u64; nw];

    let mut idx = 0;
    while idx < instructions.len() {
        let instruction = &instructions[idx];
        idx += 1;
        match instruction {
            Instruction::Gate { gate, targets } => {
                let chain = cx_chain_len(&instructions[idx - 1..]);
                if chain >= MIN_CX_CHAIN {
                    target
                        .batch_rows()
                        .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                    let start = targets[0];
                    target.batch_rows().apply_cx_chain_run(start, start + chain);
                    idx += chain - 1;
                    continue;
                }
                if let Some((w, bg)) = StabilizerBackend::classify_gate(gate, targets) {
                    let mut bits = 1u64 << bg.a_bit;
                    if bg.kind >= BatchGate::CX {
                        bits |= 1u64 << bg.b_bit;
                    }
                    if cross_word_qubits[w] & bits != 0 {
                        target
                            .batch_rows()
                            .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                    }
                    word_groups[w].push(bg);
                } else if let (Gate::Cx | Gate::Cz | Gate::Swap, &[t0, t1]) =
                    (gate, targets.as_slice())
                {
                    let w0 = t0 / 64;
                    let w1 = t1 / 64;
                    let b0 = (t0 % 64) as u8;
                    let b1 = (t1 % 64) as u8;
                    let m0 = 1u64 << b0;
                    let m1 = 1u64 << b1;
                    if cross_word_qubits[w0] & m0 != 0 || cross_word_qubits[w1] & m1 != 0 {
                        target
                            .batch_rows()
                            .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                    }
                    let kind = match gate {
                        Gate::Cx => BatchGate::CX,
                        Gate::Cz => BatchGate::CZ,
                        _ => BatchGate::SWAP,
                    };
                    cross_word.push(CrossWordGate {
                        kind,
                        w0: w0 as u16,
                        w1: w1 as u16,
                        b0,
                        b1,
                    });
                    cross_word_qubits[w0] |= m0;
                    cross_word_qubits[w1] |= m1;
                } else {
                    target
                        .batch_rows()
                        .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                    target.apply_one(gate, targets)?;
                }
            }
            _ => {
                target
                    .batch_rows()
                    .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                cross_word_qubits.fill(0);
            }
        }
    }

    target
        .batch_rows()
        .flush_all_with_cross_word(&mut word_groups, &mut cross_word);
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::backend::Backend;
    use crate::backend::stabilizer::StabilizerBackend;
    use crate::circuit::Circuit;
    use crate::gates::Gate;
    use crate::sim;

    fn run_and_count_zero(circuit: &Circuit) -> usize {
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, circuit).unwrap();
        b.classical_results().iter().filter(|x| !**x).count()
    }

    // The scalar CX row update, one gate at a time, that the word formula
    // has to reproduce bit for bit including the phase.
    fn cx_chain_reference(xw: &mut u64, zw: &mut u64, p: &mut bool, first: u8, last: u8) {
        for k in first..last {
            let (c, t) = (k, k + 1);
            let xa = (*xw >> c) & 1;
            let za = (*zw >> c) & 1;
            let xb = (*xw >> t) & 1;
            let zb = (*zw >> t) & 1;
            *p ^= (xa & zb & (xb ^ za ^ 1)) == 1;
            if xa == 1 {
                *xw ^= 1u64 << t;
            }
            if zb == 1 {
                *zw ^= 1u64 << c;
            }
        }
    }

    #[test]
    fn cx_chain_word_formula_matches_the_scalar_chain() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..4000 {
            let a = (next() % 64) as u8;
            let b = (next() % 64) as u8;
            let (first, last) = if a <= b { (a, b) } else { (b, a) };
            let (x0, z0, p0) = (next(), next(), next() & 1 == 1);
            let (mut xw, mut zw, mut p) = (x0, z0, p0);
            let (mut xr, mut zr, mut pr) = (x0, z0, p0);
            super::apply_cx_chain(&mut xw, &mut zw, &mut p, first, last);
            cx_chain_reference(&mut xr, &mut zr, &mut pr, first, last);
            assert_eq!(
                (xw, zw, p),
                (xr, zr, pr),
                "range {first}..={last} on {x0:#x} {z0:#x}"
            );
        }
    }

    #[test]
    fn cx_chain_run_matches_per_gate_rows() {
        let n = 500;
        let prefix = crate::circuits::clifford_random_pairs(n, 4, 0);
        let mut base = StabilizerBackend::new(0);
        sim::run_on(&mut base, &prefix).unwrap();
        for (start, end) in [
            (3usize, 498usize),
            (0, 64),
            (63, 65),
            (60, 200),
            (128, 192),
            (100, 127),
        ] {
            let mut a = base.clone();
            a.gate_rows_view().apply_cx_chain_run(start, end);
            let mut b = base.clone();
            let mut chain = Circuit::new(n, 0);
            for k in start..end {
                chain.add_gate(Gate::Cx, &[k, k + 1]);
            }
            for instr in &chain.instructions {
                b.apply(instr).unwrap();
            }
            let (xa, pa) = a.raw_tableau();
            let (xb, pb) = b.raw_tableau();
            let stride = 2 * n.div_ceil(64);
            let bad: Vec<usize> = (0..2 * n)
                .filter(|&r| xa[r * stride..(r + 1) * stride] != xb[r * stride..(r + 1) * stride])
                .collect();
            let badp: Vec<usize> = (0..2 * n).filter(|&r| pa[r] != pb[r]).collect();
            assert!(
                bad.is_empty() && badp.is_empty(),
                "chain {start}..{end}: {} rows differ {:?}, {} phases differ {:?}",
                bad.len(),
                &bad[..bad.len().min(8)],
                badp.len(),
                &badp[..badp.len().min(8)]
            );
        }
    }

    #[test]
    fn cx_chain_len_stops_where_the_chain_does() {
        let mut c = Circuit::new(8, 0);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Cx, &[3, 4]);
        c.add_gate(Gate::Cx, &[4, 5]);
        c.add_gate(Gate::Cx, &[6, 7]);
        assert_eq!(super::cx_chain_len(&c.instructions), 3);
        assert_eq!(super::cx_chain_len(&c.instructions[3..]), 1);
        let mut d = Circuit::new(8, 0);
        d.add_gate(Gate::Cx, &[3, 2]);
        d.add_gate(Gate::H, &[0]);
        assert_eq!(super::cx_chain_len(&d.instructions), 0);
        assert_eq!(super::cx_chain_len(&d.instructions[1..]), 0);
    }

    fn assert_runs(circuit: &Circuit) {
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, circuit).unwrap();
    }

    #[test]
    fn batch_path_193q_basic_clifford() {
        let n = 200;
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            c.add_gate(Gate::S, &[q]);
            c.add_gate(Gate::Z, &[q]);
        }
        assert_runs(&c);
    }

    #[test]
    fn batch_path_overlapping_1q_targets_segment_flush() {
        let n = 200;
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::S, &[q]);
            c.add_gate(Gate::Sdg, &[q]);
        }
        assert_runs(&c);
    }

    #[test]
    fn batch_path_cross_word_two_qubit_gates() {
        let n = 200;
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 128]);
        c.add_gate(Gate::Cz, &[5, 192]);
        c.add_gate(Gate::Swap, &[60, 199]);
        assert_runs(&c);
    }

    #[test]
    fn batch_path_all_1q_gate_types() {
        let n = 200;
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::S, &[q]);
            c.add_gate(Gate::Sdg, &[q]);
            c.add_gate(Gate::X, &[q]);
            c.add_gate(Gate::Y, &[q]);
            c.add_gate(Gate::Z, &[q]);
            c.add_gate(Gate::SX, &[q]);
            c.add_gate(Gate::SXdg, &[q]);
            c.add_gate(Gate::Id, &[q]);
        }
        assert_runs(&c);
    }

    #[test]
    fn batch_path_zero_state_unchanged_after_identity_loop() {
        let n = 200;
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::Id, &[q]);
        }
        for q in 0..n {
            c.instructions.push(crate::circuit::Instruction::Measure {
                qubit: q,
                classical_bit: q,
            });
        }
        let zeros = run_and_count_zero(&c);
        assert_eq!(zeros, n);
    }

    fn add_measure_all(c: &mut Circuit) {
        let n = c.num_qubits;
        for q in 0..n {
            c.instructions.push(crate::circuit::Instruction::Measure {
                qubit: q,
                classical_bit: q,
            });
        }
    }

    fn sequential_reference_bits(circuit: &Circuit, seed: u64) -> Vec<bool> {
        let mut b = StabilizerBackend::new(seed);
        b.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for instr in &circuit.instructions {
            b.apply(instr).unwrap();
        }
        b.classical_results().to_vec()
    }

    fn batched_bits(circuit: &Circuit, seed: u64) -> Vec<bool> {
        let mut b = StabilizerBackend::new(seed);
        b.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        b.apply_instructions(&circuit.instructions).unwrap();
        b.classical_results().to_vec()
    }

    #[test]
    fn measure_batch_matches_sequential_ghz_997q() {
        let n = 997;
        let mut c = Circuit::new(n, n);
        c.add_gate(Gate::H, &[0]);
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        add_measure_all(&mut c);
        assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
    }

    #[test]
    fn measure_batch_matches_sequential_random_clifford_300q() {
        let n = 300;
        let mut c = crate::circuits::clifford_heavy_circuit(n, 8, 42);
        c.num_classical_bits = n;
        add_measure_all(&mut c);
        assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
    }

    #[test]
    fn measure_batch_matches_sequential_interleaved_measures() {
        let n = 300;
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for q in 0..64 {
            c.instructions.push(crate::circuit::Instruction::Measure {
                qubit: q,
                classical_bit: q,
            });
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        add_measure_all(&mut c);
        assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
    }

    #[test]
    fn measure_batch_splits_on_repeated_qubit() {
        let n = 300;
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        add_measure_all(&mut c);
        for q in 0..32 {
            c.instructions.push(crate::circuit::Instruction::Measure {
                qubit: q,
                classical_bit: q,
            });
        }
        assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
    }

    #[test]
    fn measure_batch_splits_on_shared_classical_bit() {
        let n = 300;
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
        }
        for q in 0..32 {
            c.instructions.push(crate::circuit::Instruction::Measure {
                qubit: q,
                classical_bit: 0,
            });
        }
        add_measure_all(&mut c);
        assert_eq!(batched_bits(&c, 42), sequential_reference_bits(&c, 42));
    }
}
