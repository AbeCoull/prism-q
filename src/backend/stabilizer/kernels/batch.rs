use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::Gate;

use super::StabilizerBackend;
#[cfg(feature = "parallel")]
use super::MIN_QUBITS_FOR_PAR_GATES;

/// Minimum number of u64 words for word-group gate batching to be profitable.
///
/// Below this threshold, the tableau fits in L1/L2 cache and the per-gate
/// match overhead in the batched inner loop exceeds the cache-amortization
/// benefit. At nw=4 (n=193+), each row is 64 bytes (one cache line) and
/// word-group batching avoids repeated full-row iteration per gate.
pub(crate) const MIN_WORDS_FOR_BATCH: usize = 4;

/// Compact gate representation for batched word-group execution.
///
/// All gates in a word group target the same u64 word. `a_bit` and `b_bit`
/// are bit positions (0–63) within that word.
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

/// Pre-processed operation: either a batch of 1q masks or a single 2q gate.
#[derive(Clone, Copy)]
enum PrepOp {
    Masks(OneMasks),
    Gate2q(BatchGate),
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

/// Apply a pre-computed operation sequence to a single (xw, zw, phase) tuple.
#[inline(always)]
fn apply_prepared_ops(xw: &mut u64, zw: &mut u64, p: &mut bool, ops: &[PrepOp]) {
    for op in ops {
        match op {
            PrepOp::Masks(m) => apply_1q_masks(xw, zw, p, m),
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

impl StabilizerBackend {
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
        let gs = self.gate_row_start;
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
        let gs = self.gate_row_start;

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
        let gs = self.gate_row_start;
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
    /// ops are applied first, then all cross-word gates — in one pass over rows.
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
        let gs = self.gate_row_start;

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

    pub(in crate::backend::stabilizer) fn apply_instructions_word_batch(
        &mut self,
        instructions: &[Instruction],
    ) -> Result<()> {
        let nw = self.num_words;
        let mut word_groups: Vec<Vec<BatchGate>> = vec![Vec::new(); nw];
        let mut cross_word: Vec<CrossWordGate> = Vec::new();
        let mut cross_word_qubits: Vec<u64> = vec![0u64; nw];

        for instruction in instructions {
            match instruction {
                Instruction::Gate { gate, targets } => {
                    if let Some((w, bg)) = Self::classify_gate(gate, targets) {
                        let mut bits = 1u64 << bg.a_bit;
                        if bg.kind >= BatchGate::CX {
                            bits |= 1u64 << bg.b_bit;
                        }
                        if cross_word_qubits[w] & bits != 0 {
                            self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
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
                            self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
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
                        self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                        self.dispatch_gate(gate, targets)?;
                    }
                }
                _ => {
                    self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                    self.apply(instruction)?;
                }
            }
        }

        self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
        Ok(())
    }

    pub(in crate::backend::stabilizer) fn apply_gates_only_word_batch(
        &mut self,
        instructions: &[Instruction],
    ) -> Result<()> {
        let nw = self.num_words;
        let mut word_groups: Vec<Vec<BatchGate>> = vec![Vec::new(); nw];
        let mut cross_word: Vec<CrossWordGate> = Vec::new();
        let mut cross_word_qubits: Vec<u64> = vec![0u64; nw];

        for instruction in instructions {
            match instruction {
                Instruction::Gate { gate, targets } => {
                    if let Some((w, bg)) = Self::classify_gate(gate, targets) {
                        let mut bits = 1u64 << bg.a_bit;
                        if bg.kind >= BatchGate::CX {
                            bits |= 1u64 << bg.b_bit;
                        }
                        if cross_word_qubits[w] & bits != 0 {
                            self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
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
                            self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
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
                        self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                        cross_word_qubits.fill(0);
                        self.dispatch_gate(gate, targets)?;
                    }
                }
                _ => {
                    self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
                    cross_word_qubits.fill(0);
                }
            }
        }

        self.flush_all_with_cross_word(&mut word_groups, &mut cross_word);
        Ok(())
    }
}
