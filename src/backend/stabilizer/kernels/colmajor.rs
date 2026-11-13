//! Column-major mode for long Clifford runs on wide tableaus: the gate rows are transposed
//! once into per-qubit columns, each gate in the run updates whole columns, and the rows
//! are written back when the run ends.

use crate::circuit::Instruction;
use crate::gates::Gate;

use super::GateRows;

/// Narrowest register the column mode serves, the width at which the row-major flush
/// starts gathering cross-word gates into columns.
pub(crate) const MIN_QUBITS_FOR_COLS: usize = 256;

#[inline]
pub(crate) fn columns_enabled(n: usize) -> bool {
    n >= MIN_QUBITS_FOR_COLS
}

/// Whether a run with `cross` cross-word two-qubit gates moves to columns. Entry and exit
/// cost about as much as gathering and scattering two columns per tableau word through
/// the row-major cross-word flush, so the run needs about one cross-word gate per word.
#[inline]
pub(crate) fn run_takes_columns(cross: usize, num_words: usize) -> bool {
    cross >= num_words
}

/// Transpose a 64x64 bit block in place: bit `j` of word `i` moves to bit `i` of word `j`.
#[inline(always)]
fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m = 0x0000_0000_FFFF_FFFFu64;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k] ^= t << j;
            a[k + j] ^= t;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}

fn is_col_gate(gate: &Gate) -> bool {
    matches!(
        gate,
        Gate::Id
            | Gate::H
            | Gate::S
            | Gate::Sdg
            | Gate::X
            | Gate::Y
            | Gate::Z
            | Gate::SX
            | Gate::SXdg
            | Gate::Cx
            | Gate::Cz
            | Gate::Swap
    )
}

/// Length of the leading run the columns can carry, and how many of its two-qubit gates
/// cross a word boundary. Barriers stay inside a run; a conditional Clifford gate does only
/// when `conditionals` is set, since the gates-only batch skips conditionals.
pub(crate) fn col_run(instructions: &[Instruction], conditionals: bool) -> (usize, usize) {
    let mut cross = 0;
    for (i, instruction) in instructions.iter().enumerate() {
        let (gate, targets) = match instruction {
            Instruction::Gate { gate, targets } => (gate, targets),
            Instruction::Conditional { gate, targets, .. } if conditionals => (gate, targets),
            Instruction::Barrier { .. } => continue,
            _ => return (i, cross),
        };
        if !is_col_gate(gate) {
            return (i, cross);
        }
        if let [a, b] = targets.as_slice()
            && a / 64 != b / 64
        {
            cross += 1;
        }
    }
    (instructions.len(), cross)
}

/// Column buffers kept across the runs of one batched call. Qubit `q` holds its X column
/// at `2q * cw` and its Z column at `(2q + 1) * cw`, one bit per gate row.
#[derive(Default)]
pub(crate) struct ColumnScratch {
    cols: Vec<u64>,
    phase: Vec<u64>,
}

struct Columns<'a> {
    cols: &'a mut [u64],
    phase: &'a mut [u64],
    cw: usize,
}

/// Apply a run accepted by [`col_run`] to the gate rows through columns. A conditional
/// gate reads `classical_bits`, which no instruction in the run can change.
pub(crate) fn apply_run(
    rows: GateRows<'_>,
    scratch: &mut ColumnScratch,
    run: &[Instruction],
    classical_bits: &[bool],
) {
    let n = rows.n;
    let gate_rows = rows.phase.len() - rows.row_start;
    let cw = gate_rows.div_ceil(64);
    scratch.cols.resize(2 * n * cw, 0);
    scratch.phase.clear();
    scratch.phase.resize(cw, 0);
    let mut cols = Columns {
        cols: &mut scratch.cols,
        phase: &mut scratch.phase,
        cw,
    };
    cols.gather(&rows, gate_rows);
    for instruction in run {
        match instruction {
            Instruction::Gate { gate, targets } => cols.apply(gate, targets),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } if condition.evaluate(classical_bits) => cols.apply(gate, targets),
            _ => {}
        }
    }
    cols.scatter(rows, gate_rows);
}

impl Columns<'_> {
    fn gather(&mut self, rows: &GateRows<'_>, gate_rows: usize) {
        let n = rows.n;
        let nw = rows.num_words;
        let stride = 2 * nw;
        let cw = self.cw;
        let src = &rows.xz[rows.row_start * stride..];
        let mut block = [0u64; 64];
        for rb in 0..cw {
            let r0 = rb * 64;
            let live = (gate_rows - r0).min(64);
            for half in 0..2 {
                for w in 0..nw {
                    let off = half * nw + w;
                    for (i, slot) in block.iter_mut().enumerate() {
                        *slot = if i < live {
                            src[(r0 + i) * stride + off]
                        } else {
                            0
                        };
                    }
                    transpose64(&mut block);
                    let q_end = (n - w * 64).min(64);
                    for (j, &word) in block[..q_end].iter().enumerate() {
                        self.cols[(2 * (w * 64 + j) + half) * cw + rb] = word;
                    }
                }
            }
        }
        for (i, &p) in rows.phase[rows.row_start..].iter().enumerate() {
            self.phase[i / 64] |= (p as u64) << (i % 64);
        }
    }

    fn scatter(&self, rows: GateRows<'_>, gate_rows: usize) {
        let n = rows.n;
        let nw = rows.num_words;
        let stride = 2 * nw;
        let cw = self.cw;
        let dst = &mut rows.xz[rows.row_start * stride..];
        let mut block = [0u64; 64];
        for rb in 0..cw {
            let r0 = rb * 64;
            let live = (gate_rows - r0).min(64);
            for half in 0..2 {
                for w in 0..nw {
                    let off = half * nw + w;
                    let q_end = (n - w * 64).min(64);
                    for (j, slot) in block.iter_mut().enumerate() {
                        *slot = if j < q_end {
                            self.cols[(2 * (w * 64 + j) + half) * cw + rb]
                        } else {
                            0
                        };
                    }
                    transpose64(&mut block);
                    for (i, &word) in block[..live].iter().enumerate() {
                        dst[(r0 + i) * stride + off] = word;
                    }
                }
            }
        }
        for (i, p) in rows.phase[rows.row_start..].iter_mut().enumerate() {
            *p = (self.phase[i / 64] >> (i % 64)) & 1 == 1;
        }
    }

    fn qubit(&mut self, q: usize) -> (&mut [u64], &mut [u64], &mut [u64]) {
        let cw = self.cw;
        let (x, z) = self.cols[2 * q * cw..(2 * q + 2) * cw].split_at_mut(cw);
        (x, z, &mut *self.phase)
    }

    fn pair(&mut self, a: usize, b: usize) -> [&mut [u64]; 5] {
        let cw = self.cw;
        let block = 2 * cw;
        let (lo, hi, swapped) = if a < b { (a, b, false) } else { (b, a, true) };
        let (head, tail) = self.cols.split_at_mut(hi * block);
        let (xl, zl) = head[lo * block..(lo + 1) * block].split_at_mut(cw);
        let (xh, zh) = tail[..block].split_at_mut(cw);
        if swapped {
            [xh, zh, xl, zl, &mut *self.phase]
        } else {
            [xl, zl, xh, zh, &mut *self.phase]
        }
    }

    #[inline(always)]
    fn apply(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Cx | Gate::Cz | Gate::Swap => {
                let [xa, za, xb, zb, ph] = self.pair(targets[0], targets[1]);
                let words = xa.iter_mut().zip(za.iter_mut()).zip(xb.iter_mut());
                let words = words.zip(zb.iter_mut()).zip(ph.iter_mut());
                match gate {
                    Gate::Cx => {
                        for ((((xa, za), xb), zb), p) in words {
                            *p ^= *xa & *zb & !(*xb ^ *za);
                            *xb ^= *xa;
                            *za ^= *zb;
                        }
                    }
                    Gate::Cz => {
                        for ((((xa, za), xb), zb), p) in words {
                            *p ^= *xa & *xb & (*za ^ *zb);
                            *za ^= *xb;
                            *zb ^= *xa;
                        }
                    }
                    _ => {
                        for ((((xa, za), xb), zb), _) in words {
                            std::mem::swap(xa, xb);
                            std::mem::swap(za, zb);
                        }
                    }
                }
            }
            Gate::Id => {}
            _ => {
                let (xs, zs, ph) = self.qubit(targets[0]);
                let words = xs.iter_mut().zip(zs.iter_mut()).zip(ph.iter_mut());
                match gate {
                    Gate::H => {
                        for ((x, z), p) in words {
                            *p ^= *x & *z;
                            std::mem::swap(x, z);
                        }
                    }
                    Gate::S => {
                        for ((x, z), p) in words {
                            *p ^= *x & *z;
                            *z ^= *x;
                        }
                    }
                    Gate::Sdg => {
                        for ((x, z), p) in words {
                            *z ^= *x;
                            *p ^= *x & *z;
                        }
                    }
                    Gate::X => {
                        for ((_, z), p) in words {
                            *p ^= *z;
                        }
                    }
                    Gate::Y => {
                        for ((x, z), p) in words {
                            *p ^= *x ^ *z;
                        }
                    }
                    Gate::Z => {
                        for ((x, _), p) in words {
                            *p ^= *x;
                        }
                    }
                    Gate::SX => {
                        for ((x, z), p) in words {
                            *p ^= *z & !*x;
                            *x ^= *z;
                        }
                    }
                    Gate::SXdg => {
                        for ((x, z), p) in words {
                            *p ^= *x & *z;
                            *x ^= *z;
                        }
                    }
                    _ => unreachable!("col_run admits Clifford gates only"),
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "colmajor_tests.rs"]
mod tests;
