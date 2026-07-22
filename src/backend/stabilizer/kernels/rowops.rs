//! Row-level Clifford tableau operations shared by the dense stabilizer
//! backend, its SGI path, and the factored sub-tableaux.
//!
//! Rows live in a flat `xz` buffer of `2 * nw` words per row (X words then
//! Z words) with a parallel `phase` bit per row. Callers resolve a qubit to
//! a `QubitBit` once per gate; the per-row functions take the precomputed
//! coordinates so no derivation sits inside the row loop.

use super::simd::rowmul_words;

#[derive(Clone, Copy)]
pub(crate) struct QubitBit {
    pub word: usize,
    pub bit: usize,
    pub mask: u64,
}

impl QubitBit {
    #[inline(always)]
    pub(crate) fn of(q: usize) -> Self {
        QubitBit {
            word: q / 64,
            bit: q % 64,
            mask: 1u64 << (q % 64),
        }
    }
}

#[inline(always)]
pub(crate) fn h_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let xw = row[a.word];
    let zw = row[nw + a.word];
    let x = xw & a.mask;
    let z = zw & a.mask;
    *phase ^= (x != 0) && (z != 0);
    row[a.word] = (xw & !a.mask) | z;
    row[nw + a.word] = (zw & !a.mask) | x;
}

#[inline(always)]
pub(crate) fn s_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    let z = row[nw + a.word] & a.mask;
    *phase ^= (x != 0) && (z != 0);
    row[nw + a.word] ^= x;
}

#[inline(always)]
pub(crate) fn sdg_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    row[nw + a.word] ^= x;
    let z_new = row[nw + a.word] & a.mask;
    *phase ^= (x != 0) && (z_new != 0);
}

#[inline(always)]
pub(crate) fn x_row(row: &[u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let z = row[nw + a.word] & a.mask;
    *phase ^= z != 0;
}

#[inline(always)]
pub(crate) fn y_row(row: &[u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    let z = row[nw + a.word] & a.mask;
    *phase ^= (x ^ z) != 0;
}

#[inline(always)]
pub(crate) fn z_row(row: &[u64], phase: &mut bool, _nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    *phase ^= x != 0;
}

#[inline(always)]
pub(crate) fn sx_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    let z = row[nw + a.word] & a.mask;
    *phase ^= (z != 0) && (x == 0);
    row[a.word] ^= z;
}

#[inline(always)]
pub(crate) fn sxdg_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit) {
    let x = row[a.word] & a.mask;
    let z = row[nw + a.word] & a.mask;
    *phase ^= (x != 0) && (z != 0);
    row[a.word] ^= z;
}

#[inline(always)]
pub(crate) fn cx_row(row: &mut [u64], phase: &mut bool, nw: usize, c: QubitBit, t: QubitBit) {
    let xa = (row[c.word] >> c.bit) & 1;
    let za = (row[nw + c.word] >> c.bit) & 1;
    let xb = (row[t.word] >> t.bit) & 1;
    let zb = (row[nw + t.word] >> t.bit) & 1;

    *phase ^= (xa & zb & (xb ^ za ^ 1)) == 1;

    if xa == 1 {
        row[t.word] ^= t.mask;
    }
    if zb == 1 {
        row[nw + c.word] ^= c.mask;
    }
}

#[inline(always)]
pub(crate) fn cz_row(row: &mut [u64], phase: &mut bool, nw: usize, a: QubitBit, b: QubitBit) {
    let xa = (row[a.word] >> a.bit) & 1;
    let xb = (row[b.word] >> b.bit) & 1;
    let za = (row[nw + a.word] >> a.bit) & 1;
    let zb = (row[nw + b.word] >> b.bit) & 1;

    *phase ^= (xa & xb & (za ^ zb)) == 1;

    if xb == 1 {
        row[nw + a.word] ^= a.mask;
    }
    if xa == 1 {
        row[nw + b.word] ^= b.mask;
    }
}

#[inline(always)]
pub(crate) fn swap_row(row: &mut [u64], nw: usize, a: QubitBit, b: QubitBit) {
    let xa = (row[a.word] >> a.bit) & 1;
    let xb = (row[b.word] >> b.bit) & 1;
    if xa != xb {
        row[a.word] ^= a.mask;
        row[b.word] ^= b.mask;
    }

    let za = (row[nw + a.word] >> a.bit) & 1;
    let zb = (row[nw + b.word] >> b.bit) & 1;
    if za != zb {
        row[nw + a.word] ^= a.mask;
        row[nw + b.word] ^= b.mask;
    }
}

#[inline(always)]
pub(crate) fn sweep(
    xz: &mut [u64],
    phase: &mut [bool],
    nw: usize,
    par: bool,
    row_op: impl Fn(&mut [u64], &mut bool) + Send + Sync,
) {
    let stride = 2 * nw;
    #[cfg(feature = "parallel")]
    if par {
        use rayon::prelude::*;
        xz.par_chunks_mut(stride)
            .zip(phase.par_iter_mut())
            .for_each(|(row, phase)| row_op(row, phase));
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = par;
    for (row, phase) in xz.chunks_mut(stride).zip(phase.iter_mut()) {
        row_op(row, phase);
    }
}

#[inline(always)]
pub(crate) fn h_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| h_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn s_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| s_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn sdg_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| sdg_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn x_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| x_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn y_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| y_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn z_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| z_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn sx_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| sx_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn sxdg_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize) {
    let a = QubitBit::of(a);
    sweep(xz, phase, nw, par, |row, phase| sxdg_row(row, phase, nw, a));
}

#[inline(always)]
pub(crate) fn cx_all(
    xz: &mut [u64],
    phase: &mut [bool],
    nw: usize,
    par: bool,
    control: usize,
    target: usize,
) {
    let c = QubitBit::of(control);
    let t = QubitBit::of(target);
    sweep(xz, phase, nw, par, |row, phase| {
        cx_row(row, phase, nw, c, t)
    });
}

#[inline(always)]
pub(crate) fn cz_all(xz: &mut [u64], phase: &mut [bool], nw: usize, par: bool, a: usize, b: usize) {
    let a = QubitBit::of(a);
    let b = QubitBit::of(b);
    sweep(xz, phase, nw, par, |row, phase| {
        cz_row(row, phase, nw, a, b)
    });
}

#[inline(always)]
pub(crate) fn swap_all(
    xz: &mut [u64],
    phase: &mut [bool],
    nw: usize,
    par: bool,
    a: usize,
    b: usize,
) {
    let a = QubitBit::of(a);
    let b = QubitBit::of(b);
    sweep(xz, phase, nw, par, |row, _phase| swap_row(row, nw, a, b));
}

/// Multiply row `h` by row `i` (replace `h` with the Pauli product).
///
/// Fused phase+XOR: AG g-function with wordwise popcount, row XOR in the
/// same loop to avoid a separate memory pass.
pub(crate) fn rowmul(xz: &mut [u64], phase: &mut [bool], nw: usize, h: usize, i: usize) {
    debug_assert_ne!(h, i);
    let stride = 2 * nw;
    let base_h = h * stride;
    let base_i = i * stride;

    let initial_sum = if phase[i] { 2u64 } else { 0 } + if phase[h] { 2u64 } else { 0 };

    // SAFETY: h != i, so row regions [base_h..base_h+stride] and
    // [base_i..base_i+stride] are non-overlapping.
    let (dst_x, dst_z, src_x, src_z) = unsafe {
        let ptr = xz.as_mut_ptr();
        (
            std::slice::from_raw_parts_mut(ptr.add(base_h), nw),
            std::slice::from_raw_parts_mut(ptr.add(base_h + nw), nw),
            std::slice::from_raw_parts(ptr.add(base_i) as *const u64, nw),
            std::slice::from_raw_parts(ptr.add(base_i + nw) as *const u64, nw),
        )
    };

    let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
    phase[h] = (sum & 3) >= 2;
}

pub(crate) fn copy_row(xz: &mut [u64], phase: &mut [bool], nw: usize, dst: usize, src: usize) {
    let stride = 2 * nw;
    let src_start = src * stride;
    let dst_start = dst * stride;
    xz.copy_within(src_start..src_start + stride, dst_start);
    phase[dst] = phase[src];
}

pub(crate) fn zero_row(xz: &mut [u64], phase: &mut [bool], nw: usize, r: usize) {
    let stride = 2 * nw;
    let start = r * stride;
    xz[start..start + stride].fill(0);
    phase[r] = false;
}
