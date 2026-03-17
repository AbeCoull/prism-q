use smallvec::SmallVec;

use crate::backend::Backend;
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use rand::Rng;

use super::StabilizerBackend;

#[inline(always)]
pub(super) unsafe fn xor_words(dst: *mut u64, src: *const u64, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        unsafe { xor_words_avx2(dst, src, len) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { xor_words_neon(dst, src, len) };
        return;
    }
    #[allow(unreachable_code)]
    for i in 0..len {
        unsafe { *dst.add(i) ^= *src.add(i) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: *mut u64, src: *const u64, len: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        let d = _mm256_loadu_si256(dst.add(off) as *const __m256i);
        let s = _mm256_loadu_si256(src.add(off) as *const __m256i);
        _mm256_storeu_si256(dst.add(off) as *mut __m256i, _mm256_xor_si256(d, s));
    }
    let tail = chunks * 4;
    for i in tail..len {
        *dst.add(i) ^= *src.add(i);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_words_neon(dst: *mut u64, src: *const u64, len: usize) {
    use std::arch::aarch64::*;

    let chunks = len / 2;
    for i in 0..chunks {
        let off = i * 2;
        let d = vld1q_u64(dst.add(off));
        let s = vld1q_u64(src.add(off));
        vst1q_u64(dst.add(off), veorq_u64(d, s));
    }
    if len & 1 != 0 {
        *dst.add(len - 1) ^= *src.add(len - 1);
    }
}

/// Rowmul word loop: XOR x/z words from src into dst, returning the
/// accumulated phase sum (caller applies `sum & 3 >= 2`).
#[inline(always)]
pub(super) fn rowmul_words(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
    debug_assert_eq!(dst_x.len(), dst_z.len());
    debug_assert_eq!(dst_x.len(), src_x.len());
    debug_assert_eq!(dst_x.len(), src_z.len());
    let nw = dst_x.len();
    let mut sum = initial_sum;

    for w in 0..nw {
        let x1 = src_x[w];
        let z1 = src_z[w];
        let x2 = dst_x[w];
        let z2 = dst_z[w];

        let new_x = x1 ^ x2;
        let new_z = z1 ^ z2;
        dst_x[w] = new_x;
        dst_z[w] = new_z;

        if (x1 | z1 | x2 | z2) == 0 {
            continue;
        }

        let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
        let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
        sum = sum.wrapping_add(2 * pos.count_ones() as u64);
        sum = sum.wrapping_sub(nonzero.count_ones() as u64);
    }

    sum
}
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
fn has_avx2() -> bool {
    false
}

/// Minimum qubit count for parallel row iteration in gate kernels.
#[cfg(feature = "parallel")]
const MIN_QUBITS_FOR_PAR_GATES: usize = 128;

/// Minimum anti-commuting rows to justify Rayon overhead for measurement rowmul.
#[cfg(feature = "parallel")]
const MIN_ANTI_ROWS_FOR_PAR: usize = 4;

/// Send/Sync wrapper for `*mut u64` — parallel tasks access disjoint rows.
#[cfg(feature = "parallel")]
struct SendU64Ptr(*mut u64);
#[cfg(feature = "parallel")]
impl SendU64Ptr {
    #[inline(always)]
    fn ptr(&self) -> *mut u64 {
        self.0
    }
}
#[cfg(feature = "parallel")]
// SAFETY: Used only when each parallel task accesses non-overlapping row regions.
unsafe impl Send for SendU64Ptr {}
#[cfg(feature = "parallel")]
// SAFETY: The pointer itself is read-only; mutation goes through derived slices at disjoint offsets.
unsafe impl Sync for SendU64Ptr {}

/// Send/Sync wrapper for `*mut bool` used in parallel measurement pre-scan.
#[cfg(feature = "parallel")]
struct SendBoolPtr(*mut bool);
#[cfg(feature = "parallel")]
impl SendBoolPtr {
    #[inline(always)]
    fn ptr(&self) -> *mut bool {
        self.0
    }
}
#[cfg(feature = "parallel")]
// SAFETY: Used only when each parallel task accesses a distinct phase element.
unsafe impl Send for SendBoolPtr {}
#[cfg(feature = "parallel")]
// SAFETY: The pointer itself is read-only; mutation goes through distinct indices.
unsafe impl Sync for SendBoolPtr {}

/// Minimum number of u64 words for word-group gate batching to be profitable.
///
/// Below this threshold, the tableau fits in L1/L2 cache and the per-gate
/// match overhead in the batched inner loop exceeds the cache-amortization
/// benefit. At nw=4 (n=193+), each row is 64 bytes (one cache line) and
/// word-group batching avoids repeated full-row iteration per gate.
pub(super) const MIN_WORDS_FOR_BATCH: usize = 4;

/// Compact gate representation for batched word-group execution.
///
/// All gates in a word group target the same u64 word. `a_bit` and `b_bit`
/// are bit positions (0–63) within that word.
#[derive(Clone, Copy)]
pub(super) struct BatchGate {
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
    #[inline(always)]
    pub(super) fn stride(&self) -> usize {
        2 * self.num_words
    }

    #[inline(always)]
    fn apply_h(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xw = row[word];
            let zw = row[nw + word];
            let x = xw & bit_mask;
            let z = zw & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[word] = (xw & !bit_mask) | z;
            row[nw + word] = (zw & !bit_mask) | x;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_s(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[nw + word] ^= x;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_sdg(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            row[nw + word] ^= x;
            let z_new = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z_new != 0);
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_x(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &[u64], phase: &mut bool| {
            let z = row[nw + word] & bit_mask;
            *phase ^= z != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_y(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &[u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (x ^ z) != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_z(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();

        let row_op = |row: &[u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            *phase ^= x != 0;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_cx(&mut self, control: usize, target: usize) {
        let c_word = control / 64;
        let c_bit = control % 64;
        let c_mask = 1u64 << c_bit;
        let t_word = target / 64;
        let t_bit = target % 64;
        let t_mask = 1u64 << t_bit;
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xa = (row[c_word] >> c_bit) & 1;
            let za = (row[nw + c_word] >> c_bit) & 1;
            let xb = (row[t_word] >> t_bit) & 1;
            let zb = (row[nw + t_word] >> t_bit) & 1;

            *phase ^= (xa & zb & (xb ^ za ^ 1)) == 1;

            if xa == 1 {
                row[t_word] ^= t_mask;
            }
            if zb == 1 {
                row[nw + c_word] ^= c_mask;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_cz(&mut self, a: usize, b: usize) {
        let a_word = a / 64;
        let a_bit = a % 64;
        let a_mask = 1u64 << a_bit;
        let b_word = b / 64;
        let b_bit = b % 64;
        let b_mask = 1u64 << b_bit;
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let xa = (row[a_word] >> a_bit) & 1;
            let xb = (row[b_word] >> b_bit) & 1;
            let za = (row[nw + a_word] >> a_bit) & 1;
            let zb = (row[nw + b_word] >> b_bit) & 1;

            *phase ^= (xa & xb & (za ^ zb)) == 1;

            if xb == 1 {
                row[nw + a_word] ^= a_mask;
            }
            if xa == 1 {
                row[nw + b_word] ^= b_mask;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_swap(&mut self, a: usize, b: usize) {
        let a_word = a / 64;
        let a_bit = a % 64;
        let a_mask = 1u64 << a_bit;
        let b_word = b / 64;
        let b_bit = b % 64;
        let b_mask = 1u64 << b_bit;
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], _phase: &mut bool| {
            let xa = (row[a_word] >> a_bit) & 1;
            let xb = (row[b_word] >> b_bit) & 1;
            if xa != xb {
                row[a_word] ^= a_mask;
                row[b_word] ^= b_mask;
            }

            let za = (row[nw + a_word] >> a_bit) & 1;
            let zb = (row[nw + b_word] >> b_bit) & 1;
            if za != zb {
                row[nw + a_word] ^= a_mask;
                row[nw + b_word] ^= b_mask;
            }
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    pub(super) fn sgi_enabled(&self) -> bool {
        let n = self.n;
        if n < 256 {
            return false;
        }
        let total_rows = 2 * n;
        let avg = self.total_weight / total_rows;
        avg < n / 8 && self.sgi_max_active < total_rows / 16
    }

    fn sgi_apply_1q(&mut self, gate: &Gate, q: usize) {
        let word = q / 64;
        let bit_mask = 1u64 << (q % 64);
        let stride = self.stride();
        let nw = self.num_words;

        for &g in &self.qubit_active[q] {
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            match gate {
                Gate::H => {
                    let xw = row[word];
                    let zw = row[nw + word];
                    let x = xw & bit_mask;
                    let z = zw & bit_mask;
                    *phase ^= (x != 0) && (z != 0);
                    row[word] = (xw & !bit_mask) | z;
                    row[nw + word] = (zw & !bit_mask) | x;
                }
                Gate::S => {
                    let x = row[word] & bit_mask;
                    let z = row[nw + word] & bit_mask;
                    *phase ^= (x != 0) && (z != 0);
                    row[nw + word] ^= x;
                }
                Gate::Sdg => {
                    let x = row[word] & bit_mask;
                    row[nw + word] ^= x;
                    let z_new = row[nw + word] & bit_mask;
                    *phase ^= (x != 0) && (z_new != 0);
                }
                Gate::X => {
                    let z = row[nw + word] & bit_mask;
                    *phase ^= z != 0;
                }
                Gate::Y => {
                    let x = row[word] & bit_mask;
                    let z = row[nw + word] & bit_mask;
                    *phase ^= (x ^ z) != 0;
                }
                Gate::Z => {
                    let x = row[word] & bit_mask;
                    *phase ^= x != 0;
                }
                Gate::SX => {
                    let x = row[word] & bit_mask;
                    let z = row[nw + word] & bit_mask;
                    *phase ^= (z != 0) && (x == 0);
                    row[word] ^= z;
                }
                Gate::SXdg => {
                    let x = row[word] & bit_mask;
                    let z = row[nw + word] & bit_mask;
                    *phase ^= (x != 0) && (z != 0);
                    row[word] ^= z;
                }
                _ => {}
            }
        }
    }

    fn sgi_merge_active(&mut self, q_a: usize, q_b: usize) {
        self.sgi_merge_buf.clear();
        let list_a = &self.qubit_active[q_a];
        let list_b = &self.qubit_active[q_b];
        let (mut ia, mut ib) = (0, 0);
        while ia < list_a.len() && ib < list_b.len() {
            if list_a[ia] < list_b[ib] {
                self.sgi_merge_buf.push(list_a[ia]);
                ia += 1;
            } else if list_a[ia] > list_b[ib] {
                self.sgi_merge_buf.push(list_b[ib]);
                ib += 1;
            } else {
                self.sgi_merge_buf.push(list_a[ia]);
                ia += 1;
                ib += 1;
            }
        }
        if ia < list_a.len() {
            self.sgi_merge_buf.extend_from_slice(&list_a[ia..]);
        }
        if ib < list_b.len() {
            self.sgi_merge_buf.extend_from_slice(&list_b[ib..]);
        }
    }

    fn sgi_apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let c_word = ctrl / 64;
        let c_bit = ctrl % 64;
        let c_mask = 1u64 << c_bit;
        let t_word = tgt / 64;
        let t_bit = tgt % 64;
        let t_mask = 1u64 << t_bit;
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(ctrl, tgt);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            let xa = (row[c_word] >> c_bit) & 1;
            let za = (row[nw + c_word] >> c_bit) & 1;
            let xb = (row[t_word] >> t_bit) & 1;
            let zb = (row[nw + t_word] >> t_bit) & 1;

            *phase ^= (xa & zb & (xb ^ za ^ 1)) == 1;

            if xa == 1 {
                row[t_word] ^= t_mask;
            }
            if zb == 1 {
                row[nw + c_word] ^= c_mask;
            }

            let new_xa = (row[c_word] >> c_bit) & 1;
            let new_za = (row[nw + c_word] >> c_bit) & 1;
            let new_xb = (row[t_word] >> t_bit) & 1;
            let new_zb = (row[nw + t_word] >> t_bit) & 1;

            let old_a = xa | za;
            let new_a = new_xa | new_za;
            let old_b = xb | zb;
            let new_b = new_xb | new_zb;

            if new_a != 0 {
                self.sgi_new_a.push(g);
            }
            if old_a != 0 && new_a == 0 {
                self.total_weight -= 1;
            } else if old_a == 0 && new_a != 0 {
                self.total_weight += 1;
            }

            if new_b != 0 {
                self.sgi_new_b.push(g);
            }
            if old_b != 0 && new_b == 0 {
                self.total_weight -= 1;
            } else if old_b == 0 && new_b != 0 {
                self.total_weight += 1;
            }
        }

        std::mem::swap(&mut self.qubit_active[ctrl], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[tgt], &mut self.sgi_new_b);
        let ma = self.qubit_active[ctrl].len();
        let mb = self.qubit_active[tgt].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_apply_cz(&mut self, a: usize, b: usize) {
        let a_word = a / 64;
        let a_bit = a % 64;
        let a_mask = 1u64 << a_bit;
        let b_word = b / 64;
        let b_bit = b % 64;
        let b_mask = 1u64 << b_bit;
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(a, b);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];
            let phase = &mut self.phase[g as usize];

            let xa = (row[a_word] >> a_bit) & 1;
            let xb = (row[b_word] >> b_bit) & 1;
            let za = (row[nw + a_word] >> a_bit) & 1;
            let zb = (row[nw + b_word] >> b_bit) & 1;

            *phase ^= (xa & xb & (za ^ zb)) == 1;

            if xb == 1 {
                row[nw + a_word] ^= a_mask;
            }
            if xa == 1 {
                row[nw + b_word] ^= b_mask;
            }

            let new_xa = (row[a_word] >> a_bit) & 1;
            let new_za = (row[nw + a_word] >> a_bit) & 1;
            let new_xb = (row[b_word] >> b_bit) & 1;
            let new_zb = (row[nw + b_word] >> b_bit) & 1;

            let old_a_active = xa | za;
            let new_a_active = new_xa | new_za;
            let old_b_active = xb | zb;
            let new_b_active = new_xb | new_zb;

            if new_a_active != 0 {
                self.sgi_new_a.push(g);
            }
            if old_a_active != 0 && new_a_active == 0 {
                self.total_weight -= 1;
            } else if old_a_active == 0 && new_a_active != 0 {
                self.total_weight += 1;
            }

            if new_b_active != 0 {
                self.sgi_new_b.push(g);
            }
            if old_b_active != 0 && new_b_active == 0 {
                self.total_weight -= 1;
            } else if old_b_active == 0 && new_b_active != 0 {
                self.total_weight += 1;
            }
        }

        std::mem::swap(&mut self.qubit_active[a], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[b], &mut self.sgi_new_b);
        let ma = self.qubit_active[a].len();
        let mb = self.qubit_active[b].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_apply_swap(&mut self, a: usize, b: usize) {
        let a_word = a / 64;
        let a_bit = a % 64;
        let a_mask = 1u64 << a_bit;
        let b_word = b / 64;
        let b_bit = b % 64;
        let b_mask = 1u64 << b_bit;
        let stride = self.stride();
        let nw = self.num_words;

        self.sgi_merge_active(a, b);
        self.sgi_new_a.clear();
        self.sgi_new_b.clear();

        for i in 0..self.sgi_merge_buf.len() {
            let g = self.sgi_merge_buf[i];
            let row = &mut self.xz[g as usize * stride..(g as usize + 1) * stride];

            let xa = (row[a_word] >> a_bit) & 1;
            let xb = (row[b_word] >> b_bit) & 1;
            if xa != xb {
                row[a_word] ^= a_mask;
                row[b_word] ^= b_mask;
            }

            let za = (row[nw + a_word] >> a_bit) & 1;
            let zb = (row[nw + b_word] >> b_bit) & 1;
            if za != zb {
                row[nw + a_word] ^= a_mask;
                row[nw + b_word] ^= b_mask;
            }

            let new_xa = (row[a_word] >> a_bit) & 1;
            let new_za = (row[nw + a_word] >> a_bit) & 1;
            let new_xb = (row[b_word] >> b_bit) & 1;
            let new_zb = (row[nw + b_word] >> b_bit) & 1;

            if (new_xa | new_za) != 0 {
                self.sgi_new_a.push(g);
            }
            if (new_xb | new_zb) != 0 {
                self.sgi_new_b.push(g);
            }
        }

        std::mem::swap(&mut self.qubit_active[a], &mut self.sgi_new_a);
        std::mem::swap(&mut self.qubit_active[b], &mut self.sgi_new_b);
        let ma = self.qubit_active[a].len();
        let mb = self.qubit_active[b].len();
        if ma > self.sgi_max_active {
            self.sgi_max_active = ma;
        }
        if mb > self.sgi_max_active {
            self.sgi_max_active = mb;
        }
    }

    fn sgi_dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Id => {}
            Gate::X | Gate::Y | Gate::Z | Gate::H | Gate::S | Gate::Sdg | Gate::SX | Gate::SXdg => {
                self.sgi_apply_1q(gate, targets[0]);
            }
            Gate::Cx => self.sgi_apply_cx(targets[0], targets[1]),
            Gate::Cz => self.sgi_apply_cz(targets[0], targets[1]),
            Gate::Swap => self.sgi_apply_swap(targets[0], targets[1]),
            _ => {
                return Err(PrismError::BackendUnsupported {
                    backend: self.name().to_string(),
                    operation: format!(
                        "non-Clifford gate `{}` (stabilizer backend supports Clifford gates only)",
                        gate.name()
                    ),
                });
            }
        }
        Ok(())
    }

    fn sgi_measure(&mut self, qubit: usize, classical_bit: usize) {
        let n = self.n;
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);
        let stride = self.stride();

        let mut p_row: Option<usize> = None;
        for &g in &self.qubit_active[qubit] {
            let g = g as usize;
            if g >= n && g < 2 * n && self.xz[g * stride + word] & bit_mask != 0 {
                p_row = Some(g);
                break;
            }
        }

        if let Some(p_row) = p_row {
            self.sgi_measure_random(p_row, qubit, classical_bit);
        } else {
            let scratch = 2 * n;
            self.zero_row(scratch);

            let destab_active: SmallVec<[usize; 32]> = self.qubit_active[qubit]
                .iter()
                .filter_map(|&g| {
                    let g = g as usize;
                    if g < n && self.xz[g * stride + word] & bit_mask != 0 {
                        Some(g)
                    } else {
                        None
                    }
                })
                .collect();

            for g in destab_active {
                self.rowmul(scratch, g + n);
            }

            let outcome = self.phase[scratch];
            self.classical_bits[classical_bit] = outcome;
        }
    }

    fn sgi_measure_random(&mut self, p_row: usize, qubit: usize, classical_bit: usize) {
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);

        let p_base = p_row * stride;
        let p_data: SmallVec<[u64; 32]> = SmallVec::from_slice(&self.xz[p_base..p_base + stride]);
        let p_phase = self.phase[p_row];

        for i in 0..2 * n {
            if i == p_row {
                continue;
            }
            if self.xz[i * stride + word] & bit_mask != 0 {
                let initial_sum =
                    if p_phase { 2u64 } else { 0 } + if self.phase[i] { 2u64 } else { 0 };
                let row = &mut self.xz[i * stride..(i + 1) * stride];
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                self.phase[i] = (sum & 3) >= 2;
            }
        }

        let d_row = p_row - n;
        let d_base = d_row * stride;
        self.xz.copy_within(p_base..p_base + stride, d_base);
        self.phase[d_row] = self.phase[p_row];

        self.zero_row(p_row);
        let outcome: bool = self.rng.gen();
        self.phase[p_row] = outcome;
        self.xz[p_row * stride + nw + word] |= bit_mask;

        self.classical_bits[classical_bit] = outcome;

        self.rebuild_qubit_active();
    }

    fn rebuild_qubit_active(&mut self) {
        let n = self.n;
        let stride = self.stride();
        let nw = self.num_words;

        for list in &mut self.qubit_active {
            list.clear();
        }
        self.total_weight = 0;

        for g in 0..2 * n {
            let row = &self.xz[g * stride..(g + 1) * stride];
            for w in 0..nw {
                let active = row[w] | row[nw + w];
                let mut bits = active;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    let q = w * 64 + b;
                    if q < n {
                        self.qubit_active[q].push(g as u32);
                        self.total_weight += 1;
                    }
                    bits &= bits - 1;
                }
            }
        }
        self.sgi_max_active = self.qubit_active.iter().map(|a| a.len()).max().unwrap_or(0);
    }

    /// Random-outcome measurement: rowmul anti-commuting rows against the pivot,
    /// then collapse to a Z-eigenstate. Parallelizable (disjoint destinations).
    pub(super) fn measure_random(
        &mut self,
        p_row: usize,
        word: usize,
        bit_mask: u64,
        classical_bit: usize,
    ) {
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();

        let p_base = p_row * stride;
        let p_data: SmallVec<[u64; 32]> = SmallVec::from_slice(&self.xz[p_base..p_base + stride]);
        let p_phase = self.phase[p_row];

        let anti_rows: SmallVec<[usize; 16]> = (0..2 * n)
            .filter(|&r| r != p_row && self.xz[r * stride + word] & bit_mask != 0)
            .collect();

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES && anti_rows.len() >= MIN_ANTI_ROWS_FOR_PAR {
            use rayon::prelude::*;

            let xz_ptr = SendU64Ptr(self.xz.as_mut_ptr());
            let phase_ptr = SendBoolPtr(self.phase.as_mut_ptr());

            // SAFETY: Each row index in anti_rows is unique (collected from a filter
            // with no duplicates) and none equals p_row. Each row occupies
            // [row*stride .. (row+1)*stride] in xz — non-overlapping regions.
            // Phase elements are at distinct indices. p_data is a separate copy.
            anti_rows.par_iter().for_each(|&r| {
                let xz_base = xz_ptr.ptr();
                let ph_base = phase_ptr.ptr();
                let row =
                    unsafe { std::slice::from_raw_parts_mut(xz_base.add(r * stride), stride) };
                let phase = unsafe { &mut *ph_base.add(r) };

                let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                *phase = (sum & 3) >= 2;
            });
        } else {
            for &r in &anti_rows {
                let base = r * stride;
                let row = &mut self.xz[base..base + stride];
                let phase = &mut self.phase[r];
                let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
                let (rx, rz) = row.split_at_mut(nw);
                let sum = rowmul_words(
                    rx,
                    &mut rz[..nw],
                    &p_data[..nw],
                    &p_data[nw..2 * nw],
                    initial_sum,
                );
                *phase = (sum & 3) >= 2;
            }
        }

        #[cfg(not(feature = "parallel"))]
        for &r in &anti_rows {
            let base = r * stride;
            let row = &mut self.xz[base..base + stride];
            let phase = &mut self.phase[r];
            let initial_sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };
            let (rx, rz) = row.split_at_mut(nw);
            let sum = rowmul_words(
                rx,
                &mut rz[..nw],
                &p_data[..nw],
                &p_data[nw..2 * nw],
                initial_sum,
            );
            *phase = (sum & 3) >= 2;
        }

        let dest_row = p_row - n;
        self.copy_row(dest_row, p_row);

        self.zero_row(p_row);
        self.xz[p_row * stride + nw + word] |= bit_mask;

        let outcome: bool = self.rng.gen();
        self.phase[p_row] = outcome;
        self.classical_bits[classical_bit] = outcome;
    }

    #[inline(always)]
    fn apply_sx(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (z != 0) && (x == 0);
            row[word] ^= z;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
        }
    }

    #[inline(always)]
    fn apply_sxdg(&mut self, a: usize) {
        let word = a / 64;
        let bit_mask = 1u64 << (a % 64);
        let stride = self.stride();
        let nw = self.num_words;

        let row_op = |row: &mut [u64], phase: &mut bool| {
            let x = row[word] & bit_mask;
            let z = row[nw + word] & bit_mask;
            *phase ^= (x != 0) && (z != 0);
            row[word] ^= z;
        };

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, phase)| row_op(row, phase));
            return;
        }

        for (row, phase) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
            row_op(row, phase);
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

    pub(super) fn apply_instructions_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
        for (idx, instruction) in instructions.iter().enumerate() {
            if !self.sgi_enabled() {
                return self.apply_instructions_word_batch(&instructions[idx..]);
            }

            match instruction {
                Instruction::Gate { gate, targets } => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                Instruction::Measure {
                    qubit,
                    classical_bit,
                } => {
                    self.sgi_measure(*qubit, *classical_bit);
                }
                Instruction::Barrier { .. } => {}
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    if condition.evaluate(&self.classical_bits) {
                        self.sgi_dispatch_gate(gate, targets)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub(super) fn apply_instructions_word_batch(
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

    pub(super) fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Id => {}
            Gate::X => self.apply_x(targets[0]),
            Gate::Y => self.apply_y(targets[0]),
            Gate::Z => self.apply_z(targets[0]),
            Gate::H => self.apply_h(targets[0]),
            Gate::S => self.apply_s(targets[0]),
            Gate::Sdg => self.apply_sdg(targets[0]),
            Gate::SX => self.apply_sx(targets[0]),
            Gate::SXdg => self.apply_sxdg(targets[0]),
            Gate::Cx => self.apply_cx(targets[0], targets[1]),
            Gate::Cz => self.apply_cz(targets[0], targets[1]),
            Gate::Swap => self.apply_swap(targets[0], targets[1]),
            _ => {
                return Err(PrismError::BackendUnsupported {
                    backend: self.name().to_string(),
                    operation: format!(
                        "non-Clifford gate `{}` (stabilizer backend supports Clifford gates only)",
                        gate.name()
                    ),
                });
            }
        }
        Ok(())
    }

    pub(super) fn apply_gates_only_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
        for (idx, instruction) in instructions.iter().enumerate() {
            if !self.sgi_enabled() {
                return self.apply_gates_only_word_batch(&instructions[idx..]);
            }

            match instruction {
                Instruction::Gate { gate, targets } => {
                    self.sgi_dispatch_gate(gate, targets)?;
                }
                Instruction::Conditional {
                    condition,
                    gate,
                    targets,
                } => {
                    if condition.evaluate(&self.classical_bits) {
                        self.sgi_dispatch_gate(gate, targets)?;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(super) fn apply_gates_only_word_batch(
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
