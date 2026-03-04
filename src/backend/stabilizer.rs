//! Stabilizer (Clifford) simulation backend.
//!
//! Uses the Gottesman-Knill theorem: Clifford circuits can be simulated in
//! O(n²) time per gate using a stabilizer tableau of 2n Pauli strings.
//!
//! # Memory layout
//!
//! Aaronson-Gottesman (2004) tableau with (2n+1) rows:
//! - Rows 0..n: destabilizer generators
//! - Rows n..2n: stabilizer generators
//! - Row 2n: scratch row (measurement computation)
//!
//! Each row stores n X-bits, n Z-bits (bit-packed into `Vec<u64>`), and one
//! phase bit (true = -1). Total memory: O(n²/8) bytes.
//!
//! # Gate support
//!
//! Clifford gates only: H, S, Sdg, X, Y, Z, CX, CZ, SWAP, Id.
//! Non-Clifford gates (T, Rx, Ry, Rz, Fused) return `BackendUnsupported`.
//!
//! # When to prefer this backend
//!
//! - Clifford-only circuits (randomized benchmarking, error correction).
//! - Very large qubit counts (1000+) where statevector is impossible.
//! - Verification of Clifford subcircuits before layering T gates.
//!
//! # Performance characteristics
//!
//! - Gate application: O(n) per gate (iterates 2n+1 rows, constant work per row)
//! - Measurement: O(n²) worst case (rowmul is O(n), applied to up to 2n rows)
//! - Memory: n=1000 → ~500 KB, n=10000 → ~50 MB
//! - Probability extraction: O(2^n * n) — only available for n ≤ 20

use num_complex::Complex64;

use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[allow(dead_code)]
#[inline(always)]
unsafe fn xor_words(dst: *mut u64, src: *const u64, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: caller guarantees non-overlapping valid regions, has_avx2() ensures ISA.
        unsafe {
            xor_words_avx2(dst, src, len);
        }
        return;
    }
    for i in 0..len {
        // SAFETY: caller guarantees valid pointers for len elements.
        unsafe {
            *dst.add(i) ^= *src.add(i);
        }
    }
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[allow(dead_code)]
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

/// Minimum qubit count for parallel row iteration in gate kernels.
#[cfg(feature = "parallel")]
const MIN_QUBITS_FOR_PAR_GATES: usize = 128;

/// Minimum number of u64 words for word-group gate batching to be profitable.
///
/// Below this threshold, the tableau fits in L1/L2 cache and the per-gate
/// match overhead in the batched inner loop exceeds the cache-amortization
/// benefit. At nw=4 (n=193+), each row is 64 bytes (one cache line) and
/// word-group batching avoids repeated full-row iteration per gate.
const MIN_WORDS_FOR_BATCH: usize = 4;

/// Compact gate representation for batched word-group execution.
///
/// All gates in a word group target the same u64 word. `a_bit` and `b_bit`
/// are bit positions (0–63) within that word.
#[derive(Clone, Copy)]
struct BatchGate {
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

/// Clifford-only O(n^2) stabilizer simulation (Aaronson-Gottesman tableau).
pub struct StabilizerBackend {
    n: usize,
    num_words: usize,
    xz: Vec<u64>,
    phase: Vec<bool>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
}

impl StabilizerBackend {
    /// Create a new stabilizer backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            n: 0,
            num_words: 0,
            xz: Vec::new(),
            phase: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    #[inline(always)]
    fn stride(&self) -> usize {
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

    /// Multiply row `h` by row `i` (replace `h` with the Pauli product).
    ///
    /// Fused phase+XOR: Aaronson-Gottesman g-function with wordwise popcount,
    /// with the row XOR update performed in the same loop to eliminate a
    /// separate memory pass.
    fn rowmul(&mut self, h: usize, i: usize) {
        let stride = self.stride();
        let nw = self.num_words;
        let base_h = h * stride;
        let base_i = i * stride;

        let mut sum = if self.phase[i] { 2u64 } else { 0 } + if self.phase[h] { 2u64 } else { 0 };

        for w in 0..nw {
            let x1 = self.xz[base_i + w];
            let z1 = self.xz[base_i + nw + w];
            let x2 = self.xz[base_h + w];
            let z2 = self.xz[base_h + nw + w];

            self.xz[base_h + w] = x1 ^ x2;
            self.xz[base_h + nw + w] = z1 ^ z2;

            if (x1 | z1 | x2 | z2) == 0 {
                continue;
            }

            let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
            let neg = (x1 & z1 & x2 & !z2) | (x1 & !z1 & !x2 & z2) | (!x1 & z1 & x2 & z2);
            sum = sum.wrapping_add(pos.count_ones() as u64);
            sum = sum.wrapping_sub(neg.count_ones() as u64);
        }

        self.phase[h] = (sum & 3) >= 2;
    }

    fn copy_row(&mut self, dst: usize, src: usize) {
        let stride = self.stride();
        let src_start = src * stride;
        let dst_start = dst * stride;
        self.xz
            .copy_within(src_start..src_start + stride, dst_start);
        self.phase[dst] = self.phase[src];
    }

    fn zero_row(&mut self, r: usize) {
        let stride = self.stride();
        let start = r * stride;
        self.xz[start..start + stride].fill(0);
        self.phase[r] = false;
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let n = self.n;
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);
        let stride = self.stride();

        let mut p: Option<usize> = None;
        for i in n..2 * n {
            if self.xz[i * stride + word] & bit_mask != 0 {
                p = Some(i);
                break;
            }
        }

        if let Some(p_row) = p {
            self.measure_random(p_row, word, bit_mask, classical_bit);
        } else {
            let scratch = 2 * n;
            self.zero_row(scratch);

            for i in 0..n {
                if self.xz[i * stride + word] & bit_mask != 0 {
                    self.rowmul(scratch, i + n);
                }
            }

            let outcome = self.phase[scratch];
            self.classical_bits[classical_bit] = outcome;
        }
    }

    /// Random-outcome measurement: rowmul all anti-commuting rows against the
    /// pivot, then collapse the pivot to a Z-eigenstate.
    ///
    /// All rowmul(i, p_row) calls are independent (same read-only source, disjoint
    /// destinations), enabling parallel execution at high qubit counts.
    fn measure_random(&mut self, p_row: usize, word: usize, bit_mask: u64, classical_bit: usize) {
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;

            let p_base = p_row * stride;
            let p_data: Vec<u64> = self.xz[p_base..p_base + stride].to_vec();
            let p_phase = self.phase[p_row];

            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .enumerate()
                .for_each(|(row_idx, (row, phase))| {
                    if row_idx == p_row || row_idx >= 2 * n {
                        return;
                    }
                    if row[word] & bit_mask == 0 {
                        return;
                    }

                    let mut sum = if p_phase { 2u64 } else { 0 } + if *phase { 2u64 } else { 0 };

                    for w in 0..nw {
                        let x1 = p_data[w];
                        let z1 = p_data[nw + w];
                        let x2 = row[w];
                        let z2 = row[nw + w];

                        row[w] = x1 ^ x2;
                        row[nw + w] = z1 ^ z2;

                        if (x1 | z1 | x2 | z2) == 0 {
                            continue;
                        }

                        let pos =
                            (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                        let neg =
                            (x1 & z1 & x2 & !z2) | (x1 & !z1 & !x2 & z2) | (!x1 & z1 & x2 & z2);
                        sum = sum.wrapping_add(pos.count_ones() as u64);
                        sum = sum.wrapping_sub(neg.count_ones() as u64);
                    }

                    *phase = (sum & 3) >= 2;
                });
        } else {
            for i in 0..2 * n {
                if i != p_row && (self.xz[i * stride + word] & bit_mask != 0) {
                    self.rowmul(i, p_row);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..2 * n {
                if i != p_row && (self.xz[i * stride + word] & bit_mask != 0) {
                    self.rowmul(i, p_row);
                }
            }
        }

        let dest_row = p_row - n;
        self.copy_row(dest_row, p_row);

        self.zero_row(p_row);
        self.xz[p_row * stride + nw + word] |= bit_mask;

        let outcome: bool = self.rng.gen();
        self.phase[p_row] = outcome;
        self.classical_bits[classical_bit] = outcome;
    }

    /// Export the stabilizer state as a dense statevector.
    ///
    /// Constructs the 2^n amplitude vector by projecting |0...0⟩ through each
    /// stabilizer generator: |ψ⟩ = ∏_i (I + g_i)/2 |seed⟩, normalized.
    ///
    /// Each projection applies Pauli string g_i to the dense vector in O(2^n),
    /// giving O(n × 2^n) total — same complexity as `compute_probabilities`.
    pub fn export_statevector(&self) -> Result<Vec<Complex64>> {
        if self.n >= usize::BITS as usize {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits (exceeds addressable memory)",
                    self.n
                ),
            });
        }
        let dim = 1usize << self.n;
        let mut check = Vec::<Complex64>::new();
        if check.try_reserve_exact(dim).is_err() {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits ({} bytes required)",
                    self.n,
                    dim * std::mem::size_of::<Complex64>()
                ),
            });
        }
        drop(check);
        Ok(self.compute_statevector())
    }

    /// Build the dense statevector by projecting a support seed through each
    /// stabilizer generator.
    ///
    /// Algorithm:
    /// 1. Gaussian-eliminate to find the support (same as `compute_probabilities`)
    /// 2. Pick the first basis state in the support as seed
    /// 3. Apply projector (I + g_i)/2 for each original stabilizer generator
    /// 4. Normalize
    ///
    /// For Pauli g = (-1)^r × ∏_j X_j^{x_j} Z_j^{z_j}:
    ///   g|y⟩ = (-1)^{r + popcount(z & y)} |y ⊕ x_bits⟩
    fn compute_statevector(&self) -> Vec<Complex64> {
        let n = self.n;
        let dim = 1usize << n;
        let stride = self.stride();
        let nw = self.num_words;

        // Step 1: Find a seed state in the support via Gaussian elimination
        // (reuses the same diagonal-constraint logic as compute_probabilities)
        let seed = self.find_support_seed();

        // Step 2: Initialize from seed
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[seed] = Complex64::new(1.0, 0.0);

        // Step 3: Apply (I + g_i)/2 for each ORIGINAL stabilizer generator.
        // Projectors commute (stabilizer generators commute) so order is irrelevant.
        //
        // AG convention: g = (-1)^r × i^m × ∏_j X_j^{x_j} Z_j^{z_j}
        // where m = popcount(x_bits & z_bits) counts the implicit i-factor from
        // Y-type qubits (Y = iXZ, so each x=1,z=1 qubit contributes factor i).
        //
        // Action: g|y⟩ = (-1)^{r + dot(z,y)} × i^m × |y ⊕ x_bits⟩

        let powers_of_i = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];

        let mut visited_gen = vec![0u32; dim];
        let mut current_gen = 0u32;
        for i in 0..n {
            let row = i + n;
            let base = row * stride;

            let mut x_bits = 0usize;
            let mut z_bits = 0usize;
            for w in 0..nw {
                let shift = w * 64;
                if shift < usize::BITS as usize {
                    x_bits |= (self.xz[base + w] as usize) << shift;
                    z_bits |= (self.xz[base + nw + w] as usize) << shift;
                }
            }
            let r = self.phase[row];

            let m = (x_bits & z_bits).count_ones() as usize;
            let i_factor = powers_of_i[m & 3];
            let base_sign = if r { -1.0 } else { 1.0 };

            if x_bits == 0 {
                for (y, s) in state.iter_mut().enumerate() {
                    let dot_parity = (z_bits & y).count_ones() & 1;
                    let phase_val = if dot_parity == 0 {
                        base_sign
                    } else {
                        -base_sign
                    };
                    if phase_val < 0.0 {
                        *s = Complex64::new(0.0, 0.0);
                    }
                }
            } else {
                current_gen += 1;
                for y in 0..dim {
                    if visited_gen[y] == current_gen {
                        continue;
                    }
                    let partner = y ^ x_bits;
                    visited_gen[partner] = current_gen;

                    let a = state[y];
                    let b = state[partner];

                    let dot_y = (z_bits & y).count_ones() & 1;
                    let real_y = if dot_y == 0 { base_sign } else { -base_sign };
                    let gy_phase = i_factor * real_y;

                    let dot_p = (z_bits & partner).count_ones() & 1;
                    let real_p = if dot_p == 0 { base_sign } else { -base_sign };
                    let gp_phase = i_factor * real_p;

                    state[y] = (a + b * gp_phase) * 0.5;
                    state[partner] = (b + a * gy_phase) * 0.5;
                }
            }
        }

        // Step 4: Normalize
        let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > NORM_CLAMP_MIN {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for amp in &mut state {
                *amp *= inv_norm;
            }
        }

        state
    }

    /// Gaussian-eliminate the stabilizer X-part to separate diagonal (Z-only)
    /// from non-diagonal generators.
    ///
    /// Returns (stab_x, stab_z, stab_phase, diag_indices, num_pivots).
    /// stab_x and stab_z are flat arrays with stride `nw` (row i at offset `i * nw`).
    #[allow(clippy::type_complexity)]
    fn gauss_eliminate_x(&self) -> (Vec<u64>, Vec<u64>, Vec<bool>, Vec<usize>, usize) {
        let n = self.n;
        let stride = self.stride();
        let nw = self.num_words;

        let mut stab_x = vec![0u64; n * nw];
        let mut stab_z = vec![0u64; n * nw];
        let mut stab_phase = vec![false; n];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let src = (i + n) * stride;
            let dst = i * nw;
            stab_x[dst..dst + nw].copy_from_slice(&self.xz[src..src + nw]);
            stab_z[dst..dst + nw].copy_from_slice(&self.xz[src + nw..src + nw + nw]);
            stab_phase[i] = self.phase[i + n];
        }

        let mut has_pivot = vec![false; n];

        #[allow(clippy::needless_range_loop)]
        for col in 0..n {
            let w = col / 64;
            let b = col % 64;
            let mut pivot_row = None;
            for row in 0..n {
                if has_pivot[row] {
                    continue;
                }
                if (stab_x[row * nw + w] >> b) & 1 == 1 {
                    pivot_row = Some(row);
                    break;
                }
            }

            if let Some(pr) = pivot_row {
                has_pivot[pr] = true;
                let pr_off = pr * nw;

                for row in 0..n {
                    if row == pr {
                        continue;
                    }
                    let row_off = row * nw;
                    if (stab_x[row_off + w] >> b) & 1 == 1 {
                        let mut sum = if stab_phase[pr] { 2u64 } else { 0 }
                            + if stab_phase[row] { 2u64 } else { 0 };
                        for ww in 0..nw {
                            let x1 = stab_x[pr_off + ww];
                            let z1 = stab_z[pr_off + ww];
                            let x2 = stab_x[row_off + ww];
                            let z2 = stab_z[row_off + ww];

                            stab_x[row_off + ww] = x1 ^ x2;
                            stab_z[row_off + ww] = z1 ^ z2;

                            if (x1 | z1 | x2 | z2) == 0 {
                                continue;
                            }
                            let pos =
                                (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                            let neg =
                                (x1 & z1 & x2 & !z2) | (x1 & !z1 & !x2 & z2) | (!x1 & z1 & x2 & z2);
                            sum = sum.wrapping_add(pos.count_ones() as u64);
                            sum = sum.wrapping_sub(neg.count_ones() as u64);
                        }
                        stab_phase[row] = (sum & 3) >= 2;
                    }
                }
            }
        }

        let k = has_pivot.iter().filter(|&&p| p).count();
        let diag: Vec<usize> = (0..n).filter(|i| !has_pivot[*i]).collect();

        (stab_x, stab_z, stab_phase, diag, k)
    }

    fn solve_diagonal_seed(
        stab_z: &[u64],
        stab_phase: &[bool],
        diag: &[usize],
        nw: usize,
        n: usize,
    ) -> usize {
        let d = diag.len();
        if d == 0 {
            return 0;
        }

        let mut z_rows: Vec<u64> = Vec::with_capacity(d * nw);
        let mut phases: Vec<bool> = Vec::with_capacity(d);
        for &di in diag {
            z_rows.extend_from_slice(&stab_z[di * nw..(di + 1) * nw]);
            phases.push(stab_phase[di]);
        }

        let mut pivot_col = vec![usize::MAX; d];
        let mut used_cols = vec![false; n];

        for row in 0..d {
            let row_off = row * nw;
            let mut found = None;
            for col in 0..n {
                if used_cols[col] {
                    continue;
                }
                if (z_rows[row_off + col / 64] >> (col % 64)) & 1 == 1 {
                    found = Some(col);
                    break;
                }
            }

            if let Some(col) = found {
                used_cols[col] = true;
                pivot_col[row] = col;
                let w = col / 64;
                let b = col % 64;

                let pivot_z: Vec<u64> = z_rows[row_off..row_off + nw].to_vec();
                let pivot_phase = phases[row];

                #[allow(clippy::needless_range_loop)]
                for other in 0..d {
                    if other == row {
                        continue;
                    }
                    let other_off = other * nw;
                    if (z_rows[other_off + w] >> b) & 1 == 1 {
                        for ww in 0..nw {
                            z_rows[other_off + ww] ^= pivot_z[ww];
                        }
                        phases[other] ^= pivot_phase;
                    }
                }
            }
        }

        let mut seed = 0usize;
        for row in 0..d {
            if pivot_col[row] != usize::MAX && phases[row] {
                seed |= 1 << pivot_col[row];
            }
        }
        seed
    }

    fn find_support_seed(&self) -> usize {
        let nw = self.num_words;
        let (_stab_x, stab_z, stab_phase, diag, _k) = self.gauss_eliminate_x();
        Self::solve_diagonal_seed(&stab_z, &stab_phase, &diag, nw, self.n)
    }

    /// Coset-based probability extraction.
    ///
    /// After Gaussian elimination, the support is a coset of size 2^k defined
    /// by the X-parts of the k non-diagonal generators. Uses GF(2) solve to
    /// find a seed state, then Gray code enumerates all 2^k coset members.
    /// O(n^3/64) for Gaussian elimination + O(2^n) for zeroing + O(2^k) for
    /// coset enumeration (vs old O(2^n × d × n/64) brute-force).
    fn compute_probabilities(&self) -> Vec<f64> {
        let n = self.n;
        let dim = 1usize << n;
        let nw = self.num_words;
        let (stab_x, stab_z, stab_phase, diag, k) = self.gauss_eliminate_x();

        let amplitude_sq = 1.0 / (1u64 << k) as f64;

        let seed = Self::solve_diagonal_seed(&stab_z, &stab_phase, &diag, nw, n);

        if k == n {
            return vec![amplitude_sq; dim];
        }

        let mut non_diag_set = vec![true; n];
        for &di in &diag {
            non_diag_set[di] = false;
        }

        let coset_gens: Vec<usize> = (0..n)
            .filter(|&i| non_diag_set[i])
            .map(|i| {
                let mut x = 0usize;
                #[allow(clippy::needless_range_loop)]
                for w in 0..nw {
                    let shift = w * 64;
                    if shift < usize::BITS as usize {
                        x |= (stab_x[i * nw + w] as usize) << shift;
                    }
                }
                x
            })
            .collect();

        debug_assert_eq!(coset_gens.len(), k);

        let mut probs = vec![0.0f64; dim];

        let coset_size = 1usize << k;
        let mut current = seed;
        probs[current] = amplitude_sq;

        for i in 1..coset_size {
            let bit = i.trailing_zeros() as usize;
            current ^= coset_gens[bit];
            probs[current] = amplitude_sq;
        }

        probs
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
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, p)| process_row(row, p));
            return;
        }

        for (row, p) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
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

        let prepared: Vec<(usize, Vec<PrepOp>)> = word_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(w, g)| (w, prepare_word_ops(g)))
            .collect();

        #[cfg(feature = "parallel")]
        if self.n >= MIN_QUBITS_FOR_PAR_GATES {
            use rayon::prelude::*;
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
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

        for (row, p) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
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

        let stride = self.stride();
        let nw = self.num_words;

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
            self.xz
                .par_chunks_mut(stride)
                .zip(self.phase.par_iter_mut())
                .for_each(|(row, p)| row_op(row, p));
            for group in word_groups.iter_mut() {
                group.clear();
            }
            cross_word.clear();
            return;
        }

        for (row, p) in self.xz.chunks_mut(stride).zip(self.phase.iter_mut()) {
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
    fn classify_gate(gate: &Gate, targets: &[usize]) -> Option<(usize, BatchGate)> {
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

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
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
}

impl Backend for StabilizerBackend {
    fn name(&self) -> &'static str {
        "stabilizer"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let n = num_qubits;
        let nw = n.div_ceil(64);
        let total_rows = 2 * n + 1;
        let stride = 2 * nw;

        self.n = n;
        self.num_words = nw;

        self.xz = vec![0u64; total_rows * stride];
        self.phase = vec![false; total_rows];

        for i in 0..n {
            let word = i / 64;
            let bit = i % 64;
            self.xz[i * stride + word] |= 1u64 << bit;
            self.xz[(i + n) * stride + nw + word] |= 1u64 << bit;
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
            }
            Instruction::Barrier { .. } => {}
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets)?;
                }
            }
        }
        Ok(())
    }

    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        let nw = self.num_words;
        if nw < MIN_WORDS_FOR_BATCH {
            for instruction in instructions {
                self.apply(instruction)?;
            }
            return Ok(());
        }
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

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.n >= usize::BITS as usize {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probability extraction for {} qubits (exceeds addressable memory)",
                    self.n
                ),
            });
        }
        let dim = 1usize << self.n;
        let mut check = Vec::<f64>::new();
        if check.try_reserve_exact(dim).is_err() {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probability extraction for {} qubits ({} bytes required)",
                    self.n,
                    dim * std::mem::size_of::<f64>()
                ),
            });
        }
        drop(check);
        Ok(self.compute_probabilities())
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        self.export_statevector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::sim;

    #[test]
    fn test_init_tableau() {
        let mut b = StabilizerBackend::new(42);
        b.init(3, 0).unwrap();
        assert_eq!(b.n, 3);
        let stride = b.stride();
        let nw = b.num_words;
        // Destabilizer 0 = X_0
        assert_eq!(b.xz[0] & 1, 1);
        assert_eq!((b.xz[0] >> 1) & 1, 0);
        assert_eq!(b.xz[nw] & 1, 0);
        // Stabilizer 0 (row 3) = Z_0
        assert_eq!(b.xz[3 * stride] & 1, 0);
        assert_eq!(b.xz[3 * stride + nw] & 1, 1);
    }

    #[test]
    fn test_x_flips() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true]);
    }

    #[test]
    fn test_z_on_zero() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::Z, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[false]);
    }

    #[test]
    fn test_y_flips() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::Y, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true]);
    }

    #[test]
    fn test_hzh_equals_x() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Z, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true]);
    }

    #[test]
    fn test_s_squared_is_z() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true]);
    }

    #[test]
    fn test_s_sdg_cancel() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Sdg, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[false]);
    }

    #[test]
    fn test_h_superposition_deterministic() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);

        let mut b1 = StabilizerBackend::new(42);
        sim::run_on(&mut b1, &c).unwrap();
        let r1 = b1.classical_results()[0];

        let mut b2 = StabilizerBackend::new(42);
        sim::run_on(&mut b2, &c).unwrap();
        let r2 = b2.classical_results()[0];

        assert_eq!(r1, r2);
    }

    #[test]
    fn test_bell_correlated() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        for seed in [42, 100, 999, 12345] {
            let mut b = StabilizerBackend::new(seed);
            sim::run_on(&mut b, &c).unwrap();
            assert_eq!(
                b.classical_results()[0],
                b.classical_results()[1],
                "Bell state measurements must be equal (seed {seed})"
            );
        }
    }

    #[test]
    fn test_ghz_4_correlated() {
        let mut c = Circuit::new(4, 4);
        c.add_gate(Gate::H, &[0]);
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..4 {
            c.add_measure(i, i);
        }

        for seed in [42, 100, 999] {
            let mut b = StabilizerBackend::new(seed);
            sim::run_on(&mut b, &c).unwrap();
            let results = b.classical_results();
            assert!(
                results.iter().all(|&x| x == results[0]),
                "GHZ-4 measurements must be equal (seed {seed})"
            );
        }
    }

    #[test]
    fn test_swap() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Swap, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true, false]);
    }

    #[test]
    fn test_cz_on_11() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[true, true]);
    }

    #[test]
    fn test_rejects_t_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::T, &[0]);
        let mut b = StabilizerBackend::new(42);
        b.init(1, 0).unwrap();
        let err = b.apply(&c.instructions[0]).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn test_rejects_rx_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(0.5), &[0]);
        let mut b = StabilizerBackend::new(42);
        b.init(1, 0).unwrap();
        let err = b.apply(&c.instructions[0]).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn test_probs_zero_state() {
        let mut b = StabilizerBackend::new(42);
        b.init(1, 0).unwrap();
        let probs = b.probabilities().unwrap();
        assert_eq!(probs, vec![1.0, 0.0]);
    }

    #[test]
    fn test_probs_one_state() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::X, &[0]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let probs = b.probabilities().unwrap();
        assert_eq!(probs, vec![0.0, 1.0]);
    }

    #[test]
    fn test_probs_plus_state() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let probs = b.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!((probs[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_probs_bell_state() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let probs = b.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!(probs[1].abs() < 1e-12);
        assert!(probs[2].abs() < 1e-12);
        assert!((probs[3] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_probs_ghz_4() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let probs = b.probabilities().unwrap();
        assert_eq!(probs.len(), 16);
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!((probs[15] - 0.5).abs() < 1e-12);
        let rest_sum: f64 = probs[1..15].iter().sum();
        assert!(rest_sum.abs() < 1e-12);
    }

    #[test]
    fn test_1000_qubit_ghz() {
        let n = 1000;
        let mut c = Circuit::new(n, n);
        c.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            c.add_measure(i, i);
        }
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let results = b.classical_results();
        assert!(results.iter().all(|&x| x == results[0]));
    }

    #[test]
    fn test_id_no_change() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::Id, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[false]);
    }

    #[test]
    fn test_double_x_is_identity() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 0);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        assert_eq!(b.classical_results(), &[false]);
    }

    #[test]
    fn test_supports_fused_gates() {
        let b = StabilizerBackend::new(42);
        assert!(!b.supports_fused_gates());
    }

    #[test]
    fn test_rejects_cu_gate() {
        let h_mat = Gate::H.matrix_2x2();
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cu(Box::new(h_mat)), &[0, 1]);
        let mut b = StabilizerBackend::new(42);
        b.init(2, 0).unwrap();
        let err = b.apply(&c.instructions[0]).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn test_rejects_mcu_gate() {
        use crate::gates::McuData;
        let x_mat = Gate::X.matrix_2x2();
        let mcu = Gate::Mcu(Box::new(McuData {
            mat: x_mat,
            num_controls: 2,
        }));
        let mut c = Circuit::new(3, 0);
        c.add_gate(mcu, &[0, 1, 2]);
        let mut b = StabilizerBackend::new(42);
        b.init(3, 0).unwrap();
        let err = b.apply(&c.instructions[0]).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn test_export_statevector_zero_state() {
        let mut b = StabilizerBackend::new(42);
        b.init(1, 0).unwrap();
        let sv = b.export_statevector().unwrap();
        assert!((sv[0].re - 1.0).abs() < 1e-10);
        assert!(sv[0].im.abs() < 1e-10);
        assert!(sv[1].norm_sqr() < 1e-10);
    }

    #[test]
    fn test_export_statevector_one_state() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::X, &[0]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let sv = b.export_statevector().unwrap();
        assert!(sv[0].norm_sqr() < 1e-10);
        assert!((sv[1].norm_sqr() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_export_statevector_plus_state() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let sv = b.export_statevector().unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-10);
        assert!((sv[1].re - expected).abs() < 1e-10);
    }

    #[test]
    fn test_export_statevector_minus_state() {
        // H X |0⟩ = H|1⟩ = (|0⟩ - |1⟩)/√2
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::H, &[0]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let sv = b.export_statevector().unwrap();
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((sv[0].re - expected).abs() < 1e-10);
        assert!((sv[1].re + expected).abs() < 1e-10);
    }

    #[test]
    fn test_export_statevector_bell_state_matches_sv() {
        use crate::backend::statevector::StatevectorBackend;

        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);

        let mut stab = StabilizerBackend::new(42);
        sim::run_on(&mut stab, &c).unwrap();
        let stab_sv = stab.export_statevector().unwrap();

        let mut sv = StatevectorBackend::new(42);
        sim::run_on(&mut sv, &c).unwrap();
        let sv_ref = sv.state_vector();

        // Probabilities must match exactly
        for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
            assert!(
                (s.norm_sqr() - r.norm_sqr()).abs() < 1e-10,
                "prob mismatch: stab={}, sv={}",
                s.norm_sqr(),
                r.norm_sqr()
            );
        }

        // Global phase: find ratio between first non-zero pair
        let global_phase = find_global_phase(&stab_sv, sv_ref);

        // After removing global phase, amplitudes must match
        for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
            let adjusted = s * global_phase;
            assert!(
                (adjusted - r).norm() < 1e-10,
                "amplitude mismatch after phase: stab*phase={adjusted:?}, sv={r:?}"
            );
        }
    }

    #[test]
    fn test_export_statevector_ghz3_matches_sv() {
        use crate::backend::statevector::StatevectorBackend;

        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);

        let mut stab = StabilizerBackend::new(42);
        sim::run_on(&mut stab, &c).unwrap();
        let stab_sv = stab.export_statevector().unwrap();

        let mut sv = StatevectorBackend::new(42);
        sim::run_on(&mut sv, &c).unwrap();
        let sv_ref = sv.state_vector();

        let global_phase = find_global_phase(&stab_sv, sv_ref);
        for (s, r) in stab_sv.iter().zip(sv_ref.iter()) {
            let adjusted = s * global_phase;
            assert!(
                (adjusted - r).norm() < 1e-10,
                "GHZ3 mismatch: stab*phase={adjusted:?}, sv={r:?}"
            );
        }
    }

    #[test]
    fn test_export_statevector_complex_clifford_matches_sv() {
        use crate::backend::statevector::StatevectorBackend;

        // Deeper Clifford circuit with S, Sdg, CZ, SX to exercise i-phases
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::SX, &[3]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Sdg, &[1]);
        c.add_gate(Gate::H, &[3]);
        c.add_gate(Gate::S, &[2]);
        c.add_gate(Gate::Swap, &[0, 3]);

        let mut stab = StabilizerBackend::new(42);
        sim::run_on(&mut stab, &c).unwrap();
        let stab_sv = stab.export_statevector().unwrap();

        let mut sv = StatevectorBackend::new(42);
        sim::run_on(&mut sv, &c).unwrap();
        let sv_ref = sv.state_vector();

        let global_phase = find_global_phase(&stab_sv, sv_ref);
        for (i, (s, r)) in stab_sv.iter().zip(sv_ref.iter()).enumerate() {
            let adjusted = s * global_phase;
            assert!(
                (adjusted - r).norm() < 1e-10,
                "4q Clifford mismatch at index {i}: stab*phase={adjusted:?}, sv={r:?}"
            );
        }
    }

    #[test]
    fn test_export_statevector_all_paulis_match_sv() {
        use crate::backend::statevector::StatevectorBackend;

        // Y gate introduces i-phase: Y|0⟩ = i|1⟩
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Y, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::S, &[1]);

        let mut stab = StabilizerBackend::new(42);
        sim::run_on(&mut stab, &c).unwrap();
        let stab_sv = stab.export_statevector().unwrap();

        let mut sv = StatevectorBackend::new(42);
        sim::run_on(&mut sv, &c).unwrap();
        let sv_ref = sv.state_vector();

        let global_phase = find_global_phase(&stab_sv, sv_ref);
        for (i, (s, r)) in stab_sv.iter().zip(sv_ref.iter()).enumerate() {
            let adjusted = s * global_phase;
            assert!(
                (adjusted - r).norm() < 1e-10,
                "Pauli test mismatch at {i}: stab*phase={adjusted:?}, sv={r:?}"
            );
        }
    }

    /// Find the global phase ratio between two state vectors.
    /// Returns the complex scalar c such that stab * c ≈ reference.
    fn find_global_phase(stab: &[Complex64], reference: &[Complex64]) -> Complex64 {
        for (s, r) in stab.iter().zip(reference.iter()) {
            if s.norm_sqr() > 1e-10 && r.norm_sqr() > 1e-10 {
                return r / s;
            }
        }
        Complex64::new(1.0, 0.0)
    }

    #[test]
    fn test_rejects_cphase_gate() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
        let mut b = StabilizerBackend::new(42);
        b.init(2, 0).unwrap();
        let err = b.apply(&c.instructions[0]).unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }
}
