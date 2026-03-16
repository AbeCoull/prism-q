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
use smallvec::SmallVec;

use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[inline(always)]
unsafe fn xor_words(dst: *mut u64, src: *const u64, len: usize) {
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
fn rowmul_words(
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
#[derive(Clone)]
pub struct StabilizerBackend {
    n: usize,
    num_words: usize,
    xz: Vec<u64>,
    phase: Vec<bool>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    qubit_active: Vec<Vec<u32>>,
    total_weight: usize,
    sgi_merge_buf: Vec<u32>,
    sgi_new_a: Vec<u32>,
    sgi_new_b: Vec<u32>,
    sgi_max_active: usize,
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
            qubit_active: Vec::new(),
            total_weight: 0,
            sgi_merge_buf: Vec::new(),
            sgi_new_a: Vec::new(),
            sgi_new_b: Vec::new(),
            sgi_max_active: 0,
        }
    }

    pub fn raw_tableau(&self) -> (&[u64], &[bool]) {
        (&self.xz, &self.phase)
    }

    pub fn into_tableau(self) -> (Vec<u64>, Vec<bool>, usize, usize) {
        (self.xz, self.phase, self.n, self.num_words)
    }

    pub fn apply_gates_only(&mut self, instructions: &[Instruction]) -> Result<()> {
        let nw = self.num_words;
        if nw < MIN_WORDS_FOR_BATCH {
            for instruction in instructions {
                match instruction {
                    Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
                    Instruction::Conditional {
                        condition,
                        gate,
                        targets,
                    } => {
                        if condition.evaluate(&self.classical_bits) {
                            self.dispatch_gate(gate, targets)?;
                        }
                    }
                    _ => {}
                }
            }
            return Ok(());
        }

        if self.sgi_enabled() {
            return self.apply_gates_only_sgi(instructions);
        }

        self.apply_gates_only_word_batch(instructions)
    }

    fn apply_gates_only_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
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

    fn apply_gates_only_word_batch(&mut self, instructions: &[Instruction]) -> Result<()> {
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

    fn sgi_enabled(&self) -> bool {
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

    /// Multiply row `h` by row `i` (replace `h` with the Pauli product).
    ///
    /// Fused phase+XOR: AG g-function with wordwise popcount, row XOR in the
    /// same loop to avoid a separate memory pass.
    fn rowmul(&mut self, h: usize, i: usize) {
        let stride = self.stride();
        let nw = self.num_words;
        let base_h = h * stride;
        let base_i = i * stride;

        let initial_sum =
            if self.phase[i] { 2u64 } else { 0 } + if self.phase[h] { 2u64 } else { 0 };

        // SAFETY: h != i in all callers, so row regions [base_h..base_h+stride]
        // and [base_i..base_i+stride] are non-overlapping.
        let (dst_x, dst_z, src_x, src_z) = unsafe {
            let ptr = self.xz.as_mut_ptr();
            (
                std::slice::from_raw_parts_mut(ptr.add(base_h), nw),
                std::slice::from_raw_parts_mut(ptr.add(base_h + nw), nw),
                std::slice::from_raw_parts(ptr.add(base_i) as *const u64, nw),
                std::slice::from_raw_parts(ptr.add(base_i + nw) as *const u64, nw),
            )
        };

        let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
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
        self.apply_measure_with_info(qubit, classical_bit);
    }

    pub(crate) fn apply_measure_with_info(
        &mut self,
        qubit: usize,
        classical_bit: usize,
    ) -> (bool, Vec<usize>) {
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
            let p_base = p_row * stride;
            let mut support = Vec::new();
            for q in 0..n {
                if self.xz[p_base + q / 64] & (1u64 << (q % 64)) != 0 {
                    support.push(q);
                }
            }
            self.measure_random(p_row, word, bit_mask, classical_bit);
            (true, support)
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
            (false, Vec::new())
        }
    }

    pub(crate) fn batch_measure_ref_info(
        &mut self,
        measurements: &[(usize, usize)],
    ) -> (Vec<bool>, Vec<Vec<usize>>, Vec<bool>) {
        let num_meas = measurements.len();
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();

        let mut is_random = vec![false; num_meas];
        let mut random_x_support: Vec<Vec<usize>> = vec![Vec::new(); num_meas];
        let mut outcomes = vec![false; num_meas];

        let mut qubit_to_meas: Vec<usize> = vec![usize::MAX; n];
        for (mi, &(qubit, _)) in measurements.iter().enumerate() {
            qubit_to_meas[qubit] = mi;
        }

        let mut first_destab = vec![usize::MAX; num_meas];
        let mut match_count = vec![0u16; num_meas];
        let mut match_a = vec![0usize; num_meas];
        let mut match_b = vec![0usize; num_meas];

        let build_index = |first_destab: &mut [usize],
                           match_count: &mut [u16],
                           match_a: &mut [usize],
                           match_b: &mut [usize],
                           xz: &[u64],
                           qubit_to_meas: &[usize],
                           n: usize,
                           nw: usize,
                           stride: usize,
                           num_meas: usize| {
            first_destab.iter_mut().for_each(|v| *v = usize::MAX);
            match_count.iter_mut().for_each(|v| *v = 0);
            for r in 0..2 * n {
                let r_base = r * stride;
                for w in 0..nw {
                    let x_word = xz[r_base + w];
                    if x_word == 0 {
                        continue;
                    }
                    let mut bits = x_word;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as usize;
                        let q = w * 64 + b;
                        if q < n {
                            let mi = qubit_to_meas[q];
                            if mi < num_meas {
                                if r >= n {
                                    if first_destab[mi] == usize::MAX {
                                        first_destab[mi] = r;
                                    }
                                } else {
                                    let c = match_count[mi];
                                    if c == 0 {
                                        match_a[mi] = r;
                                    } else if c == 1 {
                                        match_b[mi] = r;
                                    }
                                    match_count[mi] = c.saturating_add(1);
                                }
                            }
                        }
                        bits &= bits - 1;
                    }
                }
            }
        };

        build_index(
            &mut first_destab,
            &mut match_count,
            &mut match_a,
            &mut match_b,
            &self.xz,
            &qubit_to_meas,
            n,
            nw,
            stride,
            num_meas,
        );

        for mi in 0..num_meas {
            let (qubit, classical_bit) = measurements[mi];
            if first_destab[mi] != usize::MAX {
                let (_, support) = self.apply_measure_with_info(qubit, classical_bit);
                is_random[mi] = true;
                outcomes[mi] = self.classical_bits[classical_bit];
                random_x_support[mi] = support;
                build_index(
                    &mut first_destab,
                    &mut match_count,
                    &mut match_a,
                    &mut match_b,
                    &self.xz,
                    &qubit_to_meas,
                    n,
                    nw,
                    stride,
                    num_meas,
                );
            }
        }

        let mut all_diagonal = true;
        'diag_check: for i in 0..n {
            let base = (i + n) * stride;
            for w in 0..nw {
                if self.xz[base + w] != 0 {
                    all_diagonal = false;
                    break 'diag_check;
                }
            }
        }

        if all_diagonal {
            for i in 0..n {
                let phase_i = self.phase[i + n];
                if !phase_i {
                    continue;
                }
                let base = i * stride;
                for w in 0..nw {
                    let x_word = self.xz[base + w];
                    if x_word == 0 {
                        continue;
                    }
                    let mut bits = x_word;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as usize;
                        let q = w * 64 + b;
                        if q < n {
                            let mi = qubit_to_meas[q];
                            if mi < num_meas && !is_random[mi] {
                                outcomes[mi] ^= true;
                            }
                        }
                        bits &= bits - 1;
                    }
                }
            }
            for mi in 0..num_meas {
                if !is_random[mi] {
                    self.classical_bits[measurements[mi].1] = outcomes[mi];
                }
            }
        } else {
            for mi in 0..num_meas {
                if is_random[mi] {
                    continue;
                }
                let (qubit, classical_bit) = measurements[mi];
                let word = qubit / 64;
                let bit_mask = 1u64 << (qubit % 64);
                let scratch = 2 * n;
                self.zero_row(scratch);
                for i in 0..n {
                    if self.xz[i * stride + word] & bit_mask != 0 {
                        self.rowmul(scratch, i + n);
                    }
                }
                outcomes[mi] = self.phase[scratch];
                self.classical_bits[classical_bit] = outcomes[mi];
            }
        }
        (is_random, random_x_support, outcomes)
    }

    /// Random-outcome measurement: rowmul anti-commuting rows against the pivot,
    /// then collapse to a Z-eigenstate. Parallelizable (disjoint destinations).
    fn measure_random(&mut self, p_row: usize, word: usize, bit_mask: u64, classical_bit: usize) {
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

        let mut remaining: Vec<usize> = (0..n).collect();

        for col in 0..n {
            let w = col / 64;
            let b = col % 64;
            let mut pivot_idx = None;
            for (ri, &row) in remaining.iter().enumerate() {
                if (stab_x[row * nw + w] >> b) & 1 == 1 {
                    pivot_idx = Some(ri);
                    break;
                }
            }

            if let Some(ri) = pivot_idx {
                let pr = remaining.swap_remove(ri);
                let pr_off = pr * nw;

                for row in 0..n {
                    if row == pr {
                        continue;
                    }
                    let row_off = row * nw;
                    if (stab_x[row_off + w] >> b) & 1 == 1 {
                        let initial_sum = if stab_phase[pr] { 2u64 } else { 0 }
                            + if stab_phase[row] { 2u64 } else { 0 };
                        // SAFETY: row != pr, so [row_off..row_off+nw] and
                        // [pr_off..pr_off+nw] are non-overlapping regions.
                        let (dst_x, dst_z, src_x, src_z) = unsafe {
                            let xp = stab_x.as_mut_ptr();
                            let zp = stab_z.as_mut_ptr();
                            (
                                std::slice::from_raw_parts_mut(xp.add(row_off), nw),
                                std::slice::from_raw_parts_mut(zp.add(row_off), nw),
                                std::slice::from_raw_parts(xp.add(pr_off) as *const u64, nw),
                                std::slice::from_raw_parts(zp.add(pr_off) as *const u64, nw),
                            )
                        };
                        let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
                        stab_phase[row] = (sum & 3) >= 2;
                    }
                }
            }
        }

        let k = n - remaining.len();
        let diag = remaining;

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
        let mut available_cols: Vec<usize> = (0..n).collect();

        for row in 0..d {
            let row_off = row * nw;
            let mut found = None;
            for (ci, &col) in available_cols.iter().enumerate() {
                if (z_rows[row_off + col / 64] >> (col % 64)) & 1 == 1 {
                    found = Some(ci);
                    break;
                }
            }

            if let Some(ci) = found {
                let col = available_cols.swap_remove(ci);
                pivot_col[row] = col;
                let w = col / 64;
                let b = col % 64;

                let pivot_z: SmallVec<[u64; 16]> =
                    SmallVec::from_slice(&z_rows[row_off..row_off + nw]);
                let pivot_phase = phases[row];

                #[allow(clippy::needless_range_loop)]
                for other in 0..d {
                    if other == row {
                        continue;
                    }
                    let other_off = other * nw;
                    if (z_rows[other_off + w] >> b) & 1 == 1 {
                        // SAFETY: other_off..other_off+nw and pivot_z are non-overlapping
                        // valid regions of nw u64s. pivot_z was cloned from z_rows at
                        // row_off (row != other), so the regions do not alias.
                        unsafe {
                            xor_words(z_rows.as_mut_ptr().add(other_off), pivot_z.as_ptr(), nw);
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

    fn pcc_apply_cross_word(&mut self, cross_word: &[CrossWordGate]) {
        let total_rows = 2 * self.n + 1;
        let col_words = total_rows.div_ceil(64);
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
            for (row_idx, row) in self.xz.chunks(stride).enumerate() {
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
        for (i, p) in self.phase.iter().enumerate() {
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
            for (row_idx, row) in self.xz.chunks_mut(stride).enumerate() {
                let cw = row_idx / 64;
                let cb = row_idx % 64;
                let xbit = (x_cols[x_off + cw] >> cb) & 1;
                row[word] = (row[word] & !mask) | (xbit << bit);
                let zbit = (z_cols[z_off + cw] >> cb) & 1;
                row[nw + word] = (row[nw + word] & !mask) | (zbit << bit);
            }
        }

        for (i, p) in self.phase.iter_mut().enumerate() {
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

    fn apply_instructions_sgi(&mut self, instructions: &[Instruction]) -> Result<()> {
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

    fn apply_instructions_word_batch(&mut self, instructions: &[Instruction]) -> Result<()> {
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

        self.qubit_active = (0..n).map(|q| vec![q as u32, (n + q) as u32]).collect();
        self.total_weight = 2 * n;
        self.sgi_max_active = 2;

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

        if self.sgi_enabled() {
            return self.apply_instructions_sgi(instructions);
        }

        self.apply_instructions_word_batch(instructions)
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

pub struct FilteredStabilizerBackend {
    num_qubits: usize,
    num_classical_bits: usize,
    clusters: Vec<Option<ClusterState>>,
    qubit_to_cluster: Vec<usize>,
    classical_bits: Vec<bool>,
    seed: u64,
}

struct ClusterState {
    backend: StabilizerBackend,
    qubits: Vec<usize>,
    global_to_local: Vec<usize>,
    local_classical: Vec<usize>,
}

impl FilteredStabilizerBackend {
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            num_classical_bits: 0,
            clusters: Vec::new(),
            qubit_to_cluster: Vec::new(),
            classical_bits: Vec::new(),
            seed,
        }
    }

    pub fn init_with_blocks(
        &mut self,
        num_qubits: usize,
        num_classical_bits: usize,
        blocks: &[Vec<usize>],
    ) -> Result<()> {
        self.num_qubits = num_qubits;
        self.num_classical_bits = num_classical_bits;
        self.qubit_to_cluster = vec![0; num_qubits];
        self.clusters = Vec::with_capacity(blocks.len());

        for (bi, block) in blocks.iter().enumerate() {
            for &q in block {
                self.qubit_to_cluster[q] = bi;
            }

            let mut backend = StabilizerBackend::new(self.seed.wrapping_add(bi as u64));
            backend.init(block.len(), 0)?;

            let mut g2l = vec![0usize; num_qubits];
            for (li, &q) in block.iter().enumerate() {
                g2l[q] = li;
            }

            self.clusters.push(Some(ClusterState {
                backend,
                qubits: block.clone(),
                global_to_local: g2l,
                local_classical: Vec::new(),
            }));
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn merge_clusters(&mut self, ci_a: usize, ci_b: usize) {
        if ci_a == ci_b {
            return;
        }

        let (keep, merge) = if ci_a < ci_b {
            (ci_a, ci_b)
        } else {
            (ci_b, ci_a)
        };
        let merge_state = self.clusters[merge].take().unwrap();

        let keep_state = self.clusters[keep].as_mut().unwrap();

        let old_n = keep_state.qubits.len();
        let merge_n = merge_state.qubits.len();
        let new_n = old_n + merge_n;

        let mut merged_qubits = keep_state.qubits.clone();
        merged_qubits.extend_from_slice(&merge_state.qubits);

        let mut new_backend = StabilizerBackend::new(self.seed.wrapping_add(keep as u64 * 1000));
        new_backend.init(new_n, 0).unwrap();

        copy_tableau_into(&keep_state.backend, &mut new_backend, 0);
        copy_tableau_into(&merge_state.backend, &mut new_backend, old_n);

        let mut merged_classical = keep_state.local_classical.clone();
        merged_classical.extend_from_slice(&merge_state.local_classical);
        new_backend
            .classical_bits
            .resize(merged_classical.len(), false);

        let mut g2l = vec![0usize; self.num_qubits];
        for (li, &q) in merged_qubits.iter().enumerate() {
            g2l[q] = li;
            self.qubit_to_cluster[q] = keep;
        }

        *keep_state = ClusterState {
            backend: new_backend,
            qubits: merged_qubits,
            global_to_local: g2l,
            local_classical: merged_classical,
        };
    }

    fn apply_gate_to_cluster(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let ci = self.qubit_to_cluster[targets[0]];

        if targets.len() > 1 {
            for &t in &targets[1..] {
                let other_ci = self.qubit_to_cluster[t];
                if other_ci != ci {
                    self.merge_clusters(ci, other_ci);
                    return self.apply_gate_to_cluster(gate, targets);
                }
            }
        }

        let cluster = self.clusters[ci].as_mut().unwrap();
        let local_targets: SmallVec<[usize; 4]> = targets
            .iter()
            .map(|&t| cluster.global_to_local[t])
            .collect();

        let local_inst = Instruction::Gate {
            gate: gate.clone(),
            targets: local_targets,
        };
        cluster.backend.apply(&local_inst)
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let ci = self.qubit_to_cluster[qubit];
        let cluster = self.clusters[ci].as_mut().unwrap();
        let local_q = cluster.global_to_local[qubit];

        let local_cbit = cluster
            .local_classical
            .iter()
            .position(|&cb| cb == classical_bit)
            .unwrap_or_else(|| {
                let idx = cluster.local_classical.len();
                cluster.local_classical.push(classical_bit);
                if idx >= cluster.backend.classical_bits.len() {
                    cluster.backend.classical_bits.resize(idx + 1, false);
                }
                idx
            });

        cluster.backend.apply_measure(local_q, local_cbit);
        self.classical_bits[classical_bit] = cluster.backend.classical_bits[local_cbit];
    }
}

fn copy_tableau_into(src: &StabilizerBackend, dst: &mut StabilizerBackend, qubit_offset: usize) {
    let src_n = src.n;
    let src_nw = src.num_words;
    let src_stride = 2 * src_nw;
    let dst_n = dst.n;
    let dst_nw = dst.num_words;
    let dst_stride = 2 * dst_nw;

    for i in 0..src_n {
        let src_row = i;
        let dst_row = qubit_offset + i;

        let old_word = (qubit_offset + i) / 64;
        let old_bit = (qubit_offset + i) % 64;
        dst.xz[dst_row * dst_stride + old_word] &= !(1u64 << old_bit);

        let q_word_offset = qubit_offset / 64;
        let q_bit_offset = qubit_offset % 64;
        if q_bit_offset == 0 {
            for w in 0..src_nw {
                dst.xz[dst_row * dst_stride + q_word_offset + w] = src.xz[src_row * src_stride + w];
            }
            for w in 0..src_nw {
                dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w] =
                    src.xz[src_row * src_stride + src_nw + w];
            }
        } else {
            for w in 0..src_nw {
                let val = src.xz[src_row * src_stride + w];
                dst.xz[dst_row * dst_stride + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_row * dst_stride + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
            for w in 0..src_nw {
                let val = src.xz[src_row * src_stride + src_nw + w];
                dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
        }
        dst.phase[dst_row] = src.phase[src_row];

        let src_stab = src_n + i;
        let dst_stab = dst_n + qubit_offset + i;

        let old_word_s = (qubit_offset + i) / 64;
        let old_bit_s = (qubit_offset + i) % 64;
        dst.xz[dst_stab * dst_stride + dst_nw + old_word_s] &= !(1u64 << old_bit_s);

        if q_bit_offset == 0 {
            for w in 0..src_nw {
                dst.xz[dst_stab * dst_stride + q_word_offset + w] =
                    src.xz[src_stab * src_stride + w];
            }
            for w in 0..src_nw {
                dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w] =
                    src.xz[src_stab * src_stride + src_nw + w];
            }
        } else {
            for w in 0..src_nw {
                let val = src.xz[src_stab * src_stride + w];
                dst.xz[dst_stab * dst_stride + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_stab * dst_stride + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
            for w in 0..src_nw {
                let val = src.xz[src_stab * src_stride + src_nw + w];
                dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
        }
        dst.phase[dst_stab] = src.phase[src_stab];
    }
}

impl Backend for FilteredStabilizerBackend {
    fn name(&self) -> &'static str {
        "FilteredStabilizer"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.num_classical_bits = num_classical_bits;
        self.qubit_to_cluster = vec![0; num_qubits];
        self.clusters.clear();

        for i in 0..num_qubits {
            self.qubit_to_cluster[i] = i;
            let mut backend = StabilizerBackend::new(self.seed.wrapping_add(i as u64));
            backend.init(1, 0)?;
            let mut g2l = vec![0usize; num_qubits];
            g2l[i] = 0;
            self.clusters.push(Some(ClusterState {
                backend,
                qubits: vec![i],
                global_to_local: g2l,
                local_classical: Vec::new(),
            }));
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => {
                self.apply_gate_to_cluster(gate, targets)?;
            }
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
                    self.apply_gate_to_cluster(gate, targets)?;
                }
            }
        }
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.num_qubits >= crate::backend::MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!("probability extraction for {} qubits", self.num_qubits),
            });
        }

        let mut blocks: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
        for cluster in self.clusters.iter().flatten() {
            let probs = cluster.backend.compute_probabilities();
            blocks.push((probs, cluster.qubits.clone()));
        }

        if blocks.len() == 1 && blocks[0].1.iter().enumerate().all(|(i, &q)| i == q) {
            return Ok(blocks.into_iter().next().unwrap().0);
        }

        Ok(crate::sim::merge_probabilities(&blocks, self.num_qubits))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
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

    #[test]
    fn test_xor_words_various_lengths() {
        for len in 0..=17 {
            let src: Vec<u64> = (0..len)
                .map(|i| 0xAAAA_BBBB_CCCC_0000u64 | i as u64)
                .collect();
            let original: Vec<u64> = (0..len)
                .map(|i| 0x1111_2222_3333_0000u64 | (i as u64 * 7))
                .collect();

            let mut expected = original.clone();
            for i in 0..len {
                expected[i] ^= src[i];
            }

            let mut actual = original.clone();
            if len > 0 {
                unsafe { xor_words(actual.as_mut_ptr(), src.as_ptr(), len) };
            }

            assert_eq!(actual, expected, "xor_words mismatch at len={len}");
        }
    }

    #[test]
    fn test_xor_words_all_ones_and_zeros() {
        let len = 8;
        let src = vec![u64::MAX; len];
        let mut dst = vec![0u64; len];
        unsafe { xor_words(dst.as_mut_ptr(), src.as_ptr(), len) };
        assert!(dst.iter().all(|&v| v == u64::MAX));

        unsafe { xor_words(dst.as_mut_ptr(), src.as_ptr(), len) };
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_rowmul_words_zero_src() {
        let nw = 4;
        let mut dst_x = vec![0xFFu64; nw];
        let mut dst_z = vec![0xAAu64; nw];
        let src_x = vec![0u64; nw];
        let src_z = vec![0u64; nw];
        let sum = rowmul_words(&mut dst_x, &mut dst_z, &src_x, &src_z, 0);
        assert_eq!(dst_x, vec![0xFFu64; nw]);
        assert_eq!(dst_z, vec![0xAAu64; nw]);
        assert_eq!(sum & 3, 0);
    }

    #[test]
    fn test_rowmul_words_matches_manual() {
        let nw = 3;
        let src_x = vec![0b1010u64, 0b1100u64, 0b0011u64];
        let src_z = vec![0b0110u64, 0b1010u64, 0b1001u64];
        let orig_dst_x = vec![0b1100u64, 0b0110u64, 0b1010u64];
        let orig_dst_z = vec![0b0011u64, 0b1001u64, 0b0110u64];

        let mut manual_x = orig_dst_x.clone();
        let mut manual_z = orig_dst_z.clone();
        let mut manual_sum = 4u64;
        for w in 0..nw {
            let x1 = src_x[w];
            let z1 = src_z[w];
            let x2 = manual_x[w];
            let z2 = manual_z[w];
            let new_x = x1 ^ x2;
            let new_z = z1 ^ z2;
            manual_x[w] = new_x;
            manual_z[w] = new_z;
            if (x1 | z1 | x2 | z2) != 0 {
                let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
                let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                manual_sum = manual_sum.wrapping_add(2 * pos.count_ones() as u64);
                manual_sum = manual_sum.wrapping_sub(nonzero.count_ones() as u64);
            }
        }

        let mut fn_x = orig_dst_x.clone();
        let mut fn_z = orig_dst_z.clone();
        let fn_sum = rowmul_words(&mut fn_x, &mut fn_z, &src_x, &src_z, 4);

        assert_eq!(fn_x, manual_x);
        assert_eq!(fn_z, manual_z);
        assert_eq!(fn_sum & 3, manual_sum & 3);
    }

    #[test]
    fn test_rowmul_words_phase_y_times_x() {
        let src_x = vec![1u64];
        let src_z = vec![1u64];
        let mut dst_x = vec![1u64];
        let mut dst_z = vec![0u64];
        let sum = rowmul_words(&mut dst_x, &mut dst_z, &src_x, &src_z, 0);
        assert_eq!(dst_x[0], 0);
        assert_eq!(dst_z[0], 1);
        assert!(
            (sum & 3) >= 2,
            "Y*X should give phase -1 (sum&3={})",
            sum & 3
        );
    }

    #[test]
    fn test_rowmul_refactor_preserves_ghz_correctness() {
        let n = 10;
        let mut c = Circuit::new(n, n);
        c.add_gate(Gate::H, &[0]);
        for i in 0..n - 1 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        for i in 0..n {
            c.add_measure(i, i);
        }

        for seed in 0..20u64 {
            let mut b = StabilizerBackend::new(seed);
            sim::run_on(&mut b, &c).unwrap();
            let results = b.classical_results();
            let first = results[0];
            assert!(
                results.iter().all(|&r| r == first),
                "GHZ violation at seed {seed}: {results:?}"
            );
        }
    }

    #[test]
    fn test_rowmul_refactor_preserves_probabilities() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let mut b = StabilizerBackend::new(42);
        sim::run_on(&mut b, &c).unwrap();
        let probs = b.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!(probs[1].abs() < 1e-10);
        assert!(probs[2].abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
        assert!(probs[4].abs() < 1e-10);
        assert!(probs[5].abs() < 1e-10);
        assert!((probs[6]).abs() < 1e-10);
        assert!(probs[7].abs() < 1e-10);
    }

    #[test]
    fn test_rowmul_multi_word_correctness() {
        let n = 65;
        let mut c = Circuit::new(n, n);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 64]);
        c.add_measure(0, 0);
        c.add_measure(64, 1);

        for seed in 0..10u64 {
            let mut b = StabilizerBackend::new(seed);
            sim::run_on(&mut b, &c).unwrap();
            let results = b.classical_results();
            assert_eq!(
                results[0], results[1],
                "Bell pair violation at seed {seed}: q0={}, q64={}",
                results[0], results[1]
            );
        }
    }

    #[test]
    fn test_sgi_500q_clifford_d10_matches_gate_by_gate() {
        use crate::circuits;
        let n = 500;
        let mut circuit = circuits::clifford_heavy_circuit(n, 10, 42);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let mut b1 = StabilizerBackend::new(42);
        b1.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for instr in &circuit.instructions {
            b1.apply(instr).unwrap();
        }
        let r1 = b1.classical_results().to_vec();

        let mut b2 = StabilizerBackend::new(42);
        sim::run_on(&mut b2, &circuit).unwrap();
        let r2 = b2.classical_results().to_vec();

        assert_eq!(
            r1, r2,
            "SGI 500q Clifford d10: gate-by-gate vs apply_instructions mismatch"
        );
    }

    #[test]
    fn test_sgi_300q_ghz_all_agree() {
        let n = 300;
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
        assert!(
            results.iter().all(|&x| x == results[0]),
            "GHZ 300q: not all qubits agree"
        );
    }

    #[test]
    fn test_sgi_index_consistency() {
        let n = 300;
        let mut circuit = crate::circuits::clifford_heavy_circuit(n, 5, 42);
        circuit.num_classical_bits = 0;

        let mut b = StabilizerBackend::new(42);
        b.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        b.apply_instructions(&circuit.instructions).unwrap();

        let stride = b.stride();
        let nw = b.num_words;
        for q in 0..n {
            for &g in &b.qubit_active[q] {
                let g = g as usize;
                let row = &b.xz[g * stride..(g + 1) * stride];
                let word = q / 64;
                let bit_mask = 1u64 << (q % 64);
                let x = row[word] & bit_mask;
                let z = row[nw + word] & bit_mask;
                assert!(
                    x != 0 || z != 0,
                    "qubit_active[{q}] contains generator {g} which has I on qubit {q}"
                );
            }
        }

        for g in 0..2 * n {
            let row = &b.xz[g * stride..(g + 1) * stride];
            for q in 0..n {
                let word = q / 64;
                let bit_mask = 1u64 << (q % 64);
                let x = row[word] & bit_mask;
                let z = row[nw + word] & bit_mask;
                let active = x != 0 || z != 0;
                let in_index = b.qubit_active[q].contains(&(g as u32));
                assert_eq!(
                    active, in_index,
                    "generator {g} qubit {q}: active={active} but in_index={in_index}"
                );
            }
        }
    }

    #[test]
    fn test_pcc_random_pairs_matches_gate_by_gate() {
        use crate::circuits;
        let n = 500;
        let mut circuit = circuits::clifford_random_pairs(n, 10, 42);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let mut b1 = StabilizerBackend::new(42);
        b1.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for instr in &circuit.instructions {
            b1.apply(instr).unwrap();
        }
        let r1 = b1.classical_results().to_vec();

        let mut b2 = StabilizerBackend::new(42);
        sim::run_on(&mut b2, &circuit).unwrap();
        let r2 = b2.classical_results().to_vec();

        assert_eq!(
            r1, r2,
            "PCC 500q random-pairs d10: gate-by-gate vs apply_instructions mismatch"
        );
    }

    #[test]
    fn filtered_bell_pairs_correct() {
        use crate::circuits;

        // Test pre-measurement probabilities (no RNG dependence)
        let n_pairs = 5;
        let n = n_pairs * 2;
        let circuit = circuits::independent_bell_pairs(n_pairs);

        let mut filt = FilteredStabilizerBackend::new(42);
        filt.init(n, 0).unwrap();
        for inst in &circuit.instructions {
            filt.apply(inst).unwrap();
        }

        let filt_probs = filt.probabilities().unwrap();
        let mono_probs = {
            let mut mono = StabilizerBackend::new(42);
            mono.init(n, 0).unwrap();
            mono.apply_instructions(&circuit.instructions).unwrap();
            mono.compute_probabilities()
        };

        assert_eq!(filt_probs.len(), mono_probs.len());
        for (i, (&f, &m)) in filt_probs.iter().zip(mono_probs.iter()).enumerate() {
            assert!(
                (f - m).abs() < 1e-10,
                "prob mismatch at index {i}: filtered={f}, monolithic={m}"
            );
        }
    }

    #[test]
    fn filtered_bell_pairs_measurement() {
        use crate::circuits;
        let n_pairs = 10;
        let n = n_pairs * 2;
        let mut circuit = circuits::independent_bell_pairs(n_pairs);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }

        let mut filt = FilteredStabilizerBackend::new(42);
        filt.init(n, n).unwrap();
        for inst in &circuit.instructions {
            filt.apply(inst).unwrap();
        }
        let bits = filt.classical_results();

        for i in 0..n_pairs {
            assert_eq!(
                bits[2 * i],
                bits[2 * i + 1],
                "filtered: bell pair {i} qubits disagree"
            );
        }
    }

    #[test]
    fn filtered_with_blocks_matches_monolithic() {
        use crate::circuits;
        let n_pairs = 5;
        let n = n_pairs * 2;
        let circuit = circuits::independent_bell_pairs(n_pairs);

        let blocks = circuit.independent_subsystems();
        assert_eq!(blocks.len(), n_pairs);

        let mut filt = FilteredStabilizerBackend::new(42);
        filt.init_with_blocks(n, 0, &blocks).unwrap();
        for inst in &circuit.instructions {
            filt.apply(inst).unwrap();
        }
        let filt_probs = filt.probabilities().unwrap();

        let mut mono = StabilizerBackend::new(42);
        mono.init(n, 0).unwrap();
        mono.apply_instructions(&circuit.instructions).unwrap();
        let mono_probs = mono.compute_probabilities();

        for (i, (&f, &m)) in filt_probs.iter().zip(mono_probs.iter()).enumerate() {
            assert!(
                (f - m).abs() < 1e-10,
                "prob mismatch at index {i}: filtered={f}, monolithic={m}"
            );
        }
    }
}
