//! Fused two-qubit gate kernels: the 4x4 matrix sweep behind [`PreparedGate2q`] in every
//! tier, one 4-element group or a batch of adjacent groups per step.

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use super::{SimdTier, avx2_2q_enabled, complex_mul_sse2, simd_tier};

/// Pre-broadcast 4×4 complex matrix for SIMD 2-qubit gate application.
///
/// Each of the 16 complex matrix elements is split into `rr = [re, re]` and
/// `ii = [im, im]` for the standard permute-mul-fmaddsub complex multiply.
#[cfg(target_arch = "x86_64")]
struct Mat4x4Broadcast {
    rr: [__m128d; 16],
    ii: [__m128d; 16],
}

/// 256-bit broadcast variant of [`Mat4x4Broadcast`] for AVX2 paired-group kernels.
///
/// Each register holds the same `re` (or `im`) value broadcast across 4 lanes,
/// so a single `_mm256_fmaddsub_pd` performs the complex multiply for two
/// 4-element groups simultaneously.
#[cfg(target_arch = "x86_64")]
struct Mat4x4Broadcast256 {
    rr: [__m256d; 16],
    ii: [__m256d; 16],
}

#[cfg(target_arch = "x86_64")]
impl Mat4x4Broadcast256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn from_matrix(mat: &[[Complex64; 4]; 4]) -> Self {
        let mut rr = [_mm256_setzero_pd(); 16];
        let mut ii = [_mm256_setzero_pd(); 16];
        for (r, row) in mat.iter().enumerate() {
            for (c, elem) in row.iter().enumerate() {
                let idx = r * 4 + c;
                rr[idx] = _mm256_set1_pd(elem.re);
                ii[idx] = _mm256_set1_pd(elem.im);
            }
        }
        Self { rr, ii }
    }
}

#[cfg(target_arch = "x86_64")]
impl Mat4x4Broadcast {
    #[inline(always)]
    fn from_matrix(mat: &[[Complex64; 4]; 4]) -> Self {
        // SAFETY: x86_64 guarantees SSE2, and these intrinsics only broadcast
        // scalar f64 values into SIMD registers.
        unsafe {
            let mut rr = [_mm_setzero_pd(); 16];
            let mut ii = [_mm_setzero_pd(); 16];
            for (r, row) in mat.iter().enumerate() {
                for (c, elem) in row.iter().enumerate() {
                    let idx = r * 4 + c;
                    rr[idx] = _mm_set1_pd(elem.re);
                    ii[idx] = _mm_set1_pd(elem.im);
                }
            }
            Self { rr, ii }
        }
    }
}

/// Apply one 4-element group of a fused 2q gate using FMA intrinsics.
/// Shared by both the sequential loop and the parallel per-group entry point.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn apply_fused_2q_group_fma_inner(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s0 = _mm_loadu_pd(state.add(i[0] * 2));
        let s1 = _mm_loadu_pd(state.add(i[1] * 2));
        let s2 = _mm_loadu_pd(state.add(i[2] * 2));
        let s3 = _mm_loadu_pd(state.add(i[3] * 2));

        let sf0 = _mm_shuffle_pd(s0, s0, 0b01);
        let sf1 = _mm_shuffle_pd(s1, s1, 0b01);
        let sf2 = _mm_shuffle_pd(s2, s2, 0b01);
        let sf3 = _mm_shuffle_pd(s3, s3, 0b01);

        // Two independent chains per row, the re*z terms and the im*swap(z)
        // terms, joined by one addsub (subtract in the real lanes, add in the
        // imaginary ones, which is the complex product). Nine ops against
        // eight for a single chain, but a depth of five FMA latencies instead
        // of eight: with four rows in flight the single chain left the
        // scheduler waiting on latency at about twice the port-limited cost.
        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut re = _mm_mul_pd(mat.rr[off], s0);
                let mut im = _mm_mul_pd(mat.ii[off], sf0);
                re = _mm_fmadd_pd(mat.rr[off + 1], s1, re);
                im = _mm_fmadd_pd(mat.ii[off + 1], sf1, im);
                re = _mm_fmadd_pd(mat.rr[off + 2], s2, re);
                im = _mm_fmadd_pd(mat.ii[off + 2], sf2, im);
                re = _mm_fmadd_pd(mat.rr[off + 3], s3, re);
                im = _mm_fmadd_pd(mat.ii[off + 3], sf3, im);
                _mm_storeu_pd(state.add(i[$r] * 2), _mm_addsub_pd(re, im));
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
}

/// Visit every base index of a two-qubit sweep, `step` groups at a time.
///
/// A base has zeros at bits `lo` and `hi` (`lo < hi`), so the set splits into
/// three index segments: the bits below `lo`, the bits between, and the bits
/// above `hi`. Walking them as nested loops replaces two variable-shift
/// `insert_zero_bit` calls per group with an add; in the AVX2 pair loop those
/// shifts were a third of the instruction stream and the loop read at about
/// twice its port-limited cost.
#[inline(always)]
fn for_each_2q_base(n_iter: usize, lo: usize, hi: usize, step: usize, mut f: impl FnMut(usize)) {
    let a_len = 1usize << lo;
    let b_len = 1usize << (hi - lo - 1);
    let c_len = n_iter / (a_len * b_len);
    for c in 0..c_len {
        let cbase = c << (hi + 1);
        for b in 0..b_len {
            let bbase = cbase + (b << (lo + 1));
            let mut a = 0;
            while a < a_len {
                f(bbase + a);
                a += step;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn apply_fused_2q_loop_fma(
    state: *mut f64,
    n_iter: usize,
    lo: usize,
    hi: usize,
    mask0: usize,
    mask1: usize,
    mat: &Mat4x4Broadcast,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        for_each_2q_base(n_iter, lo, hi, 1, |base| {
            let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
            apply_fused_2q_group_fma_inner(state, i, mat);
        });
    }
}

/// Apply two consecutive 4-element groups (k and k+1) of a 2q gate using AVX2.
///
/// Precondition: `iA[r]` and `iA[r] + 1` must be the indices of group A's row r
/// and group B's row r respectively, so a single 256-bit load reads both
/// `state[iA[r]]` and `state[iB[r]]` contiguously. This holds when the lower
/// gate qubit `lo` is > 0 and the caller pairs `k` with `k+1`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_fused_2q_pair_avx2_inner(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast256) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s0 = _mm256_loadu_pd(state.add(i[0] * 2));
        let s1 = _mm256_loadu_pd(state.add(i[1] * 2));
        let s2 = _mm256_loadu_pd(state.add(i[2] * 2));
        let s3 = _mm256_loadu_pd(state.add(i[3] * 2));

        let sf0 = _mm256_shuffle_pd(s0, s0, 0b0101);
        let sf1 = _mm256_shuffle_pd(s1, s1, 0b0101);
        let sf2 = _mm256_shuffle_pd(s2, s2, 0b0101);
        let sf3 = _mm256_shuffle_pd(s3, s3, 0b0101);

        // Same two chains as the 128-bit group kernel, joined by one addsub.
        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut re = _mm256_mul_pd(mat.rr[off], s0);
                let mut im = _mm256_mul_pd(mat.ii[off], sf0);
                re = _mm256_fmadd_pd(mat.rr[off + 1], s1, re);
                im = _mm256_fmadd_pd(mat.ii[off + 1], sf1, im);
                re = _mm256_fmadd_pd(mat.rr[off + 2], s2, re);
                im = _mm256_fmadd_pd(mat.ii[off + 2], sf2, im);
                re = _mm256_fmadd_pd(mat.rr[off + 3], s3, re);
                im = _mm256_fmadd_pd(mat.ii[off + 3], sf3, im);
                _mm256_storeu_pd(state.add(i[$r] * 2), _mm256_addsub_pd(re, im));
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
}

/// Apply two groups (A and B) of a 2q gate when `lo == 0` using AVX2.
///
/// With `lo == 0`, `(bit0=0, bit0=1)` of one group at fixed `hi` is contiguous
/// in memory. Loading `va_lo = [A(0,0), A(1,0)]` and `vb_lo = [B(0,0), B(1,0)]`
/// (and similarly `va_hi, vb_hi` for `bit(hi) = 1`) and transposing via
/// `_mm256_permute2f128_pd` (`0x20` / `0x31`) packs `[A(row), B(row)]` into
/// one 256-bit register for the standard two-chain multiply-addsub.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_fused_2q_pair_lo0_avx2_inner<const Q0_IS_LO: bool>(
    pa_lo: *mut f64,
    pb_lo: *mut f64,
    pa_hi: *mut f64,
    pb_hi: *mut f64,
    mat: &Mat4x4Broadcast256,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let va_lo = _mm256_loadu_pd(pa_lo);
        let vb_lo = _mm256_loadu_pd(pb_lo);
        let va_hi = _mm256_loadu_pd(pa_hi);
        let vb_hi = _mm256_loadu_pd(pb_hi);

        let p_00 = _mm256_permute2f128_pd(va_lo, vb_lo, 0x20);
        let p_10 = _mm256_permute2f128_pd(va_lo, vb_lo, 0x31);
        let p_01 = _mm256_permute2f128_pd(va_hi, vb_hi, 0x20);
        let p_11 = _mm256_permute2f128_pd(va_hi, vb_hi, 0x31);

        let (s0, s1, s2, s3) = if Q0_IS_LO {
            (p_00, p_01, p_10, p_11)
        } else {
            (p_00, p_10, p_01, p_11)
        };

        let sf0 = _mm256_shuffle_pd(s0, s0, 0b0101);
        let sf1 = _mm256_shuffle_pd(s1, s1, 0b0101);
        let sf2 = _mm256_shuffle_pd(s2, s2, 0b0101);
        let sf3 = _mm256_shuffle_pd(s3, s3, 0b0101);

        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut re = _mm256_mul_pd(mat.rr[off], s0);
                let mut im = _mm256_mul_pd(mat.ii[off], sf0);
                re = _mm256_fmadd_pd(mat.rr[off + 1], s1, re);
                im = _mm256_fmadd_pd(mat.ii[off + 1], sf1, im);
                re = _mm256_fmadd_pd(mat.rr[off + 2], s2, re);
                im = _mm256_fmadd_pd(mat.ii[off + 2], sf2, im);
                re = _mm256_fmadd_pd(mat.rr[off + 3], s3, re);
                im = _mm256_fmadd_pd(mat.ii[off + 3], sf3, im);
                _mm256_addsub_pd(re, im)
            }};
        }

        let (o_00, o_10) = if Q0_IS_LO {
            (row!(0), row!(2))
        } else {
            (row!(0), row!(1))
        };
        _mm256_storeu_pd(pa_lo, _mm256_permute2f128_pd(o_00, o_10, 0x20));
        _mm256_storeu_pd(pb_lo, _mm256_permute2f128_pd(o_00, o_10, 0x31));

        let (o_01, o_11) = if Q0_IS_LO {
            (row!(1), row!(3))
        } else {
            (row!(2), row!(3))
        };
        _mm256_storeu_pd(pa_hi, _mm256_permute2f128_pd(o_01, o_11, 0x20));
        _mm256_storeu_pd(pb_hi, _mm256_permute2f128_pd(o_01, o_11, 0x31));
    }
}

/// Pair-batched 2q kernel main loop. Pairs adjacent k/k+1 groups when `lo > 0`,
/// or transposes `(bit0=0, bit0=1)` pairs via `_mm256_permute2f128_pd` when
/// `lo == 0` and `n_iter >= 2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn apply_fused_2q_loop_avx2(
    state: *mut f64,
    n_iter: usize,
    lo: usize,
    hi: usize,
    mask0: usize,
    mask1: usize,
    mat256: &Mat4x4Broadcast256,
    mat128: &Mat4x4Broadcast,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if lo == 0 {
            if n_iter < 2 {
                for_each_2q_base(n_iter, lo, hi, 1, |base| {
                    let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
                    apply_fused_2q_group_fma_inner(state, i, mat128);
                });
                return;
            }
            let q0_is_lo = mask0 < mask1;
            let mask_hi = mask0.max(mask1);
            if hi == 1 {
                let mut k = 0;
                while k < n_iter {
                    let p = state.add(k * 8);
                    let pa_lo = p;
                    let pa_hi = p.add(4);
                    let pb_lo = p.add(8);
                    let pb_hi = p.add(12);
                    if q0_is_lo {
                        apply_fused_2q_pair_lo0_avx2_inner::<true>(
                            pa_lo, pb_lo, pa_hi, pb_hi, mat256,
                        );
                    } else {
                        apply_fused_2q_pair_lo0_avx2_inner::<false>(
                            pa_lo, pb_lo, pa_hi, pb_hi, mat256,
                        );
                    }
                    k += 2;
                }
            } else {
                let b_len = 1usize << (hi - 1);
                let c_len = n_iter / b_len;
                let hi_off = mask_hi * 2;
                for c in 0..c_len {
                    let cbase = c << (hi + 1);
                    let mut b = 0;
                    while b < b_len {
                        let base_a = cbase + (b << 1);
                        let p_lo = state.add(base_a * 2);
                        let p_hi = p_lo.add(hi_off);
                        if q0_is_lo {
                            apply_fused_2q_pair_lo0_avx2_inner::<true>(
                                p_lo,
                                p_lo.add(4),
                                p_hi,
                                p_hi.add(4),
                                mat256,
                            );
                        } else {
                            apply_fused_2q_pair_lo0_avx2_inner::<false>(
                                p_lo,
                                p_lo.add(4),
                                p_hi,
                                p_hi.add(4),
                                mat256,
                            );
                        }
                        b += 2;
                    }
                }
            }
            return;
        }
        // `lo > 0` makes the lowest segment at least two long, so every
        // group has its k + 1 neighbour adjacent and no tail is left.
        for_each_2q_base(n_iter, lo, hi, 2, |base| {
            let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
            apply_fused_2q_pair_avx2_inner(state, i, mat256);
        });
    }
}

/// Apply four consecutive groups (k to k+3) of a 2q gate using AVX-512.
///
/// Precondition: the lower gate qubit `lo` is at least 2 and `i` is group k's row
/// indices, so one 512-bit load reads row r of all four groups. Every lane runs the AVX2
/// pair kernel's two chains; `fmaddsub` by one stands in for `addsub`, which AVX-512
/// lacks, and rounds identically.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn apply_fused_2q_quad_avx512_inner(
    state: *mut f64,
    i: [usize; 4],
    mat: &Mat4x4Broadcast256,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let one = _mm512_set1_pd(1.0);
        let s0 = _mm512_loadu_pd(state.add(i[0] * 2));
        let s1 = _mm512_loadu_pd(state.add(i[1] * 2));
        let s2 = _mm512_loadu_pd(state.add(i[2] * 2));
        let s3 = _mm512_loadu_pd(state.add(i[3] * 2));

        let sf0 = _mm512_permute_pd::<0b0101_0101>(s0);
        let sf1 = _mm512_permute_pd::<0b0101_0101>(s1);
        let sf2 = _mm512_permute_pd::<0b0101_0101>(s2);
        let sf3 = _mm512_permute_pd::<0b0101_0101>(s3);

        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut re = _mm512_mul_pd(_mm512_broadcast_f64x4(mat.rr[off]), s0);
                let mut im = _mm512_mul_pd(_mm512_broadcast_f64x4(mat.ii[off]), sf0);
                re = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.rr[off + 1]), s1, re);
                im = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.ii[off + 1]), sf1, im);
                re = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.rr[off + 2]), s2, re);
                im = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.ii[off + 2]), sf2, im);
                re = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.rr[off + 3]), s3, re);
                im = _mm512_fmadd_pd(_mm512_broadcast_f64x4(mat.ii[off + 3]), sf3, im);
                _mm512_storeu_pd(state.add(i[$r] * 2), _mm512_fmaddsub_pd(one, re, im));
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
}

/// Quad-batched 2q kernel main loop. Requires `lo >= 2` so four consecutive groups map
/// to adjacent state positions; hands `lo < 2` to the AVX2 pair loop.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn apply_fused_2q_loop_avx512(
    state: *mut f64,
    n_iter: usize,
    lo: usize,
    hi: usize,
    mask0: usize,
    mask1: usize,
    mat256: &Mat4x4Broadcast256,
    mat128: &Mat4x4Broadcast,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if lo < 2 {
            apply_fused_2q_loop_avx2(state, n_iter, lo, hi, mask0, mask1, mat256, mat128);
            return;
        }
        // `lo >= 2` makes the lowest segment a multiple of four long, so every quad
        // stays inside it and no tail is left.
        for_each_2q_base(n_iter, lo, hi, 4, |base| {
            let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
            apply_fused_2q_quad_avx512_inner(state, i, mat256);
        });
    }
}

/// Apply one 4-element group of a 2q gate using SSE2 intrinsics.
///
/// Uses the 5-op complex multiply (shuffle + 2×mul + xor + add) since SSE2
/// is always available on x86_64. Safe to call from Rayon closures without
/// `#[target_feature]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_fused_2q_group_sse2(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let sign_mask = _mm_set_pd(0.0, -0.0_f64);

        let s0 = _mm_loadu_pd(state.add(i[0] * 2));
        let s1 = _mm_loadu_pd(state.add(i[1] * 2));
        let s2 = _mm_loadu_pd(state.add(i[2] * 2));
        let s3 = _mm_loadu_pd(state.add(i[3] * 2));

        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut acc = complex_mul_sse2(mat.rr[off], mat.ii[off], s0, sign_mask);
                acc = _mm_add_pd(
                    acc,
                    complex_mul_sse2(mat.rr[off + 1], mat.ii[off + 1], s1, sign_mask),
                );
                acc = _mm_add_pd(
                    acc,
                    complex_mul_sse2(mat.rr[off + 2], mat.ii[off + 2], s2, sign_mask),
                );
                acc = _mm_add_pd(
                    acc,
                    complex_mul_sse2(mat.rr[off + 3], mat.ii[off + 3], s3, sign_mask),
                );
                _mm_storeu_pd(state.add(i[$r] * 2), acc);
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
}

#[cfg(target_arch = "aarch64")]
struct Mat4x4Broadcast {
    rr: [float64x2_t; 16],
    ii_as: [float64x2_t; 16],
}

#[cfg(target_arch = "aarch64")]
impl Mat4x4Broadcast {
    #[inline(always)]
    fn from_matrix(mat: &[[Complex64; 4]; 4]) -> Self {
        // SAFETY: NEON is baseline on aarch64, and these
        // intrinsics only broadcast scalar f64 values into SIMD registers.
        unsafe {
            let mut rr = [vdupq_n_f64(0.0); 16];
            let mut ii_as = [vdupq_n_f64(0.0); 16];
            for (r, row) in mat.iter().enumerate() {
                for (c, elem) in row.iter().enumerate() {
                    let idx = r * 4 + c;
                    rr[idx] = vdupq_n_f64(elem.re);
                    ii_as[idx] = vcombine_f64(vdup_n_f64(-elem.im), vdup_n_f64(elem.im));
                }
            }
            Self { rr, ii_as }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_fused_2q_group_neon(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s0 = vld1q_f64(state.add(i[0] * 2));
        let s1 = vld1q_f64(state.add(i[1] * 2));
        let s2 = vld1q_f64(state.add(i[2] * 2));
        let s3 = vld1q_f64(state.add(i[3] * 2));

        let sf0 = vextq_f64(s0, s0, 1);
        let sf1 = vextq_f64(s1, s1, 1);
        let sf2 = vextq_f64(s2, s2, 1);
        let sf3 = vextq_f64(s3, s3, 1);

        // Two chains per row, `rr * z` and `ii_as * swap(z)` (the two halves
        // of each complex product, the imaginary half pre-signed), joined by
        // one add: nine ops at a dependency depth of five, where one chain
        // of eight left the x86 kernels latency-bound at twice their port
        // cost.
        macro_rules! row {
            ($r:expr) => {{
                let off = $r * 4;
                let mut re = vmulq_f64(mat.rr[off], s0);
                let mut im = vmulq_f64(mat.ii_as[off], sf0);
                re = vfmaq_f64(re, mat.rr[off + 1], s1);
                im = vfmaq_f64(im, mat.ii_as[off + 1], sf1);
                re = vfmaq_f64(re, mat.rr[off + 2], s2);
                im = vfmaq_f64(im, mat.ii_as[off + 2], sf2);
                re = vfmaq_f64(re, mat.rr[off + 3], s3);
                im = vfmaq_f64(im, mat.ii_as[off + 3], sf3);
                vst1q_f64(state.add(i[$r] * 2), vaddq_f64(re, im));
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
    }
}

/// Two groups at once: the group at `i` and the one at `i + 1`, adjacent
/// amplitudes when the lower target is above bit 0. Each matrix constant is
/// loaded once and feeds both groups, so the pair costs 32 constant loads
/// where two group calls cost 64, against NEON's 32 vector registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_fused_2q_pair_neon(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let p0 = state.add(i[0] * 2);
        let p1 = state.add(i[1] * 2);
        let p2 = state.add(i[2] * 2);
        let p3 = state.add(i[3] * 2);
        let a0 = vld1q_f64(p0);
        let b0 = vld1q_f64(p0.add(2));
        let a1 = vld1q_f64(p1);
        let b1 = vld1q_f64(p1.add(2));
        let a2 = vld1q_f64(p2);
        let b2 = vld1q_f64(p2.add(2));
        let a3 = vld1q_f64(p3);
        let b3 = vld1q_f64(p3.add(2));

        let af0 = vextq_f64(a0, a0, 1);
        let bf0 = vextq_f64(b0, b0, 1);
        let af1 = vextq_f64(a1, a1, 1);
        let bf1 = vextq_f64(b1, b1, 1);
        let af2 = vextq_f64(a2, a2, 1);
        let bf2 = vextq_f64(b2, b2, 1);
        let af3 = vextq_f64(a3, a3, 1);
        let bf3 = vextq_f64(b3, b3, 1);

        // Two chains per row and per group, as in the group kernel.
        macro_rules! row {
            ($r:expr, $p:expr) => {{
                let off = $r * 4;
                let c = mat.rr[off];
                let mut re_a = vmulq_f64(c, a0);
                let mut re_b = vmulq_f64(c, b0);
                let c = mat.ii_as[off];
                let mut im_a = vmulq_f64(c, af0);
                let mut im_b = vmulq_f64(c, bf0);
                let c = mat.rr[off + 1];
                re_a = vfmaq_f64(re_a, c, a1);
                re_b = vfmaq_f64(re_b, c, b1);
                let c = mat.ii_as[off + 1];
                im_a = vfmaq_f64(im_a, c, af1);
                im_b = vfmaq_f64(im_b, c, bf1);
                let c = mat.rr[off + 2];
                re_a = vfmaq_f64(re_a, c, a2);
                re_b = vfmaq_f64(re_b, c, b2);
                let c = mat.ii_as[off + 2];
                im_a = vfmaq_f64(im_a, c, af2);
                im_b = vfmaq_f64(im_b, c, bf2);
                let c = mat.rr[off + 3];
                re_a = vfmaq_f64(re_a, c, a3);
                re_b = vfmaq_f64(re_b, c, b3);
                let c = mat.ii_as[off + 3];
                im_a = vfmaq_f64(im_a, c, af3);
                im_b = vfmaq_f64(im_b, c, bf3);
                vst1q_f64($p, vaddq_f64(re_a, im_a));
                vst1q_f64($p.add(2), vaddq_f64(re_b, im_b));
            }};
        }
        row!(0, p0);
        row!(1, p1);
        row!(2, p2);
        row!(3, p3);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_fused_2q_loop_neon(
    state: *mut f64,
    n_iter: usize,
    lo: usize,
    hi: usize,
    mask0: usize,
    mask1: usize,
    mat: &Mat4x4Broadcast,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if lo == 0 {
            for_each_2q_base(n_iter, lo, hi, 1, |base| {
                let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
                apply_fused_2q_group_neon(state, i, mat);
            });
            return;
        }
        for_each_2q_base(n_iter, lo, hi, 2, |base| {
            let i = [base, base + mask1, base + mask0, base + mask0 + mask1];
            apply_fused_2q_pair_neon(state, i, mat);
        });
    }
}

/// Two-qubit gate prepared once and applied per 4-element group: the 4×4 matrix
/// in broadcast form on x86_64 and aarch64, raw elsewhere.
pub(crate) struct PreparedGate2q {
    #[cfg(target_arch = "x86_64")]
    broadcast: Mat4x4Broadcast,
    #[cfg(target_arch = "x86_64")]
    broadcast256: Option<Mat4x4Broadcast256>,
    #[cfg(target_arch = "x86_64")]
    pub(super) tier: SimdTier,
    #[cfg(target_arch = "aarch64")]
    broadcast: Mat4x4Broadcast,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    mat: [[Complex64; 4]; 4],
}

impl PreparedGate2q {
    #[inline(always)]
    pub(crate) fn new(mat: &[[Complex64; 4]; 4]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let broadcast = Mat4x4Broadcast::from_matrix(mat);
            let tier = simd_tier();
            let broadcast256 = if matches!(tier, SimdTier::Avx512 | SimdTier::Avx2Fma) {
                // SAFETY: AVX support is implied by the detected AVX2 feature.
                Some(unsafe { Mat4x4Broadcast256::from_matrix(mat) })
            } else {
                None
            };
            Self {
                broadcast,
                broadcast256,
                tier,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                broadcast: Mat4x4Broadcast::from_matrix(mat),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { mat: *mat }
        }
    }

    /// Apply the full 2q gate to the statevector sequentially.
    ///
    /// Uses 128-bit FMA on x86_64 even when AVX2 is available; the AVX2
    /// paired-group kernel is reserved for [`apply_tiled`] where the slice
    /// is known to be cache-resident.
    pub(crate) fn apply_full(
        &self,
        state: &mut [Complex64],
        num_qubits: usize,
        q0: usize,
        q1: usize,
    ) {
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let n_iter = 1usize << (num_qubits - 2);

        #[cfg(target_arch = "x86_64")]
        {
            let base = state.as_mut_ptr() as *mut f64;
            if !matches!(self.tier, SimdTier::Sse2) {
                // SAFETY: FMA support is guaranteed by the selected tier. The
                // loop builds in-bounds, disjoint 4-amplitude groups.
                unsafe {
                    apply_fused_2q_loop_fma(base, n_iter, lo, hi, mask0, mask1, &self.broadcast);
                }
                return;
            }
            // SAFETY: x86_64 guarantees SSE2, and the loop builds in-bounds,
            // disjoint 4-amplitude groups.
            unsafe {
                use crate::backend::insert_zero_bit;
                for k in 0..n_iter {
                    let idx = insert_zero_bit(insert_zero_bit(k, lo), hi);
                    let i = [idx, idx | mask1, idx | mask0, idx | mask0 | mask1];
                    apply_fused_2q_group_sse2(base, i, &self.broadcast);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            let base = state.as_mut_ptr() as *mut f64;
            // SAFETY: NEON is baseline on aarch64. The
            // loop builds in-bounds, disjoint 4-amplitude groups.
            unsafe {
                apply_fused_2q_loop_neon(base, n_iter, lo, hi, mask0, mask1, &self.broadcast);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            use crate::backend::insert_zero_bit;
            for k in 0..n_iter {
                let idx = insert_zero_bit(insert_zero_bit(k, lo), hi);
                let i = [idx, idx | mask1, idx | mask0, idx | mask0 | mask1];
                let a = [state[i[0]], state[i[1]], state[i[2]], state[i[3]]];
                for (r, &idx) in i.iter().enumerate() {
                    state[idx] = self.mat[r][0] * a[0]
                        + self.mat[r][1] * a[1]
                        + self.mat[r][2] * a[2]
                        + self.mat[r][3] * a[3];
                }
            }
        }
    }

    /// Apply the 2q gate to a slice that is known to be cache-resident.
    ///
    /// Prefers the AVX-512 / AVX2 paired-group kernels when available, falling back
    /// to 128-bit FMA only on 2-qubit slices (`n_iter < 2`). Use from `Multi2q`
    /// tile loops where the slice fits in L2/L3 and the kernel is
    /// compute-bound; the AVX2 throttle is justified by ~2× FMA throughput.
    pub(crate) fn apply_tiled(
        &self,
        state: &mut [Complex64],
        num_qubits: usize,
        q0: usize,
        q1: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            let mask0 = 1usize << q0;
            let mask1 = 1usize << q1;
            let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
            let n_iter = 1usize << (num_qubits - 2);
            let base = state.as_mut_ptr() as *mut f64;
            // SAFETY: The selected tier is guarded by runtime feature checks
            // in the constructor. The loop builds disjoint 4-amplitude groups.
            unsafe {
                match self.tier {
                    SimdTier::Avx512 if avx2_2q_enabled() => {
                        // SAFETY: broadcast256 is Some whenever tier is Avx512 (constructor invariant).
                        let mat256 = self.broadcast256.as_ref().unwrap_unchecked();
                        apply_fused_2q_loop_avx512(
                            base,
                            n_iter,
                            lo,
                            hi,
                            mask0,
                            mask1,
                            mat256,
                            &self.broadcast,
                        );
                    }
                    SimdTier::Avx2Fma if avx2_2q_enabled() => {
                        // SAFETY: broadcast256 is Some whenever tier is Avx2Fma (constructor invariant).
                        let mat256 = self.broadcast256.as_ref().unwrap_unchecked();
                        apply_fused_2q_loop_avx2(
                            base,
                            n_iter,
                            lo,
                            hi,
                            mask0,
                            mask1,
                            mat256,
                            &self.broadcast,
                        );
                    }
                    SimdTier::Avx512 | SimdTier::Avx2Fma | SimdTier::Fma => {
                        apply_fused_2q_loop_fma(
                            base,
                            n_iter,
                            lo,
                            hi,
                            mask0,
                            mask1,
                            &self.broadcast,
                        );
                    }
                    SimdTier::Sse2 => {
                        use crate::backend::insert_zero_bit;
                        for k in 0..n_iter {
                            let idx = insert_zero_bit(insert_zero_bit(k, lo), hi);
                            let i = [idx, idx | mask1, idx | mask0, idx | mask0 | mask1];
                            apply_fused_2q_group_sse2(base, i, &self.broadcast);
                        }
                    }
                }
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            self.apply_full(state, num_qubits, q0, q1);
        }
    }

    /// Apply the gate to `n_iter` groups from group `first`, counted in the
    /// order [`Self::apply_full`] visits them. `n_iter` is a power of two that
    /// divides `first`, so the range is one aligned run of the pair loop.
    ///
    /// # Safety
    /// `state` must cover every group the range names, and no other thread may
    /// touch those groups during the call.
    #[cfg(feature = "parallel")]
    pub(crate) unsafe fn apply_groups_ptr(
        &self,
        state: *mut f64,
        first: usize,
        n_iter: usize,
        q0: usize,
        q1: usize,
    ) {
        use crate::backend::insert_zero_bit;
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        // An aligned run of 2^bits groups covers whole segments of the nested
        // walk, so narrowing `lo` and `hi` to the run walks exactly that run.
        let bits = n_iter.trailing_zeros() as usize;
        let run_lo = lo.min(bits);
        let run_hi = hi.min(bits + 1);
        let start = insert_zero_bit(insert_zero_bit(first, lo), hi);

        // SAFETY: same contract as the enclosing unsafe fn. The selected tier
        // is guarded by runtime feature checks in the constructor.
        unsafe {
            let base = state.add(start * 2);
            #[cfg(target_arch = "x86_64")]
            match self.tier {
                SimdTier::Avx512 if avx2_2q_enabled() => {
                    // SAFETY: broadcast256 is Some whenever tier is Avx512 (constructor invariant).
                    let mat256 = self.broadcast256.as_ref().unwrap_unchecked();
                    apply_fused_2q_loop_avx512(
                        base,
                        n_iter,
                        run_lo,
                        run_hi,
                        mask0,
                        mask1,
                        mat256,
                        &self.broadcast,
                    );
                }
                SimdTier::Avx2Fma if avx2_2q_enabled() => {
                    // SAFETY: broadcast256 is Some whenever tier is Avx2Fma (constructor invariant).
                    let mat256 = self.broadcast256.as_ref().unwrap_unchecked();
                    apply_fused_2q_loop_avx2(
                        base,
                        n_iter,
                        run_lo,
                        run_hi,
                        mask0,
                        mask1,
                        mat256,
                        &self.broadcast,
                    );
                }
                SimdTier::Avx512 | SimdTier::Avx2Fma | SimdTier::Fma => {
                    apply_fused_2q_loop_fma(
                        base,
                        n_iter,
                        run_lo,
                        run_hi,
                        mask0,
                        mask1,
                        &self.broadcast,
                    );
                }
                SimdTier::Sse2 => {
                    for_each_2q_base(n_iter, run_lo, run_hi, 1, |b| {
                        let i = [b, b + mask1, b + mask0, b + mask0 + mask1];
                        apply_fused_2q_group_sse2(base, i, &self.broadcast);
                    });
                }
            }
            #[cfg(target_arch = "aarch64")]
            apply_fused_2q_loop_neon(base, n_iter, run_lo, run_hi, mask0, mask1, &self.broadcast);
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            for_each_2q_base(n_iter, run_lo, run_hi, 1, |b| {
                let i = [b, b + mask1, b + mask0, b + mask0 + mask1];
                let a: [Complex64; 4] = [
                    *(base.add(i[0] * 2) as *const Complex64),
                    *(base.add(i[1] * 2) as *const Complex64),
                    *(base.add(i[2] * 2) as *const Complex64),
                    *(base.add(i[3] * 2) as *const Complex64),
                ];
                for (r, &idx) in i.iter().enumerate() {
                    let result = self.mat[r][0] * a[0]
                        + self.mat[r][1] * a[1]
                        + self.mat[r][2] * a[2]
                        + self.mat[r][3] * a[3];
                    *(base.add(idx * 2) as *mut Complex64) = result;
                }
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
// SAFETY: Mat4x4Broadcast contains only SIMD register values (pure data, no pointers).
unsafe impl Send for Mat4x4Broadcast {}
#[cfg(target_arch = "x86_64")]
// SAFETY: Mat4x4Broadcast contains only SIMD register values (pure data, no pointers).
unsafe impl Sync for Mat4x4Broadcast {}
#[cfg(target_arch = "aarch64")]
// SAFETY: Mat4x4Broadcast on aarch64 contains only float64x2_t values (pure SIMD data).
unsafe impl Send for Mat4x4Broadcast {}
#[cfg(target_arch = "aarch64")]
// SAFETY: Mat4x4Broadcast on aarch64 contains only float64x2_t values (pure SIMD data).
unsafe impl Sync for Mat4x4Broadcast {}
// SAFETY: PreparedGate2q contains immutable SIMD broadcast data and no raw pointers.
unsafe impl Send for PreparedGate2q {}
// SAFETY: PreparedGate2q contains immutable SIMD broadcast data and no raw pointers.
unsafe impl Sync for PreparedGate2q {}
