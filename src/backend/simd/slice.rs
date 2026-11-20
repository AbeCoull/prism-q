//! Bulk amplitude slice primitives: negate, swap, zero, scale, norm-squared sums and the
//! distributed global half combine.

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use super::complex_mul_neon;
#[cfg(target_arch = "x86_64")]
use super::{SimdTier, complex_mul_avx512, for_each_quad, load_pd, simd_tier, store_pd};
#[cfg(all(target_arch = "x86_64", feature = "distributed"))]
use super::{complex_mul_avx2fma, complex_mul_fma};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn negate_slice_avx2(slice: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let sign = _mm256_set1_pd(-0.0_f64);
        let ptr = slice.as_mut_ptr() as *mut f64;
        let pairs = slice.len() / 2;
        for i in 0..pairs {
            let off = i * 4;
            let v = _mm256_loadu_pd(ptr.add(off));
            _mm256_storeu_pd(ptr.add(off), _mm256_xor_pd(v, sign));
        }
        if !slice.len().is_multiple_of(2) {
            let last = &mut slice[slice.len() - 1];
            *last = -*last;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn negate_slice_avx512(slice: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let sign = _mm512_set1_epi64(i64::MIN);
        let ptr = slice.as_mut_ptr() as *mut f64;
        for_each_quad!(slice.len(), |off, k| {
            let v = _mm512_castpd_si512(load_pd(ptr.add(off), k));
            store_pd(
                ptr.add(off),
                k,
                _mm512_castsi512_pd(_mm512_xor_si512(v, sign)),
            );
        });
    }
}

const MIN_SIMD_SLICE: usize = 4;

#[inline(always)]
pub(crate) fn negate_slice(slice: &mut [Complex64]) {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            SimdTier::Avx512 => {
                // SAFETY: AVX-512F detected, and the slice bounds drive all pointer
                // arithmetic inside the helper.
                unsafe { negate_slice_avx512(slice) };
                return;
            }
            SimdTier::Avx2Fma => {
                // SAFETY: AVX2 checked above, and the slice bounds drive
                // all pointer arithmetic inside the helper.
                unsafe { negate_slice_avx2(slice) };
                return;
            }
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64, and the loop
        // only touches elements inside the mutable slice.
        unsafe { negate_slice_neon(slice) };
        return;
    }
    for amp in slice.iter_mut() {
        *amp = -*amp;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn negate_slice_neon(slice: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_mut_ptr() as *mut f64;
        for i in 0..slice.len() {
            let p = ptr.add(i * 2);
            let v = vld1q_f64(p);
            vst1q_f64(p, vnegq_f64(v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn swap_slices_avx2(a: &mut [Complex64], b: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let ap = a.as_mut_ptr() as *mut f64;
        let bp = b.as_mut_ptr() as *mut f64;
        let pairs = n / 2;
        for i in 0..pairs {
            let off = i * 4;
            let va = _mm256_loadu_pd(ap.add(off));
            let vb = _mm256_loadu_pd(bp.add(off));
            _mm256_storeu_pd(ap.add(off), vb);
            _mm256_storeu_pd(bp.add(off), va);
        }
        if !n.is_multiple_of(2) {
            let last = n - 1;
            std::mem::swap(&mut a[last], &mut b[last]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn swap_slices_avx512(a: &mut [Complex64], b: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let ap = a.as_mut_ptr() as *mut f64;
        let bp = b.as_mut_ptr() as *mut f64;
        for_each_quad!(a.len(), |off, k| {
            let va = load_pd(ap.add(off), k);
            let vb = load_pd(bp.add(off), k);
            store_pd(ap.add(off), k, vb);
            store_pd(bp.add(off), k, va);
        });
    }
}

#[inline(always)]
pub(crate) fn swap_slices(a: &mut [Complex64], b: &mut [Complex64]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if a.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            SimdTier::Avx512 => {
                // SAFETY: AVX-512F detected, slices have equal length, and Rust's two
                // mutable references guarantee non-overlap.
                unsafe { swap_slices_avx512(a, b) };
                return;
            }
            SimdTier::Avx2Fma => {
                // SAFETY: AVX2 checked above, slices have equal length,
                // and Rust's two mutable references guarantee non-overlap.
                unsafe { swap_slices_avx2(a, b) };
                return;
            }
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64, slices have
        // equal length, and Rust's two mutable references guarantee non-overlap.
        unsafe { swap_slices_neon(a, b) };
        return;
    }
    for (x, y) in a.iter_mut().zip(b.iter_mut()) {
        std::mem::swap(x, y);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn swap_slices_neon(a: &mut [Complex64], b: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ap = a.as_mut_ptr() as *mut f64;
        let bp = b.as_mut_ptr() as *mut f64;
        for i in 0..a.len() {
            let off = i * 2;
            let va = vld1q_f64(ap.add(off));
            let vb = vld1q_f64(bp.add(off));
            vst1q_f64(ap.add(off), vb);
            vst1q_f64(bp.add(off), va);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn norm_sqr_sum_avx2fma(slice: &[Complex64]) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_ptr() as *const f64;
        let pairs = slice.len() / 2;

        // 4 independent accumulators hide FMA latency (4 cycles on Skylake).
        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();
        let unrolled = pairs / 4;
        let remainder = pairs % 4;

        for i in 0..unrolled {
            let base = i * 16;
            let v0 = _mm256_loadu_pd(ptr.add(base));
            let v1 = _mm256_loadu_pd(ptr.add(base + 4));
            let v2 = _mm256_loadu_pd(ptr.add(base + 8));
            let v3 = _mm256_loadu_pd(ptr.add(base + 12));
            a0 = _mm256_fmadd_pd(v0, v0, a0);
            a1 = _mm256_fmadd_pd(v1, v1, a1);
            a2 = _mm256_fmadd_pd(v2, v2, a2);
            a3 = _mm256_fmadd_pd(v3, v3, a3);
        }

        let mut acc = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
        let tail_base = unrolled * 16;
        for i in 0..remainder {
            let v = _mm256_loadu_pd(ptr.add(tail_base + i * 4));
            acc = _mm256_fmadd_pd(v, v, acc);
        }

        let hi128 = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(_mm256_castpd256_pd128(acc), hi128);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let total = _mm_add_sd(sum128, hi64);
        let mut result = _mm_cvtsd_f64(total);
        if !slice.len().is_multiple_of(2) {
            result += slice[slice.len() - 1].norm_sqr();
        }
        result
    }
}

/// Four accumulators over sixteen amplitudes per step, as the AVX2 kernel runs them,
/// then the remaining steps into one accumulator with the last under a lane mask.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn norm_sqr_sum_avx512(slice: &[Complex64]) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_ptr() as *const f64;
        let unrolled = slice.len() / 16;

        let mut a0 = _mm512_setzero_pd();
        let mut a1 = _mm512_setzero_pd();
        let mut a2 = _mm512_setzero_pd();
        let mut a3 = _mm512_setzero_pd();
        for i in 0..unrolled {
            let base = i * 32;
            let v0 = _mm512_loadu_pd(ptr.add(base));
            let v1 = _mm512_loadu_pd(ptr.add(base + 8));
            let v2 = _mm512_loadu_pd(ptr.add(base + 16));
            let v3 = _mm512_loadu_pd(ptr.add(base + 24));
            a0 = _mm512_fmadd_pd(v0, v0, a0);
            a1 = _mm512_fmadd_pd(v1, v1, a1);
            a2 = _mm512_fmadd_pd(v2, v2, a2);
            a3 = _mm512_fmadd_pd(v3, v3, a3);
        }

        let mut acc = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
        let tail = ptr.add(unrolled * 32);
        for_each_quad!(slice.len() - unrolled * 16, |off, k| {
            let v = load_pd(tail.add(off), k);
            acc = _mm512_fmadd_pd(v, v, acc);
        });
        _mm512_reduce_add_pd(acc)
    }
}

pub(crate) fn norm_sqr_sum(slice: &[Complex64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            // SAFETY: AVX-512F detected, and the helper reads only within the shared
            // slice.
            SimdTier::Avx512 => return unsafe { norm_sqr_sum_avx512(slice) },
            // SAFETY: AVX2 and FMA checked above, and the helper reads
            // only within the shared slice.
            SimdTier::Avx2Fma => return unsafe { norm_sqr_sum_avx2fma(slice) },
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64, and the
        // helper reads only within the shared slice.
        return unsafe { norm_sqr_sum_neon(slice) };
    }
    slice.iter().map(|c| c.norm_sqr()).sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn norm_sqr_sum_neon(slice: &[Complex64]) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_ptr() as *const f64;
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let unrolled = slice.len() / 4;
        let remainder = slice.len() % 4;
        for i in 0..unrolled {
            let base = i * 8;
            let v0 = vld1q_f64(ptr.add(base));
            let v1 = vld1q_f64(ptr.add(base + 2));
            let v2 = vld1q_f64(ptr.add(base + 4));
            let v3 = vld1q_f64(ptr.add(base + 6));
            a0 = vfmaq_f64(a0, v0, v0);
            a1 = vfmaq_f64(a1, v1, v1);
            a2 = vfmaq_f64(a2, v2, v2);
            a3 = vfmaq_f64(a3, v3, v3);
        }
        let mut acc = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        let tail = unrolled * 4;
        for i in 0..remainder {
            let v = vld1q_f64(ptr.add((tail + i) * 2));
            acc = vfmaq_f64(acc, v, v);
        }
        vaddvq_f64(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn norm_sqr_sum_scale_avx2fma(slice: &mut [Complex64], factor: f64) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_mut_ptr() as *mut f64;
        let f = _mm256_set1_pd(factor);
        let pairs = slice.len() / 2;

        let mut a0 = _mm256_setzero_pd();
        let mut a1 = _mm256_setzero_pd();
        let mut a2 = _mm256_setzero_pd();
        let mut a3 = _mm256_setzero_pd();
        let unrolled = pairs / 4;
        let remainder = pairs % 4;

        for i in 0..unrolled {
            let base = i * 16;
            let v0 = _mm256_loadu_pd(ptr.add(base));
            let v1 = _mm256_loadu_pd(ptr.add(base + 4));
            let v2 = _mm256_loadu_pd(ptr.add(base + 8));
            let v3 = _mm256_loadu_pd(ptr.add(base + 12));
            a0 = _mm256_fmadd_pd(v0, v0, a0);
            a1 = _mm256_fmadd_pd(v1, v1, a1);
            a2 = _mm256_fmadd_pd(v2, v2, a2);
            a3 = _mm256_fmadd_pd(v3, v3, a3);
            _mm256_storeu_pd(ptr.add(base), _mm256_mul_pd(v0, f));
            _mm256_storeu_pd(ptr.add(base + 4), _mm256_mul_pd(v1, f));
            _mm256_storeu_pd(ptr.add(base + 8), _mm256_mul_pd(v2, f));
            _mm256_storeu_pd(ptr.add(base + 12), _mm256_mul_pd(v3, f));
        }

        let mut acc = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
        let tail_base = unrolled * 16;
        for i in 0..remainder {
            let p = ptr.add(tail_base + i * 4);
            let v = _mm256_loadu_pd(p);
            acc = _mm256_fmadd_pd(v, v, acc);
            _mm256_storeu_pd(p, _mm256_mul_pd(v, f));
        }

        let hi128 = _mm256_extractf128_pd(acc, 1);
        let sum128 = _mm_add_pd(_mm256_castpd256_pd128(acc), hi128);
        let hi64 = _mm_unpackhi_pd(sum128, sum128);
        let total = _mm_add_sd(sum128, hi64);
        let mut result = _mm_cvtsd_f64(total);
        if !slice.len().is_multiple_of(2) {
            let last = &mut slice[slice.len() - 1];
            result += last.norm_sqr();
            *last *= factor;
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn norm_sqr_sum_scale_avx512(slice: &mut [Complex64], factor: f64) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_mut_ptr() as *mut f64;
        let f = _mm512_set1_pd(factor);
        let unrolled = slice.len() / 16;

        let mut a0 = _mm512_setzero_pd();
        let mut a1 = _mm512_setzero_pd();
        let mut a2 = _mm512_setzero_pd();
        let mut a3 = _mm512_setzero_pd();
        for i in 0..unrolled {
            let base = i * 32;
            let v0 = _mm512_loadu_pd(ptr.add(base));
            let v1 = _mm512_loadu_pd(ptr.add(base + 8));
            let v2 = _mm512_loadu_pd(ptr.add(base + 16));
            let v3 = _mm512_loadu_pd(ptr.add(base + 24));
            a0 = _mm512_fmadd_pd(v0, v0, a0);
            a1 = _mm512_fmadd_pd(v1, v1, a1);
            a2 = _mm512_fmadd_pd(v2, v2, a2);
            a3 = _mm512_fmadd_pd(v3, v3, a3);
            _mm512_storeu_pd(ptr.add(base), _mm512_mul_pd(v0, f));
            _mm512_storeu_pd(ptr.add(base + 8), _mm512_mul_pd(v1, f));
            _mm512_storeu_pd(ptr.add(base + 16), _mm512_mul_pd(v2, f));
            _mm512_storeu_pd(ptr.add(base + 24), _mm512_mul_pd(v3, f));
        }

        let mut acc = _mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3));
        let tail = ptr.add(unrolled * 32);
        for_each_quad!(slice.len() - unrolled * 16, |off, k| {
            let v = load_pd(tail.add(off), k);
            acc = _mm512_fmadd_pd(v, v, acc);
            store_pd(tail.add(off), k, _mm512_mul_pd(v, f));
        });
        _mm512_reduce_add_pd(acc)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn norm_sqr_sum_scale_neon(slice: &mut [Complex64], factor: f64) -> f64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let ptr = slice.as_mut_ptr() as *mut f64;
        let f = vdupq_n_f64(factor);
        let mut a0 = vdupq_n_f64(0.0);
        let mut a1 = vdupq_n_f64(0.0);
        let mut a2 = vdupq_n_f64(0.0);
        let mut a3 = vdupq_n_f64(0.0);
        let unrolled = slice.len() / 4;
        let remainder = slice.len() % 4;
        for i in 0..unrolled {
            let base = i * 8;
            let v0 = vld1q_f64(ptr.add(base));
            let v1 = vld1q_f64(ptr.add(base + 2));
            let v2 = vld1q_f64(ptr.add(base + 4));
            let v3 = vld1q_f64(ptr.add(base + 6));
            a0 = vfmaq_f64(a0, v0, v0);
            a1 = vfmaq_f64(a1, v1, v1);
            a2 = vfmaq_f64(a2, v2, v2);
            a3 = vfmaq_f64(a3, v3, v3);
            vst1q_f64(ptr.add(base), vmulq_f64(v0, f));
            vst1q_f64(ptr.add(base + 2), vmulq_f64(v1, f));
            vst1q_f64(ptr.add(base + 4), vmulq_f64(v2, f));
            vst1q_f64(ptr.add(base + 6), vmulq_f64(v3, f));
        }
        let mut acc = vaddq_f64(vaddq_f64(a0, a1), vaddq_f64(a2, a3));
        let tail = unrolled * 4;
        for i in 0..remainder {
            let p = ptr.add((tail + i) * 2);
            let v = vld1q_f64(p);
            acc = vfmaq_f64(acc, v, v);
            vst1q_f64(p, vmulq_f64(v, f));
        }
        vaddvq_f64(acc)
    }
}

/// [`norm_sqr_sum`] that scales each amplitude by `factor` once read: one pass, and
/// the same sum bit for bit, since every tier accumulates in `norm_sqr_sum`'s order.
pub(crate) fn norm_sqr_sum_scale(slice: &mut [Complex64], factor: f64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            // SAFETY: AVX-512F detected, and the helper reads and writes only within the
            // mutable slice.
            SimdTier::Avx512 => return unsafe { norm_sqr_sum_scale_avx512(slice, factor) },
            // SAFETY: AVX2 detected, and the helper reads and writes only within the
            // mutable slice.
            SimdTier::Avx2Fma => return unsafe { norm_sqr_sum_scale_avx2fma(slice, factor) },
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64, and the helper reads and writes only
        // within the mutable slice.
        return unsafe { norm_sqr_sum_scale_neon(slice, factor) };
    }
    let mut sum = 0.0;
    for amp in slice.iter_mut() {
        sum += amp.norm_sqr();
        *amp *= factor;
    }
    sum
}

/// Four norms per step from two vectors: `hadd` pairs the squared lanes and a permute puts
/// them back in amplitude order. `SCALED` multiplies by `scale`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn norm_sqr_to_slice_avx2<const SCALED: bool>(
    src: &[Complex64],
    dst: &mut [f64],
    scale: f64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(dst.len() >= src.len());
        let inp = src.as_ptr() as *const f64;
        let out = dst.as_mut_ptr();
        let s = _mm256_set1_pd(scale);
        let quads = src.len() / 4;

        for i in 0..quads {
            let base_in = i * 8;
            let base_out = i * 4;
            let v0 = _mm256_loadu_pd(inp.add(base_in));
            let v1 = _mm256_loadu_pd(inp.add(base_in + 4));
            let sq0 = _mm256_mul_pd(v0, v0);
            let sq1 = _mm256_mul_pd(v1, v1);
            let h = _mm256_hadd_pd(sq0, sq1);
            let ordered = _mm256_permute4x64_pd(h, 0b11_01_10_00);
            _mm256_storeu_pd(
                out.add(base_out),
                if SCALED {
                    _mm256_mul_pd(ordered, s)
                } else {
                    ordered
                },
            );
        }

        let tail = quads * 4;
        for (j, c) in src[tail..].iter().enumerate() {
            *out.add(tail + j) = if SCALED {
                c.norm_sqr() * scale
            } else {
                c.norm_sqr()
            };
        }
    }
}

/// Eight norms per step: the squared lanes of two vectors split into real and imaginary
/// halves by two `vpermt2pd` and added, in amplitude order. `SCALED` multiplies by
/// `scale`. The remainder runs scalar, as in the AVX2 kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn norm_sqr_to_slice_avx512<const SCALED: bool>(
    src: &[Complex64],
    dst: &mut [f64],
    scale: f64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(dst.len() >= src.len());
        let inp = src.as_ptr() as *const f64;
        let out = dst.as_mut_ptr();
        let s = _mm512_set1_pd(scale);
        let idx_re = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        let idx_im = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        let octs = src.len() / 8;

        for i in 0..octs {
            let v0 = _mm512_loadu_pd(inp.add(i * 16));
            let v1 = _mm512_loadu_pd(inp.add(i * 16 + 8));
            let sq0 = _mm512_mul_pd(v0, v0);
            let sq1 = _mm512_mul_pd(v1, v1);
            let norms = _mm512_add_pd(
                _mm512_permutex2var_pd(sq0, idx_re, sq1),
                _mm512_permutex2var_pd(sq0, idx_im, sq1),
            );
            _mm512_storeu_pd(
                out.add(i * 8),
                if SCALED {
                    _mm512_mul_pd(norms, s)
                } else {
                    norms
                },
            );
        }

        let tail = octs * 8;
        for (j, c) in src[tail..].iter().enumerate() {
            *out.add(tail + j) = if SCALED {
                c.norm_sqr() * scale
            } else {
                c.norm_sqr()
            };
        }
    }
}

pub(crate) fn norm_sqr_to_slice(src: &[Complex64], dst: &mut [f64]) {
    norm_sqr_to_slice_impl::<false>(src, dst, 1.0);
}

pub(crate) fn norm_sqr_to_slice_scaled(src: &[Complex64], dst: &mut [f64], scale: f64) {
    norm_sqr_to_slice_impl::<true>(src, dst, scale);
}

#[inline(always)]
fn norm_sqr_to_slice_impl<const SCALED: bool>(src: &[Complex64], dst: &mut [f64], scale: f64) {
    debug_assert!(dst.len() >= src.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            SimdTier::Avx512 => {
                // SAFETY: AVX-512F detected. The destination length debug assert keeps
                // all stores in bounds.
                unsafe { norm_sqr_to_slice_avx512::<SCALED>(src, dst, scale) };
                return;
            }
            SimdTier::Avx2Fma => {
                // SAFETY: AVX2 checked above. The destination length debug
                // assert keeps all stores in bounds.
                unsafe { norm_sqr_to_slice_avx2::<SCALED>(src, dst, scale) };
                return;
            }
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    #[cfg(target_arch = "aarch64")]
    if src.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64. The
        // destination length debug assert keeps all stores in bounds.
        unsafe { norm_sqr_to_slice_neon::<SCALED>(src, dst, scale) };
        return;
    }
    for (i, c) in src.iter().enumerate() {
        dst[i] = if SCALED {
            c.norm_sqr() * scale
        } else {
            c.norm_sqr()
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn norm_sqr_to_slice_neon<const SCALED: bool>(
    src: &[Complex64],
    dst: &mut [f64],
    scale: f64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let inp = src.as_ptr() as *const f64;
        let out = dst.as_mut_ptr();
        let s = vdupq_n_f64(scale);
        let pairs = src.len() / 2;
        for i in 0..pairs {
            let base = i * 4;
            let v0 = vld1q_f64(inp.add(base));
            let v1 = vld1q_f64(inp.add(base + 2));
            let sq0 = vmulq_f64(v0, v0);
            let sq1 = vmulq_f64(v1, v1);
            let ns = vpaddq_f64(sq0, sq1);
            vst1q_f64(out.add(i * 2), if SCALED { vmulq_f64(ns, s) } else { ns });
        }
        if !src.len().is_multiple_of(2) {
            let last = src.len() - 1;
            *out.add(last) = if SCALED {
                src[last].norm_sqr() * scale
            } else {
                src[last].norm_sqr()
            };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn zero_slice_avx2(slice: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let z = _mm256_setzero_pd();
        let ptr = slice.as_mut_ptr() as *mut f64;
        let pairs = slice.len() / 2;
        for i in 0..pairs {
            _mm256_storeu_pd(ptr.add(i * 4), z);
        }
        if !slice.len().is_multiple_of(2) {
            slice[slice.len() - 1] = Complex64::new(0.0, 0.0);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn zero_slice_avx512(slice: &mut [Complex64]) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let z = _mm512_setzero_pd();
        let ptr = slice.as_mut_ptr() as *mut f64;
        for_each_quad!(slice.len(), |off, k| {
            store_pd(ptr.add(off), k, z);
        });
    }
}

pub(crate) fn zero_slice(slice: &mut [Complex64]) {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE {
        match simd_tier() {
            SimdTier::Avx512 => {
                // SAFETY: AVX-512F detected, and the helper writes only within the
                // mutable slice.
                unsafe { zero_slice_avx512(slice) };
                return;
            }
            SimdTier::Avx2Fma => {
                // SAFETY: AVX2 checked above, and the helper writes only
                // within the mutable slice.
                unsafe { zero_slice_avx2(slice) };
                return;
            }
            SimdTier::Fma | SimdTier::Sse2 => {}
        }
    }
    let zero = Complex64::new(0.0, 0.0);
    for amp in slice.iter_mut() {
        *amp = zero;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn scale_complex_slice_avx512(slice: &mut [Complex64], factor: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let rr = _mm512_set1_pd(factor.re);
        let ii = _mm512_set1_pd(factor.im);
        let ptr = slice.as_mut_ptr() as *mut f64;
        for_each_quad!(slice.len(), |off, k| {
            let v = load_pd(ptr.add(off), k);
            store_pd(ptr.add(off), k, complex_mul_avx512(rr, ii, v));
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn scale_complex_slice_avx2fma(slice: &mut [Complex64], factor: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let rr = _mm256_set1_pd(factor.re);
        let ii = _mm256_set1_pd(factor.im);
        let ptr = slice.as_mut_ptr() as *mut f64;
        let pairs = slice.len() / 2;
        for i in 0..pairs {
            let off = i * 4;
            let v = _mm256_loadu_pd(ptr.add(off));
            let v_swap = _mm256_permute_pd(v, 0b0101);
            let t = _mm256_mul_pd(ii, v_swap);
            let result = _mm256_fmaddsub_pd(rr, v, t);
            _mm256_storeu_pd(ptr.add(off), result);
        }
        if !slice.len().is_multiple_of(2) {
            slice[slice.len() - 1] *= factor;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn scale_complex_slice_fma(slice: &mut [Complex64], factor: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let rr = _mm_set1_pd(factor.re);
        let ii = _mm_set1_pd(factor.im);
        let ptr = slice.as_mut_ptr() as *mut f64;
        for i in 0..slice.len() {
            let p = ptr.add(i * 2);
            let v = _mm_loadu_pd(p);
            let v_swap = _mm_shuffle_pd(v, v, 0b01);
            let t = _mm_mul_pd(ii, v_swap);
            let result = _mm_fmaddsub_pd(rr, v, t);
            _mm_storeu_pd(p, result);
        }
    }
}

pub(crate) fn scale_complex_slice(slice: &mut [Complex64], factor: Complex64) {
    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd_tier();
        if slice.len() >= MIN_SIMD_SLICE {
            match tier {
                SimdTier::Avx512 => {
                    // SAFETY: AVX-512F detected, and the helper writes only within the
                    // mutable slice.
                    unsafe { scale_complex_slice_avx512(slice, factor) };
                    return;
                }
                SimdTier::Avx2Fma => {
                    // SAFETY: AVX2 and FMA checked above, and the helper
                    // writes only within the mutable slice.
                    unsafe { scale_complex_slice_avx2fma(slice, factor) };
                    return;
                }
                SimdTier::Fma | SimdTier::Sse2 => {}
            }
        }
        if slice.len() >= 2 && tier != SimdTier::Sse2 {
            // SAFETY: FMA checked above, and the helper writes only
            // within the mutable slice.
            unsafe { scale_complex_slice_fma(slice, factor) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64, and the loop
        // writes only within the mutable slice.
        unsafe {
            let c_rr = vdupq_n_f64(factor.re);
            let c_ii_as = vcombine_f64(vdup_n_f64(-factor.im), vdup_n_f64(factor.im));
            let ptr = slice.as_mut_ptr() as *mut f64;
            for i in 0..slice.len() {
                let p = ptr.add(i * 2);
                let v = vld1q_f64(p);
                vst1q_f64(p, complex_mul_neon(c_rr, c_ii_as, v));
            }
        }
        return;
    }
    for amp in slice.iter_mut() {
        *amp *= factor;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "distributed"))]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn combine_global_half_avx512(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s_rr = _mm512_set1_pd(c_self.re);
        let s_ii = _mm512_set1_pd(c_self.im);
        let r_rr = _mm512_set1_pd(c_remote.re);
        let r_ii = _mm512_set1_pd(c_remote.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let rp = remote.as_ptr() as *const f64;
        for_each_quad!(dst.len(), |off, k| {
            let d = load_pd(dp.add(off), k);
            let r = load_pd(rp.add(off), k);
            let acc = _mm512_add_pd(
                complex_mul_avx512(s_rr, s_ii, d),
                complex_mul_avx512(r_rr, r_ii, r),
            );
            store_pd(dp.add(off), k, acc);
        });
    }
}

#[cfg(all(target_arch = "x86_64", feature = "distributed"))]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn combine_global_half_avx2fma(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s_rr = _mm256_set1_pd(c_self.re);
        let s_ii = _mm256_set1_pd(c_self.im);
        let r_rr = _mm256_set1_pd(c_remote.re);
        let r_ii = _mm256_set1_pd(c_remote.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let rp = remote.as_ptr() as *const f64;
        let pairs = dst.len() / 2;
        for i in 0..pairs {
            let off = i * 4;
            let d = _mm256_loadu_pd(dp.add(off));
            let r = _mm256_loadu_pd(rp.add(off));
            let acc = _mm256_add_pd(
                complex_mul_avx2fma(s_rr, s_ii, d),
                complex_mul_avx2fma(r_rr, r_ii, r),
            );
            _mm256_storeu_pd(dp.add(off), acc);
        }
        if !dst.len().is_multiple_of(2) {
            let last = dst.len() - 1;
            dst[last] = c_self * dst[last] + c_remote * remote[last];
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "distributed"))]
#[target_feature(enable = "fma")]
pub(super) unsafe fn combine_global_half_fma(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s_rr = _mm_set1_pd(c_self.re);
        let s_ii = _mm_set1_pd(c_self.im);
        let r_rr = _mm_set1_pd(c_remote.re);
        let r_ii = _mm_set1_pd(c_remote.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let rp = remote.as_ptr() as *const f64;
        for i in 0..dst.len() {
            let d = _mm_loadu_pd(dp.add(i * 2));
            let r = _mm_loadu_pd(rp.add(i * 2));
            let acc = _mm_add_pd(
                complex_mul_fma(s_rr, s_ii, d),
                complex_mul_fma(r_rr, r_ii, r),
            );
            _mm_storeu_pd(dp.add(i * 2), acc);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "distributed"))]
unsafe fn combine_global_half_neon(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let s_rr = vdupq_n_f64(c_self.re);
        let s_ii_as = vcombine_f64(vdup_n_f64(-c_self.im), vdup_n_f64(c_self.im));
        let r_rr = vdupq_n_f64(c_remote.re);
        let r_ii_as = vcombine_f64(vdup_n_f64(-c_remote.im), vdup_n_f64(c_remote.im));
        let dp = dst.as_mut_ptr() as *mut f64;
        let rp = remote.as_ptr() as *const f64;
        for i in 0..dst.len() {
            let d = vld1q_f64(dp.add(i * 2));
            let r = vld1q_f64(rp.add(i * 2));
            let acc = vaddq_f64(
                complex_mul_neon(s_rr, s_ii_as, d),
                complex_mul_neon(r_rr, r_ii_as, r),
            );
            vst1q_f64(dp.add(i * 2), acc);
        }
    }
}

#[cfg(feature = "distributed")]
#[inline(always)]
pub(super) fn combine_global_half_scalar(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    for (d, &r) in dst.iter_mut().zip(remote.iter()) {
        *d = c_self * *d + c_remote * r;
    }
}

/// Combine two equal-length amplitude slices in place:
/// `dst[i] = c_self * dst[i] + c_remote * remote[i]`.
///
/// Used after a distributed rank exchange for a global one qubit gate.
#[cfg(feature = "distributed")]
pub(crate) fn combine_global_half(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    debug_assert_eq!(dst.len(), remote.len());
    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd_tier();
        if dst.len() >= MIN_SIMD_SLICE {
            match tier {
                SimdTier::Avx512 => {
                    // SAFETY: AVX-512F detected; slices have equal length.
                    unsafe { combine_global_half_avx512(dst, remote, c_self, c_remote) };
                    return;
                }
                SimdTier::Avx2Fma => {
                    // SAFETY: AVX2 and FMA are available; slices have equal length.
                    unsafe { combine_global_half_avx2fma(dst, remote, c_self, c_remote) };
                    return;
                }
                SimdTier::Fma | SimdTier::Sse2 => {}
            }
        }
        if dst.len() >= 2 && tier != SimdTier::Sse2 {
            // SAFETY: FMA is available; slices have equal length.
            unsafe { combine_global_half_fma(dst, remote, c_self, c_remote) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if dst.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64; slices have equal length.
        unsafe { combine_global_half_neon(dst, remote, c_self, c_remote) };
        return;
    }
    combine_global_half_scalar(dst, remote, c_self, c_remote);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn scale_complex_to_slice_avx512(
    dst: &mut [Complex64],
    src: &[Complex64],
    factor: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(dst.len() >= src.len());
        let rr = _mm512_set1_pd(factor.re);
        let ii = _mm512_set1_pd(factor.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let sp = src.as_ptr() as *const f64;
        for_each_quad!(src.len(), |off, k| {
            let v = load_pd(sp.add(off), k);
            store_pd(dp.add(off), k, complex_mul_avx512(rr, ii, v));
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn scale_complex_to_slice_avx2fma(
    dst: &mut [Complex64],
    src: &[Complex64],
    factor: Complex64,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(dst.len() >= src.len());
        let rr = _mm256_set1_pd(factor.re);
        let ii = _mm256_set1_pd(factor.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let sp = src.as_ptr() as *const f64;
        let pairs = src.len() / 2;
        for i in 0..pairs {
            let off = i * 4;
            let v = _mm256_loadu_pd(sp.add(off));
            let v_swap = _mm256_permute_pd(v, 0b0101);
            let t = _mm256_mul_pd(ii, v_swap);
            let result = _mm256_fmaddsub_pd(rr, v, t);
            _mm256_storeu_pd(dp.add(off), result);
        }
        if !src.len().is_multiple_of(2) {
            let last = src.len() - 1;
            dst[last] = src[last] * factor;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn scale_complex_to_slice_fma(dst: &mut [Complex64], src: &[Complex64], factor: Complex64) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(dst.len() >= src.len());
        let rr = _mm_set1_pd(factor.re);
        let ii = _mm_set1_pd(factor.im);
        let dp = dst.as_mut_ptr() as *mut f64;
        let sp = src.as_ptr() as *const f64;
        for i in 0..src.len() {
            let v = _mm_loadu_pd(sp.add(i * 2));
            let v_swap = _mm_shuffle_pd(v, v, 0b01);
            let t = _mm_mul_pd(ii, v_swap);
            let result = _mm_fmaddsub_pd(rr, v, t);
            _mm_storeu_pd(dp.add(i * 2), result);
        }
    }
}

pub(crate) fn scale_complex_to_slice(dst: &mut [Complex64], src: &[Complex64], factor: Complex64) {
    assert!(
        dst.len() >= src.len(),
        "destination slice shorter than source slice"
    );
    #[cfg(target_arch = "x86_64")]
    {
        let tier = simd_tier();
        if src.len() >= MIN_SIMD_SLICE {
            match tier {
                SimdTier::Avx512 => {
                    // SAFETY: AVX-512F detected. The assert keeps every load and store
                    // in bounds, Rust references provide valid slices, and SIMD avoids
                    // the measured scalar bottleneck in large factored merge copies.
                    unsafe { scale_complex_to_slice_avx512(dst, src, factor) };
                    return;
                }
                SimdTier::Avx2Fma => {
                    // SAFETY: AVX2 and FMA checked above. The assert keeps
                    // every load and store in bounds, Rust references provide valid
                    // slices, and SIMD avoids the measured scalar bottleneck in large
                    // factored merge copies.
                    unsafe { scale_complex_to_slice_avx2fma(dst, src, factor) };
                    return;
                }
                SimdTier::Fma | SimdTier::Sse2 => {}
            }
        }
        if src.len() >= 2 && tier != SimdTier::Sse2 {
            // SAFETY: FMA checked above. The assert keeps every
            // load and store in bounds, Rust references provide valid slices,
            // and SIMD avoids the measured scalar bottleneck in factored merge
            // copies.
            unsafe { scale_complex_to_slice_fma(dst, src, factor) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if src.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is baseline on aarch64. The assert
        // keeps pointer arithmetic in bounds, Rust references provide valid
        // slices, and vector loads avoid the measured scalar bottleneck in
        // large factored merge copies.
        unsafe {
            let c_rr = vdupq_n_f64(factor.re);
            let c_ii_as = vcombine_f64(vdup_n_f64(-factor.im), vdup_n_f64(factor.im));
            let dp = dst.as_mut_ptr() as *mut f64;
            let sp = src.as_ptr() as *const f64;
            for i in 0..src.len() {
                let v = vld1q_f64(sp.add(i * 2));
                vst1q_f64(dp.add(i * 2), complex_mul_neon(c_rr, c_ii_as, v));
            }
        }
        return;
    }
    for (i, &c) in src.iter().enumerate() {
        dst[i] = c * factor;
    }
}
