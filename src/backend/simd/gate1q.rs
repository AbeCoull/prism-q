//! One-qubit gate kernels: the 2x2 matrix sweep behind [`PreparedGate1q`] in every tier,
//! the diagonal fast path, and the one-qubit density matrix sandwich.

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
use super::complex_mul_neon;
#[cfg(target_arch = "x86_64")]
use super::{
    SimdTier, complex_mul_avx2fma, complex_mul_avx512, complex_mul_fma, complex_mul_sse2,
    for_each_quad, load_pd, simd_tier, store_pd,
};

#[cfg(target_arch = "x86_64")]
struct MatBroadcast {
    m00_rr: __m128d,
    m00_ii: __m128d,
    m01_rr: __m128d,
    m01_ii: __m128d,
    m10_rr: __m128d,
    m10_ii: __m128d,
    m11_rr: __m128d,
    m11_ii: __m128d,
}

#[cfg(target_arch = "x86_64")]
impl MatBroadcast {
    #[inline(always)]
    fn from_matrix(mat: &[[Complex64; 2]; 2]) -> Self {
        // SAFETY: x86_64 guarantees SSE2, and _mm_set1_pd only broadcasts
        // scalar f64 values into SIMD registers.
        unsafe {
            Self {
                m00_rr: _mm_set1_pd(mat[0][0].re),
                m00_ii: _mm_set1_pd(mat[0][0].im),
                m01_rr: _mm_set1_pd(mat[0][1].re),
                m01_ii: _mm_set1_pd(mat[0][1].im),
                m10_rr: _mm_set1_pd(mat[1][0].re),
                m10_ii: _mm_set1_pd(mat[1][0].im),
                m11_rr: _mm_set1_pd(mat[1][1].re),
                m11_ii: _mm_set1_pd(mat[1][1].im),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
struct MatBroadcast256 {
    m00_rr: __m256d,
    m00_ii: __m256d,
    m01_rr: __m256d,
    m01_ii: __m256d,
    m10_rr: __m256d,
    m10_ii: __m256d,
    m11_rr: __m256d,
    m11_ii: __m256d,
}

#[cfg(target_arch = "x86_64")]
impl MatBroadcast256 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn from_matrix(mat: &[[Complex64; 2]; 2]) -> Self {
        Self {
            m00_rr: _mm256_set1_pd(mat[0][0].re),
            m00_ii: _mm256_set1_pd(mat[0][0].im),
            m01_rr: _mm256_set1_pd(mat[0][1].re),
            m01_ii: _mm256_set1_pd(mat[0][1].im),
            m10_rr: _mm256_set1_pd(mat[1][0].re),
            m10_ii: _mm256_set1_pd(mat[1][0].im),
            m11_rr: _mm256_set1_pd(mat[1][1].re),
            m11_ii: _mm256_set1_pd(mat[1][1].im),
        }
    }
}

#[cfg(target_arch = "x86_64")]
struct MatBroadcast512 {
    m00_rr: __m512d,
    m00_ii: __m512d,
    m01_rr: __m512d,
    m01_ii: __m512d,
    m10_rr: __m512d,
    m10_ii: __m512d,
    m11_rr: __m512d,
    m11_ii: __m512d,
}

#[cfg(target_arch = "x86_64")]
impl MatBroadcast512 {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn from_256(m: &MatBroadcast256) -> Self {
        Self {
            m00_rr: _mm512_broadcast_f64x4(m.m00_rr),
            m00_ii: _mm512_broadcast_f64x4(m.m00_ii),
            m01_rr: _mm512_broadcast_f64x4(m.m01_rr),
            m01_ii: _mm512_broadcast_f64x4(m.m01_ii),
            m10_rr: _mm512_broadcast_f64x4(m.m10_rr),
            m10_ii: _mm512_broadcast_f64x4(m.m10_ii),
            m11_rr: _mm512_broadcast_f64x4(m.m11_rr),
            m11_ii: _mm512_broadcast_f64x4(m.m11_ii),
        }
    }
}

#[cfg(target_arch = "aarch64")]
struct MatBroadcast {
    m00_rr: float64x2_t,
    m00_ii_as: float64x2_t,
    m01_rr: float64x2_t,
    m01_ii_as: float64x2_t,
    m10_rr: float64x2_t,
    m10_ii_as: float64x2_t,
    m11_rr: float64x2_t,
    m11_ii_as: float64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl MatBroadcast {
    #[inline(always)]
    fn from_matrix(mat: &[[Complex64; 2]; 2]) -> Self {
        // SAFETY: NEON is baseline on aarch64, and these
        // intrinsics only broadcast scalar f64 values into SIMD registers.
        unsafe {
            Self {
                m00_rr: vdupq_n_f64(mat[0][0].re),
                m00_ii_as: vcombine_f64(vdup_n_f64(-mat[0][0].im), vdup_n_f64(mat[0][0].im)),
                m01_rr: vdupq_n_f64(mat[0][1].re),
                m01_ii_as: vcombine_f64(vdup_n_f64(-mat[0][1].im), vdup_n_f64(mat[0][1].im)),
                m10_rr: vdupq_n_f64(mat[1][0].re),
                m10_ii_as: vcombine_f64(vdup_n_f64(-mat[1][0].im), vdup_n_f64(mat[1][0].im)),
                m11_rr: vdupq_n_f64(mat[1][1].re),
                m11_ii_as: vcombine_f64(vdup_n_f64(-mat[1][1].im), vdup_n_f64(mat[1][1].im)),
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_slices_sse2(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(lo.len(), hi.len());
        let n = lo.len();
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;
        let sign_mask = _mm_set_pd(0.0, -0.0_f64);

        for i in 0..n {
            let a_ptr = lo_ptr.add(i * 2);
            let b_ptr = hi_ptr.add(i * 2);

            let a = _mm_loadu_pd(a_ptr);
            let b = _mm_loadu_pd(b_ptr);

            let m00_a = complex_mul_sse2(mat.m00_rr, mat.m00_ii, a, sign_mask);
            let m01_b = complex_mul_sse2(mat.m01_rr, mat.m01_ii, b, sign_mask);
            let new_a = _mm_add_pd(m00_a, m01_b);

            let m10_a = complex_mul_sse2(mat.m10_rr, mat.m10_ii, a, sign_mask);
            let m11_b = complex_mul_sse2(mat.m11_rr, mat.m11_ii, b, sign_mask);
            let new_b = _mm_add_pd(m10_a, m11_b);

            _mm_storeu_pd(a_ptr, new_a);
            _mm_storeu_pd(b_ptr, new_b);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn apply_slices_fma(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(lo.len(), hi.len());
        let n = lo.len();
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;

        for i in 0..n {
            let a_ptr = lo_ptr.add(i * 2);
            let b_ptr = hi_ptr.add(i * 2);

            let a = _mm_loadu_pd(a_ptr);
            let b = _mm_loadu_pd(b_ptr);

            let m00_a = complex_mul_fma(mat.m00_rr, mat.m00_ii, a);
            let m01_b = complex_mul_fma(mat.m01_rr, mat.m01_ii, b);
            let new_a = _mm_add_pd(m00_a, m01_b);

            let m10_a = complex_mul_fma(mat.m10_rr, mat.m10_ii, a);
            let m11_b = complex_mul_fma(mat.m11_rr, mat.m11_ii, b);
            let new_b = _mm_add_pd(m10_a, m11_b);

            _mm_storeu_pd(a_ptr, new_a);
            _mm_storeu_pd(b_ptr, new_b);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_slices_neon(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(lo.len(), hi.len());
        let n = lo.len();
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;

        for i in 0..n {
            let a_ptr = lo_ptr.add(i * 2);
            let b_ptr = hi_ptr.add(i * 2);

            let a = vld1q_f64(a_ptr);
            let b = vld1q_f64(b_ptr);

            let new_a = vaddq_f64(
                complex_mul_neon(mat.m00_rr, mat.m00_ii_as, a),
                complex_mul_neon(mat.m01_rr, mat.m01_ii_as, b),
            );
            let new_b = vaddq_f64(
                complex_mul_neon(mat.m10_rr, mat.m10_ii_as, a),
                complex_mul_neon(mat.m11_rr, mat.m11_ii_as, b),
            );

            vst1q_f64(a_ptr, new_a);
            vst1q_f64(b_ptr, new_b);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_slices_avx2fma(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    mat: &MatBroadcast256,
    mat128: &MatBroadcast,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(lo.len(), hi.len());
        let n = lo.len();
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;
        let pairs = n / 2;

        for i in 0..pairs {
            let off = i * 4;
            let a = _mm256_loadu_pd(lo_ptr.add(off));
            let b = _mm256_loadu_pd(hi_ptr.add(off));

            let new_a = _mm256_add_pd(
                complex_mul_avx2fma(mat.m00_rr, mat.m00_ii, a),
                complex_mul_avx2fma(mat.m01_rr, mat.m01_ii, b),
            );
            let new_b = _mm256_add_pd(
                complex_mul_avx2fma(mat.m10_rr, mat.m10_ii, a),
                complex_mul_avx2fma(mat.m11_rr, mat.m11_ii, b),
            );

            _mm256_storeu_pd(lo_ptr.add(off), new_a);
            _mm256_storeu_pd(hi_ptr.add(off), new_b);
        }

        if !n.is_multiple_of(2) {
            let off = pairs * 4;
            apply_pair_fma(lo_ptr.add(off), hi_ptr.add(off), mat128);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn apply_slices_avx512(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast256) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert_eq!(lo.len(), hi.len());
        let m = MatBroadcast512::from_256(mat);
        let lo_ptr = lo.as_mut_ptr() as *mut f64;
        let hi_ptr = hi.as_mut_ptr() as *mut f64;
        for_each_quad!(lo.len(), |off, k| {
            let a = load_pd(lo_ptr.add(off), k);
            let b = load_pd(hi_ptr.add(off), k);
            let new_a = _mm512_add_pd(
                complex_mul_avx512(m.m00_rr, m.m00_ii, a),
                complex_mul_avx512(m.m01_rr, m.m01_ii, b),
            );
            let new_b = _mm512_add_pd(
                complex_mul_avx512(m.m10_rr, m.m10_ii, a),
                complex_mul_avx512(m.m11_rr, m.m11_ii, b),
            );
            store_pd(lo_ptr.add(off), k, new_a);
            store_pd(hi_ptr.add(off), k, new_b);
        });
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_full_loop_sse2(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let half = 1usize << target;
        let mask = half - 1;
        let num_pairs = state.len() >> 1;
        let base = state.as_mut_ptr() as *mut f64;
        let sign_mask = _mm_set_pd(0.0, -0.0_f64);

        for k in 0..num_pairs {
            let i0 = (k & !mask) << 1 | (k & mask);
            let i1 = i0 | half;
            let a_ptr = base.add(i0 * 2);
            let b_ptr = base.add(i1 * 2);

            let a = _mm_loadu_pd(a_ptr);
            let b = _mm_loadu_pd(b_ptr);

            let m00_a = complex_mul_sse2(mat.m00_rr, mat.m00_ii, a, sign_mask);
            let m01_b = complex_mul_sse2(mat.m01_rr, mat.m01_ii, b, sign_mask);
            let new_a = _mm_add_pd(m00_a, m01_b);

            let m10_a = complex_mul_sse2(mat.m10_rr, mat.m10_ii, a, sign_mask);
            let m11_b = complex_mul_sse2(mat.m11_rr, mat.m11_ii, b, sign_mask);
            let new_b = _mm_add_pd(m10_a, m11_b);

            _mm_storeu_pd(a_ptr, new_a);
            _mm_storeu_pd(b_ptr, new_b);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn apply_full_loop_fma(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let half = 1usize << target;
        let mask = half - 1;
        let num_pairs = state.len() >> 1;
        let base = state.as_mut_ptr() as *mut f64;

        for k in 0..num_pairs {
            let i0 = (k & !mask) << 1 | (k & mask);
            let i1 = i0 | half;
            let a_ptr = base.add(i0 * 2);
            let b_ptr = base.add(i1 * 2);

            let a = _mm_loadu_pd(a_ptr);
            let b = _mm_loadu_pd(b_ptr);

            let m00_a = complex_mul_fma(mat.m00_rr, mat.m00_ii, a);
            let m01_b = complex_mul_fma(mat.m01_rr, mat.m01_ii, b);
            let new_a = _mm_add_pd(m00_a, m01_b);

            let m10_a = complex_mul_fma(mat.m10_rr, mat.m10_ii, a);
            let m11_b = complex_mul_fma(mat.m11_rr, mat.m11_ii, b);
            let new_b = _mm_add_pd(m10_a, m11_b);

            _mm_storeu_pd(a_ptr, new_a);
            _mm_storeu_pd(b_ptr, new_b);
        }
    }
}

/// Requires `target >= 1`, so each half block holds whole 256-bit steps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_full_loop_avx2fma_inline(
    state: &mut [Complex64],
    target: usize,
    mat: &MatBroadcast256,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(target >= 1);
        debug_assert!(state.len().is_multiple_of(1usize << (target + 1)));
        let half = 1usize << target;
        let block_size = half << 1;
        let avx_pairs = half / 2;
        let base = state.as_mut_ptr() as *mut f64;
        let n = state.len();
        let mut offset = 0;
        while offset < n {
            let lo_ptr = base.add(offset * 2);
            let hi_ptr = base.add((offset + half) * 2);
            for i in 0..avx_pairs {
                let off = i * 4;
                let a = _mm256_loadu_pd(lo_ptr.add(off));
                let b = _mm256_loadu_pd(hi_ptr.add(off));
                let new_a = _mm256_add_pd(
                    complex_mul_avx2fma(mat.m00_rr, mat.m00_ii, a),
                    complex_mul_avx2fma(mat.m01_rr, mat.m01_ii, b),
                );
                let new_b = _mm256_add_pd(
                    complex_mul_avx2fma(mat.m10_rr, mat.m10_ii, a),
                    complex_mul_avx2fma(mat.m11_rr, mat.m11_ii, b),
                );
                _mm256_storeu_pd(lo_ptr.add(off), new_a);
                _mm256_storeu_pd(hi_ptr.add(off), new_b);
            }
            offset += block_size;
        }
    }
}

/// Requires `target >= 2`, so each half block holds whole 512-bit steps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_full_loop_avx512(state: &mut [Complex64], target: usize, mat: &MatBroadcast256) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let m = MatBroadcast512::from_256(mat);
        let half = 1usize << target;
        let block_size = half << 1;
        let base = state.as_mut_ptr() as *mut f64;
        let n = state.len();
        let mut offset = 0;
        while offset < n {
            let lo_ptr = base.add(offset * 2);
            let hi_ptr = base.add((offset + half) * 2);
            for off in (0..2 * half).step_by(8) {
                let a = _mm512_loadu_pd(lo_ptr.add(off));
                let b = _mm512_loadu_pd(hi_ptr.add(off));
                let new_a = _mm512_add_pd(
                    complex_mul_avx512(m.m00_rr, m.m00_ii, a),
                    complex_mul_avx512(m.m01_rr, m.m01_ii, b),
                );
                let new_b = _mm512_add_pd(
                    complex_mul_avx512(m.m10_rr, m.m10_ii, a),
                    complex_mul_avx512(m.m11_rr, m.m11_ii, b),
                );
                _mm512_storeu_pd(lo_ptr.add(off), new_a);
                _mm512_storeu_pd(hi_ptr.add(off), new_b);
            }
            offset += block_size;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_full_loop_q0_avx2fma(state: &mut [Complex64], mat: &MatBroadcast256) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(state.len().is_multiple_of(2));
        let diag_rr = _mm256_permute2f128_pd(mat.m00_rr, mat.m11_rr, 0x20);
        let diag_ii = _mm256_permute2f128_pd(mat.m00_ii, mat.m11_ii, 0x20);
        let off_rr = _mm256_permute2f128_pd(mat.m01_rr, mat.m10_rr, 0x20);
        let off_ii = _mm256_permute2f128_pd(mat.m01_ii, mat.m10_ii, 0x20);
        let base = state.as_mut_ptr() as *mut f64;
        let num_pairs = state.len() >> 1;
        for k in 0..num_pairs {
            let p = base.add(k * 4);
            let x = _mm256_loadu_pd(p);
            let sx = _mm256_permute2f128_pd(x, x, 0x01);
            let res = _mm256_add_pd(
                complex_mul_avx2fma(diag_rr, diag_ii, x),
                complex_mul_avx2fma(off_rr, off_ii, sx),
            );
            _mm256_storeu_pd(p, res);
        }
    }
}

/// Requires `state.len() >= 4` (multiple of 4), so each 512-bit step holds two qubit-0 pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn apply_full_loop_q0_avx512(state: &mut [Complex64], mat: &MatBroadcast256) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(state.len() >= 4 && state.len().is_multiple_of(4));
        let diag_rr = _mm512_broadcast_f64x4(_mm256_permute2f128_pd(mat.m00_rr, mat.m11_rr, 0x20));
        let diag_ii = _mm512_broadcast_f64x4(_mm256_permute2f128_pd(mat.m00_ii, mat.m11_ii, 0x20));
        let off_rr = _mm512_broadcast_f64x4(_mm256_permute2f128_pd(mat.m01_rr, mat.m10_rr, 0x20));
        let off_ii = _mm512_broadcast_f64x4(_mm256_permute2f128_pd(mat.m01_ii, mat.m10_ii, 0x20));
        let base = state.as_mut_ptr() as *mut f64;
        let num_quads = state.len() >> 2;
        for k in 0..num_quads {
            let p = base.add(k * 8);
            let x = _mm512_loadu_pd(p);
            let sx = _mm512_permutex_pd::<0x4E>(x);
            let res = _mm512_add_pd(
                complex_mul_avx512(diag_rr, diag_ii, x),
                complex_mul_avx512(off_rr, off_ii, sx),
            );
            _mm512_storeu_pd(p, res);
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_full_loop_neon(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let half = 1usize << target;
        let mask = half - 1;
        let num_pairs = state.len() >> 1;
        let base = state.as_mut_ptr() as *mut f64;

        for k in 0..num_pairs {
            let i0 = (k & !mask) << 1 | (k & mask);
            let i1 = i0 | half;
            let a_ptr = base.add(i0 * 2);
            let b_ptr = base.add(i1 * 2);

            let a = vld1q_f64(a_ptr);
            let b = vld1q_f64(b_ptr);

            let new_a = vaddq_f64(
                complex_mul_neon(mat.m00_rr, mat.m00_ii_as, a),
                complex_mul_neon(mat.m01_rr, mat.m01_ii_as, b),
            );
            let new_b = vaddq_f64(
                complex_mul_neon(mat.m10_rr, mat.m10_ii_as, a),
                complex_mul_neon(mat.m11_rr, mat.m11_ii_as, b),
            );

            vst1q_f64(a_ptr, new_a);
            vst1q_f64(b_ptr, new_b);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_pair_neon(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let a = vld1q_f64(a_ptr);
        let b = vld1q_f64(b_ptr);

        let new_a = vaddq_f64(
            complex_mul_neon(mat.m00_rr, mat.m00_ii_as, a),
            complex_mul_neon(mat.m01_rr, mat.m01_ii_as, b),
        );
        let new_b = vaddq_f64(
            complex_mul_neon(mat.m10_rr, mat.m10_ii_as, a),
            complex_mul_neon(mat.m11_rr, mat.m11_ii_as, b),
        );

        vst1q_f64(a_ptr, new_a);
        vst1q_f64(b_ptr, new_b);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_pair_sse2(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let sign_mask = _mm_set_pd(0.0, -0.0_f64);
        let a = _mm_loadu_pd(a_ptr);
        let b = _mm_loadu_pd(b_ptr);

        let m00_a = complex_mul_sse2(mat.m00_rr, mat.m00_ii, a, sign_mask);
        let m01_b = complex_mul_sse2(mat.m01_rr, mat.m01_ii, b, sign_mask);
        let new_a = _mm_add_pd(m00_a, m01_b);

        let m10_a = complex_mul_sse2(mat.m10_rr, mat.m10_ii, a, sign_mask);
        let m11_b = complex_mul_sse2(mat.m11_rr, mat.m11_ii, b, sign_mask);
        let new_b = _mm_add_pd(m10_a, m11_b);

        _mm_storeu_pd(a_ptr, new_a);
        _mm_storeu_pd(b_ptr, new_b);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn apply_pair_fma(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let a = _mm_loadu_pd(a_ptr);
        let b = _mm_loadu_pd(b_ptr);

        let m00_a = complex_mul_fma(mat.m00_rr, mat.m00_ii, a);
        let m01_b = complex_mul_fma(mat.m01_rr, mat.m01_ii, b);
        let new_a = _mm_add_pd(m00_a, m01_b);

        let m10_a = complex_mul_fma(mat.m10_rr, mat.m10_ii, a);
        let m11_b = complex_mul_fma(mat.m11_rr, mat.m11_ii, b);
        let new_b = _mm_add_pd(m10_a, m11_b);

        _mm_storeu_pd(a_ptr, new_a);
        _mm_storeu_pd(b_ptr, new_b);
    }
}

#[inline(always)]
#[allow(dead_code)]
pub(super) fn apply_slices_scalar(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    mat: &[[Complex64; 2]; 2],
) {
    debug_assert_eq!(lo.len(), hi.len());
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        let v0 = *a;
        let v1 = *b;
        *a = mat[0][0] * v0 + mat[0][1] * v1;
        *b = mat[1][0] * v0 + mat[1][1] * v1;
    }
}

/// Single-qubit gate with its matrix broadcast and SIMD tier chosen once, then
/// applied per chunk.
pub(crate) struct PreparedGate1q {
    #[cfg(target_arch = "x86_64")]
    broadcast: MatBroadcast,
    #[cfg(target_arch = "x86_64")]
    broadcast256: Option<MatBroadcast256>,
    #[cfg(target_arch = "x86_64")]
    pub(super) tier: SimdTier,
    #[cfg(target_arch = "aarch64")]
    broadcast: MatBroadcast,
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    mat: [[Complex64; 2]; 2],
}

impl PreparedGate1q {
    #[inline(always)]
    pub(crate) fn new(mat: &[[Complex64; 2]; 2]) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let broadcast = MatBroadcast::from_matrix(mat);
            let tier = simd_tier();
            let broadcast256 = if matches!(tier, SimdTier::Avx512 | SimdTier::Avx2Fma) {
                // SAFETY: AVX support is implied by the detected AVX2 feature.
                Some(unsafe { MatBroadcast256::from_matrix(mat) })
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
                broadcast: MatBroadcast::from_matrix(mat),
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self { mat: *mat }
        }
    }

    #[inline(always)]
    pub(crate) fn apply(&self, lo: &mut [Complex64], hi: &mut [Complex64]) {
        debug_assert_eq!(lo.len(), hi.len());

        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: The selected tier is guarded by runtime feature checks
            // in the constructor. Slices have matching lengths by debug assert.
            unsafe {
                match self.tier {
                    SimdTier::Avx512 => {
                        let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                        apply_slices_avx512(lo, hi, b256);
                    }
                    SimdTier::Avx2Fma => {
                        let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                        apply_slices_avx2fma(lo, hi, b256, &self.broadcast);
                    }
                    SimdTier::Fma => apply_slices_fma(lo, hi, &self.broadcast),
                    SimdTier::Sse2 => apply_slices_sse2(lo, hi, &self.broadcast),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on aarch64, and the
            // caller provides equally sized, non-overlapping slices.
            unsafe { apply_slices_neon(lo, hi, &self.broadcast) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            apply_slices_scalar(lo, hi, &self.mat);
        }
    }

    /// Apply the gate to the full state vector sequentially (one function call).
    ///
    /// Uses mask-based index iteration inside a single `#[target_feature]` call,
    /// avoiding per-chunk function call overhead that hurts small qubit counts.
    #[inline(always)]
    pub(crate) fn apply_full_sequential(&self, state: &mut [Complex64], target: usize) {
        self.apply_loop::<true>(state, target);
    }

    /// Apply the gate within a cache-resident tile, preferring AVX2 when available.
    ///
    /// Unlike `apply_full_sequential`, does not restrict AVX2 by state size.
    /// Use when the slice is known to be L2/L3-resident (e.g. MultiFused tiles).
    #[inline(always)]
    pub(crate) fn apply_tiled(&self, state: &mut [Complex64], target: usize) {
        self.apply_loop::<false>(state, target);
    }

    /// Shared body of [`Self::apply_full_sequential`] and [`Self::apply_tiled`]. With
    /// `CAP_AVX2_STATE` the AVX2 and AVX-512 loops take states of at most 8192 amplitudes.
    #[inline(always)]
    fn apply_loop<const CAP_AVX2_STATE: bool>(&self, state: &mut [Complex64], target: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: The selected tier is guarded by runtime feature checks
            // in the constructor. The target index is validated by circuit construction.
            unsafe {
                const MAX_AVX2_STATE: usize = 8192;
                let wide = !CAP_AVX2_STATE || state.len() <= MAX_AVX2_STATE;
                match self.tier {
                    SimdTier::Avx512 if wide => {
                        let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                        if target >= 2 {
                            apply_full_loop_avx512(state, target, b256);
                        } else if target == 1 {
                            apply_full_loop_avx2fma_inline(state, target, b256);
                        } else if state.len() >= 4 {
                            apply_full_loop_q0_avx512(state, b256);
                        } else {
                            apply_full_loop_q0_avx2fma(state, b256);
                        }
                    }
                    SimdTier::Avx2Fma if wide => {
                        let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                        if target >= 1 {
                            apply_full_loop_avx2fma_inline(state, target, b256);
                        } else {
                            apply_full_loop_q0_avx2fma(state, b256);
                        }
                    }
                    SimdTier::Avx512 | SimdTier::Avx2Fma | SimdTier::Fma => {
                        apply_full_loop_fma(state, target, &self.broadcast);
                    }
                    SimdTier::Sse2 => apply_full_loop_sse2(state, target, &self.broadcast),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on aarch64. The
            // target index is validated by circuit construction.
            unsafe { apply_full_loop_neon(state, target, &self.broadcast) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let half = 1usize << target;
            let mask = half - 1;
            let num_pairs = state.len() >> 1;
            for k in 0..num_pairs {
                let i0 = (k & !mask) << 1 | (k & mask);
                let i1 = i0 | half;
                let a = state[i0];
                let b = state[i1];
                state[i0] = self.mat[0][0] * a + self.mat[0][1] * b;
                state[i1] = self.mat[1][0] * a + self.mat[1][1] * b;
            }
        }
    }

    /// Apply the gate to a single amplitude pair by raw pointer.
    ///
    /// # Safety
    ///
    /// - `a_ptr` and `b_ptr` must point to valid, non-overlapping `Complex64`
    ///   values (cast as `*mut f64` pointing to the `re` field).
    /// - No other reference may alias either pointee for the duration of this call.
    #[inline(always)]
    pub(crate) unsafe fn apply_pair_ptr(&self, a_ptr: *mut f64, b_ptr: *mut f64) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                match self.tier {
                    SimdTier::Avx512 | SimdTier::Avx2Fma | SimdTier::Fma => {
                        apply_pair_fma(a_ptr, b_ptr, &self.broadcast)
                    }
                    SimdTier::Sse2 => apply_pair_sse2(a_ptr, b_ptr, &self.broadcast),
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                apply_pair_neon(a_ptr, b_ptr, &self.broadcast);
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let a = &mut *(a_ptr as *mut Complex64);
                let b = &mut *(b_ptr as *mut Complex64);
                let v0 = *a;
                let v1 = *b;
                *a = self.mat[0][0] * v0 + self.mat[0][1] * v1;
                *b = self.mat[1][0] * v0 + self.mat[1][1] * v1;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn apply_slice_pairs(&self, lo: &mut [Complex64], hi: &mut [Complex64]) {
        debug_assert_eq!(lo.len(), hi.len());
        for k in 0..lo.len() {
            // SAFETY: k < lo.len() == hi.len() by debug_assert.
            unsafe {
                let a_ptr = (lo.as_mut_ptr().add(k)) as *mut f64;
                let b_ptr = (hi.as_mut_ptr().add(k)) as *mut f64;
                self.apply_pair_ptr(a_ptr, b_ptr);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
pub(super) unsafe fn apply_diagonal_loop_fma(
    state: &mut [Complex64],
    target: usize,
    d0_rr: __m128d,
    d0_ii: __m128d,
    d1_rr: __m128d,
    d1_ii: __m128d,
    skip_lo: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let half = 1usize << target;
        let mask = half - 1;
        let num_pairs = state.len() >> 1;
        let base = state.as_mut_ptr() as *mut f64;

        if skip_lo {
            for k in 0..num_pairs {
                let i1 = ((k & !mask) << 1 | (k & mask)) | half;
                let p = base.add(i1 * 2);
                let s = _mm_loadu_pd(p);
                let r = complex_mul_fma(d1_rr, d1_ii, s);
                _mm_storeu_pd(p, r);
            }
        } else {
            for k in 0..num_pairs {
                let i0 = (k & !mask) << 1 | (k & mask);
                let i1 = i0 | half;

                let p0 = base.add(i0 * 2);
                let s0 = _mm_loadu_pd(p0);
                let r0 = complex_mul_fma(d0_rr, d0_ii, s0);
                _mm_storeu_pd(p0, r0);

                let p1 = base.add(i1 * 2);
                let s1 = _mm_loadu_pd(p1);
                let r1 = complex_mul_fma(d1_rr, d1_ii, s1);
                _mm_storeu_pd(p1, r1);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn apply_diagonal_loop_avx2fma(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(state.len().is_multiple_of(1usize << (target + 1)));
        let n = state.len();
        let base = state.as_mut_ptr() as *mut f64;

        if target == 0 {
            let num_pairs = n >> 1;
            if skip_lo {
                let d1_rr = _mm256_set1_pd(d1.re);
                let d1_ii = _mm256_set1_pd(d1.im);
                for i in 0..num_pairs {
                    let p = base.add(i * 4);
                    let v = _mm256_loadu_pd(p);
                    let r = complex_mul_avx2fma(d1_rr, d1_ii, v);
                    _mm256_storeu_pd(p, _mm256_blend_pd::<0b1100>(v, r));
                }
            } else {
                let d_rr = _mm256_setr_pd(d0.re, d0.re, d1.re, d1.re);
                let d_ii = _mm256_setr_pd(d0.im, d0.im, d1.im, d1.im);
                for i in 0..num_pairs {
                    let p = base.add(i * 4);
                    let v = _mm256_loadu_pd(p);
                    _mm256_storeu_pd(p, complex_mul_avx2fma(d_rr, d_ii, v));
                }
            }
            return;
        }

        let half = 1usize << target;
        let block_size = half << 1;
        let step_lanes = 2 * half;
        let d1_rr = _mm256_set1_pd(d1.re);
        let d1_ii = _mm256_set1_pd(d1.im);
        if skip_lo {
            let mut offset = 0;
            while offset < n {
                let hi_ptr = base.add((offset + half) * 2);
                for off in (0..step_lanes).step_by(4) {
                    let p = hi_ptr.add(off);
                    let v = _mm256_loadu_pd(p);
                    _mm256_storeu_pd(p, complex_mul_avx2fma(d1_rr, d1_ii, v));
                }
                offset += block_size;
            }
        } else {
            let d0_rr = _mm256_set1_pd(d0.re);
            let d0_ii = _mm256_set1_pd(d0.im);
            let mut offset = 0;
            while offset < n {
                let lo_ptr = base.add(offset * 2);
                let hi_ptr = base.add((offset + half) * 2);
                for off in (0..step_lanes).step_by(4) {
                    let pl = lo_ptr.add(off);
                    let vl = _mm256_loadu_pd(pl);
                    _mm256_storeu_pd(pl, complex_mul_avx2fma(d0_rr, d0_ii, vl));

                    let ph = hi_ptr.add(off);
                    let vh = _mm256_loadu_pd(ph);
                    _mm256_storeu_pd(ph, complex_mul_avx2fma(d1_rr, d1_ii, vh));
                }
                offset += block_size;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub(super) unsafe fn apply_diagonal_loop_avx512(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        debug_assert!(state.len().is_multiple_of(1usize << (target + 1)));
        let n = state.len();
        let base = state.as_mut_ptr() as *mut f64;

        if target == 0 {
            if skip_lo {
                let d1_rr = _mm512_set1_pd(d1.re);
                let d1_ii = _mm512_set1_pd(d1.im);
                for_each_quad!(n, |off, k| {
                    let p = base.add(off);
                    let v = load_pd(p, k);
                    let r = complex_mul_avx512(d1_rr, d1_ii, v);
                    store_pd(p, k, _mm512_mask_blend_pd(0xCC, v, r));
                });
            } else {
                let d_rr = _mm512_setr_pd(d0.re, d0.re, d1.re, d1.re, d0.re, d0.re, d1.re, d1.re);
                let d_ii = _mm512_setr_pd(d0.im, d0.im, d1.im, d1.im, d0.im, d0.im, d1.im, d1.im);
                for_each_quad!(n, |off, k| {
                    let p = base.add(off);
                    let v = load_pd(p, k);
                    store_pd(p, k, complex_mul_avx512(d_rr, d_ii, v));
                });
            }
            return;
        }

        if target == 1 {
            let lanes = 2 * n;
            if skip_lo {
                let d1_rr = _mm512_set1_pd(d1.re);
                let d1_ii = _mm512_set1_pd(d1.im);
                for off in (0..lanes).step_by(8) {
                    let p = base.add(off);
                    let v = _mm512_loadu_pd(p);
                    let r = complex_mul_avx512(d1_rr, d1_ii, v);
                    _mm512_storeu_pd(p, _mm512_mask_blend_pd(0xF0, v, r));
                }
            } else {
                let d_rr = _mm512_setr_pd(d0.re, d0.re, d0.re, d0.re, d1.re, d1.re, d1.re, d1.re);
                let d_ii = _mm512_setr_pd(d0.im, d0.im, d0.im, d0.im, d1.im, d1.im, d1.im, d1.im);
                for off in (0..lanes).step_by(8) {
                    let p = base.add(off);
                    let v = _mm512_loadu_pd(p);
                    _mm512_storeu_pd(p, complex_mul_avx512(d_rr, d_ii, v));
                }
            }
            return;
        }

        let half = 1usize << target;
        let block_size = half << 1;
        let step_lanes = 2 * half;
        let d1_rr = _mm512_set1_pd(d1.re);
        let d1_ii = _mm512_set1_pd(d1.im);
        if skip_lo {
            let mut offset = 0;
            while offset < n {
                let hi_ptr = base.add((offset + half) * 2);
                for off in (0..step_lanes).step_by(8) {
                    let p = hi_ptr.add(off);
                    let v = _mm512_loadu_pd(p);
                    _mm512_storeu_pd(p, complex_mul_avx512(d1_rr, d1_ii, v));
                }
                offset += block_size;
            }
        } else {
            let d0_rr = _mm512_set1_pd(d0.re);
            let d0_ii = _mm512_set1_pd(d0.im);
            let mut offset = 0;
            while offset < n {
                let lo_ptr = base.add(offset * 2);
                let hi_ptr = base.add((offset + half) * 2);
                for off in (0..step_lanes).step_by(8) {
                    let pl = lo_ptr.add(off);
                    let vl = _mm512_loadu_pd(pl);
                    _mm512_storeu_pd(pl, complex_mul_avx512(d0_rr, d0_ii, vl));

                    let ph = hi_ptr.add(off);
                    let vh = _mm512_loadu_pd(ph);
                    _mm512_storeu_pd(ph, complex_mul_avx512(d1_rr, d1_ii, vh));
                }
                offset += block_size;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn apply_diagonal_loop_neon(
    state: &mut [Complex64],
    target: usize,
    d0_rr: float64x2_t,
    d0_ii_as: float64x2_t,
    d1_rr: float64x2_t,
    d1_ii_as: float64x2_t,
    skip_lo: bool,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let half = 1usize << target;
        let mask = half - 1;
        let num_pairs = state.len() >> 1;
        let base = state.as_mut_ptr() as *mut f64;

        if skip_lo {
            for k in 0..num_pairs {
                let i1 = ((k & !mask) << 1 | (k & mask)) | half;
                let p = base.add(i1 * 2);
                let s = vld1q_f64(p);
                let r = complex_mul_neon(d1_rr, d1_ii_as, s);
                vst1q_f64(p, r);
            }
        } else {
            for k in 0..num_pairs {
                let i0 = (k & !mask) << 1 | (k & mask);
                let i1 = i0 | half;

                let p0 = base.add(i0 * 2);
                let s0 = vld1q_f64(p0);
                let r0 = complex_mul_neon(d0_rr, d0_ii_as, s0);
                vst1q_f64(p0, r0);

                let p1 = base.add(i1 * 2);
                let s1 = vld1q_f64(p1);
                let r1 = complex_mul_neon(d1_rr, d1_ii_as, s1);
                vst1q_f64(p1, r1);
            }
        }
    }
}

/// Evolve `rho -> U rho U^dagger` for a one-qubit `U` on `qubit` of the `4^n` density
/// matrix buffer `rho` (row bit at `qubit + n`, column bit at `qubit`) in one pass.
///
/// Each 2x2 `(row-bit, col-bit)` block is loaded once, takes `U` on the left and
/// `U^dagger` on the right in registers, and is stored once: the same 16 complex
/// multiplies per block as two register passes, with half the loads and stores.
#[inline(always)]
pub(crate) fn apply_1q_sandwich_blocked(
    rho: &mut [Complex64],
    n: usize,
    qubit: usize,
    mat: &[[Complex64; 2]; 2],
) {
    debug_assert_eq!(rho.len(), 1usize << (2 * n));
    debug_assert!(qubit < n);
    let rbit = 1usize << (qubit + n);
    let cbit = 1usize << qubit;
    let conj = [
        [mat[0][0].conj(), mat[0][1].conj()],
        [mat[1][0].conj(), mat[1][1].conj()],
    ];

    #[cfg(target_arch = "x86_64")]
    if matches!(simd_tier(), SimdTier::Avx512 | SimdTier::Avx2Fma) {
        // SAFETY: AVX2 detected, and every block index the loops form lies
        // below `rho.len()` because `rbit` and `cbit` are bits of it.
        unsafe {
            let u = MatBroadcast256::from_matrix(mat);
            if qubit == 0 {
                sandwich_1q_blocked_q0_avx2fma(rho, rbit, &u, &conj);
            } else {
                let v = MatBroadcast256::from_matrix(&conj);
                sandwich_1q_blocked_avx2fma(rho, rbit, cbit, &u, &v);
            }
        }
        return;
    }

    sandwich_1q_blocked_scalar(rho, rbit, cbit, mat, &conj);
}

#[inline(always)]
pub(super) fn sandwich_1q_blocked_scalar(
    rho: &mut [Complex64],
    rbit: usize,
    cbit: usize,
    u: &[[Complex64; 2]; 2],
    v: &[[Complex64; 2]; 2],
) {
    for row in (0..rho.len()).step_by(rbit << 1) {
        for col in (row..row + rbit).step_by(cbit << 1) {
            for i in col..col + cbit {
                let a = rho[i];
                let b = rho[i + cbit];
                let c = rho[i + rbit];
                let d = rho[i + rbit + cbit];
                let t00 = u[0][0] * a + u[0][1] * c;
                let t10 = u[1][0] * a + u[1][1] * c;
                let t01 = u[0][0] * b + u[0][1] * d;
                let t11 = u[1][0] * b + u[1][1] * d;
                rho[i] = v[0][0] * t00 + v[0][1] * t01;
                rho[i + cbit] = v[1][0] * t00 + v[1][1] * t01;
                rho[i + rbit] = v[0][0] * t10 + v[0][1] * t11;
                rho[i + rbit + cbit] = v[1][0] * t10 + v[1][1] * t11;
            }
        }
    }
}

/// Two blocks per vector: requires `cbit >= 2`, so `i` and `i + 1` are both block bases.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sandwich_1q_blocked_avx2fma(
    rho: &mut [Complex64],
    rbit: usize,
    cbit: usize,
    u: &MatBroadcast256,
    v: &MatBroadcast256,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let len = rho.len();
        let p = rho.as_mut_ptr() as *mut f64;
        let mut row = 0;
        while row < len {
            let mut col = row;
            while col < row + rbit {
                let mut i = col;
                while i < col + cbit {
                    let pa = p.add(2 * i);
                    let pb = p.add(2 * (i + cbit));
                    let pc = p.add(2 * (i + rbit));
                    let pd = p.add(2 * (i + rbit + cbit));
                    let a = _mm256_loadu_pd(pa);
                    let b = _mm256_loadu_pd(pb);
                    let c = _mm256_loadu_pd(pc);
                    let d = _mm256_loadu_pd(pd);
                    let t00 = _mm256_add_pd(
                        complex_mul_avx2fma(u.m00_rr, u.m00_ii, a),
                        complex_mul_avx2fma(u.m01_rr, u.m01_ii, c),
                    );
                    let t10 = _mm256_add_pd(
                        complex_mul_avx2fma(u.m10_rr, u.m10_ii, a),
                        complex_mul_avx2fma(u.m11_rr, u.m11_ii, c),
                    );
                    let t01 = _mm256_add_pd(
                        complex_mul_avx2fma(u.m00_rr, u.m00_ii, b),
                        complex_mul_avx2fma(u.m01_rr, u.m01_ii, d),
                    );
                    let t11 = _mm256_add_pd(
                        complex_mul_avx2fma(u.m10_rr, u.m10_ii, b),
                        complex_mul_avx2fma(u.m11_rr, u.m11_ii, d),
                    );
                    _mm256_storeu_pd(
                        pa,
                        _mm256_add_pd(
                            complex_mul_avx2fma(v.m00_rr, v.m00_ii, t00),
                            complex_mul_avx2fma(v.m01_rr, v.m01_ii, t01),
                        ),
                    );
                    _mm256_storeu_pd(
                        pb,
                        _mm256_add_pd(
                            complex_mul_avx2fma(v.m10_rr, v.m10_ii, t00),
                            complex_mul_avx2fma(v.m11_rr, v.m11_ii, t01),
                        ),
                    );
                    _mm256_storeu_pd(
                        pc,
                        _mm256_add_pd(
                            complex_mul_avx2fma(v.m00_rr, v.m00_ii, t10),
                            complex_mul_avx2fma(v.m01_rr, v.m01_ii, t11),
                        ),
                    );
                    _mm256_storeu_pd(
                        pd,
                        _mm256_add_pd(
                            complex_mul_avx2fma(v.m10_rr, v.m10_ii, t10),
                            complex_mul_avx2fma(v.m11_rr, v.m11_ii, t11),
                        ),
                    );
                    i += 2;
                }
                col += cbit << 1;
            }
            row += rbit << 1;
        }
    }
}

/// Qubit 0: a block's row halves `[a, b]` and `[c, d]` are each one vector, so the
/// left product runs across the pair and the right product within each vector
/// against its own half swap, with lane-varying `conj(U)` coefficients.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sandwich_1q_blocked_q0_avx2fma(
    rho: &mut [Complex64],
    rbit: usize,
    u: &MatBroadcast256,
    v: &[[Complex64; 2]; 2],
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let diag_rr = _mm256_setr_pd(v[0][0].re, v[0][0].re, v[1][1].re, v[1][1].re);
        let diag_ii = _mm256_setr_pd(v[0][0].im, v[0][0].im, v[1][1].im, v[1][1].im);
        let off_rr = _mm256_setr_pd(v[0][1].re, v[0][1].re, v[1][0].re, v[1][0].re);
        let off_ii = _mm256_setr_pd(v[0][1].im, v[0][1].im, v[1][0].im, v[1][0].im);
        let len = rho.len();
        let p = rho.as_mut_ptr() as *mut f64;
        let mut row = 0;
        while row < len {
            let mut i = row;
            while i < row + rbit {
                let px = p.add(2 * i);
                let py = p.add(2 * (i + rbit));
                let x = _mm256_loadu_pd(px);
                let y = _mm256_loadu_pd(py);
                let tx = _mm256_add_pd(
                    complex_mul_avx2fma(u.m00_rr, u.m00_ii, x),
                    complex_mul_avx2fma(u.m01_rr, u.m01_ii, y),
                );
                let ty = _mm256_add_pd(
                    complex_mul_avx2fma(u.m10_rr, u.m10_ii, x),
                    complex_mul_avx2fma(u.m11_rr, u.m11_ii, y),
                );
                let sx = _mm256_permute2f128_pd(tx, tx, 0x01);
                let sy = _mm256_permute2f128_pd(ty, ty, 0x01);
                _mm256_storeu_pd(
                    px,
                    _mm256_add_pd(
                        complex_mul_avx2fma(diag_rr, diag_ii, tx),
                        complex_mul_avx2fma(off_rr, off_ii, sx),
                    ),
                );
                _mm256_storeu_pd(
                    py,
                    _mm256_add_pd(
                        complex_mul_avx2fma(diag_rr, diag_ii, ty),
                        complex_mul_avx2fma(off_rr, off_ii, sy),
                    ),
                );
                i += 2;
            }
            row += rbit << 1;
        }
    }
}

/// Apply a diagonal single-qubit gate using a tight SIMD loop.
///
/// For gates where `d0 ≈ 1` (Z, S, T, P), set `skip_lo = true` to halve
/// memory traffic. Processes block-strided slices on AVX2/AVX-512 and pairs
/// on 128-bit tiers with 1 complex multiply per element instead of 2 + add.
pub(crate) fn apply_diagonal_sequential(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    #[cfg(target_arch = "x86_64")]
    match simd_tier() {
        SimdTier::Avx512 => {
            if target == 1 && skip_lo {
                // SAFETY: AVX2 detected, FMA is implied by SimdTier::Avx512, and
                // state.len() is a multiple of 2 << target.
                unsafe { apply_diagonal_loop_avx2fma(state, target, d0, d1, true) };
            } else {
                // SAFETY: AVX-512F detected, and state.len() is a multiple of 2 << target.
                unsafe { apply_diagonal_loop_avx512(state, target, d0, d1, skip_lo) };
            }
        }
        SimdTier::Avx2Fma => {
            // SAFETY: AVX2 detected, FMA is implied by SimdTier::Avx2Fma, and
            // state.len() is a multiple of 2 << target.
            unsafe { apply_diagonal_loop_avx2fma(state, target, d0, d1, skip_lo) };
        }
        SimdTier::Fma => {
            // SAFETY: FMA detected, and state.len() is a multiple of 2 << target.
            unsafe {
                let d0_rr = _mm_set1_pd(d0.re);
                let d0_ii = _mm_set1_pd(d0.im);
                let d1_rr = _mm_set1_pd(d1.re);
                let d1_ii = _mm_set1_pd(d1.im);
                apply_diagonal_loop_fma(state, target, d0_rr, d0_ii, d1_rr, d1_ii, skip_lo);
            }
        }
        SimdTier::Sse2 => apply_diagonal_scalar(state, target, d0, d1, skip_lo),
    }

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is baseline on aarch64. State indexing
    // is handled by the diagonal loop using validated target positions.
    unsafe {
        let d0_rr = vdupq_n_f64(d0.re);
        let d0_ii_as = vcombine_f64(vdup_n_f64(-d0.im), vdup_n_f64(d0.im));
        let d1_rr = vdupq_n_f64(d1.re);
        let d1_ii_as = vcombine_f64(vdup_n_f64(-d1.im), vdup_n_f64(d1.im));
        apply_diagonal_loop_neon(state, target, d0_rr, d0_ii_as, d1_rr, d1_ii_as, skip_lo);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    apply_diagonal_scalar(state, target, d0, d1, skip_lo);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
pub(super) fn apply_diagonal_scalar(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    let half = 1usize << target;
    let mask = half - 1;
    let num_pairs = state.len() >> 1;
    if skip_lo {
        for k in 0..num_pairs {
            let i1 = ((k & !mask) << 1 | (k & mask)) | half;
            state[i1] *= d1;
        }
    } else {
        for k in 0..num_pairs {
            let i0 = (k & !mask) << 1 | (k & mask);
            let i1 = i0 | half;
            state[i0] *= d0;
            state[i1] *= d1;
        }
    }
}
