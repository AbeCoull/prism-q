//! SIMD-accelerated kernels for single-qubit gate application and bulk
//! operations (negate, swap, norm-squared accumulation, zero, scale).
//!
//! Dispatches to the best available SIMD tier at runtime:
//!
//! | Tier | Requirement | Complex multiply cost |
//! |----------|------------------|---------------------------------|
//! | AVX2+FMA | `avx2` + `fma`   | 256-bit: 1 mul + 1 fmaddsub (2 pairs) |
//! | FMA      | `fma`            | 128-bit: 1 mul + 1 fmaddsub |
//! | NEON     | aarch64 baseline | 128-bit: 1 mul + 1 fma (pre-negated) |
//! | SSE2     | x86_64 baseline  | 128-bit: 2 mul + shuffle + xor + add |
//! | Scalar   | fallback         | Standard Complex64 ops |
//!
//! Call [`PreparedGate1q::new`] once per gate, then [`PreparedGate1q::apply`]
//! per chunk. This hoists the matrix broadcast and CPU feature detection out
//! of the inner loop.

use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn complex_mul_neon(c_rr: float64x2_t, c_ii_as: float64x2_t, z: float64x2_t) -> float64x2_t {
    let z_swap = vextq_f64(z, z, 1);
    let prod = vmulq_f64(c_rr, z);
    vfmaq_f64(prod, c_ii_as, z_swap)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn complex_mul_sse2(
    c_rr: __m128d,
    c_ii: __m128d,
    z: __m128d,
    sign_mask: __m128d,
) -> __m128d {
    let z_swap = _mm_shuffle_pd(z, z, 0b01);
    let t1 = _mm_mul_pd(c_rr, z);
    let t2 = _mm_mul_pd(c_ii, z_swap);
    let t2_neg = _mm_xor_pd(t2, sign_mask);
    _mm_add_pd(t1, t2_neg)
}

#[cfg(all(target_arch = "x86_64", any(feature = "parallel", test)))]
#[target_feature(enable = "sse2")]
unsafe fn apply_slices_sse2(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn complex_mul_fma(c_rr: __m128d, c_ii: __m128d, z: __m128d) -> __m128d {
    let z_swap = _mm_shuffle_pd(z, z, 0b01);
    let t = _mm_mul_pd(c_ii, z_swap);
    _mm_fmaddsub_pd(c_rr, z, t)
}

#[cfg(all(target_arch = "x86_64", any(feature = "parallel", test)))]
#[target_feature(enable = "fma")]
unsafe fn apply_slices_fma(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
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

#[cfg(all(target_arch = "aarch64", any(feature = "parallel", test)))]
unsafe fn apply_slices_neon(lo: &mut [Complex64], hi: &mut [Complex64], mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn complex_mul_avx2fma(c_rr: __m256d, c_ii: __m256d, z: __m256d) -> __m256d {
    let z_swap = _mm256_permute_pd(z, 0b0101);
    let t = _mm256_mul_pd(c_ii, z_swap);
    _mm256_fmaddsub_pd(c_rr, z, t)
}

#[cfg(all(target_arch = "x86_64", any(feature = "parallel", test)))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_slices_avx2fma(
    lo: &mut [Complex64],
    hi: &mut [Complex64],
    mat: &MatBroadcast256,
    mat128: &MatBroadcast,
) {
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

    if n % 2 != 0 {
        let off = pairs * 4;
        apply_pair_fma(lo_ptr.add(off), hi_ptr.add(off), mat128);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn apply_full_loop_sse2(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn apply_full_loop_fma(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_full_loop_avx2fma_inline(
    state: &mut [Complex64],
    target: usize,
    mat: &MatBroadcast256,
) {
    // SAFETY: caller guarantees target >= 2 (half >= 4, avx_pairs >= 2).
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

#[cfg(target_arch = "aarch64")]
unsafe fn apply_full_loop_neon(state: &mut [Complex64], target: usize, mat: &MatBroadcast) {
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_pair_neon(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_pair_sse2(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
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

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn apply_pair_fma(a_ptr: *mut f64, b_ptr: *mut f64, mat: &MatBroadcast) {
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

#[inline(always)]
#[allow(dead_code)]
fn apply_slices_scalar(lo: &mut [Complex64], hi: &mut [Complex64], mat: &[[Complex64; 2]; 2]) {
    debug_assert_eq!(lo.len(), hi.len());
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        let v0 = *a;
        let v1 = *b;
        *a = mat[0][0] * v0 + mat[0][1] * v1;
        *b = mat[1][0] * v0 + mat[1][1] * v1;
    }
}

#[cfg(target_arch = "x86_64")]
enum SimdTier {
    Avx2Fma,
    Fma,
    Sse2,
}

/// Bench-only kill switch for the AVX2 paired-group 2q kernel.
///
/// Reads `PRISM_NO_AVX2_2Q` once and caches the result. Set the variable
/// to disable the kernel and exercise the 128-bit FMA fallback for A/B
/// timing comparison without rebuilding.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_2q_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PRISM_NO_AVX2_2Q").is_none())
}

/// Precomputed single-qubit gate ready for repeated application.
///
/// Create once per gate via [`PreparedGate1q::new`], then call [`apply`]
/// per chunk. This avoids re-broadcasting the matrix and re-checking CPU
/// features on every chunk.
pub(crate) struct PreparedGate1q {
    #[cfg(target_arch = "x86_64")]
    broadcast: MatBroadcast,
    #[cfg(target_arch = "x86_64")]
    broadcast256: Option<MatBroadcast256>,
    #[cfg(target_arch = "x86_64")]
    tier: SimdTier,
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
            let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            let tier = if has_avx2_fma {
                SimdTier::Avx2Fma
            } else if is_x86_feature_detected!("fma") {
                SimdTier::Fma
            } else {
                SimdTier::Sse2
            };
            let broadcast256 = if has_avx2_fma {
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

    #[cfg(any(feature = "parallel", test))]
    #[inline(always)]
    pub(crate) fn apply(&self, lo: &mut [Complex64], hi: &mut [Complex64]) {
        debug_assert_eq!(lo.len(), hi.len());

        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match self.tier {
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
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match self.tier {
                    SimdTier::Avx2Fma => {
                        const MAX_AVX2_STATE: usize = 8192;
                        const MIN_AVX2_TARGET: usize = 2;
                        if target >= MIN_AVX2_TARGET && state.len() <= MAX_AVX2_STATE {
                            let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                            apply_full_loop_avx2fma_inline(state, target, b256);
                        } else {
                            apply_full_loop_fma(state, target, &self.broadcast);
                        }
                    }
                    SimdTier::Fma => apply_full_loop_fma(state, target, &self.broadcast),
                    SimdTier::Sse2 => apply_full_loop_sse2(state, target, &self.broadcast),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
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

    /// Apply the gate within a cache-resident tile, preferring AVX2 when available.
    ///
    /// Unlike `apply_full_sequential`, does not restrict AVX2 by state size.
    /// Use when the slice is known to be L2/L3-resident (e.g. MultiFused tiles).
    #[inline(always)]
    pub(crate) fn apply_tiled(&self, state: &mut [Complex64], target: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe {
                match self.tier {
                    SimdTier::Avx2Fma => {
                        if target >= 2 {
                            let b256 = self.broadcast256.as_ref().unwrap_unchecked();
                            apply_full_loop_avx2fma_inline(state, target, b256);
                        } else {
                            apply_full_loop_fma(state, target, &self.broadcast);
                        }
                    }
                    SimdTier::Fma => apply_full_loop_fma(state, target, &self.broadcast),
                    SimdTier::Sse2 => apply_full_loop_sse2(state, target, &self.broadcast),
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            unsafe { apply_full_loop_neon(state, target, &self.broadcast) };
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.apply_full_sequential(state, target);
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
        #[cfg(target_arch = "x86_64")]
        {
            match self.tier {
                SimdTier::Avx2Fma | SimdTier::Fma => apply_pair_fma(a_ptr, b_ptr, &self.broadcast),
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

    /// Apply the gate to equal-length lo/hi slice pairs.
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
unsafe fn apply_diagonal_loop_fma(
    state: &mut [Complex64],
    target: usize,
    d0_rr: __m128d,
    d0_ii: __m128d,
    d1_rr: __m128d,
    d1_ii: __m128d,
    skip_lo: bool,
) {
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

/// Apply a diagonal single-qubit gate using a tight SIMD loop.
///
/// For gates where `d0 ≈ 1` (Z, S, T, P), set `skip_lo = true` to halve
/// memory traffic. Uses the same mask-based pair iteration as the full 2×2
/// kernel but with 1 complex multiply per element instead of 2 + add.
pub(crate) fn apply_diagonal_sequential(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if has_fma() {
            unsafe {
                let d0_rr = _mm_set1_pd(d0.re);
                let d0_ii = _mm_set1_pd(d0.im);
                let d1_rr = _mm_set1_pd(d1.re);
                let d1_ii = _mm_set1_pd(d1.im);
                apply_diagonal_loop_fma(state, target, d0_rr, d0_ii, d1_rr, d1_ii, skip_lo);
            }
        } else {
            apply_diagonal_scalar(state, target, d0, d1, skip_lo);
        }
    }

    #[cfg(target_arch = "aarch64")]
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
fn apply_diagonal_scalar(
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

#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx2_fma() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn has_fma() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("fma"))
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn has_bmi2() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("bmi2"))
}

#[cfg(all(target_arch = "x86_64", any(feature = "parallel", test)))]
#[target_feature(enable = "avx2")]
unsafe fn negate_slice_avx2(slice: &mut [Complex64]) {
    let sign = _mm256_set1_pd(-0.0_f64);
    let ptr = slice.as_mut_ptr() as *mut f64;
    let pairs = slice.len() / 2;
    for i in 0..pairs {
        let off = i * 4;
        let v = _mm256_loadu_pd(ptr.add(off));
        _mm256_storeu_pd(ptr.add(off), _mm256_xor_pd(v, sign));
    }
    if slice.len() % 2 != 0 {
        let last = &mut slice[slice.len() - 1];
        *last = -*last;
    }
}

const MIN_SIMD_SLICE: usize = 4;

#[cfg(any(feature = "parallel", test))]
pub(crate) fn negate_slice(slice: &mut [Complex64]) {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        unsafe { negate_slice_avx2(slice) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        unsafe {
            let ptr = slice.as_mut_ptr() as *mut f64;
            for i in 0..slice.len() {
                let p = ptr.add(i * 2);
                let v = vld1q_f64(p);
                vst1q_f64(p, vnegq_f64(v));
            }
        }
        return;
    }
    for amp in slice.iter_mut() {
        *amp = -*amp;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn swap_slices_avx2(a: &mut [Complex64], b: &mut [Complex64]) {
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
    if n % 2 != 0 {
        let last = n - 1;
        std::mem::swap(&mut a[last], &mut b[last]);
    }
}

pub(crate) fn swap_slices(a: &mut [Complex64], b: &mut [Complex64]) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if a.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        unsafe { swap_slices_avx2(a, b) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if a.len() >= MIN_SIMD_SLICE {
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
        return;
    }
    for (x, y) in a.iter_mut().zip(b.iter_mut()) {
        std::mem::swap(x, y);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_sqr_sum_avx2fma(slice: &[Complex64]) -> f64 {
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
    if slice.len() % 2 != 0 {
        result += slice[slice.len() - 1].norm_sqr();
    }
    result
}

pub(crate) fn norm_sqr_sum(slice: &[Complex64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        return unsafe { norm_sqr_sum_avx2fma(slice) };
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
        return unsafe { norm_sqr_sum_neon(slice) };
    }
    slice.iter().map(|c| c.norm_sqr()).sum()
}

#[cfg(target_arch = "aarch64")]
unsafe fn norm_sqr_sum_neon(slice: &[Complex64]) -> f64 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn norm_sqr_to_slice_avx2(src: &[Complex64], dst: &mut [f64]) {
    debug_assert!(dst.len() >= src.len());
    let inp = src.as_ptr() as *const f64;
    let out = dst.as_mut_ptr();
    let quads = src.len() / 4;

    for i in 0..quads {
        let base_in = i * 8;
        let base_out = i * 4;
        let v0 = _mm256_loadu_pd(inp.add(base_in));
        let v1 = _mm256_loadu_pd(inp.add(base_in + 4));
        let sq0 = _mm256_mul_pd(v0, v0);
        let sq1 = _mm256_mul_pd(v1, v1);
        // hadd: lane0=[sq0[0]+sq0[1], sq1[0]+sq1[1]], lane1=[sq0[2]+sq0[3], sq1[2]+sq1[3]]
        //      = [norm0, norm2, norm1, norm3]
        let h = _mm256_hadd_pd(sq0, sq1);
        // permute to [norm0, norm1, norm2, norm3]
        let ordered = _mm256_permute4x64_pd(h, 0b11_01_10_00);
        _mm256_storeu_pd(out.add(base_out), ordered);
    }

    let tail = quads * 4;
    for (j, c) in src[tail..].iter().enumerate() {
        *out.add(tail + j) = c.norm_sqr();
    }
}

pub(crate) fn norm_sqr_to_slice(src: &[Complex64], dst: &mut [f64]) {
    debug_assert!(dst.len() >= src.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        unsafe { norm_sqr_to_slice_avx2(src, dst) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if src.len() >= MIN_SIMD_SLICE {
        unsafe {
            let inp = src.as_ptr() as *const f64;
            let out = dst.as_mut_ptr();
            let pairs = src.len() / 2;
            for i in 0..pairs {
                let base = i * 4;
                let v0 = vld1q_f64(inp.add(base));
                let v1 = vld1q_f64(inp.add(base + 2));
                let sq0 = vmulq_f64(v0, v0);
                let sq1 = vmulq_f64(v1, v1);
                let ns = vpaddq_f64(sq0, sq1);
                vst1q_f64(out.add(i * 2), ns);
            }
            if src.len() % 2 != 0 {
                let last = src.len() - 1;
                *out.add(last) = src[last].norm_sqr();
            }
        }
        return;
    }
    for (i, c) in src.iter().enumerate() {
        dst[i] = c.norm_sqr();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn norm_sqr_to_slice_scaled_avx2(src: &[Complex64], dst: &mut [f64], scale: f64) {
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
        _mm256_storeu_pd(out.add(base_out), _mm256_mul_pd(ordered, s));
    }

    let tail = quads * 4;
    for (j, c) in src[tail..].iter().enumerate() {
        *out.add(tail + j) = c.norm_sqr() * scale;
    }
}

pub(crate) fn norm_sqr_to_slice_scaled(src: &[Complex64], dst: &mut [f64], scale: f64) {
    debug_assert!(dst.len() >= src.len());
    #[cfg(target_arch = "x86_64")]
    if src.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        unsafe { norm_sqr_to_slice_scaled_avx2(src, dst, scale) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if src.len() >= MIN_SIMD_SLICE {
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
                let ns = vmulq_f64(vpaddq_f64(sq0, sq1), s);
                vst1q_f64(out.add(i * 2), ns);
            }
            if src.len() % 2 != 0 {
                let last = src.len() - 1;
                *out.add(last) = src[last].norm_sqr() * scale;
            }
        }
        return;
    }
    for (i, c) in src.iter().enumerate() {
        dst[i] = c.norm_sqr() * scale;
    }
}

#[cfg(all(target_arch = "x86_64", any(feature = "parallel", test)))]
#[target_feature(enable = "avx2")]
unsafe fn zero_slice_avx2(slice: &mut [Complex64]) {
    let z = _mm256_setzero_pd();
    let ptr = slice.as_mut_ptr() as *mut f64;
    let pairs = slice.len() / 2;
    for i in 0..pairs {
        _mm256_storeu_pd(ptr.add(i * 4), z);
    }
    if slice.len() % 2 != 0 {
        slice[slice.len() - 1] = Complex64::new(0.0, 0.0);
    }
}

#[cfg(any(feature = "parallel", test))]
pub(crate) fn zero_slice(slice: &mut [Complex64]) {
    #[cfg(target_arch = "x86_64")]
    if slice.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
        unsafe { zero_slice_avx2(slice) };
        return;
    }
    let zero = Complex64::new(0.0, 0.0);
    for amp in slice.iter_mut() {
        *amp = zero;
    }
}

#[cfg(all(target_arch = "x86_64", test))]
#[target_feature(enable = "avx2")]
unsafe fn scale_slice_avx2(slice: &mut [Complex64], factor: f64) {
    let f = _mm256_set1_pd(factor);
    let ptr = slice.as_mut_ptr() as *mut f64;
    let pairs = slice.len() / 2;
    for i in 0..pairs {
        let off = i * 4;
        let v = _mm256_loadu_pd(ptr.add(off));
        _mm256_storeu_pd(ptr.add(off), _mm256_mul_pd(v, f));
    }
    if slice.len() % 2 != 0 {
        slice[slice.len() - 1] *= factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_complex_slice_avx2fma(slice: &mut [Complex64], factor: Complex64) {
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
    if slice.len() % 2 != 0 {
        slice[slice.len() - 1] *= factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn scale_complex_slice_fma(slice: &mut [Complex64], factor: Complex64) {
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

pub(crate) fn scale_complex_slice(slice: &mut [Complex64], factor: Complex64) {
    #[cfg(target_arch = "x86_64")]
    {
        if slice.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
            unsafe { scale_complex_slice_avx2fma(slice, factor) };
            return;
        }
        if slice.len() >= 2 && has_fma() {
            unsafe { scale_complex_slice_fma(slice, factor) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if slice.len() >= MIN_SIMD_SLICE {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scale_complex_to_slice_avx2fma(
    dst: &mut [Complex64],
    src: &[Complex64],
    factor: Complex64,
) {
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
    if src.len() % 2 != 0 {
        let last = src.len() - 1;
        dst[last] = src[last] * factor;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn scale_complex_to_slice_fma(dst: &mut [Complex64], src: &[Complex64], factor: Complex64) {
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

/// Out-of-place complex scaling: `dst[i] = src[i] * factor`.
pub(crate) fn scale_complex_to_slice(dst: &mut [Complex64], src: &[Complex64], factor: Complex64) {
    assert!(
        dst.len() >= src.len(),
        "destination slice shorter than source slice"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if src.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
            // SAFETY: AVX2 and FMA support are checked above. The assert keeps
            // every load and store in bounds, Rust references provide valid
            // slices, and SIMD avoids the measured scalar bottleneck in large
            // factored merge copies.
            unsafe { scale_complex_to_slice_avx2fma(dst, src, factor) };
            return;
        }
        if src.len() >= 2 && has_fma() {
            // SAFETY: FMA support is checked above. The assert keeps every
            // load and store in bounds, Rust references provide valid slices,
            // and SIMD avoids the measured scalar bottleneck in factored merge
            // copies.
            unsafe { scale_complex_to_slice_fma(dst, src, factor) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if src.len() >= MIN_SIMD_SLICE {
        // SAFETY: NEON is available on supported aarch64 targets. The assert
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

#[cfg(test)]
fn scale_slice(slice: &mut [Complex64], factor: f64) {
    #[cfg(target_arch = "x86_64")]
    {
        if slice.len() >= MIN_SIMD_SLICE && has_avx2_fma() {
            unsafe { scale_slice_avx2(slice, factor) };
        } else {
            for amp in slice.iter_mut() {
                *amp *= factor;
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let f = vdupq_n_f64(factor);
        let ptr = slice.as_mut_ptr() as *mut f64;
        for i in 0..slice.len() {
            let p = ptr.add(i * 2);
            vst1q_f64(p, vmulq_f64(vld1q_f64(p), f));
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for amp in slice.iter_mut() {
        *amp *= factor;
    }
}

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
    #[inline(always)]
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
    let s0 = _mm_loadu_pd(state.add(i[0] * 2));
    let s1 = _mm_loadu_pd(state.add(i[1] * 2));
    let s2 = _mm_loadu_pd(state.add(i[2] * 2));
    let s3 = _mm_loadu_pd(state.add(i[3] * 2));

    let sf0 = _mm_shuffle_pd(s0, s0, 0b01);
    let sf1 = _mm_shuffle_pd(s1, s1, 0b01);
    let sf2 = _mm_shuffle_pd(s2, s2, 0b01);
    let sf3 = _mm_shuffle_pd(s3, s3, 0b01);

    macro_rules! row {
        ($r:expr) => {{
            let off = $r * 4;
            let t = _mm_mul_pd(mat.ii[off], sf0);
            let mut acc = _mm_fmaddsub_pd(mat.rr[off], s0, t);
            let t = _mm_mul_pd(mat.ii[off + 1], sf1);
            acc = _mm_add_pd(acc, _mm_fmaddsub_pd(mat.rr[off + 1], s1, t));
            let t = _mm_mul_pd(mat.ii[off + 2], sf2);
            acc = _mm_add_pd(acc, _mm_fmaddsub_pd(mat.rr[off + 2], s2, t));
            let t = _mm_mul_pd(mat.ii[off + 3], sf3);
            acc = _mm_add_pd(acc, _mm_fmaddsub_pd(mat.rr[off + 3], s3, t));
            _mm_storeu_pd(state.add(i[$r] * 2), acc);
        }};
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
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
    use crate::backend::statevector::insert_zero_bit;

    for k in 0..n_iter {
        let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
        let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
        apply_fused_2q_group_fma_inner(state, i, mat);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
unsafe fn apply_fused_2q_group_fma(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    apply_fused_2q_group_fma_inner(state, i, mat);
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
    let s0 = _mm256_loadu_pd(state.add(i[0] * 2));
    let s1 = _mm256_loadu_pd(state.add(i[1] * 2));
    let s2 = _mm256_loadu_pd(state.add(i[2] * 2));
    let s3 = _mm256_loadu_pd(state.add(i[3] * 2));

    let sf0 = _mm256_shuffle_pd(s0, s0, 0b0101);
    let sf1 = _mm256_shuffle_pd(s1, s1, 0b0101);
    let sf2 = _mm256_shuffle_pd(s2, s2, 0b0101);
    let sf3 = _mm256_shuffle_pd(s3, s3, 0b0101);

    macro_rules! row {
        ($r:expr) => {{
            let off = $r * 4;
            let t = _mm256_mul_pd(mat.ii[off], sf0);
            let mut acc = _mm256_fmaddsub_pd(mat.rr[off], s0, t);
            let t = _mm256_mul_pd(mat.ii[off + 1], sf1);
            acc = _mm256_add_pd(acc, _mm256_fmaddsub_pd(mat.rr[off + 1], s1, t));
            let t = _mm256_mul_pd(mat.ii[off + 2], sf2);
            acc = _mm256_add_pd(acc, _mm256_fmaddsub_pd(mat.rr[off + 2], s2, t));
            let t = _mm256_mul_pd(mat.ii[off + 3], sf3);
            acc = _mm256_add_pd(acc, _mm256_fmaddsub_pd(mat.rr[off + 3], s3, t));
            _mm256_storeu_pd(state.add(i[$r] * 2), acc);
        }};
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
}

/// Pair-batched 2q kernel main loop. Requires `lo > 0` so that paired k/k+1
/// values map to adjacent state positions; falls back to the 128-bit FMA
/// kernel when `lo == 0`.
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
    use crate::backend::statevector::insert_zero_bit;

    if lo == 0 {
        for k in 0..n_iter {
            let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
            let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
            apply_fused_2q_group_fma_inner(state, i, mat128);
        }
        return;
    }

    let pairs = n_iter / 2;
    for pk in 0..pairs {
        let k = pk * 2;
        let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
        let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
        apply_fused_2q_pair_avx2_inner(state, i, mat256);
    }
    if n_iter & 1 == 1 {
        let k = n_iter - 1;
        let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
        let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
        apply_fused_2q_group_fma_inner(state, i, mat128);
    }
}

/// Apply one 4-element group of a 2q gate using SSE2 intrinsics.
///
/// Uses the 5-op complex multiply (shuffle + 2×mul + xor + add) since SSE2
/// is always available on x86_64 — safe to call from Rayon closures without
/// `#[target_feature]`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn apply_fused_2q_group_sse2(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
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

#[cfg(target_arch = "aarch64")]
struct Mat4x4Broadcast {
    rr: [float64x2_t; 16],
    ii_as: [float64x2_t; 16],
}

#[cfg(target_arch = "aarch64")]
impl Mat4x4Broadcast {
    #[inline(always)]
    fn from_matrix(mat: &[[Complex64; 4]; 4]) -> Self {
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
unsafe fn complex_mul_neon_preswapped(
    c_rr: float64x2_t,
    c_ii_as: float64x2_t,
    z: float64x2_t,
    z_swap: float64x2_t,
) -> float64x2_t {
    let prod = vmulq_f64(c_rr, z);
    vfmaq_f64(prod, c_ii_as, z_swap)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_fused_2q_group_neon(state: *mut f64, i: [usize; 4], mat: &Mat4x4Broadcast) {
    let s0 = vld1q_f64(state.add(i[0] * 2));
    let s1 = vld1q_f64(state.add(i[1] * 2));
    let s2 = vld1q_f64(state.add(i[2] * 2));
    let s3 = vld1q_f64(state.add(i[3] * 2));

    let sf0 = vextq_f64(s0, s0, 1);
    let sf1 = vextq_f64(s1, s1, 1);
    let sf2 = vextq_f64(s2, s2, 1);
    let sf3 = vextq_f64(s3, s3, 1);

    macro_rules! row {
        ($r:expr) => {{
            let off = $r * 4;
            let mut acc = complex_mul_neon_preswapped(mat.rr[off], mat.ii_as[off], s0, sf0);
            acc = vaddq_f64(
                acc,
                complex_mul_neon_preswapped(mat.rr[off + 1], mat.ii_as[off + 1], s1, sf1),
            );
            acc = vaddq_f64(
                acc,
                complex_mul_neon_preswapped(mat.rr[off + 2], mat.ii_as[off + 2], s2, sf2),
            );
            acc = vaddq_f64(
                acc,
                complex_mul_neon_preswapped(mat.rr[off + 3], mat.ii_as[off + 3], s3, sf3),
            );
            vst1q_f64(state.add(i[$r] * 2), acc);
        }};
    }
    row!(0);
    row!(1);
    row!(2);
    row!(3);
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
    use crate::backend::statevector::insert_zero_bit;

    for k in 0..n_iter {
        let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
        let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
        apply_fused_2q_group_neon(state, i, mat);
    }
}

/// Precomputed two-qubit gate ready for repeated application via SIMD.
///
/// Created once per gate, then applied to each 4-element group of the
/// statevector. On x86_64 and aarch64, stores the 4×4 matrix in broadcast
/// form for SIMD kernels. On other platforms, stores the raw matrix.
pub(crate) struct PreparedGate2q {
    #[cfg(target_arch = "x86_64")]
    broadcast: Mat4x4Broadcast,
    #[cfg(target_arch = "x86_64")]
    broadcast256: Option<Mat4x4Broadcast256>,
    #[cfg(target_arch = "x86_64")]
    tier: SimdTier,
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
            let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            let tier = if has_avx2_fma {
                SimdTier::Avx2Fma
            } else if is_x86_feature_detected!("fma") {
                SimdTier::Fma
            } else {
                SimdTier::Sse2
            };
            let broadcast256 = if has_avx2_fma {
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
                unsafe {
                    apply_fused_2q_loop_fma(base, n_iter, lo, hi, mask0, mask1, &self.broadcast);
                }
                return;
            }
            unsafe {
                use crate::backend::statevector::insert_zero_bit;
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
            unsafe {
                apply_fused_2q_loop_neon(base, n_iter, lo, hi, mask0, mask1, &self.broadcast);
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            use crate::backend::statevector::insert_zero_bit;
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
    /// Prefers the AVX2 paired-group kernel when available and `lo > 0`,
    /// falling back to 128-bit FMA per-group otherwise. Use from `Multi2q`
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
            unsafe {
                match self.tier {
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
                    SimdTier::Avx2Fma | SimdTier::Fma => {
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
                        use crate::backend::statevector::insert_zero_bit;
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

    /// Apply one group at scattered indices. Safe to call from Rayon closures
    /// when callers partition indices across threads.
    ///
    /// # Safety
    /// Caller must ensure `i[0..4]` are valid indices into the state array
    /// and that no other thread is accessing the same indices.
    #[inline(always)]
    pub(crate) unsafe fn apply_group_ptr(&self, state: *mut f64, i: [usize; 4]) {
        #[cfg(target_arch = "x86_64")]
        {
            if !matches!(self.tier, SimdTier::Sse2) {
                apply_fused_2q_group_fma(state, i, &self.broadcast);
            } else {
                apply_fused_2q_group_sse2(state, i, &self.broadcast);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            apply_fused_2q_group_neon(state, i, &self.broadcast);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let a: [Complex64; 4] = [
                *(state.add(i[0] * 2) as *const Complex64),
                *(state.add(i[1] * 2) as *const Complex64),
                *(state.add(i[2] * 2) as *const Complex64),
                *(state.add(i[3] * 2) as *const Complex64),
            ];
            for (r, &idx) in i.iter().enumerate() {
                let result = self.mat[r][0] * a[0]
                    + self.mat[r][1] * a[1]
                    + self.mat[r][2] * a[2]
                    + self.mat[r][3] * a[3];
                *(state.add(idx * 2) as *mut Complex64) = result;
            }
        }
    }
}

// SAFETY: Mat4x4Broadcast contains only SIMD register values (pure data, no pointers).
#[cfg(target_arch = "x86_64")]
unsafe impl Send for Mat4x4Broadcast {}
#[cfg(target_arch = "x86_64")]
unsafe impl Sync for Mat4x4Broadcast {}
// SAFETY: Mat4x4Broadcast on aarch64 contains only float64x2_t values (pure SIMD data).
#[cfg(target_arch = "aarch64")]
unsafe impl Send for Mat4x4Broadcast {}
#[cfg(target_arch = "aarch64")]
unsafe impl Sync for Mat4x4Broadcast {}
unsafe impl Send for PreparedGate2q {}
unsafe impl Sync for PreparedGate2q {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    const EPS: f64 = 1e-12;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn assert_complex_close(a: Complex64, b: Complex64) {
        assert!(
            (a.re - b.re).abs() < EPS && (a.im - b.im).abs() < EPS,
            "expected ({}, {}i), got ({}, {}i)",
            b.re,
            b.im,
            a.re,
            a.im,
        );
    }

    fn identity() -> [[Complex64; 2]; 2] {
        [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
    }

    fn x_gate() -> [[Complex64; 2]; 2] {
        [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
    }

    fn h_gate() -> [[Complex64; 2]; 2] {
        let s = FRAC_1_SQRT_2;
        [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
    }

    #[test]
    fn test_identity_preserves_state() {
        let mut lo = vec![c(0.6, 0.2), c(0.1, -0.3)];
        let mut hi = vec![c(0.4, -0.1), c(-0.5, 0.7)];
        let lo_orig = lo.clone();
        let hi_orig = hi.clone();
        let prepared = PreparedGate1q::new(&identity());
        prepared.apply(&mut lo, &mut hi);
        for (a, b) in lo.iter().zip(lo_orig.iter()) {
            assert_complex_close(*a, *b);
        }
        for (a, b) in hi.iter().zip(hi_orig.iter()) {
            assert_complex_close(*a, *b);
        }
    }

    #[test]
    fn test_x_gate_swaps() {
        let mut lo = vec![c(1.0, 0.0)];
        let mut hi = vec![c(0.0, 0.0)];
        let prepared = PreparedGate1q::new(&x_gate());
        prepared.apply(&mut lo, &mut hi);
        assert_complex_close(lo[0], c(0.0, 0.0));
        assert_complex_close(hi[0], c(1.0, 0.0));
    }

    #[test]
    fn test_h_gate_creates_superposition() {
        let mut lo = vec![c(1.0, 0.0)];
        let mut hi = vec![c(0.0, 0.0)];
        let prepared = PreparedGate1q::new(&h_gate());
        prepared.apply(&mut lo, &mut hi);
        assert_complex_close(lo[0], c(FRAC_1_SQRT_2, 0.0));
        assert_complex_close(hi[0], c(FRAC_1_SQRT_2, 0.0));
    }

    #[test]
    fn test_multi_element_slices() {
        let mut lo = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.5, 0.5), c(0.0, 0.0)];
        let mut hi = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5, -0.5)];
        let mat = h_gate();

        let mut lo_ref = lo.clone();
        let mut hi_ref = hi.clone();
        apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

        let prepared = PreparedGate1q::new(&mat);
        prepared.apply(&mut lo, &mut hi);

        for i in 0..lo.len() {
            assert_complex_close(lo[i], lo_ref[i]);
            assert_complex_close(hi[i], hi_ref[i]);
        }
    }

    #[test]
    fn test_complex_valued_matrix() {
        let mat = [[c(0.0, 1.0), c(0.5, -0.5)], [c(0.5, 0.5), c(0.0, -1.0)]];
        let mut lo = vec![c(1.0, 0.0), c(0.0, 1.0)];
        let mut hi = vec![c(0.0, 0.0), c(1.0, 0.0)];

        let mut lo_ref = lo.clone();
        let mut hi_ref = hi.clone();
        apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

        let prepared = PreparedGate1q::new(&mat);
        prepared.apply(&mut lo, &mut hi);

        for i in 0..lo.len() {
            assert_complex_close(lo[i], lo_ref[i]);
            assert_complex_close(hi[i], hi_ref[i]);
        }
    }

    #[test]
    fn test_odd_length_slices() {
        let mut lo = vec![c(1.0, 0.0), c(0.5, 0.5), c(0.0, 1.0)];
        let mut hi = vec![c(0.0, 0.0), c(0.3, -0.2), c(0.7, 0.1)];
        let mat = h_gate();

        let mut lo_ref = lo.clone();
        let mut hi_ref = hi.clone();
        apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

        let prepared = PreparedGate1q::new(&mat);
        prepared.apply(&mut lo, &mut hi);

        for i in 0..lo.len() {
            assert_complex_close(lo[i], lo_ref[i]);
            assert_complex_close(hi[i], hi_ref[i]);
        }
    }

    #[test]
    fn test_bulk_negate() {
        let mut slice = vec![c(1.0, 2.0), c(-3.0, 0.5), c(0.0, -1.0)];
        let expected = [c(-1.0, -2.0), c(3.0, -0.5), c(0.0, 1.0)];
        negate_slice(&mut slice);
        for (a, e) in slice.iter().zip(expected.iter()) {
            assert_complex_close(*a, *e);
        }
    }

    #[test]
    fn test_bulk_swap() {
        let mut a = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let mut b = vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)];
        swap_slices(&mut a, &mut b);
        assert_complex_close(a[0], c(4.0, 0.0));
        assert_complex_close(b[0], c(1.0, 0.0));
        assert_complex_close(a[2], c(6.0, 0.0));
        assert_complex_close(b[2], c(3.0, 0.0));
    }

    #[test]
    fn test_bulk_norm_sqr_sum() {
        let slice = vec![c(3.0, 4.0), c(1.0, 0.0), c(0.0, 2.0)];
        let result = norm_sqr_sum(&slice);
        let expected = 25.0 + 1.0 + 4.0;
        assert!((result - expected).abs() < EPS);
    }

    #[test]
    fn test_bulk_zero() {
        let mut slice = vec![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
        zero_slice(&mut slice);
        for amp in &slice {
            assert_complex_close(*amp, c(0.0, 0.0));
        }
    }

    #[test]
    fn test_bulk_scale() {
        let mut slice = vec![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 0.0)];
        scale_slice(&mut slice, 2.0);
        assert_complex_close(slice[0], c(2.0, 4.0));
        assert_complex_close(slice[1], c(6.0, 8.0));
        assert_complex_close(slice[2], c(10.0, 0.0));
    }

    #[test]
    fn test_scale_complex_slice() {
        let phase = c(0.0, 1.0);
        let mut slice = vec![c(1.0, 0.0), c(0.0, 1.0), c(3.0, 4.0)];
        scale_complex_slice(&mut slice, phase);
        assert_complex_close(slice[0], c(0.0, 1.0));
        assert_complex_close(slice[1], c(-1.0, 0.0));
        assert_complex_close(slice[2], c(-4.0, 3.0));
    }

    #[test]
    fn test_scale_complex_slice_phase() {
        let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        let mut slice = vec![c(1.0, 0.0), c(1.0, 0.0), c(0.0, 1.0), c(0.5, -0.3)];
        let expected: Vec<Complex64> = slice.iter().map(|&v| v * phase).collect();
        scale_complex_slice(&mut slice, phase);
        for (a, e) in slice.iter().zip(expected.iter()) {
            assert_complex_close(*a, *e);
        }
    }

    #[test]
    fn test_scale_complex_slice_single_element() {
        let phase = c(0.0, -1.0);
        let mut slice = vec![c(2.0, 3.0)];
        let expected = slice[0] * phase;
        scale_complex_slice(&mut slice, phase);
        assert_complex_close(slice[0], expected);
    }

    #[test]
    fn test_scale_complex_to_slice_lengths() {
        let factor = Complex64::from_polar(1.3, 0.7);
        for len in [1usize, 2, 3, 4, 5, 7, 8, 16, 17, 33] {
            let src: Vec<Complex64> = (0..len)
                .map(|i| c((i as f64) + 0.25, (i as f64) * 0.5 - 1.0))
                .collect();
            let mut dst = vec![c(99.0, 99.0); len];
            scale_complex_to_slice(&mut dst, &src, factor);
            for i in 0..len {
                assert_complex_close(dst[i], src[i] * factor);
            }
        }
    }

    fn identity_4x4() -> [[Complex64; 4]; 4] {
        let z = c(0.0, 0.0);
        let o = c(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]]
    }

    fn cx_4x4() -> [[Complex64; 4]; 4] {
        let z = c(0.0, 0.0);
        let o = c(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    }

    fn cz_4x4() -> [[Complex64; 4]; 4] {
        let z = c(0.0, 0.0);
        let o = c(1.0, 0.0);
        let m = c(-1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, m]]
    }

    fn apply_2q_reference(
        state: &mut [Complex64],
        mat: &[[Complex64; 4]; 4],
        q0: usize,
        q1: usize,
    ) {
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let n = state.len();
        let n_iter = n >> 2;
        for k in 0..n_iter {
            let idx = crate::backend::statevector::insert_zero_bit(
                crate::backend::statevector::insert_zero_bit(k, lo),
                hi,
            );
            let i = [idx, idx | mask1, idx | mask0, idx | mask0 | mask1];
            let a = [state[i[0]], state[i[1]], state[i[2]], state[i[3]]];
            for (r, &ii) in i.iter().enumerate() {
                state[ii] =
                    mat[r][0] * a[0] + mat[r][1] * a[1] + mat[r][2] * a[2] + mat[r][3] * a[3];
            }
        }
    }

    #[test]
    fn test_prepared_2q_identity() {
        let mut state = vec![c(0.5, 0.1), c(0.3, -0.2), c(-0.1, 0.4), c(0.6, -0.3)];
        let orig = state.clone();
        let prepared = PreparedGate2q::new(&identity_4x4());
        prepared.apply_full(&mut state, 2, 0, 1);
        for (a, e) in state.iter().zip(orig.iter()) {
            assert_complex_close(*a, *e);
        }
    }

    #[test]
    fn test_prepared_2q_cx_on_11() {
        // |11⟩ → CX → |10⟩ (target q1 flips when control q0=1)
        let mut state = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
        let mut ref_state = state.clone();
        let mat = cx_4x4();
        let prepared = PreparedGate2q::new(&mat);
        prepared.apply_full(&mut state, 2, 0, 1);
        apply_2q_reference(&mut ref_state, &mat, 0, 1);
        for (a, e) in state.iter().zip(ref_state.iter()) {
            assert_complex_close(*a, *e);
        }
    }

    #[test]
    fn test_prepared_2q_cz_matches_reference() {
        let mut state = vec![c(0.5, 0.0), c(0.3, 0.1), c(-0.2, 0.4), c(0.6, -0.1)];
        let mut ref_state = state.clone();
        let mat = cz_4x4();
        let prepared = PreparedGate2q::new(&mat);
        prepared.apply_full(&mut state, 2, 0, 1);
        apply_2q_reference(&mut ref_state, &mat, 0, 1);
        for (a, e) in state.iter().zip(ref_state.iter()) {
            assert_complex_close(*a, *e);
        }
    }

    #[test]
    fn test_prepared_2q_3qubit_system() {
        // 3-qubit system: apply CX on q0, q2 (non-adjacent)
        let mut state = vec![c(0.0, 0.0); 8];
        state[0] = c(FRAC_1_SQRT_2, 0.0);
        state[5] = c(FRAC_1_SQRT_2, 0.0); // |101⟩
        let mut ref_state = state.clone();
        let mat = cx_4x4();
        let prepared = PreparedGate2q::new(&mat);
        prepared.apply_full(&mut state, 3, 0, 2);
        apply_2q_reference(&mut ref_state, &mat, 0, 2);
        for (i, (a, e)) in state.iter().zip(ref_state.iter()).enumerate() {
            assert!((a - e).norm() < EPS, "state[{i}]: expected {e}, got {a}");
        }
    }

    /// A dense, asymmetric 4×4 matrix that exercises every coefficient slot.
    /// Built from a non-special unitary so any indexing bug in the AVX2
    /// paired-group kernel surfaces as a numerical mismatch.
    fn dense_4x4() -> [[Complex64; 4]; 4] {
        let s = FRAC_1_SQRT_2;
        let h2 = [
            [c(0.5, 0.0), c(0.5, 0.0), c(0.5, 0.0), c(0.5, 0.0)],
            [c(0.5, 0.0), c(-0.5, 0.0), c(0.5, 0.0), c(-0.5, 0.0)],
            [c(0.5, 0.0), c(0.5, 0.0), c(-0.5, 0.0), c(-0.5, 0.0)],
            [c(0.5, 0.0), c(-0.5, 0.0), c(-0.5, 0.0), c(0.5, 0.0)],
        ];
        let phase = [
            [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
            [c(0.0, 0.0), c(s, s), c(0.0, 0.0), c(0.0, 0.0)],
            [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0)],
            [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-s, s)],
        ];
        let mut out = [[c(0.0, 0.0); 4]; 4];
        for r in 0..4 {
            for col in 0..4 {
                let mut acc = c(0.0, 0.0);
                for k in 0..4 {
                    acc += phase[r][k] * h2[k][col];
                }
                out[r][col] = acc;
            }
        }
        out
    }

    fn random_state(num_qubits: usize, seed: u64) -> Vec<Complex64> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = 1usize << num_qubits;
        let mut s = Vec::with_capacity(n);
        let mut norm = 0.0;
        for _ in 0..n {
            let re: f64 = rng.random_range(-1.0..1.0);
            let im: f64 = rng.random_range(-1.0..1.0);
            norm += re * re + im * im;
            s.push(c(re, im));
        }
        let inv = norm.sqrt().recip();
        for v in &mut s {
            v.re *= inv;
            v.im *= inv;
        }
        s
    }

    fn assert_state_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let d = (*a - *e).norm();
            assert!(
                d < 1e-10,
                "{label} state[{i}]: expected {e}, got {a} (diff {d:.2e})"
            );
        }
    }

    /// Reference test: AVX2 paired-group kernel must agree with the 128-bit
    /// FMA per-group kernel across (q0, q1) configurations covering adjacent,
    /// non-adjacent, reversed-order, and the lo == 0 fallback path.
    #[test]
    fn test_prepared_2q_apply_tiled_matches_apply_full() {
        let mat = dense_4x4();
        let configs: &[(usize, usize, usize)] = &[
            (4, 0, 1),  // adjacent, lo == 0 (forces 128-bit fallback inside apply_tiled)
            (4, 1, 0),  // reversed, lo == 0
            (4, 1, 2),  // adjacent, lo > 0 (AVX2 path)
            (4, 2, 1),  // reversed, lo > 0
            (5, 0, 4),  // far apart, lo == 0
            (5, 4, 0),  // reversed, lo == 0
            (5, 1, 4),  // far apart, lo > 0
            (5, 4, 1),  // reversed, lo > 0
            (6, 2, 5),  // mid-range, lo > 0
            (8, 0, 7),  // 8-qubit, span entire register, lo == 0
            (8, 1, 7),  // 8-qubit, span entire register, lo > 0
            (8, 7, 1),  // reversed
            (10, 3, 6), // 10-qubit AVX2 path
        ];

        for &(nq, q0, q1) in configs {
            let state_init = random_state(nq, 0xCAFE_F00D);
            let prepared = PreparedGate2q::new(&mat);

            let mut via_full = state_init.clone();
            prepared.apply_full(&mut via_full, nq, q0, q1);

            let mut via_tiled = state_init.clone();
            prepared.apply_tiled(&mut via_tiled, nq, q0, q1);

            assert_state_close(
                &via_tiled,
                &via_full,
                &format!("nq={nq} q0={q0} q1={q1} apply_tiled vs apply_full"),
            );
        }
    }
}
