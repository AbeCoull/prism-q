//! SIMD-accelerated kernels for single-qubit gate application and bulk
//! operations (negate, swap, norm-squared accumulation, zero, scale).
//!
//! Dispatches to the best available SIMD tier at runtime:
//!
//! | Tier | Requirement | Complex multiply cost |
//! |----------|------------------|---------------------------------|
//! | AVX-512  | `avx512f` + `avx512vpopcntdq`, `PRISM_AVX512=1` | 512-bit: 1 mul + 1 fmaddsub (4 pairs) |
//! | AVX2+FMA | `avx2` + `fma`   | 256-bit: 1 mul + 1 fmaddsub (2 pairs) |
//! | FMA      | `fma`            | 128-bit: 1 mul + 1 fmaddsub |
//! | NEON     | aarch64 baseline | 128-bit: 1 mul + 1 fma (pre-negated) |
//! | SSE2     | x86_64 baseline  | 128-bit: 2 mul + shuffle + xor + add |
//! | Scalar   | fallback         | Standard Complex64 ops |
//!
//! Call [`PreparedGate1q::new`] once per gate, then [`PreparedGate1q::apply`]
//! per chunk. This hoists the matrix broadcast and CPU feature detection out
//! of the inner loop.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "x86_64")]
use crate::backend::word_ops::{WordTier, word_tier};

mod gate1q;
mod gate2q;
mod kraus;
mod slice;
mod walsh;

pub(crate) use gate1q::{PreparedGate1q, apply_1q_sandwich_blocked, apply_diagonal_sequential};
pub(crate) use gate2q::PreparedGate2q;
#[cfg(target_arch = "x86_64")]
pub(crate) use kraus::PreparedKraus2q;
#[cfg(feature = "distributed")]
pub(crate) use slice::combine_global_half;
pub(crate) use slice::{
    negate_slice, norm_sqr_sum, norm_sqr_sum_scale, norm_sqr_to_slice, norm_sqr_to_slice_scaled,
    scale_complex_slice, scale_complex_to_slice, swap_slices, zero_slice,
};
pub(crate) use walsh::walsh_hadamard;

/// Visit `$n` complex values four at a time, binding the f64 offset and a lane mask that
/// is all ones except on the last partial step.
#[cfg(target_arch = "x86_64")]
macro_rules! for_each_quad {
    ($n:expr, |$off:ident, $k:ident| $body:block) => {{
        let lanes: usize = 2 * $n;
        let full = lanes & !7;
        let mut $off = 0usize;
        while $off < full {
            let $k: __mmask8 = 0xFF;
            $body
            $off += 8;
        }
        if $off < lanes {
            let $k: __mmask8 = (1u8 << (lanes - $off)) - 1;
            $body
        }
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use for_each_quad;

/// Load eight f64 lanes at `p`, or the lanes `k` selects when it is not all ones.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn load_pd(p: *const f64, k: __mmask8) -> __m512d {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if k == 0xFF {
            _mm512_loadu_pd(p)
        } else {
            _mm512_maskz_loadu_pd(k, p)
        }
    }
}

/// Store eight f64 lanes at `p`, or the lanes `k` selects when it is not all ones.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn store_pd(p: *mut f64, k: __mmask8, v: __m512d) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        if k == 0xFF {
            _mm512_storeu_pd(p, v);
        } else {
            _mm512_mask_storeu_pd(p, k, v);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn complex_mul_neon(
    c_rr: float64x2_t,
    c_ii_as: float64x2_t,
    z: float64x2_t,
) -> float64x2_t {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let z_swap = vextq_f64(z, z, 1);
        let prod = vmulq_f64(c_rr, z);
        vfmaq_f64(prod, c_ii_as, z_swap)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn complex_mul_sse2(
    c_rr: __m128d,
    c_ii: __m128d,
    z: __m128d,
    sign_mask: __m128d,
) -> __m128d {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        let z_swap = _mm_shuffle_pd(z, z, 0b01);
        let t1 = _mm_mul_pd(c_rr, z);
        let t2 = _mm_mul_pd(c_ii, z_swap);
        let t2_neg = _mm_xor_pd(t2, sign_mask);
        _mm_add_pd(t1, t2_neg)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "fma")]
pub(crate) unsafe fn complex_mul_fma(c_rr: __m128d, c_ii: __m128d, z: __m128d) -> __m128d {
    let z_swap = _mm_shuffle_pd(z, z, 0b01);
    let t = _mm_mul_pd(c_ii, z_swap);
    _mm_fmaddsub_pd(c_rr, z, t)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn complex_mul_avx2fma(c_rr: __m256d, c_ii: __m256d, z: __m256d) -> __m256d {
    let z_swap = _mm256_permute_pd(z, 0b0101);
    let t = _mm256_mul_pd(c_ii, z_swap);
    _mm256_fmaddsub_pd(c_rr, z, t)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn complex_mul_avx512(c_rr: __m512d, c_ii: __m512d, z: __m512d) -> __m512d {
    let z_swap = _mm512_permute_pd::<0b0101_0101>(z);
    let t = _mm512_mul_pd(c_ii, z_swap);
    _mm512_fmaddsub_pd(c_rr, z, t)
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum SimdTier {
    Avx512,
    Avx2Fma,
    Fma,
    Sse2,
}

/// The widest complex-arithmetic tier, read once per process. AVX-512 takes part under
/// the same detection and `PRISM_AVX512=1` opt-in as [`word_tier`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn simd_tier() -> SimdTier {
    static CACHED: std::sync::OnceLock<SimdTier> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        if has_avx2_fma() && word_tier() == WordTier::Avx512 {
            SimdTier::Avx512
        } else if has_avx2_fma() {
            SimdTier::Avx2Fma
        } else if has_fma() {
            SimdTier::Fma
        } else {
            SimdTier::Sse2
        }
    })
}

/// Bench-only kill switch for the AVX2 paired-group and AVX-512 quad-group 2q kernels.
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

/// Kill switch for the AVX2 and AVX-512 dense two-qubit Kraus kernels.
///
/// Reads `PRISM_NO_AVX2_KRAUS` once and caches the result. Set the variable
/// to route the dense sweep through the blocked scalar path instead, for
/// single-binary A/B timing and for exercising that path on SIMD hosts.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn kraus_2q_wide_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PRISM_NO_AVX2_KRAUS").is_none())
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

/// Pack the bits of `index` selected by `mask` into the low bits, lowest mask bit first.
///
/// Uses PEXT when [`has_bmi2`] reports BMI2; otherwise [`pack_masked_bits_scalar`].
#[inline]
pub(crate) fn pack_masked_bits(index: usize, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if has_bmi2() {
            // SAFETY: BMI2 detected
            return unsafe { _pext_u64(index as u64, mask) as usize };
        }
    }
    pack_masked_bits_scalar(index, mask)
}

/// Portable form of [`pack_masked_bits`], bit-identical to PEXT.
#[inline(always)]
pub(crate) fn pack_masked_bits_scalar(index: usize, mask: u64) -> usize {
    let mut result = 0usize;
    let mut bit = 0;
    let mut m = mask;
    while m != 0 {
        let pos = m.trailing_zeros() as usize;
        if index & (1usize << pos) != 0 {
            result |= 1usize << bit;
        }
        bit += 1;
        m &= m.wrapping_sub(1);
    }
    result
}

#[cfg(test)]
mod tests;
