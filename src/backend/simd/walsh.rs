//! In-place Walsh-Hadamard transform over an `f64` array, blocked for L1 and L2.

#[cfg(feature = "parallel")]
use num_complex::Complex64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use super::{SimdTier, simd_tier};

/// Levels of [`walsh_hadamard`] done while a block of `2^11` entries (16 KiB) sits in L1.
const WALSH_L1_LOG: usize = 11;
/// Levels done while a block of `2^14` entries (128 KiB) sits in L2. The levels above it
/// go in sweeps over the whole array, three to a sweep.
const WALSH_L2_LOG: usize = 14;
/// Offsets per stream that one parallel item of a sweep covers.
#[cfg(feature = "parallel")]
const WALSH_SWEEP_CHUNK: usize = 1 << 12;
/// Smallest transform split over the Rayon pool.
#[cfg(feature = "parallel")]
const WALSH_PAR_LOG: usize = 16;

/// Unnormalized Walsh-Hadamard transform in place, `a[t] = Σ_s (-1)^|s & t| a[s]`, over a
/// power-of-two length. Every entry sees its levels in increasing order whatever the
/// blocking, vector width and thread count, so the result equals one radix-2 sweep per
/// level bit for bit.
pub(crate) fn walsh_hadamard(a: &mut [f64]) {
    debug_assert!(a.len().is_power_of_two());
    #[cfg(target_arch = "x86_64")]
    if a.len() >= 4 && matches!(simd_tier(), SimdTier::Avx512 | SimdTier::Avx2Fma) {
        walsh_blocked::<true>(a);
        return;
    }
    walsh_blocked::<false>(a);
}

/// Every level below [`WALSH_L2_LOG`] one block at a time, then the rest in sweeps of up
/// to three levels. `AVX` needs AVX2 and at least four entries.
pub(super) fn walsh_blocked<const AVX: bool>(a: &mut [f64]) {
    let r = a.len().trailing_zeros() as usize;
    #[cfg(feature = "parallel")]
    let parallel = r >= WALSH_PAR_LOG;
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        a.par_chunks_mut(1 << WALSH_L2_LOG)
            .for_each(walsh_block::<AVX>);
    } else {
        a.chunks_mut(1 << WALSH_L2_LOG).for_each(walsh_block::<AVX>);
    }
    #[cfg(not(feature = "parallel"))]
    a.chunks_mut(1 << WALSH_L2_LOG).for_each(walsh_block::<AVX>);
    let mut level = WALSH_L2_LOG;
    while level < r {
        let step = (r - level).min(3);
        let h = 1usize << level;
        #[cfg(feature = "parallel")]
        if parallel {
            use crate::backend::SendPtr;
            use rayon::prelude::*;
            let span = h << step;
            let per_block = h / WALSH_SWEEP_CHUNK;
            let ptr = SendPtr(a.as_mut_ptr().cast::<Complex64>());
            (0..a.len() / span * per_block)
                .into_par_iter()
                .for_each(|item| {
                    let start = item % per_block * WALSH_SWEEP_CHUNK;
                    // SAFETY: `item / per_block * span + span <= a.len()`, and each item
                    // owns offsets `start..start + WALSH_SWEEP_CHUNK` of the streams of
                    // one superblock, so items touch disjoint entries. `start` and the
                    // chunk are multiples of 8, and `h >= 2^14`.
                    unsafe {
                        walsh_radix::<AVX>(
                            ptr.as_f64_ptr().add(item / per_block * span),
                            level,
                            step,
                            start,
                            start + WALSH_SWEEP_CHUNK,
                        )
                    };
                });
            level += step;
            continue;
        }
        for superblock in a.chunks_exact_mut(h << step) {
            // SAFETY: the superblock holds `h << step` entries, and `h >= 2^14`.
            unsafe { walsh_radix::<AVX>(superblock.as_mut_ptr(), level, step, 0, h) };
        }
        level += step;
    }
}

/// Every level of one block of at most `2^14` entries.
fn walsh_block<const AVX: bool>(block: &mut [f64]) {
    let r = block.len().trailing_zeros() as usize;
    for l1 in block.chunks_mut(1 << WALSH_L1_LOG) {
        walsh_levels::<AVX>(l1, 0, r.min(WALSH_L1_LOG));
    }
    if r > WALSH_L1_LOG {
        walsh_levels::<AVX>(block, WALSH_L1_LOG, r);
    }
}

/// Levels `lo..hi` over a slice of a multiple of `2^hi` entries, up to three per pass.
fn walsh_levels<const AVX: bool>(a: &mut [f64], lo: usize, hi: usize) {
    let mut level = lo;
    #[cfg(target_arch = "x86_64")]
    if AVX && level == 0 {
        // SAFETY: AVX2 detected; the slice holds a power of two of at least 4 entries.
        unsafe { walsh_levels01_avx2(a) };
        level = 2;
    }
    while level < hi {
        let step = (hi - level).min(3);
        let h = 1usize << level;
        for superblock in a.chunks_exact_mut(h << step) {
            // SAFETY: the superblock holds `h << step` entries; under `AVX`, `level >= 2`.
            unsafe { walsh_radix::<AVX>(superblock.as_mut_ptr(), level, step, 0, h) };
        }
        level += step;
    }
}

/// Levels `level..level + step` over offsets `start..end` of the `2^step` streams of the
/// superblock at `p`, `2^level` entries apart.
///
/// # Safety
///
/// `p` is valid for reads and writes of `2^(level + step)` entries, no other thread
/// touches offsets `start..end` of its streams, and `start <= end <= 2^level`. Under
/// `AVX`, AVX2 is detected, `level >= 2`, and `start` and `end` are multiples of 4, or
/// of 8 from level 3 up.
#[inline(always)]
unsafe fn walsh_radix<const AVX: bool>(
    p: *mut f64,
    level: usize,
    step: usize,
    start: usize,
    end: usize,
) {
    let h = 1usize << level;
    #[cfg(target_arch = "x86_64")]
    if AVX {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            match (step, h >= 8) {
                (3, true) => walsh_radix_avx2::<8, 2>(p, h, start, end),
                (2, true) => walsh_radix_avx2::<4, 2>(p, h, start, end),
                (_, true) => walsh_radix_avx2::<2, 2>(p, h, start, end),
                (3, false) => walsh_radix_avx2::<8, 1>(p, h, start, end),
                (2, false) => walsh_radix_avx2::<4, 1>(p, h, start, end),
                (_, false) => walsh_radix_avx2::<2, 1>(p, h, start, end),
            }
        }
        return;
    }
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        match step {
            3 => walsh_radix_scalar::<8>(p, h, start, end),
            2 => walsh_radix_scalar::<4>(p, h, start, end),
            _ => walsh_radix_scalar::<2>(p, h, start, end),
        }
    }
}

/// Radix-`N` butterflies on entries `p[k h + i]`, `k < N`, for each `i` in `start..end`.
///
/// # Safety
///
/// `p` is valid for reads and writes of `(N - 1) h + end` entries that no other thread
/// touches.
#[inline(always)]
unsafe fn walsh_radix_scalar<const N: usize>(p: *mut f64, h: usize, start: usize, end: usize) {
    for i in start..end {
        let mut v = [0.0f64; N];
        for (k, slot) in v.iter_mut().enumerate() {
            // SAFETY: same contract as the enclosing unsafe fn.
            *slot = unsafe { *p.add(k * h + i) };
        }
        let mut s = 1;
        while s < N {
            for k in 0..N {
                if k & s == 0 {
                    let (x, y) = (v[k], v[k + s]);
                    v[k] = x + y;
                    v[k + s] = x - y;
                }
            }
            s <<= 1;
        }
        for (k, value) in v.iter().enumerate() {
            // SAFETY: same contract as the enclosing unsafe fn.
            unsafe { *p.add(k * h + i) = *value };
        }
    }
}

/// [`walsh_radix_scalar`] on `4 W` consecutive offsets per step, `W` vectors to a stream:
/// two cover a whole cache line, so each line is loaded once even when the streams alias
/// in the cache sets.
///
/// # Safety
///
/// As [`walsh_radix_scalar`], with AVX2 detected and `start` and `end` multiples of `4 W`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::needless_range_loop)]
unsafe fn walsh_radix_avx2<const N: usize, const W: usize>(
    p: *mut f64,
    h: usize,
    start: usize,
    end: usize,
) {
    let mut v = [[_mm256_setzero_pd(); W]; N];
    let mut i = start;
    while i < end {
        for k in 0..N {
            for w in 0..W {
                // SAFETY: same contract as the enclosing unsafe fn.
                v[k][w] = unsafe { _mm256_loadu_pd(p.add(k * h + i + 4 * w)) };
            }
        }
        let mut s = 1;
        while s < N {
            for k in 0..N {
                if k & s == 0 {
                    for w in 0..W {
                        let (x, y) = (v[k][w], v[k + s][w]);
                        v[k][w] = _mm256_add_pd(x, y);
                        v[k + s][w] = _mm256_sub_pd(x, y);
                    }
                }
            }
            s <<= 1;
        }
        for k in 0..N {
            for w in 0..W {
                // SAFETY: same contract as the enclosing unsafe fn.
                unsafe { _mm256_storeu_pd(p.add(k * h + i + 4 * w), v[k][w]) };
            }
        }
        i += 4 * W;
    }
}

/// Levels 0 and 1 inside each vector of four entries.
///
/// # Safety
///
/// AVX2 detected and `a.len()` a multiple of 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn walsh_levels01_avx2(a: &mut [f64]) {
    let p = a.as_mut_ptr();
    for i in (0..a.len()).step_by(4) {
        // SAFETY: same contract as the enclosing unsafe fn; `i + 4 <= a.len()`.
        unsafe {
            let v = _mm256_loadu_pd(p.add(i));
            let swapped = _mm256_permute_pd::<0b0101>(v);
            let v = _mm256_blend_pd::<0b1010>(_mm256_add_pd(v, swapped), _mm256_sub_pd(swapped, v));
            let swapped = _mm256_permute2f128_pd::<1>(v, v);
            let v = _mm256_blend_pd::<0b1100>(_mm256_add_pd(v, swapped), _mm256_sub_pd(swapped, v));
            _mm256_storeu_pd(p.add(i), v);
        }
    }
}
