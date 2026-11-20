//! The dense two-qubit Kraus sweep for the density matrix, as a 16x16 block superoperator.

#[cfg(target_arch = "x86_64")]
use num_complex::Complex64;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Precomputed 16x16 block superoperator for the dense two-qubit Kraus sweep,
/// in 256-bit broadcast form.
///
/// Row `r` is stored as eight register pairs covering its 16 coefficients, two
/// consecutive `j` slots per register: `rr[r][k]` holds
/// `[re(s[r][2k]); 2, re(s[r][2k + 1]); 2]` and `ii[r][k]` the imaginary
/// parts. [`PreparedKraus2q::apply_block_ptr`] advances each row inner product
/// two complex terms per FMA over the always-16-wide `j` axis, so the width
/// does not depend on the qubit pair the way the block-run sweep's does.
#[cfg(target_arch = "x86_64")]
#[repr(align(64))]
pub(crate) struct PreparedKraus2q {
    rr: [[__m256d; 8]; 16],
    ii: [[__m256d; 8]; 16],
}

#[cfg(target_arch = "x86_64")]
impl PreparedKraus2q {
    /// # Safety
    ///
    /// Requires AVX, implied by a detected AVX2 feature.
    #[inline]
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(s: &[[Complex64; 16]; 16]) -> Self {
        let mut rr = [[_mm256_setzero_pd(); 8]; 16];
        let mut ii = [[_mm256_setzero_pd(); 8]; 16];
        for (r, row) in s.iter().enumerate() {
            for k in 0..8 {
                let (a, b) = (row[2 * k], row[2 * k + 1]);
                rr[r][k] = _mm256_setr_pd(a.re, a.re, b.re, b.re);
                ii[r][k] = _mm256_setr_pd(a.im, a.im, b.im, b.im);
            }
        }
        Self { rr, ii }
    }

    /// Apply the superoperator to the 16-amplitude block at `base | flats[j]`,
    /// reading all 16 slots before the first store.
    ///
    /// # Safety
    ///
    /// AVX2 and FMA must be detected, every `base | flats[j]` must be in
    /// bounds for `state` viewed as interleaved `Complex64`, and no other
    /// thread may access those 16 slots.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(crate) unsafe fn apply_block_ptr(&self, state: *mut f64, base: usize, flats: &[usize; 16]) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            let mut v = [_mm256_setzero_pd(); 8];
            for (k, pair) in v.iter_mut().enumerate() {
                let lo = _mm_loadu_pd(state.add((base | flats[2 * k]) * 2));
                let hi = _mm_loadu_pd(state.add((base | flats[2 * k + 1]) * 2));
                *pair = _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1);
            }
            for (r, &out) in flats.iter().enumerate() {
                let rr = &self.rr[r];
                let ii = &self.ii[r];
                let mut accx = _mm256_mul_pd(rr[0], v[0]);
                let mut accy = _mm256_mul_pd(ii[0], v[0]);
                for k in 1..8 {
                    accx = _mm256_fmadd_pd(rr[k], v[k], accx);
                    accy = _mm256_fmadd_pd(ii[k], v[k], accy);
                }
                let acc = _mm256_addsub_pd(accx, _mm256_shuffle_pd(accy, accy, 0b0101));
                let sum = _mm_add_pd(_mm256_castpd256_pd128(acc), _mm256_extractf128_pd(acc, 1));
                _mm_storeu_pd(state.add((base | out) * 2), sum);
            }
        }
    }

    /// [`Self::apply_block_ptr`] four complex terms per FMA. Register pairs
    /// `rr[r][2m..2m + 2]` lie side by side, so one 512-bit load of them covers
    /// coefficients `4m..4m + 4` against slots `4m..4m + 4` gathered into one register.
    ///
    /// # Safety
    ///
    /// AVX-512F must be detected; the slot contract is that of
    /// [`Self::apply_block_ptr`].
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn apply_block_ptr_avx512(
        &self,
        state: *mut f64,
        base: usize,
        flats: &[usize; 16],
    ) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            let slot = |j: usize| state.add((base | flats[j]) * 2);
            let mut v = [_mm512_setzero_pd(); 4];
            for (m, quad) in v.iter_mut().enumerate() {
                let lo = _mm256_insertf128_pd(
                    _mm256_castpd128_pd256(_mm_loadu_pd(slot(4 * m))),
                    _mm_loadu_pd(slot(4 * m + 1)),
                    1,
                );
                let hi = _mm256_insertf128_pd(
                    _mm256_castpd128_pd256(_mm_loadu_pd(slot(4 * m + 2))),
                    _mm_loadu_pd(slot(4 * m + 3)),
                    1,
                );
                *quad = _mm512_insertf64x4::<1>(_mm512_castpd256_pd512(lo), hi);
            }
            let one = _mm512_set1_pd(1.0);
            for (r, &out) in flats.iter().enumerate() {
                let rr = self.rr[r].as_ptr() as *const f64;
                let ii = self.ii[r].as_ptr() as *const f64;
                let mut accx = _mm512_mul_pd(_mm512_loadu_pd(rr), v[0]);
                let mut accy = _mm512_mul_pd(_mm512_loadu_pd(ii), v[0]);
                for (m, quad) in v.iter().enumerate().skip(1) {
                    accx = _mm512_fmadd_pd(_mm512_loadu_pd(rr.add(8 * m)), *quad, accx);
                    accy = _mm512_fmadd_pd(_mm512_loadu_pd(ii.add(8 * m)), *quad, accy);
                }
                let acc = _mm512_fmaddsub_pd(one, accx, _mm512_permute_pd::<0b0101_0101>(accy));
                let half = _mm256_add_pd(
                    _mm512_castpd512_pd256(acc),
                    _mm512_extractf64x4_pd::<1>(acc),
                );
                let sum = _mm_add_pd(_mm256_castpd256_pd128(half), _mm256_extractf128_pd(half, 1));
                _mm_storeu_pd(state.add((base | out) * 2), sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
// SAFETY: PreparedKraus2q contains immutable SIMD broadcast data and no raw pointers.
unsafe impl Send for PreparedKraus2q {}
#[cfg(target_arch = "x86_64")]
// SAFETY: PreparedKraus2q contains immutable SIMD broadcast data and no raw pointers.
unsafe impl Sync for PreparedKraus2q {}
