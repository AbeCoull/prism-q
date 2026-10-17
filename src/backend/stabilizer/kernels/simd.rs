//! Rowmul word kernels: XOR of X/Z words with Pauli phase accumulation, in
//! AVX2 (with a non-temporal store variant), NEON, and scalar forms.

#[cfg(target_arch = "x86_64")]
use crate::backend::word_ops::has_avx2;
pub(crate) use crate::backend::word_ops::xor_words_ptr as xor_words;

/// Rowmul word loop: XOR x/z words from src into dst, returning the
/// accumulated phase sum (caller applies `sum & 3 >= 2`).
#[inline(always)]
pub(crate) fn rowmul_words(
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

    #[cfg(target_arch = "x86_64")]
    if nw >= 16 && has_avx2() {
        // SAFETY: AVX2 detected at runtime; slices have equal lengths (debug_assert above).
        // The NT helper performs its own alignment check and falls back to
        // non-NT stores if alignment is insufficient.
        return unsafe { rowmul_words_avx2_nt(dst_x, dst_z, src_x, src_z, initial_sum) };
    }
    #[cfg(target_arch = "x86_64")]
    if nw >= 4 && has_avx2() {
        // SAFETY: AVX2 detected at runtime; slices have equal lengths (debug_assert above).
        return unsafe { rowmul_words_avx2(dst_x, dst_z, src_x, src_z, initial_sum) };
    }
    #[cfg(target_arch = "aarch64")]
    if nw >= 2 {
        // SAFETY: NEON is baseline on aarch64; slices have equal lengths.
        return unsafe { rowmul_words_neon(dst_x, dst_z, src_x, src_z, initial_sum) };
    }

    #[allow(unreachable_code)]
    rowmul_words_scalar(dst_x, dst_z, src_x, src_z, initial_sum)
}

fn rowmul_words_scalar(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
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
#[target_feature(enable = "avx2")]
unsafe fn rowmul_words_avx2(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;

        let nw = dst_x.len();
        let chunks = nw / 4;

        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let vzero = _mm256_setzero_si256();

        let mut vsum_pos = _mm256_setzero_si256();
        let mut vsum_nz = _mm256_setzero_si256();

        for i in 0..chunks {
            let off = i * 4;
            let vx1 = _mm256_loadu_si256(src_x.as_ptr().add(off) as *const __m256i);
            let vz1 = _mm256_loadu_si256(src_z.as_ptr().add(off) as *const __m256i);
            let vx2 = _mm256_loadu_si256(dst_x.as_ptr().add(off) as *const __m256i);
            let vz2 = _mm256_loadu_si256(dst_z.as_ptr().add(off) as *const __m256i);

            let vnew_x = _mm256_xor_si256(vx1, vx2);
            let vnew_z = _mm256_xor_si256(vz1, vz2);
            _mm256_storeu_si256(dst_x.as_mut_ptr().add(off) as *mut __m256i, vnew_x);
            _mm256_storeu_si256(dst_z.as_mut_ptr().add(off) as *mut __m256i, vnew_z);

            let vor = _mm256_or_si256(_mm256_or_si256(vx1, vz1), _mm256_or_si256(vx2, vz2));
            if _mm256_testz_si256(vor, vor) != 0 {
                continue;
            }

            let vnonzero = _mm256_and_si256(
                _mm256_and_si256(_mm256_or_si256(vnew_x, vnew_z), _mm256_or_si256(vx1, vz1)),
                _mm256_or_si256(vx2, vz2),
            );

            // _mm256_andnot_si256(a, b) = ~a & b
            let vpos = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_and_si256(vx1, vz1), _mm256_andnot_si256(vx2, vz2)),
                    _mm256_and_si256(_mm256_andnot_si256(vz1, vx1), _mm256_and_si256(vx2, vz2)),
                ),
                _mm256_and_si256(_mm256_andnot_si256(vx1, vz1), _mm256_andnot_si256(vz2, vx2)),
            );

            let lo_p = _mm256_and_si256(vpos, low_mask);
            let hi_p = _mm256_and_si256(_mm256_srli_epi16(vpos, 4), low_mask);
            let cnt_p = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_p),
                _mm256_shuffle_epi8(lookup, hi_p),
            );
            vsum_pos = _mm256_add_epi64(vsum_pos, _mm256_sad_epu8(cnt_p, vzero));

            let lo_n = _mm256_and_si256(vnonzero, low_mask);
            let hi_n = _mm256_and_si256(_mm256_srli_epi16(vnonzero, 4), low_mask);
            let cnt_n = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_n),
                _mm256_shuffle_epi8(lookup, hi_n),
            );
            vsum_nz = _mm256_add_epi64(vsum_nz, _mm256_sad_epu8(cnt_n, vzero));
        }

        let pos_lo = _mm256_castsi256_si128(vsum_pos);
        let pos_hi = _mm256_extracti128_si256(vsum_pos, 1);
        let pos_sum = _mm_add_epi64(pos_lo, pos_hi);
        let total_pos =
            (_mm_cvtsi128_si64(pos_sum) as u64).wrapping_add(_mm_extract_epi64(pos_sum, 1) as u64);

        let nz_lo = _mm256_castsi256_si128(vsum_nz);
        let nz_hi = _mm256_extracti128_si256(vsum_nz, 1);
        let nz_sum = _mm_add_epi64(nz_lo, nz_hi);
        let total_nz =
            (_mm_cvtsi128_si64(nz_sum) as u64).wrapping_add(_mm_extract_epi64(nz_sum, 1) as u64);

        let mut sum = initial_sum
            .wrapping_add(2u64.wrapping_mul(total_pos))
            .wrapping_sub(total_nz);

        let tail = chunks * 4;
        for w in tail..nw {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rowmul_words_avx2_nt(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;

        if (dst_x.as_ptr() as usize) & 31 != 0 || (dst_z.as_ptr() as usize) & 31 != 0 {
            return rowmul_words_avx2(dst_x, dst_z, src_x, src_z, initial_sum);
        }

        let nw = dst_x.len();
        let chunks = nw / 4;

        let lookup = _mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2,
            3, 3, 4,
        );
        let low_mask = _mm256_set1_epi8(0x0f);
        let vzero = _mm256_setzero_si256();

        let mut vsum_pos = _mm256_setzero_si256();
        let mut vsum_nz = _mm256_setzero_si256();

        for i in 0..chunks {
            let off = i * 4;
            let vx1 = _mm256_loadu_si256(src_x.as_ptr().add(off) as *const __m256i);
            let vz1 = _mm256_loadu_si256(src_z.as_ptr().add(off) as *const __m256i);
            let vx2 = _mm256_loadu_si256(dst_x.as_ptr().add(off) as *const __m256i);
            let vz2 = _mm256_loadu_si256(dst_z.as_ptr().add(off) as *const __m256i);

            let vnew_x = _mm256_xor_si256(vx1, vx2);
            let vnew_z = _mm256_xor_si256(vz1, vz2);
            // SAFETY: dst_x and dst_z verified 32-byte aligned at function entry;
            // chunk offsets advance by 32 bytes, preserving alignment.
            _mm256_stream_si256(dst_x.as_mut_ptr().add(off) as *mut __m256i, vnew_x);
            _mm256_stream_si256(dst_z.as_mut_ptr().add(off) as *mut __m256i, vnew_z);

            let vor = _mm256_or_si256(_mm256_or_si256(vx1, vz1), _mm256_or_si256(vx2, vz2));
            if _mm256_testz_si256(vor, vor) != 0 {
                continue;
            }

            let vnonzero = _mm256_and_si256(
                _mm256_and_si256(_mm256_or_si256(vnew_x, vnew_z), _mm256_or_si256(vx1, vz1)),
                _mm256_or_si256(vx2, vz2),
            );

            let vpos = _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_and_si256(vx1, vz1), _mm256_andnot_si256(vx2, vz2)),
                    _mm256_and_si256(_mm256_andnot_si256(vz1, vx1), _mm256_and_si256(vx2, vz2)),
                ),
                _mm256_and_si256(_mm256_andnot_si256(vx1, vz1), _mm256_andnot_si256(vz2, vx2)),
            );

            let lo_p = _mm256_and_si256(vpos, low_mask);
            let hi_p = _mm256_and_si256(_mm256_srli_epi16(vpos, 4), low_mask);
            let cnt_p = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_p),
                _mm256_shuffle_epi8(lookup, hi_p),
            );
            vsum_pos = _mm256_add_epi64(vsum_pos, _mm256_sad_epu8(cnt_p, vzero));

            let lo_n = _mm256_and_si256(vnonzero, low_mask);
            let hi_n = _mm256_and_si256(_mm256_srli_epi16(vnonzero, 4), low_mask);
            let cnt_n = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_n),
                _mm256_shuffle_epi8(lookup, hi_n),
            );
            vsum_nz = _mm256_add_epi64(vsum_nz, _mm256_sad_epu8(cnt_n, vzero));
        }

        _mm_sfence();

        let pos_lo = _mm256_castsi256_si128(vsum_pos);
        let pos_hi = _mm256_extracti128_si256(vsum_pos, 1);
        let pos_sum = _mm_add_epi64(pos_lo, pos_hi);
        let total_pos =
            (_mm_cvtsi128_si64(pos_sum) as u64).wrapping_add(_mm_extract_epi64(pos_sum, 1) as u64);

        let nz_lo = _mm256_castsi256_si128(vsum_nz);
        let nz_hi = _mm256_extracti128_si256(vsum_nz, 1);
        let nz_sum = _mm_add_epi64(nz_lo, nz_hi);
        let total_nz =
            (_mm_cvtsi128_si64(nz_sum) as u64).wrapping_add(_mm_extract_epi64(nz_sum, 1) as u64);

        let mut sum = initial_sum
            .wrapping_add(2u64.wrapping_mul(total_pos))
            .wrapping_sub(total_nz);

        let tail = chunks * 4;
        for w in tail..nw {
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
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rowmul_words_neon(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;

        let nw = dst_x.len();
        let chunks = nw / 2;

        let mut vsum_pos = vdupq_n_u64(0);
        let mut vsum_nz = vdupq_n_u64(0);

        for i in 0..chunks {
            let off = i * 2;
            let vx1 = vld1q_u64(src_x.as_ptr().add(off));
            let vz1 = vld1q_u64(src_z.as_ptr().add(off));
            let vx2 = vld1q_u64(dst_x.as_ptr().add(off));
            let vz2 = vld1q_u64(dst_z.as_ptr().add(off));

            let vnew_x = veorq_u64(vx1, vx2);
            let vnew_z = veorq_u64(vz1, vz2);
            vst1q_u64(dst_x.as_mut_ptr().add(off), vnew_x);
            vst1q_u64(dst_z.as_mut_ptr().add(off), vnew_z);

            let vor = vorrq_u64(vorrq_u64(vx1, vz1), vorrq_u64(vx2, vz2));
            if vgetq_lane_u64(vor, 0) | vgetq_lane_u64(vor, 1) == 0 {
                continue;
            }

            let vnonzero = vandq_u64(
                vandq_u64(vorrq_u64(vnew_x, vnew_z), vorrq_u64(vx1, vz1)),
                vorrq_u64(vx2, vz2),
            );

            // vbicq_u64(a, b) = a & ~b
            let term1 = vandq_u64(vandq_u64(vx1, vz1), vbicq_u64(vz2, vx2));
            let term2 = vandq_u64(vbicq_u64(vx1, vz1), vandq_u64(vx2, vz2));
            let term3 = vandq_u64(vbicq_u64(vz1, vx1), vbicq_u64(vx2, vz2));
            let vpos = vorrq_u64(vorrq_u64(term1, term2), term3);

            let pos_cnt = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(
                vpos,
            )))));
            let nz_cnt = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(
                vnonzero,
            )))));

            vsum_pos = vaddq_u64(vsum_pos, pos_cnt);
            vsum_nz = vaddq_u64(vsum_nz, nz_cnt);
        }

        let total_pos = vgetq_lane_u64(vsum_pos, 0).wrapping_add(vgetq_lane_u64(vsum_pos, 1));
        let total_nz = vgetq_lane_u64(vsum_nz, 0).wrapping_add(vgetq_lane_u64(vsum_nz, 1));

        let mut sum = initial_sum
            .wrapping_add(2u64.wrapping_mul(total_pos))
            .wrapping_sub(total_nz);

        if nw & 1 != 0 {
            let w = nw - 1;
            let x1 = src_x[w];
            let z1 = src_z[w];
            let x2 = dst_x[w];
            let z2 = dst_z[w];
            let new_x = x1 ^ x2;
            let new_z = z1 ^ z2;
            dst_x[w] = new_x;
            dst_z[w] = new_z;
            if (x1 | z1 | x2 | z2) != 0 {
                let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
                let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                sum = sum.wrapping_add(2 * pos.count_ones() as u64);
                sum = sum.wrapping_sub(nonzero.count_ones() as u64);
            }
        }

        sum
    }
}

#[cfg(all(target_arch = "x86_64", any(kani, test)))]
mod rowmul_words_spec {
    pub(super) type Kernel = unsafe fn(&mut [u64], &mut [u64], &[u64], &[u64], u64) -> u64;

    #[repr(align(32))]
    struct Aligned<const N: usize>([u64; N]);

    fn phase_masks(x1: u64, z1: u64, x2: u64, z2: u64) -> (u64, u64) {
        let nonzero = ((x1 ^ x2) | (z1 ^ z2)) & (x1 | z1) & (x2 | z2);
        let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
        (pos, nonzero)
    }

    const NIBBLE_COUNT: [u8; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

    // One word's popcount the way the kernel forms it: a table count per nibble added
    // as bytes, then the eight bytes summed as `_mm256_sad_epu8` sums them against zero.
    pub(super) fn lane_popcount(w: u64) -> u64 {
        w.to_le_bytes()
            .iter()
            .map(|&b| {
                let count = NIBBLE_COUNT[usize::from(b & 15)]
                    .wrapping_add(NIBBLE_COUNT[usize::from(b >> 4)]);
                u64::from(count.abs_diff(0))
            })
            .sum()
    }

    // `rowmul_words_scalar` with its popcounts regrouped the way the AVX2 kernels add
    // them: per nibble through the table, per 64-bit lane across chunks, then lanes 0+2
    // and 1+3, skipping all-zero chunks as the kernels do. The solver then compares
    // adder trees of one shape instead of proving popcount identities.
    fn reference<const N: usize>(
        src: &[[u64; N]; 2],
        dst: &mut [[u64; N]; 2],
        initial_sum: u64,
    ) -> u64 {
        let mut word = |w: usize| {
            let (x1, z1, x2, z2) = (src[0][w], src[1][w], dst[0][w], dst[1][w]);
            dst[0][w] = x1 ^ x2;
            dst[1][w] = z1 ^ z2;
            ((x1 | z1) | (x2 | z2), phase_masks(x1, z1, x2, z2))
        };
        let chunks = N / 4;
        let (mut lanes_pos, mut lanes_nz) = ([0u64; 4], [0u64; 4]);
        for chunk in 0..chunks {
            let words: [_; 4] = std::array::from_fn(|lane| word(4 * chunk + lane));
            if words.iter().fold(0, |any, &(or, _)| any | or) == 0 {
                continue;
            }
            for (lane, &(_, (pos, nonzero))) in words.iter().enumerate() {
                lanes_pos[lane] = lanes_pos[lane].wrapping_add(lane_popcount(pos));
                lanes_nz[lane] = lanes_nz[lane].wrapping_add(lane_popcount(nonzero));
            }
        }
        let total = |l: [u64; 4]| {
            l[0].wrapping_add(l[2])
                .wrapping_add(l[1].wrapping_add(l[3]))
        };
        let mut sum = initial_sum
            .wrapping_add(2u64.wrapping_mul(total(lanes_pos)))
            .wrapping_sub(total(lanes_nz));
        for w in 4 * chunks..N {
            let (or, (pos, nonzero)) = word(w);
            if or == 0 {
                continue;
            }
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
        sum
    }

    /// Check an AVX2 rowmul kernel against the regrouped reference on `[x, z]` rows of
    /// `N` words. The destination rows are 32-byte aligned so the non-temporal kernel
    /// takes its streaming path. Callers establish AVX2 before calling, except under
    /// Kani, which runs no CPU.
    pub(super) fn check<const N: usize>(
        kernel: Kernel,
        src: [[u64; N]; 2],
        dst: [[u64; N]; 2],
        initial_sum: u64,
    ) {
        let mut want = dst;
        let want_sum = reference(&src, &mut want, initial_sum);
        let (mut got_x, mut got_z) = (Aligned(dst[0]), Aligned(dst[1]));
        // SAFETY: all four rows hold exactly `N` words; AVX2 is established by the caller.
        let got_sum = unsafe { kernel(&mut got_x.0, &mut got_z.0, &src[0], &src[1], initial_sum) };
        assert_eq!(got_sum, want_sum);
        for (got, want) in [(&got_x.0, &want[0]), (&got_z.0, &want[1])] {
            for (g, w) in got.iter().zip(want) {
                assert_eq!(g, w);
            }
        }
    }

    /// Check that the regrouping leaves `rowmul_words_scalar`'s result unchanged.
    #[cfg(test)]
    pub(super) fn check_reference<const N: usize>(
        src: [[u64; N]; 2],
        dst: [[u64; N]; 2],
        initial_sum: u64,
    ) {
        let mut want = dst;
        let want_sum = reference(&src, &mut want, initial_sum);
        let [mut x, mut z] = dst;
        let sum = super::rowmul_words_scalar(&mut x, &mut z, &src[0], &src[1], initial_sum);
        assert_eq!(sum, want_sum);
        assert_eq!([x, z], want);
    }

    #[cfg(kani)]
    mod proofs {
        macro_rules! rowmul_words_matches_reference {
            ($kernel:ident => $($name:ident: $nw:literal),* $(,)?) => {$(
                #[kani::proof]
                #[kani::unwind(33)]
                #[kani::stub(
                    core::arch::x86_64::_mm256_shuffle_epi8,
                    crate::backend::avx2_model::shuffle_epi8
                )]
                #[kani::stub(
                    core::arch::x86_64::_mm256_sad_epu8,
                    crate::backend::avx2_model::sad_epu8
                )]
                #[kani::stub(
                    core::arch::x86_64::_mm256_testz_si256,
                    crate::backend::avx2_model::testz_si256
                )]
                #[kani::stub(core::arch::x86_64::_mm_sfence, crate::backend::avx2_model::sfence)]
                #[kani::stub(
                    core::arch::x86_64::_mm256_stream_si256,
                    crate::backend::avx2_model::stream_si256
                )]
                fn $name() {
                    super::check::<$nw>(
                        super::super::$kernel,
                        kani::any(),
                        kani::any(),
                        kani::any(),
                    );
                }
            )*};
        }

        rowmul_words_matches_reference!(
            rowmul_words_avx2 =>
            rowmul_words_avx2_nw0: 0,
            rowmul_words_avx2_nw3: 3,
            rowmul_words_avx2_nw5: 5,
        );

        rowmul_words_matches_reference!(
            rowmul_words_avx2_nt =>
            rowmul_words_avx2_nt_nw4: 4,
        );
    }
}

#[cfg(all(target_arch = "aarch64", any(kani, test)))]
mod rowmul_words_neon_spec {
    pub(super) type Kernel = unsafe fn(&mut [u64], &mut [u64], &[u64], &[u64], u64) -> u64;

    fn phase_masks(x1: u64, z1: u64, x2: u64, z2: u64) -> (u64, u64) {
        let nonzero = ((x1 ^ x2) | (z1 ^ z2)) & (x1 | z1) & (x2 | z2);
        let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
        (pos, nonzero)
    }

    // One word's popcount the way the kernel forms it: `vcntq_u8` per byte, then three
    // pairwise widening adds (`vpaddlq_u8`, `vpaddlq_u16`, `vpaddlq_u32`) up to the lane.
    pub(super) fn lane_popcount(w: u64) -> u64 {
        let bytes = w.to_le_bytes().map(|b| b.count_ones() as u8);
        let halves: [u16; 4] =
            std::array::from_fn(|i| u16::from(bytes[2 * i]) + u16::from(bytes[2 * i + 1]));
        let quads: [u32; 2] =
            std::array::from_fn(|i| u32::from(halves[2 * i]) + u32::from(halves[2 * i + 1]));
        u64::from(quads[0]) + u64::from(quads[1])
    }

    // `rowmul_words_scalar` with its popcounts regrouped the way the NEON kernel adds
    // them: per byte through `vcntq_u8`, per 64-bit lane across chunks, then lane 0 plus
    // lane 1, skipping all-zero chunks as the kernel does. The solver then compares
    // adder trees of one shape instead of proving popcount identities.
    fn reference<const N: usize>(
        src: &[[u64; N]; 2],
        dst: &mut [[u64; N]; 2],
        initial_sum: u64,
    ) -> u64 {
        let mut word = |w: usize| {
            let (x1, z1, x2, z2) = (src[0][w], src[1][w], dst[0][w], dst[1][w]);
            dst[0][w] = x1 ^ x2;
            dst[1][w] = z1 ^ z2;
            ((x1 | z1) | (x2 | z2), phase_masks(x1, z1, x2, z2))
        };
        let chunks = N / 2;
        let (mut lanes_pos, mut lanes_nz) = ([0u64; 2], [0u64; 2]);
        for chunk in 0..chunks {
            let words: [_; 2] = std::array::from_fn(|lane| word(2 * chunk + lane));
            if words[0].0 | words[1].0 == 0 {
                continue;
            }
            for (lane, &(_, (pos, nonzero))) in words.iter().enumerate() {
                lanes_pos[lane] = lanes_pos[lane].wrapping_add(lane_popcount(pos));
                lanes_nz[lane] = lanes_nz[lane].wrapping_add(lane_popcount(nonzero));
            }
        }
        let total = |l: [u64; 2]| l[0].wrapping_add(l[1]);
        let mut sum = initial_sum
            .wrapping_add(2u64.wrapping_mul(total(lanes_pos)))
            .wrapping_sub(total(lanes_nz));
        for w in 2 * chunks..N {
            let (or, (pos, nonzero)) = word(w);
            if or == 0 {
                continue;
            }
            sum = sum.wrapping_add(2 * pos.count_ones() as u64);
            sum = sum.wrapping_sub(nonzero.count_ones() as u64);
        }
        sum
    }

    /// Check the NEON rowmul kernel against the regrouped reference on `[x, z]` rows of
    /// `N` words.
    pub(super) fn check<const N: usize>(
        kernel: Kernel,
        src: [[u64; N]; 2],
        dst: [[u64; N]; 2],
        initial_sum: u64,
    ) {
        let mut want = dst;
        let want_sum = reference(&src, &mut want, initial_sum);
        let [mut got_x, mut got_z] = dst;
        // SAFETY: all four rows hold exactly `N` words; NEON is baseline on aarch64.
        let got_sum = unsafe { kernel(&mut got_x, &mut got_z, &src[0], &src[1], initial_sum) };
        assert_eq!(got_sum, want_sum);
        for (got, want) in [(&got_x, &want[0]), (&got_z, &want[1])] {
            for (g, w) in got.iter().zip(want) {
                assert_eq!(g, w);
            }
        }
    }

    /// Check that the regrouping leaves `rowmul_words_scalar`'s result unchanged.
    #[cfg(test)]
    pub(super) fn check_reference<const N: usize>(
        src: [[u64; N]; 2],
        dst: [[u64; N]; 2],
        initial_sum: u64,
    ) {
        let mut want = dst;
        let want_sum = reference(&src, &mut want, initial_sum);
        let [mut x, mut z] = dst;
        let sum = super::rowmul_words_scalar(&mut x, &mut z, &src[0], &src[1], initial_sum);
        assert_eq!(sum, want_sum);
        assert_eq!([x, z], want);
    }

    #[cfg(kani)]
    mod proofs {
        macro_rules! rowmul_words_matches_reference {
            ($($name:ident: $nw:literal),* $(,)?) => {$(
                #[kani::proof]
                #[kani::unwind(17)]
                #[kani::stub(core::arch::aarch64::vcntq_u8, crate::backend::neon_model::cntq_u8)]
                #[kani::stub(
                    core::arch::aarch64::vpaddlq_u8,
                    crate::backend::neon_model::paddlq_u8
                )]
                #[kani::stub(
                    core::arch::aarch64::vpaddlq_u16,
                    crate::backend::neon_model::paddlq_u16
                )]
                #[kani::stub(
                    core::arch::aarch64::vpaddlq_u32,
                    crate::backend::neon_model::paddlq_u32
                )]
                fn $name() {
                    super::check::<$nw>(
                        super::super::rowmul_words_neon,
                        kani::any(),
                        kani::any(),
                        kani::any(),
                    );
                }
            )*};
        }

        rowmul_words_matches_reference!(
            rowmul_words_neon_nw0: 0,
            rowmul_words_neon_nw1: 1,
            rowmul_words_neon_nw3: 3,
        );
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::rowmul_words_neon;
    use super::rowmul_words_neon_spec::{Kernel, check, check_reference, lane_popcount};
    use rand::{RngExt, SeedableRng};

    #[test]
    fn lane_popcount_matches_count_ones() {
        for shift in (0..64).step_by(8) {
            for b in 0..=255u64 {
                let w = b << shift;
                assert_eq!(lane_popcount(w), u64::from(w.count_ones()));
                assert_eq!(lane_popcount(!w), u64::from((!w).count_ones()));
            }
        }
    }

    #[test]
    fn rowmul_words_neon_matches_scalar() {
        fn run<const N: usize>(rng: &mut rand_chacha::ChaCha8Rng, kernel: Kernel) {
            for _ in 0..64 {
                let mut src: [[u64; N]; 2] = rng.random();
                let mut dst: [[u64; N]; 2] = rng.random();
                if rng.random_bool(0.25) {
                    dst = src;
                }
                if N > 0 && rng.random_bool(0.25) {
                    let start = rng.random_range(0..N);
                    let end = (start + 2).min(N);
                    for row in src.iter_mut().chain(&mut dst) {
                        row[start..end].fill(0);
                    }
                }
                let initial_sum = rng.random();
                check_reference(src, dst, initial_sum);
                check(kernel, src, dst, initial_sum);
            }
        }

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let neon: Kernel = rowmul_words_neon;
        run::<0>(&mut rng, neon);
        run::<1>(&mut rng, neon);
        run::<2>(&mut rng, neon);
        run::<3>(&mut rng, neon);
        run::<4>(&mut rng, neon);
        run::<5>(&mut rng, neon);
        run::<8>(&mut rng, neon);
        run::<9>(&mut rng, neon);
        run::<16>(&mut rng, neon);
        run::<17>(&mut rng, neon);
        run::<35>(&mut rng, neon);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::rowmul_words_spec::{Kernel, check, check_reference, lane_popcount};
    use super::{rowmul_words_avx2, rowmul_words_avx2_nt};
    use rand::{RngExt, SeedableRng};

    #[test]
    fn lane_popcount_matches_count_ones() {
        for shift in (0..64).step_by(8) {
            for b in 0..=255u64 {
                let w = b << shift;
                assert_eq!(lane_popcount(w), u64::from(w.count_ones()));
                assert_eq!(lane_popcount(!w), u64::from((!w).count_ones()));
            }
        }
    }

    #[test]
    fn rowmul_words_avx2_matches_scalar() {
        fn run<const N: usize>(rng: &mut rand_chacha::ChaCha8Rng, kernel: Kernel) {
            for _ in 0..64 {
                let mut src: [[u64; N]; 2] = rng.random();
                let mut dst: [[u64; N]; 2] = rng.random();
                if rng.random_bool(0.25) {
                    dst = src;
                }
                if N > 0 && rng.random_bool(0.25) {
                    let start = rng.random_range(0..N);
                    let end = (start + 4).min(N);
                    for row in src.iter_mut().chain(&mut dst) {
                        row[start..end].fill(0);
                    }
                }
                let initial_sum = rng.random();
                check_reference(src, dst, initial_sum);
                if is_x86_feature_detected!("avx2") {
                    check(kernel, src, dst, initial_sum);
                }
            }
        }

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let avx2: Kernel = rowmul_words_avx2;
        let nt: Kernel = rowmul_words_avx2_nt;
        run::<0>(&mut rng, avx2);
        run::<3>(&mut rng, avx2);
        run::<4>(&mut rng, avx2);
        run::<5>(&mut rng, avx2);
        run::<7>(&mut rng, avx2);
        run::<8>(&mut rng, avx2);
        run::<9>(&mut rng, avx2);
        run::<13>(&mut rng, avx2);
        run::<16>(&mut rng, nt);
        run::<17>(&mut rng, nt);
        run::<19>(&mut rng, nt);
        run::<24>(&mut rng, nt);
        run::<35>(&mut rng, nt);
    }
}
