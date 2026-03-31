#[inline(always)]
pub(crate) unsafe fn xor_words(dst: *mut u64, src: *const u64, len: usize) {
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
    if nw >= 4 && has_avx2() {
        // SAFETY: AVX2 detected at runtime; slices have equal lengths (debug_assert above).
        return unsafe { rowmul_words_avx2(dst_x, dst_z, src_x, src_z, initial_sum) };
    }
    #[cfg(target_arch = "aarch64")]
    if nw >= 2 {
        // SAFETY: NEON is mandatory on aarch64; slices have equal lengths.
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
    use std::arch::x86_64::*;

    let nw = dst_x.len();
    let chunks = nw / 4;

    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
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

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
#[target_feature(enable = "neon")]
unsafe fn rowmul_words_neon(
    dst_x: &mut [u64],
    dst_z: &mut [u64],
    src_x: &[u64],
    src_z: &[u64],
    initial_sum: u64,
) -> u64 {
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
