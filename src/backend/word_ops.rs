#[inline(always)]
pub(crate) fn xor_words(dst: &mut [u64], src: &[u64]) {
    let len = dst.len().min(src.len());

    #[cfg(target_arch = "x86_64")]
    if len >= 4 && has_avx2() {
        // SAFETY: AVX2 support is checked above. Slices provide valid pointers
        // for len u64 values.
        unsafe { xor_words_ptr(dst.as_mut_ptr(), src.as_ptr(), len) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    if len >= 2 {
        // SAFETY: NEON is available on supported aarch64 targets. Slices
        // provide valid pointers for len u64 values.
        unsafe { xor_words_ptr(dst.as_mut_ptr(), src.as_ptr(), len) };
        return;
    }

    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= s;
    }
}

#[inline(always)]
pub(crate) unsafe fn xor_words_ptr(dst: *mut u64, src: *const u64, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: AVX2 support is checked above. The caller guarantees both
        // pointers are valid for len u64 values.
        unsafe { xor_words_avx2(dst, src, len) };
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is available on supported aarch64 targets. The caller
        // guarantees both pointers are valid for len u64 values.
        unsafe { xor_words_neon(dst, src, len) };
        return;
    }

    #[allow(unreachable_code)]
    for i in 0..len {
        // SAFETY: The caller guarantees both pointers are valid for len u64 values.
        unsafe { *dst.add(i) ^= *src.add(i) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: *mut u64, src: *const u64, len: usize) {
    use std::arch::x86_64::*;

    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        // SAFETY: The caller guarantees pointers are valid for len u64 values.
        // Unaligned loads and stores are used for arbitrary slice alignment.
        unsafe {
            let d = _mm256_loadu_si256(dst.add(off) as *const __m256i);
            let s = _mm256_loadu_si256(src.add(off) as *const __m256i);
            _mm256_storeu_si256(dst.add(off) as *mut __m256i, _mm256_xor_si256(d, s));
        }
    }
    let tail = chunks * 4;
    for i in tail..len {
        // SAFETY: The caller guarantees both pointers are valid for len u64 values.
        unsafe { *dst.add(i) ^= *src.add(i) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_words_neon(dst: *mut u64, src: *const u64, len: usize) {
    use std::arch::aarch64::*;

    let chunks = len / 2;
    for i in 0..chunks {
        let off = i * 2;
        // SAFETY: The caller guarantees pointers are valid for len u64 values.
        unsafe {
            let d = vld1q_u64(dst.add(off));
            let s = vld1q_u64(src.add(off));
            vst1q_u64(dst.add(off), veorq_u64(d, s));
        }
    }
    if len & 1 != 0 {
        // SAFETY: The caller guarantees both pointers are valid for len u64 values.
        unsafe { *dst.add(len - 1) ^= *src.add(len - 1) };
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx2() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| is_x86_feature_detected!("avx2"))
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
pub(crate) fn has_avx2() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dispatcher and the AVX2 kernel must both equal a plain scalar XOR. The
    /// length is not a multiple of the SIMD stride, so the remainder loop runs too.
    #[test]
    fn xor_words_matches_scalar_reference() {
        let src: Vec<u64> = (0..37)
            .map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i + 1))
            .collect();
        let dst0: Vec<u64> = (0..37)
            .map(|i| 0xD1B5_4A32_D192_ED03u64.wrapping_add(i))
            .collect();
        let expected: Vec<u64> = dst0.iter().zip(&src).map(|(d, s)| d ^ s).collect();

        let mut dst = dst0.clone();
        xor_words(&mut dst, &src);
        assert_eq!(dst, expected, "xor_words dispatcher");

        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            let mut dst = dst0.clone();
            // SAFETY: equal lengths; avx2 verified above.
            unsafe { xor_words_avx2(dst.as_mut_ptr(), src.as_ptr(), src.len()) };
            assert_eq!(dst, expected, "xor_words_avx2 kernel");
        }
    }
}
