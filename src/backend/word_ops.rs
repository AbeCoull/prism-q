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
