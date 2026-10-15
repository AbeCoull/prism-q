//! Plain-Rust models of the AVX2 intrinsics Kani cannot execute, for `#[kani::stub]`.
//! stdarch lowers each one to an LLVM intrinsic, inline assembly, or `simd_reduce_or`;
//! every model keeps the signature of the intrinsic it replaces.

use std::arch::x86_64::__m256i;

fn bytes(v: __m256i) -> [u8; 32] {
    // SAFETY: both types are 32 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn words(v: __m256i) -> [u64; 4] {
    // SAFETY: both types are 32 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_bytes(v: [u8; 32]) -> __m256i {
    // SAFETY: both types are 32 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_words(v: [u64; 4]) -> __m256i {
    // SAFETY: both types are 32 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

/// Each byte of `b` with its top bit set yields zero; otherwise its low four bits pick a
/// byte of `a` from the same 128-bit lane.
pub(crate) fn shuffle_epi8(a: __m256i, b: __m256i) -> __m256i {
    let (a, b) = (bytes(a), bytes(b));
    let mut out = [0u8; 32];
    for (i, (o, &c)) in out.iter_mut().zip(&b).enumerate() {
        if c & 0x80 == 0 {
            *o = a[(i & 16) | usize::from(c & 15)];
        }
    }
    from_bytes(out)
}

/// Each 64-bit lane holds the sum of absolute byte differences over its eight bytes.
pub(crate) fn sad_epu8(a: __m256i, b: __m256i) -> __m256i {
    let (a, b) = (bytes(a), bytes(b));
    let (a, b) = (a.as_chunks::<8>().0, b.as_chunks::<8>().0);
    let mut out = [0u64; 4];
    for (lane, (x, y)) in out.iter_mut().zip(a.iter().zip(b)) {
        *lane = x
            .iter()
            .zip(y)
            .map(|(p, q)| u64::from(p.abs_diff(*q)))
            .sum();
    }
    from_words(out)
}

pub(crate) fn testz_si256(a: __m256i, b: __m256i) -> i32 {
    let (a, b) = (words(a), words(b));
    i32::from((a[0] & b[0]) | (a[1] & b[1]) | (a[2] & b[2]) | (a[3] & b[3]) == 0)
}

/// The ordering a non-temporal store needs is invisible to a single-threaded model,
/// so the fence is empty and [`stream_si256`] is a plain aligned store.
#[cfg(kani)]
pub(crate) fn sfence() {}

pub(crate) unsafe fn stream_si256(mem_addr: *mut __m256i, a: __m256i) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe { mem_addr.write(a) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use std::arch::x86_64::*;

    #[repr(align(32))]
    struct Slot(__m256i);

    #[test]
    fn models_match_avx2_intrinsics() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut cases: Vec<([u8; 32], [u8; 32])> = (0..=255u8)
            .map(|c| {
                (
                    rng.random(),
                    std::array::from_fn(|i| c.wrapping_add(i as u8)),
                )
            })
            .collect();
        cases.extend((0..256).map(|_| (rng.random(), rng.random())));
        cases.push(([0; 32], [0xff; 32]));
        cases.push(([0xff; 32], [0; 32]));

        for (a, b) in cases {
            let (a, b) = (from_bytes(a), from_bytes(b));
            let complement = from_words(words(b).map(|w| !w));
            let (mut modeled, mut streamed) = (Slot(from_words([0; 4])), Slot(from_words([0; 4])));
            // SAFETY: AVX2 detected; both slots are 32-byte aligned and writable.
            let (shuffle, sad, testz, testz_complement) = unsafe {
                stream_si256(&mut modeled.0, a);
                _mm256_stream_si256(&mut streamed.0, a);
                _mm_sfence();
                (
                    _mm256_shuffle_epi8(a, b),
                    _mm256_sad_epu8(a, b),
                    _mm256_testz_si256(a, b),
                    _mm256_testz_si256(b, complement),
                )
            };
            assert_eq!(bytes(shuffle_epi8(a, b)), bytes(shuffle));
            assert_eq!(words(sad_epu8(a, b)), words(sad));
            assert_eq!(testz_si256(a, b), testz);
            assert_eq!(testz_si256(b, complement), testz_complement);
            assert_eq!(bytes(modeled.0), bytes(streamed.0));
        }
    }
}
