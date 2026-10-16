//! Plain-Rust models of the NEON intrinsics Kani cannot execute, for `#[kani::stub]`.
//! stdarch lowers each one to an LLVM intrinsic; every model keeps the signature of the
//! intrinsic it replaces.

use std::arch::aarch64::{uint8x16_t, uint16x8_t, uint32x4_t, uint64x2_t};

fn bytes(v: uint8x16_t) -> [u8; 16] {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn halves(v: uint16x8_t) -> [u16; 8] {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn quads(v: uint32x4_t) -> [u32; 4] {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_bytes(v: [u8; 16]) -> uint8x16_t {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_halves(v: [u16; 8]) -> uint16x8_t {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_quads(v: [u32; 4]) -> uint32x4_t {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn from_words(v: [u64; 2]) -> uint64x2_t {
    // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
    unsafe { std::mem::transmute(v) }
}

fn pairwise_widen<const N: usize, const H: usize, A, B>(a: [A; N]) -> [B; H]
where
    A: Copy + Into<B>,
    B: std::ops::Add<Output = B>,
{
    std::array::from_fn(|i| {
        let (lo, hi): (B, B) = (a[2 * i].into(), a[2 * i + 1].into());
        lo + hi
    })
}

pub(crate) fn cntq_u8(a: uint8x16_t) -> uint8x16_t {
    from_bytes(bytes(a).map(|b| b.count_ones() as u8))
}

/// Each output lane holds the widened sum of the two input lanes it covers.
pub(crate) fn paddlq_u8(a: uint8x16_t) -> uint16x8_t {
    from_halves(pairwise_widen(bytes(a)))
}

/// Each output lane holds the widened sum of the two input lanes it covers.
pub(crate) fn paddlq_u16(a: uint16x8_t) -> uint32x4_t {
    from_quads(pairwise_widen(halves(a)))
}

/// Each output lane holds the widened sum of the two input lanes it covers.
pub(crate) fn paddlq_u32(a: uint32x4_t) -> uint64x2_t {
    from_words(pairwise_widen(quads(a)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use std::arch::aarch64::*;

    fn words(v: uint64x2_t) -> [u64; 2] {
        // SAFETY: both types are 16 bytes and every bit pattern is valid in each.
        unsafe { std::mem::transmute(v) }
    }

    #[test]
    fn models_match_neon_intrinsics() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let mut cases: Vec<[u8; 16]> = (0..=255u8)
            .map(|c| std::array::from_fn(|i| c.wrapping_add(i as u8)))
            .collect();
        cases.extend((0..256).map(|_| rng.random()));
        cases.push([0; 16]);
        cases.push([0xff; 16]);
        cases.push(std::array::from_fn(|i| if i % 2 == 0 { 0xff } else { 0 }));

        for case in cases {
            let a = from_bytes(case);
            let halves_in = from_halves(halves(paddlq_u8(a)).map(|h| h.wrapping_mul(0x1fff)));
            let quads_in =
                from_quads(quads(paddlq_u16(halves_in)).map(|q| q.wrapping_mul(0x1fff_ffff)));
            assert_eq!(bytes(cntq_u8(a)), bytes(vcntq_u8(a)));
            assert_eq!(halves(paddlq_u8(a)), halves(vpaddlq_u8(a)));
            assert_eq!(quads(paddlq_u16(halves_in)), quads(vpaddlq_u16(halves_in)));
            assert_eq!(words(paddlq_u32(quads_in)), words(vpaddlq_u32(quads_in)));
        }
    }
}
