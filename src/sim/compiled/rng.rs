use rand::RngCore;
use rand_chacha::ChaCha8Rng;

pub(super) struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    #[inline(always)]
    pub(super) fn from_seeds(s: [u64; 4]) -> Self {
        Self { s }
    }

    #[inline(always)]
    pub(super) fn from_chacha(rng: &mut ChaCha8Rng) -> Self {
        Self {
            s: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ],
        }
    }

    #[inline(always)]
    pub(super) fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) struct Xoshiro256PlusPlusX4 {
    s0: std::arch::x86_64::__m256i,
    s1: std::arch::x86_64::__m256i,
    s2: std::arch::x86_64::__m256i,
    s3: std::arch::x86_64::__m256i,
}

#[cfg(target_arch = "x86_64")]
impl Xoshiro256PlusPlusX4 {
    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller must ensure AVX2 is available (checked via is_x86_feature_detected)
    pub(super) unsafe fn from_scalar(rng: &mut Xoshiro256PlusPlus) -> Self {
        use std::arch::x86_64::*;
        let mut seeds = [0u64; 16];
        for s in &mut seeds {
            *s = rng.next_u64();
        }
        Self {
            s0: _mm256_set_epi64x(
                seeds[12] as i64,
                seeds[8] as i64,
                seeds[4] as i64,
                seeds[0] as i64,
            ),
            s1: _mm256_set_epi64x(
                seeds[13] as i64,
                seeds[9] as i64,
                seeds[5] as i64,
                seeds[1] as i64,
            ),
            s2: _mm256_set_epi64x(
                seeds[14] as i64,
                seeds[10] as i64,
                seeds[6] as i64,
                seeds[2] as i64,
            ),
            s3: _mm256_set_epi64x(
                seeds[15] as i64,
                seeds[11] as i64,
                seeds[7] as i64,
                seeds[3] as i64,
            ),
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller must ensure AVX2 is available (checked via is_x86_feature_detected)
    pub(super) unsafe fn next_m256i(&mut self) -> std::arch::x86_64::__m256i {
        use std::arch::x86_64::*;

        macro_rules! rotl64_avx2 {
            ($x:expr, $k:literal) => {
                _mm256_or_si256(_mm256_slli_epi64($x, $k), _mm256_srli_epi64($x, 64 - $k))
            };
        }

        let sum = _mm256_add_epi64(self.s0, self.s3);
        let result = _mm256_add_epi64(rotl64_avx2!(sum, 23), self.s0);

        let t = _mm256_slli_epi64(self.s1, 17);

        self.s2 = _mm256_xor_si256(self.s2, self.s0);
        self.s3 = _mm256_xor_si256(self.s3, self.s1);
        self.s1 = _mm256_xor_si256(self.s1, self.s2);
        self.s0 = _mm256_xor_si256(self.s0, self.s3);
        self.s2 = _mm256_xor_si256(self.s2, t);
        self.s3 = rotl64_avx2!(self.s3, 45);

        result
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) struct Xoshiro256PlusPlusX2 {
    s0: std::arch::aarch64::uint64x2_t,
    s1: std::arch::aarch64::uint64x2_t,
    s2: std::arch::aarch64::uint64x2_t,
    s3: std::arch::aarch64::uint64x2_t,
}

#[cfg(target_arch = "aarch64")]
impl Xoshiro256PlusPlusX2 {
    #[inline]
    // SAFETY: NEON is baseline on aarch64; caller provides valid scalar RNG
    pub(super) unsafe fn from_scalar(rng: &mut Xoshiro256PlusPlus) -> Self {
        use std::arch::aarch64::*;
        let mut seeds = [0u64; 8];
        for s in &mut seeds {
            *s = rng.next_u64();
        }
        Self {
            s0: vld1q_u64([seeds[0], seeds[4]].as_ptr()),
            s1: vld1q_u64([seeds[1], seeds[5]].as_ptr()),
            s2: vld1q_u64([seeds[2], seeds[6]].as_ptr()),
            s3: vld1q_u64([seeds[3], seeds[7]].as_ptr()),
        }
    }

    #[inline]
    // SAFETY: NEON is baseline on aarch64
    pub(super) unsafe fn next_uint64x2(&mut self) -> std::arch::aarch64::uint64x2_t {
        use std::arch::aarch64::*;

        macro_rules! rotl64_neon {
            ($x:expr, $k:literal) => {
                vorrq_u64(vshlq_n_u64($x, $k), vshrq_n_u64($x, 64 - $k))
            };
        }

        let sum = vaddq_u64(self.s0, self.s3);
        let result = vaddq_u64(rotl64_neon!(sum, 23), self.s0);

        let t = vshlq_n_u64(self.s1, 17);

        self.s2 = veorq_u64(self.s2, self.s0);
        self.s3 = veorq_u64(self.s3, self.s1);
        self.s1 = veorq_u64(self.s1, self.s2);
        self.s0 = veorq_u64(self.s0, self.s3);
        self.s2 = veorq_u64(self.s2, t);
        self.s3 = rotl64_neon!(self.s3, 45);

        result
    }
}
