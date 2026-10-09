//! Bulk randomness for compiled sampling: samplers seed from `ChaCha8Rng` for
//! per-seed determinism, then draw bulk words from xoshiro256++, which is far
//! cheaper at one u64 per random bit per 64-shot batch.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

pub(crate) struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    #[cfg(feature = "parallel")]
    #[inline(always)]
    pub(crate) fn from_seeds(s: [u64; 4]) -> Self {
        Self { s }
    }

    #[inline(always)]
    pub(crate) fn from_chacha(rng: &mut ChaCha8Rng) -> Self {
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
    pub(crate) fn next_u64(&mut self) -> u64 {
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

    #[inline(always)]
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sample Binomial(n, p): inversion for small `n * p`, BTPE otherwise.
#[inline(never)]
pub(crate) fn binomial_sample(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64) -> usize {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }

    let (pp, invert) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };
    let nf = n as f64;
    let np = nf * pp;

    let result = if np < 10.0 {
        binomial_inversion(rng, n, pp, nf)
    } else {
        binomial_btpe(rng, n, pp, nf, np)
    };

    if invert { n - result } else { result }
}

fn binomial_inversion(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64, _nf: f64) -> usize {
    let q = 1.0 - p;
    let s = p / q;
    let a = ((n + 1) as f64) * s;
    let mut r = match i32::try_from(n) {
        Ok(n) => q.powi(n),
        Err(_) => 0.0,
    };
    if r <= 0.0 {
        r = (-((n as f64) * p)).exp();
    }
    let mut u = rng.next_f64();
    let mut x = 0usize;

    loop {
        if u <= r {
            return x;
        }
        u -= r;
        x += 1;
        if x > n {
            return n;
        }
        r *= (a / x as f64) - s;
    }
}

/// BTPE rejection sampler (Kachitvichyanukul and Schmeiser, 1988), for
/// `p <= 0.5`. The triangle region accepts outright; the parallelograms and the
/// exponential tails carry their own transformed `v` into the acceptance test.
fn binomial_btpe(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64, nf: f64, np: f64) -> usize {
    let q = 1.0 - p;
    let npq = np * q;

    let fm = np + p;
    let m = fm as usize;
    let mf = m as f64;

    let p1 = (2.195 * npq.sqrt() - 4.6 * q).floor() + 0.5;
    let xm = mf + 0.5;
    let xl = xm - p1;
    let xr = xm + p1;
    let c = 0.134 + 20.5 / (15.3 + mf);

    let al = (fm - xl) / (fm - xl * p);
    let lambda_l = al * (1.0 + 0.5 * al);
    let ar = (xr - fm) / (xr * q);
    let lambda_r = ar * (1.0 + 0.5 * ar);
    let p2 = p1 * (1.0 + 2.0 * c);
    let p3 = p2 + c / lambda_l;
    let p4 = p3 + c / lambda_r;

    loop {
        let u = rng.next_f64() * p4;
        let mut v = rng.next_f64();

        if u <= p1 {
            return (xm - p1 * v + u) as usize;
        }

        let iy: usize;
        if u <= p2 {
            let x = xl + (u - p1) / c;
            v = v * c + 1.0 - (mf - x + 0.5).abs() / p1;
            if v > 1.0 || x < 0.0 {
                continue;
            }
            iy = x as usize;
        } else if u <= p3 {
            if v == 0.0 {
                continue;
            }
            let y = (xl + v.ln() / lambda_l).floor();
            if y < 0.0 {
                continue;
            }
            iy = y as usize;
            v *= (u - p2) * lambda_l;
        } else {
            if v == 0.0 {
                continue;
            }
            let y = (xr - v.ln() / lambda_r).floor();
            if y > nf {
                continue;
            }
            iy = y as usize;
            v *= (u - p3) * lambda_r;
        }
        if iy > n {
            continue;
        }

        let k = iy.abs_diff(m);
        let kf = k as f64;

        if k <= 20 || kf >= npq / 2.0 - 1.0 {
            // Explicit ratio f(y) / f(m) by the recurrence.
            let s = p / q;
            let a = s * (nf + 1.0);
            let mut f = 1.0;
            if m < iy {
                for i in (m + 1)..=iy {
                    f *= a / i as f64 - s;
                }
            } else if m > iy {
                for i in (iy + 1)..=m {
                    f /= a / i as f64 - s;
                }
            }
            if v <= f {
                return iy;
            }
            continue;
        }

        // Squeeze on log(v), then the Stirling bound on log(f(y) / f(m)).
        let rho = (kf / npq) * ((kf * (kf / 3.0 + 0.625) + 1.0 / 6.0) / npq + 0.5);
        let t = -kf * kf / (2.0 * npq);
        let log_v = v.ln();
        if log_v < t - rho {
            return iy;
        }
        if log_v > t + rho {
            continue;
        }

        let x1 = (iy + 1) as f64;
        let f1 = mf + 1.0;
        let z = nf + 1.0 - mf;
        let w = nf - iy as f64 + 1.0;
        let stirling = |x: f64| {
            let x_sq = x * x;
            (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x_sq) / x_sq) / x_sq) / x_sq)
                / x
                / 166320.0
        };

        let bound = xm * (f1 / x1).ln()
            + (nf - mf + 0.5) * (z / w).ln()
            + (iy as f64 - mf) * (w * p / (x1 * q)).ln()
            + stirling(f1)
            + stirling(z)
            + stirling(x1)
            + stirling(w);

        if log_v <= bound {
            return iy;
        }
    }
}

/// Four independent xoshiro256++ streams seeded from sixteen draws of the scalar
/// generator, lane `j` from draws `4j..4j + 4`. The AVX2 and NEON generators lay the
/// same streams across their vector lanes, so every sampler path draws the word for
/// shot word `w` from lane `w % 4` and the stream is the same on every ISA.
pub(super) struct Xoshiro256PlusPlusLanes([Xoshiro256PlusPlus; 4]);

impl Xoshiro256PlusPlusLanes {
    #[inline(always)]
    pub(super) fn from_scalar(rng: &mut Xoshiro256PlusPlus) -> Self {
        Self(std::array::from_fn(|_| Xoshiro256PlusPlus {
            s: [
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ],
        }))
    }

    #[inline(always)]
    pub(super) fn next_u64(&mut self, lane: usize) -> u64 {
        self.0[lane].next_u64()
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
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
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
    }

    #[inline]
    // SAFETY: NEON is baseline on aarch64
    pub(super) unsafe fn next_uint64x2(&mut self) -> std::arch::aarch64::uint64x2_t {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> Xoshiro256PlusPlus {
        let mut c = ChaCha8Rng::seed_from_u64(42);
        Xoshiro256PlusPlus::from_chacha(&mut c)
    }

    #[test]
    fn next_f64_in_unit_interval() {
        let mut r = rng();
        for _ in 0..10_000 {
            let x = r.next_f64();
            assert!((0.0..1.0).contains(&x), "value out of range: {x}");
        }
    }

    #[test]
    fn next_u64_changes() {
        let mut r = rng();
        let a = r.next_u64();
        let b = r.next_u64();
        assert_ne!(a, b);
    }

    #[test]
    fn binomial_sample_trivial_edges() {
        let mut r = rng();
        assert_eq!(binomial_sample(&mut r, 0, 0.5), 0);
        assert_eq!(binomial_sample(&mut r, 100, 0.0), 0);
        assert_eq!(binomial_sample(&mut r, 100, -0.5), 0);
        assert_eq!(binomial_sample(&mut r, 100, 1.0), 100);
        assert_eq!(binomial_sample(&mut r, 100, 1.5), 100);
    }

    #[test]
    fn binomial_sample_inversion_path() {
        let mut r = rng();
        let n = 50;
        let p = 0.05;
        let mut sum = 0usize;
        let trials = 4_000;
        for _ in 0..trials {
            let k = binomial_sample(&mut r, n, p);
            assert!(k <= n);
            sum += k;
        }
        let mean = sum as f64 / trials as f64;
        let expected = n as f64 * p;
        assert!(
            (mean - expected).abs() < 0.4,
            "mean {mean} not near expected {expected}"
        );
    }

    #[test]
    fn binomial_sample_btpe_path() {
        let mut r = rng();
        let n = 1000;
        let p = 0.3;
        let mut sum = 0usize;
        let trials = 2_000;
        for _ in 0..trials {
            let k = binomial_sample(&mut r, n, p);
            assert!(k <= n);
            sum += k;
        }
        let mean = sum as f64 / trials as f64;
        let expected = n as f64 * p;
        assert!(
            (mean - expected).abs() < 6.0,
            "mean {mean} not near expected {expected}"
        );
    }

    #[test]
    fn binomial_sample_btpe_invert_path() {
        let mut r = rng();
        let n = 500;
        let p = 0.85;
        let mut sum = 0usize;
        let trials = 2_000;
        for _ in 0..trials {
            let k = binomial_sample(&mut r, n, p);
            assert!(k <= n);
            sum += k;
        }
        let mean = sum as f64 / trials as f64;
        let expected = n as f64 * p;
        assert!((mean - expected).abs() < 5.0);
    }

    fn binomial_pmf(n: usize, p: f64) -> Vec<f64> {
        let (ln_p, ln_q) = (p.ln(), (1.0 - p).ln());
        let mut ln_choose = 0.0f64;
        (0..=n)
            .map(|k| {
                if k > 0 {
                    ln_choose += ((n - k + 1) as f64).ln() - (k as f64).ln();
                }
                (ln_choose + k as f64 * ln_p + (n - k) as f64 * ln_q).exp()
            })
            .collect()
    }

    // Pearson statistic over every outcome expected five or more times. The
    // bound is the mean plus six standard deviations of a chi-square with that
    // many degrees of freedom, so a correct sampler fails it about once in a
    // billion runs while a misshapen one lands orders of magnitude above it.
    fn assert_binomial_law(n: usize, p: f64) {
        let trials = 200_000;
        let mut r = rng();
        let mut counts = vec![0usize; n + 1];
        for _ in 0..trials {
            counts[binomial_sample(&mut r, n, p)] += 1;
        }
        let pmf = binomial_pmf(n, p);
        let (mut chi, mut dof) = (0.0, 0usize);
        for (k, &prob) in pmf.iter().enumerate() {
            let expected = prob * trials as f64;
            if expected >= 5.0 {
                chi += (counts[k] as f64 - expected).powi(2) / expected;
                dof += 1;
            }
        }
        let bound = dof as f64 + 6.0 * (2.0 * dof as f64).sqrt();
        assert!(
            chi < bound,
            "n={n} p={p}: chi-square {chi:.1} over {dof} bins"
        );
    }

    #[test]
    fn binomial_sample_follows_the_binomial_law() {
        for (n, p) in [
            (30, 0.2),
            (40, 0.4),
            (100, 0.5),
            (1_000, 0.02),
            (1_000, 0.3),
            (1_000, 0.97),
            (5_000, 0.5),
            (100_000, 0.25),
        ] {
            assert_binomial_law(n, p);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn from_seeds_independent_streams() {
        let mut a = Xoshiro256PlusPlus::from_seeds([1, 2, 3, 4]);
        let mut b = Xoshiro256PlusPlus::from_seeds([5, 6, 7, 8]);
        assert_ne!(a.next_u64(), b.next_u64());
    }
}
