use rand::RngCore;
use rand_chacha::ChaCha8Rng;

pub(super) struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    #[cfg(feature = "parallel")]
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

    #[inline(always)]
    pub(super) fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[inline(never)]
pub(super) fn binomial_sample(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64) -> usize {
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

    if invert {
        n - result
    } else {
        result
    }
}

fn binomial_inversion(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64, _nf: f64) -> usize {
    let q = 1.0 - p;
    let s = p / q;
    let a = ((n + 1) as f64) * s;
    let mut r = q.powi(n as i32);
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

fn binomial_btpe(rng: &mut Xoshiro256PlusPlus, n: usize, p: f64, nf: f64, np: f64) -> usize {
    let q = 1.0 - p;
    let r = p / q;
    let nr = (nf + 1.0) * r;

    let fm = np + p;
    let m = fm as usize;
    let mf = m as f64;

    let p1 = (2.195 * (np * q).sqrt() - 4.6 * q).floor() + 0.5;
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
        let v = rng.next_f64();

        let y: isize;

        if u <= p1 {
            y = (xm - p1 * v + u) as isize;
        } else if u <= p2 {
            let x1 = xl + (u - p1) / c;
            let fv = 1.0 - (mf - x1 + 0.5).abs() / p1;
            if v > fv {
                continue;
            }
            y = x1 as isize;
        } else if u <= p3 {
            let lv = if v > 0.0 { v.ln() } else { -700.0 };
            y = (xl + lv / lambda_l) as isize;
            if y < 0 {
                continue;
            }
            // v already set for step F
        } else {
            let lv = if v > 0.0 { v.ln() } else { -700.0 };
            y = (xr - lv / lambda_r) as isize;
            if y as usize > n {
                continue;
            }
        }

        if y < 0 || y as usize > n {
            continue;
        }
        let iy = y as usize;

        let k = iy.abs_diff(m);
        let kf = k as f64;

        if kf <= 20.0 || kf * kf + kf >= np * q * 3.0 {
            let mut a_val = 1.0;
            if m < iy {
                for i in (m + 1)..=iy {
                    a_val *= nr / i as f64 - r;
                }
            } else if m > iy {
                for i in (iy + 1)..=m {
                    a_val *= nr / i as f64 - r;
                }
                a_val = 1.0 / a_val;
            }

            let v_adj = if u <= p2 {
                v
            } else if u <= p3 {
                (u - p2) * lambda_l
            } else {
                (u - p3) * lambda_r
            };

            if v_adj <= a_val {
                return iy;
            }
            continue;
        }

        let rho = (kf / (np * q)) * ((kf * (kf / 3.0 + 0.625) + 1.0 / 6.0) / (np * q) + 0.5);
        let t = -kf * kf / (2.0 * np * q);

        let v_adj = if u <= p1 || u <= p2 {
            v
        } else if u <= p3 {
            (u - p2) * lambda_l
        } else {
            (u - p3) * lambda_r
        };
        let log_v = if v_adj > 0.0 { v_adj.ln() } else { -700.0 };

        if log_v < t - rho {
            return iy;
        }
        if log_v > t + rho {
            continue;
        }

        let x1 = (iy + 1) as f64;
        let f1 = (m + 1) as f64;
        let z = (n + 1 - m) as f64;
        let w = (n - iy + 1) as f64;
        let x2 = x1 * x1;
        let f2 = f1 * f1;
        let z2 = z * z;
        let w2 = w * w;

        let stirling = |x: f64, x_sq: f64| {
            (13860.0 - (462.0 - (132.0 - (99.0 - 140.0 / x_sq) / x_sq) / x_sq) / x_sq)
                / x
                / 166320.0
        };

        let bound = xm * (f1 / x1).ln()
            + (nf - mf + 0.5) * (z / w).ln()
            + ((iy as f64) - mf) * (w * r / (x1 * q)).ln()
            + stirling(f1, f2)
            + stirling(z, z2)
            + stirling(x1, x2)
            + stirling(w, w2);

        if log_v <= bound {
            return iy;
        }
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
