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
    let (ln_p, ln_q) = (p.ln(), (-p).ln_1p());
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
