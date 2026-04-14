use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use prism_q::backend::mps::svd_jacobi;

#[cfg(feature = "parallel")]
use prism_q::backend::mps::svd_faer;

fn random_matrix(m: usize, n: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..m * n)
        .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
        .collect()
}

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");

    #[cfg(feature = "bench-fast")]
    {
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(200));
        group.measurement_time(std::time::Duration::from_millis(500));
    }

    for &size in &[4, 16, 64, 128] {
        let mat = random_matrix(size, size, 0xDEAD_BEEF);

        group.bench_with_input(BenchmarkId::new("jacobi", size), &size, |b, &sz| {
            b.iter(|| svd_jacobi(&mat, sz, sz))
        });

        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("faer", size), &size, |b, &sz| {
            b.iter(|| svd_faer(&mat, sz, sz))
        });
    }

    group.finish();
}

fn bench_svd_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_rect");

    #[cfg(feature = "bench-fast")]
    {
        group.sample_size(10);
        group.warm_up_time(std::time::Duration::from_millis(200));
        group.measurement_time(std::time::Duration::from_millis(500));
    }

    for &bond in &[4, 16, 64, 128] {
        let m = 2 * bond;
        let n = bond;
        let label = format!("{}x{}", m, n);
        let mat = random_matrix(m, n, 0xDEAD_BEEF);

        group.bench_with_input(BenchmarkId::new("jacobi", &label), &(), |b, _| {
            b.iter(|| svd_jacobi(&mat, m, n))
        });

        #[cfg(feature = "parallel")]
        group.bench_with_input(BenchmarkId::new("faer", &label), &(), |b, _| {
            b.iter(|| svd_faer(&mat, m, n))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_svd, bench_svd_rectangular);
criterion_main!(benches);
