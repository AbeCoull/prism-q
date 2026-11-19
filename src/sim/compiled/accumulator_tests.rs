use super::*;
use crate::circuits;
use crate::sim::compiled::compile_measurements;

fn shot_major_packed(num_shots: usize, num_meas: usize, bits: &[(usize, usize)]) -> PackedShots {
    let m_words = num_meas.div_ceil(64);
    let mut data = vec![0u64; num_shots * m_words];
    for &(s, m) in bits {
        data[s * m_words + m / 64] |= 1u64 << (m % 64);
    }
    PackedShots::from_shot_major(data, num_shots, num_meas)
}

fn meas_major_packed(num_shots: usize, num_meas: usize, bits: &[(usize, usize)]) -> PackedShots {
    let s_words = num_shots.div_ceil(64);
    let mut data = vec![0u64; num_meas * s_words];
    for &(s, m) in bits {
        data[m * s_words + s / 64] |= 1u64 << (s % 64);
    }
    PackedShots::from_meas_major(data, num_shots, num_meas)
}

#[test]
fn optimal_chunk_size_zero_meas() {
    assert_eq!(optimal_chunk_size(0, 1 << 20), 1 << 20);
}

#[test]
fn optimal_chunk_size_rounds_down_to_64_with_a_floor_of_64() {
    assert_eq!(optimal_chunk_size(64, 8 * 200), 192);
    assert_eq!(optimal_chunk_size(64, 8 * 10), 64);
    assert_eq!(optimal_chunk_size(1000, 256 << 20), 2_097_152);
    assert_eq!(default_chunk_size(100), 16_777_216);
}

#[test]
fn histogram_w1_shot_major() {
    let packed = shot_major_packed(5, 3, &[(0, 0), (0, 2), (1, 0), (1, 2), (2, 1)]);
    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let counts = acc.into_counts();
    assert_eq!(counts.values().sum::<u64>(), 5);
    assert_eq!(counts.get(&vec![0b101]), Some(&2));
    assert_eq!(counts.get(&vec![0b010]), Some(&1));
    assert_eq!(counts.get(&vec![0]), Some(&2));
}

#[test]
fn histogram_w1_meas_major_matches_shot_major() {
    let bits = [(0, 0), (1, 2), (2, 0), (2, 2)];
    let a = shot_major_packed(3, 3, &bits);
    let b = meas_major_packed(3, 3, &bits);
    let mut ha = HistogramAccumulator::new();
    ha.accumulate(&a);
    let mut hb = HistogramAccumulator::new();
    hb.accumulate(&b);
    assert_eq!(ha.into_counts(), hb.into_counts());
}

#[test]
fn histogram_w5_path_shot_major() {
    let num_meas = 5 * 64;
    let bits = [(0, 0), (0, 64), (0, 128), (0, 192), (0, 256), (1, 300)];
    let packed = shot_major_packed(2, num_meas, &bits);
    let mut acc = HistogramAccumulator::default();
    acc.accumulate(&packed);
    let counts = acc.into_counts();
    assert_eq!(counts.values().sum::<u64>(), 2);
    assert_eq!(counts.len(), 2);
}

#[test]
fn histogram_wide_path_shot_major() {
    let num_meas = 9 * 64;
    let bits = [(0, 0), (0, 8 * 64), (1, 0)];
    let packed = shot_major_packed(2, num_meas, &bits);
    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let counts = acc.into_counts();
    assert_eq!(counts.values().sum::<u64>(), 2);
    assert_eq!(counts.len(), 2);
}

#[test]
fn histogram_wide_meas_major() {
    let num_meas = 9 * 64;
    let bits = [(0, 0), (1, 8 * 64), (2, 0)];
    let packed = meas_major_packed(3, num_meas, &bits);
    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let counts = acc.into_counts();
    assert_eq!(counts.values().sum::<u64>(), 3);
}

#[test]
fn histogram_empty_into_counts() {
    let acc = HistogramAccumulator::new();
    assert!(acc.into_counts().is_empty());
}

#[test]
fn count_vec_key_insert_and_update() {
    let mut m: FxHashMap<Vec<u64>, u64> = FxHashMap::default();
    count_vec_key(&mut m, &[1, 2, 3]);
    count_vec_key(&mut m, &[1, 2, 3]);
    count_vec_key(&mut m, &[4, 5, 6]);
    assert_eq!(m.get(&vec![1u64, 2, 3]), Some(&2));
    assert_eq!(m.get(&vec![4u64, 5, 6]), Some(&1));
}

#[test]
fn null_accumulator_sums_shots() {
    let mut acc = NullAccumulator::default();
    let a = shot_major_packed(7, 2, &[]);
    let b = shot_major_packed(3, 2, &[]);
    acc.accumulate(&a);
    acc.accumulate(&b);
    assert_eq!(acc.total_shots(), 10);
}

#[test]
fn correlator_zero_shots_returns_zero() {
    let acc = CorrelatorAccumulator::new(vec![(0, 1)]);
    assert_eq!(acc.correlators(), vec![0.0]);
    assert_eq!(acc.total_shots(), 0);
}

#[test]
fn correlator_shot_major_equals_meas_major() {
    let bits = [(0, 0), (1, 1), (2, 0), (2, 1), (3, 0)];
    let sm = shot_major_packed(4, 2, &bits);
    let mm = meas_major_packed(4, 2, &bits);
    let mut a = CorrelatorAccumulator::new(vec![(0, 1)]);
    let mut b = CorrelatorAccumulator::new(vec![(0, 1)]);
    a.accumulate(&sm);
    b.accumulate(&mm);
    assert_eq!(a.correlators(), b.correlators());
    assert_eq!(a.total_shots(), 4);
}

// Random matrices, so every bit position of every stage is exercised.
#[cfg(target_arch = "x86_64")]
#[test]
fn transpose_64x64_avx512_matches_scalar() {
    use rand::{RngExt, SeedableRng};
    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for _ in 0..32 {
        let original: [u64; 64] = rng.random();
        let mut want = original;
        transpose_64x64(&mut want);
        let mut got = original;
        // SAFETY: AVX-512F detected.
        unsafe { transpose_64x64_avx512(&mut got) };
        assert_eq!(got, want);
    }
}

#[test]
fn transpose_64x64_scalar_involutive() {
    let mut matrix = [0u64; 64];
    for (i, row) in matrix.iter_mut().enumerate() {
        *row = 0x0123_4567_89AB_CDEFu64
            .wrapping_mul(i as u64 + 1)
            .rotate_left((i % 64) as u32);
    }
    let original = matrix;
    transpose_64x64(&mut matrix);
    transpose_64x64(&mut matrix);
    assert_eq!(matrix, original);
}

#[test]
fn direct_histogram_matches_typed_counts() {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
    for words in [1, 2, 3, 4, 5, 6, 9] {
        let m = 64 * words - 3;
        for shots in [1, 63, 64, 130, 70_000] {
            let pool: Vec<Vec<u64>> = (0..8)
                .map(|_| (0..words).map(|_| rng.random::<u64>()).collect())
                .collect();
            let mut data = vec![0u64; shots * words];
            for shot in data.chunks_exact_mut(words) {
                if rng.random::<bool>() {
                    shot.copy_from_slice(&pool[rng.random_range(0..pool.len())]);
                } else {
                    shot.iter_mut().for_each(|w| *w = rng.random::<u64>());
                }
                shot[words - 1] &= (1u64 << (m % 64)) - 1;
            }
            let shot_major = PackedShots::from_shot_major(data, shots, m);
            let meas_major = shot_major.to_meas_major();
            let counts = |mut acc: HistogramAccumulator, chunk: &PackedShots| {
                acc.accumulate(chunk);
                acc.accumulate(chunk);
                acc.into_counts()
            };
            let pooled = |chunk: &PackedShots| {
                let expected = counts(HistogramAccumulator::new(), chunk);
                #[cfg(feature = "parallel")]
                for num_threads in [1, 4] {
                    let actual = rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .build()
                        .unwrap()
                        .install(|| counts(HistogramAccumulator::direct(), chunk));
                    assert_eq!(actual, expected, "{words} words, {shots} shots");
                }
                assert_eq!(counts(HistogramAccumulator::direct(), chunk), expected);
                expected
            };
            assert_eq!(pooled(&shot_major), pooled(&meas_major));
        }
    }
}

#[test]
fn chunked_histogram_matches_direct() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }
    let num_shots = 50_000;

    let mut sampler_direct = compile_measurements(&c, 42).unwrap();
    let direct_counts = sampler_direct.sample_bulk_packed(num_shots).counts();

    let mut sampler_chunked = compile_measurements(&c, 42).unwrap();
    let mut acc = HistogramAccumulator::new();
    sampler_chunked.sample_chunked(num_shots, &mut acc);
    let chunked_counts = acc.into_counts();

    assert_eq!(
        direct_counts, chunked_counts,
        "chunked histogram must match direct"
    );
}

#[test]
fn accumulator_matches_packed_counts() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(100_000);
    let direct_counts = packed.counts();

    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let acc_counts = acc.into_counts();

    assert_eq!(
        direct_counts, acc_counts,
        "accumulator counts must match PackedShots::counts"
    );
}

#[test]
fn chunked_histogram_small_chunks() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let num_shots = 10_000;

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = HistogramAccumulator::new();
    sampler.sample_chunked_with_size(num_shots, 300, &mut acc);
    let counts = acc.into_counts();

    let total: u64 = counts.values().sum();
    assert_eq!(total, num_shots as u64, "total shots must match");
    assert_eq!(counts.len(), 2, "GHZ-10 should have exactly 2 outcomes");

    let all_zeros = vec![0u64; 10usize.div_ceil(64)];
    let all_ones = {
        let mut v = vec![0u64; 10usize.div_ceil(64)];
        v[0] = (1u64 << 10) - 1;
        v
    };
    let c0 = counts.get(&all_zeros).copied().unwrap_or(0);
    let c1 = counts.get(&all_ones).copied().unwrap_or(0);
    assert!(
        (c0 as f64 / num_shots as f64 - 0.5).abs() < 0.03,
        "GHZ |0⟩^n fraction {c0}/{num_shots} should be ~0.5"
    );
    assert!(
        (c1 as f64 / num_shots as f64 - 0.5).abs() < 0.03,
        "GHZ |1⟩^n fraction {c1}/{num_shots} should be ~0.5"
    );
}

#[test]
fn correlator_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }

    let pairs = vec![(0, 1), (0, 9), (3, 7)];
    let mut sampler = compile_measurements(&c, 42).unwrap();
    let mut acc = CorrelatorAccumulator::new(pairs);
    sampler.sample_chunked(100_000, &mut acc);
    let corrs = acc.correlators();

    for (i, &c) in corrs.iter().enumerate() {
        assert!(
            (c - 1.0).abs() < 0.02,
            "GHZ pair {i} correlator {c} should be ~1.0"
        );
    }
}
