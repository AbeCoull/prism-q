use super::*;

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
fn optimal_chunk_size_aligned_and_min() {
    let c = optimal_chunk_size(64, 1 << 10);
    assert!(c >= 64);
    assert_eq!(c % 64, 0);
}

#[test]
fn default_chunk_size_nonzero() {
    assert!(default_chunk_size(100) >= 64);
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
