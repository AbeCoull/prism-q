use super::*;
use crate::circuit::Circuit;
use crate::circuits;
use crate::gates::Gate;
use crate::sim::compiled::{
    HistogramAccumulator, MarginalsAccumulator, compile_forward, compile_measurements,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[test]
fn packed_shots_roundtrip_ghz() {
    let mut c = circuits::ghz_circuit(10);
    c.num_classical_bits = 10;
    for i in 0..10 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(1000);
    assert_eq!(packed.num_shots(), 1000);
    assert_eq!(packed.num_measurements(), 10);

    let unpacked = packed.to_shots();
    assert_eq!(unpacked.len(), 1000);
    for shot in &unpacked {
        assert_eq!(shot.len(), 10);
        let first = shot[0];
        assert!(shot.iter().all(|&b| b == first));
    }
}

#[test]
fn packed_shots_matches_sample_bulk() {
    let mut c = circuits::clifford_heavy_circuit(20, 5, 42);
    c.num_classical_bits = 20;
    for i in 0..20 {
        c.add_measure(i, i);
    }

    let mut sampler1 = compile_forward(&c, 42).unwrap();
    let mut sampler2 = compile_forward(&c, 42).unwrap();

    let num_shots = 5000;
    let bulk = sampler1.sample_bulk(num_shots);
    let packed = sampler2.sample_bulk_packed(num_shots);
    let unpacked = packed.to_shots();

    assert_eq!(bulk.len(), unpacked.len());
    assert_eq!(bulk[0].len(), unpacked[0].len());

    let n = bulk[0].len();
    for q in 0..n {
        let freq1: usize = bulk.iter().filter(|s| s[q]).count();
        let freq2: usize = unpacked.iter().filter(|s| s[q]).count();
        let p1 = freq1 as f64 / num_shots as f64;
        let p2 = freq2 as f64 / num_shots as f64;
        assert!(
            (p1 - p2).abs() < 0.05,
            "qubit {q}: bulk={p1:.3}, packed={p2:.3}"
        );
    }
}

#[test]
fn packed_shots_get_bit() {
    let mut c = circuits::ghz_circuit(4);
    c.num_classical_bits = 4;
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(100);

    for s in 0..100 {
        let first = packed.get_bit(s, 0);
        for m in 1..4 {
            assert_eq!(packed.get_bit(s, m), first);
        }
    }
}

#[test]
fn packed_shots_parity_rows_shot_major() {
    let packed = PackedShots::from_shot_major(vec![0b001, 0b011, 0b110], 3, 3);
    let rows = vec![vec![0, 1], vec![1, 2], Vec::new()];
    let parity = packed.parity_rows(&rows).unwrap();
    let shots = parity.to_shots();
    assert_eq!(
        shots,
        vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![true, false, false],
        ]
    );
}

#[test]
fn packed_shots_parity_rows_meas_major() {
    let packed = PackedShots::from_meas_major(vec![0b011, 0b110, 0b100], 3, 3);
    let rows = vec![vec![0, 1], vec![1, 2], Vec::new()];
    let parity = packed.parity_rows(&rows).unwrap();
    assert_eq!(parity.layout(), ShotLayout::MeasMajor);
    assert_eq!(
        parity.to_shots(),
        vec![
            vec![true, false, false],
            vec![false, true, false],
            vec![true, false, false],
        ]
    );
}

#[test]
fn packed_shots_parity_rows_rejects_bad_index() {
    let packed = PackedShots::from_shot_major(vec![0], 1, 1);
    assert!(packed.parity_rows(&[vec![1]]).is_err());
}

#[test]
fn packed_shots_try_constructors_reject_invalid_raw_data() {
    for (name, result) in [
        (
            "shot-major shape",
            PackedShots::try_from_shot_major(vec![0], 2, 65),
        ),
        (
            "measurement-major shape",
            PackedShots::try_from_meas_major(vec![0], 65, 2),
        ),
        (
            "shot-major tail",
            PackedShots::try_from_shot_major(vec![0b1000], 1, 3),
        ),
        (
            "measurement-major tail",
            PackedShots::try_from_meas_major(vec![0b1000], 3, 1),
        ),
    ] {
        assert!(
            matches!(result.unwrap_err(), PrismError::InvalidParameter { .. }),
            "{name}"
        );
    }
}

#[test]
fn packed_shots_reports_raw_format_version() {
    let packed = PackedShots::from_shot_major(vec![0], 1, 1);
    assert_eq!(packed.raw_format_version(), PackedShots::RAW_FORMAT_VERSION);
    assert_eq!(PackedShots::RAW_FORMAT_VERSION, 1);
}

#[test]
fn packed_shots_counts_ignore_shot_major_padding() {
    let packed = PackedShots {
        data: vec![0b1000, 0b1001, 0b0000],
        num_shots: 3,
        num_measurements: 3,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::ShotMajor,
    };

    let counts = packed.counts();
    assert_eq!(counts.get(&vec![0]), Some(&2));
    assert_eq!(counts.get(&vec![1]), Some(&1));

    let mut acc = HistogramAccumulator::new();
    acc.accumulate(&packed);
    let acc_counts = acc.into_counts();
    assert_eq!(acc_counts.get(&vec![0]), Some(&2));
    assert_eq!(acc_counts.get(&vec![1]), Some(&1));
}

#[test]
fn packed_shots_marginals_ignore_meas_major_padding() {
    let packed = PackedShots {
        data: vec![0b1001],
        num_shots: 3,
        num_measurements: 1,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::MeasMajor,
    };

    let mut acc = MarginalsAccumulator::new(1);
    acc.accumulate(&packed);
    assert!((acc.marginals()[0] - (1.0 / 3.0)).abs() < 1e-12);
}

#[test]
fn packed_shots_parity_rows_clear_meas_major_padding() {
    let packed = PackedShots {
        data: vec![0b1001, 0b1000],
        num_shots: 3,
        num_measurements: 2,
        m_words: 1,
        s_words: 1,
        layout: ShotLayout::MeasMajor,
    };

    let parity = packed.parity_rows(&[vec![0], vec![0, 1]]).unwrap();
    assert_eq!(parity.raw_data(), &[0b001, 0b001]);
}

#[test]
fn packed_shots_counts() {
    let mut c = circuits::ghz_circuit(4);
    c.num_classical_bits = 4;
    for i in 0..4 {
        c.add_measure(i, i);
    }
    let mut sampler = compile_forward(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(10_000);
    let counts = packed.counts();

    assert_eq!(counts.len(), 2);
    let total: u64 = counts.values().sum();
    assert_eq!(total, 10_000);
}

#[test]
fn packed_shots_counts_support_wide_outputs() {
    let n = 600;
    let mut c = Circuit::new(n, n);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for q in 0..n {
        c.add_measure(q, q);
    }

    let mut sampler = compile_measurements(&c, 42).unwrap();
    let packed = sampler.sample_bulk_packed(256);
    let counts = packed.counts();
    let total: u64 = counts.values().sum();
    assert_eq!(total, 256);
    assert!(counts.keys().all(|key| key.len() == n.div_ceil(64)));
}

/// A random `shots x measurements` matrix in both layouts, padding clear.
/// Each bit is set with probability `2^-(1 + thinning)`.
fn random_packed_pair(
    shots: usize,
    measurements: usize,
    thinning: u32,
    seed: u64,
) -> (PackedShots, PackedShots) {
    use rand::RngExt;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let m_words = measurements.div_ceil(64);
    let tail = measurement_tail_mask(measurements);
    let mut data: Vec<u64> = (0..shots * m_words)
        .map(|_| (0..thinning).fold(rng.random::<u64>(), |word, _| word & rng.random::<u64>()))
        .collect();
    for shot in 0..shots {
        data[shot * m_words + m_words - 1] &= tail;
    }
    let shot_major = PackedShots::from_shot_major(data, shots, measurements);
    let meas_major = PackedShots::from_meas_major(
        shot_major.clone().into_meas_major_data(),
        shots,
        measurements,
    );
    (shot_major, meas_major)
}

/// Shot rows of packed bytes built one bit at a time, the layout `to_shot_major_bytes`
/// must reproduce.
fn bytes_bit_by_bit(packed: &PackedShots) -> Vec<u8> {
    let row_bytes = packed.num_measurements().div_ceil(8);
    let mut bytes = vec![0u8; packed.num_shots() * row_bytes];
    for shot in 0..packed.num_shots() {
        for measurement in 0..packed.num_measurements() {
            if packed.get_bit(shot, measurement) {
                bytes[shot * row_bytes + measurement / 8] |= 1 << (measurement % 8);
            }
        }
    }
    bytes
}

/// Histogram keyed by shot rows built one bit at a time, the map `counts` must return.
fn counts_bit_by_bit(packed: &PackedShots) -> std::collections::HashMap<Vec<u64>, u64> {
    let mut counts = std::collections::HashMap::new();
    for shot in 0..packed.num_shots() {
        let mut key = vec![0u64; packed.m_words()];
        for measurement in 0..packed.num_measurements() {
            if packed.get_bit(shot, measurement) {
                key[measurement / 64] |= 1 << (measurement % 64);
            }
        }
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

#[test]
fn layout_conversions_match_a_bit_by_bit_transpose() {
    let sizes = [1, 2, 7, 8, 9, 63, 64, 65, 127, 128, 129, 200, 300];
    let mut seed = 0;
    for &shots in &sizes {
        for &measurements in &sizes {
            for thinning in [0, 7] {
                seed += 1;
                let (shot_major, meas_major) =
                    random_packed_pair(shots, measurements, thinning, seed);
                let expected = bytes_bit_by_bit(&shot_major);
                assert_eq!(
                    meas_major.to_shot_major_bytes(),
                    expected,
                    "{shots}x{measurements} thinning {thinning}"
                );
                assert_eq!(
                    shot_major.to_shot_major_bytes(),
                    expected,
                    "{shots}x{measurements} thinning {thinning}"
                );
                assert_eq!(
                    meas_major.to_shot_major().raw_data(),
                    shot_major.raw_data(),
                    "{shots}x{measurements} thinning {thinning}"
                );
                assert_eq!(
                    meas_major.clone().into_shot_major_data(),
                    shot_major.raw_data(),
                    "{shots}x{measurements} thinning {thinning}"
                );
                assert_eq!(
                    shot_major.to_meas_major().raw_data(),
                    meas_major.raw_data(),
                    "{shots}x{measurements} thinning {thinning}"
                );
            }
        }
    }
}

#[test]
fn layout_conversions_handle_empty_matrices() {
    for (shots, measurements) in [(0usize, 5usize), (5, 0), (0, 0)] {
        let packed = PackedShots::from_meas_major(
            vec![0; measurements * shots.div_ceil(64)],
            shots,
            measurements,
        );
        assert_eq!(
            packed.to_shot_major_bytes(),
            vec![0u8; shots * measurements.div_ceil(8)]
        );
        assert_eq!(packed.to_shot_major().num_shots(), shots);
        assert_eq!(packed.counts(), counts_bit_by_bit(&packed));
    }
}

#[test]
fn xor_columns_flips_the_masked_columns_of_every_shot() {
    for (shots, measurements) in [(1, 1), (65, 3), (130, 70)] {
        let (shot_major, meas_major) = random_packed_pair(shots, measurements, 0, 99);
        let mut mask = vec![0u64; measurements.div_ceil(64)];
        for measurement in (0..measurements).step_by(3) {
            mask[measurement / 64] |= 1 << (measurement % 64);
        }
        for mut packed in [shot_major.clone(), meas_major] {
            packed.xor_columns(&mask);
            for shot in 0..shots {
                for measurement in 0..measurements {
                    assert_eq!(
                        packed.get_bit(shot, measurement),
                        shot_major.get_bit(shot, measurement) ^ (measurement % 3 == 0),
                    );
                }
            }
            assert_eq!(packed.to_shot_major_bytes(), {
                let mut expected = shot_major.clone();
                expected.xor_columns(&mask);
                bytes_bit_by_bit(&expected)
            });
        }
    }
}

#[test]
fn counts_match_a_bit_by_bit_histogram() {
    let mut seed = 1000;
    for &shots in &[1, 63, 64, 65, 200] {
        for &measurements in &[1, 63, 64, 65, 130, 600] {
            for thinning in [0, 7] {
                seed += 1;
                let (shot_major, meas_major) =
                    random_packed_pair(shots, measurements, thinning, seed);
                let expected = counts_bit_by_bit(&shot_major);
                assert_eq!(
                    shot_major.counts(),
                    expected,
                    "{shots}x{measurements} thinning {thinning}"
                );
                assert_eq!(
                    meas_major.counts(),
                    expected,
                    "{shots}x{measurements} thinning {thinning}"
                );
            }
        }
    }
}

#[test]
fn counts_match_a_bit_by_bit_histogram_past_the_parallel_floor() {
    for (measurements, seed) in [(65, 7), (520, 8)] {
        let (shot_major, meas_major) = random_packed_pair(70_000, measurements, 7, seed);
        let expected = counts_bit_by_bit(&shot_major);
        assert!(expected.len() < 70_000, "thinning must repeat outcomes");
        assert_eq!(shot_major.counts(), expected, "{measurements} shot-major");
        assert_eq!(meas_major.counts(), expected, "{measurements} meas-major");
    }
}
