use super::*;
use crate::sim::compiled::PackedShots;
use crate::sim::probability::{FactoredBlock, Probabilities};

fn basis_samples_fixture(num_shots: usize, num_qubits: usize) -> crate::backend::BasisSamples {
    let mut samples = crate::backend::BasisSamples::new(num_shots, num_qubits);
    for shot in 0..num_shots {
        for qubit in 0..num_qubits {
            if (shot * 31 + qubit * 7) % 3 == 0 {
                samples.set(shot, qubit);
            }
        }
    }
    samples
}

fn shots_per_bit(
    samples: &crate::backend::BasisSamples,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
) -> Vec<Vec<bool>> {
    let mut shots = vec![vec![false; num_classical_bits]; samples.num_shots()];
    for (index, shot) in shots.iter_mut().enumerate() {
        for &(qubit, cbit) in meas_map {
            shot[cbit] = samples.bit(index, qubit);
        }
    }
    shots
}

#[test]
fn basis_sample_fast_path_matches_per_bit_expansion() {
    for num_qubits in [4usize, 64, 70, 129] {
        let samples = basis_samples_fixture(17, num_qubits);
        let meas_map: Vec<(usize, usize)> = (0..num_qubits).map(|q| (q, q)).collect();
        assert_eq!(
            shots_from_basis_samples(&samples, &meas_map, num_qubits),
            shots_per_bit(&samples, &meas_map, num_qubits)
        );
    }
}

#[test]
fn basis_sample_fast_path_ignores_qubits_above_the_classical_register() {
    let samples = basis_samples_fixture(9, 70);
    let meas_map: Vec<(usize, usize)> = (0..5).map(|q| (q, q)).collect();
    assert_eq!(
        shots_from_basis_samples(&samples, &meas_map, 5),
        shots_per_bit(&samples, &meas_map, 5)
    );
}

// A classical register wider than the qubit register: the straddling word
// lies past the last word the samples hold, so nothing may be masked off.
#[test]
fn basis_sample_unpack_keeps_every_bit_of_a_narrow_register() {
    let samples = basis_samples_fixture(5, 70);
    let unpacked = samples.to_shots(130);
    for (shot, row) in unpacked.iter().enumerate() {
        for (qubit, &got) in row[..70].iter().enumerate() {
            assert_eq!(got, samples.bit(shot, qubit), "shot {shot} qubit {qubit}");
        }
        assert!(row[70..].iter().all(|b| !b));
    }
}

#[test]
fn basis_sample_permuted_map_takes_the_general_path() {
    let samples = basis_samples_fixture(6, 8);
    let meas_map = [(3, 0), (0, 1), (7, 2)];
    assert_eq!(
        shots_from_basis_samples(&samples, &meas_map, 3),
        shots_per_bit(&samples, &meas_map, 3)
    );
}

#[test]
fn basis_sample_counts_match_the_expanded_histogram() {
    let cases = [
        (8, 8, (0..8).map(|q| (q, q)).collect::<Vec<_>>()),
        (8, 3, vec![(3, 0), (0, 1), (7, 2)]),
        (8, 3, vec![(1, 0), (2, 0), (5, 2), (6, 0)]),
        (8, 20, vec![(1, 7), (4, 19)]),
        (8, 130, vec![(0, 0), (3, 64), (5, 129)]),
        (130, 130, (0..130).map(|q| (q, q)).collect()),
        (
            130,
            70,
            vec![(129, 0), (64, 69), (2, 64), (100, 64), (63, 63)],
        ),
    ];
    for (num_qubits, num_classical_bits, meas_map) in cases {
        let samples = basis_samples_fixture(40, num_qubits);
        let expanded = ShotsResult::from_shots(
            shots_per_bit(&samples, &meas_map, num_classical_bits),
            num_classical_bits,
        )
        .counts();
        assert_eq!(
            counts_from_basis_samples(&samples, &meas_map, num_classical_bits),
            expanded,
            "{num_qubits} qubits, {num_classical_bits} bits, map {meas_map:?}"
        );
    }
    assert!(counts_from_basis_samples(&basis_samples_fixture(0, 8), &[(0, 0)], 1).is_empty());
}

#[test]
fn build_cdf_normalizes_last_to_one() {
    let cdf = build_cdf(&[0.2, 0.3, 0.4999]);
    assert_eq!(cdf.len(), 3);
    assert!((cdf[0] - 0.2).abs() < 1e-12);
    assert!((cdf[1] - 0.5).abs() < 1e-12);
    assert!((cdf[2] - 1.0).abs() < 1e-12);
}

#[test]
fn build_cdf_empty_and_single() {
    let empty = build_cdf(&[]);
    assert!(empty.is_empty());
    let single = build_cdf(&[0.42]);
    assert_eq!(single, vec![1.0]);
}

#[test]
fn sample_from_cdf_bounds() {
    let cdf = [0.25, 0.5, 0.75, 1.0];
    assert_eq!(sample_from_cdf(&cdf, 0.0), 0);
    assert_eq!(sample_from_cdf(&cdf, 0.3), 1);
    assert_eq!(sample_from_cdf(&cdf, 0.99), 3);
    assert_eq!(sample_from_cdf(&cdf, 1.0), 3);
}

#[test]
fn bitstring_packs_bits_lsb_first() {
    let bits = vec![0b1011u64];
    let s = bitstring(&bits, 4);
    assert_eq!(s, "1101");
}

#[test]
fn bitstring_short_key_pads_zero() {
    let s = bitstring(&[], 3);
    assert_eq!(s, "000");
}

#[test]
fn shots_result_counts_and_display() {
    let result = ShotsResult::from_shots(
        vec![vec![true, false], vec![true, false], vec![false, true]],
        2,
    );
    assert_eq!(result.num_shots(), 3);
    assert_eq!(result.num_classical_bits(), 2);
    let counts = result.counts();
    assert_eq!(counts.len(), 2);
    let s = format!("{}", result);
    assert!(s.contains("10: 2"));
    assert!(s.contains("01: 1"));
}

#[test]
fn shots_result_counts_match_per_bit_keys_across_words() {
    let num_bits: usize = 130;
    let mut state = 42u64;
    let mut distinct = Vec::new();
    for _ in 0..7 {
        let shot: Vec<bool> = (0..num_bits)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state & 1 == 1
            })
            .collect();
        distinct.push(shot);
    }
    let shots: Vec<Vec<bool>> = (0..200).map(|i| distinct[(i * i) % 7].clone()).collect();

    let mut expected: HashMap<Vec<u64>, u64> = HashMap::new();
    for shot in &shots {
        let mut key = vec![0u64; num_bits.div_ceil(64)];
        for (i, &b) in shot.iter().enumerate() {
            key[i / 64] |= u64::from(b) << (i % 64);
        }
        *expected.entry(key).or_insert(0) += 1;
    }

    let counts = ShotsResult::from_shots(shots, num_bits).counts();
    assert_eq!(counts, expected);
}

#[test]
fn sample_shots_empty_meas_map_returns_all_false() {
    let probs = Probabilities::Dense(vec![1.0]);
    let shots = sample_shots(&probs, &[], 3, 4, 42);
    assert_eq!(shots.len(), 4);
    for shot in shots {
        assert_eq!(shot, vec![false, false, false]);
    }
}

#[test]
fn sample_shots_dense_deterministic() {
    let probs = Probabilities::Dense(vec![0.0, 1.0]);
    let shots = sample_shots(&probs, &[(0, 0)], 1, 5, 42);
    for shot in shots {
        assert_eq!(shot, vec![true]);
    }
}

#[test]
fn sample_shots_factored_reconstructs_global_index() {
    let probs = Probabilities::Factored {
        blocks: vec![
            FactoredBlock {
                probs: vec![0.0, 1.0],
                mask: 0b001,
            },
            FactoredBlock {
                probs: vec![0.0, 0.0, 0.0, 1.0],
                mask: 0b110,
            },
        ],
        total_qubits: 3,
    };
    let shots = sample_shots(&probs, &[(0, 0), (1, 1), (2, 2)], 3, 10, 42);
    for shot in shots {
        assert_eq!(shot, vec![true, true, true]);
    }
}

#[test]
fn packed_shots_identity_fast_path() {
    let mut data = vec![0u64; 4];
    data[0] = 0b101;
    data[1] = 0b010;
    data[2] = 0b111;
    data[3] = 0b000;
    let packed = PackedShots::from_shot_major(data, 4, 3);
    let meas_map = [(0, 0), (0, 1), (0, 2)];
    let shots = packed_shots_to_classical_bits(&packed, &meas_map, 3);
    assert_eq!(shots.len(), 4);
    assert_eq!(shots[0], vec![true, false, true]);
    assert_eq!(shots[1], vec![false, true, false]);
    assert_eq!(shots[2], vec![true, true, true]);
    assert_eq!(shots[3], vec![false, false, false]);
}

#[test]
fn packed_shots_non_identity_mapping() {
    let mut data = vec![0u64; 2];
    data[0] = 0b011;
    data[1] = 0b100;
    let packed = PackedShots::from_shot_major(data, 2, 3);
    let meas_map = [(0, 2), (0, 1), (0, 0)];
    let shots = packed_shots_to_classical_bits(&packed, &meas_map, 3);
    assert_eq!(shots[0], vec![false, true, true]);
    assert_eq!(shots[1], vec![true, false, false]);
}

#[test]
fn packed_shots_out_of_range_classical_bit_skipped() {
    let data = vec![0b111u64];
    let packed = PackedShots::from_shot_major(data, 1, 3);
    let meas_map = [(0, 0), (0, 5), (0, 1)];
    let shots = packed_shots_to_classical_bits(&packed, &meas_map, 2);
    assert_eq!(shots[0], vec![true, true]);
}
