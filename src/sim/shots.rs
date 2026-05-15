use std::collections::HashMap;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::compiled;
use super::Probabilities;

/// Result of a multi-shot simulation run.
#[derive(Debug, Clone)]
pub struct ShotsResult {
    /// Classical measurement outcomes for each shot.
    /// `shots[i][j]` is the j-th classical bit from the i-th shot.
    pub shots: Vec<Vec<bool>>,
    pub(crate) num_classical_bits: usize,
}

impl ShotsResult {
    /// Build a frequency histogram of measurement outcomes.
    ///
    /// Keys are packed `Vec<u64>` where bit `i` of word `i/64` corresponds
    /// to classical bit `i`. Use [`bitstring`] to format keys for display.
    pub fn counts(&self) -> HashMap<Vec<u64>, u64> {
        let m_words = self.num_classical_bits.div_ceil(64).max(1);
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        for shot in &self.shots {
            let mut key = vec![0u64; m_words];
            for (i, &b) in shot.iter().enumerate() {
                if b {
                    key[i / 64] |= 1u64 << (i % 64);
                }
            }
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
    }

    pub fn num_shots(&self) -> usize {
        self.shots.len()
    }

    pub fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }
}

impl std::fmt::Display for ShotsResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = self.counts();
        let mut entries: Vec<_> = counts.into_iter().collect();
        entries.sort_by_key(|e| std::cmp::Reverse(e.1));
        for (bits, count) in &entries {
            let bs = bitstring(bits, self.num_classical_bits);
            writeln!(f, "{bs}: {count}")?;
        }
        Ok(())
    }
}

/// Format a packed `Vec<u64>` key (from [`ShotsResult::counts`]) as a binary string.
///
/// Bit 0 of the first word corresponds to classical bit 0 (leftmost character).
pub fn bitstring(key: &[u64], num_bits: usize) -> String {
    let mut s = String::with_capacity(num_bits);
    for i in 0..num_bits {
        let word = i / 64;
        let bit = i % 64;
        if word < key.len() && (key[word] >> bit) & 1 == 1 {
            s.push('1');
        } else {
            s.push('0');
        }
    }
    s
}

fn build_cdf(probs: &[f64]) -> Vec<f64> {
    let mut cdf = Vec::with_capacity(probs.len());
    let mut acc = 0.0;
    for &p in probs {
        acc += p;
        cdf.push(acc);
    }
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    cdf
}

fn sample_from_cdf(cdf: &[f64], r: f64) -> usize {
    match cdf.binary_search_by(|p| p.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i.min(cdf.len() - 1),
    }
}

pub(super) fn sample_shots(
    probs: &Probabilities,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    if meas_map.is_empty() {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    let mut shots = vec![vec![false; num_classical_bits]; num_shots];

    match probs {
        Probabilities::Dense(v) => {
            let cdf = build_cdf(v);
            for shot in &mut shots {
                let r: f64 = rng.random();
                let state_idx = sample_from_cdf(&cdf, r);
                for &(qubit, cbit) in meas_map {
                    shot[cbit] = (state_idx >> qubit) & 1 == 1;
                }
            }
        }
        Probabilities::Factored { blocks, .. } => {
            let block_cdfs: Vec<Vec<f64>> = blocks.iter().map(|b| build_cdf(&b.probs)).collect();
            for shot in &mut shots {
                let mut global_idx = 0usize;
                for (block, cdf) in blocks.iter().zip(block_cdfs.iter()) {
                    let r: f64 = rng.random();
                    let local_idx = sample_from_cdf(cdf, r);
                    let mut m = block.mask;
                    let mut bit = 0;
                    while m != 0 {
                        let pos = m.trailing_zeros() as usize;
                        if local_idx & (1 << bit) != 0 {
                            global_idx |= 1 << pos;
                        }
                        bit += 1;
                        m &= m.wrapping_sub(1);
                    }
                }
                for &(qubit, cbit) in meas_map {
                    shot[cbit] = (global_idx >> qubit) & 1 == 1;
                }
            }
        }
    }

    shots
}

pub(super) fn packed_shots_to_classical_bits(
    packed: &compiled::PackedShots,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
) -> Vec<Vec<bool>> {
    let dense_identity_map = meas_map.len() == num_classical_bits
        && meas_map
            .iter()
            .enumerate()
            .all(|(idx, &(_, classical_bit))| idx == classical_bit);
    if dense_identity_map {
        return packed.to_shots();
    }

    let mut shots = vec![vec![false; num_classical_bits]; packed.num_shots()];
    let mut seen = vec![false; num_classical_bits];
    let unique_classical_bits = meas_map.iter().all(|&(_, classical_bit)| {
        if classical_bit >= num_classical_bits {
            return true;
        }
        if seen[classical_bit] {
            false
        } else {
            seen[classical_bit] = true;
            true
        }
    });

    if unique_classical_bits {
        match packed.layout() {
            compiled::ShotLayout::ShotMajor => {
                let m_words = packed.m_words();
                let data = packed.raw_data();
                for (shot_idx, shot) in shots.iter_mut().enumerate() {
                    let row = &data[shot_idx * m_words..(shot_idx + 1) * m_words];
                    for (measurement, &(_, classical_bit)) in meas_map.iter().enumerate() {
                        if classical_bit >= num_classical_bits {
                            continue;
                        }
                        let word = row[measurement / 64];
                        shot[classical_bit] = (word >> (measurement % 64)) & 1 != 0;
                    }
                }
            }
            compiled::ShotLayout::MeasMajor => {
                let s_words = packed.s_words();
                let data = packed.raw_data();
                let tail = packed.num_shots() % 64;
                let last_mask = if tail == 0 {
                    u64::MAX
                } else {
                    (1u64 << tail) - 1
                };
                for (measurement, &(_, classical_bit)) in meas_map.iter().enumerate() {
                    if classical_bit >= num_classical_bits {
                        continue;
                    }
                    let row = &data[measurement * s_words..(measurement + 1) * s_words];
                    for (sw, mut bits) in row.iter().copied().enumerate() {
                        if sw + 1 == s_words {
                            bits &= last_mask;
                        }
                        while bits != 0 {
                            let shot = sw * 64 + bits.trailing_zeros() as usize;
                            shots[shot][classical_bit] = true;
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }
        return shots;
    }

    for (measurement, &(_, classical_bit)) in meas_map.iter().enumerate() {
        if classical_bit >= num_classical_bits {
            continue;
        }
        for (shot_idx, shot) in shots.iter_mut().enumerate() {
            shot[classical_bit] = packed.get_bit(shot_idx, measurement);
        }
    }
    shots
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::compiled::PackedShots;
    use crate::sim::probability::{FactoredBlock, Probabilities};

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
        let result = ShotsResult {
            shots: vec![vec![true, false], vec![true, false], vec![false, true]],
            num_classical_bits: 2,
        };
        assert_eq!(result.num_shots(), 3);
        assert_eq!(result.num_classical_bits(), 2);
        let counts = result.counts();
        assert_eq!(counts.len(), 2);
        let s = format!("{}", result);
        assert!(s.contains("10: 2"));
        assert!(s.contains("01: 1"));
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
}
