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

    let mut indices = Vec::with_capacity(num_shots);

    match probs {
        Probabilities::Dense(v) => {
            let cdf = build_cdf(v);
            for _ in 0..num_shots {
                let r: f64 = rng.random();
                indices.push(sample_from_cdf(&cdf, r));
            }
        }
        Probabilities::Factored { blocks, .. } => {
            let block_cdfs: Vec<Vec<f64>> = blocks.iter().map(|b| build_cdf(&b.probs)).collect();
            for _ in 0..num_shots {
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
                indices.push(global_idx);
            }
        }
    }

    let mut flat = vec![false; num_shots * num_classical_bits];
    for (s, &state_idx) in indices.iter().enumerate() {
        let base = s * num_classical_bits;
        for &(qubit, cbit) in meas_map {
            flat[base + cbit] = (state_idx >> qubit) & 1 == 1;
        }
    }

    let mut shots = Vec::with_capacity(num_shots);
    for chunk in flat.chunks_exact(num_classical_bits) {
        shots.push(chunk.to_vec());
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
