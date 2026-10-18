//! Shot collections and probability-weighted shot sampling.

use std::collections::HashMap;

use crate::hash::FxHashMap;

use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::Probabilities;
use super::compiled;
use super::metadata::{ResolvedBackend, RunMetadata};

/// Marker on a result no route has stamped yet; `every_terminal_stamps_metadata`
/// asserts none reaches a caller.
pub(crate) const UNSTAMPED: &str = "unstamped";

/// Result of a multi-shot simulation run.
#[derive(Debug, Clone)]
pub struct ShotsResult {
    /// `shots[i][j]` is the j-th classical bit from the i-th shot.
    pub shots: Vec<Vec<bool>>,
    pub(crate) num_classical_bits: usize,
    /// Which engine ran, whether the answer is exact, and where the state lived.
    pub metadata: RunMetadata,
    /// `leaked[i][q]` is `true` when qubit `q` leaked at any point of shot `i`,
    /// the erasure herald of a model carrying
    /// [`NoiseChannel::Leakage`](crate::NoiseChannel::Leakage); `None` for every
    /// other run.
    pub leaked: Option<Vec<Vec<bool>>>,
}

impl ShotsResult {
    /// Build with placeholder provenance; the entry point that picked the
    /// engine overwrites it through [`ShotsResult::with_metadata`], which
    /// `every_terminal_stamps_metadata` pins.
    pub(crate) fn from_shots(shots: Vec<Vec<bool>>, num_classical_bits: usize) -> Self {
        Self {
            shots,
            num_classical_bits,
            metadata: RunMetadata::exact(ResolvedBackend::Other(UNSTAMPED)),
            leaked: None,
        }
    }

    pub(crate) fn with_leaked(mut self, leaked: Vec<Vec<bool>>) -> Self {
        self.leaked = Some(leaked);
        self
    }

    pub(crate) fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        let shots = self.shots.len();
        self.metadata = metadata.with_shots(shots);
        self
    }

    /// Keys are packed `Vec<u64>` where bit `i` of word `i/64` corresponds
    /// to classical bit `i`. Use [`bitstring`] to format keys for display.
    pub fn counts(&self) -> HashMap<Vec<u64>, u64> {
        let m_words = self.num_classical_bits.div_ceil(64).max(1);
        let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
        let mut key = vec![0u64; m_words];
        for shot in &self.shots {
            key.fill(0);
            for (word, bits) in key.iter_mut().zip(shot.chunks(64)) {
                *word = bits
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &b)| acc | (u64::from(b) << i));
            }
            match counts.get_mut(key.as_slice()) {
                Some(count) => *count += 1,
                None => {
                    counts.insert(key.clone(), 1);
                }
            }
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

#[cfg(test)]
impl ShotsResult {
    pub(crate) fn marginal(&self, bit: usize) -> f64 {
        self.shots.iter().filter(|s| s[bit]).count() as f64 / self.shots.len() as f64
    }

    pub(crate) fn coherent_fraction(&self) -> f64 {
        self.shots
            .iter()
            .filter(|s| s.iter().all(|&b| b == s[0]))
            .count() as f64
            / self.shots.len() as f64
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

pub(crate) fn build_cdf(probs: &[f64]) -> Vec<f64> {
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

/// Shots drawn per ChaCha8 substream by the block samplers. One stream per
/// block rather than per shot: cipher setup and the first keystream block cost
/// more than a whole draw when the per-shot work is short.
pub(crate) const SHOTS_PER_STREAM: usize = 256;

/// Shot count below which a block sampler stays on the calling thread.
pub(crate) const MIN_SHOTS_FOR_PAR: usize = 32;

/// Draws a call must make before the product and factored samplers split their
/// blocks across workers: 2000 factored draws (1000 shots over two sub-states)
/// ran 18% slower forked than serial.
const MIN_DRAWS_FOR_PAR: usize = 1 << 14;

/// Whether `num_shots` shots of `draws_per_shot` draws each carry enough work
/// to fill their blocks in parallel. The blocks draw the same values either way.
pub(crate) fn draws_split_across_workers(num_shots: usize, draws_per_shot: usize) -> bool {
    num_shots >= MIN_SHOTS_FOR_PAR && num_shots.saturating_mul(draws_per_shot) >= MIN_DRAWS_FOR_PAR
}

/// Fill `rows`, `per_shot` entries to a shot, one block of [`SHOTS_PER_STREAM`]
/// shots at a time. Block `b` draws from stream `b + 2` of the ChaCha8 generator
/// seeded with `seed`, so the partition, and with it every draw, is a function
/// of the shot count alone: `parallel` moves the schedule, never a value.
pub(crate) fn sample_in_shot_blocks<T: Send>(
    rows: &mut [T],
    per_shot: usize,
    seed: u64,
    parallel: bool,
    sample_block: impl Fn(&mut ChaCha8Rng, &mut [T]) + Sync,
) {
    let block_len = SHOTS_PER_STREAM * per_shot;
    let run = |block: usize, rows: &mut [T]| {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(block as u64 + 2);
        sample_block(&mut rng, rows);
    };

    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        rows.par_chunks_mut(block_len)
            .enumerate()
            .for_each(|(block, rows)| run(block, rows));
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    for (block, rows) in rows.chunks_mut(block_len).enumerate() {
        run(block, rows);
    }
}

pub(crate) fn sample_from_cdf(cdf: &[f64], r: f64) -> usize {
    match cdf.binary_search_by(|p| p.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i.min(cdf.len() - 1),
    }
}

pub(crate) fn sample_shots(
    probs: &Probabilities,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    if meas_map.is_empty() {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    let mut shots = vec![vec![false; num_classical_bits]; num_shots];

    match probs {
        Probabilities::Dense(v) => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let cdf = build_cdf(v);
            for shot in &mut shots {
                let r: f64 = rng.random();
                let state_idx = sample_from_cdf(&cdf, r);
                for &(qubit, cbit) in meas_map {
                    shot[cbit] = (state_idx >> qubit) & 1 == 1;
                }
            }
        }
        // Same blocks, streams and per-shot draw order as the factored
        // backend's native sampler, so the two routes agree shot for shot.
        Probabilities::Factored { blocks, .. } => {
            let block_cdfs: Vec<Vec<f64>> = blocks.iter().map(|b| build_cdf(&b.probs)).collect();
            let parallel = draws_split_across_workers(num_shots, blocks.len());
            sample_in_shot_blocks(&mut shots, 1, seed, parallel, |rng, block_shots| {
                for shot in block_shots {
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
            });
        }
    }

    shots
}

/// Project packed per-qubit outcomes from a native backend sampler onto
/// classical bits. Later measurements of the same classical bit win, matching
/// [`sample_shots`].
pub(crate) fn shots_from_basis_samples(
    samples: &crate::backend::BasisSamples,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
) -> Vec<Vec<bool>> {
    let dense_identity_map = meas_map.len() == num_classical_bits
        && meas_map
            .iter()
            .enumerate()
            .all(|(idx, &(qubit, classical_bit))| idx == qubit && idx == classical_bit);
    if dense_identity_map {
        return samples.to_shots(num_classical_bits);
    }

    let mut shots = vec![vec![false; num_classical_bits]; samples.num_shots()];
    for (index, shot) in shots.iter_mut().enumerate() {
        for &(qubit, cbit) in meas_map {
            shot[cbit] = samples.bit(index, qubit);
        }
    }
    shots
}

/// Re-key counts drawn in measurement-record order onto the classical
/// register, merging records that land on the same classical outcome. A later
/// measurement of a classical bit overwrites an earlier one.
pub(super) fn record_counts_to_classical_bits(
    counts: HashMap<Vec<u64>, u64>,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
) -> HashMap<Vec<u64>, u64> {
    let words = num_classical_bits.div_ceil(64).max(1);
    let in_order = meas_map.len() <= num_classical_bits
        && meas_map
            .iter()
            .enumerate()
            .all(|(idx, &(_, classical_bit))| idx == classical_bit);
    if in_order && counts.keys().all(|key| key.len() == words) {
        return counts;
    }

    let mut out: HashMap<Vec<u64>, u64> = HashMap::with_capacity(counts.len());
    for (record, count) in counts {
        let key = if in_order {
            let mut key = record;
            key.resize(words, 0);
            key
        } else {
            let mut key = vec![0u64; words];
            for (idx, &(_, classical_bit)) in meas_map.iter().enumerate() {
                if classical_bit >= num_classical_bits {
                    continue;
                }
                let bit = record
                    .get(idx / 64)
                    .is_some_and(|w| (w >> (idx % 64)) & 1 == 1);
                let mask = 1u64 << (classical_bit % 64);
                if bit {
                    key[classical_bit / 64] |= mask;
                } else {
                    key[classical_bit / 64] &= !mask;
                }
            }
            key
        };
        *out.entry(key).or_insert(0) += count;
    }
    out
}

/// Histogram of native samples keyed as [`ShotsResult::counts`] keys them,
/// equal to counting [`shots_from_basis_samples`]. Rows are counted with
/// unmeasured qubits masked off, so only distinct outcomes pay the projection
/// onto classical bits.
pub(crate) fn counts_from_basis_samples(
    samples: &crate::backend::BasisSamples,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
) -> HashMap<Vec<u64>, u64> {
    let words_per_shot = samples.words_per_shot();
    let mut mask = vec![0u64; words_per_shot];
    for &(qubit, _) in meas_map {
        mask[qubit / 64] |= 1u64 << (qubit % 64);
    }
    let m_words = num_classical_bits.div_ceil(64).max(1);
    let in_place = meas_map.iter().all(|&(qubit, cbit)| qubit == cbit);
    let classical_key = |row: &[u64]| -> Vec<u64> {
        let mut key = vec![0u64; m_words];
        if in_place {
            let live = m_words.min(row.len());
            key[..live].copy_from_slice(&row[..live]);
        } else {
            for &(qubit, cbit) in meas_map {
                let bit = (row[qubit / 64] >> (qubit % 64)) & 1;
                let word = &mut key[cbit / 64];
                *word = (*word & !(1u64 << (cbit % 64))) | (bit << (cbit % 64));
            }
        }
        key
    };

    let mut counts = HashMap::new();
    if words_per_shot == 1 {
        let mask = mask[0];
        let mut rows: FxHashMap<u64, u64> = FxHashMap::default();
        for &word in samples.words() {
            *rows.entry(word & mask).or_insert(0) += 1;
        }
        for (row, n) in rows {
            *counts.entry(classical_key(&[row])).or_insert(0) += n;
        }
    } else {
        let mut rows: FxHashMap<Vec<u64>, u64> = FxHashMap::default();
        let mut scratch = vec![0u64; words_per_shot];
        for row in samples.words().chunks_exact(words_per_shot) {
            for ((dst, &word), &m) in scratch.iter_mut().zip(row).zip(&mask) {
                *dst = word & m;
            }
            match rows.get_mut(scratch.as_slice()) {
                Some(n) => *n += 1,
                None => {
                    rows.insert(scratch.clone(), 1);
                }
            }
        }
        for (row, n) in rows {
            *counts.entry(classical_key(&row)).or_insert(0) += n;
        }
    }
    counts
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
#[path = "shots_tests.rs"]
mod tests;
