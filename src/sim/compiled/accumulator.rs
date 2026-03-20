use std::collections::HashMap;

use super::{PackedShots, ShotLayout};

const DEFAULT_TARGET_BYTES: usize = 256 * 1024 * 1024;

pub fn optimal_chunk_size(num_meas: usize, target_bytes: usize) -> usize {
    let m_words = num_meas.div_ceil(64);
    let bytes_per_shot_word = m_words * 8;
    if bytes_per_shot_word == 0 {
        return 1 << 20;
    }
    let raw = target_bytes / bytes_per_shot_word;
    let aligned = (raw / 64) * 64;
    aligned.max(64)
}

pub fn default_chunk_size(num_meas: usize) -> usize {
    optimal_chunk_size(num_meas, DEFAULT_TARGET_BYTES)
}

pub trait ShotAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots);
}

pub struct HistogramAccumulator {
    counts: HashMap<Vec<u64>, u64>,
}

impl Default for HistogramAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramAccumulator {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
        }
    }

    pub fn into_counts(self) -> HashMap<Vec<u64>, u64> {
        self.counts
    }

    pub fn counts(&self) -> &HashMap<Vec<u64>, u64> {
        &self.counts
    }
}

impl ShotAccumulator for HistogramAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        for (key, count) in chunk.counts() {
            *self.counts.entry(key).or_insert(0) += count;
        }
    }
}

pub struct MarginalsAccumulator {
    ones: Vec<u64>,
    total_shots: u64,
}

impl MarginalsAccumulator {
    pub fn new(num_measurements: usize) -> Self {
        Self {
            ones: vec![0u64; num_measurements],
            total_shots: 0,
        }
    }

    pub fn marginals(&self) -> Vec<f64> {
        if self.total_shots == 0 {
            return vec![0.0; self.ones.len()];
        }
        self.ones
            .iter()
            .map(|&c| c as f64 / self.total_shots as f64)
            .collect()
    }

    pub fn total_shots(&self) -> u64 {
        self.total_shots
    }
}

impl ShotAccumulator for MarginalsAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        let n = chunk.num_shots();
        let m = chunk.num_measurements();
        self.total_shots += n as u64;

        match chunk.layout() {
            ShotLayout::MeasMajor => {
                let s_words = chunk.s_words();
                let tail_bits = n % 64;
                let tail_mask = if tail_bits == 0 {
                    !0u64
                } else {
                    (1u64 << tail_bits) - 1
                };
                for mi in 0..m {
                    let row = chunk.meas_words(mi);
                    let mut count = 0u64;
                    if s_words > 0 {
                        for &w in &row[..s_words - 1] {
                            count += w.count_ones() as u64;
                        }
                        count += (row[s_words - 1] & tail_mask).count_ones() as u64;
                    }
                    self.ones[mi] += count;
                }
            }
            ShotLayout::ShotMajor => {
                for s in 0..n {
                    let words = chunk.shot_words(s);
                    for mi in 0..m {
                        let w = mi / 64;
                        if (words[w] >> (mi % 64)) & 1 != 0 {
                            self.ones[mi] += 1;
                        }
                    }
                }
            }
        }
    }
}

pub struct PauliExpectationAccumulator {
    observables: Vec<Vec<usize>>,
    parity_ones: Vec<u64>,
    parity_buf: Vec<u64>,
    total_shots: u64,
}

impl PauliExpectationAccumulator {
    pub fn new(observables: Vec<Vec<usize>>) -> Self {
        let k = observables.len();
        Self {
            observables,
            parity_ones: vec![0u64; k],
            parity_buf: Vec::new(),
            total_shots: 0,
        }
    }

    pub fn expectations(&self) -> Vec<f64> {
        if self.total_shots == 0 {
            return vec![0.0; self.observables.len()];
        }
        self.parity_ones
            .iter()
            .map(|&p| 1.0 - 2.0 * p as f64 / self.total_shots as f64)
            .collect()
    }

    pub fn total_shots(&self) -> u64 {
        self.total_shots
    }
}

impl ShotAccumulator for PauliExpectationAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        let n = chunk.num_shots();
        self.total_shots += n as u64;

        match chunk.layout() {
            ShotLayout::MeasMajor => {
                let s_words = chunk.s_words();
                let tail_bits = n % 64;
                let tail_mask = if tail_bits == 0 {
                    !0u64
                } else {
                    (1u64 << tail_bits) - 1
                };
                self.parity_buf.resize(s_words, 0);
                for (k, obs) in self.observables.iter().enumerate() {
                    self.parity_buf.fill(0);
                    for &mi in obs {
                        let row = chunk.meas_words(mi);
                        for (p, &r) in self.parity_buf.iter_mut().zip(row.iter()) {
                            *p ^= r;
                        }
                    }
                    let mut count = 0u64;
                    if s_words > 0 {
                        for &w in &self.parity_buf[..s_words - 1] {
                            count += w.count_ones() as u64;
                        }
                        count += (self.parity_buf[s_words - 1] & tail_mask).count_ones() as u64;
                    }
                    self.parity_ones[k] += count;
                }
            }
            ShotLayout::ShotMajor => {
                for s in 0..n {
                    let words = chunk.shot_words(s);
                    for (k, obs) in self.observables.iter().enumerate() {
                        let mut parity = false;
                        for &mi in obs {
                            let w = mi / 64;
                            if (words[w] >> (mi % 64)) & 1 != 0 {
                                parity = !parity;
                            }
                        }
                        if parity {
                            self.parity_ones[k] += 1;
                        }
                    }
                }
            }
        }
    }
}

pub struct CorrelatorAccumulator {
    pairs: Vec<(usize, usize)>,
    differ_count: Vec<u64>,
    total_shots: u64,
}

impl CorrelatorAccumulator {
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        let k = pairs.len();
        Self {
            pairs,
            differ_count: vec![0u64; k],
            total_shots: 0,
        }
    }

    pub fn correlators(&self) -> Vec<f64> {
        if self.total_shots == 0 {
            return vec![0.0; self.pairs.len()];
        }
        self.differ_count
            .iter()
            .map(|&d| 1.0 - 2.0 * d as f64 / self.total_shots as f64)
            .collect()
    }

    pub fn total_shots(&self) -> u64 {
        self.total_shots
    }
}

impl ShotAccumulator for CorrelatorAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        let n = chunk.num_shots();
        self.total_shots += n as u64;

        match chunk.layout() {
            ShotLayout::MeasMajor => {
                let s_words = chunk.s_words();
                let tail_bits = n % 64;
                let tail_mask = if tail_bits == 0 {
                    !0u64
                } else {
                    (1u64 << tail_bits) - 1
                };
                for (k, &(i, j)) in self.pairs.iter().enumerate() {
                    let row_i = chunk.meas_words(i);
                    let row_j = chunk.meas_words(j);
                    let mut count = 0u64;
                    if s_words > 0 {
                        for idx in 0..s_words - 1 {
                            count += (row_i[idx] ^ row_j[idx]).count_ones() as u64;
                        }
                        count += ((row_i[s_words - 1] ^ row_j[s_words - 1]) & tail_mask)
                            .count_ones() as u64;
                    }
                    self.differ_count[k] += count;
                }
            }
            ShotLayout::ShotMajor => {
                for s in 0..n {
                    let words = chunk.shot_words(s);
                    for (k, &(i, j)) in self.pairs.iter().enumerate() {
                        let bi = (words[i / 64] >> (i % 64)) & 1;
                        let bj = (words[j / 64] >> (j % 64)) & 1;
                        if bi != bj {
                            self.differ_count[k] += 1;
                        }
                    }
                }
            }
        }
    }
}
