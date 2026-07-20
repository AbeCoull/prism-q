use std::collections::HashMap;

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::shots::{build_cdf, sample_from_cdf};

const MAX_DENSE_COUNT_BINS: usize = 1 << 20;

#[derive(Clone)]
struct CompactMeasurementMap {
    cbits: Vec<usize>,
    qubits: Vec<usize>,
    prefix_bits: Option<usize>,
    pext_mask: Option<u64>,
}

impl CompactMeasurementMap {
    fn new(meas_map: &[(usize, usize)], num_classical_bits: usize) -> Self {
        let mut final_qubit_for_cbit = vec![None; num_classical_bits];
        for &(qubit, cbit) in meas_map {
            final_qubit_for_cbit[cbit] = Some(qubit);
        }

        let mut cbits = Vec::with_capacity(meas_map.len());
        let mut qubits = Vec::with_capacity(meas_map.len());
        for (cbit, qubit) in final_qubit_for_cbit.into_iter().enumerate() {
            if let Some(qubit) = qubit {
                cbits.push(cbit);
                qubits.push(qubit);
            }
        }

        let prefix_bits = qubits
            .iter()
            .enumerate()
            .all(|(bit, &qubit)| bit == qubit)
            .then_some(qubits.len());

        let pext_mask = if qubits.len() <= u64::BITS as usize
            && qubits.iter().all(|&qubit| qubit < u64::BITS as usize)
            && qubits.windows(2).all(|w| w[0] < w[1])
        {
            let mut mask = 0u64;
            for &qubit in &qubits {
                mask |= 1u64 << qubit;
            }
            Some(mask)
        } else {
            None
        };

        Self {
            cbits,
            qubits,
            prefix_bits,
            pext_mask,
        }
    }

    #[inline]
    fn num_bits(&self) -> usize {
        self.cbits.len()
    }

    #[inline]
    fn compact_key(&self, state_idx: usize) -> usize {
        debug_assert!(self.num_bits() < usize::BITS as usize);
        if let Some(bits) = self.prefix_bits {
            return state_idx & ((1usize << bits) - 1);
        }
        if let Some(mask) = self.pext_mask {
            return extract_masked_bits(state_idx, mask);
        }

        let mut key = 0usize;
        for (bit, &qubit) in self.qubits.iter().enumerate() {
            key |= ((state_idx >> qubit) & 1) << bit;
        }
        key
    }

    #[inline]
    fn direct_state_key(&self, state_len: usize) -> bool {
        self.prefix_bits == Some(self.num_bits()) && (1usize << self.num_bits()) == state_len
    }

    #[inline(always)]
    fn fill_shot_from_state_index(&self, state_idx: usize, shot: &mut [bool]) {
        for (&qubit, &cbit) in self.qubits.iter().zip(self.cbits.iter()) {
            shot[cbit] = (state_idx >> qubit) & 1 == 1;
        }
    }

    #[inline(always)]
    fn packed_key_from_compact(&self, key: usize, m_words: usize) -> Vec<u64> {
        let mut packed = vec![0u64; m_words];
        for (bit, &cbit) in self.cbits.iter().enumerate() {
            if (key >> bit) & 1 == 1 {
                packed[cbit / 64] |= 1u64 << (cbit % 64);
            }
        }
        packed
    }

    #[inline(always)]
    fn packed_key_from_state_index(&self, state_idx: usize, m_words: usize) -> Vec<u64> {
        let mut packed = vec![0u64; m_words];
        for (&qubit, &cbit) in self.qubits.iter().zip(self.cbits.iter()) {
            if (state_idx >> qubit) & 1 == 1 {
                packed[cbit / 64] |= 1u64 << (cbit % 64);
            }
        }
        packed
    }
}

#[cfg(target_arch = "x86_64")]
fn bmi2_available() -> bool {
    static BMI2: OnceLock<bool> = OnceLock::new();
    *BMI2.get_or_init(|| is_x86_feature_detected!("bmi2"))
}

#[inline]
fn extract_masked_bits(index: usize, mask: u64) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if bmi2_available() {
            // SAFETY: BMI2 availability is checked immediately before this call.
            return unsafe { core::arch::x86_64::_pext_u64(index as u64, mask) as usize };
        }
    }

    let mut result = 0usize;
    let mut bit = 0;
    let mut m = mask;
    while m != 0 {
        let pos = m.trailing_zeros() as usize;
        if index & (1usize << pos) != 0 {
            result |= 1usize << bit;
        }
        bit += 1;
        m &= m.wrapping_sub(1);
    }
    result
}

fn build_state_outcome_distribution(
    state: &[Complex64],
    norm_sq: f64,
    map: &CompactMeasurementMap,
    max_dense_bits: usize,
) -> Option<Vec<f64>> {
    if map.num_bits() > max_dense_bits {
        return None;
    }

    let outcomes = 1usize << map.num_bits();

    #[cfg(feature = "parallel")]
    {
        if map.direct_state_key(state.len()) {
            let mut probs = vec![0.0f64; outcomes];
            probs
                .par_iter_mut()
                .zip(state.par_iter())
                .for_each(|(p, amp)| {
                    *p = amp.norm_sqr() * norm_sq;
                });
            return Some(probs);
        }

        if state.len() >= crate::backend::MIN_PAR_ELEMS && outcomes <= (1usize << 16) {
            let probs = state
                .par_chunks(crate::backend::MIN_PAR_ELEMS)
                .enumerate()
                .map(|(chunk_idx, chunk)| {
                    let mut local = vec![0.0f64; outcomes];
                    let base = chunk_idx * crate::backend::MIN_PAR_ELEMS;
                    for (offset, amp) in chunk.iter().enumerate() {
                        let key = map.compact_key(base + offset);
                        local[key] += amp.norm_sqr() * norm_sq;
                    }
                    local
                })
                .reduce(
                    || vec![0.0f64; outcomes],
                    |mut a, b| {
                        for (dst, src) in a.iter_mut().zip(b) {
                            *dst += src;
                        }
                        a
                    },
                );
            return Some(probs);
        }
    }

    let mut probs = vec![0.0f64; outcomes];
    if map.direct_state_key(state.len()) {
        for (idx, amp) in state.iter().enumerate() {
            probs[idx] = amp.norm_sqr() * norm_sq;
        }
    } else {
        for (idx, amp) in state.iter().enumerate() {
            let key = map.compact_key(idx);
            probs[key] += amp.norm_sqr() * norm_sq;
        }
    }
    Some(probs)
}

fn packed_counts_from_dense(
    compact_counts: Vec<u64>,
    map: &CompactMeasurementMap,
    m_words: usize,
) -> HashMap<Vec<u64>, u64> {
    let nonzero = compact_counts.iter().filter(|&&count| count != 0).count();
    let mut counts = HashMap::with_capacity(nonzero);
    for (key, count) in compact_counts.into_iter().enumerate() {
        if count != 0 {
            counts.insert(map.packed_key_from_compact(key, m_words), count);
        }
    }
    counts
}

fn packed_counts_from_sparse(
    compact_counts: HashMap<usize, u64>,
    map: &CompactMeasurementMap,
    m_words: usize,
) -> HashMap<Vec<u64>, u64> {
    let mut counts = HashMap::with_capacity(compact_counts.len());
    for (key, count) in compact_counts {
        counts.insert(map.packed_key_from_compact(key, m_words), count);
    }
    counts
}

fn sample_counts_from_outcome_distribution(
    probs: &[f64],
    map: &CompactMeasurementMap,
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> HashMap<Vec<u64>, u64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let cdf = build_cdf(probs);
    let m_words = num_classical_bits.div_ceil(64).max(1);

    if probs.len() <= MAX_DENSE_COUNT_BINS {
        let mut compact_counts = vec![0u64; probs.len()];
        for _ in 0..num_shots {
            let r: f64 = rng.random();
            let key = sample_from_cdf(&cdf, r);
            compact_counts[key] += 1;
        }
        return packed_counts_from_dense(compact_counts, map, m_words);
    }

    let mut compact_counts = HashMap::with_capacity(num_shots.min(probs.len()));
    for _ in 0..num_shots {
        let r: f64 = rng.random();
        let key = sample_from_cdf(&cdf, r);
        *compact_counts.entry(key).or_insert(0) += 1;
    }
    packed_counts_from_sparse(compact_counts, map, m_words)
}

fn sorted_thresholds(num_shots: usize, seed: u64) -> Vec<(f64, usize)> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut thresholds: Vec<_> = (0..num_shots)
        .map(|shot| (rng.random::<f64>(), shot))
        .collect();
    thresholds.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    thresholds
}

fn sample_shots_streaming_from_state(
    state: &[Complex64],
    norm_sq: f64,
    map: &CompactMeasurementMap,
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    let thresholds = sorted_thresholds(num_shots, seed);
    let mut shots = vec![vec![false; num_classical_bits]; num_shots];
    let mut cumulative = 0.0f64;
    let mut next = 0usize;

    for (state_idx, amp) in state.iter().enumerate() {
        cumulative += amp.norm_sqr() * norm_sq;
        while next < thresholds.len() && thresholds[next].0 <= cumulative {
            let shot_idx = thresholds[next].1;
            map.fill_shot_from_state_index(state_idx, &mut shots[shot_idx]);
            next += 1;
        }
    }

    let fallback_idx = state.len().saturating_sub(1);
    while next < thresholds.len() {
        let shot_idx = thresholds[next].1;
        map.fill_shot_from_state_index(fallback_idx, &mut shots[shot_idx]);
        next += 1;
    }

    shots
}

fn sample_counts_streaming_from_state(
    state: &[Complex64],
    norm_sq: f64,
    map: &CompactMeasurementMap,
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> HashMap<Vec<u64>, u64> {
    let thresholds = sorted_thresholds(num_shots, seed);
    let m_words = num_classical_bits.div_ceil(64).max(1);
    let mut counts = HashMap::with_capacity(num_shots.min(state.len()));
    let mut cumulative = 0.0f64;
    let mut next = 0usize;

    for (state_idx, amp) in state.iter().enumerate() {
        cumulative += amp.norm_sqr() * norm_sq;
        let start = next;
        while next < thresholds.len() && thresholds[next].0 <= cumulative {
            next += 1;
        }
        let hits = next - start;
        if hits != 0 {
            let packed = map.packed_key_from_state_index(state_idx, m_words);
            *counts.entry(packed).or_insert(0) += hits as u64;
        }
    }

    let fallback_idx = state.len().saturating_sub(1);
    let fallback_hits = thresholds.len() - next;
    if fallback_hits != 0 {
        let packed = map.packed_key_from_state_index(fallback_idx, m_words);
        *counts.entry(packed).or_insert(0) += fallback_hits as u64;
    }

    counts
}

fn build_probs_outcome_distribution(
    probs: &[f64],
    map: &CompactMeasurementMap,
    max_dense_bits: usize,
) -> Option<Vec<f64>> {
    if map.num_bits() > max_dense_bits {
        return None;
    }

    let outcomes = 1usize << map.num_bits();

    #[cfg(feature = "parallel")]
    if probs.len() >= crate::backend::MIN_PAR_ELEMS && outcomes <= (1usize << 16) {
        let out = probs
            .par_chunks(crate::backend::MIN_PAR_ELEMS)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let mut local = vec![0.0f64; outcomes];
                let base = chunk_idx * crate::backend::MIN_PAR_ELEMS;
                for (offset, &p) in chunk.iter().enumerate() {
                    let key = map.compact_key(base + offset);
                    local[key] += p;
                }
                local
            })
            .reduce(
                || vec![0.0f64; outcomes],
                |mut a, b| {
                    for (dst, src) in a.iter_mut().zip(b) {
                        *dst += src;
                    }
                    a
                },
            );
        return Some(out);
    }

    let mut out = vec![0.0f64; outcomes];
    for (idx, &p) in probs.iter().enumerate() {
        out[map.compact_key(idx)] += p;
    }
    Some(out)
}

fn sample_shots_streaming_from_probs(
    probs: &[f64],
    map: &CompactMeasurementMap,
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    let thresholds = sorted_thresholds(num_shots, seed);
    let mut shots = vec![vec![false; num_classical_bits]; num_shots];
    let mut cumulative = 0.0f64;
    let mut next = 0usize;

    for (state_idx, &p) in probs.iter().enumerate() {
        cumulative += p;
        while next < thresholds.len() && thresholds[next].0 <= cumulative {
            let shot_idx = thresholds[next].1;
            map.fill_shot_from_state_index(state_idx, &mut shots[shot_idx]);
            next += 1;
        }
    }

    let fallback_idx = probs.len().saturating_sub(1);
    while next < thresholds.len() {
        let shot_idx = thresholds[next].1;
        map.fill_shot_from_state_index(fallback_idx, &mut shots[shot_idx]);
        next += 1;
    }

    shots
}

fn sample_counts_streaming_from_probs(
    probs: &[f64],
    map: &CompactMeasurementMap,
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> HashMap<Vec<u64>, u64> {
    let thresholds = sorted_thresholds(num_shots, seed);
    let m_words = num_classical_bits.div_ceil(64).max(1);
    let mut counts = HashMap::with_capacity(num_shots.min(probs.len()));
    let mut cumulative = 0.0f64;
    let mut next = 0usize;

    for (state_idx, &p) in probs.iter().enumerate() {
        cumulative += p;
        let start = next;
        while next < thresholds.len() && thresholds[next].0 <= cumulative {
            next += 1;
        }
        let hits = next - start;
        if hits != 0 {
            let packed = map.packed_key_from_state_index(state_idx, m_words);
            *counts.entry(packed).or_insert(0) += hits as u64;
        }
    }

    let fallback_idx = probs.len().saturating_sub(1);
    let fallback_hits = thresholds.len() - next;
    if fallback_hits != 0 {
        let packed = map.packed_key_from_state_index(fallback_idx, m_words);
        *counts.entry(packed).or_insert(0) += fallback_hits as u64;
    }

    counts
}

pub(super) fn sample_shots_from_probs(
    probs: &[f64],
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    if meas_map.is_empty() {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    let map = CompactMeasurementMap::new(meas_map, num_classical_bits);
    if map.num_bits() == 0 {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    sample_shots_streaming_from_probs(probs, &map, num_classical_bits, num_shots, seed)
}

pub(super) fn sample_counts_from_probs(
    probs: &[f64],
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> HashMap<Vec<u64>, u64> {
    sample_counts_from_probs_capped(
        probs,
        meas_map,
        num_classical_bits,
        num_shots,
        seed,
        crate::backend::max_dense_outcome_bits(),
    )
}

fn sample_counts_from_probs_capped(
    probs: &[f64],
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
    max_dense_bits: usize,
) -> HashMap<Vec<u64>, u64> {
    if num_shots == 0 {
        return HashMap::new();
    }

    let m_words = num_classical_bits.div_ceil(64).max(1);
    if meas_map.is_empty() {
        let mut counts = HashMap::with_capacity(1);
        counts.insert(vec![0u64; m_words], num_shots as u64);
        return counts;
    }

    let map = CompactMeasurementMap::new(meas_map, num_classical_bits);
    if map.num_bits() == 0 {
        let mut counts = HashMap::with_capacity(1);
        counts.insert(vec![0u64; m_words], num_shots as u64);
        return counts;
    }

    if map.num_bits() <= max_dense_bits && map.direct_state_key(probs.len()) {
        return sample_counts_from_outcome_distribution(
            probs,
            &map,
            num_classical_bits,
            num_shots,
            seed,
        );
    }

    if let Some(outcome) = build_probs_outcome_distribution(probs, &map, max_dense_bits) {
        sample_counts_from_outcome_distribution(&outcome, &map, num_classical_bits, num_shots, seed)
    } else {
        sample_counts_streaming_from_probs(probs, &map, num_classical_bits, num_shots, seed)
    }
}

pub(super) fn sample_shots_from_state(
    state: &[Complex64],
    norm_sq: f64,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    if meas_map.is_empty() {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    let map = CompactMeasurementMap::new(meas_map, num_classical_bits);
    if map.num_bits() == 0 {
        return vec![vec![false; num_classical_bits]; num_shots];
    }

    sample_shots_streaming_from_state(state, norm_sq, &map, num_classical_bits, num_shots, seed)
}

pub(super) fn sample_counts_from_state(
    state: &[Complex64],
    norm_sq: f64,
    meas_map: &[(usize, usize)],
    num_classical_bits: usize,
    num_shots: usize,
    seed: u64,
) -> HashMap<Vec<u64>, u64> {
    if num_shots == 0 {
        return HashMap::new();
    }

    let m_words = num_classical_bits.div_ceil(64).max(1);
    if meas_map.is_empty() {
        let mut counts = HashMap::with_capacity(1);
        counts.insert(vec![0u64; m_words], num_shots as u64);
        return counts;
    }

    let map = CompactMeasurementMap::new(meas_map, num_classical_bits);
    if map.num_bits() == 0 {
        let mut counts = HashMap::with_capacity(1);
        counts.insert(vec![0u64; m_words], num_shots as u64);
        return counts;
    }

    if let Some(probs) = build_state_outcome_distribution(
        state,
        norm_sq,
        &map,
        crate::backend::max_dense_outcome_bits(),
    ) {
        sample_counts_from_outcome_distribution(&probs, &map, num_classical_bits, num_shots, seed)
    } else {
        sample_counts_streaming_from_state(
            state,
            norm_sq,
            &map,
            num_classical_bits,
            num_shots,
            seed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Above the dense-outcome cap the direct-key shortcut must not build a
    /// dense CDF; counts route through the same streaming sampler as the host
    /// path, so the entry point and the streaming sampler agree byte-for-byte.
    /// Below the cap the shortcut takes the CDF sampler, which draws from a
    /// different RNG stream.
    #[test]
    fn counts_from_probs_stream_above_dense_outcome_cap() {
        let bits = 11;
        let len = 1usize << bits;
        let mut probs = vec![0.0f64; len];
        let spikes = [0usize, 123, len / 3, len / 2 + 7, len - 1];
        for &s in &spikes {
            probs[s] = 0.2;
        }
        let meas_map: Vec<(usize, usize)> = (0..bits).map(|q| (q, q)).collect();
        let map = CompactMeasurementMap::new(&meas_map, bits);
        assert!(map.direct_state_key(len));

        let above_cap = sample_counts_from_probs_capped(&probs, &meas_map, bits, 300, 42, bits - 1);
        let via_streaming = sample_counts_streaming_from_probs(&probs, &map, bits, 300, 42);
        assert_eq!(above_cap, via_streaming);

        let below_cap = sample_counts_from_probs_capped(&probs, &meas_map, bits, 300, 42, bits);
        let total: u64 = below_cap.values().sum();
        assert_eq!(total, 300);
    }
}
