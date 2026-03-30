use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

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

struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x517cc1b727220a95;
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            let word = u64::from_ne_bytes(buf);
            self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(Self::SEED);
        }
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(Self::SEED);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.hash = (self.hash.rotate_left(5) ^ i as u64).wrapping_mul(Self::SEED);
    }
}

#[derive(Clone, Default)]
struct FxBuildHasher;

impl BuildHasher for FxBuildHasher {
    type Hasher = FxHasher;
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: 0 }
    }
}

type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

#[cfg(feature = "parallel")]
const MIN_SHOTS_FOR_PAR_HISTOGRAM: usize = 65536;

pub trait ShotAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots);
}

macro_rules! inline_counts_variant {
    ($cap:expr, $variant:ident) => {{
        let mut m = FxHashMap::default();
        m.reserve($cap);
        Self::$variant(m)
    }};
}

enum InlineCounts {
    Uninit,
    W1(FxHashMap<[u64; 1], u64>),
    W2(FxHashMap<[u64; 2], u64>),
    W3(FxHashMap<[u64; 3], u64>),
    W4(FxHashMap<[u64; 4], u64>),
    W5(FxHashMap<[u64; 5], u64>),
    W6(FxHashMap<[u64; 6], u64>),
    W7(FxHashMap<[u64; 7], u64>),
    W8(FxHashMap<[u64; 8], u64>),
    Wide(FxHashMap<Vec<u64>, u64>),
}

impl InlineCounts {
    fn init(m_words: usize, capacity: usize) -> Self {
        match m_words {
            1 => inline_counts_variant!(capacity, W1),
            2 => inline_counts_variant!(capacity, W2),
            3 => inline_counts_variant!(capacity, W3),
            4 => inline_counts_variant!(capacity, W4),
            5 => inline_counts_variant!(capacity, W5),
            6 => inline_counts_variant!(capacity, W6),
            7 => inline_counts_variant!(capacity, W7),
            8 => inline_counts_variant!(capacity, W8),
            _ => inline_counts_variant!(capacity, Wide),
        }
    }

    fn into_vec_counts(self) -> HashMap<Vec<u64>, u64> {
        match self {
            Self::Uninit => HashMap::new(),
            Self::W1(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W2(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W3(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W4(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W5(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W6(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W7(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::W8(m) => m.into_iter().map(|(k, v)| (k.to_vec(), v)).collect(),
            Self::Wide(m) => m.into_iter().collect(),
        }
    }
}

pub struct HistogramAccumulator {
    counts: InlineCounts,
    batch_buf: Vec<u64>,
}

impl Default for HistogramAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl HistogramAccumulator {
    pub fn new() -> Self {
        Self {
            counts: InlineCounts::Uninit,
            batch_buf: Vec::new(),
        }
    }

    pub fn into_counts(self) -> HashMap<Vec<u64>, u64> {
        self.counts.into_vec_counts()
    }
}

#[inline]
fn count_shot_typed<const M: usize>(counts: &mut FxHashMap<[u64; M], u64>, key: [u64; M]) {
    *counts.entry(key).or_insert(0) += 1;
}

#[inline]
fn count_shot_wide(counts: &mut FxHashMap<Vec<u64>, u64>, key: &[u64]) {
    if let Some(count) = counts.get_mut(key) {
        *count += 1;
    } else {
        counts.insert(key.to_vec(), 1);
    }
}

const TRANSPOSE_MASKS: [u64; 6] = [
    0x00000000FFFFFFFF,
    0x0000FFFF0000FFFF,
    0x00FF00FF00FF00FF,
    0x0F0F0F0F0F0F0F0F,
    0x3333333333333333,
    0x5555555555555555,
];
const TRANSPOSE_DELTAS: [usize; 6] = [32, 16, 8, 4, 2, 1];

fn transpose_64x64(matrix: &mut [u64; 64]) {
    for (&delta, &mask) in TRANSPOSE_DELTAS.iter().zip(TRANSPOSE_MASKS.iter()) {
        let nmask = !mask;
        let mut i = 0;
        while i < 64 {
            for j in 0..delta {
                let a = matrix[i + j];
                let b = matrix[i + j + delta];
                matrix[i + j] = (a & mask) | ((b & mask) << delta);
                matrix[i + j + delta] = ((a >> delta) & mask) | (b & nmask);
            }
            i += 2 * delta;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn transpose_64x64_avx2(matrix: &mut [u64; 64]) {
    use std::arch::x86_64::*;

    for (&delta, &mask_val) in TRANSPOSE_DELTAS.iter().zip(TRANSPOSE_MASKS.iter()) {
        let vmask = _mm256_set1_epi64x(mask_val as i64);
        let vnmask = _mm256_set1_epi64x(!mask_val as i64);
        let shift = _mm_set_epi64x(0, delta as i64);

        let mut i = 0;
        while i < 64 {
            let mut j = 0;
            while j + 4 <= delta {
                let a = _mm256_loadu_si256(matrix.as_ptr().add(i + j) as *const __m256i);
                let b = _mm256_loadu_si256(matrix.as_ptr().add(i + j + delta) as *const __m256i);

                let a_low = _mm256_and_si256(a, vmask);
                let b_low = _mm256_and_si256(b, vmask);
                let b_shifted = _mm256_sll_epi64(b_low, shift);
                let new_a = _mm256_or_si256(a_low, b_shifted);

                let a_shifted = _mm256_srl_epi64(a, shift);
                let a_high = _mm256_and_si256(a_shifted, vmask);
                let b_high = _mm256_and_si256(b, vnmask);
                let new_b = _mm256_or_si256(a_high, b_high);

                _mm256_storeu_si256(matrix.as_mut_ptr().add(i + j) as *mut __m256i, new_a);
                _mm256_storeu_si256(
                    matrix.as_mut_ptr().add(i + j + delta) as *mut __m256i,
                    new_b,
                );

                j += 4;
            }
            while j < delta {
                let a = *matrix.get_unchecked(i + j);
                let b = *matrix.get_unchecked(i + j + delta);
                *matrix.get_unchecked_mut(i + j) = (a & mask_val) | ((b & mask_val) << delta);
                *matrix.get_unchecked_mut(i + j + delta) =
                    ((a >> delta) & mask_val) | (b & !mask_val);
                j += 1;
            }
            i += 2 * delta;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn transpose_64x64_avx2(_matrix: &mut [u64; 64]) {
    unreachable!()
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn transpose_64x64_neon(matrix: &mut [u64; 64]) {
    use std::arch::aarch64::*;

    for (&delta, &mask_val) in TRANSPOSE_DELTAS.iter().zip(TRANSPOSE_MASKS.iter()) {
        let vmask = vdupq_n_u64(mask_val);
        let vnmask = vdupq_n_u64(!mask_val);
        let vshift_left = vdupq_n_s64(delta as i64);
        let vshift_right = vdupq_n_s64(-(delta as i64));

        let mut i = 0;
        while i < 64 {
            let mut j = 0;
            while j + 2 <= delta {
                let a = vld1q_u64(matrix.as_ptr().add(i + j));
                let b = vld1q_u64(matrix.as_ptr().add(i + j + delta));

                let a_low = vandq_u64(a, vmask);
                let b_low = vandq_u64(b, vmask);
                let b_shifted = vshlq_u64(b_low, vshift_left);
                let new_a = vorrq_u64(a_low, b_shifted);

                let a_shifted = vshlq_u64(a, vshift_right);
                let a_high = vandq_u64(a_shifted, vmask);
                let b_high = vandq_u64(b, vnmask);
                let new_b = vorrq_u64(a_high, b_high);

                vst1q_u64(matrix.as_mut_ptr().add(i + j), new_a);
                vst1q_u64(matrix.as_mut_ptr().add(i + j + delta), new_b);

                j += 2;
            }
            while j < delta {
                let a = *matrix.get_unchecked(i + j);
                let b = *matrix.get_unchecked(i + j + delta);
                *matrix.get_unchecked_mut(i + j) = (a & mask_val) | ((b & mask_val) << delta);
                *matrix.get_unchecked_mut(i + j + delta) =
                    ((a >> delta) & mask_val) | (b & !mask_val);
                j += 1;
            }
            i += 2 * delta;
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn transpose_64x64_neon(_matrix: &mut [u64; 64]) {
    unreachable!()
}

#[inline]
fn transpose_64x64_dispatch(matrix: &mut [u64; 64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 feature detected at runtime
            unsafe { transpose_64x64_avx2(matrix) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on aarch64
        unsafe { transpose_64x64_neon(matrix) };
        return;
    }

    #[allow(unreachable_code)]
    transpose_64x64(matrix);
}

fn transpose_batch(
    buf: &mut [u64],
    data: &[u64],
    batch_start: usize,
    batch_len: usize,
    m: usize,
    m_words: usize,
    s_words: usize,
) {
    buf[..batch_len * m_words].fill(0);
    let sw_base = batch_start / 64;
    let mut matrix = [0u64; 64];

    for mw in 0..m_words {
        let group_start = mw * 64;
        let group_end = (group_start + 64).min(m);
        let group_len = group_end - group_start;

        for j in 0..group_len {
            matrix[j] = data[(group_start + j) * s_words + sw_base];
        }
        matrix[group_len..].fill(0);

        transpose_64x64_dispatch(&mut matrix);

        for s in 0..batch_len {
            buf[s * m_words + mw] = matrix[s];
        }
    }
}

fn accumulate_meas_major_typed<const M: usize>(
    counts: &mut FxHashMap<[u64; M], u64>,
    batch_buf: &mut Vec<u64>,
    data: &[u64],
    n: usize,
    m: usize,
    s_words: usize,
) {
    let batch_size = 64;
    let buf_size = batch_size * M;
    if batch_buf.len() < buf_size {
        batch_buf.resize(buf_size, 0);
    }

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let batch_len = batch_end - batch_start;
        transpose_batch(batch_buf, data, batch_start, batch_len, m, M, s_words);

        for s in 0..batch_len {
            let base = s * M;
            let mut key = [0u64; M];
            key.copy_from_slice(&batch_buf[base..base + M]);
            count_shot_typed(counts, key);
        }
    }
}

fn accumulate_meas_major_wide(
    counts: &mut FxHashMap<Vec<u64>, u64>,
    batch_buf: &mut Vec<u64>,
    data: &[u64],
    n: usize,
    m: usize,
    m_words: usize,
    s_words: usize,
) {
    let batch_size = 64;
    let buf_size = batch_size * m_words;
    if batch_buf.len() < buf_size {
        batch_buf.resize(buf_size, 0);
    }

    for batch_start in (0..n).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(n);
        let batch_len = batch_end - batch_start;
        transpose_batch(batch_buf, data, batch_start, batch_len, m, m_words, s_words);

        for s in 0..batch_len {
            let base = s * m_words;
            let key = &batch_buf[base..base + m_words];
            count_shot_wide(counts, key);
        }
    }
}

#[cfg(feature = "parallel")]
fn count_shard_typed<const M: usize>(
    data: &[u64],
    shard_start: usize,
    shard_end: usize,
    m: usize,
    s_words: usize,
) -> FxHashMap<[u64; M], u64> {
    let shard_len = shard_end - shard_start;
    let mut local: FxHashMap<[u64; M], u64> = FxHashMap::default();
    local.reserve(shard_len.min(65536));

    let batch_size = 64;
    let mut buf = vec![0u64; batch_size * M];

    for batch_start in (shard_start..shard_end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shard_end);
        let batch_len = batch_end - batch_start;
        transpose_batch(&mut buf, data, batch_start, batch_len, m, M, s_words);

        for s in 0..batch_len {
            let base = s * M;
            let mut key = [0u64; M];
            key.copy_from_slice(&buf[base..base + M]);
            count_shot_typed(&mut local, key);
        }
    }

    local
}

#[cfg(feature = "parallel")]
fn count_shard_shot_major_typed<const M: usize>(
    data: &[u64],
    shard_start: usize,
    shard_end: usize,
) -> FxHashMap<[u64; M], u64> {
    let shard_len = shard_end - shard_start;
    let mut local: FxHashMap<[u64; M], u64> = FxHashMap::default();
    local.reserve(shard_len.min(65536));

    for s in shard_start..shard_end {
        let base = s * M;
        let mut key = [0u64; M];
        key.copy_from_slice(&data[base..base + M]);
        count_shot_typed(&mut local, key);
    }

    local
}

#[cfg(feature = "parallel")]
fn count_shard_shot_major_wide(
    data: &[u64],
    shard_start: usize,
    shard_end: usize,
    m_words: usize,
) -> FxHashMap<Vec<u64>, u64> {
    let shard_len = shard_end - shard_start;
    let mut local = FxHashMap::default();
    local.reserve(shard_len.min(65536));

    for s in shard_start..shard_end {
        let base = s * m_words;
        let key = &data[base..base + m_words];
        count_shot_wide(&mut local, key);
    }

    local
}

#[cfg(feature = "parallel")]
fn count_shard_wide(
    data: &[u64],
    shard_start: usize,
    shard_end: usize,
    m: usize,
    m_words: usize,
    s_words: usize,
) -> FxHashMap<Vec<u64>, u64> {
    let shard_len = shard_end - shard_start;
    let mut local = FxHashMap::default();
    local.reserve(shard_len.min(65536));

    let batch_size = 64;
    let mut buf = vec![0u64; batch_size * m_words];

    for batch_start in (shard_start..shard_end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shard_end);
        let batch_len = batch_end - batch_start;
        transpose_batch(&mut buf, data, batch_start, batch_len, m, m_words, s_words);

        for s in 0..batch_len {
            let base = s * m_words;
            let key = &buf[base..base + m_words];
            count_shot_wide(&mut local, key);
        }
    }

    local
}

#[cfg(feature = "parallel")]
fn merge_shard_typed<const M: usize>(
    dst: &mut FxHashMap<[u64; M], u64>,
    src: FxHashMap<[u64; M], u64>,
) {
    for (key, count) in src {
        *dst.entry(key).or_insert(0) += count;
    }
}

#[cfg(feature = "parallel")]
fn merge_shard_wide(dst: &mut FxHashMap<Vec<u64>, u64>, src: FxHashMap<Vec<u64>, u64>) {
    for (key, count) in src {
        *dst.entry(key).or_insert(0) += count;
    }
}

macro_rules! accumulate_typed {
    ($self:expr, $chunk:expr, $n:expr, $m:expr, $s_words:expr, $data:expr, $M:expr, $counts:expr) => {{
        #[cfg(feature = "parallel")]
        {
            if $n >= MIN_SHOTS_FOR_PAR_HISTOGRAM {
                use rayon::prelude::*;

                let num_threads = rayon::current_num_threads();
                let words_per_thread = $s_words.div_ceil(num_threads).max(1);
                let shots_per_thread = words_per_thread * 64;

                let thread_maps: Vec<FxHashMap<[u64; $M], u64>> = (0..num_threads)
                    .into_par_iter()
                    .filter_map(|tid| {
                        let shard_start = tid * shots_per_thread;
                        if shard_start >= $n {
                            return None;
                        }
                        let shard_end = (shard_start + shots_per_thread).min($n);
                        Some(count_shard_typed::<$M>(
                            $data,
                            shard_start,
                            shard_end,
                            $m,
                            $s_words,
                        ))
                    })
                    .collect();

                for shard in thread_maps {
                    merge_shard_typed($counts, shard);
                }
                return;
            }
        }

        accumulate_meas_major_typed($counts, &mut $self.batch_buf, $data, $n, $m, $s_words);
    }};
}

impl ShotAccumulator for HistogramAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        let n = chunk.num_shots();
        let m = chunk.num_measurements();
        let m_words = chunk.m_words();

        if matches!(self.counts, InlineCounts::Uninit) {
            self.counts = InlineCounts::init(m_words, n.min(1 << 18));
        }

        match chunk.layout() {
            ShotLayout::MeasMajor => {
                let s_words = chunk.s_words();
                let data = chunk.raw_data();

                macro_rules! meas_major_dispatch {
                    ($($variant:ident, $M:expr);+ $(;)?) => {
                        match &mut self.counts {
                            $(InlineCounts::$variant(counts) => {
                                accumulate_typed!(self, chunk, n, m, s_words, data, $M, counts);
                            })+
                            InlineCounts::Wide(counts) => {
                                #[cfg(feature = "parallel")]
                                {
                                    if n >= MIN_SHOTS_FOR_PAR_HISTOGRAM {
                                        use rayon::prelude::*;

                                        let num_threads = rayon::current_num_threads();
                                        let words_per_thread = s_words.div_ceil(num_threads).max(1);
                                        let shots_per_thread = words_per_thread * 64;

                                        let thread_maps: Vec<FxHashMap<Vec<u64>, u64>> =
                                            (0..num_threads)
                                                .into_par_iter()
                                                .filter_map(|tid| {
                                                    let shard_start = tid * shots_per_thread;
                                                    if shard_start >= n {
                                                        return None;
                                                    }
                                                    let shard_end = (shard_start + shots_per_thread).min(n);
                                                    Some(count_shard_wide(
                                                        data, shard_start, shard_end, m, m_words, s_words,
                                                    ))
                                                })
                                                .collect();

                                        for shard in thread_maps {
                                            merge_shard_wide(counts, shard);
                                        }
                                        return;
                                    }
                                }

                                accumulate_meas_major_wide(
                                    counts,
                                    &mut self.batch_buf,
                                    data,
                                    n,
                                    m,
                                    m_words,
                                    s_words,
                                );
                            }
                            InlineCounts::Uninit => unreachable!(),
                        }
                    };
                }
                meas_major_dispatch!(
                    W1, 1; W2, 2; W3, 3; W4, 4;
                    W5, 5; W6, 6; W7, 7; W8, 8;
                );
            }
            ShotLayout::ShotMajor => {
                let data = chunk.raw_data();

                macro_rules! shot_major_dispatch {
                    ($($variant:ident, $M:expr);+ $(;)?) => {
                        match &mut self.counts {
                            $(InlineCounts::$variant(counts) => {
                                #[cfg(feature = "parallel")]
                                if n >= MIN_SHOTS_FOR_PAR_HISTOGRAM {
                                    use rayon::prelude::*;
                                    let num_threads = rayon::current_num_threads();
                                    let shots_per_thread = n.div_ceil(num_threads);

                                    let thread_maps: Vec<FxHashMap<[u64; $M], u64>> =
                                        (0..num_threads)
                                            .into_par_iter()
                                            .filter_map(|tid| {
                                                let start = tid * shots_per_thread;
                                                if start >= n { return None; }
                                                let end = (start + shots_per_thread).min(n);
                                                Some(count_shard_shot_major_typed::<$M>(
                                                    data, start, end,
                                                ))
                                            })
                                            .collect();

                                    for shard in thread_maps {
                                        merge_shard_typed(counts, shard);
                                    }
                                } else {
                                    for s in 0..n {
                                        let base = s * $M;
                                        let mut key = [0u64; $M];
                                        key.copy_from_slice(&data[base..base + $M]);
                                        count_shot_typed(counts, key);
                                    }
                                }

                                #[cfg(not(feature = "parallel"))]
                                for s in 0..n {
                                    let base = s * $M;
                                    let mut key = [0u64; $M];
                                    key.copy_from_slice(&data[base..base + $M]);
                                    count_shot_typed(counts, key);
                                }
                            })+
                            InlineCounts::Wide(counts) => {
                                #[cfg(feature = "parallel")]
                                if n >= MIN_SHOTS_FOR_PAR_HISTOGRAM {
                                    use rayon::prelude::*;
                                    let num_threads = rayon::current_num_threads();
                                    let shots_per_thread = n.div_ceil(num_threads);

                                    let thread_maps: Vec<FxHashMap<Vec<u64>, u64>> =
                                        (0..num_threads)
                                            .into_par_iter()
                                            .filter_map(|tid| {
                                                let start = tid * shots_per_thread;
                                                if start >= n { return None; }
                                                let end = (start + shots_per_thread).min(n);
                                                Some(count_shard_shot_major_wide(
                                                    data, start, end, m_words,
                                                ))
                                            })
                                            .collect();

                                    for shard in thread_maps {
                                        merge_shard_wide(counts, shard);
                                    }
                                } else {
                                    for s in 0..n {
                                        let key = chunk.shot_words(s);
                                        count_shot_wide(counts, key);
                                    }
                                }

                                #[cfg(not(feature = "parallel"))]
                                for s in 0..n {
                                    let key = chunk.shot_words(s);
                                    count_shot_wide(counts, key);
                                }
                            }
                            InlineCounts::Uninit => unreachable!(),
                        }
                    };
                }
                shot_major_dispatch!(
                    W1, 1; W2, 2; W3, 3; W4, 4;
                    W5, 5; W6, 6; W7, 7; W8, 8;
                );
            }
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
                let data = chunk.raw_data();
                let m_words = chunk.m_words();
                #[allow(clippy::needless_range_loop)]
                if m_words == 1 {
                    for mi in 0..m {
                        let shift = mi as u32;
                        let mut count = 0u64;
                        for s in 0..n {
                            count += (data[s] >> shift) & 1;
                        }
                        self.ones[mi] += count;
                    }
                } else {
                    for mi in 0..m {
                        let w = mi / 64;
                        let shift = (mi % 64) as u32;
                        let mut count = 0u64;
                        for s in 0..n {
                            count += (data[s * m_words + w] >> shift) & 1;
                        }
                        self.ones[mi] += count;
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

pub struct NullAccumulator {
    total_shots: u64,
}

impl Default for NullAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl NullAccumulator {
    pub fn new() -> Self {
        Self { total_shots: 0 }
    }

    pub fn total_shots(&self) -> u64 {
        self.total_shots
    }
}

impl ShotAccumulator for NullAccumulator {
    fn accumulate(&mut self, chunk: &PackedShots) {
        self.total_shots += chunk.num_shots() as u64;
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
