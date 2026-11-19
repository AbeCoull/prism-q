//! Bit-packed shot matrices in shot-major and measurement-major layout, with the padding,
//! transpose, and counting helpers the samplers and accumulators share.

use crate::error::{PrismError, Result};

use super::accumulator::transpose_64x64_dispatch;
use super::{ShotAccumulator, counts_from_chunks, xor_words};

/// Memory layout of a [`PackedShots`] bit matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShotLayout {
    /// Rows are shots; each row packs that shot's measurement bits.
    ShotMajor,
    /// Rows are measurements; each row packs one measurement's outcome
    /// across all shots.
    MeasMajor,
}

/// XOR `mask` into every shot of a shot-major packed buffer.
pub(crate) fn xor_mask_shot_major(
    accum: &mut [u64],
    num_shots: usize,
    m_words: usize,
    mask: &[u64],
) {
    #[cfg(feature = "parallel")]
    if num_shots >= 256 {
        use rayon::prelude::*;
        accum[..num_shots * m_words]
            .par_chunks_mut(m_words)
            .with_min_len(crate::backend::chunk_min_len(m_words))
            .for_each(|shot| xor_words(shot, mask));
        return;
    }
    for s in 0..num_shots {
        let base = s * m_words;
        xor_words(&mut accum[base..base + m_words], mask);
    }
}

/// 64x64 blocks from which a measurement-major transpose runs on the pool.
#[cfg(feature = "parallel")]
const PARALLEL_TRANSPOSE_BLOCKS: usize = 512;

/// Set bits in a 64x64 block at or below which scattering them costs less than the
/// full transpose.
const SPARSE_TRANSPOSE_BITS: u32 = 64;

/// Transpose measurement-major `data` one 64-shot block at a time into `out`, which
/// arrives zeroed and holds `row_len` elements per shot. Within `block`, the slice of
/// `out` holding the block's shots, `set_bit(block, shot, measurement)` sets one bit and
/// `write_word(block, shot, meas_word, word)` stores the 64 measurements from
/// `meas_word * 64` of shot `shot` whole, over any bits set before. Shot padding past
/// `num_shots` is never written.
fn transpose_meas_major_blocks<T: Send>(
    data: &[u64],
    num_shots: usize,
    num_measurements: usize,
    out: &mut [T],
    row_len: usize,
    set_bit: impl Fn(&mut [T], usize, usize) + Sync,
    write_word: impl Fn(&mut [T], usize, usize, u64) + Sync,
) {
    if num_shots == 0 || row_len == 0 {
        return;
    }
    let s_words = num_shots.div_ceil(64);
    let m_words = num_measurements.div_ceil(64);
    let transpose_block = |(shot_word, block): (usize, &mut [T])| {
        let shots = block.len() / row_len;
        let shot_mask = shot_tail_mask(shots);
        let mut matrix = [0u64; 64];
        for meas_word in 0..m_words {
            let first = meas_word * 64;
            let rows = (num_measurements - first).min(64);
            // Scatter bit by bit while the block stays sparse. Past the budget the full
            // transpose takes over and rewrites every word the scatter touched.
            let mut budget = SPARSE_TRANSPOSE_BITS;
            let dense = (0..rows).any(|row| {
                let mut bits = data[(first + row) * s_words + shot_word] & shot_mask;
                while bits != 0 {
                    if budget == 0 {
                        return true;
                    }
                    set_bit(block, bits.trailing_zeros() as usize, first + row);
                    budget -= 1;
                    bits &= bits - 1;
                }
                false
            });
            if !dense {
                continue;
            }
            for (row, slot) in matrix[..rows].iter_mut().enumerate() {
                *slot = data[(first + row) * s_words + shot_word] & shot_mask;
            }
            matrix[rows..].fill(0);
            transpose_64x64_dispatch(&mut matrix);
            for (shot, &word) in matrix[..shots].iter().enumerate() {
                write_word(block, shot, meas_word, word);
            }
        }
    };
    #[cfg(feature = "parallel")]
    if s_words * m_words >= PARALLEL_TRANSPOSE_BLOCKS {
        use rayon::prelude::*;
        out.par_chunks_mut(64 * row_len)
            .enumerate()
            .for_each(transpose_block);
        return;
    }
    out.chunks_mut(64 * row_len)
        .enumerate()
        .for_each(transpose_block);
}

#[inline]
pub(crate) fn measurement_tail_mask(num_measurements: usize) -> u64 {
    let tail = num_measurements % 64;
    if tail == 0 {
        u64::MAX
    } else {
        (1u64 << tail) - 1
    }
}

#[inline]
pub(crate) fn shot_tail_mask(num_shots: usize) -> u64 {
    let tail = num_shots % 64;
    if tail == 0 {
        u64::MAX
    } else {
        (1u64 << tail) - 1
    }
}

fn checked_packed_len(label: &str, rows: usize, words_per_row: usize) -> Result<usize> {
    rows.checked_mul(words_per_row)
        .ok_or_else(|| PrismError::InvalidParameter {
            message: format!("{label} packed raw shape overflows usize"),
        })
}

fn validate_packed_len(
    label: &str,
    actual: usize,
    rows: usize,
    words_per_row: usize,
) -> Result<usize> {
    let expected = checked_packed_len(label, rows, words_per_row)?;
    if actual != expected {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "{label} packed raw length {actual} does not match expected length {expected}"
            ),
        });
    }
    Ok(expected)
}

fn validate_shot_major_padding(
    data: &[u64],
    num_shots: usize,
    num_measurements: usize,
    m_words: usize,
) -> Result<()> {
    let valid_mask = measurement_tail_mask(num_measurements);
    if let Some(shot) = first_shot_major_padding_violation(data, num_shots, m_words, valid_mask) {
        return Err(PrismError::InvalidParameter {
            message: format!("shot-major raw padding bits are set in shot {shot}"),
        });
    }
    Ok(())
}

fn first_shot_major_padding_violation(
    data: &[u64],
    num_shots: usize,
    m_words: usize,
    tail_mask: u64,
) -> Option<usize> {
    if m_words == 0 || tail_mask == u64::MAX {
        return None;
    }
    let invalid_mask = !tail_mask;
    (0..num_shots).find(|&shot| data[shot * m_words + m_words - 1] & invalid_mask != 0)
}

pub(crate) fn shot_major_padding_bits_set(
    data: &[u64],
    num_shots: usize,
    m_words: usize,
    tail_mask: u64,
) -> bool {
    first_shot_major_padding_violation(data, num_shots, m_words, tail_mask).is_some()
}

fn validate_meas_major_padding(
    data: &[u64],
    num_shots: usize,
    num_measurements: usize,
    s_words: usize,
) -> Result<()> {
    if s_words == 0 {
        return Ok(());
    }
    let valid_mask = shot_tail_mask(num_shots);
    if valid_mask == u64::MAX {
        return Ok(());
    }
    let invalid_mask = !valid_mask;
    for measurement in 0..num_measurements {
        let idx = measurement * s_words + s_words - 1;
        if data[idx] & invalid_mask != 0 {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "measurement-major raw padding bits are set in measurement {measurement}"
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn clear_meas_major_shot_padding(
    data: &mut [u64],
    num_shots: usize,
    num_measurements: usize,
    s_words: usize,
) {
    if s_words == 0 {
        return;
    }
    let valid_mask = shot_tail_mask(num_shots);
    if valid_mask == u64::MAX {
        return;
    }
    for measurement in 0..num_measurements {
        data[measurement * s_words + s_words - 1] &= valid_mask;
    }
}

#[inline]
pub(crate) fn count_vec_key<S>(
    counts: &mut std::collections::HashMap<Vec<u64>, u64, S>,
    key: &[u64],
) where
    S: std::hash::BuildHasher,
{
    if let Some(count) = counts.get_mut(key) {
        *count += 1;
    } else {
        counts.insert(key.to_vec(), 1);
    }
}

#[inline]
pub(crate) fn count_vec_key_masked<S>(
    counts: &mut std::collections::HashMap<Vec<u64>, u64, S>,
    key: &[u64],
    tail_mask: u64,
    scratch: &mut Vec<u64>,
) where
    S: std::hash::BuildHasher,
{
    if key.is_empty() || tail_mask == u64::MAX || (key[key.len() - 1] & !tail_mask) == 0 {
        count_vec_key(counts, key);
        return;
    }

    scratch.clear();
    scratch.extend_from_slice(key);
    let last = scratch.len() - 1;
    scratch[last] &= tail_mask;
    if let Some(count) = counts.get_mut(scratch.as_slice()) {
        *count += 1;
    } else {
        counts.insert(scratch.clone(), 1);
    }
}

/// Bit-packed measurement outcomes for a batch of shots.
///
/// Stores a `num_shots` x `num_measurements` bit matrix in `u64` words with
/// little-endian bit order inside each word. Shot-major layout stores
/// `num_shots` rows of `m_words = ceil(num_measurements / 64)` words, so bit
/// `m` of shot `s` lives at word `s * m_words + m / 64`, bit `m % 64`.
/// Measurement-major layout stores `num_measurements` rows of `s_words =
/// ceil(num_shots / 64)` words, so the same bit lives at word
/// `m * s_words + s / 64`, bit `s % 64`. Samplers emit whichever layout they
/// can write contiguously; consumers branch on [`Self::layout`], and the
/// layout-specific row accessors panic when called in the other layout. Bits
/// past the last valid index in a row's final word are padding; the validated
/// constructors reject data with padding bits set.
#[derive(Debug, Clone)]
pub struct PackedShots {
    data: Vec<u64>,
    pub(super) num_shots: usize,
    num_measurements: usize,
    m_words: usize,
    s_words: usize,
    layout: ShotLayout,
}

impl PackedShots {
    /// Version tag for the raw word layout returned by [`Self::raw_data`].
    pub const RAW_FORMAT_VERSION: u32 = 1;

    pub fn raw_format_version(&self) -> u32 {
        Self::RAW_FORMAT_VERSION
    }

    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn layout(&self) -> ShotLayout {
        self.layout
    }

    #[inline(always)]
    pub fn get_bit(&self, shot: usize, measurement: usize) -> bool {
        match self.layout {
            ShotLayout::ShotMajor => {
                let base = shot * self.m_words;
                let w = measurement / 64;
                (self.data[base + w] >> (measurement % 64)) & 1 != 0
            }
            ShotLayout::MeasMajor => {
                let base = measurement * self.s_words;
                let w = shot / 64;
                (self.data[base + w] >> (shot % 64)) & 1 != 0
            }
        }
    }

    /// Words per measurement-major row: `ceil(num_shots / 64)`.
    pub fn s_words(&self) -> usize {
        self.s_words
    }

    /// Words per shot-major row: `ceil(num_measurements / 64)`.
    pub fn m_words(&self) -> usize {
        self.m_words
    }

    pub(crate) fn measurement_tail_mask(&self) -> u64 {
        measurement_tail_mask(self.num_measurements)
    }

    /// Row of packed measurement bits for one shot (shot-major only).
    pub fn shot_words(&self, shot: usize) -> &[u64] {
        assert!(
            self.layout == ShotLayout::ShotMajor,
            "shot_words requires ShotMajor layout"
        );
        let base = shot * self.m_words;
        &self.data[base..base + self.m_words]
    }

    /// Row of packed per-shot bits for one measurement (measurement-major
    /// only).
    pub fn meas_words(&self, m: usize) -> &[u64] {
        assert!(
            self.layout == ShotLayout::MeasMajor,
            "meas_words requires MeasMajor layout"
        );
        let base = m * self.s_words;
        &self.data[base..base + self.s_words]
    }

    /// Build from raw shot-major words, rejecting a wrong data length or set
    /// padding bits.
    pub fn try_from_shot_major(
        data: Vec<u64>,
        num_shots: usize,
        num_measurements: usize,
    ) -> Result<Self> {
        let m_words = num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        validate_packed_len("shot-major", data.len(), num_shots, m_words)?;
        validate_shot_major_padding(&data, num_shots, num_measurements, m_words)?;
        Ok(Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::ShotMajor,
        })
    }

    /// Like [`Self::try_from_shot_major`] but panics on invalid data.
    pub fn from_shot_major(data: Vec<u64>, num_shots: usize, num_measurements: usize) -> Self {
        Self::try_from_shot_major(data, num_shots, num_measurements)
            .unwrap_or_else(|e| panic!("invalid shot-major PackedShots raw data: {e}"))
    }

    /// Build from raw measurement-major words, rejecting a wrong data length
    /// or set padding bits.
    pub fn try_from_meas_major(
        data: Vec<u64>,
        num_shots: usize,
        num_measurements: usize,
    ) -> Result<Self> {
        let m_words = num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        validate_packed_len("measurement-major", data.len(), num_measurements, s_words)?;
        validate_meas_major_padding(&data, num_shots, num_measurements, s_words)?;
        Ok(Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        })
    }

    /// [`Self::from_meas_major`] for rows built with clear padding, without the scan of
    /// every row's last word, which faults in each page of rows nothing has touched.
    pub(crate) fn from_meas_major_clear_padding(
        data: Vec<u64>,
        num_shots: usize,
        num_measurements: usize,
    ) -> Self {
        Self::from_parts(data, num_shots, num_measurements, ShotLayout::MeasMajor)
    }

    /// Wrap words a sampler wrote in `layout` with clear padding. The length and padding
    /// checks of the validated constructors run in debug builds only.
    pub(super) fn from_parts(
        data: Vec<u64>,
        num_shots: usize,
        num_measurements: usize,
        layout: ShotLayout,
    ) -> Self {
        let m_words = num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        debug_assert!(match layout {
            ShotLayout::ShotMajor => {
                validate_packed_len("shot-major", data.len(), num_shots, m_words).is_ok()
                    && validate_shot_major_padding(&data, num_shots, num_measurements, m_words)
                        .is_ok()
            }
            ShotLayout::MeasMajor => {
                validate_packed_len("measurement-major", data.len(), num_measurements, s_words)
                    .is_ok()
                    && validate_meas_major_padding(&data, num_shots, num_measurements, s_words)
                        .is_ok()
            }
        });
        Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout,
        }
    }

    /// Like [`Self::try_from_meas_major`] but panics on invalid data.
    pub fn from_meas_major(data: Vec<u64>, num_shots: usize, num_measurements: usize) -> Self {
        Self::try_from_meas_major(data, num_shots, num_measurements)
            .unwrap_or_else(|e| panic!("invalid measurement-major PackedShots raw data: {e}"))
    }

    /// Return the raw words for [`PackedShots::RAW_FORMAT_VERSION`].
    ///
    /// Version 1 stores little-endian bit order within each `u64`. Shot-major
    /// layout stores `num_shots` rows of `m_words()` words. Measurement-major
    /// layout stores `num_measurements` rows of `s_words()` words. Semantic
    /// padding bits are outside `num_measurements` for shot-major data and
    /// outside `num_shots` for measurement-major data.
    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u64> {
        self.data
    }

    pub(crate) fn into_shot_major_data(self) -> Vec<u64> {
        if self.layout == ShotLayout::ShotMajor {
            return self.data;
        }
        self.meas_major_to_shot_major_words()
    }

    /// Append the shot-major words of this matrix to `out`, transposing in place.
    pub(crate) fn append_shot_major_to(&self, out: &mut Vec<u64>) {
        if self.layout == ShotLayout::ShotMajor {
            out.extend_from_slice(&self.data);
            return;
        }
        let start = out.len();
        out.resize(start + self.num_shots * self.m_words, 0);
        self.transpose_into_shot_major(&mut out[start..]);
    }

    fn meas_major_to_shot_major_words(&self) -> Vec<u64> {
        let mut shot_major = vec![0u64; self.num_shots * self.m_words];
        self.transpose_into_shot_major(&mut shot_major);
        shot_major
    }

    /// Write the shot-major words of this measurement-major matrix into zeroed `out`.
    fn transpose_into_shot_major(&self, out: &mut [u64]) {
        debug_assert_eq!(self.layout, ShotLayout::MeasMajor);
        let m_words = self.m_words;
        transpose_meas_major_blocks(
            &self.data,
            self.num_shots,
            self.num_measurements,
            out,
            m_words,
            |block, shot, measurement| {
                block[shot * m_words + measurement / 64] |= 1 << (measurement % 64);
            },
            |block, shot, meas_word, word| block[shot * m_words + meas_word] = word,
        );
    }

    /// Copy of this matrix in [`ShotLayout::ShotMajor`].
    pub fn to_shot_major(&self) -> PackedShots {
        if self.layout == ShotLayout::ShotMajor {
            return self.clone();
        }
        PackedShots {
            data: self.meas_major_to_shot_major_words(),
            layout: ShotLayout::ShotMajor,
            ..*self
        }
    }

    /// Copy of this matrix in [`ShotLayout::MeasMajor`].
    pub fn to_meas_major(&self) -> PackedShots {
        if self.layout == ShotLayout::MeasMajor {
            return self.clone();
        }
        PackedShots {
            data: self.clone().into_meas_major_data(),
            layout: ShotLayout::MeasMajor,
            ..*self
        }
    }

    /// Shot rows of `ceil(num_measurements / 8)` bytes each: measurement `j` of a shot is
    /// bit `j % 8` of byte `j / 8`, and the padding bits of the last byte are clear.
    pub fn to_shot_major_bytes(&self) -> Vec<u8> {
        let row_bytes = self.num_measurements.div_ceil(8);
        let mut bytes = vec![0u8; self.num_shots * row_bytes];
        if row_bytes == 0 {
            return bytes;
        }
        match self.layout {
            ShotLayout::ShotMajor => {
                for (shot, row) in bytes.chunks_exact_mut(row_bytes).enumerate() {
                    for (&word, out) in self.shot_words(shot).iter().zip(row.chunks_mut(8)) {
                        out.copy_from_slice(&word.to_le_bytes()[..out.len()]);
                    }
                }
            }
            ShotLayout::MeasMajor => transpose_meas_major_blocks(
                &self.data,
                self.num_shots,
                self.num_measurements,
                &mut bytes,
                row_bytes,
                |block, shot, measurement| {
                    block[shot * row_bytes + measurement / 8] |= 1 << (measurement % 8);
                },
                |block, shot, meas_word, word| {
                    let start = shot * row_bytes + meas_word * 8;
                    let take = (row_bytes - meas_word * 8).min(8);
                    block[start..start + take].copy_from_slice(&word.to_le_bytes()[..take]);
                },
            ),
        }
        bytes
    }

    /// XOR `columns`, packed one bit per measurement, into every shot.
    pub(crate) fn xor_columns(&mut self, columns: &[u64]) {
        debug_assert_eq!(columns.len(), self.m_words);
        if self.num_shots == 0 || columns.iter().all(|&word| word == 0) {
            return;
        }
        match self.layout {
            ShotLayout::ShotMajor => {
                xor_mask_shot_major(&mut self.data, self.num_shots, self.m_words, columns)
            }
            ShotLayout::MeasMajor => {
                let tail = shot_tail_mask(self.num_shots);
                for (measurement, row) in self.data.chunks_exact_mut(self.s_words).enumerate() {
                    if (columns[measurement / 64] >> (measurement % 64)) & 1 == 1 {
                        for word in row.iter_mut() {
                            *word = !*word;
                        }
                        row[self.s_words - 1] &= tail;
                    }
                }
            }
        }
    }

    /// Measurement-major words: `num_measurements` rows of `s_words` each.
    pub(crate) fn into_meas_major_data(self) -> Vec<u64> {
        if self.layout == ShotLayout::MeasMajor {
            return self.data;
        }

        let mut meas_major = vec![0u64; self.num_measurements * self.s_words];
        let mut block = [0u64; 64];
        for shot_word in 0..self.s_words {
            let first_shot = shot_word * 64;
            let shots = (self.num_shots - first_shot).min(64);
            for meas_word in 0..self.m_words {
                block.fill(0);
                for (shot, word) in block[..shots].iter_mut().enumerate() {
                    *word = self.data[(first_shot + shot) * self.m_words + meas_word];
                }
                transpose_64x64_dispatch(&mut block);
                let first_measurement = meas_word * 64;
                let measurements = (self.num_measurements - first_measurement).min(64);
                for (measurement, &word) in block[..measurements].iter().enumerate() {
                    meas_major[(first_measurement + measurement) * self.s_words + shot_word] = word;
                }
            }
        }
        meas_major
    }

    pub fn to_shots(&self) -> Vec<Vec<bool>> {
        let mut shots = vec![vec![false; self.num_measurements]; self.num_shots];
        match self.layout {
            ShotLayout::ShotMajor => {
                let last_mask = self.measurement_tail_mask();
                for (s, shot) in shots.iter_mut().enumerate() {
                    let row = &self.data[s * self.m_words..(s + 1) * self.m_words];
                    for (mw, mut bits) in row.iter().copied().enumerate() {
                        let base = mw * 64;
                        if mw + 1 == self.m_words {
                            bits &= last_mask;
                        }
                        while bits != 0 {
                            let measurement = base + bits.trailing_zeros() as usize;
                            shot[measurement] = true;
                            bits &= bits - 1;
                        }
                    }
                }
            }
            ShotLayout::MeasMajor => {
                let last_mask = shot_tail_mask(self.num_shots);
                let mut measurement = 0;
                while measurement < self.num_measurements {
                    let row =
                        &self.data[measurement * self.s_words..(measurement + 1) * self.s_words];
                    for (sw, mut bits) in row.iter().copied().enumerate() {
                        if sw + 1 == self.s_words {
                            bits &= last_mask;
                        }
                        while bits != 0 {
                            let shot = sw * 64 + bits.trailing_zeros() as usize;
                            shots[shot][measurement] = true;
                            bits &= bits - 1;
                        }
                    }
                    measurement += 1;
                }
            }
        }
        shots
    }

    /// Compute packed parity rows over this packed shot matrix.
    ///
    /// Each row lists measurement indices to XOR into one output bit.
    pub fn parity_rows(&self, rows: &[Vec<usize>]) -> Result<PackedShots> {
        validate_parity_rows(rows, self.num_measurements)?;

        match self.layout {
            ShotLayout::MeasMajor => {
                let mut data = vec![0u64; rows.len() * self.s_words];

                #[cfg(feature = "parallel")]
                if rows.len() * self.s_words >= 16_384 {
                    use rayon::prelude::*;
                    data.par_chunks_mut(self.s_words)
                        .zip(rows.par_iter())
                        .for_each(|(dst, row)| {
                            for &measurement in row {
                                let src_start = measurement * self.s_words;
                                xor_words(dst, &self.data[src_start..src_start + self.s_words]);
                            }
                        });
                    clear_meas_major_shot_padding(
                        &mut data,
                        self.num_shots,
                        rows.len(),
                        self.s_words,
                    );
                    return Ok(PackedShots::from_meas_major(
                        data,
                        self.num_shots,
                        rows.len(),
                    ));
                }

                for (out_idx, row) in rows.iter().enumerate() {
                    let dst = &mut data[out_idx * self.s_words..(out_idx + 1) * self.s_words];
                    for &measurement in row {
                        let src_start = measurement * self.s_words;
                        xor_words(dst, &self.data[src_start..src_start + self.s_words]);
                    }
                }
                clear_meas_major_shot_padding(&mut data, self.num_shots, rows.len(), self.s_words);
                Ok(PackedShots::from_meas_major(
                    data,
                    self.num_shots,
                    rows.len(),
                ))
            }
            ShotLayout::ShotMajor => {
                let out_words = rows.len().div_ceil(64);
                let mut data = vec![0u64; self.num_shots * out_words];
                for shot in 0..self.num_shots {
                    let base = shot * out_words;
                    for (out_idx, row) in rows.iter().enumerate() {
                        let mut bit = false;
                        for &measurement in row {
                            bit ^= self.get_bit(shot, measurement);
                        }
                        if bit {
                            data[base + out_idx / 64] |= 1u64 << (out_idx % 64);
                        }
                    }
                }
                Ok(PackedShots::from_shot_major(
                    data,
                    self.num_shots,
                    rows.len(),
                ))
            }
        }
    }

    /// Histogram of outcomes, keyed by `m_words()` packed words per shot.
    pub fn counts(&self) -> std::collections::HashMap<Vec<u64>, u64> {
        if self.layout == ShotLayout::ShotMajor && self.m_words > 8 {
            return self.counts_shot_major_wide();
        }
        counts_from_chunks(|acc| acc.accumulate(self))
    }

    /// Shot-major keys wider than the accumulator's inline arms go straight into the
    /// returned map; counting into an Fx map and converting it read slower at 1000
    /// measurements.
    fn counts_shot_major_wide(&self) -> std::collections::HashMap<Vec<u64>, u64> {
        let mut map = std::collections::HashMap::new();
        let mw = self.m_words;
        let tail_mask = self.measurement_tail_mask();
        if shot_major_padding_bits_set(&self.data, self.num_shots, mw, tail_mask) {
            let mut scratch = Vec::new();
            for shot in self.data.chunks_exact(mw).take(self.num_shots) {
                count_vec_key_masked(&mut map, shot, tail_mask, &mut scratch);
            }
        } else {
            for shot in self.data.chunks_exact(mw).take(self.num_shots) {
                count_vec_key(&mut map, shot);
            }
        }
        map
    }
}

pub(super) fn validate_parity_rows(rows: &[Vec<usize>], num_measurements: usize) -> Result<()> {
    for row in rows {
        for &measurement in row {
            if measurement >= num_measurements {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "measurement index {measurement} out of bounds for {num_measurements} measurements"
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "packed_tests.rs"]
mod tests;
