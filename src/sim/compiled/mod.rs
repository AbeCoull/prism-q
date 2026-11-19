//! Compiled Clifford measurement sampling: each measurement outcome is a
//! fixed reference bit XOR a parity over `rank` independent random bits, so
//! drawing shots is GF(2) linear algebra instead of circuit re-execution.

mod accumulator;
mod bts;
mod defer;
#[cfg(feature = "gpu")]
mod device;
mod forward;
mod live_forward;
mod packed;
pub(crate) mod parity;
mod propagation;
pub(crate) mod rng;
#[cfg(test)]
mod sign_truth_table_tests;
#[cfg(test)]
mod tests;

use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
#[cfg(feature = "gpu")]
use crate::gpu::kernels::bts::GpuBtsCache;
use crate::sim::ShotsResult;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use bts::{BTS_BATCH_SHOTS, bts_batched, bts_single_pass, sample_bts_meas_major};
use rng::{Xoshiro256PlusPlus, binomial_sample};

pub use accumulator::{
    CorrelatorAccumulator, HistogramAccumulator, MarginalsAccumulator, NullAccumulator,
    PauliExpectationAccumulator, ShotAccumulator, default_chunk_size, optimal_chunk_size,
};
pub(crate) use accumulator::{counts_from_chunks, for_each_chunk, marginals_from_chunks};
#[cfg(feature = "gpu")]
pub use device::DevicePackedShots;
pub use forward::compile_forward;
use forward::compile_forward_unchecked;
use packed::validate_parity_rows;
pub use packed::{PackedShots, ShotLayout};
pub(crate) use packed::{
    clear_meas_major_shot_padding, count_vec_key, count_vec_key_masked,
    shot_major_padding_bits_set, shot_tail_mask, xor_mask_shot_major,
};
pub use parity::ParityStats;
pub(crate) use parity::{ParityBlock, ParityBlocks, SparseParity};
use parity::{build_parity_blocks_if_useful, minimize_flip_row_weight};

pub(crate) use crate::backend::word_ops::xor_words;
pub(crate) use defer::{
    DeferredCircuit, Guard, GuardedEvent, GuardedKind, defer_clifford_t_circuit,
    defer_guarded_circuit, defer_measure_reset_circuit, defer_noisy_circuit,
};
pub(crate) use propagation::{
    batch_propagate_backward, batch_propagate_backward_flat, pair_rows_mut,
};
use propagation::{build_measurement_rows, compute_reference_bits};
pub(crate) use propagation::{propagate_backward, propagate_backward_words};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PauliVec {
    pub(crate) x: Vec<u64>,
    pub(crate) z: Vec<u64>,
}

impl PauliVec {
    pub fn new(num_words: usize) -> Self {
        Self {
            x: vec![0u64; num_words],
            z: vec![0u64; num_words],
        }
    }

    pub fn z_on_qubit(num_words: usize, qubit: usize) -> Self {
        let mut pv = Self::new(num_words);
        pv.z[qubit / 64] |= 1u64 << (qubit % 64);
        pv
    }

    #[inline(always)]
    pub fn is_diagonal(&self) -> bool {
        self.x.iter().all(|&w| w == 0)
    }

    #[inline(always)]
    pub fn has_x_or_y(&self, qubit: usize) -> bool {
        get_bit(&self.x, qubit)
    }
}

impl Hash for PauliVec {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.z.hash(state);
    }
}

#[inline(always)]
pub(crate) fn get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
pub(crate) fn set_bit(words: &mut [u64], qubit: usize, val: bool) {
    let word = qubit / 64;
    let bit = qubit % 64;
    if val {
        words[word] |= 1u64 << bit;
    } else {
        words[word] &= !(1u64 << bit);
    }
}

#[inline(always)]
pub(crate) fn flip_bit(words: &mut [u64], qubit: usize) {
    words[qubit / 64] ^= 1u64 << (qubit % 64);
}

fn gaussian_eliminate(rows: &mut [Vec<u64>], num_cols: usize) -> (usize, Vec<usize>) {
    let num_rows = rows.len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut current_row = 0;

    for col in 0..num_cols {
        let word = col / 64;
        let bit = col % 64;
        let mask = 1u64 << bit;

        let pivot = rows[current_row..num_rows]
            .iter()
            .position(|row| row[word] & mask != 0)
            .map(|i| i + current_row);

        let pivot_row = match pivot {
            Some(r) => r,
            None => continue,
        };

        if pivot_row != current_row {
            rows.swap(pivot_row, current_row);
        }

        let (top, rest) = rows.split_at_mut(current_row + 1);
        let pivot_data = &top[current_row];
        for row in rest.iter_mut() {
            if row[word] & mask != 0 {
                for w in 0..row.len() {
                    row[w] ^= pivot_data[w];
                }
            }
        }

        pivot_cols.push(col);
        current_row += 1;
    }

    (current_row, pivot_cols)
}

const LUT_GROUP_SIZE: usize = 8;
const LUT_MIN_RANK: usize = 8;

/// Largest flip table built. Past it, sampling XORs the flip rows each random byte selects
/// straight from the rows, drawing the same bytes, so the shots do not change.
const MAX_FLIP_LUT_BYTES: usize = 16 << 20;

/// Shot-major shots a sampler draws before it builds the flip table. Filling a group's
/// 256 entries costs about what 64 shots spend XORing its rows directly, and both draw the
/// same bytes, so the shots do not change.
const MIN_SHOTS_FOR_FLIP_LUT: usize = 64;

struct FlipLut {
    data: Vec<u64>,
    m_words: usize,
}

impl FlipLut {
    fn build(flip_rows: &[Vec<u64>], m_words: usize) -> Self {
        let rank = flip_rows.len();
        let num_full_groups = rank / LUT_GROUP_SIZE;
        let remainder_size = rank % LUT_GROUP_SIZE;
        let total_groups = num_full_groups + usize::from(remainder_size > 0);
        let entries_per_group = 1 << LUT_GROUP_SIZE;

        let mut data = vec![0u64; total_groups * entries_per_group * m_words];

        for g in 0..total_groups {
            let group_start = g * LUT_GROUP_SIZE;
            let k = if g < num_full_groups {
                LUT_GROUP_SIZE
            } else {
                remainder_size
            };
            let lut_offset = g * entries_per_group * m_words;

            for byte in 1..(1usize << k) {
                let lowest = byte & byte.wrapping_neg();
                let row_idx = group_start + lowest.trailing_zeros() as usize;
                let prev = byte ^ lowest;

                let dst_start = lut_offset + byte * m_words;
                let src_start = lut_offset + prev * m_words;

                for w in 0..m_words {
                    data[dst_start + w] = data[src_start + w] ^ flip_rows[row_idx][w];
                }
            }
        }

        Self { data, m_words }
    }

    fn table_bytes(rank: usize, m_words: usize) -> usize {
        rank.div_ceil(LUT_GROUP_SIZE) * (1 << LUT_GROUP_SIZE) * m_words * size_of::<u64>()
    }

    #[inline(always)]
    fn lookup(&self, group: usize, byte: usize) -> &[u64] {
        let offset = (group * (1 << LUT_GROUP_SIZE) + byte) * self.m_words;
        &self.data[offset..offset + self.m_words]
    }
}

/// The flip rows in groups of `LUT_GROUP_SIZE`, each group XORed in by one random byte:
/// through the table when it fits `MAX_FLIP_LUT_BYTES`, else row by row.
#[derive(Clone, Copy)]
struct FlipGroups<'a> {
    table: Option<&'a FlipLut>,
    rows: &'a [Vec<u64>],
    num_full_groups: usize,
    remainder_size: usize,
}

impl<'a> FlipGroups<'a> {
    /// Build the table on the first call that needs it.
    fn new(lut: &'a OnceLock<Option<FlipLut>>, rows: &'a [Vec<u64>], m_words: usize) -> Self {
        let table = lut.get_or_init(|| {
            (FlipLut::table_bytes(rows.len(), m_words) <= MAX_FLIP_LUT_BYTES)
                .then(|| FlipLut::build(rows, m_words))
        });
        Self {
            table: table.as_ref(),
            rows,
            num_full_groups: rows.len() / LUT_GROUP_SIZE,
            remainder_size: rows.len() % LUT_GROUP_SIZE,
        }
    }

    /// The table only if an earlier call built it, else the rows.
    fn existing(lut: &'a OnceLock<Option<FlipLut>>, rows: &'a [Vec<u64>]) -> Self {
        Self {
            table: lut.get().and_then(Option::as_ref),
            rows,
            num_full_groups: rows.len() / LUT_GROUP_SIZE,
            remainder_size: rows.len() % LUT_GROUP_SIZE,
        }
    }

    fn total_groups(&self) -> usize {
        self.num_full_groups + usize::from(self.remainder_size > 0)
    }

    #[inline(always)]
    fn xor_into(&self, dst: &mut [u64], group: usize, byte: usize) {
        match self.table {
            Some(table) => xor_words(dst, table.lookup(group, byte)),
            None => self.xor_rows(dst, group, byte),
        }
    }

    #[inline(always)]
    fn xor_rows(&self, dst: &mut [u64], group: usize, byte: usize) {
        let mut bits = byte;
        while bits != 0 {
            let row = group * LUT_GROUP_SIZE + bits.trailing_zeros() as usize;
            xor_words(dst, &self.rows[row]);
            bits &= bits - 1;
        }
    }
}

/// Draw one random byte per flip group of a single shot, a full `u64` for every eight
/// groups, and hand each group's byte to `xor`.
#[inline(always)]
fn draw_flip_groups(rng: &mut ChaCha8Rng, lut: &FlipGroups<'_>, mut xor: impl FnMut(usize, usize)) {
    let mut rand_buf = 0u64;
    let mut rand_pos = 8usize;
    for g in 0..lut.num_full_groups {
        if rand_pos >= 8 {
            rand_buf = rng.next_u64();
            rand_pos = 0;
        }
        let byte = (rand_buf & 0xFF) as usize;
        rand_buf >>= 8;
        rand_pos += 1;
        xor(g, byte);
    }
    if lut.remainder_size > 0 {
        if rand_pos >= 8 {
            rand_buf = rng.next_u64();
        }
        let mask = (1u64 << lut.remainder_size) - 1;
        xor(lut.num_full_groups, (rand_buf & mask) as usize);
    }
}

/// Sampler for a compiled Clifford measurement circuit, built by
/// [`compile_measurements`] or [`compile_forward`]. Shots XOR a random subset
/// of `rank` flip rows into fixed reference bits; deterministic per seed.
/// `try_` methods propagate GPU errors; their plain twins fall back to the
/// CPU path.
pub struct CompiledSampler {
    flip_rows: Vec<Vec<u64>>,
    ref_bits_packed: Vec<u64>,
    rank: usize,
    num_measurements: usize,
    rng: ChaCha8Rng,
    lut: OnceLock<Option<FlipLut>>,
    /// Shot-major shots drawn so far, toward [`MIN_SHOTS_FOR_FLIP_LUT`].
    shot_major_shots: usize,
    sparse: Option<SparseParity>,
    parity_blocks: Option<ParityBlocks>,
    #[cfg(feature = "gpu")]
    gpu_context: Option<std::sync::Arc<crate::gpu::GpuContext>>,
    #[cfg(feature = "gpu")]
    gpu_bts_cache: Option<GpuBtsCache>,
}

/// Compiled sampler for measurement, detector, and observable records.
pub struct CompiledDetectorSampler {
    measurement_sampler: CompiledSampler,
    detector_rows: Vec<Vec<usize>>,
    observable_rows: Vec<Vec<usize>>,
    num_measurements: usize,
}

/// Packed measurement, detector, and observable samples from one shot batch.
#[derive(Debug, Clone)]
pub struct DetectorSampleBatch {
    pub measurements: PackedShots,
    pub detectors: PackedShots,
    pub observables: PackedShots,
}

fn pack_bools(bools: &[bool]) -> Vec<u64> {
    let n_words = bools.len().div_ceil(64);
    let mut packed = vec![0u64; n_words];
    for (i, &b) in bools.iter().enumerate() {
        if b {
            packed[i / 64] |= 1u64 << (i % 64);
        }
    }
    packed
}

#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
pub(crate) struct SendPtrU64(pub(crate) *mut u64);
#[cfg(feature = "parallel")]
// SAFETY: SendPtrU64 is used only for packed shot buffers partitioned by
// disjoint word ranges before entering parallel workers.
unsafe impl Send for SendPtrU64 {}
#[cfg(feature = "parallel")]
// SAFETY: The raw pointer wrapper is shared, but each worker writes only its
// assigned non-overlapping range.
unsafe impl Sync for SendPtrU64 {}
#[cfg(feature = "parallel")]
impl SendPtrU64 {
    #[inline(always)]
    unsafe fn copy_from_slice(self, dst_offset: usize, src: &[u64]) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe { self.write_slice(dst_offset, src.as_ptr(), src.len()) }
    }

    #[inline(always)]
    pub(super) unsafe fn write_slice(self, offset: usize, src: *const u64, len: usize) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            std::ptr::copy_nonoverlapping(src, self.0.add(offset), len);
        }
    }

    /// XOR `bits` into the word at `offset`.
    ///
    /// # Safety
    ///
    /// `offset` must be in bounds, and no other thread may access that word concurrently.
    #[inline(always)]
    pub(crate) unsafe fn xor_word(self, offset: usize, bits: u64) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            *self.0.add(offset) ^= bits;
        }
    }

    /// Write `value` to the word at `offset` without reading it.
    ///
    /// # Safety
    ///
    /// `offset` must be in bounds, and no other thread may access that word concurrently.
    #[inline(always)]
    pub(crate) unsafe fn write_word(self, offset: usize, value: u64) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            self.0.add(offset).write(value);
        }
    }
}

#[inline(always)]
fn scatter_meas_major_rows(
    dst: &mut [u64],
    dst_s_words: usize,
    src: &[u64],
    src_s_words: usize,
    meas_indices: &[usize],
) {
    debug_assert_eq!(src.len(), meas_indices.len() * src_s_words);
    for (local_m, &global_m) in meas_indices.iter().enumerate() {
        let src_row = &src[local_m * src_s_words..(local_m + 1) * src_s_words];
        let dst_offset = global_m * dst_s_words;
        dst[dst_offset..dst_offset + src_s_words].copy_from_slice(src_row);
    }
}

/// Draw `num_shots` measurement-major shots of `block` from a stream seeded by `seed`.
#[inline]
fn sample_block(block: &ParityBlock, seed: u64, num_shots: usize) -> Vec<u64> {
    let mut block_chacha = ChaCha8Rng::seed_from_u64(seed);
    let mut block_rng = Xoshiro256PlusPlus::from_chacha(&mut block_chacha);
    sample_bts_meas_major(
        &block.sparse,
        num_shots,
        &block.ref_bits_packed,
        &mut block_rng,
        block.block_rank,
    )
}

#[inline(always)]
fn project_local_flip_row(local_row: &[u64], mapping: &[usize], m_words: usize) -> Vec<u64> {
    let mut global_row = vec![0u64; m_words];
    for (local_word, &word) in local_row.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            let local_m = local_word * 64 + bit;
            debug_assert!(local_m < mapping.len());
            let global_m = mapping[local_m];
            global_row[global_m / 64] |= 1u64 << (global_m % 64);
            bits &= bits - 1;
        }
    }
    global_row
}

fn build_sparse_from_filtered_blocks(
    block_samplers: &[CompiledSampler],
    meas_map: &[(usize, usize)],
    rank_offsets: &[usize],
    num_global_measurements: usize,
) -> SparseParity {
    let mut row_offsets = Vec::with_capacity(num_global_measurements + 1);
    let mut col_indices = Vec::new();

    for &(block_idx, local_meas) in meas_map {
        row_offsets.push(col_indices.len() as u32);
        let sparse = block_samplers[block_idx]
            .sparse
            .as_ref()
            .expect("filtered block samplers must retain sparse parity");
        let rank_offset = rank_offsets[block_idx] as u32;
        for &col in sparse.row_cols(local_meas) {
            col_indices.push(rank_offset + col);
        }
    }
    row_offsets.push(col_indices.len() as u32);

    let non_det_rows: Vec<u32> = (0..num_global_measurements as u32)
        .filter(|&m| row_offsets[m as usize + 1] != row_offsets[m as usize])
        .collect();

    SparseParity {
        col_indices,
        row_offsets,
        num_rows: num_global_measurements,
        non_det_rows,
    }
}

fn build_filtered_parity_blocks(
    block_samplers: Vec<CompiledSampler>,
    local_to_global: Vec<Vec<usize>>,
) -> Option<ParityBlocks> {
    let mut blocks = Vec::new();

    for (sampler, mapping) in block_samplers.into_iter().zip(local_to_global) {
        if mapping.is_empty() {
            continue;
        }

        if let Some(parity_blocks) = sampler.parity_blocks {
            for mut block in parity_blocks.blocks {
                for local_m in &mut block.meas_indices {
                    *local_m = mapping[*local_m];
                }
                blocks.push(block);
            }
            continue;
        }

        let sparse = sampler
            .sparse
            .expect("filtered block samplers must retain sparse parity");
        blocks.push(ParityBlock {
            meas_indices: mapping,
            sparse,
            block_rank: sampler.rank,
            ref_bits_packed: sampler.ref_bits_packed,
        });
    }

    ParityBlocks::from_blocks_if_useful(blocks)
}

impl CompiledSampler {
    fn from_parts(
        flip_rows: Vec<Vec<u64>>,
        ref_bits_packed: Vec<u64>,
        rank: usize,
        num_measurements: usize,
        sparse: Option<SparseParity>,
        parity_blocks: Option<ParityBlocks>,
        seed: u64,
    ) -> Self {
        Self {
            flip_rows,
            ref_bits_packed,
            rank,
            num_measurements,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: OnceLock::new(),
            shot_major_shots: 0,
            sparse,
            parity_blocks,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            #[cfg(feature = "gpu")]
            gpu_bts_cache: None,
        }
    }

    fn empty(seed: u64) -> Self {
        Self::from_parts(Vec::new(), Vec::new(), 0, 0, None, None, seed)
    }

    /// Number of independent random bits in the compiled parity system.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Opt in to GPU-accelerated BTS sampling. The sampler routes
    /// `sample_bulk_packed` through the GPU path when the circuit compiled
    /// to a flat sparse parity (no parity blocks) and the shot count
    /// crosses the GPU BTS threshold.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.gpu_context = Some(context);
        self.gpu_bts_cache = None;
        self
    }

    #[cfg(feature = "gpu")]
    fn can_use_gpu_bts(&self) -> bool {
        self.gpu_context.is_some()
            && self.sparse.is_some()
            && self.rank > 0
            && self.parity_blocks.is_none()
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn should_use_gpu_bts(&self, num_shots: usize) -> bool {
        let Some(sparse) = &self.sparse else {
            return false;
        };
        if !self.can_use_gpu_bts()
            || num_shots < crate::gpu::bts_min_shots()
            || self.rank < crate::gpu::bts_min_rank()
        {
            return false;
        }

        let stats = sparse.stats();
        let min_total_weight = sparse
            .num_rows
            .saturating_mul(crate::gpu::bts_min_weight_factor());
        stats.total_weight >= min_total_weight
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn has_gpu_context(&self) -> bool {
        self.gpu_context.is_some()
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn gpu_context(&self) -> Option<std::sync::Arc<crate::gpu::GpuContext>> {
        self.gpu_context.clone()
    }

    #[cfg(feature = "gpu")]
    fn should_try_gpu_counts(&self, total_shots: usize) -> bool {
        if !self.should_use_gpu_bts(total_shots) || self.rank <= MAX_RANK_FOR_RANK_SPACE {
            return false;
        }

        let m_words = self.num_measurements.div_ceil(64);
        if m_words == 0 || m_words > crate::gpu::kernels::bts::GPU_COUNTS_MAX_WORDS {
            return false;
        }

        let entropy_guard = total_shots.ilog2() as usize + 8;
        self.rank <= entropy_guard
    }

    #[cfg(feature = "gpu")]
    fn ensure_gpu_bts_cache(&mut self) -> Result<&mut GpuBtsCache> {
        if self.gpu_bts_cache.is_none() {
            let ctx = self
                .gpu_context
                .as_ref()
                .expect("gpu BTS cache requested without gpu_context")
                .clone();
            let sparse = self
                .sparse
                .as_ref()
                .expect("gpu BTS cache requested without sparse parity");
            self.gpu_bts_cache = Some(GpuBtsCache::new(&ctx, sparse, &self.ref_bits_packed)?);
        }
        Ok(self
            .gpu_bts_cache
            .as_mut()
            .expect("gpu BTS cache initialized above"))
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn sparse(&self) -> Option<&SparseParity> {
        self.sparse.as_ref()
    }

    pub fn parity_stats(&self) -> Option<ParityStats> {
        self.sparse.as_ref().map(|s| s.stats())
    }

    pub fn sample(&mut self) -> Vec<bool> {
        let num_meas_words = self.num_measurements.div_ceil(64);
        let mut accum = vec![0u64; num_meas_words];
        self.sample_into(&mut accum);
        self.unpack_result(&accum)
    }

    pub(crate) fn sample_bulk_words_shot_major(&mut self, num_shots: usize) -> (Vec<u64>, usize) {
        let m_words = self.num_measurements.div_ceil(64);
        let mut accum = vec![0u64; num_shots * m_words];
        let mut rand_buf = Vec::new();
        self.sample_bulk_words_shot_major_reuse(&mut accum, &mut rand_buf, num_shots);
        (accum, m_words)
    }

    pub(crate) fn sample_bulk_words_shot_major_reuse(
        &mut self,
        accum: &mut Vec<u64>,
        rand_buf: &mut Vec<u8>,
        num_shots: usize,
    ) -> usize {
        let m_words = self.num_measurements.div_ceil(64);
        let needed = num_shots * m_words;
        accum.resize(needed, 0);
        accum[..needed].fill(0);
        if num_shots == 0 || self.num_measurements == 0 || self.rank == 0 {
            return m_words;
        }

        if self.rank >= LUT_MIN_RANK {
            self.shot_major_shots = self.shot_major_shots.saturating_add(num_shots);
            let build_table = self.shot_major_shots >= MIN_SHOTS_FOR_FLIP_LUT;
            let lut = if build_table {
                FlipGroups::new(&self.lut, &self.flip_rows, m_words)
            } else {
                FlipGroups::existing(&self.lut, &self.flip_rows)
            };
            let total_groups = lut.total_groups();
            let bytes_per_shot = total_groups;
            let total_bytes = num_shots * bytes_per_shot;
            rand_buf.resize(total_bytes, 0);
            {
                let full_chunks = total_bytes / 8;
                let tail = full_chunks * 8;
                for i in 0..full_chunks {
                    let r = self.rng.next_u64();
                    rand_buf[i * 8..(i + 1) * 8].copy_from_slice(&r.to_le_bytes());
                }
                if tail < total_bytes {
                    let r = self.rng.next_u64();
                    let bytes = r.to_le_bytes();
                    rand_buf[tail..total_bytes].copy_from_slice(&bytes[..total_bytes - tail]);
                }
                if lut.remainder_size > 0 {
                    let remainder_mask = (1u8 << lut.remainder_size) - 1;
                    let last_group = lut.num_full_groups;
                    for s in 0..num_shots {
                        rand_buf[s * bytes_per_shot + last_group] &= remainder_mask;
                    }
                }
            }

            let max_batch = if m_words > 0 {
                (256 * 1024 / (m_words * 8)).max(64)
            } else {
                num_shots
            };

            let apply_seq = |dst: &mut [u64]| {
                for tile_start in (0..num_shots).step_by(max_batch) {
                    let tile_end = (tile_start + max_batch).min(num_shots);
                    for g in 0..total_groups {
                        for s in tile_start..tile_end {
                            let byte = rand_buf[s * bytes_per_shot + g] as usize;
                            let shot_base = s * m_words;
                            lut.xor_into(&mut dst[shot_base..shot_base + m_words], g, byte);
                        }
                    }
                }
            };

            #[cfg(feature = "parallel")]
            const PAR_SHOT_THRESHOLD: usize = 256;

            #[cfg(feature = "parallel")]
            if num_shots >= PAR_SHOT_THRESHOLD {
                use rayon::prelude::*;
                let shots_per_chunk =
                    (num_shots.div_ceil(rayon::current_num_threads())).max(max_batch);
                let chunk_m = shots_per_chunk * m_words;
                accum
                    .par_chunks_mut(chunk_m)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let chunk_shots = chunk.len() / m_words;
                        let chunk_start = ci * shots_per_chunk;
                        for tile_start in (0..chunk_shots).step_by(max_batch) {
                            let tile_end = (tile_start + max_batch).min(chunk_shots);
                            for g in 0..total_groups {
                                for s in tile_start..tile_end {
                                    let gs = chunk_start + s;
                                    let byte = rand_buf[gs * bytes_per_shot + g] as usize;
                                    let base = s * m_words;
                                    lut.xor_into(&mut chunk[base..base + m_words], g, byte);
                                }
                            }
                        }
                    });
            } else {
                apply_seq(accum);
            }

            #[cfg(not(feature = "parallel"))]
            apply_seq(accum);

            m_words
        } else {
            for s in 0..num_shots {
                let shot_base = s * m_words;
                let shot_accum = &mut accum[shot_base..shot_base + m_words];
                for j in 0..self.rank {
                    let bit = self.rng.next_u32() & 1;
                    if bit != 0 {
                        let row = &self.flip_rows[j];
                        xor_words(shot_accum, row);
                    }
                }
            }
            m_words
        }
    }

    fn should_use_bts(&self, num_shots: usize) -> bool {
        if let Some(sparse) = &self.sparse {
            if self.rank == 0 {
                return false;
            }
            let m_words = self.num_measurements.div_ceil(64) as u64;
            let lut_groups = (self.rank.div_ceil(LUT_GROUP_SIZE)) as u64;

            let lut_alloc_bytes = num_shots as u64 * (lut_groups + m_words * 8);
            if lut_alloc_bytes > MAX_LUT_ALLOC_BYTES {
                return true;
            }

            let s_words = num_shots.div_ceil(64);
            let stats = sparse.stats();
            let bts_work = stats.total_weight as u64 * s_words as u64;
            let lut_work = num_shots as u64 * lut_groups * m_words;
            bts_work < lut_work
        } else {
            false
        }
    }

    pub(crate) fn ref_bits_packed(&self) -> &[u64] {
        &self.ref_bits_packed
    }

    /// Packed measurement flips, one row per random bit; every sample is the reference
    /// bits XOR a subset of these rows.
    pub(crate) fn flip_rows(&self) -> &[Vec<u64>] {
        &self.flip_rows
    }

    /// Restart the random-bit stream as a sampler compiled with `seed` starts it.
    pub(crate) fn reseed(&mut self, seed: u64) {
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }

    /// Replace the reference bits with `reference`, another point of the same affine
    /// space of outcomes, so the samples no longer depend on the seed the reference was
    /// drawn with.
    pub(crate) fn rebase_reference(&mut self, reference: Vec<u64>) {
        debug_assert_eq!(reference.len(), self.ref_bits_packed.len());
        if reference == self.ref_bits_packed {
            return;
        }
        if let Some(parity_blocks) = &mut self.parity_blocks {
            for block in &mut parity_blocks.blocks {
                block.ref_bits_packed.fill(0);
                for (local, &measurement) in block.meas_indices.iter().enumerate() {
                    if get_bit(&reference, measurement) {
                        block.ref_bits_packed[local / 64] |= 1u64 << (local % 64);
                    }
                }
            }
        }
        self.ref_bits_packed = reference;
        #[cfg(feature = "gpu")]
        {
            self.gpu_bts_cache = None;
        }
    }

    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.sample_bulk_packed(num_shots).to_shots()
    }

    /// Materialise packed shots directly on the GPU.
    ///
    /// Requires a GPU context and a circuit that compiled to the flat sparse
    /// BTS path. Use
    /// [`DevicePackedShots::to_host`] to copy the full packed payload back, or
    /// [`DevicePackedShots::marginals`] / [`DevicePackedShots::counts`] to
    /// reduce on device first.
    #[cfg(feature = "gpu")]
    pub fn sample_bulk_packed_device(&mut self, num_shots: usize) -> Result<DevicePackedShots> {
        if !self.can_use_gpu_bts() {
            return Err(PrismError::BackendUnsupported {
                backend: "CompiledSampler".to_string(),
                operation: "sample_bulk_packed_device requires with_gpu() and flat sparse BTS"
                    .to_string(),
            });
        }

        let ctx = self
            .gpu_context
            .as_ref()
            .expect("gpu BTS selected without gpu_context")
            .clone();
        let rank = self.rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let cache = self.ensure_gpu_bts_cache()?;
        let data = crate::gpu::kernels::bts::launch_bts_sample_device(
            &ctx,
            &mut fast_rng,
            rank,
            num_shots,
            cache,
        )?;

        Ok(DevicePackedShots::new(
            ctx,
            data,
            num_shots,
            self.num_measurements,
            rank,
        ))
    }

    /// GPU BTS sampling with on-device meas-major to shot-major bit-transpose.
    ///
    /// `Some(Ok(data))` with `data` shot-major (`num_shots * m_words` words)
    /// when the circuit takes the GPU BTS path (flat sparse, shot threshold
    /// crossed), sparing the host `into_shot_major_data()` transpose. `None`
    /// means use the CPU path.
    #[cfg(feature = "gpu")]
    pub(crate) fn try_sample_bulk_shot_major_gpu(
        &mut self,
        num_shots: usize,
    ) -> Option<Result<Vec<u64>>> {
        if !self.should_use_gpu_bts(num_shots) {
            return None;
        }
        let ctx = self
            .gpu_context
            .as_ref()
            .expect("gpu BTS selected without gpu_context")
            .clone();
        let rank = self.rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let cache = match self.ensure_gpu_bts_cache() {
            Ok(c) => c,
            Err(e) => return Some(Err(e)),
        };
        Some(crate::gpu::kernels::bts::launch_bts_sample_shot_major_host(
            &ctx,
            &mut fast_rng,
            rank,
            num_shots,
            cache,
        ))
    }

    #[inline(always)]
    fn sample_into(&mut self, accum: &mut [u64]) {
        if self.rank == 0 {
            return;
        }

        if self.rank >= LUT_MIN_RANK {
            let lut = FlipGroups::new(&self.lut, &self.flip_rows, accum.len());
            let rng = &mut self.rng;
            match lut.table {
                Some(table) => {
                    draw_flip_groups(rng, &lut, |g, byte| xor_words(accum, table.lookup(g, byte)))
                }
                None => draw_flip_groups(rng, &lut, |g, byte| lut.xor_rows(accum, g, byte)),
            }
        } else {
            for j in 0..self.rank {
                let bit = self.rng.next_u32() & 1;
                if bit != 0 {
                    let row = &self.flip_rows[j];
                    xor_words(accum, row);
                }
            }
        }
    }

    #[inline(always)]
    fn unpack_result(&self, accum: &[u64]) -> Vec<bool> {
        let mut result = Vec::with_capacity(self.num_measurements);
        for m in 0..self.num_measurements {
            let w = m / 64;
            let ref_word = if w < self.ref_bits_packed.len() {
                self.ref_bits_packed[w]
            } else {
                0
            };
            let bit = ((accum[w] ^ ref_word) >> (m % 64)) & 1 != 0;
            result.push(bit);
        }
        result
    }

    /// Draw `num_shots` shots as a packed bit matrix. The result layout
    /// depends on the sampling path taken; branch on [`PackedShots::layout`].
    pub fn sample_bulk_packed(&mut self, num_shots: usize) -> PackedShots {
        match self.sample_bulk_packed_inner(num_shots, false) {
            Ok(packed) => packed,
            Err(_) => unreachable!("compiled sampler CPU fallback should not fail"),
        }
    }

    pub fn try_sample_bulk_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        self.sample_bulk_packed_inner(num_shots, true)
    }

    fn sample_bulk_packed_inner(
        &mut self,
        num_shots: usize,
        propagate_gpu_errors: bool,
    ) -> Result<PackedShots> {
        let m_words = self.num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        if num_shots == 0 || self.num_measurements == 0 {
            return Ok(PackedShots::from_parts(
                Vec::new(),
                num_shots,
                self.num_measurements,
                ShotLayout::ShotMajor,
            ));
        }
        if self.rank == 0 {
            let mut data = vec![0u64; num_shots * m_words];
            for s in 0..num_shots {
                let base = s * m_words;
                data[base..base + m_words].copy_from_slice(&self.ref_bits_packed);
            }
            return Ok(PackedShots::from_parts(
                data,
                num_shots,
                self.num_measurements,
                ShotLayout::ShotMajor,
            ));
        }

        if self.should_use_bts(num_shots) {
            return self.sample_bulk_packed_bts(num_shots, s_words, propagate_gpu_errors);
        }

        let (mut data, _) = self.sample_bulk_words_shot_major(num_shots);
        for s in 0..num_shots {
            let base = s * m_words;
            xor_words(&mut data[base..base + m_words], &self.ref_bits_packed);
        }
        Ok(PackedShots::from_parts(
            data,
            num_shots,
            self.num_measurements,
            ShotLayout::ShotMajor,
        ))
    }

    fn sample_bulk_packed_bts(
        &mut self,
        num_shots: usize,
        s_words: usize,
        propagate_gpu_errors: bool,
    ) -> Result<PackedShots> {
        #[cfg(not(feature = "gpu"))]
        let _ = propagate_gpu_errors;

        let num_meas = self.num_measurements;

        if let Some(pb) = &self.parity_blocks {
            let block_seeds: Vec<u64> = (0..pb.blocks.len()).map(|_| self.rng.next_u64()).collect();
            let meas_major = if pb.direct_scatter {
                let total_len = num_meas * s_words;
                #[allow(clippy::uninit_vec)]
                let mut meas_major = {
                    let mut v = Vec::with_capacity(total_len);
                    // SAFETY: Every measurement row is written exactly once by one parity
                    // block before the buffer is read. The filtered parity blocks form a
                    // partition of the global measurement rows.
                    unsafe { v.set_len(total_len) };
                    v
                };

                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;

                    let ptr = SendPtrU64(meas_major.as_mut_ptr());
                    pb.blocks
                        .par_iter()
                        .zip(block_seeds.par_iter())
                        .for_each(|(block, &seed)| {
                            let block_data = sample_block(block, seed, num_shots);

                            // SAFETY: Each parity block owns a disjoint set of global
                            // measurement rows, so these row copies target non-overlapping
                            // regions of the final meas-major output.
                            unsafe {
                                for (local_m, &global_m) in block.meas_indices.iter().enumerate() {
                                    let row =
                                        &block_data[local_m * s_words..(local_m + 1) * s_words];
                                    ptr.copy_from_slice(global_m * s_words, row);
                                }
                            }
                        });
                }

                #[cfg(not(feature = "parallel"))]
                {
                    for (block, &seed) in pb.blocks.iter().zip(block_seeds.iter()) {
                        let block_data = sample_block(block, seed, num_shots);
                        scatter_meas_major_rows(
                            &mut meas_major,
                            s_words,
                            &block_data,
                            s_words,
                            &block.meas_indices,
                        );
                    }
                }

                meas_major
            } else {
                #[cfg(feature = "parallel")]
                let block_results: Vec<(Vec<u64>, &[usize])> = {
                    use rayon::prelude::*;
                    pb.blocks
                        .par_iter()
                        .zip(block_seeds.par_iter())
                        .map(|(block, &seed)| {
                            (
                                sample_block(block, seed, num_shots),
                                block.meas_indices.as_slice(),
                            )
                        })
                        .collect()
                };

                #[cfg(not(feature = "parallel"))]
                let block_results: Vec<(Vec<u64>, &[usize])> = pb
                    .blocks
                    .iter()
                    .zip(block_seeds.iter())
                    .map(|(block, &seed)| {
                        (
                            sample_block(block, seed, num_shots),
                            block.meas_indices.as_slice(),
                        )
                    })
                    .collect();

                let mut meas_major = vec![0u64; num_meas * s_words];
                for (block_data, meas_indices) in &block_results {
                    scatter_meas_major_rows(
                        &mut meas_major,
                        s_words,
                        block_data,
                        s_words,
                        meas_indices,
                    );
                }
                meas_major
            };

            return Ok(PackedShots::from_parts(
                meas_major,
                num_shots,
                num_meas,
                ShotLayout::MeasMajor,
            ));
        }

        #[cfg(feature = "gpu")]
        if self.should_use_gpu_bts(num_shots) {
            match self.sample_bulk_packed_bts_gpu(num_shots) {
                Ok(packed) => return Ok(packed),
                Err(e) if propagate_gpu_errors => return Err(e),
                Err(_) => {}
            }
        }

        let sparse = self
            .sparse
            .as_ref()
            .expect("sparse parity required for BTS (should_use_bts guards this)");

        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);

        if num_shots <= BTS_BATCH_SHOTS {
            let data = bts_single_pass(
                sparse,
                num_shots,
                &self.ref_bits_packed,
                &mut fast_rng,
                self.rank,
            );
            return Ok(PackedShots::from_parts(
                data,
                num_shots,
                num_meas,
                ShotLayout::MeasMajor,
            ));
        }

        let data = bts_batched(
            sparse,
            num_shots,
            s_words,
            &self.ref_bits_packed,
            &mut fast_rng,
            self.rank,
        );
        Ok(PackedShots::from_parts(
            data,
            num_shots,
            num_meas,
            ShotLayout::MeasMajor,
        ))
    }

    #[cfg(feature = "gpu")]
    fn sample_bulk_packed_bts_gpu(&mut self, num_shots: usize) -> Result<PackedShots> {
        let ctx = self
            .gpu_context
            .as_ref()
            .expect("gpu BTS selected without gpu_context")
            .clone();
        let rank = self.rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let cache = self.ensure_gpu_bts_cache()?;
        let data = crate::gpu::kernels::bts::launch_bts_sample(
            &ctx,
            &mut fast_rng,
            rank,
            num_shots,
            cache,
        )?;
        Ok(PackedShots::from_parts(
            data,
            num_shots,
            self.num_measurements,
            ShotLayout::MeasMajor,
        ))
    }

    /// Stream `total_shots` through `acc` in [`default_chunk_size`] chunks,
    /// bounding peak memory.
    pub fn sample_chunked<A: ShotAccumulator>(&mut self, total_shots: usize, acc: &mut A) {
        let chunk_size = default_chunk_size(self.num_measurements);
        self.sample_chunked_with_size(total_shots, chunk_size, acc);
    }

    pub fn sample_chunked_with_size<A: ShotAccumulator>(
        &mut self,
        total_shots: usize,
        chunk_size: usize,
        acc: &mut A,
    ) {
        for_each_chunk(total_shots, chunk_size, |batch| {
            let packed = self.sample_bulk_packed(batch);
            acc.accumulate(&packed);
        });
    }

    /// Histogram of `total_shots` outcomes keyed by packed measurement words.
    /// Ranks small relative to the shot count use a closed-form multinomial
    /// draw with no per-shot work.
    ///
    /// A GPU failure falls back to the CPU sampler; [`Self::try_sample_counts`]
    /// surfaces it instead.
    pub fn sample_counts(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        match self.try_sample_counts(total_shots) {
            Ok(counts) => counts,
            Err(_) => self.sample_counts_cpu(total_shots),
        }
    }

    pub fn try_sample_counts(
        &mut self,
        total_shots: usize,
    ) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        #[cfg(feature = "gpu")]
        if self.should_try_gpu_counts(total_shots) {
            return self.sample_bulk_packed_device(total_shots)?.counts();
        }

        Ok(self.sample_counts_cpu(total_shots))
    }

    fn sample_counts_cpu(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        if self.rank == 0 {
            let mut counts = std::collections::HashMap::new();
            if total_shots > 0 {
                let m_words = self.num_measurements.div_ceil(64);
                counts.insert(self.ref_bits_packed[..m_words].to_vec(), total_shots as u64);
            }
            return counts;
        }

        if self.parity_blocks.is_none() {
            let num_outcomes = 1usize.checked_shl(self.rank as u32).unwrap_or(usize::MAX);

            if self.rank <= MAX_RANK_FOR_MULTINOMIAL
                && total_shots >= num_outcomes * MIN_SHOTS_PER_OUTCOME_MULTINOMIAL
            {
                #[cfg(feature = "parallel")]
                if num_outcomes >= MIN_KEYS_FOR_PAR_MULTINOMIAL {
                    return self.sample_counts_multinomial_pooled(total_shots);
                }
                return self.sample_counts_multinomial(total_shots);
            }

            if self.rank <= MAX_RANK_FOR_RANK_SPACE
                && total_shots >= num_outcomes * MIN_SHOTS_PER_OUTCOME
            {
                return self.sample_counts_rank_space(total_shots);
            }
        }
        let mut acc = if shots_mostly_distinct(self.rank, total_shots) {
            HistogramAccumulator::direct()
        } else {
            HistogramAccumulator::new()
        };
        self.sample_chunked(total_shots, &mut acc);
        acc.into_counts()
    }

    fn sample_counts_multinomial(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;

        let m_words = self.num_measurements.div_ceil(64);

        if total_shots == 0 || self.num_measurements == 0 {
            return HashMap::new();
        }

        let rank = self.rank;
        let num_outcomes = 1usize << rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let lut =
            (rank >= LUT_MIN_RANK).then(|| FlipGroups::new(&self.lut, &self.flip_rows, m_words));
        let mut counts = HashMap::new();
        let mut remaining = total_shots;

        for key in 0..num_outcomes {
            if remaining == 0 {
                break;
            }
            let outcomes_left = num_outcomes - key;
            let count = if outcomes_left == 1 {
                remaining
            } else {
                binomial_sample(&mut fast_rng, remaining, 1.0 / outcomes_left as f64)
            };

            if count > 0 {
                let mut outcome = self.ref_bits_packed[..m_words].to_vec();
                if let Some(lut) = lut {
                    for g in 0..lut.total_groups() {
                        let byte = (key >> (g * 8)) & 0xFF;
                        lut.xor_into(&mut outcome, g, byte);
                    }
                } else {
                    for j in 0..rank {
                        if (key >> j) & 1 != 0 {
                            xor_words(&mut outcome, &self.flip_rows[j]);
                        }
                    }
                }
                counts.insert(outcome, count as u64);
            }
            remaining -= count;
        }

        counts
    }

    #[cfg(feature = "parallel")]
    fn sample_counts_multinomial_pooled(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        use rayon::prelude::*;
        use std::collections::HashMap;

        let m_words = self.num_measurements.div_ceil(64);

        if total_shots == 0 || self.num_measurements == 0 {
            return HashMap::new();
        }

        let rank = self.rank;
        let num_outcomes = 1usize << rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let mut key_counts = vec![0u64; num_outcomes];
        let mut nonzero_keys = 0;
        let mut remaining = total_shots;
        for (key, key_count) in key_counts.iter_mut().enumerate() {
            if remaining == 0 {
                break;
            }
            let outcomes_left = num_outcomes - key;
            let count = if outcomes_left == 1 {
                remaining
            } else {
                binomial_sample(&mut fast_rng, remaining, 1.0 / outcomes_left as f64)
            };
            *key_count = count as u64;
            nonzero_keys += usize::from(count > 0);
            remaining -= count;
        }

        let lut =
            (rank >= LUT_MIN_RANK).then(|| FlipGroups::new(&self.lut, &self.flip_rows, m_words));
        let ref_bits = &self.ref_bits_packed[..m_words];
        let flip_rows = &self.flip_rows;
        let outcome = |key: usize| {
            let mut outcome = ref_bits.to_vec();
            if let Some(lut) = lut {
                for g in 0..lut.total_groups() {
                    let byte = (key >> (g * 8)) & 0xFF;
                    lut.xor_into(&mut outcome, g, byte);
                }
            } else {
                for (j, row) in flip_rows.iter().enumerate() {
                    if (key >> j) & 1 != 0 {
                        xor_words(&mut outcome, row);
                    }
                }
            }
            outcome
        };

        let keys_per_thread = num_outcomes.div_ceil(rayon::current_num_threads());
        let shards: Vec<Vec<(Vec<u64>, u64)>> = key_counts
            .par_chunks(keys_per_thread)
            .enumerate()
            .map(|(tid, shard)| {
                let shard_start = tid * keys_per_thread;
                shard
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .map(|(i, &count)| (outcome(shard_start + i), count))
                    .collect()
            })
            .collect();

        let mut counts = HashMap::with_capacity(nonzero_keys);
        for shard in shards {
            counts.extend(shard);
        }
        counts
    }

    fn sample_counts_rank_space(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;

        let m_words = self.num_measurements.div_ceil(64);

        if total_shots == 0 || self.num_measurements == 0 {
            return HashMap::new();
        }

        let rank = self.rank;
        let num_outcomes = 1usize << rank;
        let mut rank_counts = vec![0u64; num_outcomes];
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
        let lut =
            (rank >= LUT_MIN_RANK).then(|| FlipGroups::new(&self.lut, &self.flip_rows, m_words));

        if let Some(lut) = lut {
            let bytes_per_shot = lut.total_groups();
            let remainder_mask: u8 = if lut.remainder_size > 0 {
                (1u8 << lut.remainder_size) - 1
            } else {
                0xFF
            };

            let chunk_size = (32 * 1024 * 1024 / bytes_per_shot).max(64);
            let mut rand_buf = vec![0u8; chunk_size * bytes_per_shot];
            let mut remaining = total_shots;

            while remaining > 0 {
                let this_chunk = remaining.min(chunk_size);
                let total_bytes = this_chunk * bytes_per_shot;

                let full_chunks = total_bytes / 8;
                let tail = full_chunks * 8;
                for i in 0..full_chunks {
                    let r = fast_rng.next_u64();
                    rand_buf[i * 8..(i + 1) * 8].copy_from_slice(&r.to_le_bytes());
                }
                if tail < total_bytes {
                    let r = fast_rng.next_u64();
                    let bytes = r.to_le_bytes();
                    rand_buf[tail..total_bytes].copy_from_slice(&bytes[..total_bytes - tail]);
                }

                if lut.remainder_size > 0 {
                    let last_group = lut.num_full_groups;
                    for s in 0..this_chunk {
                        rand_buf[s * bytes_per_shot + last_group] &= remainder_mask;
                    }
                }

                for s in 0..this_chunk {
                    let base = s * bytes_per_shot;
                    let mut key: usize = 0;
                    for g in 0..bytes_per_shot {
                        key |= (rand_buf[base + g] as usize) << (g * 8);
                    }
                    rank_counts[key] += 1;
                }

                remaining -= this_chunk;
            }
        } else {
            let mut remaining = total_shots;
            while remaining > 0 {
                let this_chunk = remaining.min(4 * 1024 * 1024);
                for _ in 0..this_chunk {
                    let bits = fast_rng.next_u64();
                    let key = (bits as usize) & (num_outcomes - 1);
                    rank_counts[key] += 1;
                }
                remaining -= this_chunk;
            }
        }

        let mut counts = HashMap::new();
        for (key, &count) in rank_counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let mut outcome = self.ref_bits_packed[..m_words].to_vec();
            if let Some(lut) = lut {
                for g in 0..lut.total_groups() {
                    let byte = (key >> (g * 8)) & 0xFF;
                    lut.xor_into(&mut outcome, g, byte);
                }
            } else {
                for j in 0..rank {
                    if (key >> j) & 1 != 0 {
                        xor_words(&mut outcome, &self.flip_rows[j]);
                    }
                }
            }
            counts.insert(outcome, count);
        }

        counts
    }

    /// Per-measurement frequency of reading 1 across `total_shots` sampled
    /// shots.
    pub fn sample_marginals(&mut self, total_shots: usize) -> Vec<f64> {
        match self.try_sample_marginals(total_shots) {
            Ok(marginals) => marginals,
            Err(_) => self.sample_marginals_cpu(total_shots),
        }
    }

    pub fn try_sample_marginals(&mut self, total_shots: usize) -> Result<Vec<f64>> {
        #[cfg(feature = "gpu")]
        if self.should_use_gpu_bts(total_shots) {
            return self.sample_bulk_packed_device(total_shots)?.marginals();
        }

        Ok(self.sample_marginals_cpu(total_shots))
    }

    fn sample_marginals_cpu(&mut self, total_shots: usize) -> Vec<f64> {
        marginals_from_chunks(self.num_measurements, |acc| {
            self.sample_chunked(total_shots, acc)
        })
    }

    /// Sample packed detection events: event `e` holds the XOR of measurement
    /// pair `pairs[e]`'s outcomes in each shot.
    ///
    /// # Panics
    /// Panics if the sampler was compiled with no measurements.
    pub fn sample_detection_events(
        &mut self,
        pairs: &[(usize, usize)],
        num_shots: usize,
    ) -> PackedShots {
        let sparse = self.sparse.as_ref().expect("sparse parity required");
        let det_sparse = sparse.compile_detection_events(pairs);
        let num_events = det_sparse.num_rows;
        let m_words = num_events.div_ceil(64);
        let s_words = num_shots.div_ceil(64);

        let det_weight = det_sparse.stats().total_weight;
        let meas_weight = sparse.stats().total_weight;
        if self.rank > 0 && num_shots > 0 && det_weight > meas_weight + num_events {
            let rows: Vec<Vec<usize>> = pairs.iter().map(|&(a, b)| vec![a, b]).collect();
            return self
                .sample_bulk_packed(num_shots)
                .parity_rows(&rows)
                .expect("compile_detection_events bounds-checked every pair");
        }

        let ref_bit = |m: usize| (self.ref_bits_packed[m / 64] >> (m % 64)) & 1;
        let mut det_ref = vec![0u64; m_words];
        for (e, &(m_a, m_b)) in pairs.iter().enumerate() {
            det_ref[e / 64] |= (ref_bit(m_a) ^ ref_bit(m_b)) << (e % 64);
        }

        let data = if num_events == 0 || num_shots == 0 || self.rank == 0 {
            let mut data = vec![0u64; num_events * s_words];
            bts::apply_ref_bits_meas_major(&mut data, &det_ref, num_events, s_words, num_shots);
            data
        } else {
            let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
            if num_shots > BTS_BATCH_SHOTS {
                bts_batched(
                    &det_sparse,
                    num_shots,
                    s_words,
                    &det_ref,
                    &mut fast_rng,
                    self.rank,
                )
            } else {
                sample_bts_meas_major(&det_sparse, num_shots, &det_ref, &mut fast_rng, self.rank)
            }
        };

        PackedShots::from_parts(data, num_shots, num_events, ShotLayout::MeasMajor)
    }

    /// Exact outcome multiplicities over all `2^rank` equally likely
    /// random-bit assignments; counts sum to `2^rank`. `None` when `rank`
    /// exceeds 25 or no sparse parity is available.
    pub fn exact_counts(&self) -> Option<std::collections::HashMap<Vec<u64>, u64>> {
        if self.rank > MAX_RANK_FOR_GRAY_CODE {
            return None;
        }
        let sparse = self.sparse.as_ref()?;
        Some(gray_code_exact_counts(
            sparse,
            self.rank,
            &self.ref_bits_packed,
            self.num_measurements,
        ))
    }

    /// Analytic per-measurement probabilities: 0.5 for any measurement that
    /// depends on a random bit, otherwise the deterministic reference bit.
    pub(crate) fn marginal_probabilities(&self) -> Vec<f64> {
        let mut probs = vec![0.5f64; self.num_measurements];
        if let Some(sparse) = &self.sparse {
            for (m, p) in probs.iter_mut().enumerate() {
                if sparse.row_weight(m) == 0 {
                    let ref_bit = (self.ref_bits_packed[m / 64] >> (m % 64)) & 1;
                    *p = ref_bit as f64;
                }
            }
        } else {
            for (m, p) in probs.iter_mut().enumerate() {
                let mut depends_on_random = false;
                for row in &self.flip_rows {
                    let w = m / 64;
                    if w < row.len() && (row[w] >> (m % 64)) & 1 != 0 {
                        depends_on_random = true;
                        break;
                    }
                }
                if !depends_on_random {
                    let ref_bit = (self.ref_bits_packed[m / 64] >> (m % 64)) & 1;
                    *p = ref_bit as f64;
                }
            }
        }
        probs
    }

    pub fn parity_report(&self) -> String {
        let mut report = format!(
            "CompiledSampler: {} measurements, rank {}, {} flip rows\n",
            self.num_measurements,
            self.rank,
            self.flip_rows.len()
        );
        if let Some(sparse) = &self.sparse {
            let stats = sparse.stats();
            report.push_str(&format!(
                "Parity matrix: {} rows, total weight {}\n\
                 Weight range: {} to {}, mean {:.1}\n\
                 Deterministic measurements: {}\n",
                sparse.num_rows,
                stats.total_weight,
                stats.min_weight,
                stats.max_weight,
                stats.mean_weight,
                stats.num_deterministic,
            ));
            let mut histogram = [0usize; 8];
            for m in 0..sparse.num_rows {
                let w = sparse.row_weight(m);
                let bucket = w.min(7);
                histogram[bucket] += 1;
            }
            report.push_str("Weight histogram: ");
            for (i, &count) in histogram.iter().enumerate() {
                if count > 0 {
                    if i < 7 {
                        report.push_str(&format!("w{}={} ", i, count));
                    } else {
                        report.push_str(&format!("w7+={} ", count));
                    }
                }
            }
            report.push('\n');
        } else {
            report.push_str("No sparse parity matrix available\n");
        }
        report
    }
}

impl CompiledDetectorSampler {
    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn num_detectors(&self) -> usize {
        self.detector_rows.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observable_rows.len()
    }

    pub fn rank(&self) -> usize {
        self.measurement_sampler.rank()
    }

    pub fn detector_rows(&self) -> &[Vec<usize>] {
        &self.detector_rows
    }

    pub fn observable_rows(&self) -> &[Vec<usize>] {
        &self.observable_rows
    }

    #[cfg(test)]
    pub(crate) fn measurement_sampler(&self) -> &CompiledSampler {
        &self.measurement_sampler
    }

    /// Opt the underlying measurement sampler into GPU BTS sampling.
    #[cfg(feature = "gpu")]
    pub fn with_gpu(mut self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.measurement_sampler = self.measurement_sampler.with_gpu(context);
        self
    }

    /// Sample packed measurement records; errors surface only from an
    /// attached GPU context.
    pub fn sample_measurements_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        self.measurement_sampler.try_sample_bulk_packed(num_shots)
    }

    pub fn sample_detectors_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        let measurements = self.sample_measurements_packed(num_shots)?;
        measurements.parity_rows(&self.detector_rows)
    }

    pub fn sample_observables_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        let measurements = self.sample_measurements_packed(num_shots)?;
        measurements.parity_rows(&self.observable_rows)
    }

    /// Sample measurements once and derive detector and observable bits from
    /// the same shots.
    pub fn sample_packed(&mut self, num_shots: usize) -> Result<DetectorSampleBatch> {
        let measurements = self.sample_measurements_packed(num_shots)?;
        let detectors = measurements.parity_rows(&self.detector_rows)?;
        let observables = measurements.parity_rows(&self.observable_rows)?;
        Ok(DetectorSampleBatch {
            measurements,
            detectors,
            observables,
        })
    }

    pub fn sample_detectors_chunked<A: ShotAccumulator>(
        &mut self,
        total_shots: usize,
        acc: &mut A,
    ) -> Result<()> {
        let chunk_size = self.detector_chunk_size();
        let mut remaining = total_shots;
        while remaining > 0 {
            let batch = remaining.min(chunk_size);
            let packed = self.sample_detectors_packed(batch)?;
            acc.accumulate(&packed);
            remaining -= batch;
        }
        Ok(())
    }

    fn detector_chunk_size(&self) -> usize {
        default_chunk_size(self.num_measurements.max(self.detector_rows.len()))
    }

    pub fn sample_detector_counts(
        &mut self,
        total_shots: usize,
    ) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        let mut acc = HistogramAccumulator::new();
        self.sample_detectors_chunked(total_shots, &mut acc)?;
        Ok(acc.into_counts())
    }
}

fn gray_code_exact_counts(
    sparse: &SparseParity,
    rank: usize,
    ref_bits: &[u64],
    num_measurements: usize,
) -> std::collections::HashMap<Vec<u64>, u64> {
    use std::collections::HashMap;

    let m_words = num_measurements.div_ceil(64);
    let mut meas_vec = ref_bits[..m_words].to_vec();
    let total: u64 = 1u64 << rank;

    let mut col_words: Vec<Vec<u64>> = Vec::with_capacity(rank);
    for col in 0..rank {
        let mut cw = vec![0u64; m_words];
        for m in 0..num_measurements {
            let start = sparse.row_offsets[m] as usize;
            let end = sparse.row_offsets[m + 1] as usize;
            for &c in &sparse.col_indices[start..end] {
                if c as usize == col {
                    cw[m / 64] |= 1u64 << (m % 64);
                }
            }
        }
        col_words.push(cw);
    }

    let mut counts: HashMap<Vec<u64>, u64> = HashMap::new();
    *counts.entry(meas_vec.clone()).or_insert(0) += 1;

    for step in 1..total {
        let bit_to_flip = step.trailing_zeros() as usize;
        let col = &col_words[bit_to_flip];
        for (mw, cw) in meas_vec.iter_mut().zip(col.iter()) {
            *mw ^= cw;
        }
        *counts.entry(meas_vec.clone()).or_insert(0) += 1;
    }

    counts
}

const MAX_RANK_FOR_GRAY_CODE: usize = 25;
const MAX_RANK_FOR_MULTINOMIAL: usize = 22;
const MIN_SHOTS_PER_OUTCOME_MULTINOMIAL: usize = 8;
/// Outcome count (`1 << rank`) from which multinomial counts build outcomes on the pool.
/// The binomial draws stay serial in key order, so counts do not depend on the pool size.
#[cfg(feature = "parallel")]
const MIN_KEYS_FOR_PAR_MULTINOMIAL: usize = 1 << 13;
const MAX_RANK_FOR_RANK_SPACE: usize = 20;
const MIN_SHOTS_PER_OUTCOME: usize = 4;
const MAX_LUT_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

/// Whether `shots` uniform draws over the `2^rank` outcomes are expected to repeat less
/// than half a time (`shots^2 <= 2^rank`), so a count can skip the typed map.
fn shots_mostly_distinct(rank: usize, shots: usize) -> bool {
    rank >= 128 || (shots as u128).saturating_mul(shots as u128) <= 1u128 << rank
}

fn incompatible(reason: impl Into<String>) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "CompiledSampler".to_string(),
        reason: reason.into(),
    }
}

fn require_clifford(circuit: &Circuit) -> Result<()> {
    if circuit.is_clifford_only() {
        return Ok(());
    }
    Err(incompatible("circuit contains non-Clifford gates"))
}

/// [`require_clifford`], then, once the circuit measures, no resets and terminal
/// measurements with no classical conditionals.
fn require_compilable(circuit: &Circuit) -> Result<()> {
    require_clifford(circuit)?;

    let has_measurements = circuit
        .instructions
        .iter()
        .any(|inst| matches!(inst, Instruction::Measure { .. }));
    if has_measurements && circuit.has_resets() {
        return Err(incompatible(
            "compiled measurement sampling does not support reset instructions",
        ));
    }
    if has_measurements && !circuit.has_terminal_measurements_only() {
        return Err(incompatible(
            "compiled measurement sampling requires terminal measurements and does not support classical conditionals",
        ));
    }
    Ok(())
}

fn finish_sampler(
    mut flip_rows: Vec<Vec<u64>>,
    rank: usize,
    num_measurements: usize,
    ref_bits: &[bool],
    seed: u64,
) -> CompiledSampler {
    minimize_flip_row_weight(&mut flip_rows);

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_measurements);
    let ref_bits_packed = pack_bools(ref_bits);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, rank, &ref_bits_packed);

    CompiledSampler::from_parts(
        flip_rows,
        ref_bits_packed,
        rank,
        num_measurements,
        Some(sparse),
        parity_blocks,
        seed,
    )
}

fn compile_measurements_filtered(
    circuit: &Circuit,
    blocks: &[Vec<usize>],
    seed: u64,
) -> Result<CompiledSampler> {
    let num_global_measurements: usize = circuit
        .instructions
        .iter()
        .filter(|i| matches!(i, Instruction::Measure { .. }))
        .count();

    if num_global_measurements == 0 {
        return Ok(CompiledSampler::empty(seed));
    }

    let mut qubit_to_block: Vec<usize> = vec![0; circuit.num_qubits];
    for (bi, block) in blocks.iter().enumerate() {
        for &q in block {
            qubit_to_block[q] = bi;
        }
    }

    let mut block_samplers: Vec<CompiledSampler> = Vec::with_capacity(blocks.len());
    for (bi, (sub_circuit, _, _)) in circuit.partition_subcircuits(blocks).iter().enumerate() {
        let block_seed = seed.wrapping_add(bi as u64 * 0x1234_5678);
        block_samplers.push(compile_measurements(sub_circuit, block_seed)?);
    }

    let mut meas_map: Vec<(usize, usize)> = Vec::with_capacity(num_global_measurements);
    let mut block_meas_count: Vec<usize> = vec![0; blocks.len()];
    for inst in &circuit.instructions {
        if let Instruction::Measure { qubit, .. } = inst {
            let bi = qubit_to_block[*qubit];
            let local_idx = block_meas_count[bi];
            block_meas_count[bi] += 1;
            meas_map.push((bi, local_idx));
        }
    }

    let m_words = num_global_measurements.div_ceil(64);
    let total_rank: usize = block_samplers.iter().map(|s| s.rank).sum();
    let mut ref_bits_packed: Vec<u64> = vec![0u64; num_global_measurements.div_ceil(64)];

    for (gi, &(bi, li)) in meas_map.iter().enumerate() {
        let src = &block_samplers[bi].ref_bits_packed;
        let bit = (src[li / 64] >> (li % 64)) & 1;
        if bit != 0 {
            ref_bits_packed[gi / 64] |= 1u64 << (gi % 64);
        }
    }

    let mut local_to_global: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for (gi, &(bi, _li)) in meas_map.iter().enumerate() {
        local_to_global[bi].push(gi);
    }

    let mut rank_offsets = Vec::with_capacity(block_samplers.len());
    let mut rank_prefix = 0usize;
    for sampler in &block_samplers {
        rank_offsets.push(rank_prefix);
        rank_prefix += sampler.rank;
    }
    debug_assert_eq!(rank_prefix, total_rank);

    let mut flip_rows: Vec<Vec<u64>> = Vec::with_capacity(total_rank);
    for (bi, sampler) in block_samplers.iter().enumerate() {
        let mapping = &local_to_global[bi];
        for local_row in &sampler.flip_rows {
            flip_rows.push(project_local_flip_row(local_row, mapping, m_words));
        }
    }

    let sparse = build_sparse_from_filtered_blocks(
        &block_samplers,
        &meas_map,
        &rank_offsets,
        num_global_measurements,
    );
    let parity_blocks =
        build_filtered_parity_blocks(block_samplers, local_to_global).map(|mut blocks| {
            blocks.direct_scatter = true;
            blocks
        });

    Ok(CompiledSampler::from_parts(
        flip_rows,
        ref_bits_packed,
        total_rank,
        num_global_measurements,
        Some(sparse),
        parity_blocks,
        seed,
    ))
}

/// Compile a Clifford measurement circuit plus detector parity metadata.
///
/// `detector_rows` and `observable_rows` contain measurement record indices
/// in circuit measurement order. Circuits that reuse a qubit after a reset, or
/// keep using a measured qubit without one, are rewritten to terminal
/// measurements before compiling.
pub fn compile_detector_sampler(
    circuit: &Circuit,
    detector_rows: Vec<Vec<usize>>,
    observable_rows: Vec<Vec<usize>>,
    seed: u64,
) -> Result<CompiledDetectorSampler> {
    let measurement_circuit = if circuit.has_resets() || !circuit.has_terminal_measurements_only() {
        defer_measure_reset_circuit(circuit)?
    } else {
        circuit.clone()
    };

    let num_measurements = measurement_circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Measure { .. }))
        .count();
    validate_parity_rows(&detector_rows, num_measurements)?;
    validate_parity_rows(&observable_rows, num_measurements)?;

    let measurement_sampler = compile_measurements(&measurement_circuit, seed)?;
    Ok(CompiledDetectorSampler {
        measurement_sampler,
        detector_rows,
        observable_rows,
        num_measurements,
    })
}

/// Compile a Clifford circuit's measurements into a [`CompiledSampler`].
///
/// Requires terminal measurements with no resets or classical conditionals.
pub fn compile_measurements(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    require_compilable(circuit)?;

    if circuit.num_qubits >= 4 {
        let blocks = circuit.independent_subsystems();
        if blocks.len() > 1 {
            let max_block = blocks.iter().map(|b| b.len()).max().unwrap_or(0);
            if max_block < circuit.num_qubits {
                return compile_measurements_filtered(circuit, &blocks, seed);
            }
        }
    }

    if circuit.num_qubits >= 64 {
        return compile_forward_unchecked(circuit, seed);
    }

    let measurement_rows = build_measurement_rows(circuit);
    let num_measurements = measurement_rows.len();

    if num_measurements == 0 {
        return Ok(CompiledSampler::empty(seed));
    }

    let n = circuit.num_qubits;

    let x_rows: Vec<Vec<u64>> = measurement_rows
        .iter()
        .map(|(p, _, _)| p.x.clone())
        .collect();
    let signs: Vec<bool> = measurement_rows.iter().map(|(_, _, s)| *s).collect();

    let mut x_copy = x_rows.clone();
    let (rank, pivot_cols) = gaussian_eliminate(&mut x_copy, n);

    let gate_count = circuit
        .instructions
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate { .. } | Instruction::Conditional { .. }
            )
        })
        .count();

    let ref_bits: Vec<bool> = if gate_count > 2 * num_measurements {
        let mini_outcomes = compute_reference_bits(&measurement_rows, n);
        mini_outcomes
            .iter()
            .zip(signs.iter())
            .map(|(&outcome, &sign)| outcome ^ sign)
            .collect()
    } else {
        use crate::backend::Backend;
        use crate::backend::stabilizer::StabilizerBackend;
        // Two measurements may share a classical bit, so each reads into a
        // private bit numbered by its record index.
        let first_measure = circuit
            .instructions
            .iter()
            .position(|inst| matches!(inst, Instruction::Measure { .. }))
            .unwrap_or(circuit.instructions.len());
        let records: Vec<Instruction> = circuit.instructions[first_measure..]
            .iter()
            .filter_map(|inst| match inst {
                Instruction::Measure { qubit, .. } => Some(*qubit),
                _ => None,
            })
            .enumerate()
            .map(|(record, qubit)| Instruction::Measure {
                qubit,
                classical_bit: record,
            })
            .collect();
        let mut stab = StabilizerBackend::new(seed);
        stab.init(circuit.num_qubits, num_measurements)?;
        stab.apply_instructions(&circuit.instructions[..first_measure])?;
        stab.apply_instructions(&records)?;
        stab.classical_results().to_vec()
    };

    let num_meas_words = num_measurements.div_ceil(64);
    let mut flip_rows: Vec<Vec<u64>> = vec![vec![0u64; num_meas_words]; rank];

    for (j, &pcol) in pivot_cols.iter().enumerate() {
        for (i, x_row) in x_rows.iter().enumerate() {
            if get_bit(x_row, pcol) {
                flip_rows[j][i / 64] |= 1u64 << (i % 64);
            }
        }
    }

    Ok(finish_sampler(
        flip_rows,
        rank,
        num_measurements,
        &ref_bits,
        seed,
    ))
}

/// Sample shots via the compiled (Heisenberg-picture) path.
///
/// Returns `Vec<Vec<bool>>`, inherently O(num_shots) memory.
/// For bounded-memory streaming at large shot counts, use
/// `compile_measurements` + `sample_chunked` / `sample_counts` directly.
///
/// Requires the same circuit subset as [`compile_measurements`], namely
/// terminal measurements with no resets or classical conditionals.
///
/// # Examples
///
/// ```
/// use prism_q::{CircuitBuilder, run_shots_compiled};
///
/// let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();
/// let result = run_shots_compiled(&circuit, 100, 42).expect("sampling failed");
/// let counts = result.counts();
/// // Bell state: only 00 and 11 occur.
/// assert_eq!(counts.len(), 2);
/// assert_eq!(counts.get(&vec![0b00]).unwrap() + counts.get(&vec![0b11]).unwrap(), 100);
/// ```
pub fn run_shots_compiled(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    let mut sampler = compile_measurements(circuit, seed)?;
    Ok(shots_result(&mut sampler, circuit, num_shots))
}

/// Like [`run_shots_compiled`] but routes BTS sampling through the GPU when the
/// circuit compiles to a flat sparse parity matrix and `num_shots` reaches the
/// GPU BTS threshold. The threshold defaults to
/// [`crate::gpu::BTS_MIN_SHOTS_DEFAULT`] and can be overridden with
/// `PRISM_GPU_BTS_MIN_SHOTS`. Requires the `gpu` feature and a working CUDA
/// context.
#[cfg(feature = "gpu")]
pub fn run_shots_compiled_with_gpu(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    context: std::sync::Arc<crate::gpu::GpuContext>,
) -> Result<ShotsResult> {
    let mut sampler = compile_measurements(circuit, seed)?.with_gpu(context);
    Ok(shots_result(&mut sampler, circuit, num_shots))
}

fn shots_result(sampler: &mut CompiledSampler, circuit: &Circuit, num_shots: usize) -> ShotsResult {
    let packed = sampler.sample_bulk_packed(num_shots);
    let shots = crate::sim::shots::packed_shots_to_classical_bits(
        &packed,
        &circuit.measurement_map(),
        circuit.num_classical_bits,
    );
    ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(
        crate::sim::RunMetadata::exact(crate::sim::ResolvedBackend::CompiledStabilizer)
            .with_engine(crate::sim::Engine::CompiledSampler),
    )
}
