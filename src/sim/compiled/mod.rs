mod accumulator;
mod bts;
pub(crate) mod parity;
mod propagation;
pub(crate) mod rng;
#[cfg(test)]
mod tests;

use std::hash::{Hash, Hasher};

use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
#[cfg(feature = "gpu")]
use crate::gpu::kernels::bts::GpuBtsCache;
use crate::sim::ShotsResult;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

use bts::{bts_batched, bts_single_pass, sample_bts_meas_major, BTS_BATCH_SHOTS};
use rng::{binomial_sample, Xoshiro256PlusPlus};

pub use accumulator::{
    default_chunk_size, optimal_chunk_size, CorrelatorAccumulator, HistogramAccumulator,
    MarginalsAccumulator, NullAccumulator, PauliExpectationAccumulator, ShotAccumulator,
};
pub use parity::ParityStats;
use parity::{build_parity_blocks_if_useful, build_xor_dag_if_useful, minimize_flip_row_weight};
pub(crate) use parity::{ParityBlock, ParityBlocks, SparseParity, XorDag};

pub(crate) use propagation::batch_propagate_backward;
pub(crate) use propagation::propagate_backward;
use propagation::{
    build_measurement_rows, colmajor_forward_sim, compute_reference_bits, rowmul_phase,
    rowmul_phase_into,
};

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
pub(super) fn get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
pub(super) fn set_bit(words: &mut [u64], qubit: usize, val: bool) {
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

struct FlipLut {
    data: Vec<u64>,
    m_words: usize,
    num_full_groups: usize,
    remainder_size: usize,
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

        Self {
            data,
            m_words,
            num_full_groups,
            remainder_size,
        }
    }

    #[inline(always)]
    fn lookup(&self, group: usize, byte: usize) -> &[u64] {
        let offset = (group * (1 << LUT_GROUP_SIZE) + byte) * self.m_words;
        &self.data[offset..offset + self.m_words]
    }
}

pub struct CompiledSampler {
    flip_rows: Vec<Vec<u64>>,
    ref_bits_packed: Vec<u64>,
    rank: usize,
    num_measurements: usize,
    rng: ChaCha8Rng,
    lut: Option<FlipLut>,
    sparse: Option<SparseParity>,
    xor_dag: Option<XorDag>,
    parity_blocks: Option<ParityBlocks>,
    #[cfg(feature = "gpu")]
    gpu_context: Option<std::sync::Arc<crate::gpu::GpuContext>>,
    #[cfg(feature = "gpu")]
    gpu_bts_cache: Option<GpuBtsCache>,
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
struct SendPtrU64(*mut u64);
#[cfg(feature = "parallel")]
unsafe impl Send for SendPtrU64 {}
#[cfg(feature = "parallel")]
unsafe impl Sync for SendPtrU64 {}
#[cfg(feature = "parallel")]
impl SendPtrU64 {
    #[inline(always)]
    unsafe fn copy_from_slice(self, dst_offset: usize, src: &[u64]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.0.add(dst_offset), src.len());
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
    block_samplers: &[CompiledSampler],
    local_to_global: &[Vec<usize>],
) -> Option<ParityBlocks> {
    let mut blocks = Vec::new();

    for (block_idx, sampler) in block_samplers.iter().enumerate() {
        let mapping = &local_to_global[block_idx];
        if mapping.is_empty() {
            continue;
        }

        if let Some(parity_blocks) = &sampler.parity_blocks {
            for block in &parity_blocks.blocks {
                let meas_indices = block
                    .meas_indices
                    .iter()
                    .map(|&local_m| mapping[local_m])
                    .collect();
                blocks.push(ParityBlock {
                    meas_indices,
                    sparse: block.sparse.clone(),
                    block_rank: block.block_rank,
                    ref_bits_packed: block.ref_bits_packed.clone(),
                });
            }
            continue;
        }

        let sparse = sampler
            .sparse
            .as_ref()
            .expect("filtered block samplers must retain sparse parity")
            .clone();
        blocks.push(ParityBlock {
            meas_indices: mapping.clone(),
            sparse,
            block_rank: sampler.rank,
            ref_bits_packed: sampler.ref_bits_packed.clone(),
        });
    }

    ParityBlocks::from_blocks_if_useful(blocks)
}

impl CompiledSampler {
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
            && self.xor_dag.is_none()
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

        if let Some(lut) = &self.lut {
            let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);
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
                                    let entry = lut.lookup(g, byte);
                                    let base = s * m_words;
                                    xor_words(&mut chunk[base..base + m_words], entry);
                                }
                            }
                        }
                    });
            } else {
                for tile_start in (0..num_shots).step_by(max_batch) {
                    let tile_end = (tile_start + max_batch).min(num_shots);
                    for g in 0..total_groups {
                        for s in tile_start..tile_end {
                            let byte = rand_buf[s * bytes_per_shot + g] as usize;
                            let entry = lut.lookup(g, byte);
                            let shot_base = s * m_words;
                            xor_words(&mut accum[shot_base..shot_base + m_words], entry);
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            {
                for tile_start in (0..num_shots).step_by(max_batch) {
                    let tile_end = (tile_start + max_batch).min(num_shots);
                    for g in 0..total_groups {
                        for s in tile_start..tile_end {
                            let byte = rand_buf[s * bytes_per_shot + g] as usize;
                            let entry = lut.lookup(g, byte);
                            let shot_base = s * m_words;
                            xor_words(&mut accum[shot_base..shot_base + m_words], entry);
                        }
                    }
                }
            }

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

    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        self.sample_bulk_packed(num_shots).to_shots()
    }

    /// Materialise packed shots directly on the GPU.
    ///
    /// This is available only when the sampler has a GPU context and the
    /// compiled circuit routes through the flat sparse BTS path. Use
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

        Ok(DevicePackedShots {
            context: ctx,
            data,
            num_shots,
            num_measurements: self.num_measurements,
            m_words: self.num_measurements.div_ceil(64),
            s_words: num_shots.div_ceil(64),
            layout: ShotLayout::MeasMajor,
            rank,
        })
    }

    /// GPU BTS sampling with on-device meas-major to shot-major bit-transpose.
    ///
    /// Returns `Some(Ok(data))` when the compiled circuit matches the GPU BTS
    /// path (flat sparse, no xor_dag, shot threshold crossed). `data` is in
    /// shot-major layout (`num_shots * m_words` u64s) so callers can skip the
    /// host `into_shot_major_data()` transpose for downstream
    /// shot-major consumers (noise apply, etc.). `None` signals "use the CPU
    /// path for this sampler/shot-count combination".
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
    #[allow(dead_code)]
    pub(crate) fn sample_into_raw(&mut self, accum: &mut [u64]) {
        self.sample_into(accum);
    }

    #[inline(always)]
    fn sample_into(&mut self, accum: &mut [u64]) {
        if self.rank == 0 {
            return;
        }

        if let Some(lut) = &self.lut {
            let mut rand_buf = 0u64;
            let mut rand_pos = 8usize;

            for g in 0..lut.num_full_groups {
                if rand_pos >= 8 {
                    rand_buf = self.rng.next_u64();
                    rand_pos = 0;
                }
                let byte = ((rand_buf >> (rand_pos * 8)) & 0xFF) as usize;
                rand_pos += 1;
                let entry = lut.lookup(g, byte);
                xor_words(accum, entry);
            }
            if lut.remainder_size > 0 {
                if rand_pos >= 8 {
                    rand_buf = self.rng.next_u64();
                }
                let mask = (1u64 << lut.remainder_size) - 1;
                let byte = (rand_buf & mask) as usize;
                let entry = lut.lookup(lut.num_full_groups, byte);
                xor_words(accum, entry);
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
            return Ok(PackedShots {
                data: Vec::new(),
                num_shots,
                num_measurements: self.num_measurements,
                m_words,
                s_words,
                layout: ShotLayout::ShotMajor,
            });
        }
        if self.rank == 0 {
            let mut data = vec![0u64; num_shots * m_words];
            for s in 0..num_shots {
                let base = s * m_words;
                data[base..base + m_words].copy_from_slice(&self.ref_bits_packed);
            }
            return Ok(PackedShots {
                data,
                num_shots,
                num_measurements: self.num_measurements,
                m_words,
                s_words,
                layout: ShotLayout::ShotMajor,
            });
        }

        if self.should_use_bts(num_shots) {
            return self.sample_bulk_packed_bts(num_shots, m_words, s_words, propagate_gpu_errors);
        }

        let (mut data, _) = self.sample_bulk_words_shot_major(num_shots);
        for s in 0..num_shots {
            let base = s * m_words;
            xor_words(&mut data[base..base + m_words], &self.ref_bits_packed);
        }
        Ok(PackedShots {
            data,
            num_shots,
            num_measurements: self.num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::ShotMajor,
        })
    }

    fn sample_bulk_packed_bts(
        &mut self,
        num_shots: usize,
        m_words: usize,
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
                            let mut block_chacha = ChaCha8Rng::seed_from_u64(seed);
                            let mut block_rng = Xoshiro256PlusPlus::from_chacha(&mut block_chacha);
                            let block_data = sample_bts_meas_major(
                                &block.sparse,
                                num_shots,
                                &block.ref_bits_packed,
                                &mut block_rng,
                                block.block_rank,
                            );

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
                        let mut block_chacha = ChaCha8Rng::seed_from_u64(seed);
                        let mut block_rng = Xoshiro256PlusPlus::from_chacha(&mut block_chacha);
                        let block_data = sample_bts_meas_major(
                            &block.sparse,
                            num_shots,
                            &block.ref_bits_packed,
                            &mut block_rng,
                            block.block_rank,
                        );
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
                            let mut block_chacha = ChaCha8Rng::seed_from_u64(seed);
                            let mut block_rng = Xoshiro256PlusPlus::from_chacha(&mut block_chacha);
                            let data = sample_bts_meas_major(
                                &block.sparse,
                                num_shots,
                                &block.ref_bits_packed,
                                &mut block_rng,
                                block.block_rank,
                            );
                            (data, block.meas_indices.as_slice())
                        })
                        .collect()
                };

                #[cfg(not(feature = "parallel"))]
                let block_results: Vec<(Vec<u64>, &[usize])> = pb
                    .blocks
                    .iter()
                    .zip(block_seeds.iter())
                    .map(|(block, &seed)| {
                        let mut block_chacha = ChaCha8Rng::seed_from_u64(seed);
                        let mut block_rng = Xoshiro256PlusPlus::from_chacha(&mut block_chacha);
                        let data = sample_bts_meas_major(
                            &block.sparse,
                            num_shots,
                            &block.ref_bits_packed,
                            &mut block_rng,
                            block.block_rank,
                        );
                        (data, block.meas_indices.as_slice())
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

            return Ok(PackedShots {
                data: meas_major,
                num_shots,
                num_measurements: num_meas,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            });
        }

        #[cfg(feature = "gpu")]
        if self.should_use_gpu_bts(num_shots) {
            match self.sample_bulk_packed_bts_gpu(num_shots, m_words, s_words) {
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
                self.xor_dag.as_ref(),
                num_shots,
                &self.ref_bits_packed,
                &mut fast_rng,
                self.rank,
            );
            return Ok(PackedShots {
                data,
                num_shots,
                num_measurements: num_meas,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            });
        }

        let data = bts_batched(
            sparse,
            self.xor_dag.as_ref(),
            num_shots,
            s_words,
            &self.ref_bits_packed,
            &mut fast_rng,
            self.rank,
        );
        Ok(PackedShots {
            data,
            num_shots,
            num_measurements: num_meas,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        })
    }

    #[cfg(feature = "gpu")]
    fn sample_bulk_packed_bts_gpu(
        &mut self,
        num_shots: usize,
        m_words: usize,
        s_words: usize,
    ) -> Result<PackedShots> {
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
        Ok(PackedShots {
            data,
            num_shots,
            num_measurements: self.num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        })
    }

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
        let mut remaining = total_shots;
        while remaining > 0 {
            let this_batch = remaining.min(chunk_size);
            let packed = self.sample_bulk_packed(this_batch);
            acc.accumulate(&packed);
            remaining -= this_batch;
        }
    }

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
        if self.rank > 0 && self.parity_blocks.is_none() {
            let num_outcomes = 1usize << self.rank;

            if self.rank <= MAX_RANK_FOR_MULTINOMIAL
                && total_shots >= num_outcomes * MIN_SHOTS_PER_OUTCOME_MULTINOMIAL
            {
                return self.sample_counts_multinomial(total_shots);
            }

            if self.rank <= MAX_RANK_FOR_RANK_SPACE
                && total_shots >= num_outcomes * MIN_SHOTS_PER_OUTCOME
            {
                return self.sample_counts_rank_space(total_shots);
            }
        }
        let mut acc = HistogramAccumulator::new();
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
        if self.rank == 0 {
            let mut counts = HashMap::new();
            counts.insert(self.ref_bits_packed[..m_words].to_vec(), total_shots as u64);
            return counts;
        }

        let rank = self.rank;
        let num_outcomes = 1usize << rank;
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);
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
                if let Some(lut) = &self.lut {
                    let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);
                    for g in 0..total_groups {
                        let byte = (key >> (g * 8)) & 0xFF;
                        let entry = lut.lookup(g, byte);
                        xor_words(&mut outcome, entry);
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

    fn sample_counts_rank_space(
        &mut self,
        total_shots: usize,
    ) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;

        let m_words = self.num_measurements.div_ceil(64);

        if total_shots == 0 || self.num_measurements == 0 {
            return HashMap::new();
        }
        if self.rank == 0 {
            let mut counts = HashMap::new();
            counts.insert(self.ref_bits_packed[..m_words].to_vec(), total_shots as u64);
            return counts;
        }

        let rank = self.rank;
        let num_outcomes = 1usize << rank;
        let mut rank_counts = vec![0u64; num_outcomes];
        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);

        if let Some(lut) = &self.lut {
            let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);
            let bytes_per_shot = total_groups;
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
            if let Some(lut) = &self.lut {
                let total_groups = lut.num_full_groups + usize::from(lut.remainder_size > 0);
                for g in 0..total_groups {
                    let byte = (key >> (g * 8)) & 0xFF;
                    let entry = lut.lookup(g, byte);
                    xor_words(&mut outcome, entry);
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
        let mut acc = MarginalsAccumulator::new(self.num_measurements);
        self.sample_chunked(total_shots, &mut acc);
        acc.marginals()
    }

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

        if num_events == 0 || num_shots == 0 || self.rank == 0 {
            return PackedShots {
                data: vec![0u64; num_events * s_words],
                num_shots,
                num_measurements: num_events,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            };
        }

        let det_weight = det_sparse.stats().total_weight;
        let meas_weight = sparse.stats().total_weight;

        if det_weight > meas_weight + num_events {
            let meas_packed = self.sample_bulk_packed(num_shots);
            let mut data = vec![0u64; num_events * s_words];
            for (e, &(m_a, m_b)) in pairs.iter().enumerate() {
                let src_a = &meas_packed.data[m_a * s_words..(m_a + 1) * s_words];
                let src_b = &meas_packed.data[m_b * s_words..(m_b + 1) * s_words];
                let dst = &mut data[e * s_words..(e + 1) * s_words];
                for (d, (&a, &b)) in dst.iter_mut().zip(src_a.iter().zip(src_b.iter())) {
                    *d = a ^ b;
                }
            }
            return PackedShots {
                data,
                num_shots,
                num_measurements: num_events,
                m_words,
                s_words,
                layout: ShotLayout::MeasMajor,
            };
        }

        let det_ref = vec![0u64; m_words];

        let mut fast_rng = Xoshiro256PlusPlus::from_chacha(&mut self.rng);

        let data = if num_shots > BTS_BATCH_SHOTS {
            bts_batched(
                &det_sparse,
                None,
                num_shots,
                s_words,
                &det_ref,
                &mut fast_rng,
                self.rank,
            )
        } else {
            sample_bts_meas_major(&det_sparse, num_shots, &det_ref, &mut fast_rng, self.rank)
        };

        PackedShots {
            data,
            num_shots,
            num_measurements: num_events,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        }
    }

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

    pub fn marginal_probabilities(&self) -> Vec<f64> {
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

    pub fn detection_event_report(&self, pairs: &[(usize, usize)]) -> String {
        let sparse = match &self.sparse {
            Some(s) => s,
            None => return "No sparse parity matrix available\n".to_string(),
        };
        let det_sparse = sparse.compile_detection_events(pairs);
        let meas_stats = sparse.stats();
        let det_stats = det_sparse.stats();

        let mut report = format!(
            "Detection events: {} pairs\n\
             Measurement parity: total_weight={}, mean={:.2}\n\
             Detection parity:   total_weight={}, mean={:.2}\n",
            pairs.len(),
            meas_stats.total_weight,
            meas_stats.mean_weight,
            det_stats.total_weight,
            det_stats.mean_weight,
        );

        if meas_stats.total_weight > 0 {
            let reduction = 1.0 - det_stats.total_weight as f64 / meas_stats.total_weight as f64;
            report.push_str(&format!(
                "Weight reduction: {:.1}% ({:.1}x less work)\n",
                reduction * 100.0,
                if det_stats.total_weight > 0 {
                    meas_stats.total_weight as f64 / det_stats.total_weight as f64
                } else {
                    f64::INFINITY
                },
            ));
        }

        let mut histogram = [0usize; 8];
        for m in 0..det_sparse.num_rows {
            let w = det_sparse.row_weight(m);
            histogram[w.min(7)] += 1;
        }
        report.push_str("Detection weight histogram: ");
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
        report
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShotLayout {
    ShotMajor,
    MeasMajor,
}

#[derive(Debug, Clone)]
pub struct PackedShots {
    data: Vec<u64>,
    num_shots: usize,
    num_measurements: usize,
    m_words: usize,
    s_words: usize,
    layout: ShotLayout,
}

impl PackedShots {
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

    pub fn s_words(&self) -> usize {
        self.s_words
    }

    pub fn m_words(&self) -> usize {
        self.m_words
    }

    pub fn shot_words(&self, shot: usize) -> &[u64] {
        assert!(
            self.layout == ShotLayout::ShotMajor,
            "shot_words requires ShotMajor layout"
        );
        let base = shot * self.m_words;
        &self.data[base..base + self.m_words]
    }

    pub fn meas_words(&self, m: usize) -> &[u64] {
        assert!(
            self.layout == ShotLayout::MeasMajor,
            "meas_words requires MeasMajor layout"
        );
        let base = m * self.s_words;
        &self.data[base..base + self.s_words]
    }

    pub fn from_shot_major(data: Vec<u64>, num_shots: usize, num_measurements: usize) -> Self {
        let m_words = num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::ShotMajor,
        }
    }

    pub fn from_meas_major(data: Vec<u64>, num_shots: usize, num_measurements: usize) -> Self {
        let m_words = num_measurements.div_ceil(64);
        let s_words = num_shots.div_ceil(64);
        Self {
            data,
            num_shots,
            num_measurements,
            m_words,
            s_words,
            layout: ShotLayout::MeasMajor,
        }
    }

    pub fn raw_data(&self) -> &[u64] {
        &self.data
    }

    pub fn into_data(self) -> Vec<u64> {
        self.data
    }

    #[allow(dead_code)]
    pub(crate) fn into_shot_major_data(self) -> Vec<u64> {
        if self.layout == ShotLayout::ShotMajor {
            return self.data;
        }

        let mut shot_major = vec![0u64; self.num_shots * self.m_words];
        for batch_start in (0..self.num_shots).step_by(64) {
            let batch_end = (batch_start + 64).min(self.num_shots);
            let batch_len = batch_end - batch_start;
            let sw_base = batch_start / 64;
            let bit_off = batch_start % 64;
            let batch_mask = if batch_len == 64 {
                u64::MAX
            } else {
                (1u64 << batch_len) - 1
            };

            for m in 0..self.num_measurements {
                let mword = m / 64;
                let mbit = m % 64;
                let meas_row = &self.data[m * self.s_words..(m + 1) * self.s_words];
                let mut bits = meas_row[sw_base] >> bit_off;
                if bit_off != 0 && sw_base + 1 < self.s_words {
                    bits |= meas_row[sw_base + 1] << (64 - bit_off);
                }
                bits &= batch_mask;

                while bits != 0 {
                    let shot_in_batch = bits.trailing_zeros() as usize;
                    let shot_base = (batch_start + shot_in_batch) * self.m_words;
                    shot_major[shot_base + mword] |= 1u64 << mbit;
                    bits &= bits - 1;
                }
            }
        }

        shot_major
    }

    pub fn to_shots(&self) -> Vec<Vec<bool>> {
        let mut shots = Vec::with_capacity(self.num_shots);
        for s in 0..self.num_shots {
            let mut shot = Vec::with_capacity(self.num_measurements);
            for m in 0..self.num_measurements {
                shot.push(self.get_bit(s, m));
            }
            shots.push(shot);
        }
        shots
    }

    pub fn counts(&self) -> std::collections::HashMap<Vec<u64>, u64> {
        if self.m_words > 8 {
            return self.counts_wide();
        }
        self.counts_packed()
            .into_iter()
            .map(|(k, v)| (k[..self.m_words].to_vec(), v))
            .collect()
    }

    fn counts_wide(&self) -> std::collections::HashMap<Vec<u64>, u64> {
        use std::collections::HashMap;

        let mut map = HashMap::new();
        let mw = self.m_words;

        match self.layout {
            ShotLayout::ShotMajor => {
                for s in 0..self.num_shots {
                    let base = s * mw;
                    *map.entry(self.data[base..base + mw].to_vec()).or_insert(0) += 1;
                }
            }
            ShotLayout::MeasMajor => {
                let batch_size = 64;
                let mut shot_buf = vec![0u64; batch_size * mw];
                for batch_start in (0..self.num_shots).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(self.num_shots);
                    let batch_len = batch_end - batch_start;
                    shot_buf[..batch_len * mw].fill(0);

                    let sw_base = batch_start / 64;
                    let bit_off = batch_start % 64;

                    for m in 0..self.num_measurements {
                        let mword = m / 64;
                        let mbit = m % 64;
                        let meas_row = &self.data[m * self.s_words..];
                        let word = meas_row[sw_base];
                        let shifted = word >> bit_off;
                        for s in 0..batch_len {
                            if (shifted >> s) & 1 != 0 {
                                shot_buf[s * mw + mword] |= 1u64 << mbit;
                            }
                        }
                    }

                    for s in 0..batch_len {
                        let base = s * mw;
                        *map.entry(shot_buf[base..base + mw].to_vec()).or_insert(0) += 1;
                    }
                }
            }
        }

        map
    }

    fn counts_packed(&self) -> std::collections::HashMap<[u64; 8], u64> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let mw = self.m_words;
        debug_assert!(mw <= 8);

        match self.layout {
            ShotLayout::ShotMajor => {
                for s in 0..self.num_shots {
                    let base = s * mw;
                    let mut key = [0u64; 8];
                    key[..mw].copy_from_slice(&self.data[base..base + mw]);
                    *map.entry(key).or_insert(0) += 1;
                }
            }
            ShotLayout::MeasMajor => {
                let batch_size = 64;
                let mut shot_buf = vec![0u64; batch_size * mw];
                for batch_start in (0..self.num_shots).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(self.num_shots);
                    let batch_len = batch_end - batch_start;
                    shot_buf[..batch_len * mw].fill(0);

                    let sw_base = batch_start / 64;
                    let bit_off = batch_start % 64;

                    for m in 0..self.num_measurements {
                        let mword = m / 64;
                        let mbit = m % 64;
                        let meas_row = &self.data[m * self.s_words..];
                        let word = meas_row[sw_base];
                        let shifted = word >> bit_off;
                        for s in 0..batch_len {
                            if (shifted >> s) & 1 != 0 {
                                shot_buf[s * mw + mword] |= 1u64 << mbit;
                            }
                        }
                    }

                    for s in 0..batch_len {
                        let base = s * mw;
                        let mut key = [0u64; 8];
                        key[..mw].copy_from_slice(&shot_buf[base..base + mw]);
                        *map.entry(key).or_insert(0) += 1;
                    }
                }
            }
        }
        map
    }
}

/// Device-resident packed shots emitted by the GPU BTS path.
///
/// The payload stays in measurement-major layout on the device until an
/// explicit host copy via [`Self::to_host`]. Marginals and exact counts can be
/// reduced first so higher-level workflows do not have to transfer the full
/// shot matrix.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct DevicePackedShots {
    context: std::sync::Arc<crate::gpu::GpuContext>,
    data: crate::gpu::GpuBuffer<u64>,
    num_shots: usize,
    num_measurements: usize,
    m_words: usize,
    s_words: usize,
    layout: ShotLayout,
    rank: usize,
}

#[cfg(feature = "gpu")]
impl DevicePackedShots {
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    pub fn layout(&self) -> ShotLayout {
        self.layout
    }

    pub(crate) fn context(&self) -> &std::sync::Arc<crate::gpu::GpuContext> {
        &self.context
    }

    pub fn m_words(&self) -> usize {
        self.m_words
    }

    pub fn s_words(&self) -> usize {
        self.s_words
    }

    pub(crate) fn data_mut(&mut self) -> &mut crate::gpu::GpuBuffer<u64> {
        &mut self.data
    }

    /// Copy the full packed payload back to host memory.
    pub fn to_host(&self) -> Result<PackedShots> {
        let len = match self.layout {
            ShotLayout::ShotMajor => self.num_shots * self.m_words,
            ShotLayout::MeasMajor => self.num_measurements * self.s_words,
        };
        let mut host = vec![0u64; len];
        if len > 0 {
            self.data
                .copy_to_host(self.context.device(), &mut host)
                .map_err(|e| PrismError::BackendUnsupported {
                    backend: "gpu".to_string(),
                    operation: format!("copy device packed shots to host: {e}"),
                })?;
        }

        Ok(match self.layout {
            ShotLayout::ShotMajor => {
                PackedShots::from_shot_major(host, self.num_shots, self.num_measurements)
            }
            ShotLayout::MeasMajor => {
                PackedShots::from_meas_major(host, self.num_shots, self.num_measurements)
            }
        })
    }

    /// Return per-measurement one-counts without copying the full shot matrix.
    pub fn marginal_counts(&self) -> Result<Vec<u64>> {
        match self.layout {
            ShotLayout::MeasMajor => crate::gpu::kernels::bts::count_meas_major_marginals(
                &self.context,
                &self.data,
                self.num_measurements,
                self.num_shots,
                self.s_words,
            ),
            ShotLayout::ShotMajor => Ok(self.to_host()?.counts().into_iter().fold(
                vec![0u64; self.num_measurements],
                |mut acc, (key, count)| {
                    for m in 0..self.num_measurements {
                        if (key[m / 64] >> (m % 64)) & 1 != 0 {
                            acc[m] += count;
                        }
                    }
                    acc
                },
            )),
        }
    }

    /// Return per-measurement marginal probabilities.
    pub fn marginals(&self) -> Result<Vec<f64>> {
        if self.num_shots == 0 {
            return Ok(vec![0.0; self.num_measurements]);
        }
        Ok(self
            .marginal_counts()?
            .into_iter()
            .map(|count| count as f64 / self.num_shots as f64)
            .collect())
    }

    /// Return exact counts, using a GPU reduction when it reduces transfer.
    pub fn counts(&self) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        self.counts_with_rank_hint(self.rank)
    }

    pub(crate) fn counts_with_rank_hint(
        &self,
        rank_hint: usize,
    ) -> Result<std::collections::HashMap<Vec<u64>, u64>> {
        match self.layout {
            ShotLayout::MeasMajor => {
                if let Some(counts) = crate::gpu::kernels::bts::try_count_meas_major(
                    &self.context,
                    &self.data,
                    self.num_measurements,
                    self.num_shots,
                    self.m_words,
                    self.s_words,
                    rank_hint,
                )? {
                    return Ok(counts);
                }
            }
            ShotLayout::ShotMajor => {
                let raw_transfer_bytes = self
                    .num_shots
                    .saturating_mul(self.m_words)
                    .saturating_mul(std::mem::size_of::<u64>());
                if let Some(counts) = crate::gpu::kernels::bts::try_count_shot_major(
                    &self.context,
                    &self.data,
                    self.num_shots,
                    self.m_words,
                    rank_hint,
                    raw_transfer_bytes,
                )? {
                    return Ok(counts);
                }
            }
        }
        Ok(self.to_host()?.counts())
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn xor_words_avx2(dst: &mut [u64], src: &[u64]) {
    use std::arch::x86_64::*;
    let len = dst.len().min(src.len());
    let chunks = len / 4;
    let dp = dst.as_mut_ptr() as *mut __m256i;
    let sp = src.as_ptr() as *const __m256i;
    for i in 0..chunks {
        let d = _mm256_loadu_si256(dp.add(i));
        let s = _mm256_loadu_si256(sp.add(i));
        _mm256_storeu_si256(dp.add(i), _mm256_xor_si256(d, s));
    }
    let tail = chunks * 4;
    for i in tail..len {
        *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn xor_words_neon(dst: &mut [u64], src: &[u64]) {
    use std::arch::aarch64::*;
    let len = dst.len().min(src.len());
    let chunks = len / 2;
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    for i in 0..chunks {
        let off = i * 2;
        let d = vld1q_u64(dp.add(off));
        let s = vld1q_u64(sp.add(off));
        vst1q_u64(dp.add(off), veorq_u64(d, s));
    }
    let tail = chunks * 2;
    for i in tail..len {
        *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
    }
}

#[inline(always)]
pub(crate) fn xor_words(dst: &mut [u64], src: &[u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && dst.len() >= 4 {
            // SAFETY: AVX2 detected, pointers are valid u64 slices
            unsafe {
                xor_words_avx2(dst, src);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if dst.len() >= 2 {
            // SAFETY: NEON is baseline on aarch64, pointers are valid u64 slices
            unsafe {
                xor_words_neon(dst, src);
            }
            return;
        }
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d ^= s;
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
const MAX_RANK_FOR_RANK_SPACE: usize = 20;
const MIN_SHOTS_PER_OUTCOME: usize = 4;
const MAX_LUT_ALLOC_BYTES: u64 = 256 * 1024 * 1024;

pub fn compile_forward(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    if !circuit.is_clifford_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "circuit contains non-Clifford gates".to_string(),
        });
    }

    let measurements: Vec<(usize, usize)> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((*qubit, *classical_bit)),
            _ => None,
        })
        .collect();

    let num_measurements = measurements.len();
    if num_measurements == 0 {
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            #[cfg(feature = "gpu")]
            gpu_bts_cache: None,
        });
    }

    let n = circuit.num_qubits;

    let (mut xz, mut phase, nw) = colmajor_forward_sim(n, &circuit.instructions)?;
    let stride = 2 * nw;
    let m = num_measurements;
    let m_words = m.div_ceil(64);

    let rank_words = m_words;
    let total_rows = 2 * n;
    let mut gen_dep: Vec<Vec<u64>> = vec![vec![0u64; rank_words]; total_rows + 1];
    let mut ref_bits: Vec<bool> = vec![false; m];
    let mut rank = 0usize;

    let mut flip_rows: Vec<Vec<u64>> = Vec::with_capacity(m);
    let mut p_data: Vec<u64> = vec![0u64; stride];
    let mut p_dep: Vec<u64> = vec![0u64; rank_words];
    let mut scratch: Vec<u64> = vec![0u64; stride];
    let scratch_idx = total_rows;

    for (meas_idx, &(qubit, _)) in measurements.iter().enumerate() {
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);

        let mut p: Option<usize> = None;
        for i in n..2 * n {
            if xz[i * stride + word] & bit_mask != 0 {
                p = Some(i);
                break;
            }
        }

        if let Some(p_row) = p {
            // Random measurement, this is the k-th random degree of freedom
            let k = rank;
            rank += 1;
            flip_rows.push(vec![0u64; m_words]);

            flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);

            let p_base = p_row * stride;
            p_data.copy_from_slice(&xz[p_base..p_base + stride]);
            let p_phase = phase[p_row];
            p_dep.copy_from_slice(&gen_dep[p_row][..rank_words]);

            for r in 0..total_rows {
                if r == p_row {
                    continue;
                }
                if xz[r * stride + word] & bit_mask == 0 {
                    continue;
                }

                let r_base = r * stride;
                phase[r] = rowmul_phase(&p_data, &mut xz, r_base, nw, p_phase, phase[r]);
                xor_words(&mut gen_dep[r][..rank_words], &p_dep[..rank_words]);
            }

            let dest_idx = p_row - n;
            let dest_base = dest_idx * stride;
            xz.copy_within(p_row * stride..p_row * stride + stride, dest_base);
            phase[dest_idx] = p_phase;
            gen_dep[dest_idx][..rank_words].copy_from_slice(&p_dep);

            let p_base = p_row * stride;
            xz[p_base..p_base + stride].fill(0);
            xz[p_base + nw + word] |= bit_mask;
            phase[p_row] = false;

            gen_dep[p_row][..rank_words].fill(0);
            gen_dep[p_row][k / 64] |= 1u64 << (k % 64);

            ref_bits[meas_idx] = false;
        } else {
            scratch[..stride].fill(0);
            let mut scratch_phase = false;
            gen_dep[scratch_idx][..rank_words].fill(0);

            for g in 0..n {
                let d_base = g * stride;
                if xz[d_base + word] & bit_mask == 0 {
                    continue;
                }

                let s_base = (g + n) * stride;
                let s_phase = phase[g + n];
                scratch_phase =
                    rowmul_phase_into(&xz, s_base, &mut scratch, nw, s_phase, scratch_phase);

                let (lo, hi) = gen_dep.split_at_mut(scratch_idx);
                for (dst, &src) in hi[0][..rank_words].iter_mut().zip(&lo[g + n][..rank_words]) {
                    *dst ^= src;
                }
            }

            ref_bits[meas_idx] = scratch_phase;

            for (w, &dep_word) in gen_dep[scratch_idx][..rank_words].iter().enumerate() {
                let mut bits = dep_word;
                while bits != 0 {
                    let bit_pos = bits.trailing_zeros() as usize;
                    let k = w * 64 + bit_pos;
                    if k < rank {
                        flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);
                    }
                    bits &= bits - 1;
                }
            }
        }
    }

    let num_meas_words = m_words;
    minimize_flip_row_weight(&mut flip_rows);

    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_measurements);
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let ref_bits_packed = pack_bools(&ref_bits);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, rank, &ref_bits_packed);

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
        #[cfg(feature = "gpu")]
        gpu_context: None,
        #[cfg(feature = "gpu")]
        gpu_bts_cache: None,
    })
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
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            #[cfg(feature = "gpu")]
            gpu_bts_cache: None,
        });
    }

    let mut qubit_to_block: Vec<usize> = vec![0; circuit.num_qubits];
    for (bi, block) in blocks.iter().enumerate() {
        for &q in block {
            qubit_to_block[q] = bi;
        }
    }

    let mut block_samplers: Vec<CompiledSampler> = Vec::with_capacity(blocks.len());
    for (bi, block) in blocks.iter().enumerate() {
        let (sub_circuit, _qubit_map, _classical_map) = circuit.extract_subcircuit(block);
        let block_seed = seed.wrapping_add(bi as u64 * 0x1234_5678);
        block_samplers.push(compile_measurements(&sub_circuit, block_seed)?);
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

    let lut = if total_rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, m_words))
    } else {
        None
    };

    let sparse = build_sparse_from_filtered_blocks(
        &block_samplers,
        &meas_map,
        &rank_offsets,
        num_global_measurements,
    );
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let parity_blocks =
        build_filtered_parity_blocks(&block_samplers, &local_to_global).map(|mut blocks| {
            blocks.direct_scatter = true;
            blocks
        });

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank: total_rank,
        num_measurements: num_global_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
        #[cfg(feature = "gpu")]
        gpu_context: None,
        #[cfg(feature = "gpu")]
        gpu_bts_cache: None,
    })
}

/// Compile a Clifford circuit's measurements into a fast sampler.
///
/// Selects forward (SGI stabilizer + dependency tracking) or backward (Pauli
/// propagation + Gaussian elimination) based on circuit depth. Forward wins
/// for deep circuits (gate_count >= 5×measurements).
/// Requires terminal measurements and does not support reset or conditional
/// operations on measured data.
pub fn compile_measurements(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    if !circuit.is_clifford_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "circuit contains non-Clifford gates".to_string(),
        });
    }

    let has_measurements = circuit
        .instructions
        .iter()
        .any(|inst| matches!(inst, Instruction::Measure { .. }));
    if has_measurements && circuit.has_resets() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "compiled measurement sampling does not support reset instructions".to_string(),
        });
    }
    if has_measurements && !circuit.has_terminal_measurements_only() {
        return Err(PrismError::IncompatibleBackend {
            backend: "CompiledSampler".to_string(),
            reason: "compiled measurement sampling requires terminal measurements and does not support classical conditionals".to_string(),
        });
    }

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
        return compile_forward(circuit, seed);
    }

    let measurement_rows = build_measurement_rows(circuit);
    let num_measurements = measurement_rows.len();

    if num_measurements == 0 {
        return Ok(CompiledSampler {
            flip_rows: Vec::new(),
            ref_bits_packed: Vec::new(),
            rank: 0,
            num_measurements: 0,
            rng: ChaCha8Rng::seed_from_u64(seed),
            lut: None,
            sparse: None,
            xor_dag: None,
            parity_blocks: None,
            #[cfg(feature = "gpu")]
            gpu_context: None,
            #[cfg(feature = "gpu")]
            gpu_bts_cache: None,
        });
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
        use crate::backend::stabilizer::StabilizerBackend;
        use crate::backend::Backend;
        let mut stab = StabilizerBackend::new(seed);
        stab.init(circuit.num_qubits, circuit.num_classical_bits)?;
        stab.apply_instructions(&circuit.instructions)?;
        let ref_classical = stab.classical_results().to_vec();
        let classical_bit_order: Vec<usize> = measurement_rows.iter().map(|(_, c, _)| *c).collect();
        classical_bit_order
            .iter()
            .map(|&cbit| {
                if cbit < ref_classical.len() {
                    ref_classical[cbit]
                } else {
                    false
                }
            })
            .collect()
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

    minimize_flip_row_weight(&mut flip_rows);

    let lut = if rank >= LUT_MIN_RANK {
        Some(FlipLut::build(&flip_rows, num_meas_words))
    } else {
        None
    };

    let sparse = SparseParity::from_flip_rows(&flip_rows, num_measurements);
    let xor_dag = build_xor_dag_if_useful(&sparse);
    let ref_bits_packed = pack_bools(&ref_bits);
    let parity_blocks = build_parity_blocks_if_useful(&sparse, rank, &ref_bits_packed);

    Ok(CompiledSampler {
        flip_rows,
        ref_bits_packed,
        rank,
        num_measurements,
        rng: ChaCha8Rng::seed_from_u64(seed),
        lut,
        sparse: Some(sparse),
        xor_dag,
        parity_blocks,
        #[cfg(feature = "gpu")]
        gpu_context: None,
        #[cfg(feature = "gpu")]
        gpu_bts_cache: None,
    })
}

/// Sample shots via the compiled (Heisenberg-picture) path.
///
/// Returns `Vec<Vec<bool>>`, inherently O(num_shots) memory.
/// For bounded-memory streaming at large shot counts, use
/// `compile_measurements` + `sample_chunked` / `sample_counts` directly.
///
/// Requires the same circuit subset as [`compile_measurements`], namely
/// terminal measurements with no resets or classical conditionals.
pub fn run_shots_compiled(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    let mut sampler = compile_measurements(circuit, seed)?;
    let packed = sampler.sample_bulk_packed(num_shots);
    Ok(ShotsResult {
        shots: packed.to_shots(),
        num_classical_bits: circuit.num_classical_bits,
    })
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
    let packed = sampler.sample_bulk_packed(num_shots);
    Ok(ShotsResult {
        shots: packed.to_shots(),
        num_classical_bits: circuit.num_classical_bits,
    })
}
