//! Block-triangular sampling on the GPU: a device port of the CPU `sample_bts_meas_major`
//! fed by a host-generated random-bits pool, so both paths share one RNG contract, plus
//! device noise application, shot-major transpose, and outcome counting.
//!
//! The sampling kernel reads the precomputed CSR parity matrix and runs one thread per
//! (measurement, 64-shot batch) pair, emitting packed outcomes in measurement-major layout.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::Result;
use crate::gpu::{GpuBuffer, GpuDevice};
use crate::sim::compiled::parity::SparseParity;
use crate::sim::compiled::rng::Xoshiro256PlusPlus;
use crate::sim::compiled::shot_tail_mask;

use super::super::GpuContext;
use super::{
    div_ceil_grid, driver_err, launch_counted, linear_cfg, require_i32, require_u32, stream_and_fn,
};

const SAMPLE_BLOCK_SIZE: u32 = 128;
/// Columns of random bits one `bts_fill_random` thread draws; matches `BTS_FILL_COLS`.
const FILL_COLS: usize = 32;
const NOISE_BLOCK_SIZE: u32 = 128;
const POPCOUNT_BLOCK_SIZE: u32 = 256;
const COUNT_MEAS_BATCH_SIZE: u32 = 64;
const COUNT_HASH_BLOCK_SIZE: u32 = 256;
const COUNT_COMPACT_BLOCK_SIZE: u32 = 256;

/// Maximum packed measurement words per shot key for the device-side count
/// kernels; wider keys fall back to host counting.
pub(crate) const GPU_COUNTS_MAX_WORDS: usize = 8;

/// Shots per device chunk, which bounds peak memory at any shot count. Matches the CPU
/// `BTS_BATCH_SHOTS` boundary.
pub(crate) const CHUNK_SHOTS: usize = 65_536;
/// Largest sampled shot matrix, in bytes, that [`launch_bts_sample`] holds whole on the
/// device and reads back in one copy; past it the chunks come back one at a time.
const SINGLE_READBACK_BYTES: usize = 512 << 20;

const KERNEL_SOURCE: &str = include_str!("bts.cu");

pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE.to_string()
}

/// Batch count rounded up to a power of two in `[32, NOISE_BLOCK_SIZE]`, so small tiles
/// do not leave half the block idle.
fn noise_block_threads(chunk_s_words: usize) -> u32 {
    let desired = chunk_s_words.max(1).next_power_of_two();
    desired.clamp(32, NOISE_BLOCK_SIZE as usize) as u32
}

/// Reusable device and host storage for GPU BTS sampling.
///
/// The sparse parity CSR arrays and packed reference bits are uploaded once per
/// compiled sampler. `random_bits` and the chunk output buffer grow to the
/// largest chunk requested, then stay resident across repeated sampling calls.
pub(crate) struct GpuBtsCache {
    num_meas: usize,
    col_indices_dev: GpuBuffer<u32>,
    row_offsets_dev: GpuBuffer<u32>,
    ref_bits_dev: GpuBuffer<u64>,
    random_bits_dev: GpuBuffer<u64>,
    chunk_output_dev: GpuBuffer<u64>,
    chunk_output_host: Vec<u64>,
}

impl GpuBtsCache {
    pub(crate) fn new(ctx: &GpuContext, sparse: &SparseParity, ref_bits: &[u64]) -> Result<Self> {
        let device = ctx.device();
        let mut col_indices_dev =
            GpuBuffer::<u32>::alloc_zeros(device, sparse.col_indices.len().max(1))?;
        let mut row_offsets_dev =
            GpuBuffer::<u32>::alloc_zeros(device, sparse.row_offsets.len().max(1))?;
        let mut ref_bits_dev = GpuBuffer::<u64>::alloc_zeros(device, ref_bits.len().max(1))?;

        if !sparse.col_indices.is_empty() {
            col_indices_dev.copy_from_host(device, &sparse.col_indices)?;
        }
        row_offsets_dev.copy_from_host(device, &sparse.row_offsets)?;
        if !ref_bits.is_empty() {
            ref_bits_dev.copy_from_host(device, ref_bits)?;
        }

        Ok(Self {
            num_meas: sparse.num_rows,
            col_indices_dev,
            row_offsets_dev,
            ref_bits_dev,
            random_bits_dev: GpuBuffer::<u64>::alloc_zeros(device, 1)?,
            chunk_output_dev: GpuBuffer::<u64>::alloc_zeros(device, 1)?,
            chunk_output_host: Vec::new(),
        })
    }

    fn ensure_chunk_capacity(
        &mut self,
        ctx: &GpuContext,
        random_len: usize,
        output_len: usize,
    ) -> Result<()> {
        let device = ctx.device();
        let random_needed = random_len.max(1);
        if self.random_bits_dev.len() < random_needed {
            self.random_bits_dev = GpuBuffer::<u64>::alloc_zeros(device, random_needed)?;
        }
        if self.chunk_output_dev.len() < output_len.max(1) {
            self.chunk_output_dev = GpuBuffer::<u64>::alloc_zeros(device, output_len.max(1))?;
        }
        if self.chunk_output_host.len() < output_len.max(1) {
            self.chunk_output_host.resize(output_len.max(1), 0);
        }
        Ok(())
    }

    /// Fill the chunk's random bits on the device from `master_seed`, the chunk starting
    /// at absolute 64-shot batch `batch_offset`, and launch `bts_sample_meas_major` into
    /// `out` (the chunk output buffer when `None`), laid out as `num_meas` rows of
    /// `out_stride_words` words with the chunk starting at word `out_word_offset` of each
    /// row.
    #[allow(clippy::too_many_arguments)]
    fn launch_sample_chunk(
        &mut self,
        device: &GpuDevice,
        stream: &Arc<CudaStream>,
        func: &CudaFunction,
        rank: usize,
        chunk_shots: usize,
        out_stride_words: usize,
        out_word_offset: usize,
        out: Option<&mut GpuBuffer<u64>>,
        master_seed: u64,
        batch_offset: u64,
    ) -> Result<()> {
        let num_meas = self.num_meas;
        let chunk_s_words = chunk_shots.div_ceil(64);
        let chunk_random_len = chunk_s_words * rank;
        {
            let fill = device.function("bts_fill_random")?;
            let rank_i = require_i32("bts_fill_random", "rank", rank)?;
            let chunk_s_words_i = require_i32("bts_fill_random", "chunk_s_words", chunk_s_words)?;
            let threads = chunk_s_words * rank.div_ceil(FILL_COLS);
            let blocks = div_ceil_grid("bts_fill_random", "threads", threads, SAMPLE_BLOCK_SIZE)?;
            let mut random_bits_view = self
                .random_bits_dev
                .raw_mut()
                .slice_mut(0..chunk_random_len.max(1));
            let mut builder = stream.launch_builder(&fill);
            builder
                .arg(&mut random_bits_view)
                .arg(&rank_i)
                .arg(&chunk_s_words_i)
                .arg(&master_seed)
                .arg(&batch_offset);
            // SAFETY: kernel signature matches the call shape. Each thread writes the
            // columns of one (batch, column block) pair, disjoint from every other thread,
            // inside the `chunk_s_words * rank` words the view covers.
            unsafe {
                launch_counted(
                    builder,
                    linear_cfg(SAMPLE_BLOCK_SIZE, blocks),
                    "bts_fill_random",
                    device,
                    0,
                )
            }?;
        }

        let num_meas_i = require_i32("bts_sample_meas_major", "num_meas", num_meas)?;
        let chunk_s_words_i = require_i32("bts_sample_meas_major", "chunk_s_words", chunk_s_words)?;
        let rank_i = require_i32("bts_sample_meas_major", "rank", rank)?;
        let out_stride_words_i = require_i32(
            "bts_sample_meas_major",
            "out_stride_words",
            out_stride_words,
        )?;
        let out_word_offset_i =
            require_i32("bts_sample_meas_major", "out_word_offset", out_word_offset)?;
        let tail_mask_u64 = shot_tail_mask(chunk_shots);
        let batch_blocks = div_ceil_grid(
            "bts_sample_meas_major",
            "chunk_s_words",
            chunk_s_words,
            SAMPLE_BLOCK_SIZE,
        )?;
        let num_meas_grid = require_u32("bts_sample_meas_major", "num_meas", num_meas)?;
        let cfg = LaunchConfig {
            grid_dim: (num_meas_grid, batch_blocks, 1),
            block_dim: (SAMPLE_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let out = out.unwrap_or(&mut self.chunk_output_dev);
        let mut out_view = out
            .raw_mut()
            .slice_mut(0..(num_meas * out_stride_words).max(1));
        let random_bits_dev = self.random_bits_dev.raw().slice(0..chunk_random_len.max(1));
        let mut builder = stream.launch_builder(func);
        builder
            .arg(self.col_indices_dev.raw())
            .arg(self.row_offsets_dev.raw())
            .arg(self.ref_bits_dev.raw())
            .arg(&random_bits_dev)
            .arg(&num_meas_i)
            .arg(&chunk_s_words_i)
            .arg(&rank_i)
            .arg(&out_stride_words_i)
            .arg(&out_word_offset_i)
            .arg(&tail_mask_u64)
            .arg(&mut out_view);
        // SAFETY: kernel signature matches the call shape. Each thread writes one unique
        // output word, addressed by measurement row and chunk-local batch index, inside
        // the `num_meas * out_stride_words` view. All input buffers are sized to cover
        // the access pattern above.
        unsafe { launch_counted(builder, cfg, "bts_sample_meas_major", device, 0) }?;
        Ok(())
    }
}

fn next_power_of_two_or_max(n: usize) -> usize {
    n.checked_next_power_of_two().unwrap_or(usize::MAX)
}

fn highest_power_of_two_at_most(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        1usize << (usize::BITS as usize - 1 - n.leading_zeros() as usize)
    }
}

fn counts_slot_bytes(m_words: usize) -> usize {
    std::mem::size_of::<u32>() + std::mem::size_of::<u64>() + m_words * std::mem::size_of::<u64>()
}

fn counts_outcome_bound(num_shots: usize, rank: usize) -> usize {
    if rank >= usize::BITS as usize {
        num_shots
    } else {
        num_shots.min(1usize << rank)
    }
}

fn count_raw_transfer_bytes(len_words: usize) -> usize {
    len_words.saturating_mul(std::mem::size_of::<u64>())
}

fn count_compact_transfer_bound_bytes(num_shots: usize, m_words: usize, rank: usize) -> usize {
    counts_outcome_bound(num_shots, rank)
        .saturating_mul(m_words + 1)
        .saturating_mul(std::mem::size_of::<u64>())
}

fn should_try_device_count(
    num_shots: usize,
    m_words: usize,
    rank: usize,
    raw_transfer_bytes: usize,
) -> bool {
    if raw_transfer_bytes == 0 {
        return false;
    }
    count_compact_transfer_bound_bytes(num_shots, m_words, rank) < raw_transfer_bytes
}

fn plan_count_table_slots(
    ctx: &GpuContext,
    num_shots: usize,
    m_words: usize,
    rank: usize,
) -> Result<Option<usize>> {
    if m_words == 0 || m_words > GPU_COUNTS_MAX_WORDS {
        return Ok(None);
    }

    let available = ctx.vram_available()?;
    let slot_bytes = counts_slot_bytes(m_words);
    let budget = available / 2;
    let max_slots_fit = highest_power_of_two_at_most(budget / slot_bytes);
    if max_slots_fit < 64 || max_slots_fit > u32::MAX as usize + 1 {
        return Ok(None);
    }

    let distinct_bound = counts_outcome_bound(num_shots, rank);
    let target_slots = next_power_of_two_or_max(distinct_bound.saturating_mul(2).max(64));
    let table_slots = target_slots.min(max_slots_fit);
    if table_slots < 64 || table_slots > i32::MAX as usize {
        return Ok(None);
    }

    Ok(Some(table_slots))
}

/// Sample `num_shots` BTS shots on the GPU with the `sample_bts_meas_major` contract:
/// measurement-major, `num_meas * num_shots.div_ceil(64)` words, and deterministic rows
/// (empty `row_cols`) carrying only the `ref_bits` flip.
///
/// Shots are sampled in `CHUNK_SHOTS` chunks from random bits drawn on the device. The
/// parity CSR and `ref_bits` stay resident in `cache`. A result up to
/// `SINGLE_READBACK_BYTES` comes back in one copy, a larger one a chunk at a time.
pub(crate) fn launch_bts_sample(
    ctx: &GpuContext,
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    num_shots: usize,
    cache: &mut GpuBtsCache,
) -> Result<Vec<u64>> {
    let num_meas = cache.num_meas;
    let s_words = num_shots.div_ceil(64);
    if num_meas == 0 || num_shots == 0 || rank == 0 {
        return Ok(vec![0u64; num_meas * s_words]);
    }

    if num_meas * s_words * std::mem::size_of::<u64>() <= SINGLE_READBACK_BYTES {
        let device_output = launch_bts_sample_device(ctx, rng, rank, num_shots, cache)?;
        let mut output = vec![0u64; num_meas * s_words];
        device_output.copy_to_host(ctx.device(), &mut output)?;
        return Ok(output);
    }

    let (stream, func) = stream_and_fn(ctx, "bts_sample_meas_major")?;
    let master_seed = rng.next_u64();

    let mut output = vec![0u64; num_meas * s_words];
    let mut shots_done = 0usize;
    while shots_done < num_shots {
        let chunk_shots = (num_shots - shots_done).min(CHUNK_SHOTS);
        let chunk_s_words = chunk_shots.div_ceil(64);
        let chunk_random_len = chunk_s_words * rank;
        let chunk_output_len = num_meas * chunk_s_words;
        cache.ensure_chunk_capacity(ctx, chunk_random_len, chunk_output_len)?;

        cache.launch_sample_chunk(
            ctx.device(),
            stream,
            &func,
            rank,
            chunk_shots,
            chunk_s_words,
            0,
            None,
            master_seed,
            (shots_done / 64) as u64,
        )?;

        stream
            .memcpy_dtoh(
                &cache
                    .chunk_output_dev
                    .raw()
                    .slice(0..chunk_output_len.max(1)),
                &mut cache.chunk_output_host[..chunk_output_len.max(1)],
            )
            .map_err(|e| driver_err("bts output dtoh", e))?;
        ctx.device().count_drain();

        let chunk_word_offset = shots_done / 64;
        for m in 0..num_meas {
            let dst_start = m * s_words + chunk_word_offset;
            let src_start = m * chunk_s_words;
            output[dst_start..dst_start + chunk_s_words]
                .copy_from_slice(&cache.chunk_output_host[src_start..src_start + chunk_s_words]);
        }

        shots_done += chunk_shots;
    }

    Ok(output)
}

/// Same chunked contract as [`launch_bts_sample`], but the meas-major result
/// (length `num_meas * s_words` u64s) stays resident on device to feed the
/// on-device transpose and count paths.
pub(crate) fn launch_bts_sample_device(
    ctx: &GpuContext,
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    num_shots: usize,
    cache: &mut GpuBtsCache,
) -> Result<GpuBuffer<u64>> {
    let num_meas = cache.num_meas;
    let s_words = num_shots.div_ceil(64);
    let output_len = num_meas * s_words;
    if num_meas == 0 || num_shots == 0 || rank == 0 {
        return GpuBuffer::<u64>::alloc_zeros(ctx.device(), output_len.max(1));
    }

    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "bts_sample_meas_major")?;
    let mut output_dev = GpuBuffer::<u64>::alloc_zeros(device, output_len.max(1))?;
    let master_seed = rng.next_u64();

    let mut shots_done = 0usize;
    while shots_done < num_shots {
        let chunk_shots = (num_shots - shots_done).min(CHUNK_SHOTS);
        let chunk_s_words = chunk_shots.div_ceil(64);
        let chunk_random_len = chunk_s_words * rank;
        cache.ensure_chunk_capacity(ctx, chunk_random_len, 1)?;

        cache.launch_sample_chunk(
            device,
            stream,
            &func,
            rank,
            chunk_shots,
            s_words,
            shots_done / 64,
            Some(&mut output_dev),
            master_seed,
            (shots_done / 64) as u64,
        )?;

        shots_done += chunk_shots;
    }

    Ok(output_dev)
}

/// Bit-transpose a device-resident meas-major shot matrix into shot-major
/// layout on device. Result length is `num_shots * m_words` u64s.
pub(crate) fn launch_bts_transpose_meas_to_shot(
    ctx: &GpuContext,
    meas_major: &GpuBuffer<u64>,
    num_meas: usize,
    num_shots: usize,
    s_words: usize,
    m_words: usize,
) -> Result<GpuBuffer<u64>> {
    let device = ctx.device();
    let shot_major_len = (num_shots * m_words).max(1);
    let mut shot_major = GpuBuffer::<u64>::alloc_zeros(device, shot_major_len)?;
    if num_shots == 0 || num_meas == 0 || m_words == 0 || s_words == 0 {
        return Ok(shot_major);
    }

    let stream = device.stream()?;
    let func = device.function("bts_transpose_meas_to_shot")?;
    let num_meas_i = require_i32("bts_transpose_meas_to_shot", "num_meas", num_meas)?;
    let num_shots_i = require_i32("bts_transpose_meas_to_shot", "num_shots", num_shots)?;
    let s_words_i = require_i32("bts_transpose_meas_to_shot", "s_words", s_words)?;
    let m_words_i = require_i32("bts_transpose_meas_to_shot", "m_words", m_words)?;
    let m_words_grid = require_u32("bts_transpose_meas_to_shot", "m_words", m_words)?;
    let s_words_grid = require_u32("bts_transpose_meas_to_shot", "s_words", s_words)?;

    for (batch_base, launch_batches) in super::grid_y_tiles(s_words_grid) {
        let cfg = LaunchConfig {
            grid_dim: (m_words_grid, launch_batches, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch_base_i = batch_base as i32;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(meas_major.raw())
            .arg(&num_meas_i)
            .arg(&num_shots_i)
            .arg(&s_words_i)
            .arg(&m_words_i)
            .arg(&batch_base_i)
            .arg(shot_major.raw_mut());
        // SAFETY: the launches together cover every (m_word, batch) tile exactly once;
        // each tile's 64 threads produce 64 disjoint shot-major u64s. Shared tile is
        // block-local.
        unsafe { launch_counted(builder, cfg, "bts_transpose_meas_to_shot", ctx.device(), 0) }?;
    }
    Ok(shot_major)
}

/// Target buffer and event CSR arguments of the host-mask noise kernel.
pub(crate) struct NoiseApplyBase<'a> {
    pub meas_major: &'a mut GpuBuffer<u64>,
    pub num_meas: usize,
    pub s_words: usize,
    pub word_offset: usize,
    pub chunk_s_words: usize,
    pub num_events: usize,
    pub x_row_offsets: &'a GpuBuffer<u32>,
    pub x_row_indices: &'a GpuBuffer<u32>,
    pub z_row_offsets: &'a GpuBuffer<u32>,
    pub z_row_indices: &'a GpuBuffer<u32>,
}

/// [`NoiseApplyBase`] plus per-event X/Z flip masks pre-generated on the host,
/// as opposed to the on-device generation of [`NoiseMaskGenerate`] and
/// [`NoiseRowApply`].
pub(crate) struct NoiseMaskApply<'a> {
    pub base: NoiseApplyBase<'a>,
    pub x_masks: &'a GpuBuffer<u64>,
    pub z_masks: &'a GpuBuffer<u64>,
}

/// Arguments for per-(event, 64-shot batch) device noise mask generation.
/// Writes packed X and Z masks into `x_masks` and `z_masks` in
/// `[event * chunk_s_words + batch]` layout with trailing bits on the final
/// batch word cleared by `tail_mask`.
pub(crate) struct NoiseMaskGenerate<'a> {
    pub x_masks: &'a mut GpuBuffer<u64>,
    pub z_masks: &'a mut GpuBuffer<u64>,
    pub event_thresholds: &'a GpuBuffer<u64>,
    pub num_events: usize,
    pub chunk_s_words: usize,
    pub tail_mask: u64,
    pub master_seed: u64,
    pub batch_offset: u64,
}

/// Arguments for the per-(row, batch) XOR accumulator kernel. Each thread
/// walks one row's entry list in the row-major event CSR (`row_event_offsets`
/// of length `num_meas + 1`, `row_event_entries` packed as `event << 2 | flag`
/// with bit 0 = X and bit 1 = Z), reads from `x_masks` and `z_masks`, and
/// XORs the accumulated mask once into `meas_major`.
pub(crate) struct NoiseRowApply<'a> {
    pub meas_major: &'a mut GpuBuffer<u64>,
    pub num_meas: usize,
    pub s_words: usize,
    pub word_offset: usize,
    pub chunk_s_words: usize,
    pub row_event_offsets: &'a GpuBuffer<u32>,
    pub row_event_entries: &'a GpuBuffer<u32>,
    pub x_masks: &'a GpuBuffer<u64>,
    pub z_masks: &'a GpuBuffer<u64>,
}

pub(crate) fn generate_noise_masks(ctx: &GpuContext, args: NoiseMaskGenerate<'_>) -> Result<()> {
    let NoiseMaskGenerate {
        x_masks,
        z_masks,
        event_thresholds,
        num_events,
        chunk_s_words,
        tail_mask,
        master_seed,
        batch_offset,
    } = args;
    if num_events == 0 || chunk_s_words == 0 {
        return Ok(());
    }

    let mask_len = num_events.checked_mul(chunk_s_words).ok_or_else(|| {
        super::launch_limit_err("bts_generate_noise_masks", "mask_len", usize::MAX, "usize")
    })?;
    let thresholds_len = num_events.checked_mul(3).ok_or_else(|| {
        super::launch_limit_err(
            "bts_generate_noise_masks",
            "event_thresholds",
            usize::MAX,
            "usize",
        )
    })?;
    assert!(
        x_masks.len() >= mask_len && z_masks.len() >= mask_len,
        "noise mask buffers ({}, {}) smaller than required {}",
        x_masks.len(),
        z_masks.len(),
        mask_len
    );
    assert!(
        event_thresholds.len() >= thresholds_len,
        "event_thresholds buffer ({}) smaller than required {}",
        event_thresholds.len(),
        thresholds_len
    );

    let (stream, func) = stream_and_fn(ctx, "bts_generate_noise_masks")?;
    let num_events_i = require_i32("bts_generate_noise_masks", "num_events", num_events)?;
    let chunk_s_words_i = require_i32("bts_generate_noise_masks", "chunk_s_words", chunk_s_words)?;
    let block_threads = noise_block_threads(chunk_s_words);
    let batch_blocks = div_ceil_grid(
        "bts_generate_noise_masks",
        "chunk_s_words",
        chunk_s_words,
        block_threads,
    )?;
    if batch_blocks > 65_535 {
        return Err(super::launch_limit_err(
            "bts_generate_noise_masks",
            "batch_blocks",
            batch_blocks as usize,
            "gridDim.y",
        ));
    }
    let num_events_grid = require_u32("bts_generate_noise_masks", "num_events", num_events)?;
    let cfg = LaunchConfig {
        grid_dim: (num_events_grid, batch_blocks, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut x_masks_view = x_masks.raw_mut().slice_mut(0..mask_len);
    let mut z_masks_view = z_masks.raw_mut().slice_mut(0..mask_len);
    let thresholds_view = event_thresholds.raw().slice(0..thresholds_len);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&mut x_masks_view)
        .arg(&mut z_masks_view)
        .arg(&thresholds_view)
        .arg(&num_events_i)
        .arg(&chunk_s_words_i)
        .arg(&tail_mask)
        .arg(&master_seed)
        .arg(&batch_offset);
    // SAFETY: kernel signature matches the call shape. Each thread writes one
    // unique (event, batch) index inside `x_masks_view` and `z_masks_view`
    // (`num_events * chunk_s_words`), and reads three `u64` thresholds inside
    // `thresholds_view` (`num_events * 3`).
    unsafe { launch_counted(builder, cfg, "bts_generate_noise_masks", ctx.device(), 0) }
}

pub(crate) fn apply_noise_row_major(ctx: &GpuContext, args: NoiseRowApply<'_>) -> Result<()> {
    let NoiseRowApply {
        meas_major,
        num_meas,
        s_words,
        word_offset,
        chunk_s_words,
        row_event_offsets,
        row_event_entries,
        x_masks,
        z_masks,
    } = args;
    if num_meas == 0 || s_words == 0 || chunk_s_words == 0 {
        return Ok(());
    }

    let end_word = word_offset
        .checked_add(chunk_s_words)
        .expect("noise word range overflow");
    assert!(
        end_word <= s_words,
        "noise chunk words ({word_offset} + {chunk_s_words}) exceed s_words ({s_words})"
    );
    let meas_len = num_meas.checked_mul(s_words).ok_or_else(|| {
        super::launch_limit_err("bts_apply_noise_row_major", "meas_len", usize::MAX, "usize")
    })?;
    assert!(
        meas_major.len() >= meas_len,
        "meas_major buffer ({}) smaller than required {}",
        meas_major.len(),
        meas_len
    );
    assert!(
        row_event_offsets.len() > num_meas,
        "row_event_offsets buffer ({}) smaller than required {}",
        row_event_offsets.len(),
        num_meas + 1
    );

    let (stream, func) = stream_and_fn(ctx, "bts_apply_noise_row_major")?;
    let num_meas_i = require_i32("bts_apply_noise_row_major", "num_meas", num_meas)?;
    let s_words_i = require_i32("bts_apply_noise_row_major", "s_words", s_words)?;
    let word_offset_i = require_i32("bts_apply_noise_row_major", "word_offset", word_offset)?;
    let chunk_s_words_i = require_i32("bts_apply_noise_row_major", "chunk_s_words", chunk_s_words)?;
    let block_threads = noise_block_threads(chunk_s_words);
    let batch_blocks = div_ceil_grid(
        "bts_apply_noise_row_major",
        "chunk_s_words",
        chunk_s_words,
        block_threads,
    )?;
    if batch_blocks > 65_535 {
        return Err(super::launch_limit_err(
            "bts_apply_noise_row_major",
            "batch_blocks",
            batch_blocks as usize,
            "gridDim.y",
        ));
    }
    let num_meas_grid = require_u32("bts_apply_noise_row_major", "num_meas", num_meas)?;
    let cfg = LaunchConfig {
        grid_dim: (num_meas_grid, batch_blocks, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut meas_view = meas_major.raw_mut().slice_mut(0..meas_len);
    let offsets_view = row_event_offsets.raw().slice(0..num_meas + 1);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&mut meas_view)
        .arg(&num_meas_i)
        .arg(&s_words_i)
        .arg(&word_offset_i)
        .arg(&offsets_view)
        .arg(row_event_entries.raw())
        .arg(x_masks.raw())
        .arg(z_masks.raw())
        .arg(&chunk_s_words_i);
    // SAFETY: kernel signature matches the call shape. Each thread owns one
    // unique `(row, batch)` word at `row * s_words + word_offset + batch` inside
    // `meas_view` (`num_meas * s_words`), and all CSR and mask reads stay within
    // the allocated buffers.
    unsafe { launch_counted(builder, cfg, "bts_apply_noise_row_major", ctx.device(), 0) }
}

/// One thread per (event, batch) pair, with atomic XOR on rows shared
/// between events.
pub(crate) fn apply_noise_masks_meas_major(
    ctx: &GpuContext,
    args: NoiseMaskApply<'_>,
) -> Result<()> {
    let NoiseMaskApply {
        base:
            NoiseApplyBase {
                meas_major,
                num_meas,
                s_words,
                word_offset,
                chunk_s_words,
                num_events,
                x_row_offsets,
                x_row_indices,
                z_row_offsets,
                z_row_indices,
            },
        x_masks,
        z_masks,
    } = args;
    if num_meas == 0 || s_words == 0 || chunk_s_words == 0 || num_events == 0 {
        return Ok(());
    }

    let (stream, func) = stream_and_fn(ctx, "bts_apply_noise_masks_meas_major")?;
    let num_meas_i = require_i32("bts_apply_noise_masks_meas_major", "num_meas", num_meas)?;
    let s_words_i = require_i32("bts_apply_noise_masks_meas_major", "s_words", s_words)?;
    let word_offset_i = require_i32(
        "bts_apply_noise_masks_meas_major",
        "word_offset",
        word_offset,
    )?;
    let chunk_s_words_i = require_i32(
        "bts_apply_noise_masks_meas_major",
        "chunk_s_words",
        chunk_s_words,
    )?;
    let num_events_i = require_i32("bts_apply_noise_masks_meas_major", "num_events", num_events)?;
    let block_threads = noise_block_threads(chunk_s_words);
    let batch_blocks = div_ceil_grid(
        "bts_apply_noise_masks_meas_major",
        "chunk_s_words",
        chunk_s_words,
        block_threads,
    )?;
    let num_events_grid =
        require_u32("bts_apply_noise_masks_meas_major", "num_events", num_events)?;
    let cfg = LaunchConfig {
        grid_dim: (num_events_grid, batch_blocks, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw_mut())
        .arg(&num_meas_i)
        .arg(&s_words_i)
        .arg(&word_offset_i)
        .arg(x_row_offsets.raw())
        .arg(x_row_indices.raw())
        .arg(z_row_offsets.raw())
        .arg(z_row_indices.raw())
        .arg(x_masks.raw())
        .arg(z_masks.raw())
        .arg(&chunk_s_words_i)
        .arg(&num_events_i);
    // SAFETY: each thread owns one (event, batch) pair and uses atomic XOR
    // updates when multiple events touch the same measurement row.
    unsafe {
        launch_counted(
            builder,
            cfg,
            "bts_apply_noise_masks_meas_major",
            ctx.device(),
            0,
        )
    }
}

/// Flip readout error into a measurement-major record of `num_shots` shots, one
/// thread per (entry, 64-shot batch). `readout` holds three `u64`s per entry: the
/// record row, then the `p01` and `p10` thresholds scaled to `2^64`. Rows must be
/// distinct and below the record's row count.
pub(crate) fn apply_readout_meas_major(
    ctx: &GpuContext,
    meas_major: &mut GpuBuffer<u64>,
    num_shots: usize,
    readout: &GpuBuffer<u64>,
    num_entries: usize,
    master_seed: u64,
) -> Result<()> {
    let s_words = num_shots.div_ceil(64);
    if s_words == 0 || num_entries == 0 {
        return Ok(());
    }

    let (stream, func) = stream_and_fn(ctx, "bts_apply_readout_meas_major")?;
    let s_words_i = require_i32("bts_apply_readout_meas_major", "s_words", s_words)?;
    let num_entries_i = require_i32("bts_apply_readout_meas_major", "num_entries", num_entries)?;
    let entries_grid = require_u32("bts_apply_readout_meas_major", "num_entries", num_entries)?;
    let block_threads = noise_block_threads(s_words);
    let batch_blocks = div_ceil_grid(
        "bts_apply_readout_meas_major",
        "s_words",
        s_words,
        block_threads,
    )?;
    let tail_mask = shot_tail_mask(num_shots);

    for (block_base, launch_blocks) in super::grid_y_tiles(batch_blocks) {
        let cfg = LaunchConfig {
            grid_dim: (entries_grid, launch_blocks, 1),
            block_dim: (block_threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(meas_major.raw_mut())
            .arg(&s_words_i)
            .arg(readout.raw())
            .arg(&num_entries_i)
            .arg(&block_base)
            .arg(&tail_mask)
            .arg(&master_seed);
        // SAFETY: signature matches the kernel. The launches cover every (entry,
        // batch) pair once and entries name distinct rows, so each thread owns one word.
        unsafe {
            launch_counted(
                builder,
                cfg,
                "bts_apply_readout_meas_major",
                ctx.device(),
                0,
            )
        }?;
    }
    Ok(())
}

/// Sample `num_shots` BTS shots in shot-major layout, `num_shots * m_words` words,
/// transposing on device instead of through the host `into_shot_major_data()`.
pub(crate) fn launch_bts_sample_shot_major_host(
    ctx: &GpuContext,
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    num_shots: usize,
    cache: &mut GpuBtsCache,
) -> Result<Vec<u64>> {
    let num_meas = cache.num_meas;
    let m_words = num_meas.div_ceil(64);
    let s_words = num_shots.div_ceil(64);
    let shot_major_len = num_shots * m_words;
    if num_meas == 0 || num_shots == 0 || rank == 0 {
        return Ok(vec![0u64; shot_major_len]);
    }

    let meas_major = launch_bts_sample_device(ctx, rng, rank, num_shots, cache)?;
    let shot_major =
        launch_bts_transpose_meas_to_shot(ctx, &meas_major, num_meas, num_shots, s_words, m_words)?;

    let mut host = vec![0u64; shot_major_len];
    shot_major.copy_to_host(ctx.device(), &mut host)?;
    Ok(host)
}

pub(crate) fn count_meas_major_marginals(
    ctx: &GpuContext,
    meas_major: &GpuBuffer<u64>,
    num_meas: usize,
    num_shots: usize,
    s_words: usize,
) -> Result<Vec<u64>> {
    if num_meas == 0 || num_shots == 0 {
        return Ok(vec![0u64; num_meas]);
    }

    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "bts_popcount_rows")?;
    let mut row_counts = GpuBuffer::<u64>::alloc_zeros(device, num_meas.max(1))?;
    let mut host_counts = vec![0u64; num_meas];

    let num_meas_i = require_i32("bts_popcount_rows", "num_meas", num_meas)?;
    let s_words_i = require_i32("bts_popcount_rows", "s_words", s_words)?;
    let num_meas_grid = require_u32("bts_popcount_rows", "num_meas", num_meas)?;
    let tail_mask_u64 = shot_tail_mask(num_shots);
    let cfg = linear_cfg(POPCOUNT_BLOCK_SIZE, num_meas_grid);

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw())
        .arg(&num_meas_i)
        .arg(&s_words_i)
        .arg(&tail_mask_u64)
        .arg(row_counts.raw_mut());
    // SAFETY: one block owns one measurement row and writes one unique count.
    unsafe { launch_counted(builder, cfg, "bts_popcount_rows", ctx.device(), 0) }?;

    row_counts.copy_to_host(device, &mut host_counts)?;
    Ok(host_counts)
}

fn count_used_slots(
    ctx: &GpuContext,
    slot_states: &GpuBuffer<u32>,
    table_slots: usize,
) -> Result<usize> {
    if table_slots == 0 {
        return Ok(0);
    }

    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "bts_count_used_slots")?;
    let mut used_out = GpuBuffer::<u32>::alloc_zeros(device, 1)?;
    let table_capacity_i = require_i32("bts_count_used_slots", "table_slots", table_slots)?;
    let blocks = div_ceil_grid(
        "bts_count_used_slots",
        "table_slots",
        table_slots,
        COUNT_HASH_BLOCK_SIZE,
    )?;
    let cfg = linear_cfg(COUNT_HASH_BLOCK_SIZE, blocks);

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(slot_states.raw())
        .arg(&table_capacity_i)
        .arg(used_out.raw_mut());
    // SAFETY: each block scans a disjoint subset of indices and contributes
    // its local sum through one atomic add into `used_out`.
    unsafe { launch_counted(builder, cfg, "bts_count_used_slots", ctx.device(), 0) }?;

    let mut used_host = [0u32; 1];
    used_out.copy_to_host(device, &mut used_host)?;
    Ok(used_host[0] as usize)
}

/// Open-addressed device hash table the count kernels fill: `slots` entries of `m_words`
/// key words, a count, and a state word each, plus a one-word overflow flag.
struct CountTable {
    slots: usize,
    keys: GpuBuffer<u64>,
    counts: GpuBuffer<u64>,
    states: GpuBuffer<u32>,
    overflow: GpuBuffer<u32>,
}

impl CountTable {
    /// `None` when [`plan_count_table_slots`] finds no table size that fits.
    fn alloc(
        ctx: &GpuContext,
        num_shots: usize,
        m_words: usize,
        rank: usize,
    ) -> Result<Option<Self>> {
        let Some(slots) = plan_count_table_slots(ctx, num_shots, m_words, rank)? else {
            return Ok(None);
        };
        let device = ctx.device();
        Ok(Some(Self {
            slots,
            keys: GpuBuffer::<u64>::alloc_zeros(device, (slots * m_words).max(1))?,
            counts: GpuBuffer::<u64>::alloc_zeros(device, slots.max(1))?,
            states: GpuBuffer::<u32>::alloc_zeros(device, slots.max(1))?,
            overflow: GpuBuffer::<u32>::alloc_zeros(device, 1)?,
        }))
    }

    /// Compact the filled table on device; `None` when the fill overflowed it.
    fn finish(
        self,
        ctx: &GpuContext,
        m_words: usize,
        raw_transfer_bytes: usize,
    ) -> Result<Option<std::collections::HashMap<Vec<u64>, u64>>> {
        let mut overflow_host = [0u32; 1];
        self.overflow
            .copy_to_host(ctx.device(), &mut overflow_host)?;
        if overflow_host[0] != 0 {
            return Ok(None);
        }
        compact_count_table(
            ctx,
            &self.keys,
            &self.counts,
            &self.states,
            self.slots,
            m_words,
            raw_transfer_bytes,
        )
    }
}

fn compact_count_table(
    ctx: &GpuContext,
    slot_keys: &GpuBuffer<u64>,
    slot_counts: &GpuBuffer<u64>,
    slot_states: &GpuBuffer<u32>,
    table_slots: usize,
    m_words: usize,
    raw_transfer_bytes: usize,
) -> Result<Option<std::collections::HashMap<Vec<u64>, u64>>> {
    let used_slots = count_used_slots(ctx, slot_states, table_slots)?;
    if used_slots == 0 {
        return Ok(Some(std::collections::HashMap::new()));
    }

    let compact_transfer_bytes = used_slots
        .saturating_mul(m_words + 1)
        .saturating_mul(std::mem::size_of::<u64>());
    if compact_transfer_bytes >= raw_transfer_bytes {
        return Ok(None);
    }

    let device = ctx.device();
    let stream = device.stream()?;
    let mut out_keys = GpuBuffer::<u64>::alloc_zeros(device, (used_slots * m_words).max(1))?;
    let mut out_counts = GpuBuffer::<u64>::alloc_zeros(device, used_slots.max(1))?;
    let mut out_len = GpuBuffer::<u32>::alloc_zeros(device, 1)?;
    let compact_func = device.function("bts_compact_counts_upto8")?;
    let m_words_i = require_i32("bts_compact_counts_upto8", "m_words", m_words)?;
    let table_capacity_i = require_i32("bts_compact_counts_upto8", "table_slots", table_slots)?;
    let compact_blocks = div_ceil_grid(
        "bts_compact_counts_upto8",
        "table_slots",
        table_slots,
        COUNT_COMPACT_BLOCK_SIZE,
    )?;
    let compact_cfg = linear_cfg(COUNT_COMPACT_BLOCK_SIZE, compact_blocks);

    let mut compact_builder = stream.launch_builder(&compact_func);
    compact_builder
        .arg(slot_states.raw())
        .arg(slot_keys.raw())
        .arg(slot_counts.raw())
        .arg(&m_words_i)
        .arg(&table_capacity_i)
        .arg(out_keys.raw_mut())
        .arg(out_counts.raw_mut())
        .arg(out_len.raw_mut());
    // SAFETY: each occupied slot compacts into a unique output index obtained
    // from an atomic counter.
    unsafe {
        launch_counted(
            compact_builder,
            compact_cfg,
            "bts_compact_counts_upto8",
            ctx.device(),
            0,
        )
    }?;

    let mut out_len_host = [0u32; 1];
    out_len.copy_to_host(device, &mut out_len_host)?;
    let used_len = out_len_host[0] as usize;
    if used_len == 0 {
        return Ok(Some(std::collections::HashMap::new()));
    }

    let mut keys_host = vec![0u64; used_len * m_words];
    let mut counts_host = vec![0u64; used_len];
    out_keys.copy_to_host(device, &mut keys_host)?;
    out_counts.copy_to_host(device, &mut counts_host)?;

    let mut counts = std::collections::HashMap::with_capacity(used_len);
    for idx in 0..used_len {
        let key = keys_host[idx * m_words..(idx + 1) * m_words].to_vec();
        counts.insert(key, counts_host[idx]);
    }
    Ok(Some(counts))
}

/// Histogram shot-major records via an open-addressed device hash table.
/// `Ok(None)` defers to the host path (transfer heuristic, table sizing, or
/// table overflow).
pub(crate) fn try_count_shot_major(
    ctx: &GpuContext,
    shot_major: &GpuBuffer<u64>,
    num_shots: usize,
    m_words: usize,
    rank: usize,
    raw_transfer_bytes: usize,
) -> Result<Option<std::collections::HashMap<Vec<u64>, u64>>> {
    if num_shots == 0 || m_words == 0 {
        return Ok(Some(std::collections::HashMap::new()));
    }
    if !should_try_device_count(num_shots, m_words, rank, raw_transfer_bytes) {
        return Ok(None);
    }

    let Some(mut table) = CountTable::alloc(ctx, num_shots, m_words, rank)? else {
        return Ok(None);
    };

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("bts_count_shot_major_upto8")?;
    let num_shots_i = require_i32("bts_count_shot_major_upto8", "num_shots", num_shots)?;
    let m_words_i = require_i32("bts_count_shot_major_upto8", "m_words", m_words)?;
    let table_mask_u32 =
        require_u32("bts_count_shot_major_upto8", "table_slots", table.slots)?.saturating_sub(1);
    let blocks = div_ceil_grid(
        "bts_count_shot_major_upto8",
        "num_shots",
        num_shots,
        COUNT_HASH_BLOCK_SIZE,
    )?;
    let cfg = linear_cfg(COUNT_HASH_BLOCK_SIZE, blocks);

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(shot_major.raw())
        .arg(&num_shots_i)
        .arg(&m_words_i)
        .arg(table.keys.raw_mut())
        .arg(table.counts.raw_mut())
        .arg(table.states.raw_mut())
        .arg(&table_mask_u32)
        .arg(table.overflow.raw());
    // SAFETY: each thread handles at most one shot-major outcome and uses
    // atomics to claim or update hash slots.
    unsafe { launch_counted(builder, cfg, "bts_count_shot_major_upto8", ctx.device(), 0) }?;

    table.finish(ctx, m_words, raw_transfer_bytes)
}

/// Transpose to shot-major on device and count there, falling back to the
/// direct meas-major count kernel. `Ok(None)` defers to the host path.
pub(crate) fn try_count_meas_major(
    ctx: &GpuContext,
    meas_major: &GpuBuffer<u64>,
    num_meas: usize,
    num_shots: usize,
    m_words: usize,
    s_words: usize,
    rank: usize,
) -> Result<Option<std::collections::HashMap<Vec<u64>, u64>>> {
    if num_shots == 0 || num_meas == 0 {
        return Ok(Some(std::collections::HashMap::new()));
    }
    if m_words == 0 || m_words > GPU_COUNTS_MAX_WORDS {
        return Ok(None);
    }

    let raw_transfer_bytes = count_raw_transfer_bytes(num_meas.saturating_mul(s_words));
    if !should_try_device_count(num_shots, m_words, rank, raw_transfer_bytes) {
        return Ok(None);
    }

    if let Ok(shot_major) =
        launch_bts_transpose_meas_to_shot(ctx, meas_major, num_meas, num_shots, s_words, m_words)
        && let Some(counts) = try_count_shot_major(
            ctx,
            &shot_major,
            num_shots,
            m_words,
            rank,
            raw_transfer_bytes,
        )?
    {
        return Ok(Some(counts));
    }

    let Some(mut table) = CountTable::alloc(ctx, num_shots, m_words, rank)? else {
        return Ok(None);
    };

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("bts_count_meas_major_upto8")?;
    let num_meas_i = require_i32("bts_count_meas_major_upto8", "num_meas", num_meas)?;
    let num_shots_i = require_i32("bts_count_meas_major_upto8", "num_shots", num_shots)?;
    let s_words_i = require_i32("bts_count_meas_major_upto8", "s_words", s_words)?;
    let m_words_i = require_i32("bts_count_meas_major_upto8", "m_words", m_words)?;
    let table_mask_u32 =
        require_u32("bts_count_meas_major_upto8", "table_slots", table.slots)?.saturating_sub(1);
    let batches = div_ceil_grid(
        "bts_count_meas_major_upto8",
        "num_shots",
        num_shots,
        COUNT_MEAS_BATCH_SIZE,
    )?;
    let cfg = linear_cfg(COUNT_MEAS_BATCH_SIZE, batches.max(1));

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw())
        .arg(&num_meas_i)
        .arg(&num_shots_i)
        .arg(&s_words_i)
        .arg(&m_words_i)
        .arg(table.keys.raw_mut())
        .arg(table.counts.raw_mut())
        .arg(table.states.raw_mut())
        .arg(&table_mask_u32)
        .arg(table.overflow.raw());
    // SAFETY: each thread handles at most one logical shot and uses atomics to
    // claim or update hash slots.
    unsafe { launch_counted(builder, cfg, "bts_count_meas_major_upto8", ctx.device(), 0) }?;

    table.finish(ctx, m_words, raw_transfer_bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        CHUNK_SHOTS, NoiseMaskGenerate, NoiseRowApply, apply_noise_row_major, generate_noise_masks,
        launch_bts_transpose_meas_to_shot,
    };
    use crate::gpu::{GpuBuffer, GpuContext};
    use crate::sim::compiled::shot_tail_mask;

    fn try_gpu_ctx() -> Option<std::sync::Arc<GpuContext>> {
        match GpuContext::new(0) {
            Ok(ctx) => Some(ctx),
            Err(e) => {
                assert!(
                    std::env::var_os("PRISM_REQUIRE_GPU").is_none(),
                    "PRISM_REQUIRE_GPU is set but no usable GPU was found ({e})"
                );
                eprintln!("SKIP: no usable GPU ({e})");
                None
            }
        }
    }

    fn splitmix64_step(mut x: u64) -> u64 {
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn host_generate_noise_masks(
        thresholds: &[[u64; 3]],
        chunk_shots: usize,
        master_seed: u64,
        batch_offset: u64,
    ) -> (Vec<u64>, Vec<u64>) {
        let num_events = thresholds.len();
        let chunk_s_words = chunk_shots.div_ceil(64);
        let tail_mask = shot_tail_mask(chunk_shots);
        let mut x_masks = vec![0u64; num_events * chunk_s_words];
        let mut z_masks = vec![0u64; num_events * chunk_s_words];

        for (event, &[t_x, t_xy, t_p]) in thresholds.iter().enumerate() {
            if t_p == 0 {
                continue;
            }
            for batch in 0..chunk_s_words {
                let absolute_batch = batch_offset + batch as u64;
                let seed = master_seed
                    ^ ((event as u64).wrapping_mul(0x9e3779b97f4a7c15))
                    ^ splitmix64_step(absolute_batch);
                let mut s0 = splitmix64_step(seed);
                let mut s1 = splitmix64_step(s0);
                let mut s2 = splitmix64_step(s1);
                let mut s3 = splitmix64_step(s2);

                let mut x_mask = 0u64;
                let mut z_mask = 0u64;
                for bit in 0..64 {
                    let result = s0.wrapping_add(s3).rotate_left(23).wrapping_add(s0);
                    let t = s1 << 17;
                    s2 ^= s0;
                    s3 ^= s1;
                    s1 ^= s2;
                    s0 ^= s3;
                    s2 ^= t;
                    s3 = s3.rotate_left(45);

                    let bit_mask = 1u64 << bit;
                    if result < t_x {
                        z_mask |= bit_mask;
                    } else if result < t_xy {
                        x_mask |= bit_mask;
                        z_mask |= bit_mask;
                    } else if result < t_p {
                        x_mask |= bit_mask;
                    }
                }
                if batch == chunk_s_words - 1 {
                    x_mask &= tail_mask;
                    z_mask &= tail_mask;
                }
                let out_idx = event * chunk_s_words + batch;
                x_masks[out_idx] = x_mask;
                z_masks[out_idx] = z_mask;
            }
        }
        (x_masks, z_masks)
    }

    // 65,538 shot words need one tile row each, three past the 65,535 limit on
    // gridDim.y, and the last word is partial.
    #[test]
    fn transpose_past_the_grid_y_limit_matches_host() {
        let Some(ctx) = try_gpu_ctx() else { return };
        let num_meas = 3;
        let num_shots = 65_537 * 64 + 36usize;
        let s_words = num_shots.div_ceil(64);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let meas_major: Vec<u64> = (0..num_meas * s_words)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            })
            .collect();
        let device = ctx.device();
        let meas_major_dev = GpuBuffer::from_host(device, &meas_major).unwrap();

        let shot_major_dev = launch_bts_transpose_meas_to_shot(
            &ctx,
            &meas_major_dev,
            num_meas,
            num_shots,
            s_words,
            1,
        )
        .unwrap();
        let mut shot_major = vec![0u64; num_shots];
        shot_major_dev
            .copy_to_host(device, &mut shot_major)
            .unwrap();

        for (shot, &got) in shot_major.iter().enumerate() {
            let want = (0..num_meas).fold(0u64, |acc, m| {
                acc | ((meas_major[m * s_words + shot / 64] >> (shot % 64)) & 1) << m
            });
            assert_eq!(got, want, "shot {shot}");
        }
    }

    #[test]
    fn generate_noise_masks_matches_host_xoshiro_across_chunks_and_tail() {
        let Some(ctx) = try_gpu_ctx() else { return };
        let device = ctx.device();

        let thresholds: [[u64; 3]; 6] = [
            [0, 0, 0],
            [0, 0, u64::MAX / 3],
            [0, u64::MAX / 2, u64::MAX / 2],
            [u64::MAX / 4, u64::MAX / 4, u64::MAX / 4],
            [u64::MAX / 10, (u64::MAX / 10) * 3, (u64::MAX / 10) * 7],
            [0, 0, u64::MAX],
        ];
        let num_events = thresholds.len();
        let flat_thresholds: Vec<u64> = thresholds.iter().flatten().copied().collect();
        let thresholds_dev = GpuBuffer::from_host(device, &flat_thresholds).unwrap();

        let max_chunk_words = CHUNK_SHOTS.div_ceil(64);
        let max_len = num_events * max_chunk_words;
        let sentinel = vec![0xDEAD_BEEF_CAFE_BABEu64; max_len];
        let mut x_masks_dev = GpuBuffer::from_host(device, &sentinel).unwrap();
        let mut z_masks_dev = GpuBuffer::from_host(device, &sentinel).unwrap();

        let master_seed = 0x1234_5678_9ABC_DEF0u64;
        let total_shots = CHUNK_SHOTS + 129;
        let mut shot_start = 0usize;
        while shot_start < total_shots {
            let chunk_shots = (total_shots - shot_start).min(CHUNK_SHOTS);
            let chunk_s_words = chunk_shots.div_ceil(64);
            let batch_offset = (shot_start / 64) as u64;
            let tail_mask = shot_tail_mask(chunk_shots);

            // Re-dirty the active prefix to verify t_p == 0 overwrites stale bits.
            let mask_len = num_events * chunk_s_words;
            x_masks_dev
                .copy_from_host(device, &sentinel[..mask_len])
                .unwrap();
            z_masks_dev
                .copy_from_host(device, &sentinel[..mask_len])
                .unwrap();

            generate_noise_masks(
                &ctx,
                NoiseMaskGenerate {
                    x_masks: &mut x_masks_dev,
                    z_masks: &mut z_masks_dev,
                    event_thresholds: &thresholds_dev,
                    num_events,
                    chunk_s_words,
                    tail_mask,
                    master_seed,
                    batch_offset,
                },
            )
            .unwrap();

            let mut got_x = vec![0u64; max_len];
            let mut got_z = vec![0u64; max_len];
            x_masks_dev.copy_to_host(device, &mut got_x).unwrap();
            z_masks_dev.copy_to_host(device, &mut got_z).unwrap();

            let (want_x, want_z) =
                host_generate_noise_masks(&thresholds, chunk_shots, master_seed, batch_offset);
            assert_eq!(
                &got_x[..mask_len],
                want_x.as_slice(),
                "shot_start={shot_start}"
            );
            assert_eq!(
                &got_z[..mask_len],
                want_z.as_slice(),
                "shot_start={shot_start}"
            );
            for event in 0..num_events {
                let last = event * chunk_s_words + (chunk_s_words - 1);
                assert_eq!(got_x[last] & !tail_mask, 0, "x padding dirty event {event}");
                assert_eq!(got_z[last] & !tail_mask, 0, "z padding dirty event {event}");
            }
            shot_start += chunk_shots;
        }

        // Non-zero batch_offset with partial word (131 shots = 2 full words + 3 bits).
        let tail_shots = 131usize;
        let tail_words = tail_shots.div_ceil(64);
        let tail_mask = shot_tail_mask(tail_shots);
        let tail_len = num_events * tail_words;
        x_masks_dev
            .copy_from_host(device, &sentinel[..tail_len])
            .unwrap();
        z_masks_dev
            .copy_from_host(device, &sentinel[..tail_len])
            .unwrap();
        generate_noise_masks(
            &ctx,
            NoiseMaskGenerate {
                x_masks: &mut x_masks_dev,
                z_masks: &mut z_masks_dev,
                event_thresholds: &thresholds_dev,
                num_events,
                chunk_s_words: tail_words,
                tail_mask,
                master_seed,
                batch_offset: 7,
            },
        )
        .unwrap();
        let mut got_x = vec![0u64; max_len];
        let mut got_z = vec![0u64; max_len];
        x_masks_dev.copy_to_host(device, &mut got_x).unwrap();
        z_masks_dev.copy_to_host(device, &mut got_z).unwrap();
        let (want_x, want_z) = host_generate_noise_masks(&thresholds, tail_shots, master_seed, 7);
        assert_eq!(&got_x[..tail_len], want_x.as_slice());
        assert_eq!(&got_z[..tail_len], want_z.as_slice());
    }

    #[test]
    fn apply_noise_row_major_accumulates_csr_and_preserves_neighbors() {
        let Some(ctx) = try_gpu_ctx() else { return };
        let device = ctx.device();

        let num_meas = 5usize;
        let s_words = 10usize;
        let word_offset = 3usize;
        let chunk_s_words = 4usize;
        let num_events = 4usize;

        let x_masks: Vec<u64> = (0..num_events * chunk_s_words)
            .map(|i| 0x1111_2222_3333_4444u64.wrapping_add((i as u64 + 1) * 0x0102_0304_0506_0708))
            .collect();
        let z_masks: Vec<u64> = (0..num_events * chunk_s_words)
            .map(|i| {
                0x8888_7777_6666_5555u64 ^ ((i as u64 + 1).wrapping_mul(0x1020_3040_5060_7080))
            })
            .collect();

        // Row 0: empty; Row 1: e0(X); Row 2: e1(Z);
        // Row 3: e2(Y), e2(X), e2(Z) cancel + e3(Y);
        // Row 4: e0(X), e1(Z), e3(X).
        let pack = |event: u32, flag: u32| (event << 2) | flag;
        let row_event_offsets = vec![0u32, 0, 1, 2, 6, 9];
        let row_event_entries = vec![
            pack(0, 1),
            pack(1, 2),
            pack(2, 3),
            pack(2, 1),
            pack(2, 2),
            pack(3, 3),
            pack(0, 1),
            pack(1, 2),
            pack(3, 1),
        ];

        let initial_meas: Vec<u64> = (0..num_meas * s_words)
            .map(|i| 0xA5A5_5A5A_F0F0_0F0Fu64.wrapping_mul(i as u64 + 3))
            .collect();
        let mut expected = initial_meas.clone();
        for row in 0..num_meas {
            let start = row_event_offsets[row] as usize;
            let end = row_event_offsets[row + 1] as usize;
            for batch in 0..chunk_s_words {
                let mut acc = 0u64;
                for &entry in &row_event_entries[start..end] {
                    let event = (entry >> 2) as usize;
                    let flag = entry & 3;
                    let idx = event * chunk_s_words + batch;
                    if flag & 1 != 0 {
                        acc ^= x_masks[idx];
                    }
                    if flag & 2 != 0 {
                        acc ^= z_masks[idx];
                    }
                }
                expected[row * s_words + word_offset + batch] ^= acc;
            }
        }

        let mut meas_dev = GpuBuffer::from_host(device, &initial_meas).unwrap();
        let offsets_dev = GpuBuffer::from_host(device, &row_event_offsets).unwrap();
        let entries_dev = GpuBuffer::from_host(device, &row_event_entries).unwrap();
        let x_dev = GpuBuffer::from_host(device, &x_masks).unwrap();
        let z_dev = GpuBuffer::from_host(device, &z_masks).unwrap();

        apply_noise_row_major(
            &ctx,
            NoiseRowApply {
                meas_major: &mut meas_dev,
                num_meas,
                s_words,
                word_offset,
                chunk_s_words,
                row_event_offsets: &offsets_dev,
                row_event_entries: &entries_dev,
                x_masks: &x_dev,
                z_masks: &z_dev,
            },
        )
        .unwrap();

        let mut got = vec![0u64; num_meas * s_words];
        meas_dev.copy_to_host(device, &mut got).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn two_kernel_noise_split_empty_inputs_are_noops() {
        let Some(ctx) = try_gpu_ctx() else { return };
        let device = ctx.device();

        let sentinel = [0x5555_AAAA_5555_AAAAu64; 4];
        let mut x_dev = GpuBuffer::from_host(device, &sentinel).unwrap();
        let mut z_dev = GpuBuffer::from_host(device, &sentinel).unwrap();
        let thresholds_dev = GpuBuffer::<u64>::alloc_zeros(device, 3).unwrap();
        let mut meas_dev = GpuBuffer::from_host(device, &sentinel).unwrap();
        let offsets_dev = GpuBuffer::<u32>::alloc_zeros(device, 2).unwrap();
        let entries_dev = GpuBuffer::<u32>::alloc_zeros(device, 1).unwrap();

        for (num_events, chunk_s_words) in [(0, 4), (2, 0), (0, 0)] {
            generate_noise_masks(
                &ctx,
                NoiseMaskGenerate {
                    x_masks: &mut x_dev,
                    z_masks: &mut z_dev,
                    event_thresholds: &thresholds_dev,
                    num_events,
                    chunk_s_words,
                    tail_mask: u64::MAX,
                    master_seed: 42,
                    batch_offset: 0,
                },
            )
            .unwrap();
        }

        for (num_meas, s_words, chunk_s_words) in [(0, 4, 2), (2, 0, 2), (2, 4, 0)] {
            apply_noise_row_major(
                &ctx,
                NoiseRowApply {
                    meas_major: &mut meas_dev,
                    num_meas,
                    s_words,
                    word_offset: 0,
                    chunk_s_words,
                    row_event_offsets: &offsets_dev,
                    row_event_entries: &entries_dev,
                    x_masks: &x_dev,
                    z_masks: &z_dev,
                },
            )
            .unwrap();
        }

        let mut got_meas = [0u64; 4];
        meas_dev.copy_to_host(device, &mut got_meas).unwrap();
        assert_eq!(got_meas, sentinel);
    }
}
