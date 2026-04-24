//! Block-triangular sampling on the GPU.
//!
//! Ports the CPU path in `src/sim/compiled/bts.rs` (`sample_bts_meas_major`)
//! to a single-kernel device launch. The kernel consumes the precomputed
//! CSR parity matrix plus a host-generated `random_bits` pool and emits
//! packed shot outcomes in measurement-major layout.
//!
//! Work partition: one thread per (measurement, 64-shot batch) pair. Host
//! generates the full random-bits pool via the existing `Xoshiro256PlusPlus`
//! RNG so the CPU and GPU paths share one RNG contract; very large shot
//! counts stream through chunks of `CHUNK_SHOTS` shots at a time to bound
//! peak GPU memory.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::error::{PrismError, Result};
use crate::gpu::GpuBuffer;
use crate::sim::compiled::parity::SparseParity;
use crate::sim::compiled::rng::Xoshiro256PlusPlus;

use super::super::GpuContext;

const SAMPLE_BLOCK_SIZE: u32 = 128;
const NOISE_BLOCK_SIZE: u32 = 128;
const POPCOUNT_BLOCK_SIZE: u32 = 256;
const COUNT_MEAS_BATCH_SIZE: u32 = 64;
const COUNT_HASH_BLOCK_SIZE: u32 = 256;
const COUNT_COMPACT_BLOCK_SIZE: u32 = 256;

pub(crate) const GPU_COUNTS_MAX_WORDS: usize = 8;

/// Host-side random-bits pool is allocated per chunk, so peak device memory
/// stays bounded regardless of total shot count. Matches the CPU
/// `BTS_BATCH_SHOTS` chunking boundary.
const CHUNK_SHOTS: usize = 65_536;

const KERNEL_SOURCE: &str = r#"
// Consumes `random_bits` laid out as [batch][col] (row-major, batch varying
// slowest). Each thread owns one (measurement, batch) pair and produces one
// u64 of packed shot outcomes.
//
//   meas_major[m * s_words + batch] = XOR over cols of random_bits[batch][c]
//                                     flipped by ref_bits[m] when set.
//
// Deterministic rows (row_offsets[m+1] == row_offsets[m]) fall out naturally
// with acc = 0 before the ref_bit flip, matching the CPU zero-init + flip
// path in `apply_ref_bits_meas_major`.
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ unsigned long long hash_words(
    const unsigned long long *key,
    int words
) {
    unsigned long long acc = 0x9e3779b97f4a7c15ULL ^ (unsigned long long)words;
    for (int i = 0; i < words; ++i) {
        acc = mix64(acc ^ key[i] ^ ((unsigned long long)i * 0x9e3779b97f4a7c15ULL));
    }
    return acc;
}

__device__ __forceinline__ unsigned int load_state(const unsigned int *state) {
    return atomicAdd((unsigned int *)state, 0U);
}

__device__ __forceinline__ int keys_equal(
    const unsigned long long *lhs,
    const unsigned long long *rhs,
    int words
) {
    for (int i = 0; i < words; ++i) {
        if (lhs[i] != rhs[i]) return 0;
    }
    return 1;
}

extern "C" __global__ void bts_sample_meas_major(
    const unsigned int *col_indices,
    const unsigned int *row_offsets,
    const unsigned long long *ref_bits,
    const unsigned long long *random_bits,
    int num_meas,
    int s_words,
    int rank,
    int out_stride_words,
    int out_word_offset,
    unsigned long long *meas_major
) {
    int m = blockIdx.x;
    int batch = blockIdx.y * blockDim.x + threadIdx.x;
    if (m >= num_meas || batch >= s_words) return;

    unsigned int start = row_offsets[m];
    unsigned int end = row_offsets[m + 1];
    unsigned long long acc = 0ULL;
    const unsigned long long *rb = random_bits + (unsigned long long)batch * (unsigned long long)rank;
    for (unsigned int i = start; i < end; ++i) {
        acc ^= rb[col_indices[i]];
    }

    unsigned long long ref_word = ref_bits[m >> 6];
    unsigned long long ref_bit = (ref_word >> (m & 63)) & 1ULL;
    if (ref_bit != 0ULL) acc = ~acc;

    meas_major[
        (unsigned long long)m * (unsigned long long)out_stride_words +
        (unsigned long long)out_word_offset +
        (unsigned long long)batch
    ] = acc;
}

extern "C" __global__ void bts_popcount_rows(
    const unsigned long long *meas_major,
    int num_meas,
    int s_words,
    unsigned long long tail_mask,
    unsigned long long *row_counts
) {
    int m = blockIdx.x;
    if (m >= num_meas) return;

    __shared__ unsigned long long sums[256];
    unsigned long long acc = 0ULL;
    unsigned long long row_base = (unsigned long long)m * (unsigned long long)s_words;

    for (int word = threadIdx.x; word < s_words; word += blockDim.x) {
        unsigned long long bits = meas_major[row_base + (unsigned long long)word];
        if (word == s_words - 1) {
            bits &= tail_mask;
        }
        acc += (unsigned long long)__popcll(bits);
    }

    sums[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_counts[m] = sums[0];
    }
}

extern "C" __global__ void bts_count_meas_major_upto8(
    const unsigned long long *meas_major,
    int num_meas,
    int num_shots,
    int s_words,
    int m_words,
    unsigned long long *slot_keys,
    unsigned long long *slot_counts,
    unsigned int *slot_states,
    unsigned int table_mask,
    unsigned int *overflow
) {
    int lane = threadIdx.x;
    int batch = blockIdx.x;
    int shot = batch * 64 + lane;

    __shared__ unsigned long long row_bits[64];
    __shared__ unsigned long long shot_words[64][8];

    if (lane < 64) {
        #pragma unroll
        for (int mw = 0; mw < 8; ++mw) {
            shot_words[lane][mw] = 0ULL;
        }
    }
    __syncthreads();

    for (int mw = 0; mw < m_words; ++mw) {
        int rows_in_group = num_meas - mw * 64;
        if (rows_in_group > 64) rows_in_group = 64;

        if (lane < rows_in_group) {
            row_bits[lane] =
                meas_major[(unsigned long long)(mw * 64 + lane) * (unsigned long long)s_words +
                          (unsigned long long)batch];
        }
        __syncthreads();

        if (shot < num_shots) {
            unsigned long long word = 0ULL;
            #pragma unroll
            for (int bit = 0; bit < 64; ++bit) {
                if (bit < rows_in_group) {
                    word |= ((row_bits[bit] >> lane) & 1ULL) << bit;
                }
            }
            shot_words[lane][mw] = word;
        }
        __syncthreads();
    }

    if (shot >= num_shots) return;

    unsigned long long key[8];
    #pragma unroll
    for (int mw = 0; mw < 8; ++mw) {
        key[mw] = shot_words[lane][mw];
    }

    unsigned long long hash = hash_words(key, m_words);
    unsigned int slot = (unsigned int)hash & table_mask;

    for (unsigned int probe = 0; probe <= table_mask; ++probe) {
        unsigned int state = load_state(slot_states + slot);
        unsigned long long *slot_key = slot_keys + (unsigned long long)slot * (unsigned long long)m_words;

        if (state == 2U) {
            if (keys_equal(slot_key, key, m_words)) {
                atomicAdd(slot_counts + slot, 1ULL);
                return;
            }
        } else if (state == 0U) {
            if (atomicCAS(slot_states + slot, 0U, 1U) == 0U) {
                for (int mw = 0; mw < m_words; ++mw) {
                    slot_key[mw] = key[mw];
                }
                slot_counts[slot] = 1ULL;
                __threadfence();
                atomicExch(slot_states + slot, 2U);
                return;
            }
            continue;
        } else {
            while ((state = load_state(slot_states + slot)) == 1U) {
            }
            if (state == 2U && keys_equal(slot_key, key, m_words)) {
                atomicAdd(slot_counts + slot, 1ULL);
                return;
            }
        }

        slot = (slot + 1U) & table_mask;
    }

    atomicExch(overflow, 1U);
}

extern "C" __global__ void bts_count_shot_major_upto8(
    const unsigned long long *shot_major,
    int num_shots,
    int m_words,
    unsigned long long *slot_keys,
    unsigned long long *slot_counts,
    unsigned int *slot_states,
    unsigned int table_mask,
    unsigned int *overflow
) {
    int shot = blockIdx.x * blockDim.x + threadIdx.x;
    if (shot >= num_shots) return;

    unsigned long long key[8];
    #pragma unroll
    for (int mw = 0; mw < 8; ++mw) {
        key[mw] = 0ULL;
    }

    const unsigned long long *shot_key =
        shot_major + (unsigned long long)shot * (unsigned long long)m_words;
    #pragma unroll
    for (int mw = 0; mw < 8; ++mw) {
        if (mw < m_words) {
            key[mw] = shot_key[mw];
        }
    }

    unsigned long long hash = hash_words(key, m_words);
    unsigned int slot = (unsigned int)hash & table_mask;

    for (unsigned int probe = 0; probe <= table_mask; ++probe) {
        unsigned int state = load_state(slot_states + slot);
        unsigned long long *slot_key =
            slot_keys + (unsigned long long)slot * (unsigned long long)m_words;

        if (state == 2U) {
            if (keys_equal(slot_key, key, m_words)) {
                atomicAdd(slot_counts + slot, 1ULL);
                return;
            }
        } else if (state == 0U) {
            if (atomicCAS(slot_states + slot, 0U, 1U) == 0U) {
                for (int mw = 0; mw < m_words; ++mw) {
                    slot_key[mw] = key[mw];
                }
                slot_counts[slot] = 1ULL;
                __threadfence();
                atomicExch(slot_states + slot, 2U);
                return;
            }
            continue;
        } else {
            while ((state = load_state(slot_states + slot)) == 1U) {
            }
            if (state == 2U && keys_equal(slot_key, key, m_words)) {
                atomicAdd(slot_counts + slot, 1ULL);
                return;
            }
        }

        slot = (slot + 1U) & table_mask;
    }

    atomicExch(overflow, 1U);
}

extern "C" __global__ void bts_count_used_slots(
    const unsigned int *slot_states,
    int table_capacity,
    unsigned int *used_out
) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ unsigned int sums[256];
    unsigned int acc = 0U;

    for (int idx = slot; idx < table_capacity; idx += stride) {
        acc += (slot_states[idx] == 2U);
    }

    sums[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sums[threadIdx.x] += sums[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && sums[0] != 0U) {
        atomicAdd(used_out, sums[0]);
    }
}

extern "C" __global__ void bts_compact_counts_upto8(
    const unsigned int *slot_states,
    const unsigned long long *slot_keys,
    const unsigned long long *slot_counts,
    int m_words,
    int table_capacity,
    unsigned long long *out_keys,
    unsigned long long *out_counts,
    unsigned int *out_len
) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= table_capacity) return;
    if (slot_states[slot] != 2U) return;

    unsigned int out = atomicAdd(out_len, 1U);
    const unsigned long long *src_key =
        slot_keys + (unsigned long long)slot * (unsigned long long)m_words;
    unsigned long long *dst_key =
        out_keys + (unsigned long long)out * (unsigned long long)m_words;
    for (int mw = 0; mw < m_words; ++mw) {
        dst_key[mw] = src_key[mw];
    }
    out_counts[out] = slot_counts[slot];
}

// Bit-transpose a 64 x 64 block of the meas-major BTS output into shot-major
// layout. Each block handles one tile indexed by (m_word, batch). The block
// loads 64 meas-major u64s into shared memory, then each thread writes one
// shot-major u64 containing 64 measurements for its shot.
//
// Out-of-bounds tiles (m_word * 64 + i >= num_meas or batch * 64 + j >= num_shots)
// zero-fill the tile rows or skip the write so callers can safely launch with
// ceiling-divided grid shape.
extern "C" __global__ void bts_transpose_meas_to_shot(
    const unsigned long long *meas_major,
    int num_meas,
    int num_shots,
    int s_words,
    int m_words,
    unsigned long long *shot_major
) {
    int m_word = blockIdx.x;
    int batch = blockIdx.y;
    int tid = threadIdx.x;

    __shared__ unsigned long long tile[64];

    int m = m_word * 64 + tid;
    if (m < num_meas) {
        tile[tid] = meas_major[(unsigned long long)m * (unsigned long long)s_words
                              + (unsigned long long)batch];
    } else {
        tile[tid] = 0ULL;
    }
    __syncthreads();

    int shot = batch * 64 + tid;
    if (shot >= num_shots) return;

    unsigned long long out = 0ULL;
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        out |= ((tile[i] >> tid) & 1ULL) << i;
    }
    shot_major[(unsigned long long)shot * (unsigned long long)m_words
              + (unsigned long long)m_word] = out;
}

// Shared helpers for the fused noise kernel below. The xoshiro256++ stream
// per (event, 64-shot batch) is seeded from a master RNG value plus a
// splitmix64 hash of (event, absolute batch), so every bit of randomness is
// drawn on the device. Event thresholds are packed by the host as three u64
// scales of [px, px+py, px+py+pz] so the inner 64-bit compare replaces an fp
// multiply and branch per shot.
__device__ __forceinline__ unsigned long long bts_splitmix64_step(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ unsigned long long bts_rotl64(unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Per-(row, batch) fused noise generator and XOR accumulator. Each thread
// owns one (row, batch) output word, walks that row's event list, and
// accumulates the 64-bit masks in a register. The single `^=` write at the
// end replaces up to N `atomicXor` calls and removes the cross-block
// contention an event-major launch would pay on rows shared by many events.
//
// Entry layout: `event << 2 | flag`. Flag bit 0 means the event contributes
// X to this row, bit 1 means Z. A Y contribution sets both bits and applies
// both masks from the single xoshiro stream.
extern "C" __global__ void bts_generate_and_apply_noise_meas_major_by_row(
    unsigned long long *meas_major,
    int num_meas,
    int s_words,
    int out_word_offset,
    const unsigned int *row_event_offsets,
    const unsigned int *row_event_entries,
    const unsigned long long *event_thresholds,
    int chunk_s_words,
    unsigned long long master_seed,
    unsigned long long batch_offset
) {
    int row = blockIdx.x;
    int batch = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= num_meas || batch >= chunk_s_words) return;

    unsigned int start = row_event_offsets[row];
    unsigned int end = row_event_offsets[row + 1];
    if (start == end) return;

    unsigned long long absolute_batch = batch_offset + (unsigned long long)batch;
    unsigned long long batch_mix = bts_splitmix64_step(absolute_batch);
    unsigned long long acc = 0ULL;

    for (unsigned int i = start; i < end; ++i) {
        unsigned int entry = row_event_entries[i];
        unsigned int event = entry >> 2;
        unsigned int flag = entry & 3u;

        unsigned long long t_x  = event_thresholds[(unsigned long long)event * 3ULL + 0ULL];
        unsigned long long t_xy = event_thresholds[(unsigned long long)event * 3ULL + 1ULL];
        unsigned long long t_p  = event_thresholds[(unsigned long long)event * 3ULL + 2ULL];
        if (t_p == 0ULL) continue;

        // Match the event-major kernel's seed derivation exactly so both
        // paths produce identical outcomes from the same master seed.
        unsigned long long seed = master_seed
            ^ ((unsigned long long)event * 0x9e3779b97f4a7c15ULL)
            ^ batch_mix;
        unsigned long long s0 = bts_splitmix64_step(seed);
        unsigned long long s1 = bts_splitmix64_step(s0);
        unsigned long long s2 = bts_splitmix64_step(s1);
        unsigned long long s3 = bts_splitmix64_step(s2);

        unsigned long long x_mask = 0ULL;
        unsigned long long z_mask = 0ULL;
        #pragma unroll 8
        for (int bit = 0; bit < 64; ++bit) {
            unsigned long long result = bts_rotl64(s0 + s3, 23) + s0;
            unsigned long long t = s1 << 17;
            s2 ^= s0;
            s3 ^= s1;
            s1 ^= s2;
            s0 ^= s3;
            s2 ^= t;
            s3 = bts_rotl64(s3, 45);

            unsigned long long bit_mask = 1ULL << bit;
            if (result < t_x) {
                z_mask |= bit_mask;
            } else if (result < t_xy) {
                x_mask |= bit_mask;
                z_mask |= bit_mask;
            } else if (result < t_p) {
                x_mask |= bit_mask;
            }
        }

        if (flag & 1u) acc ^= x_mask;
        if (flag & 2u) acc ^= z_mask;
    }

    if (acc == 0ULL) return;
    unsigned long long dst_word = (unsigned long long)out_word_offset + (unsigned long long)batch;
    unsigned long long idx = (unsigned long long)row * (unsigned long long)s_words + dst_word;
    meas_major[idx] ^= acc;
}

extern "C" __global__ void bts_apply_noise_masks_meas_major(
    unsigned long long *meas_major,
    int num_meas,
    int s_words,
    int out_word_offset,
    const unsigned int *x_row_offsets,
    const unsigned int *x_row_indices,
    const unsigned int *z_row_offsets,
    const unsigned int *z_row_indices,
    const unsigned long long *x_masks,
    const unsigned long long *z_masks,
    int chunk_s_words,
    int num_events
) {
    int event = blockIdx.x;
    int batch = blockIdx.y * blockDim.x + threadIdx.x;
    if (event >= num_events || batch >= chunk_s_words) return;

    unsigned long long x_mask =
        x_masks[(unsigned long long)event * (unsigned long long)chunk_s_words +
                (unsigned long long)batch];
    unsigned long long z_mask =
        z_masks[(unsigned long long)event * (unsigned long long)chunk_s_words +
                (unsigned long long)batch];
    if (x_mask == 0ULL && z_mask == 0ULL) return;

    unsigned long long dst_word = (unsigned long long)out_word_offset + (unsigned long long)batch;

    if (x_mask != 0ULL) {
        unsigned int start = x_row_offsets[event];
        unsigned int end = x_row_offsets[event + 1];
        for (unsigned int i = start; i < end; ++i) {
            unsigned int row = x_row_indices[i];
            if ((int)row < num_meas) {
                atomicXor(
                    meas_major + (unsigned long long)row * (unsigned long long)s_words + dst_word,
                    x_mask
                );
            }
        }
    }

    if (z_mask != 0ULL) {
        unsigned int start = z_row_offsets[event];
        unsigned int end = z_row_offsets[event + 1];
        for (unsigned int i = start; i < end; ++i) {
            unsigned int row = z_row_indices[i];
            if ((int)row < num_meas) {
                atomicXor(
                    meas_major + (unsigned long long)row * (unsigned long long)s_words + dst_word,
                    z_mask
                );
            }
        }
    }
}

"#;

pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE.to_string()
}

fn launch_err(op: &str, err: impl std::fmt::Display) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {err}"),
    }
}

fn launch_limit_err(op: &str, name: &str, value: usize, limit: &str) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {name}={value} exceeds {limit} kernel limit"),
    }
}

fn require_i32(op: &str, name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| launch_limit_err(op, name, value, "i32"))
}

fn require_u32(op: &str, name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| launch_limit_err(op, name, value, "u32"))
}

fn div_ceil_grid(op: &str, name: &str, value: usize, block: u32) -> Result<u32> {
    Ok(require_u32(op, name, value)?.div_ceil(block).max(1))
}

/// Pick a block size for the noise kernels that doesn't leave the back half
/// of the block idle on small tiles. Clamped to `[32, NOISE_BLOCK_SIZE]` and
/// rounded to a power of two so warp-sized scheduling stays well-behaved.
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
    random_bits_host: Vec<u64>,
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
            random_bits_host: Vec::new(),
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
        if self.random_bits_host.len() < random_needed {
            self.random_bits_host.resize(random_needed, 0);
        }
        if self.chunk_output_host.len() < output_len.max(1) {
            self.chunk_output_host.resize(output_len.max(1), 0);
        }
        Ok(())
    }

    fn fill_random_bits(
        &mut self,
        rng: &mut Xoshiro256PlusPlus,
        rank: usize,
        chunk_shots: usize,
        chunk_s_words: usize,
    ) {
        // Parallel path: for a large enough pool, seed one fresh xoshiro
        // stream per worker from the master rng (matches the CPU BTS batched
        // path in `src/sim/compiled/bts.rs`). Serial fallback keeps the
        // existing single-stream sequence for small chunks or CPU-only
        // builds so determinism is not accidentally coupled to the Rayon
        // thread count.
        let required = chunk_s_words * rank;
        #[cfg(not(feature = "parallel"))]
        let _ = required;
        #[cfg(feature = "parallel")]
        {
            const MIN_PAR_DRAWS: usize = 16_384;
            let num_threads = rayon::current_num_threads();
            if num_threads > 1 && required >= MIN_PAR_DRAWS {
                let batches_per_thread = chunk_s_words.div_ceil(num_threads);
                if batches_per_thread > 0 {
                    let thread_seeds: Vec<[u64; 4]> = (0..num_threads)
                        .map(|_| {
                            [
                                rng.next_u64(),
                                rng.next_u64(),
                                rng.next_u64(),
                                rng.next_u64(),
                            ]
                        })
                        .collect();
                    let tail_mask: Option<u64> = {
                        let rem = chunk_shots % 64;
                        if rem == 0 {
                            None
                        } else {
                            Some((1u64 << rem) - 1)
                        }
                    };
                    let last_batch = chunk_s_words - 1;
                    use rayon::prelude::*;
                    self.random_bits_host[..required]
                        .par_chunks_mut(batches_per_thread * rank)
                        .enumerate()
                        .for_each(|(tid, slab)| {
                            let mut trng = Xoshiro256PlusPlus::from_seeds(thread_seeds[tid]);
                            let first_batch = tid * batches_per_thread;
                            let batches_here = slab.len() / rank;
                            for b in 0..batches_here {
                                let start = b * rank;
                                let end = start + rank;
                                for word in &mut slab[start..end] {
                                    *word = trng.next_u64();
                                }
                                let absolute_batch = first_batch + b;
                                if absolute_batch == last_batch {
                                    if let Some(mask) = tail_mask {
                                        for word in &mut slab[start..end] {
                                            *word &= mask;
                                        }
                                    }
                                }
                            }
                        });
                    return;
                }
            }
        }

        // Serial fallback.
        for batch in 0..chunk_s_words {
            let start = batch * rank;
            let end = start + rank;
            for word in &mut self.random_bits_host[start..end] {
                *word = rng.next_u64();
            }
            if batch == chunk_s_words - 1 {
                let rem = chunk_shots % 64;
                if rem != 0 {
                    let mask = (1u64 << rem) - 1;
                    for word in &mut self.random_bits_host[start..end] {
                        *word &= mask;
                    }
                }
            }
        }
    }
}

fn tail_mask(num_shots: usize) -> u64 {
    let rem = num_shots % 64;
    if rem == 0 {
        !0u64
    } else {
        (1u64 << rem) - 1
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

/// Sample `num_shots` BTS shots against a cached sparse parity matrix on the
/// GPU.
///
/// Mirrors the contract of `sample_bts_meas_major`: produces a `Vec<u64>` in
/// measurement-major layout of length `num_meas * s_words` where
/// `s_words = num_shots.div_ceil(64)`. Deterministic rows (empty
/// `row_cols`) come back zero-initialised then XOR-flipped with `ref_bits`,
/// matching the CPU path exactly.
///
/// Shot batches are streamed through device memory in groups of
/// `CHUNK_SHOTS` to keep peak VRAM bounded. The parity CSR arrays and
/// `ref_bits` stay resident inside `cache`; only the random-bit payload and
/// chunk output move each call.
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

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("bts_sample_meas_major")?;

    let mut output = vec![0u64; num_meas * s_words];
    let mut shots_done = 0usize;
    while shots_done < num_shots {
        let chunk_shots = (num_shots - shots_done).min(CHUNK_SHOTS);
        let chunk_s_words = chunk_shots.div_ceil(64);
        let chunk_random_len = chunk_s_words * rank;
        let chunk_output_len = num_meas * chunk_s_words;
        cache.ensure_chunk_capacity(ctx, chunk_random_len, chunk_output_len)?;
        cache.fill_random_bits(rng, rank, chunk_shots, chunk_s_words);

        {
            let mut random_bits_view = cache
                .random_bits_dev
                .raw_mut()
                .slice_mut(0..chunk_random_len.max(1));
            stream
                .memcpy_htod(
                    &cache.random_bits_host[..chunk_random_len],
                    &mut random_bits_view,
                )
                .map_err(|e| launch_err("upload bts random_bits", e))?;
        }

        let num_meas_i = require_i32("bts_sample_meas_major", "num_meas", num_meas)?;
        let chunk_s_words_i = require_i32("bts_sample_meas_major", "chunk_s_words", chunk_s_words)?;
        let rank_i = require_i32("bts_sample_meas_major", "rank", rank)?;
        let out_stride_words_i =
            require_i32("bts_sample_meas_major", "out_stride_words", chunk_s_words)?;
        let out_word_offset_i = 0i32;
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

        let random_bits_dev = cache
            .random_bits_dev
            .raw()
            .slice(0..chunk_random_len.max(1));
        let mut chunk_output_dev = cache
            .chunk_output_dev
            .raw_mut()
            .slice_mut(0..chunk_output_len.max(1));
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(cache.col_indices_dev.raw())
            .arg(cache.row_offsets_dev.raw())
            .arg(cache.ref_bits_dev.raw())
            .arg(&random_bits_dev)
            .arg(&num_meas_i)
            .arg(&chunk_s_words_i)
            .arg(&rank_i)
            .arg(&out_stride_words_i)
            .arg(&out_word_offset_i)
            .arg(&mut chunk_output_dev);
        // SAFETY: kernel signature matches the call shape. Each thread writes
        // one unique output word (indexed by m and batch), so there is no
        // inter-thread hazard on `chunk_output`. All input buffers are sized
        // to cover the access pattern above.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("bts_sample_meas_major", e))?;
        }

        stream
            .memcpy_dtoh(
                &chunk_output_dev,
                &mut cache.chunk_output_host[..chunk_output_len.max(1)],
            )
            .map_err(|e| launch_err("bts output dtoh", e))?;

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
    let stream = device.stream()?;
    let func = device.function("bts_sample_meas_major")?;
    let mut output_dev = GpuBuffer::<u64>::alloc_zeros(device, output_len.max(1))?;

    let mut shots_done = 0usize;
    while shots_done < num_shots {
        let chunk_shots = (num_shots - shots_done).min(CHUNK_SHOTS);
        let chunk_s_words = chunk_shots.div_ceil(64);
        let chunk_random_len = chunk_s_words * rank;
        cache.ensure_chunk_capacity(ctx, chunk_random_len, 1)?;
        cache.fill_random_bits(rng, rank, chunk_shots, chunk_s_words);

        {
            let mut random_bits_view = cache
                .random_bits_dev
                .raw_mut()
                .slice_mut(0..chunk_random_len.max(1));
            stream
                .memcpy_htod(
                    &cache.random_bits_host[..chunk_random_len],
                    &mut random_bits_view,
                )
                .map_err(|e| launch_err("upload bts random_bits", e))?;
        }

        let num_meas_i = require_i32("bts_sample_meas_major", "num_meas", num_meas)?;
        let chunk_s_words_i = require_i32("bts_sample_meas_major", "chunk_s_words", chunk_s_words)?;
        let rank_i = require_i32("bts_sample_meas_major", "rank", rank)?;
        let out_stride_words_i = require_i32("bts_sample_meas_major", "out_stride_words", s_words)?;
        let out_word_offset_i =
            require_i32("bts_sample_meas_major", "out_word_offset", shots_done / 64)?;
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

        let random_bits_dev = cache
            .random_bits_dev
            .raw()
            .slice(0..chunk_random_len.max(1));
        let mut output_view = output_dev.raw_mut().slice_mut(0..output_len.max(1));
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(cache.col_indices_dev.raw())
            .arg(cache.row_offsets_dev.raw())
            .arg(cache.ref_bits_dev.raw())
            .arg(&random_bits_dev)
            .arg(&num_meas_i)
            .arg(&chunk_s_words_i)
            .arg(&rank_i)
            .arg(&out_stride_words_i)
            .arg(&out_word_offset_i)
            .arg(&mut output_view);
        // SAFETY: kernel signature matches the call shape. Each thread writes
        // one unique output word addressed by measurement row and chunk-local
        // batch index.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("bts_sample_meas_major", e))?;
        }

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
    let cfg = LaunchConfig {
        grid_dim: (m_words_grid, s_words_grid, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw())
        .arg(&num_meas_i)
        .arg(&num_shots_i)
        .arg(&s_words_i)
        .arg(&m_words_i)
        .arg(shot_major.raw_mut());
    // SAFETY: grid covers every (m_word, batch) tile exactly once; each tile's
    // 64 threads produce 64 disjoint shot-major u64s. Shared tile is block-local.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_transpose_meas_to_shot", e))?;
    }
    Ok(shot_major)
}

/// Fields shared by both noise-apply variants. Keeps the host-mask upload
/// path and the fused device-generator path from drifting on their common
/// arguments.
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

pub(crate) struct NoiseMaskApply<'a> {
    pub base: NoiseApplyBase<'a>,
    pub x_masks: &'a GpuBuffer<u64>,
    pub z_masks: &'a GpuBuffer<u64>,
}

/// Arguments for the per-(row, batch) fused noise kernel. Each thread walks
/// one row's entry list in the row-major event CSR and XORs the generated
/// masks into a register, avoiding the cross-block `atomicXor` contention an
/// event-major launch pays on rows touched by many events. `row_event_offsets`
/// is length `num_meas + 1`; each `row_event_entries` entry is
/// `event << 2 | flag` (bit 0 = X, bit 1 = Z).
#[cfg(feature = "gpu")]
pub(crate) struct NoiseDeviceGenApplyByRow<'a> {
    pub meas_major: &'a mut GpuBuffer<u64>,
    pub num_meas: usize,
    pub s_words: usize,
    pub word_offset: usize,
    pub chunk_s_words: usize,
    pub row_event_offsets: &'a GpuBuffer<u32>,
    pub row_event_entries: &'a GpuBuffer<u32>,
    pub event_thresholds: &'a GpuBuffer<u64>,
    pub master_seed: u64,
    pub batch_offset: u64,
}

pub(crate) fn generate_and_apply_noise_masks_meas_major_by_row(
    ctx: &GpuContext,
    args: NoiseDeviceGenApplyByRow<'_>,
) -> Result<()> {
    let NoiseDeviceGenApplyByRow {
        meas_major,
        num_meas,
        s_words,
        word_offset,
        chunk_s_words,
        row_event_offsets,
        row_event_entries,
        event_thresholds,
        master_seed,
        batch_offset,
    } = args;
    if num_meas == 0 || s_words == 0 || chunk_s_words == 0 {
        return Ok(());
    }

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("bts_generate_and_apply_noise_meas_major_by_row")?;
    let num_meas_i = require_i32(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "num_meas",
        num_meas,
    )?;
    let s_words_i = require_i32(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "s_words",
        s_words,
    )?;
    let word_offset_i = require_i32(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "word_offset",
        word_offset,
    )?;
    let chunk_s_words_i = require_i32(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "chunk_s_words",
        chunk_s_words,
    )?;
    let block_threads = noise_block_threads(chunk_s_words);
    let batch_blocks = div_ceil_grid(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "chunk_s_words",
        chunk_s_words,
        block_threads,
    )?;
    let num_meas_grid = require_u32(
        "bts_generate_and_apply_noise_meas_major_by_row",
        "num_meas",
        num_meas,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_meas_grid, batch_blocks, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw_mut())
        .arg(&num_meas_i)
        .arg(&s_words_i)
        .arg(&word_offset_i)
        .arg(row_event_offsets.raw())
        .arg(row_event_entries.raw())
        .arg(event_thresholds.raw())
        .arg(&chunk_s_words_i)
        .arg(&master_seed)
        .arg(&batch_offset);
    // SAFETY: one thread per (row, batch); register accumulator replaces the
    // cross-block atomic path. No two threads write the same output index.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_generate_and_apply_noise_meas_major_by_row", e))?;
    }
    Ok(())
}

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

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("bts_apply_noise_masks_meas_major")?;
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
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_apply_noise_masks_meas_major", e))?;
    }
    Ok(())
}

/// Sample `num_shots` BTS shots on the GPU and return them in **shot-major**
/// layout (`Vec<u64>` of length `num_shots * m_words`). The transpose from
/// the native meas-major sampling layout runs on device, eliminating the
/// host `into_shot_major_data()` bit-transpose from noisy workflows.
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
    let stream = device.stream()?;
    let func = device.function("bts_popcount_rows")?;
    let mut row_counts = GpuBuffer::<u64>::alloc_zeros(device, num_meas.max(1))?;
    let mut host_counts = vec![0u64; num_meas];

    let num_meas_i = require_i32("bts_popcount_rows", "num_meas", num_meas)?;
    let s_words_i = require_i32("bts_popcount_rows", "s_words", s_words)?;
    let num_meas_grid = require_u32("bts_popcount_rows", "num_meas", num_meas)?;
    let tail_mask_u64 = tail_mask(num_shots);
    let cfg = LaunchConfig {
        grid_dim: (num_meas_grid, 1, 1),
        block_dim: (POPCOUNT_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw())
        .arg(&num_meas_i)
        .arg(&s_words_i)
        .arg(&tail_mask_u64)
        .arg(row_counts.raw_mut());
    // SAFETY: one block owns one measurement row and writes one unique count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_popcount_rows", e))?;
    }

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
    let stream = device.stream()?;
    let func = device.function("bts_count_used_slots")?;
    let mut used_out = GpuBuffer::<u32>::alloc_zeros(device, 1)?;
    let table_capacity_i = require_i32("bts_count_used_slots", "table_slots", table_slots)?;
    let blocks = div_ceil_grid(
        "bts_count_used_slots",
        "table_slots",
        table_slots,
        COUNT_HASH_BLOCK_SIZE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (COUNT_HASH_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(slot_states.raw())
        .arg(&table_capacity_i)
        .arg(used_out.raw_mut());
    // SAFETY: each block scans a disjoint subset of indices and contributes
    // its local sum through one atomic add into `used_out`.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_count_used_slots", e))?;
    }

    let mut used_host = [0u32; 1];
    used_out.copy_to_host(device, &mut used_host)?;
    Ok(used_host[0] as usize)
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
    let compact_cfg = LaunchConfig {
        grid_dim: (compact_blocks, 1, 1),
        block_dim: (COUNT_COMPACT_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

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
        compact_builder
            .launch(compact_cfg)
            .map_err(|e| launch_err("bts_compact_counts_upto8", e))?;
    }

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

    let Some(table_slots) = plan_count_table_slots(ctx, num_shots, m_words, rank)? else {
        return Ok(None);
    };

    let device = ctx.device();
    let mut slot_keys = GpuBuffer::<u64>::alloc_zeros(device, (table_slots * m_words).max(1))?;
    let mut slot_counts = GpuBuffer::<u64>::alloc_zeros(device, table_slots.max(1))?;
    let mut slot_states = GpuBuffer::<u32>::alloc_zeros(device, table_slots.max(1))?;
    let overflow = GpuBuffer::<u32>::alloc_zeros(device, 1)?;

    let stream = device.stream()?;
    let func = device.function("bts_count_shot_major_upto8")?;
    let num_shots_i = require_i32("bts_count_shot_major_upto8", "num_shots", num_shots)?;
    let m_words_i = require_i32("bts_count_shot_major_upto8", "m_words", m_words)?;
    let table_mask_u32 =
        require_u32("bts_count_shot_major_upto8", "table_slots", table_slots)?.saturating_sub(1);
    let blocks = div_ceil_grid(
        "bts_count_shot_major_upto8",
        "num_shots",
        num_shots,
        COUNT_HASH_BLOCK_SIZE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (COUNT_HASH_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(shot_major.raw())
        .arg(&num_shots_i)
        .arg(&m_words_i)
        .arg(slot_keys.raw_mut())
        .arg(slot_counts.raw_mut())
        .arg(slot_states.raw_mut())
        .arg(&table_mask_u32)
        .arg(overflow.raw());
    // SAFETY: each thread handles at most one shot-major outcome and uses
    // atomics to claim or update hash slots.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_count_shot_major_upto8", e))?;
    }

    let mut overflow_host = [0u32; 1];
    overflow.copy_to_host(device, &mut overflow_host)?;
    if overflow_host[0] != 0 {
        return Ok(None);
    }

    compact_count_table(
        ctx,
        &slot_keys,
        &slot_counts,
        &slot_states,
        table_slots,
        m_words,
        raw_transfer_bytes,
    )
}

fn try_count_meas_major_direct(
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
    let raw_transfer_bytes = count_raw_transfer_bytes(num_meas.saturating_mul(s_words));
    if !should_try_device_count(num_shots, m_words, rank, raw_transfer_bytes) {
        return Ok(None);
    }

    let Some(table_slots) = plan_count_table_slots(ctx, num_shots, m_words, rank)? else {
        return Ok(None);
    };

    let device = ctx.device();
    let mut slot_keys = GpuBuffer::<u64>::alloc_zeros(device, (table_slots * m_words).max(1))?;
    let mut slot_counts = GpuBuffer::<u64>::alloc_zeros(device, table_slots.max(1))?;
    let mut slot_states = GpuBuffer::<u32>::alloc_zeros(device, table_slots.max(1))?;
    let overflow = GpuBuffer::<u32>::alloc_zeros(device, 1)?;

    let stream = device.stream()?;
    let func = device.function("bts_count_meas_major_upto8")?;
    let num_meas_i = require_i32("bts_count_meas_major_upto8", "num_meas", num_meas)?;
    let num_shots_i = require_i32("bts_count_meas_major_upto8", "num_shots", num_shots)?;
    let s_words_i = require_i32("bts_count_meas_major_upto8", "s_words", s_words)?;
    let m_words_i = require_i32("bts_count_meas_major_upto8", "m_words", m_words)?;
    let table_mask_u32 =
        require_u32("bts_count_meas_major_upto8", "table_slots", table_slots)?.saturating_sub(1);
    let batches = div_ceil_grid(
        "bts_count_meas_major_upto8",
        "num_shots",
        num_shots,
        COUNT_MEAS_BATCH_SIZE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (batches.max(1), 1, 1),
        block_dim: (COUNT_MEAS_BATCH_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder
        .arg(meas_major.raw())
        .arg(&num_meas_i)
        .arg(&num_shots_i)
        .arg(&s_words_i)
        .arg(&m_words_i)
        .arg(slot_keys.raw_mut())
        .arg(slot_counts.raw_mut())
        .arg(slot_states.raw_mut())
        .arg(&table_mask_u32)
        .arg(overflow.raw());
    // SAFETY: each thread handles at most one logical shot and uses atomics to
    // claim or update hash slots.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("bts_count_meas_major_upto8", e))?;
    }

    let mut overflow_host = [0u32; 1];
    overflow.copy_to_host(device, &mut overflow_host)?;
    if overflow_host[0] != 0 {
        return Ok(None);
    }

    compact_count_table(
        ctx,
        &slot_keys,
        &slot_counts,
        &slot_states,
        table_slots,
        m_words,
        raw_transfer_bytes,
    )
}

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
    {
        if let Some(counts) = try_count_shot_major(
            ctx,
            &shot_major,
            num_shots,
            m_words,
            rank,
            raw_transfer_bytes,
        )? {
            return Ok(Some(counts));
        }
    }

    try_count_meas_major_direct(ctx, meas_major, num_meas, num_shots, m_words, s_words, rank)
}
