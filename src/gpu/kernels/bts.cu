
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
    unsigned long long tail_mask,
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
    if (batch == s_words - 1) acc &= tail_mask;

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


// Open-addressing insert-or-increment for a 3-state (empty / claiming / ready)
// hash table. Spins on slots mid-claim, linear-probes on collision, and sets
// `overflow` when a full sweep finds no usable slot.
//
// The mid-claim spin is bounded because pre-Volta SIMT gives the claiming lane
// no forward-progress guarantee against a spinning lane of the same warp.
// Exhausting the bound sets `overflow`, which sends counting back to the host.
#define BTS_CLAIM_SPIN_LIMIT 4096U

__device__ __forceinline__ void hash_insert_count(
    const unsigned long long *key,
    int m_words,
    unsigned long long *slot_keys,
    unsigned long long *slot_counts,
    unsigned int *slot_states,
    unsigned int table_mask,
    unsigned int *overflow
) {
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
            unsigned int spins = 0U;
            while ((state = load_state(slot_states + slot)) == 1U) {
                if (++spins >= BTS_CLAIM_SPIN_LIMIT) {
                    atomicExch(overflow, 1U);
                    return;
                }
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

    hash_insert_count(key, m_words, slot_keys, slot_counts, slot_states, table_mask, overflow);
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

    hash_insert_count(key, m_words, slot_keys, slot_counts, slot_states, table_mask, overflow);
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

