
// ============================================================================
// Stabilizer tableau kernels
// ============================================================================
//
// CPU reference: src/backend/stabilizer/mod.rs (init, layout) and
// src/backend/stabilizer/kernels/simd.rs (rowmul g-function).
//
// xz is laid out as (2n+1) rows × 2*num_words u64s per row, with X-bits in the
// first num_words words of each row and Z-bits in the second num_words words.
// phase is 2n+1 bytes.

extern "C" __global__ void stab_set_initial_tableau(
    unsigned long long *xz,
    int num_qubits,
    int num_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_qubits) return;

    int word_idx = i / 64;
    unsigned long long bit = 1ULL << (i % 64);
    int stride = 2 * num_words;

    // Destabilizer row i: X-bit i set in the X-half.
    xz[i * stride + word_idx] = bit;

    // Stabilizer row (num_qubits + i): Z-bit i set in the Z-half.
    xz[(num_qubits + i) * stride + num_words + word_idx] = bit;
}

// ============================================================================
// Word-grouped Clifford dispatch
// ============================================================================
//
// `stab_apply_word_grouped` mirrors the CPU word-group batching pattern in
// `src/backend/stabilizer/kernels/batch.rs`. The host sorts queued ops into
// groups keyed by the target word (and accumulates cross-word 2q gates
// separately). Inside the kernel each block owns one tableau row:
//
//   1. Threads stripe over the word-group list. Each thread loads one target
//      word, applies every op in that group against register-held `(xw, zw)`,
//      then writes the word back. Groups are word-disjoint, so this is safe.
//   2. The block XOR-reduces the per-group phase contributions.
//   3. Thread 0 walks the cross-word list serially. Those ops commute on
//      qubits but may still touch different bits in the same packed word, so
//      serial execution avoids shared-word RMW races.
//   4. Thread 0 writes the final row phase byte once at the end.
//
// Opcodes (kept in sync with ClifOp in src/backend/stabilizer/mod.rs):
//   0  H     1  S    2  Sdg   3  X    4  Y    5  Z    6  SX   7  SXdg
//   8  Cx   9  Cz   10  Swap
//
// Commutation discipline (enforced on the host before the launch, mirrors
// `apply_instructions_word_batch` in the CPU path):
//   - Different word groups always commute on a row (disjoint target words
//     mean disjoint qubits), so they can be applied in word-index order.
//   - A cross-word op commutes with earlier-issued word-group ops only when
//     their qubits are disjoint. The host tracks a per-word "cross-word
//     qubit mask" and issues a partial launch as soon as a newly-enqueued
//     op would violate the invariant.
//
// Layout:
//   group_words[g]     : word index for group g (strictly increasing).
//   group_offsets[g]   : start of group g's ops in ops_flat. Length
//                         num_groups + 1; group_offsets[num_groups] is the
//                         total op count.
//   ops_flat[4*i ..]   : quad (opcode, a, b, pad) for op i. a and b are
//                         absolute qubit indices; the kernel masks them with
//                         63 to get the bit position within the group's word.
//   cross_word_flat    : quad (opcode, a, b, pad) for each cross-word 2q op,
//                         in insertion order.
__device__ __forceinline__ unsigned int stab_reduce_phase_xor(
    unsigned int local_xor,
    unsigned int tid,
    unsigned int bsz,
    unsigned int *warp_xors
) {
    for (int off = 16; off > 0; off /= 2) {
        local_xor ^= __shfl_down_sync(0xffffffffu, local_xor, off);
    }
    unsigned int lane = tid & 31u;
    unsigned int warp = tid >> 5;
    if (lane == 0u) warp_xors[warp] = local_xor;
    __syncthreads();
    if (warp == 0u) {
        unsigned int nwarps = (bsz + 31u) >> 5;
        unsigned int s = (lane < nwarps) ? warp_xors[lane] : 0u;
        for (int off = 16; off > 0; off /= 2) {
            s ^= __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0u) warp_xors[0] = s;
    }
    __syncthreads();
    return warp_xors[0];
}

extern "C" __global__ void stab_apply_word_grouped(
    unsigned long long *xz,
    unsigned char *phase,
    int num_rows,
    int num_words,
    const unsigned int *group_words,
    const unsigned int *group_offsets,
    const unsigned int *ops_flat,
    int num_groups,
    const unsigned int *cross_word_flat,
    int num_cross_word
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;
    unsigned int tid = threadIdx.x;
    unsigned int bsz = blockDim.x;
    int stride = 2 * num_words;
    unsigned long long *rx = xz + row * stride;
    unsigned long long *rz = rx + num_words;
    unsigned int local_phase_xor = 0u;

    for (int g = (int)tid; g < num_groups; g += (int)bsz) {
        int w = (int)group_words[g];
        unsigned int start = group_offsets[g];
        unsigned int end = group_offsets[g + 1];
        unsigned long long xw = rx[w];
        unsigned long long zw = rz[w];
        unsigned int p = 0u;

        // Poor-man's SGI. When both words are zero, every op in the group
        // reads zero bits, applies identity, and would write zero back.
        // Skip the inner loop and the write entirely. At sparse tableaus
        // (freshly initialised or shallow circuits) this eliminates the
        // bulk of per-row work.
        if ((xw | zw) == 0ULL) {
            continue;
        }

        for (unsigned int i = start; i < end; ++i) {
            unsigned int opcode = ops_flat[i * 4];
            unsigned int a = ops_flat[i * 4 + 1];
            unsigned int b = ops_flat[i * 4 + 2];
            unsigned int abit = a & 63u;
            unsigned long long mask_a = 1ULL << abit;

            switch (opcode) {
                case 0: {
                    unsigned long long x = xw & mask_a;
                    unsigned long long z = zw & mask_a;
                    if (x != 0ULL && z != 0ULL) p ^= 1u;
                    xw = (xw & ~mask_a) | z;
                    zw = (zw & ~mask_a) | x;
                    break;
                }
                case 1: {
                    unsigned long long x = xw & mask_a;
                    unsigned long long z = zw & mask_a;
                    if (x != 0ULL && z != 0ULL) p ^= 1u;
                    zw ^= x;
                    break;
                }
                case 2: {
                    unsigned long long x = xw & mask_a;
                    zw ^= x;
                    unsigned long long z_new = zw & mask_a;
                    if (x != 0ULL && z_new != 0ULL) p ^= 1u;
                    break;
                }
                case 3: {
                    if ((zw & mask_a) != 0ULL) p ^= 1u;
                    break;
                }
                case 4: {
                    unsigned long long x = xw & mask_a;
                    unsigned long long z = zw & mask_a;
                    if ((x ^ z) != 0ULL) p ^= 1u;
                    break;
                }
                case 5: {
                    if ((xw & mask_a) != 0ULL) p ^= 1u;
                    break;
                }
                case 6: {
                    unsigned long long x = xw & mask_a;
                    unsigned long long z = zw & mask_a;
                    if (z != 0ULL && x == 0ULL) p ^= 1u;
                    xw ^= z;
                    break;
                }
                case 7: {
                    unsigned long long x = xw & mask_a;
                    unsigned long long z = zw & mask_a;
                    if (x != 0ULL && z != 0ULL) p ^= 1u;
                    xw ^= z;
                    break;
                }
                case 8: {
                    unsigned int bbit = b & 63u;
                    unsigned long long mask_b = 1ULL << bbit;
                    unsigned long long xa = (xw >> abit) & 1ULL;
                    unsigned long long za = (zw >> abit) & 1ULL;
                    unsigned long long xb = (xw >> bbit) & 1ULL;
                    unsigned long long zb = (zw >> bbit) & 1ULL;
                    if ((xa & zb & (xb ^ za ^ 1ULL)) == 1ULL) p ^= 1u;
                    if (xa == 1ULL) xw ^= mask_b;
                    if (zb == 1ULL) zw ^= mask_a;
                    break;
                }
                case 9: {
                    unsigned int bbit = b & 63u;
                    unsigned long long mask_b = 1ULL << bbit;
                    unsigned long long xa = (xw >> abit) & 1ULL;
                    unsigned long long xb = (xw >> bbit) & 1ULL;
                    unsigned long long za = (zw >> abit) & 1ULL;
                    unsigned long long zb = (zw >> bbit) & 1ULL;
                    if ((xa & xb & (za ^ zb)) == 1ULL) p ^= 1u;
                    if (xb == 1ULL) zw ^= mask_a;
                    if (xa == 1ULL) zw ^= mask_b;
                    break;
                }
                case 10: {
                    unsigned int bbit = b & 63u;
                    unsigned long long mask_b = 1ULL << bbit;
                    unsigned long long xa = (xw >> abit) & 1ULL;
                    unsigned long long xb = (xw >> bbit) & 1ULL;
                    if (xa != xb) { xw ^= mask_a; xw ^= mask_b; }
                    unsigned long long za = (zw >> abit) & 1ULL;
                    unsigned long long zb = (zw >> bbit) & 1ULL;
                    if (za != zb) { zw ^= mask_a; zw ^= mask_b; }
                    break;
                }
                default: break;
            }
        }

        rx[w] = xw;
        rz[w] = zw;
        local_phase_xor ^= p;
    }

    __shared__ unsigned int warp_xors[32];
    unsigned int phase_xor = stab_reduce_phase_xor(local_phase_xor, tid, bsz, warp_xors);
    if (tid != 0u) {
        return;
    }

    unsigned char p = phase[row] ^ (unsigned char)(phase_xor & 1u);
    for (int c = 0; c < num_cross_word; ++c) {
        unsigned int opcode = cross_word_flat[c * 4];
        unsigned int a = cross_word_flat[c * 4 + 1];
        unsigned int b = cross_word_flat[c * 4 + 2];
        int aw = (int)(a >> 6);
        int abit = (int)(a & 63u);
        unsigned long long mask_a = 1ULL << abit;
        int bw = (int)(b >> 6);
        int bbit = (int)(b & 63u);
        unsigned long long mask_b = 1ULL << bbit;
        // Poor-man's SGI for cross-word 2q. When all four target words are
        // zero, neither qubit is active in this row and the op is a no-op.
        if ((rx[aw] | rz[aw] | rx[bw] | rz[bw]) == 0ULL) {
            continue;
        }
        unsigned long long xa = (rx[aw] >> abit) & 1ULL;
        unsigned long long za = (rz[aw] >> abit) & 1ULL;
        unsigned long long xb = (rx[bw] >> bbit) & 1ULL;
        unsigned long long zb = (rz[bw] >> bbit) & 1ULL;
        switch (opcode) {
            case 8: {
                if ((xa & zb & (xb ^ za ^ 1ULL)) == 1ULL) p ^= 1;
                if (xa == 1ULL) rx[bw] ^= mask_b;
                if (zb == 1ULL) rz[aw] ^= mask_a;
                break;
            }
            case 9: {
                if ((xa & xb & (za ^ zb)) == 1ULL) p ^= 1;
                if (xb == 1ULL) rz[aw] ^= mask_a;
                if (xa == 1ULL) rz[bw] ^= mask_b;
                break;
            }
            case 10: {
                if (xa != xb) { rx[aw] ^= mask_a; rx[bw] ^= mask_b; }
                if (za != zb) { rz[aw] ^= mask_a; rz[bw] ^= mask_b; }
                break;
            }
            default: break;
        }
    }

    phase[row] = p;
}

// ============================================================================
// rowmul_words
// ============================================================================
//
// XOR source row bits into destination row and update destination phase by
// the Aaronson-Gottesman g-function. Mirrors the scalar CPU implementation at
// rowmul_words_scalar in src/backend/stabilizer/kernels/simd.rs exactly:
//
//     let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
//     let pos = (x1 & z1 & !x2 & z2)
//             | (x1 & !z1 & x2 & z2)
//             | (!x1 & z1 & x2 & !z2);
//     sum += 2 * popcount(pos)
//     sum -= popcount(nonzero)
//
// The per-word sum contributions are aggregated modulo 4 via unsigned wrapping
// arithmetic, matching CPU `u64::wrapping_add` / `wrapping_sub`. Phase update:
//     phase[dst] = ((sum + 2*src_phase + 2*dst_phase) & 3) >= 2.
//
// Launched as a single block per call. Threads partition the word loop by
// `blockDim.x` stride. Reduction proceeds in two stages:
//   1. Per-warp via __shfl_down_sync.
//   2. Across warps via __shared__ memory + a single-warp reduction.
// No cross-block reduction needed; measurement cascades issue one kernel
// launch per (src, dst) pair, so each rowmul is a distinct launch.
// Device helper shared by `stab_rowmul_words`, `stab_measure_cascade`, and
// `stab_measure_deterministic`. XORs `src_row` into `dst_row` word by word
// and computes the g-function phase contribution. Returns the contribution
// modulo 4 on every thread (broadcast via `warp_sums[0]`). Callers supply a
// 32-slot __shared__ buffer and handle the final phase byte update.
__device__ __forceinline__ unsigned long long stab_rowmul_block(
    unsigned long long *xz,
    int num_words,
    int src_row,
    int dst_row,
    int tid,
    int bsz,
    unsigned long long *warp_sums
) {
    int stride = 2 * num_words;
    unsigned long long *src_x = xz + src_row * stride;
    unsigned long long *src_z = src_x + num_words;
    unsigned long long *dst_x = xz + dst_row * stride;
    unsigned long long *dst_z = dst_x + num_words;

    unsigned long long local_sum = 0ULL;
    for (int w = tid; w < num_words; w += bsz) {
        unsigned long long x1 = src_x[w];
        unsigned long long z1 = src_z[w];
        unsigned long long x2 = dst_x[w];
        unsigned long long z2 = dst_z[w];
        unsigned long long new_x = x1 ^ x2;
        unsigned long long new_z = z1 ^ z2;
        dst_x[w] = new_x;
        dst_z[w] = new_z;

        if ((x1 | z1 | x2 | z2) != 0ULL) {
            unsigned long long nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
            unsigned long long pos = (x1 & z1 & ~x2 & z2)
                                   | (x1 & ~z1 & x2 & z2)
                                   | (~x1 & z1 & x2 & ~z2);
            local_sum += 2ULL * (unsigned long long)__popcll((long long)pos);
            local_sum -= (unsigned long long)__popcll((long long)nonzero);
        }
    }

    for (int off = 16; off > 0; off /= 2) {
        local_sum += __shfl_down_sync(0xffffffffu, local_sum, off);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        int nwarps = (bsz + 31) >> 5;
        unsigned long long s = (lane < nwarps) ? warp_sums[lane] : 0ULL;
        for (int off = 16; off > 0; off /= 2) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0) warp_sums[0] = s;
    }
    __syncthreads();
    return warp_sums[0];
}

extern "C" __global__ void stab_rowmul_words(
    unsigned long long *xz,
    unsigned char *phase,
    int num_words,
    int src_row,
    int dst_row
) {
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    __shared__ unsigned long long warp_sums[32];
    unsigned long long sum =
        stab_rowmul_block(xz, num_words, src_row, dst_row, tid, bsz, warp_sums);
    if (tid == 0) {
        unsigned long long initial = 2ULL * (unsigned long long)phase[src_row]
                                    + 2ULL * (unsigned long long)phase[dst_row];
        unsigned long long total = (sum + initial) & 3ULL;
        phase[dst_row] = (total >= 2ULL) ? 1 : 0;
    }
}

// ============================================================================
// On-device Z-basis measurement
// ============================================================================
//
// Host orchestration (src/backend/stabilizer/mod.rs:apply_measure_gpu):
//   1. Launch stab_measure_find_pivot, dtoh the 1 i32 sentinel.
//   2. If pivot < 2n (random branch):
//        a. Launch stab_measure_cascade (one block per row, 2n blocks).
//        b. Launch stab_measure_fixup (single block).
//      else (deterministic branch):
//        a. Launch stab_measure_deterministic (single block).
//        b. dtoh the 1 u8 outcome.
//
// Random-branch RNG is picked on the host so no cuRAND wiring is required.
// Reset reuses the same path and queues an X onto the Clifford batch queue
// when the outcome is 1.

// Scan stabilizer rows n..2n for the minimum row index with an X-bit at
// `target`. `out_pivot` must be initialised to a sentinel >= 2n on the host;
// atomicMin leaves the sentinel untouched when no row carries the X-bit.
extern "C" __global__ void stab_measure_find_pivot(
    const unsigned long long *xz,
    int num_qubits,
    int num_words,
    int target,
    int *out_pivot
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_qubits) return;
    int row = num_qubits + i;
    int stride = 2 * num_words;
    int word_idx = target / 64;
    unsigned long long bit = 1ULL << (target % 64);
    if ((xz[row * stride + word_idx] & bit) != 0ULL) {
        atomicMin(out_pivot, row);
    }
}

// Rowmul `pivot_row` into every row in [0, 2n) that carries an X-bit at
// `target`, skipping `pivot_row` itself. One block per candidate row; blocks
// that do not carry the X-bit early-exit. `pivot_row` is read-only throughout
// (the fixup kernel runs afterwards and is the only writer).
extern "C" __global__ void stab_measure_cascade(
    unsigned long long *xz,
    unsigned char *phase,
    int num_qubits,
    int num_words,
    int target,
    int pivot_row
) {
    int r = blockIdx.x;
    int total_rows = 2 * num_qubits;
    if (r >= total_rows) return;
    if (r == pivot_row) return;

    int stride = 2 * num_words;
    int word_idx = target / 64;
    unsigned long long bit = 1ULL << (target % 64);
    if ((xz[r * stride + word_idx] & bit) == 0ULL) return;

    int tid = threadIdx.x;
    int bsz = blockDim.x;
    __shared__ unsigned long long warp_sums[32];
    unsigned long long sum =
        stab_rowmul_block(xz, num_words, pivot_row, r, tid, bsz, warp_sums);
    if (tid == 0) {
        unsigned long long initial = 2ULL * (unsigned long long)phase[pivot_row]
                                    + 2ULL * (unsigned long long)phase[r];
        unsigned long long total = (sum + initial) & 3ULL;
        phase[r] = (total >= 2ULL) ? 1 : 0;
    }
}

// After the cascade: copy pivot row into the paired destabiliser row
// (`pivot_row - num_qubits`), zero the pivot row, set its Z-bit at `target`,
// and store the measured outcome in the pivot's phase. Single-block launch
// so thread 0 can hand-off the pre-zero phase value via shared memory.
extern "C" __global__ void stab_measure_fixup(
    unsigned long long *xz,
    unsigned char *phase,
    int num_qubits,
    int num_words,
    int target,
    int pivot_row,
    unsigned char outcome
) {
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int stride = 2 * num_words;
    int destab_row = pivot_row - num_qubits;

    __shared__ unsigned char pivot_phase_saved;
    if (tid == 0) pivot_phase_saved = phase[pivot_row];
    __syncthreads();

    for (int w = tid; w < stride; w += bsz) {
        unsigned long long pv = xz[pivot_row * stride + w];
        xz[destab_row * stride + w] = pv;
        xz[pivot_row * stride + w] = 0ULL;
    }
    __syncthreads();

    if (tid == 0) {
        phase[destab_row] = pivot_phase_saved;
        phase[pivot_row] = outcome;
        int word_idx = target / 64;
        unsigned long long bit = 1ULL << (target % 64);
        xz[pivot_row * stride + num_words + word_idx] = bit;
    }
}

// Deterministic branch: build the scratch row at index 2n by serially
// rowmul'ing stabiliser row n+i into it whenever destabiliser row i carries
// an X-bit at `target`. Single block; the serial loop keeps scratch's phase
// consistent across rowmul iterations. Thread 0 writes the scratch phase to
// `out_outcome` on exit.
extern "C" __global__ void stab_measure_deterministic(
    unsigned long long *xz,
    unsigned char *phase,
    int num_qubits,
    int num_words,
    int target,
    unsigned char *out_outcome
) {
    int tid = threadIdx.x;
    int bsz = blockDim.x;
    int scratch = 2 * num_qubits;
    int stride = 2 * num_words;

    for (int w = tid; w < stride; w += bsz) {
        xz[scratch * stride + w] = 0ULL;
    }
    if (tid == 0) phase[scratch] = 0;
    __syncthreads();

    int word_idx = target / 64;
    unsigned long long bit = 1ULL << (target % 64);
    __shared__ unsigned long long warp_sums[32];

    for (int i = 0; i < num_qubits; ++i) {
        if ((xz[i * stride + word_idx] & bit) == 0ULL) continue;
        int src_row = num_qubits + i;
        unsigned long long sum =
            stab_rowmul_block(xz, num_words, src_row, scratch, tid, bsz, warp_sums);
        if (tid == 0) {
            unsigned long long initial = 2ULL * (unsigned long long)phase[src_row]
                                        + 2ULL * (unsigned long long)phase[scratch];
            unsigned long long total = (sum + initial) & 3ULL;
            phase[scratch] = (total >= 2ULL) ? 1 : 0;
        }
        __syncthreads();
    }

    if (tid == 0) *out_outcome = phase[scratch];
}
