//! Stabilizer tableau kernels. CUDA C source compiled to PTX at runtime, plus launch
//! helpers in Rust.
//!
//! Tableau layout mirrors the CPU `StabilizerBackend`
//! (`src/backend/stabilizer/mod.rs`):
//!
//! - `xz`: `(2n+1)` rows × `2 * num_words` u64 words per row. Word ordering per row is
//!   X-bits in `[0, num_words)` then Z-bits in `[num_words, 2*num_words)`.
//! - `phase`: `(2n+1)` bytes, one per row (0 = +1, 1 = -1). Bytewise rather than
//!   bit-packed so rowmul phase writes do not require atomic RMW.
//! - Scratch row sits at index `2n` and is used only during measurement.
//!
//! Every entry-point name is prefixed `stab_` so it cannot collide with the dense
//! statevector kernels when both sources are concatenated into a single PTX module.
//!
//! Landed kernels:
//!
//! - `stab_set_initial_tableau`: identity init.
//! - `stab_apply_batch`: batched dispatch of all eleven Clifford gates
//!   (H, S, Sdg, X, Y, Z, SX, SXdg, CX, CZ, SWAP) over a host-provided
//!   op list, one kernel launch per flush.
//! - `stab_rowmul_words`: XOR source row into destination row with
//!   Aaronson-Gottesman phase update.
//! - `stab_measure_find_pivot`, `stab_measure_cascade`,
//!   `stab_measure_fixup`, `stab_measure_deterministic`: on-device Z-basis
//!   measurement. Eliminates the tableau copy-back previously needed per
//!   measure or reset.
//!
//! `stab_apply_batch` uses a one-block-per-row strategy across the full
//! `2n+1`-row tableau. Threads stripe over independent word groups inside the
//! row, which keeps reads and writes local to one row instead of striding the
//! same word across many rows. The cross-word tail stays serial on thread 0,
//! because different ops may still touch different bits in the same packed
//! u64 word.
//!
//! `stab_rowmul_words` launches a single block per call; threads partition
//! the `num_words` word loop and reduce their per-word phase contributions
//! via warp-shuffle plus shared memory.
//!
//! Measurement orchestrates four small kernels: a pivot search with an
//! atomicMin sentinel, a cascade that rowmul's the pivot into every row
//! carrying an X at the target (one block per row, most blocks early-exit),
//! a single-block fixup that moves pivot data into the paired destabilizer
//! and installs the measured Z_q, and a deterministic-branch kernel that
//! serialises rowmul's of stabilisers whose paired destabilisers anticommute
//! with Z_q into the scratch row and reads its phase.

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use crate::error::{PrismError, Result};

use super::super::{GpuContext, GpuTableau};

const BLOCK_SIZE: u32 = 128;

/// Stabilizer CUDA C source. Returned by `kernel_source()` and concatenated into the
/// combined PTX module alongside the dense kernels. No template substitutions needed:
/// all tableau shapes are passed as kernel arguments rather than compile-time constants.
const KERNEL_SOURCE: &str = r#"
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
// Commutation discipline (enforced host-side before the launch, mirrors
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
// src/backend/stabilizer/kernels/simd.rs:86 exactly:
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
"#;

/// Return the stabilizer CUDA C source for concatenation into the shared PTX module.
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

/// Initialise a freshly-allocated `GpuTableau` to the identity tableau: destabilizer
/// rows are X_i, stabilizer rows are Z_i, scratch row is all zero, phase is all zero.
///
/// Assumes `xz` and `phase` were allocated via `GpuBuffer::alloc_zeros` (so everything
/// else is already zero); this kernel only writes the identity bits.
pub(crate) fn launch_set_initial_tableau(ctx: &GpuContext, tableau: &mut GpuTableau) -> Result<()> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_set_initial_tableau")?;

    let n_usize = tableau.num_qubits();
    if n_usize == 0 {
        return Ok(());
    }
    let n = require_i32("stab_set_initial_tableau", "num_qubits", n_usize)?;
    let nw = require_i32("stab_set_initial_tableau", "num_words", tableau.num_words())?;
    let blocks = div_ceil_grid(
        "stab_set_initial_tableau",
        "num_qubits",
        n_usize,
        BLOCK_SIZE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    let xz = tableau.xz_mut().raw_mut();
    builder.arg(xz).arg(&n).arg(&nw);
    // SAFETY: kernel signature is (u64*, i32, i32); xz buffer is at least
    // (2n+1) * 2 * num_words u64s and each thread writes two disjoint words.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("stab_set_initial_tableau", e))?;
    }
    Ok(())
}

/// Clifford opcodes consumed by `stab_apply_batch`. Values are part of the ABI
/// between the host queue and the batch kernel and must stay in sync with the
/// switch inside the kernel source.
pub(crate) mod op {
    pub const H: u32 = 0;
    pub const S: u32 = 1;
    pub const SDG: u32 = 2;
    pub const X: u32 = 3;
    pub const Y: u32 = 4;
    pub const Z: u32 = 5;
    pub const SX: u32 = 6;
    pub const SXDG: u32 = 7;
    pub const CX: u32 = 8;
    pub const CZ: u32 = 9;
    pub const SWAP: u32 = 10;
}

/// Number of u32 slots per queued op: `[opcode, a, b, pad]`.
pub(crate) const CLIFOP_STRIDE: usize = 4;

const ZERO_U32: [u32; 1] = [0];

/// Apply a batch of queued Clifford ops to the device tableau in a single
/// launch.
///
/// `ops` is a flat `u32` buffer of length `CLIFOP_STRIDE * num_ops` laid out
/// as `[opcode, a, b, pad]` quads. Opcodes are the constants in [`op`]. The
/// kernel maps one block to each tableau row, parallelises the disjoint
/// same-word groups within that row, and leaves the cross-word tail serial
/// on thread 0 to avoid shared-word races. Apply a flat op list to the device
/// tableau via the word-grouped kernel.
///
/// Host-side, this streams through `ops` (quads `[opcode, a, b, pad]`),
/// sorting into word groups keyed by target-word plus a cross-word 2q list.
/// Conflicts between a newly-enqueued op and the running cross-word qubit
/// set trigger a partial launch, mirroring the CPU `flush_all_with_cross_word`
/// discipline in `src/backend/stabilizer/kernels/batch.rs`. The kernel then
/// amortises memory traffic across every op in a group: one `rx[w]`/`rz[w]`
/// read and one write per thread per group regardless of how many ops the
/// group contains.
pub(crate) fn launch_clifford_batch(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    ops: &[u32],
    scratch: &mut CliffordBatchScratch,
) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    debug_assert!(
        ops.len() % CLIFOP_STRIDE == 0,
        "ClifOp buffer length must be a multiple of {CLIFOP_STRIDE}"
    );
    let num_rows = tableau.total_rows();
    let num_words = tableau.num_words();
    if num_rows == 0 || num_words == 0 {
        return Ok(());
    }
    scratch.launch_ops(ctx, tableau, ops)
}

/// Reusable host and device scratch for GPU Clifford batch launches.
#[derive(Default)]
pub(crate) struct CliffordBatchScratch {
    num_words: usize,
    /// Per-word queued ops, flat `[opcode, a, b, pad]` quads.
    per_word_ops: Vec<Vec<u32>>,
    /// Cross-word 2q ops, flat quads in insertion order.
    cross_word: Vec<u32>,
    /// Bitmask per word of qubits already touched by a cross-word op in the
    /// current launch window. A new 1q or same-word 2q gate on a qubit in
    /// this mask forces a flush to preserve ingestion order.
    cross_word_qubits: Vec<u64>,
    group_words: Vec<u32>,
    group_offsets: Vec<u32>,
    ops_flat: Vec<u32>,
    group_words_dev: Option<CudaSlice<u32>>,
    group_offsets_dev: Option<CudaSlice<u32>>,
    ops_dev: Option<CudaSlice<u32>>,
    cross_dev: Option<CudaSlice<u32>>,
}

impl CliffordBatchScratch {
    pub(crate) fn clear(&mut self) {
        let used = self.num_words.min(self.per_word_ops.len());
        for v in &mut self.per_word_ops[..used] {
            v.clear();
        }
        self.cross_word.clear();
        let cross_used = self.cross_word_qubits.len().min(self.num_words);
        self.cross_word_qubits[..cross_used].fill(0);
        self.group_words.clear();
        self.group_offsets.clear();
        self.ops_flat.clear();
    }

    fn prepare(&mut self, num_words: usize) {
        self.num_words = num_words;
        if self.per_word_ops.len() < num_words {
            self.per_word_ops.resize_with(num_words, Vec::new);
        } else if self.per_word_ops.len() > num_words {
            self.per_word_ops.truncate(num_words);
        }
        self.cross_word_qubits.resize(num_words, 0);
        self.clear();
    }

    fn is_empty(&self) -> bool {
        self.cross_word.is_empty()
            && self.per_word_ops[..self.num_words]
                .iter()
                .all(|v| v.is_empty())
    }

    fn would_conflict(&self, opcode: u32, a: u32, b: u32) -> bool {
        if opcode <= op::SXDG {
            let w = (a as usize) >> 6;
            let bit = 1u64 << (a & 63);
            self.cross_word_qubits[w] & bit != 0
        } else {
            let aw = (a as usize) >> 6;
            let bw = (b as usize) >> 6;
            let abit = 1u64 << (a & 63);
            let bbit = 1u64 << (b & 63);
            if aw == bw {
                let bits = abit | bbit;
                self.cross_word_qubits[aw] & bits != 0
            } else {
                self.cross_word_qubits[aw] & abit != 0 || self.cross_word_qubits[bw] & bbit != 0
            }
        }
    }

    fn push(&mut self, opcode: u32, a: u32, b: u32) {
        if opcode <= op::SXDG {
            let w = (a as usize) >> 6;
            self.per_word_ops[w].extend_from_slice(&[opcode, a, b, 0]);
        } else {
            let aw = (a as usize) >> 6;
            let bw = (b as usize) >> 6;
            if aw == bw {
                self.per_word_ops[aw].extend_from_slice(&[opcode, a, b, 0]);
            } else {
                self.cross_word.extend_from_slice(&[opcode, a, b, 0]);
                self.cross_word_qubits[aw] |= 1u64 << (a & 63);
                self.cross_word_qubits[bw] |= 1u64 << (b & 63);
            }
        }
    }

    fn launch_pending(&mut self, ctx: &GpuContext, tableau: &mut GpuTableau) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        self.group_words.clear();
        self.group_offsets.clear();
        self.ops_flat.clear();
        self.group_offsets.push(0);
        for (w, v) in self.per_word_ops[..self.num_words].iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            debug_assert!(v.len() % CLIFOP_STRIDE == 0);
            self.group_words
                .push(require_u32("stab_apply_word_grouped", "group_word", w)?);
            self.ops_flat.extend_from_slice(v);
            let offset = self.ops_flat.len() / CLIFOP_STRIDE;
            self.group_offsets.push(require_u32(
                "stab_apply_word_grouped",
                "group_offset",
                offset,
            )?);
        }

        let num_groups = self.group_words.len();
        let num_cross_word = self.cross_word.len() / CLIFOP_STRIDE;
        launch_word_grouped_kernel(ctx, tableau, self, num_groups, num_cross_word)?;
        self.clear();
        Ok(())
    }

    fn launch_ops(
        &mut self,
        ctx: &GpuContext,
        tableau: &mut GpuTableau,
        ops: &[u32],
    ) -> Result<()> {
        self.prepare(tableau.num_words());
        for chunk in ops.chunks_exact(CLIFOP_STRIDE) {
            let opcode = chunk[0];
            let a = chunk[1];
            let b = chunk[2];
            if self.would_conflict(opcode, a, b) {
                self.launch_pending(ctx, tableau)?;
            }
            self.push(opcode, a, b);
        }
        self.launch_pending(ctx, tableau)
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_word_grouped_kernel(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    scratch: &mut CliffordBatchScratch,
    num_groups: usize,
    num_cross_word: usize,
) -> Result<()> {
    if num_groups == 0 && num_cross_word == 0 {
        return Ok(());
    }
    let num_rows_usize = tableau.total_rows();
    if num_rows_usize == 0 {
        return Ok(());
    }
    let num_rows = require_i32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let num_words_i = require_i32("stab_apply_word_grouped", "num_words", tableau.num_words())?;

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_apply_word_grouped")?;

    let group_words_src: &[u32] = if scratch.group_words.is_empty() {
        &ZERO_U32
    } else {
        &scratch.group_words
    };
    let group_offsets_src: &[u32] = if scratch.group_offsets.is_empty() {
        &ZERO_U32
    } else {
        &scratch.group_offsets
    };
    let ops_src: &[u32] = if scratch.ops_flat.is_empty() {
        &ZERO_U32
    } else {
        &scratch.ops_flat
    };
    let cross_src: &[u32] = if scratch.cross_word.is_empty() {
        &ZERO_U32
    } else {
        &scratch.cross_word
    };

    let ensure_u32_buffer =
        |slot: &mut Option<CudaSlice<u32>>, len: usize, op: &str| -> Result<()> {
            let needed = len.max(1);
            if slot.as_ref().map_or(true, |buf| buf.len() < needed) {
                *slot = Some(
                    stream
                        .alloc_zeros::<u32>(needed)
                        .map_err(|e| launch_err(op, e))?,
                );
            }
            Ok(())
        };
    ensure_u32_buffer(
        &mut scratch.group_words_dev,
        group_words_src.len(),
        "alloc group_words",
    )?;
    ensure_u32_buffer(
        &mut scratch.group_offsets_dev,
        group_offsets_src.len(),
        "alloc group_offsets",
    )?;
    ensure_u32_buffer(&mut scratch.ops_dev, ops_src.len(), "alloc ops_flat")?;
    ensure_u32_buffer(&mut scratch.cross_dev, cross_src.len(), "alloc cross_word")?;

    {
        let dev = scratch
            .group_words_dev
            .as_mut()
            .expect("group_words_dev allocated above");
        let mut view = dev.slice_mut(0..group_words_src.len());
        stream
            .memcpy_htod(group_words_src, &mut view)
            .map_err(|e| launch_err("upload group_words", e))?;
    }
    {
        let dev = scratch
            .group_offsets_dev
            .as_mut()
            .expect("group_offsets_dev allocated above");
        let mut view = dev.slice_mut(0..group_offsets_src.len());
        stream
            .memcpy_htod(group_offsets_src, &mut view)
            .map_err(|e| launch_err("upload group_offsets", e))?;
    }
    {
        let dev = scratch.ops_dev.as_mut().expect("ops_dev allocated above");
        let mut view = dev.slice_mut(0..ops_src.len());
        stream
            .memcpy_htod(ops_src, &mut view)
            .map_err(|e| launch_err("upload ops_flat", e))?;
    }
    {
        let dev = scratch
            .cross_dev
            .as_mut()
            .expect("cross_dev allocated above");
        let mut view = dev.slice_mut(0..cross_src.len());
        stream
            .memcpy_htod(cross_src, &mut view)
            .map_err(|e| launch_err("upload cross_word", e))?;
    }

    let num_groups_i = require_i32("stab_apply_word_grouped", "num_groups", num_groups)?;
    let num_cross_word_i =
        require_i32("stab_apply_word_grouped", "num_cross_word", num_cross_word)?;
    let block_threads = num_groups
        .next_power_of_two()
        .clamp(32, BLOCK_SIZE as usize) as u32;
    let num_rows_grid = require_u32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows_grid, 1, 1),
        block_dim: (block_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let group_words_dev = scratch
        .group_words_dev
        .as_ref()
        .expect("group_words_dev uploaded above")
        .slice(0..group_words_src.len());
    let group_offsets_dev = scratch
        .group_offsets_dev
        .as_ref()
        .expect("group_offsets_dev uploaded above")
        .slice(0..group_offsets_src.len());
    let ops_dev = scratch
        .ops_dev
        .as_ref()
        .expect("ops_dev uploaded above")
        .slice(0..ops_src.len());
    let cross_dev = scratch
        .cross_dev
        .as_ref()
        .expect("cross_dev uploaded above")
        .slice(0..cross_src.len());

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_rows)
        .arg(&num_words_i)
        .arg(&group_words_dev)
        .arg(&group_offsets_dev)
        .arg(&ops_dev)
        .arg(&num_groups_i)
        .arg(&cross_dev)
        .arg(&num_cross_word_i);
    // SAFETY: signature matches the kernel declaration. Each block owns one
    // row. Threads stripe over word-disjoint groups within that row, and
    // thread 0 alone handles the cross-word tail plus final phase write.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("stab_apply_word_grouped", e))?;
    }
    Ok(())
}

/// Block size for `stab_rowmul_words`. Chosen so a single warp-shuffle round
/// followed by one shared-memory reduction covers every supported num_words value
/// (≤ 5000 qubits ⇒ num_words ≤ 79 ⇒ one thread per word fits in a block).
const ROWMUL_BLOCK_SIZE: u32 = 128;

/// XOR `src_row` into `dst_row` and update `dst_row`'s phase per the
/// Aaronson-Gottesman g-function. Launched as a single block per call.
pub(crate) fn launch_rowmul_words(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    src_row: usize,
    dst_row: usize,
) -> Result<()> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_rowmul_words")?;

    let nw_usize = tableau.num_words();
    if nw_usize == 0 {
        return Ok(());
    }
    let nw = require_i32("stab_rowmul_words", "num_words", nw_usize)?;
    let src_i = require_i32("stab_rowmul_words", "src_row", src_row)?;
    let dst_i = require_i32("stab_rowmul_words", "dst_row", dst_row)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (ROWMUL_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder.arg(xz).arg(phase).arg(&nw).arg(&src_i).arg(&dst_i);
    // SAFETY: signature (u64*, u8*, i32, i32, i32); single block operates on
    // one (src, dst) row pair. All threads of the block write to disjoint
    // words of dst_row and a single phase byte; no inter-block hazard.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("stab_rowmul_words", e))?;
    }
    Ok(())
}

/// Block size for the measurement kernels that use one block per row
/// (`stab_measure_cascade`) and the single-block measurement kernels
/// (`stab_measure_fixup`, `stab_measure_deterministic`). Chosen so one warp-
/// shuffle round plus one shared-memory reduction cover every num_words
/// encountered in practice (≤ 5000 qubits ⇒ num_words ≤ 79).
const MEASURE_BLOCK_SIZE: u32 = 128;

/// Scan stabilizer rows `n..2n` for the minimum row index whose X-bit at
/// `target` is set. Returns `Some(row)` when a pivot exists (random branch),
/// `None` when every stabilizer commutes with `Z_target` (deterministic
/// branch). One i32 d2h roundtrip on the sentinel.
pub(crate) fn launch_measure_find_pivot(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
) -> Result<Option<usize>> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(None);
    }
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_measure_find_pivot")?;

    let sentinel = require_i32(
        "stab_measure_find_pivot",
        "sentinel",
        2usize.saturating_mul(n),
    )?;
    let num_qubits_i = require_i32("stab_measure_find_pivot", "num_qubits", n)?;
    let nw = require_i32("stab_measure_find_pivot", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_find_pivot", "target", target)?;
    let blocks = div_ceil_grid(
        "stab_measure_find_pivot",
        "num_qubits",
        n,
        MEASURE_BLOCK_SIZE,
    )?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (MEASURE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut host_pivot = [0_i32; 1];
    {
        let (xz_buf, pivot_buf) = tableau.xz_pivot_mut();
        let xz = xz_buf.raw_mut();
        let out_pivot = pivot_buf.raw_mut();
        stream
            .memcpy_htod(&[sentinel], out_pivot)
            .map_err(|e| launch_err("reset find_pivot sentinel", e))?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(xz)
            .arg(&num_qubits_i)
            .arg(&nw)
            .arg(&target_i)
            .arg(&mut *out_pivot);
        // SAFETY: signature (const u64*, i32, i32, i32, int*). The kernel only
        // reads xz at (row, target-word) positions and atomicMin's into
        // out_pivot.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("stab_measure_find_pivot", e))?;
        }
        stream
            .memcpy_dtoh(out_pivot, &mut host_pivot)
            .map_err(|e| launch_err("find_pivot dtoh", e))?;
    }
    if host_pivot[0] >= sentinel {
        Ok(None)
    } else {
        Ok(Some(host_pivot[0] as usize))
    }
}

/// Rowmul the pivot row into every non-pivot row that carries an X-bit at
/// `target`. One block per non-scratch row; blocks not participating in the
/// cascade early-exit with negligible driver overhead.
pub(crate) fn launch_measure_cascade(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
    pivot_row: usize,
) -> Result<()> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(());
    }
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_measure_cascade")?;

    let num_qubits_i = require_i32("stab_measure_cascade", "num_qubits", n)?;
    let nw = require_i32("stab_measure_cascade", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_cascade", "target", target)?;
    let pivot_i = require_i32("stab_measure_cascade", "pivot_row", pivot_row)?;
    let blocks = require_u32("stab_measure_cascade", "num_rows", 2usize.saturating_mul(n))?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (MEASURE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_qubits_i)
        .arg(&nw)
        .arg(&target_i)
        .arg(&pivot_i);
    // SAFETY: signature (u64*, u8*, i32, i32, i32, i32). Each block owns a
    // unique destination row (blockIdx.x); pivot_row is read-only throughout.
    // No inter-block hazard on xz or phase because writes target disjoint
    // rows.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("stab_measure_cascade", e))?;
    }
    Ok(())
}

/// Post-cascade fixup: move pivot data into the paired destabiliser, install
/// `Z_target` with the measured outcome at the pivot row. Single block.
pub(crate) fn launch_measure_fixup(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
    pivot_row: usize,
    outcome: bool,
) -> Result<()> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(());
    }
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_measure_fixup")?;

    let num_qubits_i = require_i32("stab_measure_fixup", "num_qubits", n)?;
    let nw = require_i32("stab_measure_fixup", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_fixup", "target", target)?;
    let pivot_i = require_i32("stab_measure_fixup", "pivot_row", pivot_row)?;
    let outcome_u8: u8 = outcome as u8;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (MEASURE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_qubits_i)
        .arg(&nw)
        .arg(&target_i)
        .arg(&pivot_i)
        .arg(&outcome_u8);
    // SAFETY: signature (u64*, u8*, i32, i32, i32, i32, u8). Single block
    // writes pivot_row and destab_row disjointly; runs only after the cascade
    // launch completes (single stream, serial ordering).
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("stab_measure_fixup", e))?;
    }
    Ok(())
}

/// Deterministic branch: rowmul stabiliser rows `n+i` for every `i` whose
/// destabiliser has an X at `target` into the scratch row, then read back the
/// scratch row's phase as the measurement outcome. One u8 d2h roundtrip.
pub(crate) fn launch_measure_deterministic(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
) -> Result<bool> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(false);
    }
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_measure_deterministic")?;

    let num_qubits_i = require_i32("stab_measure_deterministic", "num_qubits", n)?;
    let nw = require_i32(
        "stab_measure_deterministic",
        "num_words",
        tableau.num_words(),
    )?;
    let target_i = require_i32("stab_measure_deterministic", "target", target)?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (MEASURE_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut host_out = [0u8; 1];
    {
        let (xz_buf, phase_buf, outcome_buf) = tableau.xz_phase_outcome_mut();
        let xz = xz_buf.raw_mut();
        let phase = phase_buf.raw_mut();
        let out_outcome = outcome_buf.raw_mut();
        stream
            .memcpy_htod(&[0u8], out_outcome)
            .map_err(|e| launch_err("reset deterministic outcome", e))?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(xz)
            .arg(phase)
            .arg(&num_qubits_i)
            .arg(&nw)
            .arg(&target_i)
            .arg(&mut *out_outcome);
        // SAFETY: signature (u64*, u8*, i32, i32, i32, u8*). Single block runs
        // a serial loop over i=0..n; scratch row (index 2n) is the only
        // destination.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("stab_measure_deterministic", e))?;
        }
        stream
            .memcpy_dtoh(out_outcome, &mut host_out)
            .map_err(|e| launch_err("deterministic outcome dtoh", e))?;
    }
    Ok(host_out[0] != 0)
}
