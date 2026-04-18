//! Dense statevector kernels. CUDA C source compiled to PTX at runtime, plus launch
//! helpers in Rust.
//!
//! The state buffer is `2 * 2^n` f64s laid out as interleaved (re, im) pairs matching
//! `num_complex::Complex64` and CUDA's `double2` builtin. All kernels take the buffer as
//! `double2 *` for 16-byte aligned vector loads.
//!
//! # Fused-gate strategy
//!
//! `BatchPhase`, `BatchRzz`, `DiagonalBatch`, and the `all_diagonal` arm of `MultiFused`
//! are handled by dedicated batched kernels that take precomputed per-group phase LUTs
//! (built on the host by the corresponding CPU `build_*_tables` helper) plus small
//! metadata arrays (shifts / q0s / q1s / lens). One kernel launch per fused instruction
//! instead of one launch per sub-gate.
//!
//! The non-diagonal arm of `MultiFused` uses a shared-memory tiled kernel
//! (`apply_multi_fused_tiled`) for sub-gates whose target lies inside the tile, with
//! per-gate fallback launches for targets outside the tile. `Multi2q` still decomposes
//! on the host to one launch per sub-gate; rare in practice and tracked as follow-up.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::{PrismError, Result};

use super::super::{GpuContext, GpuState};

const BLOCK_SIZE: u32 = 256;

/// PTX source template. Placeholders like `{{BP_TABLE_SIZE}}` are substituted when the
/// device is constructed (see [`kernel_source`]) so the kernel's compile-time constants
/// track the CPU constants in [`crate::backend::statevector::kernels`]. Adding a new
/// placeholder requires matching entries in `kernel_source` below.
const KERNEL_SOURCE_TEMPLATE: &str = r#"
// Template constants substituted at device construction from the Rust constants.
#define TILE_Q          {{TILE_Q}}
#define TILE_SIZE       {{TILE_SIZE}}
#define BP_TABLE_SIZE   {{BP_TABLE_SIZE}}
#define BP_GROUP_SIZE   {{BP_GROUP_SIZE}}
#define BR_TABLE_SIZE   {{BR_TABLE_SIZE}}
#define BR_GROUP_SIZE   {{BR_GROUP_SIZE}}
#define DB_TABLE_SIZE   {{DB_TABLE_SIZE}}
#define DB_MAX_QUBITS   {{DB_MAX_QUBITS}}

// ============================================================================
// Initialisation
// ============================================================================

extern "C" __global__ void set_initial_state(double2 *state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        state[0] = make_double2(1.0, 0.0);
    }
}

// ============================================================================
// Single-qubit gate: generic 2x2
// ============================================================================
//
// Launch: 2^(n-1) threads. Each thread handles one (lo, hi) amplitude pair.

extern "C" __global__ void apply_gate_1q(
    double2 *state, unsigned long long pair_count, int target,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;

    unsigned long long mask = (1ULL << target) - 1;
    unsigned long long i0 = ((k & ~mask) << 1) | (k & mask);
    unsigned long long i1 = i0 | (1ULL << target);

    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = m00r*a.x - m00i*a.y + m01r*b.x - m01i*b.y;
    state[i0].y = m00r*a.y + m00i*a.x + m01r*b.y + m01i*b.x;
    state[i1].x = m10r*a.x - m10i*a.y + m11r*b.x - m11i*b.y;
    state[i1].y = m10r*a.y + m10i*a.x + m11r*b.y + m11i*b.x;
}

// Diagonal 2x2 specialisation.
// state[i0] *= d0, state[i1] *= d1 — no cross terms.

extern "C" __global__ void apply_diagonal_1q(
    double2 *state, unsigned long long pair_count, int target,
    double d0r, double d0i, double d1r, double d1i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;

    unsigned long long mask = (1ULL << target) - 1;
    unsigned long long i0 = ((k & ~mask) << 1) | (k & mask);
    unsigned long long i1 = i0 | (1ULL << target);

    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = d0r*a.x - d0i*a.y;
    state[i0].y = d0r*a.y + d0i*a.x;
    state[i1].x = d1r*b.x - d1i*b.y;
    state[i1].y = d1r*b.y + d1i*b.x;
}

// ============================================================================
// Two-qubit gates (CX / CZ / SWAP)
// ============================================================================
//
// All take `pair_count = 2^(n-2)` threads. Each thread computes a compressed index
// and expands via chained insert_zero_bit (q0, q1 sorted).

__device__ inline unsigned long long expand_2q(unsigned long long k, int lo_q, int hi_q) {
    unsigned long long lo_mask = (1ULL << lo_q) - 1;
    unsigned long long lo = k & lo_mask;
    unsigned long long mid_hi = k >> lo_q;
    mid_hi = (mid_hi << 1);                                 // insert 0 at lo_q
    unsigned long long base = (mid_hi << lo_q) | lo;        // reassemble with gap at lo_q
    // second insertion at hi_q (already accounts for +1 shift from first)
    unsigned long long hi_mask = (1ULL << hi_q) - 1;
    unsigned long long low = base & hi_mask;
    unsigned long long high = base >> hi_q;
    return (high << (hi_q + 1)) | low;
}

extern "C" __global__ void apply_cx(
    double2 *state, unsigned long long pair_count, int control, int target)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i0 = idx | (1ULL << control);
    unsigned long long i1 = i0 | (1ULL << target);
    double2 tmp = state[i0];
    state[i0] = state[i1];
    state[i1] = tmp;
}

extern "C" __global__ void apply_cz(
    double2 *state, unsigned long long pair_count, int q0, int q1)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i = idx | (1ULL << q0) | (1ULL << q1);
    state[i].x = -state[i].x;
    state[i].y = -state[i].y;
}

extern "C" __global__ void apply_swap(
    double2 *state, unsigned long long pair_count, int q0, int q1)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i01 = idx | (1ULL << q0);
    unsigned long long i10 = idx | (1ULL << q1);
    double2 tmp = state[i01];
    state[i01] = state[i10];
    state[i10] = tmp;
}

// Parity-dependent phase (Rzz and DiagEntry::Parity2q).
// state[i] *= same when ((i>>q0) ^ (i>>q1)) & 1 == 0, else state[i] *= diff.
// Launch over 2^n threads.

extern "C" __global__ void apply_parity_phase(
    double2 *state, unsigned long long dim, int q0, int q1,
    double same_r, double same_i, double diff_r, double diff_i)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    unsigned long long parity = ((i >> q0) ^ (i >> q1)) & 1ULL;
    double pr = parity ? diff_r : same_r;
    double pi = parity ? diff_i : same_i;
    double2 a = state[i];
    state[i].x = pr*a.x - pi*a.y;
    state[i].y = pr*a.y + pi*a.x;
}

// ============================================================================
// Controlled-unitary gates
// ============================================================================

extern "C" __global__ void apply_cu(
    double2 *state, unsigned long long pair_count, int control, int target,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i0 = idx | (1ULL << control);
    unsigned long long i1 = i0 | (1ULL << target);

    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = m00r*a.x - m00i*a.y + m01r*b.x - m01i*b.y;
    state[i0].y = m00r*a.y + m00i*a.x + m01r*b.y + m01i*b.x;
    state[i1].x = m10r*a.x - m10i*a.y + m11r*b.x - m11i*b.y;
    state[i1].y = m10r*a.y + m10i*a.x + m11r*b.y + m11i*b.x;
}

// Controlled-phase optimisation: state[both_set] *= phase.
// Launch over pair_count = 2^(n-2) threads. Only acts on ctrl=1, tgt=1.

extern "C" __global__ void apply_cu_phase(
    double2 *state, unsigned long long pair_count, int control, int target,
    double pr, double pi)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = control < target ? control : target;
    int hi_q = control < target ? target : control;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);
    unsigned long long i = idx | (1ULL << control) | (1ULL << target);
    double2 a = state[i];
    state[i].x = pr*a.x - pi*a.y;
    state[i].y = pr*a.y + pi*a.x;
}

// Multi-controlled unitary. `sorted` contains all controls + target, sorted ascending.
// `num_sorted = num_controls + 1`. `ctrl_mask` has bits set at control positions only;
// `tgt_mask = 1 << target`.
// Launch over 2^(n - num_sorted) threads.

extern "C" __global__ void apply_mcu(
    double2 *state, unsigned long long iter_count,
    const unsigned int *sorted, int num_sorted,
    unsigned long long ctrl_mask, unsigned long long tgt_mask,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = (int)sorted[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    unsigned long long i0 = idx | ctrl_mask;
    unsigned long long i1 = i0 | tgt_mask;

    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = m00r*a.x - m00i*a.y + m01r*b.x - m01i*b.y;
    state[i0].y = m00r*a.y + m00i*a.x + m01r*b.y + m01i*b.x;
    state[i1].x = m10r*a.x - m10i*a.y + m11r*b.x - m11i*b.y;
    state[i1].y = m10r*a.y + m10i*a.x + m11r*b.y + m11i*b.x;
}

extern "C" __global__ void apply_mcu_phase(
    double2 *state, unsigned long long iter_count,
    const unsigned int *sorted, int num_sorted,
    unsigned long long all_mask,
    double pr, double pi)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = (int)sorted[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    unsigned long long i = idx | all_mask;
    double2 a = state[i];
    state[i].x = pr*a.x - pi*a.y;
    state[i].y = pr*a.y + pi*a.x;
}

// ============================================================================
// Fused 2q: generic 4x4 matrix. mat is row-major 16 Complex64 = 32 f64.
// Launch over pair_count = 2^(n-2) threads; each processes one 4-element group.
// ============================================================================

extern "C" __global__ void apply_fused_2q(
    double2 *state, unsigned long long pair_count, int q0, int q1,
    const double *mat)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);

    // Basis ordering matches CPU (src/backend/simd.rs PreparedGate2q::apply_full, line 1598):
    //   basis index b = (q0_bit << 1) | q1_bit   i.e. q1 is LSB of the 4-element basis.
    //   b=0 → (q0=0, q1=0); b=1 → (q0=0, q1=1); b=2 → (q0=1, q1=0); b=3 → (q0=1, q1=1).
    unsigned long long i00 = idx;
    unsigned long long i01 = idx | (1ULL << q1);
    unsigned long long i10 = idx | (1ULL << q0);
    unsigned long long i11 = idx | (1ULL << q0) | (1ULL << q1);

    double2 in[4] = {state[i00], state[i01], state[i10], state[i11]};
    unsigned long long indices[4] = {i00, i01, i10, i11};

    for (int row = 0; row < 4; ++row) {
        double rr = 0.0, ri = 0.0;
        for (int col = 0; col < 4; ++col) {
            // mat row-major: 32 f64s, row r col c → mat[2*(r*4+c)]=re, mat[2*(r*4+c)+1]=im
            double mr = mat[2*(row*4 + col)];
            double mi = mat[2*(row*4 + col) + 1];
            rr += mr*in[col].x - mi*in[col].y;
            ri += mr*in[col].y + mi*in[col].x;
        }
        state[indices[row]].x = rr;
        state[indices[row]].y = ri;
    }
}

// ============================================================================
// Measurement
// ============================================================================
//
// measure_prob_one: per-block reduction of sum(|amp|^2) over elements where qubit bit is 1.
// Each block reduces 2*BLOCK_SIZE elements (using shared memory). Host sums the
// out_partials array afterward.

extern "C" __global__ void measure_prob_one(
    const double2 *state, unsigned long long dim, int qubit, double *out_partials)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    double s = 0.0;
    if (i < dim && ((i >> qubit) & 1ULL)) {
        double2 a = state[i];
        s += a.x*a.x + a.y*a.y;
    }
    unsigned long long i2 = i + blockDim.x;
    if (i2 < dim && ((i2 >> qubit) & 1ULL)) {
        double2 a = state[i2];
        s += a.x*a.x + a.y*a.y;
    }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_partials[blockIdx.x] = sdata[0];
}

// measure_collapse: zero amplitudes where qubit bit != outcome.
// Launch over 2^n threads, block size BLOCK_SIZE.

extern "C" __global__ void measure_collapse(
    double2 *state, unsigned long long dim, int qubit, int outcome)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    int bit = (int)((i >> qubit) & 1ULL);
    if (bit != outcome) {
        state[i].x = 0.0;
        state[i].y = 0.0;
    }
}

// compute_probabilities: out[i] = (amp.x² + amp.y²) * norm_sq for every basis state.
// Launched over 2^n threads. Saves half the PCIe traffic compared to dtoh'ing raw amplitudes
// and squaring host-side.

extern "C" __global__ void compute_probabilities(
    const double2 *state, unsigned long long dim, double norm_sq, double *out)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    out[i] = (a.x * a.x + a.y * a.y) * norm_sq;
}

// scale_state: state[i] *= s. Used to fold deferred pending_norm into the device buffer before
// export. Launched over 2^n threads.

extern "C" __global__ void scale_state(
    double2 *state, unsigned long long dim, double s)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    state[i].x *= s;
    state[i].y *= s;
}

// apply_multi_fused_tiled: batched non-diagonal MultiFused via shared-memory tiles.
//
// Each block loads a TILE_SIZE slice of amplitudes into shared memory, then applies every
// sub-gate whose target bit is inside the tile (target < TILE_Q) with no further global
// memory reads. Pairs (i0, i1) for a given gate stay within the tile because the target
// bit is a low bit of the global index; the high bits (> TILE_Q) are the block id.
//
// Data layout on device (uploaded per batch):
//   targets[g] : int, gate g's target qubit. All targets here satisfy target < TILE_Q.
//   matrices[g*4 .. g*4+4] : four double2 entries (m00, m01, m10, m11) for gate g.
//
// TILE_Q = 10, TILE_SIZE = 1024, block_size = 512 threads, each thread handles one pair.
// Shared memory usage: 1024 × 16 bytes = 16 KB. Pascal-friendly.
// (TILE_Q / TILE_SIZE are defined in the template header at the top of the file.)

extern "C" __global__ void apply_multi_fused_tiled(
    double2 *state, unsigned long long dim,
    const int *targets,
    const double2 *matrices,
    int num_gates)
{
    __shared__ double2 tile[TILE_SIZE];

    unsigned long long block_base = (unsigned long long)blockIdx.x * (unsigned long long)TILE_SIZE;
    if (block_base >= dim) return;

    int tid = (int)threadIdx.x;
    // Load the tile cooperatively, two amplitudes per thread (block_dim = TILE_SIZE/2).
    int a_off = tid;
    int b_off = tid + (TILE_SIZE / 2);
    tile[a_off] = state[block_base + a_off];
    tile[b_off] = state[block_base + b_off];
    __syncthreads();

    // Each thread owns one pair per gate. k is the compressed pair index (0 .. TILE_SIZE/2).
    int k = tid;

    for (int g = 0; g < num_gates; ++g) {
        int t = targets[g];
        int mask = (1 << t) - 1;
        int i0 = ((k & ~mask) << 1) | (k & mask);
        int i1 = i0 | (1 << t);

        double2 a = tile[i0];
        double2 b = tile[i1];

        double2 m00 = matrices[g * 4 + 0];
        double2 m01 = matrices[g * 4 + 1];
        double2 m10 = matrices[g * 4 + 2];
        double2 m11 = matrices[g * 4 + 3];

        // No sync needed between read and write here: each thread owns an
        // exclusive (i0, i1) pair within a single gate iteration (the compressed-
        // index mapping partitions [0, TILE_SIZE)). The trailing sync below
        // serialises writes across gate boundaries, which is the only hazard.

        tile[i0].x = m00.x * a.x - m00.y * a.y + m01.x * b.x - m01.y * b.y;
        tile[i0].y = m00.x * a.y + m00.y * a.x + m01.x * b.y + m01.y * b.x;
        tile[i1].x = m10.x * a.x - m10.y * a.y + m11.x * b.x - m11.y * b.y;
        tile[i1].y = m10.x * a.y + m10.y * a.x + m11.x * b.y + m11.y * b.x;
        __syncthreads();
    }

    // Store tile back to global memory.
    state[block_base + a_off] = tile[a_off];
    state[block_base + b_off] = tile[b_off];
}

// apply_diagonal_batch: applies a mixed batch of diagonal entries (1q / 2q / parity-2q)
// via precomputed per-group LUTs (built host-side by build_diagonal_batch_tables).
// Replaces per-entry dispatch. Launches 2^n threads; each thread folds all group phases.
//
// Table layout:
//   group_tables[g * DB_TABLE_SIZE + bits] = combined phase for bit pattern `bits` in group g
//   group_shifts[g * DB_MAX_QUBITS + j]    = qubit index of the j-th bit in group g

extern "C" __global__ void apply_diagonal_batch(
    double2 *state, unsigned long long dim,
    const double2 *group_tables,
    const int *group_shifts,
    const int *group_lens,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * DB_MAX_QUBITS;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((i >> shifts[j]) & 1ULL) << j);
        }
        double2 ph = group_tables[g * DB_TABLE_SIZE + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    double2 a = state[i];
    state[i].x = cr * a.x - ci * a.y;
    state[i].y = cr * a.y + ci * a.x;
}

// apply_batch_rzz: applies a batch of Rzz gates via precomputed parity-phase LUTs (built
// host-side by build_batch_rzz_tables). Replaces per-edge apply_parity_phase launches.
// Launches 2^n threads across the full state; each thread computes parity bits per edge
// per group, indexes the 256-entry LUT, chains multiplies.
//
// Table layout (row-major per group):
//   group_tables[g * BR_TABLE_SIZE + bits]        = combined phase for parity pattern `bits`
//   group_q0s[g * BR_GROUP_SIZE + k],
//   group_q1s[g * BR_GROUP_SIZE + k]               = qubit pair for the k-th edge in group g

extern "C" __global__ void apply_batch_rzz(
    double2 *state, unsigned long long dim,
    const double2 *group_tables,
    const int *group_q0s,
    const int *group_q1s,
    const int *group_lens,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *q0s = group_q0s + g * BR_GROUP_SIZE;
        const int *q1s = group_q1s + g * BR_GROUP_SIZE;
        int bits = 0;
        for (int k = 0; k < len; ++k) {
            bits |= (int)((((i >> q0s[k]) ^ (i >> q1s[k])) & 1ULL) << k);
        }
        double2 ph = group_tables[g * BR_TABLE_SIZE + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    double2 a = state[i];
    state[i].x = cr * a.x - ci * a.y;
    state[i].y = cr * a.y + ci * a.x;
}

// apply_batch_phase: applies a batch of controlled-phase gates sharing a control qubit via
// precomputed LUTs (built host-side by build_batch_phase_tables). Replaces per-phase
// launches of apply_cu_phase. One DRAM read/write per amplitude in the ctrl=1 subspace.
//
// Table layout (row-major per group, length MAX_BATCH_PHASE_GROUPS * BP_TABLE_SIZE):
//   group_tables[g * BP_TABLE_SIZE + bits] = combined phase for bit pattern `bits` in group g
//   group_shifts[g * BP_GROUP_SIZE + j]    = qubit index of the j-th bit in group g

extern "C" __global__ void apply_batch_phase(
    double2 *state, unsigned long long half_count,
    int control,
    const double2 *group_tables,
    const int *group_shifts,
    const int *group_lens,
    int num_groups)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_count) return;
    unsigned long long ctrl_mask = 1ULL << control;
    unsigned long long mask = ctrl_mask - 1ULL;
    unsigned long long idx = ((k & ~mask) << 1) | (k & mask) | ctrl_mask;

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * BP_GROUP_SIZE;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((idx >> shifts[j]) & 1ULL) << j);
        }
        double2 ph = group_tables[g * BP_TABLE_SIZE + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    double2 a = state[idx];
    state[idx].x = cr * a.x - ci * a.y;
    state[idx].y = cr * a.y + ci * a.x;
}

// apply_multi_fused_diagonal: batch of diagonal 1q gates in a single pass. Replaces the
// per-sub-gate host-side decomposition for `Gate::MultiFused { all_diagonal: true }`.
//
// Each thread handles one amplitude and folds all `num_gates` diagonal multiplications
// based on the bit pattern of its own index. One DRAM read + one DRAM write per thread;
// compute = num_gates complex multiplies (typically 1-10).
//
// Args: targets[g] (int); diags[2*g + 0] = d0 (double2), diags[2*g + 1] = d1 (double2).

extern "C" __global__ void apply_multi_fused_diagonal(
    double2 *state, unsigned long long dim,
    const int *targets,
    const double2 *diags,
    int num_gates)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    for (int g = 0; g < num_gates; ++g) {
        int t = targets[g];
        int bit = (int)((i >> t) & 1ULL);
        double2 p = diags[2 * g + bit];
        double nx = p.x * a.x - p.y * a.y;
        double ny = p.x * a.y + p.y * a.x;
        a.x = nx;
        a.y = ny;
    }
    state[i] = a;
}
"#;

/// Materialise the PTX source with CPU-side constants substituted into the template.
///
/// Called once per device, at `GpuDevice::new`. The substitution is the bridge between
/// the Rust constants in [`crate::backend::statevector::kernels`] (and the `MULTI_FUSED_*`
/// constants in this file) and the `#define`s at the top of [`KERNEL_SOURCE_TEMPLATE`].
/// Adding a kernel that depends on a new host-side constant: add a `{{PLACEHOLDER}}` to
/// the template header, add a matching `.replace(...)` call below, done.
pub(crate) fn kernel_source() -> String {
    use crate::backend::statevector::kernels as cpu_k;
    KERNEL_SOURCE_TEMPLATE
        .replace("{{TILE_Q}}", &MULTI_FUSED_TILE_Q.to_string())
        .replace("{{TILE_SIZE}}", &MULTI_FUSED_TILE_SIZE.to_string())
        .replace(
            "{{BP_TABLE_SIZE}}",
            &cpu_k::BATCH_PHASE_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{BP_GROUP_SIZE}}",
            &cpu_k::BATCH_PHASE_GROUP_SIZE.to_string(),
        )
        .replace(
            "{{BR_TABLE_SIZE}}",
            &cpu_k::BATCH_RZZ_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{BR_GROUP_SIZE}}",
            &cpu_k::BATCH_RZZ_GROUP_SIZE.to_string(),
        )
        .replace(
            "{{DB_TABLE_SIZE}}",
            &cpu_k::DIAG_BATCH_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{DB_MAX_QUBITS}}",
            &cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP.to_string(),
        )
}

// ============================================================================
// Rust-side launchers
// ============================================================================

fn launch_err(op: &str, err: impl std::fmt::Display) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {err}"),
    }
}

fn grid_for(count: u64) -> u32 {
    count.div_ceil(BLOCK_SIZE as u64).max(1) as u32
}

pub(crate) fn launch_set_initial_state(ctx: &GpuContext, state: &mut GpuState) -> Result<()> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("set_initial_state")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer);
    // SAFETY: kernel signature is (double2*); single-thread write within allocated range.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("set_initial_state", e))?;
    }
    Ok(())
}

pub(crate) fn launch_compute_probabilities(ctx: &GpuContext, state: &GpuState) -> Result<Vec<f64>> {
    use super::super::GpuBuffer;
    let n = state.num_qubits();
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("compute_probabilities")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    // Reuse the cached scratch buffer when large enough; grow if num_qubits increased.
    let mut scratch_slot = state.probs_scratch();
    if scratch_slot
        .as_ref()
        .map_or(true, |b| b.len() < dim as usize)
    {
        *scratch_slot = Some(GpuBuffer::<f64>::alloc_zeros(device, dim as usize)?);
    }
    let scratch = scratch_slot.as_mut().unwrap();

    let norm_sq = state.pending_norm() * state.pending_norm();
    let mut builder = stream.launch_builder(&func);
    let state_buf = state.buffer().raw();
    let out = scratch.raw_mut();
    builder.arg(state_buf).arg(&dim).arg(&norm_sq).arg(out);
    // SAFETY: signature matches; grid covers dim; scratch is at least dim elements.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("compute_probabilities", e))?;
    }
    let mut host = vec![0.0_f64; dim as usize];
    scratch.copy_to_host(device, &mut host)?;
    Ok(host)
}

#[allow(dead_code)]
pub(crate) fn launch_scale_state(ctx: &GpuContext, state: &mut GpuState, s: f64) -> Result<()> {
    let n = state.num_qubits();
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("scale_state")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&dim).arg(&s);
    // SAFETY: signature matches; grid covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("scale_state", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_gate_1q(
    ctx: &GpuContext,
    state: &mut GpuState,
    target: usize,
    matrix: [[Complex64; 2]; 2],
) -> Result<()> {
    let n = state.num_qubits();
    if target >= n {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 1);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_gate_1q")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let m00r = matrix[0][0].re;
    let m00i = matrix[0][0].im;
    let m01r = matrix[0][1].re;
    let m01i = matrix[0][1].im;
    let m10r = matrix[1][0].re;
    let m10i = matrix[1][0].im;
    let m11r = matrix[1][1].re;
    let m11i = matrix[1][1].im;
    let target_i = target as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&target_i)
        .arg(&m00r)
        .arg(&m00i)
        .arg(&m01r)
        .arg(&m01i)
        .arg(&m10r)
        .arg(&m10i)
        .arg(&m11r)
        .arg(&m11i);
    // SAFETY: signature matches kernel declaration; grid/block sized so threads <= pair_count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_gate_1q", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_diagonal_1q(
    ctx: &GpuContext,
    state: &mut GpuState,
    target: usize,
    d0: Complex64,
    d1: Complex64,
) -> Result<()> {
    let n = state.num_qubits();
    if target >= n {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 1);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_diagonal_1q")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let target_i = target as i32;
    let d0r = d0.re;
    let d0i = d0.im;
    let d1r = d1.re;
    let d1i = d1.im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&target_i)
        .arg(&d0r)
        .arg(&d0i)
        .arg(&d1r)
        .arg(&d1i);
    // SAFETY: signature matches; grid sized to pair_count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_diagonal_1q", e))?;
    }
    Ok(())
}

fn launch_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    kernel: &str,
    q0: usize,
    q1: usize,
) -> Result<()> {
    let n = state.num_qubits();
    if q0 >= n || q1 >= n || q0 == q1 {
        return Err(PrismError::InvalidQubit {
            index: q0.max(q1),
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 2);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function(kernel)?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&pair_count).arg(&q0_i).arg(&q1_i);
    // SAFETY: signature matches kernel (state, pair_count, q0, q1); grid covers iter space.
    unsafe {
        builder.launch(cfg).map_err(|e| launch_err(kernel, e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_cx(
    ctx: &GpuContext,
    state: &mut GpuState,
    control: usize,
    target: usize,
) -> Result<()> {
    launch_2q(ctx, state, "apply_cx", control, target)
}

pub(crate) fn launch_apply_cz(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
) -> Result<()> {
    launch_2q(ctx, state, "apply_cz", q0, q1)
}

pub(crate) fn launch_apply_swap(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
) -> Result<()> {
    launch_2q(ctx, state, "apply_swap", q0, q1)
}

pub(crate) fn launch_apply_parity_phase(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    same: Complex64,
    diff: Complex64,
) -> Result<()> {
    let n = state.num_qubits();
    if q0 >= n || q1 >= n || q0 == q1 {
        return Err(PrismError::InvalidQubit {
            index: q0.max(q1),
            register_size: n,
        });
    }
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_parity_phase")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let sr = same.re;
    let si = same.im;
    let dr = diff.re;
    let di = diff.im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&sr)
        .arg(&si)
        .arg(&dr)
        .arg(&di);
    // SAFETY: signature matches kernel; dim covers whole state.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_parity_phase", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_rzz(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    theta: f64,
) -> Result<()> {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    // e^{-iθ/2} = cos - i sin (parity even: both bits same)
    // e^{iθ/2}  = cos + i sin (parity odd: bits differ)
    let same = Complex64::new(c, -s);
    let diff = Complex64::new(c, s);
    launch_apply_parity_phase(ctx, state, q0, q1, same, diff)
}

pub(crate) fn launch_apply_cu(
    ctx: &GpuContext,
    state: &mut GpuState,
    control: usize,
    target: usize,
    matrix: [[Complex64; 2]; 2],
) -> Result<()> {
    let n = state.num_qubits();
    if control >= n || target >= n || control == target {
        return Err(PrismError::InvalidQubit {
            index: control.max(target),
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 2);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_cu")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let ctrl_i = control as i32;
    let tgt_i = target as i32;
    let m00r = matrix[0][0].re;
    let m00i = matrix[0][0].im;
    let m01r = matrix[0][1].re;
    let m01i = matrix[0][1].im;
    let m10r = matrix[1][0].re;
    let m10i = matrix[1][0].im;
    let m11r = matrix[1][1].re;
    let m11i = matrix[1][1].im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&ctrl_i)
        .arg(&tgt_i)
        .arg(&m00r)
        .arg(&m00i)
        .arg(&m01r)
        .arg(&m01i)
        .arg(&m10r)
        .arg(&m10i)
        .arg(&m11r)
        .arg(&m11i);
    // SAFETY: signature matches kernel; grid covers pair_count.
    unsafe {
        builder.launch(cfg).map_err(|e| launch_err("apply_cu", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_cu_phase(
    ctx: &GpuContext,
    state: &mut GpuState,
    control: usize,
    target: usize,
    phase: Complex64,
) -> Result<()> {
    let n = state.num_qubits();
    if control >= n || target >= n || control == target {
        return Err(PrismError::InvalidQubit {
            index: control.max(target),
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 2);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_cu_phase")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let ctrl_i = control as i32;
    let tgt_i = target as i32;
    let pr = phase.re;
    let pi = phase.im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&ctrl_i)
        .arg(&tgt_i)
        .arg(&pr)
        .arg(&pi);
    // SAFETY: signature matches kernel; grid covers pair_count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_cu_phase", e))?;
    }
    Ok(())
}

fn validate_mcu_qubits(n: usize, controls: &[usize], target: usize) -> Result<Vec<u32>> {
    for &c in controls {
        if c >= n {
            return Err(PrismError::InvalidQubit {
                index: c,
                register_size: n,
            });
        }
        if c == target {
            return Err(PrismError::InvalidParameter {
                message: "control qubit equals target".to_string(),
            });
        }
    }
    if target >= n {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: n,
        });
    }
    let mut sorted: Vec<u32> = controls.iter().map(|&q| q as u32).collect();
    sorted.push(target as u32);
    sorted.sort_unstable();
    Ok(sorted)
}

pub(crate) fn launch_apply_mcu(
    ctx: &GpuContext,
    state: &mut GpuState,
    controls: &[usize],
    target: usize,
    matrix: [[Complex64; 2]; 2],
) -> Result<()> {
    let n = state.num_qubits();
    let sorted = validate_mcu_qubits(n, controls, target)?;
    let num_sorted = sorted.len() as i32;
    let iter_count: u64 = 1u64 << (n - sorted.len());
    let mut ctrl_mask: u64 = 0;
    for &c in controls {
        ctrl_mask |= 1u64 << c;
    }
    let tgt_mask: u64 = 1u64 << target;

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_mcu")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(iter_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let sorted_dev = stream
        .clone_htod(&sorted)
        .map_err(|e| launch_err("upload mcu sorted", e))?;
    let m00r = matrix[0][0].re;
    let m00i = matrix[0][0].im;
    let m01r = matrix[0][1].re;
    let m01i = matrix[0][1].im;
    let m10r = matrix[1][0].re;
    let m10i = matrix[1][0].im;
    let m11r = matrix[1][1].re;
    let m11i = matrix[1][1].im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&iter_count)
        .arg(&sorted_dev)
        .arg(&num_sorted)
        .arg(&ctrl_mask)
        .arg(&tgt_mask)
        .arg(&m00r)
        .arg(&m00i)
        .arg(&m01r)
        .arg(&m01i)
        .arg(&m10r)
        .arg(&m10i)
        .arg(&m11r)
        .arg(&m11i);
    // SAFETY: signature matches; sorted_dev lives until launch returns (kept in scope).
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_mcu", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_mcu_phase(
    ctx: &GpuContext,
    state: &mut GpuState,
    controls: &[usize],
    target: usize,
    phase: Complex64,
) -> Result<()> {
    let n = state.num_qubits();
    let sorted = validate_mcu_qubits(n, controls, target)?;
    let num_sorted = sorted.len() as i32;
    let iter_count: u64 = 1u64 << (n - sorted.len());
    let mut all_mask: u64 = 1u64 << target;
    for &c in controls {
        all_mask |= 1u64 << c;
    }

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_mcu_phase")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(iter_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let sorted_dev = stream
        .clone_htod(&sorted)
        .map_err(|e| launch_err("upload mcu_phase sorted", e))?;
    let pr = phase.re;
    let pi = phase.im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&iter_count)
        .arg(&sorted_dev)
        .arg(&num_sorted)
        .arg(&all_mask)
        .arg(&pr)
        .arg(&pi);
    // SAFETY: signature matches kernel; sorted_dev retained.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_mcu_phase", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_fused_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    matrix: &[[Complex64; 4]; 4],
) -> Result<()> {
    let n = state.num_qubits();
    if q0 >= n || q1 >= n || q0 == q1 {
        return Err(PrismError::InvalidQubit {
            index: q0.max(q1),
            register_size: n,
        });
    }
    let pair_count: u64 = 1u64 << (n - 2);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_fused_2q")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(pair_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut flat = [0.0_f64; 32];
    for row in 0..4 {
        for col in 0..4 {
            flat[2 * (row * 4 + col)] = matrix[row][col].re;
            flat[2 * (row * 4 + col) + 1] = matrix[row][col].im;
        }
    }
    let mat_dev = stream
        .clone_htod(&flat)
        .map_err(|e| launch_err("upload fused_2q mat", e))?;
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&mat_dev);
    // SAFETY: signature matches; mat_dev retained until after launch.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_fused_2q", e))?;
    }
    Ok(())
}

pub(crate) fn measure_prob_one(ctx: &GpuContext, state: &GpuState, qubit: usize) -> Result<f64> {
    let n = state.num_qubits();
    if qubit >= n {
        return Err(PrismError::InvalidQubit {
            index: qubit,
            register_size: n,
        });
    }
    let dim: u64 = 1u64 << n;
    // Each block processes 2*BLOCK_SIZE elements.
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = dim.div_ceil(elems_per_block).max(1) as u32;

    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("measure_prob_one")?;
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
    };
    let mut partials = stream
        .alloc_zeros::<f64>(num_blocks as usize)
        .map_err(|e| launch_err("alloc partials", e))?;

    let qubit_i = qubit as i32;
    let mut builder = stream.launch_builder(&func);
    let state_buf = state.buffer().raw();
    builder
        .arg(state_buf)
        .arg(&dim)
        .arg(&qubit_i)
        .arg(&mut partials);
    // SAFETY: signature matches kernel; num_blocks × 2*BLOCK_SIZE covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("measure_prob_one", e))?;
    }
    let mut host_partials = vec![0.0_f64; num_blocks as usize];
    stream
        .memcpy_dtoh(&partials, &mut host_partials)
        .map_err(|e| launch_err("measure partials dtoh", e))?;
    let prob_raw: f64 = host_partials.iter().sum();
    let norm_sq = state.pending_norm() * state.pending_norm();
    Ok((prob_raw * norm_sq).clamp(0.0, 1.0))
}

pub(crate) fn measure_collapse(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    outcome: bool,
) -> Result<()> {
    let n = state.num_qubits();
    if qubit >= n {
        return Err(PrismError::InvalidQubit {
            index: qubit,
            register_size: n,
        });
    }
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("measure_collapse")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let qubit_i = qubit as i32;
    let outcome_i: i32 = if outcome { 1 } else { 0 };
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&dim).arg(&qubit_i).arg(&outcome_i);
    // SAFETY: signature matches kernel; grid covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("measure_collapse", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_multi_fused_diagonal(
    ctx: &GpuContext,
    state: &mut GpuState,
    gates: &[(usize, [[Complex64; 2]; 2])],
) -> Result<()> {
    if gates.is_empty() {
        return Ok(());
    }
    let n = state.num_qubits();
    for &(target, _) in gates {
        if target >= n {
            return Err(PrismError::InvalidQubit {
                index: target,
                register_size: n,
            });
        }
    }

    // Flatten to two device buffers: targets[i32; num_gates] and a packed `double2`
    // stream where [2*g + 0] = d0 = mat[0][0], [2*g + 1] = d1 = mat[1][1]. One allocation
    // per array, so two host-to-device copies per launch instead of five.
    let num_gates = gates.len();
    let mut targets: Vec<i32> = Vec::with_capacity(num_gates);
    let mut diags: Vec<f64> = Vec::with_capacity(num_gates * 4);
    for &(target, mat) in gates {
        targets.push(target as i32);
        diags.push(mat[0][0].re);
        diags.push(mat[0][0].im);
        diags.push(mat[1][1].re);
        diags.push(mat[1][1].im);
    }

    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_multi_fused_diagonal")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let targets_dev = stream
        .clone_htod(&targets)
        .map_err(|e| launch_err("upload targets", e))?;
    let diags_dev = stream
        .clone_htod(&diags)
        .map_err(|e| launch_err("upload diags", e))?;

    let num_gates_i = num_gates as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&targets_dev)
        .arg(&diags_dev)
        .arg(&num_gates_i);
    // SAFETY: signature matches; targets has num_gates i32s; diags has 2*num_gates double2s
    // (= 4*num_gates f64s); grid covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_multi_fused_diagonal", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_batch_phase(
    ctx: &GpuContext,
    state: &mut GpuState,
    control: usize,
    phases: &[(usize, Complex64)],
) -> Result<()> {
    use crate::backend::statevector::kernels as cpu_k;
    if phases.is_empty() {
        return Ok(());
    }
    let n = state.num_qubits();
    debug_assert!(
        n >= 1,
        "batch_phase requires at least one qubit for the control"
    );
    if control >= n {
        return Err(PrismError::InvalidQubit {
            index: control,
            register_size: n,
        });
    }
    for &(q, _) in phases {
        if q >= n {
            return Err(PrismError::InvalidQubit {
                index: q,
                register_size: n,
            });
        }
    }

    // Host-side: build the per-group LUTs using the CPU builder (reused as-is).
    let one = Complex64::new(1.0, 0.0);
    let mut groups = [cpu_k::BatchPhaseGroup {
        table: [one; cpu_k::BATCH_PHASE_TABLE_SIZE],
        shifts: [0; cpu_k::BATCH_PHASE_GROUP_SIZE],
        len: 0,
        pext_mask: 0,
    }; cpu_k::MAX_BATCH_PHASE_GROUPS];
    let num_groups = cpu_k::build_batch_phase_tables(phases, &mut groups);

    // Flatten to device-friendly arrays.
    let mut tables_flat: Vec<f64> =
        Vec::with_capacity(num_groups * cpu_k::BATCH_PHASE_TABLE_SIZE * 2);
    for group in groups.iter().take(num_groups) {
        for entry in &group.table {
            tables_flat.push(entry.re);
            tables_flat.push(entry.im);
        }
    }
    let mut shifts_flat: Vec<i32> = Vec::with_capacity(num_groups * cpu_k::BATCH_PHASE_GROUP_SIZE);
    for group in groups.iter().take(num_groups) {
        for &s in &group.shifts {
            shifts_flat.push(s as i32);
        }
    }
    let lens: Vec<i32> = groups
        .iter()
        .take(num_groups)
        .map(|g| g.len as i32)
        .collect();

    let half_count: u64 = 1u64 << (n - 1);
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_batch_phase")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(half_count), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let tables_dev = stream
        .clone_htod(&tables_flat)
        .map_err(|e| launch_err("upload batch_phase tables", e))?;
    let shifts_dev = stream
        .clone_htod(&shifts_flat)
        .map_err(|e| launch_err("upload batch_phase shifts", e))?;
    let lens_dev = stream
        .clone_htod(&lens)
        .map_err(|e| launch_err("upload batch_phase lens", e))?;

    let control_i = control as i32;
    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&half_count)
        .arg(&control_i)
        .arg(&tables_dev)
        .arg(&shifts_dev)
        .arg(&lens_dev)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; device slices sized for num_groups; grid covers half.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_batch_phase", e))?;
    }
    Ok(())
}

pub(crate) fn launch_apply_batch_rzz(
    ctx: &GpuContext,
    state: &mut GpuState,
    edges: &[(usize, usize, f64)],
) -> Result<()> {
    use crate::backend::statevector::kernels as cpu_k;
    if edges.is_empty() {
        return Ok(());
    }
    let n = state.num_qubits();
    for &(q0, q1, _) in edges {
        if q0 >= n || q1 >= n {
            return Err(PrismError::InvalidQubit {
                index: q0.max(q1),
                register_size: n,
            });
        }
    }

    let one = Complex64::new(1.0, 0.0);
    let mut groups = [cpu_k::BatchRzzGroup {
        table: [one; cpu_k::BATCH_RZZ_TABLE_SIZE],
        q0s: [0; cpu_k::BATCH_RZZ_GROUP_SIZE],
        q1s: [0; cpu_k::BATCH_RZZ_GROUP_SIZE],
        len: 0,
    }; cpu_k::MAX_BATCH_RZZ_GROUPS];
    let num_groups = cpu_k::build_batch_rzz_tables(edges, &mut groups);

    let mut tables_flat: Vec<f64> =
        Vec::with_capacity(num_groups * cpu_k::BATCH_RZZ_TABLE_SIZE * 2);
    for group in groups.iter().take(num_groups) {
        for entry in &group.table {
            tables_flat.push(entry.re);
            tables_flat.push(entry.im);
        }
    }
    let mut q0s_flat: Vec<i32> = Vec::with_capacity(num_groups * cpu_k::BATCH_RZZ_GROUP_SIZE);
    let mut q1s_flat: Vec<i32> = Vec::with_capacity(num_groups * cpu_k::BATCH_RZZ_GROUP_SIZE);
    for group in groups.iter().take(num_groups) {
        for k in 0..cpu_k::BATCH_RZZ_GROUP_SIZE {
            q0s_flat.push(group.q0s[k] as i32);
            q1s_flat.push(group.q1s[k] as i32);
        }
    }
    let lens: Vec<i32> = groups
        .iter()
        .take(num_groups)
        .map(|g| g.len as i32)
        .collect();

    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_batch_rzz")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let tables_dev = stream
        .clone_htod(&tables_flat)
        .map_err(|e| launch_err("upload batch_rzz tables", e))?;
    let q0s_dev = stream
        .clone_htod(&q0s_flat)
        .map_err(|e| launch_err("upload batch_rzz q0s", e))?;
    let q1s_dev = stream
        .clone_htod(&q1s_flat)
        .map_err(|e| launch_err("upload batch_rzz q1s", e))?;
    let lens_dev = stream
        .clone_htod(&lens)
        .map_err(|e| launch_err("upload batch_rzz lens", e))?;

    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&tables_dev)
        .arg(&q0s_dev)
        .arg(&q1s_dev)
        .arg(&lens_dev)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; device slices sized for num_groups; grid covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_batch_rzz", e))?;
    }
    Ok(())
}

/// Apply a `DiagonalBatch` via a single batched GPU kernel when groupable, falling back
/// to per-entry launches if the entries need to span more groups than the LUT allows
/// (same fallback condition as the CPU kernel).
pub(crate) fn launch_apply_diagonal_batch(
    ctx: &GpuContext,
    state: &mut GpuState,
    entries: &[crate::gates::DiagEntry],
) -> Result<()> {
    use crate::backend::statevector::kernels as cpu_k;
    use crate::gates::DiagEntry;

    if entries.is_empty() {
        return Ok(());
    }

    let Some(built) = cpu_k::build_diagonal_batch_tables(entries) else {
        // Fallback: per-entry dispatch (matches the CPU fallback path).
        let n = state.num_qubits();
        for entry in entries {
            match *entry {
                DiagEntry::Phase1q { qubit, d0, d1 } => {
                    if qubit >= n {
                        return Err(PrismError::InvalidQubit {
                            index: qubit,
                            register_size: n,
                        });
                    }
                    launch_apply_diagonal_1q(ctx, state, qubit, d0, d1)?;
                }
                DiagEntry::Phase2q { q0, q1, phase } => {
                    launch_apply_cu_phase(ctx, state, q0, q1, phase)?;
                }
                DiagEntry::Parity2q { q0, q1, same, diff } => {
                    launch_apply_parity_phase(ctx, state, q0, q1, same, diff)?;
                }
            }
        }
        return Ok(());
    };
    if built.num_groups == 0 {
        return Ok(());
    }

    let num_groups = built.num_groups;
    // Per-group shift arrays: unique_qubits is already flat and ordered (group 0 fills
    // positions [0..MAX), group 1 fills [MAX..2*MAX), etc.), so the group/pos decomposition
    // simplifies to a direct index.
    let mut shifts_flat: Vec<i32> = vec![0i32; num_groups * cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP];
    for (idx, &q) in built.unique_qubits.iter().enumerate() {
        shifts_flat[idx] = q as i32;
    }

    let mut tables_flat: Vec<f64> =
        Vec::with_capacity(num_groups * cpu_k::DIAG_BATCH_TABLE_SIZE * 2);
    for group_table in built.tables.iter().take(num_groups) {
        for entry in group_table {
            tables_flat.push(entry.re);
            tables_flat.push(entry.im);
        }
    }
    let lens: Vec<i32> = built
        .group_sizes
        .iter()
        .take(num_groups)
        .map(|&s| s as i32)
        .collect();

    let n = state.num_qubits();
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_diagonal_batch")?;
    let cfg = LaunchConfig {
        grid_dim: (grid_for(dim), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let tables_dev = stream
        .clone_htod(&tables_flat)
        .map_err(|e| launch_err("upload diag_batch tables", e))?;
    let shifts_dev = stream
        .clone_htod(&shifts_flat)
        .map_err(|e| launch_err("upload diag_batch shifts", e))?;
    let lens_dev = stream
        .clone_htod(&lens)
        .map_err(|e| launch_err("upload diag_batch lens", e))?;

    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&tables_dev)
        .arg(&shifts_dev)
        .arg(&lens_dev)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; device slices sized for num_groups; grid covers dim.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_diagonal_batch", e))?;
    }
    Ok(())
}

/// Matches the `TILE_Q` / `TILE_SIZE` in the PTX source. Targets `< TILE_Q` are processed
/// in shared memory by `apply_multi_fused_tiled`; targets `>= TILE_Q` fall back to
/// per-gate launches of `apply_gate_1q` because their pairs would span multiple tiles.
const MULTI_FUSED_TILE_Q: usize = 10;
const MULTI_FUSED_TILE_SIZE: u64 = 1 << MULTI_FUSED_TILE_Q;
const MULTI_FUSED_BLOCK_SIZE: u32 = (MULTI_FUSED_TILE_SIZE as u32) / 2;

/// The tiled kernel has fixed shared-memory load/store overhead; for MultiFused groups
/// with fewer than this many tile-local gates, per-gate launches of `apply_gate_1q` are
/// cheaper. Value chosen empirically on a GTX 1080 Ti (Pascal, compute_61). Below 3
/// gates, the per-gate path wins because launch overhead is comparable to the tile's
/// load/store cost.
///
/// **Needs re-tuning per architecture.** Launch overhead (Ampere/Ada reduce it),
/// shared-memory bandwidth, and L2 behavior all shift the crossover. On
/// A100/H100/RTX 40-series, re-run `benchmark_internal/compare_gpu.py` with values
/// 1..=5 and take the choice with the smallest regression.
const MULTI_FUSED_TILE_MIN_GATES: usize = 3;

/// Apply a non-diagonal `MultiFused` via a shared-memory tiled kernel for sub-gates with
/// low targets, plus per-gate launches for sub-gates whose target bit lies outside the
/// tile. Replaces the previous pure per-sub-gate decomposition.
pub(crate) fn launch_apply_multi_fused_nondiag(
    ctx: &GpuContext,
    state: &mut GpuState,
    gates: &[(usize, [[Complex64; 2]; 2])],
) -> Result<()> {
    if gates.is_empty() {
        return Ok(());
    }
    let n = state.num_qubits();
    // For n <= TILE_Q the tile is the full state; only one block runs and the kernel
    // collapses to the per-gate path. Fall back to the simple per-gate loop below.
    if n <= MULTI_FUSED_TILE_Q {
        for &(target, mat) in gates {
            launch_apply_gate_1q(ctx, state, target, mat)?;
        }
        return Ok(());
    }
    for &(target, _) in gates {
        if target >= n {
            return Err(PrismError::InvalidQubit {
                index: target,
                register_size: n,
            });
        }
    }

    // First pass: count tile-local sub-gates to pick the hot path without speculative
    // allocation.
    let tile_local_count = gates
        .iter()
        .filter(|(target, _)| *target < MULTI_FUSED_TILE_Q)
        .count();

    // Below the threshold, the tile's shared-memory load/store overhead outweighs the
    // savings from avoiding launches. Fall through to per-gate dispatch for the whole list.
    if tile_local_count < MULTI_FUSED_TILE_MIN_GATES {
        for &(target, mat) in gates {
            launch_apply_gate_1q(ctx, state, target, mat)?;
        }
        return Ok(());
    }

    // Second pass: flatten tile-local gate params into upload-ready arrays. External
    // (target >= TILE_Q) gates are dispatched directly in a third pass below, with no
    // intermediate Vec.
    let mut tile_targets: Vec<i32> = Vec::with_capacity(tile_local_count);
    let mut tile_matrices: Vec<f64> = Vec::with_capacity(tile_local_count * 8);
    for &(target, mat) in gates {
        if target < MULTI_FUSED_TILE_Q {
            tile_targets.push(target as i32);
            for row in mat.iter() {
                for entry in row.iter() {
                    tile_matrices.push(entry.re);
                    tile_matrices.push(entry.im);
                }
            }
        }
    }

    let dim: u64 = 1u64 << n;
    let num_tiles = dim / MULTI_FUSED_TILE_SIZE;
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("apply_multi_fused_tiled")?;
    let cfg = LaunchConfig {
        grid_dim: (num_tiles as u32, 1, 1),
        block_dim: (MULTI_FUSED_BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

    let targets_dev = stream
        .clone_htod(&tile_targets)
        .map_err(|e| launch_err("upload tiled targets", e))?;
    let matrices_dev = stream
        .clone_htod(&tile_matrices)
        .map_err(|e| launch_err("upload tiled matrices", e))?;

    let num_gates_i = tile_targets.len() as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&targets_dev)
        .arg(&matrices_dev)
        .arg(&num_gates_i);
    // SAFETY: signature matches kernel; num_tiles * TILE_SIZE = dim; device slices sized
    // for num_gates; all targets checked < MULTI_FUSED_TILE_Q above.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_multi_fused_tiled", e))?;
    }

    // Third pass: launch per-gate kernels for the external (target >= TILE_Q) sub-gates
    // whose pairs span tiles. Order matters; these must run after the tiled kernel.
    for &(target, mat) in gates {
        if target >= MULTI_FUSED_TILE_Q {
            launch_apply_gate_1q(ctx, state, target, mat)?;
        }
    }
    Ok(())
}
