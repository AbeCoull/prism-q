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

use crate::backend::statevector::kernels as cpu_k;
use crate::error::{PrismError, Result};

use super::super::{GpuContext, GpuState};
use super::{launch_err, linear_cfg, stream_and_fn};

const BLOCK_SIZE: u32 = 256;

/// The fused 2q matrix as a by-value kernel parameter, row-major re/im pairs. Mirrors
/// `struct FusedMat` in the CUDA template byte for byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct FusedMatArg {
    v: [f64; 32],
}

// SAFETY: a plain `repr(C)` array of f64 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for FusedMatArg {}

/// Widest qubit list a by-value kernel parameter carries; `1 << n` bounds `n` below it.
const MAX_PARAM_QUBITS: usize = 64;

/// Sorted MCU qubit list; mirrors `struct QubitList`.
#[repr(C)]
#[derive(Clone, Copy)]
struct QubitListArg {
    q: [i32; MAX_PARAM_QUBITS],
}

// SAFETY: a plain `repr(C)` array of i32 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for QubitListArg {}

/// Diagonal `MultiFused` sub-gates, `d[4g..4g + 4]` = (d0, d1) re/im pairs and `t[g]`
/// the target; mirrors `struct DiagList`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DiagListArg {
    d: [f64; 4 * MAX_PARAM_QUBITS],
    t: [i32; MAX_PARAM_QUBITS],
}

// SAFETY: plain `repr(C)` arrays of f64 then i32 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for DiagListArg {}

/// Tiled `MultiFused` sub-gates, `v[8g..8g + 8]` the row-major 2x2 matrix as re/im
/// pairs, the ten tile qubits at six bits each in `qubits` (sorted ascending), and each
/// gate's target as a four-bit position in that list; mirrors `struct TileGates`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TileGatesArg {
    v: [f64; 8 * MULTI_FUSED_TILE_Q],
    qubits: u64,
    targets: u64,
}

// SAFETY: a plain `repr(C)` f64 array and two u64s with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for TileGatesArg {}

/// Packed `meta` argument of the three LUT kernels: the per-group index lists, then
/// one length and one table offset per group.
const BATCH_PHASE_META_LEN: usize = cpu_k::MAX_BATCH_PHASE_GROUPS * cpu_k::BATCH_PHASE_GROUP_SIZE
    + 2 * cpu_k::MAX_BATCH_PHASE_GROUPS;
const BATCH_RZZ_META_LEN: usize =
    2 * cpu_k::MAX_BATCH_RZZ_GROUPS * cpu_k::BATCH_RZZ_GROUP_SIZE + 2 * cpu_k::MAX_BATCH_RZZ_GROUPS;
const DIAG_BATCH_META_LEN: usize = cpu_k::MAX_DIAG_BATCH_GROUPS
    * cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP
    + 2 * cpu_k::MAX_DIAG_BATCH_GROUPS;

/// The sorted MCU qubit list as a kernel parameter.
fn qubit_list(sorted: &[u32]) -> QubitListArg {
    let mut arg = QubitListArg {
        q: [0; MAX_PARAM_QUBITS],
    };
    for (d, &q) in arg.q.iter_mut().zip(sorted) {
        *d = q as i32;
    }
    arg
}

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
#define BP_MAX_GROUPS   {{BP_MAX_GROUPS}}
#define PARAM_QUBITS    {{PARAM_QUBITS}}
#define BR_MAX_GROUPS   {{BR_MAX_GROUPS}}
#define DB_MAX_GROUPS   {{DB_MAX_GROUPS}}

// The fused 2q matrix travels in kernel parameter space: a pageable host to device
// copy synchronizes the stream before it starts, so uploading 32 doubles per gate
// serialized the host against every queued kernel. Layout mirrors `FusedMatArg`.
struct FusedMat { double v[32]; };
struct QubitList { int q[PARAM_QUBITS]; };
struct DiagList { double d[4 * PARAM_QUBITS]; int t[PARAM_QUBITS]; };
struct TileGates { double v[8 * TILE_Q]; unsigned long long qubits; unsigned long long targets; };

// ============================================================================
// Shared device helpers
// ============================================================================

__device__ __forceinline__ void apply2x2(double2 *state,
    unsigned long long i0, unsigned long long i1,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    double2 a = state[i0];
    double2 b = state[i1];
    state[i0].x = m00r*a.x - m00i*a.y + m01r*b.x - m01i*b.y;
    state[i0].y = m00r*a.y + m00i*a.x + m01r*b.y + m01i*b.x;
    state[i1].x = m10r*a.x - m10i*a.y + m11r*b.x - m11i*b.y;
    state[i1].y = m10r*a.y + m10i*a.x + m11r*b.y + m11i*b.x;
}

__device__ __forceinline__ void apply_phase(double2 *state, unsigned long long i, double pr, double pi)
{
    double2 a = state[i];
    state[i].x = pr*a.x - pi*a.y;
    state[i].y = pr*a.y + pi*a.x;
}

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

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
}

// Diagonal 2x2 specialisation.
// state[i0] *= d0, state[i1] *= d1, no cross terms.

extern "C" __global__ void apply_diagonal_1q(
    double2 *state, unsigned long long pair_count, int target,
    double d0r, double d0i, double d1r, double d1i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;

    unsigned long long mask = (1ULL << target) - 1;
    unsigned long long i0 = ((k & ~mask) << 1) | (k & mask);
    unsigned long long i1 = i0 | (1ULL << target);

    apply_phase(state, i0, d0r, d0i);
    apply_phase(state, i1, d1r, d1i);
}

// ============================================================================
// Two-qubit gates (CX / CZ / SWAP)
// ============================================================================
//
// All take `pair_count = 2^(n-2)` threads. Each thread computes a compressed index
// and expands via chained insert_zero_bit (q0, q1 sorted).

__device__ __forceinline__ unsigned long long expand_2q(unsigned long long k, int lo_q, int hi_q) {
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
    apply_phase(state, i, pr, pi);
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

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
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
    apply_phase(state, i, pr, pi);
}

// Multi-controlled unitary. `sorted` contains all controls + target, sorted ascending.
// `num_sorted = num_controls + 1`. `ctrl_mask` has bits set at control positions only;
// `tgt_mask = 1 << target`.
// Launch over 2^(n - num_sorted) threads.

extern "C" __global__ void apply_mcu(
    double2 *state, unsigned long long iter_count,
    QubitList sorted, int num_sorted,
    unsigned long long ctrl_mask, unsigned long long tgt_mask,
    double m00r, double m00i, double m01r, double m01i,
    double m10r, double m10i, double m11r, double m11i)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = sorted.q[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    unsigned long long i0 = idx | ctrl_mask;
    unsigned long long i1 = i0 | tgt_mask;

    apply2x2(state, i0, i1, m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i);
}

extern "C" __global__ void apply_mcu_phase(
    double2 *state, unsigned long long iter_count,
    QubitList sorted, int num_sorted,
    unsigned long long all_mask,
    double pr, double pi)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= iter_count) return;
    unsigned long long idx = k;
    for (int i = 0; i < num_sorted; ++i) {
        int bit = sorted.q[i];
        unsigned long long mask_lo = (1ULL << bit) - 1;
        unsigned long long lo = idx & mask_lo;
        unsigned long long hi = idx >> bit;
        idx = (hi << (bit + 1)) | lo;
    }
    unsigned long long i = idx | all_mask;
    apply_phase(state, i, pr, pi);
}

// ============================================================================
// Fused 2q: generic 4x4 matrix. mat is row-major 16 Complex64 = 32 f64.
// Launch over pair_count = 2^(n-2) threads; each processes one 4-element group.
// ============================================================================

extern "C" __global__ void apply_fused_2q(
    double2 *state, unsigned long long pair_count, int q0, int q1,
    FusedMat mat)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= pair_count) return;
    int lo_q = q0 < q1 ? q0 : q1;
    int hi_q = q0 < q1 ? q1 : q0;
    unsigned long long idx = expand_2q(k, lo_q, hi_q);

    // Basis ordering matches the CPU PreparedGate2q::apply_full in src/backend/simd.rs:
    //   basis index b = (q0_bit << 1) | q1_bit   i.e. q1 is LSB of the 4-element basis.
    //   b=0 → (q0=0, q1=0); b=1 → (q0=0, q1=1); b=2 → (q0=1, q1=0); b=3 → (q0=1, q1=1).
    unsigned long long i00 = idx;
    unsigned long long i01 = idx | (1ULL << q1);
    unsigned long long i10 = idx | (1ULL << q0);
    unsigned long long i11 = idx | (1ULL << q0) | (1ULL << q1);

    double2 in[4] = {state[i00], state[i01], state[i10], state[i11]};
    unsigned long long indices[4] = {i00, i01, i10, i11};

    #pragma unroll
    for (int row = 0; row < 4; ++row) {
        double rr = 0.0, ri = 0.0;
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            // mat row-major: 32 f64s, row r col c → mat.v[2*(r*4+c)]=re, mat.v[2*(r*4+c)+1]=im
            double mr = mat.v[2*(row*4 + col)];
            double mi = mat.v[2*(row*4 + col) + 1];
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

// Single-block reduction over `partials` (length `count`) into a single f64
// at `result[0]`. Caller launches with a single block of `blockDim.x` threads;
// each thread loops with stride `blockDim.x` over `partials` to accumulate.
// Replaces a num_blocks-element D2H plus a host sum with one 8-byte D2H.
extern "C" __global__ void measure_prob_one_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    double s = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        s += partials[i];
    }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) result[0] = sdata[0];
}

// rdm_qubit: per-block reduction of the four 1q reduced-density-matrix sums over
// amplitude pairs (i0, i1 = i0 | 1<<qubit): sum |a0|^2, sum |a1|^2, and sum
// a1 * conj(a0) (re, im). Pair index p maps to i0 by inserting a zero bit at
// `qubit`. Each block reduces 2*BLOCK_SIZE pairs into out_partials[4*blockIdx..].
// Shared memory holds four blockDim.x columns.

extern "C" __global__ void rdm_qubit(
    const double2 *state, unsigned long long pairs, int qubit, double *out_partials)
{
    extern __shared__ double sdata[];
    double *s0 = sdata;
    double *s1 = sdata + blockDim.x;
    double *sr = sdata + 2 * blockDim.x;
    double *si = sdata + 3 * blockDim.x;
    unsigned long long tid = threadIdx.x;
    unsigned long long low_mask = (1ULL << qubit) - 1ULL;
    double p0 = 0.0, p1 = 0.0, rr = 0.0, ri = 0.0;
    unsigned long long p = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    for (int rep = 0; rep < 2; ++rep) {
        if (p < pairs) {
            unsigned long long i0 = ((p & ~low_mask) << 1) | (p & low_mask);
            double2 a0 = state[i0];
            double2 a1 = state[i0 | (1ULL << qubit)];
            p0 += a0.x*a0.x + a0.y*a0.y;
            p1 += a1.x*a1.x + a1.y*a1.y;
            rr += a1.x*a0.x + a1.y*a0.y;
            ri += a1.y*a0.x - a1.x*a0.y;
        }
        p += blockDim.x;
    }
    s0[tid] = p0; s1[tid] = p1; sr[tid] = rr; si[tid] = ri;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_partials[4 * blockIdx.x + 0] = s0[0];
        out_partials[4 * blockIdx.x + 1] = s1[0];
        out_partials[4 * blockIdx.x + 2] = sr[0];
        out_partials[4 * blockIdx.x + 3] = si[0];
    }
}

// Single-block reduction of the four-column partials (length 4*count) into
// result[0..4]. Same shape as measure_prob_one_finalize; replaces a
// 4*num_blocks-element D2H plus a host sum with one 32-byte D2H.
extern "C" __global__ void rdm_qubit_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    double *s0 = sdata;
    double *s1 = sdata + blockDim.x;
    double *sr = sdata + 2 * blockDim.x;
    double *si = sdata + 3 * blockDim.x;
    unsigned int tid = threadIdx.x;
    double p0 = 0.0, p1 = 0.0, rr = 0.0, ri = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        p0 += partials[4 * i + 0];
        p1 += partials[4 * i + 1];
        rr += partials[4 * i + 2];
        ri += partials[4 * i + 3];
    }
    s0[tid] = p0; s1[tid] = p1; sr[tid] = rr; si[tid] = ri;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s0[tid] += s0[tid + stride];
            s1[tid] += s1[tid + stride];
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = s0[0];
        result[1] = s1[0];
        result[2] = sr[0];
        result[3] = si[0];
    }
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
// and squaring on the host.

extern "C" __global__ void compute_probabilities(
    const double2 *state, unsigned long long dim, double norm_sq, double *out)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    out[i] = (a.x * a.x + a.y * a.y) * norm_sq;
}

// apply_multi_fused_tiled: batched non-diagonal MultiFused via shared-memory tiles.
//
// Each block loads a TILE_SIZE slice of amplitudes into shared memory, then applies every
// sub-gate whose target bit is inside the tile (target < TILE_Q) with no further global
// memory reads. Pairs (i0, i1) for a given gate stay within the tile because the target
// bit is a low bit of the global index; the high bits (> TILE_Q) are the block id.
//
// Gate data rides in parameter space: six bits of `gates.qubits` per tile qubit, four
// bits of `gates.targets` per gate (a position in the tile qubit list), and
// gates.v[8g .. 8g+8] the row-major matrix as re/im pairs.
//
// TILE_Q = 10, TILE_SIZE = 1024, block_size = 512 threads, each thread handles one pair.
// Shared memory usage: 1024 × 16 bytes = 16 KB. Pascal-friendly.
// (TILE_Q / TILE_SIZE are defined in the template header at the top of the file.)

// Scatter the TILE_Q bits of `j` onto the qubit positions `q`.
__device__ __forceinline__ unsigned long long deposit_tile_bits(int j, const int *q)
{
    unsigned long long idx = 0;
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) {
        idx |= (unsigned long long)((j >> k) & 1) << q[k];
    }
    return idx;
}

// The tile is the 2^TILE_Q amplitudes spanned by the tile qubits (sorted ascending),
// with every other bit fixed per block. Tile-local index j maps to the global index
// block_base | deposit(j); the five lowest qubits are always in the tile, so a warp's 32
// consecutive j hit 32 consecutive amplitudes and the loads stay coalesced.
extern "C" __global__ void apply_multi_fused_tiled(
    double2 *state, unsigned long long dim,
    TileGates gates,
    int num_gates)
{
    __shared__ double2 tile[TILE_SIZE];

    unsigned long long tiles = dim >> TILE_Q;
    if (blockIdx.x >= tiles) return;

    int q[TILE_Q];
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) q[k] = (int)((gates.qubits >> (6 * k)) & 63ULL);
    unsigned long long block_base = blockIdx.x;
    #pragma unroll
    for (int k = 0; k < TILE_Q; ++k) {
        unsigned long long lo = block_base & ((1ULL << q[k]) - 1ULL);
        block_base = ((block_base >> q[k]) << (q[k] + 1)) | lo;
    }

    int tid = (int)threadIdx.x;
    // Load the tile cooperatively, two amplitudes per thread (block_dim = TILE_SIZE/2).
    int a_off = tid;
    int b_off = tid + (TILE_SIZE / 2);
    unsigned long long a_idx = block_base | deposit_tile_bits(a_off, q);
    unsigned long long b_idx = block_base | deposit_tile_bits(b_off, q);
    tile[a_off] = state[a_idx];
    tile[b_off] = state[b_idx];
    __syncthreads();

    // Each thread owns one pair per gate. k is the compressed pair index (0 .. TILE_SIZE/2).
    int k = tid;

    for (int g = 0; g < num_gates; ++g) {
        int t = (int)((gates.targets >> (4 * g)) & 15ULL);
        int mask = (1 << t) - 1;
        int i0 = ((k & ~mask) << 1) | (k & mask);
        int i1 = i0 | (1 << t);

        const double *m = gates.v + 8 * g;
        double2 m00 = make_double2(m[0], m[1]);
        double2 m01 = make_double2(m[2], m[3]);
        double2 m10 = make_double2(m[4], m[5]);
        double2 m11 = make_double2(m[6], m[7]);

        // No sync needed between read and write here: each thread owns an
        // exclusive (i0, i1) pair within a single gate iteration (the compressed-
        // index mapping partitions [0, TILE_SIZE)). The trailing sync below
        // serialises writes across gate boundaries, which is the only hazard.

        apply2x2(tile, i0, i1, m00.x, m00.y, m01.x, m01.y, m10.x, m10.y, m11.x, m11.y);
        __syncthreads();
    }

    // Store tile back to global memory.
    state[a_idx] = tile[a_off];
    state[b_idx] = tile[b_off];
}

// apply_diagonal_batch: applies a mixed batch of diagonal entries (1q / 2q / parity-2q)
// via precomputed per-group LUTs (built on the host by build_diagonal_batch_tables).
// Replaces per-entry dispatch. Launches 2^n threads; each thread folds all group phases.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the bit pattern `bits`. `meta` packs
// group_shifts[DB_MAX_GROUPS * DB_MAX_QUBITS] (qubit index of the j-th bit in group g),
// then group_lens[DB_MAX_GROUPS], then group_offsets[DB_MAX_GROUPS].

extern "C" __global__ void apply_diagonal_batch(
    double2 *state, unsigned long long dim,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int *group_shifts = meta;
    const int *group_lens = meta + DB_MAX_GROUPS * DB_MAX_QUBITS;
    const int *group_offsets = group_lens + DB_MAX_GROUPS;

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *shifts = group_shifts + g * DB_MAX_QUBITS;
        int bits = 0;
        for (int j = 0; j < len; ++j) {
            bits |= (int)(((i >> shifts[j]) & 1ULL) << j);
        }
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, i, cr, ci);
}

// apply_batch_rzz: applies a batch of Rzz gates via precomputed parity-phase LUTs (built
// built on the host by build_batch_rzz_tables). Replaces per-edge apply_parity_phase launches.
// Launches 2^n threads across the full state; each thread computes parity bits per edge
// per group, indexes the 256-entry LUT, chains multiplies.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the parity pattern `bits`. `meta` packs
// group_q0s[BR_MAX_GROUPS * BR_GROUP_SIZE] and group_q1s[BR_MAX_GROUPS * BR_GROUP_SIZE]
// (qubit pair of the k-th edge in group g), then group_lens[BR_MAX_GROUPS], then
// group_offsets[BR_MAX_GROUPS].

extern "C" __global__ void apply_batch_rzz(
    double2 *state, unsigned long long dim,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    const int *group_q0s = meta;
    const int *group_q1s = meta + BR_MAX_GROUPS * BR_GROUP_SIZE;
    const int *group_lens = group_q1s + BR_MAX_GROUPS * BR_GROUP_SIZE;
    const int *group_offsets = group_lens + BR_MAX_GROUPS;

    double cr = 1.0, ci = 0.0;
    for (int g = 0; g < num_groups; ++g) {
        int len = group_lens[g];
        const int *q0s = group_q0s + g * BR_GROUP_SIZE;
        const int *q1s = group_q1s + g * BR_GROUP_SIZE;
        int bits = 0;
        for (int k = 0; k < len; ++k) {
            bits |= (int)((((i >> q0s[k]) ^ (i >> q1s[k])) & 1ULL) << k);
        }
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, i, cr, ci);
}

// apply_batch_phase: applies a batch of controlled-phase gates sharing a control qubit via
// precomputed LUTs (built on the host by build_batch_phase_tables). Replaces per-phase
// launches of apply_cu_phase. One DRAM read/write per amplitude in the ctrl=1 subspace.
//
// Table layout: group g owns `1 << group_lens[g]` entries starting at
// group_tables[group_offsets[g]], indexed by the bit pattern `bits`. `meta` packs
// group_shifts[BP_MAX_GROUPS * BP_GROUP_SIZE] (qubit index of the j-th bit in group g),
// then group_lens[BP_MAX_GROUPS], then group_offsets[BP_MAX_GROUPS].

extern "C" __global__ void apply_batch_phase(
    double2 *state, unsigned long long half_count,
    int control,
    const double2 * __restrict__ group_tables,
    const int * __restrict__ meta,
    int num_groups)
{
    unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= half_count) return;
    const int *group_shifts = meta;
    const int *group_lens = meta + BP_MAX_GROUPS * BP_GROUP_SIZE;
    const int *group_offsets = group_lens + BP_MAX_GROUPS;
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
        double2 ph = group_tables[group_offsets[g] + bits];
        double nr = cr * ph.x - ci * ph.y;
        double ni = cr * ph.y + ci * ph.x;
        cr = nr;
        ci = ni;
    }

    apply_phase(state, idx, cr, ci);
}

// apply_multi_fused_diagonal: batch of diagonal 1q gates in a single pass. Replaces the
// per sub-gate decomposition on the host for `Gate::MultiFused { all_diagonal: true }`.
//
// Each thread handles one amplitude and folds all `num_gates` diagonal multiplications
// based on the bit pattern of its own index. One DRAM read + one DRAM write per thread;
// compute = num_gates complex multiplies (typically 1-10).
//
// Gate data rides in parameter space: gates.t[g] the target, gates.d[4g .. 4g+4] the
// (d0, d1) re/im pairs. Both entries are loaded uniformly and selected per thread.

extern "C" __global__ void apply_multi_fused_diagonal(
    double2 *state, unsigned long long dim,
    DiagList gates,
    int num_gates)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double2 a = state[i];
    for (int g = 0; g < num_gates; ++g) {
        int t = gates.t[g];
        int bit = (int)((i >> t) & 1ULL);
        const double *e = gates.d + 4 * g;
        double2 p = bit ? make_double2(e[2], e[3]) : make_double2(e[0], e[1]);
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
/// Adding a kernel that depends on a new host constant: add a `{{PLACEHOLDER}}` to
/// the template header, add a matching `.replace(...)` call below, done.
pub(crate) fn kernel_source() -> String {
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
        .replace(
            "{{BP_MAX_GROUPS}}",
            &cpu_k::MAX_BATCH_PHASE_GROUPS.to_string(),
        )
        .replace("{{PARAM_QUBITS}}", &MAX_PARAM_QUBITS.to_string())
        .replace(
            "{{BR_MAX_GROUPS}}",
            &cpu_k::MAX_BATCH_RZZ_GROUPS.to_string(),
        )
        .replace(
            "{{DB_MAX_GROUPS}}",
            &cpu_k::MAX_DIAG_BATCH_GROUPS.to_string(),
        )
}

// ---- Rust-side launchers ----

fn grid_for(count: u64) -> u32 {
    count.div_ceil(BLOCK_SIZE as u64).max(1) as u32
}

/// Write amplitude 0 = 1; assumes the buffer is already zeroed.
pub(crate) fn launch_set_initial_state(ctx: &GpuContext, state: &mut GpuState) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "set_initial_state")?;
    let cfg = linear_cfg(1, 1);
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
    let (stream, func) = stream_and_fn(ctx, "compute_probabilities")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));

    // Reuse the cached scratch buffer when large enough; grow if num_qubits increased.
    let mut scratch_slot = state.probs_scratch();
    if scratch_slot.as_ref().is_none_or(|b| b.len() < dim as usize) {
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
    let (stream, func) = stream_and_fn(ctx, "apply_gate_1q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
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
    let (stream, func) = stream_and_fn(ctx, "apply_diagonal_1q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
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

/// `same` multiplies amplitudes whose two target bits agree, `diff` those
/// whose bits differ.
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
    let (stream, func) = stream_and_fn(ctx, "apply_parity_phase")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));
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
    let (stream, func) = stream_and_fn(ctx, "apply_cu")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
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
    let (stream, func) = stream_and_fn(ctx, "apply_cu_phase")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
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

/// The sorted control-plus-target qubit list is uploaded through the launcher
/// scratch for index expansion.
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

    let (stream, func) = stream_and_fn(ctx, "apply_mcu")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(iter_count));
    let sorted_arg = qubit_list(&sorted);
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
        .arg(&sorted_arg)
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
    // SAFETY: signature matches; the qubit list rides in parameter space and the grid
    // covers iter_count.
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

    let (stream, func) = stream_and_fn(ctx, "apply_mcu_phase")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(iter_count));
    let sorted_arg = qubit_list(&sorted);
    let pr = phase.re;
    let pi = phase.im;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&iter_count)
        .arg(&sorted_arg)
        .arg(&num_sorted)
        .arg(&all_mask)
        .arg(&pr)
        .arg(&pi);
    // SAFETY: signature matches; the qubit list rides in parameter space and the grid
    // covers iter_count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_mcu_phase", e))?;
    }
    Ok(())
}

/// The matrix is flattened row-major into a by-value kernel parameter.
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
    let (stream, func) = stream_and_fn(ctx, "apply_fused_2q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
    let mut mat = FusedMatArg { v: [0.0; 32] };
    for (slot, entry) in mat.v.chunks_exact_mut(2).zip(matrix.iter().flatten()) {
        slot[0] = entry.re;
        slot[1] = entry.im;
    }
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&mat);
    // SAFETY: signature matches; the matrix is copied into parameter space at launch.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_fused_2q", e))?;
    }
    Ok(())
}

/// Two-stage device reduction (per-block partials, then a single-block
/// finalize); one f64 crosses PCIe.
pub(crate) fn measure_prob_one(ctx: &GpuContext, state: &GpuState, qubit: usize) -> Result<f64> {
    let n = state.num_qubits();
    if qubit >= n {
        return Err(PrismError::InvalidQubit {
            index: qubit,
            register_size: n,
        });
    }
    let dim: u64 = 1u64 << n;
    // Each block of stage 1 processes 2*BLOCK_SIZE elements.
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = dim.div_ceil(elems_per_block).max(1) as u32;

    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("measure_prob_one")?;
    let stage2 = device.function("measure_prob_one_finalize")?;
    let stage1_cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
    };
    let stage2_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
    };

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partials =
        super::ensure_capacity(&mut scratch.measure_partials, device, num_blocks as usize)?;

    let qubit_i = qubit as i32;
    {
        let mut builder = stream.launch_builder(&stage1);
        let state_buf = state.buffer().raw();
        builder
            .arg(state_buf)
            .arg(&dim)
            .arg(&qubit_i)
            .arg(partials.raw_mut());
        // SAFETY: signature matches kernel; num_blocks * 2*BLOCK_SIZE covers dim.
        unsafe {
            builder
                .launch(stage1_cfg)
                .map_err(|e| launch_err("measure_prob_one", e))?;
        }
    }

    let result = super::ensure_capacity(&mut scratch.measure_result, device, 1)?;
    let count_u32 = num_blocks;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&count_u32)
            .arg(result.raw_mut());
        // SAFETY: signature matches kernel; one block of BLOCK_SIZE threads, grid-stride
        // loop over `count_u32` partials. Both buffers held by the scratch guard.
        unsafe {
            builder
                .launch(stage2_cfg)
                .map_err(|e| launch_err("measure_prob_one_finalize", e))?;
        }
    }

    let mut host_result = [0.0_f64];
    scratch
        .measure_result
        .as_ref()
        .unwrap()
        .copy_to_host(device, &mut host_result)?;
    let prob_raw = host_result[0];
    let norm_sq = state.pending_norm() * state.pending_norm();
    Ok((prob_raw * norm_sq).clamp(0.0, 1.0))
}

/// One-qubit reduced density matrix from the device state, `pending_norm²`
/// applied. Two reduction launches and a 32-byte readback replace the
/// full-state export the host fallback would need.
pub(crate) fn reduced_density_matrix_1q(
    ctx: &GpuContext,
    state: &GpuState,
    qubit: usize,
) -> Result<[[Complex64; 2]; 2]> {
    let n = state.num_qubits();
    if qubit >= n {
        return Err(PrismError::InvalidQubit {
            index: qubit,
            register_size: n,
        });
    }
    let pairs: u64 = 1u64 << (n - 1);
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = pairs.div_ceil(elems_per_block).max(1) as u32;

    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("rdm_qubit")?;
    let stage2 = device.function("rdm_qubit_finalize")?;
    let shared_bytes = 4 * BLOCK_SIZE * std::mem::size_of::<f64>() as u32;
    let stage1_cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: shared_bytes,
    };
    let stage2_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partials = super::ensure_capacity(
        &mut scratch.measure_partials,
        device,
        4 * num_blocks as usize,
    )?;

    let qubit_i = qubit as i32;
    {
        let mut builder = stream.launch_builder(&stage1);
        let state_buf = state.buffer().raw();
        builder
            .arg(state_buf)
            .arg(&pairs)
            .arg(&qubit_i)
            .arg(partials.raw_mut());
        // SAFETY: signature matches kernel; num_blocks * 2*BLOCK_SIZE covers pairs and
        // out_partials holds 4*num_blocks f64s.
        unsafe {
            builder
                .launch(stage1_cfg)
                .map_err(|e| launch_err("rdm_qubit", e))?;
        }
    }

    let result = super::ensure_capacity(&mut scratch.rdm_result, device, 4)?;
    let count_u32 = num_blocks;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&count_u32)
            .arg(result.raw_mut());
        // SAFETY: signature matches kernel; one block of BLOCK_SIZE threads, grid-stride
        // loop over `count_u32` four-column partials. Both buffers held by the scratch guard.
        unsafe {
            builder
                .launch(stage2_cfg)
                .map_err(|e| launch_err("rdm_qubit_finalize", e))?;
        }
    }

    let mut host_result = [0.0_f64; 4];
    scratch
        .rdm_result
        .as_ref()
        .unwrap()
        .copy_to_host(device, &mut host_result[..])?;
    let norm_sq = state.pending_norm() * state.pending_norm();
    let p0 = host_result[0] * norm_sq;
    let p1 = host_result[1] * norm_sq;
    let r = Complex64::new(host_result[2], host_result[3]) * norm_sq;
    Ok([
        [Complex64::new(p0, 0.0), r.conj()],
        [r, Complex64::new(p1, 0.0)],
    ])
}

/// Zeroes the losing branch only; renormalization is deferred to the caller
/// via `pending_norm`.
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
    let (stream, func) = stream_and_fn(ctx, "measure_collapse")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));
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

    let num_gates = gates.len();
    if num_gates > MAX_PARAM_QUBITS {
        for &(target, mat) in gates {
            launch_apply_diagonal_1q(ctx, state, target, mat[0][0], mat[1][1])?;
        }
        return Ok(());
    }
    let mut arg = DiagListArg {
        d: [0.0; 4 * MAX_PARAM_QUBITS],
        t: [0; MAX_PARAM_QUBITS],
    };
    for (g, &(target, mat)) in gates.iter().enumerate() {
        arg.t[g] = target as i32;
        arg.d[4 * g..4 * g + 4].copy_from_slice(&[
            mat[0][0].re,
            mat[0][0].im,
            mat[1][1].re,
            mat[1][1].im,
        ]);
    }

    let dim: u64 = 1u64 << n;
    let (stream, func) = stream_and_fn(ctx, "apply_multi_fused_diagonal")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));
    let num_gates_i = num_gates as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&dim).arg(&arg).arg(&num_gates_i);
    // SAFETY: signature matches; the gate list rides in parameter space with num_gates
    // entries filled; grid covers dim.
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

    // Each group ships only the `1 << len` table entries it indexes, at the offset the
    // packed metadata names.
    let mut meta = [0i32; BATCH_PHASE_META_LEN];
    let lens_at = cpu_k::MAX_BATCH_PHASE_GROUPS * cpu_k::BATCH_PHASE_GROUP_SIZE;
    let offsets_at = lens_at + cpu_k::MAX_BATCH_PHASE_GROUPS;
    let mut total = 0usize;
    for (g, group) in groups.iter().take(num_groups).enumerate() {
        for (j, &s) in group.shifts.iter().enumerate() {
            meta[g * cpu_k::BATCH_PHASE_GROUP_SIZE + j] = s as i32;
        }
        meta[lens_at + g] = group.len as i32;
        meta[offsets_at + g] = total as i32;
        total += 1 << group.len;
    }

    let half_count: u64 = 1u64 << (n - 1);
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "apply_batch_phase")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(half_count));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = super::stage_blob(
        &mut scratch.blob,
        device,
        meta.iter().copied(),
        2 * total,
        |dst| {
            let mut at = 0;
            for group in groups.iter().take(num_groups) {
                for entry in &group.table[..1 << group.len] {
                    dst[at] = entry.re;
                    dst[at + 1] = entry.im;
                    at += 2;
                }
            }
        },
    )?;

    let control_i = control as i32;
    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&half_count)
        .arg(&control_i)
        .arg(&staged.f64s)
        .arg(&staged.ints)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; every table read is below the staged length by
    // construction of the offsets; grid covers half. The scratch guard holds the upload.
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

    let mut meta = [0i32; BATCH_RZZ_META_LEN];
    let q1s_at = cpu_k::MAX_BATCH_RZZ_GROUPS * cpu_k::BATCH_RZZ_GROUP_SIZE;
    let lens_at = 2 * q1s_at;
    let offsets_at = lens_at + cpu_k::MAX_BATCH_RZZ_GROUPS;
    let mut total = 0usize;
    for (g, group) in groups.iter().take(num_groups).enumerate() {
        for k in 0..cpu_k::BATCH_RZZ_GROUP_SIZE {
            meta[g * cpu_k::BATCH_RZZ_GROUP_SIZE + k] = group.q0s[k] as i32;
            meta[q1s_at + g * cpu_k::BATCH_RZZ_GROUP_SIZE + k] = group.q1s[k] as i32;
        }
        meta[lens_at + g] = group.len as i32;
        meta[offsets_at + g] = total as i32;
        total += 1 << group.len;
    }

    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "apply_batch_rzz")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = super::stage_blob(
        &mut scratch.blob,
        device,
        meta.iter().copied(),
        2 * total,
        |dst| {
            let mut at = 0;
            for group in groups.iter().take(num_groups) {
                for entry in &group.table[..1 << group.len] {
                    dst[at] = entry.re;
                    dst[at + 1] = entry.im;
                    at += 2;
                }
            }
        },
    )?;

    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&staged.f64s)
        .arg(&staged.ints)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; every table read is below the staged length by
    // construction of the offsets; grid covers dim. The scratch guard holds the upload.
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
    // unique_qubits holds the groups concatenated in order, and a group can be shorter than
    // the stride the kernel indexes by, so the shift block is filled group by group.
    let mut meta = [0i32; DIAG_BATCH_META_LEN];
    let lens_at = cpu_k::MAX_DIAG_BATCH_GROUPS * cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP;
    let offsets_at = lens_at + cpu_k::MAX_DIAG_BATCH_GROUPS;
    let mut flat = 0;
    let mut total = 0usize;
    for (g, &size) in built.group_sizes.iter().enumerate().take(num_groups) {
        let base = g * cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP;
        for (j, &q) in built.unique_qubits[flat..flat + size].iter().enumerate() {
            meta[base + j] = q as i32;
        }
        flat += size;
        meta[lens_at + g] = size as i32;
        meta[offsets_at + g] = total as i32;
        total += 1 << size;
    }

    let n = state.num_qubits();
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "apply_diagonal_batch")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = super::stage_blob(
        &mut scratch.blob,
        device,
        meta.iter().copied(),
        2 * total,
        |dst| {
            let mut at = 0;
            for (group_table, &size) in built.tables.iter().zip(&built.group_sizes).take(num_groups)
            {
                for entry in &group_table[..1 << size] {
                    dst[at] = entry.re;
                    dst[at + 1] = entry.im;
                    at += 2;
                }
            }
        },
    )?;

    let num_groups_i = num_groups as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&dim)
        .arg(&staged.f64s)
        .arg(&staged.ints)
        .arg(&num_groups_i);
    // SAFETY: signature matches kernel; every table read is below the staged length by
    // construction of the offsets; grid covers dim. The scratch guard holds the upload.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("apply_diagonal_batch", e))?;
    }
    Ok(())
}

/// Matches the `TILE_Q` / `TILE_SIZE` in the PTX source: a tile spans `2^TILE_Q`
/// amplitudes over a chosen set of `TILE_Q` qubits.
const MULTI_FUSED_TILE_Q: usize = 10;
const MULTI_FUSED_TILE_SIZE: u64 = 1 << MULTI_FUSED_TILE_Q;
const MULTI_FUSED_BLOCK_SIZE: u32 = (MULTI_FUSED_TILE_SIZE as u32) / 2;

/// Qubits every tile carries so a warp's 32 consecutive tile indices are 32
/// consecutive amplitudes. The remaining `TILE_Q - ANCHOR_Q` tile qubits are chosen
/// per pass from the gates' targets.
const MULTI_FUSED_ANCHOR_Q: usize = 5;

/// The tiled kernel has fixed shared-memory load/store overhead; for a pass with fewer
/// than this many gates, per-gate launches of `apply_gate_1q` are cheaper. Value chosen
/// empirically on a GTX 1080 Ti (Pascal, compute_61); launch overhead, shared-memory
/// bandwidth, and L2 behavior shift the crossover on newer architectures, so re-tune
/// per device generation.
const MULTI_FUSED_TILE_MIN_GATES: usize = 3;

/// Widest register the GPU statevector admits, from the `GpuState::new` bound.
const MAX_GPU_QUBITS: usize = usize::BITS as usize - 4;

/// Apply a non-diagonal `MultiFused` as a sequence of shared-memory tiled passes.
///
/// Each pass owns a tile of `TILE_Q` qubits: the `ANCHOR_Q` lowest qubits plus up to
/// `TILE_Q - ANCHOR_Q` of the gates' higher targets, so a `MultiFused` over `n` qubits
/// costs about `(n - ANCHOR_Q) / (TILE_Q - ANCHOR_Q)` passes instead of one pass per
/// high target. Gates on distinct qubits commute, so the pass order is free. A pass
/// with fewer than `TILE_MIN_GATES` gates falls back to per-gate launches.
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
    // collapses to the per-gate path.
    if n <= MULTI_FUSED_TILE_Q {
        for &(target, mat) in gates {
            launch_apply_gate_1q(ctx, state, target, mat)?;
        }
        return Ok(());
    }
    let mut is_target = [false; MAX_GPU_QUBITS];
    for &(target, _) in gates {
        if target >= n {
            return Err(PrismError::InvalidQubit {
                index: target,
                register_size: n,
            });
        }
        is_target[target] = true;
    }

    let per_pass = MULTI_FUSED_TILE_Q - MULTI_FUSED_ANCHOR_Q;
    let low: Vec<usize> = (0..gates.len())
        .filter(|&g| gates[g].0 < MULTI_FUSED_ANCHOR_Q)
        .collect();
    let high: Vec<usize> = (0..gates.len())
        .filter(|&g| gates[g].0 >= MULTI_FUSED_ANCHOR_Q)
        .collect();
    let passes = high.len().div_ceil(per_pass).max(1);

    for pass in 0..passes {
        let chunk =
            &high[(pass * per_pass).min(high.len())..((pass + 1) * per_pass).min(high.len())];
        let pass_gates: Vec<usize> = if pass == 0 {
            low.iter().chain(chunk).copied().collect()
        } else {
            chunk.to_vec()
        };
        if pass_gates.len() < MULTI_FUSED_TILE_MIN_GATES {
            for &g in &pass_gates {
                let (target, mat) = gates[g];
                launch_apply_gate_1q(ctx, state, target, mat)?;
            }
            continue;
        }

        // Tile qubits: the anchor, this chunk's targets, then untargeted filler until
        // the tile is full. Sorted ascending, as the kernel's bit insertion requires.
        let mut tile = [0usize; MULTI_FUSED_TILE_Q];
        let mut in_tile = [false; MAX_GPU_QUBITS];
        for (slot, q) in tile.iter_mut().zip(0..MULTI_FUSED_ANCHOR_Q) {
            *slot = q;
        }
        in_tile[..MULTI_FUSED_ANCHOR_Q].fill(true);
        let mut filled = MULTI_FUSED_ANCHOR_Q;
        for &g in chunk {
            let q = gates[g].0;
            tile[filled] = q;
            in_tile[q] = true;
            filled += 1;
        }
        for untargeted_only in [true, false] {
            for q in MULTI_FUSED_ANCHOR_Q..n {
                if filled == MULTI_FUSED_TILE_Q {
                    break;
                }
                if in_tile[q] || (untargeted_only && is_target[q]) {
                    continue;
                }
                tile[filled] = q;
                in_tile[q] = true;
                filled += 1;
            }
        }
        tile.sort_unstable();
        let position = |q: usize| tile.iter().position(|&t| t == q).unwrap() as i32;

        let dim: u64 = 1u64 << n;
        let num_tiles = dim / MULTI_FUSED_TILE_SIZE;
        let (stream, func) = stream_and_fn(ctx, "apply_multi_fused_tiled")?;
        let cfg = linear_cfg(MULTI_FUSED_BLOCK_SIZE, num_tiles as u32);
        let num_gates_i = pass_gates.len() as i32;
        let mut arg = TileGatesArg {
            v: [0.0; 8 * MULTI_FUSED_TILE_Q],
            qubits: 0,
            targets: 0,
        };
        for (k, &q) in tile.iter().enumerate() {
            arg.qubits |= (q as u64) << (6 * k);
        }
        for (slot, &g) in pass_gates.iter().enumerate() {
            arg.targets |= (position(gates[g].0) as u64) << (4 * slot);
            for (d, entry) in arg.v[8 * slot..8 * slot + 8]
                .chunks_exact_mut(2)
                .zip(gates[g].1.iter().flatten())
            {
                d[0] = entry.re;
                d[1] = entry.im;
            }
        }
        let mut builder = stream.launch_builder(&func);
        let buffer = state.buffer_mut().raw_mut();
        builder.arg(buffer).arg(&dim).arg(&arg).arg(&num_gates_i);
        // SAFETY: signature matches; num_tiles * TILE_SIZE = dim; the tile holds TILE_Q
        // distinct qubits below n, every target position is inside it, and the gate list
        // rides in parameter space with num_gates entries filled.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("apply_multi_fused_tiled", e))?;
        }
    }
    Ok(())
}
