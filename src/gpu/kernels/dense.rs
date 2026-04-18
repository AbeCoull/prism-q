//! Dense statevector kernels — CUDA C source compiled to PTX at runtime, plus Rust-side
//! launch helpers.
//!
//! The state buffer is `2 * 2^n` f64s laid out as interleaved (re, im) pairs matching
//! `num_complex::Complex64` and CUDA's `double2` builtin. All kernels take the buffer as
//! `double2 *` for 16-byte aligned vector loads.
//!
//! # Fused-gate strategy
//!
//! Fused variants (`MultiFused`, `Multi2q`, `BatchPhase`, `BatchRzz`, `DiagonalBatch`) are
//! decomposed on the host into the appropriate per-element kernel launches. This avoids
//! host-side variable-length data uploads and keeps the PTX surface small. Optimised batched
//! kernels are a future refinement.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::{PrismError, Result};

use super::super::{GpuContext, GpuState};

const BLOCK_SIZE: u32 = 256;

pub(crate) const KERNEL_SOURCE: &str = r#"
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
"#;

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
