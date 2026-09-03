//! Density-matrix kernels over the embedded `2n`-qubit buffer, CUDA C source plus
//! launch helpers.
//!
//! The buffer is the statevector layout of `4^n` amplitudes with the ket index in
//! the high `n` bits and the bra index in the low `n`, so the unitary half of the
//! backend reuses the dense kernels unchanged. The kernels here are the channel,
//! measurement, and readout sweeps that have no statevector counterpart.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::{PrismError, Result};

use super::super::{GpuBuffer, GpuContext, GpuState};
use super::{ensure_scratch, launch_err, linear_cfg, stream_and_fn};

const BLOCK_SIZE: u32 = 256;

pub(crate) const KERNEL_SOURCE: &str = r#"
// ============================================================================
// Density-matrix sweeps. `n` is the circuit width; the buffer holds 4^n
// amplitudes indexed (ket << n) | bra.
// ============================================================================

__device__ __forceinline__ unsigned long long dm_insert_zero(unsigned long long m, int pos)
{
    unsigned long long low = (1ULL << pos) - 1ULL;
    return ((m & ~low) << 1) | (m & low);
}

__device__ __forceinline__ double2 dm_mul(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Base of the (q0, q1) block for compacted index m: a zero bit inserted at
// each of the four block positions in ascending order.
__device__ __forceinline__ unsigned long long dm_block_base(
    unsigned long long m, int q0, int q1, int n)
{
    int p[4] = {q0, q1, q0 + n, q1 + n};
    for (int i = 1; i < 4; ++i) {
        int v = p[i];
        int j = i - 1;
        while (j >= 0 && p[j] > v) { p[j + 1] = p[j]; --j; }
        p[j + 1] = v;
    }
    unsigned long long base = m;
    for (int i = 0; i < 4; ++i) base = dm_insert_zero(base, p[i]);
    return base;
}

// Offset of block slot 4 * tr + tc from the block base.
__device__ __forceinline__ unsigned long long dm_block_offset(int slot, int q0, int q1, int n)
{
    int tr = slot >> 2;
    int tc = slot & 3;
    unsigned long long off = 0;
    if (tr & 2) off |= 1ULL << (q0 + n);
    if (tr & 1) off |= 1ULL << (q1 + n);
    if (tc & 2) off |= 1ULL << q0;
    if (tc & 1) off |= 1ULL << q1;
    return off;
}

// out[r] = Re rho[r][r]. Launch over d = 2^n threads.
extern "C" __global__ void dm_diagonal(const double2 *state, unsigned long long d, double *out)
{
    unsigned long long r = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= d) return;
    out[r] = state[r * d + r].x;
}

// Per-block partials of sum |a|^2 over the whole buffer; finalize with
// measure_prob_one_finalize.
extern "C" __global__ void dm_norm_sqr(const double2 *state, unsigned long long len, double *out_partials)
{
    extern __shared__ double sdata[];
    unsigned long long tid = threadIdx.x;
    unsigned long long i = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    double s = 0.0;
    if (i < len) { double2 a = state[i]; s += a.x * a.x + a.y * a.y; }
    unsigned long long i2 = i + blockDim.x;
    if (i2 < len) { double2 a = state[i2]; s += a.x * a.x + a.y * a.y; }
    sdata[tid] = s;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out_partials[blockIdx.x] = sdata[0];
}

// Keep the entries whose row and column both agree with `outcome` on the
// measured qubit, scaled by `scale`; zero the rest.
extern "C" __global__ void dm_project(
    double2 *state, unsigned long long len,
    unsigned long long rmask, unsigned long long cmask, int outcome, double scale)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int row = ((i & rmask) != 0ULL) ? 1 : 0;
    int col = ((i & cmask) != 0ULL) ? 1 : 0;
    double2 a = state[i];
    if (row == outcome && col == outcome) {
        state[i] = make_double2(a.x * scale, a.y * scale);
    } else {
        state[i] = make_double2(0.0, 0.0);
    }
}

// rho -> |0><0| (x) tr_q rho. One thread per (row-clear, col-clear) entry owns
// its three siblings, so no two threads touch the same amplitude.
extern "C" __global__ void dm_reset(
    double2 *state, unsigned long long groups, int qubit, int n)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_insert_zero(dm_insert_zero(m, qubit), qubit + n);
    unsigned long long rmask = 1ULL << (qubit + n);
    unsigned long long cmask = 1ULL << qubit;
    double2 a = state[base];
    double2 b = state[base | rmask | cmask];
    state[base] = make_double2(a.x + b.x, a.y + b.y);
    state[base | rmask] = make_double2(0.0, 0.0);
    state[base | cmask] = make_double2(0.0, 0.0);
    state[base | rmask | cmask] = make_double2(0.0, 0.0);
}

extern "C" __global__ void dm_conjugate(double2 *state, unsigned long long len)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    state[i].y = -state[i].y;
}

// rho[r][c] *= f(r) * conj(f(c)) for the 2^n-entry phase table `table`.
extern "C" __global__ void dm_diagonal_sandwich(
    double2 *state, unsigned long long len, int n, const double2 *table)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned long long d = 1ULL << n;
    double2 k = table[i >> n];
    double2 b = table[i & (d - 1ULL)];
    b.y = -b.y;
    state[i] = dm_mul(state[i], dm_mul(k, b));
}

// Diagonal 16-entry block superoperator: one complex multiply per amplitude,
// slot 4 * tr + tc read from the ket bits (tr) and bra bits (tc) of (q0, q1).
extern "C" __global__ void dm_kraus_2q_diagonal(
    double2 *state, unsigned long long len, int q0, int q1, int n, const double2 *diag)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    int slot = (int)(((i >> (q0 + n)) & 1ULL) << 3)
             | (int)(((i >> (q1 + n)) & 1ULL) << 2)
             | (int)(((i >> q0) & 1ULL) << 1)
             | (int)((i >> q1) & 1ULL);
    state[i] = dm_mul(state[i], diag[slot]);
}

// Dense 16x16 block superoperator `s` (row-major, 256 complex entries), one
// block of 16 amplitudes per thread.
extern "C" __global__ void dm_kraus_2q_dense(
    double2 *state, unsigned long long groups, int q0, int q1, int n, const double2 *s)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_block_base(m, q0, q1, n);
    unsigned long long idx[16];
    double2 v[16];
    for (int k = 0; k < 16; ++k) {
        idx[k] = base | dm_block_offset(k, q0, q1, n);
        v[k] = state[idx[k]];
    }
    for (int row = 0; row < 16; ++row) {
        double re = 0.0, im = 0.0;
        for (int col = 0; col < 16; ++col) {
            double2 c = s[row * 16 + col];
            re += c.x * v[col].x - c.y * v[col].y;
            im += c.x * v[col].y + c.y * v[col].x;
        }
        state[idx[row]] = make_double2(re, im);
    }
}

// Symmetric two-qubit depolarizing on each block: B -> alpha B + beta Tr(B) I.
extern "C" __global__ void dm_depolarizing_2q(
    double2 *state, unsigned long long groups, int q0, int q1, int n, double alpha, double beta)
{
    unsigned long long m = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= groups) return;
    unsigned long long base = dm_block_base(m, q0, q1, n);
    double tr_re = 0.0, tr_im = 0.0;
    for (int t = 0; t < 4; ++t) {
        double2 a = state[base | dm_block_offset(5 * t, q0, q1, n)];
        tr_re += a.x;
        tr_im += a.y;
    }
    double shift_re = tr_re * beta;
    double shift_im = tr_im * beta;
    for (int k = 0; k < 16; ++k) {
        unsigned long long i = base | dm_block_offset(k, q0, q1, n);
        double2 a = state[i];
        double re = a.x * alpha;
        double im = a.y * alpha;
        if ((k % 5) == 0) { re += shift_re; im += shift_im; }
        state[i] = make_double2(re, im);
    }
}

// rho = |psi><psi| from the 2^n amplitudes `amps`. Launch over 4^n threads.
extern "C" __global__ void dm_outer_product(
    double2 *state, unsigned long long len, int n, const double2 *amps)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;
    unsigned long long d = 1ULL << n;
    double2 k = amps[i >> n];
    double2 b = amps[i & (d - 1ULL)];
    b.y = -b.y;
    state[i] = dm_mul(k, b);
}

// Per-block complex partials of sum_j (-1)^{popcount(j & z)} rho[j][j ^ x] for
// mask k = blockIdx.y; j runs over the 2^n rows, two per thread. Partials are
// laid out [k][block][re, im].
extern "C" __global__ void dm_pauli_expect(
    const double2 *state, unsigned long long d,
    const unsigned long long *xmasks, const unsigned long long *zmasks, double *out_partials)
{
    extern __shared__ double sdata[];
    double *sr = sdata;
    double *si = sdata + blockDim.x;
    unsigned long long tid = threadIdx.x;
    unsigned int k = blockIdx.y;
    unsigned long long x = xmasks[k];
    unsigned long long z = zmasks[k];
    double re = 0.0, im = 0.0;
    unsigned long long j = (unsigned long long)blockIdx.x * (blockDim.x * 2) + tid;
    for (int rep = 0; rep < 2; ++rep) {
        if (j < d) {
            double2 e = state[j * d + (j ^ x)];
            double sign = (__popcll(j & z) & 1) ? -1.0 : 1.0;
            re += sign * e.x;
            im += sign * e.y;
        }
        j += blockDim.x;
    }
    sr[tid] = re;
    si[tid] = im;
    __syncthreads();
    for (unsigned long long stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        unsigned long long slot = ((unsigned long long)k * gridDim.x + blockIdx.x) * 2ULL;
        out_partials[slot] = sr[0];
        out_partials[slot + 1] = si[0];
    }
}

// One block per mask: sums that mask's `count` partial pairs into result[2k..].
extern "C" __global__ void dm_pauli_expect_finalize(
    const double *partials, unsigned int count, double *result)
{
    extern __shared__ double sdata[];
    double *sr = sdata;
    double *si = sdata + blockDim.x;
    unsigned int tid = threadIdx.x;
    unsigned int k = blockIdx.x;
    double re = 0.0, im = 0.0;
    for (unsigned int i = tid; i < count; i += blockDim.x) {
        unsigned long long slot = ((unsigned long long)k * count + i) * 2ULL;
        re += partials[slot];
        im += partials[slot + 1];
    }
    sr[tid] = re;
    si[tid] = im;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sr[tid] += sr[tid + stride];
            si[tid] += si[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[2 * k] = sr[0];
        result[2 * k + 1] = si[0];
    }
}
"#;

fn grid_for(count: u64) -> u32 {
    count.div_ceil(BLOCK_SIZE as u64).max(1) as u32
}

fn shared_cfg(grid: u32, columns: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: columns * BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
    }
}

/// Buffer length of the `n`-qubit mixture, `4^n`, and a check that the device
/// state is the embedded `2n`-qubit buffer.
fn buffer_len(state: &GpuState, n: usize) -> u64 {
    debug_assert_eq!(
        state.num_qubits(),
        2 * n,
        "device buffer is not the 2n-qubit embedding"
    );
    1u64 << (2 * n)
}

/// A device buffer of exactly `len` elements in `slot`, reallocated on any size
/// change so a readback of its full length matches the request.
fn ensure_exact<'a>(
    slot: &'a mut Option<GpuBuffer<f64>>,
    device: &super::super::GpuDevice,
    len: usize,
) -> Result<&'a mut GpuBuffer<f64>> {
    let len = len.max(1);
    if slot.as_ref().is_none_or(|buf| buf.len() != len) {
        *slot = Some(GpuBuffer::<f64>::alloc_zeros(device, len)?);
    }
    Ok(slot.as_mut().unwrap())
}

fn flatten(values: &[Complex64]) -> Vec<f64> {
    let mut flat = Vec::with_capacity(2 * values.len());
    for v in values {
        flat.push(v.re);
        flat.push(v.im);
    }
    flat
}

fn check_pair(n: usize, q0: usize, q1: usize) -> Result<()> {
    if q0 >= n || q1 >= n || q0 == q1 {
        return Err(PrismError::InvalidQubit {
            index: q0.max(q1),
            register_size: n,
        });
    }
    Ok(())
}

/// `Re rho[r][r]` for every `r`, the `2^n` diagonal of the `4^n` buffer.
pub(crate) fn diagonal(ctx: &GpuContext, state: &GpuState, n: usize) -> Result<Vec<f64>> {
    buffer_len(state, n);
    let d: u64 = 1u64 << n;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "dm_diagonal")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(d));
    let mut scratch = ctx.launcher_scratch();
    let out = ensure_exact(&mut scratch.dm_diag, device, d as usize)?;
    let mut builder = stream.launch_builder(&func);
    builder.arg(state.buffer().raw()).arg(&d).arg(out.raw_mut());
    // SAFETY: signature matches the kernel; the grid covers `d` rows, each read
    // at `r * d + r` inside the `d * d` buffer, and `out` holds `d` f64s.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_diagonal", e))?;
    }
    let mut host = vec![0.0_f64; d as usize];
    out.copy_to_host(device, &mut host)?;
    Ok(host)
}

/// `sum |rho[r][c]|^2` over the buffer, the purity `Tr(rho^2)`.
pub(crate) fn norm_sqr(ctx: &GpuContext, state: &GpuState, n: usize) -> Result<f64> {
    let len = buffer_len(state, n);
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = len.div_ceil(elems_per_block).max(1) as u32;
    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("dm_norm_sqr")?;
    let stage2 = device.function("measure_prob_one_finalize")?;
    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partials =
        super::ensure_capacity(&mut scratch.measure_partials, device, num_blocks as usize)?;
    {
        let mut builder = stream.launch_builder(&stage1);
        builder
            .arg(state.buffer().raw())
            .arg(&len)
            .arg(partials.raw_mut());
        // SAFETY: signature matches the kernel; `num_blocks` blocks of two
        // elements per thread cover `len`, and `partials` holds one f64 per block.
        unsafe {
            builder
                .launch(shared_cfg(num_blocks, 1))
                .map_err(|e| launch_err("dm_norm_sqr", e))?;
        }
    }
    let result = super::ensure_capacity(&mut scratch.measure_result, device, 1)?;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&num_blocks)
            .arg(result.raw_mut());
        // SAFETY: signature matches the kernel; one block strides over
        // `num_blocks` partials, both buffers held by the scratch guard.
        unsafe {
            builder
                .launch(shared_cfg(1, 1))
                .map_err(|e| launch_err("measure_prob_one_finalize", e))?;
        }
    }
    let mut host = [0.0_f64];
    scratch
        .measure_result
        .as_ref()
        .unwrap()
        .copy_to_host(device, &mut host)?;
    Ok(host[0])
}

/// Collapse onto the `outcome` subspace of `qubit`, scaling survivors by `scale`.
pub(crate) fn project(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    qubit: usize,
    outcome: bool,
    scale: f64,
) -> Result<()> {
    let len = buffer_len(state, n);
    let rmask: u64 = 1u64 << (qubit + n);
    let cmask: u64 = 1u64 << qubit;
    let outcome_i: i32 = i32::from(outcome);
    let (stream, func) = stream_and_fn(ctx, "dm_project")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(len));
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&len)
        .arg(&rmask)
        .arg(&cmask)
        .arg(&outcome_i)
        .arg(&scale);
    // SAFETY: signature matches the kernel and the grid covers `len`; each
    // thread touches its own amplitude only.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_project", e))?;
    }
    Ok(())
}

/// `rho -> |0><0| (x) tr_q rho` for `qubit`.
pub(crate) fn reset(ctx: &GpuContext, state: &mut GpuState, n: usize, qubit: usize) -> Result<()> {
    let groups = buffer_len(state, n) >> 2;
    let qubit_i = qubit as i32;
    let n_i = n as i32;
    let (stream, func) = stream_and_fn(ctx, "dm_reset")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(groups));
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&groups)
        .arg(&qubit_i)
        .arg(&n_i);
    // SAFETY: signature matches the kernel. Inserting a zero bit at `qubit` and
    // `qubit + n` is a bijection from the `4^n / 4` compacted indices onto the
    // block bases, so the four amplitudes one thread touches are disjoint from
    // every other thread's.
    unsafe {
        builder.launch(cfg).map_err(|e| launch_err("dm_reset", e))?;
    }
    Ok(())
}

pub(crate) fn conjugate(ctx: &GpuContext, state: &mut GpuState, n: usize) -> Result<()> {
    let len = buffer_len(state, n);
    let (stream, func) = stream_and_fn(ctx, "dm_conjugate")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(len));
    let mut builder = stream.launch_builder(&func);
    builder.arg(state.buffer_mut().raw_mut()).arg(&len);
    // SAFETY: signature matches the kernel and the grid covers `len`.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_conjugate", e))?;
    }
    Ok(())
}

/// `rho[r][c] *= table[r] * conj(table[c])` for a `2^n`-entry phase table.
pub(crate) fn diagonal_sandwich(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    table: &[Complex64],
) -> Result<()> {
    let len = buffer_len(state, n);
    debug_assert_eq!(table.len(), 1usize << n);
    let n_i = n as i32;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "dm_diagonal_sandwich")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(len));
    let mut scratch = ctx.launcher_scratch();
    let table_buf = ensure_scratch(&mut scratch.f64_a, device, &flatten(table))?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&len)
        .arg(&n_i)
        .arg(table_buf.raw());
    // SAFETY: signature matches the kernel; the grid covers `len`, every table
    // read is below `2^n`, and the table is held by the scratch guard.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_diagonal_sandwich", e))?;
    }
    Ok(())
}

/// One complex multiply per amplitude by the diagonal block superoperator
/// `diag`, indexed `4 * tr + tc`.
pub(crate) fn kraus_2q_diagonal(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    q0: usize,
    q1: usize,
    diag: &[Complex64; 16],
) -> Result<()> {
    check_pair(n, q0, q1)?;
    let len = buffer_len(state, n);
    let (q0_i, q1_i, n_i) = (q0 as i32, q1 as i32, n as i32);
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "dm_kraus_2q_diagonal")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(len));
    let mut scratch = ctx.launcher_scratch();
    let diag_buf = ensure_scratch(&mut scratch.f64_a, device, &flatten(diag))?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&len)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&n_i)
        .arg(diag_buf.raw());
    // SAFETY: signature matches the kernel; the grid covers `len` and the slot
    // index is four bits, inside the 16-entry table the scratch guard holds.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_kraus_2q_diagonal", e))?;
    }
    Ok(())
}

/// Dense 16x16 block superoperator sweep, `s` indexed `[4 * tr + tc][4 * trp + tcp]`.
pub(crate) fn kraus_2q_dense(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    q0: usize,
    q1: usize,
    s: &[[Complex64; 16]; 16],
) -> Result<()> {
    check_pair(n, q0, q1)?;
    let groups = buffer_len(state, n) >> 4;
    let (q0_i, q1_i, n_i) = (q0 as i32, q1 as i32, n as i32);
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "dm_kraus_2q_dense")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(groups));
    let flat: Vec<Complex64> = s.iter().flat_map(|row| row.iter().copied()).collect();
    let mut scratch = ctx.launcher_scratch();
    let s_buf = ensure_scratch(&mut scratch.f64_a, device, &flatten(&flat))?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&groups)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&n_i)
        .arg(s_buf.raw());
    // SAFETY: signature matches the kernel. Inserting a zero bit at each of the
    // four block positions is a bijection from the compacted index onto the
    // block bases, so the 16 amplitudes one thread touches are disjoint from
    // every other thread's; the superoperator is held by the scratch guard.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_kraus_2q_dense", e))?;
    }
    Ok(())
}

/// `B -> alpha B + beta Tr(B) I` on every `(q0, q1)` block.
pub(crate) fn depolarizing_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    q0: usize,
    q1: usize,
    alpha: f64,
    beta: f64,
) -> Result<()> {
    check_pair(n, q0, q1)?;
    let groups = buffer_len(state, n) >> 4;
    let (q0_i, q1_i, n_i) = (q0 as i32, q1 as i32, n as i32);
    let (stream, func) = stream_and_fn(ctx, "dm_depolarizing_2q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(groups));
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&groups)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&n_i)
        .arg(&alpha)
        .arg(&beta);
    // SAFETY: signature matches the kernel; block bases are disjoint as in
    // `kraus_2q_dense`, so each thread owns its 16 amplitudes.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_depolarizing_2q", e))?;
    }
    Ok(())
}

/// Fill the buffer with `|psi><psi|` for the `2^n` amplitudes `amps`.
pub(crate) fn outer_product(
    ctx: &GpuContext,
    state: &mut GpuState,
    n: usize,
    amps: &[Complex64],
) -> Result<()> {
    let len = buffer_len(state, n);
    debug_assert_eq!(amps.len(), 1usize << n);
    let n_i = n as i32;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "dm_outer_product")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(len));
    let mut scratch = ctx.launcher_scratch();
    let amps_buf = ensure_scratch(&mut scratch.f64_a, device, &flatten(amps))?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer_mut().raw_mut())
        .arg(&len)
        .arg(&n_i)
        .arg(amps_buf.raw());
    // SAFETY: signature matches the kernel; the grid covers `len`, every
    // amplitude read is below `2^n`, and the table is held by the scratch guard.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| launch_err("dm_outer_product", e))?;
    }
    Ok(())
}

/// `sum_j (-1)^{popcount(j & zmask)} rho[j][j ^ xmask]` per `(xmask, zmask)`
/// pair, the complex accumulator behind `Tr(rho P)` before the `i^{num_y}`
/// factor. Two launches and a `16 * masks.len()` byte readback.
pub(crate) fn pauli_sums(
    ctx: &GpuContext,
    state: &GpuState,
    n: usize,
    masks: &[(u64, u64)],
) -> Result<Vec<Complex64>> {
    buffer_len(state, n);
    if masks.is_empty() {
        return Ok(Vec::new());
    }
    let d: u64 = 1u64 << n;
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let blocks_per_mask = d.div_ceil(elems_per_block).max(1) as u32;
    let num_masks = super::require_u32("dm_pauli_expect", "masks", masks.len())?;
    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("dm_pauli_expect")?;
    let stage2 = device.function("dm_pauli_expect_finalize")?;
    let xmasks: Vec<u64> = masks.iter().map(|m| m.0).collect();
    let zmasks: Vec<u64> = masks.iter().map(|m| m.1).collect();

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partial_len = 2 * masks.len() * blocks_per_mask as usize;
    super::ensure_capacity(&mut scratch.measure_partials, device, partial_len)?;
    ensure_scratch(&mut scratch.u64_a, device, &xmasks)?;
    ensure_scratch(&mut scratch.u64_b, device, &zmasks)?;
    ensure_exact(&mut scratch.dm_result, device, 2 * masks.len())?;
    let partials = scratch.measure_partials.as_mut().unwrap();
    {
        let cfg = LaunchConfig {
            grid_dim: (blocks_per_mask, num_masks, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 2 * BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
        };
        let mut builder = stream.launch_builder(&stage1);
        builder
            .arg(state.buffer().raw())
            .arg(&d)
            .arg(scratch.u64_a.as_ref().unwrap().raw())
            .arg(scratch.u64_b.as_ref().unwrap().raw())
            .arg(partials.raw_mut());
        // SAFETY: signature matches the kernel. `blocks_per_mask` blocks of two
        // rows per thread cover the `d` rows of each of the `num_masks` masks,
        // every read `j * d + (j ^ x)` stays inside the `d * d` buffer because
        // `x < d`, and `partials` holds two f64s per (mask, block).
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("dm_pauli_expect", e))?;
        }
    }
    let result = scratch.dm_result.as_mut().unwrap();
    {
        let cfg = LaunchConfig {
            grid_dim: (num_masks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 2 * BLOCK_SIZE * std::mem::size_of::<f64>() as u32,
        };
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&blocks_per_mask)
            .arg(result.raw_mut());
        // SAFETY: signature matches the kernel; one block per mask strides over
        // that mask's `blocks_per_mask` partial pairs, and `result` holds two
        // f64s per mask. Both buffers are held by the scratch guard.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("dm_pauli_expect_finalize", e))?;
        }
    }
    let mut host = vec![0.0_f64; 2 * masks.len()];
    result.copy_to_host(device, &mut host)?;
    Ok(host
        .chunks_exact(2)
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect())
}
