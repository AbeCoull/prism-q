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
const KERNEL_SOURCE_TEMPLATE: &str = include_str!("dense.cu");

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

/// `sum_j conj(psi[j ^ xmask]) psi[j] (-1)^{popcount(j & zmask)}` per
/// `(xmask, zmask)` pair with `pending_norm²` applied: the complex accumulator
/// behind `<P>` before the `i^{num_y}` factor and the norm. Two launches and a
/// `16 * masks.len()` byte readback replace a full-state export.
pub(crate) fn pauli_sums(
    ctx: &GpuContext,
    state: &GpuState,
    masks: &[(u64, u64)],
) -> Result<Vec<Complex64>> {
    if masks.is_empty() {
        return Ok(Vec::new());
    }
    let dim: u64 = 1u64 << state.num_qubits();
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let blocks_per_mask = dim.div_ceil(elems_per_block).max(1) as u32;
    let num_masks = super::require_u32("sv_pauli_expect", "masks", masks.len())?;
    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("sv_pauli_expect")?;
    let stage2 = device.function("dm_pauli_expect_finalize")?;
    let xmasks: Vec<u64> = masks.iter().map(|m| m.0).collect();
    let zmasks: Vec<u64> = masks.iter().map(|m| m.1).collect();
    let shared_bytes = 2 * BLOCK_SIZE * std::mem::size_of::<f64>() as u32;

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partial_len = 2 * masks.len() * blocks_per_mask as usize;
    super::ensure_capacity(&mut scratch.measure_partials, device, partial_len)?;
    super::ensure_scratch(&mut scratch.u64_a, device, &xmasks)?;
    super::ensure_scratch(&mut scratch.u64_b, device, &zmasks)?;
    super::ensure_exact(&mut scratch.pauli_result, device, 2 * masks.len())?;
    let partials = scratch.measure_partials.as_mut().unwrap();
    {
        let cfg = LaunchConfig {
            grid_dim: (blocks_per_mask, num_masks, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let mut builder = stream.launch_builder(&stage1);
        builder
            .arg(state.buffer().raw())
            .arg(&dim)
            .arg(scratch.u64_a.as_ref().unwrap().raw())
            .arg(scratch.u64_b.as_ref().unwrap().raw())
            .arg(partials.raw_mut());
        // SAFETY: signature matches the kernel. `blocks_per_mask` blocks of two
        // amplitudes per thread cover the `dim` amplitudes of each of the
        // `num_masks` masks, every partner read `j ^ x` stays below `dim`
        // because `x < dim` (masks are built from validated qubit indices), and
        // `partials` holds two f64s per (mask, block). All buffers are held by
        // the scratch guard.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| launch_err("sv_pauli_expect", e))?;
        }
    }
    let result = scratch.pauli_result.as_mut().unwrap();
    {
        let cfg = LaunchConfig {
            grid_dim: (num_masks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
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
    let norm_sq = state.pending_norm() * state.pending_norm();
    Ok(host
        .chunks_exact(2)
        .map(|pair| Complex64::new(pair[0], pair[1]) * norm_sq)
        .collect())
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
