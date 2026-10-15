//! Dense statevector kernels: the CUDA C template and its launchers. Kernels take the
//! interleaved state buffer as `double2 *` for 16-byte aligned vector loads.
//!
//! `BatchPhase`, `BatchRzz`, `DiagonalBatch`, and diagonal `MultiFused` each run as one
//! launch over per-group phase tables (built by the CPU `build_*_tables` helpers) plus
//! small shift and qubit metadata, instead of one launch per sub-gate. Non-diagonal
//! `MultiFused` runs as shared-memory tiled passes of `apply_multi_fused_tiled`;
//! `Multi2q` launches once per sub-gate, on the permutation or diagonal kernel when the
//! sub-gate's matrix is exactly one.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::backend::statevector::kernels as cpu_k;
use crate::error::{PrismError, Result};

use super::super::tiles::{Geometry, TileOp};
use super::super::{GpuContext, GpuState};
use super::{check_pair, check_qubit, driver_err, launch_limit_err, linear_cfg, stream_and_fn};

const BLOCK_SIZE: u32 = 256;

/// The fused 2q matrix as a by-value kernel parameter, `(re, im - re, re + im)` per
/// entry, row-major. Mirrors `struct FusedMat` in the CUDA template byte for byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct FusedMatArg {
    v: [f64; 48],
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

/// CUDA C source template. [`kernel_source`] fills its `{{NAME}}` placeholders so the
/// kernel's compile-time constants track the CPU constants in
/// [`crate::backend::statevector::kernels`].
const KERNEL_SOURCE_TEMPLATE: &str = include_str!("dense.cu");

/// Substitute the host constants into [`KERNEL_SOURCE_TEMPLATE`]. A new placeholder in
/// the template needs a matching `.replace` here.
pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE_TEMPLATE
        .replace("{{TILE_Q}}", &MULTI_FUSED_TILE_Q.to_string())
        .replace("{{TILE_SIZE}}", &MULTI_FUSED_TILE_SIZE.to_string())
        .replace("{{PAULI_GROUP_MASKS}}", &PAULI_GROUP_MASKS.to_string())
        .replace(
            "{{PAULI_AMPS_PER_THREAD}}",
            &PAULI_AMPS_PER_THREAD.to_string(),
        )
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
        .replace("{{QFT_TILE_Q}}", &QFT_TILE_Q.to_string())
        .replace("{{MIX_TILE_Q}}", &MIXED_TILE_Q.to_string())
        .replace("{{MIX_BLOCK_LOG}}", &MIXED_BLOCK_LOG.to_string())
        .replace("{{MIX_MAX_OPS}}", &MIXED_MAX_OPS.to_string())
        .replace("{{MIX_MAX_COEFS}}", &MIXED_MAX_COEFS.to_string())
}

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
            .map_err(|e| driver_err("set_initial_state", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

pub(crate) fn launch_compute_probabilities(ctx: &GpuContext, state: &GpuState) -> Result<Vec<f64>> {
    use super::super::GpuBuffer;
    let n = state.num_qubits();
    let dim: u64 = 1u64 << n;
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "compute_probabilities")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));

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
            .map_err(|e| driver_err("compute_probabilities", e))?;
    }
    ctx.device().count_launch(1);
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
    check_qubit(n, target)?;
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
            .map_err(|e| driver_err("apply_gate_1q", e))?;
    }
    ctx.device().count_launch(1);
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
    check_qubit(n, target)?;
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
            .map_err(|e| driver_err("apply_diagonal_1q", e))?;
    }
    ctx.device().count_launch(1);
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
    check_pair(n, q0, q1)?;
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
        builder.launch(cfg).map_err(|e| driver_err(kernel, e))?;
    }
    device.count_launch(1);
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
    check_pair(n, q0, q1)?;
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
            .map_err(|e| driver_err("apply_parity_phase", e))?;
    }
    ctx.device().count_launch(1);
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
    check_pair(n, control, target)?;
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
        builder.launch(cfg).map_err(|e| driver_err("apply_cu", e))?;
    }
    ctx.device().count_launch(1);
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
    check_pair(n, control, target)?;
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
            .map_err(|e| driver_err("apply_cu_phase", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

fn validate_mcu_qubits(n: usize, controls: &[usize], target: usize) -> Result<Vec<u32>> {
    for &c in controls {
        check_qubit(n, c)?;
        if c == target {
            return Err(PrismError::InvalidParameter {
                message: "control qubit equals target".to_string(),
            });
        }
    }
    check_qubit(n, target)?;
    let mut sorted: Vec<u32> = controls.iter().map(|&q| q as u32).collect();
    sorted.push(target as u32);
    sorted.sort_unstable();
    Ok(sorted)
}

/// The sorted control-plus-target qubit list rides in parameter space for index
/// expansion.
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
            .map_err(|e| driver_err("apply_mcu", e))?;
    }
    ctx.device().count_launch(1);
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
            .map_err(|e| driver_err("apply_mcu_phase", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Kernel route for a 4x4 matrix in the `apply_fused_2q` basis, where `q0` is the high
/// bit. Structural zeros and ones only, no tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Route2q {
    /// `CX` with the high basis bit (`q0`) as control when `control_high`.
    Cx {
        control_high: bool,
    },
    Swap,
    Cz,
    ControlledPhase(Complex64),
    /// `out[r] = scale[r] * in[src[r]]`, a permutation with phases or a diagonal.
    Monomial {
        src: [u8; 4],
        scale: [Complex64; 4],
    },
    Dense,
}

pub(crate) fn route_2q(matrix: &[[Complex64; 4]; 4]) -> Route2q {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut src = [0u8; 4];
    let mut scale = [zero; 4];
    let mut used = 0u8;
    for (r, row) in matrix.iter().enumerate() {
        let mut nonzero = row.iter().enumerate().filter(|(_, v)| **v != zero);
        let Some((c, &v)) = nonzero.next() else {
            return Route2q::Dense;
        };
        if nonzero.next().is_some() || used & (1 << c) != 0 {
            return Route2q::Dense;
        }
        used |= 1 << c;
        src[r] = c as u8;
        scale[r] = v;
    }
    if scale.iter().all(|&v| v == one) {
        match src {
            [0, 1, 3, 2] => return Route2q::Cx { control_high: true },
            [0, 3, 2, 1] => {
                return Route2q::Cx {
                    control_high: false,
                };
            }
            [0, 2, 1, 3] => return Route2q::Swap,
            _ => {}
        }
    }
    if src == [0, 1, 2, 3] && scale[..3].iter().all(|&v| v == one) {
        return if scale[3] == -one {
            Route2q::Cz
        } else {
            Route2q::ControlledPhase(scale[3])
        };
    }
    Route2q::Monomial { src, scale }
}

/// Apply a 4x4 matrix through the cheapest kernel its exact structure admits:
/// `apply_cx`, `apply_swap`, `apply_cz`, `apply_cu_phase`, `apply_monomial_2q`, or the
/// dense `apply_fused_2q`.
pub(crate) fn launch_apply_matrix_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    matrix: &[[Complex64; 4]; 4],
) -> Result<()> {
    match route_2q(matrix) {
        Route2q::Cx { control_high: true } => launch_apply_cx(ctx, state, q0, q1),
        Route2q::Cx {
            control_high: false,
        } => launch_apply_cx(ctx, state, q1, q0),
        Route2q::Swap => launch_apply_swap(ctx, state, q0, q1),
        Route2q::Cz => launch_apply_cz(ctx, state, q0, q1),
        Route2q::ControlledPhase(phase) => launch_apply_cu_phase(ctx, state, q0, q1, phase),
        Route2q::Monomial { src, scale } => {
            launch_apply_monomial_2q(ctx, state, q0, q1, src, &scale)
        }
        Route2q::Dense => launch_apply_fused_2q(ctx, state, q0, q1, matrix),
    }
}

/// Monomial scales as a by-value kernel parameter, re/im pairs per output row; mirrors
/// `struct MonoScale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MonoScaleArg {
    v: [f64; 8],
}

// SAFETY: a plain `repr(C)` array of f64 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for MonoScaleArg {}

fn launch_apply_monomial_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    src: [u8; 4],
    scale: &[Complex64; 4],
) -> Result<()> {
    let n = state.num_qubits();
    check_pair(n, q0, q1)?;
    let pair_count: u64 = 1u64 << (n - 2);
    let (stream, func) = stream_and_fn(ctx, "apply_monomial_2q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
    let mut arg = MonoScaleArg { v: [0.0; 8] };
    for (slot, entry) in arg.v.as_chunks_mut::<2>().0.iter_mut().zip(scale) {
        slot[0] = entry.re;
        slot[1] = entry.im;
    }
    let packed_src: u32 = src
        .iter()
        .enumerate()
        .map(|(r, &c)| u32::from(c) << (2 * r))
        .sum();
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&pair_count)
        .arg(&q0_i)
        .arg(&q1_i)
        .arg(&arg)
        .arg(&packed_src);
    // SAFETY: signature matches; the scales ride in parameter space and the grid covers
    // pair_count.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("apply_monomial_2q", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// The matrix rides in Gauss form as a by-value kernel parameter.
pub(crate) fn launch_apply_fused_2q(
    ctx: &GpuContext,
    state: &mut GpuState,
    q0: usize,
    q1: usize,
    matrix: &[[Complex64; 4]; 4],
) -> Result<()> {
    let n = state.num_qubits();
    check_pair(n, q0, q1)?;
    let pair_count: u64 = 1u64 << (n - 2);
    let (stream, func) = stream_and_fn(ctx, "apply_fused_2q")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count));
    let mut mat = FusedMatArg { v: [0.0; 48] };
    for (slot, entry) in mat
        .v
        .as_chunks_mut::<3>()
        .0
        .iter_mut()
        .zip(matrix.iter().flatten())
    {
        *slot = [entry.re, entry.im - entry.re, entry.re + entry.im];
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
            .map_err(|e| driver_err("apply_fused_2q", e))?;
    }
    ctx.device().count_launch(1);
    ctx.device().count_dense_2q();
    Ok(())
}

/// Two-stage device reduction (per-block partials, then a single-block
/// finalize); one f64 crosses PCIe.
pub(crate) fn measure_prob_one(ctx: &GpuContext, state: &GpuState, qubit: usize) -> Result<f64> {
    let n = state.num_qubits();
    check_qubit(n, qubit)?;
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
                .map_err(|e| driver_err("measure_prob_one", e))?;
        }
        device.count_launch(1);
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
                .map_err(|e| driver_err("measure_prob_one_finalize", e))?;
        }
        device.count_launch(0);
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
    check_qubit(n, qubit)?;
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
                .map_err(|e| driver_err("rdm_qubit", e))?;
        }
        device.count_launch(1);
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
                .map_err(|e| driver_err("rdm_qubit_finalize", e))?;
        }
        device.count_launch(0);
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
/// `16 * masks.len()` byte readback replace a full-state export. Masks sharing an
/// `xmask` run in groups of up to [`PAULI_GROUP_MASKS`], one state pass per group.
pub(crate) fn pauli_sums(
    ctx: &GpuContext,
    state: &GpuState,
    masks: &[(u64, u64)],
) -> Result<Vec<Complex64>> {
    if masks.is_empty() {
        return Ok(Vec::new());
    }
    let dim: u64 = 1u64 << state.num_qubits();
    let elems_per_block = (PAULI_AMPS_PER_THREAD * BLOCK_SIZE as usize) as u64;
    let blocks_per_mask = dim.div_ceil(elems_per_block).max(1) as u32;
    let num_masks = super::require_u32("sv_pauli_expect", "masks", masks.len())?;
    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("sv_pauli_expect")?;

    let mut order: Vec<usize> = (0..masks.len()).collect();
    order.sort_by_key(|&k| masks[k].0);
    let mut group_x = Vec::new();
    let mut group_start = Vec::new();
    for (position, &k) in order.iter().enumerate() {
        let opens = match group_start.last() {
            Some(&start) => {
                group_x.last() != Some(&masks[k].0)
                    || position - start as usize == PAULI_GROUP_MASKS
            }
            None => true,
        };
        if opens {
            group_x.push(masks[k].0);
            group_start.push(position as u64);
        }
    }
    let num_groups = super::require_u32("sv_pauli_expect", "groups", group_x.len())?;
    let mut groups = group_x;
    groups.extend(group_start);
    groups.push(masks.len() as u64);
    let members: Vec<u64> = order
        .iter()
        .map(|&k| masks[k].1)
        .chain(order.iter().map(|&k| k as u64))
        .collect();

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partial_len = 2 * masks.len() * blocks_per_mask as usize;
    super::ensure_capacity(&mut scratch.measure_partials, device, partial_len)?;
    super::ensure_scratch(&mut scratch.u64_a, device, &groups)?;
    super::ensure_scratch(&mut scratch.u64_b, device, &members)?;
    let partials = scratch.measure_partials.as_mut().unwrap();
    {
        let cfg = LaunchConfig {
            grid_dim: (blocks_per_mask, num_groups, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&stage1);
        builder
            .arg(state.buffer().raw())
            .arg(&dim)
            .arg(scratch.u64_a.as_ref().unwrap().raw())
            .arg(&num_groups)
            .arg(scratch.u64_b.as_ref().unwrap().raw())
            .arg(&num_masks)
            .arg(partials.raw_mut());
        // SAFETY: signature matches the kernel. `blocks_per_mask` blocks of
        // `PAULI_AMPS_PER_THREAD` amplitudes per thread cover the `dim` amplitudes
        // of each of the `num_groups` groups of at most `PAULI_GROUP_MASKS`
        // members, every partner read `j ^ x` stays below `dim` because
        // `x < dim` (masks are built from validated qubit indices), and every
        // member slot is a distinct mask index, so `partials` holds two f64s per
        // (slot, block). All buffers are held by the scratch guard.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("sv_pauli_expect", e))?;
        }
        device.count_launch(u64::from(num_groups));
    }
    let norm_sq = state.pending_norm() * state.pending_norm();
    super::finalize_pauli_sums(
        device,
        scratch,
        num_masks,
        blocks_per_mask,
        BLOCK_SIZE,
        norm_sq,
    )
}

/// First two moments `(<h>, <h^2>)` of each group operator
/// `h(j) = sum_i c_i (-1)^popcount(j & z_i)`, one `(zmasks, coefficients, x_bits,
/// y_bits)` request per group, read in the basis `rotate_to_z_basis` gives for
/// `x_bits` and `y_bits`. A rotated group runs on a device copy of the state, rotated
/// by the tiled 1q passes, so the state is left as it was. `Ok(None)` when that copy
/// does not fit in device memory.
pub(crate) fn group_moments(
    ctx: &GpuContext,
    state: &GpuState,
    groups: &[(Vec<u64>, Vec<f64>, u64, u64)],
) -> Result<Option<Vec<(f64, f64)>>> {
    let mut rotated = None;
    if groups.iter().any(|&(_, _, x, y)| x | y != 0) {
        match state.duplicate() {
            Ok(copy) => rotated = Some(copy),
            Err(_) => return Ok(None),
        }
    }
    let n = state.num_qubits();
    let mut fresh = true;
    let mut out = Vec::with_capacity(groups.len());
    for (zmasks, coefficients, x_bits, y_bits) in groups {
        let rotation = x_bits | y_bits;
        let source = if rotation == 0 {
            state
        } else {
            let copy = rotated.as_mut().unwrap();
            if !fresh {
                copy.buffer_mut()
                    .copy_from_device(ctx.device(), state.buffer())?;
            }
            fresh = false;
            let gates: Vec<(usize, [[Complex64; 2]; 2])> = (0..n)
                .filter(|&q| rotation >> q & 1 == 1)
                .map(|q| {
                    let t = if y_bits >> q & 1 == 1 {
                        Complex64::new(0.0, -1.0)
                    } else {
                        Complex64::ONE
                    };
                    (q, [[Complex64::ONE, t], [Complex64::ONE, -t]])
                })
                .collect();
            launch_apply_multi_fused_nondiag(ctx, copy, &gates)?;
            &*copy
        };
        out.push(group_moments_pass(ctx, source, zmasks, coefficients)?);
    }
    Ok(Some(out))
}

/// `(<h>, <h^2>)` of one group operator over `state` through `sv_group_moments`. The
/// kernel also sums `p`, so the ratio cancels the pending norm and any rotation growth.
fn group_moments_pass(
    ctx: &GpuContext,
    state: &GpuState,
    zmasks: &[u64],
    coefficients: &[f64],
) -> Result<(f64, f64)> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("sv_group_moments")?;
    let dim: u64 = 1u64 << state.num_qubits();
    let num_terms = super::require_u32("sv_group_moments", "terms", zmasks.len())?;
    let grid = dim
        .div_ceil(64)
        .div_ceil(u64::from(BLOCK_SIZE / 32))
        .min(MOMENT_MAX_BLOCKS) as u32;

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    super::ensure_scratch(&mut scratch.u64_a, device, zmasks)?;
    super::ensure_scratch(&mut scratch.f64_a, device, coefficients)?;
    let result = super::ensure_exact(&mut scratch.moment_partials, device, 3 * grid as usize)?;
    let cfg = linear_cfg(BLOCK_SIZE, grid);
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer().raw())
        .arg(&dim)
        .arg(scratch.u64_a.as_ref().unwrap().raw())
        .arg(scratch.f64_a.as_ref().unwrap().raw())
        .arg(&num_terms)
        .arg(result.raw_mut());
    // SAFETY: signature matches the kernel. The state holds `dim` amplitudes, the mask
    // and coefficient buffers hold at least `num_terms` entries, every read index is
    // checked against `dim`, and `result` holds three f64s per block. All buffers are
    // held by the scratch guard until the readback below drains the stream.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("sv_group_moments", e))?;
    }
    ctx.device().count_launch(1);
    let mut partials = vec![0.0_f64; 3 * grid as usize];
    result.copy_to_host(device, &mut partials)?;
    let (s0, s1, s2) = partials
        .as_chunks::<3>()
        .0
        .iter()
        .fold((0.0, 0.0, 0.0), |acc, p| {
            (acc.0 + p[0], acc.1 + p[1], acc.2 + p[2])
        });
    Ok(if s0 == 0.0 {
        (0.0, 0.0)
    } else {
        (s1 / s0, s2 / s0)
    })
}

/// Grid cap for `sv_group_moments`; warps stride over the remaining 64-index blocks.
const MOMENT_MAX_BLOCKS: u64 = 1024;

/// Masks one `sv_pauli_expect` block reduces per state pass.
const PAULI_GROUP_MASKS: usize = 8;

/// Amplitudes each `sv_pauli_expect` thread reads before the block reduction.
const PAULI_AMPS_PER_THREAD: usize = 16;

/// Zeroes the losing branch only; renormalization is deferred to the caller
/// via `pending_norm`.
pub(crate) fn measure_collapse(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    outcome: bool,
) -> Result<()> {
    let n = state.num_qubits();
    check_qubit(n, qubit)?;
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
            .map_err(|e| driver_err("measure_collapse", e))?;
    }
    ctx.device().count_launch(1);
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
        check_qubit(n, target)?;
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
            .map_err(|e| driver_err("apply_multi_fused_diagonal", e))?;
    }
    ctx.device().count_launch(1);
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
    check_qubit(n, control)?;
    for &(q, _) in phases {
        check_qubit(n, q)?;
    }

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
            .map_err(|e| driver_err("apply_batch_phase", e))?;
    }
    ctx.device().count_launch(1);
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
            .map_err(|e| driver_err("apply_batch_rzz", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Apply a `DiagonalBatch` in one launch, or per entry when the entries span more groups
/// than the tables allow (the CPU kernel's fallback condition).
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
        for entry in entries {
            match *entry {
                DiagEntry::Phase1q { qubit, d0, d1 } => {
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
            .map_err(|e| driver_err("apply_diagonal_batch", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Most DIF stages one `qft_dif_pass` block runs: its shared-memory tile holds
/// `2^QFT_TILE_Q` amplitudes, 32 KB.
const QFT_TILE_Q: usize = 11;

/// Log2 of the fewest consecutive amplitudes a gathered row carries above the field's
/// lowest pass: 64 bytes, so a strided row read fills whole sectors.
const QFT_MIN_ROW_LOG: usize = 2;

const QFT_BLOCK_SIZE: u32 = 256;

/// One `qft_dif_pass`: DIF stages `lo..lo + bits` of the field, in field bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QftPass {
    lo: usize,
    bits: usize,
}

/// DIF passes for a `num`-qubit field at `start`, top stage first: the stages above the
/// lowest pass split evenly into passes of at most `QFT_TILE_Q - QFT_MIN_ROW_LOG`.
fn qft_pass_plan(start: usize, num: usize) -> Vec<QftPass> {
    let low = num.min(QFT_TILE_Q - start.min(QFT_MIN_ROW_LOG));
    let passes = (num - low).div_ceil(QFT_TILE_Q - QFT_MIN_ROW_LOG);
    let mut plan = Vec::with_capacity(passes + 1);
    let mut top = num;
    for left in (1..=passes).rev() {
        let bits = (top - low).div_ceil(left);
        plan.push(QftPass {
            lo: top - bits,
            bits,
        });
        top -= bits;
    }
    plan.push(QftPass { lo: 0, bits: low });
    plan
}

/// Split of the `lo + bits` phase bits over the low, mid and high tables of a pass.
fn qft_phase_split(pass: QftPass) -> [usize; 3] {
    let total = pass.lo + pass.bits;
    let low = total.div_ceil(3);
    let mid = (total - low).div_ceil(2);
    [low, mid, total - low - mid]
}

/// Apply a [`Gate::QftBlock`](crate::gates::Gate::QftBlock) on qubits
/// `start..start + num` as tiled DIF passes, in the flag forms of the CPU
/// `apply_qft_block`: field bit reversals before or after the network as the flags
/// require, conjugate twiddles for `inverse`.
///
/// The per-stride twiddles and every pass's phase tables go up in one staged upload, so
/// a block costs one stream drain.
pub(crate) fn launch_apply_qft_block(
    ctx: &GpuContext,
    state: &mut GpuState,
    start: usize,
    num: usize,
    inverse: bool,
    swaps: bool,
    big_endian: bool,
) -> Result<()> {
    let n = state.num_qubits();
    if num == 0 {
        return Ok(());
    }
    if start + num > n {
        return Err(PrismError::InvalidQubit {
            index: start + num - 1,
            register_size: n,
        });
    }
    let reverse_input = big_endian != (inverse && !swaps);
    let reverse_output = big_endian != (swaps || inverse);
    let plan = qft_pass_plan(start, num);
    let twiddle_bits = plan.iter().map(|p| p.bits).max().unwrap_or(0);

    let sign = if inverse { -1.0 } else { 1.0 };
    let unit = |total_bits: usize| sign * std::f64::consts::TAU / (1u64 << total_bits) as f64;
    let mut phase_offsets = Vec::with_capacity(plan.len());
    let mut f64_len = 2 << twiddle_bits;
    for &pass in &plan {
        phase_offsets.push(f64_len);
        if pass.lo > 0 {
            f64_len += qft_phase_split(pass).iter().map(|b| 2 << b).sum::<usize>();
        }
    }

    let device = ctx.device();
    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = super::stage_blob(
        &mut scratch.blob,
        device,
        std::iter::empty(),
        f64_len,
        |dst| {
            let (twiddles, phases) = dst.split_at_mut(2 << twiddle_bits);
            for u in 0..twiddle_bits {
                let s = 1usize << u;
                for k in 0..s {
                    let w = Complex64::from_polar(1.0, k as f64 * unit(u + 1));
                    twiddles[2 * (s + k)] = w.re;
                    twiddles[2 * (s + k) + 1] = w.im;
                }
            }
            let mut at = 0;
            for &pass in plan.iter().filter(|p| p.lo > 0) {
                let [low_bits, mid_bits, high_bits] = qft_phase_split(pass);
                let step = unit(pass.lo + pass.bits);
                let scale = std::f64::consts::FRAC_1_SQRT_2.powi(pass.bits as i32);
                for (bits, shift, magnitude) in [
                    (low_bits, 0, scale),
                    (mid_bits, low_bits, 1.0),
                    (high_bits, low_bits + mid_bits, 1.0),
                ] {
                    for x in 0..1usize << bits {
                        let w = Complex64::from_polar(magnitude, (x << shift) as f64 * step);
                        phases[at] = w.re;
                        phases[at + 1] = w.im;
                        at += 2;
                    }
                }
            }
        },
    )?;

    if reverse_input {
        launch_qft_bit_reverse(ctx, state, start, num)?;
    }
    let (stream, func) = stream_and_fn(ctx, "qft_dif_pass")?;
    let dim: u64 = 1u64 << n;
    let start_i = start as i32;
    for (&pass, &offset) in plan.iter().zip(&phase_offsets) {
        let abs_lo = start + pass.lo;
        let row_log = abs_lo.min(QFT_TILE_Q - pass.bits);
        let groups = super::require_u32(
            "qft_dif_pass",
            "groups",
            (dim >> (pass.bits + row_log)) as usize,
        )?;
        let cfg = linear_cfg(QFT_BLOCK_SIZE, groups);
        let [low_bits, mid_bits, _] = qft_phase_split(pass);
        let phases: u64 = if pass.lo > 0 {
            staged.f64s + (offset * size_of::<f64>()) as u64
        } else {
            0
        };
        let abs_lo_i = abs_lo as i32;
        let bits_i = pass.bits as i32;
        let row_log_i = row_log as i32;
        let low_bits_i = low_bits as i32;
        let mid_bits_i = mid_bits as i32;
        let scale = std::f64::consts::FRAC_1_SQRT_2.powi(pass.bits as i32);
        let mut builder = stream.launch_builder(&func);
        let buffer = state.buffer_mut().raw_mut();
        builder
            .arg(buffer)
            .arg(&abs_lo_i)
            .arg(&bits_i)
            .arg(&row_log_i)
            .arg(&start_i)
            .arg(&staged.f64s)
            .arg(&phases)
            .arg(&low_bits_i)
            .arg(&mid_bits_i)
            .arg(&scale);
        // SAFETY: signature matches; the groups tile the state, each block's
        // `2^(bits + row_log)` amplitudes fit its `2^QFT_TILE_Q` shared tile, and every
        // twiddle and phase index stays inside the staged tables by construction of
        // their lengths. The scratch guard holds the upload.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("qft_dif_pass", e))?;
        }
        device.count_launch(1);
    }
    if reverse_output {
        launch_qft_bit_reverse(ctx, state, start, num)?;
    }
    Ok(())
}

/// Reverse the `num` index bits at `start`, tiled through shared memory when the field
/// starts at bit 0 and spans the 32 x 32 tile.
fn launch_qft_bit_reverse(
    ctx: &GpuContext,
    state: &mut GpuState,
    start: usize,
    num: usize,
) -> Result<()> {
    if num < 2 {
        return Ok(());
    }
    let dim: u64 = 1u64 << state.num_qubits();
    let num_i = num as i32;
    if start == 0 && num >= 10 {
        let (stream, func) = stream_and_fn(ctx, "qft_bit_reverse_tiled")?;
        let blocks = super::require_u32("qft_bit_reverse_tiled", "blocks", (dim >> 10) as usize)?;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&func);
        let buffer = state.buffer_mut().raw_mut();
        builder.arg(buffer).arg(&num_i);
        // SAFETY: signature matches; one block per 1024-amplitude tile index, and only
        // the block of the smaller middle of each reversed pair writes, so no two
        // blocks touch one amplitude.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("qft_bit_reverse_tiled", e))?;
        }
    } else {
        let (stream, func) = stream_and_fn(ctx, "qft_bit_reverse")?;
        let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim));
        let start_i = start as i32;
        let mut builder = stream.launch_builder(&func);
        let buffer = state.buffer_mut().raw_mut();
        builder.arg(buffer).arg(&dim).arg(&start_i).arg(&num_i);
        // SAFETY: signature matches; the grid covers dim and only the lower index of
        // each reversed pair swaps.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("qft_bit_reverse", e))?;
        }
    }
    ctx.device().count_launch(1);
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

/// Qubits one `apply_mixed_tile` block holds in shared memory: `2^11` amplitudes, 32 KB.
const MIXED_TILE_Q: usize = 11;
const MIXED_BLOCK_LOG: usize = 8;
/// Ops and parameter doubles one `apply_mixed_tile` launch carries; with the qubit list
/// they fill Pascal's 4 KB of kernel parameters.
const MIXED_MAX_OPS: usize = 128;
const MIXED_MAX_COEFS: usize = 440;

/// Tiles of `apply_mixed_tile`: the `QFT_MIN_ROW_LOG` lowest qubits in every tile, so a
/// gathered row fills whole sectors as in `qft_dif_pass`, and up to nine others.
pub(crate) const MIXED_TILE_GEOMETRY: Geometry = Geometry {
    anchor: (1 << QFT_MIN_ROW_LOG) - 1,
    budget: (MIXED_TILE_Q - QFT_MIN_ROW_LOG) as u32,
    max_ops: MIXED_MAX_OPS,
    max_coefs: MIXED_MAX_COEFS,
};

/// One tile's ops as a by-value kernel parameter; mirrors `struct MixedOps`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MixedOpsArg {
    c: [f64; MIXED_MAX_COEFS],
    op: [u32; MIXED_MAX_OPS],
    q: [u8; 16],
    num_ops: i32,
}

// SAFETY: plain `repr(C)` arrays and an i32 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for MixedOpsArg {}

const MIX_MAT1: u32 = 0;
const MIX_DIAG1: u32 = 1;
const MIX_ANTI1: u32 = 2;
const MIX_CX: u32 = 3;
const MIX_SWAP: u32 = 4;
const MIX_DIAG2: u32 = 5;
const MIX_FORM_REAL: u32 = 1;
const MIX_FORM_RX: u32 = 2;

/// Apply `ops` in order as one `apply_mixed_tile` pass over the tile on `qubits`, which
/// the lowest qubits outside it fill up to `MIXED_TILE_Q`.
///
/// The ops come from [`plan`](crate::gpu::tiles::plan) under [`MIXED_TILE_GEOMETRY`],
/// so each touches only tile qubits and together they fit the launch budget.
pub(crate) fn launch_apply_mixed_tile<'a: 'b, 'b>(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubits: u64,
    ops: impl Iterator<Item = &'b TileOp<'a>>,
) -> Result<()> {
    let n = state.num_qubits();
    if n < MIXED_TILE_Q {
        return Err(launch_limit_err(
            "apply_mixed_tile",
            "num_qubits",
            n,
            "tile width",
        ));
    }
    let mut tile = qubits;
    for q in 0..n {
        if tile.count_ones() as usize == MIXED_TILE_Q {
            break;
        }
        tile |= 1 << q;
    }
    let mut arg = MixedOpsArg {
        c: [0.0; MIXED_MAX_COEFS],
        op: [0; MIXED_MAX_OPS],
        q: [0; 16],
        num_ops: 0,
    };
    let mut position = [0u32; 64];
    for (slot, q) in (0..n).filter(|&q| tile >> q & 1 == 1).enumerate() {
        arg.q[slot] = q as u8;
        position[q] = slot as u32;
    }
    let one = Complex64::new(1.0, 0.0);
    let mut coefs = 0usize;
    for op in ops {
        let at = coefs;
        let mut put = |v: Complex64| {
            arg.c[coefs] = v.re;
            arg.c[coefs + 1] = v.im;
            coefs += 2;
        };
        let (kind, a, b, flag) = match *op {
            TileOp::Mat1 { q, m } => {
                m.iter().flatten().copied().for_each(&mut put);
                let form = if m.iter().flatten().all(|v| v.im == 0.0) {
                    MIX_FORM_REAL
                } else if [m[0][0], m[1][1]].iter().all(|v| v.im == 0.0)
                    && [m[0][1], m[1][0]].iter().all(|v| v.re == 0.0)
                {
                    MIX_FORM_RX
                } else {
                    0
                };
                (MIX_MAT1, position[q], form, false)
            }
            TileOp::Diag1 { q, d } => {
                d.iter().copied().for_each(&mut put);
                (MIX_DIAG1, position[q], 0, d[0] == one)
            }
            TileOp::Anti1 { q, d } => {
                d.iter().copied().for_each(&mut put);
                (MIX_ANTI1, position[q], 0, d[0] == one && d[1] == one)
            }
            TileOp::Cx { control, target } => (MIX_CX, position[control], position[target], false),
            TileOp::Swap { a, b } => (MIX_SWAP, position[a], position[b], false),
            TileOp::Diag2 { a, b, d } => {
                d.iter().copied().for_each(&mut put);
                let flag = d[..3].iter().all(|&v| v == one);
                (MIX_DIAG2, position[a], position[b], flag)
            }
            TileOp::Dense2q { .. } | TileOp::Gate { .. } => {
                unreachable!("the planner keeps {op:?} out of tiles")
            }
        };
        arg.op[arg.num_ops as usize] =
            kind | a << 3 | b << 7 | u32::from(flag) << 11 | (at as u32) << 12;
        arg.num_ops += 1;
    }
    let num_tiles = 1u32 << (n - MIXED_TILE_Q);
    let (stream, func) = stream_and_fn(ctx, "apply_mixed_tile")?;
    let cfg = linear_cfg(1 << MIXED_BLOCK_LOG, num_tiles);
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&arg);
    // SAFETY: signature matches; num_tiles blocks of the 2^MIXED_TILE_Q amplitudes over
    // MIXED_TILE_Q distinct qubits below n cover the state once, every op position names
    // a tile qubit, and the op list rides in parameter space with num_ops entries filled.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("apply_mixed_tile", e))?;
    }
    ctx.device().count_launch(1);
    ctx.device().count_tile();
    Ok(())
}

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
        check_qubit(n, target)?;
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
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
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
                .map_err(|e| driver_err("apply_multi_fused_tiled", e))?;
        }
        ctx.device().count_launch(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{QftPass, Route2q, qft_pass_plan, route_2q};
    use crate::gates::Gate;
    use num_complex::Complex64;

    #[test]
    fn route_2q_reads_exact_structure_only() {
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let phase = Complex64::from_polar(1.0, 0.4);
        assert_eq!(
            route_2q(&Gate::Cx.matrix_4x4()),
            Route2q::Cx { control_high: true }
        );
        assert_eq!(
            route_2q(&[[o, z, z, z], [z, z, z, o], [z, z, o, z], [z, o, z, z]]),
            Route2q::Cx {
                control_high: false
            }
        );
        assert_eq!(route_2q(&Gate::Swap.matrix_4x4()), Route2q::Swap);
        assert_eq!(route_2q(&Gate::Cz.matrix_4x4()), Route2q::Cz);
        assert_eq!(
            route_2q(&[[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, phase]]),
            Route2q::ControlledPhase(phase)
        );
        assert_eq!(
            route_2q(&[[z, o, z, z], [phase, z, z, z], [z, z, o, z], [z, z, z, o]]),
            Route2q::Monomial {
                src: [1, 0, 2, 3],
                scale: [o, phase, o, o]
            }
        );
        let mut near = Gate::Cx.matrix_4x4();
        near[0][1] = Complex64::new(1e-300, 0.0);
        assert_eq!(route_2q(&near), Route2q::Dense);
        let mut singular = Gate::Swap.matrix_4x4();
        singular[3][3] = z;
        assert_eq!(route_2q(&singular), Route2q::Dense);
        let mut repeated = Gate::Swap.matrix_4x4();
        repeated[0] = [z, o, z, z];
        assert_eq!(route_2q(&repeated), Route2q::Dense);
    }

    #[test]
    fn qft_pass_plan_splits_high_stages_evenly_above_the_tile() {
        let pass = |lo, bits| QftPass { lo, bits };
        assert_eq!(qft_pass_plan(0, 10), [pass(0, 10)]);
        assert_eq!(qft_pass_plan(0, 20), [pass(11, 9), pass(0, 11)]);
        assert_eq!(
            qft_pass_plan(0, 28),
            [pass(19, 9), pass(11, 8), pass(0, 11)]
        );
        assert_eq!(qft_pass_plan(1, 12), [pass(10, 2), pass(0, 10)]);
        assert_eq!(qft_pass_plan(5, 17), [pass(9, 8), pass(0, 9)]);
    }
}
