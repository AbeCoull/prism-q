//! Gate launchers: one and two qubit gates, controlled gates, and the 4x4 route.

use cudarc::driver::PushKernelArg;
use num_complex::Complex64;

use crate::error::{PrismError, Result};
use crate::gpu::{GpuContext, GpuState};

use super::super::{
    check_pair, check_qubit, grid_for, launch_counted, linear_cfg, stage_complex, stream_and_fn,
};
use super::{BLOCK_SIZE, FusedMatArg, UnitaryMetaArg, qubit_list};

/// Write amplitude 0 = 1; assumes the buffer is already zeroed.
pub(crate) fn launch_set_initial_state(ctx: &GpuContext, state: &mut GpuState) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "set_initial_state")?;
    let cfg = linear_cfg(1, 1);
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer);
    // SAFETY: kernel signature is (double2*); single-thread write within allocated range.
    unsafe { launch_counted(builder, cfg, "set_initial_state", ctx.device(), 0) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_gate_1q", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_diagonal_1q", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&pair_count).arg(&q0_i).arg(&q1_i);
    // SAFETY: signature matches kernel (state, pair_count, q0, q1); grid covers iter space.
    unsafe { launch_counted(builder, cfg, kernel, device, 1) }
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
pub(super) fn launch_apply_parity_phase(
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_parity_phase", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_cu", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_cu_phase", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(iter_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_mcu", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(iter_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_mcu_phase", ctx.device(), 1) }
}

/// Apply a dense `2^k x 2^k` unitary (`1 <= k <= 4`) over `targets`, with
/// `targets[0]` as the high bit of the matrix index.
pub(crate) fn launch_apply_unitary_nq(
    ctx: &GpuContext,
    state: &mut GpuState,
    targets: &[usize],
    matrix: &[Complex64],
) -> Result<()> {
    let k = targets.len();
    if k == 0 || k > 4 {
        return Err(PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("dense {k}-qubit unitary outside the 1..=4 qubit kernel range"),
        });
    }
    let n = state.num_qubits();
    for (i, &q) in targets.iter().enumerate() {
        check_qubit(n, q)?;
        if targets[..i].contains(&q) {
            return Err(PrismError::InvalidParameter {
                message: format!("duplicate target qubit {q} in dense unitary"),
            });
        }
    }
    let dim = 1usize << k;
    if matrix.len() != dim * dim {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "dense {k}-qubit unitary expects {} entries, got {}",
                dim * dim,
                matrix.len()
            ),
        });
    }

    let mut meta = UnitaryMetaArg {
        offsets: [0; 16],
        sorted: [0; 4],
        k: k as i32,
        _pad: 0,
    };
    let mut sorted = [0usize; 4];
    sorted[..k].copy_from_slice(targets);
    sorted[..k].sort_unstable();
    for (dst, &q) in meta.sorted.iter_mut().zip(&sorted[..k]) {
        *dst = q as i32;
    }
    for (m, offset) in meta.offsets.iter_mut().enumerate().take(dim) {
        let mut mask = 0u64;
        for (i, &target) in targets.iter().enumerate() {
            mask |= (((m >> (k - 1 - i)) & 1) as u64) << target;
        }
        *offset = mask;
    }

    let iter_count: u64 = 1u64 << (n - k);
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "apply_unitary_nq")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(iter_count, BLOCK_SIZE));

    let mut scratch = ctx.launcher_scratch();
    let staged = stage_complex(&mut scratch.blob, device, matrix)?;

    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder
        .arg(buffer)
        .arg(&iter_count)
        .arg(&meta)
        .arg(&staged.f64s);
    // SAFETY: signature matches `apply_unitary_nq`; targets are distinct and in bounds,
    // `staged.f64s` holds `dim * dim` complex entries in stream order while the scratch
    // guard is live, and the grid covers `iter_count`.
    unsafe { launch_counted(builder, cfg, "apply_unitary_nq", device, 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_monomial_2q", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(pair_count, BLOCK_SIZE));
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
    unsafe { launch_counted(builder, cfg, "apply_fused_2q", ctx.device(), 1) }?;
    ctx.device().count_dense_2q();
    Ok(())
}
