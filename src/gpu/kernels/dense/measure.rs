//! Measurement launchers: probabilities, reductions, collapse, and the device measure log.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::Result;
use crate::gpu::{GpuBuffer, GpuContext, GpuState};

use super::super::{
    check_pair, check_qubit, driver_err, ensure_capacity, grid_for, launch_counted, linear_cfg,
    require_i32, stream_and_fn,
};
use super::BLOCK_SIZE;

/// Per-basis-state probabilities with `pending_norm²` applied. With `parallel`, a readback
/// of `STAGED_READBACK_MIN_BYTES` or more squares one chunk at a time straight into the
/// pinned staging slots; anything smaller squares into the state's `probs_scratch`.
pub(crate) fn launch_compute_probabilities(ctx: &GpuContext, state: &GpuState) -> Result<Vec<f64>> {
    #[cfg(feature = "parallel")]
    if (std::mem::size_of::<f64>() << state.num_qubits())
        >= crate::gpu::memory::STAGED_READBACK_MIN_BYTES
    {
        return probabilities_through_staging(ctx, state);
    }
    probabilities_through_scratch(ctx, state)
}

#[cfg(feature = "parallel")]
fn probabilities_through_staging(ctx: &GpuContext, state: &GpuState) -> Result<Vec<f64>> {
    let dim = 1usize << state.num_qubits();
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "compute_probabilities")?;
    let norm_sq = state.pending_norm() * state.pending_norm();
    let state_buf = state.buffer().raw();
    let mut host = vec![0.0_f64; dim];
    // SAFETY: `host` holds `dim` initialized f64s, plain data for which every byte pattern
    // is a valid value, and the exclusive borrow covers the byte view for its lifetime.
    let dst = unsafe {
        std::slice::from_raw_parts_mut(host.as_mut_ptr().cast::<u8>(), dim * size_of::<f64>())
    };
    device.with_staging(|staging| {
        staging.stage(dst, |range, slot| {
            let start = (range.start / size_of::<f64>()) as u64;
            let count = (range.len() / size_of::<f64>()) as u64;
            let mut builder = stream.launch_builder(&func);
            builder
                .arg(state_buf)
                .arg(&start)
                .arg(&count)
                .arg(&norm_sq)
                .arg(&slot.device);
            let cfg = linear_cfg(BLOCK_SIZE, grid_for(count, BLOCK_SIZE));
            // SAFETY: signature matches; the grid covers `count` threads, each reading
            // amplitude `start + i < dim` and writing f64 `i` of the slot, which holds
            // `range.len() = 8 * count` bytes mapped into the device.
            unsafe {
                launch_counted(
                    builder,
                    cfg,
                    "compute_probabilities",
                    device,
                    u64::from(start == 0),
                )
            }
        })
    })?;
    device.count_drain();
    Ok(host)
}

pub(crate) fn probabilities_through_scratch(
    ctx: &GpuContext,
    state: &GpuState,
) -> Result<Vec<f64>> {
    let dim: u64 = 1u64 << state.num_qubits();
    let device = ctx.device();
    let (stream, func) = stream_and_fn(ctx, "compute_probabilities")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));

    let mut scratch_slot = state.probs_scratch();
    if scratch_slot.as_ref().is_none_or(|b| b.len() < dim as usize) {
        *scratch_slot = Some(GpuBuffer::<f64>::alloc_zeros(device, dim as usize)?);
    }
    let scratch = scratch_slot.as_mut().unwrap();

    let norm_sq = state.pending_norm() * state.pending_norm();
    let start = 0u64;
    let mut builder = stream.launch_builder(&func);
    let state_buf = state.buffer().raw();
    let out = scratch.raw_mut();
    builder
        .arg(state_buf)
        .arg(&start)
        .arg(&dim)
        .arg(&norm_sq)
        .arg(out);
    // SAFETY: signature matches; grid covers dim; scratch is at least dim elements.
    unsafe { launch_counted(builder, cfg, "compute_probabilities", ctx.device(), 1) }?;
    let mut host = vec![0.0_f64; dim as usize];
    scratch.copy_to_host(device, &mut host)?;
    Ok(host)
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
    let partials = ensure_capacity(&mut scratch.measure_partials, device, num_blocks as usize)?;

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
        unsafe { launch_counted(builder, stage1_cfg, "measure_prob_one", device, 1) }?;
    }

    let result = ensure_capacity(&mut scratch.measure_result, device, 1)?;
    let count_u32 = num_blocks;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&count_u32)
            .arg(result.raw_mut());
        // SAFETY: signature matches kernel; one block of BLOCK_SIZE threads, grid-stride
        // loop over `count_u32` partials. Both buffers held by the scratch guard.
        unsafe { launch_counted(builder, stage2_cfg, "measure_prob_one_finalize", device, 0) }?;
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
    let partials = ensure_capacity(
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
        unsafe { launch_counted(builder, stage1_cfg, "rdm_qubit", device, 1) }?;
    }

    let result = ensure_capacity(&mut scratch.rdm_result, device, 4)?;
    let count_u32 = num_blocks;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&count_u32)
            .arg(result.raw_mut());
        // SAFETY: signature matches kernel; one block of BLOCK_SIZE threads, grid-stride
        // loop over `count_u32` four-column partials. Both buffers held by the scratch guard.
        unsafe { launch_counted(builder, stage2_cfg, "rdm_qubit_finalize", device, 0) }?;
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

/// Two-qubit reduced density matrix over `(q0, q1)` in basis `t = 2*bit(q0) + bit(q1)`
/// with `pending_norm²` applied. Two reduction launches and a 128-byte readback replace
/// a full-state export.
pub(crate) fn reduced_density_matrix_2q(
    ctx: &GpuContext,
    state: &GpuState,
    q0: usize,
    q1: usize,
) -> Result<[[Complex64; 4]; 4]> {
    const MAX_RDM2Q_BLOCKS: u64 = 4096;
    let n = state.num_qubits();
    check_pair(n, q0, q1)?;
    let groups: u64 = 1u64 << (n - 2);
    let num_blocks = groups
        .div_ceil(2 * BLOCK_SIZE as u64)
        .clamp(1, MAX_RDM2Q_BLOCKS) as u32;

    let device = ctx.device();
    let stream = device.stream()?;
    let stage1 = device.function("rdm_2q")?;
    let stage2 = device.function("rdm_2q_finalize")?;
    let stage1_cfg = linear_cfg(BLOCK_SIZE, num_blocks);
    let stage2_cfg = linear_cfg(BLOCK_SIZE, 1);

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partials = ensure_capacity(
        &mut scratch.measure_partials,
        device,
        16 * num_blocks as usize,
    )?;

    let q0_i = q0 as i32;
    let q1_i = q1 as i32;
    {
        let mut builder = stream.launch_builder(&stage1);
        let state_buf = state.buffer().raw();
        builder
            .arg(state_buf)
            .arg(&groups)
            .arg(&q0_i)
            .arg(&q1_i)
            .arg(partials.raw_mut());
        // SAFETY: signature matches `rdm_2q`; `q0` and `q1` are distinct and below `n`,
        // threads grid-stride across `groups`, and `out_partials` holds `16 * num_blocks` f64s.
        unsafe { launch_counted(builder, stage1_cfg, "rdm_2q", device, 1) }?;
    }

    let result = ensure_capacity(&mut scratch.rdm2q_result, device, 16)?;
    let count_u32 = num_blocks;
    {
        let mut builder = stream.launch_builder(&stage2);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&count_u32)
            .arg(result.raw_mut());
        // SAFETY: signature matches `rdm_2q_finalize`; one block strides across `count_u32`
        // sixteen-column partial blocks into a 16-element output. Both buffers are held by
        // the scratch guard.
        unsafe { launch_counted(builder, stage2_cfg, "rdm_2q_finalize", device, 0) }?;
    }

    let mut host = [0.0_f64; 16];
    scratch
        .rdm2q_result
        .as_ref()
        .unwrap()
        .copy_to_host(device, &mut host[..])?;
    let norm_sq = state.pending_norm() * state.pending_norm();
    let mut rho = [[Complex64::new(0.0, 0.0); 4]; 4];
    for t in 0..4 {
        rho[t][t] = Complex64::new(host[t] * norm_sq, 0.0);
    }
    const PAIRS: [(usize, usize, usize); 6] = [
        (1, 0, 4),
        (2, 0, 6),
        (2, 1, 8),
        (3, 0, 10),
        (3, 1, 12),
        (3, 2, 14),
    ];
    for (r, c, off) in PAIRS {
        let entry = Complex64::new(host[off], host[off + 1]) * norm_sq;
        rho[r][c] = entry;
        rho[c][r] = entry.conj();
    }
    Ok(rho)
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
    check_qubit(n, qubit)?;
    let dim: u64 = 1u64 << n;
    let (stream, func) = stream_and_fn(ctx, "measure_collapse")?;
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));
    let qubit_i = qubit as i32;
    let outcome_i: i32 = if outcome { 1 } else { 0 };
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&dim).arg(&qubit_i).arg(&outcome_i);
    // SAFETY: signature matches kernel; grid covers dim.
    unsafe { launch_counted(builder, cfg, "measure_collapse", ctx.device(), 1) }
}

/// Start a run of measurements decided on the device: size the log for `capacity`
/// outcomes between readbacks. The first [`measure_logged`] after each readback uploads
/// the pending norm.
pub(crate) fn begin_measure_log(
    ctx: &GpuContext,
    state: &mut GpuState,
    capacity: usize,
) -> Result<()> {
    debug_assert!(!state.norm_in_log, "measure log begun with outcomes unread");
    if state
        .measure_log
        .as_ref()
        .is_none_or(|log| log.len() < capacity + 1)
    {
        state.measure_log = Some(GpuBuffer::<f64>::alloc_zeros(ctx.device(), capacity + 1)?);
    }
    Ok(())
}

/// Measure `qubit` with the outcome decided on the device into log slot `slot`, from the
/// uniform draw `u`, and collapse; `reset` also folds the kept half onto |0>. Requires
/// [`begin_measure_log`] with a capacity above `slot`.
pub(crate) fn measure_logged(
    ctx: &GpuContext,
    state: &mut GpuState,
    qubit: usize,
    slot: usize,
    u: f64,
    reset: bool,
) -> Result<()> {
    let n = state.num_qubits();
    check_qubit(n, qubit)?;
    let dim: u64 = 1u64 << n;
    let elems_per_block = 2u64 * BLOCK_SIZE as u64;
    let num_blocks = dim.div_ceil(elems_per_block).max(1) as u32;
    let device = ctx.device();
    let stream = device.stream()?;
    let shared_bytes = BLOCK_SIZE * std::mem::size_of::<f64>() as u32;
    let qubit_i = qubit as i32;
    let slot_i = require_i32("measure_decide", "slot", slot)?;
    let log = state
        .measure_log
        .as_mut()
        .expect("measure_logged called before begin_measure_log");
    debug_assert!(slot < log.len() - 1, "measure log slot {slot} out of range");
    if !state.norm_in_log {
        stream
            .memcpy_htod(&[state.pending_norm], &mut log.raw_mut().slice_mut(0..1))
            .map_err(|e| driver_err("upload measure log norm", e))?;
        device.count_upload();
        state.norm_in_log = true;
    }

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let partials = ensure_capacity(&mut scratch.measure_partials, device, num_blocks as usize)?;
    {
        let func = device.function("measure_prob_one")?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(state.buffer.raw())
            .arg(&dim)
            .arg(&qubit_i)
            .arg(partials.raw_mut());
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        // SAFETY: signature matches kernel; num_blocks * 2*BLOCK_SIZE covers dim.
        unsafe { launch_counted(builder, cfg, "measure_prob_one", device, 1) }?;
    }
    {
        let func = device.function("measure_decide")?;
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(scratch.measure_partials.as_ref().unwrap().raw())
            .arg(&num_blocks)
            .arg(log.raw_mut())
            .arg(&slot_i)
            .arg(&u)
            .arg(&crate::backend::NORM_CLAMP_MIN);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        // SAFETY: signature matches kernel; one block reduces `num_blocks` partials and
        // thread 0 writes log[0] and log[1 + slot], inside the log by the capacity check.
        unsafe { launch_counted(builder, cfg, "measure_decide", device, 0) }?;
    }
    let (name, count) = if reset {
        ("reset_fold_logged", dim / 2)
    } else {
        ("measure_collapse_logged", dim)
    };
    let func = device.function(name)?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer.raw_mut())
        .arg(&count)
        .arg(&qubit_i)
        .arg(log.raw())
        .arg(&slot_i);
    // SAFETY: signature matches kernel; the grid covers `count` amplitudes (or pairs),
    // and each thread writes only its own amplitude (or its own pair).
    unsafe {
        launch_counted(
            builder,
            linear_cfg(BLOCK_SIZE, grid_for(count, BLOCK_SIZE)),
            name,
            device,
            1,
        )
    }
}

/// Read back the first `slots` outcomes of the measure log as booleans and take its norm
/// as the pending norm: the one readback for a run of device measurements.
pub(crate) fn read_measure_log(
    ctx: &GpuContext,
    state: &mut GpuState,
    slots: usize,
) -> Result<Vec<bool>> {
    let device = ctx.device();
    let log = state
        .measure_log
        .as_ref()
        .expect("read_measure_log called before begin_measure_log");
    let mut host = vec![0.0_f64; slots + 1];
    device
        .stream()?
        .memcpy_dtoh(&log.raw().slice(0..slots + 1), &mut host)
        .map_err(|e| driver_err("measure log dtoh", e))?;
    device.count_drain();
    state.pending_norm = host[0];
    state.norm_in_log = false;
    Ok(host[1..].iter().map(|&v| v != 0.0).collect())
}
