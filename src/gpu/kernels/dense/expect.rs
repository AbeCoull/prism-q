//! Pauli and operator expectation reductions over the device state.

use cudarc::driver::{CudaViewMut, LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::Result;
use crate::gpu::{GpuBuffer, GpuContext, GpuState};

use super::super::{
    ensure_capacity, ensure_exact, ensure_scratch, finalize_pauli_sums, grid_y_tiles,
    launch_counted, linear_cfg, require_u32,
};
use super::BLOCK_SIZE;
use super::tiles::launch_apply_multi_fused_nondiag;

/// Grid block count per mask for `sv_pauli_expect`, capped by `PAULI_MAX_BLOCKS`
/// and the `PAULI_MAX_PARTIAL_BYTES` scratch budget.
pub(super) fn pauli_blocks_per_mask(dim: u64, num_masks: usize) -> u32 {
    let pairs = (dim >> 1).max(1);
    let elems_per_block = (PAULI_AMPS_PER_THREAD * BLOCK_SIZE as usize) as u64;
    let bytes_per_block = 2 * size_of::<f64>() * num_masks.max(1);
    let budget_blocks = (PAULI_MAX_PARTIAL_BYTES / bytes_per_block).max(1) as u64;
    pairs
        .div_ceil(elems_per_block)
        .min(PAULI_MAX_BLOCKS)
        .min(budget_blocks)
        .max(1) as u32
}

/// `sum_j conj(psi[j ^ xmask]) psi[j] (-1)^{popcount(j & zmask)}` per
/// `(xmask, zmask)` pair with `pending_norm²` applied: the complex accumulator
/// behind `<P>` before the `i^{num_y}` factor and the norm. Two launches and a
/// `16 * masks.len()` byte readback replace a full-state export. Masks sharing an
/// `xmask` run in groups of up to [`PAULI_GROUP_MASKS`], one grid-stride pass
/// per group with bounded `measure_partials` scratch.
pub(crate) fn pauli_sums(
    ctx: &GpuContext,
    state: &GpuState,
    masks: &[(u64, u64)],
) -> Result<Vec<Complex64>> {
    if masks.is_empty() {
        return Ok(Vec::new());
    }
    let dim: u64 = 1u64 << state.num_qubits();
    let blocks_per_mask = pauli_blocks_per_mask(dim, masks.len());
    let num_masks = require_u32("sv_pauli_expect", "masks", masks.len())?;
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
    let num_groups = require_u32("sv_pauli_expect", "groups", group_x.len())?;
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
    ensure_capacity(&mut scratch.measure_partials, device, partial_len)?;
    ensure_scratch(&mut scratch.u64_a, device, &groups)?;
    ensure_scratch(&mut scratch.u64_b, device, &members)?;
    let partials = scratch.measure_partials.as_mut().unwrap();
    for (group_base, launch_groups) in grid_y_tiles(num_groups) {
        let cfg = LaunchConfig {
            grid_dim: (blocks_per_mask, launch_groups, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = stream.launch_builder(&stage1);
        builder
            .arg(state.buffer().raw())
            .arg(&dim)
            .arg(scratch.u64_a.as_ref().unwrap().raw())
            .arg(&num_groups)
            .arg(&group_base)
            .arg(scratch.u64_b.as_ref().unwrap().raw())
            .arg(&num_masks)
            .arg(partials.raw_mut());
        // SAFETY: signature matches the kernel. For `x == 0`, `blocks_per_mask` blocks of
        // `BLOCK_SIZE` threads grid-stride over all `dim` amplitudes; for `x != 0`, they
        // grid-stride over `dim / 2` pairs mapped via `insert_zero_bit_u64(k, tz(x))` so
        // `i0 < dim` and `i1 = i0 ^ x < dim` because `0 < x < dim` (masks are built from
        // validated qubit indices). Each of the `launch_groups` groups from `group_base`
        // ends at or below `num_groups` and holds at most `PAULI_GROUP_MASKS` members, and
        // every member slot is a distinct mask index, so `partials` holds two f64s per
        // (slot, block) within `partial_len`. All buffers are held by the scratch guard.
        unsafe {
            launch_counted(
                builder,
                cfg,
                "sv_pauli_expect",
                device,
                u64::from(launch_groups),
            )
        }?;
    }
    let norm_sq = state.pending_norm() * state.pending_norm();
    finalize_pauli_sums(
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
    let device = ctx.device();
    let dim: u64 = 1u64 << n;
    let grid = dim
        .div_ceil(64)
        .div_ceil(u64::from(BLOCK_SIZE / 32))
        .min(MOMENT_MAX_BLOCKS) as usize;
    let stride = 3 * grid;
    let mut fresh = true;
    for (g, (zmasks, coefficients, x_bits, y_bits)) in groups.iter().enumerate() {
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
        let mut scratch = ctx.launcher_scratch();
        let scratch = &mut *scratch;
        let result = ensure_exact(&mut scratch.moment_partials, device, stride * groups.len())?;
        let mut slice = result.raw_mut().slice_mut(g * stride..(g + 1) * stride);
        launch_group_moments(
            ctx,
            source,
            zmasks,
            coefficients,
            &mut scratch.u64_a,
            &mut scratch.f64_a,
            &mut slice,
            grid as u32,
        )?;
    }
    let mut partials = vec![0.0_f64; stride * groups.len()];
    ctx.launcher_scratch()
        .moment_partials
        .as_ref()
        .expect("moment partials sized above")
        .copy_to_host(device, &mut partials)?;
    Ok(Some(
        partials
            .chunks_exact(stride)
            .map(|group| {
                let (s0, s1, s2) = group
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .fold((0.0, 0.0, 0.0), |acc, p| {
                        (acc.0 + p[0], acc.1 + p[1], acc.2 + p[2])
                    });
                if s0 == 0.0 {
                    (0.0, 0.0)
                } else {
                    (s1 / s0, s2 / s0)
                }
            })
            .collect(),
    ))
}

/// Queue `sv_group_moments` for one group operator over `state` into `result`, three
/// partials per block; the kernel also sums `p`, so the host ratio cancels the pending
/// norm and any rotation growth. Every group of a call reads back in one copy.
#[allow(clippy::too_many_arguments)]
fn launch_group_moments(
    ctx: &GpuContext,
    state: &GpuState,
    zmasks: &[u64],
    coefficients: &[f64],
    masks_slot: &mut Option<GpuBuffer<u64>>,
    coefficients_slot: &mut Option<GpuBuffer<f64>>,
    result: &mut CudaViewMut<'_, f64>,
    grid: u32,
) -> Result<()> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("sv_group_moments")?;
    let dim: u64 = 1u64 << state.num_qubits();
    let num_terms = require_u32("sv_group_moments", "terms", zmasks.len())?;
    ensure_scratch(masks_slot, device, zmasks)?;
    ensure_scratch(coefficients_slot, device, coefficients)?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer().raw())
        .arg(&dim)
        .arg(masks_slot.as_ref().unwrap().raw())
        .arg(coefficients_slot.as_ref().unwrap().raw())
        .arg(&num_terms)
        .arg(result);
    // SAFETY: signature matches the kernel. The state holds `dim` amplitudes, the mask
    // and coefficient buffers hold at least `num_terms` entries, every read index is
    // checked against `dim`, and `result` is this group's slice of three f64s per block.
    // A later group's upload into the same mask buffers lands in stream order after this
    // kernel, and a reallocated buffer is freed in stream order.
    unsafe {
        launch_counted(
            builder,
            linear_cfg(BLOCK_SIZE, grid),
            "sv_group_moments",
            device,
            1,
        )
    }
}

/// `(<H>, <H^2>)` of the operator in `table` over the device `state` from one
/// `sv_operator_moments` pass and one readback; the kernel also sums `|psi|^2`, so the
/// ratios cancel the pending norm. `(0, 0)` for a zero state.
pub(crate) fn operator_moments(
    ctx: &GpuContext,
    state: &GpuState,
    table: &crate::sim::observable::OperatorTable,
) -> Result<(f64, f64)> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("sv_operator_moments")?;
    let dim: u64 = 1u64 << state.num_qubits();
    let num_groups = require_u32("sv_operator_moments", "groups", table.xs.len())?;
    let xs: Vec<u64> = table.xs.iter().map(|&x| x as u64).collect();
    let offsets = table
        .offsets
        .iter()
        .map(|&o| require_u32("sv_operator_moments", "offset", o))
        .collect::<Result<Vec<u32>>>()?;
    let zs: Vec<u64> = table.zs.iter().map(|&z| z as u64).collect();
    let coefficients: Vec<f64> = table
        .coefficients
        .iter()
        .flat_map(|c| [c.re, c.im])
        .collect();
    let xs_dev = GpuBuffer::from_host(device, &xs)?;
    let offsets_dev = GpuBuffer::from_host(device, &offsets)?;
    let zs_dev = GpuBuffer::from_host(device, &zs)?;
    let coefficients_dev = GpuBuffer::from_host(device, &coefficients)?;
    let grid = dim.div_ceil(u64::from(BLOCK_SIZE)).min(MOMENT_MAX_BLOCKS) as u32;
    let mut partials_dev = GpuBuffer::<f64>::alloc_zeros(device, 3 * grid as usize)?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(state.buffer().raw())
        .arg(&dim)
        .arg(xs_dev.raw())
        .arg(offsets_dev.raw())
        .arg(&num_groups)
        .arg(zs_dev.raw())
        .arg(coefficients_dev.raw())
        .arg(partials_dev.raw_mut());
    // SAFETY: signature matches the kernel. The state holds `dim` amplitudes and every
    // read index `k ^ x` stays below `dim` because each X mask addresses only the
    // state's qubits; `offsets` holds `num_groups + 1` entries bounding `zs` and the
    // interleaved `coefficients`, and `partials` holds three f64s per block.
    unsafe {
        launch_counted(
            builder,
            linear_cfg(BLOCK_SIZE, grid),
            "sv_operator_moments",
            device,
            1,
        )
    }?;
    let mut partials = vec![0.0_f64; 3 * grid as usize];
    partials_dev.copy_to_host(device, &mut partials)?;
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

/// Grid cap per mask for `sv_pauli_expect`; threads grid-stride over remaining pairs.
pub(super) const PAULI_MAX_BLOCKS: u64 = 1024;

/// Byte ceiling for `LauncherScratch::measure_partials` in `pauli_sums` (16 MiB).
pub(super) const PAULI_MAX_PARTIAL_BYTES: usize = 16 * 1024 * 1024;

/// Masks one `sv_pauli_expect` block reduces per state pass.
pub(super) const PAULI_GROUP_MASKS: usize = 8;

/// Target pairs per `sv_pauli_expect` thread before capping at `PAULI_MAX_BLOCKS`.
pub(super) const PAULI_AMPS_PER_THREAD: usize = 1;
