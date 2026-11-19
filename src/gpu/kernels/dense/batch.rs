//! Batched diagonal launchers over per-group phase tables.

use cudarc::driver::PushKernelArg;
use num_complex::Complex64;

use crate::backend::statevector::kernels as cpu_k;
use crate::error::{PrismError, Result};
use crate::gpu::{GpuContext, GpuState};

use super::super::{check_qubit, grid_for, launch_counted, linear_cfg, stage_blob, stream_and_fn};
use super::gates::{launch_apply_cu_phase, launch_apply_diagonal_1q, launch_apply_parity_phase};
use super::{
    BATCH_PHASE_META_LEN, BATCH_RZZ_META_LEN, BLOCK_SIZE, DIAG_BATCH_META_LEN, DiagListArg,
    MAX_PARAM_QUBITS,
};

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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));
    let num_gates_i = num_gates as i32;
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&dim).arg(&arg).arg(&num_gates_i);
    // SAFETY: signature matches; the gate list rides in parameter space with num_gates
    // entries filled; grid covers dim.
    unsafe { launch_counted(builder, cfg, "apply_multi_fused_diagonal", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(half_count, BLOCK_SIZE));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = stage_blob(
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
    unsafe { launch_counted(builder, cfg, "apply_batch_phase", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = stage_blob(
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
    unsafe { launch_counted(builder, cfg, "apply_batch_rzz", ctx.device(), 1) }
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
    let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));

    let mut scratch = ctx.launcher_scratch();
    let scratch = &mut *scratch;
    let staged = stage_blob(
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
    unsafe { launch_counted(builder, cfg, "apply_diagonal_batch", ctx.device(), 1) }
}
