//! Shared-memory tiled launchers: the QFT block, mixed tiles, and tiled `MultiFused`.

use cudarc::driver::{LaunchConfig, PushKernelArg};
use num_complex::Complex64;

use crate::error::{PrismError, Result};
use crate::gpu::tiles::{Geometry, TileOp};
use crate::gpu::{GpuContext, GpuState};

use super::super::{
    check_qubit, grid_for, launch_counted, launch_limit_err, linear_cfg, require_u32, stage_blob,
    stream_and_fn,
};
use super::gates::launch_apply_gate_1q;
use super::{BLOCK_SIZE, TileGatesArg};

/// Most DIF stages one `qft_dif_pass` block runs: its shared-memory tile holds
/// `2^QFT_TILE_Q` amplitudes, 32 KB.
pub(super) const QFT_TILE_Q: usize = 11;

/// Log2 of the fewest consecutive amplitudes a gathered row carries above the field's
/// lowest pass: 64 bytes, so a strided row read fills whole sectors.
const QFT_MIN_ROW_LOG: usize = 2;

const QFT_BLOCK_SIZE: u32 = 256;

/// One `qft_dif_pass`: DIF stages `lo..lo + bits` of the field, in field bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct QftPass {
    pub(super) lo: usize,
    pub(super) bits: usize,
}

/// DIF passes for a `num`-qubit field at `start`, top stage first: the stages above the
/// lowest pass split evenly into passes of at most `QFT_TILE_Q - QFT_MIN_ROW_LOG`.
pub(super) fn qft_pass_plan(start: usize, num: usize) -> Vec<QftPass> {
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
    let staged = stage_blob(
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
        let groups = require_u32(
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
        unsafe { launch_counted(builder, cfg, "qft_dif_pass", device, 1) }?;
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
        let blocks = require_u32("qft_bit_reverse_tiled", "blocks", (dim >> 10) as usize)?;
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
        unsafe { launch_counted(builder, cfg, "qft_bit_reverse_tiled", ctx.device(), 1) }
    } else {
        let (stream, func) = stream_and_fn(ctx, "qft_bit_reverse")?;
        let cfg = linear_cfg(BLOCK_SIZE, grid_for(dim, BLOCK_SIZE));
        let start_i = start as i32;
        let mut builder = stream.launch_builder(&func);
        let buffer = state.buffer_mut().raw_mut();
        builder.arg(buffer).arg(&dim).arg(&start_i).arg(&num_i);
        // SAFETY: signature matches; the grid covers dim and only the lower index of
        // each reversed pair swaps.
        unsafe { launch_counted(builder, cfg, "qft_bit_reverse", ctx.device(), 1) }
    }
}

/// Matches the `TILE_Q` / `TILE_SIZE` in the PTX source: a tile spans `2^TILE_Q`
/// amplitudes over a chosen set of `TILE_Q` qubits.
pub(super) const MULTI_FUSED_TILE_Q: usize = 10;
pub(super) const MULTI_FUSED_TILE_SIZE: u64 = 1 << MULTI_FUSED_TILE_Q;
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
pub(super) const MIXED_TILE_Q: usize = 11;
pub(super) const MIXED_BLOCK_LOG: usize = 8;
/// Ops and parameter doubles one `apply_mixed_tile` launch carries; with the qubit list
/// they fill Pascal's 4 KB of kernel parameters.
pub(super) const MIXED_MAX_OPS: usize = 128;
pub(super) const MIXED_MAX_COEFS: usize = 440;

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

// State pointer + MixedOpsArg + diag_tables pointer must stay inside Pascal's 4 KB limit.
const _: () = assert!(size_of::<MixedOpsArg>() + 2 * size_of::<u64>() <= 4096);

const MIX_MAT1: u32 = 0;
const MIX_DIAG1: u32 = 1;
const MIX_ANTI1: u32 = 2;
const MIX_CX: u32 = 3;
const MIX_SWAP: u32 = 4;
const MIX_DIAG2: u32 = 5;
const MIX_DENSE2Q: u32 = 6;
const MIX_DIAG_TABLE: u32 = 7;
const MIX_FORM_REAL: u32 = 1;
const MIX_FORM_RX: u32 = 2;
/// Shortest consecutive diagonal run folded into a phase table for one tile.
const MIX_DIAG_TABLE_MIN_RUN: usize = 3;

/// Fold a run of `Diag1` and `Diag2` ops in one tile into a `2^MIXED_TILE_Q` phase table.
fn fold_mixed_diag_table(
    ops: &[&TileOp<'_>],
    position: &[u32; 64],
    one: Complex64,
    out: &mut [f64],
) {
    const TILE_LEN: usize = 1 << MIXED_TILE_Q;
    let mut tbl = [one; TILE_LEN];
    for &&op in ops {
        match op {
            TileOp::Diag1 { q, d } => {
                let a = position[q] as usize;
                let bit = 1usize << a;
                if d[0] == one {
                    let phase = d[1];
                    for (e, slot) in tbl.iter_mut().enumerate() {
                        if e & bit != 0 {
                            *slot *= phase;
                        }
                    }
                } else {
                    for (e, slot) in tbl.iter_mut().enumerate() {
                        *slot *= d[(e >> a) & 1];
                    }
                }
            }
            TileOp::Diag2 { a, b, d } => {
                let pa = position[a] as usize;
                let pb = position[b] as usize;
                if d[..3].iter().all(|&v| v == one) {
                    let mask = (1usize << pa) | (1usize << pb);
                    let phase = d[3];
                    for (e, slot) in tbl.iter_mut().enumerate() {
                        if e & mask == mask {
                            *slot *= phase;
                        }
                    }
                } else {
                    for (e, slot) in tbl.iter_mut().enumerate() {
                        let s = (((e >> pa) & 1) << 1) | ((e >> pb) & 1);
                        *slot *= d[s];
                    }
                }
            }
            _ => unreachable!("fold_mixed_diag_table called on non-diagonal op"),
        }
    }
    for (pair, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(tbl) {
        pair[0] = v.re;
        pair[1] = v.im;
    }
}

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
    let ops_vec: Vec<&TileOp<'a>> = ops.collect();
    let mut diag_runs: Vec<std::ops::Range<usize>> = Vec::new();
    let mut coefs = 0usize;
    let mut idx = 0usize;
    while idx < ops_vec.len() {
        if matches!(ops_vec[idx], TileOp::Diag1 { .. } | TileOp::Diag2 { .. }) {
            let mut end = idx + 1;
            while end < ops_vec.len()
                && matches!(ops_vec[end], TileOp::Diag1 { .. } | TileOp::Diag2 { .. })
            {
                end += 1;
            }
            if end - idx >= MIX_DIAG_TABLE_MIN_RUN {
                let table_idx = diag_runs.len() as u32;
                diag_runs.push(idx..end);
                arg.op[arg.num_ops as usize] = MIX_DIAG_TABLE | (table_idx << 12);
                arg.num_ops += 1;
                idx = end;
                continue;
            }
        }
        let op = ops_vec[idx];
        idx += 1;
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
            TileOp::Dense2q { hi, lo, m } => {
                for row in m {
                    for v in row {
                        arg.c[coefs] = v.re;
                        arg.c[coefs + 1] = v.im - v.re;
                        arg.c[coefs + 2] = v.re + v.im;
                        coefs += 3;
                    }
                }
                (MIX_DENSE2Q, position[hi], position[lo], false)
            }
            TileOp::Gate { .. } => {
                unreachable!("the planner keeps {op:?} out of tiles")
            }
        };
        arg.op[arg.num_ops as usize] =
            kind | a << 3 | b << 7 | u32::from(flag) << 11 | (at as u32) << 12;
        arg.num_ops += 1;
    }
    let mut scratch_guard = if diag_runs.is_empty() {
        None
    } else {
        Some(ctx.launcher_scratch())
    };
    let diag_tables_ptr: u64 = if let Some(scratch) = scratch_guard.as_deref_mut() {
        let f64_per_table = 2 << MIXED_TILE_Q;
        let staged = stage_blob(
            &mut scratch.blob,
            ctx.device(),
            std::iter::empty(),
            diag_runs.len() * f64_per_table,
            |dst| {
                for (i, range) in diag_runs.iter().enumerate() {
                    let start = i * f64_per_table;
                    fold_mixed_diag_table(
                        &ops_vec[range.clone()],
                        &position,
                        one,
                        &mut dst[start..start + f64_per_table],
                    );
                }
            },
        )?;
        staged.f64s
    } else {
        0
    };
    let num_tiles = 1u32 << (n - MIXED_TILE_Q);
    let (stream, func) = stream_and_fn(ctx, "apply_mixed_tile")?;
    let cfg = linear_cfg(1 << MIXED_BLOCK_LOG, num_tiles);
    let mut builder = stream.launch_builder(&func);
    let buffer = state.buffer_mut().raw_mut();
    builder.arg(buffer).arg(&arg).arg(&diag_tables_ptr);
    // SAFETY: signature matches; num_tiles blocks of the 2^MIXED_TILE_Q amplitudes over
    // MIXED_TILE_Q distinct qubits below n cover the state once, every op position names
    // a tile qubit, the op list rides in parameter space with num_ops entries filled, and
    // diag_tables_ptr holds every folded run's 2^MIXED_TILE_Q phases while scratch_guard
    // is live.
    unsafe { launch_counted(builder, cfg, "apply_mixed_tile", ctx.device(), 1) }?;
    drop(scratch_guard);
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
        unsafe { launch_counted(builder, cfg, "apply_multi_fused_tiled", ctx.device(), 1) }?;
    }
    Ok(())
}
