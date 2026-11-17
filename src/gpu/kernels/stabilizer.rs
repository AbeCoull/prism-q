//! Stabilizer tableau kernels and launchers over the `GpuTableau` layout. Phases take one
//! byte per row rather than one bit so rowmul phase writes need no atomic update, and
//! every entry point is prefixed `stab_` to stay distinct in the shared PTX module.
//!
//! `stab_rowmul_words` XORs a source row into a destination row with the
//! Aaronson-Gottesman phase update. It runs as a single block whose threads partition
//! the `num_words` loop and reduce their per-word phase contributions through a warp
//! shuffle plus shared memory.
//!
//! Z-basis measurement runs as four small kernels: a pivot search with an `atomicMin`
//! sentinel; a cascade that rowmuls the pivot into every row carrying an X at the
//! target (one block per row, most blocks exit early); a single-block fixup that moves
//! the pivot data into the paired destabilizer and installs the measured `Z_q`; and a
//! deterministic-branch kernel that serially rowmuls, into the scratch row, the
//! stabilizers whose paired destabilizers anticommute with `Z_q`, then reads its phase.
//!
//! A run of measurements on distinct qubits keeps its state in a device buffer, so its
//! collapses chain through stream order and the host reads back only a random mask
//! between chunks of collapses.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, PushKernelArg};

use crate::error::Result;

use super::super::{GpuContext, GpuTableau};
use super::{div_ceil_grid, driver_err, linear_cfg, require_i32, require_u32, stream_and_fn};

const BLOCK_SIZE: u32 = 128;

/// Stabilizer CUDA C source, used verbatim: tableau shapes are kernel arguments, not
/// compile-time constants.
const KERNEL_SOURCE: &str = include_str!("stabilizer.cu");

pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE.to_string()
}

/// Write the identity bits (destabilizer row `i` is `X_i`, stabilizer row `n + i` is
/// `Z_i`) into a tableau that must already be zeroed.
pub(crate) fn launch_set_initial_tableau(ctx: &GpuContext, tableau: &mut GpuTableau) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "stab_set_initial_tableau")?;

    let n_usize = tableau.num_qubits();
    if n_usize == 0 {
        return Ok(());
    }
    let n = require_i32("stab_set_initial_tableau", "num_qubits", n_usize)?;
    let nw = require_i32("stab_set_initial_tableau", "num_words", tableau.num_words())?;
    let blocks = div_ceil_grid(
        "stab_set_initial_tableau",
        "num_qubits",
        n_usize,
        BLOCK_SIZE,
    )?;
    let cfg = linear_cfg(BLOCK_SIZE, blocks);

    let mut builder = stream.launch_builder(&func);
    let xz = tableau.xz_mut().raw_mut();
    builder.arg(xz).arg(&n).arg(&nw);
    // SAFETY: kernel signature is (u64*, i32, i32); xz buffer is at least
    // (2n+1) * 2 * num_words u64s and each thread writes two disjoint words.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_set_initial_tableau", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Clifford opcodes consumed by `stab_apply_word_grouped`. The values must match the
/// switch inside the kernel source.
pub(crate) mod op {
    pub const H: u32 = 0;
    pub const S: u32 = 1;
    pub const SDG: u32 = 2;
    pub const X: u32 = 3;
    pub const Y: u32 = 4;
    pub const Z: u32 = 5;
    pub const SX: u32 = 6;
    pub const SXDG: u32 = 7;
    /// `a` is control, `b` is target.
    pub const CX: u32 = 8;
    pub const CZ: u32 = 9;
    pub const SWAP: u32 = 10;
}

/// Number of u32 slots per queued op: `[opcode, a, b, pad]`.
pub(crate) const CLIFOP_STRIDE: usize = 4;

const ZERO_U32: [u32; 1] = [0];

/// Apply queued Clifford ops to the device tableau.
///
/// `ops` is a flat `u32` buffer of `[opcode, a, b, pad]` quads, opcodes from [`op`].
/// Ops are grouped by target word plus a cross-word 2q list. An op on a qubit a pending
/// cross-word op already touched forces a launch first, as the CPU
/// `flush_all_with_cross_word` does. The kernel runs one block per tableau row, so
/// traffic stays inside that row instead of striding one word across many rows; its
/// threads stripe over the word groups, read and write each word once per group, and
/// leave the cross-word tail serial on thread 0 because those ops can touch different
/// bits of one packed word.
pub(crate) fn launch_clifford_batch(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    ops: &[u32],
    scratch: &mut CliffordBatchScratch,
) -> Result<()> {
    if ops.is_empty() {
        return Ok(());
    }
    debug_assert!(
        ops.len().is_multiple_of(CLIFOP_STRIDE),
        "ClifOp buffer length must be a multiple of {CLIFOP_STRIDE}"
    );
    if tableau.num_words() == 0 {
        return Ok(());
    }
    scratch.launch_ops(ctx, tableau, ops)
}

/// Reusable host and device scratch for GPU Clifford batch launches.
#[derive(Default)]
pub(crate) struct CliffordBatchScratch {
    num_words: usize,
    /// Per-word queued ops, flat `[opcode, a, b, pad]` quads.
    per_word_ops: Vec<Vec<u32>>,
    /// Cross-word 2q ops, flat quads in insertion order.
    cross_word: Vec<u32>,
    /// Bitmask per word of qubits already touched by a cross-word op in the
    /// current launch window. A new 1q or same-word 2q gate on a qubit in
    /// this mask forces a flush to preserve ingestion order.
    cross_word_qubits: Vec<u64>,
    group_words: Vec<u32>,
    group_offsets: Vec<u32>,
    ops_flat: Vec<u32>,
    group_words_dev: Option<CudaSlice<u32>>,
    group_offsets_dev: Option<CudaSlice<u32>>,
    ops_dev: Option<CudaSlice<u32>>,
    cross_dev: Option<CudaSlice<u32>>,
}

impl CliffordBatchScratch {
    pub(crate) fn clear(&mut self) {
        let used = self.num_words.min(self.per_word_ops.len());
        for v in &mut self.per_word_ops[..used] {
            v.clear();
        }
        self.cross_word.clear();
        let cross_used = self.cross_word_qubits.len().min(self.num_words);
        self.cross_word_qubits[..cross_used].fill(0);
        self.group_words.clear();
        self.group_offsets.clear();
        self.ops_flat.clear();
    }

    fn prepare(&mut self, num_words: usize) {
        self.num_words = num_words;
        self.per_word_ops.resize_with(num_words, Vec::new);
        self.cross_word_qubits.resize(num_words, 0);
        self.clear();
    }

    fn is_empty(&self) -> bool {
        self.cross_word.is_empty()
            && self.per_word_ops[..self.num_words]
                .iter()
                .all(|v| v.is_empty())
    }

    fn would_conflict(&self, opcode: u32, a: u32, b: u32) -> bool {
        if opcode <= op::SXDG {
            let w = (a as usize) >> 6;
            let bit = 1u64 << (a & 63);
            self.cross_word_qubits[w] & bit != 0
        } else {
            let aw = (a as usize) >> 6;
            let bw = (b as usize) >> 6;
            let abit = 1u64 << (a & 63);
            let bbit = 1u64 << (b & 63);
            if aw == bw {
                let bits = abit | bbit;
                self.cross_word_qubits[aw] & bits != 0
            } else {
                self.cross_word_qubits[aw] & abit != 0 || self.cross_word_qubits[bw] & bbit != 0
            }
        }
    }

    fn push(&mut self, opcode: u32, a: u32, b: u32) {
        if opcode <= op::SXDG {
            let w = (a as usize) >> 6;
            self.per_word_ops[w].extend_from_slice(&[opcode, a, b, 0]);
        } else {
            let aw = (a as usize) >> 6;
            let bw = (b as usize) >> 6;
            if aw == bw {
                self.per_word_ops[aw].extend_from_slice(&[opcode, a, b, 0]);
            } else {
                self.cross_word.extend_from_slice(&[opcode, a, b, 0]);
                self.cross_word_qubits[aw] |= 1u64 << (a & 63);
                self.cross_word_qubits[bw] |= 1u64 << (b & 63);
            }
        }
    }

    fn launch_pending(&mut self, ctx: &GpuContext, tableau: &mut GpuTableau) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }

        self.group_words.clear();
        self.group_offsets.clear();
        self.ops_flat.clear();
        self.group_offsets.push(0);
        for (w, v) in self.per_word_ops[..self.num_words].iter().enumerate() {
            if v.is_empty() {
                continue;
            }
            debug_assert!(v.len().is_multiple_of(CLIFOP_STRIDE));
            self.group_words
                .push(require_u32("stab_apply_word_grouped", "group_word", w)?);
            self.ops_flat.extend_from_slice(v);
            let offset = self.ops_flat.len() / CLIFOP_STRIDE;
            self.group_offsets.push(require_u32(
                "stab_apply_word_grouped",
                "group_offset",
                offset,
            )?);
        }

        let num_groups = self.group_words.len();
        let num_cross_word = self.cross_word.len() / CLIFOP_STRIDE;
        launch_word_grouped_kernel(ctx, tableau, self, num_groups, num_cross_word)?;
        self.clear();
        Ok(())
    }

    fn launch_ops(
        &mut self,
        ctx: &GpuContext,
        tableau: &mut GpuTableau,
        ops: &[u32],
    ) -> Result<()> {
        self.prepare(tableau.num_words());
        for chunk in ops.as_chunks::<CLIFOP_STRIDE>().0 {
            let opcode = chunk[0];
            let a = chunk[1];
            let b = chunk[2];
            if self.would_conflict(opcode, a, b) {
                self.launch_pending(ctx, tableau)?;
            }
            self.push(opcode, a, b);
        }
        self.launch_pending(ctx, tableau)
    }
}

/// Upload `host`, or one zero word when it is empty, into the start of the grow-only
/// `slot` and return a view of the uploaded words.
fn stage_u32<'a>(
    ctx: &GpuContext,
    stream: &Arc<CudaStream>,
    slot: &'a mut Option<CudaSlice<u32>>,
    host: &[u32],
    name: &str,
) -> Result<CudaView<'a, u32>> {
    let src: &[u32] = if host.is_empty() { &ZERO_U32 } else { host };
    if slot.as_ref().is_none_or(|buf| buf.len() < src.len()) {
        *slot = Some(
            stream
                .alloc_zeros::<u32>(src.len())
                .map_err(|e| driver_err(&format!("alloc {name}"), e))?,
        );
    }
    let dev = slot.as_mut().expect("slot allocated above");
    stream
        .memcpy_htod(src, &mut dev.slice_mut(0..src.len()))
        .map_err(|e| driver_err(&format!("upload {name}"), e))?;
    ctx.device().count_upload();
    let dev: &'a CudaSlice<u32> = dev;
    Ok(dev.slice(0..src.len()))
}

fn launch_word_grouped_kernel(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    scratch: &mut CliffordBatchScratch,
    num_groups: usize,
    num_cross_word: usize,
) -> Result<()> {
    if num_groups == 0 && num_cross_word == 0 {
        return Ok(());
    }
    let num_rows_usize = tableau.total_rows();
    let num_rows = require_i32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let num_words_i = require_i32("stab_apply_word_grouped", "num_words", tableau.num_words())?;

    let (stream, func) = stream_and_fn(ctx, "stab_apply_word_grouped")?;

    let group_words_dev = stage_u32(
        ctx,
        stream,
        &mut scratch.group_words_dev,
        &scratch.group_words,
        "group_words",
    )?;
    let group_offsets_dev = stage_u32(
        ctx,
        stream,
        &mut scratch.group_offsets_dev,
        &scratch.group_offsets,
        "group_offsets",
    )?;
    let ops_dev = stage_u32(
        ctx,
        stream,
        &mut scratch.ops_dev,
        &scratch.ops_flat,
        "ops_flat",
    )?;
    let cross_dev = stage_u32(
        ctx,
        stream,
        &mut scratch.cross_dev,
        &scratch.cross_word,
        "cross_word",
    )?;

    let num_groups_i = require_i32("stab_apply_word_grouped", "num_groups", num_groups)?;
    let num_cross_word_i =
        require_i32("stab_apply_word_grouped", "num_cross_word", num_cross_word)?;
    let block_threads = num_groups
        .next_power_of_two()
        .clamp(32, BLOCK_SIZE as usize) as u32;
    let num_rows_grid = require_u32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let cfg = linear_cfg(block_threads, num_rows_grid);

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_rows)
        .arg(&num_words_i)
        .arg(&group_words_dev)
        .arg(&group_offsets_dev)
        .arg(&ops_dev)
        .arg(&num_groups_i)
        .arg(&cross_dev)
        .arg(&num_cross_word_i);
    // SAFETY: signature matches the kernel declaration. Each block owns one
    // row. Threads stripe over word-disjoint groups within that row, and
    // thread 0 alone handles the cross-word tail plus final phase write.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_apply_word_grouped", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Block size for `stab_rowmul_words`. Chosen so a single warp-shuffle round
/// followed by one shared-memory reduction covers every supported num_words value
/// (≤ 5000 qubits ⇒ num_words ≤ 79 ⇒ one thread per word fits in a block).
const ROWMUL_BLOCK_SIZE: u32 = 128;

/// XOR `src_row` into `dst_row` and update `dst_row`'s phase per the
/// Aaronson-Gottesman g-function. Launched as a single block per call.
pub(crate) fn launch_rowmul_words(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    src_row: usize,
    dst_row: usize,
) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "stab_rowmul_words")?;

    let nw_usize = tableau.num_words();
    if nw_usize == 0 {
        return Ok(());
    }
    let nw = require_i32("stab_rowmul_words", "num_words", nw_usize)?;
    let src_i = require_i32("stab_rowmul_words", "src_row", src_row)?;
    let dst_i = require_i32("stab_rowmul_words", "dst_row", dst_row)?;
    let cfg = linear_cfg(ROWMUL_BLOCK_SIZE, 1);

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder.arg(xz).arg(phase).arg(&nw).arg(&src_i).arg(&dst_i);
    // SAFETY: signature (u64*, u8*, i32, i32, i32); single block operates on
    // one (src, dst) row pair. All threads of the block write to disjoint
    // words of dst_row and a single phase byte; no inter-block hazard.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_rowmul_words", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

/// Block size for the measurement kernels that use one block per row
/// (`stab_measure_cascade`) and the single-block measurement kernels
/// (`stab_measure_fixup`, `stab_measure_deterministic`). num_words bound:
/// see `ROWMUL_BLOCK_SIZE`.
const MEASURE_BLOCK_SIZE: u32 = 128;

/// Scan stabilizer rows `n..2n` for the minimum row index whose X-bit at
/// `target` is set. Returns `Some(row)` when a pivot exists (random branch),
/// `None` when every stabilizer commutes with `Z_target` (deterministic
/// branch). One i32 d2h roundtrip on the sentinel.
pub(crate) fn launch_measure_find_pivot(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
) -> Result<Option<usize>> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(None);
    }
    let (stream, func) = stream_and_fn(ctx, "stab_measure_find_pivot")?;

    let sentinel = require_i32(
        "stab_measure_find_pivot",
        "sentinel",
        2usize.saturating_mul(n),
    )?;
    let num_qubits_i = require_i32("stab_measure_find_pivot", "num_qubits", n)?;
    let nw = require_i32("stab_measure_find_pivot", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_find_pivot", "target", target)?;
    let blocks = div_ceil_grid(
        "stab_measure_find_pivot",
        "num_qubits",
        n,
        MEASURE_BLOCK_SIZE,
    )?;
    let cfg = linear_cfg(MEASURE_BLOCK_SIZE, blocks);

    let mut host_pivot = [0_i32; 1];
    {
        let (xz_buf, pivot_buf) = tableau.xz_pivot_mut();
        let xz = xz_buf.raw_mut();
        let out_pivot = pivot_buf.raw_mut();
        stream
            .memcpy_htod(&[sentinel], out_pivot)
            .map_err(|e| driver_err("reset find_pivot sentinel", e))?;
        ctx.device().count_upload();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(xz)
            .arg(&num_qubits_i)
            .arg(&nw)
            .arg(&target_i)
            .arg(&mut *out_pivot);
        // SAFETY: signature (const u64*, i32, i32, i32, int*). The kernel only
        // reads xz at (row, target-word) positions and atomicMin's into
        // out_pivot.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("stab_measure_find_pivot", e))?;
        }
        ctx.device().count_launch(0);
        stream
            .memcpy_dtoh(out_pivot, &mut host_pivot)
            .map_err(|e| driver_err("find_pivot dtoh", e))?;
        ctx.device().count_drain();
    }
    if host_pivot[0] >= sentinel {
        Ok(None)
    } else {
        Ok(Some(host_pivot[0] as usize))
    }
}

/// Rowmul the pivot row into every non-pivot row that carries an X-bit at
/// `target`. One block per non-scratch row; blocks not participating in the
/// cascade early-exit with negligible driver overhead.
pub(crate) fn launch_measure_cascade(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
    pivot_row: usize,
) -> Result<()> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(());
    }
    let (stream, func) = stream_and_fn(ctx, "stab_measure_cascade")?;

    let num_qubits_i = require_i32("stab_measure_cascade", "num_qubits", n)?;
    let nw = require_i32("stab_measure_cascade", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_cascade", "target", target)?;
    let pivot_i = require_i32("stab_measure_cascade", "pivot_row", pivot_row)?;
    let blocks = require_u32("stab_measure_cascade", "num_rows", 2usize.saturating_mul(n))?;
    let cfg = linear_cfg(MEASURE_BLOCK_SIZE, blocks);

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_qubits_i)
        .arg(&nw)
        .arg(&target_i)
        .arg(&pivot_i);
    // SAFETY: signature (u64*, u8*, i32, i32, i32, i32). Each block owns a
    // unique destination row (blockIdx.x); pivot_row is read-only throughout.
    // No inter-block hazard on xz or phase because writes target disjoint
    // rows.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_measure_cascade", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}

/// Post-cascade fixup: move pivot data into the paired destabiliser, install
/// `Z_target` with the measured outcome at the pivot row. Single block.
pub(crate) fn launch_measure_fixup(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
    pivot_row: usize,
    outcome: bool,
) -> Result<()> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(());
    }
    let (stream, func) = stream_and_fn(ctx, "stab_measure_fixup")?;

    let num_qubits_i = require_i32("stab_measure_fixup", "num_qubits", n)?;
    let nw = require_i32("stab_measure_fixup", "num_words", tableau.num_words())?;
    let target_i = require_i32("stab_measure_fixup", "target", target)?;
    let pivot_i = require_i32("stab_measure_fixup", "pivot_row", pivot_row)?;
    let outcome_u8: u8 = outcome as u8;
    let cfg = linear_cfg(MEASURE_BLOCK_SIZE, 1);

    let mut builder = stream.launch_builder(&func);
    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    builder
        .arg(xz)
        .arg(phase)
        .arg(&num_qubits_i)
        .arg(&nw)
        .arg(&target_i)
        .arg(&pivot_i)
        .arg(&outcome_u8);
    // SAFETY: signature (u64*, u8*, i32, i32, i32, i32, u8). Single block
    // writes pivot_row and destab_row disjointly; runs only after the cascade
    // launch completes (single stream, serial ordering).
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_measure_fixup", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

/// Deterministic branch: rowmul stabiliser rows `n+i` for every `i` whose
/// destabiliser has an X at `target` into the scratch row, then read back the
/// scratch row's phase as the measurement outcome. One u8 d2h roundtrip.
pub(crate) fn launch_measure_deterministic(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    target: usize,
) -> Result<bool> {
    let n = tableau.num_qubits();
    if n == 0 {
        return Ok(false);
    }
    let (stream, func) = stream_and_fn(ctx, "stab_measure_deterministic")?;

    let num_qubits_i = require_i32("stab_measure_deterministic", "num_qubits", n)?;
    let nw = require_i32(
        "stab_measure_deterministic",
        "num_words",
        tableau.num_words(),
    )?;
    let target_i = require_i32("stab_measure_deterministic", "target", target)?;
    let cfg = linear_cfg(MEASURE_BLOCK_SIZE, 1);

    let mut host_out = [0u8; 1];
    {
        let (xz_buf, phase_buf, outcome_buf) = tableau.xz_phase_outcome_mut();
        let xz = xz_buf.raw_mut();
        let phase = phase_buf.raw_mut();
        let out_outcome = outcome_buf.raw_mut();
        stream
            .memcpy_htod(&[0u8], out_outcome)
            .map_err(|e| driver_err("reset deterministic outcome", e))?;
        ctx.device().count_upload();
        let mut builder = stream.launch_builder(&func);
        builder
            .arg(xz)
            .arg(phase)
            .arg(&num_qubits_i)
            .arg(&nw)
            .arg(&target_i)
            .arg(&mut *out_outcome);
        // SAFETY: signature (u64*, u8*, i32, i32, i32, u8*). Single block runs
        // a serial loop over i=0..n; scratch row (index 2n) is the only
        // destination.
        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| driver_err("stab_measure_deterministic", e))?;
        }
        ctx.device().count_launch(0);
        stream
            .memcpy_dtoh(out_outcome, &mut host_out)
            .map_err(|e| driver_err("deterministic outcome dtoh", e))?;
        ctx.device().count_drain();
    }
    Ok(host_out[0] != 0)
}

/// Word offsets into the measurement run buffer for a run of `num_meas` measurements,
/// in the layout `stabilizer.cu` documents under "Measurement runs".
#[derive(Clone, Copy)]
pub(crate) struct MeasureRunLayout {
    num_meas: usize,
}

impl MeasureRunLayout {
    pub(crate) fn new(num_meas: usize) -> Self {
        Self { num_meas }
    }

    pub(crate) fn targets(self) -> std::ops::Range<usize> {
        1..1 + self.num_meas
    }

    pub(crate) fn pivots(self) -> std::ops::Range<usize> {
        1 + self.num_meas..1 + 2 * self.num_meas
    }

    pub(crate) fn rand(self) -> std::ops::Range<usize> {
        let start = 1 + 2 * self.num_meas;
        start..start + self.num_meas.div_ceil(32)
    }

    pub(crate) fn outcomes(self) -> std::ops::Range<usize> {
        let start = self.rand().end;
        start..start + self.num_meas
    }

    /// Words the host uploads: everything before the outcomes, which kernels write.
    pub(crate) fn upload_len(self) -> usize {
        self.rand().end
    }

    pub(crate) fn len(self) -> usize {
        self.outcomes().end
    }
}

/// The run buffer was last sized and uploaded for `layout`, so every index the run
/// kernels form from it lies inside the buffer.
fn debug_assert_run_buffer_holds(tableau: &GpuTableau, layout: MeasureRunLayout) {
    debug_assert!(
        tableau
            .measure_run
            .as_ref()
            .is_some_and(|bufs| bufs.run.len() >= layout.len()),
        "run buffer shorter than a layout of {} measurements",
        layout.num_meas
    );
}

/// Threads per block for the run kernels whose threads stripe over row words, so
/// narrow rows do not hold idle threads.
fn run_row_block(num_words: usize) -> u32 {
    num_words
        .next_power_of_two()
        .clamp(32, MEASURE_BLOCK_SIZE as usize) as u32
}

/// Upload the host part of a measurement run (cursor, targets, sentinel pivots and
/// random bits) into the tableau's grow-only run buffer.
pub(crate) fn upload_measure_run(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    layout: MeasureRunLayout,
    host: &[u32],
) -> Result<()> {
    debug_assert_eq!(host.len(), layout.upload_len());
    tableau.ensure_measure_run(layout.len())?;
    let stream = ctx.device().stream()?;
    let (_, _, bufs) = tableau.xz_phase_run_mut();
    stream
        .memcpy_htod(host, &mut bufs.run.raw_mut().slice_mut(0..host.len()))
        .map_err(|e| driver_err("upload measure run", e))?;
    ctx.device().count_upload();
    Ok(())
}

/// Read the first `host.len()` words of the run buffer back.
pub(crate) fn read_measure_run(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    host: &mut [u32],
) -> Result<()> {
    let stream = ctx.device().stream()?;
    let (_, _, bufs) = tableau.xz_phase_run_mut();
    stream
        .memcpy_dtoh(&bufs.run.raw().slice(0..host.len()), host)
        .map_err(|e| driver_err("measure run dtoh", e))?;
    ctx.device().count_drain();
    Ok(())
}

/// Store in `host_mask` the OR of each X word over the stabilizer rows: a qubit with
/// its bit clear measures deterministically. One launch and one readback.
pub(crate) fn launch_run_random_mask(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    host_mask: &mut [u64],
) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "stab_run_random_mask")?;
    let nw_usize = tableau.num_words();
    let n = require_i32("stab_run_random_mask", "num_qubits", tableau.num_qubits())?;
    let nw = require_i32("stab_run_random_mask", "num_words", nw_usize)?;
    let blocks = require_u32("stab_run_random_mask", "num_words", nw_usize)?;
    let cfg = linear_cfg(MEASURE_BLOCK_SIZE, blocks);

    let (xz_buf, _, bufs) = tableau.xz_phase_run_mut();
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(xz_buf.raw_mut())
        .arg(&n)
        .arg(&nw)
        .arg(bufs.mask.raw_mut());
    // SAFETY: signature (const u64*, i32, i32, u64*). Block `w` reads X word `w` of
    // the stabilizer rows and alone writes mask[w]; mask holds num_words words.
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| driver_err("stab_run_random_mask", e))?;
    }
    ctx.device().count_launch(1);
    stream
        .memcpy_dtoh(&bufs.mask.raw().slice(0..nw_usize), host_mask)
        .map_err(|e| driver_err("random mask dtoh", e))?;
    ctx.device().count_drain();
    Ok(())
}

/// Shape arguments shared by the run kernels that take `(run, num_meas, m)`.
struct RunArgs {
    n: i32,
    nw: i32,
    num_meas: i32,
    m: i32,
}

impl RunArgs {
    fn new(op: &str, tableau: &GpuTableau, layout: MeasureRunLayout, m: usize) -> Result<Self> {
        debug_assert!(
            m < layout.num_meas,
            "{op}: measurement {m} past {}",
            layout.num_meas
        );
        debug_assert_run_buffer_holds(tableau, layout);
        Ok(Self {
            n: require_i32(op, "num_qubits", tableau.num_qubits())?,
            nw: require_i32(op, "num_words", tableau.num_words())?,
            num_meas: require_i32(op, "num_meas", layout.num_meas)?,
            m: require_i32(op, "m", m)?,
        })
    }
}

/// Collapse measurement `m` of the run: pivot search, cascade and fixup, three launches
/// with no host copy. The cascade and fixup return at once when `m` has no pivot, and
/// the fixup takes the run's next random bit as the outcome.
pub(crate) fn launch_run_collapse(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    layout: MeasureRunLayout,
    m: usize,
) -> Result<()> {
    let n_usize = tableau.num_qubits();
    let row_block = run_row_block(tableau.num_words());
    let pivot_blocks = div_ceil_grid(
        "stab_run_find_pivot",
        "num_qubits",
        n_usize,
        MEASURE_BLOCK_SIZE,
    )?;
    let cascade_blocks = require_u32("stab_run_cascade", "num_rows", 2 * n_usize)?;
    let args = RunArgs::new("stab_run_collapse", tableau, layout, m)?;
    let (xz_buf, phase_buf, bufs) = tableau.xz_phase_run_mut();
    let xz = xz_buf.raw_mut();
    let phase = phase_buf.raw_mut();
    let run = bufs.run.raw_mut();

    let (stream, func) = stream_and_fn(ctx, "stab_run_find_pivot")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&mut *xz)
        .arg(&args.n)
        .arg(&args.nw)
        .arg(&mut *run)
        .arg(&args.num_meas)
        .arg(&args.m);
    // SAFETY: signature (const u64*, i32, i32, u32*, i32, i32). Threads read one word
    // per stabilizer row and atomicMin into pivots[m]; m < num_meas and the run buffer
    // holds the uploaded layout (checked in `RunArgs::new`), so the slot is in bounds.
    unsafe {
        builder
            .launch(linear_cfg(MEASURE_BLOCK_SIZE, pivot_blocks))
            .map_err(|e| driver_err("stab_run_find_pivot", e))?;
    }
    ctx.device().count_launch(0);

    let (_, func) = stream_and_fn(ctx, "stab_run_cascade")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&mut *xz)
        .arg(&mut *phase)
        .arg(&args.n)
        .arg(&args.nw)
        .arg(&mut *run)
        .arg(&args.num_meas)
        .arg(&args.m);
    // SAFETY: signature (u64*, u8*, i32, i32, u32*, i32, i32). Each block owns one
    // destination row; the pivot row and the run buffer are read only, at slots inside
    // the uploaded layout because m < num_meas.
    unsafe {
        builder
            .launch(linear_cfg(row_block, cascade_blocks))
            .map_err(|e| driver_err("stab_run_cascade", e))?;
    }
    ctx.device().count_launch(1);

    let (_, func) = stream_and_fn(ctx, "stab_run_fixup")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(&mut *xz)
        .arg(&mut *phase)
        .arg(&args.n)
        .arg(&args.nw)
        .arg(&mut *run)
        .arg(&args.num_meas)
        .arg(&args.m);
    // SAFETY: signature (u64*, u8*, i32, i32, u32*, i32, i32). A single block writes
    // the pivot and destabilizer rows; thread 0 alone writes the cursor and
    // outcomes[m], inside the layout because m < num_meas. Stream order puts it after
    // the cascade.
    unsafe {
        builder
            .launch(linear_cfg(row_block, 1))
            .map_err(|e| driver_err("stab_run_fixup", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

/// Resolve measurement `m` of the run into `outcomes[m]` by the serial deterministic
/// product, for a state that keeps X support.
pub(crate) fn launch_run_deterministic(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    layout: MeasureRunLayout,
    m: usize,
) -> Result<()> {
    let (stream, func) = stream_and_fn(ctx, "stab_run_deterministic")?;
    let row_block = run_row_block(tableau.num_words());
    let args = RunArgs::new("stab_run_deterministic", tableau, layout, m)?;
    let (xz_buf, phase_buf, bufs) = tableau.xz_phase_run_mut();
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(xz_buf.raw_mut())
        .arg(phase_buf.raw_mut())
        .arg(&args.n)
        .arg(&args.nw)
        .arg(bufs.run.raw_mut())
        .arg(&args.num_meas)
        .arg(&args.m);
    // SAFETY: signature (u64*, u8*, i32, i32, u32*, i32, i32). A single block runs a
    // serial loop whose only destination is the scratch row; thread 0 alone writes
    // outcomes[m], inside the uploaded layout because m < num_meas.
    unsafe {
        builder
            .launch(linear_cfg(row_block, 1))
            .map_err(|e| driver_err("stab_run_deterministic", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

/// Resolve every measurement of the run left without a pivot from the stabilizer
/// signs, for a state whose stabilizers carry no X. Two launches, no host copy.
pub(crate) fn launch_run_diagonal_finish(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    layout: MeasureRunLayout,
) -> Result<()> {
    let nw_usize = tableau.num_words();
    let n = require_i32(
        "stab_run_det_accumulate",
        "num_qubits",
        tableau.num_qubits(),
    )?;
    let nw = require_i32("stab_run_det_accumulate", "num_words", nw_usize)?;
    let num_meas = require_i32("stab_run_det_outcomes", "num_meas", layout.num_meas)?;
    let acc_blocks = require_u32("stab_run_det_accumulate", "num_words", nw_usize)?;
    let outcome_blocks = div_ceil_grid(
        "stab_run_det_outcomes",
        "num_meas",
        layout.num_meas,
        MEASURE_BLOCK_SIZE,
    )?;
    debug_assert_run_buffer_holds(tableau, layout);
    let (xz_buf, phase_buf, bufs) = tableau.xz_phase_run_mut();

    let (stream, func) = stream_and_fn(ctx, "stab_run_det_accumulate")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(xz_buf.raw_mut())
        .arg(phase_buf.raw_mut())
        .arg(&n)
        .arg(&nw)
        .arg(bufs.acc.raw_mut());
    // SAFETY: signature (const u64*, const u8*, i32, i32, u64*). Block `w` reads X
    // word `w` of the destabilizer rows and alone writes acc[w]; acc holds num_words
    // words.
    unsafe {
        builder
            .launch(linear_cfg(MEASURE_BLOCK_SIZE, acc_blocks))
            .map_err(|e| driver_err("stab_run_det_accumulate", e))?;
    }
    ctx.device().count_launch(1);

    let (_, func) = stream_and_fn(ctx, "stab_run_det_outcomes")?;
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(bufs.acc.raw_mut())
        .arg(&n)
        .arg(bufs.run.raw_mut())
        .arg(&num_meas);
    // SAFETY: signature (const u64*, i32, u32*, i32). Thread `m < num_meas` reads acc
    // and writes outcomes[m] alone; the run buffer holds the layout's words, checked
    // above.
    unsafe {
        builder
            .launch(linear_cfg(MEASURE_BLOCK_SIZE, outcome_blocks))
            .map_err(|e| driver_err("stab_run_det_outcomes", e))?;
    }
    ctx.device().count_launch(0);
    Ok(())
}

/// Apply `CX(k, k + 1)` for `k` in `start..end` in one launch with no upload, in place
/// of a word group flush at each word boundary the chain crosses.
pub(crate) fn launch_cx_chain(
    ctx: &GpuContext,
    tableau: &mut GpuTableau,
    start: usize,
    end: usize,
) -> Result<()> {
    debug_assert!(
        start < end && end < tableau.num_qubits(),
        "CX chain {start}..{end} outside {} qubits",
        tableau.num_qubits()
    );
    let (stream, func) = stream_and_fn(ctx, "stab_apply_cx_chain")?;
    let rows_usize = tableau.total_rows();
    let num_rows = require_i32("stab_apply_cx_chain", "num_rows", rows_usize)?;
    let nw = require_i32("stab_apply_cx_chain", "num_words", tableau.num_words())?;
    let start_i = require_i32("stab_apply_cx_chain", "start", start)?;
    let end_i = require_i32("stab_apply_cx_chain", "end", end)?;
    let blocks = div_ceil_grid("stab_apply_cx_chain", "num_rows", rows_usize, BLOCK_SIZE)?;

    let (xz_buf, phase_buf) = tableau.xz_phase_mut();
    let mut builder = stream.launch_builder(&func);
    builder
        .arg(xz_buf.raw_mut())
        .arg(phase_buf.raw_mut())
        .arg(&num_rows)
        .arg(&nw)
        .arg(&start_i)
        .arg(&end_i);
    // SAFETY: signature (u64*, u8*, i32, i32, i32, i32). Thread `r` alone reads and
    // writes row `r` and phase[r], and `end` is below num_qubits, so every word index
    // stays inside the row.
    unsafe {
        builder
            .launch(linear_cfg(BLOCK_SIZE, blocks))
            .map_err(|e| driver_err("stab_apply_cx_chain", e))?;
    }
    ctx.device().count_launch(1);
    Ok(())
}
