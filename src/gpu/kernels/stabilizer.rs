//! Stabilizer tableau kernels. CUDA C source compiled to PTX at runtime, plus launch
//! helpers in Rust.
//!
//! Tableau layout mirrors the CPU `StabilizerBackend`
//! (`src/backend/stabilizer/mod.rs`):
//!
//! - `xz`: `(2n+1)` rows × `2 * num_words` u64 words per row. Word ordering per row is
//!   X-bits in `[0, num_words)` then Z-bits in `[num_words, 2*num_words)`.
//! - `phase`: `(2n+1)` bytes, one per row (0 = +1, 1 = -1). Bytewise rather than
//!   bit-packed so rowmul phase writes do not require atomic RMW.
//! - Scratch row sits at index `2n` and is used only during measurement.
//!
//! Every entry-point name is prefixed `stab_` so it cannot collide with the dense
//! statevector kernels when both sources are concatenated into a single PTX module.
//!
//! Landed kernels:
//!
//! - `stab_set_initial_tableau`: identity init.
//! - `stab_apply_batch`: batched dispatch of all eleven Clifford gates
//!   (H, S, Sdg, X, Y, Z, SX, SXdg, CX, CZ, SWAP) over a host-provided
//!   op list, one kernel launch per flush.
//! - `stab_rowmul_words`: XOR source row into destination row with
//!   Aaronson-Gottesman phase update.
//! - `stab_measure_find_pivot`, `stab_measure_cascade`,
//!   `stab_measure_fixup`, `stab_measure_deterministic`: on-device Z-basis
//!   measurement. Eliminates the tableau copy-back previously needed per
//!   measure or reset.
//!
//! `stab_apply_batch` uses a one-block-per-row strategy across the full
//! `2n+1`-row tableau. Threads stripe over independent word groups inside the
//! row, which keeps reads and writes local to one row instead of striding the
//! same word across many rows. The cross-word tail stays serial on thread 0,
//! because different ops may still touch different bits in the same packed
//! u64 word.
//!
//! `stab_rowmul_words` launches a single block per call; threads partition
//! the `num_words` word loop and reduce their per-word phase contributions
//! via warp-shuffle plus shared memory.
//!
//! Measurement orchestrates four small kernels: a pivot search with an
//! atomicMin sentinel, a cascade that rowmul's the pivot into every row
//! carrying an X at the target (one block per row, most blocks early-exit),
//! a single-block fixup that moves pivot data into the paired destabilizer
//! and installs the measured Z_q, and a deterministic-branch kernel that
//! serialises rowmul's of stabilisers whose paired destabilisers anticommute
//! with Z_q into the scratch row and reads its phase.

use cudarc::driver::{CudaSlice, PushKernelArg};

use crate::error::Result;

use super::super::{GpuContext, GpuTableau};
use super::{div_ceil_grid, launch_err, linear_cfg, require_i32, require_u32, stream_and_fn};

const BLOCK_SIZE: u32 = 128;

/// Stabilizer CUDA C source. Returned by `kernel_source()` and concatenated into the
/// combined PTX module alongside the dense kernels. No template substitutions needed:
/// all tableau shapes are passed as kernel arguments rather than compile-time constants.
const KERNEL_SOURCE: &str = include_str!("stabilizer.cu");

/// Return the stabilizer CUDA C source for concatenation into the shared PTX module.
pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE.to_string()
}

/// Initialise a freshly-allocated `GpuTableau` to the identity tableau: destabilizer
/// rows are X_i, stabilizer rows are Z_i, scratch row is all zero, phase is all zero.
///
/// Assumes `xz` and `phase` were allocated via `GpuBuffer::alloc_zeros` (so everything
/// else is already zero); this kernel only writes the identity bits.
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
            .map_err(|e| launch_err("stab_set_initial_tableau", e))?;
    }
    Ok(())
}

/// Clifford opcodes consumed by `stab_apply_batch`. Values are part of the ABI
/// between the host queue and the batch kernel and must stay in sync with the
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

/// Apply a batch of queued Clifford ops to the device tableau in a single
/// launch.
///
/// `ops` is a flat `u32` buffer of length `CLIFOP_STRIDE * num_ops` laid out
/// as `[opcode, a, b, pad]` quads. Opcodes are the constants in [`op`]. The
/// kernel maps one block to each tableau row, parallelises the disjoint
/// same-word groups within that row, and leaves the cross-word tail serial
/// on thread 0 to avoid shared-word races.
///
/// Host-side, this streams through `ops` (quads `[opcode, a, b, pad]`),
/// sorting into word groups keyed by target-word plus a cross-word 2q list.
/// Conflicts between a newly-enqueued op and the running cross-word qubit
/// set trigger a partial launch, mirroring the CPU `flush_all_with_cross_word`
/// discipline in `src/backend/stabilizer/kernels/batch.rs`. The kernel then
/// amortises memory traffic across every op in a group: one `rx[w]`/`rz[w]`
/// read and one write per thread per group regardless of how many ops the
/// group contains.
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
    let num_rows = tableau.total_rows();
    let num_words = tableau.num_words();
    if num_rows == 0 || num_words == 0 {
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
        if self.per_word_ops.len() < num_words {
            self.per_word_ops.resize_with(num_words, Vec::new);
        } else if self.per_word_ops.len() > num_words {
            self.per_word_ops.truncate(num_words);
        }
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
        for chunk in ops.chunks_exact(CLIFOP_STRIDE) {
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

#[allow(clippy::too_many_arguments)]
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
    if num_rows_usize == 0 {
        return Ok(());
    }
    let num_rows = require_i32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let num_words_i = require_i32("stab_apply_word_grouped", "num_words", tableau.num_words())?;

    let (stream, func) = stream_and_fn(ctx, "stab_apply_word_grouped")?;

    let group_words_src: &[u32] = if scratch.group_words.is_empty() {
        &ZERO_U32
    } else {
        &scratch.group_words
    };
    let group_offsets_src: &[u32] = if scratch.group_offsets.is_empty() {
        &ZERO_U32
    } else {
        &scratch.group_offsets
    };
    let ops_src: &[u32] = if scratch.ops_flat.is_empty() {
        &ZERO_U32
    } else {
        &scratch.ops_flat
    };
    let cross_src: &[u32] = if scratch.cross_word.is_empty() {
        &ZERO_U32
    } else {
        &scratch.cross_word
    };

    let ensure_u32_buffer =
        |slot: &mut Option<CudaSlice<u32>>, len: usize, op: &str| -> Result<()> {
            let needed = len.max(1);
            if slot.as_ref().is_none_or(|buf| buf.len() < needed) {
                *slot = Some(
                    stream
                        .alloc_zeros::<u32>(needed)
                        .map_err(|e| launch_err(op, e))?,
                );
            }
            Ok(())
        };
    ensure_u32_buffer(
        &mut scratch.group_words_dev,
        group_words_src.len(),
        "alloc group_words",
    )?;
    ensure_u32_buffer(
        &mut scratch.group_offsets_dev,
        group_offsets_src.len(),
        "alloc group_offsets",
    )?;
    ensure_u32_buffer(&mut scratch.ops_dev, ops_src.len(), "alloc ops_flat")?;
    ensure_u32_buffer(&mut scratch.cross_dev, cross_src.len(), "alloc cross_word")?;

    {
        let dev = scratch
            .group_words_dev
            .as_mut()
            .expect("group_words_dev allocated above");
        let mut view = dev.slice_mut(0..group_words_src.len());
        stream
            .memcpy_htod(group_words_src, &mut view)
            .map_err(|e| launch_err("upload group_words", e))?;
    }
    {
        let dev = scratch
            .group_offsets_dev
            .as_mut()
            .expect("group_offsets_dev allocated above");
        let mut view = dev.slice_mut(0..group_offsets_src.len());
        stream
            .memcpy_htod(group_offsets_src, &mut view)
            .map_err(|e| launch_err("upload group_offsets", e))?;
    }
    {
        let dev = scratch.ops_dev.as_mut().expect("ops_dev allocated above");
        let mut view = dev.slice_mut(0..ops_src.len());
        stream
            .memcpy_htod(ops_src, &mut view)
            .map_err(|e| launch_err("upload ops_flat", e))?;
    }
    {
        let dev = scratch
            .cross_dev
            .as_mut()
            .expect("cross_dev allocated above");
        let mut view = dev.slice_mut(0..cross_src.len());
        stream
            .memcpy_htod(cross_src, &mut view)
            .map_err(|e| launch_err("upload cross_word", e))?;
    }

    let num_groups_i = require_i32("stab_apply_word_grouped", "num_groups", num_groups)?;
    let num_cross_word_i =
        require_i32("stab_apply_word_grouped", "num_cross_word", num_cross_word)?;
    let block_threads = num_groups
        .next_power_of_two()
        .clamp(32, BLOCK_SIZE as usize) as u32;
    let num_rows_grid = require_u32("stab_apply_word_grouped", "num_rows", num_rows_usize)?;
    let cfg = linear_cfg(block_threads, num_rows_grid);

    let group_words_dev = scratch
        .group_words_dev
        .as_ref()
        .expect("group_words_dev uploaded above")
        .slice(0..group_words_src.len());
    let group_offsets_dev = scratch
        .group_offsets_dev
        .as_ref()
        .expect("group_offsets_dev uploaded above")
        .slice(0..group_offsets_src.len());
    let ops_dev = scratch
        .ops_dev
        .as_ref()
        .expect("ops_dev uploaded above")
        .slice(0..ops_src.len());
    let cross_dev = scratch
        .cross_dev
        .as_ref()
        .expect("cross_dev uploaded above")
        .slice(0..cross_src.len());

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
            .map_err(|e| launch_err("stab_apply_word_grouped", e))?;
    }
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
            .map_err(|e| launch_err("stab_rowmul_words", e))?;
    }
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
            .map_err(|e| launch_err("reset find_pivot sentinel", e))?;
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
                .map_err(|e| launch_err("stab_measure_find_pivot", e))?;
        }
        stream
            .memcpy_dtoh(out_pivot, &mut host_pivot)
            .map_err(|e| launch_err("find_pivot dtoh", e))?;
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
            .map_err(|e| launch_err("stab_measure_cascade", e))?;
    }
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
            .map_err(|e| launch_err("stab_measure_fixup", e))?;
    }
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
            .map_err(|e| launch_err("reset deterministic outcome", e))?;
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
                .map_err(|e| launch_err("stab_measure_deterministic", e))?;
        }
        stream
            .memcpy_dtoh(out_outcome, &mut host_out)
            .map_err(|e| launch_err("deterministic outcome dtoh", e))?;
    }
    Ok(host_out[0] != 0)
}
