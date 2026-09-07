//! GPU kernels, PTX source compiled once at device construction, plus per-operation
//! launcher functions.
//!
//! The PTX module is composed by concatenating each backend's CUDA C source (dense for
//! the statevector path, stabilizer for the stabilizer path). NVRTC compiles the
//! combined source once per `GpuContext`; `KERNEL_NAMES` lists every entry point from
//! every backend so `GpuDevice::new` can pre-resolve them all.

pub(crate) mod bts;
pub(crate) mod dense;
pub(crate) mod density;
pub(crate) mod stabilizer;

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, ValidAsZeroBits,
};

use crate::error::{PrismError, Result};
use crate::gpu::GpuContext;
use crate::gpu::device::GpuDevice;
use crate::gpu::memory::GpuBuffer;

pub(super) fn stream_and_fn<'a>(
    ctx: &'a GpuContext,
    name: &'static str,
) -> Result<(&'a Arc<CudaStream>, CudaFunction)> {
    let device = ctx.device();
    Ok((device.stream()?, device.function(name)?))
}

pub(super) fn linear_cfg(block_size: u32, grid_blocks: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

pub(super) fn launch_err(op: &str, err: impl std::fmt::Display) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {err}"),
    }
}

pub(super) fn launch_limit_err(op: &str, name: &str, value: usize, limit: &str) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {name}={value} exceeds {limit} kernel limit"),
    }
}

pub(super) fn require_i32(op: &str, name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| launch_limit_err(op, name, value, "i32"))
}

pub(super) fn require_u32(op: &str, name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| launch_limit_err(op, name, value, "u32"))
}

pub(super) fn div_ceil_grid(op: &str, name: &str, value: usize, block: u32) -> Result<u32> {
    Ok(require_u32(op, name, value)?.div_ceil(block).max(1))
}

/// Scratch buffers reused across launches that need to upload small per-call
/// metadata (sorted-qubit lists, packed lookup tables, fused gate matrices).
///
/// Replaces the per-call `clone_htod` allocate-and-upload pattern with a
/// grow-only resident allocation that callers fill via `copy_from_host`. The
/// allocation is tied to the `GpuContext`; access is serialised through a
/// `Mutex` because all dense launches share the same CUDA stream anyway.
///
/// Worst case across the dense launchers is one f64 buffer plus three i32
/// buffers (`launch_apply_batch_rzz`); the four named slots cover every
/// existing call site without sharing across overlapping arguments.
#[derive(Default)]
pub(crate) struct LauncherScratch {
    pub(crate) f64_a: Option<GpuBuffer<f64>>,
    pub(crate) u64_a: Option<GpuBuffer<u64>>,
    pub(crate) u64_b: Option<GpuBuffer<u64>>,
    /// Per-block partials for [`super::dense::measure_prob_one`]. Sized by the
    /// number of grid blocks at the largest qubit count seen so far.
    pub(crate) measure_partials: Option<GpuBuffer<f64>>,
    /// One-element output for the measure-prob finalize reduction.
    pub(crate) measure_result: Option<GpuBuffer<f64>>,
    /// Four-element output for the reduced-density-matrix finalize reduction.
    /// Separate from `measure_result`: readback length must match the buffer.
    pub(crate) rdm_result: Option<GpuBuffer<f64>>,
    /// The `2^n` diagonal of a density matrix, sized to the mixture width.
    pub(crate) dm_diag: Option<GpuBuffer<f64>>,
    /// Two f64s per Pauli mask for the density-matrix expectation finalize.
    pub(crate) dm_result: Option<GpuBuffer<f64>>,
    /// One upload per gate for what does not fit parameter space; see [`stage_blob`].
    pub(crate) blob: BlobScratch,
}

const BLOB_ALIGN: usize = 16;

/// Device addresses of one staged blob: the `int` list at the start and the `double`
/// block at a 16-byte-aligned offset behind it.
#[derive(Clone, Copy)]
pub(crate) struct Staged {
    pub(crate) ints: u64,
    pub(crate) f64s: u64,
}

/// Host and device halves of the per-gate upload behind [`stage_blob`].
#[derive(Default)]
pub(crate) struct BlobScratch {
    host: Vec<u64>,
    device: Option<GpuBuffer<u64>>,
}

/// Pack `ints` and `f64_len` doubles written through `fill` into one upload and return
/// the device addresses of both halves.
///
/// Everything a kernel can take by value goes in parameter space instead; this is for
/// the lookup tables and metadata of the batched diagonal gates and the density-matrix
/// tables, which exceed the 4 KB parameter limit. The copy is synchronous from pageable
/// memory, so it costs one stream drain per gate, and the buffer is reused as soon as
/// the copy returns because that drain has already retired the kernel that read it.
pub(crate) fn stage_blob(
    scratch: &mut BlobScratch,
    device: &GpuDevice,
    ints: impl ExactSizeIterator<Item = i32>,
    f64_len: usize,
    fill: impl FnOnce(&mut [f64]),
) -> Result<Staged> {
    let int_len = ints.len();
    let f64_off = (int_len * size_of::<i32>()).next_multiple_of(BLOB_ALIGN);
    let words = (f64_off + f64_len * size_of::<f64>()) / size_of::<u64>();
    let host = &mut scratch.host;
    if host.len() < words {
        host.resize(words, 0);
    }
    // SAFETY: `host` holds at least `words` u64s, so both views lie inside it; the int
    // view ends at `f64_off`, where the f64 view starts, so they are disjoint; and a u64
    // buffer is aligned for both element types.
    unsafe {
        let base = host.as_mut_ptr().cast::<u8>();
        let int_dst = std::slice::from_raw_parts_mut(base.cast::<i32>(), int_len);
        for (d, v) in int_dst.iter_mut().zip(ints) {
            *d = v;
        }
        let f64_dst = std::slice::from_raw_parts_mut(base.add(f64_off).cast::<f64>(), f64_len);
        fill(f64_dst);
    }
    let buf = ensure_scratch(&mut scratch.device, device, &host[..words])?;
    let stream = device.stream()?;
    let (base, _sync) = buf.raw().device_ptr(stream);
    Ok(Staged {
        ints: base,
        f64s: base + f64_off as u64,
    })
}

/// Stage complex values as interleaved re/im f64 pairs.
pub(crate) fn stage_complex(
    scratch: &mut BlobScratch,
    device: &GpuDevice,
    values: &[num_complex::Complex64],
) -> Result<Staged> {
    stage_blob(
        scratch,
        device,
        std::iter::empty(),
        2 * values.len(),
        |dst| {
            for (pair, v) in dst.chunks_exact_mut(2).zip(values) {
                pair[0] = v.re;
                pair[1] = v.im;
            }
        },
    )
}

/// Ensure `slot` has at least `host.len()` elements allocated, growing if not,
/// and copy `host` into the slot. Returns the device buffer for argument
/// passing.
pub(crate) fn ensure_scratch<'a, T: DeviceRepr + ValidAsZeroBits>(
    slot: &'a mut Option<GpuBuffer<T>>,
    device: &GpuDevice,
    host: &[T],
) -> Result<&'a GpuBuffer<T>> {
    let needed = host.len().max(1);
    let realloc = match slot.as_ref() {
        Some(buf) => buf.len() < needed,
        None => true,
    };
    if realloc {
        *slot = Some(GpuBuffer::<T>::alloc_zeros(device, needed)?);
    }
    if !host.is_empty() {
        slot.as_mut().unwrap().copy_from_host(device, host)?;
    }
    Ok(slot.as_ref().unwrap())
}

/// Ensure `slot` has at least `needed` elements allocated, growing if not.
/// Returns a mutable reference suitable for kernel write-targets. Unlike
/// [`ensure_scratch`], does not perform any host-to-device copy.
pub(crate) fn ensure_capacity<'a, T: DeviceRepr + ValidAsZeroBits>(
    slot: &'a mut Option<GpuBuffer<T>>,
    device: &GpuDevice,
    needed: usize,
) -> Result<&'a mut GpuBuffer<T>> {
    let needed = needed.max(1);
    let realloc = match slot.as_ref() {
        Some(buf) => buf.len() < needed,
        None => true,
    };
    if realloc {
        *slot = Some(GpuBuffer::<T>::alloc_zeros(device, needed)?);
    }
    Ok(slot.as_mut().unwrap())
}

/// Combined CUDA C source for the GPU PTX module.
///
/// Concatenates each backend's kernel source. Any new backend that adds its own
/// module here (for example an MPS GPU path later) would append its source the same
/// way and register its entry-point names in [`KERNEL_NAMES`].
pub(crate) fn kernel_source() -> String {
    let mut src = dense::kernel_source();
    src.push('\n');
    src.push_str(&stabilizer::kernel_source());
    src.push('\n');
    src.push_str(&bts::kernel_source());
    src.push('\n');
    src.push_str(density::KERNEL_SOURCE);
    src
}

/// Every kernel entry point that appears in the materialised PTX source.
///
/// `GpuDevice::new` pre-resolves each name once so gate dispatch does not pay the
/// driver-lookup cost per launch. New backends extend this list with their own
/// entry-point names.
pub(crate) const KERNEL_NAMES: &[&str] = &[
    // Dense statevector kernels.
    "set_initial_state",
    "apply_gate_1q",
    "apply_diagonal_1q",
    "apply_cx",
    "apply_cz",
    "apply_swap",
    "apply_parity_phase",
    "apply_cu",
    "apply_cu_phase",
    "apply_mcu",
    "apply_mcu_phase",
    "apply_fused_2q",
    "measure_prob_one",
    "measure_prob_one_finalize",
    "rdm_qubit",
    "rdm_qubit_finalize",
    "measure_collapse",
    "compute_probabilities",
    "apply_multi_fused_diagonal",
    "apply_batch_phase",
    "apply_batch_rzz",
    "apply_diagonal_batch",
    "apply_multi_fused_tiled",
    // Stabilizer tableau kernels.
    "stab_set_initial_tableau",
    "stab_apply_word_grouped",
    "stab_rowmul_words",
    "stab_measure_find_pivot",
    "stab_measure_cascade",
    "stab_measure_fixup",
    "stab_measure_deterministic",
    // Block-triangular sampling.
    "bts_sample_meas_major",
    "bts_popcount_rows",
    "bts_count_meas_major_upto8",
    "bts_count_shot_major_upto8",
    "bts_count_used_slots",
    "bts_compact_counts_upto8",
    "bts_transpose_meas_to_shot",
    "bts_apply_noise_masks_meas_major",
    "bts_generate_and_apply_noise_meas_major_by_row",
    // Density-matrix sweeps over the embedded 2n-qubit buffer.
    "dm_diagonal",
    "dm_norm_sqr",
    "dm_project",
    "dm_reset",
    "dm_conjugate",
    "dm_diagonal_sandwich",
    "dm_kraus_2q_diagonal",
    "dm_kraus_2q_dense",
    "dm_depolarizing_2q",
    "dm_outer_product",
    "dm_pauli_expect",
    "dm_pauli_expect_finalize",
];

#[cfg(test)]
mod sync_tests {
    use super::{KERNEL_NAMES, kernel_source};
    use std::collections::BTreeSet;

    /// Names following each `__global__ void` marker in the PTX source.
    fn source_entry_points(src: &str) -> BTreeSet<String> {
        const MARKER: &str = "__global__ void ";
        let mut names = BTreeSet::new();
        let mut rest = src;
        while let Some(pos) = rest.find(MARKER) {
            rest = &rest[pos + MARKER.len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                names.insert(name);
            }
        }
        names
    }

    // `KERNEL_NAMES` must match the kernels in the CUDA source exactly: a missing
    // name is never pre-resolved by `GpuDevice::new`, an orphan name is dead drift.
    // Inspects the source string only, so it needs no GPU device.
    #[test]
    fn kernel_names_match_source_entry_points() {
        let src = kernel_source();
        let in_source = source_entry_points(&src);
        let declared: BTreeSet<String> = KERNEL_NAMES.iter().map(|s| s.to_string()).collect();

        let missing: Vec<&String> = in_source.difference(&declared).collect();
        let orphan: Vec<&String> = declared.difference(&in_source).collect();

        assert!(
            missing.is_empty(),
            "kernels defined in source but absent from KERNEL_NAMES \
             (GpuDevice::new would not pre-resolve them): {missing:?}"
        );
        assert!(
            orphan.is_empty(),
            "KERNEL_NAMES entries with no matching kernel in source: {orphan:?}"
        );
    }
}
