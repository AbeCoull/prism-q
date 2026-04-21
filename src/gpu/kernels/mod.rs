//! GPU kernels, PTX source compiled once at device construction, plus per-operation
//! launcher functions.
//!
//! The PTX module is composed by concatenating each backend's CUDA C source (dense for
//! the statevector path, stabilizer for the stabilizer path). NVRTC compiles the
//! combined source once per `GpuContext`; `KERNEL_NAMES` lists every entry point from
//! every backend so `GpuDevice::new` can pre-resolve them all.

pub(crate) mod bts;
pub(crate) mod dense;
pub(crate) mod stabilizer;

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
    "measure_collapse",
    "compute_probabilities",
    "scale_state",
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
];
