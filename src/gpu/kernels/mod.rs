//! GPU kernels — PTX source, compiled once at device construction, plus per-operation
//! launcher functions.

pub(crate) mod dense;

pub(crate) use dense::KERNEL_SOURCE;

/// Every kernel entry point that appears in `KERNEL_SOURCE`.
///
/// `GpuDevice::new` pre-resolves each name once so gate dispatch does not pay the
/// driver-lookup cost per launch.
pub(crate) const KERNEL_NAMES: &[&str] = &[
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
];
