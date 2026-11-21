//! Dense statevector kernels: the CUDA C template and its launchers. Kernels take the
//! interleaved state buffer as `double2 *` for 16-byte aligned vector loads.
//!
//! `BatchPhase`, `BatchRzz`, `DiagonalBatch`, and diagonal `MultiFused` each run as one
//! launch over per-group phase tables (built by the CPU `build_*_tables` helpers) plus
//! small shift and qubit metadata, instead of one launch per sub-gate. Non-diagonal
//! `MultiFused` runs as shared-memory tiled passes of `apply_multi_fused_tiled`;
//! `Multi2q` launches once per sub-gate, on the permutation or diagonal kernel when the
//! sub-gate's matrix is exactly one.

mod batch;
mod expect;
mod gates;
mod measure;
mod noise;
mod sample;
mod tiles;

use crate::backend::statevector::kernels as cpu_k;

pub(crate) use batch::{
    launch_apply_batch_phase, launch_apply_batch_rzz, launch_apply_diagonal_batch,
    launch_apply_multi_fused_diagonal,
};
pub(crate) use expect::{group_moments, operator_moments, pauli_sums};
pub(crate) use gates::{
    Route2q, launch_apply_cu, launch_apply_cu_phase, launch_apply_cx, launch_apply_cz,
    launch_apply_diagonal_1q, launch_apply_fused_2q, launch_apply_gate_1q, launch_apply_matrix_2q,
    launch_apply_mcu, launch_apply_mcu_phase, launch_apply_rzz, launch_apply_swap,
    launch_set_initial_state, route_2q,
};
#[cfg(test)]
pub(crate) use measure::probabilities_through_scratch;
pub(crate) use measure::{
    begin_measure_log, launch_compute_probabilities, measure_collapse, measure_logged,
    measure_prob_one, read_measure_log, reduced_density_matrix_1q,
};
pub(crate) use noise::{noise_damping, noise_kraus, noise_thermal};
pub(crate) use sample::{sample_sorted, walk_moves_fewer_bytes};
pub(crate) use tiles::{
    MIXED_TILE_GEOMETRY, launch_apply_mixed_tile, launch_apply_multi_fused_nondiag,
    launch_apply_qft_block,
};

use expect::{PAULI_AMPS_PER_THREAD, PAULI_GROUP_MASKS};
use tiles::{
    MIXED_BLOCK_LOG, MIXED_MAX_COEFS, MIXED_MAX_OPS, MIXED_TILE_Q, MULTI_FUSED_TILE_Q,
    MULTI_FUSED_TILE_SIZE, QFT_TILE_Q,
};

const BLOCK_SIZE: u32 = 256;

/// The fused 2q matrix as a by-value kernel parameter, `(re, im - re, re + im)` per
/// entry, row-major. Mirrors `struct FusedMat` in the CUDA template byte for byte.
#[repr(C)]
#[derive(Clone, Copy)]
struct FusedMatArg {
    v: [f64; 48],
}

// SAFETY: a plain `repr(C)` array of f64 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for FusedMatArg {}

/// Widest qubit list a by-value kernel parameter carries; `1 << n` bounds `n` below it.
const MAX_PARAM_QUBITS: usize = 64;

/// Sorted MCU qubit list; mirrors `struct QubitList`.
#[repr(C)]
#[derive(Clone, Copy)]
struct QubitListArg {
    q: [i32; MAX_PARAM_QUBITS],
}

// SAFETY: a plain `repr(C)` array of i32 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for QubitListArg {}

/// Diagonal `MultiFused` sub-gates, `d[4g..4g + 4]` = (d0, d1) re/im pairs and `t[g]`
/// the target; mirrors `struct DiagList`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DiagListArg {
    d: [f64; 4 * MAX_PARAM_QUBITS],
    t: [i32; MAX_PARAM_QUBITS],
}

// SAFETY: plain `repr(C)` arrays of f64 then i32 with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for DiagListArg {}

/// Tiled `MultiFused` sub-gates, `v[8g..8g + 8]` the row-major 2x2 matrix as re/im
/// pairs, the ten tile qubits at six bits each in `qubits` (sorted ascending), and each
/// gate's target as a four-bit position in that list; mirrors `struct TileGates`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TileGatesArg {
    v: [f64; 8 * MULTI_FUSED_TILE_Q],
    qubits: u64,
    targets: u64,
}

// SAFETY: a plain `repr(C)` f64 array and two u64s with the layout the kernel declares.
unsafe impl cudarc::driver::DeviceRepr for TileGatesArg {}

/// Most operators a one-qubit Kraus channel carries into `noise_site_kraus`.
pub(crate) const NOISE_MAX_KRAUS: usize = 16;

/// A one-qubit Kraus channel as a by-value kernel parameter: per operator its matrix then
/// its effect, row-major re/im pairs, and whether the normalized operator takes the
/// diagonal kernel; mirrors `struct KrausSet`.
#[repr(C)]
#[derive(Clone, Copy)]
struct KrausSetArg {
    v: [f64; 16 * NOISE_MAX_KRAUS],
    diagonal: [i32; NOISE_MAX_KRAUS],
    count: i32,
}

// SAFETY: plain `repr(C)` arrays of f64 and i32 then an i32 with the layout the kernel
// declares.
unsafe impl cudarc::driver::DeviceRepr for KrausSetArg {}

/// Packed `meta` argument of the three LUT kernels: the per-group index lists, then
/// one length and one table offset per group.
const BATCH_PHASE_META_LEN: usize = cpu_k::MAX_BATCH_PHASE_GROUPS * cpu_k::BATCH_PHASE_GROUP_SIZE
    + 2 * cpu_k::MAX_BATCH_PHASE_GROUPS;
const BATCH_RZZ_META_LEN: usize =
    2 * cpu_k::MAX_BATCH_RZZ_GROUPS * cpu_k::BATCH_RZZ_GROUP_SIZE + 2 * cpu_k::MAX_BATCH_RZZ_GROUPS;
const DIAG_BATCH_META_LEN: usize = cpu_k::MAX_DIAG_BATCH_GROUPS
    * cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP
    + 2 * cpu_k::MAX_DIAG_BATCH_GROUPS;

/// The sorted MCU qubit list as a kernel parameter.
fn qubit_list(sorted: &[u32]) -> QubitListArg {
    let mut arg = QubitListArg {
        q: [0; MAX_PARAM_QUBITS],
    };
    for (d, &q) in arg.q.iter_mut().zip(sorted) {
        *d = q as i32;
    }
    arg
}

/// CUDA C source template. [`kernel_source`] fills its `{{NAME}}` placeholders so the
/// kernel's compile-time constants track the CPU constants in
/// [`crate::backend::statevector::kernels`].
const KERNEL_SOURCE_TEMPLATE: &str = include_str!("../dense.cu");

/// Substitute the host constants into [`KERNEL_SOURCE_TEMPLATE`]. A new placeholder in
/// the template needs a matching `.replace` here.
pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE_TEMPLATE
        .replace("{{TILE_Q}}", &MULTI_FUSED_TILE_Q.to_string())
        .replace("{{TILE_SIZE}}", &MULTI_FUSED_TILE_SIZE.to_string())
        .replace("{{PAULI_GROUP_MASKS}}", &PAULI_GROUP_MASKS.to_string())
        .replace(
            "{{PAULI_AMPS_PER_THREAD}}",
            &PAULI_AMPS_PER_THREAD.to_string(),
        )
        .replace(
            "{{BP_TABLE_SIZE}}",
            &cpu_k::BATCH_PHASE_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{BP_GROUP_SIZE}}",
            &cpu_k::BATCH_PHASE_GROUP_SIZE.to_string(),
        )
        .replace(
            "{{BR_TABLE_SIZE}}",
            &cpu_k::BATCH_RZZ_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{BR_GROUP_SIZE}}",
            &cpu_k::BATCH_RZZ_GROUP_SIZE.to_string(),
        )
        .replace(
            "{{DB_TABLE_SIZE}}",
            &cpu_k::DIAG_BATCH_TABLE_SIZE.to_string(),
        )
        .replace(
            "{{DB_MAX_QUBITS}}",
            &cpu_k::DIAG_BATCH_MAX_QUBITS_PER_GROUP.to_string(),
        )
        .replace(
            "{{BP_MAX_GROUPS}}",
            &cpu_k::MAX_BATCH_PHASE_GROUPS.to_string(),
        )
        .replace("{{PARAM_QUBITS}}", &MAX_PARAM_QUBITS.to_string())
        .replace(
            "{{BR_MAX_GROUPS}}",
            &cpu_k::MAX_BATCH_RZZ_GROUPS.to_string(),
        )
        .replace(
            "{{DB_MAX_GROUPS}}",
            &cpu_k::MAX_DIAG_BATCH_GROUPS.to_string(),
        )
        .replace("{{QFT_TILE_Q}}", &QFT_TILE_Q.to_string())
        .replace("{{MIX_TILE_Q}}", &MIXED_TILE_Q.to_string())
        .replace("{{MIX_BLOCK_LOG}}", &MIXED_BLOCK_LOG.to_string())
        .replace("{{MIX_MAX_OPS}}", &MIXED_MAX_OPS.to_string())
        .replace("{{MIX_MAX_COEFS}}", &MIXED_MAX_COEFS.to_string())
        .replace("{{NOISE_MAX_KRAUS}}", &NOISE_MAX_KRAUS.to_string())
}

#[cfg(test)]
#[path = "dense_tests.rs"]
mod tests;
