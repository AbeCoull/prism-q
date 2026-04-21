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
//! The first kernel landed here is `stab_set_initial_tableau`. Subsequent milestones
//! add the Clifford gate kernels, `stab_rowmul_words`, and measurement kernels.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::error::{PrismError, Result};

use super::super::{GpuContext, GpuTableau};

const BLOCK_SIZE: u32 = 128;

/// Stabilizer CUDA C source. Returned by `kernel_source()` and concatenated into the
/// combined PTX module alongside the dense kernels. No template substitutions needed:
/// all tableau shapes are passed as kernel arguments rather than compile-time constants.
const KERNEL_SOURCE: &str = r#"
// ============================================================================
// Stabilizer tableau kernels
// ============================================================================
//
// CPU reference: src/backend/stabilizer/mod.rs (init, layout) and
// src/backend/stabilizer/kernels/simd.rs (rowmul g-function).
//
// xz is laid out as (2n+1) rows × 2*num_words u64s per row, with X-bits in the
// first num_words words of each row and Z-bits in the second num_words words.
// phase is 2n+1 bytes.

extern "C" __global__ void stab_set_initial_tableau(
    unsigned long long *xz,
    int num_qubits,
    int num_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_qubits) return;

    int word_idx = i / 64;
    unsigned long long bit = 1ULL << (i % 64);
    int stride = 2 * num_words;

    // Destabilizer row i: X-bit i set in the X-half.
    xz[i * stride + word_idx] = bit;

    // Stabilizer row (num_qubits + i): Z-bit i set in the Z-half.
    xz[(num_qubits + i) * stride + num_words + word_idx] = bit;
}
"#;

/// Return the stabilizer CUDA C source for concatenation into the shared PTX module.
pub(crate) fn kernel_source() -> String {
    KERNEL_SOURCE.to_string()
}

fn launch_err(op: &str, err: impl std::fmt::Display) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {err}"),
    }
}

/// Initialise a freshly-allocated `GpuTableau` to the identity tableau: destabilizer
/// rows are X_i, stabilizer rows are Z_i, scratch row is all zero, phase is all zero.
///
/// Assumes `xz` and `phase` were allocated via `GpuBuffer::alloc_zeros` (so everything
/// else is already zero); this kernel only writes the identity bits.
pub(crate) fn launch_set_initial_tableau(ctx: &GpuContext, tableau: &mut GpuTableau) -> Result<()> {
    let device = ctx.device();
    let stream = device.stream()?;
    let func = device.function("stab_set_initial_tableau")?;

    let n = tableau.num_qubits() as i32;
    let nw = tableau.num_words() as i32;
    if n <= 0 {
        return Ok(());
    }
    let blocks = (n as u32).div_ceil(BLOCK_SIZE).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };

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
