//! Shared GPU execution resource.
//!
//! GPU support in PRISM-Q is not a standalone backend. It is an execution context that CPU
//! backends (Statevector first, MPS and TensorNetwork later) opt into for their hot
//! operations. Each backend that wants GPU acceleration holds an `Option<Arc<GpuContext>>`
//! and routes through this module's kernel namespaces when the context is present.
//!
//! # Module layout
//!
//! - [`device`] — cudarc device wrapper, availability checks, VRAM queries
//! - [`memory`] — [`GpuBuffer`] RAII wrapper over device allocations
//! - `kernels::dense` — statevector kernels (CUDA C source + launch helpers)
//!
//! # Device state layout
//!
//! The statevector lives on device as a `CudaSlice<f64>` of length `2 * 2^n` holding
//! interleaved (re, im) pairs. This matches `num_complex::Complex64` and CUDA's `double2`
//! builtin, allowing zero-cost reinterpretation at kernel boundaries.

pub mod device;
pub(crate) mod kernels;
pub mod memory;

use std::sync::Arc;

use num_complex::Complex64;

use crate::error::Result;

pub use self::device::GpuDevice;
pub use self::memory::GpuBuffer;

/// Shared GPU execution context.
///
/// Holds the device handle and compiled kernel module. Cheap to clone via `Arc`. Pass by
/// `Arc<GpuContext>` so multiple backends or multiple simulations can share one device
/// initialisation.
#[derive(Debug)]
pub struct GpuContext {
    device: Arc<GpuDevice>,
}

impl GpuContext {
    /// Initialise the context for the given CUDA device ordinal.
    ///
    /// Compiles the kernel module at construction. Subsequent calls reuse the cached PTX.
    pub fn new(device_id: usize) -> Result<Arc<Self>> {
        let device = Arc::new(GpuDevice::new(device_id)?);
        Ok(Arc::new(Self { device }))
    }

    /// Whether a CUDA device is present and usable.
    pub fn is_available() -> bool {
        GpuDevice::is_available()
    }

    /// Total VRAM on the device bound to this context.
    pub fn vram_bytes(&self) -> Result<usize> {
        self.device.vram_bytes()
    }

    /// Maximum qubit count for a dense Complex64 statevector on this device.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        self.device.max_qubits_for_statevector()
    }

    pub(crate) fn device(&self) -> &GpuDevice {
        &self.device
    }

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Arc<Self> {
        Arc::new(Self {
            device: Arc::new(GpuDevice::stub_for_tests()),
        })
    }
}

/// Per-simulation device-resident state.
///
/// Owns a `GpuBuffer<f64>` holding `2 * 2^num_qubits` f64s (interleaved re/im). Tracks
/// `pending_norm` the same way the CPU statevector backend does — measurement collapse
/// accumulates into this scalar and the final scale is applied at `export_statevector` or
/// `probabilities` time.
#[derive(Debug)]
pub struct GpuState {
    context: Arc<GpuContext>,
    buffer: GpuBuffer<f64>,
    num_qubits: usize,
    pending_norm: f64,
    /// Device-side scratch buffer for `probabilities()` output, reused across calls so
    /// shot-sampling workflows don't re-allocate `2^n` f64s on every read.
    probs_scratch: std::cell::RefCell<Option<GpuBuffer<f64>>>,
}

impl GpuState {
    /// Allocate a fresh |0…0⟩ state on the device bound to `context`.
    pub fn new(context: Arc<GpuContext>, num_qubits: usize) -> Result<Self> {
        let len = 2usize.checked_shl(num_qubits as u32).ok_or_else(|| {
            crate::error::PrismError::InvalidParameter {
                message: format!("num_qubits={num_qubits} overflows addressable memory"),
            }
        })?;
        let buffer = GpuBuffer::<f64>::alloc_zeros(context.device(), len)?;
        let mut state = Self {
            context: context.clone(),
            buffer,
            num_qubits,
            pending_norm: 1.0,
            probs_scratch: std::cell::RefCell::new(None),
        };
        kernels::dense::launch_set_initial_state(&context, &mut state)?;
        Ok(state)
    }

    /// Number of qubits the buffer is sized for.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Multiplicative norm correction deferred from measurement collapse.
    pub fn pending_norm(&self) -> f64 {
        self.pending_norm
    }

    /// Read the raw (unnormalised) amplitude buffer back to host, as interleaved f64 pairs.
    pub fn copy_to_host_raw(&self) -> Result<Vec<f64>> {
        let mut host = vec![0.0_f64; self.buffer.len()];
        self.buffer.copy_to_host(self.context.device(), &mut host)?;
        Ok(host)
    }

    /// Read the amplitude buffer as `Vec<Complex64>` with the deferred `pending_norm`
    /// already applied.
    pub fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let raw = self.copy_to_host_raw()?;
        let norm = self.pending_norm;
        let out = raw
            .chunks_exact(2)
            .map(|p| Complex64::new(p[0] * norm, p[1] * norm))
            .collect();
        Ok(out)
    }

    /// Compute per-basis-state probabilities with `pending_norm²` scaling, via a GPU
    /// reduction kernel. Only `2^n` f64s cross PCIe (vs `2·2^n` for raw amplitudes).
    pub fn probabilities(&self) -> Result<Vec<f64>> {
        kernels::dense::launch_compute_probabilities(&self.context, self)
    }

    pub(crate) fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer<f64> {
        &self.buffer
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut GpuBuffer<f64> {
        &mut self.buffer
    }

    pub(crate) fn set_pending_norm(&mut self, norm: f64) {
        self.pending_norm = norm;
    }

    /// Access (and lazily allocate) the cached device-side probabilities buffer. Returned as
    /// a `RefMut` so the caller can pass it to a kernel launch for the duration of the call.
    pub(crate) fn probs_scratch(&self) -> std::cell::RefMut<'_, Option<GpuBuffer<f64>>> {
        self.probs_scratch.borrow_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::PrismError;

    #[test]
    fn stub_context_reports_available_false() {
        // Even if a device is present, this test only uses the stub path.
        let ctx = GpuContext::stub_for_tests();
        assert!(ctx.device().is_stub());
    }

    #[test]
    fn state_new_on_stub_returns_unsupported() {
        let ctx = GpuContext::stub_for_tests();
        assert!(matches!(
            GpuState::new(ctx, 4).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }
}
