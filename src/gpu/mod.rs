//! Shared GPU execution resource.
//!
//! GPU support in PRISM-Q is not a standalone backend. It is an execution context that
//! existing CPU backends (Statevector first, MPS and TensorNetwork later) can opt into for
//! their hot operations. Each backend that wants GPU acceleration holds an
//! `Option<Arc<GpuContext>>` (or owns a `GpuState`) and routes through this module's kernel
//! namespaces when the context is present.
//!
//! # Module layout
//!
//! - [`device`] — cudarc device wrapper, availability checks, VRAM queries
//! - [`memory`] — [`GpuBuffer`][memory::GpuBuffer] RAII wrapper over device allocations
//! - (future) `kernels::dense` — statevector 1q/2q/Cu/Mcu/measurement/fused kernels
//! - (future) `kernels::svd` — MPS SVD via cuSolver
//! - (future) `kernels::tn` — TensorNetwork contraction via cuBLAS
//!
//! # Scaffold status
//!
//! All structures and methods in this module are stubs that return
//! `PrismError::BackendUnsupported`. Real cudarc integration arrives with the first kernel
//! milestone.

pub mod device;
pub mod memory;

use std::sync::Arc;

use crate::error::{PrismError, Result};

use self::device::GpuDevice;
use self::memory::GpuBuffer;

/// Shared GPU execution context.
///
/// Holds the device handle and (eventually) the compiled kernel module registry. Cheap to
/// clone via `Arc`. Pass by `Arc<GpuContext>` so multiple backends or multiple simulations
/// can share one device initialisation.
#[derive(Debug)]
pub struct GpuContext {
    device: Arc<GpuDevice>,
}

impl GpuContext {
    /// Initialise the context for the given CUDA device ordinal.
    ///
    /// Loads any required kernel modules (none yet in the scaffold).
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

    #[allow(dead_code)]
    pub(crate) fn device(&self) -> &GpuDevice {
        &self.device
    }
}

/// Per-simulation device-resident state.
///
/// Owns a `GpuBuffer<f64>` holding `2 * 2^num_qubits` f64s (interleaved re/im, matching
/// `Complex64` layout and CUDA's `double2` alignment). Tracks `pending_norm` the same way the
/// CPU statevector backend does — measurement collapse accumulates into this scalar and the
/// final scale is applied at `export_statevector` or `probabilities` time.
#[derive(Debug)]
#[allow(dead_code)]
pub struct GpuState {
    context: Arc<GpuContext>,
    buffer: GpuBuffer<f64>,
    num_qubits: usize,
    pending_norm: f64,
}

impl GpuState {
    /// Allocate a fresh |0…0⟩ state on the device bound to `context`.
    pub fn new(context: Arc<GpuContext>, num_qubits: usize) -> Result<Self> {
        let _ = &context;
        let _ = num_qubits;
        Err(PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: "state allocation (scaffold only)".to_string(),
        })
    }

    /// Number of qubits the buffer is sized for.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Multiplicative norm correction deferred from measurement collapse.
    pub fn pending_norm(&self) -> f64 {
        self.pending_norm
    }

    #[allow(dead_code)]
    pub(crate) fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    #[allow(dead_code)]
    pub(crate) fn buffer(&self) -> &GpuBuffer<f64> {
        &self.buffer
    }

    #[allow(dead_code)]
    pub(crate) fn buffer_mut(&mut self) -> &mut GpuBuffer<f64> {
        &mut self.buffer
    }

    #[allow(dead_code)]
    pub(crate) fn set_pending_norm(&mut self, norm: f64) {
        self.pending_norm = norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_new_returns_unsupported() {
        assert!(matches!(
            GpuContext::new(0).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn context_is_available_false_in_scaffold() {
        assert!(!GpuContext::is_available());
    }
}
