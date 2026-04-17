//! CUDA device wrapper. Isolates cudarc so alternative backends (wgpu, ROCm) can be substituted
//! by replacing this file.
//!
//! Scaffold only: every method returns `PrismError::BackendUnsupported`.

use crate::error::{PrismError, Result};

/// Handle to a CUDA-capable device.
///
/// Owns the `CudaDevice` (cheap to clone via `Arc` internally) and the stream used for all
/// kernel launches.
#[derive(Debug)]
pub struct GpuDevice {
    device_id: usize,
}

impl GpuDevice {
    /// Open the device with the given ordinal.
    pub fn new(device_id: usize) -> Result<Self> {
        let _ = device_id;
        Err(Self::unsupported("device initialization"))
    }

    /// Query whether any CUDA-capable GPU is available on this system.
    ///
    /// Safe to call without a device; returns `false` if detection fails for any reason.
    pub fn is_available() -> bool {
        false
    }

    /// Total VRAM on the selected device in bytes.
    pub fn vram_bytes(&self) -> Result<usize> {
        Err(Self::unsupported("vram_bytes"))
    }

    /// Maximum qubits representable as a Complex64 statevector in the available VRAM.
    ///
    /// Computed as `floor(log2(vram_bytes / 16))` — each amplitude is two f64s = 16 bytes.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        Err(Self::unsupported("max_qubits_for_statevector"))
    }

    #[allow(dead_code)]
    pub(crate) fn device_id(&self) -> usize {
        self.device_id
    }

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Self {
        Self { device_id: 0 }
    }

    fn unsupported(op: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op} (scaffold only)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_returns_unsupported() {
        assert!(matches!(
            GpuDevice::new(0).unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn is_available_false_in_scaffold() {
        assert!(!GpuDevice::is_available());
    }
}
