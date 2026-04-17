//! Typed GPU memory allocation wrapper.
//!
//! Scaffold only: every method returns `PrismError::BackendUnsupported`. Real cudarc-backed
//! allocation lands when device kernels are wired.

use std::marker::PhantomData;

use crate::error::{PrismError, Result};

use super::device::GpuDevice;

/// RAII wrapper over a typed device allocation.
///
/// Pairs a typed view with the underlying device memory; drop releases the allocation.
#[derive(Debug)]
pub struct GpuBuffer<T> {
    len: usize,
    _phantom: PhantomData<T>,
}

impl<T> GpuBuffer<T> {
    /// Allocate `len` elements of `T` on the given device, zero-initialised.
    pub fn alloc_zeros(device: &GpuDevice, len: usize) -> Result<Self> {
        let _ = device;
        let _ = len;
        Err(PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: "alloc_zeros (scaffold only)".to_string(),
        })
    }

    /// Copy `host.len()` elements from host to device.
    pub fn copy_from_host(&mut self, host: &[T]) -> Result<()> {
        let _ = host;
        Err(PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: "copy_from_host (scaffold only)".to_string(),
        })
    }

    /// Copy `host.len()` elements from device into the provided host buffer.
    pub fn copy_to_host(&self, host: &mut [T]) -> Result<()> {
        let _ = host;
        Err(PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: "copy_to_host (scaffold only)".to_string(),
        })
    }

    /// Number of elements allocated.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the allocation is zero-length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_unsupported_in_scaffold() {
        let dev_err = GpuDevice::new(0).unwrap_err();
        assert!(matches!(dev_err, PrismError::BackendUnsupported { .. }));
    }
}
