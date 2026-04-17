//! Typed GPU memory allocation wrapper.
//!
//! `GpuBuffer<T>` owns a [`cudarc::driver::CudaSlice<T>`] and exposes a minimal htod/dtoh
//! interface. Additional element types can be supported by extending the `T: DeviceRepr`
//! bound as needed; today only `f64` is exercised.

use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};

use crate::error::{PrismError, Result};

use super::device::GpuDevice;

/// RAII wrapper over a typed device allocation.
pub struct GpuBuffer<T: DeviceRepr> {
    slice: CudaSlice<T>,
}

impl<T: DeviceRepr + ValidAsZeroBits> GpuBuffer<T> {
    /// Allocate `len` elements of `T` on the given device, zero-initialised.
    pub fn alloc_zeros(device: &GpuDevice, len: usize) -> Result<Self> {
        let stream = device.stream()?;
        let slice = stream
            .alloc_zeros::<T>(len)
            .map_err(|e| driver_err("alloc_zeros", e))?;
        Ok(Self { slice })
    }
}

impl<T: DeviceRepr> GpuBuffer<T> {
    /// Copy `host.len()` elements from host to device (in-place).
    pub fn copy_from_host(&mut self, device: &GpuDevice, host: &[T]) -> Result<()> {
        let stream = device.stream()?;
        stream
            .memcpy_htod(host, &mut self.slice)
            .map_err(|e| driver_err("copy_from_host", e))
    }

    /// Copy `host.len()` elements from device into the provided host buffer.
    pub fn copy_to_host(&self, device: &GpuDevice, host: &mut [T]) -> Result<()> {
        let stream = device.stream()?;
        stream
            .memcpy_dtoh(&self.slice, host)
            .map_err(|e| driver_err("copy_to_host", e))
    }
}

impl<T: DeviceRepr> GpuBuffer<T> {
    /// Number of elements allocated.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the allocation is zero-length.
    pub fn is_empty(&self) -> bool {
        self.slice.len() == 0
    }

    pub(crate) fn raw(&self) -> &CudaSlice<T> {
        &self.slice
    }

    pub(crate) fn raw_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

impl<T: DeviceRepr> std::fmt::Debug for GpuBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("len", &self.slice.len())
            .finish()
    }
}

fn driver_err(op: &str, err: impl std::fmt::Display) -> PrismError {
    PrismError::BackendUnsupported {
        backend: "gpu".to_string(),
        operation: format!("{op}: {err}"),
    }
}
