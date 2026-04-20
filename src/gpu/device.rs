//! CUDA device wrapper. Isolates cudarc so alternative backends (wgpu, ROCm) can be substituted
//! by replacing this file.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::error::{PrismError, Result};

use super::kernels::{kernel_source, KERNEL_NAMES};

/// Handle to a CUDA-capable device.
///
/// Owns the CUDA context, default stream, and compiled PTX module. The `Stub` variant exists
/// so unit tests can exercise the `with_gpu()` builder path without CUDA available.
#[derive(Debug)]
pub struct GpuDevice {
    inner: DeviceInner,
}

#[derive(Debug)]
#[allow(dead_code)]
enum DeviceInner {
    Real {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        functions: HashMap<&'static str, CudaFunction>,
    },
    Stub,
}

impl GpuDevice {
    /// Open the device with the given ordinal and compile the kernel module.
    ///
    /// PTX is compiled targeting the device's compute capability so the running NVIDIA
    /// driver can load it regardless of the toolkit NVRTC version.
    pub fn new(device_id: usize) -> Result<Self> {
        let context = CudaContext::new(device_id).map_err(|e| Self::driver_err("init", e))?;
        let stream = context.default_stream();
        let arch = detect_arch(&context).unwrap_or("compute_60");
        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let source = kernel_source();
        let ptx =
            compile_ptx_with_opts(&source, opts).map_err(|e| PrismError::BackendUnsupported {
                backend: "gpu".to_string(),
                operation: format!("PTX compilation (arch={arch}): {e}"),
            })?;
        let module = context
            .load_module(ptx)
            .map_err(|e| Self::driver_err("load_module", e))?;
        // Pre-resolve every kernel once, to amortise driver lookups away from the gate
        // dispatch hot path.
        let mut functions = HashMap::with_capacity(KERNEL_NAMES.len());
        for &name in KERNEL_NAMES {
            let func = module
                .load_function(name)
                .map_err(|e| Self::driver_err(&format!("load_function `{name}`"), e))?;
            functions.insert(name, func);
        }
        Ok(Self {
            inner: DeviceInner::Real {
                context,
                stream,
                module,
                functions,
            },
        })
    }

    /// Query whether any CUDA-capable GPU is available on this system.
    ///
    /// Safe to call without a device; returns `false` if detection fails for any reason.
    pub fn is_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// Total VRAM on the selected device in bytes.
    pub fn vram_bytes(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context
                .total_mem()
                .map_err(|e| Self::driver_err("vram_bytes", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_bytes")),
        }
    }

    /// Free VRAM currently available on the selected device in bytes.
    ///
    /// Reflects allocations by all processes sharing the device, including the
    /// current process's own outstanding `GpuBuffer`s. Useful for deciding
    /// whether a pending statevector allocation is likely to fit.
    pub fn vram_available(&self) -> Result<usize> {
        match &self.inner {
            DeviceInner::Real { context, .. } => context
                .mem_get_info()
                .map(|(free, _total)| free)
                .map_err(|e| Self::driver_err("vram_available", e)),
            DeviceInner::Stub => Err(Self::stub_unsupported("vram_available")),
        }
    }

    /// Maximum qubits representable as a Complex64 statevector in the total VRAM.
    ///
    /// Computed as `floor(log2(vram_bytes / 16))` — each amplitude is two f64s = 16 bytes.
    pub fn max_qubits_for_statevector(&self) -> Result<usize> {
        let bytes = self.vram_bytes()?;
        let elements = bytes / 16;
        if elements == 0 {
            return Ok(0);
        }
        Ok(63 - elements.leading_zeros() as usize)
    }

    #[cfg(test)]
    pub(crate) fn stub_for_tests() -> Self {
        Self {
            inner: DeviceInner::Stub,
        }
    }

    pub(crate) fn stream(&self) -> Result<&Arc<CudaStream>> {
        match &self.inner {
            DeviceInner::Real { stream, .. } => Ok(stream),
            DeviceInner::Stub => Err(Self::stub_unsupported("stream access")),
        }
    }

    pub(crate) fn function(&self, name: &str) -> Result<CudaFunction> {
        match &self.inner {
            DeviceInner::Real { functions, .. } => {
                functions
                    .get(name)
                    .cloned()
                    .ok_or_else(|| PrismError::BackendUnsupported {
                        backend: "gpu".to_string(),
                        operation: format!("unknown kernel `{name}` (not in KERNEL_NAMES)"),
                    })
            }
            DeviceInner::Stub => Err(Self::stub_unsupported(&format!("function `{name}`"))),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn is_stub(&self) -> bool {
        matches!(self.inner, DeviceInner::Stub)
    }

    fn driver_err(op: &str, err: impl std::fmt::Display) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op}: {err}"),
        }
    }

    fn stub_unsupported(op: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: "gpu".to_string(),
            operation: format!("{op} (stub device)"),
        }
    }
}

fn detect_arch(context: &Arc<CudaContext>) -> Option<&'static str> {
    let (major, minor) = context.compute_capability().ok()?;
    match (major, minor) {
        (6, 0) => Some("compute_60"),
        (6, 1) => Some("compute_61"),
        (6, 2) => Some("compute_62"),
        (7, 0) => Some("compute_70"),
        (7, 2) => Some("compute_72"),
        (7, 5) => Some("compute_75"),
        (8, 0) => Some("compute_80"),
        (8, 6) => Some("compute_86"),
        (8, 7) => Some("compute_87"),
        (8, 9) => Some("compute_89"),
        (9, 0) => Some("compute_90"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reports_as_stub() {
        let dev = GpuDevice::stub_for_tests();
        assert!(dev.is_stub());
    }

    #[test]
    fn stub_stream_returns_unsupported() {
        let dev = GpuDevice::stub_for_tests();
        assert!(matches!(
            dev.stream().unwrap_err(),
            PrismError::BackendUnsupported { .. }
        ));
    }
}
