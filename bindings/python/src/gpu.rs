//! Opaque handle to a CUDA execution context. The class exists in every build;
//! without the `gpu` feature its constructor raises `PrismError`, so the Python API
//! does not change between wheels.

use pyo3::prelude::*;

use crate::error::{PyPrismError, PyPrismResult};

/// Handle to one CUDA device and its compiled kernel module, passed to the
/// `BackendKind` GPU constructors. Construction is the point where a missing or
/// unusable device is reported; reuse one handle across simulations to compile
/// kernels once.
#[pyclass(name = "GpuContext", module = "prism_q", frozen)]
pub struct PyGpuContext {
    device_id: usize,
    #[cfg(feature = "gpu")]
    pub(crate) inner: std::sync::Arc<prism_q::gpu::GpuContext>,
}

#[cfg(not(feature = "gpu"))]
pub(crate) fn unsupported() -> PyPrismError {
    PyPrismError(prism_q::PrismError::IncompatibleBackend {
        backend: "gpu".into(),
        reason: "this build was built without GPU support; rebuild the bindings with the \
                 `gpu` feature"
            .into(),
    })
}

/// Point a failure at the `cuda12` extra when NVRTC is among what is missing.
#[cfg(feature = "gpu")]
fn with_nvrtc_hint(err: prism_q::PrismError) -> PyErr {
    let err = match err {
        prism_q::PrismError::IncompatibleBackend { backend, reason }
            if !prism_q::gpu::nvrtc_available() =>
        {
            prism_q::PrismError::IncompatibleBackend {
                backend,
                reason: format!("{reason}. `pip install \"prism-q[cuda12]\"` provides NVRTC"),
            }
        }
        other => other,
    };
    PyPrismError(err).into()
}

#[pymethods]
impl PyGpuContext {
    #[new]
    #[pyo3(signature = (device_id = 0))]
    fn new(py: Python<'_>, device_id: usize) -> PyResult<Self> {
        #[cfg(feature = "gpu")]
        {
            py.import("prism_q._cuda")?.call_method0("preload_nvrtc")?;
            let inner = py
                .detach(|| prism_q::gpu::GpuContext::new(device_id))
                .map_err(with_nvrtc_hint)?;
            Ok(Self { device_id, inner })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (py, device_id);
            Err(unsupported().into())
        }
    }

    /// Product name of the device, as the driver reports it.
    #[getter]
    fn device_name(&self) -> PyPrismResult<String> {
        #[cfg(feature = "gpu")]
        {
            Ok(self.inner.device_name()?)
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(unsupported())
        }
    }

    /// Whether this build has the `gpu` feature, independent of any device
    /// being present. Distinguishes a wheel without CUDA support from a host
    /// without a card.
    #[staticmethod]
    fn is_supported() -> bool {
        cfg!(feature = "gpu")
    }

    /// Whether this build has the `gpu` feature and a usable CUDA device.
    #[staticmethod]
    fn is_available(py: Python<'_>) -> bool {
        #[cfg(feature = "gpu")]
        {
            py.detach(prism_q::gpu::is_available)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = py;
            false
        }
    }

    fn __repr__(&self) -> String {
        format!("GpuContext(device_id={})", self.device_id)
    }
}

/// Result of `gpu_info`: `device` is set when the device opened, `reason` when not.
#[pyclass(name = "GpuInfo", module = "prism_q", frozen, get_all)]
pub struct PyGpuInfo {
    available: bool,
    device: Option<String>,
    reason: Option<String>,
}

#[pymethods]
impl PyGpuInfo {
    fn __repr__(&self) -> String {
        match (&self.device, &self.reason) {
            (Some(device), _) => format!("GpuInfo(available=True, device={device:?})"),
            (None, Some(reason)) => format!("GpuInfo(available=False, reason={reason:?})"),
            (None, None) => "GpuInfo(available=False)".to_string(),
        }
    }
}

/// Open CUDA device `device_id` and report its name, or why it cannot be used.
///
/// Opening compiles the kernels through NVRTC unless a PTX cache from an earlier run
/// matches, so a missing NVRTC shows here rather than at the first GPU run.
#[pyfunction]
#[pyo3(signature = (device_id = 0))]
pub fn gpu_info(py: Python<'_>, device_id: usize) -> PyResult<PyGpuInfo> {
    Ok(match PyGpuContext::new(py, device_id) {
        Ok(context) => PyGpuInfo {
            available: true,
            device: Some(context.device_name()?),
            reason: None,
        },
        Err(err) => PyGpuInfo {
            available: false,
            device: None,
            reason: Some(err.value(py).str()?.to_string()),
        },
    })
}
