//! Opaque handle to a CUDA execution context.
//!
//! The class is present in every build. Without the `gpu` feature the
//! constructor raises `PrismError` instead of the class being absent, so the
//! Python API surface does not change between wheels. Construction compiles the
//! kernel module, so it releases the GIL like the simulation terminals do.

use pyo3::prelude::*;

#[cfg(not(feature = "gpu"))]
use crate::error::PyPrismError;
use crate::error::PyPrismResult;

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

#[pymethods]
impl PyGpuContext {
    #[new]
    #[pyo3(signature = (device_id = 0))]
    fn new(py: Python<'_>, device_id: usize) -> PyPrismResult<Self> {
        #[cfg(feature = "gpu")]
        {
            let inner = py.detach(|| prism_q::gpu::GpuContext::new(device_id))?;
            Ok(Self { device_id, inner })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (py, device_id);
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
