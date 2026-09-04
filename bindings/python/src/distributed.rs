//! Opaque handle to an MPI world this process did not start.
//!
//! The class is present in every build; without the `distributed-mpi` feature
//! the constructor raises, which is the shape [`crate::gpu`] uses. Nothing here
//! calls `MPI_Init` or `MPI_Finalize`. mpi4py owns MPI's lifetime, and a handle
//! whose refcount drop ran `MPI_Finalize` would make every later MPI call in
//! the interpreter erroneous.

use pyo3::prelude::*;

use crate::error::PyPrismError;
use crate::error::PyPrismResult;

#[cfg(not(feature = "distributed-mpi"))]
pub(crate) fn unsupported() -> PyPrismError {
    PyPrismError(prism_q::PrismError::IncompatibleBackend {
        backend: "distributed".into(),
        reason: "this build was built without MPI support; rebuild the bindings from source \
                 with the `distributed-mpi` feature, against the same MPI implementation as \
                 mpi4py and the launcher"
            .into(),
    })
}

#[cfg(feature = "distributed-mpi")]
fn not_initialized() -> PyPrismError {
    PyPrismError(prism_q::PrismError::IncompatibleBackend {
        backend: "distributed".into(),
        reason: "MPI is not initialized; `import mpi4py.MPI` before constructing a \
                 DistributedContext, and launch the script under mpiexec"
            .into(),
    })
}

/// Handle to the MPI world communicator, passed to
/// `BackendKind.statevector_distributed`.
///
/// Construction attaches to an MPI that is already running and reports the rank
/// count; it never starts or stops MPI. Every rank runs the same script and
/// enters every collective, so a run must be reached by all of them: branching
/// on `rank` before a simulation call deadlocks the job.
#[pyclass(name = "DistributedContext", module = "prism_q", frozen)]
pub struct PyDistributedContext {
    #[cfg(feature = "distributed-mpi")]
    pub(crate) inner: std::sync::Arc<prism_q::distributed::DistributedContext>,
}

#[pymethods]
impl PyDistributedContext {
    /// Attach to the world communicator of an MPI that is already running.
    ///
    /// Raises when this build has no MPI support, and when MPI has not been
    /// initialized. Import `mpi4py.MPI` first: it calls `MPI_Init_thread` at
    /// import and registers `MPI_Finalize` at interpreter exit.
    #[new]
    fn new() -> PyPrismResult<Self> {
        #[cfg(feature = "distributed-mpi")]
        {
            let inner = prism_q::distributed::DistributedContext::attached_world()
                .ok_or_else(not_initialized)?;
            Ok(Self { inner })
        }
        #[cfg(not(feature = "distributed-mpi"))]
        {
            Err(unsupported())
        }
    }

    /// Whether this build has the `distributed-mpi` feature, independent of MPI
    /// running. Distinguishes a wheel without MPI support from a script that
    /// forgot mpi4py.
    #[staticmethod]
    fn is_supported() -> bool {
        cfg!(feature = "distributed-mpi")
    }

    #[getter]
    fn rank(&self) -> usize {
        #[cfg(feature = "distributed-mpi")]
        {
            self.inner.rank()
        }
        #[cfg(not(feature = "distributed-mpi"))]
        {
            unreachable!("no instance exists without the feature")
        }
    }

    #[getter]
    fn size(&self) -> usize {
        #[cfg(feature = "distributed-mpi")]
        {
            self.inner.size()
        }
        #[cfg(not(feature = "distributed-mpi"))]
        {
            unreachable!("no instance exists without the feature")
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DistributedContext(rank={}, size={})",
            self.rank(),
            self.size()
        )
    }
}
