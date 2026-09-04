//! `BackendKind` wrapper with ergonomic static constructors.
//!
//! The GPU and distributed constructors exist in every build and take a
//! context object, which is what fails when the matching feature is absent.
//! Nothing here owns `MPI_Init`: the distributed context attaches to an MPI
//! mpi4py already started.

use prism_q::BackendKind;
use pyo3::prelude::*;

use crate::distributed::PyDistributedContext;
use crate::error::PyPrismResult;
use crate::gpu::PyGpuContext;

/// Backend selection for a simulation. Construct via the static methods, e.g.
/// `BackendKind.auto()`, `BackendKind.mps(max_bond_dim=64)`.
#[pyclass(name = "BackendKind", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub struct PyBackendKind(pub BackendKind);

#[pymethods]
impl PyBackendKind {
    #[staticmethod]
    fn auto() -> Self {
        Self(BackendKind::Auto)
    }
    #[staticmethod]
    fn statevector() -> Self {
        Self(BackendKind::Statevector)
    }
    #[staticmethod]
    fn stabilizer() -> Self {
        Self(BackendKind::Stabilizer)
    }
    #[staticmethod]
    fn sparse() -> Self {
        Self(BackendKind::Sparse)
    }
    #[staticmethod]
    fn product_state() -> Self {
        Self(BackendKind::ProductState)
    }
    #[staticmethod]
    fn tensor_network() -> Self {
        Self(BackendKind::TensorNetwork)
    }
    #[staticmethod]
    fn factored() -> Self {
        Self(BackendKind::Factored)
    }
    #[staticmethod]
    fn factored_stabilizer() -> Self {
        Self(BackendKind::FactoredStabilizer)
    }
    #[staticmethod]
    fn stabilizer_rank() -> Self {
        Self(BackendKind::StabilizerRank)
    }
    /// Exact mixed-state evolution over `4^n` amplitudes. Never selected by
    /// `auto()`; the qubit ceiling is roughly half the statevector cap.
    #[staticmethod]
    fn density_matrix() -> Self {
        Self(BackendKind::DensityMatrix)
    }

    #[staticmethod]
    #[pyo3(signature = (max_bond_dim = 256))]
    fn mps(max_bond_dim: usize) -> Self {
        Self(BackendKind::Mps { max_bond_dim })
    }

    #[staticmethod]
    #[pyo3(signature = (num_samples = 1000))]
    fn stochastic_pauli(num_samples: usize) -> Self {
        Self(BackendKind::StochasticPauli { num_samples })
    }

    #[staticmethod]
    #[pyo3(signature = (epsilon = 0.0, max_terms = 65536))]
    fn deterministic_pauli(epsilon: f64, max_terms: usize) -> Self {
        Self(BackendKind::DeterministicPauli { epsilon, max_terms })
    }

    /// Structure-driven dispatch with the supplied device opted in. Blocks that
    /// clear the family crossover with VRAM to spare run on the device; every
    /// other block takes the path `auto()` would.
    #[staticmethod]
    fn auto_gpu(context: &PyGpuContext) -> PyPrismResult<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self(BackendKind::AutoGpu {
                context: context.inner.clone(),
            }))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = context;
            Err(crate::gpu::unsupported())
        }
    }

    /// Distributed statevector sharded across the ranks of `context`.
    ///
    /// A world of one rank raises. MPI-2 and later make a singleton `MPI_Init`
    /// succeed, so a script launched without `mpiexec` would otherwise get a
    /// correct answer from one rank at single-host speed and no signal that
    /// nothing was distributed. Pass `allow_single_rank=True` when that is
    /// deliberate.
    ///
    /// Every rank must reach this call with the same circuit and seed;
    /// disagreement is reported as an error on every rank rather than left to
    /// hang in a collective.
    #[staticmethod]
    #[pyo3(signature = (context, allow_single_rank = false))]
    fn statevector_distributed(
        context: &PyDistributedContext,
        allow_single_rank: bool,
    ) -> PyPrismResult<Self> {
        #[cfg(feature = "distributed-mpi")]
        {
            if context.inner.size() == 1 && !allow_single_rank {
                return Err(crate::error::PyPrismError(
                    prism_q::PrismError::IncompatibleBackend {
                        backend: "StatevectorDistributed".into(),
                        reason: "the MPI world has one rank, so this run would be identical to \
                                 statevector() and communicate with nobody; launch under \
                                 mpiexec, or pass allow_single_rank=True"
                            .into(),
                    },
                ));
            }
            Ok(Self(BackendKind::StatevectorDistributed {
                context: context.inner.clone(),
            }))
        }
        #[cfg(not(feature = "distributed-mpi"))]
        {
            let _ = (context, allow_single_rank);
            Err(crate::distributed::unsupported())
        }
    }

    /// Statevector on the supplied device. Circuits below the crossover
    /// (`PRISM_GPU_MIN_QUBITS`, default 14) run on the host path instead.
    #[staticmethod]
    fn statevector_gpu(context: &PyGpuContext) -> PyPrismResult<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self(BackendKind::StatevectorGpu {
                context: context.inner.clone(),
            }))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = context;
            Err(crate::gpu::unsupported())
        }
    }

    /// Stabilizer tableau on the supplied device. Clifford-only circuits, and
    /// below the crossover (`PRISM_STABILIZER_GPU_MIN_QUBITS`, default 100000)
    /// the host tableau runs instead.
    #[staticmethod]
    fn stabilizer_gpu(context: &PyGpuContext) -> PyPrismResult<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self(BackendKind::StabilizerGpu {
                context: context.inner.clone(),
            }))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = context;
            Err(crate::gpu::unsupported())
        }
    }

    /// Exact mixed state held on the supplied device. Explicit only, with no
    /// host fallback: the `4^n` buffer is budgeted against free device memory
    /// before allocation and a width that does not fit raises `PrismError`.
    #[staticmethod]
    fn density_matrix_gpu(context: &PyGpuContext) -> PyPrismResult<Self> {
        #[cfg(feature = "gpu")]
        {
            Ok(Self(BackendKind::DensityMatrixGpu {
                context: context.inner.clone(),
            }))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = context;
            Err(crate::gpu::unsupported())
        }
    }

    fn __repr__(&self) -> String {
        format!("BackendKind({:?})", self.0)
    }
}
