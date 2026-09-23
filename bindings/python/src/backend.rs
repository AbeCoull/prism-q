//! `BackendKind` wrapper with ergonomic static constructors, and the held
//! stabilizer tableau a Clifford state is exported from and imported into.
//!
//! The GPU and distributed constructors exist in every build and take a
//! context object, which is what fails when the matching feature is absent.
//! Nothing here owns `MPI_Init`: the distributed context attaches to an MPI
//! mpi4py already started.

use numpy::{IntoPyArray, PyArray1};
use prism_q::backend::Backend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::{BackendKind, SpdTruncation};
use pyo3::prelude::*;

use crate::circuit::PyCircuit;
use crate::distributed::PyDistributedContext;
use crate::error::{PyPrismResult, invalid};
use crate::gpu::PyGpuContext;
use crate::numpy_util::{bool_flags, u64_words};
use crate::sim::DEFAULT_SEED;

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
    /// The tensor network with bond truncation at the peak cap, discarding
    /// at most `tolerance` of each cut's squared weight. Approximate.
    #[staticmethod]
    fn tensor_network_bounded(tolerance: f64) -> Self {
        Self(BackendKind::TensorNetworkBounded { tolerance })
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
        Self(BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Threshold { epsilon, max_terms },
        })
    }

    /// Deterministic sparse Pauli dynamics holding a fixed term count: the
    /// smallest-magnitude surplus terms are dropped whenever the weighted sum
    /// passes `max_terms`. No threshold to guess, and growth the budget caps
    /// cannot reach the engine's internal ceiling, which is where
    /// `deterministic_pauli()` dies when its epsilon prunes nothing.
    #[staticmethod]
    #[pyo3(signature = (max_terms = 65536))]
    fn deterministic_pauli_budget(max_terms: usize) -> Self {
        Self(BackendKind::DeterministicPauli {
            truncation: SpdTruncation::Budget { max_terms },
        })
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

type TableauPy<'py> = (Bound<'py, PyArray1<u64>>, Bound<'py, PyArray1<bool>>);

/// A stabilizer tableau held across calls, so a Clifford state prepared by one
/// circuit can be exported and handed to a later run.
#[pyclass(name = "StabilizerBackend", module = "prism_q")]
pub struct PyStabilizerBackend {
    inner: StabilizerBackend,
}

#[pymethods]
impl PyStabilizerBackend {
    #[new]
    #[pyo3(signature = (seed = DEFAULT_SEED))]
    fn new(seed: u64) -> Self {
        Self {
            inner: StabilizerBackend::new(seed),
        }
    }

    /// Reset to |0...0> at the circuit's width, run it, and return the
    /// classical register.
    fn run(&mut self, circuit: &PyCircuit) -> PyPrismResult<Vec<bool>> {
        self.inner
            .init(circuit.0.num_qubits, circuit.0.num_classical_bits)?;
        self.apply(circuit)
    }

    /// Run a circuit on the held state, leaving it in place, and return the
    /// classical register. The circuit must fit the held width and classical
    /// register.
    fn apply(&mut self, circuit: &PyCircuit) -> PyPrismResult<Vec<bool>> {
        let held_qubits = self.inner.num_qubits();
        if circuit.0.num_qubits > held_qubits {
            return Err(invalid(format!(
                "circuit needs {} qubits, the held tableau has {held_qubits}",
                circuit.0.num_qubits
            )));
        }
        let held_bits = self.inner.classical_results().len();
        if circuit.0.num_classical_bits > held_bits {
            return Err(invalid(format!(
                "circuit needs {} classical bits, the held register has {held_bits}",
                circuit.0.num_classical_bits
            )));
        }
        self.inner.apply_instructions(&circuit.0.instructions)?;
        Ok(self.inner.classical_results().to_vec())
    }

    /// The tableau as `(words, phases)`: a `uint64` array of `2n + 1` rows of
    /// `2 * ceil(n / 64)` bit-packed words, and a `bool` array of row signs.
    /// `import_tableau` documents the packing.
    fn export_tableau<'py>(&self, py: Python<'py>) -> PyPrismResult<TableauPy<'py>> {
        let (words, phases) = self.inner.export_tableau()?;
        Ok((words.into_pyarray(py), phases.into_pyarray(py)))
    }

    /// Start from a tableau `export_tableau` returned instead of |0...0>.
    ///
    /// With `n = num_qubits` and `nw = ceil(n / 64)`, `words` holds `2n + 1`
    /// rows of `2 * nw` words: destabilizer rows `0..n`, stabilizer rows
    /// `n..2n`, then a scratch row whose contents are ignored. A row is its
    /// `nw` X words followed by its `nw` Z words, with qubit `q` at bit
    /// `q % 64` of word `q / 64` in each half, and `phases[r]` is true when
    /// row `r` carries sign -1.
    ///
    /// Only the lengths and the diagonal pairing (destabilizer `i`
    /// anticommutes with stabilizer `i`) are checked; rows that break the rest
    /// of the commutation structure are accepted and give wrong measurement
    /// outcomes without raising. The random stream restarts from the seed
    /// passed to the constructor.
    #[pyo3(signature = (num_qubits, words, phases, num_classical_bits = 0))]
    fn import_tableau(
        &mut self,
        num_qubits: usize,
        words: &Bound<'_, PyAny>,
        phases: &Bound<'_, PyAny>,
        num_classical_bits: usize,
    ) -> PyPrismResult<()> {
        let words = u64_words(words)?;
        let phases = bool_flags(phases)?;
        self.inner
            .init_from_tableau(num_qubits, words, phases, num_classical_bits)?;
        Ok(())
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    fn classical_bits(&self) -> Vec<bool> {
        self.inner.classical_results().to_vec()
    }

    fn __repr__(&self) -> String {
        format!("StabilizerBackend(num_qubits={})", self.inner.num_qubits())
    }
}
