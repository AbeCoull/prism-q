//! `BackendKind` wrapper with ergonomic static constructors.
//!
//! GPU and distributed variants are intentionally omitted: the bindings do not
//! enable the `gpu` or `distributed` features.

use prism_q::BackendKind;
use pyo3::prelude::*;

/// Backend selection for a simulation. Construct via the static methods, e.g.
/// `BackendKind.auto()`, `BackendKind.mps(max_bond_dim=64)`.
#[pyclass(name = "BackendKind", module = "prism_q")]
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

    fn __repr__(&self) -> String {
        format!("BackendKind({:?})", self.0)
    }
}
