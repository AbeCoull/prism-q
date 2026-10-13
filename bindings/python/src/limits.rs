//! Resource limits for one simulation or for the process.

use prism_q::Limits;
use pyo3::prelude::*;

/// Memory budget and per-cap overrides.
///
/// A cap left as `None` derives from `memory_budget_bytes`, which defaults to half of
/// physical memory. `Simulation.limits(l)` applies to one request and
/// `set_default_limits(l)` to every later call in the process; a request value beats
/// the process default, which beats the `PRISM_MAX_*` environment variables. Setting
/// `memory_budget_bytes` re-derives every memory cap below that layer.
#[pyclass(name = "Limits", module = "prism_q", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyLimits(pub(crate) Limits);

macro_rules! getters {
    ($($name:ident: $ty:ty),* $(,)?) => {
        #[pymethods]
        impl PyLimits {
            #[new]
            #[pyo3(signature = (*, $($name = None),*))]
            #[allow(clippy::too_many_arguments)]
            fn new($($name: Option<$ty>),*) -> Self {
                let mut limits = Limits::new();
                $(limits.$name = $name;)*
                Self(limits)
            }

            $(
                #[getter]
                fn $name(&self) -> Option<$ty> {
                    self.0.$name
                }
            )*

            fn __repr__(&self) -> String {
                let set: Vec<String> = [$((stringify!($name), self.0.$name.map(|v| v.to_string()))),*]
                    .into_iter()
                    .filter_map(|(name, value)| value.map(|v| format!("{name}={v}")))
                    .collect();
                format!("Limits({})", set.join(", "))
            }
        }
    };
}

getters! {
    memory_budget_bytes: u64,
    max_statevector_qubits: usize,
    max_density_matrix_qubits: usize,
    max_probability_qubits: usize,
    max_export_qubits: usize,
    max_sparse_qubits: usize,
    max_factored_merge_qubits: usize,
    max_stabilizer_cluster_qubits: usize,
    max_mps_workspace_qubits: usize,
    max_tn_peak_qubits: usize,
    max_tn_slices: usize,
    auto_mps_bond_dim: usize,
}

/// Set the limits every later call in this process resolves against, below any a
/// request sets. `Limits()` clears them.
#[pyfunction]
pub fn set_default_limits(limits: &PyLimits) {
    prism_q::set_default_limits(limits.0);
}

/// The limits last passed to `set_default_limits`.
#[pyfunction]
pub fn default_limits() -> PyLimits {
    PyLimits(prism_q::default_limits())
}
