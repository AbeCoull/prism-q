//! Braket program parsing: the circuit plus what its `#pragma braket` lines
//! declared.
//!
//! Result requests cross as plain dictionaries rather than as classes. A
//! caller translating into Braket's own schema objects switches on `type` and
//! reads the rest, and a dictionary says that without a class per variant.

use num_complex::Complex64;
use prism_q::circuit::braket::{Observable, ObservableFactor, ResultSpec, Targets};
use prism_q::circuit::openqasm;
use prism_q::sim::ResultValue;
use prism_q::{BackendKind, simulate};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::backend::PyBackendKind;
use crate::circuit::PyCircuit;
use crate::error::PyPrismResult;
use crate::noise::PyNoiseModel;
use crate::numpy_util::{complex_array, complex_matrix, f64_array};
use crate::parameter::PyParameters;
use crate::sim::DEFAULT_SEED;

fn targets_to_py<'py>(py: Python<'py>, targets: &Targets) -> PyResult<Bound<'py, PyAny>> {
    match targets {
        Targets::All => Ok(py.None().into_bound(py)),
        Targets::These(indices) => Ok(PyList::new(py, indices)?.into_any()),
        other => Err(PyNotImplementedError::new_err(format!(
            "target form {other:?} is newer than this binding"
        ))),
    }
}

/// One observable factor as a dictionary. A `hermitian` factor carries its
/// matrix as a `complex128` array, the spelling `evaluate` returns one in.
fn factor_to_py<'py>(py: Python<'py>, factor: &ObservableFactor) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    match factor {
        ObservableFactor::Pauli { axis, targets } => {
            entry.set_item("kind", axis.letter().to_ascii_lowercase().to_string())?;
            entry.set_item("targets", targets_to_py(py, targets)?)?;
        }
        ObservableFactor::Hadamard { targets } => {
            entry.set_item("kind", "h")?;
            entry.set_item("targets", targets_to_py(py, targets)?)?;
        }
        ObservableFactor::Identity { targets } => {
            entry.set_item("kind", "i")?;
            entry.set_item("targets", targets_to_py(py, targets)?)?;
        }
        ObservableFactor::Hermitian { matrix, targets } => {
            entry.set_item("kind", "hermitian")?;
            entry.set_item("targets", targets_to_py(py, targets)?)?;
            let side = matrix.len();
            let flat: Vec<Complex64> = matrix.iter().flatten().copied().collect();
            entry.set_item("matrix", complex_matrix(py, side, side, flat)?)?;
        }
        other => {
            return Err(PyNotImplementedError::new_err(format!(
                "observable factor {other:?} is newer than this binding"
            )));
        }
    }
    Ok(entry)
}

fn observable_to_py<'py>(py: Python<'py>, observable: &Observable) -> PyResult<Bound<'py, PyList>> {
    let factors = PyList::empty(py);
    for factor in &observable.factors {
        factors.append(factor_to_py(py, factor)?)?;
    }
    Ok(factors)
}

fn result_to_py<'py>(py: Python<'py>, spec: &ResultSpec) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    entry.set_item("type", spec.name())?;
    entry.set_item("requires_exact", spec.requires_exact())?;
    match spec {
        ResultSpec::StateVector => {}
        ResultSpec::DensityMatrix(targets) | ResultSpec::Probability(targets) => {
            entry.set_item("targets", targets_to_py(py, targets)?)?;
        }
        ResultSpec::Amplitude(states) => {
            entry.set_item("states", PyList::new(py, states)?)?;
        }
        ResultSpec::Expectation(observable)
        | ResultSpec::Variance(observable)
        | ResultSpec::Sample(observable) => {
            entry.set_item("observable", observable_to_py(py, observable)?)?;
        }
        other => {
            return Err(PyNotImplementedError::new_err(format!(
                "result request {other:?} is newer than this binding"
            )));
        }
    }
    Ok(entry)
}

/// A parsed Braket program.
#[pyclass(name = "BraketProgram", module = "prism_q")]
pub struct PyBraketProgram {
    circuit: prism_q::Circuit,
    parameters: prism_q::Parameters,
    results: Vec<ResultSpec>,
    noise: Option<prism_q::NoiseModel>,
}

#[pymethods]
impl PyBraketProgram {
    #[getter]
    fn circuit(&self) -> PyCircuit {
        PyCircuit(self.circuit.clone())
    }

    #[getter]
    fn parameters(&self) -> PyParameters {
        PyParameters(self.parameters.clone())
    }

    /// The noise model the `#pragma braket noise` lines built, or `None`.
    #[getter]
    fn noise(&self) -> Option<PyNoiseModel> {
        self.noise.clone().map(|inner| PyNoiseModel { inner })
    }

    /// `#pragma braket result` requests, in declaration order.
    ///
    /// Each entry carries `type`, `requires_exact`, and whichever of
    /// `targets`, `states` or `observable` that type takes. A `targets` of
    /// `None` means every qubit, which `all` and an omitted list both spell.
    #[getter]
    fn results<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for spec in &self.results {
            list.append(result_to_py(py, spec)?)?;
        }
        Ok(list)
    }

    /// Run the program and compute the results its pragmas requested.
    ///
    /// Returns one dictionary per request carrying `type` and `value`, in
    /// declaration order. Values are in Braket's own conventions rather than in
    /// PRISM-Q's: qubit 0 is the most significant bit of a basis index, the
    /// opposite of every native terminal here, so `x q[0]` lands at index 2 of a
    /// two-qubit result and not at index 1. An `expectation` or `variance` value
    /// is always a list, one entry per reported value, where Braket collapses a
    /// single-target request to a scalar.
    ///
    /// The program's own noise model is attached, and with one present the
    /// density matrix is the default backend, since a mixture has no pure state
    /// to read. `backend` overrides that.
    ///
    /// At the default `shots` of zero every value is exact and `sample` is
    /// declined, having no exact reading. Above zero the values come from a
    /// measurement record instead: each observable is rotated onto the
    /// computational basis, one pass serves `sample`, `expectation` and
    /// `variance` together, and `state_vector`, `amplitude` and `density_matrix`
    /// are declined in turn. Two observables reading one qubit in different
    /// bases cannot share a record and are rejected.
    #[pyo3(signature = (seed = DEFAULT_SEED, backend = None, shots = 0))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        backend: Option<PyBackendKind>,
        shots: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let kind = backend
            .map(|b| b.0.clone())
            .or_else(|| self.noise.is_some().then_some(BackendKind::DensityMatrix));
        let circuit = &self.circuit;
        let noise = self.noise.as_ref();
        let specs = &self.results;
        let values = py
            .detach(|| {
                let mut sim = simulate(circuit);
                if let Some(k) = &kind {
                    sim = sim.backend(k.clone());
                }
                if let Some(model) = noise {
                    sim = sim.noise(model);
                }
                let sim = sim.seed(seed);
                if shots == 0 {
                    sim.braket_results(specs)
                } else {
                    sim.braket_results_sampled(specs, shots)
                }
            })
            .map_err(crate::error::PyPrismError::from)?;

        let list = PyList::empty(py);
        for (spec, value) in specs.iter().zip(values) {
            let entry = PyDict::new(py);
            entry.set_item("type", spec.name())?;
            match value {
                ResultValue::StateVector(amplitudes) => {
                    entry.set_item("value", complex_array(py, amplitudes))?;
                }
                ResultValue::DensityMatrix(rows) => {
                    let side = rows.len();
                    let flat: Vec<Complex64> = rows.into_iter().flatten().collect();
                    let matrix = complex_matrix(py, side, side, flat)?;
                    entry.set_item("value", matrix)?;
                }
                ResultValue::Amplitude(pairs) => {
                    let mapping = PyDict::new(py);
                    for (label, amplitude) in pairs {
                        mapping.set_item(label, amplitude)?;
                    }
                    entry.set_item("value", mapping)?;
                }
                ResultValue::Probability(probabilities) => {
                    entry.set_item("value", f64_array(py, probabilities))?;
                }
                ResultValue::Expectation(values) | ResultValue::Variance(values) => {
                    entry.set_item("value", values)?;
                }
                ResultValue::Sample(series) => {
                    let rows = PyList::empty(py);
                    for single in series {
                        rows.append(f64_array(py, single))?;
                    }
                    entry.set_item("value", rows)?;
                }
                other => {
                    return Err(PyNotImplementedError::new_err(format!(
                        "result value {other:?} is newer than this binding"
                    )));
                }
            }
            list.append(entry)?;
        }
        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "BraketProgram(num_qubits={}, gates={}, results={}, noise={})",
            self.circuit.num_qubits,
            self.circuit.gate_count(),
            self.results.len(),
            self.noise.is_some()
        )
    }
}

/// Parse an OpenQASM 3.0 program under Amazon Braket's dialect.
///
/// Reads `#pragma braket` result, noise, unitary and verbatim lines, and takes
/// `gpi`, `gpi2` and `ms` angles in radians rather than the turns the native
/// reading uses.
#[pyfunction]
pub fn parse_braket(source: &str) -> PyPrismResult<PyBraketProgram> {
    let program = openqasm::parse_braket(source)?;
    Ok(PyBraketProgram {
        circuit: program.circuit,
        parameters: program.parameters,
        results: program.results,
        noise: program.noise,
    })
}
