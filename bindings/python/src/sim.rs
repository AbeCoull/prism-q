//! Simulation builder and result types.
//!
//! `PySimulation` mirrors the core `Simulate` typestate builder but stores
//! owned data and rebuilds the chain inside each terminal call, so none of the
//! Rust lifetimes or `Seeded`/`Unseeded` type parameters leak into Python.
//! Heavy terminals release the GIL via `py.allow_threads`.

use std::collections::HashMap;

use num_complex::Complex64;
use numpy::PyArray1;
use prism_q::backend::Backend;
use prism_q::{
    bitstring, simulate as core_simulate, BackendKind, Circuit, CountsResult, MarginalsResult,
    NoiseModel, RunOutcome, ShotsResult, StatevectorBackend,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backend::PyBackendKind;
use crate::circuit::PyCircuit;
use crate::error::{invalid, PyPrismResult};
use crate::noise::PyNoiseModel;
use crate::numpy_util::{complex_array, f64_array};

const DEFAULT_SEED: u64 = 42;

/// A configured simulation. Set options with `.seed()`, `.backend()`,
/// `.noise()`, then run a terminal: `.run()`, `.shots()`, `.sample_counts()`,
/// `.marginals()`, or `.state_vector()`.
#[pyclass(name = "Simulation", module = "prism_q")]
pub struct PySimulation {
    circuit: Circuit,
    seed: Option<u64>,
    kind: Option<BackendKind>,
    noise: Option<Py<PyNoiseModel>>,
}

#[pymethods]
impl PySimulation {
    /// Set the random seed (default 42).
    fn seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.seed = Some(seed);
        slf
    }

    /// Select an explicit backend.
    fn backend(mut slf: PyRefMut<'_, Self>, kind: PyBackendKind) -> PyRefMut<'_, Self> {
        slf.kind = Some(kind.0);
        slf
    }

    /// Attach a noise model. Only `.shots()` and `.sample_counts()` honor noise.
    fn noise(mut slf: PyRefMut<'_, Self>, model: Py<PyNoiseModel>) -> PyRefMut<'_, Self> {
        slf.noise = Some(model);
        slf
    }

    /// Run once and return classical bits plus the probability distribution.
    fn run(&self, py: Python<'_>) -> PyPrismResult<PyRunOutcome> {
        if self.noise.is_some() {
            return Err(invalid(
                "run() does not support noise; use shots() or sample_counts()",
            ));
        }
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let circuit = &self.circuit;
        let outcome: RunOutcome = py.allow_threads(|| {
            let mut sim = core_simulate(circuit);
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            sim.seed(seed).run()
        })?;
        Ok(PyRunOutcome::from_outcome(outcome))
    }

    /// Sample `num_shots` measurement records.
    fn shots(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyShotsResult> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let result: ShotsResult = py.allow_threads(|| {
            let mut sim = core_simulate(circuit);
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            sim.seed(seed).shots(num_shots)
        })?;
        Ok(PyShotsResult { inner: result })
    }

    /// Sample `num_shots` shots and return a frequency histogram.
    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyCountsResult> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let result: CountsResult = py.allow_threads(|| {
            let mut sim = core_simulate(circuit);
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            sim.seed(seed).sample_counts(num_shots)
        })?;
        Ok(PyCountsResult {
            counts: result.counts,
            num_classical_bits: result.num_classical_bits,
        })
    }

    /// Per-qubit marginal probabilities `(p0, p1)`.
    fn marginals(&self, py: Python<'_>) -> PyPrismResult<Vec<(f64, f64)>> {
        if self.noise.is_some() {
            return Err(invalid("marginals() does not support noise; use shots()"));
        }
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let circuit = &self.circuit;
        let result: MarginalsResult = py.allow_threads(|| {
            let mut sim = core_simulate(circuit);
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            sim.seed(seed).marginals()
        })?;
        Ok(result.marginals)
    }

    /// Exact statevector amplitudes as a `complex128` array.
    ///
    /// Always uses the statevector backend regardless of `.backend(...)`, and
    /// does not support an attached noise model.
    fn state_vector<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray1<Complex64>>> {
        if self.noise.is_some() {
            return Err(invalid("state_vector() does not support noise"));
        }
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let circuit = &self.circuit;
        let amps: Vec<Complex64> = py.allow_threads(|| {
            let mut backend = StatevectorBackend::new(seed);
            prism_q::run_on(&mut backend, circuit)?;
            backend.export_statevector()
        })?;
        Ok(complex_array(py, amps))
    }
}

impl PySimulation {
    /// Borrow the attached noise model and produce an owned copy, so the GIL can
    /// be released during the run without holding a `PyRef`.
    fn owned_noise(&self, py: Python<'_>) -> Option<NoiseModel> {
        self.noise.as_ref().map(|pn| pn.borrow(py).clone_model())
    }
}

/// Construct a [`Simulation`] for `circuit`.
#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate(circuit: &PyCircuit) -> PySimulation {
    PySimulation {
        circuit: circuit.inner().clone(),
        seed: None,
        kind: None,
        noise: None,
    }
}

/// Parse an OpenQASM string and run with automatic backend selection.
#[pyfunction]
pub fn run_qasm(source: &str, seed: u64) -> PyPrismResult<PyRunOutcome> {
    let outcome = prism_q::run_qasm(source, seed)?;
    Ok(PyRunOutcome::from_outcome(outcome))
}

fn counts_to_dict<'py>(
    py: Python<'py>,
    counts: &HashMap<Vec<u64>, u64>,
    num_bits: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, count) in counts {
        dict.set_item(bitstring(key, num_bits), count)?;
    }
    Ok(dict)
}

/// Result of a single run: classical bits and the probability distribution.
#[pyclass(name = "RunOutcome", module = "prism_q")]
pub struct PyRunOutcome {
    classical_bits: Vec<bool>,
    probabilities: Option<Vec<f64>>,
}

impl PyRunOutcome {
    fn from_outcome(outcome: RunOutcome) -> Self {
        Self {
            classical_bits: outcome.classical_bits,
            probabilities: outcome.probabilities.map(|p| p.to_vec()),
        }
    }
}

#[pymethods]
impl PyRunOutcome {
    #[getter]
    fn classical_bits(&self) -> Vec<bool> {
        self.classical_bits.clone()
    }

    /// Probability of each basis state as a `float64` array, or `None` if the
    /// backend cannot expose a dense distribution.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.probabilities
            .as_ref()
            .map(|v| f64_array(py, v.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RunOutcome(classical_bits={}, has_probabilities={})",
            self.classical_bits.len(),
            self.probabilities.is_some()
        )
    }
}

/// Result of multi-shot sampling.
#[pyclass(name = "ShotsResult", module = "prism_q")]
pub struct PyShotsResult {
    inner: ShotsResult,
}

#[pymethods]
impl PyShotsResult {
    #[getter]
    fn shots(&self) -> Vec<Vec<bool>> {
        self.inner.shots.clone()
    }

    #[getter]
    fn num_shots(&self) -> usize {
        self.inner.num_shots()
    }

    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.inner.num_classical_bits()
    }

    /// Frequency histogram keyed by bitstring.
    ///
    /// In each key, character `i` is classical bit `i`, with bit 0 leftmost
    /// (LSB-first). This follows PRISM-Q's convention that `q[0]` is the
    /// least-significant qubit, so keys read reversed relative to Qiskit.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        counts_to_dict(py, &self.inner.counts(), self.inner.num_classical_bits())
    }

    fn __repr__(&self) -> String {
        format!(
            "ShotsResult(num_shots={}, num_classical_bits={})",
            self.inner.num_shots(),
            self.inner.num_classical_bits()
        )
    }
}

/// Frequency histogram from `sample_counts`.
#[pyclass(name = "CountsResult", module = "prism_q")]
pub struct PyCountsResult {
    counts: HashMap<Vec<u64>, u64>,
    num_classical_bits: usize,
}

#[pymethods]
impl PyCountsResult {
    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }

    /// Frequency histogram keyed by bitstring.
    ///
    /// In each key, character `i` is classical bit `i`, with bit 0 leftmost
    /// (LSB-first). This follows PRISM-Q's convention that `q[0]` is the
    /// least-significant qubit, so keys read reversed relative to Qiskit.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        counts_to_dict(py, &self.counts, self.num_classical_bits)
    }

    fn __repr__(&self) -> String {
        format!(
            "CountsResult(distinct={}, num_classical_bits={})",
            self.counts.len(),
            self.num_classical_bits
        )
    }
}
