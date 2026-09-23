//! Simulation builder and result types. `PySimulation` stores owned data and rebuilds
//! the core `Simulate` chain inside each terminal, so no Rust lifetimes or typestate
//! parameters reach Python.

use std::collections::HashMap;

use num_complex::Complex64;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use prism_q::{
    BackendKind, BondReport, Circuit, CountsResult, Exactness, MarginalsResult, NoiseModel,
    ParamLink, Parameters, PauliAxis, PauliObservable, PauliTerm, Placement, Probabilities,
    ReducedDensityMatrix, RunMetadata, RunOutcome, SaveRecord, SavedValue, ShotsResult, bitstring,
    simulate as core_simulate,
};
use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::backend::PyBackendKind;
use crate::circuit::PyCircuit;
use crate::error::{PyPrismResult, invalid};
use crate::noise::PyNoiseModel;
use crate::numpy_util::{bool_matrix, complex_array, complex_matrix, f64_array};

pub(crate) const DEFAULT_SEED: u64 = 42;

/// A configured simulation. Set options with `.seed()`, `.backend()`,
/// `.noise()`, then run one terminal, which consumes the request.
#[pyclass(name = "Simulation", module = "prism_q")]
pub struct PySimulation {
    circuit: Circuit,
    seed: Option<u64>,
    kind: Option<BackendKind>,
    noise: Option<Py<PyNoiseModel>>,
    initial_state: Option<Vec<Complex64>>,
    initial_density_matrix: Option<Vec<Complex64>>,
    require_exact: bool,
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

    /// Reject a route that could return an approximate answer instead of
    /// taking it and reporting it through `result.metadata`.
    ///
    /// Automatic dispatch sends a circuit past the statevector cap to a
    /// bounded-bond MPS, which is the only route those circuits have. Call this
    /// when an approximate answer is worse than no answer.
    fn require_exact(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.require_exact = true;
        slf
    }

    /// Attach a noise model.
    ///
    /// `.shots()` and `.sample_counts()` average trajectories. `.run()`,
    /// `.marginals()`, and `.expectation_values()` answer from the exact
    /// mixture, which requires `BackendKind.density_matrix()`. Readout error
    /// is not part of that mixture, so `.run()` and `.marginals()` reject a
    /// model carrying it and `.sample_counts()` is the terminal that applies it.
    fn noise(mut slf: PyRefMut<'_, Self>, model: Py<PyNoiseModel>) -> PyRefMut<'_, Self> {
        slf.noise = Some(model);
        slf
    }

    /// Start from `amplitudes` instead of |0...0>.
    ///
    /// Takes any sequence of complex numbers, including a `complex128` NumPy
    /// array, indexed with qubit 0 in the least significant bit. The length must
    /// be `2 ** num_qubits` and the vector must be normalized. A start state
    /// runs on the statevector (dense, GPU, or distributed) or density-matrix
    /// backend only. `.shots()` and `.sample_counts()` reject one with a noise
    /// model attached, and
    /// `.expectation_gradient()` and `.density_matrix_expectation_values()`
    /// reject one outright.
    fn initial_state(
        mut slf: PyRefMut<'_, Self>,
        amplitudes: Vec<Complex64>,
    ) -> PyRefMut<'_, Self> {
        slf.initial_state = Some(amplitudes);
        slf.initial_density_matrix = None;
        slf
    }

    /// Start from the mixture `rho` instead of |0...0>.
    ///
    /// Takes a square `complex128` NumPy array, or a sequence of equal-length
    /// rows, in the layout `reduced_density_matrix()` over the whole register
    /// returns: row-major `2 ** n` by `2 ** n` with qubit 0 in the least
    /// significant bit of both indices. The matrix must be Hermitian to 1e-12
    /// per entry and have unit trace to 1e-9; positive semidefiniteness is not
    /// checked. Only `BackendKind.density_matrix()` and its GPU sibling accept
    /// one; every other backend, `auto()` included, raises `PrismError`
    /// naming itself. Replaces an earlier `.initial_state()`, and vice versa.
    fn initial_density_matrix<'py>(
        mut slf: PyRefMut<'py, Self>,
        rho: &Bound<'py, PyAny>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let (rows, cols, flat) = match rho.extract::<PyReadonlyArray2<'py, Complex64>>() {
            Ok(array) => {
                let view = array.as_array();
                let (rows, cols) = view.dim();
                (rows, cols, view.iter().copied().collect::<Vec<_>>())
            }
            Err(_) => {
                let rows: Vec<Vec<Complex64>> = rho.extract().map_err(|_| {
                    invalid(
                        "initial_density_matrix() takes a complex128 matrix or a sequence of \
                         complex rows",
                    )
                })?;
                let cols = rows.first().map_or(0, Vec::len);
                if rows.iter().any(|row| row.len() != cols) {
                    return Err(invalid("initial_density_matrix() takes rows of one length"));
                }
                (rows.len(), cols, rows.concat())
            }
        };
        if rows != cols {
            return Err(invalid(format!(
                "initial_density_matrix() takes a square matrix, got shape ({rows}, {cols})"
            )));
        }
        slf.initial_density_matrix = Some(flat);
        slf.initial_state = None;
        Ok(slf)
    }

    /// Run once and return classical bits plus the probability distribution.
    fn run(&self, py: Python<'_>) -> PyPrismResult<PyRunOutcome> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let outcome: RunOutcome = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).run()
        })?;
        Ok(PyRunOutcome::from_outcome(outcome))
    }

    /// Sample `num_shots` measurement records.
    fn shots(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyShotsResult> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result: ShotsResult = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).shots(num_shots)
        })?;
        Ok(PyShotsResult { inner: result })
    }

    /// Sample `num_shots` shots and return a frequency histogram.
    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyCountsResult> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result: CountsResult = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).sample_counts(num_shots)
        })?;
        Ok(PyCountsResult {
            counts: result.counts,
            num_classical_bits: result.num_classical_bits,
            metadata: PyRunMetadata::new(result.metadata),
        })
    }

    /// Per-qubit marginal probabilities `(p0, p1)`.
    fn marginals(&self, py: Python<'_>) -> PyPrismResult<Vec<(f64, f64)>> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result: MarginalsResult = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).marginals()
        })?;
        Ok(result.marginals)
    }

    /// Exact statevector amplitudes as a `complex128` array.
    ///
    /// Indexed with qubit 0 in the least significant bit, so `x q[0]` puts the
    /// amplitude at index 1. Honours `.backend(...)`; a backend holding no pure
    /// state declines rather than answering from a statevector the caller did
    /// not ask for. An attached noise model declines for the same reason.
    fn state_vector<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray1<Complex64>>> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let amps: Vec<Complex64> = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).state_vector()
        })?;
        Ok(complex_array(py, amps))
    }

    /// Compute `⟨H⟩` and its exact gradient with respect to the trainable
    /// parameters via the adjoint method, returning `(value, gradient)` with
    /// the gradient as a `float64` array.
    ///
    /// `hamiltonian` is a list of `(coefficient, [(qubit, axis), ...])` terms,
    /// where `axis` is one of `"X"`, `"Y"`, `"Z"` (identity factors omitted).
    /// `parameters` is a list of `(instruction_index, parameter_slot)` links,
    /// as returned by `CircuitBuilder.parameter_links()`. Runs on the
    /// statevector backend; the circuit must be unitary.
    #[pyo3(signature = (hamiltonian, parameters))]
    fn expectation_gradient<'py>(
        &self,
        py: Python<'py>,
        hamiltonian: Vec<(f64, Vec<(usize, String)>)>,
        parameters: Vec<(usize, usize)>,
    ) -> PyPrismResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if self.noise.is_some() {
            return Err(invalid("expectation_gradient() does not support noise"));
        }
        let mut terms: Vec<(f64, Vec<PauliTerm>)> = Vec::with_capacity(hamiltonian.len());
        for (coeff, factors) in hamiltonian {
            terms.push((coeff, parse_pauli_string(factors)?));
        }
        let links: Vec<ParamLink> = parameters
            .into_iter()
            .map(|(instruction, slot)| ParamLink { instruction, slot })
            .collect();
        let num_slots = links.iter().map(|l| l.slot + 1).max().unwrap_or(0);
        let params = Parameters::from_links(links, num_slots);
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            // Carried so the core rejects it rather than ignoring it here.
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).expectation_gradient(&terms, &params)
        })?;
        Ok((result.value, f64_array(py, result.gradient)))
    }

    /// Same gradient by the parameter-shift rule: two extra circuit runs per
    /// parameter instead of one backward sweep, and the only route for a
    /// backend with no adjoint pass. Takes the `expectation_gradient()`
    /// argument shape and returns the same pair.
    #[pyo3(signature = (hamiltonian, parameters))]
    fn expectation_gradient_shift<'py>(
        &self,
        py: Python<'py>,
        hamiltonian: Vec<(f64, Vec<(usize, String)>)>,
        parameters: Vec<(usize, usize)>,
    ) -> PyPrismResult<(f64, Bound<'py, PyArray1<f64>>)> {
        if self.noise.is_some() {
            return Err(invalid(
                "expectation_gradient_shift() does not support noise",
            ));
        }
        let mut terms: Vec<(f64, Vec<PauliTerm>)> = Vec::with_capacity(hamiltonian.len());
        for (coeff, factors) in hamiltonian {
            terms.push((coeff, parse_pauli_string(factors)?));
        }
        let links: Vec<ParamLink> = parameters
            .into_iter()
            .map(|(instruction, slot)| ParamLink { instruction, slot })
            .collect();
        let num_slots = links.iter().map(|l| l.slot + 1).max().unwrap_or(0);
        let params = Parameters::from_links(links, num_slots);
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            // Carried so the core rejects it rather than ignoring it here.
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).expectation_gradient_shift(&terms, &params)
        })?;
        Ok((result.value, f64_array(py, result.gradient)))
    }

    /// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on the circuit's
    /// output state, honoring the selected backend.
    ///
    /// Each observable is a list of `(qubit, axis)` factors, where `axis` is one
    /// of `"X"`, `"Y"`, `"Z"` and identity factors are omitted. The circuit must
    /// be unitary. With a noise model attached the value is the exact
    /// `Tr(rho P)`, which requires `BackendKind.density_matrix()`.
    #[pyo3(signature = (observables))]
    fn expectation_values(
        &self,
        py: Python<'_>,
        observables: Vec<Vec<(usize, String)>>,
    ) -> PyPrismResult<Vec<f64>> {
        let observables = parse_observables(observables)?;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let values = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).expectation_values(&observables)
        })?;
        Ok(values)
    }

    /// Joint probability distribution over `qubits`, `2 ** len(qubits)` entries
    /// with `qubits[0]` in the lowest bit.
    ///
    /// Generalizes `marginals()`, which reports each qubit on its own and so
    /// cannot show correlation: a Bell pair reads `(0.5, 0.5)` twice there and
    /// `[0.5, 0, 0, 0.5]` here. Honours `.backend(...)` and an attached noise
    /// model, whose answer is the marginal of the exact mixture.
    #[pyo3(signature = (qubits))]
    fn probabilities_of<'py>(
        &self,
        py: Python<'py>,
        qubits: Vec<usize>,
    ) -> PyPrismResult<Bound<'py, PyArray1<f64>>> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let values = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).probabilities_of(&qubits)
        })?;
        Ok(f64_array(py, values))
    }

    /// Reduced density matrix of `qubits`, with the route that produced it.
    ///
    /// `result.matrix[t][u]` is `<t|rho|u>` with bit `i` of `t` the state of
    /// `qubits[i]`, so `qubits[0]` is the lowest bit as it is in a basis index.
    /// The circuit must be unitary: the answer is read off one state, and a
    /// measurement or reset leaves one seeded branch of several. With a noise
    /// model attached the answer is the marginal of the exact mixture.
    #[pyo3(signature = (qubits))]
    fn reduced_density_matrix(
        &self,
        py: Python<'_>,
        qubits: Vec<usize>,
    ) -> PyPrismResult<PyReducedDensityMatrix> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let reduced: ReducedDensityMatrix = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).reduced_density_matrix(&qubits)
        })?;
        Ok(PyReducedDensityMatrix {
            purity: reduced.purity(),
            qubits: reduced.qubits,
            data: reduced.data,
            metadata: PyRunMetadata::new(reduced.metadata),
        })
    }

    /// Von Neumann entropy of `subsystem` in nats, with the Schmidt spectrum
    /// behind it.
    ///
    /// A Bell pair cut in half reads `ln 2`. The cut must leave both sides
    /// non-empty, so the whole register is rejected. `schmidt_values` is `None`
    /// where the backend holds the entropy without the spectrum, as a
    /// stabilizer cut past the export cap does.
    #[pyo3(signature = (subsystem))]
    fn entanglement_entropy(
        &self,
        py: Python<'_>,
        subsystem: Vec<usize>,
    ) -> PyPrismResult<PyEntropyResult> {
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).entanglement_entropy(&subsystem)
        })?;
        Ok(PyEntropyResult {
            subsystem: result.subsystem,
            entropy: result.entropy,
            schmidt_values: result.schmidt_values,
            metadata: PyRunMetadata::new(result.metadata),
        })
    }

    /// `Var(H) = <H^2> - <H>^2` for a weighted Pauli observable.
    ///
    /// This is the spread of the operator itself, not
    /// `ObservableExpectation.variance`, which sums per-group variances and so
    /// drops the covariance between measurement groups. `hamiltonian` takes the
    /// `observable_expectation()` term shape.
    #[pyo3(signature = (hamiltonian))]
    fn observable_variance(
        &self,
        py: Python<'_>,
        hamiltonian: Vec<(f64, Vec<(usize, String)>)>,
    ) -> PyPrismResult<PyObservableVariance> {
        let observable = build_observable(hamiltonian)?;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).observable_variance(&observable)
        })?;
        Ok(PyObservableVariance {
            variance: result.variance,
            mean: result.mean,
            metadata: PyRunMetadata::new(result.metadata),
        })
    }

    /// `expectation_values()` with the provenance of the run that served it.
    /// Under `BackendKind.auto()` a wide shallow circuit can be answered by a
    /// tensor contraction rather than by the state vector, and only the
    /// metadata says which ran.
    #[pyo3(signature = (observables))]
    fn expectation_values_reported(
        &self,
        py: Python<'_>,
        observables: Vec<Vec<(usize, String)>>,
    ) -> PyPrismResult<PyExpectationResult> {
        let observables = parse_observables(observables)?;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).expectation_values_reported(&observables)
        })?;
        Ok(PyExpectationResult {
            values: result.values,
            metadata: PyRunMetadata::new(result.metadata),
        })
    }

    /// `|<a|b>|^2` between this circuit's output state and `other`'s.
    ///
    /// Both circuits must declare the same width and both must be unitary.
    /// Each side keeps its own backend, seed and start state, so the result
    /// carries one provenance per side. Two states in the same representation
    /// contract natively at any width; a mismatched pair is served by a dense
    /// export of both, so it reaches only as far as the export cap. A noise
    /// model on either side is rejected, since the fidelity of two mixtures is
    /// not an inner product.
    #[pyo3(signature = (other))]
    fn overlap(&self, py: Python<'_>, other: &PySimulation) -> PyPrismResult<PyOverlapResult> {
        let left_noise = self.owned_noise(py);
        let right_noise = other.owned_noise(py);
        let result = py.detach(|| {
            let mut left = core_simulate(&self.circuit);
            if self.require_exact {
                left = left.require_exact();
            }
            if let Some(k) = &self.kind {
                left = left.backend(k.clone());
            }
            if let Some(nm) = &left_noise {
                left = left.noise(nm);
            }
            if let Some(amplitudes) = self.initial_state.as_deref() {
                left = left.initial_state(amplitudes);
            }
            if let Some(rho) = self.initial_density_matrix.as_deref() {
                left = left.initial_density_matrix(rho);
            }

            let mut right = core_simulate(&other.circuit);
            if other.require_exact {
                right = right.require_exact();
            }
            if let Some(k) = &other.kind {
                right = right.backend(k.clone());
            }
            if let Some(nm) = &right_noise {
                right = right.noise(nm);
            }
            if let Some(amplitudes) = other.initial_state.as_deref() {
                right = right.initial_state(amplitudes);
            }
            if let Some(rho) = other.initial_density_matrix.as_deref() {
                right = right.initial_density_matrix(rho);
            }

            left.seed(self.seed.unwrap_or(DEFAULT_SEED))
                .overlap(right.seed(other.seed.unwrap_or(DEFAULT_SEED)))
        })?;
        Ok(PyOverlapResult {
            fidelity: result.fidelity,
            left: PyRunMetadata::new(result.left),
            right: PyRunMetadata::new(result.right),
        })
    }

    /// Compute `⟨H⟩` and its grouped-measurement variance for a weighted Pauli
    /// observable on the circuit's output state.
    ///
    /// `hamiltonian` takes the `expectation_gradient()` term shape: a list of
    /// `(coefficient, [(qubit, axis), ...])` pairs with `axis` one of `"X"`,
    /// `"Y"`, `"Z"`, identity factors omitted, and an empty factor list acting
    /// as a constant offset. Identical strings merge by summing coefficients.
    /// The statevector family reports the variance (see
    /// `ObservableExpectation.variance` for the contract); every other route
    /// returns the weighted mean with `variance` of `None`.
    #[pyo3(signature = (hamiltonian))]
    fn observable_expectation(
        &self,
        py: Python<'_>,
        hamiltonian: Vec<(f64, Vec<(usize, String)>)>,
    ) -> PyPrismResult<PyObservableExpectation> {
        let observable = build_observable(hamiltonian)?;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let require_exact = self.require_exact;
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let start = self.initial_state.as_deref();
        let mixed = self.initial_density_matrix.as_deref();
        let result = py.detach(|| {
            let mut sim = core_simulate(circuit);
            if require_exact {
                sim = sim.require_exact();
            }
            if let Some(k) = &kind {
                sim = sim.backend(k.clone());
            }
            if let Some(nm) = &owned_noise {
                sim = sim.noise(nm);
            }
            if let Some(amplitudes) = start {
                sim = sim.initial_state(amplitudes);
            }
            if let Some(rho) = mixed {
                sim = sim.initial_density_matrix(rho);
            }
            sim.seed(seed).observable_expectation(&observable)
        })?;
        Ok(PyObservableExpectation {
            mean: result.mean,
            variance: result.variance,
            group_variances: result.group_variances,
            std_error: result.std_error,
            metadata: PyRunMetadata::new(result.metadata),
        })
    }

    /// Exact `Tr(rho P)` for each joint Pauli observable, evolving the
    /// density-matrix backend through the circuit and the attached noise model.
    ///
    /// Observables take the same `(qubit, axis)` form as
    /// `expectation_values()`. Measurements are read off the final mixed state
    /// without collapse, so this is the zero-variance analogue of
    /// trajectory-averaged expectation values. Always uses the density-matrix
    /// backend regardless of `.backend(...)`, so the circuit must fit that
    /// backend's qubit cap.
    #[pyo3(signature = (observables))]
    fn density_matrix_expectation_values(
        &self,
        py: Python<'_>,
        observables: Vec<Vec<(usize, String)>>,
    ) -> PyPrismResult<Vec<f64>> {
        if self.initial_state.is_some() || self.initial_density_matrix.is_some() {
            return Err(invalid(
                "density_matrix_expectation_values() does not accept a start state; call \
                 expectation_values() with BackendKind.density_matrix(), which takes one on a \
                 unitary circuit",
            ));
        }
        let observables = parse_observables(observables)?;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let circuit = &self.circuit;
        let owned_noise = self.owned_noise(py);
        let values = py.detach(|| {
            prism_q::density_matrix_expectation_values(
                circuit,
                &observables,
                owned_noise.as_ref(),
                seed,
            )
        })?;
        Ok(values)
    }
}

fn parse_observables(observables: Vec<Vec<(usize, String)>>) -> PyPrismResult<Vec<Vec<PauliTerm>>> {
    observables.into_iter().map(parse_pauli_string).collect()
}

fn build_observable(
    hamiltonian: Vec<(f64, Vec<(usize, String)>)>,
) -> PyPrismResult<PauliObservable> {
    let mut terms: Vec<(f64, Vec<PauliTerm>)> = Vec::with_capacity(hamiltonian.len());
    for (coefficient, factors) in hamiltonian {
        terms.push((coefficient, parse_pauli_string(factors)?));
    }
    Ok(PauliObservable::from_terms(terms)?)
}

pub(crate) fn parse_pauli_string(factors: Vec<(usize, String)>) -> PyPrismResult<Vec<PauliTerm>> {
    factors
        .into_iter()
        .map(|(qubit, axis)| Ok(PauliTerm::new(qubit, parse_axis(&axis)?)))
        .collect()
}

fn parse_axis(axis: &str) -> PyPrismResult<PauliAxis> {
    match axis.to_ascii_uppercase().as_str() {
        "X" => Ok(PauliAxis::X),
        "Y" => Ok(PauliAxis::Y),
        "Z" => Ok(PauliAxis::Z),
        other => Err(invalid(format!(
            "Pauli axis must be one of \"X\", \"Y\", \"Z\"; got {other:?}"
        ))),
    }
}

impl PySimulation {
    /// Borrow the attached noise model and produce an owned copy, so the GIL can
    /// be released during the run without holding a `PyRef`.
    fn owned_noise(&self, py: Python<'_>) -> Option<NoiseModel> {
        self.noise.as_ref().map(|pn| pn.borrow(py).clone_model())
    }
}

/// Start a `Simulation` builder for `circuit`.
#[pyfunction]
#[pyo3(name = "simulate")]
pub fn simulate(circuit: &PyCircuit) -> PySimulation {
    PySimulation {
        circuit: circuit.inner().clone(),
        seed: None,
        kind: None,
        noise: None,
        initial_state: None,
        initial_density_matrix: None,
        require_exact: false,
    }
}

/// Run a list of circuits, holding one backend across those that can share it.
///
/// Below 14 qubits (7 on the density matrix) the circuits split across cores,
/// which a loop cannot reach because each of those runs is single-threaded
/// inside; above that this saves only the crossing into Rust.
///
/// Results are identical to running each circuit on its own with the same seed.
/// A failing batch raises the first failure in list order.
#[pyfunction]
#[pyo3(signature = (circuits, backend = None, seed = DEFAULT_SEED))]
pub fn run_batch(
    py: Python<'_>,
    circuits: Vec<PyRef<'_, PyCircuit>>,
    backend: Option<PyBackendKind>,
    seed: u64,
) -> PyPrismResult<Vec<PyRunOutcome>> {
    let kind = backend.map(|b| b.0).unwrap_or(BackendKind::Auto);
    let owned: Vec<Circuit> = circuits.iter().map(|c| c.0.clone()).collect();
    let outcomes = py.detach(|| prism_q::sim::run_batch(&owned, kind, seed))?;
    Ok(outcomes
        .into_iter()
        .map(PyRunOutcome::from_outcome)
        .collect())
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

/// How a result was produced: which engine ran, whether the answer is exact,
/// and where the state lived.
#[pyclass(name = "RunMetadata", module = "prism_q", skip_from_py_object)]
#[derive(Clone)]
pub struct PyRunMetadata {
    inner: RunMetadata,
}

impl PyRunMetadata {
    fn new(inner: RunMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRunMetadata {
    /// Engine name after automatic dispatch, for example `"Mps"`.
    #[getter]
    fn backend(&self) -> String {
        format!("{:?}", self.inner.backend)
    }

    /// Which sampler ran when `backend` is a label several share, for example
    /// `"FrameSampler"` under `"CompiledStabilizer"`. `None` when the backend
    /// is the whole answer.
    #[getter]
    fn engine(&self) -> Option<String> {
        self.inner.engine.map(|engine| format!("{engine:?}"))
    }

    /// False when the engine that ran can discard state weight or estimate by
    /// sampling. Marks the route rather than the run; `fidelity_lower_bound`
    /// reports what this run discarded.
    #[getter]
    fn is_exact(&self) -> bool {
        self.inner.is_exact()
    }

    /// Lower bound on the fidelity of the produced state. `None` when the
    /// result is exact or the engine reports no bound; 1.0 when an approximate
    /// engine discarded nothing.
    #[getter]
    fn fidelity_lower_bound(&self) -> Option<f64> {
        self.inner.fidelity_lower_bound()
    }

    /// `"host"` or `"device"`, or `"unknown"` for a placement newer than
    /// this binding.
    #[getter]
    fn placement(&self) -> &'static str {
        match self.inner.placement {
            Placement::Host => "host",
            Placement::Device => "device",
            _ => "unknown",
        }
    }

    /// Shots drawn, `None` for an analytic result.
    #[getter]
    fn shots(&self) -> Option<usize> {
        self.inner.shots
    }

    /// Peak bond dimension against the cap, `None` unless the MPS ran.
    #[getter]
    fn bond(&self) -> Option<PyBondReport> {
        self.inner.bond.map(|inner| PyBondReport { inner })
    }

    fn __repr__(&self) -> String {
        let exact = match self.inner.exactness {
            Exactness::Exact => "exact".to_string(),
            Exactness::Approximate {
                fidelity_lower_bound: Some(bound),
            } => format!("approximate(fidelity>={bound:.6})"),
            Exactness::Approximate { .. } => "approximate".to_string(),
            _ => "unknown".to_string(),
        };
        let engine = self
            .engine()
            .map(|engine| format!(", engine={engine}"))
            .unwrap_or_default();
        format!(
            "RunMetadata(backend={}{engine}, {exact}, placement={})",
            self.backend(),
            self.placement()
        )
    }
}

/// Peak bond dimension an MPS run kept, beside the cap it ran under.
#[pyclass(name = "BondReport", module = "prism_q")]
pub struct PyBondReport {
    inner: BondReport,
}

#[pymethods]
impl PyBondReport {
    /// Widest bond any cut kept over the run, never above `cap`.
    #[getter]
    fn peak(&self) -> usize {
        self.inner.peak
    }

    /// The configured maximum bond dimension.
    #[getter]
    fn cap(&self) -> usize {
        self.inner.cap
    }

    /// True when some cut reached the cap.
    #[getter]
    fn saturated(&self) -> bool {
        self.inner.saturated()
    }

    fn __repr__(&self) -> String {
        format!(
            "BondReport(peak={}, cap={}, saturated={})",
            self.inner.peak,
            self.inner.cap,
            if self.inner.saturated() {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Operator variance of a weighted Pauli observable and the mean beside it.
#[pyclass(name = "ObservableVariance", module = "prism_q")]
pub struct PyObservableVariance {
    variance: f64,
    mean: f64,
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyObservableVariance {
    /// `<H^2> - <H>^2` on the output state.
    #[getter]
    fn variance(&self) -> f64 {
        self.variance
    }

    /// `<H>` on the same state, evaluated on the way to the variance.
    #[getter]
    fn mean(&self) -> f64 {
        self.mean
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservableVariance(variance={:.6}, mean={:.6})",
            self.variance, self.mean
        )
    }
}

/// Reduced density matrix of a qubit subset and the route that produced it.
#[pyclass(name = "ReducedDensityMatrix", module = "prism_q")]
pub struct PyReducedDensityMatrix {
    qubits: Vec<usize>,
    data: Vec<Complex64>,
    purity: f64,
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyReducedDensityMatrix {
    /// The subsystem as it was requested, which fixes the index order of
    /// `matrix`.
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.qubits.clone()
    }

    /// The matrix as a `complex128` array of side `2 ** len(qubits)`, row
    /// major, with `qubits[0]` the lowest bit of both indices.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<Complex64>>> {
        let side = 1usize << self.qubits.len();
        complex_matrix(py, side, side, self.data.clone())
    }

    /// `Tr(rho^2)`: 1 for a pure marginal, `2 ** -len(qubits)` for the
    /// maximally mixed one.
    #[getter]
    fn purity(&self) -> f64 {
        self.purity
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ReducedDensityMatrix(qubits={:?}, purity={:.6})",
            self.qubits, self.purity
        )
    }
}

/// Expectation values with the provenance of the run that produced them.
#[pyclass(name = "ExpectationResult", module = "prism_q")]
pub struct PyExpectationResult {
    values: Vec<f64>,
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyExpectationResult {
    /// One value per observable, in the order they were passed.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        f64_array(py, self.values.clone())
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!("ExpectationResult(values={} terms)", self.values.len())
    }
}

/// Squared overlap of two output states, with one provenance per side.
#[pyclass(name = "OverlapResult", module = "prism_q")]
pub struct PyOverlapResult {
    fidelity: f64,
    left: PyRunMetadata,
    right: PyRunMetadata,
}

#[pymethods]
impl PyOverlapResult {
    /// `|<a|b>|^2` over the two normalized states: 1 for the same state up to
    /// phase, 0 for orthogonal ones. The amplitude itself is not reported,
    /// since a tableau keeps no global phase and every truncation moves one.
    #[getter]
    fn fidelity(&self) -> f64 {
        self.fidelity
    }

    /// Provenance of the run the terminal was called on.
    #[getter]
    fn left(&self) -> PyRunMetadata {
        self.left.clone()
    }

    /// Provenance of the run passed as the argument.
    #[getter]
    fn right(&self) -> PyRunMetadata {
        self.right.clone()
    }

    fn __repr__(&self) -> String {
        format!("OverlapResult(fidelity={:.6})", self.fidelity)
    }
}

/// Entanglement entropy of a subsystem and the Schmidt spectrum behind it.
#[pyclass(name = "EntropyResult", module = "prism_q")]
pub struct PyEntropyResult {
    subsystem: Vec<usize>,
    entropy: f64,
    schmidt_values: Option<Vec<f64>>,
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyEntropyResult {
    /// The subsystem as it was requested, the side of the cut the entropy is
    /// read on.
    #[getter]
    fn subsystem(&self) -> Vec<usize> {
        self.subsystem.clone()
    }

    /// Von Neumann entropy in nats: a Bell pair cut in half reads `ln 2`.
    #[getter]
    fn entropy(&self) -> f64 {
        self.entropy
    }

    /// Schmidt values across the cut as a `float64` array, descending, with
    /// squares summing to 1. `None` where the backend holds the entropy
    /// without the spectrum.
    #[getter]
    fn schmidt_values<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.schmidt_values
            .as_ref()
            .map(|values| f64_array(py, values.clone()))
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "EntropyResult(subsystem={:?}, entropy={:.6})",
            self.subsystem, self.entropy
        )
    }
}

/// Weighted-observable expectation: mean, grouped-measurement variance, and
/// provenance.
#[pyclass(name = "ObservableExpectation", module = "prism_q")]
pub struct PyObservableExpectation {
    mean: f64,
    variance: Option<f64>,
    group_variances: Option<Vec<f64>>,
    std_error: Option<f64>,
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyObservableExpectation {
    /// `⟨H⟩`, including identity-term constants.
    #[getter]
    fn mean(&self) -> f64 {
        self.mean
    }

    /// Sum of per-group variances `Var(H_g)`, the variance of a grouped
    /// measurement estimate drawing one shot per commuting group. Excludes
    /// cross-group covariances, so it equals `Var(H)` only when one group
    /// covers every term. `None` on a route without the grouped evaluator.
    #[getter]
    fn variance(&self) -> Option<f64> {
        self.variance
    }

    /// Per-group `Var(H_g)` as a `float64` array in grouping order, the input
    /// to shot allocation. `None` whenever `variance` is.
    #[getter]
    fn group_variances<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.group_variances
            .as_ref()
            .map(|v| f64_array(py, v.clone()))
    }

    /// Standard error of `mean` when a sampling route estimated the per-term
    /// values, `None` for analytic routes.
    #[getter]
    fn std_error(&self) -> Option<f64> {
        self.std_error
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        match self.variance {
            Some(variance) => format!(
                "ObservableExpectation(mean={}, variance={variance})",
                self.mean
            ),
            None => format!("ObservableExpectation(mean={}, variance=None)", self.mean),
        }
    }
}

type FactoredBlockPy<'py> = (Vec<usize>, Bound<'py, PyArray1<f64>>);

/// Result of a single run: classical bits and the probability distribution.
///
/// The distribution is held in whatever form the run produced it and is only
/// made dense when `probabilities` is read, so a factored result wider than
/// memory is still usable through `probabilities_factored`.
#[pyclass(name = "RunOutcome", module = "prism_q")]
pub struct PyRunOutcome {
    classical_bits: Vec<bool>,
    probabilities: Option<Probabilities>,
    metadata: PyRunMetadata,
    saves: Vec<SaveRecord>,
}

impl PyRunOutcome {
    pub(crate) fn from_outcome(outcome: RunOutcome) -> Self {
        Self {
            classical_bits: outcome.classical_bits,
            probabilities: outcome.probabilities,
            metadata: PyRunMetadata::new(outcome.metadata),
            saves: outcome.saves,
        }
    }
}

#[pymethods]
impl PyRunOutcome {
    #[getter]
    fn classical_bits(&self) -> Vec<bool> {
        self.classical_bits.clone()
    }

    /// What each save point recorded, in the order the points were reached.
    ///
    /// One dictionary per record with `label`, `kind`, and `value`. A
    /// statevector or density matrix arrives as a `complex128` array and
    /// probabilities as `float64`; a density matrix is flat and row major over
    /// `2^n` rows.
    #[getter]
    fn saves<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for record in &self.saves {
            let entry = PyDict::new(py);
            entry.set_item("label", &record.label)?;
            match &record.value {
                SavedValue::StateVector(amps) => {
                    entry.set_item("kind", "statevector")?;
                    entry.set_item("value", complex_array(py, amps.clone()))?;
                }
                SavedValue::Probabilities(probs) => {
                    entry.set_item("kind", "probabilities")?;
                    entry.set_item("value", f64_array(py, probs.clone()))?;
                }
                SavedValue::DensityMatrix(rho) => {
                    entry.set_item("kind", "density_matrix")?;
                    entry.set_item("value", complex_array(py, rho.clone()))?;
                }
                other => {
                    return Err(PyNotImplementedError::new_err(format!(
                        "saved value {other:?} is newer than this binding"
                    )));
                }
            }
            out.append(entry)?;
        }
        Ok(out)
    }

    /// Probability of each basis state as a `float64` array, or `None` if the
    /// backend cannot expose a dense distribution.
    ///
    /// A factored result is multiplied out here, which costs `2 ** num_qubits`
    /// entries. Read `num_basis_states` first, or take the blocks from
    /// `probabilities_factored()`.
    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.probabilities
            .as_ref()
            .map(|p| f64_array(py, py.detach(|| p.to_vec())))
    }

    /// Per-block distributions of a factored result, or `None` when the run
    /// produced a dense one.
    ///
    /// Each block is `(qubits, probs)`, where `qubits` is ascending and `probs`
    /// is indexed by those qubits packed in that order, `qubits[0]` in the
    /// least significant bit. The joint probability of a basis state is the
    /// product of one entry per block, which is what `probabilities` computes.
    fn probabilities_factored<'py>(&self, py: Python<'py>) -> Option<Vec<FactoredBlockPy<'py>>> {
        let Some(Probabilities::Factored { blocks, .. }) = &self.probabilities else {
            return None;
        };
        Some(
            blocks
                .iter()
                .map(|block| {
                    let qubits = (0..u64::BITS as usize)
                        .filter(|bit| block.mask & (1 << bit) != 0)
                        .collect();
                    (qubits, f64_array(py, block.probs.clone()))
                })
                .collect(),
        )
    }

    /// Basis states the distribution covers, `2 ** num_qubits`, without
    /// materializing it. `None` when the backend exposed no distribution.
    #[getter]
    fn num_basis_states(&self) -> Option<usize> {
        self.probabilities.as_ref().map(Probabilities::len)
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        let form = match &self.probabilities {
            None => "none",
            Some(Probabilities::Dense(_)) => "dense",
            Some(Probabilities::Factored { .. }) => "factored",
            Some(_) => "other",
        };
        format!(
            "RunOutcome(classical_bits={}, probabilities={form})",
            self.classical_bits.len()
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
    /// Shot records as a `(num_shots, num_classical_bits)` bool array.
    ///
    /// Column `i` is classical bit `i`, matching the `counts` key order.
    #[getter]
    fn shots<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let rows = self.inner.num_shots();
        let cols = self.inner.num_classical_bits();
        let mut flat = Vec::with_capacity(rows * cols);
        for shot in &self.inner.shots {
            flat.extend_from_slice(shot);
        }
        bool_matrix(py, rows, cols, flat)
    }

    #[getter]
    fn num_shots(&self) -> usize {
        self.inner.num_shots()
    }

    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.inner.num_classical_bits()
    }

    /// Frequency histogram keyed by bitstring, character `i` being classical
    /// bit `i`. Bit 0 is leftmost, so keys read reversed relative to Qiskit.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        counts_to_dict(py, &self.inner.counts(), self.inner.num_classical_bits())
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        PyRunMetadata::new(self.inner.metadata.clone())
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
    metadata: PyRunMetadata,
}

#[pymethods]
impl PyCountsResult {
    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.num_classical_bits
    }

    /// Frequency histogram keyed by bitstring, character `i` being classical
    /// bit `i`. Bit 0 is leftmost, so keys read reversed relative to Qiskit.
    fn counts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        counts_to_dict(py, &self.counts, self.num_classical_bits)
    }

    #[getter]
    fn metadata(&self) -> PyRunMetadata {
        self.metadata.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "CountsResult(distinct={}, num_classical_bits={})",
            self.counts.len(),
            self.num_classical_bits
        )
    }
}
