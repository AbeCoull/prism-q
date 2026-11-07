//! Python query descriptors and route inspection results.

use prism_q::{PauliObservable, PauliTerm, Plan, PlannedExactness, Query};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::{PyPrismError, PyPrismResult};
use crate::parameter::PyParameters;
use crate::sim::{PyPauliObservable, parse_observables, widest_string};

enum OwnedQuery {
    Run,
    Shots(usize),
    Counts(usize),
    Marginals,
    Expectation(Vec<Vec<PauliTerm>>),
    Observable(PauliObservable),
    StateVector,
    ReducedDensityMatrix(Vec<usize>),
    Entropy(Vec<usize>),
    Gradient(PauliObservable, prism_q::Parameters),
    GradientShift(PauliObservable, prism_q::Parameters),
}

/// Non-consuming terminal description for `Simulation.plan`.
#[pyclass(name = "Query", module = "prism_q")]
pub(crate) struct PyQuery(OwnedQuery, usize);

impl PyQuery {
    /// Characters in the longest Pauli text the query was built from, which the
    /// parsed terms do not record; `Simulation.plan` checks it against the circuit.
    pub(crate) fn text_width(&self) -> usize {
        self.1
    }

    /// The query with its Pauli terms dropped, for planning the route of a query
    /// whose terms the circuit cannot hold.
    pub(crate) fn as_query_without_terms<'a>(&'a self, empty: &'a PauliObservable) -> Query<'a> {
        match &self.0 {
            OwnedQuery::Expectation(_) => Query::Expectation(&[]),
            OwnedQuery::Observable(_) => Query::Observable(empty),
            OwnedQuery::Gradient(_, parameters) => Query::Gradient {
                hamiltonian: empty,
                parameters,
            },
            OwnedQuery::GradientShift(_, parameters) => Query::GradientShift {
                hamiltonian: empty,
                parameters,
            },
            _ => self.as_query(),
        }
    }

    pub(crate) fn as_query(&self) -> Query<'_> {
        match &self.0 {
            OwnedQuery::Run => Query::Run,
            OwnedQuery::Shots(n) => Query::Shots(*n),
            OwnedQuery::Counts(n) => Query::Counts(*n),
            OwnedQuery::Marginals => Query::Marginals,
            OwnedQuery::Expectation(obs) => Query::Expectation(obs),
            OwnedQuery::Observable(obs) => Query::Observable(obs),
            OwnedQuery::StateVector => Query::StateVector,
            OwnedQuery::ReducedDensityMatrix(q) => Query::ReducedDensityMatrix(q),
            OwnedQuery::Entropy(q) => Query::Entropy(q),
            OwnedQuery::Gradient(hamiltonian, parameters) => Query::Gradient {
                hamiltonian,
                parameters,
            },
            OwnedQuery::GradientShift(hamiltonian, parameters) => Query::GradientShift {
                hamiltonian,
                parameters,
            },
        }
    }
}

#[pymethods]
impl PyQuery {
    #[staticmethod]
    fn run() -> Self {
        Self(OwnedQuery::Run, 0)
    }
    #[staticmethod]
    fn shots(num_shots: usize) -> Self {
        Self(OwnedQuery::Shots(num_shots), 0)
    }
    #[staticmethod]
    fn counts(num_shots: usize) -> Self {
        Self(OwnedQuery::Counts(num_shots), 0)
    }
    #[staticmethod]
    fn marginals() -> Self {
        Self(OwnedQuery::Marginals, 0)
    }
    #[staticmethod]
    fn expectation(observables: Vec<crate::sim::PauliString>) -> PyPrismResult<Self> {
        let text_width = widest_string(&observables);
        Ok(Self(
            OwnedQuery::Expectation(parse_observables(observables)?),
            text_width,
        ))
    }
    #[staticmethod]
    fn observable(observable: &PyPauliObservable) -> Self {
        Self(OwnedQuery::Observable(observable.0.clone()), observable.1)
    }
    #[staticmethod]
    fn state_vector() -> Self {
        Self(OwnedQuery::StateVector, 0)
    }
    #[staticmethod]
    fn reduced_density_matrix(qubits: Vec<usize>) -> Self {
        Self(OwnedQuery::ReducedDensityMatrix(qubits), 0)
    }
    #[staticmethod]
    fn entropy(qubits: Vec<usize>) -> Self {
        Self(OwnedQuery::Entropy(qubits), 0)
    }
    #[staticmethod]
    fn gradient(
        hamiltonian: crate::sim::Hamiltonian<'_>,
        parameters: &PyParameters,
    ) -> PyPrismResult<Self> {
        Ok(Self(
            OwnedQuery::Gradient(hamiltonian.observable()?.into_owned(), parameters.0.clone()),
            hamiltonian.text_width(),
        ))
    }
    #[staticmethod]
    fn gradient_shift(
        hamiltonian: crate::sim::Hamiltonian<'_>,
        parameters: &PyParameters,
    ) -> PyPrismResult<Self> {
        Ok(Self(
            OwnedQuery::GradientShift(hamiltonian.observable()?.into_owned(), parameters.0.clone()),
            hamiltonian.text_width(),
        ))
    }
}

pub(crate) fn rule_name(rule: prism_q::RouteRule) -> String {
    let mut snake = String::new();
    for (i, c) in format!("{rule:?}").chars().enumerate() {
        if c.is_uppercase() && i != 0 {
            snake.push('_');
        }
        snake.extend(c.to_lowercase());
    }
    snake
}

fn exactness_name(exactness: PlannedExactness) -> &'static str {
    match exactness {
        PlannedExactness::Exact => "exact",
        PlannedExactness::Approximate { .. } => "approximate",
        PlannedExactness::Estimated { .. } => "estimated",
        _ => "unknown",
    }
}

/// Route, accuracy and predictable failures before circuit execution.
#[pyclass(name = "Plan", module = "prism_q")]
pub(crate) struct PyPlan(pub(crate) Plan);

#[pymethods]
impl PyPlan {
    #[getter]
    fn evaluations(&self) -> Vec<PyPlan> {
        self.0.evaluations.iter().cloned().map(PyPlan).collect()
    }
    #[getter]
    fn backend(&self) -> String {
        format!("{:?}", self.0.backend)
    }
    #[getter]
    fn engine(&self) -> Option<String> {
        self.0.engine.map(|e| format!("{e:?}"))
    }
    #[getter]
    fn rule(&self) -> String {
        rule_name(self.0.rule)
    }
    #[getter]
    fn placement(&self) -> &'static str {
        if self.0.placement == prism_q::Placement::Device {
            "device"
        } else {
            "host"
        }
    }
    #[getter]
    fn exactness(&self) -> &'static str {
        exactness_name(self.0.exactness)
    }
    #[getter]
    fn bound_reported(&self) -> bool {
        matches!(
            self.0.exactness,
            PlannedExactness::Approximate {
                bound_reported: true
            }
        )
    }
    #[getter]
    fn samples(&self) -> Option<usize> {
        if let PlannedExactness::Estimated { samples } = self.0.exactness {
            Some(samples)
        } else {
            None
        }
    }
    #[getter]
    fn peak_memory(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        result.set_item("bytes", self.0.peak_memory.bytes)?;
        result.set_item(
            "kind",
            match self.0.peak_memory.kind {
                prism_q::MemoryEstimateKind::Exact => "exact",
                prism_q::MemoryEstimateKind::UpperBound => "upper_bound",
                _ => "unknown",
            },
        )?;
        Ok(result.unbind())
    }
    #[getter]
    fn would_fail(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0.would_fail.clone().map(|error| {
            let error: PyErr = PyPrismError(error).into();
            error.value(py).clone().unbind().into_any()
        })
    }
    #[getter]
    fn alternatives(&self, py: Python<'_>) -> Vec<(String, Py<PyAny>)> {
        self.0
            .alternatives
            .iter()
            .map(|(backend, reason)| {
                let error: PyErr = PyPrismError(reason.error.clone()).into();
                (
                    format!("{backend:?}"),
                    error.value(py).clone().unbind().into_any(),
                )
            })
            .collect()
    }
    #[getter]
    fn fallbacks(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.0
            .fallbacks
            .iter()
            .map(|fallback| {
                let result = PyDict::new(py);
                let failure = fallback.would_fail.clone().map(|error| {
                    let error: PyErr = PyPrismError(error).into();
                    error.value(py).clone().unbind().into_any()
                });
                result.set_item("would_fail", failure)?;
                result.set_item("backend", format!("{:?}", fallback.backend))?;
                result.set_item("rule", rule_name(fallback.rule))?;
                result.set_item("exactness", exactness_name(fallback.exactness))?;
                result.set_item(
                    "bound_reported",
                    matches!(
                        fallback.exactness,
                        PlannedExactness::Approximate {
                            bound_reported: true
                        }
                    ),
                )?;
                result.set_item(
                    "samples",
                    if let PlannedExactness::Estimated { samples } = fallback.exactness {
                        Some(samples)
                    } else {
                        None
                    },
                )?;
                result.set_item(
                    "placement",
                    if fallback.placement == prism_q::Placement::Device {
                        "device"
                    } else {
                        "host"
                    },
                )?;
                Ok(result.unbind())
            })
            .collect()
    }
}
