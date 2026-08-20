//! Parameter slots over a circuit's rotation angles, and the circuit held
//! across bindings that reuses the fusion and dispatch work they imply.

use std::sync::Mutex;

use prism_q::{BackendKind, Parameters, PreparedCircuit, RunOutcome};
use pyo3::prelude::*;

use crate::backend::PyBackendKind;
use crate::circuit::PyCircuit;
use crate::error::{PyPrismResult, invalid};
use crate::sim::{DEFAULT_SEED, PyRunOutcome};

/// Parameter slots over the rotation angles of a circuit.
///
/// A slot may drive several instructions: `bind` writes one angle to each and
/// the adjoint gradient accumulates their contributions into one entry.
/// Bindable gates are `rx`, `ry`, `rz`, `rzz`, `p`, and `pauli_rot`.
#[pyclass(name = "Parameters", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub struct PyParameters(pub Parameters);

#[pymethods]
impl PyParameters {
    /// Declare `num_slots` slots with no links yet.
    #[new]
    fn new(num_slots: usize) -> Self {
        Self(Parameters::new(num_slots))
    }

    /// Give every bindable gate in `circuit` its own slot, in circuit order.
    #[staticmethod]
    fn all_rotations(circuit: &PyCircuit) -> Self {
        Self(Parameters::all_rotations(circuit.inner()))
    }

    /// Record that instruction `instruction` reads `slot`.
    fn link(&mut self, instruction: usize, slot: usize) -> PyPrismResult<()> {
        if slot >= self.0.num_slots() {
            return Err(invalid(format!(
                "slot {slot} out of bounds (parameter set declares {} slots)",
                self.0.num_slots()
            )));
        }
        self.0.link(instruction, slot);
        Ok(())
    }

    /// A copy naming the slots in slot order. Names are what OpenQASM `input`
    /// declarations carry.
    fn with_names(&self, names: Vec<String>) -> PyPrismResult<Self> {
        if names.len() != self.0.num_slots() {
            return Err(invalid(format!(
                "expected {} slot names, got {}",
                self.0.num_slots(),
                names.len()
            )));
        }
        Ok(Self(self.0.clone().with_names(names)))
    }

    /// Length of the value vector `bind` expects.
    #[getter]
    fn num_slots(&self) -> usize {
        self.0.num_slots()
    }

    /// True when no instruction is linked. A set may still declare slots.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The recorded `(instruction_index, slot)` links, in the order added.
    fn links(&self) -> Vec<(usize, usize)> {
        self.0
            .links()
            .iter()
            .map(|link| (link.instruction, link.slot))
            .collect()
    }

    /// Name of `slot`, or `None` when the set is positional only.
    fn name_of(&self, slot: usize) -> Option<String> {
        self.0.name_of(slot).map(str::to_owned)
    }

    /// Slot a name refers to.
    fn slot_of(&self, name: &str) -> Option<usize> {
        self.0.slot_of(name)
    }

    /// Declared slots that no instruction reads, whose bound values are
    /// discarded. Legal, but usually a mistake worth surfacing.
    fn unread_slots(&self) -> Vec<usize> {
        self.0.unread_slots()
    }

    /// Check every link against `circuit` without binding.
    fn validate(&self, circuit: &PyCircuit) -> PyPrismResult<()> {
        self.0.validate(circuit.inner())?;
        Ok(())
    }

    /// Write `values` into a copy of `template`.
    fn bind(&self, template: &PyCircuit, values: Vec<f64>) -> PyPrismResult<PyCircuit> {
        Ok(PyCircuit(self.0.bind(template.inner(), &values)?))
    }

    /// Read the angle each slot currently holds in `circuit`.
    fn values(&self, circuit: &PyCircuit) -> PyPrismResult<Vec<f64>> {
        Ok(self.0.values(circuit.inner())?)
    }

    fn __repr__(&self) -> String {
        format!(
            "Parameters(num_slots={}, links={})",
            self.0.num_slots(),
            self.0.links().len()
        )
    }
}

/// A parameter template plus the fusion and dispatch work its structure
/// implies, held across bindings.
///
/// A variational sweep binds a new angle vector per point while the gate
/// sequence stays fixed, so fusion decides the same block structure every time
/// and dispatch picks the same backend. The constructor settles both once and
/// `run` rebuilds only what the angles change.
///
/// Automatic dispatch reads the template, so build it at angles representative
/// of the sweep. A template whose rotations are all zero reads as Clifford and
/// settles on a backend that then rejects the bound circuit.
///
/// The held backend is `Send` but not `Sync`, so the lock is what lets `run`
/// release the GIL rather than serialize the interpreter behind a sweep.
#[pyclass(name = "PreparedCircuit", module = "prism_q")]
pub struct PyPreparedCircuit {
    inner: Mutex<PreparedCircuit>,
}

impl PyPreparedCircuit {
    /// The lock is only ever taken by the methods below and never held across a
    /// call into Python, so poisoning means a panic already unwound through one
    /// of them and the prepared state is not worth recovering.
    fn locked(&self) -> std::sync::MutexGuard<'_, PreparedCircuit> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl PyPreparedCircuit {
    /// Settle the fused structure and the backend choice for `circuit` under
    /// `parameters`, with automatic backend selection unless `backend` is given.
    #[new]
    #[pyo3(signature = (circuit, parameters, backend = None))]
    fn new(
        circuit: &PyCircuit,
        parameters: &PyParameters,
        backend: Option<PyBackendKind>,
    ) -> PyPrismResult<Self> {
        let kind = backend.map_or(BackendKind::Auto, |b| b.0);
        Ok(Self {
            inner: Mutex::new(PreparedCircuit::with_backend(
                circuit.inner().clone(),
                parameters.0.clone(),
                kind,
            )?),
        })
    }

    /// Bind `values` and return the unfused circuit, ready for `simulate` or
    /// any other consumer of a plain `Circuit`.
    fn bind(&self, values: Vec<f64>) -> PyPrismResult<PyCircuit> {
        Ok(PyCircuit(self.locked().bind(&values)?.clone()))
    }

    /// Bind `values` and execute, reusing the settled backend choice and, where
    /// the backend accepts fused gates, the fusion plan.
    #[pyo3(signature = (values, seed = DEFAULT_SEED))]
    fn run(&self, py: Python<'_>, values: Vec<f64>, seed: u64) -> PyPrismResult<PyRunOutcome> {
        let outcome: RunOutcome = py.detach(|| self.locked().run(&values, seed))?;
        Ok(PyRunOutcome::from_outcome(outcome))
    }

    /// The unbound template this was built from.
    #[getter]
    fn template(&self) -> PyCircuit {
        PyCircuit(self.locked().template().clone())
    }

    #[getter]
    fn parameters(&self) -> PyParameters {
        PyParameters(self.locked().parameters().clone())
    }

    /// True when the fused structure was captured and bindings reuse it, false
    /// when every binding re-runs the pass pipeline. A performance fact, not an
    /// error: results agree either way.
    #[getter]
    fn reuses_fusion_plan(&self) -> bool {
        self.locked().reuses_fusion_plan()
    }

    fn __repr__(&self) -> String {
        let inner = self.locked();
        format!(
            "PreparedCircuit(num_qubits={}, num_slots={}, reuses_fusion_plan={})",
            inner.template().num_qubits,
            inner.parameters().num_slots(),
            inner.reuses_fusion_plan()
        )
    }
}
