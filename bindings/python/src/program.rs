//! Dynamic programs: a control-flow graph of circuits with runtime classical
//! variables, built here or parsed from OpenQASM, and run once per shot.

use prism_q::circuit::dynamic::{ClassicalExpr, ClassicalType, ClassicalValue, RotationKind};
use prism_q::circuit::openqasm;
use prism_q::{
    BackendKind, CountsResult, DynamicProgram, DynamicProgramBuilder, ShotsResult,
    simulate_program as core_simulate_program,
};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyString};

use crate::backend::PyBackendKind;
use crate::circuit::PyCircuit;
use crate::error::{PyPrismResult, invalid};
use crate::gate::PyGate;
use crate::sim::{DEFAULT_SEED, PyCountsResult, PyShotsResult};

/// A program whose instruction count can depend on measurement outcomes: basic
/// blocks of circuits joined by branches on classical expressions.
#[pyclass(
    name = "DynamicProgram",
    module = "prism_q",
    frozen,
    skip_from_py_object
)]
pub struct PyDynamicProgram(DynamicProgram);

#[pymethods]
impl PyDynamicProgram {
    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }

    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.0.num_classical_bits()
    }

    #[getter]
    fn num_blocks(&self) -> usize {
        self.0.blocks().len()
    }

    /// Runtime variable names in declaration order.
    #[getter]
    fn variables(&self) -> Vec<String> {
        self.0.variables().iter().map(|v| v.name.clone()).collect()
    }

    /// The circuit a program without runtime control flow runs, else `None`.
    fn static_circuit(&self) -> Option<PyCircuit> {
        self.0.static_circuit().cloned().map(PyCircuit)
    }

    fn __repr__(&self) -> String {
        format!(
            "DynamicProgram(num_qubits={}, num_classical_bits={}, blocks={}, variables={})",
            self.0.num_qubits(),
            self.0.num_classical_bits(),
            self.0.blocks().len(),
            self.0.variables().len()
        )
    }
}

/// Structured builder for a `DynamicProgram`. Conditions, assigned values and
/// rotation angles are OpenQASM expression strings over the declared variables
/// and the classical bits (`c[i]` for one bit, `c` for all of them as an
/// unsigned integer), or plain `bool`, `int` and `float` constants.
#[pyclass(name = "DynamicProgramBuilder", module = "prism_q")]
pub struct PyDynamicProgramBuilder {
    inner: Option<DynamicProgramBuilder>,
}

impl PyDynamicProgramBuilder {
    fn inner(&mut self) -> PyPrismResult<&mut DynamicProgramBuilder> {
        self.inner
            .as_mut()
            .ok_or_else(|| invalid("the builder has already built its program"))
    }

    fn expr(&mut self, value: &Bound<'_, PyAny>) -> PyPrismResult<ClassicalExpr> {
        let builder = self.inner()?;
        if let Ok(text) = value.cast::<PyString>() {
            let text = text.to_string();
            return Ok(builder.expr(&text)?);
        }
        Ok(ClassicalExpr::Const(constant(value)?))
    }
}

fn constant(value: &Bound<'_, PyAny>) -> PyPrismResult<ClassicalValue> {
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(ClassicalValue::Bool(flag.is_true()));
    }
    if value.is_instance_of::<PyInt>() {
        let int: i64 = value
            .extract()
            .map_err(|_| invalid("integer constant does not fit in 64 bits"))?;
        return Ok(ClassicalValue::from(int));
    }
    if value.is_instance_of::<PyFloat>() {
        let float: f64 = value
            .extract()
            .map_err(|_| invalid("float constant could not be read"))?;
        return Ok(ClassicalValue::Float(float));
    }
    Err(invalid(
        "expected an expression string or a bool, int or float constant",
    ))
}

fn classical_type(text: &str) -> PyPrismResult<ClassicalType> {
    let text = text.trim();
    let (base, width) = match text.split_once('[') {
        Some((base, rest)) => {
            let width = rest
                .strip_suffix(']')
                .and_then(|w| w.trim().parse::<u32>().ok())
                .ok_or_else(|| invalid(format!("malformed type `{text}`")))?;
            (base.trim(), Some(width))
        }
        None => (text, None),
    };
    match (base, width) {
        ("bool", None) => Ok(ClassicalType::Bool),
        ("float", None) => Ok(ClassicalType::Float),
        ("angle", None) => Ok(ClassicalType::Angle),
        ("int", width) => Ok(ClassicalType::Int {
            width: width.unwrap_or(64),
        }),
        ("uint", width) => Ok(ClassicalType::Uint {
            width: width.unwrap_or(64),
        }),
        _ => Err(invalid(format!(
            "unknown type `{text}`; expected bool, int[n], uint[n], float or angle"
        ))),
    }
}

fn rotation_kind(name: &str) -> PyPrismResult<RotationKind> {
    match name {
        "rx" => Ok(RotationKind::Rx),
        "ry" => Ok(RotationKind::Ry),
        "rz" => Ok(RotationKind::Rz),
        "p" | "phase" => Ok(RotationKind::Phase),
        "rzz" => Ok(RotationKind::Rzz),
        _ => Err(invalid(format!(
            "unknown rotation `{name}`; expected rx, ry, rz, p or rzz"
        ))),
    }
}

#[pymethods]
impl PyDynamicProgramBuilder {
    #[new]
    #[pyo3(signature = (num_qubits, num_classical_bits = 0))]
    fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            inner: Some(DynamicProgramBuilder::new(num_qubits, num_classical_bits)),
        }
    }

    /// Declare a runtime variable every shot starts at `initial`. `ty` is one of
    /// `"bool"`, `"int[n]"`, `"uint[n]"` (64 bits without `[n]`), `"float"` or
    /// `"angle"`.
    #[pyo3(signature = (name, ty, initial = None))]
    fn declare<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        ty: &str,
        initial: Option<&Bound<'_, PyAny>>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let ty = classical_type(ty)?;
        let initial = match initial {
            Some(value) => constant(value)?,
            None => match ty {
                ClassicalType::Bool => ClassicalValue::Bool(false),
                ClassicalType::Float | ClassicalType::Angle => ClassicalValue::Float(0.0),
                _ => ClassicalValue::from(0i64),
            },
        };
        slf.inner()?.declare(name, ty, initial);
        Ok(slf)
    }

    fn add_gate<'py>(
        mut slf: PyRefMut<'py, Self>,
        gate: &PyGate,
        targets: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        slf.inner()?.add_gate(gate.inner().clone(), &targets);
        Ok(slf)
    }

    fn add_measure(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        classical_bit: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.add_measure(qubit, classical_bit);
        Ok(slf)
    }

    fn add_reset(mut slf: PyRefMut<'_, Self>, qubit: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.add_reset(qubit);
        Ok(slf)
    }

    /// Append every instruction of `circuit`.
    fn append<'py>(
        mut slf: PyRefMut<'py, Self>,
        circuit: &PyCircuit,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        slf.inner()?.append(circuit.inner());
        Ok(slf)
    }

    /// Apply `kind` (`"rx"`, `"ry"`, `"rz"`, `"p"` or `"rzz"`) at the angle
    /// `angle` evaluates to when the block runs.
    fn add_rotation<'py>(
        mut slf: PyRefMut<'py, Self>,
        kind: &str,
        targets: Vec<usize>,
        angle: &Bound<'_, PyAny>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let kind = rotation_kind(kind)?;
        let angle = slf.expr(angle)?;
        slf.inner()?.add_rotation(kind, &targets, angle);
        Ok(slf)
    }

    /// Store `value` into the variable `name`, converted to its declared type.
    fn assign<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let var = slf
            .inner()?
            .variable(name)
            .ok_or_else(|| invalid(format!("no variable named `{name}`")))?;
        let value = slf.expr(value)?;
        slf.inner()?.assign(var, value);
        Ok(slf)
    }

    /// Open a loop that runs while `condition` holds; `name` labels it in the
    /// step-limit error. Close it with `end()`.
    fn begin_while<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: String,
        condition: &Bound<'_, PyAny>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let condition = slf.expr(condition)?;
        slf.inner()?.begin_while(name, condition);
        Ok(slf)
    }

    /// Open a branch taken when `condition` holds. Close it with `end()`.
    fn begin_if<'py>(
        mut slf: PyRefMut<'py, Self>,
        condition: &Bound<'_, PyAny>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let condition = slf.expr(condition)?;
        slf.inner()?.begin_if(condition);
        Ok(slf)
    }

    /// Switch the open `if` to its other arm, or reopen the `if` `end()` just
    /// closed.
    fn begin_else(mut slf: PyRefMut<'_, Self>) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.begin_else()?;
        Ok(slf)
    }

    fn end(mut slf: PyRefMut<'_, Self>) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.end()?;
        Ok(slf)
    }

    fn break_loop(mut slf: PyRefMut<'_, Self>) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.break_loop()?;
        Ok(slf)
    }

    fn continue_loop(mut slf: PyRefMut<'_, Self>) -> PyPrismResult<PyRefMut<'_, Self>> {
        slf.inner()?.continue_loop()?;
        Ok(slf)
    }

    /// Validate and return the program. The builder cannot be used afterwards.
    fn build(&mut self) -> PyPrismResult<PyDynamicProgram> {
        let builder = self
            .inner
            .take()
            .ok_or_else(|| invalid("the builder has already built its program"))?;
        Ok(PyDynamicProgram(builder.build()?))
    }
}

/// A configured run of a `DynamicProgram`. Set options with `.seed()`,
/// `.backend()` and `.max_steps()`, then call `shots` or `sample_counts`.
#[pyclass(name = "ProgramSimulation", module = "prism_q")]
pub struct PyProgramSimulation {
    program: Py<PyDynamicProgram>,
    seed: Option<u64>,
    kind: Option<BackendKind>,
    max_steps: Option<u64>,
}

impl PyProgramSimulation {
    fn run<T: Send>(
        &self,
        py: Python<'_>,
        terminal: impl FnOnce(prism_q::SimulateProgram<'_, prism_q::Seeded>) -> prism_q::Result<T>
        + Send,
    ) -> PyPrismResult<T> {
        let program = &self.program.get().0;
        let seed = self.seed.unwrap_or(DEFAULT_SEED);
        let kind = self.kind.clone();
        let max_steps = self.max_steps;
        Ok(py.detach(|| {
            let mut sim = core_simulate_program(program);
            if let Some(k) = kind {
                sim = sim.backend(k);
            }
            if let Some(bound) = max_steps {
                sim = sim.max_steps(bound);
            }
            terminal(sim.seed(seed))
        })?)
    }
}

#[pymethods]
impl PyProgramSimulation {
    /// Set the random seed (default 42). Shot `i` draws from the seed a circuit
    /// run gives shot `i`.
    fn seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.seed = Some(seed);
        slf
    }

    fn backend(mut slf: PyRefMut<'_, Self>, kind: PyBackendKind) -> PyRefMut<'_, Self> {
        slf.kind = Some(kind.0);
        slf
    }

    /// Bound the blocks one shot may run (default 1,000,000). A shot past it
    /// raises `PrismError` with `kind == "step_limit"` naming the loop.
    fn max_steps(mut slf: PyRefMut<'_, Self>, max_steps: u64) -> PyRefMut<'_, Self> {
        slf.max_steps = Some(max_steps);
        slf
    }

    fn shots(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyShotsResult> {
        let result: ShotsResult = self.run(py, |sim| sim.shots(num_shots))?;
        Ok(PyShotsResult::from_result(result))
    }

    fn sample_counts(&self, py: Python<'_>, num_shots: usize) -> PyPrismResult<PyCountsResult> {
        let result: CountsResult = self.run(py, |sim| sim.sample_counts(num_shots))?;
        Ok(PyCountsResult::from_result(result))
    }
}

/// Parse OpenQASM 3.0 with `while`, `break`, `continue` and runtime classical
/// values into a `DynamicProgram`.
#[pyfunction]
pub fn parse_qasm_dynamic(source: &str) -> PyPrismResult<PyDynamicProgram> {
    Ok(PyDynamicProgram(openqasm::parse_dynamic(source)?))
}

/// Start a `ProgramSimulation` for `program`.
#[pyfunction]
pub fn simulate_program(program: Py<PyDynamicProgram>) -> PyProgramSimulation {
    PyProgramSimulation {
        program,
        seed: None,
        kind: None,
        max_steps: None,
    }
}
