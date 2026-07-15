//! Circuit construction, OpenQASM parsing, and reusable circuit builders.

use prism_q::circuit::openqasm;
use prism_q::{Circuit, CircuitBuilder, Gate, circuits};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::error::{PyPrismResult, invalid};
use crate::gate::PyGate;

/// A quantum circuit. Construct via [`CircuitBuilder`], [`parse_qasm`], or one
/// of the reusable circuit generators under `prism_q.circuits`.
#[pyclass(name = "Circuit", module = "prism_q", from_py_object)]
#[derive(Clone)]
pub struct PyCircuit(pub Circuit);

fn check_qubit(
    num_qubits: usize,
    qubit: usize,
    label: impl std::fmt::Display,
) -> PyPrismResult<()> {
    if qubit >= num_qubits {
        return Err(invalid(format!(
            "{label} {qubit} out of range (circuit has {num_qubits} qubits)"
        )));
    }
    Ok(())
}

fn check_classical_bit(num_bits: usize, bit: usize) -> PyPrismResult<()> {
    if bit >= num_bits {
        return Err(invalid(format!(
            "classical bit {bit} out of range (circuit has {num_bits} classical bits)"
        )));
    }
    Ok(())
}

fn check_targets(num_qubits: usize, gate: &Gate, targets: &[usize]) -> PyPrismResult<()> {
    let expected = gate.num_qubits();
    if targets.len() != expected {
        return Err(invalid(format!(
            "gate `{}` expects {expected} qubits, got {}",
            gate.name(),
            targets.len()
        )));
    }
    for (idx, &target) in targets.iter().enumerate() {
        check_qubit(num_qubits, target, format!("target[{idx}]"))?;
    }
    Ok(())
}

fn check_mcu_targets(num_qubits: usize, controls: &[usize], target: usize) -> PyPrismResult<()> {
    if controls.is_empty() {
        return Err(invalid("mcu requires at least one control qubit"));
    }
    if controls.len() > u8::MAX as usize {
        return Err(invalid(format!(
            "mcu supports at most {} control qubits, got {}",
            u8::MAX,
            controls.len()
        )));
    }
    for (idx, &control) in controls.iter().enumerate() {
        check_qubit(num_qubits, control, format!("control[{idx}]"))?;
    }
    check_qubit(num_qubits, target, "target")?;
    Ok(())
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (num_qubits, num_classical_bits = 0))]
    fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self(Circuit::new(num_qubits, num_classical_bits))
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.0.num_qubits
    }

    #[getter]
    fn num_classical_bits(&self) -> usize {
        self.0.num_classical_bits
    }

    fn gate_count(&self) -> usize {
        self.0.gate_count()
    }

    fn t_count(&self) -> usize {
        self.0.t_count()
    }

    fn is_clifford_only(&self) -> bool {
        self.0.is_clifford_only()
    }

    /// Append a gate acting on `targets`.
    fn add_gate(&mut self, gate: &PyGate, targets: Vec<usize>) -> PyPrismResult<()> {
        check_targets(self.0.num_qubits, gate.inner(), &targets)?;
        self.0.add_gate(gate.inner().clone(), &targets);
        Ok(())
    }

    /// Append a measurement of `qubit` into `classical_bit`.
    fn add_measure(&mut self, qubit: usize, classical_bit: usize) -> PyPrismResult<()> {
        check_qubit(self.0.num_qubits, qubit, "qubit")?;
        check_classical_bit(self.0.num_classical_bits, classical_bit)?;
        self.0.add_measure(qubit, classical_bit);
        Ok(())
    }

    /// Append a reset of `qubit` to |0>.
    fn add_reset(&mut self, qubit: usize) -> PyPrismResult<()> {
        check_qubit(self.0.num_qubits, qubit, "qubit")?;
        self.0.add_reset(qubit);
        Ok(())
    }

    /// Append a barrier across `qubits`.
    fn add_barrier(&mut self, qubits: Vec<usize>) -> PyPrismResult<()> {
        for (idx, &qubit) in qubits.iter().enumerate() {
            check_qubit(self.0.num_qubits, qubit, format!("qubits[{idx}]"))?;
        }
        self.0.add_barrier(&qubits);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "Circuit(num_qubits={}, num_classical_bits={}, gates={})",
            self.0.num_qubits,
            self.0.num_classical_bits,
            self.0.gate_count()
        )
    }
}

impl PyCircuit {
    pub fn inner(&self) -> &Circuit {
        &self.0
    }
}

/// Fluent builder for quantum circuits. Gate methods return the builder for
/// chaining; call [`build`](Self::build) to extract the [`Circuit`].
#[pyclass(name = "CircuitBuilder", module = "prism_q")]
pub struct PyCircuitBuilder {
    inner: CircuitBuilder,
}

#[pymethods]
impl PyCircuitBuilder {
    #[new]
    #[pyo3(signature = (num_qubits, num_classical_bits = 0))]
    fn new(num_qubits: usize, num_classical_bits: usize) -> Self {
        Self {
            inner: CircuitBuilder::new_with_classical(num_qubits, num_classical_bits),
        }
    }

    fn id(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.id(q);
        Ok(slf)
    }
    fn x(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.x(q);
        Ok(slf)
    }
    fn y(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.y(q);
        Ok(slf)
    }
    fn z(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.z(q);
        Ok(slf)
    }
    fn h(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.h(q);
        Ok(slf)
    }
    fn s(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.s(q);
        Ok(slf)
    }
    fn sdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sdg(q);
        Ok(slf)
    }
    fn t(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.t(q);
        Ok(slf)
    }
    fn tdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.tdg(q);
        Ok(slf)
    }
    fn sx(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sx(q);
        Ok(slf)
    }
    fn sxdg(mut slf: PyRefMut<'_, Self>, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.sxdg(q);
        Ok(slf)
    }

    fn rx(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.rx(theta, q);
        Ok(slf)
    }
    fn ry(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.ry(theta, q);
        Ok(slf)
    }
    fn rz(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.rz(theta, q);
        Ok(slf)
    }
    fn p(mut slf: PyRefMut<'_, Self>, theta: f64, q: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        check_qubit(slf.inner.circuit().num_qubits, q, "qubit")?;
        slf.inner.p(theta, q);
        Ok(slf)
    }

    fn rzz(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.rzz(theta, q0, q1);
        Ok(slf)
    }

    fn cx(
        mut slf: PyRefMut<'_, Self>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        slf.inner.cx(control, target);
        Ok(slf)
    }

    fn cz(mut slf: PyRefMut<'_, Self>, q0: usize, q1: usize) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.cz(q0, q1);
        Ok(slf)
    }

    fn swap(
        mut slf: PyRefMut<'_, Self>,
        q0: usize,
        q1: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, q0, "q0")?;
        check_qubit(num_qubits, q1, "q1")?;
        slf.inner.swap(q0, q1);
        Ok(slf)
    }

    fn cphase(
        mut slf: PyRefMut<'_, Self>,
        theta: f64,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        slf.inner.cphase(theta, control, target);
        Ok(slf)
    }

    fn cu<'py>(
        mut slf: PyRefMut<'py, Self>,
        matrix: &Bound<'_, PyAny>,
        control: usize,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        check_qubit(num_qubits, control, "control")?;
        check_qubit(num_qubits, target, "target")?;
        let mat = crate::gate::extract_2x2(matrix)?;
        slf.inner.cu(mat, control, target);
        Ok(slf)
    }

    fn mcu<'py>(
        mut slf: PyRefMut<'py, Self>,
        matrix: &Bound<'_, PyAny>,
        controls: Vec<usize>,
        target: usize,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        check_mcu_targets(slf.inner.circuit().num_qubits, &controls, target)?;
        let mat = crate::gate::extract_2x2(matrix)?;
        slf.inner.mcu(mat, &controls, target);
        Ok(slf)
    }

    fn measure(
        mut slf: PyRefMut<'_, Self>,
        qubit: usize,
        classical_bit: usize,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let circuit = slf.inner.circuit();
        check_qubit(circuit.num_qubits, qubit, "qubit")?;
        check_classical_bit(circuit.num_classical_bits, classical_bit)?;
        slf.inner.measure(qubit, classical_bit);
        Ok(slf)
    }

    fn measure_all(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.measure_all();
        slf
    }

    fn barrier(
        mut slf: PyRefMut<'_, Self>,
        qubits: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'_, Self>> {
        let num_qubits = slf.inner.circuit().num_qubits;
        for (idx, &qubit) in qubits.iter().enumerate() {
            check_qubit(num_qubits, qubit, format!("qubits[{idx}]"))?;
        }
        slf.inner.barrier(&qubits);
        Ok(slf)
    }

    /// Append a gate acting on `targets`.
    fn gate<'py>(
        mut slf: PyRefMut<'py, Self>,
        gate: &PyGate,
        targets: Vec<usize>,
    ) -> PyPrismResult<PyRefMut<'py, Self>> {
        check_targets(slf.inner.circuit().num_qubits, gate.inner(), &targets)?;
        slf.inner.gate(gate.inner().clone(), &targets);
        Ok(slf)
    }

    /// Extract the finished circuit. The builder remains usable, so `build()`
    /// can be called repeatedly and gate methods may be chained afterward.
    fn build(&self) -> PyCircuit {
        PyCircuit(self.inner.circuit().clone())
    }
}

/// Parse an OpenQASM 3.0 (or 2.0-compatible) string into a [`Circuit`].
#[pyfunction]
pub fn parse_qasm(source: &str) -> PyPrismResult<PyCircuit> {
    Ok(PyCircuit(openqasm::parse(source)?))
}

macro_rules! circuit_fn {
    ($name:ident, $call:path, ($($arg:ident : $ty:ty),*), ($($sig:tt)*)) => {
        #[pyfunction]
        #[pyo3(signature = ($($sig)*))]
        fn $name($($arg : $ty),*) -> PyCircuit {
            PyCircuit($call($($arg),*))
        }
    };
}

circuit_fn!(qft, circuits::qft_circuit, (n: usize), (n));
circuit_fn!(random, circuits::random_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(hardware_efficient_ansatz, circuits::hardware_efficient_ansatz, (n: usize, layers: usize, seed: u64), (n, layers, seed = 42));
circuit_fn!(clifford_heavy, circuits::clifford_heavy_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(clifford_random_pairs, circuits::clifford_random_pairs, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(qaoa, circuits::qaoa_circuit, (n: usize, layers: usize, seed: u64), (n, layers, seed = 42));
circuit_fn!(single_qubit_rotation, circuits::single_qubit_rotation_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(quantum_volume, circuits::quantum_volume_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(cz_chain, circuits::cz_chain_circuit, (n: usize, depth: usize, seed: u64), (n, depth, seed = 42));
circuit_fn!(phase_estimation, circuits::phase_estimation_circuit, (n: usize), (n));
circuit_fn!(independent_bell_pairs, circuits::independent_bell_pairs, (n_pairs: usize), (n_pairs));
circuit_fn!(independent_random_blocks, circuits::independent_random_blocks, (num_blocks: usize, block_size: usize, depth: usize, seed: u64), (num_blocks, block_size, depth, seed = 42));
circuit_fn!(local_clifford_blocks, circuits::local_clifford_blocks, (num_blocks: usize, block_size: usize, depth: usize, seed: u64), (num_blocks, block_size, depth, seed = 42));

#[pyfunction]
#[pyo3(signature = (n, depth, t_fraction = 0.1, seed = 42))]
fn clifford_t(n: usize, depth: usize, t_fraction: f64, seed: u64) -> PyCircuit {
    PyCircuit(circuits::clifford_t_circuit(n, depth, t_fraction, seed))
}

#[pyfunction]
fn ghz(n: usize) -> PyPrismResult<PyCircuit> {
    if n == 0 {
        return Err(invalid("ghz requires at least one qubit"));
    }
    Ok(PyCircuit(circuits::ghz_circuit(n)))
}

#[pyfunction]
fn w_state(n: usize) -> PyPrismResult<PyCircuit> {
    if n == 0 {
        return Err(invalid("w_state requires at least one qubit"));
    }
    Ok(PyCircuit(circuits::w_state_circuit(n)))
}

/// Build and register the `prism_q.circuits` submodule.
pub fn register_circuits(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "circuits")?;
    m.add_function(wrap_pyfunction!(qft, &m)?)?;
    m.add_function(wrap_pyfunction!(ghz, &m)?)?;
    m.add_function(wrap_pyfunction!(w_state, &m)?)?;
    m.add_function(wrap_pyfunction!(random, &m)?)?;
    m.add_function(wrap_pyfunction!(hardware_efficient_ansatz, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_heavy, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_random_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(qaoa, &m)?)?;
    m.add_function(wrap_pyfunction!(single_qubit_rotation, &m)?)?;
    m.add_function(wrap_pyfunction!(clifford_t, &m)?)?;
    m.add_function(wrap_pyfunction!(quantum_volume, &m)?)?;
    m.add_function(wrap_pyfunction!(cz_chain, &m)?)?;
    m.add_function(wrap_pyfunction!(phase_estimation, &m)?)?;
    m.add_function(wrap_pyfunction!(independent_bell_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(independent_random_blocks, &m)?)?;
    m.add_function(wrap_pyfunction!(local_clifford_blocks, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
