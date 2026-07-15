//! Native QEC programs: construction, sampling, and packed-shot output.

use numpy::PyArray2;
use prism_q::{
    PackedShots, QecBasis, QecNoise, QecOptions, QecPauli, QecProgram, QecRecordRef,
    QecSampleResult, run_qec_program,
};
use pyo3::prelude::*;

use crate::error::PyPrismResult;
use crate::gate::PyGate;
use crate::numpy_util::bool_matrix;

/// Pauli basis for QEC measurements and resets.
#[pyclass(name = "QecBasis", module = "prism_q", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyQecBasis {
    X,
    Y,
    Z,
}

impl PyQecBasis {
    fn to_core(self) -> QecBasis {
        match self {
            PyQecBasis::X => QecBasis::X,
            PyQecBasis::Y => QecBasis::Y,
            PyQecBasis::Z => QecBasis::Z,
        }
    }
}

/// Reference to a prior measurement record (absolute index or lookback).
#[pyclass(name = "RecordRef", module = "prism_q", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyRecordRef(QecRecordRef);

#[pymethods]
impl PyRecordRef {
    /// Absolute measurement record index.
    #[staticmethod]
    fn absolute(index: usize) -> Self {
        Self(QecRecordRef::Absolute(index))
    }

    /// Relative lookback; `lookback(1)` is the most recent record.
    #[staticmethod]
    fn lookback(distance: usize) -> PyPrismResult<Self> {
        Ok(Self(QecRecordRef::lookback(distance)?))
    }

    fn __repr__(&self) -> String {
        format!("RecordRef({:?})", self.0)
    }
}

/// Pauli-noise annotation for a QEC program.
#[pyclass(name = "QecNoise", module = "prism_q", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyQecNoise(QecNoise);

#[pymethods]
impl PyQecNoise {
    #[staticmethod]
    fn x_error(p: f64) -> Self {
        Self(QecNoise::XError(p))
    }
    #[staticmethod]
    fn z_error(p: f64) -> Self {
        Self(QecNoise::ZError(p))
    }
    #[staticmethod]
    fn depolarize1(p: f64) -> Self {
        Self(QecNoise::Depolarize1(p))
    }
    #[staticmethod]
    fn depolarize2(p: f64) -> Self {
        Self(QecNoise::Depolarize2(p))
    }

    fn __repr__(&self) -> String {
        format!("QecNoise({:?})", self.0)
    }
}

/// A native measurement-record QEC program.
#[pyclass(name = "QecProgram", module = "prism_q")]
pub struct PyQecProgram {
    inner: QecProgram,
}

#[pymethods]
impl PyQecProgram {
    #[new]
    fn new(num_qubits: usize) -> Self {
        Self {
            inner: QecProgram::new(num_qubits),
        }
    }

    /// Parse a native QEC program from text.
    #[staticmethod]
    fn from_text(text: &str) -> PyPrismResult<Self> {
        Ok(Self {
            inner: QecProgram::from_text(text)?,
        })
    }

    /// Set runner options.
    #[pyo3(signature = (shots, seed = 42, chunk_size = None, keep_measurements = true))]
    fn set_options(
        &mut self,
        shots: usize,
        seed: u64,
        chunk_size: Option<usize>,
        keep_measurements: bool,
    ) {
        self.inner.set_options(QecOptions {
            shots,
            seed,
            chunk_size,
            keep_measurements,
        });
    }

    #[getter]
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    #[getter]
    fn num_measurements(&self) -> usize {
        self.inner.num_measurements()
    }
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Append a gate.
    fn push_gate(&mut self, gate: &PyGate, targets: Vec<usize>) -> PyPrismResult<()> {
        self.inner.push_gate(gate.inner().clone(), &targets)?;
        Ok(())
    }

    /// Reset a qubit to the +1 eigenstate of `basis`.
    fn reset(&mut self, basis: PyQecBasis, qubit: usize) -> PyPrismResult<()> {
        self.inner.reset(basis.to_core(), qubit)?;
        Ok(())
    }

    /// Measure a qubit in `basis`; returns the record index.
    fn measure(&mut self, basis: PyQecBasis, qubit: usize) -> PyPrismResult<usize> {
        Ok(self.inner.measure(basis.to_core(), qubit)?)
    }

    /// Z-basis measurement; returns the record index.
    fn measure_z(&mut self, qubit: usize) -> PyPrismResult<usize> {
        Ok(self.inner.measure_z(qubit)?)
    }

    /// X-basis measurement; returns the record index.
    fn measure_x(&mut self, qubit: usize) -> PyPrismResult<usize> {
        Ok(self.inner.measure_x(qubit)?)
    }

    /// Pauli-product (MPP) measurement from `(basis, qubit)` terms; returns the
    /// record index.
    fn measure_pauli_product(&mut self, terms: Vec<(PyQecBasis, usize)>) -> PyPrismResult<usize> {
        let terms: Vec<QecPauli> = terms
            .into_iter()
            .map(|(basis, qubit)| QecPauli::new(basis.to_core(), qubit))
            .collect();
        Ok(self.inner.measure_pauli_product(&terms)?)
    }

    /// Append a detector over `records`; returns the detector index.
    #[pyo3(signature = (records, coords = None))]
    fn detector(
        &mut self,
        records: Vec<PyRecordRef>,
        coords: Option<Vec<f64>>,
    ) -> PyPrismResult<usize> {
        let refs: Vec<QecRecordRef> = records.iter().map(|r| r.0).collect();
        let coords = coords.unwrap_or_default();
        Ok(self.inner.detector_with_coords(&refs, &coords)?)
    }

    /// Append a detector from lookback distances; returns the detector index.
    fn detector_lookback(&mut self, distances: Vec<usize>) -> PyPrismResult<usize> {
        let refs: Vec<QecRecordRef> = distances
            .into_iter()
            .map(QecRecordRef::lookback)
            .collect::<prism_q::Result<_>>()?;
        Ok(self.inner.detector(&refs)?)
    }

    /// Contribute `records` to logical observable `observable`.
    fn observable_include(
        &mut self,
        observable: usize,
        records: Vec<PyRecordRef>,
    ) -> PyPrismResult<()> {
        let refs: Vec<QecRecordRef> = records.iter().map(|r| r.0).collect();
        self.inner.observable_include(observable, &refs)?;
        Ok(())
    }

    /// Accept a shot only when the parity over `records` matches `expected`.
    fn postselect(&mut self, records: Vec<PyRecordRef>, expected: bool) -> PyPrismResult<()> {
        let refs: Vec<QecRecordRef> = records.iter().map(|r| r.0).collect();
        self.inner.postselect(&refs, expected)?;
        Ok(())
    }

    /// Append a Pauli-noise annotation on `targets`.
    fn noise(&mut self, channel: &PyQecNoise, targets: Vec<usize>) -> PyPrismResult<()> {
        self.inner.noise(channel.0, &targets)?;
        Ok(())
    }

    /// Sample the program through the compiled Clifford path.
    fn run(&self, py: Python<'_>) -> PyPrismResult<PyQecResult> {
        let program = &self.inner;
        let result = py.detach(|| run_qec_program(program))?;
        Ok(PyQecResult { inner: result })
    }

    fn __repr__(&self) -> String {
        format!(
            "QecProgram(num_qubits={}, measurements={}, detectors={}, observables={})",
            self.inner.num_qubits(),
            self.inner.num_measurements(),
            self.inner.num_detectors(),
            self.inner.num_observables()
        )
    }
}

/// Result of sampling a QEC program.
#[pyclass(name = "QecResult", module = "prism_q")]
pub struct PyQecResult {
    inner: QecSampleResult,
}

fn packed_to_2d<'py>(
    py: Python<'py>,
    packed: &PackedShots,
) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
    let n_shots = packed.num_shots();
    let n_meas = packed.num_measurements();
    let mut flat = Vec::with_capacity(n_shots * n_meas);
    for shot in 0..n_shots {
        for meas in 0..n_meas {
            flat.push(packed.get_bit(shot, meas));
        }
    }
    bool_matrix(py, n_shots, n_meas, flat)
}

#[pymethods]
impl PyQecResult {
    #[getter]
    fn total_shots(&self) -> usize {
        self.inner.total_shots
    }
    #[getter]
    fn accepted_shots(&self) -> usize {
        self.inner.accepted_shots
    }
    #[getter]
    fn discarded_shots(&self) -> usize {
        self.inner.discarded_shots
    }
    #[getter]
    fn logical_errors(&self) -> Vec<u64> {
        self.inner.logical_errors.clone()
    }

    /// Per-observable logical-error rate among accepted shots.
    fn logical_error_rates(&self) -> Vec<f64> {
        self.inner.logical_error_rates()
    }

    /// Fraction of shots accepted after postselection.
    fn survivor_rate(&self) -> f64 {
        self.inner.survivor_rate()
    }

    /// Detector records as a `(shots, num_detectors)` bool array.
    #[getter]
    fn detectors<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        packed_to_2d(py, &self.inner.detectors)
    }

    /// Observable records as a `(shots, num_observables)` bool array.
    #[getter]
    fn observables<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        packed_to_2d(py, &self.inner.observables)
    }

    /// Raw measurement records as a `(shots, num_measurements)` bool array
    /// (empty rows when `keep_measurements` is false).
    #[getter]
    fn measurements<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        packed_to_2d(py, &self.inner.measurements)
    }

    fn __repr__(&self) -> String {
        format!(
            "QecResult(total_shots={}, accepted={}, discarded={}, observables={})",
            self.inner.total_shots,
            self.inner.accepted_shots,
            self.inner.discarded_shots,
            self.inner.logical_errors.len()
        )
    }
}
