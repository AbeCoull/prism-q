//! Native QEC programs: construction, sampling, and packed-shot output.

use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use prism_q::{
    BpMethod, BpOsdDecoder, BpOsdOptions, DetectorErrorModel, MatchingDecoder, OsdMethod,
    PackedShots, PrismError, QecBasis, QecCircuitNoise, QecNoise, QecOptions, QecPauli, QecProgram,
    QecRecordRef, QecSampleResult, ShotLayout, UnionFindDecoder, run_qec_program,
    run_qec_program_reference,
};
use pyo3::prelude::*;

use crate::codec::{self, Kind, Reader, Writer};
use crate::error::PyPrismResult;
use crate::gate::PyGate;
use crate::numpy_util::{bool_matrix, f64_array, u8_matrix};
use crate::pickle::{Reduced, ReducedMember, reduce, reduce_member};

/// Pauli basis for QEC measurements and resets.
#[pyclass(name = "QecBasis", module = "prism_q", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyQecBasis {
    X,
    Y,
    Z,
}

#[pymethods]
impl PyQecBasis {
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<ReducedMember<'py>> {
        let name = match *slf.borrow() {
            PyQecBasis::X => "X",
            PyQecBasis::Y => "Y",
            PyQecBasis::Z => "Z",
        };
        reduce_member(slf.as_any(), name)
    }
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

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        let mut r = Reader::new(data, Kind::RecordRef)?;
        let record = codec::read_record(&mut r)?;
        r.finish()?;
        Ok(Self(record))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let mut w = Writer::new(Kind::RecordRef);
        codec::write_record(&mut w, &slf.get().0)?;
        reduce(slf.as_any(), w.finish())
    }

    fn __repr__(&self) -> String {
        format!("RecordRef({:?})", self.0)
    }
}

/// Pauli-noise annotation for a QEC program.
#[pyclass(name = "QecNoise", module = "prism_q", frozen, from_py_object)]
#[derive(Clone)]
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
    #[staticmethod]
    fn y_error(p: f64) -> Self {
        Self(QecNoise::YError(p))
    }
    /// X, Y, Z with probabilities `px`, `py`, `pz` per target.
    #[staticmethod]
    fn pauli_channel_1(px: f64, py: f64, pz: f64) -> Self {
        Self(QecNoise::PauliChannel1([px, py, pz]))
    }
    /// Fifteen two-qubit Pauli probabilities per target pair, in the order `IX, IY, IZ,
    /// XI, ..., ZZ` with the first letter on the first target.
    #[staticmethod]
    fn pauli_channel_2(probabilities: [f64; 15]) -> Self {
        Self(QecNoise::PauliChannel2(Box::new(probabilities)))
    }

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        let mut r = Reader::new(data, Kind::QecNoise)?;
        let channel = codec::read_qec_noise(&mut r)?;
        r.finish()?;
        Ok(Self(channel))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let mut w = Writer::new(Kind::QecNoise);
        codec::write_qec_noise(&mut w, &slf.get().0)?;
        reduce(slf.as_any(), w.finish())
    }

    fn __repr__(&self) -> String {
        format!("QecNoise({:?})", self.0)
    }
}

/// Circuit-level noise for the memory-experiment generators; a zero rate adds nothing.
#[pyclass(name = "QecCircuitNoise", module = "prism_q", frozen, from_py_object)]
#[derive(Clone, Copy)]
pub struct PyQecCircuitNoise(QecCircuitNoise);

#[pymethods]
impl PyQecCircuitNoise {
    #[new]
    #[pyo3(signature = (
        after_clifford_depolarization = 0.0,
        before_measure_flip_probability = 0.0,
        after_reset_flip_probability = 0.0,
        before_round_data_depolarization = 0.0,
    ))]
    fn new(
        after_clifford_depolarization: f64,
        before_measure_flip_probability: f64,
        after_reset_flip_probability: f64,
        before_round_data_depolarization: f64,
    ) -> Self {
        Self(QecCircuitNoise {
            after_clifford_depolarization,
            before_measure_flip_probability,
            after_reset_flip_probability,
            before_round_data_depolarization,
        })
    }

    /// Every term at rate `p`.
    #[staticmethod]
    fn uniform(p: f64) -> Self {
        Self(QecCircuitNoise::uniform(p))
    }

    #[getter]
    fn after_clifford_depolarization(&self) -> f64 {
        self.0.after_clifford_depolarization
    }
    #[getter]
    fn before_measure_flip_probability(&self) -> f64 {
        self.0.before_measure_flip_probability
    }
    #[getter]
    fn after_reset_flip_probability(&self) -> f64 {
        self.0.after_reset_flip_probability
    }
    #[getter]
    fn before_round_data_depolarization(&self) -> f64 {
        self.0.before_round_data_depolarization
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

fn generator_noise(noise: Option<PyQecCircuitNoise>) -> QecCircuitNoise {
    noise.map_or_else(QecCircuitNoise::default, |noise| noise.0)
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

    /// Render the program in the native QEC text format that `from_text` reads.
    fn to_text(&self) -> PyPrismResult<String> {
        Ok(self.inner.to_text()?)
    }

    /// Repetition-code Z memory with `distance` data qubits and `rounds` rounds.
    #[staticmethod]
    #[pyo3(signature = (distance, rounds, noise = None))]
    fn repetition_memory(
        distance: usize,
        rounds: usize,
        noise: Option<PyQecCircuitNoise>,
    ) -> PyPrismResult<Self> {
        Ok(Self {
            inner: QecProgram::repetition_memory(distance, rounds, &generator_noise(noise))?,
        })
    }

    /// Rotated surface-code memory in the X or Z logical basis.
    #[staticmethod]
    #[pyo3(signature = (distance, rounds, basis = PyQecBasis::Z, noise = None))]
    fn surface_memory(
        distance: usize,
        rounds: usize,
        basis: PyQecBasis,
        noise: Option<PyQecCircuitNoise>,
    ) -> PyPrismResult<Self> {
        Ok(Self {
            inner: QecProgram::surface_memory(
                distance,
                rounds,
                basis.to_core(),
                &generator_noise(noise),
            )?,
        })
    }

    /// Triangular 6.6.6 color-code memory in the X or Z logical basis, odd distance.
    #[staticmethod]
    #[pyo3(signature = (distance, rounds, basis = PyQecBasis::Z, noise = None))]
    fn color_memory(
        distance: usize,
        rounds: usize,
        basis: PyQecBasis,
        noise: Option<PyQecCircuitNoise>,
    ) -> PyPrismResult<Self> {
        Ok(Self {
            inner: QecProgram::color_memory(
                distance,
                rounds,
                basis.to_core(),
                &generator_noise(noise),
            )?,
        })
    }

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

    /// Apply the operations of `body` only in shots where the parity over
    /// `records` equals `expected`. `body` admits gates and resets only, so the
    /// measurement record keeps one layout across shots.
    fn feedforward(
        &mut self,
        records: Vec<PyRecordRef>,
        expected: bool,
        body: &PyQecProgram,
    ) -> PyPrismResult<()> {
        let refs: Vec<QecRecordRef> = records.iter().map(|r| r.0).collect();
        self.inner
            .feedforward(&refs, expected, body.inner.ops().to_vec())?;
        Ok(())
    }

    /// Append a Pauli-noise annotation on `targets`.
    fn noise(&mut self, channel: &PyQecNoise, targets: Vec<usize>) -> PyPrismResult<()> {
        self.inner.noise(channel.0.clone(), &targets)?;
        Ok(())
    }

    /// Sample the program through the compiled Clifford path.
    fn run(&self, py: Python<'_>) -> PyPrismResult<PyQecResult> {
        let program = &self.inner;
        let result = py.detach(|| run_qec_program(program))?;
        Ok(PyQecResult { inner: result })
    }

    /// Sample through the per-shot statevector reference path, the route that
    /// executes `feedforward`. Costs `O(shots * 2^n)`, so it suits small
    /// programs rather than bulk sampling.
    fn run_reference(&self, py: Python<'_>) -> PyPrismResult<PyQecResult> {
        let program = &self.inner;
        let result = py.detach(|| run_qec_program_reference(program))?;
        Ok(PyQecResult { inner: result })
    }

    /// Derive the detector error model from the program's noise annotations,
    /// detectors, and observables.
    fn detector_error_model(&self) -> PyPrismResult<PyDetectorErrorModel> {
        Ok(PyDetectorErrorModel {
            inner: self.inner.detector_error_model()?,
        })
    }

    #[staticmethod]
    fn _from_pickle(data: &[u8]) -> PyPrismResult<Self> {
        Ok(Self {
            inner: codec::decode_qec_program(data)?,
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Reduced<'py>> {
        let data = codec::encode_qec_program(&slf.borrow().inner)?;
        reduce(slf.as_any(), data)
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

/// Detector error model derived from a QEC program: independent error
/// mechanisms over the program's detectors and observables.
#[pyclass(name = "DetectorErrorModel", module = "prism_q", frozen)]
pub struct PyDetectorErrorModel {
    inner: DetectorErrorModel,
}

#[pymethods]
impl PyDetectorErrorModel {
    /// Parse the common detector error model text format, expanding `repeat` blocks.
    #[staticmethod]
    fn from_text(text: &str) -> PyPrismResult<Self> {
        Ok(Self {
            inner: DetectorErrorModel::from_text(text)?,
        })
    }

    /// Suggested `^` decomposition of each mechanism as lists of `(detectors,
    /// observables)` components, empty for mechanisms without one.
    fn suggested_decompositions(&self) -> Vec<Vec<(Vec<usize>, Vec<usize>)>> {
        self.inner
            .mechanisms()
            .iter()
            .map(|m| m.suggested_decomposition().to_vec())
            .collect()
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Mechanism probabilities as a `(num_mechanisms,)` float64 array.
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let probabilities = self
            .inner
            .mechanisms()
            .iter()
            .map(|m| m.probability())
            .collect();
        f64_array(py, probabilities)
    }

    /// Check matrix as a `(num_detectors, num_mechanisms)` bool array:
    /// entry `(d, m)` is true when mechanism `m` flips detector `d`.
    fn detector_matrix<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let cols = self.inner.num_mechanisms();
        let mut flat = vec![false; self.inner.num_detectors() * cols];
        for (mechanism, entry) in self.inner.mechanisms().iter().enumerate() {
            for &detector in entry.detectors() {
                flat[detector * cols + mechanism] = true;
            }
        }
        bool_matrix(py, self.inner.num_detectors(), cols, flat)
    }

    /// Observable flip matrix as a `(num_observables, num_mechanisms)` bool
    /// array: entry `(o, m)` is true when mechanism `m` flips observable `o`.
    fn observable_matrix<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let cols = self.inner.num_mechanisms();
        let mut flat = vec![false; self.inner.num_observables() * cols];
        for (mechanism, entry) in self.inner.mechanisms().iter().enumerate() {
            for &observable in entry.observables() {
                flat[observable * cols + mechanism] = true;
            }
        }
        bool_matrix(py, self.inner.num_observables(), cols, flat)
    }

    /// Coordinates per detector, empty lists for detectors without any.
    fn detector_coords(&self) -> Vec<Vec<f64>> {
        self.inner.detector_coords().to_vec()
    }

    /// Decompose hypergraph mechanisms into graphlike components (at most two
    /// detectors per mechanism); errors when a mechanism has no cover.
    fn decompose_graphlike(&self) -> PyPrismResult<PyDetectorErrorModel> {
        Ok(PyDetectorErrorModel {
            inner: self.inner.decompose_graphlike()?,
        })
    }

    /// Render the model in the common detector error model text format.
    fn to_text(&self) -> String {
        self.inner.to_text()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(mechanisms={}, detectors={}, observables={})",
            self.inner.num_mechanisms(),
            self.inner.num_detectors(),
            self.inner.num_observables()
        )
    }
}

/// Union-find decoder over a graphlike detector error model: predicts
/// observable flips from detector samples.
#[pyclass(name = "Decoder", module = "prism_q", frozen)]
pub struct PyDecoder {
    inner: UnionFindDecoder,
}

#[pymethods]
impl PyDecoder {
    /// Compile a decoder from a graphlike model (at most two detectors per
    /// mechanism; apply `decompose_graphlike` first when needed).
    #[new]
    fn new(model: &PyDetectorErrorModel) -> PyPrismResult<Self> {
        Ok(Self {
            inner: UnionFindDecoder::from_model(&model.inner)?,
        })
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Decode a `(shots, num_detectors)` bool array of detector samples into
    /// a `(shots, num_observables)` bool array of predicted observable flips.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let packed = pack_bool_rows(&detectors)?;
        let decoded = py.detach(|| self.inner.decode_packed(&packed))?;
        packed_to_2d(py, &decoded)
    }

    /// Fraction of shots whose predicted flips differ from `observables`
    /// (`(shots, num_observables)` bool) in any observable.
    fn logical_error_rate<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
        observables: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<f64> {
        let detectors = pack_bool_rows(&detectors)?;
        let observables = pack_bool_rows(&observables)?;
        Ok(py.detach(|| self.inner.logical_error_rate(&detectors, &observables))?)
    }

    fn __repr__(&self) -> String {
        format!(
            "Decoder(detectors={}, observables={})",
            self.inner.num_detectors(),
            self.inner.num_observables()
        )
    }
}

/// Exact minimum-weight perfect matching decoder over a graphlike detector
/// error model: predicts observable flips from detector samples.
#[pyclass(name = "MatchingDecoder", module = "prism_q", frozen)]
pub struct PyMatchingDecoder {
    inner: MatchingDecoder,
}

#[pymethods]
impl PyMatchingDecoder {
    /// Compile a decoder from a graphlike model (at most two detectors per
    /// mechanism; apply `decompose_graphlike` first when needed).
    #[new]
    fn new(model: &PyDetectorErrorModel) -> PyPrismResult<Self> {
        Ok(Self {
            inner: MatchingDecoder::from_model(&model.inner)?,
        })
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Decode a `(shots, num_detectors)` bool array of detector samples into
    /// a `(shots, num_observables)` bool array of predicted observable flips.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let packed = pack_bool_rows(&detectors)?;
        let decoded = py.detach(|| self.inner.decode_packed(&packed))?;
        packed_to_2d(py, &decoded)
    }

    /// Fraction of shots whose predicted flips differ from `observables`
    /// (`(shots, num_observables)` bool) in any observable.
    fn logical_error_rate<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
        observables: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<f64> {
        let detectors = pack_bool_rows(&detectors)?;
        let observables = pack_bool_rows(&observables)?;
        Ok(py.detach(|| self.inner.logical_error_rate(&detectors, &observables))?)
    }

    fn __repr__(&self) -> String {
        format!(
            "MatchingDecoder(detectors={}, observables={})",
            self.inner.num_detectors(),
            self.inner.num_observables()
        )
    }
}

/// Belief propagation with ordered-statistics post-processing over any
/// detector error model, hypergraph mechanisms included.
#[pyclass(name = "BpOsdDecoder", module = "prism_q", frozen)]
pub struct PyBpOsdDecoder {
    inner: BpOsdDecoder,
}

#[pymethods]
impl PyBpOsdDecoder {
    /// Compile a decoder. `bp_method` is `"min_sum"` (messages scaled by
    /// `min_sum_scaling`) or `"product_sum"`; `osd_method` is `"osd0"`,
    /// `"cs"` (combination sweep), or `"exhaustive"`, searching `osd_order`
    /// free columns.
    #[new]
    #[pyo3(signature = (
        model,
        *,
        max_iterations = 30,
        bp_method = "min_sum",
        min_sum_scaling = 0.625,
        osd_method = "cs",
        osd_order = 7,
    ))]
    fn new(
        model: &PyDetectorErrorModel,
        max_iterations: usize,
        bp_method: &str,
        min_sum_scaling: f64,
        osd_method: &str,
        osd_order: usize,
    ) -> PyPrismResult<Self> {
        let bp_method = match bp_method {
            "min_sum" => BpMethod::MinSum {
                scaling: min_sum_scaling,
            },
            "product_sum" => BpMethod::ProductSum,
            other => {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "unknown bp_method `{other}`; expected `min_sum` or `product_sum`"
                    ),
                }
                .into());
            }
        };
        let osd_method = match osd_method {
            "osd0" => OsdMethod::Zero,
            "cs" => OsdMethod::CombinationSweep { order: osd_order },
            "exhaustive" => OsdMethod::Exhaustive { order: osd_order },
            other => {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "unknown osd_method `{other}`; expected `osd0`, `cs`, or `exhaustive`"
                    ),
                }
                .into());
            }
        };
        let options = BpOsdOptions {
            max_iterations,
            bp_method,
            osd_method,
        };
        Ok(Self {
            inner: BpOsdDecoder::with_options(&model.inner, options)?,
        })
    }

    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Decode a `(shots, num_detectors)` bool array of detector samples into
    /// a `(shots, num_observables)` bool array of predicted observable flips.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
        let packed = pack_bool_rows(&detectors)?;
        let decoded = py.detach(|| self.inner.decode_packed(&packed))?;
        packed_to_2d(py, &decoded)
    }

    /// Fraction of shots whose predicted flips differ from `observables`
    /// (`(shots, num_observables)` bool) in any observable.
    fn logical_error_rate<'py>(
        &self,
        py: Python<'py>,
        detectors: PyReadonlyArray2<'py, bool>,
        observables: PyReadonlyArray2<'py, bool>,
    ) -> PyPrismResult<f64> {
        let detectors = pack_bool_rows(&detectors)?;
        let observables = pack_bool_rows(&observables)?;
        Ok(py.detach(|| self.inner.logical_error_rate(&detectors, &observables))?)
    }

    fn __repr__(&self) -> String {
        format!(
            "BpOsdDecoder(detectors={}, observables={})",
            self.inner.num_detectors(),
            self.inner.num_observables()
        )
    }
}

/// Pack a `(shots, columns)` bool array into shot-major words.
fn pack_bool_rows(array: &PyReadonlyArray2<'_, bool>) -> PyPrismResult<PackedShots> {
    let array = array.as_array();
    let shots = array.nrows();
    let columns = array.ncols();
    let m_words = columns.div_ceil(64);
    let mut data = vec![0u64; shots * m_words];
    for (shot, row) in array.outer_iter().enumerate() {
        let base = shot * m_words;
        for (column, &bit) in row.iter().enumerate() {
            if bit {
                data[base + column / 64] |= 1u64 << (column % 64);
            }
        }
    }
    Ok(PackedShots::try_from_shot_major(data, shots, columns)?)
}

/// Result of sampling a QEC program.
#[pyclass(name = "QecResult", module = "prism_q")]
pub struct PyQecResult {
    inner: QecSampleResult,
}

/// Byte value to its eight bits, least significant first.
const BYTE_BITS: [[bool; 8]; 256] = {
    let mut table = [[false; 8]; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut bit = 0;
        while bit < 8 {
            table[byte][bit] = (byte >> bit) & 1 == 1;
            bit += 1;
        }
        byte += 1;
    }
    table
};

/// Write the low `out.len()` bits of `word` into `out`, bit 0 first.
fn unpack_word(word: u64, out: &mut [bool]) {
    let full = out.len() / 8;
    let (bytes, tail) = out.split_at_mut(full * 8);
    for (chunk, byte) in bytes.chunks_exact_mut(8).zip(word.to_le_bytes()) {
        chunk.copy_from_slice(&BYTE_BITS[byte as usize]);
    }
    for (bit, slot) in tail.iter_mut().enumerate() {
        *slot = (word >> (full * 8 + bit)) & 1 != 0;
    }
}

pub(crate) fn packed_to_2d<'py>(
    py: Python<'py>,
    packed: &PackedShots,
) -> PyPrismResult<Bound<'py, PyArray2<bool>>> {
    let n_shots = packed.num_shots();
    let n_meas = packed.num_measurements();
    let mut flat = vec![false; n_shots * n_meas];
    if n_meas > 0 {
        match packed.layout() {
            ShotLayout::ShotMajor => {
                for (shot, row) in flat.chunks_exact_mut(n_meas).enumerate() {
                    for (&word, bits) in packed.shot_words(shot).iter().zip(row.chunks_mut(64)) {
                        unpack_word(word, bits);
                    }
                }
            }
            _ => {
                for (block, rows) in flat.chunks_mut(64 * n_meas).enumerate() {
                    for meas in 0..n_meas {
                        let mut word = packed.meas_words(meas)[block];
                        for row in rows.chunks_exact_mut(n_meas) {
                            row[meas] = word & 1 != 0;
                            word >>= 1;
                        }
                    }
                }
            }
        }
    }
    bool_matrix(py, n_shots, n_meas, flat)
}

/// Repack shot-major as `(shots, ceil(n / 8))` bytes: record `j` is bit `j % 8` of
/// byte `j / 8`, and the padding bits of the last byte are clear.
pub(crate) fn packed_to_bytes<'py>(
    py: Python<'py>,
    packed: &PackedShots,
) -> PyPrismResult<Bound<'py, PyArray2<u8>>> {
    let n_shots = packed.num_shots();
    let n_meas = packed.num_measurements();
    let row_bytes = n_meas.div_ceil(8);
    let mut flat = vec![0u8; n_shots * row_bytes];
    if row_bytes > 0 {
        match packed.layout() {
            ShotLayout::ShotMajor => {
                let tail_mask = u8::MAX >> ((8 - n_meas % 8) % 8);
                for (shot, row) in flat.chunks_exact_mut(row_bytes).enumerate() {
                    for (&word, bytes) in packed.shot_words(shot).iter().zip(row.chunks_mut(8)) {
                        bytes.copy_from_slice(&word.to_le_bytes()[..bytes.len()]);
                    }
                    row[row_bytes - 1] &= tail_mask;
                }
            }
            _ => {
                for (block, rows) in flat.chunks_mut(64 * row_bytes).enumerate() {
                    for meas in 0..n_meas {
                        let mut word = packed.meas_words(meas)[block];
                        let bit = meas % 8;
                        for row in rows.chunks_exact_mut(row_bytes) {
                            row[meas / 8] |= ((word & 1) as u8) << bit;
                            word >>= 1;
                        }
                    }
                }
            }
        }
    }
    u8_matrix(py, n_shots, row_bytes, flat)
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

    /// Detector records as `(shots, ceil(num_detectors / 8))` uint8 in little bit
    /// order, the layout `np.unpackbits(..., bitorder="little")` reverses.
    fn packed_detectors<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<u8>>> {
        packed_to_bytes(py, &self.inner.detectors)
    }

    /// Observable records in the `packed_detectors` layout.
    fn packed_observables<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<u8>>> {
        packed_to_bytes(py, &self.inner.observables)
    }

    /// Measurement records in the `packed_detectors` layout.
    fn packed_measurements<'py>(&self, py: Python<'py>) -> PyPrismResult<Bound<'py, PyArray2<u8>>> {
        packed_to_bytes(py, &self.inner.measurements)
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
