//! Native measurement-record QEC program IR, parser, and runners.
//!
//! Models QEC workloads that need measurement records, detectors, observables,
//! postselection, expectation metadata, and Pauli-noise annotations. The IR is
//! separate from `Circuit` so measurement records do not have to fit
//! final-measurement OpenQASM semantics.
//!
//! # Public surface
//!
//! - [`QecProgram`] is the IR. Construct via [`QecProgram::new`] /
//!   [`QecProgram::with_options`] and the typed `push_*` methods, or load
//!   from text via [`parse_qec_program`] / [`QecProgram::from_text`].
//! - [`run_qec_program`] is the scalable Clifford execution path. Lowers
//!   programs into the packed compiled sampler and supports Pauli noise by
//!   XORing sensitivity rows onto packed measurement records.
//! - [`run_qec_program_reference`] is the correctness oracle. One state-vector
//!   simulation per shot. Use it for small semantic cross-checks, not bulk
//!   sampling.
//! - [`compile_qec_program_rows`] lowers basis measurements and `MPP` records
//!   into the packed X/Z Pauli row representation used by sampler internals.
//!   It does not execute gates, resets, or active noise.
//!
//! [`QecSampleResult`] carries packed measurement, detector, and observable
//! shots, plus accepted and discarded shot counts after postselection and
//! per-observable logical-error counts.

mod noise;
mod parse;
mod result;
mod runner;

pub use parse::parse_qec_program;
pub use result::QecSampleResult;
pub use runner::{run_qec_program, run_qec_program_reference};

use crate::circuit::Circuit;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{get_bit, set_bit, PackedShots, PauliVec};

/// Pauli basis used by QEC measurements and Pauli products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QecBasis {
    /// X basis.
    X,
    /// Y basis.
    Y,
    /// Z basis.
    Z,
}

/// One Pauli term in an MPP-style measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QecPauli {
    /// Pauli basis for this term.
    pub basis: QecBasis,
    /// Qubit acted on by this term.
    pub qubit: usize,
}

impl QecPauli {
    /// Create a Pauli term.
    pub fn new(basis: QecBasis, qubit: usize) -> Self {
        Self { basis, qubit }
    }

    /// Create an X term.
    pub fn x(qubit: usize) -> Self {
        Self::new(QecBasis::X, qubit)
    }

    /// Create a Y term.
    pub fn y(qubit: usize) -> Self {
        Self::new(QecBasis::Y, qubit)
    }

    /// Create a Z term.
    pub fn z(qubit: usize) -> Self {
        Self::new(QecBasis::Z, qubit)
    }
}

/// Reference to a previous measurement record.
///
/// Lookbacks are resolved against the count of measurement records that exist
/// at the moment the referencing operation is appended (or, for queries like
/// [`QecProgram::detector_rows`], at the moment that operation is reached
/// during the walk). `Lookback(1)` is the most recent measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QecRecordRef {
    /// Absolute measurement record index.
    Absolute(usize),
    /// Relative lookback into prior records, where `Lookback(1)` is the most recent.
    Lookback(usize),
}

impl QecRecordRef {
    /// Create an absolute measurement reference.
    pub fn absolute(index: usize) -> Self {
        Self::Absolute(index)
    }

    /// Create a relative measurement reference.
    pub fn lookback(distance: usize) -> Result<Self> {
        if distance == 0 {
            return Err(PrismError::InvalidParameter {
                message: "measurement lookback distance must be at least 1".to_string(),
            });
        }
        Ok(Self::Lookback(distance))
    }

    fn resolve(self, next_measurement: usize) -> Result<usize> {
        match self {
            Self::Absolute(index) if index < next_measurement => Ok(index),
            Self::Absolute(index) => Err(PrismError::InvalidParameter {
                message: format!(
                    "measurement record {index} out of bounds for {next_measurement} existing records"
                ),
            }),
            Self::Lookback(distance) if distance > 0 && distance <= next_measurement => {
                Ok(next_measurement - distance)
            }
            Self::Lookback(distance) => Err(PrismError::InvalidParameter {
                message: format!(
                    "measurement lookback {distance} out of bounds for {next_measurement} existing records"
                ),
            }),
        }
    }
}

/// Pauli-noise annotation for native QEC programs.
///
/// Probabilities are validated when the annotation is appended to a
/// [`QecProgram`]. Probability zero is treated as an inactive annotation by
/// runner APIs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QecNoise {
    /// With probability `p`, apply X to each target.
    XError(f64),
    /// With probability `p`, apply Z to each target.
    ZError(f64),
    /// For each target, with total probability `p`, apply a uniformly random
    /// non-identity single-qubit Pauli. Each of X, Y, Z fires with probability
    /// `p / 3`.
    Depolarize1(f64),
    /// For each target pair, with total probability `p`, apply a uniformly
    /// random non-identity two-qubit Pauli. Each of the 15 non-identity
    /// two-qubit Paulis fires with probability `p / 15`. The target list is
    /// consumed in pairs and must have even length.
    Depolarize2(f64),
}

impl QecNoise {
    /// Channel probability.
    pub fn probability(self) -> f64 {
        match self {
            Self::XError(p) | Self::ZError(p) | Self::Depolarize1(p) | Self::Depolarize2(p) => p,
        }
    }

    /// Native text instruction name for this channel.
    pub fn name(self) -> &'static str {
        match self {
            Self::XError(_) => "X_ERROR",
            Self::ZError(_) => "Z_ERROR",
            Self::Depolarize1(_) => "DEPOLARIZE1",
            Self::Depolarize2(_) => "DEPOLARIZE2",
        }
    }
}

/// One operation in a native QEC program.
#[derive(Debug, Clone, PartialEq)]
pub enum QecOp {
    /// Standard PRISM-Q gate operation. The compiled runner requires Clifford
    /// gates; the reference runner accepts any gate the statevector backend
    /// supports.
    Gate { gate: Gate, targets: Vec<usize> },
    /// Single-qubit measurement in the requested basis. Produces one
    /// measurement record.
    Measure { basis: QecBasis, qubit: usize },
    /// Pauli-product (`MPP`) measurement. Produces one measurement record
    /// equal to the parity of the listed Pauli terms.
    MeasurePauliProduct { terms: Vec<QecPauli> },
    /// Reset a qubit to the +1 eigenstate of the requested basis.
    Reset { basis: QecBasis, qubit: usize },
    /// Detector: parity over the listed measurement records. `coords` is
    /// arbitrary passthrough metadata for visualization and downstream
    /// decoders; it does not affect sampling.
    Detector {
        records: Vec<QecRecordRef>,
        coords: Vec<f64>,
    },
    /// Logical observable parity contribution. Multiple includes for the same
    /// `observable` index XOR into a single observable row.
    ObservableInclude {
        observable: usize,
        records: Vec<QecRecordRef>,
    },
    /// Expectation-value metadata. Both runners reject programs containing
    /// this op until estimator semantics land.
    ExpectationValue {
        terms: Vec<QecPauli>,
        coefficient: f64,
    },
    /// Postselection predicate. The shot is accepted only when the parity over
    /// `records` matches `expected`.
    Postselect {
        records: Vec<QecRecordRef>,
        expected: bool,
    },
    /// Pauli-noise annotation applied at this point in the program. Zero
    /// probability is treated as inactive.
    Noise {
        channel: QecNoise,
        targets: Vec<usize>,
    },
    /// Scheduling separator. No semantic effect; carried forward for parity
    /// with native QEC text formats.
    Tick,
}

/// Packed Pauli row for one QEC measurement record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QecMeasurementRow {
    num_qubits: usize,
    pauli: PauliVec,
    weight: usize,
}

impl QecMeasurementRow {
    /// Create a row from Pauli-product terms.
    pub fn from_terms(num_qubits: usize, terms: &[QecPauli]) -> Result<Self> {
        if terms.is_empty() {
            return Err(PrismError::InvalidParameter {
                message: "QEC measurement row requires at least one Pauli term".to_string(),
            });
        }
        validate_pauli_terms(terms, num_qubits)?;

        let row_words = num_qubits.div_ceil(64);
        let mut pauli = PauliVec::new(row_words);

        for term in terms {
            match term.basis {
                QecBasis::X => set_bit(&mut pauli.x, term.qubit, true),
                QecBasis::Y => {
                    set_bit(&mut pauli.x, term.qubit, true);
                    set_bit(&mut pauli.z, term.qubit, true);
                }
                QecBasis::Z => set_bit(&mut pauli.z, term.qubit, true),
            }
        }

        Ok(Self {
            num_qubits,
            pauli,
            weight: terms.len(),
        })
    }

    /// Create a single-qubit measurement row.
    pub fn single(num_qubits: usize, basis: QecBasis, qubit: usize) -> Result<Self> {
        Self::from_terms(num_qubits, &[QecPauli::new(basis, qubit)])
    }

    /// Number of qubits covered by this row.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity Pauli terms.
    pub fn weight(&self) -> usize {
        self.weight
    }

    /// Packed X mask.
    pub fn x_mask(&self) -> &[u64] {
        &self.pauli.x
    }

    /// Packed Z mask.
    pub fn z_mask(&self) -> &[u64] {
        &self.pauli.z
    }

    /// Pauli term on one qubit, or `None` for identity (or when `qubit` is
    /// outside this row's qubit range).
    pub fn pauli_at(&self, qubit: usize) -> Option<QecBasis> {
        if qubit >= self.num_qubits {
            return None;
        }
        match (get_bit(&self.pauli.x, qubit), get_bit(&self.pauli.z, qubit)) {
            (true, false) => Some(QecBasis::X),
            (true, true) => Some(QecBasis::Y),
            (false, true) => Some(QecBasis::Z),
            (false, false) => None,
        }
    }

    /// Return non-identity Pauli terms in ascending qubit order.
    pub fn terms(&self) -> Vec<QecPauli> {
        let mut terms = Vec::with_capacity(self.weight);
        for qubit in 0..self.num_qubits {
            if let Some(basis) = self.pauli_at(qubit) {
                terms.push(QecPauli::new(basis, qubit));
            }
        }
        terms
    }

    /// Packed row storage in bytes.
    pub fn packed_bytes(&self) -> usize {
        (self.pauli.x.len() + self.pauli.z.len()) * std::mem::size_of::<u64>()
    }
}

/// Compiled QEC record rows ready for sampler lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QecCompiledRows {
    num_qubits: usize,
    measurement_rows: Vec<QecMeasurementRow>,
    detector_rows: Vec<Vec<usize>>,
    observable_rows: Vec<Vec<usize>>,
    postselection_rows: Vec<Vec<usize>>,
    postselection_expected: Vec<bool>,
}

impl QecCompiledRows {
    /// Number of qubits in the source program.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Measurement rows in record order.
    pub fn measurement_rows(&self) -> &[QecMeasurementRow] {
        &self.measurement_rows
    }

    /// Detector parity rows over measurement records.
    pub fn detector_rows(&self) -> &[Vec<usize>] {
        &self.detector_rows
    }

    /// Observable parity rows over measurement records.
    pub fn observable_rows(&self) -> &[Vec<usize>] {
        &self.observable_rows
    }

    /// Postselection parity rows.
    pub fn postselection_rows(&self) -> &[Vec<usize>] {
        &self.postselection_rows
    }

    /// Expected parity for each postselection row.
    pub fn postselection_expected(&self) -> &[bool] {
        &self.postselection_expected
    }

    /// Postselection parity rows paired with expected values.
    pub fn postselection_predicates(&self) -> impl ExactSizeIterator<Item = (&[usize], bool)> + '_ {
        self.postselection_rows
            .iter()
            .map(Vec::as_slice)
            .zip(self.postselection_expected.iter().copied())
    }

    /// Number of measurement records.
    pub fn num_measurements(&self) -> usize {
        self.measurement_rows.len()
    }

    /// Number of detector rows.
    pub fn num_detectors(&self) -> usize {
        self.detector_rows.len()
    }

    /// Number of observable rows.
    pub fn num_observables(&self) -> usize {
        self.observable_rows.len()
    }

    /// Number of postselection predicates.
    pub fn num_postselections(&self) -> usize {
        self.postselection_rows.len()
    }

    /// Packed words per X or Z mask.
    pub fn packed_row_words(&self) -> usize {
        self.num_qubits.div_ceil(64)
    }

    /// Packed measurement row storage in bytes.
    pub fn measurement_mask_bytes(&self) -> usize {
        self.measurement_rows
            .len()
            .saturating_mul(self.packed_row_words())
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u64>())
    }

    /// Compute detector records from packed measurement records.
    pub fn detector_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.detector_rows)
    }

    /// Compute logical observable records from packed measurement records.
    pub fn observable_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.observable_rows)
    }

    /// Compute postselection predicate parities from packed measurement records.
    pub fn postselection_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.postselection_rows)
    }
}

/// Options for running a native QEC program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QecOptions {
    /// Number of shots requested by runner APIs.
    pub shots: usize,
    /// RNG seed used by stochastic samplers and Pauli-noise dispatch.
    pub seed: u64,
    /// Optional chunk size for the compiled runner. When `Some(n)`, sampling
    /// proceeds in batches of at most `n` shots and intermediate measurement
    /// matrices are not held in memory together. `None` is equivalent to
    /// `Some(shots)`. `Some(0)` is rejected. Has no effect on the reference
    /// runner.
    pub chunk_size: Option<usize>,
    /// When `false`, [`QecSampleResult::measurements`] is returned with zero
    /// shots (only the column count is preserved). Detector and observable
    /// records are always populated. Set to `false` to avoid materializing
    /// large measurement-record matrices when only detectors and observables
    /// are needed.
    pub keep_measurements: bool,
}

impl Default for QecOptions {
    fn default() -> Self {
        Self {
            shots: 1024,
            seed: 42,
            chunk_size: None,
            keep_measurements: true,
        }
    }
}

/// Native QEC program expressed as measurement-record operations.
#[derive(Debug, Clone, PartialEq)]
pub struct QecProgram {
    num_qubits: usize,
    ops: Vec<QecOp>,
    options: QecOptions,
}

impl QecProgram {
    /// Create an empty program.
    pub fn new(num_qubits: usize) -> Self {
        Self::with_options(num_qubits, QecOptions::default())
    }

    /// Create an empty program with explicit options.
    pub fn with_options(num_qubits: usize, options: QecOptions) -> Self {
        Self {
            num_qubits,
            ops: Vec::new(),
            options,
        }
    }

    /// Create a program from operations, validating record references as
    /// operations are appended.
    pub fn from_ops(num_qubits: usize, options: QecOptions, ops: Vec<QecOp>) -> Result<Self> {
        let mut program = Self::with_options(num_qubits, options);
        let mut next_measurement = 0usize;
        for op in ops {
            program.validate_op(&op, next_measurement)?;
            if matches!(
                op,
                QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. }
            ) {
                next_measurement += 1;
            }
            program.ops.push(op);
        }
        Ok(program)
    }

    /// Parse a native measurement-record QEC program.
    pub fn from_text(input: &str) -> Result<Self> {
        parse_qec_program(input)
    }

    /// Number of qubits in the program.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Runner options.
    pub fn options(&self) -> QecOptions {
        self.options
    }

    /// Set runner options.
    pub fn set_options(&mut self, options: QecOptions) {
        self.options = options;
    }

    /// Operation stream.
    pub fn ops(&self) -> &[QecOp] {
        &self.ops
    }

    /// Number of measurement records produced by the operation stream.
    pub fn num_measurements(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| {
                matches!(
                    op,
                    QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. }
                )
            })
            .count()
    }

    /// Number of detector rows.
    pub fn num_detectors(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, QecOp::Detector { .. }))
            .count()
    }

    /// Number of logical observable slots.
    pub fn num_observables(&self) -> usize {
        self.ops
            .iter()
            .filter_map(|op| match op {
                QecOp::ObservableInclude { observable, .. } => Some(*observable),
                _ => None,
            })
            .max()
            .map_or(0, |max_idx| max_idx + 1)
    }

    /// Append a validated operation.
    pub fn push_op(&mut self, op: QecOp) -> Result<()> {
        self.validate_op(&op, self.num_measurements())?;
        self.ops.push(op);
        Ok(())
    }

    /// Append a gate operation.
    pub fn push_gate(&mut self, gate: Gate, targets: &[usize]) -> Result<()> {
        self.push_op(QecOp::Gate {
            gate,
            targets: targets.to_vec(),
        })
    }

    /// Append a reset operation.
    pub fn reset(&mut self, basis: QecBasis, qubit: usize) -> Result<()> {
        self.push_op(QecOp::Reset { basis, qubit })
    }

    /// Append a single-qubit measurement and return its record index.
    pub fn measure(&mut self, basis: QecBasis, qubit: usize) -> Result<usize> {
        let record = self.num_measurements();
        self.push_op(QecOp::Measure { basis, qubit })?;
        Ok(record)
    }

    /// Append a Z-basis measurement and return its record index.
    pub fn measure_z(&mut self, qubit: usize) -> Result<usize> {
        self.measure(QecBasis::Z, qubit)
    }

    /// Append an X-basis measurement and return its record index.
    pub fn measure_x(&mut self, qubit: usize) -> Result<usize> {
        self.measure(QecBasis::X, qubit)
    }

    /// Append a Pauli-product measurement and return its record index.
    pub fn measure_pauli_product(&mut self, terms: &[QecPauli]) -> Result<usize> {
        let record = self.num_measurements();
        self.push_op(QecOp::MeasurePauliProduct {
            terms: terms.to_vec(),
        })?;
        Ok(record)
    }

    /// Append a detector and return its detector index.
    pub fn detector(&mut self, records: &[QecRecordRef]) -> Result<usize> {
        self.detector_with_coords(records, &[])
    }

    /// Append a detector with coordinates and return its detector index.
    pub fn detector_with_coords(
        &mut self,
        records: &[QecRecordRef],
        coords: &[f64],
    ) -> Result<usize> {
        let detector = self.num_detectors();
        self.push_op(QecOp::Detector {
            records: records.to_vec(),
            coords: coords.to_vec(),
        })?;
        Ok(detector)
    }

    /// Append a logical observable parity contribution.
    pub fn observable_include(
        &mut self,
        observable: usize,
        records: &[QecRecordRef],
    ) -> Result<()> {
        self.push_op(QecOp::ObservableInclude {
            observable,
            records: records.to_vec(),
        })
    }

    /// Append expectation-value metadata.
    pub fn expectation_value(&mut self, terms: &[QecPauli], coefficient: f64) -> Result<()> {
        self.push_op(QecOp::ExpectationValue {
            terms: terms.to_vec(),
            coefficient,
        })
    }

    /// Append a postselection predicate.
    pub fn postselect(&mut self, records: &[QecRecordRef], expected: bool) -> Result<()> {
        self.push_op(QecOp::Postselect {
            records: records.to_vec(),
            expected,
        })
    }

    /// Append a Pauli-noise annotation.
    pub fn noise(&mut self, channel: QecNoise, targets: &[usize]) -> Result<()> {
        self.push_op(QecOp::Noise {
            channel,
            targets: targets.to_vec(),
        })
    }

    /// Resolve detector rows to absolute measurement record indices.
    pub fn detector_rows(&self) -> Result<Vec<Vec<usize>>> {
        let mut rows = Vec::new();
        let mut next_measurement = 0;
        for op in &self.ops {
            match op {
                QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. } => {
                    next_measurement += 1;
                }
                QecOp::Detector { records, .. } => {
                    rows.push(resolve_records(records, next_measurement)?);
                }
                _ => {}
            }
        }
        Ok(rows)
    }

    /// Resolve observable rows to absolute measurement record indices.
    pub fn observable_rows(&self) -> Result<Vec<Vec<usize>>> {
        let mut rows = Vec::new();
        let mut next_measurement = 0;
        for op in &self.ops {
            match op {
                QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. } => {
                    next_measurement += 1;
                }
                QecOp::ObservableInclude {
                    observable,
                    records,
                } => {
                    if rows.len() <= *observable {
                        rows.resize_with(*observable + 1, Vec::new);
                    }
                    rows[*observable].extend(resolve_records(records, next_measurement)?);
                }
                _ => {}
            }
        }
        Ok(rows)
    }

    /// Resolve postselection rows to absolute measurement record indices.
    pub fn postselection_rows(&self) -> Result<Vec<(Vec<usize>, bool)>> {
        let mut rows = Vec::new();
        let mut next_measurement = 0;
        for op in &self.ops {
            match op {
                QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. } => {
                    next_measurement += 1;
                }
                QecOp::Postselect { records, expected } => {
                    rows.push((resolve_records(records, next_measurement)?, *expected));
                }
                _ => {}
            }
        }
        Ok(rows)
    }

    /// Create an empty result with the program's current record shape.
    pub fn empty_result(&self) -> QecSampleResult {
        QecSampleResult::empty(
            self.num_measurements(),
            self.num_detectors(),
            self.num_observables(),
        )
    }

    fn validate_op(&self, op: &QecOp, next_measurement: usize) -> Result<()> {
        match op {
            QecOp::Gate { gate, targets } => {
                if gate.num_qubits() != targets.len() {
                    return Err(PrismError::GateArity {
                        gate: gate.name().to_string(),
                        expected: gate.num_qubits(),
                        got: targets.len(),
                    });
                }
                validate_qubits(targets.iter().copied(), self.num_qubits)?;
            }
            QecOp::Measure { qubit, .. } | QecOp::Reset { qubit, .. } => {
                validate_qubit(*qubit, self.num_qubits)?;
            }
            QecOp::MeasurePauliProduct { terms } => {
                if terms.is_empty() {
                    return Err(PrismError::InvalidParameter {
                        message: "Pauli-product measurement requires at least one term".to_string(),
                    });
                }
                validate_pauli_terms(terms, self.num_qubits)?;
            }
            QecOp::Detector { records, coords } => {
                resolve_records(records, next_measurement)?;
                validate_finite_values(coords, "detector coordinate")?;
            }
            QecOp::ObservableInclude { records, .. } | QecOp::Postselect { records, .. } => {
                resolve_records(records, next_measurement)?;
            }
            QecOp::ExpectationValue { terms, coefficient } => {
                if terms.is_empty() {
                    return Err(PrismError::InvalidParameter {
                        message: "expectation value requires at least one Pauli term".to_string(),
                    });
                }
                validate_pauli_terms(terms, self.num_qubits)?;
                if !coefficient.is_finite() {
                    return Err(PrismError::InvalidParameter {
                        message: "expectation-value coefficient must be finite".to_string(),
                    });
                }
            }
            QecOp::Noise { channel, targets } => {
                validate_noise(*channel, targets, self.num_qubits)?;
            }
            QecOp::Tick => {}
        }
        Ok(())
    }
}

/// Compile measurement-record operations into packed QEC row metadata.
///
/// Lowers `Measure` and `MeasurePauliProduct` ops into the same packed X/Z
/// Pauli row representation used by the compiled sampler internals, and
/// resolves detector, observable, and postselection record references to
/// absolute indices. Useful when consumers want the row-level representation
/// for custom sampler integration.
///
/// This is a sampler-row primitive, not an execution path: it rejects
/// programs containing gates, resets, active Pauli noise, or `EXP_VAL`.
/// Zero-probability noise annotations are skipped. To execute a full program
/// (including gates, resets, and noise) use [`run_qec_program`].
pub fn compile_qec_program_rows(program: &QecProgram) -> Result<QecCompiledRows> {
    let mut measurement_rows = Vec::with_capacity(program.num_measurements());

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: format!(
                        "QEC row compilation does not lower gates yet, got `{}`",
                        gate.name()
                    ),
                });
            }
            QecOp::Measure { basis, qubit } => {
                measurement_rows.push(QecMeasurementRow::single(
                    program.num_qubits(),
                    *basis,
                    *qubit,
                )?);
            }
            QecOp::MeasurePauliProduct { terms } => {
                measurement_rows.push(QecMeasurementRow::from_terms(program.num_qubits(), terms)?);
            }
            QecOp::Reset { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: "QEC row compilation does not lower resets yet".to_string(),
                });
            }
            QecOp::ExpectationValue { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: "QEC row compilation does not evaluate EXP_VAL yet".to_string(),
                });
            }
            QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
            QecOp::Noise { channel, .. } if channel.probability() == 0.0 => {}
            QecOp::Noise { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: "QEC row compilation does not support active noise annotations yet"
                        .to_string(),
                });
            }
        }
    }

    let postselection_predicates = program.postselection_rows()?;
    let mut postselection_rows = Vec::with_capacity(postselection_predicates.len());
    let mut postselection_expected = Vec::with_capacity(postselection_predicates.len());
    for (row, expected) in postselection_predicates {
        postselection_rows.push(row);
        postselection_expected.push(expected);
    }

    Ok(QecCompiledRows {
        num_qubits: program.num_qubits(),
        measurement_rows,
        detector_rows: program.detector_rows()?,
        observable_rows: program.observable_rows()?,
        postselection_rows,
        postselection_expected,
    })
}

pub(super) fn append_basis_to_z_rotation(circuit: &mut Circuit, basis: QecBasis, qubit: usize) {
    match basis {
        QecBasis::X => circuit.add_gate(Gate::H, &[qubit]),
        QecBasis::Y => {
            circuit.add_gate(Gate::Sdg, &[qubit]);
            circuit.add_gate(Gate::H, &[qubit]);
        }
        QecBasis::Z => {}
    }
}

pub(super) fn append_z_to_basis_rotation(circuit: &mut Circuit, basis: QecBasis, qubit: usize) {
    match basis {
        QecBasis::X => circuit.add_gate(Gate::H, &[qubit]),
        QecBasis::Y => {
            circuit.add_gate(Gate::H, &[qubit]);
            circuit.add_gate(Gate::S, &[qubit]);
        }
        QecBasis::Z => {}
    }
}

fn resolve_records(records: &[QecRecordRef], next_measurement: usize) -> Result<Vec<usize>> {
    records
        .iter()
        .map(|record| record.resolve(next_measurement))
        .collect()
}

fn validate_qubit(qubit: usize, num_qubits: usize) -> Result<()> {
    if qubit >= num_qubits {
        return Err(PrismError::InvalidQubit {
            index: qubit,
            register_size: num_qubits,
        });
    }
    Ok(())
}

fn validate_qubits<I>(qubits: I, num_qubits: usize) -> Result<()>
where
    I: IntoIterator<Item = usize>,
{
    for qubit in qubits {
        validate_qubit(qubit, num_qubits)?;
    }
    Ok(())
}

fn validate_pauli_terms(terms: &[QecPauli], num_qubits: usize) -> Result<()> {
    for (idx, term) in terms.iter().enumerate() {
        validate_qubit(term.qubit, num_qubits)?;
        if terms[..idx].iter().any(|prior| prior.qubit == term.qubit) {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "Pauli-product measurement contains duplicate qubit {}",
                    term.qubit
                ),
            });
        }
    }
    Ok(())
}

fn validate_finite_values(values: &[f64], label: &str) -> Result<()> {
    for value in values {
        if !value.is_finite() {
            return Err(PrismError::InvalidParameter {
                message: format!("{label} must be finite"),
            });
        }
    }
    Ok(())
}

fn validate_noise(channel: QecNoise, targets: &[usize], num_qubits: usize) -> Result<()> {
    let p = channel.probability();
    if !(0.0..=1.0).contains(&p) || !p.is_finite() {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "{} probability must be finite and in [0, 1]",
                channel.name()
            ),
        });
    }

    if targets.is_empty() {
        return Err(PrismError::InvalidParameter {
            message: format!("{} requires at least one target", channel.name()),
        });
    }

    if matches!(channel, QecNoise::Depolarize2(_)) && targets.len() % 2 != 0 {
        return Err(PrismError::InvalidParameter {
            message: "DEPOLARIZE2 requires an even number of targets".to_string(),
        });
    }

    if matches!(channel, QecNoise::Depolarize2(_)) {
        for pair in targets.chunks_exact(2) {
            if pair[0] == pair[1] {
                return Err(PrismError::InvalidParameter {
                    message: "DEPOLARIZE2 target pairs must use distinct qubits".to_string(),
                });
            }
        }
    }

    validate_qubits(targets.iter().copied(), num_qubits)
}
