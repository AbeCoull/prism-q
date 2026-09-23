//! Measurement-record QEC programs ([`QecProgram`]): parser, runners, detector error
//! models, and a union-find decoder. The IR is separate from `Circuit` so measurement
//! records need not fit final-measurement OpenQASM semantics.
//!
//! - [`run_qec_program`] lowers into the packed compiled Clifford sampler and applies
//!   Pauli noise by XORing sensitivity rows onto packed measurement records.
//! - [`run_qec_program_reference`] runs one statevector simulation per shot, as a
//!   correctness oracle for small programs.
//! - [`run_qec_program_with_strategy`] dispatches non-Clifford observable programs
//!   through exact light-cone SPD, CAMPS, then an exact tensor-network scalar fallback.
//! - [`compile_qec_program_rows`] lowers basis measurements and `MPP` records into packed
//!   X/Z Pauli rows without executing gates, resets, or noise.
//! - [`QecProgram::detector_error_model`] derives the [`DetectorErrorModel`] for export
//!   to matching and belief-propagation decoders.
//! - [`UnionFindDecoder`] decodes packed detector samples against a graphlike model.

mod camps_prefix;
/// Treewidth-aware cut-selection heuristics, benchmark-only: the dispatcher follows a
/// fixed SPD -> CAMPS -> tensor-network ladder and does not use them.
#[cfg(feature = "bench-internal")]
pub mod cut_selection;
mod decoder;
mod dem;
mod noise;
pub mod observable_reroute;
mod parse;
mod result;
mod runner;
mod t_sampler;

pub use decoder::UnionFindDecoder;
pub use dem::{DetectorErrorModel, ErrorMechanism};
pub use parse::parse_qec_program;
pub use result::{QecObservableEstimate, QecSampleResult};
#[cfg(feature = "bench-internal")]
pub use runner::{QecProfiledCounts, QecProfiledSampler, compile_qec_profiled_sampler};
pub use runner::{run_qec_program, run_qec_program_reference};
pub use t_sampler::{
    QecObservableReroute, QecTStrategy, run_qec_program_spd_rerouted, run_qec_program_with_strategy,
};

use crate::circuit::{
    Circuit, append_axis_to_z_rotation, append_parity_rotations, append_z_to_axis_rotation,
};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{PackedShots, PauliVec, get_bit, set_bit};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Pauli basis used by QEC measurements and Pauli products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QecBasis {
    X,
    Y,
    Z,
}

impl From<QecBasis> for PauliAxis {
    fn from(basis: QecBasis) -> Self {
        match basis {
            QecBasis::X => PauliAxis::X,
            QecBasis::Y => PauliAxis::Y,
            QecBasis::Z => PauliAxis::Z,
        }
    }
}

/// One Pauli term in an MPP-style measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QecPauli {
    pub basis: QecBasis,
    pub qubit: usize,
}

impl QecPauli {
    pub fn new(basis: QecBasis, qubit: usize) -> Self {
        Self { basis, qubit }
    }

    pub fn x(qubit: usize) -> Self {
        Self::new(QecBasis::X, qubit)
    }

    pub fn y(qubit: usize) -> Self {
        Self::new(QecBasis::Y, qubit)
    }

    pub fn z(qubit: usize) -> Self {
        Self::new(QecBasis::Z, qubit)
    }
}

/// Reference to a previous measurement record.
///
/// A lookback counts back from the records that precede the referencing op, so
/// `Lookback(1)` is the most recent measurement before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QecRecordRef {
    Absolute(usize),
    Lookback(usize),
}

impl QecRecordRef {
    pub fn absolute(index: usize) -> Self {
        Self::Absolute(index)
    }

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
/// Probabilities are validated on append to a [`QecProgram`]; probability zero makes
/// the annotation inactive.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum QecNoise {
    /// With probability `p`, apply X to each target.
    XError(f64),
    /// With probability `p`, apply Z to each target.
    ZError(f64),
    /// Per target, apply each of X, Y, Z with probability `p / 3`.
    Depolarize1(f64),
    /// Per target pair, apply each of the 15 non-identity two-qubit Paulis with
    /// probability `p / 15`. The target list must have even length.
    Depolarize2(f64),
}

impl QecNoise {
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
#[non_exhaustive]
pub enum QecOp {
    /// Standard PRISM-Q gate operation. The compiled runner requires Clifford
    /// gates; the reference runner accepts any gate the statevector backend
    /// supports.
    Gate { gate: Gate, targets: Vec<usize> },
    /// Single-qubit measurement. Produces one record.
    Measure { basis: QecBasis, qubit: usize },
    /// Pauli-product (`MPP`) measurement. Produces one record, the parity of the
    /// listed terms.
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
    /// Final-state estimator `coefficient * <P>`, `P` the product of `terms`. Must be
    /// terminal (no gate, measurement, reset, or active noise may follow) and may
    /// reference only qubits not single-qubit-measured since their last reset.
    /// Estimates land in [`QecSampleResult::expectation_values`] in op order.
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
    /// Feed-forward: `body` executes iff the parity over `records` equals
    /// `expected`.
    ///
    /// The predicate has the shape a detector has, because that is what
    /// adaptive correction reads. `body` admits gates and resets only: the
    /// record space is a static address space that detectors and observables
    /// index, so a measurement whose execution depends on a record would make
    /// those indices depend on the shot.
    Feedforward {
        records: Vec<QecRecordRef>,
        expected: bool,
        body: Vec<QecOp>,
    },
    /// Pauli-noise annotation applied at this point in the program.
    Noise {
        channel: QecNoise,
        targets: Vec<usize>,
    },
    /// Scheduling separator with no semantic effect, kept for the text format.
    Tick,
}

/// Packed Pauli row for one QEC measurement record.
///
/// The row names the Hermitian operator measured, with `Y` carried as both the
/// `x` and `z` bit of its qubit. A consumer that rebuilds the operator as a
/// per-qubit product of `X` and `Z` recovers `(-i)^k` times it, for `k` the
/// number of `Y` letters, since `XZ = -iY`. The row carries no sign of its own;
/// the measured eigenvalue comes from the state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QecMeasurementRow {
    num_qubits: usize,
    pauli: PauliVec,
    weight: usize,
}

impl QecMeasurementRow {
    /// Build a row from Pauli-product terms; rejects an empty list or a repeated qubit.
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

    pub fn single(num_qubits: usize, basis: QecBasis, qubit: usize) -> Result<Self> {
        Self::from_terms(num_qubits, &[QecPauli::new(basis, qubit)])
    }

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of non-identity Pauli terms.
    pub fn weight(&self) -> usize {
        self.weight
    }

    pub fn x_mask(&self) -> &[u64] {
        &self.pauli.x
    }

    pub fn z_mask(&self) -> &[u64] {
        &self.pauli.z
    }

    /// Pauli on `qubit`, or `None` for identity or an out-of-range qubit.
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

    /// Non-identity terms in ascending qubit order.
    pub fn terms(&self) -> Vec<QecPauli> {
        let mut terms = Vec::with_capacity(self.weight);
        for qubit in 0..self.num_qubits {
            if let Some(basis) = self.pauli_at(qubit) {
                terms.push(QecPauli::new(basis, qubit));
            }
        }
        terms
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

    pub fn num_measurements(&self) -> usize {
        self.measurement_rows.len()
    }

    pub fn num_detectors(&self) -> usize {
        self.detector_rows.len()
    }

    pub fn num_observables(&self) -> usize {
        self.observable_rows.len()
    }

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

    pub fn detector_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.detector_rows)
    }

    pub fn observable_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.observable_rows)
    }

    pub fn postselection_parities(&self, measurements: &PackedShots) -> Result<PackedShots> {
        measurements.parity_rows(&self.postselection_rows)
    }
}

/// Options for running a native QEC program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QecOptions {
    pub shots: usize,
    pub seed: u64,
    /// Maximum shots per compiled-runner batch, bounding peak measurement-matrix memory.
    /// `None` means one batch and `Some(0)` is rejected. The reference runner ignores it.
    pub chunk_size: Option<usize>,
    /// When `false`, [`QecSampleResult::measurements`] keeps only its column count and
    /// holds zero shots; detectors and observables are always populated.
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
    pub fn new(num_qubits: usize) -> Self {
        Self::with_options(num_qubits, QecOptions::default())
    }

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

    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    pub fn options(&self) -> QecOptions {
        self.options
    }

    pub fn set_options(&mut self, options: QecOptions) {
        self.options = options;
    }

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

    pub fn num_detectors(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, QecOp::Detector { .. }))
            .count()
    }

    /// Observable slot count, `max included index + 1`.
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

    /// Number of `EXP_VAL` ops.
    pub fn num_expectation_values(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, QecOp::ExpectationValue { .. }))
            .count()
    }

    /// The `EXP_VAL` ops in op order as `(terms, coefficient)`.
    pub fn expectation_value_ops(&self) -> Vec<(&[QecPauli], f64)> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                QecOp::ExpectationValue { terms, coefficient } => {
                    Some((terms.as_slice(), *coefficient))
                }
                _ => None,
            })
            .collect()
    }

    pub fn push_op(&mut self, op: QecOp) -> Result<()> {
        self.validate_op(&op, self.num_measurements())?;
        self.ops.push(op);
        Ok(())
    }

    pub fn push_gate(&mut self, gate: Gate, targets: &[usize]) -> Result<()> {
        self.push_op(QecOp::Gate {
            gate,
            targets: targets.to_vec(),
        })
    }

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

    pub fn expectation_value(&mut self, terms: &[QecPauli], coefficient: f64) -> Result<()> {
        self.push_op(QecOp::ExpectationValue {
            terms: terms.to_vec(),
            coefficient,
        })
    }

    pub fn postselect(&mut self, records: &[QecRecordRef], expected: bool) -> Result<()> {
        self.push_op(QecOp::Postselect {
            records: records.to_vec(),
            expected,
        })
    }

    pub fn noise(&mut self, channel: QecNoise, targets: &[usize]) -> Result<()> {
        self.push_op(QecOp::Noise {
            channel,
            targets: targets.to_vec(),
        })
    }

    /// Append a feed-forward correction; see [`QecOp::Feedforward`] for what `body` admits.
    pub fn feedforward(
        &mut self,
        records: &[QecRecordRef],
        expected: bool,
        body: Vec<QecOp>,
    ) -> Result<()> {
        self.push_op(QecOp::Feedforward {
            records: records.to_vec(),
            expected,
            body,
        })
    }

    /// Visit every non-measurement op with the count of records emitted before it.
    fn visit_ops_with_measurement_count(
        &self,
        mut visit: impl FnMut(&QecOp, usize) -> Result<()>,
    ) -> Result<()> {
        let mut next_measurement = 0;
        for op in &self.ops {
            if matches!(
                op,
                QecOp::Measure { .. } | QecOp::MeasurePauliProduct { .. }
            ) {
                next_measurement += 1;
                continue;
            }
            visit(op, next_measurement)?;
        }
        Ok(())
    }

    /// Resolve detector rows to absolute measurement record indices.
    pub fn detector_rows(&self) -> Result<Vec<Vec<usize>>> {
        let mut rows = Vec::new();
        self.visit_ops_with_measurement_count(|op, next_measurement| {
            if let QecOp::Detector { records, .. } = op {
                rows.push(resolve_records(records, next_measurement)?);
            }
            Ok(())
        })?;
        Ok(rows)
    }

    /// Resolve observable rows to absolute measurement record indices.
    pub fn observable_rows(&self) -> Result<Vec<Vec<usize>>> {
        let mut rows: Vec<Vec<usize>> = Vec::new();
        self.visit_ops_with_measurement_count(|op, next_measurement| {
            if let QecOp::ObservableInclude {
                observable,
                records,
            } = op
            {
                if rows.len() <= *observable {
                    rows.resize_with(*observable + 1, Vec::new);
                }
                rows[*observable].extend(resolve_records(records, next_measurement)?);
            }
            Ok(())
        })?;
        Ok(rows)
    }

    /// Resolve postselection rows to absolute measurement record indices.
    pub fn postselection_rows(&self) -> Result<Vec<(Vec<usize>, bool)>> {
        let mut rows = Vec::new();
        self.visit_ops_with_measurement_count(|op, next_measurement| {
            if let QecOp::Postselect { records, expected } = op {
                rows.push((resolve_records(records, next_measurement)?, *expected));
            }
            Ok(())
        })?;
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
            QecOp::Feedforward {
                records,
                body,
                expected: _,
            } => {
                if records.is_empty() {
                    return Err(PrismError::InvalidParameter {
                        message: "feed-forward predicate requires at least one record".to_string(),
                    });
                }
                if body.is_empty() {
                    return Err(PrismError::InvalidParameter {
                        message: "feed-forward body requires at least one operation".to_string(),
                    });
                }
                resolve_records(records, next_measurement)?;
                for inner in body {
                    if !matches!(inner, QecOp::Gate { .. } | QecOp::Reset { .. }) {
                        return Err(PrismError::InvalidParameter {
                            message: format!(
                                "feed-forward body admits gates and resets only, got `{}`",
                                qec_op_name(inner)
                            ),
                        });
                    }
                    self.validate_op(inner, next_measurement)?;
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

fn qec_op_name(op: &QecOp) -> &'static str {
    match op {
        QecOp::Gate { .. } => "gate",
        QecOp::Measure { .. } => "M",
        QecOp::MeasurePauliProduct { .. } => "MPP",
        QecOp::Reset { .. } => "R",
        QecOp::Detector { .. } => "DETECTOR",
        QecOp::ObservableInclude { .. } => "OBSERVABLE_INCLUDE",
        QecOp::ExpectationValue { .. } => "EXP_VAL",
        QecOp::Postselect { .. } => "POSTSELECT",
        QecOp::Feedforward { .. } => "FEEDFORWARD",
        QecOp::Noise { .. } => "noise",
        QecOp::Tick => "TICK",
    }
}

/// Compile measurement-record operations into packed QEC row metadata.
///
/// Lowers `M` and `MPP` into the packed X/Z rows the compiled sampler uses and resolves
/// detector, observable, and postselection references to absolute record indices.
/// Rejects gates, resets, active noise, `EXP_VAL`, and `FEEDFORWARD`; zero-probability
/// noise is skipped. [`run_qec_program`] executes a full program.
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
                    reason: "QEC row compilation has no row representation for `EXP_VAL`; \
                             use `run_qec_program`"
                        .to_string(),
                });
            }
            QecOp::Feedforward { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: "QEC row compilation has no row representation for `FEEDFORWARD`; \
                             use `run_qec_program_reference`"
                        .to_string(),
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

pub(crate) fn qec_terms_to_pauli(terms: &[QecPauli]) -> Vec<PauliTerm> {
    terms
        .iter()
        .map(|t| PauliTerm::new(t.qubit, t.basis.into()))
        .collect()
}

/// Reject reuse of a qubit measured in a non-Z basis before its next reset.
///
/// Both lowerings leave a basis-measured qubit in the Z frame, not the basis it named,
/// which is unobservable only while the qubit is reset before reuse (the contract
/// [`run_qec_program`] documents). Z measurements rotate nothing, and `MPP` undoes each
/// term's rotation before taking the record, so neither leaves a rotation behind.
pub(crate) fn validate_measured_qubit_reuse(program: &QecProgram) -> Result<()> {
    let reuse = |qubit: usize| PrismError::InvalidParameter {
        message: format!(
            "qubit {qubit} was measured in a non-Z basis and must be reset before it is used \
             again: a basis measurement leaves the qubit in the Z frame, not in the basis it \
             named"
        ),
    };
    let mut rotated = vec![false; program.num_qubits()];
    for op in program.ops() {
        match op {
            QecOp::Gate { targets, .. } => {
                if let Some(&qubit) = targets.iter().find(|&&q| rotated[q]) {
                    return Err(reuse(qubit));
                }
            }
            QecOp::Measure { basis, qubit } => {
                if rotated[*qubit] {
                    return Err(reuse(*qubit));
                }
                rotated[*qubit] = *basis != QecBasis::Z;
            }
            QecOp::MeasurePauliProduct { terms } => {
                if let Some(term) = terms.iter().find(|t| rotated[t.qubit]) {
                    return Err(reuse(term.qubit));
                }
            }
            QecOp::Reset { qubit, .. } => rotated[*qubit] = false,
            _ => {}
        }
    }
    Ok(())
}

/// Validate `EXP_VAL` placement for execution.
///
/// Terminality: no gate, measurement, reset, or active noise may follow an
/// `EXP_VAL` op, so "final state" is well defined on every path.
/// Liveness: an `EXP_VAL` term may not reference a qubit that was
/// single-qubit-measured after its last reset. Liveness keeps the sampled
/// post-measurement expectation equal to the measurement-stripped
/// pure-state expectation the analytical strategies evaluate (the Pauli
/// commutes with every measurement projector when their supports are
/// disjoint). Pauli-product measurements do not affect liveness: the
/// deferred lowering measures a scratch alias and the cross terms of the
/// projected state cancel exactly.
pub(crate) fn validate_qec_exp_val_placement(program: &QecProgram) -> Result<()> {
    let terminal_violation = |op_name: &str| PrismError::InvalidParameter {
        message: format!("`EXP_VAL` must be terminal: `{op_name}` appears after an `EXP_VAL` op"),
    };
    let mut seen_exp_val = false;
    let mut measured_since_reset = vec![false; program.num_qubits()];
    for op in program.ops() {
        match op {
            QecOp::ExpectationValue { terms, .. } => {
                seen_exp_val = true;
                if let Some(term) = terms.iter().find(|t| measured_since_reset[t.qubit]) {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "`EXP_VAL` term on qubit {}: qubit was measured after its last \
                             reset; expectation values are defined only on live qubits",
                            term.qubit
                        ),
                    });
                }
            }
            QecOp::Gate { gate, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation(gate.name()));
                }
            }
            QecOp::Measure { qubit, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("M"));
                }
                measured_since_reset[*qubit] = true;
            }
            QecOp::MeasurePauliProduct { .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("MPP"));
                }
            }
            QecOp::Reset { qubit, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("R"));
                }
                measured_since_reset[*qubit] = false;
            }
            QecOp::Noise { channel, .. } if channel.probability() > 0.0 => {
                if seen_exp_val {
                    return Err(terminal_violation(channel.name()));
                }
            }
            QecOp::Feedforward { .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("FEEDFORWARD"));
                }
                // A conditional reset does not restore liveness: the shots where
                // the predicate is false leave the qubit collapsed.
            }
            QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Noise { .. }
            | QecOp::Tick => {}
        }
    }
    Ok(())
}

pub(super) fn append_basis_to_z_rotation(circuit: &mut Circuit, basis: QecBasis, qubit: usize) {
    append_axis_to_z_rotation(circuit, basis.into(), qubit);
}

pub(super) fn append_z_to_basis_rotation(circuit: &mut Circuit, basis: QecBasis, qubit: usize) {
    append_z_to_axis_rotation(circuit, basis.into(), qubit);
}

/// Lower a Pauli-product measurement onto a scratch qubit holding |0>; the
/// caller measures the scratch afterward. See [`append_parity_rotations`].
pub(super) fn append_mpp_parity_rotations(
    circuit: &mut Circuit,
    terms: &[QecPauli],
    scratch: usize,
) {
    append_parity_rotations(circuit, &qec_terms_to_pauli(terms), scratch);
}

pub(super) fn qec_non_clifford_error(gate: &Gate) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "QEC compiled runner".to_string(),
        reason: format!(
            "compiled QEC runner requires Clifford gates, got `{}`",
            gate.name()
        ),
    }
}

pub(super) fn ensure_lowered_record_count(
    program: &QecProgram,
    produced: usize,
    stage: &str,
) -> Result<()> {
    if produced != program.num_measurements() {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC {stage} lowering produced {produced} records, expected {}",
                program.num_measurements()
            ),
        });
    }
    Ok(())
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
                message: format!("Pauli product contains duplicate qubit {}", term.qubit),
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

    if matches!(channel, QecNoise::Depolarize2(_)) && !targets.len().is_multiple_of(2) {
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
