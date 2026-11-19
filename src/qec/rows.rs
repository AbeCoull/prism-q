//! Packed measurement rows and [`compile_qec_program_rows`], which lowers a program's
//! records into them without executing it.

use super::program::validate_pauli_terms;
use super::{QecBasis, QecOp, QecPauli, QecProgram, qec_op_name};
use crate::error::{PrismError, Result};
use crate::sim::compiled::{PackedShots, PauliVec, get_bit, set_bit};

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

/// Compile measurement-record operations into packed QEC row metadata.
///
/// Lowers `M` and `MPP` into the packed X/Z rows the compiled sampler uses and resolves
/// detector, observable, and postselection references to absolute record indices.
/// Rejects gates, resets, active noise, `EXP_VAL`, and `FEEDFORWARD`; zero-probability
/// noise is skipped. [`run_qec_program`](super::run_qec_program) executes a full program.
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
            QecOp::HeraldedErase { .. } | QecOp::Pad { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC row compiler".to_string(),
                    reason: format!(
                        "QEC row compilation has no row representation for `{}`; \
                         use `run_qec_program`",
                        qec_op_name(op)
                    ),
                });
            }
            QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
            QecOp::Noise { .. } | QecOp::CorrelatedError { .. } | QecOp::RecordFlip { .. }
                if !op.is_active_noise() => {}
            QecOp::Noise { .. } | QecOp::CorrelatedError { .. } | QecOp::RecordFlip { .. } => {
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
