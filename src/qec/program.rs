//! The [`QecProgram`] builder: op appending, record resolution, and op validation.

use super::{
    QEC_PROBABILITY_SUM_TOLERANCE, QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram,
    QecRecordRef, QecSampleResult, parse_qec_program, parse_stim_circuit, qec_op_name,
    resolve_records,
};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

impl QecProgram {
    pub fn new(num_qubits: usize) -> Self {
        Self::with_options(num_qubits, QecOptions::default())
    }

    pub fn with_options(num_qubits: usize, options: QecOptions) -> Self {
        Self {
            num_qubits,
            ops: Vec::new(),
            options,
            num_measurements: 0,
            num_detectors: 0,
        }
    }

    /// Create a program from operations, validating record references as
    /// operations are appended.
    pub fn from_ops(num_qubits: usize, options: QecOptions, ops: Vec<QecOp>) -> Result<Self> {
        let mut program = Self::with_options(num_qubits, options);
        for (index, op) in ops.iter().enumerate() {
            let previous = index.checked_sub(1).map(|at| &ops[at]);
            program.validate_op(op, previous, program.num_measurements)?;
            program.count_op(op);
        }
        program.ops = ops;
        Ok(program)
    }

    /// Parse a native measurement-record QEC program.
    pub fn from_text(input: &str) -> Result<Self> {
        parse_qec_program(input)
    }

    /// Parse a circuit in the Stim circuit file format; see [`parse_stim_circuit`].
    pub fn from_stim(input: &str) -> Result<Self> {
        parse_stim_circuit(input)
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
        self.num_measurements
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
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

    /// True when any op is a leakage annotation, which routes
    /// [`run_qec_program`](super::run_qec_program) to the leakage sampler and fills
    /// [`QecSampleResult::heralds`].
    pub fn has_leakage(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, QecOp::Noise { channel, .. } if channel.is_leakage()))
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
        self.validate_op(&op, self.ops.last(), self.num_measurements)?;
        self.count_op(&op);
        self.ops.push(op);
        Ok(())
    }

    fn count_op(&mut self, op: &QecOp) {
        if op.produces_record() {
            self.num_measurements += 1;
        } else if matches!(op, QecOp::Detector { .. }) {
            self.num_detectors += 1;
        }
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
            if op.produces_record() {
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

    fn validate_op(
        &self,
        op: &QecOp,
        previous: Option<&QecOp>,
        next_measurement: usize,
    ) -> Result<()> {
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
                if let Some(qubit) = targets
                    .iter()
                    .enumerate()
                    .find_map(|(at, qubit)| targets[..at].contains(qubit).then_some(qubit))
                {
                    return Err(PrismError::InvalidParameter {
                        message: format!("gate `{}` names qubit {qubit} twice", gate.name()),
                    });
                }
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
                validate_records(records, next_measurement)?;
                validate_finite_values(coords, "detector coordinate")?;
            }
            QecOp::ObservableInclude { records, .. } | QecOp::Postselect { records, .. } => {
                validate_records(records, next_measurement)?;
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
                validate_records(records, next_measurement)?;
                for inner in body {
                    if !matches!(inner, QecOp::Gate { .. } | QecOp::Reset { .. }) {
                        return Err(PrismError::InvalidParameter {
                            message: format!(
                                "feed-forward body admits gates and resets only, got `{}`",
                                qec_op_name(inner)
                            ),
                        });
                    }
                    self.validate_op(inner, None, next_measurement)?;
                }
            }
            QecOp::Noise { channel, targets } => {
                validate_noise(channel, targets, self.num_qubits)?;
            }
            QecOp::CorrelatedError { branches } => {
                if branches.is_empty() {
                    return Err(PrismError::InvalidParameter {
                        message: "correlated error requires at least one branch".to_string(),
                    });
                }
                if branches.len() > MAX_CORRELATED_BRANCHES {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "correlated error chain admits at most {MAX_CORRELATED_BRANCHES} \
                             branches"
                        ),
                    });
                }
                for branch in branches {
                    validate_probability(branch.probability, "correlated error")?;
                    if branch.terms.is_empty() {
                        return Err(PrismError::InvalidParameter {
                            message: "correlated error branch requires at least one Pauli term"
                                .to_string(),
                        });
                    }
                    validate_pauli_terms(&branch.terms, self.num_qubits)?;
                }
            }
            QecOp::HeraldedErase { probability, qubit } => {
                validate_probability(*probability, "heralded erasure")?;
                validate_qubit(*qubit, self.num_qubits)?;
            }
            QecOp::RecordFlip { probability } => {
                validate_probability(*probability, "record flip")?;
                let follows_record = previous.is_some_and(|last| {
                    last.produces_record() || matches!(last, QecOp::RecordFlip { .. })
                });
                if !follows_record {
                    return Err(PrismError::InvalidParameter {
                        message: "a record flip must directly follow the op that produced its \
                                  record"
                            .to_string(),
                    });
                }
            }
            QecOp::Pad { .. } | QecOp::Tick => {}
        }
        Ok(())
    }
}

/// Branches a correlated error chain may hold: the compiled sampler packs a firing's
/// branch into 16 bits.
const MAX_CORRELATED_BRANCHES: usize = u16::MAX as usize;

fn validate_probability(p: f64, label: &str) -> Result<()> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(PrismError::InvalidParameter {
            message: format!("{label} probability must be finite and in [0, 1]"),
        });
    }
    Ok(())
}

fn validate_records(records: &[QecRecordRef], next_measurement: usize) -> Result<()> {
    records
        .iter()
        .try_for_each(|record| record.resolve(next_measurement).map(drop))
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

pub(super) fn validate_pauli_terms(terms: &[QecPauli], num_qubits: usize) -> Result<()> {
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

fn validate_noise(channel: &QecNoise, targets: &[usize], num_qubits: usize) -> Result<()> {
    let name = channel.name();
    let args = channel.args();
    let total = channel.probability();
    if args
        .iter()
        .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || !(0.0..=1.0 + QEC_PROBABILITY_SUM_TOLERANCE).contains(&total)
    {
        return Err(PrismError::InvalidParameter {
            message: if args.len() == 1 {
                format!("{name} probability must be finite and in [0, 1]")
            } else {
                format!("{name} probabilities must be finite, in [0, 1], and sum to at most 1")
            },
        });
    }

    if targets.is_empty() {
        return Err(PrismError::InvalidParameter {
            message: format!("{name} requires at least one target"),
        });
    }

    if channel.is_pair() {
        if !targets.len().is_multiple_of(2) {
            return Err(PrismError::InvalidParameter {
                message: format!("{name} requires an even number of targets"),
            });
        }
        for pair in targets.as_chunks::<2>().0 {
            if pair[0] == pair[1] {
                return Err(PrismError::InvalidParameter {
                    message: format!("{name} target pairs must use distinct qubits"),
                });
            }
        }
    }

    validate_qubits(targets.iter().copied(), num_qubits)
}
