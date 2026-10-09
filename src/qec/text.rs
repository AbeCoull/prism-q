//! Text export for native QEC programs, the inverse of [`super::parse_qec_program`].

use std::fmt::Write;

use super::{QecBasis, QecOp, QecPauli, QecProgram, QecRecordRef};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

/// Instructions that merge consecutive ops into one line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Line {
    Gate(&'static str),
    Measure(QecBasis),
    Reset(QecBasis),
    Mpp,
}

impl Line {
    fn name(self) -> &'static str {
        match self {
            Self::Gate(name) => name,
            Self::Measure(QecBasis::Z) => "M",
            Self::Measure(QecBasis::X) => "MX",
            Self::Measure(QecBasis::Y) => "MY",
            Self::Reset(QecBasis::Z) => "R",
            Self::Reset(QecBasis::X) => "RX",
            Self::Reset(QecBasis::Y) => "RY",
            Self::Mpp => "MPP",
        }
    }
}

impl QecProgram {
    /// Render the program in the native QEC text format, which
    /// [`parse_qec_program`](super::parse_qec_program) reads back to the same ops.
    ///
    /// Consecutive gates, measurements, resets, and `MPP` products of one kind share a
    /// line. Record references print as `rec[-k]` and parse back as absolute indices.
    /// A `QUBIT_COORDS` line keeps the qubit count when the highest qubit is unused.
    /// [`QecOptions`](super::QecOptions) are not part of the text, and the parser drops
    /// zero-probability Pauli noise but keeps a zero-rate leakage annotation, since
    /// `LEAK(0)` still reads herald columns.
    ///
    /// # Errors
    ///
    /// A gate outside the parser's gate set, or a `FEEDFORWARD` op, which the text
    /// format does not spell.
    pub fn to_text(&self) -> Result<String> {
        let mut out = String::with_capacity(16 * self.ops().len());
        let used = self.ops().iter().filter_map(max_qubit).max();
        if self.num_qubits() > 0 && used.is_none_or(|q| q + 1 < self.num_qubits()) {
            writeln!(out, "QUBIT_COORDS {}", self.num_qubits() - 1).expect("String write");
        }
        let mut open: Option<Line> = None;
        let mut records = 0usize;
        for op in self.ops() {
            let line = match op {
                QecOp::Gate { gate, .. } => Some(Line::Gate(gate_name(gate)?)),
                QecOp::Measure { basis, .. } => Some(Line::Measure(*basis)),
                QecOp::Reset { basis, .. } => Some(Line::Reset(*basis)),
                QecOp::MeasurePauliProduct { .. } => Some(Line::Mpp),
                _ => None,
            };
            if line != open {
                if open.is_some() {
                    out.push('\n');
                }
                if let Some(line) = line {
                    out.push_str(line.name());
                }
                open = line;
            }
            let w = &mut out;
            match op {
                QecOp::Gate { targets, .. } => {
                    for target in targets {
                        write!(w, " {target}").expect("String write");
                    }
                }
                QecOp::Measure { qubit, .. } => {
                    write!(w, " {qubit}").expect("String write");
                    records += 1;
                }
                QecOp::Reset { qubit, .. } => write!(w, " {qubit}").expect("String write"),
                QecOp::MeasurePauliProduct { terms } => {
                    w.push(' ');
                    write_product(w, terms);
                    records += 1;
                }
                QecOp::Detector {
                    records: refs,
                    coords,
                } => {
                    w.push_str("DETECTOR");
                    write_args(w, coords);
                    write_records(w, refs, records);
                    w.push('\n');
                }
                QecOp::ObservableInclude {
                    observable,
                    records: refs,
                } => {
                    write!(w, "OBSERVABLE_INCLUDE({observable})").expect("String write");
                    write_records(w, refs, records);
                    w.push('\n');
                }
                QecOp::Postselect {
                    records: refs,
                    expected,
                } => {
                    w.push_str(if *expected {
                        "POSTSELECT(1)"
                    } else {
                        "POSTSELECT"
                    });
                    write_records(w, refs, records);
                    w.push('\n');
                }
                QecOp::ExpectationValue { terms, coefficient } => {
                    write!(w, "EXP_VAL({coefficient}) ").expect("String write");
                    write_product(w, terms);
                    w.push('\n');
                }
                QecOp::Noise { channel, targets } => {
                    w.push_str(channel.name());
                    write_args(w, &channel.args());
                    for target in targets {
                        write!(w, " {target}").expect("String write");
                    }
                    w.push('\n');
                }
                QecOp::Tick => w.push_str("TICK\n"),
                QecOp::Feedforward { .. } => {
                    return Err(PrismError::InvalidParameter {
                        message: "the QEC text format has no spelling for `FEEDFORWARD`"
                            .to_string(),
                    });
                }
            }
        }
        if open.is_some() {
            out.push('\n');
        }
        Ok(out)
    }
}

fn gate_name(gate: &Gate) -> Result<&'static str> {
    Ok(match gate {
        Gate::Id => "I",
        Gate::X => "X",
        Gate::Y => "Y",
        Gate::Z => "Z",
        Gate::H => "H",
        Gate::S => "S",
        Gate::Sdg => "S_DAG",
        Gate::T => "T",
        Gate::Tdg => "T_DAG",
        Gate::Cx => "CX",
        Gate::Cz => "CZ",
        _ => {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "gate `{}` has no spelling in the QEC text format",
                    gate.name()
                ),
            });
        }
    })
}

fn max_qubit(op: &QecOp) -> Option<usize> {
    match op {
        QecOp::Gate { targets, .. } | QecOp::Noise { targets, .. } => targets.iter().copied().max(),
        QecOp::Measure { qubit, .. } | QecOp::Reset { qubit, .. } => Some(*qubit),
        QecOp::MeasurePauliProduct { terms } | QecOp::ExpectationValue { terms, .. } => {
            terms.iter().map(|term| term.qubit).max()
        }
        _ => None,
    }
}

fn write_args(out: &mut String, args: &[f64]) {
    if args.is_empty() {
        return;
    }
    out.push('(');
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write!(out, "{arg}").expect("String write");
    }
    out.push(')');
}

fn write_records(out: &mut String, refs: &[QecRecordRef], records: usize) {
    for record in refs {
        let distance = match *record {
            QecRecordRef::Absolute(index) => records - index,
            QecRecordRef::Lookback(distance) => distance,
        };
        write!(out, " rec[-{distance}]").expect("String write");
    }
}

fn write_product(out: &mut String, terms: &[QecPauli]) {
    for (i, term) in terms.iter().enumerate() {
        if i > 0 {
            out.push('*');
        }
        let letter = match term.basis {
            QecBasis::X => 'X',
            QecBasis::Y => 'Y',
            QecBasis::Z => 'Z',
        };
        write!(out, "{letter}{}", term.qubit).expect("String write");
    }
}
