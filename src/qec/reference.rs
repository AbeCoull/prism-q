//! The noiseless reference run behind [`QecDetectorReference::Noiseless`]: one noiseless
//! execution in which every random outcome is 0, and its detector and observable
//! parities.

use std::collections::BTreeMap;

use super::noise::QecFeedforwardFlips;
use super::{QecBasis, QecDetectorReference, QecOp, QecProgram};
use crate::backend::Backend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::{Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::compiled::{CompiledSampler, PackedShots, get_bit, xor_words};

/// Reference parities of a program's detectors and observables, one bit each.
pub(super) struct QecReference {
    detectors: Vec<u64>,
    observables: Vec<u64>,
}

impl QecReference {
    pub(super) fn from_records(
        records: &[u64],
        detector_rows: &[Vec<usize>],
        observable_rows: &[Vec<usize>],
    ) -> Self {
        Self {
            detectors: row_parities(records, detector_rows),
            observables: row_parities(records, observable_rows),
        }
    }

    /// XOR the reference into packed detector and observable samples.
    pub(super) fn apply(&self, detectors: &mut PackedShots, observables: &mut PackedShots) {
        detectors.xor_columns(&self.detectors);
        observables.xor_columns(&self.observables);
    }

    pub(super) fn observable(&self, observable: usize) -> bool {
        get_bit(&self.observables, observable)
    }
}

fn row_parities(records: &[u64], rows: &[Vec<usize>]) -> Vec<u64> {
    let mut parities = vec![0u64; rows.len().div_ceil(64)];
    for (index, row) in rows.iter().enumerate() {
        let parity = row
            .iter()
            .fold(false, |acc, &record| acc ^ get_bit(records, record));
        parities[index / 64] |= u64::from(parity) << (index % 64);
    }
    parities
}

/// The reference a program's results are measured against, or `None` when they report
/// raw parity: under [`QecDetectorReference::Raw`], with no detector or observable to
/// shift, or when a non-Clifford gate leaves the program without a reference run.
pub(super) fn program_reference(program: &QecProgram) -> Result<Option<QecReference>> {
    if program.options().detector_reference == QecDetectorReference::Raw
        || program.num_detectors() + program.num_observables() == 0
    {
        return Ok(None);
    }
    let Some(records) = forced_tableau_records(program)? else {
        return Ok(None);
    };
    Ok(Some(QecReference::from_records(
        &records,
        &program.detector_rows()?,
        &program.observable_rows()?,
    )))
}

/// Points of `sampler`'s noiseless outcome space at which every record that is random
/// given the records before it takes 0: the records before `corrections` land, and the
/// records after. Feed-forward corrections land in op order as the records they read
/// settle, so a correction can decide a later record that the forcing then reads.
///
/// The first point depends on the outcome space alone, not on the reference bits the
/// sampler happened to compile with. The second is the reference run whenever no reset
/// collapses a qubit that later records still depend on; see [`has_entangled_reset`].
pub(super) fn forced_zero_records(
    sampler: &CompiledSampler,
    corrections: Option<&QecFeedforwardFlips>,
) -> (Vec<u64>, Vec<u64>) {
    let mut records = sampler.ref_bits_packed().to_vec();
    let mut corrected = vec![0u64; records.len()];
    let mut pending: Vec<_> = corrections
        .map(|flips| flips.corrections().collect())
        .unwrap_or_default();
    pending.sort_by_key(|&(_, _, flips): &(&[usize], bool, &[usize])| flips[0]);
    let mut next = 0;
    let mut land = |records: &[u64], corrected: &mut [u64], until: usize| {
        while next < pending.len() && pending[next].2[0] <= until {
            let (condition, invert, flips) = pending[next];
            let fire = condition.iter().fold(invert, |acc, &record| {
                acc ^ get_bit(records, record) ^ get_bit(corrected, record)
            });
            if fire {
                for &record in flips {
                    corrected[record / 64] ^= 1u64 << (record % 64);
                }
            }
            next += 1;
        }
    };
    for (pivot, row) in echelon_by_first_record(sampler.flip_rows()) {
        land(&records, &mut corrected, pivot);
        if get_bit(&records, pivot) ^ get_bit(&corrected, pivot) {
            let word = pivot / 64;
            xor_words(&mut records[word..], &row[word..]);
        }
    }
    land(&records, &mut corrected, usize::MAX);
    xor_words(&mut corrected, &records);
    (records, corrected)
}

/// A basis of the span of `rows` with distinct lowest set bits, keyed and sorted by
/// that bit. Rows that compile in record order already have distinct lowest bits and
/// pass through without reduction.
fn echelon_by_first_record(rows: &[Vec<u64>]) -> Vec<(usize, Vec<u64>)> {
    let mut keyed: Vec<(usize, Vec<u64>)> = rows
        .iter()
        .filter_map(|row| first_record(row).map(|pivot| (pivot, row.clone())))
        .collect();
    keyed.sort_unstable_by_key(|&(pivot, _)| pivot);
    if keyed.windows(2).all(|pair| pair[0].0 < pair[1].0) {
        return keyed;
    }
    let mut basis: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    for (_, mut row) in keyed {
        while let Some(pivot) = first_record(&row) {
            if let Some(reduced) = basis.get(&pivot) {
                let word = pivot / 64;
                xor_words(&mut row[word..], &reduced[word..]);
            } else {
                basis.insert(pivot, row);
                break;
            }
        }
    }
    basis.into_iter().collect()
}

fn first_record(row: &[u64]) -> Option<usize> {
    row.iter()
        .position(|&word| word != 0)
        .map(|word| word * 64 + row[word].trailing_zeros() as usize)
}

/// Whether some reset collapses a qubit that a multi-qubit gate or Pauli-product
/// measurement touched since the qubit was last measured. Such a collapse can decide
/// later records of other qubits, and the deferred lowering, which hands the reset a
/// fresh alias, never forces it; [`forced_tableau_records`] does.
pub(super) fn has_entangled_reset(program: &QecProgram) -> bool {
    let mut entangled = vec![false; program.num_qubits()];
    let mut visit = |op: &QecOp| -> bool {
        match op {
            QecOp::Gate { targets, .. } if targets.len() > 1 => {
                for &target in targets {
                    entangled[target] = true;
                }
            }
            QecOp::MeasurePauliProduct { terms } if terms.len() > 1 => {
                for term in terms {
                    entangled[term.qubit] = true;
                }
            }
            QecOp::Measure { qubit, .. } => entangled[*qubit] = false,
            QecOp::Reset { qubit, .. } => {
                return std::mem::replace(&mut entangled[*qubit], false);
            }
            _ => {}
        }
        false
    };
    program.ops().iter().any(|op| match op {
        QecOp::Feedforward { body, .. } => body.iter().any(&mut visit),
        op => visit(op),
    })
}

/// Records of the noiseless reference run, executed op by op on a stabilizer tableau
/// whose random outcomes are all 0. A reset collapses its qubit in the reset's basis,
/// and a feed-forward body runs when its predicate holds on the records so far. `None`
/// when a gate is not Clifford.
pub(super) fn forced_tableau_records(program: &QecProgram) -> Result<Option<Vec<u64>>> {
    let gates_are_clifford = |ops: &[QecOp]| {
        ops.iter()
            .all(|op| !matches!(op, QecOp::Gate { gate, .. } if !gate.is_clifford()))
    };
    if !program.ops().iter().all(|op| match op {
        QecOp::Feedforward { body, .. } => gates_are_clifford(body),
        op => gates_are_clifford(std::slice::from_ref(op)),
    }) {
        return Ok(None);
    }

    let num_measurements = program.num_measurements();
    let scratch = program.num_qubits();
    let mut backend = StabilizerBackend::new(0);
    backend.forced_outcome = Some(false);
    backend.init(super::qec_lowered_num_qubits(program), num_measurements)?;
    let mut records = vec![0u64; num_measurements.div_ceil(64)];
    let mut next_record = 0usize;
    for op in program.ops() {
        let outcome = match op {
            QecOp::Gate { gate, targets } => {
                apply_gate(&mut backend, gate.clone(), targets)?;
                None
            }
            QecOp::Reset { basis, qubit } => {
                reset_in_basis(&mut backend, *basis, *qubit)?;
                None
            }
            QecOp::Measure { basis, qubit } => {
                rotate_basis_to_z(&mut backend, *basis, *qubit)?;
                Some(measure(&mut backend, *qubit, next_record)?)
            }
            QecOp::MeasurePauliProduct { terms } => {
                backend.reset(scratch)?;
                for term in terms {
                    rotate_basis_to_z(&mut backend, term.basis, term.qubit)?;
                    apply_gate(&mut backend, Gate::Cx, &[term.qubit, scratch])?;
                }
                for term in terms.iter().rev() {
                    rotate_z_to_basis(&mut backend, term.basis, term.qubit)?;
                }
                Some(measure(&mut backend, scratch, next_record)?)
            }
            QecOp::HeraldedErase { .. } => Some(false),
            QecOp::Pad { value } => Some(*value),
            QecOp::Feedforward {
                records: condition,
                expected,
                body,
            } => {
                let parity = super::resolve_records(condition, next_record)?
                    .into_iter()
                    .fold(false, |acc, record| acc ^ get_bit(&records, record));
                if parity == *expected {
                    for inner in body {
                        match inner {
                            QecOp::Gate { gate, targets } => {
                                apply_gate(&mut backend, gate.clone(), targets)?
                            }
                            QecOp::Reset { basis, qubit } => {
                                reset_in_basis(&mut backend, *basis, *qubit)?
                            }
                            _ => {}
                        }
                    }
                }
                None
            }
            QecOp::Noise { .. }
            | QecOp::CorrelatedError { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::ExpectationValue { .. }
            | QecOp::Tick => None,
        };
        if let Some(bit) = outcome {
            records[next_record / 64] |= u64::from(bit) << (next_record % 64);
            next_record += 1;
        }
    }
    Ok(Some(records))
}

fn apply_gate(backend: &mut StabilizerBackend, gate: Gate, targets: &[usize]) -> Result<()> {
    let mut small = SmallVec::new();
    small.extend_from_slice(targets);
    backend.apply(&Instruction::Gate {
        gate,
        targets: small,
    })
}

fn measure(backend: &mut StabilizerBackend, qubit: usize, record: usize) -> Result<bool> {
    backend.apply(&Instruction::Measure {
        qubit,
        classical_bit: record,
    })?;
    Ok(backend.classical_results()[record])
}

fn reset_in_basis(backend: &mut StabilizerBackend, basis: QecBasis, qubit: usize) -> Result<()> {
    rotate_basis_to_z(backend, basis, qubit)?;
    backend.reset(qubit)?;
    rotate_z_to_basis(backend, basis, qubit)
}

fn rotate_basis_to_z(backend: &mut StabilizerBackend, basis: QecBasis, qubit: usize) -> Result<()> {
    match basis {
        QecBasis::X => apply_gate(backend, Gate::H, &[qubit]),
        QecBasis::Y => {
            apply_gate(backend, Gate::Sdg, &[qubit])?;
            apply_gate(backend, Gate::H, &[qubit])
        }
        QecBasis::Z => Ok(()),
    }
}

fn rotate_z_to_basis(backend: &mut StabilizerBackend, basis: QecBasis, qubit: usize) -> Result<()> {
    match basis {
        QecBasis::X => apply_gate(backend, Gate::H, &[qubit]),
        QecBasis::Y => {
            apply_gate(backend, Gate::H, &[qubit])?;
            apply_gate(backend, Gate::S, &[qubit])
        }
        QecBasis::Z => Ok(()),
    }
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
