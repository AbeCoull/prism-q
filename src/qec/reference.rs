//! The noiseless reference run behind [`QecDetectorReference::Noiseless`]: one noiseless
//! execution in which every random outcome is 0, and its detector and observable
//! parities.

use std::collections::BTreeMap;

use super::noise::QecFeedforwardFlips;
use super::{QecBasis, QecDetectorReference, QecOp, QecProgram};
use crate::backend::stabilizer::kernels::rowmul_words_cached;
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
    let Some(records) = forced_tableau_records(program, true)? else {
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

/// Records of the noiseless reference run with every random outcome 0, on the inverse
/// tableau of [`InverseTableau`]. A reset collapses its qubit in the reset's basis; a
/// basis measurement leaves its qubit in the rotated frame. With `run_feedforward` a
/// feed-forward body runs when its predicate holds on the records so far; without it
/// every body is skipped. `None` when a gate is not Clifford.
pub(super) fn forced_tableau_records(
    program: &QecProgram,
    run_feedforward: bool,
) -> Result<Option<Vec<u64>>> {
    if !gates_are_clifford(program) {
        return Ok(None);
    }

    let scratch = program.num_qubits();
    let mut tableau = InverseTableau::new(super::qec_lowered_num_qubits(program));
    let mut records = vec![0u64; program.num_measurements().div_ceil(64)];
    let mut next_record = 0usize;
    let record = |records: &mut [u64], next_record: &mut usize, outcome: bool| {
        records[*next_record / 64] |= u64::from(outcome) << (*next_record % 64);
        *next_record += 1;
    };
    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => tableau.gate(gate, targets),
            QecOp::Reset { basis, qubit } => tableau.reset(*basis, *qubit),
            QecOp::Measure { basis, qubit } => {
                tableau.basis_to_z(*basis, *qubit);
                let outcome = tableau.measure_z(*qubit);
                record(&mut records, &mut next_record, outcome);
            }
            QecOp::MeasurePauliProduct { terms } => {
                tableau.reset_z(scratch);
                for term in terms {
                    tableau.basis_to_z(term.basis, term.qubit);
                    tableau.cx(term.qubit, scratch);
                }
                for term in terms.iter().rev() {
                    tableau.z_to_basis(term.basis, term.qubit);
                }
                let outcome = tableau.measure_z(scratch);
                record(&mut records, &mut next_record, outcome);
            }
            QecOp::HeraldedErase { .. } | QecOp::Pad { .. } => {
                tableau.reset_z(scratch);
                if matches!(op, QecOp::Pad { value: true }) {
                    tableau.x(scratch);
                }
                let outcome = tableau.measure_z(scratch);
                record(&mut records, &mut next_record, outcome);
            }
            QecOp::Feedforward {
                records: condition,
                expected,
                body,
            } if run_feedforward => {
                let bits = super::resolve_records(condition, next_record)?;
                let parity = bits
                    .iter()
                    .fold(false, |acc, &bit| acc ^ get_bit(&records, bit));
                if parity != *expected {
                    continue;
                }
                for inner in body {
                    match inner {
                        QecOp::Gate { gate, targets } => tableau.gate(gate, targets),
                        QecOp::Reset { basis, qubit } => tableau.reset(*basis, *qubit),
                        _ => {}
                    }
                }
            }
            QecOp::Feedforward { .. }
            | QecOp::Noise { .. }
            | QecOp::CorrelatedError { .. }
            | QecOp::RecordFlip { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::ExpectationValue { .. }
            | QecOp::Tick => {}
        }
    }
    Ok(Some(records))
}

fn gates_are_clifford(program: &QecProgram) -> bool {
    let clifford = |ops: &[QecOp]| {
        ops.iter()
            .all(|op| !matches!(op, QecOp::Gate { gate, .. } if !gate.is_clifford()))
    };
    program.ops().iter().all(|op| match op {
        QecOp::Feedforward { body, .. } => clifford(body),
        op => clifford(std::slice::from_ref(op)),
    })
}

/// Inverse tableau of the Clifford `C` that prepares the state from `|0...0>`: row `2q`
/// holds `C† X_q C` and row `2q + 1` holds `C† Z_q C`, x words then z words, with its
/// sign alongside. A gate `U` maps each row to `C† U† P U C`, a product of at most two
/// rows. A Z measurement of `q` is deterministic exactly when row `2q + 1` has no x
/// bit, and then reads its sign without the row reduction a forward tableau needs.
struct InverseTableau {
    words: usize,
    rows: Vec<u64>,
    signs: Vec<bool>,
    /// Columns with an x bit in the measured row of a collapse.
    pivots: Vec<usize>,
}

impl InverseTableau {
    fn new(num_qubits: usize) -> Self {
        let words = num_qubits.div_ceil(64);
        let stride = 2 * words;
        let mut rows = vec![0u64; 2 * num_qubits * stride];
        for q in 0..num_qubits {
            rows[2 * q * stride + q / 64] |= 1u64 << (q % 64);
            rows[(2 * q + 1) * stride + words + q / 64] |= 1u64 << (q % 64);
        }
        Self {
            words,
            rows,
            signs: vec![false; 2 * num_qubits],
            pivots: Vec::new(),
        }
    }

    fn gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Id => {}
            Gate::X => self.x(targets[0]),
            Gate::Y => {
                self.signs[2 * targets[0]] ^= true;
                self.signs[2 * targets[0] + 1] ^= true;
            }
            Gate::Z => self.signs[2 * targets[0]] ^= true,
            Gate::H => self.h(targets[0]),
            Gate::S => self.mul(2 * targets[0], 2 * targets[0] + 1, 1),
            Gate::Sdg => self.mul(2 * targets[0], 2 * targets[0] + 1, 3),
            Gate::SX | Gate::SXdg => {
                let q = targets[0];
                self.h(q);
                self.mul(
                    2 * q,
                    2 * q + 1,
                    if matches!(gate, Gate::SX) { 1 } else { 3 },
                );
                self.h(q);
            }
            Gate::Cx => self.cx(targets[0], targets[1]),
            Gate::Cz => {
                let (a, b) = (targets[0], targets[1]);
                self.mul(2 * a, 2 * b + 1, 0);
                self.mul(2 * b, 2 * a + 1, 0);
            }
            Gate::Swap => {
                let (a, b) = (targets[0], targets[1]);
                self.swap_rows(2 * a, 2 * b);
                self.swap_rows(2 * a + 1, 2 * b + 1);
            }
            _ => unreachable!("the reference run admits Clifford gates only"),
        }
    }

    /// Set row `dst` to `i^extra · src · dst`.
    #[inline(always)]
    fn mul(&mut self, dst: usize, src: usize, extra: u64) {
        let words = self.words;
        let stride = 2 * words;
        let (dst_row, src_row) = if dst < src {
            let (head, tail) = self.rows.split_at_mut(src * stride);
            (&mut head[dst * stride..(dst + 1) * stride], &tail[..stride])
        } else {
            let (head, tail) = self.rows.split_at_mut(dst * stride);
            (&mut tail[..stride], &head[src * stride..(src + 1) * stride])
        };
        let (dst_x, dst_z) = dst_row.split_at_mut(words);
        let sum = rowmul_words_cached(
            dst_x,
            dst_z,
            &src_row[..words],
            &src_row[words..],
            extra + 2 * u64::from(self.signs[dst]) + 2 * u64::from(self.signs[src]),
        );
        debug_assert_eq!(sum & 1, 0, "a gate maps a Hermitian row to a Hermitian row");
        self.signs[dst] = sum & 2 != 0;
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        let stride = 2 * self.words;
        let (lo, hi) = (a.min(b), a.max(b));
        let (head, tail) = self.rows.split_at_mut(hi * stride);
        head[lo * stride..(lo + 1) * stride].swap_with_slice(&mut tail[..stride]);
        self.signs.swap(a, b);
    }

    fn h(&mut self, q: usize) {
        self.swap_rows(2 * q, 2 * q + 1);
    }

    fn x(&mut self, q: usize) {
        self.signs[2 * q + 1] ^= true;
    }

    fn cx(&mut self, control: usize, target: usize) {
        self.mul(2 * control, 2 * target, 0);
        self.mul(2 * target + 1, 2 * control + 1, 0);
    }

    fn basis_to_z(&mut self, basis: QecBasis, q: usize) {
        match basis {
            QecBasis::X => self.h(q),
            QecBasis::Y => {
                self.mul(2 * q, 2 * q + 1, 3);
                self.h(q);
            }
            QecBasis::Z => {}
        }
    }

    fn z_to_basis(&mut self, basis: QecBasis, q: usize) {
        match basis {
            QecBasis::X => self.h(q),
            QecBasis::Y => {
                self.h(q);
                self.mul(2 * q, 2 * q + 1, 1);
            }
            QecBasis::Z => {}
        }
    }

    fn reset(&mut self, basis: QecBasis, q: usize) {
        self.basis_to_z(basis, q);
        self.reset_z(q);
        self.z_to_basis(basis, q);
    }

    fn reset_z(&mut self, q: usize) {
        if self.measure_z(q) {
            self.x(q);
        }
    }

    /// Measure Z on `q`, collapsing a random outcome onto 0.
    fn measure_z(&mut self, q: usize) -> bool {
        let words = self.words;
        let row = (2 * q + 1) * 2 * words;
        if self.rows[row..row + words].iter().all(|&word| word == 0) {
            return self.signs[2 * q + 1];
        }
        self.collapse(q);
        false
    }

    /// Collapse a random Z measurement of `q` onto 0 by changing `C` to `C V`, with `V`
    /// on the input register and `V |0...0>` the projection of `|0...0>` onto the
    /// measured row's +1 eigenspace. The measured row has x bits on the `pivots`: CNOTs
    /// from the first pivot onto the rest fix `|0...0>` and clear all x bits but the
    /// first, an H (after an S when that column reads Y) turns it into Z, and an X there
    /// sets the outcome. Each step conjugates every row on two input columns.
    fn collapse(&mut self, q: usize) {
        let words = self.words;
        let stride = 2 * words;
        let measured = 2 * q + 1;
        let mut pivots = std::mem::take(&mut self.pivots);
        pivots.clear();
        let base = measured * stride;
        for (w, &word) in self.rows[base..base + words].iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                pivots.push(w * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        let (&pivot, rest) = pivots.split_first().expect("a random row has an x bit");

        let row = &mut self.rows[base..base + stride];
        let mut sign = self.signs[measured];
        conjugate_cnots(row, &mut sign, words, pivot, rest);
        let reads_y = bit(row, 64 * words + pivot);
        conjugate_to_z(row, &mut sign, words, pivot, reads_y);
        let flip = sign;
        self.signs[measured] = false;

        for (index, row) in self.rows.chunks_exact_mut(stride).enumerate() {
            if index == measured
                || !bit(row, pivot)
                    && !bit(row, 64 * words + pivot)
                    && rest.iter().all(|&column| !bit(row, 64 * words + column))
            {
                continue;
            }
            let mut sign = self.signs[index];
            conjugate_cnots(row, &mut sign, words, pivot, rest);
            conjugate_to_z(row, &mut sign, words, pivot, reads_y);
            if flip && bit(row, 64 * words + pivot) {
                sign ^= true;
            }
            self.signs[index] = sign;
        }
        self.pivots = pivots;
    }
}

#[inline(always)]
fn bit(row: &[u64], index: usize) -> bool {
    (row[index / 64] >> (index % 64)) & 1 != 0
}

#[inline(always)]
fn flip_bit(row: &mut [u64], index: usize) {
    row[index / 64] ^= 1u64 << (index % 64);
}

/// Conjugate one row by `CX(pivot, column)` on the input register for each of `columns`.
#[inline(always)]
fn conjugate_cnots(
    row: &mut [u64],
    sign: &mut bool,
    words: usize,
    pivot: usize,
    columns: &[usize],
) {
    let x_pivot = bit(row, pivot);
    for &column in columns {
        let z_column = bit(row, 64 * words + column);
        if x_pivot && z_column && bit(row, column) == bit(row, 64 * words + pivot) {
            *sign ^= true;
        }
        if x_pivot {
            flip_bit(row, column);
        }
        if z_column {
            flip_bit(row, 64 * words + pivot);
        }
    }
}

/// Conjugate one row on the input column `pivot` by H, after S when the measured row
/// reads Y there, so the measured row's letter turns into Z.
#[inline(always)]
fn conjugate_to_z(row: &mut [u64], sign: &mut bool, words: usize, pivot: usize, reads_y: bool) {
    let x = bit(row, pivot);
    let mut z = bit(row, 64 * words + pivot);
    if reads_y && x {
        *sign ^= !z;
        z = !z;
        flip_bit(row, 64 * words + pivot);
    }
    if x && z {
        *sign ^= true;
    }
    if x != z {
        flip_bit(row, pivot);
        flip_bit(row, 64 * words + pivot);
    }
}

#[cfg(test)]
#[path = "reference_tests.rs"]
mod tests;
