#[cfg(feature = "bench-internal")]
use super::noise::QecCompiledNoiseSampler;
use super::noise::compile_qec_noisy_sampler;
use super::{
    QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram, QecSampleResult,
    append_basis_to_z_rotation, append_mpp_parity_rotations, append_z_to_basis_rotation,
    ensure_lowered_record_count, qec_non_clifford_error,
};
use crate::backend::{Backend, statevector::StatevectorBackend};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
#[cfg(feature = "bench-internal")]
use crate::sim::compiled::CompiledDetectorSampler;
use crate::sim::compiled::{PackedShots, ShotLayout, compile_detector_sampler};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Run a native QEC program through the scalable compiled Clifford path.
///
/// Lowers supported QEC operations into the packed compiled sampler rather
/// than dense state-vector simulation, so sampling cost grows with the number
/// of measurement records, not with `2^n`. Supports Clifford gates, basis
/// resets and measurements, `MPP`, detectors, observables, and postselection.
/// Active Pauli-noise annotations (`X_ERROR`, `Z_ERROR`, `DEPOLARIZE1`,
/// `DEPOLARIZE2`) are compiled into packed sensitivity rows that are XORed
/// into the noiseless measurement records.
///
/// V1 limitations:
/// - Non-Clifford gates and `EXP_VAL` are rejected.
/// - A measured qubit must be `Reset` before any later gate reuses it; the
///   compiled lowering defers measurements to the terminal records of an
///   internal circuit.
/// - [`QecOptions::chunk_size`] bounds the per-batch shot count. Setting
///   `chunk_size` together with `keep_measurements: false` keeps peak memory
///   at one chunk worth of measurement records.
pub fn run_qec_program(program: &QecProgram) -> Result<QecSampleResult> {
    let has_noise = validate_qec_compiled_program(program)?;
    let chunk_size = qec_runner_chunk_size(program.options())?;
    if program.num_measurements() == 0 {
        let measurements = PackedShots::from_shot_major(
            Vec::new(),
            program.options().shots,
            program.num_measurements(),
        );
        return qec_result_from_measurements(program, measurements);
    }

    if has_noise {
        let mut sampler = compile_qec_noisy_sampler(program)?;
        if chunk_size >= program.options().shots {
            let measurements = sampler.sample_measurements_packed(program.options().shots)?;
            return qec_result_from_measurements(program, measurements);
        }
        return qec_result_from_measurement_chunks(program, |chunk| {
            sampler.sample_measurements_packed(chunk)
        });
    }

    let circuit = lower_qec_program_to_clifford_circuit(program)?;
    let mut sampler = compile_detector_sampler(
        &circuit,
        program.detector_rows()?,
        program.observable_rows()?,
        program.options().seed,
    )?;
    if chunk_size >= program.options().shots {
        let measurements = sampler.sample_measurements_packed(program.options().shots)?;
        return qec_result_from_measurements(program, measurements);
    }
    qec_result_from_measurement_chunks(program, |chunk| sampler.sample_measurements_packed(chunk))
}

/// Internal staged QEC sampler used by benchmark harnesses.
///
/// The ordinary public execution API is [`run_qec_program`]. This type exposes
/// the same packed Clifford path in smaller pieces so benchmarks can measure
/// compile, noiseless sampling, noise application, detector projection,
/// postselection, logical counting, and total execution separately. Gated
/// behind the `bench-internal` feature; not part of the stable API.
#[cfg(feature = "bench-internal")]
pub struct QecProfiledSampler {
    sampler: QecProfiledMeasurementSampler,
    num_measurements: usize,
    detector_rows: Vec<Vec<usize>>,
    observable_rows: Vec<Vec<usize>>,
    postselection_rows: Vec<(Vec<usize>, bool)>,
    keep_measurements: bool,
}

#[cfg(feature = "bench-internal")]
enum QecProfiledMeasurementSampler {
    Empty,
    Clean(CompiledDetectorSampler),
    Noisy(QecCompiledNoiseSampler),
}

/// Internal postselection and logical count summary.
///
/// Used internally by the QEC runner and exposed publicly only under the
/// `bench-internal` feature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QecProfiledCounts {
    /// Shots accepted after postselection.
    pub accepted_shots: usize,
    /// Shots discarded by postselection.
    pub discarded_shots: usize,
    /// Per-observable logical error counts among accepted shots.
    pub logical_errors: Vec<u64>,
}

/// Compile a native QEC program into a staged sampler for benchmarks.
///
/// Prefer [`run_qec_program`] for application code. Gated behind the
/// `bench-internal` feature; not part of the stable API.
#[cfg(feature = "bench-internal")]
pub fn compile_qec_profiled_sampler(program: &QecProgram) -> Result<QecProfiledSampler> {
    let has_noise = validate_qec_compiled_program(program)?;
    qec_runner_chunk_size(program.options())?;

    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let postselection_rows = program.postselection_rows()?;
    let num_measurements = program.num_measurements();
    let sampler = if num_measurements == 0 {
        QecProfiledMeasurementSampler::Empty
    } else if has_noise {
        QecProfiledMeasurementSampler::Noisy(compile_qec_noisy_sampler(program)?)
    } else {
        let circuit = lower_qec_program_to_clifford_circuit(program)?;
        QecProfiledMeasurementSampler::Clean(compile_detector_sampler(
            &circuit,
            detector_rows.clone(),
            observable_rows.clone(),
            program.options().seed,
        )?)
    };

    Ok(QecProfiledSampler {
        sampler,
        num_measurements,
        detector_rows,
        observable_rows,
        postselection_rows,
        keep_measurements: program.options().keep_measurements,
    })
}

#[cfg(feature = "bench-internal")]
impl QecProfiledSampler {
    /// Number of measurement records produced per shot.
    pub fn num_measurements(&self) -> usize {
        self.num_measurements
    }

    /// Number of detector records produced per shot.
    pub fn num_detectors(&self) -> usize {
        self.detector_rows.len()
    }

    /// Number of observable records produced per shot.
    pub fn num_observables(&self) -> usize {
        self.observable_rows.len()
    }

    /// Number of postselection predicates.
    pub fn num_postselections(&self) -> usize {
        self.postselection_rows.len()
    }

    /// Whether active Pauli-noise rows were compiled.
    pub fn has_noise(&self) -> bool {
        matches!(self.sampler, QecProfiledMeasurementSampler::Noisy(_))
    }

    /// Sample noiseless measurement records.
    pub fn sample_noiseless_measurements_packed(
        &mut self,
        num_shots: usize,
    ) -> Result<PackedShots> {
        match &mut self.sampler {
            QecProfiledMeasurementSampler::Empty => Ok(PackedShots::from_shot_major(
                Vec::new(),
                num_shots,
                self.num_measurements,
            )),
            QecProfiledMeasurementSampler::Clean(sampler) => {
                sampler.sample_measurements_packed(num_shots)
            }
            QecProfiledMeasurementSampler::Noisy(sampler) => {
                sampler.sample_noiseless_measurements_packed(num_shots)
            }
        }
    }

    /// Apply compiled Pauli-noise rows to measurement records.
    pub fn apply_noise_to_measurements(
        &mut self,
        measurements: PackedShots,
    ) -> Result<PackedShots> {
        match &mut self.sampler {
            QecProfiledMeasurementSampler::Noisy(sampler) => {
                sampler.apply_noise_to_measurements(measurements)
            }
            QecProfiledMeasurementSampler::Empty | QecProfiledMeasurementSampler::Clean(_) => {
                Ok(measurements)
            }
        }
    }

    /// Sample measurement records with compiled noise applied.
    pub fn sample_measurements_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        let measurements = self.sample_noiseless_measurements_packed(num_shots)?;
        self.apply_noise_to_measurements(measurements)
    }

    /// Project detector parity rows from measurement records.
    pub fn detector_records(&self, measurements: &PackedShots) -> Result<PackedShots> {
        validate_qec_measurement_shape(
            "QEC detector projection input",
            measurements,
            measurements.num_shots(),
            self.num_measurements,
        )?;
        parity_rows_or_empty(measurements, &self.detector_rows)
    }

    /// Project observable parity rows from measurement records.
    pub fn observable_records(&self, measurements: &PackedShots) -> Result<PackedShots> {
        validate_qec_measurement_shape(
            "QEC observable projection input",
            measurements,
            measurements.num_shots(),
            self.num_measurements,
        )?;
        parity_rows_or_empty(measurements, &self.observable_rows)
    }

    /// Count postselection survivors and logical errors from packed records.
    pub fn postselection_and_logical_counts(
        &self,
        measurements: &PackedShots,
        observables: &PackedShots,
    ) -> Result<QecProfiledCounts> {
        validate_qec_measurement_shape(
            "QEC postselection input",
            measurements,
            observables.num_shots(),
            self.num_measurements,
        )?;
        validate_qec_measurement_shape(
            "QEC logical count observable input",
            observables,
            measurements.num_shots(),
            self.observable_rows.len(),
        )?;
        qec_count_postselection_and_logical(
            observables.num_shots(),
            &self.postselection_rows,
            measurements,
            observables,
        )
    }

    /// Build the regular QEC result from staged measurement records.
    pub fn result_from_measurements(&self, measurements: PackedShots) -> Result<QecSampleResult> {
        qec_result_from_measurement_rows(
            measurements.num_shots(),
            self.num_measurements,
            self.keep_measurements,
            &self.detector_rows,
            &self.observable_rows,
            &self.postselection_rows,
            measurements,
        )
    }
}

/// Run a native QEC program through the correctness-first reference path.
///
/// Executes one state-vector simulation per shot. Supports any gate the
/// statevector backend handles (including non-Clifford), `MPP`, all four
/// Pauli-noise channels, and postselection. Use this as a semantic oracle
/// for small programs or to cross-check the compiled runner; cost is
/// `O(shots * 2^n)`, so it is not the production performance path.
///
/// `EXP_VAL` is rejected pending estimator semantics. [`QecOptions::chunk_size`]
/// is validated for shape but not used to bound execution batches.
pub fn run_qec_program_reference(program: &QecProgram) -> Result<QecSampleResult> {
    qec_runner_chunk_size(program.options())?;
    let shots = program.options().shots;
    let num_measurements = program.num_measurements();
    let has_mpp = program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::MeasurePauliProduct { .. }));
    let scratch_qubit = program.num_qubits();
    let backend_qubits = program.num_qubits() + usize::from(has_mpp);
    let mut backend = StatevectorBackend::new(program.options().seed);
    let mut noise_rng = ChaCha8Rng::seed_from_u64(program.options().seed ^ 0xD1B5_4A32_D192_ED03);
    let m_words = num_measurements.div_ceil(64);
    let mut measurement_data = vec![0u64; shots.saturating_mul(m_words)];

    for shot in 0..shots {
        backend.init(backend_qubits, num_measurements)?;
        let mut next_record = 0;

        for op in program.ops() {
            match op {
                QecOp::Gate { gate, targets } => {
                    apply_reference_gate(&mut backend, gate.clone(), targets)?;
                }
                QecOp::Measure { basis, qubit } => {
                    let bit = measure_reference_basis(&mut backend, *basis, *qubit, next_record)?;
                    set_shot_record(&mut measurement_data, m_words, shot, next_record, bit);
                    next_record += 1;
                }
                QecOp::MeasurePauliProduct { terms } => {
                    if !has_mpp {
                        return Err(PrismError::InvalidParameter {
                            message: "internal QEC MPP scratch qubit was not allocated".to_string(),
                        });
                    }
                    let bit =
                        measure_reference_mpp(&mut backend, terms, scratch_qubit, next_record)?;
                    set_shot_record(&mut measurement_data, m_words, shot, next_record, bit);
                    next_record += 1;
                }
                QecOp::Reset { basis, qubit } => {
                    reset_reference_basis(&mut backend, *basis, *qubit)?;
                }
                QecOp::Noise { channel, targets } => {
                    apply_reference_noise(&mut backend, &mut noise_rng, *channel, targets)?;
                }
                QecOp::ExpectationValue { .. } => {
                    return Err(PrismError::IncompatibleBackend {
                        backend: "QEC reference runner".to_string(),
                        reason: "QEC reference runner does not evaluate EXP_VAL yet".to_string(),
                    });
                }
                QecOp::Detector { .. }
                | QecOp::ObservableInclude { .. }
                | QecOp::Postselect { .. }
                | QecOp::Tick => {}
            }
        }

        if next_record != num_measurements {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "QEC reference runner produced {next_record} records, expected {num_measurements}"
                ),
            });
        }
    }

    let measurements = PackedShots::from_shot_major(measurement_data, shots, num_measurements);
    qec_result_from_measurements(program, measurements)
}

fn qec_result_from_measurements(
    program: &QecProgram,
    all_measurements: PackedShots,
) -> Result<QecSampleResult> {
    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let postselection_rows = program.postselection_rows()?;
    qec_result_from_measurement_rows(
        program.options().shots,
        program.num_measurements(),
        program.options().keep_measurements,
        &detector_rows,
        &observable_rows,
        &postselection_rows,
        all_measurements,
    )
}

fn qec_result_from_measurement_rows(
    shots: usize,
    num_measurements: usize,
    keep_measurements: bool,
    detector_rows: &[Vec<usize>],
    observable_rows: &[Vec<usize>],
    postselection_rows: &[(Vec<usize>, bool)],
    all_measurements: PackedShots,
) -> Result<QecSampleResult> {
    validate_qec_measurement_shape(
        "QEC measurement sample",
        &all_measurements,
        shots,
        num_measurements,
    )?;
    let detectors = parity_rows_or_empty(&all_measurements, detector_rows)?;
    let observables = parity_rows_or_empty(&all_measurements, observable_rows)?;
    let counts = qec_count_postselection_and_logical(
        shots,
        postselection_rows,
        &all_measurements,
        &observables,
    )?;
    let measurements = if keep_measurements {
        all_measurements
    } else {
        PackedShots::from_meas_major(Vec::new(), 0, num_measurements)
    };

    QecSampleResult::new_with_total_shots(
        shots,
        measurements,
        detectors,
        observables,
        counts.accepted_shots,
        counts.discarded_shots,
        counts.logical_errors,
    )
}

fn validate_qec_measurement_shape(
    label: &str,
    measurements: &PackedShots,
    expected_shots: usize,
    expected_measurements: usize,
) -> Result<()> {
    if measurements.num_shots() == expected_shots
        && measurements.num_measurements() == expected_measurements
    {
        return Ok(());
    }
    Err(PrismError::InvalidParameter {
        message: format!(
            "{label} shape must be {expected_shots} shots by {expected_measurements} records, got {} by {}",
            measurements.num_shots(),
            measurements.num_measurements()
        ),
    })
}

fn qec_count_postselection_and_logical(
    shots: usize,
    postselection_rows: &[(Vec<usize>, bool)],
    all_measurements: &PackedShots,
    observables: &PackedShots,
) -> Result<QecProfiledCounts> {
    if observables.num_shots() != shots {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC observable shot count {} does not match {shots}",
                observables.num_shots()
            ),
        });
    }

    let mut accepted_shots = shots;
    let mut logical_errors = vec![0u64; observables.num_measurements()];
    if postselection_rows.is_empty() {
        add_qec_logical_error_counts(observables, &mut logical_errors);
    } else {
        accepted_shots = 0;
        let postselection_only: Vec<Vec<usize>> = postselection_rows
            .iter()
            .map(|(row, _)| row.clone())
            .collect();
        let postselection = all_measurements.parity_rows(&postselection_only)?;

        for shot in 0..shots {
            let accepted = postselection_rows
                .iter()
                .enumerate()
                .all(|(row_idx, (_, expected))| postselection.get_bit(shot, row_idx) == *expected);
            if accepted {
                accepted_shots += 1;
                for (observable, count) in logical_errors.iter_mut().enumerate() {
                    if observables.get_bit(shot, observable) {
                        *count += 1;
                    }
                }
            }
        }
    }

    Ok(QecProfiledCounts {
        accepted_shots,
        discarded_shots: shots - accepted_shots,
        logical_errors,
    })
}

fn qec_result_from_measurement_chunks<F>(
    program: &QecProgram,
    mut sample_chunk: F,
) -> Result<QecSampleResult>
where
    F: FnMut(usize) -> Result<PackedShots>,
{
    let options = program.options();
    let shots = options.shots;
    let num_measurements = program.num_measurements();
    let chunk_size = qec_runner_chunk_size(options)?;
    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let postselection_rows = program.postselection_rows()?;
    let postselection_only: Vec<Vec<usize>> = postselection_rows
        .iter()
        .map(|(row, _)| row.clone())
        .collect();

    let mut measurement_data = if options.keep_measurements {
        Vec::with_capacity(shots.saturating_mul(num_measurements.div_ceil(64)))
    } else {
        Vec::new()
    };
    let mut detector_data =
        Vec::with_capacity(shots.saturating_mul(detector_rows.len().div_ceil(64)));
    let mut observable_data =
        Vec::with_capacity(shots.saturating_mul(observable_rows.len().div_ceil(64)));
    let mut accepted_shots = 0usize;
    let mut logical_errors = vec![0u64; observable_rows.len()];
    let mut sampled_shots = 0usize;

    while sampled_shots < shots {
        let this_chunk = (shots - sampled_shots).min(chunk_size);
        let measurements = sample_chunk(this_chunk)?;
        if measurements.num_shots() != this_chunk
            || measurements.num_measurements() != num_measurements
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "QEC measurement chunk shape must be {this_chunk} shots by {num_measurements} records, got {} by {}",
                    measurements.num_shots(),
                    measurements.num_measurements()
                ),
            });
        }

        let detectors = parity_rows_or_empty(&measurements, &detector_rows)?;
        let observables = parity_rows_or_empty(&measurements, &observable_rows)?;

        if postselection_rows.is_empty() {
            accepted_shots += this_chunk;
            add_qec_logical_error_counts(&observables, &mut logical_errors);
        } else {
            let postselection = measurements.parity_rows(&postselection_only)?;
            for shot in 0..this_chunk {
                let accepted =
                    postselection_rows
                        .iter()
                        .enumerate()
                        .all(|(row_idx, (_, expected))| {
                            postselection.get_bit(shot, row_idx) == *expected
                        });
                if accepted {
                    accepted_shots += 1;
                    for (observable, count) in logical_errors.iter_mut().enumerate() {
                        if observables.get_bit(shot, observable) {
                            *count += 1;
                        }
                    }
                }
            }
        }

        if options.keep_measurements {
            measurement_data.extend(measurements.into_shot_major_data());
        }
        detector_data.extend(detectors.into_shot_major_data());
        observable_data.extend(observables.into_shot_major_data());
        sampled_shots += this_chunk;
    }

    let measurements = if options.keep_measurements {
        PackedShots::from_shot_major(measurement_data, shots, num_measurements)
    } else {
        PackedShots::from_meas_major(Vec::new(), 0, num_measurements)
    };
    let detectors = PackedShots::from_shot_major(detector_data, shots, detector_rows.len());
    let observables = PackedShots::from_shot_major(observable_data, shots, observable_rows.len());
    let discarded_shots = shots - accepted_shots;

    QecSampleResult::new_with_total_shots(
        shots,
        measurements,
        detectors,
        observables,
        accepted_shots,
        discarded_shots,
        logical_errors,
    )
}

fn qec_runner_chunk_size(options: QecOptions) -> Result<usize> {
    match options.chunk_size {
        Some(0) => Err(PrismError::InvalidParameter {
            message: "QEC chunk_size must be at least 1".to_string(),
        }),
        Some(chunk_size) => Ok(chunk_size),
        None => Ok(options.shots.max(1)),
    }
}

fn parity_rows_or_empty(measurements: &PackedShots, rows: &[Vec<usize>]) -> Result<PackedShots> {
    if rows.is_empty() {
        return Ok(PackedShots::from_shot_major(
            Vec::new(),
            measurements.num_shots(),
            0,
        ));
    }
    if measurements.num_shots() == 0 {
        return Ok(PackedShots::from_shot_major(Vec::new(), 0, rows.len()));
    }
    if matches!(measurements.layout(), ShotLayout::ShotMajor) {
        if let Some(words) = repeated_shot_words(measurements) {
            return Ok(parity_rows_for_repeated_shot(
                words,
                measurements.num_shots(),
                rows,
            ));
        }
        return Ok(parity_rows_for_shot_major(measurements, rows));
    }
    measurements.parity_rows(rows)
}

fn repeated_shot_words(measurements: &PackedShots) -> Option<&[u64]> {
    debug_assert!(matches!(measurements.layout(), ShotLayout::ShotMajor));
    let m_words = measurements.m_words();
    let data = measurements.raw_data();
    let first = &data[..m_words];
    for shot in 1..measurements.num_shots() {
        let base = shot * m_words;
        if &data[base..base + m_words] != first {
            return None;
        }
    }
    Some(first)
}

fn parity_rows_for_repeated_shot(
    shot_words: &[u64],
    num_shots: usize,
    rows: &[Vec<usize>],
) -> PackedShots {
    let out_words = rows.len().div_ceil(64);
    let mut pattern = vec![0u64; out_words];
    let prepared_rows = prepare_qec_parity_rows(rows);
    for (out_idx, row) in prepared_rows.iter().enumerate() {
        pattern[out_idx / 64] |= row.parity(shot_words) << (out_idx % 64);
    }

    let mut data = vec![0u64; num_shots * out_words];
    if pattern.iter().any(|&word| word != 0) {
        for shot_words in data.chunks_exact_mut(out_words) {
            shot_words.copy_from_slice(&pattern);
        }
    }
    PackedShots::from_shot_major(data, num_shots, rows.len())
}

fn parity_rows_for_shot_major(measurements: &PackedShots, rows: &[Vec<usize>]) -> PackedShots {
    debug_assert!(matches!(measurements.layout(), ShotLayout::ShotMajor));
    let out_words = rows.len().div_ceil(64);
    let m_words = measurements.m_words();
    let mut data = vec![0u64; measurements.num_shots() * out_words];
    let measurement_data = measurements.raw_data();
    let prepared_rows = prepare_qec_parity_rows(rows);

    for shot in 0..measurements.num_shots() {
        let src_base = shot * m_words;
        let dst_base = shot * out_words;
        let shot_words = &measurement_data[src_base..src_base + m_words];
        let dst = &mut data[dst_base..dst_base + out_words];
        for (out_idx, row) in prepared_rows.iter().enumerate() {
            dst[out_idx / 64] |= row.parity(shot_words) << (out_idx % 64);
        }
    }

    PackedShots::from_shot_major(data, measurements.num_shots(), rows.len())
}

enum QecPreparedParityRow {
    Empty,
    Single(QecPreparedMeasurement),
    Pair(QecPreparedMeasurement, QecPreparedMeasurement),
    Triple(
        QecPreparedMeasurement,
        QecPreparedMeasurement,
        QecPreparedMeasurement,
    ),
    Many(Vec<QecPreparedMeasurement>),
}

#[derive(Clone, Copy)]
struct QecPreparedMeasurement {
    word: usize,
    bit: u32,
}

fn prepare_qec_parity_rows(rows: &[Vec<usize>]) -> Vec<QecPreparedParityRow> {
    rows.iter()
        .map(|row| {
            let prepared: Vec<_> = row
                .iter()
                .copied()
                .map(QecPreparedMeasurement::new)
                .collect();
            match prepared.as_slice() {
                [] => QecPreparedParityRow::Empty,
                [a] => QecPreparedParityRow::Single(*a),
                [a, b] => QecPreparedParityRow::Pair(*a, *b),
                [a, b, c] => QecPreparedParityRow::Triple(*a, *b, *c),
                _ => QecPreparedParityRow::Many(prepared),
            }
        })
        .collect()
}

impl QecPreparedMeasurement {
    #[inline(always)]
    fn new(measurement: usize) -> Self {
        Self {
            word: measurement / 64,
            bit: (measurement % 64) as u32,
        }
    }

    #[inline(always)]
    fn read(self, shot_words: &[u64]) -> u64 {
        (shot_words[self.word] >> self.bit) & 1
    }
}

impl QecPreparedParityRow {
    #[inline(always)]
    fn parity(&self, shot_words: &[u64]) -> u64 {
        match self {
            Self::Empty => 0,
            Self::Single(a) => a.read(shot_words),
            Self::Pair(a, b) => a.read(shot_words) ^ b.read(shot_words),
            Self::Triple(a, b, c) => a.read(shot_words) ^ b.read(shot_words) ^ c.read(shot_words),
            Self::Many(row) => {
                let mut parity = 0u64;
                for measurement in row {
                    parity ^= measurement.read(shot_words);
                }
                parity
            }
        }
    }
}

fn add_qec_logical_error_counts(observables: &PackedShots, counts: &mut [u64]) {
    debug_assert_eq!(observables.num_measurements(), counts.len());
    if counts.is_empty() || observables.num_shots() == 0 {
        return;
    }

    match observables.layout() {
        ShotLayout::MeasMajor => {
            let full_words = observables.num_shots() / 64;
            let tail_bits = observables.num_shots() % 64;
            let tail_mask = if tail_bits == 0 {
                u64::MAX
            } else {
                (1u64 << tail_bits) - 1
            };

            for (observable, count) in counts.iter_mut().enumerate() {
                let words = observables.meas_words(observable);
                let mut total = 0u64;
                for &word in &words[..full_words] {
                    total += word.count_ones() as u64;
                }
                if tail_bits != 0 {
                    total += (words[full_words] & tail_mask).count_ones() as u64;
                }
                *count += total;
            }
        }
        ShotLayout::ShotMajor => {
            let n_obs = observables.num_measurements();
            let tail_bits = n_obs % 64;
            let tail_mask = if tail_bits == 0 {
                u64::MAX
            } else {
                (1u64 << tail_bits) - 1
            };

            for shot in 0..observables.num_shots() {
                let words = observables.shot_words(shot);
                for (word_idx, &word) in words.iter().enumerate() {
                    let valid_word = if word_idx + 1 == words.len() {
                        word & tail_mask
                    } else {
                        word
                    };
                    let mut bits = valid_word;
                    while bits != 0 {
                        let bit = bits.trailing_zeros() as usize;
                        counts[word_idx * 64 + bit] += 1;
                        bits &= bits - 1;
                    }
                }
            }
        }
    }
}

fn validate_qec_compiled_program(program: &QecProgram) -> Result<bool> {
    let mut has_noise = false;
    for op in program.ops() {
        match op {
            QecOp::Gate { gate, .. } if !gate.is_clifford() => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC compiled runner".to_string(),
                    reason: format!(
                        "compiled QEC runner requires Clifford gates, got `{}`",
                        gate.name()
                    ),
                });
            }
            QecOp::ExpectationValue { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC compiled runner".to_string(),
                    reason: "compiled QEC runner does not evaluate EXP_VAL yet".to_string(),
                });
            }
            QecOp::Noise { channel, .. } => {
                has_noise |= channel.probability() > 0.0;
            }
            _ => {}
        }
    }
    Ok(has_noise)
}

fn lower_qec_program_to_clifford_circuit(program: &QecProgram) -> Result<Circuit> {
    let has_mpp = program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::MeasurePauliProduct { .. }));
    let scratch_qubit = program.num_qubits();
    let mut circuit = Circuit::new(
        program.num_qubits() + usize::from(has_mpp),
        program.num_measurements(),
    );
    let mut next_record = 0usize;
    let mut scratch_needs_reset = false;

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => {
                if !gate.is_clifford() {
                    return Err(qec_non_clifford_error(gate));
                }
                circuit.add_gate(gate.clone(), targets);
            }
            QecOp::Measure { basis, qubit } => {
                append_basis_to_z_rotation(&mut circuit, *basis, *qubit);
                circuit.add_measure(*qubit, next_record);
                next_record += 1;
            }
            QecOp::MeasurePauliProduct { terms } => {
                if !has_mpp {
                    return Err(PrismError::InvalidParameter {
                        message: "internal QEC MPP scratch qubit was not allocated".to_string(),
                    });
                }
                if scratch_needs_reset {
                    circuit.add_reset(scratch_qubit);
                }
                append_mpp_parity_rotations(&mut circuit, terms, scratch_qubit);
                circuit.add_measure(scratch_qubit, next_record);
                next_record += 1;
                scratch_needs_reset = true;
            }
            QecOp::Reset { basis, qubit } => {
                circuit.add_reset(*qubit);
                append_z_to_basis_rotation(&mut circuit, *basis, *qubit);
            }
            QecOp::Noise { channel, .. } => {
                let probability = channel.probability();
                debug_assert!(
                    probability == 0.0,
                    "active QEC noise should use deferred lowering"
                );
                if probability > 0.0 {
                    return Err(PrismError::IncompatibleBackend {
                        backend: "QEC compiled runner".to_string(),
                        reason: "compiled QEC runner does not support active noise annotations in the clean path".to_string(),
                    });
                }
            }
            QecOp::ExpectationValue { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC compiled runner".to_string(),
                    reason: "compiled QEC runner does not evaluate EXP_VAL yet".to_string(),
                });
            }
            QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
        }
    }

    ensure_lowered_record_count(program, next_record, "compiled")?;
    Ok(circuit)
}

fn apply_reference_gate(
    backend: &mut StatevectorBackend,
    gate: Gate,
    targets: &[usize],
) -> Result<()> {
    backend.apply(&Instruction::Gate {
        gate,
        targets: qec_targets(targets),
    })
}

fn reset_reference_basis(
    backend: &mut StatevectorBackend,
    basis: QecBasis,
    qubit: usize,
) -> Result<()> {
    backend.reset(qubit)?;
    rotate_reference_z_to_basis(backend, basis, qubit)
}

fn measure_reference_basis(
    backend: &mut StatevectorBackend,
    basis: QecBasis,
    qubit: usize,
    record: usize,
) -> Result<bool> {
    rotate_reference_basis_to_z(backend, basis, qubit)?;
    backend.apply(&Instruction::Measure {
        qubit,
        classical_bit: record,
    })?;
    let outcome = backend.classical_results()[record];
    rotate_reference_z_to_basis(backend, basis, qubit)?;
    Ok(outcome)
}

fn measure_reference_mpp(
    backend: &mut StatevectorBackend,
    terms: &[QecPauli],
    scratch_qubit: usize,
    record: usize,
) -> Result<bool> {
    backend.reset(scratch_qubit)?;
    for term in terms {
        rotate_reference_basis_to_z(backend, term.basis, term.qubit)?;
        apply_reference_gate(backend, Gate::Cx, &[term.qubit, scratch_qubit])?;
    }
    for term in terms.iter().rev() {
        rotate_reference_z_to_basis(backend, term.basis, term.qubit)?;
    }
    backend.apply(&Instruction::Measure {
        qubit: scratch_qubit,
        classical_bit: record,
    })?;
    Ok(backend.classical_results()[record])
}

fn rotate_reference_basis_to_z(
    backend: &mut StatevectorBackend,
    basis: QecBasis,
    qubit: usize,
) -> Result<()> {
    match basis {
        QecBasis::X => apply_reference_gate(backend, Gate::H, &[qubit]),
        QecBasis::Y => {
            apply_reference_gate(backend, Gate::Sdg, &[qubit])?;
            apply_reference_gate(backend, Gate::H, &[qubit])
        }
        QecBasis::Z => Ok(()),
    }
}

fn rotate_reference_z_to_basis(
    backend: &mut StatevectorBackend,
    basis: QecBasis,
    qubit: usize,
) -> Result<()> {
    match basis {
        QecBasis::X => apply_reference_gate(backend, Gate::H, &[qubit]),
        QecBasis::Y => {
            apply_reference_gate(backend, Gate::H, &[qubit])?;
            apply_reference_gate(backend, Gate::S, &[qubit])
        }
        QecBasis::Z => Ok(()),
    }
}

fn apply_reference_noise(
    backend: &mut StatevectorBackend,
    rng: &mut ChaCha8Rng,
    channel: QecNoise,
    targets: &[usize],
) -> Result<()> {
    if channel.probability() == 0.0 {
        return Ok(());
    }

    match channel {
        QecNoise::XError(p) => {
            for &target in targets {
                if rng.random::<f64>() < p {
                    apply_reference_gate(backend, Gate::X, &[target])?;
                }
            }
        }
        QecNoise::ZError(p) => {
            for &target in targets {
                if rng.random::<f64>() < p {
                    apply_reference_gate(backend, Gate::Z, &[target])?;
                }
            }
        }
        QecNoise::Depolarize1(p) => {
            for &target in targets {
                if rng.random::<f64>() < p {
                    let gate = match rng.random_range(0..3) {
                        0 => Gate::X,
                        1 => Gate::Y,
                        _ => Gate::Z,
                    };
                    apply_reference_gate(backend, gate, &[target])?;
                }
            }
        }
        QecNoise::Depolarize2(p) => {
            for pair in targets.chunks_exact(2) {
                if rng.random::<f64>() < p {
                    let sample = rng.random_range(1..16);
                    let first = (sample / 4) as usize;
                    let second = (sample % 4) as usize;
                    apply_reference_pauli_index(backend, first, pair[0])?;
                    apply_reference_pauli_index(backend, second, pair[1])?;
                }
            }
        }
    }
    Ok(())
}

fn apply_reference_pauli_index(
    backend: &mut StatevectorBackend,
    pauli: usize,
    target: usize,
) -> Result<()> {
    match pauli {
        0 => Ok(()),
        1 => apply_reference_gate(backend, Gate::X, &[target]),
        2 => apply_reference_gate(backend, Gate::Y, &[target]),
        _ => apply_reference_gate(backend, Gate::Z, &[target]),
    }
}

fn qec_targets(targets: &[usize]) -> SmallVec<[usize; 4]> {
    let mut small = SmallVec::new();
    small.extend_from_slice(targets);
    small
}

fn set_shot_record(data: &mut [u64], m_words: usize, shot: usize, record: usize, value: bool) {
    if value {
        data[shot * m_words + record / 64] |= 1u64 << (record % 64);
    }
}
