use super::noise::compile_qec_noisy_sampler;
use super::{
    append_basis_to_z_rotation, append_z_to_basis_rotation, QecBasis, QecNoise, QecOp, QecOptions,
    QecPauli, QecProgram, QecSampleResult,
};
use crate::backend::{statevector::StatevectorBackend, Backend};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{compile_detector_sampler, PackedShots, ShotLayout};
use rand::{Rng, SeedableRng};
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
    let shots = program.options().shots;
    let num_measurements = program.num_measurements();
    if all_measurements.num_shots() != shots
        || all_measurements.num_measurements() != num_measurements
    {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC measurement sample shape must be {shots} shots by {num_measurements} records, got {} by {}",
                all_measurements.num_shots(),
                all_measurements.num_measurements()
            ),
        });
    }
    let detector_rows = program.detector_rows()?;
    let observable_rows = program.observable_rows()?;
    let postselection_rows = program.postselection_rows()?;
    let detectors = parity_rows_or_empty(&all_measurements, &detector_rows)?;
    let observables = parity_rows_or_empty(&all_measurements, &observable_rows)?;
    let mut accepted_shots = shots;
    let mut logical_errors = vec![0u64; observables.num_measurements()];

    if postselection_rows.is_empty() {
        add_qec_logical_error_counts(&observables, &mut logical_errors);
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

    let discarded_shots = shots - accepted_shots;
    let measurements = if program.options().keep_measurements {
        all_measurements
    } else {
        PackedShots::from_meas_major(Vec::new(), 0, num_measurements)
    };

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
    for (out_idx, row) in rows.iter().enumerate() {
        pattern[out_idx / 64] |= qec_row_parity_word(shot_words, row) << (out_idx % 64);
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

    for shot in 0..measurements.num_shots() {
        let src_base = shot * m_words;
        let dst_base = shot * out_words;
        let shot_words = &measurement_data[src_base..src_base + m_words];
        let dst = &mut data[dst_base..dst_base + out_words];
        for (out_idx, row) in rows.iter().enumerate() {
            dst[out_idx / 64] |= qec_row_parity_word(shot_words, row) << (out_idx % 64);
        }
    }

    PackedShots::from_shot_major(data, measurements.num_shots(), rows.len())
}

#[inline(always)]
fn qec_row_parity_word(shot_words: &[u64], row: &[usize]) -> u64 {
    match row {
        [] => 0,
        [a] => qec_measurement_word(shot_words, *a),
        [a, b] => qec_measurement_word(shot_words, *a) ^ qec_measurement_word(shot_words, *b),
        [a, b, c] => {
            qec_measurement_word(shot_words, *a)
                ^ qec_measurement_word(shot_words, *b)
                ^ qec_measurement_word(shot_words, *c)
        }
        _ => {
            let mut parity = 0u64;
            for &measurement in row {
                parity ^= qec_measurement_word(shot_words, measurement);
            }
            parity
        }
    }
}

#[inline(always)]
fn qec_measurement_word(shot_words: &[u64], measurement: usize) -> u64 {
    (shot_words[measurement / 64] >> (measurement % 64)) & 1
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
                    return Err(PrismError::IncompatibleBackend {
                        backend: "QEC compiled runner".to_string(),
                        reason: format!(
                            "compiled QEC runner requires Clifford gates, got `{}`",
                            gate.name()
                        ),
                    });
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
                for term in terms {
                    append_basis_to_z_rotation(&mut circuit, term.basis, term.qubit);
                }
                for term in terms {
                    circuit.add_gate(Gate::Cx, &[term.qubit, scratch_qubit]);
                }
                for term in terms.iter().rev() {
                    append_z_to_basis_rotation(&mut circuit, term.basis, term.qubit);
                }
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

    if next_record != program.num_measurements() {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC compiled lowering produced {next_record} records, expected {}",
                program.num_measurements()
            ),
        });
    }
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
