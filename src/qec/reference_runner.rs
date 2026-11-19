//! Per-shot statevector reference runner for native QEC programs, the correctness
//! oracle behind [`run_qec_program_reference`].

use super::runner::{qec_result_from_measurements, qec_runner_chunk_size};
use super::{
    QecBasis, QecNoise, QecObservableEstimate, QecOp, QecPauli, QecProgram, QecRecordRef,
    QecSampleResult, qec_lowered_num_qubits,
};
use crate::backend::{Backend, statevector::StatevectorBackend};
use crate::circuit::{Circuit, ClassicalCondition, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::PackedShots;
use crate::sim::unified_pauli::Welford;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Run a native QEC program through the per-shot statevector reference path.
///
/// Supports any gate the statevector backend handles, `MPP`, all four Pauli-noise
/// channels, postselection, and `FEEDFORWARD`. Cost is `O(shots * 2^n)`: use it as a
/// semantic oracle for small programs, not for bulk sampling.
///
/// `EXP_VAL` ops are evaluated exactly on each shot's final state and reported in
/// [`QecSampleResult::expectation_values`] as the mean and unbiased sample variance over
/// accepted shots; with noise the trajectories average to `Tr(rho P)`. The Pauli-mask
/// reduction limits `EXP_VAL` programs to 64 qubits, though statevector memory binds far
/// earlier. [`QecOptions::chunk_size`](super::QecOptions::chunk_size) is validated but
/// does not batch execution.
pub fn run_qec_program_reference(program: &QecProgram) -> Result<QecSampleResult> {
    super::validate_measured_qubit_reuse(program)?;
    qec_runner_chunk_size(program.options())?;
    super::validate_qec_exp_val_placement(program)?;
    let shots = program.options().shots;
    let num_measurements = program.num_measurements();
    let scratch_qubit = program.num_qubits();
    let backend_qubits = qec_lowered_num_qubits(program);
    let mut backend = StatevectorBackend::new(program.options().seed);
    let mut noise_rng = ChaCha8Rng::seed_from_u64(program.options().seed ^ 0xD1B5_4A32_D192_ED03);
    let m_words = num_measurements.div_ceil(64);
    let mut measurement_data = vec![0u64; shots.saturating_mul(m_words)];

    if program.num_expectation_values() > 0 && backend_qubits > usize::BITS as usize {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC reference runner".to_string(),
            reason: format!(
                "`EXP_VAL` mask reduction supports at most {} qubits, program needs {}",
                usize::BITS,
                backend_qubits
            ),
        });
    }
    let exp_val_masks = program
        .expectation_value_ops()
        .into_iter()
        .map(|(terms, coefficient)| {
            let terms = super::qec_terms_to_pauli(terms);
            crate::sim::pauli_masks(&terms, backend_qubits).map(|masks| (masks, coefficient))
        })
        .collect::<Result<Vec<_>>>()?;
    let postselection_rows = program.postselection_rows()?;
    // Record indices are shot-invariant, so the guarded regions are built once
    // rather than per shot inside the loop below.
    let feedforward_regions = build_feedforward_regions(program, backend_qubits)?;
    let mut exp_val_stats = vec![Welford::default(); exp_val_masks.len()];
    let mut exp_val_accepted = 0usize;

    for shot in 0..shots {
        backend.init(backend_qubits, num_measurements)?;
        let mut next_record = 0;
        let mut next_feedforward = 0;

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
                    let bit =
                        measure_reference_mpp(&mut backend, terms, scratch_qubit, next_record)?;
                    set_shot_record(&mut measurement_data, m_words, shot, next_record, bit);
                    next_record += 1;
                }
                QecOp::Reset { basis, qubit } => {
                    reset_reference_basis(&mut backend, *basis, *qubit)?;
                }
                QecOp::Noise { channel, targets } => {
                    apply_reference_noise(&mut backend, &mut noise_rng, channel, targets)?;
                }
                QecOp::CorrelatedError { branches } => {
                    for branch in branches {
                        if branch.probability > 0.0
                            && noise_rng.random::<f64>() < branch.probability
                        {
                            for term in &branch.terms {
                                apply_reference_pauli_index(
                                    &mut backend,
                                    usize::from(term.basis.pauli_index()),
                                    term.qubit,
                                )?;
                            }
                            break;
                        }
                    }
                }
                QecOp::HeraldedErase { probability, qubit } => {
                    let erased = *probability > 0.0 && noise_rng.random::<f64>() < *probability;
                    if erased {
                        let pauli = noise_rng.random_range(0..4);
                        apply_reference_pauli_index(&mut backend, pauli, *qubit)?;
                    }
                    set_shot_record(&mut measurement_data, m_words, shot, next_record, erased);
                    backend.classical_bits[next_record] = erased;
                    next_record += 1;
                }
                QecOp::Pad { value } => {
                    set_shot_record(&mut measurement_data, m_words, shot, next_record, *value);
                    backend.classical_bits[next_record] = *value;
                    next_record += 1;
                }
                QecOp::RecordFlip { probability } => {
                    if *probability > 0.0 && noise_rng.random::<f64>() < *probability {
                        let record = next_record - 1;
                        measurement_data[shot * m_words + record / 64] ^= 1 << (record % 64);
                        backend.classical_bits[record] ^= true;
                    }
                }
                QecOp::Feedforward { .. } => {
                    backend.apply(&feedforward_regions[next_feedforward])?;
                    next_feedforward += 1;
                }
                QecOp::ExpectationValue { .. }
                | QecOp::Detector { .. }
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

        if !exp_val_masks.is_empty() {
            let accepted = postselection_rows.iter().all(|(row, expected)| {
                let parity = row.iter().fold(false, |acc, &record| {
                    let bit =
                        (measurement_data[shot * m_words + record / 64] >> (record % 64)) & 1 == 1;
                    acc ^ bit
                });
                parity == *expected
            });
            if accepted {
                exp_val_accepted += 1;
                let state = backend.state_vector();
                let norm = crate::backend::state_norm_sqr(state);
                for (slot, &((xmask, zmask, num_y), coefficient)) in
                    exp_val_masks.iter().enumerate()
                {
                    let value = coefficient
                        * crate::sim::pauli_expectation_from_masks(
                            state, xmask, zmask, num_y, norm,
                        );
                    exp_val_stats[slot].push(value);
                }
            }
        }
    }

    let measurements = PackedShots::from_shot_major(measurement_data, shots, num_measurements);
    let result = qec_result_from_measurements(program, measurements)?;
    if exp_val_masks.is_empty() {
        return Ok(result);
    }
    let estimates = exp_val_stats
        .iter()
        .map(|stats| QecObservableEstimate {
            mean: stats.mean(),
            variance: stats.variance(),
            num_shots: exp_val_accepted,
        })
        .collect();
    Ok(result.with_expectation_values(estimates))
}

/// Lower every feed-forward op in `program` to its guarded instruction, in op
/// order.
fn build_feedforward_regions(program: &QecProgram, num_qubits: usize) -> Result<Vec<Instruction>> {
    let mut regions = Vec::new();
    let mut next_record = 0usize;
    for op in program.ops() {
        match op {
            _ if op.produces_record() => next_record += 1,
            QecOp::Feedforward {
                records,
                expected,
                body,
            } => regions.push(feedforward_region(
                records,
                *expected,
                body,
                next_record,
                num_qubits,
            )?),
            _ => {}
        }
    }
    Ok(regions)
}

/// Build the guarded instruction a feed-forward op executes as.
///
/// The QEC record space and the classical bit vector are one address space: the
/// reference runner writes record `i` to classical bit `i`, so a resolved
/// record index is the bit the condition reads with no translation. A one-gate
/// body takes the [`Instruction::Conditional`] lowering [`crate::circuit::guarded`]
/// picks for it.
fn feedforward_region(
    records: &[QecRecordRef],
    expected: bool,
    body: &[QecOp],
    next_record: usize,
    num_qubits: usize,
) -> Result<Instruction> {
    let bits = super::resolve_records(records, next_record)?;
    let mut lowered = Circuit::new(num_qubits, 0);
    for op in body {
        match op {
            QecOp::Gate { gate, targets } => lowered.add_gate(gate.clone(), targets),
            QecOp::Reset { basis, qubit } => {
                lowered.add_reset(*qubit);
                super::append_z_to_basis_rotation(&mut lowered, *basis, *qubit);
            }
            other => {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "feed-forward body admits gates and resets only, got `{}`",
                        super::qec_op_name(other)
                    ),
                });
            }
        }
    }
    let condition = ClassicalCondition::Parity {
        bits: bits.into(),
        expected,
    };
    crate::circuit::guarded(condition, lowered.instructions).ok_or_else(|| {
        PrismError::InvalidParameter {
            message: "feed-forward body is empty".to_string(),
        }
    })
}

fn apply_reference_gate(
    backend: &mut StatevectorBackend,
    gate: Gate,
    targets: &[usize],
) -> Result<()> {
    backend.apply(&Instruction::Gate {
        gate,
        targets: SmallVec::from_slice(targets),
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

/// Measure `qubit` in `basis`, leaving it in the Z frame.
///
/// The basis rotation is not undone, matching what
/// [`lower_qec_program_to_clifford_circuit`](super::runner::lower_qec_program_to_clifford_circuit)
/// emits and what the deferred noisy lowering in [`crate::qec::noise`] emits.
/// Restoring here instead would append a gate after every basis measurement, which costs
/// [`Circuit::has_terminal_measurements_only`] and the terminal sampling paths
/// it gates. A program that reuses the qubit resets it first, which is what
/// makes the convention unobservable in well-formed programs.
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
    Ok(backend.classical_results()[record])
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
    channel: &QecNoise,
    targets: &[usize],
) -> Result<()> {
    if channel.is_leakage() {
        return Err(super::leakage::leakage_rejection("QEC reference runner"));
    }
    if channel.probability() == 0.0 {
        return Ok(());
    }

    match *channel {
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
            for pair in targets.as_chunks::<2>().0 {
                if rng.random::<f64>() < p {
                    let sample = rng.random_range(1..16);
                    let first = (sample / 4) as usize;
                    let second = (sample % 4) as usize;
                    apply_reference_pauli_index(backend, first, pair[0])?;
                    apply_reference_pauli_index(backend, second, pair[1])?;
                }
            }
        }
        QecNoise::PauliChannel2(ref rates) => {
            for pair in targets.as_chunks::<2>().0 {
                let mut r = rng.random::<f64>();
                for (branch, &rate) in rates.iter().enumerate() {
                    if r < rate {
                        let sample = branch + 1;
                        apply_reference_pauli_index(backend, sample / 4, pair[0])?;
                        apply_reference_pauli_index(backend, sample % 4, pair[1])?;
                        break;
                    }
                    r -= rate;
                }
            }
        }
        _ => {
            let (px, py, pz) = channel
                .single_rates()
                .expect("every one-qubit channel has branch rates");
            for &target in targets {
                let r = rng.random::<f64>();
                let pauli = if r < px {
                    1
                } else if r < px + py {
                    2
                } else if r < px + py + pz {
                    3
                } else {
                    0
                };
                apply_reference_pauli_index(backend, pauli, target)?;
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

fn set_shot_record(data: &mut [u64], m_words: usize, shot: usize, record: usize, value: bool) {
    if value {
        data[shot * m_words + record / 64] |= 1u64 << (record % 64);
    }
}
