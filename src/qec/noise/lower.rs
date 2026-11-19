//! Lowerings of a QEC program: to the deferred circuit the compiled samplers walk, and
//! to a circuit plus noise model for the density-matrix estimator.

use super::{
    DeferredGate, QecDeferredChannel, QecDeferredFeedforward, QecDeferredNoiseEvent,
    QecDeferredProgram, QecProductTable,
};
use crate::circuit::{Circuit, GateSink, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::qec::leakage::push_deferred_leak_event;
use crate::qec::{
    QecBasis, QecNoise, QecOp, QecPauli, QecProgram, append_basis_to_z_rotation,
    append_mpp_parity_rotations, append_z_to_basis_rotation, ensure_lowered_record_count,
    qec_lowered_num_qubits, qec_non_clifford_error,
};
use crate::sim::noise::{NoiseChannel, NoiseEvent, NoiseModel};
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use smallvec::smallvec;

/// Frame letter (1 X, 2 Y, 3 Z, 0 identity) of a feed-forward body op the record path
/// applies as flips, `None` for any other op.
pub(in crate::qec) fn feedforward_body_letter(op: &QecOp) -> Option<u8> {
    match op {
        QecOp::Gate { gate, .. } => match gate {
            Gate::Id => Some(0),
            Gate::X => Some(1),
            Gate::Y => Some(2),
            Gate::Z => Some(3),
            _ => None,
        },
        _ => None,
    }
}

/// Rejection of a feed-forward body op that is not a Pauli gate.
pub(in crate::qec) fn feedforward_body_rejection(op: &QecOp) -> PrismError {
    let name = match op {
        QecOp::Gate { gate, .. } => gate.name(),
        other => crate::qec::qec_op_name(other),
    };
    PrismError::IncompatibleBackend {
        backend: "QEC compiled runner".to_string(),
        reason: format!(
            "the compiled QEC sampler applies a `FEEDFORWARD` body as record flips, which is \
             exact for Pauli gates (`X`, `Y`, `Z`, `I`) only, got `{name}`; \
             `run_qec_program_reference` executes any body"
        ),
    }
}

/// Lower a measurement-free QEC program into a circuit plus the matching
/// density-matrix noise model.
///
/// Gates map one to one; a basis reset becomes `Reset` followed by the
/// Z-to-basis rotation. Pauli-noise annotations become [`NoiseEvent`]s on the
/// instruction they follow, which the density-matrix backend applies through
/// its exact one-qubit Kraus and two-qubit depolarizing channels. Noise that
/// precedes every instruction anchors on a barrier so each event has a host.
///
/// Measurements are rejected: they carry a per-shot record stream the mixed
/// state does not hold. Callers route those programs to the reference runner.
pub(in crate::qec) fn lower_qec_program_to_density_matrix(
    program: &QecProgram,
) -> Result<(Circuit, NoiseModel)> {
    let mut circuit = Circuit::new(program.num_qubits(), 0);
    let mut after_gate: Vec<Vec<NoiseEvent>> = Vec::new();

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => circuit.add_gate(gate.clone(), targets),
            QecOp::Reset { basis, qubit } => {
                circuit.add_reset(*qubit);
                append_z_to_basis_rotation(&mut circuit, *basis, *qubit);
            }
            QecOp::Noise { channel, targets } if channel.probability() > 0.0 => {
                if circuit.instructions.is_empty() {
                    circuit.add_barrier(&[]);
                }
                let anchor = circuit.instructions.len() - 1;
                after_gate.resize_with(circuit.instructions.len(), Vec::new);
                push_density_matrix_noise_events(&mut after_gate[anchor], channel, targets);
            }
            QecOp::Measure { .. }
            | QecOp::MeasurePauliProduct { .. }
            | QecOp::HeraldedErase { .. }
            | QecOp::Pad { .. }
            | QecOp::RecordFlip { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC density-matrix estimator".to_string(),
                    reason: "the density matrix holds no measurement records; \
                             `run_qec_program` routes measuring programs to the reference runner"
                        .to_string(),
                });
            }
            QecOp::CorrelatedError { .. } if op.is_active_noise() => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC density-matrix estimator".to_string(),
                    reason: "the density-matrix channel set has no correlated Pauli product; \
                             `run_qec_program` routes such programs to the reference runner"
                        .to_string(),
                });
            }
            QecOp::Feedforward { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC density-matrix estimator".to_string(),
                    reason: "`FEEDFORWARD` reads a measurement record, and the density matrix \
                             holds none; call `run_qec_program_reference` for such programs"
                        .to_string(),
                });
            }
            QecOp::Noise { .. }
            | QecOp::CorrelatedError { .. }
            | QecOp::ExpectationValue { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
        }
    }

    after_gate.resize_with(circuit.instructions.len(), Vec::new);
    let noise = NoiseModel {
        after_gate,
        readout: Vec::new(),
    };
    Ok((circuit, noise))
}

fn push_density_matrix_noise_events(
    events: &mut Vec<NoiseEvent>,
    channel: &QecNoise,
    targets: &[usize],
) {
    match *channel {
        QecNoise::Depolarize1(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent {
                channel: NoiseChannel::Depolarizing { p },
                qubits: SmallVec::from_slice(&[q]),
            }));
        }
        QecNoise::Depolarize2(p) => {
            events.extend(targets.as_chunks::<2>().0.iter().map(|pair| NoiseEvent {
                channel: NoiseChannel::TwoQubitDepolarizing { p },
                qubits: SmallVec::from_slice(pair),
            }));
        }
        QecNoise::PauliChannel2(ref rates) => {
            let kraus = pauli_channel_2_kraus(rates);
            events.extend(targets.as_chunks::<2>().0.iter().map(|pair| NoiseEvent {
                channel: NoiseChannel::Kraus2q {
                    kraus: kraus.clone(),
                },
                qubits: SmallVec::from_slice(pair),
            }));
        }
        QecNoise::Leak(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent {
                channel: NoiseChannel::Leakage { p },
                qubits: SmallVec::from_slice(&[q]),
            }));
        }
        QecNoise::Seep(p) => {
            events.extend(targets.iter().map(|&q| NoiseEvent {
                channel: NoiseChannel::Seepage { p },
                qubits: SmallVec::from_slice(&[q]),
            }));
        }
        QecNoise::LeakTransport(p) => {
            events.extend(targets.as_chunks::<2>().0.iter().map(|pair| NoiseEvent {
                channel: NoiseChannel::LeakageTransport { p },
                qubits: SmallVec::from_slice(pair),
            }));
        }
        _ => {
            let (px, py, pz) = channel
                .single_rates()
                .expect("every one-qubit channel has branch rates");
            events.extend(targets.iter().map(|&q| NoiseEvent::pauli(q, px, py, pz)));
        }
    }
}

/// Kraus operators `sqrt(p_k) P_k` of a two-qubit Pauli channel, identity first, in the
/// `Kraus2q` packing (first target the high bit).
fn pauli_channel_2_kraus(rates: &[f64; 15]) -> Vec<[[Complex64; 4]; 4]> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let paulis = [
        [[one, zero], [zero, one]],
        [[zero, one], [one, zero]],
        [[zero, -i], [i, zero]],
        [[one, zero], [zero, -one]],
    ];
    let identity_rate = (1.0 - rates.iter().sum::<f64>()).max(0.0);
    std::iter::once((0usize, identity_rate))
        .chain(rates.iter().enumerate().map(|(b, &rate)| (b + 1, rate)))
        .filter(|&(_, rate)| rate > 0.0)
        .map(|(sample, rate)| {
            let (a, b) = (&paulis[sample / 4], &paulis[sample % 4]);
            let scale = rate.sqrt();
            let mut op = [[zero; 4]; 4];
            for (row, op_row) in op.iter_mut().enumerate() {
                for (col, entry) in op_row.iter_mut().enumerate() {
                    *entry = a[row / 2][col / 2] * b[row % 2][col % 2] * scale;
                }
            }
            op
        })
        .collect()
}

pub(in crate::qec) fn lower_qec_program_to_deferred_circuit(
    program: &QecProgram,
) -> Result<QecDeferredProgram> {
    lower_qec_program_to_deferred_circuit_inner(program, false, false)
}

/// Variant of [`lower_qec_program_to_deferred_circuit`] that admits non-Clifford
/// gates, for the SPD and CAMPS adapters that evaluate observables analytically over the
/// unitary circuit.
pub(in crate::qec) fn lower_qec_program_to_deferred_circuit_allowing_non_clifford(
    program: &QecProgram,
) -> Result<QecDeferredProgram> {
    lower_qec_program_to_deferred_circuit_inner(program, true, false)
}

/// The deferred lowering. `allow_feedforward` admits Pauli feed-forward bodies into
/// [`QecDeferredProgram::feedforwards`] for the record sampler; every other consumer
/// rejects `FEEDFORWARD`.
pub(super) fn lower_qec_program_to_deferred_circuit_inner(
    program: &QecProgram,
    allow_non_clifford: bool,
    allow_feedforward: bool,
) -> Result<QecDeferredProgram> {
    let base_qubits = qec_lowered_num_qubits(program);
    let scratch_qubit = program.num_qubits();
    let mut gates: Vec<DeferredGate> = Vec::with_capacity(qec_deferred_gate_count(program));
    touch_spare_capacity(&mut gates);
    let mut aliases: Vec<usize> = (0..base_qubits).collect();
    let mut measured_aliases = vec![false; base_qubits];
    let mut alias_positions = vec![0; base_qubits];
    let mut last_use: Vec<Option<usize>> = vec![None; base_qubits];
    let mut next_qubit = base_qubits;
    let mut next_record = 0usize;
    let mut deferred_measurements = Vec::with_capacity(program.num_measurements());
    let mut noise_events = Vec::new();
    let mut pair_tables = Vec::new();
    let mut product_tables = Vec::new();
    let mut leak_events = Vec::new();
    let mut feedforwards = Vec::new();

    for op in program.ops() {
        match op {
            QecOp::Gate { gate, targets } => {
                if !allow_non_clifford && !gate.is_clifford() {
                    return Err(qec_non_clifford_error(gate));
                }
                let mapped = map_qec_deferred_targets(targets, &aliases, &measured_aliases)?;
                gates.gate(gate.clone(), mapped.as_slice());
                for &alias in &mapped {
                    last_use[alias] = Some(gates.len());
                }
            }
            QecOp::Measure { basis, qubit } => {
                let alias = qec_deferred_target(*qubit, &aliases, &measured_aliases)?;
                append_basis_to_z_rotation(&mut gates, *basis, alias);
                if *basis != QecBasis::Z {
                    last_use[alias] = Some(gates.len());
                }
                deferred_measurements.push((alias, next_record));
                measured_aliases[alias] = true;
                next_record += 1;
            }
            QecOp::MeasurePauliProduct { terms } => {
                let scratch_alias = qec_scratch_alias(
                    gates.len(),
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut last_use,
                    &mut next_qubit,
                    scratch_qubit,
                );

                let mut mapped_terms = Vec::with_capacity(terms.len());
                for term in terms {
                    let alias = qec_deferred_target(term.qubit, &aliases, &measured_aliases)?;
                    mapped_terms.push(QecPauli::new(term.basis, alias));
                }
                append_mpp_parity_rotations(&mut gates, &mapped_terms, scratch_alias);
                if !mapped_terms.is_empty() {
                    let position = gates.len();
                    for term in &mapped_terms {
                        last_use[term.qubit] = Some(position);
                    }
                    last_use[scratch_alias] = Some(position);
                }

                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::Reset { basis, qubit } => {
                let alias = qec_assign_fresh_alias(
                    gates.len(),
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut last_use,
                    &mut next_qubit,
                    *qubit,
                );
                append_z_to_basis_rotation(&mut gates, *basis, alias);
                if *basis != QecBasis::Z {
                    last_use[alias] = Some(gates.len());
                }
            }
            QecOp::Noise { channel, targets } if channel.is_leakage() => {
                push_deferred_leak_event(
                    channel,
                    targets,
                    &aliases,
                    &measured_aliases,
                    gates.len(),
                    noise_events.len(),
                    &mut leak_events,
                )?;
            }
            QecOp::Noise { channel, targets } => {
                if channel.probability() > 0.0 {
                    let first = noise_events.len();
                    push_qec_deferred_noise_events(
                        channel,
                        targets,
                        &aliases,
                        &measured_aliases,
                        gates.len(),
                        &mut noise_events,
                        &mut pair_tables,
                    )?;
                    for event in &noise_events[first..] {
                        for &alias in &event.targets {
                            last_use[alias] = Some(event.position);
                        }
                    }
                }
            }
            QecOp::CorrelatedError { branches } => {
                if op.is_active_noise() {
                    let mut lowered = Vec::with_capacity(branches.len());
                    let mut remaining = 1.0;
                    for branch in branches {
                        let mut letters = Vec::with_capacity(branch.terms.len());
                        for term in &branch.terms {
                            if let Some(alias) =
                                qec_deferred_noise_target(term.qubit, &aliases, &measured_aliases)?
                            {
                                letters.push((alias, term.basis.pauli_index()));
                            }
                        }
                        lowered.push((remaining * branch.probability, letters));
                        remaining *= 1.0 - branch.probability;
                    }
                    push_qec_deferred_product_event(
                        &lowered,
                        gates.len(),
                        &mut noise_events,
                        &mut product_tables,
                        &mut last_use,
                    );
                }
            }
            QecOp::HeraldedErase { probability, qubit } => {
                let scratch_alias = qec_scratch_alias(
                    gates.len(),
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut last_use,
                    &mut next_qubit,
                    scratch_qubit,
                );
                if *probability > 0.0 {
                    let target = qec_deferred_noise_target(*qubit, &aliases, &measured_aliases)?;
                    let herald = (scratch_alias, 1u8);
                    let lowered: Vec<(f64, Vec<(usize, u8)>)> = (0..4u8)
                        .map(|letter| {
                            let mut letters = vec![herald];
                            if let (Some(alias), 1..=3) = (target, letter) {
                                letters.push((alias, letter));
                            }
                            (probability / 4.0, letters)
                        })
                        .collect();
                    push_qec_deferred_product_event(
                        &lowered,
                        gates.len(),
                        &mut noise_events,
                        &mut product_tables,
                        &mut last_use,
                    );
                }
                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::Pad { value } => {
                let scratch_alias = qec_scratch_alias(
                    gates.len(),
                    &mut aliases,
                    &mut measured_aliases,
                    &mut alias_positions,
                    &mut last_use,
                    &mut next_qubit,
                    scratch_qubit,
                );
                if *value {
                    gates.gate(Gate::X, &[scratch_alias]);
                    last_use[scratch_alias] = Some(gates.len());
                }
                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::RecordFlip { probability } => {
                if *probability > 0.0 {
                    // The record's alias is measured and no later gate touches it, so an
                    // X here reaches its terminal Z measurement and nothing else.
                    let &(alias, _) = deferred_measurements
                        .last()
                        .expect("validation places a record flip after a record");
                    let position = gates.len();
                    noise_events.push(QecDeferredNoiseEvent {
                        channel: QecDeferredChannel::Single([*probability, 0.0, 0.0]),
                        targets: smallvec![alias],
                        position,
                    });
                    last_use[alias] = Some(position);
                }
            }
            QecOp::Feedforward {
                records,
                expected,
                body,
            } if allow_feedforward => {
                let position = gates.len();
                let mut paulis = Vec::with_capacity(body.len());
                for inner in body {
                    let (Some(letter), QecOp::Gate { targets, .. }) =
                        (feedforward_body_letter(inner), inner)
                    else {
                        return Err(feedforward_body_rejection(inner));
                    };
                    if letter == 0 {
                        continue;
                    }
                    if let Some(alias) =
                        qec_deferred_noise_target(targets[0], &aliases, &measured_aliases)?
                    {
                        last_use[alias] = Some(position);
                        paulis.push((alias as u32, letter));
                    }
                }
                if !paulis.is_empty() {
                    feedforwards.push(QecDeferredFeedforward {
                        position,
                        condition: crate::qec::resolve_records(records, next_record)?,
                        invert: !*expected,
                        paulis,
                    });
                }
            }
            QecOp::Feedforward { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC deferred lowering".to_string(),
                    reason: "the deferred circuit holds one static map from random bits to \
                             records, which `FEEDFORWARD` makes depend on the sample; only the \
                             record sampler behind `run_qec_program` applies Pauli bodies on \
                             top of it, and `run_qec_program_reference` executes any body"
                        .to_string(),
                });
            }
            QecOp::ExpectationValue { .. }
            | QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Tick => {}
        }
    }

    ensure_lowered_record_count(program, next_record, "deferred")?;

    let measurement_qubits: Vec<usize> = deferred_measurements
        .iter()
        .map(|&(qubit, _)| qubit)
        .collect();
    debug_assert!(
        deferred_measurements
            .iter()
            .enumerate()
            .all(|(record, &(_, bit))| record == bit)
    );

    let final_qubit_aliases = aliases[..program.num_qubits()].to_vec();
    Ok(QecDeferredProgram {
        gates,
        num_aliases: next_qubit,
        noise_events,
        measurement_qubits,
        alias_positions,
        last_use,
        final_qubit_aliases,
        pair_tables,
        product_tables,
        leak_events,
        feedforwards,
    })
}

/// Alias of the scratch qubit for the next scratch measurement: a fresh one once the
/// current alias has been measured.
fn qec_scratch_alias(
    position: usize,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    alias_positions: &mut Vec<usize>,
    last_use: &mut Vec<Option<usize>>,
    next_qubit: &mut usize,
    scratch_qubit: usize,
) -> usize {
    if measured_aliases[aliases[scratch_qubit]] {
        qec_assign_fresh_alias(
            position,
            aliases,
            measured_aliases,
            alias_positions,
            last_use,
            next_qubit,
            scratch_qubit,
        )
    } else {
        aliases[scratch_qubit]
    }
}

/// Lower mutually exclusive `(absolute rate, [(alias, letter)])` branches into one
/// product event at `position`, or nothing when no branch can fire on a live alias.
fn push_qec_deferred_product_event(
    branches: &[(f64, Vec<(usize, u8)>)],
    position: usize,
    noise_events: &mut Vec<QecDeferredNoiseEvent>,
    product_tables: &mut Vec<QecProductTable>,
    last_use: &mut [Option<usize>],
) {
    let mut targets: Vec<usize> = Vec::new();
    for (_, letters) in branches {
        for &(alias, _) in letters {
            if !targets.contains(&alias) {
                targets.push(alias);
            }
        }
    }
    let p: f64 = branches.iter().map(|(rate, _)| rate).sum();
    if targets.is_empty() || p == 0.0 {
        return;
    }
    let width = targets.len();
    let mut table = QecProductTable {
        rates: Vec::with_capacity(branches.len()),
        letters: vec![0; branches.len() * width],
        width,
    };
    for (branch, (rate, letters)) in branches.iter().enumerate() {
        table.rates.push(*rate);
        for &(alias, letter) in letters {
            let slot = targets
                .iter()
                .position(|&t| t == alias)
                .expect("alias collected");
            table.letters[branch * width + slot] = letter;
        }
    }
    product_tables.push(table);
    for &alias in &targets {
        last_use[alias] = Some(position);
    }
    noise_events.push(QecDeferredNoiseEvent {
        channel: QecDeferredChannel::Products {
            p,
            table: (product_tables.len() - 1) as u32,
        },
        targets: SmallVec::from_vec(targets),
        position,
    });
}

/// Write one zero into every page of `v`'s spare capacity, from the pool when there is
/// one, so the first touch of a deep program's instruction list is not paid serially by
/// the lowering loop: a 96-byte instruction per gate means tens of megabytes of fresh
/// pages per megaquop program, and the page faults cost more than filling them.
fn touch_spare_capacity<T: Send>(v: &mut Vec<T>) {
    let per_page = (4096 / size_of::<T>().max(1)).max(1);
    let spare = v.spare_capacity_mut();
    #[cfg(feature = "parallel")]
    if spare.len() >= 64 * per_page {
        spare.par_chunks_mut(per_page).for_each(|page| {
            page[0] = std::mem::MaybeUninit::zeroed();
        });
        return;
    }
    for page in spare.chunks_mut(per_page) {
        page[0] = std::mem::MaybeUninit::zeroed();
    }
}

/// Gates the deferred lowering emits for `program`, so the list is built in one
/// allocation: the doubling copies and page faults of a growing list cost more than the
/// lowering itself.
fn qec_deferred_gate_count(program: &QecProgram) -> usize {
    let rotation = |basis: QecBasis| match basis {
        QecBasis::X => 1,
        QecBasis::Y => 2,
        QecBasis::Z => 0,
    };
    let gates: usize = program
        .ops()
        .iter()
        .map(|op| match op {
            QecOp::Gate { .. } => 1,
            QecOp::Measure { basis, .. } | QecOp::Reset { basis, .. } => rotation(*basis),
            QecOp::MeasurePauliProduct { terms } => {
                terms.iter().map(|term| 2 * rotation(term.basis) + 1).sum()
            }
            QecOp::Pad { value: true } => 1,
            _ => 0,
        })
        .sum();
    gates
}

fn qec_assign_fresh_alias(
    position: usize,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    alias_positions: &mut Vec<usize>,
    last_use: &mut Vec<Option<usize>>,
    next_qubit: &mut usize,
    logical_qubit: usize,
) -> usize {
    let alias = *next_qubit;
    aliases[logical_qubit] = alias;
    *next_qubit += 1;
    measured_aliases.push(false);
    alias_positions.push(position);
    last_use.push(None);
    alias
}

fn map_qec_deferred_targets(
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<SmallVec<[usize; 4]>> {
    let mut mapped = SmallVec::<[usize; 4]>::with_capacity(targets.len());
    for &target in targets {
        mapped.push(qec_deferred_target(target, aliases, measured_aliases)?);
    }
    Ok(mapped)
}

fn qec_deferred_target(
    target: usize,
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<usize> {
    if target >= aliases.len() {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: aliases.len(),
        });
    }
    let alias = aliases[target];
    if measured_aliases[alias] {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC compiled runner".to_string(),
            reason: "noisy QEC sampling, the detector error model and the analytical \
                     estimators require reset before reusing a measured qubit; the noiseless \
                     compiled sampler accepts the reuse"
                .to_string(),
        });
    }
    Ok(alias)
}

fn push_qec_deferred_noise_events(
    channel: &QecNoise,
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
    position: usize,
    noise_events: &mut Vec<QecDeferredNoiseEvent>,
    pair_tables: &mut Vec<[f64; 15]>,
) -> Result<()> {
    if let Some((px, py, pz)) = channel.single_rates() {
        let mut live_targets = Vec::with_capacity(targets.len());
        for &target in targets {
            if let Some(alias) = qec_deferred_noise_target(target, aliases, measured_aliases)? {
                live_targets.push(alias);
            }
        }
        if !live_targets.is_empty() {
            noise_events.push(QecDeferredNoiseEvent {
                channel: QecDeferredChannel::Single([px, py, pz]),
                targets: SmallVec::from_vec(live_targets),
                position,
            });
        }
        return Ok(());
    }

    let pair_channel = match channel {
        QecNoise::Depolarize2(p) => QecDeferredChannel::Depolarize2(*p),
        QecNoise::PauliChannel2(rates) => {
            pair_tables.push(**rates);
            QecDeferredChannel::PairTable {
                p: channel.probability(),
                table: (pair_tables.len() - 1) as u32,
            }
        }
        _ => unreachable!("one-qubit channels return above"),
    };
    for pair in targets.as_chunks::<2>().0 {
        let first = qec_deferred_noise_target(pair[0], aliases, measured_aliases)?;
        let second = qec_deferred_noise_target(pair[1], aliases, measured_aliases)?;
        let (alias, side) = match (first, second) {
            (Some(q0), Some(q1)) => {
                noise_events.push(QecDeferredNoiseEvent {
                    channel: pair_channel,
                    targets: smallvec![q0, q1],
                    position,
                });
                continue;
            }
            (Some(q), None) => (q, 0),
            (None, Some(q)) => (q, 1),
            (None, None) => continue,
        };
        noise_events.push(QecDeferredNoiseEvent {
            channel: QecDeferredChannel::Single(pair_marginal(channel, side)),
            targets: smallvec![alias],
            position,
        });
    }
    Ok(())
}

/// X, Y, Z rates of the one-qubit channel a pair channel leaves on target `side` (0 or 1)
/// once the other target is measured. `DEPOLARIZE2` leaves `DEPOLARIZE1(0.8 p)`.
fn pair_marginal(channel: &QecNoise, side: usize) -> [f64; 3] {
    match channel {
        QecNoise::Depolarize2(p) => [p * 0.8 / 3.0; 3],
        QecNoise::PauliChannel2(rates) => {
            let mut letters = [0.0; 4];
            for (branch, &rate) in rates.iter().enumerate() {
                let sample = branch + 1;
                letters[if side == 0 { sample / 4 } else { sample % 4 }] += rate;
            }
            [letters[1], letters[2], letters[3]]
        }
        _ => unreachable!("pair channels only"),
    }
}

fn qec_deferred_noise_target(
    target: usize,
    aliases: &[usize],
    measured_aliases: &[bool],
) -> Result<Option<usize>> {
    if target >= aliases.len() {
        return Err(PrismError::InvalidQubit {
            index: target,
            register_size: aliases.len(),
        });
    }
    let alias = aliases[target];
    if measured_aliases[alias] {
        return Ok(None);
    }
    Ok(Some(alias))
}
