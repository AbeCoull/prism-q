use super::{
    append_basis_to_z_rotation, append_z_to_basis_rotation, QecNoise, QecOp, QecPauli, QecProgram,
};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{
    batch_propagate_backward, compile_measurements, rng::Xoshiro256PlusPlus, xor_words,
    CompiledSampler, PackedShots,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
struct QecDeferredNoiseEvent {
    channel: QecNoise,
    targets: Vec<usize>,
    position: usize,
}

struct QecDeferredProgram {
    circuit: Circuit,
    noise_events: Vec<QecDeferredNoiseEvent>,
    measurement_qubits: Vec<usize>,
}

pub(super) struct QecCompiledNoiseSampler {
    noiseless: CompiledSampler,
    events: QecNoiseSensitivity,
    num_measurements: usize,
    rng: Xoshiro256PlusPlus,
}

impl QecCompiledNoiseSampler {
    pub(super) fn sample_measurements_packed(&mut self, num_shots: usize) -> Result<PackedShots> {
        let measurements = self.noiseless.try_sample_bulk_packed(num_shots)?;
        if self.events.is_empty() || num_shots == 0 || self.num_measurements == 0 {
            return Ok(measurements);
        }

        let m_words = self.num_measurements.div_ceil(64);
        let mut data = measurements.into_shot_major_data();
        self.events
            .apply(&mut data, num_shots, m_words, &mut self.rng);
        Ok(PackedShots::from_shot_major(
            data,
            num_shots,
            self.num_measurements,
        ))
    }
}

struct QecNoiseSensitivity {
    events: Vec<QecNoiseSensitivityEvent>,
}

impl QecNoiseSensitivity {
    fn new() -> Self {
        Self { events: Vec::new() }
    }

    fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn push_single(&mut self, x_flip: &[u64], z_flip: &[u64], px: f64, py: f64, pz: f64) {
        let x_is_zero = x_flip.iter().all(|&w| w == 0);
        let z_is_zero = z_flip.iter().all(|&w| w == 0);
        if px + py + pz == 0.0 || (x_is_zero && z_is_zero) {
            return;
        }

        let (px, py, pz) = if x_is_zero {
            (px + py, 0.0, 0.0)
        } else if z_is_zero {
            (0.0, 0.0, py + pz)
        } else if x_flip == z_flip {
            (px + pz, 0.0, 0.0)
        } else {
            (px, py, pz)
        };
        if px + py + pz == 0.0 {
            return;
        }

        self.events.push(QecNoiseSensitivityEvent::Single {
            x_flip: x_flip.to_vec(),
            z_flip: z_flip.to_vec(),
            px,
            py,
            pz,
        });
    }

    fn push_pair(
        &mut self,
        q0_x_flip: &[u64],
        q0_z_flip: &[u64],
        q1_x_flip: &[u64],
        q1_z_flip: &[u64],
        p: f64,
    ) {
        if p == 0.0 {
            return;
        }

        let m_words = q0_x_flip.len();
        let mut branch_flips = Vec::with_capacity(15 * m_words);
        let mut branch = vec![0u64; m_words];
        let mut any = false;
        for sample in 1..=15 {
            let first = sample / 4;
            let second = sample % 4;
            branch.fill(0);
            append_qec_pauli_noise_effect(&mut branch, first, q0_x_flip, q0_z_flip);
            append_qec_pauli_noise_effect(&mut branch, second, q1_x_flip, q1_z_flip);
            any |= branch.iter().any(|&w| w != 0);
            branch_flips.extend_from_slice(&branch);
        }

        if any {
            self.events
                .push(QecNoiseSensitivityEvent::Pair { branch_flips, p });
        }
    }

    fn apply(
        &self,
        data: &mut [u64],
        num_shots: usize,
        m_words: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) {
        for event in &self.events {
            match event {
                QecNoiseSensitivityEvent::Single {
                    x_flip,
                    z_flip,
                    px,
                    py,
                    pz,
                } => apply_qec_single_noise_event(
                    data,
                    num_shots,
                    m_words,
                    QecSingleNoiseView {
                        x_flip,
                        z_flip,
                        px: *px,
                        py: *py,
                        p_event: *px + *py + *pz,
                    },
                    rng,
                ),
                QecNoiseSensitivityEvent::Pair { branch_flips, p } => {
                    apply_qec_pair_noise_event(data, num_shots, m_words, branch_flips, *p, rng)
                }
            }
        }
    }
}

enum QecNoiseSensitivityEvent {
    Single {
        x_flip: Vec<u64>,
        z_flip: Vec<u64>,
        px: f64,
        py: f64,
        pz: f64,
    },
    Pair {
        branch_flips: Vec<u64>,
        p: f64,
    },
}

#[derive(Clone, Copy)]
struct QecSingleNoiseView<'a> {
    x_flip: &'a [u64],
    z_flip: &'a [u64],
    px: f64,
    py: f64,
    p_event: f64,
}

pub(super) fn compile_qec_noisy_sampler(program: &QecProgram) -> Result<QecCompiledNoiseSampler> {
    let deferred = lower_qec_program_to_deferred_circuit(program)?;
    let events = compile_qec_noise_sensitivity(&deferred)?;
    let noiseless = compile_measurements(&deferred.circuit, program.options().seed)?;
    Ok(QecCompiledNoiseSampler {
        noiseless,
        events,
        num_measurements: deferred.measurement_qubits.len(),
        rng: qec_noise_rng(program.options().seed),
    })
}

fn qec_noise_rng(seed: u64) -> Xoshiro256PlusPlus {
    let mut seed_rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(0x51A7_EC01));
    Xoshiro256PlusPlus::from_chacha(&mut seed_rng)
}

fn lower_qec_program_to_deferred_circuit(program: &QecProgram) -> Result<QecDeferredProgram> {
    let has_mpp = program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::MeasurePauliProduct { .. }));
    let base_qubits = program.num_qubits() + usize::from(has_mpp);
    let scratch_qubit = program.num_qubits();
    let mut circuit = Circuit::new(base_qubits, program.num_measurements());
    let mut aliases: Vec<usize> = (0..base_qubits).collect();
    let mut measured_aliases = vec![false; base_qubits];
    let mut next_qubit = base_qubits;
    let mut next_record = 0usize;
    let mut deferred_measurements = Vec::with_capacity(program.num_measurements());
    let mut noise_events = Vec::new();

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
                let mapped = map_qec_deferred_targets(targets, &aliases, &measured_aliases)?;
                circuit.add_gate(gate.clone(), mapped.as_slice());
            }
            QecOp::Measure { basis, qubit } => {
                let alias = qec_deferred_target(*qubit, &aliases, &measured_aliases)?;
                append_basis_to_z_rotation(&mut circuit, *basis, alias);
                deferred_measurements.push((alias, next_record));
                measured_aliases[alias] = true;
                next_record += 1;
            }
            QecOp::MeasurePauliProduct { terms } => {
                if !has_mpp {
                    return Err(PrismError::InvalidParameter {
                        message: "internal QEC MPP scratch qubit was not allocated".to_string(),
                    });
                }
                let scratch_alias = if measured_aliases[aliases[scratch_qubit]] {
                    qec_assign_fresh_alias(
                        &mut circuit,
                        &mut aliases,
                        &mut measured_aliases,
                        &mut next_qubit,
                        scratch_qubit,
                    )
                } else {
                    aliases[scratch_qubit]
                };

                let mut mapped_terms = Vec::with_capacity(terms.len());
                for term in terms {
                    let alias = qec_deferred_target(term.qubit, &aliases, &measured_aliases)?;
                    mapped_terms.push(QecPauli::new(term.basis, alias));
                }

                for term in &mapped_terms {
                    append_basis_to_z_rotation(&mut circuit, term.basis, term.qubit);
                }
                for term in &mapped_terms {
                    circuit.add_gate(Gate::Cx, &[term.qubit, scratch_alias]);
                }
                for term in mapped_terms.iter().rev() {
                    append_z_to_basis_rotation(&mut circuit, term.basis, term.qubit);
                }

                deferred_measurements.push((scratch_alias, next_record));
                measured_aliases[scratch_alias] = true;
                next_record += 1;
            }
            QecOp::Reset { basis, qubit } => {
                let alias = qec_assign_fresh_alias(
                    &mut circuit,
                    &mut aliases,
                    &mut measured_aliases,
                    &mut next_qubit,
                    *qubit,
                );
                append_z_to_basis_rotation(&mut circuit, *basis, alias);
            }
            QecOp::Noise { channel, targets } => {
                if channel.probability() > 0.0 {
                    push_qec_deferred_noise_events(
                        *channel,
                        targets,
                        &aliases,
                        &measured_aliases,
                        circuit.instructions.len(),
                        &mut noise_events,
                    )?;
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
                "QEC deferred lowering produced {next_record} records, expected {}",
                program.num_measurements()
            ),
        });
    }

    let mut measurement_qubits = Vec::with_capacity(deferred_measurements.len());
    for (qubit, classical_bit) in deferred_measurements {
        measurement_qubits.push(qubit);
        circuit.add_measure(qubit, classical_bit);
    }

    Ok(QecDeferredProgram {
        circuit,
        noise_events,
        measurement_qubits,
    })
}

fn qec_assign_fresh_alias(
    circuit: &mut Circuit,
    aliases: &mut [usize],
    measured_aliases: &mut Vec<bool>,
    next_qubit: &mut usize,
    logical_qubit: usize,
) -> usize {
    let alias = *next_qubit;
    aliases[logical_qubit] = alias;
    *next_qubit += 1;
    measured_aliases.push(false);
    circuit.num_qubits = *next_qubit;
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
            reason: "compiled QEC runner requires reset before reusing a measured qubit"
                .to_string(),
        });
    }
    Ok(alias)
}

fn push_qec_deferred_noise_events(
    channel: QecNoise,
    targets: &[usize],
    aliases: &[usize],
    measured_aliases: &[bool],
    position: usize,
    noise_events: &mut Vec<QecDeferredNoiseEvent>,
) -> Result<()> {
    match channel {
        QecNoise::XError(_) | QecNoise::ZError(_) | QecNoise::Depolarize1(_) => {
            let mut live_targets = Vec::with_capacity(targets.len());
            for &target in targets {
                if let Some(alias) = qec_deferred_noise_target(target, aliases, measured_aliases)? {
                    live_targets.push(alias);
                }
            }
            if !live_targets.is_empty() {
                noise_events.push(QecDeferredNoiseEvent {
                    channel,
                    targets: live_targets,
                    position,
                });
            }
        }
        QecNoise::Depolarize2(p) => {
            for pair in targets.chunks_exact(2) {
                let first = qec_deferred_noise_target(pair[0], aliases, measured_aliases)?;
                let second = qec_deferred_noise_target(pair[1], aliases, measured_aliases)?;
                match (first, second) {
                    (Some(q0), Some(q1)) => noise_events.push(QecDeferredNoiseEvent {
                        channel,
                        targets: vec![q0, q1],
                        position,
                    }),
                    (Some(q), None) | (None, Some(q)) => {
                        noise_events.push(QecDeferredNoiseEvent {
                            channel: QecNoise::Depolarize1(p * 0.8),
                            targets: vec![q],
                            position,
                        });
                    }
                    (None, None) => {}
                }
            }
        }
    }
    Ok(())
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

fn compile_qec_noise_sensitivity(deferred: &QecDeferredProgram) -> Result<QecNoiseSensitivity> {
    let num_measurements = deferred.measurement_qubits.len();
    let m_words = num_measurements.div_ceil(64);
    let mut x_packed = vec![vec![0u64; m_words]; deferred.circuit.num_qubits];
    let mut z_packed = vec![vec![0u64; m_words]; deferred.circuit.num_qubits];
    let mut sign_packed = vec![0u64; m_words];

    for (record, &qubit) in deferred.measurement_qubits.iter().enumerate() {
        z_packed[qubit][record / 64] |= 1u64 << (record % 64);
    }

    let gate_count = deferred
        .circuit
        .instructions
        .iter()
        .filter(|inst| matches!(inst, Instruction::Gate { .. }))
        .count();
    let mut noise_by_position = vec![Vec::new(); gate_count + 1];
    for event in &deferred.noise_events {
        if event.position > gate_count {
            return Err(PrismError::InvalidParameter {
                message: "QEC noise event position exceeds deferred gate count".to_string(),
            });
        }
        noise_by_position[event.position].push(event.clone());
    }

    let mut events = QecNoiseSensitivity::new();
    for gate_position in (0..gate_count).rev() {
        for event in &noise_by_position[gate_position + 1] {
            push_qec_noise_sensitivity_event(event, &x_packed, &z_packed, &mut events);
        }
        let (gate, targets) = match &deferred.circuit.instructions[gate_position] {
            Instruction::Gate { gate, targets } => (gate, targets.as_slice()),
            _ => {
                return Err(PrismError::InvalidParameter {
                    message: "QEC deferred circuit expected gate before terminal measurements"
                        .to_string(),
                });
            }
        };
        batch_propagate_backward(
            &mut x_packed,
            &mut z_packed,
            &mut sign_packed,
            gate,
            targets,
            m_words,
        );
    }

    for event in &noise_by_position[0] {
        push_qec_noise_sensitivity_event(event, &x_packed, &z_packed, &mut events);
    }

    Ok(events)
}

fn push_qec_noise_sensitivity_event(
    event: &QecDeferredNoiseEvent,
    x_packed: &[Vec<u64>],
    z_packed: &[Vec<u64>],
    events: &mut QecNoiseSensitivity,
) {
    match event.channel {
        QecNoise::XError(p) => {
            for &target in &event.targets {
                events.push_single(&x_packed[target], &z_packed[target], p, 0.0, 0.0);
            }
        }
        QecNoise::ZError(p) => {
            for &target in &event.targets {
                events.push_single(&x_packed[target], &z_packed[target], 0.0, 0.0, p);
            }
        }
        QecNoise::Depolarize1(p) => {
            let branch_p = p / 3.0;
            for &target in &event.targets {
                events.push_single(
                    &x_packed[target],
                    &z_packed[target],
                    branch_p,
                    branch_p,
                    branch_p,
                );
            }
        }
        QecNoise::Depolarize2(p) => {
            for pair in event.targets.chunks_exact(2) {
                events.push_pair(
                    &x_packed[pair[0]],
                    &z_packed[pair[0]],
                    &x_packed[pair[1]],
                    &z_packed[pair[1]],
                    p,
                );
            }
        }
    }
}

fn apply_qec_single_noise_event(
    data: &mut [u64],
    num_shots: usize,
    m_words: usize,
    event: QecSingleNoiseView<'_>,
    rng: &mut Xoshiro256PlusPlus,
) {
    if event.p_event == 0.0 {
        return;
    }

    if event.p_event >= 0.5 || num_shots < 32 {
        for shot in 0..num_shots {
            let r = rng.next_f64();
            let base = shot * m_words;
            apply_qec_single_noise_branch(&mut data[base..base + m_words], event, r);
        }
        return;
    }

    let ln_1mp = (1.0 - event.p_event).ln();
    let px_frac = event.px / event.p_event;
    let pxy_frac = (event.px + event.py) / event.p_event;
    let conditional_event = QecSingleNoiseView {
        x_flip: event.x_flip,
        z_flip: event.z_flip,
        px: px_frac,
        py: pxy_frac - px_frac,
        p_event: 1.0,
    };
    let mut shot = qec_geometric_sample(rng, ln_1mp);
    while shot < num_shots {
        let r = rng.next_f64();
        let base = shot * m_words;
        apply_qec_single_noise_branch(&mut data[base..base + m_words], conditional_event, r);
        shot += 1 + qec_geometric_sample(rng, ln_1mp);
    }
}

fn apply_qec_single_noise_branch(shot_words: &mut [u64], event: QecSingleNoiseView<'_>, r: f64) {
    // x_flip / z_flip are the X / Z components of the propagated measurement
    // Pauli at this point in the circuit. A Pauli error flips a measurement
    // record iff it anti-commutes with the propagated Pauli on this qubit:
    // X anti-commutes with Z, Z anti-commutes with X, Y anti-commutes with both.
    if r < event.px {
        xor_words(shot_words, event.z_flip);
    } else if r < event.px + event.py {
        xor_words(shot_words, event.x_flip);
        xor_words(shot_words, event.z_flip);
    } else if r < event.p_event {
        xor_words(shot_words, event.x_flip);
    }
}

fn apply_qec_pair_noise_event(
    data: &mut [u64],
    num_shots: usize,
    m_words: usize,
    branch_flips: &[u64],
    p: f64,
    rng: &mut Xoshiro256PlusPlus,
) {
    if p == 0.0 {
        return;
    }

    if p >= 0.5 || num_shots < 32 {
        for shot in 0..num_shots {
            if rng.next_f64() < p {
                apply_qec_pair_noise_branch(data, shot, m_words, branch_flips, rng);
            }
        }
        return;
    }

    let ln_1mp = (1.0 - p).ln();
    let mut shot = qec_geometric_sample(rng, ln_1mp);
    while shot < num_shots {
        apply_qec_pair_noise_branch(data, shot, m_words, branch_flips, rng);
        shot += 1 + qec_geometric_sample(rng, ln_1mp);
    }
}

fn apply_qec_pair_noise_branch(
    data: &mut [u64],
    shot: usize,
    m_words: usize,
    branch_flips: &[u64],
    rng: &mut Xoshiro256PlusPlus,
) {
    // branch_flips packs the 15 non-identity 2-qubit Pauli effects in fixed
    // order (see push_pair). Picking a uniform branch reproduces the
    // depolarize-2 distribution conditional on an event firing.
    let branch = qec_uniform_15(rng);
    let flip_base = branch * m_words;
    let shot_base = shot * m_words;
    xor_words(
        &mut data[shot_base..shot_base + m_words],
        &branch_flips[flip_base..flip_base + m_words],
    );
}

fn append_qec_pauli_noise_effect(branch: &mut [u64], pauli: usize, x_flip: &[u64], z_flip: &[u64]) {
    match pauli {
        0 => {}
        1 => xor_words(branch, z_flip),
        2 => {
            xor_words(branch, x_flip);
            xor_words(branch, z_flip);
        }
        _ => xor_words(branch, x_flip),
    }
}

/// Inverse-CDF sample from `Geometric(p)` (number of failures before the
/// first success). `ln_1mp` is `(1 - p).ln()` precomputed by the caller so
/// hot-path loops avoid recomputing it. Used to skip ahead to the next shot
/// where a Pauli-noise event fires when `p` is small.
#[inline(always)]
fn qec_geometric_sample(rng: &mut Xoshiro256PlusPlus, ln_1mp: f64) -> usize {
    let u: f64 = 1.0 - rng.next_f64();
    (u.ln() / ln_1mp) as usize
}

#[inline(always)]
fn qec_uniform_15(rng: &mut Xoshiro256PlusPlus) -> usize {
    const BRANCHES: u64 = 15;
    const ZONE: u64 = u64::MAX - (u64::MAX % BRANCHES);
    loop {
        let value = rng.next_u64();
        if value < ZONE {
            return (value % BRANCHES) as usize;
        }
    }
}
