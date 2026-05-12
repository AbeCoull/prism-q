use std::borrow::Cow;

use num_complex::Complex64;

use super::{smallvec, Circuit, Instruction, SmallVec};
use crate::gates::{BatchPhaseData, Gate, MultiFusedData};

use super::fusion::IDENTITY_EPS;

const MIN_BATCH_PHASES: usize = 2;

type PhaseVec = SmallVec<[(usize, Complex64); 8]>;
type TargetUserVec = SmallVec<[usize; 8]>;
type PendingPhaseVec = Vec<Option<PhaseVec>>;

fn is_controlled_phase_2q(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Gate { gate, .. }
            if gate.controlled_phase().is_some() && gate.num_qubits() == 2
    )
}

fn remove_target_user(target_users: &mut [TargetUserVec], target: usize, control: usize) {
    target_users[target].retain(|c| *c != control);
}

fn emit_phase_chain(control: usize, phases: PhaseVec, output: &mut Vec<Instruction>) {
    if phases.len() >= MIN_BATCH_PHASES {
        output.push(Instruction::Gate {
            gate: Gate::BatchPhase(Box::new(BatchPhaseData { phases })),
            targets: smallvec![control],
        });
        return;
    }

    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    for (target, phase) in phases {
        output.push(Instruction::Gate {
            gate: Gate::cu([[one, zero], [zero, phase]]),
            targets: smallvec![control, target],
        });
    }
}

fn flush_phase_control(
    control: usize,
    pending: &mut [Option<PhaseVec>],
    target_users: &mut [TargetUserVec],
    output: &mut Vec<Instruction>,
) {
    let Some(phases) = pending[control].take() else {
        return;
    };
    for &(target, _) in &phases {
        remove_target_user(target_users, target, control);
    }
    emit_phase_chain(control, phases, output);
}

fn push_pending_phase(
    control: usize,
    target: usize,
    phase: Complex64,
    pending: &mut [Option<PhaseVec>],
    target_users: &mut [TargetUserVec],
) {
    let already_indexed = pending[control]
        .as_ref()
        .is_some_and(|phases| phases.iter().any(|&(t, _)| t == target));
    if !already_indexed {
        target_users[target].push(control);
    }

    match &mut pending[control] {
        Some(v) => v.push((target, phase)),
        slot => *slot = Some(smallvec![(target, phase)]),
    }
}

fn flush_phase_target_conflicts(
    target: usize,
    pending: &mut [Option<PhaseVec>],
    target_users: &mut [TargetUserVec],
    output: &mut Vec<Instruction>,
) {
    let controls = std::mem::take(&mut target_users[target]);
    if controls.is_empty() {
        return;
    }

    let mut re_rooted: PhaseVec = SmallVec::new();
    for control in controls {
        let Some(phases) = pending[control].take() else {
            continue;
        };
        let mut kept: PhaseVec = SmallVec::new();
        for (phase_target, phase) in phases {
            if phase_target == target {
                re_rooted.push((control, phase));
            } else {
                kept.push((phase_target, phase));
            }
        }
        if !kept.is_empty() {
            pending[control] = Some(kept);
        }
    }

    if !re_rooted.is_empty() {
        emit_phase_chain(target, re_rooted, output);
    }
}

fn flush_phase_qubits_in_use(
    qs: &[usize],
    diagonal_only: bool,
    pending: &mut [Option<PhaseVec>],
    target_users: &mut [TargetUserVec],
    output: &mut Vec<Instruction>,
) {
    for &q in qs {
        flush_phase_control(q, pending, target_users, output);
    }
    if diagonal_only {
        return;
    }
    for &q in qs {
        flush_phase_target_conflicts(q, pending, target_users, output);
    }
}

pub fn fuse_controlled_phases(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_batchable_phases(&circuit) {
        return circuit;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let n = circuit.num_qubits;
    let mut pending: PendingPhaseVec = (0..n).map(|_| None).collect();
    let mut target_users: Vec<TargetUserVec> = (0..n).map(|_| SmallVec::new()).collect();

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                if let Some(phase) = gate.controlled_phase() {
                    if gate.num_qubits() == 2 {
                        let control = targets[0];
                        let target = targets[1];
                        // Incoming cphase on target, its action depends on
                        // the current diagonal-frame phase of `target`, so
                        // any pending control chain on `target` must flush first.
                        flush_phase_qubits_in_use(
                            std::slice::from_ref(&target),
                            true,
                            &mut pending,
                            &mut target_users,
                            &mut output,
                        );
                        push_pending_phase(control, target, phase, &mut pending, &mut target_users);
                        continue;
                    }
                }
                let diagonal_only = gate.num_qubits() == 1 && gate.is_diagonal_1q();
                flush_phase_qubits_in_use(
                    targets,
                    diagonal_only,
                    &mut pending,
                    &mut target_users,
                    &mut output,
                );
                output.push(inst.clone());
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                flush_phase_qubits_in_use(
                    std::slice::from_ref(qubit),
                    false,
                    &mut pending,
                    &mut target_users,
                    &mut output,
                );
                output.push(inst.clone());
            }
            Instruction::Barrier { qubits } => {
                flush_phase_qubits_in_use(
                    qubits,
                    false,
                    &mut pending,
                    &mut target_users,
                    &mut output,
                );
                output.push(inst.clone());
            }
            Instruction::Conditional { targets, .. } => {
                flush_phase_qubits_in_use(
                    targets,
                    false,
                    &mut pending,
                    &mut target_users,
                    &mut output,
                );
                output.push(inst.clone());
            }
        }
    }

    for q in 0..n {
        flush_phase_control(q, &mut pending, &mut target_users, &mut output);
    }

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

fn has_batchable_phases(circuit: &Circuit) -> bool {
    type PendingTargets = SmallVec<[usize; 8]>;

    if !circuit.instructions.iter().any(is_controlled_phase_2q) {
        return false;
    }

    fn push_pending_target(
        control: usize,
        target: usize,
        pending: &mut [PendingTargets],
        target_users: &mut [TargetUserVec],
    ) {
        if !pending[control].contains(&target) {
            target_users[target].push(control);
        }
        pending[control].push(target);
    }

    fn flush_control(
        q: usize,
        pending: &mut [PendingTargets],
        target_users: &mut [TargetUserVec],
    ) -> bool {
        let len = pending[q].len();
        for target in pending[q].drain(..) {
            remove_target_user(target_users, target, q);
        }
        len >= MIN_BATCH_PHASES
    }

    fn flush_target_conflicts(
        target: usize,
        pending: &mut [PendingTargets],
        target_users: &mut [TargetUserVec],
    ) -> bool {
        let controls = std::mem::take(&mut target_users[target]);
        let mut re_rooted = 0usize;
        for control in controls {
            let mut kept: PendingTargets = SmallVec::new();
            for pending_target in pending[control].drain(..) {
                if pending_target == target {
                    re_rooted += 1;
                } else {
                    kept.push(pending_target);
                }
            }
            pending[control] = kept;
        }
        re_rooted >= MIN_BATCH_PHASES
    }

    fn flush_qubits_in_use(
        qs: &[usize],
        diagonal_only: bool,
        pending: &mut [PendingTargets],
        target_users: &mut [TargetUserVec],
    ) -> bool {
        for &q in qs {
            if flush_control(q, pending, target_users) {
                return true;
            }
        }
        if diagonal_only {
            return false;
        }
        for &q in qs {
            if flush_target_conflicts(q, pending, target_users) {
                return true;
            }
        }
        false
    }

    let mut pending: Vec<PendingTargets> =
        (0..circuit.num_qubits).map(|_| SmallVec::new()).collect();
    let mut target_users: Vec<TargetUserVec> =
        (0..circuit.num_qubits).map(|_| SmallVec::new()).collect();
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                if gate.controlled_phase().is_some() && gate.num_qubits() == 2 {
                    let control = targets[0];
                    let target = targets[1];
                    if flush_qubits_in_use(
                        std::slice::from_ref(&target),
                        true,
                        &mut pending,
                        &mut target_users,
                    ) {
                        return true;
                    }
                    if pending[control].len() + 1 >= MIN_BATCH_PHASES {
                        return true;
                    }
                    push_pending_target(control, target, &mut pending, &mut target_users);
                } else {
                    let diagonal_only = gate.num_qubits() == 1 && gate.is_diagonal_1q();
                    if flush_qubits_in_use(targets, diagonal_only, &mut pending, &mut target_users)
                    {
                        return true;
                    }
                }
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                if flush_qubits_in_use(
                    std::slice::from_ref(qubit),
                    false,
                    &mut pending,
                    &mut target_users,
                ) {
                    return true;
                }
            }
            Instruction::Barrier { qubits } => {
                if flush_qubits_in_use(qubits, false, &mut pending, &mut target_users) {
                    return true;
                }
            }
            Instruction::Conditional { targets, .. } => {
                if flush_qubits_in_use(targets, false, &mut pending, &mut target_users) {
                    return true;
                }
            }
        }
    }
    for q in 0..circuit.num_qubits {
        if flush_control(q, &mut pending, &mut target_users) {
            return true;
        }
    }
    false
}

fn is_batchable_1q(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Gate { targets, gate, .. }
            if targets.len() == 1
                && !matches!(gate, Gate::BatchPhase(_) | Gate::MultiFused(_) | Gate::Multi2q(_) | Gate::DiagonalBatch(_))
    )
}

pub(super) fn batch_post_phase_1q(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    let mut max_run = 0usize;
    let mut run = 0usize;
    for inst in &circuit.instructions {
        if is_batchable_1q(inst) {
            run += 1;
            max_run = max_run.max(run);
        } else {
            run = 0;
        }
    }
    if max_run < 2 {
        return circuit;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut pending: Vec<(usize, [[Complex64; 2]; 2])> = Vec::new();

    let flush = |pending: &mut Vec<(usize, [[Complex64; 2]; 2])>, output: &mut Vec<Instruction>| {
        if pending.len() >= 2 {
            let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
            for &(q, _) in pending.iter() {
                if !targets.contains(&q) {
                    targets.push(q);
                }
            }
            targets.sort_unstable();
            let all_diagonal = pending
                .iter()
                .all(|(_, m)| m[0][1].norm() < IDENTITY_EPS && m[1][0].norm() < IDENTITY_EPS);
            output.push(Instruction::Gate {
                gate: Gate::MultiFused(Box::new(MultiFusedData {
                    gates: std::mem::take(pending),
                    all_diagonal,
                })),
                targets,
            });
        } else {
            for (q, mat) in pending.drain(..) {
                output.push(Instruction::Gate {
                    gate: Gate::Fused(Box::new(mat)),
                    targets: smallvec![q],
                });
            }
        }
    };

    for inst in &circuit.instructions {
        if is_batchable_1q(inst) {
            if let Instruction::Gate { gate, targets, .. } = inst {
                pending.push((targets[0], gate.matrix_2x2()));
            }
        } else {
            flush(&mut pending, &mut output);
            output.push(inst.clone());
        }
    }

    flush(&mut pending, &mut output);

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}
