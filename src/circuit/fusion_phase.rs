use std::borrow::Cow;

use num_complex::Complex64;

use super::{smallvec, Circuit, Instruction, SmallVec};
use crate::gates::{BatchPhaseData, Gate, MultiFusedData};

use super::fusion::IDENTITY_EPS;

const MIN_BATCH_PHASES: usize = 2;

pub fn fuse_controlled_phases(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_batchable_phases(&circuit) {
        return circuit;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let n = circuit.num_qubits;
    type PhaseVec = SmallVec<[(usize, Complex64); 8]>;
    let mut pending: Vec<Option<PhaseVec>> = (0..n).map(|_| None).collect();

    let flush_qubit =
        |q: usize, pending: &mut [Option<PhaseVec>], output: &mut Vec<Instruction>| {
            if let Some(phases) = pending[q].take() {
                if phases.len() >= MIN_BATCH_PHASES {
                    output.push(Instruction::Gate {
                        gate: Gate::BatchPhase(Box::new(BatchPhaseData { phases })),
                        targets: smallvec![q],
                    });
                } else {
                    for (target, phase) in phases {
                        let one = Complex64::new(1.0, 0.0);
                        let zero = Complex64::new(0.0, 0.0);
                        output.push(Instruction::Gate {
                            gate: Gate::cu([[one, zero], [zero, phase]]),
                            targets: smallvec![q, target],
                        });
                    }
                }
            }
        };

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                if let Some(phase) = gate.controlled_phase() {
                    if gate.num_qubits() == 2 {
                        let control = targets[0];
                        let target = targets[1];
                        flush_qubit(target, &mut pending, &mut output);
                        match &mut pending[control] {
                            Some(v) => v.push((target, phase)),
                            slot => *slot = Some(smallvec![(target, phase)]),
                        }
                        continue;
                    }
                }
                for &q in targets.iter() {
                    flush_qubit(q, &mut pending, &mut output);
                }
                output.push(inst.clone());
            }
            Instruction::Measure { qubit, .. } => {
                flush_qubit(*qubit, &mut pending, &mut output);
                output.push(inst.clone());
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits.iter() {
                    flush_qubit(q, &mut pending, &mut output);
                }
                output.push(inst.clone());
            }
            Instruction::Conditional { targets, .. } => {
                for &q in targets.iter() {
                    flush_qubit(q, &mut pending, &mut output);
                }
                output.push(inst.clone());
            }
        }
    }

    for q in 0..n {
        flush_qubit(q, &mut pending, &mut output);
    }

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

fn has_batchable_phases(circuit: &Circuit) -> bool {
    let mut has_pending = vec![false; circuit.num_qubits];
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                if gate.controlled_phase().is_some() && gate.num_qubits() == 2 {
                    let control = targets[0];
                    if has_pending[control] {
                        return true;
                    }
                    has_pending[control] = true;
                    has_pending[targets[1]] = false;
                } else {
                    for &q in targets.iter() {
                        has_pending[q] = false;
                    }
                }
            }
            Instruction::Measure { qubit, .. } => {
                has_pending[*qubit] = false;
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits {
                    has_pending[q] = false;
                }
            }
            Instruction::Conditional { targets, .. } => {
                for &q in targets.iter() {
                    has_pending[q] = false;
                }
            }
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
