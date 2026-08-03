//! Rzz fusion passes: [`fuse_rzz`] rewrites `CX(a,b) Rz(b) CX(a,b)` into
//! [`Gate::Rzz`], [`fuse_batch_rzz`] collects Rzz runs into [`Gate::BatchRzz`].
//! The pass pipeline lives in [`crate::circuit::fusion`].

use std::borrow::Cow;

use super::fusion::push_unique;
use super::{Circuit, Instruction, SmallVec, smallvec};
use crate::gates::{BatchRzzData, Gate};

pub(super) fn fuse_rzz(circuit: &Circuit) -> Cow<'_, Circuit> {
    let insts = &circuit.instructions;
    let n = insts.len();
    if n < 3 {
        return Cow::Borrowed(circuit);
    }

    let mut out: Option<Vec<Instruction>> = None;
    let mut i = 0;
    while i < n {
        if i + 2 < n {
            if let (
                Instruction::Gate {
                    gate: Gate::Cx,
                    targets: t1,
                },
                Instruction::Gate {
                    gate: Gate::Rz(theta),
                    targets: t2,
                },
                Instruction::Gate {
                    gate: Gate::Cx,
                    targets: t3,
                },
            ) = (&insts[i], &insts[i + 1], &insts[i + 2])
            {
                if t1.as_slice() == t3.as_slice() && t2.len() == 1 && t2[0] == t1[1] {
                    let buf = out.get_or_insert_with(|| insts[..i].to_vec());
                    buf.push(Instruction::Gate {
                        gate: Gate::Rzz(*theta),
                        targets: smallvec![t1[0], t1[1]],
                    });
                    i += 3;
                    continue;
                }
            }
        }
        if let Some(buf) = out.as_mut() {
            buf.push(insts[i].clone());
        }
        i += 1;
    }

    match out {
        Some(new_insts) => Cow::Owned(circuit.with_instructions(new_insts)),
        None => Cow::Borrowed(circuit),
    }
}

pub(super) fn fuse_batch_rzz(circuit: &Circuit) -> Cow<'_, Circuit> {
    let insts = &circuit.instructions;
    let n = insts.len();
    if n < 2 {
        return Cow::Borrowed(circuit);
    }

    let rzz_count = insts
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Rzz(_),
                    ..
                }
            )
        })
        .count();
    if rzz_count < 2 {
        return Cow::Borrowed(circuit);
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(n);
    let mut rzz_run: Vec<(usize, usize, f64)> = Vec::new();
    let mut deferred: Vec<Instruction> = Vec::new();
    let mut rzz_qubits = vec![false; circuit.num_qubits];
    let mut deferred_qubits = vec![false; circuit.num_qubits];

    for inst in insts {
        if let Instruction::Gate {
            gate: Gate::Rzz(theta),
            targets,
        } = inst
        {
            // Deferred gates are re-emitted after the whole batch. Admitting an
            // Rzz on a deferred gate's qubit would sink that gate behind an Rzz
            // it does not commute with, so close the run first.
            if deferred_qubits[targets[0]] || deferred_qubits[targets[1]] {
                flush_rzz_run(
                    &mut output,
                    &mut rzz_run,
                    &mut deferred,
                    &mut rzz_qubits,
                    &mut deferred_qubits,
                );
            }
            rzz_run.push((targets[0], targets[1], *theta));
            rzz_qubits[targets[0]] = true;
            rzz_qubits[targets[1]] = true;
            continue;
        }

        if !rzz_run.is_empty() {
            match inst {
                Instruction::Gate { gate, .. }
                    if gate.num_qubits() == 1 && gate.is_diagonal_1q() =>
                {
                    deferred.push(inst.clone());
                    continue;
                }
                Instruction::Gate { gate, targets }
                    if gate.num_qubits() == 1 && !rzz_qubits[targets[0]] =>
                {
                    deferred_qubits[targets[0]] = true;
                    deferred.push(inst.clone());
                    continue;
                }
                _ => {}
            }
        }

        flush_rzz_run(
            &mut output,
            &mut rzz_run,
            &mut deferred,
            &mut rzz_qubits,
            &mut deferred_qubits,
        );
        output.push(inst.clone());
    }

    flush_rzz_run(
        &mut output,
        &mut rzz_run,
        &mut deferred,
        &mut rzz_qubits,
        &mut deferred_qubits,
    );

    Cow::Owned(circuit.with_instructions(output))
}

fn flush_rzz_run(
    output: &mut Vec<Instruction>,
    rzz_run: &mut Vec<(usize, usize, f64)>,
    deferred: &mut Vec<Instruction>,
    rzz_qubits: &mut [bool],
    deferred_qubits: &mut [bool],
) {
    // Rzz gates are diagonal and mutually commuting, so a run longer than the
    // kernel group tables hold splits into consecutive batches.
    for chunk in rzz_run.chunks(BatchRzzData::MAX_EDGES) {
        emit_rzz_chunk(output, chunk);
    }
    output.append(deferred);
    rzz_run.clear();
    rzz_qubits.fill(false);
    deferred_qubits.fill(false);
}

fn emit_rzz_chunk(output: &mut Vec<Instruction>, chunk: &[(usize, usize, f64)]) {
    if chunk.len() < 2 {
        for &(q0, q1, theta) in chunk {
            output.push(Instruction::Gate {
                gate: Gate::Rzz(theta),
                targets: smallvec![q0, q1],
            });
        }
        return;
    }

    let mut tgts: SmallVec<[usize; 4]> = SmallVec::new();
    for &(q0, q1, _) in chunk {
        push_unique(&mut tgts, q0);
        push_unique(&mut tgts, q1);
    }
    output.push(Instruction::Gate {
        gate: Gate::BatchRzz(Box::new(BatchRzzData {
            edges: chunk.to_vec(),
        })),
        targets: tgts,
    });
}
