use std::borrow::Cow;

use super::{smallvec, Circuit, Instruction, SmallVec};
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
        Some(new_insts) => {
            let mut c = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
            c.instructions = new_insts;
            Cow::Owned(c)
        }
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

    for inst in insts {
        if let Instruction::Gate {
            gate: Gate::Rzz(theta),
            targets,
        } = inst
        {
            rzz_run.push((targets[0], targets[1], *theta));
            rzz_qubits[targets[0]] = true;
            rzz_qubits[targets[1]] = true;
            continue;
        }

        if !rzz_run.is_empty() {
            let can_pass = match inst {
                Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                    gate.is_diagonal_1q() || !rzz_qubits[targets[0]]
                }
                _ => false,
            };
            if can_pass {
                deferred.push(inst.clone());
                continue;
            }
        }

        flush_rzz_run(&mut output, &mut rzz_run, &mut deferred, &mut rzz_qubits);
        output.push(inst.clone());
    }

    flush_rzz_run(&mut output, &mut rzz_run, &mut deferred, &mut rzz_qubits);

    let mut c = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
    c.instructions = output;
    Cow::Owned(c)
}

fn flush_rzz_run(
    output: &mut Vec<Instruction>,
    rzz_run: &mut Vec<(usize, usize, f64)>,
    deferred: &mut Vec<Instruction>,
    rzz_qubits: &mut [bool],
) {
    if rzz_run.len() >= 2 {
        let mut tgts: SmallVec<[usize; 4]> = SmallVec::new();
        for &(q0, q1, _) in rzz_run.iter() {
            if !tgts.contains(&q0) {
                tgts.push(q0);
            }
            if !tgts.contains(&q1) {
                tgts.push(q1);
            }
        }
        output.push(Instruction::Gate {
            gate: Gate::BatchRzz(Box::new(BatchRzzData {
                edges: rzz_run.clone(),
            })),
            targets: tgts,
        });
    } else {
        for &(q0, q1, theta) in rzz_run.iter() {
            output.push(Instruction::Gate {
                gate: Gate::Rzz(theta),
                targets: smallvec![q0, q1],
            });
        }
    }
    output.append(deferred);
    rzz_run.clear();
    rzz_qubits.fill(false);
}
