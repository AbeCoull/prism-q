//! Rzz fusion passes: [`fuse_rzz`] rewrites `CX(a,b) Rz(b) CX(a,b)` into
//! [`Gate::Rzz`], [`fuse_batch_rzz`] collects Rzz runs into [`Gate::BatchRzz`].
//! The pass pipeline lives in [`crate::circuit::fusion`].

use std::borrow::Cow;

use super::fusion::push_unique;
use super::plan::{Place, Tracer};
use super::{Circuit, Instruction, SmallVec, smallvec};
use crate::gates::{BatchRzzData, Gate};

pub(super) fn fuse_rzz<'a>(circuit: &'a Circuit, t: &mut Tracer) -> Cow<'a, Circuit> {
    let insts = &circuit.instructions;
    let n = insts.len();
    if n < 3 {
        return Cow::Borrowed(circuit);
    }

    let mut out: Option<Vec<Instruction>> = None;
    let mut kept: Vec<usize> = Vec::new();
    let mut merged: Vec<Option<usize>> = Vec::new();
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
                    let buf = out.get_or_insert_with(|| {
                        if t.on {
                            kept.extend(0..i);
                            merged.extend(std::iter::repeat_n(None, i));
                        }
                        insts[..i].to_vec()
                    });
                    buf.push(Instruction::Gate {
                        gate: Gate::Rzz(*theta),
                        targets: smallvec![t1[0], t1[1]],
                    });
                    if t.on {
                        kept.push(i + 1);
                        merged.push(Some(i + 1));
                    }
                    i += 3;
                    continue;
                }
            }
        }
        if let Some(buf) = out.as_mut() {
            buf.push(insts[i].clone());
            if t.on {
                kept.push(i);
                merged.push(None);
            }
        }
        i += 1;
    }

    match out {
        Some(new_insts) => {
            t.begin();
            for (slot, &src) in kept.iter().enumerate() {
                // The rewritten Rzz carries the inner Rz angle, a 1q payload
                // widened onto the pair the surrounding CX gates named.
                match merged[slot] {
                    Some(rz) => t.merge(&[(rz, Place::Low)]),
                    None => t.keep(src),
                }
            }
            t.commit();
            Cow::Owned(circuit.with_instructions(new_insts))
        }
        None => Cow::Borrowed(circuit),
    }
}

pub(super) fn fuse_batch_rzz<'a>(circuit: &'a Circuit, t: &mut Tracer) -> Cow<'a, Circuit> {
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
    let mut rzz_src: Vec<usize> = Vec::new();
    let mut deferred: Vec<Instruction> = Vec::new();
    let mut deferred_src: Vec<usize> = Vec::new();
    let mut rzz_qubits = vec![false; circuit.num_qubits];
    let mut deferred_qubits = vec![false; circuit.num_qubits];
    t.begin();

    for (i, inst) in insts.iter().enumerate() {
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
                    &mut rzz_src,
                    &mut deferred,
                    &mut deferred_src,
                    &mut rzz_qubits,
                    &mut deferred_qubits,
                    t,
                );
            }
            rzz_run.push((targets[0], targets[1], *theta));
            t.note_idx(&mut rzz_src, i);
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
                    t.note_idx(&mut deferred_src, i);
                    continue;
                }
                Instruction::Gate { gate, targets }
                    if gate.num_qubits() == 1 && !rzz_qubits[targets[0]] =>
                {
                    deferred_qubits[targets[0]] = true;
                    deferred.push(inst.clone());
                    t.note_idx(&mut deferred_src, i);
                    continue;
                }
                _ => {}
            }
        }

        flush_rzz_run(
            &mut output,
            &mut rzz_run,
            &mut rzz_src,
            &mut deferred,
            &mut deferred_src,
            &mut rzz_qubits,
            &mut deferred_qubits,
            t,
        );
        output.push(inst.clone());
        t.keep(i);
    }

    flush_rzz_run(
        &mut output,
        &mut rzz_run,
        &mut rzz_src,
        &mut deferred,
        &mut deferred_src,
        &mut rzz_qubits,
        &mut deferred_qubits,
        t,
    );
    t.commit();

    Cow::Owned(circuit.with_instructions(output))
}

#[allow(clippy::too_many_arguments)]
fn flush_rzz_run(
    output: &mut Vec<Instruction>,
    rzz_run: &mut Vec<(usize, usize, f64)>,
    rzz_src: &mut Vec<usize>,
    deferred: &mut Vec<Instruction>,
    deferred_src: &mut Vec<usize>,
    rzz_qubits: &mut [bool],
    deferred_qubits: &mut [bool],
    t: &mut Tracer,
) {
    // Rzz gates are diagonal and mutually commuting, so a run longer than the
    // kernel group tables hold splits into consecutive batches.
    for (c, chunk) in rzz_run.chunks(BatchRzzData::MAX_EDGES).enumerate() {
        let base = c * BatchRzzData::MAX_EDGES;
        let srcs = if t.on {
            &rzz_src[base..base + chunk.len()]
        } else {
            &[][..]
        };
        emit_rzz_chunk(output, chunk, srcs, t);
    }
    output.append(deferred);
    for &src in deferred_src.iter() {
        t.keep(src);
    }
    deferred_src.clear();
    rzz_run.clear();
    rzz_src.clear();
    rzz_qubits.fill(false);
    deferred_qubits.fill(false);
}

fn emit_rzz_chunk(
    output: &mut Vec<Instruction>,
    chunk: &[(usize, usize, f64)],
    srcs: &[usize],
    t: &mut Tracer,
) {
    if chunk.len() < 2 {
        for (k, &(q0, q1, theta)) in chunk.iter().enumerate() {
            output.push(Instruction::Gate {
                gate: Gate::Rzz(theta),
                targets: smallvec![q0, q1],
            });
            if t.on {
                t.keep(srcs[k]);
            }
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
    if t.on {
        let entries: Vec<Vec<(usize, Place)>> =
            srcs.iter().map(|&src| vec![(src, Place::Plain)]).collect();
        t.batch(&entries);
    }
}
