//! Gate fusion pipeline: self-inverse cancellation, 1q and 2q matrix fusion, and batching
//! into the tiled and diagonal gate families, each pass gated on a qubit-count floor.
//! Fused matrices multiply on the left, so gates G1, G2, G3 fuse to G3 · G2 · G1.

use std::borrow::Cow;

use num_complex::Complex64;

use super::{Circuit, GuardedRegion, Instruction, SmallVec, smallvec};
use crate::gates::{
    DiagEntry, DiagonalBatchData, Gate, IDENTITY_EPS, MULTI_2Q_HIGH_BUDGET, Multi2qData,
    MultiFusedData, is_diagonal_2x2, is_diagonal_4x4, kron_2x2, mat_mul_2x2, mat_mul_4x4,
    multi_2q_join,
};

use super::fusion_phase::{batch_post_phase_1q, fuse_controlled_phases};
use super::fusion_rzz::{fuse_batch_rzz, fuse_rzz};

use super::plan::{Place, Tracer};

// Under miri the fusion floors drop to the reduced parallel threshold (see
// `PARALLEL_THRESHOLD_QUBITS` in `backend/mod.rs`), so the fused kernel forms
// appear at sizes the interpreter can execute. Native values are unchanged.

/// Floor for the 1q fusion, reorder, and batching passes. Below it gate execution takes
/// nanoseconds and cloning the instruction stream costs more than fusion saves.
#[cfg(not(miri))]
pub(crate) const MIN_QUBITS_FOR_FUSION: usize = 10;
#[cfg(miri)]
pub(crate) const MIN_QUBITS_FOR_FUSION: usize = 8;

#[cfg(not(miri))]
const MIN_QUBITS_FOR_MULTI_FUSION: usize = 14;
#[cfg(miri)]
const MIN_QUBITS_FOR_MULTI_FUSION: usize = 8;

/// Floor for the diagonal batch passes (BatchRzz, BatchPhase, DiagonalBatch), whose LUT
/// setup needs a large state to amortize.
#[cfg(not(miri))]
const MIN_QUBITS_FOR_DIAG_BATCH: usize = 16;
#[cfg(miri)]
const MIN_QUBITS_FOR_DIAG_BATCH: usize = 8;

/// Floor for batching the 1q runs left after `fuse_controlled_phases` (the H gates in
/// QFT). At 16q the 1 MB state sits in L3 and tiling costs more than it saves.
#[cfg(not(miri))]
const MIN_QUBITS_FOR_POST_PHASE_BATCH: usize = 18;
#[cfg(miri)]
const MIN_QUBITS_FOR_POST_PHASE_BATCH: usize = 8;

/// Floor for absorbing 1q gates into a 2q gate. The generic 4×4 kernel does ~4x the
/// FLOPs of CX/CZ plus SIMD 1q kernels; the saved memory passes win from 12q on QV and
/// random sweeps.
#[cfg(not(miri))]
const MIN_QUBITS_FOR_2Q_FUSION: usize = 12;
#[cfg(miri)]
const MIN_QUBITS_FOR_2Q_FUSION: usize = 8;

/// A/B kill switch: setting `PRISM_NO_REORDER` disables `reorder_fused2q_into_tiles`. Read
/// once per process.
#[inline]
fn reorder_2q_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PRISM_NO_REORDER").is_none())
}

/// Equal to the 2q floor because `Multi2q` batches form only from `Fused2q` gates.
const MIN_QUBITS_FOR_MULTI_2Q_FUSION: usize = MIN_QUBITS_FOR_2Q_FUSION;

const MIN_MULTI_2Q_BATCH: usize = 2;

#[inline]
pub(super) fn push_unique(qubits: &mut SmallVec<[usize; 4]>, q: usize) {
    if !qubits.contains(&q) {
        qubits.push(q);
    }
}

fn inst_qubits(inst: &Instruction) -> &[usize] {
    match inst {
        Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => targets,
        Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
            std::slice::from_ref(qubit)
        }
        Instruction::Barrier { qubits } | Instruction::Save { qubits, .. } => qubits,
        Instruction::Region(region) => region.qubits(),
    }
}

/// Clear the pending entry for qubit `q` and its partner.
fn clear_pending(q: usize, instructions: &[Instruction], pending: &mut [Option<usize>]) {
    if let Some(pi) = pending[q] {
        if let Instruction::Gate { targets, .. } = &instructions[pi] {
            for &t in targets.iter() {
                pending[t] = None;
            }
        }
    }
}

/// True if two instructions are the same self-inverse 2q gate on the same targets. CX
/// needs matching target order; CZ and SWAP are symmetric and match either way.
fn is_cancelling_pair(a: &Instruction, b: &Instruction) -> bool {
    match (a, b) {
        (
            Instruction::Gate {
                gate: ga,
                targets: ta,
            },
            Instruction::Gate {
                gate: gb,
                targets: tb,
            },
        ) => {
            if !ga.is_self_inverse_2q() || std::mem::discriminant(ga) != std::mem::discriminant(gb)
            {
                return false;
            }
            if ta.as_slice() == tb.as_slice() {
                return true;
            }
            matches!(ga, Gate::Cz | Gate::Swap)
                && ta.len() == 2
                && tb.len() == 2
                && ta[0] == tb[1]
                && ta[1] == tb[0]
        }
        _ => false,
    }
}

/// Cancel pairs of self-inverse two-qubit gates (CX·CX, CZ·CZ, SWAP·SWAP) with no
/// intervening instruction on either qubit. Returns the input borrowed when none cancel.
pub(crate) fn cancel_self_inverse_pairs<'a>(
    circuit: &'a Circuit,
    t: &mut Tracer,
) -> Cow<'a, Circuit> {
    let has_candidates = circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate { gate, .. } if gate.is_self_inverse_2q()
        )
    });
    if !has_candidates {
        return Cow::Borrowed(circuit);
    }

    let n = circuit.num_qubits;
    let len = circuit.instructions.len();
    let mut cancelled = vec![false; len];
    let mut any_cancelled = false;

    let mut pending: Vec<Option<usize>> = vec![None; n];

    for i in 0..len {
        let inst = &circuit.instructions[i];
        match inst {
            Instruction::Gate { gate, targets } if gate.is_self_inverse_2q() => {
                let (q0, q1) = (targets[0], targets[1]);

                let found = pending[q0]
                    .filter(|&pi| is_cancelling_pair(&circuit.instructions[pi], inst))
                    .or_else(|| {
                        pending[q1]
                            .filter(|&pi| is_cancelling_pair(&circuit.instructions[pi], inst))
                    });

                if let Some(pi) = found {
                    cancelled[pi] = true;
                    cancelled[i] = true;
                    any_cancelled = true;
                    clear_pending(q0, &circuit.instructions, &mut pending);
                    clear_pending(q1, &circuit.instructions, &mut pending);
                } else {
                    clear_pending(q0, &circuit.instructions, &mut pending);
                    clear_pending(q1, &circuit.instructions, &mut pending);
                    pending[q0] = Some(i);
                    pending[q1] = Some(i);
                }
            }
            _ => {
                for &q in inst_qubits(inst) {
                    if q < n {
                        clear_pending(q, &circuit.instructions, &mut pending);
                    }
                }
            }
        }
    }

    if !any_cancelled {
        return Cow::Borrowed(circuit);
    }

    t.begin();
    let mut output = Vec::with_capacity(len);
    for (j, inst) in circuit.instructions.iter().enumerate() {
        if !cancelled[j] {
            output.push(inst.clone());
            t.keep(j);
        }
    }
    t.commit();

    Cow::Owned(circuit.with_instructions(output))
}

pub(super) fn is_identity(mat: &[[Complex64; 2]; 2]) -> bool {
    (mat[0][0].re - 1.0).abs() < IDENTITY_EPS
        && mat[0][0].im.abs() < IDENTITY_EPS
        && mat[0][1].norm() < IDENTITY_EPS
        && mat[1][0].norm() < IDENTITY_EPS
        && (mat[1][1].re - 1.0).abs() < IDENTITY_EPS
        && mat[1][1].im.abs() < IDENTITY_EPS
}

#[inline]
fn gate_1q_matrix(gate: &Gate) -> [[Complex64; 2]; 2] {
    match gate {
        Gate::Fused(m) => **m,
        _ => gate.matrix_2x2(),
    }
}

#[inline]
fn accumulate_1q(slot: &mut Option<[[Complex64; 2]; 2]>, mat: [[Complex64; 2]; 2]) -> bool {
    match slot {
        Some(existing) => {
            *existing = mat_mul_2x2(&mat, existing);
            false
        }
        empty => {
            *empty = Some(mat);
            true
        }
    }
}

#[inline]
fn push_fused_1q(output: &mut Vec<Instruction>, q: usize, mat: [[Complex64; 2]; 2]) {
    output.push(Instruction::Gate {
        gate: Gate::Fused(Box::new(mat)),
        targets: smallvec![q],
    });
}

struct PendingFusion {
    matrix: [[Complex64; 2]; 2],
    target: usize,
    srcs: Vec<(usize, Place)>,
}

fn flush(pending: &mut Option<PendingFusion>, output: &mut Vec<Instruction>, t: &mut Tracer) {
    if let Some(p) = pending.take() {
        if !is_identity(&p.matrix) {
            let gate = match Gate::recognize_matrix(&p.matrix) {
                // A run collapsing to a named gate drops the stored angles, so
                // a rebinding cannot rebuild it.
                Some(named) => {
                    t.bail();
                    named
                }
                None => Gate::Fused(Box::new(p.matrix)),
            };
            output.push(Instruction::Gate {
                gate,
                targets: smallvec![p.target],
            });
            t.guard_1q(&p.srcs, is_diagonal_2x2(&p.matrix));
            t.merge(&p.srcs);
        } else {
            t.bail();
        }
    }
}

/// Fuse consecutive single-qubit gates on the same target into one `Gate::Fused`.
///
/// Gates on other qubits do not break a run. A product that matches a named gate is
/// emitted as that gate, and an identity product is dropped. Returns the input borrowed
/// when no two 1q gates fuse.
pub(crate) fn fuse_single_qubit_gates<'a>(
    circuit: &'a Circuit,
    t: &mut Tracer,
) -> Cow<'a, Circuit> {
    let n = circuit.num_qubits;
    let mut pending: Vec<Option<PendingFusion>> = (0..n).map(|_| None).collect();
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut changed = false;
    t.begin();

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = gate.matrix_2x2();
                match &mut pending[q] {
                    Some(p) => {
                        p.matrix = mat_mul_2x2(&mat, &p.matrix);
                        t.note(&mut p.srcs, i, Place::Plain);
                        changed = true;
                    }
                    slot => {
                        *slot = Some(PendingFusion {
                            matrix: mat,
                            target: q,
                            srcs: t.seed(i, Place::Plain),
                        });
                    }
                }
            }
            _ => {
                for &q in inst_qubits(inst) {
                    flush(&mut pending[q], &mut output, t);
                }
                output.push(inst.clone());
                t.keep(i);
            }
        }
    }

    for slot in &mut pending {
        flush(slot, &mut output, t);
    }

    if changed {
        t.commit();
        Cow::Owned(circuit.with_instructions(output))
    } else {
        t.discard();
        Cow::Borrowed(circuit)
    }
}

/// Reorder single-qubit gates as early as possible in the instruction stream.
///
/// Each 1q gate moves backward past instructions on other qubits, and a diagonal 1q gate
/// also commutes past CX on its control and CZ or Rzz on either qubit. Grouping the 1q
/// gates gives `fuse_multi_1q_gates` longer runs. Returns the input when no gate moves.
pub(crate) fn reorder_1q_gates<'a>(circuit: Cow<'a, Circuit>, t: &mut Tracer) -> Cow<'a, Circuit> {
    let n = circuit.num_qubits;
    // block_all[q] / block_diag[q]: index into non_1q of the last blocker
    let mut block_all: Vec<usize> = vec![usize::MAX; n];
    let mut block_diag: Vec<usize> = vec![usize::MAX; n];
    let mut last_1q_slot: Vec<usize> = vec![0; n];
    let mut non_1q: Vec<&Instruction> = Vec::new();
    let mut non_1q_idx: Vec<usize> = Vec::new();
    let mut slots: Vec<Vec<Instruction>> = vec![Vec::new()];
    let mut slot_idx: Vec<Vec<usize>> = vec![Vec::new()];
    let mut changed = false;

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let blocker = if gate.is_diagonal_1q() {
                    block_diag[q]
                } else {
                    block_all[q]
                };
                let dep_slot = if blocker == usize::MAX {
                    0
                } else {
                    blocker + 1
                };
                let slot = dep_slot.max(last_1q_slot[q]);
                if slot < non_1q.len() {
                    changed = true;
                }
                slots[slot].push(inst.clone());
                if t.on {
                    slot_idx[slot].push(i);
                }
                last_1q_slot[q] = slot;
            }
            _ => {
                let idx = non_1q.len();
                non_1q.push(inst);
                slots.push(Vec::new());
                if t.on {
                    non_1q_idx.push(i);
                    slot_idx.push(Vec::new());
                }
                match inst {
                    Instruction::Gate { gate, targets } => match gate {
                        Gate::Cx => {
                            block_all[targets[0]] = idx;
                            // block_diag[targets[0]] unchanged, diagonal commutes on control
                            block_all[targets[1]] = idx;
                            block_diag[targets[1]] = idx;
                        }
                        Gate::Cz | Gate::Rzz(_) => {
                            block_all[targets[0]] = idx;
                            block_all[targets[1]] = idx;
                            // block_diag unchanged for both, diagonal commutes on both
                        }
                        Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => {
                            for &q in targets.iter() {
                                block_all[q] = idx;
                            }
                            // block_diag unchanged, all-diagonal gate
                        }
                        _ => {
                            for &q in targets.iter() {
                                block_all[q] = idx;
                                block_diag[q] = idx;
                            }
                        }
                    },
                    _ => {
                        for &q in inst_qubits(inst) {
                            block_all[q] = idx;
                            block_diag[q] = idx;
                        }
                    }
                }
            }
        }
    }

    if !changed {
        return circuit;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    t.begin();
    for (i, non_1q_inst) in non_1q.iter().enumerate() {
        output.append(&mut slots[i]);
        if t.on {
            for &src in &slot_idx[i] {
                t.keep(src);
            }
        }
        output.push((*non_1q_inst).clone());
        if t.on {
            t.keep(non_1q_idx[i]);
        }
    }
    output.append(&mut slots[non_1q.len()]);
    if t.on {
        for &src in &slot_idx[non_1q.len()] {
            t.keep(src);
        }
    }
    t.commit();

    Cow::Owned(circuit.with_instructions(output))
}

/// Fuse single-qubit gates on distinct qubits into `Gate::MultiFused`.
///
/// A non-1q instruction flushes the pending 1q gates on its own qubits only; the rest keep
/// accumulating, which is sound because a 1q gate on q commutes with any gate not on q.
pub(crate) fn fuse_multi_1q_gates<'a>(
    circuit: Cow<'a, Circuit>,
    t: &mut Tracer,
) -> Cow<'a, Circuit> {
    if !has_multi_1q_run(&circuit) {
        return circuit;
    }

    let n = circuit.num_qubits;
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut pending: Vec<Option<[[Complex64; 2]; 2]>> = vec![None; n];
    let mut srcs: Vec<Vec<(usize, Place)>> = vec![Vec::new(); n];
    let mut pending_count = 0usize;
    t.begin();

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = gate_1q_matrix(gate);
                if accumulate_1q(&mut pending[q], mat) {
                    pending_count += 1;
                }
                t.note(&mut srcs[q], i, Place::Plain);
            }
            _ => {
                for &q in inst_qubits(inst) {
                    flush_1q_pending(
                        q,
                        &mut pending,
                        &mut pending_count,
                        &mut output,
                        &mut srcs,
                        t,
                    );
                }
                output.push(inst.clone());
                t.keep(i);
            }
        }
    }
    flush_all_pending(&mut pending, &mut pending_count, &mut output, &mut srcs, t);
    t.commit();

    Cow::Owned(circuit.with_instructions(output))
}

#[inline]
fn flush_indexed_1q(
    q: usize,
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    output: &mut Vec<Instruction>,
    srcs: &mut [Vec<(usize, Place)>],
    t: &mut Tracer,
) {
    if let Some(mat) = pending[q].take() {
        push_fused_1q(output, q, mat);
        t.merge(&std::mem::take(&mut srcs[q]));
    }
}

fn flush_1q_pending(
    q: usize,
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    pending_count: &mut usize,
    output: &mut Vec<Instruction>,
    srcs: &mut [Vec<(usize, Place)>],
    t: &mut Tracer,
) {
    if pending[q].is_some() {
        *pending_count -= 1;
        flush_indexed_1q(q, pending, output, srcs, t);
    }
}

fn flush_all_pending(
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    pending_count: &mut usize,
    output: &mut Vec<Instruction>,
    srcs: &mut [Vec<(usize, Place)>],
    t: &mut Tracer,
) {
    if *pending_count >= 2 {
        let mut gates: Vec<(usize, [[Complex64; 2]; 2])> = Vec::with_capacity(*pending_count);
        let mut entries: Vec<Vec<(usize, Place)>> = Vec::new();
        for (q, slot) in pending.iter_mut().enumerate() {
            if let Some(mat) = slot.take() {
                gates.push((q, mat));
                if t.on {
                    entries.push(std::mem::take(&mut srcs[q]));
                }
            }
        }
        let targets: SmallVec<[usize; 4]> = gates.iter().map(|&(t, _)| t).collect();
        output.push(Instruction::Gate {
            gate: Gate::MultiFused(Box::new(MultiFusedData::new(gates))),
            targets,
        });
        t.batch(&entries);
    } else {
        for (q, slot) in pending.iter_mut().enumerate() {
            if let Some(mat) = slot.take() {
                push_fused_1q(output, q, mat);
                t.merge(&std::mem::take(&mut srcs[q]));
            }
        }
    }
    *pending_count = 0;
}

fn has_multi_1q_run(circuit: &Circuit) -> bool {
    let mut total_1q = 0usize;
    for inst in &circuit.instructions {
        if let Instruction::Gate { gate, .. } = inst {
            if gate.num_qubits() == 1 {
                total_1q += 1;
                if total_1q >= 2 {
                    return true;
                }
            }
        }
    }
    false
}

/// Absorb pending 1q gates into the following two-qubit gate as a `Gate::Fused2q`.
///
/// Targets CX, CZ, `Fused2q`, and two-qubit `PauliRot`. SWAP and Cu keep their SIMD
/// kernels, and Cu stays unfused so cphase batching still sees it. A greedy forward
/// pass absorbs pre-gates; post-gates of one 2q gate become pre-gates of the next,
/// which captures most HEA-style patterns. A 1q gate with no later 2q gate to join
/// folds back into the last `Fused2q` on its qubit instead, unless that would turn a
/// diagonal block dense. Returns the input when nothing is absorbed.
pub(crate) fn fuse_2q_gates<'a>(circuit: Cow<'a, Circuit>, t: &mut Tracer) -> Cow<'a, Circuit> {
    let identity_2x2 = Gate::Id.matrix_2x2();
    let n = circuit.num_qubits;
    let mut pending_1q: Vec<Option<[[Complex64; 2]; 2]>> = vec![None; n];
    let mut srcs: Vec<Vec<(usize, Place)>> = vec![Vec::new(); n];
    let mut last_2q: Vec<Option<usize>> = vec![None; n];
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut changed = false;
    t.begin();

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = gate_1q_matrix(gate);
                accumulate_1q(&mut pending_1q[q], mat);
                t.note(&mut srcs[q], i, Place::Plain);
            }
            Instruction::Gate {
                gate: gate @ (Gate::Cx | Gate::Cz | Gate::Fused2q(_) | Gate::PauliRot(_)),
                targets,
            } if gate.num_qubits() == 2 => {
                let q0 = targets[0];
                let q1 = targets[1];
                let pre0 = pending_1q[q0].take();
                let pre1 = pending_1q[q1].take();

                if pre0.is_none() && pre1.is_none() {
                    let at = matches!(gate, Gate::Fused2q(_)).then_some(output.len());
                    last_2q[q0] = at;
                    last_2q[q1] = at;
                    output.push(inst.clone());
                    t.keep(i);
                } else {
                    let m0 = pre0.unwrap_or(identity_2x2);
                    let m1 = pre1.unwrap_or(identity_2x2);
                    let kron = kron_2x2(&m0, &m1);
                    let gate4 = gate.matrix_4x4();
                    let fused = mat_mul_4x4(&gate4, &kron);
                    last_2q[q0] = Some(output.len());
                    last_2q[q1] = Some(output.len());
                    output.push(Instruction::Gate {
                        gate: Gate::Fused2q(Box::new(fused)),
                        targets: smallvec![q0, q1],
                    });
                    if t.on {
                        let mut steps: Vec<(usize, Place)> = Vec::new();
                        for (src, _) in std::mem::take(&mut srcs[q0]) {
                            steps.push((src, Place::Low));
                        }
                        for (src, _) in std::mem::take(&mut srcs[q1]) {
                            steps.push((src, Place::High));
                        }
                        steps.push((i, Place::Plain));
                        t.merge(&steps);
                    }
                    changed = true;
                }
            }
            _ => {
                for &q in inst_qubits(inst) {
                    changed |=
                        settle_trailing_1q(q, &mut pending_1q, &last_2q, &mut output, &mut srcs, t);
                    last_2q[q] = None;
                }
                output.push(inst.clone());
                t.keep(i);
            }
        }
    }

    for q in 0..n {
        changed |= settle_trailing_1q(q, &mut pending_1q, &last_2q, &mut output, &mut srcs, t);
    }

    if changed {
        t.commit();
        Cow::Owned(circuit.with_instructions(output))
    } else {
        t.discard();
        circuit
    }
}

/// Emit the 1q run pending on `q`, folding it into the last `Fused2q` on `q` when that
/// gate is the most recent output touching `q`. A diagonal block keeps a non-diagonal
/// run out so the diagonal batch kernels still see it. Returns whether it folded.
fn settle_trailing_1q(
    q: usize,
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    last_2q: &[Option<usize>],
    output: &mut Vec<Instruction>,
    srcs: &mut [Vec<(usize, Place)>],
    t: &mut Tracer,
) -> bool {
    let (Some(u), Some(at)) = (pending[q], last_2q[q]) else {
        flush_indexed_1q(q, pending, output, srcs, t);
        return false;
    };
    let Instruction::Gate {
        gate: Gate::Fused2q(m),
        targets,
    } = &mut output[at]
    else {
        unreachable!("last_2q indexes a Fused2q");
    };
    let m_diagonal = is_diagonal_4x4(m);
    let u_diagonal = is_diagonal_2x2(&u);
    let place = if targets[0] == q {
        Place::Low
    } else {
        Place::High
    };
    let fold = !m_diagonal || u_diagonal;
    if fold {
        **m = mat_mul_4x4(&embed_1q_matrix(&u, q, targets[0]), m);
    }
    let placed: Vec<(usize, Place)> = if t.on {
        srcs[q].iter().map(|&(src, _)| (src, place)).collect()
    } else {
        Vec::new()
    };
    t.guard_output_diag_4x4(at, m_diagonal);
    if m_diagonal {
        t.guard_diag_4x4(&placed, u_diagonal);
    }
    if !fold {
        flush_indexed_1q(q, pending, output, srcs, t);
        return false;
    }
    t.extend(at, &placed);
    pending[q] = None;
    srcs[q].clear();
    true
}

#[inline]
fn swap_order_4x4(mat: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let swap = Gate::Swap.matrix_4x4();
    mat_mul_4x4(&swap, &mat_mul_4x4(mat, &swap))
}

#[inline]
fn same_unordered_pair(a0: usize, a1: usize, b0: usize, b1: usize) -> bool {
    (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0)
}

#[inline]
fn orient_2q_matrix(
    mat: &[[Complex64; 4]; 4],
    targets: &[usize],
    q0: usize,
    q1: usize,
) -> [[Complex64; 4]; 4] {
    if targets[0] == q0 && targets[1] == q1 {
        *mat
    } else {
        swap_order_4x4(mat)
    }
}

#[inline]
fn embed_1q_matrix(mat: &[[Complex64; 2]; 2], target: usize, q0: usize) -> [[Complex64; 4]; 4] {
    let id = Gate::Id.matrix_2x2();
    if target == q0 {
        kron_2x2(mat, &id)
    } else {
        kron_2x2(&id, mat)
    }
}

struct PairRun {
    q0: usize,
    q1: usize,
    acc: [[Complex64; 4]; 4],
    fused_2q_count: usize,
    has_nondiagonal_2q: bool,
    originals: Vec<Instruction>,
    srcs: Vec<(usize, Place)>,
    src_idx: Vec<usize>,
}

impl PairRun {
    fn new(
        q0: usize,
        q1: usize,
        mat: [[Complex64; 4]; 4],
        original: Instruction,
        index: usize,
        t: &mut Tracer,
    ) -> Self {
        t.guard_diag_4x4(&[(index, Place::Plain)], is_diagonal_4x4(&mat));
        Self {
            q0,
            q1,
            acc: mat,
            fused_2q_count: 1,
            has_nondiagonal_2q: !is_diagonal_4x4(&mat),
            originals: vec![original],
            srcs: t.seed(index, Place::Plain),
            src_idx: t.seed_idx(index),
        }
    }

    #[inline]
    fn can_accept_pair(&self, q0: usize, q1: usize) -> bool {
        same_unordered_pair(self.q0, self.q1, q0, q1)
    }

    #[inline]
    fn can_accept_1q(&self, q: usize) -> bool {
        q == self.q0 || q == self.q1
    }

    fn push_2q(
        &mut self,
        mat: [[Complex64; 4]; 4],
        targets: &[usize],
        original: Instruction,
        index: usize,
        t: &mut Tracer,
    ) {
        let oriented = orient_2q_matrix(&mat, targets, self.q0, self.q1);
        let place = if targets[0] == self.q0 && targets[1] == self.q1 {
            Place::Plain
        } else {
            Place::Swapped
        };
        t.guard_diag_4x4(&[(index, place)], is_diagonal_4x4(&oriented));
        self.acc = mat_mul_4x4(&oriented, &self.acc);
        self.fused_2q_count += 1;
        self.has_nondiagonal_2q |= !is_diagonal_4x4(&oriented);
        self.originals.push(original);
        t.note(&mut self.srcs, index, place);
        t.note_idx(&mut self.src_idx, index);
    }

    fn push_1q(
        &mut self,
        mat: [[Complex64; 2]; 2],
        target: usize,
        original: Instruction,
        index: usize,
        t: &mut Tracer,
    ) {
        let embedded = embed_1q_matrix(&mat, target, self.q0);
        let place = if target == self.q0 {
            Place::Low
        } else {
            Place::High
        };
        self.acc = mat_mul_4x4(&embedded, &self.acc);
        self.originals.push(original);
        t.note(&mut self.srcs, index, place);
        t.note_idx(&mut self.src_idx, index);
    }

    fn should_fuse(&self) -> bool {
        self.fused_2q_count >= 2 && self.has_nondiagonal_2q
    }
}

fn flush_pair_run(
    run: &mut Option<PairRun>,
    output: &mut Vec<Instruction>,
    changed: &mut bool,
    t: &mut Tracer,
) {
    let Some(run) = run.take() else {
        return;
    };
    if run.should_fuse() {
        output.push(Instruction::Gate {
            gate: Gate::Fused2q(Box::new(run.acc)),
            targets: smallvec![run.q0, run.q1],
        });
        t.merge(&run.srcs);
        *changed = true;
    } else {
        output.extend(run.originals);
        for src in run.src_idx {
            t.keep(src);
        }
    }
}

/// Fuse contiguous same-pair `Fused2q` runs into one larger `Fused2q`.
///
/// Consumes only `Fused2q` gates and 1q gates on the same pair, and fuses only runs
/// holding at least two 2q units. All-diagonal runs are left for the cheaper diagonal
/// batch kernels.
fn fuse_same_pair_2q_blocks<'a>(input: Cow<'a, Circuit>, t: &mut Tracer) -> Cow<'a, Circuit> {
    let circuit = input.as_ref();
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut run: Option<PairRun> = None;
    let mut changed = false;
    t.begin();

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate {
                gate: Gate::Fused2q(mat),
                targets,
            } => {
                if let Some(active) = &mut run {
                    if active.can_accept_pair(targets[0], targets[1]) {
                        active.push_2q(**mat, targets, inst.clone(), i, t);
                    } else {
                        flush_pair_run(&mut run, &mut output, &mut changed, t);
                        run = Some(PairRun::new(
                            targets[0],
                            targets[1],
                            **mat,
                            inst.clone(),
                            i,
                            t,
                        ));
                    }
                } else {
                    run = Some(PairRun::new(
                        targets[0],
                        targets[1],
                        **mat,
                        inst.clone(),
                        i,
                        t,
                    ));
                }
            }
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                if let Some(active) = &mut run {
                    if active.can_accept_1q(targets[0]) {
                        let mat = gate_1q_matrix(gate);
                        active.push_1q(mat, targets[0], inst.clone(), i, t);
                    } else {
                        flush_pair_run(&mut run, &mut output, &mut changed, t);
                        output.push(inst.clone());
                        t.keep(i);
                    }
                } else {
                    output.push(inst.clone());
                    t.keep(i);
                }
            }
            _ => {
                flush_pair_run(&mut run, &mut output, &mut changed, t);
                output.push(inst.clone());
                t.keep(i);
            }
        }
    }

    flush_pair_run(&mut run, &mut output, &mut changed, t);

    if changed {
        t.commit();
        Cow::Owned(circuit.with_instructions(output))
    } else {
        t.discard();
        input
    }
}

/// A 2q gate a `Multi2q` tile can carry as its dense 4x4. Diagonal gates stay out, since
/// the diagonal batch passes serve them without a dense multiply.
fn is_tileable_2q(inst: &Instruction) -> bool {
    matches!(
        inst,
        Instruction::Gate {
            gate: Gate::Fused2q(_) | Gate::Cx | Gate::Swap,
            targets,
        } if targets.len() == 2
    )
}

/// Reorder each run of consecutive tileable 2q gates so that gates sharing one subcube
/// tile sit next to each other, ready for `fuse_multi_2q_gates` to batch.
///
/// Each tile is filled by one scan of the run in order: a gate joins while it fits
/// the tile and no gate skipped earlier in the scan shares a qubit with it, so every
/// move commutes a gate past disjoint ones only. Quantum volume at 24 qubits takes 31
/// passes over the state, against 54 when each layer is packed on its own. Returns the
/// input when nothing moves.
pub(crate) fn reorder_fused2q_into_tiles<'a>(
    input: Cow<'a, Circuit>,
    t: &mut Tracer,
) -> Cow<'a, Circuit> {
    let circuit = input.as_ref();
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut window: Vec<(Instruction, usize)> = Vec::new();
    let mut blocked = vec![false; circuit.num_qubits];
    let mut changed = false;
    t.begin();

    for (i, inst) in circuit.instructions.iter().enumerate() {
        if is_tileable_2q(inst) {
            window.push((inst.clone(), i));
        } else {
            flush_tile_window(&mut window, &mut blocked, &mut output, &mut changed, t);
            output.push(inst.clone());
            t.keep(i);
        }
    }
    flush_tile_window(&mut window, &mut blocked, &mut output, &mut changed, t);

    if changed {
        t.commit();
        Cow::Owned(circuit.with_instructions(output))
    } else {
        t.discard();
        input
    }
}

/// Emit a run of `Fused2q` gates one tile at a time. A scan stops looking once it
/// has skipped twice as many gates as there are qubits, which keeps long runs
/// linear and lost no pass on quantum volume, HEA or random circuits.
fn flush_tile_window(
    window: &mut Vec<(Instruction, usize)>,
    blocked: &mut [bool],
    output: &mut Vec<Instruction>,
    changed: &mut bool,
    t: &mut Tracer,
) {
    let pair = |k: usize| {
        let Instruction::Gate { targets, .. } = &window[k].0 else {
            unreachable!("the window holds tileable 2q gates only");
        };
        (targets[0], targets[1])
    };
    let skip_limit = 2 * blocked.len();
    let mut rest: Vec<usize> = (0..window.len()).collect();
    let mut order: Vec<usize> = Vec::with_capacity(window.len());
    let mut left: Vec<usize> = Vec::new();
    while !rest.is_empty() {
        let mut high: SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]> = SmallVec::new();
        for (pos, &k) in rest.iter().enumerate() {
            if left.len() >= skip_limit {
                left.extend_from_slice(&rest[pos..]);
                break;
            }
            let (q0, q1) = pair(k);
            let joined = if blocked[q0] || blocked[q1] {
                None
            } else {
                multi_2q_join(&high, q0, q1)
            };
            match joined {
                Some(joined) => {
                    high = joined;
                    order.push(k);
                }
                None => {
                    blocked[q0] = true;
                    blocked[q1] = true;
                    left.push(k);
                }
            }
        }
        for &k in &left {
            let (q0, q1) = pair(k);
            blocked[q0] = false;
            blocked[q1] = false;
        }
        std::mem::swap(&mut rest, &mut left);
        left.clear();
    }
    if order.iter().enumerate().any(|(pos, &k)| pos != k) {
        *changed = true;
    }
    let mut taken: Vec<Option<(Instruction, usize)>> = window.drain(..).map(Some).collect();
    for k in order {
        let (inst, src) = taken[k].take().expect("each window slot is emitted once");
        output.push(inst);
        t.keep(src);
    }
}

/// Batch consecutive tileable 2q gates (`Fused2q`, `Cx`, `Swap`) into `Multi2q` for
/// cache-tiled execution.
///
/// A run of consecutive tileable gates grows while its gates fit one
/// subcube tile: at most [`MULTI_2Q_HIGH_BUDGET`] distinct qubits at or above
/// the tile's low bits. Each run of two or more gates becomes one `Multi2q`
/// that the statevector backend applies in one pass over the state. A run of one keeps
/// its original instruction, so a lone `Cx` stays on its permutation kernel.
///
/// Returns the input unchanged when no batch forms.
pub(crate) fn fuse_multi_2q_gates<'a>(
    circuit: Cow<'a, Circuit>,
    tracer: &mut Tracer,
) -> Cow<'a, Circuit> {
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut pending: Vec<(usize, usize, [[Complex64; 4]; 4])> = Vec::new();
    let mut pending_src: Vec<usize> = Vec::new();
    let mut high: SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]> = SmallVec::new();
    let mut changed = false;
    let source = &circuit.instructions;

    let flush = |pending: &mut Vec<(usize, usize, [[Complex64; 4]; 4])>,
                 pending_src: &mut Vec<usize>,
                 high: &mut SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]>,
                 output: &mut Vec<Instruction>,
                 changed: &mut bool,
                 tracer: &mut Tracer| {
        high.clear();
        if pending.is_empty() {
            return;
        }
        if pending.len() < MIN_MULTI_2Q_BATCH {
            pending.clear();
            for src in pending_src.drain(..) {
                output.push(source[src].clone());
                tracer.keep(src);
            }
        } else {
            let mut all_qubits: SmallVec<[usize; 4]> = SmallVec::new();
            for &(q0, q1, _) in pending.iter() {
                push_unique(&mut all_qubits, q0);
                push_unique(&mut all_qubits, q1);
            }
            all_qubits.sort_unstable();
            output.push(Instruction::Gate {
                gate: Gate::Multi2q(Box::new(Multi2qData {
                    gates: std::mem::take(pending),
                })),
                targets: all_qubits,
            });
            if tracer.on {
                let entries: Vec<Vec<(usize, Place)>> = pending_src
                    .iter()
                    .map(|&src| vec![(src, Place::Plain)])
                    .collect();
                tracer.batch(&entries);
            }
            pending_src.clear();
            *changed = true;
        }
    };

    tracer.begin();
    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if is_tileable_2q(inst) => {
                let q0 = targets[0];
                let q1 = targets[1];
                let joined = match multi_2q_join(&high, q0, q1) {
                    Some(joined) => joined,
                    None => {
                        flush(
                            &mut pending,
                            &mut pending_src,
                            &mut high,
                            &mut output,
                            &mut changed,
                            tracer,
                        );
                        multi_2q_join(&[], q0, q1).expect("one pair fits a tile")
                    }
                };
                high = joined;
                pending.push((q0, q1, gate.matrix_4x4()));
                pending_src.push(i);
            }
            _ => {
                flush(
                    &mut pending,
                    &mut pending_src,
                    &mut high,
                    &mut output,
                    &mut changed,
                    tracer,
                );
                output.push(inst.clone());
                tracer.keep(i);
            }
        }
    }
    flush(
        &mut pending,
        &mut pending_src,
        &mut high,
        &mut output,
        &mut changed,
        tracer,
    );

    if changed {
        tracer.commit();
        Cow::Owned(circuit.with_instructions(output))
    } else {
        tracer.discard();
        circuit
    }
}

/// Batch contiguous runs of diagonal gates into `DiagonalBatch` instructions.
///
/// Diagonal gates (Z, S, T, Rz, P, CZ, Rzz, CPhase) commute, so a run collapses into one
/// LUT pass. Non-diagonal 1q gates on qubits outside the run are deferred past it.
fn fuse_diagonal_batch<'a>(input: Cow<'a, Circuit>, t: &mut Tracer) -> Cow<'a, Circuit> {
    let circuit = input.as_ref();
    let insts = &circuit.instructions;
    let n = insts.len();
    if n < 2 {
        return input;
    }

    let diag_count = insts
        .iter()
        .filter(|i| matches!(i, Instruction::Gate { gate, .. } if gate.is_diag_batchable()))
        .count();
    if diag_count < 2 {
        return input;
    }
    t.bail();

    let mut output: Vec<Instruction> = Vec::with_capacity(n);
    let mut run_entries: Vec<DiagEntry> = Vec::new();
    let mut run_originals: Vec<Instruction> = Vec::new();
    let mut run_qubits = vec![false; circuit.num_qubits];
    let mut deferred: Vec<Instruction> = Vec::new();
    let mut deferred_qubits = vec![false; circuit.num_qubits];

    let flush_diag_run = |output: &mut Vec<Instruction>,
                          entries: &mut Vec<DiagEntry>,
                          originals: &mut Vec<Instruction>,
                          deferred: &mut Vec<Instruction>,
                          run_qubits: &mut [bool],
                          deferred_qubits: &mut [bool]| {
        if entries.len() >= 2 {
            let mut tgts: SmallVec<[usize; 4]> = SmallVec::new();
            for (i, &used) in run_qubits.iter().enumerate() {
                if used {
                    tgts.push(i);
                }
            }
            output.push(Instruction::Gate {
                gate: Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                    entries: std::mem::take(entries),
                })),
                targets: tgts,
            });
        } else {
            output.append(originals);
        }
        entries.clear();
        originals.clear();
        output.append(deferred);
        run_qubits.fill(false);
        deferred_qubits.fill(false);
    };

    for inst in insts {
        if let Instruction::Gate { gate, targets } = inst {
            if gate.is_diag_batchable() {
                // Deferred gates are re-emitted after the whole batch. Admitting
                // a diagonal gate on a deferred gate's qubit would sink that
                // gate behind one it does not commute with, so close the run
                // first and let this gate open a new one.
                if targets.iter().any(|t| deferred_qubits[*t]) {
                    flush_diag_run(
                        &mut output,
                        &mut run_entries,
                        &mut run_originals,
                        &mut deferred,
                        &mut run_qubits,
                        &mut deferred_qubits,
                    );
                }
                let new_entries = gate.diag_entries(targets);
                for t in targets.iter() {
                    run_qubits[*t] = true;
                }
                run_entries.extend(new_entries);
                run_originals.push(inst.clone());
                continue;
            }

            if !run_entries.is_empty() && gate.num_qubits() == 1 && !run_qubits[targets[0]] {
                deferred_qubits[targets[0]] = true;
                deferred.push(inst.clone());
                continue;
            }
        }

        flush_diag_run(
            &mut output,
            &mut run_entries,
            &mut run_originals,
            &mut deferred,
            &mut run_qubits,
            &mut deferred_qubits,
        );
        output.push(inst.clone());
    }

    flush_diag_run(
        &mut output,
        &mut run_entries,
        &mut run_originals,
        &mut deferred,
        &mut run_qubits,
        &mut deferred_qubits,
    );

    let mut c = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
    c.instructions = output;
    Cow::Owned(c)
}

/// Threads a `&Circuit -> Cow<Circuit>` pass over a `Cow<Circuit>` while
/// preserving zero-copy when both input and output are borrowed.
#[inline]
fn apply_pass<'a, F>(input: Cow<'a, Circuit>, t: &mut Tracer, pass: F) -> Cow<'a, Circuit>
where
    F: for<'b> Fn(&'b Circuit, &mut Tracer) -> Cow<'b, Circuit>,
{
    match input {
        Cow::Borrowed(c) => pass(c, t),
        Cow::Owned(c) => Cow::Owned(pass(&c, t).into_owned()),
    }
}

#[inline]
fn gated<'a, F>(
    input: Cow<'a, Circuit>,
    num_qubits: usize,
    threshold: usize,
    pass: F,
) -> Cow<'a, Circuit>
where
    F: FnOnce(Cow<'a, Circuit>) -> Cow<'a, Circuit>,
{
    if num_qubits >= threshold {
        pass(input)
    } else {
        input
    }
}

/// Fuse each guarded region's body in isolation.
///
/// Every other pass treats a region as one opaque instruction, so a body would otherwise
/// run unfused. Nothing moves across the boundary, and the recursive call handles nesting.
/// The tracer is untouched: regions carry no provenance, so the input mapping still
/// describes the output.
fn fuse_region_bodies<'a>(circuit: &'a Circuit, n: usize) -> Cow<'a, Circuit> {
    let insts = &circuit.instructions;
    let mut out: Option<Vec<Instruction>> = None;

    for (i, inst) in insts.iter().enumerate() {
        let fused_region = match inst {
            Instruction::Region(region) => {
                let body = circuit.with_instructions(region.body().to_vec());
                match fuse_at_width(&body, n, &mut Tracer::off()) {
                    Cow::Owned(fused) => Some(Instruction::Region(Box::new(GuardedRegion::new(
                        region.condition().clone(),
                        fused.instructions,
                    )))),
                    Cow::Borrowed(_) => None,
                }
            }
            _ => None,
        };

        match fused_region {
            Some(region) => out.get_or_insert_with(|| insts[..i].to_vec()).push(region),
            None => {
                if let Some(buf) = out.as_mut() {
                    buf.push(inst.clone());
                }
            }
        }
    }

    match out {
        Some(buf) => Cow::Owned(circuit.with_instructions(buf)),
        None => Cow::Borrowed(circuit),
    }
}

/// Run the fusion pipeline with every floor gated at the circuit's own width.
///
/// Returns the input borrowed when nothing fuses. Pass `supports_fused = false` for
/// backends that cannot run fused gates (the stabilizer). A backend whose buffer is wider
/// than a `num_qubits` statevector takes [`fuse_circuit_for_width`] instead.
pub fn fuse_circuit<'a>(circuit: &'a Circuit, supports_fused: bool) -> Cow<'a, Circuit> {
    fuse_circuit_for_width(circuit, supports_fused, circuit.num_qubits)
}

/// Fuse for a backend that sweeps a `state_qubits`-wide buffer.
///
/// The floors are calibrated against one statevector pass, so they gate on buffer width.
/// A density matrix holds an `n`-qubit mixture as a `2n`-qubit statevector and reaches
/// each floor at half the circuit width.
pub fn fuse_circuit_for_width<'a>(
    circuit: &'a Circuit,
    supports_fused: bool,
    state_qubits: usize,
) -> Cow<'a, Circuit> {
    if !supports_fused {
        return Cow::Borrowed(circuit);
    }
    fuse_at_width(circuit, state_qubits, &mut Tracer::off())
}

/// The pass pipeline at the circuit's own width, recording provenance into `t`.
pub(super) fn fuse_traced<'a>(circuit: &'a Circuit, t: &mut Tracer) -> Cow<'a, Circuit> {
    fuse_at_width(circuit, circuit.num_qubits, t)
}

fn fuse_at_width<'a>(circuit: &'a Circuit, n: usize, t: &mut Tracer) -> Cow<'a, Circuit> {
    let pass_r = fuse_region_bodies(circuit, n);
    let pass0 = apply_pass(pass_r, t, cancel_self_inverse_pairs);
    let pass0r = apply_pass(pass0, t, fuse_rzz);
    let pass0b = gated(pass0r, n, MIN_QUBITS_FOR_DIAG_BATCH, |c| {
        apply_pass(c, t, fuse_batch_rzz)
    });

    if n < MIN_QUBITS_FOR_FUSION {
        return pass0b;
    }

    let pass1 = apply_pass(pass0b, t, fuse_single_qubit_gates);
    let pass1r = reorder_1q_gates(pass1, t);
    let pass1c = apply_pass(pass1r, t, cancel_self_inverse_pairs);
    let pass1f = apply_pass(pass1c, t, fuse_single_qubit_gates);

    let pass_2q = gated(pass1f, n, MIN_QUBITS_FOR_2Q_FUSION, |c| fuse_2q_gates(c, t));
    let pass_2qb = gated(pass_2q, n, MIN_QUBITS_FOR_2Q_FUSION, |c| {
        fuse_same_pair_2q_blocks(c, t)
    });
    let pass2 = gated(pass_2qb, n, MIN_QUBITS_FOR_MULTI_FUSION, |c| {
        fuse_multi_1q_gates(c, t)
    });
    let pass_2qr = if n >= MIN_QUBITS_FOR_MULTI_2Q_FUSION && reorder_2q_enabled() {
        reorder_fused2q_into_tiles(pass2, t)
    } else {
        pass2
    };
    let pass_m2q = gated(pass_2qr, n, MIN_QUBITS_FOR_MULTI_2Q_FUSION, |c| {
        fuse_multi_2q_gates(c, t)
    });
    let pass_cp = gated(pass_m2q, n, MIN_QUBITS_FOR_DIAG_BATCH, |c| {
        fuse_controlled_phases(c, t)
    });
    let pass_db = gated(pass_cp, n, MIN_QUBITS_FOR_DIAG_BATCH, |c| {
        fuse_diagonal_batch(c, t)
    });
    gated(pass_db, n, MIN_QUBITS_FOR_POST_PHASE_BATCH, |c| {
        batch_post_phase_1q(c, t)
    })
}

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod tests;
