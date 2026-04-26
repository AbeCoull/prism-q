//! Single-qubit gate fusion and matrix precomputation pass.
//!
//! Scans the instruction stream and fuses consecutive single-qubit gates on the
//! same target qubit into a single `Gate::Fused` carrying the product matrix.
//! Gates on different qubits are transparent — they don't break a pending fusion.
//!
//! When the pass creates a new instruction stream, ALL single-qubit gates are
//! emitted as `Gate::Fused` with precomputed matrices — including isolated gates
//! that have no fusion partner. This avoids redundant `matrix_2x2()` dispatch
//! during simulation. Identity matrices (from inverse cancellation) are elided.
//!
//! # Matrix multiplication order
//!
//! Gates applied in circuit order G1, G2, G3 produce fused matrix M = G3 · G2 · G1.
//! The accumulator multiplies each new gate on the LEFT: `acc = G_new * acc`.
//!
//! # Flush triggers
//!
//! Two-qubit gates, measurements, and barriers flush pending fusions for every
//! qubit they touch before the instruction is emitted.

use std::borrow::Cow;

use num_complex::Complex64;

use super::{smallvec, Circuit, Instruction, SmallVec};
use crate::gates::{
    kron_2x2, mat_mul_2x2, mat_mul_4x4, DiagEntry, DiagonalBatchData, Gate, Multi2qData,
    MultiFusedData,
};

use super::fusion_phase::{batch_post_phase_1q, fuse_controlled_phases};
use super::fusion_rzz::{fuse_batch_rzz, fuse_rzz};

pub(super) const IDENTITY_EPS: f64 = 1e-12;

/// Minimum qubit count for 1q fusion, reorder, and batching passes.
///
/// Below 10 qubits, statevectors are small enough that gate execution is
/// nanoseconds. The instruction-clone cost of fusion passes (allocating output
/// Vec, cloning non-fuseable instructions) exceeds any simulation savings.
const MIN_QUBITS_FOR_FUSION: usize = 10;

/// Minimum qubit count for multi-gate tiled fusion to be profitable.
const MIN_QUBITS_FOR_MULTI_FUSION: usize = 14;

/// Minimum qubit count for diagonal-family batch passes (BatchRzz, BatchPhase,
/// DiagonalBatch) to be profitable. LUT kernel overhead needs enough state size
/// to amortize.
const MIN_QUBITS_FOR_DIAG_BATCH: usize = 16;

/// Minimum qubit count for post-phase-fusion 1q batching.
///
/// After `fuse_controlled_phases`, consecutive 1q gates (H gates in QFT) are
/// batched into MultiFused for tiled execution. At 16q (1MB, L3-resident),
/// tiling overhead exceeds savings. Profitable at 18q+ where DRAM bandwidth
/// dominates.
const MIN_QUBITS_FOR_POST_PHASE_BATCH: usize = 18;

/// Minimum qubit count for two-qubit gate fusion (absorb 1q into CX/CZ) to be profitable.
///
/// The generic 4×4 kernel does ~4x the FLOPs of specialized CX/CZ + SIMD 1q kernels.
/// Below this threshold, extra compute cost exceeds memory-pass savings.
const MIN_QUBITS_FOR_2Q_FUSION: usize = 20;

/// Minimum qubit count for multi-2q tiled fusion to be profitable.
///
/// Batches consecutive Fused2q gates into a single cache-tiled pass. Only
/// created when Fused2q gates exist (which requires ≥20q), so threshold matches.
const MIN_QUBITS_FOR_MULTI_2Q_FUSION: usize = MIN_QUBITS_FOR_2Q_FUSION;

/// Minimum batch size for multi-2q fusion (single gate not worth wrapping).
const MIN_MULTI_2Q_BATCH: usize = 2;

/// Returns the qubits touched by an instruction.
fn inst_qubits(inst: &Instruction) -> &[usize] {
    match inst {
        Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => targets,
        Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
            std::slice::from_ref(qubit)
        }
        Instruction::Barrier { qubits } => qubits,
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

/// True if two instructions form a cancelling pair (same self-inverse 2q gate, same targets).
///
/// For CX, target order must match exactly (CX(0,1) ≠ CX(1,0)).
/// For CZ and SWAP, either order matches (symmetric gates).
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
            // CZ and SWAP are symmetric — reversed order also cancels
            matches!(ga, Gate::Cz | Gate::Swap)
                && ta.len() == 2
                && tb.len() == 2
                && ta[0] == tb[1]
                && ta[1] == tb[0]
        }
        _ => false,
    }
}

/// Cancel pairs of self-inverse two-qubit gates (CX·CX, CZ·CZ, SWAP·SWAP).
///
/// Tracks pending self-inverse 2q gates per-qubit. When a matching gate appears
/// with no intervening instruction on the same qubits, both are removed.
/// Returns `Cow::Borrowed` when no cancellation opportunities exist.
pub fn cancel_self_inverse_pairs(circuit: &Circuit) -> Cow<'_, Circuit> {
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

    let output = circuit
        .instructions
        .iter()
        .enumerate()
        .filter(|(j, _)| !cancelled[*j])
        .map(|(_, inst)| inst.clone())
        .collect();

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

fn is_identity(mat: &[[Complex64; 2]; 2]) -> bool {
    (mat[0][0].re - 1.0).abs() < IDENTITY_EPS
        && mat[0][0].im.abs() < IDENTITY_EPS
        && mat[0][1].norm() < IDENTITY_EPS
        && mat[1][0].norm() < IDENTITY_EPS
        && (mat[1][1].re - 1.0).abs() < IDENTITY_EPS
        && mat[1][1].im.abs() < IDENTITY_EPS
}

struct PendingFusion {
    matrix: [[Complex64; 2]; 2],
    target: usize,
}

fn flush(pending: &mut Option<PendingFusion>, output: &mut Vec<Instruction>) {
    if let Some(p) = pending.take() {
        if !is_identity(&p.matrix) {
            let gate = match Gate::recognize_matrix(&p.matrix) {
                Some(Gate::Id) => return,
                Some(named) => named,
                None => Gate::Fused(Box::new(p.matrix)),
            };
            output.push(Instruction::Gate {
                gate,
                targets: smallvec![p.target],
            });
        }
    }
}

/// Returns true if the circuit has at least one pair of consecutive single-qubit
/// gates on the same qubit (with no intervening flush trigger on that qubit).
fn has_fusable_gates(circuit: &Circuit) -> bool {
    if circuit.instructions.len() <= 1 {
        return false;
    }
    let mut has_pending = vec![false; circuit.num_qubits];
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                if has_pending[q] {
                    return true;
                }
                has_pending[q] = true;
            }
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
                for &q in targets {
                    has_pending[q] = false;
                }
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                has_pending[*qubit] = false;
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits {
                    has_pending[q] = false;
                }
            }
        }
    }
    false
}

/// Fuse consecutive single-qubit gates on the same target into one `Gate::Fused`.
///
/// Returns a `Cow::Borrowed` reference to the original circuit when no fusion
/// opportunities exist (zero overhead), or a `Cow::Owned` new circuit with
/// fused instructions. The fused circuit produces identical simulation results.
pub fn fuse_single_qubit_gates(circuit: &Circuit) -> Cow<'_, Circuit> {
    if !has_fusable_gates(circuit) {
        return Cow::Borrowed(circuit);
    }

    let n = circuit.num_qubits;
    let mut pending: Vec<Option<PendingFusion>> = (0..n).map(|_| None).collect();
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = gate.matrix_2x2();
                match &mut pending[q] {
                    Some(p) => {
                        p.matrix = mat_mul_2x2(&mat, &p.matrix);
                    }
                    slot => {
                        *slot = Some(PendingFusion {
                            matrix: mat,
                            target: q,
                        });
                    }
                }
            }
            Instruction::Gate { targets, .. } | Instruction::Conditional { targets, .. } => {
                for &q in targets.iter() {
                    flush(&mut pending[q], &mut output);
                }
                output.push(inst.clone());
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                flush(&mut pending[*qubit], &mut output);
                output.push(inst.clone());
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits.iter() {
                    flush(&mut pending[q], &mut output);
                }
                output.push(inst.clone());
            }
        }
    }

    for slot in &mut pending {
        flush(slot, &mut output);
    }

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

/// Returns true if any 1q gate could be moved earlier without violating dependencies.
///
/// Accounts for commutation: diagonal 1q gates on CX control or CZ qubits
/// are not blocked by those gates.
fn has_reorder_opportunity(circuit: &Circuit) -> bool {
    if circuit.instructions.len() <= 1 {
        return false;
    }
    // block_all[q] = position of last instruction that blocks ALL 1q gates on q
    // block_diag[q] = position of last instruction that blocks diagonal 1q gates on q
    // None means "never blocked" — the 1q gate can move to position 0.
    let mut block_all: Vec<Option<usize>> = vec![None; circuit.num_qubits];
    let mut block_diag: Vec<Option<usize>> = vec![None; circuit.num_qubits];
    let mut last_non1q_pos: Option<usize> = None;

    for (i, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                if let Some(pos) = last_non1q_pos {
                    let blocked_at = if gate.is_diagonal_1q() {
                        block_diag[q]
                    } else {
                        block_all[q]
                    };
                    match blocked_at {
                        None => return true,
                        Some(b) if pos > b => return true,
                        _ => {}
                    }
                }
                block_all[q] = Some(i);
                block_diag[q] = Some(i);
            }
            Instruction::Gate { gate, targets } => {
                last_non1q_pos = Some(i);
                match gate {
                    Gate::Cx => {
                        block_all[targets[0]] = Some(i);
                        // block_diag[targets[0]] unchanged — diagonal commutes on control
                        block_all[targets[1]] = Some(i);
                        block_diag[targets[1]] = Some(i);
                    }
                    Gate::Cz | Gate::Rzz(_) => {
                        block_all[targets[0]] = Some(i);
                        block_all[targets[1]] = Some(i);
                        // block_diag unchanged — diagonal commutes on both
                    }
                    Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => {
                        for &q in targets.iter() {
                            block_all[q] = Some(i);
                        }
                        // block_diag unchanged — all-diagonal gate
                    }
                    _ => {
                        for &q in targets.iter() {
                            block_all[q] = Some(i);
                            block_diag[q] = Some(i);
                        }
                    }
                }
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                last_non1q_pos = Some(i);
                block_all[*qubit] = Some(i);
                block_diag[*qubit] = Some(i);
            }
            Instruction::Barrier { qubits } => {
                last_non1q_pos = Some(i);
                for &q in qubits.iter() {
                    block_all[q] = Some(i);
                    block_diag[q] = Some(i);
                }
            }
            Instruction::Conditional { targets, .. } => {
                last_non1q_pos = Some(i);
                for &q in targets.iter() {
                    block_all[q] = Some(i);
                    block_diag[q] = Some(i);
                }
            }
        }
    }
    false
}

/// Reorder single-qubit gates as early as possible in the instruction stream.
///
/// Moves each 1q gate backward past non-conflicting instructions (those that
/// don't touch the same qubit). Diagonal 1q gates can also pass through CX
/// (when on the control qubit) and CZ (on either qubit) via commutation.
///
/// This groups 1q gates together, maximizing batching opportunities for the
/// subsequent `fuse_multi_1q_gates()` pass.
///
/// Returns `Cow::Borrowed` when no reordering is possible.
pub fn reorder_1q_gates(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_reorder_opportunity(&circuit) {
        return circuit;
    }

    let n = circuit.num_qubits;
    // block_all[q] / block_diag[q]: index into non_1q of the last blocker
    let mut block_all: Vec<usize> = vec![usize::MAX; n];
    let mut block_diag: Vec<usize> = vec![usize::MAX; n];
    let mut last_1q_slot: Vec<usize> = vec![0; n];
    let mut non_1q: Vec<&Instruction> = Vec::new();
    let mut slots: Vec<Vec<Instruction>> = vec![Vec::new()];

    for inst in &circuit.instructions {
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
                slots[slot].push(inst.clone());
                last_1q_slot[q] = slot;
            }
            _ => {
                let idx = non_1q.len();
                non_1q.push(inst);
                slots.push(Vec::new());
                match inst {
                    Instruction::Gate { gate, targets } => match gate {
                        Gate::Cx => {
                            block_all[targets[0]] = idx;
                            // block_diag[targets[0]] unchanged — diagonal commutes on control
                            block_all[targets[1]] = idx;
                            block_diag[targets[1]] = idx;
                        }
                        Gate::Cz | Gate::Rzz(_) => {
                            block_all[targets[0]] = idx;
                            block_all[targets[1]] = idx;
                            // block_diag unchanged for both — diagonal commutes on both
                        }
                        Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => {
                            for &q in targets.iter() {
                                block_all[q] = idx;
                            }
                            // block_diag unchanged — all-diagonal gate
                        }
                        _ => {
                            for &q in targets.iter() {
                                block_all[q] = idx;
                                block_diag[q] = idx;
                            }
                        }
                    },
                    Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                        block_all[*qubit] = idx;
                        block_diag[*qubit] = idx;
                    }
                    Instruction::Barrier { qubits } => {
                        for &q in qubits.iter() {
                            block_all[q] = idx;
                            block_diag[q] = idx;
                        }
                    }
                    Instruction::Conditional { targets, .. } => {
                        for &q in targets.iter() {
                            block_all[q] = idx;
                            block_diag[q] = idx;
                        }
                    }
                }
            }
        }
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    for (i, non_1q_inst) in non_1q.iter().enumerate() {
        output.append(&mut slots[i]);
        output.push((*non_1q_inst).clone());
    }
    output.append(&mut slots[non_1q.len()]);

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

/// Fuse single-qubit gates on distinct qubits into `Gate::MultiFused`.
///
/// Accumulates 1q gates per-qubit across the instruction stream. When a non-1q
/// instruction is encountered, only the pending gates on the **affected qubits**
/// are flushed — gates on unrelated qubits continue accumulating. This produces
/// fewer but larger MultiFused batches than flushing the entire run at every 2q gate.
///
/// Correctness: a 1q gate on qubit q commutes with any multi-qubit gate not
/// involving q (independent subspaces), so deferring its application is safe.
pub fn fuse_multi_1q_gates(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_multi_1q_run(&circuit) {
        return circuit;
    }

    let n = circuit.num_qubits;
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut pending: Vec<Option<[[Complex64; 2]; 2]>> = vec![None; n];
    let mut pending_count = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = match gate {
                    Gate::Fused(m) => **m,
                    _ => gate.matrix_2x2(),
                };
                match &mut pending[q] {
                    Some(existing) => *existing = mat_mul_2x2(&mat, existing),
                    slot => {
                        *slot = Some(mat);
                        pending_count += 1;
                    }
                }
            }
            _ => {
                for &q in inst_qubits(inst) {
                    flush_1q_pending(q, &mut pending, &mut pending_count, &mut output);
                }
                output.push(inst.clone());
            }
        }
    }
    flush_all_pending(&mut pending, &mut pending_count, &mut output);

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

fn flush_1q_pending(
    q: usize,
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    pending_count: &mut usize,
    output: &mut Vec<Instruction>,
) {
    if let Some(mat) = pending[q].take() {
        *pending_count -= 1;
        output.push(Instruction::Gate {
            gate: Gate::Fused(Box::new(mat)),
            targets: smallvec![q],
        });
    }
}

fn flush_all_pending(
    pending: &mut [Option<[[Complex64; 2]; 2]>],
    pending_count: &mut usize,
    output: &mut Vec<Instruction>,
) {
    if *pending_count >= 2 {
        let mut gates: Vec<(usize, [[Complex64; 2]; 2])> = Vec::with_capacity(*pending_count);
        for (q, slot) in pending.iter_mut().enumerate() {
            if let Some(mat) = slot.take() {
                gates.push((q, mat));
            }
        }
        let all_diagonal = gates
            .iter()
            .all(|(_, m)| m[0][1].norm() < IDENTITY_EPS && m[1][0].norm() < IDENTITY_EPS);
        let targets: SmallVec<[usize; 4]> = gates.iter().map(|&(t, _)| t).collect();
        output.push(Instruction::Gate {
            gate: Gate::MultiFused(Box::new(MultiFusedData {
                gates,
                all_diagonal,
            })),
            targets,
        });
    } else {
        for (q, slot) in pending.iter_mut().enumerate() {
            if let Some(mat) = slot.take() {
                output.push(Instruction::Gate {
                    gate: Gate::Fused(Box::new(mat)),
                    targets: smallvec![q],
                });
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

/// Fuse adjacent single-qubit gates into two-qubit gates.
///
/// Scans for patterns where a CX or CZ gate has pending 1q gates on its qubits.
/// The 1q gates are absorbed into the 2q gate via Kronecker product,
/// producing a `Gate::Fused2q` with a 4×4 unitary.
///
/// Only CX and CZ are targeted. SWAP and Cu have specialized SIMD kernels;
/// Cu is excluded to preserve downstream cphase batching.
///
/// Algorithm: greedy forward pass, absorbing pre-gates only. Post-gates of one
/// 2q gate become pre-gates of the next, so most HEA-style patterns are captured.
///
/// Returns `Cow::Borrowed` when no fusion opportunities exist.
pub fn fuse_2q_gates(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_2q_fusion_opportunity(&circuit) {
        return circuit;
    }

    let identity_2x2 = Gate::Id.matrix_2x2();
    let n = circuit.num_qubits;
    let mut pending_1q: Vec<Option<[[Complex64; 2]; 2]>> = vec![None; n];
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());

    let flush_1q =
        |q: usize, pending: &mut [Option<[[Complex64; 2]; 2]>], output: &mut Vec<Instruction>| {
            if let Some(mat) = pending[q].take() {
                output.push(Instruction::Gate {
                    gate: Gate::Fused(Box::new(mat)),
                    targets: smallvec![q],
                });
            }
        };

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                let q = targets[0];
                let mat = match gate {
                    Gate::Fused(m) => **m,
                    _ => gate.matrix_2x2(),
                };
                match &mut pending_1q[q] {
                    Some(p) => *p = mat_mul_2x2(&mat, p),
                    slot => *slot = Some(mat),
                }
            }
            Instruction::Gate {
                gate: gate @ (Gate::Cx | Gate::Cz),
                targets,
            } => {
                let q0 = targets[0];
                let q1 = targets[1];
                let pre0 = pending_1q[q0].take();
                let pre1 = pending_1q[q1].take();

                if pre0.is_none() && pre1.is_none() {
                    output.push(inst.clone());
                } else {
                    let m0 = pre0.unwrap_or(identity_2x2);
                    let m1 = pre1.unwrap_or(identity_2x2);
                    let kron = kron_2x2(&m0, &m1);
                    let gate4 = gate.matrix_4x4();
                    let fused = mat_mul_4x4(&gate4, &kron);
                    output.push(Instruction::Gate {
                        gate: Gate::Fused2q(Box::new(fused)),
                        targets: smallvec![q0, q1],
                    });
                }
            }
            Instruction::Gate { targets, .. } => {
                for &q in targets.iter() {
                    flush_1q(q, &mut pending_1q, &mut output);
                }
                output.push(inst.clone());
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                flush_1q(*qubit, &mut pending_1q, &mut output);
                output.push(inst.clone());
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits.iter() {
                    flush_1q(q, &mut pending_1q, &mut output);
                }
                output.push(inst.clone());
            }
            Instruction::Conditional { targets, .. } => {
                for &q in targets.iter() {
                    flush_1q(q, &mut pending_1q, &mut output);
                }
                output.push(inst.clone());
            }
        }
    }

    for q in 0..n {
        flush_1q(q, &mut pending_1q, &mut output);
    }

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

/// Check whether the circuit has any CX/CZ gate preceded by a 1q gate on the same qubit.
fn has_2q_fusion_opportunity(circuit: &Circuit) -> bool {
    let mut has_pending_1q = vec![false; circuit.num_qubits];
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                has_pending_1q[targets[0]] = true;
            }
            Instruction::Gate {
                gate: Gate::Cx | Gate::Cz,
                targets,
            } => {
                if has_pending_1q[targets[0]] || has_pending_1q[targets[1]] {
                    return true;
                }
                has_pending_1q[targets[0]] = false;
                has_pending_1q[targets[1]] = false;
            }
            Instruction::Gate { targets, .. } => {
                for &q in targets.iter() {
                    has_pending_1q[q] = false;
                }
            }
            Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                has_pending_1q[*qubit] = false;
            }
            Instruction::Barrier { qubits } => {
                for &q in qubits.iter() {
                    has_pending_1q[q] = false;
                }
            }
            Instruction::Conditional { targets, .. } => {
                for &q in targets.iter() {
                    has_pending_1q[q] = false;
                }
            }
        }
    }
    false
}

/// Cache-tier classification for 2q gates based on max target qubit.
///
/// A 2q gate on (q0, q1) fits in a tile of 2^N elements iff max(q0, q1) < N.
/// L2 tiles = 16384 = 2^14 → max qubit ≤ 13.
/// L3 tiles = 131072 = 2^17 → max qubit ≤ 16.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier2q {
    L2,
    L3,
    Individual,
}

fn classify_2q_tier(q0: usize, q1: usize) -> Tier2q {
    let max_q = q0.max(q1);
    if max_q <= 13 {
        Tier2q::L2
    } else if max_q <= 16 {
        Tier2q::L3
    } else {
        Tier2q::Individual
    }
}

#[inline]
fn swap_order_4x4(mat: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let swap = Gate::Swap.matrix_4x4();
    mat_mul_4x4(&swap, &mat_mul_4x4(mat, &swap))
}

#[inline]
fn is_diagonal_4x4(mat: &[[Complex64; 4]; 4]) -> bool {
    for (r, row) in mat.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            if r != c && value.norm() >= IDENTITY_EPS {
                return false;
            }
        }
    }
    true
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
}

impl PairRun {
    fn new(q0: usize, q1: usize, mat: [[Complex64; 4]; 4], original: Instruction) -> Self {
        Self {
            q0,
            q1,
            acc: mat,
            fused_2q_count: 1,
            has_nondiagonal_2q: !is_diagonal_4x4(&mat),
            originals: vec![original],
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

    fn push_2q(&mut self, mat: [[Complex64; 4]; 4], targets: &[usize], original: Instruction) {
        let oriented = orient_2q_matrix(&mat, targets, self.q0, self.q1);
        self.acc = mat_mul_4x4(&oriented, &self.acc);
        self.fused_2q_count += 1;
        self.has_nondiagonal_2q |= !is_diagonal_4x4(&oriented);
        self.originals.push(original);
    }

    fn push_1q(&mut self, mat: [[Complex64; 2]; 2], target: usize, original: Instruction) {
        let embedded = embed_1q_matrix(&mat, target, self.q0);
        self.acc = mat_mul_4x4(&embedded, &self.acc);
        self.originals.push(original);
    }

    fn should_fuse(&self) -> bool {
        self.fused_2q_count >= 2 && self.has_nondiagonal_2q
    }
}

fn flush_pair_run(run: &mut Option<PairRun>, output: &mut Vec<Instruction>, changed: &mut bool) {
    let Some(run) = run.take() else {
        return;
    };
    if run.should_fuse() {
        output.push(Instruction::Gate {
            gate: Gate::Fused2q(Box::new(run.acc)),
            targets: smallvec![run.q0, run.q1],
        });
        *changed = true;
    } else {
        output.extend(run.originals);
    }
}

/// Fuse contiguous same-pair `Fused2q` runs into one larger `Fused2q`.
///
/// This pass is deliberately narrow: it only consumes existing `Fused2q` gates
/// and single-qubit gates on the same two qubits, and only emits a fused block
/// when at least two 2q units are present. All-diagonal runs are left alone so
/// diagonal batch passes keep their cheaper kernels.
fn fuse_same_pair_2q_blocks(input: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    let circuit = input.as_ref();
    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut run: Option<PairRun> = None;
    let mut changed = false;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate {
                gate: Gate::Fused2q(mat),
                targets,
            } => {
                if let Some(active) = &mut run {
                    if active.can_accept_pair(targets[0], targets[1]) {
                        active.push_2q(**mat, targets, inst.clone());
                    } else {
                        flush_pair_run(&mut run, &mut output, &mut changed);
                        run = Some(PairRun::new(targets[0], targets[1], **mat, inst.clone()));
                    }
                } else {
                    run = Some(PairRun::new(targets[0], targets[1], **mat, inst.clone()));
                }
            }
            Instruction::Gate { gate, targets } if gate.num_qubits() == 1 => {
                if let Some(active) = &mut run {
                    if active.can_accept_1q(targets[0]) {
                        let mat = match gate {
                            Gate::Fused(m) => **m,
                            _ => gate.matrix_2x2(),
                        };
                        active.push_1q(mat, targets[0], inst.clone());
                    } else {
                        flush_pair_run(&mut run, &mut output, &mut changed);
                        output.push(inst.clone());
                    }
                } else {
                    output.push(inst.clone());
                }
            }
            _ => {
                flush_pair_run(&mut run, &mut output, &mut changed);
                output.push(inst.clone());
            }
        }
    }

    flush_pair_run(&mut run, &mut output, &mut changed);

    if changed {
        Cow::Owned(Circuit {
            num_qubits: circuit.num_qubits,
            num_classical_bits: circuit.num_classical_bits,
            instructions: output,
        })
    } else {
        input
    }
}

/// Batch consecutive `Fused2q` gates into `Multi2q` for cache-tiled execution.
///
/// Scans for runs of consecutive `Fused2q` instructions within the same cache
/// tier (L2 or L3). Each run of ≥2 gates is replaced by a single `Multi2q`
/// gate that the statevector backend applies in a tiled pass. Individual-tier
/// gates (max qubit > 16) are left as-is.
///
/// Returns `Cow::Borrowed` when no batching opportunities exist.
pub fn fuse_multi_2q_gates(circuit: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    if !has_multi_2q_opportunity(&circuit) {
        return circuit;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len());
    let mut pending: Vec<(usize, usize, [[Complex64; 4]; 4])> = Vec::new();
    let mut current_tier: Option<Tier2q> = None;

    let flush = |pending: &mut Vec<(usize, usize, [[Complex64; 4]; 4])>,
                 tier: &mut Option<Tier2q>,
                 output: &mut Vec<Instruction>| {
        if pending.is_empty() {
            return;
        }
        let t = tier.take().unwrap();
        if t == Tier2q::Individual || pending.len() < MIN_MULTI_2Q_BATCH {
            for (q0, q1, mat) in pending.drain(..) {
                output.push(Instruction::Gate {
                    gate: Gate::Fused2q(Box::new(mat)),
                    targets: smallvec![q0, q1],
                });
            }
        } else {
            let mut all_qubits: SmallVec<[usize; 4]> = SmallVec::new();
            for &(q0, q1, _) in pending.iter() {
                if !all_qubits.contains(&q0) {
                    all_qubits.push(q0);
                }
                if !all_qubits.contains(&q1) {
                    all_qubits.push(q1);
                }
            }
            all_qubits.sort_unstable();
            output.push(Instruction::Gate {
                gate: Gate::Multi2q(Box::new(Multi2qData {
                    gates: std::mem::take(pending),
                })),
                targets: all_qubits,
            });
        }
    };

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate {
                gate: Gate::Fused2q(mat),
                targets,
            } => {
                let q0 = targets[0];
                let q1 = targets[1];
                let tier = classify_2q_tier(q0, q1);

                if let Some(ct) = current_tier {
                    if ct != tier {
                        flush(&mut pending, &mut current_tier, &mut output);
                    }
                }
                if current_tier.is_none() {
                    current_tier = Some(tier);
                }
                pending.push((q0, q1, **mat));
            }
            _ => {
                flush(&mut pending, &mut current_tier, &mut output);
                output.push(inst.clone());
            }
        }
    }
    flush(&mut pending, &mut current_tier, &mut output);

    Cow::Owned(Circuit {
        num_qubits: circuit.num_qubits,
        num_classical_bits: circuit.num_classical_bits,
        instructions: output,
    })
}

/// Check if the circuit has ≥2 consecutive Fused2q gates in a tileable tier.
fn has_multi_2q_opportunity(circuit: &Circuit) -> bool {
    let mut consecutive = 0usize;
    for inst in &circuit.instructions {
        if let Instruction::Gate {
            gate: Gate::Fused2q(_),
            targets,
        } = inst
        {
            let tier = classify_2q_tier(targets[0], targets[1]);
            if tier != Tier2q::Individual {
                consecutive += 1;
                if consecutive >= MIN_MULTI_2Q_BATCH {
                    return true;
                }
                continue;
            }
        }
        consecutive = 0;
    }
    false
}

/// Returns true if `gate` is a diagonal gate that can be absorbed into a DiagonalBatch.
fn is_diag_batchable(gate: &Gate) -> bool {
    match gate {
        Gate::Cz | Gate::Rzz(_) => true,
        _ if gate.is_diagonal_1q() => true,
        _ if gate.controlled_phase().is_some() => true,
        _ => false,
    }
}

/// Convert a gate to one or more DiagEntry values.
fn gate_to_diag_entries(gate: &Gate, targets: &[usize]) -> SmallVec<[DiagEntry; 2]> {
    match gate {
        Gate::Cz => {
            smallvec![DiagEntry::Phase2q {
                q0: targets[0],
                q1: targets[1],
                phase: Complex64::new(-1.0, 0.0),
            }]
        }
        Gate::Rzz(theta) => {
            let half = theta / 2.0;
            let same = Complex64::new((-half).cos(), (-half).sin()); // e^{-iθ/2}
            let diff = Complex64::new(half.cos(), half.sin()); // e^{iθ/2}
            smallvec![DiagEntry::Parity2q {
                q0: targets[0],
                q1: targets[1],
                same,
                diff,
            }]
        }
        _ if gate.controlled_phase().is_some() => {
            let phase = gate.controlled_phase().unwrap();
            smallvec![DiagEntry::Phase2q {
                q0: targets[0],
                q1: targets[1],
                phase,
            }]
        }
        _ => {
            let mat = gate.matrix_2x2();
            smallvec![DiagEntry::Phase1q {
                qubit: targets[0],
                d0: mat[0][0],
                d1: mat[1][1],
            }]
        }
    }
}

/// Batch contiguous runs of diagonal gates into `DiagonalBatch` instructions.
///
/// Diagonal gates (Z, S, T, Rz, P, CZ, Rzz, CPhase) commute with each other,
/// so adjacent diagonal gates can be collapsed into a single pass with precomputed
/// phase LUTs. Non-diagonal gates on non-involved qubits are deferred past the run.
fn fuse_diagonal_batch(input: Cow<'_, Circuit>) -> Cow<'_, Circuit> {
    let circuit = input.as_ref();
    let insts = &circuit.instructions;
    let n = insts.len();
    if n < 2 {
        return input;
    }

    let diag_count = insts
        .iter()
        .filter(|i| matches!(i, Instruction::Gate { gate, .. } if is_diag_batchable(gate)))
        .count();
    if diag_count < 2 {
        return input;
    }

    let mut output: Vec<Instruction> = Vec::with_capacity(n);
    let mut run_entries: Vec<DiagEntry> = Vec::new();
    let mut run_originals: Vec<Instruction> = Vec::new();
    let mut run_qubits = vec![false; circuit.num_qubits];
    let mut deferred: Vec<Instruction> = Vec::new();

    let flush_diag_run = |output: &mut Vec<Instruction>,
                          entries: &mut Vec<DiagEntry>,
                          originals: &mut Vec<Instruction>,
                          deferred: &mut Vec<Instruction>,
                          run_qubits: &mut [bool]| {
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
    };

    for inst in insts {
        if let Instruction::Gate { gate, targets } = inst {
            if is_diag_batchable(gate) {
                let new_entries = gate_to_diag_entries(gate, targets);
                for t in targets.iter() {
                    run_qubits[*t] = true;
                }
                run_entries.extend(new_entries);
                run_originals.push(inst.clone());
                continue;
            }

            if !run_entries.is_empty() && gate.num_qubits() == 1 && !run_qubits[targets[0]] {
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
        );
        output.push(inst.clone());
    }

    flush_diag_run(
        &mut output,
        &mut run_entries,
        &mut run_originals,
        &mut deferred,
        &mut run_qubits,
    );

    let mut c = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
    c.instructions = output;
    Cow::Owned(c)
}

/// Returns `Cow::Borrowed` when no fusion is profitable (zero overhead).
/// Set `supports_fused` to `false` for backends that cannot handle fused gates
/// (e.g., stabilizer).
pub fn fuse_circuit<'a>(circuit: &'a Circuit, supports_fused: bool) -> Cow<'a, Circuit> {
    if !supports_fused {
        return Cow::Borrowed(circuit);
    }

    // Pair cancellation — always applied (zero-cost when no pairs found)
    let pass0 = cancel_self_inverse_pairs(circuit);

    // CX·Rz·CX → Rzz — always applied (zero-cost when no patterns found)
    let pass0r = match pass0 {
        Cow::Borrowed(c) => fuse_rzz(c),
        Cow::Owned(c) => Cow::Owned(fuse_rzz(&c).into_owned()),
    };

    // Batch consecutive Rzz gates (≥16q) — collapses N Rzz into 1 BatchRzz pass
    let pass0b = if circuit.num_qubits >= MIN_QUBITS_FOR_DIAG_BATCH {
        match pass0r {
            Cow::Borrowed(c) => fuse_batch_rzz(c),
            Cow::Owned(c) => Cow::Owned(fuse_batch_rzz(&c).into_owned()),
        }
    } else {
        pass0r
    };

    if circuit.num_qubits < MIN_QUBITS_FOR_FUSION {
        return pass0b;
    }

    // 1q fusion + reorder — clone cost justified at ≥10q
    let pass1 = match pass0b {
        Cow::Borrowed(c) => fuse_single_qubit_gates(c),
        Cow::Owned(c) => Cow::Owned(fuse_single_qubit_gates(&c).into_owned()),
    };
    let pass1r = reorder_1q_gates(pass1);

    // Re-run cancel + 1q-fuse after reorder: commutation may expose new
    // adjacent pairs. E.g. CX·T·CX → T·CX·CX (cancellable), or
    // H·CZ·T → H·T·CZ (fusable). Zero-cost via Cow when no new opportunities.
    let pass1c = match pass1r {
        Cow::Borrowed(c) => cancel_self_inverse_pairs(c),
        Cow::Owned(c) => Cow::Owned(cancel_self_inverse_pairs(&c).into_owned()),
    };
    let pass1f = match pass1c {
        Cow::Borrowed(c) => fuse_single_qubit_gates(c),
        Cow::Owned(c) => Cow::Owned(fuse_single_qubit_gates(&c).into_owned()),
    };

    // Expensive passes — gated by qubit count
    let pass_2q = if circuit.num_qubits >= MIN_QUBITS_FOR_2Q_FUSION {
        fuse_2q_gates(pass1f)
    } else {
        pass1f
    };
    let pass_2qb = if circuit.num_qubits >= MIN_QUBITS_FOR_2Q_FUSION {
        fuse_same_pair_2q_blocks(pass_2q)
    } else {
        pass_2q
    };
    let pass2 = if circuit.num_qubits >= MIN_QUBITS_FOR_MULTI_FUSION {
        fuse_multi_1q_gates(pass_2qb)
    } else {
        pass_2qb
    };
    let pass_m2q = if circuit.num_qubits >= MIN_QUBITS_FOR_MULTI_2Q_FUSION {
        fuse_multi_2q_gates(pass2)
    } else {
        pass2
    };
    let pass_cp = if circuit.num_qubits >= MIN_QUBITS_FOR_DIAG_BATCH {
        fuse_controlled_phases(pass_m2q)
    } else {
        pass_m2q
    };
    let pass_db = if circuit.num_qubits >= MIN_QUBITS_FOR_DIAG_BATCH {
        fuse_diagonal_batch(pass_cp)
    } else {
        pass_cp
    };
    if circuit.num_qubits >= MIN_QUBITS_FOR_POST_PHASE_BATCH {
        batch_post_phase_1q(pass_db)
    } else {
        pass_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    const EPS: f64 = 1e-12;

    fn assert_mat_close(actual: &[[Complex64; 2]; 2], expected: &[[Complex64; 2]; 2]) {
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (actual[i][j] - expected[i][j]).norm() < EPS,
                    "[{i}][{j}]: expected {:?}, got {:?}",
                    expected[i][j],
                    actual[i][j]
                );
            }
        }
    }

    fn identity_mat() -> [[Complex64; 2]; 2] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        [[one, zero], [zero, one]]
    }

    fn count_fused_2q(circuit: &Circuit) -> usize {
        circuit
            .instructions
            .iter()
            .filter(|inst| {
                matches!(
                    inst,
                    Instruction::Gate {
                        gate: Gate::Fused2q(_),
                        ..
                    }
                )
            })
            .count()
    }

    #[test]
    fn test_mat_mul_identity() {
        let id = identity_mat();
        let h = Gate::H.matrix_2x2();
        assert_mat_close(&mat_mul_2x2(&id, &h), &h);
        assert_mat_close(&mat_mul_2x2(&h, &id), &h);
    }

    #[test]
    fn test_mat_mul_h_h_is_identity() {
        let h = Gate::H.matrix_2x2();
        let result = mat_mul_2x2(&h, &h);
        assert_mat_close(&result, &identity_mat());
    }

    #[test]
    fn test_mat_mul_associative() {
        let a = Gate::Rx(1.0).matrix_2x2();
        let b = Gate::Ry(0.7).matrix_2x2();
        let c = Gate::Rz(2.3).matrix_2x2();
        let ab_c = mat_mul_2x2(&mat_mul_2x2(&a, &b), &c);
        let a_bc = mat_mul_2x2(&a, &mat_mul_2x2(&b, &c));
        assert_mat_close(&ab_c, &a_bc);
    }

    fn count_fused(circuit: &Circuit) -> usize {
        circuit
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate {
                        gate: Gate::Fused(_),
                        ..
                    }
                )
            })
            .count()
    }

    fn extract_fused_matrix(inst: &Instruction) -> [[Complex64; 2]; 2] {
        match inst {
            Instruction::Gate {
                gate: Gate::Fused(m),
                ..
            } => **m,
            _ => panic!("expected Fused gate"),
        }
    }

    #[test]
    fn test_empty_circuit() {
        let c = Circuit::new(2, 0);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 0);
    }

    #[test]
    fn test_no_fusion_single_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);
        assert_eq!(count_fused(&fused), 0);
        match &fused.instructions[0] {
            Instruction::Gate {
                gate: Gate::H,
                targets,
            } => assert_eq!(targets.as_slice(), &[0]),
            _ => panic!("expected H gate"),
        }
    }

    #[test]
    fn test_no_fusion_different_qubits() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::X, &[1]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 2);
        assert_eq!(count_fused(&fused), 0);
    }

    #[test]
    fn test_fuse_adjacent_same_qubit() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);
        assert_eq!(count_fused(&fused), 1);
        let expected = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
        let actual = extract_fused_matrix(&fused.instructions[0]);
        assert_mat_close(&actual, &expected);
    }

    #[test]
    fn test_fuse_across_different_qubit() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 2);
        // H·T on q0 stays Fused (no named match), X on q1 recognized as Gate::X
        assert_eq!(count_fused(&fused), 1);

        let expected = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
        let fused_mat = extract_fused_matrix(&fused.instructions[0]);
        assert_mat_close(&fused_mat, &expected);
    }

    #[test]
    fn test_two_qubit_breaks_run() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 3);
        assert_eq!(count_fused(&fused), 0);
    }

    #[test]
    fn test_measure_breaks_run() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 3);
        assert_eq!(count_fused(&fused), 0);
    }

    #[test]
    fn test_barrier_breaks_run() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_barrier(&[0]);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 3);
        assert_eq!(count_fused(&fused), 0);
    }

    #[test]
    fn test_multiple_qubits_independent() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Ry(1.0), &[0]);
        c.add_gate(Gate::Rz(2.0), &[0]);
        c.add_gate(Gate::Ry(1.0), &[1]);
        c.add_gate(Gate::Rz(2.0), &[1]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 2);
        assert_eq!(count_fused(&fused), 2);
    }

    #[test]
    fn test_long_chain() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::Z, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);
        assert_eq!(count_fused(&fused), 1);
        let expected = mat_mul_2x2(
            &Gate::Z.matrix_2x2(),
            &mat_mul_2x2(
                &Gate::X.matrix_2x2(),
                &mat_mul_2x2(
                    &Gate::T.matrix_2x2(),
                    &mat_mul_2x2(&Gate::S.matrix_2x2(), &Gate::H.matrix_2x2()),
                ),
            ),
        );
        let actual = extract_fused_matrix(&fused.instructions[0]);
        assert_mat_close(&actual, &expected);
    }

    #[test]
    fn test_gate_count_after_fusion() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::Ry(1.0), &[0]);
        c.add_gate(Gate::Rz(2.0), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[1]);
        c.add_measure(1, 0);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.gate_count(), 3);
        assert_eq!(fused.instructions.len(), 4);
    }

    #[test]
    fn test_fused_probabilities_match_unfused() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;

        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Ry(1.23), &[1]);
        c.add_gate(Gate::Rz(0.45), &[1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.78), &[0]);
        c.add_gate(Gate::Rx(2.34), &[0]);
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::S, &[2]);
        let mut b1 = StatevectorBackend::new(42);
        b1.init(c.num_qubits, c.num_classical_bits).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let probs_unfused = b1.probabilities().unwrap();
        let fused = fuse_single_qubit_gates(&c);
        let mut b2 = StatevectorBackend::new(42);
        b2.init(fused.num_qubits, fused.num_classical_bits).unwrap();
        for inst in &fused.instructions {
            b2.apply(inst).unwrap();
        }
        let probs_fused = b2.probabilities().unwrap();

        assert_eq!(probs_unfused.len(), probs_fused.len());
        for (i, (a, b)) in probs_unfused.iter().zip(&probs_fused).enumerate() {
            assert!((a - b).abs() < 1e-10, "prob[{i}]: unfused={a}, fused={b}");
        }
    }

    #[test]
    fn test_hea_style_fusion() {
        let mut c = Circuit::new(4, 0);
        for q in 0..4 {
            c.add_gate(Gate::Ry(0.5 + q as f64), &[q]);
            c.add_gate(Gate::Rz(1.0 + q as f64), &[q]);
        }
        for q in 0..3 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..4 {
            c.add_gate(Gate::Ry(2.0 + q as f64), &[q]);
            c.add_gate(Gate::Rz(3.0 + q as f64), &[q]);
        }

        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.gate_count(), 11);
        assert_eq!(count_fused(&fused), 8);
        assert_eq!(c.gate_count(), 19);
    }

    #[test]
    fn test_fusion_matrix_order_matters() {
        // Verify that fusion respects gate ordering (non-commuting gates).
        // X then H is different from H then X.
        let mut c_xh = Circuit::new(1, 0);
        c_xh.add_gate(Gate::X, &[0]);
        c_xh.add_gate(Gate::H, &[0]);

        let mut c_hx = Circuit::new(1, 0);
        c_hx.add_gate(Gate::H, &[0]);
        c_hx.add_gate(Gate::X, &[0]);

        let fused_xh = fuse_single_qubit_gates(&c_xh);
        let fused_hx = fuse_single_qubit_gates(&c_hx);

        let mat_xh = extract_fused_matrix(&fused_xh.instructions[0]);
        let mat_hx = extract_fused_matrix(&fused_hx.instructions[0]);
        let differs = (0..2).any(|i| (0..2).any(|j| (mat_xh[i][j] - mat_hx[i][j]).norm() > EPS));
        assert!(
            differs,
            "X·H and H·X should produce different fused matrices"
        );
        let expected_xh = mat_mul_2x2(&Gate::H.matrix_2x2(), &Gate::X.matrix_2x2());
        assert_mat_close(&mat_xh, &expected_xh);

        let expected_hx = mat_mul_2x2(&Gate::X.matrix_2x2(), &Gate::H.matrix_2x2());
        assert_mat_close(&mat_hx, &expected_hx);
    }

    #[test]
    fn test_s_squared_is_z_via_fusion() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::S, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);
        // S·S recognized as Gate::Z
        assert!(matches!(
            &fused.instructions[0],
            Instruction::Gate { gate: Gate::Z, .. }
        ));
    }

    #[test]
    fn test_t_tdg_cancel_via_fusion() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Tdg, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 0);
    }

    #[test]
    fn test_identity_elision_h_h() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 0);
    }

    #[test]
    fn test_identity_elision_partial() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[1]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);
        // H·H on q0 elided, lone T on q1 recognized as Gate::T
        assert!(matches!(
            &fused.instructions[0],
            Instruction::Gate { gate: Gate::T, .. }
        ));
    }

    #[test]
    fn test_identity_elision_preserves_probabilities() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;

        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Ry(0.7), &[1]);
        c.add_gate(Gate::Rz(1.3), &[1]);

        let mut b1 = StatevectorBackend::new(42);
        b1.init(2, 0).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let p1 = b1.probabilities().unwrap();

        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 1);

        let mut b2 = StatevectorBackend::new(42);
        b2.init(2, 0).unwrap();
        for inst in &fused.instructions {
            b2.apply(inst).unwrap();
        }
        let p2 = b2.probabilities().unwrap();

        for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
            assert!((a - b).abs() < 1e-12, "prob[{i}]: unfused={a}, fused={b}");
        }
    }

    #[test]
    fn test_cphase_breaks_fusion_run() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Rz(0.5), &[0]);
        c.add_gate(Gate::cphase(0.3), &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 3);
        assert_eq!(count_fused(&fused), 0);
    }

    #[test]
    fn test_fusion_around_cphase() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::cphase(0.5), &[0, 1]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::X, &[0]);
        let fused = fuse_single_qubit_gates(&c);
        assert_eq!(fused.instructions.len(), 3);
        assert_eq!(count_fused(&fused), 2);
    }

    #[test]
    fn test_cphase_fused_probabilities_match() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;
        use crate::sim;

        let n = 4;
        let mut c = Circuit::new(n, 0);
        for i in 0..n {
            c.add_gate(Gate::H, &[i]);
            for j in (i + 1)..n {
                let theta = std::f64::consts::TAU / (1u64 << (j - i)) as f64;
                c.add_gate(Gate::cphase(theta), &[i, j]);
            }
        }
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::S, &[0]);

        let fused_circuit = fuse_single_qubit_gates(&c);

        let mut b1 = StatevectorBackend::new(42);
        sim::run_on(&mut b1, &c).unwrap();
        let p1 = b1.probabilities().unwrap();

        let mut b2 = StatevectorBackend::new(42);
        b2.init(n, 0).unwrap();
        for inst in &fused_circuit.instructions {
            b2.apply(inst).unwrap();
        }
        let p2 = b2.probabilities().unwrap();

        for (i, (a, b)) in p1.iter().zip(p2.iter()).enumerate() {
            assert!((*a - *b).abs() < EPS, "prob[{i}]: unfused={a}, fused={b}");
        }
    }

    #[test]
    fn test_cancel_adjacent_cx() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let result = cancel_self_inverse_pairs(&c);
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 0);
    }

    #[test]
    fn test_cancel_cx_with_non_conflicting_between() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]); // doesn't touch q0 or q1
        c.add_gate(Gate::Cx, &[0, 1]);
        let result = cancel_self_inverse_pairs(&c);
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 1); // only H(2) remains
    }

    #[test]
    fn test_no_cancel_cx_with_conflicting_between() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[0]); // touches q0 — breaks the chain
        c.add_gate(Gate::Cx, &[0, 1]);
        let result = cancel_self_inverse_pairs(&c);
        // No cancellation — H on q0 intervenes
        assert_eq!(result.instructions.len(), 3);
    }

    #[test]
    fn test_no_cancel_cx_reversed_targets() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 0]); // reversed — CX is NOT symmetric
        let result = cancel_self_inverse_pairs(&c);
        assert_eq!(result.instructions.len(), 2);
    }

    #[test]
    fn test_cancel_cz_symmetric() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::Cz, &[1, 0]); // reversed — CZ IS symmetric
        let result = cancel_self_inverse_pairs(&c);
        assert_eq!(result.instructions.len(), 0);
    }

    #[test]
    fn test_cancel_swap_symmetric() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Swap, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Swap, &[1, 0]); // reversed order
        let result = cancel_self_inverse_pairs(&c);
        assert_eq!(result.instructions.len(), 1); // only H(2)
    }

    #[test]
    fn test_cancel_multiple_pairs() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let result = cancel_self_inverse_pairs(&c);
        assert_eq!(result.instructions.len(), 0);
    }

    #[test]
    fn test_cancel_no_candidates() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[1]);
        let result = cancel_self_inverse_pairs(&c);
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_cancel_preserves_probabilities() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;

        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::Cx, &[0, 1]); // cancels with the first CX
        c.add_gate(Gate::H, &[1]);

        let mut b1 = StatevectorBackend::new(42);
        b1.init(3, 0).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let p1 = b1.probabilities().unwrap();

        let cancelled = cancel_self_inverse_pairs(&c);
        let mut b2 = StatevectorBackend::new(42);
        b2.init(3, 0).unwrap();
        for inst in &cancelled.instructions {
            b2.apply(inst).unwrap();
        }
        let p2 = b2.probabilities().unwrap();

        for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "prob[{i}]: original={a}, cancelled={b}"
            );
        }
    }

    #[test]
    fn test_reorder_1q_basic() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Fused(Box::new(Gate::H.matrix_2x2())), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Fused(Box::new(Gate::T.matrix_2x2())), &[2]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 3);
        assert!(
            result.instructions[0..2].iter().all(|i| matches!(
                i,
                Instruction::Gate {
                    gate: Gate::Fused(_),
                    ..
                }
            )),
            "first two instructions should be 1q Fused gates"
        );
        let targets: Vec<usize> = result.instructions[0..2]
            .iter()
            .map(|i| match i {
                Instruction::Gate { targets, .. } => targets[0],
                _ => unreachable!(),
            })
            .collect();
        assert!(targets.contains(&0) && targets.contains(&2));
        assert!(matches!(
            &result.instructions[2],
            Instruction::Gate { gate: Gate::Cx, .. }
        ));
    }

    #[test]
    fn test_reorder_no_opportunity() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Fused(Box::new(Gate::H.matrix_2x2())), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Fused(Box::new(Gate::T.matrix_2x2())), &[1]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_reorder_hea_pattern() {
        let mut c = Circuit::new(4, 0);
        let fused = |angle: f64| Gate::Fused(Box::new(Gate::Ry(angle).matrix_2x2()));
        c.add_gate(fused(0.1), &[0]);
        c.add_gate(fused(0.2), &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(fused(0.3), &[2]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(fused(0.4), &[3]);
        c.add_gate(Gate::Cx, &[2, 3]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert_eq!(result.instructions.len(), 7);
        for i in 0..4 {
            assert!(
                matches!(
                    &result.instructions[i],
                    Instruction::Gate { gate, .. } if gate.num_qubits() == 1
                ),
                "instruction {i} should be 1q gate"
            );
        }
        for i in 4..7 {
            assert!(
                matches!(
                    &result.instructions[i],
                    Instruction::Gate { gate: Gate::Cx, .. }
                ),
                "instruction {i} should be CX"
            );
        }
    }

    #[test]
    fn test_reorder_preserves_probabilities() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;

        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Ry(1.2), &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.8), &[2]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::T, &[3]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::H, &[0]);

        let mut b1 = StatevectorBackend::new(42);
        b1.init(4, 0).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let p1 = b1.probabilities().unwrap();

        let reordered = reorder_1q_gates(Cow::Borrowed(&c));
        let mut b2 = StatevectorBackend::new(42);
        b2.init(4, 0).unwrap();
        for inst in &reordered.instructions {
            b2.apply(inst).unwrap();
        }
        let p2 = b2.probabilities().unwrap();

        for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "prob[{i}]: original={a}, reordered={b}"
            );
        }
    }

    #[test]
    fn test_reorder_respects_barrier() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_barrier(&[0, 1, 2]);
        c.add_gate(Gate::T, &[2]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_reorder_diagonal_commutes_through_cx_control() {
        // Rz(0) should move past CX(0,1) since diagonal commutes on control
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.5), &[0]); // diagonal on control — should commute

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 2);
        // Rz should be moved before CX
        assert!(matches!(
            &result.instructions[0],
            Instruction::Gate {
                gate: Gate::Rz(_),
                targets
            } if targets[0] == 0
        ));
        assert!(matches!(
            &result.instructions[1],
            Instruction::Gate { gate: Gate::Cx, .. }
        ));
    }

    #[test]
    fn test_reorder_nondiagonal_blocked_by_cx_control() {
        // H(0) should NOT move past CX(0,1) — H is not diagonal
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[0]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        // No reorder — H is blocked by CX on q0
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_reorder_diagonal_blocked_on_cx_target() {
        // Rz(1) should NOT move past CX(0,1) — q1 is the target
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.5), &[1]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_reorder_diagonal_commutes_through_cz() {
        // T(0) and T(1) should both move past CZ(0,1) — diagonal commutes
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::S, &[1]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 3);
        // Both diagonal gates should precede CZ
        assert!(result.instructions[0..2]
            .iter()
            .all(|i| matches!(i, Instruction::Gate { gate, .. } if gate.num_qubits() == 1)));
        assert!(matches!(
            &result.instructions[2],
            Instruction::Gate { gate: Gate::Cz, .. }
        ));
    }

    #[test]
    fn test_reorder_commutation_preserves_probabilities() {
        use crate::backend::statevector::StatevectorBackend;
        use crate::backend::Backend;

        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.7), &[0]); // diagonal on CX control — commutes
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::T, &[1]); // diagonal — commutes through CZ
        c.add_gate(Gate::S, &[2]); // diagonal — commutes through CZ

        let mut b1 = StatevectorBackend::new(42);
        b1.init(3, 0).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let p1 = b1.probabilities().unwrap();

        let reordered = reorder_1q_gates(Cow::Borrowed(&c));
        let mut b2 = StatevectorBackend::new(42);
        b2.init(3, 0).unwrap();
        for inst in &reordered.instructions {
            b2.apply(inst).unwrap();
        }
        let p2 = b2.probabilities().unwrap();

        for (i, (a, b)) in p1.iter().zip(&p2).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "prob[{i}]: original={a}, reordered={b}"
            );
        }
    }

    #[test]
    fn test_reorder_respects_measurement() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        c.add_gate(Gate::T, &[1]);

        let result = reorder_1q_gates(Cow::Borrowed(&c));
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result.instructions.len(), 3);
        assert!(
            result.instructions[0..2]
                .iter()
                .all(|i| matches!(i, Instruction::Gate { .. })),
            "first two instructions should be 1q gates"
        );
        let targets: Vec<usize> = result.instructions[0..2]
            .iter()
            .map(|i| match i {
                Instruction::Gate { targets, .. } => targets[0],
                _ => unreachable!(),
            })
            .collect();
        assert!(targets.contains(&0) && targets.contains(&1));
        assert!(matches!(
            &result.instructions[2],
            Instruction::Measure { .. }
        ));
    }

    #[test]
    fn test_smart_multi_fusion_across_cx() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::S, &[2]);
        c.add_gate(Gate::Rx(1.0), &[3]);

        let pass1 = fuse_single_qubit_gates(&c);
        let pass2 = fuse_multi_1q_gates(pass1);

        let multi_count = pass2
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate {
                        gate: Gate::MultiFused(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            multi_count, 1,
            "q2 and q3 gates should accumulate across CX(q0,q1) into one MultiFused"
        );

        if let Instruction::Gate {
            gate: Gate::MultiFused(data),
            ..
        } = &pass2.instructions.last().unwrap()
        {
            let targets: Vec<usize> = data.gates.iter().map(|&(t, _)| t).collect();
            assert!(targets.contains(&2), "q2 should be in the MultiFused batch");
            assert!(targets.contains(&3), "q3 should be in the MultiFused batch");
        } else {
            panic!("last instruction should be MultiFused");
        }

        let mut b1 = crate::backend::statevector::StatevectorBackend::new(42);
        b1.init(c.num_qubits, 0).unwrap();
        for inst in &c.instructions {
            b1.apply(inst).unwrap();
        }
        let probs1 = b1.probabilities().unwrap();

        let mut b2 = crate::backend::statevector::StatevectorBackend::new(42);
        b2.init(pass2.num_qubits, 0).unwrap();
        for inst in &pass2.instructions {
            b2.apply(inst).unwrap();
        }
        let probs2 = b2.probabilities().unwrap();

        for (a, b) in probs1.iter().zip(probs2.iter()) {
            assert!(
                (*a - *b).abs() < 1e-12,
                "probabilities must match: {a} vs {b}"
            );
        }
    }

    #[test]
    fn reorder_exposes_cx_cancellation() {
        // CX q0,q1; T q0; CX q0,q1 — T is diagonal on CX control,
        // reorder moves it past CX, exposing CX·CX cancellation.
        let mut c = Circuit::new(10, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);

        let fused = fuse_circuit(&c, true);
        // CX pair cancelled; only T (as Fused) on q0 remains
        let gate_count = fused
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Gate { .. }))
            .count();
        assert_eq!(gate_count, 1, "CX pair should cancel after reorder");
    }

    #[test]
    fn reorder_exposes_1q_fusion() {
        // H q0; CZ q0,q1; T q0 — T is diagonal, commutes through CZ on either qubit.
        // After reorder: H q0; T q0; CZ q0,q1 — H and T fuse.
        let mut c = Circuit::new(10, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::T, &[0]);

        let fused = fuse_circuit(&c, true);
        let gates: Vec<_> = fused
            .instructions
            .iter()
            .filter_map(|i| match i {
                Instruction::Gate { gate, targets } => Some((gate.clone(), targets.clone())),
                _ => None,
            })
            .collect();
        // Should be: Fused(H·T) on q0, CZ on q0,q1
        assert_eq!(gates.len(), 2, "H and T should fuse after reorder");
        assert!(
            matches!(&gates[0].0, Gate::Fused(_)),
            "first gate should be Fused(H·T)"
        );
        assert!(matches!(&gates[1].0, Gate::Cz), "second gate should be CZ");

        // Verify fused matrix = T · H (T applied after H)
        let t_mat = Gate::T.matrix_2x2();
        let h_mat = Gate::H.matrix_2x2();
        let expected = mat_mul_2x2(&t_mat, &h_mat);
        if let Gate::Fused(mat) = &gates[0].0 {
            assert_mat_close(mat, &expected);
        }
    }

    #[test]
    fn reorder_exposes_cz_cancellation() {
        // CZ q0,q1; S q0; CZ q0,q1 — S is diagonal, commutes through CZ.
        // After reorder: S q0; CZ q0,q1; CZ q0,q1 → CZ pair cancels.
        let mut c = Circuit::new(10, 0);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Cz, &[0, 1]);

        let fused = fuse_circuit(&c, true);
        let gate_count = fused
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Gate { .. }))
            .count();
        assert_eq!(gate_count, 1, "CZ pair should cancel after reorder");
    }

    #[test]
    fn same_pair_2q_block_fuses_w_state_pairs() {
        let circuit = crate::circuits::w_state_circuit(20);
        let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
        let before = count_fused_2q(&pass_2q);
        let fused = fuse_same_pair_2q_blocks(pass_2q);
        let after = count_fused_2q(&fused);

        assert!(before >= 2, "w-state should expose paired Fused2q gates");
        assert!(
            after < before,
            "same-pair block fusion should reduce Fused2q count"
        );
    }

    #[test]
    fn same_pair_2q_block_fuses_qv_blocks() {
        let circuit = crate::circuits::quantum_volume_circuit(20, 1, 42);
        let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
        let before = count_fused_2q(&pass_2q);
        let fused = fuse_same_pair_2q_blocks(pass_2q);
        let after = count_fused_2q(&fused);

        assert!(before >= 2, "qv block should expose paired Fused2q gates");
        assert!(
            after < before,
            "same-pair block fusion should reduce Fused2q count"
        );
    }

    #[test]
    fn same_pair_2q_block_accepts_reversed_targets() {
        let mut circuit = Circuit::new(20, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::Ry(0.7), &[1]);
        circuit.add_gate(Gate::Cx, &[1, 0]);

        let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
        let before = count_fused_2q(&pass_2q);
        let fused = fuse_same_pair_2q_blocks(pass_2q);
        let after = count_fused_2q(&fused);

        assert_eq!(before, 2);
        assert_eq!(after, 1, "reversed pair order should still fuse");
        assert!(matches!(
            &fused.instructions[0],
            Instruction::Gate {
                gate: Gate::Fused2q(_),
                targets
            } if targets.as_slice() == [0, 1]
        ));
    }

    #[test]
    fn same_pair_2q_block_leaves_diagonal_runs() {
        let mut circuit = Circuit::new(20, 0);
        circuit.add_gate(Gate::Rz(0.3), &[0]);
        circuit.add_gate(Gate::Cz, &[0, 1]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::Cz, &[1, 0]);

        let pass_2q = fuse_2q_gates(Cow::Borrowed(&circuit));
        let before = count_fused_2q(&pass_2q);
        let fused = fuse_same_pair_2q_blocks(pass_2q);
        let after = count_fused_2q(&fused);

        assert_eq!(before, 2);
        assert_eq!(after, 2, "all-diagonal Fused2q runs should stay split");
    }

    #[test]
    fn qaoa_20q_fuses_to_6_instructions() {
        let circuit = crate::circuits::qaoa_circuit(20, 3, 0xDEAD_BEEF);
        let fused = fuse_circuit(&circuit, true);
        assert_eq!(fused.instructions.len(), 6);

        let batch_rzz_count = fused
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate {
                        gate: Gate::BatchRzz(_),
                        ..
                    }
                )
            })
            .count();
        let multi_fused_count = fused
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    Instruction::Gate {
                        gate: Gate::MultiFused(_),
                        ..
                    }
                )
            })
            .count();
        assert_eq!(batch_rzz_count, 3);
        assert_eq!(multi_fused_count, 3);
    }

    #[test]
    fn test_recognition_extends_clifford_prefix() {
        let mut c = Circuit::new(2, 0);
        // T, T on q0 then CX then Rx on q1
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rx(0.7), &[1]);

        // Without fusion: first gate is T (non-Clifford), so no prefix at all
        assert!(c.clifford_prefix_split().is_none());

        // After fusion: T·T recognized as S (Clifford), so prefix = [S, CX], tail = [Rx]
        let fused = fuse_single_qubit_gates(&c);
        assert!(matches!(
            &fused.instructions[0],
            Instruction::Gate { gate: Gate::S, .. }
        ));
        let split = fused.clifford_prefix_split();
        assert!(split.is_some());
        let (pre_f, tail_f) = split.unwrap();
        assert_eq!(pre_f.instructions.len(), 2); // S + CX
        assert_eq!(tail_f.instructions.len(), 1); // Rx(0.7)
    }
}
