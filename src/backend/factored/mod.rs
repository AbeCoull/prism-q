//! Factored statevector backend.
//!
//! Maintains separate sub-state vectors per entangled qubit group. When a
//! multi-qubit gate bridges two groups, they merge via tensor product. A
//! measurement or reset leaves its qubit product with the rest of the group, so
//! it splits back out into a group of its own once the group is wide enough to
//! pay for the split. For sparse-entanglement circuits this is exponentially
//! cheaper than a monolithic 2^n statevector.
//!
//! # Memory layout
//!
//! - One dense `Vec<Complex64>` per group, length 2^k for k group qubits;
//!   total cost is the sum over groups, dominated by the largest.
//! - Each group lists its global qubits sorted ascending; position = local
//!   qubit index. A global map resolves qubit to group.
//! - A split compacts its group in place and keeps the allocation, so a split
//!   qubit merging back in grows the group without allocating.
//!
//! # Gate support
//!
//! The full gate set, applied with the statevector kernels inside each group.
//! Rayon and SIMD paths engage per group at `PARALLEL_THRESHOLD_QUBITS`.
//! Shot sampling and Pauli expectations run natively per group, so both work
//! past 64 qubits.
//!
//! # When to prefer this backend
//!
//! - Entanglement confined to small qubit groups (parallel subcircuits,
//!   ancilla blocks, sparse couplings). Auto dispatch selects it on partial
//!   independence.
//!
//! # When NOT to use this backend
//!
//! - Circuits that entangle most qubits into one group; the largest group
//!   degenerates to a full statevector plus merge cost.

mod kernels;
#[cfg(test)]
mod tests;

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::{SmallVec, smallvec};

use crate::backend::memory::{dense_probability_len, dense_statevector_len};
use crate::backend::{Backend, BasisSamples, measurement_inv_norm};
use crate::backend::{reduced_density, schmidt, simd};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{DiagEntry, Gate, is_diagonal_2x2};
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

#[cfg(feature = "parallel")]
use crate::backend::statevector::kernels::apply_fused_2q;
use crate::backend::statevector::kernels::{
    apply_cu, apply_cu_phase, apply_cx, apply_cz, apply_diagonal, apply_mcu, apply_mcu_phase,
    apply_parity2q, apply_swap, apply_unitary_seq, rzz_phases,
};
#[cfg(feature = "parallel")]
use crate::backend::statevector::kernels::{
    apply_multi_1q_par, apply_single_gate_par, par_apply_unitary,
};
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ELEMS, PARALLEL_THRESHOLD_QUBITS};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use kernels::*;

type GateList = SmallVec<[(usize, [[Complex64; 2]; 2]); 4]>;

/// Narrowest sub-state a measurement or reset splits. Below it, the split and
/// the merge that usually follows cost more than the halved sweeps return.
const MIN_SPLIT_QUBITS: usize = 7;

macro_rules! seq_or_par {
    ($cond:expr, $seq:expr, $par:expr) => {{
        #[cfg(feature = "parallel")]
        if $cond {
            $par
        } else {
            $seq
        }
        #[cfg(not(feature = "parallel"))]
        $seq
    }};
}

struct SubState {
    state: Vec<Complex64>,
    /// Global qubit indices, sorted ascending. Position = local qubit index.
    qubits: SmallVec<[usize; 8]>,
}

/// Dynamic split-state backend that merges sub-states on demand.
pub struct FactoredBackend {
    num_qubits: usize,
    /// Maps global qubit index → sub-state index in `substates`.
    qubit_to_substate: Vec<usize>,
    /// Active sub-states. Slots become `None` after merge (consumed into another).
    substates: Vec<Option<SubState>>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    limits: ResolvedLimits,
}

/// Apply a fused 2q gate to one sub-state, keeping the general `PreparedGate2q` sweep below
/// the parallel threshold: the statevector kernel's adjacent pair path costs more per call
/// on the small sub-states a factored run grows through.
#[inline(always)]
fn apply_fused_2q_substate(
    state: &mut [Complex64],
    num_qubits: usize,
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    #[cfg(feature = "parallel")]
    if num_qubits >= PARALLEL_THRESHOLD_QUBITS {
        apply_fused_2q(state, num_qubits, q0, q1, mat);
        return;
    }
    simd::PreparedGate2q::new(mat).apply_full(state, num_qubits, q0, q1);
}

impl FactoredBackend {
    /// Backend under the process default [`Limits`](crate::Limits).
    pub fn new(seed: u64) -> Self {
        Self::with_limits(seed, &ResolvedLimits::process_default())
    }

    pub(crate) fn with_limits(seed: u64, limits: &ResolvedLimits) -> Self {
        Self {
            num_qubits: 0,
            qubit_to_substate: Vec::new(),
            substates: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            limits: *limits,
        }
    }

    #[inline(always)]
    fn local_qubit(sub: &SubState, global: usize) -> usize {
        sub.qubits.iter().position(|&q| q == global).unwrap()
    }

    /// Ensure all target qubits reside in the same sub-state, merging as needed.
    /// Returns the sub-state index containing all targets, or the memory-budget
    /// error when a merge would need a dense block over the statevector cap.
    fn ensure_same_substate(&mut self, targets: &[usize]) -> Result<usize> {
        let first_ss = self.qubit_to_substate[targets[0]];
        let mut need_merge: SmallVec<[usize; 4]> = SmallVec::new();

        for &q in &targets[1..] {
            let ss = self.qubit_to_substate[q];
            if ss != first_ss && !need_merge.contains(&ss) {
                need_merge.push(ss);
            }
        }

        for other_ss in need_merge {
            self.merge_substates(first_ss, other_ss)?;
        }

        Ok(first_ss)
    }

    /// Reject a merge whose dense block would exceed the statevector budget.
    /// The transient peak also holds both source blocks, but they are bounded
    /// by prior merges through this same gate, so the merged width carries the
    /// contract.
    fn check_merge_allocation(&self, total_n: usize) -> Result<()> {
        if total_n >= usize::BITS as usize {
            return Err(crate::error::PrismError::ResourceLimit(Box::new(
                crate::error::ResourceLimit {
                    backend: "factored".to_string(),
                    operation: "the dense block of merged sub-states".to_string(),
                    resource: crate::error::ResourceKind::Qubits,
                    required: total_n as u128,
                    limit: usize::BITS as u128 - 1,
                    env_var: None,
                    knob: None,
                    limit_source: crate::limits::LimitSource::Detected,
                },
            )));
        }
        let cap = self.limits.factored_merge_qubits;
        if total_n > cap.value {
            return Err(crate::limits::resource_limit_error(
                "factored",
                "the dense block of merged sub-states",
                crate::error::ResourceKind::Qubits,
                total_n as u128,
                cap.value as u128,
                crate::limits::LimitKnob::FactoredMergeQubits,
                cap.source,
            ));
        }
        Ok(())
    }

    /// Merge sub-state `src_idx` into `dst_idx` via tensor product.
    ///
    /// Both sub-states maintain `qubits` sorted ascending. The merged sub-state
    /// also has sorted qubits. When one set of qubits is wholly less than the
    /// other, the kernel reduces to a SIMD-friendly Kronecker product with a
    /// contiguous inner loop. The interleaved case falls back to scalar scatter.
    fn merge_substates(&mut self, dst_idx: usize, src_idx: usize) -> Result<()> {
        let dst_n = self.substates[dst_idx].as_ref().unwrap().qubits.len();
        let src_n = self.substates[src_idx].as_ref().unwrap().qubits.len();
        let total_n = dst_n + src_n;
        self.check_merge_allocation(total_n)?;

        // A sub-state a split shrank keeps its capacity, so a qubit merging
        // back in grows it in place instead of allocating the product.
        let wide_idx = if src_n > dst_n { src_idx } else { dst_idx };
        let wide = &self.substates[wide_idx].as_ref().unwrap().state;
        if dst_n.min(src_n) == 1 && wide.capacity() >= wide.len() << 1 {
            let mut src = self.substates[src_idx].take().unwrap();
            let dst = self.substates[dst_idx].as_mut().unwrap();
            if src_n > dst_n {
                std::mem::swap(dst, &mut src);
            }
            let q = src.qubits[0];
            let pos = dst.qubits.partition_point(|&g| g < q);
            insert_qubit(&mut dst.state, pos, [src.state[0], src.state[1]]);
            dst.qubits.insert(pos, q);
            for &g in &dst.qubits {
                self.qubit_to_substate[g] = dst_idx;
            }
            return Ok(());
        }

        let src = self.substates[src_idx].take().unwrap();
        let dst = self.substates[dst_idx].as_ref().unwrap();
        let total_dim = 1usize << total_n;

        let mut merged_qubits: SmallVec<[usize; 8]> = SmallVec::with_capacity(total_n);
        let mut dst_bit_positions: SmallVec<[usize; 8]> = SmallVec::new();
        let mut src_bit_positions: SmallVec<[usize; 8]> = SmallVec::new();

        let (mut di, mut si) = (0, 0);
        while di < dst_n || si < src_n {
            if di < dst_n && (si >= src_n || dst.qubits[di] < src.qubits[si]) {
                dst_bit_positions.push(merged_qubits.len());
                merged_qubits.push(dst.qubits[di]);
                di += 1;
            } else {
                src_bit_positions.push(merged_qubits.len());
                merged_qubits.push(src.qubits[si]);
                si += 1;
            }
        }

        let dst_low = dst_n > 0 && (src_n == 0 || dst.qubits[dst_n - 1] < src.qubits[0]);
        let src_low = src_n > 0 && (dst_n == 0 || src.qubits[src_n - 1] < dst.qubits[0]);

        let merged_state = if dst_low {
            kron_low_high(&dst.state, &src.state, dst_n, src_n)
        } else if src_low {
            kron_low_high(&src.state, &dst.state, src_n, dst_n)
        } else {
            kron_scatter(
                &dst.state,
                &src.state,
                &dst_bit_positions,
                &src_bit_positions,
                total_dim,
            )
        };

        let dst = self.substates[dst_idx].as_mut().unwrap();
        dst.state = merged_state;
        dst.qubits = merged_qubits;

        for &q in &src.qubits {
            self.qubit_to_substate[q] = dst_idx;
        }
        Ok(())
    }

    /// Central gate dispatch: translates global qubit indices to local and
    /// applies the sequential or Rayon kernel on the sub-state slice; the
    /// parallel path engages at `PARALLEL_THRESHOLD_QUBITS`.
    #[inline(always)]
    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        if let Gate::MultiFused(data) = gate {
            self.apply_multi_fused(&data.gates);
            return Ok(());
        }
        if let Gate::Multi2q(data) = gate {
            for &(q0, q1, ref mat) in &data.gates {
                let tgts = [q0, q1];
                let ss_idx = self.ensure_same_substate(&tgts)?;
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let n = sub.qubits.len();
                let lq0 = Self::local_qubit(sub, q0);
                let lq1 = Self::local_qubit(sub, q1);
                apply_fused_2q_substate(&mut sub.state, n, lq0, lq1, mat);
            }
            return Ok(());
        }
        if let Gate::BatchRzz(data) = gate {
            for &(q0, q1, theta) in &data.edges {
                self.apply_rzz_edge(q0, q1, theta)?;
            }
            return Ok(());
        }
        if let Gate::DiagonalBatch(data) = gate {
            for entry in &data.entries {
                self.apply_diag_entry(entry)?;
            }
            return Ok(());
        }
        // All other gates require targets in the same sub-state.
        // BatchPhase carries phase target qubits inside its data, not in the
        // instruction-level `targets` list, so the merge must include them.
        let ss_idx = if let Gate::BatchPhase(data) = gate {
            let mut all_qubits: SmallVec<[usize; 16]> = SmallVec::new();
            all_qubits.extend_from_slice(targets);
            for &(q, _) in &data.phases {
                if !all_qubits.contains(&q) {
                    all_qubits.push(q);
                }
            }
            self.ensure_same_substate(&all_qubits)?
        } else {
            self.ensure_same_substate(targets)?
        };

        #[cfg(feature = "parallel")]
        let par = {
            let sub = self.substates[ss_idx].as_ref().unwrap();
            sub.qubits.len() >= PARALLEL_THRESHOLD_QUBITS
        };

        match gate {
            Gate::Rzz(theta) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                let (same, diff) = rzz_phases(*theta);
                apply_parity2q(&mut sub.state, sub.qubits.len(), q0, q1, same, diff);
            }
            Gate::Cx => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                apply_cx(&mut sub.state, sub.qubits.len(), ctrl, tgt);
            }
            Gate::Cz => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_cz(&mut sub.state, sub.qubits.len(), q0, q1);
            }
            Gate::Swap => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_swap(&mut sub.state, sub.qubits.len(), q0, q1);
            }
            Gate::Cu(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                if let Some(phase) = gate.controlled_phase() {
                    apply_cu_phase(&mut sub.state, sub.qubits.len(), ctrl, tgt, phase);
                } else {
                    apply_cu(&mut sub.state, sub.qubits.len(), ctrl, tgt, **mat);
                }
            }
            Gate::Mcu(data) => {
                let nc = data.num_controls as usize;
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local_ctrls: SmallVec<[usize; 4]> = targets[..nc]
                    .iter()
                    .map(|&q| Self::local_qubit(sub, q))
                    .collect();
                let local_tgt = Self::local_qubit(sub, targets[nc]);
                if let Some(phase) = gate.controlled_phase() {
                    apply_mcu_phase(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        phase,
                    );
                } else {
                    apply_mcu(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        data.mat,
                    );
                }
            }
            Gate::Unitary(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local: SmallVec<[usize; 4]> =
                    targets.iter().map(|&q| Self::local_qubit(sub, q)).collect();
                seq_or_par!(
                    par,
                    apply_unitary_seq(&mut sub.state, &local, data.matrix()),
                    par_apply_unitary(&mut sub.state, &local, data.matrix())
                );
            }
            Gate::BatchPhase(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local_ctrl = Self::local_qubit(sub, targets[0]);
                let local_phases: SmallVec<[(usize, Complex64); 8]> = data
                    .phases
                    .iter()
                    .map(|&(gq, ph)| (Self::local_qubit(sub, gq), ph))
                    .collect();
                seq_or_par!(
                    par,
                    apply_batch_phase_seq(
                        &mut sub.state,
                        sub.qubits.len(),
                        local_ctrl,
                        &local_phases,
                    ),
                    par_apply_batch_phase(&mut sub.state, local_ctrl, &local_phases)
                );
            }
            Gate::Fused2q(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_fused_2q_substate(&mut sub.state, sub.qubits.len(), q0, q1, mat);
            }
            Gate::MultiFused(_) | Gate::Multi2q(_) | Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => {
                unreachable!()
            }
            _ => {
                let mat = gate.matrix_2x2();
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local = Self::local_qubit(sub, targets[0]);
                if gate.is_diagonal_1q() {
                    apply_diagonal(
                        &mut sub.state,
                        sub.qubits.len(),
                        local,
                        mat[0][0],
                        mat[1][1],
                    );
                } else {
                    seq_or_par!(
                        par,
                        simd::PreparedGate1q::new(&mat)
                            .apply_full_sequential(&mut sub.state, local),
                        apply_single_gate_par(&mut sub.state, local, &mat)
                    );
                }
            }
        }
        Ok(())
    }

    /// Apply one batched `Rzz` edge, merging only the pair it acts on.
    ///
    /// A batch is built without regard to independence, so merging its whole
    /// qubit union would collapse blocks the edges never connect.
    fn apply_rzz_edge(&mut self, q0: usize, q1: usize, theta: f64) -> Result<()> {
        let ss_idx = self.ensure_same_substate(&[q0, q1])?;
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let lq0 = Self::local_qubit(sub, q0);
        let lq1 = Self::local_qubit(sub, q1);
        let (same, diff) = rzz_phases(theta);
        apply_parity2q(&mut sub.state, sub.qubits.len(), lq0, lq1, same, diff);
        Ok(())
    }

    /// Apply one batched diagonal entry, merging only the qubits it acts on.
    ///
    /// Same independence contract as [`Self::apply_rzz_edge`].
    fn apply_diag_entry(&mut self, entry: &DiagEntry) -> Result<()> {
        let ss_idx = match entry {
            DiagEntry::Phase1q { qubit, .. } => self.ensure_same_substate(&[*qubit])?,
            DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                self.ensure_same_substate(&[*q0, *q1])?
            }
        };
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let n = sub.qubits.len();

        match entry {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                let lq = Self::local_qubit(sub, *qubit);
                apply_diagonal(&mut sub.state, n, lq, *d0, *d1);
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                let lq0 = Self::local_qubit(sub, *q0);
                let lq1 = Self::local_qubit(sub, *q1);
                apply_cu_phase(&mut sub.state, n, lq0, lq1, *phase);
            }
            DiagEntry::Parity2q { q0, q1, same, diff } => {
                let lq0 = Self::local_qubit(sub, *q0);
                let lq1 = Self::local_qubit(sub, *q1);
                apply_parity2q(&mut sub.state, n, lq0, lq1, *same, *diff);
            }
        }
        Ok(())
    }

    /// Apply MultiFused gates grouped by sub-state, no merging needed.
    fn apply_multi_fused(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        let mut groups: SmallVec<[(usize, GateList); 8]> = SmallVec::new();

        for &(global_q, mat) in gates {
            let ss_idx = self.qubit_to_substate[global_q];
            let sub = self.substates[ss_idx].as_ref().unwrap();
            let local = Self::local_qubit(sub, global_q);

            if let Some(entry) = groups.iter_mut().find(|(idx, _)| *idx == ss_idx) {
                entry.1.push((local, mat));
            } else {
                groups.push((ss_idx, smallvec![(local, mat)]));
            }
        }

        for (ss_idx, gate_list) in groups {
            let sub = self.substates[ss_idx].as_mut().unwrap();

            #[cfg(feature = "parallel")]
            if sub.qubits.len() >= PARALLEL_THRESHOLD_QUBITS {
                apply_multi_1q_par(&mut sub.state, &gate_list);
                continue;
            }

            for &(local_tgt, mat) in &gate_list {
                let prepared = simd::PreparedGate1q::new(&mat);
                prepared.apply_full_sequential(&mut sub.state, local_tgt);
            }
        }
    }

    fn apply_reset(&mut self, qubit: usize) {
        self.collapse(qubit, true);
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        self.classical_bits[classical_bit] = self.collapse(qubit, false);
    }

    /// Project `qubit` onto a sampled outcome and return it. A reset then
    /// rotates the qubit to `|0>`.
    ///
    /// The collapsed qubit is exactly product with the rest of its sub-state.
    /// From [`MIN_SPLIT_QUBITS`] up, the outcome branch becomes the shrunken
    /// sub-state and the qubit takes a free slot as a basis state; below it
    /// the sub-state collapses in place, which costs less than the split and
    /// the merge that usually follows.
    fn collapse(&mut self, qubit: usize, reset: bool) -> bool {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        let prob_one = branch_norm_sqr(&sub.state, local);
        let outcome = self.rng.random::<f64>() < prob_one;
        let inv_norm = measurement_inv_norm(outcome, prob_one);
        let lands_on_one = outcome && !reset;

        if sub.qubits.len() < MIN_SPLIT_QUBITS {
            collapse_in_place(&mut sub.state, local, outcome, lands_on_one, inv_norm);
            return outcome;
        }

        extract_branch(&mut sub.state, local, outcome, inv_norm);
        sub.qubits.remove(local);

        let zero = Complex64::new(0.0, 0.0);
        let mut basis = vec![zero; 2];
        basis[lands_on_one as usize] = Complex64::new(1.0, 0.0);
        // `init` made one slot per qubit, and a sub-state of two or more qubits
        // leaves fewer live sub-states than that.
        let slot = self.substates.iter().position(Option::is_none).unwrap();
        self.substates[slot] = Some(SubState {
            state: basis,
            qubits: smallvec![qubit],
        });
        self.qubit_to_substate[qubit] = slot;
        outcome
    }

    /// Append observable `index`'s masks, in local qubit coordinates, to the
    /// groups of the sub-states it touches, one entry per sub-state. Rejects
    /// out-of-range qubits and duplicate factors; `seen` comes back cleared.
    fn push_substate_pauli_masks(
        &self,
        index: usize,
        observable: &[PauliTerm],
        seen: &mut [bool],
        groups: &mut [MaskGroup],
    ) -> Result<()> {
        for term in observable {
            if term.qubit >= self.num_qubits {
                return Err(crate::error::PrismError::InvalidQubit {
                    index: term.qubit,
                    register_size: self.num_qubits,
                });
            }
            if seen[term.qubit] {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "joint Pauli observable has duplicate factor on qubit {}",
                        term.qubit
                    ),
                });
            }
            seen[term.qubit] = true;

            let ss = self.qubit_to_substate[term.qubit];
            let sub = self.substates[ss].as_ref().unwrap();
            let bit = 1usize << Self::local_qubit(sub, term.qubit);
            let group = &mut groups[ss];
            if group.last().is_none_or(|&(owner, _)| owner != index) {
                group.push((index, (0, 0, 0)));
            }
            let entry = &mut group.last_mut().unwrap().1;
            match term.axis {
                PauliAxis::X => entry.0 |= bit,
                PauliAxis::Z => entry.1 |= bit,
                PauliAxis::Y => {
                    entry.0 |= bit;
                    entry.1 |= bit;
                    entry.2 += 1;
                }
            }
        }
        for term in observable {
            seen[term.qubit] = false;
        }
        Ok(())
    }

    /// Schmidt spectrum of each sub-state the cut at `subsystem` splits, one
    /// thin SVD over that block's own amplitudes. A block wholly on one side
    /// is a product factor across the cut and contributes no spectrum.
    fn straddling_spectra(&self, subsystem: &[usize]) -> Result<Vec<Vec<f64>>> {
        let mut inside = vec![false; self.num_qubits];
        for &q in subsystem {
            inside[q] = true;
        }
        let mut spectra = Vec::new();
        for sub in self.substates.iter().flatten() {
            let local: SmallVec<[usize; 8]> = (0..sub.qubits.len())
                .filter(|&i| inside[sub.qubits[i]])
                .collect();
            if local.is_empty() || local.len() == sub.qubits.len() {
                continue;
            }
            spectra.push(schmidt::dense_schmidt_values(
                self.name(),
                &sub.state,
                sub.qubits.len(),
                &local,
                &self.limits,
            )?);
        }
        Ok(spectra)
    }
}

impl crate::backend::sealed::Sealed for FactoredBackend {
    fn limits(&self) -> crate::limits::ResolvedLimits {
        self.limits
    }
}

impl Backend for FactoredBackend {
    fn name(&self) -> &'static str {
        "factored"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::Factored
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        #[cfg(feature = "parallel")]
        crate::backend::init_thread_pool();

        self.num_qubits = num_qubits;
        self.qubit_to_substate.clear();
        self.substates.clear();

        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        for q in 0..num_qubits {
            self.qubit_to_substate.push(q);
            self.substates.push(Some(SubState {
                state: vec![one, zero],
                qubits: smallvec![q],
            }));
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
                Ok(())
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit);
                Ok(())
            }
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Save { label, .. } => {
                Err(crate::backend::save_not_applied("Factored", label))
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets)?;
                }
                Ok(())
            }
            Instruction::Region(region) => self.apply_region(region),
        }
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit);
        Ok(())
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_ref().unwrap();
        let local = Self::local_qubit(sub, qubit);
        let mask = 1usize << local;

        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        let mut r = Complex64::new(0.0, 0.0);
        for idx in 0..sub.state.len() {
            let amp = sub.state[idx];
            if idx & mask == 0 {
                p0 += amp.norm_sqr();
                r += sub.state[idx | mask] * amp.conj();
            } else {
                p1 += amp.norm_sqr();
            }
        }

        Ok([
            [Complex64::new(p0, 0.0), r.conj()],
            [r, Complex64::new(p1, 0.0)],
        ])
    }

    fn supports_two_qubit_kraus(&self) -> bool {
        true
    }

    /// Blocks are exactly unentangled, so the answer is the product of one
    /// partial trace per block that meets `subsystem`, each over
    /// `2^(n_b + k_b)` amplitudes, multiplied in through the bits of the row
    /// and column index that block's qubits occupy, then scaled to trace one.
    fn reduced_density_matrix(&mut self, subsystem: &[usize]) -> Result<Vec<Complex64>> {
        schmidt::validate_qubit_set(subsystem, self.num_qubits)?;
        let dim =
            reduced_density::reduced_density_side(self.name(), subsystem.len(), &self.limits)?;
        let mut rho = vec![Complex64::new(1.0, 0.0); dim * dim];
        for (ss_idx, sub) in self.substates.iter().enumerate() {
            let Some(sub) = sub else { continue };
            let positions: Vec<usize> = (0..subsystem.len())
                .filter(|&i| self.qubit_to_substate[subsystem[i]] == ss_idx)
                .collect();
            if positions.is_empty() {
                continue;
            }
            let local: Vec<usize> = positions
                .iter()
                .map(|&i| Self::local_qubit(sub, subsystem[i]))
                .collect();
            let factor =
                reduced_density::dense_reduced_density(&sub.state, sub.qubits.len(), &local);
            let side = 1usize << local.len();
            reduced_density::multiply_block_factor(&mut rho, dim, &positions, &factor, side);
        }
        reduced_density::normalize_trace(&mut rho, dim);
        Ok(rho)
    }

    /// Sorted Kronecker product of the spectra of the blocks the cut splits,
    /// a single `1` when it splits none. The product is priced against the
    /// dense export cap, past which only [`Backend::entanglement_entropy`]
    /// answers.
    fn schmidt_values(&mut self, subsystem: &[usize]) -> Result<Vec<f64>> {
        schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        let spectra = self.straddling_spectra(subsystem)?;
        let width = spectra
            .iter()
            .map(|s| s.len().next_power_of_two().trailing_zeros() as usize)
            .sum();
        dense_statevector_len(self.name(), "Schmidt values", width, &self.limits)?;
        let mut values = vec![1.0];
        for spectrum in &spectra {
            values = values
                .iter()
                .flat_map(|&a| spectrum.iter().map(move |&b| a * b))
                .collect();
        }
        Ok(schmidt::finish_schmidt_values(values))
    }

    /// Sum of the split blocks' entropies, so the product spectrum is never
    /// built and the export cap bounds only each block's own SVD.
    fn entanglement_entropy(&mut self, subsystem: &[usize]) -> Result<f64> {
        schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        Ok(self
            .straddling_spectra(subsystem)?
            .iter()
            .map(|s| schmidt::entropy_of_schmidt_values(s))
            .sum())
    }

    /// Apply a Kraus branch to one qubit without boxing a `Gate::Fused`.
    ///
    /// The default routes through `apply`, which allocates per call, and a
    /// non-Pauli channel on a factored run reaches here once per damping event.
    /// A single target never merges sub-states, so kernel selection matches the
    /// `Gate::Fused` arm of `dispatch_gate`.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        if is_diagonal_2x2(matrix) {
            let n = sub.qubits.len();
            apply_diagonal(&mut sub.state, n, local, matrix[0][0], matrix[1][1]);
        } else {
            #[cfg(feature = "parallel")]
            if sub.qubits.len() >= PARALLEL_THRESHOLD_QUBITS {
                apply_single_gate_par(&mut sub.state, local, matrix);
                return Ok(());
            }
            simd::PreparedGate1q::new(matrix).apply_full_sequential(&mut sub.state, local);
        }
        Ok(())
    }

    /// Tensor the live sub-states back into a joint amplitude vector.
    ///
    /// The state is a product over blocks, so the joint amplitude at a global
    /// index is the product of each block's amplitude at the index restricted to
    /// that block's qubits. Sub-states stay normalized through measurement, so no
    /// pending norm is carried.
    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(
            self.name(),
            "statevector export",
            self.num_qubits,
            &self.limits,
        )?;

        let active: SmallVec<[&SubState; 16]> = self
            .substates
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        if active.len() == 1 && active[0].qubits.len() == self.num_qubits {
            return Ok(active[0].state.clone());
        }

        let mut joint = vec![Complex64::new(0.0, 0.0); dim];
        for (index, amp) in joint.iter_mut().enumerate() {
            let mut product = Complex64::new(1.0, 0.0);
            for sub in &active {
                let mut local = 0usize;
                for (bit, &q) in sub.qubits.iter().enumerate() {
                    local |= ((index >> q) & 1) << bit;
                }
                product *= sub.state[local];
            }
            *amp = product;
        }
        Ok(joint)
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn set_classical_bit(&mut self, bit: usize, value: bool) -> Result<()> {
        self.classical_bits[bit] = value;
        Ok(())
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        dense_probability_len(self.name(), self.num_qubits, &self.limits)?;
        let active: SmallVec<[&SubState; 16]> = self
            .substates
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        if active.len() == 1 && active[0].qubits.len() == self.num_qubits {
            let st = &active[0].state;
            let mut probs = vec![0.0_f64; st.len()];
            #[cfg(feature = "parallel")]
            if active[0].qubits.len() >= PARALLEL_THRESHOLD_QUBITS {
                let src_chunks = st.par_chunks(MIN_PAR_ELEMS);
                let dst_chunks = probs.par_chunks_mut(MIN_PAR_ELEMS);
                src_chunks.zip(dst_chunks).for_each(|(s, d)| {
                    simd::norm_sqr_to_slice(s, d);
                });
                return Ok(probs);
            }
            simd::norm_sqr_to_slice(st, &mut probs);
            return Ok(probs);
        }

        let blocks: Vec<(Vec<f64>, Vec<usize>)> = active
            .iter()
            .map(|sub| {
                let mut probs = vec![0.0_f64; sub.state.len()];
                simd::norm_sqr_to_slice(&sub.state, &mut probs);
                let qubits: Vec<usize> = sub.qubits.to_vec();
                (probs, qubits)
            })
            .collect();

        Ok(crate::sim::merge_probabilities(&blocks, self.num_qubits))
    }

    fn block_probabilities(&self) -> Option<crate::sim::Probabilities> {
        // `FactoredBlock::mask` is a u64 and `Probabilities::len` is
        // `1 << total_qubits`, so a wider register has no representable lazy
        // form either; the dense terminal declines it.
        if self.num_qubits > 64 {
            return None;
        }

        let active: SmallVec<[&SubState; 16]> = self
            .substates
            .iter()
            .filter_map(|opt| opt.as_ref())
            .collect();

        if active.len() < 2 {
            return None;
        }

        let blocks = active
            .iter()
            .map(|sub| {
                let mut probs = vec![0.0_f64; sub.state.len()];
                simd::norm_sqr_to_slice(&sub.state, &mut probs);
                let mask = sub.qubits.iter().fold(0u64, |m, &q| m | 1 << q);
                crate::sim::FactoredBlock { probs, mask }
            })
            .collect();

        Some(crate::sim::Probabilities::Factored {
            blocks,
            total_qubits: self.num_qubits,
        })
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Draws one local index per sub-state and concatenates them, so the cost
    /// is the sum of the block dimensions rather than their product.
    ///
    /// Shots draw in blocks of 256, one ChaCha8 stream per block, and within a
    /// shot the sub-states are visited in slot order with one draw each: the
    /// streams, order and count the dense factored sampler uses, so the two
    /// agree shot for shot at any thread count.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let blocks: Vec<(Vec<f64>, &[usize])> = self
            .substates
            .iter()
            .filter_map(|opt| opt.as_ref())
            .map(|sub| {
                let mut probs = vec![0.0_f64; sub.state.len()];
                simd::norm_sqr_to_slice(&sub.state, &mut probs);
                (crate::sim::shots::build_cdf(&probs), sub.qubits.as_slice())
            })
            .collect();

        use crate::sim::shots::{
            draws_split_across_workers, sample_from_cdf, sample_in_shot_blocks,
        };

        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        let words_per_shot = samples.words_per_shot();
        let parallel = draws_split_across_workers(num_shots, blocks.len());
        sample_in_shot_blocks(
            samples.words_mut(),
            words_per_shot,
            seed,
            parallel,
            |rng, block_words| {
                for shot_words in block_words.chunks_mut(words_per_shot) {
                    for (cdf, qubits) in &blocks {
                        let r: f64 = rng.random();
                        let local = sample_from_cdf(cdf, r);
                        for (bit, &qubit) in qubits.iter().enumerate() {
                            if (local >> bit) & 1 == 1 {
                                shot_words[qubit / 64] |= 1u64 << (qubit % 64);
                            }
                        }
                    }
                }
            },
        );
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// A joint Pauli factorizes across independent sub-states, so the value is
    /// the product of the per-block expectations. Blocks the observable does
    /// not touch contribute one and are skipped.
    ///
    /// Every observable that touches a given sub-state is evaluated in one
    /// traversal of it, so the block is read once rather than once per
    /// observable. The cross-block product stays per observable.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let mut groups: Vec<MaskGroup> = vec![Vec::new(); self.substates.len()];
        let mut seen = vec![false; self.num_qubits];
        for (index, observable) in observables.iter().enumerate() {
            self.push_substate_pauli_masks(index, observable, &mut seen, &mut groups)?;
        }

        let mut products = vec![1.0f64; observables.len()];
        let mut masks: Vec<(usize, usize, u32)> = Vec::new();
        for (ss, group) in groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let sub = self.substates[ss].as_ref().unwrap();
            masks.clear();
            masks.extend(group.iter().map(|&(_, entry)| entry));
            let (values, _) = crate::sim::pauli_expectations_from_masks(&sub.state, &masks);
            for (&(index, _), value) in group.iter().zip(values) {
                products[index] *= value;
            }
        }
        Ok(products)
    }
}

/// The strings reaching one sub-state, as `(string index, (x mask, z mask, Y
/// count))` in string order.
type MaskGroup = Vec<(usize, (usize, usize, u32))>;
