//! Factored statevector backend.
//!
//! Maintains separate sub-state vectors per entangled qubit group. When a
//! multi-qubit gate bridges two groups, they merge via tensor product. Groups
//! never split. For sparse-entanglement circuits this is exponentially cheaper
//! than a monolithic 2^n statevector.
//!
//! # Memory layout
//!
//! - One dense `Vec<Complex64>` per group, length 2^k for k group qubits;
//!   total cost is the sum over groups, dominated by the largest.
//! - Each group lists its global qubits sorted ascending; position = local
//!   qubit index. A global map resolves qubit to group.
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

#[cfg(test)]
mod tests;

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::{SmallVec, smallvec};

use crate::backend::memory::{dense_probability_len, dense_statevector_len};
use crate::backend::simd;
use crate::backend::statevector::insert_zero_bit;
use crate::backend::{
    Backend, BasisSamples, MCU_QUBIT_BUF, is_phase_one, measurement_inv_norm, sorted_mcu_qubits,
};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{DiagEntry, Gate, is_diagonal_2x2};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

#[cfg(feature = "parallel")]
use crate::backend::statevector::SendPtr;
#[cfg(feature = "parallel")]
use crate::backend::{
    MIN_PAR_ELEMS, MIN_PAR_ITERS, PARALLEL_THRESHOLD_QUBITS, chunk_min_len as par_chunk_min_len,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

type GateList = SmallVec<[(usize, [[Complex64; 2]; 2]); 4]>;

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
}

impl FactoredBackend {
    /// Create a new factored backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            qubit_to_substate: Vec::new(),
            substates: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Translate a global qubit index to a local index within its sub-state.
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
    fn check_merge_allocation(total_n: usize) -> Result<()> {
        if total_n >= usize::BITS as usize {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "factored".to_string(),
                reason: format!(
                    "merging entangled sub-states needs a {total_n}-qubit dense block, \
                     exceeding addressable memory"
                ),
            });
        }
        let cap = crate::backend::max_factored_merge_qubits();
        if total_n > cap {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "factored".to_string(),
                reason: format!(
                    "merging entangled sub-states needs a {total_n}-qubit dense block, \
                     exceeding the cap of {cap} on this machine \
                     (set PRISM_MAX_FACTORED_MERGE_QUBITS to override)"
                ),
            });
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
        Self::check_merge_allocation(total_n)?;

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

    /// Central gate dispatch. Translates global qubit indices to local and
    /// applies the sequential or Rayon kernel on the sub-state slice; the
    /// parallel path engages at `PARALLEL_THRESHOLD_QUBITS`.
    #[inline(always)]
    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        if let Gate::MultiFused(data) = gate {
            self.apply_multi_fused(&data.gates, data.all_diagonal);
            return Ok(());
        }
        if let Gate::Multi2q(data) = gate {
            for &(q0, q1, ref mat) in &data.gates {
                let tgts = [q0, q1];
                let ss_idx = self.ensure_same_substate(&tgts)?;
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let lq0 = Self::local_qubit(sub, q0);
                let lq1 = Self::local_qubit(sub, q1);
                let prepared = simd::PreparedGate2q::new(mat);
                prepared.apply_full(&mut sub.state, sub.qubits.len(), lq0, lq1);
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

        macro_rules! seq_or_par {
            ($seq:expr, $par:expr) => {{
                #[cfg(feature = "parallel")]
                if par {
                    $par
                } else {
                    $seq
                }
                #[cfg(not(feature = "parallel"))]
                $seq
            }};
        }

        match gate {
            Gate::Rzz(theta) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                seq_or_par!(
                    apply_rzz_seq(&mut sub.state, q0, q1, *theta),
                    par_apply_rzz(&mut sub.state, q0, q1, *theta)
                );
            }
            Gate::Cx => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                seq_or_par!(
                    apply_cx_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt),
                    par_apply_cx(&mut sub.state, ctrl, tgt)
                );
            }
            Gate::Cz => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                seq_or_par!(
                    apply_cz_seq(&mut sub.state, sub.qubits.len(), q0, q1),
                    par_apply_cz(&mut sub.state, q0, q1)
                );
            }
            Gate::Swap => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                seq_or_par!(
                    apply_swap_seq(&mut sub.state, sub.qubits.len(), q0, q1),
                    par_apply_swap(&mut sub.state, q0, q1)
                );
            }
            Gate::Cu(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                if let Some(phase) = gate.controlled_phase() {
                    seq_or_par!(
                        apply_cu_phase_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt, phase),
                        par_apply_cu_phase(&mut sub.state, sub.qubits.len(), ctrl, tgt, phase)
                    );
                } else {
                    seq_or_par!(
                        apply_cu_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt, **mat),
                        par_apply_cu(&mut sub.state, sub.qubits.len(), ctrl, tgt, **mat)
                    );
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
                    seq_or_par!(
                        apply_mcu_phase_seq(
                            &mut sub.state,
                            sub.qubits.len(),
                            &local_ctrls,
                            local_tgt,
                            phase,
                        ),
                        par_apply_mcu_phase(
                            &mut sub.state,
                            sub.qubits.len(),
                            &local_ctrls,
                            local_tgt,
                            phase,
                        )
                    );
                } else {
                    seq_or_par!(
                        apply_mcu_seq(
                            &mut sub.state,
                            sub.qubits.len(),
                            &local_ctrls,
                            local_tgt,
                            data.mat,
                        ),
                        par_apply_mcu(
                            &mut sub.state,
                            sub.qubits.len(),
                            &local_ctrls,
                            local_tgt,
                            data.mat,
                        )
                    );
                }
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
                    apply_batch_phase_seq(
                        &mut sub.state,
                        sub.qubits.len(),
                        local_ctrl,
                        &local_phases,
                    ),
                    par_apply_batch_phase(&mut sub.state, local_ctrl, &local_phases)
                );
            }
            Gate::BatchRzz(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for &(q0, q1, theta) in &data.edges {
                    let lq0 = Self::local_qubit(sub, q0);
                    let lq1 = Self::local_qubit(sub, q1);
                    seq_or_par!(
                        apply_rzz_seq(&mut sub.state, lq0, lq1, theta),
                        par_apply_rzz(&mut sub.state, lq0, lq1, theta)
                    );
                }
            }
            Gate::DiagonalBatch(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for entry in &data.entries {
                    match entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            let lq = Self::local_qubit(sub, *qubit);
                            let skip_lo = (d0.re - 1.0).abs() < 1e-15 && d0.im.abs() < 1e-15;
                            seq_or_par!(
                                simd::apply_diagonal_sequential(
                                    &mut sub.state,
                                    lq,
                                    *d0,
                                    *d1,
                                    skip_lo,
                                ),
                                par_apply_diagonal(&mut sub.state, lq, *d0, *d1, skip_lo)
                            );
                        }
                        DiagEntry::Phase2q { q0, q1, phase } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            seq_or_par!(
                                apply_cu_phase_seq(
                                    &mut sub.state,
                                    sub.qubits.len(),
                                    lq0,
                                    lq1,
                                    *phase,
                                ),
                                par_apply_cu_phase(
                                    &mut sub.state,
                                    sub.qubits.len(),
                                    lq0,
                                    lq1,
                                    *phase,
                                )
                            );
                        }
                        DiagEntry::Parity2q { q0, q1, same, diff } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            seq_or_par!(
                                apply_parity2q_seq(&mut sub.state, lq0, lq1, *same, *diff),
                                par_apply_parity2q(&mut sub.state, lq0, lq1, *same, *diff)
                            );
                        }
                    }
                }
            }
            Gate::Fused2q(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                seq_or_par!(
                    simd::PreparedGate2q::new(mat).apply_full(
                        &mut sub.state,
                        sub.qubits.len(),
                        q0,
                        q1,
                    ),
                    par_apply_fused2q(&mut sub.state, sub.qubits.len(), q0, q1, mat)
                );
            }
            Gate::MultiFused(_) | Gate::Multi2q(_) => unreachable!(),
            _ => {
                let mat = gate.matrix_2x2();
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local = Self::local_qubit(sub, targets[0]);
                if gate.is_diagonal_1q() {
                    let skip_lo = is_phase_one(mat[0][0]);
                    seq_or_par!(
                        simd::apply_diagonal_sequential(
                            &mut sub.state,
                            local,
                            mat[0][0],
                            mat[1][1],
                            skip_lo,
                        ),
                        par_apply_diagonal(&mut sub.state, local, mat[0][0], mat[1][1], skip_lo)
                    );
                } else {
                    seq_or_par!(
                        simd::PreparedGate1q::new(&mat)
                            .apply_full_sequential(&mut sub.state, local),
                        par_apply_1q(&mut sub.state, local, &mat)
                    );
                }
            }
        }
        Ok(())
    }

    /// Apply MultiFused gates grouped by sub-state, no merging needed.
    fn apply_multi_fused(&mut self, gates: &[(usize, [[Complex64; 2]; 2])], _all_diagonal: bool) {
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
                par_apply_multi_1q(&mut sub.state, &gate_list);
                continue;
            }

            for &(local_tgt, mat) in &gate_list {
                let prepared = simd::PreparedGate1q::new(&mat);
                prepared.apply_full_sequential(&mut sub.state, local_tgt);
            }
        }
    }

    fn apply_reset(&mut self, qubit: usize) {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        let mask = 1usize << local;
        let n = sub.state.len();
        let zero = Complex64::new(0.0, 0.0);

        let mut prob_one = 0.0f64;
        for i in 0..n {
            if (i & mask) != 0 {
                prob_one += sub.state[i].norm_sqr();
            }
        }

        let outcome = self.rng.random::<f64>() < prob_one;
        let inv_norm = measurement_inv_norm(outcome, prob_one);

        for i in 0..n {
            if (i & mask) != 0 {
                continue;
            }
            let source = if outcome { i | mask } else { i };
            sub.state[i] = sub.state[source] * inv_norm;
            sub.state[i | mask] = zero;
        }
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        let mask = 1usize << local;
        let n = sub.state.len();

        let mut prob_one = 0.0f64;
        for i in 0..n {
            if (i & mask) != 0 {
                prob_one += sub.state[i].norm_sqr();
            }
        }

        let outcome = self.rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;

        let inv_norm = measurement_inv_norm(outcome, prob_one);
        let zero = Complex64::new(0.0, 0.0);

        for i in 0..n {
            let bit_set = (i & mask) != 0;
            if bit_set == outcome {
                sub.state[i] *= inv_norm;
            } else {
                sub.state[i] = zero;
            }
        }
    }

    /// Reduce a joint Pauli observable to per-sub-state masks in local qubit
    /// coordinates. Rejects out-of-range qubits and duplicate factors.
    fn substate_pauli_masks(&self, observable: &[PauliTerm]) -> Result<Vec<(usize, usize, u32)>> {
        let mut masks = vec![(0usize, 0usize, 0u32); self.substates.len()];
        let mut seen = vec![false; self.num_qubits];
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
            let entry = &mut masks[ss];
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
        Ok(masks)
    }
}

impl Backend for FactoredBackend {
    fn name(&self) -> &'static str {
        "factored"
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

    /// Apply a Kraus branch to one qubit without boxing a `Gate::Fused`.
    ///
    /// The default routes through `apply`, which allocates per call, and this is
    /// the one override a live path depends on: a non-Pauli channel on a factored
    /// run reaches here once per damping event. A single target never merges
    /// sub-states, so kernel selection matches the `Gate::Fused` arm of
    /// `dispatch_gate` exactly.
    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        #[cfg(feature = "parallel")]
        let par = sub.qubits.len() >= PARALLEL_THRESHOLD_QUBITS;

        if is_diagonal_2x2(matrix) {
            let skip_lo = is_phase_one(matrix[0][0]);
            #[cfg(feature = "parallel")]
            if par {
                par_apply_diagonal(&mut sub.state, local, matrix[0][0], matrix[1][1], skip_lo);
                return Ok(());
            }
            simd::apply_diagonal_sequential(
                &mut sub.state,
                local,
                matrix[0][0],
                matrix[1][1],
                skip_lo,
            );
        } else {
            #[cfg(feature = "parallel")]
            if par {
                par_apply_1q(&mut sub.state, local, matrix);
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
        let dim = dense_statevector_len(self.name(), "statevector export", self.num_qubits)?;

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

    fn probabilities(&self) -> Result<Vec<f64>> {
        dense_probability_len(self.name(), self.num_qubits)?;
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
    /// Blocks are visited in slot order and each consumes one draw per shot,
    /// which is the order and the count the dense factored sampler uses.
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

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        for shot in 0..num_shots {
            for (cdf, qubits) in &blocks {
                let r: f64 = rng.random();
                let local = crate::sim::shots::sample_from_cdf(cdf, r);
                for (bit, &qubit) in qubits.iter().enumerate() {
                    if (local >> bit) & 1 == 1 {
                        samples.set(shot, qubit);
                    }
                }
            }
        }
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// A joint Pauli factorizes across independent sub-states, so the value is
    /// the product of the per-block expectations. Blocks the observable does
    /// not touch contribute one and are skipped.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let norms: Vec<f64> = self
            .substates
            .iter()
            .map(|slot| {
                slot.as_ref()
                    .map_or(0.0, |sub| crate::backend::state_norm_sqr(&sub.state))
            })
            .collect();

        observables
            .iter()
            .map(|observable| {
                let masks = self.substate_pauli_masks(observable)?;
                let mut product = 1.0f64;
                for (ss, slot) in self.substates.iter().enumerate() {
                    let Some(sub) = slot else { continue };
                    let (xmask, zmask, num_y) = masks[ss];
                    if xmask == 0 && zmask == 0 {
                        continue;
                    }
                    product *= crate::sim::pauli_expectation_from_masks(
                        &sub.state, xmask, zmask, num_y, norms[ss],
                    );
                }
                Ok(product)
            })
            .collect()
    }
}

/// Kronecker product where `low_state` occupies the low `low_n` bits of the
/// merged index and `high_state` occupies the upper `high_n` bits.
///
/// `merged[h * low_dim + l] = low_state[l] * high_state[h]`. Inner loop is a
/// contiguous SIMD scaled copy, parallelized across `h` for large merges.
fn kron_low_high(
    low_state: &[Complex64],
    high_state: &[Complex64],
    low_n: usize,
    high_n: usize,
) -> Vec<Complex64> {
    let low_dim = 1usize << low_n;
    let high_dim = 1usize << high_n;
    let total_dim = low_dim * high_dim;
    let mut merged = vec![Complex64::new(0.0, 0.0); total_dim];

    #[cfg(feature = "parallel")]
    if (low_n + high_n) >= PARALLEL_THRESHOLD_QUBITS {
        merged
            .par_chunks_mut(low_dim)
            .with_min_len(par_chunk_min_len(low_dim))
            .enumerate()
            .for_each(|(h, chunk)| {
                simd::scale_complex_to_slice(chunk, low_state, high_state[h]);
            });
        return merged;
    }

    for (h, chunk) in merged.chunks_mut(low_dim).enumerate() {
        simd::scale_complex_to_slice(chunk, low_state, high_state[h]);
    }
    merged
}

/// General Kronecker product where dst and src bits are interleaved across
/// the merged index. Per-element scatter, no SIMD inner loop.
fn kron_scatter(
    dst_state: &[Complex64],
    src_state: &[Complex64],
    dst_bit_positions: &[usize],
    src_bit_positions: &[usize],
    total_dim: usize,
) -> Vec<Complex64> {
    let mut merged = vec![Complex64::new(0.0, 0.0); total_dim];

    for (merged_idx, amp) in merged.iter_mut().enumerate() {
        let mut dst_local = 0usize;
        for (local_bit, &merged_bit) in dst_bit_positions.iter().enumerate() {
            dst_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
        }
        let mut src_local = 0usize;
        for (local_bit, &merged_bit) in src_bit_positions.iter().enumerate() {
            src_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
        }
        *amp = dst_state[dst_local] * src_state[src_local];
    }
    merged
}

#[inline(always)]
fn apply_cx_seq(state: &mut [Complex64], num_qubits: usize, control: usize, target: usize) {
    let ctrl_mask = 1usize << control;
    let tgt_mask = 1usize << target;
    let n = 1usize << num_qubits;

    for i in 0..n {
        if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
            state.swap(i, i | tgt_mask);
        }
    }
}

#[inline(always)]
fn apply_rzz_seq(state: &mut [Complex64], q0: usize, q1: usize, theta: f64) {
    let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
    let phases = [phase_same, phase_diff];

    for (i, amp) in state.iter_mut().enumerate() {
        let parity = ((i >> q0) ^ (i >> q1)) & 1;
        *amp *= phases[parity];
    }
}

#[inline(always)]
fn apply_parity2q_seq(
    state: &mut [Complex64],
    q0: usize,
    q1: usize,
    same: Complex64,
    diff: Complex64,
) {
    let phases = [same, diff];
    for (i, amp) in state.iter_mut().enumerate() {
        let parity = ((i >> q0) ^ (i >> q1)) & 1;
        *amp *= phases[parity];
    }
}

#[cfg(feature = "parallel")]
fn par_apply_parity2q(
    state: &mut [Complex64],
    q0: usize,
    q1: usize,
    same: Complex64,
    diff: Complex64,
) {
    let phases = [same, diff];
    state
        .par_chunks_mut(MIN_PAR_ELEMS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * MIN_PAR_ELEMS;
            for (j, amp) in chunk.iter_mut().enumerate() {
                let i = base + j;
                let parity = ((i >> q0) ^ (i >> q1)) & 1;
                *amp *= phases[parity];
            }
        });
}

fn apply_cz_seq(state: &mut [Complex64], num_qubits: usize, q0: usize, q1: usize) {
    let mask0 = 1usize << q0;
    let mask1 = 1usize << q1;
    let n = 1usize << num_qubits;

    for (i, amp) in state.iter_mut().enumerate().take(n) {
        if (i & mask0) != 0 && (i & mask1) != 0 {
            *amp = -*amp;
        }
    }
}

#[inline(always)]
fn apply_swap_seq(state: &mut [Complex64], _num_qubits: usize, q0: usize, q1: usize) {
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    let hi_half = 1usize << hi;
    let block_size = hi_half << 1;

    for chunk in state.chunks_mut(block_size) {
        let (lo_group, hi_group) = chunk.split_at_mut(hi_half);
        for (lo_sub, hi_sub) in lo_group
            .chunks_mut(lo_block)
            .zip(hi_group.chunks_mut(lo_block))
        {
            let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
            let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
            simd::swap_slices(lo_sub_hi, hi_sub_lo);
        }
    }
}

#[inline(always)]
fn apply_cu_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let prepared = simd::PreparedGate1q::new(&mat);

    if control > target {
        let ctrl_half = 1usize << control;
        let block_size = ctrl_half << 1;
        for chunk in state.chunks_mut(block_size) {
            let (_, hi) = chunk.split_at_mut(ctrl_half);
            prepared.apply_full_sequential(hi, target);
        }
    } else {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let num_iters = 1usize << (num_qubits - 2);
        let base_ptr = state.as_mut_ptr() as *mut f64;
        for i in 0..num_iters {
            let base = insert_zero_bit(insert_zero_bit(i, control), target);
            let idx0 = base | ctrl_mask;
            let idx1 = idx0 | tgt_mask;
            // SAFETY: indices from insert_zero_bit bijection are in-bounds and disjoint.
            unsafe {
                prepared.apply_pair_ptr(base_ptr.add(idx0 * 2), base_ptr.add(idx1 * 2));
            }
        }
    }
}

#[inline(always)]
fn apply_cu_phase_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
    phase: Complex64,
) {
    let (lo, hi) = if control < target {
        (control, target)
    } else {
        (target, control)
    };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    let hi_half = 1usize << hi;
    let block_size = hi_half << 1;

    let n = 1usize << num_qubits;
    for start in (0..n).step_by(block_size) {
        let hi_start = start + hi_half;
        for sub_start in (hi_start..hi_start + hi_half).step_by(lo_block) {
            let range_start = sub_start + lo_half;
            let range_end = range_start + lo_half;
            for amp in &mut state[range_start..range_end] {
                *amp *= phase;
            }
        }
    }
}

#[inline(always)]
fn apply_mcu_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
    let tgt_mask = 1usize << target;
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let prepared = simd::PreparedGate1q::new(&mat);
    let base_ptr = state.as_mut_ptr() as *mut f64;

    for i in 0..num_iters {
        let mut base = i;
        for &q in sorted {
            base = insert_zero_bit(base, q);
        }
        let idx0 = base | ctrl_mask;
        let idx1 = idx0 | tgt_mask;
        // SAFETY: indices from insert_zero_bit bijection are in-bounds and disjoint.
        unsafe {
            prepared.apply_pair_ptr(base_ptr.add(idx0 * 2), base_ptr.add(idx1 * 2));
        }
    }
}

#[inline(always)]
fn apply_mcu_phase_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    phase: Complex64,
) {
    let all_mask: usize = controls
        .iter()
        .chain(std::iter::once(&target))
        .map(|&q| 1usize << q)
        .fold(0, |a, b| a | b);
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    for i in 0..num_iters {
        let mut base = i;
        for &q in sorted {
            base = insert_zero_bit(base, q);
        }
        state[base | all_mask] *= phase;
    }
}

/// Scalar batch-phase kernel for sub-state slices.
///
/// For sub-states in the factored backend (typically < 14 qubits), the
/// per-element phase accumulation is fast enough without a LUT.
#[inline(always)]
fn apply_batch_phase_seq(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    phases: &[(usize, Complex64)],
) {
    let ctrl_mask = 1usize << control;
    let one = Complex64::new(1.0, 0.0);
    let n = 1usize << num_qubits;

    for (i, amp) in state.iter_mut().enumerate().take(n) {
        if (i & ctrl_mask) == 0 {
            continue;
        }
        let mut combined = one;
        for &(tgt, phase) in phases {
            if (i >> tgt) & 1 != 0 {
                combined *= phase;
            }
        }
        if !is_phase_one(combined) {
            *amp *= combined;
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_1q(state: &mut [Complex64], target: usize, mat: &[[Complex64; 2]; 2]) {
    let half = 1usize << target;
    let block_size = half << 1;
    let prepared = simd::PreparedGate1q::new(mat);

    const MIN_TILE: usize = 8192;
    let tile_size = MIN_TILE.max(block_size);
    let num_tiles = state.len() / tile_size;

    if block_size <= MIN_TILE && num_tiles >= 4 {
        state.par_chunks_mut(MIN_TILE).for_each(|tile| {
            prepared.apply_full_sequential(tile, target);
        });
    } else if num_tiles >= 4 {
        state.par_chunks_mut(block_size).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(half);
            prepared.apply(lo, hi);
        });
    } else {
        let sub_tile = MIN_TILE.min(half);
        for block in state.chunks_mut(block_size) {
            let (lo, hi) = block.split_at_mut(half);
            lo.par_chunks_mut(sub_tile)
                .zip(hi.par_chunks_mut(sub_tile))
                .for_each(|(lo_t, hi_t)| {
                    prepared.apply(lo_t, hi_t);
                });
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_diagonal(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    const MIN_TILE: usize = 8192;
    let half = 1usize << target;
    let block_size = half << 1;
    let tile_size = MIN_TILE.max(block_size);

    state.par_chunks_mut(tile_size).for_each(|tile| {
        simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
    });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_cx(state: &mut [Complex64], control: usize, target: usize) {
    if control > target {
        let ctrl_half = 1usize << control;
        let block_size = ctrl_half << 1;
        let tgt_half = 1usize << target;
        let tgt_block = tgt_half << 1;

        state
            .par_chunks_mut(block_size)
            .with_min_len(par_chunk_min_len(block_size))
            .for_each(|chunk| {
                let (_, hi) = chunk.split_at_mut(ctrl_half);
                for sub in hi.chunks_mut(tgt_block) {
                    let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                    simd::swap_slices(sub_lo, sub_hi);
                }
            });
    } else {
        let tgt_half = 1usize << target;
        let block_size = tgt_half << 1;
        let ctrl_mask = 1usize << control;

        state
            .par_chunks_mut(block_size)
            .with_min_len(par_chunk_min_len(block_size))
            .for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(tgt_half);
                for k in 0..tgt_half {
                    if k & ctrl_mask != 0 {
                        std::mem::swap(&mut lo[k], &mut hi[k]);
                    }
                }
            });
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_rzz(state: &mut [Complex64], q0: usize, q1: usize, theta: f64) {
    let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
    let phases = [phase_same, phase_diff];

    state
        .par_chunks_mut(MIN_PAR_ELEMS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * MIN_PAR_ELEMS;
            for (j, amp) in chunk.iter_mut().enumerate() {
                let i = base + j;
                let parity = ((i >> q0) ^ (i >> q1)) & 1;
                *amp *= phases[parity];
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_cz(state: &mut [Complex64], q0: usize, q1: usize) {
    let (lo_q, hi_q) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let lo_half = 1usize << lo_q;
    let lo_block = lo_half << 1;
    let hi_half = 1usize << hi_q;
    let block_size = hi_half << 1;

    state
        .par_chunks_mut(block_size)
        .with_min_len(par_chunk_min_len(block_size))
        .for_each(|chunk| {
            let (_, hi_group) = chunk.split_at_mut(hi_half);
            for sub in hi_group.chunks_mut(lo_block) {
                let (_, sub_hi) = sub.split_at_mut(lo_half);
                simd::negate_slice(sub_hi);
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_swap(state: &mut [Complex64], q0: usize, q1: usize) {
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    let hi_half = 1usize << hi;
    let block_size = hi_half << 1;

    state
        .par_chunks_mut(block_size)
        .with_min_len(par_chunk_min_len(block_size))
        .for_each(|chunk| {
            let (lo_group, hi_group) = chunk.split_at_mut(hi_half);
            let lo_subs = lo_group.chunks_mut(lo_block);
            let hi_subs = hi_group.chunks_mut(lo_block);
            for (lo_sub, hi_sub) in lo_subs.zip(hi_subs) {
                let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
                let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
                simd::swap_slices(lo_sub_hi, hi_sub_lo);
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_cu(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let prepared = simd::PreparedGate1q::new(&mat);

    if control > target {
        let ctrl_half = 1usize << control;
        let block_size = ctrl_half << 1;

        state
            .par_chunks_mut(block_size)
            .with_min_len(par_chunk_min_len(block_size))
            .for_each(|chunk| {
                let (_, hi) = chunk.split_at_mut(ctrl_half);
                prepared.apply_full_sequential(hi, target);
            });
    } else {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let num_iters = 1usize << (num_qubits - 2);
        let ptr = SendPtr(state.as_mut_ptr());

        // SAFETY: insert_zero_bit bijection produces disjoint index pairs.
        (0..num_iters)
            .into_par_iter()
            .with_min_len(MIN_PAR_ITERS)
            .for_each(move |i| {
                let base = insert_zero_bit(insert_zero_bit(i, control), target);
                let idx0 = base | ctrl_mask;
                let idx1 = idx0 | tgt_mask;
                // SAFETY: idx0 and idx1 are in bounds and unique for this iteration.
                // The outer iterator maps every pair once, so Rayon tasks do not alias.
                unsafe {
                    let fp = ptr.as_f64_ptr();
                    prepared.apply_pair_ptr(fp.add(idx0 * 2), fp.add(idx1 * 2));
                }
            });
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_cu_phase(
    state: &mut [Complex64],
    _num_qubits: usize,
    control: usize,
    target: usize,
    phase: Complex64,
) {
    let (lo, hi) = if control < target {
        (control, target)
    } else {
        (target, control)
    };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    let hi_half = 1usize << hi;
    let block_size = hi_half << 1;

    state
        .par_chunks_mut(block_size)
        .with_min_len(par_chunk_min_len(block_size))
        .for_each(|chunk| {
            let hi_group = &mut chunk[hi_half..];
            for sub in hi_group.chunks_mut(lo_block) {
                simd::scale_complex_slice(&mut sub[lo_half..], phase);
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_mcu(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
    let tgt_mask = 1usize << target;
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let ptr = SendPtr(state.as_mut_ptr());
    let prepared = simd::PreparedGate1q::new(&mat);

    // SAFETY: insert_zero_bit bijection produces disjoint index pairs.
    (0..num_iters)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |i| {
            let mut base = i;
            for &q in sorted {
                base = insert_zero_bit(base, q);
            }
            let idx0 = base | ctrl_mask;
            let idx1 = idx0 | tgt_mask;
            // SAFETY: idx0 and idx1 are in bounds and unique for this iteration.
            // The outer iterator maps every pair once, so Rayon tasks do not alias.
            unsafe {
                let fp = ptr.as_f64_ptr();
                prepared.apply_pair_ptr(fp.add(idx0 * 2), fp.add(idx1 * 2));
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_mcu_phase(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    phase: Complex64,
) {
    let all_mask: usize = controls
        .iter()
        .chain(std::iter::once(&target))
        .map(|&q| 1usize << q)
        .fold(0, |a, b| a | b);
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let ptr = SendPtr(state.as_mut_ptr());

    // SAFETY: insert_zero_bit bijection produces disjoint indices.
    (0..num_iters)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |i| {
            let mut base = i;
            for &q in sorted {
                base = insert_zero_bit(base, q);
            }
            let idx = base | all_mask;
            // SAFETY: idx is in bounds and unique for this iteration. The
            // insert_zero_bit mapping excludes all special qubits before masks are set.
            unsafe {
                let val = ptr.load(idx);
                ptr.store(idx, val * phase);
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_batch_phase(state: &mut [Complex64], control: usize, phases: &[(usize, Complex64)]) {
    let ctrl_half = 1usize << control;
    let block_size = ctrl_half << 1;
    let one = Complex64::new(1.0, 0.0);

    state
        .par_chunks_mut(block_size)
        .with_min_len(par_chunk_min_len(block_size))
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let block_start = chunk_idx * block_size;
            let (_, hi) = chunk.split_at_mut(ctrl_half);

            for (local_i, amp) in hi.iter_mut().enumerate() {
                let i = block_start + ctrl_half + local_i;
                let mut combined = one;
                for &(tgt, phase) in phases {
                    if (i >> tgt) & 1 != 0 {
                        combined *= phase;
                    }
                }
                if !is_phase_one(combined) {
                    *amp *= combined;
                }
            }
        });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_apply_fused2q(
    state: &mut [Complex64],
    num_qubits: usize,
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    let mask0 = 1usize << q0;
    let mask1 = 1usize << q1;
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let n_iter = 1usize << (num_qubits - 2);
    let ptr = SendPtr(state.as_mut_ptr());
    let prepared = simd::PreparedGate2q::new(mat);

    // SAFETY: insert_zero_bit bijection produces disjoint index groups.
    (0..n_iter)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |k| {
            let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
            let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
            // SAFETY: the four indices are in bounds and the pair of inserted
            // zero bits gives each iteration a disjoint 4-amplitude group.
            unsafe {
                prepared.apply_group_ptr(ptr.as_f64_ptr(), i);
            }
        });
}

#[cfg(feature = "parallel")]
fn par_apply_multi_1q(state: &mut [Complex64], gates: &[(usize, [[Complex64; 2]; 2])]) {
    if gates.is_empty() {
        return;
    }
    if gates.len() == 1 {
        par_apply_1q(state, gates[0].0, &gates[0].1);
        return;
    }

    const MULTI_TILE: usize = 16384;
    const L3_TILE: usize = 131072;

    const fn max_target_for_tile(tile_size: usize) -> usize {
        let mut t = 0usize;
        while (1usize << (t + 1)) <= tile_size {
            t += 1;
        }
        t - 1
    }

    let max_l2_target = max_target_for_tile(MULTI_TILE);
    let max_l3_target = max_target_for_tile(L3_TILE);

    let mut small_gates: SmallVec<[(usize, simd::PreparedGate1q); 16]> = SmallVec::new();
    let mut medium_gates: SmallVec<[(usize, simd::PreparedGate1q); 4]> = SmallVec::new();
    let mut large_gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 4]> = SmallVec::new();

    for &(target, mat) in gates {
        if target <= max_l2_target {
            small_gates.push((target, simd::PreparedGate1q::new(&mat)));
        } else if target <= max_l3_target {
            medium_gates.push((target, simd::PreparedGate1q::new(&mat)));
        } else {
            large_gates.push((target, mat));
        }
    }

    if !small_gates.is_empty() {
        let outer_block = 1usize << (max_l2_target + 1);
        let tile_size = MULTI_TILE.max(outer_block);
        state
            .par_chunks_mut(tile_size)
            .with_min_len(par_chunk_min_len(tile_size))
            .for_each(|tile| {
                for &(target, ref prepared) in &small_gates {
                    prepared.apply_tiled(tile, target);
                }
            });
    }

    if !medium_gates.is_empty() {
        let outer_block = 1usize << (max_l3_target + 1);
        let tile_size = L3_TILE.max(outer_block);
        state
            .par_chunks_mut(tile_size)
            .with_min_len(par_chunk_min_len(tile_size))
            .for_each(|tile| {
                for &(target, ref prepared) in &medium_gates {
                    prepared.apply_tiled(tile, target);
                }
            });
    }

    for (target, mat) in large_gates {
        par_apply_1q(state, target, &mat);
    }
}
