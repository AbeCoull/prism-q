//! Factored statevector backend.
//!
//! Maintains separate sub-state vectors per entangled qubit group. When a
//! multi-qubit gate bridges two groups, they merge via tensor product. Groups
//! never split. For sparse-entanglement circuits this is exponentially cheaper
//! than a monolithic 2^n statevector.

#[cfg(test)]
mod tests;

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::{smallvec, SmallVec};

use crate::backend::simd;
use crate::backend::statevector::insert_zero_bit;
use crate::backend::{is_phase_one, measurement_inv_norm, sorted_mcu_qubits, Backend};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{DiagEntry, Gate};

#[cfg(feature = "parallel")]
use crate::backend::statevector::SendPtr;
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ELEMS, MIN_PAR_ITERS, PARALLEL_THRESHOLD_QUBITS};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

type GateList = SmallVec<[(usize, [[Complex64; 2]; 2]); 4]>;

#[cfg(feature = "parallel")]
#[inline(always)]
fn par_chunk_min_len(chunk_size: usize) -> usize {
    (MIN_PAR_ELEMS / chunk_size).max(1)
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
    /// Returns the sub-state index containing all targets.
    fn ensure_same_substate(&mut self, targets: &[usize]) -> usize {
        let first_ss = self.qubit_to_substate[targets[0]];
        let mut need_merge: SmallVec<[usize; 4]> = SmallVec::new();

        for &q in &targets[1..] {
            let ss = self.qubit_to_substate[q];
            if ss != first_ss && !need_merge.contains(&ss) {
                need_merge.push(ss);
            }
        }

        for other_ss in need_merge {
            self.merge_substates(first_ss, other_ss);
        }

        first_ss
    }

    /// Merge sub-state `src_idx` into `dst_idx` via tensor product.
    ///
    /// Single-pass with correct bit placement: the merged state vector has
    /// qubit-to-bit mapping determined by the sorted merge of both qubit lists.
    fn merge_substates(&mut self, dst_idx: usize, src_idx: usize) {
        let src = self.substates[src_idx].take().unwrap();
        let dst = self.substates[dst_idx].as_ref().unwrap();

        let dst_n = dst.qubits.len();
        let src_n = src.qubits.len();
        let total_n = dst_n + src_n;
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

        let dst_state = &dst.state;
        let zero = Complex64::new(0.0, 0.0);
        let mut merged_state = vec![zero; total_dim];

        for (merged_idx, amp) in merged_state.iter_mut().enumerate() {
            let mut dst_local = 0usize;
            for (local_bit, &merged_bit) in dst_bit_positions.iter().enumerate() {
                dst_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
            }
            let mut src_local = 0usize;
            for (local_bit, &merged_bit) in src_bit_positions.iter().enumerate() {
                src_local |= ((merged_idx >> merged_bit) & 1) << local_bit;
            }

            *amp = dst_state[dst_local] * src.state[src_local];
        }

        let dst = self.substates[dst_idx].as_mut().unwrap();
        dst.state = merged_state;
        dst.qubits = merged_qubits;

        for &q in &src.qubits {
            self.qubit_to_substate[q] = dst_idx;
        }
    }

    /// Central gate dispatch. Translates global qubit indices to local and
    /// calls sequential SIMD kernels on the sub-state slice.
    #[inline(always)]
    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        if let Gate::MultiFused(data) = gate {
            self.apply_multi_fused(&data.gates, data.all_diagonal);
            return Ok(());
        }
        if let Gate::Multi2q(data) = gate {
            for &(q0, q1, ref mat) in &data.gates {
                let tgts = [q0, q1];
                let ss_idx = self.ensure_same_substate(&tgts);
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let lq0 = Self::local_qubit(sub, q0);
                let lq1 = Self::local_qubit(sub, q1);
                let prepared = simd::PreparedGate2q::new(mat);
                prepared.apply_full(&mut sub.state, sub.qubits.len(), lq0, lq1);
            }
            return Ok(());
        }
        // All other gates require targets in the same sub-state.
        let ss_idx = self.ensure_same_substate(targets);

        #[cfg(feature = "parallel")]
        {
            let sub = self.substates[ss_idx].as_ref().unwrap();
            if sub.qubits.len() >= PARALLEL_THRESHOLD_QUBITS {
                return self.dispatch_gate_par(ss_idx, gate, targets);
            }
        }

        match gate {
            Gate::Rzz(theta) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_rzz_seq(&mut sub.state, q0, q1, *theta);
            }
            Gate::Cx => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                apply_cx_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt);
            }
            Gate::Cz => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_cz_seq(&mut sub.state, sub.qubits.len(), q0, q1);
            }
            Gate::Swap => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                apply_swap_seq(&mut sub.state, sub.qubits.len(), q0, q1);
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let ctrl = Self::local_qubit(sub, targets[0]);
                    let tgt = Self::local_qubit(sub, targets[1]);
                    apply_cu_phase_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt, phase);
                } else {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let ctrl = Self::local_qubit(sub, targets[0]);
                    let tgt = Self::local_qubit(sub, targets[1]);
                    apply_cu_seq(&mut sub.state, sub.qubits.len(), ctrl, tgt, **mat);
                }
            }
            Gate::Mcu(data) => {
                let nc = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let local_ctrls: SmallVec<[usize; 4]> = targets[..nc]
                        .iter()
                        .map(|&q| Self::local_qubit(sub, q))
                        .collect();
                    let local_tgt = Self::local_qubit(sub, targets[nc]);
                    apply_mcu_phase_seq(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        phase,
                    );
                } else {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let local_ctrls: SmallVec<[usize; 4]> = targets[..nc]
                        .iter()
                        .map(|&q| Self::local_qubit(sub, q))
                        .collect();
                    let local_tgt = Self::local_qubit(sub, targets[nc]);
                    apply_mcu_seq(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        data.mat,
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
                apply_batch_phase_seq(&mut sub.state, sub.qubits.len(), local_ctrl, &local_phases);
            }
            Gate::BatchRzz(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for &(q0, q1, theta) in &data.edges {
                    let lq0 = Self::local_qubit(sub, q0);
                    let lq1 = Self::local_qubit(sub, q1);
                    apply_rzz_seq(&mut sub.state, lq0, lq1, theta);
                }
            }
            Gate::DiagonalBatch(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for entry in &data.entries {
                    match entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            let lq = Self::local_qubit(sub, *qubit);
                            let skip_lo = (d0.re - 1.0).abs() < 1e-15 && d0.im.abs() < 1e-15;
                            simd::apply_diagonal_sequential(&mut sub.state, lq, *d0, *d1, skip_lo);
                        }
                        DiagEntry::Phase2q { q0, q1, phase } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            apply_cu_phase_seq(&mut sub.state, sub.qubits.len(), lq0, lq1, *phase);
                        }
                        DiagEntry::Parity2q { q0, q1, same, diff } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            let n = sub.state.len();
                            for i in 0..n {
                                let parity = ((i >> lq0) ^ (i >> lq1)) & 1;
                                sub.state[i] *= if parity == 0 { *same } else { *diff };
                            }
                        }
                    }
                }
            }
            Gate::Fused2q(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                let prepared = simd::PreparedGate2q::new(mat);
                prepared.apply_full(&mut sub.state, sub.qubits.len(), q0, q1);
            }
            Gate::MultiFused(_) | Gate::Multi2q(_) => unreachable!(),
            _ => {
                let mat = gate.matrix_2x2();
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local = Self::local_qubit(sub, targets[0]);
                if gate.is_diagonal_1q() {
                    let skip_lo = is_phase_one(mat[0][0]);
                    simd::apply_diagonal_sequential(
                        &mut sub.state,
                        local,
                        mat[0][0],
                        mat[1][1],
                        skip_lo,
                    );
                } else {
                    let prepared = simd::PreparedGate1q::new(&mat);
                    prepared.apply_full_sequential(&mut sub.state, local);
                }
            }
        }
        Ok(())
    }

    /// Apply MultiFused gates grouped by sub-state — no merging needed.
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

    #[cfg(feature = "parallel")]
    fn dispatch_gate_par(&mut self, ss_idx: usize, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Rzz(theta) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                par_apply_rzz(&mut sub.state, q0, q1, *theta);
            }
            Gate::Cx => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let ctrl = Self::local_qubit(sub, targets[0]);
                let tgt = Self::local_qubit(sub, targets[1]);
                par_apply_cx(&mut sub.state, ctrl, tgt);
            }
            Gate::Cz => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                par_apply_cz(&mut sub.state, q0, q1);
            }
            Gate::Swap => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                par_apply_swap(&mut sub.state, q0, q1);
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let ctrl = Self::local_qubit(sub, targets[0]);
                    let tgt = Self::local_qubit(sub, targets[1]);
                    par_apply_cu_phase(&mut sub.state, sub.qubits.len(), ctrl, tgt, phase);
                } else {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let ctrl = Self::local_qubit(sub, targets[0]);
                    let tgt = Self::local_qubit(sub, targets[1]);
                    par_apply_cu(&mut sub.state, sub.qubits.len(), ctrl, tgt, **mat);
                }
            }
            Gate::Mcu(data) => {
                let nc = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let local_ctrls: SmallVec<[usize; 4]> = targets[..nc]
                        .iter()
                        .map(|&q| Self::local_qubit(sub, q))
                        .collect();
                    let local_tgt = Self::local_qubit(sub, targets[nc]);
                    par_apply_mcu_phase(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        phase,
                    );
                } else {
                    let sub = self.substates[ss_idx].as_mut().unwrap();
                    let local_ctrls: SmallVec<[usize; 4]> = targets[..nc]
                        .iter()
                        .map(|&q| Self::local_qubit(sub, q))
                        .collect();
                    let local_tgt = Self::local_qubit(sub, targets[nc]);
                    par_apply_mcu(
                        &mut sub.state,
                        sub.qubits.len(),
                        &local_ctrls,
                        local_tgt,
                        data.mat,
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
                par_apply_batch_phase(&mut sub.state, local_ctrl, &local_phases);
            }
            Gate::BatchRzz(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for &(q0, q1, theta) in &data.edges {
                    let lq0 = Self::local_qubit(sub, q0);
                    let lq1 = Self::local_qubit(sub, q1);
                    par_apply_rzz(&mut sub.state, lq0, lq1, theta);
                }
            }
            Gate::DiagonalBatch(data) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                for entry in &data.entries {
                    match entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            let lq = Self::local_qubit(sub, *qubit);
                            let skip_lo = (d0.re - 1.0).abs() < 1e-15 && d0.im.abs() < 1e-15;
                            par_apply_diagonal(&mut sub.state, lq, *d0, *d1, skip_lo);
                        }
                        DiagEntry::Phase2q { q0, q1, phase } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            par_apply_cu_phase(&mut sub.state, sub.qubits.len(), lq0, lq1, *phase);
                        }
                        DiagEntry::Parity2q { q0, q1, same, diff } => {
                            let lq0 = Self::local_qubit(sub, *q0);
                            let lq1 = Self::local_qubit(sub, *q1);
                            let phases = [*same, *diff];
                            sub.state
                                .par_chunks_mut(MIN_PAR_ELEMS)
                                .enumerate()
                                .for_each(|(chunk_idx, chunk)| {
                                    let base = chunk_idx * MIN_PAR_ELEMS;
                                    for (j, amp) in chunk.iter_mut().enumerate() {
                                        let i = base + j;
                                        let parity = ((i >> lq0) ^ (i >> lq1)) & 1;
                                        *amp *= phases[parity];
                                    }
                                });
                        }
                    }
                }
            }
            Gate::Fused2q(mat) => {
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let q0 = Self::local_qubit(sub, targets[0]);
                let q1 = Self::local_qubit(sub, targets[1]);
                par_apply_fused2q(&mut sub.state, sub.qubits.len(), q0, q1, mat);
            }
            Gate::MultiFused(_) | Gate::Multi2q(_) => unreachable!(),
            _ => {
                let mat = gate.matrix_2x2();
                let sub = self.substates[ss_idx].as_mut().unwrap();
                let local = Self::local_qubit(sub, targets[0]);
                if gate.is_diagonal_1q() {
                    let skip_lo = is_phase_one(mat[0][0]);
                    par_apply_diagonal(&mut sub.state, local, mat[0][0], mat[1][1], skip_lo);
                } else {
                    par_apply_1q(&mut sub.state, local, &mat);
                }
            }
        }
        Ok(())
    }

    fn apply_reset(&mut self, qubit: usize) {
        let ss_idx = self.qubit_to_substate[qubit];
        let sub = self.substates[ss_idx].as_mut().unwrap();
        let local = Self::local_qubit(sub, qubit);

        let mask = 1usize << local;
        let n = sub.state.len();
        let zero = Complex64::new(0.0, 0.0);

        let mut prob_zero = 0.0f64;
        for i in 0..n {
            if (i & mask) == 0 {
                prob_zero += sub.state[i].norm_sqr();
            }
        }

        if prob_zero > 0.0 {
            let inv_norm = 1.0 / prob_zero.sqrt();
            for i in 0..n {
                if (i & mask) == 0 {
                    sub.state[i] *= inv_norm;
                } else {
                    sub.state[i] = zero;
                }
            }
        } else {
            for amp in sub.state.iter_mut() {
                *amp = zero;
            }
            sub.state[0] = Complex64::new(1.0, 0.0);
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
}

impl Backend for FactoredBackend {
    fn name(&self) -> &'static str {
        "factored"
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

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
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

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
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
    let mut sorted_buf = [0usize; 10];
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
    let mut sorted_buf = [0usize; 10];
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
    let mut sorted_buf = [0usize; 10];
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
    let mut sorted_buf = [0usize; 10];
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
