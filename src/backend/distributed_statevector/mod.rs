//! Distributed state vector backend.
//!
//! Partitions the `2^n` amplitude vector across `P = 2^p` ranks. The low
//! `n - p` qubits are *local* (indexing each rank's contiguous slice); the top
//! `p` qubits are *global* (selecting the rank id). Each rank holds a local
//! slice of length `2^(n-p)` inside an inner [`StatevectorBackend`]. Gates on
//! local qubits reuse the existing SIMD kernels.
//!
//! # Memory layout
//!
//! The global amplitude index is `rank * 2^(n-p) + local_index`. Qubit `q` with
//! `q < n - p` is bit `q` of `local_index`; qubit `q >= n - p` is bit
//! `q - (n - p)` of the rank id. Qubit 0 remains the least significant bit, so
//! `|0...0>` lives at `local_index = 0` on rank 0.
//!
//! # Status
//!
//! Implemented: local gates, one qubit gates on rank bits (exchange plus the
//! `combine_global_half` SIMD kernel; diagonals scale in place), and two-qubit
//! and controlled gates spanning rank bits (Cx, Cz, Swap, Rzz, Cu, Mcu). A
//! global control is constant per rank, so it gates the whole slice with no
//! communication; controlled diagonals (Cz, controlled phase, Rzz) never
//! communicate. Plus gathers for `probabilities` and `export_statevector`. With
//! a single rank ([`SerialComm`]), every qubit is local.
//!
//! Not implemented yet: measurement, reset, conditionals, and sampling. Fused
//! and batched gates are not applied across rank bits (fusion is disabled on
//! multi-rank runs).

#[cfg(test)]
mod tests;

use std::sync::Arc;

use num_complex::Complex64;

use crate::backend::simd;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::{dense_probability_len, dense_statevector_len, Backend};
use crate::circuit::{smallvec, Instruction};
use crate::distributed::DistributedContext;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

const BACKEND_NAME: &str = "distributed_statevector";

/// Distributed state vector backend over an [`Arc<DistributedContext>`].
pub struct DistributedStatevectorBackend {
    context: Arc<DistributedContext>,
    inner: StatevectorBackend,
    num_qubits: usize,
    global_qubits: usize,
    recv: Vec<Complex64>,
}

impl DistributedStatevectorBackend {
    /// Create a backend bound to the given rank context and RNG seed.
    pub fn new(context: Arc<DistributedContext>, seed: u64) -> Self {
        Self {
            context,
            inner: StatevectorBackend::new(seed),
            num_qubits: 0,
            global_qubits: 0,
            recv: Vec::new(),
        }
    }

    #[inline]
    fn local_qubits(&self) -> usize {
        self.num_qubits - self.global_qubits
    }

    #[inline]
    fn is_single_rank(&self) -> bool {
        self.context.size() == 1
    }

    #[inline]
    fn targets_are_local(&self, targets: &[usize]) -> bool {
        let local = self.local_qubits();
        targets.iter().all(|&q| q < local)
    }

    /// Bit position within the rank id for global qubit `q` (`q >= local`).
    #[inline]
    fn global_bit(&self, q: usize) -> usize {
        q - self.local_qubits()
    }

    /// Whether this rank holds the `|1>` half of global qubit `q`.
    #[inline]
    fn rank_bit_set(&self, q: usize) -> bool {
        (self.context.rank() >> self.global_bit(q)) & 1 == 1
    }

    /// Apply a one qubit gate whose target is stored in the rank id.
    ///
    /// Exchange with the partner rank, then write this rank's half of the 2x2
    /// result.
    fn apply_global_1q(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let len = self.inner.state.len();
        if self.recv.len() != len {
            self.recv.resize(len, Complex64::new(0.0, 0.0));
        }
        self.context
            .comm()
            .sendrecv_c64(partner, &self.inner.state, &mut self.recv);

        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };
        simd::combine_global_half(&mut self.inner.state, &self.recv, c_self, c_remote);
    }

    /// Apply a diagonal one qubit gate whose target is stored in the rank id.
    ///
    /// The rank bit is constant across the local slice, so this only scales the
    /// slice by `d0` or `d1`.
    fn apply_global_diagonal_1q(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        let factor = if self.rank_bit_set(target) { d1 } else { d0 };
        simd::scale_complex_slice(&mut self.inner.state, factor);
    }

    /// True if every global control qubit has its rank bit set on this rank.
    ///
    /// A global control is constant across a rank's slice (it equals the rank
    /// id bit), so a controlled gate is either fully active or fully inactive on
    /// a given rank. Every rank evaluates this identically for its own id, and
    /// exchange partners share the same global control bits (they differ only in
    /// the target bit), so collectives stay in lockstep.
    #[inline]
    fn global_controls_satisfied(&self, global_controls: &[usize]) -> bool {
        global_controls.iter().all(|&c| self.rank_bit_set(c))
    }

    /// Apply a 2x2 matrix to a local target qubit, gated by a set of local
    /// control qubits (all must be 1). With no controls this is a plain 1q gate.
    fn apply_local_controlled_1q(
        &mut self,
        local_controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        if local_controls.is_empty() {
            self.inner
                .apply_1q_matrix(target, &mat)
                .expect("local 1q matrix");
            return;
        }
        let ctrl_mask: usize = local_controls.iter().map(|&c| 1usize << c).sum();
        let half = 1usize << target;
        let state = &mut self.inner.state;
        let pairs = state.len() >> 1;
        for k in 0..pairs {
            let i0 = (k & !(half - 1)) << 1 | (k & (half - 1));
            if i0 & ctrl_mask != ctrl_mask {
                continue;
            }
            let i1 = i0 | half;
            let a = state[i0];
            let b = state[i1];
            state[i0] = mat[0][0] * a + mat[0][1] * b;
            state[i1] = mat[1][0] * a + mat[1][1] * b;
        }
    }

    /// Apply a 2x2 matrix to a global target qubit, gated by local control
    /// qubits (all must be 1). Exchanges with the partner rank, then combines
    /// only the controlled indices; uncontrolled indices keep their own value.
    fn apply_global_controlled_1q(
        &mut self,
        local_controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let len = self.inner.state.len();
        if self.recv.len() != len {
            self.recv.resize(len, Complex64::new(0.0, 0.0));
        }
        self.context
            .comm()
            .sendrecv_c64(partner, &self.inner.state, &mut self.recv);

        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };

        if local_controls.is_empty() {
            simd::combine_global_half(&mut self.inner.state, &self.recv, c_self, c_remote);
            return;
        }
        let ctrl_mask: usize = local_controls.iter().map(|&c| 1usize << c).sum();
        for (i, amp) in self.inner.state.iter_mut().enumerate() {
            if i & ctrl_mask == ctrl_mask {
                *amp = c_self * *amp + c_remote * self.recv[i];
            }
        }
    }

    /// Apply a controlled gate (one target, zero or more controls) whose qubit
    /// set may span local and global qubits. Covers Cx, Cu, and Mcu uniformly.
    fn apply_controlled_dist(
        &mut self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        let local = self.local_qubits();
        let (local_controls, global_controls): (Vec<usize>, Vec<usize>) =
            controls.iter().partition(|&&c| c < local);

        // A global control that is 0 on this rank deactivates the whole gate.
        if !self.global_controls_satisfied(&global_controls) {
            return;
        }

        if target < local {
            self.apply_local_controlled_1q(&local_controls, target, mat);
        } else {
            self.apply_global_controlled_1q(&local_controls, target, mat);
        }
    }

    /// Apply a controlled diagonal gate `diag(1, phase)` on the all-ones corner
    /// of its qubit set. Covers Cz, controlled phase, and diagonal Mcu with no
    /// communication: a global qubit contributes a constant rank-bit factor, and
    /// local qubits restrict which slice indices receive the phase.
    fn apply_controlled_phase_dist(&mut self, qubits: &[usize], phase: Complex64) {
        let local = self.local_qubits();
        let (local_qubits, global_qubits): (Vec<usize>, Vec<usize>) =
            qubits.iter().partition(|&&q| q < local);

        // Every global qubit in the corner must have its rank bit set.
        if !global_qubits.iter().all(|&q| self.rank_bit_set(q)) {
            return;
        }

        if local_qubits.is_empty() {
            simd::scale_complex_slice(&mut self.inner.state, phase);
            return;
        }
        let mask: usize = local_qubits.iter().map(|&q| 1usize << q).sum();
        for (i, amp) in self.inner.state.iter_mut().enumerate() {
            if i & mask == mask {
                *amp *= phase;
            }
        }
    }

    /// Apply `Rzz(theta)` across any local/global split. Rzz is fully diagonal,
    /// `phase = exp(-i theta/2)` when the two qubit bits agree and
    /// `exp(i theta/2)` when they differ, so no communication is needed: a
    /// global qubit contributes a constant rank bit to the parity.
    fn apply_rzz_dist(&mut self, q0: usize, q1: usize, theta: f64) {
        let local = self.local_qubits();
        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        let phases = [phase_same, phase_diff];

        match (q0 < local, q1 < local) {
            (true, true) => {
                for (i, amp) in self.inner.state.iter_mut().enumerate() {
                    let parity = ((i >> q0) ^ (i >> q1)) & 1;
                    *amp *= phases[parity];
                }
            }
            (false, false) => {
                let parity =
                    ((self.rank_bit_set(q0) as usize) ^ (self.rank_bit_set(q1) as usize)) & 1;
                simd::scale_complex_slice(&mut self.inner.state, phases[parity]);
            }
            (true, false) | (false, true) => {
                let (local_q, global_q) = if q0 < local { (q0, q1) } else { (q1, q0) };
                let global_bit = self.rank_bit_set(global_q) as usize;
                for (i, amp) in self.inner.state.iter_mut().enumerate() {
                    let parity = (((i >> local_q) & 1) ^ global_bit) & 1;
                    *amp *= phases[parity];
                }
            }
        }
    }

    /// Apply `SWAP(a, b)` across any local/global split.
    ///
    /// Local-local delegates to the inner kernel. When a global qubit is
    /// involved, SWAP exchanges the `|01>` and `|10>` amplitudes of the pair:
    /// only indices where the two qubit bits differ move.
    fn apply_swap_dist(&mut self, a: usize, b: usize) {
        let local = self.local_qubits();
        match (a < local, b < local) {
            (true, true) => {
                self.inner
                    .apply(&Instruction::Gate {
                        gate: Gate::Swap,
                        targets: smallvec![a, b],
                    })
                    .expect("local swap");
            }
            (false, false) => {
                // Both global: ranks differing in exactly these two bits and with
                // opposite values swap their entire slices. Equal-bit ranks are
                // unaffected (|00>, |11> map to themselves).
                if self.rank_bit_set(a) == self.rank_bit_set(b) {
                    return;
                }
                let partner = self.context.rank()
                    ^ (1usize << self.global_bit(a))
                    ^ (1usize << self.global_bit(b));
                let len = self.inner.state.len();
                if self.recv.len() != len {
                    self.recv.resize(len, Complex64::new(0.0, 0.0));
                }
                self.context
                    .comm()
                    .sendrecv_c64(partner, &self.inner.state, &mut self.recv);
                self.inner.state.copy_from_slice(&self.recv);
            }
            (true, false) | (false, true) => {
                let (local_q, global_q) = if a < local { (a, b) } else { (b, a) };
                let partner = self.context.rank() ^ (1usize << self.global_bit(global_q));
                let len = self.inner.state.len();
                if self.recv.len() != len {
                    self.recv.resize(len, Complex64::new(0.0, 0.0));
                }
                self.context
                    .comm()
                    .sendrecv_c64(partner, &self.inner.state, &mut self.recv);
                // This rank's global bit is fixed. Amplitudes whose local bit
                // differs from the global bit belong to the |01>/|10> subspace
                // and take the partner's value at the local-bit-flipped index.
                let global_bit = self.rank_bit_set(global_q);
                let half = 1usize << local_q;
                let len = self.inner.state.len();
                for i in 0..len {
                    let local_bit = (i >> local_q) & 1 == 1;
                    if local_bit != global_bit {
                        let partner_idx = i ^ half;
                        self.inner.state[i] = self.recv[partner_idx];
                    }
                }
            }
        }
    }

    /// Dispatch a multi-qubit gate whose qubit set spans at least one global
    /// qubit. Fusion is disabled in multi-rank mode, so gates arrive as raw
    /// primitives (Cx, Cz, Swap, Cu, Mcu, Rzz).
    fn apply_global_multi_qubit(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Cx => {
                self.apply_controlled_dist(&targets[..1], targets[1], Gate::X.matrix_2x2());
                Ok(())
            }
            Gate::Cz => {
                self.apply_controlled_phase_dist(
                    &[targets[0], targets[1]],
                    -Complex64::new(1.0, 0.0),
                );
                Ok(())
            }
            Gate::Swap => {
                self.apply_swap_dist(targets[0], targets[1]);
                Ok(())
            }
            Gate::Rzz(theta) => {
                self.apply_rzz_dist(targets[0], targets[1], *theta);
                Ok(())
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_controlled_phase_dist(&[targets[0], targets[1]], phase);
                } else {
                    self.apply_controlled_dist(&targets[..1], targets[1], **mat);
                }
                Ok(())
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                let controls = &targets[..num_ctrl];
                let target = targets[num_ctrl];
                if let Some(phase) = gate.controlled_phase() {
                    let mut corner: Vec<usize> = controls.to_vec();
                    corner.push(target);
                    self.apply_controlled_phase_dist(&corner, phase);
                } else {
                    self.apply_controlled_dist(controls, target, data.mat);
                }
                Ok(())
            }
            _ => Err(self.unsupported("fused or batched gate spanning a global qubit")),
        }
    }

    fn unsupported(&self, operation: &str) -> PrismError {
        PrismError::BackendUnsupported {
            backend: BACKEND_NAME.to_string(),
            operation: operation.to_string(),
        }
    }
}

impl Backend for DistributedStatevectorBackend {
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    fn supports_fused_gates(&self) -> bool {
        // Fusion assumes every target indexes the local slice.
        self.is_single_rank()
    }

    fn supports_qft_block(&self) -> bool {
        self.is_single_rank() && self.inner.supports_qft_block()
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let size = self.context.size();
        if !size.is_power_of_two() {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!("rank count {size} is not a power of two"),
            });
        }
        let p = size.trailing_zeros() as usize;
        let min_local = crate::distributed::min_local_qubits();
        if size > 1 && num_qubits < p + min_local {
            return Err(PrismError::BackendUnsupported {
                backend: BACKEND_NAME.to_string(),
                operation: format!(
                    "{num_qubits} qubits across {size} ranks leaves fewer than \
                     {min_local} local qubits per rank"
                ),
            });
        }

        self.num_qubits = num_qubits;
        self.global_qubits = p;
        let local_qubits = num_qubits - p;
        self.inner.init(local_qubits, num_classical_bits)?;

        // inner.init seeds index 0 on every rank; only rank 0 owns |0...0>.
        if self.context.rank() != 0 {
            if let Some(amp) = self.inner.state.get_mut(0) {
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        if self.global_qubits == 0 {
            return self.inner.apply(instruction);
        }
        match instruction {
            Instruction::Gate { gate, targets } => {
                if self.targets_are_local(targets) {
                    return self.inner.apply(instruction);
                }
                if gate.num_qubits() == 1 {
                    let target = targets[0];
                    let mat = gate.matrix_2x2();
                    if gate.is_diagonal_1q() {
                        self.apply_global_diagonal_1q(target, mat[0][0], mat[1][1]);
                    } else {
                        self.apply_global_1q(target, mat);
                    }
                    return Ok(());
                }
                self.apply_global_multi_qubit(gate, targets)
            }
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Measure { .. } => Err(self.unsupported("distributed measurement")),
            Instruction::Reset { .. } => Err(self.unsupported("distributed reset")),
            Instruction::Conditional { .. } => {
                Err(self.unsupported("distributed classical conditional"))
            }
        }
    }

    fn classical_results(&self) -> &[bool] {
        self.inner.classical_results()
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let local = self.inner.probabilities()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_probability_len(BACKEND_NAME, self.num_qubits)?;
        Ok(self.context.comm().allgather_f64(&local))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let local = self.inner.export_statevector()?;
        if self.global_qubits == 0 {
            return Ok(local);
        }
        dense_statevector_len(BACKEND_NAME, "statevector export", self.num_qubits)?;
        Ok(self.context.comm().allgather_c64(&local))
    }
}
