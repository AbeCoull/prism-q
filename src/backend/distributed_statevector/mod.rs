//! Distributed state vector backend.
//!
//! Splits the `2^n` amplitude vector across `P = 2^p` ranks. The low `n - p`
//! qubits index the local slice. The top `p` qubits select the rank. Each rank
//! stores `2^(n - p)` amplitudes in an inner [`StatevectorBackend`].
//!
//! # Memory layout
//!
//! Global index: `rank * 2^(n - p) + local_index`. If `q < n - p`, qubit `q` is
//! bit `q` of `local_index`; otherwise it is bit `q - (n - p)` of the rank id.
//! Qubit 0 is the least significant bit. `|0...0>` is index 0 on rank 0.
//!
//! # Status
//!
//! Implemented: local gates, rank bit one qubit gates, two qubit gates, controlled
//! gates across rank bits, `probabilities`, and `export_statevector`. A global
//! control is constant on a rank, so it gates the whole slice with no
//! communication. Diagonal controlled gates never communicate. With one rank
//! ([`SerialComm`]), every qubit is local.
//!
//! Fusion runs in every mode. Local fused gates dispatch to the inner SIMD
//! kernels. Fused or batched gates that span rank bits are decomposed into the
//! paths above. A general two qubit gate over one global qubit needs one
//! pairwise exchange; over two global qubits it gathers a group of four ranks.
//!
//! Once a rank resolves its global qubit bits, the remaining gate is local and
//! dispatches to the inner backend. The only manual amplitude loops combine the
//! received buffers after communication.
//!
//! Measurement, reset, and classical conditionals are supported. Measurement
//! probabilities are summed with `Allreduce`. Each rank uses the same seeded RNG,
//! so ranks agree without exchanging the draw. Reset follows the statevector
//! convention: project onto `|0>`, renormalize, and reinitialize to `|0...0>`
//! when the zero branch is empty.
//!
//! # Communication cost
//!
//! Only gates that are not diagonal and touch a global target communicate. A
//! global one qubit gate, or a two qubit gate over one global qubit, costs one
//! pairwise exchange of the local slice. A two qubit gate over two global qubits
//! gathers four ranks. Global one qubit exchange is tiled by
//! [`crate::distributed::exchange_chunk`], which bounds the receive buffer.
//!
//! [`DistributedStatevectorBackend::exchange_messages`] and
//! [`DistributedStatevectorBackend::exchange_amplitudes`] expose rank local
//! communication volume. Use these counters to evaluate qubit reordering and
//! routing, since one host cannot measure real network latency.
//!
//! Not implemented yet: distributed shot sampling, and automatic qubit reorder
//! to keep busy qubits local.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::backend::simd;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::{dense_probability_len, dense_statevector_len, measurement_inv_norm, Backend};
use crate::circuit::{smallvec, Instruction, SmallVec};
use crate::distributed::DistributedContext;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

const BACKEND_NAME: &str = "distributed_statevector";

/// Whether a 2x2 matrix is diagonal.
#[inline]
fn is_diagonal_2x2(mat: &[[Complex64; 2]; 2]) -> bool {
    mat[0][1].norm() < 1e-12 && mat[1][0].norm() < 1e-12
}

/// Distributed state vector backend over an [`Arc<DistributedContext>`].
pub struct DistributedStatevectorBackend {
    context: Arc<DistributedContext>,
    inner: StatevectorBackend,
    num_qubits: usize,
    global_qubits: usize,
    recv: Vec<Complex64>,
    seed: u64,
    /// Max amplitudes exchanged per message for a global one qubit gate.
    /// Tiling the exchange bounds the receive buffer to `exchange_chunk`.
    exchange_chunk: usize,
    /// Count of `sendrecv` messages issued by this rank, and the total
    /// amplitudes exchanged. Reorder and routing passes should minimize these
    /// counters.
    exchange_messages: u64,
    exchange_amplitudes: u64,
    /// RNG for measurement decisions, seeded identically on every rank and
    /// advanced in lockstep. Outcomes are derived from `Allreduce`d global
    /// probabilities, so all ranks agree without exchanging the draw.
    meas_rng: ChaCha8Rng,
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
            seed,
            exchange_chunk: crate::distributed::exchange_chunk(),
            exchange_messages: 0,
            exchange_amplitudes: 0,
            meas_rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Override the exchange chunk size in amplitudes. Tests use this to cover
    /// the tiled path without using the process environment.
    #[cfg(test)]
    pub(crate) fn set_exchange_chunk(&mut self, chunk: usize) {
        self.exchange_chunk = chunk.max(1);
    }

    /// Number of `sendrecv` messages this rank has issued since `init`.
    ///
    /// Cost proxy for this backend. One host cannot measure real network
    /// latency, so routing changes are evaluated against this count.
    pub fn exchange_messages(&self) -> u64 {
        self.exchange_messages
    }

    /// Total amplitudes this rank has sent across all exchanges since `init`.
    pub fn exchange_amplitudes(&self) -> u64 {
        self.exchange_amplitudes
    }

    /// Record a pairwise exchange of `amplitudes` for the cost counters.
    #[inline]
    fn count_exchange(&mut self, amplitudes: usize) {
        self.exchange_messages += 1;
        self.exchange_amplitudes += amplitudes as u64;
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
    /// result. The combine is elementwise, so the exchange is tiled in chunks of
    /// [`crate::distributed::exchange_chunk`] amplitudes, bounding the receive
    /// buffer to `chunk` instead of a full slice copy. The default chunk is
    /// the whole slice (single message), so behavior is unchanged unless tuned.
    fn apply_global_1q(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(target));
        let (c_self, c_remote) = if self.rank_bit_set(target) {
            (mat[1][1], mat[1][0])
        } else {
            (mat[0][0], mat[0][1])
        };
        let len = self.inner.state.len();
        let chunk = self.exchange_chunk.min(len).max(1);
        if self.recv.len() != chunk {
            self.recv.resize(chunk, Complex64::new(0.0, 0.0));
        }
        let mut off = 0;
        while off < len {
            let end = (off + chunk).min(len);
            self.count_exchange(end - off);
            let recv = &mut self.recv[..end - off];
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state[off..end], recv);
            simd::combine_global_half(&mut self.inner.state[off..end], recv, c_self, c_remote);
            off = end;
        }
    }

    /// Apply a diagonal one qubit gate whose target is stored in the rank id.
    ///
    /// The rank bit is constant across the local slice, so this only scales the
    /// slice by `d0` or `d1`.
    fn apply_global_diagonal_1q(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        let factor = if self.rank_bit_set(target) { d1 } else { d0 };
        simd::scale_complex_slice(&mut self.inner.state, factor);
    }

    /// Apply a 2x2 matrix to a local target qubit, gated by a set of local
    /// control qubits (all must be 1). The whole operation is local, so it
    /// dispatches to the inner backend's SIMD and parallel controlled kernels.
    fn apply_local_controlled_1q(
        &mut self,
        local_controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        let gate = match local_controls.len() {
            0 => {
                self.inner
                    .apply_1q_matrix(target, &mat)
                    .expect("local 1q matrix");
                return;
            }
            1 => Gate::cu(mat),
            n => Gate::mcu(mat, n as u8),
        };
        let mut targets: SmallVec<[usize; 4]> = local_controls.iter().copied().collect();
        targets.push(target);
        self.inner
            .apply(&Instruction::Gate { gate, targets })
            .expect("local controlled 1q");
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
        self.count_exchange(len);
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
        let mut local_controls: SmallVec<[usize; 4]> = SmallVec::new();
        for &c in controls {
            if c < local {
                local_controls.push(c);
            } else if !self.rank_bit_set(c) {
                // A zero global control disables the gate on this rank.
                return;
            }
        }

        if target < local {
            self.apply_local_controlled_1q(&local_controls, target, mat);
        } else {
            self.apply_global_controlled_1q(&local_controls, target, mat);
        }
    }

    /// Apply a controlled diagonal gate `diag(1, phase)` on the all ones corner
    /// of its qubit set. Covers Cz, controlled phase, and diagonal Mcu with no
    /// communication: a global qubit contributes a constant rank bit, and
    /// local qubits restrict which slice indices receive the phase.
    ///
    /// The residual on local qubits is another controlled phase gate, so it uses
    /// the inner backend kernels.
    fn apply_controlled_phase_dist(&mut self, qubits: &[usize], phase: Complex64) {
        let local = self.local_qubits();
        let mut local_qubits: SmallVec<[usize; 8]> = SmallVec::new();
        for &q in qubits {
            if q < local {
                local_qubits.push(q);
            } else if !self.rank_bit_set(q) {
                // A zero global corner bit makes the gate inactive on this rank.
                return;
            }
        }
        self.apply_local_corner_phase(&local_qubits, phase);
    }

    /// Apply `phase` on the all ones corner through the inner backend.
    fn apply_local_corner_phase(&mut self, local_qubits: &[usize], phase: Complex64) {
        let z = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        match local_qubits.len() {
            0 => simd::scale_complex_slice(&mut self.inner.state, phase),
            1 => self
                .inner
                .apply_1q_matrix(local_qubits[0], &[[one, z], [z, phase]])
                .expect("local diagonal phase"),
            n => {
                let mat = [[one, z], [z, phase]];
                let gate = if n == 2 {
                    Gate::cu(mat)
                } else {
                    Gate::mcu(mat, (n - 1) as u8)
                };
                self.inner
                    .apply(&Instruction::Gate {
                        gate,
                        targets: local_qubits.iter().copied().collect(),
                    })
                    .expect("local controlled phase");
            }
        }
    }

    /// Apply `Rzz(theta)` across any local or global split. Rzz is diagonal,
    /// `phase = exp(-i theta/2)` when the two qubit bits agree and
    /// `exp(i theta/2)` when they differ, so no communication is needed: a
    /// global qubit contributes a constant rank bit to the parity.
    fn apply_rzz_dist(&mut self, q0: usize, q1: usize, theta: f64) {
        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        self.apply_rzz_phases_dist(q0, q1, phase_same, phase_diff);
    }

    /// Apply a parity diagonal two qubit phase. Shared by `Rzz` and
    /// `Parity2q`; it needs no communication.
    fn apply_rzz_phases_dist(
        &mut self,
        q0: usize,
        q1: usize,
        phase_same: Complex64,
        phase_diff: Complex64,
    ) {
        let local = self.local_qubits();

        match (q0 < local, q1 < local) {
            (true, true) => {
                // Both qubits are local, so use the inner diagonal batch kernel.
                use crate::gates::{DiagEntry, DiagonalBatchData};
                let entry = DiagEntry::Parity2q {
                    q0,
                    q1,
                    same: phase_same,
                    diff: phase_diff,
                };
                self.inner
                    .apply(&Instruction::Gate {
                        gate: Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                            entries: vec![entry],
                        })),
                        targets: smallvec![q0, q1],
                    })
                    .expect("local parity diagonal");
            }
            (false, false) => {
                let parity =
                    ((self.rank_bit_set(q0) as usize) ^ (self.rank_bit_set(q1) as usize)) & 1;
                let factor = [phase_same, phase_diff][parity];
                simd::scale_complex_slice(&mut self.inner.state, factor);
            }
            (true, false) | (false, true) => {
                // One global qubit is fixed on this rank. The residual is a
                // diagonal one qubit gate.
                let (local_q, global_q) = if q0 < local { (q0, q1) } else { (q1, q0) };
                let gbit = self.rank_bit_set(global_q) as usize;
                // Local bit 0 uses parity gbit. Local bit 1 uses gbit ^ 1.
                let d0 = [phase_same, phase_diff][gbit];
                let d1 = [phase_same, phase_diff][gbit ^ 1];
                let z = Complex64::new(0.0, 0.0);
                self.inner
                    .apply_1q_matrix(local_q, &[[d0, z], [z, d1]])
                    .expect("local parity residual");
            }
        }
    }

    /// Apply `SWAP(a, b)` across any local or global split.
    ///
    /// Local pairs delegate to the inner kernel. With a global qubit, only the
    /// `|01>` and `|10>` amplitudes move.
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
                // Both are global. Ranks that differ in these two bits swap
                // slices. Equal bits stay in place.
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
                self.count_exchange(len);
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
                self.count_exchange(len);
                self.context
                    .comm()
                    .sendrecv_c64(partner, &self.inner.state, &mut self.recv);
                // The global bit is fixed on this rank. Entries where the local
                // bit differs take the partner value at the flipped local index.
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

    /// Apply a general 4x4 two qubit unitary across any local or global split.
    ///
    /// `mat` uses basis index `2*b0 + b1`, with `q0` as the high bit. Local
    /// pairs delegate to the inner kernel. One global qubit needs one exchange;
    /// two global qubits gather the rank group that shares both rank bits.
    fn apply_2q_dist(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let local = self.local_qubits();
        match (q0 < local, q1 < local) {
            (true, true) => self.apply_local_fused_2q(q0, q1, mat),
            (true, false) | (false, true) => self.apply_2q_one_global(q0, q1, mat),
            (false, false) => self.apply_2q_two_global(q0, q1, mat),
        }
    }

    /// Apply a fully local 4x4 gate through the inner backend's tiled kernel.
    fn apply_local_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        self.inner
            .apply(&Instruction::Gate {
                gate: Gate::Fused2q(Box::new(*mat)),
                targets: smallvec![q0, q1],
            })
            .expect("local fused 2q");
    }

    /// One qubit is local and one is global. Exchange with the partner rank,
    /// then recompute each amplitude from the four inputs of the 2x2 block.
    fn apply_2q_one_global(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let local = self.local_qubits();
        let (local_q, global_q, global_is_q0) = if q0 < local {
            (q0, q1, false)
        } else {
            (q1, q0, true)
        };
        let partner = self.context.rank() ^ (1usize << self.global_bit(global_q));
        let len = self.inner.state.len();
        if self.recv.len() != len {
            self.recv.resize(len, Complex64::new(0.0, 0.0));
        }
        self.count_exchange(len);
        self.context
            .comm()
            .sendrecv_c64(partner, &self.inner.state, &mut self.recv);

        let g = self.rank_bit_set(global_q) as usize;
        let half = 1usize << local_q;
        // Basis index in `mat` is `2*b_q0 + b_q1`.
        let basis = |gbit: usize, lbit: usize| -> usize {
            if global_is_q0 {
                (gbit << 1) | lbit
            } else {
                (lbit << 1) | gbit
            }
        };
        // Output depends on both local bit siblings, so snapshot first.
        let local_snapshot = self.inner.state.clone();
        for i in 0..len {
            let l = (i >> local_q) & 1;
            let row = basis(g, l);
            let sib0 = i & !half; // local bit 0 sibling index
            let sib1 = i | half; // local bit 1 sibling index
            let mut acc = mat[row][basis(g, 0)] * local_snapshot[sib0];
            acc += mat[row][basis(g, 1)] * local_snapshot[sib1];
            acc += mat[row][basis(1 - g, 0)] * self.recv[sib0];
            acc += mat[row][basis(1 - g, 1)] * self.recv[sib1];
            self.inner.state[i] = acc;
        }
    }

    /// Both qubits global. The four `(q0, q1)` combinations live on four ranks
    /// that share every other rank bit. Gather the other three slices, then each
    /// rank computes its output row of `mat` from the four gathered slices.
    fn apply_2q_two_global(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let b0 = self.global_bit(q0);
        let b1 = self.global_bit(q1);
        let rank = self.context.rank();
        let g0 = (rank >> b0) & 1;
        let g1 = (rank >> b1) & 1;
        let rank_basis = (g0 << 1) | g1;

        let len = self.inner.state.len();
        // Gather only partner slices. The local term reads from `inner.state`.
        let mut partners: [Option<Vec<Complex64>>; 4] = [None, None, None, None];
        for (c, slot) in partners.iter_mut().enumerate() {
            if c == rank_basis {
                continue;
            }
            let c0 = (c >> 1) & 1;
            let c1 = c & 1;
            let partner = (rank & !(1 << b0) & !(1 << b1)) | (c0 << b0) | (c1 << b1);
            let mut buf = vec![Complex64::new(0.0, 0.0); len];
            self.exchange_messages += 1;
            self.exchange_amplitudes += len as u64;
            self.context
                .comm()
                .sendrecv_c64(partner, &self.inner.state, &mut buf);
            *slot = Some(buf);
        }
        // Write into a fresh buffer so local reads see the old amplitudes.
        let mut out = vec![Complex64::new(0.0, 0.0); len];
        let self_coeff = mat[rank_basis][rank_basis];
        for i in 0..len {
            let mut acc = self_coeff * self.inner.state[i];
            for (c, slot) in partners.iter().enumerate() {
                if let Some(slice) = slot {
                    acc += mat[rank_basis][c] * slice[i];
                }
            }
            out[i] = acc;
        }
        self.inner.state = out;
    }

    /// Dispatch a gate that spans at least one global qubit.
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
            Gate::Fused2q(mat) => {
                self.apply_2q_dist(targets[0], targets[1], mat);
                Ok(())
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_2q_dist(q0, q1, mat);
                }
                Ok(())
            }
            Gate::MultiFused(data) => {
                for &(q, ref mat) in &data.gates {
                    if q < self.local_qubits() {
                        self.inner.apply_1q_matrix(q, mat).expect("local 1q matrix");
                    } else if is_diagonal_2x2(mat) {
                        self.apply_global_diagonal_1q(q, mat[0][0], mat[1][1]);
                    } else {
                        self.apply_global_1q(q, *mat);
                    }
                }
                Ok(())
            }
            Gate::BatchPhase(data) => {
                let control = targets[0];
                for &(target, phase) in &data.phases {
                    self.apply_controlled_phase_dist(&[control, target], phase);
                }
                Ok(())
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    self.apply_rzz_dist(q0, q1, theta);
                }
                Ok(())
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    self.apply_diag_entry_dist(entry);
                }
                Ok(())
            }
            _ => Err(self.unsupported("gate spanning a global qubit")),
        }
    }

    /// Apply a single [`DiagEntry`] across any local or global split.
    fn apply_diag_entry_dist(&mut self, entry: &crate::gates::DiagEntry) {
        use crate::gates::DiagEntry;
        match *entry {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                if qubit < self.local_qubits() {
                    self.inner
                        .apply_1q_matrix(
                            qubit,
                            &[
                                [d0, Complex64::new(0.0, 0.0)],
                                [Complex64::new(0.0, 0.0), d1],
                            ],
                        )
                        .expect("local diagonal 1q");
                } else {
                    self.apply_global_diagonal_1q(qubit, d0, d1);
                }
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                self.apply_controlled_phase_dist(&[q0, q1], phase);
            }
            DiagEntry::Parity2q {
                q0, q1, same, diff, ..
            } => {
                // These are the parity phases for Rzz(theta).
                self.apply_rzz_phases_dist(q0, q1, same, diff);
            }
        }
    }

    /// Total scaled weight of the `qubit == outcome` subspace across ranks.
    fn prob_outcome_global(&self, qubit: usize, outcome: bool) -> f64 {
        let norm_sq = self.inner.pending_norm * self.inner.pending_norm;
        let local_prob = if qubit < self.local_qubits() {
            let half = 1usize << qubit;
            let block_size = half << 1;
            let mut acc = 0.0f64;
            for block in self.inner.state.chunks(block_size) {
                let (lo, hi) = block.split_at(half);
                acc += simd::norm_sqr_sum(if outcome { hi } else { lo });
            }
            acc
        } else if self.rank_bit_set(qubit) == outcome {
            simd::norm_sqr_sum(&self.inner.state)
        } else {
            0.0
        };
        self.context.comm().allreduce_sum_f64(local_prob) * norm_sq
    }

    /// Total weight of the `qubit == 1` subspace across all ranks. Used by
    /// measurement and as `P(qubit = 1)`.
    fn prob_one_global(&self, qubit: usize) -> f64 {
        self.prob_outcome_global(qubit, true)
    }

    /// Measure `qubit`, collapse the state, and record the bit. Deterministic
    /// across ranks: the outcome is drawn from the lockstep `meas_rng` against an
    /// `Allreduce`d probability, so every rank collapses to the same branch.
    fn measure_dist(&mut self, qubit: usize, classical_bit: usize) {
        let prob_one = self.prob_one_global(qubit);
        let outcome = self.meas_rng.random::<f64>() < prob_one;
        self.inner.classical_bits[classical_bit] = outcome;
        self.collapse(qubit, outcome);
        self.inner.pending_norm *= measurement_inv_norm(outcome, prob_one);
    }

    /// Zero the amplitudes inconsistent with `qubit == outcome`.
    fn collapse(&mut self, qubit: usize, outcome: bool) {
        let zero = Complex64::new(0.0, 0.0);
        if qubit < self.local_qubits() {
            let half = 1usize << qubit;
            let block_size = half << 1;
            for block in self.inner.state.chunks_mut(block_size) {
                let (lo, hi) = block.split_at_mut(half);
                if outcome {
                    simd::zero_slice(lo);
                } else {
                    simd::zero_slice(hi);
                }
            }
        } else if self.rank_bit_set(qubit) != outcome {
            // This rank holds the eliminated branch entirely.
            for amp in self.inner.state.iter_mut() {
                *amp = zero;
            }
        }
    }

    /// Reset `qubit` to `|0>` using the statevector backend's projection
    /// convention.
    fn reset_dist(&mut self, qubit: usize) {
        let prob_zero = self.prob_outcome_global(qubit, false);
        if prob_zero > 0.0 {
            self.collapse(qubit, false);
            self.inner.pending_norm *= 1.0 / prob_zero.sqrt();
        } else {
            simd::zero_slice(&mut self.inner.state);
            if self.context.rank() == 0 {
                if let Some(amp) = self.inner.state.get_mut(0) {
                    *amp = Complex64::new(1.0, 0.0);
                }
            }
            self.inner.pending_norm = 1.0;
        }
    }

    /// Route a gate to the local fast path or the distributed paths by whether
    /// its qubits span a rank bit.
    fn apply_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        if self.global_qubits == 0 || self.targets_are_local(targets) {
            return self.inner.apply(&Instruction::Gate {
                gate: gate.clone(),
                targets: targets.into(),
            });
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
        // Fusion runs in every mode. Fully local fused gates dispatch to the
        // inner backend's tiled SIMD kernels; fused or batched gates that span a
        // rank bit are decomposed into primitives at apply time.
        true
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
        self.meas_rng = ChaCha8Rng::seed_from_u64(self.seed);
        self.exchange_messages = 0;
        self.exchange_amplitudes = 0;
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
        match instruction {
            // Measurement routes through the distributed path even at a single
            // rank, so `meas_rng` is the sole measurement RNG and outcomes
            // match across every rank count for a given seed.
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.measure_dist(*qubit, *classical_bit);
                Ok(())
            }
            Instruction::Reset { qubit } => {
                self.reset_dist(*qubit);
                Ok(())
            }
            Instruction::Barrier { .. } => Ok(()),
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(self.inner.classical_results()) {
                    self.apply_gate(gate, targets)
                } else {
                    Ok(())
                }
            }
            Instruction::Gate { gate, targets } => self.apply_gate(gate, targets),
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

    fn qubit_probability(&self, qubit: usize) -> Result<f64> {
        Ok(self.prob_one_global(qubit))
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.reset_dist(qubit);
        Ok(())
    }
}
