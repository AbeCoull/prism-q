//! Circuit qubit to physical position map of [`DistributedStatevectorBackend`]: eviction
//! victims, relabel exchanges, and translation of gates, masks and gathered vectors.

use std::borrow::Cow;

use super::DistributedStatevectorBackend;
use super::shard::{for_each_gate_qubit, gather_indexed, scatter_indexed};
use crate::backend::Backend;
use crate::circuit::{Instruction, SmallVec, smallvec};
use crate::gates::{DiagEntry, Gate};

impl DistributedStatevectorBackend {
    #[inline]
    pub(super) fn local_qubits(&self) -> usize {
        self.num_qubits - self.global_qubits
    }

    #[inline]
    pub(super) fn is_single_rank(&self) -> bool {
        self.context.size() == 1
    }

    /// Bit position within the rank id for global qubit `q` (`q >= local`).
    #[inline]
    pub(super) fn global_bit(&self, q: usize) -> usize {
        q - self.local_qubits()
    }

    /// Whether this rank holds the `|1>` half of global qubit `q`.
    #[inline]
    pub(super) fn rank_bit_set(&self, q: usize) -> bool {
        (self.context.rank() >> self.global_bit(q)) & 1 == 1
    }

    /// Advance the instruction tick and mark every circuit qubit the
    /// instruction references. Marked qubits are exempt from eviction until the
    /// next instruction. Identical on every rank because the instruction stream
    /// is identical.
    pub(super) fn touch_instruction(&mut self, gate: &Gate, targets: &[usize]) {
        self.tick += 1;
        let tick = self.tick;
        for_each_gate_qubit(gate, targets, |q| self.last_used[q] = tick);
    }

    pub(super) fn refresh_map_identity(&mut self) {
        self.map_identity = self.qubit_map.iter().enumerate().all(|(q, &p)| q == p);
    }

    /// Apply SWAP as a pure relabeling: exchange the two circuit qubits' map
    /// entries. No amplitudes move and no rank communicates.
    pub(super) fn swap_circuit_qubits(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let pa = self.qubit_map[a];
        let pb = self.qubit_map[b];
        self.qubit_map.swap(a, b);
        self.phys_map.swap(pa, pb);
        self.refresh_map_identity();
    }

    /// Local position holding the least recently used circuit qubit that the
    /// current instruction does not reference. `None` when every local qubit is
    /// referenced this tick.
    pub(super) fn pick_victim(&self) -> Option<usize> {
        let local = self.local_qubits();
        let mut best: Option<(u64, usize)> = None;
        for pos in 0..local {
            let used = self.last_used[self.phys_map[pos]];
            if used == self.tick {
                continue;
            }
            match best {
                Some((b, _)) if used >= b => {}
                _ => best = Some((used, pos)),
            }
        }
        best.map(|(_, pos)| pos)
    }

    /// Bring each requested circuit qubit into a local position. Best effort:
    /// stops when no eviction victim remains, leaving the rest to the direct
    /// exchange paths. Each relabel costs one half-slice exchange. Inside a
    /// planned window every requested qubit is already local and this returns
    /// without work.
    pub(super) fn make_local(&mut self, req: &[usize]) {
        for &q in req {
            let pos = self.qubit_map[q];
            if pos < self.local_qubits() {
                continue;
            }
            let Some(victim) = self.pick_victim() else {
                return;
            };
            self.relabel_swap(victim, pos);
        }
    }

    /// Exchange the moving half of a local/global SWAP with the partner rank.
    /// Only amplitudes whose local bit differs from this rank's bit of the
    /// global position move, so each rank exchanges half its slice. Both ranks
    /// enumerate their moving halves in ascending index order, which the
    /// single-bit XOR relation between the two sets preserves, so the k-th
    /// received amplitude lands at the k-th moving index. Tiled by
    /// `exchange_chunk` like the direct global exchange. Pure data movement:
    /// the qubit map is untouched.
    pub(super) fn half_slice_swap(&mut self, local_pos: usize, global_pos: usize) {
        let partner = self.context.rank() ^ (1usize << self.global_bit(global_pos));
        let gbit = self.rank_bit_set(global_pos);
        let stride = 1usize << local_pos;
        let fixed = if gbit { 0 } else { stride };
        let moving = self.inner.state.len() / 2;
        let chunk = self.exchange_chunk.min(moving).max(1);
        self.ensure_pack(chunk);
        self.ensure_recv(chunk);
        let index_of =
            |flat: usize| ((flat >> local_pos) << (local_pos + 1)) | fixed | (flat & (stride - 1));
        let mut off = 0;
        while off < moving {
            let count = (off + chunk).min(moving) - off;
            gather_indexed(&mut self.pack[..count], &self.inner.state, |k| {
                index_of(off + k)
            });
            self.count_exchange(count);
            self.context
                .comm()
                .sendrecv_c64(partner, &self.pack[..count], &mut self.recv[..count]);
            scatter_indexed(
                &mut self.inner.state,
                &self.recv[..count],
                |k| index_of(off + k),
                |_, remote| remote,
            );
            off += count;
        }
    }

    /// Physically swap the qubits at a local and a global position, then update
    /// the map.
    pub(super) fn relabel_swap(&mut self, local_pos: usize, global_pos: usize) {
        self.half_slice_swap(local_pos, global_pos);

        let local_q = self.phys_map[local_pos];
        let global_q = self.phys_map[global_pos];
        self.qubit_map[local_q] = global_pos;
        self.qubit_map[global_q] = local_pos;
        self.phys_map.swap(local_pos, global_pos);
        self.refresh_map_identity();
    }

    /// Physically swap the qubits at positions `a < b` and update the map.
    /// When both positions are local, this runs the inner SWAP kernel. When one
    /// position is local and one is global, this reuses the half-slice relabel
    /// exchange. When both positions are global, this exchanges full slices
    /// between rank pairs whose two bits differ.
    pub(super) fn swap_physical_positions(&mut self, a: usize, b: usize) {
        debug_assert!(
            a < b,
            "positions must be ordered: branch selection assumes a < b"
        );
        let local = self.local_qubits();
        if b < local {
            self.inner
                .apply(&Instruction::Gate {
                    gate: Gate::Swap,
                    targets: smallvec![a, b],
                })
                .expect("local SWAP cannot fail");
        } else if a < local {
            self.relabel_swap(a, b);
            return;
        } else {
            self.swap_global_slices(a, b);
        }
        let qa = self.phys_map[a];
        let qb = self.phys_map[b];
        self.qubit_map[qa] = b;
        self.qubit_map[qb] = a;
        self.phys_map.swap(a, b);
        self.refresh_map_identity();
    }

    /// Physically reorder the state until every circuit qubit occupies its own
    /// position. Runs in lockstep on every rank because the maps are identical.
    /// Each misplaced qubit costs at most one exchange. The identity map
    /// returns without work.
    pub(super) fn restore_identity_map(&mut self) {
        while !self.map_identity {
            let Some(pos) = (0..self.num_qubits).find(|&p| self.phys_map[p] != p) else {
                break;
            };
            let src = self.qubit_map[pos];
            self.swap_physical_positions(pos, src);
        }
    }

    /// Translate an instruction into physical positions: map the targets and
    /// rewrite qubit indices stored inside batched gate data. Borrows the gate
    /// unchanged while the map is the identity.
    pub(super) fn to_physical<'g>(
        &self,
        gate: &'g Gate,
        targets: &[usize],
    ) -> (Cow<'g, Gate>, SmallVec<[usize; 4]>) {
        if self.map_identity {
            return (Cow::Borrowed(gate), targets.into());
        }
        let ptargets: SmallVec<[usize; 4]> = targets.iter().map(|&q| self.qubit_map[q]).collect();
        // A payload whose every index maps to itself is borrowed unchanged:
        // once any relabel leaves the map non-identity, a per-application deep
        // clone of every batched payload is the steady state otherwise.
        let fixed = |q: usize| self.qubit_map[q] == q;
        let pgate = match gate {
            Gate::MultiFused(data) => {
                if data.gates.iter().all(|&(q, _)| fixed(q)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.gates {
                        entry.0 = self.qubit_map[entry.0];
                    }
                    Cow::Owned(Gate::MultiFused(data))
                }
            }
            Gate::Multi2q(data) => {
                if data.gates.iter().all(|&(q0, q1, _)| fixed(q0) && fixed(q1)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.gates {
                        entry.0 = self.qubit_map[entry.0];
                        entry.1 = self.qubit_map[entry.1];
                    }
                    Cow::Owned(Gate::Multi2q(data))
                }
            }
            Gate::BatchPhase(data) => {
                if data.phases.iter().all(|&(q, _)| fixed(q)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.phases {
                        entry.0 = self.qubit_map[entry.0];
                    }
                    Cow::Owned(Gate::BatchPhase(data))
                }
            }
            Gate::BatchRzz(data) => {
                if data.edges.iter().all(|&(q0, q1, _)| fixed(q0) && fixed(q1)) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.edges {
                        entry.0 = self.qubit_map[entry.0];
                        entry.1 = self.qubit_map[entry.1];
                    }
                    Cow::Owned(Gate::BatchRzz(data))
                }
            }
            Gate::DiagonalBatch(data) => {
                let entry_fixed = |entry: &DiagEntry| match *entry {
                    DiagEntry::Phase1q { qubit, .. } => fixed(qubit),
                    DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                        fixed(q0) && fixed(q1)
                    }
                };
                if data.entries.iter().all(entry_fixed) {
                    Cow::Borrowed(gate)
                } else {
                    let mut data = data.clone();
                    for entry in &mut data.entries {
                        match entry {
                            DiagEntry::Phase1q { qubit, .. } => *qubit = self.qubit_map[*qubit],
                            DiagEntry::Phase2q { q0, q1, .. }
                            | DiagEntry::Parity2q { q0, q1, .. } => {
                                *q0 = self.qubit_map[*q0];
                                *q1 = self.qubit_map[*q1];
                            }
                        }
                    }
                    Cow::Owned(Gate::DiagonalBatch(data))
                }
            }
            _ => Cow::Borrowed(gate),
        };
        (pgate, ptargets)
    }

    /// Whether every physical position the translated instruction touches,
    /// including indices inside batched gate data, is below the local boundary.
    pub(super) fn instruction_qubits_local(&self, gate: &Gate, targets: &[usize]) -> bool {
        let local = self.local_qubits();
        let mut all = true;
        for_each_gate_qubit(gate, targets, |q| all &= q < local);
        all
    }

    /// Translate a circuit-qubit bit mask into physical positions.
    pub(super) fn to_physical_mask(&self, mask: usize) -> usize {
        if self.map_identity {
            return mask;
        }
        let mut out = 0usize;
        for (q, &pos) in self.qubit_map.iter().enumerate() {
            out |= ((mask >> q) & 1) << pos;
        }
        out
    }

    /// Reorder a gathered dense vector from physical to circuit qubit order.
    pub(super) fn unpermuted<T: Copy + Default>(&self, phys: Vec<T>) -> Vec<T> {
        if self.map_identity {
            return phys;
        }
        let mut out = vec![T::default(); phys.len()];
        for (c, slot) in out.iter_mut().enumerate() {
            let mut p = 0usize;
            for (q, &pos) in self.qubit_map.iter().enumerate() {
                p |= ((c >> q) & 1) << pos;
            }
            *slot = phys[p];
        }
        out
    }

    /// Physical position of a circuit qubit. Identity before `init` runs the
    /// map setup or while no relabeling has occurred.
    #[inline]
    pub(super) fn physical_qubit(&self, qubit: usize) -> usize {
        if self.map_identity {
            qubit
        } else {
            self.qubit_map[qubit]
        }
    }
}
