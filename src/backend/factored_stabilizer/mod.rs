//! Factored stabilizer simulation backend.
//!
//! Keeps one sub-tableau per disentangled qubit group, merging two groups when
//! a gate bridges them (O((a+b)²/64) per merge) and splitting when measurement
//! reveals product structure.
//!
//! # Memory layout
//!
//! - One Aaronson-Gottesman sub-tableau per group: 2n+1 rows of bit-packed
//!   X words then Z words plus a phase bit per row, for an n-qubit group.
//! - Each group lists its global qubits; merging tensors two tableaux, and
//!   measurement or reset triggers a split check that factors disentangled
//!   qubits back out.
//!
//! # Gate support
//!
//! Clifford gates only: H, S, Sdg, SX, SXdg, X, Y, Z, Id, CX, CZ, SWAP.
//! Non-Clifford gates return `BackendUnsupported`, and fusion is skipped
//! (`supports_fused_gates` is false). Pauli expectations are answered per
//! cluster and multiplied, so marginals and observables run at any width.
//!
//! # When to prefer this backend
//!
//! - Large Clifford circuits with sparse entanglement, where per-group
//!   tableaux beat one O(n²) tableau. Auto dispatch routes such circuits here.
//!
//! # When NOT to use this backend
//!
//! - Densely entangling Clifford circuits; merges converge to one joint
//!   tableau and the dense stabilizer backend does the same work without
//!   merge and split overhead.
//! - Any circuit containing a non-Clifford gate.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::stabilizer::kernels::{self, rowops};
use crate::backend::stabilizer::{dense, diagnostics};
use crate::backend::{
    Backend, dense_probability_len, dense_statevector_len, reduced_density, reserve_dense_output,
    schmidt,
};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};
use std::borrow::Cow;

#[cfg(feature = "parallel")]
use crate::backend::MIN_QUBITS_FOR_PAR_GATES;

#[cfg(test)]
mod tests;

/// How far a sub-tableau's rows can be trusted beyond the stabilizer group.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Destabs {
    /// Gates sweep only the stabilizer half, so destabilizer rows are stale.
    Stale,
    /// Destabilizer `i` anticommutes with stabilizer `i` and commutes with the
    /// other stabilizers.
    Live,
    /// Live, with stabilizer row `n + q` the only stabilizer set at one of
    /// qubit `q`'s two bits (its pivot) and destabilizer `q` the unit Pauli on
    /// the other axis. Every generator then lies inside one entanglement
    /// component.
    Reduced,
}

struct SubTableau {
    n: usize,
    num_words: usize,
    xz: Vec<u64>,
    phase: Vec<bool>,
    qubits: SmallVec<[usize; 8]>,
    destabs: Destabs,
}

impl SubTableau {
    fn new_single(global_qubit: usize) -> Self {
        let n = 1;
        let num_words = 1;
        let stride = 2;
        let total_rows = 3;
        let mut xz = vec![0u64; total_rows * stride];
        let phase = vec![false; total_rows];
        xz[0] = 1;
        xz[stride + num_words] = 1;
        SubTableau {
            n,
            num_words,
            xz,
            phase,
            qubits: SmallVec::from_elem(global_qubit, 1),
            destabs: Destabs::Reduced,
        }
    }

    #[inline(always)]
    fn stride(&self) -> usize {
        2 * self.num_words
    }

    /// Position of `global` in the cluster's qubit list, which every merge
    /// and split keeps sorted.
    fn local_qubit(&self, global: usize) -> usize {
        self.qubits.binary_search(&global).unwrap()
    }

    #[inline(always)]
    fn bit(&self, row: usize, axis: usize, q: usize) -> bool {
        self.xz[row * self.stride() + axis + q / 64] >> (q % 64) & 1 != 0
    }

    /// Word offset of the axis holding qubit `q`'s pivot in a reduced
    /// tableau: 0 for X, `num_words` for Z.
    #[inline(always)]
    fn pivot_axis(&self, q: usize) -> usize {
        if self.bit(q, self.num_words, q) {
            0
        } else {
            self.num_words
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn par_rows(&self) -> bool {
        self.n >= MIN_QUBITS_FOR_PAR_GATES
    }

    #[cfg(not(feature = "parallel"))]
    #[inline(always)]
    fn par_rows(&self) -> bool {
        false
    }

    /// Stabilizer-half row window for gate kernels (rows n..2n+1), skipping
    /// the destabilizers and marking them stale for the next measurement to
    /// rebuild.
    #[inline(always)]
    fn gate_rows(&mut self) -> (&mut [u64], &mut [bool]) {
        self.destabs = Destabs::Stale;
        let start = self.n * self.stride();
        (&mut self.xz[start..], &mut self.phase[self.n..])
    }

    #[inline(always)]
    fn apply_h(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::h_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_s(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::s_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_sdg(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::sdg_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_x(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::x_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_y(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::y_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_z(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::z_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_sx(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::sx_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_sxdg(&mut self, a: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::sxdg_all(xz, phase, nw, par, a);
    }

    #[inline(always)]
    fn apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::cx_all(xz, phase, nw, par, ctrl, tgt);
    }

    #[inline(always)]
    fn apply_cz(&mut self, a: usize, b: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::cz_all(xz, phase, nw, par, a, b);
    }

    #[inline(always)]
    fn apply_swap(&mut self, a: usize, b: usize) {
        let par = self.par_rows();
        let nw = self.num_words;
        let (xz, phase) = self.gate_rows();
        rowops::swap_all(xz, phase, nw, par, a, b);
    }

    /// Batched gate application over this cluster's rows, reusing the dense
    /// backend's word grouping. Worth the buffering only once a row spans
    /// enough words that one pass per gate is the cost; below that the cluster
    /// sits in cache and the per-gate path is already free.
    fn apply_gate_run(&mut self, instructions: &[Instruction]) -> Result<()> {
        if self.num_words < kernels::MIN_WORDS_FOR_BATCH {
            for instruction in instructions {
                let Instruction::Gate { gate, targets } = instruction else {
                    unreachable!("a gate run holds gates only")
                };
                self.dispatch_gate(gate, targets)?;
            }
            return Ok(());
        }
        kernels::apply_gates_word_batch(self, instructions)
    }

    fn dispatch_gate(&mut self, gate: &Gate, local_targets: &[usize]) -> Result<()> {
        match gate {
            Gate::H => self.apply_h(local_targets[0]),
            Gate::S => self.apply_s(local_targets[0]),
            Gate::Sdg => self.apply_sdg(local_targets[0]),
            Gate::X => self.apply_x(local_targets[0]),
            Gate::Y => self.apply_y(local_targets[0]),
            Gate::Z => self.apply_z(local_targets[0]),
            Gate::SX => self.apply_sx(local_targets[0]),
            Gate::SXdg => self.apply_sxdg(local_targets[0]),
            Gate::Id => {}
            Gate::Cx => self.apply_cx(local_targets[0], local_targets[1]),
            Gate::Cz => self.apply_cz(local_targets[0], local_targets[1]),
            Gate::Swap => self.apply_swap(local_targets[0], local_targets[1]),
            _ => {
                return Err(PrismError::BackendUnsupported {
                    backend: "factored-stabilizer".to_string(),
                    operation: format!(
                        "non-Clifford gate `{}` (factored stabilizer supports Clifford gates only)",
                        gate.name()
                    ),
                });
            }
        }
        Ok(())
    }

    fn rowmul(&mut self, h: usize, i: usize) {
        rowops::rowmul(&mut self.xz, &mut self.phase, self.num_words, h, i);
    }

    fn zero_row(&mut self, r: usize) {
        rowops::zero_row(&mut self.xz, &mut self.phase, self.num_words, r);
    }

    fn set_unit_row(&mut self, row: usize, axis: usize, q: usize) {
        self.zero_row(row);
        let at = row * self.stride() + axis + q / 64;
        self.xz[at] |= 1u64 << (q % 64);
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        let stride = self.stride();
        let (lo, hi) = (a.min(b), a.max(b));
        let (head, tail) = self.xz.split_at_mut(hi * stride);
        head[lo * stride..(lo + 1) * stride].swap_with_slice(&mut tail[..stride]);
        self.phase.swap(a, b);
    }

    /// Multiply stabilizer row `pivot` into every other stabilizer row set at
    /// bit (`axis`, `q`), leaving `pivot` the only one.
    fn eliminate(&mut self, axis: usize, q: usize, pivot: usize) {
        let at = axis + q / 64;
        let mask = 1u64 << (q % 64);
        for r in self.n..2 * self.n {
            if r != pivot && self.xz[r * self.stride() + at] & mask != 0 {
                self.rowmul(r, pivot);
            }
        }
    }

    fn reduce(&mut self) {
        rowops::materialize_destabilizers(&mut self.xz, &mut self.phase, self.n, self.num_words);
        self.destabs = Destabs::Reduced;
    }

    /// Measure Z on `local_q`, leaving the tableau reduced with stabilizer row
    /// `n + local_q` equal to ±Z there and destabilizer `local_q` equal to X.
    ///
    /// A tableau that is not reduced is reduced once. On a reduced one a
    /// random outcome first moves the qubit's pivot to its X bit, with at most
    /// two pivot exchanges, so that row is the only stabilizer anticommuting
    /// with Z; the collapse then rewrites it and clears one bit in the others.
    /// That is O(k · words) against the O(k² · words) of a reduction.
    fn measure(&mut self, local_q: usize, rng: &mut ChaCha8Rng) -> bool {
        if self.destabs != Destabs::Reduced {
            self.reduce();
        }
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();
        let word = local_q / 64;
        let bit_mask = 1u64 << (local_q % 64);
        let row_q = n + local_q;

        let Some(first) = (n..2 * n).find(|&r| self.xz[r * stride + word] & bit_mask != 0) else {
            debug_assert!(self.is_unit_row(row_q, nw, local_q));
            return self.phase[row_q];
        };

        if self.pivot_axis(local_q) == 0 {
            debug_assert_eq!(first, row_q);
        } else if self.bit(row_q, 0, local_q) {
            self.eliminate(0, local_q, row_q);
        } else {
            // Commuting reduced rows have a symmetric off-pivot pattern: row
            // `first` set at X on `local_q` means row `row_q` is set at the
            // off-pivot bit of qubit `d`. That bit becomes d's pivot, held by
            // `row_q`, and X on `local_q` passes to row `first`.
            let d = first - n;
            let old_axis = self.pivot_axis(d);
            let new_axis = nw - old_axis;
            debug_assert!(self.bit(row_q, new_axis, d));
            self.eliminate(0, local_q, first);
            self.eliminate(new_axis, d, row_q);
            self.swap_rows(first, row_q);
            self.set_unit_row(d, old_axis, d);
        }

        let outcome: bool = rng.random();
        let z_word = nw + word;
        for r in n..2 * n {
            let at = r * stride + z_word;
            if r != row_q && self.xz[at] & bit_mask != 0 {
                self.xz[at] &= !bit_mask;
                self.phase[r] ^= outcome;
            }
        }
        self.set_unit_row(row_q, nw, local_q);
        self.phase[row_q] = outcome;
        self.set_unit_row(local_q, 0, local_q);
        outcome
    }

    fn is_unit_row(&self, row: usize, axis: usize, q: usize) -> bool {
        let base = row * self.stride();
        (0..self.stride()).all(|w| {
            let expected = if w == axis + q / 64 {
                1u64 << (q % 64)
            } else {
                0
            };
            self.xz[base + w] == expected
        })
    }

    fn reset_qubit(&mut self, local_q: usize, rng: &mut ChaCha8Rng) {
        self.measure(local_q, rng);
        self.phase[self.n + local_q] = false;
    }

    /// Rows with current destabilizers for a read-only query: borrowed when
    /// they are live, a materialized copy when gates have left them stale.
    fn rows_with_destabilizers(&self) -> rowops::TableauRows<'_> {
        if self.destabs != Destabs::Stale {
            return (Cow::Borrowed(&self.xz), Cow::Borrowed(&self.phase));
        }
        let mut xz = self.xz.clone();
        let mut phase = self.phase.clone();
        rowops::materialize_destabilizers(&mut xz, &mut phase, self.n, self.num_words);
        (Cow::Owned(xz), Cow::Owned(phase))
    }

    fn compute_probabilities(&self, limits: &ResolvedLimits) -> Result<Vec<f64>> {
        dense::dense_probabilities(
            &self.xz,
            &self.phase,
            self.n,
            self.num_words,
            self.stride(),
            "factored-stabilizer",
            limits,
        )
    }

    fn compute_statevector(&self, limits: &ResolvedLimits) -> Result<Vec<Complex64>> {
        dense::dense_statevector(
            &self.xz,
            &self.phase,
            self.n,
            self.num_words,
            self.stride(),
            "factored-stabilizer",
            "statevector",
            limits,
        )
    }
}

/// Factored stabilizer simulation: dynamic sub-tableau partitioning.
pub struct FactoredStabilizerBackend {
    num_qubits: usize,
    qubit_to_sub: Vec<usize>,
    subs: Vec<Option<SubTableau>>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    /// Merged-cluster qubit cap is `limits.stabilizer_cluster_qubits`, fixed at
    /// construction so the per-merge check is a field compare.
    limits: ResolvedLimits,
}

impl kernels::BatchTarget for SubTableau {
    fn num_words(&self) -> usize {
        self.num_words
    }

    fn batch_rows(&mut self) -> kernels::GateRows<'_> {
        self.destabs = Destabs::Stale;
        kernels::GateRows {
            n: self.n,
            num_words: self.num_words,
            row_start: self.n,
            xz: &mut self.xz,
            phase: &mut self.phase,
        }
    }

    fn apply_one(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        self.dispatch_gate(gate, targets)
    }
}

impl FactoredStabilizerBackend {
    /// Backend under the process default [`Limits`](crate::Limits).
    pub fn new(seed: u64) -> Self {
        Self::with_limits(seed, &ResolvedLimits::process_default())
    }

    pub(crate) fn with_limits(seed: u64, limits: &ResolvedLimits) -> Self {
        Self {
            num_qubits: 0,
            limits: *limits,
            qubit_to_sub: Vec::new(),
            subs: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Start from a tableau in the layout
    /// [`StabilizerBackend::init_from_tableau`](crate::backend::stabilizer::StabilizerBackend::init_from_tableau)
    /// documents, held as one cluster over every qubit until a measurement or
    /// reset splits it. Same checks as that import, and the joint tableau is
    /// held to the merged-cluster qubit cap.
    pub fn init_from_tableau(
        &mut self,
        num_qubits: usize,
        mut words: Vec<u64>,
        mut phases: Vec<bool>,
        num_classical_bits: usize,
    ) -> Result<()> {
        let n = num_qubits;
        rowops::check_imported_rows(n, &words, &phases)?;
        if n > self.limits.stabilizer_cluster_qubits.value {
            return Err(crate::backend::stabilizer_cluster_error(
                n,
                self.limits.stabilizer_cluster_qubits,
            ));
        }
        let num_words = n.div_ceil(64);
        words[2 * n * 2 * num_words..].fill(0);
        phases[2 * n] = false;
        self.num_qubits = n;
        self.qubit_to_sub = vec![0; n];
        self.subs.clear();
        self.subs.push(Some(SubTableau {
            n,
            num_words,
            xz: words,
            phase: phases,
            qubits: (0..n).collect(),
            destabs: Destabs::Live,
        }));
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    /// The state as one joint tableau in the layout
    /// [`StabilizerBackend::init_from_tableau`](crate::backend::stabilizer::StabilizerBackend::init_from_tableau)
    /// accepts: each cluster's rows, destabilizers materialized, scattered to
    /// their global qubit columns, with the clusters' generator pairs laid out
    /// one cluster after another.
    pub fn export_tableau(&self) -> (Vec<u64>, Vec<bool>) {
        let n = self.num_qubits;
        let nw = n.div_ceil(64);
        let stride = 2 * nw;
        let mut xz = vec![0u64; (2 * n + 1) * stride];
        let mut phase = vec![false; 2 * n + 1];
        let mut offset = 0;
        for sub in self.subs.iter().flatten() {
            let (sub_xz, sub_phase) = sub.rows_with_destabilizers();
            let k = sub.n;
            let sub_stride = sub.stride();
            for r in 0..2 * k {
                let dst = if r < k {
                    offset + r
                } else {
                    n + offset + r - k
                };
                phase[dst] = sub_phase[r];
                let src = &sub_xz[r * sub_stride..(r + 1) * sub_stride];
                let row = &mut xz[dst * stride..(dst + 1) * stride];
                for (local, &global) in sub.qubits.iter().enumerate() {
                    let bit = 1u64 << (local % 64);
                    if src[local / 64] & bit != 0 {
                        row[global / 64] |= 1u64 << (global % 64);
                    }
                    if src[sub.num_words + local / 64] & bit != 0 {
                        row[nw + global / 64] |= 1u64 << (global % 64);
                    }
                }
            }
            offset += k;
        }
        (xz, phase)
    }

    fn ensure_same_sub(&mut self, targets: &[usize]) -> Result<usize> {
        let first = self.qubit_to_sub[targets[0]];
        let mut need_merge: SmallVec<[usize; 4]> = SmallVec::new();
        for &q in &targets[1..] {
            let s = self.qubit_to_sub[q];
            if s != first && !need_merge.contains(&s) {
                need_merge.push(s);
            }
        }
        for other in need_merge {
            self.merge_subs(first, other)?;
        }
        Ok(first)
    }

    fn merge_subs(&mut self, dst_idx: usize, src_idx: usize) -> Result<()> {
        let total_n =
            self.subs[dst_idx].as_ref().unwrap().n + self.subs[src_idx].as_ref().unwrap().n;
        if total_n > self.limits.stabilizer_cluster_qubits.value {
            return Err(crate::backend::stabilizer_cluster_error(
                total_n,
                self.limits.stabilizer_cluster_qubits,
            ));
        }

        let src = self.subs[src_idx].take().unwrap();
        let d = self.subs[dst_idx].as_mut().unwrap();

        let a = d.n;
        let b = src.n;
        let new_nw = total_n.div_ceil(64);
        let new_stride = 2 * new_nw;
        let total_rows = 2 * total_n + 1;

        let mut merged_qubits: SmallVec<[usize; 8]> = SmallVec::with_capacity(total_n);
        let mut dst_positions: SmallVec<[usize; 8]> = SmallVec::new();
        let mut src_positions: SmallVec<[usize; 8]> = SmallVec::new();

        let (mut di, mut si) = (0, 0);
        while di < a || si < b {
            if di < a && (si >= b || d.qubits[di] < src.qubits[si]) {
                dst_positions.push(merged_qubits.len());
                merged_qubits.push(d.qubits[di]);
                di += 1;
            } else {
                src_positions.push(merged_qubits.len());
                merged_qubits.push(src.qubits[si]);
                si += 1;
            }
        }
        let dst_runs = position_runs(&dst_positions);
        let src_runs = position_runs(&src_positions);

        // Destabilizer rows are not copied: a merge only happens on the way
        // into a cross-cluster gate, which stales them immediately, so the
        // merged cluster starts lazy and the next measurement rebuilds them.
        let mut new_xz = vec![0u64; total_rows * new_stride];
        let mut new_phase = vec![false; total_rows];
        for r in 0..a {
            remap_row(
                &d.xz,
                d.stride(),
                a + r,
                &dst_runs,
                d.num_words,
                &mut new_xz,
                new_stride,
                total_n + r,
                new_nw,
            );
            new_phase[total_n + r] = d.phase[a + r];
        }
        d.xz = new_xz;
        d.phase = new_phase;
        for r in 0..b {
            remap_row(
                &src.xz,
                src.stride(),
                b + r,
                &src_runs,
                src.num_words,
                &mut d.xz,
                new_stride,
                total_n + a + r,
                new_nw,
            );
            d.phase[total_n + a + r] = src.phase[b + r];
        }

        d.n = total_n;
        d.num_words = new_nw;
        d.qubits = merged_qubits;
        d.destabs = Destabs::Stale;

        for &q in &src.qubits {
            self.qubit_to_sub[q] = dst_idx;
        }
        Ok(())
    }

    /// Factor a reduced cluster into its entanglement components.
    ///
    /// Reduced rows keep each generator inside one component, and commutation
    /// makes the off-pivot pattern symmetric (row `c` is set on qubit `d`
    /// exactly when row `d` is set on `c`). A search that ORs each reached
    /// qubit's row support into a visited mask therefore finds every component
    /// in O(k · words).
    fn try_split(&mut self, sub_idx: usize) -> bool {
        let sub = self.subs[sub_idx].as_ref().unwrap();
        let k = sub.n;
        if k <= 1 {
            return false;
        }
        debug_assert_eq!(sub.destabs, Destabs::Reduced);
        let nw = sub.num_words;
        let stride = sub.stride();

        let mut seen = vec![0u64; nw];
        let mut qubit_component = vec![0usize; k];
        let mut stack: Vec<usize> = Vec::new();
        let mut num_components = 0usize;
        for start in 0..k {
            if seen[start / 64] >> (start % 64) & 1 != 0 {
                continue;
            }
            seen[start / 64] |= 1u64 << (start % 64);
            stack.push(start);
            let mut size = 0;
            while let Some(c) = stack.pop() {
                qubit_component[c] = num_components;
                size += 1;
                let (row_x, row_z) = sub.xz[(k + c) * stride..(k + c + 1) * stride].split_at(nw);
                for (w, seen_w) in seen.iter_mut().enumerate() {
                    let mut fresh = (row_x[w] | row_z[w]) & !*seen_w;
                    *seen_w |= fresh;
                    while fresh != 0 {
                        stack.push(w * 64 + fresh.trailing_zeros() as usize);
                        fresh &= fresh - 1;
                    }
                }
            }
            if size == k {
                return false;
            }
            num_components += 1;
        }

        let sub = self.subs[sub_idx].take().unwrap();
        let mut comp_qubits: Vec<SmallVec<[usize; 8]>> = vec![SmallVec::new(); num_components];
        for (q, &c) in qubit_component.iter().enumerate() {
            comp_qubits[c].push(q);
        }

        for local_qs in &comp_qubits {
            let cn = local_qs.len();
            let cnw = cn.div_ceil(64);
            let cstride = 2 * cnw;
            let mut cxz = vec![0u64; (2 * cn + 1) * cstride];
            let mut cphase = vec![false; 2 * cn + 1];
            let runs = consecutive_runs(local_qs);

            // Destabilizer row lq and stabilizer row k + lq both belong to
            // qubit lq, so the pair lands at local rows ci and cn + ci.
            for (ci, &lq) in local_qs.iter().enumerate() {
                for (src_row, dst_row) in [(lq, ci), (k + lq, cn + ci)] {
                    let src = &sub.xz[src_row * stride..(src_row + 1) * stride];
                    let dst = &mut cxz[dst_row * cstride..(dst_row + 1) * cstride];
                    let (dst_x, dst_z) = dst.split_at_mut(cnw);
                    for &(start, len, to) in &runs {
                        or_bit_run(&src[..nw], start, len, dst_x, to);
                        or_bit_run(&src[nw..], start, len, dst_z, to);
                    }
                    cphase[dst_row] = sub.phase[src_row];
                }
            }

            let new_sub = SubTableau {
                n: cn,
                num_words: cnw,
                xz: cxz,
                phase: cphase,
                qubits: local_qs.iter().map(|&lq| sub.qubits[lq]).collect(),
                destabs: Destabs::Reduced,
            };

            let slot = self.find_free_slot();
            for &gq in &new_sub.qubits {
                self.qubit_to_sub[gq] = slot;
            }
            self.subs[slot] = Some(new_sub);
        }

        true
    }

    /// Ebits across the cut at `subsystem`, summed over the clusters it
    /// crosses. A cluster lying wholly on one side of the cut carries none.
    fn cut_rank(&self, subsystem: &[usize]) -> usize {
        let mut local: Vec<Vec<usize>> = vec![Vec::new(); self.subs.len()];
        for &q in subsystem {
            let index = self.qubit_to_sub[q];
            local[index].push(self.subs[index].as_ref().unwrap().local_qubit(q));
        }
        let mut total = 0;
        for (sub, local) in self.subs.iter().zip(&local) {
            let Some(sub) = sub else { continue };
            if local.is_empty() || local.len() == sub.n {
                continue;
            }
            total += diagnostics::subsystem_rank(&sub.xz, sub.n, sub.num_words, local);
        }
        total
    }

    fn find_free_slot(&mut self) -> usize {
        for (i, s) in self.subs.iter().enumerate() {
            if s.is_none() {
                return i;
            }
        }
        self.subs.push(None);
        self.subs.len() - 1
    }
}

impl crate::backend::sealed::Sealed for FactoredStabilizerBackend {
    fn limits(&self) -> crate::limits::ResolvedLimits {
        self.limits
    }
}

impl Backend for FactoredStabilizerBackend {
    fn name(&self) -> &'static str {
        "factored-stabilizer"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::FactoredStabilizer
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.qubit_to_sub.clear();
        self.subs.clear();
        for q in 0..num_qubits {
            self.qubit_to_sub.push(q);
            self.subs.push(Some(SubTableau::new_single(q)));
        }
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    /// Batch each run of gates that stays inside one cluster.
    ///
    /// A run ends at the first instruction that is not a gate or whose targets
    /// reach outside the cluster the run is on, so a merge still happens where
    /// the per-instruction path would have done it. Targets are rewritten to
    /// cluster-local indices once per run into a reused buffer, which also
    /// retires the per-gate `SmallVec` the default loop built.
    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        let mut run: Vec<Instruction> = Vec::new();
        let mut at = 0;
        while at < instructions.len() {
            let Instruction::Gate { targets, .. } = &instructions[at] else {
                self.apply(&instructions[at])?;
                at += 1;
                continue;
            };
            let cluster = self.ensure_same_sub(targets)?;

            run.clear();
            while at < instructions.len() {
                let Instruction::Gate { gate, targets } = &instructions[at] else {
                    break;
                };
                if targets.iter().any(|&q| self.qubit_to_sub[q] != cluster) {
                    break;
                }
                let sub = self.subs[cluster].as_ref().unwrap();
                let local: SmallVec<[usize; 4]> =
                    targets.iter().map(|&t| sub.local_qubit(t)).collect();
                run.push(Instruction::Gate {
                    gate: gate.clone(),
                    targets: local,
                });
                at += 1;
            }

            self.subs[cluster].as_mut().unwrap().apply_gate_run(&run)?;
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => {
                let ss = self.ensure_same_sub(targets)?;
                let sub = self.subs[ss].as_mut().unwrap();
                let mut local = SmallVec::<[usize; 4]>::new();
                for &t in targets.iter() {
                    local.push(sub.local_qubit(t));
                }
                sub.dispatch_gate(gate, &local)?;
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                let ss = self.qubit_to_sub[*qubit];
                let sub = self.subs[ss].as_mut().unwrap();
                let local_q = sub.local_qubit(*qubit);
                let outcome = sub.measure(local_q, &mut self.rng);
                self.classical_bits[*classical_bit] = outcome;
                self.try_split(ss);
            }
            Instruction::Reset { qubit } => {
                let ss = self.qubit_to_sub[*qubit];
                let sub = self.subs[ss].as_mut().unwrap();
                let local_q = sub.local_qubit(*qubit);
                sub.reset_qubit(local_q, &mut self.rng);
                self.try_split(ss);
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied(
                    "FactoredStabilizer",
                    label,
                ));
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    let ss = self.ensure_same_sub(targets)?;
                    let sub = self.subs[ss].as_mut().unwrap();
                    let mut local = SmallVec::<[usize; 4]>::new();
                    for &t in targets.iter() {
                        local.push(sub.local_qubit(t));
                    }
                    sub.dispatch_gate(gate, &local)?;
                }
            }
            Instruction::Region(region) => self.apply_region(region)?,
        }
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        dense_probability_len(self.name(), self.num_qubits, &self.limits)?;
        let active: Vec<&SubTableau> = self.subs.iter().filter_map(|s| s.as_ref()).collect();

        if active.len() == 1 && active[0].n == self.num_qubits {
            return active[0].compute_probabilities(&self.limits);
        }

        let blocks: Vec<(Vec<f64>, Vec<usize>)> = active
            .iter()
            .map(|sub| {
                let probs = sub.compute_probabilities(&self.limits)?;
                let qubits: Vec<usize> = sub.qubits.to_vec();
                Ok((probs, qubits))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(crate::sim::merge_probabilities(&blocks, self.num_qubits))
    }

    fn block_probabilities(&self) -> Option<crate::sim::Probabilities> {
        // `FactoredBlock::mask` is a u64 and `Probabilities::len` is
        // `1 << total_qubits`, so a wider register has no representable lazy
        // form either; the dense terminal declines it.
        if self.num_qubits > 64 {
            return None;
        }

        let active: Vec<&SubTableau> = self.subs.iter().filter_map(|s| s.as_ref()).collect();
        if active.len() < 2 {
            return None;
        }

        let mut blocks = Vec::with_capacity(active.len());
        for sub in active {
            let probs = sub.compute_probabilities(&self.limits).ok()?;
            let mask = sub.qubits.iter().fold(0u64, |m, &q| m | 1 << q);
            blocks.push(crate::sim::FactoredBlock { probs, mask });
        }

        Some(crate::sim::Probabilities::Factored {
            blocks,
            total_qubits: self.num_qubits,
        })
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Clusters are mutually unentangled, so a joint string factors into one
    /// string per cluster it touches and the value is the product of the
    /// cluster expectations. Every cluster's packed string lives in one scratch
    /// buffer, so a request packs its factors in a single pass and allocates
    /// nothing per string. A cluster with stale destabilizers is rebuilt once,
    /// on the first string that reaches it.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        for observable in observables {
            crate::sim::validate_observable(observable, self.num_qubits)?;
        }

        let mut rows: Vec<Option<rowops::TableauRows<'_>>> = Vec::with_capacity(self.subs.len());
        let mut offsets = Vec::with_capacity(self.subs.len());
        let mut total = 0;
        let mut max_nw = 0;
        for sub in &self.subs {
            offsets.push(total);
            rows.push(None);
            if let Some(sub) = sub {
                total += 2 * sub.num_words;
                max_nw = max_nw.max(sub.num_words);
            }
        }
        let mut scratch = vec![0u64; total + 2 * max_nw];
        let (terms, acc) = scratch.split_at_mut(total);
        let mut marked = vec![false; self.subs.len()];
        let mut touched: SmallVec<[usize; 8]> = SmallVec::new();

        observables
            .iter()
            .map(|observable| {
                for term in observable {
                    let s = self.qubit_to_sub[term.qubit];
                    let sub = self.subs[s].as_ref().unwrap();
                    let nw = sub.num_words;
                    let region = &mut terms[offsets[s]..offsets[s] + 2 * nw];
                    if !marked[s] {
                        marked[s] = true;
                        touched.push(s);
                        region.fill(0);
                    }
                    let q = rowops::QubitBit::of(sub.local_qubit(term.qubit));
                    match term.axis {
                        PauliAxis::X => region[q.word] |= q.mask,
                        PauliAxis::Z => region[nw + q.word] |= q.mask,
                        PauliAxis::Y => {
                            region[q.word] |= q.mask;
                            region[nw + q.word] |= q.mask;
                        }
                    }
                }

                let mut product = 1.0f64;
                for &s in &touched {
                    marked[s] = false;
                    if product == 0.0 {
                        continue;
                    }
                    let sub = self.subs[s].as_ref().unwrap();
                    let (xz, phase) = rows[s].get_or_insert_with(|| sub.rows_with_destabilizers());
                    let nw = sub.num_words;
                    product *= rowops::pauli_expectation(
                        xz,
                        phase,
                        sub.n,
                        nw,
                        &terms[offsets[s]..offsets[s] + 2 * nw],
                        &mut acc[..2 * nw],
                    );
                }
                touched.clear();
                Ok(product)
            })
            .collect()
    }

    /// `2^r` copies of `2^(-r/2)` for the rank `r` of the cut, as the
    /// stabilizer backend, priced against the dense export cap.
    fn schmidt_values(&mut self, subsystem: &[usize]) -> Result<Vec<f64>> {
        schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        diagnostics::flat_spectrum(self.name(), self.cut_rank(subsystem), &self.limits)
    }

    /// The cut's rank times `ln 2`, at any width. Clusters are unentangled, so
    /// a cut that crosses several of them adds up and one lying wholly on
    /// either side contributes nothing.
    fn entanglement_entropy(&mut self, subsystem: &[usize]) -> Result<f64> {
        schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        Ok(self.cut_rank(subsystem) as f64 * std::f64::consts::LN_2)
    }

    /// Kronecker product of the per-cluster marginals, each the projector
    /// its sub-tableau's generators inside the subsystem define.
    fn reduced_density_matrix(&mut self, subsystem: &[usize]) -> Result<Vec<Complex64>> {
        schmidt::validate_qubit_set(subsystem, self.num_qubits)?;
        let dim =
            reduced_density::reduced_density_side(self.name(), subsystem.len(), &self.limits)?;
        let mut rho = vec![Complex64::new(1.0, 0.0); dim * dim];
        for (index, sub) in self.subs.iter().enumerate() {
            let Some(sub) = sub else { continue };
            let positions: Vec<usize> = (0..subsystem.len())
                .filter(|&i| self.qubit_to_sub[subsystem[i]] == index)
                .collect();
            if positions.is_empty() {
                continue;
            }
            let local: Vec<usize> = positions
                .iter()
                .map(|&i| sub.local_qubit(subsystem[i]))
                .collect();
            let side = 1usize << local.len();
            let factor = diagnostics::subsystem_density(
                &sub.xz,
                &sub.phase,
                sub.n,
                sub.num_words,
                &local,
                side,
            );
            reduced_density::multiply_block_factor(&mut rho, dim, &positions, &factor, side);
        }
        Ok(rho)
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        let ss = self.qubit_to_sub[qubit];
        let sub = self.subs[ss].as_mut().unwrap();
        let local_q = sub.local_qubit(qubit);
        sub.reset_qubit(local_q, &mut self.rng);
        self.try_split(ss);
        Ok(())
    }

    /// Tensor the live sub-tableaux back into a joint amplitude vector.
    ///
    /// Clusters are mutually unentangled, so the joint amplitude at a global
    /// index is the product of each cluster's amplitude at the index restricted
    /// to that cluster's qubits. `init` gives every qubit a cluster and the
    /// merge and split passes preserve the partition, so the product covers the
    /// whole register.
    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(self.name(), "statevector", self.num_qubits, &self.limits)?;
        let active: Vec<&SubTableau> = self.subs.iter().filter_map(|s| s.as_ref()).collect();

        if active.len() == 1 && active[0].n == self.num_qubits {
            return active[0].compute_statevector(&self.limits);
        }

        let blocks: Vec<(Vec<Complex64>, &[usize])> = active
            .iter()
            .map(|sub| {
                Ok((
                    sub.compute_statevector(&self.limits)?,
                    sub.qubits.as_slice(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;

        let mut joint = Vec::new();
        reserve_dense_output(&mut joint, dim, self.name(), "statevector")?;
        joint.resize(dim, Complex64::new(0.0, 0.0));
        for (index, amp) in joint.iter_mut().enumerate() {
            let mut product = Complex64::new(1.0, 0.0);
            for (state, qubits) in &blocks {
                let mut local = 0usize;
                for (bit, &q) in qubits.iter().enumerate() {
                    local |= ((index >> q) & 1) << bit;
                }
                product *= state[local];
            }
            *amp = product;
        }
        Ok(joint)
    }
}

/// Stretches of local qubits that keep their spacing in the merged order, as
/// `(start, len, shift)`: local bits `start..start + len` land at `start + shift`.
/// Merged qubit lists are sorted, so a run only breaks where the other
/// cluster's qubits interleave, and a run count near the smaller cluster's
/// width is the norm.
fn position_runs(positions: &[usize]) -> SmallVec<[(usize, usize, usize); 8]> {
    let mut runs = SmallVec::new();
    let mut start = 0;
    while start < positions.len() {
        let shift = positions[start] - start;
        let mut end = start + 1;
        while end < positions.len() && positions[end] - end == shift {
            end += 1;
        }
        runs.push((start, end - start, shift));
        start = end;
    }
    runs
}

/// OR bits `start..start + len` of `src` into `dst` at `start + shift`, a word
/// at a time. Every moved bit must fit in `dst`.
fn or_shifted_bits(src: &[u64], start: usize, len: usize, shift: usize, dst: &mut [u64]) {
    let end = start + len;
    let (dw_off, sh) = (shift / 64, shift % 64);
    let mut bit = start;
    while bit < end {
        let w = bit / 64;
        let lo = bit % 64;
        let hi = (end - w * 64).min(64);
        let chunk = src[w] & ((u64::MAX >> (64 - (hi - lo))) << lo);
        let dw = w + dw_off;
        dst[dw] |= chunk << sh;
        if sh != 0 {
            let spill = chunk >> (64 - sh);
            if spill != 0 {
                dst[dw + 1] |= spill;
            }
        }
        bit = w * 64 + hi;
    }
}

#[allow(clippy::too_many_arguments)]
fn remap_row(
    src_xz: &[u64],
    src_stride: usize,
    src_row: usize,
    runs: &[(usize, usize, usize)],
    src_nw: usize,
    dst_xz: &mut [u64],
    dst_stride: usize,
    dst_row: usize,
    dst_nw: usize,
) {
    let src_base = src_row * src_stride;
    let dst_base = dst_row * dst_stride;
    let (src_x, src_z) = src_xz[src_base..src_base + 2 * src_nw].split_at(src_nw);
    let (dst_x, dst_z) = dst_xz[dst_base..dst_base + 2 * dst_nw].split_at_mut(dst_nw);
    for &(start, len, shift) in runs {
        or_shifted_bits(src_x, start, len, shift, dst_x);
        or_shifted_bits(src_z, start, len, shift, dst_z);
    }
}

/// Maximal stretches of consecutive values in sorted `positions`, as
/// `(start, len, to)`: values `start..start + len` sit at indices `to..to + len`.
fn consecutive_runs(positions: &[usize]) -> SmallVec<[(usize, usize, usize); 8]> {
    let mut runs = SmallVec::new();
    let mut to = 0;
    while to < positions.len() {
        let start = positions[to];
        let mut len = 1;
        while to + len < positions.len() && positions[to + len] == start + len {
            len += 1;
        }
        runs.push((start, len, to));
        to += len;
    }
    runs
}

/// OR bits `start..start + len` of `src` into `dst` at `to..to + len`, one
/// destination word at a time.
fn or_bit_run(src: &[u64], start: usize, len: usize, dst: &mut [u64], to: usize) {
    let mut done = 0;
    while done < len {
        let at = to + done;
        let take = (64 - at % 64).min(len - done);
        let from = start + done;
        let (w, off) = (from / 64, from % 64);
        let mut chunk = src[w] >> off;
        if off + take > 64 {
            chunk |= src[w + 1] << (64 - off);
        }
        chunk &= u64::MAX >> (64 - take);
        dst[at / 64] |= chunk << (at % 64);
        done += take;
    }
}
