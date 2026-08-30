//! Sparse state-vector simulation backend.
//!
//! Stores only non-zero amplitudes in a map keyed by basis-state index, giving
//! O(k) memory where k is the number of non-zero basis states. Amplitudes below
//! a configurable epsilon are pruned after each gate to maintain sparsity.
//!
//! Every gate walks the map, so the map hashes basis-state indices with the
//! crate's multiply-xor hasher rather than the stdlib default.
//!
//! # Memory layout
//!
//! - `FxHashMap<usize, Complex64>` keyed by basis-state index, non-zero
//!   amplitudes only.
//! - A retained swap-buffer map lets amplitude-moving gates rebuild the state
//!   without reallocating.
//!
//! # Gate support
//!
//! The full gate set, including MCU and every fused or batched variant.
//! Diagonal gates scale amplitudes in place without moving keys. Shot
//! sampling and Pauli expectations run natively on the map.
//!
//! # When to prefer this backend
//!
//! - States with few non-zero amplitudes (computational basis states, limited superposition).
//! - Large qubit counts where the state stays sparse throughout the circuit.
//! - Classical-like circuits with limited branching.
//!
//! # When NOT to use this backend
//!
//! - After a layer of Hadamard gates (state becomes maximally dense).
//! - Small qubit counts where dense statevector is faster due to HashMap overhead.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const MIN_STATES_FOR_PAR: usize = 4096;

use crate::backend::{
    Backend, BasisSamples, dense_probability_len, dense_statevector_len, is_phase_one,
    reserve_dense_output,
};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{Gate, diag_entries_phase, is_diagonal_2x2};
use crate::hash::FxHashMap;
use crate::sim::unified_pauli::PauliTerm;

const DEFAULT_EPSILON: f64 = 1e-16;

/// Sparse state-vector backend, O(k) where k is the number of non-zero amplitudes.
pub struct SparseBackend {
    num_qubits: usize,
    state: FxHashMap<usize, Complex64>,
    swap_buf: FxHashMap<usize, Complex64>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    epsilon: f64,
    /// [`crate::backend::max_sparse_entries`] read once at construction, so the
    /// per-gate growth check is a field compare rather than an atomic load.
    entry_cap: usize,
}

impl SparseBackend {
    /// Create a new sparse backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            state: FxHashMap::default(),
            swap_buf: FxHashMap::default(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            epsilon: DEFAULT_EPSILON,
            entry_cap: crate::backend::max_sparse_entries(),
        }
    }

    /// Number of nonzero amplitudes currently stored, the `k` in the O(k) cost
    /// of every gate walk.
    pub fn entry_count(&self) -> usize {
        self.state.len()
    }

    #[inline(always)]
    fn prune(&mut self) {
        let eps = self.epsilon;
        self.state.retain(|_, amp| amp.norm_sqr() >= eps);
    }

    /// Reject a gate whose worst-case fan-out would grow the map past the
    /// entry budget. `factor` is the per-source-entry fan-out of the caller
    /// (2 for a branching 1q gate, 4 for a dense 2q), so rejection can fire
    /// one gate early on a state that would have deduplicated below the cap.
    #[inline(always)]
    fn check_entry_growth(&self, factor: usize) -> Result<()> {
        let projected = self.state.len().saturating_mul(factor);
        if projected > self.entry_cap {
            return Err(self.entry_growth_error(projected));
        }
        Ok(())
    }

    #[cold]
    fn entry_growth_error(&self, projected: usize) -> crate::error::PrismError {
        crate::error::PrismError::IncompatibleBackend {
            backend: "sparse".to_string(),
            reason: format!(
                "state holds {} entries and this gate can reach {projected}, \
                 exceeding the cap of {} entries on this machine \
                 (set PRISM_MAX_SPARSE_QUBITS to override)",
                self.state.len(),
                self.entry_cap
            ),
        }
    }

    #[inline(always)]
    fn apply_single_qubit(&mut self, target: usize, mat: [[Complex64; 2]; 2]) -> Result<()> {
        if is_diagonal_2x2(&mat) {
            self.apply_diagonal_1q(target, mat[0][0], mat[1][1]);
            return Ok(());
        }
        self.check_entry_growth(2)?;

        let mask = 1usize << target;
        let zero = Complex64::new(0.0, 0.0);
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len() * 2);

        for (&idx, &amp) in &self.state {
            let bit = (idx >> target) & 1;
            let partner = idx ^ mask;

            *self.swap_buf.entry(idx).or_insert(zero) += mat[bit][bit] * amp;
            *self.swap_buf.entry(partner).or_insert(zero) += mat[1 - bit][bit] * amp;
        }

        std::mem::swap(&mut self.state, &mut self.swap_buf);
        self.prune();
        Ok(())
    }

    /// A diagonal 2x2 scales amplitudes in place: no partner entries, no map rebuild.
    /// A sub-unit diagonal (a Kraus operator) can shrink amplitudes below epsilon,
    /// so only that case pays the prune pass.
    #[inline(always)]
    fn apply_diagonal_1q(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        for (idx, amp) in self.state.iter_mut() {
            *amp *= if (*idx >> target) & 1 == 1 { d1 } else { d0 };
        }
        if d0.norm_sqr() < 1.0 - 1e-12 || d1.norm_sqr() < 1.0 - 1e-12 {
            self.prune();
        }
    }

    /// CX is a deterministic 1:1 index mapping. No near-zero amplitudes are created.
    #[inline(always)]
    fn apply_cx(&mut self, control: usize, target: usize) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len());
        self.swap_buf.extend(self.state.drain().map(|(idx, amp)| {
            if idx & ctrl_mask != 0 {
                (idx ^ tgt_mask, amp)
            } else {
                (idx, amp)
            }
        }));
        std::mem::swap(&mut self.state, &mut self.swap_buf);
    }

    #[inline(always)]
    fn apply_cz(&mut self, q0: usize, q1: usize) {
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        for (&idx, amp) in self.state.iter_mut() {
            if idx & mask0 != 0 && idx & mask1 != 0 {
                *amp = -*amp;
            }
        }
    }

    #[inline(always)]
    fn apply_swap(&mut self, q0: usize, q1: usize) {
        let m0 = 1usize << q0;
        let m1 = 1usize << q1;
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len());
        self.swap_buf.extend(self.state.drain().map(|(idx, amp)| {
            let bit0 = (idx >> q0) & 1;
            let bit1 = (idx >> q1) & 1;
            if bit0 != bit1 {
                (idx ^ m0 ^ m1, amp)
            } else {
                (idx, amp)
            }
        }));
        std::mem::swap(&mut self.state, &mut self.swap_buf);
    }

    #[inline(always)]
    fn apply_cu(&mut self, control: usize, target: usize, mat: [[Complex64; 2]; 2]) -> Result<()> {
        self.check_entry_growth(2)?;
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let zero = Complex64::new(0.0, 0.0);
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len() * 2);

        for (&idx, &amp) in &self.state {
            if idx & ctrl_mask == 0 {
                *self.swap_buf.entry(idx).or_insert(zero) += amp;
            } else {
                let bit = (idx >> target) & 1;
                let partner = idx ^ tgt_mask;
                *self.swap_buf.entry(idx).or_insert(zero) += mat[bit][bit] * amp;
                *self.swap_buf.entry(partner).or_insert(zero) += mat[1 - bit][bit] * amp;
            }
        }

        std::mem::swap(&mut self.state, &mut self.swap_buf);
        self.prune();
        Ok(())
    }

    #[inline(always)]
    fn apply_mcu(
        &mut self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) -> Result<()> {
        self.check_entry_growth(2)?;
        let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
        let tgt_mask = 1usize << target;
        let zero = Complex64::new(0.0, 0.0);
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len() * 2);

        for (&idx, &amp) in &self.state {
            if idx & ctrl_mask != ctrl_mask {
                *self.swap_buf.entry(idx).or_insert(zero) += amp;
            } else {
                let bit = (idx >> target) & 1;
                let partner = idx ^ tgt_mask;
                *self.swap_buf.entry(idx).or_insert(zero) += mat[bit][bit] * amp;
                *self.swap_buf.entry(partner).or_insert(zero) += mat[1 - bit][bit] * amp;
            }
        }

        std::mem::swap(&mut self.state, &mut self.swap_buf);
        self.prune();
        Ok(())
    }

    #[inline(always)]
    fn apply_cu_phase(&mut self, control: usize, target: usize, phase: Complex64) {
        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        for (&idx, amp) in self.state.iter_mut() {
            if idx & ctrl_mask != 0 && idx & tgt_mask != 0 {
                *amp *= phase;
            }
        }
    }

    #[inline(always)]
    fn apply_mcu_phase(&mut self, controls: &[usize], target: usize, phase: Complex64) {
        let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
        let tgt_mask = 1usize << target;
        for (&idx, amp) in self.state.iter_mut() {
            if idx & ctrl_mask == ctrl_mask && idx & tgt_mask != 0 {
                *amp *= phase;
            }
        }
    }

    #[inline(always)]
    fn apply_rzz(&mut self, q0: usize, q1: usize, theta: f64) {
        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        for (idx, amp) in self.state.iter_mut() {
            let parity = ((*idx >> q0) ^ (*idx >> q1)) & 1;
            *amp *= if parity == 0 { phase_same } else { phase_diff };
        }
    }

    /// One walk over the map with the per-edge phase pair precomputed, instead of
    /// one walk (and two `from_polar`) per edge.
    fn apply_batch_rzz(&mut self, edges: &[(usize, usize, f64)]) {
        let phases: Vec<(usize, usize, [Complex64; 2])> = edges
            .iter()
            .map(|&(q0, q1, theta)| {
                let same = Complex64::from_polar(1.0, -theta / 2.0);
                let diff = Complex64::from_polar(1.0, theta / 2.0);
                (q0, q1, [same, diff])
            })
            .collect();

        for (idx, amp) in self.state.iter_mut() {
            let mut combined = Complex64::new(1.0, 0.0);
            for &(q0, q1, pair) in &phases {
                combined *= pair[((*idx >> q0) ^ (*idx >> q1)) & 1];
            }
            *amp *= combined;
        }
    }

    fn apply_batch_phase(&mut self, control: usize, phases: &[(usize, Complex64)]) {
        let ctrl_mask = 1usize << control;
        let one = Complex64::new(1.0, 0.0);
        for (&idx, amp) in self.state.iter_mut() {
            if idx & ctrl_mask == 0 {
                continue;
            }
            let mut combined = one;
            for &(target, phase) in phases {
                if idx & (1usize << target) != 0 {
                    combined *= phase;
                }
            }
            if !is_phase_one(combined) {
                *amp *= combined;
            }
        }
    }

    /// Exact structural test for a monomial 4x4: one nonzero per source basis
    /// state and per destination. Structural zeros only, no tolerance, so a
    /// matrix product carrying float dust off its pattern stays on the general
    /// path.
    fn monomial_4x4(mat: &[[Complex64; 4]; 4]) -> Option<([usize; 4], [Complex64; 4])> {
        let zero = Complex64::new(0.0, 0.0);
        let mut dest = [usize::MAX; 4];
        let mut scale = [zero; 4];
        let mut used = 0u8;
        for (col, mat_row) in mat.iter().enumerate() {
            for (row, &coeff) in mat_row.iter().enumerate() {
                if coeff != zero {
                    if dest[row] != usize::MAX {
                        return None;
                    }
                    dest[row] = col;
                    scale[row] = coeff;
                }
            }
        }
        for &d in &dest {
            if d == usize::MAX || used & (1 << d) != 0 {
                return None;
            }
            used |= 1 << d;
        }
        Some((dest, scale))
    }

    /// A monomial 4x4 maps each occupied basis state to exactly one
    /// destination, so no cancellation can arise, the invariant `apply_cx`
    /// states. A sub-unit scale (a normalized Kraus branch routed through a
    /// fused payload) can still shrink amplitudes below epsilon, so only that
    /// case pays the prune pass, as `apply_diagonal_1q` does.
    fn apply_monomial_2q(&mut self, q0: usize, q1: usize, dest: [usize; 4], scale: [Complex64; 4]) {
        if dest == [0, 1, 2, 3] {
            for (idx, amp) in self.state.iter_mut() {
                let row = ((*idx >> q0) & 1) * 2 + ((*idx >> q1) & 1);
                *amp *= scale[row];
            }
        } else {
            let mask0 = 1usize << q0;
            let mask1 = 1usize << q1;
            self.swap_buf.clear();
            self.swap_buf.reserve(self.state.len());
            self.swap_buf.extend(self.state.drain().map(|(idx, amp)| {
                let row = ((idx >> q0) & 1) * 2 + ((idx >> q1) & 1);
                let col = dest[row];
                let new_idx = idx & !(mask0 | mask1) | (((col >> 1) & 1) << q0) | ((col & 1) << q1);
                (new_idx, amp * scale[row])
            }));
            std::mem::swap(&mut self.state, &mut self.swap_buf);
        }
        if scale.iter().any(|c| c.norm_sqr() < 1.0 - 1e-12) {
            self.prune();
        }
    }

    fn apply_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) -> Result<()> {
        if let Some((dest, scale)) = Self::monomial_4x4(mat) {
            self.apply_monomial_2q(q0, q1, dest, scale);
            return Ok(());
        }
        self.check_entry_growth(4)?;
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let zero = Complex64::new(0.0, 0.0);
        self.swap_buf.clear();
        self.swap_buf.reserve(self.state.len() * 2);

        for (&idx, &amp) in &self.state {
            let bit0 = (idx >> q0) & 1;
            let bit1 = (idx >> q1) & 1;
            let row = bit0 * 2 + bit1;
            let base = idx & !(mask0 | mask1);

            for (col, mat_row) in mat.iter().enumerate() {
                let coeff = mat_row[row];
                if coeff == zero {
                    continue;
                }
                let col_bit0 = (col >> 1) & 1;
                let col_bit1 = col & 1;
                let dest = base | (col_bit0 << q0) | (col_bit1 << q1);
                *self.swap_buf.entry(dest).or_insert(zero) += coeff * amp;
            }
        }

        std::mem::swap(&mut self.state, &mut self.swap_buf);
        self.prune();
        Ok(())
    }

    fn masked_prob(&self, mask: usize, bit_set: bool) -> f64 {
        #[cfg(feature = "parallel")]
        if self.state.len() >= MIN_STATES_FOR_PAR {
            return self
                .state
                .par_iter()
                .filter(|&(&idx, _)| (idx & mask != 0) == bit_set)
                .map(|(_, amp)| amp.norm_sqr())
                .sum();
        }
        self.state
            .iter()
            .filter(|&(&idx, _)| (idx & mask != 0) == bit_set)
            .map(|(_, amp)| amp.norm_sqr())
            .sum()
    }

    fn apply_reset(&mut self, qubit: usize) {
        let mask = 1usize << qubit;
        let prob_one = self.masked_prob(mask, true);
        let outcome = self.rng.random::<f64>() < prob_one;
        let inv_norm = crate::backend::measurement_inv_norm(outcome, prob_one);

        if !outcome {
            self.state.retain(|&idx, amp| {
                if idx & mask == 0 {
                    *amp *= inv_norm;
                    true
                } else {
                    false
                }
            });
            return;
        }

        let folded: Vec<(usize, Complex64)> = self
            .state
            .drain()
            .filter(|(idx, _)| idx & mask != 0)
            .map(|(idx, amp)| (idx ^ mask, amp * inv_norm))
            .collect();
        self.state.extend(folded);
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let mask = 1usize << qubit;
        let prob_one = self.masked_prob(mask, true);

        let outcome = self.rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;

        let inv_norm = crate::backend::measurement_inv_norm(outcome, prob_one);

        self.state.retain(|&idx, amp| {
            let matches = (idx & mask != 0) == outcome;
            if matches {
                *amp *= inv_norm;
            }
            matches
        });
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Rzz(theta) => {
                self.apply_rzz(targets[0], targets[1], *theta);
            }
            Gate::Cx => {
                self.apply_cx(targets[0], targets[1]);
            }
            Gate::Cz => {
                self.apply_cz(targets[0], targets[1]);
            }
            Gate::Swap => {
                self.apply_swap(targets[0], targets[1]);
            }
            Gate::Cu(mat) => {
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_cu_phase(targets[0], targets[1], phase);
                } else {
                    self.apply_cu(targets[0], targets[1], **mat)?;
                }
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_mcu_phase(&targets[..num_ctrl], targets[num_ctrl], phase);
                } else {
                    self.apply_mcu(&targets[..num_ctrl], targets[num_ctrl], data.mat)?;
                }
            }
            Gate::BatchPhase(data) => {
                self.apply_batch_phase(targets[0], &data.phases);
            }
            Gate::BatchRzz(data) => {
                self.apply_batch_rzz(&data.edges);
            }
            Gate::DiagonalBatch(data) => {
                for (idx, amp) in self.state.iter_mut() {
                    *amp *= diag_entries_phase(*idx, &data.entries);
                }
            }
            Gate::MultiFused(data) => {
                for &(target, mat) in &data.gates {
                    self.apply_single_qubit(target, mat)?;
                }
            }
            Gate::Fused2q(mat) => {
                self.apply_fused_2q(targets[0], targets[1], mat)?;
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_fused_2q(q0, q1, mat)?;
                }
            }
            other => {
                debug_assert!(
                    targets.len() == 1,
                    "sparse dispatch_gate: unexpected multi-qubit gate {:?}",
                    other
                );
                let mat = other.matrix_2x2();
                self.apply_single_qubit(targets[0], mat)?;
            }
        }
        Ok(())
    }
}

impl Backend for SparseBackend {
    fn name(&self) -> &'static str {
        "sparse"
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::Sparse
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.state.clear();
        self.state.insert(0, Complex64::new(1.0, 0.0));
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit);
            }
            Instruction::Barrier { .. } => {}
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets)?;
                }
            }
            Instruction::Region(region) => self.apply_region(region)?,
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit);
        Ok(())
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.apply_single_qubit(qubit, *matrix)
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let mask = 1usize << qubit;
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        let mut r = Complex64::new(0.0, 0.0);

        for (&idx, &amp) in &self.state {
            if idx & mask == 0 {
                p0 += amp.norm_sqr();
                if let Some(&amp_one) = self.state.get(&(idx | mask)) {
                    r += amp_one * amp.conj();
                }
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
        let dim = dense_probability_len(self.name(), self.num_qubits)?;
        let mut probs = Vec::new();
        reserve_dense_output(&mut probs, dim, self.name(), "probabilities")?;
        probs.resize(dim, 0.0f64);
        for (&idx, amp) in &self.state {
            probs[idx] = amp.norm_sqr();
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Samples from a CDF over the `k` stored amplitudes instead of `2^n`.
    ///
    /// Entries are ordered by basis index, which is the order the dense route
    /// accumulates its CDF in, so the same seed draws the same basis states
    /// the dense path would have drawn.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let mut indices: Vec<usize> = self.state.keys().copied().collect();
        indices.sort_unstable();
        let probs: Vec<f64> = indices
            .iter()
            .map(|idx| self.state[idx].norm_sqr())
            .collect();
        let cdf = crate::sim::shots::build_cdf(&probs);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        for shot in 0..num_shots {
            let r: f64 = rng.random();
            samples.set_index(shot, indices[crate::sim::shots::sample_from_cdf(&cdf, r)]);
        }
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Every observable is accumulated in one walk of the stored amplitudes.
    /// A Z-only observable has `xmask == 0`, so its partner is the entry
    /// itself and it costs a popcount instead of a hash lookup; the two
    /// families are accumulated separately for that reason.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let masks = observables
            .iter()
            .map(|observable| crate::sim::pauli_masks(observable, self.num_qubits))
            .collect::<Result<Vec<_>>>()?;

        let norm: f64 = self.state.values().map(|amp| amp.norm_sqr()).sum();
        if norm == 0.0 {
            return Ok(vec![0.0; masks.len()]);
        }

        let z_only: Vec<usize> = masks
            .iter()
            .filter(|&&(xmask, _, _)| xmask == 0)
            .map(|&(_, zmask, _)| zmask)
            .collect();
        let general: Vec<(usize, usize)> = masks
            .iter()
            .filter(|&&(xmask, _, _)| xmask != 0)
            .map(|&(xmask, zmask, _)| (xmask, zmask))
            .collect();

        let mut z_sum = vec![0.0f64; z_only.len()];
        let mut g_sum = vec![Complex64::new(0.0, 0.0); general.len()];
        for (&idx, &amp) in &self.state {
            let parity_sign = |mask: usize| {
                if (idx & mask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                }
            };
            let norm_sqr = amp.norm_sqr();
            for (slot, &zmask) in z_sum.iter_mut().zip(&z_only) {
                *slot += norm_sqr * parity_sign(zmask);
            }
            for (slot, &(xmask, zmask)) in g_sum.iter_mut().zip(&general) {
                let Some(&partner) = self.state.get(&(idx ^ xmask)) else {
                    continue;
                };
                *slot += partner.conj() * amp * parity_sign(zmask);
            }
        }

        Ok(crate::sim::finish_expectations(
            &masks, &z_sum, &g_sum, norm,
        ))
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(self.name(), "statevector export", self.num_qubits)?;
        let mut sv = Vec::new();
        reserve_dense_output(&mut sv, dim, self.name(), "statevector export")?;
        sv.resize(dim, Complex64::new(0.0, 0.0));
        for (&idx, &amp) in &self.state {
            sv[idx] = amp;
        }
        Ok(sv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::sim;

    const EPS: f64 = 1e-12;

    fn run_sparse(circuit: &Circuit) -> SparseBackend {
        let mut b = SparseBackend::new(42);
        sim::run_on(&mut b, circuit).unwrap();
        b
    }

    fn run_sparse_probs(circuit: &Circuit) -> Vec<f64> {
        let b = run_sparse(circuit);
        b.probabilities().unwrap()
    }

    #[test]
    fn test_init_zero_state() {
        let mut b = SparseBackend::new(42);
        b.init(3, 0).unwrap();
        assert_eq!(b.state.len(), 1);
        assert!((b.state[&0].re - 1.0).abs() < EPS);
    }

    #[test]
    fn test_x_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::X, &[0]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        assert!(b.state.contains_key(&1));
        assert!((b.state[&1].norm() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_h_creates_superposition() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 2);
        assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
        assert!((b.state[&1].norm_sqr() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_hh_is_identity() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[0]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        assert!((b.state[&0].re - 1.0).abs() < EPS);
    }

    #[test]
    fn test_cx_bell_state() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 2);
        assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
        assert!((b.state[&3].norm_sqr() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_cz_phase() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Cz, &[0, 1]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        assert!((b.state[&3].re - (-1.0)).abs() < EPS);
    }

    #[test]
    fn test_swap() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Swap, &[0, 1]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        assert!(b.state.contains_key(&1));
    }

    #[test]
    fn test_rx_pi() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(std::f64::consts::PI), &[0]);
        let probs = run_sparse_probs(&c);
        assert!(probs[0].abs() < EPS);
        assert!((probs[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_rz_preserves_sparsity() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rz(1.234), &[0]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        assert!((b.state[&0].norm() - 1.0).abs() < EPS);
    }

    #[test]
    fn test_measure_collapses() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        let outcome = b.classical_results()[0];
        if outcome {
            assert!(b.state.contains_key(&1));
        } else {
            assert!(b.state.contains_key(&0));
        }
    }

    #[test]
    fn test_measure_deterministic() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);

        let b1 = run_sparse(&c);
        let b2 = run_sparse(&c);
        assert_eq!(b1.classical_results()[0], b2.classical_results()[0]);
    }

    #[test]
    fn test_probs_bell() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let probs = run_sparse_probs(&c);
        assert!((probs[0] - 0.5).abs() < EPS);
        assert!(probs[1].abs() < EPS);
        assert!(probs[2].abs() < EPS);
        assert!((probs[3] - 0.5).abs() < EPS);
    }

    #[test]
    fn test_probs_zero_state() {
        let c = Circuit::new(3, 0);
        let probs = run_sparse_probs(&c);
        assert!((probs[0] - 1.0).abs() < EPS);
        let rest: f64 = probs[1..].iter().sum();
        assert!(rest.abs() < EPS);
    }

    #[test]
    fn test_pruning() {
        let mut b = SparseBackend::new(42);
        b.init(1, 0).unwrap();
        b.state.insert(1, Complex64::new(1e-20, 0.0));
        assert_eq!(b.state.len(), 2);
        b.prune();
        assert_eq!(b.state.len(), 1);
        assert!(b.state.contains_key(&0));
    }

    #[test]
    fn test_fused_gate() {
        let h_mat = Gate::H.matrix_2x2();
        let t_mat = Gate::T.matrix_2x2();
        let zero = Complex64::new(0.0, 0.0);
        let mut fused = [[zero; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    fused[i][j] += t_mat[i][k] * h_mat[k][j];
                }
            }
        }

        let mut c1 = Circuit::new(1, 0);
        c1.add_gate(Gate::H, &[0]);
        c1.add_gate(Gate::T, &[0]);
        let p1 = run_sparse_probs(&c1);

        let mut c2 = Circuit::new(1, 0);
        c2.add_gate(Gate::Fused(Box::new(fused)), &[0]);
        let p2 = run_sparse_probs(&c2);

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < EPS);
        }
    }

    #[test]
    fn test_ghz_4_sparse() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 2);
        assert!((b.state[&0].norm_sqr() - 0.5).abs() < EPS);
        assert!((b.state[&15].norm_sqr() - 0.5).abs() < EPS);
    }

    #[test]
    fn test_cu_phase_applies_phase() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::cphase(std::f64::consts::FRAC_PI_4), &[0, 1]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 1);
        let expected = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
        assert!((b.state[&3] - expected).norm() < EPS);
    }

    #[test]
    fn test_cu_phase_no_action_control_zero() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::cphase(1.0), &[0, 1]);
        let b = run_sparse(&c);
        let h = 1.0 / 2.0_f64.sqrt();
        assert!((b.state[&0].re - h).abs() < EPS);
        assert!((b.state[&2].re - h).abs() < EPS);
        assert!(!b.state.contains_key(&1));
        assert!(!b.state.contains_key(&3));
    }

    #[test]
    fn test_cu_phase_matches_cz() {
        let mut c1 = Circuit::new(2, 0);
        c1.add_gate(Gate::H, &[0]);
        c1.add_gate(Gate::H, &[1]);
        c1.add_gate(Gate::cphase(std::f64::consts::PI), &[0, 1]);

        let mut c2 = Circuit::new(2, 0);
        c2.add_gate(Gate::H, &[0]);
        c2.add_gate(Gate::H, &[1]);
        c2.add_gate(Gate::Cz, &[0, 1]);

        let b1 = run_sparse(&c1);
        let b2 = run_sparse(&c2);

        for (&idx, &amp1) in &b1.state {
            let amp2 = b2
                .state
                .get(&idx)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
        }
    }

    #[test]
    fn test_diagonal_1q_in_place() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::P(1.234), &[0]);
        let b = run_sparse(&c);
        assert_eq!(b.state.len(), 2);
        let h = 1.0 / 2.0_f64.sqrt();
        assert!((b.state[&0].re - h).abs() < EPS);
        assert!((b.state[&1] - Complex64::from_polar(h, 1.234)).norm() < EPS);
    }

    #[test]
    fn test_batch_rzz_matches_individual() {
        use crate::gates::BatchRzzData;

        let edges = vec![(0usize, 1usize, 0.7f64), (1, 2, 1.3)];

        let mut c1 = Circuit::new(3, 0);
        for q in 0..3 {
            c1.add_gate(Gate::H, &[q]);
        }
        for &(q0, q1, theta) in &edges {
            c1.add_gate(Gate::Rzz(theta), &[q0, q1]);
        }
        let b1 = run_sparse(&c1);

        let mut c2 = Circuit::new(3, 0);
        for q in 0..3 {
            c2.add_gate(Gate::H, &[q]);
        }
        c2.add_gate(Gate::BatchRzz(Box::new(BatchRzzData { edges })), &[0, 1, 2]);
        let b2 = run_sparse(&c2);

        for (&idx, &amp1) in &b1.state {
            let amp2 = b2
                .state
                .get(&idx)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
        }
    }

    #[test]
    fn test_diagonal_batch_matches_individual() {
        use crate::gates::{DiagEntry, DiagonalBatchData};

        let mut c1 = Circuit::new(3, 0);
        for q in 0..3 {
            c1.add_gate(Gate::H, &[q]);
        }
        c1.add_gate(Gate::S, &[0]);
        c1.add_gate(Gate::Cz, &[0, 1]);
        c1.add_gate(Gate::Rzz(0.9), &[1, 2]);
        let b1 = run_sparse(&c1);

        let s_mat = Gate::S.matrix_2x2();
        let mut c2 = Circuit::new(3, 0);
        for q in 0..3 {
            c2.add_gate(Gate::H, &[q]);
        }
        c2.add_gate(
            Gate::DiagonalBatch(Box::new(DiagonalBatchData {
                entries: vec![
                    DiagEntry::Phase1q {
                        qubit: 0,
                        d0: s_mat[0][0],
                        d1: s_mat[1][1],
                    },
                    DiagEntry::Phase2q {
                        q0: 0,
                        q1: 1,
                        phase: Complex64::new(-1.0, 0.0),
                    },
                    DiagEntry::Parity2q {
                        q0: 1,
                        q1: 2,
                        same: Complex64::from_polar(1.0, -0.45),
                        diff: Complex64::from_polar(1.0, 0.45),
                    },
                ],
            })),
            &[0, 1, 2],
        );
        let b2 = run_sparse(&c2);

        for (&idx, &amp1) in &b1.state {
            let amp2 = b2
                .state
                .get(&idx)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            assert!((amp1 - amp2).norm() < EPS, "mismatch at idx {idx}");
        }
    }

    #[test]
    fn test_batch_phase_matches_individual() {
        use crate::gates::BatchPhaseData;
        use smallvec::smallvec;

        let phase1 = Complex64::from_polar(1.0, 0.5);
        let phase2 = Complex64::from_polar(1.0, 1.2);

        let mut c1 = Circuit::new(3, 0);
        c1.add_gate(Gate::H, &[0]);
        c1.add_gate(Gate::H, &[1]);
        c1.add_gate(Gate::H, &[2]);
        c1.add_gate(Gate::cphase(0.5), &[0, 1]);
        c1.add_gate(Gate::cphase(1.2), &[0, 2]);
        let p1 = run_sparse_probs(&c1);

        let mut c2 = Circuit::new(3, 0);
        c2.add_gate(Gate::H, &[0]);
        c2.add_gate(Gate::H, &[1]);
        c2.add_gate(Gate::H, &[2]);
        c2.add_gate(
            Gate::BatchPhase(Box::new(BatchPhaseData {
                phases: smallvec![(1, phase1), (2, phase2)],
            })),
            &[0, 1, 2],
        );
        let p2 = run_sparse_probs(&c2);

        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < EPS, "probs mismatch: {a} vs {b}");
        }
    }
}
