//! Stabilizer (Clifford) simulation backend.
//!
//! Uses the Gottesman-Knill theorem: Clifford circuits can be simulated in
//! O(n²) time per gate using a stabilizer tableau of 2n Pauli strings.
//!
//! # Memory layout
//!
//! Aaronson-Gottesman (2004) tableau with (2n+1) rows:
//! - Rows 0..n: destabilizer generators
//! - Rows n..2n: stabilizer generators
//! - Row 2n: scratch row (measurement computation)
//!
//! Each row stores n X-bits, n Z-bits (bit-packed into `Vec<u64>`), and one
//! phase bit (true = -1). Total memory: O(n²/8) bytes.
//!
//! # Gate support
//!
//! Clifford gates only: H, S, Sdg, X, Y, Z, CX, CZ, SWAP, Id.
//! Non-Clifford gates (T, Rx, Ry, Rz, Fused) return `BackendUnsupported`.
//!
//! # When to prefer this backend
//!
//! - Clifford-only circuits (randomized benchmarking, error correction).
//! - Very large qubit counts (1000+) where statevector is impossible.
//! - Verification of Clifford subcircuits before layering T gates.
//!
//! # Performance characteristics
//!
//! - Gate application: O(n) per gate (iterates 2n+1 rows, constant work per row)
//! - Measurement: O(n²) worst case (rowmul is O(n), applied to up to 2n rows)
//! - Memory: n=1000 → ~500 KB, n=10000 → ~50 MB
//! - Probability extraction: O(2^n * n) — only available for n ≤ 20

use num_complex::Complex64;
use smallvec::SmallVec;

use crate::backend::{Backend, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub(crate) mod kernels;
#[cfg(test)]
mod tests;

use kernels::{rowmul_words, xor_words, MIN_WORDS_FOR_BATCH};

/// Clifford-only O(n^2) stabilizer simulation (Aaronson-Gottesman tableau).
#[derive(Clone)]
pub struct StabilizerBackend {
    pub(super) n: usize,
    pub(super) num_words: usize,
    pub(super) xz: Vec<u64>,
    pub(super) phase: Vec<bool>,
    pub(super) classical_bits: Vec<bool>,
    pub(super) rng: ChaCha8Rng,
    pub(super) qubit_active: Vec<Vec<u32>>,
    pub(super) total_weight: usize,
    pub(super) sgi_merge_buf: Vec<u32>,
    pub(super) sgi_new_a: Vec<u32>,
    pub(super) sgi_new_b: Vec<u32>,
    pub(super) sgi_max_active: usize,
    pub(super) lazy_destab: bool,
    pub(super) gate_row_start: usize,
}

impl StabilizerBackend {
    /// Create a new stabilizer backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            n: 0,
            num_words: 0,
            xz: Vec::new(),
            phase: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            qubit_active: Vec::new(),
            total_weight: 0,
            sgi_merge_buf: Vec::new(),
            sgi_new_a: Vec::new(),
            sgi_new_b: Vec::new(),
            sgi_max_active: 0,
            lazy_destab: false,
            gate_row_start: 0,
        }
    }

    pub fn new_lazy(seed: u64) -> Self {
        let mut s = Self::new(seed);
        s.lazy_destab = true;
        s
    }

    pub fn enable_lazy_destab(&mut self) {
        if self.lazy_destab || self.n == 0 {
            return;
        }
        self.lazy_destab = true;
        self.gate_row_start = self.n;
        let n = self.n;
        self.qubit_active = (0..n).map(|q| vec![(n + q) as u32]).collect();
        self.total_weight = n;
        self.sgi_max_active = 1;
    }

    pub(super) fn ensure_destabilizers(&mut self) {
        if !self.lazy_destab {
            return;
        }
        self.materialize_destabilizers();
        self.lazy_destab = false;
        self.gate_row_start = 0;
        let n = self.n;
        for q in 0..n {
            if !self.qubit_active[q].contains(&(q as u32)) {
                self.qubit_active[q].push(q as u32);
            }
        }
        self.total_weight = self.qubit_active.iter().map(|v| v.len()).sum();
        self.sgi_max_active = self.qubit_active.iter().map(|v| v.len()).max().unwrap_or(0);
    }

    fn materialize_destabilizers(&mut self) {
        let n = self.n;
        if n == 0 {
            return;
        }
        let nw = self.num_words;
        let stride = self.stride();

        for i in 0..n {
            let base = i * stride;
            for w in 0..stride {
                self.xz[base + w] = 0;
            }
            self.phase[i] = false;
        }

        let mut stab_copy: Vec<u64> = self.xz[n * stride..2 * n * stride].to_vec();
        let mut stab_phase: Vec<bool> = self.phase[n..2 * n].to_vec();

        for col in 0..n {
            let mut pivot = None;
            for row in col..n {
                let word = col / 64;
                let bit = col % 64;
                if stab_copy[row * stride + word] & (1u64 << bit) != 0 {
                    pivot = Some(row);
                    break;
                }
            }

            if pivot.is_none() {
                for row in col..n {
                    let word = col / 64;
                    let bit = col % 64;
                    if stab_copy[row * stride + nw + word] & (1u64 << bit) != 0 {
                        pivot = Some(row);
                        break;
                    }
                }

                if let Some(p) = pivot {
                    if p != col {
                        let col_off = col * stride;
                        let p_off = p * stride;
                        for w in 0..stride {
                            stab_copy.swap(col_off + w, p_off + w);
                        }
                        stab_phase.swap(col, p);
                    }

                    let word = col / 64;
                    let bit = col % 64;
                    let bit_mask = 1u64 << bit;

                    for row in 0..n {
                        if row == col {
                            continue;
                        }
                        if stab_copy[row * stride + nw + word] & bit_mask != 0 {
                            let src: Vec<u64> =
                                stab_copy[col * stride..(col + 1) * stride].to_vec();
                            let sp = stab_phase[col];
                            let dst = &mut stab_copy[row * stride..(row + 1) * stride];
                            let initial =
                                if sp { 2u64 } else { 0 } + if stab_phase[row] { 2u64 } else { 0 };
                            let (dx, dz) = dst.split_at_mut(nw);
                            let sum = rowmul_words(
                                dx,
                                &mut dz[..nw],
                                &src[..nw],
                                &src[nw..2 * nw],
                                initial,
                            );
                            stab_phase[row] = (sum & 3) >= 2;
                        }
                    }

                    self.xz[col * stride + word] |= bit_mask;
                    self.phase[col] = false;
                } else {
                    self.xz[col * stride + col / 64] |= 1u64 << (col % 64);
                    self.phase[col] = false;
                }
                continue;
            }

            let p = pivot.unwrap();
            if p != col {
                let col_off = col * stride;
                let p_off = p * stride;
                for w in 0..stride {
                    stab_copy.swap(col_off + w, p_off + w);
                }
                stab_phase.swap(col, p);
            }

            let word = col / 64;
            let bit = col % 64;
            let bit_mask = 1u64 << bit;

            for row in 0..n {
                if row == col {
                    continue;
                }
                if stab_copy[row * stride + word] & bit_mask != 0 {
                    let src: Vec<u64> = stab_copy[col * stride..(col + 1) * stride].to_vec();
                    let sp = stab_phase[col];
                    let dst = &mut stab_copy[row * stride..(row + 1) * stride];
                    let initial =
                        if sp { 2u64 } else { 0 } + if stab_phase[row] { 2u64 } else { 0 };
                    let (dx, dz) = dst.split_at_mut(nw);
                    let sum =
                        rowmul_words(dx, &mut dz[..nw], &src[..nw], &src[nw..2 * nw], initial);
                    stab_phase[row] = (sum & 3) >= 2;
                }
            }

            self.xz[col * stride + nw + word] |= bit_mask;
            self.phase[col] = false;
        }
    }

    pub fn raw_tableau(&self) -> (&[u64], &[bool]) {
        (&self.xz, &self.phase)
    }

    pub fn into_tableau(self) -> (Vec<u64>, Vec<bool>, usize, usize) {
        (self.xz, self.phase, self.n, self.num_words)
    }

    pub fn apply_gates_only(&mut self, instructions: &[Instruction]) -> Result<()> {
        let nw = self.num_words;
        if nw < MIN_WORDS_FOR_BATCH {
            for instruction in instructions {
                match instruction {
                    Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
                    Instruction::Conditional {
                        condition,
                        gate,
                        targets,
                    } => {
                        if condition.evaluate(&self.classical_bits) {
                            self.dispatch_gate(gate, targets)?;
                        }
                    }
                    _ => {}
                }
            }
            return Ok(());
        }

        if self.sgi_enabled() {
            return self.apply_gates_only_sgi(instructions);
        }

        self.apply_gates_only_word_batch(instructions)
    }

    /// Multiply row `h` by row `i` (replace `h` with the Pauli product).
    ///
    /// Fused phase+XOR: AG g-function with wordwise popcount, row XOR in the
    /// same loop to avoid a separate memory pass.
    pub(super) fn rowmul(&mut self, h: usize, i: usize) {
        let stride = self.stride();
        let nw = self.num_words;
        let base_h = h * stride;
        let base_i = i * stride;

        let initial_sum =
            if self.phase[i] { 2u64 } else { 0 } + if self.phase[h] { 2u64 } else { 0 };

        // SAFETY: h != i in all callers, so row regions [base_h..base_h+stride]
        // and [base_i..base_i+stride] are non-overlapping.
        let (dst_x, dst_z, src_x, src_z) = unsafe {
            let ptr = self.xz.as_mut_ptr();
            (
                std::slice::from_raw_parts_mut(ptr.add(base_h), nw),
                std::slice::from_raw_parts_mut(ptr.add(base_h + nw), nw),
                std::slice::from_raw_parts(ptr.add(base_i) as *const u64, nw),
                std::slice::from_raw_parts(ptr.add(base_i + nw) as *const u64, nw),
            )
        };

        let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
        self.phase[h] = (sum & 3) >= 2;
    }

    pub(super) fn copy_row(&mut self, dst: usize, src: usize) {
        let stride = self.stride();
        let src_start = src * stride;
        let dst_start = dst * stride;
        self.xz
            .copy_within(src_start..src_start + stride, dst_start);
        self.phase[dst] = self.phase[src];
    }

    pub(super) fn zero_row(&mut self, r: usize) {
        let stride = self.stride();
        let start = r * stride;
        self.xz[start..start + stride].fill(0);
        self.phase[r] = false;
    }

    pub(super) fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        self.ensure_destabilizers();
        self.apply_measure_with_info(qubit, classical_bit);
    }

    pub(super) fn apply_reset(&mut self, qubit: usize) -> Result<()> {
        self.ensure_destabilizers();
        let prev_len = self.classical_bits.len();
        self.classical_bits.push(false);
        let scratch = prev_len;
        self.apply_measure_with_info(qubit, scratch);
        let outcome = self.classical_bits[scratch];
        self.classical_bits.truncate(prev_len);
        if outcome {
            self.dispatch_gate(&Gate::X, &[qubit])?;
        }
        Ok(())
    }

    pub(crate) fn apply_measure_with_info(
        &mut self,
        qubit: usize,
        classical_bit: usize,
    ) -> (bool, Vec<usize>) {
        let n = self.n;
        let word = qubit / 64;
        let bit_mask = 1u64 << (qubit % 64);
        let stride = self.stride();

        let mut p: Option<usize> = None;
        for i in n..2 * n {
            if self.xz[i * stride + word] & bit_mask != 0 {
                p = Some(i);
                break;
            }
        }

        if let Some(p_row) = p {
            let p_base = p_row * stride;
            let mut support = Vec::new();
            for q in 0..n {
                if self.xz[p_base + q / 64] & (1u64 << (q % 64)) != 0 {
                    support.push(q);
                }
            }
            self.measure_random(p_row, word, bit_mask, classical_bit);
            (true, support)
        } else {
            let scratch = 2 * n;
            self.zero_row(scratch);

            for i in 0..n {
                if self.xz[i * stride + word] & bit_mask != 0 {
                    self.rowmul(scratch, i + n);
                }
            }

            let outcome = self.phase[scratch];
            self.classical_bits[classical_bit] = outcome;
            (false, Vec::new())
        }
    }

    pub(crate) fn batch_measure_ref_info(
        &mut self,
        measurements: &[(usize, usize)],
    ) -> (Vec<bool>, Vec<Vec<usize>>, Vec<bool>) {
        self.ensure_destabilizers();
        let num_meas = measurements.len();
        let n = self.n;
        let nw = self.num_words;
        let stride = self.stride();

        let mut is_random = vec![false; num_meas];
        let mut random_x_support: Vec<Vec<usize>> = vec![Vec::new(); num_meas];
        let mut outcomes = vec![false; num_meas];

        let mut qubit_to_meas: Vec<usize> = vec![usize::MAX; n];
        for (mi, &(qubit, _)) in measurements.iter().enumerate() {
            qubit_to_meas[qubit] = mi;
        }

        let mut first_destab = vec![usize::MAX; num_meas];
        let mut match_count = vec![0u16; num_meas];
        let mut match_a = vec![0usize; num_meas];
        let mut match_b = vec![0usize; num_meas];

        let build_index = |first_destab: &mut [usize],
                           match_count: &mut [u16],
                           match_a: &mut [usize],
                           match_b: &mut [usize],
                           xz: &[u64],
                           qubit_to_meas: &[usize],
                           n: usize,
                           nw: usize,
                           stride: usize,
                           num_meas: usize| {
            first_destab.iter_mut().for_each(|v| *v = usize::MAX);
            match_count.iter_mut().for_each(|v| *v = 0);
            for r in 0..2 * n {
                let r_base = r * stride;
                for w in 0..nw {
                    let x_word = xz[r_base + w];
                    if x_word == 0 {
                        continue;
                    }
                    let mut bits = x_word;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as usize;
                        let q = w * 64 + b;
                        if q < n {
                            let mi = qubit_to_meas[q];
                            if mi < num_meas {
                                if r >= n {
                                    if first_destab[mi] == usize::MAX {
                                        first_destab[mi] = r;
                                    }
                                } else {
                                    let c = match_count[mi];
                                    if c == 0 {
                                        match_a[mi] = r;
                                    } else if c == 1 {
                                        match_b[mi] = r;
                                    }
                                    match_count[mi] = c.saturating_add(1);
                                }
                            }
                        }
                        bits &= bits - 1;
                    }
                }
            }
        };

        build_index(
            &mut first_destab,
            &mut match_count,
            &mut match_a,
            &mut match_b,
            &self.xz,
            &qubit_to_meas,
            n,
            nw,
            stride,
            num_meas,
        );

        for mi in 0..num_meas {
            let (qubit, classical_bit) = measurements[mi];
            if first_destab[mi] != usize::MAX {
                let (_, support) = self.apply_measure_with_info(qubit, classical_bit);
                is_random[mi] = true;
                outcomes[mi] = self.classical_bits[classical_bit];
                random_x_support[mi] = support;
                build_index(
                    &mut first_destab,
                    &mut match_count,
                    &mut match_a,
                    &mut match_b,
                    &self.xz,
                    &qubit_to_meas,
                    n,
                    nw,
                    stride,
                    num_meas,
                );
            }
        }

        let mut all_diagonal = true;
        'diag_check: for i in 0..n {
            let base = (i + n) * stride;
            for w in 0..nw {
                if self.xz[base + w] != 0 {
                    all_diagonal = false;
                    break 'diag_check;
                }
            }
        }

        if all_diagonal {
            for i in 0..n {
                let phase_i = self.phase[i + n];
                if !phase_i {
                    continue;
                }
                let base = i * stride;
                for w in 0..nw {
                    let x_word = self.xz[base + w];
                    if x_word == 0 {
                        continue;
                    }
                    let mut bits = x_word;
                    while bits != 0 {
                        let b = bits.trailing_zeros() as usize;
                        let q = w * 64 + b;
                        if q < n {
                            let mi = qubit_to_meas[q];
                            if mi < num_meas && !is_random[mi] {
                                outcomes[mi] ^= true;
                            }
                        }
                        bits &= bits - 1;
                    }
                }
            }
            for mi in 0..num_meas {
                if !is_random[mi] {
                    self.classical_bits[measurements[mi].1] = outcomes[mi];
                }
            }
        } else {
            for mi in 0..num_meas {
                if is_random[mi] {
                    continue;
                }
                let (qubit, classical_bit) = measurements[mi];
                let word = qubit / 64;
                let bit_mask = 1u64 << (qubit % 64);
                let scratch = 2 * n;
                self.zero_row(scratch);
                for i in 0..n {
                    if self.xz[i * stride + word] & bit_mask != 0 {
                        self.rowmul(scratch, i + n);
                    }
                }
                outcomes[mi] = self.phase[scratch];
                self.classical_bits[classical_bit] = outcomes[mi];
            }
        }
        (is_random, random_x_support, outcomes)
    }

    /// Export the stabilizer state as a dense statevector.
    ///
    /// Constructs the 2^n amplitude vector by projecting |0...0⟩ through each
    /// stabilizer generator: |ψ⟩ = ∏_i (I + g_i)/2 |seed⟩, normalized.
    ///
    /// Each projection applies Pauli string g_i to the dense vector in O(2^n),
    /// giving O(n × 2^n) total — same complexity as `compute_probabilities`.
    pub fn export_statevector(&self) -> Result<Vec<Complex64>> {
        if self.n >= usize::BITS as usize {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits (exceeds addressable memory)",
                    self.n
                ),
            });
        }
        let dim = 1usize << self.n;
        let mut check = Vec::<Complex64>::new();
        if check.try_reserve_exact(dim).is_err() {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits ({} bytes required)",
                    self.n,
                    dim * std::mem::size_of::<Complex64>()
                ),
            });
        }
        drop(check);
        Ok(self.compute_statevector())
    }

    /// Build the dense statevector by projecting a support seed through each
    /// stabilizer generator.
    ///
    /// Algorithm:
    /// 1. Gaussian-eliminate to find the support (same as `compute_probabilities`)
    /// 2. Pick the first basis state in the support as seed
    /// 3. Apply projector (I + g_i)/2 for each original stabilizer generator
    /// 4. Normalize
    ///
    /// For Pauli g = (-1)^r × ∏_j X_j^{x_j} Z_j^{z_j}:
    ///   g|y⟩ = (-1)^{r + popcount(z & y)} |y ⊕ x_bits⟩
    fn compute_statevector(&self) -> Vec<Complex64> {
        let n = self.n;
        let dim = 1usize << n;
        let stride = self.stride();
        let nw = self.num_words;

        // Step 1: Find a seed state in the support via Gaussian elimination
        // (reuses the same diagonal-constraint logic as compute_probabilities)
        let seed = self.find_support_seed();

        // Step 2: Initialize from seed
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[seed] = Complex64::new(1.0, 0.0);

        // Step 3: Apply (I + g_i)/2 for each ORIGINAL stabilizer generator.
        // Projectors commute (stabilizer generators commute) so order is irrelevant.
        //
        // AG convention: g = (-1)^r × i^m × ∏_j X_j^{x_j} Z_j^{z_j}
        // where m = popcount(x_bits & z_bits) counts the implicit i-factor from
        // Y-type qubits (Y = iXZ, so each x=1,z=1 qubit contributes factor i).
        //
        // Action: g|y⟩ = (-1)^{r + dot(z,y)} × i^m × |y ⊕ x_bits⟩

        let powers_of_i = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];

        let mut visited_gen = vec![0u32; dim];
        let mut current_gen = 0u32;
        for i in 0..n {
            let row = i + n;
            let base = row * stride;

            let mut x_bits = 0usize;
            let mut z_bits = 0usize;
            for w in 0..nw {
                let shift = w * 64;
                if shift < usize::BITS as usize {
                    x_bits |= (self.xz[base + w] as usize) << shift;
                    z_bits |= (self.xz[base + nw + w] as usize) << shift;
                }
            }
            let r = self.phase[row];

            let m = (x_bits & z_bits).count_ones() as usize;
            let i_factor = powers_of_i[m & 3];
            let base_sign = if r { -1.0 } else { 1.0 };

            if x_bits == 0 {
                for (y, s) in state.iter_mut().enumerate() {
                    let dot_parity = (z_bits & y).count_ones() & 1;
                    let phase_val = if dot_parity == 0 {
                        base_sign
                    } else {
                        -base_sign
                    };
                    if phase_val < 0.0 {
                        *s = Complex64::new(0.0, 0.0);
                    }
                }
            } else {
                current_gen += 1;
                for y in 0..dim {
                    if visited_gen[y] == current_gen {
                        continue;
                    }
                    let partner = y ^ x_bits;
                    visited_gen[partner] = current_gen;

                    let a = state[y];
                    let b = state[partner];

                    let dot_y = (z_bits & y).count_ones() & 1;
                    let real_y = if dot_y == 0 { base_sign } else { -base_sign };
                    let gy_phase = i_factor * real_y;

                    let dot_p = (z_bits & partner).count_ones() & 1;
                    let real_p = if dot_p == 0 { base_sign } else { -base_sign };
                    let gp_phase = i_factor * real_p;

                    state[y] = (a + b * gp_phase) * 0.5;
                    state[partner] = (b + a * gy_phase) * 0.5;
                }
            }
        }

        // Step 4: Normalize
        let norm_sq: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > NORM_CLAMP_MIN {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for amp in &mut state {
                *amp *= inv_norm;
            }
        }

        state
    }

    /// Gaussian-eliminate the stabilizer X-part to separate diagonal (Z-only)
    /// from non-diagonal generators.
    ///
    /// Returns (stab_x, stab_z, stab_phase, diag_indices, num_pivots).
    /// stab_x and stab_z are flat arrays with stride `nw` (row i at offset `i * nw`).
    #[allow(clippy::type_complexity)]
    fn gauss_eliminate_x(&self) -> (Vec<u64>, Vec<u64>, Vec<bool>, Vec<usize>, usize) {
        let n = self.n;
        let stride = self.stride();
        let nw = self.num_words;

        let mut stab_x = vec![0u64; n * nw];
        let mut stab_z = vec![0u64; n * nw];
        let mut stab_phase = vec![false; n];

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let src = (i + n) * stride;
            let dst = i * nw;
            stab_x[dst..dst + nw].copy_from_slice(&self.xz[src..src + nw]);
            stab_z[dst..dst + nw].copy_from_slice(&self.xz[src + nw..src + nw + nw]);
            stab_phase[i] = self.phase[i + n];
        }

        let mut remaining: Vec<usize> = (0..n).collect();

        for col in 0..n {
            let w = col / 64;
            let b = col % 64;
            let mut pivot_idx = None;
            for (ri, &row) in remaining.iter().enumerate() {
                if (stab_x[row * nw + w] >> b) & 1 == 1 {
                    pivot_idx = Some(ri);
                    break;
                }
            }

            if let Some(ri) = pivot_idx {
                let pr = remaining.swap_remove(ri);
                let pr_off = pr * nw;

                for row in 0..n {
                    if row == pr {
                        continue;
                    }
                    let row_off = row * nw;
                    if (stab_x[row_off + w] >> b) & 1 == 1 {
                        let initial_sum = if stab_phase[pr] { 2u64 } else { 0 }
                            + if stab_phase[row] { 2u64 } else { 0 };
                        // SAFETY: row != pr, so [row_off..row_off+nw] and
                        // [pr_off..pr_off+nw] are non-overlapping regions.
                        let (dst_x, dst_z, src_x, src_z) = unsafe {
                            let xp = stab_x.as_mut_ptr();
                            let zp = stab_z.as_mut_ptr();
                            (
                                std::slice::from_raw_parts_mut(xp.add(row_off), nw),
                                std::slice::from_raw_parts_mut(zp.add(row_off), nw),
                                std::slice::from_raw_parts(xp.add(pr_off) as *const u64, nw),
                                std::slice::from_raw_parts(zp.add(pr_off) as *const u64, nw),
                            )
                        };
                        let sum = rowmul_words(dst_x, dst_z, src_x, src_z, initial_sum);
                        stab_phase[row] = (sum & 3) >= 2;
                    }
                }
            }
        }

        let k = n - remaining.len();
        let diag = remaining;

        (stab_x, stab_z, stab_phase, diag, k)
    }

    fn solve_diagonal_seed(
        stab_z: &[u64],
        stab_phase: &[bool],
        diag: &[usize],
        nw: usize,
        n: usize,
    ) -> usize {
        let d = diag.len();
        if d == 0 {
            return 0;
        }

        let mut z_rows: Vec<u64> = Vec::with_capacity(d * nw);
        let mut phases: Vec<bool> = Vec::with_capacity(d);
        for &di in diag {
            z_rows.extend_from_slice(&stab_z[di * nw..(di + 1) * nw]);
            phases.push(stab_phase[di]);
        }

        let mut pivot_col = vec![usize::MAX; d];
        let mut available_cols: Vec<usize> = (0..n).collect();

        for row in 0..d {
            let row_off = row * nw;
            let mut found = None;
            for (ci, &col) in available_cols.iter().enumerate() {
                if (z_rows[row_off + col / 64] >> (col % 64)) & 1 == 1 {
                    found = Some(ci);
                    break;
                }
            }

            if let Some(ci) = found {
                let col = available_cols.swap_remove(ci);
                pivot_col[row] = col;
                let w = col / 64;
                let b = col % 64;

                let pivot_z: SmallVec<[u64; 16]> =
                    SmallVec::from_slice(&z_rows[row_off..row_off + nw]);
                let pivot_phase = phases[row];

                #[allow(clippy::needless_range_loop)]
                for other in 0..d {
                    if other == row {
                        continue;
                    }
                    let other_off = other * nw;
                    if (z_rows[other_off + w] >> b) & 1 == 1 {
                        // SAFETY: other_off..other_off+nw and pivot_z are non-overlapping
                        // valid regions of nw u64s. pivot_z was cloned from z_rows at
                        // row_off (row != other), so the regions do not alias.
                        unsafe {
                            xor_words(z_rows.as_mut_ptr().add(other_off), pivot_z.as_ptr(), nw);
                        }
                        phases[other] ^= pivot_phase;
                    }
                }
            }
        }

        let mut seed = 0usize;
        for row in 0..d {
            if pivot_col[row] != usize::MAX && phases[row] {
                seed |= 1 << pivot_col[row];
            }
        }
        seed
    }

    fn find_support_seed(&self) -> usize {
        let nw = self.num_words;
        let (_stab_x, stab_z, stab_phase, diag, _k) = self.gauss_eliminate_x();
        Self::solve_diagonal_seed(&stab_z, &stab_phase, &diag, nw, self.n)
    }

    /// Coset-based probability extraction.
    ///
    /// After Gaussian elimination, the support is a coset of size 2^k defined
    /// by the X-parts of the k non-diagonal generators. Uses GF(2) solve to
    /// find a seed state, then Gray code enumerates all 2^k coset members.
    /// O(n^3/64) for Gaussian elimination + O(2^n) for zeroing + O(2^k) for
    /// coset enumeration (vs old O(2^n × d × n/64) brute-force).
    pub(super) fn compute_probabilities(&self) -> Vec<f64> {
        let n = self.n;
        let dim = 1usize << n;
        let nw = self.num_words;
        let (stab_x, stab_z, stab_phase, diag, k) = self.gauss_eliminate_x();

        let amplitude_sq = 1.0 / (1u64 << k) as f64;

        let seed = Self::solve_diagonal_seed(&stab_z, &stab_phase, &diag, nw, n);

        if k == n {
            return vec![amplitude_sq; dim];
        }

        let mut non_diag_set = vec![true; n];
        for &di in &diag {
            non_diag_set[di] = false;
        }

        let coset_gens: Vec<usize> = (0..n)
            .filter(|&i| non_diag_set[i])
            .map(|i| {
                let mut x = 0usize;
                #[allow(clippy::needless_range_loop)]
                for w in 0..nw {
                    let shift = w * 64;
                    if shift < usize::BITS as usize {
                        x |= (stab_x[i * nw + w] as usize) << shift;
                    }
                }
                x
            })
            .collect();

        debug_assert_eq!(coset_gens.len(), k);

        let mut probs = vec![0.0f64; dim];

        let coset_size = 1usize << k;
        let mut current = seed;
        probs[current] = amplitude_sq;

        for i in 1..coset_size {
            let bit = i.trailing_zeros() as usize;
            current ^= coset_gens[bit];
            probs[current] = amplitude_sq;
        }

        probs
    }
}

impl Backend for StabilizerBackend {
    fn name(&self) -> &'static str {
        "stabilizer"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        let n = num_qubits;
        let nw = n.div_ceil(64);
        let total_rows = 2 * n + 1;
        let stride = 2 * nw;

        self.n = n;
        self.num_words = nw;

        self.xz = vec![0u64; total_rows * stride];
        self.phase = vec![false; total_rows];

        for i in 0..n {
            let word = i / 64;
            let bit = i % 64;
            self.xz[i * stride + word] |= 1u64 << bit;
            self.xz[(i + n) * stride + nw + word] |= 1u64 << bit;
        }

        self.qubit_active = (0..n).map(|q| vec![q as u32, (n + q) as u32]).collect();
        self.total_weight = 2 * n;
        self.sgi_max_active = 2;

        let want_lazy = self.lazy_destab;
        self.lazy_destab = false;
        self.gate_row_start = 0;

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        if want_lazy {
            self.enable_lazy_destab();
        }
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        if self.lazy_destab
            && matches!(
                instruction,
                Instruction::Gate { .. } | Instruction::Conditional { .. }
            )
        {
            // Lazy destabilizer mode is optimized for bulk apply paths.
            // If callers drive the backend instruction-by-instruction, switch
            // back to an eager tableau before the first gate to preserve the
            // standard `apply` semantics.
            self.ensure_destabilizers();
        }
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit)?;
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
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit)
    }

    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        let nw = self.num_words;
        if nw < MIN_WORDS_FOR_BATCH {
            for instruction in instructions {
                self.apply(instruction)?;
            }
            return Ok(());
        }

        if self.sgi_enabled() {
            return self.apply_instructions_sgi(instructions);
        }

        self.apply_instructions_word_batch(instructions)
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.n >= usize::BITS as usize {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probability extraction for {} qubits (exceeds addressable memory)",
                    self.n
                ),
            });
        }
        let dim = 1usize << self.n;
        let mut check = Vec::<f64>::new();
        if check.try_reserve_exact(dim).is_err() {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probability extraction for {} qubits ({} bytes required)",
                    self.n,
                    dim * std::mem::size_of::<f64>()
                ),
            });
        }
        drop(check);
        Ok(self.compute_probabilities())
    }

    fn num_qubits(&self) -> usize {
        self.n
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        self.export_statevector()
    }
}

pub struct FilteredStabilizerBackend {
    num_qubits: usize,
    num_classical_bits: usize,
    clusters: Vec<Option<ClusterState>>,
    qubit_to_cluster: Vec<usize>,
    classical_bits: Vec<bool>,
    seed: u64,
}

struct ClusterState {
    backend: StabilizerBackend,
    qubits: Vec<usize>,
    global_to_local: Vec<usize>,
    local_classical: Vec<usize>,
}

impl FilteredStabilizerBackend {
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            num_classical_bits: 0,
            clusters: Vec::new(),
            qubit_to_cluster: Vec::new(),
            classical_bits: Vec::new(),
            seed,
        }
    }

    pub fn init_with_blocks(
        &mut self,
        num_qubits: usize,
        num_classical_bits: usize,
        blocks: &[Vec<usize>],
    ) -> Result<()> {
        self.num_qubits = num_qubits;
        self.num_classical_bits = num_classical_bits;
        self.qubit_to_cluster = vec![0; num_qubits];
        self.clusters = Vec::with_capacity(blocks.len());

        for (bi, block) in blocks.iter().enumerate() {
            for &q in block {
                self.qubit_to_cluster[q] = bi;
            }

            let mut backend = StabilizerBackend::new(self.seed.wrapping_add(bi as u64));
            backend.init(block.len(), 0)?;

            let mut g2l = vec![0usize; num_qubits];
            for (li, &q) in block.iter().enumerate() {
                g2l[q] = li;
            }

            self.clusters.push(Some(ClusterState {
                backend,
                qubits: block.clone(),
                global_to_local: g2l,
                local_classical: Vec::new(),
            }));
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn merge_clusters(&mut self, ci_a: usize, ci_b: usize) {
        if ci_a == ci_b {
            return;
        }

        let (keep, merge) = if ci_a < ci_b {
            (ci_a, ci_b)
        } else {
            (ci_b, ci_a)
        };
        let merge_state = self.clusters[merge].take().unwrap();

        let keep_state = self.clusters[keep].as_mut().unwrap();

        let old_n = keep_state.qubits.len();
        let merge_n = merge_state.qubits.len();
        let new_n = old_n + merge_n;

        let mut merged_qubits = keep_state.qubits.clone();
        merged_qubits.extend_from_slice(&merge_state.qubits);

        let mut new_backend = StabilizerBackend::new(self.seed.wrapping_add(keep as u64 * 1000));
        new_backend.init(new_n, 0).unwrap();

        copy_tableau_into(&keep_state.backend, &mut new_backend, 0);
        copy_tableau_into(&merge_state.backend, &mut new_backend, old_n);

        let mut merged_classical = keep_state.local_classical.clone();
        merged_classical.extend_from_slice(&merge_state.local_classical);
        new_backend
            .classical_bits
            .resize(merged_classical.len(), false);

        let mut g2l = vec![0usize; self.num_qubits];
        for (li, &q) in merged_qubits.iter().enumerate() {
            g2l[q] = li;
            self.qubit_to_cluster[q] = keep;
        }

        *keep_state = ClusterState {
            backend: new_backend,
            qubits: merged_qubits,
            global_to_local: g2l,
            local_classical: merged_classical,
        };
    }

    fn apply_gate_to_cluster(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let ci = self.qubit_to_cluster[targets[0]];

        if targets.len() > 1 {
            for &t in &targets[1..] {
                let other_ci = self.qubit_to_cluster[t];
                if other_ci != ci {
                    self.merge_clusters(ci, other_ci);
                    return self.apply_gate_to_cluster(gate, targets);
                }
            }
        }

        let cluster = self.clusters[ci].as_mut().unwrap();
        let local_targets: SmallVec<[usize; 4]> = targets
            .iter()
            .map(|&t| cluster.global_to_local[t])
            .collect();

        let local_inst = Instruction::Gate {
            gate: gate.clone(),
            targets: local_targets,
        };
        cluster.backend.apply(&local_inst)
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let ci = self.qubit_to_cluster[qubit];
        let cluster = self.clusters[ci].as_mut().unwrap();
        let local_q = cluster.global_to_local[qubit];

        let local_cbit = cluster
            .local_classical
            .iter()
            .position(|&cb| cb == classical_bit)
            .unwrap_or_else(|| {
                let idx = cluster.local_classical.len();
                cluster.local_classical.push(classical_bit);
                if idx >= cluster.backend.classical_bits.len() {
                    cluster.backend.classical_bits.resize(idx + 1, false);
                }
                idx
            });

        cluster.backend.apply_measure(local_q, local_cbit);
        self.classical_bits[classical_bit] = cluster.backend.classical_bits[local_cbit];
    }

    fn apply_reset_cluster(&mut self, qubit: usize) -> Result<()> {
        let ci = self.qubit_to_cluster[qubit];
        let cluster = self.clusters[ci].as_mut().unwrap();
        let local_q = cluster.global_to_local[qubit];
        cluster.backend.apply_reset(local_q)
    }
}

fn copy_tableau_into(src: &StabilizerBackend, dst: &mut StabilizerBackend, qubit_offset: usize) {
    let src_n = src.n;
    let src_nw = src.num_words;
    let src_stride = 2 * src_nw;
    let dst_n = dst.n;
    let dst_nw = dst.num_words;
    let dst_stride = 2 * dst_nw;

    for i in 0..src_n {
        let src_row = i;
        let dst_row = qubit_offset + i;

        let old_word = (qubit_offset + i) / 64;
        let old_bit = (qubit_offset + i) % 64;
        dst.xz[dst_row * dst_stride + old_word] &= !(1u64 << old_bit);

        let q_word_offset = qubit_offset / 64;
        let q_bit_offset = qubit_offset % 64;
        if q_bit_offset == 0 {
            for w in 0..src_nw {
                dst.xz[dst_row * dst_stride + q_word_offset + w] = src.xz[src_row * src_stride + w];
            }
            for w in 0..src_nw {
                dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w] =
                    src.xz[src_row * src_stride + src_nw + w];
            }
        } else {
            for w in 0..src_nw {
                let val = src.xz[src_row * src_stride + w];
                dst.xz[dst_row * dst_stride + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_row * dst_stride + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
            for w in 0..src_nw {
                let val = src.xz[src_row * src_stride + src_nw + w];
                dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_row * dst_stride + dst_nw + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
        }
        dst.phase[dst_row] = src.phase[src_row];

        let src_stab = src_n + i;
        let dst_stab = dst_n + qubit_offset + i;

        let old_word_s = (qubit_offset + i) / 64;
        let old_bit_s = (qubit_offset + i) % 64;
        dst.xz[dst_stab * dst_stride + dst_nw + old_word_s] &= !(1u64 << old_bit_s);

        if q_bit_offset == 0 {
            for w in 0..src_nw {
                dst.xz[dst_stab * dst_stride + q_word_offset + w] =
                    src.xz[src_stab * src_stride + w];
            }
            for w in 0..src_nw {
                dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w] =
                    src.xz[src_stab * src_stride + src_nw + w];
            }
        } else {
            for w in 0..src_nw {
                let val = src.xz[src_stab * src_stride + w];
                dst.xz[dst_stab * dst_stride + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_stab * dst_stride + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
            for w in 0..src_nw {
                let val = src.xz[src_stab * src_stride + src_nw + w];
                dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w] |= val << q_bit_offset;
                if q_word_offset + w + 1 < dst_nw {
                    dst.xz[dst_stab * dst_stride + dst_nw + q_word_offset + w + 1] |=
                        val >> (64 - q_bit_offset);
                }
            }
        }
        dst.phase[dst_stab] = src.phase[src_stab];
    }
}

impl Backend for FilteredStabilizerBackend {
    fn name(&self) -> &'static str {
        "FilteredStabilizer"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.num_classical_bits = num_classical_bits;
        self.qubit_to_cluster = vec![0; num_qubits];
        self.clusters.clear();

        for i in 0..num_qubits {
            self.qubit_to_cluster[i] = i;
            let mut backend = StabilizerBackend::new(self.seed.wrapping_add(i as u64));
            backend.init(1, 0)?;
            let mut g2l = vec![0usize; num_qubits];
            g2l[i] = 0;
            self.clusters.push(Some(ClusterState {
                backend,
                qubits: vec![i],
                global_to_local: g2l,
                local_classical: Vec::new(),
            }));
        }

        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => {
                self.apply_gate_to_cluster(gate, targets)?;
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(*qubit, *classical_bit);
            }
            Instruction::Reset { qubit } => {
                self.apply_reset_cluster(*qubit)?;
            }
            Instruction::Barrier { .. } => {}
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.apply_gate_to_cluster(gate, targets)?;
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset_cluster(qubit)
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.num_qubits >= crate::backend::MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!("probability extraction for {} qubits", self.num_qubits),
            });
        }

        let mut blocks: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
        for cluster in self.clusters.iter().flatten() {
            let probs = cluster.backend.compute_probabilities();
            blocks.push((probs, cluster.qubits.clone()));
        }

        if blocks.len() == 1 && blocks[0].1.iter().enumerate().all(|(i, &q)| i == q) {
            return Ok(blocks.into_iter().next().unwrap().0);
        }

        Ok(crate::sim::merge_probabilities(&blocks, self.num_qubits))
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        false
    }
}
