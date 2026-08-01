//! Homological noisy sampling for Clifford circuits: precomputed syndrome
//! classes give O(1) noise work per shot, plus exact analytic noisy
//! marginals.

use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::sim::ShotsResult;
use crate::sim::compiled::batch_propagate_backward;
use crate::sim::compiled::{PackedShots, ShotAccumulator, default_chunk_size, xor_words};
use crate::sim::noise::NoiseModel;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Dense binary matrix over GF(2) stored as packed u64 words per row.
/// Row-major: row i is stored in words[i * row_words .. (i+1) * row_words].
struct F2DenseMatrix {
    #[cfg(test)]
    num_rows: usize,
    #[cfg(test)]
    num_cols: usize,
    row_words: usize,
    data: Vec<u64>,
}

impl F2DenseMatrix {
    fn new(num_rows: usize, num_cols: usize) -> Self {
        let row_words = num_cols.div_ceil(64);
        Self {
            #[cfg(test)]
            num_rows,
            #[cfg(test)]
            num_cols,
            row_words,
            data: vec![0u64; num_rows * row_words],
        }
    }

    #[inline(always)]
    fn set(&mut self, row: usize, col: usize) {
        self.data[row * self.row_words + col / 64] |= 1u64 << (col % 64);
    }

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> bool {
        (self.data[row * self.row_words + col / 64] >> (col % 64)) & 1 != 0
    }

    #[cfg(test)]
    fn row(&self, row: usize) -> &[u64] {
        let start = row * self.row_words;
        &self.data[start..start + self.row_words]
    }

    #[cfg(test)]
    fn xor_row(&mut self, dst_row: usize, src_row: usize) {
        let rw = self.row_words;
        let (dst_start, src_start) = (dst_row * rw, src_row * rw);
        if dst_start < src_start {
            let (left, right) = self.data.split_at_mut(src_start);
            for w in 0..rw {
                left[dst_start + w] ^= right[w];
            }
        } else {
            let (left, right) = self.data.split_at_mut(dst_start);
            for w in 0..rw {
                right[w] ^= left[src_start + w];
            }
        }
    }

    #[cfg(test)]
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let rw = self.row_words;
        let (a_start, b_start) = (a * rw, b * rw);
        for w in 0..rw {
            self.data.swap(a_start + w, b_start + w);
        }
    }
}

/// Compute the kernel (null space) of a binary matrix over GF(2).
///
/// Given M ∈ F₂^{m×n}, returns a basis for ker(M) = {x ∈ F₂^n : Mx = 0}.
/// Uses row reduction on the augmented matrix [M | I_n]^T approach:
/// transpose M, row-reduce M^T, read off kernel vectors.
///
/// Returns: Vec of kernel basis vectors, each as a Vec<u64> packed bitvector of length n.
#[cfg(test)]
fn gf2_kernel(matrix: &F2DenseMatrix) -> Vec<Vec<u64>> {
    let m = matrix.num_rows;
    let n = matrix.num_cols;
    let n_words = n.div_ceil(64);

    let aug_cols = m + n;
    let mut aug = F2DenseMatrix::new(n, aug_cols);

    for r in 0..m {
        for c in 0..n {
            if matrix.get(r, c) {
                aug.set(c, r);
            }
        }
    }
    for i in 0..n {
        aug.set(i, m + i);
    }

    let mut pivot_row = 0;
    for col in 0..m {
        let mut found = None;
        for r in pivot_row..n {
            if aug.get(r, col) {
                found = Some(r);
                break;
            }
        }
        let Some(pr) = found else { continue };

        aug.swap_rows(pivot_row, pr);

        for r in 0..n {
            if r != pivot_row && aug.get(r, col) {
                aug.xor_row(r, pivot_row);
            }
        }
        pivot_row += 1;
    }

    let mut kernel = Vec::new();
    let m_words = m.div_ceil(64);
    for r in 0..n {
        let row = aug.row(r);
        let mt_zero = row[..m_words].iter().enumerate().all(|(w, &val)| {
            if w == m_words - 1 && !m.is_multiple_of(64) {
                val & ((1u64 << (m % 64)) - 1) == 0
            } else {
                val == 0
            }
        });
        if mt_zero {
            let mut kv = vec![0u64; n_words];
            for c in 0..n {
                if aug.get(r, m + c) {
                    kv[c / 64] |= 1u64 << (c % 64);
                }
            }
            kernel.push(kv);
        }
    }

    kernel
}

/// Error chain complex for a Clifford circuit with noise.
///
/// Represents the chain complex C₂ →∂₂→ C₁ →∂₁→ C₀ where:
/// - C₀ = F₂^m (measurement/detector space)
/// - C₁ = F₂^p (error location space)
/// - C₂ = F₂^s (stabilizer space)
/// - ∂₁ = E (the m×p error propagation matrix)
pub struct ErrorChainComplex {
    /// E-matrix: m × p binary matrix. E[d][e] = 1 if error e flips measurement d.
    e_matrix: F2DenseMatrix,
    /// Error probabilities: p_total[e] = px + py + pz for error location e.
    error_probs: Vec<f64>,
    num_measurements: usize,
    num_errors: usize,
    /// dim(im(∂₂) ∩ ker(∂₁)): stabilizer generators undetectable by measurements
    boundary_dim: usize,
    /// dim(H₁) = dim(ker(∂₁)/im(∂₂)): independent logical error classes
    homology_dim: usize,
}

/// Precomputed sampler for O(r + 1) per-shot noisy measurement sampling.
///
/// Combines a compiled sampler (quantum randomness, O(r) per shot) with
/// precomputed syndrome class probabilities (noise randomness, O(1) per shot).
/// The syndrome classes are elements of im(E) ⊆ F₂^m where E is the
/// error-to-measurement propagation matrix.
pub struct HomologicalSampler {
    compiled: crate::sim::compiled::CompiledSampler,
    /// Syndrome rank = dim(im(E))
    syndrome_rank: usize,
    /// 2^r class probabilities (for diagnostics)
    #[allow(dead_code)]
    class_probs: Vec<f64>,
    /// 2^r cumulative probabilities for sampling
    class_cdf: Vec<f64>,
    /// 2^r detection signatures: for class c, which measurements are flipped.
    /// Stored as packed u64 vectors, each of length ceil(m/64).
    class_detections: Vec<Vec<u64>>,
    /// dim(im(∂₂) ∩ ker(∂₁)): undetectable stabilizer generators
    boundary_dim: usize,
    /// dim(H₁ = ker(∂₁)/im(∂₂)): independent logical error classes
    homology_dim: usize,
    rng: ChaCha8Rng,
}

impl ErrorChainComplex {
    /// Build the error chain complex from a Clifford circuit and noise model.
    ///
    /// Uses backward Pauli propagation (same as the compiled noisy sampler)
    /// to determine which measurements are sensitive to each error location.
    pub fn build(circuit: &Circuit, noise: &NoiseModel, _seed: u64) -> Result<Self> {
        let m = circuit
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .count();
        if m == 0 {
            return Ok(Self {
                e_matrix: F2DenseMatrix::new(0, 0),
                error_probs: Vec::new(),
                num_measurements: 0,
                num_errors: 0,
                boundary_dim: circuit.num_qubits,
                homology_dim: 0,
            });
        }

        let m_words = m.div_ceil(64);
        let n = circuit.num_qubits;

        let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
        let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
        let mut sign_packed = vec![0u64; m_words];

        let mut meas_idx = m;
        for instr in circuit.instructions.iter().rev() {
            if let Instruction::Measure { qubit, .. } = instr {
                meas_idx -= 1;
                let word = meas_idx / 64;
                let bit = meas_idx % 64;
                z_packed[*qubit][word] |= 1u64 << bit;
            }
        }

        let mut error_probs = Vec::new();
        let mut e_cols: Vec<Vec<u64>> = Vec::new();

        for (instr_idx, instr) in circuit.instructions.iter().enumerate().rev() {
            match instr {
                Instruction::Gate { gate, targets } => {
                    let noise_events = &noise.after_gate[instr_idx];
                    for event in noise_events {
                        let (px, py, pz) = event.pauli_probs();
                        let q = event.qubit();
                        let p_total = px + py + pz;
                        if p_total < 1e-15 {
                            continue;
                        }

                        let x_sens = &z_packed[q];
                        let z_sens = &x_packed[q];

                        if px > 1e-15 && x_sens.iter().any(|&w| w != 0) {
                            error_probs.push(px);
                            e_cols.push(x_sens.clone());
                        }

                        if pz > 1e-15 && z_sens.iter().any(|&w| w != 0) {
                            error_probs.push(pz);
                            e_cols.push(z_sens.clone());
                        }

                        if py > 1e-15 {
                            let mut y_sens = vec![0u64; m_words];
                            for w in 0..m_words {
                                y_sens[w] = x_sens[w] ^ z_sens[w];
                            }
                            if y_sens.iter().any(|&w| w != 0) {
                                error_probs.push(py);
                                e_cols.push(y_sens);
                            }
                        }
                    }

                    batch_propagate_backward(
                        &mut x_packed,
                        &mut z_packed,
                        &mut sign_packed,
                        gate,
                        targets.as_slice(),
                        m_words,
                    );
                }
                Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Barrier { .. } => {}
                Instruction::Conditional { gate, targets, .. } => {
                    batch_propagate_backward(
                        &mut x_packed,
                        &mut z_packed,
                        &mut sign_packed,
                        gate,
                        targets.as_slice(),
                        m_words,
                    );
                }
            }
        }

        let p = error_probs.len();
        let mut e_matrix = F2DenseMatrix::new(m, p);

        for (col, col_data) in e_cols.iter().enumerate() {
            for (w, &word) in col_data.iter().enumerate() {
                if word == 0 {
                    continue;
                }
                let base = w * 64;
                let mut bits = word;
                while bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    let row = base + bit;
                    if row < m {
                        e_matrix.set(row, col);
                    }
                    bits &= bits - 1;
                }
            }
        }

        let (boundary_dim, homology_dim) = Self::compute_boundary_space(circuit, n);

        Ok(Self {
            e_matrix,
            error_probs,
            num_measurements: m,
            num_errors: p,
            boundary_dim,
            homology_dim,
        })
    }

    /// Forward-propagate stabilizer generators and compute ∂₂ boundary space.
    ///
    /// Returns (boundary_dim, homology_dim) where:
    /// - boundary_dim = dim(im(∂₂) ∩ ker(∂₁)) = stabilizers undetectable by measurements
    /// - homology_dim = dim(H₁) = independent logical error classes
    ///
    /// Algorithm: forward-propagate Z_0,...,Z_{n-1} through the circuit to get
    /// output stabilizer generators. Build X-projection onto measured qubits.
    /// rank(X_proj) counts stabilizers with detectable X components;
    /// H₁ = ker(σ) / (S ∩ ker(σ)) has dim = n - num_measured + rank(X_proj).
    fn compute_boundary_space(circuit: &Circuit, n: usize) -> (usize, usize) {
        if n == 0 {
            return (0, 0);
        }

        let n_words = n.div_ceil(64);
        let mut stab_x: Vec<Vec<u64>> = vec![vec![0u64; n_words]; n];
        let mut stab_z: Vec<Vec<u64>> = vec![vec![0u64; n_words]; n];
        let mut stab_sign = vec![0u64; n_words];

        for i in 0..n {
            stab_z[i][i / 64] |= 1u64 << (i % 64);
        }

        for instr in circuit.instructions.iter() {
            match instr {
                Instruction::Gate { gate, targets } => {
                    batch_propagate_backward(
                        &mut stab_x,
                        &mut stab_z,
                        &mut stab_sign,
                        gate,
                        targets.as_slice(),
                        n_words,
                    );
                }
                Instruction::Conditional { gate, targets, .. } => {
                    batch_propagate_backward(
                        &mut stab_x,
                        &mut stab_z,
                        &mut stab_sign,
                        gate,
                        targets.as_slice(),
                        n_words,
                    );
                }
                _ => {}
            }
        }

        let mut measured = vec![false; n];
        for instr in &circuit.instructions {
            if let Instruction::Measure { qubit, .. } = instr {
                measured[*qubit] = true;
            }
        }
        let num_measured = measured.iter().filter(|&&b| b).count();
        let measured_indices: Vec<usize> = (0..n).filter(|&q| measured[q]).collect();

        if num_measured == 0 {
            return (n, 0);
        }

        let proj_words = num_measured.div_ceil(64);
        let mut proj = vec![0u64; n * proj_words];

        for stab_idx in 0..n {
            for (proj_col, &q) in measured_indices.iter().enumerate() {
                let x_bit = (stab_x[q][stab_idx / 64] >> (stab_idx % 64)) & 1;
                if x_bit != 0 {
                    proj[stab_idx * proj_words + proj_col / 64] |= 1u64 << (proj_col % 64);
                }
            }
        }

        let mut rank = 0;
        let mut pivot_row = 0;
        for col in 0..num_measured {
            let mut found = None;
            for r in pivot_row..n {
                if (proj[r * proj_words + col / 64] >> (col % 64)) & 1 != 0 {
                    found = Some(r);
                    break;
                }
            }
            let Some(pr) = found else { continue };

            if pr != pivot_row {
                for w in 0..proj_words {
                    proj.swap(pivot_row * proj_words + w, pr * proj_words + w);
                }
            }

            for r in 0..n {
                if r != pivot_row && (proj[r * proj_words + col / 64] >> (col % 64)) & 1 != 0 {
                    for w in 0..proj_words {
                        proj[r * proj_words + w] ^= proj[pivot_row * proj_words + w];
                    }
                }
            }

            pivot_row += 1;
            rank += 1;
        }

        let boundary_dim = n - rank;
        let homology_dim = n - num_measured + rank;
        (boundary_dim, homology_dim)
    }

    /// dim(im(∂₂) ∩ ker(∂₁)): stabilizer generators undetectable by the
    /// measurements.
    pub fn boundary_dim(&self) -> usize {
        self.boundary_dim
    }

    /// dim(H₁): independent logical error classes.
    pub fn homology_dim(&self) -> usize {
        self.homology_dim
    }

    /// Compute exact noisy marginals analytically. No sampling, no rank limit.
    ///
    /// For each measurement j, the noisy probability is:
    ///   p_j^noisy = p_j + (1 - 2·p_j) · (1 - f_j) / 2
    /// where f_j = Π_{e: E(j,e)=1} (1 - 2·p_e) is the flip attenuation factor
    /// and p_j is the noiseless marginal (0, 0.5, or 1).
    ///
    /// Cost: O(nnz(E)). Works for any qubit count.
    pub fn noisy_marginals(&self, noiseless_marginals: &[f64]) -> Vec<f64> {
        let m = self.num_measurements;
        let p = self.num_errors;
        if m == 0 || p == 0 {
            return noiseless_marginals.to_vec();
        }

        let mut flip_factor = vec![1.0f64; m];
        let rw = self.e_matrix.row_words;

        for e in 0..p {
            let factor = 1.0 - 2.0 * self.error_probs[e];
            if (factor - 1.0).abs() < 1e-15 {
                continue;
            }

            let col_word = e / 64;
            let col_bit = 1u64 << (e % 64);

            for (j, ff) in flip_factor.iter_mut().enumerate() {
                if self.e_matrix.data[j * rw + col_word] & col_bit != 0 {
                    *ff *= factor;
                }
            }
        }

        let mut result = Vec::with_capacity(m);
        for j in 0..m {
            let p_j = noiseless_marginals[j];
            let p_flip = (1.0 - flip_factor[j]) / 2.0;
            result.push(p_j + (1.0 - 2.0 * p_j) * p_flip);
        }
        result
    }
}

const MAX_SYNDROME_RANK: usize = 20;

impl HomologicalSampler {
    /// Build a sampler from a circuit and noise model.
    ///
    /// Computes the E-matrix (error-to-measurement propagation), finds a basis
    /// for im(E), and precomputes 2^r syndrome class probabilities where
    /// r = rank(E), rejecting circuits with r above `MAX_SYNDROME_RANK`.
    /// Also builds a compiled sampler for quantum randomness.
    ///
    /// Total per-shot cost: O(r_quantum + 1) where r_quantum is the stabilizer
    /// rank (number of random measurements), versus O(p) for brute-force
    /// where p is the number of error locations.
    pub fn compile(circuit: &Circuit, noise: &NoiseModel, seed: u64) -> Result<Self> {
        let ecc = ErrorChainComplex::build(circuit, noise, seed)?;
        let m = ecc.num_measurements;
        let p = ecc.num_errors;
        let compiled = crate::sim::compiled::compile_measurements(circuit, seed)?;

        if m == 0 || p == 0 {
            return Ok(Self {
                compiled,
                syndrome_rank: 0,
                class_probs: vec![1.0],
                class_cdf: vec![1.0],
                class_detections: vec![vec![0u64; m.div_ceil(64)]],
                boundary_dim: ecc.boundary_dim,
                homology_dim: ecc.homology_dim,
                rng: ChaCha8Rng::seed_from_u64(seed),
            });
        }

        let m_words = m.div_ceil(64);

        let mut work = ecc.e_matrix.data.clone();
        let rw = ecc.e_matrix.row_words;
        let mut pivot_cols = Vec::new();
        let mut pivot_row = 0;

        for col in 0..p {
            let mut found = None;
            for r in pivot_row..m {
                if (work[r * rw + col / 64] >> (col % 64)) & 1 != 0 {
                    found = Some(r);
                    break;
                }
            }
            let Some(pr) = found else { continue };

            if pr != pivot_row {
                for w in 0..rw {
                    work.swap(pivot_row * rw + w, pr * rw + w);
                }
            }

            for r in 0..m {
                if r != pivot_row && (work[r * rw + col / 64] >> (col % 64)) & 1 != 0 {
                    for w in 0..rw {
                        work[r * rw + w] ^= work[pivot_row * rw + w];
                    }
                }
            }

            pivot_cols.push(col);
            pivot_row += 1;
        }

        let r = pivot_cols.len();
        if r > MAX_SYNDROME_RANK {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "HomologicalSampler".to_string(),
                reason: format!("syndrome rank {r} too large (max {MAX_SYNDROME_RANK})"),
            });
        }

        // Extract r-bit coordinates from RREF: col j's coordinate at basis i
        // is work[i][j] in the reduced matrix.
        let mut col_coords = vec![0usize; p];
        for (basis_idx, &_pivot_col) in pivot_cols.iter().enumerate() {
            for j in 0..p {
                if (work[basis_idx * rw + j / 64] >> (j % 64)) & 1 != 0 {
                    col_coords[j] |= 1 << basis_idx;
                }
            }
        }

        let num_classes = 1usize << r;
        let mut class_detections = Vec::with_capacity(num_classes);
        for c in 0..num_classes {
            let mut det = vec![0u64; m_words];
            for (basis_idx, &pivot_col) in pivot_cols.iter().enumerate() {
                if (c >> basis_idx) & 1 != 0 {
                    for row in 0..m {
                        if ecc.e_matrix.get(row, pivot_col) {
                            det[row / 64] ^= 1u64 << (row % 64);
                        }
                    }
                }
            }
            class_detections.push(det);
        }

        // F₂^r probability convolution: P[c] = (1-p_j) P[c] + p_j P[c ⊕ coord_j]
        let mut class_probs = vec![0.0_f64; num_classes];
        class_probs[0] = 1.0;

        for (j, &coord) in col_coords.iter().enumerate() {
            let pj = ecc.error_probs[j];
            if pj < 1e-15 {
                continue;
            }
            if coord == 0 {
                continue;
            }
            let mut new_probs = vec![0.0_f64; num_classes];
            for c in 0..num_classes {
                new_probs[c] = (1.0 - pj) * class_probs[c] + pj * class_probs[c ^ coord];
            }
            class_probs = new_probs;
        }

        let mut class_cdf = vec![0.0_f64; num_classes];
        class_cdf[0] = class_probs[0];
        for c in 1..num_classes {
            class_cdf[c] = class_cdf[c - 1] + class_probs[c];
        }
        let total = class_cdf[num_classes - 1];
        if total > 0.0 {
            for v in &mut class_cdf {
                *v /= total;
            }
        }

        Ok(Self {
            compiled,
            syndrome_rank: r,
            class_probs,
            class_cdf,
            class_detections,
            boundary_dim: ecc.boundary_dim,
            homology_dim: ecc.homology_dim,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    /// rank(E): number of independent syndrome classes is `2^rank`.
    pub fn syndrome_rank(&self) -> usize {
        self.syndrome_rank
    }

    /// dim(im(∂₂) ∩ ker(∂₁)): stabilizer generators undetectable by the
    /// measurements.
    pub fn boundary_dim(&self) -> usize {
        self.boundary_dim
    }

    /// dim(H₁): independent logical error classes.
    pub fn homology_dim(&self) -> usize {
        self.homology_dim
    }

    /// Cost: O(r_quantum) for compiled sampler + O(1) for noise class lookup.
    pub fn sample(&mut self) -> Vec<bool> {
        let mut outcome = self.compiled.sample();

        let u: f64 = rand::RngExt::random(&mut self.rng);
        let class = match self
            .class_cdf
            .binary_search_by(|p| p.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i,
            Err(i) => i.min(self.class_cdf.len() - 1),
        };

        let det = &self.class_detections[class];
        for (mi, bit) in outcome.iter_mut().enumerate() {
            let det_bit = (det[mi / 64] >> (mi % 64)) & 1 != 0;
            *bit ^= det_bit;
        }
        outcome
    }

    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        (0..num_shots).map(|_| self.sample()).collect()
    }

    /// Sample `num_shots` shots into a shot-major [`PackedShots`] buffer.
    pub fn sample_packed(&mut self, num_shots: usize) -> PackedShots {
        let m = self.compiled.num_measurements();
        let m_words = m.div_ceil(64);
        if num_shots == 0 || m == 0 {
            return PackedShots::from_shot_major(Vec::new(), num_shots, m);
        }

        let mut accum = Vec::new();
        let mut rand_buf = Vec::new();
        self.compiled
            .sample_bulk_words_shot_major_reuse(&mut accum, &mut rand_buf, num_shots);

        let ref_bits = self.compiled.ref_bits_packed();
        for s in 0..num_shots {
            let base = s * m_words;
            xor_words(&mut accum[base..base + m_words], ref_bits);
        }

        for s in 0..num_shots {
            let u: f64 = rand::RngExt::random(&mut self.rng);
            let class = match self
                .class_cdf
                .binary_search_by(|p| p.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) => i,
                Err(i) => i.min(self.class_cdf.len() - 1),
            };

            let det = &self.class_detections[class];
            let base = s * m_words;
            xor_words(&mut accum[base..base + m_words], det);
        }

        PackedShots::from_shot_major(accum, num_shots, m)
    }

    /// Stream shots into `acc` in default-sized chunks instead of one matrix.
    pub fn sample_chunked<A: ShotAccumulator>(&mut self, total_shots: usize, acc: &mut A) {
        let chunk_size = default_chunk_size(self.compiled.num_measurements());
        crate::sim::compiled::for_each_chunk(total_shots, chunk_size, |batch| {
            let packed = self.sample_packed(batch);
            acc.accumulate(&packed);
        });
    }

    /// Sample `total_shots` shots and return per-measurement `P(bit = 1)`.
    pub fn sample_marginals(&mut self, total_shots: usize) -> Vec<f64> {
        crate::sim::compiled::marginals_from_chunks(self.compiled.num_measurements(), |acc| {
            self.sample_chunked(total_shots, acc)
        })
    }
}

/// Run noisy shot sampling using the homological sampler.
///
/// For Clifford circuits whose syndrome rank is at most `MAX_SYNDROME_RANK`
/// (20), precomputes class probabilities and samples in O(1) per shot.
pub fn run_shots_homological(
    circuit: &Circuit,
    noise: &NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    let sampler = HomologicalSampler::compile(circuit, noise, seed)?;
    run_shots_homological_inner(sampler, circuit, num_shots)
}

pub(crate) fn run_shots_homological_inner(
    mut sampler: HomologicalSampler,
    circuit: &Circuit,
    num_shots: usize,
) -> Result<ShotsResult> {
    let classical_bit_order: Vec<usize> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure { classical_bit, .. } => Some(*classical_bit),
            _ => None,
        })
        .collect();
    let num_classical = circuit.num_classical_bits;

    let raw_shots = sampler.sample_bulk(num_shots);

    let mut shots = Vec::with_capacity(num_shots);
    for raw in &raw_shots {
        let mut out = vec![false; num_classical];
        for (mi, &cbit) in classical_bit_order.iter().enumerate() {
            if cbit < num_classical {
                out[cbit] = raw[mi];
            }
        }
        shots.push(out);
    }

    Ok(ShotsResult::from_shots(shots, circuit.num_classical_bits))
}

/// Compute exact noisy marginals analytically. No sampling, no rank limit.
///
/// Builds the error chain complex and compiled sampler, then computes
/// exact per-measurement noisy probabilities in O(nnz(E)) time.
/// Works for any qubit count, not limited by syndrome rank.
pub fn noisy_marginals_analytical(
    circuit: &Circuit,
    noise: &NoiseModel,
    seed: u64,
) -> Result<Vec<f64>> {
    let ecc = ErrorChainComplex::build(circuit, noise, seed)?;
    let compiled = crate::sim::compiled::compile_measurements(circuit, seed)?;
    let noiseless = compiled.marginal_probabilities();
    let noisy = ecc.noisy_marginals(&noiseless);

    let classical_bit_order: Vec<usize> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure { classical_bit, .. } => Some(*classical_bit),
            _ => None,
        })
        .collect();
    let num_classical = circuit.num_classical_bits;

    let mut result = vec![0.5f64; num_classical];
    for (mi, &cbit) in classical_bit_order.iter().enumerate() {
        if cbit < num_classical && mi < noisy.len() {
            result[cbit] = noisy[mi];
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "homological_tests.rs"]
mod tests;
