use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::sim::compiled::batch_propagate_backward;
use crate::sim::compiled::{default_chunk_size, xor_words, PackedShots, ShotAccumulator};
use crate::sim::noise::NoiseModel;
use crate::sim::ShotsResult;
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
            if w == m_words - 1 && m % 64 != 0 {
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
    /// Number of measurements (detectors)
    num_measurements: usize,
    /// Number of error locations
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
    /// Compiled sampler for quantum randomness (noiseless measurement distribution)
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
    /// Number of measurements
    #[allow(dead_code)]
    num_measurements: usize,
    /// dim(im(∂₂) ∩ ker(∂₁)): undetectable stabilizer generators
    boundary_dim: usize,
    /// dim(H₁ = ker(∂₁)/im(∂₂)): independent logical error classes
    homology_dim: usize,
    /// RNG for noise sampling
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
                    let noise_ops = &noise.after_gate[instr_idx];
                    for op in noise_ops {
                        let q = op.qubit;
                        let p_total = op.px + op.py + op.pz;
                        if p_total < 1e-15 {
                            continue;
                        }

                        let x_sens = &z_packed[q];
                        let z_sens = &x_packed[q];

                        if op.px > 1e-15 && x_sens.iter().any(|&w| w != 0) {
                            error_probs.push(op.px);
                            e_cols.push(x_sens.clone());
                        }

                        if op.pz > 1e-15 && z_sens.iter().any(|&w| w != 0) {
                            error_probs.push(op.pz);
                            e_cols.push(z_sens.clone());
                        }

                        if op.py > 1e-15 {
                            let mut y_sens = vec![0u64; m_words];
                            for w in 0..m_words {
                                y_sens[w] = x_sens[w] ^ z_sens[w];
                            }
                            if y_sens.iter().any(|&w| w != 0) {
                                error_probs.push(op.py);
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
                Instruction::Measure { .. } | Instruction::Barrier { .. } => {}
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

    pub fn boundary_dim(&self) -> usize {
        self.boundary_dim
    }

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

impl HomologicalSampler {
    /// Build a sampler from a circuit and noise model.
    ///
    /// Computes the E-matrix (error-to-measurement propagation), finds a basis
    /// for im(E), and precomputes 2^r syndrome class probabilities where
    /// r = rank(E). Also builds a compiled sampler for quantum randomness.
    ///
    /// Total per-shot cost: O(r_quantum + 1) where r_quantum is the stabilizer
    /// rank (number of random measurements) — versus O(p) for brute-force
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
                num_measurements: m,
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
        if r > 20 {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "HomologicalSampler".to_string(),
                reason: format!("syndrome rank {} too large (max 20)", r),
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
            num_measurements: m,
            boundary_dim: ecc.boundary_dim,
            homology_dim: ecc.homology_dim,
            rng: ChaCha8Rng::seed_from_u64(seed),
        })
    }

    pub fn syndrome_rank(&self) -> usize {
        self.syndrome_rank
    }

    pub fn boundary_dim(&self) -> usize {
        self.boundary_dim
    }

    pub fn homology_dim(&self) -> usize {
        self.homology_dim
    }

    /// Sample a single shot: returns measurement outcomes.
    ///
    /// Cost: O(r_quantum) for compiled sampler + O(1) for noise class lookup.
    pub fn sample(&mut self) -> Vec<bool> {
        let mut outcome = self.compiled.sample();

        let u: f64 = rand::Rng::gen(&mut self.rng);
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

    /// Sample multiple shots.
    pub fn sample_bulk(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        (0..num_shots).map(|_| self.sample()).collect()
    }

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
            let u: f64 = rand::Rng::gen(&mut self.rng);
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

    pub fn sample_chunked<A: ShotAccumulator>(&mut self, total_shots: usize, acc: &mut A) {
        let chunk_size = default_chunk_size(self.compiled.num_measurements());
        let mut remaining = total_shots;
        while remaining > 0 {
            let batch = remaining.min(chunk_size);
            let packed = self.sample_packed(batch);
            acc.accumulate(&packed);
            remaining -= batch;
        }
    }

    pub fn sample_marginals(&mut self, total_shots: usize) -> Vec<f64> {
        let mut acc =
            crate::sim::compiled::MarginalsAccumulator::new(self.compiled.num_measurements());
        self.sample_chunked(total_shots, &mut acc);
        acc.marginals()
    }
}

/// Run noisy shot sampling using the homological sampler.
///
/// For Clifford circuits where the homology dimension h is small (≤ 20),
/// precomputes class probabilities and samples in O(1) per shot.
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

    Ok(ShotsResult {
        shots,
        probabilities: None,
    })
}

/// Compute exact noisy marginals analytically. No sampling, no rank limit.
///
/// Builds the error chain complex and compiled sampler, then computes
/// exact per-measurement noisy probabilities in O(nnz(E)) time.
/// Works for any qubit count — not limited by syndrome rank.
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
mod tests {
    use super::*;
    use crate::circuits;

    #[test]
    fn gf2_kernel_identity() {
        // Identity matrix: kernel is trivial (empty)
        let mut m = F2DenseMatrix::new(3, 3);
        m.set(0, 0);
        m.set(1, 1);
        m.set(2, 2);
        let k = gf2_kernel(&m);
        assert!(k.is_empty(), "Identity matrix should have trivial kernel");
    }

    #[test]
    fn gf2_kernel_zero_matrix() {
        // Zero matrix: kernel is the full space
        let m = F2DenseMatrix::new(3, 4);
        let k = gf2_kernel(&m);
        assert_eq!(k.len(), 4, "Zero 3×4 matrix should have 4-dim kernel");
    }

    #[test]
    fn gf2_kernel_rank_deficient() {
        // [1 1 0]
        // [0 1 1]
        // Row 0 + Row 1 = [1 0 1], so rank = 2, kernel dim = 3 - 2 = 1
        let mut m = F2DenseMatrix::new(2, 3);
        m.set(0, 0);
        m.set(0, 1);
        m.set(1, 1);
        m.set(1, 2);
        let k = gf2_kernel(&m);
        assert_eq!(k.len(), 1, "rank-2 2×3 matrix should have 1-dim kernel");
        // Kernel vector should be [1, 1, 1] (x₀ = x₁ = x₂)
        // Row 0: x₀ + x₁ = 0 → x₀ = x₁
        // Row 1: x₁ + x₂ = 0 → x₁ = x₂
        let kv = &k[0];
        assert_eq!(kv[0] & 0b111, 0b111, "kernel vector should be [1,1,1]");
    }

    #[test]
    fn gf2_kernel_verifies() {
        // Verify Mx = 0 for all kernel vectors
        let mut m = F2DenseMatrix::new(3, 5);
        // Some arbitrary matrix
        m.set(0, 0);
        m.set(0, 2);
        m.set(0, 4);
        m.set(1, 1);
        m.set(1, 3);
        m.set(2, 0);
        m.set(2, 1);
        m.set(2, 2);

        let k = gf2_kernel(&m);
        for kv in &k {
            // Check M · kv = 0
            for r in 0..3 {
                let mut dot = 0u32;
                for c in 0..5 {
                    if m.get(r, c) && (kv[c / 64] >> (c % 64)) & 1 != 0 {
                        dot ^= 1;
                    }
                }
                assert_eq!(dot, 0, "kernel vector should satisfy Mx = 0");
            }
        }
    }

    #[test]
    fn homological_ghz_compiles() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        assert!(sampler.syndrome_rank() <= n, "syndrome rank should be ≤ n");
    }

    #[test]
    fn homological_ghz_samples() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let mut sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let shots = sampler.sample_bulk(1000);
        assert_eq!(shots.len(), 1000);
        assert_eq!(shots[0].len(), n);
    }

    #[test]
    fn homological_bell_pairs() {
        let n = 4;
        let mut circuit = circuits::independent_bell_pairs(n / 2);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        // Bell pairs with noise should have non-trivial syndrome rank
        assert!(sampler.syndrome_rank() > 0);
    }

    #[test]
    fn homological_class_probs_sum_to_one() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let sum: f64 = sampler.class_probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "class probabilities should sum to 1, got {sum}"
        );
    }

    #[test]
    fn homological_matches_brute_force_statistics() {
        let n = 4;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.05);
        let num_shots = 10000;

        // Homological sampler
        let homo_result = run_shots_homological(&circuit, &noise, num_shots, 42).unwrap();

        // Brute-force sampler
        let brute_result =
            crate::sim::noise::run_shots_noisy(&circuit, &noise, num_shots, 42).unwrap();

        // Compare per-bit marginal probabilities
        let m = n;
        for bit in 0..m {
            let homo_ones: usize = homo_result.shots.iter().filter(|s| s[bit]).count();
            let brute_ones: usize = brute_result.shots.iter().filter(|s| s[bit]).count();
            let homo_p = homo_ones as f64 / num_shots as f64;
            let brute_p = brute_ones as f64 / num_shots as f64;
            let diff = (homo_p - brute_p).abs();
            assert!(
                diff < 0.05,
                "bit {bit}: homological p={homo_p:.4}, brute p={brute_p:.4}, diff={diff:.4}"
            );
        }
    }

    #[test]
    fn boundary_trivial_circuit_has_zero_homology() {
        let n = 4;
        let mut circuit = crate::circuit::Circuit::new(n, n);
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        assert_eq!(ecc.boundary_dim(), n);
        assert_eq!(ecc.homology_dim(), 0);
    }

    #[test]
    fn boundary_ghz_has_one_logical_qubit() {
        for n in [3, 5, 8] {
            let mut circuit = circuits::ghz_circuit(n);
            circuit.num_classical_bits = n;
            for i in 0..n {
                circuit.add_measure(i, i);
            }
            let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
            let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
            assert_eq!(
                ecc.homology_dim(),
                1,
                "GHZ-{n} should have 1 logical error class"
            );
            assert_eq!(ecc.boundary_dim(), n - 1);
        }
    }

    #[test]
    fn boundary_bell_pair_has_one_logical() {
        let mut circuit = crate::circuit::Circuit::new(2, 2);
        circuit.add_gate(crate::gates::Gate::H, &[0]);
        circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        assert_eq!(ecc.homology_dim(), 1);
        assert_eq!(ecc.boundary_dim(), 1);
    }

    #[test]
    fn boundary_independent_bell_pairs() {
        let n_pairs = 3;
        let n = n_pairs * 2;
        let mut circuit = circuits::independent_bell_pairs(n_pairs);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        assert_eq!(
            ecc.homology_dim(),
            n_pairs,
            "{n_pairs} bell pairs should have {n_pairs} logical error classes"
        );
    }

    #[test]
    fn boundary_exposed_via_sampler() {
        let n = 4;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        assert_eq!(sampler.homology_dim(), 1);
        assert_eq!(sampler.boundary_dim(), n - 1);
    }

    #[test]
    fn boundary_partial_measurement() {
        let mut circuit = crate::circuit::Circuit::new(3, 1);
        circuit.add_gate(crate::gates::Gate::H, &[0]);
        circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let ecc = ErrorChainComplex::build(&circuit, &noise, 42).unwrap();
        // 3 qubits, 1 measured: ker(σ) has dim 2*3-1=5
        // Stabilizers: X₀X₁, Z₀Z₁, Z₂ (3 generators)
        // X-projection on qubit 0: X₀X₁ has X on q0 → rank(A) = 1
        // boundary_dim = 3-1 = 2, homology_dim = 3-1+1 = 3
        assert_eq!(ecc.boundary_dim(), 2);
        assert_eq!(ecc.homology_dim(), 3);
    }

    #[test]
    fn packed_matches_unpacked() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

        let unpacked = s1.sample_bulk(500);
        let packed = s2.sample_packed(500);

        assert_eq!(packed.num_shots(), 500);
        assert_eq!(packed.num_measurements(), n);

        for (s, shot) in unpacked.iter().enumerate() {
            for (m, &val) in shot.iter().enumerate() {
                assert_eq!(packed.get_bit(s, m), val, "mismatch at shot={s} meas={m}");
            }
        }
    }

    #[test]
    fn marginals_matches_unpacked() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

        let num_shots = 10_000;
        let unpacked = s1.sample_bulk(num_shots);
        let marginals = s2.sample_marginals(num_shots);

        assert_eq!(marginals.len(), n);
        for m in 0..n {
            let unpacked_p = unpacked.iter().filter(|s| s[m]).count() as f64 / num_shots as f64;
            assert!(
                (marginals[m] - unpacked_p).abs() < 1e-10,
                "marginal mismatch at meas={m}: packed={}, unpacked={unpacked_p}",
                marginals[m],
            );
        }
    }

    #[test]
    fn analytical_marginals_match_sampled_small() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let analytical = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();

        let mut sampler = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let sampled = sampler.sample_marginals(100_000);

        assert_eq!(analytical.len(), n);
        assert_eq!(sampled.len(), n);
        for i in 0..n {
            assert!(
                (analytical[i] - sampled[i]).abs() < 0.01,
                "bit {i}: analytical={:.6}, sampled={:.6}",
                analytical[i],
                sampled[i],
            );
        }
    }

    #[test]
    fn analytical_marginals_ghz_50q() {
        let n = 50;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        assert_eq!(marginals.len(), n);
        for (i, &p) in marginals.iter().enumerate() {
            assert!(p > 0.0 && p < 1.0, "bit {i}: marginal {p} out of range");
            assert!(
                (p - 0.5).abs() < 0.05,
                "bit {i}: GHZ marginal should be near 0.5, got {p}"
            );
        }
    }

    #[test]
    fn analytical_marginals_ghz_100q() {
        let n = 100;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        assert_eq!(marginals.len(), n);
        for (i, &p) in marginals.iter().enumerate() {
            assert!(p > 0.0 && p < 1.0, "bit {i}: marginal {p} out of range");
        }
    }

    #[test]
    fn analytical_marginals_bell_pairs_100q() {
        let n_pairs = 50;
        let n = n_pairs * 2;
        let mut circuit = circuits::independent_bell_pairs(n_pairs);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        assert_eq!(marginals.len(), n);
        for (i, &p) in marginals.iter().enumerate() {
            assert!(
                (p - 0.5).abs() < 0.05,
                "bit {i}: bell pair marginal should be near 0.5, got {p}"
            );
        }
    }

    #[test]
    fn analytical_marginals_clifford_1000q() {
        let n = 1000;
        let mut circuit = circuits::clifford_heavy_circuit(n, 2, 42);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.001);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        assert_eq!(marginals.len(), n);
        for (i, &p) in marginals.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "bit {i}: marginal {p} out of range"
            );
        }
    }

    #[test]
    fn analytical_marginals_deterministic_bits() {
        let mut circuit = crate::circuit::Circuit::new(4, 4);
        for i in 0..4 {
            circuit.add_gate(crate::gates::Gate::X, &[i]);
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        for (i, &p) in marginals.iter().enumerate() {
            assert!(
                p > 0.95,
                "bit {i}: X-then-measure should give p(1) near 1.0, got {p}"
            );
        }
    }

    #[test]
    fn analytical_marginals_no_noise() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);

        let marginals = noisy_marginals_analytical(&circuit, &noise, 42).unwrap();
        for (i, &p) in marginals.iter().enumerate() {
            assert!(
                (p - 0.5).abs() < 1e-10,
                "bit {i}: GHZ with no noise should have marginal 0.5, got {p}"
            );
        }
    }

    #[test]
    fn chunked_accumulator_matches_packed() {
        let n = 6;
        let mut circuit = circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let mut s1 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();
        let mut s2 = HomologicalSampler::compile(&circuit, &noise, 42).unwrap();

        let num_shots = 5_000;
        let packed = s1.sample_packed(num_shots);
        let direct_counts = packed.counts();

        let mut acc = super::super::compiled::HistogramAccumulator::new();
        s2.sample_chunked(num_shots, &mut acc);
        let chunked_counts = acc.into_counts();

        assert_eq!(direct_counts, chunked_counts);
    }
}
