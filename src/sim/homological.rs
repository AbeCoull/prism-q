//! Error chain complex of a noisy Clifford circuit and the exact analytic noisy
//! marginals read off it.

use crate::circuit::{Circuit, Instruction};
use crate::error::Result;
use crate::sim::compiled::batch_propagate_backward;
use crate::sim::noise::NoiseModel;

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

    #[cfg(test)]
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

/// Basis of ker(M) over GF(2) for M ∈ F₂^{m×n}, each vector packed into `u64`
/// words of length n, read off a row reduction of `M^T` augmented with `I_n`.
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

fn has_bit(words: &[u64], i: usize) -> bool {
    (words[i / 64] >> (i % 64)) & 1 != 0
}

fn first_bit(words: &[u64]) -> Option<usize> {
    words
        .iter()
        .position(|&w| w != 0)
        .map(|w| w * 64 + words[w].trailing_zeros() as usize)
}

/// Multiply generator `src` into every generator in `targets`, with generators stored
/// as one bit column across the per-qubit `x` and `z` rows.
fn multiply_into(x: &mut [Vec<u64>], z: &mut [Vec<u64>], src: usize, targets: &[u64]) {
    for row in x.iter_mut().chain(z.iter_mut()) {
        if (row[src / 64] >> (src % 64)) & 1 != 0 {
            for (word, &t) in row.iter_mut().zip(targets) {
                *word ^= t;
            }
        }
    }
}

fn clear_generator(x: &mut [Vec<u64>], z: &mut [Vec<u64>], g: usize) {
    for row in x.iter_mut().chain(z.iter_mut()) {
        row[g / 64] &= !(1u64 << (g % 64));
    }
}

/// Rank over GF(2) of the first `num_rows` rows, reducing them in place.
fn gf2_rank(matrix: &mut F2DenseMatrix, num_rows: usize, num_cols: usize) -> usize {
    let row_words = matrix.row_words;
    let rows = &mut matrix.data;
    let mut rank = 0;
    for col in 0..num_cols {
        let Some(pr) = (rank..num_rows).find(|&r| has_bit(&rows[r * row_words..], col)) else {
            continue;
        };

        if pr != rank {
            for w in 0..row_words {
                rows.swap(rank * row_words + w, pr * row_words + w);
            }
        }

        for r in 0..num_rows {
            if r != rank && has_bit(&rows[r * row_words..], col) {
                for w in 0..row_words {
                    rows[r * row_words + w] ^= rows[rank * row_words + w];
                }
            }
        }

        rank += 1;
    }
    rank
}

/// Whether Z on `qubit` is a product of the sign-free generators.
fn z_in_span(x: &[Vec<u64>], z: &[Vec<u64>], qubit: usize) -> bool {
    let n = x.len();
    let rank = |with_z: bool| {
        let mut matrix = F2DenseMatrix::new(n + 1, 2 * n);
        for g in 0..n {
            for q in 0..n {
                if has_bit(&x[q], g) {
                    matrix.set(g, q);
                }
                if has_bit(&z[q], g) {
                    matrix.set(g, n + q);
                }
            }
        }
        if with_z {
            matrix.set(n, n + qubit);
        }
        gf2_rank(&mut matrix, n + 1, 2 * n)
    };
    rank(true) == rank(false)
}

/// Measure Z on `qubit` against sign-free generators. When the outcome is random, put
/// Z on `qubit` in place of the first anticommuting generator, or in a free slot when
/// it commutes with a mixed state's generators without being one of their products,
/// and return that slot.
fn collapse_onto_z(
    x: &mut [Vec<u64>],
    z: &mut [Vec<u64>],
    qubit: usize,
    free_slots: &mut usize,
) -> Option<usize> {
    let mut anticommuting = x[qubit].clone();
    let slot = match first_bit(&anticommuting) {
        Some(a) => {
            anticommuting[a / 64] &= !(1u64 << (a % 64));
            multiply_into(x, z, a, &anticommuting);
            clear_generator(x, z, a);
            a
        }
        None if *free_slots > 0 && !z_in_span(x, z, qubit) => {
            let mut occupied = vec![0u64; anticommuting.len()];
            for row in x.iter().chain(z.iter()) {
                for (o, &w) in occupied.iter_mut().zip(row) {
                    *o |= w;
                }
            }
            *free_slots -= 1;
            (0..x.len())
                .find(|&g| !has_bit(&occupied, g))
                .expect("independent generators fewer than n leave a zero slot")
        }
        None => return None,
    };
    z[qubit][slot / 64] |= 1u64 << (slot % 64);
    Some(slot)
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
    /// dim(im(∂₂) ∩ ker(∂₁)): trivial errors no final measurement detects
    boundary_dim: usize,
    /// dim(H₁) = dim(ker(∂₁)/im(∂₂)): independent logical error classes
    homology_dim: usize,
}

impl ErrorChainComplex {
    /// Build the error chain complex from a Clifford circuit and noise model.
    ///
    /// Uses backward Pauli propagation (same as the compiled noisy sampler)
    /// to determine which measurements are sensitive to each error location. A circuit
    /// with a non-Clifford or guarded gate, or a model with any channel other than a
    /// single-qubit Pauli, returns `IncompatibleBackend`.
    pub fn build(circuit: &Circuit, noise: &NoiseModel, _seed: u64) -> Result<Self> {
        noise.validate_for(circuit)?;
        noise.ensure_no_guarded_events(circuit, "error chain complex")?;
        noise.ensure_chain_complex_channels()?;
        if !circuit.is_clifford_only() {
            return Err(crate::error::PrismError::IncompatibleBackend {
                backend: "error chain complex".to_string(),
                reason: "a non-Clifford gate maps a Pauli error to a sum of Paulis, which \
                         no single error column describes"
                    .to_string(),
            });
        }
        let m = circuit
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. }))
            .count();
        if m == 0 {
            let (boundary_dim, homology_dim) =
                Self::compute_boundary_space(circuit, circuit.num_qubits);
            return Ok(Self {
                e_matrix: F2DenseMatrix::new(0, 0),
                error_probs: Vec::new(),
                num_measurements: 0,
                num_errors: 0,
                boundary_dim,
                homology_dim,
            });
        }

        let m_words = m.div_ceil(64);
        let n = circuit.num_qubits;

        let mut x_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
        let mut z_packed: Vec<Vec<u64>> = vec![vec![0u64; m_words]; n];
        let mut sign_packed = vec![0u64; m_words];

        let mut error_probs = Vec::new();
        let mut e_cols: Vec<Vec<u64>> = Vec::new();

        let mut meas_idx = m;
        for (instr_idx, instr) in circuit.instructions.iter().enumerate().rev() {
            for event in &noise.after_gate[instr_idx] {
                // An inert channel contributes no column, whatever its
                // variant.
                if event.channel.is_inert() {
                    continue;
                }
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

            match instr {
                Instruction::Gate { gate, targets } => {
                    batch_propagate_backward(
                        &mut x_packed,
                        &mut z_packed,
                        &mut sign_packed,
                        gate,
                        targets.as_slice(),
                        m_words,
                    );
                }
                Instruction::Measure { qubit, .. } => {
                    meas_idx -= 1;
                    z_packed[*qubit][meas_idx / 64] |= 1u64 << (meas_idx % 64);
                }
                Instruction::Reset { qubit } => {
                    x_packed[*qubit].fill(0);
                    z_packed[*qubit].fill(0);
                }
                Instruction::Barrier { .. } | Instruction::Save { .. } => {}
                Instruction::Conditional { .. } => {
                    return Err(crate::error::PrismError::IncompatibleBackend {
                        backend: "error chain complex".to_string(),
                        reason: "a guarded gate runs only on the shots its condition holds, \
                                 and an error on a guard bit can decide that, so which \
                                 measurements an earlier error reaches varies per shot; one \
                                 error matrix cannot describe it"
                            .to_string(),
                    });
                }
                Instruction::Region(_) => {
                    return Err(crate::error::PrismError::IncompatibleBackend {
                        backend: "HomologicalDetectorModel".to_string(),
                        reason: "error propagation does not support guarded regions".to_string(),
                    });
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
    /// Returns (boundary_dim, homology_dim) for Pauli errors just before the final
    /// measurements, those followed by no other operation on their qubit:
    /// - boundary_dim = dim(im(∂₂) ∩ ker(∂₁)) = trivial errors no final measurement detects
    /// - homology_dim = dim(H₁) = independent logical error classes
    ///
    /// Algorithm: forward-propagate Z_0,...,Z_{n-1} through the circuit, collapsing the
    /// sign-free generators at every earlier measurement and reset. A reset with a random
    /// outcome also drops a generator whose sign it scrambles, when one exists, which
    /// leaves a mixed state.
    ///
    /// An error is trivial when it commutes with every generator, and ker(σ) is the
    /// Paulis with no X on a measured qubit, so dim(H₁) is the rank of the commutation
    /// constraints on ker(σ): one row per generator, its Z on unmeasured qubits beside
    /// its X on all qubits. dim(ker σ) = 2n - num_measured.
    fn compute_boundary_space(circuit: &Circuit, n: usize) -> (usize, usize) {
        if n == 0 {
            return (0, 0);
        }

        let mut last_op = vec![None; n];
        for (idx, instr) in circuit.instructions.iter().enumerate() {
            match instr {
                Instruction::Gate { targets, .. } => {
                    for &q in targets {
                        last_op[q] = Some(idx);
                    }
                }
                Instruction::Measure { qubit, .. } | Instruction::Reset { qubit } => {
                    last_op[*qubit] = Some(idx);
                }
                _ => {}
            }
        }

        let n_words = n.div_ceil(64);
        let mut stab_x: Vec<Vec<u64>> = vec![vec![0u64; n_words]; n];
        let mut stab_z: Vec<Vec<u64>> = vec![vec![0u64; n_words]; n];
        let mut stab_sign = vec![0u64; n_words];
        let mut free_slots = 0;

        for i in 0..n {
            stab_z[i][i / 64] |= 1u64 << (i % 64);
        }

        for (idx, instr) in circuit.instructions.iter().enumerate() {
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
                Instruction::Measure { qubit, .. } if last_op[*qubit] != Some(idx) => {
                    collapse_onto_z(&mut stab_x, &mut stab_z, *qubit, &mut free_slots);
                }
                Instruction::Reset { qubit } => {
                    let collapsed =
                        collapse_onto_z(&mut stab_x, &mut stab_z, *qubit, &mut free_slots);
                    if let Some(a) = collapsed {
                        let mut scrambled = stab_z[*qubit].clone();
                        scrambled[a / 64] &= !(1u64 << (a % 64));
                        if let Some(c) = first_bit(&scrambled) {
                            scrambled[c / 64] &= !(1u64 << (c % 64));
                            multiply_into(&mut stab_x, &mut stab_z, c, &scrambled);
                            clear_generator(&mut stab_x, &mut stab_z, c);
                            free_slots += 1;
                        }
                    }
                }
                _ => {}
            }
        }

        let unmeasured: Vec<usize> = (0..n)
            .filter(|&q| {
                !matches!(
                    last_op[q].map(|idx| &circuit.instructions[idx]),
                    Some(Instruction::Measure { .. })
                )
            })
            .collect();
        let num_measured = n - unmeasured.len();

        let num_cols = unmeasured.len() + n;
        let mut constraints = F2DenseMatrix::new(n, num_cols);
        for g in 0..n {
            for (col, &q) in unmeasured.iter().enumerate() {
                if has_bit(&stab_z[q], g) {
                    constraints.set(g, col);
                }
            }
            for (q, row) in stab_x.iter().enumerate() {
                if has_bit(row, g) {
                    constraints.set(g, unmeasured.len() + q);
                }
            }
        }

        let rank = gf2_rank(&mut constraints, n, num_cols);
        let boundary_dim = 2 * n - num_measured - rank;
        let homology_dim = rank;
        (boundary_dim, homology_dim)
    }

    /// dim(im(∂₂) ∩ ker(∂₁)): independent Pauli errors at the end of the circuit that
    /// leave the state unchanged and that no final measurement (one with no later
    /// operation on its qubit) detects.
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

/// Compute exact noisy marginals analytically. No sampling, no rank limit.
///
/// Builds the error chain complex and compiled sampler, then computes
/// exact per-measurement noisy probabilities in O(nnz(E)) time.
/// Works for any qubit count, not limited by syndrome rank.
///
/// Readout error needs no syndrome class on a marginal: a bit reported 1 was
/// either measured 1 and not flipped or measured 0 and flipped, so the reported
/// probability is `p(1 - p10) + (1 - p) p01`. Per-shot sampling is still the
/// only route for correlations between bits.
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
            let p = noisy[mi];
            result[cbit] = match noise.readout.get(cbit).and_then(Option::as_ref) {
                Some(err) => p * (1.0 - err.p10) + (1.0 - p) * err.p01,
                None => p,
            };
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "homological_tests.rs"]
mod tests;
