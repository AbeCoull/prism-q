//! Product-state simulation backend.
//!
//! Represents each qubit independently as a 2-element amplitude vector (α, β),
//! exact for circuits with no entangling gates.
//!
//! # Memory layout
//!
//! - `Vec<[Complex64; 2]>`: one state per qubit, `[α, β]` where `|ψ⟩ = α|0⟩ + β|1⟩`.
//! - Total memory: O(n), 32 bytes per qubit vs O(2^n) for statevector.
//!
//! # Gate support
//!
//! All single-qubit gates are applied as a 2×2 matrix-vector multiply on the
//! per-qubit state; `MultiFused` batches and single-qubit `DiagonalBatch`
//! entries apply the same way per target. Entangling gates return
//! `BackendUnsupported` since product states cannot represent entanglement.
//!
//! # Queries
//!
//! Shots and Pauli expectations answer from the per-qubit states: a shot is one
//! Bernoulli draw per qubit and an observable is one factor per qubit it names,
//! so neither builds the `2^n` vector [`Backend::probabilities`] returns.
//!
//! # When to prefer this backend
//!
//! - Circuits with zero entangling gates (e.g., single-qubit randomized
//!   benchmarking), at any width.
//!
//! # When NOT to use this backend
//!
//! - Circuits containing any entangling gate; the run stops with
//!   `BackendUnsupported` at the first one.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::backend::{
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_probability_len, dense_statevector_len, overlap,
    reserve_dense_output,
};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::{DiagEntry, Gate};
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Per-qubit O(n) backend for non-entangling circuits.
pub struct ProductStateBackend {
    num_qubits: usize,
    qubits: Vec<[Complex64; 2]>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    limits: ResolvedLimits,
}

impl ProductStateBackend {
    /// Backend under the process default [`Limits`](crate::Limits).
    pub fn new(seed: u64) -> Self {
        Self::with_limits(seed, &ResolvedLimits::process_default())
    }

    pub(crate) fn with_limits(seed: u64, limits: &ResolvedLimits) -> Self {
        Self {
            num_qubits: 0,
            qubits: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            limits: *limits,
        }
    }

    #[inline(always)]
    fn apply_single_qubit_matrix(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let [a, b] = self.qubits[target];
        self.qubits[target] = [mat[0][0] * a + mat[0][1] * b, mat[1][0] * a + mat[1][1] * b];
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Rzz(_)
            | Gate::Cx
            | Gate::Cz
            | Gate::Swap
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::BatchPhase(_)
            | Gate::BatchRzz(_)
            | Gate::Fused2q(_)
            | Gate::Multi2q(_)
            | Gate::Unitary(_) => Err(PrismError::BackendUnsupported {
                backend: "productstate".to_string(),
                operation: format!(
                    "entangling gate `{}` (product state backend supports single-qubit gates only)",
                    gate.name()
                ),
            }),
            Gate::MultiFused(data) => {
                for &(target, mat) in &data.gates {
                    self.apply_single_qubit_matrix(target, mat);
                }
                Ok(())
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    match entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            let [a, b] = self.qubits[*qubit];
                            self.qubits[*qubit] = [*d0 * a, *d1 * b];
                        }
                        DiagEntry::Phase2q { .. } | DiagEntry::Parity2q { .. } => {
                            return Err(PrismError::BackendUnsupported {
                                backend: "productstate".to_string(),
                                operation:
                                    "multi-qubit diagonal batch (product state backend supports single-qubit gates only)"
                                        .to_string(),
                            });
                        }
                    }
                }
                Ok(())
            }
            _ => {
                let target = targets[0];
                self.apply_single_qubit_matrix(target, gate.matrix_2x2());
                Ok(())
            }
        }
    }
}

impl crate::backend::sealed::Sealed for ProductStateBackend {
    fn limits(&self) -> crate::limits::ResolvedLimits {
        self.limits
    }
}

impl Backend for ProductStateBackend {
    fn name(&self) -> &'static str {
        "productstate"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::ProductState
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        let zero_state = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

        if self.qubits.len() == num_qubits {
            self.qubits.fill(zero_state);
        } else {
            self.qubits = vec![zero_state; num_qubits];
        }

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
                let [alpha, beta] = self.qubits[*qubit];
                let prob_one = beta.norm_sqr().clamp(0.0, 1.0);
                let outcome = self.rng.random::<f64>() < prob_one;
                self.classical_bits[*classical_bit] = outcome;

                if outcome {
                    let norm = prob_one.clamp(NORM_CLAMP_MIN, 1.0).sqrt();
                    self.qubits[*qubit] = [Complex64::new(0.0, 0.0), beta / norm];
                } else {
                    let norm = (1.0 - prob_one).clamp(NORM_CLAMP_MIN, 1.0).sqrt();
                    self.qubits[*qubit] = [alpha / norm, Complex64::new(0.0, 0.0)];
                }
            }
            Instruction::Reset { qubit } => {
                self.qubits[*qubit] = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied("ProductState", label));
            }
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
        self.qubits[qubit] = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        Ok(())
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.apply_single_qubit_matrix(qubit, *matrix);
        Ok(())
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let [alpha, beta] = self.qubits[qubit];
        let r = beta * alpha.conj();
        Ok([
            [Complex64::new(alpha.norm_sqr(), 0.0), r.conj()],
            [r, Complex64::new(beta.norm_sqr(), 0.0)],
        ])
    }

    fn schmidt_values(&mut self, subsystem: &[usize]) -> Result<Vec<f64>> {
        crate::backend::schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        Ok(vec![1.0])
    }

    /// Product of the per-qubit inner products, `O(n)` and no width cap: the
    /// state factorizes, so the overlap with another product state does.
    fn overlap_sq(&self, other: &dyn Backend) -> Result<f64> {
        if let Some(product) = other
            .as_any()
            .and_then(|any| any.downcast_ref::<ProductStateBackend>())
            && product.num_qubits == self.num_qubits
        {
            let (mut inner_sq, mut left, mut right) = (1.0, 1.0, 1.0);
            for (a, b) in self.qubits.iter().zip(&product.qubits) {
                inner_sq *= (a[0].conj() * b[0] + a[1].conj() * b[1]).norm_sqr();
                left *= a[0].norm_sqr() + a[1].norm_sqr();
                right *= b[0].norm_sqr() + b[1].norm_sqr();
            }
            return Ok(overlap::normalized(inner_sq, left, right));
        }
        overlap::export_overlap_sq(
            self.name(),
            self.num_qubits(),
            || self.export_statevector(),
            other,
            &self.limits,
        )
    }

    /// Kronecker product of the per-qubit `2 x 2` factors, each qubit of
    /// `subsystem` taken as the next higher bit, `4^k` multiplies in all,
    /// scaled to trace one.
    fn reduced_density_matrix(&mut self, subsystem: &[usize]) -> Result<Vec<Complex64>> {
        crate::backend::schmidt::validate_qubit_set(subsystem, self.num_qubits)?;
        let side = crate::backend::reduced_density::reduced_density_side(
            self.name(),
            subsystem.len(),
            &self.limits,
        )?;
        let mut rho = vec![Complex64::new(1.0, 0.0)];
        let mut dim = 1usize;
        for &qubit in subsystem {
            let amp = self.qubits[qubit];
            let mut next = vec![Complex64::new(0.0, 0.0); 4 * dim * dim];
            for (b, &a) in amp.iter().enumerate() {
                for (bp, &ap) in amp.iter().enumerate() {
                    let factor = a * ap.conj();
                    for r in 0..dim {
                        let out = &mut next[(b * dim + r) * 2 * dim + bp * dim..][..dim];
                        for (entry, &old) in out.iter_mut().zip(&rho[r * dim..(r + 1) * dim]) {
                            *entry = factor * old;
                        }
                    }
                }
            }
            rho = next;
            dim *= 2;
        }
        crate::backend::reduced_density::normalize_trace(&mut rho, side);
        Ok(rho)
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let dim = dense_probability_len(self.name(), self.num_qubits, &self.limits)?;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use rayon::prelude::*;
            let n = self.num_qubits;
            let qubit_probs: Vec<[f64; 2]> = self
                .qubits
                .iter()
                .map(|q| [q[0].norm_sqr(), q[1].norm_sqr()])
                .collect();
            let mut probs = Vec::new();
            reserve_dense_output(&mut probs, dim, self.name(), "probabilities")?;
            probs.resize(dim, 0.0);
            probs.par_iter_mut().enumerate().for_each(|(idx, prob)| {
                let mut p = 1.0f64;
                for q in 0..n {
                    p *= qubit_probs[q][(idx >> q) & 1];
                }
                *prob = p;
            });
            return Ok(probs);
        }

        let mut probs = Vec::new();
        reserve_dense_output(&mut probs, dim, self.name(), "probabilities")?;
        probs.push(1.0f64);
        for q in 0..self.num_qubits {
            let p0 = self.qubits[q][0].norm_sqr();
            let p1 = self.qubits[q][1].norm_sqr();
            let len = probs.len();
            for i in 0..len {
                probs.push(probs[i] * p1);
            }
            for p in probs.iter_mut().take(len) {
                *p *= p0;
            }
        }
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Qubits are independent, so a shot is `n` Bernoulli draws on the
    /// per-qubit `|β|²` rather than one draw from a `2^n` CDF: `O(shots·n)`
    /// with the probabilities computed once up front. Shots draw in blocks of
    /// 256, one ChaCha8 stream per block, which split across workers once the
    /// call makes enough draws; the words are the same at any thread count.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let prob_one: Vec<f64> = self
            .qubits
            .iter()
            .map(|[alpha, beta]| {
                let weight_one = beta.norm_sqr();
                let norm = alpha.norm_sqr() + weight_one;
                if norm > 0.0 {
                    (weight_one / norm).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect();

        use crate::sim::shots::{draws_split_across_workers, sample_in_shot_blocks};

        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        let words_per_shot = samples.words_per_shot();
        let parallel = draws_split_across_workers(num_shots, self.num_qubits);
        sample_in_shot_blocks(
            samples.words_mut(),
            words_per_shot,
            seed,
            parallel,
            |rng, block_words| {
                for shot_words in block_words.chunks_mut(words_per_shot) {
                    for (qubit, &p1) in prob_one.iter().enumerate() {
                        if rng.random::<f64>() < p1 {
                            shot_words[qubit / 64] |= 1u64 << (qubit % 64);
                        }
                    }
                }
            },
        );
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// A product state factorizes a joint Pauli into one single-qubit
    /// expectation per factor, so a term costs `O(1)` per qubit it names and
    /// identity qubits contribute nothing.
    ///
    /// Each factor divides by its own qubit's `⟨φ|φ⟩`, which is how the whole
    /// value ends up divided by `⟨ψ|ψ⟩`: the untouched qubits cancel between
    /// numerator and denominator.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        observables
            .iter()
            .map(|observable| {
                crate::sim::validate_observable(observable, self.num_qubits)?;
                let mut product = 1.0f64;
                for term in observable {
                    let [alpha, beta] = self.qubits[term.qubit];
                    let norm = alpha.norm_sqr() + beta.norm_sqr();
                    if norm == 0.0 {
                        return Ok(0.0);
                    }
                    let off_diagonal = alpha.conj() * beta;
                    let factor = match term.axis {
                        PauliAxis::X => 2.0 * off_diagonal.re,
                        PauliAxis::Y => 2.0 * off_diagonal.im,
                        PauliAxis::Z => alpha.norm_sqr() - beta.norm_sqr(),
                    };
                    product *= factor / norm;
                }
                Ok(product)
            })
            .collect()
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(
            self.name(),
            "statevector export",
            self.num_qubits,
            &self.limits,
        )?;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
            use rayon::prelude::*;
            let n = self.num_qubits;
            let qubits = &self.qubits;
            let mut sv = Vec::new();
            reserve_dense_output(&mut sv, dim, self.name(), "statevector export")?;
            sv.resize(dim, Complex64::new(0.0, 0.0));
            sv.par_iter_mut().enumerate().for_each(|(idx, amp_out)| {
                let mut amp = Complex64::new(1.0, 0.0);
                for q in 0..n {
                    amp *= qubits[q][(idx >> q) & 1];
                }
                *amp_out = amp;
            });
            return Ok(sv);
        }

        // Qubit 0 is LSB: index bit 0 selects qubit 0's state.
        let mut sv = Vec::new();
        reserve_dense_output(&mut sv, dim, self.name(), "statevector export")?;
        sv.push(Complex64::new(1.0, 0.0));
        for q in 0..self.num_qubits {
            let a = self.qubits[q][0];
            let b = self.qubits[q][1];
            let len = sv.len();
            for i in 0..len {
                sv.push(sv[i] * b);
            }
            for s in sv.iter_mut().take(len) {
                *s *= a;
            }
        }
        Ok(sv)
    }
}

#[cfg(test)]
#[path = "product_tests.rs"]
mod tests;
