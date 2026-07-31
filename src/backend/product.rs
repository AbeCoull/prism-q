//! Product-state simulation backend.
//!
//! Represents each qubit independently as a 2-element amplitude vector (α, β).
//! Exact for circuits with no entangling gates, providing O(n) memory simulation.
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
//! - Circuits with zero entangling gates (e.g., single-qubit randomized benchmarking).
//! - As a fast validator for single-qubit gate correctness.
//! - Scales to arbitrarily many qubits with constant per-qubit cost.
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
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_probability_len, dense_statevector_len,
    reserve_dense_output,
};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::{DiagEntry, Gate};
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Per-qubit O(n) backend for non-entangling circuits.
pub struct ProductStateBackend {
    num_qubits: usize,
    qubits: Vec<[Complex64; 2]>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
}

impl ProductStateBackend {
    /// Create a new product-state backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            qubits: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
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
            | Gate::Multi2q(_) => Err(PrismError::BackendUnsupported {
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

impl Backend for ProductStateBackend {
    fn name(&self) -> &'static str {
        "productstate"
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

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let dim = dense_probability_len(self.name(), self.num_qubits)?;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= 14 {
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
    /// with the probabilities computed once up front.
    fn sample_basis_states(&self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
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

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut samples = BasisSamples::new(num_shots, self.num_qubits);
        for shot in 0..num_shots {
            for (qubit, &p1) in prob_one.iter().enumerate() {
                if rng.random::<f64>() < p1 {
                    samples.set(shot, qubit);
                }
            }
        }
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
        let dim = dense_statevector_len(self.name(), "statevector export", self.num_qubits)?;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= 14 {
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
mod tests {
    use super::*;
    use crate::circuit::smallvec;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;

    fn init_backend(n: usize) -> ProductStateBackend {
        let mut b = ProductStateBackend::new(42);
        b.init(n, 0).unwrap();
        b
    }

    #[test]
    fn test_init_all_zero() {
        let b = init_backend(4);
        let probs = b.probabilities().unwrap();
        assert!((probs[0] - 1.0).abs() < EPS);
        for p in &probs[1..] {
            assert!(p.abs() < EPS);
        }
    }

    #[test]
    fn test_h_creates_superposition() {
        let mut b = init_backend(1);
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        let probs = b.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < EPS);
        assert!((probs[1] - 0.5).abs() < EPS);
    }

    #[test]
    fn test_x_flips() {
        let mut b = init_backend(1);
        b.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec![0],
        })
        .unwrap();
        let probs = b.probabilities().unwrap();
        assert!(probs[0].abs() < EPS);
        assert!((probs[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_parametric_gates() {
        for gate in [Gate::Rx(PI / 3.0), Gate::Ry(PI / 4.0), Gate::Rz(PI / 6.0)] {
            let mut b = init_backend(1);
            b.apply(&Instruction::Gate {
                gate,
                targets: smallvec![0],
            })
            .unwrap();
            let probs = b.probabilities().unwrap();
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < EPS, "probabilities must sum to 1");
        }
    }

    #[test]
    fn test_multi_qubit_independent() {
        let mut b = init_backend(3);
        b.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec![0],
        })
        .unwrap();
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![1],
        })
        .unwrap();

        let probs = b.probabilities().unwrap();
        // q0=1, q1=+, q2=0 → indices with bit0=1, bit2=0
        // |001⟩ (idx 1) = 0.5, |011⟩ (idx 3) = 0.5
        assert!((probs[1] - 0.5).abs() < EPS);
        assert!((probs[3] - 0.5).abs() < EPS);
        for (i, &p) in probs.iter().enumerate() {
            if i != 1 && i != 3 {
                assert!(p.abs() < EPS, "prob[{i}] should be 0, got {p}");
            }
        }
    }

    #[test]
    fn test_rejects_cx() {
        let mut b = init_backend(2);
        let result = b.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec![0, 1],
        });
        assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
    }

    #[test]
    fn test_rejects_cz() {
        let mut b = init_backend(2);
        let result = b.apply(&Instruction::Gate {
            gate: Gate::Cz,
            targets: smallvec![0, 1],
        });
        assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
    }

    #[test]
    fn test_rejects_swap() {
        let mut b = init_backend(2);
        let result = b.apply(&Instruction::Gate {
            gate: Gate::Swap,
            targets: smallvec![0, 1],
        });
        assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
    }

    #[test]
    fn test_rejects_cu() {
        let mut b = init_backend(2);
        let result = b.apply(&Instruction::Gate {
            gate: Gate::Cu(Box::new(Gate::H.matrix_2x2())),
            targets: smallvec![0, 1],
        });
        assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
    }

    #[test]
    fn test_rejects_mcu() {
        use crate::gates::McuData;
        let mut b = init_backend(3);
        let result = b.apply(&Instruction::Gate {
            gate: Gate::Mcu(Box::new(McuData {
                mat: Gate::X.matrix_2x2(),
                num_controls: 2,
            })),
            targets: smallvec![0, 1, 2],
        });
        assert!(matches!(result, Err(PrismError::BackendUnsupported { .. })));
    }

    #[test]
    fn test_measurement_collapses() {
        let mut b = ProductStateBackend::new(42);
        b.init(1, 1).unwrap();
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        b.apply(&Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        })
        .unwrap();
        let outcome = b.classical_results()[0];

        let probs = b.probabilities().unwrap();
        if outcome {
            assert!(probs[0].abs() < EPS);
            assert!((probs[1] - 1.0).abs() < EPS);
        } else {
            assert!((probs[0] - 1.0).abs() < EPS);
            assert!(probs[1].abs() < EPS);
        }
    }

    #[test]
    fn test_probabilities_tensor_product() {
        let mut b = init_backend(3);
        // q0: H → equal superposition
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        // q1: X → |1⟩
        b.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec![1],
        })
        .unwrap();
        // q2: stays |0⟩

        let probs = b.probabilities().unwrap();
        assert_eq!(probs.len(), 8);

        // q0=0/1 (50/50), q1=1 (certain), q2=0 (certain)
        // |010⟩ (idx 2) = 0.5, |011⟩ (idx 3) = 0.5
        assert!((probs[2] - 0.5).abs() < EPS);
        assert!((probs[3] - 0.5).abs() < EPS);
        for (i, &p) in probs.iter().enumerate() {
            if i != 2 && i != 3 {
                assert!(p.abs() < EPS, "prob[{i}] should be 0, got {p}");
            }
        }
    }

    #[test]
    fn test_sample_basis_states_matches_the_dense_distribution() {
        let mut b = init_backend(6);
        for q in 0..6 {
            b.apply(&Instruction::Gate {
                gate: Gate::Ry(0.3 + 0.2 * q as f64),
                targets: smallvec![q],
            })
            .unwrap();
        }

        let probs = b.probabilities().unwrap();
        let shots = 40_000;
        let samples = b.sample_basis_states(shots, 42).unwrap();

        let mut counts = vec![0usize; probs.len()];
        for shot in 0..shots {
            let index = (0..6)
                .filter(|&q| samples.bit(shot, q))
                .fold(0, |acc, q| acc | 1 << q);
            counts[index] += 1;
        }
        for (index, &p) in probs.iter().enumerate() {
            let frequency = counts[index] as f64 / shots as f64;
            let band = 4.0 * (p * (1.0 - p) / shots as f64).sqrt() + 0.002;
            assert!(
                (frequency - p).abs() < band,
                "basis state {index}: sampled {frequency:.6} against {p:.6}"
            );
        }
    }

    #[test]
    fn test_sample_basis_states_repeats_from_the_seed() {
        let mut b = init_backend(3);
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![1],
        })
        .unwrap();
        let first = b.sample_basis_states(64, 42).unwrap();
        let second = b.sample_basis_states(64, 42).unwrap();
        let other_seed = b.sample_basis_states(64, 43).unwrap();

        let bits = |s: &BasisSamples| -> Vec<bool> {
            (0..64)
                .flat_map(|shot| (0..3).map(move |q| (shot, q)))
                .map(|(shot, q)| s.bit(shot, q))
                .collect()
        };
        assert_eq!(bits(&first), bits(&second));
        assert_ne!(bits(&first), bits(&other_seed));
    }

    #[test]
    fn test_pauli_expectations_are_normalization_independent() {
        let mut b = init_backend(2);
        b.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        let scaled = b.qubits[1].map(|amp| amp * 3.0);
        b.qubits[1] = scaled;

        let values = b
            .pauli_expectations(&[
                vec![PauliTerm::x(0)],
                vec![PauliTerm::z(1)],
                vec![PauliTerm::x(0), PauliTerm::z(1)],
                vec![],
            ])
            .unwrap();
        for (got, want) in values.iter().zip(&[1.0, 1.0, 1.0, 1.0]) {
            assert!((got - want).abs() < EPS, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_pauli_expectations_reject_bad_observables() {
        let b = init_backend(2);
        assert!(matches!(
            b.pauli_expectations(&[vec![PauliTerm::z(2)]]),
            Err(PrismError::InvalidQubit { .. })
        ));
        assert!(matches!(
            b.pauli_expectations(&[vec![PauliTerm::z(0), PauliTerm::x(0)]]),
            Err(PrismError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn test_fused_gate() {
        use crate::gates::mat_mul_2x2;

        let fused_mat = mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
        let mut b = init_backend(1);
        b.apply(&Instruction::Gate {
            gate: Gate::Fused(Box::new(fused_mat)),
            targets: smallvec![0],
        })
        .unwrap();

        let mut b2 = init_backend(1);
        b2.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec![0],
        })
        .unwrap();
        b2.apply(&Instruction::Gate {
            gate: Gate::T,
            targets: smallvec![0],
        })
        .unwrap();

        let p1 = b.probabilities().unwrap();
        let p2 = b2.probabilities().unwrap();
        for (a, b) in p1.iter().zip(&p2) {
            assert!((a - b).abs() < EPS);
        }
    }
}
