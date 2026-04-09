//! Product-state simulation backend.
//!
//! Represents each qubit independently as a 2-element amplitude vector (α, β).
//! Exact for circuits with no entangling gates, providing O(n) memory simulation.
//!
//! # Memory layout
//!
//! - `Vec<[Complex64; 2]>` — one state per qubit, `[α, β]` where `|ψ⟩ = α|0⟩ + β|1⟩`.
//! - Total memory: O(n) — 32 bytes per qubit vs O(2^n) for statevector.
//!
//! # Gate support
//!
//! All single-qubit gates are applied as a 2×2 matrix–vector multiply on the
//! per-qubit state. Two-qubit and multi-qubit gates return `BackendUnsupported`
//! since product states cannot represent entanglement.
//!
//! # When to prefer this backend
//!
//! - Circuits with zero entangling gates (e.g., single-qubit randomized benchmarking).
//! - As a fast validator for single-qubit gate correctness.
//! - Scales to arbitrarily many qubits with constant per-qubit cost.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::backend::{Backend, MAX_PROB_QUBITS, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

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

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Cx
            | Gate::Cz
            | Gate::Swap
            | Gate::Cu(_)
            | Gate::Mcu(_)
            | Gate::BatchPhase(_)
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
                    let [a, b] = self.qubits[target];
                    self.qubits[target] =
                        [mat[0][0] * a + mat[0][1] * b, mat[1][0] * a + mat[1][1] * b];
                }
                Ok(())
            }
            _ => {
                let target = targets[0];
                let mat = gate.matrix_2x2();
                let [a, b] = self.qubits[target];
                self.qubits[target] =
                    [mat[0][0] * a + mat[0][1] * b, mat[1][0] * a + mat[1][1] * b];
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
                let outcome = self.rng.gen::<f64>() < prob_one;
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

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.num_qubits > MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probabilities for {} qubits (max {})",
                    self.num_qubits, MAX_PROB_QUBITS
                ),
            });
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= 14 {
            use rayon::prelude::*;
            let n = self.num_qubits;
            let dim = 1usize << n;
            let qubit_probs: Vec<[f64; 2]> = self
                .qubits
                .iter()
                .map(|q| [q[0].norm_sqr(), q[1].norm_sqr()])
                .collect();
            let probs: Vec<f64> = (0..dim)
                .into_par_iter()
                .map(|idx| {
                    let mut p = 1.0f64;
                    for q in 0..n {
                        p *= qubit_probs[q][(idx >> q) & 1];
                    }
                    p
                })
                .collect();
            return Ok(probs);
        }

        let mut probs = vec![1.0f64];
        for q in 0..self.num_qubits {
            let p0 = self.qubits[q][0].norm_sqr();
            let p1 = self.qubits[q][1].norm_sqr();
            let len = probs.len();
            probs.reserve(len);
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

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        if self.num_qubits > MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits (max {})",
                    self.num_qubits, MAX_PROB_QUBITS
                ),
            });
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= 14 {
            use rayon::prelude::*;
            let n = self.num_qubits;
            let dim = 1usize << n;
            let qubits = &self.qubits;
            let sv: Vec<Complex64> = (0..dim)
                .into_par_iter()
                .map(|idx| {
                    let mut amp = Complex64::new(1.0, 0.0);
                    for q in 0..n {
                        amp *= qubits[q][(idx >> q) & 1];
                    }
                    amp
                })
                .collect();
            return Ok(sv);
        }

        // Qubit 0 is LSB: index bit 0 selects qubit 0's state.
        let mut sv = vec![Complex64::new(1.0, 0.0)];
        for q in 0..self.num_qubits {
            let a = self.qubits[q][0]; // α = ⟨0|ψ_q⟩
            let b = self.qubits[q][1]; // β = ⟨1|ψ_q⟩
            let len = sv.len();
            sv.reserve(len);
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
