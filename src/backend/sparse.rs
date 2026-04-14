//! Sparse state-vector simulation backend.
//!
//! Stores only non-zero amplitudes in a `HashMap<usize, Complex64>`, giving O(k) memory
//! where k is the number of non-zero basis states. Amplitudes below a configurable
//! epsilon are pruned after each gate to maintain sparsity.
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

use std::collections::HashMap;

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
const MIN_STATES_FOR_PAR: usize = 4096;

use crate::backend::{is_phase_one, Backend, MAX_PROB_QUBITS};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::{DiagEntry, Gate};

const DEFAULT_EPSILON: f64 = 1e-16;

/// Sparse state-vector backend — O(k) where k is the number of non-zero amplitudes.
pub struct SparseBackend {
    num_qubits: usize,
    state: HashMap<usize, Complex64>,
    swap_buf: HashMap<usize, Complex64>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    epsilon: f64,
}

impl SparseBackend {
    /// Create a new sparse backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            state: HashMap::new(),
            swap_buf: HashMap::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            epsilon: DEFAULT_EPSILON,
        }
    }

    #[inline(always)]
    fn prune(&mut self) {
        let eps = self.epsilon;
        self.state.retain(|_, amp| amp.norm_sqr() >= eps);
    }

    #[inline(always)]
    fn apply_single_qubit(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
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
    }

    /// CX is a deterministic 1:1 index mapping — no near-zero amplitudes created.
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
    fn apply_cu(&mut self, control: usize, target: usize, mat: [[Complex64; 2]; 2]) {
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
    }

    #[inline(always)]
    fn apply_mcu(&mut self, controls: &[usize], target: usize, mat: [[Complex64; 2]; 2]) {
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

    fn apply_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
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
    }

    fn apply_reset(&mut self, qubit: usize) {
        let mask = 1usize << qubit;

        #[cfg(feature = "parallel")]
        let prob_zero: f64 = if self.state.len() >= MIN_STATES_FOR_PAR {
            self.state
                .par_iter()
                .filter(|(&idx, _)| idx & mask == 0)
                .map(|(_, amp)| amp.norm_sqr())
                .sum()
        } else {
            self.state
                .iter()
                .filter(|(&idx, _)| idx & mask == 0)
                .map(|(_, amp)| amp.norm_sqr())
                .sum()
        };

        #[cfg(not(feature = "parallel"))]
        let prob_zero: f64 = self
            .state
            .iter()
            .filter(|(&idx, _)| idx & mask == 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        if prob_zero > 0.0 {
            let inv_norm = 1.0 / prob_zero.sqrt();
            self.state.retain(|&idx, amp| {
                if idx & mask == 0 {
                    *amp *= inv_norm;
                    true
                } else {
                    false
                }
            });
        } else {
            self.state.clear();
            self.state.insert(0, Complex64::new(1.0, 0.0));
        }
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let mask = 1usize << qubit;

        #[cfg(feature = "parallel")]
        let prob_one: f64 = if self.state.len() >= MIN_STATES_FOR_PAR {
            self.state
                .par_iter()
                .filter(|(&idx, _)| idx & mask != 0)
                .map(|(_, amp)| amp.norm_sqr())
                .sum()
        } else {
            self.state
                .iter()
                .filter(|(&idx, _)| idx & mask != 0)
                .map(|(_, amp)| amp.norm_sqr())
                .sum()
        };

        #[cfg(not(feature = "parallel"))]
        let prob_one: f64 = self
            .state
            .iter()
            .filter(|(&idx, _)| idx & mask != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

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

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Rzz(theta) => {
                let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
                let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
                let q0 = targets[0];
                let q1 = targets[1];
                for (idx, amp) in self.state.iter_mut() {
                    let parity = ((*idx >> q0) ^ (*idx >> q1)) & 1;
                    *amp *= if parity == 0 { phase_same } else { phase_diff };
                }
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
                    self.apply_cu(targets[0], targets[1], **mat);
                }
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                if let Some(phase) = gate.controlled_phase() {
                    self.apply_mcu_phase(&targets[..num_ctrl], targets[num_ctrl], phase);
                } else {
                    self.apply_mcu(&targets[..num_ctrl], targets[num_ctrl], data.mat);
                }
            }
            Gate::BatchPhase(data) => {
                self.apply_batch_phase(targets[0], &data.phases);
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
                    let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
                    for (idx, amp) in self.state.iter_mut() {
                        let parity = ((*idx >> q0) ^ (*idx >> q1)) & 1;
                        *amp *= if parity == 0 { phase_same } else { phase_diff };
                    }
                }
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    match entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            let mask = 1usize << qubit;
                            for (idx, amp) in self.state.iter_mut() {
                                if (*idx & mask) != 0 {
                                    *amp *= d1;
                                } else {
                                    *amp *= d0;
                                }
                            }
                        }
                        DiagEntry::Phase2q { q0, q1, phase } => {
                            let mask = (1usize << q0) | (1usize << q1);
                            for (idx, amp) in self.state.iter_mut() {
                                if (*idx & mask) == mask {
                                    *amp *= phase;
                                }
                            }
                        }
                        DiagEntry::Parity2q { q0, q1, same, diff } => {
                            for (idx, amp) in self.state.iter_mut() {
                                let parity = ((*idx >> q0) ^ (*idx >> q1)) & 1;
                                *amp *= if parity == 0 { *same } else { *diff };
                            }
                        }
                    }
                }
            }
            Gate::MultiFused(data) => {
                for &(target, mat) in &data.gates {
                    self.apply_single_qubit(target, mat);
                }
            }
            Gate::Fused2q(mat) => {
                self.apply_fused_2q(targets[0], targets[1], mat);
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_fused_2q(q0, q1, mat);
                }
            }
            other => {
                debug_assert!(
                    targets.len() == 1,
                    "sparse dispatch_gate: unexpected multi-qubit gate {:?}",
                    other
                );
                let mat = other.matrix_2x2();
                self.apply_single_qubit(targets[0], mat);
            }
        }
    }
}

impl Backend for SparseBackend {
    fn name(&self) -> &'static str {
        "sparse"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.state.clear();
        self.state.insert(0, Complex64::new(1.0, 0.0));
        self.classical_bits = vec![false; num_classical_bits];
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets),
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
                    self.dispatch_gate(gate, targets);
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit);
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
        let dim = 1usize << self.num_qubits;
        let mut probs = vec![0.0f64; dim];
        for (&idx, amp) in &self.state {
            probs[idx] = amp.norm_sqr();
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
        let dim = 1usize << self.num_qubits;
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];
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
