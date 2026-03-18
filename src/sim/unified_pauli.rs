use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::compiled::{flip_bit, propagate_backward, PauliVec};

const SQRT_2: f64 = std::f64::consts::SQRT_2;
const MIN_CLIFF_GATES_FOR_COALESCE: usize = 16;

// ---------------------------------------------------------------------------
// Clifford Map: pre-compiled symplectic transformation over GF(2)
// ---------------------------------------------------------------------------

struct CliffordMap {
    num_words: usize,
    col_xx: Vec<u64>,
    col_xz: Vec<u64>,
    col_zx: Vec<u64>,
    col_zz: Vec<u64>,
}

impl CliffordMap {
    fn identity(n: usize) -> Self {
        let num_words = n.div_ceil(64);
        let mut col_xx = vec![0u64; n * num_words];
        let mut col_zz = vec![0u64; n * num_words];
        for q in 0..n {
            col_xx[q * num_words + q / 64] = 1u64 << (q % 64);
            col_zz[q * num_words + q / 64] = 1u64 << (q % 64);
        }
        Self {
            num_words,
            col_xx,
            col_xz: vec![0u64; n * num_words],
            col_zx: vec![0u64; n * num_words],
            col_zz,
        }
    }

    #[inline(always)]
    fn col_range(&self, q: usize) -> std::ops::Range<usize> {
        q * self.num_words..(q + 1) * self.num_words
    }

    fn apply_h(&mut self, q: usize) {
        let r = self.col_range(q);
        for i in r {
            std::mem::swap(&mut self.col_xx[i], &mut self.col_xz[i]);
            std::mem::swap(&mut self.col_zx[i], &mut self.col_zz[i]);
        }
    }

    fn apply_s(&mut self, q: usize) {
        let r = self.col_range(q);
        for i in r {
            self.col_xz[i] ^= self.col_xx[i];
            self.col_zz[i] ^= self.col_zx[i];
        }
    }

    fn apply_sx(&mut self, q: usize) {
        let r = self.col_range(q);
        for i in r {
            self.col_xx[i] ^= self.col_xz[i];
            self.col_zx[i] ^= self.col_zz[i];
        }
    }

    fn apply_cx(&mut self, ctrl: usize, tgt: usize) {
        let nw = self.num_words;
        for w in 0..nw {
            self.col_xx[tgt * nw + w] ^= self.col_xx[ctrl * nw + w];
            self.col_xz[tgt * nw + w] ^= self.col_xz[ctrl * nw + w];
            self.col_zx[ctrl * nw + w] ^= self.col_zx[tgt * nw + w];
            self.col_zz[ctrl * nw + w] ^= self.col_zz[tgt * nw + w];
        }
    }

    fn apply_cz(&mut self, q0: usize, q1: usize) {
        let nw = self.num_words;
        for w in 0..nw {
            self.col_xz[q0 * nw + w] ^= self.col_xx[q1 * nw + w];
            self.col_xz[q1 * nw + w] ^= self.col_xx[q0 * nw + w];
            self.col_zz[q0 * nw + w] ^= self.col_zx[q1 * nw + w];
            self.col_zz[q1 * nw + w] ^= self.col_zx[q0 * nw + w];
        }
    }

    fn apply_swap(&mut self, q0: usize, q1: usize) {
        let nw = self.num_words;
        for w in 0..nw {
            let r0 = q0 * nw + w;
            let r1 = q1 * nw + w;
            self.col_xx.swap(r0, r1);
            self.col_xz.swap(r0, r1);
            self.col_zx.swap(r0, r1);
            self.col_zz.swap(r0, r1);
        }
    }

    fn apply_gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::H => self.apply_h(targets[0]),
            Gate::S | Gate::Sdg => self.apply_s(targets[0]),
            Gate::SX | Gate::SXdg => self.apply_sx(targets[0]),
            Gate::X | Gate::Y | Gate::Z | Gate::Id => {}
            Gate::Cx => self.apply_cx(targets[0], targets[1]),
            Gate::Cz => self.apply_cz(targets[0], targets[1]),
            Gate::Swap => self.apply_swap(targets[0], targets[1]),
            _ => {}
        }
    }

    fn apply(&self, pauli: &mut PauliVec) {
        let nw = self.num_words;

        let mut scratch = [0u64; 32];
        if nw <= 16 {
            scratch[..nw].copy_from_slice(&pauli.x);
            scratch[16..16 + nw].copy_from_slice(&pauli.z);
            pauli.x.fill(0);
            pauli.z.fill(0);

            for word in 0..nw {
                let mut xw = scratch[word];
                while xw != 0 {
                    let bit = xw.trailing_zeros() as usize;
                    let q = word * 64 + bit;
                    let base = q * nw;
                    for w in 0..nw {
                        pauli.x[w] ^= self.col_xx[base + w];
                        pauli.z[w] ^= self.col_xz[base + w];
                    }
                    xw &= xw - 1;
                }

                let mut zw = scratch[16 + word];
                while zw != 0 {
                    let bit = zw.trailing_zeros() as usize;
                    let q = word * 64 + bit;
                    let base = q * nw;
                    for w in 0..nw {
                        pauli.x[w] ^= self.col_zx[base + w];
                        pauli.z[w] ^= self.col_zz[base + w];
                    }
                    zw &= zw - 1;
                }
            }
        } else {
            let old_x = pauli.x.clone();
            let old_z = pauli.z.clone();
            pauli.x.fill(0);
            pauli.z.fill(0);

            for word in 0..nw {
                let mut xw = old_x[word];
                while xw != 0 {
                    let bit = xw.trailing_zeros() as usize;
                    let q = word * 64 + bit;
                    let base = q * nw;
                    for w in 0..nw {
                        pauli.x[w] ^= self.col_xx[base + w];
                        pauli.z[w] ^= self.col_xz[base + w];
                    }
                    xw &= xw - 1;
                }

                let mut zw = old_z[word];
                while zw != 0 {
                    let bit = zw.trailing_zeros() as usize;
                    let q = word * 64 + bit;
                    let base = q * nw;
                    for w in 0..nw {
                        pauli.x[w] ^= self.col_zx[base + w];
                        pauli.z[w] ^= self.col_zz[base + w];
                    }
                    zw &= zw - 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Coalesced circuit representation
// ---------------------------------------------------------------------------

enum CoalescedOp {
    Map(CliffordMap),
    SmallCliff(Vec<(Gate, SmallVec<[usize; 4]>)>),
    T { qubit: usize, is_dagger: bool },
}

fn coalesce_cliffords(circuit: &Circuit) -> Vec<CoalescedOp> {
    let n = circuit.num_qubits;
    let mut ops = Vec::new();
    let mut cliff_buf: Vec<(Gate, SmallVec<[usize; 4]>)> = Vec::new();

    for inst in &circuit.instructions {
        if let Instruction::Gate { gate, targets } = inst {
            match gate {
                Gate::T => {
                    flush_cliff_buf(n, &mut cliff_buf, &mut ops);
                    ops.push(CoalescedOp::T {
                        qubit: targets[0],
                        is_dagger: false,
                    });
                }
                Gate::Tdg => {
                    flush_cliff_buf(n, &mut cliff_buf, &mut ops);
                    ops.push(CoalescedOp::T {
                        qubit: targets[0],
                        is_dagger: true,
                    });
                }
                _ => {
                    cliff_buf.push((gate.clone(), SmallVec::from_slice(targets)));
                }
            }
        } else {
            flush_cliff_buf(n, &mut cliff_buf, &mut ops);
        }
    }
    flush_cliff_buf(n, &mut cliff_buf, &mut ops);
    ops
}

fn flush_cliff_buf(
    n: usize,
    buf: &mut Vec<(Gate, SmallVec<[usize; 4]>)>,
    ops: &mut Vec<CoalescedOp>,
) {
    if buf.is_empty() {
        return;
    }
    if buf.len() < MIN_CLIFF_GATES_FOR_COALESCE {
        ops.push(CoalescedOp::SmallCliff(std::mem::take(buf)));
    } else {
        let mut map = CliffordMap::identity(n);
        for (gate, targets) in buf.drain(..) {
            map.apply_gate(&gate, &targets);
        }
        ops.push(CoalescedOp::Map(map));
    }
}

// ---------------------------------------------------------------------------
// SPP (Stochastic Pauli Propagation)
// ---------------------------------------------------------------------------

#[inline(always)]
fn branch_t_gate(
    pauli: &mut PauliVec,
    qubit: usize,
    is_dagger: bool,
    rng: &mut impl Rng,
) -> Complex64 {
    if !pauli.has_x_or_y(qubit) {
        return Complex64::new(1.0, 0.0);
    }

    let has_z = (pauli.z[qubit / 64] >> (qubit % 64)) & 1 != 0;
    let is_y = has_z;

    let keep = rng.gen_bool(0.5);
    if !keep {
        flip_bit(&mut pauli.z, qubit);
    }

    let sign = match (is_y, keep, is_dagger) {
        (false, _, false) => 1.0,
        (false, true, true) => 1.0,
        (false, false, true) => -1.0,
        (true, true, false) => 1.0,
        (true, false, false) => -1.0,
        (true, _, true) => 1.0,
    };

    Complex64::new(sign * SQRT_2, 0.0)
}

fn backward_propagate_coalesced(
    ops: &[CoalescedOp],
    observable: &PauliVec,
    rng: &mut impl Rng,
) -> (PauliVec, Complex64) {
    let mut pauli = PauliVec {
        x: observable.x.clone(),
        z: observable.z.clone(),
    };
    let mut weight = Complex64::new(1.0, 0.0);

    for op in ops.iter().rev() {
        match op {
            CoalescedOp::Map(map) => {
                map.apply(&mut pauli);
            }
            CoalescedOp::SmallCliff(gates) => {
                for (gate, targets) in gates.iter().rev() {
                    propagate_backward(&mut pauli, gate, targets);
                }
            }
            CoalescedOp::T { qubit, is_dagger } => {
                weight *= branch_t_gate(&mut pauli, *qubit, *is_dagger, rng);
            }
        }
    }

    (pauli, weight)
}

fn count_t_gates(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::T | Gate::Tdg,
                    ..
                }
            )
        })
        .count()
}

pub struct SppResult {
    pub expectations: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub num_samples: usize,
    pub t_count: usize,
    pub nonzero_fraction: f64,
}

fn estimate_qubit_expectation(
    ops: &[CoalescedOp],
    qubit: usize,
    num_words: usize,
    num_samples: usize,
    seed: u64,
) -> (f64, f64, usize) {
    let obs = PauliVec::z_on_qubit(num_words, qubit);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut nonzero = 0usize;

    for i in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            weight.re
        } else {
            0.0
        };

        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = if num_samples > 1 {
        m2 / (num_samples - 1) as f64
    } else {
        0.0
    };
    let std_error = (variance / num_samples as f64).sqrt();
    (mean, std_error, nonzero)
}

pub fn run_spp(circuit: &Circuit, num_samples: usize, seed: u64) -> Result<SppResult> {
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_t_gates(circuit);
    let ops = coalesce_cliffords(circuit);

    #[cfg(feature = "parallel")]
    let results: Vec<(f64, f64, usize)> = {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|q| {
                estimate_qubit_expectation(
                    &ops,
                    q,
                    num_words,
                    num_samples,
                    seed.wrapping_add(q as u64),
                )
            })
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(f64, f64, usize)> = (0..n)
        .map(|q| {
            estimate_qubit_expectation(&ops, q, num_words, num_samples, seed.wrapping_add(q as u64))
        })
        .collect();

    let mut expectations = Vec::with_capacity(n);
    let mut std_errors = Vec::with_capacity(n);
    let mut total_nonzero = 0usize;

    for (mean, std_error, nonzero) in &results {
        expectations.push(*mean);
        std_errors.push(*std_error);
        total_nonzero += nonzero;
    }

    let nonzero_fraction = total_nonzero as f64 / (n * num_samples) as f64;

    Ok(SppResult {
        expectations,
        std_errors,
        num_samples,
        t_count,
        nonzero_fraction,
    })
}

pub fn spp_to_probabilities(result: &SppResult) -> Vec<f64> {
    result
        .expectations
        .iter()
        .flat_map(|ez| {
            let p0 = (1.0 + ez) / 2.0;
            [p0.clamp(0.0, 1.0), (1.0 - ez).clamp(0.0, 2.0) / 2.0]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Deterministic Sparse Pauli Dynamics (SPD)
// ---------------------------------------------------------------------------

use std::collections::HashMap;

struct WeightedPauliSum {
    terms: HashMap<PauliVec, Complex64>,
}

impl WeightedPauliSum {
    fn new() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    fn insert(&mut self, pauli: PauliVec, coeff: Complex64) {
        let entry = self.terms.entry(pauli).or_insert(Complex64::new(0.0, 0.0));
        *entry += coeff;
    }

    fn conjugate_all_backward(&mut self, gate: &Gate, targets: &[usize]) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();
        for (mut pauli, coeff) in old_terms {
            propagate_backward(&mut pauli, gate, targets);
            self.insert(pauli, coeff);
        }
    }

    fn conjugate_all_map(&mut self, map: &CliffordMap) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();
        for (mut pauli, coeff) in old_terms {
            map.apply(&mut pauli);
            self.insert(pauli, coeff);
        }
    }

    fn branch_t_deterministic(&mut self, qubit: usize, is_dagger: bool) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();
        let inv_sqrt2 = 1.0 / SQRT_2;

        for (pauli, coeff) in old_terms {
            if !pauli.has_x_or_y(qubit) {
                self.insert(pauli, coeff);
                continue;
            }

            let has_z = (pauli.z[qubit / 64] >> (qubit % 64)) & 1 != 0;
            let is_y = has_z;

            let pauli_keep = pauli.clone();
            let mut pauli_flip = pauli;
            flip_bit(&mut pauli_flip.z, qubit);

            let (sign_keep, sign_flip) = match (is_y, is_dagger) {
                (false, false) => (1.0, 1.0),
                (false, true) => (1.0, -1.0),
                (true, false) => (1.0, -1.0),
                (true, true) => (1.0, 1.0),
            };

            self.insert(pauli_keep, coeff * sign_keep * inv_sqrt2);
            self.insert(pauli_flip, coeff * sign_flip * inv_sqrt2);
        }
    }

    fn truncate(&mut self, epsilon: f64) -> f64 {
        let mut discarded = 0.0;
        self.terms.retain(|_, coeff| {
            if coeff.norm() < epsilon {
                discarded += coeff.norm();
                false
            } else {
                true
            }
        });
        discarded
    }

    fn diagonal_expectation(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for (pauli, coeff) in &self.terms {
            if pauli.is_diagonal() {
                sum += coeff;
            }
        }
        sum.re
    }
}

pub struct SpdResult {
    pub expectations: Vec<f64>,
    pub t_count: usize,
    pub max_terms: usize,
    pub total_discarded: f64,
}

pub fn run_spd(circuit: &Circuit, epsilon: f64, max_terms: usize) -> Result<SpdResult> {
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_t_gates(circuit);
    let ops = coalesce_cliffords(circuit);

    let mut expectations = Vec::with_capacity(n);
    let mut peak_terms = 0usize;
    let mut total_discarded = 0.0;

    for q in 0..n {
        let mut sum = WeightedPauliSum::new();
        sum.insert(PauliVec::z_on_qubit(num_words, q), Complex64::new(1.0, 0.0));

        for op in ops.iter().rev() {
            match op {
                CoalescedOp::Map(map) => sum.conjugate_all_map(map),
                CoalescedOp::SmallCliff(gates) => {
                    for (gate, targets) in gates.iter().rev() {
                        sum.conjugate_all_backward(gate, targets);
                    }
                }
                CoalescedOp::T { qubit, is_dagger } => {
                    sum.branch_t_deterministic(*qubit, *is_dagger);
                }
            }

            if max_terms > 0 && sum.terms.len() > max_terms {
                total_discarded += sum.truncate(epsilon);
            }

            if sum.terms.len() > peak_terms {
                peak_terms = sum.terms.len();
            }
        }

        if epsilon > 0.0 {
            total_discarded += sum.truncate(epsilon);
        }

        expectations.push(sum.diagonal_expectation());
    }

    Ok(SpdResult {
        expectations,
        t_count,
        max_terms: peak_terms,
        total_discarded,
    })
}

pub fn spd_to_probabilities(result: &SpdResult) -> Vec<f64> {
    result
        .expectations
        .iter()
        .flat_map(|ez| {
            let p0 = (1.0 + ez) / 2.0;
            [p0.clamp(0.0, 1.0), (1.0 - p0).clamp(0.0, 1.0)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_no_t_gates_matches_propagate_backward() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::S, &[2]);

        let obs = PauliVec::z_on_qubit(1, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ops = coalesce_cliffords(&circuit);
        let (result, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);

        let mut expected = PauliVec::z_on_qubit(1, 1);
        for inst in circuit.instructions.iter().rev() {
            if let Instruction::Gate { gate, targets } = inst {
                propagate_backward(&mut expected, gate, targets);
            }
        }

        assert_eq!(result.x, expected.x);
        assert_eq!(result.z, expected.z);
        assert!((weight - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn test_h_t_h_expectation_converges() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let obs = PauliVec::z_on_qubit(1, 0);
        let num_samples = 100_000;
        let mut sum = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ops = coalesce_cliffords(&circuit);

        for _ in 0..num_samples {
            let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
            if pauli.is_diagonal() {
                sum += weight.re;
            }
        }
        let mean = sum / num_samples as f64;

        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (mean - exact_z).abs() < 0.02,
            "mean={mean}, expected≈{exact_z}"
        );
    }

    #[test]
    fn test_branch_t_gate_passthrough_iz() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut pauli_i = PauliVec::new(1);
        let w = branch_t_gate(&mut pauli_i, 0, false, &mut rng);
        assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);

        let mut pauli_z = PauliVec::z_on_qubit(1, 0);
        let w = branch_t_gate(&mut pauli_z, 0, false, &mut rng);
        assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn test_branch_t_gate_x_branches() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let num_samples = 10_000;
        let mut x_count = 0;
        let mut y_count = 0;

        for _ in 0..num_samples {
            let mut pauli = PauliVec::new(1);
            pauli.x[0] = 1;
            let w = branch_t_gate(&mut pauli, 0, false, &mut rng);
            assert!((w.norm() - SQRT_2).abs() < 1e-14);
            if pauli.z[0] == 0 {
                x_count += 1;
            } else {
                y_count += 1;
            }
        }

        let ratio = x_count as f64 / num_samples as f64;
        assert!(
            (ratio - 0.5).abs() < 0.03,
            "expected ~50/50, got {x_count}/{y_count}"
        );
    }

    fn marginal_p0(full_probs: &[f64], _n: usize, qubit: usize) -> f64 {
        let mut p0 = 0.0;
        for (i, &p) in full_probs.iter().enumerate() {
            if (i >> qubit) & 1 == 0 {
                p0 += p;
            }
        }
        p0
    }

    #[test]
    fn test_run_spp_vs_statevector_3q_2t() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::H, &[0]);

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        let spp = run_spp(&circuit, 200_000, 42).unwrap();
        assert_eq!(spp.t_count, 2);

        for q in 0..3 {
            let exact_p0 = marginal_p0(&sv_probs, 3, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            let err = (spp.expectations[q] - exact_ez).abs();
            assert!(
                err < 3.0 * spp.std_errors[q] + 0.01,
                "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
                spp.expectations[q],
                3.0 * spp.std_errors[q]
            );
        }
    }

    #[test]
    fn test_run_spp_vs_statevector_4q_4t() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        let spp = run_spp(&circuit, 200_000, 42).unwrap();
        assert_eq!(spp.t_count, 4);

        for q in 0..4 {
            let exact_p0 = marginal_p0(&sv_probs, 4, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            let err = (spp.expectations[q] - exact_ez).abs();
            assert!(
                err < 3.0 * spp.std_errors[q] + 0.01,
                "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
                spp.expectations[q],
                3.0 * spp.std_errors[q]
            );
        }
    }

    #[test]
    fn test_spp_to_probabilities() {
        let result = SppResult {
            expectations: vec![0.5, -0.3],
            std_errors: vec![0.01, 0.01],
            num_samples: 1000,
            t_count: 2,
            nonzero_fraction: 0.8,
        };
        let probs = spp_to_probabilities(&result);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.75).abs() < 1e-14);
        assert!((probs[1] - 0.25).abs() < 1e-14);
        assert!((probs[2] - 0.35).abs() < 1e-14);
        assert!((probs[3] - 0.65).abs() < 1e-14);
    }

    #[test]
    fn test_spd_no_t_gates_exact() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::S, &[2]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 0);
        assert_eq!(spd.max_terms, 1);

        let spp = run_spp(&circuit, 10_000, 42).unwrap();
        for q in 0..3 {
            assert!(
                (spd.expectations[q] - spp.expectations[q]).abs() < 0.05,
                "qubit {q}: spd={}, spp={}",
                spd.expectations[q],
                spp.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_h_t_h_exact() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (spd.expectations[0] - exact_z).abs() < 1e-10,
            "spd={}, exact={exact_z}",
            spd.expectations[0]
        );
    }

    #[test]
    fn test_spd_vs_statevector_3q_2t() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::H, &[0]);

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 2);

        for q in 0..3 {
            let exact_p0 = marginal_p0(&sv_probs, 3, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            assert!(
                (spd.expectations[q] - exact_ez).abs() < 1e-10,
                "qubit {q}: spd={}, exact={exact_ez}",
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_vs_statevector_4q_4t() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 4);

        for q in 0..4 {
            let exact_p0 = marginal_p0(&sv_probs, 4, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            assert!(
                (spd.expectations[q] - exact_ez).abs() < 1e-10,
                "qubit {q}: spd={}, exact={exact_ez}",
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_truncation() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let exact = run_spd(&circuit, 0.0, 0).unwrap();
        let approx = run_spd(&circuit, 1e-6, 0).unwrap();

        for q in 0..4 {
            assert!(
                (exact.expectations[q] - approx.expectations[q]).abs() < 1e-4,
                "qubit {q}: exact={}, approx={}",
                exact.expectations[q],
                approx.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_with_tdg() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Tdg, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (spd.expectations[0] - exact_z).abs() < 1e-10,
            "spd={}, exact={exact_z}",
            spd.expectations[0]
        );
    }

    #[test]
    fn test_spd_to_probabilities() {
        let result = SpdResult {
            expectations: vec![0.5, -0.3],
            t_count: 2,
            max_terms: 4,
            total_discarded: 0.0,
        };
        let probs = spd_to_probabilities(&result);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.75).abs() < 1e-14);
        assert!((probs[1] - 0.25).abs() < 1e-14);
        assert!((probs[2] - 0.35).abs() < 1e-14);
        assert!((probs[3] - 0.65).abs() < 1e-14);
    }
}
