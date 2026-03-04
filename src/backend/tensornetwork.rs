//! Tensor-network simulation backend.
//!
//! Represents the quantum state as a network of tensors. Gate application
//! appends gate tensors to the network (deferred contraction). Contraction
//! happens lazily when `probabilities()` or measurement is requested.
//!
//! # When to prefer this backend
//!
//! - Circuits with low treewidth (shallow or geometrically local).
//! - Circuits where full statevector is infeasible (>30 qubits) but structure
//!   permits efficient contraction.
//!
//! # Contraction strategy
//!
//! Greedy min-size heuristic: repeatedly contract the pair of tensors whose
//! result has the smallest total element count. O(T²) where T = tensor count.

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::{Backend, MAX_PROB_QUBITS, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type LegId = usize;
#[cfg(feature = "parallel")]
const MIN_PAR_ELEMS: usize = 4096;

/// Dense multidimensional tensor with named legs for contraction.
///
/// Legs with matching `LegId` across two tensors are contracted (summed over)
/// when those tensors are pairwise contracted.
#[derive(Clone, Debug)]
struct Tensor {
    data: Vec<Complex64>,
    shape: SmallVec<[usize; 6]>,
    legs: SmallVec<[LegId; 6]>,
}

impl Tensor {
    fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// Transpose a tensor by permuting its axes.
///
/// `perm[new_axis] = old_axis`. The output tensor has shape
/// `[input.shape[perm[0]], input.shape[perm[1]], ...]`.
fn transpose(t: &Tensor, perm: &[usize]) -> Tensor {
    let rank = t.rank();
    debug_assert_eq!(perm.len(), rank);

    let new_shape: SmallVec<[usize; 6]> = perm.iter().map(|&p| t.shape[p]).collect();
    let new_legs: SmallVec<[LegId; 6]> = perm.iter().map(|&p| t.legs[p]).collect();

    let total = t.num_elements();
    let mut new_data = vec![Complex64::new(0.0, 0.0); total];

    let mut old_strides: SmallVec<[usize; 6]> = SmallVec::new();
    let mut stride = 1usize;
    for _ in 0..rank {
        old_strides.push(0);
    }
    for i in (0..rank).rev() {
        old_strides[i] = stride;
        stride *= t.shape[i];
    }

    let mut new_strides: SmallVec<[usize; 6]> = SmallVec::new();
    stride = 1;
    for _ in 0..rank {
        new_strides.push(0);
    }
    for i in (0..rank).rev() {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    // Permuted strides: for each old axis, what stride does it contribute in the new layout?
    let perm_strides: SmallVec<[usize; 6]> = (0..rank)
        .map(|old_ax| {
            let new_ax = perm.iter().position(|&p| p == old_ax).unwrap();
            new_strides[new_ax]
        })
        .collect();

    #[cfg(feature = "parallel")]
    if total >= MIN_PAR_ELEMS {
        let old_strides_ref = &old_strides;
        let perm_strides_ref = &perm_strides;
        let src = &t.data;
        new_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(new_linear, out)| {
                let mut old_linear = 0usize;
                let mut rem = new_linear;
                for a in 0..rank {
                    let idx = rem / new_strides[a];
                    rem %= new_strides[a];
                    let old_ax = perm[a];
                    old_linear += idx * old_strides_ref[old_ax];
                }
                let _ = perm_strides_ref;
                *out = src[old_linear];
            });
    } else {
        for old_linear in 0..total {
            let mut new_linear = 0usize;
            let mut rem = old_linear;
            for i in 0..rank {
                let idx = rem / old_strides[i];
                rem %= old_strides[i];
                new_linear += idx * perm_strides[i];
            }
            new_data[new_linear] = t.data[old_linear];
        }
    }

    #[cfg(not(feature = "parallel"))]
    for old_linear in 0..total {
        let mut new_linear = 0usize;
        let mut rem = old_linear;
        for i in 0..rank {
            let idx = rem / old_strides[i];
            rem %= old_strides[i];
            new_linear += idx * perm_strides[i];
        }
        new_data[new_linear] = t.data[old_linear];
    }

    Tensor {
        data: new_data,
        shape: new_shape,
        legs: new_legs,
    }
}

/// Contract two tensors over shared legs (matching LegId).
///
/// Standard tensordot: find shared legs, reshape both to 2D matrices,
/// multiply, reshape result.
fn contract(a: &Tensor, b: &Tensor) -> Tensor {
    let mut a_shared: SmallVec<[usize; 4]> = SmallVec::new();
    let mut b_shared: SmallVec<[usize; 4]> = SmallVec::new();
    for (ai, &a_leg) in a.legs.iter().enumerate() {
        for (bi, &b_leg) in b.legs.iter().enumerate() {
            if a_leg == b_leg {
                a_shared.push(ai);
                b_shared.push(bi);
            }
        }
    }

    let a_free: SmallVec<[usize; 6]> = (0..a.rank()).filter(|i| !a_shared.contains(i)).collect();
    let b_free: SmallVec<[usize; 6]> = (0..b.rank()).filter(|i| !b_shared.contains(i)).collect();

    let mut a_perm: SmallVec<[usize; 6]> = SmallVec::new();
    a_perm.extend_from_slice(&a_free);
    a_perm.extend_from_slice(&a_shared);

    let mut b_perm: SmallVec<[usize; 6]> = SmallVec::new();
    b_perm.extend_from_slice(&b_shared);
    b_perm.extend_from_slice(&b_free);

    let a_t = if a_perm.iter().enumerate().all(|(i, &p)| i == p) {
        a.clone()
    } else {
        transpose(a, &a_perm)
    };

    let b_t = if b_perm.iter().enumerate().all(|(i, &p)| i == p) {
        b.clone()
    } else {
        transpose(b, &b_perm)
    };

    let m: usize = a_free.iter().map(|&i| a.shape[i]).product::<usize>().max(1);
    let k: usize = a_shared
        .iter()
        .map(|&i| a.shape[i])
        .product::<usize>()
        .max(1);
    let n: usize = b_free.iter().map(|&i| b.shape[i]).product::<usize>().max(1);

    let zero = Complex64::new(0.0, 0.0);
    let mut c_data = vec![zero; m * n];

    #[cfg(feature = "parallel")]
    if m * n >= MIN_PAR_ELEMS {
        let a_data = &a_t.data;
        let b_data = &b_t.data;
        c_data.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            for j in 0..k {
                let a_val = a_data[i * k + j];
                if a_val == zero {
                    continue;
                }
                let b_row = &b_data[j * n..(j + 1) * n];
                for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                    *c_elem += a_val * b_val;
                }
            }
        });
    } else {
        for i in 0..m {
            for j in 0..k {
                let a_val = a_t.data[i * k + j];
                if a_val == zero {
                    continue;
                }
                let b_row = &b_t.data[j * n..(j + 1) * n];
                let c_row = &mut c_data[i * n..(i + 1) * n];
                for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                    *c_elem += a_val * b_val;
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    for i in 0..m {
        for j in 0..k {
            let a_val = a_t.data[i * k + j];
            if a_val == zero {
                continue;
            }
            let b_row = &b_t.data[j * n..(j + 1) * n];
            let c_row = &mut c_data[i * n..(i + 1) * n];
            for (c_elem, &b_val) in c_row.iter_mut().zip(b_row) {
                *c_elem += a_val * b_val;
            }
        }
    }

    let mut result_shape: SmallVec<[usize; 6]> = SmallVec::new();
    let mut result_legs: SmallVec<[LegId; 6]> = SmallVec::new();
    for &i in &a_free {
        result_shape.push(a.shape[i]);
        result_legs.push(a.legs[i]);
    }
    for &i in &b_free {
        result_shape.push(b.shape[i]);
        result_legs.push(b.legs[i]);
    }

    if result_shape.is_empty() {
        result_shape.push(1);
    }

    Tensor {
        data: c_data,
        shape: result_shape,
        legs: result_legs,
    }
}

/// Find the number of shared legs between two tensors.
fn shared_leg_count(a: &Tensor, b: &Tensor) -> usize {
    let mut count = 0;
    for &a_leg in &a.legs {
        for &b_leg in &b.legs {
            if a_leg == b_leg {
                count += 1;
            }
        }
    }
    count
}

/// Compute the result size if two tensors were contracted.
fn contraction_result_size(a: &Tensor, b: &Tensor) -> usize {
    let mut a_free_size = 1usize;
    let mut b_free_size = 1usize;
    for (ai, &a_leg) in a.legs.iter().enumerate() {
        let shared = b.legs.contains(&a_leg);
        if !shared {
            a_free_size *= a.shape[ai];
        }
    }
    for (bi, &b_leg) in b.legs.iter().enumerate() {
        let shared = a.legs.contains(&b_leg);
        if !shared {
            b_free_size *= b.shape[bi];
        }
    }
    a_free_size * b_free_size
}

/// Contract an entire tensor network using a greedy min-size heuristic.
///
/// Repeatedly finds the pair with the smallest contraction result and
/// contracts them, until a single tensor remains.
fn greedy_contract(tensors: &mut Vec<Tensor>) -> Tensor {
    debug_assert!(!tensors.is_empty());

    while tensors.len() > 1 {
        let len = tensors.len();

        #[cfg(feature = "parallel")]
        let (best_i, best_j) = if len >= 50 {
            let t_ref: &[Tensor] = tensors;
            let (_, bi, bj) = (0..len)
                .into_par_iter()
                .flat_map_iter(|i| {
                    (i + 1..len).map(move |j| {
                        let shared = shared_leg_count(&t_ref[i], &t_ref[j]);
                        let cost = contraction_result_size(&t_ref[i], &t_ref[j]);
                        let priority = if shared > 0 { 0usize } else { 1 };
                        ((priority, cost), i, j)
                    })
                })
                .min_by_key(|&(key, _, _)| key)
                .unwrap();
            (bi, bj)
        } else {
            find_best_pair(tensors, len)
        };

        #[cfg(not(feature = "parallel"))]
        let (best_i, best_j) = find_best_pair(tensors, len);

        let b_tensor = tensors.swap_remove(best_j);
        let a_tensor = tensors.swap_remove(best_i);
        let result = contract(&a_tensor, &b_tensor);
        tensors.push(result);
    }

    tensors.pop().unwrap()
}

fn find_best_pair(tensors: &[Tensor], len: usize) -> (usize, usize) {
    let mut best_i = 0;
    let mut best_j = 1;
    let mut best_cost = usize::MAX;
    let mut found_shared = false;

    for i in 0..len {
        for j in (i + 1)..len {
            let shared = shared_leg_count(&tensors[i], &tensors[j]);
            if shared > 0 {
                let cost = contraction_result_size(&tensors[i], &tensors[j]);
                if !found_shared || cost < best_cost {
                    best_i = i;
                    best_j = j;
                    best_cost = cost;
                    found_shared = true;
                }
            } else if !found_shared {
                let cost = contraction_result_size(&tensors[i], &tensors[j]);
                if cost < best_cost {
                    best_i = i;
                    best_j = j;
                    best_cost = cost;
                }
            }
        }
    }
    (best_i, best_j)
}

/// Tensor-network simulation backend with deferred contraction.
pub struct TensorNetworkBackend {
    num_qubits: usize,
    tensors: Vec<Tensor>,
    output_legs: Vec<LegId>,
    next_leg: usize,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
}

impl TensorNetworkBackend {
    /// Create a new tensor-network backend with the given RNG seed.
    pub fn new(seed: u64) -> Self {
        Self {
            num_qubits: 0,
            tensors: Vec::new(),
            output_legs: Vec::new(),
            next_leg: 0,
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn fresh_leg(&mut self) -> LegId {
        let id = self.next_leg;
        self.next_leg += 1;
        id
    }

    fn apply_1q_matrix(&mut self, target: usize, mat: &[[Complex64; 2]; 2]) {
        let in_leg = self.output_legs[target];
        let out_leg = self.fresh_leg();

        // Rank-2 tensor: shape [2, 2], legs [out, in]
        // data[out_idx * 2 + in_idx] = mat[out_idx][in_idx]
        let data = vec![mat[0][0], mat[0][1], mat[1][0], mat[1][1]];
        self.tensors.push(Tensor {
            data,
            shape: smallvec::smallvec![2, 2],
            legs: smallvec::smallvec![out_leg, in_leg],
        });

        self.output_legs[target] = out_leg;
    }

    fn apply_2q_matrix(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let in0 = self.output_legs[q0];
        let in1 = self.output_legs[q1];
        let out0 = self.fresh_leg();
        let out1 = self.fresh_leg();

        // Rank-4 tensor: shape [2, 2, 2, 2], legs [out0, out1, in0, in1]
        // Index: mat[i0*2 + i1][j0*2 + j1] → data[out0 * 8 + out1 * 4 + in0 * 2 + in1]
        let mut data = vec![Complex64::new(0.0, 0.0); 16];
        for i0 in 0..2usize {
            for i1 in 0..2usize {
                for j0 in 0..2usize {
                    for j1 in 0..2usize {
                        data[i0 * 8 + i1 * 4 + j0 * 2 + j1] = mat[i0 * 2 + i1][j0 * 2 + j1];
                    }
                }
            }
        }

        self.tensors.push(Tensor {
            data,
            shape: SmallVec::from_slice(&[2, 2, 2, 2]),
            legs: SmallVec::from_slice(&[out0, out1, in0, in1]),
        });

        self.output_legs[q0] = out0;
        self.output_legs[q1] = out1;
    }

    /// Build the full 2^m × 2^m matrix for an MCU gate.
    fn mcu_full_matrix(num_controls: usize, mat: &[[Complex64; 2]; 2]) -> Vec<Vec<Complex64>> {
        let m = num_controls + 1;
        let dim = 1usize << m;
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let mut full = vec![vec![zero; dim]; dim];
        for (i, row) in full.iter_mut().enumerate().take(dim - 2) {
            row[i] = one;
        }
        full[dim - 2][dim - 2] = mat[0][0];
        full[dim - 2][dim - 1] = mat[0][1];
        full[dim - 1][dim - 2] = mat[1][0];
        full[dim - 1][dim - 1] = mat[1][1];
        full
    }

    fn apply_nq_matrix(&mut self, qubits: &[usize], full_mat: &[Vec<Complex64>]) {
        let m = qubits.len();
        let dim = 1usize << m;

        let in_legs: SmallVec<[LegId; 6]> = qubits.iter().map(|&q| self.output_legs[q]).collect();
        let out_legs: SmallVec<[LegId; 6]> = (0..m).map(|_| self.fresh_leg()).collect();

        // Rank-2m tensor: shape [2]^(2m), legs [out0..outm, in0..inm]
        let total = dim * dim;
        let mut data = vec![Complex64::new(0.0, 0.0); total];

        for (out_idx, row) in full_mat.iter().enumerate() {
            for (in_idx, &val) in row.iter().enumerate() {
                let mut flat = 0usize;
                for bit in 0..m {
                    let out_bit = (out_idx >> (m - 1 - bit)) & 1;
                    flat = flat * 2 + out_bit;
                }
                for bit in 0..m {
                    let in_bit = (in_idx >> (m - 1 - bit)) & 1;
                    flat = flat * 2 + in_bit;
                }
                data[flat] = val;
            }
        }

        let mut shape: SmallVec<[usize; 6]> = SmallVec::new();
        let mut legs: SmallVec<[LegId; 6]> = SmallVec::new();
        for i in 0..m {
            shape.push(2);
            legs.push(out_legs[i]);
        }
        for i in 0..m {
            shape.push(2);
            legs.push(in_legs[i]);
        }

        self.tensors.push(Tensor { data, shape, legs });

        for (i, &q) in qubits.iter().enumerate() {
            self.output_legs[q] = out_legs[i];
        }
    }

    /// Contract the full network and return the amplitude vector in
    /// computational basis order.
    fn contract_to_statevector(&self) -> Result<Vec<Complex64>> {
        if self.num_qubits > MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "contraction for {} qubits (max {})",
                    self.num_qubits, MAX_PROB_QUBITS
                ),
            });
        }

        let mut tensors = self.tensors.clone();
        let result = greedy_contract(&mut tensors);

        // The result tensor's legs should be exactly the output_legs.
        // PRISM-Q convention: q[0] = LSB of state index. In row-major
        // tensor layout, the last axis is LSB. So we want leg order
        // [q_{n-1}, q_{n-2}, ..., q_0] — reversed.
        let target_order: Vec<LegId> = self.output_legs.iter().rev().copied().collect();
        let perm: SmallVec<[usize; 6]> = target_order
            .iter()
            .map(|target_leg| {
                result
                    .legs
                    .iter()
                    .position(|l| l == target_leg)
                    .unwrap_or(0)
            })
            .collect();

        let needs_perm = perm.iter().enumerate().any(|(i, &p)| i != p);
        let ordered = if needs_perm {
            transpose(&result, &perm)
        } else {
            result
        };

        Ok(ordered.data)
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap => {
                self.apply_2q_matrix(targets[0], targets[1], &gate.matrix_4x4());
            }
            Gate::Cu(mat) => {
                let m4 = Gate::Cu(mat.clone()).matrix_4x4();
                self.apply_2q_matrix(targets[0], targets[1], &m4);
            }
            Gate::Fused2q(mat) => {
                self.apply_2q_matrix(targets[0], targets[1], mat);
            }
            Gate::Mcu(data) => {
                let qubits: Vec<usize> = targets.to_vec();
                let full = Self::mcu_full_matrix(data.num_controls as usize, &data.mat);
                self.apply_nq_matrix(&qubits, &full);
            }
            Gate::BatchPhase(data) => {
                let control = targets[0];
                let one = Complex64::new(1.0, 0.0);
                let zero = Complex64::new(0.0, 0.0);
                for &(target_qubit, phase) in &data.phases {
                    let mat = [[one, zero], [zero, phase]];
                    let m4 = Gate::Cu(Box::new(mat)).matrix_4x4();
                    self.apply_2q_matrix(control, target_qubit, &m4);
                }
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    let m4 = Gate::Rzz(theta).matrix_4x4();
                    self.apply_2q_matrix(q0, q1, &m4);
                }
            }
            Gate::MultiFused(data) => {
                for &(target, ref mat) in &data.gates {
                    self.apply_1q_matrix(target, mat);
                }
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_2q_matrix(q0, q1, mat);
                }
            }
            _ => {
                let mat = gate.matrix_2x2();
                self.apply_1q_matrix(targets[0], &mat);
            }
        }
    }
}

impl Backend for TensorNetworkBackend {
    fn name(&self) -> &'static str {
        "tensornetwork"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.tensors = Vec::new();
        self.next_leg = 0;
        self.classical_bits = vec![false; num_classical_bits];

        self.output_legs = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let leg = self.fresh_leg();
            self.output_legs.push(leg);
            self.tensors.push(Tensor {
                data: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                shape: SmallVec::from_buf_and_len([2, 0, 0, 0, 0, 0], 1),
                legs: SmallVec::from_buf_and_len([leg, 0, 0, 0, 0, 0], 1),
            });
        }

        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets),
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                use rand::Rng;

                let amplitudes = self.contract_to_statevector()?;
                let n = self.num_qubits;

                let mut prob_one = 0.0f64;
                for (idx, amp) in amplitudes.iter().enumerate() {
                    if (idx >> (n - 1 - qubit)) & 1 == 1 {
                        prob_one += amp.norm_sqr();
                    }
                }

                let outcome = self.rng.gen::<f64>() < prob_one;
                self.classical_bits[*classical_bit] = outcome;

                let mut collapsed = amplitudes;
                let mut norm_sq = 0.0f64;
                for (idx, amp) in collapsed.iter_mut().enumerate() {
                    let bit = (idx >> (n - 1 - qubit)) & 1 == 1;
                    if bit != outcome {
                        *amp = Complex64::new(0.0, 0.0);
                    } else {
                        norm_sq += amp.norm_sqr();
                    }
                }
                let norm = norm_sq.clamp(NORM_CLAMP_MIN, 1.0).sqrt();
                for amp in &mut collapsed {
                    *amp /= norm;
                }

                self.tensors.clear();
                self.next_leg = 0;
                self.output_legs.clear();
                let mut shape: SmallVec<[usize; 6]> = SmallVec::new();
                let mut legs: SmallVec<[LegId; 6]> = SmallVec::new();
                for _ in 0..n {
                    let leg = self.fresh_leg();
                    self.output_legs.push(leg);
                    shape.push(2);
                    legs.push(leg);
                }
                self.tensors.push(Tensor {
                    data: collapsed,
                    shape,
                    legs,
                });
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

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let amplitudes = self.contract_to_statevector()?;
        #[cfg(feature = "parallel")]
        if amplitudes.len() >= MIN_PAR_ELEMS {
            return Ok(amplitudes.par_iter().map(|a| a.norm_sqr()).collect());
        }
        Ok(amplitudes.iter().map(|a| a.norm_sqr()).collect())
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_fused_gates(&self) -> bool {
        true
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        self.contract_to_statevector()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::statevector::StatevectorBackend;
    use crate::backend::Backend;
    use crate::circuit::Circuit;
    use crate::gates::{Gate, MultiFusedData};

    const EPS: f64 = 1e-10;

    fn assert_probs_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < EPS,
                "prob[{i}]: TN={x}, expected={y}, diff={}",
                (x - y).abs()
            );
        }
    }

    #[test]
    fn test_init_zero_state() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(3, 0).unwrap();
        let probs = tn.probabilities().unwrap();
        assert_eq!(probs.len(), 8);
        assert!((probs[0] - 1.0).abs() < EPS);
        for &p in &probs[1..] {
            assert!(p.abs() < EPS);
        }
    }

    #[test]
    fn test_single_qubit_h() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(1, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        let probs = tn.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < EPS);
        assert!((probs[1] - 0.5).abs() < EPS);
    }

    #[test]
    fn test_single_qubit_x() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(1, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        let probs = tn.probabilities().unwrap();
        assert!(probs[0].abs() < EPS);
        assert!((probs[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_two_qubit_cx_bell() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(2, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec::smallvec![0, 1],
        })
        .unwrap();
        let probs = tn.probabilities().unwrap();
        assert!((probs[0] - 0.5).abs() < EPS);
        assert!(probs[1].abs() < EPS);
        assert!(probs[2].abs() < EPS);
        assert!((probs[3] - 0.5).abs() < EPS);
    }

    #[test]
    fn test_parametric_rx() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(1, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::Rx(std::f64::consts::PI),
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        let probs = tn.probabilities().unwrap();
        assert!(probs[0].abs() < EPS);
        assert!((probs[1] - 1.0).abs() < EPS);
    }

    #[test]
    fn test_measure_deterministic() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(1, 1).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::X,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        tn.apply(&Instruction::Measure {
            qubit: 0,
            classical_bit: 0,
        })
        .unwrap();
        assert!(tn.classical_results()[0]);
    }

    #[test]
    fn test_measure_seeded() {
        let run = |seed| {
            let mut tn = TensorNetworkBackend::new(seed);
            tn.init(1, 1).unwrap();
            tn.apply(&Instruction::Gate {
                gate: Gate::H,
                targets: smallvec::smallvec![0],
            })
            .unwrap();
            tn.apply(&Instruction::Measure {
                qubit: 0,
                classical_bit: 0,
            })
            .unwrap();
            tn.classical_results()[0]
        };
        let r1 = run(42);
        let r2 = run(42);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_fused_gate() {
        let ht_mat = crate::gates::mat_mul_2x2(&Gate::T.matrix_2x2(), &Gate::H.matrix_2x2());
        let mut tn_fused = TensorNetworkBackend::new(42);
        tn_fused.init(1, 0).unwrap();
        tn_fused
            .apply(&Instruction::Gate {
                gate: Gate::Fused(Box::new(ht_mat)),
                targets: smallvec::smallvec![0],
            })
            .unwrap();

        let mut tn_individual = TensorNetworkBackend::new(42);
        tn_individual.init(1, 0).unwrap();
        tn_individual
            .apply(&Instruction::Gate {
                gate: Gate::H,
                targets: smallvec::smallvec![0],
            })
            .unwrap();
        tn_individual
            .apply(&Instruction::Gate {
                gate: Gate::T,
                targets: smallvec::smallvec![0],
            })
            .unwrap();

        assert_probs_close(
            &tn_fused.probabilities().unwrap(),
            &tn_individual.probabilities().unwrap(),
        );
    }

    #[test]
    fn test_multi_fused() {
        let h_mat = Gate::H.matrix_2x2();
        let t_mat = Gate::T.matrix_2x2();
        let x_mat = Gate::X.matrix_2x2();

        let mut tn_mf = TensorNetworkBackend::new(42);
        tn_mf.init(3, 0).unwrap();
        tn_mf
            .apply(&Instruction::Gate {
                gate: Gate::MultiFused(Box::new(MultiFusedData {
                    gates: vec![(0, h_mat), (1, t_mat), (2, x_mat)],
                })),
                targets: smallvec::smallvec![0, 1, 2],
            })
            .unwrap();

        let mut tn_ind = TensorNetworkBackend::new(42);
        tn_ind.init(3, 0).unwrap();
        tn_ind
            .apply(&Instruction::Gate {
                gate: Gate::H,
                targets: smallvec::smallvec![0],
            })
            .unwrap();
        tn_ind
            .apply(&Instruction::Gate {
                gate: Gate::T,
                targets: smallvec::smallvec![1],
            })
            .unwrap();
        tn_ind
            .apply(&Instruction::Gate {
                gate: Gate::X,
                targets: smallvec::smallvec![2],
            })
            .unwrap();

        assert_probs_close(
            &tn_mf.probabilities().unwrap(),
            &tn_ind.probabilities().unwrap(),
        );
    }

    #[test]
    fn test_golden_vs_statevector() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Ry(0.7), &[2]);
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::Rx(1.2), &[3]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::H, &[3]);

        let mut sv = StatevectorBackend::new(42);
        sv.init(4, 0).unwrap();
        for inst in &c.instructions {
            sv.apply(inst).unwrap();
        }
        let sv_probs = sv.probabilities().unwrap();

        let mut tn = TensorNetworkBackend::new(42);
        tn.init(4, 0).unwrap();
        for inst in &c.instructions {
            tn.apply(inst).unwrap();
        }
        let tn_probs = tn.probabilities().unwrap();

        assert_probs_close(&tn_probs, &sv_probs);
    }

    #[test]
    fn test_export_statevector() {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(2, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec::smallvec![0, 1],
        })
        .unwrap();

        let sv = tn.export_statevector().unwrap();
        assert_eq!(sv.len(), 4);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv[0].re - h).abs() < EPS);
        assert!(sv[1].norm() < EPS);
        assert!(sv[2].norm() < EPS);
        assert!((sv[3].re - h).abs() < EPS);
    }

    #[test]
    fn test_cu_gate() {
        let rz_mat = Gate::Rz(0.5).matrix_2x2();

        let mut tn = TensorNetworkBackend::new(42);
        tn.init(2, 0).unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        tn.apply(&Instruction::Gate {
            gate: Gate::Cu(Box::new(rz_mat)),
            targets: smallvec::smallvec![0, 1],
        })
        .unwrap();

        let mut sv = StatevectorBackend::new(42);
        sv.init(2, 0).unwrap();
        sv.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: smallvec::smallvec![0],
        })
        .unwrap();
        sv.apply(&Instruction::Gate {
            gate: Gate::Cu(Box::new(rz_mat)),
            targets: smallvec::smallvec![0, 1],
        })
        .unwrap();

        assert_probs_close(&tn.probabilities().unwrap(), &sv.probabilities().unwrap());
    }
}
