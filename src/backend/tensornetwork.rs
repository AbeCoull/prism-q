//! Tensor-network simulation backend.
//!
//! Represents the quantum state as a network of tensors. Gate application
//! appends gate tensors to the network (deferred contraction). Contraction
//! happens lazily when `probabilities()` or measurement is requested.
//!
//! # Memory layout
//!
//! - Each tensor: contiguous `Vec<Complex64>` plus a shape vector and leg ids
//!   (up to 6 held inline).
//! - Gates append tensors, so memory grows with gate count until a query
//!   contracts the network; measurement leaves one dense rank-n tensor.
//!
//! # Gate support
//!
//! The full gate set: 1q and 2q gates as rank-2/rank-4 tensors, MCU as one
//! dense multi-qubit tensor, batched variants expanded per entry.
//! `Gate::QftBlock` is expanded to textbook gates before dispatch.
//!
//! # When to prefer this backend
//!
//! - Circuits with low treewidth (shallow or geometrically local).
//! - Circuits where full statevector is infeasible (>30 qubits) but structure
//!   permits efficient contraction.
//!
//! # When NOT to use this backend
//!
//! - High-treewidth circuits, where contraction intermediates outgrow the
//!   dense statevector.
//! - Shot- or observable-heavy workloads; both route through `probabilities()`
//!   (see below).
//!
//! # Contraction strategy
//!
//! Greedy min-size heuristic: repeatedly contract the pair of tensors whose
//! result has the smallest total element count, preferring pairs sharing a leg.
//! O(T·r·log T) for T tensors of rank at most r.
//!
//! # Observables contract natively; shots stay on the dense route
//!
//! `Backend::pauli_expectations` and `Backend::reduced_density_matrix_1q` both
//! answer by doubling the network against its conjugate: the bra copy's legs are
//! shifted clear of the ket id space, and each qubit's boundary is either closed
//! against its twin (a trace) or joined through an operator. No `2^n` vector is
//! built, so neither query passes the dense ceiling.
//!
//! An identity factor is a closed leg rather than an appended tensor, so a
//! weight-`k` observable adds `k` tensors to a network of `2T`, not `n`.
//!
//! `Backend::sample_basis_states` is not overridden, and shots continue to route
//! through `probabilities()`. Sampling one bitstring from a deferred network
//! means contracting it once per qubit to get each conditional, so a shot costs
//! `n` contractions against the single contraction the dense route pays for the
//! whole distribution. That is strictly worse whenever the statevector fits, and
//! when it does not fit the per-qubit contractions do not fit either: the cost is
//! set by treewidth, which conditioning does not reduce.
//!
//! `expectation_zero_state` remains a separate path, contracting `⟨0|U†PU|0⟩`
//! from a circuit rather than an evolved backend, and is what the QEC estimator
//! ladder uses.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::{Backend, NORM_CLAMP_MIN, dense_statevector_len, tensor_probability_len};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

type LegId = usize;

#[cfg(feature = "parallel")]
use crate::backend::MIN_PAR_ELEMS;

/// `m*k*n` at or above which a contraction goes to faer instead of the scalar
/// loop below.
///
/// A 64-cubed complex product. Routing everything to faer instead costs 23% on
/// `tn/scalar_depth_20q/4`, where the operands are too small to cover its
/// packing.
#[cfg(feature = "parallel")]
const MIN_FAER_GEMM_WORK: usize = 1 << 18;

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
        self.legs.len()
    }
}

/// Fill `out` with transposed elements, `out[0]` being output index `start`.
///
/// The output index is walked as an odometer over the permuted axes, so the
/// source offset advances by addition and only a nonzero `start` needs division.
///
/// `steps[a]` is the source stride of the axis that output axis `a` came from.
fn transpose_range(
    out: &mut [Complex64],
    src: &[Complex64],
    start: usize,
    new_shape: &[usize],
    new_strides: &[usize],
    steps: &[usize],
) {
    let rank = new_shape.len();
    let mut counter: SmallVec<[usize; 6]> = SmallVec::from_elem(0usize, rank);
    let mut src_idx = 0usize;
    if start != 0 {
        let mut rem = start;
        for a in 0..rank {
            counter[a] = rem / new_strides[a];
            rem %= new_strides[a];
            src_idx += counter[a] * steps[a];
        }
    }

    for slot in out.iter_mut() {
        *slot = src[src_idx];
        for a in (0..rank).rev() {
            counter[a] += 1;
            src_idx += steps[a];
            if counter[a] < new_shape[a] {
                break;
            }
            counter[a] = 0;
            src_idx -= steps[a] * new_shape[a];
        }
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

    let steps: SmallVec<[usize; 6]> = perm.iter().map(|&old_ax| old_strides[old_ax]).collect();

    #[cfg(feature = "parallel")]
    if total >= MIN_PAR_ELEMS {
        let src = &t.data;
        new_data
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, out)| {
                transpose_range(
                    out,
                    src,
                    chunk_idx * MIN_PAR_ELEMS,
                    &new_shape,
                    &new_strides,
                    &steps,
                );
            });

        return Tensor {
            data: new_data,
            shape: new_shape,
            legs: new_legs,
        };
    }

    transpose_range(&mut new_data, &t.data, 0, &new_shape, &new_strides, &steps);

    Tensor {
        data: new_data,
        shape: new_shape,
        legs: new_legs,
    }
}

/// Multiply the row-major `m` by `k` and `k` by `n` operands into `c`.
///
/// A row-major array is its own transpose read column-major, so `C = A*B` is
/// issued as `C^T = B^T * A^T` over the same buffers with no repacking.
#[cfg(feature = "parallel")]
fn faer_gemm(a: &[Complex64], b: &[Complex64], c: &mut [Complex64], m: usize, k: usize, n: usize) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};

    matmul(
        MatMut::from_column_major_slice_mut(c, n, m),
        Accum::Replace,
        MatRef::from_column_major_slice(b, n, k),
        MatRef::from_column_major_slice(a, k, m),
        Complex64::new(1.0, 0.0),
        Par::rayon(0),
    );
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
        Cow::Borrowed(a)
    } else {
        Cow::Owned(transpose(a, &a_perm))
    };

    let b_t = if b_perm.iter().enumerate().all(|(i, &p)| i == p) {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(transpose(b, &b_perm))
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
    if m * k * n >= MIN_FAER_GEMM_WORK {
        faer_gemm(&a_t.data, &b_t.data, &mut c_data, m, k, n);
    } else if m * n >= MIN_PAR_ELEMS {
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

/// Contraction candidates ordered by result element count, then by the higher
/// slot id descending, then the lower.
///
/// Newest-first on ties keeps the contraction on its frontier instead of letting
/// fresh tensors accumulate legs. Reversing the tie-break raises peak
/// intermediates 16x on `hardware_efficient_ansatz(50, 3)`.
type PairQueue = BinaryHeap<Reverse<(usize, Reverse<usize>, Reverse<usize>)>>;

/// Record `slot` against each of its legs and queue it against every live tensor
/// already sharing one, pruning contracted slots off the holder lists it walks.
///
/// A tensor carrying one leg twice would otherwise queue against itself; no
/// valid circuit produces one.
fn queue_slot_pairs(
    slots: &[Option<Tensor>],
    slot: usize,
    leg_holders: &mut Vec<SmallVec<[usize; 2]>>,
    queue: &mut PairQueue,
) {
    let tensor = slots[slot].as_ref().expect("slot just filled");
    for &leg in &tensor.legs {
        if leg >= leg_holders.len() {
            leg_holders.resize(leg + 1, SmallVec::new());
        }
        leg_holders[leg].retain(|held| slots[*held].is_some());
        for &other in leg_holders[leg].iter().filter(|&&held| held != slot) {
            let cost = contraction_result_size(
                tensor,
                slots[other].as_ref().expect("holder list pruned above"),
            );
            queue.push(Reverse((
                cost,
                Reverse(other.max(slot)),
                Reverse(other.min(slot)),
            )));
        }
        leg_holders[leg].push(slot);
    }
}

/// Pop the cheapest queued pair whose members are both still live.
///
/// Returns `None` once no two live tensors share a leg.
fn pop_live_pair(queue: &mut PairQueue, slots: &[Option<Tensor>]) -> Option<(usize, usize)> {
    while let Some(Reverse((_, Reverse(j), Reverse(i)))) = queue.pop() {
        if slots[i].is_some() && slots[j].is_some() {
            return Some((i, j));
        }
    }
    None
}

/// Multiply out the tensors left once no pair shares a leg, smallest first.
///
/// Reached when every connected component has collapsed to a single tensor, and
/// stable from there: merging two tensors that share no leg with anything cannot
/// create a shared leg. For disjoint operands the greedy cost reduces to the
/// product of their element counts, so smallest-first is the same choice a scan
/// over every pair would make, without the scan.
fn join_disjoint(mut slots: Vec<Option<Tensor>>) -> Tensor {
    let mut by_size: BinaryHeap<Reverse<(usize, usize)>> = slots
        .iter()
        .enumerate()
        .filter_map(|(idx, held)| held.as_ref().map(|t| Reverse((t.num_elements(), idx))))
        .collect();

    while by_size.len() > 1 {
        let Reverse((_, i)) = by_size.pop().expect("two or more queued");
        let Reverse((_, j)) = by_size.pop().expect("two or more queued");
        let a_tensor = slots[i].take().expect("queued slots are live");
        let b_tensor = slots[j].take().expect("queued slots are live");
        let merged = contract(&a_tensor, &b_tensor);
        by_size.push(Reverse((merged.num_elements(), slots.len())));
        slots.push(Some(merged));
    }

    let Reverse((_, last)) = by_size.pop().expect("the network is never empty");
    slots[last].take().expect("queued slots are live")
}

/// Contract an entire tensor network with the greedy min-size heuristic.
///
/// Only pairs involving the tensor a contraction produced change cost, so
/// candidates come off a leg-indexed queue.
fn greedy_contract(tensors: &mut Vec<Tensor>) -> Tensor {
    debug_assert!(!tensors.is_empty());

    let mut slots: Vec<Option<Tensor>> = std::mem::take(tensors).into_iter().map(Some).collect();
    let mut leg_holders: Vec<SmallVec<[usize; 2]>> = Vec::new();
    let mut queue: PairQueue = BinaryHeap::new();

    for slot in 0..slots.len() {
        queue_slot_pairs(&slots, slot, &mut leg_holders, &mut queue);
    }

    while let Some((i, j)) = pop_live_pair(&mut queue, &slots) {
        let a_tensor = slots[i].take().expect("popped pair is live");
        let b_tensor = slots[j].take().expect("popped pair is live");
        slots.push(Some(contract(&a_tensor, &b_tensor)));
        queue_slot_pairs(&slots, slots.len() - 1, &mut leg_holders, &mut queue);
    }

    join_disjoint(slots)
}

struct ScalarExpectationNetwork {
    num_qubits: usize,
    tensors: Vec<Tensor>,
    ket_legs: Vec<LegId>,
    bra_legs: Vec<LegId>,
    next_leg: LegId,
}

impl ScalarExpectationNetwork {
    fn new(num_qubits: usize) -> Self {
        let mut network = Self {
            num_qubits,
            tensors: Vec::with_capacity(num_qubits * 4),
            ket_legs: Vec::with_capacity(num_qubits),
            bra_legs: Vec::with_capacity(num_qubits),
            next_leg: 0,
        };
        let zero_state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        for _ in 0..num_qubits {
            let ket_leg = network.fresh_leg();
            network.ket_legs.push(ket_leg);
            network.tensors.push(Tensor {
                data: zero_state.clone(),
                shape: smallvec::smallvec![2],
                legs: smallvec::smallvec![ket_leg],
            });

            let bra_leg = network.fresh_leg();
            network.bra_legs.push(bra_leg);
            network.tensors.push(Tensor {
                data: zero_state.clone(),
                shape: smallvec::smallvec![2],
                legs: smallvec::smallvec![bra_leg],
            });
        }
        network
    }

    fn fresh_leg(&mut self) -> LegId {
        let leg = self.next_leg;
        self.next_leg += 1;
        leg
    }

    fn validate_qubit(&self, qubit: usize) -> Result<()> {
        if qubit >= self.num_qubits {
            return Err(PrismError::InvalidQubit {
                index: qubit,
                register_size: self.num_qubits,
            });
        }
        Ok(())
    }

    fn append_1q_matrix(
        &mut self,
        target: usize,
        mat: &[[Complex64; 2]; 2],
        conjugate: bool,
    ) -> Result<()> {
        self.validate_qubit(target)?;
        let in_leg = if conjugate {
            self.bra_legs[target]
        } else {
            self.ket_legs[target]
        };
        let out_leg = self.fresh_leg();
        let data = if conjugate {
            vec![
                mat[0][0].conj(),
                mat[0][1].conj(),
                mat[1][0].conj(),
                mat[1][1].conj(),
            ]
        } else {
            vec![mat[0][0], mat[0][1], mat[1][0], mat[1][1]]
        };
        self.tensors.push(Tensor {
            data,
            shape: smallvec::smallvec![2, 2],
            legs: smallvec::smallvec![out_leg, in_leg],
        });
        if conjugate {
            self.bra_legs[target] = out_leg;
        } else {
            self.ket_legs[target] = out_leg;
        }
        Ok(())
    }

    fn append_2q_matrix(
        &mut self,
        q0: usize,
        q1: usize,
        mat: &[[Complex64; 4]; 4],
        conjugate: bool,
    ) -> Result<()> {
        self.validate_qubit(q0)?;
        self.validate_qubit(q1)?;
        let (in0, in1) = if conjugate {
            (self.bra_legs[q0], self.bra_legs[q1])
        } else {
            (self.ket_legs[q0], self.ket_legs[q1])
        };
        let out0 = self.fresh_leg();
        let out1 = self.fresh_leg();
        let mut data = vec![Complex64::new(0.0, 0.0); 16];
        for i0 in 0..2usize {
            for i1 in 0..2usize {
                for j0 in 0..2usize {
                    for j1 in 0..2usize {
                        let value = mat[i0 * 2 + i1][j0 * 2 + j1];
                        data[i0 * 8 + i1 * 4 + j0 * 2 + j1] =
                            if conjugate { value.conj() } else { value };
                    }
                }
            }
        }
        self.tensors.push(Tensor {
            data,
            shape: SmallVec::from_slice(&[2, 2, 2, 2]),
            legs: SmallVec::from_slice(&[out0, out1, in0, in1]),
        });
        if conjugate {
            self.bra_legs[q0] = out0;
            self.bra_legs[q1] = out1;
        } else {
            self.ket_legs[q0] = out0;
            self.ket_legs[q1] = out1;
        }
        Ok(())
    }

    fn append_nq_matrix(
        &mut self,
        qubits: &[usize],
        full_mat: &[Vec<Complex64>],
        conjugate: bool,
    ) -> Result<()> {
        for &qubit in qubits {
            self.validate_qubit(qubit)?;
        }
        let m = qubits.len();
        let dim = 1usize << m;
        if full_mat.len() != dim || full_mat.iter().any(|row| row.len() != dim) {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "tensor-network scalar expected a {dim} by {dim} matrix for {} targets",
                    qubits.len()
                ),
            });
        }
        let in_legs: SmallVec<[LegId; 6]> = if conjugate {
            qubits.iter().map(|&q| self.bra_legs[q]).collect()
        } else {
            qubits.iter().map(|&q| self.ket_legs[q]).collect()
        };
        let out_legs: SmallVec<[LegId; 6]> = (0..m).map(|_| self.fresh_leg()).collect();
        let mut data = vec![Complex64::new(0.0, 0.0); dim * dim];

        for (out_idx, row) in full_mat.iter().enumerate() {
            for (in_idx, &raw) in row.iter().enumerate() {
                let value = if conjugate { raw.conj() } else { raw };
                let mut flat = 0usize;
                for bit in 0..m {
                    let out_bit = (out_idx >> (m - 1 - bit)) & 1;
                    flat = flat * 2 + out_bit;
                }
                for bit in 0..m {
                    let in_bit = (in_idx >> (m - 1 - bit)) & 1;
                    flat = flat * 2 + in_bit;
                }
                data[flat] = value;
            }
        }

        let mut shape: SmallVec<[usize; 6]> = SmallVec::new();
        let mut legs: SmallVec<[LegId; 6]> = SmallVec::new();
        for &leg in &out_legs {
            shape.push(2);
            legs.push(leg);
        }
        for &leg in &in_legs {
            shape.push(2);
            legs.push(leg);
        }
        self.tensors.push(Tensor { data, shape, legs });

        let legs = if conjugate {
            &mut self.bra_legs
        } else {
            &mut self.ket_legs
        };
        for (idx, &qubit) in qubits.iter().enumerate() {
            legs[qubit] = out_legs[idx];
        }
        Ok(())
    }

    fn append_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let num_qubits = self.num_qubits;
        for_each_gate_tensor(gate, targets, num_qubits, |op| match op {
            GateTensorOp::OneQ(q, mat) => {
                self.append_1q_matrix(q, &mat, false)?;
                self.append_1q_matrix(q, &mat, true)
            }
            GateTensorOp::TwoQ(q0, q1, mat) => {
                self.append_2q_matrix(q0, q1, &mat, false)?;
                self.append_2q_matrix(q0, q1, &mat, true)
            }
            GateTensorOp::NQ(qubits, full) => {
                self.append_nq_matrix(qubits, &full, false)?;
                self.append_nq_matrix(qubits, &full, true)
            }
        })
    }

    fn append_observable(&mut self, terms: &[PauliTerm]) -> Result<()> {
        let mut axes = vec![None; self.num_qubits];
        for term in terms {
            self.validate_qubit(term.qubit)?;
            if axes[term.qubit].is_some() {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "tensor-network scalar observable has duplicate factor on qubit {}",
                        term.qubit
                    ),
                });
            }
            axes[term.qubit] = Some(term.axis);
        }

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);
        for (qubit, axis) in axes.into_iter().enumerate() {
            let data = match axis {
                None => vec![one, zero, zero, one],
                Some(PauliAxis::X) => vec![zero, one, one, zero],
                Some(PauliAxis::Y) => vec![zero, neg_i, i, zero],
                Some(PauliAxis::Z) => vec![one, zero, zero, neg_one],
            };
            self.tensors.push(Tensor {
                data,
                shape: smallvec::smallvec![2, 2],
                legs: smallvec::smallvec![self.bra_legs[qubit], self.ket_legs[qubit]],
            });
        }
        Ok(())
    }

    fn contract(mut self) -> Result<f64> {
        if self.tensors.is_empty() {
            return Ok(1.0);
        }
        let result = greedy_contract(&mut self.tensors);
        if result.data.len() != 1 || !result.legs.is_empty() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "tensor-network scalar contraction left {} amplitudes and {} open legs",
                    result.data.len(),
                    result.legs.len()
                ),
            });
        }
        Ok(result.data[0].re)
    }
}

/// Contract `<0| U^dag P U |0>` without materializing a full statevector.
pub(crate) fn expectation_zero_state(circuit: &Circuit, pauli_terms: &[PauliTerm]) -> Result<f64> {
    let mut network = ScalarExpectationNetwork::new(circuit.num_qubits);
    for instruction in &circuit.instructions {
        match instruction {
            Instruction::Gate { gate, targets } => network.append_gate(gate, targets)?,
            Instruction::Barrier { .. } => {}
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. } => {
                return Err(PrismError::BackendUnsupported {
                    backend: "tensor_network_scalar".to_string(),
                    operation: format!("non-unitary instruction {instruction:?}"),
                });
            }
        }
    }
    network.append_observable(pauli_terms)?;
    network.contract()
}

/// Bench-visible wrapper over [`expectation_zero_state`]; not stable API.
#[cfg(feature = "bench-internal")]
pub fn scalar_expectation(circuit: &Circuit, pauli_terms: &[PauliTerm]) -> Result<f64> {
    expectation_zero_state(circuit, pauli_terms)
}

/// Elementary tensor operation a gate decomposes into when appended to a
/// tensor network. `NQ` carries the full `2^k × 2^k` matrix for
/// multi-controlled unitaries.
enum GateTensorOp<'a> {
    OneQ(usize, [[Complex64; 2]; 2]),
    TwoQ(usize, usize, [[Complex64; 4]; 4]),
    NQ(&'a [usize], Vec<Vec<Complex64>>),
}

/// Decompose `gate` into the elementary 1q/2q/nq matrix operations a tensor
/// network applies, invoking `emit` once per operation in application order.
///
/// Shared by the deferred-contraction backend ([`TensorNetworkBackend`], which
/// appends ket tensors only) and the scalar-expectation network
/// ([`ScalarExpectationNetwork`], which appends a conjugated ket+bra pair). The
/// two differ only in their sink, so routing both through this keeps their gate
/// coverage and matrices identical.
fn for_each_gate_tensor<'a, F>(
    gate: &Gate,
    targets: &'a [usize],
    num_qubits: usize,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(GateTensorOp<'a>) -> Result<()>,
{
    let check_qubit = |q: usize| -> Result<()> {
        if q >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: q,
                register_size: num_qubits,
            });
        }
        Ok(())
    };
    let check_arity = |expected: usize| -> Result<()> {
        if targets.len() != expected {
            return Err(PrismError::GateArity {
                gate: gate.name().to_string(),
                expected,
                got: targets.len(),
            });
        }
        for &t in targets {
            check_qubit(t)?;
        }
        Ok(())
    };

    match gate {
        Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap | Gate::Cu(_) | Gate::Fused2q(_) => {
            check_arity(2)?;
            emit(GateTensorOp::TwoQ(
                targets[0],
                targets[1],
                gate.matrix_4x4(),
            ))
        }
        Gate::Mcu(data) => {
            check_arity(data.num_controls as usize + 1)?;
            let full = TensorNetworkBackend::mcu_full_matrix(data.num_controls as usize, &data.mat);
            emit(GateTensorOp::NQ(targets, full))
        }
        Gate::BatchPhase(data) => {
            if targets.is_empty() {
                return Err(PrismError::GateArity {
                    gate: gate.name().to_string(),
                    expected: 1,
                    got: 0,
                });
            }
            check_qubit(targets[0])?;
            let one = Complex64::new(1.0, 0.0);
            let zero = Complex64::new(0.0, 0.0);
            for &(target_qubit, phase) in &data.phases {
                let mat = [
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, one, zero],
                    [zero, zero, zero, phase],
                ];
                emit(GateTensorOp::TwoQ(targets[0], target_qubit, mat))?;
            }
            Ok(())
        }
        Gate::BatchRzz(data) => {
            for &(q0, q1, theta) in &data.edges {
                emit(GateTensorOp::TwoQ(q0, q1, Gate::Rzz(theta).matrix_4x4()))?;
            }
            Ok(())
        }
        Gate::DiagonalBatch(data) => {
            for entry in &data.entries {
                if let Some((q, mat)) = entry.as_1q_matrix() {
                    emit(GateTensorOp::OneQ(q, mat))?;
                } else if let Some((q0, q1, mat)) = entry.as_2q_matrix() {
                    emit(GateTensorOp::TwoQ(q0, q1, mat))?;
                }
            }
            Ok(())
        }
        Gate::MultiFused(data) => {
            for &(target, ref mat) in &data.gates {
                emit(GateTensorOp::OneQ(target, *mat))?;
            }
            Ok(())
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, ref mat) in &data.gates {
                emit(GateTensorOp::TwoQ(q0, q1, *mat))?;
            }
            Ok(())
        }
        Gate::QftBlock { .. } => Err(PrismError::BackendUnsupported {
            backend: "tensor_network".to_string(),
            operation: "QFT block scalar contraction without prior expansion".to_string(),
        }),
        _ => {
            check_arity(1)?;
            emit(GateTensorOp::OneQ(targets[0], gate.matrix_2x2()))
        }
    }
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

    fn replace_with_statevector(&mut self, state: Vec<Complex64>) {
        let n = self.num_qubits;
        self.tensors.clear();
        self.next_leg = 0;
        self.output_legs.clear();

        for _ in 0..n {
            let leg = self.fresh_leg();
            self.output_legs.push(leg);
        }

        let mut shape: SmallVec<[usize; 6]> = SmallVec::new();
        let mut legs: SmallVec<[LegId; 6]> = SmallVec::new();
        for q in (0..n).rev() {
            shape.push(2);
            legs.push(self.output_legs[q]);
        }

        self.tensors.push(Tensor {
            data: state,
            shape,
            legs,
        });
    }

    fn append_1q_matrix(&mut self, target: usize, mat: &[[Complex64; 2]; 2]) {
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

    fn apply_reset(&mut self, qubit: usize) -> Result<()> {
        use rand::RngExt;

        let amplitudes = self.contract_to_statevector()?;
        let mask = 1usize << qubit;

        let mut prob_one = 0.0f64;
        for (idx, amp) in amplitudes.iter().enumerate() {
            if idx & mask != 0 {
                prob_one += amp.norm_sqr();
            }
        }
        let outcome = self.rng.random::<f64>() < prob_one;
        let inv_norm = crate::backend::measurement_inv_norm(outcome, prob_one);

        let mut collapsed = vec![Complex64::new(0.0, 0.0); amplitudes.len()];
        for (idx, amp) in amplitudes.iter().enumerate() {
            if (idx & mask != 0) == outcome {
                collapsed[idx & !mask] = amp * inv_norm;
            }
        }

        self.replace_with_statevector(collapsed);
        Ok(())
    }

    /// Leg ids the bra copy uses, one entry per live ket leg.
    ///
    /// Every leg is shifted clear of the ket id space by [`Self::next_leg`], so a
    /// bra tensor contracts only against other bra tensors. Callers then map back
    /// to the ket id the legs they want summed over: an output leg mapped to
    /// itself closes that qubit against its twin, which is a trace with identity.
    fn bra_leg_map(&self) -> Vec<LegId> {
        (0..self.next_leg).map(|leg| leg + self.next_leg).collect()
    }

    /// Pair every tensor with a conjugated twin whose legs are read off `bra_legs`.
    fn double_through(&self, bra_legs: &[LegId]) -> Vec<Tensor> {
        let mut network = Vec::with_capacity(self.tensors.len() * 2);
        for tensor in &self.tensors {
            network.push(tensor.clone());
            network.push(Tensor {
                data: tensor.data.iter().map(Complex64::conj).collect(),
                shape: tensor.shape.clone(),
                legs: tensor.legs.iter().map(|&leg| bra_legs[leg]).collect(),
            });
        }
        network
    }

    /// Build the network for `tr_{q != qubit} |psi><psi|`, leaving `qubit`'s ket
    /// and bra indices open.
    ///
    /// Returns the network and the two open leg ids as `(ket, bra)`.
    fn double_for_partial_trace(&self, qubit: usize) -> (Vec<Tensor>, LegId, LegId) {
        let ket_leg = self.output_legs[qubit];
        let mut bra_legs = self.bra_leg_map();
        for (q, &leg) in self.output_legs.iter().enumerate() {
            if q != qubit {
                bra_legs[leg] = leg;
            }
        }

        let network = self.double_through(&bra_legs);
        (network, ket_leg, ket_leg + self.next_leg)
    }

    /// Contract `<psi|P|psi>` for one joint Pauli observable, unnormalized.
    ///
    /// Qubits the observable omits carry identity, and an identity factor is a
    /// leg closed directly against its twin rather than a tensor appended, which
    /// keeps `n - k` tensors out of the network for a weight-`k` observable.
    fn contract_pauli_sandwich(&self, axes: &[Option<PauliAxis>]) -> f64 {
        let mut bra_legs = self.bra_leg_map();
        for (q, axis) in axes.iter().enumerate() {
            if axis.is_none() {
                let leg = self.output_legs[q];
                bra_legs[leg] = leg;
            }
        }

        let mut network = self.double_through(&bra_legs);

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        for (q, axis) in axes.iter().enumerate() {
            let Some(axis) = axis else { continue };
            let data = match axis {
                PauliAxis::X => vec![zero, one, one, zero],
                PauliAxis::Y => vec![zero, -i, i, zero],
                PauliAxis::Z => vec![one, zero, zero, -one],
            };
            let ket_leg = self.output_legs[q];
            network.push(Tensor {
                data,
                shape: smallvec::smallvec![2, 2],
                legs: smallvec::smallvec![bra_legs[ket_leg], ket_leg],
            });
        }

        let result = greedy_contract(&mut network);
        debug_assert!(result.legs.is_empty(), "sandwich leaves every leg paired");
        result.data[0].re
    }

    /// Contract the full network and return the amplitude vector in
    /// computational basis order.
    fn contract_to_statevector(&self) -> Result<Vec<Complex64>> {
        dense_statevector_len(self.name(), "contraction", self.num_qubits)?;

        let mut tensors = self.tensors.clone();
        let result = greedy_contract(&mut tensors);

        // The result tensor's legs should be exactly the output_legs.
        // PRISM-Q convention: q[0] = LSB of state index. In row-major
        // tensor layout, the last axis is LSB. Use leg order
        // [q_{n-1}, q_{n-2}, ..., q_0], reversed.
        let target_order: Vec<LegId> = self.output_legs.iter().rev().copied().collect();
        let perm: SmallVec<[usize; 6]> = target_order
            .iter()
            .map(|target_leg| {
                result
                    .legs
                    .iter()
                    .position(|l| l == target_leg)
                    .expect("greedy_contract consumes every tensor, so output legs survive")
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

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let num_qubits = self.num_qubits;
        for_each_gate_tensor(gate, targets, num_qubits, |op| {
            match op {
                GateTensorOp::OneQ(q, mat) => self.append_1q_matrix(q, &mat),
                GateTensorOp::TwoQ(q0, q1, mat) => self.apply_2q_matrix(q0, q1, &mat),
                GateTensorOp::NQ(qubits, full) => self.apply_nq_matrix(qubits, &full),
            }
            Ok(())
        })
    }
}

impl Backend for TensorNetworkBackend {
    fn name(&self) -> &'static str {
        "tensornetwork"
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::TensorNetwork
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.tensors = Vec::new();
        self.next_leg = 0;
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);

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
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                use rand::RngExt;

                let amplitudes = self.contract_to_statevector()?;

                let mut prob_one = 0.0f64;
                for (idx, amp) in amplitudes.iter().enumerate() {
                    if (idx >> qubit) & 1 == 1 {
                        prob_one += amp.norm_sqr();
                    }
                }

                let outcome = self.rng.random::<f64>() < prob_one;
                self.classical_bits[*classical_bit] = outcome;

                let mut collapsed = amplitudes;
                let mut norm_sq = 0.0f64;
                for (idx, amp) in collapsed.iter_mut().enumerate() {
                    let bit = (idx >> qubit) & 1 == 1;
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

                self.replace_with_statevector(collapsed);
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

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.append_1q_matrix(qubit, matrix);
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        tensor_probability_len(self.name(), self.num_qubits)?;
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

    /// Contracts the network against its conjugate with every other qubit's
    /// output leg closed, so the cost is set by the doubled network's treewidth
    /// rather than by `2^n`.
    ///
    /// # Panics
    ///
    /// If `qubit` is outside the register.
    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let (mut network, ket_leg, bra_leg) = self.double_for_partial_trace(qubit);
        let rho = greedy_contract(&mut network);

        let axis = |leg: LegId| {
            rho.legs
                .iter()
                .position(|&l| l == leg)
                .expect("partial trace leaves both open legs on the result")
        };
        let ket_stride = if axis(ket_leg) == 0 { 2 } else { 1 };
        let bra_stride = if axis(bra_leg) == 0 { 2 } else { 1 };

        Ok([
            [rho.data[0], rho.data[bra_stride]],
            [rho.data[ket_stride], rho.data[ket_stride + bra_stride]],
        ])
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Contracts `<psi|P|psi>` directly, so no `2^n` vector is built and the
    /// dense query ceiling does not apply.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidQubit`] for a factor outside the register, and
    /// [`PrismError::InvalidParameter`] for two factors on one qubit.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let mut axes: Vec<Option<PauliAxis>> = vec![None; self.num_qubits];
        let mut expectations = Vec::with_capacity(observables.len());
        let norm_sq = self.contract_pauli_sandwich(&axes);

        for observable in observables {
            axes.iter_mut().for_each(|axis| *axis = None);
            for term in observable {
                if term.qubit >= self.num_qubits {
                    return Err(PrismError::InvalidQubit {
                        index: term.qubit,
                        register_size: self.num_qubits,
                    });
                }
                if axes[term.qubit].is_some() {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "tensor-network observable has duplicate factor on qubit {}",
                            term.qubit
                        ),
                    });
                }
                axes[term.qubit] = Some(term.axis);
            }
            expectations.push(self.contract_pauli_sandwich(&axes) / norm_sq);
        }

        Ok(expectations)
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
    use crate::backend::Backend;
    use crate::backend::statevector::StatevectorBackend;
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
                    all_diagonal: false,
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

    #[test]
    fn test_scalar_expectation_matches_statevector() {
        let circuit = crate::circuits::hardware_efficient_ansatz(8, 2, 42);
        let terms = [PauliTerm::z(1), PauliTerm::x(5)];

        let expected =
            crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
        let actual = expectation_zero_state(&circuit, &terms).unwrap();
        assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
    }

    // Idle qubits leave one disconnected component each, which is what
    // join_disjoint exists for.
    #[test]
    fn test_scalar_expectation_with_idle_qubits() {
        let mut circuit = Circuit::new(9, 0);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_gate(Gate::Ry(0.7), &[6]);
        let terms = [PauliTerm::z(2), PauliTerm::z(3), PauliTerm::x(6)];

        let expected =
            crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
        let actual = expectation_zero_state(&circuit, &terms).unwrap();
        assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
    }

    // Two entangled components of unequal size, so join_disjoint merges tensors
    // rather than bare scalars and its smallest-first order is observable.
    #[test]
    fn test_scalar_expectation_unequal_disjoint_components() {
        let mut circuit = Circuit::new(10, 0);
        for &q in &[0usize, 1, 2, 3, 4] {
            circuit.add_gate(Gate::Ry(0.3 + q as f64 * 0.1), &[q]);
        }
        for &(a, b) in &[(0usize, 1usize), (1, 2), (2, 3), (3, 4)] {
            circuit.add_gate(Gate::Cx, &[a, b]);
        }
        circuit.add_gate(Gate::H, &[7]);
        circuit.add_gate(Gate::Cx, &[7, 8]);
        let terms = [PauliTerm::z(0), PauliTerm::x(4), PauliTerm::z(7)];

        let expected =
            crate::sim::run_expectation_values(&circuit, &[terms.to_vec()], 42).unwrap()[0];
        let actual = expectation_zero_state(&circuit, &terms).unwrap();
        assert!((actual - expected).abs() < EPS, "{actual} vs {expected}");
    }
}
