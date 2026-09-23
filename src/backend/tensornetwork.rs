//! Tensor-network simulation backend.
//!
//! Represents the quantum state as a network of tensors. Gate application
//! appends gate tensors to the network (deferred contraction). Contraction
//! happens lazily when `probabilities()` or another query is requested.
//!
//! # Memory layout
//!
//! - Each tensor: contiguous `Vec<Complex64>` plus a shape vector and leg ids
//!   (up to 6 held inline).
//! - Gates append tensors, so memory grows with gate count until a query
//!   contracts the network. Measurement and reset absorb a projector into the
//!   tensor holding the measured qubit's output leg and keep the deferred
//!   form: the outcome marginal contracts the doubled network, so mid-circuit
//!   measurement carries no dense width ceiling and does not grow the network.
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
//!
//! # Contraction strategy
//!
//! A metadata-only planner picks the pair order, then the kernel replays it.
//! The baseline plan is the greedy min-size heuristic: repeatedly contract the
//! pair of tensors whose result has the smallest total element count,
//! preferring pairs sharing a leg. O(T·r·log T) for T tensors of rank at most
//! r. When the greedy plan's peak intermediate reaches
//! `RESTART_PEAK_THRESHOLD`, the planner reruns with seeded multiplicative
//! noise on the size key, `PLAN_RESTARTS_PER_TEMPERATURE` passes at each
//! `PLAN_NOISE_TEMPERATURES` entry, and keeps the tree with the smallest peak
//! intermediate, the greedy tree included, so the peak never rises. Planning
//! touches shapes and legs only, so a restart costs a heap walk, not data
//! movement. The winning plan's peak is held to `PRISM_MAX_TN_PEAK_QUBITS`
//! (a memory-derived `2^q` elements by default) before the replay allocates
//! anything, so a contraction the host cannot hold errors instead of aborting.
//!
//! # Index slicing at the memory ceiling
//!
//! A plan whose peak clears the cap is not rejected outright. Legs are fixed
//! one at a time, greedily by the peak each choice buys, and the network is
//! contracted once per assignment of the fixed legs and summed: a
//! multiplicative time factor in exchange for a divided peak. Only a plan
//! still over the cap once `PRISM_MAX_TN_SLICES` assignments are on the table
//! is rejected, and the rejection names the cap. Slices are independent, so
//! the parallel build runs them through Rayon, but only as many at once as fit
//! under the cap together: the cap bounds the live intermediates of a sliced
//! run the way it bounds an unsliced one. Sliced legs are shared by exactly two
//! tensors, so the slice results sum; open legs stay whole. Every terminal
//! contracts through the same path, and the result is exact either way.
//!
//! # Bounded contraction
//!
//! [`TensorNetworkBackend::with_tolerance`] trades exactness for reach. With
//! a tolerance set, a plan over the cap first has an intermediate factored
//! across the cut that separates its partner-facing legs from the rest, the
//! new bond kept only as far as the tolerance on that cut's relative
//! discarded weight allows. The factored halves go back into the network and
//! the rest is replanned, so the peak comes down without the whole tensor
//! ever forming. Slicing then covers whatever truncation leaves above the
//! cap. Without a tolerance, slicing is the only lever and the answer is
//! exact.
//!
//! # Observables and shots both contract natively
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
//! `Backend::sample_basis_states` answers below the dense ceiling from one
//! contraction of the full distribution, a measured 66x cheaper than the
//! sweep at 16 qubits and 32 shots. Past the ceiling it samples qubit by
//! qubit: each bit is drawn from the conditioned single-qubit marginal, and
//! the outcome projector is absorbed before the next qubit's marginal, so a
//! shot costs `n` doubled contractions whose peak is set by treewidth rather
//! than `2^n`, each planned on the first shot and replayed from a per-call
//! cache on the rest.
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

use crate::backend::{
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_statevector_len, reserve_dense_output,
    tensor_peak_cap_elements, tensor_peak_error, tensor_probability_len,
};
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

/// Shape and legs of a tensor, all the planner reads.
#[derive(Clone)]
struct TensorMeta {
    shape: SmallVec<[usize; 6]>,
    legs: SmallVec<[LegId; 6]>,
}

impl TensorMeta {
    fn of(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape.clone(),
            legs: tensor.legs.clone(),
        }
    }

    fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

/// Pair order for one contraction, with the two counts plans are ranked by.
///
/// `pairs` holds slot indices, inputs first, each result appended at the next
/// index. `peak` is the largest result element count; `total` sums them and
/// breaks peak ties.
struct ContractionPlan {
    pairs: Vec<(usize, usize)>,
    peak: usize,
    total: usize,
}

/// Noisy passes per temperature once the greedy plan's peak intermediate
/// reaches [`RESTART_PEAK_THRESHOLD`].
///
/// Sized against the temperature sweep on `hardware_efficient_ansatz(n, 7)`
/// scalar networks: first improvements appeared as late as a temperature's
/// 23rd pass, and a fully fruitless 32-pass sweep cost 110 ms per
/// temperature against the multi-second contraction the threshold
/// guarantees.
const PLAN_RESTARTS_PER_TEMPERATURE: u64 = 32;

/// Peak intermediate element count at which the noisy restarts run.
///
/// Below it the contraction is cheap enough that extra planning passes cost
/// more than a better tree returns; the depth-swept bench rows peak an order
/// of magnitude under this and stay on the single greedy pass.
const RESTART_PEAK_THRESHOLD: usize = 1 << 22;

/// Noise scales for the restart sweep, in doublings of the size key.
///
/// Measured on `hardware_efficient_ansatz(n, 7)` scalar networks under the
/// per-pass seeding: 0.25 and below found nothing at n = 50 in 32 passes,
/// 2.0 and 4.0 found nothing at either width, and 0.5 and 1.0 carried every
/// improvement seen.
const PLAN_NOISE_TEMPERATURES: [f64; 2] = [0.5, 1.0];

/// Fixed base seed for the restart noise, so one network always maps to one
/// tree.
///
/// Planning runs inside `&self` queries that cannot reach the run rng, and
/// drawing from it would shift the measurement outcome stream relative to the
/// other backends. Each (temperature, pass) reseeds from this base, so a
/// pass's noise does not depend on how early the passes before it aborted.
const PLAN_NOISE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn contraction_result_size(a: &TensorMeta, b: &TensorMeta) -> usize {
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

/// Compute the free legs of `a` then `b`, in each operand's axis order,
/// matching the result [`contract`] builds for the same pair.
fn contract_meta(a: &TensorMeta, b: &TensorMeta) -> TensorMeta {
    let mut shape: SmallVec<[usize; 6]> = SmallVec::new();
    let mut legs: SmallVec<[LegId; 6]> = SmallVec::new();
    for (ai, &leg) in a.legs.iter().enumerate() {
        if !b.legs.contains(&leg) {
            shape.push(a.shape[ai]);
            legs.push(leg);
        }
    }
    for (bi, &leg) in b.legs.iter().enumerate() {
        if !a.legs.contains(&leg) {
            shape.push(b.shape[bi]);
            legs.push(leg);
        }
    }
    TensorMeta { shape, legs }
}

/// Contraction candidates ordered by size key, then by the higher slot id
/// descending, then the lower.
///
/// The key is the result element count, scaled by Gumbel noise on a noisy
/// planning pass. Newest-first on ties keeps the contraction on its frontier
/// instead of letting fresh tensors accumulate legs. Reversing the tie-break
/// raises peak intermediates 16x on `hardware_efficient_ansatz(50, 3)`.
type PairQueue = BinaryHeap<Reverse<(u64, Reverse<usize>, Reverse<usize>)>>;

/// Compute the size key for one candidate pair.
///
/// The noisy arm multiplies by `2^(temperature * gumbel)` and compares the
/// result through its bit pattern, which orders positive floats numerically.
fn pair_key(cost: usize, noise: Option<(&mut ChaCha8Rng, f64)>) -> u64 {
    match noise {
        None => cost as u64,
        Some((rng, temperature)) => {
            use rand::RngExt;
            let uniform: f64 = rng.random::<f64>();
            let gumbel = -(-uniform.max(f64::MIN_POSITIVE).ln()).ln();
            ((cost as f64) * (temperature * gumbel).exp2()).to_bits()
        }
    }
}

/// Record `slot` against each of its legs and queue it against every live tensor
/// already sharing one, pruning contracted slots off the holder lists it walks.
///
/// A tensor carrying one leg twice would otherwise queue against itself; no
/// valid circuit produces one.
fn queue_slot_pairs(
    slots: &[Option<TensorMeta>],
    slot: usize,
    leg_holders: &mut Vec<SmallVec<[usize; 2]>>,
    queue: &mut PairQueue,
    noise: &mut Option<(&mut ChaCha8Rng, f64)>,
) {
    let meta = slots[slot].as_ref().expect("slot just filled");
    for &leg in &meta.legs {
        if leg >= leg_holders.len() {
            leg_holders.resize(leg + 1, SmallVec::new());
        }
        leg_holders[leg].retain(|held| slots[*held].is_some());
        for &other in leg_holders[leg].iter().filter(|&&held| held != slot) {
            let cost = contraction_result_size(
                meta,
                slots[other].as_ref().expect("holder list pruned above"),
            );
            queue.push(Reverse((
                pair_key(
                    cost,
                    noise
                        .as_mut()
                        .map(|(rng, temperature)| (&mut **rng, *temperature)),
                ),
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
fn pop_live_pair(queue: &mut PairQueue, slots: &[Option<TensorMeta>]) -> Option<(usize, usize)> {
    while let Some(Reverse((_, Reverse(j), Reverse(i)))) = queue.pop() {
        if slots[i].is_some() && slots[j].is_some() {
            return Some((i, j));
        }
    }
    None
}

/// Run one greedy planning pass over the metadata, deterministic when `noise`
/// is `None` and Gumbel-perturbed otherwise.
///
/// Returns `None` as soon as a result exceeds `abort_above` elements: the pass
/// can no longer beat the plan holding that peak, and abandoning it keeps a
/// failed restart from paying for a full walk. Peak ties complete, since they
/// can still win on `total`.
fn plan_pairs(
    mut slots: Vec<Option<TensorMeta>>,
    mut noise: Option<(&mut ChaCha8Rng, f64)>,
    abort_above: usize,
) -> Option<ContractionPlan> {
    let mut leg_holders: Vec<SmallVec<[usize; 2]>> = Vec::new();
    let mut queue: PairQueue = BinaryHeap::new();

    for slot in 0..slots.len() {
        queue_slot_pairs(&slots, slot, &mut leg_holders, &mut queue, &mut noise);
    }

    let mut plan = ContractionPlan {
        pairs: Vec::new(),
        peak: 0,
        total: 0,
    };
    while let Some((i, j)) = pop_live_pair(&mut queue, &slots) {
        let a = slots[i].take().expect("popped pair is live");
        let b = slots[j].take().expect("popped pair is live");
        let result = contract_meta(&a, &b);
        let elements = result.num_elements();
        if elements > abort_above {
            return None;
        }
        plan.peak = plan.peak.max(elements);
        plan.total += elements;
        plan.pairs.push((i, j));
        slots.push(Some(result));
        queue_slot_pairs(
            &slots,
            slots.len() - 1,
            &mut leg_holders,
            &mut queue,
            &mut noise,
        );
    }
    Some(plan)
}

/// Plan greedily, then rerun with noise when the greedy peak intermediate
/// reaches [`RESTART_PEAK_THRESHOLD`], keeping the best plan by peak then
/// total.
///
/// The greedy plan competes, so the peak never rises. The restart arm walks
/// `tensors` a second time; below the threshold the single pass pays one
/// metadata copy and no more.
fn plan_with_restarts(tensors: &[Tensor]) -> ContractionPlan {
    #[cfg(test)]
    PLANNER_CALLS.with(|calls| calls.set(calls.get() + 1));
    let slots: Vec<Option<TensorMeta>> = tensors.iter().map(|t| Some(TensorMeta::of(t))).collect();
    let mut plan = plan_pairs(slots, None, usize::MAX).expect("unbounded pass completes");
    if plan.peak >= RESTART_PEAK_THRESHOLD {
        let metas: Vec<TensorMeta> = tensors.iter().map(TensorMeta::of).collect();
        for (temp_index, &temperature) in PLAN_NOISE_TEMPERATURES.iter().enumerate() {
            for pass in 0..PLAN_RESTARTS_PER_TEMPERATURE {
                let pass_seed = PLAN_NOISE_SEED ^ (((temp_index as u64) << 32) | pass);
                let mut rng = ChaCha8Rng::seed_from_u64(pass_seed);
                let slots: Vec<Option<TensorMeta>> = metas.iter().cloned().map(Some).collect();
                let Some(candidate) = plan_pairs(slots, Some((&mut rng, temperature)), plan.peak)
                else {
                    continue;
                };
                if (candidate.peak, candidate.total) < (plan.peak, plan.total) {
                    plan = candidate;
                }
            }
        }
    }
    plan
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

/// Contract an entire tensor network along a planned pair order.
///
/// Planning walks metadata only; the replay in [`contract_within`] is where
/// data moves.
fn greedy_contract(
    tensors: &mut Vec<Tensor>,
    limits: ContractionLimits,
    backend: &str,
    operation: &str,
) -> Result<Tensor> {
    debug_assert!(!tensors.is_empty());

    let plan = plan_with_restarts(tensors);
    contract_within(tensors, &plan, limits, backend, operation)
}

/// Peak-intermediate cap, slice budget and truncation tolerance one
/// contraction runs under.
///
/// `tolerance` is the per-cut relative squared weight a bounded contraction
/// may discard; `None` keeps the contraction exact.
#[derive(Clone, Copy)]
struct ContractionLimits {
    peak_cap: usize,
    slice_budget: usize,
    tolerance: Option<f64>,
}

impl ContractionLimits {
    fn from_env() -> Self {
        Self {
            peak_cap: tensor_peak_cap_elements(),
            slice_budget: max_slices(),
            tolerance: None,
        }
    }

    fn with_tolerance(tolerance: Option<f64>) -> Self {
        Self {
            tolerance,
            ..Self::from_env()
        }
    }
}

/// Independent contractions a sliced run may sum over before the peak cap is
/// reported unreachable.
///
/// Slicing trades a multiplicative time factor for a divided peak, so the
/// budget is the worst-case slowdown a caller takes in place of a rejection.
const DEFAULT_SLICE_BUDGET: usize = 1 << 10;

/// Slice budget from `PRISM_MAX_TN_SLICES`, cached for the process. A budget
/// of 1 turns slicing off, so a plan over the cap is rejected.
fn max_slices() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        crate::env_knobs::usize_knob("PRISM_MAX_TN_SLICES", DEFAULT_SLICE_BUDGET, 1)
    })
}

/// Legs fixed for a sliced contraction, in the order they were chosen.
///
/// `count` is the product of `dims`: one contraction per assignment. `peak`
/// is the largest intermediate one slice plans, in elements.
struct SlicePlan {
    legs: SmallVec<[LegId; 4]>,
    dims: SmallVec<[usize; 4]>,
    count: usize,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    peak: usize,
}

thread_local! {
    static LAST_SLICE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    static LAST_DISCARDED: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Slices the most recent contraction on this thread summed over, 1 when it
/// ran whole. Diagnostic for tests and tuning; not stable API.
#[doc(hidden)]
pub fn last_slice_count() -> usize {
    LAST_SLICE_COUNT.with(std::cell::Cell::get)
}

/// Relative squared weight the most recent contraction on this thread
/// discarded, summed over its cuts.
fn last_discarded() -> f64 {
    LAST_DISCARDED.with(std::cell::Cell::get)
}

/// Cost of one slice of a plan, and the legs a further slice could act on.
///
/// `peak` is the largest intermediate, `total` sums them all, and `hot` holds
/// every leg carried by an intermediate over the cap. Ranking on `total` as
/// well as `peak` is what carries the search off a plateau, where several
/// intermediates sit at the peak and no single leg reaches all of them.
struct SliceCost {
    peak: usize,
    total: usize,
    hot: Vec<LegId>,
}

/// Replay `plan` over metadata with every leg in `sliced` pinned to extent 1.
///
/// Pinning rather than dropping the axis keeps every planned pair sharing the
/// legs it was planned on, so one tree serves every slice.
fn sliced_cost(
    metas: &[TensorMeta],
    plan: &ContractionPlan,
    sliced: &[LegId],
    cap: usize,
) -> SliceCost {
    let mut slots: Vec<Option<TensorMeta>> = metas
        .iter()
        .map(|meta| {
            let mut meta = meta.clone();
            for (axis, leg) in meta.legs.iter().enumerate() {
                if sliced.contains(leg) {
                    meta.shape[axis] = 1;
                }
            }
            Some(meta)
        })
        .collect();

    let mut cost = SliceCost {
        peak: 0,
        total: 0,
        hot: Vec::new(),
    };
    for &(i, j) in &plan.pairs {
        let a = slots[i].take().expect("planned pair is live");
        let b = slots[j].take().expect("planned pair is live");
        let result = contract_meta(&a, &b);
        let elements = result.num_elements();
        cost.peak = cost.peak.max(elements);
        cost.total += elements;
        if elements > cap {
            for &leg in &result.legs {
                if !cost.hot.contains(&leg) {
                    cost.hot.push(leg);
                }
            }
        }
        slots.push(Some(result));
    }
    cost
}

/// Legs whose slices sum: shared by exactly two tensors, so fixing one splits
/// a contraction rather than an output index.
///
/// An open leg fixed the same way would want a scatter instead of a sum, and
/// the terminals holding open legs carry their own dense ceiling anyway.
fn sliceable_legs(metas: &[TensorMeta]) -> Vec<(LegId, usize)> {
    let mut seen: Vec<(LegId, usize, usize)> = Vec::new();
    for meta in metas {
        for (axis, &leg) in meta.legs.iter().enumerate() {
            match seen.iter_mut().find(|(held, _, _)| *held == leg) {
                Some((_, _, holders)) => *holders += 1,
                None => seen.push((leg, meta.shape[axis], 1)),
            }
        }
    }
    seen.into_iter()
        .filter(|&(_, dim, holders)| holders == 2 && dim > 1)
        .map(|(leg, dim, _)| (leg, dim))
        .collect()
}

/// Pick sliced legs greedily by the peak each one buys, re-evaluating the peak
/// after every choice, until the per-slice peak fits under the cap.
///
/// Candidates come from the legs of the intermediates that are over the cap: a
/// leg none of them carries cannot bring the peak down, so the search stays on
/// a handful of ranks rather than on the leg count of the network. `None` when
/// the slice budget runs out first, or when no remaining candidate improves
/// the pair the search ranks on.
fn choose_slices(
    metas: &[TensorMeta],
    plan: &ContractionPlan,
    limits: ContractionLimits,
) -> Option<SlicePlan> {
    let candidates = sliceable_legs(metas);
    let mut legs: SmallVec<[LegId; 4]> = SmallVec::new();
    let mut dims: SmallVec<[usize; 4]> = SmallVec::new();
    let mut count = 1usize;
    let mut cost = sliced_cost(metas, plan, &legs, limits.peak_cap);

    while cost.peak > limits.peak_cap {
        let mut best: Option<(SliceCost, LegId, usize)> = None;
        for &leg in &cost.hot {
            if legs.contains(&leg) {
                continue;
            }
            let Some(&(_, dim)) = candidates.iter().find(|&&(held, _)| held == leg) else {
                continue;
            };
            if count.saturating_mul(dim) > limits.slice_budget {
                continue;
            }
            legs.push(leg);
            let candidate = sliced_cost(metas, plan, &legs, limits.peak_cap);
            legs.pop();
            if best.as_ref().is_none_or(|(held, _, _)| {
                (candidate.peak, candidate.total) < (held.peak, held.total)
            }) {
                best = Some((candidate, leg, dim));
            }
        }
        let (candidate, leg, dim) = best?;
        if (candidate.peak, candidate.total) >= (cost.peak, cost.total) {
            return None;
        }
        legs.push(leg);
        dims.push(dim);
        count *= dim;
        cost = candidate;
    }

    (!legs.is_empty()).then_some(SlicePlan {
        legs,
        dims,
        count,
        peak: cost.peak,
    })
}

/// Copy `tensor` with each axis named in `pinned` held at one index and kept
/// at extent 1.
fn pin_axes(tensor: &Tensor, pinned: &[(usize, usize)]) -> Tensor {
    let rank = tensor.rank();
    let mut strides: SmallVec<[usize; 6]> = SmallVec::from_elem(1usize, rank);
    for axis in (0..rank.saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * tensor.shape[axis + 1];
    }

    let mut shape = tensor.shape.clone();
    let mut source = 0usize;
    for &(axis, index) in pinned {
        source += index * strides[axis];
        shape[axis] = 1;
    }

    let total: usize = shape.iter().product();
    let mut data = Vec::with_capacity(total);
    let mut counter: SmallVec<[usize; 6]> = SmallVec::from_elem(0usize, rank);
    for _ in 0..total {
        data.push(tensor.data[source]);
        for axis in (0..rank).rev() {
            if shape[axis] == 1 {
                continue;
            }
            counter[axis] += 1;
            source += strides[axis];
            if counter[axis] < shape[axis] {
                break;
            }
            counter[axis] = 0;
            source -= strides[axis] * shape[axis];
        }
    }

    Tensor {
        data,
        shape,
        legs: tensor.legs.clone(),
    }
}

/// The network for slice `index`, every sliced leg pinned to the value that
/// index names in mixed radix over [`SlicePlan::dims`].
fn slice_network(tensors: &[Tensor], slice: &SlicePlan, index: usize) -> Vec<Tensor> {
    let mut values: SmallVec<[usize; 4]> = SmallVec::new();
    let mut rest = index;
    for &dim in &slice.dims {
        values.push(rest % dim);
        rest /= dim;
    }

    tensors
        .iter()
        .map(|tensor| {
            let pinned: SmallVec<[(usize, usize); 4]> = tensor
                .legs
                .iter()
                .enumerate()
                .filter_map(|(axis, leg)| {
                    slice
                        .legs
                        .iter()
                        .position(|held| held == leg)
                        .map(|which| (axis, values[which]))
                })
                .collect();
            if pinned.is_empty() {
                tensor.clone()
            } else {
                pin_axes(tensor, &pinned)
            }
        })
        .collect()
}

/// Add `addend` into `total` elementwise, taking it whole when there is no
/// running total yet.
fn accumulate_slice(total: Option<Tensor>, addend: Tensor) -> Tensor {
    let Some(mut total) = total else {
        return addend;
    };
    debug_assert_eq!(total.legs, addend.legs, "slices leave the same open legs");
    for (slot, term) in total.data.iter_mut().zip(&addend.data) {
        *slot += term;
    }
    total
}

/// Sum the slices of `plan` over every assignment of the sliced legs.
///
/// Every slice replays the same tree over its own copy of the network, so the
/// results carry identical legs and add elementwise.
///
/// The parallel arm runs slices in waves of as many as fit under `peak_cap`
/// together, `peak_cap / slice.peak` of them. Each slice's intermediates reach
/// up to `slice.peak`, so letting every worker take one at once would hold the
/// sum of their peaks, which the cap was sized never to allow.
fn contract_slices(
    tensors: &[Tensor],
    plan: &ContractionPlan,
    slice: &SlicePlan,
    peak_cap: usize,
) -> Tensor {
    let run = |index: usize| {
        let mut network = slice_network(tensors, slice, index);
        replay_plan(&mut network, plan)
    };

    #[cfg(feature = "parallel")]
    {
        let wave = (peak_cap / slice.peak.max(1)).clamp(1, slice.count);
        let mut total = None;
        for start in (0..slice.count).step_by(wave) {
            let end = (start + wave).min(slice.count);
            let part = (start..end)
                .into_par_iter()
                .map(run)
                .reduce_with(|left, right| accumulate_slice(Some(left), right))
                .expect("a wave holds at least one slice");
            total = Some(accumulate_slice(total, part));
        }
        total.expect("a slice plan holds at least one slice")
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = peak_cap;
        (0..slice.count)
            .fold(None, |total, index| {
                Some(accumulate_slice(total, run(index)))
            })
            .expect("a slice plan holds at least one slice")
    }
}

/// Replay `plan` over `tensors`, holding the peak to the tensor-network cap.
///
/// A contraction whose planned peak fits runs whole. One that does not is
/// brought under the cap by the levers the limits allow, in order: bond
/// truncation when a tolerance is set, then index slicing over whatever is
/// still above the cap. Only a contraction over the cap after both is
/// rejected, and `backend` and `operation` name it. Every terminal contracts
/// through here, so all of them take that route.
fn contract_within(
    tensors: &mut Vec<Tensor>,
    plan: &ContractionPlan,
    limits: ContractionLimits,
    backend: &str,
    operation: &str,
) -> Result<Tensor> {
    LAST_DISCARDED.with(|weight| weight.set(0.0));
    if plan.peak <= limits.peak_cap {
        LAST_SLICE_COUNT.with(|count| count.set(1));
        return Ok(replay_plan(tensors, plan));
    }
    match limits.tolerance {
        Some(tolerance) => contract_truncated(tensors, limits, tolerance, backend, operation),
        None => slice_to_fit(tensors, plan, limits, backend, operation),
    }
}

/// Sum the slices that bring `plan` under the cap, or reject when the slice
/// budget cannot reach it.
fn slice_to_fit(
    tensors: &mut Vec<Tensor>,
    plan: &ContractionPlan,
    limits: ContractionLimits,
    backend: &str,
    operation: &str,
) -> Result<Tensor> {
    let metas: Vec<TensorMeta> = tensors.iter().map(TensorMeta::of).collect();
    let Some(slice) = choose_slices(&metas, plan, limits) else {
        return Err(tensor_peak_error(
            backend,
            operation,
            plan.peak,
            limits.peak_cap,
        ));
    };
    LAST_SLICE_COUNT.with(|count| count.set(slice.count));
    let summed = contract_slices(tensors, plan, &slice, limits.peak_cap);
    tensors.clear();
    Ok(summed)
}

/// Contract with intermediate bonds truncated at `tolerance`, slicing
/// whatever truncation leaves above the cap.
///
/// Each pass plans the live network, replays it as far as the cap allows, and
/// factors the operand of the first pair that would cross it. Keeping that
/// operand in factored form is what pays: the next plan contracts the small
/// side against the partner rather than the whole tensor. A pass that fails
/// to lower the peak and total it ranks on hands the rest to slicing, which
/// also bounds the loop, since both counts fall on every pass that continues.
fn contract_truncated(
    tensors: &mut Vec<Tensor>,
    limits: ContractionLimits,
    tolerance: f64,
    backend: &str,
    operation: &str,
) -> Result<Tensor> {
    let mut discarded = 0.0f64;
    let mut ranked = (usize::MAX, usize::MAX);
    loop {
        let plan = plan_with_restarts(tensors);
        if plan.peak <= limits.peak_cap {
            LAST_SLICE_COUNT.with(|count| count.set(1));
            LAST_DISCARDED.with(|weight| weight.set(discarded));
            return Ok(replay_plan(tensors, &plan));
        }
        if (plan.peak, plan.total) >= ranked {
            let summed = slice_to_fit(tensors, &plan, limits, backend, operation)?;
            LAST_DISCARDED.with(|weight| weight.set(discarded));
            return Ok(summed);
        }
        ranked = (plan.peak, plan.total);
        let Some(step) = truncate_one(tensors, &plan, limits.peak_cap, tolerance) else {
            let plan = plan_with_restarts(tensors);
            let summed = slice_to_fit(tensors, &plan, limits, backend, operation)?;
            LAST_DISCARDED.with(|weight| weight.set(discarded));
            return Ok(summed);
        };
        discarded += step;
    }
}

/// Replay `plan` until a pair would cross `cap`, then factor one of that
/// pair's operands and put both halves back among the live tensors.
///
/// Returns the relative squared weight the cut discarded, or `None` when
/// neither operand admits a cut that shrinks the blocked result. `tensors`
/// holds the network as far as the replay got either way, so the caller
/// replans rather than reusing `plan`.
fn truncate_one(
    tensors: &mut Vec<Tensor>,
    plan: &ContractionPlan,
    cap: usize,
    tolerance: f64,
) -> Option<f64> {
    let mut slots: Vec<Option<Tensor>> = std::mem::take(tensors).into_iter().map(Some).collect();
    let mut blocked = None;
    for &(i, j) in &plan.pairs {
        let a = slots[i].take().expect("planned pair is live");
        let b = slots[j].take().expect("planned pair is live");
        if contraction_result_size(&TensorMeta::of(&a), &TensorMeta::of(&b)) > cap {
            blocked = Some((a, b));
            break;
        }
        slots.push(Some(contract(&a, &b)));
    }

    let mut live: Vec<Tensor> = slots.into_iter().flatten().collect();
    let Some((a, b)) = blocked else {
        *tensors = live;
        return None;
    };

    let shared: SmallVec<[LegId; 6]> = a
        .legs
        .iter()
        .filter(|leg| b.legs.contains(leg))
        .copied()
        .collect();
    let bond = 1 + live
        .iter()
        .chain([&a, &b])
        .flat_map(|tensor| tensor.legs.iter().copied())
        .max()
        .expect("the blocked pair carries legs");
    let (first, second) = if free_size(&a, &shared) >= free_size(&b, &shared) {
        (a, b)
    } else {
        (b, a)
    };

    let cut = split_bond(&first, &shared, bond, tolerance)
        .map(|parts| (parts, false))
        .or_else(|| split_bond(&second, &shared, bond, tolerance).map(|parts| (parts, true)));

    let Some(((far, near, discarded), second_was_cut)) = cut else {
        live.push(first);
        live.push(second);
        *tensors = live;
        return None;
    };
    live.push(far);
    live.push(near);
    live.push(if second_was_cut { first } else { second });
    *tensors = live;
    Some(discarded)
}

/// Element count of the legs `tensor` does not share with its partner.
fn free_size(tensor: &Tensor, shared: &[LegId]) -> usize {
    tensor
        .legs
        .iter()
        .enumerate()
        .filter(|(_, leg)| !shared.contains(leg))
        .map(|(axis, _)| tensor.shape[axis])
        .product::<usize>()
        .max(1)
}

/// Factor `tensor` across the cut that puts the legs in `near` on one side,
/// joining the halves through a new `bond` truncated at `tolerance`.
///
/// Returns the far half, the near half carrying the singular values, and the
/// relative squared weight the cut discarded. `None` when the cut leaves one
/// side empty, or when the kept rank would not shrink the contraction the
/// caller is trying to fit, in which case holding the tensor whole is both
/// cheaper and exact.
fn split_bond(
    tensor: &Tensor,
    near: &[LegId],
    bond: LegId,
    tolerance: f64,
) -> Option<(Tensor, Tensor, f64)> {
    let far_axes: SmallVec<[usize; 6]> = (0..tensor.rank())
        .filter(|&axis| !near.contains(&tensor.legs[axis]))
        .collect();
    let near_axes: SmallVec<[usize; 6]> = (0..tensor.rank())
        .filter(|&axis| near.contains(&tensor.legs[axis]))
        .collect();
    if far_axes.is_empty() || near_axes.is_empty() {
        return None;
    }

    let rows: usize = far_axes.iter().map(|&axis| tensor.shape[axis]).product();
    let cols: usize = near_axes.iter().map(|&axis| tensor.shape[axis]).product();

    let mut perm: SmallVec<[usize; 6]> = far_axes.clone();
    perm.extend_from_slice(&near_axes);
    let ordered = if perm
        .iter()
        .enumerate()
        .all(|(axis, &source)| axis == source)
    {
        Cow::Borrowed(tensor)
    } else {
        Cow::Owned(transpose(tensor, &perm))
    };

    let zero = Complex64::new(0.0, 0.0);
    let mut column_major = vec![zero; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            column_major[col * rows + row] = ordered.data[row * cols + col];
        }
    }
    let factored = crate::backend::mps::svd(&column_major, rows, cols);

    let total: f64 = factored.s.iter().map(|value| value * value).sum();
    if total <= 0.0 {
        return None;
    }
    let chi = kept_rank(&factored.s, total * tolerance);
    if chi >= rows {
        return None;
    }
    let discarded = factored.s[chi..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        / total;

    let mut far_shape: SmallVec<[usize; 6]> =
        far_axes.iter().map(|&axis| tensor.shape[axis]).collect();
    let mut far_legs: SmallVec<[LegId; 6]> =
        far_axes.iter().map(|&axis| tensor.legs[axis]).collect();
    far_shape.push(chi);
    far_legs.push(bond);
    let mut far_data = vec![zero; rows * chi];
    for row in 0..rows {
        for rank in 0..chi {
            far_data[row * chi + rank] = factored.u[rank * rows + row];
        }
    }

    let mut near_shape: SmallVec<[usize; 6]> = smallvec::smallvec![chi];
    let mut near_legs: SmallVec<[LegId; 6]> = smallvec::smallvec![bond];
    for &axis in &near_axes {
        near_shape.push(tensor.shape[axis]);
        near_legs.push(tensor.legs[axis]);
    }
    let mut near_data = vec![zero; chi * cols];
    for rank in 0..chi {
        let value = factored.s[rank];
        for col in 0..cols {
            near_data[rank * cols + col] = factored.vt[rank * cols + col] * value;
        }
    }

    Some((
        Tensor {
            data: far_data,
            shape: far_shape,
            legs: far_legs,
        },
        Tensor {
            data: near_data,
            shape: near_shape,
            legs: near_legs,
        },
        discarded,
    ))
}

/// Shortest prefix of the descending `values` that leaves at most `budget` of
/// squared weight behind, and never fewer than one.
fn kept_rank(values: &[f64], budget: f64) -> usize {
    let mut discarded = 0.0f64;
    let mut kept = values.len();
    for (index, &value) in values.iter().enumerate().rev() {
        discarded += value * value;
        if discarded > budget {
            break;
        }
        kept = index;
    }
    kept.max(1)
}

/// Contract `tensors` along `plan`. Pairs the plan leaves uncontracted share
/// no leg and go to [`join_disjoint`].
fn replay_plan(tensors: &mut Vec<Tensor>, plan: &ContractionPlan) -> Tensor {
    let mut slots: Vec<Option<Tensor>> = std::mem::take(tensors).into_iter().map(Some).collect();
    for &(i, j) in &plan.pairs {
        let a_tensor = slots[i].take().expect("planned pair is live");
        let b_tensor = slots[j].take().expect("planned pair is live");
        debug_assert!(
            a_tensor.legs.iter().any(|leg| b_tensor.legs.contains(leg)),
            "planned pair shares a leg"
        );
        slots.push(Some(contract(&a_tensor, &b_tensor)));
    }

    join_disjoint(slots)
}

/// One sweep position's plan, with the fingerprint of the metadata it was
/// planned from.
struct CachedPlan {
    fingerprint: u64,
    plan: ContractionPlan,
}

/// FNV-1a fold of the tensor count, then each tensor's rank, shape, and leg
/// ids in order: everything the planner reads.
fn metadata_fingerprint(tensors: &[Tensor]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let fold = |hash: u64, word: usize| (hash ^ word as u64).wrapping_mul(PRIME);
    let mut hash = fold(OFFSET, tensors.len());
    for tensor in tensors {
        hash = fold(hash, tensor.rank());
        for &dim in &tensor.shape {
            hash = fold(hash, dim);
        }
        for &leg in &tensor.legs {
            hash = fold(hash, leg);
        }
    }
    hash
}

/// The plan in `slot` when its fingerprint matches `tensors`; otherwise plan
/// afresh, overwrite the slot, and return that.
fn cached_plan<'s>(tensors: &[Tensor], slot: &'s mut Option<CachedPlan>) -> &'s ContractionPlan {
    let fingerprint = metadata_fingerprint(tensors);
    if slot
        .as_ref()
        .is_none_or(|cached| cached.fingerprint != fingerprint)
    {
        *slot = Some(CachedPlan {
            fingerprint,
            plan: plan_with_restarts(tensors),
        });
    }
    &slot.as_ref().expect("slot filled above").plan
}

#[cfg(test)]
thread_local! {
    static PLANNER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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

    fn contract(self, limits: ContractionLimits) -> Result<f64> {
        if self.tensors.is_empty() {
            return Ok(1.0);
        }
        let plan = plan_with_restarts(&self.tensors);
        self.contract_on(&plan, limits)
    }

    fn contract_on(mut self, plan: &ContractionPlan, limits: ContractionLimits) -> Result<f64> {
        let result = contract_within(
            &mut self.tensors,
            plan,
            limits,
            "tensor_network_scalar",
            "scalar expectation",
        )?;
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
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied("TensorNetwork", label));
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. }
            | Instruction::Region(_) => {
                return Err(PrismError::BackendUnsupported {
                    backend: "tensor_network_scalar".to_string(),
                    operation: format!("non-unitary instruction {instruction:?}"),
                });
            }
        }
    }
    network.append_observable(pauli_terms)?;
    network.contract(ContractionLimits::from_env())
}

/// Largest greedy-tree intermediate, in elements, under which the `Auto`
/// expectation route takes the scalar path. Measured over 24 circuits: every
/// family the scalar path beat the statevector on stayed under it for every
/// observable, every family it lost to crossed it, and the bounded pass that
/// decides costs 1 to 22 ms.
pub(crate) const AUTO_EXPECTATION_PEAK_BOUND: usize = 1 << 12;

/// `<0| U^dag P U |0>` for every observable, each contracted on a greedy tree
/// that stays under `bound` elements. `None` when any observable's tree
/// crosses the bound, or the circuit holds an instruction the network cannot
/// append, so the caller falls back to a dense route. The plan the dry run
/// produced is the plan contracted, so nothing is planned twice.
pub(crate) fn bounded_expectations_zero_state(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    bound: usize,
) -> Option<Result<Vec<f64>>> {
    let circuit = crate::circuit::expand_qft_blocks(circuit);
    let mut planned = Vec::with_capacity(observables.len());
    for observable in observables {
        let mut network = ScalarExpectationNetwork::new(circuit.num_qubits);
        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { gate, targets } => network.append_gate(gate, targets).ok()?,
                Instruction::Barrier { .. } => {}
                _ => return None,
            }
        }
        if let Err(e) = network.append_observable(observable) {
            return Some(Err(e));
        }
        let slots: Vec<Option<TensorMeta>> = network
            .tensors
            .iter()
            .map(|t| Some(TensorMeta::of(t)))
            .collect();
        let plan = plan_pairs(slots, None, bound)?;
        planned.push((network, plan));
    }
    Some(
        planned
            .into_iter()
            .map(|(network, plan)| network.contract_on(&plan, ContractionLimits::from_env()))
            .collect(),
    )
}

/// Bench-visible wrapper over [`expectation_zero_state`]; not stable API.
#[cfg(feature = "bench-internal")]
pub fn scalar_expectation(circuit: &Circuit, pauli_terms: &[PauliTerm]) -> Result<f64> {
    expectation_zero_state(circuit, pauli_terms)
}

/// [`scalar_expectation`] with the peak cap, slice budget and truncation
/// tolerance supplied rather than read from the environment, so one process
/// can hold rows on both sides of the ceiling; not stable API.
#[cfg(feature = "bench-internal")]
pub fn scalar_expectation_capped(
    circuit: &Circuit,
    pauli_terms: &[PauliTerm],
    peak_cap: usize,
    slice_budget: usize,
    tolerance: Option<f64>,
) -> Result<f64> {
    let mut network = ScalarExpectationNetwork::new(circuit.num_qubits);
    for instruction in &circuit.instructions {
        let Instruction::Gate { gate, targets } = instruction else {
            continue;
        };
        network.append_gate(gate, targets)?;
    }
    network.append_observable(pauli_terms)?;
    network.contract(ContractionLimits {
        peak_cap,
        slice_budget,
        tolerance,
    })
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
        Gate::Unitary(data) => {
            check_arity(data.num_qubits())?;
            let dim = 1usize << data.num_qubits();
            let full: Vec<Vec<Complex64>> = data
                .matrix()
                .chunks(dim)
                .map(<[Complex64]>::to_vec)
                .collect();
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
    tolerance: Option<f64>,
    truncation_discarded: std::cell::Cell<f64>,
}

impl TensorNetworkBackend {
    pub fn new(seed: u64) -> Self {
        Self::build(seed, None)
    }

    /// A backend whose contractions may truncate an intermediate bond when
    /// the planned peak crosses the memory cap, discarding at most
    /// `tolerance` of a cut tensor's squared weight per cut.
    ///
    /// A tolerance of zero discards nothing and stays exact.
    ///
    /// # Panics
    ///
    /// When `tolerance` is negative or not finite.
    pub fn with_tolerance(seed: u64, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "tensor-network tolerance must be a finite fraction at or above zero, got {tolerance}"
        );
        Self::build(seed, Some(tolerance))
    }

    fn build(seed: u64, tolerance: Option<f64>) -> Self {
        Self {
            num_qubits: 0,
            tensors: Vec::new(),
            output_legs: Vec::new(),
            next_leg: 0,
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            tolerance,
            truncation_discarded: std::cell::Cell::new(0.0),
        }
    }

    /// Cumulative fraction of squared weight the contractions of this backend
    /// have discarded since [`Backend::init`].
    ///
    /// The total sums one relative discard per cut rather than measuring the
    /// final state, so a run that truncates heavily can carry it past 1,
    /// where the fidelity it implies clamps to zero and certifies nothing.
    ///
    /// [`Backend::init`]: crate::backend::Backend::init
    pub fn truncation_discarded(&self) -> f64 {
        self.truncation_discarded.get()
    }

    /// Limits every terminal of this backend contracts under.
    fn limits(&self) -> ContractionLimits {
        ContractionLimits::with_tolerance(self.tolerance)
    }

    /// Add what the contraction just finished discarded to the running total.
    fn book_truncation(&self) {
        self.truncation_discarded
            .set(self.truncation_discarded.get() + last_discarded());
    }

    fn fresh_leg(&mut self) -> LegId {
        let id = self.next_leg;
        self.next_leg += 1;
        id
    }

    /// Draw the outcome for `qubit` from its reduced density matrix and append
    /// the renormalizing projector, keeping the network in deferred form.
    ///
    /// The marginal contracts the doubled network, so no `2^n` vector is built
    /// and measurement carries no dense width ceiling. One rng draw per call,
    /// in program order, matching every other backend's outcome stream.
    fn collapse_qubit(&mut self, qubit: usize, reset: bool) -> Result<bool> {
        use rand::RngExt;

        let uniform = self.rng.random::<f64>();
        self.collapse_qubit_with(qubit, reset, uniform, None)
    }

    /// Collapse `qubit` with a caller-supplied uniform draw, so the native
    /// sampler can drive collapses from its own seeded stream without
    /// touching the run rng. `plan` is the sampler's cache slot for this
    /// position; `None` plans the marginal afresh.
    fn collapse_qubit_with(
        &mut self,
        qubit: usize,
        reset: bool,
        uniform: f64,
        plan: Option<&mut Option<CachedPlan>>,
    ) -> Result<bool> {
        let rho = self.marginal_1q(qubit, plan)?;
        let trace = (rho[0][0].re + rho[1][1].re).max(NORM_CLAMP_MIN);
        let prob_one = (rho[1][1].re / trace).clamp(0.0, 1.0);
        let outcome = uniform < prob_one;
        let inv_norm = crate::backend::measurement_inv_norm(outcome, prob_one);
        self.append_collapse(qubit, outcome, reset, inv_norm);
        Ok(outcome)
    }

    /// Draw one shot into `samples` by fixing qubits in index order, each bit
    /// from its conditioned marginal with the outcome projector absorbed.
    fn sample_one_shot(
        &mut self,
        rng: &mut ChaCha8Rng,
        shot: usize,
        samples: &mut BasisSamples,
        mut plans: Option<&mut [Option<CachedPlan>]>,
    ) -> Result<()> {
        use rand::RngExt;

        for qubit in 0..self.num_qubits {
            let uniform = rng.random::<f64>();
            let plan = plans.as_deref_mut().map(|plans| &mut plans[qubit]);
            if self.collapse_qubit_with(qubit, false, uniform, plan)? {
                samples.set(shot, qubit);
            }
        }
        Ok(())
    }

    /// Qubit-by-qubit conditional sampling, one doubled-network contraction
    /// per qubit per shot; the module docstring carries the cost trade.
    ///
    /// Each position's contraction is planned on the first shot and replayed
    /// on the rest: the projector is absorbed into its owner and leg ids
    /// restart with the state, so the doubled network at a position carries
    /// the same metadata in every shot. The cache lives for this call only,
    /// and a fingerprint of that metadata guards every replay.
    fn sample_native(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let mut plans: Vec<Option<CachedPlan>> = std::iter::repeat_with(|| None)
            .take(self.num_qubits)
            .collect();
        self.sample_sweep(num_shots, seed, Some(&mut plans))
    }

    /// The sweep behind [`Self::sample_native`], with the plan cache as a
    /// parameter so a reference run can go without one.
    ///
    /// A pre-flight plans the first marginal and reserves its peak
    /// intermediate as a feasibility proxy: later marginals open a different
    /// qubit and can plan a different tree, so the gate is heuristic, not a
    /// guarantee. The state is restored after every shot, errors included.
    fn sample_sweep(
        &mut self,
        num_shots: usize,
        seed: u64,
        mut plans: Option<&mut [Option<CachedPlan>]>,
    ) -> Result<BasisSamples> {
        let n = self.num_qubits;
        let mut samples = BasisSamples::new(num_shots, n);

        let (network, _, _) = self.double_for_partial_trace(0);
        let slots: Vec<Option<TensorMeta>> =
            network.iter().map(|t| Some(TensorMeta::of(t))).collect();
        let plan = plan_pairs(slots, None, usize::MAX).expect("unbounded pass completes");
        let mut probe: Vec<Complex64> = Vec::new();
        reserve_dense_output(&mut probe, plan.peak, self.name(), "native sampling")?;
        drop(probe);

        let tensors = self.tensors.clone();
        let output_legs = self.output_legs.clone();
        let next_leg = self.next_leg;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for shot in 0..num_shots {
            let drawn = self.sample_one_shot(&mut rng, shot, &mut samples, plans.as_deref_mut());
            self.tensors.clone_from(&tensors);
            self.output_legs.clone_from(&output_legs);
            self.next_leg = next_leg;
            drawn?;
        }
        Ok(samples)
    }

    /// Absorb the rank-2 tensor that keeps only `outcome` on `qubit`, scaled by
    /// `inv_norm`, into the tensor holding the qubit's output leg, mapping the
    /// kept branch to `|0>` when `reset` is set.
    ///
    /// Contracting into the owner instead of appending keeps the tensor count
    /// constant across measurements, so a measure or reset loop costs one
    /// doubled contraction per event rather than growing the network it
    /// contracts.
    fn append_collapse(&mut self, qubit: usize, outcome: bool, reset: bool, inv_norm: f64) {
        let in_leg = self.output_legs[qubit];
        let out_leg = self.fresh_leg();
        let zero = Complex64::new(0.0, 0.0);
        let scale = Complex64::new(inv_norm, 0.0);

        let in_idx = usize::from(outcome);
        let out_idx = if reset { 0 } else { in_idx };
        let mut data = vec![zero; 4];
        data[out_idx * 2 + in_idx] = scale;

        let projector = Tensor {
            data,
            shape: smallvec::smallvec![2, 2],
            legs: smallvec::smallvec![out_leg, in_leg],
        };
        let owner = self
            .tensors
            .iter()
            .position(|t| t.legs.contains(&in_leg))
            .expect("every output leg has an owner");
        self.tensors[owner] = contract(&self.tensors[owner], &projector);
        self.output_legs[qubit] = out_leg;
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
        self.collapse_qubit(qubit, true)?;
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

    /// Contract `tr_{q != qubit} |psi><psi|` to its four entries, replaying
    /// or filling `plan` when the caller holds a cache slot.
    fn marginal_1q(
        &self,
        qubit: usize,
        plan: Option<&mut Option<CachedPlan>>,
    ) -> Result<[[Complex64; 2]; 2]> {
        let (mut network, ket_leg, bra_leg) = self.double_for_partial_trace(qubit);
        let operation = "reduced density matrix";
        let rho = match plan {
            Some(slot) => {
                let plan = cached_plan(&network, slot);
                contract_within(&mut network, plan, self.limits(), self.name(), operation)?
            }
            None => greedy_contract(&mut network, self.limits(), self.name(), operation)?,
        };
        self.book_truncation();

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

    /// Contract `<psi|P|psi>` for one joint Pauli observable, unnormalized.
    ///
    /// Qubits the observable omits carry identity, and an identity factor is a
    /// leg closed directly against its twin rather than a tensor appended, which
    /// keeps `n - k` tensors out of the network for a weight-`k` observable.
    fn contract_pauli_sandwich(&self, axes: &[Option<PauliAxis>]) -> Result<f64> {
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

        let result = greedy_contract(
            &mut network,
            self.limits(),
            self.name(),
            "pauli expectation",
        )?;
        self.book_truncation();
        debug_assert!(result.legs.is_empty(), "sandwich leaves every leg paired");
        Ok(result.data[0].re)
    }

    /// Contract the full network and return the amplitude vector in
    /// computational basis order.
    fn contract_to_statevector(&self) -> Result<Vec<Complex64>> {
        dense_statevector_len(self.name(), "contraction", self.num_qubits)?;

        let mut tensors = self.tensors.clone();
        let result = greedy_contract(&mut tensors, self.limits(), self.name(), "contraction")?;
        self.book_truncation();

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

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::TensorNetwork
    }

    /// `Exact` without a tolerance, whatever the contraction did, since every
    /// other route to the cap preserves the value. A tolerance reports
    /// `Approximate` whether or not this run cut anything, since the route
    /// could have.
    ///
    /// The bound is 1 minus the summed per-cut relative discarded weights: a
    /// first-order truncation estimate, not a certificate. Errors compound
    /// across cuts, and summing the weights understates the compounded error,
    /// since the strict bound on the infidelity is the square of the summed
    /// square roots. The two agree only when a single cut truncates. A
    /// doubled contraction, which `pauli_expectations` and
    /// `reduced_density_matrix_1q` both run, cuts in the doubled space, so
    /// what the bound describes there is the quadratic form rather than the
    /// state.
    fn exactness(&self) -> crate::sim::Exactness {
        match self.tolerance {
            Some(tolerance) if tolerance > 0.0 => crate::sim::Exactness::Approximate {
                fidelity_lower_bound: Some((1.0 - self.truncation_discarded.get()).max(0.0)),
            },
            _ => crate::sim::Exactness::Exact,
        }
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.tensors = Vec::new();
        self.next_leg = 0;
        self.truncation_discarded.set(0.0);
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
                let outcome = self.collapse_qubit(*qubit, false)?;
                self.classical_bits[*classical_bit] = outcome;
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit)?;
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied("TensorNetwork", label));
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

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Draw shots from the dense distribution below the ceiling and from the
    /// qubit-by-qubit conditional sweep past it.
    ///
    /// One full-distribution contraction plus a draw per shot undercuts the
    /// per-shot sweep by a measured 66x at 16 qubits and 32 shots on the
    /// chain shape, so the dense arm answers wherever `probabilities()` can;
    /// the sweep is the route that exists past that ceiling.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let n = self.num_qubits;
        if n == 0 || num_shots == 0 {
            return Ok(BasisSamples::new(num_shots, n));
        }

        if tensor_probability_len(self.name(), n).is_ok() {
            use rand::RngExt;

            let mut samples = BasisSamples::new(num_shots, n);
            let cdf = crate::sim::shots::build_cdf(&self.probabilities()?);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for shot in 0..num_shots {
                let r = rng.random::<f64>();
                samples.set_index(shot, crate::sim::shots::sample_from_cdf(&cdf, r));
            }
            return Ok(samples);
        }

        self.sample_native(num_shots, seed)
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
        self.marginal_1q(qubit, None)
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
        let norm_sq = self.contract_pauli_sandwich(&axes)?;

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
            expectations.push(self.contract_pauli_sandwich(&axes)? / norm_sq);
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

    fn scalar_network(circuit: &Circuit, terms: &[PauliTerm]) -> ScalarExpectationNetwork {
        let mut network = ScalarExpectationNetwork::new(circuit.num_qubits);
        for instruction in &circuit.instructions {
            let Instruction::Gate { gate, targets } = instruction else {
                continue;
            };
            network.append_gate(gate, targets).unwrap();
        }
        network.append_observable(terms).unwrap();
        network
    }

    // hardware_efficient_ansatz(30, 7) is a recorded case of the greedy tree
    // peaking at 16.8M elements, past RESTART_PEAK_THRESHOLD, so the restart
    // arm runs. Planning walks metadata only, so no contraction executes here.
    #[test]
    fn test_plan_restarts_deterministic_and_never_worse() {
        let circuit = crate::circuits::hardware_efficient_ansatz(30, 7, 42);
        let terms = [PauliTerm::z(0), PauliTerm::z(15)];
        let network = scalar_network(&circuit, &terms);
        let slots: Vec<Option<TensorMeta>> = network
            .tensors
            .iter()
            .map(|t| Some(TensorMeta::of(t)))
            .collect();

        let greedy = plan_pairs(slots, None, usize::MAX).unwrap();
        assert!(
            greedy.peak >= RESTART_PEAK_THRESHOLD,
            "fixture no longer reaches the restart arm: greedy peak {}",
            greedy.peak
        );

        let best_a = plan_with_restarts(&network.tensors);
        let best_b = plan_with_restarts(&network.tensors);
        assert_eq!(best_a.pairs, best_b.pairs);
        assert!(best_a.peak <= greedy.peak);
        println!(
            "greedy peak {} restart peak {} ({} pairs)",
            greedy.peak,
            best_a.peak,
            best_a.pairs.len()
        );
    }

    #[test]
    fn test_slices_sum_to_the_unsliced_contraction() {
        let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
        let terms = [PauliTerm::z(0), PauliTerm::z(6)];
        let exact = scalar_network(&circuit, &terms)
            .contract(ContractionLimits::from_env())
            .unwrap();

        let network = scalar_network(&circuit, &terms);
        let plan = plan_with_restarts(&network.tensors);
        let limits = ContractionLimits {
            peak_cap: 1 << 8,
            slice_budget: 1 << 10,
            tolerance: None,
        };
        assert!(
            plan.peak > limits.peak_cap,
            "fixture plans {} elements, already under the cap",
            plan.peak
        );

        let metas: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();
        let slice = choose_slices(&metas, &plan, limits).expect("the fixture slices");
        assert!(slice.count > 1, "the fixture did not slice");
        assert!(slice.peak <= limits.peak_cap);
        // One slice per wave, three per wave, and every slice in one wave.
        for cap in [slice.peak, slice.peak * 3, usize::MAX] {
            let summed = contract_slices(&network.tensors, &plan, &slice, cap);
            assert!(
                (summed.data[0].re - exact).abs() < 1e-10,
                "{} vs {exact} over {} slices at cap {cap}",
                summed.data[0].re,
                slice.count
            );
        }
    }

    #[test]
    fn test_slice_choice_refuses_what_the_budget_cannot_reach() {
        let circuit = crate::circuits::hardware_efficient_ansatz(12, 4, 42);
        let terms = [PauliTerm::z(0), PauliTerm::z(6)];
        let network = scalar_network(&circuit, &terms);
        let plan = plan_with_restarts(&network.tensors);
        let metas: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();

        let limits = ContractionLimits {
            peak_cap: 1 << 8,
            slice_budget: 2,
            tolerance: None,
        };
        assert!(choose_slices(&metas, &plan, limits).is_none());
    }

    #[test]
    fn test_pinned_axes_keep_their_extent_at_one() {
        let tensor = Tensor {
            data: (0..6).map(|i| Complex64::new(i as f64, 0.0)).collect(),
            shape: smallvec::smallvec![2, 3],
            legs: smallvec::smallvec![7, 8],
        };

        let rows = pin_axes(&tensor, &[(0, 1)]);
        assert_eq!(rows.shape.as_slice(), &[1, 3]);
        assert_eq!(rows.data, vec_of(&[3.0, 4.0, 5.0]));

        let cols = pin_axes(&tensor, &[(1, 2)]);
        assert_eq!(cols.shape.as_slice(), &[2, 1]);
        assert_eq!(cols.data, vec_of(&[2.0, 5.0]));

        let one = pin_axes(&tensor, &[(0, 1), (1, 0)]);
        assert_eq!(one.shape.as_slice(), &[1, 1]);
        assert_eq!(one.data, vec_of(&[3.0]));
    }

    #[test]
    fn test_kept_rank_drops_only_what_the_budget_covers() {
        // Squared weights 1, 0.25, 0.01 and 0.0001, and the budget is an
        // absolute squared weight, not a fraction.
        let values = [1.0, 0.5, 0.1, 0.01];
        assert_eq!(kept_rank(&values, 0.0), 4);
        assert_eq!(kept_rank(&values, 0.0001), 3);
        assert_eq!(kept_rank(&values, 0.011), 2);
        assert_eq!(kept_rank(&values, 10.0), 1, "never fewer than one");
    }

    // A cut at zero tolerance keeps the full rank, so contracting the halves
    // back has to reproduce the tensor: the split is exact and the only
    // thing a tolerance buys is a shorter bond.
    #[test]
    fn test_a_zero_tolerance_cut_reconstructs_the_tensor() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let tensor = Tensor {
            data: (0..32)
                .map(|_| {
                    use rand::RngExt;
                    Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5)
                })
                .collect(),
            shape: smallvec::smallvec![2, 2, 2, 4],
            legs: smallvec::smallvec![10, 11, 12, 13],
        };

        let (far, near, discarded) = split_bond(&tensor, &[13], 99, 0.0).expect("the cut applies");
        assert_eq!(discarded, 0.0);
        assert!(far.legs.contains(&99) && near.legs.contains(&99));

        let rebuilt = contract(&far, &near);
        let perm: SmallVec<[usize; 6]> = tensor
            .legs
            .iter()
            .map(|leg| {
                rebuilt
                    .legs
                    .iter()
                    .position(|held| held == leg)
                    .expect("the halves carry every original leg")
            })
            .collect();
        let ordered = transpose(&rebuilt, &perm);
        assert_eq!(ordered.shape, tensor.shape);
        for (got, want) in ordered.data.iter().zip(&tensor.data) {
            assert!((got - want).norm() < 1e-12, "{got} vs {want}");
        }
    }

    // A cut that cannot shrink the blocked contraction declines, so the
    // caller holds the tensor whole rather than paying for an SVD and two
    // factors that buy nothing.
    #[test]
    fn test_a_cut_that_cannot_shrink_declines() {
        let tensor = Tensor {
            data: vec![Complex64::new(1.0, 0.0); 8],
            shape: smallvec::smallvec![2, 4],
            legs: smallvec::smallvec![10, 11],
        };
        assert!(
            split_bond(&tensor, &[10, 11], 99, 0.5).is_none(),
            "no far side"
        );
        assert!(split_bond(&tensor, &[], 99, 0.5).is_none(), "no near side");
    }

    fn vec_of(values: &[f64]) -> Vec<Complex64> {
        values.iter().map(|&v| Complex64::new(v, 0.0)).collect()
    }

    // Every noise stream must land on the same scalar: replay correctness for
    // arbitrary plan orders is the risk the planner split introduces.
    #[test]
    fn test_noisy_plans_execute_to_the_same_scalar() {
        let circuit = crate::circuits::hardware_efficient_ansatz(8, 3, 42);
        let terms = [PauliTerm::z(0), PauliTerm::x(4)];
        let expected = expectation_zero_state(&circuit, &terms).unwrap();

        for seed in 0..5u64 {
            let network = scalar_network(&circuit, &terms);
            let slots: Vec<Option<TensorMeta>> = network
                .tensors
                .iter()
                .map(|t| Some(TensorMeta::of(t)))
                .collect();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let plan = plan_pairs(slots, Some((&mut rng, 1.0)), usize::MAX).unwrap();

            let mut slots: Vec<Option<Tensor>> = network.tensors.into_iter().map(Some).collect();
            for &(i, j) in &plan.pairs {
                let a = slots[i].take().unwrap();
                let b = slots[j].take().unwrap();
                slots.push(Some(contract(&a, &b)));
            }
            let result = join_disjoint(slots);
            assert_eq!(result.data.len(), 1);
            assert!(
                (result.data[0].re - expected).abs() < EPS,
                "seed {seed}: {} vs {expected}",
                result.data[0].re
            );
        }
    }

    // Mid-circuit measure and reset must agree with the statevector on the
    // seeded outcome stream, not just on the marginals, and on the final
    // distribution after further gates.
    #[test]
    fn test_mid_circuit_measure_reset_matches_statevector() {
        let mut c = Circuit::new(5, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Ry(0.7), &[2]);
        c.add_measure(1, 0);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::H, &[1]);
        c.add_reset(0);
        c.add_gate(Gate::Cx, &[0, 3]);
        c.add_measure(2, 1);
        c.add_gate(Gate::Ry(0.3), &[4]);

        for seed in [42u64, 7, 12345] {
            let mut sv = StatevectorBackend::new(seed);
            sv.init(5, 2).unwrap();
            let mut tn = TensorNetworkBackend::new(seed);
            tn.init(5, 2).unwrap();
            for inst in &c.instructions {
                sv.apply(inst).unwrap();
                tn.apply(inst).unwrap();
            }
            assert_eq!(
                tn.classical_results(),
                sv.classical_results(),
                "seed {seed}"
            );
            assert_probs_close(&tn.probabilities().unwrap(), &sv.probabilities().unwrap());
        }
    }

    // A measurement past the dense query ceiling must succeed and keep the
    // deferred form: no rank-n tensor, and the network still answers
    // expectation queries that never build a 2^n vector.
    #[test]
    fn test_measurement_past_dense_ceiling_keeps_network() {
        let n = 30;
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(n, 1).unwrap();
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
        tn.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: smallvec::smallvec![1, 2],
        })
        .unwrap();
        tn.apply(&Instruction::Measure {
            qubit: 1,
            classical_bit: 0,
        })
        .unwrap();

        assert!(tn.tensors.len() > 1);
        assert!(tn.tensors.iter().all(|t| t.rank() < 6));

        let outcome = tn.classical_results()[0];
        let expected = if outcome { -1.0 } else { 1.0 };
        let exps = tn
            .pauli_expectations(&[vec![PauliTerm::z(0)], vec![PauliTerm::z(2)]])
            .unwrap();
        assert!(
            (exps[0] - expected).abs() < EPS,
            "{} vs {expected}",
            exps[0]
        );
        assert!(
            (exps[1] - expected).abs() < EPS,
            "{} vs {expected}",
            exps[1]
        );
    }

    // Joint distribution check for the conditional sweep against the dense
    // route with per-outcome binomial bands, plus the contract that sampling
    // leaves the state untouched. Calls the sweep directly: the public path
    // takes the dense arm at this width.
    #[test]
    fn test_native_sampling_matches_dense_distribution() {
        let circuit = crate::circuits::cz_chain_circuit(6, 3, 42);
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(6, 0).unwrap();
        for inst in &circuit.instructions {
            tn.apply(inst).unwrap();
        }
        let probs_before = tn.probabilities().unwrap();

        let shots = 2000usize;
        let samples = tn.sample_native(shots, 42).unwrap();

        let mut counts = vec![0usize; 1 << 6];
        for shot in 0..shots {
            let mut index = 0usize;
            for q in 0..6 {
                if samples.bit(shot, q) {
                    index |= 1 << q;
                }
            }
            counts[index] += 1;
        }
        for (index, (&count, &p)) in counts.iter().zip(&probs_before).enumerate() {
            let freq = count as f64 / shots as f64;
            let sigma = (p * (1.0 - p) / shots as f64).sqrt().max(1e-3);
            assert!(
                (freq - p).abs() < 6.0 * sigma,
                "outcome {index}: {freq} vs {p}"
            );
        }

        assert_probs_close(&tn.probabilities().unwrap(), &probs_before);
    }

    // Per-qubit counts convergence where the dense route cannot answer at
    // all: 30 independent rotations, marginals known analytically.
    #[test]
    fn test_native_sampling_past_dense_ceiling() {
        let n = 30;
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(n, 0).unwrap();
        for q in 0..n {
            tn.apply(&Instruction::Gate {
                gate: Gate::Ry(0.9),
                targets: smallvec::smallvec![q],
            })
            .unwrap();
        }

        let shots = 500usize;
        let samples = tn.sample_basis_states(shots, 42).unwrap();
        let p_one = (0.45f64).sin().powi(2);
        let sigma = (p_one * (1.0 - p_one) / shots as f64).sqrt();
        for q in [0usize, 7, 15, 29] {
            let count = (0..shots).filter(|&shot| samples.bit(shot, q)).count();
            let freq = count as f64 / shots as f64;
            assert!(
                (freq - p_one).abs() < 5.0 * sigma,
                "qubit {q}: {freq} vs {p_one}"
            );
        }
    }

    fn loaded_backend(circuit: &Circuit) -> TensorNetworkBackend {
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(circuit.num_qubits, 0).unwrap();
        for inst in &circuit.instructions {
            tn.apply(inst).unwrap();
        }
        tn
    }

    fn planner_calls() -> usize {
        PLANNER_CALLS.with(|calls| calls.get())
    }

    fn assert_plan_cache_transparent(circuit: &Circuit, shots: usize) {
        let mut tn = loaded_backend(circuit);
        let cached = tn.sample_native(shots, 42).unwrap();
        let uncached = tn.sample_sweep(shots, 42, None).unwrap();
        assert_eq!(cached.words, uncached.words);
    }

    #[test]
    fn test_plan_cache_shots_match_uncached_sweep_on_chain() {
        assert_plan_cache_transparent(&crate::circuits::cz_chain_circuit(12, 4, 42), 8);
    }

    #[test]
    fn test_plan_cache_shots_match_uncached_sweep_on_random_circuit() {
        assert_plan_cache_transparent(&crate::circuits::random_circuit(8, 5, 42), 8);
    }

    #[test]
    fn test_plan_cache_recomputes_on_fingerprint_mismatch() {
        let circuit = crate::circuits::cz_chain_circuit(8, 3, 42);
        let mut tn = loaded_backend(&circuit);
        let mut plans: Vec<Option<CachedPlan>> = std::iter::repeat_with(|| None).take(8).collect();
        tn.sample_sweep(1, 42, Some(&mut plans)).unwrap();
        let genuine = plans[3].as_ref().unwrap().fingerprint;
        plans[3].as_mut().unwrap().fingerprint = !genuine;

        let before = planner_calls();
        tn.sample_sweep(1, 42, Some(&mut plans)).unwrap();
        assert_eq!(planner_calls() - before, 1);
        assert_eq!(plans[3].as_ref().unwrap().fingerprint, genuine);
    }

    fn leg_network(legs: &[[LegId; 2]]) -> Vec<Tensor> {
        legs.iter()
            .map(|pair| Tensor {
                data: vec![Complex64::new(1.0, 0.0); 4],
                shape: smallvec::smallvec![2, 2],
                legs: pair.iter().copied().collect(),
            })
            .collect()
    }

    #[test]
    fn test_cached_plan_replans_when_only_leg_ids_differ() {
        let first = leg_network(&[[0, 1], [1, 2]]);
        let second = leg_network(&[[0, 1], [1, 3]]);
        let mut slot = None;
        let before = planner_calls();
        cached_plan(&first, &mut slot);
        let stored = slot.as_ref().unwrap().fingerprint;
        cached_plan(&first, &mut slot);
        assert_eq!(planner_calls() - before, 1);
        cached_plan(&second, &mut slot);
        assert_eq!(planner_calls() - before, 2);
        assert_ne!(slot.as_ref().unwrap().fingerprint, stored);
    }

    #[test]
    fn test_plan_cache_plans_each_position_once() {
        let n = 12;
        let mut tn = loaded_backend(&crate::circuits::cz_chain_circuit(n, 4, 42));
        let before = planner_calls();
        tn.sample_native(8, 42).unwrap();
        assert_eq!(planner_calls() - before, n);
    }

    #[test]
    fn test_sample_basis_states_repeats_from_the_seed() {
        let circuit = crate::circuits::cz_chain_circuit(6, 3, 42);
        let mut tn = TensorNetworkBackend::new(42);
        tn.init(6, 0).unwrap();
        for inst in &circuit.instructions {
            tn.apply(inst).unwrap();
        }

        let first = tn.sample_basis_states(64, 42).unwrap();
        let second = tn.sample_basis_states(64, 42).unwrap();
        let other = tn.sample_basis_states(64, 43).unwrap();

        let bits = |s: &BasisSamples| -> Vec<bool> {
            (0..64)
                .flat_map(|shot| (0..6).map(move |q| (shot, q)))
                .map(|(shot, q)| s.bit(shot, q))
                .collect()
        };
        assert_eq!(bits(&first), bits(&second));
        assert_ne!(bits(&first), bits(&other));
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
