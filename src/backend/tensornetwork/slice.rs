//! Index slicing, bounded truncation and plan caching at the memory ceiling.

use std::borrow::Cow;

use num_complex::Complex64;
use smallvec::SmallVec;

use crate::backend::tensor_peak_error;
use crate::error::Result;
use crate::limits::{LimitSource, ResolvedLimits};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::LegId;
use super::plan::{
    ContractionPlan, TensorMeta, contract_meta, contraction_result_size, join_disjoint,
    plan_with_restarts,
};
use super::tensor::{BufferPool, Tensor, contract, contract_with, transpose};

/// Peak-intermediate cap, slice budget and truncation tolerance one
/// contraction runs under.
///
/// `tolerance` is the per-cut relative squared weight a bounded contraction
/// may discard; `None` keeps the contraction exact.
#[derive(Clone, Copy)]
pub(super) struct ContractionLimits {
    pub(super) peak_cap: usize,
    pub(super) peak_source: LimitSource,
    pub(super) slice_budget: usize,
    pub(super) tolerance: Option<f64>,
}

impl ContractionLimits {
    pub(super) fn new(limits: &ResolvedLimits, tolerance: Option<f64>) -> Self {
        Self {
            peak_cap: limits.tn_peak_elements(),
            peak_source: limits.tn_peak_qubits.source,
            slice_budget: limits.tn_slices.value,
            tolerance,
        }
    }
}

/// Legs fixed for a sliced contraction, in the order they were chosen.
///
/// `count` is the product of `dims`: one contraction per assignment. `peak`
/// is the largest intermediate one slice plans, in elements.
pub(super) struct SlicePlan {
    legs: SmallVec<[LegId; 4]>,
    dims: SmallVec<[usize; 4]>,
    pub(super) count: usize,
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    pub(super) peak: usize,
}

thread_local! {
    pub(super) static LAST_SLICE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(1) };
    pub(super) static LAST_DISCARDED: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Slices the most recent contraction on this thread summed over, 1 when it
/// ran whole. Diagnostic for tests and tuning; not stable API.
#[doc(hidden)]
pub fn last_slice_count() -> usize {
    LAST_SLICE_COUNT.with(std::cell::Cell::get)
}

/// Relative squared weight the most recent contraction on this thread
/// discarded, summed over its cuts.
pub(super) fn last_discarded() -> f64 {
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
pub(super) fn choose_slices(
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
pub(super) fn pin_axes(tensor: &Tensor, pinned: &[(usize, usize)]) -> Tensor {
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
pub(super) fn contract_slices(
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
pub(super) fn contract_within(
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
            limits.peak_source,
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
pub(super) fn split_bond(
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
pub(super) fn kept_rank(values: &[f64], budget: f64) -> usize {
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
pub(super) fn replay_plan(tensors: &mut Vec<Tensor>, plan: &ContractionPlan) -> Tensor {
    let mut slots: Vec<Option<Tensor>> = std::mem::take(tensors).into_iter().map(Some).collect();
    let mut pool = BufferPool::default();
    for &(i, j) in &plan.pairs {
        let a_tensor = slots[i].take().expect("planned pair is live");
        let b_tensor = slots[j].take().expect("planned pair is live");
        debug_assert!(
            a_tensor.legs.iter().any(|leg| b_tensor.legs.contains(leg)),
            "planned pair shares a leg"
        );
        slots.push(Some(contract_with(&a_tensor, &b_tensor, &mut pool)));
        pool.give(a_tensor.data);
        pool.give(b_tensor.data);
    }

    join_disjoint(slots, &mut pool)
}

/// One sweep position's plan, with the fingerprint of the metadata it was
/// planned from.
pub(super) struct CachedPlan {
    pub(super) fingerprint: u64,
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
pub(super) fn cached_plan<'s>(
    tensors: &[Tensor],
    slot: &'s mut Option<CachedPlan>,
) -> &'s ContractionPlan {
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
