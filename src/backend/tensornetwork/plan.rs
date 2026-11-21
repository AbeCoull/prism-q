//! Contraction planning over tensor shapes and legs.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::error::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::LegId;
use super::slice::{ContractionLimits, contract_within};
use super::tensor::{BufferPool, Tensor, contract_with};

/// Shape and legs of a tensor, all the planner reads.
#[derive(Clone)]
pub(super) struct TensorMeta {
    pub(super) shape: SmallVec<[usize; 6]>,
    pub(super) legs: SmallVec<[LegId; 6]>,
}

impl TensorMeta {
    pub(super) fn of(tensor: &Tensor) -> Self {
        Self {
            shape: SmallVec::from_slice(&tensor.shape),
            legs: SmallVec::from_slice(&tensor.legs),
        }
    }

    pub(super) fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }
}

/// Pair order for one contraction, with the two counts plans are ranked by.
///
/// `pairs` holds slot indices, inputs first, each result appended at the next
/// index. `peak` is the largest result element count; `total` sums them and
/// breaks peak ties.
pub(super) struct ContractionPlan {
    pub(super) pairs: Vec<(usize, usize)>,
    pub(super) peak: usize,
    pub(super) total: usize,
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
pub(super) const RESTART_PEAK_THRESHOLD: usize = 1 << 22;

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

pub(super) fn contraction_result_size(a: &TensorMeta, b: &TensorMeta) -> usize {
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
/// matching the result [`contract`](super::tensor::contract) builds for the same pair.
pub(super) fn contract_meta(a: &TensorMeta, b: &TensorMeta) -> TensorMeta {
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

/// Noisy-pass candidates as `(key, higher slot, lower slot)`, reversed into
/// [`PairQueue`] order.
type PairHeap = BinaryHeap<Reverse<(u64, Reverse<usize>, Reverse<usize>)>>;

/// Candidate pairs of one planning pass, popped by size key, then by the
/// higher slot id descending, then the lower.
///
/// The key is the result element count, scaled by Gumbel noise on a noisy
/// planning pass. Newest-first on ties keeps the contraction on its frontier
/// instead of letting fresh tensors accumulate legs. Reversing the tie-break
/// raises peak intermediates 16x on `hardware_efficient_ansatz(50, 3)`.
///
/// The exact pass needs no heap. Each pair is queued when its higher slot
/// fills, so the higher slot never decreases from one push to the next, and
/// one slot's pairs go in ascending lower slot. A stack of `(lower, higher)`
/// pairs per size, sizes ascending, therefore holds each size's next pair on
/// top, and pops come out in the heap's order without its sift. Stacks below
/// `lowest` are empty.
enum PairQueue<'r> {
    Exact {
        sizes: Vec<(usize, Vec<(usize, usize)>)>,
        lowest: usize,
    },
    Noisy {
        heap: PairHeap,
        rng: &'r mut ChaCha8Rng,
        temperature: f64,
    },
}

/// Compute the noisy size key for one candidate pair.
///
/// Multiplies by `2^(temperature * gumbel)` and compares the result through its
/// bit pattern, which orders positive floats numerically.
fn pair_key(cost: usize, rng: &mut ChaCha8Rng, temperature: f64) -> u64 {
    use rand::RngExt;
    let uniform: f64 = rng.random::<f64>();
    let gumbel = -(-uniform.max(f64::MIN_POSITIVE).ln()).ln();
    ((cost as f64) * (temperature * gumbel).exp2()).to_bits()
}

impl PairQueue<'_> {
    /// Queue `slot` against every slot in `others`, each lower than `slot` and
    /// listed in the order the legs were walked.
    ///
    /// Each entry carries the extent of the leg it was found through, so the
    /// extents of one slot's entries multiply to the shared size and the result
    /// size follows from the two element counts without walking either leg list.
    ///
    /// The noisy arm draws one key per entry, in that order and twice for a pair
    /// sharing two legs. The exact arm drops the repeat, which would carry the
    /// same key.
    fn push_pairs(
        &mut self,
        elements: &[usize],
        slot: usize,
        others: &mut SmallVec<[(usize, usize); 8]>,
    ) {
        let cost =
            |other: usize, shared: usize| (elements[slot] / shared) * (elements[other] / shared);
        match self {
            Self::Exact { sizes, lowest } => {
                others.sort_unstable_by_key(|&(other, _)| other);
                let mut rest = &others[..];
                while let Some(&(other, extent)) = rest.first() {
                    let mut shared = extent;
                    rest = &rest[1..];
                    while let Some(&(_, extent)) = rest.first().filter(|&&(next, _)| next == other)
                    {
                        shared *= extent;
                        rest = &rest[1..];
                    }
                    let size = cost(other, shared);
                    let at = match sizes.binary_search_by_key(&size, |&(key, _)| key) {
                        Ok(at) => at,
                        Err(at) => {
                            sizes.insert(at, (size, Vec::new()));
                            at
                        }
                    };
                    sizes[at].1.push((other, slot));
                    *lowest = (*lowest).min(at);
                }
            }
            Self::Noisy {
                heap,
                rng,
                temperature,
            } => {
                for &(other, _) in others.iter() {
                    let shared = others
                        .iter()
                        .filter(|&&(held, _)| held == other)
                        .map(|&(_, extent)| extent)
                        .product();
                    let key = pair_key(cost(other, shared), rng, *temperature);
                    heap.push(Reverse((key, Reverse(slot), Reverse(other))));
                }
            }
        }
    }

    /// Pop the cheapest queued pair whose members are both still live, lower
    /// slot first.
    ///
    /// Returns `None` once no two live tensors share a leg.
    fn pop_live(&mut self, live: &[bool]) -> Option<(usize, usize)> {
        let live = |i: usize, j: usize| live[i] && live[j];
        match self {
            Self::Exact { sizes, lowest } => {
                while let Some((_, stack)) = sizes.get_mut(*lowest) {
                    match stack.pop() {
                        Some((i, j)) if live(i, j) => return Some((i, j)),
                        Some(_) => {}
                        None => *lowest += 1,
                    }
                }
                None
            }
            Self::Noisy { heap, .. } => {
                while let Some(Reverse((_, Reverse(j), Reverse(i)))) = heap.pop() {
                    if live(i, j) {
                        return Some((i, j));
                    }
                }
                None
            }
        }
    }
}

/// Record `slot` against each of its legs and queue it against every live tensor
/// already sharing one.
///
/// Each leg keeps two holder entries, and `slot` takes the first that is empty
/// or names a contracted tensor. The `held != slot` test keeps a tensor
/// carrying one leg twice from queuing against itself. The backend builds such
/// a tensor only from a gate that names one qubit twice, which is outside its
/// contract: every network built from distinct targets carries each leg at most
/// once per tensor.
///
/// # Panics
///
/// If a leg already has two live holders: contraction is pairwise, so a third
/// tensor on one leg has no meaning.
fn queue_slot_pairs(
    meta: &TensorMeta,
    live: &[bool],
    elements: &[usize],
    slot: usize,
    leg_holders: &mut [[usize; 2]],
    queue: &mut PairQueue,
) {
    let mut others: SmallVec<[(usize, usize); 8]> = SmallVec::new();
    for (&leg, &extent) in meta.legs.iter().zip(&meta.shape) {
        let mut placed = false;
        for held in &mut leg_holders[leg] {
            if *held != slot && live.get(*held).is_some_and(|&alive| alive) {
                others.push((*held, extent));
            } else if !placed {
                *held = slot;
                placed = true;
            }
        }
        assert!(placed, "a leg joins at most two tensors");
    }
    queue.push_pairs(elements, slot, &mut others);
}

/// Run one greedy planning pass over the metadata, deterministic when `noise`
/// is `None` and Gumbel-perturbed otherwise.
///
/// Returns `None` as soon as a result exceeds `abort_above` elements: the pass
/// can no longer beat the plan holding that peak, and abandoning it keeps a
/// failed restart from paying for a full walk. Peak ties complete, since they
/// can still win on `total`. Also `None` once the summed elements pass
/// `total_budget`.
pub(super) fn plan_pairs(
    mut slots: Vec<TensorMeta>,
    noise: Option<(&mut ChaCha8Rng, f64)>,
    abort_above: usize,
    total_budget: usize,
) -> Option<ContractionPlan> {
    let legs = slots
        .iter()
        .flat_map(|meta| meta.legs.iter().copied())
        .max()
        .map_or(0, |leg| leg + 1);
    let mut leg_holders = vec![[usize::MAX; 2]; legs];
    let mut queue = match noise {
        None => PairQueue::Exact {
            sizes: Vec::new(),
            lowest: 0,
        },
        Some((rng, temperature)) => PairQueue::Noisy {
            heap: BinaryHeap::new(),
            rng,
            temperature,
        },
    };

    let mut live = vec![true; slots.len()];
    let mut elements: Vec<usize> = slots.iter().map(TensorMeta::num_elements).collect();
    for (slot, meta) in slots.iter().enumerate() {
        queue_slot_pairs(meta, &live, &elements, slot, &mut leg_holders, &mut queue);
    }

    let mut plan = ContractionPlan {
        pairs: Vec::new(),
        peak: 0,
        total: 0,
    };
    while let Some((i, j)) = queue.pop_live(&live) {
        live[i] = false;
        live[j] = false;
        let result = contract_meta(&slots[i], &slots[j]);
        let size = result.num_elements();
        if size > abort_above {
            return None;
        }
        plan.peak = plan.peak.max(size);
        plan.total += size;
        if plan.total > total_budget {
            return None;
        }
        plan.pairs.push((i, j));
        live.push(true);
        elements.push(size);
        queue_slot_pairs(
            &result,
            &live,
            &elements,
            slots.len(),
            &mut leg_holders,
            &mut queue,
        );
        slots.push(result);
    }
    Some(plan)
}

#[cfg(test)]
thread_local! {
    pub(super) static PLANNER_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Plan greedily, then rerun with noise when the greedy peak intermediate
/// reaches [`RESTART_PEAK_THRESHOLD`], keeping the best plan by peak then
/// total.
///
/// The greedy plan competes, so the peak never rises. The restart arm walks
/// `tensors` a second time; below the threshold the single pass pays one
/// metadata copy and no more.
pub(super) fn plan_with_restarts(tensors: &[Tensor]) -> ContractionPlan {
    #[cfg(test)]
    PLANNER_CALLS.with(|calls| calls.set(calls.get() + 1));
    let slots: Vec<TensorMeta> = tensors.iter().map(TensorMeta::of).collect();
    let greedy = plan_pairs(slots, None, usize::MAX, usize::MAX).expect("unbounded pass completes");
    if greedy.peak < RESTART_PEAK_THRESHOLD {
        return greedy;
    }
    let metas: Vec<TensorMeta> = tensors.iter().map(TensorMeta::of).collect();
    let bound = greedy.peak;
    let restart = |index: u64| {
        let temp_index = index / PLAN_RESTARTS_PER_TEMPERATURE;
        let pass = index % PLAN_RESTARTS_PER_TEMPERATURE;
        let temperature = PLAN_NOISE_TEMPERATURES[temp_index as usize];
        let pass_seed = PLAN_NOISE_SEED ^ ((temp_index << 32) | pass);
        let mut rng = ChaCha8Rng::seed_from_u64(pass_seed);
        plan_pairs(
            metas.clone(),
            Some((&mut rng, temperature)),
            bound,
            usize::MAX,
        )
    };
    let passes = PLAN_NOISE_TEMPERATURES.len() as u64 * PLAN_RESTARTS_PER_TEMPERATURE;
    #[cfg(feature = "parallel")]
    let candidates: Vec<Option<ContractionPlan>> =
        (0..passes).into_par_iter().map(restart).collect();
    #[cfg(not(feature = "parallel"))]
    let candidates: Vec<Option<ContractionPlan>> = (0..passes).map(restart).collect();
    candidates
        .into_iter()
        .flatten()
        .fold(greedy, |best, candidate| {
            if (candidate.peak, candidate.total) < (best.peak, best.total) {
                candidate
            } else {
                best
            }
        })
}

/// Multiply out the tensors left once no pair shares a leg, smallest first.
///
/// Reached when every connected component has collapsed to a single tensor, and
/// stable from there: merging two tensors that share no leg with anything cannot
/// create a shared leg. For disjoint operands the greedy cost reduces to the
/// product of their element counts, so smallest-first is the same choice a scan
/// over every pair would make, without the scan.
pub(super) fn join_disjoint(mut slots: Vec<Option<Tensor>>, pool: &mut BufferPool) -> Tensor {
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
        let merged = contract_with(&a_tensor, &b_tensor, pool);
        pool.give(a_tensor.data);
        pool.give(b_tensor.data);
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
pub(super) fn greedy_contract(
    tensors: &mut Vec<Tensor>,
    limits: ContractionLimits,
    backend: &str,
    operation: &str,
) -> Result<Tensor> {
    debug_assert!(!tensors.is_empty());

    let plan = plan_with_restarts(tensors);
    contract_within(tensors, &plan, limits, backend, operation)
}
