//! Cache tiling, subcube gathers and tier routing for batched gate passes.

use super::*;

/// Largest qubit target whose full period (2^(t+1) elements) fits within `tile_size`.
#[inline(always)]
pub(crate) const fn max_target_for_tile(tile_size: usize) -> usize {
    tile_size.ilog2() as usize - 1
}

/// Amplitudes per L2 tile: the tile budget, 2^14 on the x86 cores measured.
#[inline(always)]
fn multi_gate_l2_tile() -> usize {
    cache::tile_budget_bytes() / size_of::<Complex64>()
}

/// Highest target the L2 tier takes in a sequential pass.
#[inline(always)]
pub(crate) fn multi_gate_max_l2_target() -> usize {
    max_target_for_tile(multi_gate_l2_tile())
}

/// Amplitudes per L2 tile for a parallel pass over `state_len` amplitudes: the budget,
/// cut to leave the pool its tile count (see [`cache::parallel_tile_len`]).
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn multi_gate_l2_tile_par(state_len: usize) -> usize {
    cache::parallel_tile_len(state_len, multi_gate_l2_tile(), MIN_PAR_ELEMS)
}

/// The in-place tile a `Multi2q` tiered pass over a `num_qubits` state runs with: the
/// subcube tile, so a batch that sits below it lands in one tier and keeps its order.
#[inline(always)]
pub(crate) fn multi_2q_small_tile(num_qubits: usize) -> usize {
    1 << multi_2q_tile_bits_for(num_qubits)
}

const MULTI_GATE_L3_TILE: usize = 131_072;
pub(crate) const MULTI_GATE_MAX_L3_TARGET: usize = max_target_for_tile(MULTI_GATE_L3_TILE);

/// Length of the longest prefix of `gates` that fits the in-place tile or a
/// [`subcube_plan`] on a `num_qubits` state. Used by [`StatevectorBackend::apply_multi_2q`]
/// and the density-matrix registers to split batches whose tiers descend across
/// `max(q0, q1)` into ordered sub-runs. At least 1 for a non-empty list, since one
/// gate is always in order.
pub(crate) fn multi_2q_ordered_prefix(
    gates: &[(usize, usize, [[Complex64; 4]; 4])],
    num_qubits: usize,
) -> usize {
    let tile_bits = multi_2q_tile_bits_for(num_qubits);
    let budget = multi_2q_high_budget_for(num_qubits);
    let low_bits = multi_2q_low_bits();
    let mut high: SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]> = SmallVec::new();
    let mut in_tile = true;
    let mut over_budget = false;
    for (i, &(q0, q1, _)) in gates.iter().enumerate() {
        in_tile &= q0.max(q1) < tile_bits;
        for q in [q0, q1] {
            if q >= low_bits && !high.contains(&q) {
                if high.len() == budget {
                    over_budget = true;
                } else {
                    high.push(q);
                }
            }
        }
        if over_budget && !in_tile {
            return i.max(1);
        }
    }
    gates.len()
}

/// Tile geometry for a `Multi2q` batch that reaches past the lowest tile bits:
/// a tile is `2^high.len()` contiguous runs of `2^low` amplitudes, one run per
/// setting of the `high` qubits, and `low + high.len() == tile_bits`.
/// Gathered into one buffer, run `c` occupies `[c << low, (c + 1) << low)`, so
/// a low qubit keeps its bit and the `j`th high qubit becomes bit `low + j`.
pub(crate) struct SubcubePlan {
    pub(super) tile_bits: usize,
    pub(super) low: usize,
    pub(super) high: SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]>,
    /// Qubit index to bit position inside the gathered tile.
    pub(super) map: Vec<usize>,
}

impl SubcubePlan {
    /// Index bits outside the tile, ascending: the ones an outer index walks.
    pub(super) fn rest(&self, num_qubits: usize) -> Vec<usize> {
        (self.low..num_qubits)
            .filter(|q| !self.high.contains(q))
            .collect()
    }
}

/// The subcube tile for `gates`, or `None` when the batch is served better by
/// the in-place tiles (every qubit below the tile bits, or the state is one
/// tile) or cannot be tiled at all (more than [`multi_2q_high_budget`]
/// distinct qubits at or above [`multi_2q_low_bits`]).
pub(crate) fn subcube_plan(
    gates: &[(usize, usize, [[Complex64; 4]; 4])],
    num_qubits: usize,
) -> Option<SubcubePlan> {
    qubit_subcube_plan(gates.iter().flat_map(|&(q0, q1, _)| [q0, q1]), num_qubits)
}

/// [`subcube_plan`] over `qubits`, the qubits a batch needs inside the tile (the X and
/// Y qubits of a `MultiPauliRot` batch, the targets of a `MultiFused` one), with the
/// same `None` cases.
pub(crate) fn qubit_subcube_plan(
    qubits: impl Iterator<Item = usize> + Clone,
    num_qubits: usize,
) -> Option<SubcubePlan> {
    let tile_bits = multi_2q_tile_bits_for(num_qubits);
    if num_qubits <= tile_bits {
        return None;
    }
    let max_q = qubits.clone().max()?;
    if max_q < tile_bits {
        return None;
    }
    let budget = multi_2q_high_budget_for(num_qubits);
    let low_bits = multi_2q_low_bits();
    let mut high: SmallVec<[usize; MULTI_2Q_HIGH_BUDGET]> = SmallVec::new();
    for q in qubits {
        if q >= low_bits && !high.contains(&q) {
            if high.len() == budget {
                return None;
            }
            high.push(q);
        }
    }
    // A high qubit that lands inside the contiguous run rides there instead,
    // which lengthens the run and can free another; iterate to a fixed point.
    let mut low = tile_bits - high.len();
    loop {
        let before = high.len();
        high.retain(|&mut q| q >= low);
        if high.len() == before {
            break;
        }
        low = tile_bits - high.len();
    }
    high.sort_unstable();
    let mut map = vec![usize::MAX; num_qubits];
    for (q, slot) in map.iter_mut().enumerate().take(low) {
        *slot = q;
    }
    for (j, &h) in high.iter().enumerate() {
        map[h] = low + j;
    }
    Some(SubcubePlan {
        tile_bits,
        low,
        high,
        map,
    })
}

fn prepare_2q(
    gates: &[(usize, usize, [[Complex64; 4]; 4])],
) -> Vec<(usize, usize, simd::PreparedGate2q)> {
    gates
        .iter()
        .map(|&(q0, q1, ref mat)| (q0, q1, simd::PreparedGate2q::new(mat)))
        .collect()
}

/// The tier of the tiered `Multi2q` pass that takes a gate whose high qubit is `max_q`:
/// 0 for the in-place tile (up to `max_l2_target`), 1 for the L3 tile, 2 for a full pass.
/// The in-place tile can reach past the L3 one, and then takes the gates both would.
#[inline(always)]
fn multi_2q_tier(max_q: usize, max_l2_target: usize) -> usize {
    if max_q <= max_l2_target {
        0
    } else if max_q <= MULTI_GATE_MAX_L3_TARGET {
        1
    } else {
        2
    }
}

pub(crate) type TargetedGate2q = (usize, usize, [[Complex64; 4]; 4]);

/// Sort `gates` into the in-place tile, L3 tile and untiled tiers of
/// [`multi_2q_tier`], preparing the tiled ones, so each gate lands in exactly one.
#[inline(always)]
pub(crate) fn split_multi_2q_tiers<'a>(
    gates: &'a [TargetedGate2q],
    max_l2_target: usize,
    small_gates: &mut SmallVec<[(usize, usize, simd::PreparedGate2q); 2]>,
    medium_gates: &mut SmallVec<[(usize, usize, simd::PreparedGate2q); 2]>,
    large_gates: &mut SmallVec<[&'a TargetedGate2q; 4]>,
) {
    for gate in gates {
        let &(q0, q1, ref mat) = gate;
        match multi_2q_tier(q0.max(q1), max_l2_target) {
            0 => small_gates.push((q0, q1, simd::PreparedGate2q::new(mat))),
            1 => medium_gates.push((q0, q1, simd::PreparedGate2q::new(mat))),
            _ => large_gates.push(gate),
        }
    }
}

fn tiers_keep_order(gates: &[(usize, usize, [[Complex64; 4]; 4])], num_qubits: usize) -> bool {
    let max_small = max_target_for_tile(multi_2q_small_tile(num_qubits));
    let tier =
        |&(q0, q1, _): &(usize, usize, [[Complex64; 4]; 4])| multi_2q_tier(q0.max(q1), max_small);
    gates
        .windows(2)
        .all(|pair| tier(&pair[0]) <= tier(&pair[1]))
}

/// Whether a `Multi2q` batch keeps list order in a single tiered or subcube pass
/// on a `num_qubits` state.
#[cfg(test)]
pub(crate) fn multi_2q_keeps_order(
    gates: &[(usize, usize, [[Complex64; 4]; 4])],
    num_qubits: usize,
) -> bool {
    gates.len() <= 1
        || subcube_plan(gates, num_qubits).is_some()
        || tiers_keep_order(gates, num_qubits)
}

thread_local! {
    /// The gathered tile, one per thread and kept between batches: a fresh
    /// allocation per subcube paid more in page faults than the sweeps it
    /// saved.
    static SUBCUBE_TILE: std::cell::RefCell<Vec<Complex64>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

pub(super) fn with_subcube_tile<R>(tile_bits: usize, f: impl FnOnce(&mut [Complex64]) -> R) -> R {
    SUBCUBE_TILE.with(|tile| {
        let mut tile = tile.borrow_mut();
        if tile.len() != 1 << tile_bits {
            tile.resize(1 << tile_bits, Complex64::new(0.0, 0.0));
        }
        f(&mut tile)
    })
}

/// Offset of the subcube `outer` selects: bit `j` of `outer` lands on qubit `rest[j]`.
#[inline(always)]
fn subcube_base(rest: &[usize], outer: usize) -> usize {
    let mut base = 0usize;
    for (j, &p) in rest.iter().enumerate() {
        base |= ((outer >> j) & 1) << p;
    }
    base
}

/// Offset of run `c` of the subcube at `base`: bit `j` of `c` lands on `plan.high[j]`.
#[inline(always)]
fn subcube_run_offset(plan: &SubcubePlan, base: usize, c: usize) -> usize {
    let mut off = base;
    for (j, &h) in plan.high.iter().enumerate() {
        off |= ((c >> j) & 1) << h;
    }
    off
}

/// Gather the subcube at `base` into `tile`, run `c` at `[c << low, (c + 1) << low)`.
/// Returns false when every gathered amplitude is zero: a zero subcube stays zero under
/// any unitary, so it needs no gates and no write back. Early in a circuit most subcubes
/// are zero; a dense one exits the scan at its first amplitude.
///
/// # Safety
/// `state` must hold the subcube at `base` laid out for `plan`, `tile` must hold
/// `2^plan.tile_bits` amplitudes, and no other access to this subcube may overlap.
#[inline(always)]
unsafe fn gather_subcube(
    state: *const Complex64,
    tile: &mut [Complex64],
    plan: &SubcubePlan,
    base: usize,
) -> bool {
    let run = 1usize << plan.low;
    let runs = 1usize << plan.high.len();
    for c in 0..runs {
        // SAFETY: same contract as the enclosing unsafe fn; the run is
        // inside the subcube and the tile holds `runs << low` elements.
        unsafe {
            std::ptr::copy_nonoverlapping(
                state.add(subcube_run_offset(plan, base, c)),
                tile.as_mut_ptr().add(c << plan.low),
                run,
            );
        }
    }
    !tile[..runs << plan.low]
        .iter()
        .all(|a| a.re == 0.0 && a.im == 0.0)
}

/// Write `tile` back over the subcube at `base`, the inverse of [`gather_subcube`].
///
/// # Safety
/// The contract of [`gather_subcube`].
#[inline(always)]
unsafe fn scatter_subcube(
    tile: &[Complex64],
    state: *mut Complex64,
    plan: &SubcubePlan,
    base: usize,
) {
    let run = 1usize << plan.low;
    for c in 0..1usize << plan.high.len() {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            std::ptr::copy_nonoverlapping(
                tile.as_ptr().add(c << plan.low),
                state.add(subcube_run_offset(plan, base, c)),
                run,
            );
        }
    }
}

/// Gather the subcube `outer` selects into this thread's tile, apply `gates`
/// there in order through the contiguous tiled kernel, and scatter it back.
/// Applying in place across the runs instead was measured slower at every
/// width: the strided group walk gives up the paired AVX2 kernel.
///
/// # Safety
/// `state` must point at `2^num_qubits` amplitudes laid out for `plan` and
/// `rest`, and no other access to this subcube may overlap the call.
unsafe fn apply_subcube(
    state: *mut Complex64,
    plan: &SubcubePlan,
    rest: &[usize],
    outer: usize,
    gates: &[(usize, usize, simd::PreparedGate2q)],
) {
    let base = subcube_base(rest, outer);
    with_subcube_tile(plan.tile_bits, |tile| {
        // SAFETY: same contract as the enclosing unsafe fn.
        if !unsafe { gather_subcube(state, tile, plan, base) } {
            return;
        }
        for &(q0, q1, ref prepared) in gates {
            prepared.apply_tiled(tile, plan.tile_bits, plan.map[q0], plan.map[q1]);
        }
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe { scatter_subcube(tile, state, plan, base) };
    });
}

/// [`apply_subcube`] for a `MultiPauliRot` batch: gather the subcube, apply
/// `rotations` there in order, and scatter it back. The body stays apart from the
/// `Multi2q` one, sharing only the gather and scatter: one generic body for both read
/// 5% slower on `statevector/qv/16`.
///
/// # Safety
/// `state` must point at `2^num_qubits` amplitudes laid out for `plan` and
/// `rest`, and no other access to this subcube may overlap the call.
pub(super) unsafe fn apply_pauli_rot_subcube(
    state: *mut Complex64,
    plan: &SubcubePlan,
    rest: &[usize],
    outer: usize,
    rotations: &[TiledPauliRot],
) {
    let base = subcube_base(rest, outer);
    with_subcube_tile(plan.tile_bits, |tile| {
        // SAFETY: same contract as the enclosing unsafe fn.
        if !unsafe { gather_subcube(state, tile, plan, base) } {
            return;
        }
        for rotation in rotations {
            rotation.apply(tile, base);
        }
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe { scatter_subcube(tile, state, plan, base) };
    });
}

/// [`apply_subcube`] for a `MultiFused` batch, each gate keyed by its bit in the
/// gathered tile. Kept apart from the `Multi2q` body for the reason
/// [`apply_pauli_rot_subcube`] gives.
///
/// # Safety
/// `state` must point at `2^num_qubits` amplitudes laid out for `plan` and
/// `rest`, and no other access to this subcube may overlap the call.
unsafe fn apply_multi_1q_subcube(
    state: *mut Complex64,
    plan: &SubcubePlan,
    rest: &[usize],
    outer: usize,
    gates: &[(usize, simd::PreparedGate1q)],
) {
    let base = subcube_base(rest, outer);
    with_subcube_tile(plan.tile_bits, |tile| {
        // SAFETY: same contract as the enclosing unsafe fn.
        if !unsafe { gather_subcube(state, tile, plan, base) } {
            return;
        }
        for &(bit, ref prepared) in gates {
            prepared.apply_tiled(tile, bit);
        }
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe { scatter_subcube(tile, state, plan, base) };
    });
}

/// Run `gates`, keyed by tile bit, over every subcube of `state` that `plan` selects:
/// one pass over the state.
fn apply_multi_1q_subcubes(
    state: &mut [Complex64],
    plan: &SubcubePlan,
    gates: &[(usize, simd::PreparedGate1q)],
) {
    #[cfg(test)]
    crate::backend::statevector::sweeps::tick();
    let num_qubits = state.len().trailing_zeros() as usize;
    let rest = plan.rest(num_qubits);
    let subcubes = 1usize << (num_qubits - plan.tile_bits);
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        let ptr = SendPtr(state.as_mut_ptr());
        (0..subcubes).into_par_iter().for_each(|outer| {
            // SAFETY: `ptr` holds 2^num_qubits amplitudes and distinct `outer`
            // values address disjoint subcubes, so no two tasks touch the same
            // amplitude.
            unsafe { apply_multi_1q_subcube(ptr.as_complex_ptr(), plan, &rest, outer, gates) };
        });
        return;
    }
    let ptr = state.as_mut_ptr();
    for outer in 0..subcubes {
        // SAFETY: `ptr` holds 2^num_qubits amplitudes and each `outer` addresses
        // its own subcube, applied one at a time here.
        unsafe { apply_multi_1q_subcube(ptr, plan, &rest, outer, gates) };
    }
}

/// Targets folded into one shared high-target traversal. `lanes * tile_len` is
/// fixed at the L2 budget, so each lane's run shrinks as `2^-k` while the
/// stream count grows; past three the tiles are too short to prefetch.
#[cfg(feature = "parallel")]
pub(crate) const MAX_SHARED_1Q_TARGETS: usize = 3;

/// Apply gates on `k` distinct targets in one traversal instead of `k`.
///
/// Gates on disjoint qubits commute, so the stages run in target order.
/// For targets above the L2 tier only: below it the tiled pass keeps a tile
/// cache resident across gates, which this shape cannot. The L3 tier's targets
/// join it where their tiles would not stay resident either.
#[cfg(feature = "parallel")]
fn apply_multi_1q_shared(state: &mut [Complex64], gates: &[(usize, [[Complex64; 2]; 2])]) {
    #[cfg(test)]
    crate::backend::statevector::sweeps::tick();
    let mut sorted: SmallVec<[(usize, [[Complex64; 2]; 2]); 4]> = gates.into();
    sorted.sort_unstable_by_key(|&(target, _)| target);
    debug_assert!(
        sorted.windows(2).all(|pair| pair[0].0 < pair[1].0),
        "repeated target in a shared 1q pass"
    );
    let k = sorted.len();
    let lanes = 1usize << k;
    let low_target = sorted[0].0;

    let prepared: SmallVec<[simd::PreparedGate1q; 4]> = sorted
        .iter()
        .map(|(_, mat)| simd::PreparedGate1q::new(mat))
        .collect();

    let mut offsets: SmallVec<[usize; 16]> = SmallVec::with_capacity(lanes);
    for j in 0..lanes {
        let mut offset = 0usize;
        for (axis, &(target, _)) in sorted.iter().enumerate() {
            if (j >> axis) & 1 == 1 {
                offset |= 1usize << target;
            }
        }
        offsets.push(offset);
    }

    let run = 1usize << low_target;
    let tile_len = (multi_gate_l2_tile() / lanes).min(run);
    let tiles_per_run = run / tile_len;
    let blocks = (state.len() >> k) >> low_target;

    let base_of = |block: usize| {
        let mut base = block << low_target;
        for &(target, _) in sorted.iter() {
            base = insert_zero_bit(base, target);
        }
        base
    };

    let start_of = |unit: usize| base_of(unit / tiles_per_run) + (unit % tiles_per_run) * tile_len;
    let units = blocks * tiles_per_run;

    let ptr = SendPtr(state.as_mut_ptr());
    (0..units).into_par_iter().for_each(|unit| {
        let start = start_of(unit);
        for (axis, prep) in prepared.iter().enumerate() {
            let stride = 1usize << axis;
            for j in 0..lanes {
                if j & stride != 0 {
                    continue;
                }
                // SAFETY: the target weights are super-increasing, so
                // the target bits and tile index partition the buffer
                // into disjoint `tile_len` runs; no two lanes or units
                // alias.
                unsafe {
                    let base = ptr.as_complex_ptr();
                    let lo = std::slice::from_raw_parts_mut(base.add(start + offsets[j]), tile_len);
                    let hi = std::slice::from_raw_parts_mut(
                        base.add(start + offsets[j | stride]),
                        tile_len,
                    );
                    prep.apply(lo, hi);
                }
            }
        }
    });
}

pub(crate) type TargetedGate1q = (usize, [[Complex64; 2]; 2]);

/// Sort `gates` into the L2 tile, L3 tile and untiled tiers by target. Both upper
/// tiers run in ascending target order, so folding the L3 tier into the untiled one
/// leaves every amplitude bitwise unchanged.
#[inline(always)]
pub(crate) fn split_multi_1q_tiers(
    gates: &[TargetedGate1q],
    max_l2_target: usize,
    medium_tiled: bool,
    small_gates: &mut SmallVec<[(usize, simd::PreparedGate1q); 16]>,
    medium_gates: &mut SmallVec<[(usize, simd::PreparedGate1q); 4]>,
    large_gates: &mut SmallVec<[TargetedGate1q; 4]>,
) {
    for &(target, mat) in gates {
        if target <= max_l2_target {
            small_gates.push((target, simd::PreparedGate1q::new(&mat)));
        } else if target <= MULTI_GATE_MAX_L3_TARGET && medium_tiled {
            medium_gates.push((target, simd::PreparedGate1q::new(&mat)));
        } else {
            large_gates.push((target, mat));
        }
    }
    medium_gates.sort_unstable_by_key(|&(target, _)| target);
    large_gates.sort_unstable_by_key(|&(target, _)| target);
}

/// Whether the L3 tier beats folding its targets into the shared passes: every
/// worker needs its own tile, and the concurrent tiles must fit the last-level cache.
/// An unreported cache keeps the tier.
#[cfg(feature = "parallel")]
pub(crate) fn keep_l3_tier(state_len: usize) -> bool {
    let Some(cache) = crate::backend::cache::last_level_cache() else {
        return true;
    };
    let tiles = state_len / MULTI_GATE_L3_TILE;
    let workers = rayon::current_num_threads();
    let tile_bytes = MULTI_GATE_L3_TILE * std::mem::size_of::<Complex64>();
    tiles >= workers && crate::backend::cache::tiles_fit(cache, tile_bytes, workers)
}

/// Sort diagonal `gates` into the L2 tile, L3 tile and untiled tiers by target, as
/// `(target, d0, d1)` with the tiled tiers also carrying `is_phase_one(d0)`.
#[inline(always)]
pub(crate) fn split_multi_1q_diagonal_tiers(
    gates: &[TargetedGate1q],
    max_l2_target: usize,
    small_gates: &mut SmallVec<[(usize, Complex64, Complex64, bool); 16]>,
    medium_gates: &mut SmallVec<[(usize, Complex64, Complex64, bool); 4]>,
    large_gates: &mut SmallVec<[(usize, Complex64, Complex64); 4]>,
) {
    for &(target, mat) in gates {
        let d0 = mat[0][0];
        let d1 = mat[1][1];
        let skip_lo = is_phase_one(d0);
        if target <= max_l2_target {
            small_gates.push((target, d0, d1, skip_lo));
        } else if target <= MULTI_GATE_MAX_L3_TARGET {
            medium_gates.push((target, d0, d1, skip_lo));
        } else {
            large_gates.push((target, d0, d1));
        }
    }
}

/// Passes over a `num_qubits` state that [`StatevectorBackend::apply_multi_1q`] makes
/// for a non-diagonal `gates` batch, on the route [`multi_1q_route`] picks. Callers
/// weigh it against the passes of another route.
pub(crate) fn multi_1q_sweeps(gates: &[(usize, [[Complex64; 2]; 2])], num_qubits: usize) -> usize {
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        let state_len = 1usize << num_qubits;
        return multi_1q_route(
            gates,
            num_qubits,
            max_target_for_tile(multi_gate_l2_tile_par(state_len)),
            keep_l3_tier(state_len),
            MAX_SHARED_1Q_TARGETS,
        )
        .1;
    }
    multi_1q_route(gates, num_qubits, multi_gate_max_l2_target(), true, 1).1
}

/// The passes [`split_multi_1q_tiers`] leads to with these tier bounds, when the gates
/// above the tiles share traversals `shared` at a time.
fn tier_sweeps(
    gates: &[(usize, [[Complex64; 2]; 2])],
    max_l2_target: usize,
    medium_tiled: bool,
    shared: usize,
) -> usize {
    let (mut small, mut medium, mut large) = (false, false, 0usize);
    for &(target, _) in gates {
        if target <= max_l2_target {
            small = true;
        } else if target <= MULTI_GATE_MAX_L3_TARGET && medium_tiled {
            medium = true;
        } else {
            large += 1;
        }
    }
    small as usize + medium as usize + large.div_ceil(shared)
}

/// How a `MultiFused` batch runs. Every route applies the targets inside the
/// in-place tile first, in list order, then the rest by ascending target, so the
/// routes agree bitwise.
pub(crate) enum Multi1qRoute {
    /// The tile tiers, then shared traversals of the targets above them.
    Tiered,
    /// One subcube gather over every target.
    Whole(SubcubePlan),
    /// The in-place tile pass, then the targets above it cut into runs of the
    /// `Multi2q` high budget, each `(len, plan)`: one subcube gather per run with a
    /// plan, shared traversals for a run whose gather would save no pass.
    Split(SmallVec<[(usize, Option<SubcubePlan>); 2]>),
}

/// The route for `gates` on a `num_qubits` state and its pass count. A gather route
/// is taken only when it makes strictly fewer passes than the tiers, whose targets
/// above `max_l2_target` share traversals `shared` at a time.
pub(crate) fn multi_1q_route(
    gates: &[TargetedGate1q],
    num_qubits: usize,
    max_l2_target: usize,
    medium_tiled: bool,
    shared: usize,
) -> (Multi1qRoute, usize) {
    let tiered = tier_sweeps(gates, max_l2_target, medium_tiled, shared);
    if tiered <= 1 {
        return (Multi1qRoute::Tiered, tiered);
    }
    if let Some(plan) = qubit_subcube_plan(gates.iter().map(|&(target, _)| target), num_qubits) {
        return (Multi1qRoute::Whole(plan), 1);
    }
    let mut above: SmallVec<[usize; 16]> = gates
        .iter()
        .map(|&(target, _)| target)
        .filter(|&target| target > max_l2_target)
        .collect();
    above.sort_unstable();
    let mut passes = usize::from(above.len() < gates.len());
    let mut runs = SmallVec::new();
    for run in above.chunks(multi_2q_high_budget_for(num_qubits)) {
        let plan = if run.len() > shared {
            qubit_subcube_plan(run.iter().copied(), num_qubits)
        } else {
            None
        };
        passes += if plan.is_some() {
            1
        } else {
            run.len().div_ceil(shared)
        };
        runs.push((run.len(), plan));
    }
    if passes < tiered {
        (Multi1qRoute::Split(runs), passes)
    } else {
        (Multi1qRoute::Tiered, tiered)
    }
}

/// The gates of a [`Multi1qRoute::Whole`] gather keyed by their bit in `plan`'s tile:
/// `small` in its list order, then `above` by ascending target.
fn whole_subcube_gates(
    plan: &SubcubePlan,
    small: SmallVec<[(usize, simd::PreparedGate1q); 16]>,
    above: &[TargetedGate1q],
) -> SmallVec<[(usize, simd::PreparedGate1q); 16]> {
    small
        .into_iter()
        .map(|(target, prepared)| (plan.map[target], prepared))
        .chain(
            above
                .iter()
                .map(|&(target, ref mat)| (plan.map[target], simd::PreparedGate1q::new(mat))),
        )
        .collect()
}

/// Apply `above`, the gates past the in-place tile by ascending target, in the runs
/// of a [`Multi1qRoute::Split`].
fn apply_multi_1q_runs(
    state: &mut [Complex64],
    above: &[TargetedGate1q],
    runs: &[(usize, Option<SubcubePlan>)],
) {
    let mut rest = above;
    for &(len, ref plan) in runs {
        let (run, tail) = rest.split_at(len);
        rest = tail;
        if let Some(plan) = plan {
            let tiled: SmallVec<[(usize, simd::PreparedGate1q); 16]> = run
                .iter()
                .map(|&(target, ref mat)| (plan.map[target], simd::PreparedGate1q::new(mat)))
                .collect();
            apply_multi_1q_subcubes(state, plan, &tiled);
            continue;
        }
        #[cfg(feature = "parallel")]
        if state.len() >= 1 << kernel_parallel_qubits() {
            for chunk in run.chunks(MAX_SHARED_1Q_TARGETS) {
                if chunk.len() > 1 {
                    apply_multi_1q_shared(state, chunk);
                } else {
                    apply_single_gate_par(state, chunk[0].0, &chunk[0].1);
                }
            }
            continue;
        }
        for &(target, ref mat) in run {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            simd::PreparedGate1q::new(mat).apply_full_sequential(state, target);
        }
    }
}

/// Apply a `MultiFused` batch on the route [`multi_1q_route`] picks. Shared with the
/// factored backend, whose blocks are bare statevector slices.
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn apply_multi_1q_par(state: &mut [Complex64], gates: &[(usize, [[Complex64; 2]; 2])]) {
    if gates.is_empty() {
        return;
    }
    if gates.len() == 1 {
        apply_single_gate_par(state, gates[0].0, &gates[0].1);
        return;
    }
    let medium_tiled = keep_l3_tier(state.len());
    let (route, _) = multi_1q_route(
        gates,
        state.len().trailing_zeros() as usize,
        max_target_for_tile(multi_gate_l2_tile_par(state.len())),
        medium_tiled,
        MAX_SHARED_1Q_TARGETS,
    );
    apply_multi_1q_par_routed(state, gates, medium_tiled, &route);
}

/// [`apply_multi_1q_par`] on `route`, with the L3 tier on or folded into the shared
/// passes when the route is [`Multi1qRoute::Tiered`].
#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_multi_1q_par_routed(
    state: &mut [Complex64],
    gates: &[(usize, [[Complex64; 2]; 2])],
    medium_tiled: bool,
    route: &Multi1qRoute,
) {
    let tile_size = multi_gate_l2_tile_par(state.len());
    let mut small_gates = SmallVec::new();
    let mut medium_gates = SmallVec::new();
    let mut large_gates = SmallVec::new();
    split_multi_1q_tiers(
        gates,
        max_target_for_tile(tile_size),
        medium_tiled && matches!(route, Multi1qRoute::Tiered),
        &mut small_gates,
        &mut medium_gates,
        &mut large_gates,
    );

    if let Multi1qRoute::Whole(plan) = route {
        let tiled = whole_subcube_gates(plan, small_gates, &large_gates);
        apply_multi_1q_subcubes(state, plan, &tiled);
        return;
    }

    if !small_gates.is_empty() {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        tile_pass(state, tile_size, true, |tile| {
            for &(target, ref prepared) in &small_gates {
                prepared.apply_tiled(tile, target);
            }
        });
    }

    if !medium_gates.is_empty() {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        tile_pass(state, MULTI_GATE_L3_TILE, true, |tile| {
            for &(target, ref prepared) in &medium_gates {
                prepared.apply_tiled(tile, target);
            }
        });
    }

    if let Multi1qRoute::Split(runs) = route {
        apply_multi_1q_runs(state, &large_gates, runs);
        return;
    }

    for chunk in large_gates.chunks(MAX_SHARED_1Q_TARGETS) {
        if chunk.len() > 1 {
            apply_multi_1q_shared(state, chunk);
        } else {
            apply_single_gate_par(state, chunk[0].0, &chunk[0].1);
        }
    }
}

impl StatevectorBackend {
    /// Apply multiple single-qubit gates in a multi-tier tiled pass.
    ///
    /// Three tiers based on gate target qubit:
    /// - L2 tier (target 0..13): 256KB tiles, all applied per tile in L2 cache
    /// - L3 tier (target 14..16): 2MB tiles, applied per tile in L3 cache where
    ///   one tile per worker fits it, otherwise folded into the shared traversal
    /// - Individual (target 17+): one shared traversal, or a single
    ///   full-state pass when only one gate lands there
    ///
    /// Where that costs more passes, the targets above the L2 tier are gathered into
    /// subcube tiles as `Multi2q` gathers its high qubits (see [`subcube_plan`]), up
    /// to the same high budget per pass, together with the low targets when they all
    /// fit one tile.
    #[inline(always)]
    pub(crate) fn apply_multi_1q(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_single_gate(gates[0].0, gates[0].1);
            return;
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            apply_multi_1q_par(&mut self.state, gates);
            return;
        }

        let max_l2_target = multi_gate_max_l2_target();
        let (route, _) = multi_1q_route(gates, self.num_qubits, max_l2_target, true, 1);
        let mut small_gates = SmallVec::new();
        let mut medium_gates = SmallVec::new();
        let mut large_gates = SmallVec::new();
        split_multi_1q_tiers(
            gates,
            max_l2_target,
            matches!(route, Multi1qRoute::Tiered),
            &mut small_gates,
            &mut medium_gates,
            &mut large_gates,
        );

        if let Multi1qRoute::Whole(plan) = &route {
            let tiled = whole_subcube_gates(plan, small_gates, &large_gates);
            apply_multi_1q_subcubes(&mut self.state, plan, &tiled);
            return;
        }

        if !small_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            tile_pass(&mut self.state, multi_gate_l2_tile(), false, |tile| {
                for &(target, ref prepared) in &small_gates {
                    prepared.apply_tiled(tile, target);
                }
            });
        }

        if !medium_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            tile_pass(&mut self.state, MULTI_GATE_L3_TILE, false, |tile| {
                for &(target, ref prepared) in &medium_gates {
                    prepared.apply_tiled(tile, target);
                }
            });
        }

        if let Multi1qRoute::Split(runs) = &route {
            apply_multi_1q_runs(&mut self.state, &large_gates, runs);
            return;
        }

        for (target, mat) in large_gates {
            self.apply_single_gate(target, mat);
        }
    }

    pub(crate) fn apply_multi_1q_diagonal(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_diagonal_gate(gates[0].0, gates[0].1[0][0], gates[0].1[1][1]);
            return;
        }

        #[cfg(feature = "parallel")]
        let parallel = self.num_qubits >= kernel_parallel_qubits();
        #[cfg(not(feature = "parallel"))]
        let parallel = false;
        #[cfg(feature = "parallel")]
        let tile_size = if parallel {
            multi_gate_l2_tile_par(self.state.len())
        } else {
            multi_gate_l2_tile()
        };
        #[cfg(not(feature = "parallel"))]
        let tile_size = multi_gate_l2_tile();

        let mut small_gates = SmallVec::new();
        let mut medium_gates = SmallVec::new();
        let mut large_gates = SmallVec::new();
        split_multi_1q_diagonal_tiers(
            gates,
            max_target_for_tile(tile_size),
            &mut small_gates,
            &mut medium_gates,
            &mut large_gates,
        );

        if !small_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            tile_pass(&mut self.state, tile_size, parallel, |tile| {
                for &(target, d0, d1, skip_lo) in &small_gates {
                    simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                }
            });
        }

        if !medium_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            tile_pass(&mut self.state, MULTI_GATE_L3_TILE, parallel, |tile| {
                for &(target, d0, d1, skip_lo) in &medium_gates {
                    simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                }
            });
        }

        for (target, d0, d1) in large_gates {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let skip_lo = is_phase_one(d0);
            #[cfg(feature = "parallel")]
            if parallel {
                let block_size = 1usize << (target + 1);
                let num_blocks = self.state.len() / block_size;
                if num_blocks < 4 {
                    apply_diagonal_high_target_par(&mut self.state, target, d0, d1, skip_lo);
                    continue;
                }
                let tile_size = block_size.max(MIN_PAR_ELEMS);
                self.state.par_chunks_mut(tile_size).for_each(|tile| {
                    simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                });
                continue;
            }
            simd::apply_diagonal_sequential(&mut self.state, target, d0, d1, skip_lo);
        }
    }

    fn apply_multi_2q_subcube(
        &mut self,
        gates: &[(usize, usize, [[Complex64; 4]; 4])],
        plan: &SubcubePlan,
    ) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        let prepared = prepare_2q(gates);
        let rest = plan.rest(self.num_qubits);
        let state = self.state.as_mut_ptr();
        for outer in 0..1usize << (self.num_qubits - plan.tile_bits) {
            // SAFETY: `state` holds 2^num_qubits amplitudes and each `outer`
            // addresses its own subcube, applied one at a time here.
            unsafe { apply_subcube(state, plan, &rest, outer, &prepared) };
        }
    }

    #[cfg(feature = "parallel")]
    fn apply_multi_2q_subcube_par(
        &mut self,
        gates: &[(usize, usize, [[Complex64; 4]; 4])],
        plan: &SubcubePlan,
    ) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        let prepared = prepare_2q(gates);
        let rest = plan.rest(self.num_qubits);
        let ptr = SendPtr(self.state.as_mut_ptr());
        (0..1usize << (self.num_qubits - plan.tile_bits))
            .into_par_iter()
            .for_each(|outer| {
                // SAFETY: `ptr` holds 2^num_qubits amplitudes and distinct
                // `outer` values address disjoint subcubes, so no two tasks
                // touch the same amplitude.
                unsafe { apply_subcube(ptr.as_complex_ptr(), plan, &rest, outer, &prepared) };
            });
    }

    /// Apply multiple two-qubit gates in a cache-tiled pass.
    ///
    /// A batch whose qubits reach past the lowest tile bits takes the subcube
    /// path (see [`subcube_plan`]): each subcube (256 KB by default, set by
    /// `PRISM_MULTI_2Q_TILE_BITS`) is gathered into a per-thread tile, every
    /// gate is applied there in order, and the tile is scattered back, so the
    /// whole batch costs one pass over the state plus in-cache arithmetic.
    /// A batch that sits entirely below the tile bits (14 by default) tiles the
    /// state in place. Any other list falls back to the tiered pass by
    /// `max(q0, q1)`: L2 tiles up to qubit 13, L3 tiles up to 16, one full pass
    /// per gate above that. Batches exceeding the subcube budget whose tiers
    /// descend in `max(q0, q1)` split at [`multi_2q_ordered_prefix`] and apply
    /// each sub-run recursively so list order always holds. In a parallel pass
    /// the L3 tier stays one sweep even where the state is a single L3 tile: at
    /// 17 qubits per-gate parallel passes in its place read 28% slower on
    /// `statevector/random_d10/17`.
    pub(crate) fn apply_multi_2q(&mut self, gates: &[(usize, usize, [[Complex64; 4]; 4])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_fused_2q(gates[0].0, gates[0].1, &gates[0].2);
            return;
        }
        if let Some(plan) = subcube_plan(gates, self.num_qubits) {
            #[cfg(feature = "parallel")]
            if self.num_qubits >= kernel_parallel_qubits() {
                self.apply_multi_2q_subcube_par(gates, &plan);
                return;
            }
            self.apply_multi_2q_subcube(gates, &plan);
            return;
        }
        if !tiers_keep_order(gates, self.num_qubits) {
            let split = multi_2q_ordered_prefix(gates, self.num_qubits);
            if split < gates.len() {
                self.apply_multi_2q(&gates[..split]);
                self.apply_multi_2q(&gates[split..]);
                return;
            }
        }

        #[cfg(feature = "parallel")]
        let parallel = self.num_qubits >= kernel_parallel_qubits();
        #[cfg(not(feature = "parallel"))]
        let parallel = false;

        let tile_size = multi_2q_small_tile(self.num_qubits);
        let mut small_gates = SmallVec::new();
        let mut medium_gates = SmallVec::new();
        let mut large_gates = SmallVec::new();
        split_multi_2q_tiers(
            gates,
            max_target_for_tile(tile_size),
            &mut small_gates,
            &mut medium_gates,
            &mut large_gates,
        );

        if !small_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let tile_qubits = tile_size.trailing_zeros() as usize;
            tile_pass(&mut self.state, tile_size, parallel, |tile| {
                let n = tile.len().trailing_zeros() as usize;
                for &(q0, q1, ref prepared) in &small_gates {
                    prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                }
            });
        }

        if !medium_gates.is_empty() {
            #[cfg(test)]
            crate::backend::statevector::sweeps::tick();
            let tile_qubits = MULTI_GATE_L3_TILE.trailing_zeros() as usize;
            tile_pass(&mut self.state, MULTI_GATE_L3_TILE, parallel, |tile| {
                let n = tile.len().trailing_zeros() as usize;
                for &(q0, q1, ref prepared) in &medium_gates {
                    prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                }
            });
        }

        self.apply_multi_2q_untiled(&large_gates);
    }

    /// Apply `gates`, the untiled tier of [`split_multi_2q_tiers`], one full pass each.
    #[inline(always)]
    fn apply_multi_2q_untiled(&mut self, gates: &[&TargetedGate2q]) {
        for &&(q0, q1, ref mat) in gates {
            self.apply_fused_2q(q0, q1, mat);
        }
    }
}

#[cfg(test)]
#[path = "multi_1q_tier_tests.rs"]
mod multi_1q_tier_tests;
