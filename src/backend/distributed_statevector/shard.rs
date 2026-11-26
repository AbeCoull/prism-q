//! Shard kernels over one rank's slice and a received block, plus the circuit qubits a
//! gate touches and the ones it must hold in local positions.

use num_complex::Complex64;

#[cfg(feature = "parallel")]
use crate::backend::SendPtr;
use crate::backend::is_phase_one;
use crate::backend::simd;
use crate::backend::statevector::kernels::{
    apply_cu_phase, apply_diagonal, apply_parity2q, rzz_phases,
};
#[cfg(feature = "parallel")]
use crate::backend::statevector::rdm_sum_add;
use crate::backend::statevector::{StatevectorBackend, rdm_pair_sums};
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ELEMS, MIN_PAR_REDUCE_ELEMS, chunk_min_len};
use crate::circuit::SmallVec;
use crate::gates::{
    BatchPhaseData, BatchRzzData, DiagEntry, DiagonalBatchData, Gate, MultiFusedData,
    is_diagonal_2x2,
};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Shard length at which the combine loops below fan out to Rayon. Not tied to
/// `PARALLEL_THRESHOLD_QUBITS`, whose move to 15 was measured on dense kernels only.
#[cfg(feature = "parallel")]
pub(super) const PAR_SHARD_LEN: usize = 1 << 14;

pub(super) fn scale_shard(state: &mut [Complex64], factor: Complex64) {
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .for_each(|tile| simd::scale_complex_slice(tile, factor));
        return;
    }
    simd::scale_complex_slice(state, factor);
}

/// Scale the amplitudes whose index has every bit of `mask` set.
pub(super) fn scale_shard_masked(state: &mut [Complex64], mask: usize, factor: Complex64) {
    let tile = |base: usize, tile: &mut [Complex64]| {
        for (k, amp) in tile.iter_mut().enumerate() {
            if (base + k) & mask == mask {
                *amp *= factor;
            }
        }
    };
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, chunk)| tile(t * MIN_PAR_ELEMS, chunk));
        return;
    }
    tile(0, state);
}

pub(super) fn zero_shard(state: &mut [Complex64]) {
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .for_each(simd::zero_slice);
        return;
    }
    simd::zero_slice(state);
}

/// `dst[i] = c_self * dst[i] + c_remote * remote[i]` over a received block.
pub(super) fn combine_shard(
    dst: &mut [Complex64],
    remote: &[Complex64],
    c_self: Complex64,
    c_remote: Complex64,
) {
    #[cfg(feature = "parallel")]
    if dst.len() >= PAR_SHARD_LEN {
        dst.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(remote.par_chunks(MIN_PAR_ELEMS))
            .for_each(|(d, r)| simd::combine_global_half(d, r, c_self, c_remote));
        return;
    }
    simd::combine_global_half(dst, remote, c_self, c_remote);
}

/// `pack[k] = state[index_of(k)]` for every `k`.
pub(super) fn gather_indexed(
    pack: &mut [Complex64],
    state: &[Complex64],
    index_of: impl Fn(usize) -> usize + Sync,
) {
    #[cfg(feature = "parallel")]
    if pack.len() >= PAR_SHARD_LEN {
        pack.par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, tile)| {
                let base = t * MIN_PAR_ELEMS;
                for (k, slot) in tile.iter_mut().enumerate() {
                    *slot = state[index_of(base + k)];
                }
            });
        return;
    }
    for (k, slot) in pack.iter_mut().enumerate() {
        *slot = state[index_of(k)];
    }
}

/// `state[index_of(k)] = f(state[index_of(k)], recv[k])` for every `k`.
/// `index_of` must be injective on `0..recv.len()`: the parallel arm relies on
/// it to keep the tasks' writes disjoint.
pub(super) fn scatter_indexed(
    state: &mut [Complex64],
    recv: &[Complex64],
    index_of: impl Fn(usize) -> usize + Sync,
    f: impl Fn(Complex64, Complex64) -> Complex64 + Sync,
) {
    #[cfg(feature = "parallel")]
    if recv.len() >= PAR_SHARD_LEN {
        let ptr = SendPtr(state.as_mut_ptr());
        recv.par_chunks(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(t, tile)| {
                let base = t * MIN_PAR_ELEMS;
                for (k, &r) in tile.iter().enumerate() {
                    let i = index_of(base + k);
                    // SAFETY: `index_of` is injective and maps into `state`, so
                    // each index is read and written by exactly one task and no
                    // two tasks touch the same amplitude.
                    unsafe { ptr.store(i, f(ptr.load(i), r)) };
                }
            });
        return;
    }
    for (k, &r) in recv.iter().enumerate() {
        let i = index_of(k);
        state[i] = f(state[i], r);
    }
}

/// Visit the `(lo, hi)` halves of every `2^(local_q + 1)` block of `state`
/// together with the same halves of `recv`, as four equal-length tiles whose
/// `k`-th elements share a basis index apart from the `local_q` bit.
pub(super) fn for_each_pair_tile<F>(
    state: &mut [Complex64],
    recv: &mut [Complex64],
    local_q: usize,
    f: F,
) where
    F: Fn(&mut [Complex64], &mut [Complex64], &mut [Complex64], &mut [Complex64]) + Sync,
{
    let half = 1usize << local_q;
    let block = half << 1;
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        if state.len() / block >= 4 {
            state
                .par_chunks_mut(block)
                .zip(recv.par_chunks_mut(block))
                .with_min_len(chunk_min_len(block))
                .for_each(|(s, r)| {
                    let (lo, hi) = s.split_at_mut(half);
                    let (rlo, rhi) = r.split_at_mut(half);
                    f(lo, hi, rlo, rhi);
                });
        } else {
            for (s, r) in state.chunks_mut(block).zip(recv.chunks_mut(block)) {
                let (lo, hi) = s.split_at_mut(half);
                let (rlo, rhi) = r.split_at_mut(half);
                lo.par_chunks_mut(MIN_PAR_ELEMS)
                    .zip(hi.par_chunks_mut(MIN_PAR_ELEMS))
                    .zip(rlo.par_chunks_mut(MIN_PAR_ELEMS))
                    .zip(rhi.par_chunks_mut(MIN_PAR_ELEMS))
                    .for_each(|(((lo, hi), rlo), rhi)| f(lo, hi, rlo, rhi));
            }
        }
        return;
    }
    for (s, r) in state.chunks_mut(block).zip(recv.chunks_mut(block)) {
        let (lo, hi) = s.split_at_mut(half);
        let (rlo, rhi) = r.split_at_mut(half);
        f(lo, hi, rlo, rhi);
    }
}

/// Butterfly step over a received block: `pack[i] = forward[0] * state[i] +
/// forward[1] * remote[i]` is the partial sum forwarded to the next partner,
/// then `state[i] = keep[0] * state[i] + keep[1] * remote[i]`.
pub(super) fn butterfly_shard(
    state: &mut [Complex64],
    remote: &[Complex64],
    pack: &mut [Complex64],
    keep: [Complex64; 2],
    forward: [Complex64; 2],
) {
    let tile = |state: &mut [Complex64], remote: &[Complex64], pack: &mut [Complex64]| {
        for ((s, &r), p) in state.iter_mut().zip(remote).zip(pack.iter_mut()) {
            let own = *s;
            *p = forward[0] * own + forward[1] * r;
            *s = keep[0] * own + keep[1] * r;
        }
    };
    #[cfg(feature = "parallel")]
    if state.len() >= PAR_SHARD_LEN {
        state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .zip(remote.par_chunks(MIN_PAR_ELEMS))
            .zip(pack.par_chunks_mut(MIN_PAR_ELEMS))
            .for_each(|((s, r), p)| tile(s, r, p));
        return;
    }
    tile(state, remote, pack);
}

/// `dst[i] += src[i]` over a received block.
pub(super) fn add_shard(dst: &mut [Complex64], src: &[Complex64]) {
    let tile = |dst: &mut [Complex64], src: &[Complex64]| {
        for (d, &s) in dst.iter_mut().zip(src) {
            *d += s;
        }
    };
    #[cfg(feature = "parallel")]
    if dst.len() >= PAR_SHARD_LEN {
        dst.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(src.par_chunks(MIN_PAR_ELEMS))
            .for_each(|(d, s)| tile(d, s));
        return;
    }
    tile(dst, src);
}

/// `sum |a|^2` over the `qubit == outcome` half of every block of a shard,
/// parallel above `MIN_PAR_REDUCE_ELEMS` like `state_norm_sqr`.
pub(super) fn half_norm_sqr(state: &[Complex64], qubit: usize, outcome: bool) -> f64 {
    fn select(block: &[Complex64], half: usize, outcome: bool) -> &[Complex64] {
        if outcome {
            &block[half..]
        } else {
            &block[..half]
        }
    }
    let half = 1usize << qubit;
    let block = half << 1;
    #[cfg(feature = "parallel")]
    if state.len() >= MIN_PAR_REDUCE_ELEMS {
        if state.len() / block >= 4 {
            return state
                .par_chunks(block)
                .with_min_len(chunk_min_len(block))
                .map(|b| simd::norm_sqr_sum(select(b, half, outcome)))
                .sum();
        }
        return state
            .chunks(block)
            .map(|b| {
                select(b, half, outcome)
                    .par_chunks(MIN_PAR_ELEMS)
                    .map(simd::norm_sqr_sum)
                    .sum::<f64>()
            })
            .sum();
    }
    state
        .chunks(block)
        .map(|b| simd::norm_sqr_sum(select(b, half, outcome)))
        .sum()
}

/// `(sum |lo|^2, sum |hi|^2, sum hi * conj(lo))` over paired halves, parallel
/// above `MIN_PAR_REDUCE_ELEMS`.
pub(super) fn pair_rdm_sums(lo: &[Complex64], hi: &[Complex64]) -> (f64, f64, Complex64) {
    #[cfg(feature = "parallel")]
    if lo.len() >= MIN_PAR_REDUCE_ELEMS {
        return lo
            .par_chunks(MIN_PAR_ELEMS)
            .zip(hi.par_chunks(MIN_PAR_ELEMS))
            .map(|(l, h)| rdm_pair_sums(l, h))
            .reduce(|| (0.0, 0.0, Complex64::new(0.0, 0.0)), rdm_sum_add);
    }
    rdm_pair_sums(lo, hi)
}

/// Visit every circuit qubit an instruction touches: the instruction targets
/// plus qubit indices stored inside batched gate data. Indices may repeat.
pub(super) fn for_each_gate_qubit(gate: &Gate, targets: &[usize], mut f: impl FnMut(usize)) {
    for &q in targets {
        f(q);
    }
    match gate {
        Gate::BatchPhase(data) => {
            for &(target, _) in &data.phases {
                f(target);
            }
        }
        Gate::BatchRzz(data) => {
            for &(q0, q1, _) in &data.edges {
                f(q0);
                f(q1);
            }
        }
        Gate::MultiFused(data) => {
            for &(q, _) in &data.gates {
                f(q);
            }
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, _) in &data.gates {
                f(q0);
                f(q1);
            }
        }
        Gate::DiagonalBatch(data) => {
            for entry in &data.entries {
                match *entry {
                    DiagEntry::Phase1q { qubit, .. } => f(qubit),
                    DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                        f(q0);
                        f(q1);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Circuit qubits that must occupy local positions for the gate to apply
/// without a per-gate amplitude exchange. Diagonal action and control bits
/// are free on global qubits, so only non-diagonal application targets count.
pub(super) fn required_local_qubits(gate: &Gate, targets: &[usize]) -> SmallVec<[usize; 8]> {
    let mut req: SmallVec<[usize; 8]> = SmallVec::new();
    fn push(req: &mut SmallVec<[usize; 8]>, q: usize) {
        if !req.contains(&q) {
            req.push(q);
        }
    }
    match gate {
        Gate::Cx => push(&mut req, targets[1]),
        Gate::Cz
        | Gate::Swap
        | Gate::Rzz(_)
        | Gate::BatchPhase(_)
        | Gate::BatchRzz(_)
        | Gate::DiagonalBatch(_) => {}
        Gate::Cu(_) | Gate::Mcu(_) => {
            if gate.controlled_phase().is_none() {
                let (target, mat) = match gate {
                    Gate::Mcu(data) => (targets[data.num_controls as usize], &data.mat),
                    Gate::Cu(mat) => (targets[1], &**mat),
                    _ => unreachable!("outer match arm is Cu | Mcu"),
                };
                if !is_diagonal_2x2(mat) {
                    push(&mut req, target);
                }
            }
        }
        Gate::Fused2q(_) => {
            push(&mut req, targets[0]);
            push(&mut req, targets[1]);
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, _) in &data.gates {
                push(&mut req, q0);
                push(&mut req, q1);
            }
        }
        Gate::MultiFused(data) => {
            for &(q, ref mat) in &data.gates {
                if !is_diagonal_2x2(mat) {
                    push(&mut req, q);
                }
            }
        }
        Gate::Unitary(_) => {
            for &q in targets {
                push(&mut req, q);
            }
        }
        g if g.num_qubits() == 1 && !g.is_diagonal_1q() => {
            push(&mut req, targets[0]);
        }
        _ => {}
    }
    req
}

const ONE: Complex64 = Complex64::new(1.0, 0.0);

#[inline(always)]
fn rank_bit_set(rank: usize, local: usize, q: usize) -> bool {
    (rank >> (q - local)) & 1 == 1
}

/// Whether `entries` fits the lookup table groups of `DiagonalBatch` without
/// hitting the per-entry fallback.
pub(super) fn diagonal_batch_fits(entries: &[DiagEntry]) -> bool {
    const MAX_BITS: usize = 64;
    let mut used = [false; MAX_BITS];
    let mut parent: [u8; MAX_BITS] = std::array::from_fn(|i| i as u8);
    let find_root = |parent: &mut [u8; MAX_BITS], mut x: u8| -> u8 {
        while parent[x as usize] != x {
            let p = parent[x as usize];
            parent[x as usize] = parent[p as usize];
            x = parent[x as usize];
        }
        x
    };
    for entry in entries {
        match *entry {
            DiagEntry::Phase1q { qubit, .. } => {
                if qubit >= MAX_BITS {
                    return false;
                }
                used[qubit] = true;
            }
            DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                if q0 >= MAX_BITS || q1 >= MAX_BITS {
                    return false;
                }
                used[q0] = true;
                used[q1] = true;
                let r0 = find_root(&mut parent, q0 as u8);
                let r1 = find_root(&mut parent, q1 as u8);
                if r0 != r1 {
                    parent[r0 as usize] = r1;
                }
            }
        }
    }
    let mut sizes = [0usize; MAX_BITS];
    for (q, &is_used) in used.iter().enumerate() {
        if is_used {
            let r = find_root(&mut parent, q as u8) as usize;
            sizes[r] += 1;
        }
    }
    let mut components: SmallVec<[usize; MAX_BITS]> =
        sizes.into_iter().filter(|&len| len > 0).collect();
    components.sort_unstable_by(|a, b| b.cmp(a));
    let mut groups = [0usize; DiagonalBatchData::MAX_GROUPS];
    components.iter().all(|&len| {
        match groups
            .iter_mut()
            .find(|used| **used + len <= DiagonalBatchData::MAX_QUBITS_PER_GROUP)
        {
            Some(group) => {
                *group += len;
                true
            }
            None => false,
        }
    })
}

pub(super) struct MultiFusedSplit {
    pub(super) scalar: Complex64,
    pub(super) local_gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 32]>,
    pub(super) global_nondiag: SmallVec<[(usize, [[Complex64; 2]; 2]); 8]>,
}

pub(super) fn split_multi_fused(
    gates: &[(usize, [[Complex64; 2]; 2])],
    local: usize,
    rank: usize,
) -> MultiFusedSplit {
    let mut scalar = ONE;
    let mut local_gates = SmallVec::new();
    let mut global_nondiag = SmallVec::new();
    for &(q, mat) in gates {
        if q < local {
            local_gates.push((q, mat));
        } else if is_diagonal_2x2(&mat) {
            scalar *= if rank_bit_set(rank, local, q) {
                mat[1][1]
            } else {
                mat[0][0]
            };
        } else {
            global_nondiag.push((q, mat));
        }
    }
    MultiFusedSplit {
        scalar,
        local_gates,
        global_nondiag,
    }
}

pub(super) enum BatchPhaseSplit {
    Inactive,
    GlobalControl {
        scalar: Complex64,
        local_1q: SmallVec<[(usize, Complex64); 16]>,
    },
    LocalControl {
        ctrl_phase: Complex64,
        local_phases: SmallVec<[(usize, Complex64); 16]>,
    },
}

pub(super) fn split_batch_phase(
    control: usize,
    phases: &[(usize, Complex64)],
    local: usize,
    rank: usize,
) -> BatchPhaseSplit {
    if control >= local {
        if !rank_bit_set(rank, local, control) {
            return BatchPhaseSplit::Inactive;
        }
        let mut scalar = ONE;
        let mut local_1q = SmallVec::new();
        for &(target, phase) in phases {
            if target < local {
                local_1q.push((target, phase));
            } else if rank_bit_set(rank, local, target) {
                scalar *= phase;
            }
        }
        BatchPhaseSplit::GlobalControl { scalar, local_1q }
    } else {
        let mut ctrl_phase = ONE;
        let mut local_phases = SmallVec::new();
        for &(target, phase) in phases {
            if target < local {
                local_phases.push((target, phase));
            } else if rank_bit_set(rank, local, target) {
                ctrl_phase *= phase;
            }
        }
        BatchPhaseSplit::LocalControl {
            ctrl_phase,
            local_phases,
        }
    }
}

pub(super) struct BatchRzzSplit {
    pub(super) scalar: Complex64,
    pub(super) local_1q: SmallVec<[(usize, f64); 16]>,
    pub(super) local_edges: Vec<(usize, usize, f64)>,
}

pub(super) fn split_batch_rzz(
    edges: &[(usize, usize, f64)],
    local: usize,
    rank: usize,
) -> BatchRzzSplit {
    let mut scalar_angle = 0.0f64;
    let mut local_1q: SmallVec<[(usize, f64); 16]> = SmallVec::new();
    let mut local_edges = Vec::new();
    for &(q0, q1, theta) in edges {
        match (q0 < local, q1 < local) {
            (true, true) => local_edges.push((q0, q1, theta)),
            (false, false) => {
                let diff = rank_bit_set(rank, local, q0) ^ rank_bit_set(rank, local, q1);
                scalar_angle += if diff { 0.5 * theta } else { -0.5 * theta };
            }
            (true, false) | (false, true) => {
                let (lq, gq) = if q0 < local { (q0, q1) } else { (q1, q0) };
                let signed = if rank_bit_set(rank, local, gq) {
                    -theta
                } else {
                    theta
                };
                match local_1q.iter_mut().find(|(q, _)| *q == lq) {
                    Some((_, acc)) => *acc += signed,
                    None => local_1q.push((lq, signed)),
                }
            }
        }
    }
    local_1q.retain(|(_, angle)| *angle != 0.0);
    let scalar = if scalar_angle == 0.0 {
        ONE
    } else {
        Complex64::from_polar(1.0, scalar_angle)
    };
    BatchRzzSplit {
        scalar,
        local_1q,
        local_edges,
    }
}

pub(super) fn split_diagonal_batch(
    entries: &[DiagEntry],
    local: usize,
    rank: usize,
) -> (Complex64, Vec<DiagEntry>) {
    let mut scalar = ONE;
    let mut local_entries = Vec::with_capacity(entries.len());
    for entry in entries {
        match *entry {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                if qubit < local {
                    local_entries.push(DiagEntry::Phase1q { qubit, d0, d1 });
                } else {
                    scalar *= if rank_bit_set(rank, local, qubit) {
                        d1
                    } else {
                        d0
                    };
                }
            }
            DiagEntry::Phase2q { q0, q1, phase } => match (q0 < local, q1 < local) {
                (true, true) => local_entries.push(DiagEntry::Phase2q { q0, q1, phase }),
                (false, false) => {
                    if rank_bit_set(rank, local, q0) && rank_bit_set(rank, local, q1) {
                        scalar *= phase;
                    }
                }
                (true, false) | (false, true) => {
                    let (lq, gq) = if q0 < local { (q0, q1) } else { (q1, q0) };
                    if rank_bit_set(rank, local, gq) {
                        local_entries.push(DiagEntry::Phase1q {
                            qubit: lq,
                            d0: ONE,
                            d1: phase,
                        });
                    }
                }
            },
            DiagEntry::Parity2q { q0, q1, same, diff } => match (q0 < local, q1 < local) {
                (true, true) => local_entries.push(DiagEntry::Parity2q { q0, q1, same, diff }),
                (false, false) => {
                    let parity =
                        (rank_bit_set(rank, local, q0) ^ rank_bit_set(rank, local, q1)) as usize;
                    scalar *= [same, diff][parity];
                }
                (true, false) | (false, true) => {
                    let (lq, gq) = if q0 < local { (q0, q1) } else { (q1, q0) };
                    let g = rank_bit_set(rank, local, gq) as usize;
                    local_entries.push(DiagEntry::Phase1q {
                        qubit: lq,
                        d0: [same, diff][g],
                        d1: [same, diff][g ^ 1],
                    });
                }
            },
        }
    }
    (scalar, local_entries)
}

pub(super) fn apply_multi_fused_local(
    inner: &mut StatevectorBackend,
    data: &MultiFusedData,
    local: usize,
    rank: usize,
) -> SmallVec<[(usize, [[Complex64; 2]; 2]); 8]> {
    let mut split = split_multi_fused(&data.gates, local, rank);
    if split.local_gates.is_empty() {
        if !is_phase_one(split.scalar) {
            scale_shard(&mut inner.state, split.scalar);
        }
    } else {
        if !is_phase_one(split.scalar) {
            for row in &mut split.local_gates[0].1 {
                for entry in row {
                    *entry *= split.scalar;
                }
            }
        }
        if data.all_diagonal
            || split
                .local_gates
                .iter()
                .all(|(_, mat)| is_diagonal_2x2(mat))
        {
            inner.apply_multi_1q_diagonal(&split.local_gates);
        } else {
            inner.apply_multi_1q(&split.local_gates);
        }
    }
    split.global_nondiag
}

pub(super) fn apply_batch_phase_local(
    inner: &mut StatevectorBackend,
    control: usize,
    phases: &[(usize, Complex64)],
    local: usize,
    rank: usize,
) {
    match split_batch_phase(control, phases, local, rank) {
        BatchPhaseSplit::Inactive => {}
        BatchPhaseSplit::GlobalControl { scalar, local_1q } => match local_1q.len() {
            0 => {
                if !is_phase_one(scalar) {
                    scale_shard(&mut inner.state, scalar);
                }
            }
            1 => {
                let (q, phase) = local_1q[0];
                apply_diagonal(
                    &mut inner.state,
                    inner.num_qubits,
                    q,
                    scalar,
                    scalar * phase,
                );
            }
            _ => {
                let entries: Vec<DiagEntry> = local_1q
                    .iter()
                    .enumerate()
                    .map(|(i, &(qubit, phase))| {
                        let s = if i == 0 { scalar } else { ONE };
                        DiagEntry::Phase1q {
                            qubit,
                            d0: s,
                            d1: s * phase,
                        }
                    })
                    .collect();
                if diagonal_batch_fits(&entries) {
                    inner.dispatch_gate(
                        &Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(entries))),
                        &[],
                    );
                } else {
                    let gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 32]> =
                        entries.iter().filter_map(DiagEntry::as_1q_matrix).collect();
                    inner.apply_multi_1q_diagonal(&gates);
                }
            }
        },
        BatchPhaseSplit::LocalControl {
            ctrl_phase,
            local_phases,
        } => {
            if local_phases.is_empty() {
                if !is_phase_one(ctrl_phase) {
                    apply_diagonal(&mut inner.state, inner.num_qubits, control, ONE, ctrl_phase);
                }
            } else if is_phase_one(ctrl_phase) {
                if local_phases.len() == 1 {
                    apply_cu_phase(
                        &mut inner.state,
                        inner.num_qubits,
                        control,
                        local_phases[0].0,
                        local_phases[0].1,
                    );
                } else {
                    inner.dispatch_gate(
                        &Gate::BatchPhase(Box::new(BatchPhaseData::new(local_phases))),
                        &[control],
                    );
                }
            } else if local_phases.len() < DiagonalBatchData::MAX_QUBITS_PER_GROUP {
                let mut entries = Vec::with_capacity(1 + local_phases.len());
                entries.push(DiagEntry::Phase1q {
                    qubit: control,
                    d0: ONE,
                    d1: ctrl_phase,
                });
                for (q1, phase) in local_phases {
                    entries.push(DiagEntry::Phase2q {
                        q0: control,
                        q1,
                        phase,
                    });
                }
                inner.dispatch_gate(
                    &Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(entries))),
                    &[],
                );
            } else {
                inner.dispatch_gate(
                    &Gate::BatchPhase(Box::new(BatchPhaseData::new(local_phases))),
                    &[control],
                );
                apply_diagonal(&mut inner.state, inner.num_qubits, control, ONE, ctrl_phase);
            }
        }
    }
}

fn apply_rzz_1q_residual(
    inner: &mut StatevectorBackend,
    local_1q: &[(usize, f64)],
    scalar: Complex64,
) {
    let zero = Complex64::new(0.0, 0.0);
    match local_1q.len() {
        0 => {
            if !is_phase_one(scalar) {
                scale_shard(&mut inner.state, scalar);
            }
        }
        1 => {
            let (q, angle) = local_1q[0];
            let (d0, d1) = rzz_phases(angle);
            apply_diagonal(
                &mut inner.state,
                inner.num_qubits,
                q,
                d0 * scalar,
                d1 * scalar,
            );
        }
        _ => {
            let mut gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 32]> = local_1q
                .iter()
                .map(|&(q, angle)| {
                    let (d0, d1) = rzz_phases(angle);
                    (q, [[d0, zero], [zero, d1]])
                })
                .collect();
            if !is_phase_one(scalar) {
                gates[0].1[0][0] *= scalar;
                gates[0].1[1][1] *= scalar;
            }
            inner.apply_multi_1q_diagonal(&gates);
        }
    }
}

pub(super) fn apply_batch_rzz_local(
    inner: &mut StatevectorBackend,
    edges: &[(usize, usize, f64)],
    local: usize,
    rank: usize,
) {
    let split = split_batch_rzz(edges, local, rank);
    if split.local_edges.is_empty() {
        apply_rzz_1q_residual(inner, &split.local_1q, split.scalar);
    } else if split.local_edges.len() == 1 && split.local_1q.is_empty() {
        let (q0, q1, theta) = split.local_edges[0];
        let (same, diff) = rzz_phases(theta);
        apply_parity2q(
            &mut inner.state,
            inner.num_qubits,
            q0,
            q1,
            same * split.scalar,
            diff * split.scalar,
        );
    } else if split.local_1q.is_empty() && is_phase_one(split.scalar) {
        inner.dispatch_gate(
            &Gate::BatchRzz(Box::new(BatchRzzData::new(split.local_edges))),
            &[],
        );
    } else {
        let mut entries = Vec::with_capacity(
            split
                .local_1q
                .len()
                .max(usize::from(!is_phase_one(split.scalar)))
                + split.local_edges.len(),
        );
        for (i, &(qubit, angle)) in split.local_1q.iter().enumerate() {
            let (mut d0, mut d1) = rzz_phases(angle);
            if i == 0 && !is_phase_one(split.scalar) {
                d0 *= split.scalar;
                d1 *= split.scalar;
            }
            entries.push(DiagEntry::Phase1q { qubit, d0, d1 });
        }
        if split.local_1q.is_empty() && !is_phase_one(split.scalar) {
            entries.push(DiagEntry::Phase1q {
                qubit: split.local_edges[0].0,
                d0: split.scalar,
                d1: split.scalar,
            });
        }
        for &(q0, q1, theta) in &split.local_edges {
            let (same, diff) = rzz_phases(theta);
            entries.push(DiagEntry::Parity2q { q0, q1, same, diff });
        }
        if diagonal_batch_fits(&entries) {
            inner.dispatch_gate(
                &Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(entries))),
                &[],
            );
        } else {
            if split.local_edges.len() == 1 {
                let (q0, q1, theta) = split.local_edges[0];
                let (same, diff) = rzz_phases(theta);
                apply_parity2q(&mut inner.state, inner.num_qubits, q0, q1, same, diff);
            } else {
                inner.dispatch_gate(
                    &Gate::BatchRzz(Box::new(BatchRzzData::new(split.local_edges))),
                    &[],
                );
            }
            apply_rzz_1q_residual(inner, &split.local_1q, split.scalar);
        }
    }
}

pub(super) fn apply_diagonal_batch_local(
    inner: &mut StatevectorBackend,
    entries: &[DiagEntry],
    local: usize,
    rank: usize,
) {
    let (scalar, mut local_entries) = split_diagonal_batch(entries, local, rank);
    match local_entries.len() {
        0 => {
            if !is_phase_one(scalar) {
                scale_shard(&mut inner.state, scalar);
            }
        }
        1 => match local_entries[0] {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                apply_diagonal(
                    &mut inner.state,
                    inner.num_qubits,
                    qubit,
                    d0 * scalar,
                    d1 * scalar,
                );
            }
            DiagEntry::Parity2q { q0, q1, same, diff } => {
                apply_parity2q(
                    &mut inner.state,
                    inner.num_qubits,
                    q0,
                    q1,
                    same * scalar,
                    diff * scalar,
                );
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                if is_phase_one(scalar) {
                    apply_cu_phase(&mut inner.state, inner.num_qubits, q0, q1, phase);
                } else {
                    local_entries.push(DiagEntry::Phase1q {
                        qubit: q0,
                        d0: scalar,
                        d1: scalar,
                    });
                    inner.dispatch_gate(
                        &Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(local_entries))),
                        &[],
                    );
                }
            }
        },
        _ => {
            if !is_phase_one(scalar) {
                if let Some(DiagEntry::Phase1q { d0, d1, .. }) = local_entries
                    .iter_mut()
                    .find(|e| matches!(e, DiagEntry::Phase1q { .. }))
                {
                    *d0 *= scalar;
                    *d1 *= scalar;
                } else {
                    let qubit = match local_entries[0] {
                        DiagEntry::Phase1q { qubit, .. } => qubit,
                        DiagEntry::Phase2q { q0, .. } | DiagEntry::Parity2q { q0, .. } => q0,
                    };
                    local_entries.push(DiagEntry::Phase1q {
                        qubit,
                        d0: scalar,
                        d1: scalar,
                    });
                }
            }
            inner.dispatch_gate(
                &Gate::DiagonalBatch(Box::new(DiagonalBatchData::new(local_entries))),
                &[],
            );
        }
    }
}
