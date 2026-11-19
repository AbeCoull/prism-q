//! Shard kernels over one rank's slice and a received block, plus the circuit qubits a
//! gate touches and the ones it must hold in local positions.

use num_complex::Complex64;

#[cfg(feature = "parallel")]
use crate::backend::SendPtr;
use crate::backend::simd;
use crate::backend::statevector::rdm_pair_sums;
#[cfg(feature = "parallel")]
use crate::backend::statevector::rdm_sum_add;
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ELEMS, MIN_PAR_REDUCE_ELEMS, chunk_min_len};
use crate::circuit::SmallVec;
use crate::gates::{DiagEntry, Gate, is_diagonal_2x2};
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
