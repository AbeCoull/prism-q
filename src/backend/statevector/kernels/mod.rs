//! Gate application kernels for the statevector backend.
//!
//! Each kernel has a sequential implementation and, when the `parallel` feature
//! is enabled, a Rayon-parallelized variant dispatched above
//! `kernel_parallel_qubits()`.

mod batch_tables;
mod mixed;
mod pauli_rot;
mod qft;
mod tiling;

pub(crate) use batch_tables::*;
use pauli_rot::*;
pub(crate) use qft::*;
pub(crate) use tiling::*;

use num_complex::Complex64;
use rand::RngExt;
use smallvec::SmallVec;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use super::StatevectorBackend;
use crate::backend::insert_zero_bit;
#[cfg(feature = "parallel")]
use crate::backend::kernel_parallel_qubits;
use crate::backend::simd;
use crate::backend::{MCU_QUBIT_BUF, is_phase_one, measurement_inv_norm, sorted_mcu_qubits};
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ELEMS, SendPtr};
use crate::circuit::{QftTextbookStep, qft_textbook_steps};
use crate::gates::{
    BatchPhaseData, BatchRzzData, DiagEntry, DiagonalBatchData, Gate, MAX_UNITARY_QUBITS,
    MULTI_2Q_HIGH_BUDGET, diag_entries_phase, multi_2q_high_budget_for, multi_2q_low_bits,
    multi_2q_tile_bits_for, pauli_rot_masks,
};
use crate::sim::unified_pauli::PauliAxis;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::backend::cache;
#[cfg(feature = "parallel")]
use crate::backend::{MIN_PAR_ITERS, chunk_min_len};

#[inline(always)]
fn adjacent_2q_indices(offset: usize, stride: usize, q0_is_lo: bool) -> [usize; 4] {
    if q0_is_lo {
        [
            offset,
            offset + (stride << 1),
            offset + stride,
            offset + (stride * 3),
        ]
    } else {
        [
            offset,
            offset + stride,
            offset + (stride << 1),
            offset + (stride * 3),
        ]
    }
}

/// Run `f` on the high half of every `2 * half` block of `state`.
///
/// From [`kernel_parallel_qubits`] up, the blocks run as Rayon tasks, or, with fewer than
/// four blocks, each high half splits into tiles of `MIN_PAR_ELEMS.max(sub_block)`. `f`
/// must accept any slice that is a whole number of `sub_block` runs.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn for_each_hi_half(
    state: &mut [Complex64],
    num_qubits: usize,
    half: usize,
    sub_block: usize,
    f: impl Fn(&mut [Complex64]) + Sync,
) {
    let block_size = half << 1;
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        if state.len() / block_size >= 4 {
            state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| f(&mut chunk[half..]));
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(sub_block);
            for chunk in state.chunks_mut(block_size) {
                chunk[half..].par_chunks_mut(inner_tile).for_each(&f);
            }
        }
        return;
    }
    for chunk in state.chunks_mut(block_size) {
        f(&mut chunk[half..]);
    }
}

/// Run `f` on the `(lo, hi)` halves of every `2 * half` block of `state`, split as
/// [`for_each_hi_half`] splits, with both halves tiled in step.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn for_each_half_pair(
    state: &mut [Complex64],
    num_qubits: usize,
    half: usize,
    sub_block: usize,
    f: impl Fn(&mut [Complex64], &mut [Complex64]) + Sync,
) {
    let block_size = half << 1;
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        if state.len() / block_size >= 4 {
            state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| {
                    let (lo, hi) = chunk.split_at_mut(half);
                    f(lo, hi);
                });
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(sub_block);
            for chunk in state.chunks_mut(block_size) {
                let (lo, hi) = chunk.split_at_mut(half);
                lo.par_chunks_mut(inner_tile)
                    .zip(hi.par_chunks_mut(inner_tile))
                    .for_each(|(lo_tile, hi_tile)| f(lo_tile, hi_tile));
            }
        }
        return;
    }
    for chunk in state.chunks_mut(block_size) {
        let (lo, hi) = chunk.split_at_mut(half);
        f(lo, hi);
    }
}

/// Sum `f` over the high half of every `2 * half` block of `state`, split as
/// [`for_each_hi_half`] splits with `MIN_PAR_ELEMS` tiles.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn sum_hi_halves(
    state: &[Complex64],
    num_qubits: usize,
    half: usize,
    f: impl Fn(&[Complex64]) -> f64 + Sync,
) -> f64 {
    let block_size = half << 1;
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        return if state.len() / block_size >= 4 {
            state
                .par_chunks(block_size)
                .with_min_len(chunk_min_len(block_size))
                .map(|chunk| f(&chunk[half..]))
                .sum::<f64>()
        } else {
            state
                .chunks(block_size)
                .map(|chunk| chunk[half..].par_chunks(MIN_PAR_ELEMS).map(&f).sum::<f64>())
                .sum::<f64>()
        };
    }
    let mut sum = 0.0f64;
    for chunk in state.chunks(block_size) {
        sum += f(&chunk[half..]);
    }
    sum
}

/// [`sum_hi_halves`] for an `f` that also writes the half it reads.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn sum_hi_halves_mut(
    state: &mut [Complex64],
    num_qubits: usize,
    half: usize,
    f: impl Fn(&mut [Complex64]) -> f64 + Sync,
) -> f64 {
    let block_size = half << 1;
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        return if state.len() / block_size >= 4 {
            state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .map(|chunk| f(&mut chunk[half..]))
                .sum::<f64>()
        } else {
            state
                .chunks_mut(block_size)
                .map(|chunk| {
                    chunk[half..]
                        .par_chunks_mut(MIN_PAR_ELEMS)
                        .map(&f)
                        .sum::<f64>()
                })
                .sum::<f64>()
        };
    }
    let mut sum = 0.0f64;
    for chunk in state.chunks_mut(block_size) {
        sum += f(&mut chunk[half..]);
    }
    sum
}

/// Reduce `f` under `add` over the `(lo, hi)` halves of every `2 * half` block of
/// `state`, split as [`sum_hi_halves`] splits, with both halves tiled in step.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn reduce_half_pairs<T: Send>(
    state: &[Complex64],
    num_qubits: usize,
    half: usize,
    zero: impl Fn() -> T + Sync + Send,
    add: impl Fn(T, T) -> T + Sync + Send,
    f: impl Fn(&[Complex64], &[Complex64]) -> T + Sync,
) -> T {
    let block_size = half << 1;
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        return if state.len() / block_size >= 4 {
            state
                .par_chunks(block_size)
                .with_min_len(chunk_min_len(block_size))
                .map(|chunk| {
                    let (lo, hi) = chunk.split_at(half);
                    f(lo, hi)
                })
                .reduce(&zero, &add)
        } else {
            state
                .chunks(block_size)
                .map(|chunk| {
                    let (lo, hi) = chunk.split_at(half);
                    lo.par_chunks(MIN_PAR_ELEMS)
                        .zip(hi.par_chunks(MIN_PAR_ELEMS))
                        .map(|(lo_tile, hi_tile)| f(lo_tile, hi_tile))
                        .reduce(&zero, &add)
                })
                .fold(zero(), &add)
        };
    }
    let mut acc = zero();
    for chunk in state.chunks(block_size) {
        let (lo, hi) = chunk.split_at(half);
        acc = add(acc, f(lo, hi));
    }
    acc
}

/// Run `f` on each `tile_size` tile of `state`, as Rayon tasks when `parallel`.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
fn tile_pass(
    state: &mut [Complex64],
    tile_size: usize,
    parallel: bool,
    f: impl Fn(&mut [Complex64]) + Sync,
) {
    #[cfg(feature = "parallel")]
    if parallel {
        state
            .par_chunks_mut(tile_size)
            .with_min_len(chunk_min_len(tile_size))
            .for_each(&f);
        return;
    }
    for tile in state.chunks_mut(tile_size) {
        f(tile);
    }
}

/// Fold the `|1>` branch of a reset onto the `|0>` branch over one aligned
/// `(lo, hi)` pair, then clear `hi`.
#[inline(always)]
fn fold_reset_pair(lo: &mut [Complex64], hi: &mut [Complex64], outcome: bool) {
    if outcome {
        lo.copy_from_slice(hi);
    }
    simd::zero_slice(hi);
}

/// Diagonal `(d0, d1)` on `target` when the state holds fewer than four
/// `2^(target+1)` blocks, where tiling by the block leaves at most three Rayon
/// tasks. Scales each half of every block in parallel sub-tiles instead.
#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_diagonal_high_target_par(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    let half = 1usize << target;
    for block in state.chunks_mut(half << 1) {
        let (lo, hi) = block.split_at_mut(half);
        lo.par_chunks_mut(MIN_PAR_ELEMS)
            .zip(hi.par_chunks_mut(MIN_PAR_ELEMS))
            .for_each(|(lo_t, hi_t)| {
                if !skip_lo {
                    simd::scale_complex_slice(lo_t, d0);
                }
                simd::scale_complex_slice(hi_t, d1);
            });
    }
}

/// Apply one gate over the whole state, tiling or splitting by block size.
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn apply_single_gate_par(
    state: &mut [Complex64],
    target: usize,
    mat: &[[Complex64; 2]; 2],
) {
    #[cfg(test)]
    super::sweeps::tick();
    let half = 1usize << target;
    let block_size = half << 1;
    let prepared = simd::PreparedGate1q::new(mat);

    const MIN_TILE: usize = 8192;

    let tile_size = MIN_TILE.max(block_size);
    let num_tiles = state.len() / tile_size;

    if block_size <= MIN_TILE && num_tiles >= 4 {
        state.par_chunks_mut(MIN_TILE).for_each(|tile| {
            prepared.apply_full_sequential(tile, target);
        });
    } else if num_tiles >= 4 {
        state.par_chunks_mut(block_size).for_each(|chunk| {
            let (lo, hi) = chunk.split_at_mut(half);
            prepared.apply(lo, hi);
        });
    } else {
        let sub_tile = MIN_TILE.min(half);
        for block in state.chunks_mut(block_size) {
            let (lo, hi) = block.split_at_mut(half);
            lo.par_chunks_mut(sub_tile)
                .zip(hi.par_chunks_mut(sub_tile))
                .for_each(|(lo_t, hi_t)| {
                    prepared.apply(lo_t, hi_t);
                });
        }
    }
}

/// Gather offsets of a dense unitary's `2^k` amplitudes from a base index:
/// matrix index `m` reads bit `k - 1 - i` for `targets[i]`, so `targets[0]`
/// is the most significant matrix-index bit.
#[inline(always)]
fn unitary_offsets(targets: &[usize]) -> [usize; 1 << MAX_UNITARY_QUBITS] {
    let k = targets.len();
    let mut offsets = [0usize; 1 << MAX_UNITARY_QUBITS];
    for (m, offset) in offsets.iter_mut().enumerate().take(1 << k) {
        for (i, &target) in targets.iter().enumerate() {
            *offset |= ((m >> (k - 1 - i)) & 1) << target;
        }
    }
    offsets
}

/// Targets sorted ascending, the order `insert_zero_bit` needs to place a base
/// index around every target bit.
#[inline(always)]
fn sorted_unitary_targets(targets: &[usize]) -> [usize; MAX_UNITARY_QUBITS] {
    let mut sorted = [0usize; MAX_UNITARY_QUBITS];
    sorted[..targets.len()].copy_from_slice(targets);
    sorted[..targets.len()].sort_unstable();
    sorted
}

/// `state[base + offsets[r]] = sum_c mat[r * dim + c] * state[base + offsets[c]]`
/// over the `dim` amplitudes one base index selects, `amps` the gather buffer.
///
/// # Safety
/// `base + off` must index the buffer behind `ptr` for every entry of
/// `offsets`, and nothing else may touch those amplitudes meanwhile.
#[inline(always)]
unsafe fn apply_unitary_at(
    ptr: *mut Complex64,
    base: usize,
    offsets: &[usize],
    mat: &[Complex64],
    amps: &mut [Complex64],
) {
    let dim = offsets.len();
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        for (slot, &off) in amps.iter_mut().zip(offsets) {
            *slot = *ptr.add(base + off);
        }
        for (row, &off) in offsets.iter().enumerate() {
            let coeffs = mat.get_unchecked(row * dim..(row + 1) * dim);
            let mut acc = Complex64::new(0.0, 0.0);
            for (c, a) in coeffs.iter().zip(amps.iter()) {
                acc += c * a;
            }
            *ptr.add(base + off) = acc;
        }
    }
}

/// Apply a dense `2^k x 2^k` matrix over `targets` in one gather-scatter pass,
/// `targets[0]` the most significant matrix-index bit. Shared with the
/// factored backend, whose blocks are bare statevector slices.
#[inline(always)]
pub(crate) fn apply_unitary_seq(state: &mut [Complex64], targets: &[usize], mat: &[Complex64]) {
    let k = targets.len();
    let dim = 1usize << k;
    let sorted = sorted_unitary_targets(targets);
    let offsets = unitary_offsets(targets);
    let num_bases = state.len() >> k;
    let ptr = state.as_mut_ptr();
    let mut amps = [Complex64::new(0.0, 0.0); 1 << MAX_UNITARY_QUBITS];
    for i in 0..num_bases {
        let mut base = i;
        for &q in &sorted[..k] {
            base = insert_zero_bit(base, q);
        }
        // SAFETY: insert_zero_bit over the ascending targets maps each base
        // index onto an in-bounds index with every target bit clear, so the
        // `dim` offsets select in-bounds amplitudes disjoint across bases.
        unsafe {
            apply_unitary_at(ptr, base, &offsets[..dim], mat, &mut amps[..dim]);
        }
    }
}

/// [`apply_unitary_seq`] with the base indices split across Rayon tasks.
#[cfg(feature = "parallel")]
#[inline(always)]
pub(crate) fn par_apply_unitary(state: &mut [Complex64], targets: &[usize], mat: &[Complex64]) {
    let k = targets.len();
    let dim = 1usize << k;
    let sorted = sorted_unitary_targets(targets);
    let offsets = unitary_offsets(targets);
    let num_bases = state.len() >> k;
    let ptr = SendPtr(state.as_mut_ptr());
    (0..num_bases)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |i| {
            let mut amps = [Complex64::new(0.0, 0.0); 1 << MAX_UNITARY_QUBITS];
            let mut base = i;
            for &q in &sorted[..k] {
                base = insert_zero_bit(base, q);
            }
            // SAFETY: insert_zero_bit over the ascending targets maps each
            // base index onto an in-bounds index with every target bit clear,
            // so the `dim` offsets select in-bounds amplitudes disjoint across
            // bases, and no two Rayon tasks share an amplitude.
            unsafe {
                apply_unitary_at(
                    ptr.as_complex_ptr(),
                    base,
                    &offsets[..dim],
                    mat,
                    &mut amps[..dim],
                );
            }
        });
}

/// `Rzz(theta)` phases on even and odd parity: `e^{-i theta/2}` and `e^{i theta/2}`.
#[inline(always)]
pub(crate) fn rzz_phases(theta: f64) -> (Complex64, Complex64) {
    (
        Complex64::from_polar(1.0, -theta / 2.0),
        Complex64::from_polar(1.0, theta / 2.0),
    )
}

#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
pub(crate) fn apply_diagonal(
    state: &mut [Complex64],
    num_qubits: usize,
    target: usize,
    d0: Complex64,
    d1: Complex64,
) {
    let skip_lo = is_phase_one(d0);
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        apply_diagonal_par(state, target, d0, d1, skip_lo);
        return;
    }
    simd::apply_diagonal_sequential(state, target, d0, d1, skip_lo);
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_diagonal_par(
    state: &mut [Complex64],
    target: usize,
    d0: Complex64,
    d1: Complex64,
    skip_lo: bool,
) {
    let half = 1usize << target;
    let block_size = half << 1;
    let num_blocks = state.len() / block_size;

    if num_blocks < 4 {
        apply_diagonal_high_target_par(state, target, d0, d1, skip_lo);
        return;
    }

    const MIN_TILE: usize = 8192;
    let tile_size = MIN_TILE.max(block_size);
    state.par_chunks_mut(tile_size).for_each(|tile| {
        simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
    });
}

#[inline(always)]
pub(crate) fn apply_cx(state: &mut [Complex64], num_qubits: usize, control: usize, target: usize) {
    let ctrl_half = 1usize << control;
    let tgt_half = 1usize << target;
    if control > target {
        let tgt_block = tgt_half << 1;
        for_each_hi_half(state, num_qubits, ctrl_half, tgt_block, |hi| {
            for sub in hi.chunks_mut(tgt_block) {
                let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                simd::swap_slices(sub_lo, sub_hi);
            }
        });
    } else {
        let ctrl_block = ctrl_half << 1;
        for_each_half_pair(state, num_qubits, tgt_half, ctrl_block, |lo, hi| {
            for (lo_sub, hi_sub) in lo.chunks_mut(ctrl_block).zip(hi.chunks_mut(ctrl_block)) {
                let (_, lo_active) = lo_sub.split_at_mut(ctrl_half);
                let (_, hi_active) = hi_sub.split_at_mut(ctrl_half);
                simd::swap_slices(lo_active, hi_active);
            }
        });
    }
}

#[inline(always)]
pub(crate) fn apply_cz(state: &mut [Complex64], num_qubits: usize, q0: usize, q1: usize) {
    let (lo_q, hi_q) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let lo_half = 1usize << lo_q;
    let lo_block = lo_half << 1;
    for_each_hi_half(state, num_qubits, 1usize << hi_q, lo_block, |hi_group| {
        for sub in hi_group.chunks_mut(lo_block) {
            simd::negate_slice(&mut sub[lo_half..]);
        }
    });
}

/// Multiply each amplitude by `same` where bits `q0` and `q1` agree and by `diff`
/// where they differ.
#[cfg_attr(not(feature = "parallel"), allow(unused_variables))]
#[inline(always)]
pub(crate) fn apply_parity2q(
    state: &mut [Complex64],
    num_qubits: usize,
    q0: usize,
    q1: usize,
    same: Complex64,
    diff: Complex64,
) {
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        apply_parity2q_par(state, q0, q1, same, diff);
        return;
    }

    let phases = [same, diff];
    for (i, amp) in state.iter_mut().enumerate() {
        let parity = ((i >> q0) ^ (i >> q1)) & 1;
        *amp *= phases[parity];
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_parity2q_par(
    state: &mut [Complex64],
    q0: usize,
    q1: usize,
    same: Complex64,
    diff: Complex64,
) {
    let phases = [same, diff];
    state
        .par_chunks_mut(MIN_PAR_ELEMS)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * MIN_PAR_ELEMS;
            for (j, amp) in chunk.iter_mut().enumerate() {
                let i = base + j;
                let parity = ((i >> q0) ^ (i >> q1)) & 1;
                *amp *= phases[parity];
            }
        });
}

#[inline(always)]
pub(crate) fn apply_cu(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let prepared = simd::PreparedGate1q::new(&mat);
    let ctrl_half = 1usize << control;
    let tgt_half = 1usize << target;
    if control > target {
        for_each_hi_half(state, num_qubits, ctrl_half, tgt_half << 1, |hi| {
            prepared.apply_full_sequential(hi, target)
        });
    } else {
        let ctrl_block = ctrl_half << 1;
        for_each_half_pair(state, num_qubits, tgt_half, ctrl_block, |lo, hi| {
            for (lo_sub, hi_sub) in lo.chunks_mut(ctrl_block).zip(hi.chunks_mut(ctrl_block)) {
                let (_, lo_active) = lo_sub.split_at_mut(ctrl_half);
                let (_, hi_active) = hi_sub.split_at_mut(ctrl_half);
                prepared.apply(lo_active, hi_active);
            }
        });
    }
}

#[inline(always)]
pub(crate) fn apply_mcu(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        apply_mcu_par(state, num_qubits, controls, target, mat);
        return;
    }

    let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
    let tgt_mask = 1usize << target;
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let prepared = simd::PreparedGate1q::new(&mat);
    let base_ptr = state.as_mut_ptr() as *mut f64;

    for i in 0..num_iters {
        let mut base = i;
        for &q in sorted {
            base = insert_zero_bit(base, q);
        }
        let idx0 = base | ctrl_mask;
        let idx1 = idx0 | tgt_mask;
        // SAFETY: indices from insert_zero_bit bijection are in-bounds and disjoint.
        unsafe {
            prepared.apply_pair_ptr(base_ptr.add(idx0 * 2), base_ptr.add(idx1 * 2));
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_mcu_par(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    mat: [[Complex64; 2]; 2],
) {
    let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
    let tgt_mask = 1usize << target;
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let ptr = SendPtr(state.as_mut_ptr());
    let prepared = simd::PreparedGate1q::new(&mat);

    // SAFETY: insert_zero_bit bijection gives disjoint index pairs per iteration.
    (0..num_iters)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |i| {
            let mut base = i;
            for &q in sorted {
                base = insert_zero_bit(base, q);
            }
            let idx0 = base | ctrl_mask;
            let idx1 = idx0 | tgt_mask;
            // SAFETY: idx0 and idx1 are in bounds and unique for this iteration.
            // The outer iterator maps every pair once, so Rayon tasks do not alias.
            unsafe {
                let fp = ptr.as_f64_ptr();
                prepared.apply_pair_ptr(fp.add(idx0 * 2), fp.add(idx1 * 2));
            }
        });
}

#[inline(always)]
pub(crate) fn apply_cu_phase(
    state: &mut [Complex64],
    num_qubits: usize,
    control: usize,
    target: usize,
    phase: Complex64,
) {
    let (lo, hi) = if control < target {
        (control, target)
    } else {
        (target, control)
    };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    for_each_hi_half(state, num_qubits, 1usize << hi, lo_block, |hi_group| {
        for sub in hi_group.chunks_mut(lo_block) {
            simd::scale_complex_slice(&mut sub[lo_half..], phase);
        }
    });
}

#[inline(always)]
pub(crate) fn apply_mcu_phase(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    phase: Complex64,
) {
    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        apply_mcu_phase_par(state, num_qubits, controls, target, phase);
        return;
    }

    let all_mask: usize = controls
        .iter()
        .chain(std::iter::once(&target))
        .map(|&q| 1usize << q)
        .fold(0, |a, b| a | b);
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    for i in 0..num_iters {
        let mut base = i;
        for &q in sorted {
            base = insert_zero_bit(base, q);
        }
        state[base | all_mask] *= phase;
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_mcu_phase_par(
    state: &mut [Complex64],
    num_qubits: usize,
    controls: &[usize],
    target: usize,
    phase: Complex64,
) {
    let all_mask: usize = controls
        .iter()
        .chain(std::iter::once(&target))
        .map(|&q| 1usize << q)
        .fold(0, |a, b| a | b);
    let mut sorted_buf = [0usize; MCU_QUBIT_BUF];
    let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
    let sorted = &sorted_buf[..num_special];

    let num_iters = 1usize << (num_qubits - num_special);
    let ptr = SendPtr(state.as_mut_ptr());

    // SAFETY: insert_zero_bit bijection gives disjoint indices per iteration.
    (0..num_iters)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |i| {
            let mut base = i;
            for &q in sorted {
                base = insert_zero_bit(base, q);
            }
            let idx = base | all_mask;
            // SAFETY: idx is in bounds and unique for this iteration. The
            // insert_zero_bit mapping excludes all special qubits before masks are set.
            unsafe {
                let val = ptr.load(idx);
                ptr.store(idx, val * phase);
            }
        });
}

#[inline(always)]
pub(crate) fn apply_swap(state: &mut [Complex64], num_qubits: usize, q0: usize, q1: usize) {
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let lo_half = 1usize << lo;
    let lo_block = lo_half << 1;
    for_each_half_pair(
        state,
        num_qubits,
        1usize << hi,
        lo_block,
        |lo_group, hi_group| {
            for (lo_sub, hi_sub) in lo_group
                .chunks_mut(lo_block)
                .zip(hi_group.chunks_mut(lo_block))
            {
                let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
                let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
                simd::swap_slices(lo_sub_hi, hi_sub_lo);
            }
        },
    );
}

#[inline(always)]
pub(crate) fn apply_fused_2q(
    state: &mut [Complex64],
    num_qubits: usize,
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    if q0.max(q1) - q0.min(q1) == 1 {
        #[cfg(feature = "parallel")]
        if num_qubits >= kernel_parallel_qubits() {
            apply_fused_2q_adjacent_par(state, q0, q1, mat);
            return;
        }

        apply_fused_2q_adjacent(state, q0, q1, mat);
        return;
    }

    #[cfg(feature = "parallel")]
    if num_qubits >= kernel_parallel_qubits() {
        apply_fused_2q_par(state, num_qubits, q0, q1, mat);
        return;
    }

    let prepared = simd::PreparedGate2q::new(mat);
    prepared.apply_full(state, num_qubits, q0, q1);
}

#[inline(always)]
fn apply_fused_2q_adjacent(
    state: &mut [Complex64],
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    let lo = q0.min(q1);
    let stride = 1usize << lo;
    let block_size = stride << 2;
    let q0_is_lo = q0 == lo;
    let prepared = simd::PreparedGate2q::new(mat);

    for chunk in state.chunks_mut(block_size) {
        let ptr = chunk.as_mut_ptr() as *mut f64;
        for offset in 0..stride {
            let i = adjacent_2q_indices(offset, stride, q0_is_lo);
            // SAFETY: each offset maps to one disjoint 4-amplitude group in
            // this chunk, and all indices are below block_size.
            unsafe {
                prepared.apply_group_ptr(ptr, i);
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_fused_2q_adjacent_group(
    ptr: SendPtr,
    group: usize,
    lo: usize,
    stride: usize,
    block_size: usize,
    q0_is_lo: bool,
    prepared: &simd::PreparedGate2q,
) {
    let block = group >> lo;
    let offset = group & (stride - 1);
    let base = block * block_size;
    let local = adjacent_2q_indices(offset, stride, q0_is_lo);
    let i = [
        base + local[0],
        base + local[1],
        base + local[2],
        base + local[3],
    ];
    // SAFETY: adjacent group mapping partitions the state into disjoint
    // 4-amplitude groups, and group < state.len() / 4.
    unsafe {
        prepared.apply_group_ptr(ptr.as_f64_ptr(), i);
    }
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_fused_2q_par(
    state: &mut [Complex64],
    num_qubits: usize,
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    let n_iter = 1usize << (num_qubits - 2);
    let tile = MIN_PAR_ITERS.min(n_iter);
    let ptr = SendPtr(state.as_mut_ptr());
    let prepared = simd::PreparedGate2q::new(mat);

    (0..n_iter / tile).into_par_iter().for_each(move |t| {
        // SAFETY: aligned runs of the group order are disjoint and each
        // lies inside the state.
        unsafe {
            prepared.apply_groups_ptr(ptr.as_f64_ptr(), t * tile, tile, q0, q1);
        }
    });
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn apply_fused_2q_adjacent_par(
    state: &mut [Complex64],
    q0: usize,
    q1: usize,
    mat: &[[Complex64; 4]; 4],
) {
    let lo = q0.min(q1);
    let stride = 1usize << lo;
    let block_size = stride << 2;
    let q0_is_lo = q0 == lo;
    let n_groups = state.len() >> 2;
    let ptr = SendPtr(state.as_mut_ptr());
    let prepared = simd::PreparedGate2q::new(mat);

    (0..n_groups)
        .into_par_iter()
        .with_min_len(MIN_PAR_ITERS)
        .for_each(move |group| {
            apply_fused_2q_adjacent_group(ptr, group, lo, stride, block_size, q0_is_lo, &prepared);
        });
}

impl StatevectorBackend {
    #[inline(always)]
    pub(super) fn apply_single_gate(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            apply_single_gate_par(&mut self.state, target, &mat);
            return;
        }

        #[cfg(test)]
        super::sweeps::tick();
        let prepared = simd::PreparedGate1q::new(&mat);
        prepared.apply_full_sequential(&mut self.state, target);
    }

    #[inline(always)]
    pub(super) fn apply_diagonal_gate(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_diagonal(&mut self.state, self.num_qubits, target, d0, d1);
    }

    #[inline(always)]
    pub(super) fn apply_cx(&mut self, control: usize, target: usize) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_cx(&mut self.state, self.num_qubits, control, target);
    }

    #[inline(always)]
    pub(super) fn apply_cz(&mut self, q0: usize, q1: usize) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_cz(&mut self.state, self.num_qubits, q0, q1);
    }

    #[inline(always)]
    pub(super) fn apply_rzz(&mut self, q0: usize, q1: usize, theta: f64) {
        #[cfg(test)]
        super::sweeps::tick();
        let (same, diff) = rzz_phases(theta);
        apply_parity2q(&mut self.state, self.num_qubits, q0, q1, same, diff);
    }

    #[inline(always)]
    pub(super) fn apply_cu(&mut self, control: usize, target: usize, mat: [[Complex64; 2]; 2]) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_cu(&mut self.state, self.num_qubits, control, target, mat);
    }

    #[inline(always)]
    pub(super) fn apply_unitary(&mut self, targets: &[usize], mat: &[Complex64]) {
        #[cfg(test)]
        super::sweeps::tick();
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            par_apply_unitary(&mut self.state, targets, mat);
            return;
        }
        apply_unitary_seq(&mut self.state, targets, mat);
    }

    #[inline(always)]
    pub(super) fn apply_mcu(
        &mut self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_mcu(&mut self.state, self.num_qubits, controls, target, mat);
    }

    #[inline(always)]
    pub(super) fn apply_cu_phase(&mut self, control: usize, target: usize, phase: Complex64) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_cu_phase(&mut self.state, self.num_qubits, control, target, phase);
    }

    #[inline(always)]
    pub(super) fn apply_mcu_phase(&mut self, controls: &[usize], target: usize, phase: Complex64) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_mcu_phase(&mut self.state, self.num_qubits, controls, target, phase);
    }

    #[inline(always)]
    pub(super) fn apply_swap(&mut self, q0: usize, q1: usize) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_swap(&mut self.state, self.num_qubits, q0, q1);
    }

    #[inline(always)]
    pub(crate) fn apply_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        #[cfg(test)]
        super::sweeps::tick();
        apply_fused_2q(&mut self.state, self.num_qubits, q0, q1, mat);
    }

    #[inline(always)]
    pub(super) fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let prob_one = self.qubit_probability_one(qubit);
        let outcome = self.rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;
        self.collapse_qubit(qubit, outcome, prob_one);
    }

    /// Project `qubit` onto `outcome` without drawing, given the probability of
    /// one the caller read with [`Self::qubit_probability_one`]: the collapse
    /// half of a measurement, for a caller that chose the outcome itself.
    pub(crate) fn collapse_qubit(&mut self, qubit: usize, outcome: bool, prob_one: f64) {
        #[cfg(test)]
        super::sweeps::tick();
        for_each_half_pair(
            &mut self.state,
            self.num_qubits,
            1usize << qubit,
            1,
            |lo, hi| {
                simd::zero_slice(if outcome { lo } else { hi });
            },
        );
        self.pending_norm *= measurement_inv_norm(outcome, prob_one);
    }

    /// Reset `qubit` along the branch where it measured `outcome`: collapse
    /// onto that outcome and fold it down to `|0>`, as [`Self::apply_reset`]
    /// does after its own draw.
    pub(crate) fn reset_qubit_from(&mut self, qubit: usize, outcome: bool, prob_one: f64) {
        #[cfg(test)]
        super::sweeps::tick();
        let inv_norm = measurement_inv_norm(outcome, prob_one);
        for_each_half_pair(
            &mut self.state,
            self.num_qubits,
            1usize << qubit,
            1,
            |lo, hi| {
                fold_reset_pair(lo, hi, outcome);
            },
        );
        self.pending_norm *= inv_norm;
    }

    #[inline(always)]
    pub(super) fn qubit_probability_one(&self, qubit: usize) -> f64 {
        #[cfg(test)]
        super::sweeps::tick();
        let norm_sq = self.pending_norm * self.pending_norm;
        sum_hi_halves(
            &self.state,
            self.num_qubits,
            1usize << qubit,
            simd::norm_sqr_sum,
        ) * norm_sq
    }

    /// Scale the `|1>` half of `qubit` by `factor` in the pass that reads P(1), and
    /// return P(1) from before the scaling, summed as [`Self::qubit_probability_one`]
    /// sums it.
    pub(super) fn scale_qubit_one(&mut self, qubit: usize, factor: f64) -> f64 {
        #[cfg(test)]
        super::sweeps::tick();
        let norm_sq = self.pending_norm * self.pending_norm;
        sum_hi_halves_mut(&mut self.state, self.num_qubits, 1usize << qubit, |hi| {
            simd::norm_sqr_sum_scale(hi, factor)
        }) * norm_sq
    }

    /// Multiply the amplitudes by `pending_norm` and reset it to one.
    pub(super) fn fold_pending_norm(&mut self) {
        #[cfg(test)]
        super::sweeps::tick();
        let factor = Complex64::new(self.pending_norm, 0.0);
        self.pending_norm = 1.0;

        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.state
                .par_chunks_mut(MIN_PAR_ELEMS)
                .for_each(|chunk| simd::scale_complex_slice(chunk, factor));
            return;
        }
        simd::scale_complex_slice(&mut self.state, factor);
    }

    #[inline(always)]
    pub(super) fn reduced_density_matrix_one(&self, qubit: usize) -> [[Complex64; 2]; 2] {
        let norm_sq = self.pending_norm * self.pending_norm;
        let (p0, p1, r) = reduce_half_pairs(
            &self.state,
            self.num_qubits,
            1usize << qubit,
            || (0.0, 0.0, Complex64::new(0.0, 0.0)),
            super::rdm_sum_add,
            super::rdm_pair_sums,
        );

        let scale = Complex64::new(norm_sq, 0.0);
        let r = r * scale;
        [
            [Complex64::new(p0 * norm_sq, 0.0), r.conj()],
            [r, Complex64::new(p1 * norm_sq, 0.0)],
        ]
    }

    /// `rho[t][t'] = sum_e psi[idx(t, e)] * conj(psi[idx(t', e)])`, where `t`
    /// packs `q0` in bit 1 and `q1` in bit 0 and `e` runs over the remaining
    /// `n - 2` qubits. Requires `q0 != q1`; the bit insertion is a bijection
    /// onto the block bases only for distinct positions.
    pub(super) fn reduced_density_matrix_two(&self, q0: usize, q1: usize) -> [[Complex64; 4]; 4] {
        debug_assert_ne!(q0, q1);
        let offsets = [
            0usize,
            1usize << q1,
            1usize << q0,
            (1usize << q0) | (1usize << q1),
        ];
        let (lo, hi) = (q0.min(q1), q0.max(q1));
        let state = self.state.as_slice();
        let groups = state.len() >> 2;
        let block = |g: usize| {
            let base = insert_zero_bit(insert_zero_bit(g, lo), hi);
            let mut amps = [Complex64::new(0.0, 0.0); 4];
            for (t, off) in offsets.iter().enumerate() {
                amps[t] = state[base | off];
            }
            let mut out = [[Complex64::new(0.0, 0.0); 4]; 4];
            for (t, row) in out.iter_mut().enumerate() {
                for (tp, entry) in row.iter_mut().enumerate() {
                    *entry = amps[t] * amps[tp].conj();
                }
            }
            out
        };
        let add = |mut a: [[Complex64; 4]; 4], b: [[Complex64; 4]; 4]| {
            for (row, brow) in a.iter_mut().zip(b.iter()) {
                for (entry, bentry) in row.iter_mut().zip(brow.iter()) {
                    *entry += bentry;
                }
            }
            a
        };
        let zero = || [[Complex64::new(0.0, 0.0); 4]; 4];

        #[cfg(feature = "parallel")]
        let mut rho = if self.num_qubits >= kernel_parallel_qubits() {
            (0..groups)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .map(block)
                .reduce(zero, add)
        } else {
            (0..groups).map(block).fold(zero(), add)
        };
        #[cfg(not(feature = "parallel"))]
        let mut rho = (0..groups).map(block).fold(zero(), add);

        let scale = self.pending_norm * self.pending_norm;
        for row in rho.iter_mut() {
            for entry in row.iter_mut() {
                *entry *= scale;
            }
        }
        rho
    }

    #[inline(always)]
    /// Reset `qubit` to `|0>` as one trajectory of the reset channel: sample
    /// the measurement outcome, collapse onto it, then fold the `|1>` branch
    /// down to `|0>`. See the [`Backend::reset`](crate::backend::Backend::reset)
    /// contract for why a projection onto `|0>` is not equivalent.
    pub(super) fn apply_reset(&mut self, qubit: usize) {
        let prob_one = self.qubit_probability_one(qubit);
        let outcome = self.rng.random::<f64>() < prob_one;
        self.reset_qubit_from(qubit, outcome, prob_one);
    }
}
