//! Gate application kernels for the statevector backend.
//!
//! Each kernel has a sequential implementation and, when the `parallel` feature
//! is enabled, a Rayon-parallelized variant dispatched above
//! `PARALLEL_THRESHOLD_QUBITS`.

use num_complex::Complex64;
use rand::Rng;
use smallvec::SmallVec;

use super::insert_zero_bit;
use super::StatevectorBackend;
#[cfg(feature = "parallel")]
use super::{SendPtr, MIN_PAR_ELEMS, PARALLEL_THRESHOLD_QUBITS};
use crate::backend::simd;
use crate::backend::{is_phase_one, measurement_inv_norm, sorted_mcu_qubits};
use crate::gates::DiagEntry;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::backend::MIN_PAR_ITERS;

#[cfg(feature = "parallel")]
#[inline(always)]
fn chunk_min_len(chunk_size: usize) -> usize {
    (MIN_PAR_ELEMS / chunk_size).max(1)
}

/// Largest qubit target whose full period (2^(t+1) elements) fits within `tile_size`.
#[inline(always)]
const fn max_target_for_tile(tile_size: usize) -> usize {
    let mut t = 0usize;
    while (1usize << (t + 1)) <= tile_size {
        t += 1;
    }
    t - 1
}

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

pub(crate) const BATCH_PHASE_GROUP_SIZE: usize = 10;
pub(crate) const BATCH_PHASE_TABLE_SIZE: usize = 1024;
pub(crate) const MAX_BATCH_PHASE_GROUPS: usize = 4;

pub(crate) const BATCH_RZZ_GROUP_SIZE: usize = 8;
pub(crate) const BATCH_RZZ_TABLE_SIZE: usize = 256;
pub(crate) const MAX_BATCH_RZZ_GROUPS: usize = 4;
#[cfg(target_arch = "x86_64")]
const BATCH_RZZ_BMI2_MAX_UNIQUE: usize = 10;
#[cfg(target_arch = "x86_64")]
const BATCH_RZZ_BMI2_TABLE_SIZE: usize = 1024;

pub(crate) const DIAG_BATCH_MAX_QUBITS_PER_GROUP: usize = 10;
pub(crate) const DIAG_BATCH_TABLE_SIZE: usize = 1024; // 2^10
pub(crate) const MAX_DIAG_BATCH_GROUPS: usize = 4;

#[derive(Clone, Copy)]
pub(crate) struct BatchRzzGroup {
    pub(crate) table: [Complex64; BATCH_RZZ_TABLE_SIZE],
    pub(crate) q0s: [usize; BATCH_RZZ_GROUP_SIZE],
    pub(crate) q1s: [usize; BATCH_RZZ_GROUP_SIZE],
    pub(crate) len: usize,
}

pub(crate) fn build_batch_rzz_tables(
    edges: &[(usize, usize, f64)],
    groups: &mut [BatchRzzGroup; MAX_BATCH_RZZ_GROUPS],
) -> usize {
    let num_groups = edges.len().div_ceil(BATCH_RZZ_GROUP_SIZE);
    debug_assert!(num_groups <= MAX_BATCH_RZZ_GROUPS);

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let start = g * BATCH_RZZ_GROUP_SIZE;
        let end = (start + BATCH_RZZ_GROUP_SIZE).min(edges.len());
        let group_len = end - start;
        group.len = group_len;

        for (k, &(q0, q1, _)) in edges[start..end].iter().enumerate() {
            group.q0s[k] = q0;
            group.q1s[k] = q1;
        }

        for bits in 0..BATCH_RZZ_TABLE_SIZE {
            let mut angle = 0.0f64;
            for k in 0..group_len {
                let theta = edges[start + k].2;
                let parity = (bits >> k) & 1;
                angle += if parity == 0 {
                    -theta / 2.0
                } else {
                    theta / 2.0
                };
            }
            group.table[bits] = Complex64::from_polar(1.0, angle);
        }
    }
    num_groups
}

#[inline(always)]
fn extract_rzz_bits(i: usize, group: &BatchRzzGroup) -> usize {
    let mut bits = 0usize;
    for k in 0..group.len {
        bits |= (((i >> group.q0s[k]) ^ (i >> group.q1s[k])) & 1) << k;
    }
    bits
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct BatchRzzBmi2Group {
    table: [Complex64; BATCH_RZZ_BMI2_TABLE_SIZE],
    pext_mask: u64,
}

#[cfg(target_arch = "x86_64")]
fn build_batch_rzz_bmi2_tables(
    edges: &[(usize, usize, f64)],
    groups: &mut [BatchRzzBmi2Group; MAX_BATCH_RZZ_GROUPS],
) -> Option<usize> {
    let num_groups = edges.len().div_ceil(BATCH_RZZ_GROUP_SIZE);
    if num_groups > MAX_BATCH_RZZ_GROUPS {
        return None;
    }

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let start = g * BATCH_RZZ_GROUP_SIZE;
        let end = (start + BATCH_RZZ_GROUP_SIZE).min(edges.len());
        let group_len = end - start;

        let mut mask = 0u64;
        for &(q0, q1, _) in &edges[start..end] {
            mask |= (1u64 << q0) | (1u64 << q1);
        }
        let num_unique = mask.count_ones() as usize;
        if num_unique > BATCH_RZZ_BMI2_MAX_UNIQUE {
            return None;
        }

        group.pext_mask = mask;

        let mut q0_pos = [0u8; BATCH_RZZ_GROUP_SIZE];
        let mut q1_pos = [0u8; BATCH_RZZ_GROUP_SIZE];
        for (k, &(q0, q1, _)) in edges[start..end].iter().enumerate() {
            q0_pos[k] = (mask & ((1u64 << q0) - 1)).count_ones() as u8;
            q1_pos[k] = (mask & ((1u64 << q1) - 1)).count_ones() as u8;
        }

        let table_size = 1usize << num_unique;
        for c in 0..table_size {
            let mut angle = 0.0f64;
            for k in 0..group_len {
                let parity = ((c >> q0_pos[k]) ^ (c >> q1_pos[k])) & 1;
                let theta = edges[start + k].2;
                angle += if parity == 0 {
                    -theta / 2.0
                } else {
                    theta / 2.0
                };
            }
            group.table[c] = Complex64::from_polar(1.0, angle);
        }
        for c in table_size..BATCH_RZZ_BMI2_TABLE_SIZE {
            group.table[c] = Complex64::new(1.0, 0.0);
        }
    }
    Some(num_groups)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,fma")]
unsafe fn apply_batch_rzz_bmi2(
    state: &mut [Complex64],
    groups: &[BatchRzzBmi2Group; MAX_BATCH_RZZ_GROUPS],
    num_groups: usize,
) {
    use std::arch::x86_64::_pext_u64;
    let one = Complex64::new(1.0, 0.0);
    for (i, amp) in state.iter_mut().enumerate() {
        let mut combined = one;
        for group in groups.iter().take(num_groups) {
            let bits = _pext_u64(i as u64, group.pext_mask) as usize;
            combined *= group.table[bits];
        }
        *amp *= combined;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "bmi2,fma")]
unsafe fn batch_rzz_tile_bmi2(
    tile: &mut [Complex64],
    base_idx: usize,
    groups: &[BatchRzzBmi2Group; MAX_BATCH_RZZ_GROUPS],
    num_groups: usize,
) {
    use std::arch::x86_64::_pext_u64;
    let one = Complex64::new(1.0, 0.0);
    for (j, amp) in tile.iter_mut().enumerate() {
        let i = base_idx + j;
        let mut combined = one;
        for group in groups.iter().take(num_groups) {
            let bits = _pext_u64(i as u64, group.pext_mask) as usize;
            combined *= group.table[bits];
        }
        *amp *= combined;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BatchPhaseGroup {
    pub(crate) table: [Complex64; BATCH_PHASE_TABLE_SIZE],
    pub(crate) shifts: [usize; BATCH_PHASE_GROUP_SIZE],
    pub(crate) len: usize,
    pub(crate) pext_mask: u64,
}

/// Build lookup tables for batched controlled-phase application.
///
/// Partitions `phases` into groups of up to 10. For each group, builds a
/// 1024-entry table mapping every combination of target-bit patterns to the
/// combined phase. Returns the number of groups filled.
pub(crate) fn build_batch_phase_tables(
    phases: &[(usize, Complex64)],
    groups: &mut [BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS],
) -> usize {
    let one = Complex64::new(1.0, 0.0);
    let num_groups = phases.len().div_ceil(BATCH_PHASE_GROUP_SIZE);
    debug_assert!(num_groups <= MAX_BATCH_PHASE_GROUPS);

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let start = g * BATCH_PHASE_GROUP_SIZE;
        let end = (start + BATCH_PHASE_GROUP_SIZE).min(phases.len());
        let group_len = end - start;
        group.len = group_len;

        let mut sorted: [(usize, Complex64); BATCH_PHASE_GROUP_SIZE] =
            [(0, one); BATCH_PHASE_GROUP_SIZE];
        sorted[..group_len].copy_from_slice(&phases[start..end]);
        sorted[..group_len].sort_unstable_by_key(|&(q, _)| q);

        let mut mask = 0u64;
        for (j, &(q, _)) in sorted.iter().enumerate().take(group_len) {
            group.shifts[j] = q;
            mask |= 1u64 << q;
        }
        group.pext_mask = mask;

        group.table[0] = one;
        for (j, &(_, phase)) in sorted.iter().enumerate().take(group_len) {
            let prev_entries = 1usize << j;
            for k in 0..prev_entries {
                group.table[k | prev_entries] = group.table[k] * phase;
            }
        }
    }
    num_groups
}

/// Extract bits at specified positions from `idx` and pack them into a
/// contiguous low-order pattern for table lookup.
#[inline(always)]
fn extract_bits(idx: usize, shifts: &[usize; BATCH_PHASE_GROUP_SIZE], len: usize) -> usize {
    let mut bits = 0usize;
    for (j, &shift) in shifts.iter().enumerate().take(len) {
        bits |= ((idx >> shift) & 1) << j;
    }
    bits
}

/// Pre-built LUT data for `apply_diagonal_batch`.
///
/// Returned by [`build_diagonal_batch_tables`]. The statevector backend consumes this
/// directly; the GPU backend uploads the tables + per-group metadata to the device and
/// launches a batched kernel.
pub(crate) struct DiagBatchTables {
    pub(crate) tables: [[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    pub(crate) unique_qubits: SmallVec<[usize; 32]>,
    pub(crate) group_sizes: [usize; MAX_DIAG_BATCH_GROUPS],
    /// Per-group PEXT masks used by the x86_64 BMI2 CPU path only. The GPU path
    /// rebuilds the equivalent `group_shifts` array from `unique_qubits + group_sizes`
    /// because PTX has no PEXT; non-x86_64 CPU builds don't use this either. Kept
    /// populated regardless (the fill is trivial) so the struct has one layout.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) group_pext_masks: [u64; MAX_DIAG_BATCH_GROUPS],
    pub(crate) num_groups: usize,
}

/// Group `entries` by qubit affinity and build per-group phase LUTs.
///
/// Returns `None` when the grouping requires more than `MAX_DIAG_BATCH_GROUPS` groups, or
/// when a 2-qubit entry spans two groups. Both cases need the per-element fallback path.
/// Returns `Some` with `num_groups == 0` when `entries` is empty (no-op).
pub(crate) fn build_diagonal_batch_tables(entries: &[DiagEntry]) -> Option<DiagBatchTables> {
    let mut unique_qubits = SmallVec::<[usize; 32]>::new();
    let add_qubit = |q: usize, uq: &mut SmallVec<[usize; 32]>| {
        if !uq.contains(&q) {
            uq.push(q);
        }
    };
    for e in entries {
        match e {
            DiagEntry::Phase1q { qubit, .. } => add_qubit(*qubit, &mut unique_qubits),
            DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } => {
                add_qubit(*q0, &mut unique_qubits);
                add_qubit(*q1, &mut unique_qubits);
            }
        }
    }
    unique_qubits.sort_unstable();
    let num_unique = unique_qubits.len();

    let one = Complex64::new(1.0, 0.0);
    let empty_tables = [[one; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS];

    if num_unique == 0 {
        return Some(DiagBatchTables {
            tables: empty_tables,
            unique_qubits,
            group_sizes: [0; MAX_DIAG_BATCH_GROUPS],
            group_pext_masks: [0; MAX_DIAG_BATCH_GROUPS],
            num_groups: 0,
        });
    }

    let num_groups = num_unique.div_ceil(DIAG_BATCH_MAX_QUBITS_PER_GROUP);
    if num_groups > MAX_DIAG_BATCH_GROUPS {
        return None;
    }

    let mut qubit_group = [0u8; 64];
    let mut qubit_pos = [0u8; 64];
    let mut group_sizes = [0usize; MAX_DIAG_BATCH_GROUPS];
    let mut group_pext_masks = [0u64; MAX_DIAG_BATCH_GROUPS];
    for (idx, &q) in unique_qubits.iter().enumerate() {
        let g = idx / DIAG_BATCH_MAX_QUBITS_PER_GROUP;
        let pos = idx % DIAG_BATCH_MAX_QUBITS_PER_GROUP;
        qubit_group[q] = g as u8;
        qubit_pos[q] = pos as u8;
        group_sizes[g] = pos + 1;
        group_pext_masks[g] |= 1u64 << q;
    }

    let mut tables = empty_tables;

    for e in entries {
        match e {
            DiagEntry::Phase1q { qubit, d0, d1 } => {
                let g = qubit_group[*qubit] as usize;
                let p = qubit_pos[*qubit] as usize;
                let k = group_sizes[g];
                for (bits, entry) in tables[g][..1 << k].iter_mut().enumerate() {
                    if (bits >> p) & 1 == 1 {
                        *entry *= d1;
                    } else {
                        *entry *= d0;
                    }
                }
            }
            DiagEntry::Phase2q { q0, q1, phase } => {
                let g0 = qubit_group[*q0] as usize;
                let g1 = qubit_group[*q1] as usize;
                let p0 = qubit_pos[*q0] as usize;
                let p1 = qubit_pos[*q1] as usize;
                if g0 != g1 {
                    return None;
                }
                let g = g0;
                let k = group_sizes[g];
                for (bits, entry) in tables[g][..1 << k].iter_mut().enumerate() {
                    if ((bits >> p0) & 1 == 1) && ((bits >> p1) & 1 == 1) {
                        *entry *= phase;
                    }
                }
            }
            DiagEntry::Parity2q { q0, q1, same, diff } => {
                let g0 = qubit_group[*q0] as usize;
                let g1 = qubit_group[*q1] as usize;
                let p0 = qubit_pos[*q0] as usize;
                let p1 = qubit_pos[*q1] as usize;
                if g0 != g1 {
                    return None;
                }
                let g = g0;
                let k = group_sizes[g];
                for (bits, entry) in tables[g][..1 << k].iter_mut().enumerate() {
                    let parity = ((bits >> p0) ^ (bits >> p1)) & 1;
                    *entry *= if parity == 0 { *same } else { *diff };
                }
            }
        }
    }

    Some(DiagBatchTables {
        tables,
        unique_qubits,
        group_sizes,
        group_pext_masks,
        num_groups,
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,fma")]
unsafe fn apply_batch_phase_bmi2(
    state: &mut [Complex64],
    ctrl_bit: usize,
    groups: &[BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS],
    num_groups: usize,
) {
    use std::arch::x86_64::_pext_u64;
    let ctrl_mask = 1usize << ctrl_bit;
    let half = state.len() >> 1;
    for k in 0..half {
        let i = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
        let mut combined = Complex64::new(1.0, 0.0);
        for group in groups.iter().take(num_groups) {
            let bits = _pext_u64(i as u64, group.pext_mask) as usize;
            combined *= group.table[bits];
        }
        // SAFETY: insert_zero_bit(k, ctrl_bit) | ctrl_mask < state.len() for k < half
        *state.get_unchecked_mut(i) *= combined;
    }
}

#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[target_feature(enable = "bmi2,fma")]
unsafe fn batch_phase_tile_bmi2(
    tile: &mut [Complex64],
    base_idx: usize,
    ctrl_bit: usize,
    groups: &[BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS],
    num_groups: usize,
) {
    use std::arch::x86_64::_pext_u64;
    let ctrl_mask = 1usize << ctrl_bit;
    let tile_bits = tile.len().trailing_zeros() as usize;

    if ctrl_bit >= tile_bits {
        if (base_idx & ctrl_mask) == 0 {
            return;
        }
        for (j, amp) in tile.iter_mut().enumerate() {
            let i = base_idx + j;
            let mut combined = Complex64::new(1.0, 0.0);
            for group in groups.iter().take(num_groups) {
                let bits = _pext_u64(i as u64, group.pext_mask) as usize;
                combined *= group.table[bits];
            }
            *amp *= combined;
        }
    } else {
        let half = tile.len() >> 1;
        for k in 0..half {
            let j = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
            let i = base_idx + j;
            let mut combined = Complex64::new(1.0, 0.0);
            for group in groups.iter().take(num_groups) {
                let bits = _pext_u64(i as u64, group.pext_mask) as usize;
                combined *= group.table[bits];
            }
            // SAFETY: j < tile.len() guaranteed by insert_zero_bit mapping
            *tile.get_unchecked_mut(j) *= combined;
        }
    }
}

impl StatevectorBackend {
    #[inline(always)]
    pub(super) fn apply_single_gate(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_single_gate_par(target, mat);
            return;
        }

        let prepared = simd::PreparedGate1q::new(&mat);
        prepared.apply_full_sequential(&mut self.state, target);
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_single_gate_par(&mut self, target: usize, mat: [[Complex64; 2]; 2]) {
        let half = 1usize << target;
        let block_size = half << 1;
        let prepared = simd::PreparedGate1q::new(&mat);
        let state_len = self.state.len();

        const MIN_TILE: usize = 8192;

        let tile_size = MIN_TILE.max(block_size);
        let num_tiles = state_len / tile_size;

        if block_size <= MIN_TILE && num_tiles >= 4 {
            self.state.par_chunks_mut(MIN_TILE).for_each(|tile| {
                prepared.apply_full_sequential(tile, target);
            });
        } else if num_tiles >= 4 {
            self.state.par_chunks_mut(block_size).for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(half);
                prepared.apply(lo, hi);
            });
        } else {
            let sub_tile = MIN_TILE.min(half);
            for block in self.state.chunks_mut(block_size) {
                let (lo, hi) = block.split_at_mut(half);
                lo.par_chunks_mut(sub_tile)
                    .zip(hi.par_chunks_mut(sub_tile))
                    .for_each(|(lo_t, hi_t)| {
                        prepared.apply(lo_t, hi_t);
                    });
            }
        }
    }

    #[inline(always)]
    pub(super) fn apply_diagonal_gate(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_diagonal_gate_par(target, d0, d1);
            return;
        }

        let skip_lo = is_phase_one(d0);
        simd::apply_diagonal_sequential(&mut self.state, target, d0, d1, skip_lo);
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_diagonal_gate_par(&mut self, target: usize, d0: Complex64, d1: Complex64) {
        let skip_lo = is_phase_one(d0);

        const MIN_TILE: usize = 8192;
        let half = 1usize << target;
        let block_size = half << 1;
        let tile_size = MIN_TILE.max(block_size);

        self.state.par_chunks_mut(tile_size).for_each(|tile| {
            simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
        });
    }

    #[inline(always)]
    pub(super) fn apply_cx(&mut self, control: usize, target: usize) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_cx_par(control, target);
            return;
        }

        let ctrl_mask = 1usize << control;
        let tgt_mask = 1usize << target;
        let n = 1usize << self.num_qubits;

        for i in 0..n {
            if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
                let j = i | tgt_mask;
                self.state.swap(i, j);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_cx_par(&mut self, control: usize, target: usize) {
        if control > target {
            let ctrl_half = 1usize << control;
            let block_size = ctrl_half << 1;
            let tgt_half = 1usize << target;
            let tgt_block = tgt_half << 1;
            let num_blocks = self.state.len() / block_size;

            if num_blocks >= 4 {
                self.state
                    .par_chunks_mut(block_size)
                    .with_min_len(chunk_min_len(block_size))
                    .for_each(|chunk| {
                        let (_, hi) = chunk.split_at_mut(ctrl_half);
                        for sub in hi.chunks_mut(tgt_block) {
                            let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                            simd::swap_slices(sub_lo, sub_hi);
                        }
                    });
            } else {
                let inner_tile = MIN_PAR_ELEMS.max(tgt_block);
                for chunk in self.state.chunks_mut(block_size) {
                    let (_, hi) = chunk.split_at_mut(ctrl_half);
                    hi.par_chunks_mut(inner_tile).for_each(|tile| {
                        for sub in tile.chunks_mut(tgt_block) {
                            let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                            simd::swap_slices(sub_lo, sub_hi);
                        }
                    });
                }
            }
        } else {
            let tgt_half = 1usize << target;
            let block_size = tgt_half << 1;
            let ctrl_mask = 1usize << control;
            let num_blocks = self.state.len() / block_size;

            if num_blocks >= 4 {
                self.state
                    .par_chunks_mut(block_size)
                    .with_min_len(chunk_min_len(block_size))
                    .for_each(|chunk| {
                        let (lo, hi) = chunk.split_at_mut(tgt_half);
                        for k in 0..tgt_half {
                            if k & ctrl_mask != 0 {
                                std::mem::swap(&mut lo[k], &mut hi[k]);
                            }
                        }
                    });
            } else {
                for chunk in self.state.chunks_mut(block_size) {
                    let (lo, hi) = chunk.split_at_mut(tgt_half);
                    lo.par_chunks_mut(MIN_PAR_ELEMS)
                        .zip(hi.par_chunks_mut(MIN_PAR_ELEMS))
                        .enumerate()
                        .for_each(|(tile_idx, (lo_tile, hi_tile))| {
                            let offset = tile_idx * MIN_PAR_ELEMS;
                            for k in 0..lo_tile.len() {
                                if (offset + k) & ctrl_mask != 0 {
                                    std::mem::swap(&mut lo_tile[k], &mut hi_tile[k]);
                                }
                            }
                        });
                }
            }
        }
    }

    #[inline(always)]
    pub(super) fn apply_cz(&mut self, q0: usize, q1: usize) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_cz_par(q0, q1);
            return;
        }

        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let n = 1usize << self.num_qubits;

        for i in 0..n {
            if (i & mask0) != 0 && (i & mask1) != 0 {
                self.state[i] = -self.state[i];
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_cz_par(&mut self, q0: usize, q1: usize) {
        let (lo_q, hi_q) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let lo_half = 1usize << lo_q;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi_q;
        let block_size = hi_half << 1;
        let num_blocks = self.state.len() / block_size;

        if num_blocks >= 4 {
            self.state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| {
                    let (_, hi_group) = chunk.split_at_mut(hi_half);
                    for sub in hi_group.chunks_mut(lo_block) {
                        let (_, sub_hi) = sub.split_at_mut(lo_half);
                        simd::negate_slice(sub_hi);
                    }
                });
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(lo_block);
            for chunk in self.state.chunks_mut(block_size) {
                let (_, hi_group) = chunk.split_at_mut(hi_half);
                hi_group.par_chunks_mut(inner_tile).for_each(|tile| {
                    for sub in tile.chunks_mut(lo_block) {
                        let (_, sub_hi) = sub.split_at_mut(lo_half);
                        simd::negate_slice(sub_hi);
                    }
                });
            }
        }
    }

    #[inline(always)]
    pub(super) fn apply_rzz(&mut self, q0: usize, q1: usize, theta: f64) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_rzz_par(q0, q1, theta);
            return;
        }

        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        let phases = [phase_same, phase_diff];

        for (i, amp) in self.state.iter_mut().enumerate() {
            let parity = ((i >> q0) ^ (i >> q1)) & 1;
            *amp *= phases[parity];
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_rzz_par(&mut self, q0: usize, q1: usize, theta: f64) {
        let phase_same = Complex64::from_polar(1.0, -theta / 2.0);
        let phase_diff = Complex64::from_polar(1.0, theta / 2.0);
        let phases = [phase_same, phase_diff];

        self.state
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
    pub(super) fn apply_batch_rzz(&mut self, edges: &[(usize, usize, f64)]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_batch_rzz_par(edges);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_bmi2() && simd::has_fma() {
            let one = Complex64::new(1.0, 0.0);
            let mut bmi2_groups = [BatchRzzBmi2Group {
                table: [one; BATCH_RZZ_BMI2_TABLE_SIZE],
                pext_mask: 0,
            }; MAX_BATCH_RZZ_GROUPS];
            if let Some(num_groups) = build_batch_rzz_bmi2_tables(edges, &mut bmi2_groups) {
                // SAFETY: has_bmi2() && has_fma() verified above
                unsafe { apply_batch_rzz_bmi2(&mut self.state, &bmi2_groups, num_groups) };
                return;
            }
        }

        let one = Complex64::new(1.0, 0.0);
        let mut groups = [BatchRzzGroup {
            table: [one; BATCH_RZZ_TABLE_SIZE],
            q0s: [0; BATCH_RZZ_GROUP_SIZE],
            q1s: [0; BATCH_RZZ_GROUP_SIZE],
            len: 0,
        }; MAX_BATCH_RZZ_GROUPS];
        let num_groups = build_batch_rzz_tables(edges, &mut groups);

        for (i, amp) in self.state.iter_mut().enumerate() {
            let mut combined = one;
            for group in groups.iter().take(num_groups) {
                let bits = extract_rzz_bits(i, group);
                combined *= group.table[bits];
            }
            *amp *= combined;
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_batch_rzz_par(&mut self, edges: &[(usize, usize, f64)]) {
        #[cfg(target_arch = "x86_64")]
        if simd::has_bmi2() && simd::has_fma() {
            let one = Complex64::new(1.0, 0.0);
            let mut bmi2_groups = [BatchRzzBmi2Group {
                table: [one; BATCH_RZZ_BMI2_TABLE_SIZE],
                pext_mask: 0,
            }; MAX_BATCH_RZZ_GROUPS];
            if let Some(num_groups) = build_batch_rzz_bmi2_tables(edges, &mut bmi2_groups) {
                self.state
                    .par_chunks_mut(MIN_PAR_ELEMS)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base = chunk_idx * MIN_PAR_ELEMS;
                        // SAFETY: has_bmi2() && has_fma() verified above
                        unsafe {
                            batch_rzz_tile_bmi2(chunk, base, &bmi2_groups, num_groups);
                        }
                    });
                return;
            }
        }

        let one = Complex64::new(1.0, 0.0);
        let mut groups = [BatchRzzGroup {
            table: [one; BATCH_RZZ_TABLE_SIZE],
            q0s: [0; BATCH_RZZ_GROUP_SIZE],
            q1s: [0; BATCH_RZZ_GROUP_SIZE],
            len: 0,
        }; MAX_BATCH_RZZ_GROUPS];
        let num_groups = build_batch_rzz_tables(edges, &mut groups);

        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * MIN_PAR_ELEMS;
                for (j, amp) in chunk.iter_mut().enumerate() {
                    let i = base + j;
                    let mut combined = Complex64::new(1.0, 0.0);
                    for group in groups.iter().take(num_groups) {
                        let bits = extract_rzz_bits(i, group);
                        combined *= group.table[bits];
                    }
                    *amp *= combined;
                }
            });
    }

    #[inline(always)]
    pub(super) fn apply_cu(&mut self, control: usize, target: usize, mat: [[Complex64; 2]; 2]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_cu_par(control, target, mat);
            return;
        }

        let prepared = simd::PreparedGate1q::new(&mat);

        if control > target {
            let ctrl_half = 1usize << control;
            let block_size = ctrl_half << 1;
            for chunk in self.state.chunks_mut(block_size) {
                let (_, hi) = chunk.split_at_mut(ctrl_half);
                prepared.apply_full_sequential(hi, target);
            }
        } else {
            let ctrl_mask = 1usize << control;
            let tgt_mask = 1usize << target;
            let num_iters = 1usize << (self.num_qubits - 2);
            let base_ptr = self.state.as_mut_ptr() as *mut f64;
            for i in 0..num_iters {
                let base = insert_zero_bit(insert_zero_bit(i, control), target);
                let idx0 = base | ctrl_mask;
                let idx1 = idx0 | tgt_mask;
                // SAFETY: indices from insert_zero_bit bijection are in-bounds and disjoint.
                unsafe {
                    prepared.apply_pair_ptr(base_ptr.add(idx0 * 2), base_ptr.add(idx1 * 2));
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_cu_par(&mut self, control: usize, target: usize, mat: [[Complex64; 2]; 2]) {
        if control > target {
            let ctrl_half = 1usize << control;
            let block_size = ctrl_half << 1;
            let prepared = simd::PreparedGate1q::new(&mat);
            let num_blocks = self.state.len() / block_size;

            if num_blocks >= 4 {
                self.state
                    .par_chunks_mut(block_size)
                    .with_min_len(chunk_min_len(block_size))
                    .for_each(|chunk| {
                        let (_, hi) = chunk.split_at_mut(ctrl_half);
                        prepared.apply_full_sequential(hi, target);
                    });
            } else {
                let tgt_half = 1usize << target;
                let tgt_block = tgt_half << 1;
                let inner_tile = MIN_PAR_ELEMS.max(tgt_block);
                for chunk in self.state.chunks_mut(block_size) {
                    let (_, hi) = chunk.split_at_mut(ctrl_half);
                    hi.par_chunks_mut(inner_tile).for_each(|tile| {
                        prepared.apply_full_sequential(tile, target);
                    });
                }
            }
        } else {
            let ctrl_mask = 1usize << control;
            let tgt_mask = 1usize << target;
            let lo = control;
            let hi = target;
            let num_iters = 1usize << (self.num_qubits - 2);
            let ptr = SendPtr(self.state.as_mut_ptr());
            let prepared = simd::PreparedGate1q::new(&mat);

            // SAFETY: insert_zero_bit bijection → disjoint index pairs per iteration.
            (0..num_iters)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .for_each(move |i| {
                    let base = insert_zero_bit(insert_zero_bit(i, lo), hi);
                    let idx0 = base | ctrl_mask;
                    let idx1 = idx0 | tgt_mask;
                    unsafe {
                        let fp = ptr.as_f64_ptr();
                        prepared.apply_pair_ptr(fp.add(idx0 * 2), fp.add(idx1 * 2));
                    }
                });
        }
    }

    #[inline(always)]
    pub(super) fn apply_mcu(
        &mut self,
        controls: &[usize],
        target: usize,
        mat: [[Complex64; 2]; 2],
    ) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_mcu_par(controls, target, mat);
            return;
        }

        let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
        let tgt_mask = 1usize << target;
        let mut sorted_buf = [0usize; 10];
        let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
        let sorted = &sorted_buf[..num_special];

        let num_iters = 1usize << (self.num_qubits - num_special);
        let prepared = simd::PreparedGate1q::new(&mat);
        let base_ptr = self.state.as_mut_ptr() as *mut f64;

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
    fn apply_mcu_par(&mut self, controls: &[usize], target: usize, mat: [[Complex64; 2]; 2]) {
        let ctrl_mask: usize = controls.iter().map(|&q| 1usize << q).fold(0, |a, b| a | b);
        let tgt_mask = 1usize << target;
        let mut sorted_buf = [0usize; 10];
        let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
        let sorted = &sorted_buf[..num_special];

        let num_iters = 1usize << (self.num_qubits - num_special);
        let ptr = SendPtr(self.state.as_mut_ptr());
        let prepared = simd::PreparedGate1q::new(&mat);

        // SAFETY: insert_zero_bit bijection → disjoint index pairs per iteration.
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
                unsafe {
                    let fp = ptr.as_f64_ptr();
                    prepared.apply_pair_ptr(fp.add(idx0 * 2), fp.add(idx1 * 2));
                }
            });
    }

    #[inline(always)]
    pub(super) fn apply_cu_phase(&mut self, control: usize, target: usize, phase: Complex64) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_cu_phase_par(control, target, phase);
            return;
        }

        let (lo, hi) = if control < target {
            (control, target)
        } else {
            (target, control)
        };
        let lo_half = 1usize << lo;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi;
        let block_size = hi_half << 1;

        for chunk in self.state.chunks_mut(block_size) {
            let hi_group = &mut chunk[hi_half..];
            for sub in hi_group.chunks_mut(lo_block) {
                simd::scale_complex_slice(&mut sub[lo_half..], phase);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_cu_phase_par(&mut self, control: usize, target: usize, phase: Complex64) {
        let (lo, hi) = if control < target {
            (control, target)
        } else {
            (target, control)
        };
        let lo_half = 1usize << lo;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi;
        let block_size = hi_half << 1;
        let num_blocks = self.state.len() / block_size;

        if num_blocks >= 4 {
            self.state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| {
                    let hi_group = &mut chunk[hi_half..];
                    for sub in hi_group.chunks_mut(lo_block) {
                        simd::scale_complex_slice(&mut sub[lo_half..], phase);
                    }
                });
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(lo_block);
            for chunk in self.state.chunks_mut(block_size) {
                let hi_group = &mut chunk[hi_half..];
                hi_group.par_chunks_mut(inner_tile).for_each(|tile| {
                    for sub in tile.chunks_mut(lo_block) {
                        simd::scale_complex_slice(&mut sub[lo_half..], phase);
                    }
                });
            }
        }
    }

    #[inline(always)]
    pub(super) fn apply_mcu_phase(&mut self, controls: &[usize], target: usize, phase: Complex64) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_mcu_phase_par(controls, target, phase);
            return;
        }

        let all_mask: usize = controls
            .iter()
            .chain(std::iter::once(&target))
            .map(|&q| 1usize << q)
            .fold(0, |a, b| a | b);
        let mut sorted_buf = [0usize; 10];
        let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
        let sorted = &sorted_buf[..num_special];

        let num_iters = 1usize << (self.num_qubits - num_special);
        for i in 0..num_iters {
            let mut base = i;
            for &q in sorted {
                base = insert_zero_bit(base, q);
            }
            self.state[base | all_mask] *= phase;
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_mcu_phase_par(&mut self, controls: &[usize], target: usize, phase: Complex64) {
        let all_mask: usize = controls
            .iter()
            .chain(std::iter::once(&target))
            .map(|&q| 1usize << q)
            .fold(0, |a, b| a | b);
        let mut sorted_buf = [0usize; 10];
        let num_special = sorted_mcu_qubits(controls, target, &mut sorted_buf);
        let sorted = &sorted_buf[..num_special];

        let num_iters = 1usize << (self.num_qubits - num_special);
        let ptr = SendPtr(self.state.as_mut_ptr());

        // SAFETY: insert_zero_bit bijection → disjoint indices per iteration.
        (0..num_iters)
            .into_par_iter()
            .with_min_len(MIN_PAR_ITERS)
            .for_each(move |i| {
                let mut base = i;
                for &q in sorted {
                    base = insert_zero_bit(base, q);
                }
                let idx = base | all_mask;
                unsafe {
                    let val = ptr.load(idx);
                    ptr.store(idx, val * phase);
                }
            });
    }

    #[inline(always)]
    pub(super) fn apply_swap(&mut self, q0: usize, q1: usize) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_swap_par(q0, q1);
            return;
        }

        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let lo_half = 1usize << lo;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi;
        let block_size = hi_half << 1;

        for chunk in self.state.chunks_mut(block_size) {
            let (lo_group, hi_group) = chunk.split_at_mut(hi_half);
            for (lo_sub, hi_sub) in lo_group
                .chunks_mut(lo_block)
                .zip(hi_group.chunks_mut(lo_block))
            {
                let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
                let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
                simd::swap_slices(lo_sub_hi, hi_sub_lo);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_swap_par(&mut self, q0: usize, q1: usize) {
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let lo_half = 1usize << lo;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi;
        let block_size = hi_half << 1;
        let num_blocks = self.state.len() / block_size;

        if num_blocks >= 4 {
            self.state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| {
                    let (lo_group, hi_group) = chunk.split_at_mut(hi_half);
                    let lo_subs = lo_group.chunks_mut(lo_block);
                    let hi_subs = hi_group.chunks_mut(lo_block);
                    for (lo_sub, hi_sub) in lo_subs.zip(hi_subs) {
                        let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
                        let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
                        simd::swap_slices(lo_sub_hi, hi_sub_lo);
                    }
                });
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(lo_block);
            for chunk in self.state.chunks_mut(block_size) {
                let (lo_group, hi_group) = chunk.split_at_mut(hi_half);
                lo_group
                    .par_chunks_mut(inner_tile)
                    .zip(hi_group.par_chunks_mut(inner_tile))
                    .for_each(|(lo_tile, hi_tile)| {
                        for (lo_sub, hi_sub) in lo_tile
                            .chunks_mut(lo_block)
                            .zip(hi_tile.chunks_mut(lo_block))
                        {
                            let (_, lo_sub_hi) = lo_sub.split_at_mut(lo_half);
                            let (hi_sub_lo, _) = hi_sub.split_at_mut(lo_half);
                            simd::swap_slices(lo_sub_hi, hi_sub_lo);
                        }
                    });
            }
        }
    }

    #[inline(always)]
    pub(super) fn apply_fused_2q(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        if q0.max(q1) - q0.min(q1) == 1 {
            #[cfg(feature = "parallel")]
            if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
                self.apply_fused_2q_adjacent_par(q0, q1, mat);
                return;
            }

            self.apply_fused_2q_adjacent(q0, q1, mat);
            return;
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_fused_2q_par(q0, q1, mat);
            return;
        }

        let prepared = simd::PreparedGate2q::new(mat);
        prepared.apply_full(&mut self.state, self.num_qubits, q0, q1);
    }

    #[inline(always)]
    fn apply_fused_2q_adjacent(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let lo = q0.min(q1);
        let stride = 1usize << lo;
        let block_size = stride << 2;
        let q0_is_lo = q0 == lo;
        let prepared = simd::PreparedGate2q::new(mat);

        for chunk in self.state.chunks_mut(block_size) {
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
    fn apply_fused_2q_par(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let mask0 = 1usize << q0;
        let mask1 = 1usize << q1;
        let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let n_iter = 1usize << (self.num_qubits - 2);
        let ptr = SendPtr(self.state.as_mut_ptr());
        let prepared = simd::PreparedGate2q::new(mat);

        // SAFETY: insert_zero_bit bijection → disjoint index groups per iteration.
        (0..n_iter)
            .into_par_iter()
            .with_min_len(MIN_PAR_ITERS)
            .for_each(move |k| {
                let base = insert_zero_bit(insert_zero_bit(k, lo), hi);
                let i = [base, base | mask1, base | mask0, base | mask0 | mask1];
                // SAFETY: disjoint groups, ptr valid for state lifetime.
                unsafe {
                    prepared.apply_group_ptr(ptr.as_f64_ptr(), i);
                }
            });
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_fused_2q_adjacent_par(&mut self, q0: usize, q1: usize, mat: &[[Complex64; 4]; 4]) {
        let lo = q0.min(q1);
        let stride = 1usize << lo;
        let block_size = stride << 2;
        let q0_is_lo = q0 == lo;
        let n_groups = self.state.len() >> 2;
        let ptr = SendPtr(self.state.as_mut_ptr());
        let prepared = simd::PreparedGate2q::new(mat);

        (0..n_groups)
            .into_par_iter()
            .with_min_len(MIN_PAR_ITERS)
            .for_each(move |group| {
                Self::apply_fused_2q_adjacent_group(
                    ptr, group, lo, stride, block_size, q0_is_lo, &prepared,
                );
            });
    }

    #[inline(always)]
    pub(super) fn apply_batch_phase(&mut self, control: usize, phases: &[(usize, Complex64)]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_batch_phase_par(control, phases);
            return;
        }

        let one = Complex64::new(1.0, 0.0);

        let mut groups = [BatchPhaseGroup {
            table: [one; BATCH_PHASE_TABLE_SIZE],
            shifts: [0; BATCH_PHASE_GROUP_SIZE],
            len: 0,
            pext_mask: 0,
        }; MAX_BATCH_PHASE_GROUPS];
        let num_groups = build_batch_phase_tables(phases, &mut groups);

        #[cfg(target_arch = "x86_64")]
        if simd::has_bmi2() && simd::has_fma() {
            // SAFETY: BMI2+FMA availability verified above.
            unsafe {
                apply_batch_phase_bmi2(&mut self.state, control, &groups, num_groups);
            }
            return;
        }

        let ctrl_mask = 1usize << control;
        let half = 1usize << (self.num_qubits - 1);
        for k in 0..half {
            let i = insert_zero_bit(k, control) | ctrl_mask;
            let mut combined = one;
            for group in groups.iter().take(num_groups) {
                let bits = extract_bits(i, &group.shifts, group.len);
                combined *= group.table[bits];
            }
            self.state[i] *= combined;
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_batch_phase_par(&mut self, control: usize, phases: &[(usize, Complex64)]) {
        let one = Complex64::new(1.0, 0.0);

        let mut groups = [BatchPhaseGroup {
            table: [one; BATCH_PHASE_TABLE_SIZE],
            shifts: [0; BATCH_PHASE_GROUP_SIZE],
            len: 0,
            pext_mask: 0,
        }; MAX_BATCH_PHASE_GROUPS];
        let num_groups = build_batch_phase_tables(phases, &mut groups);

        #[cfg(target_arch = "x86_64")]
        let use_bmi2 = simd::has_bmi2() && simd::has_fma();

        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(tile_idx, tile)| {
                let base = tile_idx * MIN_PAR_ELEMS;

                #[cfg(target_arch = "x86_64")]
                if use_bmi2 {
                    // SAFETY: BMI2+FMA availability verified above.
                    unsafe {
                        batch_phase_tile_bmi2(tile, base, control, &groups, num_groups);
                    }
                    return;
                }

                let ctrl_mask = 1usize << control;
                let tile_bits = tile.len().trailing_zeros() as usize;
                if control >= tile_bits {
                    if (base & ctrl_mask) == 0 {
                        return;
                    }
                    for (j, amp) in tile.iter_mut().enumerate() {
                        let i = base + j;
                        let mut combined = one;
                        for group in groups.iter().take(num_groups) {
                            let bits = extract_bits(i, &group.shifts, group.len);
                            combined *= group.table[bits];
                        }
                        *amp *= combined;
                    }
                } else {
                    let half = tile.len() >> 1;
                    for k in 0..half {
                        let j = insert_zero_bit(k, control) | ctrl_mask;
                        let i = base + j;
                        let mut combined = one;
                        for group in groups.iter().take(num_groups) {
                            let bits = extract_bits(i, &group.shifts, group.len);
                            combined *= group.table[bits];
                        }
                        tile[j] *= combined;
                    }
                }
            });
    }

    #[inline(always)]
    pub(super) fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_measure_par(qubit, classical_bit);
            return;
        }

        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;
        let zero = Complex64::new(0.0, 0.0);

        let mut prob_one = 0.0f64;
        for block in self.state.chunks(block_size) {
            for amp in &block[half..] {
                prob_one += amp.norm_sqr();
            }
        }
        prob_one *= norm_sq;

        let outcome = self.rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;

        let inv_norm = measurement_inv_norm(outcome, prob_one);

        for block in self.state.chunks_mut(block_size) {
            if outcome {
                for amp in &mut block[..half] {
                    *amp = zero;
                }
            } else {
                for amp in &mut block[half..] {
                    *amp = zero;
                }
            }
        }

        self.pending_norm *= inv_norm;
    }

    #[inline(always)]
    pub(super) fn qubit_probability_one(&self, qubit: usize) -> f64 {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            return self.qubit_probability_one_par(qubit);
        }

        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        let mut prob_one = 0.0f64;
        for block in self.state.chunks(block_size) {
            prob_one += simd::norm_sqr_sum(&block[half..]);
        }
        prob_one * norm_sq
    }

    #[cfg(feature = "parallel")]
    fn qubit_probability_one_par(&self, qubit: usize) -> f64 {
        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        self.state
            .par_chunks(block_size)
            .with_min_len(chunk_min_len(block_size))
            .map(|chunk| simd::norm_sqr_sum(&chunk[half..]))
            .sum::<f64>()
            * norm_sq
    }

    #[inline(always)]
    pub(super) fn reduced_density_matrix_one(&self, qubit: usize) -> [[Complex64; 2]; 2] {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            return self.reduced_density_matrix_one_par(qubit);
        }

        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        let mut r = Complex64::new(0.0, 0.0);
        for block in self.state.chunks(block_size) {
            let (lo, hi) = block.split_at(half);
            for i in 0..half {
                let a0 = lo[i];
                let a1 = hi[i];
                p0 += a0.norm_sqr();
                p1 += a1.norm_sqr();
                r += a1 * a0.conj();
            }
        }

        let scale = Complex64::new(norm_sq, 0.0);
        let r = r * scale;
        [
            [Complex64::new(p0 * norm_sq, 0.0), r.conj()],
            [r, Complex64::new(p1 * norm_sq, 0.0)],
        ]
    }

    #[cfg(feature = "parallel")]
    fn reduced_density_matrix_one_par(&self, qubit: usize) -> [[Complex64; 2]; 2] {
        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        let (p0, p1, r) = self
            .state
            .par_chunks(block_size)
            .with_min_len(chunk_min_len(block_size))
            .map(|block| {
                let (lo, hi) = block.split_at(half);
                let mut p0 = 0.0f64;
                let mut p1 = 0.0f64;
                let mut r = Complex64::new(0.0, 0.0);
                for i in 0..half {
                    let a0 = lo[i];
                    let a1 = hi[i];
                    p0 += a0.norm_sqr();
                    p1 += a1.norm_sqr();
                    r += a1 * a0.conj();
                }
                (p0, p1, r)
            })
            .reduce(
                || (0.0, 0.0, Complex64::new(0.0, 0.0)),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
            );

        let scale = Complex64::new(norm_sq, 0.0);
        let r = r * scale;
        [
            [Complex64::new(p0 * norm_sq, 0.0), r.conj()],
            [r, Complex64::new(p1 * norm_sq, 0.0)],
        ]
    }

    #[inline(always)]
    pub(super) fn apply_reset(&mut self, qubit: usize) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_reset_par(qubit);
            return;
        }

        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;
        let zero = Complex64::new(0.0, 0.0);

        let mut prob_zero = 0.0f64;
        for block in self.state.chunks(block_size) {
            for amp in &block[..half] {
                prob_zero += amp.norm_sqr();
            }
        }
        prob_zero *= norm_sq;

        if prob_zero > 0.0 {
            for block in self.state.chunks_mut(block_size) {
                for amp in &mut block[half..] {
                    *amp = zero;
                }
            }
            let inv_norm = 1.0 / prob_zero.sqrt();
            self.pending_norm *= inv_norm;
        } else {
            for amp in self.state.iter_mut() {
                *amp = zero;
            }
            self.state[0] = Complex64::new(1.0, 0.0);
            self.pending_norm = 1.0;
        }
    }

    /// Apply multiple single-qubit gates in a multi-tier tiled pass.
    ///
    /// Three tiers based on gate target qubit:
    /// - **L2 tier** (target 0–13): 256KB tiles, all applied per-tile in L2 cache
    /// - **L3 tier** (target 14–16): 2MB tiles, applied per-tile in L3 cache
    /// - **Individual** (target 17+): separate full-state passes
    #[inline(always)]
    pub(super) fn apply_multi_1q(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_single_gate(gates[0].0, gates[0].1);
            return;
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_multi_1q_par(gates);
            return;
        }

        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, simd::PreparedGate1q); 16]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, simd::PreparedGate1q); 4]> = SmallVec::new();
        let mut large_gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 4]> = SmallVec::new();

        for &(target, mat) in gates {
            if target <= max_l2_target {
                small_gates.push((target, simd::PreparedGate1q::new(&mat)));
            } else if target <= max_l3_target {
                medium_gates.push((target, simd::PreparedGate1q::new(&mat)));
            } else {
                large_gates.push((target, mat));
            }
        }

        if !small_gates.is_empty() {
            let outer_block = 1usize << (max_l2_target + 1);
            let tile_size = MULTI_TILE.max(outer_block);
            for tile in self.state.chunks_mut(tile_size) {
                for &(target, ref prepared) in &small_gates {
                    prepared.apply_tiled(tile, target);
                }
            }
        }

        if !medium_gates.is_empty() {
            let outer_block = 1usize << (max_l3_target + 1);
            let tile_size = L3_TILE.max(outer_block);
            for tile in self.state.chunks_mut(tile_size) {
                for &(target, ref prepared) in &medium_gates {
                    prepared.apply_tiled(tile, target);
                }
            }
        }

        for (target, mat) in large_gates {
            self.apply_single_gate(target, mat);
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_multi_1q_par(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, simd::PreparedGate1q); 16]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, simd::PreparedGate1q); 4]> = SmallVec::new();
        let mut large_gates: SmallVec<[(usize, [[Complex64; 2]; 2]); 4]> = SmallVec::new();

        for &(target, mat) in gates {
            if target <= max_l2_target {
                small_gates.push((target, simd::PreparedGate1q::new(&mat)));
            } else if target <= max_l3_target {
                medium_gates.push((target, simd::PreparedGate1q::new(&mat)));
            } else {
                large_gates.push((target, mat));
            }
        }

        if !small_gates.is_empty() {
            let outer_block = 1usize << (max_l2_target + 1);
            let tile_size = MULTI_TILE.max(outer_block);
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    for &(target, ref prepared) in &small_gates {
                        prepared.apply_tiled(tile, target);
                    }
                });
        }

        if !medium_gates.is_empty() {
            let outer_block = 1usize << (max_l3_target + 1);
            let tile_size = L3_TILE.max(outer_block);
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    for &(target, ref prepared) in &medium_gates {
                        prepared.apply_tiled(tile, target);
                    }
                });
        }

        for (target, mat) in large_gates {
            self.apply_single_gate(target, mat);
        }
    }

    pub(super) fn apply_multi_1q_diagonal(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_diagonal_gate(gates[0].0, gates[0].1[0][0], gates[0].1[1][1]);
            return;
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_multi_1q_diagonal_par(gates);
            return;
        }

        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, Complex64, Complex64, bool); 16]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, Complex64, Complex64, bool); 4]> = SmallVec::new();
        let mut large_gates: SmallVec<[(usize, Complex64, Complex64); 4]> = SmallVec::new();

        for &(target, mat) in gates {
            let d0 = mat[0][0];
            let d1 = mat[1][1];
            let skip_lo = is_phase_one(d0);
            if target <= max_l2_target {
                small_gates.push((target, d0, d1, skip_lo));
            } else if target <= max_l3_target {
                medium_gates.push((target, d0, d1, skip_lo));
            } else {
                large_gates.push((target, d0, d1));
            }
        }

        if !small_gates.is_empty() {
            let outer_block = 1usize << (max_l2_target + 1);
            let tile_size = MULTI_TILE.max(outer_block);
            for tile in self.state.chunks_mut(tile_size) {
                for &(target, d0, d1, skip_lo) in &small_gates {
                    simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                }
            }
        }

        if !medium_gates.is_empty() {
            let outer_block = 1usize << (max_l3_target + 1);
            let tile_size = L3_TILE.max(outer_block);
            for tile in self.state.chunks_mut(tile_size) {
                for &(target, d0, d1, skip_lo) in &medium_gates {
                    simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                }
            }
        }

        for (target, d0, d1) in large_gates {
            let skip_lo = is_phase_one(d0);
            simd::apply_diagonal_sequential(&mut self.state, target, d0, d1, skip_lo);
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_multi_1q_diagonal_par(&mut self, gates: &[(usize, [[Complex64; 2]; 2])]) {
        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, Complex64, Complex64, bool); 16]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, Complex64, Complex64, bool); 4]> = SmallVec::new();
        let mut large_gates: SmallVec<[(usize, Complex64, Complex64); 4]> = SmallVec::new();

        for &(target, mat) in gates {
            let d0 = mat[0][0];
            let d1 = mat[1][1];
            let skip_lo = is_phase_one(d0);
            if target <= max_l2_target {
                small_gates.push((target, d0, d1, skip_lo));
            } else if target <= max_l3_target {
                medium_gates.push((target, d0, d1, skip_lo));
            } else {
                large_gates.push((target, d0, d1));
            }
        }

        if !small_gates.is_empty() {
            let outer_block = 1usize << (max_l2_target + 1);
            let tile_size = MULTI_TILE.max(outer_block);
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    for &(target, d0, d1, skip_lo) in &small_gates {
                        simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                    }
                });
        }

        if !medium_gates.is_empty() {
            let outer_block = 1usize << (max_l3_target + 1);
            let tile_size = L3_TILE.max(outer_block);
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    for &(target, d0, d1, skip_lo) in &medium_gates {
                        simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
                    }
                });
        }

        for (target, d0, d1) in large_gates {
            let skip_lo = is_phase_one(d0);
            let min_tile = 1usize << (target + 1);
            let tile_size = min_tile.max(MIN_PAR_ELEMS);
            self.state.par_chunks_mut(tile_size).for_each(|tile| {
                simd::apply_diagonal_sequential(tile, target, d0, d1, skip_lo);
            });
        }
    }

    /// Apply multiple two-qubit gates in a cache-tiled pass.
    ///
    /// Partitions gates by `max(q0, q1)` into three tiers matching `apply_multi_1q`:
    /// - **L2** (max qubit ≤ 13): 16K-element tiles (256 KB)
    /// - **L3** (max qubit ≤ 16): 131K-element tiles (2 MB)
    /// - **Individual** (max qubit > 16): per-gate full-state passes
    ///
    /// Within each tier the gates are applied sequentially per tile, keeping data
    /// cache-resident. `PreparedGate2q::apply_full` works on sub-slices because it
    /// uses `1 << (num_qubits - 2)` for iteration, relative to slice length.
    #[inline(always)]
    pub(super) fn apply_multi_2q(&mut self, gates: &[(usize, usize, [[Complex64; 4]; 4])]) {
        if gates.is_empty() {
            return;
        }
        if gates.len() == 1 {
            self.apply_fused_2q(gates[0].0, gates[0].1, &gates[0].2);
            return;
        }

        #[cfg(feature = "parallel")]
        if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS {
            self.apply_multi_2q_par(gates);
            return;
        }

        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, usize, simd::PreparedGate2q); 2]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, usize, simd::PreparedGate2q); 2]> = SmallVec::new();

        for &(q0, q1, ref mat) in gates {
            let max_q = q0.max(q1);
            if max_q <= max_l2_target {
                small_gates.push((q0, q1, simd::PreparedGate2q::new(mat)));
            } else if max_q <= max_l3_target {
                medium_gates.push((q0, q1, simd::PreparedGate2q::new(mat)));
            }
        }

        if !small_gates.is_empty() {
            let tile_size = MULTI_TILE;
            let tile_qubits = tile_size.trailing_zeros() as usize;
            for tile in self.state.chunks_mut(tile_size) {
                let n = tile.len().trailing_zeros() as usize;
                for &(q0, q1, ref prepared) in &small_gates {
                    prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                }
            }
        }

        if !medium_gates.is_empty() {
            let tile_size = L3_TILE;
            let tile_qubits = tile_size.trailing_zeros() as usize;
            for tile in self.state.chunks_mut(tile_size) {
                let n = tile.len().trailing_zeros() as usize;
                for &(q0, q1, ref prepared) in &medium_gates {
                    prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                }
            }
        }

        for &(q0, q1, ref mat) in gates {
            if q0.max(q1) > max_l3_target {
                self.apply_fused_2q(q0, q1, mat);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_multi_2q_par(&mut self, gates: &[(usize, usize, [[Complex64; 4]; 4])]) {
        const MULTI_TILE: usize = 16384;
        const L3_TILE: usize = 131072;

        let max_l2_target = max_target_for_tile(MULTI_TILE);
        let max_l3_target = max_target_for_tile(L3_TILE);

        let mut small_gates: SmallVec<[(usize, usize, simd::PreparedGate2q); 2]> = SmallVec::new();
        let mut medium_gates: SmallVec<[(usize, usize, simd::PreparedGate2q); 2]> = SmallVec::new();

        for &(q0, q1, ref mat) in gates {
            let max_q = q0.max(q1);
            if max_q <= max_l2_target {
                small_gates.push((q0, q1, simd::PreparedGate2q::new(mat)));
            } else if max_q <= max_l3_target {
                medium_gates.push((q0, q1, simd::PreparedGate2q::new(mat)));
            }
        }

        if !small_gates.is_empty() {
            let tile_size = MULTI_TILE;
            let tile_qubits = tile_size.trailing_zeros() as usize;
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    let n = tile.len().trailing_zeros() as usize;
                    for &(q0, q1, ref prepared) in &small_gates {
                        prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                    }
                });
        }

        if !medium_gates.is_empty() {
            let tile_size = L3_TILE;
            let tile_qubits = tile_size.trailing_zeros() as usize;
            self.state
                .par_chunks_mut(tile_size)
                .with_min_len(chunk_min_len(tile_size))
                .for_each(|tile| {
                    let n = tile.len().trailing_zeros() as usize;
                    for &(q0, q1, ref prepared) in &medium_gates {
                        prepared.apply_tiled(tile, n.min(tile_qubits), q0, q1);
                    }
                });
        }

        for &(q0, q1, ref mat) in gates {
            if q0.max(q1) > max_l3_target {
                self.apply_fused_2q(q0, q1, mat);
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_measure_par(&mut self, qubit: usize, classical_bit: usize) {
        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        let prob_one: f64 = self
            .state
            .par_chunks(block_size)
            .with_min_len(chunk_min_len(block_size))
            .map(|chunk| simd::norm_sqr_sum(&chunk[half..]))
            .sum::<f64>()
            * norm_sq;

        let outcome = self.rng.random::<f64>() < prob_one;
        self.classical_bits[classical_bit] = outcome;

        let inv_norm = measurement_inv_norm(outcome, prob_one);

        self.state
            .par_chunks_mut(block_size)
            .with_min_len(chunk_min_len(block_size))
            .for_each(|chunk| {
                let (lo, hi) = chunk.split_at_mut(half);
                if outcome {
                    simd::zero_slice(lo);
                } else {
                    simd::zero_slice(hi);
                }
            });

        self.pending_norm *= inv_norm;
    }

    #[cfg(feature = "parallel")]
    fn apply_reset_par(&mut self, qubit: usize) {
        let half = 1usize << qubit;
        let block_size = half << 1;
        let norm_sq = self.pending_norm * self.pending_norm;

        let prob_zero: f64 = self
            .state
            .par_chunks(block_size)
            .with_min_len(chunk_min_len(block_size))
            .map(|chunk| simd::norm_sqr_sum(&chunk[..half]))
            .sum::<f64>()
            * norm_sq;

        if prob_zero > 0.0 {
            self.state
                .par_chunks_mut(block_size)
                .with_min_len(chunk_min_len(block_size))
                .for_each(|chunk| {
                    let (_, hi) = chunk.split_at_mut(half);
                    simd::zero_slice(hi);
                });
            let inv_norm = 1.0 / prob_zero.sqrt();
            self.pending_norm *= inv_norm;
        } else {
            self.state
                .par_chunks_mut(crate::backend::MIN_PAR_ELEMS)
                .for_each(simd::zero_slice);
            self.state[0] = Complex64::new(1.0, 0.0);
            self.pending_norm = 1.0;
        }
    }

    pub(super) fn apply_diagonal_batch(&mut self, entries: &[DiagEntry]) {
        let Some(built) = build_diagonal_batch_tables(entries) else {
            self.apply_diagonal_batch_fallback(entries);
            return;
        };
        if built.num_groups == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_bmi2() && simd::has_fma() {
            // SAFETY: has_bmi2() + has_fma() confirmed above
            unsafe {
                apply_diagonal_batch_bmi2(
                    &mut self.state,
                    &built.tables,
                    &built.group_pext_masks,
                    built.num_groups,
                );
            }
            return;
        }

        apply_diagonal_batch_scalar(
            &mut self.state,
            &built.tables,
            &built.unique_qubits,
            &built.group_sizes,
            built.num_groups,
        );
    }

    fn apply_diagonal_batch_fallback(&mut self, entries: &[DiagEntry]) {
        let one = Complex64::new(1.0, 0.0);
        for (i, amp) in self.state.iter_mut().enumerate() {
            let mut phase = one;
            for e in entries {
                match e {
                    DiagEntry::Phase1q { qubit, d0, d1 } => {
                        if (i >> qubit) & 1 == 1 {
                            phase *= d1;
                        } else {
                            phase *= d0;
                        }
                    }
                    DiagEntry::Phase2q { q0, q1, phase: p } => {
                        if ((i >> q0) & 1 == 1) && ((i >> q1) & 1 == 1) {
                            phase *= p;
                        }
                    }
                    DiagEntry::Parity2q { q0, q1, same, diff } => {
                        let parity = ((i >> q0) ^ (i >> q1)) & 1;
                        phase *= if parity == 0 { *same } else { *diff };
                    }
                }
            }
            *amp *= phase;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2,fma")]
unsafe fn apply_diagonal_batch_bmi2(
    state: &mut [Complex64],
    tables: &[[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    pext_masks: &[u64; MAX_DIAG_BATCH_GROUPS],
    num_groups: usize,
) {
    use std::arch::x86_64::_pext_u64;
    let one = Complex64::new(1.0, 0.0);
    for (i, amp) in state.iter_mut().enumerate() {
        let mut combined = one;
        for g in 0..num_groups {
            let bits = _pext_u64(i as u64, pext_masks[g]) as usize;
            combined *= tables[g][bits];
        }
        *amp *= combined;
    }
}

fn apply_diagonal_batch_scalar(
    state: &mut [Complex64],
    tables: &[[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    unique_qubits: &SmallVec<[usize; 32]>,
    group_sizes: &[usize; MAX_DIAG_BATCH_GROUPS],
    num_groups: usize,
) {
    let one = Complex64::new(1.0, 0.0);
    let mut group_shifts = [[0usize; DIAG_BATCH_MAX_QUBITS_PER_GROUP]; MAX_DIAG_BATCH_GROUPS];
    for (idx, &q) in unique_qubits.iter().enumerate() {
        group_shifts[idx / DIAG_BATCH_MAX_QUBITS_PER_GROUP]
            [idx % DIAG_BATCH_MAX_QUBITS_PER_GROUP] = q;
    }

    for (i, amp) in state.iter_mut().enumerate() {
        let mut combined = one;
        for g in 0..num_groups {
            let k = group_sizes[g];
            let mut bits = 0usize;
            for (p, &shift) in group_shifts[g][..k].iter().enumerate() {
                bits |= ((i >> shift) & 1) << p;
            }
            combined *= tables[g][bits];
        }
        *amp *= combined;
    }
}
