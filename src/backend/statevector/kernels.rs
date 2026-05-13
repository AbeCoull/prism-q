//! Gate application kernels for the statevector backend.
//!
//! Each kernel has a sequential implementation and, when the `parallel` feature
//! is enabled, a Rayon-parallelized variant dispatched above
//! `PARALLEL_THRESHOLD_QUBITS`.

use num_complex::Complex64;
use rand::Rng;
use smallvec::SmallVec;
use std::sync::{Arc, Mutex, OnceLock};

use super::insert_zero_bit;
use super::StatevectorBackend;
#[cfg(feature = "parallel")]
use super::{SendPtr, MIN_PAR_ELEMS, PARALLEL_THRESHOLD_QUBITS};
use crate::backend::simd;
use crate::backend::{is_phase_one, measurement_inv_norm, sorted_mcu_qubits};
use crate::circuit::{qft_textbook_steps, QftTextbookStep};
use crate::gates::{DiagEntry, Gate};
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

#[inline(always)]
fn swap_slices_kernel(a: &mut [Complex64], b: &mut [Complex64]) {
    debug_assert_eq!(a.len(), b.len());
    if a.len() < 4 {
        for (x, y) in a.iter_mut().zip(b.iter_mut()) {
            std::mem::swap(x, y);
        }
    } else {
        simd::swap_slices(a, b);
    }
}

#[inline(always)]
fn negate_slice_kernel(slice: &mut [Complex64]) {
    if slice.len() < 4 {
        for amp in slice {
            *amp = -*amp;
        }
    } else {
        simd::negate_slice(slice);
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

type QftTwiddleTable = Arc<[Complex64]>;

struct CachedQftTwiddles {
    table: QftTwiddleTable,
    last_used: u64,
}

struct QftTwiddleCache {
    entries: Vec<Option<CachedQftTwiddles>>,
    bytes: usize,
    tick: u64,
}

impl QftTwiddleCache {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            bytes: 0,
            tick: 0,
        }
    }

    #[inline]
    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }
}

const QFT_TWIDDLE_CACHE_DEFAULT_LIMIT_BYTES: usize = 256 * 1024 * 1024;

// Soft LRU cap for QFT twiddles. One oversized table may occupy the cache
// alone, which keeps large repeated QFT runs from rebuilding twiddles every
// sample. Set the environment value to 0 to disable caching.
#[inline]
fn qft_twiddle_cache_limit_bytes() -> usize {
    static LIMIT: OnceLock<usize> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("PRISM_QFT_TWIDDLE_CACHE_LIMIT_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb.saturating_mul(1024 * 1024))
            .unwrap_or(QFT_TWIDDLE_CACHE_DEFAULT_LIMIT_BYTES)
    })
}

#[inline(always)]
fn qft_twiddle_table_bytes(table_len: usize) -> usize {
    table_len.saturating_mul(std::mem::size_of::<Complex64>())
}

fn qft_twiddles_scaled(n: usize) -> QftTwiddleTable {
    static CACHE: OnceLock<Mutex<QftTwiddleCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(QftTwiddleCache::new()));

    {
        let mut guard = cache.lock().unwrap();
        let tick = guard.next_tick();
        if let Some(Some(entry)) = guard.entries.get_mut(n) {
            entry.last_used = tick;
            return Arc::clone(&entry.table);
        }
    }

    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let qft_size = 1usize << n;
    let mut twiddles = Vec::with_capacity(qft_size / 2);
    for k in 0..qft_size / 2 {
        let angle = std::f64::consts::TAU * k as f64 / qft_size as f64;
        twiddles.push(Complex64::from_polar(inv_sqrt2, angle));
    }
    let twiddles: Arc<[Complex64]> = twiddles.into();
    let table_bytes = qft_twiddle_table_bytes(twiddles.len());
    let cache_limit = qft_twiddle_cache_limit_bytes();
    if cache_limit == 0 {
        return twiddles;
    }

    let mut guard = cache.lock().unwrap();
    let tick = guard.next_tick();
    if guard.entries.len() <= n {
        guard.entries.resize_with(n + 1, || None);
    }
    if let Some(existing) = &mut guard.entries[n] {
        existing.last_used = tick;
        return Arc::clone(&existing.table);
    }

    while guard.bytes.saturating_add(table_bytes) > cache_limit {
        let Some((idx, _)) = guard
            .entries
            .iter()
            .enumerate()
            .filter_map(|(idx, entry)| entry.as_ref().map(|entry| (idx, entry.last_used)))
            .min_by_key(|&(_, last_used)| last_used)
        else {
            break;
        };
        if let Some(evicted) = guard.entries[idx].take() {
            guard.bytes = guard
                .bytes
                .saturating_sub(qft_twiddle_table_bytes(evicted.table.len()));
        }
    }

    guard.entries[n] = Some(CachedQftTwiddles {
        table: Arc::clone(&twiddles),
        last_used: tick,
    });
    guard.bytes = guard.bytes.saturating_add(table_bytes);
    twiddles
}

#[inline(always)]
fn reverse_low_bits(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    x.reverse_bits() >> (usize::BITS as usize - bits)
}

#[inline]
fn apply_bit_reverse_permutation(state: &mut [Complex64], bits: usize) {
    let len = state.len();
    debug_assert_eq!(len, 1usize << bits);
    if len <= 2 {
        return;
    }

    #[cfg(feature = "parallel")]
    if bits >= PARALLEL_THRESHOLD_QUBITS {
        let ptr = SendPtr(state.as_mut_ptr());
        (0..len)
            .into_par_iter()
            .with_min_len(MIN_PAR_ITERS)
            .for_each(move |i| {
                let j = reverse_low_bits(i, bits);
                if j > i {
                    // SAFETY: bit reversal is an involution. The `j > i`
                    // guard assigns each pair to exactly one iteration, and
                    // fixed points are skipped. The safe alternative is the
                    // serial `state.swap` fallback below; above the parallel
                    // threshold this O(N) final pass is part of the measured
                    // whole-state QFT hot path.
                    unsafe {
                        let a = ptr.load(i);
                        let b = ptr.load(j);
                        ptr.store(i, b);
                        ptr.store(j, a);
                    }
                }
            });
        return;
    }

    for i in 0..len {
        let j = reverse_low_bits(i, bits);
        if j > i {
            state.swap(i, j);
        }
    }
}

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

/// Run a single FFT stage in parallel.
///
/// Run one FFT stage in parallel.
///
/// Large group counts split by group. High-stride stages split inside each
/// group so all cores still get work.
#[cfg(feature = "parallel")]
#[inline]
fn fft_stage_par(
    state: &mut [Complex64],
    stride: usize,
    block_size: usize,
    twiddle_step: usize,
    twiddles_scaled: &[Complex64],
    inv_sqrt2: f64,
) {
    let total = state.len();
    let num_groups = total / block_size;

    let apply_pairs = |lo: &mut [Complex64], hi: &mut [Complex64], k_base: usize| {
        debug_assert_eq!(lo.len(), hi.len());
        for j in 0..lo.len() {
            let k = k_base + j;
            let w = twiddles_scaled[k * twiddle_step];
            let (out_lo, out_hi) = radix2_butterfly_values(lo[j], hi[j], w, inv_sqrt2);
            lo[j] = out_lo;
            hi[j] = out_hi;
        }
    };

    if num_groups >= 4 && block_size >= MIN_PAR_ELEMS {
        // Many large groups: split state by group.
        state.par_chunks_mut(block_size).for_each(|group| {
            let (lo, hi) = group.split_at_mut(stride);
            apply_pairs(lo, hi, 0);
        });
    } else if block_size < MIN_PAR_ELEMS {
        // Small groups: bundle work so each Rayon job has enough elements.
        let task_size = MIN_PAR_ELEMS;
        state.par_chunks_mut(task_size).for_each(|task_chunk| {
            let mut g = 0;
            while g + block_size <= task_chunk.len() {
                let group = &mut task_chunk[g..g + block_size];
                let (lo, hi) = group.split_at_mut(stride);
                apply_pairs(lo, hi, 0);
                g += block_size;
            }
        });
    } else {
        // Few groups: split each large group into zipped chunks.
        const MIN_PAR_PAIRS: usize = MIN_PAR_ELEMS / 2;
        for group_start in (0..total).step_by(block_size) {
            let group = &mut state[group_start..group_start + block_size];
            let (lo, hi) = group.split_at_mut(stride);
            let sub = MIN_PAR_PAIRS.min(stride).max(1);
            lo.par_chunks_mut(sub)
                .zip(hi.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(chunk_idx, (lo_c, hi_c))| {
                    let k_base = chunk_idx * sub;
                    apply_pairs(lo_c, hi_c, k_base);
                });
        }
    }
}

/// Scalar radix-2 DIF butterfly. `w` includes the `1/sqrt(2)` factor.
#[inline(always)]
fn radix2_butterfly_values(
    a: Complex64,
    b: Complex64,
    w: Complex64,
    inv_sqrt2: f64,
) -> (Complex64, Complex64) {
    let lo = Complex64::new((a.re + b.re) * inv_sqrt2, (a.im + b.im) * inv_sqrt2);
    let hi = (a - b) * w;
    (lo, hi)
}

/// Apply one full radix-2 DIF stage in place to `state` at the given stride.
/// `twiddle_step = total_state_len / (2 * stride)` selects the right
/// subsample of the twiddle table.
#[inline]
fn run_radix2_stage_seq(
    state: &mut [Complex64],
    stride: usize,
    twiddles_scaled: &[Complex64],
    twiddle_step: usize,
    inv_sqrt2: f64,
) {
    let block_size = stride << 1;
    let total = state.len();
    let mut group_start = 0;
    while group_start < total {
        for k in 0..stride {
            let i0 = group_start + k;
            let i1 = i0 + stride;
            let w = twiddles_scaled[k * twiddle_step];
            let (lo, hi) = radix2_butterfly_values(state[i0], state[i1], w, inv_sqrt2);
            state[i0] = lo;
            state[i1] = hi;
        }
        group_start += block_size;
    }
}

/// Scalar radix-4 butterfly math. Twiddles include the `1/sqrt(2)` factor.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn radix4_butterfly_values(
    a: Complex64,
    b: Complex64,
    c: Complex64,
    d: Complex64,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
    inv_sqrt2: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let t_ac_sum = Complex64::new((a.re + c.re) * inv_sqrt2, (a.im + c.im) * inv_sqrt2);
    let t_ac_diff = (a - c) * w_2s_k;
    let t_bd_sum = Complex64::new((b.re + d.re) * inv_sqrt2, (b.im + d.im) * inv_sqrt2);
    let t_bd_diff = (b - d) * w_2s_kps;

    let oa = Complex64::new(
        (t_ac_sum.re + t_bd_sum.re) * inv_sqrt2,
        (t_ac_sum.im + t_bd_sum.im) * inv_sqrt2,
    );
    let ob = (t_ac_sum - t_bd_sum) * w_s_k;
    let oc = Complex64::new(
        (t_ac_diff.re + t_bd_diff.re) * inv_sqrt2,
        (t_ac_diff.im + t_bd_diff.im) * inv_sqrt2,
    );
    let od = (t_ac_diff - t_bd_diff) * w_s_k;
    (oa, ob, oc, od)
}

/// Apply one fused radix-4 butterfly at state index `i`.
#[inline(always)]
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn radix4_butterfly_scalar(
    state: &mut [Complex64],
    i: usize,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
    inv_sqrt2: f64,
) {
    let s = inner_stride;
    let (oa, ob, oc, od) = radix4_butterfly_values(
        state[i],
        state[i + s],
        state[i + 2 * s],
        state[i + 3 * s],
        w_2s_k,
        w_2s_kps,
        w_s_k,
        inv_sqrt2,
    );
    state[i] = oa;
    state[i + s] = ob;
    state[i + 2 * s] = oc;
    state[i + 3 * s] = od;
}

/// Apply one radix-4 butterfly across four disjoint quarter slices.
#[inline(always)]
#[cfg_attr(
    any(target_arch = "aarch64", not(feature = "parallel")),
    allow(dead_code)
)]
#[allow(clippy::too_many_arguments)]
fn radix4_butterfly_quartet_scalar(
    qa: &mut [Complex64],
    qb: &mut [Complex64],
    qc: &mut [Complex64],
    qd: &mut [Complex64],
    j: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
    inv_sqrt2: f64,
) {
    let (oa, ob, oc, od) = radix4_butterfly_values(
        qa[j], qb[j], qc[j], qd[j], w_2s_k, w_2s_kps, w_s_k, inv_sqrt2,
    );
    qa[j] = oa;
    qb[j] = ob;
    qc[j] = oc;
    qd[j] = od;
}

/// FMA variant for one radix-4 butterfly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn radix4_butterfly_fma(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    use std::arch::x86_64::{
        _mm_add_pd, _mm_loadu_pd, _mm_mul_pd, _mm_set1_pd, _mm_storeu_pd, _mm_sub_pd,
    };
    let s = inner_stride;
    let inv_sqrt2_v = _mm_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

    let a_ptr = base_ptr;
    let b_ptr = base_ptr.add(s * 2);
    let c_ptr = base_ptr.add(2 * s * 2);
    let d_ptr = base_ptr.add(3 * s * 2);

    let a = _mm_loadu_pd(a_ptr);
    let b = _mm_loadu_pd(b_ptr);
    let c = _mm_loadu_pd(c_ptr);
    let d = _mm_loadu_pd(d_ptr);

    let w_2s_k_rr = _mm_set1_pd(w_2s_k.re);
    let w_2s_k_ii = _mm_set1_pd(w_2s_k.im);
    let w_2s_kps_rr = _mm_set1_pd(w_2s_kps.re);
    let w_2s_kps_ii = _mm_set1_pd(w_2s_kps.im);
    let w_s_k_rr = _mm_set1_pd(w_s_k.re);
    let w_s_k_ii = _mm_set1_pd(w_s_k.im);

    let t_ac_sum = _mm_mul_pd(_mm_add_pd(a, c), inv_sqrt2_v);
    let t_ac_diff = simd::complex_mul_fma(w_2s_k_rr, w_2s_k_ii, _mm_sub_pd(a, c));
    let t_bd_sum = _mm_mul_pd(_mm_add_pd(b, d), inv_sqrt2_v);
    let t_bd_diff = simd::complex_mul_fma(w_2s_kps_rr, w_2s_kps_ii, _mm_sub_pd(b, d));

    _mm_storeu_pd(
        a_ptr,
        _mm_mul_pd(_mm_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
    );
    _mm_storeu_pd(
        b_ptr,
        simd::complex_mul_fma(w_s_k_rr, w_s_k_ii, _mm_sub_pd(t_ac_sum, t_bd_sum)),
    );
    _mm_storeu_pd(
        c_ptr,
        _mm_mul_pd(_mm_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
    );
    _mm_storeu_pd(
        d_ptr,
        simd::complex_mul_fma(w_s_k_rr, w_s_k_ii, _mm_sub_pd(t_ac_diff, t_bd_diff)),
    );
}

/// AVX2-256 SIMD variant: processes TWO radix-4 butterflies per call (at
/// consecutive `k` and `k+1`, four complex amplitudes each, eight elements
/// total). Each load/store is one 256-bit op covering two adjacent complex
/// values (which are 32 bytes contiguous in memory). Twiddles are still
/// loaded scalar-by-scalar since the stage's `twiddle_step` is typically
/// > 1 for high-stride pairs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_pair_avx2fma(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k0: Complex64,
    w_2s_k1: Complex64,
    w_2s_kps0: Complex64,
    w_2s_kps1: Complex64,
    w_s_k0: Complex64,
    w_s_k1: Complex64,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_set_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };
    let s = inner_stride;
    let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

    let a_ptr = base_ptr;
    let b_ptr = base_ptr.add(s * 2);
    let c_ptr = base_ptr.add(2 * s * 2);
    let d_ptr = base_ptr.add(3 * s * 2);

    let a = _mm256_loadu_pd(a_ptr);
    let b = _mm256_loadu_pd(b_ptr);
    let c = _mm256_loadu_pd(c_ptr);
    let d = _mm256_loadu_pd(d_ptr);

    // c_rr layout: [w0.re, w0.re, w1.re, w1.re], broadcasts each twiddle's
    // real part to the two lanes corresponding to that butterfly's complex.
    let w_2s_k_rr = _mm256_set_pd(w_2s_k1.re, w_2s_k1.re, w_2s_k0.re, w_2s_k0.re);
    let w_2s_k_ii = _mm256_set_pd(w_2s_k1.im, w_2s_k1.im, w_2s_k0.im, w_2s_k0.im);
    let w_2s_kps_rr = _mm256_set_pd(w_2s_kps1.re, w_2s_kps1.re, w_2s_kps0.re, w_2s_kps0.re);
    let w_2s_kps_ii = _mm256_set_pd(w_2s_kps1.im, w_2s_kps1.im, w_2s_kps0.im, w_2s_kps0.im);
    let w_s_k_rr = _mm256_set_pd(w_s_k1.re, w_s_k1.re, w_s_k0.re, w_s_k0.re);
    let w_s_k_ii = _mm256_set_pd(w_s_k1.im, w_s_k1.im, w_s_k0.im, w_s_k0.im);

    let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
    let t_ac_diff = simd::complex_mul_avx2fma(w_2s_k_rr, w_2s_k_ii, _mm256_sub_pd(a, c));
    let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
    let t_bd_diff = simd::complex_mul_avx2fma(w_2s_kps_rr, w_2s_kps_ii, _mm256_sub_pd(b, d));

    _mm256_storeu_pd(
        a_ptr,
        _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
    );
    _mm256_storeu_pd(
        b_ptr,
        simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
    );
    _mm256_storeu_pd(
        c_ptr,
        _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
    );
    _mm256_storeu_pd(
        d_ptr,
        simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
    );
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_pair_slices_avx2fma(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k0: Complex64,
    w_2s_k1: Complex64,
    w_2s_kps0: Complex64,
    w_2s_kps1: Complex64,
    w_s_k0: Complex64,
    w_s_k1: Complex64,
) {
    use std::arch::x86_64::{
        _mm256_add_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_set_pd,
        _mm256_storeu_pd, _mm256_sub_pd,
    };
    let inv_sqrt2_v = _mm256_set1_pd(std::f64::consts::FRAC_1_SQRT_2);

    let a = _mm256_loadu_pd(a_ptr);
    let b = _mm256_loadu_pd(b_ptr);
    let c = _mm256_loadu_pd(c_ptr);
    let d = _mm256_loadu_pd(d_ptr);

    let w_2s_k_rr = _mm256_set_pd(w_2s_k1.re, w_2s_k1.re, w_2s_k0.re, w_2s_k0.re);
    let w_2s_k_ii = _mm256_set_pd(w_2s_k1.im, w_2s_k1.im, w_2s_k0.im, w_2s_k0.im);
    let w_2s_kps_rr = _mm256_set_pd(w_2s_kps1.re, w_2s_kps1.re, w_2s_kps0.re, w_2s_kps0.re);
    let w_2s_kps_ii = _mm256_set_pd(w_2s_kps1.im, w_2s_kps1.im, w_2s_kps0.im, w_2s_kps0.im);
    let w_s_k_rr = _mm256_set_pd(w_s_k1.re, w_s_k1.re, w_s_k0.re, w_s_k0.re);
    let w_s_k_ii = _mm256_set_pd(w_s_k1.im, w_s_k1.im, w_s_k0.im, w_s_k0.im);

    let t_ac_sum = _mm256_mul_pd(_mm256_add_pd(a, c), inv_sqrt2_v);
    let t_ac_diff = simd::complex_mul_avx2fma(w_2s_k_rr, w_2s_k_ii, _mm256_sub_pd(a, c));
    let t_bd_sum = _mm256_mul_pd(_mm256_add_pd(b, d), inv_sqrt2_v);
    let t_bd_diff = simd::complex_mul_avx2fma(w_2s_kps_rr, w_2s_kps_ii, _mm256_sub_pd(b, d));

    _mm256_storeu_pd(
        a_ptr,
        _mm256_mul_pd(_mm256_add_pd(t_ac_sum, t_bd_sum), inv_sqrt2_v),
    );
    _mm256_storeu_pd(
        b_ptr,
        simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_sum, t_bd_sum)),
    );
    _mm256_storeu_pd(
        c_ptr,
        _mm256_mul_pd(_mm256_add_pd(t_ac_diff, t_bd_diff), inv_sqrt2_v),
    );
    _mm256_storeu_pd(
        d_ptr,
        simd::complex_mul_avx2fma(w_s_k_rr, w_s_k_ii, _mm256_sub_pd(t_ac_diff, t_bd_diff)),
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_twiddle_parts(
    w: Complex64,
) -> (
    std::arch::aarch64::float64x2_t,
    std::arch::aarch64::float64x2_t,
) {
    use std::arch::aarch64::{vcombine_f64, vdup_n_f64, vdupq_n_f64};
    (
        vdupq_n_f64(w.re),
        vcombine_f64(vdup_n_f64(-w.im), vdup_n_f64(w.im)),
    )
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_neon(
    base_ptr: *mut f64,
    inner_stride: usize,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};
    let s = inner_stride;
    let inv_sqrt2_v = vdupq_n_f64(std::f64::consts::FRAC_1_SQRT_2);

    let a_ptr = base_ptr;
    let b_ptr = base_ptr.add(s * 2);
    let c_ptr = base_ptr.add(2 * s * 2);
    let d_ptr = base_ptr.add(3 * s * 2);

    let a = vld1q_f64(a_ptr);
    let b = vld1q_f64(b_ptr);
    let c = vld1q_f64(c_ptr);
    let d = vld1q_f64(d_ptr);

    let (w_2s_k_rr, w_2s_k_ii_as) = neon_twiddle_parts(w_2s_k);
    let (w_2s_kps_rr, w_2s_kps_ii_as) = neon_twiddle_parts(w_2s_kps);
    let (w_s_k_rr, w_s_k_ii_as) = neon_twiddle_parts(w_s_k);

    let t_ac_sum = vmulq_f64(vaddq_f64(a, c), inv_sqrt2_v);
    let t_ac_diff = simd::complex_mul_neon(w_2s_k_rr, w_2s_k_ii_as, vsubq_f64(a, c));
    let t_bd_sum = vmulq_f64(vaddq_f64(b, d), inv_sqrt2_v);
    let t_bd_diff = simd::complex_mul_neon(w_2s_kps_rr, w_2s_kps_ii_as, vsubq_f64(b, d));

    vst1q_f64(a_ptr, vmulq_f64(vaddq_f64(t_ac_sum, t_bd_sum), inv_sqrt2_v));
    vst1q_f64(
        b_ptr,
        simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_sum, t_bd_sum)),
    );
    vst1q_f64(
        c_ptr,
        vmulq_f64(vaddq_f64(t_ac_diff, t_bd_diff), inv_sqrt2_v),
    );
    vst1q_f64(
        d_ptr,
        simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_diff, t_bd_diff)),
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn radix4_butterfly_slices_neon(
    a_ptr: *mut f64,
    b_ptr: *mut f64,
    c_ptr: *mut f64,
    d_ptr: *mut f64,
    w_2s_k: Complex64,
    w_2s_kps: Complex64,
    w_s_k: Complex64,
) {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64, vsubq_f64};
    let inv_sqrt2_v = vdupq_n_f64(std::f64::consts::FRAC_1_SQRT_2);

    let a = vld1q_f64(a_ptr);
    let b = vld1q_f64(b_ptr);
    let c = vld1q_f64(c_ptr);
    let d = vld1q_f64(d_ptr);

    let (w_2s_k_rr, w_2s_k_ii_as) = neon_twiddle_parts(w_2s_k);
    let (w_2s_kps_rr, w_2s_kps_ii_as) = neon_twiddle_parts(w_2s_kps);
    let (w_s_k_rr, w_s_k_ii_as) = neon_twiddle_parts(w_s_k);

    let t_ac_sum = vmulq_f64(vaddq_f64(a, c), inv_sqrt2_v);
    let t_ac_diff = simd::complex_mul_neon(w_2s_k_rr, w_2s_k_ii_as, vsubq_f64(a, c));
    let t_bd_sum = vmulq_f64(vaddq_f64(b, d), inv_sqrt2_v);
    let t_bd_diff = simd::complex_mul_neon(w_2s_kps_rr, w_2s_kps_ii_as, vsubq_f64(b, d));

    vst1q_f64(a_ptr, vmulq_f64(vaddq_f64(t_ac_sum, t_bd_sum), inv_sqrt2_v));
    vst1q_f64(
        b_ptr,
        simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_sum, t_bd_sum)),
    );
    vst1q_f64(
        c_ptr,
        vmulq_f64(vaddq_f64(t_ac_diff, t_bd_diff), inv_sqrt2_v),
    );
    vst1q_f64(
        d_ptr,
        simd::complex_mul_neon(w_s_k_rr, w_s_k_ii_as, vsubq_f64(t_ac_diff, t_bd_diff)),
    );
}

#[inline]
fn apply_radix4_groups(
    group: &mut [Complex64],
    s: usize,
    k_base: usize,
    step_outer: usize,
    step_inner: usize,
    twiddles_scaled: &[Complex64],
    inv_sqrt2: f64,
) {
    debug_assert_eq!(group.len() % (s * 4), 0);
    let groups = group.len() / (s * 4);
    #[cfg(target_arch = "aarch64")]
    let _ = inv_sqrt2;

    #[cfg(target_arch = "x86_64")]
    {
        if simd::has_avx2_fma() && s >= 2 {
            // SAFETY: has_avx2_fma() is runtime checked. Adjacent pairs stay
            // inside the same radix-4 group because s >= 2. The safe
            // alternative is the scalar radix-4 loop below; this SIMD path is
            // kept because the measured QFT speedup depends on avoiding scalar
            // Complex64 arithmetic in the fused stage-pair hot loop.
            unsafe {
                let base_ptr = group.as_mut_ptr() as *mut f64;
                for g in 0..groups {
                    let group_off = g * s * 4;
                    let mut k_offset = 0usize;
                    while k_offset + 2 <= s {
                        let k = k_base + k_offset;
                        let w_2s_k0 = twiddles_scaled[k * step_outer];
                        let w_2s_k1 = twiddles_scaled[(k + 1) * step_outer];
                        let w_2s_kps0 = twiddles_scaled[(k + s) * step_outer];
                        let w_2s_kps1 = twiddles_scaled[(k + s + 1) * step_outer];
                        let w_s_k0 = twiddles_scaled[k * step_inner];
                        let w_s_k1 = twiddles_scaled[(k + 1) * step_inner];
                        let pair_base = base_ptr.add((group_off + k_offset) * 2);
                        radix4_butterfly_pair_avx2fma(
                            pair_base, s, w_2s_k0, w_2s_k1, w_2s_kps0, w_2s_kps1, w_s_k0, w_s_k1,
                        );
                        k_offset += 2;
                    }
                    if k_offset < s {
                        let k = k_base + k_offset;
                        let w_2s_k = twiddles_scaled[k * step_outer];
                        let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                        let w_s_k = twiddles_scaled[k * step_inner];
                        let pair_base = base_ptr.add((group_off + k_offset) * 2);
                        radix4_butterfly_fma(pair_base, s, w_2s_k, w_2s_kps, w_s_k);
                    }
                }
            }
            return;
        }
        if simd::has_fma() {
            // SAFETY: has_fma() is runtime checked. The safe alternative is
            // the scalar radix-4 loop below; this SIMD path is kept because the
            // measured QFT speedup depends on avoiding scalar Complex64
            // arithmetic in the fused stage-pair hot loop.
            unsafe {
                let base_ptr = group.as_mut_ptr() as *mut f64;
                for g in 0..groups {
                    let group_off = g * s * 4;
                    for k_offset in 0..s {
                        let k = k_base + k_offset;
                        let w_2s_k = twiddles_scaled[k * step_outer];
                        let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                        let w_s_k = twiddles_scaled[k * step_inner];
                        let pair_base = base_ptr.add((group_off + k_offset) * 2);
                        radix4_butterfly_fma(pair_base, s, w_2s_k, w_2s_kps, w_s_k);
                    }
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 targets provide NEON. The loop maps each radix-4
        // butterfly to four in-bounds amplitudes inside `group`; the safe
        // alternative is the scalar fallback below. This SIMD counterpart is
        // required so Apple Silicon does not pay scalar Complex64 overhead in
        // the same measured QFT hot loop as x86.
        unsafe {
            let base_ptr = group.as_mut_ptr() as *mut f64;
            for g in 0..groups {
                let group_off = g * s * 4;
                for k_offset in 0..s {
                    let k = k_base + k_offset;
                    let w_2s_k = twiddles_scaled[k * step_outer];
                    let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                    let w_s_k = twiddles_scaled[k * step_inner];
                    let pair_base = base_ptr.add((group_off + k_offset) * 2);
                    radix4_butterfly_neon(pair_base, s, w_2s_k, w_2s_kps, w_s_k);
                }
            }
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        for g in 0..groups {
            let base = g * s * 4;
            for k_offset in 0..s {
                let k = k_base + k_offset;
                let w_2s_k = twiddles_scaled[k * step_outer];
                let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                let w_s_k = twiddles_scaled[k * step_inner];
                radix4_butterfly_scalar(
                    group,
                    base + k_offset,
                    s,
                    w_2s_k,
                    w_2s_kps,
                    w_s_k,
                    inv_sqrt2,
                );
            }
        }
    }
}

#[inline]
fn fft_stage_pair_in_slice(
    slice: &mut [Complex64],
    inner_stride: usize,
    total: usize,
    twiddles_scaled: &[Complex64],
    inv_sqrt2: f64,
) {
    let s = inner_stride;
    let block_size = s << 2;
    let step_outer = total / block_size;
    let step_inner = total / (s << 1);
    apply_radix4_groups(
        slice,
        s,
        0,
        step_outer,
        step_inner,
        twiddles_scaled,
        inv_sqrt2,
    );
}

/// Run two DIF FFT stages as one radix-4 pass.
///
/// `inner_stride = 2^(outer_stage - 1)`. Falls back to a sequential loop when
/// the `parallel` feature is off.
#[inline]
fn fft_stage_pair_par(
    state: &mut [Complex64],
    inner_stride: usize,
    twiddles_scaled: &[Complex64],
    inv_sqrt2: f64,
) {
    let s = inner_stride;
    let block_size = s << 2;
    let total = state.len();
    let step_outer = total / block_size;
    let step_inner = total / (s << 1);

    let apply_group = |group: &mut [Complex64], k_base: usize| {
        apply_radix4_groups(
            group,
            s,
            k_base,
            step_outer,
            step_inner,
            twiddles_scaled,
            inv_sqrt2,
        );
    };

    #[cfg(feature = "parallel")]
    {
        let num_groups = total / block_size;
        if num_groups >= 4 && block_size >= MIN_PAR_ELEMS {
            // Case A: many independent groups, split state by block.
            state.par_chunks_mut(block_size).for_each(|chunk| {
                apply_group(chunk, 0);
            });
            return;
        }
        if block_size < MIN_PAR_ELEMS {
            // Case B: small groups, bundle work per Rayon task.
            let task_size = MIN_PAR_ELEMS;
            state.par_chunks_mut(task_size).for_each(|task_chunk| {
                apply_group(task_chunk, 0);
            });
            return;
        }
        // Case C: 1-3 large groups, split inside each group.
        const MIN_PAR_PAIRS: usize = MIN_PAR_ELEMS / 4;
        for group_start in (0..total).step_by(block_size) {
            let group = &mut state[group_start..group_start + block_size];
            let (q0, rest) = group.split_at_mut(s);
            let (q1, rest) = rest.split_at_mut(s);
            let (q2, q3) = rest.split_at_mut(s);
            let sub = MIN_PAR_PAIRS.min(s).max(1);
            q0.par_chunks_mut(sub)
                .zip(q1.par_chunks_mut(sub))
                .zip(q2.par_chunks_mut(sub))
                .zip(q3.par_chunks_mut(sub))
                .enumerate()
                .for_each(|(chunk_idx, (((qa, qb), qc), qd))| {
                    let k_base = chunk_idx * sub;
                    let n_local = qa.len();
                    #[cfg(target_arch = "x86_64")]
                    if simd::has_avx2_fma() && n_local >= 2 {
                        // SAFETY: the four slices are disjoint quarters of
                        // one radix-4 group. Each AVX2 call handles adjacent
                        // complex values inside the same local chunk. The
                        // safe alternative is the scalar quartet loop below;
                        // this SIMD path is kept because the measured QFT
                        // speedup depends on avoiding scalar Complex64
                        // arithmetic in the fused stage-pair hot loop.
                        unsafe {
                            let qa_ptr = qa.as_mut_ptr() as *mut f64;
                            let qb_ptr = qb.as_mut_ptr() as *mut f64;
                            let qc_ptr = qc.as_mut_ptr() as *mut f64;
                            let qd_ptr = qd.as_mut_ptr() as *mut f64;
                            let mut j = 0usize;
                            while j + 2 <= n_local {
                                let k = k_base + j;
                                let w_2s_k0 = twiddles_scaled[k * step_outer];
                                let w_2s_k1 = twiddles_scaled[(k + 1) * step_outer];
                                let w_2s_kps0 = twiddles_scaled[(k + s) * step_outer];
                                let w_2s_kps1 = twiddles_scaled[(k + s + 1) * step_outer];
                                let w_s_k0 = twiddles_scaled[k * step_inner];
                                let w_s_k1 = twiddles_scaled[(k + 1) * step_inner];
                                radix4_butterfly_pair_slices_avx2fma(
                                    qa_ptr.add(j * 2),
                                    qb_ptr.add(j * 2),
                                    qc_ptr.add(j * 2),
                                    qd_ptr.add(j * 2),
                                    w_2s_k0,
                                    w_2s_k1,
                                    w_2s_kps0,
                                    w_2s_kps1,
                                    w_s_k0,
                                    w_s_k1,
                                );
                                j += 2;
                            }
                            if j == n_local {
                                return;
                            }
                            let k = k_base + j;
                            let w_2s_k = twiddles_scaled[k * step_outer];
                            let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                            let w_s_k = twiddles_scaled[k * step_inner];
                            radix4_butterfly_quartet_scalar(
                                qa, qb, qc, qd, j, w_2s_k, w_2s_kps, w_s_k, inv_sqrt2,
                            );
                            return;
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        // SAFETY: the zipped Rayon chunks own disjoint
                        // quarter slices. Each iteration touches the same
                        // local offset in all four quarters, matching the
                        // scalar fallback below without aliasing. This SIMD
                        // counterpart is required so Apple Silicon does not
                        // pay scalar Complex64 overhead in the same measured
                        // QFT hot loop as x86.
                        unsafe {
                            let qa_ptr = qa.as_mut_ptr() as *mut f64;
                            let qb_ptr = qb.as_mut_ptr() as *mut f64;
                            let qc_ptr = qc.as_mut_ptr() as *mut f64;
                            let qd_ptr = qd.as_mut_ptr() as *mut f64;
                            for j in 0..n_local {
                                let k = k_base + j;
                                let w_2s_k = twiddles_scaled[k * step_outer];
                                let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                                let w_s_k = twiddles_scaled[k * step_inner];
                                radix4_butterfly_slices_neon(
                                    qa_ptr.add(j * 2),
                                    qb_ptr.add(j * 2),
                                    qc_ptr.add(j * 2),
                                    qd_ptr.add(j * 2),
                                    w_2s_k,
                                    w_2s_kps,
                                    w_s_k,
                                );
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for j in 0..n_local {
                            let k = k_base + j;
                            let w_2s_k = twiddles_scaled[k * step_outer];
                            let w_2s_kps = twiddles_scaled[(k + s) * step_outer];
                            let w_s_k = twiddles_scaled[k * step_inner];
                            radix4_butterfly_quartet_scalar(
                                qa, qb, qc, qd, j, w_2s_k, w_2s_kps, w_s_k, inv_sqrt2,
                            );
                        }
                    }
                });
        }
    }

    #[cfg(not(feature = "parallel"))]
    apply_group(state, 0);
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

        if control > target {
            let ctrl_half = 1usize << control;
            let block_size = ctrl_half << 1;
            let tgt_half = 1usize << target;
            let tgt_block = tgt_half << 1;

            for chunk in self.state.chunks_mut(block_size) {
                let (_, hi) = chunk.split_at_mut(ctrl_half);
                for sub in hi.chunks_mut(tgt_block) {
                    let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                    swap_slices_kernel(sub_lo, sub_hi);
                }
            }
        } else {
            let ctrl_half = 1usize << control;
            let ctrl_block = ctrl_half << 1;
            let tgt_half = 1usize << target;
            let block_size = tgt_half << 1;

            for chunk in self.state.chunks_mut(block_size) {
                let (lo, hi) = chunk.split_at_mut(tgt_half);
                for (lo_sub, hi_sub) in lo.chunks_mut(ctrl_block).zip(hi.chunks_mut(ctrl_block)) {
                    let (_, lo_active) = lo_sub.split_at_mut(ctrl_half);
                    let (_, hi_active) = hi_sub.split_at_mut(ctrl_half);
                    swap_slices_kernel(lo_active, hi_active);
                }
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
                            swap_slices_kernel(sub_lo, sub_hi);
                        }
                    });
            } else {
                let inner_tile = MIN_PAR_ELEMS.max(tgt_block);
                for chunk in self.state.chunks_mut(block_size) {
                    let (_, hi) = chunk.split_at_mut(ctrl_half);
                    hi.par_chunks_mut(inner_tile).for_each(|tile| {
                        for sub in tile.chunks_mut(tgt_block) {
                            let (sub_lo, sub_hi) = sub.split_at_mut(tgt_half);
                            swap_slices_kernel(sub_lo, sub_hi);
                        }
                    });
                }
            }
        } else {
            let ctrl_half = 1usize << control;
            let ctrl_block = ctrl_half << 1;
            let tgt_half = 1usize << target;
            let block_size = tgt_half << 1;
            let num_blocks = self.state.len() / block_size;

            if num_blocks >= 4 {
                self.state
                    .par_chunks_mut(block_size)
                    .with_min_len(chunk_min_len(block_size))
                    .for_each(|chunk| {
                        let (lo, hi) = chunk.split_at_mut(tgt_half);
                        for (lo_sub, hi_sub) in
                            lo.chunks_mut(ctrl_block).zip(hi.chunks_mut(ctrl_block))
                        {
                            let (_, lo_active) = lo_sub.split_at_mut(ctrl_half);
                            let (_, hi_active) = hi_sub.split_at_mut(ctrl_half);
                            swap_slices_kernel(lo_active, hi_active);
                        }
                    });
            } else {
                let inner_tile = MIN_PAR_ELEMS.max(ctrl_block);
                for chunk in self.state.chunks_mut(block_size) {
                    let (lo, hi) = chunk.split_at_mut(tgt_half);
                    lo.par_chunks_mut(inner_tile)
                        .zip(hi.par_chunks_mut(inner_tile))
                        .for_each(|(lo_tile, hi_tile)| {
                            for (lo_sub, hi_sub) in lo_tile
                                .chunks_mut(ctrl_block)
                                .zip(hi_tile.chunks_mut(ctrl_block))
                            {
                                let (_, lo_active) = lo_sub.split_at_mut(ctrl_half);
                                let (_, hi_active) = hi_sub.split_at_mut(ctrl_half);
                                swap_slices_kernel(lo_active, hi_active);
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

        let (lo_q, hi_q) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
        let lo_half = 1usize << lo_q;
        let lo_block = lo_half << 1;
        let hi_half = 1usize << hi_q;
        let block_size = hi_half << 1;

        for chunk in self.state.chunks_mut(block_size) {
            let (_, hi_group) = chunk.split_at_mut(hi_half);
            for sub in hi_group.chunks_mut(lo_block) {
                let (_, sub_hi) = sub.split_at_mut(lo_half);
                negate_slice_kernel(sub_hi);
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
                        negate_slice_kernel(sub_hi);
                    }
                });
        } else {
            let inner_tile = MIN_PAR_ELEMS.max(lo_block);
            for chunk in self.state.chunks_mut(block_size) {
                let (_, hi_group) = chunk.split_at_mut(hi_half);
                hi_group.par_chunks_mut(inner_tile).for_each(|tile| {
                    for sub in tile.chunks_mut(lo_block) {
                        let (_, sub_hi) = sub.split_at_mut(lo_half);
                        negate_slice_kernel(sub_hi);
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

            // SAFETY: insert_zero_bit bijection gives disjoint index pairs per iteration.
            (0..num_iters)
                .into_par_iter()
                .with_min_len(MIN_PAR_ITERS)
                .for_each(move |i| {
                    let base = insert_zero_bit(insert_zero_bit(i, lo), hi);
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

        // SAFETY: insert_zero_bit bijection gives disjoint index groups per iteration.
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

    /// Apply a whole-state QFT with a tiled DIF FFT.
    ///
    /// The DIF output is bit-reversed; the final pass restores textbook order.
    pub(super) fn apply_qft_block(&mut self, start: usize, num: usize) {
        assert!(start + num <= self.num_qubits);
        assert!(num >= 1);

        assert_eq!(
            start, 0,
            "QftBlock with non-zero start is not supported by the FFT kernel; \
             sim::expand_qft_blocks should pre-expand before reaching this point"
        );

        let n = num;
        let qft_size = 1usize << n;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        let twiddles_scaled = qft_twiddles_scaled(n);
        let twiddles_scaled = twiddles_scaled.as_ref();

        let total = self.state.len();
        assert_eq!(
            total, qft_size,
            "QftBlock currently requires whole-state QFT"
        );

        // Cache-tiled DIF FFT:
        //   - High-stride stages run as full-state passes.
        //   - Low-stride stages run together inside each L2-sized tile.
        const TILE_BITS: usize = 13;

        let tile_bits = TILE_BITS.min(n);

        // Phase 1: high-stride stages.
        //
        // Fuse adjacent high-stride stages as radix-4 pairs to cut memory
        // traffic. If the count is odd, run one radix-2 stage first.
        let high_stride_count = n - tile_bits;
        let mut stage_top = n;
        let mut stages_left = high_stride_count;

        let run_radix2_stage = |state: &mut Vec<Complex64>, stage: usize| {
            let stride = 1usize << stage;
            let block_size = stride << 1;
            let twiddle_step = total / block_size;

            #[cfg(feature = "parallel")]
            if state.len() >= 1usize << PARALLEL_THRESHOLD_QUBITS {
                fft_stage_par(
                    state,
                    stride,
                    block_size,
                    twiddle_step,
                    twiddles_scaled,
                    inv_sqrt2,
                );
                return;
            }

            run_radix2_stage_seq(state, stride, twiddles_scaled, twiddle_step, inv_sqrt2);
        };

        if stages_left % 2 == 1 {
            stage_top -= 1;
            run_radix2_stage(&mut self.state, stage_top);
            stages_left -= 1;
        }

        while stages_left > 0 {
            // Fuse pair (stage_top - 1, stage_top - 2): outer stride
            // = 2^(stage_top - 1), inner stride = 2^(stage_top - 2).
            let inner_stride = 1usize << (stage_top - 2);
            fft_stage_pair_par(&mut self.state, inner_stride, twiddles_scaled, inv_sqrt2);
            stage_top -= 2;
            stages_left -= 2;
        }

        // Phase 2: low-stride stages on cache-resident tiles.
        if tile_bits == 0 {
            return;
        }
        let tile_size = 1usize << tile_bits;

        let apply_low_stages_in_tile = |tile: &mut [Complex64], twiddles_scaled: &[Complex64]| {
            let mut stage_top = tile_bits;
            let mut stages_left = tile_bits;
            if stages_left % 2 == 1 {
                stage_top -= 1;
                let stride = 1usize << stage_top;
                let block_size = stride << 1;
                let twiddle_step = total / block_size;
                run_radix2_stage_seq(tile, stride, twiddles_scaled, twiddle_step, inv_sqrt2);
                stages_left -= 1;
            }
            while stages_left > 0 {
                let inner_stride = 1usize << (stage_top - 2);
                fft_stage_pair_in_slice(tile, inner_stride, total, twiddles_scaled, inv_sqrt2);
                stage_top -= 2;
                stages_left -= 2;
            }
        };

        #[cfg(feature = "parallel")]
        let low_done_parallel =
            if self.num_qubits >= PARALLEL_THRESHOLD_QUBITS && total / tile_size >= 4 {
                self.state.par_chunks_mut(tile_size).for_each(|tile| {
                    apply_low_stages_in_tile(tile, twiddles_scaled);
                });
                true
            } else {
                false
            };
        #[cfg(not(feature = "parallel"))]
        let low_done_parallel = false;

        if !low_done_parallel {
            for tile in self.state.chunks_mut(tile_size) {
                apply_low_stages_in_tile(tile, twiddles_scaled);
            }
        }

        apply_bit_reverse_permutation(&mut self.state, n);
    }

    #[inline]
    pub(super) fn apply_qft_block_textbook(&mut self, start: usize, num: usize) {
        assert!(start + num <= self.num_qubits);
        assert!(num >= 1);

        let h = Gate::H.matrix_2x2();
        for step in qft_textbook_steps(start, num) {
            match step {
                QftTextbookStep::Hadamard(q) => self.apply_single_gate(q, h),
                QftTextbookStep::CPhase {
                    control,
                    target,
                    theta,
                } => self.apply_cu_phase(control, target, Complex64::from_polar(1.0, theta)),
                QftTextbookStep::Swap(a, b) => self.apply_swap(a, b),
            }
        }
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
    /// - **L2 tier** (target 0..13): 256KB tiles, all applied per tile in L2 cache
    /// - **L3 tier** (target 14..16): 2MB tiles, applied per tile in L3 cache
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
