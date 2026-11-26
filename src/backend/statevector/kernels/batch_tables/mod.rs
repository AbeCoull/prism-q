//! Lookup tables and tile kernels for the batched diagonal gates.

use super::*;

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
#[cfg(target_arch = "x86_64")]
const BATCH_RZZ_BMI2_MAX_GROUPS: usize = 8;

const _: () = assert!(
    MAX_BATCH_PHASE_GROUPS * BATCH_PHASE_GROUP_SIZE == BatchPhaseData::MAX_PHASES
        && MAX_BATCH_RZZ_GROUPS * BATCH_RZZ_GROUP_SIZE == BatchRzzData::MAX_EDGES,
    "fusion splits these runs against the gate-level cap, which must match the table shape"
);

pub(crate) const DIAG_BATCH_MAX_QUBITS_PER_GROUP: usize = DiagonalBatchData::MAX_QUBITS_PER_GROUP;
pub(crate) const DIAG_BATCH_TABLE_SIZE: usize = 1 << DIAG_BATCH_MAX_QUBITS_PER_GROUP;
pub(crate) const MAX_DIAG_BATCH_GROUPS: usize = DiagonalBatchData::MAX_GROUPS;

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
    assert!(
        num_groups <= MAX_BATCH_RZZ_GROUPS,
        "BatchRzz carries {} edges, past the {} the group tables hold",
        edges.len(),
        BatchRzzData::MAX_EDGES
    );

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let start = g * BATCH_RZZ_GROUP_SIZE;
        let end = (start + BATCH_RZZ_GROUP_SIZE).min(edges.len());
        let group_len = end - start;
        group.len = group_len;

        for (k, &(q0, q1, _)) in edges[start..end].iter().enumerate() {
            group.q0s[k] = q0;
            group.q1s[k] = q1;
        }

        for (bits, entry) in group.table.iter_mut().enumerate() {
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
            *entry = Complex64::from_polar(1.0, angle);
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

/// Pack `edges` into lookup groups of at most [`BATCH_RZZ_BMI2_MAX_UNIQUE`] qubits
/// each, placing every edge in the group it widens least. A group takes any number of
/// edges among its qubits, so a ring plus a random pairing fits in a few groups where
/// runs of consecutive edges would not. `None` when the edges need more than
/// [`BATCH_RZZ_BMI2_MAX_GROUPS`] groups.
#[cfg(target_arch = "x86_64")]
fn build_batch_rzz_bmi2_tables(
    edges: &[(usize, usize, f64)],
    groups: &mut [BatchRzzBmi2Group; BATCH_RZZ_BMI2_MAX_GROUPS],
) -> Option<usize> {
    debug_assert!(edges.len() <= u64::BITS as usize);
    let mut masks = [0u64; BATCH_RZZ_BMI2_MAX_GROUPS];
    let mut members = [0u64; BATCH_RZZ_BMI2_MAX_GROUPS];
    let mut num_groups = 0;
    for (e, &(q0, q1, _)) in edges.iter().enumerate() {
        let pair = (1u64 << q0) | (1u64 << q1);
        let best = (0..num_groups)
            .filter(|&g| (masks[g] | pair).count_ones() as usize <= BATCH_RZZ_BMI2_MAX_UNIQUE)
            .min_by_key(|&g| (masks[g] | pair).count_ones() - masks[g].count_ones());
        let g = match best {
            Some(g) => g,
            None if num_groups < BATCH_RZZ_BMI2_MAX_GROUPS => {
                num_groups += 1;
                num_groups - 1
            }
            None => return None,
        };
        masks[g] |= pair;
        members[g] |= 1u64 << e;
    }

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let mask = masks[g];
        group.pext_mask = mask;
        let unique = mask.count_ones() as usize;
        // Flipping table bit `k` flips the parity of every edge on it, which
        // multiplies the phase by `e^{i theta}` or its conjugate.
        let mut incident = [[(0u8, ONE); BatchRzzData::MAX_EDGES]; BATCH_RZZ_BMI2_MAX_UNIQUE];
        let mut degree = [0usize; BATCH_RZZ_BMI2_MAX_UNIQUE];
        let mut base = ONE;
        let mut bits = members[g];
        while bits != 0 {
            let (q0, q1, theta) = edges[bits.trailing_zeros() as usize];
            let p0 = (mask & ((1u64 << q0) - 1)).count_ones() as usize;
            let p1 = (mask & ((1u64 << q1) - 1)).count_ones() as usize;
            let flip = Complex64::from_polar(1.0, theta);
            incident[p0][degree[p0]] = (p1 as u8, flip);
            degree[p0] += 1;
            incident[p1][degree[p1]] = (p0 as u8, flip);
            degree[p1] += 1;
            base *= Complex64::from_polar(1.0, -theta / 2.0);
            bits &= bits - 1;
        }

        group.table[0] = base;
        for c in 1..1usize << unique {
            let k = c.trailing_zeros() as usize;
            let prev = c & (c - 1);
            let mut phase = group.table[prev];
            for &(other, flip) in &incident[k][..degree[k]] {
                phase *= if (prev >> other) & 1 == 0 {
                    flip
                } else {
                    flip.conj()
                };
            }
            group.table[c] = phase;
        }
    }
    Some(num_groups)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,fma")]
unsafe fn batch_rzz_tile_bmi2(
    tile: &mut [Complex64],
    base_idx: usize,
    groups: &[BatchRzzBmi2Group; BATCH_RZZ_BMI2_MAX_GROUPS],
    num_groups: usize,
) {
    let Some((first, rest)) = groups[..num_groups].split_first() else {
        return;
    };
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm_loadu_pd, _mm256_castpd128_pd256, _mm256_insertf128_pd, _mm256_loadu_pd,
            _mm256_movedup_pd, _mm256_permute_pd, _mm256_storeu_pd, _pext_u64,
        };
        let len = tile.len();
        let ptr = tile.as_mut_ptr() as *mut f64;
        let mut j = 0usize;
        while j + 1 < len {
            let i0 = (base_idx + j) as u64;
            let i1 = i0 + 1;
            let t0 = _mm_loadu_pd(
                std::ptr::from_ref(&first.table[_pext_u64(i0, first.pext_mask) as usize])
                    .cast::<f64>(),
            );
            let t1 = _mm_loadu_pd(
                std::ptr::from_ref(&first.table[_pext_u64(i1, first.pext_mask) as usize])
                    .cast::<f64>(),
            );
            let mut combined = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0), t1, 1);
            for group in rest {
                let g0 = _mm_loadu_pd(
                    std::ptr::from_ref(&group.table[_pext_u64(i0, group.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let g1 = _mm_loadu_pd(
                    std::ptr::from_ref(&group.table[_pext_u64(i1, group.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let phase = _mm256_insertf128_pd(_mm256_castpd128_pd256(g0), g1, 1);
                combined = simd::complex_mul_avx2fma(
                    _mm256_movedup_pd(combined),
                    _mm256_permute_pd(combined, 0b1111),
                    phase,
                );
            }
            let p = ptr.add(j * 2);
            let amp = _mm256_loadu_pd(p);
            let res = simd::complex_mul_avx2fma(
                _mm256_movedup_pd(combined),
                _mm256_permute_pd(combined, 0b1111),
                amp,
            );
            _mm256_storeu_pd(p, res);
            j += 2;
        }
        if j < len {
            let i = (base_idx + j) as u64;
            let mut combined = first.table[_pext_u64(i, first.pext_mask) as usize];
            for group in rest {
                combined *= group.table[_pext_u64(i, group.pext_mask) as usize];
            }
            tile[j] *= combined;
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct BatchPhaseGroup {
    pub(crate) table: [Complex64; BATCH_PHASE_TABLE_SIZE],
    pub(crate) shifts: [usize; BATCH_PHASE_GROUP_SIZE],
    pub(crate) len: usize,
    pub(crate) pext_mask: u64,
}

const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// Product of the first `num_groups` rzz group phases at basis index `i`.
#[inline(always)]
fn rzz_phase_at(i: usize, groups: &[BatchRzzGroup], num_groups: usize) -> Complex64 {
    let mut combined = ONE;
    for group in groups.iter().take(num_groups) {
        combined *= group.table[extract_rzz_bits(i, group)];
    }
    combined
}

/// Product of the first `num_groups` batch phase group phases at basis index `i`.
#[inline(always)]
fn batch_phase_at(i: usize, groups: &[BatchPhaseGroup], num_groups: usize) -> Complex64 {
    let mut combined = ONE;
    for group in groups.iter().take(num_groups) {
        combined *= group.table[extract_bits(i, &group.shifts, group.len)];
    }
    combined
}

impl BatchRzzGroup {
    const EMPTY: Self = Self {
        table: [ONE; BATCH_RZZ_TABLE_SIZE],
        q0s: [0; BATCH_RZZ_GROUP_SIZE],
        q1s: [0; BATCH_RZZ_GROUP_SIZE],
        len: 0,
    };
}

#[cfg(target_arch = "x86_64")]
impl BatchRzzBmi2Group {
    const EMPTY: Self = Self {
        table: [ONE; BATCH_RZZ_BMI2_TABLE_SIZE],
        pext_mask: 0,
    };
}

impl BatchPhaseGroup {
    const EMPTY: Self = Self {
        table: [ONE; BATCH_PHASE_TABLE_SIZE],
        shifts: [0; BATCH_PHASE_GROUP_SIZE],
        len: 0,
        pext_mask: 0,
    };
}

/// Lookup tables the batched diagonal kernels fill per gate.
///
/// Heap-resident and reused across gates, so a gate rewrites only the table prefix its
/// groups index instead of initialising a 64 KB array on the stack. Each table allocates
/// on the first gate of its kind.
#[derive(Default)]
pub(crate) struct BatchTableScratch {
    rzz: Option<Box<[BatchRzzGroup; MAX_BATCH_RZZ_GROUPS]>>,
    #[cfg(target_arch = "x86_64")]
    rzz_bmi2: Option<Box<[BatchRzzBmi2Group; BATCH_RZZ_BMI2_MAX_GROUPS]>>,
    phase: Option<Box<[BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS]>>,
    diag: Option<Box<DiagBatchTables>>,
}

impl BatchTableScratch {
    fn rzz(&mut self) -> &mut [BatchRzzGroup; MAX_BATCH_RZZ_GROUPS] {
        self.rzz
            .get_or_insert_with(|| Box::new([BatchRzzGroup::EMPTY; MAX_BATCH_RZZ_GROUPS]))
    }

    #[cfg(target_arch = "x86_64")]
    fn rzz_bmi2(&mut self) -> &mut [BatchRzzBmi2Group; BATCH_RZZ_BMI2_MAX_GROUPS] {
        self.rzz_bmi2
            .get_or_insert_with(|| Box::new([BatchRzzBmi2Group::EMPTY; BATCH_RZZ_BMI2_MAX_GROUPS]))
    }

    fn phase(&mut self) -> &mut [BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS] {
        self.phase
            .get_or_insert_with(|| Box::new([BatchPhaseGroup::EMPTY; MAX_BATCH_PHASE_GROUPS]))
    }

    fn diag(&mut self) -> &mut DiagBatchTables {
        self.diag
            .get_or_insert_with(|| Box::new(DiagBatchTables::empty()))
    }
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
    let num_groups = phases.len().div_ceil(BATCH_PHASE_GROUP_SIZE);
    assert!(
        num_groups <= MAX_BATCH_PHASE_GROUPS,
        "BatchPhase carries {} entries, past the {} the group tables hold",
        phases.len(),
        BatchPhaseData::MAX_PHASES
    );

    for (g, group) in groups.iter_mut().enumerate().take(num_groups) {
        let start = g * BATCH_PHASE_GROUP_SIZE;
        let end = (start + BATCH_PHASE_GROUP_SIZE).min(phases.len());
        let group_len = end - start;
        group.len = group_len;

        let mut sorted: [(usize, Complex64); BATCH_PHASE_GROUP_SIZE] =
            [(0, ONE); BATCH_PHASE_GROUP_SIZE];
        sorted[..group_len].copy_from_slice(&phases[start..end]);
        sorted[..group_len].sort_unstable_by_key(|&(q, _)| q);

        let mut mask = 0u64;
        for (j, &(q, _)) in sorted.iter().enumerate().take(group_len) {
            group.shifts[j] = q;
            mask |= 1u64 << q;
        }
        group.pext_mask = mask;

        group.table[0] = ONE;
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

impl DiagBatchTables {
    pub(super) fn empty() -> Self {
        Self {
            tables: [[ONE; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
            unique_qubits: SmallVec::new(),
            group_sizes: [0; MAX_DIAG_BATCH_GROUPS],
            group_pext_masks: [0; MAX_DIAG_BATCH_GROUPS],
            num_groups: 0,
        }
    }
}

/// Root of `x` in the qubit-affinity forest, with path halving.
fn find_root(parent: &mut [u8; 64], mut x: u8) -> u8 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Group `entries` by qubit affinity and build per-group phase LUTs.
///
/// A two-qubit entry folds its phase into a single group's table, so both of its qubits
/// have to share a group. Qubits joined by such entries therefore form components, and the
/// components are packed into groups as units, largest first.
///
/// Returns `None` when a component is wider than `DIAG_BATCH_MAX_QUBITS_PER_GROUP` or the
/// packing needs more than `MAX_DIAG_BATCH_GROUPS` groups. Both cases need the per-element
/// fallback path. Returns `Some` with `num_groups == 0` when `entries` is empty (no-op).
#[cfg(any(test, feature = "gpu"))]
pub(crate) fn build_diagonal_batch_tables(entries: &[DiagEntry]) -> Option<DiagBatchTables> {
    let mut out = DiagBatchTables::empty();
    fill_diagonal_batch_tables(entries, &mut out).then_some(out)
}

/// [`build_diagonal_batch_tables`] into a reused `out`, touching only the table prefix
/// each group indexes. Returns false when the entries are not groupable; `out` is then
/// stale and the caller takes the per-element fallback.
pub(super) fn fill_diagonal_batch_tables(entries: &[DiagEntry], out: &mut DiagBatchTables) -> bool {
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

    if num_unique == 0 {
        out.unique_qubits.clear();
        out.group_sizes = [0; MAX_DIAG_BATCH_GROUPS];
        out.group_pext_masks = [0; MAX_DIAG_BATCH_GROUPS];
        out.num_groups = 0;
        return true;
    }

    let mut dense = [0u8; 64];
    for (idx, &q) in unique_qubits.iter().enumerate() {
        dense[q] = idx as u8;
    }
    let mut parent: [u8; 64] = std::array::from_fn(|i| i as u8);
    for e in entries {
        if let DiagEntry::Phase2q { q0, q1, .. } | DiagEntry::Parity2q { q0, q1, .. } = e {
            let ra = find_root(&mut parent, dense[*q0]);
            let rb = find_root(&mut parent, dense[*q1]);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }

    let mut comp_of = [0u8; 64];
    let mut comp_len: SmallVec<[usize; 64]> = SmallVec::new();
    let mut slot_of = [u8::MAX; 64];
    for &q in &unique_qubits {
        let root = find_root(&mut parent, dense[q]) as usize;
        if slot_of[root] == u8::MAX {
            slot_of[root] = comp_len.len() as u8;
            comp_len.push(0);
        }
        comp_of[q] = slot_of[root];
        comp_len[slot_of[root] as usize] += 1;
    }
    let mut order: SmallVec<[u8; 64]> = (0..comp_len.len() as u8).collect();
    order.sort_unstable_by_key(|&c| std::cmp::Reverse(comp_len[c as usize]));

    let mut group_of_comp = [0u8; 64];
    let mut group_sizes = [0usize; MAX_DIAG_BATCH_GROUPS];
    for &c in &order {
        let len = comp_len[c as usize];
        let Some(g) = group_sizes
            .iter()
            .position(|&used| used + len <= DIAG_BATCH_MAX_QUBITS_PER_GROUP)
        else {
            return false;
        };
        group_of_comp[c as usize] = g as u8;
        group_sizes[g] += len;
    }
    let num_groups = group_sizes.iter().filter(|&&size| size > 0).count();

    let mut qubit_group = [0u8; 64];
    let mut qubit_pos = [0u8; 64];
    let mut group_pext_masks = [0u64; MAX_DIAG_BATCH_GROUPS];
    let sorted = std::mem::take(&mut unique_qubits);
    for (g, mask) in group_pext_masks.iter_mut().enumerate().take(num_groups) {
        let members = sorted
            .iter()
            .filter(|&&q| group_of_comp[comp_of[q] as usize] as usize == g);
        for (pos, &q) in members.enumerate() {
            qubit_group[q] = g as u8;
            qubit_pos[q] = pos as u8;
            *mask |= 1u64 << q;
            unique_qubits.push(q);
        }
    }

    let tables = &mut out.tables;
    for (table, &k) in tables.iter_mut().zip(&group_sizes).take(num_groups) {
        table[..1 << k].fill(ONE);
    }

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
                let g = qubit_group[*q0] as usize;
                let p0 = qubit_pos[*q0] as usize;
                let p1 = qubit_pos[*q1] as usize;
                debug_assert_eq!(g, qubit_group[*q1] as usize);
                let k = group_sizes[g];
                for (bits, entry) in tables[g][..1 << k].iter_mut().enumerate() {
                    if ((bits >> p0) & 1 == 1) && ((bits >> p1) & 1 == 1) {
                        *entry *= phase;
                    }
                }
            }
            DiagEntry::Parity2q { q0, q1, same, diff } => {
                let g = qubit_group[*q0] as usize;
                let p0 = qubit_pos[*q0] as usize;
                let p1 = qubit_pos[*q1] as usize;
                debug_assert_eq!(g, qubit_group[*q1] as usize);
                let k = group_sizes[g];
                for (bits, entry) in tables[g][..1 << k].iter_mut().enumerate() {
                    let parity = ((bits >> p0) ^ (bits >> p1)) & 1;
                    *entry *= if parity == 0 { *same } else { *diff };
                }
            }
        }
    }

    out.unique_qubits = unique_qubits;
    out.group_sizes = group_sizes;
    out.group_pext_masks = group_pext_masks;
    out.num_groups = num_groups;
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,fma")]
unsafe fn apply_batch_phase_bmi2(
    state: &mut [Complex64],
    ctrl_bit: usize,
    groups: &[BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS],
    num_groups: usize,
) {
    debug_assert!(
        state.len().is_power_of_two() && ctrl_bit < state.len().trailing_zeros() as usize
    );
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe { batch_phase_tile_bmi2(state, 0, ctrl_bit, groups, num_groups) };
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi2,fma")]
unsafe fn batch_phase_tile_bmi2(
    tile: &mut [Complex64],
    base_idx: usize,
    ctrl_bit: usize,
    groups: &[BatchPhaseGroup; MAX_BATCH_PHASE_GROUPS],
    num_groups: usize,
) {
    let Some((first, rest)) = groups[..num_groups].split_first() else {
        return;
    };
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm_loadu_pd, _mm_storeu_pd, _mm256_castpd128_pd256, _mm256_castpd256_pd128,
            _mm256_extractf128_pd, _mm256_insertf128_pd, _mm256_loadu_pd, _mm256_movedup_pd,
            _mm256_permute_pd, _mm256_storeu_pd, _pext_u64,
        };
        let ctrl_mask = 1usize << ctrl_bit;
        let tile_bits = tile.len().trailing_zeros() as usize;
        let ptr = tile.as_mut_ptr() as *mut f64;

        if ctrl_bit >= tile_bits {
            if (base_idx & ctrl_mask) == 0 {
                return;
            }
            let len = tile.len();
            let mut j = 0usize;
            while j + 1 < len {
                let i0 = (base_idx + j) as u64;
                let i1 = i0 + 1;
                let t0 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i0, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let t1 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i1, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let mut combined = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0), t1, 1);
                for group in rest {
                    let g0 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i0, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let g1 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i1, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let phase = _mm256_insertf128_pd(_mm256_castpd128_pd256(g0), g1, 1);
                    combined = simd::complex_mul_avx2fma(
                        _mm256_movedup_pd(combined),
                        _mm256_permute_pd(combined, 0b1111),
                        phase,
                    );
                }
                let p = ptr.add(j * 2);
                let amp = _mm256_loadu_pd(p);
                let res = simd::complex_mul_avx2fma(
                    _mm256_movedup_pd(combined),
                    _mm256_permute_pd(combined, 0b1111),
                    amp,
                );
                _mm256_storeu_pd(p, res);
                j += 2;
            }
            if j < len {
                let i = (base_idx + j) as u64;
                let mut combined = first.table[_pext_u64(i, first.pext_mask) as usize];
                for group in rest {
                    combined *= group.table[_pext_u64(i, group.pext_mask) as usize];
                }
                tile[j] *= combined;
            }
        } else if ctrl_bit >= 1 {
            let half = tile.len() >> 1;
            let mut k = 0usize;
            while k + 1 < half {
                let j0 = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
                let i0 = (base_idx + j0) as u64;
                let i1 = i0 + 1;
                let t0 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i0, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let t1 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i1, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let mut combined = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0), t1, 1);
                for group in rest {
                    let g0 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i0, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let g1 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i1, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let phase = _mm256_insertf128_pd(_mm256_castpd128_pd256(g0), g1, 1);
                    combined = simd::complex_mul_avx2fma(
                        _mm256_movedup_pd(combined),
                        _mm256_permute_pd(combined, 0b1111),
                        phase,
                    );
                }
                let p = ptr.add(j0 * 2);
                let amp = _mm256_loadu_pd(p);
                let res = simd::complex_mul_avx2fma(
                    _mm256_movedup_pd(combined),
                    _mm256_permute_pd(combined, 0b1111),
                    amp,
                );
                _mm256_storeu_pd(p, res);
                k += 2;
            }
            if k < half {
                let j = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
                let i = (base_idx + j) as u64;
                let mut combined = first.table[_pext_u64(i, first.pext_mask) as usize];
                for group in rest {
                    combined *= group.table[_pext_u64(i, group.pext_mask) as usize];
                }
                // SAFETY: j < tile.len() guaranteed by insert_zero_bit mapping
                *tile.get_unchecked_mut(j) *= combined;
            }
        } else {
            let half = tile.len() >> 1;
            let mut k = 0usize;
            while k + 1 < half {
                let j0 = (k << 1) | 1;
                let j1 = j0 + 2;
                let i0 = (base_idx + j0) as u64;
                let i1 = (base_idx + j1) as u64;
                let t0 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i0, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let t1 = _mm_loadu_pd(
                    std::ptr::from_ref(&first.table[_pext_u64(i1, first.pext_mask) as usize])
                        .cast::<f64>(),
                );
                let mut combined = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0), t1, 1);
                for group in rest {
                    let g0 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i0, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let g1 = _mm_loadu_pd(
                        std::ptr::from_ref(&group.table[_pext_u64(i1, group.pext_mask) as usize])
                            .cast::<f64>(),
                    );
                    let phase = _mm256_insertf128_pd(_mm256_castpd128_pd256(g0), g1, 1);
                    combined = simd::complex_mul_avx2fma(
                        _mm256_movedup_pd(combined),
                        _mm256_permute_pd(combined, 0b1111),
                        phase,
                    );
                }
                let p0 = ptr.add(j0 * 2);
                let p1 = ptr.add(j1 * 2);
                let amp = _mm256_insertf128_pd(
                    _mm256_castpd128_pd256(_mm_loadu_pd(p0)),
                    _mm_loadu_pd(p1),
                    1,
                );
                let res = simd::complex_mul_avx2fma(
                    _mm256_movedup_pd(combined),
                    _mm256_permute_pd(combined, 0b1111),
                    amp,
                );
                _mm_storeu_pd(p0, _mm256_castpd256_pd128(res));
                _mm_storeu_pd(p1, _mm256_extractf128_pd(res, 1));
                k += 2;
            }
            if k < half {
                let j = (k << 1) | 1;
                let i = (base_idx + j) as u64;
                let mut combined = first.table[_pext_u64(i, first.pext_mask) as usize];
                for group in rest {
                    combined *= group.table[_pext_u64(i, group.pext_mask) as usize];
                }
                // SAFETY: j < tile.len() guaranteed by (k << 1) | 1 for k < half
                *tile.get_unchecked_mut(j) *= combined;
            }
        }
    }
}

impl StatevectorBackend {
    #[inline(always)]
    pub(in crate::backend::statevector) fn apply_batch_rzz(
        &mut self,
        edges: &[(usize, usize, f64)],
    ) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_batch_rzz_par(edges);
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::has_bmi2() {
            let bmi2_groups = self.batch_tables.rzz_bmi2();
            if let Some(num_groups) = build_batch_rzz_bmi2_tables(edges, bmi2_groups) {
                // SAFETY: AVX2 detected, along with BMI2 and FMA.
                unsafe { batch_rzz_tile_bmi2(&mut self.state, 0, bmi2_groups, num_groups) };
                return;
            }
        }

        let groups = self.batch_tables.rzz();
        let num_groups = build_batch_rzz_tables(edges, groups);

        for (i, amp) in self.state.iter_mut().enumerate() {
            *amp *= rzz_phase_at(i, groups, num_groups);
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_batch_rzz_par(&mut self, edges: &[(usize, usize, f64)]) {
        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::has_bmi2() {
            let bmi2_groups = self.batch_tables.rzz_bmi2();
            if let Some(num_groups) = build_batch_rzz_bmi2_tables(edges, bmi2_groups) {
                let bmi2_groups = &*bmi2_groups;
                self.state
                    .par_chunks_mut(MIN_PAR_ELEMS)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let base = chunk_idx * MIN_PAR_ELEMS;
                        // SAFETY: AVX2 detected, along with BMI2 and FMA.
                        unsafe {
                            batch_rzz_tile_bmi2(chunk, base, bmi2_groups, num_groups);
                        }
                    });
                return;
            }
        }

        let groups = self.batch_tables.rzz();
        let num_groups = build_batch_rzz_tables(edges, groups);
        let groups = &*groups;

        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let base = chunk_idx * MIN_PAR_ELEMS;
                for (j, amp) in chunk.iter_mut().enumerate() {
                    *amp *= rzz_phase_at(base + j, groups, num_groups);
                }
            });
    }

    #[inline(always)]
    pub(in crate::backend::statevector) fn apply_batch_phase(
        &mut self,
        control: usize,
        phases: &[(usize, Complex64)],
    ) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_batch_phase_par(control, phases);
            return;
        }

        let groups = self.batch_tables.phase();
        let num_groups = build_batch_phase_tables(phases, groups);

        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::has_bmi2() {
            // SAFETY: AVX2 detected, along with BMI2 and FMA.
            unsafe {
                apply_batch_phase_bmi2(&mut self.state, control, groups, num_groups);
            }
            return;
        }

        let ctrl_mask = 1usize << control;
        let half = 1usize << (self.num_qubits - 1);
        for k in 0..half {
            let i = insert_zero_bit(k, control) | ctrl_mask;
            self.state[i] *= batch_phase_at(i, groups, num_groups);
        }
    }

    #[cfg(feature = "parallel")]
    #[inline(always)]
    fn apply_batch_phase_par(&mut self, control: usize, phases: &[(usize, Complex64)]) {
        let groups = self.batch_tables.phase();
        let num_groups = build_batch_phase_tables(phases, groups);
        let groups = &*groups;

        #[cfg(target_arch = "x86_64")]
        let use_bmi2 = simd::has_avx2_fma() && simd::has_bmi2();

        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(tile_idx, tile)| {
                let base = tile_idx * MIN_PAR_ELEMS;

                #[cfg(target_arch = "x86_64")]
                if use_bmi2 {
                    // SAFETY: AVX2 detected, along with BMI2 and FMA.
                    unsafe {
                        batch_phase_tile_bmi2(tile, base, control, groups, num_groups);
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
                        *amp *= batch_phase_at(base + j, groups, num_groups);
                    }
                } else {
                    let half = tile.len() >> 1;
                    for k in 0..half {
                        let j = insert_zero_bit(k, control) | ctrl_mask;
                        tile[j] *= batch_phase_at(base + j, groups, num_groups);
                    }
                }
            });
    }

    pub(in crate::backend::statevector) fn apply_diagonal_batch(&mut self, entries: &[DiagEntry]) {
        #[cfg(test)]
        crate::backend::statevector::sweeps::tick();
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.apply_diagonal_batch_par(entries);
            return;
        }

        if !fill_diagonal_batch_tables(entries, self.batch_tables.diag()) {
            self.apply_diagonal_batch_fallback(entries);
            return;
        }
        let built = &*self.batch_tables.diag();
        if built.num_groups == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::has_bmi2() {
            // SAFETY: AVX2 detected, along with BMI2 and FMA.
            unsafe {
                diagonal_batch_tile_bmi2(
                    &mut self.state,
                    0,
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

    #[cfg(feature = "parallel")]
    fn apply_diagonal_batch_par(&mut self, entries: &[DiagEntry]) {
        if !fill_diagonal_batch_tables(entries, self.batch_tables.diag()) {
            self.apply_diagonal_batch_fallback(entries);
            return;
        }
        let built = &*self.batch_tables.diag();
        if built.num_groups == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        if simd::has_avx2_fma() && simd::has_bmi2() {
            self.state
                .par_chunks_mut(MIN_PAR_ELEMS)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * MIN_PAR_ELEMS;
                    // SAFETY: AVX2 detected, along with BMI2 and FMA.
                    unsafe {
                        diagonal_batch_tile_bmi2(
                            chunk,
                            base,
                            &built.tables,
                            &built.group_pext_masks,
                            built.num_groups,
                        );
                    }
                });
            return;
        }

        let group_shifts = diag_batch_group_shifts(&built.unique_qubits, &built.group_sizes);
        self.state
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                diagonal_batch_tile_scalar(
                    chunk,
                    chunk_idx * MIN_PAR_ELEMS,
                    &built.tables,
                    &group_shifts,
                    &built.group_sizes,
                    built.num_groups,
                );
            });
    }

    fn apply_diagonal_batch_fallback(&mut self, entries: &[DiagEntry]) {
        #[cfg(feature = "parallel")]
        if self.num_qubits >= kernel_parallel_qubits() {
            self.state
                .par_chunks_mut(MIN_PAR_ELEMS)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let base = chunk_idx * MIN_PAR_ELEMS;
                    for (j, amp) in chunk.iter_mut().enumerate() {
                        *amp *= diag_entries_phase(base + j, entries);
                    }
                });
            return;
        }
        for (i, amp) in self.state.iter_mut().enumerate() {
            *amp *= diag_entries_phase(i, entries);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,bmi2,fma")]
pub(super) unsafe fn diagonal_batch_tile_bmi2(
    tile: &mut [Complex64],
    base_idx: usize,
    tables: &[[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    pext_masks: &[u64; MAX_DIAG_BATCH_GROUPS],
    num_groups: usize,
) {
    if num_groups == 0 {
        return;
    }
    let first_table = &tables[0];
    let first_mask = pext_masks[0];
    let rest_tables = &tables[1..num_groups];
    let rest_masks = &pext_masks[1..num_groups];
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::{
            _mm_loadu_pd, _mm256_castpd128_pd256, _mm256_insertf128_pd, _mm256_loadu_pd,
            _mm256_movedup_pd, _mm256_permute_pd, _mm256_storeu_pd, _pext_u64,
        };
        let len = tile.len();
        let ptr = tile.as_mut_ptr() as *mut f64;
        let mut j = 0usize;
        while j + 1 < len {
            let i0 = (base_idx + j) as u64;
            let i1 = i0 + 1;
            let t0 = _mm_loadu_pd(
                std::ptr::from_ref(&first_table[_pext_u64(i0, first_mask) as usize]).cast::<f64>(),
            );
            let t1 = _mm_loadu_pd(
                std::ptr::from_ref(&first_table[_pext_u64(i1, first_mask) as usize]).cast::<f64>(),
            );
            let mut combined = _mm256_insertf128_pd(_mm256_castpd128_pd256(t0), t1, 1);
            for (tg, &mg) in rest_tables.iter().zip(rest_masks) {
                let g0 =
                    _mm_loadu_pd(std::ptr::from_ref(&tg[_pext_u64(i0, mg) as usize]).cast::<f64>());
                let g1 =
                    _mm_loadu_pd(std::ptr::from_ref(&tg[_pext_u64(i1, mg) as usize]).cast::<f64>());
                let phase = _mm256_insertf128_pd(_mm256_castpd128_pd256(g0), g1, 1);
                combined = simd::complex_mul_avx2fma(
                    _mm256_movedup_pd(combined),
                    _mm256_permute_pd(combined, 0b1111),
                    phase,
                );
            }
            let p = ptr.add(j * 2);
            let amp = _mm256_loadu_pd(p);
            let res = simd::complex_mul_avx2fma(
                _mm256_movedup_pd(combined),
                _mm256_permute_pd(combined, 0b1111),
                amp,
            );
            _mm256_storeu_pd(p, res);
            j += 2;
        }
        if j < len {
            let i = (base_idx + j) as u64;
            let mut combined = first_table[_pext_u64(i, first_mask) as usize];
            for (tg, &mg) in rest_tables.iter().zip(rest_masks) {
                combined *= tg[_pext_u64(i, mg) as usize];
            }
            tile[j] *= combined;
        }
    }
}

/// Per-group qubit positions, the non-x86_64 stand-in for the PEXT masks.
///
/// `unique_qubits` holds the groups concatenated in order, and a group can be shorter than
/// `DIAG_BATCH_MAX_QUBITS_PER_GROUP`, so `group_sizes` drives the split.
pub(super) fn diag_batch_group_shifts(
    unique_qubits: &[usize],
    group_sizes: &[usize; MAX_DIAG_BATCH_GROUPS],
) -> [[usize; DIAG_BATCH_MAX_QUBITS_PER_GROUP]; MAX_DIAG_BATCH_GROUPS] {
    let mut group_shifts = [[0usize; DIAG_BATCH_MAX_QUBITS_PER_GROUP]; MAX_DIAG_BATCH_GROUPS];
    let mut flat = 0;
    for (shifts, &size) in group_shifts.iter_mut().zip(group_sizes) {
        shifts[..size].copy_from_slice(&unique_qubits[flat..flat + size]);
        flat += size;
    }
    group_shifts
}

pub(super) fn diagonal_batch_tile_scalar(
    tile: &mut [Complex64],
    base_idx: usize,
    tables: &[[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    group_shifts: &[[usize; DIAG_BATCH_MAX_QUBITS_PER_GROUP]; MAX_DIAG_BATCH_GROUPS],
    group_sizes: &[usize; MAX_DIAG_BATCH_GROUPS],
    num_groups: usize,
) {
    for (j, amp) in tile.iter_mut().enumerate() {
        let i = base_idx + j;
        let mut combined = ONE;
        for (g, shifts) in group_shifts.iter().enumerate().take(num_groups) {
            let mut bits = 0usize;
            for (p, &shift) in shifts[..group_sizes[g]].iter().enumerate() {
                bits |= ((i >> shift) & 1) << p;
            }
            combined *= tables[g][bits];
        }
        *amp *= combined;
    }
}

fn apply_diagonal_batch_scalar(
    state: &mut [Complex64],
    tables: &[[Complex64; DIAG_BATCH_TABLE_SIZE]; MAX_DIAG_BATCH_GROUPS],
    unique_qubits: &[usize],
    group_sizes: &[usize; MAX_DIAG_BATCH_GROUPS],
    num_groups: usize,
) {
    let group_shifts = diag_batch_group_shifts(unique_qubits, group_sizes);
    diagonal_batch_tile_scalar(state, 0, tables, &group_shifts, group_sizes, num_groups);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests;
