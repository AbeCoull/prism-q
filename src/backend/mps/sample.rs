//! Dense expansion of the MPS chain into amplitudes and the conditional site weights
//! native sampling draws from.

use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::{MpsBackend, ONE, ZERO};

/// Leaves per Rayon task in the dense expansion, as a bit count. The split
/// depth is chosen to leave at least `2^6` amplitudes under each prefix, so
/// the gate is per-task work rather than total element count: each element
/// here costs a site contraction, not the few flops a statevector sweep pays.
#[cfg(feature = "parallel")]
const MIN_LEAVES_PER_TASK_BITS: usize = 6;
/// Cap on the split depth, bounding the serial prefix walk and the number of
/// scratch stacks allocated.
#[cfg(feature = "parallel")]
const MAX_SPLIT_BITS: usize = 6;

/// Raw output pointer for the dense expansion's Rayon split.
#[cfg(feature = "parallel")]
#[derive(Copy, Clone)]
struct DenseOutPtr<T>(*mut T);

#[cfg(feature = "parallel")]
// SAFETY: the expansion partitions basis indices by their top-site bits before
// entering Rayon, so each task writes a disjoint set of elements.
unsafe impl<T: Send> Send for DenseOutPtr<T> {}
#[cfg(feature = "parallel")]
// SAFETY: same partition as Send; sharing the wrapper grants no overlap.
unsafe impl<T: Send> Sync for DenseOutPtr<T> {}

#[cfg(feature = "parallel")]
impl<T> DenseOutPtr<T> {
    #[inline(always)]
    unsafe fn store(self, idx: usize, val: T) {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            *self.0.add(idx) = val;
        }
    }
}

impl MpsBackend {
    /// Conditional outcome weights at `site`, given the boundary vector `left`
    /// that carries the bits already fixed to its left.
    ///
    /// Writes `w[i * bond_right + β] = Σ_α left[α] · A[α, i, β]` into `w` and
    /// returns the unnormalized weight of each outcome. The conditioned left
    /// environment is the rank-one `left[α]·conj(left[α'])`, which collapses
    /// the four-index contraction a general environment owes to `O(χ²)`.
    pub(super) fn site_conditional_weights(
        &self,
        site: usize,
        left: &[Complex64],
        right: &[Complex64],
        w: &mut [Complex64],
    ) -> [f64; 2] {
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;

        let w = &mut w[..2 * br];
        w.fill(ZERO);
        for (alpha, &l) in left[..bl].iter().enumerate() {
            if l == ZERO {
                continue;
            }
            let row = &t.data[alpha * (2 * br)..(alpha + 1) * (2 * br)];
            for (dst, &src) in w.iter_mut().zip(row) {
                *dst += l * src;
            }
        }

        let mut prob = [0.0f64; 2];
        for (i, prob_out) in prob.iter_mut().enumerate() {
            let wi = &w[i * br..(i + 1) * br];
            let mut acc = ZERO;
            for (beta, &w_beta) in wi.iter().enumerate() {
                if w_beta == ZERO {
                    continue;
                }
                let row = &right[beta * br..(beta + 1) * br];
                let inner: Complex64 = row.iter().zip(wi).map(|(&r, &v)| r * v.conj()).sum();
                acc += w_beta * inner;
            }
            *prob_out = acc.re;
        }
        prob
    }

    /// Contract site `site` against the right-side vector `src`, selecting
    /// physical index `bit`, into `dst` of length `bond_left`.
    #[inline(always)]
    fn contract_site_right(
        &self,
        site: usize,
        bit: usize,
        src: &[Complex64],
        dst: &mut [Complex64],
    ) {
        let t = &self.sites[site];
        debug_assert_eq!(src.len(), t.bond_right);
        debug_assert_eq!(dst.len(), t.bond_left);
        for (alpha, slot) in dst.iter_mut().enumerate() {
            let mut acc = ZERO;
            for (beta, &v) in src.iter().enumerate() {
                acc += t.data[t.idx(alpha, bit, beta)] * v;
            }
            *slot = acc;
        }
    }

    /// Depth-first expansion of the subtree below `depth`, writing one
    /// amplitude per leaf through `store`.
    ///
    /// Sites are consumed right to left, so `scratch[d]` holds the vector on
    /// the left bond of site `n - d` and the deepest level decides site 0.
    /// With an unpermuted layout that makes consecutive leaves differ in the
    /// lowest basis bit, so leaf writes run in address order.
    fn expand_subtree(
        &self,
        depth: usize,
        basis: usize,
        scratch: &mut [Vec<Complex64>],
        store: &mut impl FnMut(usize, Complex64),
    ) {
        let n = self.num_qubits;
        if depth == n {
            store(basis, scratch[n][0]);
            return;
        }
        let site = n - 1 - depth;
        let logical = self.site_to_logical[site];
        for bit in 0..2 {
            let (lo, hi) = scratch.split_at_mut(depth + 1);
            self.contract_site_right(site, bit, &lo[depth], &mut hi[0]);
            self.expand_subtree(depth + 1, basis | (bit << logical), scratch, store);
        }
    }

    /// Scratch stack for [`MpsBackend::expand_subtree`], one buffer per depth.
    fn expansion_scratch(&self) -> Vec<Vec<Complex64>> {
        let n = self.num_qubits;
        let mut scratch = Vec::with_capacity(n + 1);
        scratch.push(vec![ONE]);
        for depth in 1..=n {
            scratch.push(vec![ZERO; self.sites[n - depth].bond_left]);
        }
        scratch
    }

    /// Every `(basis prefix, left-bond vector)` pair at `depth`, in DFS order,
    /// grown from `root`.
    #[cfg(feature = "parallel")]
    fn expansion_prefixes(&self, depth: usize, root: Complex64) -> Vec<(usize, Vec<Complex64>)> {
        let mut scratch = self.expansion_scratch();
        scratch[0][0] = root;
        let mut out = Vec::with_capacity(1 << depth);
        self.collect_prefixes(0, depth, 0, &mut scratch, &mut out);
        out
    }

    #[cfg(feature = "parallel")]
    fn collect_prefixes(
        &self,
        depth: usize,
        stop: usize,
        basis: usize,
        scratch: &mut [Vec<Complex64>],
        out: &mut Vec<(usize, Vec<Complex64>)>,
    ) {
        if depth == stop {
            out.push((basis, scratch[depth].clone()));
            return;
        }
        let site = self.num_qubits - 1 - depth;
        let logical = self.site_to_logical[site];
        for bit in 0..2 {
            let (lo, hi) = scratch.split_at_mut(depth + 1);
            self.contract_site_right(site, bit, &lo[depth], &mut hi[0]);
            self.collect_prefixes(depth + 1, stop, basis | (bit << logical), scratch, out);
        }
    }

    /// Fill `out` with `convert(amplitude)` per basis index, sharing every
    /// prefix contraction across the `2^(n-k)` basis states below it.
    ///
    /// [`MpsBackend::chain_amplitude`] recomputes that prefix per amplitude and
    /// allocates once per site, so a dense fill costs `2^n * sum_k bl_k*br_k`
    /// against `sum_k 2^(k+1)*bl_k*br_k` here. The parallel split fixes the top
    /// `m` sites, whose basis-bit sets are disjoint across prefixes.
    ///
    /// The root vector carries `1/sqrt(⟨ψ|ψ⟩)`, so a truncated chain expands to
    /// the normalized state at no cost per leaf.
    pub(super) fn expand_dense<T: Copy + Send>(
        &self,
        out: &mut [T],
        convert: impl Fn(Complex64) -> T + Sync,
    ) {
        let root = self.normalized_root();
        #[cfg(feature = "parallel")]
        {
            let split = self
                .num_qubits
                .saturating_sub(MIN_LEAVES_PER_TASK_BITS)
                .min(MAX_SPLIT_BITS);
            if split > 0 {
                let ptr = DenseOutPtr(out.as_mut_ptr());
                self.expansion_prefixes(split, root)
                    .into_par_iter()
                    .for_each(|(basis, vector)| {
                        let mut scratch = self.expansion_scratch();
                        scratch[split].copy_from_slice(&vector);
                        self.expand_subtree(split, basis, &mut scratch, &mut |idx, amp| {
                            // SAFETY: this task owns every basis index whose top
                            // `split` site bits match its prefix, and those sets
                            // partition the output, so no two tasks reach the
                            // same element.
                            unsafe { ptr.store(idx, convert(amp)) }
                        });
                    });
                return;
            }
        }

        let mut scratch = self.expansion_scratch();
        scratch[0][0] = root;
        self.expand_subtree(0, 0, &mut scratch, &mut |idx, amp| out[idx] = convert(amp));
    }

    /// A zero-norm chain keeps the unit root, the same guard
    /// [`Backend::pauli_expectations`](crate::backend::Backend::pauli_expectations)
    /// applies.
    fn normalized_root(&self) -> Complex64 {
        let norm = self.pauli_env_prefixes()[self.num_qubits][0].re;
        if norm > 0.0 {
            Complex64::new(1.0 / norm.sqrt(), 0.0)
        } else {
            ONE
        }
    }
}
