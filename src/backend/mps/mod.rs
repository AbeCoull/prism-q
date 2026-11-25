//! Matrix Product State (MPS) simulation backend.
//!
//! Represents an n-qubit state as a chain of rank-3 tensors:
//!
//! ```text
//!   A[0] - A[1] - A[2] - ... - A[n-1]
//! ```
//!
//! Each tensor `A[k]` has shape (χ_left, 2, χ_right) where χ is the bond dimension.
//! Single-qubit gates are absorbed into the local tensor in O(χ²).
//! Two-qubit gates on adjacent sites use SVD truncation in O(χ³).
//! Non-adjacent two-qubit gates are routed via SWAP chains.
//!
//! # Memory layout
//!
//! - Each site tensor: contiguous `Vec<Complex64>` of length bond_left × 2 × bond_right.
//! - Element A[α, i, β] at index: `α * (2 * bond_right) + i * bond_right + β`.
//! - Bond dimension capped at `max_bond_dim`, which has no default: `MpsBackend::new`
//!   takes it, and automatic dispatch passes 256.
//! - SVD truncation applies both a relative tolerance (default 1e-12) and the
//!   bond cap.
//!
//! # Gate support
//!
//! Every single-qubit gate is absorbed into its site tensor. Adjacent
//! two-qubit gates (`Cx`, `Cz`, `Swap`, `Rzz`, `Cu`, `Fused2q`) go through the
//! SVD kernel, and non-adjacent pairs route via SWAP chains. `Mcu` expands to
//! a dense matrix over the gathered sites, `BatchPhase` sweeps its control
//! toward each target (a single phase reduces to the pair kernel), and the
//! other fused families (`Multi2q`, `BatchRzz`, `DiagonalBatch`, `MultiFused`)
//! decompose into the single-site and adjacent-pair kernels.
//!
//! # When to prefer this backend
//!
//! - 1D circuits with limited entanglement growth.
//! - Variational circuits with hardware-efficient ansatz.
//! - Large qubit counts (50+) with low entanglement.
//!
//! # When NOT to use this backend
//!
//! - Circuits with high entanglement across the chain (many Hadamards + random CX).
//! - Small qubit counts where statevector is faster and exact.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::backend::{
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_probability_len, dense_statevector_len, overlap,
    reserve_dense_output, schmidt, simd,
};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{Gate, mcu_matrix};
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::PauliTerm;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

mod env;
mod linalg;
mod nsite;
mod sample;
mod svd;

pub(crate) use env::MpsPauliAxis;
use env::mps_pauli_factors;
use linalg::{cu_phase_swap_matrix, swap_gate_qubits};

// Benchmark-only surface: benches link the crate externally and cannot see
// pub(crate), so the SVD entry points are pub; doc(hidden) keeps them out of
// the rendered API.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub use svd::svd_faer;
#[doc(hidden)]
pub use svd::{SvdResult, svd, svd_jacobi};

#[cfg(test)]
use svd::l2_norm;
use svd::{
    ThinQr, WeightSide, cut_rank, discards_resolvable_weight, fill_isometry_site_data,
    fill_scaled_u_site_data, fill_scaled_vt_data, fill_vt_site_data, scratch_slice,
    truncated_svd_rank,
};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const DEFAULT_SVD_EPSILON: f64 = 1e-12;
/// Summed relative squared weight, in the unit of
/// [`MpsBackend::truncation_discarded`], at or under which a run reports
/// `Exact`: the weight of one singular value at the default threshold.
const EXACT_DISCARD_BOUND: f64 = DEFAULT_SVD_EPSILON * DEFAULT_SVD_EPSILON;
/// Bond dimension at or above which a threshold cut takes the orthogonality
/// center whatever it drops. The environment of an ungauged chain amplifies
/// what a cut sheds by its own conditioning, which grows with the bonds:
/// brickwork chains at bonds to 128 read 1e-15 from the statevector at every
/// threshold to 1e-12, while at bond 256 a 16-qubit chain read 2.8e-7 from
/// values dropped at 8.6e-16 of the largest, under [`SVD_RESOLUTION`], and
/// another seed read 1.0e-8 with nothing dropped at all.
const GAUGE_RANK: usize = 128;
/// Relative size under which a value dropped by a cut on a chain under
/// [`GAUGE_RANK`] is taken as the factorization's own rounding.
const SVD_RESOLUTION: f64 = 4.0 * f64::EPSILON;
#[cfg(feature = "parallel")]
const MIN_BOND_FOR_PAR: usize = 32;
/// Shot count at or above which native sampling splits the shot loop across
/// the Rayon pool; per-shot substreams keep the output identical either way.
#[cfg(feature = "parallel")]
const MIN_SHOTS_FOR_PAR: usize = 32;

pub(crate) fn check_density_workspace(k: usize, chi: usize, limits: &ResolvedLimits) -> Result<()> {
    check_contraction_workspace(
        k,
        chi,
        limits,
        "a reduced density matrix over the span these qubits occupy",
    )
}

fn check_contraction_workspace(
    k: usize,
    chi: usize,
    limits: &ResolvedLimits,
    operation: &str,
) -> Result<()> {
    // Transfer object, half-contracted form and successor over 4^k open index pairs.
    let workspace = (7 * (chi as u128).pow(2)) << (2 * k);
    if workspace > limits.mps_workspace_elements() {
        return Err(crate::backend::workspace_allocation_error(
            "mps", operation, workspace, limits,
        ));
    }
    Ok(())
}

#[derive(Clone)]
struct SiteTensor {
    bond_left: usize,
    bond_right: usize,
    data: Vec<Complex64>,
}

impl SiteTensor {
    fn new_zero_state() -> Self {
        Self {
            bond_left: 1,
            bond_right: 1,
            data: vec![ONE, ZERO],
        }
    }

    #[inline(always)]
    fn idx(&self, alpha: usize, i: usize, beta: usize) -> usize {
        alpha * (2 * self.bond_right) + i * self.bond_right + beta
    }
}

/// Isometry deviation [`MpsBackend::assert_gauge`] admits. A move refactorizes
/// the site it leaves, so the deviation is that factorization's own and does
/// not accumulate: 420 moves over the eight-site fixture in `mps_tests.rs`
/// take the worst site from 4.4e-16 to 1.1e-15.
#[cfg(test)]
const GAUGE_TOLERANCE: f64 = 1e-14;

/// Matrix Product State backend with bounded bond dimension.
pub struct MpsBackend {
    num_qubits: usize,
    max_bond_dim: usize,
    svd_epsilon: f64,
    sites: Vec<SiteTensor>,
    logical_to_site: Vec<usize>,
    site_to_logical: Vec<usize>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    truncation_discarded: f64,
    /// Site the chain is canonical about, or `None` when no gauge is
    /// recorded. `Some` claims that every site left of it is a left isometry
    /// and every site right of it a right isometry, which is what makes a
    /// cut's discarded singular weight the state's 2-norm error. The two-site
    /// path establishes it where a cut is going to happen and maintains it
    /// from there; a write that cannot hold the claim clears it.
    center: Option<usize>,
    /// Widest bond any cut has written since [`Backend::init`], which decides
    /// with [`GAUGE_RANK`] whether a threshold cut takes the center and is the
    /// peak the result reports through [`Backend::bond_report`]. Read as a
    /// high-water mark so the decision costs a compare per gate rather than a
    /// scan of the chain.
    bond_high_water: usize,
    /// `limits.mps_workspace_elements()` read once at construction, so the
    /// per-gate check is a field compare.
    workspace_cap: u128,
    limits: ResolvedLimits,
    /// Grow-only scratch for [`Self::apply_adjacent_two_qubit`], held across
    /// calls so the four contraction buffers cost no allocation per gate.
    /// Every element of each call's window is written by assignment before
    /// any read, so stale contents never leak between calls.
    scratch_theta: Vec<Complex64>,
    scratch_right_t: Vec<Complex64>,
    scratch_theta_prime: Vec<Complex64>,
    scratch_mat: Vec<Complex64>,
    /// Steps [`MpsBackend::move_center`] has taken since construction, so a
    /// test can price the walk a gate order asks for.
    #[cfg(test)]
    center_steps: usize,
    /// Relative squared weight the walk's factorizations have left behind in
    /// the columns they dropped, so a test can hold the walk to its own
    /// rounding.
    #[cfg(test)]
    qr_discarded: f64,
    /// Grow-only scratch for one step of [`MpsBackend::move_center`], which
    /// runs one to three times per two-qubit gate once the center is
    /// maintained: the factorization buffers and the neighbour the step
    /// absorbs `R` into.
    qr: ThinQr,
    scratch_absorb: Vec<Complex64>,
}

// Manual so clones start with empty scratch: the buffers are write-before-read
// and the stabilizer-rank path clones branch states per shot, so copying their
// contents would be pure memcpy waste.
impl Clone for MpsBackend {
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            max_bond_dim: self.max_bond_dim,
            svd_epsilon: self.svd_epsilon,
            sites: self.sites.clone(),
            logical_to_site: self.logical_to_site.clone(),
            site_to_logical: self.site_to_logical.clone(),
            classical_bits: self.classical_bits.clone(),
            rng: self.rng.clone(),
            truncation_discarded: self.truncation_discarded,
            center: self.center,
            bond_high_water: self.bond_high_water,
            #[cfg(test)]
            center_steps: self.center_steps,
            #[cfg(test)]
            qr_discarded: self.qr_discarded,
            workspace_cap: self.workspace_cap,
            limits: self.limits,
            scratch_theta: Vec::new(),
            scratch_right_t: Vec::new(),
            scratch_theta_prime: Vec::new(),
            scratch_mat: Vec::new(),
            qr: ThinQr::default(),
            scratch_absorb: Vec::new(),
        }
    }
}

impl MpsBackend {
    /// Backend under the process default [`Limits`](crate::Limits).
    pub fn new(seed: u64, max_bond_dim: usize) -> Self {
        Self::with_limits(seed, max_bond_dim, &ResolvedLimits::process_default())
    }

    pub(crate) fn with_limits(seed: u64, max_bond_dim: usize, limits: &ResolvedLimits) -> Self {
        Self {
            num_qubits: 0,
            max_bond_dim,
            svd_epsilon: DEFAULT_SVD_EPSILON,
            sites: Vec::new(),
            logical_to_site: Vec::new(),
            site_to_logical: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            truncation_discarded: 0.0,
            center: None,
            bond_high_water: 1,
            #[cfg(test)]
            center_steps: 0,
            #[cfg(test)]
            qr_discarded: 0.0,
            workspace_cap: limits.mps_workspace_elements(),
            limits: *limits,
            scratch_theta: Vec::new(),
            scratch_right_t: Vec::new(),
            scratch_theta_prime: Vec::new(),
            scratch_mat: Vec::new(),
            qr: ThinQr::default(),
            scratch_absorb: Vec::new(),
        }
    }

    /// Create an MPS backend that keeps every non-zero singular value.
    pub(crate) fn new_exact(seed: u64) -> Self {
        let mut backend = Self::new(seed, usize::MAX);
        backend.svd_epsilon = 0.0;
        backend
    }

    /// Set the relative singular-value truncation threshold: each SVD drops
    /// values at or below `epsilon` times the largest, booked in
    /// [`Self::truncation_discarded`].
    ///
    /// The default 1e-12 keeps every numerically meaningful value. A larger
    /// threshold trades fidelity for bond dimension, and at a fixed cap can
    /// lower the total discarded weight by cutting low-weight Schmidt tails
    /// before the cap discards real weight.
    ///
    /// # Panics
    /// Panics unless `0 <= epsilon < 1`.
    pub fn set_svd_epsilon(&mut self, epsilon: f64) {
        assert!(
            (0.0..1.0).contains(&epsilon),
            "svd epsilon must lie in [0, 1)"
        );
        self.svd_epsilon = epsilon;
    }

    /// Zero the cumulative SVD-truncation weight. [`Backend::init`] does this,
    /// so a caller needs it only to scope the total to a sequence shorter than a
    /// run: the CAMPS T-gate path, which must reject silent truncation, brackets
    /// its sequence with this and [`Self::truncation_discarded`].
    ///
    /// [`Backend::init`]: crate::backend::Backend::init
    pub fn reset_truncation_tracking(&mut self) {
        self.truncation_discarded = 0.0;
    }

    /// Cumulative fraction of the state's squared weight discarded by SVD
    /// truncation since [`Backend::init`] or the last
    /// [`Self::reset_truncation_tracking`].
    ///
    /// The unit is squared: a cut that moves the state by `d` in the 2-norm
    /// books `d * d`, so a total of 1e-24 answers for a distance of 1e-12 and
    /// for an infidelity of 1e-24. Comparing it against a distance reads
    /// twelve orders too small.
    ///
    /// At the default threshold a meaningful value means the bond cap discarded
    /// real weight; after [`Self::set_svd_epsilon`] both sources contribute.
    ///
    /// A cut that can lose weight takes the orthogonality center onto its own
    /// sites first, so what it books is the error it made rather than a figure
    /// against a non-orthogonal environment: every cut that can reach the bond
    /// cap, every threshold cut once the chain's bonds have reached
    /// `GAUGE_RANK`, under that a two-site or block threshold cut that would
    /// drop a value above `SVD_RESOLUTION` of its largest, and every cut on a
    /// chain already carrying a center, which a measurement leaves behind on
    /// any chain whose cuts can lose weight at all. The center is parked by
    /// [`svd()`] rather than by the exact walk, so the environment is
    /// orthonormal to that factorization's isometry.
    ///
    /// The total sums one relative discard per SVD rather than measuring the
    /// final state, so a chain that truncates heavily can carry it past 1,
    /// where the fidelity it implies clamps to zero and certifies nothing.
    ///
    /// Truncation does not renormalize the chain. Every read rescales instead:
    /// expectation values, shot sampling, the probability terminal and
    /// [`Self::export_statevector`] all describe the normalized state.
    ///
    /// [`Backend::init`]: crate::backend::Backend::init
    pub fn truncation_discarded(&self) -> f64 {
        self.truncation_discarded
    }

    fn record_truncation(&mut self, singular_values: &[f64], chi_kept: usize) {
        if chi_kept >= singular_values.len() {
            return;
        }
        let total: f64 = singular_values.iter().map(|s| s * s).sum();
        if total <= 0.0 {
            return;
        }
        let discarded: f64 = singular_values[chi_kept..].iter().map(|s| s * s).sum();
        self.truncation_discarded += discarded / total;
    }

    /// Widest bond the chain holds now, 1 for a product state. The run's peak
    /// is `Backend::bond_report`.
    pub fn current_max_bond_dim(&self) -> usize {
        self.sites.iter().map(|s| s.bond_right).max().unwrap_or(1)
    }

    /// Restore the internal MPS site order to logical qubit order without
    /// changing the represented logical state.
    #[cfg(test)]
    pub(crate) fn canonicalize_logical_order(&mut self) -> Result<()> {
        let swap_mat = Gate::Swap.matrix_4x4();
        for target_site in 0..self.num_qubits {
            while self.site_to_logical[target_site] != target_site {
                let logical = target_site;
                let site = self.logical_to_site[logical];
                debug_assert!(site > target_site);
                self.apply_virtual_swap(site - 1, &swap_mat)?;
            }
        }
        Ok(())
    }

    /// Site currently hosting logical qubit `q`. SWAP routing changes this
    /// mapping: non-adjacent two-qubit gates run as chains of nearest-neighbour
    /// SWAPs and grow bond dimension in proportion to site distance. CAMPS
    /// anchor selection uses this to score candidate cascades.
    pub(crate) fn site_for_qubit(&self, q: usize) -> usize {
        self.logical_to_site[q]
    }

    /// Whether `P(q = |1⟩) = (1 − Re⟨Z_q⟩)/2` is below `tol`. For a qubit just
    /// disentangled by an OFD cascade, this is the qubit being in the pure `|0⟩`
    /// state on its site.
    pub(crate) fn is_qubit_in_zero_state(&self, q: usize, tol: f64) -> Result<bool> {
        let z = self.pauli_expectation(&[(q, MpsPauliAxis::Z)])?;
        let p_one = 0.5 * (1.0 - z.re);
        Ok(p_one < tol)
    }

    #[inline(always)]
    fn site_for_logical(&self, qubit: usize) -> usize {
        self.logical_to_site[qubit]
    }

    #[inline(always)]
    fn logical_for_site(&self, site: usize) -> usize {
        self.site_to_logical[site]
    }

    fn swap_layout_labels(&mut self, left_site: usize) {
        let left_logical = self.site_to_logical[left_site];
        let right_logical = self.site_to_logical[left_site + 1];
        self.site_to_logical.swap(left_site, left_site + 1);
        self.logical_to_site[left_logical] = left_site + 1;
        self.logical_to_site[right_logical] = left_site;
    }

    fn apply_virtual_swap(
        &mut self,
        left_site: usize,
        swap_mat: &[[Complex64; 4]; 4],
    ) -> Result<()> {
        self.apply_adjacent_two_qubit(swap_mat, left_site, true)?;
        self.swap_layout_labels(left_site);
        Ok(())
    }

    /// Apply a single-qubit unitary to `site` in place.
    ///
    /// A unitary leaves the gauge alone: `sum_i conj(u[i][j]) u[i][k]` is
    /// `delta_jk`, which collapses the site's isometry sum back to the one it
    /// had, so a recorded center survives untouched. An arbitrary matrix does
    /// not, which is why [`Backend::apply_1q_matrix`] drops the center first.
    #[inline(always)]
    fn apply_single_qubit_gate(&mut self, site: usize, u: &[[Complex64; 2]; 2]) {
        let t = &mut self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;
        let prepared = simd::PreparedGate1q::new(u);
        for alpha in 0..bl {
            let base = alpha * (2 * br);
            let (lo, hi) = t.data[base..base + 2 * br].split_at_mut(br);
            prepared.apply_slice_pairs(lo, hi);
        }
    }

    /// Walk the orthogonality center to `to`, one site at a time.
    ///
    /// Each step refactorizes the site being left and absorbs the triangular
    /// factor into the neighbour it moves toward, so the walk is exact: it
    /// consults neither `svd_epsilon` nor the bond cap and books nothing
    /// through [`Self::truncation_discarded`]. Every site it crosses comes out
    /// an isometry whether or not it was one, and sites beyond `to` keep the
    /// gauge they had, so a chain canonical about [`Self::center`] comes out
    /// canonical about `to`. A chain that was not is repaired only over the
    /// span walked, which is why an unrecorded gauge goes through
    /// [`Self::establish_center`] instead. Bonds never grow: a step caps the
    /// bond it rewrites at the rank of the site it factorizes.
    ///
    /// # Panics
    /// Panics unless `to` indexes a site and a center is recorded.
    fn move_center(&mut self, to: usize) {
        assert!(
            to < self.sites.len(),
            "orthogonality center {to} is outside a chain of {} sites",
            self.sites.len()
        );
        let mut at = self.center.expect("the walk starts from a recorded center");
        #[cfg(test)]
        {
            self.center_steps += at.abs_diff(to);
        }
        while at < to {
            self.shift_center_right(at);
            at += 1;
        }
        while at > to {
            self.shift_center_left(at);
            at -= 1;
        }
        self.center = Some(to);
    }

    /// Record a center at `to` on a chain whose gauge is unknown.
    ///
    /// The first leg runs to the end away from `to`, so it crosses every site
    /// and the center it lands on is real whatever the sites held before. The
    /// second leg is then an ordinary move. Costs `n - 1` steps plus the
    /// distance from that end.
    fn establish_center(&mut self, to: usize) {
        let last = self.sites.len() - 1;
        let (from, via) = if 2 * to <= last { (last, 0) } else { (0, last) };
        self.center = Some(from);
        self.move_center(via);
        self.move_center(to);
    }

    /// Put the center on the pair starting at `left_site` and name the side
    /// the two-site update should leave it on.
    ///
    /// The update writes an isometry into one site of the pair and the
    /// singular weight into the other, and which is which is a write
    /// convention rather than extra work. Leaving the center on the far side
    /// in the direction of travel carries it through a monotone run of
    /// adjacent gates at no extra factorization.
    fn prepare_center(&mut self, left_site: usize) -> WeightSide {
        match self.center {
            Some(at) if at <= left_site => {
                self.move_center(left_site);
                WeightSide::Right
            }
            Some(_) => {
                self.move_center(left_site + 1);
                WeightSide::Left
            }
            None => {
                self.establish_center(left_site);
                WeightSide::Right
            }
        }
    }

    /// Whether a cut of `rank` singular values takes the center before it is
    /// made: it can reach the bond cap, or a threshold is set and the chain's
    /// bonds have reached [`GAUGE_RANK`]. Under that a threshold cut is judged
    /// from its spectrum by [`discards_resolvable_weight`].
    fn cut_can_lose(&self, rank: u128) -> bool {
        rank > self.max_bond_dim as u128
            || (self.svd_epsilon > 0.0
                && (rank >= GAUGE_RANK as u128 || self.bond_high_water >= GAUGE_RANK))
    }

    /// Whether any cut of this chain can lose weight, and so whether a gauge
    /// is worth recording at all. An unbounded cap with a threshold of zero
    /// keeps every singular value of every cut, which is what the exact
    /// constructor asks for, and a center there would put a walk on every
    /// later two-qubit gate for an error no cut makes.
    fn cuts_can_lose_weight(&self) -> bool {
        self.svd_epsilon > 0.0 || self.max_bond_dim < usize::MAX
    }

    /// Drop the recorded center unless the write lands on it.
    ///
    /// A write that is not an isometry on `site` breaks the gauge the record
    /// claims for every site but the center, and the next consumer then
    /// truncates against a non-orthogonal environment with nothing looking
    /// wrong. Rebuilding on demand is cheaper than repairing blind.
    fn invalidate_center_unless(&mut self, site: usize) {
        if self.center != Some(site) {
            self.center = None;
        }
    }

    /// One rightward step of [`Self::move_center`]: `site` keeps `Q` across
    /// its right bond and `R` multiplies into the site to its right.
    fn shift_center_right(&mut self, site: usize) {
        let bl = self.sites[site].bond_left;
        let bond = self.sites[site].bond_right;
        let rows = bl * 2;

        let data = &self.sites[site].data;
        let mat = scratch_slice(&mut self.scratch_mat, rows * bond);
        for alpha in 0..bl {
            for i in 0..2 {
                for gamma in 0..bond {
                    mat[gamma * rows + alpha * 2 + i] = data[alpha * (2 * bond) + i * bond + gamma];
                }
            }
        }

        self.qr.factorize(mat, rows, bond);
        let chi = self.qr.rank;
        #[cfg(test)]
        {
            self.qr_discarded += self.qr.discarded;
        }

        let br = self.sites[site + 1].bond_right;
        let next = &self.sites[site + 1].data;
        let r = &self.qr.r;
        let absorb = scratch_slice(&mut self.scratch_absorb, chi * 2 * br);
        let width = 2 * br;
        let row = |gamma: usize, out: &mut [Complex64]| {
            out.fill(ZERO);
            for delta in 0..bond {
                let weight = r[gamma * bond + delta];
                for (o, x) in out
                    .iter_mut()
                    .zip(&next[delta * width..delta * width + width])
                {
                    *o += weight * x;
                }
            }
        };

        #[cfg(feature = "parallel")]
        if chi >= MIN_BOND_FOR_PAR {
            absorb
                .par_chunks_mut(width)
                .enumerate()
                .for_each(|(gamma, out)| row(gamma, out));
        } else {
            absorb
                .chunks_mut(width)
                .enumerate()
                .for_each(|(gamma, out)| row(gamma, out));
        }

        #[cfg(not(feature = "parallel"))]
        absorb
            .chunks_mut(width)
            .enumerate()
            .for_each(|(gamma, out)| row(gamma, out));

        let right = &mut self.sites[site + 1];
        right.bond_left = chi;
        right.data.clear();
        right.data.extend_from_slice(absorb);

        fill_isometry_site_data(&mut self.sites[site].data, &self.qr.q, rows, bl, chi);
        self.sites[site].bond_right = chi;
    }

    /// One leftward step of [`Self::move_center`], the same factorization on
    /// the conjugate transpose: `site` keeps the rows of `Q†` across its left
    /// bond and `R†` multiplies into the site to its left.
    fn shift_center_left(&mut self, site: usize) {
        let bond = self.sites[site].bond_left;
        let br = self.sites[site].bond_right;
        let rows = 2 * br;

        let data = &self.sites[site].data;
        let mat = scratch_slice(&mut self.scratch_mat, rows * bond);
        for (out, x) in mat.iter_mut().zip(data.iter()) {
            *out = x.conj();
        }

        self.qr.factorize(mat, rows, bond);
        let chi = self.qr.rank;
        #[cfg(test)]
        {
            self.qr_discarded += self.qr.discarded;
        }

        let bl = self.sites[site - 1].bond_left;
        let prev = &self.sites[site - 1].data;
        let r = &self.qr.r;
        let absorb = scratch_slice(&mut self.scratch_absorb, bl * 2 * chi);
        // Summed over the old bond innermost, so the site being read and the
        // row of `R` both run stride 1 and each output element takes a single
        // store.
        let row = |alpha: usize, out: &mut [Complex64]| {
            for i in 0..2 {
                let base = alpha * (2 * bond) + i * bond;
                let amps = &prev[base..base + bond];
                for gamma in 0..chi {
                    let mut acc = ZERO;
                    for (amp, factor) in amps.iter().zip(&r[gamma * bond..gamma * bond + bond]) {
                        acc += amp * factor.conj();
                    }
                    out[i * chi + gamma] = acc;
                }
            }
        };

        #[cfg(feature = "parallel")]
        if bl >= MIN_BOND_FOR_PAR {
            absorb
                .par_chunks_mut(2 * chi)
                .enumerate()
                .for_each(|(alpha, out)| row(alpha, out));
        } else {
            absorb
                .chunks_mut(2 * chi)
                .enumerate()
                .for_each(|(alpha, out)| row(alpha, out));
        }

        #[cfg(not(feature = "parallel"))]
        absorb
            .chunks_mut(2 * chi)
            .enumerate()
            .for_each(|(alpha, out)| row(alpha, out));

        let left = &mut self.sites[site - 1];
        left.bond_right = chi;
        left.data.clear();
        left.data.extend_from_slice(absorb);

        let t = &mut self.sites[site];
        t.bond_left = chi;
        t.data.clear();
        t.data
            .extend(self.qr.q[..rows * chi].iter().map(|x| x.conj()));
    }

    /// Largest deviation from the orthogonality condition over the sites away
    /// from `center`: a site left of it owes `A† A = I` summed over its left
    /// bond and physical index, a site right of it owes `A A† = I` summed over
    /// its physical index and right bond.
    #[cfg(test)]
    fn gauge_deviation(&self, center: usize) -> f64 {
        let mut worst = 0.0f64;
        for (site, t) in self.sites.iter().enumerate() {
            if site < center {
                for g in 0..t.bond_right {
                    for h in 0..t.bond_right {
                        let mut sum = ZERO;
                        for alpha in 0..t.bond_left {
                            for i in 0..2 {
                                sum +=
                                    t.data[t.idx(alpha, i, g)].conj() * t.data[t.idx(alpha, i, h)];
                            }
                        }
                        let target = if g == h { ONE } else { ZERO };
                        worst = worst.max((sum - target).norm());
                    }
                }
            } else if site > center {
                for a in 0..t.bond_left {
                    for b in 0..t.bond_left {
                        let mut sum = ZERO;
                        for i in 0..2 {
                            for beta in 0..t.bond_right {
                                sum += t.data[t.idx(a, i, beta)] * t.data[t.idx(b, i, beta)].conj();
                            }
                        }
                        let target = if a == b { ONE } else { ZERO };
                        worst = worst.max((sum - target).norm());
                    }
                }
            }
        }
        worst
    }

    #[cfg(test)]
    fn assert_gauge(&self, center: usize) {
        let worst = self.gauge_deviation(center);
        assert!(
            worst <= GAUGE_TOLERANCE,
            "gauge deviation {worst:.3e} about center {center} exceeds {GAUGE_TOLERANCE:.0e}"
        );
    }

    /// Apply a two-qubit gate to the pair at `left_site` and `left_site + 1`.
    ///
    /// The cut this makes is exact in the 2-norm only against an orthonormal
    /// environment, so the pair takes the orthogonality center first wherever
    /// the cut can lose anything: before the factorization when
    /// [`Self::cut_can_lose`] says so from the rank and the chain's bonds, and
    /// after it when the spectrum shows a threshold cut dropping a resolvable
    /// value, in which case the pair is cut again in gauge. A pair under both
    /// marks stays off the walk entirely. Maintaining a center already
    /// recorded costs one to three steps, well under rebuilding it, so a chain
    /// that has one keeps it on every pair.
    fn apply_adjacent_two_qubit(
        &mut self,
        gate: &[[Complex64; 4]; 4],
        left_site: usize,
        left_is_first_qubit: bool,
    ) -> Result<()> {
        if !self.cut_adjacent_pair(gate, left_site, left_is_first_qubit)? {
            self.establish_center(left_site);
            self.cut_adjacent_pair(gate, left_site, left_is_first_qubit)?;
        }
        Ok(())
    }

    /// One attempt at [`Self::apply_adjacent_two_qubit`]. Returns `false`,
    /// having written nothing, when the chain has no center and the cut would
    /// drop a value the factorization resolves.
    fn cut_adjacent_pair(
        &mut self,
        gate: &[[Complex64; 4]; 4],
        left_site: usize,
        left_is_first_qubit: bool,
    ) -> Result<bool> {
        let right_site = left_site + 1;
        let bl = self.sites[left_site].bond_left;
        let bond_mid = self.sites[left_site].bond_right;
        let br = self.sites[right_site].bond_right;

        // Live contraction buffers: theta, theta_prime, and the SVD input plus
        // its internals at 4*bl*br each (about five at peak), and the right
        // transpose at 2*bond_mid*br.
        let workspace = 20u128 * bl as u128 * br as u128 + 2u128 * bond_mid as u128 * br as u128;
        if workspace > self.workspace_cap {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "two-qubit gate application",
                workspace,
                &self.limits,
            ));
        }

        let maintained = self.center.is_some() || self.cut_can_lose(cut_rank(bl, bond_mid, br));
        let weight = if maintained {
            self.prepare_center(left_site)
        } else {
            WeightSide::Right
        };

        // A move rewrites the bonds it crosses at the rank it finds, so the
        // shapes are read again here.
        let bl = self.sites[left_site].bond_left;
        let bond_mid = self.sites[left_site].bond_right;
        let br = self.sites[right_site].bond_right;

        let g = if left_is_first_qubit {
            *gate
        } else {
            swap_gate_qubits(gate)
        };

        // 1. Contract: Θ[α, i, j, β] = Σ_γ A[k][α,i,γ] · A[k+1][γ,j,β]
        let left_data = &self.sites[left_site].data;
        let right_data = &self.sites[right_site].data;
        let chunk_size = 2 * 2 * br;
        let theta = scratch_slice(&mut self.scratch_theta, bl * chunk_size);

        // Transpose right: (bond_mid, 2, br) → (2, br, bond_mid) for sequential gamma access.
        let right_t = scratch_slice(&mut self.scratch_right_t, bond_mid * 2 * br);
        for gamma in 0..bond_mid {
            for j in 0..2usize {
                for beta in 0..br {
                    right_t[j * br * bond_mid + beta * bond_mid + gamma] =
                        right_data[gamma * (2 * br) + j * br + beta];
                }
            }
        }

        #[cfg(feature = "parallel")]
        if bl >= MIN_BOND_FOR_PAR {
            let right_ref = &right_t;
            theta
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(alpha, chunk)| {
                    for i in 0..2 {
                        for j in 0..2 {
                            for beta in 0..br {
                                let mut val = ZERO;
                                let rt_base = j * br * bond_mid + beta * bond_mid;
                                for gamma in 0..bond_mid {
                                    val += left_data[alpha * (2 * bond_mid) + i * bond_mid + gamma]
                                        * right_ref[rt_base + gamma];
                                }
                                chunk[i * (2 * br) + j * br + beta] = val;
                            }
                        }
                    }
                });
        } else {
            for alpha in 0..bl {
                for i in 0..2 {
                    for j in 0..2 {
                        for beta in 0..br {
                            let mut val = ZERO;
                            let rt_base = j * br * bond_mid + beta * bond_mid;
                            for gamma in 0..bond_mid {
                                val += left_data[alpha * (2 * bond_mid) + i * bond_mid + gamma]
                                    * right_t[rt_base + gamma];
                            }
                            theta[alpha * chunk_size + i * (2 * br) + j * br + beta] = val;
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        for alpha in 0..bl {
            for i in 0..2 {
                for j in 0..2 {
                    for beta in 0..br {
                        let mut val = ZERO;
                        let rt_base = j * br * bond_mid + beta * bond_mid;
                        for gamma in 0..bond_mid {
                            val += left_data[alpha * (2 * bond_mid) + i * bond_mid + gamma]
                                * right_t[rt_base + gamma];
                        }
                        theta[alpha * chunk_size + i * (2 * br) + j * br + beta] = val;
                    }
                }
            }
        }

        // 2. Apply gate: Θ'[α, i', j', β] = Σ_{i,j} G[i'2+j', i2+j] · Θ[α,i,j,β]
        let theta = &self.scratch_theta[..bl * chunk_size];
        let theta_prime = scratch_slice(&mut self.scratch_theta_prime, bl * chunk_size);

        #[cfg(feature = "parallel")]
        if bl >= MIN_BOND_FOR_PAR {
            let theta_ref = &theta;
            theta_prime
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(alpha, chunk)| {
                    for ip in 0..2 {
                        for jp in 0..2 {
                            for beta in 0..br {
                                let mut val = ZERO;
                                for i in 0..2 {
                                    for j in 0..2 {
                                        val += g[ip * 2 + jp][i * 2 + j]
                                            * theta_ref
                                                [alpha * chunk_size + i * (2 * br) + j * br + beta];
                                    }
                                }
                                chunk[ip * (2 * br) + jp * br + beta] = val;
                            }
                        }
                    }
                });
        } else {
            for alpha in 0..bl {
                for ip in 0..2 {
                    for jp in 0..2 {
                        for beta in 0..br {
                            let mut val = ZERO;
                            for i in 0..2 {
                                for j in 0..2 {
                                    val += g[ip * 2 + jp][i * 2 + j]
                                        * theta[alpha * chunk_size + i * (2 * br) + j * br + beta];
                                }
                            }
                            theta_prime[alpha * chunk_size + ip * (2 * br) + jp * br + beta] = val;
                        }
                    }
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        for alpha in 0..bl {
            for ip in 0..2 {
                for jp in 0..2 {
                    for beta in 0..br {
                        let mut val = ZERO;
                        for i in 0..2 {
                            for j in 0..2 {
                                val += g[ip * 2 + jp][i * 2 + j]
                                    * theta[alpha * chunk_size + i * (2 * br) + j * br + beta];
                            }
                        }
                        theta_prime[alpha * chunk_size + ip * (2 * br) + jp * br + beta] = val;
                    }
                }
            }
        }

        // 3. Reshape to matrix M: rows = bl*2, cols = 2*br
        //    M[(α*2+i), (j*br+β)] = Θ'[α, i, j, β]
        let rows = bl * 2;
        let cols = 2 * br;
        let theta_prime = &self.scratch_theta_prime[..bl * chunk_size];
        let mat = scratch_slice(&mut self.scratch_mat, rows * cols);
        for alpha in 0..bl {
            for i in 0..2 {
                for j in 0..2 {
                    for beta in 0..br {
                        let r = alpha * 2 + i;
                        let c = j * br + beta;
                        mat[c * rows + r] =
                            theta_prime[alpha * (2 * 2 * br) + i * (2 * br) + j * br + beta];
                    }
                }
            }
        }

        let svd_result = svd(mat, rows, cols);
        let chi_new = truncated_svd_rank(&svd_result.s, self.svd_epsilon, self.max_bond_dim);
        if !maintained && discards_resolvable_weight(&svd_result.s, chi_new) {
            return Ok(false);
        }
        self.record_truncation(&svd_result.s, chi_new);

        // A[k] has shape (bl, 2, chi_new) and A[k+1] shape (chi_new, 2, br).
        // U is column-major, U[col * rows + row]; V† is row-major,
        // Vt[row * cols + col]. Whichever side takes diag(S) carries the
        // weight, and the other comes out an isometry. Both sites are written
        // back through the buffers they already own, so a chain at a steady
        // bond allocates nothing per gate.
        let mut left_data = std::mem::take(&mut self.sites[left_site].data);
        let mut right_data = std::mem::take(&mut self.sites[right_site].data);
        match weight {
            WeightSide::Right => {
                fill_isometry_site_data(
                    &mut left_data,
                    &svd_result.u,
                    svd_result.u_rows,
                    bl,
                    chi_new,
                );
                fill_scaled_vt_data(&mut right_data, &svd_result, chi_new, 2, br);
            }
            WeightSide::Left => {
                fill_scaled_u_site_data(&mut left_data, &svd_result, bl, chi_new);
                fill_vt_site_data(&mut right_data, &svd_result, chi_new, 2, br);
            }
        }

        self.sites[left_site] = SiteTensor {
            bond_left: bl,
            bond_right: chi_new,
            data: left_data,
        };
        self.sites[right_site] = SiteTensor {
            bond_left: chi_new,
            bond_right: br,
            data: right_data,
        };
        self.bond_high_water = self.bond_high_water.max(chi_new);
        if maintained {
            self.center = Some(match weight {
                WeightSide::Right => right_site,
                WeightSide::Left => left_site,
            });
        }
        Ok(true)
    }

    fn apply_two_qubit_gate(
        &mut self,
        gate: &[[Complex64; 4]; 4],
        q0: usize,
        q1: usize,
    ) -> Result<()> {
        let p0 = self.site_for_logical(q0);
        let p1 = self.site_for_logical(q1);
        let k = p0.min(p1);
        let m = p0.max(p1);
        let left_is_first = p0 < p1;

        if m - k == 1 {
            self.apply_adjacent_two_qubit(gate, k, left_is_first)?;
        } else {
            let swap_mat = Gate::Swap.matrix_4x4();
            for s in (k + 1..m).rev() {
                self.apply_virtual_swap(s, &swap_mat)?;
            }
            self.apply_adjacent_two_qubit(gate, k, left_is_first)?;
        }
        Ok(())
    }

    /// Apply `instructions` in order, except that a run of two-qubit gates on
    /// adjacent, disjoint site pairs with increasing left sites enters from
    /// whichever end the recorded center is nearer. The gates of such a run
    /// commute, and the walk crosses the run once from either end, so the
    /// choice moves the entry cost and nothing else: on a brick layer that is
    /// one step against the width of the chain. A run is counted in gate
    /// entries by [`pair_entries`], so it spans instruction boundaries and
    /// reaches into `Multi2q` and `BatchRzz` lists, which is how fusion
    /// delivers a brick layer; an instruction without entries, a measurement
    /// or a barrier included, ends the run where it stands.
    fn apply_by_center(&mut self, instructions: &[Instruction]) -> Result<()> {
        let (mut i, mut k) = (0, 0);
        while i < instructions.len() {
            if pair_entries(&instructions[i]) == 0 {
                self.apply(&instructions[i])?;
                i += 1;
                continue;
            }
            let (mut ri, mut rk) = (i, k);
            let mut len = 0;
            let (mut first, mut last) = (0, 0);
            'scan: while ri < instructions.len() {
                let entries = pair_entries(&instructions[ri]);
                if entries == 0 {
                    break;
                }
                while rk < entries {
                    let (q0, q1) = pair_entry(&instructions[ri], rk);
                    let p0 = self.site_for_logical(q0);
                    let p1 = self.site_for_logical(q1);
                    let left = p0.min(p1);
                    if p0.abs_diff(p1) != 1 || (len > 0 && left < last + 2) {
                        break 'scan;
                    }
                    if len == 0 {
                        first = left;
                    }
                    last = left;
                    len += 1;
                    rk += 1;
                }
                ri += 1;
                rk = 0;
            }
            if len == 0 {
                self.apply_pair_entry(&instructions[i], k)?;
                k += 1;
                if k == pair_entries(&instructions[i]) {
                    i += 1;
                    k = 0;
                }
                continue;
            }
            if len > 1 && self.entry_steps(last) < self.entry_steps(first) {
                let (mut ii, mut kk) = (ri, rk);
                while (ii, kk) != (i, k) {
                    if kk == 0 {
                        ii -= 1;
                        kk = pair_entries(&instructions[ii]);
                    }
                    kk -= 1;
                    self.apply_pair_entry(&instructions[ii], kk)?;
                }
            } else {
                let (mut ii, mut kk) = (i, k);
                while (ii, kk) != (ri, rk) {
                    self.apply_pair_entry(&instructions[ii], kk)?;
                    kk += 1;
                    if kk == pair_entries(&instructions[ii]) {
                        ii += 1;
                        kk = 0;
                    }
                }
            }
            i = ri;
            k = rk;
        }
        Ok(())
    }

    /// Apply entry `k` of `instruction`, one of the [`pair_entries`] it owes.
    fn apply_pair_entry(&mut self, instruction: &Instruction, k: usize) -> Result<()> {
        match instruction {
            Instruction::Gate {
                gate: Gate::Multi2q(data),
                ..
            } => {
                let (q0, q1, mat) = &data.gates[k];
                self.apply_two_qubit_gate(mat, *q0, *q1)
            }
            Instruction::Gate {
                gate: Gate::BatchRzz(data),
                ..
            } => {
                let (q0, q1, theta) = data.edges[k];
                self.apply_two_qubit_gate(&Gate::Rzz(theta).matrix_4x4(), q0, q1)
            }
            Instruction::Gate {
                gate: Gate::Fused2q(mat),
                targets,
            } => self.apply_two_qubit_gate(mat, targets[0], targets[1]),
            Instruction::Gate { gate, targets } => {
                self.apply_two_qubit_gate(&gate.matrix_4x4(), targets[0], targets[1])
            }
            _ => unreachable!("an instruction without pair entries"),
        }
    }

    /// Steps [`Self::prepare_center`] walks to reach the pair at `left_site`.
    fn entry_steps(&self, left_site: usize) -> usize {
        match self.center {
            Some(at) if at <= left_site => left_site - at,
            Some(at) => at - left_site - 1,
            None => 0,
        }
    }

    /// Apply BatchPhase via bubble routing: sweep control toward targets,
    /// applying each CU-phase+SWAP as a single 2-site operation. O(k) tensor
    /// ops instead of O(k²) for k non-adjacent phases.
    fn apply_batch_phase_bubble(
        &mut self,
        control: usize,
        phases: &[(usize, Complex64)],
    ) -> Result<()> {
        if phases.is_empty() {
            return Ok(());
        }

        let mut right: Vec<(usize, Complex64)> = Vec::new();
        let mut left: Vec<(usize, Complex64)> = Vec::new();
        for &(target, phase) in phases {
            if target > control {
                right.push((target, phase));
            } else if target < control {
                left.push((target, phase));
            }
        }
        right.sort_by_key(|&(t, _)| t);
        left.sort_by_key(|&(t, _)| std::cmp::Reverse(t));

        let swap_mat = Gate::Swap.matrix_4x4();

        // Right sweep: bubble control rightward
        if !right.is_empty() {
            let mut cur_pos = control;
            let last_idx = right.len() - 1;
            for (idx, &(target, phase)) in right.iter().enumerate() {
                while cur_pos + 1 < target {
                    self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true)?;
                    cur_pos += 1;
                }
                if idx < last_idx {
                    let combined = cu_phase_swap_matrix(phase);
                    self.apply_adjacent_two_qubit(&combined, cur_pos, true)?;
                    cur_pos += 1;
                } else {
                    let mat = [[ONE, ZERO], [ZERO, phase]];
                    let g = crate::gates::cu_matrix_4x4(&mat);
                    self.apply_adjacent_two_qubit(&g, cur_pos, true)?;
                }
            }
            while cur_pos > control {
                cur_pos -= 1;
                self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true)?;
            }
        }

        // Left sweep: bubble control leftward
        if !left.is_empty() {
            let mut cur_pos = control;
            let last_idx = left.len() - 1;
            for (idx, &(target, phase)) in left.iter().enumerate() {
                while cur_pos > target + 1 {
                    cur_pos -= 1;
                    self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true)?;
                }
                if idx < last_idx {
                    let combined = cu_phase_swap_matrix(phase);
                    self.apply_adjacent_two_qubit(&combined, cur_pos - 1, true)?;
                    cur_pos -= 1;
                } else {
                    let mat = [[ONE, ZERO], [ZERO, phase]];
                    let g = crate::gates::cu_matrix_4x4(&mat);
                    self.apply_adjacent_two_qubit(&g, cur_pos - 1, false)?;
                }
            }
            while cur_pos < control {
                self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true)?;
                cur_pos += 1;
            }
        }
        Ok(())
    }

    fn apply_reset(&mut self, qubit: usize) -> Result<()> {
        let weights = self.born_weights(qubit)?;
        let outcome = self.draw_outcome(weights);
        self.collapse_site_to_zero(qubit, outcome, weights);
        Ok(())
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) -> Result<()> {
        let weights = self.born_weights(qubit)?;
        let measured = self.draw_outcome(weights);
        self.classical_bits[classical_bit] = measured == 1;
        self.project_site_outcome(qubit, measured, weights);
        Ok(())
    }

    /// Draw an outcome from Born weights, which sum to the squared norm the
    /// truncations so far have left rather than to one.
    fn draw_outcome(&mut self, weights: [f64; 2]) -> usize {
        let total = weights[0] + weights[1];
        usize::from(self.rng.random::<f64>() * total < weights[1])
    }

    /// Unnormalized weight of each outcome at `site`, taken with the
    /// orthogonality center walked onto it.
    ///
    /// Every other site is then an isometry and both environments contract to
    /// identities, which leaves the weights as the site's own row sums: `O(χ²)`
    /// against the `O(n·χ⁴)` of the two environment sweeps a site off the
    /// center needs. The projection that follows writes `site` alone, the one
    /// site under no isometry claim, so the center survives it and the next
    /// measurement pays the distance between the two sites rather than a
    /// rebuild from the end of the chain. A chain
    /// [`Self::cuts_can_lose_weight`] rules out keeps no record past the
    /// projection: the walk is still the cheaper way to the weights, and the
    /// gauge would only tax the gates that follow.
    ///
    /// Errors when the weights sum to zero or to a value that is not finite:
    /// the chain then holds no state a projection could rescale.
    fn born_weights(&mut self, site: usize) -> Result<[f64; 2]> {
        match self.center {
            Some(_) => self.move_center(site),
            None => self.establish_center(site),
        }

        let t = &self.sites[site];
        let br = t.bond_right;
        let mut prob = [0.0f64; 2];
        for alpha in 0..t.bond_left {
            let row = &t.data[alpha * (2 * br)..(alpha + 1) * (2 * br)];
            for (outcome, weight) in prob.iter_mut().enumerate() {
                *weight += row[outcome * br..(outcome + 1) * br]
                    .iter()
                    .map(|x| x.norm_sqr())
                    .sum::<f64>();
            }
        }
        require_weight(prob)?;
        Ok(prob)
    }

    /// Project onto `outcome`, rescale the chain to unit norm and return the
    /// outcome's share of the chain's weight, zero when that share is at or
    /// under [`NORM_CLAMP_MIN`].
    ///
    /// Takes both Born weights from the caller, which read them off the center
    /// in [`Self::born_weights`]. The share rather than the weight decides,
    /// since truncation leaves the chain's own weight anywhere under one. The
    /// write lands on the center, so the gauge comes through it intact.
    fn project_site_outcome(&mut self, site: usize, outcome: usize, weights: [f64; 2]) -> f64 {
        debug_assert_eq!(self.center, Some(site), "a projection off the center");
        if !self.cuts_can_lose_weight() {
            self.center = None;
        }
        let prob = weights[outcome];
        let share = prob / (weights[0] + weights[1]);
        if share <= NORM_CLAMP_MIN {
            return 0.0;
        }
        let inv_sqrt_prob = 1.0 / prob.sqrt();
        let scale = Complex64::new(inv_sqrt_prob, 0.0);
        let other = 1 - outcome;

        let t = &mut self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;
        for alpha in 0..bl {
            for beta in 0..br {
                let idx_m = alpha * (2 * br) + outcome * br + beta;
                let idx_o = alpha * (2 * br) + other * br + beta;
                t.data[idx_m] *= scale;
                t.data[idx_o] = ZERO;
            }
        }
        share
    }

    /// Collapse a site onto `outcome` and return it to the zero state.
    ///
    /// Writing the surviving component straight into the zero slot folds in the
    /// X that a trajectory reset would otherwise apply after projecting onto
    /// outcome 1. Takes both Born weights and judges the outcome by its share,
    /// as [`Self::project_site_outcome`] does.
    fn collapse_site_to_zero(&mut self, site: usize, outcome: usize, weights: [f64; 2]) {
        debug_assert_eq!(self.center, Some(site), "a reset off the center");
        if !self.cuts_can_lose_weight() {
            self.center = None;
        }
        let prob = weights[outcome];
        if prob / (weights[0] + weights[1]) <= NORM_CLAMP_MIN {
            return;
        }
        let scale = Complex64::new(1.0 / prob.sqrt(), 0.0);
        let t = &mut self.sites[site];
        let br = t.bond_right;
        for alpha in 0..t.bond_left {
            let base = alpha * (2 * br);
            for beta in 0..br {
                let zero_idx = base + beta;
                t.data[zero_idx] = t.data[base + outcome * br + beta] * scale;
                t.data[zero_idx + br] = ZERO;
            }
        }
    }

    /// Project a logical qubit onto a requested computational-basis outcome.
    ///
    /// Returns the branch-local Born probability. A zero return means the
    /// requested branch should be discarded by the caller. Errors as
    /// [`Self::born_weights`] does.
    pub(crate) fn project_z_outcome(&mut self, qubit: usize, outcome: bool) -> Result<f64> {
        let site = self.site_for_logical(qubit);
        let outcome = usize::from(outcome);
        let weights = self.born_weights(site)?;
        Ok(self.project_site_outcome(site, outcome, weights))
    }

    /// Sites left of a recorded center are left isometries and sites right of
    /// it right isometries, so each environment starts at the center or at the
    /// site itself instead of at the chain end, and the center's own site needs
    /// neither.
    fn reduced_density_site(&self, site: usize) -> [[Complex64; 2]; 2] {
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut rho = [[ZERO; 2]; 2];

        if self.center == Some(site) {
            for alpha in 0..bl {
                for beta in 0..br {
                    let a = [t.data[t.idx(alpha, 0, beta)], t.data[t.idx(alpha, 1, beta)]];
                    for (row, rho_row) in rho.iter_mut().enumerate() {
                        for (col, rho_cell) in rho_row.iter_mut().enumerate() {
                            *rho_cell += a[row] * a[col].conj();
                        }
                    }
                }
            }
            crate::backend::reduced_density::normalize_trace(rho.as_flattened_mut(), 2);
            return rho;
        }

        let (from, to) = match self.center {
            Some(c) if c < site => (c, site),
            Some(c) => (site, c),
            None => (0, self.num_qubits - 1),
        };
        let l_env = self.compute_left_env(from, site);
        let r_env = self.compute_right_env(site, to);

        for (row, rho_row) in rho.iter_mut().enumerate() {
            for (col, rho_cell) in rho_row.iter_mut().enumerate() {
                let mut val = ZERO;
                for alpha in 0..bl {
                    for alpha_p in 0..bl {
                        let l_val = l_env[alpha * bl + alpha_p];
                        if l_val == ZERO {
                            continue;
                        }
                        for beta in 0..br {
                            for beta_p in 0..br {
                                let r_val = r_env[beta * br + beta_p];
                                if r_val == ZERO {
                                    continue;
                                }
                                val += l_val
                                    * t.data[t.idx(alpha, row, beta)]
                                    * t.data[t.idx(alpha_p, col, beta_p)].conj()
                                    * r_val;
                            }
                        }
                    }
                }
                *rho_cell = val;
            }
        }

        crate::backend::reduced_density::normalize_trace(rho.as_flattened_mut(), 2);
        rho
    }

    /// Schmidt values across the bond between sites `bond` and `bond + 1`.
    ///
    /// With the center on either site of the bond, every other site is an
    /// isometry, so the singular values of the center site read as a matrix
    /// across that bond are the spectrum of the whole state. The site's data
    /// is already that matrix transposed in column-major order, and the
    /// transpose has the same singular values, so nothing is reshaped. The
    /// center walks to whichever site of the bond is nearer, or is
    /// established at the one nearer its end of the chain when none is
    /// recorded; the walk is exact and books nothing.
    fn schmidt_values_at_bond(&mut self, bond: usize) -> Vec<f64> {
        let last = self.sites.len() - 1;
        let site = match self.center {
            Some(at) if at > bond => bond + 1,
            Some(_) => bond,
            None if 2 * bond < last => bond,
            None => bond + 1,
        };
        match self.center {
            Some(_) => self.move_center(site),
            None => self.establish_center(site),
        }
        let t = &self.sites[site];
        let (rows, cols) = if site == bond {
            (t.bond_right, 2 * t.bond_left)
        } else {
            (2 * t.bond_right, t.bond_left)
        };
        schmidt::finish_schmidt_values(svd(&t.data, rows, cols).s)
    }

    /// Schmidt values from the eigenvalues of the reduced density matrix over
    /// the chain sites `sites`, ascending and at most half the chain. The
    /// center moves to the nearest site of the span so both environments are
    /// identities.
    fn schmidt_values_by_reduced_density(&mut self, sites: &[usize]) -> Result<Vec<f64>> {
        let what = "Schmidt values across a cut that is not contiguous in chain order";
        let k = sites.len();
        schmidt::check_schmidt_side(self.name(), what, k, &self.limits)?;
        let (first, last) = (sites[0], sites[k - 1]);
        let chi = self.sites[first..=last]
            .iter()
            .fold(1usize, |chi, t| chi.max(t.bond_left).max(t.bond_right));
        check_contraction_workspace(k, chi, &self.limits, what)?;
        let end = self.sites.len() - 1;
        match self.center {
            Some(at) => self.move_center(at.clamp(first, last)),
            None if first <= end - last => self.establish_center(first),
            None => self.establish_center(last),
        }
        let rho = self.reduced_density_sites(sites);
        Ok(schmidt::schmidt_values_from_density(&rho, 1 << k))
    }

    /// Reduced density matrix over the chain sites `sites` (ascending), row
    /// major `2^k x 2^k` with the leftmost site in the high bit of both
    /// indices. Requires the center inside `sites[0]..=sites[k - 1]`: the
    /// sites on either side of that span then contract to identities, and the
    /// sweep over the span carries a `(ket, bra, bond, bond)` transfer object
    /// that grows fourfold at each site it keeps open.
    fn reduced_density_sites(&self, sites: &[usize]) -> Vec<Complex64> {
        let (first, last) = (sites[0], sites[sites.len() - 1]);
        let bl = self.sites[first].bond_left;
        let mut env: Vec<Complex64> = (0..bl * bl)
            .map(|x| if x / bl == x % bl { ONE } else { ZERO })
            .collect();
        let mut dim = 1usize;
        let mut kept = sites.iter().peekable();
        for site in first..=last {
            let keep = kept.next_if_eq(&&site).is_some();
            let t = &self.sites[site];
            let (bl, br) = (t.bond_left, t.bond_right);
            let width = 2 * br;
            // half[o, a', i, b] = sum_a env[o, a, a'] A[a, i, b]
            let mut half = vec![ZERO; dim * dim * bl * width];
            for (o, rows) in half.chunks_exact_mut(bl * width).enumerate() {
                for (ap, out) in rows.chunks_exact_mut(width).enumerate() {
                    for a in 0..bl {
                        let e = env[(o * bl + a) * bl + ap];
                        for (x, amp) in out.iter_mut().zip(&t.data[a * width..(a + 1) * width]) {
                            *x += e * amp;
                        }
                    }
                }
            }
            // next[o', b, b'] = sum_{a', i, i'} half[o, a', i, b] conj(A[a', i', b'])
            // with `i == i'` traced, or `i` and `i'` appended to `o` as the
            // low bit of the ket and bra when the site is kept.
            let next_dim = if keep { 2 * dim } else { dim };
            let mut next = vec![ZERO; next_dim * next_dim * br * br];
            for (o, rows) in half.chunks_exact(bl * width).enumerate() {
                let (ket, bra) = (o / dim, o % dim);
                for (ap, row) in rows.chunks_exact(width).enumerate() {
                    for i in 0..2 {
                        let ket_row = &row[i * br..(i + 1) * br];
                        for ip in 0..2 {
                            if !keep && ip != i {
                                continue;
                            }
                            let o_next = if keep {
                                (2 * ket + i) * next_dim + 2 * bra + ip
                            } else {
                                o
                            };
                            let bra_row = &t.data[ap * width + ip * br..ap * width + (ip + 1) * br];
                            let dst = &mut next[o_next * br * br..(o_next + 1) * br * br];
                            for (b, &x) in ket_row.iter().enumerate() {
                                for (y, d) in bra_row.iter().zip(&mut dst[b * br..(b + 1) * br]) {
                                    *d += x * y.conj();
                                }
                            }
                        }
                    }
                }
            }
            env = next;
            dim = next_dim;
        }
        let br = self.sites[last].bond_right;
        env.chunks_exact(br * br)
            .map(|block| (0..br).map(|b| block[b * br + b]).sum())
            .collect()
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Rzz(_) => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cx => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cz => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Swap => {
                let g = Gate::Swap.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cu(_) => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                let n = num_ctrl + 1;
                self.check_n_qubit_gate_matrix(n)?;
                let all_qubits: Vec<usize> = targets
                    .iter()
                    .map(|&qubit| self.site_for_logical(qubit))
                    .collect();
                let dim = 1usize << n;
                let gate_mat = mcu_matrix(num_ctrl, &data.mat);
                self.apply_n_qubit_gate(&gate_mat, dim, &all_qubits)?;
            }
            Gate::Unitary(data) => {
                let n = data.num_qubits();
                self.check_n_qubit_gate_matrix(n)?;
                let sites: Vec<usize> = targets
                    .iter()
                    .map(|&qubit| self.site_for_logical(qubit))
                    .collect();
                self.apply_n_qubit_gate(data.matrix(), 1 << n, &sites)?;
            }
            Gate::BatchPhase(data) => {
                if let [(target, phase)] = data.phases[..] {
                    let g = crate::gates::cu_matrix_4x4(&[[ONE, ZERO], [ZERO, phase]]);
                    self.apply_two_qubit_gate(&g, targets[0], target)?;
                } else {
                    let control = self.site_for_logical(targets[0]);
                    let phases: Vec<(usize, Complex64)> = data
                        .phases
                        .iter()
                        .map(|&(qubit, phase)| (self.site_for_logical(qubit), phase))
                        .collect();
                    self.apply_batch_phase_bubble(control, &phases)?;
                }
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    let g = Gate::Rzz(theta).matrix_4x4();
                    self.apply_two_qubit_gate(&g, q0, q1)?;
                }
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    if let Some((q, mat)) = entry.as_1q_matrix() {
                        self.apply_single_qubit_gate(self.site_for_logical(q), &mat);
                    } else if let Some((q0, q1, mat)) = entry.as_2q_matrix() {
                        self.apply_two_qubit_gate(&mat, q0, q1)?;
                    }
                }
            }
            Gate::MultiFused(data) => {
                for &(target, ref mat) in &data.gates {
                    self.apply_single_qubit_gate(self.site_for_logical(target), mat);
                }
            }
            Gate::Fused2q(mat) => {
                self.apply_two_qubit_gate(mat, targets[0], targets[1])?;
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_two_qubit_gate(mat, q0, q1)?;
                }
            }
            single_qubit => {
                let mat = single_qubit.matrix_2x2();
                self.apply_single_qubit_gate(self.site_for_logical(targets[0]), &mat);
            }
        }
        Ok(())
    }
}

/// Two-qubit gate entries `instruction` contributes to a run of
/// [`MpsBackend::apply_by_center`]: one for a gate the adjacent kernel takes
/// whole, one per entry of a `Multi2q` or `BatchRzz`, none for anything else.
fn pair_entries(instruction: &Instruction) -> usize {
    match instruction {
        Instruction::Gate { gate, .. } => match gate {
            Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap | Gate::Cu(_) | Gate::Fused2q(_) => 1,
            Gate::Multi2q(data) => data.gates.len(),
            Gate::BatchRzz(data) => data.edges.len(),
            _ => 0,
        },
        _ => 0,
    }
}

/// Logical qubits of entry `k` of `instruction`, one of the
/// [`pair_entries`] it owes.
fn pair_entry(instruction: &Instruction, k: usize) -> (usize, usize) {
    match instruction {
        Instruction::Gate {
            gate: Gate::Multi2q(data),
            ..
        } => {
            let (q0, q1, _) = &data.gates[k];
            (*q0, *q1)
        }
        Instruction::Gate {
            gate: Gate::BatchRzz(data),
            ..
        } => {
            let (q0, q1, _) = data.edges[k];
            (q0, q1)
        }
        Instruction::Gate { targets, .. } => (targets[0], targets[1]),
        _ => unreachable!("an instruction without pair entries"),
    }
}

/// Refuse a chain whose Born weights at one site sum to zero or to a value
/// that is not finite: no projection or sample can rescale it back to a state.
fn require_weight(weights: [f64; 2]) -> Result<()> {
    let total = weights[0] + weights[1];
    if total.is_finite() && total > 0.0 {
        return Ok(());
    }
    let message = if total.is_finite() {
        "MPS truncation discarded the whole state; raise max_bond_dim or lower svd_epsilon"
            .to_string()
    } else {
        format!(
            "MPS state is not finite (squared norm {total}); check the gate parameters, and \
             max_bond_dim and svd_epsilon if truncation ran"
        )
    };
    Err(crate::error::PrismError::InvalidParameter { message })
}

impl crate::backend::sealed::Sealed for MpsBackend {
    fn limits(&self) -> crate::limits::ResolvedLimits {
        self.limits
    }

    fn damping_probability(&mut self, qubit: usize) -> Option<Result<f64>> {
        let site = self.site_for_logical(qubit);
        Some(self.born_weights(site).map(|[w0, w1]| w1 / (w0 + w1)))
    }
}

impl Backend for MpsBackend {
    fn name(&self) -> &'static str {
        "mps"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::Mps
    }

    /// A run whose [`Self::truncation_discarded`] stays at or under
    /// `EXACT_DISCARD_BOUND` reports `Exact` at any bond cap; past it, the run
    /// reports `Approximate` with the bound that total implies.
    fn exactness(&self) -> crate::sim::Exactness {
        if self.truncation_discarded <= EXACT_DISCARD_BOUND {
            return crate::sim::Exactness::Exact;
        }
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some((1.0 - self.truncation_discarded).max(0.0)),
        }
    }

    fn bond_report(&self) -> Option<crate::sim::BondReport> {
        Some(crate::sim::BondReport {
            peak: self.bond_high_water,
            cap: self.max_bond_dim,
        })
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.truncation_discarded = 0.0;
        self.center = None;
        self.bond_high_water = 1;
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);
        self.sites = (0..num_qubits)
            .map(|_| SiteTensor::new_zero_state())
            .collect();
        self.logical_to_site = (0..num_qubits).collect();
        self.site_to_logical = (0..num_qubits).collect();
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => {
                let list = matches!(gate, Gate::Multi2q(_) | Gate::BatchRzz(_));
                if list && pair_entries(instruction) > 0 {
                    self.apply_by_center(std::slice::from_ref(instruction))?;
                } else {
                    self.dispatch_gate(gate, targets)?;
                }
            }
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(self.site_for_logical(*qubit), *classical_bit)?;
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(self.site_for_logical(*qubit))?;
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied("Mps", label));
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets)?;
                }
            }
            Instruction::Region(region) => self.apply_region(region)?,
        }
        Ok(())
    }

    fn apply_instructions(&mut self, instructions: &[Instruction]) -> Result<()> {
        #[cfg(feature = "parallel")]
        return crate::backend::hybrid::on_narrow_pool(true, || self.apply_by_center(instructions));
        #[cfg(not(feature = "parallel"))]
        self.apply_by_center(instructions)
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(self.site_for_logical(qubit))
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        let site = self.site_for_logical(qubit);
        self.invalidate_center_unless(site);
        self.apply_single_qubit_gate(site, matrix);
        Ok(())
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        Ok(self.reduced_density_site(self.site_for_logical(qubit)))
    }

    /// Contracted out of the chain rather than traced out of a vector: with the
    /// center inside the span, the sites on either side of it collapse to
    /// identities and one sweep carries a transfer object that grows fourfold
    /// per kept site. The sweep indexes by chain position, so a SWAP-routed
    /// layout is permuted back into the requested qubit order at the end.
    fn reduced_density_matrix(&mut self, subsystem: &[usize]) -> Result<Vec<Complex64>> {
        schmidt::validate_qubit_set(subsystem, self.num_qubits)?;
        let k = subsystem.len();
        let dim =
            crate::backend::reduced_density::reduced_density_side(self.name(), k, &self.limits)?;

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&i| self.site_for_logical(subsystem[i]));
        let sites: Vec<usize> = order
            .iter()
            .map(|&i| self.site_for_logical(subsystem[i]))
            .collect();

        let (first, last) = (sites[0], sites[k - 1]);
        let chi = self.sites[first..=last]
            .iter()
            .fold(1usize, |chi, t| chi.max(t.bond_left).max(t.bond_right));
        check_density_workspace(k, chi, &self.limits)?;

        let end = self.sites.len() - 1;
        match self.center {
            Some(at) => self.move_center(at.clamp(first, last)),
            None if first <= end - last => self.establish_center(first),
            None => self.establish_center(last),
        }
        let contracted = self.reduced_density_sites(&sites);

        // Bit `k - 1 - j` of a contracted index is the site at position `j`,
        // which hosts `subsystem[order[j]]` and so becomes bit `order[j]`.
        let permuted: Vec<usize> = (0..dim)
            .map(|t| {
                (0..k)
                    .filter(|&j| (t >> (k - 1 - j)) & 1 == 1)
                    .fold(0, |index, j| index | (1 << order[j]))
            })
            .collect();
        let mut rho = vec![ZERO; dim * dim];
        for (t, row) in contracted.chunks_exact(dim).enumerate() {
            for (tp, &amp) in row.iter().enumerate() {
                rho[permuted[t] * dim + permuted[tp]] = amp;
            }
        }
        crate::backend::reduced_density::normalize_trace(&mut rho, dim);
        Ok(rho)
    }

    fn supports_two_qubit_kraus(&self) -> bool {
        true
    }

    /// The one-SVD route when the subsystem is a run of sites at either end
    /// of the chain; otherwise the eigenvalues of the reduced density matrix
    /// over the smaller side of the cut, under the dense cap. The chain order
    /// is never re-sorted for it.
    fn schmidt_values(&mut self, subsystem: &[usize]) -> Result<Vec<f64>> {
        schmidt::validate_subsystem(subsystem, self.num_qubits)?;
        let n = self.num_qubits;
        let mut sites: Vec<usize> = subsystem
            .iter()
            .map(|&q| self.site_for_logical(q))
            .collect();
        sites.sort_unstable();
        let k = sites.len();
        if sites[k - 1] - sites[0] + 1 == k {
            if sites[0] == 0 {
                return Ok(self.schmidt_values_at_bond(k - 1));
            }
            if sites[k - 1] == n - 1 {
                return Ok(self.schmidt_values_at_bond(sites[0] - 1));
            }
        }
        if 2 * k > n {
            let mut inside = vec![false; n];
            for &site in &sites {
                inside[site] = true;
            }
            sites = (0..n).filter(|&site| !inside[site]).collect();
        }
        self.schmidt_values_by_reduced_density(&sites)
    }

    /// [`MpsBackend::inner_product`] when both chains carry the same site
    /// layout, `O(n chi^3)` and no width cap. The contraction pairs sites by
    /// chain position, so a chain whose virtual swaps left a different order
    /// takes the dense route instead.
    fn overlap_sq(&self, other: &dyn Backend) -> Result<f64> {
        if let Some(chain) = other
            .as_any()
            .and_then(|any| any.downcast_ref::<MpsBackend>())
            && chain.num_qubits == self.num_qubits
            && chain.logical_to_site == self.logical_to_site
        {
            let inner = self.inner_product(chain)?;
            let left = self.inner_product(self)?.re;
            let right = chain.inner_product(chain)?.re;
            return Ok(overlap::normalized(inner.norm_sqr(), left, right));
        }
        overlap::export_overlap_sq(
            self.name(),
            self.num_qubits(),
            || self.export_statevector(),
            other,
            &self.limits,
        )
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn set_classical_bit(&mut self, bit: usize, value: bool) -> Result<()> {
        self.classical_bits[bit] = value;
        Ok(())
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let dim = dense_probability_len(self.name(), self.num_qubits, &self.limits)?;
        let mut probs = Vec::new();
        reserve_dense_output(&mut probs, dim, self.name(), "probabilities")?;
        probs.resize(dim, 0.0);
        self.expand_dense(&mut probs, |amp| amp.norm_sqr());
        Ok(probs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Sequential conditional sampling over the chain: `O(n·χ²)` per shot after
    /// one `O(n·χ³)` sweep for the right environments.
    ///
    /// Each shot draws from its own ChaCha8 substream keyed on `(seed, shot)`
    /// (streams 2 and up: 0 is the backend's own draw stream, 1 the
    /// trajectory noise stream), so the sampled words are identical at any
    /// thread count and the shot loop parallelizes at or above
    /// `MIN_SHOTS_FOR_PAR`.
    ///
    /// Sites are visited left to right and each bit is recorded against the
    /// logical qubit currently hosted there, so a layout permuted by SWAP
    /// routing needs no canonicalization pass.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let n = self.num_qubits;
        let mut samples = BasisSamples::new(num_shots, n);
        if n == 0 {
            return Ok(samples);
        }

        let right = self.right_environments();
        let mut first = vec![ZERO; 2 * self.sites[0].bond_right];
        require_weight(self.site_conditional_weights(0, &[ONE], &right[0], &mut first))?;
        let max_bond = self
            .sites
            .iter()
            .map(|site| site.bond_left.max(site.bond_right))
            .max()
            .unwrap_or(1);

        let words_per_shot = samples.words_per_shot();
        let sample_shot =
            |shot: usize, shot_words: &mut [u64], left: &mut [Complex64], w: &mut [Complex64]| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                rng.set_stream(shot as u64 + 2);
                left[0] = ONE;
                left[1..].fill(ZERO);
                for (site, right_env) in right.iter().enumerate() {
                    let br = self.sites[site].bond_right;
                    let prob = self.site_conditional_weights(site, left, right_env, w);

                    let total = prob[0] + prob[1];
                    let prob_one = if total > 0.0 {
                        (prob[1] / total).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let outcome = usize::from(rng.random::<f64>() < prob_one);
                    if outcome == 1 {
                        let qubit = self.logical_for_site(site);
                        shot_words[qubit / 64] |= 1u64 << (qubit % 64);
                    }

                    let scale = 1.0 / prob[outcome].max(NORM_CLAMP_MIN).sqrt();
                    left[..br].copy_from_slice(&w[outcome * br..(outcome + 1) * br]);
                    for value in &mut left[..br] {
                        *value *= scale;
                    }
                }
            };

        #[cfg(feature = "parallel")]
        if num_shots >= MIN_SHOTS_FOR_PAR {
            samples
                .words_mut()
                .par_chunks_mut(words_per_shot)
                .enumerate()
                .for_each_init(
                    || (vec![ZERO; max_bond], vec![ZERO; 2 * max_bond]),
                    |(left, w), (shot, shot_words)| sample_shot(shot, shot_words, left, w),
                );
            return Ok(samples);
        }

        let mut left = vec![ZERO; max_bond];
        let mut w = vec![ZERO; 2 * max_bond];
        for shot in 0..num_shots {
            let start = shot * words_per_shot;
            sample_shot(
                shot,
                &mut samples.words_mut()[start..start + words_per_shot],
                &mut left,
                &mut w,
            );
        }
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// With two or more observables, identity environments are swept once in
    /// each direction and shared, so an observable contracts only the sites
    /// between its first and last
    /// non-identity factor: a single-qubit observable costs no environment
    /// step at all. Divided by `⟨ψ|ψ⟩` because a truncated chain is not exactly
    /// normalized.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        if self.num_qubits == 0 {
            return Ok(vec![1.0; observables.len()]);
        }
        if observables.len() < 2 {
            // One observable costs the same two sweeps either way, and keeping
            // the plain contraction here leaves a route that shares no code
            // with the windowed one for a test to compare against.
            let norm = self.pauli_expectation(&[])?.re;
            return observables
                .iter()
                .map(|observable| {
                    let value = self.pauli_expectation(&mps_pauli_factors(observable))?;
                    Ok(if norm == 0.0 { 0.0 } else { value.re / norm })
                })
                .collect();
        }
        self.windowed_pauli_expectations(observables)
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(
            self.name(),
            "statevector export",
            self.num_qubits,
            &self.limits,
        )?;
        let mut state = Vec::new();
        reserve_dense_output(&mut state, dim, self.name(), "statevector export")?;
        state.resize(dim, ZERO);
        self.expand_dense(&mut state, |amp| amp);
        Ok(state)
    }
}

#[cfg(test)]
#[path = "mps_tests.rs"]
mod tests;
