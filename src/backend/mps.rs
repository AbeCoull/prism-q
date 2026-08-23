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
//! - SVD truncation uses relative tolerance (default 1e-12) AND bond dim cap.
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
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_probability_len, dense_statevector_len,
    reserve_dense_output, simd,
};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::Gate;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const DEFAULT_SVD_EPSILON: f64 = 1e-12;
const MAX_SVD_SWEEPS: usize = 100;
const SVD_CONVERGENCE: f64 = 1e-14;
#[cfg(feature = "parallel")]
const MIN_BOND_FOR_PAR: usize = 32;
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

fn cx_matrix_4x4() -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
}

fn cz_matrix_4x4() -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    let m = Complex64::new(-1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, m]]
}

fn swap_matrix_4x4() -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, o]]
}

fn cu_matrix_4x4(mat: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    [
        [o, z, z, z],
        [z, o, z, z],
        [z, z, mat[0][0], mat[0][1]],
        [z, z, mat[1][0], mat[1][1]],
    ]
}

/// CU-phase followed by SWAP: applies controlled-phase then exchanges qubits.
/// Invariant under qubit reorder (symmetric since phase acts on |11⟩ only).
fn cu_phase_swap_matrix(phase: Complex64) -> [[Complex64; 4]; 4] {
    let z = ZERO;
    let o = ONE;
    [[o, z, z, z], [z, z, o, z], [z, o, z, z], [z, z, z, phase]]
}

fn swap_gate_qubits(g: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut out = [[ZERO; 4]; 4];
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for l in 0..2 {
                    out[j * 2 + i][l * 2 + k] = g[i * 2 + j][k * 2 + l];
                }
            }
        }
    }
    out
}

/// Build a 2^N × 2^N MCU gate matrix (row-major, flattened).
///
/// `qubit_order[i]` maps block-position `i` to the MCU role index:
/// roles 0..num_controls are controls, role num_controls is the target.
/// The matrix is identity except when all controls are |1⟩, where `mat` is
/// applied to the target qubit.
fn mcu_matrix(
    num_controls: usize,
    mat: &[[Complex64; 2]; 2],
    qubit_order: &[usize],
) -> Vec<Complex64> {
    let n = num_controls + 1;
    let dim = 1usize << n;
    let mut gate = vec![ZERO; dim * dim];

    // Block position p → bit (n-1-p) in state index (MSB = leftmost).
    let mut ctrl_mask: usize = 0;
    let mut tgt_bit: usize = 0;
    for (pos, &role) in qubit_order.iter().enumerate() {
        let bit = n - 1 - pos;
        if role < num_controls {
            ctrl_mask |= 1 << bit;
        } else {
            tgt_bit = bit;
        }
    }

    for s_in in 0..dim {
        if (s_in & ctrl_mask) == ctrl_mask {
            let t_in = (s_in >> tgt_bit) & 1;
            for (t_out, row) in mat.iter().enumerate() {
                let s_out = (s_in & !(1 << tgt_bit)) | (t_out << tgt_bit);
                gate[s_out * dim + s_in] = row[t_in];
            }
        } else {
            gate[s_in * dim + s_in] = ONE;
        }
    }
    gate
}

// Benchmark-only surface: benches link the crate externally and cannot see pub(crate),
// so the SVD items are pub; doc(hidden) keeps them out of the rendered API.
/// Thin SVD result: A = U · diag(S) · V†
#[doc(hidden)]
pub struct SvdResult {
    pub u: Vec<Complex64>,
    pub u_rows: usize,
    pub s: Vec<f64>,
    pub vt: Vec<Complex64>,
    pub vt_cols: usize,
}

/// Compute thin SVD via the best available algorithm.
///
/// With the `parallel` feature: uses faer for matrices where m*n >= 256,
/// Jacobi for smaller. Without: always Jacobi.
#[doc(hidden)]
pub fn svd(a: &[Complex64], m: usize, n: usize) -> SvdResult {
    #[cfg(feature = "parallel")]
    if m * n >= 256 {
        return svd_faer(a, m, n);
    }
    svd_jacobi(a, m, n)
}

fn truncated_svd_rank(singular_values: &[f64], epsilon: f64, max_bond_dim: usize) -> usize {
    let s_max = singular_values.first().copied().unwrap_or(0.0);
    let threshold = epsilon * s_max;
    singular_values
        .iter()
        .take_while(|&&s| s > threshold)
        .count()
        .max(1)
        .min(max_bond_dim)
}

fn svd_left_site_data(svd_result: &SvdResult, bond_left: usize, chi: usize) -> Vec<Complex64> {
    let mut left_data = vec![ZERO; bond_left * 2 * chi];
    for alpha in 0..bond_left {
        for i in 0..2 {
            for gamma in 0..chi {
                let r = alpha * 2 + i;
                left_data[alpha * (2 * chi) + i * chi + gamma] =
                    svd_result.u[gamma * svd_result.u_rows + r];
            }
        }
    }
    left_data
}

fn fill_scaled_vt_data(
    out: &mut Vec<Complex64>,
    svd_result: &SvdResult,
    chi: usize,
    physical_dim: usize,
    bond_right: usize,
) {
    out.clear();
    out.resize(chi * physical_dim * bond_right, ZERO);
    for gamma in 0..chi {
        let s_val = Complex64::new(svd_result.s[gamma], 0.0);
        for s in 0..physical_dim {
            for beta in 0..bond_right {
                let c = s * bond_right + beta;
                out[gamma * (physical_dim * bond_right) + s * bond_right + beta] =
                    s_val * svd_result.vt[gamma * svd_result.vt_cols + c];
            }
        }
    }
}

/// Compute thin SVD using Jacobi one-sided rotations (column-major storage).
///
/// Returns U (m×k), S (k), V† (k×n) where k = min(m, n),
/// sorted by descending singular values.
#[doc(hidden)]
pub fn svd_jacobi(a: &[Complex64], m: usize, n: usize) -> SvdResult {
    let k = m.min(n);
    let transpose = m < n;

    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };

    let mut work = vec![ZERO; work_m * work_n];
    if transpose {
        for col_b in 0..work_n {
            for row_b in 0..work_m {
                work[col_b * work_m + row_b] = a[row_b * m + col_b].conj();
            }
        }
    } else {
        work.copy_from_slice(&a[..work_m * work_n]);
    }

    let mut v = vec![ZERO; work_n * work_n];
    for i in 0..work_n {
        v[i * work_n + i] = ONE;
    }

    let frob_sq: f64 = work.iter().map(|x| x.norm_sqr()).sum();
    let tol = SVD_CONVERGENCE * frob_sq;

    for _sweep in 0..MAX_SVD_SWEEPS {
        let mut off_diag = 0.0f64;

        for p in 0..work_n {
            for q in (p + 1)..work_n {
                let mut g_pp = 0.0f64;
                let mut g_qq = 0.0f64;
                let mut g_pq = ZERO;

                for r in 0..work_m {
                    let ap = work[p * work_m + r];
                    let aq = work[q * work_m + r];
                    g_pp += ap.norm_sqr();
                    g_qq += aq.norm_sqr();
                    g_pq += ap.conj() * aq;
                }

                off_diag += g_pq.norm_sqr();

                if g_pq.norm_sqr() < NORM_CLAMP_MIN {
                    continue;
                }

                let beta_norm = g_pq.norm();
                let phase = g_pq / beta_norm;

                let tau = (g_qq - g_pp) / (2.0 * beta_norm);
                let t = if tau >= 0.0 {
                    -1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let c_cx = Complex64::new(c, 0.0);
                let s_cx = Complex64::new(s, 0.0);
                let phase_conj = phase.conj();

                for r in 0..work_m {
                    let ap = work[p * work_m + r];
                    let aq = work[q * work_m + r];
                    work[p * work_m + r] = c_cx * ap + s_cx * phase_conj * aq;
                    work[q * work_m + r] = -s_cx * ap + c_cx * phase_conj * aq;
                }

                for r in 0..work_n {
                    let vp = v[p * work_n + r];
                    let vq = v[q * work_n + r];
                    v[p * work_n + r] = c_cx * vp + s_cx * phase_conj * vq;
                    v[q * work_n + r] = -s_cx * vp + c_cx * phase_conj * vq;
                }
            }
        }

        if off_diag < tol {
            break;
        }
    }

    let mut singular_values = vec![0.0f64; work_n];
    let mut u_work = vec![ZERO; work_m * work_n];

    for j in 0..work_n {
        let mut norm_sq = 0.0f64;
        for r in 0..work_m {
            norm_sq += work[j * work_m + r].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        singular_values[j] = norm;
        if norm > NORM_CLAMP_MIN {
            let inv_norm = 1.0 / norm;
            for r in 0..work_m {
                u_work[j * work_m + r] = work[j * work_m + r] * inv_norm;
            }
        }
    }

    let mut order: Vec<usize> = (0..work_n).collect();
    order.sort_by(|&a, &b| singular_values[b].partial_cmp(&singular_values[a]).unwrap());

    let mut s_sorted = vec![0.0f64; k];
    let mut u_sorted = vec![ZERO; work_m * k];
    let mut vt_sorted = vec![ZERO; k * work_n];

    for (new_idx, &old_idx) in order.iter().take(k).enumerate() {
        s_sorted[new_idx] = singular_values[old_idx];

        for r in 0..work_m {
            u_sorted[new_idx * work_m + r] = u_work[old_idx * work_m + r];
        }

        for r in 0..work_n {
            vt_sorted[new_idx * work_n + r] = v[old_idx * work_n + r].conj();
        }
    }

    if transpose {
        // SVD is computed from A^H (shape n x m): A^H = U_h * S * V_h^H
        // So A = V_h * S * U_h^H
        // U_A = V_h (m×k col-major): conj of vt_sorted (which stores V_h^H row-major)
        // V_A^H = U_h^H (k×n row-major): conj of u_sorted (which stores U_h col-major)
        SvdResult {
            u: vt_sorted.iter().map(|x| x.conj()).collect(),
            u_rows: m,
            s: s_sorted,
            vt: u_sorted.iter().map(|x| x.conj()).collect(),
            vt_cols: n,
        }
    } else {
        SvdResult {
            u: u_sorted,
            u_rows: work_m,
            s: s_sorted,
            vt: vt_sorted,
            vt_cols: work_n,
        }
    }
}

/// Compute thin SVD using the faer library (SIMD-accelerated bidiag + D&C).
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn svd_faer(a: &[Complex64], m: usize, n: usize) -> SvdResult {
    use faer::MatRef;

    let mat = MatRef::from_column_major_slice(a, m, n);

    let result = match mat.thin_svd() {
        Ok(svd) => svd,
        Err(_) => return svd_jacobi(a, m, n),
    };
    let k = m.min(n);

    let u_mat = result.U();
    let s_col = result.S().column_vector();
    let v_mat = result.V();

    let mut u = vec![ZERO; m * k];
    for j in 0..k {
        for i in 0..m {
            u[j * m + i] = u_mat[(i, j)];
        }
    }

    let s: Vec<f64> = (0..k).map(|i| s_col[i].re).collect();

    let mut vt = vec![ZERO; k * n];
    for i in 0..k {
        for j in 0..n {
            vt[i * n + j] = v_mat[(j, i)].conj();
        }
    }

    SvdResult {
        u,
        u_rows: m,
        s,
        vt,
        vt_cols: n,
    }
}

/// Pauli axis for [`MpsBackend::pauli_expectation`]. Identity factors
/// are conveyed by omission from the factor list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsPauliAxis {
    X,
    Y,
    Z,
}

fn mps_pauli_factors(observable: &[PauliTerm]) -> Vec<(usize, MpsPauliAxis)> {
    observable
        .iter()
        .map(|term| {
            let axis = match term.axis {
                PauliAxis::X => MpsPauliAxis::X,
                PauliAxis::Y => MpsPauliAxis::Y,
                PauliAxis::Z => MpsPauliAxis::Z,
            };
            (term.qubit, axis)
        })
        .collect()
}

#[inline]
fn pauli_matrix_entry(axis: Option<MpsPauliAxis>, j: usize, k: usize) -> Complex64 {
    match (axis, j, k) {
        (None, 0, 0) | (None, 1, 1) => ONE,
        (None, _, _) => ZERO,
        (Some(MpsPauliAxis::X), 0, 1) | (Some(MpsPauliAxis::X), 1, 0) => ONE,
        (Some(MpsPauliAxis::X), _, _) => ZERO,
        (Some(MpsPauliAxis::Y), 0, 1) => Complex64::new(0.0, -1.0),
        (Some(MpsPauliAxis::Y), 1, 0) => Complex64::new(0.0, 1.0),
        (Some(MpsPauliAxis::Y), _, _) => ZERO,
        (Some(MpsPauliAxis::Z), 0, 0) => ONE,
        (Some(MpsPauliAxis::Z), 1, 1) => Complex64::new(-1.0, 0.0),
        (Some(MpsPauliAxis::Z), _, _) => ZERO,
    }
}

/// Matrix Product State backend with bounded bond dimension.
#[derive(Clone)]
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
    /// [`crate::backend::mps_workspace_cap_elements`] read once at
    /// construction, so the per-gate check is a field compare rather than an
    /// atomic load.
    workspace_cap: u128,
}

impl MpsBackend {
    /// Create a new MPS backend with the given RNG seed and maximum bond dimension.
    pub fn new(seed: u64, max_bond_dim: usize) -> Self {
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
            workspace_cap: crate::backend::mps_workspace_cap_elements(),
        }
    }

    /// Create an MPS backend that keeps every non-zero singular value.
    pub(crate) fn new_exact(seed: u64) -> Self {
        let mut backend = Self::new(seed, usize::MAX);
        backend.svd_epsilon = 0.0;
        backend
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

    /// Cumulative relative state weight discarded by SVD truncation since
    /// [`Backend::init`] or the last [`Self::reset_truncation_tracking`].
    ///
    /// [`Backend::init`]: crate::backend::Backend::init
    /// Epsilon-threshold truncation contributes negligibly (`~svd_epsilon²`); a
    /// meaningful value indicates the bond-dimension cap discarded real weight.
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

    /// Peak bond dimension across all internal bonds. Returns 1 for a
    /// freshly initialized product state. Intended for diagnostics and
    /// regression tests that monitor entanglement growth.
    pub fn current_max_bond_dim(&self) -> usize {
        self.sites.iter().map(|s| s.bond_right).max().unwrap_or(1)
    }

    /// Configured bond-dimension cap. SVD truncation engages when a bond
    /// would exceed this value.
    pub fn max_bond_dim_cap(&self) -> usize {
        self.max_bond_dim
    }

    /// Restore the internal MPS site order to logical qubit order without
    /// changing the represented logical state.
    pub fn canonicalize_logical_order(&mut self) -> Result<()> {
        let swap_mat = swap_matrix_4x4();
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

    /// Inner product between two MPS values that share the same site
    /// layout. Returns an error if the qubit counts or layout differ.
    ///
    /// Contracts left-to-right through the environment tensor in two
    /// passes per site, avoiding the naive four-loop sum.
    pub fn inner_product(&self, other: &Self) -> Result<Complex64> {
        if self.num_qubits != other.num_qubits {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "MPS inner product requires equal qubit counts; got {} vs {}",
                    self.num_qubits, other.num_qubits,
                ),
            });
        }
        if self.logical_to_site != other.logical_to_site {
            return Err(crate::error::PrismError::InvalidParameter {
                message: "MPS inner product requires identical logical_to_site permutations"
                    .to_string(),
            });
        }
        if self.num_qubits == 0 {
            return Ok(ONE);
        }

        let mut env: Vec<Complex64> = vec![ONE; 1];
        let mut env_cols: usize = 1;

        for i in 0..self.num_qubits {
            let a = &self.sites[i];
            let b = &other.sites[i];
            let bl_a = a.bond_left;
            let br_a = a.bond_right;
            let bl_b = b.bond_left;
            let br_b = b.bond_right;

            // Step 1: tmp[β, j, α'] = Σ_α env[α, β] · conj(A[α, j, α'])
            let mut tmp = vec![ZERO; bl_b * 2 * br_a];
            for alpha in 0..bl_a {
                for j in 0..2 {
                    for alpha_p in 0..br_a {
                        let a_val = a.data[a.idx(alpha, j, alpha_p)].conj();
                        if a_val == ZERO {
                            continue;
                        }
                        for beta in 0..bl_b {
                            let e_ab = env[alpha * env_cols + beta];
                            tmp[beta * (2 * br_a) + j * br_a + alpha_p] += a_val * e_ab;
                        }
                    }
                }
            }

            // Step 2: new_env[α', β'] = Σ_{β, j} tmp[β, j, α'] · B[β, j, β']
            let mut new_env = vec![ZERO; br_a * br_b];
            for beta in 0..bl_b {
                for j in 0..2 {
                    for beta_p in 0..br_b {
                        let b_val = b.data[b.idx(beta, j, beta_p)];
                        if b_val == ZERO {
                            continue;
                        }
                        for alpha_p in 0..br_a {
                            let t = tmp[beta * (2 * br_a) + j * br_a + alpha_p];
                            new_env[alpha_p * br_b + beta_p] += t * b_val;
                        }
                    }
                }
            }

            env = new_env;
            env_cols = br_b;
        }

        Ok(env[0])
    }

    /// Compute `⟨ψ|P|ψ⟩` for a Pauli string `P = ⊗_i P_i` with one
    /// factor per listed `(qubit, axis)`. Missing factors are implicit
    /// `I`. Walks the contraction once `(O(N·χ³))` and applies each
    /// site's Pauli matrix inline; no MPS clone, no gate apply pass.
    /// Duplicate factors on the same qubit are rejected.
    pub fn pauli_expectation(&self, pauli_factors: &[(usize, MpsPauliAxis)]) -> Result<Complex64> {
        if self.num_qubits == 0 {
            return Ok(ONE);
        }

        let site_axis = self.pauli_site_axes(pauli_factors)?;

        let mut env: Vec<Complex64> = vec![ONE; 1];
        let mut env_cols: usize = 1;

        for (i, a) in self.sites.iter().enumerate().take(self.num_qubits) {
            env = self.contract_pauli_env(i, &env, env_cols, site_axis[i]);
            env_cols = a.bond_right;
        }

        Ok(env[0])
    }

    /// Site index of each requested Pauli factor, `None` where the string acts
    /// as identity. Rejects out-of-range qubits and duplicate factors.
    fn pauli_site_axes(
        &self,
        pauli_factors: &[(usize, MpsPauliAxis)],
    ) -> Result<Vec<Option<MpsPauliAxis>>> {
        let mut site_axis: Vec<Option<MpsPauliAxis>> = vec![None; self.num_qubits];
        for &(qubit, axis) in pauli_factors {
            if qubit >= self.num_qubits {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!(
                        "MPS Pauli expectation: qubit {qubit} out of range (n={})",
                        self.num_qubits
                    ),
                });
            }
            let site = self.logical_to_site[qubit];
            if site_axis[site].is_some() {
                return Err(crate::error::PrismError::InvalidParameter {
                    message: format!("MPS Pauli expectation: duplicate factor on qubit {qubit}"),
                });
            }
            site_axis[site] = Some(axis);
        }
        Ok(site_axis)
    }

    /// Absorb one site into the `⟨ψ|P|ψ⟩` environment, applying `axis` as the
    /// site's Pauli matrix (identity when `None`).
    ///
    /// `env` is indexed `[bra bond, ket bond]` with `env_cols` columns; the
    /// result is indexed the same way with `bond_right` columns.
    fn contract_pauli_env(
        &self,
        site: usize,
        env: &[Complex64],
        env_cols: usize,
        axis: Option<MpsPauliAxis>,
    ) -> Vec<Complex64> {
        let a = &self.sites[site];
        let bl = a.bond_left;
        let br = a.bond_right;

        // Step 1: tmp[β, j, α'] = Σ_α env[α, β] · conj(A[α, j, α'])
        let mut tmp = vec![ZERO; bl * 2 * br];
        for alpha in 0..bl {
            for j in 0..2 {
                for alpha_p in 0..br {
                    let a_val = a.data[a.idx(alpha, j, alpha_p)].conj();
                    if a_val == ZERO {
                        continue;
                    }
                    for beta in 0..bl {
                        let e_ab = env[alpha * env_cols + beta];
                        tmp[beta * (2 * br) + j * br + alpha_p] += a_val * e_ab;
                    }
                }
            }
        }

        // Step 2: new_env[α', β'] = Σ_{β, j, k} tmp[β, j, α'] · M[j,k] · A[β, k, β']
        // with M = local Pauli matrix at this site (or identity if no factor).
        let mut new_env = vec![ZERO; br * br];
        for beta in 0..bl {
            for j in 0..2 {
                for k in 0..2 {
                    let m_jk = pauli_matrix_entry(axis, j, k);
                    if m_jk == ZERO {
                        continue;
                    }
                    for beta_p in 0..br {
                        let a_bk = a.data[a.idx(beta, k, beta_p)];
                        if a_bk == ZERO {
                            continue;
                        }
                        let ma_val = m_jk * a_bk;
                        for alpha_p in 0..br {
                            let t = tmp[beta * (2 * br) + j * br + alpha_p];
                            new_env[alpha_p * br + beta_p] += t * ma_val;
                        }
                    }
                }
            }
        }

        new_env
    }

    /// Identity environments left of every site, plus `⟨ψ|ψ⟩` as the last
    /// entry, in the `[bra bond, ket bond]` convention of
    /// [`MpsBackend::contract_pauli_env`].
    ///
    /// Entry `s` is the contraction of sites `0..s`, so it has `bond_left(s)`
    /// columns and entry `num_qubits` is the `1x1` norm.
    fn pauli_env_prefixes(&self) -> Vec<Vec<Complex64>> {
        let mut prefixes = Vec::with_capacity(self.num_qubits + 1);
        prefixes.push(vec![ONE; 1]);
        let mut env_cols = 1usize;
        for site in 0..self.num_qubits {
            let next = self.contract_pauli_env(site, prefixes.last().unwrap(), env_cols, None);
            env_cols = self.sites[site].bond_right;
            prefixes.push(next);
        }
        prefixes
    }

    /// MPS site index currently hosting logical qubit `q`. SWAP-routing
    /// changes this mapping; non-adjacent two-qubit gates are realized
    /// as sequences of nearest-neighbour SWAPs and grow bond dimension
    /// in proportion to site-coordinate distance. Anchor-selection
    /// heuristics in CAMPS use this to score candidate cascades.
    pub fn site_for_qubit(&self, q: usize) -> usize {
        self.logical_to_site[q]
    }

    /// Test whether logical qubit `q` has zero marginal probability of
    /// being measured as `|1⟩`, within `tol`. For a state where qubit
    /// `q` has just been disentangled by an OFD cascade, this is
    /// equivalent to qubit `q` being in the pure `|0⟩` state on that
    /// site.
    ///
    /// Computed as `(1 − Re⟨Z_q⟩)/2 < tol` via [`Self::pauli_expectation`].
    pub fn is_qubit_in_zero_state(&self, q: usize, tol: f64) -> Result<bool> {
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

    fn apply_adjacent_two_qubit(
        &mut self,
        gate: &[[Complex64; 4]; 4],
        left_site: usize,
        left_is_first_qubit: bool,
    ) -> Result<()> {
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
            ));
        }

        let g = if left_is_first_qubit {
            *gate
        } else {
            swap_gate_qubits(gate)
        };

        // 1. Contract: Θ[α, i, j, β] = Σ_γ A[k][α,i,γ] · A[k+1][γ,j,β]
        let left_data = &self.sites[left_site].data;
        let right_data = &self.sites[right_site].data;
        let chunk_size = 2 * 2 * br;
        let mut theta = vec![ZERO; bl * chunk_size];

        // Transpose right: (bond_mid, 2, br) → (2, br, bond_mid) for sequential gamma access.
        let mut right_t = vec![ZERO; bond_mid * 2 * br];
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
        let mut theta_prime = vec![ZERO; bl * chunk_size];

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
        let mut mat = vec![ZERO; rows * cols];
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

        let svd_result = svd(&mat, rows, cols);
        let chi_new = truncated_svd_rank(&svd_result.s, self.svd_epsilon, self.max_bond_dim);
        self.record_truncation(&svd_result.s, chi_new);

        // A[k] from U: shape (bl, 2, chi_new)
        // U is column-major: U[col * rows + row]
        let left_data = svd_left_site_data(&svd_result, bl, chi_new);

        // A[k+1] from S·V†: shape (chi_new, 2, br)
        // V† is row-major: Vt[row * cols + col]
        let mut right_data = Vec::new();
        fill_scaled_vt_data(&mut right_data, &svd_result, chi_new, 2, br);

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
        Ok(())
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
            let swap_mat = swap_matrix_4x4();
            for s in (k + 1..m).rev() {
                self.apply_virtual_swap(s, &swap_mat)?;
            }
            self.apply_adjacent_two_qubit(gate, k, left_is_first)?;
        }
        Ok(())
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
        if phases.len() == 1 {
            let (target, phase) = phases[0];
            let mat = [[ONE, ZERO], [ZERO, phase]];
            let g = cu_matrix_4x4(&mat);
            return self.apply_two_qubit_gate(&g, control, target);
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

        let swap_mat = swap_matrix_4x4();

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
                    let g = cu_matrix_4x4(&mat);
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
                    let g = cu_matrix_4x4(&mat);
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

    /// Contract N adjacent site tensors into Θ with shape (χ_left, 2^N, χ_right).
    ///
    /// Returns (theta, bl, br) where theta is indexed as Θ[α, s, β] at
    /// `α * (dim * br) + s * br + β`, with dim = 2^n.
    fn contract_n_sites(&self, start: usize, n: usize) -> (Vec<Complex64>, usize, usize) {
        debug_assert!(n >= 2);
        let bl = self.sites[start].bond_left;
        let br = self.sites[start + n - 1].bond_right;

        let t0 = &self.sites[start];
        let mut theta = t0.data.clone();
        let mut cur_dim = 2usize;
        let mut cur_right = t0.bond_right;
        let mut new_theta = Vec::new();

        for k in 1..n {
            let t = &self.sites[start + k];
            let bond_mid = cur_right;
            let next_right = t.bond_right;
            let new_dim = cur_dim * 2;

            let nsite_chunk = new_dim * next_right;
            let needed = bl * nsite_chunk;
            new_theta.clear();
            new_theta.resize(needed, ZERO);
            let t_data = &t.data;
            let theta_ref = &theta;

            #[cfg(feature = "parallel")]
            if bl >= MIN_BOND_FOR_PAR {
                new_theta
                    .par_chunks_mut(nsite_chunk)
                    .enumerate()
                    .for_each(|(alpha, chunk)| {
                        for s in 0..cur_dim {
                            for j in 0..2 {
                                for beta in 0..next_right {
                                    let mut val = ZERO;
                                    for gamma in 0..bond_mid {
                                        val += theta_ref
                                            [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                            * t_data
                                                [gamma * (2 * next_right) + j * next_right + beta];
                                    }
                                    let new_s = s * 2 + j;
                                    chunk[new_s * next_right + beta] = val;
                                }
                            }
                        }
                    });
            } else {
                for alpha in 0..bl {
                    for s in 0..cur_dim {
                        for j in 0..2 {
                            for beta in 0..next_right {
                                let mut val = ZERO;
                                for gamma in 0..bond_mid {
                                    val += theta_ref
                                        [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                        * t_data[gamma * (2 * next_right) + j * next_right + beta];
                                }
                                let new_s = s * 2 + j;
                                new_theta[alpha * nsite_chunk + new_s * next_right + beta] = val;
                            }
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            for alpha in 0..bl {
                for s in 0..cur_dim {
                    for j in 0..2 {
                        for beta in 0..next_right {
                            let mut val = ZERO;
                            for gamma in 0..bond_mid {
                                val += theta_ref
                                    [alpha * (cur_dim * bond_mid) + s * bond_mid + gamma]
                                    * t_data[gamma * (2 * next_right) + j * next_right + beta];
                            }
                            let new_s = s * 2 + j;
                            new_theta[alpha * nsite_chunk + new_s * next_right + beta] = val;
                        }
                    }
                }
            }
            std::mem::swap(&mut theta, &mut new_theta);
            cur_dim = new_dim;
            cur_right = next_right;
        }

        (theta, bl, br)
    }

    /// Apply a 2^N × 2^N gate matrix (row-major) to a contracted tensor in-place.
    fn apply_gate_to_theta(
        theta: &[Complex64],
        gate: &[Complex64],
        dim: usize,
        bl: usize,
        br: usize,
    ) -> Vec<Complex64> {
        let gate_chunk = dim * br;
        let mut out = vec![ZERO; bl * gate_chunk];

        #[cfg(feature = "parallel")]
        if bl >= MIN_BOND_FOR_PAR {
            out.par_chunks_mut(gate_chunk)
                .enumerate()
                .for_each(|(alpha, chunk)| {
                    for sp in 0..dim {
                        for beta in 0..br {
                            let mut val = ZERO;
                            for s in 0..dim {
                                val +=
                                    gate[sp * dim + s] * theta[alpha * gate_chunk + s * br + beta];
                            }
                            chunk[sp * br + beta] = val;
                        }
                    }
                });
            return out;
        }

        for alpha in 0..bl {
            for sp in 0..dim {
                for beta in 0..br {
                    let mut val = ZERO;
                    for s in 0..dim {
                        val += gate[sp * dim + s] * theta[alpha * gate_chunk + s * br + beta];
                    }
                    out[alpha * gate_chunk + sp * br + beta] = val;
                }
            }
        }
        out
    }

    /// Decompose a contracted tensor back into N site tensors via N-1 SVDs.
    ///
    /// Left-to-right sweep: at each step, reshape the remainder to
    /// (χ × 2) rows × (remaining_dim × χ_right) cols, SVD, truncate,
    /// extract left site from U, and continue with S·V† as the new remainder.
    fn decompose_n_sites(
        &mut self,
        theta: &[Complex64],
        start: usize,
        n: usize,
        bl: usize,
        br: usize,
    ) {
        let mut remaining = theta.to_vec();
        let mut cur_bl = bl;
        let mut remaining_dim = 1usize << n;
        let mut mat = Vec::new();
        let mut new_remaining = Vec::new();

        for k in 0..n - 1 {
            remaining_dim /= 2;
            let rows = cur_bl * 2;
            let cols = remaining_dim * br;

            // Reshape to column-major matrix for SVD: M[r, c] at mat[c * rows + r]
            // r = α * 2 + i, c = s_rest * br + β
            mat.clear();
            mat.resize(rows * cols, ZERO);
            #[cfg(feature = "parallel")]
            if cur_bl >= MIN_BOND_FOR_PAR * 2 {
                let rem = &remaining;
                let rd = remaining_dim;
                mat.par_iter_mut().enumerate().for_each(|(out_idx, elem)| {
                    let c = out_idx / rows;
                    let r = out_idx % rows;
                    let alpha = r / 2;
                    let i = r & 1;
                    let s_rest = c / br;
                    let beta = c % br;
                    let combined_s = i * rd + s_rest;
                    *elem = rem[alpha * (2 * rd * br) + combined_s * br + beta];
                });
            } else {
                for alpha in 0..cur_bl {
                    for i in 0..2 {
                        for s_rest in 0..remaining_dim {
                            for beta in 0..br {
                                let r = alpha * 2 + i;
                                let c = s_rest * br + beta;
                                let combined_s = i * remaining_dim + s_rest;
                                mat[c * rows + r] = remaining
                                    [alpha * (2 * remaining_dim * br) + combined_s * br + beta];
                            }
                        }
                    }
                }
            }

            #[cfg(not(feature = "parallel"))]
            for alpha in 0..cur_bl {
                for i in 0..2 {
                    for s_rest in 0..remaining_dim {
                        for beta in 0..br {
                            let r = alpha * 2 + i;
                            let c = s_rest * br + beta;
                            let combined_s = i * remaining_dim + s_rest;
                            mat[c * rows + r] = remaining
                                [alpha * (2 * remaining_dim * br) + combined_s * br + beta];
                        }
                    }
                }
            }

            let svd_result = svd(&mat, rows, cols);
            let chi_new = truncated_svd_rank(&svd_result.s, self.svd_epsilon, self.max_bond_dim);
            self.record_truncation(&svd_result.s, chi_new);

            // Extract left site from U: shape (cur_bl, 2, chi_new)
            let left_data = svd_left_site_data(&svd_result, cur_bl, chi_new);

            self.sites[start + k] = SiteTensor {
                bond_left: cur_bl,
                bond_right: chi_new,
                data: left_data,
            };

            if k < n - 2 {
                // Build remainder from S·V†: shape (chi_new, remaining_dim, br)
                fill_scaled_vt_data(&mut new_remaining, &svd_result, chi_new, remaining_dim, br);
                std::mem::swap(&mut remaining, &mut new_remaining);
                cur_bl = chi_new;
            } else {
                // Last site: S·V† reshaped to (chi_new, 2, br)
                let mut right_data = Vec::new();
                fill_scaled_vt_data(&mut right_data, &svd_result, chi_new, 2, br);
                self.sites[start + n - 1] = SiteTensor {
                    bond_left: chi_new,
                    bond_right: br,
                    data: right_data,
                };
            }
        }
    }

    /// Reject a dense `2^n x 2^n` gate matrix over the workspace budget before
    /// it is built.
    ///
    /// The N-site path holds two of them at peak, the assembled matrix and the
    /// role-reordered copy, and both are allocated before
    /// [`Self::apply_adjacent_n_qubit`] reaches its own theta check.
    fn check_n_qubit_gate_matrix(&self, n: usize) -> Result<()> {
        // Rejected outright rather than by comparison: `2^(2n + 1)` is not
        // representable here, and the cap is itself `u128::MAX` when memory
        // detection is unavailable, so a comparison would pass and leave the
        // caller to shift `1usize` by more than its width.
        if n >= 63 {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate matrix",
                u128::MAX,
            ));
        }
        let elements = 2u128 << (2 * n);
        if elements > self.workspace_cap {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate matrix",
                elements,
            ));
        }
        Ok(())
    }

    fn apply_adjacent_n_qubit(
        &mut self,
        gate: &[Complex64],
        dim: usize,
        start_site: usize,
    ) -> Result<()> {
        let n = dim.trailing_zeros() as usize; // dim = 2^n
        // Contraction, gate application, and decomposition each hold a
        // (bl, 2^n, br) tensor, with the decomposition remainder and SVD input
        // making about four live at peak.
        let bl = self.sites[start_site].bond_left;
        let br = self.sites[start_site + n - 1].bond_right;
        let workspace = 4u128 * dim as u128 * bl as u128 * br as u128;
        if workspace > self.workspace_cap {
            return Err(crate::backend::workspace_allocation_error(
                "mps",
                "multi-qubit gate application",
                workspace,
            ));
        }

        let (theta, bl, br) = self.contract_n_sites(start_site, n);
        let theta_prime = Self::apply_gate_to_theta(&theta, gate, dim, bl, br);
        self.decompose_n_sites(&theta_prime, start_site, n, bl, br);
        Ok(())
    }

    /// Apply an N-qubit gate to arbitrary (possibly non-adjacent) qubits.
    ///
    /// SWAP-routes qubits into a contiguous block, applies the gate, then
    /// reverses the SWAPs. Returns nothing; modifies `self.sites` in place.
    fn apply_n_qubit_gate(
        &mut self,
        gate: &[Complex64],
        dim: usize,
        qubits: &[usize],
    ) -> Result<()> {
        let n = qubits.len();
        debug_assert_eq!(dim, 1 << n);

        let mut indexed: Vec<(usize, usize)> = qubits
            .iter()
            .enumerate()
            .map(|(role, &pos)| (pos, role))
            .collect();
        indexed.sort_unstable_by_key(|&(pos, _)| pos);

        let sorted_positions: Vec<usize> = indexed.iter().map(|&(pos, _)| pos).collect();
        let start = sorted_positions[0];

        let contiguous = sorted_positions.last().unwrap() - start + 1 == n;

        if contiguous {
            let qubit_order: Vec<usize> = indexed.iter().map(|&(_, role)| role).collect();
            let reordered_gate = Self::reorder_n_gate(gate, dim, &qubit_order, n);
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start)?;
        } else {
            let swap_mat = swap_matrix_4x4();
            let mut current_positions = sorted_positions;
            let mut swap_log: Vec<usize> = Vec::new();

            for i in 1..n {
                let target_pos = start + i;
                while current_positions[i] > target_pos {
                    let s = current_positions[i] - 1;
                    self.apply_adjacent_two_qubit(&swap_mat, s, true)?;
                    swap_log.push(s);
                    // Update any tracked qubit that was displaced
                    for cp in &mut current_positions[(i + 1)..n] {
                        if *cp == s {
                            *cp = s + 1;
                        }
                    }
                    current_positions[i] -= 1;
                }
            }

            let qubit_order: Vec<usize> = indexed.iter().map(|&(_, role)| role).collect();
            let reordered_gate = Self::reorder_n_gate(gate, dim, &qubit_order, n);
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start)?;

            for &s in swap_log.iter().rev() {
                self.apply_adjacent_two_qubit(&swap_mat, s, true)?;
            }
        }
        Ok(())
    }

    /// Reorder an N-qubit gate matrix to match the physical qubit ordering.
    ///
    /// `qubit_order[block_pos]` = the MCU role index for that block position.
    /// The original gate assumes role order 0,1,...,N-1 (controls then target).
    /// This function permutes the matrix indices to match the block ordering.
    fn reorder_n_gate(
        gate: &[Complex64],
        dim: usize,
        qubit_order: &[usize],
        n: usize,
    ) -> Vec<Complex64> {
        let identity_order = qubit_order.iter().enumerate().all(|(i, &r)| r == i);
        if identity_order {
            return gate.to_vec();
        }

        let mut out = vec![ZERO; dim * dim];
        for s_in in 0..dim {
            for s_out in 0..dim {
                let mut logical_in = 0usize;
                let mut logical_out = 0usize;
                for (pos, &role) in qubit_order.iter().enumerate() {
                    let bit_pos = n - 1 - pos; // bit position for block pos
                    let role_bit = n - 1 - role; // bit position in role order
                    if s_in & (1 << bit_pos) != 0 {
                        logical_in |= 1 << role_bit;
                    }
                    if s_out & (1 << bit_pos) != 0 {
                        logical_out |= 1 << role_bit;
                    }
                }
                out[s_out * dim + s_in] = gate[logical_out * dim + logical_in];
            }
        }
        out
    }

    fn compute_left_env(&self, site: usize) -> Vec<Complex64> {
        let mut env = vec![ONE];
        let mut env_dim = 1usize;

        for s in 0..site {
            let t = &self.sites[s];
            let bl = t.bond_left;
            let br = t.bond_right;
            debug_assert_eq!(env_dim, bl);

            #[cfg(feature = "parallel")]
            if br >= MIN_BOND_FOR_PAR {
                let mut new_env = vec![ZERO; br * br];
                new_env.par_iter_mut().enumerate().for_each(|(idx, val)| {
                    let gamma = idx / br;
                    let gamma_p = idx % br;
                    let mut sum = ZERO;
                    for alpha in 0..bl {
                        for alpha_p in 0..bl {
                            let env_val = env[alpha * bl + alpha_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                sum += env_val
                                    * t.data[t.idx(alpha, i, gamma)]
                                    * t.data[t.idx(alpha_p, i, gamma_p)].conj();
                            }
                        }
                    }
                    *val = sum;
                });
                env = new_env;
                env_dim = br;
                continue;
            }

            let mut new_env = vec![ZERO; br * br];
            for gamma in 0..br {
                for gamma_p in 0..br {
                    let mut val = ZERO;
                    for alpha in 0..bl {
                        for alpha_p in 0..bl {
                            let env_val = env[alpha * bl + alpha_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                val += env_val
                                    * t.data[t.idx(alpha, i, gamma)]
                                    * t.data[t.idx(alpha_p, i, gamma_p)].conj();
                            }
                        }
                    }
                    new_env[gamma * br + gamma_p] = val;
                }
            }
            env = new_env;
            env_dim = br;
        }
        env
    }

    /// Absorb one site into a right environment, in the two passes
    /// [`Self::inner_product`] uses: `O(χ³)` instead of the four-index sum's
    /// `O(χ⁴)`. `tmp` carries the intermediate across a sweep so the inner
    /// buffer is allocated once rather than per site.
    ///
    /// [`Self::compute_right_env`] deliberately keeps the four-index form
    /// instead of calling this. Its callers are measurement and reset, where
    /// projection has just zeroed half of each site tensor and the environments
    /// are mostly zero; the `env_val == ZERO` skip there beats dense `O(χ³)`
    /// arithmetic by 14% on `mps/hotspots/measure_reset_32q_r3` (496 µs against
    /// 574 µs). The sweep below runs on unprojected chains where the
    /// environments are dense and the asymptotics decide instead. Measure both
    /// before unifying them.
    fn contract_right_env(
        &self,
        site: usize,
        env: &[Complex64],
        tmp: &mut Vec<Complex64>,
    ) -> Vec<Complex64> {
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;

        // tmp[α, i, β'] = Σ_β A[α, i, β] · env[β, β']
        tmp.clear();
        tmp.resize(bl * 2 * br, ZERO);
        for alpha in 0..bl {
            for i in 0..2 {
                for beta in 0..br {
                    let a = t.data[t.idx(alpha, i, beta)];
                    if a == ZERO {
                        continue;
                    }
                    let dst = (alpha * 2 + i) * br;
                    let src = beta * br;
                    for beta_p in 0..br {
                        tmp[dst + beta_p] += a * env[src + beta_p];
                    }
                }
            }
        }

        // new_env[α, α'] = Σ_{i, β'} tmp[α, i, β'] · conj(A[α', i, β'])
        let mut new_env = vec![ZERO; bl * bl];
        for alpha_p in 0..bl {
            for i in 0..2 {
                for beta_p in 0..br {
                    let a = t.data[t.idx(alpha_p, i, beta_p)].conj();
                    if a == ZERO {
                        continue;
                    }
                    for alpha in 0..bl {
                        new_env[alpha * bl + alpha_p] += tmp[(alpha * 2 + i) * br + beta_p] * a;
                    }
                }
            }
        }
        new_env
    }

    fn compute_right_env(&self, site: usize) -> Vec<Complex64> {
        let mut env = vec![ONE];
        let mut env_dim = 1usize;

        for s in (site + 1..self.num_qubits).rev() {
            let t = &self.sites[s];
            let bl = t.bond_left;
            let br = t.bond_right;
            debug_assert_eq!(env_dim, br);

            #[cfg(feature = "parallel")]
            if bl >= MIN_BOND_FOR_PAR {
                let mut new_env = vec![ZERO; bl * bl];
                new_env.par_iter_mut().enumerate().for_each(|(idx, val)| {
                    let alpha = idx / bl;
                    let alpha_p = idx % bl;
                    let mut sum = ZERO;
                    for beta in 0..br {
                        for beta_p in 0..br {
                            let env_val = env[beta * br + beta_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                sum += t.data[t.idx(alpha, i, beta)]
                                    * t.data[t.idx(alpha_p, i, beta_p)].conj()
                                    * env_val;
                            }
                        }
                    }
                    *val = sum;
                });
                env = new_env;
                env_dim = bl;
                continue;
            }

            let mut new_env = vec![ZERO; bl * bl];
            for alpha in 0..bl {
                for alpha_p in 0..bl {
                    let mut val = ZERO;
                    for beta in 0..br {
                        for beta_p in 0..br {
                            let env_val = env[beta * br + beta_p];
                            if env_val == ZERO {
                                continue;
                            }
                            for i in 0..2 {
                                val += t.data[t.idx(alpha, i, beta)]
                                    * t.data[t.idx(alpha_p, i, beta_p)].conj()
                                    * env_val;
                            }
                        }
                    }
                    new_env[alpha * bl + alpha_p] = val;
                }
            }
            env = new_env;
            env_dim = bl;
        }
        env
    }

    /// Right environment of every site, in one backward sweep.
    ///
    /// `envs[s]` is what [`Self::compute_right_env`] returns for `s`, but the
    /// sweep shares each partial contraction with the site to its left, so all
    /// `n` cost one pass instead of `n`.
    fn right_environments(&self) -> Vec<Vec<Complex64>> {
        let n = self.num_qubits;
        let mut envs: Vec<Vec<Complex64>> = Vec::with_capacity(n);
        let mut tmp = Vec::new();
        envs.push(vec![ONE]);
        for site in (1..n).rev() {
            let next = self.contract_right_env(site, envs.last().unwrap(), &mut tmp);
            envs.push(next);
        }
        envs.reverse();
        envs
    }

    fn apply_reset(&mut self, qubit: usize) {
        let prob = self.site_outcome_probabilities(qubit);
        let outcome = usize::from(self.rng.random::<f64>() < prob[1].clamp(0.0, 1.0));
        self.collapse_site_to_zero(qubit, outcome, prob[outcome].clamp(0.0, 1.0));
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let prob = self.site_outcome_probabilities(qubit);

        let measured = usize::from(self.rng.random::<f64>() < prob[1].clamp(0.0, 1.0));
        self.classical_bits[classical_bit] = measured == 1;
        self.project_site_outcome(qubit, measured, prob[measured].clamp(0.0, 1.0));
    }

    fn site_outcome_probabilities(&self, site: usize) -> [f64; 2] {
        let l_env = self.compute_left_env(site);
        let r_env = self.compute_right_env(site);
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;

        let mut prob = [0.0f64; 2];
        for (outcome, prob_out) in prob.iter_mut().enumerate() {
            #[cfg(feature = "parallel")]
            if bl >= MIN_BOND_FOR_PAR {
                let val: Complex64 = (0..bl)
                    .into_par_iter()
                    .map(|alpha| {
                        let mut sum = ZERO;
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
                                    sum += l_val
                                        * t.data[t.idx(alpha, outcome, beta)]
                                        * t.data[t.idx(alpha_p, outcome, beta_p)].conj()
                                        * r_val;
                                }
                            }
                        }
                        sum
                    })
                    .sum();
                *prob_out = val.re;
                continue;
            }

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
                                * t.data[t.idx(alpha, outcome, beta)]
                                * t.data[t.idx(alpha_p, outcome, beta_p)].conj()
                                * r_val;
                        }
                    }
                }
            }
            *prob_out = val.re;
        }
        prob
    }

    /// Take the Born probability from the caller so the site environments are
    /// contracted once per measurement rather than twice, the convention
    /// [`Self::collapse_site_to_zero`] already uses for reset.
    fn project_site_outcome(&mut self, site: usize, outcome: usize, prob: f64) -> f64 {
        if prob <= NORM_CLAMP_MIN {
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
        prob
    }

    /// Collapse a site onto `outcome` and return it to the zero state.
    ///
    /// Writing the surviving component straight into the zero slot folds in the
    /// X that a trajectory reset would otherwise apply after projecting onto
    /// outcome 1, and takes the Born probability from the caller so the site
    /// environments are contracted once per reset rather than twice.
    fn collapse_site_to_zero(&mut self, site: usize, outcome: usize, prob: f64) {
        if prob <= NORM_CLAMP_MIN {
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
    /// requested branch should be discarded by the caller.
    pub(crate) fn project_z_outcome(&mut self, qubit: usize, outcome: bool) -> f64 {
        let site = self.site_for_logical(qubit);
        let outcome = usize::from(outcome);
        let prob = self.site_outcome_probabilities(site)[outcome].clamp(0.0, 1.0);
        self.project_site_outcome(site, outcome, prob)
    }

    fn reduced_density_site(&self, site: usize) -> [[Complex64; 2]; 2] {
        let l_env = self.compute_left_env(site);
        let r_env = self.compute_right_env(site);
        let t = &self.sites[site];
        let bl = t.bond_left;
        let br = t.bond_right;
        let mut rho = [[ZERO; 2]; 2];

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

        rho
    }

    /// Conditional outcome weights at `site`, given the boundary vector `left`
    /// that carries the bits already fixed to its left.
    ///
    /// Writes `w[i * bond_right + β] = Σ_α left[α] · A[α, i, β]` into `w` and
    /// returns the unnormalized weight of each outcome. The conditioned left
    /// environment is the rank-one `left[α]·conj(left[α'])`, which collapses
    /// the four-index contraction of [`Self::site_outcome_probabilities`] to
    /// `O(χ²)`.
    fn site_conditional_weights(
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

    /// Every `(basis prefix, left-bond vector)` pair at `depth`, in DFS order.
    #[cfg(feature = "parallel")]
    fn expansion_prefixes(&self, depth: usize) -> Vec<(usize, Vec<Complex64>)> {
        let mut scratch = self.expansion_scratch();
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
    fn expand_dense<T: Copy + Send>(&self, out: &mut [T], convert: impl Fn(Complex64) -> T + Sync) {
        #[cfg(feature = "parallel")]
        {
            let split = self
                .num_qubits
                .saturating_sub(MIN_LEAVES_PER_TASK_BITS)
                .min(MAX_SPLIT_BITS);
            if split > 0 {
                let ptr = DenseOutPtr(out.as_mut_ptr());
                self.expansion_prefixes(split)
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
        self.expand_subtree(0, 0, &mut scratch, &mut |idx, amp| out[idx] = convert(amp));
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Rzz(_) => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cx => {
                let g = cx_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cz => {
                let g = cz_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Swap => {
                let g = swap_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1])?;
            }
            Gate::Cu(mat) => {
                let g = cu_matrix_4x4(mat);
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
                let role_order: Vec<usize> = (0..n).collect();
                let gate_mat = mcu_matrix(num_ctrl, &data.mat, &role_order);
                self.apply_n_qubit_gate(&gate_mat, dim, &all_qubits)?;
            }
            Gate::BatchPhase(data) => {
                let control = self.site_for_logical(targets[0]);
                let phases: Vec<(usize, Complex64)> = data
                    .phases
                    .iter()
                    .map(|&(qubit, phase)| (self.site_for_logical(qubit), phase))
                    .collect();
                self.apply_batch_phase_bubble(control, &phases)?;
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

impl Backend for MpsBackend {
    fn name(&self) -> &'static str {
        "mps"
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::Mps
    }

    fn exactness(&self) -> crate::sim::Exactness {
        crate::sim::Exactness::Approximate {
            fidelity_lower_bound: Some((1.0 - self.truncation_discarded).max(0.0)),
        }
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.truncation_discarded = 0.0;
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
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                self.apply_measure(self.site_for_logical(*qubit), *classical_bit);
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(self.site_for_logical(*qubit));
            }
            Instruction::Barrier { .. } => {}
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

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(self.site_for_logical(qubit));
        Ok(())
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.apply_single_qubit_gate(self.site_for_logical(qubit), matrix);
        Ok(())
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        Ok(self.reduced_density_site(self.site_for_logical(qubit)))
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        let dim = dense_probability_len(self.name(), self.num_qubits)?;
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
        let max_bond = self
            .sites
            .iter()
            .map(|site| site.bond_left.max(site.bond_right))
            .max()
            .unwrap_or(1);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut left = vec![ZERO; max_bond];
        let mut w = vec![ZERO; 2 * max_bond];

        for shot in 0..num_shots {
            left[0] = ONE;
            left[1..].fill(ZERO);
            for (site, right_env) in right.iter().enumerate() {
                let br = self.sites[site].bond_right;
                let prob = self.site_conditional_weights(site, &left, right_env, &mut w);

                let total = prob[0] + prob[1];
                let prob_one = if total > 0.0 {
                    (prob[1] / total).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let outcome = usize::from(rng.random::<f64>() < prob_one);
                if outcome == 1 {
                    samples.set(shot, self.logical_for_site(site));
                }

                let scale = 1.0 / prob[outcome].max(NORM_CLAMP_MIN).sqrt();
                left[..br].copy_from_slice(&w[outcome * br..(outcome + 1) * br]);
                for value in &mut left[..br] {
                    *value *= scale;
                }
            }
        }
        Ok(samples)
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Identity environments are swept once and shared, so an observable
    /// contracts only the sites between its first and last non-identity
    /// factor: a single-qubit observable costs one site instead of a full
    /// chain walk. Divided by `⟨ψ|ψ⟩` because a truncated chain is not
    /// exactly normalized.
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

        let prefixes = self.pauli_env_prefixes();
        let rights = self.right_environments();
        let norm = prefixes[self.num_qubits][0].re;

        observables
            .iter()
            .map(|observable| {
                let site_axis = self.pauli_site_axes(&mps_pauli_factors(observable))?;
                if norm == 0.0 {
                    return Ok(0.0);
                }
                let Some(first) = site_axis.iter().position(Option::is_some) else {
                    return Ok(1.0);
                };
                let last = site_axis.iter().rposition(Option::is_some).unwrap();

                let mut env = self.contract_pauli_env(
                    first,
                    &prefixes[first],
                    self.sites[first].bond_left,
                    site_axis[first],
                );
                let mut cols = self.sites[first].bond_right;
                for (site, axis) in site_axis.iter().enumerate().take(last + 1).skip(first + 1) {
                    env = self.contract_pauli_env(site, &env, cols, *axis);
                    cols = self.sites[site].bond_right;
                }

                // The identity cap right of `last` carries the ket bond first,
                // so closing the two chains is a trace of the two matrices.
                let right = &rights[last];
                let mut value = ZERO;
                for bra in 0..cols {
                    for ket in 0..cols {
                        value += env[bra * cols + ket] * right[ket * cols + bra];
                    }
                }
                Ok(value.re / norm)
            })
            .collect()
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        let dim = dense_statevector_len(self.name(), "statevector export", self.num_qubits)?;
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
