//! Matrix Product State (MPS) simulation backend.
//!
//! Represents an n-qubit state as a chain of rank-3 tensors:
//!
//! ```text
//!   A[0] — A[1] — A[2] — ... — A[n-1]
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
//! - Bond dimension capped at `max_bond_dim` (configurable, default 64).
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
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::backend::{simd, Backend, NORM_CLAMP_MIN};
use crate::circuit::Instruction;
use crate::error::{PrismError, Result};
use crate::gates::Gate;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const DEFAULT_SVD_EPSILON: f64 = 1e-12;
const MAX_SVD_SWEEPS: usize = 100;
const SVD_CONVERGENCE: f64 = 1e-14;
const MAX_PROB_QUBITS: usize = 20;
#[cfg(feature = "parallel")]
const MIN_DIM_FOR_PAR: usize = 1 << 14;
#[cfg(feature = "parallel")]
const MIN_BOND_FOR_PAR: usize = 32;

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

/// Reindex a 4×4 gate matrix to swap the two qubit roles.
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
        // We computed SVD of A^H (shape n×m): A^H = U_h * S * V_h^H
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
    use faer::Mat;

    let mat = Mat::<faer::complex_native::c64>::from_fn(m, n, |i, j| {
        let v = a[j * m + i];
        faer::complex_native::c64::new(v.re, v.im)
    });

    let result = mat.thin_svd();
    let k = m.min(n);

    let u_mat = result.u();
    let s_vec = result.s_diagonal();
    let v_mat = result.v();

    let mut u = vec![ZERO; m * k];
    for j in 0..k {
        for i in 0..m {
            let v = u_mat.read(i, j);
            u[j * m + i] = Complex64::new(v.re, v.im);
        }
    }

    let s: Vec<f64> = (0..k).map(|i| s_vec.read(i).re).collect();

    let mut vt = vec![ZERO; k * n];
    for i in 0..k {
        for j in 0..n {
            let v = v_mat.read(j, i);
            vt[i * n + j] = Complex64::new(v.re, -v.im);
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
        }
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

    fn apply_virtual_swap(&mut self, left_site: usize, swap_mat: &[[Complex64; 4]; 4]) {
        self.apply_adjacent_two_qubit(swap_mat, left_site, true);
        self.swap_layout_labels(left_site);
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
    ) {
        let right_site = left_site + 1;
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

        let s_max = svd_result.s.first().copied().unwrap_or(0.0);
        let thresh = self.svd_epsilon * s_max;
        let mut chi_new = svd_result
            .s
            .iter()
            .take_while(|&&s| s > thresh)
            .count()
            .max(1);
        chi_new = chi_new.min(self.max_bond_dim);

        // A[k] from U: shape (bl, 2, chi_new)
        // U is column-major: U[col * rows + row]
        let mut left_data = vec![ZERO; bl * 2 * chi_new];
        for alpha in 0..bl {
            for i in 0..2 {
                for gamma in 0..chi_new {
                    let r = alpha * 2 + i;
                    left_data[alpha * (2 * chi_new) + i * chi_new + gamma] =
                        svd_result.u[gamma * svd_result.u_rows + r];
                }
            }
        }

        // A[k+1] from S·V†: shape (chi_new, 2, br)
        // V† is row-major: Vt[row * cols + col]
        let mut right_data = vec![ZERO; chi_new * 2 * br];
        for gamma in 0..chi_new {
            let s_val = Complex64::new(svd_result.s[gamma], 0.0);
            for j in 0..2 {
                for beta in 0..br {
                    let c = j * br + beta;
                    right_data[gamma * (2 * br) + j * br + beta] =
                        s_val * svd_result.vt[gamma * svd_result.vt_cols + c];
                }
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
    }

    fn apply_two_qubit_gate(&mut self, gate: &[[Complex64; 4]; 4], q0: usize, q1: usize) {
        let p0 = self.site_for_logical(q0);
        let p1 = self.site_for_logical(q1);
        let k = p0.min(p1);
        let m = p0.max(p1);
        let left_is_first = p0 < p1;

        if m - k == 1 {
            self.apply_adjacent_two_qubit(gate, k, left_is_first);
        } else {
            let swap_mat = swap_matrix_4x4();
            for s in (k + 1..m).rev() {
                self.apply_virtual_swap(s, &swap_mat);
            }
            self.apply_adjacent_two_qubit(gate, k, left_is_first);
        }
    }

    /// Apply BatchPhase via bubble routing: sweep control toward targets,
    /// applying each CU-phase+SWAP as a single 2-site operation. O(k) tensor
    /// ops instead of O(k²) for k non-adjacent phases.
    fn apply_batch_phase_bubble(&mut self, control: usize, phases: &[(usize, Complex64)]) {
        if phases.is_empty() {
            return;
        }
        if phases.len() == 1 {
            let (target, phase) = phases[0];
            let mat = [[ONE, ZERO], [ZERO, phase]];
            let g = cu_matrix_4x4(&mat);
            self.apply_two_qubit_gate(&g, control, target);
            return;
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
                    self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true);
                    cur_pos += 1;
                }
                if idx < last_idx {
                    let combined = cu_phase_swap_matrix(phase);
                    self.apply_adjacent_two_qubit(&combined, cur_pos, true);
                    cur_pos += 1;
                } else {
                    let mat = [[ONE, ZERO], [ZERO, phase]];
                    let g = cu_matrix_4x4(&mat);
                    self.apply_adjacent_two_qubit(&g, cur_pos, true);
                }
            }
            while cur_pos > control {
                cur_pos -= 1;
                self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true);
            }
        }

        // Left sweep: bubble control leftward
        if !left.is_empty() {
            let mut cur_pos = control;
            let last_idx = left.len() - 1;
            for (idx, &(target, phase)) in left.iter().enumerate() {
                while cur_pos > target + 1 {
                    cur_pos -= 1;
                    self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true);
                }
                if idx < last_idx {
                    let combined = cu_phase_swap_matrix(phase);
                    self.apply_adjacent_two_qubit(&combined, cur_pos - 1, true);
                    cur_pos -= 1;
                } else {
                    let mat = [[ONE, ZERO], [ZERO, phase]];
                    let g = cu_matrix_4x4(&mat);
                    self.apply_adjacent_two_qubit(&g, cur_pos - 1, false);
                }
            }
            while cur_pos < control {
                self.apply_adjacent_two_qubit(&swap_mat, cur_pos, true);
                cur_pos += 1;
            }
        }
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

            let s_max = svd_result.s.first().copied().unwrap_or(0.0);
            let thresh = self.svd_epsilon * s_max;
            let mut chi_new = svd_result
                .s
                .iter()
                .take_while(|&&s| s > thresh)
                .count()
                .max(1);
            chi_new = chi_new.min(self.max_bond_dim);

            // Extract left site from U: shape (cur_bl, 2, chi_new)
            let mut left_data = vec![ZERO; cur_bl * 2 * chi_new];
            for alpha in 0..cur_bl {
                for i in 0..2 {
                    for gamma in 0..chi_new {
                        let r = alpha * 2 + i;
                        left_data[alpha * (2 * chi_new) + i * chi_new + gamma] =
                            svd_result.u[gamma * svd_result.u_rows + r];
                    }
                }
            }

            self.sites[start + k] = SiteTensor {
                bond_left: cur_bl,
                bond_right: chi_new,
                data: left_data,
            };

            if k < n - 2 {
                // Build remainder from S·V†: shape (chi_new, remaining_dim, br)
                let nr_len = chi_new * remaining_dim * br;
                new_remaining.clear();
                new_remaining.resize(nr_len, ZERO);
                for gamma in 0..chi_new {
                    let s_val = Complex64::new(svd_result.s[gamma], 0.0);
                    for s_rest in 0..remaining_dim {
                        for beta in 0..br {
                            let c = s_rest * br + beta;
                            new_remaining[gamma * (remaining_dim * br) + s_rest * br + beta] =
                                s_val * svd_result.vt[gamma * svd_result.vt_cols + c];
                        }
                    }
                }
                std::mem::swap(&mut remaining, &mut new_remaining);
                cur_bl = chi_new;
            } else {
                // Last site: S·V† reshaped to (chi_new, 2, br)
                let mut right_data = vec![ZERO; chi_new * 2 * br];
                for gamma in 0..chi_new {
                    let s_val = Complex64::new(svd_result.s[gamma], 0.0);
                    for j in 0..2 {
                        for beta in 0..br {
                            let c = j * br + beta;
                            right_data[gamma * (2 * br) + j * br + beta] =
                                s_val * svd_result.vt[gamma * svd_result.vt_cols + c];
                        }
                    }
                }
                self.sites[start + n - 1] = SiteTensor {
                    bond_left: chi_new,
                    bond_right: br,
                    data: right_data,
                };
            }
        }
    }

    /// Apply an N-qubit gate on N adjacent sites starting at `start_site`.
    fn apply_adjacent_n_qubit(&mut self, gate: &[Complex64], dim: usize, start_site: usize) {
        let n = dim.trailing_zeros() as usize; // dim = 2^n
        let (theta, bl, br) = self.contract_n_sites(start_site, n);
        let theta_prime = Self::apply_gate_to_theta(&theta, gate, dim, bl, br);
        self.decompose_n_sites(&theta_prime, start_site, n, bl, br);
    }

    /// Apply an N-qubit gate to arbitrary (possibly non-adjacent) qubits.
    ///
    /// SWAP-routes qubits into a contiguous block, applies the gate, then
    /// reverses the SWAPs. Returns nothing; modifies `self.sites` in place.
    fn apply_n_qubit_gate(&mut self, gate: &[Complex64], dim: usize, qubits: &[usize]) {
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
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start);
        } else {
            let swap_mat = swap_matrix_4x4();
            let mut current_positions = sorted_positions.clone();
            let mut swap_log: Vec<usize> = Vec::new();

            for i in 1..n {
                let target_pos = start + i;
                while current_positions[i] > target_pos {
                    let s = current_positions[i] - 1;
                    self.apply_adjacent_two_qubit(&swap_mat, s, true);
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
            self.apply_adjacent_n_qubit(&reordered_gate, dim, start);

            for &s in swap_log.iter().rev() {
                self.apply_adjacent_two_qubit(&swap_mat, s, true);
            }
        }
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

    fn apply_reset(&mut self, qubit: usize) {
        let l_env = self.compute_left_env(qubit);
        let r_env = self.compute_right_env(qubit);
        let t = &self.sites[qubit];
        let bl = t.bond_left;
        let br = t.bond_right;

        let mut prob_zero = 0.0f64;
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
                        prob_zero += (l_val
                            * t.data[t.idx(alpha, 0, beta)]
                            * t.data[t.idx(alpha_p, 0, beta_p)].conj()
                            * r_val)
                            .re;
                    }
                }
            }
        }

        if prob_zero > NORM_CLAMP_MIN {
            let inv_sqrt = 1.0 / prob_zero.sqrt();
            let scale = Complex64::new(inv_sqrt, 0.0);
            let t = &mut self.sites[qubit];
            for alpha in 0..bl {
                for beta in 0..br {
                    let idx_0 = alpha * (2 * br) + beta;
                    let idx_1 = alpha * (2 * br) + br + beta;
                    t.data[idx_0] *= scale;
                    t.data[idx_1] = ZERO;
                }
            }
        } else {
            self.apply_single_qubit_gate(qubit, &[[ZERO, ONE], [ONE, ZERO]]);
        }
    }

    fn apply_measure(&mut self, qubit: usize, classical_bit: usize) {
        let l_env = self.compute_left_env(qubit);
        let r_env = self.compute_right_env(qubit);
        let t = &self.sites[qubit];
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

        let measured = if self.rng.random::<f64>() < prob[1] {
            1usize
        } else {
            0usize
        };
        self.classical_bits[classical_bit] = measured == 1;

        let inv_sqrt_prob = 1.0 / prob[measured].clamp(NORM_CLAMP_MIN, 1.0).sqrt();
        let scale = Complex64::new(inv_sqrt_prob, 0.0);
        let other = 1 - measured;

        let t = &mut self.sites[qubit];
        for alpha in 0..bl {
            for beta in 0..br {
                let idx_m = alpha * (2 * br) + measured * br + beta;
                let idx_o = alpha * (2 * br) + other * br + beta;
                t.data[idx_m] *= scale;
                t.data[idx_o] = ZERO;
            }
        }
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

    fn chain_amplitude(&self, basis: usize) -> Complex64 {
        let n = self.num_qubits;
        let mut vec_data = vec![ONE];
        for site in 0..n {
            let logical = self.logical_for_site(site);
            let bit = (basis >> logical) & 1;
            let t = &self.sites[site];
            let br = t.bond_right;
            let new_vec: Vec<Complex64> = (0..br)
                .map(|beta| {
                    vec_data
                        .iter()
                        .enumerate()
                        .map(|(alpha, &v)| v * t.data[t.idx(alpha, bit, beta)])
                        .sum()
                })
                .collect();
            vec_data = new_vec;
        }
        vec_data[0]
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Rzz(_) => {
                let g = gate.matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1]);
            }
            Gate::Cx => {
                let g = cx_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1]);
            }
            Gate::Cz => {
                let g = cz_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1]);
            }
            Gate::Swap => {
                let g = swap_matrix_4x4();
                self.apply_two_qubit_gate(&g, targets[0], targets[1]);
            }
            Gate::Cu(mat) => {
                let g = cu_matrix_4x4(mat);
                self.apply_two_qubit_gate(&g, targets[0], targets[1]);
            }
            Gate::Mcu(data) => {
                let num_ctrl = data.num_controls as usize;
                let all_qubits: Vec<usize> = targets
                    .iter()
                    .map(|&qubit| self.site_for_logical(qubit))
                    .collect();
                let n = num_ctrl + 1;
                let dim = 1usize << n;
                let role_order: Vec<usize> = (0..n).collect();
                let gate_mat = mcu_matrix(num_ctrl, &data.mat, &role_order);
                self.apply_n_qubit_gate(&gate_mat, dim, &all_qubits);
            }
            Gate::BatchPhase(data) => {
                let control = self.site_for_logical(targets[0]);
                let phases: Vec<(usize, Complex64)> = data
                    .phases
                    .iter()
                    .map(|&(qubit, phase)| (self.site_for_logical(qubit), phase))
                    .collect();
                self.apply_batch_phase_bubble(control, &phases);
            }
            Gate::BatchRzz(data) => {
                for &(q0, q1, theta) in &data.edges {
                    let g = Gate::Rzz(theta).matrix_4x4();
                    self.apply_two_qubit_gate(&g, q0, q1);
                }
            }
            Gate::DiagonalBatch(data) => {
                for entry in &data.entries {
                    if let Some((q, mat)) = entry.as_1q_matrix() {
                        self.apply_single_qubit_gate(self.site_for_logical(q), &mat);
                    } else if let Some((q0, q1, mat)) = entry.as_2q_matrix() {
                        self.apply_two_qubit_gate(&mat, q0, q1);
                    }
                }
            }
            Gate::MultiFused(data) => {
                for &(target, ref mat) in &data.gates {
                    self.apply_single_qubit_gate(self.site_for_logical(target), mat);
                }
            }
            Gate::Fused2q(mat) => {
                self.apply_two_qubit_gate(mat, targets[0], targets[1]);
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in &data.gates {
                    self.apply_two_qubit_gate(mat, q0, q1);
                }
            }
            single_qubit => {
                let mat = single_qubit.matrix_2x2();
                self.apply_single_qubit_gate(self.site_for_logical(targets[0]), &mat);
            }
        }
    }
}

impl Backend for MpsBackend {
    fn name(&self) -> &'static str {
        "mps"
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.classical_bits = vec![false; num_classical_bits];
        self.sites = (0..num_qubits)
            .map(|_| SiteTensor::new_zero_state())
            .collect();
        self.logical_to_site = (0..num_qubits).collect();
        self.site_to_logical = (0..num_qubits).collect();
        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets),
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
                    self.dispatch_gate(gate, targets);
                }
            }
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(self.site_for_logical(qubit));
        Ok(())
    }

    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        Ok(self.reduced_density_site(self.site_for_logical(qubit)))
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        if self.num_qubits > MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "probabilities for {} qubits (max {})",
                    self.num_qubits, MAX_PROB_QUBITS
                ),
            });
        }

        let dim = 1usize << self.num_qubits;

        #[cfg(feature = "parallel")]
        if dim >= MIN_DIM_FOR_PAR {
            return Ok((0..dim)
                .into_par_iter()
                .map(|basis| self.chain_amplitude(basis).norm_sqr())
                .collect());
        }

        Ok((0..dim)
            .map(|basis| self.chain_amplitude(basis).norm_sqr())
            .collect())
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        if self.num_qubits > MAX_PROB_QUBITS {
            return Err(PrismError::BackendUnsupported {
                backend: self.name().to_string(),
                operation: format!(
                    "statevector export for {} qubits (max {})",
                    self.num_qubits, MAX_PROB_QUBITS
                ),
            });
        }

        let dim = 1usize << self.num_qubits;

        #[cfg(feature = "parallel")]
        if dim >= MIN_DIM_FOR_PAR {
            return Ok((0..dim)
                .into_par_iter()
                .map(|basis| self.chain_amplitude(basis))
                .collect());
        }

        Ok((0..dim).map(|basis| self.chain_amplitude(basis)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::sim;

    const EPS: f64 = 1e-10;

    fn run_mps(circuit: &Circuit) -> MpsBackend {
        let mut b = MpsBackend::new(42, 64);
        sim::run_on(&mut b, circuit).unwrap();
        b
    }

    fn run_mps_probs(circuit: &Circuit) -> Vec<f64> {
        let b = run_mps(circuit);
        b.probabilities().unwrap()
    }

    fn assert_probs_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < EPS, "prob[{i}]: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_svd_2x2() {
        let a = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let r = svd(&a, 2, 2);
        assert_eq!(r.s.len(), 2);
        assert!(r.s[0] >= r.s[1]);

        let mut recon = [ZERO; 4];
        for c in 0..2 {
            for row in 0..2 {
                for kk in 0..2 {
                    recon[c * 2 + row] += r.u[kk * r.u_rows + row]
                        * Complex64::new(r.s[kk], 0.0)
                        * r.vt[kk * r.vt_cols + c];
                }
            }
        }
        for i in 0..4 {
            assert!(
                (recon[i] - a[i]).norm() < 1e-10,
                "recon[{i}] = {:?}, expected {:?}",
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_svd_rank_deficient() {
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let r = svd(&a, 2, 2);
        assert!(r.s[1] < 1e-10, "second singular value should be ~0");
    }

    #[test]
    fn test_svd_identity() {
        let a = vec![ONE, ZERO, ZERO, ONE];
        let r = svd(&a, 2, 2);
        assert!((r.s[0] - 1.0).abs() < 1e-10);
        assert!((r.s[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_svd_wide_matrix() {
        let a = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 1.0),
        ];
        let r = svd(&a, 2, 3);
        assert_eq!(r.u_rows, 2);
        assert_eq!(r.vt_cols, 3);

        let mut recon = [ZERO; 6];
        for c in 0..3 {
            for row in 0..2 {
                for kk in 0..2 {
                    recon[c * 2 + row] += r.u[kk * r.u_rows + row]
                        * Complex64::new(r.s[kk], 0.0)
                        * r.vt[kk * r.vt_cols + c];
                }
            }
        }
        for i in 0..6 {
            assert!(
                (recon[i] - a[i]).norm() < 1e-10,
                "recon[{i}] = {:?}, expected {:?}",
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn test_init_zero_state() {
        let mut b = MpsBackend::new(42, 64);
        b.init(3, 0).unwrap();
        assert_eq!(b.sites.len(), 3);
        for s in &b.sites {
            assert_eq!(s.bond_left, 1);
            assert_eq!(s.bond_right, 1);
            assert_eq!(s.data.len(), 2);
            assert!((s.data[0] - ONE).norm() < EPS);
            assert!((s.data[1] - ZERO).norm() < EPS);
        }
    }

    #[test]
    fn test_x_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::X, &[0]);
        assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0]);
    }

    #[test]
    fn test_h_gate() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        assert_probs_close(&run_mps_probs(&c), &[0.5, 0.5]);
    }

    #[test]
    fn test_hh_is_identity() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[0]);
        assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0]);
    }

    #[test]
    fn test_rz_preserves_zero() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rz(1.234), &[0]);
        assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0]);
    }

    #[test]
    fn test_rx_pi() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(std::f64::consts::PI), &[0]);
        assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0]);
    }

    #[test]
    fn test_bell_state() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        assert_probs_close(&run_mps_probs(&c), &[0.5, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_bell_bond_dim() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let b = run_mps(&c);
        assert_eq!(b.sites[0].bond_right, 2);
        assert_eq!(b.sites[1].bond_left, 2);
    }

    #[test]
    fn test_cx_no_flip() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Cx, &[0, 1]);
        assert_probs_close(&run_mps_probs(&c), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cz_phase() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Cz, &[0, 1]);
        assert_probs_close(&run_mps_probs(&c), &[0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_swap() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::Swap, &[0, 1]);
        assert_probs_close(&run_mps_probs(&c), &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ghz_3() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        let probs = run_mps_probs(&c);
        assert_probs_close(&probs, &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_non_adjacent_cx() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::Cx, &[0, 2]);
        assert_probs_close(
            &run_mps_probs(&c),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        );
    }

    #[test]
    fn test_measure_deterministic() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 0);
        let b = run_mps(&c);
        assert!(b.classical_results()[0]);
    }

    #[test]
    fn test_measure_seeded() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        let b1 = run_mps(&c);
        let b2 = run_mps(&c);
        assert_eq!(b1.classical_results()[0], b2.classical_results()[0]);
    }

    #[test]
    fn test_fused_gate() {
        let h_mat = Gate::H.matrix_2x2();
        let t_mat = Gate::T.matrix_2x2();
        let mut fused = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    fused[i][j] += t_mat[i][k] * h_mat[k][j];
                }
            }
        }

        let mut c1 = Circuit::new(1, 0);
        c1.add_gate(Gate::H, &[0]);
        c1.add_gate(Gate::T, &[0]);
        let p1 = run_mps_probs(&c1);

        let mut c2 = Circuit::new(1, 0);
        c2.add_gate(Gate::Fused(Box::new(fused)), &[0]);
        let p2 = run_mps_probs(&c2);

        assert_probs_close(&p1, &p2);
    }

    #[test]
    fn test_supports_fused_gates() {
        let b = MpsBackend::new(42, 64);
        assert!(b.supports_fused_gates());
    }

    #[test]
    fn test_probabilities_cap() {
        let mut b = MpsBackend::new(42, 64);
        b.init(21, 0).unwrap();
        assert!(b.probabilities().is_err());
    }

    #[test]
    fn test_mcu_matrix_toffoli() {
        let x_mat = Gate::X.matrix_2x2();
        let order = vec![0, 1, 2]; // ctrl0, ctrl1, target — identity order
        let gate = mcu_matrix(2, &x_mat, &order);
        // 8×8 matrix: identity for states 0..5, then X on target for states 6,7
        // state 6 = |110⟩, state 7 = |111⟩ → swap these
        assert!((gate[6 * 8 + 6] - ZERO).norm() < 1e-12); // 6→6 should be 0
        assert!((gate[7 * 8 + 6] - ONE).norm() < 1e-12); // 6→7
        assert!((gate[6 * 8 + 7] - ONE).norm() < 1e-12); // 7→6
        assert!((gate[7 * 8 + 7] - ZERO).norm() < 1e-12); // 7→7 should be 0
                                                          // Diagonal entries for 0..5 should be 1
        for s in 0..6 {
            assert!((gate[s * 8 + s] - ONE).norm() < 1e-12, "state {s}");
        }
    }

    fn assert_mps_matches_statevector(circuit: &crate::circuit::Circuit) {
        use crate::backend::statevector::StatevectorBackend;

        let mut sv = StatevectorBackend::new(42);
        sv.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for inst in &circuit.instructions {
            sv.apply(inst).unwrap();
        }
        let sv_probs = sv.probabilities().unwrap();

        let mut mps = MpsBackend::new(42, 128);
        mps.init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        for inst in &circuit.instructions {
            mps.apply(inst).unwrap();
        }
        let mps_probs = mps.probabilities().unwrap();

        for (i, (a, b)) in sv_probs.iter().zip(&mps_probs).enumerate() {
            assert!((a - b).abs() < 1e-10, "prob[{i}]: sv={a}, mps={b}");
        }
    }

    #[test]
    fn test_toffoli_adjacent() {
        use crate::circuit::Circuit;
        use crate::gates::McuData;

        let x_mat = Gate::X.matrix_2x2();
        let mut c = Circuit::new(3, 0);
        // Set controls to |1⟩
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        // Toffoli: should flip target
        c.add_gate(
            Gate::Mcu(Box::new(McuData {
                mat: x_mat,
                num_controls: 2,
            })),
            &[0, 1, 2],
        );
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_toffoli_no_flip() {
        use crate::circuit::Circuit;
        use crate::gates::McuData;

        let x_mat = Gate::X.matrix_2x2();
        let mut c = Circuit::new(3, 0);
        // Only one control set — should NOT flip target
        c.add_gate(Gate::X, &[0]);
        c.add_gate(
            Gate::Mcu(Box::new(McuData {
                mat: x_mat,
                num_controls: 2,
            })),
            &[0, 1, 2],
        );
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_toffoli_non_adjacent() {
        use crate::circuit::Circuit;
        use crate::gates::McuData;

        let x_mat = Gate::X.matrix_2x2();
        let mut c = Circuit::new(5, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[2]);
        c.add_gate(
            Gate::Mcu(Box::new(McuData {
                mat: x_mat,
                num_controls: 2,
            })),
            &[0, 2, 4],
        );
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_cccx() {
        use crate::circuit::Circuit;
        use crate::gates::McuData;

        let x_mat = Gate::X.matrix_2x2();
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(Gate::X, &[2]);
        c.add_gate(
            Gate::Mcu(Box::new(McuData {
                mat: x_mat,
                num_controls: 3,
            })),
            &[0, 1, 2, 3],
        );
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_mcu_arbitrary_unitary() {
        use crate::circuit::Circuit;
        use crate::gates::McuData;

        let ry_mat = Gate::Ry(std::f64::consts::FRAC_PI_4).matrix_2x2();
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::X, &[1]);
        c.add_gate(
            Gate::Mcu(Box::new(McuData {
                mat: ry_mat,
                num_controls: 2,
            })),
            &[0, 1, 2],
        );
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_non_adjacent_layout_tracks_logical_targets() {
        let mut c = Circuit::new(6, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::X, &[5]);
        c.add_gate(Gate::Cx, &[0, 5]);
        c.add_gate(Gate::Ry(0.37), &[0]);
        c.add_gate(Gate::Rz(-0.52), &[5]);
        c.add_gate(Gate::Swap, &[0, 3]);
        c.add_gate(Gate::S, &[3]);
        c.add_gate(Gate::Cx, &[1, 4]);
        c.add_gate(Gate::H, &[4]);
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_measure_after_non_adjacent_routing_uses_logical_qubit() {
        let mut c = Circuit::new(5, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::Cx, &[0, 4]);
        c.add_measure(4, 0);
        assert_mps_matches_statevector(&c);

        let b = run_mps(&c);
        assert_eq!(b.classical_results(), &[true]);
    }

    #[test]
    fn test_reset_after_non_adjacent_routing_uses_logical_qubit() {
        let mut c = Circuit::new(5, 0);
        c.add_gate(Gate::X, &[0]);
        c.add_gate(Gate::Cx, &[0, 4]);
        c.add_reset(0);
        c.add_gate(Gate::H, &[4]);
        assert_mps_matches_statevector(&c);
    }

    #[test]
    fn test_batch_phase_decomposition() {
        use crate::circuit::Circuit;
        use crate::gates::BatchPhaseData;

        let phase1 = Complex64::from_polar(1.0, 0.5);
        let phase2 = Complex64::from_polar(1.0, 1.2);

        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(
            Gate::BatchPhase(Box::new(BatchPhaseData {
                phases: smallvec::smallvec![(1, phase1), (2, phase2)],
            })),
            &[0, 1, 2],
        );
        assert_mps_matches_statevector(&c);
    }
}
