//! Thin SVD and thin QR for the MPS chain, plus the site-fill helpers the
//! two-site update and the gauge moves factorize through.

use num_complex::Complex64;

use super::{JACOBI_REL_TOL, MAX_SVD_SWEEPS, ONE, SVD_RESOLUTION, ZERO};
use crate::backend::NORM_CLAMP_MIN;

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

/// Return a length-`len` window of `buf`, growing it only past its high-water
/// mark, so a steady-state call allocates and clears nothing. Callers
/// overwrite every element of the window before reading it.
pub(super) fn scratch_slice(buf: &mut Vec<Complex64>, len: usize) -> &mut [Complex64] {
    if buf.len() < len {
        buf.resize(len, ZERO);
    }
    &mut buf[..len]
}

/// Orthogonalization passes a column gets before it is accepted or dropped.
const QR_MAX_PASSES: usize = 3;
/// A pass leaving the residual above this fraction of its norm before the
/// pass has cancelled nothing significant, so the direction has settled.
const QR_PASS_RETAIN: f64 = 0.5;
/// Relative floor below which a column adds no direction to `Q`: what is left
/// of it is the rounding noise of the columns before it, and a normalized
/// noise vector is not orthogonal to them.
const QR_RANK_REL_TOL: f64 = 1e-14;
/// Thin QR factorization: after [`ThinQr::factorize`], `q` holds `rank`
/// orthonormal columns of length `rows` in column-major order, `r` is `rank`
/// by `cols` in row-major order, and `q · r` reproduces the input. The
/// buffers are grow-only scratch held across steps, so what lies past the
/// stated extent is stale.
#[derive(Default)]
pub(super) struct ThinQr {
    pub(super) q: Vec<Complex64>,
    pub(super) r: Vec<Complex64>,
    v: Vec<Complex64>,
    pub(super) rank: usize,
}

pub(super) fn l2_norm(v: &[Complex64]) -> f64 {
    v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt()
}

impl ThinQr {
    /// Factorize a column-major `rows` by `cols` matrix by modified
    /// Gram-Schmidt, reorthogonalizing a column until a pass stops shrinking
    /// it.
    ///
    /// A gauge move wants the isometry rather than the spectrum, and one
    /// Gram-Schmidt pass with reorthogonalization reaches it at rounding where
    /// [`svd`] sweeps to convergence.
    pub(super) fn factorize(&mut self, a: &[Complex64], rows: usize, cols: usize) {
        let rank_cap = rows.min(cols);
        let q = scratch_slice(&mut self.q, rows * rank_cap);
        let r = scratch_slice(&mut self.r, rank_cap * cols);
        let v = scratch_slice(&mut self.v, rows);
        r.fill(ZERO);
        let mut rank = 0usize;

        for j in 0..cols {
            v.copy_from_slice(&a[j * rows..j * rows + rows]);
            let column_norm = l2_norm(v);
            let mut previous = column_norm;
            let mut residual = column_norm;
            for _ in 0..QR_MAX_PASSES {
                for i in 0..rank {
                    let basis = &q[i * rows..i * rows + rows];
                    // Four accumulators: one would be a loop-carried floating
                    // point chain the compiler cannot reassociate.
                    let mut acc = [ZERO; 4];
                    let mut basis_chunks = basis.chunks_exact(4);
                    let mut v_chunks = v.chunks_exact(4);
                    for (b4, x4) in basis_chunks.by_ref().zip(v_chunks.by_ref()) {
                        for k in 0..4 {
                            acc[k] += b4[k].conj() * x4[k];
                        }
                    }
                    for (b, x) in basis_chunks.remainder().iter().zip(v_chunks.remainder()) {
                        acc[0] += b.conj() * x;
                    }
                    let dot = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                    for (b, x) in basis.iter().zip(v.iter_mut()) {
                        *x -= dot * b;
                    }
                    r[i * cols + j] += dot;
                }
                residual = l2_norm(v);
                if residual > QR_PASS_RETAIN * previous {
                    break;
                }
                previous = residual;
            }

            if rank < rank_cap && residual > QR_RANK_REL_TOL * column_norm {
                let inv = 1.0 / residual;
                for (out, x) in q[rank * rows..rank * rows + rows].iter_mut().zip(v.iter()) {
                    *out = x * inv;
                }
                r[rank * cols + j] = Complex64::new(residual, 0.0);
                rank += 1;
            }
        }

        if rank == 0 {
            // A numerically zero matrix offers no direction to keep. The basis
            // column keeps `q` an isometry and leaves `r` zero, so the product
            // is still zero.
            q[..rows].fill(ZERO);
            q[0] = ONE;
            rank = 1;
        }

        self.rank = rank;
    }
}

/// Most singular values a two-site cut carries. The pair bonds give
/// `2 * bl.min(br)`, and the middle bond gives `4 * bond_mid`, because the
/// contraction factors through it at rank `bond_mid` and a 4x4 gate is a sum of
/// at most four products that each lift the rank by one factor.
pub(super) fn cut_rank(bond_left: usize, bond_mid: usize, bond_right: usize) -> u128 {
    (2 * bond_left.min(bond_right)).min(4 * bond_mid) as u128
}

/// Most singular values any cut of an `n`-site block decomposition carries:
/// the cut after site `k` splits `(bond_left, 2^n, bond_right)` into
/// `bond_left * 2^(k+1)` rows against `bond_right * 2^(n-k-1)` columns.
pub(super) fn widest_block_cut(bond_left: usize, bond_right: usize, n: usize) -> u128 {
    (0..n - 1)
        .map(|k| ((bond_left as u128) << (k + 1)).min((bond_right as u128) << (n - 1 - k)))
        .max()
        .unwrap_or(0)
}

/// Whether keeping `chi` of `singular_values` drops one above
/// [`SVD_RESOLUTION`] of the largest.
pub(super) fn discards_resolvable_weight(singular_values: &[f64], chi: usize) -> bool {
    match (singular_values.first(), singular_values.get(chi)) {
        (Some(&s_max), Some(&next)) => next > SVD_RESOLUTION * s_max,
        _ => false,
    }
}

pub(super) fn truncated_svd_rank(
    singular_values: &[f64],
    epsilon: f64,
    max_bond_dim: usize,
) -> usize {
    let s_max = singular_values.first().copied().unwrap_or(0.0);
    let threshold = epsilon * s_max;
    singular_values
        .iter()
        .take_while(|&&s| s > threshold)
        .count()
        .max(1)
        .min(max_bond_dim)
}

/// Reshape `chi` orthonormal columns of length `stride` into a site of shape
/// `(bond_left, 2, chi)`, whose row index packs `alpha * 2 + i` and whose
/// column index is the site's right bond.
///
/// Sized rather than cleared, here and in the three helpers below: every
/// element is written, so a buffer handed back by a site it is about to
/// overwrite costs neither an allocation nor a memset.
pub(super) fn fill_isometry_site_data(
    out: &mut Vec<Complex64>,
    columns: &[Complex64],
    stride: usize,
    bond_left: usize,
    chi: usize,
) {
    out.resize(bond_left * 2 * chi, ZERO);
    for alpha in 0..bond_left {
        for i in 0..2 {
            let r = alpha * 2 + i;
            for gamma in 0..chi {
                out[alpha * (2 * chi) + i * chi + gamma] = columns[gamma * stride + r];
            }
        }
    }
}

pub(super) fn fill_scaled_vt_data(
    out: &mut Vec<Complex64>,
    svd_result: &SvdResult,
    chi: usize,
    physical_dim: usize,
    bond_right: usize,
) {
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

/// Which site of an adjacent pair carries the singular weight after a
/// two-site update. The partner comes out an isometry, so the weight site is
/// the orthogonality center the update leaves behind.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum WeightSide {
    Left,
    Right,
}

/// Write `U · diag(S)` into a site of shape `(bond_left, 2, chi)`, the mirror
/// of [`fill_scaled_vt_data`].
pub(super) fn fill_scaled_u_site_data(
    out: &mut Vec<Complex64>,
    svd_result: &SvdResult,
    bond_left: usize,
    chi: usize,
) {
    out.resize(bond_left * 2 * chi, ZERO);
    for alpha in 0..bond_left {
        for i in 0..2 {
            let r = alpha * 2 + i;
            for gamma in 0..chi {
                out[alpha * (2 * chi) + i * chi + gamma] =
                    svd_result.u[gamma * svd_result.u_rows + r] * svd_result.s[gamma];
            }
        }
    }
}

/// Write the first `chi` rows of V† into a site of shape
/// `(chi, physical_dim, bond_right)`. The rows are orthonormal, so the site
/// carries no weight, the mirror of [`fill_isometry_site_data`].
pub(super) fn fill_vt_site_data(
    out: &mut Vec<Complex64>,
    svd_result: &SvdResult,
    chi: usize,
    physical_dim: usize,
    bond_right: usize,
) {
    let row_len = physical_dim * bond_right;
    out.resize(chi * row_len, ZERO);
    for gamma in 0..chi {
        let src = gamma * svd_result.vt_cols;
        out[gamma * row_len..gamma * row_len + row_len]
            .copy_from_slice(&svd_result.vt[src..src + row_len]);
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

    for _sweep in 0..MAX_SVD_SWEEPS {
        let mut rotated = false;

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

                if g_pq.norm_sqr() <= JACOBI_REL_TOL * JACOBI_REL_TOL * g_pp * g_qq {
                    continue;
                }
                rotated = true;

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

        if !rotated {
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
