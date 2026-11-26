//! Thin SVD and thin QR for the MPS chain, plus the site-fill helpers the
//! two-site update and the gauge moves factorize through.

use num_complex::Complex64;

use super::{ONE, SVD_RESOLUTION, ZERO};
use crate::backend::NORM_CLAMP_MIN;

const MAX_SVD_SWEEPS: usize = 100;
/// Relative pair threshold of the one-sided Jacobi sweep: a pair rotates while
/// `|a_p^H a_q|` exceeds this times `|a_p| |a_q|`. Judged against the pair's
/// own norms rather than the whole matrix, so the columns of `U` come out
/// orthogonal on the small singular values too; an absolute residual left
/// them 0.56 apart on an 8x8 whose values span six decades.
const JACOBI_REL_TOL: f64 = 1e-15;

/// Thin SVD result: A = U · diag(S) · V†
#[doc(hidden)]
#[derive(Default)]
pub struct SvdResult {
    pub u: Vec<Complex64>,
    pub u_rows: usize,
    pub s: Vec<f64>,
    pub vt: Vec<Complex64>,
    pub vt_cols: usize,
}

/// Grow-only scratch and result buffers for [`svd_with_workspace`] and
/// [`svd_singular_values_with_workspace`], kept across gates so steady-state
/// cuts allocate nothing.
#[derive(Default)]
pub(super) struct SvdWorkspace {
    pub(super) result: SvdResult,
    #[cfg(feature = "parallel")]
    s_cplx: Vec<Complex64>,
    #[cfg(feature = "parallel")]
    probe_x: Vec<Complex64>,
    #[cfg(feature = "parallel")]
    probe_residual: Vec<Complex64>,
    jacobi_work: Vec<Complex64>,
    jacobi_v: Vec<Complex64>,
    jacobi_s: Vec<f64>,
    jacobi_u: Vec<Complex64>,
    jacobi_order: Vec<usize>,
    #[cfg(feature = "parallel")]
    faer_scratch: Option<faer::dyn_stack::MemBuffer>,
}

#[cfg(feature = "parallel")]
fn ensure_mem_stack(
    slot: &mut Option<faer::dyn_stack::MemBuffer>,
    req: faer::dyn_stack::StackReq,
) -> &mut faer::dyn_stack::MemStack {
    use faer::dyn_stack::{MemBuffer, MemStack};
    let needs_alloc = slot
        .as_mut()
        .is_none_or(|buf| !MemStack::new(buf).can_hold(req));
    if needs_alloc {
        *slot = Some(MemBuffer::new(req));
    }
    MemStack::new(slot.as_mut().expect("buffer initialized above"))
}

/// Thin SVD through faer when the `parallel` feature is on and `m * n >= 256`,
/// one-sided Jacobi otherwise.
#[doc(hidden)]
pub fn svd(a: &[Complex64], m: usize, n: usize) -> SvdResult {
    let mut ws = SvdWorkspace::default();
    svd_with_workspace(&mut ws, a, m, n);
    ws.result
}

/// Thin SVD into `ws.result`, reusing `ws`'s scratch allocations.
pub(super) fn svd_with_workspace(ws: &mut SvdWorkspace, a: &[Complex64], m: usize, n: usize) {
    #[cfg(feature = "parallel")]
    if m * n >= 256 {
        svd_faer_into(ws, a, m, n);
        return;
    }
    svd_jacobi_into(ws, a, m, n, true);
}

/// Singular values only into `ws.result.s`, skipping singular vector computation.
///
/// Uses bidiagonal QR iteration (`divide_and_conquer = false`) with
/// `ComputeSvdVectors::No`: without singular vectors `probe_residual` cannot
/// guard against faer divide-and-conquer error (`~1e-8`) on rank-deficient
/// graded spectra, and bidiagonal QR iteration without vector accumulation
/// runs in `O(k^2)`.
pub(crate) fn svd_singular_values_with_workspace<'a>(
    ws: &'a mut SvdWorkspace,
    a: &[Complex64],
    m: usize,
    n: usize,
) -> &'a [f64] {
    #[cfg(feature = "parallel")]
    if m * n >= 256 && faer_svd_core(ws, a, m, n, false, false) {
        return &ws.result.s;
    }
    svd_jacobi_into(ws, a, m, n, false);
    &ws.result.s
}

/// Singular values only of column-major `a` (`m` by `n`), in descending order.
pub(crate) fn svd_singular_values(a: &[Complex64], m: usize, n: usize) -> Vec<f64> {
    let mut ws = SvdWorkspace::default();
    svd_singular_values_with_workspace(&mut ws, a, m, n);
    ws.result.s
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
/// Element count from which [`ThinQr`] factorizes through faer's blocked
/// Householder QR instead of Gram-Schmidt. Swept on `mps/brickwork_d24`: the
/// 64 by 256 class and up gains 7% to 9% on the cap-256 rows, while 128 by 64
/// (8192) sent to faer costs 9% to 15% on the cap-64 rows.
#[cfg(feature = "parallel")]
const QR_FAER_MIN_ELEMS: usize = 16384;
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
    #[cfg(feature = "parallel")]
    qr_work: Vec<Complex64>,
    #[cfg(feature = "parallel")]
    q_coeff: Vec<Complex64>,
    #[cfg(feature = "parallel")]
    faer_scratch: Option<faer::dyn_stack::MemBuffer>,
    pub(super) rank: usize,
    /// Squared weight the last [`ThinQr::factorize`] left with the columns it
    /// dropped, relative to the whole input. A column is dropped only where
    /// the columns before it already span it, so this reads the factorization's
    /// own rounding. Test-only: the walk runs one to three times per two-qubit
    /// gate and no shipped path reads this.
    #[cfg(test)]
    pub(super) discarded: f64,
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
        #[cfg(feature = "parallel")]
        if rows * cols >= QR_FAER_MIN_ELEMS {
            self.factorize_faer(a, rows, cols);
            return;
        }
        self.factorize_gram_schmidt(a, rows, cols)
    }

    /// Blocked Householder QR through faer, for the shapes where Gram-Schmidt's
    /// column-at-a-time projections lose to a blocked reflector sweep.
    #[cfg(feature = "parallel")]
    fn factorize_faer(&mut self, a: &[Complex64], rows: usize, cols: usize) {
        use faer::dyn_stack::StackReq;
        use faer::linalg::householder::{
            apply_block_householder_sequence_on_the_left_in_place_scratch,
            apply_block_householder_sequence_on_the_left_in_place_with_conj,
        };
        use faer::linalg::qr::no_pivoting::factor::{
            qr_in_place, qr_in_place_scratch, recommended_block_size,
        };
        use faer::{Conj, MatMut, MatRef};

        let k = rows.min(cols);
        let block_size = recommended_block_size::<Complex64>(rows, cols);
        let par = faer::get_global_parallelism();

        let qr_work = scratch_slice(&mut self.qr_work, rows * cols);
        qr_work.copy_from_slice(&a[..rows * cols]);
        let q_coeff = scratch_slice(&mut self.q_coeff, block_size * k);
        q_coeff.fill(ZERO);

        let req = StackReq::any_of(&[
            qr_in_place_scratch::<Complex64>(rows, cols, block_size, par, Default::default()),
            apply_block_householder_sequence_on_the_left_in_place_scratch::<Complex64>(
                rows, block_size, k,
            ),
        ]);
        let stack = ensure_mem_stack(&mut self.faer_scratch, req);

        qr_in_place(
            MatMut::from_column_major_slice_mut(qr_work, rows, cols),
            MatMut::from_column_major_slice_mut(q_coeff, block_size, k),
            par,
            stack,
            Default::default(),
        );

        let r = scratch_slice(&mut self.r, k * cols);
        r.fill(ZERO);
        for (j, col) in qr_work[..rows * cols].chunks_exact(rows).enumerate() {
            for (i, &val) in col.iter().take(k.min(j + 1)).enumerate() {
                r[i * cols + j] = val;
            }
        }

        let q = scratch_slice(&mut self.q, rows * k);
        q.fill(ZERO);
        for i in 0..k {
            q[i * rows + i] = ONE;
        }
        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            MatRef::from_column_major_slice(qr_work, rows, cols).subcols(0, k),
            MatRef::from_column_major_slice(q_coeff, block_size, k),
            Conj::No,
            MatMut::from_column_major_slice_mut(q, rows, k),
            par,
            stack,
        );

        self.rank = k;
        #[cfg(test)]
        {
            self.discarded = 0.0;
        }
    }

    fn factorize_gram_schmidt(&mut self, a: &[Complex64], rows: usize, cols: usize) {
        let rank_cap = rows.min(cols);
        let q = scratch_slice(&mut self.q, rows * rank_cap);
        let r = scratch_slice(&mut self.r, rank_cap * cols);
        let v = scratch_slice(&mut self.v, rows);
        r.fill(ZERO);
        let mut rank = 0usize;
        #[cfg(test)]
        let (mut total, mut dropped) = (0.0f64, 0.0f64);

        for j in 0..cols {
            v.copy_from_slice(&a[j * rows..j * rows + rows]);
            let column_norm = l2_norm(v);
            #[cfg(test)]
            {
                total += column_norm * column_norm;
            }
            let mut previous = column_norm;
            let mut residual = column_norm;
            for _ in 0..QR_MAX_PASSES {
                for i in 0..rank {
                    let basis = &q[i * rows..i * rows + rows];
                    // Four accumulators: one would be a loop-carried floating
                    // point chain the compiler cannot reassociate.
                    let mut acc = [ZERO; 4];
                    let (basis_quads, basis_rest) = basis.as_chunks::<4>();
                    let (v_quads, v_rest) = v.as_chunks::<4>();
                    for (b4, x4) in basis_quads.iter().zip(v_quads) {
                        for k in 0..4 {
                            acc[k] += b4[k].conj() * x4[k];
                        }
                    }
                    for (b, x) in basis_rest.iter().zip(v_rest) {
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
            } else {
                #[cfg(test)]
                {
                    dropped += residual * residual;
                }
            }
        }
        #[cfg(test)]
        {
            self.discarded = if total > 0.0 { dropped / total } else { 0.0 };
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
    let mut ws = SvdWorkspace::default();
    svd_jacobi_into(&mut ws, a, m, n, true);
    ws.result
}

fn svd_jacobi_into(
    ws: &mut SvdWorkspace,
    a: &[Complex64],
    m: usize,
    n: usize,
    compute_vectors: bool,
) {
    let k = m.min(n);
    if k == 0 {
        ws.result.s.clear();
        if compute_vectors {
            ws.result.u.clear();
            ws.result.vt.clear();
            ws.result.u_rows = m;
            ws.result.vt_cols = n;
        }
        return;
    }

    let transpose = m < n;
    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };

    ws.jacobi_work.resize(work_m * work_n, ZERO);
    let work = &mut ws.jacobi_work[..work_m * work_n];
    if transpose {
        for col_b in 0..work_n {
            for row_b in 0..work_m {
                work[col_b * work_m + row_b] = a[row_b * m + col_b].conj();
            }
        }
    } else {
        work.copy_from_slice(&a[..work_m * work_n]);
    }

    if compute_vectors {
        ws.jacobi_v.resize(work_n * work_n, ZERO);
        ws.jacobi_v.fill(ZERO);
        for i in 0..work_n {
            ws.jacobi_v[i * work_n + i] = ONE;
        }
    }
    let v = &mut ws.jacobi_v;

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

                if compute_vectors {
                    for r in 0..work_n {
                        let vp = v[p * work_n + r];
                        let vq = v[q * work_n + r];
                        v[p * work_n + r] = c_cx * vp + s_cx * phase_conj * vq;
                        v[q * work_n + r] = -s_cx * vp + c_cx * phase_conj * vq;
                    }
                }
            }
        }

        if !rotated {
            break;
        }
    }

    ws.jacobi_s.resize(work_n, 0.0);
    if compute_vectors {
        ws.jacobi_u.resize(work_m * work_n, ZERO);
        ws.jacobi_u.fill(ZERO);
    }

    for j in 0..work_n {
        let mut norm_sq = 0.0f64;
        for r in 0..work_m {
            norm_sq += work[j * work_m + r].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        ws.jacobi_s[j] = norm;
        if compute_vectors && norm > NORM_CLAMP_MIN {
            let inv_norm = 1.0 / norm;
            for r in 0..work_m {
                ws.jacobi_u[j * work_m + r] = work[j * work_m + r] * inv_norm;
            }
        }
    }

    ws.jacobi_order.clear();
    ws.jacobi_order.extend(0..work_n);
    ws.jacobi_order
        .sort_by(|&a, &b| ws.jacobi_s[b].partial_cmp(&ws.jacobi_s[a]).unwrap());

    ws.result.s.resize(k, 0.0);
    for (new_idx, &old_idx) in ws.jacobi_order.iter().take(k).enumerate() {
        ws.result.s[new_idx] = ws.jacobi_s[old_idx];
    }

    if !compute_vectors {
        return;
    }

    ws.result.u_rows = m;
    ws.result.vt_cols = n;
    ws.result.u.resize(m * k, ZERO);
    ws.result.vt.resize(k * n, ZERO);

    for (new_idx, &old_idx) in ws.jacobi_order.iter().take(k).enumerate() {
        if transpose {
            for r in 0..work_n {
                ws.result.u[new_idx * m + r] = ws.jacobi_v[old_idx * work_n + r];
            }
            for r in 0..work_m {
                ws.result.vt[new_idx * n + r] = ws.jacobi_u[old_idx * work_m + r].conj();
            }
        } else {
            for r in 0..work_m {
                ws.result.u[new_idx * work_m + r] = ws.jacobi_u[old_idx * work_m + r];
            }
            for r in 0..work_n {
                ws.result.vt[new_idx * work_n + r] = ws.jacobi_v[old_idx * work_n + r].conj();
            }
        }
    }
}

/// Compute thin SVD using the faer library (SIMD-accelerated bidiag + D&C).
///
/// faer's divide-and-conquer stage can return `Ok` on a rank-deficient input
/// with NaN singular vectors, or with finite ones that miss the input by 1e-8,
/// so output that is non-finite or fails a probe product is recomputed by QR
/// iteration on the bidiagonal, and by [`svd_jacobi`] if that fails too.
#[cfg(feature = "parallel")]
#[doc(hidden)]
pub fn svd_faer(a: &[Complex64], m: usize, n: usize) -> SvdResult {
    let mut ws = SvdWorkspace::default();
    svd_faer_into(&mut ws, a, m, n);
    ws.result
}

#[cfg(feature = "parallel")]
fn svd_faer_into(ws: &mut SvdWorkspace, a: &[Complex64], m: usize, n: usize) {
    if faer_svd_core(ws, a, m, n, true, true) {
        return;
    }
    if faer_svd_core(ws, a, m, n, false, true) {
        return;
    }
    svd_jacobi_into(ws, a, m, n, true);
}

#[cfg(feature = "parallel")]
fn faer_svd_core(
    ws: &mut SvdWorkspace,
    a: &[Complex64],
    m: usize,
    n: usize,
    divide_and_conquer: bool,
    compute_vectors: bool,
) -> bool {
    use faer::diag::DiagMut;
    use faer::linalg::svd::{self as faer_svd, ComputeSvdVectors, SvdParams};
    use faer::{Auto, MatMut, MatRef};

    let k = m.min(n);
    let par = faer::get_global_parallelism();
    let mut params = <SvdParams as Auto<Complex64>>::auto();
    if !divide_and_conquer {
        params.recursion_threshold = usize::MAX;
    }
    let vec_mode = if compute_vectors {
        ComputeSvdVectors::Thin
    } else {
        ComputeSvdVectors::No
    };
    let req = faer_svd::svd_scratch::<Complex64>(m, n, vec_mode, vec_mode, par, params.into());
    let stack = ensure_mem_stack(&mut ws.faer_scratch, req);

    ws.s_cplx.resize(k, ZERO);
    ws.result.s.resize(k, 0.0);

    let ok = if compute_vectors {
        ws.result.u_rows = m;
        ws.result.vt_cols = n;
        ws.result.u.resize(m * k, ZERO);
        ws.result.vt.resize(k * n, ZERO);
        faer_svd::svd(
            MatRef::from_column_major_slice(a, m, n),
            DiagMut::from_slice_mut(&mut ws.s_cplx[..k]),
            Some(MatMut::from_column_major_slice_mut(
                &mut ws.result.u[..m * k],
                m,
                k,
            )),
            // faer writes column-major V (n x k); indexing (j, i) at i * n + j matches
            // row-major V^H (k x n) once conjugated in place below.
            Some(MatMut::from_column_major_slice_mut(
                &mut ws.result.vt[..k * n],
                n,
                k,
            )),
            par,
            stack,
            params.into(),
        )
        .is_ok()
    } else {
        faer_svd::svd(
            MatRef::from_column_major_slice(a, m, n),
            DiagMut::from_slice_mut(&mut ws.s_cplx[..k]),
            None,
            None,
            par,
            stack,
            params.into(),
        )
        .is_ok()
    };

    if !ok {
        return false;
    }

    for (dst, src) in ws.result.s[..k].iter_mut().zip(&ws.s_cplx[..k]) {
        *dst = src.re;
    }
    if !ws.result.s.iter().all(|x| x.is_finite()) {
        return false;
    }

    if !compute_vectors {
        return true;
    }

    for z in &mut ws.result.vt[..k * n] {
        *z = z.conj();
    }

    let finite = |z: &Complex64| z.re.is_finite() && z.im.is_finite();
    if !(ws.result.u.iter().all(finite) && ws.result.vt.iter().all(finite)) {
        return false;
    }

    probe_residual_with_scratch(&mut ws.probe_x, &mut ws.probe_residual, a, &ws.result)
        <= SVD_PROBE_TOL
}

/// Largest probe residual a faer factorization may leave before the next
/// fallback recomputes it. Sound factorizations up to 512 wide read under 2e-15;
/// faer's divide-and-conquer on rank-64 cuts 128 and 256 wide reads 5e-9 to 1e-7.
#[cfg(feature = "parallel")]
const SVD_PROBE_TOL: f64 = 1e-12;

/// `|A x - U S V^H x| / (|A|_F |x|)` for a fixed unit-modulus probe `x`: one
/// matrix-vector product, about `1 / k` of the factorization it checks.
#[cfg(feature = "parallel")]
fn probe_residual_with_scratch(
    probe_x: &mut Vec<Complex64>,
    residual: &mut Vec<Complex64>,
    a: &[Complex64],
    r: &SvdResult,
) -> f64 {
    const GOLDEN_ANGLE: f64 = 2.399_963_229_728_653;
    let (m, n) = (r.u_rows, r.vt_cols);
    if m == 0 || n == 0 {
        return 0.0;
    }
    if probe_x.len() < n {
        let start = probe_x.len();
        probe_x.extend((start..n).map(|j| Complex64::from_polar(1.0, j as f64 * GOLDEN_ANGLE)));
    }
    let x = &probe_x[..n];
    residual.clear();
    residual.resize(m, ZERO);
    let mut a_norm_sqr = 0.0;
    for (col, &xj) in a.chunks_exact(m).zip(x) {
        for (out, &aij) in residual.iter_mut().zip(col) {
            *out += aij * xj;
            a_norm_sqr += aij.norm_sqr();
        }
    }
    for (g, &sg) in r.s.iter().enumerate() {
        let row = &r.vt[g * n..(g + 1) * n];
        let y = row.iter().zip(x).map(|(v, x)| v * x).sum::<Complex64>() * sg;
        for (out, &ug) in residual.iter_mut().zip(&r.u[g * m..(g + 1) * m]) {
            *out -= ug * y;
        }
    }
    let scale = (a_norm_sqr * n as f64).sqrt();
    if scale > 0.0 {
        l2_norm(residual) / scale
    } else {
        0.0
    }
}
