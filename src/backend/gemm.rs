//! Row-major complex matrix products for the MPS and tensor network backends: faer
//! at or above `MIN_FAER_GEMM_WORK`, scalar loops below it.

use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
#[cfg(feature = "parallel")]
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// `m * k * n` at or above which a product multiplies through faer rather than
/// the scalar loops: a 64-cubed complex product, which one pass of an MPS environment
/// step clears at bond 64 and not at 32. Routing every tensor network contraction to
/// faer instead costs 23% on `tn/scalar_depth_20q/4`, where the operands are too small
/// to cover its packing.
#[cfg(feature = "parallel")]
pub(crate) const MIN_FAER_GEMM_WORK: usize = 1 << 18;

/// `out = xᵀ · y` over row-major `x` (`r` by `m`), `y` (`r` by `n`) and
/// `out` (`m` by `n`), with `y` conjugated when `CONJ_Y`.
pub(crate) fn gemm_tn<const CONJ_Y: bool>(
    out: &mut [Complex64],
    x: &[Complex64],
    y: &[Complex64],
    r: usize,
    m: usize,
    n: usize,
) {
    debug_assert_eq!((x.len(), y.len(), out.len()), (r * m, r * n, m * n));
    if m == 0 || n == 0 {
        return;
    }
    if r == 0 {
        out.fill(ZERO);
        return;
    }
    #[cfg(feature = "parallel")]
    if r * m * n >= MIN_FAER_GEMM_WORK {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        let out = MatMut::from_column_major_slice_mut(out, n, m);
        let y = MatRef::from_column_major_slice(y, n, r);
        let x = MatRef::from_column_major_slice(x, m, r).transpose();
        if CONJ_Y {
            matmul(out, Accum::Replace, y.conjugate(), x, ONE, Par::rayon(0));
        } else {
            matmul(out, Accum::Replace, y, x, ONE, Par::rayon(0));
        }
        return;
    }
    out.fill(ZERO);
    for (x_row, y_row) in x.chunks_exact(m).zip(y.chunks_exact(n)) {
        for (&xv, out_row) in x_row.iter().zip(out.chunks_exact_mut(n)) {
            for (o, &yv) in out_row.iter_mut().zip(y_row) {
                *o += xv * if CONJ_Y { yv.conj() } else { yv };
            }
        }
    }
}

/// `out = x · y` over row-major `x` (`m` by `k`), `y` (`k` by `n`) and `out`
/// (`m` by `n`).
pub(crate) fn gemm_nn(
    out: &mut [Complex64],
    x: &[Complex64],
    y: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!((x.len(), y.len(), out.len()), (m * k, k * n, m * n));
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        out.fill(ZERO);
        return;
    }
    #[cfg(feature = "parallel")]
    if m * k * n >= MIN_FAER_GEMM_WORK {
        faer_gemm_nn(out, x, y, m, k, n);
        return;
    }
    out.fill(ZERO);
    for (out_row, x_row) in out.chunks_exact_mut(n).zip(x.chunks_exact(k)) {
        for (&xv, y_row) in x_row.iter().zip(y.chunks_exact(n)) {
            for (o, &yv) in out_row.iter_mut().zip(y_row) {
                *o += xv * yv;
            }
        }
    }
}

/// [`gemm_nn`] through faer at any size, overwriting `out`.
///
/// A row-major array is its own transpose read column-major, so `out = x · y` is
/// issued as `outᵀ = yᵀ · xᵀ` over the same buffers with no repacking.
#[cfg(feature = "parallel")]
pub(crate) fn faer_gemm_nn(
    out: &mut [Complex64],
    x: &[Complex64],
    y: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, MatMut, MatRef, Par};
    matmul(
        MatMut::from_column_major_slice_mut(out, n, m),
        Accum::Replace,
        MatRef::from_column_major_slice(y, n, k),
        MatRef::from_column_major_slice(x, k, m),
        ONE,
        Par::rayon(0),
    );
}

/// `out = x · yᴴ` over row-major `x` (`m` by `k`), `y` (`n` by `k`) and `out`
/// (`m` by `n`).
pub(crate) fn gemm_nh(
    out: &mut [Complex64],
    x: &[Complex64],
    y: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert_eq!((x.len(), y.len(), out.len()), (m * k, n * k, m * n));
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        out.fill(ZERO);
        return;
    }
    #[cfg(feature = "parallel")]
    if m * k * n >= MIN_FAER_GEMM_WORK {
        use faer::linalg::matmul::matmul;
        use faer::{Accum, MatMut, MatRef, Par};
        matmul(
            MatMut::from_column_major_slice_mut(out, n, m),
            Accum::Replace,
            MatRef::from_column_major_slice(y, k, n).adjoint(),
            MatRef::from_column_major_slice(x, k, m),
            ONE,
            Par::rayon(0),
        );
        return;
    }
    for (out_row, x_row) in out.chunks_exact_mut(n).zip(x.chunks_exact(k)) {
        for (o, y_row) in out_row.iter_mut().zip(y.chunks_exact(k)) {
            *o = x_row
                .iter()
                .zip(y_row)
                .fold(ZERO, |acc, (&xv, &yv)| acc + xv * yv.conj());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn random_vec(rng: &mut ChaCha8Rng, len: usize) -> Vec<Complex64> {
        (0..len)
            .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
            .collect()
    }

    #[test]
    fn gemm_zero_dimensions_do_not_panic_and_zero_fill_empty_inner() {
        let sentinel = Complex64::new(9.0, -3.0);

        let mut empty = [];
        let z12 = [ZERO; 12];
        gemm_nn(&mut empty, &[], &z12, 0, 3, 4);
        gemm_nn(&mut empty, &z12, &[], 3, 4, 0);
        gemm_nh(&mut empty, &[], &z12, 0, 3, 4);
        gemm_nh(&mut empty, &z12, &[], 3, 4, 0);
        gemm_tn::<false>(&mut empty, &[], &z12, 3, 0, 4);
        gemm_tn::<true>(&mut empty, &z12, &[], 3, 4, 0);

        let mut out_nn = vec![sentinel; 6];
        gemm_nn(&mut out_nn, &[], &[], 2, 0, 3);
        assert!(out_nn.iter().all(|&z| z == ZERO));

        let mut out_nh = vec![sentinel; 6];
        gemm_nh(&mut out_nh, &[], &[], 2, 0, 3);
        assert!(out_nh.iter().all(|&z| z == ZERO));

        let mut out_tn = vec![sentinel; 6];
        gemm_tn::<false>(&mut out_tn, &[], &[], 0, 2, 3);
        assert!(out_tn.iter().all(|&z| z == ZERO));
        let mut out_tn_conj = vec![sentinel; 6];
        gemm_tn::<true>(&mut out_tn_conj, &[], &[], 0, 2, 3);
        assert!(out_tn_conj.iter().all(|&z| z == ZERO));
    }

    #[test]
    fn gemm_nn_and_gemm_nh_match_reference_across_threshold() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for &(m, k, n) in &[(7, 5, 11), (32, 32, 32), (64, 64, 64), (70, 65, 60)] {
            let x = random_vec(&mut rng, m * k);
            let y_nn = random_vec(&mut rng, k * n);
            let mut out_nn = vec![Complex64::new(1.0, 1.0); m * n];
            gemm_nn(&mut out_nn, &x, &y_nn, m, k, n);
            for i in 0..m {
                for j in 0..n {
                    let mut ref_val = ZERO;
                    for p in 0..k {
                        ref_val += x[i * k + p] * y_nn[p * n + j];
                    }
                    assert!(
                        (out_nn[i * n + j] - ref_val).norm() < 1e-12,
                        "gemm_nn ({m},{k},{n})[{i},{j}]"
                    );
                }
            }

            let y_nh = random_vec(&mut rng, n * k);
            let mut out_nh = vec![Complex64::new(1.0, 1.0); m * n];
            gemm_nh(&mut out_nh, &x, &y_nh, m, k, n);
            for i in 0..m {
                for j in 0..n {
                    let mut ref_val = ZERO;
                    for p in 0..k {
                        ref_val += x[i * k + p] * y_nh[j * k + p].conj();
                    }
                    assert!(
                        (out_nh[i * n + j] - ref_val).norm() < 1e-12,
                        "gemm_nh ({m},{k},{n})[{i},{j}]"
                    );
                }
            }
        }
    }
}
