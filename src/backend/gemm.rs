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
