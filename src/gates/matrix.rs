//! Products, adjoints and shape predicates on dense 2x2 and 4x4 matrices.

use super::{Gate, IDENTITY_EPS};
use num_complex::Complex64;

/// Controlled `mat` as a 4x4 with the control on the high bit of the basis index.
#[inline]
pub(crate) fn cu_matrix_4x4(mat: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    let z = Complex64::new(0.0, 0.0);
    let o = Complex64::new(1.0, 0.0);
    [
        [o, z, z, z],
        [z, o, z, z],
        [z, z, mat[0][0], mat[0][1]],
        [z, z, mat[1][0], mat[1][1]],
    ]
}

/// Row-major `2^n` by `2^n` matrix of a gate applying `mat` to the target when all
/// `num_controls` controls are set, with `n = num_controls + 1`: the controls on the high
/// bits of the basis index and the target on bit 0.
pub(crate) fn mcu_matrix(num_controls: usize, mat: &[[Complex64; 2]; 2]) -> Vec<Complex64> {
    let m = num_controls + 1;
    let dim = 1usize << m;
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut full = vec![zero; dim * dim];
    for i in 0..dim - 2 {
        full[i * dim + i] = one;
    }
    let corner = (dim - 2) * dim + dim - 2;
    full[corner..corner + 2].copy_from_slice(&mat[0]);
    full[corner + dim..corner + dim + 2].copy_from_slice(&mat[1]);
    full
}

/// Kronecker product of two 2×2 matrices: A ⊗ B → 4×4.
///
/// Result indices: `(i*2+j, k*2+l) = A[i][k] * B[j][l]`
/// where i,k index A (targets\[0\]) and j,l index B (targets\[1\]).
#[inline]
pub(crate) fn kron_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 4]; 4] {
    let mut result = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..2 {
        for k in 0..2 {
            let aik = a[i][k];
            for j in 0..2 {
                for l in 0..2 {
                    result[i * 2 + j][k * 2 + l] = aik * b[j][l];
                }
            }
        }
    }
    result
}

#[inline]
pub(crate) fn mat_mul_4x4(a: &[[Complex64; 4]; 4], b: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let zero = Complex64::new(0.0, 0.0);
    let mut result = [[zero; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = zero;
            for k in 0..4 {
                sum += a[i][k] * b[k][j];
            }
            result[i][j] = sum;
        }
    }
    result
}

/// `mat` with its two qubits exchanged, `SWAP * mat * SWAP`.
#[inline]
pub(crate) fn swap_qubit_order_4x4(mat: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let swap = Gate::Swap.matrix_4x4();
    mat_mul_4x4(&swap, &mat_mul_4x4(mat, &swap))
}

pub(super) fn adjoint_4x4(m: &[[Complex64; 4]; 4]) -> [[Complex64; 4]; 4] {
    let mut result = [[Complex64::new(0.0, 0.0); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = m[j][i].conj();
        }
    }
    result
}

pub(super) fn adjoint_2x2(m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [m[0][0].conj(), m[1][0].conj()],
        [m[0][1].conj(), m[1][1].conj()],
    ]
}

#[inline]
pub(crate) fn mat_mul_2x2(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Whether a 2x2 matrix is diagonal (both off-diagonal norms below `IDENTITY_EPS`).
#[inline]
pub(crate) fn is_diagonal_2x2(mat: &[[Complex64; 2]; 2]) -> bool {
    mat[0][1].norm() < IDENTITY_EPS && mat[1][0].norm() < IDENTITY_EPS
}

/// Whether a 2x2 matrix is antidiagonal (both diagonal norms below `IDENTITY_EPS`).
#[inline]
pub(crate) fn is_antidiagonal_2x2(mat: &[[Complex64; 2]; 2]) -> bool {
    mat[0][0].norm() < IDENTITY_EPS && mat[1][1].norm() < IDENTITY_EPS
}

/// Whether a 4x4 matrix is diagonal (all off-diagonal norms below `IDENTITY_EPS`).
#[inline]
pub(crate) fn is_diagonal_4x4(mat: &[[Complex64; 4]; 4]) -> bool {
    for (r, row) in mat.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            if r != c && value.norm() >= IDENTITY_EPS {
                return false;
            }
        }
    }
    true
}

pub(super) fn matrices_equal(a: &[[Complex64; 2]; 2], b: &[[Complex64; 2]; 2], eps: f64) -> bool {
    (0..2).all(|i| (0..2).all(|j| (a[i][j] - b[i][j]).norm() <= eps))
}

pub(super) fn matrices_equal_4x4(
    a: &[[Complex64; 4]; 4],
    b: &[[Complex64; 4]; 4],
    eps: f64,
) -> bool {
    (0..4).all(|i| (0..4).all(|j| (a[i][j] - b[i][j]).norm() <= eps))
}
