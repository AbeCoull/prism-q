//! Eigendecomposition of the small Hermitian and unitary matrices a gate or an
//! observable carries.
//!
//! Matrices cross this module column major, the layout
//! [`svd_jacobi`](crate::backend::mps::svd_jacobi) uses, so no transpose is needed.

use num_complex::Complex64;

use crate::backend::mps::svd_jacobi;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Largest gap between two eigenvalues still read as one degenerate cluster.
///
/// A unitary is diagonalized through two commuting Hermitian parts, and only a
/// cluster the first part cannot separate is handed to the second, so this
/// decides how much rounding a shared eigenvalue may carry.
const CLUSTER_TOLERANCE: f64 = 1e-9;

/// Eigenvalues ascending and the matching orthonormal eigenvectors of a
/// Hermitian matrix, the eigenvectors column major.
///
/// The matrix is shifted past its spectral radius before it is factorized,
/// which makes it positive definite: the singular values of a positive
/// definite Hermitian matrix are its eigenvalues rather than their magnitudes,
/// and its left factor diagonalizes it. Without the shift a pair of
/// eigenvalues equal in magnitude and opposite in sign shares one singular
/// subspace, and the factor returned for it need not diagonalize anything.
pub(crate) fn hermitian_eigen(matrix: &[Complex64], dim: usize) -> (Vec<f64>, Vec<Complex64>) {
    let peak = matrix
        .iter()
        .fold(0.0f64, |peak, entry| peak.max(entry.norm()));
    let shift = peak * dim as f64 + 1.0;
    let mut shifted = matrix.to_vec();
    for index in 0..dim {
        shifted[index * dim + index] += shift;
    }
    let factored = svd_jacobi(&shifted, dim, dim);
    // Descending singular values, so both halves are reversed onto ascending
    // eigenvalues.
    let values: Vec<f64> = factored.s.iter().rev().map(|value| value - shift).collect();
    let mut vectors = vec![ZERO; dim * dim];
    for column in 0..dim {
        let source = dim - 1 - column;
        vectors[column * dim..column * dim + dim]
            .copy_from_slice(&factored.u[source * dim..source * dim + dim]);
    }
    (values, vectors)
}

/// Eigenvalues and orthonormal eigenvectors of a unitary matrix, the
/// eigenvectors column major.
///
/// A unitary is normal, so its Hermitian and anti-Hermitian parts
/// `A = (U + U*)/2` and `B = (U - U*)/2i` commute and share its eigenvectors.
/// `A` is diagonalized first; only a cluster of `A` eigenvalues too close to
/// separate is handed to `B`, which splits `e^(i theta)` from `e^(-i theta)`
/// where `cos(theta)` alone cannot. Where both are degenerate the matrix is a
/// scalar on that subspace and any orthonormal basis of it diagonalizes.
///
/// Eigenvalues are read back as `(V* U V)` diagonal entries and normalized, so
/// a cluster boundary that split one eigenvalue slightly wrong shows up as a
/// residual rather than as a silently wrong phase.
pub(crate) fn unitary_eigen(matrix: &[Complex64], dim: usize) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut hermitian = vec![ZERO; dim * dim];
    let mut skew = vec![ZERO; dim * dim];
    for column in 0..dim {
        for row in 0..dim {
            let entry = matrix[column * dim + row];
            let adjoint = matrix[row * dim + column].conj();
            hermitian[column * dim + row] = (entry + adjoint) * 0.5;
            skew[column * dim + row] = (entry - adjoint) * Complex64::new(0.0, -0.5);
        }
    }

    let (values, mut vectors) = hermitian_eigen(&hermitian, dim);
    let mut start = 0;
    while start < dim {
        let mut end = start + 1;
        while end < dim && values[end] - values[start] <= CLUSTER_TOLERANCE {
            end += 1;
        }
        if end - start > 1 {
            split_cluster(&skew, dim, &mut vectors, start, end);
        }
        start = end;
    }

    let eigenvalues = (0..dim)
        .map(|column| {
            let vector = &vectors[column * dim..column * dim + dim];
            let applied = multiply_vector(matrix, dim, vector);
            let value: Complex64 = (0..dim).map(|row| vector[row].conj() * applied[row]).sum();
            value / value.norm()
        })
        .collect();
    (eigenvalues, vectors)
}

/// `U^t` for a real exponent, on a unitary of any width, column major.
///
/// The principal power: each eigenvalue angle is brought onto `(-pi, pi]`
/// before it is scaled, so the root follows the eigenvalue rather than any
/// angle the caller wrote.
pub(crate) fn unitary_power(matrix: &[Complex64], dim: usize, t: f64) -> Vec<Complex64> {
    let (values, vectors) = unitary_eigen(matrix, dim);
    let scaled: Vec<Complex64> = values
        .iter()
        .map(|value| Complex64::from_polar(1.0, t * crate::gates::principal_angle(value.arg())))
        .collect();
    let mut out = vec![ZERO; dim * dim];
    for column in 0..dim {
        for row in 0..dim {
            out[column * dim + row] = (0..dim)
                .map(|k| vectors[k * dim + row] * scaled[k] * vectors[k * dim + column].conj())
                .sum();
        }
    }
    out
}

/// Re-diagonalize the columns `start..end` of `vectors` against `operator`,
/// which the caller has checked commutes with what they already diagonalize.
fn split_cluster(
    operator: &[Complex64],
    dim: usize,
    vectors: &mut [Complex64],
    start: usize,
    end: usize,
) {
    let width = end - start;
    let mut projected = vec![ZERO; width * width];
    for column in 0..width {
        let applied = multiply_vector(
            operator,
            dim,
            &vectors[(start + column) * dim..(start + column) * dim + dim],
        );
        for (row, cell) in projected[column * width..column * width + width]
            .iter_mut()
            .enumerate()
        {
            let left = &vectors[(start + row) * dim..(start + row) * dim + dim];
            *cell = (0..dim)
                .map(|index| left[index].conj() * applied[index])
                .sum();
        }
    }
    let (_, rotation) = hermitian_eigen(&projected, width);
    let block: Vec<Complex64> = vectors[start * dim..end * dim].to_vec();
    for column in 0..width {
        for row in 0..dim {
            vectors[(start + column) * dim + row] = (0..width)
                .map(|inner| block[inner * dim + row] * rotation[column * width + inner])
                .sum();
        }
    }
}

fn multiply_vector(matrix: &[Complex64], dim: usize, vector: &[Complex64]) -> Vec<Complex64> {
    (0..dim)
        .map(|row| {
            (0..dim)
                .map(|column| matrix[column * dim + row] * vector[column])
                .sum()
        })
        .collect()
}

#[cfg(test)]
#[path = "spectral_tests.rs"]
mod tests;
