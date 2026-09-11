//! Schmidt spectrum helpers shared by the backends: subsystem validation, the
//! entropy formula, the eigenvalues of a reduced density matrix, and the
//! reshape-and-SVD route over a dense amplitude vector.

use num_complex::Complex64;

use super::mps::svd;
use crate::error::{PrismError, Result};

/// Relative floor under which a singular value is numerical noise. The SVD
/// resolves the spectrum to about `1e-16` of its largest value, and a value at
/// this floor carries under `1e-27` nats.
const SCHMIDT_VALUE_FLOOR: f64 = 1e-14;

/// Reject a subsystem that is empty, out of range, or names a qubit twice.
pub(crate) fn validate_qubit_set(subsystem: &[usize], num_qubits: usize) -> Result<()> {
    let mut seen = vec![false; num_qubits];
    for &qubit in subsystem {
        if qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: qubit,
                register_size: num_qubits,
            });
        }
        if seen[qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!("subsystem names qubit {qubit} twice"),
            });
        }
        seen[qubit] = true;
    }
    if subsystem.is_empty() {
        return Err(PrismError::InvalidParameter {
            message: "subsystem must name at least one qubit".to_string(),
        });
    }
    Ok(())
}

/// [`validate_qubit_set`], and reject the whole register too, so a cut always
/// has two non-empty sides.
pub(crate) fn validate_subsystem(subsystem: &[usize], num_qubits: usize) -> Result<()> {
    validate_qubit_set(subsystem, num_qubits)?;
    if subsystem.len() == num_qubits {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "subsystem must leave both sides of the cut non-empty, got {} of {num_qubits} \
                 qubits",
                subsystem.len()
            ),
        });
    }
    Ok(())
}

/// Sort descending, drop values under [`SCHMIDT_VALUE_FLOOR`] of the largest,
/// and scale so the squares sum to 1.
pub(crate) fn finish_schmidt_values(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_unstable_by(|a, b| b.total_cmp(a));
    let floor = values.first().copied().unwrap_or(0.0) * SCHMIDT_VALUE_FLOOR;
    values.truncate(
        values
            .iter()
            .position(|&s| s <= floor)
            .unwrap_or(values.len()),
    );
    let total: f64 = values.iter().map(|s| s * s).sum();
    debug_assert!(total > 0.0, "a state with no singular weight");
    let scale = 1.0 / total.sqrt();
    for s in &mut values {
        *s *= scale;
    }
    values
}

/// Schmidt values from a `dim x dim` reduced density matrix, the square roots
/// of its eigenvalues. Those are resolved to about `1e-16` of the largest, so
/// a weight under [`SCHMIDT_VALUE_FLOOR`] of it is noise and is dropped here,
/// before the square root would lift it above the floor on the values.
pub(crate) fn schmidt_values_from_density(rho: &[Complex64], dim: usize) -> Vec<f64> {
    let mut weights = hermitian_eigenvalues(rho, dim);
    let floor = weights.iter().copied().fold(0.0, f64::max) * SCHMIDT_VALUE_FLOOR;
    weights.retain(|&w| w > floor);
    finish_schmidt_values(weights.into_iter().map(f64::sqrt).collect())
}

/// Per-pair stopping criterion of [`hermitian_eigenvalues`]: relative to the
/// two diagonal entries the pair couples, or to the largest diagonal entry once
/// one of the two has collapsed to rounding, which every eigenvalue of a
/// rank-deficient matrix does.
const EIGEN_PAIR_TOL: f64 = 1e-15;
const MAX_EIGEN_SWEEPS: usize = 60;

/// Matrix width above which [`hermitian_eigenvalues`] hands the matrix to faer,
/// the same `m * n >= 256` threshold as `mps::svd`.
#[cfg(feature = "parallel")]
const JACOBI_EIGEN_ELEMENTS: usize = 256;

/// Smaller-side width past which a build without `parallel` declines the
/// reduced-density route: `2^8` is the last width the Jacobi sweep resolves
/// in about a second, and the next doubling costs eight times that.
#[cfg(not(feature = "parallel"))]
const MAX_JACOBI_SIDE: usize = 8;

/// Eigenvalues of a Hermitian positive semidefinite `dim x dim` matrix, in
/// any order, to about `1e-16` of the largest.
///
/// Above `JACOBI_EIGEN_ELEMENTS` under `parallel` the matrix goes to
/// `svd_faer`, since the singular values of a semidefinite matrix are its
/// eigenvalues. `svd_jacobi` is not used below it: that sweep stops on a
/// criterion absolute in the Frobenius norm, which leaves an eigenvalue near
/// `1e-7` of the largest unresolved, and on the squared spectrum a density
/// matrix carries that is a Schmidt value near `3e-4`. This cyclic Jacobi stops
/// per pair on [`EIGEN_PAIR_TOL`] instead, at `O(dim^3)` per sweep.
fn hermitian_eigenvalues(a: &[Complex64], dim: usize) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    if dim * dim >= JACOBI_EIGEN_ELEMENTS {
        return super::mps::svd_faer(a, dim, dim).s;
    }
    let mut a = a.to_vec();
    let at = |p: usize, q: usize| p * dim + q;
    for _ in 0..MAX_EIGEN_SWEEPS {
        let mut rotated = false;
        let max_diag = (0..dim).map(|i| a[at(i, i)].re).fold(0.0, f64::max);
        for p in 0..dim {
            for q in p + 1..dim {
                let apq = a[at(p, q)];
                let (app, aqq) = (a[at(p, p)].re, a[at(q, q)].re);
                let apq_abs = apq.norm();
                if apq_abs <= EIGEN_PAIR_TOL * (app * aqq).abs().sqrt()
                    || apq_abs <= EIGEN_PAIR_TOL * max_diag
                {
                    continue;
                }
                rotated = true;
                // `D = diag(1, e^{-i phi})` makes the pair real, then the
                // real rotation `R` zeros it; the update is `G^H A G` with
                // `G = D R`.
                let phase = apq / apq_abs;
                let theta = (aqq - app) / (2.0 * apq_abs);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let phase_conj = phase.conj();
                for k in 0..dim {
                    let (akp, akq) = (a[at(k, p)], a[at(k, q)]);
                    a[at(k, p)] = c * akp - s * phase_conj * akq;
                    a[at(k, q)] = s * akp + c * phase_conj * akq;
                }
                for k in 0..dim {
                    let (apk, aqk) = (a[at(p, k)], a[at(q, k)]);
                    a[at(p, k)] = c * apk - s * phase * aqk;
                    a[at(q, k)] = s * apk + c * phase * aqk;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    (0..dim).map(|i| a[at(i, i)].re).collect()
}

/// `-sum p ln p` in nats over `p = s^2 / sum s^2`.
pub(crate) fn entropy_of_schmidt_values(values: &[f64]) -> f64 {
    let total: f64 = values.iter().map(|s| s * s).sum();
    let mut entropy = 0.0;
    for s in values {
        let p = s * s / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Width check for the reduced-density route over a smaller side of `side`
/// qubits. The matrix holds `4^side` entries, the bytes of a `2 * side`-qubit
/// statevector, so that is the width priced against the dense export cap.
/// Without `parallel` the eigensolver is the Jacobi sweep, which bounds the
/// side at [`MAX_JACOBI_SIDE`] as well.
pub(crate) fn check_schmidt_side(backend: &str, what: &str, side: usize) -> Result<()> {
    #[cfg(not(feature = "parallel"))]
    if side > MAX_JACOBI_SIDE {
        return Err(PrismError::BackendUnsupported {
            backend: backend.to_string(),
            operation: format!(
                "{what} with {side} qubits on the smaller side (max {MAX_JACOBI_SIDE} without \
                 the parallel feature, whose eigensolver is a Jacobi sweep)"
            ),
        });
    }
    let width = 2 * side;
    if width > export_cap() {
        return Err(export_cap_exceeded(
            backend,
            format!(
                "{what} with {side} qubits on the smaller side, whose reduced density matrix \
                 is the size of a statevector for {width} qubits"
            ),
        ));
    }
    Ok(())
}

/// Width check for the dense route on `num_qubits` qubits. Its transient is
/// the gathered copy of the state plus the thin SVD's wide factor, two vectors
/// of the state's length, so it is priced as one statevector of
/// `num_qubits + 1` qubits against the dense export cap.
fn check_dense_schmidt_width(backend: &str, num_qubits: usize) -> Result<()> {
    let width = num_qubits + 1;
    if width > export_cap() {
        return Err(export_cap_exceeded(
            backend,
            format!(
                "Schmidt values across a cut, whose gathered copy and thin factor together are \
                 the size of a statevector for {width} qubits"
            ),
        ));
    }
    Ok(())
}

/// The dense export cap as `dense_statevector_len` applies it, so a check
/// against it formats no message on the path that passes.
pub(crate) fn export_cap() -> usize {
    super::memory::max_dense_statevector_qubits().min(usize::BITS as usize - 1)
}

/// The error a diagnostic raises past the dense export cap: the same variant a
/// statevector export raises there, so nothing reads it as a missing terminal,
/// with `what` naming the allocation that was priced.
pub(crate) fn export_cap_exceeded(backend: &str, what: String) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: backend.to_string(),
        reason: format!(
            "{what} (max {} on this machine, set PRISM_MAX_EXPORT_QUBITS to override)",
            export_cap()
        ),
    }
}

/// Schmidt values of a dense `2^num_qubits` amplitude vector across
/// `subsystem`, which the caller has validated: the vector reshaped as a
/// matrix with the smaller side of the cut gathered into the row index, then
/// one thin SVD. Any overall scale on `state` drops out.
pub(crate) fn dense_schmidt_values(
    backend: &str,
    state: &[Complex64],
    num_qubits: usize,
    subsystem: &[usize],
) -> Result<Vec<f64>> {
    check_dense_schmidt_width(backend, num_qubits)?;

    let k = subsystem.len();
    let side = k.min(num_qubits - k);
    let named = subsystem.iter().fold(0usize, |mask, &q| mask | (1 << q));
    let full = state.len() - 1;
    let (row_mask, col_mask) = if k == side {
        (named, !named & full)
    } else {
        (!named & full, named)
    };
    let rows = 1usize << side;
    let cols = state.len() >> side;
    let mut mat = vec![Complex64::new(0.0, 0.0); state.len()];
    // `x = (x - mask) & mask` steps through the submasks of `mask` in
    // increasing order, which is the deposit of 0, 1, 2, ... into its bits.
    let mut col_bits = 0usize;
    for column in mat.chunks_exact_mut(rows) {
        let mut row_bits = 0usize;
        for out in column {
            *out = state[col_bits | row_bits];
            row_bits = row_bits.wrapping_sub(row_mask) & row_mask;
        }
        col_bits = col_bits.wrapping_sub(col_mask) & col_mask;
    }
    Ok(finish_schmidt_values(svd(&mat, rows, cols).s))
}
