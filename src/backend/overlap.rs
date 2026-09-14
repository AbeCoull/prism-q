//! Overlap helpers shared by the backends: the width two states must agree
//! on, the normalization every route applies, and the dense-export route a
//! pair of unlike representations takes.

use num_complex::Complex64;

use super::{Backend, NORM_CLAMP_MIN, dense_statevector_len};
use crate::error::{PrismError, Result};

/// Reject two states of different widths: an inner product across registers
/// of different sizes is not defined.
pub(crate) fn require_equal_width(left: usize, right: usize) -> Result<()> {
    if left == right {
        return Ok(());
    }
    Err(PrismError::InvalidParameter {
        message: format!(
            "a state overlap needs two states of the same width; got {left} and {right} qubits"
        ),
    })
}

/// `|<a|b>|^2 / (<a|a><b|b>)` from the three quantities each route computes
/// its own way. A state with no weight left reads 0 rather than an infinity.
pub(crate) fn normalized(inner_sq: f64, left_norm_sq: f64, right_norm_sq: f64) -> f64 {
    let scale = left_norm_sq * right_norm_sq;
    if scale <= NORM_CLAMP_MIN {
        return 0.0;
    }
    inner_sq / scale
}

/// [`normalized`] over two dense amplitude vectors of equal length.
pub(crate) fn dense_overlap_sq(left: &[Complex64], right: &[Complex64]) -> f64 {
    let mut inner = Complex64::new(0.0, 0.0);
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (a, b) in left.iter().zip(right) {
        inner += a.conj() * b;
        left_norm += a.norm_sqr();
        right_norm += b.norm_sqr();
    }
    normalized(inner.norm_sqr(), left_norm, right_norm)
}

/// Both states exported under the dense cap and dotted, the route a pair with
/// no shared representation takes. It reaches exactly as far as the cap does,
/// so a wider pair answers only when both sides share a representation with a
/// native route.
///
/// The left side passes its own name, width and exporter rather than itself,
/// since the trait default that calls this holds a `&Self` of unknown size and
/// so has no `&dyn Backend` view of itself to hand over.
pub(crate) fn export_overlap_sq(
    name: &str,
    num_qubits: usize,
    export: impl FnOnce() -> Result<Vec<Complex64>>,
    right: &dyn Backend,
) -> Result<f64> {
    require_equal_width(num_qubits, right.num_qubits())?;
    dense_statevector_len(name, "state overlap", num_qubits)?;
    let a = export()?;
    let b = right.export_statevector()?;
    Ok(dense_overlap_sq(&a, &b))
}
