//! NumPy conversion helpers.

use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::error::{PyPrismResult, invalid};

/// Move a `Vec<f64>` into a 1-D `float64` NumPy array.
pub fn f64_array(py: Python<'_>, values: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    values.into_pyarray(py)
}

/// Move a `Vec<Complex64>` into a 1-D `complex128` NumPy array.
pub fn complex_array(py: Python<'_>, values: Vec<Complex64>) -> Bound<'_, PyArray1<Complex64>> {
    values.into_pyarray(py)
}

/// Build a row-major `(rows, cols)` boolean NumPy matrix from a flat buffer.
pub fn bool_matrix(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    flat: Vec<bool>,
) -> PyPrismResult<Bound<'_, PyArray2<bool>>> {
    let array = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| invalid(format!("failed to shape ({rows}, {cols}) bool matrix: {e}")))?;
    Ok(array.into_pyarray(py))
}
