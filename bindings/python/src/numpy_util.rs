//! NumPy conversion helpers.

use num_complex::Complex64;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
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

/// Build a row-major `(rows, cols)` `complex128` NumPy matrix from a flat
/// buffer.
pub fn complex_matrix(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    flat: Vec<Complex64>,
) -> PyPrismResult<Bound<'_, PyArray2<Complex64>>> {
    let array = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
        invalid(format!(
            "failed to shape ({rows}, {cols}) complex matrix: {e}"
        ))
    })?;
    Ok(array.into_pyarray(py))
}

/// Build a row-major `(rows, cols)` `float64` NumPy matrix from a flat buffer.
pub fn f64_matrix(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    flat: Vec<f64>,
) -> PyPrismResult<Bound<'_, PyArray2<f64>>> {
    let array = Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
        invalid(format!(
            "failed to shape ({rows}, {cols}) float matrix: {e}"
        ))
    })?;
    Ok(array.into_pyarray(py))
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

/// Read a 1-D `uint64` NumPy array, or any sequence of ints, into a `Vec`.
pub fn u64_words(value: &Bound<'_, PyAny>) -> PyPrismResult<Vec<u64>> {
    if let Ok(array) = value.cast::<PyArray1<u64>>() {
        return Ok(array.readonly().as_array().to_vec());
    }
    value.extract().map_err(|e| {
        invalid(format!(
            "expected a uint64 array or a sequence of ints: {e}"
        ))
    })
}

/// Read a 1-D `bool` NumPy array, or any sequence of bools, into a `Vec`.
pub fn bool_flags(value: &Bound<'_, PyAny>) -> PyPrismResult<Vec<bool>> {
    if let Ok(array) = value.cast::<PyArray1<bool>>() {
        return Ok(array.readonly().as_array().to_vec());
    }
    value
        .extract()
        .map_err(|e| invalid(format!("expected a bool array or a sequence of bools: {e}")))
}
