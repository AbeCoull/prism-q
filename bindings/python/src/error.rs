//! Error bridging between `prism_q::PrismError` and Python exceptions.
//!
//! The orphan rule forbids implementing `From<prism_q::PrismError> for PyErr`
//! directly, so a local newtype carries the conversion. Binding methods return
//! `Result<T, PyPrismError>`; PyO3 raises the registered `PrismError` exception.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(
    prism_q,
    PrismError,
    PyException,
    "Error raised by PRISM-Q operations."
);

/// Newtype bridging `prism_q::PrismError` to `PyErr` across the orphan rule.
pub struct PyPrismError(pub prism_q::PrismError);

impl From<prism_q::PrismError> for PyPrismError {
    fn from(err: prism_q::PrismError) -> Self {
        PyPrismError(err)
    }
}

impl From<PyPrismError> for PyErr {
    fn from(err: PyPrismError) -> PyErr {
        PrismError::new_err(err.0.to_string())
    }
}

/// Result alias for binding methods that surface `PrismError` to Python.
pub type PyPrismResult<T> = Result<T, PyPrismError>;

/// Construct an `InvalidParameter` error for binding-layer validation failures.
pub fn invalid(message: impl Into<String>) -> PyPrismError {
    PyPrismError(prism_q::PrismError::InvalidParameter {
        message: message.into(),
    })
}
