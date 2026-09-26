//! Error bridging between `prism_q::PrismError` and the Python `PrismError` exception.

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

/// Stable discriminant for a [`prism_q::PrismError`], surfaced as the raised
/// exception's `kind` attribute so callers need not match on message text.
fn error_kind(err: &prism_q::PrismError) -> &'static str {
    use prism_q::PrismError as E;
    match err {
        E::Parse { .. } => "parse",
        E::UnsupportedConstruct { .. } => "unsupported_construct",
        E::InvalidQubit { .. } => "invalid_qubit",
        E::InvalidClassicalBit { .. } => "invalid_classical_bit",
        E::GateArity { .. } => "gate_arity",
        E::BackendUnsupported { .. } => "backend_unsupported",
        E::InvalidParameter { .. } => "invalid_parameter",
        E::UndefinedRegister { .. } => "undefined_register",
        E::ExportUnsupported { .. } => "export_unsupported",
        E::IncompatibleBackend { .. } => "incompatible_backend",
        E::ResourceLimit { .. } => "resource_limit",
        _ => "other",
    }
}

impl From<PyPrismError> for PyErr {
    fn from(err: PyPrismError) -> PyErr {
        let kind = error_kind(&err.0);
        let py_err = PrismError::new_err(err.0.to_string());
        Python::attach(|py| {
            // Failing to annotate must not displace the error being reported.
            let value = py_err.value(py);
            let _ = value.setattr("kind", kind);
            if let prism_q::PrismError::ResourceLimit(data) = &err.0 {
                let _ = value.setattr("resource", data.resource.to_string());
                let _ = value.setattr("required", data.required);
                let _ = value.setattr("limit", data.limit);
                let _ = value.setattr("env_var", data.env_var);
            }
        });
        py_err
    }
}

pub type PyPrismResult<T> = Result<T, PyPrismError>;

/// Construct an `InvalidParameter` error for binding-layer validation failures.
pub fn invalid(message: impl Into<String>) -> PyPrismError {
    PyPrismError(prism_q::PrismError::InvalidParameter {
        message: message.into(),
    })
}
