//! Error types for PRISM-Q.
//!
//! All public APIs return structured errors — never panics for user-facing operations.
//! Internal debug_assert! may fire in debug builds for invariant violations.

use thiserror::Error;

/// Top-level error type for PRISM-Q operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum PrismError {
    /// OpenQASM parse error with source line number.
    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// Encountered a valid OpenQASM construct that PRISM-Q v0 does not support.
    #[error("unsupported construct at line {line}: `{construct}`")]
    UnsupportedConstruct { construct: String, line: usize },

    /// Qubit index exceeds register size.
    #[error("invalid qubit index {index} (register size: {register_size})")]
    InvalidQubit { index: usize, register_size: usize },

    /// Classical bit index exceeds register size.
    #[error("invalid classical bit index {index} (register size: {register_size})")]
    InvalidClassicalBit { index: usize, register_size: usize },

    /// Gate applied to wrong number of qubits.
    #[error("gate `{gate}`: expected {expected} qubit(s), got {got}")]
    GateArity {
        gate: String,
        expected: usize,
        got: usize,
    },

    /// Backend does not support the requested operation.
    #[error("backend `{backend}` does not support: {operation}")]
    BackendUnsupported { backend: String, operation: String },

    /// Invalid gate parameter (e.g., NaN rotation angle).
    #[error("invalid parameter: {message}")]
    InvalidParameter { message: String },

    /// Reference to a register name that was never declared.
    #[error("undefined register `{name}` at line {line}")]
    UndefinedRegister { name: String, line: usize },

    /// Incompatible backend for the given circuit.
    #[error("backend `{backend}` is incompatible: {reason}")]
    IncompatibleBackend { backend: String, reason: String },
}

/// Convenience alias used throughout PRISM-Q.
pub type Result<T> = std::result::Result<T, PrismError>;
