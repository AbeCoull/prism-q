//! Error types for PRISM-Q.
//!
//! Invalid input returns [`PrismError`]. API misuse (out-of-bounds indices,
//! wrong-variant accessors) panics, and each such method says so under `# Panics`.

use thiserror::Error;

/// Top-level error type for PRISM-Q operations.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum PrismError {
    /// OpenQASM parse error with source line number.
    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// Valid OpenQASM that PRISM-Q does not support.
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

    /// Circuit holds an instruction with no OpenQASM 3.0 spelling.
    #[error("cannot export instruction {index} to OpenQASM 3.0: {reason}")]
    ExportUnsupported { index: usize, reason: String },

    /// Incompatible backend for the given circuit.
    #[error("backend `{backend}` is incompatible: {reason}")]
    IncompatibleBackend { backend: String, reason: String },

    /// An allocation the run needs is over a memory cap; the numbers are on
    /// [`ResourceLimit`], boxed so the error stays small on every `Result`.
    #[error("{0}")]
    ResourceLimit(Box<ResourceLimit>),
}

/// Payload of [`PrismError::ResourceLimit`]. `limit` is the cap detected on this
/// machine or set through `env_var`, when a variable overrides it.
#[derive(Debug, Error, Clone, PartialEq)]
#[error(
    "backend `{backend}`: {operation} needs {required} {resource}, exceeding the cap of \
     {limit} on this machine{}",
    override_hint(.env_var)
)]
#[non_exhaustive]
pub struct ResourceLimit {
    pub backend: String,
    pub operation: String,
    pub resource: ResourceKind,
    pub required: u128,
    pub limit: u128,
    pub env_var: Option<&'static str>,
}

/// Unit of a [`PrismError::ResourceLimit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResourceKind {
    /// Qubits of a dense statevector, or of an allocation priced as one.
    Qubits,
    /// Complex amplitudes of backend workspace.
    Amplitudes,
    /// Stored nonzero entries of a sparse state.
    Entries,
    /// Complex elements of a tensor-network intermediate.
    Elements,
    DeviceBytes,
}

impl std::fmt::Display for ResourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ResourceKind::Qubits => "qubits",
            ResourceKind::Amplitudes => "amplitudes",
            ResourceKind::Entries => "entries",
            ResourceKind::Elements => "elements",
            ResourceKind::DeviceBytes => "bytes of device memory",
        })
    }
}

fn override_hint(env_var: &Option<&'static str>) -> String {
    env_var.map_or_else(String::new, |var| format!(" (set {var} to override)"))
}

pub type Result<T> = std::result::Result<T, PrismError>;
