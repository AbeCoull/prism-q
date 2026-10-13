//! Error types for PRISM-Q.
//!
//! Invalid input returns [`PrismError`]. API misuse (out-of-bounds indices,
//! wrong-variant accessors) panics, and each such method says so under `# Panics`.

use thiserror::Error;

use crate::limits::{LimitKnob, LimitSource};

/// Top-level error type for PRISM-Q operations.
#[derive(Debug, Error, Clone, PartialEq)]
#[non_exhaustive]
pub enum PrismError {
    /// OpenQASM parse error with source line number.
    #[error("parse error at line {line}: {message}")]
    Parse { line: usize, message: String },

    /// Valid OpenQASM that PRISM-Q does not support. A loop keyword names the
    /// entry point that reads it.
    #[error("unsupported construct at line {line}: `{construct}`{}", loop_hint(.construct))]
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

    /// A shot of a dynamic program ran more blocks than its step bound allows.
    /// `region` names the loop it was in.
    #[error(
        "dynamic program stopped after {max_steps} steps in {region}; raise the bound with          `max_steps` if the loop is meant to run that long"
    )]
    StepLimit { region: String, max_steps: u64 },

    /// An allocation the run needs is over a memory cap; the numbers are on
    /// [`ResourceLimit`], boxed so the error stays small on every `Result`.
    #[error("{0}")]
    ResourceLimit(Box<ResourceLimit>),
}

/// Payload of [`PrismError::ResourceLimit`]. `limit` is the cap `knob` names, resolved
/// from the layer `limit_source` names. `env_var` is the variable that overrides it,
/// `None` when [`Limits`](crate::Limits) set the cap, since a request or process
/// default outranks the environment.
#[derive(Debug, Error, Clone, PartialEq)]
#[error(
    "backend `{backend}`: {operation} needs {required} {resource}, exceeding the cap of \
     {limit} {}{}",
    origin(.limit_source, .env_var),
    override_hint(.knob, .env_var)
)]
#[non_exhaustive]
pub struct ResourceLimit {
    pub backend: String,
    pub operation: String,
    pub resource: ResourceKind,
    pub required: u128,
    pub limit: u128,
    pub env_var: Option<&'static str>,
    /// The [`Limits`](crate::Limits) cap the allocation ran into; `None` for a bound
    /// no limit moves, such as free device memory or the address width.
    pub knob: Option<LimitKnob>,
    pub limit_source: LimitSource,
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

/// Where a program declined for a loop keyword can go instead.
fn loop_hint(construct: &str) -> &'static str {
    match construct {
        "while" | "break" | "continue" => {
            "; `openqasm::parse_dynamic` reads it into a control-flow graph"
        }
        _ => "",
    }
}

fn origin(source: &LimitSource, env_var: &Option<&'static str>) -> String {
    match (source, env_var) {
        (LimitSource::Env, Some(var)) => format!("set by {var}"),
        (LimitSource::ProcessDefault, _) => "set by the process default limits".to_string(),
        (LimitSource::Request, _) => "set by this request's limits".to_string(),
        _ => "on this machine".to_string(),
    }
}

fn override_hint(knob: &Option<LimitKnob>, env_var: &Option<&'static str>) -> String {
    let setter = knob.map(|knob| match knob {
        LimitKnob::DensityMatrixQubits => {
            "Limits::max_density_matrix_qubits and max_statevector_qubits".to_string()
        }
        _ => format!("Limits::{}", knob.name()),
    });
    match (setter, env_var) {
        (Some(setter), Some(var)) => format!(" (override with {setter} or {var})"),
        (Some(setter), None) => format!(" (override with {setter})"),
        (None, Some(var)) => format!(" (set {var} to override)"),
        (None, None) => String::new(),
    }
}

pub type Result<T> = std::result::Result<T, PrismError>;
