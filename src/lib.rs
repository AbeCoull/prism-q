//! PRISM-Q — Performance Rust Interoperable Simulator for Quantum
//!
//! A performance-first quantum circuit simulator with pluggable backends.
//!
//! # Quick start
//!
//! ```
//! use prism_q::run_qasm;
//!
//! let qasm = r#"
//!     OPENQASM 3.0;
//!     include "stdgates.inc";
//!     qubit[2] q;
//!     bit[2] c;
//!     h q[0];
//!     cx q[0], q[1];
//! "#;
//!
//! let result = run_qasm(qasm, 42).expect("parse/sim failed");
//! let probs = result.probabilities.expect("no probabilities").to_vec();
//! // Bell state: ~50% |00⟩, ~50% |11⟩
//! assert!((probs[0] - 0.5).abs() < 1e-10);
//! assert!((probs[3] - 0.5).abs() < 1e-10);
//! ```
//!
//! # Input model
//!
//! The primary entrypoint accepts OpenQASM 3.0 strings (`&str`). See
//! [`circuit::openqasm`] for the supported subset.
//!
//! # Backends
//!
//! - [`StatevectorBackend`] — full state-vector simulation (implemented)
//! - [`StabilizerBackend`] — Clifford-only O(n²) simulation (implemented)
//! - [`SparseBackend`] — sparse state-vector O(k) simulation (implemented)
//! - [`MpsBackend`] — Matrix Product State O(nχ²) simulation (implemented)
//! - [`ProductStateBackend`] — per-qubit O(n) simulation for non-entangling circuits (implemented)
//! - [`TensorNetworkBackend`] — deferred contraction for low-treewidth circuits (implemented)
//! - [`FactoredBackend`] — dynamic split-state simulation for sparse-entanglement circuits (implemented)

pub mod backend;
pub mod circuit;
pub mod circuits;
pub mod error;
pub mod gates;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod sim;

pub use backend::factored::FactoredBackend;
pub use backend::factored_stabilizer::FactoredStabilizerBackend;
pub use backend::mps::MpsBackend;
pub use backend::product::ProductStateBackend;
pub use backend::sparse::SparseBackend;
pub use backend::stabilizer::StabilizerBackend;
pub use backend::statevector::StatevectorBackend;
pub use backend::tensornetwork::TensorNetworkBackend;
pub use circuit::builder::CircuitBuilder;
pub use circuit::{Circuit, ClassicalCondition, Instruction, SvgOptions, TextOptions};
pub use error::{PrismError, Result};
pub use gates::{BatchPhaseData, Gate, McuData, Multi2qData, MultiFusedData};
pub use sim::compiled::{
    compile_forward, compile_measurements, run_shots_compiled, CompiledSampler,
    CorrelatorAccumulator, HistogramAccumulator, MarginalsAccumulator, NullAccumulator,
    PackedShots, ParityStats, PauliExpectationAccumulator, ShotAccumulator, ShotLayout,
};
pub use sim::homological::{
    noisy_marginals_analytical, run_shots_homological, ErrorChainComplex, HomologicalSampler,
};
pub use sim::noise::{
    compile_noisy, run_shots_noisy, NoiseChannel, NoiseEvent, NoiseModel, NoisyCompiledSampler,
    ReadoutError,
};
pub use sim::stabilizer_rank::{
    run_stabilizer_rank, run_stabilizer_rank_approx, stabilizer_overlap_sq, StabRankResult,
};
pub use sim::unified_pauli::{
    run_spd, run_spp, spd_to_probabilities, spp_to_probabilities, SpdResult, SppResult,
};
pub use sim::{
    bitstring, run, run_counts, run_marginals, run_on, run_qasm, run_shots, run_shots_with,
    run_shots_with_noise, run_with, BackendKind, FactoredBlock, Probabilities, ShotsResult,
    SimulationResult,
};
