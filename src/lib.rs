//! PRISM-Q: Performance Rust Interoperable Simulator for Quantum
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
//! // Bell state: ~50% |00>, ~50% |11>
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
//! - [`StatevectorBackend`]: full state-vector simulation
//! - [`StabilizerBackend`]: Clifford-only O(n^2) simulation
//! - [`SparseBackend`]: sparse state-vector O(k) simulation
//! - [`MpsBackend`]: Matrix Product State O(n * chi^2) simulation
//! - [`ProductStateBackend`]: per-qubit O(n) simulation for non-entangling circuits
//! - [`TensorNetworkBackend`]: deferred contraction for low-treewidth circuits
//! - [`FactoredBackend`]: dynamic split-state simulation for sparse-entanglement circuits
//! - [`FactoredStabilizerBackend`]: per-group tableaux for large sparse-entanglement Clifford circuits
//! - [`backend::density_matrix::DensityMatrixBackend`]: 4^n density matrix for exact channel semantics
//! - `DistributedStatevectorBackend`: statevector sharded across ranks (`distributed` feature)
//!
//! # Native QEC
//!
//! Measurement-record QEC programs use [`QecProgram`] or [`parse_qec_program`].
//! [`run_qec_program`] executes supported Clifford QEC programs through packed
//! compiled sampling with Pauli-noise annotations. [`run_qec_program_reference`]
//! runs small correctness checks through the reference path.

#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod backend;
pub mod circuit;
pub mod circuits;
#[cfg(feature = "distributed")]
pub mod distributed;
pub mod error;
pub mod gates;
#[cfg(feature = "gpu")]
pub mod gpu;
mod hash;
pub mod qec;
pub mod sim;

#[cfg(feature = "distributed")]
pub use backend::distributed_statevector::DistributedStatevectorBackend;
pub use backend::factored::FactoredBackend;
pub use backend::factored_stabilizer::FactoredStabilizerBackend;
pub use backend::mps::MpsBackend;
pub use backend::product::ProductStateBackend;
pub use backend::sparse::SparseBackend;
pub use backend::stabilizer::StabilizerBackend;
pub use backend::statevector::StatevectorBackend;
pub use backend::tensornetwork::TensorNetworkBackend;
pub use circuit::builder::CircuitBuilder;
pub use circuit::{
    Circuit, ClassicalCondition, Instruction, ParamLink, Parameters, PreparedCircuit, SvgOptions,
    TextOptions,
};
#[cfg(feature = "distributed-mpi")]
pub use distributed::MpiComm;
#[cfg(feature = "distributed")]
pub use distributed::{DistributedContext, RankComm, SerialComm};
pub use error::{PrismError, Result};
pub use gates::{BatchPhaseData, Gate, GeneratorKind, McuData, Multi2qData, MultiFusedData};
pub use qec::{
    DetectorErrorModel, ErrorMechanism, QecBasis, QecCompiledRows, QecMeasurementRow, QecNoise,
    QecObservableEstimate, QecObservableReroute, QecOp, QecOptions, QecPauli, QecProgram,
    QecRecordRef, QecSampleResult, QecTStrategy, UnionFindDecoder, compile_qec_program_rows,
    parse_qec_program, run_qec_program, run_qec_program_reference, run_qec_program_spd_rerouted,
    run_qec_program_with_strategy,
};
#[cfg(feature = "bench-internal")]
pub use qec::{QecProfiledCounts, QecProfiledSampler, compile_qec_profiled_sampler};
pub use sim::compiled::{
    CompiledDetectorSampler, CompiledSampler, CorrelatorAccumulator, DetectorSampleBatch,
    HistogramAccumulator, MarginalsAccumulator, NullAccumulator, PackedShots, ParityStats,
    PauliExpectationAccumulator, ShotAccumulator, ShotLayout, compile_detector_sampler,
    compile_forward, compile_measurements, run_shots_compiled,
};
pub use sim::gradient::{
    ExpectationGradient, run_expectation_gradient, run_expectation_gradient_shift,
};
pub use sim::homological::{
    ErrorChainComplex, HomologicalSampler, noisy_marginals_analytical, run_shots_homological,
};
pub use sim::noise::{
    GateFilter, NoiseBuilder, NoiseChannel, NoiseEvent, NoiseModel, NoisyCompiledSampler,
    ReadoutError, compile_noisy, density_matrix_expectation_values, run_shots_noisy,
};
pub use sim::stabilizer_rank::{
    StabRankResult, run_stabilizer_rank, run_stabilizer_rank_approx, stabilizer_inner_product,
    stabilizer_overlap_sq,
};
pub use sim::unified_pauli::{
    PauliAxis, PauliTerm, SpdObservableResult, SpdResult, SppObservableResult, SppResult,
    inverse_light_cone, run_spd, run_spd_observable, run_spd_observable_light_cone, run_spp,
    run_spp_observable,
};
pub use sim::{
    BackendKind, CountsResult, Exactness, ExpectationResult, FactoredBlock, MarginalsResult,
    ObservableExpectation, PauliObservable, Placement, Probabilities, ResolvedBackend, RunMetadata,
    RunOutcome, Seeded, ShotsResult, Simulate, Unseeded, bitstring, run_expectation_values,
    run_observable_expectation, run_on, run_on_state, run_qasm, simulate,
};

#[cfg(feature = "gpu")]
pub use sim::compiled::{DevicePackedShots, run_shots_compiled_with_gpu};
