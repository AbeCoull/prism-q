//! Measurement-record QEC programs ([`QecProgram`]): parser, runners, detector error
//! models, and decoders. The IR is separate from `Circuit` so measurement
//! records need not fit final-measurement OpenQASM semantics.
//!
//! - [`run_qec_program`] lowers into the packed compiled Clifford sampler and applies
//!   Pauli noise by XORing sensitivity rows onto packed measurement records.
//! - [`run_qec_program_reference`] runs one statevector simulation per shot, as a
//!   correctness oracle for small programs.
//! - [`run_qec_program_with_strategy`] dispatches non-Clifford observable programs
//!   through exact light-cone SPD, CAMPS, then an exact tensor-network scalar fallback.
//! - [`compile_qec_program_rows`] lowers basis measurements and `MPP` records into packed
//!   X/Z Pauli rows without executing gates, resets, or noise.
//! - [`QecProgram::detector_error_model`] derives the [`DetectorErrorModel`] for export
//!   to matching and belief-propagation decoders.
//! - [`UnionFindDecoder`] and the exact [`MatchingDecoder`] decode packed detector
//!   samples against a graphlike model; [`BpOsdDecoder`] decodes any model.
//! - [`QecProgram::surface_memory`] and its repetition and color code siblings generate
//!   memory experiments with circuit-level noise.

mod bposd;
pub(crate) mod camps_prefix;
mod compiled;
/// Treewidth-aware cut-selection heuristics, benchmark-only: the dispatcher follows a
/// fixed SPD -> CAMPS -> tensor-network ladder and does not use them.
#[cfg(feature = "bench-internal")]
pub mod cut_selection;
mod decoder;
mod dem;
mod dem_text;
mod generators;
mod leakage;
mod matching;
mod noise;
pub mod observable_reroute;
mod parity_walk;
mod parse;
mod program;
mod reference;
mod reference_runner;
mod result;
mod rows;
mod runner;
mod t_sampler;
mod text;

pub use bposd::{BpMethod, BpOsdDecoder, BpOsdOptions, OsdMethod};
pub use compiled::QecCompiledSampler;
pub use decoder::UnionFindDecoder;
pub use dem::{DetectorErrorModel, ErrorMechanism};
pub use generators::QecCircuitNoise;
pub use matching::MatchingDecoder;
pub use parse::{parse_qec_program, parse_stim_circuit};
pub use reference_runner::run_qec_program_reference;
pub use result::{QecObservableEstimate, QecSampleResult};
pub use rows::{QecCompiledRows, QecMeasurementRow, compile_qec_program_rows};
pub use runner::run_qec_program;
#[cfg(feature = "bench-internal")]
pub use runner::{QecProfiledCounts, QecProfiledSampler, compile_qec_profiled_sampler};
pub use t_sampler::{
    QecObservableReroute, QecTStrategy, run_qec_program_spd_rerouted, run_qec_program_with_strategy,
};

use crate::circuit::{
    GateSink, append_axis_to_z_rotation, append_parity_rotations, append_z_to_axis_rotation,
};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Rounding slack on the branch sum of a multi-probability channel, so `0.1, 0.2, 0.7`
/// is accepted.
pub(super) const QEC_PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Exclusive bound on qubit and observable indices read from QEC text, and on observable
/// indices a program accepts, since runners size buffers from the highest index.
pub(super) const QEC_MAX_INDEX: usize = 1 << 24;

/// Pauli basis used by QEC measurements and Pauli products.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QecBasis {
    X,
    Y,
    Z,
}

impl QecBasis {
    /// Pauli index with `X = 1`, `Y = 2`, `Z = 3`, leaving 0 for the identity.
    pub(crate) const fn pauli_index(self) -> u8 {
        match self {
            QecBasis::X => 1,
            QecBasis::Y => 2,
            QecBasis::Z => 3,
        }
    }
}

impl From<QecBasis> for PauliAxis {
    fn from(basis: QecBasis) -> Self {
        match basis {
            QecBasis::X => PauliAxis::X,
            QecBasis::Y => PauliAxis::Y,
            QecBasis::Z => PauliAxis::Z,
        }
    }
}

/// One Pauli term in an MPP-style measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QecPauli {
    pub basis: QecBasis,
    pub qubit: usize,
}

impl QecPauli {
    pub fn new(basis: QecBasis, qubit: usize) -> Self {
        Self { basis, qubit }
    }

    pub fn x(qubit: usize) -> Self {
        Self::new(QecBasis::X, qubit)
    }

    pub fn y(qubit: usize) -> Self {
        Self::new(QecBasis::Y, qubit)
    }

    pub fn z(qubit: usize) -> Self {
        Self::new(QecBasis::Z, qubit)
    }
}

/// One branch of a [`QecOp::CorrelatedError`] chain.
#[derive(Debug, Clone, PartialEq)]
pub struct QecCorrelatedBranch {
    /// Probability that the branch fires, given that no earlier branch of its chain did.
    pub probability: f64,
    pub terms: Vec<QecPauli>,
}

/// Reference to a previous measurement record.
///
/// A lookback counts back from the records that precede the referencing op, so
/// `Lookback(1)` is the most recent measurement before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum QecRecordRef {
    Absolute(usize),
    Lookback(usize),
}

impl QecRecordRef {
    pub fn absolute(index: usize) -> Self {
        Self::Absolute(index)
    }

    pub fn lookback(distance: usize) -> Result<Self> {
        if distance == 0 {
            return Err(PrismError::InvalidParameter {
                message: "measurement lookback distance must be at least 1".to_string(),
            });
        }
        Ok(Self::Lookback(distance))
    }

    fn resolve(self, next_measurement: usize) -> Result<usize> {
        match self {
            Self::Absolute(index) if index < next_measurement => Ok(index),
            Self::Absolute(index) => Err(PrismError::InvalidParameter {
                message: format!(
                    "measurement record {index} out of bounds for {next_measurement} existing records"
                ),
            }),
            Self::Lookback(distance) if distance > 0 && distance <= next_measurement => {
                Ok(next_measurement - distance)
            }
            Self::Lookback(distance) => Err(PrismError::InvalidParameter {
                message: format!(
                    "measurement lookback {distance} out of bounds for {next_measurement} existing records"
                ),
            }),
        }
    }
}

/// Noise annotation for native QEC programs: Pauli channels, and the leakage channels
/// [`run_qec_program`] samples as heralded erasure.
///
/// Probabilities are validated on append to a [`QecProgram`]; probability zero makes
/// the annotation inactive.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QecNoise {
    /// With probability `p`, apply X to each target.
    XError(f64),
    /// With probability `p`, apply Z to each target.
    ZError(f64),
    /// Per target, apply each of X, Y, Z with probability `p / 3`.
    Depolarize1(f64),
    /// Per target pair, apply each of the 15 non-identity two-qubit Paulis with
    /// probability `p / 15`. The target list must have even length.
    Depolarize2(f64),
    /// With probability `p`, apply Y to each target.
    YError(f64),
    /// Per target, apply X, Y, or Z with the listed probability, in that order.
    PauliChannel1([f64; 3]),
    /// Per target pair, apply `P0 P1` with its listed probability, in the order `IX, IY,
    /// IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ` (`P0` on the first target of the
    /// pair). The target list must have even length. Boxed so [`QecOp`] stays 64 bytes.
    PauliChannel2(Box<[f64; 15]>),
    /// Per target, an unleaked qubit leaks with probability `p`. Each target adds one
    /// column to [`QecSampleResult::heralds`], set when the qubit is leaked just after
    /// this annotation, so `LEAK(0)` reads the flags without changing them.
    ///
    /// While leaked, a qubit is erased: a two-qubit gate on it fully depolarizes both
    /// of its qubits, its own measurement reports 1, and an `MPP` over it reports a
    /// uniformly random parity. A reset or a [`QecNoise::Seep`] ends the leak.
    Leak(f64),
    /// Per target, a leaked qubit returns with probability `p`, in a uniformly random
    /// computational basis state.
    Seep(f64),
    /// Per target pair, when exactly one of the two qubits is leaked the other leaks
    /// too with probability `p`. The target list must have even length.
    LeakTransport(f64),
}

impl QecNoise {
    /// Probability that the channel fires on one target or pair: a non-identity Pauli, or
    /// a leakage transition.
    pub fn probability(&self) -> f64 {
        match self {
            Self::XError(p)
            | Self::ZError(p)
            | Self::Depolarize1(p)
            | Self::Depolarize2(p)
            | Self::YError(p)
            | Self::Leak(p)
            | Self::Seep(p)
            | Self::LeakTransport(p) => *p,
            Self::PauliChannel1(probabilities) => probabilities.iter().sum(),
            Self::PauliChannel2(probabilities) => probabilities.iter().sum(),
        }
    }

    /// True for the channels that carry a leak flag between annotations.
    pub fn is_leakage(&self) -> bool {
        matches!(self, Self::Leak(_) | Self::Seep(_) | Self::LeakTransport(_))
    }

    /// Native text instruction name for this channel.
    pub fn name(&self) -> &'static str {
        match self {
            Self::XError(_) => "X_ERROR",
            Self::ZError(_) => "Z_ERROR",
            Self::Depolarize1(_) => "DEPOLARIZE1",
            Self::Depolarize2(_) => "DEPOLARIZE2",
            Self::YError(_) => "Y_ERROR",
            Self::PauliChannel1(_) => "PAULI_CHANNEL_1",
            Self::PauliChannel2(_) => "PAULI_CHANNEL_2",
            Self::Leak(_) => "LEAK",
            Self::Seep(_) => "SEEP",
            Self::LeakTransport(_) => "LEAK_TRANSPORT",
        }
    }

    /// Numeric arguments in text order.
    pub(crate) fn args(&self) -> Vec<f64> {
        match self {
            Self::XError(p)
            | Self::ZError(p)
            | Self::Depolarize1(p)
            | Self::Depolarize2(p)
            | Self::YError(p)
            | Self::Leak(p)
            | Self::Seep(p)
            | Self::LeakTransport(p) => vec![*p],
            Self::PauliChannel1(probabilities) => probabilities.to_vec(),
            Self::PauliChannel2(probabilities) => probabilities.to_vec(),
        }
    }

    /// Whether targets are consumed in pairs.
    pub(crate) fn is_pair(&self) -> bool {
        matches!(
            self,
            Self::Depolarize2(_) | Self::PauliChannel2(_) | Self::LeakTransport(_)
        )
    }

    /// Single-qubit branch probabilities `(px, py, pz)`, `None` for a pair or leakage
    /// channel.
    pub(crate) fn single_rates(&self) -> Option<(f64, f64, f64)> {
        match *self {
            Self::XError(p) => Some((p, 0.0, 0.0)),
            Self::YError(p) => Some((0.0, p, 0.0)),
            Self::ZError(p) => Some((0.0, 0.0, p)),
            Self::Depolarize1(p) => Some((p / 3.0, p / 3.0, p / 3.0)),
            Self::PauliChannel1(ref p) => Some((p[0], p[1], p[2])),
            Self::Depolarize2(_)
            | Self::PauliChannel2(_)
            | Self::Leak(_)
            | Self::Seep(_)
            | Self::LeakTransport(_) => None,
        }
    }
}

/// One operation in a native QEC program.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum QecOp {
    /// Standard PRISM-Q gate operation. The compiled runner requires Clifford
    /// gates; the reference runner accepts any gate the statevector backend
    /// supports.
    Gate { gate: Gate, targets: Vec<usize> },
    /// Single-qubit measurement. Produces one record.
    Measure { basis: QecBasis, qubit: usize },
    /// Pauli-product (`MPP`) measurement. Produces one record, the parity of the
    /// listed terms.
    MeasurePauliProduct { terms: Vec<QecPauli> },
    /// Reset a qubit to the +1 eigenstate of the requested basis.
    Reset { basis: QecBasis, qubit: usize },
    /// Detector: parity over the listed measurement records. `coords` is
    /// arbitrary passthrough metadata for visualization and downstream
    /// decoders; it does not affect sampling.
    Detector {
        records: Vec<QecRecordRef>,
        coords: Vec<f64>,
    },
    /// Logical observable parity contribution. Multiple includes for the same
    /// `observable` index XOR into a single observable row.
    ObservableInclude {
        observable: usize,
        records: Vec<QecRecordRef>,
    },
    /// Final-state estimator `coefficient * <P>`, `P` the product of `terms`. Must be
    /// terminal (no gate, measurement, reset, or active noise may follow) and may
    /// reference only qubits not single-qubit-measured since their last reset.
    /// Estimates land in [`QecSampleResult::expectation_values`] in op order.
    ExpectationValue {
        terms: Vec<QecPauli>,
        coefficient: f64,
    },
    /// Postselection predicate. The shot is accepted only when the parity over
    /// `records` matches `expected`.
    Postselect {
        records: Vec<QecRecordRef>,
        expected: bool,
    },
    /// Feed-forward: `body` executes iff the parity over `records` equals
    /// `expected`.
    ///
    /// The predicate has the shape a detector has, because that is what
    /// adaptive correction reads. `body` admits gates and resets only: the
    /// record space is a static address space that detectors and observables
    /// index, so a measurement whose execution depends on a record would make
    /// those indices depend on the shot.
    ///
    /// [`run_qec_program`] samples a body of Pauli gates (`X`, `Y`, `Z`, `I`) on the
    /// compiled path as record flips. Any other body needs
    /// [`run_qec_program_reference`].
    Feedforward {
        records: Vec<QecRecordRef>,
        expected: bool,
        body: Vec<QecOp>,
    },
    /// Pauli-noise annotation applied at this point in the program.
    Noise {
        channel: QecNoise,
        targets: Vec<usize>,
    },
    /// Scheduling separator with no semantic effect, kept for the text format.
    Tick,
    /// Correlated Pauli-product error, `E`/`CORRELATED_ERROR` with its
    /// `ELSE_CORRELATED_ERROR` chain: branch `i` fires with its probability only when no
    /// earlier branch fired, so at most one branch's terms are applied.
    CorrelatedError { branches: Vec<QecCorrelatedBranch> },
    /// Heralded erasure: with probability `probability`, apply a uniformly random Pauli
    /// (identity included) to `qubit` and record 1; otherwise record 0. Produces one
    /// record.
    HeraldedErase { probability: f64, qubit: usize },
    /// Constant measurement record holding `value`. Produces one record.
    Pad { value: bool },
    /// Readout error: with probability `probability`, flip the record of the op directly
    /// before it and leave the state alone. Must directly follow an op that produces a
    /// record, or another `RecordFlip` of that record.
    RecordFlip { probability: f64 },
}

impl QecOp {
    /// True for a noise op that can fire: a Pauli annotation, correlated error, heralded
    /// erasure, or record flip with a nonzero probability.
    pub(crate) fn is_active_noise(&self) -> bool {
        match self {
            Self::Noise { channel, .. } => channel.probability() > 0.0,
            Self::CorrelatedError { branches } => branches.iter().any(|b| b.probability > 0.0),
            Self::HeraldedErase { probability, .. } | Self::RecordFlip { probability } => {
                *probability > 0.0
            }
            _ => false,
        }
    }

    /// True for the ops that append one measurement record.
    pub(crate) fn produces_record(&self) -> bool {
        matches!(
            self,
            Self::Measure { .. }
                | Self::MeasurePauliProduct { .. }
                | Self::HeraldedErase { .. }
                | Self::Pad { .. }
        )
    }
}
/// Options for running a native QEC program.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QecOptions {
    pub shots: usize,
    pub seed: u64,
    /// Maximum shots per compiled-runner batch, bounding peak measurement-matrix memory.
    /// `None` means one batch and `Some(0)` is rejected. The reference runner ignores it.
    /// Seeded noise draws in fixed 8192-shot units and does not depend on it.
    pub chunk_size: Option<usize>,
    /// When `false`, [`QecSampleResult::measurements`] keeps only its column count and
    /// holds zero shots; detectors and observables are always populated.
    pub keep_measurements: bool,
    /// What reported detector and observable bits are measured against.
    pub detector_reference: QecDetectorReference,
}

impl Default for QecOptions {
    fn default() -> Self {
        Self {
            shots: 1024,
            seed: 42,
            chunk_size: None,
            keep_measurements: true,
            detector_reference: QecDetectorReference::Noiseless,
        }
    }
}

/// Baseline that reported detector and observable bits are measured against.
///
/// The noiseless reference is one noiseless run of the program in which every
/// measurement whose outcome is random takes outcome 0, including the collapse a reset
/// performs. A detector or observable that is deterministic without noise then reports
/// 1 exactly when noise flipped it; one that is random reports a uniform bit, as raw
/// parity does. Programs with non-Clifford gates have no such run and report raw
/// parity under either setting. Postselection and erasure heralds always read raw
/// records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum QecDetectorReference {
    /// The parity XOR its value in the noiseless reference run.
    #[default]
    Noiseless,
    /// The parity of the measurement records the detector or observable names.
    Raw,
}

/// Native QEC program expressed as measurement-record operations.
#[derive(Debug, Clone, PartialEq)]
pub struct QecProgram {
    num_qubits: usize,
    ops: Vec<QecOp>,
    options: QecOptions,
    num_measurements: usize,
    num_detectors: usize,
}

fn qec_op_name(op: &QecOp) -> &'static str {
    match op {
        QecOp::Gate { .. } => "gate",
        QecOp::Measure { .. } => "M",
        QecOp::MeasurePauliProduct { .. } => "MPP",
        QecOp::Reset { .. } => "R",
        QecOp::Detector { .. } => "DETECTOR",
        QecOp::ObservableInclude { .. } => "OBSERVABLE_INCLUDE",
        QecOp::ExpectationValue { .. } => "EXP_VAL",
        QecOp::Postselect { .. } => "POSTSELECT",
        QecOp::Feedforward { .. } => "FEEDFORWARD",
        QecOp::Noise { .. } => "noise",
        QecOp::CorrelatedError { .. } => "E",
        QecOp::HeraldedErase { .. } => "HERALDED_ERASE",
        QecOp::Pad { .. } => "MPAD",
        QecOp::RecordFlip { .. } => "RECORD_FLIP",
        QecOp::Tick => "TICK",
    }
}

pub(crate) fn qec_terms_to_pauli(terms: &[QecPauli]) -> Vec<PauliTerm> {
    terms
        .iter()
        .map(|t| PauliTerm::new(t.qubit, t.basis.into()))
        .collect()
}

/// Reject reuse of a qubit measured in a non-Z basis before its next reset.
///
/// Both lowerings leave a basis-measured qubit in the Z frame, not the basis it named,
/// which is unobservable only while the qubit is reset before reuse (the contract
/// [`run_qec_program`] documents). Z measurements rotate nothing, and `MPP` undoes each
/// term's rotation before taking the record, so neither leaves a rotation behind.
pub(crate) fn validate_measured_qubit_reuse(program: &QecProgram) -> Result<()> {
    let reuse = |qubit: usize| PrismError::InvalidParameter {
        message: format!(
            "qubit {qubit} was measured in a non-Z basis and must be reset before it is used \
             again: a basis measurement leaves the qubit in the Z frame, not in the basis it \
             named"
        ),
    };
    let mut rotated = vec![false; program.num_qubits()];
    for op in program.ops() {
        match op {
            QecOp::Gate { targets, .. } => {
                if let Some(&qubit) = targets.iter().find(|&&q| rotated[q]) {
                    return Err(reuse(qubit));
                }
            }
            QecOp::Measure { basis, qubit } => {
                if rotated[*qubit] {
                    return Err(reuse(*qubit));
                }
                rotated[*qubit] = *basis != QecBasis::Z;
            }
            QecOp::MeasurePauliProduct { terms } => {
                if let Some(term) = terms.iter().find(|t| rotated[t.qubit]) {
                    return Err(reuse(term.qubit));
                }
            }
            QecOp::Reset { qubit, .. } => rotated[*qubit] = false,
            _ => {}
        }
    }
    Ok(())
}

/// Validate `EXP_VAL` placement for execution.
///
/// Terminality: no gate, measurement, reset, or active noise may follow an
/// `EXP_VAL` op, so "final state" is well defined on every path.
/// Liveness: an `EXP_VAL` term may not reference a qubit that was
/// single-qubit-measured after its last reset. Liveness keeps the sampled
/// post-measurement expectation equal to the measurement-stripped
/// pure-state expectation the analytical strategies evaluate (the Pauli
/// commutes with every measurement projector when their supports are
/// disjoint). Pauli-product measurements do not affect liveness: the
/// deferred lowering measures a scratch alias and the cross terms of the
/// projected state cancel exactly.
pub(crate) fn validate_qec_exp_val_placement(program: &QecProgram) -> Result<()> {
    let terminal_violation = |op_name: &str| PrismError::InvalidParameter {
        message: format!("`EXP_VAL` must be terminal: `{op_name}` appears after an `EXP_VAL` op"),
    };
    let mut seen_exp_val = false;
    let mut measured_since_reset = vec![false; program.num_qubits()];
    for op in program.ops() {
        match op {
            QecOp::ExpectationValue { terms, .. } => {
                seen_exp_val = true;
                if let Some(term) = terms.iter().find(|t| measured_since_reset[t.qubit]) {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "`EXP_VAL` term on qubit {}: qubit was measured after its last \
                             reset; expectation values are defined only on live qubits",
                            term.qubit
                        ),
                    });
                }
            }
            QecOp::Gate { gate, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation(gate.name()));
                }
            }
            QecOp::Measure { qubit, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("M"));
                }
                measured_since_reset[*qubit] = true;
            }
            QecOp::MeasurePauliProduct { .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("MPP"));
                }
            }
            QecOp::Reset { qubit, .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("R"));
                }
                measured_since_reset[*qubit] = false;
            }
            QecOp::Noise { channel, .. } if channel.probability() > 0.0 => {
                if seen_exp_val {
                    return Err(terminal_violation(channel.name()));
                }
            }
            QecOp::CorrelatedError { .. } if op.is_active_noise() => {
                if seen_exp_val {
                    return Err(terminal_violation("E"));
                }
            }
            QecOp::HeraldedErase { .. } | QecOp::Pad { .. } | QecOp::RecordFlip { .. } => {
                if seen_exp_val {
                    return Err(terminal_violation(qec_op_name(op)));
                }
            }
            QecOp::Feedforward { .. } => {
                if seen_exp_val {
                    return Err(terminal_violation("FEEDFORWARD"));
                }
                // A conditional reset does not restore liveness: the shots where
                // the predicate is false leave the qubit collapsed.
            }
            QecOp::Detector { .. }
            | QecOp::ObservableInclude { .. }
            | QecOp::Postselect { .. }
            | QecOp::Noise { .. }
            | QecOp::CorrelatedError { .. }
            | QecOp::Tick => {}
        }
    }
    Ok(())
}

pub(super) fn append_basis_to_z_rotation<S: GateSink>(sink: &mut S, basis: QecBasis, qubit: usize) {
    append_axis_to_z_rotation(sink, basis.into(), qubit);
}

pub(super) fn append_z_to_basis_rotation<S: GateSink>(sink: &mut S, basis: QecBasis, qubit: usize) {
    append_z_to_axis_rotation(sink, basis.into(), qubit);
}

/// Lower a Pauli-product measurement onto a scratch qubit holding |0>; the
/// caller measures the scratch afterward. See [`append_parity_rotations`].
pub(super) fn append_mpp_parity_rotations<S: GateSink>(
    sink: &mut S,
    terms: &[QecPauli],
    scratch: usize,
) {
    append_parity_rotations(sink, &qec_terms_to_pauli(terms), scratch);
}

/// Qubit count of a lowered program, including the scratch qubit at index
/// `program.num_qubits()` when the program measures any Pauli product, heralds an
/// erasure, or pads a record.
pub(super) fn qec_lowered_num_qubits(program: &QecProgram) -> usize {
    let needs_scratch = program.ops().iter().any(|op| {
        matches!(
            op,
            QecOp::MeasurePauliProduct { .. } | QecOp::HeraldedErase { .. } | QecOp::Pad { .. }
        )
    });
    program.num_qubits() + usize::from(needs_scratch)
}

pub(super) fn qec_non_clifford_error(gate: &Gate) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: "QEC compiled runner".to_string(),
        reason: format!(
            "compiled QEC runner requires Clifford gates, got `{}`",
            gate.name()
        ),
    }
}

pub(super) fn ensure_lowered_record_count(
    program: &QecProgram,
    produced: usize,
    stage: &str,
) -> Result<()> {
    if produced != program.num_measurements() {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC {stage} lowering produced {produced} records, expected {}",
                program.num_measurements()
            ),
        });
    }
    Ok(())
}

fn resolve_records(records: &[QecRecordRef], next_measurement: usize) -> Result<Vec<usize>> {
    records
        .iter()
        .map(|record| record.resolve(next_measurement))
        .collect()
}
