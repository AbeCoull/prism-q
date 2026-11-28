//! Simulation orchestration: routes a circuit to a backend and answers the
//! terminal queries. Entry points: [`simulate`], [`run_qasm`], [`run_on`].

mod batch;
pub mod braket;
mod branching;
pub mod calibration;
mod clifford_register;
pub mod compiled;
mod decomposed;
mod diagnostic;
mod dispatch;
mod dynamic;
mod expectation;
pub mod gradient;
pub mod homological;
mod initial_state;
mod metadata;
pub mod noise;
pub(crate) mod observable;
mod plan;
mod probability;
mod route;
pub use plan::{
    MemoryEstimate, MemoryEstimateKind, Plan, PlanFallback, PlannedExactness, Query, RejectReason,
    RouteRule,
};
use route::*;
mod shot_routes;
pub(crate) mod shots;
mod simulate;
pub mod stabilizer_rank;
mod terminal_sampling;
pub(crate) mod trajectory;
pub mod unified_pauli;

pub use batch::run_batch;
use batch::*;
pub(crate) use batch::{PreparedRoute, prepared_route};
#[cfg(feature = "parallel")]
pub(crate) use batch::{runs_split_across_workers, state_splits_across_workers};
pub use braket::ResultValue;
pub(crate) use decomposed::merge_probabilities;
use decomposed::{
    MIN_DECOMPOSITION_QUBITS, PreparedBlocks, run_decomposed, run_decomposed_prefused,
    should_decompose,
};
use diagnostic::*;
pub use dispatch::BackendKind;
use dispatch::{
    BackendPlan, ExecutionPlan, Family, MAX_AUTO_MID_CIRCUIT_BRANCH_T, MIN_BLOCK_FOR_FACTORED_STAB,
    MIN_FACTORED_STABILIZER_QUBITS, MIN_QUBITS_FOR_SPD_AUTO, accel_for, approximate_route_name,
    auto_selects_cpu_statevector, auto_spd_work_budget, build_statevector,
    has_temporal_clifford_opportunity, initial_state_plan, mps_apply_cost, plan_for_family,
    plan_temporal_clifford, resolve, resolve_backend, run_temporal_clifford, statevector_work,
    validate_explicit_backend, validate_kind_parameters,
};
pub use dynamic::{SimulateProgram, simulate_program};
#[cfg(all(test, feature = "gpu"))]
pub(crate) use expectation::host_group_moments;
use expectation::*;
pub use expectation::{run_expectation_values, run_observable_expectation};
use initial_state::*;
pub use metadata::{
    BondReport, Engine, Exactness, ExpectationResult, Placement, ResolvedBackend, RunMetadata,
};
#[cfg(feature = "distributed")]
pub(crate) use observable::pauli_sandwich;
pub use observable::{ObservableExpectation, PauliObservable};
pub(crate) use observable::{
    finish_expectations, i_pow, pauli_expectation_from_masks, pauli_expectations_from_masks,
    pauli_masks, pauli_sandwiches_from_masks, validate_observable,
};
pub use probability::{FactoredBlock, Probabilities, ProbabilitiesIter};
use shot_routes::*;
pub(crate) use shot_routes::{
    run_counts_with, run_shots_with, run_shots_with_noise, supports_compiled_measurement_sampling,
};
pub use shots::{ShotsResult, bitstring};
pub use simulate::{Seeded, Simulate, Unseeded, simulate};

use std::collections::HashMap;

use num_complex::Complex64;

use crate::backend::Backend;
use crate::backend::sparse::MAX_SPARSE_INDEX_QUBITS;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::parity_ancilla::rewrite_parity_ancillas;
use crate::circuit::{Circuit, Instruction, SaveRecord, SaveSpec, SavedValue};
use crate::error::{PrismError, Result};
use crate::limits::{Limits, ResolvedLimits};
use crate::sim::noise::NoiseModel;
use shots::{
    counts_from_basis_samples, packed_shots_to_classical_bits, record_counts_to_classical_bits,
    sample_shots, shots_from_basis_samples,
};
use terminal_sampling::{
    sample_counts_from_probs, sample_counts_from_state, sample_counts_from_walk,
    sample_shots_from_probs, sample_shots_from_state, sample_shots_from_walk,
};
use unified_pauli::{PauliTerm, SpdTruncation};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SimOptions {
    pub(crate) probabilities: bool,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            probabilities: true,
        }
    }
}

impl SimOptions {
    pub(crate) fn classical_only() -> Self {
        Self {
            probabilities: false,
        }
    }
}

/// Result of a generic simulation run.
#[derive(Debug, Clone)]
pub struct RunOutcome {
    /// Classical measurement outcomes, indexed by classical bit number.
    /// `true` = measured |1⟩.
    pub classical_bits: Vec<bool>,
    /// Probability of each computational basis state (length 2^n).
    ///
    /// `None` means the selected backend cannot expose a dense probability
    /// distribution for this circuit. Other probability extraction failures
    /// are returned as errors by the query that produced this result.
    pub probabilities: Option<Probabilities>,
    /// Which engine ran, whether the answer is exact, and where the state lived.
    pub metadata: RunMetadata,
    /// What each [`Instruction::Save`] point recorded, in the order the points
    /// were reached. Empty unless the circuit carries save points.
    pub saves: Vec<SaveRecord>,
}

/// Frequency histogram returned by query-aware count sampling.
#[derive(Debug, Clone)]
pub struct CountsResult {
    /// Histogram keyed by packed classical bits; same key layout as
    /// [`ShotsResult::counts`], formattable with [`bitstring`].
    pub counts: HashMap<Vec<u64>, u64>,
    pub num_classical_bits: usize,
    pub metadata: RunMetadata,
}

impl CountsResult {
    pub fn into_counts(self) -> HashMap<Vec<u64>, u64> {
        self.counts
    }
}

/// Per-qubit marginal probabilities returned by query-aware marginal sampling.
#[derive(Debug, Clone)]
pub struct MarginalsResult {
    /// `(P(0), P(1))` per qubit, indexed by qubit number.
    pub marginals: Vec<(f64, f64)>,
    pub metadata: RunMetadata,
}

impl MarginalsResult {
    pub fn into_vec(self) -> Vec<(f64, f64)> {
        self.marginals
    }
}

/// Reduced density matrix of a qubit subset, returned by
/// [`Simulate::reduced_density_matrix`].
#[derive(Debug, Clone)]
pub struct ReducedDensityMatrix {
    /// The subsystem as it was requested, which fixes the index order below.
    pub qubits: Vec<usize>,
    /// Row major with side `2^k` over `k = qubits.len()`: `data[t * 2^k + t']`
    /// is `<t|rho|t'>`, where bit `i` of `t` is the state of `qubits[i]`, so
    /// `qubits[0]` is the lowest bit as `q[0]` is in a basis index. Trace one,
    /// Hermitian to rounding.
    pub data: Vec<Complex64>,
    pub metadata: RunMetadata,
}

impl ReducedDensityMatrix {
    /// `Tr(rho^2)`: 1 for a pure marginal, `2^-k` for the maximally mixed one.
    pub fn purity(&self) -> f64 {
        self.data.iter().map(|entry| entry.norm_sqr()).sum()
    }
}

/// Operator variance of a weighted Pauli observable, returned by
/// [`Simulate::observable_variance`].
#[derive(Debug, Clone)]
pub struct ObservableVariance {
    /// `<H^2> - <H>^2` on the output state.
    pub variance: f64,
    /// `<H>` on the same state, evaluated on the way to the variance.
    pub mean: f64,
    pub metadata: RunMetadata,
}

/// Entanglement entropy of a subsystem, returned by
/// [`Simulate::entanglement_entropy`].
#[derive(Debug, Clone)]
pub struct EntropyResult {
    /// The subsystem as it was requested, the side of the cut the entropy is
    /// read on.
    pub subsystem: Vec<usize>,
    /// Von Neumann entropy of the subsystem in nats: a Bell pair reads `ln 2`.
    pub entropy: f64,
    /// Schmidt values across the cut, descending, with squares summing to 1.
    /// `None` where the backend holds the entropy without the spectrum that
    /// stands behind it, as a stabilizer cut past the export cap does: its
    /// `2^r` equal values do not fit. A factored cut past that cap does the
    /// same, holding each block's spectrum but not their product.
    pub schmidt_values: Option<Vec<f64>>,
    pub metadata: RunMetadata,
}

/// Overlap between the output states of two runs, returned by
/// [`Simulate::overlap`].
#[derive(Debug, Clone)]
pub struct OverlapResult {
    /// `|<a|b>|^2` over the two normalized states: 1 for the same state up to
    /// phase, 0 for orthogonal ones. The amplitude itself is not reported,
    /// since a tableau keeps no global phase and every MPS truncation moves
    /// one.
    pub fidelity: f64,
    /// Provenance of the run the terminal was called on, the left of the
    /// inner product.
    pub left: RunMetadata,
    /// Provenance of the run passed as the argument.
    pub right: RunMetadata,
}

/// A caller-supplied start: a `2^n` amplitude vector, or a `4^n` mixture in
/// the density-matrix layout.
#[derive(Debug, Clone, Copy)]
pub(crate) enum StartState<'c> {
    Amplitudes(&'c [Complex64]),
    DensityMatrix(&'c [Complex64]),
}

impl StartState<'_> {
    /// Load the start into `backend`, whose register width the buffer sets.
    fn load(self, backend: &mut dyn Backend, num_classical_bits: usize) -> Result<()> {
        match self {
            Self::Amplitudes(amplitudes) => {
                backend.init_from_amplitudes(amplitudes.to_vec(), num_classical_bits)
            }
            Self::DensityMatrix(rho) => {
                backend.init_from_density_matrix(rho.to_vec(), num_classical_bits)
            }
        }
    }
}

fn require_exact_mixture(kind: &BackendKind, terminal: &str) -> Result<()> {
    if kind.is_density_matrix() {
        return Ok(());
    }
    let reason = if terminal == "entanglement entropy" {
        "entanglement entropy requires a pure state without noise; a noisy circuit evolves a \
         mixed state, which has no Schmidt decomposition"
            .into()
    } else {
        format!(
            "{terminal} under a noise model reads the exact mixed state, which only the \
             density-matrix backend holds; select it, or average trajectories through `shots` \
             or `sample_counts`"
        )
    };
    Err(PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason,
    })
}

/// Gate for the terminals that answer from the state, mixed or pure, which
/// readout error is not part of.
///
/// The condition is the one under which a draw would actually differ. Entries
/// past the circuit's bit count never apply, matching what
/// `sample_exact_noisy_shots` consumes, so a model built for a wider circuit is
/// not rejected on bits this one does not have, and a zero-rate entry flips
/// nothing, so a sweep that starts at zero is not rejected at its first point.
fn reject_readout_at(
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    terminal: &str,
) -> Result<()> {
    let inert = noise_model
        .readout
        .iter()
        .take(circuit.num_classical_bits)
        .all(|entry| entry.as_ref().is_none_or(|readout| readout.is_inert()));
    if inert {
        return Ok(());
    }
    Err(PrismError::InvalidParameter {
        message: format!(
            "{terminal} answers from the mixed state, which readout error is not part of: it \
             acts on the measurement record and is indexed by classical bit, not qubit. Drop \
             it from the model, or use `shots` or `sample_counts`, which apply it"
        ),
    })
}

/// Expand the gate forms `backend` has no native kernel for, `QftBlock` and
/// `PauliRot`, or on a backend with the `QftBlock` kernel fold expanded QFTs into
/// blocks. The stream stays borrowed when nothing changes.
pub(crate) fn expand_for_backend<'c>(
    backend: &dyn Backend,
    circuit: &'c Circuit,
) -> std::borrow::Cow<'c, Circuit> {
    use std::borrow::Cow;
    let expanded = if backend.supports_qft_block() {
        crate::circuit::recognize_qft_blocks(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    if backend.supports_pauli_rotation() {
        return expanded;
    }
    match expanded {
        Cow::Borrowed(borrowed) => crate::circuit::expand_pauli_rotations(borrowed),
        Cow::Owned(owned) => {
            let rotations = crate::circuit::expand_pauli_rotations(&owned);
            if let Cow::Owned(expanded_rotations) = rotations {
                return Cow::Owned(expanded_rotations);
            }
            Cow::Owned(owned)
        }
    }
}

/// Fuse `circuit` against what `backend` accepts and how wide a buffer it
/// sweeps, the two backend facts the pass pipeline is gated on.
pub(crate) fn fuse_for_backend<'a>(
    backend: &dyn Backend,
    circuit: &'a Circuit,
) -> std::borrow::Cow<'a, Circuit> {
    crate::circuit::fusion::fuse_circuit_for_profile(
        circuit,
        backend.supports_fused_gates(),
        backend.fusion_state_qubits(circuit.num_qubits),
        backend.fusion_profile(circuit.num_qubits),
    )
}

/// Build the statevector `kind` selects, fuse `circuit` for it, initialize it, run
/// `before_apply`, then evolve the state. The hook sits after the `init` that rejects an
/// oversized register and before the `2^n` evolution. The flag reports whether a device
/// was requested.
fn evolved_statevector<T>(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    limits: &ResolvedLimits,
    before_apply: impl FnOnce() -> Result<T>,
) -> Result<(StatevectorBackend, bool, T)> {
    let accel = accel_for(kind, Family::Statevector, circuit.num_qubits);
    let mut backend = build_statevector(&accel, seed, limits);
    let expanded = expand_for_backend(&backend, circuit);
    let fused = fuse_for_backend(&backend, &expanded);
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    let hooked = before_apply()?;
    backend.apply_instructions(&fused.instructions)?;
    Ok((backend, !matches!(accel, dispatch::Accel::Cpu), hooked))
}

/// [`apply_fused_circuit`] for a terminal that has nowhere to put a save.
///
/// Only [`Simulate::run`] returns save records, so every other terminal
/// declines a circuit carrying a save point rather than running it and
/// discarding what the points recorded.
fn apply_fused_without_saves(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    terminal: &str,
) -> Result<()> {
    decline_saves(circuit, backend.name(), terminal)?;
    apply_fused_circuit(backend, circuit).map(|_| ())
}

/// Fuse `circuit` for `backend` and apply it, leaving initialization to the
/// caller. The start-state analogue of [`execute`], which owns the |0...0⟩ init.
fn apply_fused_circuit(backend: &mut dyn Backend, circuit: &Circuit) -> Result<Vec<SaveRecord>> {
    let expanded = expand_for_backend(&*backend, circuit);
    let fused = fuse_for_backend(&*backend, &expanded);
    apply_recording_saves(backend, &fused.instructions)
}

/// Reject a result that turned out approximate, for a caller that opted out.
///
/// The backstop behind [`reject_approximate_route`], which decides from the
/// circuit alone. Sparse Pauli dynamics truncates on coefficient magnitudes it
/// only learns while propagating, so whether it stayed exact is not knowable
/// before the run.
fn ensure_exact_result(require_exact: bool, metadata: &RunMetadata) -> Result<()> {
    if require_exact && !metadata.is_exact() {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{:?}", metadata.backend),
            reason: "require_exact rejects an approximate result; this engine discarded state \
                     weight while running, which the route could not predict"
                .into(),
        });
    }
    Ok(())
}

/// Reject an approximate route for a caller that opted out of one.
fn reject_approximate_route(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Result<()> {
    reject_predicted_route(approximate_route_name(kind, circuit, limits))
}

/// The engine is named in the error, since a caller who asked for exactness
/// wants to know which one would have answered.
fn reject_predicted_route(engine: Option<&str>) -> Result<()> {
    match engine {
        Some(engine) => Err(PrismError::IncompatibleBackend {
            backend: engine.into(),
            reason: "require_exact rejects a route that can discard state weight; drop the \
                     requirement to accept the approximation, which the result reports, or \
                     select a backend that represents this circuit exactly"
                .into(),
        }),
        None => Ok(()),
    }
}

/// `subject` names what needs the unitary circuit with its verb, so the
/// sentence reads for a plural terminal ("expectation values require") and for
/// a singular one ("a reduced density matrix requires").
fn require_unitary_circuit(kind: &BackendKind, circuit: &Circuit, subject: &str) -> Result<()> {
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "{subject} a unitary circuit without measurements, resets, or conditionals"
            ),
        });
    }
    Ok(())
}

/// [`require_unitary_circuit`] for the expectation terminals. When resets are the
/// only obstacle on another backend, the error names the density matrix, which
/// evolves them as channels.
fn require_unitary_expectation_circuit(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    let checked = require_unitary_circuit(kind, circuit, "expectation values require");
    if checked.is_err()
        && !kind.is_density_matrix()
        && evolves_resets_as_channels(&BackendKind::DensityMatrix, circuit)
    {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "expectation values require a unitary circuit without measurements, \
                     resets, or conditionals; `BackendKind::DensityMatrix` evolves resets as \
                     channels"
                .into(),
        });
    }
    checked
}

/// Whether the expectation terminals evolve `circuit` on the density matrix with
/// its resets as channels: a density-matrix `kind`, a reset, and no measurement,
/// conditional, or region, each of which the unitary check still rejects.
fn evolves_resets_as_channels(kind: &BackendKind, circuit: &Circuit) -> bool {
    kind.is_density_matrix()
        && circuit.has_resets()
        && !circuit.instructions.iter().any(|inst| {
            matches!(
                inst,
                Instruction::Measure { .. }
                    | Instruction::Conditional { .. }
                    | Instruction::Region(_)
            )
        })
}

/// Provenance of a result read off the exact mixture: the density matrix,
/// placed wherever `kind` holds it.
fn exact_mixture_metadata(kind: &BackendKind) -> RunMetadata {
    let metadata =
        RunMetadata::exact(ResolvedBackend::DensityMatrix).with_reason(RouteRule::ExactMixture);
    #[cfg(feature = "gpu")]
    if matches!(kind, BackendKind::DensityMatrixGpu { .. }) {
        let mut on_device = metadata;
        on_device.placement = Placement::Device;
        return on_device;
    }
    #[cfg(not(feature = "gpu"))]
    let _ = kind;
    metadata
}

/// Exact output distribution of `circuit` under `noise_model`, evolved once on
/// the density matrix. The mixture carries every measurement branch at once,
/// so it answers for a whole shot only when the measurements are terminal.
fn exact_noisy_probabilities(
    kind: &BackendKind,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    initial_state: Option<StartState<'_>>,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<Probabilities> {
    Ok(Probabilities::Dense(noise::density_matrix_probabilities(
        kind,
        circuit,
        noise_model,
        initial_state,
        seed,
        limits,
    )?))
}

/// Draw classical bits from the exact noisy distribution, then apply readout
/// error. Readout acts on the outcome rather than on the state, so it is not
/// carried by the distribution and has to be applied per draw, on a stream of
/// its own so it does not track the state draws.
fn sample_exact_noisy_shots(
    probs: &Probabilities,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Vec<Vec<bool>> {
    let bits = circuit.num_classical_bits;
    let mut shots = sample_shots(probs, &circuit.measurement_map(), bits, num_shots, seed);
    if noise_model.readout.iter().any(Option::is_some) {
        let readout = trajectory::written_readout(circuit, &noise_model.readout);
        let mut rng = trajectory::noise_rng(seed);
        for shot in &mut shots {
            trajectory::apply_readout_errors(shot, &readout, &mut rng);
        }
    }
    shots
}

#[inline]
fn probs_only_result(
    probs: Vec<f64>,
    num_classical_bits: usize,
    metadata: RunMetadata,
) -> RunOutcome {
    RunOutcome {
        probabilities: Some(Probabilities::Dense(probs)),
        classical_bits: vec![false; num_classical_bits],
        metadata,
        saves: Vec::new(),
    }
}

fn decline_saves(circuit: &Circuit, backend: impl std::fmt::Display, terminal: &str) -> Result<()> {
    if let Some((_, label)) = circuit.save_points().next() {
        return Err(PrismError::IncompatibleBackend {
            backend: backend.to_string(),
            reason: format!("{terminal} cannot return save point `{label}`"),
        });
    }
    Ok(())
}

fn try_backend_probabilities(backend: &dyn Backend) -> Result<Option<Probabilities>> {
    if let Some(factored) = backend.block_probabilities() {
        return Ok(Some(factored));
    }
    match backend.probabilities() {
        Ok(probs) => Ok(Some(Probabilities::Dense(probs))),
        Err(PrismError::BackendUnsupported { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Core execution: fuse, init, apply, extract.
fn execute(backend: &mut dyn Backend, circuit: &Circuit, opts: &SimOptions) -> Result<RunOutcome> {
    let expanded = expand_for_backend(&*backend, circuit);
    let fused = fuse_for_backend(&*backend, &expanded);
    execute_circuit(backend, &fused, opts)
}

/// Shared init → apply → extract logic.
fn execute_circuit(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<RunOutcome> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let saves = apply_recording_saves(backend, &circuit.instructions)?;

    let probabilities = if opts.probabilities {
        try_backend_probabilities(backend)?
    } else {
        None
    };

    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
        metadata: backend_metadata(backend),
        saves,
    })
}

/// Apply `instructions`, reading the state at each save point.
///
/// The stream is handed to the backend in the segments between save points, so
/// no backend implements a save and none can batch across one. A circuit with
/// no save points takes one call.
fn apply_recording_saves(
    backend: &mut dyn Backend,
    instructions: &[Instruction],
) -> Result<Vec<SaveRecord>> {
    let first = instructions
        .iter()
        .position(|i| matches!(i, Instruction::Save { .. }));
    let Some(first) = first else {
        backend.apply_instructions(instructions)?;
        return Ok(Vec::new());
    };

    let mut saves = Vec::new();
    let mut start = 0;
    for (i, inst) in instructions.iter().enumerate().skip(first) {
        let Instruction::Save { spec, label, .. } = inst else {
            continue;
        };
        backend.apply_instructions(&instructions[start..i])?;
        saves.push(SaveRecord {
            label: label.clone(),
            value: read_save(backend, *spec, label)?,
        });
        start = i + 1;
    }
    backend.apply_instructions(&instructions[start..])?;
    Ok(saves)
}

/// Read one save off the backend that is mid-circuit.
///
/// Every spec maps to an export the backend already offers, so a backend that
/// cannot produce the requested form fails here with its own message rather
/// than needing a save implementation of its own.
fn read_save(backend: &mut dyn Backend, spec: SaveSpec, label: &str) -> Result<SavedValue> {
    match spec {
        SaveSpec::StateVector => backend.export_statevector().map(SavedValue::StateVector),
        SaveSpec::Probabilities => backend.probabilities().map(SavedValue::Probabilities),
        SaveSpec::DensityMatrix => {
            let whole: Vec<usize> = (0..backend.num_qubits()).collect();
            backend
                .reduced_density_matrix(&whole)
                .map(SavedValue::DensityMatrix)
        }
    }
    .map_err(|err| name_save_error(err, label))
}

fn name_save_error(mut error: PrismError, label: &str) -> PrismError {
    let text = match &mut error {
        PrismError::InvalidParameter { message } => message,
        PrismError::IncompatibleBackend { reason, .. } => reason,
        PrismError::BackendUnsupported { operation, .. } => operation,
        PrismError::ResourceLimit(limit) => &mut limit.operation,
        _ => return error,
    };
    *text = format!("save point `{label}`: {text}");
    error
}

/// Provenance read off the engine that ran, after it ran. Exactness and
/// placement are reports rather than predictions: the MPS bound reflects what
/// this run discarded, and the placement reflects where the amplitudes ended up
/// after any device fallback.
fn route_metadata(backend: &dyn Backend, expected_device: bool, reason: RouteRule) -> RunMetadata {
    let mut metadata = backend_metadata(backend);
    metadata.reason = if expected_device && metadata.placement == Placement::Host {
        RouteRule::DeviceInitializationFallback
    } else {
        reason
    };
    metadata
}

pub(crate) fn backend_metadata(backend: &dyn Backend) -> RunMetadata {
    let mut metadata =
        RunMetadata::new(backend.resolved(), backend.exactness(), backend.placement());
    metadata.bond = backend.bond_report();
    metadata
}

/// Build the backend `kind` selects and run the circuit on it. For a
/// pre-constructed backend, use [`run_on`].
pub(crate) fn run_with(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    run_with_internal(kind, circuit, seed, SimOptions::default(), limits)
}

fn run_with_internal(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    validate_explicit_backend(&kind, circuit, limits)?;
    // The distributed backend runs the whole circuit across ranks in lockstep.
    // Subsystem decomposition, Clifford+T, and temporal-Clifford shortcuts all
    // reshape execution per sub-block, which would desynchronize the collective
    // calls every rank must issue in the same order. Dispatch directly.
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        let mut backend = resolve_backend(&kind, circuit, false, limits).build(seed, limits);
        if opts.probabilities {
            // The gather cap follows from the register width alone, so a
            // register past it is rejected here rather than after the run has
            // been paid for. Same check `probabilities()` applies, so the
            // message does not depend on where it surfaced.
            crate::backend::dense_probability_len(backend.name(), circuit.num_qubits, limits)?;
        }
        return execute(&mut *backend, circuit, &opts).map(|mut result| {
            result.metadata.reason = RouteRule::ExplicitBackend;
            result
        });
    }
    let route = plan_probability_route(&kind, circuit, limits);
    run_route(&kind, circuit, seed, opts, &route, limits)
}

/// Execute one seed of a planned probability route. Shot loops plan once and
/// call this per seed.
fn run_route(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
    route: &ProbabilityRoute,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    match route {
        ProbabilityRoute::FactoredStabilizer => {
            let mut backend =
                crate::backend::factored_stabilizer::FactoredStabilizerBackend::with_limits(
                    seed, limits,
                );
            let fs_opts = if circuit.num_qubits >= usize::BITS as usize {
                SimOptions {
                    probabilities: false,
                }
            } else {
                opts
            };
            execute(&mut backend, circuit, &fs_opts).map(|mut r| {
                r.metadata.reason = backend_rule(kind, circuit, &BackendPlan::FactoredStabilizer);
                r
            })
        }
        ProbabilityRoute::Decomposed(components) => {
            run_decomposed(kind, components, circuit, seed, &opts, limits).map(|mut r| {
                r.metadata.reason = RouteRule::IndependentBlocks;
                r
            })
        }
        ProbabilityRoute::TemporalClifford {
            has_partial_independence,
        } => match plan_temporal_clifford(kind, circuit, limits) {
            Some(tc) => run_temporal_clifford(&tc, seed, opts.probabilities).map(|mut r| {
                r.metadata.reason = RouteRule::TemporalCliffordPrefix;
                r
            }),
            None => run_direct(kind, circuit, seed, opts, *has_partial_independence, limits),
        },
        ProbabilityRoute::Direct {
            has_partial_independence,
        } => run_direct(kind, circuit, seed, opts, *has_partial_independence, limits),
    }
}

fn run_direct(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
    has_partial_independence: bool,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    let route = resolve(kind, circuit, has_partial_independence, limits);
    if !matches!(route, ExecutionPlan::Backend(_)) {
        decline_saves(circuit, format_args!("{kind:?}"), "a probability-only run")?;
    }
    match route {
            ExecutionPlan::Backend(plan) => {
                let mut backend = plan.build(seed, limits);
                execute(&mut *backend, circuit, &opts).map(|mut r| {
                    r.metadata.reason = if plan.is_gpu() && r.metadata.placement == Placement::Host { RouteRule::DeviceInitializationFallback } else { backend_rule(kind, circuit, &plan) }; r
                })
            }
            ExecutionPlan::StabilizerRank => {
                if opts.probabilities {
                    crate::backend::statevector_probability_len(
                        "stabilizer_rank",
                        circuit.num_qubits,
                        limits,
                    )?;
                }
                let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
                Ok(probs_only_result(
                    sr.probabilities,
                    circuit.num_classical_bits,
                    RunMetadata::exact(ResolvedBackend::StabilizerRank)
                        .with_reason(RouteRule::ExplicitBackend),
                ))
            }
            ExecutionPlan::StochasticPauli { num_samples } => {
                Err(crate::error::PrismError::IncompatibleBackend {
                    backend: format!(
                        "{:?}",
                        BackendKind::StochasticPauli { num_samples }
                    ),
                    reason: "StochasticPauli produces marginal estimates only; use `simulate(...).marginals()`".into(),
                })
            }
            ExecutionPlan::DeterministicPauli { truncation } => {
                Err(crate::error::PrismError::IncompatibleBackend {
                    backend: format!(
                        "{:?}",
                        BackendKind::DeterministicPauli { truncation }
                    ),
                    reason: "DeterministicPauli produces marginals only; use `simulate(...).marginals()`".into(),
                })
            }
            ExecutionPlan::PauliPath => Err(reject_pauli_path("a single run")),
    }
}

/// Execute a circuit on a pre-constructed backend. For automatic dispatch,
/// use [`simulate`].
pub fn run_on(backend: &mut dyn Backend, circuit: &Circuit) -> Result<RunOutcome> {
    execute(backend, circuit, &SimOptions::default())
}

/// Execute a circuit on a pre-constructed backend from a start state other than
/// |0...0⟩, the [`run_on`] sibling for a caller holding its own amplitudes.
///
/// The backend must accept one; see [`Backend::init_from_amplitudes`] for the
/// validation applied to `initial_state` and for which backends decline it.
pub fn run_on_state(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    initial_state: &[Complex64],
) -> Result<RunOutcome> {
    check_initial_state_len(StartState::Amplitudes(initial_state), circuit.num_qubits)?;
    backend.init_from_amplitudes(initial_state.to_vec(), circuit.num_classical_bits)?;
    let saves = apply_fused_circuit(backend, circuit)?;
    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities: try_backend_probabilities(backend)?,
        metadata: backend_metadata(backend),
        saves,
    })
}

/// Width below which separate runs split across Rayon workers, above the kernels'
/// own parallel floor: on a 4-core host a 200-circuit two-layer ansatz batch split
/// one circuit per core took 18% of the unsplit time at 14 qubits and 53% at 16, and
/// 10% more at 18. Under miri it follows the kernels' floor down.
/// Bracketed by `batch/hea_l2_sweep` (`run_batch/16` and `run_batch/18` against `loop`).
#[cfg(all(feature = "parallel", not(miri)))]
const RUN_SPLIT_QUBITS: usize = 17;
#[cfg(all(feature = "parallel", miri))]
const RUN_SPLIT_QUBITS: usize = crate::backend::PARALLEL_THRESHOLD_QUBITS;

/// Parse an OpenQASM string and execute with automatic backend selection.
pub fn run_qasm(qasm: &str, seed: u64) -> Result<RunOutcome> {
    let circuit = crate::circuit::openqasm::parse(qasm)?;
    simulate(&circuit).seed(seed).run()
}

/// Components to decompose with, and whether every gate is Clifford.
type Decomposition = (Vec<Vec<usize>>, bool);

/// Independence analysis shared by the routing prelude in
/// `run_with_internal`, the shots slow path, and the terminal fast-path
/// candidacy. Returns the decomposition when full decomposition should fire,
/// plus the partial-independence flag otherwise.
fn analyze_independence(circuit: &Circuit) -> (Option<Decomposition>, bool) {
    if circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        let subsystems = circuit.subsystems_and_clifford();
        let partial = subsystems.interaction.len() > 1;
        let components = subsystems.closed.unwrap_or(subsystems.interaction);
        if components.len() > 1 && should_decompose(&components, circuit.num_qubits) {
            return (Some((components, subsystems.clifford_only)), false);
        }
        return (None, partial);
    }
    (None, false)
}

/// T count after lowering of a Clifford+T circuit holding a T gate, guarded ones included.
fn auto_clifford_t_count(circuit: &Circuit) -> Option<usize> {
    (circuit.is_clifford_plus_t() && circuit.has_t_gates()).then(|| circuit.t_count())
}

/// Narrowest register `Auto` samples through the Clifford register rather than the
/// statevector. At 12 qubits the statevector still wins, 0.8 against 1.3 ms for 1024
/// shots; at 16 the register takes 1.5 ms against 3.3.
/// No bench row brackets this yet.
#[cfg(not(prism_lowered_thresholds))]
const MIN_AUTO_CLIFFORD_REGISTER_QUBITS: usize = 16;
#[cfg(prism_lowered_thresholds)]
const MIN_AUTO_CLIFFORD_REGISTER_QUBITS: usize = 4;

pub(super) fn has_nonunitary_or_classical_ops(circuit: &Circuit) -> bool {
    circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_)
        )
    })
}

/// Pair-expansion budget per commuting group. The dedicated pass pays a scratch
/// copy and a basis rotation per group that the pair expansion avoids. On the
/// 2000-string Jordan-Wigner fixture, sending every multi-term group through it
/// cost about 3x this budget's reduction at n=16 and n=20, and about 0.6x at
/// n=12.
/// Bracketed by `expectation/pauli_sum_grouped` (`grouped/16`, `grouped/20`).
const MAX_PAIR_MASKS_PER_GROUP: usize = 20;

pub(crate) const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seed for item `index` (a shot, trajectory, block, qubit or observable) of a run
/// seeded `seed`. Both inputs pass through SplitMix64, so runs on nearby seeds draw
/// unrelated streams rather than shifted copies of one another.
pub(crate) const fn mix_seed(seed: u64, index: usize) -> u64 {
    splitmix64(splitmix64(seed) ^ index as u64)
}

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "gpu"))]
mod gpu_stub_tests;

#[cfg(test)]
mod terminal_candidate_matrix_tests;

#[cfg(test)]
mod diagnostic_terminal_tests;

#[cfg(test)]
mod route_ledger_tests;
