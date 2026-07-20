//! Simulation orchestration.
//!
//! Connects the circuit IR to a backend. This module is deliberately thin,
//! the complexity lives in the backends and the parser.

pub mod compiled;
mod decomposed;
mod dispatch;
pub mod homological;
pub mod noise;
mod probability;
pub(crate) mod shots;
pub mod stabilizer_rank;
mod terminal_sampling;
mod trajectory;
pub mod unified_pauli;

pub(crate) use decomposed::merge_probabilities;
use decomposed::{
    MIN_DECOMPOSITION_QUBITS, run_decomposed, run_decomposed_prefused, should_decompose,
};
pub use dispatch::BackendKind;
use dispatch::{
    AUTO_APPROX_MAX_TERMS, AUTO_SPD_MAX_TERMS, BackendPlan, ExecutionPlan, Family,
    MAX_AUTO_T_COUNT_APPROX, MAX_AUTO_T_COUNT_EXACT, MAX_AUTO_T_COUNT_SHOTS,
    MAX_STABILIZER_RANK_QUBITS, MIN_BLOCK_FOR_FACTORED_STAB, MIN_FACTORED_STABILIZER_QUBITS,
    MIN_QUBITS_FOR_SPD_AUTO, accel_for, auto_selects_cpu_statevector, build_statevector,
    has_temporal_clifford_opportunity, plan_for_family, plan_temporal_clifford, resolve,
    resolve_backend, run_temporal_clifford, stabilizer_rank_budget, validate_explicit_backend,
};
pub use probability::{FactoredBlock, Probabilities, ProbabilitiesIter};
pub use shots::{ShotsResult, bitstring};

use std::collections::HashMap;

use num_complex::Complex64;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use shots::{packed_shots_to_classical_bits, sample_shots};
use terminal_sampling::{
    sample_counts_from_probs, sample_counts_from_state, sample_shots_from_probs,
    sample_shots_from_state,
};
use unified_pauli::{PauliAxis, PauliTerm};

type TerminalStatevector = (StatevectorBackend, Vec<(usize, usize)>);

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
}

/// Frequency histogram returned by query-aware count sampling.
#[derive(Debug, Clone)]
pub struct CountsResult {
    pub counts: HashMap<Vec<u64>, u64>,
    pub num_classical_bits: usize,
}

impl CountsResult {
    pub fn into_counts(self) -> HashMap<Vec<u64>, u64> {
        self.counts
    }
}

/// Per-qubit marginal probabilities returned by query-aware marginal sampling.
#[derive(Debug, Clone)]
pub struct MarginalsResult {
    pub marginals: Vec<(f64, f64)>,
}

impl MarginalsResult {
    pub fn into_vec(self) -> Vec<(f64, f64)> {
        self.marginals
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Unseeded;

#[derive(Debug, Clone, Copy)]
pub struct Seeded {
    seed: u64,
}

/// Builder for query-aware simulation requests.
pub struct Simulate<'c, SeedState> {
    circuit: &'c Circuit,
    kind: BackendKind,
    seed: SeedState,
    noise_model: Option<&'c noise::NoiseModel>,
}

impl<'c, SeedState> Simulate<'c, SeedState> {
    #[inline]
    pub fn backend(mut self, kind: BackendKind) -> Self {
        self.kind = kind;
        self
    }

    #[inline]
    pub fn noise(mut self, model: &'c noise::NoiseModel) -> Self {
        self.noise_model = Some(model);
        self
    }

    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu(self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.backend(BackendKind::StatevectorGpu { context })
    }

    /// Automatic backend selection with GPU acceleration opted in via `context`.
    ///
    /// Routes like [`BackendKind::Auto`], but a selected statevector or
    /// stabilizer block that clears the qubit crossover with VRAM to spare runs
    /// on the device. Unsupported cases fall back to the identical CPU path.
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_auto(self, context: std::sync::Arc<crate::gpu::GpuContext>) -> Self {
        self.backend(BackendKind::AutoGpu { context })
    }

    /// Distribute the exact state vector across the ranks of `context`.
    ///
    /// With a single rank this behaves like [`Simulate::backend`] with
    /// [`BackendKind::Statevector`].
    #[cfg(feature = "distributed")]
    pub fn distributed(
        self,
        context: std::sync::Arc<crate::distributed::DistributedContext>,
    ) -> Self {
        self.backend(BackendKind::StatevectorDistributed { context })
    }
}

impl<'c> Simulate<'c, Unseeded> {
    #[inline]
    pub fn seed(self, seed: u64) -> Simulate<'c, Seeded> {
        Simulate {
            circuit: self.circuit,
            kind: self.kind,
            seed: Seeded { seed },
            noise_model: self.noise_model,
        }
    }
}

impl<'c> Simulate<'c, Seeded> {
    #[inline]
    fn seed_value(&self) -> u64 {
        self.seed.seed
    }

    #[inline]
    pub fn run(self) -> Result<RunOutcome> {
        let seed = self.seed_value();
        if self.noise_model.is_some() {
            return Err(crate::error::PrismError::BackendUnsupported {
                backend: format!("{:?}", self.kind),
                operation: "single-run noisy simulation through `run`".into(),
            });
        }
        run_with_internal(self.kind, self.circuit, seed, SimOptions::default())
    }

    #[inline]
    pub fn shots(self, num_shots: usize) -> Result<ShotsResult> {
        let seed = self.seed_value();
        if let Some(noise_model) = self.noise_model {
            run_shots_with_noise(self.kind, self.circuit, noise_model, num_shots, seed)
        } else {
            run_shots_with(self.kind, self.circuit, num_shots, seed)
        }
    }

    #[inline]
    pub fn sample_counts(self, num_shots: usize) -> Result<CountsResult> {
        let seed = self.seed_value();
        let counts = if let Some(noise_model) = self.noise_model {
            run_shots_with_noise(self.kind, self.circuit, noise_model, num_shots, seed)?.counts()
        } else {
            run_counts_with(self.kind, self.circuit, num_shots, seed)?
        };
        Ok(CountsResult {
            counts,
            num_classical_bits: self.circuit.num_classical_bits,
        })
    }

    #[inline]
    pub fn marginals(self) -> Result<MarginalsResult> {
        let seed = self.seed_value();
        if self.noise_model.is_some() {
            return Err(crate::error::PrismError::BackendUnsupported {
                backend: format!("{:?}", self.kind),
                operation: "marginals with inline noise model".into(),
            });
        }
        run_marginals_result_with(self.kind, self.circuit, seed)
    }

    /// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on the circuit's
    /// output state, honoring the selected backend.
    ///
    /// Each observable is a product of single-qubit Paulis (identity factors
    /// omitted). The circuit must be unitary. Clifford circuits propagate each
    /// observable exactly; non-Clifford circuits use the state vector.
    #[inline]
    pub fn expectation_values(self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let seed = self.seed_value();
        if self.noise_model.is_some() {
            return Err(PrismError::BackendUnsupported {
                backend: format!("{:?}", self.kind),
                operation: "expectation values with an inline noise model".into(),
            });
        }
        run_expectation_values_with(self.kind, self.circuit, observables, seed)
    }
}

#[inline]
pub fn simulate(circuit: &Circuit) -> Simulate<'_, Unseeded> {
    Simulate {
        circuit,
        kind: BackendKind::Auto,
        seed: Unseeded,
        noise_model: None,
    }
}

#[inline]
fn probs_only_result(probs: Vec<f64>) -> RunOutcome {
    RunOutcome {
        probabilities: Some(Probabilities::Dense(probs)),
        classical_bits: vec![],
    }
}

fn try_backend_probabilities(backend: &dyn Backend) -> Result<Option<Probabilities>> {
    match backend.probabilities() {
        Ok(probs) => Ok(Some(Probabilities::Dense(probs))),
        Err(PrismError::BackendUnsupported { .. }) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Core execution: fuse, init, apply, extract.
fn execute(backend: &mut dyn Backend, circuit: &Circuit, opts: &SimOptions) -> Result<RunOutcome> {
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    execute_circuit(backend, &fused, opts)
}

/// Shared init → apply → extract logic.
fn execute_circuit(
    backend: &mut dyn Backend,
    circuit: &Circuit,
    opts: &SimOptions,
) -> Result<RunOutcome> {
    backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
    backend.apply_instructions(&circuit.instructions)?;

    let probabilities = if opts.probabilities {
        try_backend_probabilities(backend)?
    } else {
        None
    };

    Ok(RunOutcome {
        classical_bits: backend.classical_results().to_vec(),
        probabilities,
    })
}

/// Execute a circuit with automatic backend selection.
///
/// The simplest entry point. Uses [`BackendKind::Auto`] to select the
/// optimal backend based on circuit properties, then runs the circuit.
#[cfg(test)]
fn run(circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    run_with(BackendKind::Auto, circuit, seed)
}

/// Execute a circuit with explicit backend selection.
///
/// Constructs the backend internally based on [`BackendKind`], then runs
/// the circuit. For a pre-constructed backend instance, use [`run_on`].
pub(crate) fn run_with(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
    run_with_internal(kind, circuit, seed, SimOptions::default())
}

fn run_with_internal(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
    opts: SimOptions,
) -> Result<RunOutcome> {
    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }
    // The distributed backend runs the whole circuit across ranks in lockstep.
    // Subsystem decomposition, Clifford+T, and temporal-Clifford shortcuts all
    // reshape execution per sub-block, which would desynchronize the collective
    // calls every rank must issue in the same order. Dispatch directly.
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        let mut backend = resolve_backend(&kind, circuit, false).build(seed);
        return execute(&mut *backend, circuit, &opts);
    }
    let (decompose, has_partial_independence) = analyze_independence(circuit);
    if let Some(components) = decompose {
        let max_block = components.iter().map(|c| c.len()).max().unwrap_or(0);
        if kind.is_auto()
            && circuit.is_clifford_only()
            && circuit.num_qubits >= MIN_FACTORED_STABILIZER_QUBITS
            && max_block >= MIN_BLOCK_FOR_FACTORED_STAB
        {
            let mut backend =
                crate::backend::factored_stabilizer::FactoredStabilizerBackend::new(seed);
            let fs_opts = if circuit.num_qubits > 64 {
                SimOptions {
                    probabilities: false,
                }
            } else {
                opts
            };
            return execute(&mut backend, circuit, &fs_opts);
        }
        return run_decomposed(&kind, &components, circuit, seed, &opts);
    }
    if kind.is_auto() && circuit.num_qubits <= MAX_STABILIZER_RANK_QUBITS {
        if let Some((t, sr_budget)) = auto_clifford_t_budget(circuit) {
            if t <= MAX_AUTO_T_COUNT_APPROX
                && t <= sr_budget
                && !has_nonunitary_or_classical_ops(circuit)
            {
                let sr = if t <= MAX_AUTO_T_COUNT_EXACT {
                    stabilizer_rank::run_stabilizer_rank(circuit, seed)?
                } else {
                    stabilizer_rank::run_stabilizer_rank_approx(
                        circuit,
                        AUTO_APPROX_MAX_TERMS,
                        seed,
                    )?
                };
                return Ok(probs_only_result(sr.probabilities));
            }
        }
    }
    if let Some(tc) = plan_temporal_clifford(&kind, circuit) {
        return run_temporal_clifford(&tc, seed, opts.probabilities);
    }
    match resolve(&kind, circuit, has_partial_independence) {
        ExecutionPlan::Backend(plan) => {
            let mut backend = plan.build(seed);
            execute(&mut *backend, circuit, &opts)
        }
        ExecutionPlan::StabilizerRank => {
            let sr = stabilizer_rank::run_stabilizer_rank(circuit, seed)?;
            Ok(probs_only_result(sr.probabilities))
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
        ExecutionPlan::DeterministicPauli { epsilon, max_terms } => {
            Err(crate::error::PrismError::IncompatibleBackend {
                backend: format!(
                    "{:?}",
                    BackendKind::DeterministicPauli { epsilon, max_terms }
                ),
                reason: "DeterministicPauli produces marginals only; use `simulate(...).marginals()`".into(),
            })
        }
    }
}

/// Execute a circuit on a pre-constructed backend.
///
/// Use this when you need direct control over the backend instance
/// (e.g., testing a specific backend). For automatic dispatch, use [`simulate`].
pub fn run_on(backend: &mut dyn Backend, circuit: &Circuit) -> Result<RunOutcome> {
    execute(backend, circuit, &SimOptions::default())
}

/// Parse an OpenQASM string and execute with automatic backend selection.
pub fn run_qasm(qasm: &str, seed: u64) -> Result<RunOutcome> {
    let circuit = crate::circuit::openqasm::parse(qasm)?;
    simulate(&circuit).seed(seed).run()
}

/// Execute a circuit multiple times, collecting measurement outcomes.
#[cfg(test)]
fn run_shots(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<ShotsResult> {
    run_shots_with(BackendKind::Auto, circuit, num_shots, seed)
}

pub(crate) fn supports_compiled_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && !circuit.has_resets()
        && circuit.has_terminal_measurements_only()
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
}

fn supports_deferred_measurement_sampling(circuit: &Circuit) -> bool {
    circuit.is_clifford_only()
        && (circuit.has_resets() || !circuit.has_terminal_measurements_only())
        && circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Measure { .. }))
        && !circuit
            .instructions
            .iter()
            .any(|inst| matches!(inst, Instruction::Conditional { .. }))
}

fn is_clifford_sampler_kind(kind: &BackendKind) -> bool {
    if kind.is_auto() {
        return true;
    }
    match kind {
        BackendKind::Stabilizer | BackendKind::FactoredStabilizer => true,
        #[cfg(feature = "gpu")]
        BackendKind::StabilizerGpu { .. } => true,
        _ => false,
    }
}

fn should_use_compiled_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 2
        && supports_compiled_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

fn should_use_deferred_clifford_sampling(
    kind: &BackendKind,
    circuit: &Circuit,
    num_shots: usize,
) -> bool {
    num_shots >= 2
        && supports_deferred_measurement_sampling(circuit)
        && is_clifford_sampler_kind(kind)
}

fn compile_measurements_for_kind(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<compiled::CompiledSampler> {
    #[cfg(not(feature = "gpu"))]
    let _ = kind;

    let sampler = compiled::compile_measurements(circuit, seed)?;

    #[cfg(feature = "gpu")]
    if let BackendKind::StabilizerGpu { context } = kind {
        return Ok(sampler.with_gpu(context.clone()));
    }

    Ok(sampler)
}

/// Independence analysis shared by the routing prelude in
/// `run_with_internal`, the shots slow path, and the terminal fast-path
/// candidacy. Returns the components to decompose with when full
/// decomposition should fire, plus the partial-independence flag otherwise.
fn analyze_independence(circuit: &Circuit) -> (Option<Vec<Vec<usize>>>, bool) {
    if circuit.num_qubits >= MIN_DECOMPOSITION_QUBITS {
        let components = circuit.independent_subsystems();
        if components.len() > 1 {
            if should_decompose(&components, circuit.num_qubits) {
                return (Some(components), false);
            }
            return (None, true);
        }
    }
    (None, false)
}

/// `(t_count, stabilizer_rank_budget)` when the auto Clifford+T family gate
/// passes. Callers apply their own per-entry-point T-count ceilings.
fn auto_clifford_t_budget(circuit: &Circuit) -> Option<(usize, usize)> {
    (circuit.is_clifford_plus_t() && circuit.has_t_gates()).then(|| {
        (
            circuit.t_count(),
            stabilizer_rank_budget(circuit.num_qubits),
        )
    })
}

/// Mirrors the routing precedence of `run_with_internal` (decomposition, then
/// Clifford+T stabilizer rank, then temporal Clifford, then family choice)
/// through the shared predicates. Keep the check order aligned when either
/// side changes.
fn auto_terminal_statevector_candidate(circuit: &Circuit) -> bool {
    let (decompose, has_partial_independence) = analyze_independence(circuit);
    if decompose.is_some() {
        return false;
    }

    if !auto_selects_cpu_statevector(circuit, has_partial_independence) {
        return false;
    }

    if circuit.num_qubits <= MAX_STABILIZER_RANK_QUBITS {
        if let Some((t, sr_budget)) = auto_clifford_t_budget(circuit) {
            if t <= MAX_AUTO_T_COUNT_APPROX && t <= sr_budget {
                return false;
            }
        }
    }

    !has_temporal_clifford_opportunity(&BackendKind::Auto, circuit)
}

fn terminal_statevector_candidate(kind: &BackendKind, circuit: &Circuit) -> bool {
    if kind.is_auto() {
        return auto_terminal_statevector_candidate(circuit);
    }
    match kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => false,
    }
}

fn try_terminal_statevector_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<Option<TerminalStatevector>> {
    if !circuit.has_terminal_measurements_only() {
        return Ok(None);
    }

    let meas_map = circuit.measurement_map();
    if meas_map.is_empty() {
        return Ok(None);
    }

    let stripped = circuit.without_measurements();
    if !terminal_statevector_candidate(kind, &stripped) {
        return Ok(None);
    }

    let accel = accel_for(kind, Family::Statevector, stripped.num_qubits);
    let mut backend = build_statevector(&accel, seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(&stripped)
    } else {
        crate::circuit::expand_qft_blocks(&stripped)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    backend.apply_instructions(&fused.instructions)?;

    Ok(Some((backend, meas_map)))
}

/// Execute a circuit multiple times with automatic backend selection and return counts.
///
/// Use this when only a frequency histogram is needed. Optimized paths can
/// avoid materializing per-shot bit vectors.
#[cfg(test)]
fn run_counts(circuit: &Circuit, num_shots: usize, seed: u64) -> Result<HashMap<Vec<u64>, u64>> {
    run_counts_with(BackendKind::Auto, circuit, num_shots, seed)
}

/// Execute a circuit multiple times with explicit backend selection and return counts.
///
/// For Clifford circuits with terminal measurements and no resets, Auto,
/// Stabilizer, FactoredStabilizer, and explicit `StabilizerGpu` route through
/// the compiled sampler's optimized counting path. Explicit `StabilizerGpu`
/// carries its GPU context into the compiled sampler so large shot runs avoid
/// the raw tableau measurement round-trips. Other circuits fall back to
/// per-shot simulation with counting.
///
/// Optimized terminal statevector paths sample counts directly from the output
/// distribution. The distribution is equivalent to materializing shots first,
/// but finite seeded counts may differ from `run_shots_with(...).counts()`.
pub(crate) fn run_counts_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<HashMap<Vec<u64>, u64>> {
    if should_use_compiled_clifford_sampling(&kind, circuit, num_shots) {
        let mut sampler = compile_measurements_for_kind(&kind, circuit, seed)?;
        return sampler.try_sample_counts(num_shots);
    }

    if let Some((backend, meas_map)) = try_terminal_statevector_backend(&kind, circuit, seed)? {
        if backend.is_gpu_resident() {
            let probs = backend.probabilities()?;
            return Ok(sample_counts_from_probs(
                &probs,
                &meas_map,
                circuit.num_classical_bits,
                num_shots,
                seed,
            ));
        }
        return Ok(sample_counts_from_state(
            backend.state_vector(),
            backend.probability_scale(),
            &meas_map,
            circuit.num_classical_bits,
            num_shots,
            seed,
        ));
    }

    let result = run_shots_with(kind, circuit, num_shots, seed)?;
    Ok(result.counts())
}

/// Compute per-qubit marginal probabilities: P(q_i = 0) and P(q_i = 1) for each qubit.
///
/// Returns a `Vec<(f64, f64)>` of length `num_qubits`, where each element is `(p0, p1)`.
///
/// For Clifford+T circuits, uses Sparse Pauli Dynamics (SPD), Heisenberg-picture
/// backward propagation that scales with Pauli complexity, not 2^n. This is orders
/// of magnitude faster than statevector for structured Clifford+T circuits (25-400x
/// at 14-22 qubits).
///
/// For pure Clifford circuits, exact marginals still come from backend
/// probabilities. Sampled parity-matrix marginals remain available through
/// `compile_measurements(...).sample_marginals(...)`.
///
/// For other circuits, falls back to statevector probabilities and extracts
/// marginals.
#[cfg(test)]
fn run_marginals(circuit: &Circuit, seed: u64) -> Result<Vec<(f64, f64)>> {
    run_marginals_result_with(BackendKind::Auto, circuit, seed).map(MarginalsResult::into_vec)
}

/// Compute per-qubit marginal probabilities with explicit backend selection.
#[cfg(test)]
fn run_marginals_with(kind: BackendKind, circuit: &Circuit, seed: u64) -> Result<Vec<(f64, f64)>> {
    run_marginals_result_with(kind, circuit, seed).map(MarginalsResult::into_vec)
}

fn expectations_to_marginals(expectations: &[f64]) -> Vec<(f64, f64)> {
    expectations
        .iter()
        .map(|ez| {
            let p0 = ((1.0 + ez) / 2.0).clamp(0.0, 1.0);
            (p0, 1.0 - p0)
        })
        .collect()
}

fn has_nonunitary_or_classical_ops(circuit: &Circuit) -> bool {
    circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
        )
    })
}

fn supports_pauli_marginal_backend(circuit: &Circuit) -> bool {
    circuit.is_clifford_plus_t() && !has_nonunitary_or_classical_ops(circuit)
}

fn validate_pauli_marginal_backend(kind: &BackendKind, circuit: &Circuit) -> Result<()> {
    if !circuit.is_clifford_plus_t() {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli marginal backends require Clifford+T gates".into(),
        });
    }
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli marginal backends require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }
    Ok(())
}

fn run_marginals_result_with(
    kind: BackendKind,
    circuit: &Circuit,
    seed: u64,
) -> Result<MarginalsResult> {
    let n = circuit.num_qubits;

    match &kind {
        BackendKind::StochasticPauli { num_samples } => {
            validate_pauli_marginal_backend(&kind, circuit)?;
            let spp = unified_pauli::run_spp(circuit, *num_samples, seed)?;
            return Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spp.expectations),
            });
        }
        BackendKind::DeterministicPauli { epsilon, max_terms } => {
            validate_pauli_marginal_backend(&kind, circuit)?;
            let spd = unified_pauli::run_spd(circuit, *epsilon, *max_terms)?;
            return Ok(MarginalsResult {
                marginals: expectations_to_marginals(&spd.expectations),
            });
        }
        _ => {}
    }

    if kind.is_auto()
        && supports_pauli_marginal_backend(circuit)
        && circuit.has_t_gates()
        && n >= MIN_QUBITS_FOR_SPD_AUTO
    {
        let spd = unified_pauli::run_spd(circuit, 0.0, AUTO_SPD_MAX_TERMS)?;
        return Ok(MarginalsResult {
            marginals: expectations_to_marginals(&spd.expectations),
        });
    }

    let result = run_with(kind, circuit, seed)?;
    if let Some(probs) = &result.probabilities {
        Ok(MarginalsResult {
            marginals: probs.marginals(),
        })
    } else {
        Err(PrismError::BackendUnsupported {
            backend: "simulate".into(),
            operation: format!(
                "marginals for {} qubits without backend probability output",
                circuit.num_qubits
            ),
        })
    }
}

/// Compute `⟨ψ|P|ψ⟩` for each joint Pauli observable on a unitary circuit's
/// output state, using automatic backend selection. See
/// [`Simulate::expectation_values`] for explicit backend control.
pub fn run_expectation_values(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<Vec<f64>> {
    run_expectation_values_with(BackendKind::Auto, circuit, observables, seed)
}

fn run_expectation_values_with(
    kind: BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<Vec<f64>> {
    if has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "expectation values require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }

    match &kind {
        BackendKind::StochasticPauli { num_samples } => observables
            .iter()
            .enumerate()
            .map(|(i, obs)| {
                unified_pauli::run_spp_observable(
                    circuit,
                    obs,
                    *num_samples,
                    seed.wrapping_add(i as u64),
                )
                .map(|r| r.mean)
            })
            .collect(),
        BackendKind::DeterministicPauli { epsilon, max_terms } => observables
            .iter()
            .map(|obs| {
                unified_pauli::run_spd_observable(circuit, obs, *epsilon, *max_terms)
                    .map(|r| r.mean)
            })
            .collect(),
        _ if kind.is_auto()
            || matches!(
                kind,
                BackendKind::Stabilizer | BackendKind::FactoredStabilizer
            ) =>
        {
            if circuit.is_clifford_only() {
                observables
                    .iter()
                    .map(|obs| {
                        unified_pauli::run_spd_observable(circuit, obs, 0.0, 0).map(|r| r.mean)
                    })
                    .collect()
            } else if kind.is_auto() {
                if circuit.num_qubits > max_statevector_qubits() {
                    return Err(PrismError::IncompatibleBackend {
                        backend: format!("{kind:?}"),
                        reason: format!(
                            "expectation values for a {}-qubit non-Clifford circuit exceed the statevector cap ({} qubits); no exact accelerated path exists for arbitrary Pauli observables here",
                            circuit.num_qubits,
                            max_statevector_qubits()
                        ),
                    });
                }
                expectation_values_statevector(&kind, circuit, observables, seed)
            } else {
                Err(PrismError::IncompatibleBackend {
                    backend: format!("{kind:?}"),
                    reason: "stabilizer backends require a Clifford-only circuit".into(),
                })
            }
        }
        BackendKind::Statevector => {
            expectation_values_statevector(&kind, circuit, observables, seed)
        }
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => {
            expectation_values_statevector(&kind, circuit, observables, seed)
        }
        other => {
            #[cfg(feature = "gpu")]
            let supported = "expectation values support Auto, AutoGpu, Statevector, StatevectorGpu, Stabilizer, FactoredStabilizer, StochasticPauli, and DeterministicPauli";
            #[cfg(not(feature = "gpu"))]
            let supported = "expectation values support Auto, Statevector, Stabilizer, FactoredStabilizer, StochasticPauli, and DeterministicPauli";
            Err(PrismError::IncompatibleBackend {
                backend: format!("{other:?}"),
                reason: supported.into(),
            })
        }
    }
}

fn expectation_values_statevector(
    kind: &BackendKind,
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    seed: u64,
) -> Result<Vec<f64>> {
    // Validate before the 2^n simulation so bad observables fail cheaply.
    let masks = observables
        .iter()
        .map(|obs| pauli_masks(obs, circuit.num_qubits))
        .collect::<Result<Vec<_>>>()?;

    let accel = accel_for(kind, Family::Statevector, circuit.num_qubits);
    let mut backend = build_statevector(&accel, seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if backend.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, backend.supports_fused_gates());
    backend.init(fused.num_qubits, fused.num_classical_bits)?;
    backend.apply_instructions(&fused.instructions)?;

    let exported;
    let state: &[Complex64] = if backend.is_gpu_resident() {
        exported = backend.export_statevector()?;
        &exported
    } else {
        backend.state_vector()
    };
    let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
    Ok(masks
        .iter()
        .map(|&(xmask, zmask, num_y)| {
            pauli_expectation_from_masks(state, xmask, zmask, num_y, norm)
        })
        .collect())
}

/// Validate a joint Pauli observable and reduce it to `(Xmask, Zmask, #Y)`,
/// where `Xmask` covers X and Y factors and `Zmask` covers Z and Y factors.
fn pauli_masks(observable: &[PauliTerm], num_qubits: usize) -> Result<(usize, usize, u32)> {
    let mut xmask = 0usize;
    let mut zmask = 0usize;
    let mut num_y = 0u32;
    let mut seen = vec![false; num_qubits];
    for term in observable {
        if term.qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
        let bit = 1usize << term.qubit;
        match term.axis {
            PauliAxis::X => xmask |= bit,
            PauliAxis::Z => zmask |= bit,
            PauliAxis::Y => {
                xmask |= bit;
                zmask |= bit;
                num_y += 1;
            }
        }
    }
    Ok((xmask, zmask, num_y))
}

/// Exact `⟨ψ|P|ψ⟩` from the reduced observable masks, where `P` acts as
/// `P|j⟩ = i^{#Y}·(-1)^{popcount(j & Zmask)}·|j ⊕ Xmask⟩`. Normalization
/// independent, so raw backend amplitudes are fine.
fn pauli_expectation_from_masks(
    state: &[Complex64],
    xmask: usize,
    zmask: usize,
    num_y: u32,
    norm: f64,
) -> f64 {
    if norm == 0.0 {
        return 0.0;
    }

    let mut acc = Complex64::new(0.0, 0.0);
    for (j, &amp) in state.iter().enumerate() {
        let partner = state[j ^ xmask];
        let sign = if (j & zmask).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        acc += partner.conj() * amp * sign;
    }

    let i_pow = match num_y % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    (acc * i_pow).re / norm
}

/// Multi-shot execution for the distributed statevector backend.
///
/// Every rank runs this function in lockstep. Circuits with only terminal
/// measurements run once and sample basis indices without gathering the dense
/// state on any rank. Circuits with mid-circuit measurements run once per shot,
/// prefused, with per-shot seeds matching the generic slow path.
#[cfg(feature = "distributed")]
fn run_shots_distributed(
    context: std::sync::Arc<crate::distributed::DistributedContext>,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    use crate::backend::distributed_statevector::DistributedStatevectorBackend;

    let meas_map = circuit.measurement_map();
    if meas_map.is_empty() {
        // No measurements means every shot is all false, but init must still
        // run so invalid rank counts and local qubit floor violations surface as
        // errors instead of fabricated output.
        let mut backend = DistributedStatevectorBackend::new(context, seed);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        return Ok(ShotsResult {
            shots: vec![vec![false; circuit.num_classical_bits]; num_shots],
            num_classical_bits: circuit.num_classical_bits,
        });
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let mut backend = DistributedStatevectorBackend::new(context, seed);
        execute(&mut backend, &stripped, &SimOptions::classical_only())?;
        let indices = backend.sample_state_indices(num_shots, seed)?;
        let shots = indices
            .iter()
            .map(|&idx| {
                let mut shot = vec![false; circuit.num_classical_bits];
                for &(qubit, cbit) in &meas_map {
                    shot[cbit] = (idx >> qubit) & 1 == 1;
                }
                shot
            })
            .collect();
        return Ok(ShotsResult {
            shots,
            num_classical_bits: circuit.num_classical_bits,
        });
    }

    let probe = DistributedStatevectorBackend::new(context.clone(), seed);
    let expanded: std::borrow::Cow<'_, Circuit> = if probe.supports_qft_block() {
        std::borrow::Cow::Borrowed(circuit)
    } else {
        crate::circuit::expand_qft_blocks(circuit)
    };
    let fused = crate::circuit::fusion::fuse_circuit(&expanded, probe.supports_fused_gates());
    let opts = SimOptions::classical_only();
    let mut shots = Vec::with_capacity(num_shots);
    for i in 0..num_shots {
        let shot_seed = seed.wrapping_add(i as u64);
        let mut backend = DistributedStatevectorBackend::new(context.clone(), shot_seed);
        let result = execute_circuit(&mut backend, &fused, &opts)?;
        shots.push(result.classical_bits);
    }
    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

/// Execute a circuit multiple times with explicit backend selection.
pub(crate) fn run_shots_with(
    kind: BackendKind,
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    // The distributed backend runs every rank in lockstep, so shot execution
    // must not route through shortcuts that reshape the collective call
    // sequence. Dispatch directly.
    #[cfg(feature = "distributed")]
    if let BackendKind::StatevectorDistributed { context } = &kind {
        return run_shots_distributed(context.clone(), circuit, num_shots, seed);
    }

    // Compiled sampler: O(n²·m) compile + O(r·m/64) per shot with LUT.
    // Always polynomial, avoids the O(2^k) probability computation path.
    // Explicit `StabilizerGpu` attaches its CUDA context here so repeated shot
    // runs use the compiled GPU sampling path instead of the raw GPU tableau
    // measurement loop, but only for circuits the compiled sampler models
    // exactly.
    if should_use_compiled_clifford_sampling(&kind, circuit, num_shots) {
        let mut sampler = compile_measurements_for_kind(&kind, circuit, seed)?;
        let packed = sampler.try_sample_bulk_packed(num_shots)?;
        let meas_map = circuit.measurement_map();
        return Ok(ShotsResult {
            shots: packed_shots_to_classical_bits(&packed, &meas_map, circuit.num_classical_bits),
            num_classical_bits: circuit.num_classical_bits,
        });
    }

    if should_use_deferred_clifford_sampling(&kind, circuit, num_shots) {
        if let Ok(deferred) = compiled::defer_measure_reset_circuit(circuit) {
            let mut sampler = compile_measurements_for_kind(&kind, &deferred, seed)?;
            let packed = sampler.try_sample_bulk_packed(num_shots)?;
            let meas_map = deferred.measurement_map();
            return Ok(ShotsResult {
                shots: packed_shots_to_classical_bits(
                    &packed,
                    &meas_map,
                    circuit.num_classical_bits,
                ),
                num_classical_bits: circuit.num_classical_bits,
            });
        }
    }

    if let Some((backend, meas_map)) = try_terminal_statevector_backend(&kind, circuit, seed)? {
        let shots = if backend.is_gpu_resident() {
            let probs = backend.probabilities()?;
            sample_shots_from_probs(
                &probs,
                &meas_map,
                circuit.num_classical_bits,
                num_shots,
                seed,
            )
        } else {
            sample_shots_from_state(
                backend.state_vector(),
                backend.probability_scale(),
                &meas_map,
                circuit.num_classical_bits,
                num_shots,
                seed,
            )
        };
        return Ok(ShotsResult {
            shots,
            num_classical_bits: circuit.num_classical_bits,
        });
    }

    if matches!(kind, BackendKind::StabilizerRank) && circuit.has_t_gates() {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }
    if kind.is_auto()
        && circuit.has_terminal_measurements_only()
        && circuit.num_qubits > MAX_STABILIZER_RANK_QUBITS
    {
        if let Some((t, sr_budget)) = auto_clifford_t_budget(circuit) {
            if t <= MAX_AUTO_T_COUNT_SHOTS && t <= sr_budget {
                return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
            }
        }
    }

    if circuit.has_terminal_measurements_only() {
        let stripped = circuit.without_measurements();
        let result = run_with_internal(kind.clone(), &stripped, seed, SimOptions::default())?;
        if let Some(probs) = result.probabilities {
            let meas_map = circuit.measurement_map();
            let shots = sample_shots(
                &probs,
                &meas_map,
                circuit.num_classical_bits,
                num_shots,
                seed,
            );
            return Ok(ShotsResult {
                shots,
                num_classical_bits: circuit.num_classical_bits,
            });
        }
    }

    // Slow path: mid-circuit measurements require per-shot simulation.
    // Pre-compute seed-independent analysis to avoid redundant work.
    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }

    let (decompose, has_partial_independence) = analyze_independence(circuit);

    if matches!(kind, BackendKind::StabilizerRank) {
        return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
    }
    if matches!(
        kind,
        BackendKind::StochasticPauli { .. } | BackendKind::DeterministicPauli { .. }
    ) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "Pauli propagation backends do not support mid-circuit measurements".into(),
        });
    }
    if kind.is_auto() {
        if let Some((t, sr_budget)) = auto_clifford_t_budget(circuit) {
            if t <= MAX_AUTO_T_COUNT_SHOTS && t <= sr_budget {
                return stabilizer_rank::run_stabilizer_rank_shots(circuit, num_shots, seed);
            }
        }
    }

    if has_temporal_clifford_opportunity(&kind, circuit) {
        if decompose.is_none() {
            if let Some(tc) = plan_temporal_clifford(&kind, circuit) {
                return collect_shots(circuit, num_shots, seed, |shot_seed| {
                    Ok(run_temporal_clifford(&tc, shot_seed, false)?.classical_bits)
                });
            }
        }
        // Decomposable circuits with a temporal prefix keep the per-shot
        // full-pipeline route; the prefix spans blocks that decomposition
        // would otherwise split.
        let opts = SimOptions::classical_only();
        return collect_shots(circuit, num_shots, seed, |shot_seed| {
            Ok(run_with_internal(kind.clone(), circuit, shot_seed, opts)?.classical_bits)
        });
    }

    let opts = SimOptions::classical_only();

    if let Some(ref comps) = decompose {
        let partitions = circuit.partition_subcircuits(comps);
        let block_plans: Vec<BackendPlan> = partitions
            .iter()
            .map(|(sub, _, _)| {
                if !kind.is_auto() {
                    validate_explicit_backend(&kind, sub)?;
                }
                Ok(resolve_backend(&kind, sub, false))
            })
            .collect::<Result<_>>()?;
        let fused_blocks: Vec<_> = partitions
            .iter()
            .zip(&block_plans)
            .map(|((sub, _, _), plan)| {
                crate::circuit::fusion::fuse_circuit(sub, plan.supports_fused())
            })
            .collect();

        collect_shots(circuit, num_shots, seed, |shot_seed| {
            let result = run_decomposed_prefused(
                &block_plans,
                comps,
                &partitions,
                &fused_blocks,
                shot_seed,
                &opts,
                circuit,
            )?;
            Ok(result.classical_bits)
        })
    } else {
        let plan = resolve_backend(&kind, circuit, has_partial_independence);
        let fused = crate::circuit::fusion::fuse_circuit(circuit, plan.supports_fused());

        collect_shots(circuit, num_shots, seed, |shot_seed| {
            let mut backend = plan.build(shot_seed);
            Ok(execute_circuit(&mut *backend, &fused, &opts)?.classical_bits)
        })
    }
}

fn collect_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
    mut shot: impl FnMut(u64) -> Result<Vec<bool>>,
) -> Result<ShotsResult> {
    let mut shots = Vec::with_capacity(num_shots);
    for i in 0..num_shots {
        shots.push(shot(seed.wrapping_add(i as u64))?);
    }
    Ok(ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

/// Family choice for auto-routed non-Pauli noise trajectories. Restricted to
/// families whose trajectory operations (1q Kraus, qubit probability, reduced
/// density matrix, reset) are supported; the statevector leaf carries the
/// kind's acceleration.
fn general_noise_plan(kind: &BackendKind, circuit: &Circuit) -> BackendPlan {
    let family = if !circuit.has_entangling_gates() {
        Family::ProductState
    } else if circuit.num_qubits > max_statevector_qubits() {
        if circuit.is_sparse_friendly() {
            Family::Sparse
        } else {
            Family::Mps
        }
    } else {
        Family::Statevector
    };
    plan_for_family(kind, family, circuit.num_qubits)
}

/// Execute a noisy circuit for multiple shots with explicit backend selection.
///
/// For Clifford circuits with Auto/Stabilizer/FactoredStabilizer backends,
/// uses the compiled noisy sampler (fast O(n²·m) compile + O(events·m/64) per shot).
/// For all other cases, falls back to per-shot simulation with noise injection.
/// The compiled noisy path is limited to terminal measurements with no resets
/// or classical conditionals.
pub(crate) fn run_shots_with_noise(
    kind: BackendKind,
    circuit: &Circuit,
    noise_model: &noise::NoiseModel,
    num_shots: usize,
    seed: u64,
) -> Result<ShotsResult> {
    // Trajectory execution runs shots on Rayon worker threads, whose
    // scheduling order differs per rank. Per-shot distributed backends would
    // issue collectives out of lockstep and deadlock or corrupt exchanges.
    // Reject until a lockstep noisy path exists.
    #[cfg(feature = "distributed")]
    if matches!(kind, BackendKind::StatevectorDistributed { .. }) {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "noisy shot sampling is not supported on the distributed backend; \
                     trajectory execution cannot keep rank collectives in lockstep"
                .into(),
        });
    }

    if !kind.supports_noisy_per_shot() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "this backend does not support noisy per-shot simulation".into(),
        });
    }

    let is_stabilizer_kind = kind.is_stabilizer_family();

    if is_stabilizer_kind && !noise_model.is_pauli_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "stabilizer backends only support Pauli/depolarizing noise; use {} for amplitude damping, phase damping, thermal relaxation, custom Kraus, or readout errors",
                BackendKind::general_noise_backend_names()
            ),
        });
    }

    if !noise_model.is_pauli_only() && !kind.supports_general_noise() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: format!(
                "non-Pauli noise requires {}",
                BackendKind::general_noise_backend_names()
            ),
        });
    }

    if is_stabilizer_kind && !circuit.is_clifford_only() {
        return Err(crate::error::PrismError::IncompatibleBackend {
            backend: format!("{kind:?}"),
            reason: "circuit contains non-Clifford gates".into(),
        });
    }

    if !kind.is_auto() {
        validate_explicit_backend(&kind, circuit)?;
    }

    if noise_model.is_pauli_only() {
        let use_compiled = (kind.is_auto()
            || matches!(
                kind,
                BackendKind::Stabilizer | BackendKind::FactoredStabilizer
            ))
            && supports_compiled_measurement_sampling(circuit)
            || {
                #[cfg(feature = "gpu")]
                {
                    matches!(kind, BackendKind::StabilizerGpu { .. })
                        && supports_compiled_measurement_sampling(circuit)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    false
                }
            };

        if use_compiled {
            #[cfg(feature = "gpu")]
            if let BackendKind::StabilizerGpu { context } = &kind {
                return noise::run_shots_noisy_with_gpu(
                    circuit,
                    noise_model,
                    num_shots,
                    seed,
                    context.clone(),
                );
            }
            return noise::run_shots_noisy(circuit, noise_model, num_shots, seed);
        }
    }

    let plan = if kind.is_auto() && !noise_model.is_pauli_only() {
        general_noise_plan(&kind, circuit)
    } else {
        resolve_backend(&kind, circuit, false)
    };
    trajectory::run_trajectories(
        |s| plan.build(s),
        circuit,
        noise_model,
        num_shots,
        seed,
        plan.is_gpu(),
    )
}

#[cfg(test)]
mod tests {
    use super::dispatch::min_clifford_prefix_gates;
    use super::*;
    use crate::backend::mps::MpsBackend;
    use crate::backend::product::ProductStateBackend;
    use crate::backend::sparse::SparseBackend;
    use crate::backend::stabilizer::StabilizerBackend;
    use crate::backend::statevector::StatevectorBackend;
    use crate::backend::tensornetwork::TensorNetworkBackend;
    use crate::circuit::smallvec;
    use crate::gates::Gate;

    fn make_clifford_circuit() -> Circuit {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::S, &[0]);
        c
    }

    fn make_product_circuit() -> Circuit {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Rx(1.0), &[1]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::Y, &[3]);
        c
    }

    fn make_general_circuit() -> Circuit {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c
    }

    #[derive(Debug, Clone, Copy)]
    enum ProbabilityFailure {
        Unsupported,
        Invalid,
    }

    struct ProbabilityFailureBackend {
        failure: ProbabilityFailure,
        classical_bits: Vec<bool>,
        num_qubits: usize,
    }

    impl ProbabilityFailureBackend {
        fn new(failure: ProbabilityFailure) -> Self {
            Self {
                failure,
                classical_bits: Vec::new(),
                num_qubits: 0,
            }
        }
    }

    impl Backend for ProbabilityFailureBackend {
        fn name(&self) -> &'static str {
            "probability_failure"
        }

        fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
            self.num_qubits = num_qubits;
            self.classical_bits = vec![false; num_classical_bits];
            Ok(())
        }

        fn apply(&mut self, _instruction: &Instruction) -> Result<()> {
            Ok(())
        }

        fn classical_results(&self) -> &[bool] {
            &self.classical_bits
        }

        fn probabilities(&self) -> Result<Vec<f64>> {
            match self.failure {
                ProbabilityFailure::Unsupported => Err(PrismError::BackendUnsupported {
                    backend: self.name().to_string(),
                    operation: "probabilities".to_string(),
                }),
                ProbabilityFailure::Invalid => Err(PrismError::InvalidParameter {
                    message: "probability extraction failed".to_string(),
                }),
            }
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
    }

    fn assert_pauli_marginals_reject(circuit: &Circuit) {
        for backend in [
            BackendKind::StochasticPauli { num_samples: 100 },
            BackendKind::DeterministicPauli {
                epsilon: 0.0,
                max_terms: 0,
            },
        ] {
            assert!(matches!(
                simulate(circuit)
                    .backend(backend)
                    .seed(42)
                    .marginals()
                    .unwrap_err(),
                PrismError::IncompatibleBackend { .. }
            ));
        }
    }

    #[test]
    fn test_circuit_is_clifford_only() {
        assert!(make_clifford_circuit().is_clifford_only());
        assert!(!make_general_circuit().is_clifford_only());
        assert!(!make_product_circuit().is_clifford_only());
    }

    #[test]
    fn test_circuit_has_entangling_gates() {
        assert!(make_clifford_circuit().has_entangling_gates());
        assert!(make_general_circuit().has_entangling_gates());
        assert!(!make_product_circuit().has_entangling_gates());
    }

    #[test]
    fn test_auto_selects_product() {
        let circuit = make_product_circuit();
        let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
        assert_eq!(backend.name(), "productstate");
    }

    #[test]
    fn test_auto_selects_stabilizer() {
        let circuit = make_clifford_circuit();
        let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
        assert_eq!(backend.name(), "stabilizer");
    }

    #[test]
    fn test_auto_selects_statevector() {
        let circuit = make_general_circuit();
        let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
        assert_eq!(backend.name(), "statevector");
    }

    #[test]
    fn test_run_with_auto_matches_explicit() {
        let circuit = make_general_circuit();
        let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_run_with_explicit_backends() {
        let circuit = make_clifford_circuit();
        assert!(run_with(BackendKind::Statevector, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Stabilizer, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Sparse, &circuit, 42).is_ok());
        assert!(run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).is_ok());
    }

    #[test]
    fn test_run_auto_clifford_probs_match_statevector() {
        let circuit = make_clifford_circuit();
        let auto_result = run(&circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &circuit).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, b) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_run_qasm() {
        let qasm = "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];";
        let result = run_qasm(qasm, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_empty_circuit_is_clifford_and_no_entangling() {
        let c = Circuit::new(2, 0);
        assert!(c.is_clifford_only());
        assert!(!c.has_entangling_gates());
    }

    #[test]
    fn test_validate_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit(); // has T gate
        let result = run_with(BackendKind::Stabilizer, &circuit, 42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_product_rejects_entangling() {
        let circuit = make_clifford_circuit(); // has CX
        let result = run_with(BackendKind::ProductState, &circuit, 42);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_passes_for_compatible() {
        let clifford = make_clifford_circuit();
        assert!(run_with(BackendKind::Stabilizer, &clifford, 42).is_ok());

        let product = make_product_circuit();
        assert!(run_with(BackendKind::ProductState, &product, 42).is_ok());
    }

    #[test]
    fn test_auto_moderate_qubit_count_uses_statevector() {
        let mut circuit = Circuit::new(20, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let backend = resolve_backend(&BackendKind::Auto, &circuit, false).build(42);
        assert_eq!(backend.name(), "statevector");
    }

    #[test]
    fn test_auto_selects_factored_with_partial_independence() {
        let mut circuit = Circuit::new(10, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let backend = resolve_backend(&BackendKind::Auto, &circuit, true).build(42);
        assert_eq!(backend.name(), "factored");
    }

    #[test]
    fn test_auto_ignores_partial_independence_when_no_entangling() {
        let circuit = make_product_circuit();
        let backend = resolve_backend(&BackendKind::Auto, &circuit, true).build(42);
        assert_eq!(backend.name(), "productstate");
    }

    #[test]
    fn test_classical_only_skips_probabilities() {
        let qasm =
            "OPENQASM 3.0;\nqubit[2] q;\nbit[1] c;\nh q[0];\ncx q[0], q[1];\nc[0] = measure q[0];";
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with_internal(
            BackendKind::Statevector,
            &circuit,
            42,
            SimOptions::classical_only(),
        )
        .unwrap();
        assert!(result.probabilities.is_none());
        assert_eq!(result.classical_bits.len(), 1);
    }

    #[test]
    fn test_default_options_include_probabilities() {
        let circuit = make_general_circuit();
        let result = run_with_internal(
            BackendKind::Statevector,
            &circuit,
            42,
            SimOptions::default(),
        )
        .unwrap();
        assert!(result.probabilities.is_some());
    }

    #[test]
    fn test_run_on_always_computes_probabilities() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        let result = run_on(&mut backend, &circuit).unwrap();
        assert!(result.probabilities.is_some());
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_clifford_matches_statevector() {
        // Circuit with a long Clifford prefix followed by non-Clifford gates.
        // At 10q with 20+ prefix gates, temporal decomposition should trigger.
        let mut c = Circuit::new(10, 0);

        // Clifford prefix: 22 gates (above min_clifford_prefix_gates(10)=20)
        for i in 0..10 {
            c.add_gate(Gate::H, &[i]);
        }
        for i in 0..9 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Sdg, &[3]);
        c.add_gate(Gate::SX, &[7]);

        // Non-Clifford tail
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Rx(0.7), &[1]);
        c.add_gate(Gate::Cx, &[2, 3]);
        c.add_gate(Gate::Rz(1.2), &[2]);

        let (prefix, _tail) = c.clifford_prefix_split().unwrap();
        assert!(prefix.gate_count() >= min_clifford_prefix_gates(c.num_qubits));

        let auto_result = run(&c, 42).unwrap();

        // Pure statevector reference (no temporal decomposition)
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        assert_eq!(auto_probs.len(), sv_probs.len());
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!(
                (a - s).abs() < 1e-10,
                "temporal decomp mismatch: auto={a}, sv={s}"
            );
        }
    }

    #[test]
    fn test_temporal_clifford_complex_circuit_matches_sv() {
        // 3q circuit: prefix too short for temporal (min_clifford_prefix_gates(3)=16)
        // but auto must still match statevector.
        let mut c = Circuit::new(3, 0);

        // Clifford prefix with i-phase generators
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Y, &[1]);
        c.add_gate(Gate::S, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::SXdg, &[2]);
        c.add_gate(Gate::Cz, &[1, 2]);
        c.add_gate(Gate::Swap, &[0, 2]);
        c.add_gate(Gate::S, &[1]);

        // Non-Clifford tail
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Ry(0.3), &[1]);
        c.add_gate(Gate::Cx, &[1, 2]);

        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!(
                (a - s).abs() < 1e-10,
                "complex temporal mismatch: auto={a}, sv={s}"
            );
        }
    }

    #[test]
    fn test_temporal_clifford_skipped_when_prefix_too_short() {
        // Short Clifford prefix, should NOT use temporal decomposition,
        // but result must still be correct.
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]); // 2 Clifford gates < min_clifford_prefix_gates(2)=16

        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let sv_result = run_on(&mut sv, &c).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (a, s) in auto_probs.iter().zip(sv_probs.iter()) {
            assert!((a - s).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decomposed_random_blocks_matches_monolithic() {
        let circuit = crate::circuits::independent_random_blocks(10, 2, 5, 0xDEAD_BEEF);
        let decomposed = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let monolithic = run_on(&mut sv, &circuit).unwrap();
        let d_probs = decomposed.probabilities.unwrap().to_vec();
        let m_probs = monolithic.probabilities.unwrap().to_vec();
        assert_eq!(d_probs.len(), m_probs.len());
        for (d, m) in d_probs.iter().zip(m_probs.iter()) {
            assert!(
                (d - m).abs() < 1e-10,
                "mismatch: decomposed={d}, monolithic={m}"
            );
        }
    }

    #[test]
    fn test_per_block_clifford_dispatch() {
        // 6-qubit circuit: qubits 0-2 = Clifford block, qubits 3-5 = non-Clifford block.
        // The two blocks are independent (no gates bridge them).
        // Under Auto dispatch with decomposition, block 0-2 should use Stabilizer
        // and block 3-5 should use Statevector. Correctness is checked by comparing
        // against a monolithic statevector run.
        let mut c = Circuit::new(6, 0);

        // Block A (Clifford): GHZ state on qubits 0,1,2
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::S, &[0]);

        // Block B (non-Clifford): qubits 3,4,5
        c.add_gate(Gate::H, &[3]);
        c.add_gate(Gate::T, &[3]);
        c.add_gate(Gate::Cx, &[3, 4]);
        c.add_gate(Gate::Rx(0.7), &[5]);
        c.add_gate(Gate::Cx, &[4, 5]);

        let components = c.independent_subsystems();
        assert_eq!(components.len(), 2);

        let (sub_a, _, _) = c.extract_subcircuit(&components[0]);
        assert!(sub_a.is_clifford_only());
        let backend_a = resolve_backend(&BackendKind::Auto, &sub_a, false).build(42);
        assert_eq!(backend_a.name(), "stabilizer");

        let (sub_b, _, _) = c.extract_subcircuit(&components[1]);
        assert!(!sub_b.is_clifford_only());
        let backend_b = resolve_backend(&BackendKind::Auto, &sub_b, false).build(43);
        assert_eq!(backend_b.name(), "statevector");

        // End-to-end: auto (decomposed) must match monolithic statevector
        let auto_result = run(&c, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let mono_result = run_on(&mut sv, &c).unwrap();
        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let mono_probs = mono_result.probabilities.unwrap().to_vec();
        assert_eq!(auto_probs.len(), mono_probs.len());
        for (a, m) in auto_probs.iter().zip(mono_probs.iter()) {
            assert!((a - m).abs() < 1e-10, "prob mismatch: auto={a}, mono={m}");
        }
    }

    #[test]
    fn test_decomposed_bell_pairs_matches_monolithic() {
        let circuit = crate::circuits::independent_bell_pairs(10);
        let decomposed = run(&circuit, 42).unwrap();
        let mut sv = StatevectorBackend::new(42);
        let monolithic = run_on(&mut sv, &circuit).unwrap();
        let d_probs = decomposed.probabilities.unwrap().to_vec();
        let m_probs = monolithic.probabilities.unwrap().to_vec();
        assert_eq!(d_probs.len(), m_probs.len());
        for (d, m) in d_probs.iter().zip(m_probs.iter()) {
            assert!(
                (d - m).abs() < 1e-10,
                "mismatch: decomposed={d}, monolithic={m}"
            );
        }
    }

    #[test]
    fn test_measurement_normalization_statevector() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "statevector post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_measurement_normalization_mps() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "MPS post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_measurement_normalization_sparse() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Sparse, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "sparse post-measurement probs sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_conditional_gate_execution() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[1] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        // q[0] measured as 1, so if-gate fires, q[1] flipped to |1⟩
        // Final state: |11⟩ = index 3
        let probs = result.probabilities.unwrap().to_vec();
        assert!(
            probs[3] > 0.99,
            "conditional gate should flip q[1]: probs={probs:?}"
        );
        assert!(result.classical_bits[0]);
    }

    fn make_bell_with_measure() -> Circuit {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        crate::circuit::openqasm::parse(qasm).unwrap()
    }

    #[test]
    fn test_shots_deterministic() {
        let circuit = make_bell_with_measure();
        let a = run_shots(&circuit, 10, 42).unwrap();
        let b = run_shots(&circuit, 10, 42).unwrap();
        assert_eq!(a.shots, b.shots);
    }

    #[test]
    fn test_shots_distribution_convergence() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 10000, 42).unwrap();
        let counts = result.counts();
        let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
        assert!(
            (4500..=5500).contains(&n_00),
            "|00> count {n_00} outside [4500, 5500]"
        );
        assert!(
            (4500..=5500).contains(&n_11),
            "|11> count {n_11} outside [4500, 5500]"
        );
        assert_eq!(n_01, 0, "|01> should never appear in Bell state");
        assert_eq!(n_10, 0, "|10> should never appear in Bell state");
    }

    #[test]
    fn test_shots_single_valid_outcome() {
        let circuit = make_bell_with_measure();
        let shots_result = run_shots(&circuit, 1, 42).unwrap();
        let shot = &shots_result.shots[0];
        assert_eq!(shot[0], shot[1], "Bell state: both bits must agree");
    }

    #[test]
    fn test_shots_all_zero() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[3] c;
            c[0] = measure q[0];
            c[1] = measure q[1];
            c[2] = measure q[2];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(
                shot.iter().all(|&b| !b),
                "shot {i} should be all-zero: {shot:?}"
            );
        }
    }

    #[test]
    fn test_shots_mid_circuit_measurement() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(shot[0], "shot {i}: q[0] should always be 1");
            assert!(shot[1], "shot {i}: q[1] should always be 1 (conditional)");
        }
    }

    #[test]
    fn test_shots_counts_sum() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 500, 42).unwrap();
        let counts = result.counts();
        let total: u64 = counts.values().sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn test_run_counts_factored_stabilizer() {
        let circuit = make_bell_with_measure();
        let counts = run_counts_with(BackendKind::FactoredStabilizer, &circuit, 128, 42).unwrap();
        let total: u64 = counts.values().sum();
        let bell_total = counts.get(&vec![0u64]).copied().unwrap_or(0)
            + counts.get(&vec![3u64]).copied().unwrap_or(0);

        assert_eq!(total, 128);
        assert_eq!(bell_total, 128);
    }

    fn assert_unit_norm(state: &[num_complex::Complex64], label: &str) {
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "{label}: norm = {norm}, expected 1.0"
        );
    }

    #[test]
    fn test_export_norm_statevector_bell() {
        let circuit = make_clifford_circuit();
        let mut backend = StatevectorBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/bell");
    }

    #[test]
    fn test_export_norm_statevector_parametric() {
        let circuit = crate::circuits::hardware_efficient_ansatz(6, 3, 42);
        let mut backend = StatevectorBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "statevector/hea_6q");
    }

    #[test]
    fn test_export_norm_stabilizer() {
        let circuit = make_clifford_circuit();
        let mut backend = StabilizerBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "stabilizer");
    }

    #[test]
    fn test_export_norm_sparse() {
        let circuit = make_general_circuit();
        let mut backend = SparseBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "sparse");
    }

    #[test]
    fn test_export_norm_mps() {
        let circuit = make_general_circuit();
        let mut backend = MpsBackend::new(64, 42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "mps");
    }

    #[test]
    fn test_export_norm_product_state() {
        let circuit = make_product_circuit();
        let mut backend = ProductStateBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "productstate");
    }

    #[test]
    fn test_export_norm_tensor_network() {
        let circuit = make_general_circuit();
        let mut backend = TensorNetworkBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert_unit_norm(&backend.export_statevector().unwrap(), "tensornetwork");
    }

    #[test]
    fn test_export_norm_after_measurement() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[1] c;
            h q[0];
            cx q[0], q[1];
            h q[2];
            c[0] = measure q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        for backend_kind in [
            BackendKind::Statevector,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
        ] {
            let label = format!("{backend_kind:?}/post-measure");
            let mut backend = resolve_backend(&backend_kind, &circuit, false).build(42);
            run_on(backend.as_mut(), &circuit).unwrap();
            let state = backend.export_statevector().unwrap();
            assert_unit_norm(&state, &label);
        }
    }

    #[test]
    fn test_export_norm_qft() {
        let circuit = crate::circuits::qft_circuit(8);
        for (kind, label) in [
            (BackendKind::Statevector, "statevector/qft8"),
            (BackendKind::Sparse, "sparse/qft8"),
            (BackendKind::Mps { max_bond_dim: 128 }, "mps/qft8"),
            (BackendKind::TensorNetwork, "tn/qft8"),
        ] {
            let mut backend = resolve_backend(&kind, &circuit, false).build(42);
            run_on(backend.as_mut(), &circuit).unwrap();
            let state = backend.export_statevector().unwrap();
            assert_unit_norm(&state, label);
        }
    }

    #[test]
    fn test_export_factored_unsupported() {
        let circuit = make_general_circuit();
        let mut backend = crate::backend::factored::FactoredBackend::new(42);
        run_on(&mut backend, &circuit).unwrap();
        assert!(backend.export_statevector().is_err());
    }

    #[test]
    fn test_shots_random_convergence() {
        let circuit = make_bell_with_measure();
        let result = run_shots(&circuit, 10000, rand::random()).unwrap();
        let counts = result.counts();
        let n_00 = counts.get(&vec![0u64]).copied().unwrap_or(0);
        let n_11 = counts.get(&vec![3u64]).copied().unwrap_or(0);
        let n_01 = counts.get(&vec![2u64]).copied().unwrap_or(0);
        let n_10 = counts.get(&vec![1u64]).copied().unwrap_or(0);
        // p=0.5, n=10000 → σ=50. Bounds at ±10σ: failure prob < 10^-23.
        assert!(
            (4500..=5500).contains(&n_00),
            "|00> count {n_00} outside [4500, 5500]"
        );
        assert!(
            (4500..=5500).contains(&n_11),
            "|11> count {n_11} outside [4500, 5500]"
        );
        assert_eq!(n_01, 0, "|01> should never appear in Bell state");
        assert_eq!(n_10, 0, "|10> should never appear in Bell state");
    }

    #[test]
    fn test_has_terminal_measurements_only() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        assert!(c.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[1] c;
            c[0] = measure q[0];
            h q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            x q[0];
            c[0] = measure q[0];
            if (c[0]) x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[1] c;
            h q[0];
            c[0] = measure q[0];
            x q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());
    }

    #[test]
    fn test_measurement_map() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            bit[3] c;
            c[2] = measure q[0];
            c[0] = measure q[2];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let map = circuit.measurement_map();
        assert_eq!(map, vec![(0, 2), (2, 0), (1, 1)]);
    }

    #[test]
    fn test_fast_path_deterministic_x() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[1] c;
            x q[0];
            c[0] = measure q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(circuit.has_terminal_measurements_only());
        let result = run_shots(&circuit, 100, 42).unwrap();
        for (i, shot) in result.shots.iter().enumerate() {
            assert!(shot[0], "shot {i}: X|0> should always measure 1");
        }
    }

    #[test]
    fn test_fast_path_preserves_classical_bit_index() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            bit[3] c;
            x q[0];
            c[2] = measure q[0];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        let result = run_shots(&circuit, 16, 42).unwrap();
        assert_eq!(result.num_classical_bits, 3);
        for shot in &result.shots {
            assert_eq!(shot, &vec![false, false, true]);
        }
    }

    #[test]
    fn test_terminal_statevector_sampling_matches_probability_path() {
        let mut c = Circuit::new(5, 5);
        for q in 0..5 {
            c.add_gate(Gate::Ry(0.17 + q as f64 * 0.11), &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Rz(0.41), &[3]);
        c.add_gate(Gate::Rx(0.23), &[4]);
        c.add_measure(3, 1);
        c.add_measure(0, 4);

        let stripped = c.without_measurements();
        let reference = run_with_internal(
            BackendKind::Statevector,
            &stripped,
            42,
            SimOptions::default(),
        )
        .unwrap();
        let probs = reference.probabilities.unwrap();
        let expected =
            shots::sample_shots(&probs, &c.measurement_map(), c.num_classical_bits, 256, 42);

        let actual = run_shots_with(BackendKind::Statevector, &c, 256, 42).unwrap();
        assert_eq!(actual.shots, expected);
    }

    #[test]
    fn test_terminal_statevector_counts_match_probability_path_all_measured() {
        let mut c = Circuit::new(4, 4);
        c.add_gate(Gate::Ry(0.31), &[0]);
        c.add_gate(Gate::Ry(0.47), &[1]);
        c.add_gate(Gate::Cx, &[0, 2]);
        c.add_gate(Gate::Rx(0.19), &[3]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        c.add_measure(2, 2);
        c.add_measure(3, 3);

        let stripped = c.without_measurements();
        let reference = run_with_internal(
            BackendKind::Statevector,
            &stripped,
            7,
            SimOptions::default(),
        )
        .unwrap();
        let probs = reference.probabilities.unwrap();
        let expected_shots =
            shots::sample_shots(&probs, &c.measurement_map(), c.num_classical_bits, 512, 7);
        let expected = ShotsResult {
            shots: expected_shots,
            num_classical_bits: c.num_classical_bits,
        }
        .counts();

        let actual = run_counts_with(BackendKind::Statevector, &c, 512, 7).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_terminal_statevector_duplicate_classical_bit_uses_last_measurement() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 0);
        c.add_measure(1, 0);

        let shots = run_shots_with(BackendKind::Statevector, &c, 16, 42).unwrap();
        for shot in &shots.shots {
            assert_eq!(shot, &vec![false]);
        }

        let counts = run_counts_with(BackendKind::Statevector, &c, 16, 42).unwrap();
        assert_eq!(counts.get(&vec![0]), Some(&16));
    }

    #[test]
    fn test_terminal_statevector_counts_wide_classical_register() {
        let mut c = Circuit::new(2, 72);
        c.add_gate(Gate::X, &[0]);
        c.add_measure(0, 70);

        let counts = run_counts_with(BackendKind::Statevector, &c, 10, 11).unwrap();
        let mut expected = vec![0u64; 2];
        expected[1] = 1u64 << 6;
        assert_eq!(counts.get(&expected), Some(&10));
    }

    #[test]
    fn test_terminal_statevector_subset_counts_sum_to_shots() {
        let mut c = Circuit::new(5, 5);
        for q in 0..5 {
            c.add_gate(Gate::Ry(0.21 + q as f64 * 0.07), &[q]);
        }
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[3, 4]);
        c.add_measure(1, 4);
        c.add_measure(4, 0);

        let counts = run_counts_with(BackendKind::Statevector, &c, 1024, 42).unwrap();
        assert_eq!(counts.values().sum::<u64>(), 1024);
        assert!(counts.keys().all(|key| key[0] & !0b1_0001 == 0));
    }

    #[test]
    fn test_fast_path_no_measurements() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        let result = run_shots(&c, 50, 42).unwrap();
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
            assert!(!shot[0] && !shot[1], "no measurements → all-false");
        }
    }

    #[test]
    fn test_shots_cached_fusion_matches_uncached() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            x q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());

        let cached = run_shots_with(BackendKind::Statevector, &circuit, 20, 42).unwrap();
        for i in 0..20 {
            let seed_i = 42u64.wrapping_add(i as u64);
            let single = run_with_internal(
                BackendKind::Statevector,
                &circuit,
                seed_i,
                SimOptions::default(),
            )
            .unwrap();
            assert_eq!(cached.shots[i], single.classical_bits, "shot {i} mismatch");
        }
    }

    #[test]
    fn test_shots_decomposed_cached() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[8] q;
            bit[8] c;
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            x q[1];
            c[1] = measure q[1];
            h q[4];
            cx q[4], q[5];
            c[4] = measure q[4];
            x q[5];
            c[5] = measure q[5];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();
        assert!(!circuit.has_terminal_measurements_only());
        let comps = circuit.independent_subsystems();
        assert!(comps.len() > 1, "circuit should decompose");

        let result = run_shots_with(BackendKind::Statevector, &circuit, 10, 42).unwrap();
        assert_eq!(result.shots.len(), 10);
        for shot in &result.shots {
            assert_eq!(shot.len(), 8);
        }
    }

    #[test]
    fn test_shots_temporal_clifford_fallback() {
        let mut c = Circuit::new(4, 4);
        for i in 0..4 {
            c.add_gate(Gate::H, &[i]);
        }
        for i in 0..3 {
            c.add_gate(Gate::Cx, &[i, i + 1]);
        }
        c.add_gate(Gate::T, &[0]);
        c.add_measure(0, 0);
        c.add_gate(Gate::X, &[1]);
        c.add_measure(1, 1);

        let result = run_shots_with(BackendKind::Auto, &c, 10, 42).unwrap();
        assert_eq!(result.shots.len(), 10);
        for shot in &result.shots {
            assert_eq!(shot.len(), 4);
        }
    }

    #[test]
    fn test_stabilizer_rank_dispatch() {
        let circuit = make_general_circuit();
        let result = run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert_eq!(probs.len(), 8);
        let total: f64 = probs.iter().sum();
        assert!((total - 1.0).abs() < 1e-10);

        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (sr, sv)) in probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr - sv).abs() < 1e-10,
                "prob[{i}]: stab_rank={sr}, statevector={sv}"
            );
        }
    }

    #[test]
    fn test_stabilizer_rank_rejects_no_t() {
        let circuit = make_clifford_circuit();
        let result = run_with(BackendKind::StabilizerRank, &circuit, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_clifford_plus_t_probabilities() {
        let circuit = make_general_circuit();
        assert!(circuit.is_clifford_plus_t());
        assert!(circuit.has_t_gates());

        let auto_result = run_with(BackendKind::Auto, &circuit, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &circuit, 42).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-10,
                "prob[{i}]: auto={a}, statevector={s}"
            );
        }
    }

    #[test]
    fn test_auto_clifford_plus_t_shots() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        let result = run_shots_with(BackendKind::Auto, &c, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
        }
    }

    #[test]
    fn test_decomposed_mixed_clifford_and_t() {
        // Two independent subsystems: q0-q1 (Clifford+T), q2-q3 (Clifford-only)
        // Under decomposition, q2-q3 should route to Stabilizer, q0-q1 to StabilizerRank
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[2, 3]);

        let subs = c.independent_subsystems();
        assert_eq!(subs.len(), 2);

        let auto_result = run_with(BackendKind::Auto, &c, 42).unwrap();
        let sv_result = run_with(BackendKind::Statevector, &c, 42).unwrap();

        let auto_probs = auto_result.probabilities.unwrap().to_vec();
        let sv_probs = sv_result.probabilities.unwrap().to_vec();
        for (i, (a, s)) in auto_probs.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (a - s).abs() < 1e-10,
                "prob[{i}]: auto={a}, statevector={s}"
            );
        }
    }

    #[test]
    fn test_run_shots_with_noise_clifford_uses_compiled() {
        let n = 10;
        let mut circuit = crate::circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        assert!(result.shots[0].len() == n);
    }

    #[test]
    fn test_run_shots_with_noise_statevector_brute() {
        let mut circuit = Circuit::new(3, 3);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        let result =
            run_shots_with_noise(BackendKind::Statevector, &circuit, &noise, 50, 42).unwrap();
        assert_eq!(result.shots.len(), 50);
        assert_eq!(result.shots[0].len(), 3);
    }

    #[test]
    fn test_run_shots_with_noise_auto_non_clifford() {
        let mut circuit = Circuit::new(3, 3);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_measure(0, 0);
        circuit.add_measure(1, 1);
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.001);
        let result = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_run_shots_with_stabilizer_gpu_falls_back_for_reset_circuits() {
        let mut circuit = Circuit::new(1, 1);
        circuit.add_gate(Gate::X, &[0]);
        circuit.add_reset(0);
        circuit.add_measure(0, 0);

        let cpu = run_shots_with(BackendKind::Stabilizer, &circuit, 8, 42).unwrap();
        let gpu = run_shots_with(
            BackendKind::StabilizerGpu {
                context: crate::gpu::GpuContext::stub_for_tests(),
            },
            &circuit,
            8,
            42,
        )
        .unwrap();

        assert_eq!(gpu.shots, cpu.shots);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_run_shots_with_stabilizer_gpu_falls_back_for_conditionals() {
        let mut circuit = Circuit::new(2, 2);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_measure(0, 0);
        circuit.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: crate::circuit::smallvec![1],
        });
        circuit.add_measure(1, 1);

        let cpu = run_shots_with(BackendKind::Stabilizer, &circuit, 256, 42).unwrap();
        let gpu = run_shots_with(
            BackendKind::StabilizerGpu {
                context: crate::gpu::GpuContext::stub_for_tests(),
            },
            &circuit,
            256,
            42,
        )
        .unwrap();

        assert_eq!(gpu.shots, cpu.shots);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_run_shots_with_noise_stabilizer_gpu_matches_stabilizer() {
        let n = 8;
        let mut circuit = crate::circuits::ghz_circuit(n);
        circuit.num_classical_bits = n;
        for i in 0..n {
            circuit.add_measure(i, i);
        }
        let noise = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);

        let cpu = run_shots_with_noise(BackendKind::Stabilizer, &circuit, &noise, 128, 42).unwrap();
        let gpu = run_shots_with_noise(
            BackendKind::StabilizerGpu {
                context: crate::gpu::GpuContext::stub_for_tests(),
            },
            &circuit,
            &noise,
            128,
            42,
        )
        .unwrap();

        assert_eq!(gpu.shots, cpu.shots);
    }

    #[test]
    fn test_run_marginals_bell_pair() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        let m = run_marginals(&c, 42).unwrap();
        assert_eq!(m.len(), 2);
        assert!((m[0].0 - 0.5).abs() < 1e-10);
        assert!((m[0].1 - 0.5).abs() < 1e-10);
        assert!((m[1].0 - 0.5).abs() < 1e-10);
        assert!((m[1].1 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_run_marginals_x_gate() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::X, &[0]);
        let m = run_marginals(&c, 42).unwrap();
        assert!((m[0].0 - 0.0).abs() < 1e-10);
        assert!((m[0].1 - 1.0).abs() < 1e-10);
        assert!((m[1].0 - 1.0).abs() < 1e-10);
        assert!((m[1].1 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_run_handles_backend_probability_failures() {
        let circuit = Circuit::new(1, 0);

        let mut unsupported = ProbabilityFailureBackend::new(ProbabilityFailure::Unsupported);
        let result = run_on(&mut unsupported, &circuit).unwrap();
        assert!(result.classical_bits.is_empty());
        assert!(result.probabilities.is_none());

        let mut invalid = ProbabilityFailureBackend::new(ProbabilityFailure::Invalid);
        let err = run_on(&mut invalid, &circuit).unwrap_err();

        assert!(matches!(err, PrismError::InvalidParameter { .. }));
    }

    #[test]
    fn test_run_marginals_rejects_missing_probability_output() {
        let circuit = Circuit::new(65, 0);
        let run_result = simulate(&circuit)
            .backend(BackendKind::FactoredStabilizer)
            .seed(42)
            .run()
            .unwrap();
        assert!(run_result.probabilities.is_none());

        let err = simulate(&circuit)
            .backend(BackendKind::FactoredStabilizer)
            .seed(42)
            .marginals()
            .unwrap_err();
        assert!(matches!(err, PrismError::BackendUnsupported { .. }));
    }

    #[test]
    fn test_run_marginals_clifford_t_spd_path() {
        let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
        let m_spd = run_marginals(&c, 42).unwrap();
        assert_eq!(m_spd.len(), 14);
        for (p0, p1) in &m_spd {
            assert!(*p0 >= 0.0 && *p0 <= 1.0);
            assert!((p0 + p1 - 1.0).abs() < 1e-10);
        }

        let m_sv = run_marginals_with(BackendKind::Statevector, &c, 42).unwrap();
        for i in 0..14 {
            assert!(
                (m_spd[i].0 - m_sv[i].0).abs() < 1e-6,
                "qubit {i}: SPD p0={} vs SV p0={}",
                m_spd[i].0,
                m_sv[i].0
            );
        }
    }

    // ── Dispatch validation ───────────────────────────────────────────

    #[test]
    fn test_simulate_builder_run_matches_run() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Ry(0.31), &[2]);

        let expected = run(&c, 42).unwrap();
        let actual = simulate(&c).seed(42).run().unwrap();

        assert_eq!(actual.classical_bits, expected.classical_bits);
        assert_eq!(
            actual.probabilities.unwrap().to_vec(),
            expected.probabilities.unwrap().to_vec()
        );
    }

    #[test]
    fn test_simulate_builder_sample_counts_matches_run_counts() {
        let mut c = Circuit::new(4, 4);
        c.add_gate(Gate::Ry(0.25), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rx(0.17), &[2]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);
        c.add_measure(2, 2);
        c.add_measure(3, 3);

        let expected = run_counts(&c, 256, 42).unwrap();
        let actual = simulate(&c).seed(42).sample_counts(256).unwrap();

        assert_eq!(actual.num_classical_bits, c.num_classical_bits);
        assert_eq!(actual.counts, expected);
    }

    #[test]
    fn test_simulate_builder_marginals_matches_run_marginals() {
        let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
        let expected = run_marginals(&c, 42).unwrap();
        let actual = simulate(&c).seed(42).marginals().unwrap();

        assert_eq!(actual.marginals.len(), expected.len());
        for (a, b) in actual.marginals.iter().zip(expected.iter()) {
            assert!((a.0 - b.0).abs() < 1e-12);
            assert!((a.1 - b.1).abs() < 1e-12);
        }
    }

    #[test]
    fn test_validate_factored_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit();
        assert!(matches!(
            run_with(BackendKind::FactoredStabilizer, &circuit, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_stabilizer_rank_rejects_no_t_gates() {
        let circuit = make_clifford_circuit();
        assert!(matches!(
            run_with(BackendKind::StabilizerRank, &circuit, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_validate_factored_stabilizer_accepts_clifford() {
        assert!(
            run_with(
                BackendKind::FactoredStabilizer,
                &make_clifford_circuit(),
                42
            )
            .is_ok()
        );
    }

    // ── Pauli backend error paths ───────────────────────────────────────

    #[test]
    fn test_pauli_backends_reject_mid_circuit_measurements() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            h q[0];
            c[0] = measure q[0];
            cx q[0], q[1];
            c[1] = measure q[1];
        "#;
        let circuit = crate::circuit::openqasm::parse(qasm).unwrap();

        assert!(matches!(
            run_shots_with(
                BackendKind::StochasticPauli { num_samples: 100 },
                &circuit,
                10,
                42
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            run_shots_with(
                BackendKind::DeterministicPauli {
                    epsilon: 1e-3,
                    max_terms: 1000
                },
                &circuit,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    // ── Noisy simulation error paths ────────────────────────────────────

    #[test]
    fn test_pauli_backends_reject_generic_run() {
        let c = crate::circuits::clifford_t_circuit(4, 2, 0.1, 42);

        assert!(matches!(
            simulate(&c)
                .backend(BackendKind::StochasticPauli { num_samples: 100 })
                .seed(42)
                .run()
                .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            simulate(&c)
                .backend(BackendKind::DeterministicPauli {
                    epsilon: 0.0,
                    max_terms: 0
                })
                .seed(42)
                .run()
                .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_pauli_backends_return_marginals_through_builder() {
        let c = crate::circuits::clifford_t_circuit(4, 2, 0.1, 42);

        let spp = simulate(&c)
            .backend(BackendKind::StochasticPauli { num_samples: 1_000 })
            .seed(42)
            .marginals()
            .unwrap();
        let spd = simulate(&c)
            .backend(BackendKind::DeterministicPauli {
                epsilon: 0.0,
                max_terms: 0,
            })
            .seed(42)
            .marginals()
            .unwrap();

        assert_eq!(spp.marginals.len(), c.num_qubits);
        assert_eq!(spd.marginals.len(), c.num_qubits);
        assert!(
            spp.marginals
                .iter()
                .chain(spd.marginals.iter())
                .all(|(p0, p1)| *p0 >= 0.0 && *p0 <= 1.0 && (p0 + p1 - 1.0).abs() < 1e-10)
        );
    }

    #[test]
    fn test_pauli_marginals_reject_non_clifford_t_gates() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(0.25), &[0]);

        assert_pauli_marginals_reject(&c);
    }

    #[test]
    fn test_pauli_marginals_reject_measurements_resets_and_conditionals() {
        let mut measured = Circuit::new(1, 1);
        measured.add_gate(Gate::H, &[0]);
        measured.add_gate(Gate::T, &[0]);
        measured.add_measure(0, 0);

        let mut reset = Circuit::new(1, 0);
        reset.add_gate(Gate::T, &[0]);
        reset.add_reset(0);

        let mut conditional = Circuit::new(2, 1);
        conditional.add_measure(0, 0);
        conditional.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::T,
            targets: smallvec![1],
        });

        for circuit in [&measured, &reset, &conditional] {
            assert_pauli_marginals_reject(circuit);
        }
    }

    #[test]
    fn test_noise_rejects_stabilizer_rank() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(matches!(
            run_shots_with_noise(BackendKind::StabilizerRank, &circuit, &nm, 10, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_rejects_pauli_backends() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);

        assert!(matches!(
            run_shots_with_noise(
                BackendKind::StochasticPauli { num_samples: 100 },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            run_shots_with_noise(
                BackendKind::DeterministicPauli {
                    epsilon: 1e-3,
                    max_terms: 1000
                },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_stabilizer_rejects_non_pauli_noise() {
        let circuit = make_clifford_circuit();
        let nm = noise::NoiseModel {
            after_gate: {
                let mut ag = vec![Vec::new(); circuit.instructions.len()];
                ag[0].push(noise::NoiseEvent {
                    channel: noise::NoiseChannel::AmplitudeDamping { gamma: 0.1 },
                    qubits: smallvec![0],
                });
                ag
            },
            readout: vec![None; circuit.num_qubits],
        };
        let err = run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err();
        match err {
            crate::error::PrismError::IncompatibleBackend { reason, .. } => {
                assert!(reason.contains("Statevector"));
                assert!(reason.contains("Sparse"));
                assert!(reason.contains("Factored"));
            }
            other => panic!("expected IncompatibleBackend, got {other:?}"),
        }
    }

    #[test]
    fn test_noise_auto_general_noise_avoids_stabilizer_dispatch() {
        let circuit = make_clifford_circuit();
        let nm = noise::NoiseModel::with_amplitude_damping(&circuit, 0.1);
        let result = run_shots_with_noise(BackendKind::Auto, &circuit, &nm, 16, 42);
        assert!(result.is_ok());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_noise_stabilizer_gpu_rejects_non_pauli_noise() {
        let circuit = make_clifford_circuit();
        let nm = noise::NoiseModel {
            after_gate: {
                let mut ag = vec![Vec::new(); circuit.instructions.len()];
                ag[0].push(noise::NoiseEvent {
                    channel: noise::NoiseChannel::AmplitudeDamping { gamma: 0.1 },
                    qubits: smallvec![0],
                });
                ag
            },
            readout: vec![None; circuit.num_qubits],
        };
        assert!(matches!(
            run_shots_with_noise(
                BackendKind::StabilizerGpu {
                    context: crate::gpu::GpuContext::stub_for_tests(),
                },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[test]
    fn test_noise_stabilizer_rejects_non_clifford() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(matches!(
            run_shots_with_noise(BackendKind::Stabilizer, &circuit, &nm, 10, 42).unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_noise_stabilizer_gpu_rejects_non_clifford() {
        let circuit = make_general_circuit();
        let nm = noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
        assert!(matches!(
            run_shots_with_noise(
                BackendKind::StabilizerGpu {
                    context: crate::gpu::GpuContext::stub_for_tests(),
                },
                &circuit,
                &nm,
                10,
                42,
            )
            .unwrap_err(),
            crate::error::PrismError::IncompatibleBackend { .. }
        ));
    }

    // ── Backend smoke tests ─────────────────────────────────────────────

    fn assert_probs_match(kind: BackendKind, circuit: &Circuit, expected: &[f64], tol: f64) {
        let label = format!("{kind:?}");
        let result = run_with(kind, circuit, 42).unwrap();
        let probs = result.probabilities.unwrap().to_vec();
        assert_eq!(probs.len(), expected.len(), "{label}: length mismatch");
        for (i, (a, b)) in probs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < tol,
                "{label}: prob[{i}] = {a}, expected {b}"
            );
        }
    }

    #[test]
    fn test_smoke_all_backends_clifford() {
        let circuit = make_clifford_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();

        for kind in [
            BackendKind::Stabilizer,
            BackendKind::FactoredStabilizer,
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
            BackendKind::TensorNetwork,
            BackendKind::Factored,
        ] {
            assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
        }
    }

    #[test]
    fn test_smoke_all_backends_general() {
        let circuit = make_general_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();

        for kind in [
            BackendKind::Sparse,
            BackendKind::Mps { max_bond_dim: 64 },
            BackendKind::TensorNetwork,
            BackendKind::Factored,
        ] {
            assert_probs_match(kind, &circuit, &sv_probs, 1e-8);
        }
    }

    #[test]
    fn test_smoke_product_state() {
        let circuit = make_product_circuit();
        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();
        assert_probs_match(BackendKind::ProductState, &circuit, &sv_probs, 1e-8);
    }

    #[test]
    fn test_smoke_stabilizer_rank() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::T, &[2]);

        let sv_probs = run_with(BackendKind::Statevector, &circuit, 42)
            .unwrap()
            .probabilities
            .unwrap()
            .to_vec();
        assert_probs_match(BackendKind::StabilizerRank, &circuit, &sv_probs, 1e-6);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod terminal_gpu_stub_tests {
    use super::*;
    use crate::gates::Gate;
    use crate::gpu::GpuContext;

    fn terminal_circuit(n: usize) -> Circuit {
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            c.add_measure(q, q);
        }
        c
    }

    fn auto_gpu_stub() -> BackendKind {
        BackendKind::AutoGpu {
            context: GpuContext::stub_for_tests(),
        }
    }

    /// The terminal fast path resolves the accel through the capability table.
    /// On the stub the VRAM gate fails closed, so `AutoGpu` must take the
    /// identical host path as `Auto`: same sampler, same RNG stream, byte-equal
    /// counts.
    #[test]
    fn auto_gpu_terminal_counts_match_auto_on_stub() {
        let circuit = terminal_circuit(16);
        let auto_counts = run_counts_with(BackendKind::Auto, &circuit, 500, 42).unwrap();
        let gpu_counts = run_counts_with(auto_gpu_stub(), &circuit, 500, 42).unwrap();
        assert_eq!(auto_counts, gpu_counts);
    }

    #[test]
    fn auto_gpu_terminal_shots_match_auto_on_stub() {
        let circuit = terminal_circuit(16);
        let auto_shots = run_shots_with(BackendKind::Auto, &circuit, 64, 42).unwrap();
        let gpu_shots = run_shots_with(auto_gpu_stub(), &circuit, 64, 42).unwrap();
        assert_eq!(auto_shots.shots, gpu_shots.shots);
    }

    /// Explicit `StatevectorGpu` is a terminal-fast-path candidate. Above the
    /// crossover it resolves hard, so the stub's failed allocation surfaces
    /// instead of falling back, proving the device path was reached.
    #[test]
    fn statevector_gpu_terminal_counts_hard_above_crossover_on_stub() {
        let circuit = terminal_circuit(16);
        let kind = BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        };
        let err = run_counts_with(kind, &circuit, 100, 42).unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::BackendUnsupported { .. }
        ));
    }

    /// Below the crossover the explicit GPU kind resolves to the host and must
    /// match explicit `Statevector` byte-exact through the terminal path.
    #[test]
    fn statevector_gpu_terminal_counts_below_crossover_match_statevector() {
        let circuit = terminal_circuit(6);
        let kind = BackendKind::StatevectorGpu {
            context: GpuContext::stub_for_tests(),
        };
        let sv = run_counts_with(BackendKind::Statevector, &circuit, 200, 42).unwrap();
        let gpu = run_counts_with(kind, &circuit, 200, 42).unwrap();
        assert_eq!(sv, gpu);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod expectation_gpu_stub_tests {
    use super::unified_pauli::{PauliAxis, PauliTerm};
    use super::*;
    use crate::gates::Gate;
    use crate::gpu::GpuContext;

    fn dense_circuit(n: usize) -> Circuit {
        let mut c = Circuit::new(n, 0);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        c
    }

    fn observables() -> Vec<Vec<PauliTerm>> {
        vec![
            vec![PauliTerm::new(0, PauliAxis::Z)],
            vec![
                PauliTerm::new(1, PauliAxis::X),
                PauliTerm::new(2, PauliAxis::Y),
            ],
        ]
    }

    #[test]
    fn auto_gpu_expectation_matches_auto_on_stub() {
        let circuit = dense_circuit(16);
        let auto_vals =
            run_expectation_values_with(BackendKind::Auto, &circuit, &observables(), 42).unwrap();
        let gpu_vals = run_expectation_values_with(
            BackendKind::AutoGpu {
                context: GpuContext::stub_for_tests(),
            },
            &circuit,
            &observables(),
            42,
        )
        .unwrap();
        assert_eq!(auto_vals, gpu_vals);
    }

    /// Explicit `StatevectorGpu` expectation values resolve hard above the
    /// crossover, so the stub's failed allocation surfaces.
    #[test]
    fn statevector_gpu_expectation_hard_above_crossover_on_stub() {
        let circuit = dense_circuit(16);
        let err = run_expectation_values_with(
            BackendKind::StatevectorGpu {
                context: GpuContext::stub_for_tests(),
            },
            &circuit,
            &observables(),
            42,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            crate::error::PrismError::BackendUnsupported { .. }
        ));
    }

    #[test]
    fn statevector_gpu_expectation_below_crossover_matches_statevector() {
        let circuit = dense_circuit(6);
        let sv =
            run_expectation_values_with(BackendKind::Statevector, &circuit, &observables(), 42)
                .unwrap();
        let gpu = run_expectation_values_with(
            BackendKind::StatevectorGpu {
                context: GpuContext::stub_for_tests(),
            },
            &circuit,
            &observables(),
            42,
        )
        .unwrap();
        assert_eq!(sv, gpu);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod noise_gpu_stub_tests {
    use super::*;
    use crate::gates::Gate;
    use crate::gpu::GpuContext;

    fn noisy_circuit(n: usize) -> Circuit {
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in 0..n - 1 {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..n {
            c.add_measure(q, q);
        }
        c
    }

    /// AutoGpu + non-Pauli noise resolves the trajectory plan through the
    /// capability table. On the stub the VRAM gate fails closed, so the run
    /// must be byte-identical to `Auto`.
    #[test]
    fn auto_gpu_general_noise_matches_auto_on_stub() {
        let circuit = noisy_circuit(14);
        let noise = noise::NoiseModel::with_amplitude_damping(&circuit, 0.05);
        let auto_shots = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 16, 42).unwrap();
        let gpu_shots = run_shots_with_noise(
            BackendKind::AutoGpu {
                context: GpuContext::stub_for_tests(),
            },
            &circuit,
            &noise,
            16,
            42,
        )
        .unwrap();
        assert_eq!(auto_shots.shots, gpu_shots.shots);
    }

    /// A non-entangling circuit resolves to the product-state family for
    /// general noise under AutoGpu, per the capability table (no GPU row).
    #[test]
    fn auto_gpu_general_noise_product_circuit_matches_auto() {
        let mut circuit = Circuit::new(6, 6);
        for q in 0..6 {
            circuit.add_gate(Gate::Rx(0.4), &[q]);
        }
        for q in 0..6 {
            circuit.add_measure(q, q);
        }
        let noise = noise::NoiseModel::with_amplitude_damping(&circuit, 0.05);
        let auto_shots = run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 64, 42).unwrap();
        let gpu_shots = run_shots_with_noise(
            BackendKind::AutoGpu {
                context: GpuContext::stub_for_tests(),
            },
            &circuit,
            &noise,
            64,
            42,
        )
        .unwrap();
        assert_eq!(auto_shots.shots, gpu_shots.shots);
    }
}

#[cfg(test)]
mod terminal_candidate_matrix_tests {
    use super::*;
    use crate::gates::Gate;

    fn rx_cx_chain(circuit: &mut Circuit, qubits: std::ops::Range<usize>) {
        for q in qubits.clone() {
            circuit.add_gate(Gate::Rx(0.3), &[q]);
        }
        for q in qubits.start..qubits.end - 1 {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    #[test]
    fn dense_entangled_circuit_is_candidate() {
        let mut circuit = Circuit::new(8, 0);
        rx_cx_chain(&mut circuit, 0..8);
        assert!(auto_terminal_statevector_candidate(&circuit));
    }

    #[test]
    fn decomposable_circuit_is_not_candidate() {
        let mut circuit = Circuit::new(12, 0);
        rx_cx_chain(&mut circuit, 0..6);
        rx_cx_chain(&mut circuit, 6..12);
        assert!(!auto_terminal_statevector_candidate(&circuit));
    }

    #[test]
    fn partial_independent_circuit_is_not_candidate() {
        let mut circuit = Circuit::new(10, 0);
        rx_cx_chain(&mut circuit, 0..8);
        rx_cx_chain(&mut circuit, 8..10);
        assert!(!auto_terminal_statevector_candidate(&circuit));
    }

    /// A Clifford+T circuit inside the stabilizer-rank budget routes to the
    /// rank engine; one T past the budget falls back to the statevector and
    /// becomes a candidate. Pins the shared budget helper at its boundary.
    #[test]
    fn clifford_t_budget_boundary_flips_candidacy() {
        let n = 10;
        let budget = stabilizer_rank_budget(n);
        assert_eq!(budget, 2);

        let mut within = Circuit::new(n, 0);
        within.add_gate(Gate::H, &[0]);
        for q in 0..n - 1 {
            within.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..budget {
            within.add_gate(Gate::T, &[q]);
        }
        assert!(!auto_terminal_statevector_candidate(&within));

        let mut beyond = Circuit::new(n, 0);
        beyond.add_gate(Gate::H, &[0]);
        for q in 0..n - 1 {
            beyond.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..budget + 1 {
            beyond.add_gate(Gate::T, &[q]);
        }
        assert!(auto_terminal_statevector_candidate(&beyond));
    }

    #[test]
    fn temporal_prefix_circuit_is_not_candidate() {
        let n = 8;
        let mut circuit = Circuit::new(n, 0);
        circuit.add_gate(Gate::H, &[0]);
        for _ in 0..3 {
            for q in 0..n - 1 {
                circuit.add_gate(Gate::Cx, &[q, q + 1]);
            }
        }
        for q in 0..n {
            circuit.add_gate(Gate::Rx(0.3), &[q]);
        }
        assert!(has_temporal_clifford_opportunity(
            &BackendKind::Auto,
            &circuit
        ));
        assert!(!auto_terminal_statevector_candidate(&circuit));
    }
}
