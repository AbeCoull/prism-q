//! T-gate strategy dispatch for native QEC programs.
//!
//! Public surface: one strategy enum and a single entry point
//! [`run_qec_program_with_strategy`]. [`QecTStrategy::Auto`] is the
//! production dispatcher: exact light-cone SPD, CAMPS, then the private exact
//! tensor-network scalar fallback. [`QecTStrategy::Reference`] is the
//! analytical correctness anchor. [`QecTStrategy::Spd`] and
//! [`QecTStrategy::Camps`] are direct analytical paths.

use super::observable_reroute::{min_cone_z_representative, xor_z_support};
#[cfg(test)]
use super::{QecBasis, QecOptions, QecRecordRef};
use super::{QecObservableEstimate, QecOp, QecProgram, QecSampleResult, run_qec_program_reference};
use crate::backend::Backend;
use crate::backend::mps::MpsBackend;
use crate::circuit::{Circuit, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::PackedShots;
use crate::sim::unified_pauli::{PauliTerm, run_spd_observable_light_cone};

/// Strategy used to sample a QEC program that may contain T gates.
///
/// Production path is [`QecTStrategy::Auto`], which tries exact light-cone SPD,
/// then CAMPS, then the private exact tensor-network scalar fallback when CAMPS
/// cannot run. [`QecTStrategy::Reference`] is the analytical correctness
/// anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QecTStrategy {
    /// Production dispatcher. Exact light-cone SPD, CAMPS, then tensor network.
    Auto,
    /// One state-vector simulation per shot. Correctness oracle.
    Reference,
    /// Clifford-augmented matrix product state Pauli expectation path.
    Camps,
    /// Sparse Pauli decomposition after the joint-observable upgrade.
    Spd,
}

#[derive(Debug, Clone)]
pub struct QecObservableReroute {
    pub observable: usize,
    pub stabilizers: Vec<Vec<usize>>,
}

impl QecTStrategy {
    /// Short human label used in benchmark groups and decision tables.
    pub fn label(self) -> &'static str {
        match self {
            QecTStrategy::Auto => "auto",
            QecTStrategy::Reference => "reference",
            QecTStrategy::Camps => "camps",
            QecTStrategy::Spd => "spd",
        }
    }
}

/// Run a QEC program under the chosen T-gate sampling strategy.
///
/// Use [`QecTStrategy::Auto`] for production observable runs. The
/// [`QecTStrategy::Reference`] strategy is always available and matches
/// [`run_qec_program_reference`] exactly.
pub fn run_qec_program_with_strategy(
    program: &QecProgram,
    strategy: QecTStrategy,
) -> Result<QecSampleResult> {
    match strategy {
        QecTStrategy::Auto => run_qec_program_auto(program),
        QecTStrategy::Reference => run_qec_program_reference(program),
        QecTStrategy::Spd => run_qec_program_spd(program),
        QecTStrategy::Camps => run_qec_program_camps(program),
    }
}

#[cfg(test)]
mod auto_test_hooks {
    use std::cell::Cell;

    thread_local! {
        static FORCE_SPD_NONEXACT: Cell<bool> = const { Cell::new(false) };
        static FORCE_CAMPS_FAILURE: Cell<bool> = const { Cell::new(false) };
    }

    pub(super) fn force_spd_nonexact() -> bool {
        FORCE_SPD_NONEXACT.with(Cell::get)
    }

    pub(super) fn force_camps_failure() -> bool {
        FORCE_CAMPS_FAILURE.with(Cell::get)
    }

    pub(super) fn set_force_spd_nonexact(value: bool) {
        FORCE_SPD_NONEXACT.with(|flag| flag.set(value));
    }

    pub(super) fn set_force_camps_failure(value: bool) {
        FORCE_CAMPS_FAILURE.with(|flag| flag.set(value));
    }
}

#[cfg(test)]
fn run_qec_program_spd_for_auto(program: &QecProgram) -> Result<QecSampleResult> {
    let mut result = run_qec_program_spd(program)?;
    if auto_test_hooks::force_spd_nonexact() {
        if let Some(estimates) = result.observable_expectations.as_mut() {
            for estimate in estimates {
                estimate.variance = estimate.variance.max(1.0);
            }
        }
    }
    Ok(result)
}

#[cfg(not(test))]
fn run_qec_program_spd_for_auto(program: &QecProgram) -> Result<QecSampleResult> {
    run_qec_program_spd(program)
}

#[cfg(test)]
fn run_qec_program_camps_for_auto(program: &QecProgram) -> Result<QecSampleResult> {
    if auto_test_hooks::force_camps_failure() {
        return Err(PrismError::BackendUnsupported {
            backend: "QEC CAMPS".to_string(),
            operation: "forced CAMPS failure for auto-dispatch test".to_string(),
        });
    }
    run_qec_program_camps(program)
}

#[cfg(not(test))]
fn run_qec_program_camps_for_auto(program: &QecProgram) -> Result<QecSampleResult> {
    run_qec_program_camps(program)
}

fn run_qec_program_auto(program: &QecProgram) -> Result<QecSampleResult> {
    match run_qec_program_spd_for_auto(program) {
        Ok(result) if analytical_result_has_no_truncation(&result) => Ok(result),
        Ok(_) => run_qec_program_auto_after_camps(
            program,
            "light-cone SPD exceeded its exact truncation budget".to_string(),
        ),
        Err(spd_error) => run_qec_program_auto_after_camps(
            program,
            format!("light-cone SPD failed ({spd_error})"),
        ),
    }
}

fn run_qec_program_auto_after_camps(
    program: &QecProgram,
    spd_context: String,
) -> Result<QecSampleResult> {
    match run_qec_program_camps_for_auto(program) {
        Ok(result) => Ok(result),
        Err(camps_error) => {
            run_qec_program_tensor_network_observable(program).map_err(|tn_error| {
                PrismError::IncompatibleBackend {
                    backend: "QEC auto T-strategy".to_string(),
                    reason: format!(
                        "{spd_context} and CAMPS fallback failed ({camps_error}); \
                         tensor-network scalar fallback failed: {tn_error}"
                    ),
                }
            })
        }
    }
}

/// Discarded-mass variance at or below this is treated as numerically exact.
/// SPD reports `variance = total_discarded²`, and a handful of legitimate
/// sub-`QEC_SPD_EPSILON` terms can leave a tiny nonzero residue that must not
/// force the costlier CAMPS / tensor-network fallback. A discarded amplitude of
/// `1e-6` (variance `1e-12`) is far below the statistical noise of any shot
/// count the QEC paths target.
const QEC_SPD_NEGLIGIBLE_VARIANCE: f64 = 1e-12;

fn analytical_result_has_no_truncation(result: &QecSampleResult) -> bool {
    result
        .observable_expectations
        .as_ref()
        .map(|estimates| {
            estimates
                .iter()
                .all(|estimate| estimate.variance <= QEC_SPD_NEGLIGIBLE_VARIANCE)
        })
        .unwrap_or(true)
}

/// Default truncation tolerance for SPD on QEC programs.
const QEC_SPD_EPSILON: f64 = 1e-10;
/// Default term-count cap for SPD on QEC programs. Sub-percent error is
/// expected for `t ≤ 20`; larger T-counts may exceed this and report a
/// non-zero `total_discarded`.
const QEC_SPD_MAX_TERMS: usize = 16_384;

/// Lower a QEC program into a unitary Circuit plus per-record qubit map
/// for the joint-Pauli SPD / CAMPS analytical paths.
///
/// Delegates to the deferred-lowering machinery in `qec/noise.rs`
/// (`lower_qec_program_to_deferred_circuit_allowing_non_clifford`),
/// which handles Reset (fresh qubit alias), MPP (scratch qubit + CNOT
/// chain), basis-rotated measurements (`H` / `S†H` prefix), and the
/// trailing measurement record bookkeeping. The trailing
/// `Instruction::Measure` ops appended by the deferred lowering are
/// stripped from the returned Circuit so the strategies can evaluate
/// `⟨0^n| U† P U |0^n⟩` against the pure unitary.
///
/// **Still rejected:**
/// - Active `Noise` channels: SPD / CAMPS do not have an
///   `apply_noise_to_measurements`-style hook.
/// - `ExpectationValue`: unmodified rejection.
/// - `Detector`: analytical strategies do not emit detector rows yet.
fn lower_qec_program_for_pauli_observable(program: &QecProgram) -> Result<(Circuit, Vec<usize>)> {
    for op in program.ops() {
        match op {
            QecOp::Noise { channel, .. } if channel.probability() > 0.0 => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC SPD / CAMPS".to_string(),
                    reason: format!(
                        "analytical strategies evaluate pure-state expectations and \
                         cannot absorb Pauli noise channels (got `{channel:?}`); \
                         route noisy QEC programs through `run_qec_program_reference` \
                         (statevector per shot with stochastic noise) or extend \
                         CAMPS to an `MPDO` density-matrix path"
                    ),
                });
            }
            QecOp::ExpectationValue { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC SPD / CAMPS".to_string(),
                    reason: "EXP_VAL op is reserved for the dedicated \
                             expectation runner"
                        .to_string(),
                });
            }
            QecOp::Detector { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "QEC SPD / CAMPS".to_string(),
                    reason: "analytical strategies do not emit detector records yet".to_string(),
                });
            }
            _ => {}
        }
    }

    let deferred =
        super::noise::lower_qec_program_to_deferred_circuit_allowing_non_clifford(program)?;
    let mut circuit = Circuit::new(
        deferred.circuit.num_qubits,
        deferred.circuit.num_classical_bits,
    );
    for inst in &deferred.circuit.instructions {
        if matches!(inst, crate::circuit::Instruction::Gate { .. }) {
            circuit.instructions.push(inst.clone());
        }
    }
    Ok((circuit, deferred.measurement_qubits))
}

/// Convert an observable-row record list into a joint Pauli observable.
/// Each record contributes a single Z factor on the measured qubit; an
/// even number of factors on the same qubit cancels (Z² = I).
fn observable_row_to_pauli_terms(
    row: &[usize],
    record_to_qubit: &[usize],
) -> Result<Vec<PauliTerm>> {
    let mut parity: SmallVec<[(usize, bool); 8]> = SmallVec::new();
    for &record in row {
        let qubit = *record_to_qubit
            .get(record)
            .ok_or_else(|| PrismError::InvalidParameter {
                message: format!(
                    "observable references record {record} but only {} records exist",
                    record_to_qubit.len()
                ),
            })?;
        if let Some(entry) = parity.iter_mut().find(|(q, _)| *q == qubit) {
            entry.1 = !entry.1;
        } else {
            parity.push((qubit, true));
        }
    }
    let mut terms: Vec<PauliTerm> = parity
        .into_iter()
        .filter(|(_, odd)| *odd)
        .map(|(qubit, _)| PauliTerm::z(qubit))
        .collect();
    terms.sort_by_key(|t| t.qubit);
    Ok(terms)
}

/// Max number of postselection rows allowed for the analytical
/// conditional-expectation path. Each additional row doubles the
/// number of Pauli evaluations; 12 caps the inner sum at 4,096 terms
/// per observable.
const MAX_POSTSEL_FOR_CONDITIONAL_EXPECTATION: usize = 12;

/// Resolve postselection rows into (qubit-set, sign) pairs ready for
/// the conditional-expectation expansion. Returns
/// `(Vec<(Vec<usize>, bool)>, Vec<f64>)`: per-row qubit list and per-row
/// `ε_i ∈ {+1, -1}` based on the `expected` outcome bit.
fn lower_postselection_rows(
    rows: &[(Vec<usize>, bool)],
    record_to_qubit: &[usize],
) -> Result<Vec<(Vec<usize>, f64)>> {
    if rows.len() > MAX_POSTSEL_FOR_CONDITIONAL_EXPECTATION {
        return Err(PrismError::BackendUnsupported {
            backend: "QEC analytical conditional expectation".to_string(),
            operation: format!(
                "{} postselection rows exceeds the cap ({}); use the \
                 reference runner for large-postsel circuits",
                rows.len(),
                MAX_POSTSEL_FOR_CONDITIONAL_EXPECTATION
            ),
        });
    }
    let mut out = Vec::with_capacity(rows.len());
    for (records, expected) in rows {
        let mut qubits = Vec::with_capacity(records.len());
        for &record in records {
            let qubit =
                *record_to_qubit
                    .get(record)
                    .ok_or_else(|| PrismError::InvalidParameter {
                        message: format!(
                            "postselection references record {record} but only {} records exist",
                            record_to_qubit.len()
                        ),
                    })?;
            qubits.push(qubit);
        }
        let sign = if *expected { -1.0 } else { 1.0 };
        out.push((qubits, sign));
    }
    Ok(out)
}

/// XOR-merge a sequence of qubit lists into a sorted, deduplicated
/// `Vec<PauliTerm>` of `Z` factors (each qubit appears with parity 1).
fn merge_qubit_groups_into_z_terms<'a, I>(groups: I) -> Vec<PauliTerm>
where
    I: IntoIterator<Item = &'a [usize]>,
{
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for group in groups {
        for &q in group {
            *counts.entry(q).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, c)| c % 2 == 1)
        .map(|(q, _)| PauliTerm::z(q))
        .collect()
}

fn validate_z_stabilizers(stabilizers: &[Vec<usize>], num_qubits: usize) -> Result<()> {
    for stabilizer in stabilizers {
        for &qubit in stabilizer {
            if qubit >= num_qubits {
                return Err(PrismError::InvalidQubit {
                    index: qubit,
                    register_size: num_qubits,
                });
            }
        }
    }
    Ok(())
}

/// Per-Pauli-string evaluation outcome. `discarded_sq` carries the
/// squared truncation weight (zero for exact strategies like CAMPS and
/// tensor-network); the analytical SPD strategy reports its light-cone
/// truncation error here so it can be attributed to the right observable.
struct ConditionalEval {
    mean: f64,
    discarded_sq: f64,
}

/// Generic conditional-expectation evaluator for the analytical
/// strategies. Implements `⟨O⟩_post = ⟨O·Π⟩ / ⟨Π⟩` where
/// `Π = ∏_j (I + ε_j Z(S_j))/2`. Expands the product into a sum over
/// `2^J` subsets `T ⊆ {1..J}` and evaluates each Pauli-string
/// expectation via the supplied closure. Returns
/// `(post_means, per_observable_discarded_sq, accept_rate)`, where the
/// discarded-weight vector is aligned with `observable_terms_list` so each
/// observable's truncation error is tracked independently.
fn evaluate_conditional_expectations<F>(
    observable_terms_list: &[Vec<PauliTerm>],
    postsel: &[(Vec<usize>, f64)],
    mut eval: F,
) -> Result<(Vec<f64>, Vec<f64>, f64)>
where
    F: FnMut(&[PauliTerm]) -> Result<ConditionalEval>,
{
    let j = postsel.len();
    let subset_count = 1usize << j;
    let scale = 1.0 / (1usize << j) as f64;

    let mut pi_sum = 0.0;
    let mut obs_pi_sums = vec![0.0; observable_terms_list.len()];
    let mut obs_discarded = vec![0.0; observable_terms_list.len()];

    let obs_qubits: Vec<Vec<usize>> = observable_terms_list
        .iter()
        .map(|terms| terms.iter().map(|t| t.qubit).collect())
        .collect();

    for mask in 0..subset_count {
        let mut sign = 1.0;
        let mut qubit_groups: Vec<&[usize]> = Vec::with_capacity(j + 1);
        for (j_idx, (qubits, eps)) in postsel.iter().enumerate() {
            if (mask >> j_idx) & 1 == 1 {
                sign *= *eps;
                qubit_groups.push(qubits.as_slice());
            }
        }
        let pi_pauli = merge_qubit_groups_into_z_terms(qubit_groups.iter().copied());
        let pi_value = eval(&pi_pauli)?;
        pi_sum += sign * pi_value.mean;

        for (obs_idx, obs_q) in obs_qubits.iter().enumerate() {
            qubit_groups.push(obs_q.as_slice());
            let combined_pauli = merge_qubit_groups_into_z_terms(qubit_groups.iter().copied());
            qubit_groups.pop();
            let outcome = eval(&combined_pauli)?;
            obs_pi_sums[obs_idx] += sign * outcome.mean;
            obs_discarded[obs_idx] += outcome.discarded_sq;
        }
    }

    let accept_rate = (pi_sum * scale).clamp(0.0, 1.0);
    if accept_rate <= 1e-12 {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "QEC postselection has acceptance rate {accept_rate:.2e}; cannot \
                 condition an expectation on a zero-measure event"
            ),
        });
    }
    let post_means: Vec<f64> = obs_pi_sums
        .iter()
        .map(|s| (s * scale) / accept_rate)
        .collect();
    Ok((post_means, obs_discarded, accept_rate))
}

fn build_qec_result_from_observable_means(
    program: &QecProgram,
    means: Vec<f64>,
    estimates: Vec<QecObservableEstimate>,
) -> Result<QecSampleResult> {
    build_qec_result_with_acceptance(program, means, estimates, 1.0)
}

fn accepted_shots_for_rate(total_shots: usize, accept_rate: f64) -> usize {
    (((total_shots as f64) * accept_rate).round() as usize).min(total_shots)
}

fn build_qec_result_with_acceptance(
    program: &QecProgram,
    means: Vec<f64>,
    estimates: Vec<QecObservableEstimate>,
    accept_rate: f64,
) -> Result<QecSampleResult> {
    if program.num_detectors() > 0 {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC analytical result".to_string(),
            reason: "analytical QEC T strategies do not emit detector records yet".to_string(),
        });
    }

    let total_shots = program.options().shots;
    let accepted_shots = accepted_shots_for_rate(total_shots, accept_rate);
    let discarded_shots = total_shots - accepted_shots;
    let num_observables = program.num_observables();
    let num_detectors = program.num_detectors();
    let num_measurements = program.num_measurements();
    let measurements = PackedShots::from_meas_major(Vec::new(), 0, num_measurements);
    // Detector records are empty because analytical strategies reject
    // detectors. Observable records are synthesized so their popcount
    // marginals equal `logical_errors`, but the analytical path has no
    // per-shot accept mask: the `count` one-bits occupy positions
    // `[0, accepted_shots)` and the rest is inert padding up to
    // `total_shots`. Consumers must therefore divide by `accepted_shots`
    // (e.g. via [`QecSampleResult::logical_error_rates`]), not
    // `total_shots`, and must not align observable rows shot-for-shot with
    // detector rows. Exact expectations live in `observable_expectations`.
    let detector_words = total_shots.div_ceil(64) * num_detectors;
    let detectors =
        PackedShots::from_meas_major(vec![0u64; detector_words], total_shots, num_detectors);

    let mut logical_errors = Vec::with_capacity(num_observables);
    for &mean in &means {
        let rate = ((1.0 - mean) * 0.5).clamp(0.0, 1.0);
        let count = if accepted_shots == 0 {
            0
        } else {
            (rate * accepted_shots as f64).round() as u64
        };
        logical_errors.push(count);
    }
    let observables = packed_observable_records_from_logical_errors(total_shots, &logical_errors);

    let result = QecSampleResult::new_with_total_shots(
        total_shots,
        measurements,
        detectors,
        observables,
        accepted_shots,
        discarded_shots,
        logical_errors,
    )?;
    result.with_observable_expectations(estimates)
}

fn packed_observable_records_from_logical_errors(
    total_shots: usize,
    logical_errors: &[u64],
) -> PackedShots {
    let num_observables = logical_errors.len();
    let s_words = total_shots.div_ceil(64);
    let mut data = vec![0u64; num_observables * s_words];
    for (observable, &count) in logical_errors.iter().enumerate() {
        let mut remaining = (count as usize).min(total_shots);
        for word in 0..s_words {
            let bits = remaining.min(64);
            data[observable * s_words + word] = match bits {
                0 => 0,
                64 => u64::MAX,
                n => (1u64 << n) - 1,
            };
            remaining -= bits;
            if remaining == 0 {
                break;
            }
        }
    }
    PackedShots::from_meas_major(data, total_shots, num_observables)
}

/// Default MPS bond-dimension cap for CAMPS. Matches the
/// auto-dispatch ceiling used elsewhere in PRISM-Q (`MPS(256)` in
/// `sim/dispatch.rs`).
const QEC_CAMPS_MAX_BOND_DIM: usize = 256;

/// CAMPS analytical path for QEC observables.
///
/// Tracks the Clifford prefix separately from the MPS, applies T/Tdg through
/// the CAMPS disentangler update, and evaluates Z-string observables directly
/// from the MPS plus prefix. The path shares SPD's restricted lowering and
/// Z-only observable scope.
fn run_qec_program_camps(program: &QecProgram) -> Result<QecSampleResult> {
    use super::camps_prefix::{
        SignedCliffordPrefix, apply_t_via_camps, evaluate_z_observable_camps,
    };
    use crate::circuit::Instruction;

    let (circuit, record_to_qubit) = lower_qec_program_for_pauli_observable(program)?;
    let observable_rows = program.observable_rows()?;
    let postsel_rows = program.postselection_rows()?;

    let mut backend = MpsBackend::new(program.options().seed, QEC_CAMPS_MAX_BOND_DIM);
    backend.init(circuit.num_qubits, 0)?;

    let mut prefix = SignedCliffordPrefix::identity(circuit.num_qubits);
    for inst in &circuit.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            continue;
        };
        match gate {
            Gate::T => {
                apply_t_via_camps(&mut prefix, &mut backend, targets[0], false, 1e-10)?;
            }
            Gate::Tdg => {
                apply_t_via_camps(&mut prefix, &mut backend, targets[0], true, 1e-10)?;
            }
            _ => {
                prefix.apply_state_gate(gate, targets).map_err(|_| {
                    PrismError::InvalidParameter {
                        message: format!(
                            "CAMPS dispatcher: gate {gate:?} is neither Clifford nor T/Tdg; \
                             extend `lower_qec_program_for_pauli_observable` or add a CAMPS \
                             fallback to handle this instruction"
                        ),
                    }
                })?;
            }
        }
    }

    let mut observable_terms_list = Vec::with_capacity(observable_rows.len());
    for row in observable_rows.iter() {
        observable_terms_list.push(observable_row_to_pauli_terms(row, &record_to_qubit)?);
    }

    // Clifford gates are absorbed into a [`SignedCliffordPrefix`]; T/Tdg
    // gates dispatch through [`apply_t_via_camps`] (Liu & Clark
    // Algorithm 1 OFD). The observable `⟨ψ|Π Z_q|ψ⟩` is evaluated as
    // `⟨ϕ| C†(Π Z_q) C |ϕ⟩` by composing twisted Pauli rows and calling
    // [`crate::backend::mps::MpsBackend::pauli_expectation`].
    let eval = |terms: &[PauliTerm]| -> Result<f64> {
        let qubits: Vec<usize> = terms.iter().map(|t| t.qubit).collect();
        evaluate_z_observable_camps(&prefix, &backend, &qubits)
    };

    if postsel_rows.is_empty() {
        let mut means = Vec::with_capacity(observable_terms_list.len());
        let mut estimates = Vec::with_capacity(observable_terms_list.len());
        for terms in &observable_terms_list {
            let mean = eval(terms)?;
            means.push(mean);
            estimates.push(QecObservableEstimate {
                mean,
                variance: 0.0,
                num_shots: program.options().shots,
            });
        }
        return build_qec_result_from_observable_means(program, means, estimates);
    }

    let postsel = lower_postselection_rows(&postsel_rows, &record_to_qubit)?;
    let (means, _obs_discarded, accept_rate) =
        evaluate_conditional_expectations(&observable_terms_list, &postsel, |terms| {
            Ok(ConditionalEval {
                mean: eval(terms)?,
                discarded_sq: 0.0,
            })
        })?;
    let estimate_shots = accepted_shots_for_rate(program.options().shots, accept_rate);
    let estimates: Vec<_> = means
        .iter()
        .map(|&m| QecObservableEstimate {
            mean: m,
            variance: 0.0,
            num_shots: estimate_shots,
        })
        .collect();
    build_qec_result_with_acceptance(program, means, estimates, accept_rate)
}

// Private exact fallback for Auto. Contracts each scalar observable directly
// as <0| U^dagger P U |0> and does not materialize a dense statevector.
fn run_qec_program_tensor_network_observable(program: &QecProgram) -> Result<QecSampleResult> {
    let (circuit, record_to_qubit) = lower_qec_program_for_pauli_observable(program)?;
    let observable_rows = program.observable_rows()?;
    let postsel_rows = program.postselection_rows()?;

    let mut observable_terms_list = Vec::with_capacity(observable_rows.len());
    for row in observable_rows.iter() {
        observable_terms_list.push(observable_row_to_pauli_terms(row, &record_to_qubit)?);
    }

    let eval = |terms: &[PauliTerm]| -> Result<f64> {
        crate::backend::tensornetwork::expectation_zero_state(&circuit, terms)
    };

    if postsel_rows.is_empty() {
        let mut means = Vec::with_capacity(observable_terms_list.len());
        let mut estimates = Vec::with_capacity(observable_terms_list.len());
        for terms in &observable_terms_list {
            let mean = eval(terms)?;
            means.push(mean);
            estimates.push(QecObservableEstimate {
                mean,
                variance: 0.0,
                num_shots: program.options().shots,
            });
        }
        return build_qec_result_from_observable_means(program, means, estimates);
    }

    let postsel = lower_postselection_rows(&postsel_rows, &record_to_qubit)?;
    let (means, _obs_discarded, accept_rate) =
        evaluate_conditional_expectations(&observable_terms_list, &postsel, |terms| {
            Ok(ConditionalEval {
                mean: eval(terms)?,
                discarded_sq: 0.0,
            })
        })?;
    let estimate_shots = accepted_shots_for_rate(program.options().shots, accept_rate);
    let estimates: Vec<_> = means
        .iter()
        .map(|&mean| QecObservableEstimate {
            mean,
            variance: 0.0,
            num_shots: estimate_shots,
        })
        .collect();
    build_qec_result_with_acceptance(program, means, estimates, accept_rate)
}

/// Tolerance on `⟨ψ|S|ψ⟩ = +1` for a rerouting stabilizer.
const REROUTE_STABILIZER_TOL: f64 = 1e-6;

/// Verify that the Z-string distinguishing the original observable support from
/// the rerouted support is a genuine `+1` stabilizer of the evaluated state.
///
/// Rerouting replaces observable `O` with `O' = O · S`, which preserves the
/// expectation only when `S |ψ⟩ = +|ψ⟩`. The reroute module documents this as a
/// caller obligation but cannot check it (it has no state). Here the state is
/// available cheaply via SPD on `S` alone, so validate it rather than silently
/// evaluate a different operator. `S = supp(original) ⊕ supp(rerouted)`; an
/// empty support means no reroute happened and is trivially valid.
fn verify_reroute_is_state_stabilizer(
    circuit: &Circuit,
    original_support: &[usize],
    rerouted_support: &[usize],
) -> Result<()> {
    let stab_support = xor_z_support(original_support, rerouted_support);
    if stab_support.is_empty() {
        return Ok(());
    }
    let stab_terms: Vec<PauliTerm> = stab_support.iter().copied().map(PauliTerm::z).collect();
    let stab =
        run_spd_observable_light_cone(circuit, &stab_terms, QEC_SPD_EPSILON, QEC_SPD_MAX_TERMS)?;
    if (stab.mean - 1.0).abs() > REROUTE_STABILIZER_TOL {
        return Err(PrismError::InvalidParameter {
            message: format!(
                "reroute stabilizer on qubits {stab_support:?} is not a +1 stabilizer of the \
                 state (⟨S⟩ = {:.6}); rerouting would evaluate a different operator",
                stab.mean
            ),
        });
    }
    Ok(())
}

pub fn run_qec_program_spd_rerouted(
    program: &QecProgram,
    reroutes: &[QecObservableReroute],
) -> Result<QecSampleResult> {
    let (circuit, record_to_qubit) = lower_qec_program_for_pauli_observable(program)?;
    let observable_rows = program.observable_rows()?;
    let postsel_rows = program.postselection_rows()?;
    if !postsel_rows.is_empty() {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC SPD rerouted".to_string(),
            reason: "rerouted analytical SPD does not yet combine stabilizer rerouting with postselection projectors".to_string(),
        });
    }

    // Reroute stabilizers and the observable support must live in the same
    // qubit space. Observable support comes from `record_to_qubit`, which
    // resolves to lowered circuit qubits; a `Reset` reassigns a logical
    // qubit to a fresh higher index, so a stabilizer expressed in original
    // logical-qubit indices would silently land on the wrong physical qubit.
    // Reject such programs rather than evaluate a different operator.
    if program
        .ops()
        .iter()
        .any(|op| matches!(op, QecOp::Reset { .. }))
    {
        return Err(PrismError::IncompatibleBackend {
            backend: "QEC SPD rerouted".to_string(),
            reason: "stabilizer rerouting is not supported for programs containing RESET; \
                     resets relabel logical qubits, making stabilizer qubit indices ambiguous"
                .to_string(),
        });
    }

    let mut reroute_by_observable: Vec<Option<&QecObservableReroute>> =
        vec![None; observable_rows.len()];
    for reroute in reroutes {
        let slot = reroute_by_observable
            .get_mut(reroute.observable)
            .ok_or_else(|| PrismError::InvalidParameter {
                message: format!(
                    "reroute references observable {} but program has {} observables",
                    reroute.observable,
                    observable_rows.len()
                ),
            })?;
        if slot.is_some() {
            return Err(PrismError::InvalidParameter {
                message: format!("duplicate reroute for observable {}", reroute.observable),
            });
        }
        validate_z_stabilizers(&reroute.stabilizers, program.num_qubits())?;
        *slot = Some(reroute);
    }

    let mut means = Vec::with_capacity(observable_rows.len());
    let mut estimates = Vec::with_capacity(observable_rows.len());
    for (observable, row) in observable_rows.iter().enumerate() {
        let mut terms = observable_row_to_pauli_terms(row, &record_to_qubit)?;
        if let Some(reroute) = reroute_by_observable[observable] {
            let support: Vec<usize> = terms.iter().map(|term| term.qubit).collect();
            let route = min_cone_z_representative(&circuit, &support, &reroute.stabilizers)?;
            verify_reroute_is_state_stabilizer(&circuit, &support, &route.rerouted_support)?;
            terms = route
                .rerouted_support
                .iter()
                .copied()
                .map(PauliTerm::z)
                .collect();
        }
        let result =
            run_spd_observable_light_cone(&circuit, &terms, QEC_SPD_EPSILON, QEC_SPD_MAX_TERMS)?;
        means.push(result.mean);
        estimates.push(QecObservableEstimate {
            mean: result.mean,
            variance: result.total_discarded * result.total_discarded,
            num_shots: program.options().shots,
        });
    }

    build_qec_result_from_observable_means(program, means, estimates)
}

/// SPD analytical path for QEC observables.
///
/// Backward-propagates the observable through its inverse light cone, branching each
/// T into two weighted Pauli terms, truncating terms below
/// `QEC_SPD_EPSILON` when the sum exceeds `QEC_SPD_MAX_TERMS`. Returns
/// the diagonal sum on `|0^n⟩` per observable. Truncation error is
/// reported via `QecObservableEstimate::variance = total_discarded²`.
fn run_qec_program_spd(program: &QecProgram) -> Result<QecSampleResult> {
    let (circuit, record_to_qubit) = lower_qec_program_for_pauli_observable(program)?;
    let observable_rows = program.observable_rows()?;
    let postsel_rows = program.postselection_rows()?;
    let mut observable_terms_list = Vec::with_capacity(observable_rows.len());
    for row in observable_rows.iter() {
        observable_terms_list.push(observable_row_to_pauli_terms(row, &record_to_qubit)?);
    }

    if postsel_rows.is_empty() {
        let mut means = Vec::with_capacity(observable_terms_list.len());
        let mut estimates = Vec::with_capacity(observable_terms_list.len());
        for terms in &observable_terms_list {
            let result =
                run_spd_observable_light_cone(&circuit, terms, QEC_SPD_EPSILON, QEC_SPD_MAX_TERMS)?;
            means.push(result.mean);
            estimates.push(QecObservableEstimate {
                mean: result.mean,
                variance: result.total_discarded * result.total_discarded,
                num_shots: program.options().shots,
            });
        }
        return build_qec_result_from_observable_means(program, means, estimates);
    }

    let postsel = lower_postselection_rows(&postsel_rows, &record_to_qubit)?;
    let (means, obs_discarded, accept_rate) =
        evaluate_conditional_expectations(&observable_terms_list, &postsel, |terms| {
            let result =
                run_spd_observable_light_cone(&circuit, terms, QEC_SPD_EPSILON, QEC_SPD_MAX_TERMS)?;
            Ok(ConditionalEval {
                mean: result.mean,
                discarded_sq: result.total_discarded * result.total_discarded,
            })
        })?;
    let estimate_shots = accepted_shots_for_rate(program.options().shots, accept_rate);
    let estimates: Vec<_> = means
        .iter()
        .zip(obs_discarded.iter())
        .map(|(&m, &discarded_sq)| QecObservableEstimate {
            mean: m,
            variance: discarded_sq,
            num_shots: estimate_shots,
        })
        .collect();
    build_qec_result_with_acceptance(program, means, estimates, accept_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SEED: u64 = 0xDEAD_BEEF;

    struct AutoHookGuard;

    impl AutoHookGuard {
        fn force_spd_nonexact_and_camps_failure() -> Self {
            super::auto_test_hooks::set_force_spd_nonexact(true);
            super::auto_test_hooks::set_force_camps_failure(true);
            Self
        }
    }

    impl Drop for AutoHookGuard {
        fn drop(&mut self) {
            super::auto_test_hooks::set_force_spd_nonexact(false);
            super::auto_test_hooks::set_force_camps_failure(false);
        }
    }

    fn options(shots: usize) -> QecOptions {
        QecOptions {
            shots,
            seed: TEST_SEED,
            chunk_size: Some(128),
            keep_measurements: false,
        }
    }

    fn h_t_h_program(shots: usize) -> QecProgram {
        let mut program = QecProgram::with_options(1, options(shots));
        program.push_gate(Gate::H, &[0]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        program.push_gate(Gate::H, &[0]).unwrap();
        let m0 = program.measure_z(0).unwrap();
        program
            .observable_include(0, &[QecRecordRef::absolute(m0)])
            .unwrap();
        program
    }

    fn entangled_xor_program(shots: usize) -> QecProgram {
        let mut program = QecProgram::with_options(3, options(shots));
        program.push_gate(Gate::H, &[0]).unwrap();
        program.push_gate(Gate::Cx, &[0, 1]).unwrap();
        program.push_gate(Gate::Cx, &[1, 2]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        for qubit in 0..3 {
            program.push_gate(Gate::H, &[qubit]).unwrap();
        }
        let mut records = Vec::new();
        for qubit in 0..3 {
            records.push(QecRecordRef::absolute(program.measure_z(qubit).unwrap()));
        }
        program.observable_include(0, &records).unwrap();
        program
    }

    fn postselected_program(shots: usize) -> QecProgram {
        let mut program = QecProgram::with_options(1, options(shots));
        program.push_gate(Gate::H, &[0]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        let m0 = program.measure_z(0).unwrap();
        program.reset(QecBasis::Z, 0).unwrap();
        program.push_gate(Gate::H, &[0]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        let m1 = program.measure_z(0).unwrap();
        program
            .postselect(&[QecRecordRef::absolute(m0)], false)
            .unwrap();
        program
            .observable_include(0, &[QecRecordRef::absolute(m1)])
            .unwrap();
        program
    }

    fn deterministic_t_zero_program(shots: usize) -> QecProgram {
        let mut program = QecProgram::with_options(1, options(shots));
        program.push_gate(Gate::T, &[0]).unwrap();
        let m0 = program.measure_z(0).unwrap();
        program
            .observable_include(0, &[QecRecordRef::absolute(m0)])
            .unwrap();
        program
    }

    fn deterministic_t_one_program(shots: usize) -> QecProgram {
        let mut program = QecProgram::with_options(1, options(shots));
        program.push_gate(Gate::X, &[0]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        let m0 = program.measure_z(0).unwrap();
        program
            .observable_include(0, &[QecRecordRef::absolute(m0)])
            .unwrap();
        program
    }

    fn estimates(result: &QecSampleResult) -> &[QecObservableEstimate] {
        result
            .observable_expectations
            .as_ref()
            .expect("analytical result must populate observable expectations")
    }

    fn assert_estimates_close(actual: &QecSampleResult, expected: &QecSampleResult) {
        let actual_estimates = estimates(actual);
        let expected_estimates = estimates(expected);
        assert_eq!(actual_estimates.len(), expected_estimates.len());
        for (idx, (actual, expected)) in actual_estimates
            .iter()
            .zip(expected_estimates.iter())
            .enumerate()
        {
            assert!(
                (actual.mean - expected.mean).abs() < 1e-10,
                "observable {idx}: tensor-network mean {} differs from expected {}",
                actual.mean,
                expected.mean
            );
            assert_eq!(actual.variance, 0.0);
        }
        assert_eq!(actual.accepted_shots, expected.accepted_shots);
        assert_eq!(actual.discarded_shots, expected.discarded_shots);
        assert_eq!(actual.logical_errors, expected.logical_errors);
    }

    #[test]
    fn tensor_network_observable_matches_spd_on_small_non_truncated_programs() {
        let fixtures = vec![
            h_t_h_program(512),
            entangled_xor_program(512),
            postselected_program(512),
        ];
        for program in fixtures {
            let spd = run_qec_program_spd(&program).unwrap();
            assert!(analytical_result_has_no_truncation(&spd));
            let tensor_network = run_qec_program_tensor_network_observable(&program).unwrap();
            assert_estimates_close(&tensor_network, &spd);
        }
    }

    #[test]
    fn tensor_network_observable_matches_reference_on_deterministic_t_fixtures() {
        let fixtures = vec![
            deterministic_t_zero_program(128),
            deterministic_t_one_program(128),
        ];
        for program in fixtures {
            let reference = run_qec_program_reference(&program).unwrap();
            let tensor_network = run_qec_program_tensor_network_observable(&program).unwrap();
            assert_eq!(tensor_network.accepted_shots, reference.accepted_shots);
            assert_eq!(tensor_network.discarded_shots, reference.discarded_shots);
            assert_eq!(tensor_network.logical_errors, reference.logical_errors);
        }
    }

    #[test]
    fn auto_uses_tensor_network_when_spd_nonexact_and_camps_fails() {
        let program = h_t_h_program(256);
        let expected = run_qec_program_tensor_network_observable(&program).unwrap();
        let _guard = AutoHookGuard::force_spd_nonexact_and_camps_failure();
        let actual = run_qec_program_auto(&program).unwrap();
        assert_estimates_close(&actual, &expected);
    }
}
