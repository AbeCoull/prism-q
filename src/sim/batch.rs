//! Routes held across many runs: prepared templates and [`run_batch`].

use super::*;

/// The execution route a template settles on, for a caller that holds one
/// circuit across many bindings and fuses it itself.
///
/// Resolving the route from the template rather than from a bound circuit
/// matters twice over: it is the work being amortized, and a fused stream
/// misreports the circuit to dispatch (a fused Clifford circuit no longer
/// looks Clifford).
pub(crate) struct PreparedRoute {
    pub(super) plan: BackendPlan,
    /// Resolved when the route was settled; every point runs under them.
    limits: ResolvedLimits,
    supports_fused: bool,
    fusion_state_qubits: usize,
    fusion_profile: crate::circuit::fusion::FusionProfile,
    /// Backend held across points. `init` reuses its state buffer when the
    /// width matches, so a sweep pays one `2^n` allocation rather than one per
    /// point. Rebuilt when the seed changes, since the seed feeds its RNG.
    held: Option<(u64, Box<dyn Backend + Send>)>,
    /// Set when the template reads the RNG, so every point gets a fresh backend
    /// and measures what a solo run with its seed measures.
    draws_randomness: bool,
    pub(super) values: PreparedValues,
    pub(super) observable: PreparedObservable,
}

/// Who answers [`Simulate::expectation_values`] for a prepared template.
#[derive(Clone, Copy)]
pub(super) enum PreparedValues {
    Held,
    /// `Auto` from [`TENSOR_ROUTE_MIN_QUBITS`]: the bound circuit is offered to
    /// the scalar tensor route first, as `Simulate` offers it.
    TensorFirst,
    /// `Simulate` would take a route the held backend does not reproduce.
    Simulate,
}

/// Who answers [`Simulate::observable_expectation`] for a prepared template.
#[derive(Clone, Copy)]
pub(super) enum PreparedObservable {
    /// Term by term on the held backend, with no variance.
    Held,
    /// The grouped evaluator on the held statevector, with its variance.
    Grouped,
    /// `Simulate` would take a route the held backend does not reproduce.
    Simulate,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ShiftTensorVerdict {
    Undecided,
    UseTensor,
    UseHeld,
}

thread_local! {
    static SHIFT_TENSOR_VERDICT: std::cell::Cell<Option<ShiftTensorVerdict>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    pub(super) static TENSOR_EXPECTATION_DECISIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub(super) struct ShiftTensorScope {
    previous: Option<ShiftTensorVerdict>,
}

impl ShiftTensorScope {
    pub(super) fn enter() -> Self {
        Self::enter_with(Some(ShiftTensorVerdict::Undecided))
    }

    pub(super) fn enter_with(verdict: Option<ShiftTensorVerdict>) -> Self {
        let previous = SHIFT_TENSOR_VERDICT.with(|cell| cell.replace(verdict));
        Self { previous }
    }

    pub(super) fn current() -> Option<ShiftTensorVerdict> {
        SHIFT_TENSOR_VERDICT.with(|cell| cell.get())
    }
}

impl Drop for ShiftTensorScope {
    fn drop(&mut self) {
        SHIFT_TENSOR_VERDICT.with(|cell| cell.set(self.previous));
    }
}

impl PreparedRoute {
    /// True when the chosen backend accepts fused gates, so the caller should
    /// hand `run` a fused stream rather than the bound template.
    pub(crate) fn supports_fused(&self) -> bool {
        self.supports_fused
    }

    /// The width `simulate` fuses at for the held backend, so a stream fused
    /// here matches the one a direct run applies.
    pub(crate) fn fusion_state_qubits(&self) -> usize {
        self.fusion_state_qubits
    }

    pub(crate) fn fusion_profile(&self) -> crate::circuit::fusion::FusionProfile {
        self.fusion_profile
    }

    /// Apply `circuit` verbatim, with no further fusion.
    pub(crate) fn run(&mut self, circuit: &Circuit, seed: u64) -> Result<RunOutcome> {
        if self.draws_randomness {
            let mut backend = self.plan.build(seed, &self.limits);
            return execute_circuit(&mut *backend, circuit, &SimOptions::default());
        }
        execute_circuit(self.held_backend(seed), circuit, &SimOptions::default())
    }

    /// [`Simulate::expectation_values`] on `bound`, evaluated by applying
    /// `applied` verbatim, or `None` when the caller should ask `Simulate`.
    pub(crate) fn expectation_values(
        &mut self,
        bound: &Circuit,
        applied: &Circuit,
        observables: &[Vec<PauliTerm>],
        seed: u64,
        require_exact: bool,
    ) -> Option<Result<Vec<f64>>> {
        match self.values {
            PreparedValues::Simulate => return None,
            PreparedValues::TensorFirst => {
                let cached = ShiftTensorScope::current();
                if cached != Some(ShiftTensorVerdict::UseHeld) {
                    #[cfg(test)]
                    TENSOR_EXPECTATION_DECISIONS.with(|c| c.set(c.get() + 1));
                    if let Some(result) =
                        decide_tensor_expectations(bound, observables, &self.limits)
                    {
                        if cached == Some(ShiftTensorVerdict::Undecided) {
                            SHIFT_TENSOR_VERDICT
                                .with(|c| c.set(Some(ShiftTensorVerdict::UseTensor)));
                        }
                        return Some(result.and_then(|planned| planned.contract(&self.limits)));
                    }
                    if cached == Some(ShiftTensorVerdict::Undecided) {
                        SHIFT_TENSOR_VERDICT.with(|c| c.set(Some(ShiftTensorVerdict::UseHeld)));
                    }
                }
            }
            PreparedValues::Held => {}
        }
        if require_exact {
            let route = ExpectationRoute::Native(self.plan.clone());
            if let Err(error) =
                plan::require_exact_expectation(&route, &BackendKind::Auto, bound, &self.limits)
            {
                return Some(Err(error));
            }
        }
        Some(
            self.held_expectation_values(applied, observables, seed)
                .and_then(|(values, metadata)| {
                    ensure_exact_result(require_exact, &metadata)?;
                    Ok(values)
                }),
        )
    }

    /// [`Simulate::observable_expectation`] on the circuit `applied` was bound
    /// from, or `None` when the caller should ask `Simulate`.
    pub(crate) fn observable_expectation(
        &mut self,
        applied: &Circuit,
        observable: &PauliObservable,
        seed: u64,
    ) -> Option<Result<ObservableExpectation>> {
        match self.observable {
            PreparedObservable::Simulate => None,
            PreparedObservable::Held => Some(
                self.held_expectation_values(applied, &observable_vecs(observable), seed)
                    .map(|(values, metadata)| {
                        weighted_observable_result(observable, &values, None, metadata)
                    }),
            ),
            PreparedObservable::Grouped => Some(self.held_grouped(applied, observable, seed)),
        }
    }

    fn held_backend(&mut self, seed: u64) -> &mut dyn Backend {
        if !matches!(&self.held, Some((s, _)) if *s == seed) {
            self.held = Some((seed, self.plan.build(seed, &self.limits)));
        }
        let (_, backend) = self.held.as_mut().expect("just built");
        &mut **backend
    }

    fn held_expectation_values(
        &mut self,
        circuit: &Circuit,
        observables: &[Vec<PauliTerm>],
        seed: u64,
    ) -> Result<(Vec<f64>, RunMetadata)> {
        for observable in observables {
            validate_observable(observable, circuit.num_qubits)?;
        }
        let backend = self.held_backend(seed);
        execute_circuit(&mut *backend, circuit, &SimOptions::classical_only())?;
        Ok((
            backend.pauli_expectations(observables)?,
            backend_metadata(backend),
        ))
    }

    fn held_grouped(
        &mut self,
        circuit: &Circuit,
        observable: &PauliObservable,
        seed: u64,
    ) -> Result<ObservableExpectation> {
        for (_, factors) in observable.terms() {
            validate_observable(factors, circuit.num_qubits)?;
        }
        let backend = self.held_backend(seed);
        execute_circuit(&mut *backend, circuit, &SimOptions::classical_only())?;
        let statevector = backend
            .as_any()
            .and_then(|any| any.downcast_ref::<StatevectorBackend>())
            .expect("a grouped answer is settled only on a statevector plan");
        grouped_expectation_on_state(statevector, observable)
    }
}

/// Settle the route for `template`, or `None` when it takes one that reshapes
/// execution (decomposition, stabilizer rank, temporal Clifford) or expands
/// `QftBlock`, where a caller-supplied stream has nowhere to go.
pub(crate) fn prepared_route(
    kind: &BackendKind,
    template: &Circuit,
    request: Option<&Limits>,
) -> Option<PreparedRoute> {
    let limits = &ResolvedLimits::resolve(request);
    if limits.validate().is_err() {
        return None;
    }
    if validate_explicit_backend(kind, template, limits).is_err() {
        return None;
    }
    let ProbabilityRoute::Direct {
        has_partial_independence,
    } = plan_probability_route(kind, template, limits)
    else {
        return None;
    };
    let ExecutionPlan::Backend(plan) = resolve(kind, template, has_partial_independence, limits)
    else {
        return None;
    };
    let probe = plan.build(0, limits);
    let has_qft_block = crate::circuit::any_gate(&template.instructions, &mut |gate| {
        matches!(gate, crate::gates::Gate::QftBlock { .. })
    });
    if has_qft_block && !probe.supports_qft_block() {
        return None;
    }
    let has_pauli_rot = crate::circuit::any_gate(&template.instructions, &mut |gate| {
        matches!(gate, crate::gates::Gate::PauliRot(_))
    });
    if has_pauli_rot && !probe.supports_pauli_rotation() {
        return None;
    }
    let (values, observable) = prepared_expectations(kind, template, &*probe, limits);
    Some(PreparedRoute {
        supports_fused: probe.supports_fused_gates(),
        fusion_state_qubits: probe.fusion_state_qubits(template.num_qubits),
        fusion_profile: probe.fusion_profile(template.num_qubits),
        plan,
        limits: *limits,
        held: None,
        draws_randomness: draws_randomness(template),
        values,
        observable,
    })
}

/// Settle which expectation terminals the held backend `probe` answers the
/// way `Simulate` answers them.
///
/// Both decline a template the terminals reject, whose error `Simulate`
/// reports, and a Clifford template under `Auto` or a stabilizer kind, which
/// `Simulate` propagates through the Pauli route rather than a backend. The
/// weighted observable takes the grouped evaluator wherever `Simulate` would,
/// so its variance survives, and declines when the held backend is not the
/// statevector that evaluator reads.
pub(super) fn prepared_expectations(
    kind: &BackendKind,
    template: &Circuit,
    probe: &dyn Backend,
    limits: &ResolvedLimits,
) -> (PreparedValues, PreparedObservable) {
    let clifford_route =
        (kind.is_auto() || kind.is_stabilizer_family()) && template.is_clifford_only();
    if draws_randomness(template)
        || template.save_count() > 0
        || clifford_route
        || !probe.supports_pauli_expectation()
    {
        return (PreparedValues::Simulate, PreparedObservable::Simulate);
    }
    let fits_dense = template.num_qubits <= limits.statevector_qubits.value;
    let values = if kind.is_auto() && fits_dense && template.num_qubits >= TENSOR_ROUTE_MIN_QUBITS {
        PreparedValues::TensorFirst
    } else {
        PreparedValues::Held
    };
    let grouped = match kind {
        BackendKind::Statevector => true,
        #[cfg(feature = "gpu")]
        BackendKind::StatevectorGpu { .. } => true,
        _ => kind.is_auto() && fits_dense,
    };
    let holds_statevector = probe
        .as_any()
        .is_some_and(|any| any.is::<StatevectorBackend>());
    let observable = match (grouped, holds_statevector) {
        (false, _) => PreparedObservable::Held,
        (true, true) => PreparedObservable::Grouped,
        (true, false) => PreparedObservable::Simulate,
    };
    (values, observable)
}

/// Run several circuits, holding one backend across those that can share it.
///
/// Under the `parallel` feature, a batch whose circuits all sit at 16 qubits or
/// fewer splits across Rayon workers, each holding its own backend: a 200-point
/// sweep of a two-layer hardware-efficient ansatz ran 3.2x to 4.8x faster than a
/// loop over [`simulate`] at 4 to 12 qubits on a four-core i7-6700K, and one circuit
/// per core beat the kernels' own threads at 14 and 16. A wider circuit, a density
/// matrix from 9 qubits, or a GPU or distributed kind keeps the whole batch on one
/// thread.
///
/// On one thread the saving is the backend construction and its `2^n`
/// allocation, which `init` reuses when the next circuit has the same width;
/// route analysis is planned per circuit either way. On the same sweep that read
/// within about 10% of the loop, in both directions.
///
/// Results are identical to running each circuit on its own with the same seed.
/// A circuit that draws randomness (a measurement, a reset, or a classical
/// condition) gets a backend of its own, because a held backend would carry its
/// RNG forward into the next circuit and change what the next one measures.
///
/// # Errors
/// Returns the error of the first failing circuit in list order. A caller that
/// wants every circuit's result should call this per circuit.
pub fn run_batch(circuits: &[Circuit], kind: BackendKind, seed: u64) -> Result<Vec<RunOutcome>> {
    let limits = &ResolvedLimits::process_default();
    limits.validate()?;
    #[cfg(feature = "parallel")]
    if circuits.len() > 1
        && circuits
            .iter()
            .all(|c| runs_split_across_workers(&kind, c.num_qubits))
    {
        use rayon::prelude::*;
        let outcomes: Vec<Result<RunOutcome>> = circuits
            .par_iter()
            .map_init(
                || None,
                |held, circuit| run_batch_entry(&kind, circuit, seed, held, limits),
            )
            .collect();
        return outcomes.into_iter().collect();
    }

    let mut held = None;
    circuits
        .iter()
        .map(|circuit| run_batch_entry(&kind, circuit, seed, &mut held, limits))
        .collect()
}

pub(super) type HeldBackend = Option<(dispatch::BackendPlan, usize, Box<dyn Backend + Send>)>;

pub(super) fn run_batch_entry(
    kind: &BackendKind,
    circuit: &Circuit,
    seed: u64,
    held: &mut HeldBackend,
    limits: &ResolvedLimits,
) -> Result<RunOutcome> {
    let Some(plan) = batch_plan(kind, circuit, limits) else {
        *held = None;
        return run_with_internal(kind.clone(), circuit, seed, SimOptions::default(), limits);
    };
    validate_explicit_backend(kind, circuit, limits)?;

    let reusable = matches!(
        &*held,
        Some((p, width, _)) if *width == circuit.num_qubits && dispatch::same_plan(p, &plan)
    );
    if !reusable {
        *held = Some((plan.clone(), circuit.num_qubits, plan.build(seed, limits)));
    }
    let (_, _, backend) = held.as_mut().expect("just built");
    execute(&mut **backend, circuit, &SimOptions::default()).map(|mut r| {
        r.metadata.reason = if plan.is_gpu() && r.metadata.placement == Placement::Host {
            RouteRule::DeviceInitializationFallback
        } else {
            backend_rule(kind, circuit, &plan)
        };
        r
    })
}

/// Whether separate runs of `num_qubits`-wide circuits on `kind` should split across
/// Rayon workers: only below `RUN_SPLIT_QUBITS`, counting a density matrix at
/// twice its width since each worker holds a state, and never on a kind bound to one
/// device or rank context.
#[cfg(feature = "parallel")]
pub(crate) fn runs_split_across_workers(kind: &BackendKind, num_qubits: usize) -> bool {
    if bound_to_one_context(kind) {
        return false;
    }
    let width = if kind.is_density_matrix() {
        2 * num_qubits
    } else {
        num_qubits
    };
    width < RUN_SPLIT_QUBITS
}

/// Whether per-shot runs on `kind` should split across Rayon workers, given the
/// resolved family and width of each state one shot holds. See
/// [`state_splits_across_workers`] for the per-state rule.
#[cfg(feature = "parallel")]
pub(super) fn shots_split_across_workers(
    kind: &BackendKind,
    states: &[(ResolvedBackend, usize)],
) -> bool {
    !bound_to_one_context(kind)
        && states
            .iter()
            .all(|&(family, width)| state_splits_across_workers(family, width))
}

/// Whether separate runs each holding one `width`-qubit state of `family` should
/// split across Rayon workers: only below the width where that family's own
/// kernels go parallel, since each worker holds a state and a shot task nested
/// inside kernel joins can pile stolen frames onto one worker stack.
///
/// A tableau parallelizes its row loops only from `MIN_QUBITS_FOR_PAR_GATES`
/// qubits, and a product state parallelizes no gate, measurement or reset. A
/// density matrix counts at twice its width.
#[cfg(feature = "parallel")]
pub(crate) fn state_splits_across_workers(family: ResolvedBackend, width: usize) -> bool {
    use crate::backend::{MIN_QUBITS_FOR_PAR_GATES, PARALLEL_THRESHOLD_QUBITS};
    match family {
        ResolvedBackend::Stabilizer | ResolvedBackend::FactoredStabilizer => {
            width < MIN_QUBITS_FOR_PAR_GATES
        }
        ResolvedBackend::ProductState => true,
        ResolvedBackend::DensityMatrix => 2 * width < PARALLEL_THRESHOLD_QUBITS,
        _ => width < PARALLEL_THRESHOLD_QUBITS,
    }
}

#[cfg(feature = "parallel")]
pub(super) fn bound_to_one_context(kind: &BackendKind) -> bool {
    match kind {
        #[cfg(feature = "gpu")]
        BackendKind::AutoGpu { .. }
        | BackendKind::StatevectorGpu { .. }
        | BackendKind::DensityMatrixGpu { .. }
        | BackendKind::StabilizerGpu { .. } => true,
        #[cfg(feature = "distributed")]
        BackendKind::StatevectorDistributed { .. } => true,
        _ => false,
    }
}

/// The plan a circuit can share, or `None` when it must run on its own.
///
/// A circuit that draws randomness needs a fresh RNG to match its solo run, and
/// anything that is not a direct backend route has no single plan to hold.
pub(super) fn batch_plan(
    kind: &BackendKind,
    circuit: &Circuit,
    limits: &ResolvedLimits,
) -> Option<dispatch::BackendPlan> {
    if draws_randomness(circuit) {
        return None;
    }
    let ProbabilityRoute::Direct {
        has_partial_independence,
    } = plan_probability_route(kind, circuit, limits)
    else {
        return None;
    };
    match resolve(kind, circuit, has_partial_independence, limits) {
        ExecutionPlan::Backend(plan) => Some(plan),
        _ => None,
    }
}

/// True when a run of `circuit` reads the backend RNG, so a backend carried over
/// from an earlier run would change its outcome.
pub(super) fn draws_randomness(circuit: &Circuit) -> bool {
    circuit.instructions.iter().any(|i| {
        matches!(
            i,
            Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_)
        )
    })
}
