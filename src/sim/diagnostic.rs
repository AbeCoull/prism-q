//! Backends built for the state diagnostics: statevector export, reduced density
//! matrix, entanglement entropy and overlap.

use super::*;

/// Which state diagnostic a terminal is asking for, carried into dispatch so
/// the route can be judged before the circuit runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Diagnostic {
    StateVector,
    ReducedDensityMatrix,
    Entropy,
    Overlap,
}

impl Diagnostic {
    pub(super) fn terminal(self) -> &'static str {
        match self {
            Diagnostic::StateVector => "a statevector",
            Diagnostic::ReducedDensityMatrix => "a reduced density matrix",
            Diagnostic::Entropy => "entanglement entropy",
            Diagnostic::Overlap => "a state overlap",
        }
    }
}

/// Whether the backend `plan` builds implements `diagnostic` on its own
/// representation.
///
/// Read from the plan rather than from a built state so [`BackendKind::Auto`]
/// can re-route before paying for the run. A plan added without the kernel
/// reads as declining here, which is what its trait default does.
pub(super) fn plan_answers(plan: &BackendPlan, diagnostic: Diagnostic) -> bool {
    match diagnostic {
        Diagnostic::ReducedDensityMatrix => {
            #[cfg(feature = "distributed")]
            if matches!(plan, BackendPlan::Distributed(_)) {
                return true;
            }
            matches!(
                plan,
                BackendPlan::Statevector { .. }
                    | BackendPlan::Sparse
                    | BackendPlan::Factored
                    | BackendPlan::Mps { .. }
                    | BackendPlan::TensorNetwork { .. }
                    | BackendPlan::ProductState
                    | BackendPlan::DensityMatrix { .. }
                    | BackendPlan::Stabilizer { .. }
                    | BackendPlan::FactoredStabilizer
            )
        }
        Diagnostic::Entropy => matches!(
            plan,
            BackendPlan::Statevector { .. }
                | BackendPlan::Factored
                | BackendPlan::Mps { .. }
                | BackendPlan::ProductState
                | BackendPlan::Stabilizer { .. }
                | BackendPlan::FactoredStabilizer
        ),
        Diagnostic::StateVector | Diagnostic::Overlap => {
            !matches!(plan, BackendPlan::DensityMatrix { .. })
        }
    }
}

/// The error for a route that propagates an observable and holds no state,
/// naming the route so the caller knows which engine `kind` reached.
pub(super) fn stateless_route(
    kind: &BackendKind,
    diagnostic: Diagnostic,
    route: &str,
) -> PrismError {
    PrismError::IncompatibleBackend {
        backend: format!("{kind:?}"),
        reason: format!(
            "{} needs a backend that holds a state; the {route} route returns \
             probabilities only",
            diagnostic.terminal()
        ),
    }
}

/// Price a diagnostic's output against the dense export cap before the circuit
/// runs: a reduced density matrix holds `4^k` entries, which depends on `k`
/// and the backend's name alone, both known once the plan is built.
pub(super) fn check_diagnostic_width(
    backend: &str,
    diagnostic: Diagnostic,
    k: usize,
    limits: &ResolvedLimits,
) -> Result<()> {
    match diagnostic {
        Diagnostic::ReducedDensityMatrix => {
            crate::backend::reduced_density::reduced_density_side(backend, k, limits)?;
            Ok(())
        }
        Diagnostic::StateVector => {
            if k > limits.export_cap() {
                return Err(crate::backend::schmidt::export_cap_exceeded(
                    backend,
                    "a dense statevector".to_string(),
                    k,
                    limits,
                ));
            }
            Ok(())
        }
        Diagnostic::Entropy | Diagnostic::Overlap => Ok(()),
    }
}

/// Build the backend `kind` resolves to, run `circuit` on it, and hand it back
/// for one terminal read of its state.
///
/// Resolution goes straight to a single backend, as the native expectation
/// path does: a diagnostic is read off one state, and the decomposed route
/// holds one per independent block. Under [`BackendKind::Auto`] a resolved
/// plan that cannot answer `diagnostic` is replaced by the statevector while
/// the circuit fits its cap, since the route was the dispatcher's choice and
/// not the caller's; an explicit kind keeps its backend and declines.
pub(super) fn diagnostic_backend(
    kind: &BackendKind,
    circuit: &Circuit,
    initial_state: Option<StartState<'_>>,
    seed: u64,
    diagnostic: Diagnostic,
    subsystem_len: usize,
    limits: &ResolvedLimits,
) -> Result<(Box<dyn Backend>, RouteRule)> {
    if let Some(state) = initial_state {
        let (mut backend, reason) = backend_from_initial_state(kind, circuit, state, seed, limits)?;
        check_diagnostic_width(backend.name(), diagnostic, subsystem_len, limits)?;
        apply_fused_without_saves(&mut *backend, circuit, "state diagnostics")?;
        return Ok((backend, reason));
    }
    let (plan, rule) = decide_diagnostic(kind, circuit, None, diagnostic, limits)?;
    let mut backend: Box<dyn Backend> = plan.build(seed, limits);
    check_diagnostic_width(backend.name(), diagnostic, subsystem_len, limits)?;
    execute(&mut *backend, circuit, &SimOptions::classical_only())?;
    let reason = route_metadata(&*backend, plan.is_gpu(), rule).reason;
    Ok((backend, reason))
}
