//! A circuit held across many parameter bindings, reusing the fusion plan its
//! structure implies.

use super::Circuit;
use super::fusion::fuse_circuit;
use super::parameter::Parameters;
use super::plan::FusionPlan;
use crate::error::Result;
use crate::sim::{BackendKind, PreparedRoute, RunOutcome, prepared_route};

/// A parameter template plus the fusion and dispatch work its structure
/// implies, held across bindings.
///
/// [`new`](Self::new) settles the fused block structure and the backend once, and
/// [`run`](Self::run) rebuilds only the block matrices the angles change. A binding that
/// collapses a block to the identity or a named gate, or flips its diagonality, falls
/// back to the full pass pipeline, so the result matches an independently fused circuit.
pub struct PreparedCircuit {
    template: Circuit,
    params: Parameters,
    kind: BackendKind,
    plan: Option<FusionPlan>,
    route: Option<PreparedRoute>,
    skeleton: Circuit,
    bound: Circuit,
    fused: Circuit,
    /// Set when `fused` last held a fallback stream rather than the skeleton,
    /// so the next replay restores the recorded structure before patching it.
    fused_off_plan: bool,
}

impl Clone for PreparedCircuit {
    fn clone(&self) -> Self {
        Self {
            template: self.template.clone(),
            params: self.params.clone(),
            kind: self.kind.clone(),
            plan: self.plan.clone(),
            route: prepared_route(&self.kind, &self.template),
            skeleton: self.skeleton.clone(),
            bound: self.bound.clone(),
            fused: self.fused.clone(),
            fused_off_plan: self.fused_off_plan,
        }
    }
}

impl PreparedCircuit {
    /// Settle the fused structure and the backend choice for `template` under
    /// `params`, with automatic backend selection.
    ///
    /// # Errors
    /// Returns [`PrismError::InvalidParameter`](crate::PrismError::InvalidParameter)
    /// when `params` does not validate against `template`.
    pub fn new(template: Circuit, params: Parameters) -> Result<Self> {
        Self::with_backend(template, params, BackendKind::Auto)
    }

    /// Settle against an explicit backend.
    ///
    /// # Errors
    /// Same conditions as [`new`](Self::new).
    pub fn with_backend(template: Circuit, params: Parameters, kind: BackendKind) -> Result<Self> {
        params.validate(&template)?;
        let (skeleton, plan) = FusionPlan::capture(&template);
        let route = prepared_route(&kind, &template);
        let bound = template.clone();
        let fused = skeleton.clone();
        Ok(Self {
            template,
            params,
            kind,
            plan,
            route,
            skeleton,
            bound,
            fused,
            fused_off_plan: false,
        })
    }

    pub fn template(&self) -> &Circuit {
        &self.template
    }

    pub fn parameters(&self) -> &Parameters {
        &self.params
    }

    /// True when bindings reuse the captured fused structure, false when every binding
    /// re-runs the pass pipeline. Results agree either way.
    pub fn reuses_fusion_plan(&self) -> bool {
        self.plan.is_some()
    }

    /// Bind `values` and return the unfused circuit.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`Parameters::bind`].
    pub fn bind(&mut self, values: &[f64]) -> Result<&Circuit> {
        self.params.check_values(values)?;
        self.params.write_angles(&mut self.bound, values);
        Ok(&self.bound)
    }

    /// Bind `values` and return the circuit fused for a backend that accepts
    /// fused gates.
    ///
    /// The result is legal only on such a backend: a fused Clifford circuit no
    /// longer reads as Clifford, so handing this to an explicit stabilizer run
    /// fails. Prefer [`run`](Self::run), which picks the right form.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`Parameters::bind`].
    pub fn bind_fused(&mut self, values: &[f64]) -> Result<&Circuit> {
        self.params.check_values(values)?;
        self.params.write_angles(&mut self.bound, values);

        if let Some(plan) = &self.plan {
            if self.fused_off_plan {
                self.fused.clone_from(&self.skeleton);
                self.fused_off_plan = false;
            }
            if plan.replay(&self.bound, &mut self.fused) {
                return Ok(&self.fused);
            }
            self.fused_off_plan = true;
        }
        self.fused = fuse_circuit(&self.bound, true).into_owned();
        Ok(&self.fused)
    }

    /// Bind `values` and execute, reusing the settled backend choice and, where
    /// the backend accepts fused gates, the fusion plan.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`Parameters::bind`], plus whatever the
    /// backend reports.
    pub fn run(&mut self, values: &[f64], seed: u64) -> Result<RunOutcome> {
        let fused = self
            .route
            .as_ref()
            .is_some_and(PreparedRoute::supports_fused);
        if fused {
            self.bind_fused(values)?;
        } else {
            self.bind(values)?;
        }
        let Self {
            kind,
            route,
            bound,
            fused: fused_circuit,
            ..
        } = self;
        let circuit = if fused { &*fused_circuit } else { &*bound };
        match route {
            Some(route) => route.run(circuit, seed),
            None => crate::sim::simulate(circuit)
                .backend(kind.clone())
                .seed(seed)
                .run(),
        }
    }
}
