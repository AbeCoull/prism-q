//! A circuit held across many parameter bindings, reusing the fusion plan its
//! structure implies.

use super::Circuit;
use super::fusion::fuse_circuit;
use super::parameter::Parameters;
use super::plan::FusionPlan;
use crate::error::Result;
use crate::sim::unified_pauli::PauliTerm;
use crate::sim::{
    BackendKind, ObservableExpectation, PauliObservable, PreparedRoute, RunOutcome, prepared_route,
    simulate,
};

/// A parameter template plus the fusion and dispatch work its structure
/// implies, held across bindings.
///
/// [`new`](Self::new) settles the fused block structure and the backend once, and
/// [`run`](Self::run) rebuilds only the block matrices the angles change. A binding that
/// collapses a block to the identity or a named gate, or flips its diagonality, falls
/// back to the full pass pipeline, so the result matches an independently fused circuit.
///
/// Each terminal answers what the [`Simulate`](crate::sim::Simulate) terminal of the same
/// name answers on the bound circuit under the same backend kind and seed. Where the held
/// backend would answer differently, such as a route that bypasses backends or a
/// template the terminal rejects, the call binds and hands the circuit to `simulate`.
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
        Ok(Self::assemble(template, params, kind, plan, skeleton))
    }

    fn assemble(
        template: Circuit,
        params: Parameters,
        kind: BackendKind,
        plan: Option<FusionPlan>,
        skeleton: Circuit,
    ) -> Self {
        let route = prepared_route(&kind, &template);
        let bound = template.clone();
        let fused = skeleton.clone();
        Self {
            template,
            params,
            kind,
            plan,
            route,
            skeleton,
            bound,
            fused,
            fused_off_plan: false,
        }
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
        let fused = self.bind_for_route(values)?;
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
            None => simulate(circuit).backend(kind.clone()).seed(seed).run(),
        }
    }

    /// Bind `values` and compute `⟨ψ|P|ψ⟩` for each joint Pauli observable, as
    /// [`Simulate::expectation_values`](crate::sim::Simulate::expectation_values) does.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`Parameters::bind`], plus whatever that
    /// terminal reports.
    pub fn expectation_values(
        &mut self,
        values: &[f64],
        observables: &[Vec<PauliTerm>],
        seed: u64,
    ) -> Result<Vec<f64>> {
        let fused = self.bind_for_route(values)?;
        let Self {
            kind,
            route,
            bound,
            fused: fused_circuit,
            ..
        } = self;
        let applied = if fused { &*fused_circuit } else { &*bound };
        if let Some(result) = route
            .as_mut()
            .and_then(|route| route.expectation_values(bound, applied, observables, seed))
        {
            return result;
        }
        simulate(bound)
            .backend(kind.clone())
            .seed(seed)
            .expectation_values(observables)
    }

    /// Bind `values` and compute `⟨H⟩` for a weighted Pauli observable, as
    /// [`Simulate::observable_expectation`](crate::sim::Simulate::observable_expectation)
    /// does, variance included.
    ///
    /// # Errors
    /// Same arity and finiteness conditions as [`Parameters::bind`], plus whatever that
    /// terminal reports.
    pub fn observable_expectation(
        &mut self,
        values: &[f64],
        observable: &PauliObservable,
        seed: u64,
    ) -> Result<ObservableExpectation> {
        let fused = self.bind_for_route(values)?;
        let Self {
            kind,
            route,
            bound,
            fused: fused_circuit,
            ..
        } = self;
        let applied = if fused { &*fused_circuit } else { &*bound };
        if let Some(result) = route
            .as_mut()
            .and_then(|route| route.observable_expectation(applied, observable, seed))
        {
            return result;
        }
        simulate(bound)
            .backend(kind.clone())
            .seed(seed)
            .observable_expectation(observable)
    }

    /// [`run`](Self::run) on each binding in order, under one seed.
    ///
    /// Under the `parallel` feature the bindings split across Rayon workers, each on a
    /// copy of this circuit, at the widths where [`run_batch`](crate::sim::run_batch)
    /// splits. Results are identical to calling `run` in a loop either way.
    ///
    /// # Errors
    /// The error of the first failing binding in list order.
    pub fn run_many<V: AsRef<[f64]> + Sync>(
        &mut self,
        bindings: &[V],
        seed: u64,
    ) -> Result<Vec<RunOutcome>> {
        self.map_bindings(bindings, |prepared, values| prepared.run(values, seed))
    }

    /// [`expectation_values`](Self::expectation_values) on each binding in order,
    /// split as [`run_many`](Self::run_many) splits.
    ///
    /// # Errors
    /// The error of the first failing binding in list order.
    pub fn expectation_values_many<V: AsRef<[f64]> + Sync>(
        &mut self,
        bindings: &[V],
        observables: &[Vec<PauliTerm>],
        seed: u64,
    ) -> Result<Vec<Vec<f64>>> {
        self.map_bindings(bindings, |prepared, values| {
            prepared.expectation_values(values, observables, seed)
        })
    }

    /// [`observable_expectation`](Self::observable_expectation) on each binding in
    /// order, split as [`run_many`](Self::run_many) splits.
    ///
    /// # Errors
    /// The error of the first failing binding in list order.
    pub fn observable_expectation_many<V: AsRef<[f64]> + Sync>(
        &mut self,
        bindings: &[V],
        observable: &PauliObservable,
        seed: u64,
    ) -> Result<Vec<ObservableExpectation>> {
        self.map_bindings(bindings, |prepared, values| {
            prepared.observable_expectation(values, observable, seed)
        })
    }

    /// Bind `values` in the form the held route applies, returning true when that
    /// is the fused stream.
    fn bind_for_route(&mut self, values: &[f64]) -> Result<bool> {
        let fused = self
            .route
            .as_ref()
            .is_some_and(PreparedRoute::supports_fused);
        if fused {
            self.bind_fused(values)?;
        } else {
            self.bind(values)?;
        }
        Ok(fused)
    }

    /// Evaluate `eval` per binding. A worker copy starts from the settled parts
    /// rather than from `self`, whose held backend is not `Sync`; results do not
    /// depend on binding history, so the copy answers as `self` would.
    fn map_bindings<V, T, F>(&mut self, bindings: &[V], eval: F) -> Result<Vec<T>>
    where
        V: AsRef<[f64]> + Sync,
        T: Send,
        F: Fn(&mut Self, &[f64]) -> Result<T> + Sync,
    {
        #[cfg(feature = "parallel")]
        if bindings.len() > 1
            && crate::sim::runs_split_across_workers(&self.kind, self.template.num_qubits)
        {
            use rayon::prelude::*;
            let Self {
                template,
                params,
                kind,
                plan,
                skeleton,
                ..
            } = &*self;
            let results: Vec<Result<T>> = bindings
                .par_iter()
                .map_init(
                    || {
                        Self::assemble(
                            template.clone(),
                            params.clone(),
                            kind.clone(),
                            plan.clone(),
                            skeleton.clone(),
                        )
                    },
                    |worker, values| eval(worker, values.as_ref()),
                )
                .collect();
            return results.into_iter().collect();
        }
        bindings
            .iter()
            .map(|values| eval(self, values.as_ref()))
            .collect()
    }
}
