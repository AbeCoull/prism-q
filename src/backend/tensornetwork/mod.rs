//! Tensor-network simulation backend.
//!
//! Represents the quantum state as a network of tensors. Gate application
//! appends gate tensors to the network (deferred contraction). Contraction
//! happens lazily when `probabilities()` or another query is requested.
//!
//! # Memory layout
//!
//! - Each tensor: contiguous `Vec<Complex64>` plus a shape vector and leg ids
//!   (up to 6 held inline).
//! - Gates append tensors, so memory grows with gate count until a query
//!   contracts the network. Measurement and reset absorb a projector into the
//!   tensor holding the measured qubit's output leg and keep the deferred
//!   form: the outcome marginal contracts the doubled network, so mid-circuit
//!   measurement carries no dense width ceiling and does not grow the network.
//!
//! # Gate support
//!
//! The full gate set: 1q and 2q gates as rank-2/rank-4 tensors, MCU as one
//! dense multi-qubit tensor, batched variants expanded per entry.
//! `Gate::QftBlock` is expanded to textbook gates before dispatch.
//!
//! # When to prefer this backend
//!
//! - Circuits with low treewidth (shallow or geometrically local).
//! - Circuits where full statevector is infeasible (>30 qubits) but structure
//!   permits efficient contraction.
//!
//! # When NOT to use this backend
//!
//! - High-treewidth circuits, where contraction intermediates outgrow the
//!   dense statevector.
//!
//! # Contraction strategy
//!
//! A metadata-only planner picks the pair order, then the kernel replays it.
//! The baseline plan is the greedy min-size heuristic: repeatedly contract the
//! pair of tensors whose result has the smallest total element count,
//! preferring pairs sharing a leg. O(T·r·log T) for T tensors of rank at most
//! r. When the greedy plan's peak intermediate reaches
//! `RESTART_PEAK_THRESHOLD`, the planner reruns with seeded multiplicative
//! noise on the size key, `PLAN_RESTARTS_PER_TEMPERATURE` passes at each
//! `PLAN_NOISE_TEMPERATURES` entry, and keeps the tree with the smallest peak
//! intermediate, the greedy tree included, so the peak never rises. Planning
//! touches shapes and legs only, so a restart costs a heap walk, not data
//! movement. The winning plan's peak is held to `PRISM_MAX_TN_PEAK_QUBITS`
//! (a memory-derived `2^q` elements by default) before the replay allocates
//! anything, so a contraction the host cannot hold errors instead of aborting.
//!
//! # Index slicing at the memory ceiling
//!
//! A plan whose peak clears the cap is not rejected outright. Legs are fixed
//! one at a time, greedily by the peak each choice buys, and the network is
//! contracted once per assignment of the fixed legs and summed: a
//! multiplicative time factor in exchange for a divided peak. Only a plan
//! still over the cap once `PRISM_MAX_TN_SLICES` assignments are on the table
//! is rejected, and the rejection names the cap. Slices are independent, so
//! the parallel build runs them through Rayon, but only as many at once as fit
//! under the cap together: the cap bounds the live intermediates of a sliced
//! run the way it bounds an unsliced one. Sliced legs are shared by exactly two
//! tensors, so the slice results sum; open legs stay whole. Every terminal
//! contracts through the same path, and the result is exact either way.
//!
//! # Bounded contraction
//!
//! [`TensorNetworkBackend::with_tolerance`] trades exactness for reach. With
//! a tolerance set, a plan over the cap first has an intermediate factored
//! across the cut that separates its partner-facing legs from the rest, the
//! new bond kept only as far as the tolerance on that cut's relative
//! discarded weight allows. The factored halves go back into the network and
//! the rest is replanned, so the peak comes down without the whole tensor
//! ever forming. Slicing then covers whatever truncation leaves above the
//! cap. Without a tolerance, slicing is the only lever and the answer is
//! exact.
//!
//! # Observables and shots both contract natively
//!
//! `Backend::pauli_expectations`, `Backend::reduced_density_matrix_1q` and
//! `Backend::reduced_density_matrix` answer by doubling the network against its
//! conjugate: the bra copy's legs are shifted clear of the ket id space, and each
//! qubit's boundary is closed against its twin (a trace), joined through an
//! operator, or left open. No `2^n` vector is built, so none of them passes the
//! dense ceiling.
//!
//! An identity factor is a closed leg rather than an appended tensor, so a
//! weight-`k` observable adds `k` tensors to a network of `2T`, not `n`.
//!
//! `Backend::sample_basis_states` answers below the dense ceiling from one
//! contraction of the full distribution, a measured 66x cheaper than the
//! sweep at 16 qubits and 32 shots. Past the ceiling it samples qubit by
//! qubit: each bit is drawn from the conditioned single-qubit marginal, and
//! the outcome projector is absorbed before the next qubit's marginal, so a
//! shot costs `n` doubled contractions whose peak is set by treewidth rather
//! than `2^n`, each planned on the first shot and replayed from a per-call
//! cache on the rest.
//!
//! `expectation_zero_state` remains a separate path, contracting `⟨0|U†PU|0⟩`
//! from a circuit rather than an evolved backend over the observable's backward
//! light cone. Explicit `TensorNetwork` expectations, the `Auto` bounded route
//! and the QEC estimator ladder use it.

use num_complex::Complex64;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use smallvec::SmallVec;

use crate::backend::{
    Backend, BasisSamples, NORM_CLAMP_MIN, dense_statevector_len, reserve_dense_output, simd,
    tensor_probability_len,
};
use crate::circuit::Instruction;
use crate::error::Result;
use crate::gates::{DiagEntry, Gate};
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
use crate::backend::MIN_PAR_ELEMS;

mod expectation;
mod plan;
mod slice;
mod tensor;

use expectation::{
    GateTensorOp, append_gate_tensor, for_each_gate_tensor, observable_axes, pauli_tensor,
};
use plan::{TensorMeta, greedy_contract, plan_pairs};
use slice::{CachedPlan, ContractionLimits, contract_cached, last_discarded};
use tensor::{Tensor, contract, transpose};

pub(crate) use expectation::{
    AUTO_EXPECTATION_PEAK_BOUND, BoundedExpectations, cone_entangling_gates,
    expectation_zero_state, plan_bounded_expectations, plan_expectations,
};
#[cfg(feature = "bench-internal")]
pub use expectation::{scalar_expectation, scalar_expectation_capped};
#[doc(hidden)]
pub use slice::last_slice_count;

type LegId = usize;

const UNITARY_EPS: f64 = 1e-9;

fn is_unitary_phase(phase: Complex64) -> bool {
    phase.is_finite() && (phase.norm_sqr() - 1.0).abs() <= UNITARY_EPS
}

fn is_unitary_dense(mat: &[Complex64], dim: usize) -> bool {
    if mat.iter().any(|entry| !entry.is_finite()) {
        return false;
    }
    for i in 0..dim {
        for j in i..dim {
            let mut dot = Complex64::new(0.0, 0.0);
            for row in 0..dim {
                dot += mat[row * dim + i].conj() * mat[row * dim + j];
            }
            if i == j {
                dot -= Complex64::new(1.0, 0.0);
            }
            let norm = dot.norm();
            if !norm.is_finite() || norm > UNITARY_EPS {
                return false;
            }
        }
    }
    true
}

/// Tensor-network simulation backend with deferred contraction.
pub struct TensorNetworkBackend {
    num_qubits: usize,
    tensors: Vec<Tensor>,
    output_legs: Vec<LegId>,
    next_leg: usize,
    is_unitary: Vec<bool>,
    classical_bits: Vec<bool>,
    rng: ChaCha8Rng,
    tolerance: Option<f64>,
    /// Discard from the marginals of mid-circuit measurements and resets, which
    /// every later answer inherits.
    run_discarded: f64,
    /// Discard booked so far by the query still being answered.
    query_discarded: std::cell::Cell<f64>,
    worst_query_discarded: std::cell::Cell<f64>,
    limits: ResolvedLimits,
}

impl TensorNetworkBackend {
    /// Backend under the process default [`Limits`](crate::Limits).
    pub fn new(seed: u64) -> Self {
        Self::with_limits(seed, None, &ResolvedLimits::process_default())
    }

    /// A backend whose contractions may truncate an intermediate bond when
    /// the planned peak crosses the memory cap, discarding at most
    /// `tolerance` of a cut tensor's squared weight per cut.
    ///
    /// A tolerance of zero discards nothing and stays exact.
    ///
    /// # Panics
    ///
    /// When `tolerance` is negative or not finite.
    pub fn with_tolerance(seed: u64, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "tensor-network tolerance must be a finite fraction at or above zero, got {tolerance}"
        );
        Self::with_limits(seed, Some(tolerance), &ResolvedLimits::process_default())
    }

    /// A backend under `limits`, truncating up to `tolerance` per cut when set.
    /// The tolerance is checked as [`Self::with_tolerance`] checks it, by the
    /// caller.
    pub(crate) fn with_limits(seed: u64, tolerance: Option<f64>, limits: &ResolvedLimits) -> Self {
        Self {
            num_qubits: 0,
            tensors: Vec::new(),
            output_legs: Vec::new(),
            next_leg: 0,
            is_unitary: Vec::new(),
            classical_bits: Vec::new(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            tolerance,
            run_discarded: 0.0,
            query_discarded: std::cell::Cell::new(0.0),
            worst_query_discarded: std::cell::Cell::new(0.0),
            limits: *limits,
        }
    }

    /// Fraction of squared weight behind the reported bound: the worst single
    /// query answered since [`Backend::init`], plus what the marginals of
    /// mid-circuit measurements and resets discarded.
    ///
    /// A query is one shot of the conditional sweep, one observable of
    /// `pauli_expectations` with the norm contraction it divides by, or one
    /// other terminal contraction. Each truncates its own copy of the network,
    /// so queries do not add up, while the cuts inside one query do. A shot's
    /// cuts depend on its outcomes, so a later shot can still raise the total.
    /// The sum takes one relative discard per cut rather than measuring the
    /// final state, so a query that truncates heavily can carry it past 1,
    /// where the fidelity it implies clamps to zero and certifies nothing.
    ///
    /// [`Backend::init`]: crate::backend::Backend::init
    pub fn truncation_discarded(&self) -> f64 {
        self.run_discarded + self.worst_query_discarded.get()
    }

    /// Limits every terminal of this backend contracts under.
    fn contraction_limits(&self) -> ContractionLimits {
        ContractionLimits::new(&self.limits, self.tolerance)
    }

    /// Add what the contraction just finished discarded to the open query.
    fn book_truncation(&self) {
        self.query_discarded
            .set(self.query_discarded.get() + last_discarded());
    }

    /// Close the open query, keeping its discard when it is the worst so far.
    fn close_query(&self) {
        let discarded = self.query_discarded.replace(0.0);
        if discarded > self.worst_query_discarded.get() {
            self.worst_query_discarded.set(discarded);
        }
    }

    fn fresh_leg(&mut self) -> LegId {
        let id = self.next_leg;
        self.next_leg += 1;
        id
    }

    /// Draw the outcome for `qubit` from its reduced density matrix and append
    /// the renormalizing projector, keeping the network in deferred form.
    ///
    /// The marginal contracts the doubled network, so no `2^n` vector is built
    /// and measurement carries no dense width ceiling. One rng draw per call,
    /// in program order, matching every other backend's outcome stream.
    fn collapse_qubit(&mut self, qubit: usize, reset: bool) -> Result<bool> {
        use rand::RngExt;

        let uniform = self.rng.random::<f64>();
        let outcome = self.collapse_qubit_with(qubit, reset, uniform, None);
        self.run_discarded += self.query_discarded.replace(0.0);
        outcome
    }

    /// Collapse `qubit` with a caller-supplied uniform draw, so the native
    /// sampler can drive collapses from its own seeded stream without
    /// touching the run rng. `plan` is the sampler's cache slot for this
    /// position; `None` plans the marginal afresh.
    fn collapse_qubit_with(
        &mut self,
        qubit: usize,
        reset: bool,
        uniform: f64,
        plan: Option<&mut Option<CachedPlan>>,
    ) -> Result<bool> {
        let rho = self.marginal_1q(qubit, plan)?;
        let trace = (rho[0][0].re + rho[1][1].re).max(NORM_CLAMP_MIN);
        let prob_one = (rho[1][1].re / trace).clamp(0.0, 1.0);
        let outcome = uniform < prob_one;
        let inv_norm = crate::backend::measurement_inv_norm(outcome, prob_one);
        self.append_collapse(qubit, outcome, reset, inv_norm);
        Ok(outcome)
    }

    /// Draw one shot into `samples` by fixing qubits in index order, each bit
    /// from its conditioned marginal with the outcome projector absorbed.
    fn sample_one_shot(
        &mut self,
        rng: &mut ChaCha8Rng,
        shot: usize,
        samples: &mut BasisSamples,
        mut plans: Option<&mut [Option<CachedPlan>]>,
    ) -> Result<()> {
        use rand::RngExt;

        for qubit in 0..self.num_qubits {
            let uniform = rng.random::<f64>();
            let plan = plans.as_deref_mut().map(|plans| &mut plans[qubit]);
            if self.collapse_qubit_with(qubit, false, uniform, plan)? {
                samples.set(shot, qubit);
            }
        }
        Ok(())
    }

    /// Qubit-by-qubit conditional sampling, one doubled-network contraction
    /// per qubit per shot; the module docstring carries the cost trade.
    ///
    /// Each position's contraction is planned, and past the cap sliced, on the
    /// first shot and replayed on the rest: the projector is absorbed into its
    /// owner and leg ids restart with the state, so the doubled network at a
    /// position carries the same metadata in every shot. The cache lives for
    /// this call only, and a fingerprint of that metadata guards every replay.
    fn sample_native(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let mut plans: Vec<Option<CachedPlan>> = std::iter::repeat_with(|| None)
            .take(self.num_qubits)
            .collect();
        self.sample_sweep(num_shots, seed, Some(&mut plans))
    }

    /// The sweep behind [`Self::sample_native`], with the plan cache as a
    /// parameter so a reference run can go without one.
    ///
    /// A pre-flight plans the first marginal and reserves its peak
    /// intermediate as a feasibility proxy: later marginals open a different
    /// qubit and can plan a different tree, so the gate is heuristic, not a
    /// guarantee. The state is restored after every shot, errors included.
    fn sample_sweep(
        &mut self,
        num_shots: usize,
        seed: u64,
        mut plans: Option<&mut [Option<CachedPlan>]>,
    ) -> Result<BasisSamples> {
        let n = self.num_qubits;
        let mut samples = BasisSamples::new(num_shots, n);
        if n == 0 {
            return Ok(samples);
        }

        let network = self.double_for_partial_trace(&[0]);
        let slots: Vec<TensorMeta> = network.iter().map(TensorMeta::of).collect();
        let plan =
            plan_pairs(slots, None, usize::MAX, usize::MAX).expect("unbounded pass completes");
        let probe_len = plan.peak.min(self.contraction_limits().peak_cap);
        let mut probe: Vec<Complex64> = Vec::new();
        reserve_dense_output(&mut probe, probe_len, self.name(), "native sampling")?;
        drop(probe);

        let tensors = self.tensors.clone();
        let output_legs = self.output_legs.clone();
        let next_leg = self.next_leg;
        let is_unitary = self.is_unitary.clone();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for shot in 0..num_shots {
            let drawn = self.sample_one_shot(&mut rng, shot, &mut samples, plans.as_deref_mut());
            self.close_query();
            self.tensors.clone_from(&tensors);
            self.output_legs.clone_from(&output_legs);
            self.next_leg = next_leg;
            self.is_unitary.clone_from(&is_unitary);
            drawn?;
        }
        Ok(samples)
    }

    /// Absorb the rank-2 tensor that keeps only `outcome` on `qubit`, scaled by
    /// `inv_norm`, into the tensor holding the qubit's output leg, mapping the
    /// kept branch to `|0>` when `reset` is set.
    ///
    /// Contracting into the owner instead of appending keeps the tensor count
    /// constant across measurements, so a measure or reset loop costs one
    /// doubled contraction per event rather than growing the network it
    /// contracts.
    fn append_collapse(&mut self, qubit: usize, outcome: bool, reset: bool, inv_norm: f64) {
        let in_leg = self.output_legs[qubit];
        let out_leg = self.fresh_leg();
        let zero = Complex64::new(0.0, 0.0);
        let scale = Complex64::new(inv_norm, 0.0);

        let in_idx = usize::from(outcome);
        let out_idx = if reset { 0 } else { in_idx };
        let mut data = vec![zero; 4];
        data[out_idx * 2 + in_idx] = scale;

        let projector = Tensor {
            data,
            shape: smallvec::smallvec![2, 2],
            legs: smallvec::smallvec![out_leg, in_leg],
        };
        let owner = self
            .tensors
            .iter()
            .position(|t| t.legs.contains(&in_leg))
            .expect("every output leg has an owner");
        self.tensors[owner] = contract(&self.tensors[owner], &projector);
        self.output_legs[qubit] = out_leg;
        self.is_unitary[qubit] = false;
    }

    fn append_matrix(&mut self, qubits: &[usize], data: Vec<Complex64>) {
        append_gate_tensor(
            &mut self.tensors,
            &mut self.output_legs,
            &mut self.next_leg,
            qubits,
            data,
            false,
        );
    }

    fn apply_reset(&mut self, qubit: usize) -> Result<()> {
        self.collapse_qubit(qubit, true)?;
        Ok(())
    }

    /// Leg ids the bra copy uses, one entry per live ket leg.
    ///
    /// Every leg is shifted clear of the ket id space by [`Self::next_leg`], so a
    /// bra tensor contracts only against other bra tensors. Callers then map back
    /// to the ket id the legs they want summed over: an output leg mapped to
    /// itself closes that qubit against its twin, which is a trace with identity.
    fn bra_leg_map(&self) -> Vec<LegId> {
        (0..self.next_leg).map(|leg| leg + self.next_leg).collect()
    }

    /// Double the backward light cone of open and non-unitary qubits against
    /// its conjugate.
    ///
    /// Unitary components with closed output legs cancel to identity and are
    /// skipped before doubling. Any closed leg held by a single kept tensor
    /// (either a non-unitary output leg or an internal leg whose downstream
    /// unitary gate was pruned) is closed directly against its bra twin.
    fn double_cone(&self, is_open: impl Fn(usize) -> bool) -> Vec<Tensor> {
        let mut needed = vec![false; self.next_leg];
        let mut open_legs = vec![false; self.next_leg];
        for (q, &leg) in self.output_legs.iter().enumerate() {
            if is_open(q) {
                open_legs[leg] = true;
                needed[leg] = true;
            } else if !self.is_unitary[q] {
                needed[leg] = true;
            }
        }

        let mut keep = vec![false; self.tensors.len()];
        let mut kept_count = 0usize;
        for (slot, tensor) in keep.iter_mut().zip(&self.tensors).rev() {
            if tensor.legs.iter().any(|&leg| needed[leg]) {
                *slot = true;
                kept_count += 1;
                for &leg in &tensor.legs {
                    needed[leg] = true;
                }
            }
        }
        if kept_count == 0 {
            return Vec::new();
        }

        let mut holders = vec![0u8; self.next_leg];
        for (tensor, &kept) in self.tensors.iter().zip(&keep) {
            if kept {
                for &leg in &tensor.legs {
                    holders[leg] += 1;
                }
            }
        }

        let mut bra_legs = self.bra_leg_map();
        for leg in 0..self.next_leg {
            if holders[leg] == 1 && !open_legs[leg] {
                bra_legs[leg] = leg;
            }
        }

        let mut network = Vec::with_capacity(kept_count * 2);
        for (tensor, &kept) in self.tensors.iter().zip(&keep) {
            if kept {
                network.push(tensor.clone());
                network.push(Tensor {
                    data: tensor.data.iter().map(Complex64::conj).collect(),
                    shape: tensor.shape.clone(),
                    legs: tensor.legs.iter().map(|&leg| bra_legs[leg]).collect(),
                });
            }
        }
        network
    }

    /// Build the network for the partial trace of `|psi><psi|` over every qubit
    /// outside `kept`, leaving each kept qubit's ket and bra indices open.
    ///
    /// A kept qubit's ket leg is its output leg and its bra leg is that id plus
    /// [`Self::next_leg`].
    fn double_for_partial_trace(&self, kept: &[usize]) -> Vec<Tensor> {
        let mut open = vec![false; self.num_qubits];
        for &q in kept {
            open[q] = true;
        }
        self.double_cone(|q| open[q])
    }

    /// Contract `tr_{q != qubit} |psi><psi|` to its four entries, replaying
    /// or filling `plan` when the caller holds a cache slot.
    fn marginal_1q(
        &self,
        qubit: usize,
        plan: Option<&mut Option<CachedPlan>>,
    ) -> Result<[[Complex64; 2]; 2]> {
        let mut network = self.double_for_partial_trace(&[qubit]);
        let ket_leg = self.output_legs[qubit];
        let bra_leg = ket_leg + self.next_leg;
        let operation = "reduced density matrix";
        let rho = match plan {
            Some(slot) => contract_cached(
                &mut network,
                slot,
                self.contraction_limits(),
                self.name(),
                operation,
            )?,
            None => greedy_contract(
                &mut network,
                self.contraction_limits(),
                self.name(),
                operation,
            )?,
        };
        self.book_truncation();

        let axis = |leg: LegId| {
            rho.legs
                .iter()
                .position(|&l| l == leg)
                .expect("partial trace leaves both open legs on the result")
        };
        let ket_stride = if axis(ket_leg) == 0 { 2 } else { 1 };
        let bra_stride = if axis(bra_leg) == 0 { 2 } else { 1 };

        Ok([
            [rho.data[0], rho.data[bra_stride]],
            [rho.data[ket_stride], rho.data[ket_stride + bra_stride]],
        ])
    }

    /// Contract `<psi|P|psi>` for one joint Pauli observable, unnormalized.
    ///
    /// Qubits the observable omits carry identity, and an identity factor is a
    /// leg closed directly against its twin rather than a tensor appended, which
    /// keeps `n - k` tensors out of the network for a weight-`k` observable.
    fn contract_pauli_sandwich(&self, axes: &[Option<PauliAxis>]) -> Result<f64> {
        let mut network = self.double_cone(|q| axes[q].is_some());
        if network.is_empty() {
            return Ok(1.0);
        }
        for (q, &axis) in axes.iter().enumerate() {
            if let Some(axis) = axis {
                let ket_leg = self.output_legs[q];
                network.push(pauli_tensor(Some(axis), ket_leg + self.next_leg, ket_leg));
            }
        }

        let result = greedy_contract(
            &mut network,
            self.contraction_limits(),
            self.name(),
            "pauli expectation",
        )?;
        self.book_truncation();
        debug_assert!(result.legs.is_empty(), "sandwich leaves every leg paired");
        Ok(result.data[0].re)
    }

    /// Contract the full network and return the amplitude vector in
    /// computational basis order.
    fn contract_to_statevector(&self) -> Result<Vec<Complex64>> {
        dense_statevector_len(self.name(), "contraction", self.num_qubits, &self.limits)?;
        if self.num_qubits == 0 {
            return Ok(vec![Complex64::new(1.0, 0.0)]);
        }

        let mut tensors = self.tensors.clone();
        let result = greedy_contract(
            &mut tensors,
            self.contraction_limits(),
            self.name(),
            "contraction",
        )?;
        self.book_truncation();
        self.close_query();

        // The result tensor's legs should be exactly the output_legs.
        // PRISM-Q convention: q[0] = LSB of state index. In row-major
        // tensor layout, the last axis is LSB. Use leg order
        // [q_{n-1}, q_{n-2}, ..., q_0], reversed.
        let target_order: Vec<LegId> = self.output_legs.iter().rev().copied().collect();
        let perm: SmallVec<[usize; 6]> = target_order
            .iter()
            .map(|target_leg| {
                result
                    .legs
                    .iter()
                    .position(|l| l == target_leg)
                    .expect("greedy_contract consumes every tensor, so output legs survive")
            })
            .collect();

        let needs_perm = perm.iter().enumerate().any(|(i, &p)| i != p);
        let ordered = if needs_perm {
            transpose(&result, &perm)
        } else {
            result
        };

        Ok(ordered.data)
    }

    fn dispatch_gate(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        let num_qubits = self.num_qubits;
        for_each_gate_tensor(gate, targets, num_qubits, |op| {
            match op {
                GateTensorOp::OneQ(q, mat) => self.append_matrix(&[q], mat.as_flattened().to_vec()),
                GateTensorOp::TwoQ(q0, q1, mat) => {
                    self.append_matrix(&[q0, q1], mat.as_flattened().to_vec())
                }
                GateTensorOp::NQ(qubits, matrix) => self.append_matrix(qubits, matrix),
            }
            Ok(())
        })?;
        self.record_gate_unitarity(gate, targets);
        Ok(())
    }

    fn record_gate_unitarity(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::Fused(mat) => {
                if !is_unitary_dense(mat.as_flattened(), 2) {
                    self.is_unitary[targets[0]] = false;
                }
            }
            Gate::Fused2q(mat) => {
                if !is_unitary_dense(mat.as_flattened(), 4) {
                    self.is_unitary[targets[0]] = false;
                    self.is_unitary[targets[1]] = false;
                }
            }
            Gate::Cu(mat) => {
                if !is_unitary_dense(mat.as_flattened(), 2) {
                    self.is_unitary[targets[0]] = false;
                    self.is_unitary[targets[1]] = false;
                }
            }
            Gate::Mcu(data) => {
                if !is_unitary_dense(data.mat().as_flattened(), 2) {
                    for &q in targets {
                        self.is_unitary[q] = false;
                    }
                }
            }
            Gate::MultiFused(data) => {
                for &(q, ref mat) in data.gates() {
                    if !is_unitary_dense(mat.as_flattened(), 2) {
                        self.is_unitary[q] = false;
                    }
                }
            }
            Gate::Multi2q(data) => {
                for &(q0, q1, ref mat) in data.gates() {
                    if !is_unitary_dense(mat.as_flattened(), 4) {
                        self.is_unitary[q0] = false;
                        self.is_unitary[q1] = false;
                    }
                }
            }
            Gate::BatchPhase(data) => {
                for &(q1, phase) in data.phases() {
                    if !is_unitary_phase(phase) {
                        self.is_unitary[targets[0]] = false;
                        self.is_unitary[q1] = false;
                    }
                }
            }
            Gate::DiagonalBatch(data) => {
                for entry in data.entries() {
                    match *entry {
                        DiagEntry::Phase1q { qubit, d0, d1 } => {
                            if !(is_unitary_phase(d0) && is_unitary_phase(d1)) {
                                self.is_unitary[qubit] = false;
                            }
                        }
                        DiagEntry::Phase2q { q0, q1, phase } => {
                            if !is_unitary_phase(phase) {
                                self.is_unitary[q0] = false;
                                self.is_unitary[q1] = false;
                            }
                        }
                        DiagEntry::Parity2q {
                            q0, q1, same, diff, ..
                        } => {
                            if !(is_unitary_phase(same) && is_unitary_phase(diff)) {
                                self.is_unitary[q0] = false;
                                self.is_unitary[q1] = false;
                            }
                        }
                    }
                }
            }
            Gate::Unitary(data) => {
                let dim = 1usize << data.num_qubits();
                if !is_unitary_dense(data.matrix(), dim) {
                    for &q in targets {
                        self.is_unitary[q] = false;
                    }
                }
            }
            _ => {}
        }
    }
}

impl crate::backend::sealed::Sealed for TensorNetworkBackend {
    fn limits(&self) -> crate::limits::ResolvedLimits {
        self.limits
    }
}

impl Backend for TensorNetworkBackend {
    fn name(&self) -> &'static str {
        "tensornetwork"
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn resolved(&self) -> crate::sim::ResolvedBackend {
        crate::sim::ResolvedBackend::TensorNetwork
    }

    /// `Exact` without a tolerance, whatever the contraction did, since every
    /// other route to the cap preserves the value. A tolerance reports
    /// `Approximate` whether or not this run cut anything, since the route
    /// could have.
    ///
    /// The bound is 1 minus [`TensorNetworkBackend::truncation_discarded`]: the
    /// relative discarded weights summed over the cuts of the worst single
    /// query, one shot, one observable or one export, rather than over every
    /// query the run answered. It is a first-order truncation estimate, not a
    /// certificate. Errors compound across cuts, and summing the weights
    /// understates the compounded error, since the strict bound on the
    /// infidelity is the square of the summed square roots. The two agree only
    /// when a single cut truncates. A doubled contraction, which
    /// `pauli_expectations` and both reduced density matrix methods run, cuts
    /// in the doubled space, so what the bound describes there is the
    /// quadratic form rather than the state. A truncated
    /// `reduced_density_matrix` is made Hermitian again by averaging it with
    /// its conjugate transpose; positivity is not restored.
    fn exactness(&self) -> crate::sim::Exactness {
        match self.tolerance {
            Some(tolerance) if tolerance > 0.0 => crate::sim::Exactness::Approximate {
                fidelity_lower_bound: Some((1.0 - self.truncation_discarded()).max(0.0)),
            },
            _ => crate::sim::Exactness::Exact,
        }
    }

    fn init(&mut self, num_qubits: usize, num_classical_bits: usize) -> Result<()> {
        self.num_qubits = num_qubits;
        self.tensors = Vec::new();
        self.next_leg = 0;
        self.is_unitary = vec![true; num_qubits];
        self.run_discarded = 0.0;
        self.query_discarded.set(0.0);
        self.worst_query_discarded.set(0.0);
        crate::backend::init_classical_bits(&mut self.classical_bits, num_classical_bits);

        self.output_legs = Vec::with_capacity(num_qubits);
        for _ in 0..num_qubits {
            let leg = self.fresh_leg();
            self.output_legs.push(leg);
            self.tensors.push(Tensor {
                data: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                shape: SmallVec::from_buf_and_len([2, 0, 0, 0, 0, 0], 1),
                legs: SmallVec::from_buf_and_len([leg, 0, 0, 0, 0, 0], 1),
            });
        }

        Ok(())
    }

    fn apply(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Gate { gate, targets } => self.dispatch_gate(gate, targets)?,
            Instruction::Measure {
                qubit,
                classical_bit,
            } => {
                let outcome = self.collapse_qubit(*qubit, false)?;
                self.classical_bits[*classical_bit] = outcome;
            }
            Instruction::Reset { qubit } => {
                self.apply_reset(*qubit)?;
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(crate::backend::save_not_applied("TensorNetwork", label));
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } => {
                if condition.evaluate(&self.classical_bits) {
                    self.dispatch_gate(gate, targets)?;
                }
            }
            Instruction::Region(region) => self.apply_region(region)?,
        }
        Ok(())
    }

    fn reset(&mut self, qubit: usize) -> Result<()> {
        self.apply_reset(qubit)
    }

    fn apply_1q_matrix(&mut self, qubit: usize, matrix: &[[Complex64; 2]; 2]) -> Result<()> {
        self.append_matrix(&[qubit], matrix.as_flattened().to_vec());
        self.is_unitary[qubit] = false;
        Ok(())
    }

    fn classical_results(&self) -> &[bool] {
        &self.classical_bits
    }

    fn set_classical_bit(&mut self, bit: usize, value: bool) -> Result<()> {
        self.classical_bits[bit] = value;
        Ok(())
    }

    fn probabilities(&self) -> Result<Vec<f64>> {
        tensor_probability_len(self.name(), self.num_qubits, &self.limits)?;
        let amplitudes = self.contract_to_statevector()?;
        let mut probs = vec![0.0_f64; amplitudes.len()];
        #[cfg(feature = "parallel")]
        if amplitudes.len() >= MIN_PAR_ELEMS {
            amplitudes
                .par_chunks(MIN_PAR_ELEMS)
                .zip(probs.par_chunks_mut(MIN_PAR_ELEMS))
                .for_each(|(src, dst)| simd::norm_sqr_to_slice(src, dst));
            return Ok(probs);
        }
        simd::norm_sqr_to_slice(&amplitudes, &mut probs);
        Ok(probs)
    }

    fn supports_native_sampling(&self) -> bool {
        true
    }

    /// Draw shots from the dense distribution below the ceiling and from the
    /// qubit-by-qubit conditional sweep past it.
    ///
    /// One full-distribution contraction plus a draw per shot undercuts the
    /// per-shot sweep by a measured 66x at 16 qubits and 32 shots on the
    /// chain shape, so the dense arm answers wherever `probabilities()` can;
    /// the sweep is the route that exists past that ceiling.
    fn sample_basis_states(&mut self, num_shots: usize, seed: u64) -> Result<BasisSamples> {
        let n = self.num_qubits;
        if n == 0 || num_shots == 0 {
            return Ok(BasisSamples::new(num_shots, n));
        }

        if tensor_probability_len(self.name(), n, &self.limits).is_ok() {
            use rand::RngExt;

            let mut samples = BasisSamples::new(num_shots, n);
            let cdf = crate::sim::shots::build_cdf(&self.probabilities()?);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            for shot in 0..num_shots {
                let r = rng.random::<f64>();
                samples.set_index(shot, crate::sim::shots::sample_from_cdf(&cdf, r));
            }
            return Ok(samples);
        }

        self.sample_native(num_shots, seed)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Contracts the network against its conjugate with every other qubit's
    /// output leg closed, so the cost is set by the doubled network's treewidth
    /// rather than by `2^n`.
    ///
    /// # Panics
    ///
    /// If `qubit` is outside the register.
    fn reduced_density_matrix_1q(&self, qubit: usize) -> Result<[[Complex64; 2]; 2]> {
        let rho = self.marginal_1q(qubit, None);
        self.close_query();
        rho
    }

    /// One doubled contraction with the subsystem's ket and bra legs open, so
    /// the cost follows the doubled network's treewidth, widened by those open
    /// legs, rather than `2^n`.
    fn reduced_density_matrix(&mut self, subsystem: &[usize]) -> Result<Vec<Complex64>> {
        crate::backend::schmidt::validate_qubit_set(subsystem, self.num_qubits)?;
        let k = subsystem.len();
        let dim =
            crate::backend::reduced_density::reduced_density_side(self.name(), k, &self.limits)?;

        let mut network = self.double_for_partial_trace(subsystem);
        let rho = greedy_contract(
            &mut network,
            self.contraction_limits(),
            self.name(),
            "reduced density matrix",
        )?;
        self.book_truncation();
        self.close_query();
        debug_assert_eq!(rho.rank(), 2 * k);

        // Row major with the last axis lowest, so `subsystem[0]` takes the last
        // ket axis and the last bra axis: ket legs carry the row index.
        let kets = subsystem.iter().rev().map(|&q| self.output_legs[q]);
        let bras = kets.clone().map(|leg| leg + self.next_leg);
        let perm: SmallVec<[usize; 6]> = kets
            .chain(bras)
            .map(|target| {
                rho.legs
                    .iter()
                    .position(|&leg| leg == target)
                    .expect("partial trace leaves every kept leg open on the result")
            })
            .collect();
        let mut data = if perm.iter().enumerate().any(|(i, &p)| i != p) {
            transpose(&rho, &perm).data
        } else {
            rho.data
        };

        if self.tolerance.is_some_and(|tolerance| tolerance > 0.0) {
            for t in 0..dim {
                data[t * dim + t].im = 0.0;
                for tp in t + 1..dim {
                    let mean = 0.5 * (data[t * dim + tp] + data[tp * dim + t].conj());
                    data[t * dim + tp] = mean;
                    data[tp * dim + t] = mean.conj();
                }
            }
        }
        crate::backend::reduced_density::normalize_trace(&mut data, dim);
        Ok(data)
    }

    fn supports_two_qubit_kraus(&self) -> bool {
        true
    }

    fn supports_pauli_expectation(&self) -> bool {
        true
    }

    /// Contracts `<psi|P|psi>` directly, so no `2^n` vector is built and the
    /// dense query ceiling does not apply.
    ///
    /// # Errors
    ///
    /// [`PrismError::InvalidQubit`](crate::PrismError::InvalidQubit) for a factor
    /// outside the register, and
    /// [`PrismError::InvalidParameter`](crate::PrismError::InvalidParameter) for two
    /// factors on one qubit.
    fn pauli_expectations(&self, observables: &[Vec<PauliTerm>]) -> Result<Vec<f64>> {
        let mut axes: Vec<Option<PauliAxis>> = vec![None; self.num_qubits];
        let mut expectations = Vec::with_capacity(observables.len());
        let (norm_sq, norm_discarded) = if self.is_unitary.iter().all(|&u| u) {
            (1.0, 0.0)
        } else {
            let norm_sq = self.contract_pauli_sandwich(&axes)?;
            (norm_sq, self.query_discarded.replace(0.0))
        };

        for observable in observables {
            observable_axes(observable, &mut axes, "tensor-network")?;
            expectations.push(self.contract_pauli_sandwich(&axes)? / norm_sq);
            self.query_discarded
                .set(self.query_discarded.get() + norm_discarded);
            self.close_query();
        }

        Ok(expectations)
    }

    fn supports_fused_gates(&self) -> bool {
        true
    }

    fn export_statevector(&self) -> Result<Vec<Complex64>> {
        self.contract_to_statevector()
    }
}

#[cfg(test)]
mod tests;
