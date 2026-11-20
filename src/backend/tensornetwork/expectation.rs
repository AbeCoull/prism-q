//! Scalar expectation networks and the gate tensors both networks append.

use std::borrow::Cow;

use num_complex::Complex64;
use smallvec::SmallVec;

use crate::backend::tensor_peak_error;
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::{Gate, mcu_matrix};
#[cfg(feature = "bench-internal")]
use crate::limits::LimitSource;
use crate::limits::ResolvedLimits;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

use super::LegId;
use super::plan::{ContractionPlan, TensorMeta, plan_pairs, plan_with_restarts};
use super::slice::{
    ContractionLimits, LAST_DISCARDED, LAST_SLICE_COUNT, SlicePlan, choose_slices, contract_slices,
    contract_within,
};
use super::tensor::Tensor;

/// Rank-`2k` tensor of a `k`-qubit gate on legs `[out.., in..]`, conjugated for
/// the bra copy when `conjugate` is set.
///
/// A row-major `2^k` by `2^k` matrix is already the tensor's data in that leg
/// order, the first target the most significant bit on each side.
fn gate_tensor(
    mut data: Vec<Complex64>,
    out_legs: &[LegId],
    in_legs: &[LegId],
    conjugate: bool,
) -> Tensor {
    if conjugate {
        data.iter_mut().for_each(|value| *value = value.conj());
    }
    Tensor {
        data,
        shape: SmallVec::from_elem(2, out_legs.len() + in_legs.len()),
        legs: out_legs.iter().chain(in_legs).copied().collect(),
    }
}

/// Push the gate tensor for `data` on `qubits`, reading each qubit's input leg
/// from `legs` and replacing it with an output leg minted from `next_leg`.
pub(super) fn append_gate_tensor(
    tensors: &mut Vec<Tensor>,
    legs: &mut [LegId],
    next_leg: &mut LegId,
    qubits: &[usize],
    data: Vec<Complex64>,
    conjugate: bool,
) {
    let in_legs: SmallVec<[LegId; 6]> = qubits.iter().map(|&q| legs[q]).collect();
    let out_legs: SmallVec<[LegId; 6]> = (*next_leg..*next_leg + qubits.len()).collect();
    *next_leg += qubits.len();
    tensors.push(gate_tensor(data, &out_legs, &in_legs, conjugate));
    for (&q, &leg) in qubits.iter().zip(&out_legs) {
        legs[q] = leg;
    }
}

/// The Pauli on `axis` as a tensor on legs `[bra, ket]`, identity for `None`.
pub(super) fn pauli_tensor(axis: Option<PauliAxis>, bra_leg: LegId, ket_leg: LegId) -> Tensor {
    let gate = match axis {
        None => Gate::Id,
        Some(PauliAxis::X) => Gate::X,
        Some(PauliAxis::Y) => Gate::Y,
        Some(PauliAxis::Z) => Gate::Z,
    };
    let data = gate.matrix_2x2().as_flattened().to_vec();
    gate_tensor(data, &[bra_leg], &[ket_leg], false)
}

/// Reset `axes`, one slot per qubit, and fill it from a joint Pauli observable.
///
/// The checks match [`crate::sim::validate_observable`]; `owner` leads the
/// duplicate-factor message.
pub(super) fn observable_axes(
    terms: &[PauliTerm],
    axes: &mut [Option<PauliAxis>],
    owner: &str,
) -> Result<()> {
    axes.fill(None);
    for term in terms {
        if term.qubit >= axes.len() {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: axes.len(),
            });
        }
        if axes[term.qubit].is_some() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "{owner} observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        axes[term.qubit] = Some(term.axis);
    }
    Ok(())
}

pub(super) struct ScalarExpectationNetwork {
    pub(super) tensors: Vec<Tensor>,
    pub(super) ket_legs: Vec<LegId>,
    pub(super) bra_legs: Vec<LegId>,
    pub(super) next_leg: LegId,
}

impl ScalarExpectationNetwork {
    /// Network over `num_qubits` with a `|0>` ket and bra only for the qubits
    /// `in_cone` marks; the others never receive a tensor.
    pub(super) fn new(num_qubits: usize, in_cone: &[bool]) -> Self {
        let mut network = Self {
            tensors: Vec::new(),
            ket_legs: vec![LegId::MAX; num_qubits],
            bra_legs: vec![LegId::MAX; num_qubits],
            next_leg: 0,
        };
        let zero_state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
        for q in (0..num_qubits).filter(|&q| in_cone[q]) {
            let ket_leg = network.fresh_leg();
            network.ket_legs[q] = ket_leg;
            network.tensors.push(Tensor {
                data: zero_state.clone(),
                shape: smallvec::smallvec![2],
                legs: smallvec::smallvec![ket_leg],
            });

            let bra_leg = network.fresh_leg();
            network.bra_legs[q] = bra_leg;
            network.tensors.push(Tensor {
                data: zero_state.clone(),
                shape: smallvec::smallvec![2],
                legs: smallvec::smallvec![bra_leg],
            });
        }
        network
    }

    fn fresh_leg(&mut self) -> LegId {
        let leg = self.next_leg;
        self.next_leg += 1;
        leg
    }

    /// Append the gate tensor for `data` on `qubits` to the ket side, or to the
    /// conjugated bra side when `conjugate` is set.
    fn append_matrix(&mut self, qubits: &[usize], data: Vec<Complex64>, conjugate: bool) {
        let legs = if conjugate {
            &mut self.bra_legs
        } else {
            &mut self.ket_legs
        };
        append_gate_tensor(
            &mut self.tensors,
            legs,
            &mut self.next_leg,
            qubits,
            data,
            conjugate,
        );
    }

    /// `<0| U^dag P U |0>` over the backward light cone of `terms`, rejecting any
    /// instruction that is not a unitary gate or a barrier. A gate whose qubits all
    /// lie outside the cone cancels against its adjoint, and a qubit no kept gate
    /// touches closes to `<0|0> = 1`, so neither enters the network. Gates are
    /// screened by the qubits they touch before any matrix is built, so the build
    /// costs the cone rather than the circuit.
    pub(super) fn for_observable(circuit: &Circuit, terms: &[PauliTerm]) -> Result<Self> {
        let num_qubits = circuit.num_qubits;
        let mut axes = vec![None; num_qubits];
        observable_axes(terms, &mut axes, "tensor-network scalar")?;

        for instruction in &circuit.instructions {
            match instruction {
                Instruction::Gate { .. } | Instruction::Barrier { .. } => {}
                Instruction::Save { label, .. } => {
                    return Err(crate::backend::save_not_applied("TensorNetwork", label));
                }
                Instruction::Measure { .. }
                | Instruction::Reset { .. }
                | Instruction::Conditional { .. }
                | Instruction::Region(_) => {
                    return Err(PrismError::BackendUnsupported {
                        backend: "tensor_network_scalar".to_string(),
                        operation: format!("non-unitary instruction {instruction:?}"),
                    });
                }
            }
        }

        let mut in_cone: Vec<bool> = axes.iter().map(Option::is_some).collect();
        let mut reached = vec![false; circuit.instructions.len()];
        for (hit, instruction) in reached.iter_mut().zip(&circuit.instructions).rev() {
            let Instruction::Gate { gate, targets } = instruction else {
                continue;
            };
            let mut touches = false;
            for_each_gate_tensor_qubit(gate, targets, |q| {
                if q >= num_qubits {
                    return Err(PrismError::InvalidQubit {
                        index: q,
                        register_size: num_qubits,
                    });
                }
                touches |= in_cone[q];
                Ok(())
            })?;
            if touches {
                *hit = true;
                for_each_gate_tensor_qubit(gate, targets, |q| {
                    in_cone[q] = true;
                    Ok(())
                })?;
            }
        }

        let mut ops = Vec::new();
        for (instruction, reached) in circuit.instructions.iter().zip(reached) {
            if let (true, Instruction::Gate { gate, targets }) = (reached, instruction) {
                for_each_gate_tensor(gate, targets, num_qubits, |op| {
                    ops.push(op);
                    Ok(())
                })?;
            }
        }

        let mut in_cone: Vec<bool> = axes.iter().map(Option::is_some).collect();
        let mut keep = vec![false; ops.len()];
        for (kept, op) in keep.iter_mut().zip(&ops).rev() {
            let qubits = op.qubits();
            if qubits.iter().any(|&q| in_cone[q]) {
                *kept = true;
                for q in qubits {
                    in_cone[q] = true;
                }
            }
        }

        let mut network = Self::new(num_qubits, &in_cone);
        for (op, kept) in ops.into_iter().zip(keep) {
            if kept {
                network.append_op(op);
            }
        }
        for (qubit, axis) in axes.into_iter().enumerate() {
            if in_cone[qubit] {
                network.tensors.push(pauli_tensor(
                    axis,
                    network.bra_legs[qubit],
                    network.ket_legs[qubit],
                ));
            }
        }
        Ok(network)
    }

    pub(super) fn append_op(&mut self, op: GateTensorOp<'_>) {
        match op {
            GateTensorOp::OneQ(q, mat) => {
                self.append_matrix(&[q], mat.as_flattened().to_vec(), false);
                self.append_matrix(&[q], mat.as_flattened().to_vec(), true);
            }
            GateTensorOp::TwoQ(q0, q1, mat) => {
                self.append_matrix(&[q0, q1], mat.as_flattened().to_vec(), false);
                self.append_matrix(&[q0, q1], mat.as_flattened().to_vec(), true);
            }
            GateTensorOp::NQ(qubits, matrix) => {
                self.append_matrix(qubits, matrix.clone(), false);
                self.append_matrix(qubits, matrix, true);
            }
        }
    }

    fn contract_on(self, plan: &ContractionPlan, limits: ContractionLimits) -> Result<f64> {
        self.contract_prepared(plan, None, limits)
    }

    fn contract_prepared(
        mut self,
        plan: &ContractionPlan,
        slice: Option<&SlicePlan>,
        limits: ContractionLimits,
    ) -> Result<f64> {
        if self.tensors.is_empty() {
            return Ok(1.0);
        }
        let result = if let Some(slice) = slice {
            LAST_DISCARDED.with(|weight| weight.set(0.0));
            LAST_SLICE_COUNT.with(|count| count.set(slice.count));
            contract_slices(&self.tensors, plan, slice, limits.peak_cap)
        } else {
            contract_within(
                &mut self.tensors,
                plan,
                limits,
                "tensor_network_scalar",
                "scalar expectation",
            )?
        };
        if result.data.len() != 1 || !result.legs.is_empty() {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "tensor-network scalar contraction left {} amplitudes and {} open legs",
                    result.data.len(),
                    result.legs.len()
                ),
            });
        }
        Ok(result.data[0].re)
    }
}

/// Plan total, in elements, from which the fused circuit's network competes
/// with the unfused one.
///
/// Fusion trades tensor count for tensor size, and which side contracts faster
/// follows the plan total: on a 4-row grid with `Z` on every qubit the fused
/// plan totals 573k elements against 3.1M at 16 qubits and runs 3x faster,
/// while at 20 qubits it totals 5.6M against 4.0M and runs 1.4x slower.
/// Fusing and planning the second network costs about 0.3 ms, under 5% of a
/// contraction this size; a local observable's cone stays far below it.
pub(super) const FUSED_PLAN_TOTAL: usize = 1 << 18;

/// Contract `<0| U^dag P U |0>` without materializing a full statevector.
pub(crate) fn expectation_zero_state(
    circuit: &Circuit,
    pauli_terms: &[PauliTerm],
    limits: &ResolvedLimits,
) -> Result<f64> {
    contract_scalar(circuit, pauli_terms, ContractionLimits::new(limits, None))
}

/// Contract the observable's network, or the fused circuit's when its plan is
/// lower by peak then total.
fn contract_scalar(
    circuit: &Circuit,
    pauli_terms: &[PauliTerm],
    limits: ContractionLimits,
) -> Result<f64> {
    let (network, plan) = plan_scalar(circuit, pauli_terms)?;
    network.contract_on(&plan, limits)
}

fn plan_scalar(
    circuit: &Circuit,
    pauli_terms: &[PauliTerm],
) -> Result<(ScalarExpectationNetwork, ContractionPlan)> {
    let network = ScalarExpectationNetwork::for_observable(circuit, pauli_terms)?;
    if network.tensors.is_empty() {
        return Ok((
            network,
            ContractionPlan {
                pairs: Vec::new(),
                peak: 1,
                total: 0,
            },
        ));
    }
    let plan = plan_with_restarts(&network.tensors);
    if plan.total >= FUSED_PLAN_TOTAL
        && let Cow::Owned(fused) = crate::circuit::fusion::fuse_circuit(circuit, true)
    {
        let fused_network = ScalarExpectationNetwork::for_observable(&fused, pauli_terms)?;
        let fused_plan = plan_with_restarts(&fused_network.tensors);
        if (fused_plan.peak, fused_plan.total) < (plan.peak, plan.total) {
            return Ok((fused_network, fused_plan));
        }
    }
    Ok((network, plan))
}

/// Largest greedy-tree intermediate, in elements, under which the `Auto`
/// expectation route below the statevector cap takes the scalar path. Measured
/// over 24 circuits: every family the scalar path beat the statevector on stayed
/// under it for every observable, every family it lost to crossed it, and the
/// bounded pass that decides costs 1 to 22 ms.
pub(crate) const AUTO_EXPECTATION_PEAK_BOUND: usize = 1 << 12;

/// Scalar networks with carried contraction trees and optional slice plans.
pub(crate) struct BoundedExpectations {
    planned: Vec<(ScalarExpectationNetwork, ContractionPlan, Option<SlicePlan>)>,
    failure: Option<PrismError>,
}

impl BoundedExpectations {
    pub(crate) fn with_limits(mut self, limits: &ResolvedLimits) -> Self {
        let limits = ContractionLimits::new(limits, None);
        for (network, plan, slice) in &mut self.planned {
            if plan.peak > limits.peak_cap {
                let metas: Vec<_> = network.tensors.iter().map(TensorMeta::of).collect();
                *slice = choose_slices(&metas, plan, limits);
                if slice.is_none() {
                    self.failure = Some(tensor_peak_error(
                        "tensor_network_scalar",
                        "scalar expectation",
                        plan.peak,
                        limits.peak_cap,
                        limits.peak_source,
                    ));
                    break;
                }
            }
        }
        self
    }

    pub(crate) fn failure(&self) -> Option<&PrismError> {
        self.failure.as_ref()
    }

    pub(crate) fn contract(self, limits: &ResolvedLimits) -> Result<Vec<f64>> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let limits = ContractionLimits::new(limits, None);
        self.planned
            .into_iter()
            .map(|(network, plan, slice)| network.contract_prepared(&plan, slice.as_ref(), limits))
            .collect()
    }
}

/// Plan `<0| U^dag P U |0>` for every observable with no intermediate over
/// `peak_bound` elements and at most `budget` elements summed over every tree.
/// `None` when a tree crosses either, or the circuit holds an instruction the
/// network cannot append, so the caller falls back to another route.
pub(crate) fn plan_bounded_expectations(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    peak_bound: usize,
    budget: usize,
) -> Option<Result<BoundedExpectations>> {
    let circuit = crate::circuit::expand_qft_blocks(circuit);
    let mut planned = Vec::with_capacity(observables.len());
    let mut spent = 0usize;
    for observable in observables {
        let mut axes = vec![None; circuit.num_qubits];
        if let Err(e) = observable_axes(observable, &mut axes, "tensor-network scalar") {
            return Some(Err(e));
        }
        let network = ScalarExpectationNetwork::for_observable(&circuit, observable).ok()?;
        let slots: Vec<TensorMeta> = network.tensors.iter().map(TensorMeta::of).collect();
        let plan = plan_pairs(slots, None, peak_bound, budget - spent)?;
        spent += plan.total;
        planned.push((network, plan, None));
    }
    Some(Ok(BoundedExpectations {
        planned,
        failure: None,
    }))
}

/// Gates on two or more qubits in the inverse light cones of `observables`,
/// summed over the observables; stops counting once the sum passes `limit`.
/// Observable qubits must lie in the register.
pub(crate) fn cone_entangling_gates(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    limit: usize,
) -> usize {
    let mut in_cone = vec![false; circuit.num_qubits];
    let mut count = 0;
    for observable in observables {
        in_cone.fill(false);
        for term in observable {
            in_cone[term.qubit] = true;
        }
        for instruction in circuit.instructions.iter().rev() {
            let Instruction::Gate { gate, targets } = instruction else {
                continue;
            };
            let mut touches = false;
            let mut width = 0;
            for_each_gate_tensor_qubit(gate, targets, |q| {
                touches |= in_cone[q];
                width += 1;
                Ok(())
            })
            .expect("the visitor never fails");
            if !touches {
                continue;
            }
            for_each_gate_tensor_qubit(gate, targets, |q| {
                in_cone[q] = true;
                Ok(())
            })
            .expect("the visitor never fails");
            if width > 1 {
                count += 1;
                if count > limit {
                    return count;
                }
            }
        }
    }
    count
}

pub(crate) fn plan_expectations(
    circuit: &Circuit,
    observables: &[Vec<PauliTerm>],
    limits: &ResolvedLimits,
) -> Result<BoundedExpectations> {
    let circuit = crate::circuit::expand_qft_blocks(circuit);
    let planned = observables
        .iter()
        .map(|observable| {
            let (network, plan) = plan_scalar(&circuit, observable)?;
            Ok((network, plan, None))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BoundedExpectations {
        planned,
        failure: None,
    }
    .with_limits(limits))
}

/// Bench-visible wrapper over `expectation_zero_state`; not stable API.
#[cfg(feature = "bench-internal")]
pub fn scalar_expectation(circuit: &Circuit, pauli_terms: &[PauliTerm]) -> Result<f64> {
    expectation_zero_state(circuit, pauli_terms, &ResolvedLimits::process_default())
}

/// [`scalar_expectation`] with the peak cap, slice budget and truncation
/// tolerance supplied rather than read from the environment, so one process
/// can hold rows on both sides of the ceiling; not stable API.
#[cfg(feature = "bench-internal")]
pub fn scalar_expectation_capped(
    circuit: &Circuit,
    pauli_terms: &[PauliTerm],
    peak_cap: usize,
    slice_budget: usize,
    tolerance: Option<f64>,
) -> Result<f64> {
    contract_scalar(
        circuit,
        pauli_terms,
        ContractionLimits {
            peak_cap,
            peak_source: LimitSource::Request,
            slice_budget,
            tolerance,
        },
    )
}

/// Elementary tensor operation a gate decomposes into when appended to a
/// tensor network. `NQ` carries the full row-major `2^k × 2^k` matrix for
/// multi-controlled unitaries.
pub(super) enum GateTensorOp<'a> {
    OneQ(usize, [[Complex64; 2]; 2]),
    TwoQ(usize, usize, [[Complex64; 4]; 4]),
    NQ(&'a [usize], Vec<Complex64>),
}

impl GateTensorOp<'_> {
    pub(super) fn qubits(&self) -> SmallVec<[usize; 4]> {
        match self {
            GateTensorOp::OneQ(q, _) => smallvec::smallvec![*q],
            GateTensorOp::TwoQ(q0, q1, _) => smallvec::smallvec![*q0, *q1],
            GateTensorOp::NQ(qubits, _) => SmallVec::from_slice(qubits),
        }
    }
}

/// Call `visit` with every qubit an operation of [`for_each_gate_tensor`] on
/// `gate` touches, without building a matrix. `BatchPhase` lists only its control
/// in `targets`, so its phase targets are visited from the data.
fn for_each_gate_tensor_qubit<F>(gate: &Gate, targets: &[usize], mut visit: F) -> Result<()>
where
    F: FnMut(usize) -> Result<()>,
{
    for &q in targets {
        visit(q)?;
    }
    if let Gate::BatchPhase(data) = gate {
        for &(q, _) in &data.phases {
            visit(q)?;
        }
    }
    Ok(())
}

/// Decompose `gate` into the elementary 1q/2q/nq matrix operations a tensor
/// network applies, invoking `emit` once per operation in application order.
///
/// Shared by the deferred-contraction backend
/// ([`TensorNetworkBackend`](super::TensorNetworkBackend), which appends ket tensors
/// only) and the scalar-expectation network
/// ([`ScalarExpectationNetwork`], which appends a conjugated ket+bra pair). The
/// two differ only in their sink, so routing both through this keeps their gate
/// coverage and matrices identical.
pub(super) fn for_each_gate_tensor<'a, F>(
    gate: &Gate,
    targets: &'a [usize],
    num_qubits: usize,
    mut emit: F,
) -> Result<()>
where
    F: FnMut(GateTensorOp<'a>) -> Result<()>,
{
    let check_qubit = |q: usize| -> Result<()> {
        if q >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: q,
                register_size: num_qubits,
            });
        }
        Ok(())
    };
    let check_arity = |expected: usize| -> Result<()> {
        if targets.len() != expected {
            return Err(PrismError::GateArity {
                gate: gate.name().to_string(),
                expected,
                got: targets.len(),
            });
        }
        for &t in targets {
            check_qubit(t)?;
        }
        Ok(())
    };

    match gate {
        Gate::Rzz(_) | Gate::Cx | Gate::Cz | Gate::Swap | Gate::Cu(_) | Gate::Fused2q(_) => {
            check_arity(2)?;
            emit(GateTensorOp::TwoQ(
                targets[0],
                targets[1],
                gate.matrix_4x4(),
            ))
        }
        Gate::Mcu(data) => {
            check_arity(data.num_controls as usize + 1)?;
            let full = mcu_matrix(data.num_controls as usize, &data.mat);
            emit(GateTensorOp::NQ(targets, full))
        }
        Gate::Unitary(data) => {
            check_arity(data.num_qubits())?;
            emit(GateTensorOp::NQ(targets, data.matrix().to_vec()))
        }
        Gate::BatchPhase(data) => {
            if targets.is_empty() {
                return Err(PrismError::GateArity {
                    gate: gate.name().to_string(),
                    expected: 1,
                    got: 0,
                });
            }
            check_qubit(targets[0])?;
            let one = Complex64::new(1.0, 0.0);
            let zero = Complex64::new(0.0, 0.0);
            for &(target_qubit, phase) in &data.phases {
                let mat = [
                    [one, zero, zero, zero],
                    [zero, one, zero, zero],
                    [zero, zero, one, zero],
                    [zero, zero, zero, phase],
                ];
                emit(GateTensorOp::TwoQ(targets[0], target_qubit, mat))?;
            }
            Ok(())
        }
        Gate::BatchRzz(data) => {
            for &(q0, q1, theta) in &data.edges {
                emit(GateTensorOp::TwoQ(q0, q1, Gate::Rzz(theta).matrix_4x4()))?;
            }
            Ok(())
        }
        Gate::DiagonalBatch(data) => {
            for entry in &data.entries {
                if let Some((q, mat)) = entry.as_1q_matrix() {
                    emit(GateTensorOp::OneQ(q, mat))?;
                } else if let Some((q0, q1, mat)) = entry.as_2q_matrix() {
                    emit(GateTensorOp::TwoQ(q0, q1, mat))?;
                }
            }
            Ok(())
        }
        Gate::MultiFused(data) => {
            for &(target, ref mat) in &data.gates {
                emit(GateTensorOp::OneQ(target, *mat))?;
            }
            Ok(())
        }
        Gate::Multi2q(data) => {
            for &(q0, q1, ref mat) in &data.gates {
                emit(GateTensorOp::TwoQ(q0, q1, *mat))?;
            }
            Ok(())
        }
        Gate::QftBlock { .. } => Err(PrismError::BackendUnsupported {
            backend: "tensor_network".to_string(),
            operation: "QFT block scalar contraction without prior expansion".to_string(),
        }),
        _ => {
            check_arity(1)?;
            emit(GateTensorOp::OneQ(targets[0], gate.matrix_2x2()))
        }
    }
}
