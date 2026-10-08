//! Gradients of expectation values, by the adjoint method and by parameter
//! shift.
//!
//! Both compute `⟨H⟩ = ⟨0|U†HU|0⟩` and `d⟨H⟩/dθ` for a Hermitian
//! `H = Σ c_k P_k`. The adjoint method ([`run_expectation_gradient`]) is exact
//! in one pair of statevectors and costs one circuit evaluation regardless of
//! the parameter count, but runs only on the statevector backend. Parameter
//! shift ([`run_expectation_gradient_shift`]) costs two evaluations per
//! trainable gate and reaches any backend with a native observable path,
//! including widths past the statevector cap.
//!
//! The differentiated circuit must be unitary (no measurement, reset, or
//! conditional) on both paths. Differentiable gates are `Rx`, `Ry`, `Rz`,
//! `Rzz`, `P`, and `PauliRot` for both: those are the `Gate` variants carrying
//! a rotation angle, so the shift rule reaches no gate the adjoint rejects.

use std::borrow::Cow;

use num_complex::Complex64;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits, reserve_dense_output};
use crate::circuit::parameter::{ParamLink, Parameters, angle_mut, angle_of};
use crate::circuit::{Circuit, Instruction, PreparedCircuit, SmallVec, smallvec};
use crate::error::{PrismError, Result};
use crate::gates::{Gate, GeneratorKind, is_diagonal_2x2, pauli_rot_masks};

use super::noise::NoiseModel;
use super::unified_pauli::PauliTerm;
use super::{BackendKind, i_pow, pauli_masks, pauli_sandwiches_from_masks};

/// Expectation value and its gradient with respect to each parameter slot.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectationGradient {
    /// `⟨H⟩` at the circuit's current parameter values.
    pub value: f64,
    /// `d⟨H⟩/dθ`, one entry per parameter slot.
    pub gradient: Vec<f64>,
}

/// Compute `⟨H⟩` and its exact gradient with respect to the trainable
/// parameters using the adjoint method on the statevector backend.
///
/// `hamiltonian` is a weighted Pauli sum `Σ c_k P_k` with real coefficients;
/// each `P_k` is a joint Pauli string (identity factors omitted). `params`
/// declares which gate instructions are trainable and how they map to the
/// gradient vector. The returned gradient has length `params.num_slots()`.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, Parameters, PauliTerm, run_expectation_gradient};
///
/// let theta = 0.5_f64;
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::Rx(theta), &[0]);
///
/// let hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
/// let params = Parameters::all_rotations(&circuit);
/// let g = run_expectation_gradient(&circuit, &hamiltonian, &params, 42)?;
/// // <Z0> = cos(theta), d<Z0>/dtheta = -sin(theta).
/// assert!((g.value - theta.cos()).abs() < 1e-12);
/// assert!((g.gradient[0] + theta.sin()).abs() < 1e-9);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn run_expectation_gradient(
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
    seed: u64,
) -> Result<ExpectationGradient> {
    if super::has_nonunitary_or_classical_ops(circuit) {
        return Err(PrismError::IncompatibleBackend {
            backend: "Statevector".into(),
            reason: "adjoint gradients require a unitary circuit without measurements, resets, or conditionals".into(),
        });
    }

    if circuit.instructions.iter().any(|inst| {
        matches!(
            inst,
            Instruction::Gate {
                gate: Gate::QftBlock { .. },
                ..
            }
        )
    }) {
        return Err(PrismError::IncompatibleBackend {
            backend: "Statevector".into(),
            reason: "adjoint gradients do not support QftBlock; expand it to primitive gates first"
                .into(),
        });
    }

    params.validate(circuit)?;

    if circuit.num_qubits > max_statevector_qubits() {
        return Err(PrismError::ResourceLimit(Box::new(
            crate::error::ResourceLimit {
                backend: "Statevector".into(),
                operation: "adjoint gradients, which hold two statevectors,".to_string(),
                resource: crate::error::ResourceKind::Qubits,
                required: circuit.num_qubits as u128,
                limit: max_statevector_qubits() as u128,
                env_var: Some("PRISM_MAX_SV_QUBITS"),
            },
        )));
    }

    // Validate and reduce observables before the 2^n simulation.
    let mut masked = Vec::with_capacity(hamiltonian.len());
    for (coeff, terms) in hamiltonian {
        let (xmask, zmask, num_y) = pauli_masks(terms, circuit.num_qubits)?;
        masked.push((*coeff, xmask, zmask, num_y));
    }

    // A gate outside the Hamiltonian's inverse light cone conjugates the
    // back-propagated observable trivially, so ⟨H⟩ and every gradient entry
    // are unchanged when it is dropped.
    let in_cone = observable_light_cone(circuit, hamiltonian);
    let kept: Vec<usize> = (0..circuit.instructions.len())
        .filter(|&i| in_cone[i])
        .collect();

    // The forward pass has to land |φ⟩ = U|0...0⟩ and nothing else, so it runs
    // the kept gates through the ordinary fusion pipeline. The sweep below
    // rebuilds every intermediate state by inverting `circuit.instructions` one
    // entry at a time, so its 1:1 gate-to-generator view of the trainable gates
    // survives whatever shape the fused stream takes.
    let mut phi = StatevectorBackend::new(seed);
    phi.init(circuit.num_qubits, circuit.num_classical_bits)?;
    let forward = kept_subcircuit(circuit, &kept);
    let expanded = super::expand_for_backend(&phi, &forward);
    let fused = super::fuse_for_backend(&phi, &expanded);
    phi.apply_instructions(&fused.instructions)?;

    let (value, lambda_state) = build_lambda_and_value(phi.state_vector(), &masked)?;

    let mut gradient = vec![0.0; params.num_slots()];

    // In-cone links sorted by instruction index. An out-of-cone trainable gate
    // has a provably zero gradient, so its links carry nothing to accumulate.
    let mut links: Vec<_> = params
        .links()
        .iter()
        .filter(|l| in_cone[l.instruction])
        .copied()
        .collect();
    links.sort_by_key(|l| l.instruction);

    // The sweep stops at the earliest in-cone trainable gate: nothing before
    // it contributes, so a non-trainable prefix costs no inverse applications.
    // If no trainable gate reaches the observable, the gradient is zero
    // everywhere.
    let Some(earliest) = links.first().map(|l| l.instruction) else {
        return Ok(ExpectationGradient { value, gradient });
    };

    let mut lambda = StatevectorBackend::new(seed);
    lambda.init_from_state(lambda_state, circuit.num_classical_bits)?;

    // The sweep is the in-cone tail from the earliest trainable gate. Sweep
    // position `s` owns `links[first_link[s]..first_link[s + 1]]`.
    let sweep = &kept[kept.partition_point(|&i| i < earliest)..];
    let mut first_link = Vec::with_capacity(sweep.len() + 1);
    let mut cursor = 0;
    for &i in sweep {
        first_link.push(cursor);
        while cursor < links.len() && links[cursor].instruction == i {
            cursor += 1;
        }
    }
    first_link.push(cursor);

    let trainable: Vec<bool> = first_link.windows(2).map(|w| w[0] < w[1]).collect();
    let footprints: Vec<Footprint> = sweep
        .iter()
        .map(|&i| {
            let Instruction::Gate { gate, targets } = &circuit.instructions[i] else {
                unreachable!("the light cone keeps gate instructions only")
            };
            Footprint::of(gate, targets)
        })
        .collect();
    let plan = plan_sweep(&footprints, &trainable, circuit.num_qubits);

    let mut masks: Vec<(usize, usize, u32)> = Vec::new();
    let (mut evaluated, mut inverted) = (0, 0);
    for step in 0..=plan.last_step {
        let evaluate = evaluated..plan.evaluate.partition_point(|&(t, _)| t <= step);
        evaluated = evaluate.end;
        if !evaluate.is_empty() {
            masks.clear();
            masks.extend(plan.evaluate[evaluate.clone()].iter().map(|&(_, s)| {
                let f = &footprints[s];
                (f.xmask, f.zmask, f.num_y)
            }));
            let values =
                pauli_sandwiches_from_masks(lambda.state_vector(), phi.state_vector(), &masks);
            for (&(_, s), value) in plan.evaluate[evaluate].iter().zip(&values) {
                for link in &links[first_link[s]..first_link[s + 1]] {
                    gradient[link.slot] += value.im;
                }
            }
        }

        let invert = inverted..plan.invert.partition_point(|&(t, _)| t <= step);
        inverted = invert.end;
        if !invert.is_empty() {
            let stretch = circuit.with_instructions(
                plan.invert[invert]
                    .iter()
                    .map(|&(_, s)| inverse_instruction(&circuit.instructions[sweep[s]]))
                    .collect(),
            );
            let expanded = super::expand_for_backend(&phi, &stretch);
            let fused = super::fuse_for_backend(&phi, &expanded);
            phi.apply_instructions(&fused.instructions)?;
            lambda.apply_instructions(&fused.instructions)?;
        }
    }

    Ok(ExpectationGradient { value, gradient })
}

/// How a gate acts on one qubit: within the algebra of `I` and one Pauli, or
/// anything. Two gates commute when their actions agree on every shared qubit
/// and neither is `Any`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalAction {
    Z,
    X,
    Y,
    Any,
}

impl LocalAction {
    fn from_bits(x: bool, z: bool) -> Option<Self> {
        match (x, z) {
            (false, false) => None,
            (true, false) => Some(Self::X),
            (false, true) => Some(Self::Z),
            (true, true) => Some(Self::Y),
        }
    }

    fn bit(self) -> u8 {
        match self {
            Self::Z => 1,
            Self::X => 2,
            Self::Y => 4,
            Self::Any => 7,
        }
    }
}

/// What the sweep planner knows about how one gate commutes.
///
/// `pauli` marks a gate in the span of `I` and one Pauli string with masks
/// `(xmask, zmask, num_y)`, the generator for a rotation. Two such gates commute
/// exactly when their strings do, which the per-qubit `support` test misses for
/// strings that anticommute on an even number of qubits; every other pair falls
/// back to it.
struct Footprint {
    pauli: bool,
    xmask: usize,
    zmask: usize,
    num_y: u32,
    support: SmallVec<[(usize, LocalAction); 4]>,
}

impl Footprint {
    fn pauli((xmask, zmask, num_y): (usize, usize, u32), targets: &[usize]) -> Self {
        let support = targets
            .iter()
            .filter_map(|&q| {
                LocalAction::from_bits(xmask >> q & 1 == 1, zmask >> q & 1 == 1).map(|a| (q, a))
            })
            .collect();
        Self {
            pauli: true,
            xmask,
            zmask,
            num_y,
            support,
        }
    }

    fn of(gate: &Gate, targets: &[usize]) -> Self {
        use LocalAction::{Any, X, Z};
        if let Some(kind) = gate.pauli_generator() {
            return Self::pauli(generator_masks(kind, targets), targets);
        }
        let bit = || 1usize << targets[0];
        let diagonal_or_any = |diagonal: bool| if diagonal { Z } else { Any };
        let support: SmallVec<[(usize, LocalAction); 4]> = match gate {
            Gate::Id => SmallVec::new(),
            Gate::X | Gate::SX | Gate::SXdg => return Self::pauli((bit(), 0, 0), targets),
            Gate::Y => return Self::pauli((bit(), bit(), 1), targets),
            Gate::Z | Gate::S | Gate::Sdg | Gate::T | Gate::Tdg => {
                return Self::pauli((0, bit(), 0), targets);
            }
            Gate::Cx => smallvec![(targets[0], Z), (targets[1], X)],
            Gate::Cu(mat) => smallvec![
                (targets[0], Z),
                (targets[1], diagonal_or_any(is_diagonal_2x2(mat)))
            ],
            Gate::Mcu(data) => {
                let (&target, controls) = targets.split_last().expect("Mcu has a target");
                controls
                    .iter()
                    .map(|&q| (q, Z))
                    .chain([(target, diagonal_or_any(is_diagonal_2x2(&data.mat)))])
                    .collect()
            }
            Gate::Fused(mat) => smallvec![(targets[0], diagonal_or_any(is_diagonal_2x2(mat)))],
            Gate::BatchPhase(data) => targets
                .iter()
                .copied()
                .chain(data.phases.iter().map(|&(q, _)| q))
                .map(|q| (q, Z))
                .collect(),
            Gate::Cz | Gate::BatchRzz(_) | Gate::DiagonalBatch(_) => {
                targets.iter().map(|&q| (q, Z)).collect()
            }
            Gate::MultiFused(data) => {
                let action = diagonal_or_any(data.all_diagonal);
                targets.iter().map(|&q| (q, action)).collect()
            }
            _ => targets.iter().map(|&q| (q, Any)).collect(),
        };
        Self {
            pauli: false,
            xmask: 0,
            zmask: 0,
            num_y: 0,
            support,
        }
    }

    fn commutes_with(&self, other: &Footprint) -> bool {
        if self.pauli && other.pauli {
            let anticommuting =
                (self.xmask & other.zmask).count_ones() + (self.zmask & other.xmask).count_ones();
            return anticommuting.is_multiple_of(2);
        }
        self.support.iter().all(|&(q, a)| {
            other
                .support
                .iter()
                .all(|&(p, b)| p != q || (a == b && a != LocalAction::Any))
        })
    }

    /// Whether a conflicting action on any one shared qubit already stops this
    /// gate commuting with another, as it does for all but a Pauli string over
    /// several qubits.
    fn decides_locally(&self) -> bool {
        !self.pauli || self.support.len() == 1
    }

    fn action_on(&self, qubit: usize) -> LocalAction {
        self.support
            .iter()
            .find_map(|&(q, a)| (q == qubit).then_some(a))
            .expect("a gate is listed only under qubits it acts on")
    }
}

/// Pauli masks `(xmask, zmask, num_y)` of a rotation's generator.
fn generator_masks(kind: GeneratorKind<'_>, targets: &[usize]) -> (usize, usize, u32) {
    match kind {
        GeneratorKind::RotX => (1usize << targets[0], 0, 0),
        GeneratorKind::RotY => {
            let bit = 1usize << targets[0];
            (bit, bit, 1)
        }
        GeneratorKind::RotZ => (0, 1usize << targets[0], 0),
        GeneratorKind::RotZz => (0, (1usize << targets[0]) | (1usize << targets[1]), 0),
        // The projector generator differs from Z by the identity, whose
        // sandwich `⟨λ|φ⟩ = ⟨φ|H|φ⟩` is real and contributes nothing to the
        // imaginary part, so `P(θ) = e^{iθ/2} Rz(θ)` differentiates as `Rz`.
        GeneratorKind::Phase => (0, 1usize << targets[0], 0),
        GeneratorKind::RotPauli(axes) => pauli_rot_masks(targets, axes),
    }
}

fn inverse_instruction(instruction: &Instruction) -> Instruction {
    let Instruction::Gate { gate, targets } = instruction else {
        unreachable!("the light cone keeps gate instructions only")
    };
    Instruction::Gate {
        gate: gate.inverse(),
        targets: targets.clone(),
    }
}

/// Gates the planner checks above a gate on one qubit before it counts every
/// gate left there as blocking. The bound keeps planning near linear through a
/// long commuting stretch; past it the plan stays exact but may take more steps.
const SWEEP_SCAN_LIMIT: usize = 64;

/// Backward-sweep schedule, as `(step, sweep position)` pairs sorted by step.
///
/// Step `t` reads the sandwiches of `evaluate`'s entries at `t` from one state
/// pair, then applies the inverses of `invert`'s entries at `t`, highest
/// position first.
struct SweepPlan {
    evaluate: Vec<(usize, usize)>,
    invert: Vec<(usize, usize)>,
    last_step: usize,
}

/// Schedule the backward sweep so every trainable gate is evaluated at the
/// earliest step its sandwich is exact.
///
/// The states always sit after the gates not yet inverted, applied in order. A
/// gate may be inverted once every gate above it that it does not commute with
/// has been, since it then commutes past the rest to the end of the circuit. A
/// trainable gate's sandwich at that frontier equals the one at its own place
/// once every gate left above it commutes with its generator. A top-down pass
/// gives each gate its earliest step, and a bottom-up pass keeps only the gates
/// some trainable gate below transitively waits on, so the earliest trainable
/// gate is never inverted.
fn plan_sweep(footprints: &[Footprint], trainable: &[bool], num_qubits: usize) -> SweepPlan {
    let len = footprints.len();
    let mut on_qubit: Vec<Vec<usize>> = vec![Vec::new(); num_qubits];
    let rows: Vec<SmallVec<[usize; 4]>> = footprints
        .iter()
        .enumerate()
        .map(|(s, f)| {
            f.support
                .iter()
                .map(|&(q, _)| {
                    on_qubit[q].push(s);
                    on_qubit[q].len() - 1
                })
                .collect()
        })
        .collect();

    // `blockers[blocker_start[s]..blocker_start[s + 1]]` holds gates above `s`
    // that it does not commute with. A scan on a qubit stops once its blockers
    // there take two different actions, or one `Any`, from gates that decide
    // locally: every later gate on the qubit then fails to commute with one of
    // them and waits on it already. A scan that hits the limit instead leaves a
    // `(s, qubit, row)` spill standing for every gate from `row` up.
    let mut blocker_start = Vec::with_capacity(len + 1);
    let mut blockers: Vec<usize> = Vec::new();
    let mut spills: Vec<(usize, usize, usize)> = Vec::new();
    blocker_start.push(0);
    for (s, f) in footprints.iter().enumerate() {
        for (&(q, _), &row) in f.support.iter().zip(&rows[s]) {
            let list = &on_qubit[q];
            let limit = list.len().min(row + 1 + SWEEP_SCAN_LIMIT);
            let mut covered = 0u8;
            let mut next = row + 1;
            while next < limit && covered.count_ones() < 2 {
                let above = &footprints[list[next]];
                if !f.commutes_with(above) {
                    blockers.push(list[next]);
                    if above.decides_locally() {
                        covered |= above.action_on(q).bit();
                    }
                }
                next += 1;
            }
            if covered.count_ones() < 2 && next < list.len() {
                spills.push((s, q, next));
            }
        }
        blocker_start.push(blockers.len());
    }

    // A trainable gate is evaluated one step after the last of its blockers is
    // inverted and inverted in the step it is evaluated; any other gate is
    // inverted in the step its last blocker is.
    let mut step = vec![0usize; len];
    let mut latest_from: Vec<Vec<usize>> = on_qubit.iter().map(|l| vec![0; l.len()]).collect();
    let mut spill = spills.len();
    for s in (0..len).rev() {
        let mut latest = blockers[blocker_start[s]..blocker_start[s + 1]]
            .iter()
            .map(|&i| step[i])
            .max();
        while spill > 0 && spills[spill - 1].0 == s {
            spill -= 1;
            let (_, q, row) = spills[spill];
            latest = latest.max(Some(latest_from[q][row]));
        }
        step[s] = match latest {
            Some(t) if trainable[s] => t + 1,
            Some(t) => t,
            None => 0,
        };
        for (&(q, _), &row) in footprints[s].support.iter().zip(&rows[s]) {
            let above = latest_from[q].get(row + 1).copied().unwrap_or(0);
            latest_from[q][row] = step[s].max(above);
        }
    }

    let mut needed = vec![false; len];
    let mut needed_from = vec![usize::MAX; num_qubits];
    let mut spill = 0;
    for s in 0..len {
        needed[s] = needed[s]
            || footprints[s]
                .support
                .iter()
                .zip(&rows[s])
                .any(|(&(q, _), &row)| row >= needed_from[q]);
        let waits = needed[s] || trainable[s];
        if waits {
            for &i in &blockers[blocker_start[s]..blocker_start[s + 1]] {
                needed[i] = true;
            }
        }
        while spill < spills.len() && spills[spill].0 == s {
            let (_, q, row) = spills[spill];
            if waits {
                needed_from[q] = needed_from[q].min(row);
            }
            spill += 1;
        }
    }

    let last_step = (0..len)
        .filter(|&s| trainable[s])
        .map(|s| step[s])
        .max()
        .unwrap_or(0);
    let mut evaluate: Vec<(usize, usize)> = (0..len)
        .filter(|&s| trainable[s])
        .map(|s| (step[s], s))
        .collect();
    evaluate.sort_unstable();
    let mut invert: Vec<(usize, usize)> = (0..len)
        .filter(|&s| needed[s] && step[s] < last_step)
        .map(|s| (step[s], s))
        .collect();
    invert.sort_unstable_by_key(|&(t, s)| (t, std::cmp::Reverse(s)));
    SweepPlan {
        evaluate,
        invert,
        last_step,
    }
}

/// The gates `kept` indexes, as a circuit of the original width so the fusion
/// floors read the same qubit count. Borrowed when the cone keeps every
/// instruction.
fn kept_subcircuit<'a>(circuit: &'a Circuit, kept: &[usize]) -> Cow<'a, Circuit> {
    if kept.len() == circuit.instructions.len() {
        return Cow::Borrowed(circuit);
    }
    let instructions = kept
        .iter()
        .map(|&i| circuit.instructions[i].clone())
        .collect();
    Cow::Owned(circuit.with_instructions(instructions))
}

/// Per-instruction flag: true if the gate lies in the Hamiltonian's inverse
/// light cone (its support is connected to some observable term through the
/// gates that follow it).
fn observable_light_cone(circuit: &Circuit, hamiltonian: &[(f64, Vec<PauliTerm>)]) -> Vec<bool> {
    let union: Vec<PauliTerm> = hamiltonian
        .iter()
        .flat_map(|(_, terms)| terms.iter().copied())
        .collect();
    super::unified_pauli::inverse_light_cone(circuit, &union)
}

/// Hamiltonian terms sharing one X mask. The gather reads `phi[i ^ xmask]`
/// once per group; each `(Zmask, factor)` pair then contributes its own sign to
/// that one amplitude.
struct TermGroup {
    xmask: usize,
    terms: Vec<(usize, Complex64)>,
}

/// Group masked terms by X mask, ascending, so the diagonal terms (X mask 0)
/// lead and the gather's first read is the sequential stream.
fn group_terms_by_xmask(masked: &[(f64, usize, usize, u32)]) -> Vec<TermGroup> {
    let mut flat: Vec<(usize, usize, Complex64)> = masked
        .iter()
        .map(|&(coeff, xmask, zmask, num_y)| {
            (xmask, zmask, Complex64::new(coeff, 0.0) * i_pow(num_y))
        })
        .collect();
    flat.sort_by_key(|&(xmask, _, _)| xmask);

    let mut groups: Vec<TermGroup> = Vec::with_capacity(flat.len());
    for (xmask, zmask, factor) in flat {
        match groups.last_mut() {
            Some(group) if group.xmask == xmask => group.terms.push((zmask, factor)),
            _ => groups.push(TermGroup {
                xmask,
                terms: vec![(zmask, factor)],
            }),
        }
    }
    groups
}

/// Gather `|λ⟩ = Σ c_k P_k|φ⟩` over `out`, the slice of `λ` starting at `base`,
/// and return that slice's share of `Re⟨φ|λ⟩`.
///
/// Each output element is written once, from
/// `Σ_k factor_k · (-1)^popcount((i ⊕ Xmask_k) & Zmask_k) · phi[i ⊕ Xmask_k]`, so
/// the terms batch into one pass over the register instead of one scattering
/// pass each.
#[inline(always)]
fn gather_lambda_chunk(
    groups: &[TermGroup],
    phi: &[Complex64],
    base: usize,
    out: &mut [Complex64],
) -> f64 {
    let mut value = 0.0;
    for (offset, slot) in out.iter_mut().enumerate() {
        let i = base + offset;
        let mut acc = Complex64::new(0.0, 0.0);
        for group in groups {
            let j = i ^ group.xmask;
            let mut weight = Complex64::new(0.0, 0.0);
            for &(zmask, factor) in &group.terms {
                let sign = if (j & zmask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                weight += factor * sign;
            }
            // SAFETY: callers pass `out` as a chunk of a buffer of `phi`'s
            // length starting at `base`, so `i < phi.len()`, and every X mask
            // below the power-of-two `phi.len()`, so `i ^ xmask` stays in range.
            // `build_lambda_and_value` asserts both before it fans out.
            let amp = unsafe { *phi.get_unchecked(j) };
            acc += weight * amp;
        }
        // SAFETY: same range argument with an X mask of zero.
        let p = unsafe { *phi.get_unchecked(i) };
        value += p.re * acc.re + p.im * acc.im;
        *slot = acc;
    }
    value
}

/// Build `|λ⟩ = Σ c_k P_k|φ⟩` into a fresh buffer and return `(⟨H⟩, |λ⟩)`,
/// where `⟨H⟩ = Re⟨φ|λ⟩`.
///
/// # Panics
///
/// If `phi`'s length is not a power of two or a term's X mask indexes past it.
fn build_lambda_and_value(
    phi: &[Complex64],
    masked: &[(f64, usize, usize, u32)],
) -> Result<(f64, Vec<Complex64>)> {
    let dim = phi.len();
    let mut lambda: Vec<Complex64> = Vec::new();
    reserve_dense_output(
        &mut lambda,
        dim,
        "Statevector",
        "adjoint gradient lambda state",
    )?;
    lambda.resize(dim, Complex64::new(0.0, 0.0));

    let groups = group_terms_by_xmask(masked);
    assert!(
        dim.is_power_of_two() && groups.iter().all(|g| g.xmask < dim),
        "observable masks must index the state dimension"
    );

    #[cfg(feature = "parallel")]
    if dim >= (1 << crate::backend::PARALLEL_THRESHOLD_QUBITS) {
        use crate::backend::MIN_PAR_ELEMS;
        use rayon::prelude::*;

        let value = lambda
            .par_chunks_mut(MIN_PAR_ELEMS)
            .enumerate()
            .map(|(chunk, out)| gather_lambda_chunk(&groups, phi, chunk * MIN_PAR_ELEMS, out))
            .sum();
        return Ok((value, lambda));
    }

    let value = gather_lambda_chunk(&groups, phi, 0, &mut lambda);
    Ok((value, lambda))
}

/// Compute `⟨H⟩` and its gradient by the parameter-shift rule, routing every
/// evaluation through automatic backend selection.
///
/// Unlike [`run_expectation_gradient`] this places no ceiling on the qubit
/// count of its own: it inherits whatever the selected backend can represent,
/// holding one backend state at a time from 17 qubits up. Below that, under the
/// `parallel` feature, the shifted evaluations split across Rayon workers with a
/// state each, and the gradient is bit-identical to the sequential sum. It also
/// accepts `QftBlock`. The price is `1 + 2 * params.links().len()` circuit
/// evaluations against the adjoint's one, so prefer the adjoint wherever it
/// applies. Select an explicit backend with [`crate::simulate`] and
/// `expectation_gradient_shift`.
///
/// # Examples
///
/// ```
/// use prism_q::{Circuit, Gate, Parameters, PauliTerm, run_expectation_gradient_shift};
///
/// let theta = 0.5_f64;
/// let mut circuit = Circuit::new(2, 0);
/// circuit.add_gate(Gate::Rx(theta), &[0]);
///
/// let hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
/// let params = Parameters::all_rotations(&circuit);
/// let g = run_expectation_gradient_shift(&circuit, &hamiltonian, &params, 42)?;
/// assert!((g.gradient[0] + theta.sin()).abs() < 1e-9);
/// # Ok::<(), prism_q::PrismError>(())
/// ```
pub fn run_expectation_gradient_shift(
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
    seed: u64,
) -> Result<ExpectationGradient> {
    shift_gradient(
        &BackendKind::Auto,
        circuit,
        hamiltonian,
        params,
        None,
        None,
        seed,
    )
}

/// Parameter-shift gradient on the backend `kind` selects, optionally from a
/// start state.
///
/// Every differentiable gate is `exp(-iθG/2)` with `G` of eigenvalues `±1`, so
/// `⟨H⟩` is a degree-1 trigonometric polynomial in each angle and
/// `d⟨H⟩/dθ = (f(θ+π/2) - f(θ-π/2)) / 2` is exact. `P(θ) = diag(1, e^{iθ})` has
/// the projector `|1⟩⟨1|` for a generator, eigenvalues `{0, 1}` rather than
/// `{-1, +1}`, but `P(θ) = e^{iθ/2} Rz(θ)`: the θ-dependent factor is a scalar
/// wherever the gate sits, so it cancels against its conjugate in `⟨ψ|H|ψ⟩` and
/// the same shift applies unchanged.
///
/// Gates sharing a parameter slot are shifted one at a time and summed. Shifting
/// them together is a different quantity: two `Rx(θ)` on one qubit under `⟨Z⟩`
/// give `cos 2θ`, whose joint ±π/2 shift is zero rather than `-2 sin 2θ`.
///
/// Under `noise` every forward evaluation reads the exact mixture, so `kind`
/// must be a density-matrix kind, which the caller checks. The channels do not
/// depend on the shifted angle, so `⟨H⟩` stays a degree-1 trigonometric
/// polynomial in it and the shift is still exact.
pub(crate) fn shift_gradient(
    kind: &BackendKind,
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
    noise: Option<&NoiseModel>,
    initial_state: Option<super::StartState<'_>>,
    seed: u64,
) -> Result<ExpectationGradient> {
    params.validate(circuit)?;
    // Below the fusion floor there is no plan to replay, and the prepared copy
    // each worker builds costs more than the route and allocation it saves.
    if initial_state.is_none()
        && noise.is_none()
        && circuit.num_qubits >= crate::circuit::fusion::MIN_QUBITS_FOR_FUSION
    {
        return prepared_shift_gradient(kind, circuit, hamiltonian, params, seed);
    }
    if initial_state.is_some() || noise.is_some() {
        super::require_unitary_circuit(kind, circuit, "expectation values require")?;
    }

    let observables: Vec<Vec<PauliTerm>> =
        hamiltonian.iter().map(|(_, terms)| terms.clone()).collect();
    let evaluate = |c: &Circuit| -> Result<f64> {
        if let BackendKind::PauliPath { epsilon, max_terms } = kind {
            let per_term =
                super::pauli_path_expectations(c, noise, &observables, *epsilon, *max_terms)?
                    .into_values();
            return Ok(hamiltonian
                .iter()
                .zip(per_term)
                .map(|((coeff, _), v)| coeff * v)
                .sum());
        }
        let per_term = match (noise, initial_state) {
            (Some(noise), _) => super::noise::dm_expectation_values(
                kind,
                c,
                &observables,
                Some(noise),
                initial_state,
                seed,
            )?,
            (None, Some(state)) => {
                super::expectation_values_from_initial_state(kind, c, state, &observables, seed)?
                    .into_values()
            }
            (None, None) => {
                super::run_expectation_values_with(kind.clone(), c, &observables, seed)?
            }
        };
        Ok(hamiltonian
            .iter()
            .zip(per_term)
            .map(|((coeff, _), v)| coeff * v)
            .sum())
    };

    let value = evaluate(circuit)?;
    let mut gradient = vec![0.0; params.num_slots()];
    if params.is_empty() {
        return Ok(ExpectationGradient { value, gradient });
    }

    let shift = std::f64::consts::FRAC_PI_2;
    let term = |shifted: &mut Circuit, instruction: usize| -> Result<f64> {
        let base = *angle_mut(&mut shifted.instructions[instruction]);
        *angle_mut(&mut shifted.instructions[instruction]) = base + shift;
        let plus = evaluate(shifted)?;
        *angle_mut(&mut shifted.instructions[instruction]) = base - shift;
        let minus = evaluate(shifted)?;
        *angle_mut(&mut shifted.instructions[instruction]) = base;
        Ok(0.5 * (plus - minus))
    };

    // Terms are summed into their slots in link order on both paths, so the
    // parallel gradient is bit-identical to the sequential one.
    let links = params.links();
    #[cfg(feature = "parallel")]
    if links.len() > 1 && super::runs_split_across_workers(kind, circuit.num_qubits) {
        use rayon::prelude::*;
        let terms: Vec<Result<f64>> = links
            .par_iter()
            .map_init(
                || circuit.clone(),
                |shifted, link| term(shifted, link.instruction),
            )
            .collect();
        for (link, t) in links.iter().zip(terms) {
            gradient[link.slot] += t?;
        }
        return Ok(ExpectationGradient { value, gradient });
    }

    let mut shifted = circuit.clone();
    for link in links {
        gradient[link.slot] += term(&mut shifted, link.instruction)?;
    }

    Ok(ExpectationGradient { value, gradient })
}

/// [`shift_gradient`] with neither noise nor a start state: every evaluation binds
/// one [`PreparedCircuit`], so the route is settled and the fusion plan captured
/// once rather than per shifted circuit.
fn prepared_shift_gradient(
    kind: &BackendKind,
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &Parameters,
    seed: u64,
) -> Result<ExpectationGradient> {
    // One slot per linked instruction rather than per parameter slot, since a
    // shift moves one gate even where its slot drives several.
    let links = params.links();
    let mut sites: Vec<usize> = links.iter().map(|link| link.instruction).collect();
    sites.sort_unstable();
    sites.dedup();
    let base: Vec<f64> = sites
        .iter()
        .map(|&i| angle_of(&circuit.instructions[i]))
        .collect();
    let site_params = Parameters::from_links(
        sites
            .iter()
            .enumerate()
            .map(|(slot, &instruction)| ParamLink { instruction, slot })
            .collect(),
        sites.len(),
    );

    let observables: Vec<Vec<PauliTerm>> =
        hamiltonian.iter().map(|(_, terms)| terms.clone()).collect();
    let energy = |prepared: &mut PreparedCircuit, values: &[f64]| -> Result<f64> {
        let per_term = prepared.expectation_values(values, &observables, seed)?;
        Ok(hamiltonian
            .iter()
            .zip(per_term)
            .map(|((coeff, _), v)| coeff * v)
            .sum())
    };

    let mut prepared = PreparedCircuit::with_backend(circuit.clone(), site_params, kind.clone())?;
    let value = energy(&mut prepared, &base)?;
    let mut gradient = vec![0.0; params.num_slots()];
    if params.is_empty() {
        return Ok(ExpectationGradient { value, gradient });
    }

    // Terms are summed into their slots in link order however the links were
    // split, so the gradient does not depend on the thread count.
    let shift = std::f64::consts::FRAC_PI_2;
    let terms = prepared.map_split(links, |worker, link| {
        let site = sites
            .binary_search(&link.instruction)
            .expect("every link names a site");
        let mut values = base.clone();
        values[site] = base[site] + shift;
        let plus = energy(worker, &values)?;
        values[site] = base[site] - shift;
        let minus = energy(worker, &values)?;
        Ok(0.5 * (plus - minus))
    })?;
    for (link, term) in links.iter().zip(terms) {
        gradient[link.slot] += term;
    }

    Ok(ExpectationGradient { value, gradient })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::unified_pauli::PauliTerm;

    fn z_obs(qubit: usize) -> Vec<(f64, Vec<PauliTerm>)> {
        vec![(1.0, vec![PauliTerm::z(qubit)])]
    }

    #[test]
    fn single_rx_gradient_matches_analytic() {
        // Rx(θ)|0>, <Z> = cos θ, d/dθ = -sin θ.
        let theta = 0.7;
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(theta), &[0]);
        let mut params = Parameters::new(1);
        params.link(0, 0);

        let g = run_expectation_gradient(&c, &z_obs(0), &params, 42).unwrap();
        assert!((g.value - theta.cos()).abs() < 1e-12);
        assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
    }

    #[test]
    fn ry_generator_carries_num_y() {
        // Ry(θ)|0>, <Z> = cos θ, d/dθ = -sin θ. Generator Y has num_y = 1.
        let theta = 1.3;
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Ry(theta), &[0]);
        let mut params = Parameters::new(1);
        params.link(0, 0);

        let g = run_expectation_gradient(&c, &z_obs(0), &params, 42).unwrap();
        assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
    }

    #[test]
    fn phase_projector_gradient() {
        // H then P(θ): |ψ> = (|0> + e^{iθ}|1>)/√2, <X> = cos θ, d/dθ = -sin θ.
        let theta = 0.9;
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::P(theta), &[0]);
        let mut params = Parameters::new(1);
        params.link(1, 0);

        let obs = vec![(1.0, vec![PauliTerm::x(0)])];
        let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        assert!((g.value - theta.cos()).abs() < 1e-12);
        assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
    }

    #[test]
    fn shared_parameter_accumulates() {
        // Two Rx gates on separate qubits sharing one slot; each contributes
        // -sin θ to <Z0 + Z1>, so the shared gradient is -2 sin θ.
        let theta = 0.4;
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Rx(theta), &[0]);
        c.add_gate(Gate::Rx(theta), &[1]);
        let mut params = Parameters::new(1);
        params.link(0, 0);
        params.link(1, 0);

        let obs = vec![(1.0, vec![PauliTerm::z(0)]), (1.0, vec![PauliTerm::z(1)])];
        let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        assert_eq!(g.gradient.len(), 1);
        assert!((g.gradient[0] - (-2.0 * theta.sin())).abs() < 1e-9);
    }

    fn hea_with_shared_slots(n: usize) -> (Circuit, Parameters) {
        let circuit = crate::circuits::hardware_efficient_ansatz(n, 2, 7);
        let mut params = Parameters::new(3);
        let rotations = circuit.instructions.iter().enumerate().filter(|(_, inst)| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::Ry(_) | Gate::Rz(_),
                    ..
                }
            )
        });
        for (k, (i, _)) in rotations.enumerate() {
            params.link(i, k % 3);
        }
        (circuit, params)
    }

    fn slot_per_link(params: &Parameters) -> Parameters {
        let links = params.links();
        Parameters::from_links(
            links
                .iter()
                .enumerate()
                .map(|(slot, link)| ParamLink {
                    instruction: link.instruction,
                    slot,
                })
                .collect(),
            links.len(),
        )
    }

    fn weighted_energy(ham: &[(f64, Vec<PauliTerm>)], per_term: Vec<f64>) -> f64 {
        ham.iter()
            .zip(per_term)
            .map(|((coeff, _), v)| coeff * v)
            .sum()
    }

    // Ten qubits splits the links across workers under `parallel` and replays a
    // fusion plan; the reference sums the same prepared terms in link order on
    // one thread, so the bits must agree.
    #[test]
    fn shift_gradient_matches_a_sequential_sum_bit_for_bit() {
        let (circuit, params) = hea_with_shared_slots(10);
        let ham = vec![
            (0.5, vec![PauliTerm::z(0), PauliTerm::z(1)]),
            (1.5, vec![PauliTerm::x(3)]),
        ];
        let observables: Vec<Vec<PauliTerm>> = ham.iter().map(|(_, t)| t.clone()).collect();
        let links = params.links();
        let sites = slot_per_link(&params);
        let base = sites.values(&circuit).unwrap();
        let mut prepared = PreparedCircuit::new(circuit.clone(), sites).unwrap();
        assert!(prepared.reuses_fusion_plan());
        let mut energy = |values: &[f64]| {
            weighted_energy(
                &ham,
                prepared
                    .expectation_values(values, &observables, 42)
                    .unwrap(),
            )
        };
        let shift = std::f64::consts::FRAC_PI_2;
        let mut expected = vec![0.0; 3];
        for (k, link) in links.iter().enumerate() {
            let mut values = base.clone();
            values[k] = base[k] + shift;
            let plus = energy(&values);
            values[k] = base[k] - shift;
            let minus = energy(&values);
            expected[link.slot] += 0.5 * (plus - minus);
        }

        let g = run_expectation_gradient_shift(&circuit, &ham, &params, 42).unwrap();
        assert_eq!(g.gradient, expected);
    }

    // Angles of pi/2 on half the rotations shift to 0 and pi, which collapse a
    // fused block to the identity or a named gate and force the replay onto the
    // full pass pipeline.
    #[test]
    fn shift_gradient_matches_fresh_fusion_on_degenerate_shifts() {
        for n in [6, 10] {
            let (mut circuit, params) = hea_with_shared_slots(n);
            for link in params.links().iter().step_by(2) {
                *angle_mut(&mut circuit.instructions[link.instruction]) =
                    std::f64::consts::FRAC_PI_2;
            }
            let ham: Vec<(f64, Vec<PauliTerm>)> = (0..n - 1)
                .map(|q| (1.0, vec![PauliTerm::z(q), PauliTerm::z(q + 1)]))
                .chain([(0.7, vec![PauliTerm::x(2)])])
                .collect();
            let observables: Vec<Vec<PauliTerm>> = ham.iter().map(|(_, t)| t.clone()).collect();
            let energy = |c: &Circuit| {
                weighted_energy(
                    &ham,
                    super::super::run_expectation_values_with(
                        BackendKind::Auto,
                        c,
                        &observables,
                        42,
                    )
                    .unwrap(),
                )
            };
            let shift = std::f64::consts::FRAC_PI_2;
            let mut expected = vec![0.0; params.num_slots()];
            for link in params.links() {
                let mut shifted = circuit.clone();
                let base = *angle_mut(&mut shifted.instructions[link.instruction]);
                *angle_mut(&mut shifted.instructions[link.instruction]) = base + shift;
                let plus = energy(&shifted);
                *angle_mut(&mut shifted.instructions[link.instruction]) = base - shift;
                let minus = energy(&shifted);
                expected[link.slot] += 0.5 * (plus - minus);
            }

            if n == 10 {
                let sites = slot_per_link(&params);
                let base = sites.values(&circuit).unwrap();
                let mut prepared = PreparedCircuit::new(circuit.clone(), sites).unwrap();
                assert!(prepared.reuses_fusion_plan());
                let replayed = prepared.bind_fused(&base).unwrap().instructions.len();
                let fell_back = (0..base.len()).any(|k| {
                    let mut values = base.clone();
                    values[k] -= shift;
                    prepared.bind_fused(&values).unwrap().instructions.len() != replayed
                });
                assert!(fell_back, "no shifted binding left the captured plan");
            }

            let g = run_expectation_gradient_shift(&circuit, &ham, &params, 42).unwrap();
            assert!((g.value - energy(&circuit)).abs() < 1e-10, "{n}q value");
            for (slot, (got, want)) in g.gradient.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() < 1e-10,
                    "{n}q slot {slot}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn empty_params_returns_value_only() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(0.5), &[0]);
        let g = run_expectation_gradient(&c, &z_obs(0), &Parameters::new(0), 42).unwrap();
        assert!(g.gradient.is_empty());
        assert!((g.value - 0.5f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn nondifferentiable_trainable_gate_is_rejected() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let mut params = Parameters::new(1);
        params.link(0, 0);
        assert!(run_expectation_gradient(&c, &z_obs(0), &params, 42).is_err());
    }

    #[test]
    fn nonunitary_circuit_is_rejected() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::Rx(0.3), &[0]);
        c.add_measure(0, 0);
        assert!(run_expectation_gradient(&c, &z_obs(0), &Parameters::new(0), 42).is_err());
    }

    #[test]
    fn single_term_hamiltonian_matches_analytic() {
        // Ry(theta)|0>, <X> = sin theta, d/dtheta = cos theta.
        let theta = 0.6;
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Ry(theta), &[0]);
        let params = Parameters::all_rotations(&c);

        let obs = vec![(1.0, vec![PauliTerm::x(0)])];
        let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        assert!((g.value - theta.sin()).abs() < 1e-12);
        assert!((g.gradient[0] - theta.cos()).abs() < 1e-12);
    }

    #[test]
    fn diagonal_hamiltonian_matches_analytic() {
        // Two independent Ry rotations under Z0, Z1 and Z0Z1: every term is
        // diagonal, so nothing moves an amplitude.
        let (a, b) = (0.4, 1.1);
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::Ry(a), &[0]);
        c.add_gate(Gate::Ry(b), &[1]);
        let params = Parameters::all_rotations(&c);

        let obs = vec![
            (1.0, vec![PauliTerm::z(0)]),
            (0.5, vec![PauliTerm::z(1)]),
            (0.25, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        ];
        let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        let value = a.cos() + 0.5 * b.cos() + 0.25 * a.cos() * b.cos();
        assert!((g.value - value).abs() < 1e-12);
        assert!((g.gradient[0] - (-a.sin() - 0.25 * a.sin() * b.cos())).abs() < 1e-12);
        assert!((g.gradient[1] - (-0.5 * b.sin() - 0.25 * a.cos() * b.sin())).abs() < 1e-12);
    }

    #[test]
    fn grouped_x_mask_terms_match_parameter_shift() {
        // X0, Y0, X0Z1 and Y0Z2 all carry the X mask of qubit 0, four terms
        // reading the same amplitude.
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Ry(0.4), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rx(0.9), &[1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Rz(0.3), &[2]);
        let params = Parameters::all_rotations(&c);

        let obs = vec![
            (1.0, vec![PauliTerm::x(0)]),
            (-0.5, vec![PauliTerm::y(0)]),
            (0.75, vec![PauliTerm::x(0), PauliTerm::z(1)]),
            (0.25, vec![PauliTerm::y(0), PauliTerm::z(2)]),
        ];
        let adjoint = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        let shift = run_expectation_gradient_shift(&c, &obs, &params, 42).unwrap();
        assert!((adjoint.value - shift.value).abs() < 1e-12);
        for (slot, (&got, &want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
            assert!((got - want).abs() < 1e-12, "slot {slot}: {got} vs {want}");
        }
    }

    #[test]
    fn parallel_threshold_width_matches_parameter_shift() {
        // 2^15 amplitudes, the width at which the gather fans out to Rayon.
        let mut c = Circuit::new(15, 0);
        c.add_gate(Gate::Ry(0.3), &[0]);
        c.add_gate(Gate::Cx, &[0, 7]);
        c.add_gate(Gate::Rx(0.8), &[7]);
        c.add_gate(Gate::Cx, &[7, 14]);
        c.add_gate(Gate::Rz(0.5), &[14]);
        let params = Parameters::all_rotations(&c);

        let obs = vec![
            (1.0, vec![PauliTerm::z(0)]),
            (0.5, vec![PauliTerm::x(7)]),
            (-0.25, vec![PauliTerm::y(7), PauliTerm::z(14)]),
            (0.75, vec![PauliTerm::x(7), PauliTerm::z(0)]),
        ];
        let adjoint = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        let shift = run_expectation_gradient_shift(&c, &obs, &params, 42).unwrap();
        assert!((adjoint.value - shift.value).abs() < 1e-12);
        for (slot, (&got, &want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
            assert!((got - want).abs() < 1e-12, "slot {slot}: {got} vs {want}");
        }
    }

    #[test]
    fn the_gather_matches_a_term_by_term_reference() {
        // Sum the terms one at a time, the shape the gather replaces, and
        // compare on one register. Split across two chunks so the gather's base
        // arithmetic is covered as well.
        let dim = 1usize << 6;
        let phi: Vec<Complex64> = (0..dim)
            .map(|i| Complex64::new((i as f64 * 0.37).sin(), (i as f64 * 0.11).cos()))
            .collect();
        let masked = vec![
            (1.0, 0usize, 0b101usize, 0u32),
            (-0.5, 0b010, 0b100, 1),
            (0.75, 0b010, 0b001, 0),
            (0.25, 0b110, 0b011, 2),
        ];

        let mut want = vec![Complex64::new(0.0, 0.0); dim];
        for &(coeff, xmask, zmask, num_y) in &masked {
            let factor = Complex64::new(coeff, 0.0) * i_pow(num_y);
            for (j, &amp) in phi.iter().enumerate() {
                let sign = if (j & zmask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                want[j ^ xmask] += factor * sign * amp;
            }
        }
        let want_value: f64 = phi.iter().zip(&want).map(|(p, l)| (p.conj() * l).re).sum();

        let groups = group_terms_by_xmask(&masked);
        let mut got = vec![Complex64::new(0.0, 0.0); dim];
        let (lo, hi) = got.split_at_mut(dim / 2);
        let got_value = gather_lambda_chunk(&groups, &phi, 0, lo)
            + gather_lambda_chunk(&groups, &phi, dim / 2, hi);

        assert!((want_value - got_value).abs() < 1e-12);
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            assert!((w - g).norm() < 1e-12, "slot {i}: {w} vs {g}");
        }
    }

    #[test]
    fn pauli_rot_generators_commute_on_an_even_anticommuting_count() {
        use crate::sim::unified_pauli::PauliAxis;
        let gate = |axes: &[PauliAxis], targets: &[usize]| {
            Footprint::pauli(
                generator_masks(GeneratorKind::RotPauli(axes), targets),
                targets,
            )
        };
        let x0y1 = gate(&[PauliAxis::X, PauliAxis::Y], &[0, 1]);
        let y0x1 = gate(&[PauliAxis::Y, PauliAxis::X], &[0, 1]);
        let wide = gate(
            &[PauliAxis::Y, PauliAxis::X, PauliAxis::X, PauliAxis::X],
            &[0, 1, 2, 3],
        );
        let x1y2 = gate(&[PauliAxis::X, PauliAxis::Y], &[1, 2]);

        assert!(x0y1.commutes_with(&y0x1));
        assert!(x0y1.commutes_with(&x0y1));
        assert!(!wide.commutes_with(&x1y2));
        assert!(wide.commutes_with(&y0x1));
    }

    #[test]
    fn footprints_commute_through_control_and_target_actions() {
        let cx = Footprint::of(&Gate::Cx, &[0, 1]);
        let commutes = |gate: Gate, targets: &[usize]| {
            let f = Footprint::of(&gate, targets);
            assert_eq!(f.commutes_with(&cx), cx.commutes_with(&f));
            f.commutes_with(&cx)
        };
        assert!(commutes(Gate::Rz(0.3), &[0]));
        assert!(commutes(Gate::Rx(0.3), &[1]));
        assert!(commutes(Gate::Cz, &[0, 2]));
        assert!(commutes(Gate::H, &[2]));
        assert!(!commutes(Gate::Rz(0.3), &[1]));
        assert!(!commutes(Gate::Ry(0.3), &[0]));
        assert!(!commutes(Gate::Rzz(0.3), &[0, 1]));
        assert!(!commutes(Gate::H, &[0]));
        assert!(!commutes(Gate::Cx, &[1, 0]));
    }

    fn plan_for(circuit: &Circuit, params: &Parameters) -> SweepPlan {
        let mut linked: Vec<usize> = params.links().iter().map(|l| l.instruction).collect();
        linked.sort_unstable();
        let start = linked[0];
        let sweep: Vec<usize> = (start..circuit.instructions.len()).collect();
        let trainable: Vec<bool> = sweep
            .iter()
            .map(|i| linked.binary_search(i).is_ok())
            .collect();
        let footprints: Vec<Footprint> = sweep
            .iter()
            .map(|&i| {
                let Instruction::Gate { gate, targets } = &circuit.instructions[i] else {
                    unreachable!()
                };
                Footprint::of(gate, targets)
            })
            .collect();
        plan_sweep(&footprints, &trainable, circuit.num_qubits)
    }

    // Interleaved Ry and Rz on each qubit evaluate as one Rz step and one Ry
    // step per layer, where a run that must commute internally splits at every
    // qubit.
    #[test]
    fn a_hardware_efficient_layer_takes_two_steps() {
        let (n, layers) = (8, 3);
        let circuit = crate::circuits::hardware_efficient_ansatz(n, layers, 7);
        let plan = plan_for(&circuit, &Parameters::all_rotations(&circuit));
        assert_eq!(plan.evaluate.len(), 2 * n * layers);
        assert!(
            plan.last_step <= 2 * layers,
            "{} steps for {layers} layers",
            plan.last_step + 1
        );
        let stretches = plan
            .invert
            .iter()
            .map(|&(t, _)| t)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(stretches.len() <= 2 * layers);
    }

    #[test]
    fn a_qaoa_layer_takes_two_steps() {
        let circuit = crate::circuits::qaoa_circuit(8, 3, 7);
        let plan = plan_for(&circuit, &Parameters::all_rotations(&circuit));
        assert_eq!(plan.last_step, 5);
    }

    #[test]
    fn the_plan_never_inverts_the_earliest_trainable_gate_or_an_idle_one() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::Ry(0.4), &[0]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(0.7), &[1]);
        let mut params = Parameters::new(2);
        params.link(0, 0);
        params.link(3, 1);

        let plan = plan_for(&c, &params);
        let inverted: Vec<usize> = plan.invert.iter().map(|&(_, s)| s).collect();
        assert_eq!(inverted, [3, 2]);
        assert_eq!(plan.last_step, 1);
    }

    fn assert_adjoint_matches_shift(
        circuit: &Circuit,
        ham: &[(f64, Vec<PauliTerm>)],
        params: &Parameters,
    ) {
        let adjoint = run_expectation_gradient(circuit, ham, params, 42).unwrap();
        let shift = run_expectation_gradient_shift(circuit, ham, params, 42).unwrap();
        let n = circuit.num_qubits;
        assert!((adjoint.value - shift.value).abs() < 1e-10, "{n}q value");
        for (slot, (got, want)) in adjoint.gradient.iter().zip(&shift.gradient).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "{n}q slot {slot}: {got} vs {want}"
            );
        }
    }

    fn mixed_hamiltonian(n: usize) -> Vec<(f64, Vec<PauliTerm>)> {
        (0..n - 1)
            .map(|q| {
                (
                    0.5 + 0.1 * q as f64,
                    vec![PauliTerm::z(q), PauliTerm::z(q + 1)],
                )
            })
            .chain([
                (0.7, vec![PauliTerm::x(1)]),
                (-0.4, vec![PauliTerm::y(0), PauliTerm::x(n - 1)]),
            ])
            .collect()
    }

    // Widths on both sides of the fusion floors, so the inverted stretches run
    // both unfused and through the 1q, 2q, and batch passes.
    #[test]
    fn frontier_sweep_matches_parameter_shift_on_hea_and_qaoa() {
        for n in [5, 12, 16] {
            let ham = mixed_hamiltonian(n);
            let (hea, shared) = hea_with_shared_slots(n);
            assert_adjoint_matches_shift(&hea, &ham, &shared);
            assert_adjoint_matches_shift(&hea, &ham, &Parameters::all_rotations(&hea));

            let qaoa = crate::circuits::qaoa_circuit(n, 2, 11);
            let mut params = Parameters::new(4);
            for (k, link) in Parameters::all_rotations(&qaoa).links().iter().enumerate() {
                let layer = k / (2 * n - 1);
                let mixer = k % (2 * n - 1) >= n - 1;
                params.link(link.instruction, 2 * layer + usize::from(mixer));
            }
            assert_adjoint_matches_shift(&qaoa, &ham, &params);
        }
    }

    // Fixed gates of every commutation kind sit between trainable ones, some
    // trainable gates share a slot, one rotation stays fixed, and a fixed
    // prefix precedes the earliest trainable gate.
    #[test]
    fn frontier_sweep_matches_parameter_shift_with_fixed_gates_between() {
        use crate::sim::unified_pauli::PauliAxis;
        for n in [4, 12, 16] {
            let mut c = Circuit::new(n, 0);
            for q in 0..n {
                c.add_gate(Gate::H, &[q]);
            }
            for layer in 0..2 {
                for q in 0..n {
                    let angle = 0.3 + 0.17 * (q + layer * n) as f64;
                    c.add_gate(Gate::Ry(angle), &[q]);
                    c.add_gate(if q % 3 == 0 { Gate::T } else { Gate::SX }, &[q]);
                    c.add_gate(Gate::Rz(angle * 0.5), &[q]);
                }
                for q in (layer..n - 1).step_by(2) {
                    c.add_gate(Gate::Cx, &[q, q + 1]);
                    c.add_gate(Gate::Rzz(0.2 + 0.05 * q as f64), &[q, q + 1]);
                }
                c.add_gate(Gate::Cz, &[0, n - 1]);
                c.add_gate(Gate::Swap, &[1, 2]);
                c.add_gate(Gate::P(0.6), &[n - 1]);
                c.add_pauli_rotation(
                    0.45,
                    &[
                        PauliTerm::new(0, PauliAxis::X),
                        PauliTerm::new(2, PauliAxis::Y),
                        PauliTerm::new(3, PauliAxis::Z),
                    ],
                );
                c.add_gate(Gate::Rx(0.9), &[n / 2]);
            }

            let mut params = Parameters::new(5);
            let rotations = c
                .instructions
                .iter()
                .enumerate()
                .filter(|(_, inst)| {
                    matches!(inst, Instruction::Gate { gate, .. } if gate.pauli_generator().is_some())
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            for (k, &i) in rotations.iter().enumerate() {
                if k % 7 != 3 {
                    params.link(i, k % 5);
                }
            }
            assert_adjoint_matches_shift(&c, &mixed_hamiltonian(n), &params);
        }
    }

    #[test]
    fn a_full_light_cone_borrows_the_circuit_it_was_cut_from() {
        let mut c = Circuit::new(2, 1);
        c.add_gate(Gate::Rx(0.3), &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Ry(0.7), &[1]);

        let all: Vec<usize> = (0..c.instructions.len()).collect();
        assert!(matches!(kept_subcircuit(&c, &all), Cow::Borrowed(_)));

        let pruned = kept_subcircuit(&c, &[0, 2]);
        assert!(matches!(pruned, Cow::Owned(_)));
        assert_eq!(pruned.num_qubits, c.num_qubits);
        assert_eq!(pruned.num_classical_bits, c.num_classical_bits);
        let gates: Vec<&Gate> = pruned
            .instructions
            .iter()
            .map(|inst| {
                let Instruction::Gate { gate, .. } = inst else {
                    unreachable!()
                };
                gate
            })
            .collect();
        assert!(matches!(gates.as_slice(), [Gate::Rx(_), Gate::Ry(_)]));
    }
}
