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
use crate::circuit::parameter::{Parameters, angle_mut};
use crate::circuit::{Circuit, Instruction, SmallVec, smallvec};
use crate::error::{PrismError, Result};
use crate::gates::{
    BatchRzzData, DiagEntry, DiagonalBatchData, Gate, GeneratorKind, MultiFusedData,
    is_diagonal_2x2, pauli_rot_masks,
};

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
        return Err(PrismError::IncompatibleBackend {
            backend: "Statevector".into(),
            reason: format!(
                "adjoint gradients for {} qubits exceed the statevector cap ({} qubits); the gradient path holds two statevectors",
                circuit.num_qubits,
                max_statevector_qubits()
            ),
        });
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

    // In-cone links sorted by descending instruction index, matching the
    // reverse sweep. A cursor walks this list so the per-instruction lookup
    // stays O(params), not O(instructions). An out-of-cone trainable gate has
    // a provably zero gradient, so its links carry nothing to accumulate.
    let mut links: Vec<_> = params
        .links()
        .iter()
        .filter(|l| in_cone[l.instruction])
        .copied()
        .collect();
    links.sort_unstable_by_key(|l| std::cmp::Reverse(l.instruction));

    // The sweep stops at the earliest in-cone trainable gate: nothing before
    // it contributes, so a non-trainable prefix costs no inverse applications.
    // If no trainable gate reaches the observable, the gradient is zero
    // everywhere.
    let Some(earliest) = links.last().map(|l| l.instruction) else {
        return Ok(ExpectationGradient { value, gradient });
    };

    let mut lambda = StatevectorBackend::new(seed);
    lambda.init_from_state(lambda_state, circuit.num_classical_bits)?;

    // The sweep is the in-cone tail from the earliest trainable gate, walked
    // backwards. Its trainable gates group into commuting runs, which share
    // one state pair for their sandwiches and one fused pass for their
    // inverses; a non-trainable gate carries no contribution and splits runs.
    let sweep = &kept[kept.partition_point(|&i| i < earliest)..];
    let mut cursor = 0;
    let mut end = sweep.len();
    let mut run: Vec<RunGate> = Vec::new();
    let mut masks: Vec<(usize, usize, u32)> = Vec::new();

    while end > 0 {
        let i = sweep[end - 1];
        if cursor >= links.len() || links[cursor].instruction != i {
            let inverse = inverse_instruction(&circuit.instructions[i]);
            phi.apply(&inverse)?;
            lambda.apply(&inverse)?;
            end -= 1;
            continue;
        }

        run.clear();
        let mut start = end;
        let mut lookahead = cursor;
        while start > 0 {
            let index = sweep[start - 1];
            if lookahead >= links.len() || links[lookahead].instruction != index {
                break;
            }
            let Instruction::Gate { gate, targets } = &circuit.instructions[index] else {
                unreachable!("the light cone keeps gate instructions only")
            };
            let kind = gate
                .pauli_generator()
                .expect("trainable instruction validated as differentiable");
            let candidate = RunGate::new(index, kind, targets);
            if run.iter().any(|member| !member.commutes_with(&candidate)) {
                break;
            }
            while lookahead < links.len() && links[lookahead].instruction == index {
                lookahead += 1;
            }
            run.push(candidate);
            start -= 1;
        }

        masks.clear();
        masks.extend(run.iter().map(|g| (g.xmask, g.zmask, g.num_y)));
        let values = pauli_sandwiches_from_masks(lambda.state_vector(), phi.state_vector(), &masks);

        for (g, value) in run.iter().zip(&values) {
            while cursor < links.len() && links[cursor].instruction == g.index {
                gradient[links[cursor].slot] += value.im;
                cursor += 1;
            }
        }

        // The earliest in-cone trainable gate is the last one evaluated; its
        // inverse and every gate before it can be skipped.
        let applied = run.len() - usize::from(sweep[start] == earliest);
        for inverse in run_inverse_instructions(circuit, &run[..applied]) {
            phi.apply(&inverse)?;
            lambda.apply(&inverse)?;
        }
        end = start;
    }

    Ok(ExpectationGradient { value, gradient })
}

/// One trainable gate staged in a commuting run: its instruction index and the
/// Pauli masks of its generator.
struct RunGate {
    index: usize,
    xmask: usize,
    zmask: usize,
    num_y: u32,
}

impl RunGate {
    fn new(index: usize, kind: GeneratorKind<'_>, targets: &[usize]) -> Self {
        let (xmask, zmask, num_y) = match kind {
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
        };
        Self {
            index,
            xmask,
            zmask,
            num_y,
        }
    }

    /// Two Pauli strings commute exactly when they anticommute on an even
    /// number of qubits. `exp(-iθP/2)` then commutes with the other string as
    /// well, which is what lets a run share one state pair: conjugating a
    /// member's generator by the inverses of the members that follow it leaves
    /// the generator, so every sandwich in the run reads the same `⟨λ|` and
    /// `|φ⟩` as it would at its own position.
    fn commutes_with(&self, other: &RunGate) -> bool {
        let anticommuting =
            (self.xmask & other.zmask).count_ones() + (self.zmask & other.xmask).count_ones();
        anticommuting.is_multiple_of(2)
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

/// Inverses of a commuting run, collapsed into a batch gate where the run's
/// gate type has one: `MultiFused` for single-qubit rotations on distinct
/// qubits, `BatchRzz` for an Rzz layer, `DiagonalBatch` for a mixed diagonal
/// run. Anything else falls back to one instruction per gate. Order within a
/// run is free because its gates commute.
fn run_inverse_instructions(circuit: &Circuit, run: &[RunGate]) -> Vec<Instruction> {
    let gates: Vec<(&Gate, &[usize])> = run
        .iter()
        .map(|g| {
            let Instruction::Gate { gate, targets } = &circuit.instructions[g.index] else {
                unreachable!("the light cone keeps gate instructions only")
            };
            (gate, targets.as_slice())
        })
        .collect();

    if gates.len() > 1 {
        if let Some(fused) = multi_fused_inverse(&gates) {
            return vec![fused];
        }
        if let Some(batched) = batch_rzz_inverse(&gates) {
            return batched;
        }
        if let Some(diagonal) = diagonal_batch_inverse(&gates) {
            return vec![diagonal];
        }
    }

    gates
        .iter()
        .map(|&(gate, targets)| Instruction::Gate {
            gate: gate.inverse(),
            targets: targets.iter().copied().collect(),
        })
        .collect()
}

fn multi_fused_inverse(gates: &[(&Gate, &[usize])]) -> Option<Instruction> {
    let mut fused: Vec<(usize, [[Complex64; 2]; 2])> = Vec::with_capacity(gates.len());
    for &(gate, targets) in gates {
        if !matches!(gate, Gate::Rx(_) | Gate::Ry(_) | Gate::Rz(_) | Gate::P(_))
            || fused.iter().any(|&(q, _)| q == targets[0])
        {
            return None;
        }
        fused.push((targets[0], gate.inverse().matrix_2x2()));
    }
    let all_diagonal = fused.iter().all(|(_, mat)| is_diagonal_2x2(mat));
    let targets: SmallVec<[usize; 4]> = fused.iter().map(|&(q, _)| q).collect();
    Some(Instruction::Gate {
        gate: Gate::MultiFused(Box::new(MultiFusedData {
            gates: fused,
            all_diagonal,
        })),
        targets,
    })
}

fn batch_rzz_inverse(gates: &[(&Gate, &[usize])]) -> Option<Vec<Instruction>> {
    let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(gates.len());
    for &(gate, targets) in gates {
        let Gate::Rzz(theta) = gate else {
            return None;
        };
        edges.push((targets[0], targets[1], -theta));
    }
    Some(
        edges
            .chunks(BatchRzzData::MAX_EDGES)
            .map(|chunk| match chunk {
                [(q0, q1, theta)] => Instruction::Gate {
                    gate: Gate::Rzz(*theta),
                    targets: smallvec![*q0, *q1],
                },
                _ => {
                    let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
                    for &(q0, q1, _) in chunk {
                        for q in [q0, q1] {
                            if !targets.contains(&q) {
                                targets.push(q);
                            }
                        }
                    }
                    Instruction::Gate {
                        gate: Gate::BatchRzz(Box::new(BatchRzzData {
                            edges: chunk.to_vec(),
                        })),
                        targets,
                    }
                }
            })
            .collect(),
    )
}

/// A run of diagonal rotations collapsed into one `DiagonalBatch` sweep. The
/// kernel has no entry cap: a payload whose connected components outgrow the
/// phase tables falls back to a per-element pass, still one traversal.
fn diagonal_batch_inverse(gates: &[(&Gate, &[usize])]) -> Option<Instruction> {
    let mut entries: Vec<DiagEntry> = Vec::with_capacity(gates.len());
    let mut targets: SmallVec<[usize; 4]> = SmallVec::new();
    for &(gate, gate_targets) in gates {
        if !matches!(gate, Gate::Rz(_) | Gate::P(_) | Gate::Rzz(_)) {
            return None;
        }
        entries.extend(gate.inverse().diag_entries(gate_targets));
        for &q in gate_targets {
            if !targets.contains(&q) {
                targets.push(q);
            }
        }
    }
    targets.sort_unstable();
    Some(Instruction::Gate {
        gate: Gate::DiagonalBatch(Box::new(DiagonalBatchData { entries })),
        targets,
    })
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
/// count of its own: it holds one backend state at a time and inherits whatever
/// the selected backend can represent. It also accepts `QftBlock`. The price is
/// `1 + 2 * params.links().len()` circuit evaluations against the adjoint's
/// one, so prefer the adjoint wherever it applies. Select an explicit backend
/// with [`crate::simulate`] and `expectation_gradient_shift`.
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
    let mut shifted = circuit.clone();
    for link in params.links() {
        let base = *angle_mut(&mut shifted.instructions[link.instruction]);
        *angle_mut(&mut shifted.instructions[link.instruction]) = base + shift;
        let plus = evaluate(&shifted)?;
        *angle_mut(&mut shifted.instructions[link.instruction]) = base - shift;
        let minus = evaluate(&shifted)?;
        *angle_mut(&mut shifted.instructions[link.instruction]) = base;
        gradient[link.slot] += 0.5 * (plus - minus);
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
        // 2^14 amplitudes, the width at which the gather fans out to Rayon.
        let mut c = Circuit::new(14, 0);
        c.add_gate(Gate::Ry(0.3), &[0]);
        c.add_gate(Gate::Cx, &[0, 7]);
        c.add_gate(Gate::Rx(0.8), &[7]);
        c.add_gate(Gate::Cx, &[7, 13]);
        c.add_gate(Gate::Rz(0.5), &[13]);
        let params = Parameters::all_rotations(&c);

        let obs = vec![
            (1.0, vec![PauliTerm::z(0)]),
            (0.5, vec![PauliTerm::x(7)]),
            (-0.25, vec![PauliTerm::y(7), PauliTerm::z(13)]),
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
        let gate = |index: usize, axes: &[PauliAxis], targets: &[usize]| {
            RunGate::new(index, GeneratorKind::RotPauli(axes), targets)
        };
        let x0y1 = gate(0, &[PauliAxis::X, PauliAxis::Y], &[0, 1]);
        let y0x1 = gate(1, &[PauliAxis::Y, PauliAxis::X], &[0, 1]);
        let wide = gate(
            2,
            &[PauliAxis::Y, PauliAxis::X, PauliAxis::X, PauliAxis::X],
            &[0, 1, 2, 3],
        );
        let x1y2 = gate(3, &[PauliAxis::X, PauliAxis::Y], &[1, 2]);

        assert!(x0y1.commutes_with(&y0x1));
        assert!(x0y1.commutes_with(&x0y1));
        assert!(!wide.commutes_with(&x1y2));
        assert!(wide.commutes_with(&y0x1));
    }

    #[test]
    fn a_run_collapses_to_one_batch_gate_only_when_its_type_has_one() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(Gate::Rz(0.3), &[0]);
        c.add_gate(Gate::Rz(0.5), &[1]);
        c.add_gate(Gate::Rz(0.7), &[0]);
        c.add_gate(Gate::Rzz(0.9), &[0, 1]);
        c.add_gate(Gate::Rzz(1.1), &[2, 3]);
        c.add_gate(Gate::Rx(1.3), &[0]);

        let run = |indices: &[usize]| -> Vec<RunGate> {
            indices
                .iter()
                .map(|&i| {
                    let Instruction::Gate { gate, targets } = &c.instructions[i] else {
                        unreachable!()
                    };
                    RunGate::new(i, gate.pauli_generator().unwrap(), targets)
                })
                .collect()
        };

        let distinct = run_inverse_instructions(&c, &run(&[1, 0]));
        assert!(matches!(
            distinct.as_slice(),
            [Instruction::Gate {
                gate: Gate::MultiFused(_),
                ..
            }]
        ));

        let rzz_layer = run_inverse_instructions(&c, &run(&[4, 3]));
        assert!(matches!(
            rzz_layer.as_slice(),
            [Instruction::Gate {
                gate: Gate::BatchRzz(_),
                ..
            }]
        ));

        let repeated_qubit = run_inverse_instructions(&c, &run(&[2, 0]));
        assert!(matches!(
            repeated_qubit.as_slice(),
            [Instruction::Gate {
                gate: Gate::DiagonalBatch(_),
                ..
            }]
        ));

        let non_diagonal = run_inverse_instructions(&c, &run(&[5, 4]));
        assert_eq!(non_diagonal.len(), 2);
    }

    #[test]
    fn a_mixed_diagonal_run_collapses_into_one_diagonal_batch() {
        // The Rz layer and the Rzz chain of an Ising ansatz merge into one run:
        // every generator is Z type, so nothing splits them.
        let n = 6;
        let mut c = Circuit::new(n, 0);
        for q in 0..n - 1 {
            c.add_gate(Gate::Rzz(0.31 + 0.07 * q as f64), &[q, q + 1]);
        }
        for q in 0..3 {
            c.add_gate(Gate::Rz(0.4 + 0.11 * q as f64), &[q]);
        }

        let run: Vec<RunGate> = (0..c.instructions.len())
            .rev()
            .map(|i| {
                let Instruction::Gate { gate, targets } = &c.instructions[i] else {
                    unreachable!()
                };
                RunGate::new(i, gate.pauli_generator().unwrap(), targets)
            })
            .collect();

        let inverses = run_inverse_instructions(&c, &run);
        let [
            Instruction::Gate {
                gate: Gate::DiagonalBatch(data),
                targets,
            },
        ] = inverses.as_slice()
        else {
            panic!("expected one DiagonalBatch, got {inverses:?}")
        };
        assert_eq!(data.entries.len(), 8);
        assert_eq!(targets.as_slice(), &[0, 1, 2, 3, 4, 5]);
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
