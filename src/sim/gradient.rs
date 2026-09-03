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

use num_complex::Complex64;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits, reserve_dense_output};
use crate::circuit::parameter::{Parameters, angle_mut};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::{Gate, GeneratorKind, pauli_rot_masks};

use super::noise::NoiseModel;
use super::unified_pauli::PauliTerm;
use super::{BackendKind, i_pow, pauli_masks, pauli_sandwich};

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

    // Validate and reduce observables before the 2^n simulation.
    let mut masked = Vec::with_capacity(hamiltonian.len());
    for (coeff, terms) in hamiltonian {
        let (xmask, zmask, num_y) = pauli_masks(terms, circuit.num_qubits)?;
        masked.push((*coeff, xmask, zmask, num_y));
    }

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

    // Forward pass, unfused, to keep a 1:1 gate-to-generator correspondence.
    let mut phi = StatevectorBackend::new(seed);
    phi.init(circuit.num_qubits, circuit.num_classical_bits)?;
    phi.apply_instructions(&circuit.instructions)?;

    let (value, lambda_state) = build_lambda_and_value(phi.state_vector(), &masked)?;

    let num_params = params.num_slots();
    let mut gradient = vec![0.0; num_params];
    if params.is_empty() {
        return Ok(ExpectationGradient { value, gradient });
    }

    // Inverse light cone of the Hamiltonian: a trainable gate outside it has a
    // provably zero gradient (its generator commutes through the back-evolved
    // observable), so its sandwich is skipped.
    let in_cone = observable_light_cone(circuit, hamiltonian);

    // Links sorted by descending instruction index, matching the reverse sweep.
    // A cursor walks this list so the per-instruction lookup stays O(params),
    // not O(instructions).
    let mut links = params.links().to_vec();
    links.sort_unstable_by_key(|l| std::cmp::Reverse(l.instruction));

    // Stop the backward sweep at the earliest in-cone trainable gate: nothing
    // before it contributes, so a non-trainable (or out-of-cone) prefix costs
    // no inverse applications. If no trainable gate reaches the observable, the
    // gradient is zero everywhere.
    let earliest = links
        .iter()
        .filter(|l| in_cone[l.instruction])
        .map(|l| l.instruction)
        .min();
    let Some(earliest) = earliest else {
        return Ok(ExpectationGradient { value, gradient });
    };

    let mut lambda = StatevectorBackend::new(seed);
    lambda.init_from_state(lambda_state, circuit.num_classical_bits)?;

    let mut cursor = 0;
    for i in (earliest..circuit.instructions.len()).rev() {
        let (gate, targets) = match &circuit.instructions[i] {
            Instruction::Gate { gate, targets } => (gate, targets),
            Instruction::Barrier { .. } => continue,
            _ => unreachable!("non-unitary instructions rejected above"),
        };

        if cursor < links.len() && links[cursor].instruction == i {
            if in_cone[i] {
                let kind = gate
                    .pauli_generator()
                    .expect("trainable instruction validated as differentiable");
                let contrib =
                    gradient_contribution(kind, targets, lambda.state_vector(), phi.state_vector());
                while cursor < links.len() && links[cursor].instruction == i {
                    gradient[links[cursor].slot] += contrib;
                    cursor += 1;
                }
            } else {
                // Out of the light cone: contribution is zero, but the cursor
                // still advances past this instruction's links.
                while cursor < links.len() && links[cursor].instruction == i {
                    cursor += 1;
                }
            }
        }

        // The earliest in-cone trainable gate is the last one evaluated; its
        // inverse and every gate before it can be skipped.
        if i > earliest {
            let inverse = Instruction::Gate {
                gate: gate.inverse(),
                targets: targets.clone(),
            };
            phi.apply(&inverse)?;
            lambda.apply(&inverse)?;
        }
    }

    Ok(ExpectationGradient { value, gradient })
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

/// Build `|λ⟩ = Σ c_k P_k|φ⟩` into a fresh buffer and return `(⟨H⟩, |λ⟩)`,
/// where `⟨H⟩ = Re⟨φ|λ⟩`.
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

    for &(coeff, xmask, zmask, num_y) in masked {
        let factor = Complex64::new(coeff, 0.0) * i_pow(num_y);
        for (j, &amp) in phi.iter().enumerate() {
            let sign = if (j & zmask).count_ones() & 1 == 1 {
                -1.0
            } else {
                1.0
            };
            lambda[j ^ xmask] += factor * sign * amp;
        }
    }

    let value: f64 = phi
        .iter()
        .zip(&lambda)
        .map(|(p, l)| (p.conj() * l).re)
        .sum();
    Ok((value, lambda))
}

/// Gradient contribution `d⟨H⟩/dθ` of a single trainable gate, from the
/// generator sandwich `⟨λ|G|φ⟩` with `|φ⟩` on the output side of the gate.
fn gradient_contribution(
    kind: GeneratorKind<'_>,
    targets: &[usize],
    lambda: &[Complex64],
    phi: &[Complex64],
) -> f64 {
    match kind {
        GeneratorKind::RotX => {
            let x = 1usize << targets[0];
            pauli_sandwich(lambda, phi, x, 0, 0).im
        }
        GeneratorKind::RotY => {
            let b = 1usize << targets[0];
            pauli_sandwich(lambda, phi, b, b, 1).im
        }
        GeneratorKind::RotZ => {
            let z = 1usize << targets[0];
            pauli_sandwich(lambda, phi, 0, z, 0).im
        }
        GeneratorKind::RotZz => {
            let z = (1usize << targets[0]) | (1usize << targets[1]);
            pauli_sandwich(lambda, phi, 0, z, 0).im
        }
        GeneratorKind::Phase => {
            let bit = 1usize << targets[0];
            let mut acc = Complex64::new(0.0, 0.0);
            for (j, &amp) in phi.iter().enumerate() {
                if j & bit != 0 {
                    acc += lambda[j].conj() * amp;
                }
            }
            -2.0 * acc.im
        }
        GeneratorKind::RotPauli(axes) => {
            let (xmask, zmask, num_y) = pauli_rot_masks(targets, axes);
            pauli_sandwich(lambda, phi, xmask, zmask, num_y).im
        }
    }
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
    initial_state: Option<&[Complex64]>,
    seed: u64,
) -> Result<ExpectationGradient> {
    params.validate(circuit)?;
    if initial_state.is_some() || noise.is_some() {
        super::require_unitary_circuit(kind, circuit)?;
    }

    let observables: Vec<Vec<PauliTerm>> =
        hamiltonian.iter().map(|(_, terms)| terms.clone()).collect();
    let evaluate = |c: &Circuit| -> Result<f64> {
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
}
