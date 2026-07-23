//! Adjoint-method gradients of expectation values.
//!
//! Computes the exact gradient of `⟨H⟩ = ⟨0|U†HU|0⟩` with respect to a vector
//! of circuit parameters in a cost independent of the parameter count, by
//! back-propagating two statevectors through the circuit. For `U = U_L…U_1`
//! and a Hermitian Hamiltonian `H = Σ c_k P_k`:
//!
//! 1. Forward: apply `U` unfused, keep `|φ⟩ = U|0⟩`.
//! 2. Build `|λ⟩ = H|φ⟩`; the value `⟨H⟩ = Re⟨φ|λ⟩`.
//! 3. Backward `i = L…1`: for each trainable gate with generator `G_i`,
//!    accumulate the gradient from `⟨λ|G_i|φ⟩` (with `|φ⟩` on the output side
//!    of gate `i`), then step both states back through `U_i†`.
//!
//! Only the statevector backend is supported. The differentiated circuit must
//! be unitary (no measurement, reset, or conditional). Differentiable gates are
//! `Rx`, `Ry`, `Rz`, `Rzz`, and `P`; a trainable link on any other gate is an
//! error.

use num_complex::Complex64;

use crate::backend::statevector::StatevectorBackend;
use crate::backend::{Backend, max_statevector_qubits, reserve_dense_output};
use crate::circuit::{Circuit, Instruction};
use crate::error::{PrismError, Result};
use crate::gates::{Gate, GeneratorKind};

use super::unified_pauli::PauliTerm;
use super::{i_pow, pauli_masks, pauli_sandwich};

/// Binds one trainable gate instruction to a slot in the parameter vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamLink {
    /// Index into [`Circuit::instructions`].
    pub instruction: usize,
    /// Slot in the parameter (theta) vector this gate feeds.
    pub param: usize,
}

/// Maps trainable gate instructions to parameter-vector slots for the adjoint
/// gradient. Many instructions may share one slot (weight sharing); their
/// contributions accumulate. Built by the parametric [`crate::CircuitBuilder`]
/// methods or constructed directly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParameterMap {
    links: Vec<ParamLink>,
}

impl ParameterMap {
    /// An empty map (no trainable parameters).
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    /// Build a map from explicit instruction-to-slot links.
    pub fn from_links(links: Vec<ParamLink>) -> Self {
        Self { links }
    }

    /// Record that `instruction` feeds parameter slot `param`.
    pub fn push(&mut self, instruction: usize, param: usize) {
        self.links.push(ParamLink { instruction, param });
    }

    /// The recorded links.
    pub fn links(&self) -> &[ParamLink] {
        &self.links
    }

    /// True if no trainable parameters are recorded.
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Number of distinct parameters, `max slot + 1` (0 if empty). The returned
    /// gradient vector has this length.
    pub fn num_params(&self) -> usize {
        self.links.iter().map(|l| l.param + 1).max().unwrap_or(0)
    }

    fn validate(&self, circuit: &Circuit) -> Result<()> {
        let n = circuit.instructions.len();
        for link in &self.links {
            if link.instruction >= n {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "parameter link references instruction {} but the circuit has {n} instructions",
                        link.instruction
                    ),
                });
            }
            match &circuit.instructions[link.instruction] {
                Instruction::Gate { gate, .. } if gate.pauli_generator().is_some() => {}
                Instruction::Gate { gate, .. } => {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "instruction {} (`{}`) is not analytically differentiable; supported gates are rx, ry, rz, rzz, p",
                            link.instruction,
                            gate.name()
                        ),
                    });
                }
                _ => {
                    return Err(PrismError::InvalidParameter {
                        message: format!(
                            "parameter link references instruction {} which is not a gate",
                            link.instruction
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

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
/// gradient vector. The returned gradient has length `params.num_params()`.
pub fn run_expectation_gradient(
    circuit: &Circuit,
    hamiltonian: &[(f64, Vec<PauliTerm>)],
    params: &ParameterMap,
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

    let num_params = params.num_params();
    let mut gradient = vec![0.0; num_params];
    if params.is_empty() {
        return Ok(ExpectationGradient { value, gradient });
    }

    // Inverse light cone of the Hamiltonian: a trainable gate outside it has a
    // provably zero gradient (its generator commutes through the back-evolved
    // observable), so its sandwich is skipped.
    let in_cone = observable_light_cone(circuit, hamiltonian);

    let mut slots_of: Vec<Vec<usize>> = vec![Vec::new(); circuit.instructions.len()];
    for link in params.links() {
        slots_of[link.instruction].push(link.param);
    }

    // Stop the backward sweep at the earliest in-cone trainable gate: nothing
    // before it contributes, so a non-trainable (or out-of-cone) prefix costs
    // no inverse applications. If no trainable gate reaches the observable, the
    // gradient is zero everywhere.
    let earliest = (0..circuit.instructions.len()).find(|&i| in_cone[i] && !slots_of[i].is_empty());
    let Some(earliest) = earliest else {
        return Ok(ExpectationGradient { value, gradient });
    };

    let mut lambda = StatevectorBackend::new(seed);
    lambda.init_from_state(lambda_state, circuit.num_classical_bits)?;

    for i in (earliest..circuit.instructions.len()).rev() {
        let (gate, targets) = match &circuit.instructions[i] {
            Instruction::Gate { gate, targets } => (gate, targets),
            Instruction::Barrier { .. } => continue,
            _ => unreachable!("non-unitary instructions rejected above"),
        };

        if in_cone[i] && !slots_of[i].is_empty() {
            let kind = gate
                .pauli_generator()
                .expect("trainable instruction validated as differentiable");
            let contrib =
                gradient_contribution(kind, targets, lambda.state_vector(), phi.state_vector());
            for &slot in &slots_of[i] {
                gradient[slot] += contrib;
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
    kind: GeneratorKind,
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
    }
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
        let mut params = ParameterMap::new();
        params.push(0, 0);

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
        let mut params = ParameterMap::new();
        params.push(0, 0);

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
        let mut params = ParameterMap::new();
        params.push(1, 0);

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
        let mut params = ParameterMap::new();
        params.push(0, 0);
        params.push(1, 0);

        let obs = vec![(1.0, vec![PauliTerm::z(0)]), (1.0, vec![PauliTerm::z(1)])];
        let g = run_expectation_gradient(&c, &obs, &params, 42).unwrap();
        assert_eq!(g.gradient.len(), 1);
        assert!((g.gradient[0] - (-2.0 * theta.sin())).abs() < 1e-9);
    }

    #[test]
    fn empty_params_returns_value_only() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::Rx(0.5), &[0]);
        let g = run_expectation_gradient(&c, &z_obs(0), &ParameterMap::new(), 42).unwrap();
        assert!(g.gradient.is_empty());
        assert!((g.value - 0.5f64.cos()).abs() < 1e-12);
    }

    #[test]
    fn nondifferentiable_trainable_gate_is_rejected() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        let mut params = ParameterMap::new();
        params.push(0, 0);
        assert!(run_expectation_gradient(&c, &z_obs(0), &params, 42).is_err());
    }

    #[test]
    fn nonunitary_circuit_is_rejected() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::Rx(0.3), &[0]);
        c.add_measure(0, 0);
        assert!(run_expectation_gradient(&c, &z_obs(0), &ParameterMap::new(), 42).is_err());
    }
}
