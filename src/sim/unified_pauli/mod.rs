//! Pauli propagation engines for circuits of Clifford gates and Pauli
//! rotations: SPP samples backward Heisenberg propagation stochastically, SPD
//! carries the full weighted Pauli sum with optional truncation, and the Pauli
//! path engine carries the same sum through a noise model. None materializes a
//! state vector.
//!
//! Z-axis rotations (`T`, `Tdg`, `Rz`, `P`) and the two-qubit `Rzz` branch
//! natively, as do `Rx` and `Ry` in SPD and the Pauli path engine; SPP's `Rx`
//! and `Ry`, and multi-qubit `PauliRot` strings everywhere, lower to Clifford
//! conjugation around one `Rz` before the run, a `Fused` matrix lowers to the
//! named gate it equals up to phase or to its ZYZ Euler triple, and a `Cu` with
//! a diagonal or Pauli target lowers to Z rotations and Cliffords.

use num_complex::Complex64;

use std::borrow::Cow;

use crate::circuit::clifford_t::{is_pauli_native, lower_to_pauli_forms};
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::PauliVec;

mod pauli_path;
mod spd;
mod spp;

pub(crate) use pauli_path::run_pauli_path_weighted_observable;
pub use pauli_path::{PauliPathResult, run_pauli_path_observable};
pub(super) use spd::propagate_register_rotations;
pub use spd::{
    SpdObservableResult, SpdResult, SpdTruncation, inverse_light_cone, run_spd, run_spd_observable,
    run_spd_observable_budgeted, run_spd_observable_light_cone, run_spd_with,
};
pub(crate) use spd::{
    run_spd_exact_within, run_spd_observable_with, run_spd_weighted_observable_with,
};
pub(crate) use spp::Welford;
pub use spp::{SppObservableResult, SppResult, run_spp, run_spp_observable};

/// Absolute ceiling on the weighted-Pauli term count, enforced even in exact
/// mode (`max_terms == 0`). Without a per-step truncation budget the term set
/// grows as `2^(in-cone branching rotations)`; this caps transient memory the
/// same way the stabilizer-rank backend does. Exceeding it is an error, not a
/// silent truncation: callers wanting bounded approximate evaluation pass a
/// nonzero `max_terms` instead.
const SPD_MAX_TERMS_CEILING: usize = 1 << 20;

#[inline]
fn check_spd_term_ceiling(len: usize, backend: &str) -> Result<()> {
    if len > SPD_MAX_TERMS_CEILING {
        return Err(spd_term_ceiling_error(backend));
    }
    Ok(())
}

fn spd_term_ceiling_error(backend: &str) -> PrismError {
    PrismError::BackendUnsupported {
        backend: backend.into(),
        operation: format!(
            "weighted-Pauli sum exceeded {SPD_MAX_TERMS_CEILING} terms; pass a nonzero \
             max_terms for bounded approximate truncation"
        ),
    }
}

/// Rotation angle of the Z-axis rotation a gate lowers to, if any.
///
/// `P(λ)` and `Rz(λ)` differ by the global phase `e^{-iλ/2}`, which conjugation
/// discards, so both report `λ`. Clifford Z rotations (`Z`, `S`, `Sdg`) are
/// deliberately absent: they conjugate exactly through the Clifford path and
/// would otherwise branch into a term with a rounding-noise coefficient.
#[inline]
fn z_rotation_angle(gate: &Gate) -> Option<f64> {
    match gate {
        Gate::T => Some(std::f64::consts::FRAC_PI_4),
        Gate::Tdg => Some(-std::f64::consts::FRAC_PI_4),
        Gate::Rz(theta) | Gate::P(theta) => Some(*theta),
        _ => None,
    }
}

#[inline]
fn zz_rotation_angle(gate: &Gate) -> Option<f64> {
    match gate {
        Gate::Rzz(theta) => Some(*theta),
        _ => None,
    }
}

/// Validate the circuit and lower every gate `native` rejects in one walk.
///
/// Multi-qubit `PauliRot` strings, and `Rx` and `Ry` unless `native` keeps them,
/// expand through the shared CNOT-ladder lowering into Clifford conjugation
/// around one `Rz`, and `Fused` matrices and `Cu` gates take the forms
/// `lower_to_pauli_forms` documents; `Rzz` stays native and branches directly on
/// the `Z⊗Z` anticommutation test. The lowering tests sit on the walk's
/// rejection path, so a circuit needing none pays no second scan and borrows
/// through unchanged.
fn validate_and_lower<'c>(
    circuit: &'c Circuit,
    backend: &'static str,
    native: fn(&Gate) -> bool,
) -> Result<Cow<'c, Circuit>> {
    let reject = |operation: String| PrismError::BackendUnsupported {
        backend: backend.to_string(),
        operation,
    };
    let mut needs_lowering = false;
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => {
                if native(gate) {
                    continue;
                }
                lower_to_pauli_forms(gate, targets, &mut |_, _| {}).map_err(reject)?;
                needs_lowering = true;
            }
            Instruction::Barrier { .. } => {}
            Instruction::Save { .. }
            | Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. }
            | Instruction::Region(_) => {
                return Err(PrismError::IncompatibleBackend {
                    backend: backend.to_string(),
                    reason: "Pauli propagation requires a unitary circuit without measurements, \
                         resets, or conditionals"
                        .to_string(),
                });
            }
        }
    }
    if !needs_lowering {
        return Ok(Cow::Borrowed(circuit));
    }

    let mut out: Vec<Instruction> = Vec::with_capacity(circuit.instructions.len() * 2);
    for inst in &circuit.instructions {
        let Instruction::Gate { gate, targets } = inst else {
            out.push(inst.clone());
            continue;
        };
        if native(gate) {
            out.push(inst.clone());
            continue;
        }
        lower_to_pauli_forms(gate, targets, &mut |gate, tgts| {
            out.push(Instruction::Gate {
                gate,
                targets: SmallVec::from_slice(tgts),
            });
        })
        .map_err(reject)?;
    }
    Ok(Cow::Owned(circuit.with_instructions(out)))
}

/// Gates the deterministic engines take without lowering: the shared native set
/// plus `Rx` and `Ry`, which branch on their own axis.
fn is_spd_native(gate: &Gate) -> bool {
    is_pauli_native(gate) || matches!(gate, Gate::Rx(_) | Gate::Ry(_))
}

fn count_branching_gates(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| match inst {
            Instruction::Gate { gate, .. } => {
                z_rotation_angle(gate).is_some()
                    || zz_rotation_angle(gate).is_some()
                    || matches!(gate, Gate::Rx(_) | Gate::Ry(_))
            }
            _ => false,
        })
        .count()
}

/// Pauli axis for a joint-observable term. Identity factors are omitted
/// from the term list and contribute trivially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PauliAxis {
    X,
    Y,
    Z,
}

impl PauliAxis {
    /// Uppercase letter naming the axis, the spelling Pauli strings carry.
    pub fn letter(self) -> char {
        match self {
            PauliAxis::X => 'X',
            PauliAxis::Y => 'Y',
            PauliAxis::Z => 'Z',
        }
    }

    /// Inverse of [`letter`](Self::letter), accepting either case.
    pub fn from_letter(letter: char) -> Option<Self> {
        match letter.to_ascii_uppercase() {
            'X' => Some(PauliAxis::X),
            'Y' => Some(PauliAxis::Y),
            'Z' => Some(PauliAxis::Z),
            _ => None,
        }
    }
}

/// One non-identity factor of a joint Pauli observable. Ordered by qubit,
/// then axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PauliTerm {
    pub qubit: usize,
    pub axis: PauliAxis,
}

impl PauliTerm {
    pub fn new(qubit: usize, axis: PauliAxis) -> Self {
        Self { qubit, axis }
    }

    pub fn x(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::X)
    }

    pub fn y(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::Y)
    }

    pub fn z(qubit: usize) -> Self {
        Self::new(qubit, PauliAxis::Z)
    }
}

fn pauli_vec_from_terms(
    num_qubits: usize,
    terms: &[PauliTerm],
) -> std::result::Result<(PauliVec, Complex64), crate::error::PrismError> {
    let num_words = num_qubits.div_ceil(64);
    let mut pv = PauliVec::new(num_words);
    let mut coeff = Complex64::new(1.0, 0.0);
    let mut seen = vec![false; num_qubits];
    for term in terms {
        if term.qubit >= num_qubits {
            return Err(crate::error::PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(crate::error::PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
        match term.axis {
            PauliAxis::X => {
                pv.x[term.qubit / 64] |= 1u64 << (term.qubit % 64);
            }
            PauliAxis::Z => {
                pv.z[term.qubit / 64] |= 1u64 << (term.qubit % 64);
            }
            PauliAxis::Y => {
                pv.x[term.qubit / 64] |= 1u64 << (term.qubit % 64);
                pv.z[term.qubit / 64] |= 1u64 << (term.qubit % 64);
                coeff *= Complex64::new(0.0, 1.0);
            }
        }
    }
    Ok((pv, coeff))
}

#[inline(always)]
fn pv_get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
fn clifford_conjugation_phase(gate: &Gate, targets: &[usize], x: &[u64], z: &[u64]) -> Complex64 {
    match gate {
        Gate::H => {
            let q = targets[0];
            if pv_get_bit(x, q) && pv_get_bit(z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::S => {
            let q = targets[0];
            if pv_get_bit(x, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Sdg => {
            let q = targets[0];
            if pv_get_bit(x, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::X => {
            let q = targets[0];
            if pv_get_bit(z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Y => {
            let q = targets[0];
            let xq = pv_get_bit(x, q);
            let zq = pv_get_bit(z, q);
            if xq ^ zq {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Z => {
            let q = targets[0];
            if pv_get_bit(x, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SX => {
            let q = targets[0];
            if pv_get_bit(z, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SXdg => {
            let q = targets[0];
            if pv_get_bit(z, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            if pv_get_bit(x, q0) && pv_get_bit(x, q1) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        _ => Complex64::new(1.0, 0.0),
    }
}

#[cfg(test)]
mod tests;
