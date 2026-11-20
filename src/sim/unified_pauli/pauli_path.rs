//! Pauli path engine: the SPD weighted sum carried backward through a noise model
//! as well as the circuit.

use super::spd::{
    InlinePauli, SpdKey, SpdTruncation, WeightedPauliSum, weighted_sum, with_spd_key,
};
use super::{PauliTerm, SPD_MAX_TERMS_CEILING, is_spd_native, spd_term_ceiling_error};
use crate::circuit::clifford_t::lower_to_pauli_forms;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{PauliVec, flip_bit, get_bit};
use crate::sim::noise::{NoiseChannel, NoiseModel, thermal_relaxation_rates};

/// Backward action of one noise channel on a weighted Pauli sum.
///
/// A unital Pauli channel scales each letter by its own Pauli eigenvalue.
/// Amplitude damping is not unital: its adjoint sends `Z` to
/// `(1 - gamma) Z + gamma I`, and the sum carries that identity branch as a
/// second term rather than dropping it, so the engine stays exact on the
/// channel rather than on its unital part.
#[derive(Clone, Copy)]
enum ChannelAction {
    /// Eigenvalues for `X`, `Y`, `Z` on one qubit, plus the identity weight the
    /// `Z` letter picks up.
    Local {
        lambda: [f64; 3],
        z_to_identity: f64,
    },
    /// One eigenvalue for every restriction to the pair that is not `I (x) I`.
    Depolarizing2q { lambda: f64 },
}

/// The backward action of `channel`, or `None` for a channel with no
/// Pauli-basis form.
fn channel_action(channel: &NoiseChannel) -> Option<ChannelAction> {
    let damping = |gamma: f64| {
        let keep = (1.0 - gamma).max(0.0);
        ChannelAction::Local {
            lambda: [keep.sqrt(), keep.sqrt(), keep],
            z_to_identity: gamma,
        }
    };
    match channel {
        NoiseChannel::Pauli { px, py, pz } => {
            let pi = 1.0 - px - py - pz;
            Some(ChannelAction::Local {
                lambda: [pi + px - py - pz, pi + py - px - pz, pi + pz - px - py],
                z_to_identity: 0.0,
            })
        }
        NoiseChannel::Depolarizing { p } => {
            let lambda = 1.0 - 4.0 * p / 3.0;
            Some(ChannelAction::Local {
                lambda: [lambda; 3],
                z_to_identity: 0.0,
            })
        }
        NoiseChannel::PhaseDamping { gamma } => {
            let keep = (1.0 - gamma).max(0.0).sqrt();
            Some(ChannelAction::Local {
                lambda: [keep, keep, 1.0],
                z_to_identity: 0.0,
            })
        }
        NoiseChannel::AmplitudeDamping { gamma } => Some(damping(*gamma)),
        NoiseChannel::ThermalRelaxation {
            t1,
            t2,
            gate_time,
            excited_population,
        } => {
            let (gad, gpd) = thermal_relaxation_rates(*t1, *t2, *gate_time);
            let dephase = (1.0 - gpd).max(0.0).sqrt();
            let ChannelAction::Local { lambda, .. } = damping(gad) else {
                unreachable!("amplitude damping is a local action")
            };
            Some(ChannelAction::Local {
                lambda: [lambda[0] * dephase, lambda[1] * dephase, lambda[2]],
                // The identity the `Z` letter picks up is the steady-state
                // polarization, `1 - 2 * excited`, so it vanishes at the
                // maximally mixed steady state and inverts above it.
                z_to_identity: gad * (1.0 - 2.0 * *excited_population),
            })
        }
        NoiseChannel::TwoQubitDepolarizing { p } => Some(ChannelAction::Depolarizing2q {
            lambda: 1.0 - 16.0 * p / 15.0,
        }),
        NoiseChannel::Custom { .. }
        | NoiseChannel::Kraus2q { .. }
        | NoiseChannel::Leakage { .. }
        | NoiseChannel::Seepage { .. }
        | NoiseChannel::LeakageTransport { .. }
        | NoiseChannel::QuasiStatic { .. } => None,
    }
}

/// Index into [`ChannelAction::Local`]'s eigenvalue triple for the letter on
/// `qubit`, or `None` for the identity letter.
#[inline]
fn pauli_letter<K: SpdKey>(pauli: &K, qubit: usize) -> Option<usize> {
    match (get_bit(pauli.x(), qubit), get_bit(pauli.z(), qubit)) {
        (false, false) => None,
        (true, false) => Some(0),
        (true, true) => Some(1),
        (false, true) => Some(2),
    }
}

impl<K: SpdKey> WeightedPauliSum<K> {
    /// Backward-apply one noise channel.
    ///
    /// The unital part scales coefficients in place. The amplitude-damping
    /// identity branch merges into a term already present or appends one.
    fn apply_backward_channel(&mut self, action: &ChannelAction, qubits: &[usize]) {
        match *action {
            ChannelAction::Local {
                lambda,
                z_to_identity,
            } => {
                let q = qubits[0];
                let len = self.terms.len();
                if z_to_identity != 0.0 {
                    self.add_identity_branches(q, z_to_identity);
                }
                for (pauli, coeff) in self.terms[..len].iter_mut() {
                    if let Some(letter) = pauli_letter(pauli, q) {
                        *coeff *= lambda[letter];
                    }
                }
            }
            ChannelAction::Depolarizing2q { lambda } => {
                let (a, b) = (qubits[0], qubits[1]);
                for (pauli, coeff) in self.terms.iter_mut() {
                    if pauli_letter(pauli, a).is_some() || pauli_letter(pauli, b).is_some() {
                        *coeff *= lambda;
                    }
                }
            }
        }
    }

    /// Add `weight` times the identity branch of every term with a `Z` on
    /// `qubit`, from coefficients not yet scaled by the channel. The branch
    /// clears the qubit's z bit, so it can land only on a term with the identity
    /// letter there; those are probed against the `Z` terms.
    fn add_identity_branches(&mut self, qubit: usize, weight: f64) {
        self.in_support.clear();
        self.in_support.extend(
            self.terms
                .iter()
                .enumerate()
                .filter(|(_, (pauli, _))| pauli_letter(pauli, qubit) == Some(2))
                .map(|(i, _)| i),
        );
        if self.in_support.is_empty() {
            return;
        }
        self.partner.clear();
        self.partner.resize(self.in_support.len(), usize::MAX);
        self.index.clear();
        self.index.reserve(self.in_support.len());
        for (k, &i) in self.in_support.iter().enumerate() {
            self.index.insert(self.terms[i].0.clone(), k);
        }
        for (i, (pauli, _)) in self.terms.iter().enumerate() {
            if pauli_letter(pauli, qubit).is_none() {
                self.probe.clone_from(pauli);
                flip_bit(self.probe.words_mut().1, qubit);
                if let Some(&k) = self.index.get(&self.probe) {
                    self.partner[k] = i;
                }
            }
        }
        for k in 0..self.in_support.len() {
            let i = self.in_support[k];
            let branch = self.terms[i].1 * weight;
            match self.partner[k] {
                usize::MAX => {
                    let mut identity = self.terms[i].0.clone();
                    flip_bit(identity.words_mut().1, qubit);
                    self.terms.push((identity, branch));
                }
                j => self.terms[j].1 += branch,
            }
        }
    }
}

/// One step of the backward walk: a gate to conjugate through, or a channel to
/// apply.
enum PathOp {
    Gate(Gate, SmallVec<[usize; 4]>),
    Channel(ChannelAction, SmallVec<[usize; 2]>),
}

/// Flatten `circuit` and `noise` into the ordered step list the backward walk
/// consumes.
///
/// Lowering a `PauliRot` string or a `Fused` matrix rewrites one instruction
/// into several, so the noise events cannot stay indexed by circuit position.
/// Emitting a gate's own channels directly after it sidesteps the remapping.
fn pauli_path_ops(circuit: &Circuit, noise: &NoiseModel, backend: &str) -> Result<Vec<PathOp>> {
    if noise.readout.iter().any(|r| r.is_some()) {
        return Err(PrismError::IncompatibleBackend {
            backend: backend.to_string(),
            reason: "readout error has no Pauli-basis form on an expectation value; use \
                     BackendKind::DensityMatrix"
                .to_string(),
        });
    }
    let mut ops = Vec::with_capacity(circuit.instructions.len() * 2);
    for (index, inst) in circuit.instructions.iter().enumerate() {
        match inst {
            Instruction::Gate { gate, targets } if is_spd_native(gate) => {
                ops.push(PathOp::Gate(gate.clone(), targets.clone()));
            }
            Instruction::Gate { gate, targets } => {
                lower_to_pauli_forms(gate, targets, &mut |g, tgts| {
                    ops.push(PathOp::Gate(g, SmallVec::from_slice(tgts)));
                })
                .map_err(|operation| PrismError::BackendUnsupported {
                    backend: backend.to_string(),
                    operation,
                })?;
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
        for event in noise.after_gate.get(index).into_iter().flatten() {
            let Some(action) = channel_action(&event.channel) else {
                return Err(PrismError::BackendUnsupported {
                    backend: backend.to_string(),
                    operation: format!(
                        "channel `{:?}` has no Pauli-basis form; use BackendKind::DensityMatrix",
                        event.channel
                    ),
                });
            };
            ops.push(PathOp::Channel(action, event.qubits.clone()));
        }
    }
    Ok(ops)
}

/// Result of one Pauli path evaluation.
#[derive(Debug, Clone)]
pub struct PauliPathResult {
    /// `Tr(rho P)` on the noisy evolution; exact when nothing was truncated.
    pub mean: f64,
    /// Peak size of the weighted Pauli sum during propagation.
    pub peak_terms: usize,
    /// Total coefficient magnitude discarded by truncation (0 for exact runs).
    pub total_discarded: f64,
}

/// `Tr(rho P)` for a joint Pauli observable on a noisy circuit, by backward
/// Pauli propagation with coefficient truncation.
///
/// The observable propagates back through the circuit as a weighted Pauli sum:
/// Clifford gates conjugate it, `Rz`, `Rzz`, `Rx`, `Ry` and `PauliRot` split
/// each anticommuting term in two, and every noise channel scales the terms by the
/// channel's Pauli eigenvalues. Noise shrinks coefficients while rotations grow
/// the term count, so the sum stays small whenever noise outpaces the branching
/// rotations.
///
/// # Truncation
///
/// With `max_terms == 0` the run is exact and errors at the shared term
/// ceiling. With `max_terms > 0`, terms below `epsilon` are dropped once the sum
/// exceeds the budget and `total_discarded` bounds the error: every channel and
/// every Clifford conjugation is a contraction in the Pauli 1-norm, so a dropped
/// term's remaining contribution is at most its magnitude.
///
/// # Errors
///
/// Rejects a circuit carrying measurements, resets, or conditionals, a gate that
/// is neither Clifford nor a Pauli rotation, and a noise channel with no
/// Pauli-basis form (custom Kraus, two-qubit Kraus, readout error), naming the
/// density matrix as the route that serves those.
pub fn run_pauli_path_observable(
    circuit: &Circuit,
    noise: &NoiseModel,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<PauliPathResult> {
    run_pauli_path_weighted_observable(circuit, noise, &[(1.0, observable)], epsilon, max_terms)
}

/// [`run_pauli_path_observable`] on `Σ weight · P`, propagated as one sum, with
/// truncation and `total_discarded` applying to the merged sum. Falls back to
/// one propagation per term when the merged sum outgrows the term ceiling.
pub(crate) fn run_pauli_path_weighted_observable(
    circuit: &Circuit,
    noise: &NoiseModel,
    terms: &[(f64, &[PauliTerm])],
    epsilon: f64,
    max_terms: usize,
) -> Result<PauliPathResult> {
    let ops = pauli_path_ops(circuit, noise, "PauliPath")?;
    let truncation = SpdTruncation::Threshold { epsilon, max_terms };
    let n = circuit.num_qubits;

    let merged = |terms: &[(f64, &[PauliTerm])]| -> Result<Option<PauliPathResult>> {
        with_spd_key!(n, K => Ok(propagate_pauli_path(
            &ops,
            weighted_sum::<K>(n, terms)?,
            &truncation,
        )))
    };
    if let Some(result) = merged(terms)? {
        return Ok(result);
    }
    if terms.len() < 2 {
        return Err(spd_term_ceiling_error("PauliPath"));
    }
    let mut folded = PauliPathResult {
        mean: 0.0,
        peak_terms: 0,
        total_discarded: 0.0,
    };
    for &(weight, observable) in terms {
        let result =
            merged(&[(1.0, observable)])?.ok_or_else(|| spd_term_ceiling_error("PauliPath"))?;
        folded.mean += weight * result.mean;
        folded.peak_terms = folded.peak_terms.max(result.peak_terms);
        folded.total_discarded += weight.abs() * result.total_discarded;
    }
    Ok(folded)
}

fn propagate_pauli_path<K: SpdKey>(
    ops: &[PathOp],
    mut sum: WeightedPauliSum<K>,
    truncation: &SpdTruncation,
) -> Option<PauliPathResult> {
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for op in ops.iter().rev() {
        match op {
            PathOp::Gate(gate, targets) => sum.apply_backward(gate, targets),
            PathOp::Channel(action, qubits) => sum.apply_backward_channel(action, qubits),
        }
        truncation.enforce(&mut sum, &mut total_discarded);
        if sum.terms.len() > SPD_MAX_TERMS_CEILING {
            return None;
        }
        peak_terms = peak_terms.max(sum.terms.len());
    }

    truncation.final_sweep(&mut sum, &mut total_discarded);

    Some(PauliPathResult {
        mean: sum.diagonal_expectation(),
        peak_terms,
        total_discarded,
    })
}
