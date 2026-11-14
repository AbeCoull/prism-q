//! Stabilizer rank simulation for Clifford+T circuits.
//!
//! Pauli-offset representation: every branch of the T = αI + βZ expansion is
//! written as `branch_k = w_k · P_k · |ψ_0⟩`, where `|ψ_0⟩` is the result of
//! running every Clifford in the circuit on `|0…0⟩` with no Z insertions, and
//! `P_k` is a signed Pauli string accumulating the Zs that earlier T-positions
//! chose, each conjugated through the Cliffords that followed it
//! (Heisenberg picture: `U Z_q U†` is a signed Pauli for any Clifford `U`).
//!
//! A single shared [`StabilizerBackend`] evolves `|ψ_0⟩`; each branch is a
//! `(weight, SignedPauli)` pair. Accumulation reconstructs the statevector once
//! and routes amplitudes through each branch's Pauli, sidestepping the global
//! phase ambiguity inherent in per-branch tableau export.
//!
//! Two modes:
//! - **Exact probabilities** (n ≤ 25): read off the Clifford register, which sums
//!   the branches that land on one basis state of the qubits the T gates reach.
//!   A register wider than 20 qubits falls back to reconstructing `|ψ_0⟩` and
//!   summing weighted Pauli-shifted amplitudes.
//!   The approximate route cuts the same register back to a coefficient budget
//!   after each T gate and reports the exact norm each cut drops.
//! - **Measurement sampling** (any n): terminal measurements go through a
//!   Clifford tableau over a dense register of the qubits the T gates reach;
//!   otherwise keep coherent weighted MPS branches,
//!   project requested measurement outcomes, and contract pairwise branch
//!   overlaps without materializing a dense statevector.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::borrow::Cow;
use std::f64::consts::FRAC_PI_4;

use crate::backend::Backend;
use crate::backend::mps::MpsBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::clifford_t::lower_to_clifford_t;
use crate::circuit::{Circuit, GuardedRegion, Instruction, SmallVec, any_gate};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

use super::clifford_register::{MAX_REGISTER_QUBITS, Pruning, TerminalSampler, register_price};

/// Letter-level signed Pauli string. Each qubit's `(x, z)` bit pair encodes
/// the Pauli letter directly: (0,0)=I, (1,0)=X, (0,1)=Z, (1,1)=Y. `phase4` is
/// the extra `i^{phase4}` global factor multiplying the product, so `Y_q` is
/// stored as `x=1,z=1,phase4=0` (the `i` in `Y=iXZ` is baked into the letter
/// convention, not the global phase).
///
/// This shares its letter and `phase4` convention with
/// [`crate::qec::camps_prefix`]'s `SignedPauli`, but the two are kept
/// separate on purpose: this type tracks a single observable string with
/// dense `Vec<bool>` storage and forward `G·P·G†` conjugation, while the
/// CAMPS variant maintains a full inverse Clifford tableau with packed
/// `Vec<u64>` rows and `rowmul`. Any change to the (x,z)->letter or
/// `phase4` convention must be mirrored in both.
#[derive(Clone, Debug)]
struct SignedPauli {
    x: Vec<bool>,
    z: Vec<bool>,
    phase4: u8,
}

impl SignedPauli {
    fn identity(n: usize) -> Self {
        Self {
            x: vec![false; n],
            z: vec![false; n],
            phase4: 0,
        }
    }

    /// `P ← Z_q · P`. Letter table for (Z) · (letter): Z·I=Z, Z·X=iY,
    /// Z·Y=-iX, Z·Z=I.
    fn mul_z_on_left(&mut self, q: usize) {
        let xb = self.x[q];
        let zb = self.z[q];
        match (xb, zb) {
            (false, false) => {
                self.z[q] = true;
            }
            (true, false) => {
                self.z[q] = true;
                self.phase4 = (self.phase4 + 1) & 3;
            }
            (true, true) => {
                self.z[q] = false;
                self.phase4 = (self.phase4 + 3) & 3;
            }
            (false, true) => {
                self.z[q] = false;
            }
        }
    }

    /// `P ← G · P · G†` for supported Clifford gates.
    fn conjugate_by(&mut self, gate: &Gate, targets: &[usize]) -> Result<()> {
        match gate {
            Gate::Id => Ok(()),
            Gate::H => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                self.x[q] = zb;
                self.z[q] = xb;
                if xb && zb {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::S => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                match (xb, zb) {
                    (true, false) => {
                        self.z[q] = true;
                    }
                    (true, true) => {
                        self.z[q] = false;
                        self.phase4 = (self.phase4 + 2) & 3;
                    }
                    _ => {}
                }
                Ok(())
            }
            Gate::Sdg => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                match (xb, zb) {
                    (true, false) => {
                        self.z[q] = true;
                        self.phase4 = (self.phase4 + 2) & 3;
                    }
                    (true, true) => {
                        self.z[q] = false;
                    }
                    _ => {}
                }
                Ok(())
            }
            Gate::SX => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                match (xb, zb) {
                    (true, true) => {
                        self.x[q] = false;
                    }
                    (false, true) => {
                        self.x[q] = true;
                        self.phase4 = (self.phase4 + 2) & 3;
                    }
                    _ => {}
                }
                Ok(())
            }
            Gate::SXdg => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                match (xb, zb) {
                    (true, true) => {
                        self.x[q] = false;
                        self.phase4 = (self.phase4 + 2) & 3;
                    }
                    (false, true) => {
                        self.x[q] = true;
                    }
                    _ => {}
                }
                Ok(())
            }
            Gate::X => {
                let q = targets[0];
                if self.z[q] {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::Y => {
                let q = targets[0];
                let (xb, zb) = (self.x[q], self.z[q]);
                if xb != zb {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::Z => {
                let q = targets[0];
                if self.x[q] {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::Cx => {
                let c = targets[0];
                let t = targets[1];
                let (xc, zc, xt, zt) = (self.x[c], self.z[c], self.x[t], self.z[t]);
                self.x[t] = xt ^ xc;
                self.z[c] = zc ^ zt;
                if xc && zt && (xt == zc) {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::Cz => {
                let a = targets[0];
                let b = targets[1];
                let (xa, za, xb_, zb_) = (self.x[a], self.z[a], self.x[b], self.z[b]);
                self.z[a] = za ^ xb_;
                self.z[b] = zb_ ^ xa;
                if xa && xb_ && (za != zb_) {
                    self.phase4 = (self.phase4 + 2) & 3;
                }
                Ok(())
            }
            Gate::Swap => {
                let a = targets[0];
                let b = targets[1];
                self.x.swap(a, b);
                self.z.swap(a, b);
                Ok(())
            }
            _ => Err(PrismError::BackendUnsupported {
                backend: "stabilizer_rank".into(),
                operation: format!("Pauli conjugation by non-Clifford gate `{}`", gate.name()),
            }),
        }
    }

    /// `P|input⟩ = phase · |x_out⟩`. Returns `(phase, x_out)`.
    fn act_on_basis(&self, input: usize) -> (Complex64, usize) {
        let mut x_out = input;
        let mut phase4 = self.phase4 as u32;
        for q in 0..self.x.len() {
            let xq = self.x[q];
            let zq = self.z[q];
            let bit = (input >> q) & 1 == 1;
            match (xq, zq) {
                (false, false) => {}
                (true, false) => {
                    x_out ^= 1 << q;
                }
                (false, true) => {
                    if bit {
                        phase4 += 2;
                    }
                }
                (true, true) => {
                    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩.
                    x_out ^= 1 << q;
                    phase4 += if bit { 3 } else { 1 };
                }
            }
        }
        let phase = match (phase4 & 3) as u8 {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };
        (phase, x_out)
    }
}

const MAX_STATEVECTOR_QUBITS: usize = 25;
const MAX_TERMS: usize = 1 << 20; // 1M terms safety limit

/// Reject the instruction kinds the probability path cannot take, then lower
/// the gates to Clifford plus `T`/`Tdg`.
fn lower_probability_circuit(circuit: &Circuit) -> Result<Cow<'_, Circuit>> {
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { .. } | Instruction::Barrier { .. } => {}
            Instruction::Save { label, .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason: format!(
                        "save point `{label}` has no single state to read: the route                          carries a weighted sum of stabilizer branches"
                    ),
                });
            }
            Instruction::Measure { .. } | Instruction::Reset { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason:
                        "stabilizer-rank probabilities require a unitary circuit without measurements or resets"
                            .to_string(),
                });
            }
            Instruction::Conditional { .. } | Instruction::Region(_) => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason:
                        "stabilizer-rank probabilities require a unitary circuit without conditionals"
                            .to_string(),
                });
            }
        }
    }
    lower_clifford_t_circuit(circuit)
}

fn is_clifford_t_literal(gate: &Gate) -> bool {
    gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg)
}

/// Lower every gate, guarded gates and region bodies included, to Clifford
/// plus `T`/`Tdg`; borrows through when the circuit already is one. `Rz`, `P`,
/// and `Rzz` must sit on the pi/4 grid, `Fused` and `Cu` take the forms
/// `lower_to_clifford_t` documents, and anything else is rejected by name.
fn lower_clifford_t_circuit(circuit: &Circuit) -> Result<Cow<'_, Circuit>> {
    if !any_gate(&circuit.instructions, &mut |gate| {
        !is_clifford_t_literal(gate)
    }) {
        return Ok(Cow::Borrowed(circuit));
    }
    let mut out = Vec::with_capacity(circuit.instructions.len() * 2);
    lower_instructions(&circuit.instructions, &mut out)?;
    Ok(Cow::Owned(circuit.with_instructions(out)))
}

fn lower_instructions(instructions: &[Instruction], out: &mut Vec<Instruction>) -> Result<()> {
    let reject = |operation: String| PrismError::BackendUnsupported {
        backend: "stabilizer_rank".into(),
        operation,
    };
    for inst in instructions {
        match inst {
            Instruction::Gate { gate, targets } if !is_clifford_t_literal(gate) => {
                lower_to_clifford_t(gate, targets, &mut |gate, tgts| {
                    out.push(Instruction::Gate {
                        gate,
                        targets: SmallVec::from_slice(tgts),
                    });
                })
                .map_err(reject)?;
            }
            Instruction::Conditional {
                condition,
                gate,
                targets,
            } if !is_clifford_t_literal(gate) => {
                lower_to_clifford_t(gate, targets, &mut |gate, tgts| {
                    out.push(Instruction::Conditional {
                        condition: condition.clone(),
                        gate,
                        targets: SmallVec::from_slice(tgts),
                    });
                })
                .map_err(reject)?;
            }
            Instruction::Region(region) => {
                let mut body = Vec::with_capacity(region.body().len());
                lower_instructions(region.body(), &mut body)?;
                out.push(Instruction::Region(Box::new(GuardedRegion::new(
                    region.condition().clone(),
                    body,
                ))));
            }
            other => out.push(other.clone()),
        }
    }
    Ok(())
}

fn t_coefficients() -> (Complex64, Complex64) {
    let exp_i_pi_4 = Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
    let alpha = (Complex64::new(1.0, 0.0) + exp_i_pi_4) / 2.0;
    let beta = (Complex64::new(1.0, 0.0) - exp_i_pi_4) / 2.0;
    (alpha, beta)
}

fn tdg_coefficients() -> (Complex64, Complex64) {
    let (alpha, beta) = t_coefficients();
    (alpha.conj(), beta.conj())
}

/// A weighted Pauli-offset branch: `weight · offset · |ψ_0⟩`.
struct WeightedBranch {
    weight: Complex64,
    offset: SignedPauli,
}

/// A coherent branch for shot sampling. MPS storage keeps branch phases
/// explicit and avoids dense statevector materialization.
#[derive(Clone)]
struct WeightedMpsBranch {
    weight: Complex64,
    state: MpsBackend,
}

/// Result of stabilizer rank probability computation.
#[derive(Debug, Clone)]
pub struct StabRankResult {
    pub probabilities: Vec<f64>,
    /// Terms held at the end: `2^k` merged coefficients for a Clifford register of
    /// width `k` on the exact route, the branch count on the branch fallback, and the
    /// nonzero register coefficients, at most the budget, on the pruned route.
    pub num_terms: usize,
    pub t_count: usize,
    /// Nonzero register coefficients pruning zeroed, summed over the cuts (0 for exact).
    pub pruned_count: usize,
    /// Sum over the cuts of the exact 2-norm each one dropped, 0 for an exact run. It
    /// bounds the distance from the exact state to the kept one; see
    /// [`run_stabilizer_rank_approx`].
    pub discarded_weight: f64,
    /// Squared norm of the kept state before renormalization, 1.0 for an exact run.
    /// Equals `1 - sum of squared cut norms`.
    pub kept_norm_sqr: f64,
}

impl StabRankResult {
    /// Certified lower bound on the fidelity with the exact state.
    ///
    /// `((1 + N^2 - D^2) / (2 N))^2` with `N^2` = [`Self::kept_norm_sqr`] and `D` =
    /// [`Self::discarded_weight`], 0 when `1 + N^2 <= D^2`, and never below `1 - D^2`.
    /// Returns 1.0 for a run that pruned nothing. Twice `sqrt(1 - bound)` bounds the
    /// summed absolute difference from the exact distribution.
    pub fn fidelity_bound(&self) -> f64 {
        let (n2, d) = (self.kept_norm_sqr, self.discarded_weight);
        let numerator = 1.0 + n2 - d * d;
        if numerator <= 0.0 || n2 <= 0.0 {
            return 0.0;
        }
        (numerator * numerator / (4.0 * n2)).min(1.0)
    }
}

#[cfg(feature = "parallel")]
const MIN_TERMS_FOR_PAR: usize = 16;

/// Run stabilizer rank simulation for exact probabilities.
///
/// Probabilities come from the Clifford register: the branches of the
/// `T = α·I + β·Z` expansion that land on one basis state of the `k` qubits the
/// T gates reach are summed into one coefficient, so the cost follows `2^k`
/// with `k <= min(n, t)` and no T-count limit applies. A register wider than
/// 20 qubits falls back to one branch per T choice, at most `2^20` of them.
/// `n <= 25`.
pub fn run_stabilizer_rank(circuit: &Circuit, seed: u64) -> Result<StabRankResult> {
    exact_probabilities(circuit, seed, MAX_REGISTER_QUBITS)
}

fn exact_probabilities(
    circuit: &Circuit,
    seed: u64,
    max_register_width: usize,
) -> Result<StabRankResult> {
    let n = circuit.num_qubits;
    let circuit = lower_probability_circuit(circuit)?;
    require_dense_width(n)?;
    if let Some(sampler) = TerminalSampler::compile_every_qubit(&circuit, max_register_width)? {
        return Ok(StabRankResult {
            probabilities: sampler.distribution(),
            num_terms: 1 << sampler.register_width(),
            t_count: circuit.t_count(),
            pruned_count: 0,
            discarded_weight: 0.0,
            kept_norm_sqr: 1.0,
        });
    }
    let (mut backend, mut branches) = stabilizer_rank_setup(&circuit, seed)?;

    let mut t_count = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => match gate {
                Gate::T => {
                    t_count += 1;
                    expand_t(&mut branches, targets[0], false)?;
                }
                Gate::Tdg => {
                    t_count += 1;
                    expand_t(&mut branches, targets[0], true)?;
                }
                _ => {
                    backend.apply(inst)?;
                    conjugate_all(&mut branches, gate, targets)?;
                }
            },
            _ => {
                backend.apply(inst)?;
            }
        }
    }

    accumulate_probabilities(&backend, &branches, n).map(|probabilities| StabRankResult {
        probabilities,
        num_terms: branches.len(),
        t_count,
        pruned_count: 0,
        discarded_weight: 0.0,
        kept_norm_sqr: 1.0,
    })
}

fn require_dense_width(n: usize) -> Result<()> {
    if n > MAX_STATEVECTOR_QUBITS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!(
                "exact probabilities for {} qubits (max {})",
                n, MAX_STATEVECTOR_QUBITS
            ),
        });
    }
    Ok(())
}

/// Initial state for the branch runners, on a circuit `lower_probability_circuit`
/// and `require_dense_width` have already taken.
fn stabilizer_rank_setup(
    circuit: &Circuit,
    seed: u64,
) -> Result<(StabilizerBackend, Vec<WeightedBranch>)> {
    let n = circuit.num_qubits;
    let mut backend = StabilizerBackend::new(seed);
    backend.init(n, circuit.num_classical_bits)?;

    let branches = vec![WeightedBranch {
        weight: Complex64::new(1.0, 0.0),
        offset: SignedPauli::identity(n),
    }];
    Ok((backend, branches))
}

fn conjugate_all(branches: &mut [WeightedBranch], gate: &Gate, targets: &[usize]) -> Result<()> {
    #[cfg(feature = "parallel")]
    if branches.len() >= MIN_TERMS_FOR_PAR {
        use rayon::prelude::*;
        branches
            .par_iter_mut()
            .with_min_len(crate::backend::MIN_PAR_ITERS)
            .try_for_each(|b| b.offset.conjugate_by(gate, targets))?;
        return Ok(());
    }
    for b in branches.iter_mut() {
        b.offset.conjugate_by(gate, targets)?;
    }
    Ok(())
}

/// Reconstruct `|ψ_0⟩` once, then route each branch's contribution through
/// its Pauli offset. The global phase of `|ψ_0⟩` is a common factor across
/// every branch and cancels in `|·|²`.
fn accumulate_probabilities(
    backend: &StabilizerBackend,
    branches: &[WeightedBranch],
    n: usize,
) -> Result<Vec<f64>> {
    let dim = 1usize << n;
    let zero = Complex64::new(0.0, 0.0);
    let psi0 = backend.export_statevector()?;

    #[cfg(feature = "parallel")]
    if branches.len() >= MIN_TERMS_FOR_PAR {
        use rayon::prelude::*;
        let total_amps = branches
            .par_iter()
            .map(|b| {
                let mut partial = vec![zero; dim];
                for (y, amp) in psi0.iter().enumerate() {
                    let (phase, x_out) = b.offset.act_on_basis(y);
                    partial[x_out] += b.weight * phase * amp;
                }
                partial
            })
            .reduce(
                || vec![zero; dim],
                |mut a, b| {
                    for (ai, bi) in a.iter_mut().zip(b.iter()) {
                        *ai += bi;
                    }
                    a
                },
            );
        return Ok(total_amps.iter().map(|a| a.norm_sqr()).collect());
    }

    let mut total_amps = vec![zero; dim];
    for b in branches {
        for (y, amp) in psi0.iter().enumerate() {
            let (phase, x_out) = b.offset.act_on_basis(y);
            total_amps[x_out] += b.weight * phase * amp;
        }
    }
    Ok(total_amps.iter().map(|a| a.norm_sqr()).collect())
}

/// Split each branch by T = α·I + β·Z on `qubit`. The β branch left-multiplies
/// the offset by `Z_qubit` (since the T expansion inserts Z to the left of the
/// existing accumulated Pauli, in the order the gates have been processed).
fn expand_t(branches: &mut Vec<WeightedBranch>, qubit: usize, is_dagger: bool) -> Result<()> {
    let new_count =
        branches
            .len()
            .checked_mul(2)
            .ok_or_else(|| PrismError::BackendUnsupported {
                backend: "stabilizer_rank".into(),
                operation: "term count overflow".into(),
            })?;
    if new_count > MAX_TERMS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!("too many terms ({} > {})", new_count, MAX_TERMS),
        });
    }

    expand_t_unbounded(branches, qubit, is_dagger);
    Ok(())
}

/// Approximate stabilizer rank simulation with bounded term count.
///
/// Runs the Clifford register of [`run_stabilizer_rank`] and, after each T gate, zeroes
/// all but the `max_terms` largest register coefficients (at least 2), ties kept in
/// index order. Cut `j` drops a vector of exact norm `d_j`, orthogonal to what it keeps;
/// `discarded_weight` reports `D = sum d_j` and `kept_norm_sqr` the kept norm `N^2 = 1 -
/// sum d_j^2`. The rotations are unitary, so the exact state lies within `D` of the kept
/// one, and [`StabRankResult::fidelity_bound`] turns `(N, D)` into a fidelity floor `F`.
/// The returned probabilities are renormalized; their summed absolute difference from
/// the exact distribution is at most `2 sqrt(1 - F) <= 2 min(1, D)`. One cut gives `F =
/// N^2` exactly. A run that prunes nothing is exact.
///
/// `n <= 25`, and a register wider than 20 qubits returns `BackendUnsupported`. The
/// route is deterministic and ignores `seed`.
pub fn run_stabilizer_rank_approx(
    circuit: &Circuit,
    max_terms: usize,
    seed: u64,
) -> Result<StabRankResult> {
    let _ = seed;
    let n = circuit.num_qubits;
    let circuit = lower_probability_circuit(circuit)?;
    require_dense_width(n)?;
    let too_wide = || PrismError::BackendUnsupported {
        backend: "stabilizer_rank".into(),
        operation: format!(
            "pruned probabilities with a Clifford register wider than {MAX_REGISTER_QUBITS} qubits"
        ),
    };
    let t_count = circuit.t_count();
    if n.min(t_count) > MAX_REGISTER_QUBITS
        && register_price(&circuit, MAX_REGISTER_QUBITS)?.is_none()
    {
        return Err(too_wide());
    }
    let mut prune = Pruning::new(max_terms.max(2));
    let sampler =
        TerminalSampler::compile_every_qubit_pruned(&circuit, MAX_REGISTER_QUBITS, &mut prune)?
            .ok_or_else(too_wide)?;
    let mut probabilities = sampler.distribution();
    for p in &mut probabilities {
        *p /= prune.kept_norm_sqr;
    }
    Ok(StabRankResult {
        probabilities,
        num_terms: prune.held,
        t_count,
        pruned_count: prune.pruned,
        discarded_weight: prune.discarded,
        kept_norm_sqr: prune.kept_norm_sqr,
    })
}

fn expand_t_unbounded(branches: &mut Vec<WeightedBranch>, qubit: usize, is_dagger: bool) {
    let (alpha, beta) = if is_dagger {
        tdg_coefficients()
    } else {
        t_coefficients()
    };

    let orig_len = branches.len();
    let mut new_branches = Vec::with_capacity(orig_len);

    for b in branches.iter_mut() {
        let mut z_offset = b.offset.clone();
        z_offset.mul_z_on_left(qubit);
        new_branches.push(WeightedBranch {
            weight: b.weight * beta,
            offset: z_offset,
        });
        b.weight *= alpha;
    }

    branches.extend(new_branches);
}

/// Stabilizer inner product |⟨φ₁|φ₂⟩|² via combined stabilizer group method.
///
/// Merges the two generator sets into 2n rows, eliminates over GF(2) to find
/// the rank r of the merged strings, and reads |⟨φ₁|φ₂⟩|² = 2^{n-r}. A row
/// that eliminates to the identity string names a Pauli both groups hold; if
/// either group holds it with the opposite sign the states are orthogonal.
///
/// Each row keeps its two group factors apart rather than as one merged Pauli.
/// Rows of different groups can anticommute, so a merged product would depend
/// on the order the elimination happened to multiply them in, and a leftover
/// row would carry reordering signs that say nothing about the two groups. A
/// factor stays inside its own group, whose elements all commute, so it stays
/// a signed Pauli and the two signs compare directly.
pub fn stabilizer_overlap_sq(s1: &StabilizerBackend, s2: &StabilizerBackend, n: usize) -> f64 {
    let nw = n.div_ceil(64);
    let stride = 2 * nw;
    let total_rows = 2 * n;

    let (xz1, phase1) = s1.raw_tableau();
    let (xz2, phase2) = s2.raw_tableau();

    let mut left = vec![0u64; total_rows * stride];
    let mut right = vec![0u64; total_rows * stride];
    let mut left_sign = vec![false; total_rows];
    let mut right_sign = vec![false; total_rows];

    for i in 0..n {
        let src = (i + n) * stride;
        left[i * stride..(i + 1) * stride].copy_from_slice(&xz1[src..src + stride]);
        left_sign[i] = phase1[i + n];
        let dst = (i + n) * stride;
        right[dst..dst + stride].copy_from_slice(&xz2[src..src + stride]);
        right_sign[i + n] = phase2[i + n];
    }

    // The row's Pauli string, which the elimination pivots on, is the product
    // of its two factors, and their strings multiply as the XOR.
    let has_bit = |left: &[u64], right: &[u64], row: usize, word: usize, mask: u64| {
        (left[row * stride + word] ^ right[row * stride + word]) & mask != 0
    };

    let mut rank = 0usize;
    for col in 0..2 * n {
        let qubit = col % n;
        let half = if col < n { 0 } else { nw };
        let word = half + qubit / 64;
        let mask = 1u64 << (qubit % 64);

        let Some(pivot) = (rank..total_rows).find(|&r| has_bit(&left, &right, r, word, mask))
        else {
            continue;
        };
        if pivot != rank {
            for w in 0..stride {
                left.swap(rank * stride + w, pivot * stride + w);
                right.swap(rank * stride + w, pivot * stride + w);
            }
            left_sign.swap(rank, pivot);
            right_sign.swap(rank, pivot);
        }

        for row in rank + 1..total_rows {
            if !has_bit(&left, &right, row, word, mask) {
                continue;
            }
            multiply_within_group(&mut left, &mut left_sign, rank, row, nw, stride);
            multiply_within_group(&mut right, &mut right_sign, rank, row, nw, stride);
        }
        rank += 1;
    }

    // Every row past the rank has the identity string, so its two factors are
    // the same Pauli drawn from the two groups. Opposite signs put P in one
    // group and -P in the other, and the overlap is 0.
    for row in rank..total_rows {
        debug_assert!(
            (0..stride).all(|w| left[row * stride + w] == right[row * stride + w]),
            "a row past the rank eliminated to the identity string"
        );
        if left_sign[row] != right_sign[row] {
            return 0.0;
        }
    }

    // |⟨φ₁|φ₂⟩|² = 2^{n-r} where r is the combined rank of the stabilizer groups.
    // r ≥ n always (each group alone has n independent generators).
    // r = n → identical states (overlap = 1). r = 2n → minimum nonzero overlap (2^{-n}).
    2.0_f64.powi(n as i32 - rank as i32)
}

/// `rows[dst] <- rows[dst] * rows[src]` in Aaronson-Gottesman normal form,
/// both factors drawn from one stabilizer group. `dst` is past `src`, as the
/// forward elimination above leaves it.
fn multiply_within_group(
    rows: &mut [u64],
    signs: &mut [bool],
    src: usize,
    dst: usize,
    nw: usize,
    stride: usize,
) {
    debug_assert!(src < dst);
    let initial = 2 * u64::from(signs[src]) + 2 * u64::from(signs[dst]);
    let (head, tail) = rows.split_at_mut(dst * stride);
    let (src_x, src_z) = head[src * stride..src * stride + stride].split_at(nw);
    let (dst_x, dst_z) = tail[..stride].split_at_mut(nw);
    let sum =
        crate::backend::stabilizer::kernels::rowmul_words(dst_x, dst_z, src_x, src_z, initial);
    debug_assert_eq!(sum & 1, 0, "elements of one stabilizer group commute");
    signs[dst] = (sum & 3) == 2;
}

/// Phase-sensitive stabilizer inner product for small validation fixtures,
/// through a dense export capped at the probability-extraction width.
/// Large-qubit shot sampling uses MPS branch contraction instead.
pub fn stabilizer_inner_product(
    s1: &StabilizerBackend,
    s2: &StabilizerBackend,
    n: usize,
) -> Result<Complex64> {
    if n > MAX_STATEVECTOR_QUBITS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!(
                "dense stabilizer inner product validation for {} qubits (max {})",
                n, MAX_STATEVECTOR_QUBITS
            ),
        });
    }
    let v1 = s1.export_statevector()?;
    let v2 = s2.export_statevector()?;
    Ok(v1.iter().zip(v2.iter()).map(|(a, b)| a.conj() * b).sum())
}

fn validate_stabilizer_rank_shot_circuit(circuit: &Circuit) -> Result<()> {
    validate_shot_instructions(&circuit.instructions)
}

fn validate_shot_instructions(instructions: &[Instruction]) -> Result<()> {
    for inst in instructions {
        let gate = match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => gate,
            Instruction::Region(region) => {
                validate_shot_instructions(region.body())?;
                continue;
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Barrier { .. }
            | Instruction::Save { .. } => {
                continue;
            }
        };
        if !(gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg)) {
            return Err(PrismError::BackendUnsupported {
                backend: "stabilizer_rank".into(),
                operation: format!("non-Clifford+T gate `{}`", gate.name()),
            });
        }
    }
    Ok(())
}

fn initial_mps_branches(circuit: &Circuit, seed: u64) -> Result<Vec<WeightedMpsBranch>> {
    validate_stabilizer_rank_shot_circuit(circuit)?;
    let mut state = MpsBackend::new_exact(seed);
    state.init(circuit.num_qubits, circuit.num_classical_bits)?;
    Ok(vec![WeightedMpsBranch {
        weight: Complex64::new(1.0, 0.0),
        state,
    }])
}

fn apply_mps_gate(
    branches: &mut Vec<WeightedMpsBranch>,
    gate: &Gate,
    targets: &[usize],
) -> Result<()> {
    match gate {
        Gate::T => expand_t_mps(branches, targets[0], false),
        Gate::Tdg => expand_t_mps(branches, targets[0], true),
        _ if gate.is_clifford() => {
            let inst = Instruction::Gate {
                gate: gate.clone(),
                targets: SmallVec::from_slice(targets),
            };
            for branch in branches {
                branch.state.apply(&inst)?;
            }
            Ok(())
        }
        _ => Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!("non-Clifford+T gate `{}`", gate.name()),
        }),
    }
}

fn expand_t_mps(
    branches: &mut Vec<WeightedMpsBranch>,
    qubit: usize,
    is_dagger: bool,
) -> Result<()> {
    let new_count =
        branches
            .len()
            .checked_mul(2)
            .ok_or_else(|| PrismError::BackendUnsupported {
                backend: "stabilizer_rank".into(),
                operation: "term count overflow".into(),
            })?;
    if new_count > MAX_TERMS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!("too many terms ({} > {})", new_count, MAX_TERMS),
        });
    }

    let (alpha, beta) = if is_dagger {
        tdg_coefficients()
    } else {
        t_coefficients()
    };
    let z_inst = Instruction::Gate {
        gate: Gate::Z,
        targets: SmallVec::from_slice(&[qubit]),
    };

    let orig_len = branches.len();
    let mut new_branches = Vec::with_capacity(orig_len);
    for branch in branches.iter_mut() {
        let mut z_branch = branch.clone();
        z_branch.weight *= beta;
        z_branch.state.apply(&z_inst)?;
        new_branches.push(z_branch);
        branch.weight *= alpha;
    }
    branches.extend(new_branches);
    Ok(())
}

fn weighted_mps_norm_sq(branches: &[WeightedMpsBranch]) -> Result<f64> {
    if branches.is_empty() {
        return Ok(0.0);
    }

    // Gram term (i, j) is the conjugate of (j, i), so the upper triangle plus
    // the diagonal carries the whole sum; the scratch pair is reused across
    // every overlap in the sweep.
    let mut total = Complex64::new(0.0, 0.0);
    let mut tmp = Vec::new();
    let mut next_env = Vec::new();
    for (i, left) in branches.iter().enumerate() {
        let left_weight = left.weight.conj();
        let diag = left
            .state
            .inner_product_with_scratch(&left.state, &mut tmp, &mut next_env)?;
        total += left_weight * left.weight * diag;
        for right in &branches[i + 1..] {
            let overlap =
                left.state
                    .inner_product_with_scratch(&right.state, &mut tmp, &mut next_env)?;
            let term = left_weight * right.weight * overlap;
            total += term + term.conj();
        }
    }

    if !total.re.is_finite() || !total.im.is_finite() {
        return Err(PrismError::InvalidParameter {
            message: "stabilizer-rank MPS branch norm is not finite".to_string(),
        });
    }
    if total.re < -1e-8 || total.im.abs() > 1e-7 {
        return Err(PrismError::InvalidParameter {
            message: format!("invalid stabilizer-rank MPS branch norm {total:?}"),
        });
    }
    Ok(total.re.max(0.0))
}

fn project_mps_branches(
    branches: &[WeightedMpsBranch],
    qubit: usize,
    outcome: bool,
) -> Vec<WeightedMpsBranch> {
    let mut projected = Vec::with_capacity(branches.len());
    for branch in branches {
        let mut next = branch.clone();
        let prob = next.state.project_z_outcome(qubit, outcome);
        if prob <= crate::backend::NORM_CLAMP_MIN {
            continue;
        }
        next.weight *= prob.sqrt();
        projected.push(next);
    }
    projected
}

fn normalize_mps_branches(branches: &mut [WeightedMpsBranch], norm_sq: f64) -> Result<()> {
    if norm_sq <= crate::backend::NORM_CLAMP_MIN {
        return Err(PrismError::InvalidParameter {
            message: "stabilizer-rank projection eliminated every branch".to_string(),
        });
    }
    let scale = 1.0 / norm_sq.sqrt();
    for branch in branches {
        branch.weight *= scale;
    }
    Ok(())
}

fn sample_mps_measurement(
    branches: &mut Vec<WeightedMpsBranch>,
    qubit: usize,
    rng: &mut ChaCha8Rng,
) -> Result<bool> {
    let mut zero = project_mps_branches(branches, qubit, false);
    let mut one = project_mps_branches(branches, qubit, true);
    let norm_zero = weighted_mps_norm_sq(&zero)?;
    let norm_one = weighted_mps_norm_sq(&one)?;
    let denom = norm_zero + norm_one;
    if denom <= crate::backend::NORM_CLAMP_MIN {
        return Err(PrismError::InvalidParameter {
            message: "stabilizer-rank measurement has zero total probability".to_string(),
        });
    }

    let outcome = if norm_zero <= crate::backend::NORM_CLAMP_MIN {
        true
    } else if norm_one <= crate::backend::NORM_CLAMP_MIN {
        false
    } else {
        rng.random::<f64>() < (norm_one / denom).clamp(0.0, 1.0)
    };

    if outcome {
        normalize_mps_branches(&mut one, norm_one)?;
        *branches = one;
    } else {
        normalize_mps_branches(&mut zero, norm_zero)?;
        *branches = zero;
    }
    Ok(outcome)
}

fn apply_reset_mps(
    branches: &mut Vec<WeightedMpsBranch>,
    qubit: usize,
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    let measured_one = sample_mps_measurement(branches, qubit, rng)?;
    if measured_one {
        apply_mps_gate(branches, &Gate::X, &[qubit])?;
    }
    Ok(())
}

fn process_mps_instruction(
    branches: &mut Vec<WeightedMpsBranch>,
    inst: &Instruction,
    classical_bits: &mut [bool],
    rng: &mut ChaCha8Rng,
) -> Result<()> {
    match inst {
        Instruction::Gate { gate, targets } => apply_mps_gate(branches, gate, targets),
        Instruction::Measure {
            qubit,
            classical_bit,
        } => {
            let outcome = sample_mps_measurement(branches, *qubit, rng)?;
            classical_bits[*classical_bit] = outcome;
            Ok(())
        }
        Instruction::Reset { qubit } => apply_reset_mps(branches, *qubit, rng),
        Instruction::Barrier { .. } => Ok(()),
        Instruction::Save { label, .. } => Err(PrismError::IncompatibleBackend {
            backend: "stabilizer_rank".into(),
            reason: format!(
                "save point `{label}` has no single state to read: the route carries a \
                 weighted sum of stabilizer branches"
            ),
        }),
        Instruction::Conditional {
            condition,
            gate,
            targets,
        } => {
            if condition.evaluate(classical_bits) {
                apply_mps_gate(branches, gate, targets)?;
            }
            Ok(())
        }
        Instruction::Region(region) => {
            if region.condition().evaluate(classical_bits) {
                for inner in region.body() {
                    process_mps_instruction(branches, inner, classical_bits, rng)?;
                }
            }
            Ok(())
        }
    }
}

pub(super) fn sample_mps_branches_online(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<super::ShotsResult> {
    validate_stabilizer_rank_shot_circuit(circuit)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shots = Vec::with_capacity(num_shots);
    for _ in 0..num_shots {
        let mut branches = initial_mps_branches(circuit, seed)?;
        let mut classical_bits = vec![false; circuit.num_classical_bits];
        for inst in &circuit.instructions {
            process_mps_instruction(&mut branches, inst, &mut classical_bits, &mut rng)?;
        }
        shots.push(classical_bits);
    }
    Ok(
        super::ShotsResult::from_shots(shots, circuit.num_classical_bits).with_metadata(
            super::RunMetadata::exact(super::ResolvedBackend::StabilizerRank),
        ),
    )
}

/// Shot sampling on a Clifford+T circuit.
///
/// Samples from the coherent output distribution `|⟨x|ψ⟩|²`. The T branches
/// `T = αI + βZ` must be summed into a single amplitude before squaring;
/// sampling each branch independently as a classical mixture discards the
/// interference between branches and produces the wrong outcome distribution
/// (for example `H·T·H` would collapse to `[0.5, 0.5]` instead of
/// `[cos²(π/8), sin²(π/8)]`).
///
/// Terminal measurements of a unitary circuit are drawn from a Clifford tableau over a
/// dense register holding only the qubits the T gates reach, at most 20 of them; a
/// circuit whose register would grow past 20 returns `BackendUnsupported`. A circuit with
/// mid-circuit measurements, guards or resets takes the same register once its
/// measurements move to the end, which holds when every guard applies a Pauli and no gate
/// reuses a measured qubit before a reset. Otherwise its measurements are sampled by
/// projecting coherent weighted branches and contracting branch overlaps, up to 20 T
/// gates, guarded ones included; past that it returns `BackendUnsupported` before the
/// first shot. Neither route is bound by the dense statevector cap.
pub fn run_stabilizer_rank_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<super::ShotsResult> {
    let lowered = lower_clifford_t_circuit(circuit)?;
    let circuit = &*lowered;
    if !circuit.has_t_gates() {
        return super::run_shots_with(
            super::BackendKind::Stabilizer,
            circuit,
            num_shots,
            seed,
            &crate::limits::ResolvedLimits::process_default(),
        );
    }

    if circuit.has_terminal_measurements_only() && !circuit.has_resets() {
        let Some(sampler) = TerminalSampler::compile(circuit, MAX_REGISTER_QUBITS)? else {
            return Err(PrismError::BackendUnsupported {
                backend: "stabilizer_rank".into(),
                operation: format!(
                    "terminal sampling with a Clifford register wider than {MAX_REGISTER_QUBITS} \
                     qubits"
                ),
            });
        };
        return Ok(super::ShotsResult::from_shots(
            sampler.sample(num_shots, seed),
            circuit.num_classical_bits,
        )
        .with_metadata(super::RunMetadata::exact(
            super::ResolvedBackend::StabilizerRank,
        )));
    }

    if let Some(sampler) = deferred_sampler(circuit)? {
        return Ok(super::ShotsResult::from_shots(
            sampler.sample(num_shots, seed),
            circuit.num_classical_bits,
        )
        .with_metadata(super::RunMetadata::exact(
            super::ResolvedBackend::StabilizerRank,
        )));
    }
    reject_branch_overflow(circuit)?;
    sample_mps_branches_online(circuit, num_shots, seed)
}

/// The register sampler of the lowered `circuit` with its measurements deferred to the
/// end, `None` when the deferral declines or the register would pass
/// [`MAX_REGISTER_QUBITS`]. Past that many T gates a tableau dry run checks the width
/// before any amplitude is built.
fn deferred_sampler(circuit: &Circuit) -> Result<Option<TerminalSampler>> {
    let Ok(deferred) = super::compiled::defer_clifford_t_circuit(circuit) else {
        return Ok(None);
    };
    if circuit.t_count() > MAX_REGISTER_QUBITS
        && register_price(&deferred, MAX_REGISTER_QUBITS)?.is_none()
    {
        return Ok(None);
    }
    TerminalSampler::compile(&deferred, MAX_REGISTER_QUBITS)
}

/// Refuse a circuit whose branches would pass [`MAX_TERMS`] before the first shot: every
/// T gate, guarded ones included, may double them.
fn reject_branch_overflow(circuit: &Circuit) -> Result<()> {
    let t = circuit.t_count();
    if 1usize
        .checked_shl(t as u32)
        .is_some_and(|terms| terms <= MAX_TERMS)
    {
        return Ok(());
    }
    Err(PrismError::BackendUnsupported {
        backend: "stabilizer_rank".into(),
        operation: format!(
            "mid-circuit sampling of {t} T gates: the measurements do not defer to a \
             Clifford register of at most {MAX_REGISTER_QUBITS} qubits, and the branches \
             would pass {MAX_TERMS} terms"
        ),
    })
}

/// The lowered `circuit` with its measurements deferred to the end, `None` when it does
/// not lower to Clifford+T or a guard or measured qubit keeps a measurement in place.
pub(super) fn deferred_register_circuit(circuit: &Circuit) -> Option<Circuit> {
    let lowered = lower_clifford_t_circuit(circuit).ok()?;
    super::compiled::defer_clifford_t_circuit(&lowered).ok()
}

/// Width and rotation work of the register [`run_stabilizer_rank_shots`] samples terminal
/// shots of `circuit` from, decided from the tableau alone. `None` past `max_width`
/// qubits or when the circuit does not lower to Clifford+T.
pub(super) fn terminal_register_price(circuit: &Circuit, max_width: usize) -> Option<(usize, u64)> {
    let lowered = lower_clifford_t_circuit(circuit).ok()?;
    register_price(&lowered, max_width).ok().flatten()
}

#[cfg(feature = "bench-internal")]
pub use super::clifford_register::{CliffordRegisterCensus, RegisterPart};

/// Counts of the register [`run_stabilizer_rank_shots`] builds for the terminal
/// measurements of `circuit`, with the width cap lifted so a circuit past it still reports
/// its width and rotation work.
#[cfg(feature = "bench-internal")]
pub fn clifford_register_census(circuit: &Circuit) -> Result<CliffordRegisterCensus> {
    super::clifford_register::census(&*lower_clifford_t_circuit(circuit)?)
}

#[cfg(test)]
#[path = "stabilizer_rank_tests.rs"]
mod tests;
