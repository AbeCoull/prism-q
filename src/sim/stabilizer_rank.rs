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
//! - **Exact probabilities** (n ≤ 25): reconstruct `|ψ_0⟩`, sum weighted
//!   Pauli-shifted amplitudes, compute |amplitude|² for each basis state.
//! - **Measurement sampling** (any n): keep coherent weighted MPS branches,
//!   project requested measurement outcomes, and contract pairwise branch
//!   overlaps without materializing a dense statevector.

use num_complex::Complex64;
use rand::RngExt;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::FRAC_PI_4;

use crate::backend::Backend;
use crate::backend::mps::MpsBackend;
use crate::backend::stabilizer::StabilizerBackend;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

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

fn validate_stabilizer_rank_circuit(circuit: &Circuit) -> Result<()> {
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, .. } => {
                if !(gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg)) {
                    return Err(PrismError::BackendUnsupported {
                        backend: "stabilizer_rank".into(),
                        operation: format!("non-Clifford+T gate `{}`", gate.name()),
                    });
                }
            }
            Instruction::Measure { .. } | Instruction::Reset { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason:
                        "stabilizer-rank probabilities require a unitary circuit without measurements or resets"
                            .to_string(),
                });
            }
            Instruction::Conditional { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason:
                        "stabilizer-rank probabilities require a unitary circuit without conditionals"
                            .to_string(),
                });
            }
            Instruction::Barrier { .. } => {}
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
    /// Probability distribution over computational basis states.
    pub probabilities: Vec<f64>,
    /// Number of stabilizer terms in the decomposition.
    pub num_terms: usize,
    /// T-gate count in the circuit.
    pub t_count: usize,
    /// Number of terms pruned during approximate simulation (0 for exact).
    pub pruned_count: usize,
}

/// Minimum term count for parallel Clifford gate application.
#[cfg(feature = "parallel")]
const MIN_TERMS_FOR_PAR: usize = 16;

/// Run stabilizer rank simulation for exact probabilities.
///
/// Clifford gates update all terms in O(n²). T gates double the term count
/// via T = α·I + β·Z decomposition. n ≤ 25, total terms ≤ 2²⁰.
pub fn run_stabilizer_rank(circuit: &Circuit, seed: u64) -> Result<StabRankResult> {
    let n = circuit.num_qubits;
    let (mut backend, mut branches) = stabilizer_rank_setup(circuit, seed)?;

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
    })
}

/// Shared entry validation and initial state for the exact and approximate runners.
fn stabilizer_rank_setup(
    circuit: &Circuit,
    seed: u64,
) -> Result<(StabilizerBackend, Vec<WeightedBranch>)> {
    let n = circuit.num_qubits;
    if n > MAX_STATEVECTOR_QUBITS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!(
                "exact probabilities for {} qubits (max {})",
                n, MAX_STATEVECTOR_QUBITS
            ),
        });
    }
    validate_stabilizer_rank_circuit(circuit)?;

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
/// Like [`run_stabilizer_rank`] but prunes low-weight terms after each T gate
/// to keep term count ≤ `max_terms`. Russian roulette: below-threshold terms
/// are killed (probability 1 - w/w_max) or promoted (w → w_max).
pub fn run_stabilizer_rank_approx(
    circuit: &Circuit,
    max_terms: usize,
    seed: u64,
) -> Result<StabRankResult> {
    let n = circuit.num_qubits;
    let (mut backend, mut branches) = stabilizer_rank_setup(circuit, seed)?;

    let max_terms = max_terms.max(2);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut t_count = 0usize;
    let mut pruned_total = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => match gate {
                Gate::T => {
                    t_count += 1;
                    expand_t_unbounded(&mut branches, targets[0], false);
                    pruned_total += prune_terms(&mut branches, max_terms, &mut rng);
                }
                Gate::Tdg => {
                    t_count += 1;
                    expand_t_unbounded(&mut branches, targets[0], true);
                    pruned_total += prune_terms(&mut branches, max_terms, &mut rng);
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
        pruned_count: pruned_total,
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

/// Prune branches by descending weight magnitude.
fn prune_terms(
    branches: &mut Vec<WeightedBranch>,
    max_terms: usize,
    _rng: &mut ChaCha8Rng,
) -> usize {
    if branches.len() <= max_terms {
        return 0;
    }

    branches.sort_by(|a, b| {
        b.weight
            .norm_sqr()
            .partial_cmp(&a.weight.norm_sqr())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pruned = branches.len() - max_terms;
    branches.truncate(max_terms);
    pruned
}

/// Stabilizer inner product |⟨φ₁|φ₂⟩|² via combined stabilizer group method.
///
/// Merges generators into a 2n-row tableau, Gaussian-eliminates to find rank r.
/// Sign conflict (P and -P both present) → 0. Otherwise |⟨φ₁|φ₂⟩|² = 2^{n-r}.
pub fn stabilizer_overlap_sq(s1: &StabilizerBackend, s2: &StabilizerBackend, n: usize) -> f64 {
    let nw = n.div_ceil(64);
    let stride = 2 * nw;

    let (xz1, phase1) = s1.raw_tableau();
    let (xz2, phase2) = s2.raw_tableau();

    let mut combined_x = vec![0u64; 2 * n * nw];
    let mut combined_z = vec![0u64; 2 * n * nw];
    let mut combined_phase = vec![false; 2 * n];

    for i in 0..n {
        let src1 = (i + n) * stride;
        let src2 = (i + n) * stride;
        for w in 0..nw {
            combined_x[i * nw + w] = xz1[src1 + w];
            combined_z[i * nw + w] = xz1[src1 + nw + w];
            combined_x[(i + n) * nw + w] = xz2[src2 + w];
            combined_z[(i + n) * nw + w] = xz2[src2 + nw + w];
        }
        combined_phase[i] = phase1[i + n];
        combined_phase[i + n] = phase2[i + n];
    }

    // Gaussian elimination on the combined 2n × 2n Pauli system
    let mut rank = 0usize;
    let total_rows = 2 * n;

    // Iterate over 2n columns (X-block then Z-block) for full rank determination
    for col in 0..(2 * n) {
        let word = (col % n) / 64;
        let bit = 1u64 << ((col % n) % 64);
        let is_x_col = col < n;

        let mut pivot = None;
        for row in rank..total_rows {
            let has = if is_x_col {
                combined_x[row * nw + word] & bit != 0
            } else {
                combined_z[row * nw + word] & bit != 0
            };
            if has {
                pivot = Some(row);
                break;
            }
        }

        let pivot = match pivot {
            Some(p) => p,
            None => continue,
        };

        if pivot != rank {
            for w in 0..nw {
                combined_x.swap(rank * nw + w, pivot * nw + w);
                combined_z.swap(rank * nw + w, pivot * nw + w);
            }
            combined_phase.swap(rank, pivot);
        }

        for row in 0..total_rows {
            if row == rank {
                continue;
            }
            let has_bit = if is_x_col {
                combined_x[row * nw + word] & bit != 0
            } else {
                combined_z[row * nw + word] & bit != 0
            };
            if !has_bit {
                continue;
            }

            // AG rowmul: row ← row × rank (exact same phase logic as stabilizer.rs)
            let mut sum = if combined_phase[row] { 2u64 } else { 0 }
                + if combined_phase[rank] { 2u64 } else { 0 };

            for w in 0..nw {
                let x1 = combined_x[row * nw + w];
                let z1 = combined_z[row * nw + w];
                let x2 = combined_x[rank * nw + w];
                let z2 = combined_z[rank * nw + w];

                let new_x = x1 ^ x2;
                let new_z = z1 ^ z2;

                if (x1 | z1 | x2 | z2) != 0 {
                    let nonzero = (new_x | new_z) & (x1 | z1) & (x2 | z2);
                    let pos = (x1 & z1 & !x2 & z2) | (x1 & !z1 & x2 & z2) | (!x1 & z1 & x2 & !z2);
                    sum = sum.wrapping_add(2 * pos.count_ones() as u64);
                    sum = sum.wrapping_sub(nonzero.count_ones() as u64);
                }

                combined_x[row * nw + w] = new_x;
                combined_z[row * nw + w] = new_z;
            }

            combined_phase[row] = (sum & 3) >= 2;
        }

        rank += 1;
    }

    // Check for sign conflicts: any row that is all-zero X,Z but phase=true
    // means P and -P are both in the combined group → overlap = 0
    for row in rank..total_rows {
        let all_zero =
            (0..nw).all(|w| combined_x[row * nw + w] == 0 && combined_z[row * nw + w] == 0);
        if all_zero && combined_phase[row] {
            return 0.0;
        }
    }

    // |⟨φ₁|φ₂⟩|² = 2^{n-r} where r is the combined rank of the stabilizer groups.
    // r ≥ n always (each group alone has n independent generators).
    // r = n → identical states (overlap = 1). r = 2n → minimum nonzero overlap (2^{-n}).
    2.0_f64.powi(n as i32 - rank as i32)
}

/// Phase-sensitive stabilizer inner product for small validation fixtures.
///
/// Dense export is intentionally limited to the same size as probability
/// extraction. Large-qubit shot sampling uses MPS branch contraction below.
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
    for inst in &circuit.instructions {
        let gate = match inst {
            Instruction::Gate { gate, .. } | Instruction::Conditional { gate, .. } => gate,
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Barrier { .. } => {
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

    let mut total = Complex64::new(0.0, 0.0);
    for left in branches {
        let left_weight = left.weight.conj();
        for right in branches {
            let overlap = left.state.inner_product(&right.state)?;
            total += left_weight * right.weight * overlap;
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
    }
}

fn build_mps_branches_for_unitary(circuit: &Circuit, seed: u64) -> Result<Vec<WeightedMpsBranch>> {
    let mut branches = initial_mps_branches(circuit, seed)?;
    let mut classical_bits = vec![false; circuit.num_classical_bits];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { .. } | Instruction::Barrier { .. } => {
                process_mps_instruction(&mut branches, inst, &mut classical_bits, &mut rng)?;
            }
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: "stabilizer_rank".into(),
                    reason: "unitary branch preparation cannot include measurements, resets, or conditionals"
                        .to_string(),
                });
            }
        }
    }
    Ok(branches)
}

fn sample_terminal_mps_branches(
    base_branches: &[WeightedMpsBranch],
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<super::ShotsResult> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shots = Vec::with_capacity(num_shots);
    for _ in 0..num_shots {
        let mut branches = base_branches.to_vec();
        let mut classical_bits = vec![false; circuit.num_classical_bits];
        for inst in &circuit.instructions {
            if matches!(
                inst,
                Instruction::Measure { .. } | Instruction::Barrier { .. }
            ) {
                process_mps_instruction(&mut branches, inst, &mut classical_bits, &mut rng)?;
            }
        }
        shots.push(classical_bits);
    }
    Ok(super::ShotsResult::from_shots(
        shots,
        circuit.num_classical_bits,
    ))
}

fn sample_mps_branches_online(
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
    Ok(super::ShotsResult::from_shots(
        shots,
        circuit.num_classical_bits,
    ))
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
/// Terminal and mid-circuit measurements are sampled by projecting coherent
/// weighted branches and contracting branch overlaps. This avoids the dense
/// statevector cap used by probability-vector APIs.
pub fn run_stabilizer_rank_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<super::ShotsResult> {
    if !circuit.has_t_gates() {
        return super::run_shots_with(super::BackendKind::Stabilizer, circuit, num_shots, seed);
    }

    validate_stabilizer_rank_shot_circuit(circuit)?;

    if circuit.has_terminal_measurements_only() && !circuit.has_resets() {
        let stripped = circuit.without_measurements();
        let base_branches = build_mps_branches_for_unitary(&stripped, seed)?;
        return sample_terminal_mps_branches(&base_branches, circuit, num_shots, seed);
    }

    sample_mps_branches_online(circuit, num_shots, seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::SmallVec;

    #[test]
    fn test_pure_clifford() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);

        let result = run_stabilizer_rank(&c, 42).unwrap();
        assert_eq!(result.num_terms, 1);
        assert_eq!(result.t_count, 0);
        assert!((result.probabilities[0] - 0.5).abs() < 1e-10);
        assert!((result.probabilities[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_single_t() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_stabilizer_rank(&c, 42).unwrap();
        assert_eq!(result.num_terms, 2);
        assert_eq!(result.t_count, 1);

        let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
        assert!(
            (result.probabilities[0] - p0_expected).abs() < 1e-10,
            "P(0) = {}, expected {}",
            result.probabilities[0],
            p0_expected
        );
    }

    #[test]
    fn shots_preserve_t_branch_interference() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);

        let num_shots = 20_000;
        let result = run_stabilizer_rank_shots(&c, num_shots, 42).unwrap();
        let zeros = result.shots.iter().filter(|s| !s[0]).count();
        let p0 = zeros as f64 / num_shots as f64;
        let expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
        assert!(
            (p0 - expected).abs() < 0.02,
            "P(0) = {p0}, expected {expected} (a classical T-branch mixture would give 0.5)"
        );
    }

    #[test]
    fn shots_without_t_bypass_statevector_qubit_cap() {
        let n = MAX_STATEVECTOR_QUBITS + 5;
        let mut c = Circuit::new(n, n);
        for q in 0..n {
            c.add_gate(Gate::H, &[q]);
            c.add_measure(q, q);
        }

        let result = run_stabilizer_rank_shots(&c, 16, 42).unwrap();
        assert_eq!(result.shots.len(), 16);
        assert!(result.shots.iter().all(|shot| shot.len() == n));
    }

    #[test]
    fn shots_with_t_bypass_statevector_qubit_cap_terminal() {
        let n = MAX_STATEVECTOR_QUBITS + 5;
        let mut c = Circuit::new(n, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);

        let result = run_stabilizer_rank_shots(&c, 32, 42).unwrap();
        assert_eq!(result.shots.len(), 32);
        assert!(result.shots.iter().all(|shot| shot.len() == 1));

        let public_result =
            crate::sim::run_shots_with(crate::sim::BackendKind::StabilizerRank, &c, 8, 42).unwrap();
        assert_eq!(public_result.shots.len(), 8);

        let auto_result =
            crate::sim::run_shots_with(crate::sim::BackendKind::Auto, &c, 8, 42).unwrap();
        assert_eq!(auto_result.shots.len(), 8);
    }

    #[test]
    fn shots_with_t_bypass_statevector_qubit_cap_mid_circuit() {
        let n = MAX_STATEVECTOR_QUBITS + 5;
        let mut c = Circuit::new(n, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);
        c.instructions.push(Instruction::Conditional {
            condition: crate::circuit::ClassicalCondition::BitIsOne(0),
            gate: Gate::X,
            targets: SmallVec::from_slice(&[1]),
        });
        c.add_reset(0);
        c.add_measure(1, 1);

        let result = run_stabilizer_rank_shots(&c, 32, 42).unwrap();
        assert_eq!(result.shots.len(), 32);
        assert!(result.shots.iter().all(|shot| shot.len() == 2));
    }

    #[test]
    fn forced_mps_projection_has_expected_probability() {
        let mut plus = MpsBackend::new_exact(0);
        plus.init(1, 0).unwrap();
        plus.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();

        let mut zero = plus.clone();
        let mut one = plus;
        let p0 = zero.project_z_outcome(0, false);
        let p1 = one.project_z_outcome(0, true);

        assert!((p0 - 0.5).abs() < 1e-12);
        assert!((p1 - 0.5).abs() < 1e-12);
        assert!(zero.inner_product(&one).unwrap().norm() < 1e-12);
    }

    #[test]
    fn test_rejects_reset_in_probability_path() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_reset(0);

        assert!(run_stabilizer_rank(&c, 42).is_err());
        assert!(run_stabilizer_rank_approx(&c, 8, 42).is_err());
    }

    #[test]
    fn test_rejects_measurement_in_probability_path() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_measure(0, 0);

        assert!(run_stabilizer_rank(&c, 42).is_err());
        assert!(run_stabilizer_rank_approx(&c, 8, 42).is_err());
    }

    #[test]
    fn test_multi_t_with_separating_clifford() {
        // Regression: prior to absorbing the deterministic Z-eigenvalue
        // into the branch weight, this circuit returned [0.5, 0.5]
        // because the AG tableau dropped the global -1 phase on the
        // |1⟩ branch. The T-count scaling sweep surfaced the bug.
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_stabilizer_rank(&c, 42).unwrap();
        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        for (i, (&sr, &sv)) in result.probabilities.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr - sv).abs() < 1e-10,
                "P({i}) mismatch: stab_rank = {sr}, statevector = {sv}"
            );
        }
    }

    /// Regression for prior two-T multi-qubit reconstruction failure.
    /// Per-branch tableau export picked inconsistent implicit global
    /// phases when support shifted between branches. The Pauli-offset
    /// representation should preserve the interference pattern.
    #[test]
    fn test_two_t_multi_qubit_bisect_stages() {
        type Stage<'a> = (&'a str, &'a [(Gate, &'a [usize])]);
        let stages: &[Stage] = &[
            ("ghz_only", &[(Gate::H, &[0]), (Gate::Cx, &[0, 1])]),
            (
                "ghz_t",
                &[(Gate::H, &[0]), (Gate::Cx, &[0, 1]), (Gate::T, &[0])],
            ),
            (
                "ghz_t_h",
                &[
                    (Gate::H, &[0]),
                    (Gate::Cx, &[0, 1]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                ],
            ),
            (
                "ghz_t_h_t",
                &[
                    (Gate::H, &[0]),
                    (Gate::Cx, &[0, 1]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                    (Gate::T, &[0]),
                ],
            ),
            (
                "ghz_t_h_t_h0",
                &[
                    (Gate::H, &[0]),
                    (Gate::Cx, &[0, 1]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                ],
            ),
            (
                "ghz_t_h_t_h0_h1",
                &[
                    (Gate::H, &[0]),
                    (Gate::Cx, &[0, 1]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                    (Gate::T, &[0]),
                    (Gate::H, &[0]),
                    (Gate::H, &[1]),
                ],
            ),
        ];
        let mut failures = Vec::new();
        for (label, gates) in stages {
            let mut c = Circuit::new(2, 0);
            for (gate, targets) in *gates {
                c.add_gate(gate.clone(), targets);
            }
            let result = run_stabilizer_rank(&c, 42).unwrap();
            let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
            let sv_probs = sv.probabilities.unwrap().to_vec();
            let max_diff = result
                .probabilities
                .iter()
                .zip(sv_probs.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            if max_diff > 1e-9 {
                failures.push(format!(
                    "{label}: sr={:?} sv={:?}",
                    result.probabilities, sv_probs
                ));
            }
        }
        assert!(failures.is_empty(), "fails:\n  {}", failures.join("\n  "));
    }

    /// Companion to the bisect test: minimal multi-qubit two-T fixture.
    /// Same root cause: cross-branch phase reconstruction.
    #[test]
    fn test_two_t_multi_qubit_entangled_matches_statevector() {
        // Surface fixture: H_0, CX(0,1), T_0, H_0, T_0, H_0, H_1.
        // Both stabilizer_rank and statevector should agree on the
        // full 2q probability vector.
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        let result = run_stabilizer_rank(&c, 42).unwrap();
        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        for (i, (&sr, &sv)) in result.probabilities.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr - sv).abs() < 1e-10,
                "P({i}) mismatch: stab_rank = {sr}, statevector = {sv}"
            );
        }
    }

    #[test]
    fn test_multi_qubit_multi_t_post_cliffords_matches_statevector() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Cx, &[0, 2]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cz, &[0, 1]);
        c.add_gate(Gate::Tdg, &[2]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::Swap, &[0, 2]);

        let sr = run_stabilizer_rank(&c, 42).unwrap();
        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        for (i, (sr_p, sv_p)) in sr.probabilities.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr_p - sv_p).abs() < 1e-10,
                "prob[{i}]: stab_rank={sr_p}, statevector={sv_p}"
            );
        }
    }

    #[test]
    fn test_matches_statevector() {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[2]);
        c.add_gate(Gate::T, &[2]);
        c.add_gate(Gate::Cx, &[2, 1]);

        let sr = run_stabilizer_rank(&c, 42).unwrap();
        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        for (i, (sr_p, sv_p)) in sr.probabilities.iter().zip(sv_probs.iter()).enumerate() {
            assert!(
                (sr_p - sv_p).abs() < 1e-10,
                "prob[{i}]: stab_rank={sr_p}, statevector={sv_p}"
            );
        }
    }

    #[test]
    fn test_tdg() {
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Tdg, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_stabilizer_rank(&c, 42).unwrap();
        assert_eq!(result.t_count, 1);

        let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
        assert!((result.probabilities[0] - p0_expected).abs() < 1e-10);
    }

    #[test]
    fn test_term_count_scaling() {
        let mut c = Circuit::new(4, 0);
        for q in 0..4 {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::T, &[q]);
        }

        let result = run_stabilizer_rank(&c, 42).unwrap();
        assert_eq!(result.t_count, 4);
        assert_eq!(result.num_terms, 16); // 2^4

        let total: f64 = result.probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_overlap_identical_states() {
        let mut b1 = StabilizerBackend::new(42);
        b1.init(3, 0).unwrap();
        let inst_h = Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        };
        let inst_cx = Instruction::Gate {
            gate: Gate::Cx,
            targets: SmallVec::from_slice(&[0, 1]),
        };
        b1.apply(&inst_h).unwrap();
        b1.apply(&inst_cx).unwrap();

        let b2 = b1.clone();
        let overlap = stabilizer_overlap_sq(&b1, &b2, 3);
        assert!(
            (overlap - 1.0).abs() < 1e-10,
            "overlap of identical states should be 1, got {}",
            overlap
        );
    }

    #[test]
    fn test_overlap_orthogonal_states() {
        // |0⟩ and |1⟩ are orthogonal
        let mut b1 = StabilizerBackend::new(42);
        b1.init(1, 0).unwrap();
        // b1 = |0⟩

        let mut b2 = StabilizerBackend::new(42);
        b2.init(1, 0).unwrap();
        let inst_x = Instruction::Gate {
            gate: Gate::X,
            targets: SmallVec::from_slice(&[0]),
        };
        b2.apply(&inst_x).unwrap();
        // b2 = |1⟩

        let overlap = stabilizer_overlap_sq(&b1, &b2, 1);
        assert!(
            overlap < 1e-10,
            "overlap of |0⟩ and |1⟩ should be 0, got {}",
            overlap
        );
    }

    #[test]
    fn test_overlap_bell_with_basis() {
        // |Φ+⟩ = (|00⟩+|11⟩)/√2 vs |00⟩: |⟨00|Φ+⟩|² = 1/2
        let mut bell = StabilizerBackend::new(42);
        bell.init(2, 0).unwrap();
        let inst_h = Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        };
        let inst_cx = Instruction::Gate {
            gate: Gate::Cx,
            targets: SmallVec::from_slice(&[0, 1]),
        };
        bell.apply(&inst_h).unwrap();
        bell.apply(&inst_cx).unwrap();

        let mut basis = StabilizerBackend::new(42);
        basis.init(2, 0).unwrap();

        let overlap = stabilizer_overlap_sq(&bell, &basis, 2);
        assert!(
            (overlap - 0.5).abs() < 1e-10,
            "|⟨00|Φ+⟩|² should be 0.5, got {}",
            overlap
        );
    }

    #[test]
    fn test_overlap_plus_with_basis() {
        // |+⟩ vs |0⟩: |⟨0|+⟩|² = 1/2
        let mut plus = StabilizerBackend::new(42);
        plus.init(1, 0).unwrap();
        let inst_h = Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        };
        plus.apply(&inst_h).unwrap();

        let mut zero = StabilizerBackend::new(42);
        zero.init(1, 0).unwrap();

        let overlap = stabilizer_overlap_sq(&plus, &zero, 1);
        assert!(
            (overlap - 0.5).abs() < 1e-10,
            "|⟨0|+⟩|² should be 0.5, got {}",
            overlap
        );
    }

    #[test]
    fn test_stabilizer_inner_product_matches_dense_export() {
        let mut b1 = StabilizerBackend::new(42);
        b1.init(2, 0).unwrap();
        b1.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        b1.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: SmallVec::from_slice(&[0, 1]),
        })
        .unwrap();

        let mut b2 = StabilizerBackend::new(7);
        b2.init(2, 0).unwrap();
        b2.apply(&Instruction::Gate {
            gate: Gate::H,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        b2.apply(&Instruction::Gate {
            gate: Gate::S,
            targets: SmallVec::from_slice(&[0]),
        })
        .unwrap();
        b2.apply(&Instruction::Gate {
            gate: Gate::Cx,
            targets: SmallVec::from_slice(&[0, 1]),
        })
        .unwrap();

        let b1_vec = b1.export_statevector().unwrap();
        let b2_vec = b2.export_statevector().unwrap();
        let expected: Complex64 = b1_vec
            .iter()
            .zip(b2_vec.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let actual = stabilizer_inner_product(&b1, &b2, 2).unwrap();
        assert!((actual - expected).norm() < 1e-12);
    }

    #[test]
    fn test_too_many_terms() {
        let mut c = Circuit::new(1, 0);
        // 21 T gates would need 2^21 > MAX_TERMS terms
        for _ in 0..21 {
            c.add_gate(Gate::T, &[0]);
        }
        let result = run_stabilizer_rank(&c, 42);
        assert!(result.is_err());
    }

    #[test]
    fn test_approx_small_circuit_exact() {
        // With budget > 2^t, approximate = exact
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::T, &[1]);

        let exact = run_stabilizer_rank(&c, 42).unwrap();
        let approx = run_stabilizer_rank_approx(&c, 1024, 42).unwrap();

        assert_eq!(approx.num_terms, exact.num_terms);
        assert_eq!(approx.pruned_count, 0);
        for (e, a) in exact.probabilities.iter().zip(approx.probabilities.iter()) {
            assert!((e - a).abs() < 1e-10);
        }
    }

    #[test]
    fn test_approx_prunes_terms() {
        let mut c = Circuit::new(4, 0);
        for q in 0..4 {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::T, &[q]);
        }
        // 4 T gates → 16 terms exact. Budget of 8 should prune.
        let result = run_stabilizer_rank_approx(&c, 8, 42).unwrap();
        assert!(result.num_terms <= 8);
        assert!(result.pruned_count > 0);

        let total: f64 = result.probabilities.iter().sum();
        // Approximate, so not exactly 1.0, but should be in a reasonable range
        assert!(total > 0.5 && total < 2.0, "total = {total}");
    }

    #[test]
    fn test_approx_handles_many_t_gates() {
        // 10 T gates → 1024 exact terms, budget 32 should work without error
        let mut c = Circuit::new(3, 0);
        for _ in 0..10 {
            c.add_gate(Gate::H, &[0]);
            c.add_gate(Gate::T, &[0]);
        }
        let result = run_stabilizer_rank_approx(&c, 32, 42).unwrap();
        assert!(result.num_terms <= 32);
        assert_eq!(result.t_count, 10);
    }
}

#[cfg(test)]
mod more_tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::gates::Gate;

    #[test]
    fn rejects_too_many_qubits() {
        let n = MAX_STATEVECTOR_QUBITS + 1;
        let c = Circuit::new(n, 0);
        assert!(run_stabilizer_rank(&c, 0).is_err());
        assert!(run_stabilizer_rank_approx(&c, 0, 16).is_err());
    }

    #[test]
    fn approx_path_runs_with_pruning() {
        let mut c = Circuit::new(3, 0);
        for q in 0..3 {
            c.add_gate(Gate::H, &[q]);
            c.add_gate(Gate::T, &[q]);
            c.add_gate(Gate::T, &[q]);
        }
        let result = run_stabilizer_rank_approx(&c, 42, 4).unwrap();
        let total: f64 = result.probabilities.iter().sum();
        assert!(total.is_finite());
    }

    #[test]
    fn approx_path_tdg_runs() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Tdg, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Tdg, &[1]);
        let result = run_stabilizer_rank_approx(&c, 7, 16).unwrap();
        assert!(result.probabilities.iter().all(|p| p.is_finite()));
    }
}
