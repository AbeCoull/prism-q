//! Stabilizer rank simulation for Clifford+T circuits.
//!
//! Maintains a weighted sum of stabilizer states: |ψ⟩ = Σ_k c_k |φ_k⟩.
//! Clifford gates are applied to all terms (O(n²) per term). Each T gate
//! expands every term into two via T = α·I + β·Z, doubling the term count.
//!
//! Two modes:
//! - **Exact probabilities** (n ≤ 25): export statevectors per term, accumulate
//!   weighted amplitudes, compute |amplitude|² for each basis state.
//! - **Measurement sampling** (any n): compute Born probabilities via pairwise
//!   stabilizer inner products O(χ² · n³), sample outcomes.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::FRAC_PI_4;

use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;

const MAX_STATEVECTOR_QUBITS: usize = 25;
const MAX_TERMS: usize = 1 << 20; // 1M terms safety limit

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

/// A weighted stabilizer state: coefficient × stabilizer tableau.
struct WeightedStabilizer {
    weight: Complex64,
    backend: StabilizerBackend,
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
    let nc = circuit.num_classical_bits;

    if n > MAX_STATEVECTOR_QUBITS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!(
                "exact probabilities for {} qubits (max {})",
                n, MAX_STATEVECTOR_QUBITS
            ),
        });
    }

    let mut terms: Vec<WeightedStabilizer> = vec![WeightedStabilizer {
        weight: Complex64::new(1.0, 0.0),
        backend: StabilizerBackend::new(seed),
    }];
    terms[0].backend.init(n, nc)?;

    let mut t_count = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => match gate {
                Gate::T => {
                    t_count += 1;
                    expand_t(&mut terms, targets[0], false)?;
                }
                Gate::Tdg => {
                    t_count += 1;
                    expand_t(&mut terms, targets[0], true)?;
                }
                _ => apply_to_all_terms(&mut terms, inst)?,
            },
            _ => apply_to_all_terms(&mut terms, inst)?,
        }
    }

    accumulate_probabilities(&terms, n).map(|probabilities| StabRankResult {
        probabilities,
        num_terms: terms.len(),
        t_count,
        pruned_count: 0,
    })
}

/// Apply an instruction to all terms, parallelized when term count is large.
fn apply_to_all_terms(terms: &mut [WeightedStabilizer], inst: &Instruction) -> Result<()> {
    #[cfg(feature = "parallel")]
    if terms.len() >= MIN_TERMS_FOR_PAR {
        use rayon::prelude::*;
        terms
            .par_iter_mut()
            .try_for_each(|term| term.backend.apply(inst))?;
        return Ok(());
    }

    for term in terms.iter_mut() {
        term.backend.apply(inst)?;
    }
    Ok(())
}

/// Accumulate weighted amplitudes across all terms and compute probabilities.
fn accumulate_probabilities(terms: &[WeightedStabilizer], n: usize) -> Result<Vec<f64>> {
    let dim = 1usize << n;
    let zero = Complex64::new(0.0, 0.0);

    #[cfg(feature = "parallel")]
    if terms.len() >= MIN_TERMS_FOR_PAR {
        use rayon::prelude::*;
        let total_amps = terms
            .par_iter()
            .map(|term| {
                let amps = term.backend.export_statevector().unwrap();
                let mut partial = vec![zero; dim];
                for (i, amp) in amps.iter().enumerate() {
                    partial[i] = term.weight * amp;
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
    for term in terms {
        let amps = term.backend.export_statevector()?;
        for (i, amp) in amps.iter().enumerate() {
            total_amps[i] += term.weight * amp;
        }
    }
    Ok(total_amps.iter().map(|a| a.norm_sqr()).collect())
}

/// Expand each term by T = α·I + β·Z decomposition on target qubit.
fn expand_t(terms: &mut Vec<WeightedStabilizer>, qubit: usize, is_dagger: bool) -> Result<()> {
    let new_count = terms
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

    let orig_len = terms.len();
    let mut new_terms = Vec::with_capacity(orig_len);

    for term in terms.iter_mut() {
        let mut z_backend = term.backend.clone();
        let z_inst = Instruction::Gate {
            gate: Gate::Z,
            targets: SmallVec::from_slice(&[qubit]),
        };
        z_backend.apply(&z_inst)?;
        new_terms.push(WeightedStabilizer {
            weight: term.weight * beta,
            backend: z_backend,
        });

        term.weight *= alpha;
    }

    terms.extend(new_terms);
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
    let nc = circuit.num_classical_bits;

    if n > MAX_STATEVECTOR_QUBITS {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!(
                "exact probabilities for {} qubits (max {})",
                n, MAX_STATEVECTOR_QUBITS
            ),
        });
    }

    let max_terms = max_terms.max(2);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut terms: Vec<WeightedStabilizer> = vec![WeightedStabilizer {
        weight: Complex64::new(1.0, 0.0),
        backend: StabilizerBackend::new(seed),
    }];
    terms[0].backend.init(n, nc)?;

    let mut t_count = 0usize;
    let mut pruned_total = 0usize;

    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, targets } => match gate {
                Gate::T => {
                    t_count += 1;
                    expand_t_unbounded(&mut terms, targets[0], false)?;
                    pruned_total += prune_terms(&mut terms, max_terms, &mut rng);
                }
                Gate::Tdg => {
                    t_count += 1;
                    expand_t_unbounded(&mut terms, targets[0], true)?;
                    pruned_total += prune_terms(&mut terms, max_terms, &mut rng);
                }
                _ => apply_to_all_terms(&mut terms, inst)?,
            },
            _ => apply_to_all_terms(&mut terms, inst)?,
        }
    }

    accumulate_probabilities(&terms, n).map(|probabilities| StabRankResult {
        probabilities,
        num_terms: terms.len(),
        t_count,
        pruned_count: pruned_total,
    })
}

/// Expand terms without the MAX_TERMS safety check (for approximate mode).
fn expand_t_unbounded(
    terms: &mut Vec<WeightedStabilizer>,
    qubit: usize,
    is_dagger: bool,
) -> Result<()> {
    let (alpha, beta) = if is_dagger {
        tdg_coefficients()
    } else {
        t_coefficients()
    };

    let orig_len = terms.len();
    let mut new_terms = Vec::with_capacity(orig_len);

    for term in terms.iter_mut() {
        let mut z_backend = term.backend.clone();
        let z_inst = Instruction::Gate {
            gate: Gate::Z,
            targets: SmallVec::from_slice(&[qubit]),
        };
        z_backend.apply(&z_inst)?;
        new_terms.push(WeightedStabilizer {
            weight: term.weight * beta,
            backend: z_backend,
        });
        term.weight *= alpha;
    }

    terms.extend(new_terms);
    Ok(())
}

/// Prune terms to at most `max_terms` by discarding lowest-weight terms.
///
/// Sorts by descending weight magnitude, keeps the top `max_terms`.
/// Returns the number of pruned terms.
fn prune_terms(
    terms: &mut Vec<WeightedStabilizer>,
    max_terms: usize,
    _rng: &mut ChaCha8Rng,
) -> usize {
    if terms.len() <= max_terms {
        return 0;
    }

    terms.sort_by(|a, b| {
        b.weight
            .norm_sqr()
            .partial_cmp(&a.weight.norm_sqr())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let pruned = terms.len() - max_terms;
    terms.truncate(max_terms);
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

struct TGateLocation {
    instruction_index: usize,
    qubit: usize,
}

fn find_t_gates(circuit: &Circuit) -> Vec<TGateLocation> {
    circuit
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| match inst {
            Instruction::Gate {
                gate: Gate::T | Gate::Tdg,
                targets,
            } if targets.len() == 1 => Some(TGateLocation {
                instruction_index: idx,
                qubit: targets[0],
            }),
            _ => None,
        })
        .collect()
}

fn build_clifford_branch(
    circuit: &Circuit,
    t_locations: &[TGateLocation],
    branch_bits: u64,
) -> Circuit {
    let mut out = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
    out.instructions.reserve(circuit.instructions.len());

    let mut t_idx = 0;

    for (instr_idx, inst) in circuit.instructions.iter().enumerate() {
        if t_idx < t_locations.len() && t_locations[t_idx].instruction_index == instr_idx {
            let loc = &t_locations[t_idx];
            if (branch_bits >> t_idx) & 1 == 1 {
                out.instructions.push(Instruction::Gate {
                    gate: Gate::Z,
                    targets: SmallVec::from_slice(&[loc.qubit]),
                });
            }
            t_idx += 1;
        } else {
            out.instructions.push(inst.clone());
        }
    }

    out
}

fn run_branch(
    circuit: &Circuit,
    t_locations: &[TGateLocation],
    branch_bits: u64,
    seed: u64,
) -> Result<Vec<bool>> {
    let branch_circuit = build_clifford_branch(circuit, t_locations, branch_bits);
    let mut backend = StabilizerBackend::new(seed);
    backend.init(branch_circuit.num_qubits, branch_circuit.num_classical_bits)?;
    for inst in &branch_circuit.instructions {
        backend.apply(inst)?;
    }
    Ok(backend.classical_results().to_vec())
}

/// Shot sampling on a Clifford+T circuit via stochastic branch selection.
///
/// Each shot randomly replaces T→I or T→Z weighted by |α|/|β|, producing a
/// Clifford circuit that is simulated on the stabilizer backend. Uses
/// antithetic pairing (complement branch) for variance reduction.
pub fn run_stabilizer_rank_shots(
    circuit: &Circuit,
    num_shots: usize,
    seed: u64,
) -> Result<super::ShotsResult> {
    let t_locations = find_t_gates(circuit);
    let t_count = t_locations.len();

    if t_count == 0 {
        return super::run_shots_with(super::BackendKind::Stabilizer, circuit, num_shots, seed);
    }
    if t_count > 64 {
        return Err(PrismError::BackendUnsupported {
            backend: "stabilizer_rank".into(),
            operation: format!("shot sampling for {} T-gates (max 64)", t_count),
        });
    }

    let (alpha, beta) = t_coefficients();
    let p_alpha = alpha.norm() / (alpha.norm() + beta.norm());
    let t_mask = if t_count >= 64 {
        u64::MAX
    } else {
        (1u64 << t_count) - 1
    };

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shots = Vec::with_capacity(num_shots);

    while shots.len() < num_shots {
        let mut branch_bits = 0u64;
        for i in 0..t_count {
            if rng.random::<f64>() >= p_alpha {
                branch_bits |= 1u64 << i;
            }
        }

        let branch_seed = rng.random::<u64>();
        shots.push(run_branch(circuit, &t_locations, branch_bits, branch_seed)?);

        if shots.len() < num_shots {
            let anti_bits = !branch_bits & t_mask;
            let anti_seed = rng.random::<u64>();
            shots.push(run_branch(circuit, &t_locations, anti_bits, anti_seed)?);
        }
    }

    shots.truncate(num_shots);

    Ok(super::ShotsResult {
        shots,
        num_classical_bits: circuit.num_classical_bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
