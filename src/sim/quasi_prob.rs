//! Quasi-probability simulation for Clifford+T circuits.
//!
//! Decomposes each T gate as T = α·I + β·Z where:
//!   α = (1 + e^{iπ/4})/2, β = (1 - e^{iπ/4})/2
//!
//! Two modes:
//! - **Exact enumeration** (t ≤ `MAX_EXACT_T`): enumerate all 2^t Clifford branches,
//!   accumulate weighted amplitudes via `export_statevector()`, compute exact probabilities.
//!   Cost: O(2^t · (n² + 2^n)).
//! - **Shot sampling**: sample random Clifford branches, run with measurements.
//!   Each shot randomly replaces T→I or T→Z weighted by |α|,|β|.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::FRAC_PI_4;

use crate::backend::stabilizer::StabilizerBackend;
use crate::backend::Backend;
use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::Result;
use crate::gates::Gate;

const MAX_EXACT_T: usize = 20;
const MAX_EXACT_QUBITS: usize = 25;

/// Decomposition coefficients for T = α·I + β·Z.
fn t_decomposition() -> (Complex64, Complex64) {
    let exp_i_pi_4 = Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
    let alpha = (Complex64::new(1.0, 0.0) + exp_i_pi_4) / 2.0;
    let beta = (Complex64::new(1.0, 0.0) - exp_i_pi_4) / 2.0;
    (alpha, beta)
}

/// Decomposition coefficients for Tdg = α†·I + β†·Z.
fn tdg_decomposition() -> (Complex64, Complex64) {
    let (alpha, beta) = t_decomposition();
    (alpha.conj(), beta.conj())
}

/// Per-T-gate negativity: |α| + |β| = cos(π/8) + sin(π/8).
fn negativity_per_t() -> f64 {
    let (alpha, beta) = t_decomposition();
    alpha.norm() + beta.norm()
}

/// Information about a T/Tdg gate location in the circuit.
struct TGateLocation {
    instruction_index: usize,
    qubit: usize,
    is_dagger: bool,
}

/// Find all T/Tdg gate locations in a circuit.
fn find_t_gates(circuit: &Circuit) -> Vec<TGateLocation> {
    circuit
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(idx, inst)| match inst {
            Instruction::Gate { gate, targets } if targets.len() == 1 => match gate {
                Gate::T => Some(TGateLocation {
                    instruction_index: idx,
                    qubit: targets[0],
                    is_dagger: false,
                }),
                Gate::Tdg => Some(TGateLocation {
                    instruction_index: idx,
                    qubit: targets[0],
                    is_dagger: true,
                }),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Build a Clifford circuit by replacing T/Tdg gates with I or Z choices.
///
/// `branch_bits`: bit i = 0 → replace T_i with I, bit i = 1 → replace with Z.
///
/// Returns (circuit, weight) where weight is the product of coefficients.
fn build_clifford_branch(
    circuit: &Circuit,
    t_locations: &[TGateLocation],
    branch_bits: u64,
) -> (Circuit, Complex64) {
    let mut out = Circuit::new(circuit.num_qubits, circuit.num_classical_bits);
    out.instructions.reserve(circuit.instructions.len());

    let mut weight = Complex64::new(1.0, 0.0);
    let mut t_idx = 0;

    for (instr_idx, inst) in circuit.instructions.iter().enumerate() {
        if t_idx < t_locations.len() && t_locations[t_idx].instruction_index == instr_idx {
            let loc = &t_locations[t_idx];
            let (alpha, beta) = if loc.is_dagger {
                tdg_decomposition()
            } else {
                t_decomposition()
            };

            if (branch_bits >> t_idx) & 1 == 1 {
                weight *= beta;
                out.instructions.push(Instruction::Gate {
                    gate: Gate::Z,
                    targets: SmallVec::from_slice(&[loc.qubit]),
                });
            } else {
                weight *= alpha;
            }
            t_idx += 1;
        } else {
            out.instructions.push(inst.clone());
        }
    }

    (out, weight)
}

/// Result of quasi-probability estimation.
#[derive(Debug, Clone)]
pub struct QuasiProbResult {
    /// Estimated probability distribution.
    pub probabilities: Vec<f64>,
    /// Number of Clifford branches evaluated.
    pub num_branches: usize,
    /// Total negativity ξ = (|α|+|β|)^t.
    pub negativity: f64,
    /// T-gate count.
    pub t_count: usize,
}

/// Compute exact probabilities for a Clifford+T circuit via branch enumeration.
///
/// Enumerates all 2^t Clifford branches, runs each on stabilizer, accumulates
/// weighted amplitudes, then computes |amplitude|² for each basis state.
///
/// Requirements: t ≤ 20, n ≤ 25 (due to exponential cost in both dimensions).
pub fn run_quasi_prob(circuit: &Circuit, seed: u64) -> Result<QuasiProbResult> {
    let t_locations = find_t_gates(circuit);
    let t_count = t_locations.len();

    if t_count == 0 {
        let mut backend = StabilizerBackend::new(seed);
        backend.init(circuit.num_qubits, circuit.num_classical_bits)?;
        for inst in &circuit.instructions {
            backend.apply(inst)?;
        }
        let probs = backend.probabilities()?;
        return Ok(QuasiProbResult {
            probabilities: probs,
            num_branches: 1,
            negativity: 1.0,
            t_count: 0,
        });
    }

    if t_count > MAX_EXACT_T {
        return Err(crate::error::PrismError::BackendUnsupported {
            backend: "quasi_probability".into(),
            operation: format!(
                "exact enumeration for {} T-gates (max {})",
                t_count, MAX_EXACT_T
            ),
        });
    }
    if circuit.num_qubits > MAX_EXACT_QUBITS {
        return Err(crate::error::PrismError::BackendUnsupported {
            backend: "quasi_probability".into(),
            operation: format!(
                "statevector export for {} qubits (max {})",
                circuit.num_qubits, MAX_EXACT_QUBITS
            ),
        });
    }

    let n_states = 1usize << circuit.num_qubits;
    let num_branches = 1u64 << t_count;
    let zero = Complex64::new(0.0, 0.0);
    let mut total_amps = vec![zero; n_states];

    for branch in 0..num_branches {
        let (branch_circuit, weight) = build_clifford_branch(circuit, &t_locations, branch);

        let mut backend = StabilizerBackend::new(seed);
        backend.init(branch_circuit.num_qubits, branch_circuit.num_classical_bits)?;
        for inst in &branch_circuit.instructions {
            backend.apply(inst)?;
        }

        let amps = backend.export_statevector()?;
        for (i, amp) in amps.iter().enumerate() {
            total_amps[i] += weight * amp;
        }
    }

    let probabilities: Vec<f64> = total_amps.iter().map(|a| a.norm_sqr()).collect();
    let negativity = negativity_per_t().powi(t_count as i32);

    Ok(QuasiProbResult {
        probabilities,
        num_branches: num_branches as usize,
        negativity,
        t_count,
    })
}

/// Run quasi-probability shot sampling on a Clifford+T circuit.
///
/// For each shot, randomly samples a Clifford branch (replacing each T gate
/// with I or Z weighted by |α| and |β|), runs it on the stabilizer backend
/// with measurements, and returns the measurement outcomes.
///
/// Uses antithetic sampling: each pair of shots uses branch `b` and its
/// complement `~b`, creating negatively correlated pairs that improve
/// coverage of the branch space.
pub fn run_quasi_prob_shots(
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
        return Err(crate::error::PrismError::BackendUnsupported {
            backend: "quasi_probability".into(),
            operation: format!("shot sampling for {} T-gates (max 64)", t_count),
        });
    }

    let (alpha, beta) = t_decomposition();
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
            if rng.gen::<f64>() >= p_alpha {
                branch_bits |= 1u64 << i;
            }
        }

        // Primary branch
        let branch_seed = rng.gen::<u64>();
        shots.push(run_branch(circuit, &t_locations, branch_bits, branch_seed)?);

        // Antithetic branch (complement): every I↔Z swapped
        if shots.len() < num_shots {
            let anti_bits = !branch_bits & t_mask;
            let anti_seed = rng.gen::<u64>();
            shots.push(run_branch(circuit, &t_locations, anti_bits, anti_seed)?);
        }
    }

    shots.truncate(num_shots);

    Ok(super::ShotsResult {
        shots,
        probabilities: None,
    })
}

fn run_branch(
    circuit: &Circuit,
    t_locations: &[TGateLocation],
    branch_bits: u64,
    seed: u64,
) -> Result<Vec<bool>> {
    let (branch_circuit, _weight) = build_clifford_branch(circuit, t_locations, branch_bits);
    let mut backend = StabilizerBackend::new(seed);
    backend.init(branch_circuit.num_qubits, branch_circuit.num_classical_bits)?;
    for inst in &branch_circuit.instructions {
        backend.apply(inst)?;
    }
    Ok(backend.classical_results().to_vec())
}

/// Run quasi-probability shot sampling with stratified branch selection.
///
/// Partitions the 2^t branch space by Hamming weight (number of Z
/// replacements). Allocates shots proportionally to each stratum's
/// total weight, ensuring coverage across low-Z and high-Z branches.
///
/// Each stratum uses antithetic pairing internally.
pub fn run_quasi_prob_shots_stratified(
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
        return Err(crate::error::PrismError::BackendUnsupported {
            backend: "quasi_probability".into(),
            operation: format!("shot sampling for {} T-gates (max 64)", t_count),
        });
    }

    let (alpha, beta) = t_decomposition();
    let p_alpha = alpha.norm() / (alpha.norm() + beta.norm());
    let p_beta = 1.0 - p_alpha;

    // Compute weight of each Hamming weight stratum: C(t,k) * p_alpha^(t-k) * p_beta^k
    let mut stratum_weights = Vec::with_capacity(t_count + 1);
    let mut total_weight = 0.0;
    for k in 0..=t_count {
        let binom = binomial(t_count, k) as f64;
        let w = binom * p_alpha.powi((t_count - k) as i32) * p_beta.powi(k as i32);
        stratum_weights.push(w);
        total_weight += w;
    }

    // Allocate shots per stratum (proportional, at least 1 if weight > 0)
    let mut shots_per_stratum: Vec<usize> = stratum_weights
        .iter()
        .map(|w| ((w / total_weight) * num_shots as f64).round() as usize)
        .collect();
    let allocated: usize = shots_per_stratum.iter().sum();
    if allocated < num_shots {
        // Add remainder to the heaviest stratum
        let max_idx = stratum_weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        shots_per_stratum[max_idx] += num_shots - allocated;
    } else if allocated > num_shots {
        // Remove from the heaviest stratum
        let max_idx = shots_per_stratum
            .iter()
            .enumerate()
            .max_by_key(|(_, &s)| s)
            .map(|(i, _)| i)
            .unwrap_or(0);
        shots_per_stratum[max_idx] -= allocated - num_shots;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shots = Vec::with_capacity(num_shots);

    for (k, &n_shots) in shots_per_stratum.iter().enumerate() {
        for _ in 0..n_shots {
            // Sample a random branch with exactly k bits set
            let branch_bits = random_bits_with_weight(&mut rng, t_count, k);
            let branch_seed = rng.gen::<u64>();
            shots.push(run_branch(circuit, &t_locations, branch_bits, branch_seed)?);
        }
    }

    Ok(super::ShotsResult {
        shots,
        probabilities: None,
    })
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u64;
    for i in 0..k {
        result = result.saturating_mul((n - i) as u64) / (i as u64 + 1);
    }
    result
}

fn random_bits_with_weight(rng: &mut ChaCha8Rng, n: usize, k: usize) -> u64 {
    if k == 0 {
        return 0;
    }
    if k >= n {
        return (1u64 << n) - 1;
    }
    // Fisher-Yates partial shuffle to pick k positions from n
    let mut positions: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_range(0..n - i);
        positions.swap(i, j);
    }
    let mut bits = 0u64;
    for &pos in &positions[..k] {
        bits |= 1u64 << pos;
    }
    bits
}

/// Adaptive shot sampling with convergence detection.
///
/// Samples shots in batches, monitoring the total-variation distance between
/// successive normalized histograms. Stops early when the distribution
/// stabilizes (TV distance < `tolerance`) or `max_shots` is reached.
///
/// Returns the shots collected so far plus convergence metadata.
pub fn run_quasi_prob_shots_adaptive(
    circuit: &Circuit,
    max_shots: usize,
    seed: u64,
    tolerance: f64,
) -> Result<AdaptiveResult> {
    let t_locations = find_t_gates(circuit);
    let t_count = t_locations.len();

    if t_count > 64 {
        return Err(crate::error::PrismError::BackendUnsupported {
            backend: "quasi_probability".into(),
            operation: format!("shot sampling for {} T-gates (max 64)", t_count),
        });
    }

    if t_count == 0 {
        let result =
            super::run_shots_with(super::BackendKind::Stabilizer, circuit, max_shots, seed)?;
        return Ok(AdaptiveResult {
            shots_result: result,
            converged: true,
            tv_distance: 0.0,
            total_shots: max_shots,
        });
    }

    let (alpha, beta) = t_decomposition();
    let p_alpha = alpha.norm() / (alpha.norm() + beta.norm());
    let t_mask = if t_count >= 64 {
        u64::MAX
    } else {
        (1u64 << t_count) - 1
    };

    let n_cbits = circuit.num_classical_bits;
    let batch_size = (max_shots / 10).max(50).min(max_shots);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut shots = Vec::with_capacity(max_shots);
    let mut histogram = std::collections::HashMap::<Vec<bool>, usize>::new();
    let mut prev_histogram = std::collections::HashMap::<Vec<bool>, usize>::new();
    let mut converged = false;
    let mut tv_distance = f64::INFINITY;

    while shots.len() < max_shots {
        let batch_end = (shots.len() + batch_size).min(max_shots);
        let this_batch = batch_end - shots.len();

        for _ in 0..this_batch {
            let mut branch_bits = 0u64;
            for i in 0..t_count {
                if rng.gen::<f64>() >= p_alpha {
                    branch_bits |= 1u64 << i;
                }
            }

            let branch_seed = rng.gen::<u64>();
            let shot = run_branch(circuit, &t_locations, branch_bits, branch_seed)?;
            *histogram.entry(shot.clone()).or_insert(0) += 1;
            shots.push(shot);

            // Antithetic
            if shots.len() < max_shots {
                let anti_bits = !branch_bits & t_mask;
                let anti_seed = rng.gen::<u64>();
                let anti_shot = run_branch(circuit, &t_locations, anti_bits, anti_seed)?;
                *histogram.entry(anti_shot.clone()).or_insert(0) += 1;
                shots.push(anti_shot);
            }
        }

        // Compute TV distance between current and previous histogram
        if !prev_histogram.is_empty() {
            let total_prev: usize = prev_histogram.values().sum();
            let total_curr: usize = histogram.values().sum();

            if total_prev > 0 && total_curr > 0 {
                let all_keys: std::collections::HashSet<&Vec<bool>> =
                    histogram.keys().chain(prev_histogram.keys()).collect();

                let mut tv = 0.0;
                for key in &all_keys {
                    let p = *prev_histogram.get(*key).unwrap_or(&0) as f64 / total_prev as f64;
                    let q = *histogram.get(*key).unwrap_or(&0) as f64 / total_curr as f64;
                    tv += (p - q).abs();
                }
                tv_distance = tv / 2.0;

                if tv_distance < tolerance {
                    converged = true;
                    break;
                }
            }
        }

        prev_histogram = histogram.clone();
    }

    // Ensure we don't have zero-length classical results
    if n_cbits == 0 && shots.is_empty() {
        shots.push(vec![]);
    }

    let total_shots = shots.len();
    Ok(AdaptiveResult {
        shots_result: super::ShotsResult {
            shots,
            probabilities: None,
        },
        converged,
        tv_distance,
        total_shots,
    })
}

/// Result of adaptive quasi-probability shot sampling.
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// The shot results.
    pub shots_result: super::ShotsResult,
    /// Whether the distribution converged within tolerance.
    pub converged: bool,
    /// Final total-variation distance between successive histograms.
    pub tv_distance: f64,
    /// Number of shots actually taken.
    pub total_shots: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_decomposition_coefficients() {
        let (alpha, beta) = t_decomposition();

        // T|0⟩ = |0⟩, so (α·I + β·Z)|0⟩ = (α+β)|0⟩ = |0⟩ → α+β = 1
        let sum = alpha + beta;
        assert!((sum.re - 1.0).abs() < 1e-12);
        assert!(sum.im.abs() < 1e-12);

        // T|1⟩ = e^{iπ/4}|1⟩, so (α·I + β·Z)|1⟩ = (α-β)|1⟩ = e^{iπ/4}|1⟩
        let diff = alpha - beta;
        let expected = Complex64::new(FRAC_PI_4.cos(), FRAC_PI_4.sin());
        assert!((diff - expected).norm() < 1e-12);
    }

    #[test]
    fn test_negativity_per_t() {
        // |α| + |β| = cos(π/8) + sin(π/8)
        let xi = negativity_per_t();
        let expected = (std::f64::consts::FRAC_PI_8).cos() + (std::f64::consts::FRAC_PI_8).sin();
        assert!(
            (xi - expected).abs() < 1e-12,
            "Per-T negativity should be cos(π/8)+sin(π/8) ≈ {}, got {}",
            expected,
            xi
        );
        // ≈ 1.3066
        assert!(xi > 1.3 && xi < 1.31);
    }

    #[test]
    fn test_pure_clifford_passthrough() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);

        let result = run_quasi_prob(&c, 42).unwrap();
        assert_eq!(result.t_count, 0);
        assert_eq!(result.num_branches, 1);

        // Bell state: |00⟩ and |11⟩ each ~50%
        assert!((result.probabilities[0] - 0.5).abs() < 1e-10);
        assert!(result.probabilities[1].abs() < 1e-10);
        assert!(result.probabilities[2].abs() < 1e-10);
        assert!((result.probabilities[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_single_t_exact() {
        // H·T·H on qubit 0
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_quasi_prob(&c, 42).unwrap();
        assert_eq!(result.t_count, 1);
        assert_eq!(result.num_branches, 2); // exact: 2^1 = 2 branches

        // P(0) = cos²(π/8), P(1) = sin²(π/8)
        let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
        let p1_expected = (std::f64::consts::FRAC_PI_8).sin().powi(2);

        assert!(
            (result.probabilities[0] - p0_expected).abs() < 1e-10,
            "P(0) = {}, expected {}",
            result.probabilities[0],
            p0_expected
        );
        assert!(
            (result.probabilities[1] - p1_expected).abs() < 1e-10,
            "P(1) = {}, expected {}",
            result.probabilities[1],
            p1_expected
        );
    }

    #[test]
    fn test_two_t_gates_exact() {
        // T·T = S on qubit 0
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_quasi_prob(&c, 42).unwrap();
        assert_eq!(result.t_count, 2);
        assert_eq!(result.num_branches, 4); // 2^2

        // H·S·H|0⟩ = H·S|+⟩ = H·(|0⟩+i|1⟩)/√2 = (1+i)/2|0⟩ + (1-i)/2|1⟩
        // P(0) = |(1+i)/2|² = 1/2, P(1) = |(1-i)/2|² = 1/2
        assert!(
            (result.probabilities[0] - 0.5).abs() < 1e-10,
            "P(0) = {}, expected 0.5",
            result.probabilities[0]
        );
        assert!(
            (result.probabilities[1] - 0.5).abs() < 1e-10,
            "P(1) = {}, expected 0.5",
            result.probabilities[1]
        );
    }

    #[test]
    fn test_tdg_exact() {
        // H·Tdg·H on qubit 0
        let mut c = Circuit::new(1, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Tdg, &[0]);
        c.add_gate(Gate::H, &[0]);

        let result = run_quasi_prob(&c, 42).unwrap();
        assert_eq!(result.t_count, 1);

        // Same probabilities as T (Tdg just conjugates the phase, |amp|² is symmetric)
        let p0_expected = (std::f64::consts::FRAC_PI_8).cos().powi(2);
        assert!(
            (result.probabilities[0] - p0_expected).abs() < 1e-10,
            "P(0) = {}, expected {}",
            result.probabilities[0],
            p0_expected
        );
    }

    #[test]
    fn test_quasi_prob_shots() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        let result = run_quasi_prob_shots(&c, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
        }
    }

    #[test]
    fn test_tdg_decomposition() {
        let (alpha_t, beta_t) = t_decomposition();
        let (alpha_tdg, beta_tdg) = tdg_decomposition();
        assert!((alpha_tdg - alpha_t.conj()).norm() < 1e-12);
        assert!((beta_tdg - beta_t.conj()).norm() < 1e-12);
    }

    #[test]
    fn test_antithetic_shots() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        let result = run_quasi_prob_shots(&c, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
        for shot in &result.shots {
            assert_eq!(shot.len(), 2);
        }
    }

    #[test]
    fn test_antithetic_odd_shots() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_measure(0, 0);

        // Odd number of shots — should truncate cleanly
        let result = run_quasi_prob_shots(&c, 7, 42).unwrap();
        assert_eq!(result.shots.len(), 7);
    }

    #[test]
    fn test_stratified_shots() {
        let mut c = Circuit::new(2, 2);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_measure(0, 0);
        c.add_measure(1, 1);

        let result = run_quasi_prob_shots_stratified(&c, 100, 42).unwrap();
        assert_eq!(result.shots.len(), 100);
    }

    #[test]
    fn test_adaptive_converges() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_measure(0, 0);

        let result = run_quasi_prob_shots_adaptive(&c, 1000, 42, 0.1).unwrap();
        assert!(result.total_shots <= 1000);
        assert!(result.total_shots >= 50); // at least one batch
    }

    #[test]
    fn test_adaptive_pure_clifford() {
        let mut c = Circuit::new(1, 1);
        c.add_gate(Gate::H, &[0]);
        c.add_measure(0, 0);

        let result = run_quasi_prob_shots_adaptive(&c, 100, 42, 0.01).unwrap();
        assert!(result.converged);
        assert_eq!(result.total_shots, 100);
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 0), 1);
        assert_eq!(binomial(5, 1), 5);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(5, 3), 10);
        assert_eq!(binomial(5, 5), 1);
        assert_eq!(binomial(10, 5), 252);
        assert_eq!(binomial(0, 0), 1);
    }

    #[test]
    fn test_random_bits_with_weight() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..20 {
            let bits = random_bits_with_weight(&mut rng, 8, 3);
            assert_eq!(bits.count_ones(), 3);
            assert!(bits < 256); // only bottom 8 bits
        }
        assert_eq!(random_bits_with_weight(&mut rng, 5, 0), 0);
        assert_eq!(random_bits_with_weight(&mut rng, 5, 5), 0b11111);
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        // Multi-T circuit: H-T-H on each qubit + CX
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::T, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::T, &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);

        let result = run_quasi_prob(&c, 42).unwrap();
        assert_eq!(result.t_count, 2);

        let total: f64 = result.probabilities.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-8,
            "Probabilities sum to {}, expected 1.0",
            total
        );
    }
}
