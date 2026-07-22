use num_complex::Complex64;
use rand::SeedableRng;
use rand::{Rng, RngExt};
use rand_chacha::ChaCha8Rng;

use crate::circuit::{Circuit, Instruction, SmallVec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::compiled::{PauliVec, flip_bit, propagate_backward};

const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Absolute ceiling on the weighted-Pauli term count, enforced even in exact
/// mode (`max_terms == 0`). Without a per-step truncation budget the term set
/// grows as `2^(in-cone T count)`; this caps transient memory the same way the
/// stabilizer-rank backend does. Exceeding it is an error, not a silent
/// truncation: callers wanting bounded approximate evaluation pass a nonzero
/// `max_terms` instead.
const SPD_MAX_TERMS_CEILING: usize = 1 << 20;

#[inline]
fn check_spd_term_ceiling(len: usize) -> Result<()> {
    if len > SPD_MAX_TERMS_CEILING {
        return Err(PrismError::BackendUnsupported {
            backend: "SPD".into(),
            operation: format!(
                "weighted-Pauli sum exceeded {SPD_MAX_TERMS_CEILING} terms; pass a nonzero \
                 max_terms for bounded approximate truncation"
            ),
        });
    }
    Ok(())
}

fn validate_clifford_t_unitary(circuit: &Circuit, backend: &'static str) -> Result<()> {
    for inst in &circuit.instructions {
        match inst {
            Instruction::Gate { gate, .. } => {
                if !(gate.is_clifford() || matches!(gate, Gate::T | Gate::Tdg)) {
                    return Err(PrismError::BackendUnsupported {
                        backend: backend.to_string(),
                        operation: format!("non-Clifford+T gate `{}`", gate.name()),
                    });
                }
            }
            Instruction::Barrier { .. } => {}
            Instruction::Measure { .. }
            | Instruction::Reset { .. }
            | Instruction::Conditional { .. } => {
                return Err(PrismError::IncompatibleBackend {
                    backend: backend.to_string(),
                    reason: "Pauli propagation requires a unitary Clifford+T circuit without measurements, resets, or conditionals"
                        .to_string(),
                });
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Coalesced circuit representation
// ---------------------------------------------------------------------------

enum CoalescedOp {
    SmallCliff(Vec<(Gate, SmallVec<[usize; 4]>)>),
    T { qubit: usize, is_dagger: bool },
}

fn coalesce_cliffords(circuit: &Circuit) -> Vec<CoalescedOp> {
    let mut ops = Vec::new();
    let mut cliff_buf: Vec<(Gate, SmallVec<[usize; 4]>)> = Vec::new();

    for inst in &circuit.instructions {
        if let Instruction::Gate { gate, targets } = inst {
            match gate {
                Gate::T => {
                    flush_cliff_buf(&mut cliff_buf, &mut ops);
                    ops.push(CoalescedOp::T {
                        qubit: targets[0],
                        is_dagger: false,
                    });
                }
                Gate::Tdg => {
                    flush_cliff_buf(&mut cliff_buf, &mut ops);
                    ops.push(CoalescedOp::T {
                        qubit: targets[0],
                        is_dagger: true,
                    });
                }
                _ => {
                    cliff_buf.push((gate.clone(), SmallVec::from_slice(targets)));
                }
            }
        } else {
            flush_cliff_buf(&mut cliff_buf, &mut ops);
        }
    }
    flush_cliff_buf(&mut cliff_buf, &mut ops);
    ops
}

fn flush_cliff_buf(buf: &mut Vec<(Gate, SmallVec<[usize; 4]>)>, ops: &mut Vec<CoalescedOp>) {
    if buf.is_empty() {
        return;
    }
    ops.push(CoalescedOp::SmallCliff(std::mem::take(buf)));
}

// ---------------------------------------------------------------------------
// SPP (Stochastic Pauli Propagation)
// ---------------------------------------------------------------------------

#[inline(always)]
fn branch_t_gate(
    pauli: &mut PauliVec,
    qubit: usize,
    is_dagger: bool,
    rng: &mut impl Rng,
) -> Complex64 {
    // PauliVec stores the Y letter as the ordered product XZ. Since
    // actual Y = i XZ, the T flip branch is imaginary in this basis:
    // T contributes -i/sqrt(2), and Tdg contributes +i/sqrt(2).
    // SPP samples one branch with probability 1/2, so the per-shot
    // flip weight is +/-i*sqrt(2).
    if !pauli.has_x_or_y(qubit) {
        return Complex64::new(1.0, 0.0);
    }

    let keep = rng.random_bool(0.5);
    if keep {
        return Complex64::new(SQRT_2, 0.0);
    }

    flip_bit(&mut pauli.z, qubit);
    if is_dagger {
        Complex64::new(0.0, SQRT_2)
    } else {
        Complex64::new(0.0, -SQRT_2)
    }
}

fn backward_propagate_coalesced(
    ops: &[CoalescedOp],
    observable: &PauliVec,
    rng: &mut impl Rng,
) -> (PauliVec, Complex64) {
    let mut pauli = PauliVec {
        x: observable.x.clone(),
        z: observable.z.clone(),
    };
    let mut weight = Complex64::new(1.0, 0.0);

    for op in ops.iter().rev() {
        match op {
            CoalescedOp::SmallCliff(gates) => {
                for (gate, targets) in gates.iter().rev() {
                    // Phase track Clifford conjugation, mirroring the
                    // SPD path (`conjugate_all_backward_phased`); the
                    // bare `propagate_backward` is Pauli-frame only
                    // and drops the `-1` from e.g. `HYH = -Y`.
                    weight *= clifford_conjugation_phase(gate, targets, &pauli);
                    propagate_backward(&mut pauli, gate, targets);
                }
            }
            CoalescedOp::T { qubit, is_dagger } => {
                weight *= branch_t_gate(&mut pauli, *qubit, *is_dagger, rng);
            }
        }
    }

    (pauli, weight)
}

fn count_t_gates(circuit: &Circuit) -> usize {
    circuit
        .instructions
        .iter()
        .filter(|inst| {
            matches!(
                inst,
                Instruction::Gate {
                    gate: Gate::T | Gate::Tdg,
                    ..
                }
            )
        })
        .count()
}

pub struct SppResult {
    pub expectations: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub num_samples: usize,
    pub t_count: usize,
    pub nonzero_fraction: f64,
}

fn estimate_qubit_expectation(
    ops: &[CoalescedOp],
    qubit: usize,
    num_words: usize,
    num_samples: usize,
    seed: u64,
) -> (f64, f64, usize) {
    let obs = PauliVec::z_on_qubit(num_words, qubit);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut nonzero = 0usize;

    for i in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            weight.re
        } else {
            0.0
        };

        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = if num_samples > 1 {
        m2 / (num_samples - 1) as f64
    } else {
        0.0
    };
    let std_error = (variance / num_samples as f64).sqrt();
    (mean, std_error, nonzero)
}

/// Pauli axis for a joint-observable term. Identity factors are omitted
/// from the term list and contribute trivially.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliAxis {
    X,
    Y,
    Z,
}

/// One non-identity factor of a joint Pauli observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// Result of running SPP on a joint Pauli observable.
#[derive(Debug, Clone)]
pub struct SppObservableResult {
    pub mean: f64,
    pub std_error: f64,
    pub variance: f64,
    pub num_samples: usize,
    pub nonzero_fraction: f64,
    pub t_count: usize,
}

/// Result of running SPD on a joint Pauli observable.
#[derive(Debug, Clone)]
pub struct SpdObservableResult {
    pub mean: f64,
    pub t_count: usize,
    pub peak_terms: usize,
    pub total_discarded: f64,
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

/// Estimate `⟨0^n| U† P U |0^n⟩` for joint Pauli observable `P` on a
/// Clifford+T circuit `U` via stochastic Pauli propagation.
///
/// Each sample backward-propagates the observable through the circuit
/// (Clifford segments as coalesced gate runs, T gates via a stochastic
/// Pauli branch that records a complex weight). The
/// contribution is `Re(weight)` when the final Pauli is diagonal in
/// `{I, Z}` (i.e. evaluates trivially on `|0^n⟩`), else zero.
pub fn run_spp_observable(
    circuit: &Circuit,
    observable: &[PauliTerm],
    num_samples: usize,
    seed: u64,
) -> Result<SppObservableResult> {
    validate_clifford_t_unitary(circuit, "SPP observable")?;
    let n = circuit.num_qubits;
    let t_count = count_t_gates(circuit);
    let ops = coalesce_cliffords(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut nonzero = 0usize;

    for i in 0..num_samples {
        let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
        let val = if pauli.is_diagonal() {
            nonzero += 1;
            (obs_coeff * weight).re
        } else {
            0.0
        };
        let delta = val - mean;
        mean += delta / (i + 1) as f64;
        let delta2 = val - mean;
        m2 += delta * delta2;
    }

    let variance = if num_samples > 1 {
        m2 / (num_samples - 1) as f64
    } else {
        0.0
    };
    let std_error = (variance / num_samples.max(1) as f64).sqrt();
    let nonzero_fraction = nonzero as f64 / num_samples.max(1) as f64;

    Ok(SppObservableResult {
        mean,
        std_error,
        variance,
        num_samples,
        nonzero_fraction,
        t_count,
    })
}

pub fn run_spp(circuit: &Circuit, num_samples: usize, seed: u64) -> Result<SppResult> {
    validate_clifford_t_unitary(circuit, "SPP")?;
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_t_gates(circuit);
    let ops = coalesce_cliffords(circuit);

    #[cfg(feature = "parallel")]
    let results: Vec<(f64, f64, usize)> = {
        use rayon::prelude::*;
        (0..n)
            .into_par_iter()
            .map(|q| {
                estimate_qubit_expectation(
                    &ops,
                    q,
                    num_words,
                    num_samples,
                    seed.wrapping_add(q as u64),
                )
            })
            .collect()
    };

    #[cfg(not(feature = "parallel"))]
    let results: Vec<(f64, f64, usize)> = (0..n)
        .map(|q| {
            estimate_qubit_expectation(&ops, q, num_words, num_samples, seed.wrapping_add(q as u64))
        })
        .collect();

    let mut expectations = Vec::with_capacity(n);
    let mut std_errors = Vec::with_capacity(n);
    let mut total_nonzero = 0usize;

    for (mean, std_error, nonzero) in &results {
        expectations.push(*mean);
        std_errors.push(*std_error);
        total_nonzero += nonzero;
    }

    let nonzero_fraction = total_nonzero as f64 / (n * num_samples) as f64;

    Ok(SppResult {
        expectations,
        std_errors,
        num_samples,
        t_count,
        nonzero_fraction,
    })
}

#[cfg(test)]
fn spp_to_probabilities(result: &SppResult) -> Vec<f64> {
    super::expectations_to_marginals(&result.expectations)
        .into_iter()
        .flat_map(|(p0, p1)| [p0, p1])
        .collect()
}

// ---------------------------------------------------------------------------
// Deterministic Sparse Pauli Dynamics (SPD)
// ---------------------------------------------------------------------------

use std::collections::HashMap;

#[inline(always)]
fn pv_get_bit(words: &[u64], qubit: usize) -> bool {
    (words[qubit / 64] >> (qubit % 64)) & 1 != 0
}

#[inline(always)]
fn clifford_conjugation_phase(gate: &Gate, targets: &[usize], pauli: &PauliVec) -> Complex64 {
    match gate {
        Gate::H => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) && pv_get_bit(&pauli.z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::S => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Sdg => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::X => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Y => {
            let q = targets[0];
            let xq = pv_get_bit(&pauli.x, q);
            let zq = pv_get_bit(&pauli.z, q);
            if xq ^ zq {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Z => {
            let q = targets[0];
            if pv_get_bit(&pauli.x, q) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SX => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(0.0, 1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::SXdg => {
            let q = targets[0];
            if pv_get_bit(&pauli.z, q) {
                Complex64::new(0.0, -1.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        Gate::Cz => {
            let q0 = targets[0];
            let q1 = targets[1];
            if pv_get_bit(&pauli.x, q0) && pv_get_bit(&pauli.x, q1) {
                Complex64::new(-1.0, 0.0)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
        _ => Complex64::new(1.0, 0.0),
    }
}

struct WeightedPauliSum {
    terms: HashMap<PauliVec, Complex64>,
    scratch: Vec<(PauliVec, Complex64)>,
}

impl WeightedPauliSum {
    fn new() -> Self {
        Self {
            terms: HashMap::new(),
            scratch: Vec::new(),
        }
    }

    fn insert(&mut self, pauli: PauliVec, coeff: Complex64) {
        let entry = self.terms.entry(pauli).or_insert(Complex64::new(0.0, 0.0));
        *entry += coeff;
    }

    fn conjugate_all_backward_phased(&mut self, gate: &Gate, targets: &[usize]) {
        // Clifford conjugation is a bijection on Pauli strings: the symplectic
        // action on the (x, z) bits is invertible, so distinct keys map to
        // distinct keys and no coefficient merging occurs. Reuse `scratch` to
        // avoid a per-gate heap allocation, drain into it (which keeps the
        // map's bucket capacity), then re-insert directly without the
        // accumulate path.
        self.scratch.clear();
        self.scratch.extend(self.terms.drain());
        for (mut pauli, coeff) in self.scratch.drain(..) {
            let phase = clifford_conjugation_phase(gate, targets, &pauli);
            propagate_backward(&mut pauli, gate, targets);
            self.terms.insert(pauli, coeff * phase);
        }
    }

    fn branch_t_deterministic(&mut self, qubit: usize, is_dagger: bool) {
        let old_terms: Vec<(PauliVec, Complex64)> = self.terms.drain().collect();
        let inv_sqrt2 = 1.0 / SQRT_2;
        let keep_coeff = Complex64::new(inv_sqrt2, 0.0);
        let flip_coeff = if is_dagger {
            Complex64::new(0.0, inv_sqrt2)
        } else {
            Complex64::new(0.0, -inv_sqrt2)
        };

        for (pauli, coeff) in old_terms {
            if !pauli.has_x_or_y(qubit) {
                self.insert(pauli, coeff);
                continue;
            }

            let pauli_keep = pauli.clone();
            let mut pauli_flip = pauli;
            flip_bit(&mut pauli_flip.z, qubit);

            self.insert(pauli_keep, coeff * keep_coeff);
            self.insert(pauli_flip, coeff * flip_coeff);
        }
    }

    fn truncate(&mut self, epsilon: f64) -> f64 {
        let mut discarded = 0.0;
        self.terms.retain(|_, coeff| {
            if coeff.norm() < epsilon {
                discarded += coeff.norm();
                false
            } else {
                true
            }
        });
        discarded
    }

    fn diagonal_expectation(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for (pauli, coeff) in &self.terms {
            if pauli.is_diagonal() {
                sum += coeff;
            }
        }
        sum.re
    }
}

pub struct SpdResult {
    pub expectations: Vec<f64>,
    pub t_count: usize,
    pub max_terms: usize,
    pub total_discarded: f64,
}

/// Deterministic SPD on a joint Pauli observable.
///
/// Starts with the single weighted term `(observable, 1.0)`, backward-
/// propagates through every gate, branches each T into two terms with
/// `α / β` coefficients, and truncates terms whose magnitude falls below
/// `epsilon` whenever the sum exceeds `max_terms`. Returns
/// `⟨0^n| U† P U |0^n⟩` as the sum of remaining diagonal-term
/// coefficients.
pub fn run_spd_observable(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    validate_clifford_t_unitary(circuit, "SPD observable")?;
    let n = circuit.num_qubits;
    let t_count = count_t_gates(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;

    let mut sum = WeightedPauliSum::new();
    sum.insert(obs, obs_coeff);
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for inst in circuit.instructions.iter().rev() {
        if let Instruction::Gate { gate, targets } = inst {
            match gate {
                Gate::T => sum.branch_t_deterministic(targets[0], false),
                Gate::Tdg => sum.branch_t_deterministic(targets[0], true),
                _ => sum.conjugate_all_backward_phased(gate, targets),
            }
        }
        if max_terms > 0 && sum.terms.len() > max_terms {
            total_discarded += sum.truncate(epsilon);
        }
        check_spd_term_ceiling(sum.terms.len())?;
        if sum.terms.len() > peak_terms {
            peak_terms = sum.terms.len();
        }
    }

    if epsilon > 0.0 {
        total_discarded += sum.truncate(epsilon);
    }

    Ok(SpdObservableResult {
        mean: sum.diagonal_expectation(),
        t_count,
        peak_terms,
        total_discarded,
    })
}

pub fn run_spd(circuit: &Circuit, epsilon: f64, max_terms: usize) -> Result<SpdResult> {
    validate_clifford_t_unitary(circuit, "SPD")?;
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let t_count = count_t_gates(circuit);

    let mut expectations = Vec::with_capacity(n);
    let mut peak_terms = 0usize;
    let mut total_discarded = 0.0;

    for q in 0..n {
        let mut sum = WeightedPauliSum::new();
        sum.insert(PauliVec::z_on_qubit(num_words, q), Complex64::new(1.0, 0.0));

        for inst in circuit.instructions.iter().rev() {
            if let Instruction::Gate { gate, targets } = inst {
                match gate {
                    Gate::T => sum.branch_t_deterministic(targets[0], false),
                    Gate::Tdg => sum.branch_t_deterministic(targets[0], true),
                    _ => sum.conjugate_all_backward_phased(gate, targets),
                }
            }

            if max_terms > 0 && sum.terms.len() > max_terms {
                total_discarded += sum.truncate(epsilon);
            }
            check_spd_term_ceiling(sum.terms.len())?;

            if sum.terms.len() > peak_terms {
                peak_terms = sum.terms.len();
            }
        }

        if epsilon > 0.0 {
            total_discarded += sum.truncate(epsilon);
        }

        expectations.push(sum.diagonal_expectation());
    }

    Ok(SpdResult {
        expectations,
        t_count,
        max_terms: peak_terms,
        total_discarded,
    })
}

/// Inverse light cone of a Pauli observable under a circuit, computed
/// conservatively by gate-graph reachability.
///
/// Returns, for each instruction index in `circuit.instructions`, whether the
/// gate at that index can affect the backward-propagated observable. A gate is
/// in the cone if its support intersects the current cone-qubit set when the
/// circuit is traversed in reverse from the observable.
///
/// Exactness: a gate whose target set is disjoint from the cone-qubit set at
/// its backward-traversal depth conjugates the propagated observable trivially
/// (`U_k^dag P_k U_k = P_k` because the support of `P_k` is contained in the
/// cone set at depth `k`, and `U_k` acts as identity outside its targets).
/// Removing those gates from the backward pass is therefore exact.
pub fn inverse_light_cone(circuit: &Circuit, observable: &[PauliTerm]) -> Vec<bool> {
    let mut cone: std::collections::HashSet<usize> = observable.iter().map(|t| t.qubit).collect();
    let n_inst = circuit.instructions.len();
    let mut keep = vec![false; n_inst];

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        let targets: &[usize] = match inst {
            Instruction::Gate { targets, .. } => targets,
            _ => continue,
        };
        let touches = targets.iter().any(|q| cone.contains(q));
        if touches {
            keep[idx] = true;
            for &q in targets {
                cone.insert(q);
            }
        }
    }
    keep
}

/// SPD on a joint Pauli observable, restricted to the inverse light cone.
///
/// Identical in result to `run_spd_observable` for any Clifford+T circuit, but
/// skips gates whose support is disjoint from the propagated observable's
/// causal cone. For QEC syndrome and detector observables with bounded
/// spatial support, this turns the SPD cliff from a function of total T-count
/// into a function of in-cone T-count.
pub fn run_spd_observable_light_cone(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    validate_clifford_t_unitary(circuit, "light-cone SPD observable")?;
    let n = circuit.num_qubits;
    let t_count = count_t_gates(circuit);
    let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;
    let keep = inverse_light_cone(circuit, observable);

    let mut sum = WeightedPauliSum::new();
    sum.insert(obs, obs_coeff);
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        if !keep[idx] {
            continue;
        }
        if let Instruction::Gate { gate, targets } = inst {
            match gate {
                Gate::T => sum.branch_t_deterministic(targets[0], false),
                Gate::Tdg => sum.branch_t_deterministic(targets[0], true),
                _ => sum.conjugate_all_backward_phased(gate, targets),
            }
        }
        if max_terms > 0 && sum.terms.len() > max_terms {
            total_discarded += sum.truncate(epsilon);
        }
        check_spd_term_ceiling(sum.terms.len())?;
        if sum.terms.len() > peak_terms {
            peak_terms = sum.terms.len();
        }
    }

    if epsilon > 0.0 {
        total_discarded += sum.truncate(epsilon);
    }

    Ok(SpdObservableResult {
        mean: sum.diagonal_expectation(),
        t_count,
        peak_terms,
        total_discarded,
    })
}

#[cfg(test)]
fn spd_to_probabilities(result: &SpdResult) -> Vec<f64> {
    super::expectations_to_marginals(&result.expectations)
        .into_iter()
        .flat_map(|(p0, p1)| [p0, p1])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_no_t_gates_matches_propagate_backward() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::S, &[2]);

        let obs = PauliVec::z_on_qubit(1, 1);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ops = coalesce_cliffords(&circuit);
        let (result, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);

        let mut expected = PauliVec::z_on_qubit(1, 1);
        for inst in circuit.instructions.iter().rev() {
            if let Instruction::Gate { gate, targets } = inst {
                propagate_backward(&mut expected, gate, targets);
            }
        }

        assert_eq!(result.x, expected.x);
        assert_eq!(result.z, expected.z);
        assert!((weight - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn test_h_t_h_expectation_converges() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let obs = PauliVec::z_on_qubit(1, 0);
        let num_samples = 100_000;
        let mut sum = 0.0;
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ops = coalesce_cliffords(&circuit);

        for _ in 0..num_samples {
            let (pauli, weight) = backward_propagate_coalesced(&ops, &obs, &mut rng);
            if pauli.is_diagonal() {
                sum += weight.re;
            }
        }
        let mean = sum / num_samples as f64;

        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (mean - exact_z).abs() < 0.02,
            "mean={mean}, expected≈{exact_z}"
        );
    }

    #[test]
    fn test_branch_t_gate_passthrough_iz() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let mut pauli_i = PauliVec::new(1);
        let w = branch_t_gate(&mut pauli_i, 0, false, &mut rng);
        assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);

        let mut pauli_z = PauliVec::z_on_qubit(1, 0);
        let w = branch_t_gate(&mut pauli_z, 0, false, &mut rng);
        assert!((w - Complex64::new(1.0, 0.0)).norm() < 1e-14);
    }

    #[test]
    fn test_branch_t_gate_x_branches() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let num_samples = 10_000;
        let mut x_count = 0;
        let mut y_count = 0;

        for _ in 0..num_samples {
            let mut pauli = PauliVec::new(1);
            pauli.x[0] = 1;
            let w = branch_t_gate(&mut pauli, 0, false, &mut rng);
            assert!((w.norm() - SQRT_2).abs() < 1e-14);
            if pauli.z[0] == 0 {
                x_count += 1;
            } else {
                y_count += 1;
            }
        }

        let ratio = x_count as f64 / num_samples as f64;
        assert!(
            (ratio - 0.5).abs() < 0.03,
            "expected ~50/50, got {x_count}/{y_count}"
        );
    }

    fn marginal_p0(full_probs: &[f64], _n: usize, qubit: usize) -> f64 {
        let mut p0 = 0.0;
        for (i, &p) in full_probs.iter().enumerate() {
            if (i >> qubit) & 1 == 0 {
                p0 += p;
            }
        }
        p0
    }

    #[test]
    fn test_run_spp_vs_statevector_3q_2t() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::H, &[0]);

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        let spp = run_spp(&circuit, 200_000, 42).unwrap();
        assert_eq!(spp.t_count, 2);

        for q in 0..3 {
            let exact_p0 = marginal_p0(&sv_probs, 3, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            let err = (spp.expectations[q] - exact_ez).abs();
            assert!(
                err < 3.0 * spp.std_errors[q] + 0.01,
                "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
                spp.expectations[q],
                3.0 * spp.std_errors[q]
            );
        }
    }

    #[test]
    fn test_run_spp_vs_statevector_4q_4t() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        let spp = run_spp(&circuit, 200_000, 42).unwrap();
        assert_eq!(spp.t_count, 4);

        for q in 0..4 {
            let exact_p0 = marginal_p0(&sv_probs, 4, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            let err = (spp.expectations[q] - exact_ez).abs();
            assert!(
                err < 3.0 * spp.std_errors[q] + 0.01,
                "qubit {q}: spp={}, exact={exact_ez}, err={err}, 3σ={}",
                spp.expectations[q],
                3.0 * spp.std_errors[q]
            );
        }
    }

    #[test]
    fn test_spp_to_probabilities() {
        let result = SppResult {
            expectations: vec![0.5, -0.3],
            std_errors: vec![0.01, 0.01],
            num_samples: 1000,
            t_count: 2,
            nonzero_fraction: 0.8,
        };
        let probs = spp_to_probabilities(&result);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.75).abs() < 1e-14);
        assert!((probs[1] - 0.25).abs() < 1e-14);
        assert!((probs[2] - 0.35).abs() < 1e-14);
        assert!((probs[3] - 0.65).abs() < 1e-14);
    }

    #[test]
    fn test_spd_no_t_gates_exact() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::S, &[2]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 0);
        assert_eq!(spd.max_terms, 1);

        let spp = run_spp(&circuit, 10_000, 42).unwrap();
        for q in 0..3 {
            assert!(
                (spd.expectations[q] - spp.expectations[q]).abs() < 0.05,
                "qubit {q}: spd={}, spp={}",
                spd.expectations[q],
                spp.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_h_t_h_exact() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (spd.expectations[0] - exact_z).abs() < 1e-10,
            "spd={}, exact={exact_z}",
            spd.expectations[0]
        );
    }

    #[test]
    fn test_spd_vs_statevector_3q_2t() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[1]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::H, &[0]);

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 2);

        for q in 0..3 {
            let exact_p0 = marginal_p0(&sv_probs, 3, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            assert!(
                (spd.expectations[q] - exact_ez).abs() < 1e-10,
                "qubit {q}: spd={}, exact={exact_ez}",
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_vs_statevector_4q_4t() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::Cx, &[2, 3]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert_eq!(spd.t_count, 4);

        for q in 0..4 {
            let exact_p0 = marginal_p0(&sv_probs, 4, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            assert!(
                (spd.expectations[q] - exact_ez).abs() < 1e-10,
                "qubit {q}: spd={}, exact={exact_ez}",
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_truncation() {
        let mut circuit = Circuit::new(4, 0);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::T, &[3]);
        for q in 0..4 {
            circuit.add_gate(Gate::H, &[q]);
        }

        let exact = run_spd(&circuit, 0.0, 0).unwrap();
        let approx = run_spd(&circuit, 1e-6, 0).unwrap();

        for q in 0..4 {
            assert!(
                (exact.expectations[q] - approx.expectations[q]).abs() < 1e-4,
                "qubit {q}: exact={}, approx={}",
                exact.expectations[q],
                approx.expectations[q]
            );
        }
    }

    #[test]
    fn test_spd_with_tdg() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Tdg, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        let exact_z = std::f64::consts::FRAC_1_SQRT_2;
        assert!(
            (spd.expectations[0] - exact_z).abs() < 1e-10,
            "spd={}, exact={exact_z}",
            spd.expectations[0]
        );
    }

    #[test]
    fn test_spd_to_probabilities() {
        let result = SpdResult {
            expectations: vec![0.5, -0.3],
            t_count: 2,
            max_terms: 4,
            total_discarded: 0.0,
        };
        let probs = spd_to_probabilities(&result);
        assert_eq!(probs.len(), 4);
        assert!((probs[0] - 0.75).abs() < 1e-14);
        assert!((probs[1] - 0.25).abs() < 1e-14);
        assert!((probs[2] - 0.35).abs() < 1e-14);
        assert!((probs[3] - 0.65).abs() < 1e-14);
    }

    #[test]
    fn test_spd_h_t_h_t_h_phase_regression() {
        let mut circuit = Circuit::new(1, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &circuit, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();
        let exact_ez = 2.0 * sv_probs[0] - 1.0;

        let spd = run_spd(&circuit, 0.0, 0).unwrap();
        assert!(
            (spd.expectations[0] - exact_ez).abs() < 1e-10,
            "spd={}, exact={exact_ez}",
            spd.expectations[0]
        );
    }

    #[test]
    fn test_spd_vs_statevector_14q_clifford_t() {
        let c = crate::circuits::clifford_t_circuit(14, 10, 0.1, 42);
        let spd = run_spd(&c, 0.0, 0).unwrap();

        let sv = crate::sim::run_with(crate::sim::BackendKind::Statevector, &c, 42).unwrap();
        let sv_probs = sv.probabilities.unwrap().to_vec();

        for q in 0..14 {
            let exact_p0 = marginal_p0(&sv_probs, 14, q);
            let exact_ez = 2.0 * exact_p0 - 1.0;
            assert!(
                (spd.expectations[q] - exact_ez).abs() < 1e-8,
                "qubit {q}: spd={}, exact={exact_ez}",
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn coalesce_cliffords_long_clifford_run_stays_phase_correct() {
        // Regression: the SmallCliff path threads
        // `clifford_conjugation_phase` through every gate so global
        // signs from `HYH = -Y`, `SYS† = -X`, etc. are preserved.
        // Confirm SPP matches the analytical SPD on a long Clifford
        // run.
        let mut circuit = Circuit::new(4, 0);
        for _ in 0..20 {
            for q in 0..4 {
                circuit.add_gate(Gate::H, &[q]);
                circuit.add_gate(Gate::S, &[q]);
            }
            circuit.add_gate(Gate::Cx, &[0, 1]);
            circuit.add_gate(Gate::Cz, &[2, 3]);
        }
        let spp = run_spp(&circuit, 4_000, 42).unwrap();
        let spd = run_spd(&circuit, 1e-10, 16_384).unwrap();
        for q in 0..4 {
            assert!(
                (spp.expectations[q] - spd.expectations[q]).abs() < 0.08,
                "qubit {q}: spp={}, spd={}",
                spp.expectations[q],
                spd.expectations[q]
            );
        }
    }

    #[test]
    fn run_spp_pure_clifford() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let result = run_spp(&circuit, 16, 42).unwrap();
        assert_eq!(result.expectations.len(), 2);
        assert!(result.t_count == 0);
    }

    #[test]
    fn run_spd_pure_clifford() {
        let mut circuit = Circuit::new(2, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        let result = run_spd(&circuit, 0.0, 1024).unwrap();
        let probs = spd_to_probabilities(&result);
        assert_eq!(probs.len(), 4);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 2.0).abs() < 1e-9);
    }

    #[test]
    fn light_cone_excludes_disjoint_gates() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[3]);
        circuit.add_gate(Gate::T, &[3]);
        circuit.add_gate(Gate::H, &[3]);

        let obs = [PauliTerm::z(0)];
        let keep = inverse_light_cone(&circuit, &obs);
        assert_eq!(keep, vec![true, true, true, false, false, false]);
    }

    #[test]
    fn light_cone_follows_entangling_gates() {
        let mut circuit = Circuit::new(3, 0);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::H, &[2]);
        circuit.add_gate(Gate::H, &[0]);

        let obs = [PauliTerm::z(2)];
        let keep = inverse_light_cone(&circuit, &obs);
        assert_eq!(keep, vec![true, true, true, false]);
    }

    #[test]
    fn light_cone_spd_matches_unrestricted_spd() {
        let mut circuit = Circuit::new(5, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::H, &[4]);
        circuit.add_gate(Gate::T, &[4]);
        circuit.add_gate(Gate::Cx, &[3, 4]);
        circuit.add_gate(Gate::T, &[3]);

        let obs = [PauliTerm::z(0)];
        let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
        let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
        assert!(
            (full.mean - cone.mean).abs() < 1e-12,
            "full={} cone={}",
            full.mean,
            cone.mean
        );
        assert!(cone.peak_terms <= full.peak_terms);
    }

    #[test]
    fn light_cone_skips_most_gates_on_disjoint_t_block() {
        let mut circuit = Circuit::new(6, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::T, &[0]);
        circuit.add_gate(Gate::H, &[0]);
        for _ in 0..6 {
            circuit.add_gate(Gate::H, &[3]);
            circuit.add_gate(Gate::T, &[3]);
            circuit.add_gate(Gate::Cx, &[3, 4]);
            circuit.add_gate(Gate::T, &[4]);
            circuit.add_gate(Gate::Cx, &[4, 5]);
            circuit.add_gate(Gate::T, &[5]);
        }

        let obs = [PauliTerm::z(0)];
        let keep = inverse_light_cone(&circuit, &obs);
        let kept = keep.iter().filter(|b| **b).count();
        assert_eq!(kept, 3, "only the H-T-H block on q0 should be kept");

        let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
        let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
        assert!((full.mean - cone.mean).abs() < 1e-12);
    }

    #[test]
    fn light_cone_spd_matches_on_entangled_observable() {
        let mut circuit = Circuit::new(4, 0);
        circuit.add_gate(Gate::H, &[0]);
        circuit.add_gate(Gate::Cx, &[0, 1]);
        circuit.add_gate(Gate::T, &[1]);
        circuit.add_gate(Gate::Cx, &[1, 2]);
        circuit.add_gate(Gate::T, &[2]);
        circuit.add_gate(Gate::H, &[3]);
        circuit.add_gate(Gate::T, &[3]);

        let obs = [PauliTerm::z(0), PauliTerm::z(2)];
        let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
        let cone = run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
        assert!(
            (full.mean - cone.mean).abs() < 1e-12,
            "full={} cone={}",
            full.mean,
            cone.mean
        );
    }
}
