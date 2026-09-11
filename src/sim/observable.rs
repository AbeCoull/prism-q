//! Weighted Pauli-sum observables: construction and arithmetic, qubit-wise-
//! commuting grouping, and the grouped moment accumulation the statevector
//! route evaluates mean and variance with. Also holds the joint-Pauli mask
//! reduction and the expectation kernels the backends share.

use std::sync::OnceLock;

use num_complex::Complex64;

use crate::circuit::Circuit;
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::sim::RunMetadata;
use crate::sim::unified_pauli::{PauliAxis, PauliTerm};

/// Weighted sum of joint Pauli observables, `H = sum_k c_k P_k`.
///
/// Terms are kept canonical: factors sorted by qubit, identical Pauli strings
/// merged by summing coefficients, terms ordered by string. An empty factor
/// list is the identity and contributes its coefficient as a constant offset.
/// The qubit-wise-commuting grouping the grouped evaluation route uses is
/// computed lazily and cached; mutation invalidates the cache.
#[derive(Debug, Clone, Default)]
pub struct PauliObservable {
    terms: Vec<(f64, Vec<PauliTerm>)>,
    grouping: OnceLock<Grouping>,
}

impl PauliObservable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from `(coefficient, factors)` pairs, the Hamiltonian shape
    /// [`Simulate::expectation_gradient`] takes.
    ///
    /// [`Simulate::expectation_gradient`]: crate::sim::Simulate::expectation_gradient
    pub fn from_terms(terms: impl IntoIterator<Item = (f64, Vec<PauliTerm>)>) -> Result<Self> {
        let mut observable = Self::new();
        for (coefficient, factors) in terms {
            observable.add_term(coefficient, factors)?;
        }
        Ok(observable)
    }

    /// Add `coefficient` times the Pauli string `factors`, merging into an
    /// existing term with the same string.
    ///
    /// # Errors
    /// Rejects a non-finite coefficient and duplicate factors on one qubit.
    pub fn add_term(&mut self, coefficient: f64, mut factors: Vec<PauliTerm>) -> Result<()> {
        if !coefficient.is_finite() {
            return Err(PrismError::InvalidParameter {
                message: format!("observable coefficient {coefficient} is not finite"),
            });
        }
        factors.sort_unstable_by_key(|term| term.qubit);
        if let Some(pair) = factors
            .windows(2)
            .find(|pair| pair[0].qubit == pair[1].qubit)
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    pair[0].qubit
                ),
            });
        }
        self.merge_term(coefficient, factors);
        Ok(())
    }

    fn merge_term(&mut self, coefficient: f64, factors: Vec<PauliTerm>) {
        match self
            .terms
            .binary_search_by(|(_, existing)| existing.as_slice().cmp(&factors))
        {
            Ok(i) => self.terms[i].0 += coefficient,
            Err(i) => self.terms.insert(i, (coefficient, factors)),
        }
        self.grouping = OnceLock::new();
    }

    /// Canonical `(coefficient, factors)` pairs, ordered by Pauli string.
    pub fn terms(&self) -> &[(f64, Vec<PauliTerm>)] {
        &self.terms
    }

    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Number of qubit-wise-commuting groups, computing the grouping if
    /// needed. Identity terms belong to no group.
    pub fn num_groups(&self) -> usize {
        self.grouping().groups.len()
    }

    pub(crate) fn grouping(&self) -> &Grouping {
        self.grouping.get_or_init(|| compute_grouping(&self.terms))
    }
}

impl std::ops::Add for PauliObservable {
    type Output = PauliObservable;

    fn add(mut self, rhs: PauliObservable) -> PauliObservable {
        for (coefficient, factors) in rhs.terms {
            self.merge_term(coefficient, factors);
        }
        self
    }
}

impl std::ops::Sub for PauliObservable {
    type Output = PauliObservable;

    fn sub(self, rhs: PauliObservable) -> PauliObservable {
        self + (-rhs)
    }
}

impl std::ops::Neg for PauliObservable {
    type Output = PauliObservable;

    fn neg(mut self) -> PauliObservable {
        for (coefficient, _) in &mut self.terms {
            *coefficient = -*coefficient;
        }
        self
    }
}

impl std::ops::Mul<f64> for PauliObservable {
    type Output = PauliObservable;

    fn mul(mut self, rhs: f64) -> PauliObservable {
        for (coefficient, _) in &mut self.terms {
            *coefficient *= rhs;
        }
        self
    }
}

/// Weighted-observable expectation with the grouped-measurement variance.
#[derive(Debug, Clone)]
pub struct ObservableExpectation {
    /// `<H> = sum_k c_k <P_k>`, including identity-term constants.
    pub mean: f64,
    /// Sum of per-group variances `Var(H_g) = <H_g^2> - <H_g>^2`, each exact
    /// in the output state. This is the variance of a grouped measurement
    /// estimate drawing one shot per commuting group; with `S_g` shots on
    /// group `g` the estimator variance is `sum_g Var(H_g) / S_g`. It equals
    /// `Var(H)` of the full operator only when one group covers every term,
    /// since cross-group covariances are excluded. `None` on a route that
    /// evaluates term by term without the grouped traversal.
    pub variance: Option<f64>,
    /// Per-group `Var(H_g)` in grouping order, the input to shot allocation.
    pub group_variances: Option<Vec<f64>>,
    /// Standard error of `mean` when a sampling route estimated the per-term
    /// values, `None` for analytic routes.
    pub std_error: Option<f64>,
    pub metadata: RunMetadata,
}

/// Qubit-wise-commuting grouping over an observable's non-identity terms.
#[derive(Debug, Clone)]
pub(crate) struct Grouping {
    pub(crate) groups: Vec<QwcGroup>,
}

/// One commuting set: member term indices plus per-qubit axis-assignment
/// words. A qubit's assigned axis is X when only its `axis_x` bit is set, Z
/// when only `axis_z`, Y when both; unset bits are unconstrained.
#[derive(Debug, Clone)]
pub(crate) struct QwcGroup {
    pub(crate) term_indices: Vec<usize>,
    axis_x: Vec<u64>,
    axis_z: Vec<u64>,
}

impl QwcGroup {
    fn accepts(&self, tx: &[u64], tz: &[u64]) -> bool {
        for w in 0..tx.len() {
            let shared = (tx[w] | tz[w]) & (self.axis_x[w] | self.axis_z[w]);
            if ((tx[w] ^ self.axis_x[w]) | (tz[w] ^ self.axis_z[w])) & shared != 0 {
                return false;
            }
        }
        true
    }

    fn absorb(&mut self, index: usize, tx: &[u64], tz: &[u64]) {
        for w in 0..tx.len() {
            self.axis_x[w] |= tx[w];
            self.axis_z[w] |= tz[w];
        }
        self.term_indices.push(index);
    }

    /// Whether every assigned axis is Z, so members evaluate on the
    /// unrotated state.
    pub(crate) fn is_z_only(&self) -> bool {
        self.axis_x.iter().all(|&word| word == 0)
    }

    /// Rotation taking every assigned axis to Z: H on X qubits, Sdg then H on
    /// Y qubits. Conjugation by it sends each member string to a plus-sign Z
    /// string on the same support.
    pub(crate) fn basis_rotation_circuit(&self, num_qubits: usize) -> Circuit {
        let mut circuit = Circuit::new(num_qubits, 0);
        for qubit in 0..num_qubits.min(self.axis_x.len() * 64) {
            let bit = 1u64 << (qubit % 64);
            if self.axis_x[qubit / 64] & bit != 0 {
                if self.axis_z[qubit / 64] & bit != 0 {
                    circuit.add_gate(Gate::Sdg, &[qubit]);
                }
                circuit.add_gate(Gate::H, &[qubit]);
            }
        }
        circuit
    }
}

fn compute_grouping(terms: &[(f64, Vec<PauliTerm>)]) -> Grouping {
    let max_qubit = terms
        .iter()
        .flat_map(|(_, factors)| factors.iter())
        .map(|term| term.qubit)
        .max();
    let num_words = max_qubit.map_or(0, |q| q / 64 + 1);

    // First-fit-decreasing on factor count, index-stable for determinism.
    let mut order: Vec<usize> = (0..terms.len())
        .filter(|&i| !terms[i].1.is_empty())
        .collect();
    order.sort_by(|&a, &b| terms[b].1.len().cmp(&terms[a].1.len()).then(a.cmp(&b)));

    let mut groups: Vec<QwcGroup> = Vec::new();
    let mut tx = vec![0u64; num_words];
    let mut tz = vec![0u64; num_words];
    for &index in &order {
        tx.fill(0);
        tz.fill(0);
        for term in &terms[index].1 {
            let bit = 1u64 << (term.qubit % 64);
            match term.axis {
                PauliAxis::X => tx[term.qubit / 64] |= bit,
                PauliAxis::Z => tz[term.qubit / 64] |= bit,
                PauliAxis::Y => {
                    tx[term.qubit / 64] |= bit;
                    tz[term.qubit / 64] |= bit;
                }
            }
        }
        match groups.iter_mut().find(|group| group.accepts(&tx, &tz)) {
            Some(group) => group.absorb(index, &tx, &tz),
            None => groups.push(QwcGroup {
                term_indices: vec![index],
                axis_x: tx.clone(),
                axis_z: tz.clone(),
            }),
        }
    }
    Grouping { groups }
}

/// First two moments `(sum_j p_j h(j), sum_j p_j h(j)^2)` of one group
/// operator `h(j) = sum_i c_i (-1)^popcount(j & z_i)`, normalized by `norm`.
///
/// The z-only accumulator family of `pauli_expectations_from_masks`, combined
/// per element before squaring so the group variance comes from the same
/// traversal as its mean.
pub(crate) fn weighted_group_moments(
    state: &[Complex64],
    zmasks: &[usize],
    coefficients: &[f64],
    norm: f64,
) -> (f64, f64) {
    if norm == 0.0 {
        return (0.0, 0.0);
    }

    let accumulate = |acc: &mut (f64, f64), base: usize, block: &[Complex64]| {
        for (offset, amp) in block.iter().enumerate() {
            let j = base + offset;
            let mut h = 0.0;
            for (&zmask, &c) in zmasks.iter().zip(coefficients) {
                h += if (j & zmask).count_ones() & 1 == 1 {
                    -c
                } else {
                    c
                };
            }
            let weighted = amp.norm_sqr() * h;
            acc.0 += weighted;
            acc.1 += weighted * h;
        }
    };

    #[cfg(feature = "parallel")]
    if state.len() >= crate::backend::MIN_PAR_REDUCE_ELEMS {
        use rayon::prelude::*;
        let chunk = crate::backend::MIN_PAR_ELEMS;
        let (m1, m2) = state
            .par_chunks(chunk)
            .enumerate()
            .fold(
                || (0.0, 0.0),
                |mut acc, (c, block)| {
                    accumulate(&mut acc, c * chunk, block);
                    acc
                },
            )
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        return (m1 / norm, m2 / norm);
    }

    let mut acc = (0.0, 0.0);
    accumulate(&mut acc, 0, state);
    (acc.0 / norm, acc.1 / norm)
}

#[cfg(test)]
#[path = "observable_tests.rs"]
mod tests;

/// Reject out-of-range qubits and duplicate factors in a joint Pauli
/// observable.
///
/// Same checks [`pauli_masks`] makes, without its `1 << qubit` mask width, so
/// it also covers the backends that run past 64 qubits.
pub(crate) fn validate_observable(observable: &[PauliTerm], num_qubits: usize) -> Result<()> {
    let mut seen = vec![false; num_qubits];
    for term in observable {
        if term.qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
    }
    Ok(())
}

/// Validate a joint Pauli observable and reduce it to `(Xmask, Zmask, #Y)`,
/// where `Xmask` covers X and Y factors and `Zmask` covers Z and Y factors.
pub(crate) fn pauli_masks(
    observable: &[PauliTerm],
    num_qubits: usize,
) -> Result<(usize, usize, u32)> {
    let mut xmask = 0usize;
    let mut zmask = 0usize;
    let mut num_y = 0u32;
    let mut seen = vec![false; num_qubits];
    for term in observable {
        if term.qubit >= num_qubits {
            return Err(PrismError::InvalidQubit {
                index: term.qubit,
                register_size: num_qubits,
            });
        }
        if seen[term.qubit] {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "joint Pauli observable has duplicate factor on qubit {}",
                    term.qubit
                ),
            });
        }
        seen[term.qubit] = true;
        let bit = 1usize << term.qubit;
        match term.axis {
            PauliAxis::X => xmask |= bit,
            PauliAxis::Z => zmask |= bit,
            PauliAxis::Y => {
                xmask |= bit;
                zmask |= bit;
                num_y += 1;
            }
        }
    }
    Ok((xmask, zmask, num_y))
}

/// Complex Pauli sandwich `⟨λ|P|φ⟩`, where `P` acts as
/// `P|j⟩ = i^{#Y}·(-1)^{popcount(j & Zmask)}·|j ⊕ Xmask⟩`. Returns the raw
/// (unnormalized) complex value. The adjoint gradient engine uses this with
/// distinct `λ` and `φ`; `pauli_expectation_from_masks` is the `λ = φ` case.
///
/// Inlined explicitly: this is the adjoint engine's inner reduction, called
/// once per parameter, and leaving the decision to LTO ties it to how many
/// other callers the function happens to have.
#[inline]
pub(crate) fn pauli_sandwich(
    lambda: &[Complex64],
    phi: &[Complex64],
    xmask: usize,
    zmask: usize,
    num_y: u32,
) -> Complex64 {
    let term = |j: usize, amp: Complex64| {
        let partner = lambda[j ^ xmask];
        let sign = if (j & zmask).count_ones() & 1 == 1 {
            -1.0
        } else {
            1.0
        };
        partner.conj() * amp * sign
    };

    // Higher threshold than gate kernels: the sandwich is a single lightweight
    // O(N) reduction, so Rayon fan-out only pays off past 2^16 elements. Below
    // that (and for a multi-term Hamiltonian's many small reductions) the
    // sequential path is faster.
    #[cfg(feature = "parallel")]
    const SANDWICH_MIN_PAR_QUBITS: usize = 16;
    #[cfg(feature = "parallel")]
    let acc: Complex64 = if phi.len() >= (1 << SANDWICH_MIN_PAR_QUBITS) {
        use rayon::prelude::*;
        phi.par_iter()
            .enumerate()
            .map(|(j, &amp)| term(j, amp))
            .sum()
    } else {
        phi.iter().enumerate().map(|(j, &amp)| term(j, amp)).sum()
    };
    #[cfg(not(feature = "parallel"))]
    let acc: Complex64 = phi.iter().enumerate().map(|(j, &amp)| term(j, amp)).sum();

    acc * i_pow(num_y)
}

/// `i^{num_y}`, the phase a joint Pauli picks up from its Y factors.
#[inline]
pub(crate) fn i_pow(num_y: u32) -> Complex64 {
    match num_y % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

/// Exact `⟨ψ|P|ψ⟩` from the reduced observable masks. Normalization
/// independent, so raw backend amplitudes are fine.
pub(crate) fn pauli_expectation_from_masks(
    state: &[Complex64],
    xmask: usize,
    zmask: usize,
    num_y: u32,
    norm: f64,
) -> f64 {
    if norm == 0.0 {
        return 0.0;
    }
    pauli_sandwich(state, state, xmask, zmask, num_y).re / norm
}

/// Exact `⟨ψ|P_i|ψ⟩` for every mask triple in one traversal of `state`.
///
/// Same value as [`pauli_expectation_from_masks`] per entry, to within the
/// association of the sum. A Z-only observable has `xmask == 0` and therefore
/// no Y factor, so its contribution is `±|amp|^2` and needs neither the partner
/// load nor complex arithmetic; the two families are accumulated separately for
/// that reason.
pub(crate) fn pauli_expectations_from_masks(
    state: &[Complex64],
    masks: &[(usize, usize, u32)],
    norm: f64,
) -> Vec<f64> {
    if norm == 0.0 {
        return vec![0.0; masks.len()];
    }
    if masks.len() < 2 {
        return masks
            .iter()
            .map(|&(xmask, zmask, num_y)| {
                pauli_expectation_from_masks(state, xmask, zmask, num_y, norm)
            })
            .collect();
    }

    let z_only: Vec<usize> = masks
        .iter()
        .filter(|&&(xmask, _, _)| xmask == 0)
        .map(|&(_, zmask, _)| zmask)
        .collect();
    let general: Vec<(usize, usize)> = masks
        .iter()
        .filter(|&&(xmask, _, _)| xmask != 0)
        .map(|&(xmask, zmask, _)| (xmask, zmask))
        .collect();

    let accumulate = |z_acc: &mut [f64], g_acc: &mut [Complex64], base: usize, len: usize| {
        for j in base..base + len {
            let amp = state[j];
            let n2 = amp.norm_sqr();
            for (slot, &zmask) in z_acc.iter_mut().zip(z_only.iter()) {
                *slot += if (j & zmask).count_ones() & 1 == 1 {
                    -n2
                } else {
                    n2
                };
            }
            for (slot, &(xmask, zmask)) in g_acc.iter_mut().zip(general.iter()) {
                let partner = state[j ^ xmask];
                let sign = if (j & zmask).count_ones() & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
                *slot += partner.conj() * amp * sign;
            }
        }
    };

    let zeros = || {
        (
            vec![0.0f64; z_only.len()],
            vec![Complex64::new(0.0, 0.0); general.len()],
        )
    };
    let (mut z_sum, mut g_sum) = zeros();

    #[cfg(feature = "parallel")]
    if state.len() >= crate::backend::MIN_PAR_REDUCE_ELEMS {
        use rayon::prelude::*;
        let chunk = crate::backend::MIN_PAR_ELEMS;
        let (z, g) = state
            .par_chunks(chunk)
            .enumerate()
            .fold(zeros, |mut acc, (c, block)| {
                accumulate(&mut acc.0, &mut acc.1, c * chunk, block.len());
                acc
            })
            .reduce(zeros, |mut a, b| {
                for (slot, v) in a.0.iter_mut().zip(b.0) {
                    *slot += v;
                }
                for (slot, v) in a.1.iter_mut().zip(b.1) {
                    *slot += v;
                }
                a
            });
        return finish_expectations(masks, &z, &g, norm);
    }

    accumulate(&mut z_sum, &mut g_sum, 0, state.len());
    finish_expectations(masks, &z_sum, &g_sum, norm)
}

/// Interleave the two accumulator families back into observable order.
///
/// Entries of `z_sum` and `g_sum` are in `masks` order within their family:
/// the `i`-th `xmask == 0` entry of `masks` reads `z_sum[i]`, and the `i`-th
/// `xmask != 0` entry reads `g_sum[i]`.
pub(crate) fn finish_expectations(
    masks: &[(usize, usize, u32)],
    z_sum: &[f64],
    g_sum: &[Complex64],
    norm: f64,
) -> Vec<f64> {
    let (mut zi, mut gi) = (0, 0);
    masks
        .iter()
        .map(|&(xmask, _, num_y)| {
            if xmask == 0 {
                zi += 1;
                z_sum[zi - 1] / norm
            } else {
                gi += 1;
                (g_sum[gi - 1] * i_pow(num_y)).re / norm
            }
        })
        .collect()
}
