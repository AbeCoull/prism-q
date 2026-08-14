//! Weighted Pauli-sum observables: construction and arithmetic, qubit-wise-
//! commuting grouping, and the grouped moment accumulation the statevector
//! route evaluates mean and variance with.

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
