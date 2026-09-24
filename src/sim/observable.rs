//! Weighted Pauli-sum observables: construction and arithmetic, qubit-wise-
//! commuting grouping, and the grouped moment accumulation the statevector
//! route evaluates mean and variance with. Also holds the joint-Pauli mask
//! reduction and the expectation kernels the backends share.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use num_complex::Complex64;

use crate::error::{PrismError, Result};
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

    /// The constant term's coefficient and the rest of the sum.
    ///
    /// `Var(H + cI) = Var(H)`, so a variance squares the traceless part rather
    /// than the whole sum: at a large `c` the constant dominates both `<H^2>`
    /// and `<H>^2` and the difference loses the spread it was meant to report.
    pub fn split_identity(&self) -> (f64, PauliObservable) {
        let mut offset = 0.0;
        let mut rest = PauliObservable::new();
        for (coefficient, string) in &self.terms {
            if string.is_empty() {
                offset += coefficient;
            } else {
                rest.merge_term(*coefficient, string.clone());
            }
        }
        (offset, rest)
    }

    /// `H^2` as a Pauli sum, the second moment [`Simulate::observable_variance`]
    /// reads `Var(H) = <H^2> - <H>^2` from.
    ///
    /// Every coefficient of the square is real. Two Pauli strings either
    /// commute, and their product carries no phase, or anticommute, and the
    /// `(j, k)` and `(k, j)` products carry opposite imaginary phases that
    /// cancel. Phases are tracked as powers of `i` so that cancellation is
    /// exact rather than a subtraction of two nearly equal floats.
    ///
    /// Costs `T^2` string products over `T` terms, so it suits the tensor
    /// products and small Hermitian matrices an observable request names
    /// rather than a molecular Hamiltonian.
    ///
    /// [`Simulate::observable_variance`]: crate::sim::Simulate::observable_variance
    pub fn square(&self) -> PauliObservable {
        let mut accumulated: BTreeMap<Vec<PauliTerm>, f64> = BTreeMap::new();
        for (left, left_string) in &self.terms {
            for (right, right_string) in &self.terms {
                let (phase, product) = multiply_pauli_strings(left_string, right_string);
                if phase % 2 == 1 {
                    continue;
                }
                let sign = if phase == 0 { 1.0 } else { -1.0 };
                *accumulated.entry(product).or_insert(0.0) += sign * left * right;
            }
        }
        let norm = self.terms.iter().map(|(c, _)| c.abs()).sum::<f64>();
        let tolerance = f64::EPSILON * norm * norm * self.terms.len().max(1) as f64;
        let mut squared = PauliObservable::new();
        for (string, coefficient) in accumulated {
            if coefficient.abs() > tolerance {
                squared.merge_term(coefficient, string);
            }
        }
        squared
    }
}

/// Product of two sorted Pauli strings as `(power of i, string)`.
fn multiply_pauli_strings(left: &[PauliTerm], right: &[PauliTerm]) -> (u32, Vec<PauliTerm>) {
    let mut phase = 0u32;
    let mut product = Vec::with_capacity(left.len() + right.len());
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        let (a, b) = (left[i], right[j]);
        match a.qubit.cmp(&b.qubit) {
            std::cmp::Ordering::Less => {
                product.push(a);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                product.push(b);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                if let Some((step, axis)) = multiply_pauli_axes(a.axis, b.axis) {
                    phase = (phase + step) % 4;
                    product.push(PauliTerm::new(a.qubit, axis));
                }
                i += 1;
                j += 1;
            }
        }
    }
    product.extend_from_slice(&left[i..]);
    product.extend_from_slice(&right[j..]);
    (phase, product)
}

/// `a * b` on one qubit as `(power of i, axis)`, `None` when the two axes
/// agree and the product is the identity.
fn multiply_pauli_axes(a: PauliAxis, b: PauliAxis) -> Option<(u32, PauliAxis)> {
    use PauliAxis::{X, Y, Z};
    match (a, b) {
        (X, Y) => Some((1, Z)),
        (Y, Z) => Some((1, X)),
        (Z, X) => Some((1, Y)),
        (Y, X) => Some((3, Z)),
        (Z, Y) => Some((3, X)),
        (X, Z) => Some((3, Y)),
        _ => None,
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

    /// Qubit masks `(x, y)` of the group's X-assigned and Y-assigned qubits.
    /// [`rotate_to_z_basis`] on them sends each member string to a plus-sign Z
    /// string on the same support. Statevector widths fit one word.
    pub(crate) fn rotation_masks(&self) -> (usize, usize) {
        let (x, z) = (self.axis_x[0], self.axis_z[0]);
        ((x & !z) as usize, (x & z) as usize)
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
/// `h` is built a block of [`MOMENT_BLOCK`] indices at a time. Within a block
/// the high index bits are fixed, so each term reduces to a signed
/// coefficient on its low-bit pattern, and a Walsh-Hadamard transform over
/// those patterns yields `h` at every index of the block. That costs one
/// scalar step per term per block plus `log2(MOMENT_BLOCK)` butterflies per
/// index, where a per-index sum over terms costs a parity and an add per term
/// per index.
pub(crate) fn weighted_group_moments(
    state: &[Complex64],
    zmasks: &[usize],
    coefficients: &[f64],
    norm: f64,
) -> (f64, f64) {
    if norm == 0.0 {
        return (0.0, 0.0);
    }

    let accumulate = |acc: &mut (f64, f64), base: usize, chunk: &[Complex64]| {
        for (b, block) in chunk.chunks(MOMENT_BLOCK).enumerate() {
            let block_base = base + b * MOMENT_BLOCK;
            let mut h = [0.0f64; MOMENT_BLOCK];
            for (&zmask, &c) in zmasks.iter().zip(coefficients) {
                let flip = u64::from((block_base & zmask).count_ones() & 1) << 63;
                h[zmask & (MOMENT_BLOCK - 1)] += f64::from_bits(c.to_bits() ^ flip);
            }
            let mut half = 1;
            while half < MOMENT_BLOCK {
                for pair in h.chunks_exact_mut(2 * half) {
                    let (lo, hi) = pair.split_at_mut(half);
                    for (a, b) in lo.iter_mut().zip(hi) {
                        (*a, *b) = (*a + *b, *a - *b);
                    }
                }
                half *= 2;
            }
            for (amp, &hj) in block.iter().zip(&h) {
                let weighted = amp.norm_sqr() * hj;
                acc.0 += weighted;
                acc.1 += weighted * hj;
            }
        }
    };

    #[cfg(feature = "parallel")]
    if state.len() >= crate::backend::MIN_PAR_REDUCE_ELEMS {
        use rayon::prelude::*;
        let chunk = crate::backend::MIN_PAR_ELEMS.max(MOMENT_BLOCK);
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

/// Indices per Walsh-Hadamard block in [`weighted_group_moments`]. A state
/// shorter than a block reads only the leading entries, which are exact
/// because every mask then fits below the state length.
const MOMENT_BLOCK: usize = 64;

/// Copy `state` into `out` rotated so each qubit in `x_bits` reads in the X
/// basis and each in `y_bits` in the Y basis, as `H` and `H S†` do. The
/// butterflies drop the `1/sqrt(2)`, so the squared norm of `out` is that of
/// `state` times `2^popcount(x_bits | y_bits)`, an exact power of two for the
/// caller to fold into the moments' `norm`.
///
/// Qubits below `log2(`[`ROTATION_BLOCK`]`)` rotate a block at a time while
/// the block is copied in and still in L1. The rest take one sweep per pair of
/// qubits, each pair a radix-4 butterfly that loads and stores an amplitude
/// once for two levels.
pub(crate) fn rotate_to_z_basis(
    state: &[Complex64],
    out: &mut Vec<Complex64>,
    x_bits: usize,
    y_bits: usize,
) {
    out.resize(state.len(), Complex64::ZERO);
    let block = ROTATION_BLOCK.min(state.len());
    let levels = |range: std::ops::Range<usize>| {
        range
            .filter(|&q| (x_bits | y_bits) >> q & 1 == 1)
            .map(|q| (q, y_bits >> q & 1 == 1))
            .collect::<Vec<_>>()
    };
    let low = levels(0..block.trailing_zeros() as usize);
    let high = levels(block.trailing_zeros() as usize..state.len().trailing_zeros() as usize);

    let fill = |(dst, src): (&mut [Complex64], &[Complex64])| {
        dst.copy_from_slice(src);
        rotate_levels(dst, &low);
    };

    #[cfg(feature = "parallel")]
    if state.len() >= 1 << crate::backend::PARALLEL_THRESHOLD_QUBITS {
        use rayon::prelude::*;
        out.par_chunks_mut(block)
            .zip(state.par_chunks(block))
            .for_each(fill);
        for pair in high.chunks(2) {
            rotate_pair_par(out, pair);
        }
        return;
    }

    out.chunks_mut(block)
        .zip(state.chunks(block))
        .for_each(fill);
    rotate_levels(out, &high);
}

/// Amplitudes per block in [`rotate_to_z_basis`], 16 KiB.
const ROTATION_BLOCK: usize = 1 << 10;

/// `-i b` for a Y level, `b` for an X level: the second column of the
/// unnormalized `H` or `H S†`.
#[inline(always)]
fn twiddle<const Y: bool>(b: Complex64) -> Complex64 {
    if Y { Complex64::new(b.im, -b.re) } else { b }
}

#[inline(always)]
fn butterfly2<const Y: bool>(lo: &mut [Complex64], hi: &mut [Complex64]) {
    for (a, b) in lo.iter_mut().zip(hi) {
        let t = twiddle::<Y>(*b);
        (*a, *b) = (*a + t, *a - t);
    }
}

/// Levels `q1 < q2` on one amplitude from each quarter, the quarters indexed
/// by the two target bits as `00 01 10 11`.
#[inline(always)]
fn quad<const Y1: bool, const Y2: bool>(
    a: &mut Complex64,
    b: &mut Complex64,
    c: &mut Complex64,
    d: &mut Complex64,
) {
    let (tb, td) = (twiddle::<Y1>(*b), twiddle::<Y1>(*d));
    let (a1, b1, c1, d1) = (*a + tb, *a - tb, *c + td, *c - td);
    let (tc, td) = (twiddle::<Y2>(c1), twiddle::<Y2>(d1));
    (*a, *b, *c, *d) = (a1 + tc, b1 + td, a1 - tc, b1 - td);
}

#[inline(always)]
fn butterfly4<const Y1: bool, const Y2: bool>(
    a: &mut [Complex64],
    b: &mut [Complex64],
    c: &mut [Complex64],
    d: &mut [Complex64],
) {
    let n = a.len();
    let (b, c, d) = (&mut b[..n], &mut c[..n], &mut d[..n]);
    for i in 0..n {
        quad::<Y1, Y2>(&mut a[i], &mut b[i], &mut c[i], &mut d[i]);
    }
}

/// Apply ascending `(qubit, is_y)` levels to `amps`, pairing levels into
/// radix-4 sweeps from [`MIN_RADIX4_QUBIT`] up.
fn rotate_levels(amps: &mut [Complex64], levels: &[(usize, bool)]) {
    let mut rest = levels;
    while let Some((&(q1, y1), tail)) = rest.split_first() {
        match tail.first() {
            Some(&(q2, y2)) if q1 >= MIN_RADIX4_QUBIT || (q1, q2) == (0, 1) => {
                match (y1, y2) {
                    (false, false) => rotate4::<false, false>(amps, q1, q2),
                    (false, true) => rotate4::<false, true>(amps, q1, q2),
                    (true, false) => rotate4::<true, false>(amps, q1, q2),
                    (true, true) => rotate4::<true, true>(amps, q1, q2),
                }
                rest = &tail[1..];
            }
            _ => {
                if y1 {
                    rotate2::<true>(amps, q1);
                } else {
                    rotate2::<false>(amps, q1);
                }
                rest = tail;
            }
        }
    }
}

/// Lowest qubit a radix-4 sweep starts from, apart from qubits 0 and 1, which
/// pair as fixed groups of four. Starting at qubit 1 or 2 leaves quarters of 2
/// or 4 amplitudes, and those sweeps measured slower than two radix-2 sweeps.
const MIN_RADIX4_QUBIT: usize = 3;

fn rotate2<const Y: bool>(amps: &mut [Complex64], q: usize) {
    for pair in amps.chunks_exact_mut(2 << q) {
        let (lo, hi) = pair.split_at_mut(1 << q);
        butterfly2::<Y>(lo, hi);
    }
}

fn rotate4<const Y1: bool, const Y2: bool>(amps: &mut [Complex64], q1: usize, q2: usize) {
    if (q1, q2) == (0, 1) {
        for chunk in amps.chunks_exact_mut(4) {
            if let [a, b, c, d] = chunk {
                quad::<Y1, Y2>(a, b, c, d);
            }
        }
        return;
    }
    for quad in amps.chunks_exact_mut(2 << q2) {
        let (lo, hi) = quad.split_at_mut(1 << q2);
        for (l, h) in lo
            .chunks_exact_mut(2 << q1)
            .zip(hi.chunks_exact_mut(2 << q1))
        {
            let (a, b) = l.split_at_mut(1 << q1);
            let (c, d) = h.split_at_mut(1 << q1);
            butterfly4::<Y1, Y2>(a, b, c, d);
        }
    }
}

/// [`rotate_levels`] on one or two levels at or above `log2(`[`ROTATION_BLOCK`]`)`,
/// split into runs of [`MIN_PAR_ELEMS`](crate::backend::MIN_PAR_ELEMS) across
/// the pool.
#[cfg(feature = "parallel")]
fn rotate_pair_par(amps: &mut [Complex64], levels: &[(usize, bool)]) {
    use rayon::prelude::*;
    let run = crate::backend::MIN_PAR_ELEMS;
    match *levels {
        [(q, y)] => amps.par_chunks_mut(2 << q).for_each(|pair| {
            let (lo, hi) = pair.split_at_mut(1 << q);
            lo.par_chunks_mut(run)
                .zip(hi.par_chunks_mut(run))
                .for_each(|(lo, hi)| {
                    if y {
                        butterfly2::<true>(lo, hi)
                    } else {
                        butterfly2::<false>(lo, hi)
                    }
                });
        }),
        [(q1, y1), (q2, y2)] => amps.par_chunks_mut(2 << q2).for_each(|quad| {
            let (lo, hi) = quad.split_at_mut(1 << q2);
            lo.par_chunks_mut(2 << q1)
                .zip(hi.par_chunks_mut(2 << q1))
                .for_each(|(l, h)| {
                    let (a, b) = l.split_at_mut(1 << q1);
                    let (c, d) = h.split_at_mut(1 << q1);
                    a.par_chunks_mut(run)
                        .zip(b.par_chunks_mut(run))
                        .zip(c.par_chunks_mut(run))
                        .zip(d.par_chunks_mut(run))
                        .for_each(|(((a, b), c), d)| match (y1, y2) {
                            (false, false) => butterfly4::<false, false>(a, b, c, d),
                            (false, true) => butterfly4::<false, true>(a, b, c, d),
                            (true, false) => butterfly4::<true, false>(a, b, c, d),
                            (true, true) => butterfly4::<true, true>(a, b, c, d),
                        });
                });
        }),
        _ => unreachable!(),
    }
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

/// Rayon fan-out threshold for the sandwich reductions. Higher than the gate
/// kernels': a sandwich is a single lightweight O(N) reduction, so fan-out only
/// pays off past 2^16 elements. Below that (and for a multi-term Hamiltonian's
/// many small reductions) the sequential path is faster.
#[cfg(feature = "parallel")]
const SANDWICH_MIN_PAR_QUBITS: usize = 16;

/// Complex Pauli sandwich `⟨λ|P|φ⟩`, where `P` acts as
/// `P|j⟩ = i^{#Y}·(-1)^{popcount(j & Zmask)}·|j ⊕ Xmask⟩`. Returns the raw
/// (unnormalized) complex value. The adjoint gradient engine uses this with
/// distinct `λ` and `φ`; `pauli_expectation_from_masks` is the `λ = φ` case.
///
/// Inlined explicitly: the single-mask fallback of
/// [`pauli_sandwiches_from_masks`], [`pauli_expectation_from_masks`] and the
/// distributed backend all reduce through this, and leaving the decision to LTO
/// ties it to how many callers the function happens to have.
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

/// Complex Pauli sandwiches `⟨λ|P_i|φ⟩` for every mask triple in one traversal
/// of the pair.
///
/// Same value as [`pauli_sandwich`] per entry, to within the association of the
/// sum. A mask with `xmask == 0` reads `λ` at the loop index rather than at a
/// partner index, so the two families are accumulated separately as in
/// [`pauli_expectations_from_masks`].
pub(crate) fn pauli_sandwiches_from_masks(
    lambda: &[Complex64],
    phi: &[Complex64],
    masks: &[(usize, usize, u32)],
) -> Vec<Complex64> {
    if masks.len() < 2 {
        return masks
            .iter()
            .map(|&(xmask, zmask, num_y)| pauli_sandwich(lambda, phi, xmask, zmask, num_y))
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

    let accumulate = |z_acc: &mut [Complex64], g_acc: &mut [Complex64], base: usize, len: usize| {
        for j in base..base + len {
            let amp = phi[j];
            let aligned = lambda[j].conj() * amp;
            for (slot, &zmask) in z_acc.iter_mut().zip(z_only.iter()) {
                *slot += if (j & zmask).count_ones() & 1 == 1 {
                    -aligned
                } else {
                    aligned
                };
            }
            for (slot, &(xmask, zmask)) in g_acc.iter_mut().zip(general.iter()) {
                let partner = lambda[j ^ xmask];
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
            vec![Complex64::new(0.0, 0.0); z_only.len()],
            vec![Complex64::new(0.0, 0.0); general.len()],
        )
    };
    let (mut z_sum, mut g_sum) = zeros();

    #[cfg(feature = "parallel")]
    if phi.len() >= (1 << SANDWICH_MIN_PAR_QUBITS) {
        use rayon::prelude::*;
        let chunk = crate::backend::MIN_PAR_ELEMS;
        let (z, g) = phi
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
        return finish_sandwiches(masks, &z, &g);
    }

    accumulate(&mut z_sum, &mut g_sum, 0, phi.len());
    finish_sandwiches(masks, &z_sum, &g_sum)
}

/// Interleave the two sandwich accumulator families back into mask order, the
/// [`finish_expectations`] split applied to the unnormalized complex values.
fn finish_sandwiches(
    masks: &[(usize, usize, u32)],
    z_sum: &[Complex64],
    g_sum: &[Complex64],
) -> Vec<Complex64> {
    let (mut zi, mut gi) = (0, 0);
    masks
        .iter()
        .map(|&(xmask, _, num_y)| {
            let raw = if xmask == 0 {
                zi += 1;
                z_sum[zi - 1]
            } else {
                gi += 1;
                g_sum[gi - 1]
            };
            raw * i_pow(num_y)
        })
        .collect()
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
