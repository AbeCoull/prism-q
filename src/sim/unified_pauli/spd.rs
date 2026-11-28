//! Deterministic sparse Pauli dynamics (SPD): the full weighted Pauli sum carried
//! backward, with optional truncation and a light-cone variant.

use num_complex::Complex64;

use std::borrow::Cow;
use std::hash::{Hash, Hasher};

use super::{
    PauliAxis, PauliTerm, SPD_MAX_TERMS_CEILING, check_spd_term_ceiling,
    clifford_conjugation_phase, count_branching_gates, is_spd_native, pauli_vec_from_terms,
    snap_trig_pair, snapped_sin_cos, spd_term_ceiling_error, validate_and_lower, z_rotation_angle,
    zz_rotation_angle,
};
use crate::circuit::{Circuit, Instruction, SmallVec, smallvec};
use crate::error::{PrismError, Result};
use crate::gates::Gate;
use crate::hash::{FxBuildHasher, FxHashMap};
use crate::sim::compiled::{PauliVec, flip_bit, get_bit, propagate_backward_words};

/// Result of running SPD on a joint Pauli observable.
#[derive(Debug, Clone)]
pub struct SpdObservableResult {
    /// `⟨P⟩`; exact when nothing was truncated.
    pub mean: f64,
    pub t_count: usize,
    /// Peak size of the weighted Pauli sum during propagation.
    pub peak_terms: usize,
    /// Total coefficient magnitude discarded by truncation (0 for exact runs).
    pub total_discarded: f64,
}

/// Pauli string as `x` and `z` words in the [`PauliVec`] letter convention: the term key
/// of a [`WeightedPauliSum`].
pub(super) trait SpdKey: Clone + Eq + Hash {
    fn from_words(x: &[u64], z: &[u64]) -> Self;
    fn x(&self) -> &[u64];
    fn z(&self) -> &[u64];
    fn words_mut(&mut self) -> (&mut [u64], &mut [u64]);
}

/// Pauli string on at most `64 * W` qubits, held inline so a term owns no heap block.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct InlinePauli<const W: usize> {
    x: [u64; W],
    z: [u64; W],
}

impl<const W: usize> Hash for InlinePauli<W> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for w in 0..W {
            state.write_u64(self.x[w]);
            state.write_u64(self.z[w]);
        }
    }
}

impl<const W: usize> SpdKey for InlinePauli<W> {
    #[inline(always)]
    fn from_words(x: &[u64], z: &[u64]) -> Self {
        let mut key = Self {
            x: [0; W],
            z: [0; W],
        };
        key.x[..x.len()].copy_from_slice(x);
        key.z[..z.len()].copy_from_slice(z);
        key
    }

    #[inline(always)]
    fn x(&self) -> &[u64] {
        &self.x
    }

    #[inline(always)]
    fn z(&self) -> &[u64] {
        &self.z
    }

    #[inline(always)]
    fn words_mut(&mut self) -> (&mut [u64], &mut [u64]) {
        (&mut self.x, &mut self.z)
    }
}

impl SpdKey for PauliVec {
    fn from_words(x: &[u64], z: &[u64]) -> Self {
        Self {
            x: x.to_vec(),
            z: z.to_vec(),
        }
    }

    #[inline(always)]
    fn x(&self) -> &[u64] {
        &self.x
    }

    #[inline(always)]
    fn z(&self) -> &[u64] {
        &self.z
    }

    #[inline(always)]
    fn words_mut(&mut self) -> (&mut [u64], &mut [u64]) {
        (&mut self.x, &mut self.z)
    }
}

/// Bind `$key` to the narrowest [`SpdKey`] for `$num_qubits` and evaluate `$body`, so the
/// key width is chosen once per run. Past 128 qubits a term falls back to heap words.
macro_rules! with_spd_key {
    ($num_qubits:expr, $key:ident => $body:expr) => {
        match usize::div_ceil($num_qubits, 64) {
            0 | 1 => {
                type $key = InlinePauli<1>;
                $body
            }
            2 => {
                type $key = InlinePauli<2>;
                $body
            }
            _ => {
                type $key = PauliVec;
                $body
            }
        }
    };
}
pub(super) use with_spd_key;

/// Weighted Pauli sum held as a list of distinct Pauli strings.
///
/// Clifford conjugation is a bijection on Pauli strings, so a Clifford step
/// rewrites each term in place and never merges two. A rotation's flip
/// multiplies by its axis, which preserves anticommutation with that axis, so
/// the flip of an anticommuting term can land only on another anticommuting
/// term; those are the only terms a rotation hashes.
pub(super) struct WeightedPauliSum<K> {
    pub(super) terms: Vec<(K, Complex64)>,
    pub(super) in_support: Vec<usize>,
    pub(super) partner: Vec<usize>,
    pub(super) probe: K,
    /// Pauli string to term position, cleared per rotation but kept across them so the
    /// table is allocated once per run.
    pub(super) index: FxHashMap<K, usize>,
}

impl<K: SpdKey> WeightedPauliSum<K> {
    pub(super) fn new() -> Self {
        Self {
            terms: Vec::new(),
            in_support: Vec::new(),
            partner: Vec::new(),
            probe: K::from_words(&[], &[]),
            index: FxHashMap::default(),
        }
    }

    /// Append a term whose Pauli string is not already in the sum.
    pub(super) fn insert(&mut self, pauli: K, coeff: Complex64) {
        self.terms.push((pauli, coeff));
    }

    fn conjugate_all_backward_phased(&mut self, gate: &Gate, targets: &[usize]) {
        for (pauli, coeff) in self.terms.iter_mut() {
            let phase = clifford_conjugation_phase(gate, targets, pauli.x(), pauli.z());
            let (x, z) = pauli.words_mut();
            propagate_backward_words(x, z, gate, targets);
            *coeff *= phase;
        }
    }

    /// Split every anticommuting term `P` into `cos P` and `-i sin` times its
    /// flip. A term whose flip is already present pairs with it and both update
    /// in place; an unpaired term appends its flip, or becomes it when `cos` is
    /// zero.
    #[inline(always)]
    fn branch(
        &mut self,
        sin: f64,
        cos: f64,
        anticommutes: impl Fn(&K) -> bool,
        flip: impl Fn(&mut K),
    ) {
        self.branch_with(sin, cos, anticommutes, flip, |_, c| {
            Complex64::new(c.im * sin, -c.re * sin)
        });
    }

    /// [`branch`](Self::branch) with the flipped coefficient taken from
    /// `flipped_coeff(source, c)`, for an axis whose phase depends on the
    /// source term's letter. `sin == 0` still means no flip.
    #[inline(always)]
    fn branch_with(
        &mut self,
        sin: f64,
        cos: f64,
        anticommutes: impl Fn(&K) -> bool,
        flip: impl Fn(&mut K),
        flipped_coeff: impl Fn(&K, Complex64) -> Complex64,
    ) {
        let (sin, cos) = snap_trig_pair(sin, cos);
        if sin == 0.0 {
            if cos != 1.0 {
                for (pauli, coeff) in &mut self.terms {
                    if anticommutes(pauli) {
                        *coeff *= cos;
                    }
                }
            }
            return;
        }
        self.in_support.clear();
        self.in_support.extend(
            self.terms
                .iter()
                .enumerate()
                .filter(|(_, (pauli, _))| anticommutes(pauli))
                .map(|(i, _)| i),
        );

        self.partner.clear();
        if self.in_support.len() > 1 {
            self.index.clear();
            self.index.reserve(self.in_support.len());
            for &i in &self.in_support {
                self.index.insert(self.terms[i].0.clone(), i);
            }
            for &i in &self.in_support {
                self.probe.clone_from(&self.terms[i].0);
                flip(&mut self.probe);
                self.partner
                    .push(self.index.get(&self.probe).copied().unwrap_or(usize::MAX));
            }
        } else {
            self.partner.resize(self.in_support.len(), usize::MAX);
        }

        for k in 0..self.in_support.len() {
            let i = self.in_support[k];
            let j = self.partner[k];
            if j != usize::MAX {
                if i < j {
                    let (ci, cj) = (self.terms[i].1, self.terms[j].1);
                    let from_j = flipped_coeff(&self.terms[j].0, cj);
                    let from_i = flipped_coeff(&self.terms[i].0, ci);
                    self.terms[i].1 = ci * cos + from_j;
                    self.terms[j].1 = cj * cos + from_i;
                }
            } else if cos == 0.0 {
                let (pauli, coeff) = &mut self.terms[i];
                *coeff = flipped_coeff(pauli, *coeff);
                flip(pauli);
            } else {
                let mut pauli = self.terms[i].0.clone();
                flip(&mut pauli);
                let coeff = self.terms[i].1;
                let flipped = flipped_coeff(&self.terms[i].0, coeff);
                self.terms[i].1 = coeff * cos;
                self.terms.push((pauli, flipped));
            }
        }
    }

    /// Merge terms that share a Pauli string.
    fn coalesce(&mut self) {
        let mut merged: FxHashMap<K, Complex64> =
            FxHashMap::with_capacity_and_hasher(self.terms.len(), FxBuildHasher);
        for (pauli, coeff) in self.terms.drain(..) {
            *merged.entry(pauli).or_insert(Complex64::new(0.0, 0.0)) += coeff;
        }
        self.terms.extend(merged);
    }

    /// Split every in-support term under backward conjugation by a Z rotation,
    /// which sends `P` to `cos(theta) P + (-i sin(theta)) P Z_q`. Takes the
    /// evaluated `(sin, cos)` because `run_spd` replays the circuit once per
    /// qubit and the trigonometry is per gate, not per replay. A branch whose
    /// coefficient is exactly zero is not inserted, so `Rz(0)` and `P(0)` leave
    /// the term count alone.
    fn branch_z_rotation(&mut self, qubit: usize, sin: f64, cos: f64) {
        self.branch(
            sin,
            cos,
            |pauli| get_bit(pauli.x(), qubit),
            |pauli| flip_bit(pauli.words_mut().1, qubit),
        );
    }

    /// Two-qubit analogue of [`branch_z_rotation`](Self::branch_z_rotation)
    /// for `Rzz`: a term splits only when it anticommutes with `Z⊗Z` on the
    /// pair, i.e. exactly one of the two qubits carries an X or Y letter. The
    /// flip branch right-multiplies by `Z_a Z_b`, a pure z-bit flip on both
    /// qubits in the ordered `X^x Z^z` letter convention.
    fn branch_zz_rotation(&mut self, a: usize, b: usize, sin: f64, cos: f64) {
        self.branch(
            sin,
            cos,
            |pauli| get_bit(pauli.x(), a) != get_bit(pauli.x(), b),
            |pauli| {
                let z = pauli.words_mut().1;
                flip_bit(z, a);
                flip_bit(z, b);
            },
        );
    }

    /// X-axis analogue of [`branch_z_rotation`](Self::branch_z_rotation): a
    /// term anticommutes with `X_q` when its z bit is set, and `P X_q` is minus
    /// the term with its x bit toggled, so the flip carries `+i sin`.
    fn branch_x_rotation(&mut self, qubit: usize, sin: f64, cos: f64) {
        self.branch(
            -sin,
            cos,
            |pauli| get_bit(pauli.z(), qubit),
            |pauli| flip_bit(pauli.words_mut().0, qubit),
        );
    }

    /// Y-axis analogue of [`branch_z_rotation`](Self::branch_z_rotation): a
    /// term anticommutes with `Y_q` when its x and z bits differ, and the flip
    /// toggles both.
    fn branch_y_rotation(&mut self, qubit: usize, sin: f64, cos: f64) {
        self.branch_with(
            sin,
            cos,
            |pauli| get_bit(pauli.x(), qubit) != get_bit(pauli.z(), qubit),
            |pauli| {
                let (x, z) = pauli.words_mut();
                flip_bit(x, qubit);
                flip_bit(z, qubit);
            },
            // With Y = iXZ: X·Y = iZ and Z·Y = -iX, so -i sin times the phase is
            // real, +sin from an X source and -sin from a Z source.
            |source, c| {
                c * if get_bit(source.z(), qubit) {
                    -sin
                } else {
                    sin
                }
            },
        );
    }

    /// Pauli-string analogue of [`branch_z_rotation`](Self::branch_z_rotation) for
    /// `exp(-i θ P / 2)` with `(px, pz)` the words of `P` and `num_y` its Y count: a term
    /// `Q` splits when it anticommutes with `P`, and its flip is the key of `Q P`. With
    /// `P = i^num_y X^px Z^pz`, `X^qx Z^qz X^px Z^pz = (-1)^|qz & px| X^(qx^px) Z^(qz^pz)`,
    /// so the flip carries `-i^(num_y + 1) sin` and the sign of the source's z bits
    /// passing the string's x bits.
    fn branch_pauli_rotation(&mut self, px: &[u64], pz: &[u64], num_y: u32, sin: f64, cos: f64) {
        let flip_weight = match num_y & 3 {
            0 => Complex64::new(0.0, -sin),
            1 => Complex64::new(sin, 0.0),
            2 => Complex64::new(0.0, sin),
            _ => Complex64::new(-sin, 0.0),
        };
        self.branch_with(
            sin,
            cos,
            |pauli| {
                let overlap = (pauli.x().iter().zip(pauli.z()))
                    .zip(px.iter().zip(pz))
                    .fold(0, |acc, ((qx, qz), (px, pz))| acc ^ (qx & pz) ^ (qz & px));
                overlap.count_ones() & 1 == 1
            },
            |pauli| {
                let (x, z) = pauli.words_mut();
                for (w, p) in x.iter_mut().zip(px) {
                    *w ^= p;
                }
                for (w, p) in z.iter_mut().zip(pz) {
                    *w ^= p;
                }
            },
            |source, c| {
                let passed = (source.z().iter().zip(px)).fold(0, |acc, (qz, px)| acc ^ (qz & px));
                if passed.count_ones() & 1 == 1 {
                    -c * flip_weight
                } else {
                    c * flip_weight
                }
            },
        );
    }

    /// Backward-apply one gate: a single discriminant match, so the Clifford
    /// path pays no rotation-probe chain per instruction. Angle cases mirror
    /// [`z_rotation_angle`] and [`zz_rotation_angle`].
    #[inline(always)]
    pub(super) fn apply_backward(&mut self, gate: &Gate, targets: &[usize]) {
        match gate {
            Gate::T => self.branch_z_rotation_angle(targets[0], std::f64::consts::FRAC_PI_4),
            Gate::Tdg => self.branch_z_rotation_angle(targets[0], -std::f64::consts::FRAC_PI_4),
            Gate::Rz(theta) | Gate::P(theta) => self.branch_z_rotation_angle(targets[0], *theta),
            Gate::Rzz(theta) => {
                let (sin, cos) = snapped_sin_cos(*theta);
                self.branch_zz_rotation(targets[0], targets[1], sin, cos);
            }
            Gate::Rx(theta) => {
                let (sin, cos) = snapped_sin_cos(*theta);
                self.branch_x_rotation(targets[0], sin, cos);
            }
            Gate::Ry(theta) => {
                let (sin, cos) = snapped_sin_cos(*theta);
                self.branch_y_rotation(targets[0], sin, cos);
            }
            Gate::PauliRot(data) => {
                let Some((first, _)) = self.terms.first() else {
                    return;
                };
                let (px, pz, num_y) = pauli_rotation_words(first.x().len(), targets, data.axes());
                let (sin, cos) = snapped_sin_cos(data.theta());
                self.branch_pauli_rotation(&px, &pz, num_y, sin, cos);
            }
            _ => self.conjugate_all_backward_phased(gate, targets),
        }
    }

    #[inline(always)]
    fn branch_z_rotation_angle(&mut self, qubit: usize, theta: f64) {
        let (sin, cos) = snapped_sin_cos(theta);
        self.branch_z_rotation(qubit, sin, cos);
    }

    pub(super) fn truncate(&mut self, epsilon: f64) -> f64 {
        let threshold = epsilon.max(0.0);
        let threshold_sqr = threshold * threshold;
        let mut discarded = 0.0;
        self.terms.retain(|(_, coeff)| {
            if coeff.norm_sqr() < threshold_sqr {
                discarded += coeff.norm();
                false
            } else {
                true
            }
        });
        discarded
    }

    /// Drop exactly the smallest-magnitude surplus terms until at most
    /// `max_terms` remain, returning the 1-norm of what was dropped. Ties at
    /// the cut fall to term order, so which of two equal-magnitude terms
    /// survives is not stable across runs; the returned bound holds either
    /// way.
    fn truncate_to_budget(&mut self, max_terms: usize) -> f64 {
        let surplus = self.terms.len().saturating_sub(max_terms);
        if surplus == 0 {
            return 0.0;
        }
        self.terms.select_nth_unstable_by(surplus - 1, |a, b| {
            a.1.norm_sqr().total_cmp(&b.1.norm_sqr())
        });
        let mut discarded = 0.0;
        for (_, coeff) in self.terms.drain(..surplus) {
            discarded += coeff.norm();
        }
        discarded
    }

    pub(super) fn diagonal_expectation(&self) -> f64 {
        let mut sum = Complex64::new(0.0, 0.0);
        for (pauli, coeff) in &self.terms {
            if pauli.x().iter().all(|&w| w == 0) {
                sum += coeff;
            }
        }
        sum.re
    }
}

/// Words of the Pauli string `axes[i]` on `targets[i]` over `num_words` words, and its Y
/// count.
fn pauli_rotation_words(
    num_words: usize,
    targets: &[usize],
    axes: &[PauliAxis],
) -> (SmallVec<[u64; 2]>, SmallVec<[u64; 2]>, u32) {
    let mut x: SmallVec<[u64; 2]> = smallvec![0; num_words];
    let mut z: SmallVec<[u64; 2]> = smallvec![0; num_words];
    let mut num_y = 0;
    for (&q, axis) in targets.iter().zip(axes) {
        match axis {
            PauliAxis::X => flip_bit(&mut x, q),
            PauliAxis::Z => flip_bit(&mut z, q),
            PauliAxis::Y => {
                flip_bit(&mut x, q);
                flip_bit(&mut z, q);
                num_y += 1;
            }
        }
    }
    (x, z, num_y)
}

/// `U† P U` as weighted one-word Pauli strings in the letter convention of the Clifford
/// register, where `X`, `Y` and `Z` letters are Hermitian: `P` is the string `(x, z)` and `U`
/// the product of `exp(-i θ i^phase4 R)` over `rotations` given as `(R.x, R.z, phase4, θ)`,
/// the first applied first.
pub(in crate::sim) fn propagate_register_rotations(
    x: u64,
    z: u64,
    rotations: impl DoubleEndedIterator<Item = (u64, u64, u8, f64)>,
) -> Vec<(u64, u64, Complex64)> {
    const I_POW: [Complex64; 4] = [
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, -1.0),
    ];
    let mut sum = WeightedPauliSum::<InlinePauli<1>>::new();
    sum.insert(InlinePauli { x: [x], z: [z] }, Complex64::new(1.0, 0.0));
    for (rx, rz, phase4, theta) in rotations.rev() {
        // One rotation `E` sends `P` anticommuting with `R` to `cos 2θ P + i sin 2θ R P`.
        let (sin, cos) = snapped_sin_cos(2.0 * theta);
        sum.branch_with(
            sin,
            cos,
            |p| ((p.x[0] & rz) ^ (p.z[0] & rx)).count_ones() & 1 == 1,
            |p| {
                p.x[0] ^= rx;
                p.z[0] ^= rz;
            },
            |source, c| {
                let quarter = 1
                    + u32::from(phase4)
                    + crate::sim::clifford_register::letter_phase(rx, rz, source.x[0], source.z[0]);
                c * sin * I_POW[(quarter & 3) as usize]
            },
        );
    }
    sum.terms
        .into_iter()
        .map(|(p, coeff)| (p.x[0], p.z[0], coeff))
        .collect()
}

/// Per-qubit `⟨Z_q⟩` values from a deterministic sparse Pauli dynamics run.
pub struct SpdResult {
    /// `⟨Z_q⟩` per qubit; exact when nothing was truncated.
    pub expectations: Vec<f64>,
    /// Branching gates in the circuit: `T`, `Tdg`, `Rz`, `P`, `Rzz`, `Rx`, `Ry`
    /// and `PauliRot`.
    pub t_count: usize,
    /// Peak size of the weighted Pauli sum across all per-qubit runs, not the
    /// truncation budget passed in.
    pub max_terms: usize,
    /// Total coefficient magnitude discarded by truncation (0 for exact runs).
    pub total_discarded: f64,
}

/// Truncation policy for deterministic sparse Pauli dynamics.
///
/// Both arms report the discarded coefficient magnitude on the result, and
/// that sum bounds the error by `|error| <= sum |discarded|`. Neither is a
/// fidelity bound.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpdTruncation {
    /// Drop terms below `epsilon` whenever the weighted sum exceeds
    /// `max_terms`; `max_terms == 0` disables mid-run pruning. An `epsilon`
    /// too small to hold the growth reaches the internal term ceiling and
    /// errors there rather than returning an over-truncated value.
    Threshold { epsilon: f64, max_terms: usize },
    /// Hold the sum at `max_terms` by dropping the smallest-magnitude surplus
    /// terms, exactly enough to return to the budget. No threshold to guess,
    /// and growth the budget already caps cannot reach the ceiling.
    Budget { max_terms: usize },
}

impl SpdTruncation {
    /// Whether the policy can drop a term. A threshold at or below zero cannot,
    /// since no coefficient magnitude falls under it.
    pub(crate) fn can_discard(&self) -> bool {
        match *self {
            SpdTruncation::Threshold { epsilon, .. } => epsilon > 0.0,
            SpdTruncation::Budget { .. } => true,
        }
    }

    /// Reject a budget that would drop every term or outrun the internal
    /// ceiling. Called by every entry point, so a policy built by hand cannot
    /// reach the propagation loop unchecked.
    pub(crate) fn validate(&self) -> Result<()> {
        if let SpdTruncation::Budget { max_terms } = *self
            && (max_terms == 0 || max_terms > SPD_MAX_TERMS_CEILING)
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "budgeted SPD needs a term budget between 1 and {SPD_MAX_TERMS_CEILING}"
                ),
            });
        }
        Ok(())
    }

    /// Mid-run enforcement after one instruction. Discarded mass accumulates
    /// into `total_discarded` only when a drop fires, so the exact path keeps
    /// no unconditional float add in the per-instruction loop.
    pub(super) fn enforce<K: SpdKey>(
        &self,
        sum: &mut WeightedPauliSum<K>,
        total_discarded: &mut f64,
    ) {
        match *self {
            SpdTruncation::Threshold { epsilon, max_terms } => {
                if max_terms > 0 && sum.terms.len() > max_terms {
                    *total_discarded += sum.truncate(epsilon);
                }
            }
            SpdTruncation::Budget { max_terms } => {
                if sum.terms.len() > max_terms {
                    *total_discarded += sum.truncate_to_budget(max_terms);
                }
            }
        }
    }

    /// Terminal sweep after the backward pass.
    pub(super) fn final_sweep<K: SpdKey>(
        &self,
        sum: &mut WeightedPauliSum<K>,
        total_discarded: &mut f64,
    ) {
        if let SpdTruncation::Threshold { epsilon, .. } = *self
            && epsilon > 0.0
        {
            *total_discarded += sum.truncate(epsilon);
        }
    }
}

/// Deterministic SPD on a joint Pauli observable.
///
/// Starts with the single weighted term `(observable, 1.0)`, backward-
/// propagates through every gate, splits each term anticommuting with the
/// Pauli string of an `Rz`, `Rzz`, `Rx`, `Ry` or `PauliRot` into `cos(theta)`
/// and `sin(theta)` branches, and truncates terms whose
/// magnitude falls below `epsilon` whenever the sum exceeds `max_terms`.
/// Returns `⟨0^n| U† P U |0^n⟩` as the sum of remaining diagonal-term
/// coefficients.
///
/// # Truncation
///
/// Every rotation with a non-Clifford angle branches, so a variational circuit
/// branches on each of its rotations rather than only on its `T` gates, and the
/// term count is bounded by `2^(in-support rotations)`. With `max_terms == 0`
/// the run is exact and errors once the sum passes an internal ceiling. With
/// `max_terms > 0`, sub-`epsilon` terms are dropped whenever the sum exceeds
/// the budget and `total_discarded` bounds the resulting error, per the
/// composable 1-norm bound `|error| ≤ Σ |discarded|` (each dropped term's
/// exact continuation is unitary conjugation of a unit-norm operator, so its
/// terminal contribution is at most its magnitude). An `epsilon` too small to
/// prune the growth still reaches the ceiling and errors there: the run never
/// returns a silently over-truncated value. See [`run_spd_observable_budgeted`]
/// for the policy with no threshold to guess.
pub fn run_spd_observable(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    run_spd_observable_with(
        circuit,
        observable,
        &SpdTruncation::Threshold { epsilon, max_terms },
    )
}

/// Deterministic SPD on a joint Pauli observable under a fixed term budget.
///
/// Same backward propagation as [`run_spd_observable`], but whenever the sum
/// exceeds `max_terms` the smallest-magnitude surplus terms are dropped,
/// exactly enough to return to the budget. No threshold needs guessing, the
/// run cannot die at the term ceiling on account of growth the budget already
/// caps, and `total_discarded` is the realized 1-norm error bound for the
/// term count held. `max_terms` must be at least 1.
pub fn run_spd_observable_budgeted(
    circuit: &Circuit,
    observable: &[PauliTerm],
    max_terms: usize,
) -> Result<SpdObservableResult> {
    run_spd_observable_with(circuit, observable, &SpdTruncation::Budget { max_terms })
}

pub(crate) fn run_spd_observable_with(
    circuit: &Circuit,
    observable: &[PauliTerm],
    truncation: &SpdTruncation,
) -> Result<SpdObservableResult> {
    run_spd_weighted_observable_with(circuit, &[(1.0, observable)], truncation)
}

/// Seed a sum with `Σ weight · observable`, merging terms that share a Pauli
/// string.
pub(super) fn weighted_sum<K: SpdKey>(
    num_qubits: usize,
    terms: &[(f64, &[PauliTerm])],
) -> Result<WeightedPauliSum<K>> {
    let mut sum = WeightedPauliSum::new();
    for &(weight, observable) in terms {
        let (pauli, phase) = pauli_vec_from_terms(num_qubits, observable)?;
        sum.insert(K::from_words(&pauli.x, &pauli.z), phase * weight);
    }
    if sum.terms.len() > 1 {
        sum.coalesce();
    }
    Ok(sum)
}

/// `⟨Σ weight · P⟩` by one backward propagation of the whole weighted sum.
///
/// Truncation applies to the merged sum: a term budget caps the whole sum,
/// `epsilon` compares weighted coefficients, and `total_discarded` bounds the
/// error on the weighted value. A merged sum that outgrows the term ceiling
/// falls back to one propagation per term, so an exact run fails only where a
/// single term would.
pub(crate) fn run_spd_weighted_observable_with(
    circuit: &Circuit,
    terms: &[(f64, &[PauliTerm])],
    truncation: &SpdTruncation,
) -> Result<SpdObservableResult> {
    truncation.validate()?;
    let lowered = validate_and_lower(circuit, "SPD observable", is_spd_native)?;
    let circuit = lowered.as_ref();
    let n = circuit.num_qubits;
    let t_count = count_branching_gates(circuit);

    let merged = |terms: &[(f64, &[PauliTerm])]| -> Result<Option<SpdObservableResult>> {
        with_spd_key!(n, K => Ok(propagate_spd(circuit, weighted_sum::<K>(n, terms)?, truncation)))
    };
    if let Some(result) = merged(terms)? {
        return Ok(SpdObservableResult { t_count, ..result });
    }
    if terms.len() < 2 {
        return Err(spd_term_ceiling_error("SPD"));
    }
    let mut folded = SpdObservableResult {
        mean: 0.0,
        t_count,
        peak_terms: 0,
        total_discarded: 0.0,
    };
    for &(weight, observable) in terms {
        let result = merged(&[(1.0, observable)])?.ok_or_else(|| spd_term_ceiling_error("SPD"))?;
        folded.mean += weight * result.mean;
        folded.peak_terms = folded.peak_terms.max(result.peak_terms);
        folded.total_discarded += weight.abs() * result.total_discarded;
    }
    Ok(folded)
}

/// Backward-propagate `sum` through a lowered circuit, or `None` once it
/// outgrows the term ceiling. The caller fills `t_count`.
fn propagate_spd<K: SpdKey>(
    circuit: &Circuit,
    mut sum: WeightedPauliSum<K>,
    truncation: &SpdTruncation,
) -> Option<SpdObservableResult> {
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for inst in circuit.instructions.iter().rev() {
        if let Instruction::Gate { gate, targets } = inst {
            sum.apply_backward(gate, targets);
        }
        truncation.enforce(&mut sum, &mut total_discarded);
        if sum.terms.len() > SPD_MAX_TERMS_CEILING {
            return None;
        }
        if sum.terms.len() > peak_terms {
            peak_terms = sum.terms.len();
        }
    }

    truncation.final_sweep(&mut sum, &mut total_discarded);

    Some(SpdObservableResult {
        mean: sum.diagonal_expectation(),
        t_count: 0,
        peak_terms,
        total_discarded,
    })
}

/// Compute `⟨Z_q⟩` for every qubit of a unitary circuit of Clifford gates and
/// Pauli rotations, via deterministic sparse Pauli dynamics.
///
/// Each qubit's `Z_q` propagates backward as a weighted Pauli sum; every
/// rotation with a non-Clifford angle doubles the in-support terms. When the
/// sum exceeds `max_terms`, terms with coefficient magnitude below `epsilon`
/// are dropped. `max_terms == 0` disables truncation: exact, but the sum must
/// stay under a hard term ceiling. See [`run_spd_observable`] for a single
/// joint observable and for the truncation contract.
pub fn run_spd(circuit: &Circuit, epsilon: f64, max_terms: usize) -> Result<SpdResult> {
    run_spd_with(circuit, &SpdTruncation::Threshold { epsilon, max_terms })
}

/// [`run_spd`] under any truncation policy, including the budget one that has
/// no threshold to guess.
pub fn run_spd_with(circuit: &Circuit, truncation: &SpdTruncation) -> Result<SpdResult> {
    truncation.validate()?;
    Ok(run_spd_bounded(circuit, truncation, None)?
        .0
        .expect("an unbounded run always completes"))
}

/// Exact [`run_spd`] that gives up with `Ok(None)` once the term count summed
/// over every instruction of every pass exceeds `work_budget`, or the sum
/// outgrows the term ceiling, so the caller can take another exact route.
///
/// The exact sum never loses a term: conjugation permutes the keys, a rotation
/// keeps each key or swaps it for its flip, and a cancelled coefficient stays
/// in the list at zero. So the current count times the instructions left in the
/// pass, plus one term per instruction of every pass not yet started, is a
/// floor on the work still to come, and the run stops once that floor passes
/// the budget. It returns `None` for exactly the circuits that would overrun
/// the budget if run to the end.
pub(crate) fn run_spd_exact_within(
    circuit: &Circuit,
    work_budget: usize,
) -> Result<Option<SpdResult>> {
    Ok(run_spd_bounded(circuit, &EXACT, Some(work_budget))?.0)
}

pub(super) const EXACT: SpdTruncation = SpdTruncation::Threshold {
    epsilon: 0.0,
    max_terms: 0,
};

/// A `work_budget` requires [`EXACT`] truncation, under which the term count
/// cannot shrink. Also returns the term steps run, counted only under a budget.
pub(super) fn run_spd_bounded(
    circuit: &Circuit,
    truncation: &SpdTruncation,
    work_budget: Option<usize>,
) -> Result<(Option<SpdResult>, usize)> {
    let lowered = validate_and_lower(circuit, "SPD", is_spd_native)?;
    let circuit = lowered.as_ref();
    with_spd_key!(circuit.num_qubits, K => run_spd_lowered::<K>(circuit, truncation, work_budget))
}

enum PrecomputedRot {
    Single {
        sin: f64,
        cos: f64,
    },
    Pair {
        sin: f64,
        cos: f64,
    },
    X {
        sin: f64,
        cos: f64,
    },
    Y {
        sin: f64,
        cos: f64,
    },
    String {
        x: SmallVec<[u64; 2]>,
        z: SmallVec<[u64; 2]>,
        num_y: u32,
        sin: f64,
        cos: f64,
    },
}

fn precompute_rotations(circuit: &Circuit) -> (Vec<Option<PrecomputedRot>>, usize) {
    let num_words = circuit.num_qubits.div_ceil(64).max(1);
    let mut t_count = 0usize;
    let rotations = circuit
        .instructions
        .iter()
        .map(|inst| {
            let Instruction::Gate { gate, targets } = inst else {
                return None;
            };
            let rot = if let Some(theta) = z_rotation_angle(gate) {
                let (sin, cos) = snapped_sin_cos(theta);
                Some(PrecomputedRot::Single { sin, cos })
            } else if let Some(theta) = zz_rotation_angle(gate) {
                let (sin, cos) = snapped_sin_cos(theta);
                Some(PrecomputedRot::Pair { sin, cos })
            } else {
                match gate {
                    Gate::Rx(theta) => {
                        let (sin, cos) = snapped_sin_cos(*theta);
                        Some(PrecomputedRot::X { sin, cos })
                    }
                    Gate::Ry(theta) => {
                        let (sin, cos) = snapped_sin_cos(*theta);
                        Some(PrecomputedRot::Y { sin, cos })
                    }
                    Gate::PauliRot(data) => {
                        let (x, z, num_y) = pauli_rotation_words(num_words, targets, data.axes());
                        let (sin, cos) = snapped_sin_cos(data.theta());
                        Some(PrecomputedRot::String {
                            x,
                            z,
                            num_y,
                            sin,
                            cos,
                        })
                    }
                    _ => None,
                }
            };
            if rot.is_some() {
                t_count += 1;
            }
            rot
        })
        .collect();
    (rotations, t_count)
}

#[inline(always)]
fn apply_precomputed_backward<K: SpdKey>(
    sum: &mut WeightedPauliSum<K>,
    rot: Option<&PrecomputedRot>,
    gate: &Gate,
    targets: &[usize],
) {
    match rot {
        Some(PrecomputedRot::Single { sin, cos }) => {
            sum.branch_z_rotation(targets[0], *sin, *cos);
        }
        Some(PrecomputedRot::Pair { sin, cos }) => {
            sum.branch_zz_rotation(targets[0], targets[1], *sin, *cos);
        }
        Some(PrecomputedRot::X { sin, cos }) => {
            sum.branch_x_rotation(targets[0], *sin, *cos);
        }
        Some(PrecomputedRot::Y { sin, cos }) => {
            sum.branch_y_rotation(targets[0], *sin, *cos);
        }
        Some(PrecomputedRot::String {
            x,
            z,
            num_y,
            sin,
            cos,
        }) => {
            sum.branch_pauli_rotation(x, z, *num_y, *sin, *cos);
        }
        None => sum.conjugate_all_backward_phased(gate, targets),
    }
}

/// Circuit lowered once with precomputed rotation trig for repeated light-cone SPD queries.
pub(crate) struct PreparedSpd<'c> {
    circuit: Cow<'c, Circuit>,
    t_count: usize,
    rotations: Vec<Option<PrecomputedRot>>,
}

impl<'c> PreparedSpd<'c> {
    pub(crate) fn new(circuit: &'c Circuit, backend: &'static str) -> Result<Self> {
        let lowered = validate_and_lower(circuit, backend, is_spd_native)?;
        let (rotations, t_count) = precompute_rotations(lowered.as_ref());
        Ok(Self {
            circuit: lowered,
            t_count,
            rotations,
        })
    }

    pub(crate) fn eval_light_cone(
        &self,
        observable: &[PauliTerm],
        epsilon: f64,
        max_terms: usize,
    ) -> Result<SpdObservableResult> {
        let circuit = self.circuit.as_ref();
        let n = circuit.num_qubits;
        let (obs, obs_coeff) = pauli_vec_from_terms(n, observable)?;
        let keep = inverse_light_cone(circuit, observable);
        with_spd_key!(n, K => propagate_light_cone(
            circuit,
            &self.rotations,
            &keep,
            K::from_words(&obs.x, &obs.z),
            obs_coeff,
            epsilon,
            max_terms,
            self.t_count,
        ))
    }
}

fn run_spd_lowered<K: SpdKey>(
    circuit: &Circuit,
    truncation: &SpdTruncation,
    work_budget: Option<usize>,
) -> Result<(Option<SpdResult>, usize)> {
    let n = circuit.num_qubits;
    let num_words = n.div_ceil(64);
    let (rotations, t_count) = precompute_rotations(circuit);

    let zeros = vec![0u64; num_words];
    let identity = K::from_words(&zeros, &zeros);
    let mut expectations = Vec::with_capacity(n);
    let mut peak_terms = 0usize;
    let mut total_discarded = 0.0;
    let mut work = 0usize;

    for q in 0..n {
        let later_pass_floor = (n - 1 - q).saturating_mul(circuit.instructions.len());
        let mut sum = WeightedPauliSum::new();
        let mut z_q = identity.clone();
        flip_bit(z_q.words_mut().1, q);
        sum.insert(z_q, Complex64::new(1.0, 0.0));

        for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
            if let Instruction::Gate { gate, targets } = inst {
                apply_precomputed_backward(&mut sum, rotations[idx].as_ref(), gate, targets);
            }

            truncation.enforce(&mut sum, &mut total_discarded);
            if let Some(budget) = work_budget {
                let terms = sum.terms.len();
                work += terms;
                let floor = work
                    .saturating_add(terms.saturating_mul(idx))
                    .saturating_add(later_pass_floor);
                if floor > budget || terms > SPD_MAX_TERMS_CEILING {
                    return Ok((None, work));
                }
            } else {
                check_spd_term_ceiling(sum.terms.len(), "SPD")?;
            }

            if sum.terms.len() > peak_terms {
                peak_terms = sum.terms.len();
            }
        }

        truncation.final_sweep(&mut sum, &mut total_discarded);

        expectations.push(sum.diagonal_expectation());
    }

    Ok((
        Some(SpdResult {
            expectations,
            t_count,
            max_terms: peak_terms,
            total_discarded,
        }),
        work,
    ))
}

/// Inverse light cone of a Pauli observable under a circuit, computed
/// conservatively by gate-graph reachability.
///
/// One flag per entry of `circuit.instructions`: whether its support meets the
/// cone-qubit set as the circuit is walked in reverse from the observable.
///
/// Dropping the unflagged gates is exact: the support of `P_k` lies inside the
/// cone set at depth `k`, and `U_k` acts as identity outside its targets, so
/// `U_k^dag P_k U_k = P_k`.
pub fn inverse_light_cone(circuit: &Circuit, observable: &[PauliTerm]) -> Vec<bool> {
    let n_inst = circuit.instructions.len();
    let mut keep = vec![false; n_inst];
    if observable.is_empty() {
        return keep;
    }
    let mut cone = vec![0u64; circuit.num_qubits.div_ceil(64)];
    for term in observable {
        if let Some(word) = cone.get_mut(term.qubit / 64) {
            *word |= 1u64 << (term.qubit % 64);
        }
    }

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        let targets: &[usize] = match inst {
            Instruction::Gate { targets, .. } => targets,
            _ => continue,
        };
        let touches = targets
            .iter()
            .any(|&q| cone.get(q / 64).is_some_and(|&w| (w >> (q % 64)) & 1 != 0));
        if touches {
            keep[idx] = true;
            for &q in targets {
                if let Some(word) = cone.get_mut(q / 64) {
                    *word |= 1u64 << (q % 64);
                }
            }
        }
    }
    keep
}

/// SPD on a joint Pauli observable, restricted to the inverse light cone.
///
/// Identical in result to `run_spd_observable`, but skips gates whose support
/// is disjoint from the propagated observable's causal cone. For QEC syndrome
/// and detector observables with bounded spatial support, this turns the SPD
/// cliff from a function of the total branching-gate count into a function of
/// the in-cone count.
pub fn run_spd_observable_light_cone(
    circuit: &Circuit,
    observable: &[PauliTerm],
    epsilon: f64,
    max_terms: usize,
) -> Result<SpdObservableResult> {
    PreparedSpd::new(circuit, "light-cone SPD observable")?
        .eval_light_cone(observable, epsilon, max_terms)
}

#[allow(clippy::too_many_arguments)]
fn propagate_light_cone<K: SpdKey>(
    circuit: &Circuit,
    rotations: &[Option<PrecomputedRot>],
    keep: &[bool],
    observable: K,
    coeff: Complex64,
    epsilon: f64,
    max_terms: usize,
    t_count: usize,
) -> Result<SpdObservableResult> {
    let mut sum = WeightedPauliSum::new();
    sum.insert(observable, coeff);
    let mut peak_terms = sum.terms.len();
    let mut total_discarded = 0.0;

    for (idx, inst) in circuit.instructions.iter().enumerate().rev() {
        if !keep[idx] {
            continue;
        }
        if let Instruction::Gate { gate, targets } = inst {
            apply_precomputed_backward(&mut sum, rotations[idx].as_ref(), gate, targets);
        }
        if max_terms > 0 && sum.terms.len() > max_terms {
            total_discarded += sum.truncate(epsilon);
        }
        check_spd_term_ceiling(sum.terms.len(), "SPD")?;
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
