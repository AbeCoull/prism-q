//! Belief propagation with ordered-statistics post-processing (BP+OSD) over
//! detector error models, hypergraph mechanisms included.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use super::DetectorErrorModel;
use super::decoder::{ShotDecoder, decode_batch, logical_error_rate};
use super::dem::symptom_label;
use crate::error::{PrismError, Result};
use crate::sim::compiled::PackedShots;

/// Largest exhaustive OSD order: `2^order` candidate patterns per shot.
const MAX_EXHAUSTIVE_ORDER: usize = 24;
/// Bound on `|tanh|` products before `atanh`, keeping product-sum messages finite.
const TANH_CLAMP: f64 = 1.0 - 1e-15;

/// Check-node update rule for belief propagation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BpMethod {
    /// Exact sum-product update in the log-likelihood domain.
    ProductSum,
    /// Min-sum update with every check-to-variable message multiplied by
    /// `scaling` in `(0, 1]`.
    MinSum { scaling: f64 },
}

/// Ordered-statistics post-processing applied when belief propagation does not
/// reach a solution of the syndrome.
///
/// Columns are ranked by posterior error likelihood; Gaussian elimination in
/// that order picks the most reliable information set, and the non-pivot
/// ("free") columns are the search space. Every candidate fixes the free bits
/// and solves the pivots, and the candidate of least total weight
/// `sum ln((1-p)/p)` wins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsdMethod {
    /// The single candidate with every free bit zero.
    Zero,
    /// OSD-0 plus every single free bit and every pair among the first
    /// `order` free bits.
    CombinationSweep { order: usize },
    /// Every pattern over the first `order` free bits, at most 24. With
    /// `order` at least the free-column count this is exact maximum-likelihood
    /// error decoding.
    Exhaustive { order: usize },
}

/// Configuration for [`BpOsdDecoder`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BpOsdOptions {
    /// Flooding iterations before falling back to OSD; zero runs OSD on the
    /// priors alone.
    pub max_iterations: usize,
    pub bp_method: BpMethod,
    pub osd_method: OsdMethod,
}

impl Default for BpOsdOptions {
    /// 30 iterations of min-sum at scaling 0.625, then OSD-CS of order 7.
    fn default() -> Self {
        Self {
            max_iterations: 30,
            bp_method: BpMethod::MinSum { scaling: 0.625 },
            osd_method: OsdMethod::CombinationSweep { order: 7 },
        }
    }
}

/// BP+OSD decoder compiled from a detector error model.
///
/// Each mechanism flipping at least one detector becomes a column of the
/// check matrix (detectors are rows) with prior log-likelihood ratio
/// `ln((1-p)/p)`; mechanisms sharing one detector set collapse to the most
/// probable of them, and mechanisms flipping no detector cannot be inferred.
/// No graphlike restriction applies, so color-code and qLDPC models decode
/// directly. Belief propagation runs on the Tanner graph and stops as soon as
/// its hard decision reproduces the syndrome; otherwise ordered-statistics
/// decoding ([`OsdMethod`]) returns the least-weight solution among its
/// candidates. Decoding uses no randomness.
#[derive(Debug, Clone)]
pub struct BpOsdDecoder {
    num_detectors: usize,
    num_observables: usize,
    obs_words: usize,
    num_columns: usize,
    rank: usize,
    check_offsets: Vec<u32>,
    slot_column: Vec<u32>,
    column_offsets: Vec<u32>,
    column_slots: Vec<u32>,
    slot_check: Vec<u32>,
    prior: Vec<f64>,
    column_obs: Vec<u64>,
    nonnegative_priors: bool,
    max_check_degree: usize,
    options: BpOsdOptions,
}

impl BpOsdDecoder {
    /// Compile a decoder with [`BpOsdOptions::default`].
    ///
    /// # Errors
    ///
    /// As [`Self::with_options`].
    pub fn from_model(model: &DetectorErrorModel) -> Result<Self> {
        Self::with_options(model, BpOsdOptions::default())
    }

    /// Compile a decoder from any detector error model.
    ///
    /// # Errors
    ///
    /// Mechanism probabilities must lie in `[0, 1)`; a zero-probability
    /// mechanism is skipped. A min-sum scaling outside `(0, 1]` or an
    /// exhaustive order above 24 is rejected.
    pub fn with_options(model: &DetectorErrorModel, options: BpOsdOptions) -> Result<Self> {
        if let BpMethod::MinSum { scaling } = options.bp_method
            && !(scaling > 0.0 && scaling <= 1.0)
        {
            return Err(PrismError::InvalidParameter {
                message: format!("min-sum scaling {scaling} lies outside (0, 1]"),
            });
        }
        if let OsdMethod::Exhaustive { order } = options.osd_method
            && order > MAX_EXHAUSTIVE_ORDER
        {
            return Err(PrismError::InvalidParameter {
                message: format!(
                    "exhaustive OSD order {order} exceeds {MAX_EXHAUSTIVE_ORDER}; use the \
                     combination sweep for larger searches"
                ),
            });
        }

        let mut columns: Vec<(&[usize], f64, &[usize])> = Vec::new();
        let mut slots: HashMap<&[usize], usize> = HashMap::new();
        for mechanism in model.mechanisms() {
            let p = mechanism.probability();
            if !(0.0..1.0).contains(&p) {
                return Err(PrismError::InvalidParameter {
                    message: format!(
                        "mechanism `{}` has probability {p}, outside [0, 1)",
                        symptom_label(mechanism)
                    ),
                });
            }
            if p == 0.0 || mechanism.detectors().is_empty() {
                continue;
            }
            match slots.entry(mechanism.detectors()) {
                Entry::Occupied(slot) => {
                    let column = &mut columns[*slot.get()];
                    if p > column.1 {
                        column.1 = p;
                        column.2 = mechanism.observables();
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(columns.len());
                    columns.push((mechanism.detectors(), p, mechanism.observables()));
                }
            }
        }
        Ok(Self::from_columns(
            model.num_detectors(),
            model.num_observables(),
            &columns,
            options,
        ))
    }

    /// Build from `(detectors, probability, observables)` columns with
    /// ascending, in-range detector indices and probabilities in `(0, 1)`.
    fn from_columns(
        num_detectors: usize,
        num_observables: usize,
        columns: &[(&[usize], f64, &[usize])],
        options: BpOsdOptions,
    ) -> Self {
        let obs_words = num_observables.div_ceil(64);
        let num_columns = columns.len();
        let mut check_offsets = vec![0u32; num_detectors + 1];
        for (detectors, ..) in columns {
            for &d in *detectors {
                check_offsets[d + 1] += 1;
            }
        }
        for d in 0..num_detectors {
            check_offsets[d + 1] += check_offsets[d];
        }
        let nnz = check_offsets[num_detectors] as usize;
        let mut cursor = check_offsets.clone();
        let mut slot_column = vec![0u32; nnz];
        let mut slot_check = vec![0u32; nnz];
        let mut column_offsets = vec![0u32; num_columns + 1];
        let mut column_slots = Vec::with_capacity(nnz);
        let mut prior = Vec::with_capacity(num_columns);
        let mut column_obs = vec![0u64; num_columns * obs_words];
        for (column, (detectors, p, observables)) in columns.iter().enumerate() {
            for &d in *detectors {
                let slot = cursor[d];
                cursor[d] += 1;
                slot_column[slot as usize] = column as u32;
                slot_check[slot as usize] = d as u32;
                column_slots.push(slot);
            }
            column_offsets[column + 1] = column_slots.len() as u32;
            prior.push(((1.0 - p) / p).ln());
            for &o in *observables {
                column_obs[column * obs_words + o / 64] |= 1u64 << (o % 64);
            }
        }
        let max_check_degree = (0..num_detectors)
            .map(|d| (check_offsets[d + 1] - check_offsets[d]) as usize)
            .max()
            .unwrap_or(0);
        let nonnegative_priors = prior.iter().all(|&llr| llr >= 0.0);

        let mut decoder = Self {
            num_detectors,
            num_observables,
            obs_words,
            num_columns,
            rank: 0,
            check_offsets,
            slot_column,
            column_offsets,
            column_slots,
            slot_check,
            prior,
            column_obs,
            nonnegative_priors,
            max_check_degree,
            options,
        };
        let mut scratch = decoder.scratch();
        for (position, column) in scratch.order.iter_mut().enumerate() {
            *column = position as u32;
        }
        scratch.use_all_rows();
        decoder.rank = decoder.eliminate(&mut scratch, num_columns, num_detectors);
        decoder
    }

    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    pub fn options(&self) -> BpOsdOptions {
        self.options
    }

    /// Decode packed detector samples into predicted observable flips.
    ///
    /// Layouts and bit order as [`super::UnionFindDecoder::decode_packed`].
    ///
    /// # Errors
    ///
    /// The input measurement count must equal the model's detector count, and
    /// a syndrome outside the column space of the check matrix rejects the
    /// batch, naming the first such shot.
    pub fn decode_packed(&self, detectors: &PackedShots) -> Result<PackedShots> {
        decode_batch(self, detectors)
    }

    /// Decode `detectors` and return the fraction of shots whose predicted
    /// flips differ from `observables` in any observable; `0.0` for no shots.
    ///
    /// # Errors
    ///
    /// As [`Self::decode_packed`], and `observables` must hold one bit per
    /// observable for the same shot count.
    pub fn logical_error_rate(
        &self,
        detectors: &PackedShots,
        observables: &PackedShots,
    ) -> Result<f64> {
        logical_error_rate(&self.decode_packed(detectors)?, observables)
    }

    /// Decode one shot into `s.solution` (one byte per column) and XOR its
    /// observable flips into `out_row`. Returns the solution's prior weight.
    fn solve_shot(
        &self,
        row: &[u64],
        out_row: &mut [u64],
        s: &mut BpOsdScratch,
    ) -> std::result::Result<f64, Unexplained> {
        let mut any = false;
        for (d, bit) in s.syndrome.iter_mut().enumerate() {
            *bit = (row[d / 64] >> (d % 64)) as u8 & 1;
            any |= *bit != 0;
        }
        s.solution.fill(0);
        if !any && self.nonnegative_priors {
            return Ok(0.0);
        }

        if self.belief_propagation(s) {
            self.refine_support(s);
        } else {
            for (position, column) in s.order.iter_mut().enumerate() {
                *column = position as u32;
            }
            let posterior = &s.posterior;
            s.order.sort_unstable_by(|&a, &b| {
                posterior[a as usize]
                    .total_cmp(&posterior[b as usize])
                    .then(a.cmp(&b))
            });
            s.use_all_rows();
            self.ordered_statistics(s, self.num_columns, self.rank)?;
        }
        let mut weight = 0.0;
        for column in 0..self.num_columns {
            if s.solution[column] != 0 {
                weight += self.prior[column];
                let base = column * self.obs_words;
                for (word, mask) in out_row
                    .iter_mut()
                    .zip(&self.column_obs[base..base + self.obs_words])
                {
                    *word ^= mask;
                }
            }
        }
        Ok(weight)
    }

    /// Flooding belief propagation. Returns whether the hard decision, left in
    /// `s.solution`, satisfies the syndrome; `s.posterior` holds the final
    /// log-likelihood ratios either way.
    fn belief_propagation(&self, s: &mut BpOsdScratch) -> bool {
        s.posterior.copy_from_slice(&self.prior);
        for (column, &llr) in self.prior.iter().enumerate() {
            s.solution[column] = u8::from(llr < 0.0);
        }
        if self.satisfies_syndrome(s) {
            return true;
        }
        for (slot, &column) in self.slot_column.iter().enumerate() {
            s.v2c[slot] = self.prior[column as usize];
        }
        for _ in 0..self.options.max_iterations {
            for check in 0..self.num_detectors {
                let begin = self.check_offsets[check] as usize;
                let end = self.check_offsets[check + 1] as usize;
                let flip = s.syndrome[check] != 0;
                match self.options.bp_method {
                    BpMethod::MinSum { scaling } => {
                        let mut negative = flip;
                        let mut min1 = f64::INFINITY;
                        let mut min2 = f64::INFINITY;
                        let mut arg = begin;
                        for slot in begin..end {
                            let q = s.v2c[slot];
                            negative ^= q < 0.0;
                            let magnitude = q.abs();
                            if magnitude < min1 {
                                min2 = min1;
                                min1 = magnitude;
                                arg = slot;
                            } else if magnitude < min2 {
                                min2 = magnitude;
                            }
                        }
                        for slot in begin..end {
                            let magnitude = if slot == arg { min2 } else { min1 };
                            let sign_negative = negative ^ (s.v2c[slot] < 0.0);
                            s.c2v[slot] = if sign_negative {
                                -scaling * magnitude
                            } else {
                                scaling * magnitude
                            };
                        }
                    }
                    BpMethod::ProductSum => {
                        let degree = end - begin;
                        let mut forward = 1.0;
                        for at in 0..degree {
                            s.partial[at] = forward;
                            forward *= (0.5 * s.v2c[begin + at]).tanh();
                        }
                        let mut backward = 1.0;
                        for at in (0..degree).rev() {
                            let product = (s.partial[at] * backward).clamp(-TANH_CLAMP, TANH_CLAMP);
                            let message = 2.0 * product.atanh();
                            s.c2v[begin + at] = if flip { -message } else { message };
                            backward *= (0.5 * s.v2c[begin + at]).tanh();
                        }
                    }
                }
            }
            for column in 0..self.num_columns {
                let begin = self.column_offsets[column] as usize;
                let end = self.column_offsets[column + 1] as usize;
                let mut total = self.prior[column];
                for &slot in &self.column_slots[begin..end] {
                    total += s.c2v[slot as usize];
                }
                s.posterior[column] = total;
                s.solution[column] = u8::from(total < 0.0);
                for &slot in &self.column_slots[begin..end] {
                    s.v2c[slot as usize] = total - s.c2v[slot as usize];
                }
            }
            if self.satisfies_syndrome(s) {
                return true;
            }
        }
        false
    }

    fn satisfies_syndrome(&self, s: &BpOsdScratch) -> bool {
        (0..self.num_detectors).all(|check| {
            let begin = self.check_offsets[check] as usize;
            let end = self.check_offsets[check + 1] as usize;
            let parity = self.slot_column[begin..end]
                .iter()
                .fold(0u8, |acc, &column| acc ^ s.solution[column as usize]);
            parity == s.syndrome[check]
        })
    }

    /// A converged BP decision satisfies the syndrome but can carry a
    /// zero-syndrome cycle (a stabilizer or a logical) on top of a lighter
    /// solution. Rerun the OSD search restricted to the decision's support and
    /// the checks it touches, which keeps the cost proportional to its size.
    fn refine_support(&self, s: &mut BpOsdScratch) {
        let mut count = 0;
        for column in 0..self.num_columns {
            if s.solution[column] != 0 {
                s.order[count] = column as u32;
                count += 1;
            }
        }
        if count < 2 {
            return;
        }
        let posterior = &s.posterior;
        s.order[..count].sort_unstable_by(|&a, &b| {
            posterior[a as usize]
                .total_cmp(&posterior[b as usize])
                .then(a.cmp(&b))
        });
        s.row_stamp += 1;
        s.rows.clear();
        for &column in &s.order[..count] {
            let begin = self.column_offsets[column as usize] as usize;
            let end = self.column_offsets[column as usize + 1] as usize;
            for &slot in &self.column_slots[begin..end] {
                let check = self.slot_check[slot as usize] as usize;
                if s.row_mark[check] != s.row_stamp {
                    s.row_mark[check] = s.row_stamp;
                    s.row_local[check] = s.rows.len() as u32;
                    s.rows.push(check as u32);
                }
            }
        }
        let support_weight = |s: &BpOsdScratch| -> f64 {
            s.order[..count]
                .iter()
                .filter(|&&column| s.solution[column as usize] != 0)
                .map(|&column| self.prior[column as usize])
                .sum()
        };
        let decision_weight = support_weight(s);
        let max_rank = count.min(s.rows.len());
        let refined = self.ordered_statistics(s, count, max_rank);
        debug_assert!(refined.is_ok(), "the BP decision explains the syndrome");
        if support_weight(s) > decision_weight {
            for &column in &s.order[..count] {
                s.solution[column as usize] = 1;
            }
        }
    }

    /// Reduce the submatrix on rows `s.rows` and columns `s.order[..count]`,
    /// with the syndrome as right-hand side, to reduced row-echelon form over
    /// GF(2), stopping after `max_rank` pivots. Returns the rank; pivot
    /// positions land in `s.pivot_position` and the reduced rows in `s.matrix`
    /// at a stride of `s.stride` words.
    fn eliminate(&self, s: &mut BpOsdScratch, count: usize, max_rank: usize) -> usize {
        let words = count.div_ceil(64).max(1);
        s.stride = words;
        let rows = s.rows.len();
        s.matrix[..rows * words].fill(0);
        for (position, &column) in s.order[..count].iter().enumerate() {
            let begin = self.column_offsets[column as usize] as usize;
            let end = self.column_offsets[column as usize + 1] as usize;
            for &slot in &self.column_slots[begin..end] {
                let local = s.row_local[self.slot_check[slot as usize] as usize] as usize;
                s.matrix[local * words + position / 64] |= 1u64 << (position % 64);
            }
        }
        for (local, &check) in s.rows.iter().enumerate() {
            s.rhs[local] = s.syndrome[check as usize];
        }

        let mut rank = 0;
        for position in 0..count {
            if rank == max_rank || rank == rows {
                break;
            }
            let word = position / 64;
            let bit = 1u64 << (position % 64);
            let Some(pivot) = (rank..rows).find(|&r| s.matrix[r * words + word] & bit != 0) else {
                continue;
            };
            if pivot != rank {
                for w in 0..words {
                    s.matrix.swap(pivot * words + w, rank * words + w);
                }
                s.rhs.swap(pivot, rank);
            }
            let (head, tail) = s.matrix.split_at_mut(rank * words);
            let (pivot_row, tail) = tail.split_at_mut(words);
            for (r, target) in head.chunks_exact_mut(words).enumerate() {
                if target[word] & bit != 0 {
                    for (t, p) in target.iter_mut().zip(pivot_row.iter()) {
                        *t ^= p;
                    }
                    s.rhs[r] ^= s.rhs[rank];
                }
            }
            for (offset, target) in tail[..(rows - rank - 1) * words]
                .chunks_exact_mut(words)
                .enumerate()
            {
                if target[word] & bit != 0 {
                    for (t, p) in target.iter_mut().zip(pivot_row.iter()) {
                        *t ^= p;
                    }
                    s.rhs[rank + 1 + offset] ^= s.rhs[rank];
                }
            }
            s.pivot_position[rank] = position as u32;
            rank += 1;
        }
        rank
    }

    /// Search the OSD candidates over columns `s.order[..count]` (sorted most
    /// likely error first) and rows `s.rows`, and write the least-weight
    /// solution over those columns into `s.solution`.
    fn ordered_statistics(
        &self,
        s: &mut BpOsdScratch,
        count: usize,
        max_rank: usize,
    ) -> std::result::Result<(), Unexplained> {
        let rank = self.eliminate(s, count, max_rank);
        if s.rhs[rank..s.rows.len()].iter().any(|&bit| bit != 0) {
            return Err(Unexplained);
        }

        let words = s.stride;
        let rank_words = rank.div_ceil(64);
        s.is_pivot[..count].fill(false);
        for &position in &s.pivot_position[..rank] {
            s.is_pivot[position as usize] = true;
        }
        s.free.clear();
        for position in 0..count {
            if !s.is_pivot[position] {
                s.free.push(position as u32);
            }
        }
        let candidates = match self.options.osd_method {
            OsdMethod::Zero => 0,
            OsdMethod::CombinationSweep { .. } => s.free.len(),
            OsdMethod::Exhaustive { order } => order.min(s.free.len()),
        };

        // Reduced pivot-row column of each candidate free position, packed over
        // pivot rows, and the right-hand side packed the same way.
        s.base[..rank_words].fill(0);
        for r in 0..rank {
            s.base[r / 64] |= u64::from(s.rhs[r]) << (r % 64);
        }
        s.columns[..candidates * rank_words].fill(0);
        for (candidate, &position) in s.free[..candidates].iter().enumerate() {
            let word = position as usize / 64;
            let shift = position as usize % 64;
            for r in 0..rank {
                let bit = (s.matrix[r * words + word] >> shift) & 1;
                s.columns[candidate * rank_words + r / 64] |= bit << (r % 64);
            }
        }

        let column_weight =
            |s: &BpOsdScratch, position: u32| self.prior[s.order[position as usize] as usize];
        let pivot_weight = |s: &BpOsdScratch, x: &[u64]| {
            let mut total = 0.0;
            for (w, &bits) in x.iter().enumerate() {
                let mut bits = bits;
                while bits != 0 {
                    let r = w * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    total += self.prior[s.order[s.pivot_position[r] as usize] as usize];
                }
            }
            total
        };

        s.trial[..rank_words].copy_from_slice(&s.base[..rank_words]);
        let mut best_weight = pivot_weight(s, &s.trial[..rank_words]);
        let mut best = Flips::None;
        match self.options.osd_method {
            OsdMethod::Zero => {}
            OsdMethod::CombinationSweep { order } => {
                for f in 0..candidates {
                    for w in 0..rank_words {
                        s.trial[w] = s.base[w] ^ s.columns[f * rank_words + w];
                    }
                    let weight =
                        column_weight(s, s.free[f]) + pivot_weight(s, &s.trial[..rank_words]);
                    if weight < best_weight {
                        best_weight = weight;
                        best = Flips::One(f);
                    }
                }
                let pairs = order.min(candidates);
                for f1 in 0..pairs {
                    for f2 in f1 + 1..pairs {
                        for w in 0..rank_words {
                            s.trial[w] = s.base[w]
                                ^ s.columns[f1 * rank_words + w]
                                ^ s.columns[f2 * rank_words + w];
                        }
                        let weight = column_weight(s, s.free[f1])
                            + column_weight(s, s.free[f2])
                            + pivot_weight(s, &s.trial[..rank_words]);
                        if weight < best_weight {
                            best_weight = weight;
                            best = Flips::Two(f1, f2);
                        }
                    }
                }
            }
            OsdMethod::Exhaustive { .. } => {
                // Gray-code walk: each step toggles one free column.
                let mut pattern = 0u32;
                for step in 1..1u32 << candidates {
                    let f = step.trailing_zeros() as usize;
                    pattern ^= 1 << f;
                    for w in 0..rank_words {
                        s.trial[w] ^= s.columns[f * rank_words + w];
                    }
                    let mut free_weight = 0.0;
                    let mut bits = pattern;
                    while bits != 0 {
                        free_weight += column_weight(s, s.free[bits.trailing_zeros() as usize]);
                        bits &= bits - 1;
                    }
                    let weight = free_weight + pivot_weight(s, &s.trial[..rank_words]);
                    if weight < best_weight {
                        best_weight = weight;
                        best = Flips::Pattern(pattern);
                    }
                }
            }
        }

        for &column in &s.order[..count] {
            s.solution[column as usize] = 0;
        }
        s.trial[..rank_words].copy_from_slice(&s.base[..rank_words]);
        let flip = |s: &mut BpOsdScratch, f: usize| {
            s.solution[s.order[s.free[f] as usize] as usize] = 1;
            for w in 0..rank_words {
                s.trial[w] ^= s.columns[f * rank_words + w];
            }
        };
        match best {
            Flips::None => {}
            Flips::One(f) => flip(s, f),
            Flips::Two(f1, f2) => {
                flip(s, f1);
                flip(s, f2);
            }
            Flips::Pattern(pattern) => {
                for f in 0..candidates {
                    if pattern >> f & 1 == 1 {
                        flip(s, f);
                    }
                }
            }
        }
        for r in 0..rank {
            let column = s.order[s.pivot_position[r] as usize] as usize;
            s.solution[column] = (s.trial[r / 64] >> (r % 64)) as u8 & 1;
        }
        Ok(())
    }
}

/// Free-column flips of the best OSD candidate.
#[derive(Debug, Clone, Copy)]
enum Flips {
    None,
    One(usize),
    Two(usize, usize),
    Pattern(u32),
}

/// A syndrome outside the check matrix's column space.
pub(super) struct Unexplained;

impl ShotDecoder for BpOsdDecoder {
    type Scratch = BpOsdScratch;
    type Failure = Unexplained;
    #[cfg(feature = "parallel")]
    const PARALLEL_SHOT_THRESHOLD: usize = 64;
    #[cfg(feature = "parallel")]
    const SHOT_CHUNK: usize = 16;

    fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    fn num_observables(&self) -> usize {
        self.num_observables
    }

    fn scratch(&self) -> BpOsdScratch {
        BpOsdScratch::new(self)
    }

    fn decode_shot(
        &self,
        row: &[u64],
        out_row: &mut [u64],
        scratch: &mut BpOsdScratch,
    ) -> std::result::Result<(), Unexplained> {
        self.solve_shot(row, out_row, scratch).map(|_| ())
    }

    fn failure_error(_: Unexplained, shot: usize) -> PrismError {
        PrismError::InvalidParameter {
            message: format!(
                "shot {shot}: the syndrome lies outside the column space of the model's \
                 check matrix, so it is impossible under the model"
            ),
        }
    }
}

/// Reusable per-shot BP and OSD buffers, sized once from the decoder.
pub(super) struct BpOsdScratch {
    syndrome: Vec<u8>,
    v2c: Vec<f64>,
    c2v: Vec<f64>,
    partial: Vec<f64>,
    posterior: Vec<f64>,
    solution: Vec<u8>,
    order: Vec<u32>,
    rows: Vec<u32>,
    row_local: Vec<u32>,
    row_mark: Vec<u64>,
    row_stamp: u64,
    stride: usize,
    matrix: Vec<u64>,
    rhs: Vec<u8>,
    pivot_position: Vec<u32>,
    is_pivot: Vec<bool>,
    free: Vec<u32>,
    base: Vec<u64>,
    trial: Vec<u64>,
    columns: Vec<u64>,
}

impl BpOsdScratch {
    fn use_all_rows(&mut self) {
        self.rows.clear();
        for (check, local) in self.row_local.iter_mut().enumerate() {
            *local = check as u32;
            self.rows.push(check as u32);
        }
    }

    fn new(decoder: &BpOsdDecoder) -> Self {
        let rows = decoder.num_detectors;
        let columns = decoder.num_columns;
        let nnz = decoder.slot_column.len();
        let row_words = columns.div_ceil(64);
        let rank_words = rows.min(columns).div_ceil(64);
        Self {
            syndrome: vec![0; rows],
            v2c: vec![0.0; nnz],
            c2v: vec![0.0; nnz],
            partial: vec![0.0; decoder.max_check_degree],
            posterior: vec![0.0; columns],
            solution: vec![0; columns],
            order: vec![0; columns],
            rows: Vec::with_capacity(rows),
            row_local: vec![0; rows],
            row_mark: vec![0; rows],
            row_stamp: 0,
            stride: row_words,
            matrix: vec![0; rows * row_words.max(1)],
            rhs: vec![0; rows],
            pivot_position: vec![0; rows.min(columns)],
            is_pivot: vec![false; columns],
            free: Vec::with_capacity(columns),
            base: vec![0; rank_words],
            trial: vec![0; rank_words],
            columns: vec![0; columns * rank_words],
        }
    }
}

#[cfg(test)]
#[path = "bposd_tests.rs"]
mod tests;
