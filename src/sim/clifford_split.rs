//! Terminal shots of a unitary Clifford+T circuit from its register in the product gauge,
//! split into parts that each draw from their own table, with a part too wide for one
//! table cut in two.

use num_complex::Complex64;
use rand::Rng;
use std::f64::consts::FRAC_1_SQRT_2;

use super::parts::{ProductGauge, compress};
use super::{
    MeasuredRows, RegisterPauli, bit, diagonalize, find, fold_table, groups, join, measured_rows,
};
use crate::backend::simd::walsh_hadamard;
use crate::circuit::Circuit;
use crate::error::Result;
use crate::sim::shots::{draws_split_across_workers, sample_in_shot_blocks};

/// Admission limits of the split register. Each one decides which circuits the register
/// takes and how their shots are drawn, so changing one moves seeded shots.
#[derive(Clone, Copy, Debug)]
pub(super) struct SplitLimits {
    /// Most index bits in one pack of small tables.
    pack_bits: usize,
    /// Widest part held whole before a cut is tried.
    whole_width: usize,
    /// Widest part held whole at all: one table of `2^width` entries.
    table_width: usize,
    side_width: usize,
    /// Most row pairs across a cut taken ahead of a whole part up to `table_width`.
    cheap_pairs: usize,
    /// Most row pairs across any cut. A shot pays about `(E + 3) 2^E` amplitude units for
    /// a cut with `E` pairs.
    max_pairs: usize,
    max_anchors: usize,
    /// `log2` of the most amplitudes summed over whole parts and cut sides.
    total_log2: usize,
}

#[cfg(not(prism_lowered_thresholds))]
pub(super) const SPLIT_LIMITS: SplitLimits = SplitLimits {
    pack_bits: 12,
    whole_width: 20,
    table_width: 24,
    side_width: 20,
    cheap_pairs: 7,
    max_pairs: 12,
    max_anchors: 64,
    total_log2: 25,
};
#[cfg(prism_lowered_thresholds)]
pub(super) const SPLIT_LIMITS: SplitLimits = SplitLimits {
    pack_bits: 2,
    whole_width: 2,
    table_width: 3,
    side_width: 2,
    cheap_pairs: 1,
    max_pairs: 2,
    max_anchors: 64,
    total_log2: 4,
};

/// Letters of a Pauli on register bits, `x` then `z`, with no sign.
type Letters = (u64, u64);

/// Classes of register bits under "an excess rotation links every bit of its mask", by
/// lowest bit.
pub(super) fn units(width: usize, excess: &[(RegisterPauli, f64)]) -> Vec<u64> {
    let mut parent: Vec<usize> = (0..width).collect();
    for (p, _) in excess {
        join(&mut parent, p.x | p.z);
    }
    let mut units = groups(&mut parent);
    units.sort_unstable_by_key(|u| u.trailing_zeros());
    units
}

/// A basis of the letter strings of `rows`, as `x | z << 64`.
pub(super) fn row_basis(rows: impl Iterator<Item = RegisterPauli>) -> Vec<u128> {
    let mut basis: Vec<u128> = Vec::new();
    for row in rows {
        let mut v = u128::from(row.x) | u128::from(row.z) << 64;
        for b in &basis {
            v = v.min(v ^ b);
        }
        if v != 0 {
            basis.push(v);
        }
    }
    basis
}

/// Whether two letter strings, as `x | z << 64`, anticommute on the register bits of `mask`.
fn anticommute_on(a: u128, b: u128, mask: u64) -> bool {
    let (ax, az, bx, bz) = (a as u64, (a >> 64) as u64, b as u64, (b >> 64) as u64);
    (((ax & bz) ^ (az & bx)) & mask).count_ones() & 1 == 1
}

/// The commutation form of each unit over `basis`: row `i` of a unit's form has bit `j`
/// set when basis rows `i` and `j` anticommute on the unit's bits.
pub(super) fn unit_forms(units: &[u64], basis: &[u128]) -> Vec<Vec<u64>> {
    units
        .iter()
        .map(|&u| {
            basis
                .iter()
                .map(|&a| {
                    basis.iter().enumerate().fold(0u64, |row, (j, &b)| {
                        row | u64::from(anticommute_on(a, b, u)) << j
                    })
                })
                .collect()
        })
        .collect()
}

/// Parts of the register by the canonical merge, by lowest bit: the connected components
/// of the binary matroid of the nonzero unit forms, a unit with a zero form alone. A form
/// that eliminates to zero against earlier ones joins every unit of its fundamental
/// circuit, and the fundamental circuits of one elimination span the components, so the
/// parts depend on the row space alone.
pub(super) fn canonical_parts(units: &[u64], forms: &[Vec<u64>]) -> Vec<u64> {
    let mut unit_parent: Vec<usize> = (0..units.len()).collect();
    let mut eliminated: Vec<(Vec<u64>, usize, u64)> = Vec::new();
    for (i, form) in forms.iter().enumerate() {
        let mut v = form.clone();
        let mut combo = 1u64 << i;
        for (row, pivot, sum) in &eliminated {
            if bit(&v, *pivot) {
                for (d, s) in v.iter_mut().zip(row) {
                    *d ^= s;
                }
                combo ^= sum;
            }
        }
        match v.iter().position(|&w| w != 0) {
            Some(w) => {
                let pivot = 64 * w + v[w].trailing_zeros() as usize;
                eliminated.push((v, pivot, combo));
            }
            None => join(&mut unit_parent, combo),
        }
    }
    let mut parts = vec![0u64; units.len()];
    for (i, &u) in units.iter().enumerate() {
        parts[find(&mut unit_parent, i)] |= u;
    }
    parts.retain(|&m| m != 0);
    parts.sort_unstable_by_key(|p| p.trailing_zeros());
    parts
}

/// Rank of the pieces of `basis` on the register bits of `mask`.
fn piece_rank(basis: &[u128], mask: u64) -> usize {
    let cut = u128::from(mask) | u128::from(mask) << 64;
    row_basis(basis.iter().map(|&row| {
        let v = row & cut;
        RegisterPauli {
            x: v as u64,
            z: (v >> 64) as u64,
            phase4: 0,
        }
    }))
    .len()
}

/// Rank over F2 of a matrix given by its rows.
pub(super) fn f2_rank(rows: &[u64]) -> usize {
    let mut rows: Vec<u64> = rows.iter().copied().filter(|&r| r != 0).collect();
    let mut rank = 0;
    while let Some(p) = rows.pop() {
        rank += 1;
        let low = p & p.wrapping_neg();
        for r in &mut rows {
            if *r & low != 0 {
                *r ^= p;
            }
        }
        rows.retain(|&r| r != 0);
    }
    rank
}

/// A part cut in two along units, so that no excess rotation crosses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CutChoice {
    /// Register bits of the side holding the part's lowest unit.
    pub(super) left: u64,
    /// Units on the left side, bit `i` for unit `i` of the part.
    inside: u64,
    pub(super) left_width: usize,
    pub(super) right_width: usize,
    /// Row pairs that anticommute across the cut, `E`.
    pub(super) pairs: usize,
    /// `40 (2^kL + 2^kR) + 1024 (E + 3) 2^E`: amplitude updates to build both sides and
    /// draw 1024 shots, a constant in place of the run's shot count.
    pub(super) work: u128,
}

impl CutChoice {
    fn key(&self) -> (u128, usize, usize, u64) {
        (
            self.work,
            self.left_width.max(self.right_width),
            self.pairs,
            self.inside,
        )
    }
}

/// The cut of a part into two unions of `units`, lowest by work, then the wider side,
/// then the row pairs, then the set of units on the left read as an integer. `forms[i]` is
/// the commutation form of `units[i]`. Every cut is tried up to 12 units; past that, three
/// descents by single units, from the shortest prefixes holding a quarter, a half and
/// three quarters of the width. `None` for a part of one unit.
pub(super) fn frozen_cut(units: &[u64], forms: &[&[u64]]) -> Option<CutChoice> {
    let m = units.len();
    if m < 2 {
        return None;
    }
    let widths: Vec<usize> = units.iter().map(|u| u.count_ones() as usize).collect();
    let width: usize = widths.iter().sum();
    let rows = forms[0].len();
    let all = if m == 64 { u64::MAX } else { (1u64 << m) - 1 };
    let mut acc = vec![0u64; rows];
    let mut evaluate = |inside: u64| {
        acc.fill(0);
        let (mut left, mut left_width) = (0u64, 0usize);
        let mut rest = inside;
        while rest != 0 {
            let i = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            left |= units[i];
            left_width += widths[i];
            for (a, f) in acc.iter_mut().zip(forms[i]) {
                *a ^= f;
            }
        }
        let pairs = f2_rank(&acc) / 2;
        let right_width = width - left_width;
        let work = 40 * ((1u128 << left_width) + (1u128 << right_width))
            + 1024 * (pairs as u128 + 3) * (1u128 << pairs);
        CutChoice {
            left,
            inside,
            left_width,
            right_width,
            pairs,
            work,
        }
    };
    if m <= 12 {
        return (0..(1u64 << (m - 1)) - 1)
            .map(|rest| evaluate(1 | rest << 1))
            .min_by_key(CutChoice::key);
    }
    let mut best: Option<CutChoice> = None;
    for quarters in 1..=3 {
        let mut prefix = 0usize;
        let mut covered = 0usize;
        while 4 * covered < quarters * width {
            covered += widths[prefix];
            prefix += 1;
        }
        let prefix = prefix.min(m - 1);
        let mut current = evaluate((1u64 << prefix) - 1);
        for _ in 0..8 {
            let mut moved = false;
            for i in 1..m {
                let trial = current.inside ^ 1 << i;
                if trial == all {
                    continue;
                }
                let candidate = evaluate(trial);
                if candidate.key() < current.key() {
                    current = candidate;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
        if best.is_none_or(|b| current.key() < b.key()) {
            best = Some(current);
        }
    }
    best
}

struct Part {
    mask: u64,
    /// Indices into the gauge's excess rotations, in circuit order.
    excess: Vec<usize>,
    /// Rank of the register rows cut down to the part.
    rank: usize,
    /// `None` for a part held whole.
    cut: Option<CutChoice>,
    /// Whether a whole part reads its table out by [`WalshReadout`] rather than from its
    /// dense amplitudes, whichever [`SplitPlan::work`] prices lower.
    walsh: bool,
}

/// The register of a terminal Clifford+T circuit split into parts and tiered, from the
/// tableau alone.
pub(in crate::sim) struct SplitPlan {
    gauge: ProductGauge,
    measures: Vec<(usize, usize)>,
    num_classical_bits: usize,
    rows: MeasuredRows,
    /// Each row the uniform rows leave, with its index, as register letters; a `phase4` of
    /// 2 is the sign `-1`.
    register_rows: Vec<(usize, RegisterPauli)>,
    parts: Vec<Part>,
    limits: SplitLimits,
}

impl SplitPlan {
    /// The plan of the lowered `circuit`, or `None` when the split declines it: past
    /// 64 register qubits, a unit wider than 24, a part neither held whole nor cut, or
    /// more than `2^25` amplitudes in all.
    pub(in crate::sim) fn new(circuit: &Circuit) -> Result<Option<Self>> {
        Self::with_limits(circuit, SPLIT_LIMITS)
    }

    fn with_limits(circuit: &Circuit, limits: SplitLimits) -> Result<Option<Self>> {
        let mut measures = Vec::new();
        let Some(gauge) = ProductGauge::walk(
            circuit,
            limits.max_anchors,
            limits.table_width,
            &mut measures,
        )?
        else {
            return Ok(None);
        };
        let register = &gauge.register;
        let rows = measured_rows(&register.prefix, &register.slot, &measures);
        let register_rows: Vec<(usize, RegisterPauli)> = (0..rows.rows.len())
            .filter(|&r| !rows.uniform[r])
            .map(|r| (r, register.restrict(&rows.rows[r])))
            .collect();
        let units = units(register.width, &gauge.excess);
        let basis = row_basis(register_rows.iter().map(|&(_, p)| p));
        let forms = unit_forms(&units, &basis);
        let mut parts = Vec::new();
        let mut amplitudes = 0u128;
        for mask in canonical_parts(&units, &forms) {
            let width = mask.count_ones() as usize;
            let rank = piece_rank(&basis, mask);
            let cut = if width <= limits.whole_width {
                None
            } else {
                let inside: Vec<usize> =
                    (0..units.len()).filter(|&i| units[i] & mask != 0).collect();
                let part_units: Vec<u64> = inside.iter().map(|&i| units[i]).collect();
                let part_forms: Vec<&[u64]> = inside.iter().map(|&i| &forms[i][..]).collect();
                let cut = (rank == width)
                    .then(|| frozen_cut(&part_units, &part_forms))
                    .flatten()
                    .filter(|c| c.left_width.max(c.right_width) <= limits.side_width);
                match cut {
                    Some(c) if c.pairs <= limits.cheap_pairs => Some(c),
                    _ if width <= limits.table_width => None,
                    Some(c) if c.pairs <= limits.max_pairs => Some(c),
                    _ => return Ok(None),
                }
            };
            amplitudes += match cut {
                None => 1u128 << width,
                Some(c) => (1u128 << c.left_width) + (1u128 << c.right_width),
            };
            let excess: Vec<usize> = (0..gauge.excess.len())
                .filter(|&i| {
                    let (p, _) = gauge.excess[i];
                    (p.x | p.z) & mask != 0
                })
                .collect();
            let walsh = cut.is_none()
                && rank > 0
                && walsh_readout_work(rank, excess.len())
                    < dense_readout_work(width, rank, excess.len());
            parts.push(Part {
                mask,
                excess,
                rank,
                cut,
                walsh,
            });
        }
        if amplitudes > 1u128 << limits.total_log2 {
            return Ok(None);
        }
        Ok(Some(Self {
            gauge,
            measures,
            num_classical_bits: circuit.num_classical_bits,
            rows,
            register_rows,
            parts,
            limits,
        }))
    }

    /// Register amplitude updates to build every table: the cheaper readout of each whole
    /// part, and for each side of a cut one pass for the anchors, one per excess rotation
    /// and about `3 (k + 1)` to diagonalize its rows, plus the pass that phases it.
    pub(in crate::sim) fn work(&self) -> f64 {
        let pass = |width: usize| 2f64.powi(width as i32);
        self.parts
            .iter()
            .map(|part| match part.cut {
                None if part.rank == 0 => 0.0,
                None if part.walsh => walsh_readout_work(part.rank, part.excess.len()),
                None => {
                    let width = part.mask.count_ones() as usize;
                    dense_readout_work(width, part.rank, part.excess.len())
                }
                Some(cut) => {
                    let side = |mask: u64, width: usize| {
                        let excess = part
                            .excess
                            .iter()
                            .filter(|&&i| {
                                let (p, _) = self.gauge.excess[i];
                                (p.x | p.z) & mask != 0
                            })
                            .count();
                        (excess + 2 + 3 * (width + 1)) as f64 * pass(width)
                    };
                    side(cut.left, cut.left_width) + side(part.mask & !cut.left, cut.right_width)
                }
            })
            .sum()
    }

    /// Amplitude units each shot pays past its table draws: `(E + 3) 2^E` per cut.
    pub(in crate::sim) fn units_per_shot(&self) -> u64 {
        self.parts
            .iter()
            .filter_map(|part| part.cut)
            .map(|cut| (cut.pairs as u64 + 3) << cut.pairs)
            .sum()
    }

    /// Build every table and the classical key of every drawn bit.
    pub(in crate::sim) fn sampler(self) -> SplitSampler {
        let words = self.num_classical_bits.div_ceil(64).max(1);
        let mut writer = vec![usize::MAX; self.num_classical_bits];
        for &(q, cb) in &self.measures {
            writer[cb] = self.rows.row_of_qubit[q];
        }
        let row_keys: Vec<Vec<u64>> = self
            .rows
            .inverse
            .iter()
            .map(|column| {
                let mut key = vec![0u64; words];
                for (cb, &row) in writer.iter().enumerate() {
                    if row != usize::MAX && bit(column, row) {
                        key[cb >> 6] |= 1 << (cb & 63);
                    }
                }
                key
            })
            .collect();

        let coin_rows: Vec<usize> = (0..self.rows.uniform.len())
            .filter(|&r| self.rows.uniform[r])
            .collect();
        let coin_keys: Vec<KeyTable> = coin_rows
            .chunks(64)
            .map(|chunk| {
                let keys: Vec<&[u64]> = chunk.iter().map(|&r| &row_keys[r][..]).collect();
                KeyTable::new(&keys, words)
            })
            .collect();

        let mut signs: Vec<bool> = self
            .register_rows
            .iter()
            .map(|(_, p)| p.phase4 == 2)
            .collect();
        let mut packs: Vec<Pack> = Vec::new();
        let mut draws: Vec<(u32, Draw)> = Vec::new();
        for part in &self.parts {
            let pieces: Vec<Letters> = self
                .register_rows
                .iter()
                .map(|(_, p)| (p.x & part.mask, p.z & part.mask))
                .collect();
            let elements = match part.cut {
                None => ordered_basis(&mut pieces.clone(), part.mask),
                Some(cut) => cut_elements(&mut pieces.clone(), part.mask, cut),
            };
            let element_keys = self.expand(&pieces, &elements, &row_keys, &mut signs);
            let lowest = part.mask.trailing_zeros();
            match part.cut {
                None if part.rank == 0 => {}
                None => {
                    let table = self.whole_table(part, &elements);
                    if part.rank <= self.limits.pack_bits {
                        match packs
                            .iter_mut()
                            .find(|pack| pack.bits + part.rank <= self.limits.pack_bits)
                        {
                            Some(pack) => pack.push(table, element_keys),
                            None => packs.push(Pack {
                                lowest,
                                bits: part.rank,
                                members: vec![table],
                                keys: element_keys,
                            }),
                        }
                    } else {
                        let mut cumulative = table;
                        running_sums(&mut cumulative);
                        let keys: Vec<&[u64]> = element_keys.iter().map(|k| &k[..]).collect();
                        draws.push((
                            lowest,
                            Draw::Table {
                                cumulative,
                                keys: KeyTable::new(&keys, words),
                            },
                        ));
                    }
                }
                Some(cut) => {
                    let cut_draw = self.cut_draw(part, cut, &elements, &element_keys, words);
                    draws.push((lowest, Draw::Cut(Box::new(cut_draw))));
                }
            }
        }
        for pack in packs {
            let lowest = pack.lowest;
            draws.push((lowest, pack.into_draw(words)));
        }
        draws.sort_by_key(|&(lowest, _)| lowest);

        let mut base = vec![0u64; words];
        for (&(r, _), &sign) in self.register_rows.iter().zip(&signs) {
            if sign {
                for (b, k) in base.iter_mut().zip(&row_keys[r]) {
                    *b ^= k;
                }
            }
        }
        let max_pairs = self
            .parts
            .iter()
            .filter_map(|part| part.cut)
            .map(|cut| cut.pairs)
            .max();
        SplitSampler {
            coin_keys,
            draws: draws.into_iter().map(|(_, draw)| draw).collect(),
            base,
            max_pairs,
            num_classical_bits: self.num_classical_bits,
        }
    }

    /// Expand each row's piece on a part over the part's `elements`: flip the row's sign by
    /// the expansion's, and add the row's key to each element it uses. Returns one key per
    /// element.
    fn expand(
        &self,
        pieces: &[Letters],
        elements: &[Letters],
        row_keys: &[Vec<u64>],
        signs: &mut [bool],
    ) -> Vec<Vec<u64>> {
        let words = row_keys.first().map_or(1, Vec::len);
        let mut keys = vec![vec![0u64; words]; elements.len()];
        if elements.is_empty() {
            return keys;
        }
        let expander = Expander::new(elements);
        for (i, piece) in pieces.iter().enumerate() {
            if *piece == (0, 0) {
                continue;
            }
            let (combo, negative) = expander.expand(elements, *piece);
            signs[i] ^= negative;
            let key = &row_keys[self.register_rows[i].0];
            let mut rest = combo;
            while rest != 0 {
                let e = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                for (d, s) in keys[e].iter_mut().zip(key) {
                    *d ^= s;
                }
            }
        }
        keys
    }

    /// The part's table: entry `t` is the probability that basis row `j` reads bit `j` of
    /// `t` for every `j`.
    fn whole_table(&self, part: &Part, basis: &[Letters]) -> Vec<f64> {
        if part.walsh {
            self.walsh_table(part, basis)
        } else {
            self.dense_table(part, basis)
        }
    }

    /// [`Self::whole_table`] as the Walsh transform of the expectation value of every
    /// product of basis rows, `P(t) = 2^-r Σ_s (-1)^|s & t| <Π_{j in s} B_j>`.
    fn walsh_table(&self, part: &Part, basis: &[Letters]) -> Vec<f64> {
        let readout = WalshReadout::new(&self.gauge, part.mask, &part.excess, basis);
        let mut table = vec![0.0f64; 1 << basis.len()];
        readout.fill(&mut table);
        walsh_hadamard(&mut table);
        let scale = 0.5f64.powi(basis.len() as i32);
        for p in &mut table {
            *p = (*p * scale).max(0.0);
        }
        table
    }

    /// [`Self::whole_table`] from the part's dense amplitudes, diagonalized row by row.
    fn dense_table(&self, part: &Part, basis: &[Letters]) -> Vec<f64> {
        let width = part.mask.count_ones() as usize;
        let mut amps = self.gauge.part_state(part.mask, &part.excess);
        let mut rows: Vec<RegisterPauli> = basis
            .iter()
            .map(|&(x, z)| RegisterPauli {
                x: compress(x, part.mask),
                z: compress(z, part.mask),
                phase4: 0,
            })
            .collect();
        let independent = rows.len();
        let (pivots, flips, _) = diagonalize(&mut amps, &mut rows, independent, width);
        let mut table = fold_table(&amps, &pivots, width);
        drop(amps);
        let flipped = flips
            .iter()
            .enumerate()
            .fold(0usize, |m, (j, &f)| m | usize::from(f) << j);
        if flipped != 0 {
            for t in 0..table.len() {
                if t < t ^ flipped {
                    table.swap(t, t ^ flipped);
                }
            }
        }
        table
    }

    fn cut_draw(
        &self,
        part: &Part,
        cut: CutChoice,
        elements: &[Letters],
        element_keys: &[Vec<u64>],
        words: usize,
    ) -> CutDraw {
        let pairs = cut.pairs;
        let right = part.mask & !cut.left;
        let inside_left = cut.left_width - pairs;
        let (u, rest) = elements.split_at(pairs);
        let (v, rest) = rest.split_at(pairs);
        let (inside_l, inside_r) = rest.split_at(inside_left);
        let keys = |range: std::ops::Range<usize>| -> Vec<&[u64]> {
            element_keys[range].iter().map(|k| &k[..]).collect()
        };
        let side = |mask: u64, inside: &[Letters], inside_keys: Vec<&[u64]>| {
            let mut zs: Vec<Letters> = u.iter().map(|&(x, z)| (x & mask, z & mask)).collect();
            zs.extend_from_slice(inside);
            let riders: Vec<Letters> = v.iter().map(|&(x, z)| (x & mask, z & mask)).collect();
            let excess: Vec<usize> = part
                .excess
                .iter()
                .copied()
                .filter(|&i| {
                    let (p, _) = self.gauge.excess[i];
                    (p.x | p.z) & mask != 0
                })
                .collect();
            let (amps, cumulative) = self.side_state(mask, &excess, &zs, &riders);
            let mut bit_keys = keys(0..pairs);
            bit_keys.extend(inside_keys);
            Side {
                amps,
                cumulative,
                keys: KeyTable::new(&bit_keys, words),
            }
        };
        let left_side = side(cut.left, inside_l, keys(2 * pairs..2 * pairs + inside_left));
        let right_side = side(
            right,
            inside_r,
            keys(2 * pairs + inside_left..elements.len()),
        );
        CutDraw {
            pairs,
            left: left_side,
            right: right_side,
            pair_keys: KeyTable::new(&keys(pairs..2 * pairs), words),
        }
    }

    /// Amplitudes of one side of a cut over the basis where `zs[j]` reads bit `j`, phased
    /// so that `riders[j]` flips bit `j` with no phase, and their running sums of `|ψ|^2`.
    fn side_state(
        &self,
        mask: u64,
        excess: &[usize],
        zs: &[Letters],
        riders: &[Letters],
    ) -> (Vec<Complex64>, Vec<f64>) {
        let width = mask.count_ones() as usize;
        debug_assert_eq!(zs.len(), width);
        let mut amps = self.gauge.part_state(mask, excess);
        let mut rows: Vec<RegisterPauli> = zs
            .iter()
            .chain(riders)
            .map(|&(x, z)| RegisterPauli {
                x: compress(x, mask),
                z: compress(z, mask),
                phase4: 0,
            })
            .collect();
        let (pivots, flips, _) = diagonalize(&mut amps, &mut rows, width, width);
        let riders = &rows[width..];
        let flipped = pivots
            .iter()
            .zip(&flips)
            .fold(0usize, |m, (&c, &f)| m | usize::from(f) << c);
        let mut scatter = vec![[0usize; 256]; width.div_ceil(8)];
        for (byte, lookup) in scatter.iter_mut().enumerate() {
            for (value, entry) in lookup.iter_mut().enumerate() {
                *entry = (0..8)
                    .filter(|&j| (value >> j) & 1 == 1 && 8 * byte + j < width)
                    .fold(0, |m, j| m | 1 << pivots[8 * byte + j]);
            }
        }
        let index = |t: usize| {
            scatter
                .iter()
                .enumerate()
                .fold(flipped, |b, (byte, lookup)| {
                    b ^ lookup[(t >> (8 * byte)) & 255]
                })
        };
        let pairs = riders.len();
        let block = 1usize << pairs;
        let mut phase = vec![0u32; block];
        let mut psi = vec![Complex64::new(0.0, 0.0); amps.len()];
        let mut cumulative = vec![0.0f64; amps.len()];
        let mut total = 0.0;
        let conj_powers = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ];
        for (o, (psi, cumulative)) in psi
            .chunks_exact_mut(block)
            .zip(cumulative.chunks_exact_mut(block))
            .enumerate()
        {
            for x in 0..block {
                let t = o << pairs | x;
                if x > 0 {
                    let j = (usize::BITS - 1 - x.leading_zeros()) as usize;
                    let from = index(t ^ 1 << j);
                    let rider = riders[j];
                    debug_assert_eq!(rider.x, 1 << pivots[j]);
                    phase[x] = phase[x ^ 1 << j]
                        + u32::from(rider.phase4)
                        + (rider.x & rider.z).count_ones()
                        + 2 * (from as u64 & rider.z).count_ones();
                }
                psi[x] = amps[index(t)] * conj_powers[(phase[x] & 3) as usize];
                total += psi[x].norm_sqr();
                cumulative[x] = total;
            }
        }
        (psi, cumulative)
    }
}

/// Register amplitude updates one entry of a Walsh readout costs per term the fill
/// evaluates, its allocation included: 2.6 to 4.9 ns per entry on parts of 13 to 18 rows
/// with no excess, threaded, against 0.24 to 0.38 ns per dense register update.
const WALSH_TERM_UNITS: f64 = 10.0;
/// Register amplitude updates one entry of a Walsh readout costs per transform level:
/// 0.02 to 0.05 ns while the table fits in cache, 0.16 ns at `2^24` entries.
const WALSH_LEVEL_UNITS: f64 = 0.25;
/// Table entries one task of the Walsh fill covers.
const WALSH_FILL_CHUNK_LOG: usize = 10;
/// Smallest Walsh fill split over the Rayon pool.
#[cfg(feature = "parallel")]
const WALSH_FILL_PAR_LOG: usize = 12;

/// Register amplitude updates to read a whole part out of its dense amplitudes: one pass
/// for the anchors, one per excess rotation and about `3 (r + 1)` to diagonalize the rows.
fn dense_readout_work(width: usize, rank: usize, excess: usize) -> f64 {
    (excess + 1 + 3 * (rank + 1)) as f64 * 2f64.powi(width as i32)
}

/// Register amplitude updates of a [`WalshReadout`] of `2^rank` entries: the fill
/// evaluates about `1.5^excess` terms per entry, as many as half the rotations splitting
/// each one in two, and the transform takes `rank` levels.
fn walsh_readout_work(rank: usize, excess: usize) -> f64 {
    let terms = 1.5f64.powi(excess as i32);
    2f64.powi(rank as i32) * (WALSH_TERM_UNITS * terms + WALSH_LEVEL_UNITS * rank as f64)
}

/// Expectation values on one part in the product gauge of every product of its basis
/// rows. A Pauli string is propagated back through the part's excess rotations, and each
/// term is read off the anchors' Bloch vectors, whose components are `0` or `±1/√2`.
struct WalshReadout {
    /// The basis rows on the part's bits, packed.
    rows: Vec<RegisterPauli>,
    /// Bits whose anchor has `<X> = 0`, and those with `<Y> = 0`.
    blind: [u64; 2],
    /// Bits whose anchor has a negative `<X>`, `<Y>` and `<Z>`.
    negative: [u64; 3],
    /// `2^(-w/2)` by weight `w`.
    scale: [f64; 65],
    /// `(R, cos 2θ, sin 2θ)` for each excess rotation `exp(-i θ R)` on the part, packed,
    /// in circuit order.
    excess: Vec<(RegisterPauli, f64, f64)>,
}

impl WalshReadout {
    fn new(gauge: &ProductGauge, mask: u64, excess: &[usize], basis: &[Letters]) -> Self {
        let mut blind = [0u64; 2];
        let mut negative = [0u64; 3];
        let mut rest = mask;
        let mut i = 0;
        while rest != 0 {
            let [a, b] = gauge.anchors[rest.trailing_zeros() as usize];
            rest &= rest - 1;
            let ab = a.conj() * b;
            let bloch = [2.0 * ab.re, 2.0 * ab.im, a.norm_sqr() - b.norm_sqr()];
            for (axis, &v) in bloch.iter().enumerate() {
                if v == 0.0 {
                    debug_assert!(axis < 2, "an anchor has <Z> = cos(π/4)");
                    blind[axis] |= 1 << i;
                } else {
                    debug_assert!((v.abs() - FRAC_1_SQRT_2).abs() < 1e-12);
                    negative[axis] |= u64::from(v < 0.0) << i;
                }
            }
            i += 1;
        }
        let mut scale = [0.0; 65];
        for (w, s) in scale.iter_mut().enumerate() {
            let half = 0.5f64.powi((w / 2) as i32);
            *s = if w % 2 == 1 {
                half * FRAC_1_SQRT_2
            } else {
                half
            };
        }
        let pack = |x: u64, z: u64, phase4: u8| RegisterPauli {
            x: compress(x, mask),
            z: compress(z, mask),
            phase4,
        };
        Self {
            rows: basis.iter().map(|&(x, z)| pack(x, z, 0)).collect(),
            blind,
            negative,
            scale,
            excess: excess
                .iter()
                .map(|&j| {
                    let (p, theta) = gauge.excess[j];
                    let (sin, cos) = (2.0 * theta).sin_cos();
                    (pack(p.x, p.z, p.phase4), cos, sin)
                })
                .collect(),
        }
    }

    /// `table[s] = <Π_{j in s} B_j>` for every subset `s` of the rows. Each task walks one
    /// aligned chunk in Gray-code order, so a step multiplies one row in.
    fn fill(&self, table: &mut [f64]) {
        let chunk = table.len().min(1 << WALSH_FILL_CHUNK_LOG);
        let fill_chunk = |(c, entries): (usize, &mut [f64])| {
            let low = entries.len().trailing_zeros();
            let mut q = self.product(c << low);
            for i in 0..entries.len() {
                entries[i ^ (i >> 1)] = self.value(q, self.excess.len());
                let next = i + 1;
                if next < entries.len() {
                    q = q.mul(self.rows[next.trailing_zeros() as usize]);
                }
            }
        };
        #[cfg(feature = "parallel")]
        if table.len() >= 1 << WALSH_FILL_PAR_LOG {
            use rayon::prelude::*;
            table.par_chunks_mut(chunk).enumerate().for_each(fill_chunk);
            return;
        }
        table.chunks_mut(chunk).enumerate().for_each(fill_chunk);
    }

    fn product(&self, subset: usize) -> RegisterPauli {
        let mut q = RegisterPauli {
            x: 0,
            z: 0,
            phase4: 0,
        };
        let mut rest = subset;
        while rest != 0 {
            q = q.mul(self.rows[rest.trailing_zeros() as usize]);
            rest &= rest - 1;
        }
        q
    }

    /// `<q>` after the first `rotations` excess rotations: `exp(iθR) q exp(-iθR)` is `q`
    /// when `q` commutes with `R`, else `cos 2θ q - i sin 2θ q R`.
    fn value(&self, q: RegisterPauli, rotations: usize) -> f64 {
        if rotations == 0 {
            return self.product_value(q);
        }
        let (r, cos, sin) = self.excess[rotations - 1];
        if ((q.x & r.z) ^ (q.z & r.x)).count_ones() & 1 == 0 {
            return self.value(q, rotations - 1);
        }
        let mut qr = q.mul(r);
        qr.phase4 = (qr.phase4 + 3) & 3;
        cos * self.value(q, rotations - 1) + sin * self.value(qr, rotations - 1)
    }

    /// `<q>` on the product of the anchors.
    #[inline(always)]
    fn product_value(&self, q: RegisterPauli) -> f64 {
        debug_assert_eq!(q.phase4 & 1, 0, "a Hermitian term has a real sign");
        let (xs, ys, zs) = (q.x & !q.z, q.x & q.z, q.z & !q.x);
        if (xs & self.blind[0]) | (ys & self.blind[1]) != 0 {
            return 0.0;
        }
        let flips = (xs & self.negative[0]) | (ys & self.negative[1]) | (zs & self.negative[2]);
        let magnitude = self.scale[(q.x | q.z).count_ones() as usize];
        if (flips.count_ones() + u32::from(q.phase4 >> 1)) & 1 == 1 {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// Basis of the pieces on a part by the ordered reduction over the part's `x` bits
/// ascending, then its `z` bits: the pivot rows in the order found.
fn ordered_basis(pieces: &mut [Letters], mask: u64) -> Vec<Letters> {
    ordered_reduction(pieces, mask)
        .into_iter()
        .map(|i| pieces[i])
        .collect()
}

/// For each column, `x` bits of `mask` ascending then its `z` bits, the first row in list
/// order that is not yet a pivot and has the column's letter becomes its pivot and is
/// multiplied into every other row with that letter. Returns the pivots in the order found.
fn ordered_reduction(rows: &mut [Letters], mask: u64) -> Vec<usize> {
    let mut pivots = Vec::new();
    for z in [false, true] {
        let mut rest = mask;
        while rest != 0 {
            let b = rest.trailing_zeros();
            rest &= rest - 1;
            let has = |row: &Letters| ((if z { row.1 } else { row.0 }) >> b) & 1 == 1;
            let Some(pivot) = (0..rows.len()).find(|i| !pivots.contains(i) && has(&rows[*i]))
            else {
                continue;
            };
            pivots.push(pivot);
            let p = rows[pivot];
            for (i, row) in rows.iter_mut().enumerate() {
                if i != pivot && has(row) {
                    row.0 ^= p.0;
                    row.1 ^= p.1;
                }
            }
        }
    }
    pivots
}

/// The elements a cut part draws, as letters on the part: the `u` row of each pair, the
/// `v` row of each pair, the inside rows of the left side, then those of the right.
fn cut_elements(pieces: &mut [Letters], mask: u64, cut: CutChoice) -> Vec<Letters> {
    let left = cut.left;
    let right = mask & !left;
    let mut rows = ordered_basis(pieces, mask);
    let anti = |a: Letters, b: Letters| (((a.0 & b.1) ^ (a.1 & b.0)) & left).count_ones() & 1 == 1;
    let (mut u, mut v) = (Vec::new(), Vec::new());
    loop {
        let found = (0..rows.len()).find_map(|i| {
            (i + 1..rows.len())
                .find(|&j| anti(rows[i], rows[j]))
                .map(|j| (i, j))
        });
        let Some((i, j)) = found else {
            break;
        };
        let pv = rows.remove(j);
        let pu = rows.remove(i);
        for row in rows.iter_mut() {
            if anti(*row, pv) {
                row.0 ^= pu.0;
                row.1 ^= pu.1;
            }
            if anti(*row, pu) {
                row.0 ^= pv.0;
                row.1 ^= pv.1;
            }
        }
        u.push(pu);
        v.push(pv);
    }
    debug_assert_eq!(u.len(), cut.pairs);
    let inside = |columns: u64| {
        let mut reduced = rows.clone();
        let pivots = ordered_reduction(&mut reduced, columns);
        (0..reduced.len())
            .filter(|i| !pivots.contains(i))
            .map(|i| reduced[i])
            .collect::<Vec<Letters>>()
    };
    let inside_left = inside(right);
    let inside_right = inside(left);
    debug_assert_eq!(inside_left.len(), cut.left_width - cut.pairs);
    debug_assert_eq!(inside_right.len(), cut.right_width - cut.pairs);
    debug_assert!(inside_left.iter().all(|&(x, z)| (x | z) & right == 0));
    debug_assert!(inside_right.iter().all(|&(x, z)| (x | z) & left == 0));
    u.into_iter()
        .chain(v)
        .chain(inside_left)
        .chain(inside_right)
        .collect()
}

/// Expands a letter string over independent commuting Paulis.
struct Expander {
    /// Each element reduced against the earlier ones, as `x | z << 64`, with its pivot bit
    /// and the elements it sums.
    rows: Vec<(u128, u32, u64)>,
}

impl Expander {
    fn new(elements: &[Letters]) -> Self {
        let mut expander = Self { rows: Vec::new() };
        for (i, &(x, z)) in elements.iter().enumerate() {
            let (v, combo) = expander.reduce(u128::from(x) | u128::from(z) << 64, 1u64 << i);
            debug_assert_ne!(v, 0, "the elements are independent");
            expander.rows.push((v, v.trailing_zeros(), combo));
        }
        expander
    }

    fn reduce(&self, mut v: u128, mut combo: u64) -> (u128, u64) {
        for &(row, pivot, sum) in &self.rows {
            if (v >> pivot) & 1 == 1 {
                v ^= row;
                combo ^= sum;
            }
        }
        (v, combo)
    }

    /// The elements whose product is `piece` up to a sign, and whether that sign is `-1`.
    fn expand(&self, elements: &[Letters], piece: Letters) -> (u64, bool) {
        let (rest, combo) = self.reduce(u128::from(piece.0) | u128::from(piece.1) << 64, 0);
        debug_assert_eq!(rest, 0, "the piece lies in the span of the elements");
        let mut product = RegisterPauli {
            x: 0,
            z: 0,
            phase4: 0,
        };
        let mut bits = combo;
        while bits != 0 {
            let (x, z) = elements[bits.trailing_zeros() as usize];
            bits &= bits - 1;
            product = product.mul(RegisterPauli { x, z, phase4: 0 });
        }
        debug_assert!(product.x == piece.0 && product.z == piece.1 && product.phase4 & 1 == 0);
        (combo, product.phase4 == 2)
    }
}

/// Replace each entry by the sum of the entries up to it, adding in index order.
fn running_sums(table: &mut [f64]) {
    let mut total = 0.0;
    for p in table {
        total += *p;
        *p = total;
    }
}

/// Classical keys of the bits of a drawn index, one table of 256 keys per index byte.
struct KeyTable {
    words: usize,
    /// `256 * words` entries per index byte: the XOR of the keys of the byte's set bits.
    entries: Vec<u64>,
}

impl KeyTable {
    fn new(bit_keys: &[&[u64]], words: usize) -> Self {
        let bytes = bit_keys.len().div_ceil(8);
        let mut entries = vec![0u64; bytes * 256 * words];
        for (byte, table) in entries.chunks_exact_mut(256 * words).enumerate() {
            for value in 1..256usize {
                let low = value.trailing_zeros() as usize;
                let (done, entry) = table.split_at_mut(value * words);
                let previous = &done[(value & (value - 1)) * words..][..words];
                let key = bit_keys.get(8 * byte + low);
                for (w, e) in entry[..words].iter_mut().enumerate() {
                    *e = previous[w] ^ key.map_or(0, |k| k[w]);
                }
            }
        }
        Self { words, entries }
    }

    #[inline(always)]
    fn apply(&self, index: u64, key: &mut [u64]) {
        for (byte, table) in self.entries.chunks_exact(256 * self.words).enumerate() {
            let value = ((index >> (8 * byte)) & 255) as usize;
            for (k, e) in key.iter_mut().zip(&table[value * self.words..]) {
                *k ^= e;
            }
        }
    }
}

/// Small whole tables multiplied into one, the first member's index bits lowest.
struct Pack {
    lowest: u32,
    bits: usize,
    members: Vec<Vec<f64>>,
    keys: Vec<Vec<u64>>,
}

impl Pack {
    fn push(&mut self, table: Vec<f64>, keys: Vec<Vec<u64>>) {
        self.bits += keys.len();
        self.members.push(table);
        self.keys.extend(keys);
    }

    fn into_draw(self, words: usize) -> Draw {
        let mut cumulative = vec![0.0f64; 1 << self.bits];
        for (t, entry) in cumulative.iter_mut().enumerate() {
            let mut shift = 0;
            let mut product = 1.0;
            for (m, member) in self.members.iter().enumerate() {
                let bits = member.len().trailing_zeros();
                let p = member[(t >> shift) & ((1 << bits) - 1)];
                product = if m == 0 { p } else { product * p };
                shift += bits;
            }
            *entry = product;
        }
        running_sums(&mut cumulative);
        let keys: Vec<&[u64]> = self.keys.iter().map(|k| &k[..]).collect();
        Draw::Table {
            cumulative,
            keys: KeyTable::new(&keys, words),
        }
    }
}

struct Side {
    /// `ψ` over the side's basis: the `u` pieces in the low bits, the inside rows above.
    amps: Vec<Complex64>,
    cumulative: Vec<f64>,
    keys: KeyTable,
}

struct CutDraw {
    pairs: usize,
    left: Side,
    right: Side,
    /// Keys of the `v` row of each pair.
    pair_keys: KeyTable,
}

enum Draw {
    Table {
        cumulative: Vec<f64>,
        keys: KeyTable,
    },
    Cut(Box<CutDraw>),
}

/// The index a word picks from a table of running sums: the count of entries at most
/// `u C[last]`, `u` the word's top 53 bits as a fraction, capped at the last entry.
#[inline(always)]
fn pick(cumulative: &[f64], word: u64) -> usize {
    let u = (word >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
    let draw = u * cumulative[cumulative.len() - 1];
    cumulative
        .partition_point(|&c| c <= draw)
        .min(cumulative.len() - 1)
}

/// Per-block scratch for the pair draw of a cut.
struct PairScratch {
    sums: Vec<Complex64>,
    cumulative: Vec<f64>,
}

impl CutDraw {
    /// Draw both sides and the pairs from three words, XORing the drawn bits' keys into
    /// `key`.
    fn draw(&self, words: [u64; 3], scratch: &mut PairScratch, key: &mut [u64]) {
        let left = pick(&self.left.cumulative, words[0]);
        let right = pick(&self.right.cumulative, words[1]);
        self.left.keys.apply(left as u64, key);
        self.right.keys.apply(right as u64, key);
        let cumulative = self.pair_sums(left, right, scratch);
        self.pair_keys.apply(pick(cumulative, words[2]) as u64, key);
    }

    /// Running sums over `b` of the weight of the `v` rows reading `b` given side indices
    /// `left` and `right`: `|Σ_x (-1)^(b·x) ψ_L[o_L, x] ψ_R[o_R, x ^ d]|^2`, `o` each
    /// side's inside rows and `d` the `u` rows' outcomes.
    fn pair_sums<'a>(&self, left: usize, right: usize, scratch: &'a mut PairScratch) -> &'a [f64] {
        let block = 1usize << self.pairs;
        let low = block - 1;
        let d = (left ^ right) & low;
        let psi_left = &self.left.amps[left & !low..][..block];
        let psi_right = &self.right.amps[right & !low..][..block];
        let sums = &mut scratch.sums[..block];
        for (x, sum) in sums.iter_mut().enumerate() {
            *sum = psi_left[x] * psi_right[x ^ d];
        }
        let mut half = 1;
        while half < block {
            for chunk in sums.chunks_exact_mut(2 * half) {
                let (lo, hi) = chunk.split_at_mut(half);
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (s, t) = (*a, *b);
                    *a = s + t;
                    *b = s - t;
                }
            }
            half *= 2;
        }
        let cumulative = &mut scratch.cumulative[..block];
        let mut total = 0.0;
        for (c, s) in cumulative.iter_mut().zip(sums.iter()) {
            total += s.norm_sqr();
            *c = total;
        }
        cumulative
    }
}

/// Terminal shots of a split register. Every shot reads the same number of 64-bit words:
/// one per 64 uniform rows, then one per table and three per cut in order of lowest
/// register bit. Shot `s` takes its words from stream `s / 256 + 2` of the ChaCha8
/// generator seeded with the run seed, after those of the earlier shots in its block.
pub(in crate::sim) struct SplitSampler {
    /// Keys of the uniform rows, 64 to a coin word.
    coin_keys: Vec<KeyTable>,
    draws: Vec<Draw>,
    /// Classical bits every shot starts from.
    base: Vec<u64>,
    max_pairs: Option<usize>,
    num_classical_bits: usize,
}

impl SplitSampler {
    /// Compile `circuit`, lowered to Clifford+T, or return `None` when the split declines.
    pub(in crate::sim) fn compile(circuit: &Circuit) -> Result<Option<Self>> {
        Ok(SplitPlan::new(circuit)?.map(SplitPlan::sampler))
    }

    fn words_per_shot(&self) -> usize {
        self.coin_keys.len()
            + self
                .draws
                .iter()
                .map(|draw| match draw {
                    Draw::Table { .. } => 1,
                    Draw::Cut(_) => 3,
                })
                .sum::<usize>()
    }

    pub(in crate::sim) fn sample(&self, num_shots: usize, seed: u64) -> Vec<Vec<bool>> {
        let mut shots = vec![Vec::new(); num_shots];
        let parallel = draws_split_across_workers(num_shots, self.words_per_shot().max(1));
        sample_in_shot_blocks(&mut shots, 1, seed, parallel, |rng, block| {
            let mut scratch = self.scratch();
            let mut key = vec![0u64; self.base.len()];
            for shot in block {
                self.draw(rng, &mut scratch, &mut key);
                *shot = (0..self.num_classical_bits)
                    .map(|cb| (key[cb >> 6] >> (cb & 63)) & 1 == 1)
                    .collect();
            }
        });
        shots
    }

    fn scratch(&self) -> PairScratch {
        let block = self.max_pairs.map_or(0, |pairs| 1usize << pairs);
        PairScratch {
            sums: vec![Complex64::new(0.0, 0.0); block],
            cumulative: vec![0.0; block],
        }
    }

    /// Draw one shot's classical bits into `key`.
    fn draw<R: Rng>(&self, rng: &mut R, scratch: &mut PairScratch, key: &mut [u64]) {
        key.copy_from_slice(&self.base);
        for coins in &self.coin_keys {
            coins.apply(rng.next_u64(), key);
        }
        for draw in &self.draws {
            match draw {
                Draw::Table { cumulative, keys } => {
                    keys.apply(pick(cumulative, rng.next_u64()) as u64, key);
                }
                Draw::Cut(cut) => {
                    let words = [rng.next_u64(), rng.next_u64(), rng.next_u64()];
                    cut.draw(words, scratch, key);
                }
            }
        }
    }
}

#[cfg(test)]
impl SplitSampler {
    /// The exact distribution the draws induce, indexed with classical bit `i` at bit `i`.
    pub(in crate::sim) fn distribution(&self) -> Vec<f64> {
        let bits = self.num_classical_bits;
        assert!(bits <= 20, "dense over 2^{bits} outcomes");
        let size = 1usize << bits;
        let key_of = |table: &KeyTable, index: usize| {
            let mut key = vec![0u64; self.base.len()];
            table.apply(index as u64, &mut key);
            key[0] as usize
        };
        let weights = |cumulative: &[f64]| {
            let total = cumulative[cumulative.len() - 1];
            (0..cumulative.len())
                .map(|i| (cumulative[i] - if i == 0 { 0.0 } else { cumulative[i - 1] }) / total)
                .collect::<Vec<f64>>()
        };
        let mut factors: Vec<Vec<f64>> = Vec::new();
        for coins in &self.coin_keys {
            for b in 0..64 {
                let mut d = vec![0.0; size];
                d[0] += 0.5;
                d[key_of(coins, 1 << b)] += 0.5;
                factors.push(d);
            }
        }
        let mut scratch = self.scratch();
        for draw in &self.draws {
            let mut d = vec![0.0; size];
            match draw {
                Draw::Table { cumulative, keys } => {
                    for (i, p) in weights(cumulative).into_iter().enumerate() {
                        d[key_of(keys, i)] += p;
                    }
                }
                Draw::Cut(cut) => {
                    let right_weights = weights(&cut.right.cumulative);
                    for (left, pl) in weights(&cut.left.cumulative).into_iter().enumerate() {
                        for (right, &pr) in right_weights.iter().enumerate() {
                            if pl * pr == 0.0 {
                                continue;
                            }
                            let side_key =
                                key_of(&cut.left.keys, left) ^ key_of(&cut.right.keys, right);
                            let pairs = weights(cut.pair_sums(left, right, &mut scratch));
                            for (b, q) in pairs.into_iter().enumerate() {
                                d[side_key ^ key_of(&cut.pair_keys, b)] += pl * pr * q;
                            }
                        }
                    }
                }
            }
            factors.push(d);
        }
        let mut spectrum = vec![1.0f64; size];
        for mut d in factors {
            walsh(&mut d);
            for (s, v) in spectrum.iter_mut().zip(&d) {
                *s *= v;
            }
        }
        walsh(&mut spectrum);
        let base = self.base[0] as usize;
        let mut dist = vec![0.0; size];
        for (k, s) in spectrum.iter().enumerate() {
            dist[k ^ base] = s / size as f64;
        }
        dist
    }
}

#[cfg(test)]
fn walsh(v: &mut [f64]) {
    let mut half = 1;
    while half < v.len() {
        for chunk in v.chunks_exact_mut(2 * half) {
            let (lo, hi) = chunk.split_at_mut(half);
            for (a, b) in lo.iter_mut().zip(hi) {
                let (s, t) = (*a, *b);
                *a = s + t;
                *b = s - t;
            }
        }
        half *= 2;
    }
}

#[cfg(test)]
#[path = "clifford_split_tests.rs"]
mod tests;
