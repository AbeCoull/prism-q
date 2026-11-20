//! Forward stabilizer compile of a Clifford circuit's measurements, taken for registers of
//! 64 or more qubits.

use crate::circuit::{Circuit, Instruction};
use crate::error::Result;

use super::propagation::{colmajor_forward_sim, rowmul_phase};
use super::{CompiledSampler, finish_sampler, live_forward, require_compilable, xor_words};

/// Compile a Clifford circuit's measurements via forward stabilizer
/// simulation; [`compile_measurements`](super::compile_measurements) routes
/// registers of 64 or more qubits here, and the same circuit requirements apply.
///
/// # Examples
///
/// ```
/// use prism_q::{CircuitBuilder, HistogramAccumulator, compile_forward};
///
/// let circuit = CircuitBuilder::new(2).h(0).cx(0, 1).measure_all().build();
/// let mut sampler = compile_forward(&circuit, 42).expect("compile failed");
/// assert_eq!(sampler.rank(), 1);
///
/// let mut acc = HistogramAccumulator::new();
/// sampler.sample_chunked(1000, &mut acc);
/// let counts = acc.into_counts();
/// // Bell state: only 00 and 11 occur.
/// assert_eq!(counts.len(), 2);
/// assert_eq!(counts.values().sum::<u64>(), 1000);
/// ```
pub fn compile_forward(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    require_compilable(circuit)?;
    compile_forward_unchecked(circuit, seed)
}

pub(super) fn compile_forward_unchecked(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    let density_cap = LIVE_FORWARD_DENSITY * circuit.num_qubits;
    let bound = live_forward::peak_live_lower_bound(circuit);
    if bound * bound > density_cap {
        return compile_forward_dense(circuit, seed);
    }
    let schedule = live_forward::LiveSchedule::plan(circuit);
    if schedule.peak_live * schedule.peak_live <= density_cap {
        return live_forward::compile_forward_live(circuit, seed, &schedule);
    }
    compile_forward_dense(circuit, seed)
}

/// Take the live-slot compile while the peak live count squared is at most this many
/// times the register. Its cost per record grows with the square of the live count and
/// the dense tableau's with the register; on random Clifford circuits that keep every
/// qubit live the two meet near 200 times the register.
pub(super) const LIVE_FORWARD_DENSITY: usize = 64;

/// Row words a random collapse in [`compile_forward_dense`] multiplies before its row
/// pass splits across the pool: two tasks of
/// [`MIN_PAR_ELEMS`](crate::backend::MIN_PAR_ELEMS) words.
#[cfg(feature = "parallel")]
const MIN_PAR_COLLAPSE_WORDS: usize = 2 * crate::backend::MIN_PAR_ELEMS;

/// [`compile_forward`] over the whole register at once, measured at the end.
pub(crate) fn compile_forward_dense(circuit: &Circuit, seed: u64) -> Result<CompiledSampler> {
    let measurements: Vec<(usize, usize)> = circuit
        .instructions
        .iter()
        .filter_map(|inst| match inst {
            Instruction::Measure {
                qubit,
                classical_bit,
            } => Some((*qubit, *classical_bit)),
            _ => None,
        })
        .collect();

    let num_measurements = measurements.len();
    if num_measurements == 0 {
        return Ok(CompiledSampler::empty(seed));
    }

    let n = circuit.num_qubits;

    let (mut xz, mut phase, nw) = colmajor_forward_sim(n, &circuit.instructions)?;
    let stride = 2 * nw;
    let m = num_measurements;
    let m_words = m.div_ceil(64);

    let rank_words = m_words;
    let total_rows = 2 * n;
    let mut gen_dep: Vec<u64> = vec![0u64; (total_rows + 1) * rank_words];
    let dep_row = |r: usize| r * rank_words..(r + 1) * rank_words;
    let mut ref_bits: Vec<bool> = vec![false; m];
    let mut rank = 0usize;

    let mut flip_rows: Vec<Vec<u64>> = Vec::with_capacity(m);
    let mut p_data: Vec<u64> = vec![0u64; stride];
    let mut p_dep: Vec<u64> = vec![0u64; rank_words];
    let mut scratch: Vec<u64> = vec![0u64; stride];
    let scratch_idx = total_rows;
    let mut columns = XWordColumns::new(total_rows);
    let mut hits: Vec<u64> = vec![0u64; columns.row_words];
    let destab_words = n.div_ceil(64);
    let mut held: Vec<u64> = vec![0u64; destab_words];
    let mut held_valid = false;
    let mut scratch_phase = false;

    for (meas_idx, &(qubit, _)) in measurements.iter().enumerate() {
        let word = qubit / 64;
        let bit = qubit % 64;
        let bit_mask = 1u64 << bit;
        columns.load(&xz, stride, word);

        if let Some(p_row) = first_row_from(columns.column(bit), n) {
            // Random measurement, this is the k-th random degree of freedom
            let k = rank;
            rank += 1;
            flip_rows.push(vec![0u64; m_words]);

            flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);

            let p_base = p_row * stride;
            p_data.copy_from_slice(&xz[p_base..p_base + stride]);
            let p_phase = phase[p_row];
            p_dep.copy_from_slice(&gen_dep[dep_row(p_row)]);

            hits.copy_from_slice(columns.column(bit));
            hits[p_row / 64] &= !(1u64 << (p_row % 64));

            // Dependencies only reach random bits below k, so the words past them are
            // zero in every row.
            let live = k.div_ceil(64);
            let collapse = |row: &mut [u64], sign: &mut bool, dep: &mut [u64]| {
                *sign = rowmul_phase(&p_data, row, nw, p_phase, *sign);
                xor_words(&mut dep[..live], &p_dep[..live]);
            };
            #[cfg(feature = "parallel")]
            let hit_rows: usize = hits.iter().map(|w| w.count_ones() as usize).sum();
            #[cfg(feature = "parallel")]
            let done_parallel = if hit_rows * stride >= MIN_PAR_COLLAPSE_WORDS {
                use rayon::prelude::*;
                xz.par_chunks_mut(64 * stride)
                    .zip(phase.par_chunks_mut(64))
                    .zip(gen_dep[..total_rows * rank_words].par_chunks_mut(64 * rank_words))
                    .zip(hits.par_iter())
                    .with_min_len(crate::backend::chunk_min_len(64 * stride))
                    .for_each(|(((rows, signs), deps), word_hits)| {
                        for_each_row(std::slice::from_ref(word_hits), signs.len(), |i| {
                            let row = &mut rows[i * stride..(i + 1) * stride];
                            let dep = &mut deps[i * rank_words..(i + 1) * rank_words];
                            collapse(row, &mut signs[i], dep);
                        });
                    });
                true
            } else {
                false
            };
            #[cfg(not(feature = "parallel"))]
            let done_parallel = false;
            if !done_parallel {
                for_each_row(&hits, total_rows, |r| {
                    let row = &mut xz[r * stride..(r + 1) * stride];
                    collapse(row, &mut phase[r], &mut gen_dep[dep_row(r)]);
                });
            }
            // Every multiplied row's X word moved by the pivot's.
            for (w, &rows) in hits.iter().enumerate() {
                if rows != 0 {
                    columns.update_rows(w, rows, p_data[word]);
                }
            }

            let dest_idx = p_row - n;
            let dest_base = dest_idx * stride;
            columns.update(dest_idx, xz[dest_base + word], p_data[word]);
            xz.copy_within(p_base..p_base + stride, dest_base);
            phase[dest_idx] = p_phase;
            gen_dep[dep_row(dest_idx)].copy_from_slice(&p_dep);

            columns.update(p_row, p_data[word], 0);
            xz[p_base..p_base + stride].fill(0);
            xz[p_base + nw + word] |= bit_mask;
            phase[p_row] = false;

            gen_dep[dep_row(p_row)].fill(0);
            gen_dep[p_row * rank_words + k / 64] |= 1u64 << (k % 64);

            ref_bits[meas_idx] = false;
            held_valid = false;
        } else {
            // The scratch row holds the product of the stabilizers in `held`. They
            // commute and square to +I, so the product this measurement needs is
            // the held one times the stabilizers in the symmetric difference.
            let live = rank.div_ceil(64);
            let column = columns.column(bit);
            let tail = (1u64 << (n % 64)).wrapping_sub(1);
            let mut direct = 0u32;
            let mut changed = 0u32;
            for w in 0..destab_words {
                let mask = if w + 1 == destab_words && tail != 0 {
                    tail
                } else {
                    !0
                };
                let now = column[w] & mask;
                direct += now.count_ones();
                changed += (now ^ held[w]).count_ones();
                hits[w] = now ^ held[w];
                held[w] = now;
            }
            if !held_valid || changed > direct {
                scratch[..stride].fill(0);
                scratch_phase = false;
                gen_dep[scratch_idx * rank_words..][..live].fill(0);
                hits[..destab_words].copy_from_slice(&held);
            }
            held_valid = true;

            for_each_row(&hits[..destab_words], n, |g| {
                let s_base = (g + n) * stride;
                let s_phase = phase[g + n];
                scratch_phase = rowmul_phase(
                    &xz[s_base..s_base + stride],
                    &mut scratch,
                    nw,
                    s_phase,
                    scratch_phase,
                );

                let (lo, hi) = gen_dep.split_at_mut(scratch_idx * rank_words);
                for (dst, &src) in hi[..live]
                    .iter_mut()
                    .zip(&lo[(g + n) * rank_words..][..live])
                {
                    *dst ^= src;
                }
            });

            ref_bits[meas_idx] = scratch_phase;

            for_each_row(&gen_dep[scratch_idx * rank_words..][..live], rank, |k| {
                flip_rows[k][meas_idx / 64] |= 1u64 << (meas_idx % 64);
            });
        }
    }

    Ok(finish_sampler(
        flip_rows,
        rank,
        num_measurements,
        &ref_bits,
        seed,
    ))
}

/// X bits of one 64-qubit word of a row-major tableau, transposed to one row bitset
/// per qubit, so a measurement reads the rows it touches from a contiguous column
/// instead of one strided word per row.
struct XWordColumns {
    word: usize,
    row_words: usize,
    total_rows: usize,
    bits: Vec<u64>,
}

impl XWordColumns {
    fn new(total_rows: usize) -> Self {
        let row_words = total_rows.div_ceil(64);
        Self {
            word: usize::MAX,
            row_words,
            total_rows,
            bits: vec![0u64; 64 * row_words],
        }
    }

    /// Point the index at `word`, rebuilding it only when the word changes.
    fn load(&mut self, xz: &[u64], stride: usize, word: usize) {
        if self.word == word {
            return;
        }
        self.word = word;
        self.bits.fill(0);
        for r in 0..self.total_rows {
            let mut v = xz[r * stride + word];
            while v != 0 {
                let b = v.trailing_zeros() as usize;
                self.bits[b * self.row_words + r / 64] |= 1u64 << (r % 64);
                v &= v - 1;
            }
        }
    }

    /// Record that the X word of `row` changed from `old` to `new`.
    #[inline]
    fn update(&mut self, row: usize, old: u64, new: u64) {
        let mut diff = old ^ new;
        while diff != 0 {
            let b = diff.trailing_zeros() as usize;
            self.bits[b * self.row_words + row / 64] ^= 1u64 << (row % 64);
            diff &= diff - 1;
        }
    }

    /// Record that the X word of every row set in `rows`, word `w` of a row bitset,
    /// changed by `diff`.
    #[inline]
    fn update_rows(&mut self, w: usize, rows: u64, mut diff: u64) {
        while diff != 0 {
            let b = diff.trailing_zeros() as usize;
            self.bits[b * self.row_words + w] ^= rows;
            diff &= diff - 1;
        }
    }

    fn column(&self, bit: usize) -> &[u64] {
        &self.bits[bit * self.row_words..(bit + 1) * self.row_words]
    }
}

/// First set row of `column` at or after `from`.
fn first_row_from(column: &[u64], from: usize) -> Option<usize> {
    let mut w = from / 64;
    let mut bits = column.get(w)? & (!0u64 << (from % 64));
    loop {
        if bits != 0 {
            return Some(w * 64 + bits.trailing_zeros() as usize);
        }
        w += 1;
        bits = *column.get(w)?;
    }
}

/// Call `f` on each set row of `column` below `end`, in ascending order.
#[inline]
fn for_each_row(column: &[u64], end: usize, mut f: impl FnMut(usize)) {
    for (w, &word) in column.iter().enumerate().take(end.div_ceil(64)) {
        let mut bits = word;
        if (w + 1) * 64 > end {
            bits &= (1u64 << (end % 64)).wrapping_sub(1);
        }
        while bits != 0 {
            f(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}
