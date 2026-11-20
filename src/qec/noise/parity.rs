//! The parity noise sampler: compiled faults landed straight on detector and observable
//! rows.

#[cfg(feature = "parallel")]
use super::qec_noise_units_in_parallel;
use super::{
    QEC_NOISE_UNIT_SHOTS, QecNoiseDraw, QecProductTable, draw_qec_unit, qec_noise_unit_rng,
};
#[cfg(feature = "parallel")]
use crate::backend::SendMutPtr;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Output rows per bucket when a unit lands its flips: 256 rows of one unit's 128 words
/// are 256 KB, inside the L2 share of one core.
const QEC_SCATTER_BLOCK_ROWS: usize = 256;

/// Drawn flips a unit buffers before landing them, 8 MB of packed entries.
const QEC_SCATTER_FLUSH: usize = 1 << 20;

/// Buffered flips from which a unit lands its buckets on the pool instead of in place.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_LAND_FLIPS: usize = 1 << 15;

/// Compiled noise events whose flips land on measurement-major detector and observable
/// rows instead of measurement records. Branch `b` flips the output rows
/// `outputs[branch_offsets[b]..branch_offsets[b + 1]]`.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(in crate::qec) struct QecParityNoise {
    events: Vec<QecParityNoiseEvent>,
    fire_rates: Vec<f64>,
    branch_offsets: Vec<u32>,
    outputs: Vec<u32>,
    pair_tables: Vec<[f64; 15]>,
    product_tables: Vec<QecProductTable>,
    seed: u64,
}

/// Branches run from `first_branch`: Z, Y, X for a single-qubit event, the 15
/// depolarize-2 branches in `qec_pair_branch_flips` order for a pair.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(in crate::qec) struct QecParityNoiseEvent {
    pub(in crate::qec) draw: QecNoiseDraw,
    pub(in crate::qec) first_branch: usize,
}

impl QecParityNoise {
    /// Zeroed rows of `row_words` words for `num_rows` outputs with the noise of all
    /// `shots` shots XORed in, drawn exactly as the record path draws.
    pub(in crate::qec) fn sample_rows(
        &self,
        num_rows: usize,
        row_words: usize,
        shots: usize,
    ) -> Vec<u64> {
        let mut rows = vec![0u64; num_rows * row_words];
        let units = shots.div_ceil(QEC_NOISE_UNIT_SHOTS);
        if units == 0 || row_words == 0 {
            return rows;
        }
        #[cfg(feature = "parallel")]
        {
            let rows_len = rows.len();
            let rows = SendMutPtr(rows.as_mut_ptr());
            let flip = move |offset: usize, bit: u64| {
                debug_assert!(offset < rows_len);
                // SAFETY: `offset` is an output row's word for a shot of the unit being
                // landed, so it lies in `rows`. Unit `unit` writes only words
                // `unit * U / 64..(unit + 1) * U / 64` of each row, with
                // `U = QEC_NOISE_UNIT_SHOTS` a multiple of 64, so no two units touch the
                // same word, and within a unit each block of rows lands on one thread.
                unsafe { rows.xor_word(offset, bit) }
            };
            let clear = move |offset: usize| {
                debug_assert!(offset < rows_len);
                // SAFETY: `offset` is a word of the unit being landed, in `rows` and
                // written by one thread only, for the reasons given in `flip`.
                unsafe { rows.write_word(offset, 0) }
            };
            if qec_noise_units_in_parallel(
                units,
                shots,
                self.events.iter().map(|event| &event.draw),
            ) {
                (0..units).into_par_iter().for_each(|unit| {
                    self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear)
                });
            } else {
                for unit in 0..units {
                    self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear);
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            let cells = std::cell::Cell::from_mut(rows.as_mut_slice()).as_slice_of_cells();
            let flip = |offset: usize, bit: u64| cells[offset].set(cells[offset].get() ^ bit);
            let clear = |offset: usize| cells[offset].set(0);
            for unit in 0..units {
                self.scatter_unit(unit, shots, row_words, num_rows, &flip, &clear);
            }
        }
        rows
    }

    /// Draw unit `unit` into one bucket of flips per block of output rows, then land the
    /// buckets, so the words a bucket's flips touch stay in cache: the rows span far more
    /// memory than the cache and a fault's outputs are spread over all of it.
    fn scatter_unit(
        &self,
        unit: usize,
        shots: usize,
        row_words: usize,
        num_rows: usize,
        flip: &impl LandFlip,
        clear: &impl LandClear,
    ) {
        let first_word = unit * QEC_NOISE_UNIT_SHOTS / 64;
        let span = UnitSpan {
            first_word,
            words: (row_words - first_word).min(QEC_NOISE_UNIT_SHOTS / 64),
            row_words,
            num_rows,
        };
        let mut buckets: Vec<LandBucket> = (0..num_rows.div_ceil(QEC_SCATTER_BLOCK_ROWS))
            .map(|_| LandBucket::default())
            .collect();
        let mut buffered = 0usize;
        self.draw_unit(unit, shots, |output, word, bit| {
            buckets[output / QEC_SCATTER_BLOCK_ROWS].flips.push(
                ((output as u64) << 32)
                    | (((word - first_word) as u64) << 6)
                    | u64::from(bit.trailing_zeros()),
            );
            buffered += 1;
            if buffered == QEC_SCATTER_FLUSH {
                land_buckets(&mut buckets, span, flip, clear);
                buffered = 0;
            }
        });
        land_buckets(&mut buckets, span, flip, clear);
    }

    /// Draw unit `unit` of a `shots`-shot run, calling `flip(output, word, bit)` for every
    /// output bit a fault toggles.
    #[inline(always)]
    fn draw_unit(&self, unit: usize, shots: usize, mut flip: impl FnMut(usize, usize, u64)) {
        let first_shot = unit * QEC_NOISE_UNIT_SHOTS;
        let unit_shots = (shots - first_shot).min(QEC_NOISE_UNIT_SHOTS);
        let mut rng = qec_noise_unit_rng(self.seed, unit);
        draw_qec_unit(
            &self.events,
            |event| &event.draw,
            &self.fire_rates,
            &self.pair_tables,
            &self.product_tables,
            &mut rng,
            unit_shots,
            |_, event, shot, branch| {
                let shot = first_shot + shot;
                let word = shot / 64;
                let bit = 1u64 << (shot % 64);
                let branch = event.first_branch + branch;
                let range =
                    self.branch_offsets[branch] as usize..self.branch_offsets[branch + 1] as usize;
                for &output in &self.outputs[range] {
                    flip(output as usize, word, bit);
                }
            },
        );
    }

    #[cfg(test)]
    pub(in crate::qec) fn event_count(&self) -> usize {
        self.events.len()
    }

    pub(in crate::qec) fn reseed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub(in crate::qec) fn from_parts(
        seed: u64,
        events: Vec<QecParityNoiseEvent>,
        branch_offsets: Vec<u32>,
        outputs: Vec<u32>,
        pair_tables: Vec<[f64; 15]>,
        product_tables: Vec<QecProductTable>,
    ) -> Self {
        let fire_rates = events.iter().map(|event| event.draw.rate()).collect();
        Self {
            events,
            fire_rates,
            branch_offsets,
            outputs,
            pair_tables,
            product_tables,
            seed,
        }
    }
}

/// `flip(offset, bit)` XORs `bit` into word `offset` of the rows, shared across the
/// pool's threads when there is one, so a unit's blocks land in parallel.
#[cfg(feature = "parallel")]
trait LandFlip: Fn(usize, u64) + Sync {}
#[cfg(feature = "parallel")]
impl<F: Fn(usize, u64) + Sync> LandFlip for F {}
#[cfg(not(feature = "parallel"))]
trait LandFlip: Fn(usize, u64) {}
#[cfg(not(feature = "parallel"))]
impl<F: Fn(usize, u64)> LandFlip for F {}

/// `clear(offset)` writes zero to word `offset` of the rows without reading it.
#[cfg(feature = "parallel")]
trait LandClear: Fn(usize) + Sync {}
#[cfg(feature = "parallel")]
impl<F: Fn(usize) + Sync> LandClear for F {}
#[cfg(not(feature = "parallel"))]
trait LandClear: Fn(usize) {}
#[cfg(not(feature = "parallel"))]
impl<F: Fn(usize)> LandClear for F {}

/// Words `first_word..first_word + words` of each of `num_rows` rows of `row_words`
/// words: the part of the rows one noise unit writes.
#[derive(Clone, Copy)]
struct UnitSpan {
    first_word: usize,
    words: usize,
    row_words: usize,
    num_rows: usize,
}

/// Flips drawn for one block of rows, packed `output << 32 | unit_word << 6 | bit`, and
/// whether the block's span has been cleared.
#[derive(Default)]
struct LandBucket {
    flips: Vec<u64>,
    cleared: bool,
}

/// Land every bucket's flips, a bucket per thread when there is a pool and enough of
/// them, then empty the buckets. Before its first flip lands, a block's span is cleared
/// in row order: the rows are freshly zeroed, so this only faults the pages in, and a
/// page faulted in by a plain write costs less than one faulted in by a flip's read.
fn land_buckets(
    buckets: &mut [LandBucket],
    span: UnitSpan,
    flip: &impl LandFlip,
    clear: &impl LandClear,
) {
    let land_bucket = |block: usize, bucket: &mut LandBucket| {
        if bucket.flips.is_empty() {
            return;
        }
        if !bucket.cleared {
            bucket.cleared = true;
            let first_row = block * QEC_SCATTER_BLOCK_ROWS;
            for row in first_row..(first_row + QEC_SCATTER_BLOCK_ROWS).min(span.num_rows) {
                let base = row * span.row_words + span.first_word;
                clear(base);
                clear(base + span.words - 1);
            }
        }
        for &packed in &bucket.flips {
            let output = (packed >> 32) as usize;
            let word = ((packed >> 6) & 0x03FF_FFFF) as usize;
            flip(
                output * span.row_words + span.first_word + word,
                1u64 << (packed & 63),
            );
        }
        bucket.flips.clear();
    };
    #[cfg(feature = "parallel")]
    if buckets
        .iter()
        .map(|bucket| bucket.flips.len())
        .sum::<usize>()
        >= QEC_PARALLEL_LAND_FLIPS
    {
        buckets
            .par_iter_mut()
            .enumerate()
            .for_each(|(block, bucket)| land_bucket(block, bucket));
        return;
    }
    for (block, bucket) in buckets.iter_mut().enumerate() {
        land_bucket(block, bucket);
    }
}
