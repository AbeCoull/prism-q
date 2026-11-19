//! Record flips of a program's Pauli feed-forward corrections.

use super::record::{Frame, record_walk_order, walk_record_frame};
use super::{QecDeferredFeedforward, QecDeferredProgram};
use crate::sim::compiled::xor_words;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Record flips of a program's Pauli feed-forward corrections, applied in op order
/// once noise has landed.
///
/// A Pauli fired at a position flips the same later records whatever shot it fires
/// in, so correction `k` XORs `parity(condition) ^ invert` into each record of
/// `flips`. Every condition record precedes the correction and every flipped record
/// follows it, so applying corrections in op order lets a condition read records an
/// earlier correction flipped.
pub(in crate::qec) struct QecFeedforwardFlips {
    corrections: Vec<QecFeedforwardCorrection>,
}

struct QecFeedforwardCorrection {
    condition: Vec<usize>,
    invert: bool,
    flips: Vec<usize>,
}

/// Words of every record row the feed-forward pass holds at once: 256 words (2 KB) per
/// row keep a block of a few hundred records in L2 while each correction reads its
/// condition rows and flips its targets.
const QEC_FEEDFORWARD_BLOCK_WORDS: usize = 256;

/// Row words a feed-forward pass touches from which its blocks run on the pool.
#[cfg(feature = "parallel")]
const QEC_PARALLEL_FEEDFORWARD_WORDS: usize = 1 << 18;

impl QecFeedforwardFlips {
    /// `None` when no correction of the program can flip a record.
    pub(super) fn compile(deferred: &QecDeferredProgram) -> Option<Self> {
        if deferred.feedforwards.is_empty() {
            return None;
        }
        let (aliases_by_last_use, record_of_alias) = record_walk_order(deferred);
        let batch_flips = |batch: &[QecDeferredFeedforward]| {
            feedforward_batch_flips(deferred, &aliases_by_last_use, &record_of_alias, batch)
        };
        #[cfg(feature = "parallel")]
        let batches: Vec<Vec<Vec<usize>>> = deferred
            .feedforwards
            .par_chunks(64)
            .map(batch_flips)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let batches: Vec<Vec<Vec<usize>>> =
            deferred.feedforwards.chunks(64).map(batch_flips).collect();
        let flips = batches.into_iter().flatten();

        let corrections: Vec<QecFeedforwardCorrection> = deferred
            .feedforwards
            .iter()
            .zip(flips)
            .filter(|(_, flips)| !flips.is_empty())
            .map(|(feedforward, flips)| QecFeedforwardCorrection {
                condition: feedforward.condition.clone(),
                invert: feedforward.invert,
                flips,
            })
            .collect();
        (!corrections.is_empty()).then_some(Self { corrections })
    }

    /// Each correction as `(condition, invert, flips)` in op order; `flips` is ascending
    /// and never empty.
    pub(in crate::qec) fn corrections(
        &self,
    ) -> impl Iterator<Item = (&[usize], bool, &[usize])> + '_ {
        self.corrections.iter().map(|correction| {
            (
                correction.condition.as_slice(),
                correction.invert,
                correction.flips.as_slice(),
            )
        })
    }

    /// Apply every correction to measurement-major `data` of `num_shots` shots, a block
    /// of row words at a time.
    pub(super) fn apply(&self, data: &mut [u64], num_shots: usize) {
        let s_words = num_shots.div_ceil(64);
        let tail = match num_shots % 64 {
            0 => u64::MAX,
            bits => (1u64 << bits) - 1,
        };
        let num_blocks = s_words.div_ceil(QEC_FEEDFORWARD_BLOCK_WORDS);
        let num_records = data.len() / s_words;
        let mut blocks: Vec<Vec<&mut [u64]>> = (0..num_blocks)
            .map(|_| Vec::with_capacity(num_records))
            .collect();
        for row in data.chunks_exact_mut(s_words) {
            for (block, words) in blocks
                .iter_mut()
                .zip(row.chunks_mut(QEC_FEEDFORWARD_BLOCK_WORDS))
            {
                block.push(words);
            }
        }
        let apply_block = |(index, rows): (usize, &mut Vec<&mut [u64]>)| {
            self.apply_block(
                rows,
                if index + 1 == num_blocks {
                    tail
                } else {
                    u64::MAX
                },
            );
        };
        #[cfg(feature = "parallel")]
        if num_blocks > 1 {
            let touched_rows: usize = self
                .corrections
                .iter()
                .map(|correction| correction.condition.len() + correction.flips.len())
                .sum();
            if touched_rows * s_words >= QEC_PARALLEL_FEEDFORWARD_WORDS {
                blocks.par_iter_mut().enumerate().for_each(apply_block);
                return;
            }
        }
        blocks.iter_mut().enumerate().for_each(apply_block);
    }

    /// Apply every correction to one block of record rows, `tail` masking the block's
    /// last word.
    #[inline(always)]
    fn apply_block(&self, rows: &mut [&mut [u64]], tail: u64) {
        let words = rows[0].len();
        let mut fire = [0u64; QEC_FEEDFORWARD_BLOCK_WORDS];
        let fire = &mut fire[..words];
        for correction in &self.corrections {
            fire.fill(if correction.invert { u64::MAX } else { 0 });
            for &record in &correction.condition {
                xor_words(fire, rows[record]);
            }
            fire[words - 1] &= tail;
            if fire.iter().all(|&word| word == 0) {
                continue;
            }
            for &record in &correction.flips {
                xor_words(rows[record], fire);
            }
        }
    }
}

/// The records each correction of `batch` (at most 64, positions ascending) flips: one
/// forward frame walk from the first correction, with correction `k` of the batch
/// injected as a fault in lane `k`.
fn feedforward_batch_flips(
    deferred: &QecDeferredProgram,
    aliases_by_last_use: &[u32],
    record_of_alias: &[u32],
    batch: &[QecDeferredFeedforward],
) -> Vec<Vec<usize>> {
    debug_assert!(!batch.is_empty() && batch.len() <= 64);
    let mut flips = vec![Vec::new(); batch.len()];
    let mut frame = Frame::new(1, deferred.num_aliases);
    let mut next = 0usize;
    walk_record_frame(
        deferred,
        aliases_by_last_use,
        record_of_alias,
        &mut frame,
        batch[0].position,
        |position, frame| {
            while next < batch.len() && batch[next].position == position {
                for &(alias, letter) in &batch[next].paulis {
                    let slot = frame.slot(alias as usize);
                    frame.flip(slot, next, letter as usize);
                }
                next += 1;
            }
        },
        |record, x_row| {
            let mut lanes = x_row[0];
            while lanes != 0 {
                flips[lanes.trailing_zeros() as usize].push(record);
                lanes &= lanes - 1;
            }
        },
    );
    for record_list in &mut flips {
        record_list.sort_unstable();
    }
    flips
}
