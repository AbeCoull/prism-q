//! Block-triangular sampling (BTS): draws one random u64 per rank column per
//! 64-shot batch and forms each measurement row as the XOR of its parity
//! columns' words, emitting measurement-major packed shots.

use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use super::SendPtrU64;
use super::parity::SparseParity;
#[cfg(target_arch = "aarch64")]
use super::rng::Xoshiro256PlusPlusX2;
#[cfg(target_arch = "x86_64")]
use super::rng::Xoshiro256PlusPlusX4;
use super::rng::{Xoshiro256PlusPlus, Xoshiro256PlusPlusLanes};
use super::shot_tail_mask;

pub(super) const BTS_BATCH_SHOTS: usize = 65536;

struct BtsOutput<'a> {
    ptr: *mut u64,
    rows: usize,
    stride: usize,
    offset: usize,
    words: usize,
    borrow: PhantomData<&'a mut [u64]>,
}

impl<'a> BtsOutput<'a> {
    fn contiguous(output: &'a mut [u64], rows: usize, words: usize) -> Self {
        assert_eq!(Some(output.len()), rows.checked_mul(words));
        Self {
            ptr: output.as_mut_ptr(),
            rows,
            stride: words,
            offset: 0,
            words,
            borrow: PhantomData,
        }
    }

    /// Zero the exclusive word range in each measurement row.
    ///
    /// # Safety
    /// Every range `ptr + row * stride + offset .. + words` must be valid for writes and
    /// exclusively borrowed for `'a`; the ranges must not overlap.
    unsafe fn new(ptr: *mut u64, rows: usize, stride: usize, offset: usize, words: usize) -> Self {
        // SAFETY: same contract as the enclosing unsafe fn.
        unsafe {
            if stride == words && offset == 0 {
                ptr.write_bytes(0, rows * words);
            } else {
                for row in 0..rows {
                    ptr.add(row * stride + offset).write_bytes(0, words);
                }
            }
        }
        Self {
            ptr,
            rows,
            stride,
            offset,
            words,
            borrow: PhantomData,
        }
    }

    #[inline(always)]
    fn row_mut(&mut self, row: usize) -> &mut [u64] {
        assert!(row < self.rows);
        // SAFETY: Both constructors retain an initialized, exclusive range for every row.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr.add(row * self.stride + self.offset),
                self.words,
            )
        }
    }

    fn apply_ref_bits(&mut self, ref_bits: &[u64], num_shots: usize) {
        if self.stride == self.words && self.offset == 0 {
            // SAFETY: The initialized exclusive rows are contiguous at this stride.
            let output =
                unsafe { std::slice::from_raw_parts_mut(self.ptr, self.rows * self.words) };
            apply_ref_bits_meas_major(output, ref_bits, self.rows, self.words, num_shots);
        } else {
            let words = self.words;
            for row in 0..self.rows {
                let bit = (ref_bits[row / 64] >> (row % 64)) & 1;
                apply_ref_bits_meas_major(self.row_mut(row), &[bit], 1, words, num_shots);
            }
        }
    }
}

#[cfg(feature = "parallel")]
impl SendPtrU64 {
    #[inline(always)]
    fn as_mut_ptr(self) -> *mut u64 {
        self.0
    }
}

#[inline(always)]
fn xor_reduce_scalar(cols: &[u32], random_bits: &[u64]) -> u64 {
    match cols.len() {
        0 => 0,
        1 => random_bits[cols[0] as usize],
        2 => random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize],
        3 => {
            random_bits[cols[0] as usize]
                ^ random_bits[cols[1] as usize]
                ^ random_bits[cols[2] as usize]
        }
        4 => {
            (random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize])
                ^ (random_bits[cols[2] as usize] ^ random_bits[cols[3] as usize])
        }
        _ => {
            let mut chunks = cols.chunks_exact(4);
            let mut acc = 0u64;
            for chunk in &mut chunks {
                acc ^= (random_bits[chunk[0] as usize] ^ random_bits[chunk[1] as usize])
                    ^ (random_bits[chunk[2] as usize] ^ random_bits[chunk[3] as usize]);
            }
            for &c in chunks.remainder() {
                acc ^= random_bits[c as usize];
            }
            acc
        }
    }
}

pub(super) fn bts_single_pass(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    sample_bts_meas_major(sparse, num_shots, ref_bits, rng, rank)
}

pub(super) fn bts_batched(
    sparse: &SparseParity,
    num_shots: usize,
    total_s_words: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;
    let total_len = num_meas
        .checked_mul(total_s_words)
        .expect("BTS output size overflows usize");
    let mut output: Vec<u64> = Vec::with_capacity(total_len);

    #[cfg(feature = "parallel")]
    {
        let num_threads = rayon::current_num_threads();
        if num_threads > 1 {
            let shots_per_thread = (num_shots.div_ceil(num_threads) / 64) * 64;
            if shots_per_thread >= 64 {
                let thread_seeds: Vec<[u64; 4]> = (0..num_threads)
                    .map(|_| {
                        [
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                            rng.next_u64(),
                        ]
                    })
                    .collect();

                let chunks: Vec<(usize, usize)> = (0..num_threads)
                    .map(|t| {
                        let start = t * shots_per_thread;
                        let end = if t + 1 == num_threads {
                            num_shots
                        } else {
                            (t + 1) * shots_per_thread
                        };
                        (start, end.min(num_shots))
                    })
                    .filter(|(s, e)| s < e)
                    .collect();

                {
                    use rayon::prelude::*;
                    let ptr = SendPtrU64(output.as_mut_ptr());
                    let total_sw = total_s_words;
                    let nm = num_meas;

                    chunks
                        .into_par_iter()
                        .enumerate()
                        .for_each(|(t, (shot_start, shot_end))| {
                            let chunk_shots = shot_end - shot_start;
                            let word_offset = shot_start / 64;
                            let mut thread_rng = Xoshiro256PlusPlus::from_seeds(thread_seeds[t]);

                            let mut chunk_done = 0usize;
                            while chunk_done < chunk_shots {
                                let batch_shots = (chunk_shots - chunk_done).min(BTS_BATCH_SHOTS);
                                let batch_s_words = batch_shots.div_ceil(64);
                                let batch_offset = word_offset + chunk_done / 64;
                                // SAFETY: Shot chunks have disjoint, 64-aligned word ranges
                                // in every row. Each batch stays in its worker's range and
                                // the allocation remains live until all workers finish.
                                let mut batch_output = unsafe {
                                    BtsOutput::new(
                                        ptr.as_mut_ptr(),
                                        nm,
                                        total_sw,
                                        batch_offset,
                                        batch_s_words,
                                    )
                                };
                                sample_bts_into(
                                    sparse,
                                    batch_shots,
                                    ref_bits,
                                    &mut thread_rng,
                                    rank,
                                    &mut batch_output,
                                );

                                chunk_done += batch_shots;
                            }
                        });
                }

                // SAFETY: The completed batches initialized every word of every row.
                unsafe { output.set_len(total_len) };
                return output;
            }
        }
    }

    let mut shots_done = 0usize;

    while shots_done < num_shots {
        let batch_shots = (num_shots - shots_done).min(BTS_BATCH_SHOTS);
        let batch_s_words = batch_shots.div_ceil(64);
        let word_offset = shots_done / 64;

        // SAFETY: Every batch owns a disjoint word range in each allocated row,
        // and no reference into output survives the batch.
        let mut batch_output = unsafe {
            BtsOutput::new(
                output.as_mut_ptr(),
                num_meas,
                total_s_words,
                word_offset,
                batch_s_words,
            )
        };
        sample_bts_into(sparse, batch_shots, ref_bits, rng, rank, &mut batch_output);

        shots_done += batch_shots;
    }

    // SAFETY: The completed batches initialized every word of every row.
    unsafe { output.set_len(total_len) };
    output
}

pub(super) fn sample_bts_meas_major(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    let words = num_shots.div_ceil(64);
    let len = sparse
        .num_rows
        .checked_mul(words)
        .expect("BTS output size overflows usize");
    let mut output = vec![0; len];
    let mut dest = BtsOutput::contiguous(&mut output, sparse.num_rows, words);
    sample_bts_into(sparse, num_shots, ref_bits, rng, rank, &mut dest);
    output
}

fn sample_bts_into(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    output: &mut BtsOutput<'_>,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && num_shots >= 256 {
            // SAFETY: AVX2 detected, all pointer arithmetic bounded by allocation sizes
            return unsafe {
                sample_bts_meas_major_avx2(sparse, num_shots, ref_bits, rng, rank, output)
            };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_shots >= 128 {
            // SAFETY: NEON is baseline on aarch64, pointers are valid
            return unsafe {
                sample_bts_meas_major_neon(sparse, num_shots, ref_bits, rng, rank, output)
            };
        }
    }

    sample_bts_meas_major_scalar(sparse, num_shots, ref_bits, rng, rank, output)
}

/// The portable path, drawing shot word `w` from lane `w % 4` of the four-lane stream
/// the vector paths lay across their registers, so all three agree bitwise.
fn sample_bts_meas_major_scalar(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    output: &mut BtsOutput<'_>,
) {
    let s_words = num_shots.div_ceil(64);
    let mut lanes = Xoshiro256PlusPlusLanes::from_scalar(rng);
    let mut random_bits = vec![0u64; rank];

    for batch in 0..s_words {
        let lane = batch % 4;
        for r in random_bits.iter_mut().take(rank) {
            *r = lanes.next_u64(lane);
        }
        if batch == s_words - 1 {
            let mask = shot_tail_mask(num_shots);
            if mask != u64::MAX {
                for r in random_bits.iter_mut().take(rank) {
                    *r &= mask;
                }
            }
        }

        for &m in &sparse.non_det_rows {
            let m = m as usize;
            let cols = sparse.row_cols(m);
            let acc = xor_reduce_scalar(cols, &random_bits);
            output.row_mut(m)[batch] = acc;
        }
    }

    output.apply_ref_bits(ref_bits, num_shots);
}

pub(super) fn apply_ref_bits_meas_major(
    meas_major: &mut [u64],
    ref_bits: &[u64],
    num_meas: usize,
    s_words: usize,
    num_shots: usize,
) {
    for m in 0..num_meas {
        let ref_bit = (ref_bits[m / 64] >> (m % 64)) & 1;
        if ref_bit != 0 {
            let row = &mut meas_major[m * s_words..(m + 1) * s_words];
            for w in row.iter_mut() {
                *w ^= !0u64;
            }
        }
    }
    super::clear_meas_major_shot_padding(meas_major, num_shots, num_meas, s_words);
}

#[cfg(target_arch = "x86_64")]
const BTS_QUAD_TILE: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sample_bts_meas_major_avx2(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    output: &mut BtsOutput<'_>,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;

        let s_words = num_shots.div_ceil(64);
        let s_quads = num_shots.div_ceil(256);

        let mut vrng = Xoshiro256PlusPlusX4::from_scalar(rng);

        let tile = if rank == 0 {
            s_quads
        } else {
            (16384 / (rank * 32)).clamp(1, BTS_QUAD_TILE).min(s_quads)
        };

        let rem = num_shots % 256;
        let full_quads = if rem == 0 { s_quads } else { s_quads - 1 };

        if tile >= 2 && full_quads >= tile {
            let mut random_tile: Vec<__m256i> = vec![_mm256_setzero_si256(); rank * tile];

            let mut quad_start = 0;
            while quad_start + tile <= full_quads {
                for t in 0..tile {
                    for r in 0..rank {
                        random_tile[r * tile + t] = vrng.next_m256i();
                    }
                }

                for &m in &sparse.non_det_rows {
                    let m = m as usize;
                    let cols = sparse.row_cols(m);
                    let row = output.row_mut(m);
                    let out_base = quad_start * 4;

                    match cols.len() {
                        0 => unreachable!(),
                        1 => {
                            let c0 = cols[0] as usize * tile;
                            for t in 0..tile {
                                _mm256_storeu_si256(
                                    row[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                    random_tile[c0 + t],
                                );
                            }
                        }
                        2 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            for t in 0..tile {
                                _mm256_storeu_si256(
                                    row[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                    _mm256_xor_si256(random_tile[c0 + t], random_tile[c1 + t]),
                                );
                            }
                        }
                        3 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            let c2 = cols[2] as usize * tile;
                            for t in 0..tile {
                                _mm256_storeu_si256(
                                    row[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                    _mm256_xor_si256(
                                        _mm256_xor_si256(random_tile[c0 + t], random_tile[c1 + t]),
                                        random_tile[c2 + t],
                                    ),
                                );
                            }
                        }
                        4 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            let c2 = cols[2] as usize * tile;
                            let c3 = cols[3] as usize * tile;
                            for t in 0..tile {
                                _mm256_storeu_si256(
                                    row[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                    _mm256_xor_si256(
                                        _mm256_xor_si256(random_tile[c0 + t], random_tile[c1 + t]),
                                        _mm256_xor_si256(random_tile[c2 + t], random_tile[c3 + t]),
                                    ),
                                );
                            }
                        }
                        _ => {
                            for t in 0..tile {
                                let a = xor_reduce_avx2_tiled(cols, &random_tile, tile, t);
                                _mm256_storeu_si256(
                                    row[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                    a,
                                );
                            }
                        }
                    }
                }

                quad_start += tile;
            }

            bts_avx2_remainder(
                sparse,
                output,
                &mut vrng,
                &mut random_tile,
                rank,
                s_words,
                s_quads,
                quad_start,
                tile,
                rem,
            );
        } else {
            let mut random_avx: Vec<__m256i> = vec![_mm256_setzero_si256(); rank];
            bts_avx2_per_quad(
                sparse,
                output,
                &mut vrng,
                &mut random_avx,
                rank,
                s_words,
                s_quads,
                0,
                rem,
            );
        }

        output.apply_ref_bits(ref_bits, num_shots);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_avx2_remainder(
    sparse: &SparseParity,
    output: &mut BtsOutput<'_>,
    vrng: &mut Xoshiro256PlusPlusX4,
    random_tile: &mut [std::arch::x86_64::__m256i],
    rank: usize,
    s_words: usize,
    s_quads: usize,
    quad_start: usize,
    tile: usize,
    rem: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;

        for quad in quad_start..s_quads {
            let base_sw = quad * 4;
            let words_this_quad = (s_words - base_sw).min(4);

            for r in 0..rank {
                random_tile[r * tile] = vrng.next_m256i();
            }

            if quad == s_quads - 1 && rem != 0 {
                let full_words = rem / 64;
                let tail_bits = rem % 64;
                let mut mask_buf = [!0u64; 4];
                for val in mask_buf
                    .iter_mut()
                    .skip(full_words + usize::from(tail_bits > 0))
                {
                    *val = 0;
                }
                if tail_bits > 0 {
                    mask_buf[full_words] = (1u64 << tail_bits) - 1;
                }
                let mask_vec = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
                for r in 0..rank {
                    random_tile[r * tile] = _mm256_and_si256(random_tile[r * tile], mask_vec);
                }
            }

            for &m in &sparse.non_det_rows {
                let m = m as usize;
                let cols = sparse.row_cols(m);
                let acc = match cols.len() {
                    0 => unreachable!(),
                    1 => random_tile[cols[0] as usize * tile],
                    2 => _mm256_xor_si256(
                        random_tile[cols[0] as usize * tile],
                        random_tile[cols[1] as usize * tile],
                    ),
                    3 => _mm256_xor_si256(
                        _mm256_xor_si256(
                            random_tile[cols[0] as usize * tile],
                            random_tile[cols[1] as usize * tile],
                        ),
                        random_tile[cols[2] as usize * tile],
                    ),
                    _ => xor_reduce_avx2_tiled(cols, random_tile, tile, 0),
                };

                let out_ptr = output.row_mut(m)[base_sw..].as_mut_ptr();
                if words_this_quad == 4 {
                    _mm256_storeu_si256(out_ptr as *mut __m256i, acc);
                } else {
                    let mut tmp = [0u64; 4];
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
                    for (w, &val) in tmp.iter().enumerate().take(words_this_quad) {
                        *out_ptr.add(w) = val;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_avx2_per_quad(
    sparse: &SparseParity,
    output: &mut BtsOutput<'_>,
    vrng: &mut Xoshiro256PlusPlusX4,
    random_avx: &mut [std::arch::x86_64::__m256i],
    rank: usize,
    s_words: usize,
    s_quads: usize,
    start_quad: usize,
    rem: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;

        for quad in start_quad..s_quads {
            let base_sw = quad * 4;
            let words_this_quad = (s_words - base_sw).min(4);

            for avx in random_avx.iter_mut().take(rank) {
                *avx = vrng.next_m256i();
            }

            if quad == s_quads - 1 && rem != 0 {
                let full_words = rem / 64;
                let tail_bits = rem % 64;
                let mut mask_buf = [!0u64; 4];
                for val in mask_buf
                    .iter_mut()
                    .skip(full_words + usize::from(tail_bits > 0))
                {
                    *val = 0;
                }
                if tail_bits > 0 {
                    mask_buf[full_words] = (1u64 << tail_bits) - 1;
                }
                let mask_vec = _mm256_loadu_si256(mask_buf.as_ptr() as *const __m256i);
                for avx in random_avx.iter_mut().take(rank) {
                    *avx = _mm256_and_si256(*avx, mask_vec);
                }
            }

            for &m in &sparse.non_det_rows {
                let m = m as usize;
                let cols = sparse.row_cols(m);
                let acc = match cols.len() {
                    0 => unreachable!(),
                    1 => random_avx[cols[0] as usize],
                    2 => {
                        _mm256_xor_si256(random_avx[cols[0] as usize], random_avx[cols[1] as usize])
                    }
                    3 => _mm256_xor_si256(
                        _mm256_xor_si256(
                            random_avx[cols[0] as usize],
                            random_avx[cols[1] as usize],
                        ),
                        random_avx[cols[2] as usize],
                    ),
                    4 => _mm256_xor_si256(
                        _mm256_xor_si256(
                            random_avx[cols[0] as usize],
                            random_avx[cols[1] as usize],
                        ),
                        _mm256_xor_si256(
                            random_avx[cols[2] as usize],
                            random_avx[cols[3] as usize],
                        ),
                    ),
                    _ => xor_reduce_avx2(cols, random_avx),
                };

                let out_ptr = output.row_mut(m)[base_sw..].as_mut_ptr();
                if words_this_quad == 4 {
                    _mm256_storeu_si256(out_ptr as *mut __m256i, acc);
                } else {
                    let mut tmp = [0u64; 4];
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, acc);
                    for (w, &val) in tmp.iter().enumerate().take(words_this_quad) {
                        *out_ptr.add(w) = val;
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xor_reduce_avx2_tiled(
    cols: &[u32],
    random_tile: &[std::arch::x86_64::__m256i],
    tile: usize,
    t: usize,
) -> std::arch::x86_64::__m256i {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let mut chunks = cols.chunks_exact(4);
        let mut acc = _mm256_setzero_si256();
        for chunk in &mut chunks {
            acc = _mm256_xor_si256(
                acc,
                _mm256_xor_si256(
                    _mm256_xor_si256(
                        random_tile[chunk[0] as usize * tile + t],
                        random_tile[chunk[1] as usize * tile + t],
                    ),
                    _mm256_xor_si256(
                        random_tile[chunk[2] as usize * tile + t],
                        random_tile[chunk[3] as usize * tile + t],
                    ),
                ),
            );
        }
        for &c in chunks.remainder() {
            acc = _mm256_xor_si256(acc, random_tile[c as usize * tile + t]);
        }
        acc
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn xor_reduce_avx2(
    cols: &[u32],
    random: &[std::arch::x86_64::__m256i],
) -> std::arch::x86_64::__m256i {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::x86_64::*;
        let mut chunks = cols.chunks_exact(4);
        let mut acc = _mm256_setzero_si256();
        for chunk in &mut chunks {
            acc = _mm256_xor_si256(
                acc,
                _mm256_xor_si256(
                    _mm256_xor_si256(random[chunk[0] as usize], random[chunk[1] as usize]),
                    _mm256_xor_si256(random[chunk[2] as usize], random[chunk[3] as usize]),
                ),
            );
        }
        for &c in chunks.remainder() {
            acc = _mm256_xor_si256(acc, random[c as usize]);
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
const BTS_PAIR_TILE: usize = 8;

#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_neon(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
    output: &mut BtsOutput<'_>,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;

        let s_words = num_shots.div_ceil(64);
        let s_pairs = num_shots.div_ceil(128);

        // Pair p carries lanes 2(p % 2) and 2(p % 2) + 1 of the four-lane stream.
        let mut vrng = [
            Xoshiro256PlusPlusX2::from_scalar(rng),
            Xoshiro256PlusPlusX2::from_scalar(rng),
        ];

        let tile = if rank == 0 {
            s_pairs
        } else {
            (16384 / (rank * 16)).clamp(1, BTS_PAIR_TILE).min(s_pairs)
        };

        let rem = num_shots % 128;
        let full_pairs = if rem == 0 { s_pairs } else { s_pairs - 1 };

        if tile >= 2 && full_pairs >= tile {
            let mut random_tile: Vec<uint64x2_t> = vec![vdupq_n_u64(0); rank * tile];

            let mut pair_start = 0;
            while pair_start + tile <= full_pairs {
                for t in 0..tile {
                    let lanes = &mut vrng[(pair_start + t) % 2];
                    for r in 0..rank {
                        random_tile[r * tile + t] = lanes.next_uint64x2();
                    }
                }

                for &m in &sparse.non_det_rows {
                    let m = m as usize;
                    let cols = sparse.row_cols(m);
                    let row = output.row_mut(m);
                    let out_base = pair_start * 2;

                    match cols.len() {
                        0 => unreachable!(),
                        1 => {
                            let c0 = cols[0] as usize * tile;
                            for t in 0..tile {
                                vst1q_u64(
                                    row[out_base + t * 2..].as_mut_ptr(),
                                    random_tile[c0 + t],
                                );
                            }
                        }
                        2 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            for t in 0..tile {
                                vst1q_u64(
                                    row[out_base + t * 2..].as_mut_ptr(),
                                    veorq_u64(random_tile[c0 + t], random_tile[c1 + t]),
                                );
                            }
                        }
                        3 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            let c2 = cols[2] as usize * tile;
                            for t in 0..tile {
                                vst1q_u64(
                                    row[out_base + t * 2..].as_mut_ptr(),
                                    veorq_u64(
                                        veorq_u64(random_tile[c0 + t], random_tile[c1 + t]),
                                        random_tile[c2 + t],
                                    ),
                                );
                            }
                        }
                        4 => {
                            let c0 = cols[0] as usize * tile;
                            let c1 = cols[1] as usize * tile;
                            let c2 = cols[2] as usize * tile;
                            let c3 = cols[3] as usize * tile;
                            for t in 0..tile {
                                vst1q_u64(
                                    row[out_base + t * 2..].as_mut_ptr(),
                                    veorq_u64(
                                        veorq_u64(random_tile[c0 + t], random_tile[c1 + t]),
                                        veorq_u64(random_tile[c2 + t], random_tile[c3 + t]),
                                    ),
                                );
                            }
                        }
                        _ => {
                            for t in 0..tile {
                                let a = xor_reduce_neon_tiled(cols, &random_tile, tile, t);
                                vst1q_u64(row[out_base + t * 2..].as_mut_ptr(), a);
                            }
                        }
                    }
                }

                pair_start += tile;
            }

            bts_neon_per_pair(
                sparse, output, &mut vrng, rank, s_words, s_pairs, pair_start, rem,
            );
        } else {
            bts_neon_per_pair(sparse, output, &mut vrng, rank, s_words, s_pairs, 0, rem);
        }

        output.apply_ref_bits(ref_bits, num_shots);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_neon_per_pair(
    sparse: &SparseParity,
    output: &mut BtsOutput<'_>,
    vrng: &mut [Xoshiro256PlusPlusX2; 2],
    rank: usize,
    s_words: usize,
    s_pairs: usize,
    start_pair: usize,
    rem: usize,
) {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;

        let mut random_neon: Vec<uint64x2_t> = vec![vdupq_n_u64(0); rank];

        for pair in start_pair..s_pairs {
            let base_sw = pair * 2;
            let words_this_pair = (s_words - base_sw).min(2);

            let lanes = &mut vrng[pair % 2];
            for nval in random_neon.iter_mut().take(rank) {
                *nval = lanes.next_uint64x2();
            }

            if pair == s_pairs - 1 && rem != 0 {
                let full_words = rem / 64;
                let tail_bits = rem % 64;
                let mut mask_buf = [!0u64; 2];
                for val in mask_buf
                    .iter_mut()
                    .skip(full_words + usize::from(tail_bits > 0))
                {
                    *val = 0;
                }
                if tail_bits > 0 {
                    mask_buf[full_words] = (1u64 << tail_bits) - 1;
                }
                let mask_vec = vld1q_u64(mask_buf.as_ptr());
                for nval in random_neon.iter_mut().take(rank) {
                    *nval = vandq_u64(*nval, mask_vec);
                }
            }

            for &m in &sparse.non_det_rows {
                let m = m as usize;
                let cols = sparse.row_cols(m);
                let acc = match cols.len() {
                    0 => unreachable!(),
                    1 => random_neon[cols[0] as usize],
                    2 => veorq_u64(random_neon[cols[0] as usize], random_neon[cols[1] as usize]),
                    3 => veorq_u64(
                        veorq_u64(random_neon[cols[0] as usize], random_neon[cols[1] as usize]),
                        random_neon[cols[2] as usize],
                    ),
                    4 => veorq_u64(
                        veorq_u64(random_neon[cols[0] as usize], random_neon[cols[1] as usize]),
                        veorq_u64(random_neon[cols[2] as usize], random_neon[cols[3] as usize]),
                    ),
                    _ => xor_reduce_neon(cols, &random_neon),
                };

                let out_ptr = output.row_mut(m)[base_sw..].as_mut_ptr();
                if words_this_pair == 2 {
                    vst1q_u64(out_ptr, acc);
                } else {
                    *out_ptr = vgetq_lane_u64(acc, 0);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_reduce_neon_tiled(
    cols: &[u32],
    random_tile: &[std::arch::aarch64::uint64x2_t],
    tile: usize,
    t: usize,
) -> std::arch::aarch64::uint64x2_t {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let mut chunks = cols.chunks_exact(4);
        let mut acc = vdupq_n_u64(0);
        for chunk in &mut chunks {
            acc = veorq_u64(
                acc,
                veorq_u64(
                    veorq_u64(
                        random_tile[chunk[0] as usize * tile + t],
                        random_tile[chunk[1] as usize * tile + t],
                    ),
                    veorq_u64(
                        random_tile[chunk[2] as usize * tile + t],
                        random_tile[chunk[3] as usize * tile + t],
                    ),
                ),
            );
        }
        for &c in chunks.remainder() {
            acc = veorq_u64(acc, random_tile[c as usize * tile + t]);
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_reduce_neon(
    cols: &[u32],
    random: &[std::arch::aarch64::uint64x2_t],
) -> std::arch::aarch64::uint64x2_t {
    // SAFETY: same contract as the enclosing unsafe fn.
    unsafe {
        use std::arch::aarch64::*;
        let mut chunks = cols.chunks_exact(4);
        let mut acc = vdupq_n_u64(0);
        for chunk in &mut chunks {
            acc = veorq_u64(
                acc,
                veorq_u64(
                    veorq_u64(random[chunk[0] as usize], random[chunk[1] as usize]),
                    veorq_u64(random[chunk[2] as usize], random[chunk[3] as usize]),
                ),
            );
        }
        for &c in chunks.remainder() {
            acc = veorq_u64(acc, random[c as usize]);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    fn rng(seed: u64) -> Xoshiro256PlusPlus {
        let mut c = ChaCha8Rng::seed_from_u64(seed);
        Xoshiro256PlusPlus::from_chacha(&mut c)
    }

    fn scalar_samples(
        sparse: &SparseParity,
        num_shots: usize,
        ref_bits: &[u64],
        rng: &mut Xoshiro256PlusPlus,
        rank: usize,
    ) -> Vec<u64> {
        let words = num_shots.div_ceil(64);
        let mut data = vec![0; sparse.num_rows * words];
        let mut output = BtsOutput::contiguous(&mut data, sparse.num_rows, words);
        sample_bts_meas_major_scalar(sparse, num_shots, ref_bits, rng, rank, &mut output);
        data
    }

    #[test]
    fn xor_reduce_scalar_arities() {
        let bits = vec![0x01u64, 0x02, 0x04, 0x08, 0x10, 0x20];
        assert_eq!(xor_reduce_scalar(&[], &bits), 0);
        assert_eq!(xor_reduce_scalar(&[0], &bits), 0x01);
        assert_eq!(xor_reduce_scalar(&[0, 1], &bits), 0x03);
        assert_eq!(xor_reduce_scalar(&[0, 1, 2], &bits), 0x07);
        assert_eq!(xor_reduce_scalar(&[0, 1, 2, 3], &bits), 0x0F);
        assert_eq!(xor_reduce_scalar(&[0, 1, 2, 3, 4], &bits), 0x1F);
        assert_eq!(xor_reduce_scalar(&[0, 1, 2, 3, 4, 5], &bits), 0x3F);
    }

    #[test]
    fn apply_ref_bits_flips_set_rows() {
        let num_meas = 3;
        let s_words = 1;
        let mut m = vec![0u64; num_meas * s_words];
        m[0] = 0x0F;
        m[1] = 0x0F;
        m[2] = 0x0F;
        let ref_bits = vec![0b101u64];
        apply_ref_bits_meas_major(&mut m, &ref_bits, num_meas, s_words, 64);
        assert_eq!(m[0], !0x0Fu64);
        assert_eq!(m[1], 0x0F);
        assert_eq!(m[2], !0x0Fu64);
    }

    #[test]
    fn sample_bts_small_runs() {
        let rank = 2;
        let num_meas = 4;
        let flip_rows = vec![vec![0b0011u64], vec![0b0101u64]];
        let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
        let ref_bits = vec![0u64];
        let mut r = rng(42);
        let num_shots = 32;
        let out = sample_bts_meas_major(&sparse, num_shots, &ref_bits, &mut r, rank);
        assert_eq!(out.len(), num_meas * num_shots.div_ceil(64));
    }

    #[test]
    #[should_panic(expected = "BTS output size overflows usize")]
    fn batched_output_rejects_size_overflow() {
        let sparse = SparseParity::from_flip_rows(&[vec![1]], 64);
        bts_batched(
            &sparse,
            usize::MAX,
            usize::MAX.div_ceil(64),
            &[0],
            &mut rng(42),
            1,
        );
    }

    #[test]
    #[should_panic(expected = "BTS output size overflows usize")]
    fn single_pass_output_rejects_size_overflow() {
        let sparse = SparseParity::from_flip_rows(&[vec![1]], 64);
        sample_bts_meas_major(&sparse, usize::MAX, &[0], &mut rng(42), 1);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn contiguous_output_rejects_size_overflow() {
        BtsOutput::contiguous(&mut [], 64, usize::MAX.div_ceil(64));
    }

    // Every path lays the same four xoshiro lanes over the shot words, so the vector
    // kernel and the portable loop agree bitwise at any shot count and rank, and
    // both leave the scalar generator at the same point.
    #[test]
    fn vector_and_scalar_paths_draw_one_stream() {
        let num_meas = 40;
        let cases = [
            (64, 1),
            (100, 3),
            (128, 2),
            (200, 7),
            (256, 0),
            (300, 5),
            (1000, 17),
            (4096, 70),
            (5000, 2),
        ];
        for (num_shots, rank) in cases {
            let mut setup = rng(7 + rank as u64);
            let mask = (1u64 << num_meas) - 1;
            let flip_rows: Vec<Vec<u64>> =
                (0..rank).map(|_| vec![setup.next_u64() & mask]).collect();
            let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
            let ref_bits = vec![setup.next_u64() & mask];
            let mut scalar_rng = rng(99);
            let mut vector_rng = rng(99);
            let scalar = scalar_samples(&sparse, num_shots, &ref_bits, &mut scalar_rng, rank);
            let words = num_shots.div_ceil(64);
            let mut vector = vec![0; num_meas * words];
            let mut output = BtsOutput::contiguous(&mut vector, num_meas, words);
            #[cfg(target_arch = "x86_64")]
            {
                if !is_x86_feature_detected!("avx2") {
                    return;
                }
                // SAFETY: AVX2 detected
                unsafe {
                    sample_bts_meas_major_avx2(
                        &sparse,
                        num_shots,
                        &ref_bits,
                        &mut vector_rng,
                        rank,
                        &mut output,
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is baseline on aarch64
            unsafe {
                sample_bts_meas_major_neon(
                    &sparse,
                    num_shots,
                    &ref_bits,
                    &mut vector_rng,
                    rank,
                    &mut output,
                );
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            sample_bts_meas_major_scalar(
                &sparse,
                num_shots,
                &ref_bits,
                &mut vector_rng,
                rank,
                &mut output,
            );
            assert_eq!(scalar, vector, "{num_shots} shots, rank {rank}");
            assert_eq!(
                scalar_rng.next_u64(),
                vector_rng.next_u64(),
                "both paths consume the sixteen seeding draws"
            );
        }
    }

    #[test]
    fn strided_output_preserves_stream_and_neighbouring_words() {
        const SENTINEL: u64 = 0x1234_5678_9abc_def0;
        let num_meas = 73;
        let ref_bits = [0x0123_4567_89ab_cdef, 0b1_0101_0011];
        for rank in [0, 7, 70] {
            let mut setup = rng(42);
            let flip_rows: Vec<Vec<u64>> = (0..rank)
                .map(|_| vec![setup.next_u64(), setup.next_u64() & 0x7f])
                .collect();
            let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
            for num_shots in [
                0usize, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 65535, 65536, 65537,
            ] {
                let words = num_shots.div_ceil(64);
                let stride = words + 5;
                let mut scalar_rng = rng(42);
                let expected = scalar_samples(&sparse, num_shots, &ref_bits, &mut scalar_rng, rank);
                let mut data = vec![SENTINEL; num_meas * stride];
                let mut actual_rng = rng(42);
                // SAFETY: Each row owns its middle words exclusively; prefix and suffix
                // sentinels are outside the destination and remain initialized.
                let mut output =
                    unsafe { BtsOutput::new(data.as_mut_ptr(), num_meas, stride, 3, words) };
                sample_bts_into(
                    &sparse,
                    num_shots,
                    &ref_bits,
                    &mut actual_rng,
                    rank,
                    &mut output,
                );
                for row in 0..num_meas {
                    let actual = &data[row * stride..(row + 1) * stride];
                    assert_eq!(&actual[..3], &[SENTINEL; 3]);
                    assert_eq!(
                        &actual[3..3 + words],
                        &expected[row * words..(row + 1) * words],
                        "{num_shots} shots, rank {rank}, row {row}"
                    );
                    assert_eq!(&actual[3 + words..], &[SENTINEL; 2]);
                }
                assert_eq!(actual_rng.next_u64(), scalar_rng.next_u64());
            }
        }
    }

    fn check_batched_stream(num_threads: usize) {
        for rank in [0, 7] {
            let num_meas = 9;
            let ref_bits = [0b1_0101_0011];
            let mut setup = rng(42);
            let flip_rows: Vec<Vec<u64>> =
                (0..rank).map(|_| vec![setup.next_u64() & 0x7f]).collect();
            let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
            for num_shots in [
                0usize,
                63,
                65,
                127,
                129,
                255,
                257,
                65535,
                65536,
                65537,
                num_threads * BTS_BATCH_SHOTS + 65,
            ] {
                let words = num_shots.div_ceil(64);
                let mut expected = vec![0; num_meas * words];
                let mut expected_rng = rng(42);
                let mut copy_batches =
                    |start: usize, shots: usize, rng: &mut Xoshiro256PlusPlus| {
                        for done in (0..shots).step_by(BTS_BATCH_SHOTS) {
                            let batch_shots = (shots - done).min(BTS_BATCH_SHOTS);
                            let batch_words = batch_shots.div_ceil(64);
                            let batch = scalar_samples(&sparse, batch_shots, &ref_bits, rng, rank);
                            for row in 0..num_meas {
                                let offset = row * words + (start + done) / 64;
                                expected[offset..offset + batch_words].copy_from_slice(
                                    &batch[row * batch_words..(row + 1) * batch_words],
                                );
                            }
                        }
                    };
                #[cfg(feature = "parallel")]
                {
                    let shots_per_thread = (num_shots.div_ceil(num_threads) / 64) * 64;
                    if num_threads > 1 && shots_per_thread >= 64 {
                        for thread in 0..num_threads {
                            let seed = std::array::from_fn(|_| expected_rng.next_u64());
                            let mut thread_rng = Xoshiro256PlusPlus::from_seeds(seed);
                            let start = thread * shots_per_thread;
                            let end = if thread + 1 == num_threads {
                                num_shots
                            } else {
                                ((thread + 1) * shots_per_thread).min(num_shots)
                            };
                            if start < end {
                                copy_batches(start, end - start, &mut thread_rng);
                            }
                        }
                    } else {
                        copy_batches(0, num_shots, &mut expected_rng);
                    }
                }
                #[cfg(not(feature = "parallel"))]
                copy_batches(0, num_shots, &mut expected_rng);

                let mut actual_rng = rng(42);
                let actual =
                    bts_batched(&sparse, num_shots, words, &ref_bits, &mut actual_rng, rank);
                assert_eq!(
                    actual, expected,
                    "{num_shots} shots, rank {rank}, {num_threads} threads"
                );
                assert_eq!(actual_rng.next_u64(), expected_rng.next_u64());
            }
        }
    }

    #[test]
    fn bts_batched_preserves_batch_and_worker_streams() {
        #[cfg(feature = "parallel")]
        for num_threads in [1, 2, 3, 8] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .unwrap()
                .install(|| check_batched_stream(num_threads));
        }
        #[cfg(not(feature = "parallel"))]
        check_batched_stream(1);
    }
}
