use super::parity::{SparseParity, XorDag};
use super::rng::Xoshiro256PlusPlus;
#[cfg(target_arch = "aarch64")]
use super::rng::Xoshiro256PlusPlusX2;
#[cfg(target_arch = "x86_64")]
use super::rng::Xoshiro256PlusPlusX4;

pub(super) const BTS_BATCH_SHOTS: usize = 65536;

pub(super) fn bts_single_pass(
    sparse: &SparseParity,
    xor_dag: Option<&XorDag>,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    if let Some(dag) = xor_dag {
        sample_bts_meas_major_dag(sparse, dag, num_shots, ref_bits, rng, rank)
    } else {
        sample_bts_meas_major(sparse, num_shots, ref_bits, rng, rank)
    }
}

pub(super) fn bts_batched(
    sparse: &SparseParity,
    xor_dag: Option<&XorDag>,
    num_shots: usize,
    total_s_words: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;

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
                        let end = ((t + 1) * shots_per_thread).min(num_shots);
                        (start, end)
                    })
                    .filter(|(s, e)| s < e)
                    .collect();

                let mut output = vec![0u64; num_meas * total_s_words];
                let out_ptr = output.as_mut_ptr();

                {
                    use rayon::prelude::*;
                    let out_slice = &mut output[..];
                    let total_sw = total_s_words;
                    let nm = num_meas;

                    chunks
                        .into_par_iter()
                        .enumerate()
                        .map(|(t, (shot_start, shot_end))| {
                            let chunk_shots = shot_end - shot_start;
                            let word_offset = shot_start / 64;
                            let mut thread_rng = Xoshiro256PlusPlus::from_seeds(thread_seeds[t]);

                            let mut results = Vec::new();
                            let mut chunk_done = 0usize;
                            while chunk_done < chunk_shots {
                                let batch_shots = (chunk_shots - chunk_done).min(BTS_BATCH_SHOTS);
                                let batch_offset = word_offset + chunk_done / 64;
                                let batch_data = sample_bts_meas_major(
                                    sparse,
                                    batch_shots,
                                    ref_bits,
                                    &mut thread_rng,
                                    rank,
                                );
                                results.push((batch_data, batch_shots, batch_offset));
                                chunk_done += batch_shots;
                            }
                            results
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .for_each(|thread_results| {
                            for (batch_data, batch_shots, batch_offset) in thread_results {
                                let batch_s_words = batch_shots.div_ceil(64);
                                for m in 0..nm {
                                    let src =
                                        &batch_data[m * batch_s_words..(m + 1) * batch_s_words];
                                    let dst_start = m * total_sw + batch_offset;
                                    out_slice[dst_start..dst_start + batch_s_words]
                                        .copy_from_slice(src);
                                }
                            }
                        });
                }

                let _ = out_ptr;
                return output;
            }
        }
    }

    let mut output = vec![0u64; num_meas * total_s_words];
    let mut shots_done = 0usize;

    while shots_done < num_shots {
        let batch_shots = (num_shots - shots_done).min(BTS_BATCH_SHOTS);
        let batch_s_words = batch_shots.div_ceil(64);
        let word_offset = shots_done / 64;

        let batch_data = bts_single_pass(sparse, xor_dag, batch_shots, ref_bits, rng, rank);

        for m in 0..num_meas {
            let src = &batch_data[m * batch_s_words..(m + 1) * batch_s_words];
            let dst_start = m * total_s_words + word_offset;
            output[dst_start..dst_start + batch_s_words].copy_from_slice(src);
        }

        shots_done += batch_shots;
    }

    output
}

pub(super) fn sample_bts_meas_major(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && num_shots >= 256 {
            // SAFETY: AVX2 detected, all pointer arithmetic bounded by allocation sizes
            return unsafe { sample_bts_meas_major_avx2(sparse, num_shots, ref_bits, rng, rank) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if num_shots >= 128 {
            // SAFETY: NEON is baseline on aarch64, pointers are valid
            return unsafe { sample_bts_meas_major_neon(sparse, num_shots, ref_bits, rng, rank) };
        }
    }

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_bits = vec![0u64; rank];

    for batch in 0..s_words {
        for r in random_bits.iter_mut().take(rank) {
            *r = rng.next_u64();
        }
        if batch == s_words - 1 {
            let rem = num_shots % 64;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                for r in random_bits.iter_mut().take(rank) {
                    *r &= mask;
                }
            }
        }

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => 0u64,
                1 => random_bits[cols[0] as usize],
                2 => random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize],
                3 => {
                    random_bits[cols[0] as usize]
                        ^ random_bits[cols[1] as usize]
                        ^ random_bits[cols[2] as usize]
                }
                _ => {
                    let mut a = random_bits[cols[0] as usize] ^ random_bits[cols[1] as usize];
                    for &c in &cols[2..] {
                        a ^= random_bits[c as usize];
                    }
                    a
                }
            };
            meas_major[m * s_words + batch] = acc;
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

pub(super) fn apply_ref_bits_meas_major(
    meas_major: &mut [u64],
    ref_bits: &[u64],
    num_meas: usize,
    s_words: usize,
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
}

fn sample_bts_meas_major_dag(
    sparse: &SparseParity,
    dag: &XorDag,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut random_bits = vec![0u64; rank];

    for batch in 0..s_words {
        for r in random_bits.iter_mut().take(rank) {
            *r = rng.next_u64();
        }
        if batch == s_words - 1 {
            let rem = num_shots % 64;
            if rem != 0 {
                let mask = (1u64 << rem) - 1;
                for r in random_bits.iter_mut().take(rank) {
                    *r &= mask;
                }
            }
        }

        for (m, entry) in dag.entries.iter().enumerate() {
            let mut acc = if let Some(p) = entry.parent {
                meas_major[p * s_words + batch]
            } else {
                0u64
            };
            for &c in &entry.residual_cols {
                acc ^= random_bits[c as usize];
            }
            meas_major[m * s_words + batch] = acc;
        }
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

const BTS_QUAD_TILE: usize = 8;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sample_bts_meas_major_avx2(
    sparse: &SparseParity,
    num_shots: usize,
    ref_bits: &[u64],
    rng: &mut Xoshiro256PlusPlus,
    rank: usize,
) -> Vec<u64> {
    use std::arch::x86_64::*;

    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);
    let s_quads = num_shots.div_ceil(256);

    let mut meas_major = vec![0u64; num_meas * s_words];
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

            for m in 0..num_meas {
                let cols = sparse.row_cols(m);
                let out_base = m * s_words + quad_start * 4;

                match cols.len() {
                    0 => {
                        let z = _mm256_setzero_si256();
                        for t in 0..tile {
                            _mm256_storeu_si256(
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                z,
                            );
                        }
                    }
                    1 => {
                        let c0 = cols[0] as usize * tile;
                        for t in 0..tile {
                            _mm256_storeu_si256(
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                random_tile[c0 + t],
                            );
                        }
                    }
                    2 => {
                        let c0 = cols[0] as usize * tile;
                        let c1 = cols[1] as usize * tile;
                        for t in 0..tile {
                            _mm256_storeu_si256(
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
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
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
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
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
                                _mm256_xor_si256(
                                    _mm256_xor_si256(random_tile[c0 + t], random_tile[c1 + t]),
                                    _mm256_xor_si256(random_tile[c2 + t], random_tile[c3 + t]),
                                ),
                            );
                        }
                    }
                    _ => {
                        for t in 0..tile {
                            let mut a = _mm256_xor_si256(
                                random_tile[cols[0] as usize * tile + t],
                                random_tile[cols[1] as usize * tile + t],
                            );
                            for &c in &cols[2..] {
                                a = _mm256_xor_si256(a, random_tile[c as usize * tile + t]);
                            }
                            _mm256_storeu_si256(
                                meas_major[out_base + t * 4..].as_mut_ptr() as *mut __m256i,
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
            &mut meas_major,
            &mut vrng,
            &mut random_tile,
            rank,
            num_meas,
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
            &mut meas_major,
            &mut vrng,
            &mut random_avx,
            rank,
            num_meas,
            s_words,
            s_quads,
            0,
            rem,
        );
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_avx2_remainder(
    sparse: &SparseParity,
    meas_major: &mut [u64],
    vrng: &mut Xoshiro256PlusPlusX4,
    random_tile: &mut [std::arch::x86_64::__m256i],
    rank: usize,
    num_meas: usize,
    s_words: usize,
    s_quads: usize,
    quad_start: usize,
    tile: usize,
    rem: usize,
) {
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

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => _mm256_setzero_si256(),
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
                _ => {
                    let mut a = _mm256_xor_si256(
                        random_tile[cols[0] as usize * tile],
                        random_tile[cols[1] as usize * tile],
                    );
                    for &c in &cols[2..] {
                        a = _mm256_xor_si256(a, random_tile[c as usize * tile]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_avx2_per_quad(
    sparse: &SparseParity,
    meas_major: &mut [u64],
    vrng: &mut Xoshiro256PlusPlusX4,
    random_avx: &mut [std::arch::x86_64::__m256i],
    rank: usize,
    num_meas: usize,
    s_words: usize,
    s_quads: usize,
    start_quad: usize,
    rem: usize,
) {
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

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => _mm256_setzero_si256(),
                1 => random_avx[cols[0] as usize],
                2 => _mm256_xor_si256(random_avx[cols[0] as usize], random_avx[cols[1] as usize]),
                3 => _mm256_xor_si256(
                    _mm256_xor_si256(random_avx[cols[0] as usize], random_avx[cols[1] as usize]),
                    random_avx[cols[2] as usize],
                ),
                4 => _mm256_xor_si256(
                    _mm256_xor_si256(random_avx[cols[0] as usize], random_avx[cols[1] as usize]),
                    _mm256_xor_si256(random_avx[cols[2] as usize], random_avx[cols[3] as usize]),
                ),
                _ => {
                    let mut a = _mm256_xor_si256(
                        random_avx[cols[0] as usize],
                        random_avx[cols[1] as usize],
                    );
                    for &c in &cols[2..] {
                        a = _mm256_xor_si256(a, random_avx[c as usize]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
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

#[cfg(not(target_arch = "x86_64"))]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_avx2(
    _sparse: &SparseParity,
    _num_shots: usize,
    _ref_bits: &[u64],
    _rng: &mut Xoshiro256PlusPlus,
    _rank: usize,
) -> Vec<u64> {
    unreachable!()
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
) -> Vec<u64> {
    use std::arch::aarch64::*;

    let num_meas = sparse.num_rows;
    let s_words = num_shots.div_ceil(64);
    let s_pairs = num_shots.div_ceil(128);

    let mut meas_major = vec![0u64; num_meas * s_words];
    let mut vrng = Xoshiro256PlusPlusX2::from_scalar(rng);

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
                for r in 0..rank {
                    random_tile[r * tile + t] = vrng.next_uint64x2();
                }
            }

            for m in 0..num_meas {
                let cols = sparse.row_cols(m);
                let out_base = m * s_words + pair_start * 2;

                match cols.len() {
                    0 => {
                        let z = vdupq_n_u64(0);
                        for t in 0..tile {
                            vst1q_u64(meas_major[out_base + t * 2..].as_mut_ptr(), z);
                        }
                    }
                    1 => {
                        let c0 = cols[0] as usize * tile;
                        for t in 0..tile {
                            vst1q_u64(
                                meas_major[out_base + t * 2..].as_mut_ptr(),
                                random_tile[c0 + t],
                            );
                        }
                    }
                    2 => {
                        let c0 = cols[0] as usize * tile;
                        let c1 = cols[1] as usize * tile;
                        for t in 0..tile {
                            vst1q_u64(
                                meas_major[out_base + t * 2..].as_mut_ptr(),
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
                                meas_major[out_base + t * 2..].as_mut_ptr(),
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
                                meas_major[out_base + t * 2..].as_mut_ptr(),
                                veorq_u64(
                                    veorq_u64(random_tile[c0 + t], random_tile[c1 + t]),
                                    veorq_u64(random_tile[c2 + t], random_tile[c3 + t]),
                                ),
                            );
                        }
                    }
                    _ => {
                        for t in 0..tile {
                            let mut a = veorq_u64(
                                random_tile[cols[0] as usize * tile + t],
                                random_tile[cols[1] as usize * tile + t],
                            );
                            for &c in &cols[2..] {
                                a = veorq_u64(a, random_tile[c as usize * tile + t]);
                            }
                            vst1q_u64(meas_major[out_base + t * 2..].as_mut_ptr(), a);
                        }
                    }
                }
            }

            pair_start += tile;
        }

        bts_neon_remainder(
            sparse,
            &mut meas_major,
            &mut vrng,
            rank,
            num_meas,
            s_words,
            s_pairs,
            pair_start,
            rem,
        );
    } else {
        bts_neon_per_pair(
            sparse,
            &mut meas_major,
            &mut vrng,
            rank,
            num_meas,
            s_words,
            s_pairs,
            0,
            rem,
        );
    }

    apply_ref_bits_meas_major(&mut meas_major, ref_bits, num_meas, s_words);
    meas_major
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_neon_remainder(
    sparse: &SparseParity,
    meas_major: &mut [u64],
    vrng: &mut Xoshiro256PlusPlusX2,
    rank: usize,
    num_meas: usize,
    s_words: usize,
    s_pairs: usize,
    pair_start: usize,
    rem: usize,
) {
    use std::arch::aarch64::*;

    let mut random_neon: Vec<uint64x2_t> = vec![vdupq_n_u64(0); rank];

    for pair in pair_start..s_pairs {
        let base_sw = pair * 2;
        let words_this_pair = (s_words - base_sw).min(2);

        for nval in random_neon.iter_mut().take(rank) {
            *nval = vrng.next_uint64x2();
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

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => vdupq_n_u64(0),
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
                _ => {
                    let mut a =
                        veorq_u64(random_neon[cols[0] as usize], random_neon[cols[1] as usize]);
                    for &c in &cols[2..] {
                        a = veorq_u64(a, random_neon[c as usize]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
            if words_this_pair == 2 {
                vst1q_u64(out_ptr, acc);
            } else {
                *out_ptr = vgetq_lane_u64(acc, 0);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn bts_neon_per_pair(
    sparse: &SparseParity,
    meas_major: &mut [u64],
    vrng: &mut Xoshiro256PlusPlusX2,
    rank: usize,
    num_meas: usize,
    s_words: usize,
    s_pairs: usize,
    start_pair: usize,
    rem: usize,
) {
    use std::arch::aarch64::*;

    let mut random_neon: Vec<uint64x2_t> = vec![vdupq_n_u64(0); rank];

    for pair in start_pair..s_pairs {
        let base_sw = pair * 2;
        let words_this_pair = (s_words - base_sw).min(2);

        for nval in random_neon.iter_mut().take(rank) {
            *nval = vrng.next_uint64x2();
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

        for m in 0..num_meas {
            let cols = sparse.row_cols(m);
            let acc = match cols.len() {
                0 => vdupq_n_u64(0),
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
                _ => {
                    let mut a =
                        veorq_u64(random_neon[cols[0] as usize], random_neon[cols[1] as usize]);
                    for &c in &cols[2..] {
                        a = veorq_u64(a, random_neon[c as usize]);
                    }
                    a
                }
            };

            let out_ptr = meas_major[m * s_words + base_sw..].as_mut_ptr();
            if words_this_pair == 2 {
                vst1q_u64(out_ptr, acc);
            } else {
                *out_ptr = vgetq_lane_u64(acc, 0);
            }
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
unsafe fn sample_bts_meas_major_neon(
    _sparse: &SparseParity,
    _num_shots: usize,
    _ref_bits: &[u64],
    _rng: &mut Xoshiro256PlusPlus,
    _rank: usize,
) -> Vec<u64> {
    unreachable!()
}
