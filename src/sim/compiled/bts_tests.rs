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
        let flip_rows: Vec<Vec<u64>> = (0..rank).map(|_| vec![setup.next_u64() & mask]).collect();
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

// The AVX-512 kernel regroups the four-lane stream into eight-word columns, so it must
// match the portable loop bitwise across partial octets, partial tiles (rank 17 tiles
// four octets, rank 300 one), and every column count the row XOR specialises.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_path_draws_the_same_stream() {
    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let num_meas = 40;
    for rank in [0, 1, 2, 3, 5, 17, 70, 300] {
        let mut setup = rng(11 + rank as u64);
        let mask = (1u64 << num_meas) - 1;
        let flip_rows: Vec<Vec<u64>> = (0..rank).map(|_| vec![setup.next_u64() & mask]).collect();
        let sparse = SparseParity::from_flip_rows(&flip_rows, num_meas);
        let ref_bits = vec![setup.next_u64() & mask];
        for num_shots in [512, 513, 576, 1000, 1024, 1536, 2048 + 64, 4672, 5000] {
            let mut scalar_rng = rng(99);
            let mut vector_rng = rng(99);
            let scalar = scalar_samples(&sparse, num_shots, &ref_bits, &mut scalar_rng, rank);
            let words = num_shots.div_ceil(64);
            let mut vector = vec![0; num_meas * words];
            let mut output = BtsOutput::contiguous(&mut vector, num_meas, words);
            // SAFETY: AVX-512F detected.
            unsafe {
                sample_bts_meas_major_avx512(
                    &sparse,
                    num_shots,
                    &ref_bits,
                    &mut vector_rng,
                    rank,
                    &mut output,
                );
            }
            assert_eq!(scalar, vector, "{num_shots} shots, rank {rank}");
            assert_eq!(scalar_rng.next_u64(), vector_rng.next_u64());
        }
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
        let flip_rows: Vec<Vec<u64>> = (0..rank).map(|_| vec![setup.next_u64() & 0x7f]).collect();
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
            let mut copy_batches = |start: usize, shots: usize, rng: &mut Xoshiro256PlusPlus| {
                for done in (0..shots).step_by(BTS_BATCH_SHOTS) {
                    let batch_shots = (shots - done).min(BTS_BATCH_SHOTS);
                    let batch_words = batch_shots.div_ceil(64);
                    let batch = scalar_samples(&sparse, batch_shots, &ref_bits, rng, rank);
                    for row in 0..num_meas {
                        let offset = row * words + (start + done) / 64;
                        expected[offset..offset + batch_words]
                            .copy_from_slice(&batch[row * batch_words..(row + 1) * batch_words]);
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
            let actual = bts_batched(&sparse, num_shots, words, &ref_bits, &mut actual_rng, rank);
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
