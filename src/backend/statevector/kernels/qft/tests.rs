use super::*;

fn cached_sizes(cache: &RwLock<QftTwiddleCache>) -> Vec<usize> {
    let guard = cache.read().unwrap();
    (0..guard.entries.len())
        .filter(|&n| guard.entries[n].is_some())
        .collect()
}

// A table for size n holds 2^(n-1) entries, so 2^(n+3) bytes: n=1 is 16,
// n=2 is 32, n=3 is 64, n=5 is 256.
#[test]
fn evicts_least_recently_used_over_the_cap() {
    let cache = RwLock::new(QftTwiddleCache::new(96));
    cached_qft_twiddles(&cache, 2);
    cached_qft_twiddles(&cache, 3);
    assert_eq!(cached_sizes(&cache), vec![2, 3]);
    cached_qft_twiddles(&cache, 2);
    cached_qft_twiddles(&cache, 1);
    assert_eq!(cached_sizes(&cache), vec![1, 2]);
    assert_eq!(cache.read().unwrap().bytes, 48);
}

#[test]
fn oversized_table_sits_alone_until_the_next_insert() {
    let cache = RwLock::new(QftTwiddleCache::new(96));
    cached_qft_twiddles(&cache, 2);
    cached_qft_twiddles(&cache, 5);
    assert_eq!(cached_sizes(&cache), vec![5]);
    assert_eq!(cache.read().unwrap().bytes, 256);
    cached_qft_twiddles(&cache, 1);
    assert_eq!(cached_sizes(&cache), vec![1]);
    assert_eq!(cache.read().unwrap().bytes, 16);
}

#[test]
fn concurrent_misses_share_one_table() {
    let cache = RwLock::new(QftTwiddleCache::new(
        QFT_TWIDDLE_CACHE_DEFAULT_LIMIT_MB * 1024 * 1024,
    ));
    let tables: Vec<QftTwiddleTable> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| cached_qft_twiddles(&cache, 6)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert_eq!(cached_sizes(&cache), vec![6]);
    let guard = cache.read().unwrap();
    let cached = &guard.entries[6].as_ref().unwrap().table;
    assert!(tables.iter().all(|t| Arc::ptr_eq(t, cached)));
    assert_eq!(guard.bytes, qft_twiddle_table_bytes(cached.len()));
}

#[test]
fn compact_twiddles_match_whole_state_table_bitwise() {
    // The whole-state table has to reach the compact table's widest stride.
    let n = qft_max_tile_bits().max(16);
    let total = 1usize << n;
    for inverse in [false, true] {
        let whole = qft_twiddles_scaled(n, inverse);
        let compact = qft_compact_twiddles(inverse);
        let mut s = 1;
        while s < compact.len() {
            let step = total / (2 * s);
            for k in 0..s {
                assert_eq!(
                    compact[s + k],
                    whole[k * step],
                    "inverse {inverse}, stride {s}, k {k}"
                );
            }
            s <<= 1;
        }
    }
}

#[test]
fn qft_tiles_leave_paired_stages_and_eight_tiles() {
    let widths: Vec<usize> = (10..=26).map(|n| qft_tile_bits_for(n, 15, 3)).collect();
    assert_eq!(
        widths,
        [
            10, 11, 12, 13, 13, 13, 13, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14
        ]
    );
}

#[test]
fn tiled_bit_reversal_matches_index_reversal() {
    for bits in [3, 9, 10, 11, 12, 15, 16, 17] {
        let mut state: Vec<Complex64> = (0..1usize << bits)
            .map(|i| Complex64::new(i as f64, -(i as f64)))
            .collect();
        apply_bit_reverse_permutation(&mut state, bits);
        for (i, amp) in state.iter().enumerate() {
            let src = reverse_low_bits(i, bits) as f64;
            assert_eq!(*amp, Complex64::new(src, -src), "bits {bits}, index {i}");
        }
    }
}

#[test]
fn qft_block_odd_and_threshold_widths_match_textbook() {
    use crate::test_support::random_state;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let cases = [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (13, 13),
        (14, 14),
        (15, 15),
        (15, 1),
        (15, 2),
        (15, 3),
        (15, 13),
    ];
    for (state_bits, num) in cases {
        let state = random_state(
            state_bits,
            &mut ChaCha8Rng::seed_from_u64(100 + (state_bits * 31 + num) as u64),
        );
        for f in 0..8 {
            let (inverse, swaps, big_endian) = (f & 1 != 0, f & 2 != 0, f & 4 != 0);

            let mut textbook = StatevectorBackend::new(42);
            textbook.num_qubits = state_bits;
            textbook.state = state.clone();
            textbook.apply_qft_block_textbook(0, num, inverse, swaps, big_endian);

            for use_default in [false, true] {
                let mut got = StatevectorBackend::new(42);
                got.num_qubits = state_bits;
                got.state = state.clone();
                if use_default {
                    got.apply_qft_block(num, inverse, swaps, big_endian);
                } else {
                    got.apply_qft_block_with(num, inverse, swaps, big_endian, false);
                }
                let diff = got
                    .state
                    .iter()
                    .zip(&textbook.state)
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0, f64::max);
                assert!(
                    diff < 1e-12,
                    "state_bits {state_bits}, num {num}, form ({inverse}, {swaps}, {big_endian}), \
                     default {use_default}: diff {diff:e}"
                );
            }
        }
    }
}
