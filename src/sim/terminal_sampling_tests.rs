use super::*;

// Above the dense-outcome cap the direct-key shortcut must not build a
// dense CDF; counts route through the same streaming sampler as the host
// path, so the entry point and the streaming sampler agree byte-for-byte.
// Below the cap the shortcut takes the CDF sampler, which draws from a
// different RNG stream.
#[test]
fn counts_from_probs_stream_above_dense_outcome_cap() {
    let bits = 11;
    let len = 1usize << bits;
    let mut probs = vec![0.0f64; len];
    let spikes = [0usize, 123, len / 3, len / 2 + 7, len - 1];
    for &s in &spikes {
        probs[s] = 0.2;
    }
    let meas_map: Vec<(usize, usize)> = (0..bits).map(|q| (q, q)).collect();
    let map = CompactMeasurementMap::new(&meas_map, bits);
    assert!(map.direct_state_key(len));

    let above_cap = sample_counts_from_probs_capped(&probs, &meas_map, bits, 300, 42, bits - 1);
    let via_streaming = streaming_counts(&probs, |&p| p, &map, bits, 300, 42);
    assert_eq!(above_cap, via_streaming);

    let below_cap = sample_counts_from_probs_capped(&probs, &meas_map, bits, 300, 42, bits);
    let total: u64 = below_cap.values().sum();
    assert_eq!(total, 300);
}

fn merged_counts(
    probs: &[f64],
    sorted: &[f64],
    map: &CompactMeasurementMap,
) -> HashMap<Vec<u64>, u64> {
    merged_cdf_hits(probs, sorted)
        .expect("nonnegative weights")
        .into_iter()
        .map(|(key, count)| (map.packed_key_from_compact(key, 1), count))
        .collect()
}

// The merge must land every uniform where the binary search does, including a
// uniform equal to a cumulative weight that zero-probability bins repeat, a
// weight sum that rounds past 1 before the last bin is pinned, and draws past
// every nonzero bin.
#[test]
fn merged_cdf_counts_match_searched_cdf() {
    let bits = 3;
    let meas_map: Vec<(usize, usize)> = (0..bits).map(|q| (q, q)).collect();
    let map = CompactMeasurementMap::new(&meas_map, bits);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let random: Vec<f64> = (0..64).map(|_| rng.random()).collect();
    let weights: Vec<f64> = (0..8).map(|_| rng.random()).collect();
    let total: f64 = weights.iter().sum();
    let distributions: Vec<Vec<f64>> = vec![
        weights.iter().map(|w| w / total).collect(),
        vec![0.25, 0.0, 0.25, 0.0, 0.5, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
        vec![0.1, 0.2, 0.3, 0.4 + 1e-15, 0.0, 0.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vec![0.125; 8],
    ];
    for probs in &distributions {
        let cdf = build_cdf(probs);
        let mut uniforms = random.clone();
        uniforms.extend(cdf.iter().copied().filter(|&c| c < 1.0));
        uniforms.extend([
            0.0,
            0.25 - f64::EPSILON / 8.0,
            0.5,
            1.0 - f64::EPSILON / 2.0,
        ]);
        uniforms.sort_unstable_by(f64::total_cmp);
        assert_eq!(
            merged_counts(probs, &uniforms, &map),
            searched_cdf_counts(probs, uniforms.iter().copied(), &map, 1),
            "{probs:?}"
        );
    }
    assert!(merged_cdf_hits(&[0.5, -1e-18, 0.5], &[0.5]).is_none());
}

#[test]
fn merged_counts_match_searched_counts_end_to_end() {
    let bits = 16;
    let len = 1usize << bits;
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let mut probs: Vec<f64> = (0..len)
        .map(|i| if i % 5 == 0 { 0.0 } else { rng.random::<f64>() })
        .collect();
    probs[len - 64..].fill(0.0);
    let total: f64 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= total);
    let meas_map: Vec<(usize, usize)> = (0..bits).map(|q| (q, q)).collect();
    let map = CompactMeasurementMap::new(&meas_map, bits);

    for shots in [1_000, 100_000] {
        let counts = sample_counts_from_outcome_distribution(&probs, &map, bits, shots, 42);
        let mut draws = ChaCha8Rng::seed_from_u64(42);
        let searched = searched_cdf_counts(&probs, (0..shots).map(|_| draws.random()), &map, 1);
        assert_eq!(counts, searched, "{shots} shots");
    }
}
