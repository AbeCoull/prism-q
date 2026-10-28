use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const TOL: f64 = 1e-12;

fn random_state(bits: usize, seed: u64) -> Vec<Complex64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state: Vec<Complex64> = (0..1usize << bits)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect();
    let norm = state.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    state.iter_mut().for_each(|a| *a /= norm);
    state
}

fn max_diff(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

#[derive(Clone, Copy, Debug)]
enum Route {
    Gathered,
    Paired,
    Textbook,
}

fn run(state: &[Complex64], num: usize, form: (bool, bool, bool), route: Route) -> Vec<Complex64> {
    let (inverse, swaps, big_endian) = form;
    let mut backend = StatevectorBackend::new(42);
    backend.num_qubits = state.len().ilog2() as usize;
    backend.state = state.to_vec();
    match route {
        Route::Gathered => backend.apply_qft_block_with(num, inverse, swaps, big_endian, true),
        Route::Paired => backend.apply_qft_block_with(num, inverse, swaps, big_endian, false),
        Route::Textbook => backend.apply_qft_block_textbook(0, num, inverse, swaps, big_endian),
    }
    backend.state
}

fn forms() -> impl Iterator<Item = (bool, bool, bool)> {
    (0..8).map(|f| (f & 1 != 0, f & 2 != 0, f & 4 != 0))
}

fn assert_routes_agree(state_bits: usize, num: usize, textbook: bool) {
    let state = random_state(state_bits, 42 + (state_bits * 64 + num) as u64);
    for form in forms() {
        let gathered = run(&state, num, form, Route::Gathered);
        let mut against = vec![Route::Paired];
        if textbook {
            against.push(Route::Textbook);
        }
        for route in against {
            let diff = max_diff(&gathered, &run(&state, num, form, route));
            assert!(
                diff < TOL,
                "{num} of {state_bits} qubits, (inverse, swaps, big_endian) {form:?}: \
                     gathered vs {route:?} differ by {diff:e}"
            );
        }
    }
}

// Reference DIF stages straight from the definition: stage `t` twiddles position
// `k` of its block by `exp(2 pi i k / 2^(t + 1))`, read from the whole-state table.
fn dif_stages(state: &mut [Complex64], n: usize, stages: std::ops::Range<usize>, inverse: bool) {
    let table = qft_twiddles_scaled(n, inverse);
    for chunk in state.chunks_mut(1 << n) {
        for t in stages.clone().rev() {
            let stride = 1usize << t;
            let step = (1usize << n) / (2 * stride);
            run_radix2_stage_seq(chunk, stride, &table, step, std::f64::consts::FRAC_1_SQRT_2);
        }
    }
}

#[test]
fn gathered_passes_match_the_dif_definition_for_any_split() {
    // Splits cover odd and even pass widths, rows clamped to the offsets below a
    // narrow pass, and several 2^n chunks in one state.
    let cases: &[(usize, usize, &[usize])] = &[
        (10, 10, &[3, 3, 3]),
        (12, 12, &[1, 2, 4]),
        (12, 10, &[5, 4]),
        (14, 13, &[2, 7]),
        (16, 16, &[9, 6]),
        (17, 15, &[6]),
    ];
    for &(state_bits, n, split) in cases {
        for inverse in [false, true] {
            let mut expected = random_state(state_bits, 42 + state_bits as u64);
            let mut actual = expected.clone();
            let bottom = n - split.iter().sum::<usize>();
            dif_stages(&mut expected, n, bottom..n, inverse);
            let mut top = n;
            for &bits in split {
                qft_gather_pass(&mut actual, QftGatherPass::new(top - bits, bits, inverse));
                top -= bits;
            }
            let diff = max_diff(&actual, &expected);
            assert!(
                diff < TOL,
                "{n} of {state_bits} qubits, split {split:?}, inverse {inverse}: {diff:e}"
            );
        }
    }
}

// The textbook circuit costs one full pass per gate, so the widest registers check
// against the paired route alone, which the narrower ones tie to the textbook.
#[test]
fn gathered_route_matches_paired_and_textbook_routes() {
    for n in QFT_GATHER_MIN_QUBITS..=22 {
        assert_routes_agree(n, n, n <= 20);
    }
}

#[test]
fn gathered_route_matches_on_sub_register_chunks() {
    for (state_bits, num) in [(19, 16), (20, 18), (21, 19), (22, 20)] {
        assert_routes_agree(state_bits, num, state_bits <= 20);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn row_kernels_match_scalar_butterflies() {
    if !simd::has_avx2_fma() {
        return;
    }
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let rows: Vec<Vec<Complex64>> = (0..4).map(|r| random_state(4, 7 + r)).collect();
    let w = [0.3, 1.7, -2.4].map(|angle| Complex64::from_polar(inv_sqrt2, angle));

    let (mut lo, mut hi) = (rows[0].clone(), rows[1].clone());
    // SAFETY: AVX2 detected above; the rows share one even length.
    unsafe { radix2_rows_avx2fma(&mut lo, &mut hi, w[0]) };
    for j in 0..lo.len() {
        let (elo, ehi) = radix2_butterfly_values(rows[0][j], rows[1][j], w[0], inv_sqrt2);
        assert!((lo[j] - elo).norm() < TOL && (hi[j] - ehi).norm() < TOL);
    }

    let mut q: [Vec<Complex64>; 4] = std::array::from_fn(|r| rows[r].clone());
    let [qa, qb, qc, qd] = q.each_mut().map(Vec::as_mut_slice);
    // SAFETY: AVX2 detected above; the rows share one even length.
    unsafe { radix4_rows_avx2fma(qa, qb, qc, qd, w[0], w[1], w[2]) };
    for j in 0..rows[0].len() {
        let expected = radix4_butterfly_values(
            rows[0][j], rows[1][j], rows[2][j], rows[3][j], w[0], w[1], w[2], inv_sqrt2,
        );
        let expected = [expected.0, expected.1, expected.2, expected.3];
        for (row, e) in q.iter().zip(expected) {
            assert!((row[j] - e).norm() < TOL);
        }
    }

    let phases = qft_gather_phases(16, true);
    let (anchor, step) = (phases.phase(12345), 37);
    let mut out = vec![Complex64::new(0.0, 0.0); rows[0].len()];
    // SAFETY: AVX2 detected above; `out` and the row share one even length.
    unsafe { qft_scatter_row_avx2fma(&mut out, &rows[0], anchor, &phases.low, step) };
    for (j, o) in out.iter().enumerate() {
        let expected = rows[0][j] * (anchor * phases.low[j * step]);
        assert!((o - expected).norm() < TOL);
    }
}

// Row lengths 2 and 6 run the masked two-amplitude step, and the radix-4 quad must
// match two AVX2 pair calls on the same offsets.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_row_kernels_match_avx2() {
    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let w =
        [0.3, 1.7, -2.4].map(|angle| Complex64::from_polar(std::f64::consts::FRAC_1_SQRT_2, angle));
    let phases = qft_gather_phases(16, true);
    for len in [2usize, 4, 6, 8, 10, 16] {
        let rows: Vec<Vec<Complex64>> = (0..4)
            .map(|r| random_state(5, 7 + r)[..len].to_vec())
            .collect();
        let close = |got: &[Complex64], want: &[Complex64], what: &str| {
            assert!(max_diff(got, want) < TOL, "{what} len {len}");
        };

        let (mut want_lo, mut want_hi) = (rows[0].clone(), rows[1].clone());
        let (mut got_lo, mut got_hi) = (rows[0].clone(), rows[1].clone());
        // SAFETY: AVX-512F detected, which implies AVX2 and FMA; the rows share one even
        // length.
        unsafe {
            radix2_rows_avx2fma(&mut want_lo, &mut want_hi, w[0]);
            radix2_rows_avx512(&mut got_lo, &mut got_hi, w[0]);
        }
        close(&got_lo, &want_lo, "radix2 lo");
        close(&got_hi, &want_hi, "radix2 hi");

        let mut want: [Vec<Complex64>; 4] = std::array::from_fn(|r| rows[r].clone());
        let mut got = want.clone();
        let [qa, qb, qc, qd] = want.each_mut().map(Vec::as_mut_slice);
        // SAFETY: as above.
        unsafe { radix4_rows_avx2fma(qa, qb, qc, qd, w[0], w[1], w[2]) };
        let [qa, qb, qc, qd] = got.each_mut().map(Vec::as_mut_slice);
        // SAFETY: as above.
        unsafe { radix4_rows_avx512(qa, qb, qc, qd, w[0], w[1], w[2]) };
        for (g, e) in got.iter().zip(&want) {
            close(g, e, "radix4 rows");
        }

        let (anchor, step) = (phases.phase(12345), 37);
        let mut want = vec![Complex64::new(0.0, 0.0); len];
        let mut got = want.clone();
        // SAFETY: as above; `out` and the row share one even length.
        unsafe {
            qft_scatter_row_avx2fma(&mut want, &rows[0], anchor, &phases.low, step);
            qft_scatter_row_avx512(&mut got, &rows[0], anchor, &phases.low, step);
        }
        close(&got, &want, "scatter");
    }

    let twiddles: Vec<[Complex64; 4]> = (0..3)
        .map(|t| std::array::from_fn(|j| Complex64::from_polar(1.0, 0.4 * (4 * t + j) as f64)))
        .collect();
    let rows: Vec<Vec<Complex64>> = (0..4).map(|r| random_state(2, 11 + r)).collect();
    let mut want: [Vec<Complex64>; 4] = std::array::from_fn(|r| rows[r].clone());
    let mut got = want.clone();
    // SAFETY: AVX-512F detected; every row holds four amplitudes, two AVX2 pairs.
    unsafe {
        let [pa, pb, pc, pd] = want.each_mut().map(|r| r.as_mut_ptr() as *mut f64);
        for j in [0, 2] {
            radix4_butterfly_pair_slices_avx2fma(
                pa.add(2 * j),
                pb.add(2 * j),
                pc.add(2 * j),
                pd.add(2 * j),
                twiddles[0][j],
                twiddles[0][j + 1],
                twiddles[1][j],
                twiddles[1][j + 1],
                twiddles[2][j],
                twiddles[2][j + 1],
            );
        }
        let [pa, pb, pc, pd] = got.each_mut().map(|r| r.as_mut_ptr() as *mut f64);
        radix4_butterfly_quad_avx512(pa, pb, pc, pd, &twiddles[0], &twiddles[1], &twiddles[2]);
    }
    for (g, e) in got.iter().zip(&want) {
        assert!(max_diff(g, e) < TOL, "radix4 quad");
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "256 MiB states, run with --release")]
fn gathered_route_matches_paired_route_at_24_qubits() {
    assert_routes_agree(24, 24, false);
}
