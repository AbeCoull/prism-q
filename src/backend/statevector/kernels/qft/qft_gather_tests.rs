use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::test_support::random_state;

const TOL: f64 = 1e-12;

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
    let state = random_state(
        state_bits,
        &mut ChaCha8Rng::seed_from_u64(42 + (state_bits * 64 + num) as u64),
    );
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
            run_radix2_stage_seq(chunk, stride, &table, step);
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
            let mut expected = random_state(
                state_bits,
                &mut ChaCha8Rng::seed_from_u64(42 + state_bits as u64),
            );
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
    let rows: Vec<Vec<Complex64>> = (0..4)
        .map(|r| random_state(4, &mut ChaCha8Rng::seed_from_u64(7 + r)))
        .collect();
    let w = [0.3, 1.7, -2.4].map(|angle| Complex64::from_polar(INV_SQRT2, angle));

    let (mut lo, mut hi) = (rows[0].clone(), rows[1].clone());
    // SAFETY: AVX2 detected above; the rows share one even length.
    unsafe { radix2_rows_avx2fma(&mut lo, &mut hi, w[0]) };
    for j in 0..lo.len() {
        let (elo, ehi) = radix2_butterfly_values(rows[0][j], rows[1][j], w[0]);
        assert!((lo[j] - elo).norm() < TOL && (hi[j] - ehi).norm() < TOL);
    }

    let mut q: [Vec<Complex64>; 4] = std::array::from_fn(|r| rows[r].clone());
    let [qa, qb, qc, qd] = q.each_mut().map(Vec::as_mut_slice);
    // SAFETY: AVX2 detected above; the rows share one even length.
    unsafe { radix4_rows_avx2fma(qa, qb, qc, qd, w[0], w[1], w[2]) };
    for j in 0..rows[0].len() {
        let expected = radix4_butterfly_values(
            rows[0][j], rows[1][j], rows[2][j], rows[3][j], w[0], w[1], w[2],
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
            .map(|r| random_state(5, &mut ChaCha8Rng::seed_from_u64(7 + r))[..len].to_vec())
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
    let rows: Vec<Vec<Complex64>> = (0..4)
        .map(|r| random_state(2, &mut ChaCha8Rng::seed_from_u64(11 + r)))
        .collect();
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

// Every stage pair from inner stride four up, through the AVX-512 groups and through the
// AVX2 pair kernel at the same offsets, compared bit for bit. In-tile pairs read the
// compact table at steps 1 and 2; whole-state pairs read the full table at outer steps
// from 1 to 2^(n - 4), which puts the shuffled step 2 on the outer twiddles too.
#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_radix4_groups_match_avx2_bitwise() {
    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let bits = |v: &[Complex64]| -> Vec<[u64; 2]> {
        v.iter().map(|a| [a.re.to_bits(), a.im.to_bits()]).collect()
    };
    for n in [4usize, 7, 10, 12] {
        let fft_len = 1usize << n;
        for inverse in [false, true] {
            let whole = qft_twiddles_scaled(n, inverse);
            let compact = qft_compact_twiddles(inverse);
            let mut want = random_state(n, &mut ChaCha8Rng::seed_from_u64(42 + n as u64));
            let mut got = want.clone();
            for top in (4..=n).rev() {
                let s = 1usize << (top - 2);
                let groups = fft_len / (4 * s);
                for (table, step_outer, step_inner) in [
                    (&compact[2 * s..4 * s], 1, 2),
                    (&whole[..], fft_len / (4 * s), fft_len / (2 * s)),
                ] {
                    let (pg, pw) = (got.as_mut_ptr() as *mut f64, want.as_mut_ptr() as *mut f64);
                    // SAFETY: AVX-512F detected, which implies AVX2 and FMA. Both states hold
                    // `groups` radix-4 groups of stride `s`, a power of two of at least four,
                    // and every twiddle index stays below `table.len()`.
                    unsafe {
                        radix4_groups_avx512(pg, groups, s, 0, table, step_outer, step_inner);
                        for g in 0..groups {
                            for k in (0..s).step_by(2) {
                                radix4_butterfly_pair_avx2fma(
                                    pw.add((g * 4 * s + k) * 2),
                                    s,
                                    table[k * step_outer],
                                    table[(k + 1) * step_outer],
                                    table[(k + s) * step_outer],
                                    table[(k + s + 1) * step_outer],
                                    table[k * step_inner],
                                    table[(k + 1) * step_inner],
                                );
                            }
                        }
                    }
                    assert!(
                        bits(&got) == bits(&want),
                        "n {n}, s {s}, steps {step_outer}/{step_inner}, inverse {inverse}"
                    );
                }
            }
        }
    }
}

fn radix2_stage_scalar(
    state: &mut [Complex64],
    stride: usize,
    twiddles_scaled: &[Complex64],
    twiddle_step: usize,
) {
    let block_size = stride << 1;
    let total = state.len();
    let mut group_start = 0;
    while group_start < total {
        for k in 0..stride {
            let i0 = group_start + k;
            let i1 = i0 + stride;
            let w = twiddles_scaled[k * twiddle_step];
            let (lo, hi) = radix2_butterfly_values(state[i0], state[i1], w);
            state[i0] = lo;
            state[i1] = hi;
        }
        group_start += block_size;
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_radix4_groups_match_reference_bitwise() {
    if !simd::has_avx2_fma() {
        return;
    }
    let bits = |v: &[Complex64]| -> Vec<[u64; 2]> {
        v.iter().map(|a| [a.re.to_bits(), a.im.to_bits()]).collect()
    };
    for n in [3usize, 4, 7, 10] {
        let fft_len = 1usize << n;
        for inverse in [false, true] {
            let whole = qft_twiddles_scaled(n, inverse);
            let compact = qft_compact_twiddles(inverse);
            let mut want_pair = random_state(n, &mut ChaCha8Rng::seed_from_u64(42 + n as u64));
            let mut want_scalar = want_pair.clone();
            let mut got = want_pair.clone();
            for top in (3..=n).rev() {
                let s = 1usize << (top - 2);
                let groups = fft_len / (4 * s);
                for (table, step_outer, step_inner) in [
                    (&compact[2 * s..4 * s], 1, 2),
                    (&whole[..], fft_len / (4 * s), fft_len / (2 * s)),
                ] {
                    let (pg, pw) = (
                        got.as_mut_ptr() as *mut f64,
                        want_pair.as_mut_ptr() as *mut f64,
                    );
                    // SAFETY: AVX2 and FMA detected above. Both states hold `groups` radix-4
                    // groups of stride `s >= 2`, and twiddle indices stay below `table.len()`.
                    unsafe {
                        radix4_groups_avx2fma(pg, groups, s, 0, table, step_outer, step_inner);
                        for g in 0..groups {
                            for k in (0..s).step_by(2) {
                                radix4_butterfly_pair_avx2fma(
                                    pw.add((g * 4 * s + k) * 2),
                                    s,
                                    table[k * step_outer],
                                    table[(k + 1) * step_outer],
                                    table[(k + s) * step_outer],
                                    table[(k + s + 1) * step_outer],
                                    table[k * step_inner],
                                    table[(k + 1) * step_inner],
                                );
                            }
                        }
                    }
                    for g in 0..groups {
                        let base = g * 4 * s;
                        for k in 0..s {
                            radix4_butterfly_scalar(
                                &mut want_scalar,
                                base + k,
                                s,
                                table[k * step_outer],
                                table[(k + s) * step_outer],
                                table[k * step_inner],
                            );
                        }
                    }
                    assert!(
                        bits(&got) == bits(&want_pair),
                        "bitwise mismatch: n {n}, s {s}, steps {step_outer}/{step_inner}, inverse {inverse}"
                    );
                    assert!(
                        max_diff(&got, &want_scalar) < TOL,
                        "scalar diff: n {n}, s {s}, steps {step_outer}/{step_inner}, inverse {inverse}"
                    );
                }
            }
        }
    }

    // Exercise k_base > 0 with step_outer == 1 and step_outer > 1.
    let n = 6usize;
    let s = 4usize;
    let groups = 2usize;
    let k_base = 2usize;
    let whole = qft_twiddles_scaled(n, false);
    for (step_outer, step_inner) in [(1usize, 2usize), (2usize, 4usize)] {
        let mut got = random_state(5, &mut ChaCha8Rng::seed_from_u64(314));
        let mut want = got.clone();
        // SAFETY: AVX2 and FMA detected above; `got` holds 2 groups of `4 * 4 = 16` amplitudes
        // and `whole` has 32 entries, exceeding `(2 + 7) * 2` and `(2 + 3) * 4`.
        unsafe {
            radix4_groups_avx2fma(
                got.as_mut_ptr() as *mut f64,
                groups,
                s,
                k_base,
                &whole,
                step_outer,
                step_inner,
            );
        }
        for g in 0..groups {
            let base = g * 4 * s;
            for k_offset in 0..s {
                let k = k_base + k_offset;
                radix4_butterfly_scalar(
                    &mut want,
                    base + k_offset,
                    s,
                    whole[k * step_outer],
                    whole[(k + s) * step_outer],
                    whole[k * step_inner],
                );
            }
        }
        assert!(max_diff(&got, &want) < TOL);
    }
}

#[test]
fn apply_radix4_groups_and_runs_match_scalar() {
    for s in [1usize, 2, 4, 8] {
        let groups = 3usize;
        let len = groups * 4 * s;
        for inverse in [false, true] {
            let whole = qft_twiddles_scaled(6, inverse);
            for (step_outer, step_inner) in [(1usize, 2usize), (2usize, 4usize)] {
                let mut got: Vec<Complex64> =
                    random_state(7, &mut ChaCha8Rng::seed_from_u64(99 + s as u64))[..len].to_vec();
                let mut want = got.clone();
                apply_radix4_groups(&mut got, s, 0, step_outer, step_inner, &whole);
                for g in 0..groups {
                    let base = g * 4 * s;
                    for k in 0..s {
                        radix4_butterfly_scalar(
                            &mut want,
                            base + k,
                            s,
                            whole[k * step_outer],
                            whole[(k + s) * step_outer],
                            whole[k * step_inner],
                        );
                    }
                }
                assert!(max_diff(&got, &want) < TOL, "s {s}, inverse {inverse}");
            }
        }
    }

    #[cfg(feature = "parallel")]
    {
        for len in [1usize, 2, 3, 4, 5, 7, 16] {
            let w0 = random_state(4, &mut ChaCha8Rng::seed_from_u64(501))[..len].to_vec();
            let w1 = random_state(4, &mut ChaCha8Rng::seed_from_u64(502))[..len].to_vec();
            let w2 = random_state(4, &mut ChaCha8Rng::seed_from_u64(503))[..len].to_vec();
            let mut got: [Vec<Complex64>; 4] = std::array::from_fn(|r| {
                random_state(4, &mut ChaCha8Rng::seed_from_u64(510 + r as u64))[..len].to_vec()
            });
            let mut want = got.clone();
            {
                let [qa, qb, qc, qd] = got.each_mut().map(Vec::as_mut_slice);
                radix4_runs(qa, qb, qc, qd, &w0, &w1, &w2);
            }
            {
                let [qa, qb, qc, qd] = want.each_mut().map(Vec::as_mut_slice);
                for j in 0..len {
                    radix4_butterfly_quartet_scalar(qa, qb, qc, qd, j, w0[j], w1[j], w2[j]);
                }
            }
            for (g, e) in got.iter().zip(&want) {
                assert!(max_diff(g, e) < TOL, "radix4_runs len {len}");
            }
        }
    }
}

#[test]
fn radix2_stages_match_scalar_across_strides_and_steps() {
    for inverse in [false, true] {
        let table = qft_twiddles_scaled(8, inverse);
        for stride in [1usize, 2, 4, 8, 16] {
            for groups in [1usize, 2, 3, 4, 5] {
                let len = groups * 2 * stride;
                for step in [1usize, 2, 5] {
                    let mut got = random_state(8, &mut ChaCha8Rng::seed_from_u64(700 + len as u64))
                        [..len]
                        .to_vec();
                    let mut want = got.clone();
                    run_radix2_stage_seq(&mut got, stride, &table, step);
                    radix2_stage_scalar(&mut want, stride, &table, step);
                    assert!(
                        max_diff(&got, &want) < TOL,
                        "stride {stride}, groups {groups}, step {step}, inverse {inverse}"
                    );
                }
            }
        }

        for pairs in [1usize, 2, 3, 4, 5, 6, 7, 8, 15] {
            for k_base in [0usize, 3] {
                for step in [1usize, 2, 5] {
                    let mut got_lo =
                        random_state(4, &mut ChaCha8Rng::seed_from_u64(801))[..pairs].to_vec();
                    let mut got_hi =
                        random_state(4, &mut ChaCha8Rng::seed_from_u64(802))[..pairs].to_vec();
                    let mut want_lo = got_lo.clone();
                    let mut want_hi = got_hi.clone();
                    apply_radix2_span(&mut got_lo, &mut got_hi, k_base, step, &table);
                    for j in 0..pairs {
                        let w = table[(k_base + j) * step];
                        let (lo, hi) = radix2_butterfly_values(want_lo[j], want_hi[j], w);
                        want_lo[j] = lo;
                        want_hi[j] = hi;
                    }
                    assert!(
                        max_diff(&got_lo, &want_lo) < TOL && max_diff(&got_hi, &want_hi) < TOL,
                        "span pairs {pairs}, k_base {k_base}, step {step}, inverse {inverse}"
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx512_radix2_stages_match_avx2_bitwise() {
    if !crate::backend::word_ops::avx512_test_host() {
        return;
    }
    let bits = |v: &[Complex64]| -> Vec<[u64; 2]> {
        v.iter().map(|a| [a.re.to_bits(), a.im.to_bits()]).collect()
    };
    let table = qft_twiddles_scaled(8, true);

    for groups in [4usize, 8] {
        let mut got = random_state(4, &mut ChaCha8Rng::seed_from_u64(901))[..groups * 2].to_vec();
        let mut want = got.clone();
        // SAFETY: AVX-512F detected above, which implies AVX2 and FMA; `got` and `want`
        // hold `groups` pairs.
        unsafe {
            radix2_stride1_avx512(got.as_mut_ptr() as *mut f64, groups, table[0]);
            radix2_stride1_avx2fma(want.as_mut_ptr() as *mut f64, 0, groups, table[0]);
        }
        assert_eq!(bits(&got), bits(&want), "stride1 groups {groups}");
    }

    for step in [1usize, 2, 5] {
        for groups in [2usize, 4, 6] {
            let mut got =
                random_state(5, &mut ChaCha8Rng::seed_from_u64(902))[..groups * 4].to_vec();
            let mut want = got.clone();
            // SAFETY: AVX-512F detected above; `got` and `want` hold `groups` 4-element groups.
            unsafe {
                radix2_stride2_avx512(got.as_mut_ptr() as *mut f64, groups, table[0], table[step]);
                radix2_stride2_avx2fma(
                    want.as_mut_ptr() as *mut f64,
                    0,
                    groups,
                    table[0],
                    table[step],
                );
            }
            assert_eq!(
                bits(&got),
                bits(&want),
                "stride2 groups {groups}, step {step}"
            );
        }

        for stride in [4usize, 8, 16] {
            let groups = 3usize;
            let mut got = random_state(7, &mut ChaCha8Rng::seed_from_u64(903))
                [..groups * 2 * stride]
                .to_vec();
            let mut want = got.clone();
            // SAFETY: AVX-512F detected above; both slices hold `groups` blocks of `2 * stride`.
            unsafe {
                radix2_groups_avx512(got.as_mut_ptr() as *mut f64, groups, stride, &table, step);
                radix2_groups_avx2fma(want.as_mut_ptr() as *mut f64, groups, stride, &table, step);
            }
            assert_eq!(bits(&got), bits(&want), "stride {stride}, step {step}");
        }

        for pairs in [4usize, 8, 12] {
            for k_base in [0usize, 4] {
                let mut got_lo =
                    random_state(4, &mut ChaCha8Rng::seed_from_u64(904))[..pairs].to_vec();
                let mut got_hi =
                    random_state(4, &mut ChaCha8Rng::seed_from_u64(905))[..pairs].to_vec();
                let mut want_lo = got_lo.clone();
                let mut want_hi = got_hi.clone();
                // SAFETY: AVX-512F detected above; `pairs` is a multiple of 4 and twiddle bounds
                // fit within `table.len() == 128`.
                unsafe {
                    radix2_span_avx512(
                        got_lo.as_mut_ptr() as *mut f64,
                        got_hi.as_mut_ptr() as *mut f64,
                        pairs,
                        k_base,
                        table.as_ptr(),
                        step,
                    );
                    radix2_span_avx2fma(
                        want_lo.as_mut_ptr() as *mut f64,
                        want_hi.as_mut_ptr() as *mut f64,
                        0,
                        pairs,
                        k_base,
                        table.as_ptr(),
                        step,
                    );
                }
                assert_eq!(bits(&got_lo), bits(&want_lo), "span lo pairs {pairs}");
                assert_eq!(bits(&got_hi), bits(&want_hi), "span hi pairs {pairs}");
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn fft_stage_par_all_arms_match_sequential() {
    let n = 14usize;
    let total = 1usize << n;
    let base = random_state(n, &mut ChaCha8Rng::seed_from_u64(1234));
    for inverse in [false, true] {
        let table = qft_twiddles_scaled(n, inverse);
        for stride in [1usize, 2, 4, 1024, 2048, 4096, 8192] {
            let block_size = stride << 1;
            let twiddle_step = total / block_size;
            let mut par = base.clone();
            let mut seq = base.clone();
            let mut scalar = base.clone();
            fft_stage_par(&mut par, stride, block_size, twiddle_step, &table);
            run_radix2_stage_seq(&mut seq, stride, &table, twiddle_step);
            radix2_stage_scalar(&mut scalar, stride, &table, twiddle_step);
            assert!(
                max_diff(&par, &seq) < TOL && max_diff(&par, &scalar) < TOL,
                "stride {stride}, inverse {inverse}"
            );
        }
    }
}

#[test]
fn qft_gather_pass_reuses_thread_local_scratch_across_sizes() {
    // Sequential path (state_bits < kernel_parallel_qubits()): shrink then grow `len`,
    // poisoning the scratch buffer before smaller and larger passes.
    let seq_bits = 12usize;
    let mut actual_seq = random_state(seq_bits, &mut ChaCha8Rng::seed_from_u64(2025));
    let mut expected_seq = actual_seq.clone();

    let poison = Complex64::new(999.0, -999.0);
    qft_gather_pass(&mut actual_seq, QftGatherPass::new(8, 4, false));
    dif_stages(&mut expected_seq, 12, 8..12, false);

    with_qft_gather_buf(4096, |buf| buf.fill(poison));
    qft_gather_pass(&mut actual_seq, QftGatherPass::new(4, 2, true));
    dif_stages(&mut expected_seq, 6, 4..6, true);

    with_qft_gather_buf(64, |buf| buf.fill(poison));
    qft_gather_pass(&mut actual_seq, QftGatherPass::new(9, 3, true));
    dif_stages(&mut expected_seq, 12, 9..12, true);

    assert!(
        max_diff(&actual_seq, &expected_seq) < TOL,
        "sequential gather scratch reuse mismatch"
    );

    #[cfg(feature = "parallel")]
    {
        // Parallel path (state_bits >= kernel_parallel_qubits()): large -> tiny -> large.
        let par_bits = 16usize;
        let mut actual_par = random_state(par_bits, &mut ChaCha8Rng::seed_from_u64(2026));
        let mut expected_par = actual_par.clone();

        qft_gather_pass(&mut actual_par, QftGatherPass::new(10, 6, false));
        dif_stages(&mut expected_par, 16, 10..16, false);

        qft_gather_pass(&mut actual_par, QftGatherPass::new(3, 2, true));
        dif_stages(&mut expected_par, 5, 3..5, true);

        qft_gather_pass(&mut actual_par, QftGatherPass::new(9, 7, false));
        dif_stages(&mut expected_par, 16, 9..16, false);

        assert!(
            max_diff(&actual_par, &expected_par) < TOL,
            "parallel gather scratch reuse mismatch"
        );
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "256 MiB states, run with --release")]
fn gathered_route_matches_paired_route_at_24_qubits() {
    assert_routes_agree(24, 24, false);
}
