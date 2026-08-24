use super::*;
use std::f64::consts::FRAC_1_SQRT_2;

const EPS: f64 = 1e-12;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn assert_complex_close(a: Complex64, b: Complex64) {
    assert!(
        (a.re - b.re).abs() < EPS && (a.im - b.im).abs() < EPS,
        "expected ({}, {}i), got ({}, {}i)",
        b.re,
        b.im,
        a.re,
        a.im,
    );
}

fn identity() -> [[Complex64; 2]; 2] {
    [[c(1.0, 0.0), c(0.0, 0.0)], [c(0.0, 0.0), c(1.0, 0.0)]]
}

fn x_gate() -> [[Complex64; 2]; 2] {
    [[c(0.0, 0.0), c(1.0, 0.0)], [c(1.0, 0.0), c(0.0, 0.0)]]
}

fn h_gate() -> [[Complex64; 2]; 2] {
    let s = FRAC_1_SQRT_2;
    [[c(s, 0.0), c(s, 0.0)], [c(s, 0.0), c(-s, 0.0)]]
}

#[test]
fn test_identity_preserves_state() {
    let mut lo = vec![c(0.6, 0.2), c(0.1, -0.3)];
    let mut hi = vec![c(0.4, -0.1), c(-0.5, 0.7)];
    let lo_orig = lo.clone();
    let hi_orig = hi.clone();
    let prepared = PreparedGate1q::new(&identity());
    prepared.apply(&mut lo, &mut hi);
    for (a, b) in lo.iter().zip(lo_orig.iter()) {
        assert_complex_close(*a, *b);
    }
    for (a, b) in hi.iter().zip(hi_orig.iter()) {
        assert_complex_close(*a, *b);
    }
}

#[test]
fn test_x_gate_swaps() {
    let mut lo = vec![c(1.0, 0.0)];
    let mut hi = vec![c(0.0, 0.0)];
    let prepared = PreparedGate1q::new(&x_gate());
    prepared.apply(&mut lo, &mut hi);
    assert_complex_close(lo[0], c(0.0, 0.0));
    assert_complex_close(hi[0], c(1.0, 0.0));
}

#[test]
fn test_h_gate_creates_superposition() {
    let mut lo = vec![c(1.0, 0.0)];
    let mut hi = vec![c(0.0, 0.0)];
    let prepared = PreparedGate1q::new(&h_gate());
    prepared.apply(&mut lo, &mut hi);
    assert_complex_close(lo[0], c(FRAC_1_SQRT_2, 0.0));
    assert_complex_close(hi[0], c(FRAC_1_SQRT_2, 0.0));
}

#[test]
fn test_multi_element_slices() {
    let mut lo = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.5, 0.5), c(0.0, 0.0)];
    let mut hi = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5, -0.5)];
    let mat = h_gate();

    let mut lo_ref = lo.clone();
    let mut hi_ref = hi.clone();
    apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

    let prepared = PreparedGate1q::new(&mat);
    prepared.apply(&mut lo, &mut hi);

    for i in 0..lo.len() {
        assert_complex_close(lo[i], lo_ref[i]);
        assert_complex_close(hi[i], hi_ref[i]);
    }
}

#[test]
fn test_complex_valued_matrix() {
    let mat = [[c(0.0, 1.0), c(0.5, -0.5)], [c(0.5, 0.5), c(0.0, -1.0)]];
    let mut lo = vec![c(1.0, 0.0), c(0.0, 1.0)];
    let mut hi = vec![c(0.0, 0.0), c(1.0, 0.0)];

    let mut lo_ref = lo.clone();
    let mut hi_ref = hi.clone();
    apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

    let prepared = PreparedGate1q::new(&mat);
    prepared.apply(&mut lo, &mut hi);

    for i in 0..lo.len() {
        assert_complex_close(lo[i], lo_ref[i]);
        assert_complex_close(hi[i], hi_ref[i]);
    }
}

#[test]
fn test_odd_length_slices() {
    let mut lo = vec![c(1.0, 0.0), c(0.5, 0.5), c(0.0, 1.0)];
    let mut hi = vec![c(0.0, 0.0), c(0.3, -0.2), c(0.7, 0.1)];
    let mat = h_gate();

    let mut lo_ref = lo.clone();
    let mut hi_ref = hi.clone();
    apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

    let prepared = PreparedGate1q::new(&mat);
    prepared.apply(&mut lo, &mut hi);

    for i in 0..lo.len() {
        assert_complex_close(lo[i], lo_ref[i]);
        assert_complex_close(hi[i], hi_ref[i]);
    }
}

#[test]
fn test_bulk_negate() {
    let mut slice = vec![c(1.0, 2.0), c(-3.0, 0.5), c(0.0, -1.0)];
    let expected = [c(-1.0, -2.0), c(3.0, -0.5), c(0.0, 1.0)];
    negate_slice(&mut slice);
    for (a, e) in slice.iter().zip(expected.iter()) {
        assert_complex_close(*a, *e);
    }
}

#[test]
fn test_bulk_swap() {
    let mut a = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
    let mut b = vec![c(4.0, 0.0), c(5.0, 0.0), c(6.0, 0.0)];
    swap_slices(&mut a, &mut b);
    assert_complex_close(a[0], c(4.0, 0.0));
    assert_complex_close(b[0], c(1.0, 0.0));
    assert_complex_close(a[2], c(6.0, 0.0));
    assert_complex_close(b[2], c(3.0, 0.0));
}

#[test]
fn test_bulk_norm_sqr_sum() {
    let slice = vec![c(3.0, 4.0), c(1.0, 0.0), c(0.0, 2.0)];
    let result = norm_sqr_sum(&slice);
    let expected = 25.0 + 1.0 + 4.0;
    assert!((result - expected).abs() < EPS);
}

#[test]
fn test_bulk_zero() {
    let mut slice = vec![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)];
    zero_slice(&mut slice);
    for amp in &slice {
        assert_complex_close(*amp, c(0.0, 0.0));
    }
}

#[test]
fn test_bulk_scale() {
    let mut slice = vec![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 0.0)];
    scale_slice(&mut slice, 2.0);
    assert_complex_close(slice[0], c(2.0, 4.0));
    assert_complex_close(slice[1], c(6.0, 8.0));
    assert_complex_close(slice[2], c(10.0, 0.0));
}

#[test]
fn test_scale_complex_slice() {
    let phase = c(0.0, 1.0);
    let mut slice = vec![c(1.0, 0.0), c(0.0, 1.0), c(3.0, 4.0)];
    scale_complex_slice(&mut slice, phase);
    assert_complex_close(slice[0], c(0.0, 1.0));
    assert_complex_close(slice[1], c(-1.0, 0.0));
    assert_complex_close(slice[2], c(-4.0, 3.0));
}

#[test]
fn test_scale_complex_slice_phase() {
    let phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let mut slice = vec![c(1.0, 0.0), c(1.0, 0.0), c(0.0, 1.0), c(0.5, -0.3)];
    let expected: Vec<Complex64> = slice.iter().map(|&v| v * phase).collect();
    scale_complex_slice(&mut slice, phase);
    for (a, e) in slice.iter().zip(expected.iter()) {
        assert_complex_close(*a, *e);
    }
}

#[test]
fn test_scale_complex_slice_single_element() {
    let phase = c(0.0, -1.0);
    let mut slice = vec![c(2.0, 3.0)];
    let expected = slice[0] * phase;
    scale_complex_slice(&mut slice, phase);
    assert_complex_close(slice[0], expected);
}

#[test]
fn test_scale_complex_to_slice_lengths() {
    let factor = Complex64::from_polar(1.3, 0.7);
    for len in [1usize, 2, 3, 4, 5, 7, 8, 16, 17, 33] {
        let src: Vec<Complex64> = (0..len)
            .map(|i| c((i as f64) + 0.25, (i as f64) * 0.5 - 1.0))
            .collect();
        let mut dst = vec![c(99.0, 99.0); len];
        scale_complex_to_slice(&mut dst, &src, factor);
        for i in 0..len {
            assert_complex_close(dst[i], src[i] * factor);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "distributed"))]
#[test]
fn combine_global_half_x86_tiers_match_scalar() {
    let c_self = c(0.3, -0.4);
    let c_remote = c(-0.2, 0.7);
    for len in [1usize, 2, 3, 4, 5, 7, 16, 17, 33] {
        let dst0: Vec<Complex64> = (0..len)
            .map(|i| c((i as f64) * 0.13 - 0.4, 0.2 - (i as f64) * 0.07))
            .collect();
        let remote: Vec<Complex64> = (0..len)
            .map(|i| c(0.5 - (i as f64) * 0.11, (i as f64) * 0.03 + 0.1))
            .collect();
        let mut expected = dst0.clone();
        combine_global_half_scalar(&mut expected, &remote, c_self, c_remote);

        if has_fma() {
            let mut actual = dst0.clone();
            // SAFETY: FMA checked above, and the slices have equal length.
            unsafe { combine_global_half_fma(&mut actual, &remote, c_self, c_remote) };
            assert_state_close(&actual, &expected, &format!("fma len={len}"));
        }

        if has_avx2_fma() {
            let mut actual = dst0.clone();
            // SAFETY: AVX2 and FMA checked above, and the slices have equal length.
            unsafe { combine_global_half_avx2fma(&mut actual, &remote, c_self, c_remote) };
            assert_state_close(&actual, &expected, &format!("avx2fma len={len}"));
        }

        let mut actual = dst0.clone();
        combine_global_half(&mut actual, &remote, c_self, c_remote);
        assert_state_close(&actual, &expected, &format!("dispatch len={len}"));
    }
}

fn identity_4x4() -> [[Complex64; 4]; 4] {
    let z = c(0.0, 0.0);
    let o = c(1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, o]]
}

fn cx_4x4() -> [[Complex64; 4]; 4] {
    let z = c(0.0, 0.0);
    let o = c(1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
}

fn cz_4x4() -> [[Complex64; 4]; 4] {
    let z = c(0.0, 0.0);
    let o = c(1.0, 0.0);
    let m = c(-1.0, 0.0);
    [[o, z, z, z], [z, o, z, z], [z, z, o, z], [z, z, z, m]]
}

fn apply_2q_reference(state: &mut [Complex64], mat: &[[Complex64; 4]; 4], q0: usize, q1: usize) {
    let mask0 = 1usize << q0;
    let mask1 = 1usize << q1;
    let (lo, hi) = if q0 < q1 { (q0, q1) } else { (q1, q0) };
    let n = state.len();
    let n_iter = n >> 2;
    for k in 0..n_iter {
        let idx = crate::backend::statevector::insert_zero_bit(
            crate::backend::statevector::insert_zero_bit(k, lo),
            hi,
        );
        let i = [idx, idx | mask1, idx | mask0, idx | mask0 | mask1];
        let a = [state[i[0]], state[i[1]], state[i[2]], state[i[3]]];
        for (r, &ii) in i.iter().enumerate() {
            state[ii] = mat[r][0] * a[0] + mat[r][1] * a[1] + mat[r][2] * a[2] + mat[r][3] * a[3];
        }
    }
}

#[test]
fn test_prepared_2q_identity() {
    let mut state = vec![c(0.5, 0.1), c(0.3, -0.2), c(-0.1, 0.4), c(0.6, -0.3)];
    let orig = state.clone();
    let prepared = PreparedGate2q::new(&identity_4x4());
    prepared.apply_full(&mut state, 2, 0, 1);
    for (a, e) in state.iter().zip(orig.iter()) {
        assert_complex_close(*a, *e);
    }
}

#[test]
fn test_prepared_2q_cx_on_11() {
    // |11⟩ → CX → |10⟩ (target q1 flips when control q0=1)
    let mut state = vec![c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    let mut ref_state = state.clone();
    let mat = cx_4x4();
    let prepared = PreparedGate2q::new(&mat);
    prepared.apply_full(&mut state, 2, 0, 1);
    apply_2q_reference(&mut ref_state, &mat, 0, 1);
    for (a, e) in state.iter().zip(ref_state.iter()) {
        assert_complex_close(*a, *e);
    }
}

#[test]
fn test_prepared_2q_cz_matches_reference() {
    let mut state = vec![c(0.5, 0.0), c(0.3, 0.1), c(-0.2, 0.4), c(0.6, -0.1)];
    let mut ref_state = state.clone();
    let mat = cz_4x4();
    let prepared = PreparedGate2q::new(&mat);
    prepared.apply_full(&mut state, 2, 0, 1);
    apply_2q_reference(&mut ref_state, &mat, 0, 1);
    for (a, e) in state.iter().zip(ref_state.iter()) {
        assert_complex_close(*a, *e);
    }
}

#[test]
fn test_prepared_2q_3qubit_system() {
    // 3-qubit system: apply CX on q0, q2 (non-adjacent)
    let mut state = vec![c(0.0, 0.0); 8];
    state[0] = c(FRAC_1_SQRT_2, 0.0);
    state[5] = c(FRAC_1_SQRT_2, 0.0); // |101⟩
    let mut ref_state = state.clone();
    let mat = cx_4x4();
    let prepared = PreparedGate2q::new(&mat);
    prepared.apply_full(&mut state, 3, 0, 2);
    apply_2q_reference(&mut ref_state, &mat, 0, 2);
    for (i, (a, e)) in state.iter().zip(ref_state.iter()).enumerate() {
        assert!((a - e).norm() < EPS, "state[{i}]: expected {e}, got {a}");
    }
}

/// A dense, asymmetric 4×4 matrix that exercises every coefficient slot.
/// Built from a non-special unitary so any indexing bug in the AVX2
/// paired-group kernel surfaces as a numerical mismatch.
fn dense_4x4() -> [[Complex64; 4]; 4] {
    let s = FRAC_1_SQRT_2;
    let h2 = [
        [c(0.5, 0.0), c(0.5, 0.0), c(0.5, 0.0), c(0.5, 0.0)],
        [c(0.5, 0.0), c(-0.5, 0.0), c(0.5, 0.0), c(-0.5, 0.0)],
        [c(0.5, 0.0), c(0.5, 0.0), c(-0.5, 0.0), c(-0.5, 0.0)],
        [c(0.5, 0.0), c(-0.5, 0.0), c(-0.5, 0.0), c(0.5, 0.0)],
    ];
    let phase = [
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(s, s), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 1.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-s, s)],
    ];
    let mut out = [[c(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for col in 0..4 {
            let mut acc = c(0.0, 0.0);
            for k in 0..4 {
                acc += phase[r][k] * h2[k][col];
            }
            out[r][col] = acc;
        }
    }
    out
}

fn random_state(num_qubits: usize, seed: u64) -> Vec<Complex64> {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = 1usize << num_qubits;
    let mut s = Vec::with_capacity(n);
    let mut norm = 0.0;
    for _ in 0..n {
        let re: f64 = rng.random_range(-1.0..1.0);
        let im: f64 = rng.random_range(-1.0..1.0);
        norm += re * re + im * im;
        s.push(c(re, im));
    }
    let inv = norm.sqrt().recip();
    for v in &mut s {
        v.re *= inv;
        v.im *= inv;
    }
    s
}

fn assert_state_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let d = (*a - *e).norm();
        assert!(
            d < 1e-10,
            "{label} state[{i}]: expected {e}, got {a} (diff {d:.2e})"
        );
    }
}

// Reference test: AVX2 paired-group kernel must agree with the 128-bit
// FMA per-group kernel across (q0, q1) configurations covering adjacent,
// non-adjacent, reversed-order, and the lo == 0 fallback path.
#[test]
fn test_prepared_2q_apply_tiled_matches_apply_full() {
    let mat = dense_4x4();
    let configs: &[(usize, usize, usize)] = &[
        (4, 0, 1),  // adjacent, lo == 0 (forces 128-bit fallback inside apply_tiled)
        (4, 1, 0),  // reversed, lo == 0
        (4, 1, 2),  // adjacent, lo > 0 (AVX2 path)
        (4, 2, 1),  // reversed, lo > 0
        (5, 0, 4),  // far apart, lo == 0
        (5, 4, 0),  // reversed, lo == 0
        (5, 1, 4),  // far apart, lo > 0
        (5, 4, 1),  // reversed, lo > 0
        (6, 2, 5),  // mid-range, lo > 0
        (8, 0, 7),  // 8-qubit, span entire register, lo == 0
        (8, 1, 7),  // 8-qubit, span entire register, lo > 0
        (8, 7, 1),  // reversed
        (10, 3, 6), // 10-qubit AVX2 path
    ];

    for &(nq, q0, q1) in configs {
        let state_init = random_state(nq, 0xCAFE_F00D);
        let prepared = PreparedGate2q::new(&mat);

        let mut via_full = state_init.clone();
        prepared.apply_full(&mut via_full, nq, q0, q1);

        let mut via_tiled = state_init.clone();
        prepared.apply_tiled(&mut via_tiled, nq, q0, q1);

        assert_state_close(
            &via_tiled,
            &via_full,
            &format!("nq={nq} q0={q0} q1={q1} apply_tiled vs apply_full"),
        );
    }
}

// PreparedKraus2q must agree with scalar evaluation of the same 16x16
// superoperator on scattered block slots, at a contiguous and a strided slot
// layout. Runs only where AVX2 and FMA are detected; the dispatcher never
// selects it elsewhere.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_prepared_kraus_2q_matches_scalar_reference() {
    if !has_avx2_fma() {
        return;
    }
    use rand::RngExt;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut s = [[c(0.0, 0.0); 16]; 16];
    for row in s.iter_mut() {
        for e in row.iter_mut() {
            *e = c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        }
    }

    let flats_variants: [[usize; 16]; 2] =
        [std::array::from_fn(|j| j), std::array::from_fn(|j| 3 * j)];
    for flats in &flats_variants {
        let len = flats[15] + 1;
        let mut buf: Vec<Complex64> = (0..len)
            .map(|_| c(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        let mut reference = buf.clone();

        let v: [Complex64; 16] = std::array::from_fn(|j| reference[flats[j]]);
        for (r, row) in s.iter().enumerate() {
            let mut acc = c(0.0, 0.0);
            for (coeff, val) in row.iter().zip(v.iter()) {
                acc += coeff * val;
            }
            reference[flats[r]] = acc;
        }

        // SAFETY: AVX2 and FMA checked above.
        let prepared = unsafe { PreparedKraus2q::new(&s) };
        // SAFETY: AVX2 and FMA checked above; every slot index is in bounds
        // and no other thread touches the buffer.
        unsafe { prepared.apply_block_ptr(buf.as_mut_ptr() as *mut f64, 0, flats) };
        assert_state_close(&buf, &reference, &format!("stride {}", flats[1]));
    }
}

// Drive every SIMD tier the host supports against the scalar reference. The
// dispatcher runs only the best tier, so SSE2/FMA never execute on an AVX2 host
// without forcing them here.

#[cfg(target_arch = "x86_64")]
fn available_tiers() -> Vec<SimdTier> {
    let mut tiers = vec![SimdTier::Sse2];
    if has_fma() {
        tiers.push(SimdTier::Fma);
    }
    if has_avx2_fma() {
        tiers.push(SimdTier::Avx2Fma);
    }
    tiers
}

#[cfg(target_arch = "x86_64")]
fn tier_name(tier: &SimdTier) -> &'static str {
    match tier {
        SimdTier::Avx2Fma => "avx2fma",
        SimdTier::Fma => "fma",
        SimdTier::Sse2 => "sse2",
    }
}

/// All four entries distinct, so an indexing or sign bug cannot cancel out.
#[cfg(target_arch = "x86_64")]
fn asymmetric_gate() -> [[Complex64; 2]; 2] {
    [[c(0.3, 0.1), c(-0.2, 0.4)], [c(0.5, -0.3), c(0.1, 0.2)]]
}

#[cfg(target_arch = "x86_64")]
#[test]
fn apply_slices_every_tier_matches_scalar() {
    let mat = asymmetric_gate();
    // Length deliberately not a multiple of the widest tier's stride so the
    // scalar remainder loop is exercised too.
    let lo0: Vec<Complex64> = (0..37)
        .map(|i| c(0.11 * i as f64, -0.07 * i as f64))
        .collect();
    let hi0: Vec<Complex64> = (0..37)
        .map(|i| c(-0.05 * i as f64, 0.09 * i as f64))
        .collect();

    let mut lo_ref = lo0.clone();
    let mut hi_ref = hi0.clone();
    apply_slices_scalar(&mut lo_ref, &mut hi_ref, &mat);

    for tier in available_tiers() {
        let mut prepared = PreparedGate1q::new(&mat);
        let name = tier_name(&tier);
        prepared.tier = tier;
        let mut lo = lo0.clone();
        let mut hi = hi0.clone();
        prepared.apply(&mut lo, &mut hi);
        for i in 0..lo.len() {
            assert!(
                (lo[i] - lo_ref[i]).norm() < EPS && (hi[i] - hi_ref[i]).norm() < EPS,
                "tier {name} diverged from scalar at index {i}"
            );
        }
    }
}

/// Scalar reference for a full-state single-qubit apply.
#[cfg(target_arch = "x86_64")]
fn scalar_full_apply(state: &mut [Complex64], target: usize, mat: &[[Complex64; 2]; 2]) {
    let half = 1usize << target;
    let mut base = 0usize;
    while base < state.len() {
        for k in base..base + half {
            let v0 = state[k];
            let v1 = state[k + half];
            state[k] = mat[0][0] * v0 + mat[0][1] * v1;
            state[k + half] = mat[1][0] * v0 + mat[1][1] * v1;
        }
        base += half * 2;
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn apply_full_sequential_every_tier_matches_scalar() {
    let mat = asymmetric_gate();
    // Cover the AVX2 size/target guard boundary (target >= 2 and small state
    // takes the 256-bit path; otherwise the 128-bit fallback).
    for nq in [3usize, 5, 8] {
        let dim = 1usize << nq;
        let state0: Vec<Complex64> = (0..dim)
            .map(|i| c(0.013 * i as f64 + 0.1, -0.017 * i as f64))
            .collect();
        for target in 0..nq {
            let mut reference = state0.clone();
            scalar_full_apply(&mut reference, target, &mat);

            for tier in available_tiers() {
                let mut prepared = PreparedGate1q::new(&mat);
                let name = tier_name(&tier);
                prepared.tier = tier;
                let mut state = state0.clone();
                prepared.apply_full_sequential(&mut state, target);
                for i in 0..dim {
                    assert!(
                        (state[i] - reference[i]).norm() < EPS,
                        "tier {name} nq={nq} target={target} diverged at index {i}"
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn norm_sqr_sum_avx2fma_matches_scalar() {
    if !has_avx2_fma() {
        return;
    }
    let slice: Vec<Complex64> = (0..101)
        .map(|i| c(0.07 * i as f64 - 1.0, 0.03 * i as f64 + 0.2))
        .collect();
    let scalar: f64 = slice.iter().map(|z| z.norm_sqr()).sum();
    // SAFETY: guarded by has_avx2_fma() above.
    let simd = unsafe { norm_sqr_sum_avx2fma(&slice) };
    assert!(
        (simd - scalar).abs() < 1e-9,
        "avx2fma norm_sqr_sum={simd} vs scalar={scalar}"
    );
}

// Completeness tripwire: pin the count of `#[target_feature]` kernels so adding
// one without a paired per-tier test fails here. Reads source text only, so it
// runs on every arch.
#[test]
fn target_feature_kernel_count_is_pinned() {
    // Split the needle so this test's own source does not self-count.
    let needle = concat!("target_feature", "(enable");
    let root = env!("CARGO_MANIFEST_DIR");
    // Per-file expected counts; the breakdown makes a drift easy to localise.
    let expected: [(&str, usize); 4] = [
        ("src/backend/simd.rs", 30),
        ("src/backend/word_ops.rs", 2),
        ("src/backend/stabilizer/kernels/simd.rs", 3),
        ("src/backend/statevector/kernels.rs", 11),
    ];
    for (file, want) in expected {
        let path = format!("{root}/{file}");
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let got = text.matches(needle).count();
        assert_eq!(
            got, want,
            "{file} has {got} `#[target_feature]` kernels, pinned at {want}. \
             If you added or removed a tiered kernel, add/adjust its per-tier \
             equivalence test and update this pin."
        );
    }
}
