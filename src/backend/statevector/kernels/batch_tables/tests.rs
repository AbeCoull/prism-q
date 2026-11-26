use super::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const NUM_QUBITS: usize = 12;

// The scalar and BMI2 kernels index their tables differently (shift loop vs
// PEXT over a mask), so a table shape that indexes one bit per unique qubit
// silently diverges from a producer that emits a repeated qubit. Both batch
// families are checked against the same amplitude for every basis index.
#[test]
fn batch_phase_pext_matches_scalar_lookup() {
    if !std::is_x86_feature_detected!("bmi2") {
        panic!("bmi2 is required to check the PEXT path against the scalar path");
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let one = Complex64::new(1.0, 0.0);

    for case in 0..64 {
        let len = 1 + case % BatchPhaseData::MAX_PHASES.min(NUM_QUBITS - 1);
        let mut targets: Vec<usize> = (1..NUM_QUBITS).collect();
        for i in (1..targets.len()).rev() {
            targets.swap(i, rng.random_range(0..=i));
        }
        let phases: Vec<(usize, Complex64)> = targets[..len]
            .iter()
            .map(|&q| (q, Complex64::from_polar(1.0, rng.random_range(-3.0..3.0))))
            .collect();

        let mut groups = [BatchPhaseGroup {
            table: [one; BATCH_PHASE_TABLE_SIZE],
            shifts: [0; BATCH_PHASE_GROUP_SIZE],
            len: 0,
            pext_mask: 0,
        }; MAX_BATCH_PHASE_GROUPS];
        let num_groups = build_batch_phase_tables(&phases, &mut groups);

        for idx in 0..(1usize << NUM_QUBITS) {
            let scalar = batch_phase_at(idx, &groups, num_groups);
            let mut pext = one;
            for group in groups.iter().take(num_groups) {
                // SAFETY: BMI2 checked above.
                let bits = unsafe { std::arch::x86_64::_pext_u64(idx as u64, group.pext_mask) };
                pext *= group.table[bits as usize];
            }
            assert!(
                (scalar - pext).norm() < 1e-12,
                "case {case} idx {idx}: scalar {scalar} vs pext {pext}"
            );
        }
    }
}

// Pair-aware grouping can leave every group short of
// `DIAG_BATCH_MAX_QUBITS_PER_GROUP`, so the scalar path can no longer split
// `unique_qubits` at a fixed stride. Two six-qubit chains pin that shape: they
// cannot share a group, and neither fills one.
#[test]
fn diagonal_batch_pext_matches_scalar_lookup_with_short_groups() {
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("bmi2")
        || !std::is_x86_feature_detected!("fma")
    {
        panic!("avx2, bmi2, and fma are required to check the PEXT path against the scalar path");
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut entries = Vec::new();
    for chain in [0usize, 6] {
        for q in chain..chain + 5 {
            entries.push(DiagEntry::Phase2q {
                q0: q,
                q1: q + 1,
                phase: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
            });
        }
        entries.push(DiagEntry::Parity2q {
            q0: chain,
            q1: chain + 3,
            same: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
            diff: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
        });
        entries.push(DiagEntry::Phase1q {
            qubit: chain + 2,
            d0: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
            d1: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
        });
    }

    let built = build_diagonal_batch_tables(&entries).expect("groupable");
    assert_eq!(built.num_groups, 2);
    assert_eq!(built.group_sizes[..2], [6, 6]);

    let dim = 1usize << NUM_QUBITS;
    let mut scalar: Vec<Complex64> = (0..dim)
        .map(|i| Complex64::from_polar(1.0, i as f64 * 0.017))
        .collect();
    let mut pext = scalar.clone();
    let shifts = diag_batch_group_shifts(&built.unique_qubits, &built.group_sizes);
    diagonal_batch_tile_scalar(
        &mut scalar,
        0,
        &built.tables,
        &shifts,
        &built.group_sizes,
        built.num_groups,
    );
    // SAFETY: BMI2 and FMA checked above.
    unsafe {
        diagonal_batch_tile_bmi2(
            &mut pext,
            0,
            &built.tables,
            &built.group_pext_masks,
            built.num_groups,
        );
    }
    for (idx, (a, b)) in scalar.iter().zip(&pext).enumerate() {
        assert!((a - b).norm() < 1e-12, "idx {idx}: scalar {a} vs pext {b}");
    }
}

// A 24-qubit ring plus four pairs of a matching: runs of eight consecutive
// edges would put the pairs in a sixteen-qubit group, past the PEXT table.
#[test]
fn batch_rzz_pext_packs_a_ring_with_a_matching() {
    if !std::is_x86_feature_detected!("bmi2") {
        panic!("bmi2 is required to check the PEXT path against the scalar path");
    }
    let one = Complex64::new(1.0, 0.0);
    let mut edges: Vec<(usize, usize, f64)> = (0..24)
        .map(|q| (q, (q + 1) % 24, 0.3 + 0.01 * q as f64))
        .collect();
    edges.extend([(3, 17, 0.7), (5, 11, -0.4), (20, 8, 1.1), (1, 14, 0.2)]);
    edges.extend([(6, 22, 0.9), (9, 2, -1.3), (13, 19, 0.5), (4, 15, 0.8)]);

    let mut bmi2_groups = [BatchRzzBmi2Group {
        table: [one; BATCH_RZZ_BMI2_TABLE_SIZE],
        pext_mask: 0,
    }; BATCH_RZZ_BMI2_MAX_GROUPS];
    let num_groups = build_batch_rzz_bmi2_tables(&edges, &mut bmi2_groups)
        .expect("the ring and matching fit the PEXT groups");

    for idx in (0..1usize << 24).step_by(9973) {
        let mut angle = 0.0;
        for &(q0, q1, theta) in &edges {
            let parity = ((idx >> q0) ^ (idx >> q1)) & 1;
            angle += if parity == 0 {
                -theta / 2.0
            } else {
                theta / 2.0
            };
        }
        let expected = Complex64::from_polar(1.0, angle);
        let mut pext = one;
        for group in bmi2_groups.iter().take(num_groups) {
            // SAFETY: BMI2 checked above.
            let bits = unsafe { std::arch::x86_64::_pext_u64(idx as u64, group.pext_mask) };
            pext *= group.table[bits as usize];
        }
        assert!(
            (expected - pext).norm() < 1e-12,
            "idx {idx}: {expected} vs {pext}"
        );
    }
}

#[test]
fn batch_rzz_pext_matches_scalar_lookup() {
    if !std::is_x86_feature_detected!("bmi2") {
        panic!("bmi2 is required to check the PEXT path against the scalar path");
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let one = Complex64::new(1.0, 0.0);

    for case in 0..64 {
        let num_edges = 1 + case % BatchRzzData::MAX_EDGES;
        let edges: Vec<(usize, usize, f64)> = (0..num_edges)
            .map(|_| {
                let q0 = rng.random_range(0..NUM_QUBITS);
                let mut q1 = rng.random_range(0..NUM_QUBITS);
                while q1 == q0 {
                    q1 = rng.random_range(0..NUM_QUBITS);
                }
                (q0, q1, rng.random_range(-3.0..3.0))
            })
            .collect();

        let mut scalar_groups = [BatchRzzGroup {
            table: [one; BATCH_RZZ_TABLE_SIZE],
            q0s: [0; BATCH_RZZ_GROUP_SIZE],
            q1s: [0; BATCH_RZZ_GROUP_SIZE],
            len: 0,
        }; MAX_BATCH_RZZ_GROUPS];
        let num_groups = build_batch_rzz_tables(&edges, &mut scalar_groups);

        let mut bmi2_groups = [BatchRzzBmi2Group {
            table: [one; BATCH_RZZ_BMI2_TABLE_SIZE],
            pext_mask: 0,
        }; BATCH_RZZ_BMI2_MAX_GROUPS];
        let Some(bmi2_num_groups) = build_batch_rzz_bmi2_tables(&edges, &mut bmi2_groups) else {
            continue;
        };

        for idx in 0..(1usize << NUM_QUBITS) {
            let mut scalar = one;
            for group in scalar_groups.iter().take(num_groups) {
                scalar *= group.table[extract_rzz_bits(idx, group)];
            }
            let mut pext = one;
            for group in bmi2_groups.iter().take(bmi2_num_groups) {
                // SAFETY: BMI2 checked above.
                let bits = unsafe { std::arch::x86_64::_pext_u64(idx as u64, group.pext_mask) };
                pext *= group.table[bits as usize];
            }
            assert!(
                (scalar - pext).norm() < 1e-12,
                "case {case} idx {idx}: scalar {scalar} vs pext {pext}"
            );
        }
    }
}

fn sample_tile(len: usize, seed_offset: usize) -> Vec<Complex64> {
    (0..len)
        .map(|i| {
            let x = (i + seed_offset) as f64;
            Complex64::new((x * 0.13).cos() * 0.7 + 0.1, (x * 0.07).sin() * 0.5 - 0.2)
        })
        .collect()
}

fn assert_tile_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (idx, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*a - *b).norm() < 1e-12,
            "{label} idx {idx}: got {a} vs expected {b}"
        );
    }
}

#[test]
fn batch_rzz_tile_bmi2_matches_scalar_across_groups_and_lengths() {
    if !simd::has_avx2_fma() || !simd::has_bmi2() {
        return;
    }
    type RzzEdgeCase = (usize, Vec<(usize, usize, f64)>);
    let edge_sets: Vec<RzzEdgeCase> = vec![
        (0, vec![]),
        (1, vec![(0, 1, 0.4), (1, 2, -0.7), (2, 3, 1.1)]),
        (
            2,
            (0..6)
                .map(|i| (2 * i, 2 * i + 1, 0.25 + 0.1 * i as f64))
                .collect(),
        ),
        (
            4,
            (0..20)
                .map(|i| (2 * i, 2 * i + 1, -0.5 + 0.07 * i as f64))
                .collect(),
        ),
    ];

    for (expected_groups, edges) in edge_sets {
        let mut bmi2_groups = [BatchRzzBmi2Group::EMPTY; BATCH_RZZ_BMI2_MAX_GROUPS];
        let num_groups = build_batch_rzz_bmi2_tables(&edges, &mut bmi2_groups).expect("fits");
        assert_eq!(num_groups, expected_groups);

        for base_idx in [0usize, 1, 5, 16] {
            for len in [0usize, 1, 2, 3, 7, 64] {
                let tile0 = sample_tile(len, base_idx);
                let mut expected = tile0.clone();
                for (j, amp) in expected.iter_mut().enumerate() {
                    let idx = base_idx + j;
                    let mut angle = 0.0;
                    for &(q0, q1, theta) in &edges {
                        let parity = ((idx >> q0) ^ (idx >> q1)) & 1;
                        angle += if parity == 0 {
                            -theta / 2.0
                        } else {
                            theta / 2.0
                        };
                    }
                    *amp *= Complex64::from_polar(1.0, angle);
                }

                let mut actual = tile0;
                // SAFETY: AVX2, BMI2, and FMA detected above.
                unsafe { batch_rzz_tile_bmi2(&mut actual, base_idx, &bmi2_groups, num_groups) };
                assert_tile_close(
                    &actual,
                    &expected,
                    &format!("rzz groups={num_groups} base={base_idx} len={len}"),
                );
            }
        }
    }
}

#[test]
fn diagonal_batch_tile_bmi2_matches_scalar_edge_cases() {
    if !simd::has_avx2_fma() || !simd::has_bmi2() {
        return;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut group_cases: Vec<(usize, Vec<DiagEntry>)> = vec![(0, vec![])];

    for num_chains in [1usize, 2, 4] {
        let mut entries = Vec::new();
        for c in 0..num_chains {
            let start = c * 10;
            for q in start..start + 9 {
                entries.push(DiagEntry::Phase2q {
                    q0: q,
                    q1: q + 1,
                    phase: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
                });
            }
            entries.push(DiagEntry::Phase1q {
                qubit: start,
                d0: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
                d1: Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
            });
        }
        group_cases.push((num_chains, entries));
    }

    for (expected_groups, entries) in group_cases {
        let built = build_diagonal_batch_tables(&entries).expect("groupable");
        assert_eq!(built.num_groups, expected_groups);
        let shifts = diag_batch_group_shifts(&built.unique_qubits, &built.group_sizes);

        for base_idx in [0usize, 1, 5, 16] {
            for len in [0usize, 1, 2, 3, 17, 64] {
                let tile0 = sample_tile(len, base_idx);
                let mut expected = tile0.clone();
                diagonal_batch_tile_scalar(
                    &mut expected,
                    base_idx,
                    &built.tables,
                    &shifts,
                    &built.group_sizes,
                    built.num_groups,
                );

                let mut actual = tile0;
                // SAFETY: AVX2, BMI2, and FMA detected above.
                unsafe {
                    diagonal_batch_tile_bmi2(
                        &mut actual,
                        base_idx,
                        &built.tables,
                        &built.group_pext_masks,
                        built.num_groups,
                    );
                }
                assert_tile_close(
                    &actual,
                    &expected,
                    &format!("diag groups={} base={base_idx} len={len}", built.num_groups),
                );
            }
        }
    }
}

#[test]
fn apply_batch_phase_bmi2_and_tile_match_scalar() {
    if !simd::has_avx2_fma() || !simd::has_bmi2() {
        return;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let phase_cases: Vec<(usize, Vec<(usize, Complex64)>)> = vec![
        (0, vec![]),
        (
            1,
            (0..5)
                .map(|q| (q + 1, Complex64::from_polar(1.0, 0.3 * (q + 1) as f64)))
                .collect(),
        ),
        (
            2,
            (0..12)
                .map(|q| {
                    (
                        q + 1,
                        Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
                    )
                })
                .collect(),
        ),
        (
            4,
            (0..35)
                .map(|q| {
                    (
                        q + 1,
                        Complex64::from_polar(1.0, rng.random_range(-3.0..3.0)),
                    )
                })
                .collect(),
        ),
    ];

    for (expected_groups, phases) in phase_cases {
        let mut groups = [BatchPhaseGroup::EMPTY; MAX_BATCH_PHASE_GROUPS];
        let num_groups = build_batch_phase_tables(&phases, &mut groups);
        assert_eq!(num_groups, expected_groups);

        for nq in [1usize, 2, 3, 6, 12] {
            for ctrl_bit in [0usize, 1, 5] {
                if ctrl_bit >= nq {
                    continue;
                }
                let state0 = sample_tile(1usize << nq, ctrl_bit);
                let mut expected = state0.clone();
                let ctrl_mask = 1usize << ctrl_bit;
                let half = 1usize << (nq - 1);
                for k in 0..half {
                    let i = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
                    expected[i] *= batch_phase_at(i, &groups, num_groups);
                }

                let mut actual = state0;
                // SAFETY: AVX2, BMI2, and FMA detected above.
                unsafe { apply_batch_phase_bmi2(&mut actual, ctrl_bit, &groups, num_groups) };
                assert_tile_close(
                    &actual,
                    &expected,
                    &format!("apply_batch_phase groups={num_groups} nq={nq} ctrl={ctrl_bit}"),
                );
            }
        }

        let scalar_tile = |tile: &mut [Complex64], base_idx: usize, ctrl_bit: usize| {
            let ctrl_mask = 1usize << ctrl_bit;
            let tile_bits = tile.len().trailing_zeros() as usize;
            if ctrl_bit >= tile_bits {
                if (base_idx & ctrl_mask) == 0 {
                    return;
                }
                for (j, amp) in tile.iter_mut().enumerate() {
                    *amp *= batch_phase_at(base_idx + j, &groups, num_groups);
                }
            } else {
                let half = tile.len() >> 1;
                for k in 0..half {
                    let j = insert_zero_bit(k, ctrl_bit) | ctrl_mask;
                    tile[j] *= batch_phase_at(base_idx + j, &groups, num_groups);
                }
            }
        };

        for &(ctrl_bit, len, base_idx) in &[
            (0usize, 2usize, 0usize),
            (0, 6, 4),
            (0, 64, 128),
            (1, 16, 0),
            (1, 64, 64),
            (3, 16, 32),
            (3, 64, 0),
            (6, 0, 0),
            (6, 1, 0),
            (6, 1, 64),
            (6, 3, 64),
            (6, 17, 64),
            (6, 64, 0),
            (6, 64, 64),
        ] {
            let tile0 = sample_tile(len, base_idx + ctrl_bit);
            let mut expected = tile0.clone();
            scalar_tile(&mut expected, base_idx, ctrl_bit);

            let mut actual = tile0;
            // SAFETY: AVX2, BMI2, and FMA detected above.
            unsafe { batch_phase_tile_bmi2(&mut actual, base_idx, ctrl_bit, &groups, num_groups) };
            assert_tile_close(
                &actual,
                &expected,
                &format!(
                    "batch_phase_tile groups={num_groups} ctrl={ctrl_bit} base={base_idx} len={len}"
                ),
            );
        }
    }
}
