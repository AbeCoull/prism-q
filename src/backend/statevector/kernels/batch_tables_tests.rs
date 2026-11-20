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
    if !std::is_x86_feature_detected!("bmi2") || !std::is_x86_feature_detected!("fma") {
        panic!("bmi2 and fma are required to check the PEXT path against the scalar path");
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
