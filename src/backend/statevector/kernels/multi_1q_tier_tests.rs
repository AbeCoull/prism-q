use super::*;
use crate::backend::Backend;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn random_unitary(rng: &mut ChaCha8Rng) -> [[Complex64; 2]; 2] {
    let mut angle = || rng.random::<f64>() * std::f64::consts::TAU;
    let (theta, phi, lambda) = (angle() / 2.0, angle(), angle());
    let (c, s) = (theta.cos(), theta.sin());
    [
        [Complex64::new(c, 0.0), -Complex64::from_polar(s, lambda)],
        [
            Complex64::from_polar(s, phi),
            Complex64::from_polar(c, phi + lambda),
        ],
    ]
}

fn random_state(num_qubits: usize, rng: &mut ChaCha8Rng) -> Vec<Complex64> {
    (0..1usize << num_qubits)
        .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
        .collect()
}

fn apply_scalar(state: &mut [Complex64], target: usize, mat: &[[Complex64; 2]; 2]) {
    let bit = 1usize << target;
    for i in (0..state.len()).filter(|i| i & bit == 0) {
        let (a, b) = (state[i], state[i | bit]);
        state[i] = mat[0][0] * a + mat[0][1] * b;
        state[i | bit] = mat[1][0] * a + mat[1][1] * b;
    }
}

fn assert_matches(got: &[Complex64], want: &[Complex64], label: &str) {
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert!(
            (got - want).norm() < 1e-12,
            "{label}, amplitude {i}: {got} vs {want}"
        );
    }
}

// Targets span the L2, L3 and untiled tiers in no particular order. The route
// depends on the thread count, so every route must agree bitwise.
#[cfg(feature = "parallel")]
#[test]
fn medium_tier_routes_match_per_gate_application() {
    const NUM_QUBITS: usize = 18;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = random_state(NUM_QUBITS, &mut rng);
    let gates: Vec<TargetedGate1q> = [17, 0, 14, 5, 16, 13, 15]
        .into_iter()
        .map(|target| (target, random_unitary(&mut rng)))
        .collect();

    let mut expected = start.clone();
    for (target, mat) in &gates {
        apply_scalar(&mut expected, *target, mat);
    }
    let (gathered, _) = multi_1q_route(
        &gates,
        NUM_QUBITS,
        max_target_for_tile(multi_gate_l2_tile_par(start.len())),
        false,
        MAX_SHARED_1Q_TARGETS,
    );
    assert!(!matches!(gathered, Multi1qRoute::Tiered));
    let routes = [
        (true, Multi1qRoute::Tiered),
        (false, Multi1qRoute::Tiered),
        (false, gathered),
    ]
    .map(|(medium_tiled, route)| {
        let mut state = start.clone();
        apply_multi_1q_par_routed(&mut state, &gates, medium_tiled, &route);
        state
    });
    assert_matches(&routes[0], &expected, "L3 tier");
    for route in &routes[1..] {
        assert!(
            route.iter().zip(&routes[0]).all(|(a, b)| a == b),
            "the routes must agree bitwise, or results depend on the thread count"
        );
    }
}

// A shared traversal whose lowest target sits below the per-lane tile: the run
// under that target is shorter than the tile, as on the parallel route once the
// pool cuts the L2 tile below a quarter of the budget.
#[cfg(feature = "parallel")]
#[test]
fn shared_traversal_handles_runs_shorter_than_the_lane_tile() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for (num_qubits, targets) in [
        (10usize, vec![6usize, 7]),
        (10, vec![9, 6, 8]),
        (15, vec![12, 13]),
        (15, vec![12, 13, 14]),
    ] {
        let start = random_state(num_qubits, &mut rng);
        let gates: Vec<TargetedGate1q> = targets
            .iter()
            .map(|&target| (target, random_unitary(&mut rng)))
            .collect();
        let mut expected = start.clone();
        for (target, mat) in &gates {
            apply_scalar(&mut expected, *target, mat);
        }
        let mut state = start;
        apply_multi_1q_shared(&mut state, &gates);
        assert_matches(
            &state,
            &expected,
            &format!("{num_qubits}q targets {targets:?}"),
        );
    }
}

// Every shape the router separates: targets in the tile only, a full layer, and
// one to ten targets at the top of the register, alone and with low targets.
// Under `parallel` the routed result must also equal the tiers bitwise.
#[test]
fn gathered_routes_match_per_gate_application() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for num_qubits in [14usize, 16, 18, 20] {
        let start = random_state(num_qubits, &mut rng);
        let mut cases: Vec<Vec<usize>> = vec![
            (0..num_qubits.min(12)).rev().collect(),
            (0..num_qubits).collect(),
        ];
        for k in 1..=10 {
            let high: Vec<usize> = (num_qubits - k..num_qubits).rev().collect();
            let mut mixed: Vec<usize> = [0, 3, 7, 11]
                .into_iter()
                .filter(|&q| q < num_qubits - k)
                .collect();
            mixed.extend(&high);
            cases.push(high);
            cases.push(mixed);
        }
        for targets in cases {
            let gates: Vec<TargetedGate1q> = targets
                .iter()
                .map(|&target| (target, random_unitary(&mut rng)))
                .collect();
            let mut expected = start.clone();
            for (target, mat) in &gates {
                apply_scalar(&mut expected, *target, mat);
            }
            let mut backend = StatevectorBackend::new(42);
            backend.init(num_qubits, 0).unwrap();
            backend.state.copy_from_slice(&start);
            backend.apply_multi_1q(&gates);
            let label = format!("{num_qubits}q targets {targets:?}");
            assert_matches(&backend.state, &expected, &label);

            #[cfg(feature = "parallel")]
            if num_qubits >= crate::backend::PARALLEL_THRESHOLD_QUBITS {
                let mut tiered = start.clone();
                let medium_tiled = keep_l3_tier(tiered.len());
                apply_multi_1q_par_routed(&mut tiered, &gates, medium_tiled, &Multi1qRoute::Tiered);
                assert!(tiered == backend.state, "{label}: routes differ bitwise");
            }
        }
    }
}

fn noncommuting_2q_pair() -> ([[Complex64; 4]; 4], [[Complex64; 4]; 4]) {
    let cx = crate::gates::Gate::Cx.matrix_4x4();
    let (s0, c0) = (0.37f64).sin_cos();
    let (s1, c1) = (0.61f64).sin_cos();
    let u0 = [
        [Complex64::new(c0, 0.0), Complex64::new(-s0, 0.0)],
        [Complex64::new(s0, 0.0), Complex64::new(c0, 0.0)],
    ];
    let u1 = [
        [Complex64::new(c1, 0.0), Complex64::new(0.0, -s1)],
        [Complex64::new(0.0, -s1), Complex64::new(c1, 0.0)],
    ];
    let mut dense = [[Complex64::new(0.0, 0.0); 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            dense[r][c] = u0[r >> 1][c >> 1] * u1[r & 1][c & 1];
        }
    }
    (cx, dense)
}

fn run_multi_2q_reference(
    start: &[Complex64],
    num_qubits: usize,
    gates: &[TargetedGate2q],
) -> Vec<Complex64> {
    let mut reference = StatevectorBackend::new(42);
    reference.init(num_qubits, 0).unwrap();
    reference.state.copy_from_slice(start);
    for &(q0, q1, ref mat) in gates {
        reference.apply_fused_2q(q0, q1, mat);
    }
    reference.state
}

fn run_multi_2q_batched(
    start: &[Complex64],
    num_qubits: usize,
    gates: &[TargetedGate2q],
) -> (Vec<Complex64>, usize) {
    let mut batched = StatevectorBackend::new(42);
    batched.init(num_qubits, 0).unwrap();
    batched.state.copy_from_slice(start);
    let _ = crate::backend::statevector::sweeps::take_ticks();
    batched.apply_multi_2q(gates);
    let ticks = crate::backend::statevector::sweeps::take_ticks();
    (batched.state, ticks)
}

#[test]
fn multi_2q_preserves_order_across_non_monotonic_tiers() {
    const NUM_QUBITS: usize = 20;
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let start = random_state(NUM_QUBITS, &mut rng);
    let (mat_a, mat_b) = noncommuting_2q_pair();

    let tile_bits = crate::gates::multi_2q_tile_bits_for(NUM_QUBITS);
    let budget = crate::gates::multi_2q_high_budget_for(NUM_QUBITS);
    let low = crate::gates::multi_2q_low_bits();
    assert!(
        low + budget < NUM_QUBITS - 1,
        "test requires room above high budget"
    );

    // Build filler gates touching distinct qubits in `low..` to push past `budget`.
    let push_over_budget = |gates: &mut Vec<TargetedGate2q>, skip_q: usize| {
        let mut distinct = 0usize;
        let mut q = low;
        while distinct <= budget && q < NUM_QUBITS - 1 {
            if q != skip_q {
                gates.push((q, q + 1, mat_b));
                distinct += 2;
            }
            q += 2;
        }
    };

    // 1. L3 -> L2 with subcube_plan == None (when L3 tier exists).
    if tile_bits <= MULTI_GATE_MAX_L3_TARGET {
        let mut l3_to_l2 = vec![(0, tile_bits, mat_a)];
        push_over_budget(&mut l3_to_l2, tile_bits);
        l3_to_l2.push((0, 1, mat_b));
        assert!(subcube_plan(&l3_to_l2, NUM_QUBITS).is_none());
        assert!(!multi_2q_keeps_order(&l3_to_l2, NUM_QUBITS));
        let expected = run_multi_2q_reference(&start, NUM_QUBITS, &l3_to_l2);
        let (actual, _) = run_multi_2q_batched(&start, NUM_QUBITS, &l3_to_l2);
        assert_matches(&actual, &expected, "L3 -> L2 over budget");
    }

    // 2. Untiled -> L2 with subcube_plan == None.
    let mut untiled_to_l2 = vec![(0, NUM_QUBITS - 1, mat_a)];
    push_over_budget(&mut untiled_to_l2, NUM_QUBITS - 1);
    untiled_to_l2.push((0, 1, mat_b));
    assert!(subcube_plan(&untiled_to_l2, NUM_QUBITS).is_none());
    assert!(!multi_2q_keeps_order(&untiled_to_l2, NUM_QUBITS));
    let expected = run_multi_2q_reference(&start, NUM_QUBITS, &untiled_to_l2);
    let (actual, _) = run_multi_2q_batched(&start, NUM_QUBITS, &untiled_to_l2);
    assert_matches(&actual, &expected, "untiled -> L2 over budget");

    // 3. Multi-split untiled -> L3 -> L2 -> untiled -> L2 requiring > 2 splits.
    let mut multi_split = vec![(0, NUM_QUBITS - 1, mat_a)];
    push_over_budget(&mut multi_split, NUM_QUBITS - 1);
    multi_split.push((0, tile_bits.min(MULTI_GATE_MAX_L3_TARGET), mat_b));
    multi_split.push((0, 1, mat_a));
    multi_split.push((0, NUM_QUBITS - 1, mat_b));
    push_over_budget(&mut multi_split, NUM_QUBITS - 1);
    multi_split.push((0, 1, mat_a));
    assert!(subcube_plan(&multi_split, NUM_QUBITS).is_none());
    assert!(!multi_2q_keeps_order(&multi_split, NUM_QUBITS));
    let expected = run_multi_2q_reference(&start, NUM_QUBITS, &multi_split);
    let (actual, _) = run_multi_2q_batched(&start, NUM_QUBITS, &multi_split);
    assert_matches(&actual, &expected, "multi-split descending tiers");

    // 4. L3 -> L2 and untiled -> L2 with subcube_plan == Some (single pass).
    for (label, hi) in [
        ("L3 -> L2 subcube", tile_bits.min(MULTI_GATE_MAX_L3_TARGET)),
        ("untiled -> L2 subcube", NUM_QUBITS - 1),
    ] {
        let gates = vec![(0, hi, mat_a), (0, 1, mat_b)];
        assert!(subcube_plan(&gates, NUM_QUBITS).is_some());
        let expected = run_multi_2q_reference(&start, NUM_QUBITS, &gates);
        let (actual, ticks) = run_multi_2q_batched(&start, NUM_QUBITS, &gates);
        assert_eq!(ticks, 1, "{label}: expected 1 subcube pass");
        assert_matches(&actual, &expected, label);
    }

    // 5. Empty, single-gate, and monotonic ascending tiers over budget.
    let (empty_actual, empty_ticks) = run_multi_2q_batched(&start, NUM_QUBITS, &[]);
    assert_eq!(empty_ticks, 0);
    assert_matches(&empty_actual, &start, "empty batch");

    let single = vec![(0, 1, mat_a)];
    let expected = run_multi_2q_reference(&start, NUM_QUBITS, &single);
    let (actual, _) = run_multi_2q_batched(&start, NUM_QUBITS, &single);
    assert_matches(&actual, &expected, "single gate");

    let mut ascending = vec![(0, 1, mat_a)];
    for q in (low..NUM_QUBITS - 1).step_by(2) {
        ascending.push((q, q + 1, mat_b));
    }
    ascending.push((0, NUM_QUBITS - 1, mat_a));
    assert!(subcube_plan(&ascending, NUM_QUBITS).is_none());
    assert!(multi_2q_keeps_order(&ascending, NUM_QUBITS));
    let expected = run_multi_2q_reference(&start, NUM_QUBITS, &ascending);
    let (actual, _) = run_multi_2q_batched(&start, NUM_QUBITS, &ascending);
    assert_matches(&actual, &expected, "ascending tiers over budget");
}

#[test]
fn multi_2q_recursive_split_keeps_subcube_tiling() {
    const NUM_QUBITS: usize = 20;
    let mut rng = ChaCha8Rng::seed_from_u64(43);
    let start = random_state(NUM_QUBITS, &mut rng);
    let (mat_a, mat_b) = noncommuting_2q_pair();

    let tile_bits = crate::gates::multi_2q_tile_bits_for(NUM_QUBITS);
    let budget = crate::gates::multi_2q_high_budget_for(NUM_QUBITS);
    let low = crate::gates::multi_2q_low_bits();
    assert!(budget >= 2, "budget must hold at least 2 high qubits");
    assert!(low + budget <= NUM_QUBITS - 2);

    // Run 1 fills the high-qubit budget starting with an untiled gate on qubit NUM_QUBITS - 1.
    let mut gates = vec![(0, NUM_QUBITS - 1, mat_a)];
    for offset in 0..(budget - 1) {
        gates.push((0, low + offset, mat_b));
    }
    let split_expected = gates.len();
    // Run 2 introduces a fresh high qubit above tile_bits followed by a low gate,
    // descending across tiers so the batch splits into two subcube-tiled runs.
    let fresh_high = (NUM_QUBITS - 2).max(tile_bits);
    gates.push((0, fresh_high, mat_a));
    gates.push((0, 1, mat_b));

    assert!(subcube_plan(&gates, NUM_QUBITS).is_none());
    assert!(!multi_2q_keeps_order(&gates, NUM_QUBITS));
    assert_eq!(multi_2q_ordered_prefix(&gates, NUM_QUBITS), split_expected);
    assert!(subcube_plan(&gates[..split_expected], NUM_QUBITS).is_some());
    assert!(subcube_plan(&gates[split_expected..], NUM_QUBITS).is_some());

    let expected = run_multi_2q_reference(&start, NUM_QUBITS, &gates);
    let (actual, ticks) = run_multi_2q_batched(&start, NUM_QUBITS, &gates);
    assert_eq!(
        ticks,
        2,
        "expected 2 subcube passes instead of {} per-gate passes",
        gates.len()
    );
    assert_matches(&actual, &expected, "recursive split keeps subcube tiling");
}
