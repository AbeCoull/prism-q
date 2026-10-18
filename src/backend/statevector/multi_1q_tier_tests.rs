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
            if num_qubits >= PARALLEL_THRESHOLD_QUBITS {
                let mut tiered = start.clone();
                let medium_tiled = keep_l3_tier(tiered.len());
                apply_multi_1q_par_routed(&mut tiered, &gates, medium_tiled, &Multi1qRoute::Tiered);
                assert!(tiered == backend.state, "{label}: routes differ bitwise");
            }
        }
    }
}
