use super::*;
use crate::Limits;
use crate::limits::ResolvedLimits;
use crate::sim::route::{MIN_AUTO_REGISTER_EXPECTATION_QUBITS, decide_register_subset};
use crate::sim::{BackendKind, Query, ResolvedBackend, RouteRule, simulate};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const EPS: f64 = 1e-10;

fn random_clifford_t(n: usize, gates: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    let ones = [Gate::H, Gate::S, Gate::Sdg, Gate::X, Gate::Y, Gate::Z];
    let mut t_left = t_count;
    for g in 0..gates {
        if t_left > 0 && rng.random_range(0..gates - g) < t_left {
            let gate = if rng.random_bool(0.5) {
                Gate::T
            } else {
                Gate::Tdg
            };
            c.add_gate(gate, &[rng.random_range(0..n)]);
            t_left -= 1;
        } else if rng.random_bool(0.4) {
            let a = rng.random_range(0..n);
            let b = (a + rng.random_range(1..n)) % n;
            c.add_gate(
                [Gate::Cx, Gate::Cz][rng.random_range(0..2)].clone(),
                &[a, b],
            );
        } else {
            c.add_gate(
                ones[rng.random_range(0..ones.len())].clone(),
                &[rng.random_range(0..n)],
            );
        }
    }
    c
}

fn random_observables(n: usize, count: usize, seed: u64) -> Vec<Vec<PauliTerm>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let axes = [PauliAxis::X, PauliAxis::Y, PauliAxis::Z];
    (0..count)
        .map(|_| {
            let weight = rng.random_range(1..=4.min(n));
            let mut qubits: Vec<usize> = Vec::new();
            while qubits.len() < weight {
                let q = rng.random_range(0..n);
                if !qubits.contains(&q) {
                    qubits.push(q);
                }
            }
            qubits
                .into_iter()
                .map(|q| PauliTerm::new(q, axes[rng.random_range(0..3)]))
                .collect()
        })
        .collect()
}

fn statevector_values(circuit: &Circuit, observables: &[Vec<PauliTerm>]) -> Vec<f64> {
    simulate(circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(observables)
        .unwrap()
}

fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < EPS, "{label}[{i}]: {g} against {w}");
    }
}

#[test]
fn every_engine_matches_the_statevector() {
    for seed in 0..24u64 {
        let n = 6 + (seed % 4) as usize;
        let circuit = random_clifford_t(n, 70, 4 + (seed % 9) as usize, seed);
        let observables = random_observables(n, 12, seed + 100);
        let want = statevector_values(&circuit, &observables);
        for (label, max_dense, max_excess) in [
            ("cheapest", MAX_REGISTER_QUBITS, MAX_PROPAGATED_EXCESS),
            ("propagate", 0, 64),
            ("dense", 64, 0),
        ] {
            let plan =
                RegisterExpectations::plan_with(&circuit, &observables, max_dense, max_excess)
                    .unwrap();
            assert_close(&plan.values(), &want, &format!("seed {seed} {label}"));
        }
    }
}

// H and T on every qubit anchors 22 one-qubit parts. Behind a CX ladder a T on the last
// qubit has the axis `Z` on every qubit, an excess rotation that links one part past the
// dense cap.
#[test]
fn a_part_past_the_dense_cap_answers_through_its_excess_rotations() {
    let n = 22;
    let mut circuit = Circuit::new(n, 0);
    for q in 0..n {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for (i, q) in [n - 1, 7, 15, 3].into_iter().enumerate() {
        circuit.add_gate(if i % 2 == 0 { Gate::T } else { Gate::Tdg }, &[q]);
        circuit.add_gate(Gate::Cx, &[q, (q + 5) % n]);
    }
    let observables = random_observables(n, 16, 7)
        .into_iter()
        .chain([
            vec![PauliTerm::z(n - 1)],
            vec![PauliTerm::x(7), PauliTerm::y(12)],
        ])
        .collect::<Vec<_>>();
    let plan = RegisterExpectations::plan(&circuit, &observables).unwrap();
    assert!(
        plan.parts.iter().any(|part| {
            part.mask.count_ones() as usize > MAX_REGISTER_QUBITS
                && matches!(part.engine, Engine::Propagate)
        }),
        "no part past the dense cap propagates"
    );
    assert_close(
        &plan.values(),
        &statevector_values(&circuit, &observables),
        "wide part",
    );
}

#[test]
fn an_observable_that_flips_a_zero_qubit_reads_zero() {
    let mut circuit = Circuit::new(3, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let observables = vec![
        vec![PauliTerm::x(2)],
        vec![PauliTerm::y(2), PauliTerm::z(0)],
        vec![PauliTerm::z(0), PauliTerm::z(1)],
    ];
    let plan = RegisterExpectations::plan(&circuit, &observables).unwrap();
    assert!(plan.pulled[0].is_none() && plan.pulled[1].is_none());
    let values = plan.values();
    assert_eq!(&values[..2], &[0.0, 0.0]);
    assert_close(
        &values,
        &statevector_values(&circuit, &observables),
        "zero qubit",
    );
}

#[test]
fn propagation_matches_a_dense_rotation() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..40 {
        let rotations: Vec<(u64, u64, u8, f64)> = (0..5)
            .map(|_| {
                let (x, z) = (rng.random_range(0..16u64), rng.random_range(0..16u64));
                let (x, z) = if x | z == 0 { (1, 0) } else { (x, z) };
                (
                    x,
                    z,
                    2 * rng.random_range(0..2u8),
                    rng.random_range(-1.0..1.0),
                )
            })
            .collect();
        let (px, pz) = (rng.random_range(0..16u64), rng.random_range(0..16u64));
        let mut amps: Vec<Complex64> = (0..16)
            .map(|_| Complex64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
            .collect();
        let norm = amps.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
        amps.iter_mut().for_each(|a| *a /= norm);
        let start = amps.clone();
        for &(x, z, phase4, theta) in &rotations {
            apply_rotation(&mut amps, RegisterPauli { x, z, phase4 }, theta);
        }
        let want = dense_value(&amps, px as usize, pz as usize);
        let got: f64 = propagate_register_rotations(px, pz, rotations.iter().copied())
            .into_iter()
            .map(|(x, z, c)| c.re * dense_value(&start, x as usize, z as usize))
            .sum();
        assert!((got - want).abs() < EPS, "{got} against {want}");
    }
}

fn wide_layers(n: usize, t_count: usize, seed: u64) -> Circuit {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    let mut t_left = t_count;
    for layer in 0..6 {
        for q in 0..n {
            if t_left > 0 && rng.random_bool(0.2) {
                c.add_gate(Gate::T, &[q]);
                t_left -= 1;
            }
            c.add_gate(
                [Gate::H, Gate::S, Gate::Z][rng.random_range(0..3)].clone(),
                &[q],
            );
        }
        for q in (layer % 2..n - 1).step_by(2) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

fn route(
    circuit: &Circuit,
    limits: Option<Limits>,
    query: Query<'_>,
) -> (ResolvedBackend, RouteRule) {
    let request = simulate(circuit).seed(42);
    let request = match limits {
        Some(limits) => request.limits(limits),
        None => request,
    };
    let plan = request.plan(query).unwrap();
    (plan.backend, plan.rule)
}

const REGISTER: (ResolvedBackend, RouteRule) =
    (ResolvedBackend::StabilizerRank, RouteRule::CliffordRegister);

#[test]
fn auto_expectations_and_marginals_take_the_register_on_clifford_t() {
    let circuit = wide_layers(16, 12, 3);
    let observables = random_observables(16, 8, 5);
    assert_eq!(
        route(&circuit, None, Query::Expectation(&observables)),
        REGISTER
    );
    assert_eq!(route(&circuit, None, Query::Marginals), REGISTER);

    let result = simulate(&circuit)
        .seed(42)
        .expectation_values_reported(&observables)
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::StabilizerRank);
    assert_eq!(result.metadata.reason, RouteRule::CliffordRegister);
    assert_close(
        &result.values,
        &statevector_values(&circuit, &observables),
        "auto",
    );

    let marginals = simulate(&circuit).seed(42).marginals().unwrap();
    assert_eq!(marginals.metadata.reason, RouteRule::CliffordRegister);
    let dense = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .marginals()
        .unwrap();
    for (q, (got, want)) in marginals.marginals.iter().zip(&dense.marginals).enumerate() {
        assert!((got.0 - want.0).abs() < EPS, "qubit {q}");
    }
}

#[test]
fn few_qubit_distributions_read_the_walsh_transform_of_the_register() {
    let circuit = wide_layers(16, 12, 9);
    for qubits in [vec![3], vec![0, 15], vec![5, 2, 9], vec![1, 4, 7, 10, 13]] {
        let plan = decide_register_subset(&circuit, &qubits, &ResolvedLimits::process_default());
        assert!(plan.is_some(), "{qubits:?} declined");
        let got = simulate(&circuit)
            .seed(42)
            .probabilities_of(&qubits)
            .unwrap();
        let want = simulate(&circuit)
            .backend(BackendKind::Statevector)
            .seed(42)
            .probabilities_of(&qubits)
            .unwrap();
        assert_close(&got, &want, &format!("{qubits:?}"));
    }
}

#[test]
fn routes_that_keep_their_engine() {
    let observables = vec![
        vec![PauliTerm::z(1)],
        vec![PauliTerm::x(0), PauliTerm::z(2)],
    ];

    let mut clifford = wide_layers(16, 0, 3);
    clifford.add_gate(Gate::S, &[0]);
    assert_eq!(
        route(&clifford, None, Query::Expectation(&observables)).1,
        RouteRule::SpdExact
    );

    let mut rotated = wide_layers(16, 6, 3);
    rotated.add_gate(Gate::Rx(0.3), &[0]);
    assert_ne!(
        route(&rotated, None, Query::Expectation(&observables)),
        REGISTER
    );

    let narrow = wide_layers(MIN_AUTO_REGISTER_EXPECTATION_QUBITS - 1, 6, 3);
    assert_ne!(
        route(&narrow, None, Query::Expectation(&observables)),
        REGISTER
    );
}

// Past the cap any exact plan is taken; a part past the dense cap whose excess also passes
// the propagation limit has no exact engine, and the route falls through to the tree.
#[test]
fn past_the_cap_the_register_takes_every_exact_plan() {
    let limits = || Some(Limits::new().max_statevector_qubits(20));
    let observables = vec![
        vec![PauliTerm::z(30)],
        vec![PauliTerm::x(4), PauliTerm::y(40)],
    ];
    let wide = wide_layers(48, 30, 11);
    assert_eq!(
        route(&wide, limits(), Query::Expectation(&observables)),
        REGISTER
    );

    let n = 24;
    let mut tangled = Circuit::new(n, 0);
    for q in 0..n {
        tangled.add_gate(Gate::H, &[q]);
        tangled.add_gate(Gate::T, &[q]);
    }
    for q in 0..n - 1 {
        tangled.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for i in 0..MAX_PROPAGATED_EXCESS + 1 {
        let q = (7 * i + 3) % n;
        tangled.add_gate(Gate::T, &[q]);
        tangled.add_gate(Gate::Cx, &[q, (q + 11) % n]);
    }
    let observables = vec![vec![PauliTerm::z(n - 1)]];
    assert!(RegisterExpectations::plan(&tangled, &observables).is_none());
    assert_ne!(
        route(&tangled, limits(), Query::Expectation(&observables)),
        REGISTER
    );
}
