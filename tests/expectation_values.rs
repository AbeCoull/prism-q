//! Public API coverage for `run_expectation_values` and
//! `Simulate::expectation_values`.

mod common;

use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, PauliAxis, PauliTerm, run_expectation_values, simulate};

const TOL: f64 = 1e-10;

fn bell() -> Circuit {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c
}

fn assert_close(got: &[f64], want: &[f64], tol: f64) {
    common::assert_probs_close(got, want, tol, "expectation");
}

#[test]
fn clifford_expectations_are_exact() {
    // Bell state: <ZZ>=<XX>=1, <YY>=-1, <Z0>=0, and the empty (identity) = 1.
    let vals = run_expectation_values(
        &bell(),
        &[
            vec![PauliTerm::z(0), PauliTerm::z(1)],
            vec![PauliTerm::x(0), PauliTerm::x(1)],
            vec![PauliTerm::y(0), PauliTerm::y(1)],
            vec![PauliTerm::z(0)],
            vec![],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[1.0, 1.0, -1.0, 0.0, 1.0], TOL);
}

#[test]
fn plus_state_single_qubit() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    let vals = run_expectation_values(
        &c,
        &[
            vec![PauliTerm::x(0)],
            vec![PauliTerm::y(0)],
            vec![PauliTerm::z(0)],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[1.0, 0.0, 0.0], TOL);
}

#[test]
fn non_clifford_statevector_route_matches_analytic() {
    let theta = 0.7_f64;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    // Rx(theta)|0>: <X>=0, <Y>=-sin(theta), <Z>=cos(theta).
    let vals = run_expectation_values(
        &c,
        &[
            vec![PauliTerm::x(0)],
            vec![PauliTerm::y(0)],
            vec![PauliTerm::z(0)],
        ],
        42,
    )
    .unwrap();
    assert_close(&vals, &[0.0, -theta.sin(), theta.cos()], TOL);
}

#[test]
fn clifford_route_matches_statevector_route() {
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::S, &[0]);
    let observables = [
        vec![PauliTerm::x(0), PauliTerm::y(1), PauliTerm::z(2)],
        vec![PauliTerm::z(0), PauliTerm::z(2)],
        vec![PauliTerm::new(1, PauliAxis::X)],
    ];
    let auto = run_expectation_values(&c, &observables, 42).unwrap();
    let sv = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    assert_close(&auto, &sv, TOL);
}

#[test]
fn clifford_t_deterministic_pauli_matches_statevector() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::H, &[1]);
    let observables = [
        vec![PauliTerm::x(0), PauliTerm::x(1)],
        vec![PauliTerm::z(0)],
    ];
    let sv = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    let spd = simulate(&c)
        .backend(BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        })
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    assert_close(&spd, &sv, 1e-9);
}

#[test]
fn stochastic_pauli_seeds_each_observable_independently() {
    // H,T,H gives a stochastic estimate; two identical observables must draw
    // from distinct sample streams rather than sharing one seed.
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::T, &[0]);
    c.add_gate(Gate::H, &[0]);
    let z0 = vec![PauliTerm::z(0)];
    let vals = simulate(&c)
        .backend(BackendKind::StochasticPauli { num_samples: 64 })
        .seed(42)
        .expectation_values(&[z0.clone(), z0])
        .unwrap();
    assert!(vals[0] != vals[1], "identical observables shared a seed");
}

#[test]
fn invalid_observable_is_rejected() {
    use prism_q::PrismError::{InvalidParameter, InvalidQubit};
    let c = bell();
    assert!(matches!(
        run_expectation_values(&c, &[vec![PauliTerm::z(5)]], 42).unwrap_err(),
        InvalidQubit { .. }
    ));
    assert!(matches!(
        run_expectation_values(&c, &[vec![PauliTerm::z(0), PauliTerm::x(0)]], 42).unwrap_err(),
        InvalidParameter { .. }
    ));
}

#[test]
fn non_unitary_circuit_is_rejected() {
    let qasm = "OPENQASM 3.0;\nqubit[1] q;\nbit[1] c;\nh q[0];\nc[0] = measure q[0];";
    let c = prism_q::circuit::openqasm::parse(qasm).unwrap();
    let observables: [&[Vec<PauliTerm>]; 2] = [&[], &[vec![PauliTerm::z(0)]]];
    for backend in [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::DeterministicPauli {
            epsilon: 0.0,
            max_terms: 0,
        },
        BackendKind::StochasticPauli { num_samples: 16 },
    ] {
        for obs in observables {
            let err = simulate(&c)
                .backend(backend.clone())
                .seed(42)
                .expectation_values(obs)
                .unwrap_err();
            assert!(
                matches!(err, prism_q::PrismError::IncompatibleBackend { .. }),
                "{backend:?} accepted a non-unitary circuit"
            );
        }
    }
}

// ===== backends that hold a polynomial-size state =====

/// Observables that mix all three axes over a non-Clifford, entangled state.
/// Wide enough that a backend confusing qubit order or dropping a factor
/// cannot land on the statevector value by accident.
fn mixed_axis_observables() -> [Vec<PauliTerm>; 5] {
    [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1), PauliTerm::y(3)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)],
        vec![
            PauliTerm::y(0),
            PauliTerm::x(2),
            PauliTerm::z(4),
            PauliTerm::y(5),
        ],
        vec![],
    ]
}

fn entangled_non_clifford(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.3 + 0.15 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::T, &[1]);
    c.add_gate(Gate::Cx, &[0, n - 1]);
    c
}

fn assert_matches_statevector(label: &str, backend: BackendKind, circuit: &Circuit) {
    let observables = mixed_axis_observables();
    let sv = simulate(circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    let got = simulate(circuit)
        .backend(backend)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    common::assert_probs_close(&got, &sv, 1e-9, label);
}

#[test]
fn mps_expectation_values_match_statevector() {
    assert_matches_statevector(
        "mps expectation",
        BackendKind::Mps {
            max_bond_dim: 1 << 8,
        },
        &entangled_non_clifford(6),
    );
}

// SWAP routing permutes the site layout, so the observable's logical qubits
// no longer index the sites they started on.
#[test]
fn mps_expectation_values_survive_swap_routing() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Ry(0.25 * (q + 1) as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 5]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[1, 4]);
    assert_matches_statevector(
        "mps expectation swap routed",
        BackendKind::Mps {
            max_bond_dim: 1 << 8,
        },
        &c,
    );
}

#[test]
fn sparse_expectation_values_match_statevector() {
    assert_matches_statevector(
        "sparse expectation",
        BackendKind::Sparse,
        &entangled_non_clifford(6),
    );
}

#[test]
fn factored_expectation_values_match_statevector() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Ry(0.4 + 0.1 * q as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[4, 5]);
    assert_matches_statevector("factored expectation blocks", BackendKind::Factored, &c);
}

#[test]
fn factored_expectation_values_match_statevector_when_fully_merged() {
    assert_matches_statevector(
        "factored expectation merged",
        BackendKind::Factored,
        &entangled_non_clifford(6),
    );
}

// The product state factorizes an observable into one closed-form factor per
// qubit. The statevector is the reference for the signs, which is the half of
// those closed forms hand algebra gets wrong.
#[test]
fn product_expectation_values_match_statevector() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Ry(0.35 + 0.2 * q as f64), &[q]);
        c.add_gate(Gate::Rz(0.15 + 0.25 * q as f64), &[q]);
    }
    c.add_gate(Gate::T, &[3]);
    assert_matches_statevector("product expectation", BackendKind::ProductState, &c);
}

// Every single-qubit axis on every axis eigenstate, so a swapped sign or a
// swapped real and imaginary part in one closed form cannot hide behind the
// others averaging out.
#[test]
fn product_expectation_values_cover_each_axis_eigenstate() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::X, &[2]);
    c.add_gate(Gate::H, &[3]);
    c.add_gate(Gate::Z, &[3]);
    c.add_gate(Gate::H, &[4]);
    c.add_gate(Gate::S, &[4]);
    c.add_gate(Gate::H, &[5]);
    c.add_gate(Gate::Sdg, &[5]);

    let axes = [PauliAxis::X, PauliAxis::Y, PauliAxis::Z];
    let observables: Vec<Vec<PauliTerm>> = (0..6)
        .flat_map(|q| axes.iter().map(move |&axis| vec![PauliTerm::new(q, axis)]))
        .collect();

    // q0 |0>, q1 |+>, q2 |1>, q3 |->, q4 |+i>, q5 |-i>.
    #[rustfmt::skip]
    let want = [
        0.0, 0.0, 1.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        -1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, -1.0, 0.0,
    ];
    let got = simulate(&c)
        .backend(BackendKind::ProductState)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    assert_close(&got, &want, TOL);
}

// The tensor network contracts to a dense statevector by construction, so it
// has no native observable path. The rejection has to name the backend that
// cannot serve the request, not the route that selected it.
#[test]
fn backends_without_a_native_path_name_themselves() {
    let c = entangled_non_clifford(5);
    let observables = [vec![PauliTerm::z(0)]];
    let err = simulate(&c)
        .backend(BackendKind::TensorNetwork)
        .seed(42)
        .expectation_values(&observables)
        .unwrap_err();
    match err {
        prism_q::PrismError::BackendUnsupported {
            backend: reported, ..
        } => assert_eq!(reported, "tensornetwork"),
        other => panic!("expected a BackendUnsupported naming tensornetwork, got {other:?}"),
    }
}

// The native path validates observables before it runs, so a bad qubit index
// costs nothing on a circuit the statevector could not hold.
#[test]
fn invalid_observable_is_rejected_on_the_native_path() {
    use prism_q::PrismError::{InvalidParameter, InvalidQubit};
    let c = entangled_non_clifford(5);
    let mps = BackendKind::Mps {
        max_bond_dim: 1 << 8,
    };
    assert!(matches!(
        simulate(&c)
            .backend(mps.clone())
            .seed(42)
            .expectation_values(&[vec![PauliTerm::z(9)]])
            .unwrap_err(),
        InvalidQubit { .. }
    ));
    assert!(matches!(
        simulate(&c)
            .backend(mps)
            .seed(42)
            .expectation_values(&[vec![PauliTerm::z(0), PauliTerm::x(0)]])
            .unwrap_err(),
        InvalidParameter { .. }
    ));
}

#[test]
fn statevector_route_matches_scalar_reference_above_the_parallel_norm_threshold() {
    // 2^16 amplitudes is where the dense route's normalization pass switches to
    // a parallel SIMD reduction, and 2^12 is below it. Both must agree with a
    // scalar sweep of the exported amplitudes.
    use prism_q::backend::Backend;
    use prism_q::backend::statevector::StatevectorBackend;
    use prism_q::circuits;

    for n in [12usize, 16] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, 42);
        let observables: Vec<Vec<PauliTerm>> = vec![
            vec![PauliTerm::z(0), PauliTerm::z(n - 1)],
            vec![PauliTerm::x(n / 2)],
        ];
        let got = run_expectation_values(&circuit, &observables, 42).unwrap();

        let mut backend = StatevectorBackend::new(42);
        prism_q::sim::run_on(&mut backend, &circuit).unwrap();
        let state = backend.export_statevector().unwrap();
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();

        let zz = 1usize | (1usize << (n - 1));
        let mut expect_zz = 0.0f64;
        for (i, amp) in state.iter().enumerate() {
            let sign = if (i & zz).count_ones() & 1 == 1 {
                -1.0
            } else {
                1.0
            };
            expect_zz += sign * amp.norm_sqr();
        }
        let xmask = 1usize << (n / 2);
        let mut expect_x = 0.0f64;
        for (i, amp) in state.iter().enumerate() {
            expect_x += (state[i ^ xmask].conj() * amp).re;
        }

        assert_close(&got, &[expect_zz / norm, expect_x / norm], TOL);
    }
}

// A batch of observables shares one traversal of the amplitude buffer, while a
// batch of one keeps the single-observable reduction. The two must agree, and
// the batch must return values in request order: the shared pass splits the
// list into a Z-only family and a family carrying an X or Y factor, then
// interleaves them back. Sizes straddle the parallel threshold at 2^16, and the
// observable order alternates between the two families so a mismatched
// interleave cannot pass.
#[test]
fn batched_observables_match_one_at_a_time() {
    use prism_q::circuits;

    for n in [12usize, 16, 17] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, 42);
        let observables: Vec<Vec<PauliTerm>> = vec![
            vec![PauliTerm::x(0), PauliTerm::y(1)],
            vec![PauliTerm::z(2), PauliTerm::z(n - 1)],
            vec![PauliTerm::y(n / 2)],
            vec![PauliTerm::z(n - 2)],
            vec![],
        ];

        let batched = run_expectation_values(&circuit, &observables, 42).unwrap();
        let one_at_a_time: Vec<f64> = observables
            .iter()
            .map(|obs| run_expectation_values(&circuit, std::slice::from_ref(obs), 42).unwrap()[0])
            .collect();

        assert_close(&batched, &one_at_a_time, TOL);
    }
}
