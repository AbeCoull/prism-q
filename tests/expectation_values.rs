//! Public API coverage for `run_expectation_values` and
//! `Simulate::expectation_values`.

mod common;

use prism_q::gates::Gate;
use prism_q::{
    BackendKind, Circuit, NoiseModel, PauliAxis, PauliObservable, PauliTerm,
    density_matrix_expectation_values, run_expectation_values, run_observable_expectation,
    simulate,
};

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
fn tensor_network_expectation_values_match_statevector() {
    assert_matches_statevector(
        "tensor network expectation",
        BackendKind::TensorNetwork,
        &entangled_non_clifford(6),
    );
}

// ---- batched traversal vs the per-observable loop ----

// Each of these backends evaluates a batch of observables in one pass over its
// state and a batch of one with the single-observable reduction. Batching
// reassociates the sum across observables and never within one, so the two must
// agree. The list alternates between the Z-only family and the family carrying
// an X or Y factor, since the batched pass splits on exactly that and has to
// interleave the halves back into request order; the empty observable is the
// identity, which touches no site at all. Comparing against the same backend
// one observable at a time rather than against the statevector, because a
// cross-backend check cannot witness a shared batched helper breaking.
fn batch_observables() -> [Vec<PauliTerm>; 6] {
    [
        vec![PauliTerm::x(0), PauliTerm::y(1)],
        vec![PauliTerm::z(2), PauliTerm::z(5)],
        vec![PauliTerm::y(4)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(3)],
        vec![],
        vec![PauliTerm::x(3), PauliTerm::z(5)],
    ]
}

fn assert_batch_matches_per_observable_loop(label: &str, backend: BackendKind, circuit: &Circuit) {
    let observables = batch_observables();
    let batched = simulate(circuit)
        .backend(backend.clone())
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    let one_at_a_time: Vec<f64> = observables
        .iter()
        .map(|obs| {
            simulate(circuit)
                .backend(backend.clone())
                .seed(42)
                .expectation_values(std::slice::from_ref(obs))
                .unwrap()[0]
        })
        .collect();
    common::assert_probs_close(&batched, &one_at_a_time, 1e-12, label);
}

#[test]
fn factored_batch_matches_the_per_observable_loop() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Ry(0.4 + 0.1 * q as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Cx, &[4, 5]);
    assert_batch_matches_per_observable_loop("factored batch blocks", BackendKind::Factored, &c);
    assert_batch_matches_per_observable_loop(
        "factored batch merged",
        BackendKind::Factored,
        &entangled_non_clifford(6),
    );
}

#[test]
fn sparse_batch_matches_the_per_observable_loop() {
    assert_batch_matches_per_observable_loop(
        "sparse batch",
        BackendKind::Sparse,
        &entangled_non_clifford(6),
    );
}

// The identity environment right of an observable's last site is closed against
// the swept environment as a trace, so it enters transposed. A chain of real
// site tensors makes both matrices real symmetric and hides a wrong transpose;
// the diagonal T alone does not help, because its phase cancels in the
// environment contraction. Rx after T is what puts an imaginary part in the
// environments, and the statevector value is the independent authority for it.
fn complex_environment_chain(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.3 + 0.15 * q as f64), &[q]);
    }
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..n {
        c.add_gate(Gate::T, &[q]);
        c.add_gate(Gate::Rx(0.2 + 0.1 * q as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, n - 1]);
    c.add_gate(Gate::Cx, &[2, 4]);
    c
}

#[test]
fn mps_batch_matches_the_per_observable_loop() {
    let mps = BackendKind::Mps {
        max_bond_dim: 1 << 8,
    };
    assert_batch_matches_per_observable_loop("mps batch", mps.clone(), &entangled_non_clifford(6));
    assert_batch_matches_per_observable_loop(
        "mps batch complex environments",
        mps.clone(),
        &complex_environment_chain(6),
    );
    assert_matches_statevector(
        "mps expectation complex environments",
        mps,
        &complex_environment_chain(6),
    );
}

// SWAP routing permutes the site layout, so the first and last site an
// observable touches are not its first and last logical qubit, and the shared
// identity environments are read at the routed positions.
#[test]
fn mps_batch_matches_the_per_observable_loop_under_swap_routing() {
    let mut c = Circuit::new(6, 0);
    for q in 0..6 {
        c.add_gate(Gate::Ry(0.25 * (q + 1) as f64), &[q]);
    }
    c.add_gate(Gate::Cx, &[0, 5]);
    c.add_gate(Gate::T, &[2]);
    c.add_gate(Gate::Cx, &[1, 4]);
    assert_batch_matches_per_observable_loop(
        "mps batch swap routed",
        BackendKind::Mps {
            max_bond_dim: 1 << 8,
        },
        &c,
    );
}

#[test]
fn density_matrix_batch_matches_the_per_observable_loop() {
    let circuit = entangled_non_clifford(6);
    let observables = batch_observables();
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let batched =
        density_matrix_expectation_values(&circuit, &observables, Some(&noise), 42).unwrap();
    let one_at_a_time: Vec<f64> = observables
        .iter()
        .map(|obs| {
            density_matrix_expectation_values(&circuit, std::slice::from_ref(obs), Some(&noise), 42)
                .unwrap()[0]
        })
        .collect();
    common::assert_probs_close(&batched, &one_at_a_time, 1e-12, "density matrix batch");
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

// ---- Weighted observables ----

#[test]
fn grouped_observable_matches_the_per_term_weighted_sum() {
    use prism_q::circuits;

    for n in [8usize, 12] {
        let circuit = circuits::hardware_efficient_ansatz(n, 2, 42);
        let terms = circuits::jordan_wigner_hamiltonian(n, 300, 42);
        let observable = PauliObservable::from_terms(terms).unwrap();
        assert!(observable.num_groups() < observable.num_terms());

        let grouped = run_observable_expectation(&circuit, &observable, 42).unwrap();
        assert!(grouped.variance.is_some());
        assert_eq!(
            grouped.group_variances.as_ref().unwrap().len(),
            observable.num_groups()
        );

        let obs_vecs: Vec<Vec<PauliTerm>> = observable
            .terms()
            .iter()
            .map(|(_, factors)| factors.clone())
            .collect();
        let values = run_expectation_values(&circuit, &obs_vecs, 42).unwrap();
        let weighted: f64 = observable
            .terms()
            .iter()
            .zip(&values)
            .map(|((c, _), v)| c * v)
            .sum();
        assert!(
            (grouped.mean - weighted).abs() < 1e-12,
            "grouped mean {} vs per-term weighted sum {} at n={n}",
            grouped.mean,
            weighted
        );
    }
}

// Z-string products carry no phase, so <H^2> expands client-side through the
// per-term path and pins the single-group variance to Var(H) exactly.
#[test]
fn single_group_variance_matches_the_operator_square() {
    use prism_q::circuits;

    let n = 6;
    let circuit = circuits::hardware_efficient_ansatz(n, 2, 7);
    let strings: Vec<(f64, usize)> = vec![
        (0.5, 0b000001),
        (-1.25, 0b001010),
        (2.0, 0b110100),
        (0.75, 0b000110),
    ];
    let factors_of = |mask: usize| -> Vec<PauliTerm> {
        (0..n)
            .filter(|q| mask >> q & 1 == 1)
            .map(PauliTerm::z)
            .collect()
    };
    let observable =
        PauliObservable::from_terms(strings.iter().map(|&(c, mask)| (c, factors_of(mask))))
            .unwrap();
    assert_eq!(observable.num_groups(), 1);

    let result = run_observable_expectation(&circuit, &observable, 42).unwrap();

    let mut square_terms: Vec<Vec<PauliTerm>> = Vec::new();
    let mut square_coefficients = Vec::new();
    for &(ca, ma) in &strings {
        for &(cb, mb) in &strings {
            square_terms.push(factors_of(ma ^ mb));
            square_coefficients.push(ca * cb);
        }
    }
    let values = run_expectation_values(&circuit, &square_terms, 42).unwrap();
    let h_square: f64 = square_coefficients
        .iter()
        .zip(&values)
        .map(|(c, v)| c * v)
        .sum();
    let expected = h_square - result.mean * result.mean;
    assert!(
        (result.variance.unwrap() - expected).abs() < 1e-10,
        "grouped variance {} vs <H^2>-<H>^2 = {expected}",
        result.variance.unwrap()
    );
}

#[test]
fn clifford_route_reports_the_weighted_mean_without_variance() {
    let observable = PauliObservable::from_terms([
        (2.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-1.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
        (0.5, vec![]),
    ])
    .unwrap();
    let result = run_observable_expectation(&bell(), &observable, 42).unwrap();
    assert!((result.mean - 1.5).abs() < TOL);
    assert!(result.variance.is_none());
    assert!(result.std_error.is_none());
}

#[test]
fn explicit_statevector_reaches_the_grouped_route_on_clifford_circuits() {
    let observable = PauliObservable::from_terms([
        (2.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-1.0, vec![PauliTerm::x(0), PauliTerm::x(1)]),
        (0.5, vec![]),
    ])
    .unwrap();
    let result = simulate(&bell())
        .backend(BackendKind::Statevector)
        .seed(42)
        .observable_expectation(&observable)
        .unwrap();
    assert!((result.mean - 1.5).abs() < TOL);
    // ZZ and XX split into two groups, and each is deterministic on the Bell
    // state, so every group variance vanishes.
    let group_variances = result.group_variances.unwrap();
    assert_eq!(group_variances.len(), 2);
    assert!(result.variance.unwrap().abs() < TOL);
}

#[test]
fn cross_backend_weighted_means_agree() {
    use prism_q::circuits;

    let n = 6;
    let circuit = circuits::hardware_efficient_ansatz(n, 1, 3);
    let terms = circuits::jordan_wigner_hamiltonian(n, 60, 5);
    let observable = PauliObservable::from_terms(terms).unwrap();

    let reference = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .observable_expectation(&observable)
        .unwrap();
    assert!(reference.variance.is_some());

    for kind in [
        BackendKind::Mps { max_bond_dim: 64 },
        BackendKind::DensityMatrix,
        BackendKind::Factored,
    ] {
        let result = simulate(&circuit)
            .backend(kind.clone())
            .seed(42)
            .observable_expectation(&observable)
            .unwrap();
        assert!(
            (result.mean - reference.mean).abs() < 1e-9,
            "{kind:?} mean {} vs statevector {}",
            result.mean,
            reference.mean
        );
        assert!(result.variance.is_none(), "{kind:?} has no grouped route");
    }
}

#[test]
fn noisy_route_matches_density_matrix_expectations() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.05);

    let observable = PauliObservable::from_terms([
        (2.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (-0.5, vec![PauliTerm::x(0)]),
    ])
    .unwrap();

    let result = simulate(&circuit)
        .backend(BackendKind::DensityMatrix)
        .noise(&noise)
        .seed(42)
        .observable_expectation(&observable)
        .unwrap();

    let obs_vecs: Vec<Vec<PauliTerm>> = observable
        .terms()
        .iter()
        .map(|(_, factors)| factors.clone())
        .collect();
    let values = density_matrix_expectation_values(&circuit, &obs_vecs, Some(&noise), 42).unwrap();
    let weighted: f64 = observable
        .terms()
        .iter()
        .zip(&values)
        .map(|((c, _), v)| c * v)
        .sum();
    assert!((result.mean - weighted).abs() < 1e-12);
    assert!(result.variance.is_none());
}

#[test]
fn initial_state_route_returns_the_weighted_mean() {
    use num_complex::Complex64;

    let s = std::f64::consts::FRAC_1_SQRT_2;
    let plus = [Complex64::new(s, 0.0), Complex64::new(s, 0.0)];
    let theta = 0.7_f64;
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rz(theta), &[0]);

    let observable = PauliObservable::from_terms([(2.0, vec![PauliTerm::x(0)])]).unwrap();
    let result = simulate(&circuit)
        .initial_state(&plus)
        .seed(42)
        .observable_expectation(&observable)
        .unwrap();
    assert!((result.mean - 2.0 * theta.cos()).abs() < TOL);
    assert!(result.variance.is_none());
}
