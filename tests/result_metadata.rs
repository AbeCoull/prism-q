//! Provenance contract: every terminal stamps a result, dispatch onto an
//! approximate backend says so, and `require_exact` rejects that route instead.

mod common;

use common::SEED;
use prism_q::{
    BackendKind, Circuit, CircuitBuilder, Exactness, Gate, PauliTerm, Placement, ResolvedBackend,
    simulate,
};

fn bell() -> Circuit {
    CircuitBuilder::new_with_classical(2, 2)
        .h(0)
        .cx(0, 1)
        .measure(0, 0)
        .measure(1, 1)
        .build()
}

/// Brickwork entangling layers with T gates between them. One CX ladder from a
/// product state leaves every bond at rank 2, which a small cap does not
/// truncate; the offset second layer is what pushes the chain past it.
fn entangling_brickwork(num_qubits: usize, layers: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::H, &[q]);
    }
    for layer in 0..layers {
        for q in (layer % 2..num_qubits.saturating_sub(1)).step_by(2) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
        for q in 0..num_qubits {
            circuit.add_gate(Gate::T, &[q]);
            circuit.add_gate(Gate::H, &[q]);
        }
    }
    circuit
}

#[test]
fn small_circuit_reports_an_exact_backend() {
    let outcome = simulate(&bell()).seed(SEED).run().unwrap();
    assert!(outcome.metadata.is_exact(), "{:?}", outcome.metadata);
    assert_eq!(outcome.metadata.fidelity_lower_bound(), None);
}

// The flag marks the route and the bound reports the run, so a bond the
// circuit never fills reports approximate at fidelity 1.
#[test]
fn mps_reports_a_fidelity_bound() {
    let mut circuit = Circuit::new(6, 0);
    circuit.add_gate(Gate::H, &[0]);
    for q in 0..5 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }

    let roomy = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 64 })
        .seed(SEED)
        .run()
        .unwrap();
    assert_eq!(
        roomy.metadata.fidelity_lower_bound(),
        Some(1.0),
        "a GHZ chain needs bond 2, so bond 64 discards nothing"
    );

    let clamped = simulate(&entangling_brickwork(10, 6))
        .backend(BackendKind::Mps { max_bond_dim: 2 })
        .seed(SEED)
        .run()
        .unwrap();
    let bound = clamped.metadata.fidelity_lower_bound().unwrap();
    assert!(
        bound < 1.0,
        "bond 2 on a ladder with T gates on every layer must discard weight, got {bound}"
    );
}

#[test]
fn require_exact_leaves_an_exact_route_alone() {
    let outcome = simulate(&bell()).seed(SEED).require_exact().run().unwrap();
    assert!(outcome.metadata.is_exact());
}

// A sampled estimate carries a standard error; an evaluated one reports
// exactness instead of an interval of width zero.
#[test]
fn sampled_expectations_carry_a_standard_error() {
    // T branching is what gives the estimator variance, and it only fires when
    // the propagated Pauli anticommutes with Z, so the observable has to reach
    // the T through a basis change.
    let circuit = entangling_brickwork(4, 2);
    let observables = vec![vec![PauliTerm::z(0)]];

    let sampled = simulate(&circuit)
        .backend(BackendKind::StochasticPauli { num_samples: 4000 })
        .seed(SEED)
        .expectation_values_reported(&observables)
        .unwrap();
    let errors = sampled
        .std_errors
        .clone()
        .expect("sampled route reports an error");
    assert_eq!(errors.len(), 1);
    assert!(errors[0] > 0.0, "got {}", errors[0]);
    assert_eq!(sampled.metadata.shots, Some(4000));
    assert!(!sampled.metadata.is_exact());

    let analytic = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_values_reported(&observables)
        .unwrap();
    assert!(analytic.std_errors.is_none());
    assert!(analytic.metadata.is_exact());

    let half_width = errors[0] * 5.0;
    assert!(
        (sampled.values[0] - analytic.values[0]).abs() <= half_width.max(0.05),
        "sampled {} vs analytic {}",
        sampled.values[0],
        analytic.values[0]
    );
}

// Placement is readable off the result. Without a device it is `Host`
// everywhere, stated rather than left to a guess about whether a run landed on
// the device.
#[test]
fn placement_is_reported() {
    let outcome = simulate(&bell()).seed(SEED).run().unwrap();
    #[cfg(not(feature = "gpu"))]
    assert_eq!(outcome.metadata.placement, Placement::Host);
    #[cfg(feature = "gpu")]
    assert!(matches!(
        outcome.metadata.placement,
        Placement::Host | Placement::Device
    ));
}

// The placeholder `from_shots` installs must never reach a caller. Every public
// terminal is walked here; a route that forgets to stamp fails this and nothing
// else.
#[test]
fn every_terminal_stamps_metadata() {
    let measured = bell();
    let mut unitary = Circuit::new(2, 0);
    unitary.add_gate(Gate::H, &[0]);
    unitary.add_gate(Gate::Cx, &[0, 1]);

    let kinds = [
        BackendKind::Auto,
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Sparse,
        BackendKind::Mps { max_bond_dim: 16 },
        BackendKind::Factored,
        BackendKind::TensorNetwork,
        BackendKind::DensityMatrix,
    ];

    for kind in kinds {
        let label = format!("{kind:?}");
        let run = simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .run()
            .unwrap();
        assert_ne!(
            run.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "run on {label}"
        );

        let shots = simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .shots(8)
            .unwrap();
        assert_ne!(
            shots.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "shots on {label}"
        );
        assert_eq!(shots.metadata.shots, Some(8), "shots on {label}");
        assert_ne!(
            simulate(&measured)
                .backend(kind.clone())
                .seed(SEED)
                .shots(0)
                .unwrap()
                .metadata
                .backend,
            ResolvedBackend::Other("unstamped"),
            "zero shots on {label}"
        );

        let counts = simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .sample_counts(8)
            .unwrap();
        assert_ne!(
            counts.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "counts on {label}"
        );

        let marginals = simulate(&measured)
            .backend(kind.clone())
            .seed(SEED)
            .marginals()
            .unwrap();
        assert_ne!(
            marginals.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "marginals on {label}"
        );

        let expectations = simulate(&unitary)
            .backend(kind.clone())
            .seed(SEED)
            .expectation_values_reported(&[vec![PauliTerm::z(0)]])
            .unwrap();
        assert_ne!(
            expectations.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "expectations on {label}"
        );
    }
}

// A zero-shot request runs nothing, so there is no backend to read provenance
// off and the route has to supply it.
#[test]
fn zero_shots_still_names_the_route() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(1, 1);

    for kind in [BackendKind::Auto, BackendKind::Statevector] {
        let shots = simulate(&circuit)
            .backend(kind.clone())
            .seed(SEED)
            .shots(0)
            .unwrap();
        assert!(shots.shots.is_empty());
        assert_ne!(
            shots.metadata.backend,
            ResolvedBackend::Other("unstamped"),
            "zero shots on {kind:?}"
        );
    }
}

// Mid-circuit measurement forces the per-shot route, where each shot evolves
// its own state and the ensemble keeps the weakest claim.
#[test]
fn per_shot_route_stamps_metadata() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(Gate::T, &[1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(1, 1);

    let shots = simulate(&circuit)
        .backend(BackendKind::Mps { max_bond_dim: 8 })
        .seed(SEED)
        .shots(8)
        .unwrap();
    assert_eq!(shots.metadata.backend, ResolvedBackend::Mps);
    assert!(matches!(
        shots.metadata.exactness,
        Exactness::Approximate { .. }
    ));
}
