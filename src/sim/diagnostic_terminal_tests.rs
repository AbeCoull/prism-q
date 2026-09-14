use super::*;
use crate::gates::Gate;

fn clifford_t_chain(n: usize) -> Circuit {
    let mut circuit = Circuit::new(n, 0);
    circuit.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    circuit.add_gate(Gate::T, &[0]);
    circuit
}

/// The routes that propagate an observable instead of holding a state, each
/// with the name its error must carry.
fn stateless_kinds() -> [(BackendKind, &'static str); 4] {
    [
        (BackendKind::StabilizerRank, "stabilizer-rank"),
        (
            BackendKind::StochasticPauli { num_samples: 64 },
            "stochastic Pauli",
        ),
        (
            BackendKind::DeterministicPauli {
                epsilon: 1e-3,
                max_terms: 1024,
            },
            "deterministic Pauli",
        ),
        (
            BackendKind::PauliPath {
                epsilon: 1e-3,
                max_terms: 1024,
            },
            "Pauli path",
        ),
    ]
}

fn assert_no_state(error: PrismError, terminal: &str, route: &str) {
    match error {
        PrismError::IncompatibleBackend { reason, .. } => {
            assert!(
                reason.starts_with(terminal) && reason.contains(&format!("the {route} route")),
                "{reason}"
            );
        }
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

#[test]
fn stateless_routes_decline_every_diagnostic() {
    let circuit = clifford_t_chain(6);
    for (kind, route) in stateless_kinds() {
        assert_no_state(
            simulate(&circuit)
                .backend(kind.clone())
                .seed(42)
                .reduced_density_matrix(&[0, 1])
                .unwrap_err(),
            "a reduced density matrix",
            route,
        );
        assert_no_state(
            simulate(&circuit)
                .backend(kind.clone())
                .seed(42)
                .entanglement_entropy(&[0, 1])
                .unwrap_err(),
            "entanglement entropy",
            route,
        );
        assert_no_state(
            simulate(&circuit)
                .backend(kind.clone())
                .seed(42)
                .overlap(simulate(&circuit).backend(kind).seed(42))
                .unwrap_err(),
            "a state overlap",
            route,
        );
    }
}

// A mixture is only held by the density matrix, so a noise model on any other
// backend is rejected before the run rather than answered from one trajectory.
#[test]
fn a_noise_model_outside_the_density_matrix_is_rejected() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = noise::NoiseBuilder::new()
        .after_gates(
            noise::GateFilter::all(),
            noise::NoiseChannel::Depolarizing { p: 0.02 },
        )
        .build(&circuit)
        .unwrap();
    for kind in [
        BackendKind::Statevector,
        BackendKind::Mps { max_bond_dim: 8 },
    ] {
        assert!(matches!(
            simulate(&circuit)
                .backend(kind.clone())
                .noise(&noise)
                .seed(42)
                .reduced_density_matrix(&[0])
                .unwrap_err(),
            PrismError::IncompatibleBackend { .. }
        ));
        assert!(matches!(
            simulate(&circuit)
                .backend(kind)
                .noise(&noise)
                .seed(42)
                .entanglement_entropy(&[0])
                .unwrap_err(),
            PrismError::IncompatibleBackend { .. }
        ));
    }
}

// The density matrix the noise model forces holds a mixture, which has no
// Schmidt decomposition, so the entropy terminal declines under any model.
#[test]
fn the_entropy_terminal_declines_under_a_noise_model() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = noise::NoiseBuilder::new()
        .after_gates(
            noise::GateFilter::all(),
            noise::NoiseChannel::Depolarizing { p: 0.02 },
        )
        .build(&circuit)
        .unwrap();
    assert_eq!(
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .seed(42)
            .entanglement_entropy(&[0])
            .unwrap_err(),
        PrismError::BackendUnsupported {
            backend: "density_matrix".to_string(),
            operation: "Schmidt values of a mixed state".to_string(),
        }
    );
}

// Readout error is indexed by classical bit and never reaches the state, so a
// model carrying it is rejected rather than silently ignored. The register
// declares the bits the model is indexed by and measures none of them, since a
// measurement would fail the unitary check first.
#[test]
fn readout_error_is_rejected_by_both_diagnostics() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = noise::NoiseBuilder::new()
        .uniform_readout_error(0.01, 0.02)
        .build(&circuit)
        .unwrap();
    assert!(matches!(
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .seed(42)
            .reduced_density_matrix(&[0])
            .unwrap_err(),
        PrismError::InvalidParameter { .. }
    ));
    assert!(matches!(
        simulate(&circuit)
            .backend(BackendKind::DensityMatrix)
            .noise(&noise)
            .seed(42)
            .entanglement_entropy(&[0])
            .unwrap_err(),
        PrismError::InvalidParameter { .. }
    ));
}

// An MPS at a bond cap can discard weight, so a caller who opted out of an
// approximate answer is turned away before the chain is built.
#[test]
fn require_exact_rejects_the_bounded_mps_route() {
    let circuit = clifford_t_chain(6);
    assert!(matches!(
        simulate(&circuit)
            .backend(BackendKind::Mps { max_bond_dim: 4 })
            .require_exact()
            .seed(42)
            .entanglement_entropy(&[0, 1])
            .unwrap_err(),
        PrismError::IncompatibleBackend { .. }
    ));
}

// Both terminals validate their subsystem before any state is allocated, so a
// bad index costs nothing and reports the same error the backends do.
#[test]
fn the_subsystem_is_validated_before_the_run() {
    let circuit = clifford_t_chain(4);
    assert_eq!(
        simulate(&circuit)
            .seed(42)
            .reduced_density_matrix(&[0, 4])
            .unwrap_err(),
        PrismError::InvalidQubit {
            index: 4,
            register_size: 4,
        }
    );
    assert!(matches!(
        simulate(&circuit)
            .seed(42)
            .entanglement_entropy(&[0, 1, 2, 3])
            .unwrap_err(),
        PrismError::InvalidParameter { .. }
    ));
}

// A measurement, reset or conditional leaves one seeded branch of several, not
// the state a diagnostic is defined on, so both terminals reject one before
// any state is allocated.
#[test]
fn a_non_unitary_circuit_is_rejected_by_both_diagnostics() {
    let mut measured = Circuit::new(2, 1);
    measured.add_gate(Gate::H, &[0]);
    measured.add_gate(Gate::Cx, &[0, 1]);
    measured.add_measure(0, 0);

    let mut reset = Circuit::new(2, 0);
    reset.add_gate(Gate::H, &[0]);
    reset.add_gate(Gate::Cx, &[0, 1]);
    reset.add_reset(0);

    let mut conditional = Circuit::new(2, 1);
    conditional.add_gate(Gate::H, &[0]);
    conditional.instructions.extend(crate::circuit::guarded(
        crate::circuit::ClassicalCondition::BitIsOne(0),
        vec![Instruction::Gate {
            gate: Gate::X,
            targets: crate::circuit::SmallVec::from_slice(&[1]),
        }],
    ));

    for circuit in [measured, reset, conditional] {
        for (error, subject) in [
            (
                simulate(&circuit)
                    .seed(42)
                    .reduced_density_matrix(&[0])
                    .unwrap_err(),
                "a reduced density matrix requires a unitary circuit",
            ),
            (
                simulate(&circuit)
                    .seed(42)
                    .entanglement_entropy(&[0])
                    .unwrap_err(),
                "entanglement entropy requires a unitary circuit",
            ),
        ] {
            match error {
                PrismError::IncompatibleBackend { reason, .. } => {
                    assert!(reason.starts_with(subject), "{reason}");
                }
                other => panic!("expected IncompatibleBackend, got {other:?}"),
            }
        }
    }
}

// A stabilizer state's spectrum is flat, so the tableau reads both the entropy
// and the values off one rank, and `Auto` keeps the route instead of falling
// back to the statevector.
#[test]
fn the_stabilizer_answers_the_entropy_and_the_flat_spectrum() {
    let mut bell = Circuit::new(2, 0);
    bell.add_gate(Gate::H, &[0]);
    bell.add_gate(Gate::Cx, &[0, 1]);
    let half = std::f64::consts::FRAC_1_SQRT_2;
    for kind in [BackendKind::Auto, BackendKind::Stabilizer] {
        let result = simulate(&bell)
            .backend(kind)
            .seed(42)
            .entanglement_entropy(&[0])
            .unwrap();
        assert!((result.entropy - std::f64::consts::LN_2).abs() < 1e-12);
        assert_eq!(result.metadata.backend, ResolvedBackend::Stabilizer);
        let values = result.schmidt_values.unwrap();
        assert_eq!(values.len(), 2, "{values:?}");
        assert!(
            values.iter().all(|v| (v - half).abs() < 1e-12),
            "{values:?}"
        );
    }
}

// Past the export cap the `2^r` values of a wide cut do not fit, while the
// rank behind them still does, so the terminal reports the entropy with no
// spectrum rather than declining.
#[test]
fn a_wide_stabilizer_cut_answers_the_entropy_without_the_spectrum() {
    let width = crate::backend::schmidt::export_cap() + 2;
    let mut circuit = Circuit::new(2 * width, 0);
    for q in 0..width {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::Cx, &[q, q + width]);
    }
    let subsystem: Vec<usize> = (0..width).collect();
    let result = simulate(&circuit)
        .backend(BackendKind::Stabilizer)
        .seed(42)
        .entanglement_entropy(&subsystem)
        .unwrap();
    assert!(
        (result.entropy - width as f64 * std::f64::consts::LN_2).abs() < 1e-9,
        "entropy {}",
        result.entropy
    );
    assert!(result.schmidt_values.is_none());
}

// A mixture has no statevector to dot, and Uhlmann fidelity is not an inner
// product, so the density matrix declines the overlap by name.
#[test]
fn the_density_matrix_declines_the_overlap() {
    let mut bell = Circuit::new(2, 0);
    bell.add_gate(Gate::H, &[0]);
    bell.add_gate(Gate::Cx, &[0, 1]);
    assert_eq!(
        simulate(&bell)
            .backend(BackendKind::DensityMatrix)
            .seed(42)
            .overlap(simulate(&bell).backend(BackendKind::DensityMatrix).seed(42))
            .unwrap_err(),
        PrismError::BackendUnsupported {
            backend: "density_matrix".to_string(),
            operation: "state overlap".to_string(),
        }
    );
}

// Two registers of different sizes have no inner product, and the check comes
// before either circuit runs.
#[test]
fn the_overlap_terminal_rejects_a_width_mismatch() {
    let two = Circuit::new(2, 0);
    let three = Circuit::new(3, 0);
    match simulate(&two)
        .seed(42)
        .overlap(simulate(&three).seed(42))
        .unwrap_err()
    {
        PrismError::InvalidParameter { message } => {
            assert!(
                message.starts_with("a state overlap needs two circuits of the same width"),
                "{message}"
            );
        }
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

// The overlap is an inner product of two pure states, so a noise model on
// either side is rejected rather than answered off one trajectory.
#[test]
fn the_overlap_terminal_declines_under_a_noise_model() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    let noise = noise::NoiseBuilder::new()
        .after_gates(
            noise::GateFilter::all(),
            noise::NoiseChannel::Depolarizing { p: 0.02 },
        )
        .build(&circuit)
        .unwrap();
    for (left, right) in [(Some(&noise), None), (None, Some(&noise))] {
        let mut a = simulate(&circuit);
        if let Some(model) = left {
            a = a.noise(model);
        }
        let mut b = simulate(&circuit);
        if let Some(model) = right {
            b = b.noise(model);
        }
        match a.seed(42).overlap(b.seed(42)).unwrap_err() {
            PrismError::IncompatibleBackend { reason, .. } => {
                assert!(
                    reason.starts_with("a state overlap is an inner product of two pure states"),
                    "{reason}"
                );
            }
            other => panic!("expected IncompatibleBackend, got {other:?}"),
        }
    }
}

// A measurement leaves one seeded branch of several, not the state an overlap
// is defined on, so the terminal rejects it before any state is allocated.
#[test]
fn the_overlap_terminal_rejects_a_non_unitary_circuit() {
    let mut measured = Circuit::new(2, 1);
    measured.add_gate(Gate::H, &[0]);
    measured.add_measure(0, 0);
    let unitary = Circuit::new(2, 1);
    match simulate(&measured)
        .seed(42)
        .overlap(simulate(&unitary).seed(42))
        .unwrap_err()
    {
        PrismError::IncompatibleBackend { reason, .. } => {
            assert!(
                reason.starts_with("a state overlap requires a unitary circuit"),
                "{reason}"
            );
        }
        other => panic!("expected IncompatibleBackend, got {other:?}"),
    }
}

// Auto picks the factored backend for a partially independent circuit, which
// holds no spectrum. The route was the dispatcher's choice, so it falls back
// to the statevector.
#[test]
fn auto_falls_back_to_the_statevector_when_its_route_cannot_answer() {
    let mut split = Circuit::new(10, 0);
    terminal_candidate_matrix_tests::rx_cx_chain(&mut split, 0..8);
    terminal_candidate_matrix_tests::rx_cx_chain(&mut split, 8..10);
    let result = simulate(&split)
        .seed(42)
        .entanglement_entropy(&[8])
        .unwrap();
    assert_eq!(result.metadata.backend, ResolvedBackend::Statevector);
    assert!(result.entropy > 0.0, "entropy {}", result.entropy);
}
