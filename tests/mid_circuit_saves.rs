//! Mid-circuit save points: what they record, where they decline.

use prism_q::{BackendKind, Circuit, Gate, SaveSpec, SavedValue, simulate};

const SEED: u64 = 42;
const EPS: f64 = 1e-12;

/// Four layers of entangling rotations, deterministic in `n`.
fn body(circuit: &mut Circuit, n: usize, layers: usize, offset: usize) {
    for d in 0..layers {
        for q in 0..n {
            circuit.add_gate(Gate::Rx(0.17 * (offset + d * n + q + 1) as f64), &[q]);
        }
        for q in 0..n.saturating_sub(1) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
}

fn saved_statevector(value: &SavedValue) -> &[num_complex::Complex64] {
    match value {
        SavedValue::StateVector(amps) => amps,
        other => panic!("expected a statevector, got {other:?}"),
    }
}

fn assert_close(a: &[num_complex::Complex64], b: &[num_complex::Complex64]) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).norm() < EPS,
            "amplitude {i}: {x} vs {y} differ by {}",
            (x - y).norm()
        );
    }
}

// The contract: what a save records at a point equals running the circuit that
// ends at that point. Run on every backend that accepts a dense statevector.
#[test]
fn a_save_matches_running_the_prefix_alone() {
    for (kind, n) in [
        (BackendKind::Statevector, 6),
        (BackendKind::Sparse, 6),
        (BackendKind::Mps { max_bond_dim: 64 }, 6),
        (BackendKind::TensorNetwork { tolerance: None }, 6),
        (BackendKind::Auto, 6),
    ] {
        let mut prefix = Circuit::new(n, n);
        body(&mut prefix, n, 2, 0);

        let mut whole = prefix.clone();
        whole.add_save(SaveSpec::StateVector, "midpoint");
        body(&mut whole, n, 2, 100);

        let reference = simulate(&prefix)
            .backend(kind.clone())
            .seed(SEED)
            .state_vector()
            .unwrap_or_else(|e| panic!("{kind:?} prefix: {e}"));

        let outcome = simulate(&whole)
            .backend(kind.clone())
            .seed(SEED)
            .run()
            .unwrap_or_else(|e| panic!("{kind:?} whole: {e}"));

        assert_eq!(outcome.saves.len(), 1, "{kind:?}");
        assert_eq!(outcome.saves[0].label, "midpoint", "{kind:?}");
        assert_close(saved_statevector(&outcome.saves[0].value), &reference);
    }
}

// Above MIN_QUBITS_FOR_DIAG_BATCH every fusion pass is live, so this is the one
// that would catch a pass reordering a gate across the save point.
#[test]
fn fusion_does_not_move_a_gate_across_a_save() {
    let n = 16;
    let mut prefix = Circuit::new(n, n);
    body(&mut prefix, n, 3, 0);

    let mut whole = prefix.clone();
    whole.add_save(SaveSpec::StateVector, "after_prefix");
    body(&mut whole, n, 3, 500);

    let reference = simulate(&prefix)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .state_vector()
        .unwrap();
    let outcome = simulate(&whole)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap();

    assert_close(saved_statevector(&outcome.saves[0].value), &reference);
}

#[test]
fn saves_come_back_in_the_order_their_points_were_reached() {
    let n = 5;
    let mut circuit = Circuit::new(n, n);
    let mut prefixes = Vec::new();
    for step in 0..3 {
        body(&mut circuit, n, 1, step * 50);
        let mut upto = circuit.clone();
        upto.instructions
            .retain(|i| !matches!(i, prism_q::Instruction::Save { .. }));
        prefixes.push(upto);
        circuit.add_save(SaveSpec::StateVector, format!("step{step}"));
    }

    let outcome = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap();

    assert_eq!(outcome.saves.len(), 3);
    assert_eq!(circuit.save_count(), 3);
    for (step, record) in outcome.saves.iter().enumerate() {
        assert_eq!(record.label, format!("step{step}"));
        let reference = simulate(&prefixes[step])
            .backend(BackendKind::Statevector)
            .seed(SEED)
            .state_vector()
            .unwrap();
        assert_close(saved_statevector(&record.value), &reference);
    }
}

#[test]
fn a_probabilities_save_matches_the_prefix_distribution() {
    let n = 6;
    let mut prefix = Circuit::new(n, n);
    body(&mut prefix, n, 2, 0);

    let mut whole = prefix.clone();
    whole.add_save(SaveSpec::Probabilities, "mid");
    body(&mut whole, n, 2, 70);

    let reference = simulate(&prefix)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap()
        .probabilities
        .unwrap()
        .to_vec();

    let outcome = simulate(&whole)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap();
    let SavedValue::Probabilities(saved) = &outcome.saves[0].value else {
        panic!("expected probabilities, got {:?}", outcome.saves[0].value);
    };
    assert_eq!(saved.len(), reference.len());
    for (i, (a, b)) in saved.iter().zip(&reference).enumerate() {
        assert!((a - b).abs() < 1e-10, "state {i}: {a} vs {b}");
    }
}

#[test]
fn a_density_matrix_save_matches_the_prefix_on_the_density_matrix_backend() {
    let n = 3;
    let mut prefix = Circuit::new(n, n);
    body(&mut prefix, n, 2, 0);

    let mut whole = prefix.clone();
    whole.add_save(SaveSpec::DensityMatrix, "rho");
    body(&mut whole, n, 2, 20);

    let reference = simulate(&prefix)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .reduced_density_matrix(&(0..n).collect::<Vec<_>>())
        .unwrap();

    let outcome = simulate(&whole)
        .backend(BackendKind::DensityMatrix)
        .seed(SEED)
        .run()
        .unwrap();
    let SavedValue::DensityMatrix(saved) = &outcome.saves[0].value else {
        panic!(
            "expected a density matrix, got {:?}",
            outcome.saves[0].value
        );
    };
    assert_close(saved, &reference.data);
}

// A save after a measurement records the collapsed state, so the two halves of
// a mid-circuit measurement are observable.
#[test]
fn a_save_after_a_measurement_records_the_collapsed_state() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_measure(0, 0);
    circuit.add_save(SaveSpec::Probabilities, "after_collapse");

    let outcome = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .run()
        .unwrap();
    let SavedValue::Probabilities(p) = &outcome.saves[0].value else {
        panic!("expected probabilities");
    };
    let collapsed = if outcome.classical_bits[0] { 3 } else { 0 };
    assert!(
        (p[collapsed] - 1.0).abs() < 1e-10,
        "collapsed state should carry all the weight, got {p:?}"
    );
}

#[test]
fn a_circuit_with_no_save_points_returns_no_records() {
    let mut circuit = Circuit::new(4, 4);
    body(&mut circuit, 4, 2, 0);
    let outcome = simulate(&circuit).seed(SEED).run().unwrap();
    assert!(outcome.saves.is_empty());
    assert_eq!(circuit.save_count(), 0);
}

#[test]
fn terminals_other_than_run_decline_a_save() {
    let mut circuit = Circuit::new(4, 4);
    body(&mut circuit, 4, 2, 0);
    circuit.add_save(SaveSpec::StateVector, "mid");
    body(&mut circuit, 4, 1, 90);

    let err = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(8)
        .unwrap_err()
        .to_string();
    assert!(err.contains("save point"), "{err}");

    let err = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .marginals()
        .unwrap_err()
        .to_string();
    assert!(err.contains("save point"), "{err}");
}

#[test]
fn a_route_that_cannot_read_a_state_declines_by_name() {
    // Clifford plus T with no measurement routes to stabilizer rank, which
    // carries a weighted sum of branches rather than one state.
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_save(SaveSpec::StateVector, "mid");
    circuit.add_gate(Gate::T, &[1]);

    let err = simulate(&circuit)
        .backend(BackendKind::StabilizerRank)
        .seed(SEED)
        .run()
        .unwrap_err()
        .to_string();
    assert!(err.contains("save point") || err.contains("save"), "{err}");
}

#[test]
fn a_save_has_no_openqasm_spelling() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_save(SaveSpec::StateVector, "mid");
    let err = prism_q::circuit::qasm_export::to_qasm3(&circuit)
        .unwrap_err()
        .to_string();
    assert!(err.contains("save point `mid`"), "{err}");
}
