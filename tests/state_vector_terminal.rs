//! The `state_vector` terminal: its index convention, the backends that
//! answer it, and the states that have no amplitude vector to report.

mod common;

use std::f64::consts::FRAC_1_SQRT_2;

use num_complex::Complex64;
use prism_q::CircuitBuilder;
use prism_q::backend::Backend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::gates::Gate;
use prism_q::sim::noise::NoiseModel;
use prism_q::{BackendKind, Circuit, PrismError, circuits, sim, simulate};

use common::{SEED, SV_EPS};

fn assert_state_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).norm() < SV_EPS,
            "{label}: [{i}] expected {want}, got {got}"
        );
    }
}

fn amp(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

fn zero() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

// Qubit 0 is the least significant bit, so `x` on it lands at index 1 and not
// at index 2. Every consumer that reindexes amplitudes reads this contract.
#[test]
fn qubit_zero_is_the_least_significant_bit() {
    let mut builder = CircuitBuilder::new(2);
    builder.x(0);
    let state = simulate(&builder.build())
        .seed(SEED)
        .state_vector()
        .unwrap();
    assert_state_close(&state, &[zero(), amp(1.0), zero(), zero()], "x on qubit 0");

    let mut builder = CircuitBuilder::new(2);
    builder.x(1);
    let state = simulate(&builder.build())
        .seed(SEED)
        .state_vector()
        .unwrap();
    assert_state_close(&state, &[zero(), zero(), amp(1.0), zero()], "x on qubit 1");
}

#[test]
fn bell_state_amplitudes() {
    let mut builder = CircuitBuilder::new(2);
    builder.h(0).cx(0, 1);
    let state = simulate(&builder.build())
        .seed(SEED)
        .state_vector()
        .unwrap();
    assert_state_close(
        &state,
        &[amp(FRAC_1_SQRT_2), zero(), zero(), amp(FRAC_1_SQRT_2)],
        "bell",
    );
}

// The terminal must report what the backend holds, not what the statevector
// would have held, so each route is checked against its own export.
#[test]
fn every_pure_state_backend_answers_with_its_own_export() {
    // An entangling block plus two idle qubits, so the factored route keeps
    // more than one block. The product route takes a circuit with no
    // entanglement at all, which is the only shape it accepts.
    let mut entangled = CircuitBuilder::new(4);
    entangled.h(0).cx(0, 1).rx(0.7, 2).ry(0.3, 3);
    let mut product = CircuitBuilder::new(4);
    product.h(0).rx(0.7, 1).ry(0.3, 2).t(3);

    let groups: [(&str, Circuit, Vec<BackendKind>); 2] = [
        (
            "entangled",
            entangled.build(),
            vec![
                BackendKind::Auto,
                BackendKind::Statevector,
                BackendKind::Sparse,
                BackendKind::Factored,
                BackendKind::TensorNetwork { tolerance: None },
                BackendKind::Mps { max_bond_dim: 32 },
            ],
        ),
        (
            "product",
            product.build(),
            vec![BackendKind::Auto, BackendKind::ProductState],
        ),
    ];

    for (label, circuit, kinds) in groups {
        let mut reference = StatevectorBackend::new(SEED);
        sim::run_on(&mut reference, &circuit).unwrap();
        let expected = reference.export_statevector().unwrap();
        for kind in kinds {
            let state = simulate(&circuit)
                .seed(SEED)
                .backend(kind.clone())
                .state_vector()
                .unwrap_or_else(|e| panic!("{label} on {kind:?} declined: {e}"));
            assert_state_close(&state, &expected, &format!("{label} on {kind:?}"));
        }
    }
}

#[test]
fn clifford_circuit_answers_on_the_stabilizer_route() {
    let circuit = circuits::ghz_circuit(3);
    let state = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::Stabilizer)
        .state_vector()
        .unwrap();
    let mut expected = vec![zero(); 8];
    expected[0] = amp(FRAC_1_SQRT_2);
    expected[7] = amp(FRAC_1_SQRT_2);
    assert_state_close(&state, &expected, "stabilizer ghz");
}

#[test]
fn initial_state_is_honoured() {
    let mut builder = CircuitBuilder::new(1);
    builder.gate(Gate::X, &[0]);
    let start = vec![zero(), amp(1.0)];
    let state = simulate(&builder.build())
        .seed(SEED)
        .initial_state(&start)
        .state_vector()
        .unwrap();
    assert_state_close(&state, &[amp(1.0), zero()], "x on |1>");
}

// A mixture has no amplitude vector. The decline has to name that rather than
// report an arbitrary trajectory, which is what a shot request would sample.
#[test]
fn a_noise_model_declines() {
    let circuit = circuits::ghz_circuit(2);
    let noise = NoiseModel::uniform_depolarizing(&circuit, 0.1);
    let err = simulate(&circuit)
        .seed(SEED)
        .noise(&noise)
        .state_vector()
        .unwrap_err();
    let PrismError::IncompatibleBackend { reason, .. } = err else {
        panic!("expected IncompatibleBackend");
    };
    assert!(
        reason.contains("reduced_density_matrix"),
        "the decline should name the terminal that answers: {reason}"
    );
}

#[test]
fn the_density_matrix_backend_declines() {
    let circuit = circuits::ghz_circuit(2);
    let err = simulate(&circuit)
        .seed(SEED)
        .backend(BackendKind::DensityMatrix)
        .state_vector()
        .unwrap_err();
    assert!(
        matches!(err, PrismError::BackendUnsupported { .. }),
        "expected the backend to decline, got {err:?}"
    );
}

// The answer is read off one output state, so a circuit that leaves several
// seeded branches has no single vector to report.
#[test]
fn a_measurement_declines() {
    let mut circuit = Circuit::new(1, 1);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_measure(0, 0);
    let err = simulate(&circuit).seed(SEED).state_vector().unwrap_err();
    assert!(
        matches!(err, PrismError::IncompatibleBackend { .. }),
        "expected a decline, got {err:?}"
    );
}
