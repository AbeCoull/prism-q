//! Declines and argument checks for `Backend::schmidt_values` and
//! `Backend::entanglement_entropy`, and agreement of the `Simulate` terminal
//! over them with the statevector. The values themselves are pinned in
//! `golden_small_circuits.rs`, and the MPS routes in its own unit tests.

mod common;

use common::circuits::{CircuitCase, exact_small_cases, product_separable_cases};
use common::{MPS_EPS, PRODUCT_EPS, SEED, SV_EPS};
use prism_q::PrismError;
use prism_q::backend::Backend;
use prism_q::backend::density_matrix::DensityMatrixBackend;
use prism_q::backend::factored::FactoredBackend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::mps::MpsBackend;
use prism_q::backend::product::ProductStateBackend;
use prism_q::backend::sparse::SparseBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::backend::statevector::StatevectorBackend;
use prism_q::backend::tensornetwork::TensorNetworkBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits::ghz_circuit;
use prism_q::gates::Gate;
use prism_q::sim;
use prism_q::{BackendKind, simulate};

fn assert_declines(backend: &mut dyn Backend, operation: &str) {
    sim::run_on(backend, &ghz_circuit(4)).unwrap();
    let expected = PrismError::BackendUnsupported {
        backend: backend.name().to_string(),
        operation: operation.to_string(),
    };
    assert_eq!(backend.schmidt_values(&[0, 1]).unwrap_err(), expected);
    assert_eq!(backend.entanglement_entropy(&[0, 1]).unwrap_err(), expected);
}

#[test]
fn stabilizer_declines_schmidt_values() {
    assert_declines(&mut StabilizerBackend::new(SEED), "Schmidt values");
}

#[test]
fn factored_stabilizer_declines_schmidt_values() {
    assert_declines(&mut FactoredStabilizerBackend::new(SEED), "Schmidt values");
}

#[test]
fn density_matrix_declines_schmidt_values_of_a_mixed_state() {
    assert_declines(
        &mut DensityMatrixBackend::new(SEED),
        "Schmidt values of a mixed state",
    );
}

#[test]
fn sparse_declines_schmidt_values() {
    assert_declines(&mut SparseBackend::new(SEED), "Schmidt values");
}

#[test]
fn tensor_network_declines_schmidt_values() {
    assert_declines(&mut TensorNetworkBackend::new(SEED), "Schmidt values");
}

#[test]
fn factored_declines_schmidt_values() {
    assert_declines(&mut FactoredBackend::new(SEED), "Schmidt values");
}

// The same four rejections on every backend that answers, on a circuit all
// three can run.
#[test]
fn a_subsystem_is_a_proper_non_empty_set_of_distinct_qubits() {
    let mut circuit = Circuit::new(4, 0);
    for q in 0..4 {
        circuit.add_gate(Gate::Ry(0.5 * q as f64 + 0.2), &[q]);
    }
    let backends: [Box<dyn Backend>; 3] = [
        Box::new(StatevectorBackend::new(SEED)),
        Box::new(MpsBackend::new(SEED, 64)),
        Box::new(ProductStateBackend::new(SEED)),
    ];
    for mut backend in backends {
        sim::run_on(backend.as_mut(), &circuit).unwrap();
        let name = backend.name();
        assert!(
            matches!(
                backend.schmidt_values(&[]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took an empty subsystem"
        );
        assert!(
            matches!(
                backend.entanglement_entropy(&[0, 1, 2, 3]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took the whole register"
        );
        assert!(
            matches!(
                backend.schmidt_values(&[1, 1]),
                Err(PrismError::InvalidParameter { .. })
            ),
            "{name} took a repeated qubit"
        );
        assert_eq!(
            backend.schmidt_values(&[0, 4]).unwrap_err(),
            PrismError::InvalidQubit {
                index: 4,
                register_size: 4
            },
            "{name} took an out-of-range qubit"
        );
        assert!(
            backend.entanglement_entropy(&[3, 0]).is_ok(),
            "{name} rejected a valid cut"
        );
    }
}

/// The terminal's entropy for one explicit backend kind, with the spectrum it
/// carries checked for the ordering and normalization the contract states.
fn terminal_entropy(kind: BackendKind, circuit: &Circuit, subsystem: &[usize]) -> f64 {
    let result = simulate(circuit)
        .backend(kind)
        .seed(SEED)
        .entanglement_entropy(subsystem)
        .unwrap();
    let values = result
        .schmidt_values
        .as_deref()
        .expect("a backend answering from a spectrum returns it");
    assert!(
        values.windows(2).all(|pair| pair[0] >= pair[1]),
        "Schmidt values out of order: {values:?}"
    );
    let total: f64 = values.iter().map(|s| s * s).sum();
    assert!((total - 1.0).abs() < 1e-12, "squares sum to {total}");
    result.entropy
}

/// The terminal on each backend it can route to, against the statevector
/// kernel, at every cut of every case the backend accepts.
fn assert_terminal_matches_statevector(rows: &[(BackendKind, f64)], cases: &[CircuitCase]) {
    for case in cases {
        let circuit = case.circuit();
        let mut sv = StatevectorBackend::new(SEED);
        sim::run_on(&mut sv, &circuit).unwrap();
        for cut in 1..circuit.num_qubits {
            let subsystem: Vec<usize> = (0..cut).collect();
            let expected = sv.entanglement_entropy(&subsystem).unwrap();
            for (kind, eps) in rows {
                if matches!(kind, BackendKind::Mps { .. }) && !case.capabilities.safe_for_mps {
                    continue;
                }
                let entropy = terminal_entropy(kind.clone(), &circuit, &subsystem);
                assert!(
                    (entropy - expected).abs() < *eps,
                    "{kind:?} {} at cut {cut}: {entropy} against {expected}",
                    case.name
                );
            }
        }
    }
}

#[test]
fn the_terminal_matches_the_statevector_on_the_small_corpus() {
    assert_terminal_matches_statevector(
        &[
            (BackendKind::Statevector, SV_EPS),
            (BackendKind::Mps { max_bond_dim: 64 }, MPS_EPS),
        ],
        &exact_small_cases(),
    );
}

#[test]
fn the_terminal_matches_the_statevector_on_the_separable_corpus() {
    assert_terminal_matches_statevector(
        &[(BackendKind::ProductState, PRODUCT_EPS)],
        &product_separable_cases(),
    );
}

// A backend with no spectrum and no entropy of its own is named by the
// terminal, not by the route that selected it.
#[test]
fn the_terminal_names_the_backend_that_declines() {
    let circuit = ghz_circuit(4);
    for (kind, name, operation) in [
        (BackendKind::Sparse, "sparse", "Schmidt values"),
        (
            BackendKind::TensorNetwork,
            "tensornetwork",
            "Schmidt values",
        ),
        (BackendKind::Stabilizer, "stabilizer", "Schmidt values"),
        (BackendKind::Factored, "factored", "Schmidt values"),
        (
            BackendKind::DensityMatrix,
            "density_matrix",
            "Schmidt values of a mixed state",
        ),
    ] {
        assert_eq!(
            simulate(&circuit)
                .backend(kind)
                .seed(SEED)
                .entanglement_entropy(&[0, 1])
                .unwrap_err(),
            PrismError::BackendUnsupported {
                backend: name.to_string(),
                operation: operation.to_string(),
            }
        );
    }
}
