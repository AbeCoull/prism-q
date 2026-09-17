//! The `Auto` expectation and marginals terminals take the scalar tensor path
//! on a wide, shallow, unitary circuit whose bounded plan fits, and stay on
//! the statevector otherwise.

use prism_q::circuits::{brickwork_circuit, hardware_efficient_ansatz, random_circuit};
use prism_q::{BackendKind, Circuit, PauliTerm, ResolvedBackend, simulate};

const SEED: u64 = 0xDEAD_BEEF;

fn two_local(n: usize) -> Vec<Vec<PauliTerm>> {
    (0..8)
        .map(|k| {
            let a = (k * n / 8) % n;
            let b = (a + n / 3 + 1) % n;
            vec![PauliTerm::z(a), PauliTerm::z(b)]
        })
        .collect()
}

fn auto_backend(circuit: &Circuit, observables: &[Vec<PauliTerm>]) -> ResolvedBackend {
    simulate(circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .expectation_values_reported(observables)
        .unwrap()
        .metadata
        .backend
}

#[test]
fn auto_expectations_route_a_wide_shallow_circuit_to_the_tensor_path() {
    let circuit = random_circuit(20, 10, SEED);
    let observables = two_local(20);
    let auto = simulate(&circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .expectation_values_reported(&observables)
        .unwrap();
    assert_eq!(auto.metadata.backend, ResolvedBackend::TensorNetwork);

    let dense = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&observables)
        .unwrap();
    for (got, want) in auto.values.iter().zip(&dense) {
        assert!((got - want).abs() < 1e-10, "{got} vs {want}");
    }
}

#[test]
fn auto_marginals_route_a_wide_shallow_circuit_to_the_tensor_path() {
    let circuit = brickwork_circuit(20, 4, SEED);
    let auto = simulate(&circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .marginals()
        .unwrap();
    assert_eq!(auto.metadata.backend, ResolvedBackend::TensorNetwork);

    let dense = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .marginals()
        .unwrap();
    assert_eq!(auto.marginals.len(), dense.marginals.len());
    for ((p0, p1), (q0, q1)) in auto.marginals.iter().zip(&dense.marginals) {
        assert!((p0 - q0).abs() < 1e-10 && (p1 - q1).abs() < 1e-10);
    }
}

#[test]
fn a_deep_circuit_stays_on_the_statevector() {
    let circuit = hardware_efficient_ansatz(20, 8, SEED);
    assert_eq!(
        auto_backend(&circuit, &two_local(20)),
        ResolvedBackend::Statevector
    );
}

#[test]
fn a_narrow_circuit_stays_on_the_statevector() {
    let circuit = random_circuit(16, 10, SEED);
    assert_eq!(
        auto_backend(&circuit, &two_local(16)),
        ResolvedBackend::Statevector
    );
}

#[test]
fn a_measured_circuit_keeps_its_marginals_off_the_tensor_path() {
    let mut circuit = random_circuit(20, 10, SEED);
    circuit.num_classical_bits = 1;
    circuit.add_measure(0, 0);
    let auto = simulate(&circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .marginals()
        .unwrap();
    assert_ne!(auto.metadata.backend, ResolvedBackend::TensorNetwork);
}
