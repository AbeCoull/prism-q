//! Width ceiling of the sparse basis index.
//!
//! Sparse gate kernels mask the global basis index with `1usize << qubit`, so a
//! wider circuit aliases qubits onto each other. Release builds mask the shift
//! count and return a wrong answer; debug builds panic. Every assertion here is
//! on the rejection or on the routing, so both profiles test the same thing.

use prism_q::backend::sparse::SparseBackend;
use prism_q::gates::Gate;
use prism_q::sim::ResolvedBackend;
use prism_q::{
    BackendKind, Circuit, NoiseModel, PauliObservable, PauliTerm, PrismError, run_on, sim,
};

const WIDTH: usize = usize::BITS as usize;

// Sparse-friendly (every gate is diagonal or a permutation), entangling, and
// not Clifford, so `Auto` reaches the width branch rather than the stabilizer
// or Clifford+T routes. The CX chain spans the register: a fixture that gates
// only a few qubits leaves the rest as isolated singletons and runs on the
// decomposed route instead of the family under test.
fn phase_chain(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 2);
    c.add_gate(Gate::X, &[0]);
    for q in 1..n {
        c.add_gate(Gate::Cx, &[q - 1, q]);
    }
    c.add_gate(Gate::P(0.3), &[n - 1]);
    c
}

#[test]
fn sparse_rejects_a_circuit_wider_than_the_basis_index() {
    let circuit = phase_chain(WIDTH + 1);
    let mut backend = SparseBackend::new(42);
    match run_on(&mut backend, &circuit).unwrap_err() {
        PrismError::IncompatibleBackend { backend, reason } => {
            assert_eq!(backend, "sparse");
            assert!(reason.contains("basis-index width"), "reason: {reason}");
        }
        other => panic!("expected an incompatible-backend rejection, got {other:?}"),
    }
}

// The widest circuit the index addresses uses qubit `WIDTH - 1`, whose mask is
// the last shift that does not wrap. Guarding one qubit lower would drop a
// width the representation handles correctly.
#[test]
fn sparse_runs_at_the_widest_addressable_circuit() {
    let mut circuit = phase_chain(WIDTH);
    circuit.add_measure(WIDTH - 1, 0);
    circuit.add_measure(WIDTH - 2, 1);
    let mut backend = SparseBackend::new(42);
    let outcome = run_on(&mut backend, &circuit).expect("a circuit at the index width must run");
    assert!(
        outcome.classical_bits[0],
        "the chain must carry the flip to qubit {}, the last addressable index",
        WIDTH - 1
    );
    assert!(outcome.classical_bits[1], "and to the qubit below it");
}

// Auto routes wide sparse-friendly circuits to Sparse. Past the index width the
// alternative representation has to carry them, or the rejection above turns a
// regime Auto serves into an error.
#[test]
fn auto_routes_past_the_index_width_to_mps() {
    let circuit = phase_chain(WIDTH + 6);
    let outcome = sim::simulate(&circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .run()
        .expect("auto must serve a circuit past the sparse index width");
    assert_eq!(outcome.metadata.backend, ResolvedBackend::Mps);
}

// The noisy Auto route repeats the width branch of the exact one, on its own
// selection function.
#[test]
fn noisy_auto_routes_past_the_index_width_to_mps() {
    let circuit = phase_chain(WIDTH + 6);
    // The route is reached only by noise that is not Pauli-only.
    let noise = NoiseModel::with_amplitude_damping(&circuit, 0.05);
    let shots = sim::simulate(&circuit)
        .backend(BackendKind::Auto)
        .seed(42)
        .noise(&noise)
        .shots(2)
        .expect("auto must serve a noisy circuit past the sparse index width");
    assert_eq!(shots.metadata.backend, ResolvedBackend::Mps);
}

// Every observable terminal reduces its Pauli factors to `1 << qubit` masks.
// Each one has to reach the width rejection before that shift runs, or a debug
// build panics where the documented error belongs.
#[test]
fn wide_observable_terminals_reject_rather_than_shift() {
    let circuit = phase_chain(WIDTH + 6);
    let top = WIDTH + 5;

    let values = sim::simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .expectation_values(&[vec![PauliTerm::z(top)]]);
    assert!(values.is_err(), "expectation_values must reject");

    let observable = PauliObservable::from_terms([(1.0, vec![PauliTerm::z(top)])]).unwrap();
    let grouped = sim::simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(42)
        .observable_expectation(&observable);
    assert!(grouped.is_err(), "observable_expectation must reject");
}
