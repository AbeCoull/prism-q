//! Bench circuits that `tests/integration/bench_fixture_routing.rs` also builds, so its
//! routing checks run on the circuits the rows time.

use prism_q::circuit::Circuit;
use prism_q::gates::Gate;

/// `depth` layers of H on every qubit, then CX on every fourth bond of the chain, the
/// first bond advancing one per layer: a quarter of the dense layer's CX, with the
/// whole chain connected from `depth >= 4` so the decomposed route cannot claim it.
pub fn sparse_entanglement_circuit(n_qubits: usize, depth: usize) -> Circuit {
    let mut circuit = Circuit::new(n_qubits, 0);

    for layer in 0..depth {
        for q in 0..n_qubits {
            circuit.add_gate(Gate::H, &[q]);
        }
        for q in (layer % 4..n_qubits.saturating_sub(1)).step_by(4) {
            circuit.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }

    circuit
}
