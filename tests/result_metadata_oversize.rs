//! Cap-crossing half of the contract that a result discloses an approximation.
//! Isolated in its own test binary: it overrides `PRISM_MAX_SV_QUBITS`, which is
//! cached per process, so it must not share a process with tests expecting the
//! real cap.

use prism_q::{Circuit, Gate, PrismError, ResolvedBackend, simulate};

const SEED: u64 = 42;
const CAP: usize = 4;

fn set_cap() {
    // SAFETY: this binary runs cap-crossing tests only, the variable is set
    // before any cap query, and the value is the same at every call site.
    unsafe { std::env::set_var("PRISM_MAX_SV_QUBITS", "4") };
}

/// Not sparse-friendly, so Auto takes the MPS branch of the oversize split
/// rather than the sparse one.
fn oversize_dense(num_qubits: usize) -> Circuit {
    let mut circuit = Circuit::new(num_qubits, 0);
    for q in 0..num_qubits {
        circuit.add_gate(Gate::H, &[q]);
        circuit.add_gate(Gate::T, &[q]);
    }
    for q in 0..num_qubits - 1 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in 0..num_qubits {
        circuit.add_gate(Gate::T, &[q]);
    }
    circuit
}

// The dispatch-level half of the approximation contract: the same query above
// and below the statevector cap must differ in the flag, asserted at the
// `Simulate` level rather than on the backend.
#[test]
fn auto_reports_approximate_above_the_statevector_cap() {
    set_cap();
    let below = simulate(&oversize_dense(CAP - 2))
        .seed(SEED)
        .marginals()
        .unwrap();
    let above = simulate(&oversize_dense(CAP + 2))
        .seed(SEED)
        .marginals()
        .unwrap();

    assert!(below.metadata.is_exact(), "{:?}", below.metadata);
    assert!(!above.metadata.is_exact(), "{:?}", above.metadata);
    assert_eq!(above.metadata.backend, ResolvedBackend::Mps);
}

#[test]
fn require_exact_rejects_the_approximate_route() {
    set_cap();
    let circuit = oversize_dense(CAP + 2);
    let err = simulate(&circuit)
        .seed(SEED)
        .require_exact()
        .marginals()
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. } if backend == "Mps"),
        "{err:?}"
    );

    // The same builder without the requirement answers.
    assert!(simulate(&circuit).seed(SEED).marginals().is_ok());
}

// Sparse stabilizer-rank approximation is chosen ahead of the family tree, so
// the tree never sees it and the route check has to cover it separately.
#[test]
fn require_exact_rejects_the_approximate_stabilizer_rank_route() {
    set_cap();
    let mut circuit = Circuit::new(6, 0);
    for q in 0..6 {
        circuit.add_gate(Gate::H, &[q]);
    }
    for q in 0..5 {
        circuit.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for _ in 0..4 {
        for q in 0..6 {
            circuit.add_gate(Gate::T, &[q]);
        }
    }

    let err = simulate(&circuit)
        .seed(SEED)
        .require_exact()
        .run()
        .unwrap_err();
    assert!(
        matches!(&err, PrismError::IncompatibleBackend { backend, .. }
            if backend == "StabilizerRank" || backend == "Mps"),
        "{err:?}"
    );
    assert!(simulate(&circuit).seed(SEED).run().is_ok());
}
