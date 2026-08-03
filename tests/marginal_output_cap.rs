//! Marginals either need the dense gather or they do not. Isolated in its own
//! test binary: it overrides `PRISM_MAX_PROB_QUBITS`, which the cap helper
//! caches per process.

use std::sync::Once;

use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, simulate};

const SEED: u64 = 42;
const PROB_CAP: usize = 4;
const N: usize = PROB_CAP + 2;
const EPS: f64 = 1e-12;

fn small_prob_cap() {
    static SET: Once = Once::new();
    SET.call_once(|| {
        // SAFETY: set exactly once, and every reader in this binary is gated
        // behind this `Once`, so no thread queries the cap while it is written.
        unsafe { std::env::set_var("PRISM_MAX_PROB_QUBITS", "4") };
    });
}

fn assert_marginals(kind: BackendKind, circuit: &Circuit, want: &[(f64, f64)], label: &str) {
    let got = simulate(circuit)
        .backend(kind)
        .seed(SEED)
        .marginals()
        .expect("marginals come from per-qubit expectations, not the dense vector")
        .into_vec();
    assert_eq!(got.len(), want.len(), "{label}");
    for (q, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g.0 - w.0).abs() < EPS && (g.1 - w.1).abs() < EPS,
            "{label} marginal[{q}]: got {g:?}, want {w:?}"
        );
    }
}

// GHZ keeps the whole register in one component, so the route is direct
// resolution onto a single backend. Every qubit is an even mixture of |0> and
// |1>, which needs no reference run (the reference would hit the same cap).
#[test]
fn direct_route_marginals_answer_past_the_dense_cap() {
    small_prob_cap();
    let mut circuit = Circuit::new(N, 0);
    circuit.add_gate(Gate::H, &[0]);
    for q in 1..N {
        circuit.add_gate(Gate::Cx, &[q - 1, q]);
    }

    let want = vec![(0.5, 0.5); N];
    assert_marginals(BackendKind::Sparse, &circuit, &want, "sparse");
    assert_marginals(
        BackendKind::Mps { max_bond_dim: 64 },
        &circuit,
        &want,
        "mps",
    );
}

// No entangling gate, so the route is subsystem decomposition, whose merge
// builds the 2^n distribution the cap rejects. The product state is the one
// backend carried past that split.
#[test]
fn product_state_marginals_answer_past_the_dense_cap() {
    small_prob_cap();
    let mut circuit = Circuit::new(N, 0);
    circuit.add_gate(Gate::X, &[0]);
    for q in 1..N {
        circuit.add_gate(Gate::H, &[q]);
    }

    let mut want = vec![(0.5, 0.5); N];
    want[0] = (0.0, 1.0);
    assert_marginals(BackendKind::ProductState, &circuit, &want, "product");
}
