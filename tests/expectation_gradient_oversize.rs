//! Parameter-shift gradients above the statevector cap. Isolated in its own
//! test binary: it overrides `PRISM_MAX_SV_QUBITS`, which is cached per
//! process. The adjoint cannot run at this width, so the reference is closed
//! form rather than a statevector comparison.

use prism_q::gates::Gate;
use prism_q::{BackendKind, Circuit, ParameterMap, PauliTerm, run_expectation_gradient, simulate};

const SEED: u64 = 42;
const TOL: f64 = 1e-9;

struct Fixture {
    circuit: Circuit,
    params: ParameterMap,
    hamiltonian: Vec<(f64, Vec<PauliTerm>)>,
    want: Vec<f64>,
}

// Rx(theta_q) on every qubit, then CX(0->1) and CX(2->3). Under the Heisenberg
// conjugation of CX(0->1), Z0 is unchanged and Z0*Z1 becomes Z1, so
// <H> = cos(theta_0) + 0.5 cos(theta_1) for H = Z0 + 0.5 Z0 Z1. Slots 0 and 1
// therefore carry -sin(theta_0) and -0.5 sin(theta_1); every other slot is
// exactly zero.
fn fixture(n: usize) -> Fixture {
    let mut circuit = Circuit::new(n, 0);
    let angles: Vec<f64> = (0..n).map(|q| 0.2 + 0.1 * q as f64).collect();
    for (q, &theta) in angles.iter().enumerate() {
        circuit.add_gate(Gate::Rx(theta), &[q]);
    }
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[2, 3]);

    let params = ParameterMap::all_rotations(&circuit);
    let hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0)]),
        (0.5, vec![PauliTerm::z(0), PauliTerm::z(1)]),
    ];
    let mut want = vec![0.0; n];
    want[0] = -angles[0].sin();
    want[1] = -0.5 * angles[1].sin();
    Fixture {
        circuit,
        params,
        hamiltonian,
        want,
    }
}

fn assert_gradient(label: &str, gradient: &[f64], want: &[f64]) {
    assert_eq!(gradient.len(), want.len());
    for (slot, (&got, &expected)) in gradient.iter().zip(want).enumerate() {
        assert!(
            (got - expected).abs() < TOL,
            "{label} slot {slot}: got {got}, want {expected}"
        );
    }
}

#[test]
fn shift_gradient_above_the_cap_on_mps_and_factored() {
    // SAFETY: single test in this binary; the variable is set before any cap
    // query and no other thread is running.
    unsafe { std::env::set_var("PRISM_MAX_SV_QUBITS", "4") };

    let f = fixture(8);

    // The adjoint declines at this width: it holds two statevectors.
    assert!(run_expectation_gradient(&f.circuit, &f.hamiltonian, &f.params, SEED).is_err());

    for kind in [BackendKind::Mps { max_bond_dim: 16 }, BackendKind::Factored] {
        let label = format!("{kind:?}");
        let g = simulate(&f.circuit)
            .backend(kind)
            .seed(SEED)
            .expectation_gradient_shift(&f.hamiltonian, &f.params)
            .unwrap();
        assert_gradient(&label, &g.gradient, &f.want);
    }

    // Auto routes past the cap to a native observable path of its own.
    let g = simulate(&f.circuit)
        .seed(SEED)
        .expectation_gradient_shift(&f.hamiltonian, &f.params)
        .unwrap();
    assert_gradient("auto", &g.gradient, &f.want);
}
