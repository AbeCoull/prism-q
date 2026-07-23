//! Adjoint-gradient correctness: `run_expectation_gradient` and
//! `Simulate::expectation_gradient` validated against central finite
//! differences and the parameter-shift rule, at fixed seed 42.

use prism_q::circuits;
use prism_q::{
    Circuit, Gate, Instruction, ParameterMap, PauliTerm, run_expectation_gradient,
    run_expectation_values, simulate,
};

const SEED: u64 = 42;

type Hamiltonian = Vec<(f64, Vec<PauliTerm>)>;

/// Weighted expectation value `Σ c_k ⟨P_k⟩` from the forward API.
fn expval(circuit: &Circuit, hamiltonian: &Hamiltonian) -> f64 {
    let observables: Vec<Vec<PauliTerm>> = hamiltonian.iter().map(|(_, p)| p.clone()).collect();
    let per_term = run_expectation_values(circuit, &observables, SEED).unwrap();
    hamiltonian
        .iter()
        .zip(per_term)
        .map(|((c, _), v)| c * v)
        .sum()
}

/// Return a copy of `circuit` with `delta` added to the angle of every gate
/// bound to parameter `slot`.
fn shift_slot(circuit: &Circuit, params: &ParameterMap, slot: usize, delta: f64) -> Circuit {
    let mut out = circuit.clone();
    for link in params.links().iter().filter(|l| l.param == slot) {
        if let Instruction::Gate { gate, .. } = &mut out.instructions[link.instruction] {
            *gate = shifted_gate(gate, delta);
        }
    }
    out
}

fn shifted_gate(gate: &Gate, delta: f64) -> Gate {
    match gate {
        Gate::Rx(t) => Gate::Rx(t + delta),
        Gate::Ry(t) => Gate::Ry(t + delta),
        Gate::Rz(t) => Gate::Rz(t + delta),
        Gate::Rzz(t) => Gate::Rzz(t + delta),
        Gate::P(t) => Gate::P(t + delta),
        other => panic!("gate {} is not differentiable", other.name()),
    }
}

/// Central finite-difference gradient of `⟨H⟩` for one slot.
fn finite_diff(
    circuit: &Circuit,
    hamiltonian: &Hamiltonian,
    params: &ParameterMap,
    slot: usize,
) -> f64 {
    let eps = 1e-5;
    let plus = expval(&shift_slot(circuit, params, slot, eps), hamiltonian);
    let minus = expval(&shift_slot(circuit, params, slot, -eps), hamiltonian);
    (plus - minus) / (2.0 * eps)
}

/// Parameter-shift gradient of `⟨H⟩` for one slot. Valid for Rx/Ry/Rz/Rzz/P
/// (generator eigenvalue gap 1, shift π/2, coefficient 1/2).
fn param_shift(
    circuit: &Circuit,
    hamiltonian: &Hamiltonian,
    params: &ParameterMap,
    slot: usize,
) -> f64 {
    let s = std::f64::consts::FRAC_PI_2;
    let plus = expval(&shift_slot(circuit, params, slot, s), hamiltonian);
    let minus = expval(&shift_slot(circuit, params, slot, -s), hamiltonian);
    (plus - minus) / 2.0
}

/// Mark every analytically differentiable gate as its own trainable slot.
fn all_rotations_trainable(circuit: &Circuit) -> ParameterMap {
    let mut params = ParameterMap::new();
    let mut slot = 0;
    for (i, inst) in circuit.instructions.iter().enumerate() {
        if let Instruction::Gate { gate, .. } = inst {
            if gate.pauli_generator().is_some() {
                params.push(i, slot);
                slot += 1;
            }
        }
    }
    params
}

#[test]
fn single_rx_matches_analytic() {
    let theta = 0.6;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    let mut params = ParameterMap::new();
    params.push(0, 0);

    let obs = vec![(1.0, vec![PauliTerm::z(0)])];
    let g = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    assert!((g.value - theta.cos()).abs() < 1e-12);
    assert!((g.gradient[0] - (-theta.sin())).abs() < 1e-9);
    assert!((g.gradient[0] - finite_diff(&c, &obs, &params, 0)).abs() < 1e-6);
    assert!((g.gradient[0] - param_shift(&c, &obs, &params, 0)).abs() < 1e-9);
}

#[test]
fn hea_multiterm_hamiltonian_all_params() {
    let c = circuits::hardware_efficient_ansatz(4, 2, SEED);
    let params = all_rotations_trainable(&c);
    let obs: Hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (0.7, vec![PauliTerm::x(0)]),
        (0.5, vec![PauliTerm::y(2)]),
        (-0.3, vec![PauliTerm::z(3)]),
    ];

    let g = simulate(&c)
        .seed(SEED)
        .expectation_gradient(&obs, &params)
        .unwrap();

    assert!((g.value - expval(&c, &obs)).abs() < 1e-10);
    assert_eq!(g.gradient.len(), params.num_params());
    for slot in 0..params.num_params() {
        let fd = finite_diff(&c, &obs, &params, slot);
        let ps = param_shift(&c, &obs, &params, slot);
        assert!(
            (g.gradient[slot] - fd).abs() < 1e-6,
            "slot {slot}: adjoint {} vs finite-diff {fd}",
            g.gradient[slot]
        );
        assert!(
            (g.gradient[slot] - ps).abs() < 1e-9,
            "slot {slot}: adjoint {} vs param-shift {ps}",
            g.gradient[slot]
        );
    }
}

#[test]
fn qaoa_layer_rzz_and_rx() {
    let c = circuits::qaoa_circuit(6, 1, SEED);
    let params = all_rotations_trainable(&c);
    let obs: Hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (1.0, vec![PauliTerm::z(2), PauliTerm::z(3)]),
        (0.4, vec![PauliTerm::x(5)]),
    ];

    let g = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    assert!((g.value - expval(&c, &obs)).abs() < 1e-10);
    for slot in 0..params.num_params() {
        let fd = finite_diff(&c, &obs, &params, slot);
        assert!(
            (g.gradient[slot] - fd).abs() < 1e-6,
            "slot {slot}: adjoint {} vs finite-diff {fd}",
            g.gradient[slot]
        );
    }
}

#[test]
fn value_matches_forward_expectation() {
    let c = circuits::hardware_efficient_ansatz(5, 3, SEED);
    let params = all_rotations_trainable(&c);
    let obs: Hamiltonian = vec![
        (1.2, vec![PauliTerm::z(0)]),
        (-0.5, vec![PauliTerm::x(1), PauliTerm::x(2)]),
    ];
    let g = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    assert!((g.value - expval(&c, &obs)).abs() < 1e-10);
}

#[test]
fn builder_records_and_differentiates() {
    use prism_q::CircuitBuilder;
    let theta = 0.9;
    let mut b = CircuitBuilder::new(2);
    b.h(0).rz_param(0, theta, 0).cx(0, 1).ry_param(1, 0.4, 1);
    let (circuit, params) = b.build_parametric();

    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(0), PauliTerm::z(1)])];
    let g = run_expectation_gradient(&circuit, &obs, &params, SEED).unwrap();
    assert_eq!(g.gradient.len(), 2);
    for slot in 0..2 {
        let fd = finite_diff(&circuit, &obs, &params, slot);
        assert!((g.gradient[slot] - fd).abs() < 1e-6);
    }
}

#[test]
fn out_of_cone_gate_has_zero_gradient() {
    // Observable Z0; an Rx on qubit 2 has no path to qubit 0, so its gradient
    // is exactly zero (light-cone skip), while the in-cone Rx on qubit 0 is
    // -sin(theta).
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Rx(0.5), &[0]);
    c.add_gate(Gate::Rx(0.7), &[2]);
    let mut params = ParameterMap::new();
    params.push(0, 0);
    params.push(1, 1);

    let obs = vec![(1.0, vec![PauliTerm::z(0)])];
    let g = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    assert!((g.gradient[0] - (-0.5f64.sin())).abs() < 1e-9);
    assert_eq!(g.gradient[1], 0.0);
    assert!((g.gradient[1] - finite_diff(&c, &obs, &params, 1)).abs() < 1e-6);
}

#[test]
fn nontrainable_prefix_is_skipped_correctly() {
    // A fixed entangling prefix (H, CX, CX) precedes the trainable ansatz. The
    // backward sweep must stop at the earliest trainable gate; the gradient
    // still matches finite differences.
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Ry(0.6), &[0]);
    c.add_gate(Gate::Rz(0.4), &[1]);
    c.add_gate(Gate::Rx(0.9), &[2]);
    let mut params = ParameterMap::new();
    params.push(3, 0);
    params.push(4, 1);
    params.push(5, 2);

    let obs: Hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(2)]),
        (0.5, vec![PauliTerm::x(1)]),
    ];
    let g = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    for slot in 0..3 {
        let fd = finite_diff(&c, &obs, &params, slot);
        assert!(
            (g.gradient[slot] - fd).abs() < 1e-6,
            "slot {slot}: adjoint {} vs finite-diff {fd}",
            g.gradient[slot]
        );
    }
}

#[test]
fn out_of_range_link_is_rejected() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(0.3), &[0]);
    let mut params = ParameterMap::new();
    params.push(5, 0);
    let obs = vec![(1.0, vec![PauliTerm::z(0)])];
    assert!(run_expectation_gradient(&c, &obs, &params, SEED).is_err());
}
