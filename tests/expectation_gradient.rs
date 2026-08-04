//! Gradient correctness: the adjoint entry points validated against central
//! finite differences and a hand-rolled parameter-shift rule, and the shipped
//! parameter-shift path validated against the adjoint, at the fixed test seed.

mod common;

use common::SEED;
use num_complex::Complex64;
use prism_q::circuits;
use prism_q::{
    BackendKind, Circuit, Gate, Instruction, ParameterMap, PauliTerm, run_expectation_gradient,
    run_expectation_gradient_shift, run_expectation_values, simulate,
};

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
    let params = ParameterMap::all_rotations(&c);
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
    let params = ParameterMap::all_rotations(&c);
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
    let params = ParameterMap::all_rotations(&c);
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
    b.h(0)
        .rz(theta, 0)
        .trainable(0)
        .cx(0, 1)
        .ry(0.4, 1)
        .trainable(1);
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
fn builder_shared_slot_accumulates() {
    use prism_q::CircuitBuilder;
    // Two Rx gates on separate qubits sharing slot 0 via the builder marker.
    let theta = 0.4;
    let mut b = CircuitBuilder::new(2);
    b.rx(theta, 0).trainable(0).rx(theta, 1).trainable(0);
    let (circuit, params) = b.build_parametric();
    assert_eq!(params.num_params(), 1);

    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(0)]), (1.0, vec![PauliTerm::z(1)])];
    let g = run_expectation_gradient(&circuit, &obs, &params, SEED).unwrap();
    assert!((g.gradient[0] - (-2.0 * theta.sin())).abs() < 1e-9);
    assert!((g.gradient[0] - finite_diff(&circuit, &obs, &params, 0)).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "differentiable gate")]
fn trainable_on_nondifferentiable_gate_panics() {
    use prism_q::CircuitBuilder;
    CircuitBuilder::new(1).h(0).trainable(0);
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
fn shift_matches_adjoint_on_statevector() {
    let c = circuits::hardware_efficient_ansatz(4, 2, SEED);
    let params = ParameterMap::all_rotations(&c);
    let obs: Hamiltonian = vec![
        (1.0, vec![PauliTerm::z(0), PauliTerm::z(1)]),
        (0.7, vec![PauliTerm::x(0)]),
        (0.5, vec![PauliTerm::y(2)]),
        (-0.3, vec![PauliTerm::z(3)]),
    ];

    let adjoint = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    let shift = simulate(&c)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_gradient_shift(&obs, &params)
        .unwrap();

    assert!((shift.value - adjoint.value).abs() < 1e-12);
    assert_eq!(shift.gradient.len(), params.num_params());
    for slot in 0..params.num_params() {
        assert!(
            (shift.gradient[slot] - adjoint.gradient[slot]).abs() < 1e-9,
            "slot {slot}: shift {} vs adjoint {}",
            shift.gradient[slot],
            adjoint.gradient[slot]
        );
    }
}

#[test]
fn shift_matches_adjoint_on_phase_and_rzz() {
    // P(theta) = e^{i theta/2} Rz(theta) and the scalar cancels in <H>, so P
    // takes the same pi/2 shift as the rotations despite its {0, 1} generator
    // eigenvalues.
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::P(0.9), &[0]);
    c.add_gate(Gate::Rzz(0.4), &[0, 1]);
    let mut params = ParameterMap::new();
    params.push(2, 0);
    params.push(3, 1);

    let obs: Hamiltonian = vec![
        (1.0, vec![PauliTerm::x(0)]),
        (0.6, vec![PauliTerm::y(0), PauliTerm::x(1)]),
    ];
    let adjoint = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    let shift = run_expectation_gradient_shift(&c, &obs, &params, SEED).unwrap();
    for slot in 0..2 {
        assert!(
            (shift.gradient[slot] - adjoint.gradient[slot]).abs() < 1e-9,
            "slot {slot}: shift {} vs adjoint {}",
            shift.gradient[slot],
            adjoint.gradient[slot]
        );
    }
}

#[test]
fn shift_accumulates_a_shared_slot() {
    // Two Rx on one qubit bound to a single slot: <Z> = cos(2 theta), so the
    // shared gradient is -2 sin(2 theta). Each bound gate must be shifted on
    // its own; shifting both at once gives zero here.
    let theta = 0.35;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    c.add_gate(Gate::Rx(theta), &[0]);
    let mut params = ParameterMap::new();
    params.push(0, 0);
    params.push(1, 0);

    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
    let shift = run_expectation_gradient_shift(&c, &obs, &params, SEED).unwrap();
    assert_eq!(shift.gradient.len(), 1);
    assert!((shift.gradient[0] - (-2.0 * (2.0 * theta).sin())).abs() < 1e-9);

    let adjoint = run_expectation_gradient(&c, &obs, &params, SEED).unwrap();
    assert!((shift.gradient[0] - adjoint.gradient[0]).abs() < 1e-12);
}

#[test]
fn shift_differentiates_a_circuit_holding_a_qft_block() {
    // The adjoint rejects QftBlock outright; parameter shift differentiates
    // through it because it only ever calls the forward observable path.
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::Rx(0.4), &[0]);
    c.add_gate(Gate::QftBlock { start: 0, num: 3 }, &[0, 1, 2]);
    let mut params = ParameterMap::new();
    params.push(0, 0);

    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(2)])];
    assert!(run_expectation_gradient(&c, &obs, &params, SEED).is_err());

    let shift = run_expectation_gradient_shift(&c, &obs, &params, SEED).unwrap();
    assert!((shift.gradient[0] - finite_diff(&c, &obs, &params, 0)).abs() < 1e-6);
}

#[test]
fn shift_differentiates_from_a_start_state() {
    // The adjoint declines a start state: its backward pass inverts the circuit
    // back to |0...0>. Parameter shift only ever runs forward. Starting from
    // |1> on q0, Rx(theta) gives <Z0> = -cos(theta), so the gradient is
    // sin(theta).
    let theta = 0.8;
    let mut c = Circuit::new(1, 0);
    c.add_gate(Gate::Rx(theta), &[0]);
    let mut params = ParameterMap::new();
    params.push(0, 0);

    let start = [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
    let g = simulate(&c)
        .initial_state(&start)
        .seed(SEED)
        .expectation_gradient_shift(&obs, &params)
        .unwrap();

    assert!((g.value - (-theta.cos())).abs() < 1e-12);
    assert!((g.gradient[0] - theta.sin()).abs() < 1e-9);
}

#[test]
fn shift_on_a_backend_without_an_observable_path_names_it() {
    let mut c = Circuit::new(2, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Rx(0.3), &[0]);
    let mut params = ParameterMap::new();
    params.push(2, 0);

    let obs: Hamiltonian = vec![(1.0, vec![PauliTerm::z(0)])];
    let err = simulate(&c)
        .backend(BackendKind::TensorNetwork)
        .seed(SEED)
        .expectation_gradient_shift(&obs, &params)
        .unwrap_err()
        .to_string();
    assert!(err.contains("tensornetwork"), "{err}");
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
