//! Direct-circuit tests for the joint-observable SPP/SPD API.

use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{run_spd_observable, run_spp_observable, PauliAxis, PauliTerm};

const SEED: u64 = 0xDEAD_BEEF;

#[test]
fn spp_recovers_single_qubit_z_expectation_on_h_t_h_circuit() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::H, &[0]);

    let observable = [PauliTerm::z(0)];
    let result = run_spp_observable(&circuit, &observable, 8_000, SEED).unwrap();
    // ⟨0|H T H Z H T† H|0⟩ = ⟨0| (HTH)† Z (HTH) |0⟩ = cos(π/4) = 1/√2
    let expected = 1.0 / std::f64::consts::SQRT_2;
    assert!(
        (result.mean - expected).abs() < 0.05,
        "SPP mean {:.4} diverged from expected {expected:.4}",
        result.mean
    );
}

#[test]
fn spd_matches_spp_on_two_qubit_zz_observable() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::T, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::H, &[0]);

    let observable = [PauliTerm::z(0), PauliTerm::z(1)];
    let spd = run_spd_observable(&circuit, &observable, 1e-10, 1024).unwrap();
    let spp = run_spp_observable(&circuit, &observable, 16_000, SEED).unwrap();
    assert!(
        (spd.mean - spp.mean).abs() < 0.05,
        "SPD mean {:.4} diverged from SPP mean {:.4}",
        spd.mean,
        spp.mean
    );
}

#[test]
fn spp_handles_x_and_y_pauli_factors() {
    // Pure Clifford: ⟨+|X|+⟩ = +1. Use Y to verify mixed PauliVec layout.
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);

    let x_result = run_spp_observable(&circuit, &[PauliTerm::x(0)], 1_000, SEED).unwrap();
    assert!(
        (x_result.mean - 1.0).abs() < 0.05,
        "⟨X⟩ on |+⟩ should be +1, got {:.4}",
        x_result.mean
    );

    // ⟨+|Y|+⟩ = 0.
    let y_result = run_spp_observable(&circuit, &[PauliTerm::y(0)], 2_000, SEED).unwrap();
    assert!(
        y_result.mean.abs() < 0.05,
        "⟨Y⟩ on |+⟩ should be 0, got {:.4}",
        y_result.mean
    );
}

#[test]
fn y_observable_uses_physical_pauli_phase() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::S, &[0]);

    let spd = run_spd_observable(&circuit, &[PauliTerm::y(0)], 0.0, 0).unwrap();
    let spp = run_spp_observable(&circuit, &[PauliTerm::y(0)], 2_000, SEED).unwrap();

    assert!(
        (spd.mean - 1.0).abs() < 1e-12,
        "<Y> on S H |0> should be +1, got {:.4}",
        spd.mean
    );
    assert!(
        (spp.mean - 1.0).abs() < 0.05,
        "SPP <Y> on S H |0> should be +1, got {:.4}",
        spp.mean
    );
}

#[test]
fn stochastic_t_branch_preserves_y_expectation() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::T, &[0]);

    let expected = 1.0 / std::f64::consts::SQRT_2;
    let spd = run_spd_observable(&circuit, &[PauliTerm::y(0)], 0.0, 0).unwrap();
    let spp = run_spp_observable(&circuit, &[PauliTerm::y(0)], 12_000, SEED).unwrap();

    assert!(
        (spd.mean - expected).abs() < 1e-12,
        "SPD <Y> on T H |0> should be {expected:.4}, got {:.4}",
        spd.mean
    );
    assert!(
        (spp.mean - expected).abs() < 0.05,
        "SPP <Y> on T H |0> should be {expected:.4}, got {:.4}",
        spp.mean
    );
}

#[test]
fn pauli_term_constructor_helpers_match_axis_enum() {
    assert_eq!(PauliTerm::x(0).axis, PauliAxis::X);
    assert_eq!(PauliTerm::y(3).axis, PauliAxis::Y);
    assert_eq!(PauliTerm::z(7).axis, PauliAxis::Z);
    assert_eq!(PauliTerm::z(7).qubit, 7);
}

#[test]
fn invalid_qubit_index_is_rejected() {
    let circuit = Circuit::new(2, 0);
    let err = run_spp_observable(&circuit, &[PauliTerm::z(5)], 10, SEED)
        .expect_err("out-of-range qubit must error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("InvalidQubit"),
        "expected InvalidQubit, got {msg}"
    );
}

#[test]
fn duplicate_pauli_factors_are_rejected() {
    let circuit = Circuit::new(2, 0);
    let observable = [PauliTerm::z(0), PauliTerm::z(0)];

    for msg in [
        format!(
            "{:?}",
            run_spp_observable(&circuit, &observable, 10, SEED)
                .expect_err("SPP must reject duplicate Pauli factors")
        ),
        format!(
            "{:?}",
            run_spd_observable(&circuit, &observable, 0.0, 0)
                .expect_err("SPD must reject duplicate Pauli factors")
        ),
    ] {
        assert!(
            msg.contains("duplicate factor"),
            "expected duplicate-factor rejection, got {msg}"
        );
    }
}

#[test]
fn spd_rejects_non_clifford_non_t_gates() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rx(0.37), &[0]);

    let err = run_spd_observable(&circuit, &[PauliTerm::z(0)], 0.0, 0)
        .expect_err("SPD must reject gates outside Clifford+T");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("non-Clifford+T"),
        "expected non-Clifford+T rejection, got {msg}"
    );
}
