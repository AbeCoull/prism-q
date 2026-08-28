//! Direct-circuit tests for the joint-observable SPP/SPD API.

mod common;

use common::SEED;
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::{
    BackendKind, PauliAxis, PauliTerm, run_spd_observable, run_spp_observable, simulate,
};

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
fn spd_rejects_gates_outside_the_rotation_family() {
    let mut circuit = Circuit::new(2, 0);
    let phase = num_complex::Complex64::from_polar(1.0, 0.37);
    let zero = num_complex::Complex64::new(0.0, 0.0);
    let one = num_complex::Complex64::new(1.0, 0.0);
    circuit.add_gate(Gate::Cu(Box::new([[one, zero], [zero, phase]])), &[0, 1]);

    let err = run_spd_observable(&circuit, &[PauliTerm::z(0)], 0.0, 0)
        .expect_err("SPD must reject gates it cannot branch or lower");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("cu"),
        "rejection should name the gate, got {msg}"
    );

    let mut supported = Circuit::new(1, 0);
    supported.add_gate(Gate::Rz(0.37), &[0]);
    run_spd_observable(&supported, &[PauliTerm::z(0)], 0.0, 0)
        .expect("SPD must accept a Z-axis rotation");
}

// Rx and Ry lower to Clifford conjugation around one Rz, so both engines
// accept them; correctness is pinned by the statevector comparisons below.
#[test]
fn engines_accept_arbitrary_axis_rotations() {
    let mut circuit = Circuit::new(1, 0);
    circuit.add_gate(Gate::Rx(0.37), &[0]);
    circuit.add_gate(Gate::Ry(-1.1), &[0]);

    run_spd_observable(&circuit, &[PauliTerm::z(0)], 0.0, 0)
        .expect("SPD must accept Rx and Ry via lowering");
    run_spp_observable(&circuit, &[PauliTerm::z(0)], 100, SEED)
        .expect("SPP must accept Rx and Ry via lowering");
}

/// The generalized rotation branch must reduce to the T rule exactly, not
/// approximately: both engines take the same code path at `theta = pi/4`, so a
/// last-ulp difference would mean the generalization, not the rounding, is off.
#[test]
fn rz_at_quarter_pi_is_bit_identical_to_t() {
    let observable = [PauliTerm::z(0), PauliTerm::y(1)];
    for (angle, t_gate) in [
        (std::f64::consts::FRAC_PI_4, Gate::T),
        (-std::f64::consts::FRAC_PI_4, Gate::Tdg),
    ] {
        let build = |rot: &Gate| {
            let mut c = Circuit::new(2, 0);
            c.add_gate(Gate::H, &[0]);
            c.add_gate(Gate::H, &[1]);
            c.add_gate(rot.clone(), &[0]);
            c.add_gate(Gate::Cx, &[0, 1]);
            c.add_gate(rot.clone(), &[1]);
            c.add_gate(Gate::H, &[0]);
            c
        };
        let with_t = build(&t_gate);
        let with_rz = build(&Gate::Rz(angle));

        let spd_t = run_spd_observable(&with_t, &observable, 0.0, 0).unwrap();
        let spd_rz = run_spd_observable(&with_rz, &observable, 0.0, 0).unwrap();
        assert_eq!(spd_t.mean.to_bits(), spd_rz.mean.to_bits());
        assert_eq!(spd_t.peak_terms, spd_rz.peak_terms);

        let spp_t = run_spp_observable(&with_t, &observable, 4_000, SEED).unwrap();
        let spp_rz = run_spp_observable(&with_rz, &observable, 4_000, SEED).unwrap();
        assert_eq!(spp_t.mean.to_bits(), spp_rz.mean.to_bits());
        assert_eq!(spp_t.variance.to_bits(), spp_rz.variance.to_bits());
    }
}

#[test]
fn phase_gate_matches_rz_of_the_same_angle() {
    let theta = 0.731;
    let build = |rot: Gate| {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(rot, &[1]);
        c.add_gate(Gate::H, &[1]);
        c
    };
    let observable = [PauliTerm::z(1)];
    let with_p = run_spd_observable(&build(Gate::P(theta)), &observable, 0.0, 0).unwrap();
    let with_rz = run_spd_observable(&build(Gate::Rz(theta)), &observable, 0.0, 0).unwrap();
    assert_eq!(with_p.mean.to_bits(), with_rz.mean.to_bits());
}

/// QAOA-shaped layers: a `Cx`-`Rz`-`Cx` cost term plus a single-qubit rotation,
/// one rotation angle per layer.
fn variational_circuit(angles: &[f64]) -> Circuit {
    let mut c = Circuit::new(3, 0);
    for q in 0..3 {
        c.add_gate(Gate::H, &[q]);
    }
    for (layer, &theta) in angles.iter().enumerate() {
        let q = layer % 3;
        c.add_gate(Gate::Cx, &[q, (q + 1) % 3]);
        c.add_gate(Gate::Rz(theta), &[(q + 1) % 3]);
        c.add_gate(Gate::Cx, &[q, (q + 1) % 3]);
        c.add_gate(Gate::Rz(theta), &[q]);
        c.add_gate(Gate::H, &[q]);
    }
    c
}

#[test]
fn spd_arbitrary_angles_match_the_statevector() {
    let circuit = variational_circuit(&[0.37, 1.94, -0.62, 2.71]);
    let observables = [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1)],
        vec![PauliTerm::y(0), PauliTerm::y(2)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)],
    ];

    let exact = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();

    for (obs, expected) in observables.iter().zip(exact) {
        let spd = run_spd_observable(&circuit, obs, 0.0, 0).unwrap();
        assert!(
            (spd.mean - expected).abs() < 1e-12,
            "SPD {:.15} vs statevector {expected:.15} for {obs:?}",
            spd.mean
        );
    }
}

#[test]
fn spp_arbitrary_angles_match_the_statevector() {
    let circuit = variational_circuit(&[0.37, 1.94, -0.62]);
    let observables = [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::z(1), PauliTerm::x(2)],
    ];

    let exact = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();

    for (obs, expected) in observables.iter().zip(exact) {
        let spp = run_spp_observable(&circuit, obs, 200_000, SEED).unwrap();
        assert!(
            (spp.mean - expected).abs() < 3.0 * spp.std_error + 0.01,
            "SPP {:.6} +/- {:.6} vs statevector {expected:.6}",
            spp.mean,
            spp.std_error
        );
    }
}

/// Sampling variance is set by the product of `|cos| + |sin|` over the
/// branching rotations, so it peaks at the T angle and collapses toward the
/// Clifford angles. That gradient is the reason the capability is worth having:
/// near-Clifford angles cost far fewer samples for the same error.
#[test]
fn spp_variance_peaks_at_the_t_angle() {
    let quarter = std::f64::consts::FRAC_PI_4;
    let sweep: Vec<f64> = [0.02, 0.4, quarter, 1.1, std::f64::consts::FRAC_PI_2 - 0.02]
        .into_iter()
        .map(|theta| {
            let circuit = variational_circuit(&[theta; 4]);
            run_spp_observable(&circuit, &[PauliTerm::z(0)], 20_000, SEED)
                .unwrap()
                .variance
        })
        .collect();

    let peak = sweep[2];
    assert!(
        sweep.iter().all(|v| *v <= peak),
        "T angle should hold the maximum variance: {sweep:?}"
    );
    for near_clifford in [sweep[0], sweep[4]] {
        assert!(
            near_clifford < 0.2 * peak,
            "near-Clifford variance {near_clifford:.4} should sit far under the peak {peak:.4}"
        );
    }
}

#[test]
fn zero_angle_rotations_do_not_branch() {
    let mut circuit = Circuit::new(2, 0);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Rz(0.0), &[0]);
    circuit.add_gate(Gate::P(0.0), &[0]);
    circuit.add_gate(Gate::Rzz(0.0), &[0, 1]);
    circuit.add_gate(Gate::Cx, &[0, 1]);

    let spd = run_spd_observable(&circuit, &[PauliTerm::x(0), PauliTerm::x(1)], 0.0, 0).unwrap();
    assert_eq!(spd.peak_terms, 1);
    assert_eq!(spd.t_count, 3);
    assert!((spd.mean - 1.0).abs() < 1e-15, "got {}", spd.mean);
}

// The two-qubit anticommutation rule: Rzz branches a term only when exactly
// one of the pair carries an X or Y letter.
#[test]
fn rzz_branches_only_on_anticommuting_terms() {
    let build = || {
        let mut c = Circuit::new(2, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Rzz(0.37), &[0, 1]);
        c
    };

    // X on one qubit of the pair anticommutes with Z⊗Z: one branch fires.
    let anti = run_spd_observable(&build(), &[PauliTerm::x(0)], 0.0, 0).unwrap();
    assert_eq!(anti.peak_terms, 2);
    assert!(
        (anti.mean - 0.37f64.cos()).abs() < 1e-12,
        "got {}",
        anti.mean
    );

    // X⊗X and Z⊗I commute with Z⊗Z: no branch, and the rotation is
    // transparent to the propagated term.
    let xx = run_spd_observable(&build(), &[PauliTerm::x(0), PauliTerm::x(1)], 0.0, 0).unwrap();
    assert_eq!(xx.peak_terms, 1);
    assert!((xx.mean - 1.0).abs() < 1e-12, "got {}", xx.mean);

    let z = run_spd_observable(&build(), &[PauliTerm::z(0)], 0.0, 0).unwrap();
    assert_eq!(z.peak_terms, 1);
    assert!(z.mean.abs() < 1e-12, "got {}", z.mean);
}

// Native Rzz and its Cx-Rz-Cx lowering take different code paths, so agree to
// tolerance rather than bitwise (hash-order accumulation shifts the last ulp).
#[test]
fn rzz_matches_its_cx_rz_cx_lowering() {
    let theta = 1.94;
    let native = {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Rzz(theta), &[0, 1]);
        c.add_gate(Gate::Rzz(-0.62), &[1, 2]);
        c
    };
    let lowered = {
        let mut c = Circuit::new(3, 0);
        c.add_gate(Gate::H, &[0]);
        c.add_gate(Gate::H, &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Rz(theta), &[1]);
        c.add_gate(Gate::Cx, &[0, 1]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c.add_gate(Gate::Rz(-0.62), &[2]);
        c.add_gate(Gate::Cx, &[1, 2]);
        c
    };

    for obs in [
        vec![PauliTerm::x(0)],
        vec![PauliTerm::x(0), PauliTerm::y(1)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)],
    ] {
        let a = run_spd_observable(&native, &obs, 0.0, 0).unwrap();
        let b = run_spd_observable(&lowered, &obs, 0.0, 0).unwrap();
        assert!(
            (a.mean - b.mean).abs() < 1e-12,
            "native {:.15} vs lowered {:.15} for {obs:?}",
            a.mean,
            b.mean
        );
    }
}

/// QAOA-shaped ansatz: an `Rzz` cost ring plus an `Rx` mixer layer, one
/// `(gamma, beta)` pair per layer.
fn qaoa_circuit(n: usize, angles: &[(f64, f64)]) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::H, &[q]);
    }
    for &(gamma, beta) in angles {
        for q in 0..n {
            c.add_gate(Gate::Rzz(gamma), &[q, (q + 1) % n]);
        }
        for q in 0..n {
            c.add_gate(Gate::Rx(beta), &[q]);
        }
    }
    c
}

#[test]
fn spd_qaoa_angle_sweep_matches_the_statevector() {
    let quarter = std::f64::consts::FRAC_PI_4;
    for theta in [0.02, 0.4, quarter, 1.1, std::f64::consts::FRAC_PI_2 - 0.02] {
        let circuit = qaoa_circuit(4, &[(theta, 0.5), (0.8 * theta, 0.3)]);
        let observables = [
            vec![PauliTerm::z(0), PauliTerm::z(1)],
            vec![PauliTerm::x(2)],
            vec![PauliTerm::y(0), PauliTerm::z(3)],
        ];

        let exact = simulate(&circuit)
            .backend(BackendKind::Statevector)
            .seed(SEED)
            .expectation_values(&observables)
            .unwrap();

        for (obs, expected) in observables.iter().zip(exact) {
            let spd = run_spd_observable(&circuit, obs, 0.0, 0).unwrap();
            assert!(
                (spd.mean - expected).abs() < 1e-9,
                "SPD {:.15} vs statevector {expected:.15} at theta {theta:.3} for {obs:?}",
                spd.mean
            );
        }
    }
}

#[test]
fn spp_variance_peaks_at_the_t_angle_for_rzz() {
    let quarter = std::f64::consts::FRAC_PI_4;
    let sweep: Vec<f64> = [0.02, 0.4, quarter, 1.1, std::f64::consts::FRAC_PI_2 - 0.02]
        .into_iter()
        .map(|theta| {
            let circuit = qaoa_circuit(3, &[(theta, 0.0); 4]);
            run_spp_observable(&circuit, &[PauliTerm::x(0)], 20_000, SEED)
                .unwrap()
                .variance
        })
        .collect();

    let peak = sweep[2];
    assert!(
        sweep.iter().all(|v| *v <= peak),
        "T angle should hold the maximum variance: {sweep:?}"
    );
    for near_clifford in [sweep[0], sweep[4]] {
        assert!(
            near_clifford < 0.2 * peak,
            "near-Clifford variance {near_clifford:.4} should sit far under the peak {peak:.4}"
        );
    }
}

#[test]
fn light_cone_spd_matches_full_spd_on_rzz_circuits() {
    let circuit = qaoa_circuit(5, &[(0.37, 0.5), (1.1, 0.3)]);
    for obs in [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::z(1), PauliTerm::x(3)],
    ] {
        let full = run_spd_observable(&circuit, &obs, 0.0, 0).unwrap();
        let cone = prism_q::run_spd_observable_light_cone(&circuit, &obs, 0.0, 0).unwrap();
        assert!(
            (full.mean - cone.mean).abs() < 1e-12,
            "full {:.15} vs cone {:.15} for {obs:?}",
            full.mean,
            cone.mean
        );
    }
}

// Multi-qubit arbitrary-axis strings reach the engines through the shared
// CNOT-ladder lowering.
#[test]
fn pauli_rot_strings_match_the_statevector() {
    let mut circuit = Circuit::new(3, 0);
    for q in 0..3 {
        circuit.add_gate(Gate::H, &[q]);
    }
    circuit.add_pauli_rotation(0.37, &[PauliTerm::x(0), PauliTerm::y(1)]);
    circuit.add_pauli_rotation(-1.1, &[PauliTerm::y(0), PauliTerm::z(1), PauliTerm::x(2)]);
    circuit.add_pauli_rotation(1.94, &[PauliTerm::z(1), PauliTerm::z(2)]);

    let observables = [
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1), PauliTerm::y(2)],
        vec![PauliTerm::z(0), PauliTerm::z(1), PauliTerm::z(2)],
    ];
    let exact = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .expectation_values(&observables)
        .unwrap();

    for (obs, expected) in observables.iter().zip(exact) {
        let spd = run_spd_observable(&circuit, obs, 0.0, 0).unwrap();
        assert!(
            (spd.mean - expected).abs() < 1e-9,
            "SPD {:.15} vs statevector {expected:.15} for {obs:?}",
            spd.mean
        );
        assert_eq!(spd.t_count, 3, "one branching rotation per Pauli string");
    }
}
