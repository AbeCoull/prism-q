//! End-to-end distance-3 repetition-code fixtures for the `EXP_VAL`
//! estimator.
//!
//! `e2e-d3-t`: logical |+> encoding, one syndrome round with ancilla resets
//! (exercising reset-alias translation of `EXP_VAL` terms), deterministic
//! postselection, transversal T, and X/Y/Z logical expectation values with
//! closed-form references. Runs the analytical ladder, the compiled-runner
//! routing, and the sampled reference.
//!
//! `e2e-d3-s`: Clifford-only encoding under X noise with detectors,
//! sampled through the compiled runner's reference route, compared against
//! closed-form means within a 5-sigma band.

mod qec_common;

use prism_q::{
    Gate, QecBasis, QecNoise, QecPauli, QecProgram, QecRecordRef, QecTStrategy, run_qec_program,
    run_qec_program_reference, run_qec_program_with_strategy,
};
use qec_common::ANALYTICAL_STRATEGIES;

const STAT_SHOTS: usize = 4_000;

fn syndrome_round(program: &mut QecProgram) -> (usize, usize) {
    program.push_gate(Gate::Cx, &[0, 3]).unwrap();
    program.push_gate(Gate::Cx, &[1, 3]).unwrap();
    program.push_gate(Gate::Cx, &[1, 4]).unwrap();
    program.push_gate(Gate::Cx, &[2, 4]).unwrap();
    let m0 = program.measure_z(3).unwrap();
    let m1 = program.measure_z(4).unwrap();
    (m0, m1)
}

fn e2e_d3_t_program() -> QecProgram {
    let mut program = QecProgram::with_options(5, qec_common::qec_options(STAT_SHOTS, 2048, false));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::Cx, &[0, 2]).unwrap();
    let (m0, _m1) = syndrome_round(&mut program);
    program.reset(QecBasis::Z, 3).unwrap();
    program.reset(QecBasis::Z, 4).unwrap();
    program
        .postselect(&[QecRecordRef::absolute(m0)], false)
        .unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::T, &[1]).unwrap();
    program.push_gate(Gate::T, &[2]).unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::x(1), QecPauli::x(2)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::y(0), QecPauli::x(1), QecPauli::x(2)], -0.5)
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], 1.0)
        .unwrap();
    program
}

// State after transversal T on the logical |+>: (|000> + e^{i3pi/4}|111>)/sqrt(2).
// <X_L> = cos(3pi/4), <Y0 X1 X2> = sin(3pi/4) (scaled by -0.5), <Z0 Z1> = 1.
fn e2e_d3_t_expected() -> [f64; 3] {
    let theta = 3.0 * std::f64::consts::FRAC_PI_4;
    [theta.cos(), -0.5 * theta.sin(), 1.0]
}

#[test]
fn e2e_d3_t() {
    let program = e2e_d3_t_program();
    let expected = e2e_d3_t_expected();

    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let estimates = result.expectation_values.expect("e2e-d3-t estimates");
        assert_eq!(estimates.len(), 3, "e2e-d3-t: three EXP_VAL ops in order");
        for (slot, (estimate, reference)) in estimates.iter().zip(expected.iter()).enumerate() {
            assert!(
                (estimate.mean - reference).abs() < 1e-6,
                "e2e-d3-t strategy {strategy:?} slot {slot}: {:.8} vs closed form {reference:.8}",
                estimate.mean
            );
        }
    }

    let routed = run_qec_program(&program).unwrap();
    let auto = run_qec_program_with_strategy(&program, QecTStrategy::Auto).unwrap();
    let routed_estimates = routed.expectation_values.expect("estimates");
    let auto_estimates = auto.expectation_values.expect("estimates");
    for (slot, (a, b)) in routed_estimates
        .iter()
        .zip(auto_estimates.iter())
        .enumerate()
    {
        assert!(
            (a.mean - b.mean).abs() < 1e-12,
            "e2e-d3-t slot {slot}: compiled entry point must match the Auto ladder"
        );
    }

    // The syndrome round is deterministic on the logical |+> state, so the
    // sampled reference is exact shot for shot.
    let reference = run_qec_program_reference(&program).unwrap();
    assert_eq!(reference.accepted_shots, STAT_SHOTS);
    let estimates = reference.expectation_values.expect("estimates");
    for (slot, (estimate, expected)) in estimates.iter().zip(expected.iter()).enumerate() {
        assert!(
            (estimate.mean - expected).abs() < 1e-9,
            "e2e-d3-t reference slot {slot}: {:.10} vs closed form {expected:.10}",
            estimate.mean
        );
    }
}

#[test]
fn e2e_d3_s() {
    let p = 0.05;
    let mut program = QecProgram::with_options(5, qec_common::qec_options(STAT_SHOTS, 2048, false));
    program.noise(QecNoise::XError(p), &[0, 1, 2]).unwrap();
    let (m0, m1) = syndrome_round(&mut program);
    program.detector(&[QecRecordRef::absolute(m0)]).unwrap();
    program.detector(&[QecRecordRef::absolute(m1)]).unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1), QecPauli::z(2)], 1.0)
        .unwrap();
    program.expectation_value(&[QecPauli::z(0)], 2.0).unwrap();

    // Independent X flips with probability p on each data qubit:
    // <Z0 Z1 Z2> = (1-2p)^3 and 2*<Z0> = 2*(1-2p).
    let expected = [(1.0 - 2.0 * p).powi(3), 2.0 * (1.0 - 2.0 * p)];

    let result = run_qec_program(&program).unwrap();
    assert_eq!(result.total_shots, STAT_SHOTS);
    let estimates = result.expectation_values.expect("e2e-d3-s estimates");
    assert_eq!(estimates.len(), 2, "e2e-d3-s: two EXP_VAL ops in order");
    for (slot, (estimate, reference)) in estimates.iter().zip(expected.iter()).enumerate() {
        assert!(
            estimate.variance > 0.0,
            "e2e-d3-s slot {slot}: sampled path must report nonzero variance"
        );
        assert_eq!(estimate.num_shots, STAT_SHOTS);
        let tol = 5.0 * (estimate.variance / estimate.num_shots as f64).sqrt() + 0.02;
        assert!(
            (estimate.mean - reference).abs() < tol,
            "e2e-d3-s slot {slot}: {:.4} vs closed form {reference:.4} (tol {tol:.4})",
            estimate.mean
        );
    }

    // Each detector fires when an odd number of its two data qubits flipped:
    // rate 2p(1-p) = 0.095.
    let detector_shots = result.detectors.to_shots();
    assert_eq!(result.detectors.num_measurements(), 2);
    for detector in 0..2 {
        let fired = detector_shots.iter().filter(|shot| shot[detector]).count() as f64;
        let rate = fired / STAT_SHOTS as f64;
        assert!(
            (0.05..0.15).contains(&rate),
            "e2e-d3-s detector {detector}: firing rate {rate:.4} outside the plausible band"
        );
    }
}
