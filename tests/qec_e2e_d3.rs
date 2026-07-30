//! End-to-end distance-3 repetition-code fixtures for the `EXP_VAL`
//! estimator, each against closed-form references: `e2e-d3-t` (noiseless
//! transversal T with reset-alias translation and postselection), `e2e-d3-s`
//! (Clifford encoding under X noise with detectors, sampled), and `e2e-d3-d`
//! (depolarized transversal T on the exact density-matrix route).

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
        qec_common::assert_exact_estimates(
            qec_common::estimates(&result),
            &expected,
            1e-6,
            &format!("e2e-d3-t strategy {strategy:?}"),
        );
    }

    let routed = run_qec_program(&program).unwrap();
    let auto = run_qec_program_with_strategy(&program, QecTStrategy::Auto).unwrap();
    qec_common::assert_exact_estimates(
        qec_common::estimates(&routed),
        &qec_common::means(&auto),
        1e-12,
        "e2e-d3-t compiled entry point vs Auto ladder",
    );

    // The syndrome round is deterministic on the logical |+> state, so the
    // sampled reference is exact shot for shot.
    let reference = run_qec_program_reference(&program).unwrap();
    assert_eq!(reference.accepted_shots, STAT_SHOTS);
    qec_common::assert_exact_estimates(
        qec_common::estimates(&reference),
        &expected,
        1e-9,
        "e2e-d3-t reference",
    );
}

#[test]
fn e2e_d3_d() {
    // Logical |+> plus transversal T, then DEPOLARIZE1(p) on every data
    // qubit. No syndrome round, so the program carries no measurement
    // records and routes to the density-matrix estimator. Depolarizing keeps
    // (1-4p/3) of a term per data qubit it touches, so the weight-3 logical
    // terms pick up the cube and Z0*Z1 the square.
    let p = 0.06;
    let mut program = QecProgram::with_options(3, qec_common::qec_options(STAT_SHOTS, 2048, false));
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::Cx, &[0, 2]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::T, &[1]).unwrap();
    program.push_gate(Gate::T, &[2]).unwrap();
    program.noise(QecNoise::Depolarize1(p), &[0, 1, 2]).unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::x(1), QecPauli::x(2)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::y(0), QecPauli::x(1), QecPauli::x(2)], -0.5)
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], 1.0)
        .unwrap();

    let [x_l, y_l, zz] = e2e_d3_t_expected();
    let decay = 1.0 - 4.0 * p / 3.0;
    let expected = [x_l * decay.powi(3), y_l * decay.powi(3), zz * decay.powi(2)];

    let result = run_qec_program(&program).unwrap();
    let estimates = qec_common::estimates(&result);
    qec_common::assert_exact_estimates(estimates, &expected, 1e-10, "e2e-d3-d");
    qec_common::assert_zero_variance(estimates, "e2e-d3-d");
    assert!(estimates.iter().all(|e| e.num_shots == STAT_SHOTS));

    let reference = run_qec_program_reference(&program).unwrap();
    qec_common::assert_within_sampling_error(
        qec_common::estimates(&reference),
        &expected,
        0.01,
        "e2e-d3-d reference",
    );
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
    let estimates = qec_common::estimates(&result);
    qec_common::assert_within_sampling_error(estimates, &expected, 0.02, "e2e-d3-s");
    for (slot, estimate) in estimates.iter().enumerate() {
        assert!(
            estimate.variance > 0.0,
            "e2e-d3-s slot {slot}: measurement records keep this program on the sampled path"
        );
        assert_eq!(estimate.num_shots, STAT_SHOTS);
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
