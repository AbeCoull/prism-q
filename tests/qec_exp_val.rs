//! Semantics tests for the QEC `EXP_VAL` estimator: placement validation,
//! coefficient scaling, reference-vs-analytic agreement, postselection
//! composition, and entry-point routing.

mod qec_common;

use prism_q::{
    Gate, QecBasis, QecNoise, QecPauli, QecProgram, QecRecordRef, QecTStrategy, run_qec_program,
    run_qec_program_reference, run_qec_program_with_strategy,
};
use qec_common::ANALYTICAL_STRATEGIES;

const STAT_SHOTS: usize = 4_000;

fn program_with_shots(num_qubits: usize, shots: usize) -> QecProgram {
    QecProgram::with_options(num_qubits, qec_common::qec_options(shots, 2048, false))
}

#[test]
fn exp_val_requires_terminal_placement() {
    let mut gate_after = program_with_shots(1, 16);
    gate_after
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    gate_after.push_gate(Gate::X, &[0]).unwrap();

    let mut measure_after = program_with_shots(1, 16);
    measure_after
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    measure_after.measure_z(0).unwrap();

    for program in [&gate_after, &measure_after] {
        let err = run_qec_program(program).unwrap_err();
        assert!(format!("{err}").contains("must be terminal"));
        let err = run_qec_program_reference(program).unwrap_err();
        assert!(format!("{err}").contains("must be terminal"));
        for &strategy in &ANALYTICAL_STRATEGIES {
            let err = run_qec_program_with_strategy(program, strategy).unwrap_err();
            assert!(
                format!("{err}").contains("must be terminal"),
                "strategy {strategy:?} accepted a non-terminal EXP_VAL"
            );
        }
    }
}

#[test]
fn exp_val_rejects_measured_unreset_qubit() {
    let mut measured = program_with_shots(1, 16);
    measured.measure_z(0).unwrap();
    measured.expectation_value(&[QecPauli::x(0)], 1.0).unwrap();
    let err = run_qec_program_reference(&measured).unwrap_err();
    assert!(format!("{err}").contains("live qubits"));

    let mut reset_between = program_with_shots(1, 16);
    reset_between.measure_z(0).unwrap();
    reset_between.reset(QecBasis::Z, 0).unwrap();
    reset_between
        .expectation_value(&[QecPauli::x(0)], 1.0)
        .unwrap();
    let reference = run_qec_program_reference(&reset_between).unwrap();
    let ref_mean = reference.expectation_values.expect("estimates")[0].mean;
    assert!(ref_mean.abs() < 1e-12, "<X> on a reset qubit must be 0");
    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&reset_between, strategy).unwrap();
        let mean = result.expectation_values.expect("estimates")[0].mean;
        assert!(
            mean.abs() < 1e-9,
            "strategy {strategy:?}: <X> on a reset qubit must be 0, got {mean}"
        );
    }
}

#[test]
fn exp_val_reference_rejects_programs_beyond_mask_width() {
    let mut program = program_with_shots(65, 4);
    program.expectation_value(&[QecPauli::z(64)], 1.0).unwrap();
    let err = run_qec_program_reference(&program).unwrap_err();
    assert!(format!("{err}").contains("at most 64 qubits"));
}

#[test]
fn exp_val_coefficient_scales_mean_and_variance() {
    let build = |coefficient: f64| {
        let mut program = program_with_shots(1, 512);
        program.noise(QecNoise::XError(0.3), &[0]).unwrap();
        program
            .expectation_value(&[QecPauli::z(0)], coefficient)
            .unwrap();
        program
    };
    let unit = run_qec_program(&build(1.0)).unwrap();
    let scaled = run_qec_program(&build(-0.5)).unwrap();
    let unit = &unit.expectation_values.expect("estimates")[0];
    let scaled = &scaled.expectation_values.expect("estimates")[0];
    assert!((scaled.mean - (-0.5) * unit.mean).abs() < 1e-12);
    assert!((scaled.variance - 0.25 * unit.variance).abs() < 1e-12);
    assert!(unit.variance > 0.0, "noisy sampling must report spread");
}

#[test]
fn exp_val_reference_matches_analytic_on_live_qubits_after_measurement() {
    // GHZ on three qubits with a T phase, then a collapsing measurement of
    // qubit 2. EXP_VAL terms live on the unmeasured qubits, so the
    // trajectory-averaged post-measurement expectation must equal the
    // measurement-stripped pure-state expectation. Z0 has genuine per-shot
    // spread (+1 or -1 per collapse branch), X0*X1 and Z0*Z1 are
    // deterministic.
    let mut program = program_with_shots(3, STAT_SHOTS);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::Cx, &[1, 2]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.measure_z(2).unwrap();
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::x(1)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], 1.0)
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_estimates = reference.expectation_values.expect("estimates");
    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let estimates = result.expectation_values.expect("estimates");
        assert_eq!(estimates.len(), 3);
        for (slot, (analytic, sampled)) in estimates.iter().zip(ref_estimates.iter()).enumerate() {
            let tol = 5.0 * (sampled.variance / sampled.num_shots.max(1) as f64).sqrt() + 0.01;
            assert!(
                (analytic.mean - sampled.mean).abs() < tol,
                "strategy {strategy:?} slot {slot}: analytic {:.4} vs reference {:.4} (tol {tol:.4})",
                analytic.mean,
                sampled.mean
            );
        }
    }
}

#[test]
fn exp_val_after_mpp_matches_reference() {
    // MPP measures a scratch alias, so data qubits stay live and an X-basis
    // EXP_VAL on them stays exact: on a Bell pair Z0*Z1 measures +1
    // deterministically and <X0*X1> remains +1.
    let mut program = program_with_shots(2, 256);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program
        .measure_pauli_product(&[QecPauli::z(0), QecPauli::z(1)])
        .unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::x(1)], 1.0)
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_mean = reference.expectation_values.expect("estimates")[0].mean;
    assert!((ref_mean - 1.0).abs() < 1e-12);
    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let mean = result.expectation_values.expect("estimates")[0].mean;
        assert!(
            (mean - 1.0).abs() < 1e-9,
            "strategy {strategy:?}: <X0*X1> after MPP should be 1, got {mean}"
        );
    }
}

#[test]
fn exp_val_composes_with_postselection() {
    // Bell pair, measure qubit 1, postselect on outcome 0. The accepted
    // ensemble collapses qubit 0 to |0>, so the conditioned <Z0> is +1 on
    // both paths, and num_shots reflects roughly half the shots surviving.
    let mut program = program_with_shots(2, STAT_SHOTS);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    let record = program.measure_z(1).unwrap();
    program
        .postselect(&[QecRecordRef::absolute(record)], false)
        .unwrap();
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let sampled = &reference.expectation_values.expect("estimates")[0];
    assert!((sampled.mean - 1.0).abs() < 1e-12);
    assert_eq!(sampled.num_shots, reference.accepted_shots);
    assert!(sampled.num_shots > STAT_SHOTS / 3 && sampled.num_shots < 2 * STAT_SHOTS / 3);

    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let estimate = &result.expectation_values.expect("estimates")[0];
        assert!(
            (estimate.mean - 1.0).abs() < 1e-9,
            "strategy {strategy:?}: conditioned <Z0> should be 1, got {}",
            estimate.mean
        );
        assert_eq!(estimate.num_shots, STAT_SHOTS / 2);
    }
}

#[test]
fn exp_val_routing_pinned() {
    let base = |num_qubits: usize| {
        let mut program = program_with_shots(num_qubits, 256);
        program.push_gate(Gate::H, &[0]).unwrap();
        program.push_gate(Gate::T, &[0]).unwrap();
        program
    };

    let mut noiseless = base(1);
    noiseless.expectation_value(&[QecPauli::x(0)], 1.0).unwrap();
    let routed = run_qec_program(&noiseless).unwrap();
    let auto = run_qec_program_with_strategy(&noiseless, QecTStrategy::Auto).unwrap();
    let routed_estimates = routed.expectation_values.expect("estimates");
    let auto_estimates = auto.expectation_values.expect("estimates");
    for (a, b) in routed_estimates.iter().zip(auto_estimates.iter()) {
        assert!(
            (a.mean - b.mean).abs() < 1e-12 && a.num_shots == b.num_shots,
            "noiseless EXP_VAL programs must route to the analytical Auto ladder"
        );
    }

    let mut noisy = base(1);
    noisy.noise(QecNoise::XError(0.1), &[0]).unwrap();
    noisy.expectation_value(&[QecPauli::x(0)], 1.0).unwrap();
    let routed = run_qec_program(&noisy).unwrap();
    let reference = run_qec_program_reference(&noisy).unwrap();
    assert_eq!(
        routed.expectation_values.expect("estimates"),
        reference.expectation_values.expect("estimates"),
        "noisy EXP_VAL programs must route to the reference runner"
    );

    let mut with_detector = base(2);
    with_detector.push_gate(Gate::Cx, &[0, 1]).unwrap();
    with_detector.measure_z(1).unwrap();
    with_detector
        .detector(&[QecRecordRef::lookback(1).unwrap()])
        .unwrap();
    with_detector
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let routed = run_qec_program(&with_detector).unwrap();
    let reference = run_qec_program_reference(&with_detector).unwrap();
    assert_eq!(
        routed.expectation_values.expect("estimates"),
        reference.expectation_values.expect("estimates"),
        "non-Clifford EXP_VAL programs with detectors must fall back to the reference runner"
    );
    assert_eq!(routed.detectors.num_measurements(), 1);
    assert_eq!(routed.detectors.num_shots(), 256);
}

#[test]
fn exp_val_with_detectors_splits_clifford_programs() {
    // GHZ on three qubits, measure qubit 2 with a detector on the record.
    // The detector must come from real packed sampling (fires on the |111>
    // branch, rate 1/2) while the estimates stay analytical and exact:
    // stripped <Z0*Z1> = 1 (scaled by -2.0) and stripped <Z0> = 0.
    let mut program = program_with_shots(3, STAT_SHOTS);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::Cx, &[1, 2]).unwrap();
    program.measure_z(2).unwrap();
    program
        .detector(&[QecRecordRef::lookback(1).unwrap()])
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], -2.0)
        .unwrap();
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();

    let result = run_qec_program(&program).unwrap();
    let estimates = result.expectation_values.expect("estimates");
    assert_eq!(estimates.len(), 2);
    assert!(
        (estimates[0].mean - (-2.0)).abs() < 1e-9,
        "split path slot 0: {} vs closed form -2.0",
        estimates[0].mean
    );
    assert!(
        estimates[0].variance <= 1e-12 && estimates[1].variance <= 1e-12,
        "analytical estimates on a Clifford program must be exact"
    );
    assert!(estimates[1].mean.abs() < 1e-9);

    assert_eq!(result.detectors.num_measurements(), 1);
    assert_eq!(result.detectors.num_shots(), STAT_SHOTS);
    let fired = result
        .detectors
        .to_shots()
        .iter()
        .filter(|shot| shot[0])
        .count();
    let rate = fired as f64 / STAT_SHOTS as f64;
    assert!(
        (0.4..0.6).contains(&rate),
        "detector firing rate {rate:.4} outside the sampled band around 0.5"
    );
}

#[test]
fn exp_val_xy_strings_across_strategies() {
    // Non-measuring Clifford+T fixture: the reference runner is exact per
    // shot (zero variance), so every analytical strategy must reproduce its
    // means for X-, Y-, and mixed strings. Pins the CAMPS X/Y extension
    // end to end.
    let mut program = program_with_shots(2, 64);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::T, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::T, &[1]).unwrap();
    program.push_gate(Gate::H, &[1]).unwrap();
    program.expectation_value(&[QecPauli::x(0)], 1.0).unwrap();
    program.expectation_value(&[QecPauli::y(0)], 1.0).unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::y(1)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::x(1)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::y(0), QecPauli::y(1)], -2.0)
        .unwrap();

    let reference = run_qec_program_reference(&program).unwrap();
    let ref_estimates = reference.expectation_values.expect("estimates");
    for estimate in ref_estimates.iter() {
        assert!(estimate.variance < 1e-30, "identical shots leave no spread");
    }
    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        let estimates = result.expectation_values.expect("estimates");
        assert_eq!(estimates.len(), ref_estimates.len());
        for (slot, (analytic, exact)) in estimates.iter().zip(ref_estimates.iter()).enumerate() {
            assert!(
                (analytic.mean - exact.mean).abs() < 1e-6,
                "strategy {strategy:?} slot {slot}: {:.8} vs exact {:.8}",
                analytic.mean,
                exact.mean
            );
        }
    }
}
