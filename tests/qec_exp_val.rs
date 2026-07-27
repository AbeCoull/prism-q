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
    // Sampled path, called directly: `run_qec_program` sends this program to
    // the exact density-matrix estimator, which reports no spread to scale.
    let build = |coefficient: f64| {
        let mut program = program_with_shots(1, 512);
        program.noise(QecNoise::XError(0.3), &[0]).unwrap();
        program
            .expectation_value(&[QecPauli::z(0)], coefficient)
            .unwrap();
        program
    };
    let unit = run_qec_program_reference(&build(1.0)).unwrap();
    let scaled = run_qec_program_reference(&build(-0.5)).unwrap();
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
    for &strategy in &ANALYTICAL_STRATEGIES {
        let result = run_qec_program_with_strategy(&program, strategy).unwrap();
        qec_common::assert_within_sampling_error(
            qec_common::estimates(&reference),
            &qec_common::means(&result),
            0.01,
            &format!("strategy {strategy:?} vs reference"),
        );
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
    qec_common::assert_exact_estimates(
        qec_common::estimates(&routed),
        &qec_common::means(&auto),
        1e-12,
        "noiseless EXP_VAL programs route to the analytical Auto ladder",
    );

    // No measurement records and no postselection, so this is eligible for the
    // exact route. DEPOLARIZE1 gives the sampled path genuine per-shot spread
    // (the Y and Z branches flip the X observable), so a zero variance here
    // can only come from the exact route. <X0> = cos(pi/4) * (1 - 4p/3).
    let p = 0.3;
    let mut noisy = base(1);
    noisy.noise(QecNoise::Depolarize1(p), &[0]).unwrap();
    noisy.expectation_value(&[QecPauli::x(0)], 1.0).unwrap();
    let expected = [std::f64::consts::FRAC_1_SQRT_2 * (1.0 - 4.0 * p / 3.0)];
    let routed = run_qec_program(&noisy).unwrap();
    let estimates = qec_common::estimates(&routed);
    qec_common::assert_exact_estimates(
        estimates,
        &expected,
        1e-10,
        "eligible noisy EXP_VAL programs route to the density-matrix estimator",
    );
    qec_common::assert_zero_variance(estimates, "density-matrix route");
    let sampled = run_qec_program_reference(&noisy).unwrap();
    assert!(
        qec_common::estimates(&sampled)[0].variance > 0.0,
        "fixture must separate the routes: the sampled path has to report spread"
    );

    // Measurement records force per-shot sampling: Z0 collapses to +1 or -1
    // with the measured partner, so the reference runner keeps this program.
    let mut noisy_measured = program_with_shots(2, 256);
    noisy_measured.push_gate(Gate::H, &[0]).unwrap();
    noisy_measured.push_gate(Gate::Cx, &[0, 1]).unwrap();
    noisy_measured.noise(QecNoise::XError(0.1), &[0]).unwrap();
    noisy_measured.measure_z(1).unwrap();
    noisy_measured
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let routed =
        qec_common::assert_matches_reference_runner(&noisy_measured, "measurement records");
    assert!(
        qec_common::estimates(&routed)[0].variance > 0.0,
        "sampled path must report spread"
    );
    assert_eq!(routed.measurements.num_measurements(), 1);

    // Postselection conditions the estimate on an accepted subensemble, which
    // Tr(rho O) cannot express. The empty-record predicate never matches, so
    // the reference runner reports an estimate over zero accepted shots.
    let mut noisy_postselected = program_with_shots(1, 256);
    noisy_postselected
        .noise(QecNoise::XError(0.3), &[0])
        .unwrap();
    noisy_postselected.postselect(&[], true).unwrap();
    noisy_postselected
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let routed = qec_common::assert_matches_reference_runner(&noisy_postselected, "postselection");
    assert_eq!(routed.accepted_shots, 0);

    let mut with_detector = base(2);
    with_detector.push_gate(Gate::Cx, &[0, 1]).unwrap();
    with_detector.measure_z(1).unwrap();
    with_detector
        .detector(&[QecRecordRef::lookback(1).unwrap()])
        .unwrap();
    with_detector
        .expectation_value(&[QecPauli::z(0)], 1.0)
        .unwrap();
    let routed = qec_common::assert_matches_reference_runner(
        &with_detector,
        "non-Clifford program with detectors",
    );
    assert_eq!(routed.detectors.num_measurements(), 1);
    assert_eq!(routed.detectors.num_shots(), 256);
}

#[test]
fn exp_val_noisy_density_matrix_matches_closed_form() {
    // Bell pair under all four QEC Pauli channels. Each channel rescales a
    // Pauli expectation by a fixed factor: X_ERROR(a) and Z_ERROR(b) give
    // (1-2a) and (1-2b) on terms that anticommute with them, DEPOLARIZE1(c)
    // gives (1-4c/3) on a term touching its target (two of the three Paulis
    // anticommute), and DEPOLARIZE2(d) gives (1-16d/15) on a two-qubit term
    // (7 of the 15 non-identity two-qubit Paulis commute with it). On the
    // Bell state <Z0*Z1> = <X0*X1> = 1 and <Y0*Y1> = -1.
    let (a, b, c, d) = (0.1, 0.2, 0.15, 0.05);
    let mut program = program_with_shots(2, STAT_SHOTS);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.noise(QecNoise::XError(a), &[0]).unwrap();
    program.noise(QecNoise::ZError(b), &[1]).unwrap();
    program.noise(QecNoise::Depolarize1(c), &[0]).unwrap();
    program.noise(QecNoise::Depolarize2(d), &[0, 1]).unwrap();
    program
        .expectation_value(&[QecPauli::z(0), QecPauli::z(1)], 1.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::x(0), QecPauli::x(1)], 2.0)
        .unwrap();
    program
        .expectation_value(&[QecPauli::y(0), QecPauli::y(1)], -0.5)
        .unwrap();

    let shared = (1.0 - 4.0 * c / 3.0) * (1.0 - 16.0 * d / 15.0);
    let expected = [
        (1.0 - 2.0 * a) * shared,
        2.0 * (1.0 - 2.0 * b) * shared,
        0.5 * (1.0 - 2.0 * a) * (1.0 - 2.0 * b) * shared,
    ];

    let result = run_qec_program(&program).unwrap();
    let estimates = qec_common::estimates(&result);
    qec_common::assert_exact_estimates(estimates, &expected, 1e-10, "density-matrix estimate");
    qec_common::assert_zero_variance(estimates, "density-matrix estimate");
    assert!(estimates.iter().all(|e| e.num_shots == STAT_SHOTS));
    assert_eq!(result.total_shots, STAT_SHOTS);
    assert_eq!(result.accepted_shots, STAT_SHOTS);
    assert_eq!(result.measurements.num_measurements(), 0);

    // The per-shot reference runner estimates the same quantity by sampling,
    // which cross-checks the channel lowering against the trajectory path.
    let reference = run_qec_program_reference(&program).unwrap();
    qec_common::assert_within_sampling_error(
        qec_common::estimates(&reference),
        &expected,
        0.01,
        "reference runner",
    );
}

#[test]
fn exp_val_noisy_density_matrix_absorbs_resets() {
    // Both paths implement the reset channel, so resets stay on the exact
    // route. Resetting qubit 1 of a Bell pair traces it out, leaving qubit 0
    // maximally mixed (<Z0> = 0) and qubit 1 in |+>, whose <X1> decays to
    // 1-2p under Z_ERROR(p). The reference runner samples one branch of the
    // reset per shot, so its mean converges to the same values.
    let p = 0.25;
    let mut program = program_with_shots(2, STAT_SHOTS);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.reset(QecBasis::X, 1).unwrap();
    program.noise(QecNoise::ZError(p), &[1]).unwrap();
    program.expectation_value(&[QecPauli::x(1)], 1.0).unwrap();
    program.expectation_value(&[QecPauli::z(0)], 1.0).unwrap();

    let expected = [1.0 - 2.0 * p, 0.0];

    let result = run_qec_program(&program).unwrap();
    let estimates = qec_common::estimates(&result);
    qec_common::assert_exact_estimates(estimates, &expected, 1e-10, "reset under noise");
    qec_common::assert_zero_variance(estimates, "reset under noise");

    let reference = run_qec_program_reference(&program).unwrap();
    qec_common::assert_within_sampling_error(
        qec_common::estimates(&reference),
        &expected,
        0.01,
        "reset under noise reference",
    );
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
