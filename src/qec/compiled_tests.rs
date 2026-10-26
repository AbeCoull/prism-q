use super::*;
use crate::gates::Gate;
use crate::qec::{QecOp, QecOptions, QecRecordRef, run_qec_program, run_qec_program_reference};

const NOISY: &str = "R 0 1 2 3 4
X_ERROR(0.1) 0 2 4
CX 0 1 2 1 2 3 4 3
DEPOLARIZE1(0.05) 1 3
MR 1 3
DETECTOR rec[-2]
DETECTOR rec[-1]
M 0 2 4
OBSERVABLE_INCLUDE(0) rec[-1]";

const REPRO: &str = "R 0\nX 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n";

fn with_options(text: &str, options: QecOptions) -> QecProgram {
    let mut program = QecProgram::from_text(text).unwrap();
    program.set_options(options);
    program
}

fn feedforward_program(options: QecOptions) -> QecProgram {
    let mut program = with_options("H 0\nX_ERROR(0.2) 1\nM 0", options);
    program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    program.measure_z(1).unwrap();
    program
        .detector(&[
            QecRecordRef::lookback(1).unwrap(),
            QecRecordRef::lookback(2).unwrap(),
        ])
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::lookback(1).unwrap()])
        .unwrap();
    program
}

/// One program per sampler route, keyed by a label.
fn route_programs(options: QecOptions) -> Vec<(&'static str, QecProgram)> {
    vec![
        ("noisy", with_options(NOISY, options)),
        (
            "random detector",
            with_options(
                &format!("{NOISY}\nR 0\nH 0\nM 0\nDETECTOR rec[-1]"),
                options,
            ),
        ),
        (
            "clean",
            with_options(
                "H 0\nCX 0 1\nX 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]",
                options,
            ),
        ),
        (
            "postselected",
            with_options(&format!("{NOISY}\nPOSTSELECT rec[-2]"), options),
        ),
        (
            "leaky",
            with_options(
                "R 0 1 2\nCX 0 1\nLEAK(0.1) 0 1\nCX 1 2\nM 0 1 2\nDETECTOR rec[-1] rec[-2]",
                options,
            ),
        ),
        ("feedforward", feedforward_program(options)),
        ("repro", with_options(REPRO, options)),
        ("empty", with_options("R 0\nX_ERROR(0.5) 0", options)),
    ]
}

fn assert_same_result(a: &QecSampleResult, b: &QecSampleResult, label: &str) {
    assert_eq!(a.total_shots, b.total_shots, "{label}");
    assert_eq!(a.accepted_shots, b.accepted_shots, "{label}");
    assert_eq!(a.logical_errors, b.logical_errors, "{label}");
    assert_eq!(
        a.measurements.to_shots(),
        b.measurements.to_shots(),
        "{label}"
    );
    assert_eq!(a.detectors.to_shots(), b.detectors.to_shots(), "{label}");
    assert_eq!(
        a.observables.to_shots(),
        b.observables.to_shots(),
        "{label}"
    );
    assert_eq!(
        a.heralds.as_ref().map(PackedShots::to_shots),
        b.heralds.as_ref().map(PackedShots::to_shots),
        "{label}"
    );
}

fn option_variants() -> Vec<QecOptions> {
    let mut variants = Vec::new();
    for chunk_size in [None, Some(37)] {
        for keep_measurements in [true, false] {
            for detector_reference in [QecDetectorReference::Noiseless, QecDetectorReference::Raw] {
                variants.push(QecOptions {
                    shots: 300,
                    seed: 11,
                    chunk_size,
                    keep_measurements,
                    detector_reference,
                });
            }
        }
    }
    variants
}

#[test]
fn a_seeded_sample_matches_a_run_at_that_seed_on_every_route() {
    for options in option_variants() {
        for (label, program) in route_programs(options) {
            let mut sampler = program.compile_sampler().unwrap();
            for seed in [11, 12, 0xDEAD_BEEF] {
                let mut at_seed = program.clone();
                at_seed.set_options(QecOptions { seed, ..options });
                let expected = run_qec_program(&at_seed).unwrap();
                let sampled = sampler.sample(options.shots, Some(seed)).unwrap();
                assert_same_result(&sampled, &expected, &format!("{label} {options:?} {seed}"));
            }
        }
    }
}

#[test]
fn unseeded_samples_continue_one_stream_from_the_program_seed() {
    let options = QecOptions {
        shots: 256,
        ..QecOptions::default()
    };
    let program = with_options(NOISY, options);
    let mut sampler = program.compile_sampler().unwrap();
    let first = sampler.sample(256, None).unwrap();
    assert_same_result(&first, &run_qec_program(&program).unwrap(), "first call");
    let second = sampler.sample(256, None).unwrap();
    assert_ne!(first.detectors.to_shots(), second.detectors.to_shots());

    let mut again = program.compile_sampler().unwrap();
    assert_same_result(
        &again.sample(256, None).unwrap(),
        &first,
        "recompiled first",
    );
    assert_same_result(
        &again.sample(256, None).unwrap(),
        &second,
        "recompiled second",
    );

    sampler.sample(256, Some(options.seed)).unwrap();
    assert_same_result(
        &sampler.sample(256, None).unwrap(),
        &second,
        "reseeded stream",
    );
}

#[test]
fn noiseless_flipped_records_report_no_detection_events() {
    for keep_measurements in [true, false] {
        let options = QecOptions {
            shots: 100,
            keep_measurements,
            ..QecOptions::default()
        };
        let program = with_options(REPRO, options);
        for result in [
            run_qec_program(&program).unwrap(),
            run_qec_program_reference(&program).unwrap(),
        ] {
            assert!(
                result
                    .detectors
                    .to_shots()
                    .iter()
                    .flatten()
                    .all(|&bit| !bit)
            );
            assert!(
                result
                    .observables
                    .to_shots()
                    .iter()
                    .flatten()
                    .all(|&bit| !bit)
            );
            assert_eq!(result.logical_error_rates(), vec![0.0]);
        }

        let raw = with_options(
            REPRO,
            QecOptions {
                detector_reference: QecDetectorReference::Raw,
                ..options
            },
        );
        for result in [
            run_qec_program(&raw).unwrap(),
            run_qec_program_reference(&raw).unwrap(),
        ] {
            assert!(result.detectors.to_shots().iter().flatten().all(|&bit| bit));
            assert_eq!(result.logical_error_rates(), vec![1.0]);
        }
    }
}

#[test]
fn programs_off_the_packed_path_do_not_compile() {
    let exp_val = QecProgram::from_text("H 0\nEXP_VAL X0").unwrap();
    assert!(exp_val.compile_sampler().is_err());
    let non_clifford = QecProgram::from_text("H 0\nT 0\nM 0").unwrap();
    assert!(non_clifford.compile_sampler().is_err());
}

#[test]
fn sampler_reports_the_program_shape() {
    let sampler = QecProgram::from_text(NOISY)
        .unwrap()
        .compile_sampler()
        .unwrap();
    assert_eq!(sampler.num_measurements(), 5);
    assert_eq!(sampler.num_detectors(), 2);
    assert_eq!(sampler.num_observables(), 1);
}
