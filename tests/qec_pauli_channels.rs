//! `Y_ERROR`, `PAULI_CHANNEL_1`, and `PAULI_CHANNEL_2`: parsing, the compiled sampler
//! against the exact branch rates and the reference oracle, detector error model
//! extraction, and the exact density-matrix estimate.

use prism_q::{
    PackedShots, QecNoise, QecPauli, QecProgram, parse_qec_program, run_qec_program,
    run_qec_program_reference,
};

mod qec_common;

const SHOTS: usize = 40_000;
const PC2: [f64; 15] = [
    0.010, 0.020, 0.015, 0.030, 0.005, 0.025, 0.012, 0.018, 0.022, 0.008, 0.016, 0.027, 0.011,
    0.014, 0.019,
];

fn pc2_text() -> String {
    PC2.iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Bell pairs `(q, q + n)` for each channel qubit `q`, the channel, then a `ZZ` and an
/// `XX` detector per pair: the detectors spell out which Pauli fired on each qubit.
fn bell_tomography(n: usize, channel: &str, targets: &str, keep: bool) -> QecProgram {
    let mut text = String::new();
    for q in 0..n {
        text.push_str(&format!("H {q}\nCX {q} {}\n", q + n));
    }
    text.push_str(&format!("{channel} {targets}\n"));
    for q in 0..n {
        text.push_str(&format!(
            "MPP Z{q}*Z{p} X{q}*X{p}\nDETECTOR rec[-2]\nDETECTOR rec[-1]\n",
            p = q + n
        ));
    }
    let mut program = parse_qec_program(&text).unwrap();
    program.set_options(qec_common::qec_options(SHOTS, 8192, keep));
    program
}

/// Letter index (0 I, 1 X, 2 Y, 3 Z) per qubit from its `ZZ` and `XX` detector bits.
fn letter(zz: bool, xx: bool) -> usize {
    match (zz, xx) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

fn histogram(detectors: &PackedShots, qubits: usize) -> Vec<f64> {
    let mut counts = vec![0usize; 1 << (2 * qubits)];
    for shot in 0..detectors.num_shots() {
        let mut index = 0;
        for q in 0..qubits {
            let l = letter(
                detectors.get_bit(shot, 2 * q),
                detectors.get_bit(shot, 2 * q + 1),
            );
            index = index * 4 + l;
        }
        counts[index] += 1;
    }
    counts
        .iter()
        .map(|&c| c as f64 / detectors.num_shots() as f64)
        .collect()
}

#[track_caller]
fn assert_histogram(label: &str, actual: &[f64], expected: &[f64]) {
    for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let sigma = (e * (1.0 - e) / SHOTS as f64)
            .sqrt()
            .max(1.0 / SHOTS as f64);
        assert!(
            (a - e).abs() <= 5.0 * sigma,
            "{label}: outcome {index} rate {a:.5}, expected {e:.5}"
        );
    }
}

fn pc2_expected() -> Vec<f64> {
    let mut expected = vec![0.0; 16];
    expected[0] = 1.0 - PC2.iter().sum::<f64>();
    expected[1..].copy_from_slice(&PC2);
    expected
}

#[test]
fn parser_reads_new_channels() {
    let program = parse_qec_program(&format!(
        "Y_ERROR(0.1) 0\nPAULI_CHANNEL_1(0.1, 0.2, 0.3) 0 1\nPAULI_CHANNEL_2({}) 0 1",
        pc2_text()
    ))
    .unwrap();
    let channels: Vec<&QecNoise> = program
        .ops()
        .iter()
        .filter_map(|op| match op {
            prism_q::QecOp::Noise { channel, .. } => Some(channel),
            _ => None,
        })
        .collect();
    assert_eq!(channels[0], &QecNoise::YError(0.1));
    assert_eq!(channels[1], &QecNoise::PauliChannel1([0.1, 0.2, 0.3]));
    assert_eq!(channels[2], &QecNoise::PauliChannel2(Box::new(PC2)));
    assert!((channels[1].probability() - 0.6).abs() < 1e-15);
}

#[test]
fn parser_rejects_malformed_channels() {
    for text in [
        "PAULI_CHANNEL_1(0.1, 0.2) 0",
        "PAULI_CHANNEL_1(0.5, 0.4, 0.3) 0",
        "PAULI_CHANNEL_1(-0.1, 0.2, 0.3) 0",
        "PAULI_CHANNEL_2(0.1) 0 1",
        "Y_ERROR(1.5) 0",
    ] {
        assert!(parse_qec_program(text).is_err(), "accepted `{text}`");
    }
    let odd = format!("PAULI_CHANNEL_2({}) 0 1 2", pc2_text());
    assert!(parse_qec_program(&odd).is_err());
    let repeated = format!("PAULI_CHANNEL_2({}) 1 1", pc2_text());
    assert!(parse_qec_program(&repeated).is_err());

    let mut program = QecProgram::new(2);
    assert!(
        program
            .noise(QecNoise::PauliChannel1([0.6, 0.6, 0.0]), &[0])
            .is_err()
    );
    assert!(
        program
            .noise(QecNoise::PauliChannel2(Box::new(PC2)), &[0])
            .is_err()
    );
}

#[test]
fn one_qubit_channels_sample_their_branch_rates() {
    for (channel, rates) in [
        ("Y_ERROR(0.2)", [0.8, 0.0, 0.2, 0.0]),
        ("PAULI_CHANNEL_1(0.05, 0.1, 0.15)", [0.7, 0.05, 0.1, 0.15]),
        ("PAULI_CHANNEL_1(0.5, 0.2, 0.25)", [0.05, 0.5, 0.2, 0.25]),
    ] {
        for keep in [false, true] {
            let program = bell_tomography(1, channel, "0", keep);
            let result = run_qec_program(&program).unwrap();
            assert_histogram(channel, &histogram(&result.detectors, 1), &rates);
        }
        let reference =
            run_qec_program_reference(&bell_tomography(1, channel, "0", false)).unwrap();
        assert_histogram(
            &format!("{channel} reference"),
            &histogram(&reference.detectors, 1),
            &rates,
        );
    }
}

#[test]
fn pauli_channel_2_samples_every_branch_on_both_paths() {
    let channel = format!("PAULI_CHANNEL_2({})", pc2_text());
    let expected = pc2_expected();
    for keep in [false, true] {
        let program = bell_tomography(2, &channel, "0 1", keep);
        let result = run_qec_program(&program).unwrap();
        assert_histogram(
            &format!("PAULI_CHANNEL_2 keep={keep}"),
            &histogram(&result.detectors, 2),
            &expected,
        );
    }
    let program = bell_tomography(2, &channel, "0 1", false);
    let reference = run_qec_program_reference(&program).unwrap();
    assert_histogram(
        "PAULI_CHANNEL_2 reference",
        &histogram(&reference.detectors, 2),
        &expected,
    );
}

#[test]
fn dense_pauli_channel_2_uses_the_per_shot_draw() {
    let mut dense = [0.0; 15];
    dense[4] = 0.35;
    dense[14] = 0.25;
    let text = dense
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let program = bell_tomography(2, &format!("PAULI_CHANNEL_2({text})"), "0 1", false);
    let result = run_qec_program(&program).unwrap();
    let mut expected = vec![0.0; 16];
    expected[0] = 0.4;
    expected[5] = 0.35;
    expected[15] = 0.25;
    assert_histogram(
        "dense PAULI_CHANNEL_2",
        &histogram(&result.detectors, 2),
        &expected,
    );
}

#[test]
fn pauli_channel_2_with_a_measured_partner_keeps_the_marginal() {
    let text = format!(
        "R 0 1\nH 0\nM 1\nPAULI_CHANNEL_2({}) 1 0\nH 0\nM 0\nDETECTOR rec[-1]",
        pc2_text()
    );
    let mut program = parse_qec_program(&text).unwrap();
    program.set_options(qec_common::qec_options(SHOTS, 8192, false));
    let result = run_qec_program(&program).unwrap();
    // Qubit 0 is the second target; Y or Z on it flips the X-basis readout.
    let flips: f64 = PC2
        .iter()
        .enumerate()
        .filter(|&(branch, _)| matches!((branch + 1) % 4, 2 | 3))
        .map(|(_, &p)| p)
        .sum();
    assert_histogram(
        "measured-partner marginal",
        &[(0..result.detectors.num_shots())
            .filter(|&shot| result.detectors.get_bit(shot, 0))
            .count() as f64
            / SHOTS as f64],
        &[flips],
    );
}

#[test]
fn detector_error_model_carries_every_branch_exactly() {
    let channel = format!("PAULI_CHANNEL_2({})", pc2_text());
    let model = bell_tomography(2, &channel, "0 1", false)
        .detector_error_model()
        .unwrap();
    assert_eq!(model.num_mechanisms(), 15);
    for mechanism in model.mechanisms() {
        let mut index = 0;
        for q in 0..2 {
            let zz = mechanism.detectors().contains(&(2 * q));
            let xx = mechanism.detectors().contains(&(2 * q + 1));
            index = index * 4 + letter(zz, xx);
        }
        assert_eq!(mechanism.probability(), PC2[index - 1], "branch {index}");
    }

    let model = bell_tomography(1, "PAULI_CHANNEL_1(0.05, 0.1, 0.15)", "0", false)
        .detector_error_model()
        .unwrap();
    let mut rates: Vec<f64> = model.mechanisms().iter().map(|m| m.probability()).collect();
    rates.sort_by(f64::total_cmp);
    assert_eq!(rates, [0.05, 0.1, 0.15]);

    let model = bell_tomography(1, "Y_ERROR(0.2)", "0", false)
        .detector_error_model()
        .unwrap();
    assert_eq!(model.num_mechanisms(), 1);
    assert_eq!(model.mechanisms()[0].detectors(), [0, 1]);
}

#[test]
fn density_matrix_estimates_are_exact() {
    let mut program = QecProgram::new(4);
    for q in 0..2 {
        program.push_gate(prism_q::Gate::H, &[q]).unwrap();
        program.push_gate(prism_q::Gate::Cx, &[q, q + 2]).unwrap();
    }
    program
        .noise(QecNoise::PauliChannel2(Box::new(PC2)), &[0, 1])
        .unwrap();
    program
        .noise(QecNoise::PauliChannel1([0.05, 0.1, 0.15]), &[2])
        .unwrap();
    program.noise(QecNoise::YError(0.2), &[3]).unwrap();
    let observables = [
        [QecPauli::z(0), QecPauli::z(2)],
        [QecPauli::x(0), QecPauli::x(2)],
        [QecPauli::z(1), QecPauli::z(3)],
        [QecPauli::x(1), QecPauli::x(3)],
    ];
    for terms in &observables {
        program.expectation_value(terms, 1.0).unwrap();
    }

    let anticommuting = |qubit: usize, flips: [bool; 4]| -> f64 {
        PC2.iter()
            .enumerate()
            .filter(|&(branch, _)| {
                let sample = branch + 1;
                flips[if qubit == 0 { sample / 4 } else { sample % 4 }]
            })
            .map(|(_, &p)| p)
            .sum()
    };
    let z_flips = [false, true, true, false];
    let x_flips = [false, false, true, true];
    let (px, py, pz) = (0.05, 0.1, 0.15);
    let expected = [
        (1.0 - 2.0 * anticommuting(0, z_flips)) * (1.0 - 2.0 * (px + py)),
        (1.0 - 2.0 * anticommuting(0, x_flips)) * (1.0 - 2.0 * (py + pz)),
        (1.0 - 2.0 * anticommuting(1, z_flips)) * (1.0 - 2.0 * 0.2),
        (1.0 - 2.0 * anticommuting(1, x_flips)) * (1.0 - 2.0 * 0.2),
    ];
    let result = run_qec_program(&program).unwrap();
    qec_common::assert_exact_estimates(
        qec_common::estimates(&result),
        &expected,
        1e-10,
        "density-matrix Pauli channels",
    );
}
