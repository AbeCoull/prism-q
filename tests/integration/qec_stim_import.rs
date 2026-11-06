//! Stim circuit file import: generated memory circuits against the builder API and the
//! native text parser, inverted measurement targets, record-controlled Paulis, and the
//! errors for instructions outside the supported subset.

use prism_q::{
    Gate, PrismError, QecBasis, QecNoise, QecOp, QecProgram, QecRecordRef, QecSampleResult,
    parse_qec_program, parse_stim_circuit, run_qec_program, run_qec_program_reference,
};

use crate::qec_common;

const COLOR_CODE_MEMORY_XYZ_D5: &str = include_str!("../fixtures/qec/color_code_memory_xyz_d5.txt");

const REPETITION_MEMORY: &str = "
X_ERROR(0.05) 0 1 2
MPP Z0*Z1 Z1*Z2
DETECTOR rec[-2]
DETECTOR rec[-1]
REPEAT 2 {
    X_ERROR(0.05) 0 1 2
    MPP Z0*Z1 Z1*Z2
    DETECTOR rec[-2] rec[-4]
    DETECTOR rec[-1] rec[-3]
}
M 0 1 2
DETECTOR rec[-3] rec[-2] rec[-5]
DETECTOR rec[-2] rec[-1] rec[-4]
OBSERVABLE_INCLUDE(0) rec[-3]
";

// Distance-2 rotated surface code Z memory: data 0..3, X check X0X1X2X3 on ancilla 4,
// Z checks Z0Z2 and Z1Z3 on ancillas 5 and 6, logical Z0Z1.
const SURFACE_D2_MEMORY: &str = "
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(0, 1) 2
QUBIT_COORDS(1, 1) 3
QUBIT_COORDS(0.5, 0.5) 4
QUBIT_COORDS(0, 0.5) 5
QUBIT_COORDS(1, 0.5) 6
R 0 1 2 3 4 5 6
X_ERROR(0.01) 0 1 2 3
TICK
H 4
TICK
CX 4 0 4 1 4 2 4 3
DEPOLARIZE2(0.01) 4 0 4 1 4 2 4 3
TICK
H 4
CX 0 5 2 5 1 6 3 6
DEPOLARIZE2(0.01) 0 5 2 5 1 6 3 6
TICK
X_ERROR(0.01) 4 5 6
MR 4 5 6
DETECTOR(0, 0.5, 0) rec[-2]
DETECTOR(1, 0.5, 0) rec[-1]
REPEAT 2 {
    TICK
    DEPOLARIZE1(0.01) 0 1 2 3
    H 4
    TICK
    CX 4 0 4 1 4 2 4 3
    DEPOLARIZE2(0.01) 4 0 4 1 4 2 4 3
    TICK
    H 4
    CX 0 5 2 5 1 6 3 6
    DEPOLARIZE2(0.01) 0 5 2 5 1 6 3 6
    TICK
    X_ERROR(0.01) 4 5 6
    MR 4 5 6
    SHIFT_COORDS(0, 0, 1)
    DETECTOR(0.5, 0.5, 0) rec[-3] rec[-6]
    DETECTOR(0, 0.5, 0) rec[-2] rec[-5]
    DETECTOR(1, 0.5, 0) rec[-1] rec[-4]
}
X_ERROR(0.01) 0 1 2 3
M 0 1 2 3
DETECTOR(0, 0.5, 1) rec[-4] rec[-2] rec[-6]
DETECTOR(1, 0.5, 1) rec[-3] rec[-1] rec[-5]
OBSERVABLE_INCLUDE(0) rec[-4] rec[-3]
";

fn sample(mut program: QecProgram, shots: usize) -> QecSampleResult {
    program.set_options(qec_common::qec_options(shots, shots, true));
    run_qec_program(&program).unwrap()
}

fn assert_same_samples(a: &QecSampleResult, b: &QecSampleResult) {
    assert_eq!(a.measurements.to_shots(), b.measurements.to_shots());
    assert_eq!(a.detectors.to_shots(), b.detectors.to_shots());
    assert_eq!(a.observables.to_shots(), b.observables.to_shots());
}

fn without_coords(program: &QecProgram) -> Vec<QecOp> {
    program
        .ops()
        .iter()
        .cloned()
        .map(|mut op| {
            if let QecOp::Detector { coords, .. } = &mut op {
                coords.clear();
            }
            op
        })
        .collect()
}

fn detector_coords(program: &QecProgram) -> Vec<Vec<f64>> {
    program
        .ops()
        .iter()
        .filter_map(|op| match op {
            QecOp::Detector { coords, .. } => Some(coords.clone()),
            _ => None,
        })
        .collect()
}

fn parse_error(text: &str) -> (usize, String) {
    match parse_stim_circuit(text).unwrap_err() {
        PrismError::Parse { line, message } => (line, message),
        other => panic!("expected a parse error, got {other}"),
    }
}

#[test]
fn stim_repetition_memory_matches_the_builder_api() {
    let shots = 10_000;
    let built = qec_common::repetition_memory(3, 3, QecNoise::XError(0.05), shots);
    let program = parse_stim_circuit(REPETITION_MEMORY).unwrap();
    assert_eq!(program.num_qubits(), built.num_qubits());
    assert_eq!(program.ops(), built.ops());

    let result = sample(program, shots);
    assert_same_samples(&result, &sample(built, shots));

    let shots_by_detector = result.detectors.to_shots();
    for detector in 0..2 {
        let fired = shots_by_detector.iter().filter(|s| s[detector]).count();
        let rate = fired as f64 / shots as f64;
        assert!(
            (rate - 0.095).abs() < 0.012,
            "round-0 detector {detector} fired at {rate}, expected 2 * 0.05 * 0.95"
        );
    }
}

#[test]
fn stim_surface_memory_with_repeat_matches_native_text() {
    let stim = parse_stim_circuit(SURFACE_D2_MEMORY).unwrap();
    let native = parse_qec_program(SURFACE_D2_MEMORY).unwrap();
    assert_eq!(stim.num_detectors(), 2 + 2 * 3 + 2);
    assert_eq!(without_coords(&stim), without_coords(&native));
    assert_same_samples(&sample(stim, 4096), &sample(native, 4096));

    let noiseless = parse_stim_circuit(&SURFACE_D2_MEMORY.replace("(0.01)", "(0)")).unwrap();
    let result = sample(noiseless, 256);
    assert!(
        result
            .detectors
            .to_shots()
            .iter()
            .chain(&result.observables.to_shots())
            .flatten()
            .all(|&bit| !bit)
    );
}

#[test]
fn stim_shift_coords_offsets_detector_coordinates() {
    let stim = detector_coords(&parse_stim_circuit(SURFACE_D2_MEMORY).unwrap());
    assert_eq!(stim[0], [0.0, 0.5, 0.0]);
    assert_eq!(stim[2], [0.5, 0.5, 1.0]);
    assert_eq!(stim[5], [0.5, 0.5, 2.0]);
    assert_eq!(stim[8], [0.0, 0.5, 3.0]);

    let native = detector_coords(&parse_qec_program(SURFACE_D2_MEMORY).unwrap());
    assert_eq!(native[5], [0.5, 0.5, 0.0]);
    assert_eq!(native[8], [0.0, 0.5, 1.0]);
}

#[test]
fn stim_generated_color_code_matches_native_text() {
    let stim = parse_stim_circuit(COLOR_CODE_MEMORY_XYZ_D5).unwrap();
    let native = parse_qec_program(COLOR_CODE_MEMORY_XYZ_D5).unwrap();
    assert_eq!(without_coords(&stim), without_coords(&native));
    assert_same_samples(&sample(stim, 2048), &sample(native, 2048));
}

#[test]
fn stim_inverted_measurements_invert_only_the_record() {
    let program = parse_stim_circuit(
        "RX 1 5 6
         M !0
         MX !1
         MPP !Z2*Z3 Z2*!Z3 !Z2*!Z3 Z2*Z3
         MR !4
         M 4
         MXX !5 6
         X_ERROR(0.5) 0",
    )
    .unwrap();
    let expected = [true, true, true, true, false, false, true, false, true];
    for shot in sample(program, 64).measurements.to_shots() {
        assert_eq!(shot, expected);
    }

    let mut reused = parse_stim_circuit("M !0\nM 0\nMPP !Z0\nM 0\nR 0\nM !0").unwrap();
    reused.set_options(qec_common::qec_options(16, 16, true));
    for shot in run_qec_program_reference(&reused)
        .unwrap()
        .measurements
        .to_shots()
    {
        assert_eq!(shot, [true, false, true, false, true]);
    }
}

#[test]
fn stim_record_controlled_paulis_lower_to_feedforward() {
    let program = parse_stim_circuit("X 0\nM 0\nCX rec[-1] 1\nM 1").unwrap();
    let mut built = QecProgram::new(2);
    built.push_gate(Gate::X, &[0]).unwrap();
    built.measure_z(0).unwrap();
    built
        .feedforward(
            &[QecRecordRef::absolute(0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    built.measure_z(1).unwrap();
    assert_eq!(program.ops(), built.ops());

    let program = parse_stim_circuit(
        "X 0
         M 0
         H 2
         X 5
         CX 5 3 rec[-1] 1
         CZ 2 rec[-1]
         CY rec[-1] 4
         H 2
         M 1 2 3 4",
    )
    .unwrap();
    for shot in sample(program, 64).measurements.to_shots() {
        assert_eq!(shot, [true, true, true, true, true]);
    }
}

#[test]
fn stim_reader_rejects_unsupported_instructions_with_the_line() {
    let (line, message) = parse_error("R 0\nHERALDED_PAULI_CHANNEL_1(0.1, 0, 0, 0, 0) 0");
    assert_eq!(line, 2);
    assert!(message.contains("unsupported Stim instruction `HERALDED_PAULI_CHANNEL_1`"));

    let (line, message) = parse_error("H 0\nT 0");
    assert_eq!(line, 2);
    assert!(message.contains("`T` is a native QEC extension"));
    assert!(parse_qec_program("H 0\nT 0").is_ok());

    for (text, expected) in [
        ("M 0\nCX sweep[0] 1", "sweep target `sweep[0]`"),
        ("M 0\nCX 1 rec[-1]", "record-controlled pair `1 rec[-1]`"),
        ("MPP(0.01) Z0", "measurement-error"),
        ("MPAD(0.1) 0", "flip-probability"),
        ("M 0\nDETECTOR !rec[-1]", "inverted target"),
        ("H !0", "inverted target"),
        (
            "OBSERVABLE_INCLUDE(0) X0",
            "expected measurement record target",
        ),
    ] {
        let (_, message) = parse_error(text);
        assert!(message.contains(expected), "{text}: {message}");
    }
}

fn record_rates(result: &QecSampleResult) -> Vec<f64> {
    let shots = result.measurements.to_shots();
    (0..shots[0].len())
        .map(|record| shots.iter().filter(|s| s[record]).count() as f64 / shots.len() as f64)
        .collect()
}

fn reference(text: &str, shots: usize) -> QecSampleResult {
    let mut program = parse_stim_circuit(text).unwrap();
    program.set_options(qec_common::qec_options(shots, shots, true));
    run_qec_program_reference(&program).unwrap()
}

#[test]
fn stim_measurement_error_flips_only_the_record() {
    assert_eq!(
        parse_stim_circuit("M(0.2) 0\nM 0").unwrap().ops(),
        [
            QecOp::Measure {
                basis: QecBasis::Z,
                qubit: 0
            },
            QecOp::RecordFlip { probability: 0.2 },
            QecOp::Measure {
                basis: QecBasis::Z,
                qubit: 0
            },
        ]
    );

    // 4000 shots put one standard deviation of the 0.2 rate at 0.0063.
    let shots = 4000;
    for text in ["M(0.2) 0\nM 0", "M(0.2) 0\nH 0\nMX 0", "M(0.2) !0\nM 0"] {
        let rates = record_rates(&reference(text, shots));
        let first = if text.contains('!') { 0.8 } else { 0.2 };
        assert!((rates[0] - first).abs() < 0.03, "{text}: {rates:?}");
        assert_eq!(rates[1], 0.0, "{text}: the qubit itself was never flipped");
    }
}

#[test]
fn stim_measure_reset_and_basis_errors_flip_the_record() {
    let program = parse_stim_circuit("MR(0.2) 0\nM 0").unwrap();
    assert_eq!(
        program.ops()[1..3],
        [
            QecOp::RecordFlip { probability: 0.2 },
            QecOp::Reset {
                basis: QecBasis::Z,
                qubit: 0
            },
        ]
    );

    let shots = 4000;
    for text in [
        "MR(0.2) 0\nM 0",
        "RX 0\nMX(0.2) 0\nRY 1\nMY(0.2) 1",
        "RX 0\nMRX(0.2) 0\nMX 0",
    ] {
        let compiled = record_rates(&sample(parse_stim_circuit(text).unwrap(), shots));
        let reference = record_rates(&reference(text, shots));
        for rates in [&compiled, &reference] {
            assert!((rates[0] - 0.2).abs() < 0.03, "{text}: {rates:?}");
            let second = if text.contains("MY(") { 0.2 } else { 0.0 };
            assert!((rates[1] - second).abs() < 0.03, "{text}: {rates:?}");
        }
    }
}

#[test]
fn stim_measurement_error_is_a_detector_error_model_mechanism() {
    let program = parse_stim_circuit(
        "M(0.125) 0 1
         DETECTOR rec[-2]
         DETECTOR rec[-1] rec[-2]
         OBSERVABLE_INCLUDE(0) rec[-1]",
    )
    .unwrap();
    let model = program.detector_error_model().unwrap();
    let mechanisms: Vec<_> = model
        .mechanisms()
        .iter()
        .map(|m| {
            (
                m.detectors().to_vec(),
                m.observables().to_vec(),
                m.probability(),
            )
        })
        .collect();
    assert_eq!(
        mechanisms,
        [(vec![0, 1], vec![], 0.125), (vec![1], vec![0], 0.125),]
    );
}

#[test]
fn stim_measurement_error_samples_like_a_flip_before_a_last_measurement() {
    let noisy = SURFACE_D2_MEMORY
        .replace("MR 4 5 6", "MR(0.02) 4 5 6")
        .replace("M 0 1 2 3", "M(0.02) 0 1 2 3");
    let stim = parse_stim_circuit(&noisy).unwrap();
    let native = parse_qec_program(&noisy).unwrap();
    assert_ne!(without_coords(&stim), without_coords(&native));
    assert_eq!(
        stim.detector_error_model().unwrap().mechanisms(),
        native.detector_error_model().unwrap().mechanisms()
    );
    assert_same_samples(&sample(stim, 4096), &sample(native, 4096));
}

#[test]
fn stim_measurement_error_agrees_between_compiled_and_reference_runners() {
    let text = "R 0 1
                H 0
                CX 0 1
                MR(0.1) 0
                M(0.2) 1
                DETECTOR rec[-1] rec[-2]
                OBSERVABLE_INCLUDE(0) rec[-1]";
    let shots = 4000;
    let compiled = sample(parse_stim_circuit(text).unwrap(), shots);
    let reference = reference(text, shots);
    for result in [&compiled, &reference] {
        let fired = result.detectors.to_shots().iter().filter(|s| s[0]).count();
        let rate = fired as f64 / shots as f64;
        assert!(
            (rate - 0.26).abs() < 0.03,
            "detector fired at {rate}, expected 0.1 + 0.2 - 2 * 0.02"
        );
    }
}
