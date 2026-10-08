//! `QecProgram::to_text` against `parse_qec_program`: every instruction the parser
//! reads comes back as the same ops.

use prism_q::{Gate, QecBasis, QecNoise, QecPauli, QecProgram, QecRecordRef, parse_qec_program};

const EVERY_INSTRUCTION: &str = "
    QUBIT_COORDS(0, 0) 5
    R 0 1 2
    RX 3
    RY 4
    I 0
    X 0
    Y 1
    Z 2
    H 0 1
    S 0
    S_DAG 1
    T 2
    T_DAG 2
    CX 0 1 1 2
    CZ 2 3
    TICK
    X_ERROR(0.01) 0 1
    Y_ERROR(0.02) 2
    Z_ERROR(0.03) 3
    DEPOLARIZE1(0.04) 0
    DEPOLARIZE2(0.05) 0 1 2 3
    PAULI_CHANNEL_1(0.01, 0.02, 0.03) 4
    PAULI_CHANNEL_2(0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015) 0 4
    LEAK(0.01) 0 1
    LEAK(0) 2
    SEEP(0.2) 0
    LEAK_TRANSPORT(0.1) 0 1 2 3
    M(0.01) 0
    MX 3
    MY 4
    MR 1
    MPP X2*Z3 Y0
    DETECTOR(1.5, -2, 0.125) rec[-1] rec[-3]
    DETECTOR rec[-2]
    OBSERVABLE_INCLUDE(2) rec[-4]
    POSTSELECT rec[-5]
    POSTSELECT(1) rec[-6]
    REPEAT 2 {
        R 2
        MRX 2
        DETECTOR rec[-1]
    }
    EXP_VAL(-0.5) X2*Z5
";

#[test]
fn every_instruction_round_trips() {
    let program = parse_qec_program(EVERY_INSTRUCTION).unwrap();
    let text = program.to_text().unwrap();
    let parsed = parse_qec_program(&text).unwrap();
    assert_eq!(parsed.num_qubits(), program.num_qubits());
    assert_eq!(parsed.ops(), program.ops());
    assert_eq!(parsed.to_text().unwrap(), text);
}

#[test]
fn consecutive_ops_share_a_line() {
    let program = parse_qec_program("R 0 1\nH 0 1\nCX 0 1\nM 0 1\nMPP Z0 Z1").unwrap();
    assert_eq!(
        program.to_text().unwrap(),
        "R 0 1\nH 0 1\nCX 0 1\nM 0 1\nMPP Z0 Z1\n"
    );
}

#[test]
fn built_programs_resolve_to_the_same_rows() {
    let mut program = QecProgram::new(4);
    program.push_gate(Gate::H, &[0]).unwrap();
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.noise(QecNoise::YError(0.1), &[1]).unwrap();
    let a = program.measure_z(0).unwrap();
    program.measure(QecBasis::X, 1).unwrap();
    program
        .measure_pauli_product(&[QecPauli::x(2), QecPauli::y(3)])
        .unwrap();
    program
        .detector_with_coords(
            &[
                QecRecordRef::absolute(a),
                QecRecordRef::lookback(1).unwrap(),
            ],
            &[0.1, 0.2],
        )
        .unwrap();
    program
        .observable_include(0, &[QecRecordRef::lookback(2).unwrap()])
        .unwrap();
    program
        .postselect(&[QecRecordRef::absolute(a)], true)
        .unwrap();

    let parsed = parse_qec_program(&program.to_text().unwrap()).unwrap();
    assert_eq!(parsed.num_qubits(), 4);
    assert_eq!(
        parsed.detector_rows().unwrap(),
        program.detector_rows().unwrap()
    );
    assert_eq!(
        parsed.observable_rows().unwrap(),
        program.observable_rows().unwrap()
    );
    assert_eq!(
        parsed.postselection_rows().unwrap(),
        program.postselection_rows().unwrap()
    );
    assert_eq!(
        parsed.detector_error_model().unwrap(),
        program.detector_error_model().unwrap()
    );
}

#[test]
fn unused_high_qubits_keep_the_register_width() {
    let mut program = QecProgram::new(7);
    program.push_gate(Gate::H, &[1]).unwrap();
    let text = program.to_text().unwrap();
    assert_eq!(text, "QUBIT_COORDS 6\nH 1\n");
    assert_eq!(parse_qec_program(&text).unwrap().num_qubits(), 7);
    assert_eq!(QecProgram::new(0).to_text().unwrap(), "");
}

#[test]
fn unspellable_ops_are_rejected() {
    let mut program = QecProgram::new(2);
    program.push_gate(Gate::Swap, &[0, 1]).unwrap();
    assert!(program.to_text().is_err());

    let mut program = QecProgram::new(1);
    let record = program.measure_z(0).unwrap();
    program.reset(QecBasis::Z, 0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(record)],
            true,
            vec![prism_q::QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .unwrap();
    assert!(program.to_text().is_err());
}
