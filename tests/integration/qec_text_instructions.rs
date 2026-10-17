//! QEC text instructions beyond the original subset: square-root and axis-cycling
//! gates, `CY` / `SWAP` / `ISWAP` / `CXSWAP` / `SWAPCX`, pair measurements, `MPAD`,
//! correlated-error chains, and heralded erasure, each checked against a hand-derived
//! outcome on the compiled (stabilizer) and reference runners.

use prism_q::{
    Gate, PackedShots, QecBasis, QecOp, QecOptions, QecPauli, QecProgram, QecSampleResult,
    compile_qec_program_rows, parse_qec_program, run_qec_program, run_qec_program_reference,
};

const SEED: u64 = 42;

const COLOR_CODE_MEMORY_XYZ: &str = include_str!("../fixtures/qec/color_code_memory_xyz.txt");
const LATTICE_SURGERY_ZZ: &str = include_str!("../fixtures/qec/lattice_surgery_zz.txt");

fn seeded(text: &str, shots: usize) -> QecProgram {
    let mut program = parse_qec_program(text).unwrap();
    program.set_options(QecOptions {
        shots,
        seed: SEED,
        chunk_size: None,
        keep_measurements: true,
    });
    program
}

fn rate(shots: &PackedShots, column: usize) -> f64 {
    (0..shots.num_shots())
        .filter(|&shot| shots.get_bit(shot, column))
        .count() as f64
        / shots.num_shots() as f64
}

fn assert_rate(observed: f64, p: f64, n: usize, label: &str) {
    let tolerance = 5.0 * (p * (1.0 - p) / n as f64).sqrt().max(1e-3);
    assert!(
        (observed - p).abs() <= tolerance,
        "{label}: observed {observed}, expected {p} +- {tolerance}"
    );
}

fn records(result: &QecSampleResult, shot: usize) -> Vec<bool> {
    (0..result.measurements.num_measurements())
        .map(|record| result.measurements.get_bit(shot, record))
        .collect()
}

/// Both runners produce `expected` on every shot.
#[track_caller]
fn assert_constant_records(text: &str, expected: &[bool]) {
    let program = seeded(text, 32);
    for (runner, result) in [
        ("compiled", run_qec_program(&program).unwrap()),
        ("reference", run_qec_program_reference(&program).unwrap()),
    ] {
        assert_eq!(result.measurements.num_measurements(), expected.len());
        for shot in 0..32 {
            assert_eq!(records(&result, shot), expected, "{runner}: {text}");
        }
    }
}

fn gates(program: &QecProgram) -> Vec<(Gate, Vec<usize>)> {
    program
        .ops()
        .iter()
        .map(|op| match op {
            QecOp::Gate { gate, targets } => (gate.clone(), targets.clone()),
            other => panic!("expected a gate, got {other:?}"),
        })
        .collect()
}

#[test]
fn square_root_and_axis_cycle_gates_lower_to_clifford_sequences() {
    let cases: [(&str, &[Gate]); 8] = [
        ("SQRT_X 0", &[Gate::SX]),
        ("SQRT_X_DAG 0", &[Gate::SXdg]),
        ("SQRT_Y 0", &[Gate::Sdg, Gate::SX, Gate::S]),
        ("SQRT_Y_DAG 0", &[Gate::Sdg, Gate::SXdg, Gate::S]),
        ("SQRT_Z 0", &[Gate::S]),
        ("SQRT_Z_DAG 0", &[Gate::Sdg]),
        ("C_XYZ 0", &[Gate::Sdg, Gate::H]),
        ("C_ZYX 0", &[Gate::H, Gate::S]),
    ];
    for (text, expected) in cases {
        let lowered = gates(&parse_qec_program(text).unwrap());
        let expected: Vec<(Gate, Vec<usize>)> = expected
            .iter()
            .map(|gate| (gate.clone(), vec![0]))
            .collect();
        assert_eq!(lowered, expected, "{text}");
    }
    let two = gates(&parse_qec_program("SQRT_X 0 1").unwrap());
    assert_eq!(two, vec![(Gate::SX, vec![0]), (Gate::SX, vec![1])]);
}

#[test]
fn square_root_gates_rotate_by_a_quarter_turn() {
    // Two half turns make a full Pauli; a root followed by its dagger is the identity.
    assert_constant_records("R 0\nSQRT_X 0\nSQRT_X 0\nM 0", &[true]);
    assert_constant_records("R 0\nSQRT_X_DAG 0\nSQRT_X_DAG 0\nM 0", &[true]);
    assert_constant_records("R 0\nSQRT_X 0\nSQRT_X_DAG 0\nM 0", &[false]);
    // SQRT_X|0> is a -1 eigenstate of Y: SX = e^{i pi/4} Rx(pi/2) sends |0> to |-i>.
    assert_constant_records("R 0\nSQRT_X 0\nMY 0", &[true]);
    assert_constant_records("R 0\nSQRT_X_DAG 0\nMY 0", &[false]);
    // SQRT_Y|0> = |+>, SQRT_Y_DAG|0> = |->, and two SQRT_Y make a Y.
    assert_constant_records("R 0\nSQRT_Y 0\nMX 0", &[false]);
    assert_constant_records("R 0\nSQRT_Y_DAG 0\nMX 0", &[true]);
    assert_constant_records("R 0\nSQRT_Y 0\nSQRT_Y 0\nM 0", &[true]);
    assert_constant_records("R 0\nSQRT_Y 0\nSQRT_Y_DAG 0\nM 0", &[false]);
    // SQRT_Z|+> = |+i>, SQRT_Z_DAG|+> = |-i>.
    assert_constant_records("RX 0\nSQRT_Z 0\nMY 0", &[false]);
    assert_constant_records("RX 0\nSQRT_Z_DAG 0\nMY 0", &[true]);
}

#[test]
fn axis_cycles_send_x_to_y_to_z_and_back() {
    // C_XYZ: X -> Y -> Z -> X, eigenvalues preserved.
    assert_constant_records("RX 0\nC_XYZ 0\nMY 0", &[false]);
    assert_constant_records("RY 0\nC_XYZ 0\nM 0", &[false]);
    assert_constant_records("R 0\nC_XYZ 0\nMX 0", &[false]);
    assert_constant_records("R 0\nX 0\nC_XYZ 0\nMX 0", &[true]);
    assert_constant_records("RX 0\nC_XYZ 0\nC_XYZ 0\nC_XYZ 0\nMX 0", &[false]);
    // C_ZYX is the inverse cycle: X -> Z -> Y -> X.
    assert_constant_records("RX 0\nC_ZYX 0\nM 0", &[false]);
    assert_constant_records("R 0\nC_ZYX 0\nMY 0", &[false]);
    assert_constant_records("RY 0\nC_ZYX 0\nMX 0", &[false]);
    assert_constant_records("RX 0\nZ 0\nC_ZYX 0\nM 0", &[true]);
    assert_constant_records("RX 0\nC_XYZ 0\nC_ZYX 0\nMX 0", &[false]);
}

#[test]
fn two_qubit_gate_spellings_lower_and_act_as_defined() {
    let cy = gates(&parse_qec_program("CY 0 1").unwrap());
    assert_eq!(
        cy,
        vec![
            (Gate::Sdg, vec![1]),
            (Gate::Cx, vec![0, 1]),
            (Gate::S, vec![1])
        ]
    );
    let swap = |a: usize, b: usize| {
        vec![
            (Gate::Cx, vec![a, b]),
            (Gate::Cx, vec![b, a]),
            (Gate::Cx, vec![a, b]),
        ]
    };
    assert_eq!(
        gates(&parse_qec_program("SWAP 0 1 2 3").unwrap()),
        [swap(0, 1), swap(2, 3)].concat()
    );
    assert_eq!(
        gates(&parse_qec_program("ISWAP 0 1").unwrap()),
        [
            vec![
                (Gate::S, vec![0]),
                (Gate::S, vec![1]),
                (Gate::Cz, vec![0, 1])
            ],
            swap(0, 1)
        ]
        .concat()
    );
    assert_eq!(
        gates(&parse_qec_program("CXSWAP 0 1").unwrap()),
        vec![(Gate::Cx, vec![1, 0]), (Gate::Cx, vec![0, 1])]
    );
    assert_eq!(
        gates(&parse_qec_program("SWAPCX 0 1").unwrap()),
        vec![(Gate::Cx, vec![0, 1]), (Gate::Cx, vec![1, 0])]
    );

    // CY flips the target iff the control is set, and acts as Y on it.
    assert_constant_records("R 0 1\nCY 0 1\nM 0 1", &[false, false]);
    assert_constant_records("R 0 1\nX 0\nCY 0 1\nM 0 1", &[true, true]);
    assert_constant_records("R 0\nRX 1\nX 0\nCY 0 1\nMX 1", &[true]);
    assert_constant_records("R 0\nRX 1\nCY 0 1\nMX 1", &[false]);
    // SWAP exchanges the qubits.
    assert_constant_records("R 0 1\nX 0\nSWAP 0 1\nM 0 1", &[false, true]);
    // ISWAP swaps and phases |01> and |10> by i: |+>|0> becomes |0>|+i>.
    assert_constant_records("R 0 1\nX 0\nISWAP 0 1\nM 0 1", &[false, true]);
    assert_constant_records("RX 0\nR 1\nISWAP 0 1\nM 0\nMY 1", &[false, false]);
    assert_constant_records("RX 0\nR 1\nISWAP_DAG 0 1\nM 0\nMY 1", &[false, true]);
    assert_constant_records(
        "RX 0\nR 1\nISWAP 0 1\nISWAP_DAG 0 1\nMX 0\nM 1",
        &[false, false],
    );
    // CXSWAP is CX then SWAP; SWAPCX is SWAP then CX.
    assert_constant_records("R 0 1\nX 1\nCXSWAP 0 1\nM 0 1", &[true, false]);
    assert_constant_records("R 0 1\nX 0\nCXSWAP 0 1\nM 0 1", &[true, true]);
    assert_constant_records("R 0 1\nX 1\nSWAPCX 0 1\nM 0 1", &[true, true]);
    assert_constant_records("R 0 1\nX 0\nSWAPCX 0 1\nM 0 1", &[false, true]);

    for text in ["CY 0", "SWAP 0 1 2", "ISWAP", "CXSWAP 1"] {
        let err = parse_qec_program(text).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("even number") || message.contains("at least one target"),
            "{text}: {message}"
        );
    }
}

#[test]
fn pair_measurements_lower_to_pauli_products() {
    let program = parse_qec_program("MXX 0 1\nMYY 2 3\nMZZ 0 1 2 3").unwrap();
    assert_eq!(program.num_measurements(), 4);
    assert_eq!(
        program.ops(),
        &[
            QecOp::MeasurePauliProduct {
                terms: vec![QecPauli::x(0), QecPauli::x(1)],
            },
            QecOp::MeasurePauliProduct {
                terms: vec![QecPauli::y(2), QecPauli::y(3)],
            },
            QecOp::MeasurePauliProduct {
                terms: vec![QecPauli::z(0), QecPauli::z(1)],
            },
            QecOp::MeasurePauliProduct {
                terms: vec![QecPauli::z(2), QecPauli::z(3)],
            },
        ]
    );

    assert_constant_records("RX 0 1\nMXX 0 1", &[false]);
    assert_constant_records("RX 0 1\nZ 0\nMXX 0 1", &[true]);
    assert_constant_records("RY 0 1\nMYY 0 1", &[false]);
    assert_constant_records("RY 0 1\nX 1\nMYY 0 1", &[true]);
    assert_constant_records("R 0 1\nMZZ 0 1", &[false]);
    assert_constant_records("R 0 1\nX 0\nMZZ 0 1", &[true]);
    // A pair record agrees with the MPP spelling and with a detector over both.
    assert_constant_records(
        "R 0 1 2 3\nX 1\nMZZ 0 1 2 3\nMPP Z0*Z1\nDETECTOR rec[-1] rec[-3]",
        &[true, false, true],
    );

    let err = parse_qec_program("MZZ(0.1) 0 1").unwrap_err();
    assert!(format!("{err}").contains("measurement-error"));
    let err = parse_qec_program("MXX 0 1 2").unwrap_err();
    assert!(format!("{err}").contains("even number"));
    let err = parse_qec_program("MXX 0 0").unwrap_err();
    assert!(format!("{err}").contains("duplicate"));
}

#[test]
fn mpad_appends_constant_records() {
    let program = parse_qec_program("MPAD 1 0 1").unwrap();
    assert_eq!(program.num_measurements(), 3);
    assert_eq!(
        program.ops(),
        &[
            QecOp::Pad { value: true },
            QecOp::Pad { value: false },
            QecOp::Pad { value: true }
        ]
    );
    assert_constant_records("MPAD 1 0 1", &[true, false, true]);
    assert_constant_records("R 0\nM 0\nMPAD 1\nDETECTOR rec[-1] rec[-2]", &[false, true]);
    assert_constant_records(
        "R 0 1\nMPP Z0*Z1\nMPAD 0\nMPP Z0*Z1\nMPAD 1",
        &[false, false, false, true],
    );

    // The noisy sampler keeps the padded record constant and the detector reads it.
    let noisy = seeded("R 0\nX_ERROR(0.5) 0\nM 0\nMPAD 1\nDETECTOR rec[-1]", 4096);
    let result = run_qec_program(&noisy).unwrap();
    assert_eq!(rate(&result.measurements, 1), 1.0);
    assert_eq!(rate(&result.detectors, 0), 1.0);
    assert_rate(rate(&result.measurements, 0), 0.5, 4096, "flip before pad");

    let err = parse_qec_program("MPAD 2").unwrap_err();
    assert!(format!("{err}").contains("0 or 1"));
    let err = compile_qec_program_rows(&parse_qec_program("MPAD 0").unwrap()).unwrap_err();
    assert!(format!("{err}").contains("MPAD"));
}

#[test]
fn correlated_error_is_deterministic_at_zero_and_one() {
    let program = parse_qec_program("E(0.25) X0 Y1 Z2\nCORRELATED_ERROR(0.5) Z0*X3").unwrap();
    let branches = |op: &QecOp| match op {
        QecOp::CorrelatedError { branches } => branches.clone(),
        other => panic!("expected a correlated error, got {other:?}"),
    };
    let first = branches(&program.ops()[0]);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].probability, 0.25);
    assert_eq!(
        first[0].terms,
        vec![QecPauli::x(0), QecPauli::y(1), QecPauli::z(2)]
    );
    let second = branches(&program.ops()[1]);
    assert_eq!(second[0].terms, vec![QecPauli::z(0), QecPauli::x(3)]);
    assert_eq!(program.num_qubits(), 4);

    assert_constant_records("R 0 1 2\nE(1) X0 Y1 Z2\nM 0 1 2", &[true, true, false]);
    assert_constant_records("R 0 1 2\nE(0) X0 Y1 Z2\nM 0 1 2", &[false, false, false]);
    assert_constant_records(
        "RX 0 1\nCORRELATED_ERROR(1) Z0 Y1\nMX 0 1\nDETECTOR rec[-1] rec[-2]",
        &[true, true],
    );
    assert_constant_records("R 0 1\nE(1) X0\nE(1) X0 X1\nM 0 1", &[false, true]);

    for text in [
        "E(1.5) X0",
        "E(-1) X0",
        "E X0",
        "E(0.5)",
        "E(0.5) 0",
        "E(0.5) X0 Z0",
    ] {
        let err = parse_qec_program(text).unwrap_err();
        assert!(!format!("{err}").is_empty(), "{text}");
    }
}

#[test]
fn else_correlated_error_fires_only_when_the_chain_did_not() {
    let program = parse_qec_program("E(0.1) X0\nELSE_CORRELATED_ERROR(0.5) X1").unwrap();
    assert_eq!(program.ops().len(), 1);
    let QecOp::CorrelatedError { branches } = &program.ops()[0] else {
        panic!("chain is one op");
    };
    assert_eq!(branches.len(), 2);
    assert_eq!(branches[1].probability, 0.5);
    assert_eq!(branches[1].terms, vec![QecPauli::x(1)]);

    assert_constant_records(
        "R 0 1\nE(0) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1",
        &[false, true],
    );
    assert_constant_records(
        "R 0 1\nE(1) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1",
        &[true, false],
    );
    assert_constant_records(
        "R 0 1 2\nE(0) X0\nELSE_CORRELATED_ERROR(0) X1\nELSE_CORRELATED_ERROR(1) X2\nM 0 1 2",
        &[false, false, true],
    );

    for text in [
        "ELSE_CORRELATED_ERROR(0.5) X0",
        "X_ERROR(0.1) 0\nELSE_CORRELATED_ERROR(0.5) X0",
        "E(0.1) X0\nTICK\nELSE_CORRELATED_ERROR(0.5) X0",
    ] {
        let err = parse_qec_program(text).unwrap_err();
        assert!(
            format!("{err}").contains("must directly follow"),
            "{text}: {err}"
        );
    }
}

#[test]
fn correlated_error_chains_sample_their_branch_rates() {
    let n = 20_000;
    let text = "R 0 1 2\nE(0.3) X0 X1\nELSE_CORRELATED_ERROR(0.5) X2\nM 0 1 2";
    let compiled = run_qec_program(&seeded(text, n)).unwrap();
    let reference = run_qec_program_reference(&seeded(text, 2000)).unwrap();
    for (label, result, shots) in [("compiled", &compiled, n), ("reference", &reference, 2000)] {
        assert_rate(rate(&result.measurements, 0), 0.3, shots, label);
        assert_rate(rate(&result.measurements, 2), 0.35, shots, label);
        for shot in 0..shots {
            let bits = records(result, shot);
            assert_eq!(bits[0], bits[1], "{label}: the product fires as a whole");
            assert!(!(bits[0] && bits[2]), "{label}: branches are exclusive");
        }
    }

    // A chain whose branches exhaust the probability fires exactly one branch per shot.
    let text = "R 0 1\nE(0.5) X0\nELSE_CORRELATED_ERROR(1) X1\nM 0 1";
    let result = run_qec_program(&seeded(text, n)).unwrap();
    assert_rate(rate(&result.measurements, 0), 0.5, n, "first branch");
    for shot in 0..n {
        let bits = records(&result, shot);
        assert!(bits[0] ^ bits[1], "exactly one branch fires");
    }

    // The detector error model carries the chain's absolute rates.
    let dem = parse_qec_program(
        "R 0 1\nE(0.1) X0\nELSE_CORRELATED_ERROR(0.5) X1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]",
    )
    .unwrap()
    .detector_error_model()
    .unwrap();
    let mut mechanisms: Vec<(Vec<usize>, f64)> = dem
        .mechanisms()
        .iter()
        .map(|m| (m.detectors().to_vec(), m.probability()))
        .collect();
    mechanisms.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(mechanisms.len(), 2);
    assert_eq!(mechanisms[0].0, vec![0]);
    assert!((mechanisms[0].1 - 0.1).abs() < 1e-12);
    assert_eq!(mechanisms[1].0, vec![1]);
    assert!((mechanisms[1].1 - 0.45).abs() < 1e-12);
}

#[test]
fn heralded_erase_records_the_herald_and_erases_on_it() {
    let program = parse_qec_program("HERALDED_ERASE(0.25) 0 1").unwrap();
    assert_eq!(program.num_measurements(), 2);
    assert_eq!(
        program.ops(),
        &[
            QecOp::HeraldedErase {
                probability: 0.25,
                qubit: 0
            },
            QecOp::HeraldedErase {
                probability: 0.25,
                qubit: 1
            },
        ]
    );

    assert_constant_records(
        "R 0\nHERALDED_ERASE(0) 0\nM 0\nDETECTOR rec[-2]",
        &[false, false],
    );
    // At p = 1 the herald is certain and the qubit is maximally mixed.
    let n = 8192;
    let text = "R 0\nHERALDED_ERASE(1) 0\nM 0\nDETECTOR rec[-2]";
    for (label, result, shots) in [
        ("compiled", run_qec_program(&seeded(text, n)).unwrap(), n),
        (
            "reference",
            run_qec_program_reference(&seeded(text, 2000)).unwrap(),
            2000,
        ),
    ] {
        assert_eq!(rate(&result.measurements, 0), 1.0, "{label}");
        assert_eq!(rate(&result.detectors, 0), 1.0, "{label}");
        assert_rate(rate(&result.measurements, 1), 0.5, shots, label);
    }
    // An X-basis qubit is erased too: Y and Z flip its record.
    let result = run_qec_program(&seeded("RX 0\nHERALDED_ERASE(1) 0\nMX 0", n)).unwrap();
    assert_rate(rate(&result.measurements, 1), 0.5, n, "x basis");

    // At p = 0.3 the data record flips only on heralded shots, half of them.
    let text = "R 0\nHERALDED_ERASE(0.3) 0\nM 0";
    for (label, result, shots) in [
        ("compiled", run_qec_program(&seeded(text, n)).unwrap(), n),
        (
            "reference",
            run_qec_program_reference(&seeded(text, 2000)).unwrap(),
            2000,
        ),
    ] {
        assert_rate(rate(&result.measurements, 0), 0.3, shots, label);
        let mut heralded = 0;
        let mut flipped = 0;
        for shot in 0..shots {
            let bits = records(&result, shot);
            assert!(bits[0] || !bits[1], "{label}: a flip without a herald");
            heralded += usize::from(bits[0]);
            flipped += usize::from(bits[1]);
        }
        assert_rate(flipped as f64 / heralded as f64, 0.5, heralded, label);
    }

    let dem =
        parse_qec_program("R 0\nHERALDED_ERASE(0.2) 0\nM 0\nDETECTOR rec[-2]\nDETECTOR rec[-1]")
            .unwrap()
            .detector_error_model()
            .unwrap();
    let mut mechanisms: Vec<(Vec<usize>, f64)> = dem
        .mechanisms()
        .iter()
        .map(|m| (m.detectors().to_vec(), m.probability()))
        .collect();
    mechanisms.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(mechanisms.len(), 2);
    assert_eq!(mechanisms[0].0, vec![0]);
    assert!((mechanisms[0].1 - 0.1).abs() < 1e-12, "{mechanisms:?}");
    assert_eq!(mechanisms[1].0, vec![0, 1]);
    assert!((mechanisms[1].1 - 0.1).abs() < 1e-12, "{mechanisms:?}");

    for text in [
        "HERALDED_ERASE(2) 0",
        "HERALDED_ERASE 0",
        "HERALDED_ERASE(0.1)",
    ] {
        assert!(parse_qec_program(text).is_err(), "{text}");
    }
    let err =
        compile_qec_program_rows(&parse_qec_program("HERALDED_ERASE(0) 0").unwrap()).unwrap_err();
    assert!(format!("{err}").contains("HERALDED_ERASE"));
}

#[test]
fn new_instructions_round_trip_through_text() {
    let text = "
        R 0 1 2 3
        SQRT_X 0
        SQRT_X_DAG 1
        SQRT_Y 2
        SQRT_Y_DAG 3
        SQRT_Z 0
        SQRT_Z_DAG 1
        C_XYZ 2
        C_ZYX 3
        CY 0 1
        SWAP 0 1 2 3
        ISWAP 0 1
        ISWAP_DAG 2 3
        CXSWAP 0 1
        SWAPCX 2 3
        E(0.01) X0 Y1
        ELSE_CORRELATED_ERROR(0.02) Z2
        ELSE_CORRELATED_ERROR(0.03) X3 Z0
        CORRELATED_ERROR(0.04) Z3
        HERALDED_ERASE(0.05) 0 1
        MXX 0 1
        MYY 2 3
        MZZ 0 1 2 3
        MPAD 0 1
        DETECTOR rec[-1] rec[-8]
    ";
    let program = parse_qec_program(text).unwrap();
    let rendered = program.to_text().unwrap();
    let parsed = parse_qec_program(&rendered).unwrap();
    assert_eq!(parsed.ops(), program.ops());
    assert_eq!(parsed.num_qubits(), program.num_qubits());
    assert_eq!(parsed.to_text().unwrap(), rendered);
    assert!(rendered.contains("SQRT_X 0\nSQRT_X_DAG 1\n"), "{rendered}");
    assert!(
        rendered.contains("CX 0 1 1 0 0 1 2 3 3 2 2 3\n"),
        "{rendered}"
    );
    assert!(rendered.contains("E(0.01) X0 Y1\nELSE_CORRELATED_ERROR(0.02) Z2\n"));
    assert!(rendered.contains("HERALDED_ERASE(0.05) 0\nHERALDED_ERASE(0.05) 1\n"));
    assert!(rendered.contains("MPAD 0\nMPAD 1\n"));
}

#[test]
fn heralded_erase_and_pad_run_with_leakage_annotations() {
    let text = "R 0 1\nLEAK(0.1) 0\nHERALDED_ERASE(0.2) 1\nMPAD 1\nM 0 1";
    let n = 8192;
    let result = run_qec_program(&seeded(text, n)).unwrap();
    assert_eq!(result.heralds.as_ref().unwrap().num_measurements(), 1);
    assert_rate(rate(&result.measurements, 0), 0.2, n, "erasure herald");
    assert_eq!(rate(&result.measurements, 1), 1.0);
    assert_rate(rate(&result.measurements, 2), 0.1, n, "leaked record");
}

#[test]
fn exp_val_rejects_records_and_active_chains_after_it() {
    for text in [
        "R 0\nEXP_VAL Z0\nMPAD 1",
        "R 0\nEXP_VAL Z0\nHERALDED_ERASE(0) 0",
        "R 0\nEXP_VAL Z0\nE(0.1) X0",
    ] {
        let err = run_qec_program(&seeded(text, 8)).unwrap_err();
        assert!(
            format!("{err}").contains("must be terminal"),
            "{text}: {err}"
        );
    }
    let result = run_qec_program(&seeded("R 0\nE(0) X0\nEXP_VAL Z0", 8)).unwrap();
    let estimate = result.expectation_values.unwrap()[0];
    assert!((estimate.mean - 1.0).abs() < 1e-9);
}

#[test]
fn dropped_records_match_kept_projection_for_new_ops() {
    let shots = 8192;
    for text in [
        "R 0 1\nE(0.2) X0 Z1\nELSE_CORRELATED_ERROR(0.5) X1\nM 0\nMX 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0 1\nHERALDED_ERASE(0.3) 0\nCX 0 1\nM 0 1\nDETECTOR rec[-3]\nDETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]",
        "R 0\nX_ERROR(0.1) 0\nMPAD 1\nM 0\nMPAD 0\nDETECTOR rec[-3] rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2] rec[-3]",
        LATTICE_SURGERY_ZZ,
        COLOR_CODE_MEMORY_XYZ,
    ] {
        let mut program = seeded(text, shots);
        let kept = run_qec_program(&program).unwrap();
        program.set_options(QecOptions {
            keep_measurements: false,
            ..program.options()
        });
        let dropped = run_qec_program(&program).unwrap();
        assert_eq!(dropped.measurements.num_shots(), 0);
        let projected = kept
            .measurements
            .parity_rows(&program.detector_rows().unwrap())
            .unwrap();
        assert_eq!(kept.detectors.to_shots(), projected.to_shots(), "{text}");
        assert_eq!(dropped.detectors.to_shots(), kept.detectors.to_shots(), "{text}");
        assert_eq!(
            dropped.observables.to_shots(),
            kept.observables.to_shots(),
            "{text}"
        );
        assert_eq!(dropped.logical_errors, kept.logical_errors, "{text}");
    }
}

/// Replace every `(p)` probability in a fixture with zero.
fn noiseless(text: &str) -> String {
    text.replace("(0.001)", "(0)")
}

#[test]
fn color_code_memory_xyz_fixture_imports_and_runs() {
    let program = seeded(COLOR_CODE_MEMORY_XYZ, 512);
    assert_eq!(program.num_qubits(), 7);
    assert_eq!(program.num_measurements(), 6 * 3 + 7);
    assert_eq!(program.num_detectors(), 3 + 6 * 2 + 3);
    assert_eq!(program.num_observables(), 1);
    assert!(program.ops().iter().any(|op| matches!(
        op,
        QecOp::Gate {
            gate: Gate::Sdg,
            ..
        }
    )));

    let clean = seeded(&noiseless(COLOR_CODE_MEMORY_XYZ), 256);
    for result in [
        run_qec_program(&clean).unwrap(),
        run_qec_program_reference(&clean).unwrap(),
    ] {
        for detector in 0..clean.num_detectors() {
            assert_eq!(
                rate(&result.detectors, detector),
                0.0,
                "detector {detector}"
            );
        }
        assert_eq!(result.logical_errors, vec![0]);
    }

    let noisy = run_qec_program(&program).unwrap();
    assert_eq!(noisy.total_shots, 512);
    let dem = program.detector_error_model().unwrap();
    assert!(dem.num_mechanisms() > 0);
    assert_eq!(dem.num_detectors(), program.num_detectors());
}

#[test]
fn lattice_surgery_fixture_imports_and_runs() {
    let program = seeded(LATTICE_SURGERY_ZZ, 512);
    assert_eq!(program.num_qubits(), 5);
    assert_eq!(program.num_measurements(), 13);
    assert_eq!(program.num_detectors(), 7);
    assert_eq!(program.num_observables(), 2);

    let clean = seeded(&noiseless(LATTICE_SURGERY_ZZ), 256);
    for result in [
        run_qec_program(&clean).unwrap(),
        run_qec_program_reference(&clean).unwrap(),
    ] {
        for detector in 0..clean.num_detectors() {
            assert_eq!(
                rate(&result.detectors, detector),
                0.0,
                "detector {detector}"
            );
        }
        assert_eq!(result.logical_errors, vec![0, 0]);
        assert_eq!(rate(&result.measurements, 6), 0.0, "padded record");
        assert_rate(rate(&result.measurements, 5), 0.5, 256, "split MX");
    }

    let noisy = run_qec_program(&program).unwrap();
    assert_eq!(noisy.total_shots, 512);
    let dem = program.detector_error_model().unwrap();
    assert!(dem.mechanisms().iter().any(|m| m.detectors() == [2]));
}

#[test]
fn measurement_bases_still_reject_inverted_targets() {
    for text in [
        "MXX !0 1",
        "MPAD !1",
        "HERALDED_ERASE(0.1) !0",
        "E(0.1) !X0",
    ] {
        let err = parse_qec_program(text).unwrap_err();
        let message = format!("{err}");
        assert!(
            message.contains("inverted target") || message.contains("0 or 1"),
            "{text}: {message}"
        );
    }
    let _ = QecBasis::X;
}
