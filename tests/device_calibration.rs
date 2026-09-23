//! Device calibration import: the text form round-trips into a `NoiseModel`
//! whose events carry the table's values, bad input names its line, and a
//! preset's model agrees between the exact mixture and trajectory sampling.

mod common;

use common::SEED;
use num_complex::Complex64;
use prism_q::sim::calibration::presets;
use prism_q::sim::noise::{NoiseChannel, ReadoutError};
use prism_q::{
    BackendKind, Circuit, DeviceCalibration, Gate, GateCalibration, PrismError, QubitCalibration,
    simulate,
};

const FIXTURE: &str = include_str!("fixtures/device_calibration.txt");

fn four_qubit_circuit(measure: bool) -> Circuit {
    let mut c = Circuit::new(4, if measure { 4 } else { 0 });
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 1]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Rx(0.4), &[3]);
    c.add_gate(Gate::Cx, &[2, 3]);
    c.add_gate(Gate::Id, &[1]);
    if measure {
        for q in 0..4 {
            c.add_measure(q, q);
        }
    }
    c
}

#[test]
fn fixture_import_reproduces_qubit_values_and_family_counts() {
    let calibration = DeviceCalibration::parse(FIXTURE).unwrap();
    assert_eq!(calibration.num_qubits(), 4);
    assert_eq!(calibration.qubit(3).p01, 0.0);
    assert_eq!(calibration.gate2q_on(2, 1).time, 250e-9);
    assert_eq!(calibration.gate2q_on(0, 1).error, 8e-3);

    let circuit = four_qubit_circuit(true);
    let model = calibration.to_noise_model(&circuit).unwrap();
    assert_eq!(model.after_gate.len(), circuit.instructions.len());

    let expected: [(f64, f64); 4] = [
        (120e-6, 80e-6),
        (95e-6, 110e-6),
        (150e-6, 150e-6),
        (80e-6, 60e-6),
    ];
    let mut seen = [false; 4];
    for events in &model.after_gate {
        for event in events {
            if let NoiseChannel::ThermalRelaxation {
                t1,
                t2,
                excited_population,
                ..
            } = event.channel
            {
                let q = event.qubit();
                assert_eq!((t1, t2), expected[q], "qubit {q}");
                assert_eq!(excited_population, 0.0);
                seen[q] = true;
            }
        }
    }
    assert_eq!(seen, [true; 4]);

    // 1q gates: thermal + depolarizing; 2q gates: two thermal + one joint.
    let counts: Vec<usize> = model.after_gate.iter().map(Vec::len).collect();
    assert_eq!(counts, vec![2, 3, 3, 2, 3, 2, 0, 0, 0, 0]);
    let thermal_time = |idx: usize| match model.after_gate[idx][0].channel {
        NoiseChannel::ThermalRelaxation { gate_time, .. } => gate_time,
        ref other => panic!("instruction {idx}: {other:?}"),
    };
    assert_eq!(thermal_time(0), 35e-9);
    assert_eq!(thermal_time(1), 300e-9);
    assert_eq!(thermal_time(2), 250e-9);
    assert_eq!(
        model.after_gate[0][1].channel,
        NoiseChannel::Depolarizing { p: 3e-4 }
    );
    assert_eq!(
        model.after_gate[2][2].channel,
        NoiseChannel::TwoQubitDepolarizing { p: 5e-3 }
    );
    assert_eq!(model.after_gate[2][2].qubits.as_slice(), &[1, 2]);
    assert_eq!(
        model.after_gate[4][2].channel,
        NoiseChannel::TwoQubitDepolarizing { p: 8e-3 }
    );

    assert_eq!(
        model.readout,
        vec![
            Some(ReadoutError {
                p01: 0.02,
                p10: 0.03
            }),
            Some(ReadoutError {
                p01: 0.01,
                p10: 0.025
            }),
            Some(ReadoutError {
                p01: 0.015,
                p10: 0.04
            }),
            None,
        ]
    );
    model.validate_for(&circuit).unwrap();
}

#[test]
fn parse_error_names_the_line_and_field() {
    let cases: [(&str, usize, &str); 6] = [
        (
            "qubit 0 t1=1e-4 t2=3e-4\ngate1q time=1e-8 error=0\ngate2q time=1e-7 error=0",
            1,
            "t2 = 0.0003 exceeds twice t1",
        ),
        (
            "gate1q time=1e-8 error=0\nqubit 0 t1=1e-4 t3=1e-4\ngate2q time=1e-7 error=0",
            2,
            "unknown field `t3`",
        ),
        (
            "qubit 0 t1=1e-4 t2=1e-4\ngate1q time=1e-8\ngate2q time=1e-7 error=0",
            2,
            "field `error` is missing",
        ),
        (
            "qubit 0 t1=1e-4 t2=1e-4\ngate1q time=1e-8 error=0\ngate2q time=1e-7 error=1.5",
            3,
            "error = 1.5 must be finite and in [0, 1]",
        ),
        (
            "qubit 0 t1=1e-4 t2=1e-4\nqubit 1 t1=1e-4 t2=1e-4\ngate1q time=1e-8 error=0\ngate2q time=1e-7 error=0\ngate2q 0 3 time=1e-7 error=0",
            5,
            "gate2q 0 3: qubit 3 is outside the 2-qubit table",
        ),
        (
            "qubit 1 t1=1e-4 t2=1e-4\ngate1q time=1e-8 error=0\ngate2q time=1e-7 error=0",
            3,
            "qubit 0 is not given",
        ),
    ];
    for (text, line, needle) in cases {
        match DeviceCalibration::parse(text) {
            Err(PrismError::Parse { line: at, message }) => {
                assert_eq!(at, line, "{text:?}: {message}");
                assert!(message.contains(needle), "{text:?}: {message}");
            }
            other => panic!("{text:?}: expected a parse error, got {other:?}"),
        }
    }
}

#[test]
fn construction_names_the_qubit_and_field() {
    let good = QubitCalibration {
        t1: 1e-4,
        t2: 1e-4,
        p01: 0.0,
        p10: 0.0,
    };
    let gate = GateCalibration {
        time: 1e-8,
        error: 1e-3,
    };
    let bad = QubitCalibration { p10: 1.2, ..good };
    let err = DeviceCalibration::new(vec![good, bad], gate, gate).unwrap_err();
    assert!(
        err.to_string()
            .contains("qubit 1: p10 = 1.2 must be finite and in [0, 1]"),
        "{err}"
    );

    let err = DeviceCalibration::new(vec![good], GateCalibration { time: 0.0, ..gate }, gate)
        .unwrap_err();
    assert!(err.to_string().contains("gate1q: time = 0"), "{err}");

    let err = DeviceCalibration::new(vec![good, good], gate, gate)
        .unwrap()
        .with_pair(1, 1, gate)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("gate2q 1 1: a pair needs two distinct qubits"),
        "{err}"
    );
}

#[test]
fn model_rejects_a_wider_circuit_and_a_three_qubit_gate() {
    let calibration = presets::superconducting_transmon(2);
    let err = calibration
        .to_noise_model(&four_qubit_circuit(false))
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("circuit has 4 qubits but the calibration covers 2"),
        "{err}"
    );

    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut c = Circuit::new(3, 0);
    c.add_gate(Gate::mcu([[zero, one], [one, zero]], 2), &[0, 1, 2]);
    let err = presets::trapped_ion(3).to_noise_model(&c).unwrap_err();
    assert!(
        err.to_string()
            .contains("gate `mcu` at instruction 0 acts on 3 qubits"),
        "{err}"
    );
}

// The exact mixture and trajectory sampling both apply the preset's readout
// rates, so the two shot routes are compared on <Z> per qubit.
#[test]
fn preset_model_agrees_between_density_matrix_and_trajectories() {
    let num_shots = 20000;
    let circuit = four_qubit_circuit(true);
    let noise = presets::superconducting_transmon(4)
        .to_noise_model(&circuit)
        .unwrap();
    assert!(noise.has_noise());

    let z_from = |kind: BackendKind| -> Vec<f64> {
        let shots = simulate(&circuit)
            .backend(kind)
            .noise(&noise)
            .seed(SEED)
            .shots(num_shots)
            .unwrap();
        (0..4)
            .map(|q| {
                let ones = shots.shots.iter().filter(|record| record[q]).count() as f64;
                1.0 - 2.0 * ones / num_shots as f64
            })
            .collect()
    };
    let exact = z_from(BackendKind::DensityMatrix);
    let sampled = z_from(BackendKind::Statevector);

    let sigma = (2.0 / num_shots as f64).sqrt();
    for q in 0..4 {
        assert!(
            (sampled[q] - exact[q]).abs() <= 5.0 * sigma,
            "qubit {q}: trajectory <Z> = {}, density matrix = {}",
            sampled[q],
            exact[q]
        );
    }
}
