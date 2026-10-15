//! Leakage annotations on the compiled QEC sampler: herald columns, forced records,
//! partner erasure, chunking invariance, and agreement in distribution with
//! trajectory leakage on the same circuit.

use std::collections::HashMap;

use prism_q::circuit::Circuit;
use prism_q::{
    BackendKind, Gate, NoiseChannel, NoiseEvent, NoiseModel, PackedShots, QecOptions, QecProgram,
    QecSampleResult, run_qec_program, run_qec_program_reference, simulate,
};

const SEED: u64 = 42;

fn program(text: &str, shots: usize, chunk_size: Option<usize>) -> QecProgram {
    let mut program = QecProgram::from_text(text).unwrap();
    program.set_options(QecOptions {
        shots,
        seed: SEED,
        chunk_size,
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

#[test]
fn leaked_qubit_reads_one_and_erases_its_partner() {
    let n = 20_000;
    for text in [
        "R 0 1\nLEAK(1) 0\nCX 0 1\nM 0 1",
        "R 0 1\nH 1\nLEAK(1) 0\nCX 1 0\nH 1\nM 0 1",
        "R 0 1\nX 0\nLEAK(1) 0\nX 0\nMX 0\nM 1",
    ] {
        let result = run_qec_program(&program(text, n, None)).unwrap();
        assert_eq!(rate(&result.measurements, 0), 1.0, "{text}");
        let heralds = result.heralds.as_ref().unwrap();
        assert_eq!(heralds.num_measurements(), 1);
        assert_eq!(rate(heralds, 0), 1.0);
        let expected = if text.contains("CX") { 0.5 } else { 0.0 };
        assert_rate(rate(&result.measurements, 1), expected, n, text);
    }
}

#[test]
fn pauli_noise_naming_a_leaked_qubit_is_skipped_in_program_order() {
    let n = 4_000;
    let skipped = "R 0 1\nLEAK(1) 0\nDEPOLARIZE2(0.9375) 0 1\nX_ERROR(1) 0\nM 0 1";
    let result = run_qec_program(&program(skipped, n, None)).unwrap();
    assert_eq!(rate(&result.measurements, 0), 1.0);
    assert_eq!(rate(&result.measurements, 1), 0.0);

    let applied = "R 0 1\nDEPOLARIZE2(0.9375) 0 1\nLEAK(1) 0\nM 0 1";
    let result = run_qec_program(&program(applied, n, None)).unwrap();
    assert_rate(
        rate(&result.measurements, 1),
        0.5,
        n,
        "applied before the leak",
    );
}

#[test]
fn every_pauli_channel_draws_beside_leakage() {
    let n = 20_000;
    let text = "R 0 1 2 3 4 5
        LEAK(0) 0
        Y_ERROR(1) 0
        PAULI_CHANNEL_1(0, 0, 1) 1
        X_ERROR(1) 1
        PAULI_CHANNEL_2(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 2 3
        PAULI_CHANNEL_2(0, 0.25, 0, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) 4 5
        M 0 1 2 3 4 5";
    let result = run_qec_program(&program(text, n, None)).unwrap();
    for column in 0..4 {
        assert_eq!(rate(&result.measurements, column), 1.0, "column {column}");
    }
    // `XI` and `IY` each fire a quarter of the time, never together.
    assert_rate(rate(&result.measurements, 4), 0.25, n, "pair table first");
    assert_rate(rate(&result.measurements, 5), 0.25, n, "pair table second");
    let both = (0..n)
        .filter(|&shot| {
            result.measurements.get_bit(shot, 4) && result.measurements.get_bit(shot, 5)
        })
        .count();
    assert_eq!(both, 0);
}

#[test]
fn heralds_follow_leak_seep_and_reset() {
    let n = 40_000;
    let text = "R 0 1
        LEAK(0.25) 0 1
        SEEP(0.4) 0
        LEAK(0) 0 1
        R 1
        LEAK(0) 1
        M 0 1";
    let result = run_qec_program(&program(text, n, None)).unwrap();
    let heralds = result.heralds.as_ref().unwrap();
    assert_eq!(heralds.num_measurements(), 5);
    assert_rate(rate(heralds, 0), 0.25, n, "leak 0");
    assert_rate(rate(heralds, 1), 0.25, n, "leak 1");
    assert_rate(rate(heralds, 2), 0.25 * 0.6, n, "after seep");
    assert_rate(rate(heralds, 3), 0.25, n, "qubit 1 unchanged");
    assert_eq!(rate(heralds, 4), 0.0, "reset clears the flag");
    // A leaked qubit reads 1; one that seeped back reads 1 half the time.
    assert_rate(
        rate(&result.measurements, 0),
        0.25 * 0.6 + 0.25 * 0.4 * 0.5,
        n,
        "qubit 0 record",
    );
    for shot in 0..n {
        if heralds.get_bit(shot, 2) {
            assert!(result.measurements.get_bit(shot, 0));
        }
    }
}

#[test]
fn chunked_runs_draw_the_same_records_and_heralds() {
    // Noiseless records here are deterministic, so only the noise units decide them,
    // and those draw the same bits whatever the chunking.
    let text = "R 0 1 2 3
        X 0
        CX 0 1 1 2
        LEAK(0.1) 0 1 2
        DEPOLARIZE2(0.05) 1 2
        CX 1 3 2 3
        LEAK_TRANSPORT(0.5) 1 3
        X_ERROR(0.02) 3
        SEEP(0.3) 1
        M 0 1 2 3
        DETECTOR rec[-1] rec[-2]";
    let whole = run_qec_program(&program(text, 20_000, None)).unwrap();
    let chunked = run_qec_program(&program(text, 20_000, Some(1_000))).unwrap();
    let odd = run_qec_program(&program(text, 20_000, Some(3_333))).unwrap();
    for other in [&chunked, &odd] {
        assert_eq!(
            whole.heralds.as_ref().unwrap().raw_data(),
            other.heralds.as_ref().unwrap().raw_data()
        );
        for shot in 0..20_000 {
            for record in 0..4 {
                assert_eq!(
                    whole.measurements.get_bit(shot, record),
                    other.measurements.get_bit(shot, record),
                    "shot {shot} record {record}"
                );
            }
        }
    }
}

#[test]
fn engines_without_leak_flags_decline_leakage() {
    let program = program("R 0\nLEAK(0.1) 0\nM 0\nDETECTOR rec[-1]", 16, None);
    assert!(program.detector_error_model().is_err());
    assert!(run_qec_program_reference(&program).is_err());
    let plain = run_qec_program(&self::program("R 0\nX_ERROR(0.1) 0\nM 0", 16, None)).unwrap();
    assert!(plain.heralds.is_none());
}

/// Joint counts of the three records and the herald of qubit 0.
fn qec_counts(result: &QecSampleResult) -> HashMap<[bool; 4], usize> {
    let heralds = result.heralds.as_ref().unwrap();
    let mut counts = HashMap::new();
    for shot in 0..result.total_shots {
        let key = [
            result.measurements.get_bit(shot, 0),
            result.measurements.get_bit(shot, 1),
            result.measurements.get_bit(shot, 2),
            heralds.get_bit(shot, 0),
        ];
        *counts.entry(key).or_default() += 1;
    }
    counts
}

#[test]
fn erasure_heralds_agree_with_trajectory_leakage_in_distribution() {
    let n = 40_000;
    let text = "R 0 1 2
        H 0
        CX 0 1
        LEAK(0.2) 0
        CX 0 2
        LEAK_TRANSPORT(0.3) 0 2
        CX 1 2
        H 1
        M 0 1 2";
    let qec = qec_counts(&run_qec_program(&program(text, n, None)).unwrap());

    let mut circuit = Circuit::new(3, 3);
    circuit.add_gate(Gate::H, &[0]);
    circuit.add_gate(Gate::Cx, &[0, 1]);
    circuit.add_gate(Gate::Cx, &[0, 2]);
    circuit.add_gate(Gate::Cx, &[1, 2]);
    circuit.add_gate(Gate::H, &[1]);
    for q in 0..3 {
        circuit.add_measure(q, q);
    }
    let mut noise = NoiseModel::uniform_depolarizing(&circuit, 0.0);
    for slot in &mut noise.after_gate {
        slot.clear();
    }
    noise.after_gate[1].push(NoiseEvent {
        channel: NoiseChannel::Leakage { p: 0.2 },
        qubits: [0].into_iter().collect(),
    });
    noise.after_gate[2].push(NoiseEvent {
        channel: NoiseChannel::LeakageTransport { p: 0.3 },
        qubits: [0, 2].into_iter().collect(),
    });
    let result = simulate(&circuit)
        .backend(BackendKind::Statevector)
        .noise(&noise)
        .seed(SEED)
        .shots(n)
        .unwrap();
    let leaked = result.leaked.as_ref().unwrap();
    let mut trajectory: HashMap<[bool; 4], usize> = HashMap::new();
    for (shot, flags) in result.shots.iter().zip(leaked) {
        *trajectory
            .entry([shot[0], shot[1], shot[2], flags[0]])
            .or_default() += 1;
    }

    let mut keys: Vec<_> = qec.keys().chain(trajectory.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    for key in keys {
        let a = *qec.get(&key).unwrap_or(&0) as f64 / n as f64;
        let b = *trajectory.get(&key).unwrap_or(&0) as f64 / n as f64;
        let p = (a + b) / 2.0;
        let tolerance = 5.0 * (2.0 * p * (1.0 - p) / n as f64).sqrt() + 1e-3;
        assert!(
            (a - b).abs() <= tolerance,
            "{key:?}: compiled {a}, trajectory {b}, tolerance {tolerance}"
        );
    }
}
