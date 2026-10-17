//! QEC feed-forward: a record-conditioned correction consuming the
//! guarded-region contract, Pauli bodies on the compiled record path, and the
//! routes that reject the rest.

use crate::qec_common;

use prism_q::{
    Gate, PackedShots, QecBasis, QecNoise, QecOp, QecOptions, QecPauli, QecProgram, QecRecordRef,
    QecSampleResult, run_qec_program, run_qec_program_reference,
};

// An unset `chunk_size` resolves to the shot count, so passing `shots` keeps
// these cases on the single batch they have always run as.
fn options(shots: usize) -> QecOptions {
    qec_common::qec_options(shots, shots, true)
}

/// Teleportation-style correction: measure a qubit prepared in |1>, then flip a
/// second qubit back to |0> when the record reads 1.
fn corrected_program(expected: bool) -> QecProgram {
    let mut program = QecProgram::with_options(2, options(16));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.push_gate(Gate::X, &[1]).unwrap();
    program.measure_z(0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            expected,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    program.measure_z(1).unwrap();
    program
}

#[test]
fn feedforward_fires_on_a_matching_record() {
    let result = run_qec_program_reference(&corrected_program(true)).unwrap();
    let measurements = &result.measurements;
    for shot in 0..16 {
        assert!(
            measurements.get_bit(shot, 0),
            "record 0 is the |1> preparation"
        );
        assert!(
            !measurements.get_bit(shot, 1),
            "the correction returned q1 to 0"
        );
    }
}

#[test]
fn feedforward_skips_on_a_mismatching_record() {
    let result = run_qec_program_reference(&corrected_program(false)).unwrap();
    let measurements = &result.measurements;
    for shot in 0..16 {
        assert!(measurements.get_bit(shot, 1), "the correction did not run");
    }
}

// The predicate is a parity over records, which is the shape a detector has and
// what a register comparison cannot express.
#[test]
fn feedforward_reads_the_parity_of_several_records() {
    let mut program = QecProgram::with_options(3, options(8));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.measure_z(0).unwrap();
    program.measure_z(1).unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(0), QecRecordRef::absolute(1)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![2],
            }],
        )
        .unwrap();
    program.measure_z(2).unwrap();
    let result = run_qec_program_reference(&program).unwrap();
    let measurements = &result.measurements;
    // Records read 1 and 0, so the parity is 1 and the correction fires.
    for shot in 0..8 {
        assert!(measurements.get_bit(shot, 2));
    }
}

// The record space and the classical bit vector are one address space: the
// runner writes record `i` to classical bit `i`, so an absolute reference and a
// lookback that resolve to the same index select the same bit.
#[test]
fn absolute_and_lookback_references_agree() {
    let mut absolute = corrected_program(true);
    absolute.set_options(options(4));
    let mut lookback = QecProgram::with_options(2, options(4));
    lookback.push_gate(Gate::X, &[0]).unwrap();
    lookback.push_gate(Gate::X, &[1]).unwrap();
    lookback.measure_z(0).unwrap();
    lookback
        .feedforward(
            &[QecRecordRef::absolute(0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    lookback.measure_z(1).unwrap();

    let a = run_qec_program_reference(&absolute).unwrap();
    let b = run_qec_program_reference(&lookback).unwrap();
    for shot in 0..4 {
        for record in 0..2 {
            assert_eq!(
                a.measurements.get_bit(shot, record),
                b.measurements.get_bit(shot, record),
                "shot {shot} record {record}"
            );
        }
    }
}

#[test]
fn feedforward_body_rejects_a_record_producing_op() {
    let mut program = QecProgram::with_options(2, options(1));
    program.measure_z(0).unwrap();
    let err = program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            true,
            vec![QecOp::Measure {
                basis: QecBasis::Z,
                qubit: 1,
            }],
        )
        .expect_err("a conditional record would move every later record index");
    assert!(
        format!("{err}").contains("gates and resets only"),
        "unexpected error: {err}"
    );
}

#[test]
fn feedforward_predicate_needs_a_record() {
    let mut program = QecProgram::with_options(1, options(1));
    program.measure_z(0).unwrap();
    let err = program
        .feedforward(
            &[],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .expect_err("empty predicate");
    assert!(format!("{err}").contains("at least one record"), "{err}");
}

#[test]
fn feedforward_reference_out_of_bounds_is_rejected() {
    let mut program = QecProgram::with_options(1, options(1));
    let err = program
        .feedforward(
            &[QecRecordRef::absolute(0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .expect_err("no records exist yet");
    assert!(format!("{err}").contains("out of bounds"), "{err}");
}

// Only a Pauli body is a fixed set of record flips. Any other body is rejected
// by name, not sent down a silent slow path.
#[test]
fn the_compiled_qec_sampler_rejects_a_non_pauli_body() {
    let bodies = [
        QecOp::Gate {
            gate: Gate::H,
            targets: vec![1],
        },
        QecOp::Reset {
            basis: QecBasis::Z,
            qubit: 1,
        },
    ];
    for body in bodies {
        let mut program = QecProgram::with_options(2, options(16));
        program.measure_z(0).unwrap();
        program
            .feedforward(&[QecRecordRef::absolute(0)], true, vec![body])
            .unwrap();
        program.measure_z(1).unwrap();
        let err = run_qec_program(&program).expect_err("non-Pauli body");
        let text = format!("{err}");
        assert!(text.contains("FEEDFORWARD"), "{text}");
        assert!(text.contains("run_qec_program_reference"), "{text}");
    }
}

#[test]
fn the_compiled_qec_sampler_rejects_feedforward_with_leakage() {
    let mut program = corrected_program(true);
    program.noise(QecNoise::Leak(0.1), &[1]).unwrap();
    let err = run_qec_program(&program).expect_err("no leakage feed-forward stage");
    assert!(format!("{err}").contains("leakage"), "{err}");
}

#[test]
fn the_compiled_path_reproduces_exact_feedforward_answers() {
    let fired = run_qec_program(&corrected_program(true)).unwrap();
    let skipped = run_qec_program(&corrected_program(false)).unwrap();
    for shot in 0..16 {
        assert!(fired.measurements.get_bit(shot, 0));
        assert!(
            !fired.measurements.get_bit(shot, 1),
            "the correction returned q1 to 0"
        );
        assert!(
            skipped.measurements.get_bit(shot, 1),
            "the correction did not run"
        );
    }

    let mut parity = QecProgram::with_options(3, options(70));
    parity.push_gate(Gate::X, &[0]).unwrap();
    parity.measure_z(0).unwrap();
    parity.measure_z(1).unwrap();
    parity
        .feedforward(
            &[QecRecordRef::absolute(0), QecRecordRef::absolute(1)],
            true,
            vec![
                QecOp::Gate {
                    gate: Gate::Y,
                    targets: vec![2],
                },
                QecOp::Gate {
                    gate: Gate::Id,
                    targets: vec![1],
                },
            ],
        )
        .unwrap();
    parity.measure_z(2).unwrap();
    let result = run_qec_program(&parity).unwrap();
    for shot in 0..70 {
        assert!(result.measurements.get_bit(shot, 2), "shot {shot}");
    }
}

// `MPAD` produces a record, so a lookback across it reaches one record further
// back than a count of measurements alone would place it.
#[test]
fn a_lookback_counts_padded_records() {
    let mut program = QecProgram::with_options(2, options(8));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.measure_z(0).unwrap();
    program.push_op(QecOp::Pad { value: false }).unwrap();
    program
        .feedforward(
            &[QecRecordRef::lookback(2).unwrap()],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    program.measure_z(1).unwrap();
    for result in [
        run_qec_program_reference(&program).unwrap(),
        run_qec_program(&program).unwrap(),
    ] {
        for shot in 0..8 {
            assert!(result.measurements.get_bit(shot, 2), "shot {shot}");
        }
    }
}

#[test]
fn a_compiled_condition_reads_a_padded_record() {
    let mut program = QecProgram::with_options(1, options(8));
    program.push_op(QecOp::Pad { value: true }).unwrap();
    program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .unwrap();
    program.measure_z(0).unwrap();
    let result = run_qec_program(&program).unwrap();
    for shot in 0..8 {
        assert!(result.measurements.get_bit(shot, 1), "shot {shot}");
    }
}

const STAT_SHOTS: usize = 20_000;

fn x_body(qubit: usize) -> Vec<QecOp> {
    vec![QecOp::Gate {
        gate: Gate::X,
        targets: vec![qubit],
    }]
}

/// Teleport the +1 eigenstate of `basis` from q0 to q2 with X and Z corrections, then
/// measure q2 in `basis`, which reads 0 in every noiseless shot.
fn teleportation(basis: QecBasis, noise: Option<f64>, options: QecOptions) -> QecProgram {
    let mut program = QecProgram::with_options(3, options);
    program.reset(basis, 0).unwrap();
    program.push_gate(Gate::H, &[1]).unwrap();
    program.push_gate(Gate::Cx, &[1, 2]).unwrap();
    if let Some(p) = noise {
        program.noise(QecNoise::Depolarize1(p), &[0, 1, 2]).unwrap();
    }
    program.push_gate(Gate::Cx, &[0, 1]).unwrap();
    program.push_gate(Gate::H, &[0]).unwrap();
    let m0 = program.measure_z(0).unwrap();
    let m1 = program.measure_z(1).unwrap();
    program
        .feedforward(&[QecRecordRef::absolute(m1)], true, x_body(2))
        .unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(m0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::Z,
                targets: vec![2],
            }],
        )
        .unwrap();
    if let Some(p) = noise {
        program.noise(QecNoise::Depolarize1(p), &[2]).unwrap();
    }
    program.measure(basis, 2).unwrap();
    program
}

/// Distance-3 repetition memory whose corrections return the checks to the code space
/// each round: X on q0 when `Z0 Z1` reads 1, X on q2 when `Z1 Z2` does. Each correction
/// flips the next round's check, which the next correction then reads.
fn corrected_repetition_memory(rounds: usize, p: f64, options: QecOptions) -> QecProgram {
    let mut program = QecProgram::with_options(3, options);
    for _ in 0..rounds {
        program.noise(QecNoise::Depolarize1(p), &[0, 1, 2]).unwrap();
        let left = program
            .measure_pauli_product(&[QecPauli::z(0), QecPauli::z(1)])
            .unwrap();
        let right = program
            .measure_pauli_product(&[QecPauli::z(1), QecPauli::z(2)])
            .unwrap();
        for record in [left, right] {
            program.detector(&[QecRecordRef::absolute(record)]).unwrap();
        }
        program
            .feedforward(&[QecRecordRef::absolute(left)], true, x_body(0))
            .unwrap();
        program
            .feedforward(&[QecRecordRef::absolute(right)], true, x_body(2))
            .unwrap();
    }
    let readout: Vec<usize> = (0..3).map(|q| program.measure_z(q).unwrap()).collect();
    for pair in readout.windows(2) {
        program
            .detector(&[
                QecRecordRef::absolute(pair[0]),
                QecRecordRef::absolute(pair[1]),
            ])
            .unwrap();
    }
    program
        .observable_include(0, &[QecRecordRef::absolute(readout[0])])
        .unwrap();
    program
}

/// A random record steers a chain: q1 copies r0, q2 takes `Y` when r1 reads 0, and q3
/// flips on the parity of r0 and r2. Noiseless, r1 = r0, r2 = !r0, and r3 = 1.
fn chained_corrections(noise: Option<f64>, options: QecOptions) -> QecProgram {
    let mut program = QecProgram::with_options(4, options);
    program.push_gate(Gate::H, &[0]).unwrap();
    let r0 = program.measure_z(0).unwrap();
    if let Some(p) = noise {
        program.noise(QecNoise::Depolarize1(p), &[1, 2, 3]).unwrap();
    }
    program
        .feedforward(&[QecRecordRef::absolute(r0)], true, x_body(1))
        .unwrap();
    let r1 = program.measure_z(1).unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(r1)],
            false,
            vec![QecOp::Gate {
                gate: Gate::Y,
                targets: vec![2],
            }],
        )
        .unwrap();
    let r2 = program.measure_z(2).unwrap();
    if let Some(p) = noise {
        program.noise(QecNoise::XError(p), &[3]).unwrap();
    }
    program
        .feedforward(
            &[QecRecordRef::absolute(r0), QecRecordRef::absolute(r2)],
            true,
            x_body(3),
        )
        .unwrap();
    let r3 = program.measure_z(3).unwrap();
    program
        .detector(&[QecRecordRef::absolute(r0), QecRecordRef::absolute(r1)])
        .unwrap();
    program
        .detector(&[QecRecordRef::absolute(r2), QecRecordRef::absolute(r3)])
        .unwrap();
    program
}

fn count_ones(shots: &PackedShots, column: usize) -> usize {
    (0..shots.num_shots())
        .filter(|&shot| shots.get_bit(shot, column))
        .count()
}

fn count_parity(shots: &PackedShots, a: usize, b: usize) -> usize {
    (0..shots.num_shots())
        .filter(|&shot| shots.get_bit(shot, a) != shots.get_bit(shot, b))
        .count()
}

/// Two rates of `shots` samples each agree within five standard deviations of their
/// difference under the pooled rate.
#[track_caller]
fn assert_counts_agree(label: &str, compiled: usize, reference: usize, shots: usize) {
    let n = shots as f64;
    let pooled = (compiled + reference) as f64 / (2.0 * n);
    let tolerance = 5.0 * (2.0 * pooled * (1.0 - pooled) / n).sqrt();
    let diff = (compiled as f64 - reference as f64).abs() / n;
    assert!(
        diff <= tolerance,
        "{label}: compiled {compiled}, reference {reference} of {shots}, diff {diff:.4}, \
         tolerance {tolerance:.4}"
    );
}

/// Compare every record marginal, every pairwise record parity, every detector rate,
/// and every logical error count of the two runs.
fn assert_runs_agree(compiled: &QecSampleResult, reference: &QecSampleResult, shots: usize) {
    let records = compiled.measurements.num_measurements();
    assert_eq!(records, reference.measurements.num_measurements());
    for a in 0..records {
        assert_counts_agree(
            &format!("record {a}"),
            count_ones(&compiled.measurements, a),
            count_ones(&reference.measurements, a),
            shots,
        );
        for b in a + 1..records {
            assert_counts_agree(
                &format!("records {a} ^ {b}"),
                count_parity(&compiled.measurements, a, b),
                count_parity(&reference.measurements, a, b),
                shots,
            );
        }
    }
    for detector in 0..compiled.detectors.num_measurements() {
        assert_counts_agree(
            &format!("detector {detector}"),
            count_ones(&compiled.detectors, detector),
            count_ones(&reference.detectors, detector),
            shots,
        );
    }
    for (observable, (&a, &b)) in compiled
        .logical_errors
        .iter()
        .zip(&reference.logical_errors)
        .enumerate()
    {
        assert_counts_agree(
            &format!("observable {observable}"),
            a as usize,
            b as usize,
            shots,
        );
    }
}

/// Compiled options split the run into chunks that are not word multiples, so the
/// corrections run on partial words and on a short final chunk.
fn chunked_options(shots: usize) -> QecOptions {
    qec_common::qec_options(shots, 3_000, true)
}

#[test]
fn noiseless_teleportation_is_exact_on_the_compiled_path() {
    for basis in [QecBasis::X, QecBasis::Y, QecBasis::Z] {
        let program = teleportation(basis, None, chunked_options(4_000));
        let result = run_qec_program(&program).unwrap();
        for shot in 0..4_000 {
            assert!(
                !result.measurements.get_bit(shot, 2),
                "{basis:?} basis, shot {shot}"
            );
        }
        for record in 0..2 {
            let ones = count_ones(&result.measurements, record);
            assert!(
                (1_600..2_400).contains(&ones),
                "{basis:?} record {record} is a fair coin, got {ones}"
            );
        }
    }
}

#[test]
fn noisy_teleportation_matches_the_reference() {
    for basis in [QecBasis::X, QecBasis::Y] {
        let compiled = run_qec_program(&teleportation(
            basis,
            Some(0.05),
            chunked_options(STAT_SHOTS),
        ))
        .unwrap();
        let reference =
            run_qec_program_reference(&teleportation(basis, Some(0.05), options(STAT_SHOTS)))
                .unwrap();
        assert_runs_agree(&compiled, &reference, STAT_SHOTS);
    }
}

#[test]
fn a_corrected_repetition_memory_matches_the_reference() {
    let compiled = run_qec_program(&corrected_repetition_memory(
        3,
        0.05,
        chunked_options(STAT_SHOTS),
    ))
    .unwrap();
    let reference =
        run_qec_program_reference(&corrected_repetition_memory(3, 0.05, options(STAT_SHOTS)))
            .unwrap();
    assert_runs_agree(&compiled, &reference, STAT_SHOTS);
    assert!(compiled.logical_errors[0] > 0);
}

// The parity route samples detectors without records, which a correction needs, so
// dropping the records must leave detectors and logical counts untouched.
#[test]
fn dropping_records_keeps_corrected_detectors() {
    let kept = run_qec_program(&corrected_repetition_memory(
        3,
        0.05,
        chunked_options(5_000),
    ))
    .unwrap();
    let dropped = run_qec_program(&corrected_repetition_memory(
        3,
        0.05,
        qec_common::qec_options(5_000, 3_000, false),
    ))
    .unwrap();
    for detector in 0..kept.detectors.num_measurements() {
        for shot in 0..5_000 {
            assert_eq!(
                kept.detectors.get_bit(shot, detector),
                dropped.detectors.get_bit(shot, detector),
                "detector {detector}, shot {shot}"
            );
        }
    }
    assert_eq!(kept.logical_errors, dropped.logical_errors);
}

#[test]
fn noiseless_chained_corrections_hold_per_shot() {
    let result = run_qec_program(&chained_corrections(None, chunked_options(4_000))).unwrap();
    let records = &result.measurements;
    for shot in 0..4_000 {
        let r0 = records.get_bit(shot, 0);
        assert_eq!(records.get_bit(shot, 1), r0, "shot {shot}");
        assert_eq!(records.get_bit(shot, 2), !r0, "shot {shot}");
        assert!(records.get_bit(shot, 3), "shot {shot}");
    }
    let ones = count_ones(records, 0);
    assert!(
        (1_600..2_400).contains(&ones),
        "r0 is a fair coin, got {ones}"
    );
}

#[test]
fn noisy_chained_corrections_match_the_reference() {
    let compiled =
        run_qec_program(&chained_corrections(Some(0.1), chunked_options(STAT_SHOTS))).unwrap();
    let reference =
        run_qec_program_reference(&chained_corrections(Some(0.1), options(STAT_SHOTS))).unwrap();
    assert_runs_agree(&compiled, &reference, STAT_SHOTS);
}

#[test]
fn the_detector_error_model_rejects_feedforward() {
    let mut program = corrected_program(true);
    program
        .detector(&[QecRecordRef::lookback(1).unwrap()])
        .unwrap();
    let err = program
        .detector_error_model()
        .expect_err("a conditional correction is not a linear fault propagation");
    assert!(format!("{err}").contains("FEEDFORWARD"), "{err}");
}

#[test]
fn feedforward_after_an_expectation_value_breaks_terminality() {
    let mut program = QecProgram::with_options(2, options(1));
    program.measure_z(0).unwrap();
    program.expectation_value(&[QecPauli::z(1)], 1.0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![1],
            }],
        )
        .unwrap();
    let err = run_qec_program_reference(&program).expect_err("EXP_VAL must stay terminal");
    assert!(format!("{err}").contains("FEEDFORWARD"), "{err}");
}

// A conditional reset in the X basis: the body runs, so q1 lands in |+> and a
// following X-basis measurement reads 0 every shot.
#[test]
fn feedforward_body_carries_a_basis_reset() {
    let mut program = QecProgram::with_options(2, options(16));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.push_gate(Gate::X, &[1]).unwrap();
    program.measure_z(0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            true,
            vec![QecOp::Reset {
                basis: QecBasis::X,
                qubit: 1,
            }],
        )
        .unwrap();
    program.measure_x(1).unwrap();
    let result = run_qec_program_reference(&program).unwrap();
    for shot in 0..16 {
        assert!(
            !result.measurements.get_bit(shot, 1),
            "the reset left q1 in the +1 X eigenstate"
        );
    }
}

// A feed-forward emits no record, so detector rows keep indexing the records
// the static walk assigned them.
#[test]
fn detectors_stay_correct_across_a_feedforward() {
    let mut program = QecProgram::with_options(2, options(32));
    program.push_gate(Gate::X, &[0]).unwrap();
    program.measure_z(0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::lookback(1).unwrap()],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .unwrap();
    program.measure_z(0).unwrap();
    program
        .detector(&[
            QecRecordRef::lookback(1).unwrap(),
            QecRecordRef::lookback(2).unwrap(),
        ])
        .unwrap();
    assert_eq!(program.num_measurements(), 2);
    let result = run_qec_program_reference(&program).unwrap();
    let detectors = &result.detectors;
    for shot in 0..32 {
        assert!(
            detectors.get_bit(shot, 0),
            "record 0 reads 1 and the correction makes record 1 read 0"
        );
    }
}

// The address-space claim, checked against the other executor: the same
// predicate written directly as a circuit guard produces the same outcomes.
#[test]
fn the_equivalent_circuit_guard_agrees() {
    use prism_q::circuit::{Circuit, ClassicalCondition, guarded};

    let program = corrected_program(true);
    let qec = run_qec_program_reference(&program).unwrap();

    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(Gate::X, &[0]);
    circuit.add_gate(Gate::X, &[1]);
    circuit.add_measure(0, 0);
    circuit.instructions.extend(guarded(
        ClassicalCondition::Parity {
            bits: vec![0].into(),
            expected: true,
        },
        vec![prism_q::Instruction::Gate {
            gate: Gate::X,
            targets: prism_q::circuit::SmallVec::from_slice(&[1]),
        }],
    ));
    circuit.add_measure(1, 1);
    let outcome = prism_q::simulate(&circuit).seed(42).run().unwrap();

    for record in 0..2 {
        assert_eq!(
            qec.measurements.get_bit(0, record),
            outcome.classical_bits[record],
            "record {record}"
        );
    }
}

#[test]
fn feedforward_body_must_not_be_empty() {
    let mut program = QecProgram::with_options(1, options(1));
    program.measure_z(0).unwrap();
    let err = program
        .feedforward(&[QecRecordRef::absolute(0)], true, Vec::new())
        .expect_err("empty body");
    assert!(format!("{err}").contains("at least one operation"), "{err}");
}

// No top-level gate, so the row compiler reaches the feed-forward rather than
// stopping on the gate rejection two arms above it.
#[test]
fn the_row_compiler_rejects_feedforward() {
    let mut program = QecProgram::with_options(1, options(1));
    program.measure_z(0).unwrap();
    program
        .feedforward(
            &[QecRecordRef::absolute(0)],
            true,
            vec![QecOp::Gate {
                gate: Gate::X,
                targets: vec![0],
            }],
        )
        .unwrap();
    let err = prism_q::compile_qec_program_rows(&program).expect_err("no row representation");
    assert!(format!("{err}").contains("FEEDFORWARD"), "{err}");
}
