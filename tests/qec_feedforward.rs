//! QEC feed-forward: a record-conditioned correction consuming the
//! guarded-region contract, and the routes that reject it.

use prism_q::{
    Gate, QecBasis, QecOp, QecOptions, QecPauli, QecProgram, QecRecordRef, run_qec_program,
    run_qec_program_reference,
};

fn options(shots: usize) -> QecOptions {
    QecOptions {
        shots,
        seed: 42,
        keep_measurements: true,
        ..QecOptions::default()
    }
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

// The compiled QEC sampler evaluates a static affine map, which a
// record-conditioned branch makes depend on the sample. The rejection is
// explicit, not a silent slow path.
#[test]
fn the_compiled_qec_sampler_rejects_feedforward() {
    let err = run_qec_program(&corrected_program(true)).expect_err("affine sampler");
    let text = format!("{err}");
    assert!(text.contains("FEEDFORWARD"), "{text}");
    assert!(text.contains("run_qec_program_reference"), "{text}");
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

/// A conditional reset in the X basis: the body runs, so q1 lands in |+> and a
/// following X-basis measurement reads 0 every shot.
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
