use super::*;
use crate::qec::QecPauli;
use crate::qec::QecRecordRef;
use crate::qec::noise::compile_qec_noisy_sampler_on_compiled_base;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn records(program: &QecProgram) -> Vec<bool> {
    let packed = forced_tableau_records(program, true).unwrap().unwrap();
    (0..program.num_measurements())
        .map(|record| get_bit(&packed, record))
        .collect()
}

fn forced_zero(program: &QecProgram) -> (Vec<u64>, Vec<u64>) {
    let sampler = compile_qec_noisy_sampler_on_compiled_base(program).unwrap();
    forced_zero_records(sampler.compiled_base().unwrap(), sampler.feedforward())
}

/// A random Clifford program over `num_qubits` qubits: gates, basis measurements,
/// Pauli products, resets of measured qubits (and, with `entangled_resets`, of any
/// qubit), Pauli feed-forward, then Z measurements of every qubit not yet measured. A
/// measured qubit is reset before anything touches it again, as the packed path needs.
fn random_program(seed: u64, num_qubits: usize, entangled_resets: bool) -> QecProgram {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut program = QecProgram::new(num_qubits);
    let mut measured = vec![false; num_qubits];
    let mut entangled = vec![false; num_qubits];
    let one_qubit = [
        Gate::H,
        Gate::S,
        Gate::Sdg,
        Gate::X,
        Gate::Y,
        Gate::Z,
        Gate::SX,
    ];
    let bases = [QecBasis::X, QecBasis::Y, QecBasis::Z];
    for _ in 0..40 {
        let a = rng.random_range(0..num_qubits);
        let b = (a + rng.random_range(1..num_qubits)) % num_qubits;
        match rng.random_range(0..7) {
            0 | 1 if !measured[a] => {
                let gate = one_qubit[rng.random_range(0..one_qubit.len())].clone();
                program.push_gate(gate, &[a]).unwrap();
            }
            2 if !measured[a] && !measured[b] => {
                let gate = if rng.random_bool(0.5) {
                    Gate::Cx
                } else {
                    Gate::Cz
                };
                program.push_gate(gate, &[a, b]).unwrap();
                entangled[a] = true;
                entangled[b] = true;
            }
            3 if !measured[a] => {
                let basis = bases[rng.random_range(0..3)];
                program.measure(basis, a).unwrap();
                measured[a] = true;
                entangled[a] = false;
            }
            4 if entangled_resets || !entangled[a] => {
                program.reset(bases[rng.random_range(0..3)], a).unwrap();
                measured[a] = false;
                entangled[a] = false;
            }
            5 if !measured[a] && !measured[b] => {
                program
                    .measure_pauli_product(&[
                        QecPauli::new(bases[rng.random_range(0..3)], a),
                        QecPauli::new(bases[rng.random_range(0..3)], b),
                    ])
                    .unwrap();
                entangled[a] = true;
                entangled[b] = true;
            }
            6 if program.num_measurements() > 0 && !measured[a] => {
                let lookback = rng.random_range(1..=program.num_measurements().min(3));
                let letter = [Gate::X, Gate::Y, Gate::Z][rng.random_range(0..3)].clone();
                program
                    .feedforward(
                        &[QecRecordRef::lookback(lookback).unwrap()],
                        rng.random_bool(0.5),
                        vec![QecOp::Gate {
                            gate: letter,
                            targets: vec![a],
                        }],
                    )
                    .unwrap();
            }
            _ => {}
        }
    }
    for (qubit, &done) in measured.iter().enumerate() {
        if !done {
            program.measure_z(qubit).unwrap();
        }
    }
    program
}

#[test]
fn random_outcomes_take_zero_and_settle_the_records_after_them() {
    for (text, expected) in [
        ("H 0\nM 0 0", vec![false, false]),
        ("X 0\nH 0\nM 0", vec![false]),
        ("H 0\nCX 0 1\nX 1\nM 0 1", vec![false, true]),
        ("RX 0\nMX 0\nRY 1\nMY 1", vec![false, false]),
        ("H 0\nMPP X0*X1\nM 0", vec![false, false]),
        (
            "MPAD 1 0\nHERALDED_ERASE(0.5) 0\nM 0",
            vec![true, false, false, false],
        ),
    ] {
        let program = QecProgram::from_text(text).unwrap();
        assert_eq!(records(&program), expected, "{text}");
    }
}

#[test]
fn a_reset_collapses_its_entangled_partner_onto_outcome_zero() {
    let program = QecProgram::from_text("H 0\nCX 0 1\nX 1\nR 0\nM 1").unwrap();
    assert!(has_entangled_reset(&program));
    assert_eq!(records(&program), vec![true]);

    let x_basis = QecProgram::from_text("H 0\nCX 0 1\nRX 0\nMX 1").unwrap();
    assert_eq!(records(&x_basis), vec![false]);
}

#[test]
fn only_resets_of_qubits_gated_since_their_measurement_are_entangled() {
    for (text, entangled) in [
        ("H 0\nCX 0 1\nMR 0 1\nM 0 1", false),
        ("R 0 1\nH 0\nS 0\nR 0", false),
        ("H 0\nCX 0 1\nMR 0\nR 1\nM 1", true),
        ("MPP X0*X1\nR 0", true),
    ] {
        let program = QecProgram::from_text(text).unwrap();
        assert_eq!(has_entangled_reset(&program), entangled, "{text}");
    }
}

#[test]
fn forced_zero_matches_the_tableau_run_without_entangled_resets() {
    let mut compared = 0;
    for seed in 0..400 {
        let program = random_program(seed, 2 + (seed as usize % 5), false);
        assert!(!has_entangled_reset(&program), "seed {seed}");
        let (uncorrected, corrected) = forced_zero(&program);
        let packed = forced_tableau_records(&program, true).unwrap().unwrap();
        assert_eq!(corrected, packed, "seed {seed}");
        if !program
            .ops()
            .iter()
            .any(|op| matches!(op, QecOp::Feedforward { .. }))
        {
            let skipped = forced_tableau_records(&program, false).unwrap().unwrap();
            assert_eq!(uncorrected, skipped, "seed {seed}");
        }
        compared += 1;
    }
    assert_eq!(compared, 400);
}

#[test]
fn forced_zero_points_ignore_the_compile_seed() {
    for seed in 0..100 {
        let mut program = random_program(seed, 3, true);
        let first = forced_zero(&program);
        let mut options = program.options();
        options.seed = seed ^ 0x5EED;
        program.set_options(options);
        assert_eq!(forced_zero(&program), first, "seed {seed}");
    }
}

#[test]
fn echelon_reduces_rows_that_share_a_first_record() {
    let rows = vec![vec![0b0111], vec![0b0101], vec![0b1100], vec![0b1000]];
    let basis = echelon_by_first_record(&rows);
    let pivots: Vec<usize> = basis.iter().map(|&(pivot, _)| pivot).collect();
    assert_eq!(pivots, vec![0, 1, 2, 3]);
    for (pivot, row) in &basis {
        assert_eq!(first_record(row), Some(*pivot));
    }
    assert_eq!(echelon_by_first_record(&[vec![0b11], vec![0b11]]).len(), 1);
}

#[test]
fn reference_parities_follow_the_rows() {
    let reference =
        QecReference::from_records(&[0b101], &[vec![0], vec![1], vec![0, 2]], &[vec![2]]);
    assert_eq!(reference.detectors, vec![0b001]);
    assert!(reference.observable(0));
}

#[test]
fn non_clifford_programs_have_no_reference_run() {
    let program = QecProgram::from_text("H 0\nT 0\nM 0\nDETECTOR rec[-1]").unwrap();
    assert!(forced_tableau_records(&program, true).unwrap().is_none());
    assert!(program_reference(&program).unwrap().is_none());
}

fn gf2_rank(mut rows: Vec<Vec<u64>>) -> usize {
    let mut rank = 0;
    let bits = rows.first().map_or(0, |row| row.len() * 64);
    for bit in 0..bits {
        let (word, mask) = (bit / 64, 1u64 << (bit % 64));
        let Some(pivot) = (rank..rows.len()).find(|&r| rows[r][word] & mask != 0) else {
            continue;
        };
        rows.swap(rank, pivot);
        let pivot_row = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && row[word] & mask != 0 {
                xor_words(row, &pivot_row);
            }
        }
        rank += 1;
    }
    rank
}

#[test]
fn frame_records_cover_the_compiled_outcome_space() {
    use crate::qec::noise::compile_qec_noisy_sampler;
    const SHOTS: usize = 4096;
    for seed in 0..400 {
        let program = random_program(seed, 2 + (seed as usize % 5), seed % 2 == 1);
        let m = program.num_measurements();
        let shot_rows = |shots: &PackedShots| -> Vec<Vec<u64>> {
            (0..SHOTS)
                .map(|shot| {
                    let mut row = vec![0u64; m.div_ceil(64).max(1)];
                    for record in 0..m {
                        row[record / 64] |= u64::from(shots.get_bit(shot, record)) << (record % 64);
                    }
                    row
                })
                .collect()
        };
        let differences = |rows: &[Vec<u64>]| -> Vec<Vec<u64>> {
            rows.iter()
                .map(|row| {
                    let mut diff = row.clone();
                    xor_words(&mut diff, &rows[0]);
                    diff
                })
                .collect()
        };
        let mut frame = compile_qec_noisy_sampler(&program).unwrap();
        let mut compiled = compile_qec_noisy_sampler_on_compiled_base(&program).unwrap();
        let frame_rows = shot_rows(&frame.sample_measurements_packed(0, SHOTS, SHOTS).unwrap());
        let compiled_rows = shot_rows(
            &compiled
                .sample_measurements_packed(0, SHOTS, SHOTS)
                .unwrap(),
        );
        let frame_span = differences(&frame_rows);
        let compiled_span = differences(&compiled_rows);
        let mut joint = frame_span.clone();
        joint.extend(compiled_span.iter().cloned());
        let mut offset = frame_rows[0].clone();
        xor_words(&mut offset, &compiled_rows[0]);
        joint.push(offset);
        let rank = gf2_rank(compiled_span);
        assert_eq!(gf2_rank(frame_span), rank, "seed {seed}");
        assert_eq!(gf2_rank(joint), rank, "seed {seed}");
    }
}
