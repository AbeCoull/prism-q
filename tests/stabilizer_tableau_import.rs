//! Round trips through the stabilizer tableau export and import: a run split
//! at a checkpoint reproduces the outcome sequence of the uninterrupted run.

mod common;

use common::SEED;
use prism_q::backend::Backend;
use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::circuit::Circuit;
use prism_q::circuits;
use prism_q::error::PrismError;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

const N: usize = 500;
const NUM_MEASURE: usize = 200;

/// A random Clifford circuit with measurements spread through its second half,
/// returned whole and as the two halves.
fn split_circuit(n: usize) -> (Circuit, Circuit, Circuit) {
    let gates = circuits::clifford_random_pairs(n, 6, SEED).instructions;
    let mid = gates.len() / 2;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut first = Circuit::new(n, 0);
    first.instructions = gates[..mid].to_vec();

    let mut second = Circuit::new(n, NUM_MEASURE);
    let per = (gates.len() - mid) / NUM_MEASURE;
    let mut bit = 0;
    for (i, inst) in gates[mid..].iter().enumerate() {
        second.instructions.push(inst.clone());
        if i % per == per - 1 && bit < NUM_MEASURE {
            second.add_measure(rng.random_range(0..n), bit);
            bit += 1;
        }
    }
    assert_eq!(bit, NUM_MEASURE);

    let mut full = Circuit::new(n, NUM_MEASURE);
    full.instructions = first
        .instructions
        .iter()
        .chain(&second.instructions)
        .cloned()
        .collect();
    (full, first, second)
}

fn outcomes<B: Backend>(backend: &mut B, circuit: &Circuit) -> Vec<bool> {
    backend
        .init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    backend.apply_instructions(&circuit.instructions).unwrap();
    backend.classical_results().to_vec()
}

fn assert_mixed(outcomes: &[bool]) {
    assert!(outcomes.iter().any(|&b| b) && outcomes.iter().any(|&b| !b));
}

#[test]
fn stabilizer_import_resumes_outcome_sequence_500q() {
    let (full, first, second) = split_circuit(N);
    let expected = outcomes(&mut StabilizerBackend::new(SEED), &full);
    assert_mixed(&expected);

    let mut prep = StabilizerBackend::new(SEED);
    outcomes(&mut prep, &first);
    let (words, phases) = prep.export_tableau().unwrap();

    let mut resumed = StabilizerBackend::new(SEED);
    resumed
        .init_from_tableau(N, words, phases, NUM_MEASURE)
        .unwrap();
    resumed.apply_instructions(&second.instructions).unwrap();
    assert_eq!(resumed.classical_results(), expected.as_slice());
}

#[test]
fn lazy_stabilizer_export_resumes_on_eager_and_lazy_importers() {
    let n = 300;
    let (full, first, second) = split_circuit(n);
    let expected = outcomes(&mut StabilizerBackend::new(SEED), &full);

    let mut prep = StabilizerBackend::new_lazy(SEED);
    outcomes(&mut prep, &first);
    let (words, phases) = prep.export_tableau().unwrap();

    for mut resumed in [
        StabilizerBackend::new(SEED),
        StabilizerBackend::new_lazy(SEED),
    ] {
        resumed
            .init_from_tableau(n, words.clone(), phases.clone(), NUM_MEASURE)
            .unwrap();
        resumed.apply_instructions(&second.instructions).unwrap();
        assert_eq!(resumed.classical_results(), expected.as_slice());
    }
}

#[test]
fn factored_import_resumes_outcome_sequence_500q() {
    let (full, first, second) = split_circuit(N);
    let expected = outcomes(&mut FactoredStabilizerBackend::new(SEED), &full);
    assert_mixed(&expected);

    let mut prep = FactoredStabilizerBackend::new(SEED);
    outcomes(&mut prep, &first);
    let (words, phases) = prep.export_tableau();

    let mut resumed = FactoredStabilizerBackend::new(SEED);
    resumed
        .init_from_tableau(N, words, phases, NUM_MEASURE)
        .unwrap();
    resumed.apply_instructions(&second.instructions).unwrap();
    assert_eq!(resumed.classical_results(), expected.as_slice());
}

#[test]
fn tableau_crosses_between_stabilizer_and_factored() {
    let (full, first, second) = split_circuit(N);
    let expected = outcomes(&mut StabilizerBackend::new(SEED), &full);

    let mut dense = StabilizerBackend::new(SEED);
    outcomes(&mut dense, &first);
    let (words, phases) = dense.export_tableau().unwrap();
    let mut factored = FactoredStabilizerBackend::new(SEED);
    factored
        .init_from_tableau(N, words, phases, NUM_MEASURE)
        .unwrap();
    factored.apply_instructions(&second.instructions).unwrap();
    assert_eq!(factored.classical_results(), expected.as_slice());

    let mut factored = FactoredStabilizerBackend::new(SEED);
    outcomes(&mut factored, &first);
    let (words, phases) = factored.export_tableau();
    let mut dense = StabilizerBackend::new(SEED);
    dense
        .init_from_tableau(N, words, phases, NUM_MEASURE)
        .unwrap();
    dense.apply_instructions(&second.instructions).unwrap();
    assert_eq!(dense.classical_results(), expected.as_slice());
}

#[test]
fn import_then_export_returns_the_rows_with_a_cleared_scratch_row() {
    let (full, _, _) = split_circuit(70);
    let mut prep = StabilizerBackend::new(SEED);
    outcomes(&mut prep, &full);
    let (words, phases) = prep.export_tableau().unwrap();

    let mut resumed = StabilizerBackend::new(SEED);
    resumed
        .init_from_tableau(70, words.clone(), phases.clone(), 0)
        .unwrap();
    let (back_words, back_phases) = resumed.export_tableau().unwrap();

    let stride = 2 * 70usize.div_ceil(64);
    let scratch = 2 * 70 * stride;
    assert_eq!(&back_words[..scratch], &words[..scratch]);
    assert!(back_words[scratch..].iter().all(|&w| w == 0));
    assert_eq!(&back_phases[..2 * 70], &phases[..2 * 70]);
    assert!(!back_phases[2 * 70]);
}

fn identity_rows(n: usize) -> (Vec<u64>, Vec<bool>) {
    let mut backend = StabilizerBackend::new(SEED);
    backend.init(n, 0).unwrap();
    backend.export_tableau().unwrap()
}

fn invalid_message(err: PrismError) -> String {
    match err {
        PrismError::InvalidParameter { message } => message,
        other => panic!("expected InvalidParameter, got {other:?}"),
    }
}

#[test]
fn import_rejects_wrong_word_count() {
    let (words, phases) = identity_rows(3);
    let mut backend = StabilizerBackend::new(SEED);
    let err = backend
        .init_from_tableau(3, words[..words.len() - 1].to_vec(), phases, 0)
        .unwrap_err();
    assert!(invalid_message(err).contains("words"));
}

#[test]
fn import_rejects_wrong_phase_count() {
    let (words, mut phases) = identity_rows(3);
    phases.push(false);
    let mut backend = StabilizerBackend::new(SEED);
    let err = backend.init_from_tableau(3, words, phases, 0).unwrap_err();
    assert!(invalid_message(err).contains("phases"));
}

#[test]
fn import_rejects_commuting_diagonal_pair() {
    let (mut words, phases) = identity_rows(3);
    // Destabilizer 0 becomes Z0, which commutes with stabilizer 0 (also Z0).
    words[0] = 0;
    words[1] = 1;
    let mut backend = StabilizerBackend::new(SEED);
    let err = backend
        .init_from_tableau(3, words.clone(), phases.clone(), 0)
        .unwrap_err();
    assert!(invalid_message(err).contains("row 0"));

    let mut factored = FactoredStabilizerBackend::new(SEED);
    let err = factored.init_from_tableau(3, words, phases, 0).unwrap_err();
    assert!(invalid_message(err).contains("row 0"));
}

#[test]
fn factored_import_rejects_wrong_lengths() {
    let (words, phases) = identity_rows(3);
    let mut factored = FactoredStabilizerBackend::new(SEED);
    let err = factored.init_from_tableau(4, words, phases, 0).unwrap_err();
    assert!(invalid_message(err).contains("words"));
}
