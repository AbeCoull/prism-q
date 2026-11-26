//! Correctness for `StabilizerBackend` against the statevector reference, on
//! Clifford-eligible circuit families only; the `is_clifford` guard in
//! `check` rejects anything outside the stabilizer formalism.

use crate::common;

use common::{SEED, STAB_EPS, assert_backend_matches_sv, is_clifford};
use prism_q::backend::stabilizer::StabilizerBackend;
use prism_q::circuits;

fn check(label: &str, circuit: &prism_q::circuit::Circuit) {
    assert!(
        is_clifford(circuit),
        "{label}: circuit contains non-Clifford gates; \
         this test must only be called on Clifford circuits"
    );
    let mut backend = StabilizerBackend::new(SEED);
    assert_backend_matches_sv(&mut backend, circuit, STAB_EPS, label);
}

// ===== clifford_heavy =====

#[test]
fn stabilizer_clifford_heavy_4q() {
    check(
        "clifford_heavy 4q d10",
        &circuits::clifford_heavy_circuit(4, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_8q() {
    check(
        "clifford_heavy 8q d10",
        &circuits::clifford_heavy_circuit(8, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_12q() {
    check(
        "clifford_heavy 12q d10",
        &circuits::clifford_heavy_circuit(12, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_16q() {
    check(
        "clifford_heavy 16q d10",
        &circuits::clifford_heavy_circuit(16, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_heavy_20q() {
    check(
        "clifford_heavy 20q d10",
        &circuits::clifford_heavy_circuit(20, 10, SEED),
    );
}

// ===== clifford_random_pairs =====

#[test]
fn stabilizer_clifford_random_pairs_8q() {
    check(
        "clifford_random_pairs 8q d10",
        &circuits::clifford_random_pairs(8, 10, SEED),
    );
}

#[test]
fn stabilizer_clifford_random_pairs_16q() {
    check(
        "clifford_random_pairs 16q d10",
        &circuits::clifford_random_pairs(16, 10, SEED),
    );
}

// ===== ghz =====

#[test]
fn stabilizer_ghz_12q() {
    check("ghz 12q", &circuits::ghz_circuit(12));
}

#[test]
fn stabilizer_ghz_20q() {
    check("ghz 20q", &circuits::ghz_circuit(20));
}

// ===== bell_pairs =====

#[test]
fn stabilizer_bell_pairs_4q() {
    check("bell_pairs 4q", &circuits::independent_bell_pairs(2));
}

#[test]
fn stabilizer_bell_pairs_12q() {
    check("bell_pairs 12q", &circuits::independent_bell_pairs(6));
}

#[test]
fn stabilizer_bell_pairs_20q() {
    check("bell_pairs 20q", &circuits::independent_bell_pairs(10));
}

// ===== local_clifford_blocks =====

#[test]
fn stabilizer_local_clifford_blocks_12q() {
    check(
        "local_clifford_blocks 3x4q d10",
        &circuits::local_clifford_blocks(3, 4, 10, SEED),
    );
}

#[test]
fn stabilizer_local_clifford_blocks_16q() {
    check(
        "local_clifford_blocks 4x4q d10",
        &circuits::local_clifford_blocks(4, 4, 10, SEED),
    );
}

#[test]
fn stabilizer_local_clifford_blocks_20q() {
    check(
        "local_clifford_blocks 5x4q d10",
        &circuits::local_clifford_blocks(5, 4, 10, SEED),
    );
}

// ===== applicability gate sanity =====

#[test]
fn is_clifford_rejects_t_gate() {
    let qft = circuits::qft_circuit(4);
    assert!(
        !is_clifford(&qft),
        "QFT should be flagged non-Clifford (uses controlled-phase)"
    );
}

#[test]
fn is_clifford_accepts_clifford_heavy() {
    let c = circuits::clifford_heavy_circuit(4, 10, SEED);
    assert!(is_clifford(&c), "clifford_heavy must be all-Clifford");
}

// ===== wide cross-word batching (n = 320, nw = 5, 4 <= cross < nw) =====

fn wide_cross_word_observables() -> Vec<Vec<prism_q::PauliTerm>> {
    use prism_q::PauliTerm;
    vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::x(1), PauliTerm::z(64 + 7)],
        vec![PauliTerm::y(192), PauliTerm::x(3)],
        vec![PauliTerm::z(128 + 5), PauliTerm::z(4)],
        vec![PauliTerm::x(256 + 20), PauliTerm::y(64 + 13)],
        vec![PauliTerm::z(100), PauliTerm::z(200), PauliTerm::z(310)],
    ]
}

fn build_wide_cross_word_circuit(n: usize, seed: u64, with_measures: bool) -> prism_q::Circuit {
    use prism_q::gates::Gate;
    let mut c = circuits::clifford_random_pairs(n, 12, seed);
    c.num_classical_bits = n;
    if with_measures {
        c.add_measure(0, 0);
    }

    // Mix 1q, same-word 2q, and 4 cross-word gates (4 >= 4 and 4 < nw = 5)
    // with unsorted and shared words (word 0 has qubits 1, 2, 3, 4; word 1
    // has 65, 71, 77; word 4 has 267, 276).
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::S, &[65]);
    c.add_gate(Gate::SX, &[4]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[192, 3]);
    c.add_gate(Gate::Cz, &[64 + 7, 256 + 11]);
    c.add_gate(Gate::Swap, &[128 + 5, 4]);
    c.add_gate(Gate::Cx, &[256 + 20, 64 + 13]);
    c.add_gate(Gate::Y, &[3]);
    c.add_gate(Gate::Sdg, &[192]);

    if with_measures {
        for q in 0..n {
            c.add_measure(q, q);
        }
    }
    c
}

#[test]
fn stabilizer_cross_word_batch_matches_per_instruction_320q() {
    use prism_q::backend::Backend;
    use prism_q::sim;

    let n = 320;
    let obs = wide_cross_word_observables();

    for seed in [SEED, 0xDEAD_BEEF] {
        let circuit_no_tail_meas = build_wide_cross_word_circuit(n, seed, false);
        let mut ref_obs_backend = StabilizerBackend::new(seed);
        ref_obs_backend.init(n, n).unwrap();
        for instr in &circuit_no_tail_meas.instructions {
            ref_obs_backend.apply(instr).unwrap();
        }
        let expected_obs = ref_obs_backend.pauli_expectations(&obs).unwrap();

        for mut batch_obs_backend in [
            StabilizerBackend::new(seed),
            StabilizerBackend::new_lazy(seed),
        ] {
            sim::run_on(&mut batch_obs_backend, &circuit_no_tail_meas).unwrap();
            let actual_obs = batch_obs_backend.pauli_expectations(&obs).unwrap();
            assert_eq!(
                actual_obs, expected_obs,
                "320q cross-word expectations mismatch for seed {seed}"
            );
        }

        let circuit_with_meas = build_wide_cross_word_circuit(n, seed, true);
        let mut ref_meas_backend = StabilizerBackend::new(seed);
        ref_meas_backend.init(n, n).unwrap();
        for instr in &circuit_with_meas.instructions {
            ref_meas_backend.apply(instr).unwrap();
        }
        let expected_bits = ref_meas_backend.classical_results().to_vec();

        for mut batch_meas_backend in [
            StabilizerBackend::new(seed),
            StabilizerBackend::new_lazy(seed),
        ] {
            sim::run_on(&mut batch_meas_backend, &circuit_with_meas).unwrap();
            assert_eq!(
                batch_meas_backend.classical_results(),
                expected_bits.as_slice(),
                "320q cross-word measurement outcomes mismatch for seed {seed}"
            );
        }
    }
}

#[test]
fn factored_stabilizer_cross_word_batch_matches_stabilizer_320q() {
    use prism_q::backend::Backend;
    use prism_q::backend::factored_stabilizer::FactoredStabilizerBackend;
    use prism_q::gates::Gate;
    use prism_q::sim;

    let n = 320;
    let mut c = prism_q::Circuit::new(n, 16);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    for q in (0..n).step_by(3) {
        c.add_gate(Gate::H, &[q]);
    }
    // Single-cluster gate run with 4 cross-word gates (< nw = 5) so
    // SubTableau routes through word batching and pcc_apply_cross_word.
    c.add_gate(Gate::S, &[65]);
    c.add_gate(Gate::Cx, &[1, 2]);
    c.add_gate(Gate::Cx, &[192, 3]);
    c.add_gate(Gate::Cz, &[64 + 7, 256 + 11]);
    c.add_gate(Gate::Swap, &[128 + 5, 4]);
    c.add_gate(Gate::Cx, &[256 + 20, 64 + 13]);
    c.add_gate(Gate::SX, &[4]);

    let obs = wide_cross_word_observables();

    let mut stab = StabilizerBackend::new(SEED);
    stab.init(n, 16).unwrap();
    for instr in &c.instructions {
        stab.apply(instr).unwrap();
    }
    let expected_obs = stab.pauli_expectations(&obs).unwrap();

    let mut factored = FactoredStabilizerBackend::new(SEED);
    sim::run_on(&mut factored, &c).unwrap();
    let actual_obs = factored.pauli_expectations(&obs).unwrap();
    assert_eq!(actual_obs, expected_obs);

    let mut c_meas = c;
    for (bit, q) in [0, 1, 3, 4, 65, 71, 77, 128, 133, 192, 256, 267, 276, 319]
        .into_iter()
        .enumerate()
    {
        c_meas.add_measure(q, bit);
    }

    let mut stab_meas = StabilizerBackend::new(SEED);
    stab_meas.init(n, 16).unwrap();
    for instr in &c_meas.instructions {
        stab_meas.apply(instr).unwrap();
    }
    let mut factored_meas = FactoredStabilizerBackend::new(SEED);
    sim::run_on(&mut factored_meas, &c_meas).unwrap();
    assert_eq!(
        factored_meas.classical_results(),
        stab_meas.classical_results()
    );
}
