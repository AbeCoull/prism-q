//! Native shot sampling on the backends that hold a polynomial-size state.
//!
//! Sparse, Factored, and MPS draw measurement outcomes from their own
//! representation instead of a dense `2^n` probability vector.
//!
//! Most cases run below the dense cap, where the statevector distribution is
//! available as the reference. The oversize section at the bottom runs at 40
//! qubits, where it is not: those cases use GHZ and Bell-pair states, whose
//! support is known in closed form, and they open by pinning that the dense
//! route rejects the same circuit. No environment override is involved; a `2^40`
//! amplitude vector is eight terabytes.
//!
//! Three properties are checked per backend, and only the second is
//! statistical:
//!
//! 1. Support. Every sampled bitstring has non-zero reference probability. A
//!    sampler that walks the wrong conditional, or reads the wrong site for a
//!    permuted MPS layout, emits impossible outcomes and fails here exactly.
//! 2. Frequency. Per-outcome frequencies sit inside a four-sigma binomial band.
//! 3. Determinism. One seed replays bit for bit, across separate runs.

mod common;

use std::collections::HashMap;

use common::{SEED, sv_reference_probs};
use prism_q::circuit::Circuit;
use prism_q::gates::Gate;
use prism_q::sim::{BackendKind, ShotsResult, simulate};

/// Shots per sampling case. Sets the width of the frequency band below.
const SHOTS: usize = 20_000;

/// Reference probability under which an outcome counts as unreachable. Well
/// above the amplitude noise floor of every backend here and far below the
/// smallest weight any of these circuits puts on a reachable outcome.
const SUPPORT_FLOOR: f64 = 1e-12;

/// Frequency band: four standard errors of a binomial at [`SHOTS`] draws, plus
/// a floor that keeps near-deterministic outcomes from demanding an exact hit.
fn frequency_band(p: f64) -> f64 {
    4.0 * (p * (1.0 - p) / SHOTS as f64).sqrt() + 0.002
}

fn measure_all(circuit: &Circuit) -> Circuit {
    let mut measured = Circuit::new(circuit.num_qubits, circuit.num_qubits);
    measured.instructions = circuit.instructions.clone();
    for q in 0..circuit.num_qubits {
        measured.add_measure(q, q);
    }
    measured
}

/// Basis-state index of one shot, reading classical bit `q` as qubit `q`.
fn shot_index(shot: &[bool]) -> usize {
    shot.iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .map(|(i, _)| 1usize << i)
        .sum()
}

fn outcome_frequencies(result: &ShotsResult) -> HashMap<usize, f64> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for shot in &result.shots {
        *counts.entry(shot_index(shot)).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(index, count)| (index, count as f64 / result.shots.len() as f64))
        .collect()
}

/// Run `circuit` with terminal measurements on `kind` and check the sampled
/// distribution against the statevector reference.
fn check_sampling(label: &str, kind: BackendKind, circuit: &Circuit) {
    let reference = sv_reference_probs(circuit);
    let measured = measure_all(circuit);

    let result = simulate(&measured)
        .backend(kind.clone())
        .seed(SEED)
        .shots(SHOTS)
        .unwrap();
    assert_eq!(result.shots.len(), SHOTS, "{label}: wrong shot count");

    let observed = outcome_frequencies(&result);
    for (&index, &frequency) in &observed {
        assert!(
            reference[index] > SUPPORT_FLOOR,
            "{label}: sampled basis state {index} has reference probability {:.3e}, \
             so the sampler drew an outcome the state cannot produce",
            reference[index]
        );
        let band = frequency_band(reference[index]);
        assert!(
            (frequency - reference[index]).abs() < band,
            "{label}: outcome {index} frequency {frequency:.6} vs reference {:.6} \
             (band {band:.6} at {SHOTS} shots)",
            reference[index]
        );
    }

    for (index, &p) in reference.iter().enumerate() {
        if p > 4.0 * frequency_band(p) {
            assert!(
                observed.contains_key(&index),
                "{label}: outcome {index} carries probability {p:.6} and was never sampled"
            );
        }
    }

    let replay = simulate(&measured)
        .backend(kind)
        .seed(SEED)
        .shots(SHOTS)
        .unwrap();
    assert_eq!(
        result.shots, replay.shots,
        "{label}: same seed produced different shots"
    );
}

/// Counts must agree with the shot histogram drawn from the same seed. The
/// counts entry point has no shortcut for these backends, so it routes through
/// the same native sampler.
fn check_counts_match_shots(label: &str, kind: BackendKind, circuit: &Circuit) {
    let measured = measure_all(circuit);
    let shots = simulate(&measured)
        .backend(kind.clone())
        .seed(SEED)
        .shots(SHOTS)
        .unwrap();
    let counts = simulate(&measured)
        .backend(kind)
        .seed(SEED)
        .sample_counts(SHOTS)
        .unwrap();
    assert_eq!(
        shots.counts(),
        counts.into_counts(),
        "{label}: counts disagree with the shot histogram at the same seed"
    );
}

fn ghz(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c
}

/// Sparse-friendly: a permutation layer over a small superposition keeps the
/// amplitude map far below `2^n` entries.
fn sparse_friendly(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::H, &[1]);
    c.add_gate(Gate::T, &[0]);
    for q in 0..n - 1 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    c.add_gate(Gate::X, &[n - 1]);
    c
}

// ===== sparse =====

#[test]
fn sparse_samples_ghz_distribution() {
    check_sampling("sparse ghz 10q", BackendKind::Sparse, &ghz(10));
}

#[test]
fn sparse_samples_sparse_friendly_distribution() {
    check_sampling(
        "sparse mixed 10q",
        BackendKind::Sparse,
        &sparse_friendly(10),
    );
}

#[test]
fn sparse_counts_match_shots() {
    check_counts_match_shots("sparse ghz 10q", BackendKind::Sparse, &ghz(10));
}

// ===== factored =====

#[test]
fn factored_samples_independent_blocks() {
    check_sampling(
        "factored blocks 10q",
        BackendKind::Factored,
        &prism_q::circuits::independent_bell_pairs(5),
    );
}

#[test]
fn factored_samples_single_merged_block() {
    check_sampling("factored ghz 10q", BackendKind::Factored, &ghz(10));
}

#[test]
fn factored_counts_match_shots() {
    check_counts_match_shots(
        "factored blocks 10q",
        BackendKind::Factored,
        &prism_q::circuits::independent_bell_pairs(5),
    );
}

// ===== mps =====

const MPS: BackendKind = BackendKind::Mps {
    max_bond_dim: 1 << 8,
};

#[test]
fn mps_samples_ghz_distribution() {
    check_sampling("mps ghz 10q", MPS, &ghz(10));
}

#[test]
fn mps_samples_rotation_chain_distribution() {
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.add_gate(Gate::Ry(0.5 + 0.1 * q as f64), &[q]);
    }
    for q in 0..7 {
        c.add_gate(Gate::Cx, &[q, q + 1]);
    }
    check_sampling("mps rotation chain 8q", MPS, &c);
}

/// Long-range gates route through SWAP chains, so the site holding a logical
/// qubit is no longer its index. A sampler that reads the site index straight
/// through returns permuted bitstrings, which the support check rejects.
#[test]
fn mps_samples_swap_routed_layout() {
    let mut c = Circuit::new(8, 0);
    c.add_gate(Gate::H, &[0]);
    c.add_gate(Gate::Cx, &[0, 7]);
    c.add_gate(Gate::Ry(0.7), &[3]);
    c.add_gate(Gate::Cx, &[3, 6]);
    c.add_gate(Gate::T, &[5]);
    c.add_gate(Gate::Cx, &[1, 5]);
    check_sampling("mps swap routed 8q", MPS, &c);
}

#[test]
fn mps_counts_match_shots() {
    check_counts_match_shots("mps ghz 10q", MPS, &ghz(10));
}

// ===== above the dense cap =====

/// Qubit count for the oversize cases. A `2^40` amplitude vector is eight
/// terabytes, so no machine's memory-derived cap admits it and the dense route
/// is unreachable by construction rather than by an environment override.
const OVERSIZE_QUBITS: usize = 40;

fn oversize_ghz() -> Circuit {
    measure_all(&ghz(OVERSIZE_QUBITS))
}

/// Pins the premise the oversize cases rest on: the dense route cannot serve
/// this circuit, so anything that does is not going through it.
#[test]
fn oversize_dense_route_is_unavailable() {
    let err = simulate(&oversize_ghz())
        .backend(BackendKind::Statevector)
        .seed(SEED)
        .shots(8)
        .unwrap_err();
    assert!(
        matches!(
            err,
            prism_q::PrismError::IncompatibleBackend { .. }
                | prism_q::PrismError::BackendUnsupported { .. }
        ),
        "expected the statevector route to reject {OVERSIZE_QUBITS} qubits, got {err:?}"
    );
}

/// GHZ has exactly two outcomes, all zeros and all ones, so an oversize sample
/// is checkable without any reference vector.
fn assert_oversize_ghz_shots(label: &str, kind: BackendKind) {
    let circuit = oversize_ghz();
    let shots = 512;

    let result = simulate(&circuit)
        .backend(kind.clone())
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.shots.len(), shots, "{label}: wrong shot count");

    let mut ones = 0usize;
    for shot in &result.shots {
        let set = shot.iter().filter(|&&b| b).count();
        assert!(
            set == 0 || set == OVERSIZE_QUBITS,
            "{label}: GHZ shot has {set} of {OVERSIZE_QUBITS} bits set, so the chain broke"
        );
        if set == OVERSIZE_QUBITS {
            ones += 1;
        }
    }
    let fraction = ones as f64 / shots as f64;
    assert!(
        (fraction - 0.5).abs() < frequency_band(0.5) * (SHOTS as f64 / shots as f64).sqrt(),
        "{label}: all-ones fraction {fraction:.4} is not a fair coin"
    );

    let counts = simulate(&circuit)
        .backend(kind.clone())
        .seed(SEED)
        .sample_counts(shots)
        .unwrap();
    assert_eq!(
        counts.into_counts().len(),
        2,
        "{label}: GHZ counts must have exactly two outcomes"
    );

    let replay = simulate(&circuit)
        .backend(kind)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(
        result.shots, replay.shots,
        "{label}: same seed produced different shots"
    );
}

#[test]
fn mps_samples_above_the_dense_cap() {
    assert_oversize_ghz_shots(
        "mps 40q",
        BackendKind::Mps {
            max_bond_dim: 1 << 4,
        },
    );
}

#[test]
fn sparse_samples_above_the_dense_cap() {
    assert_oversize_ghz_shots("sparse 40q", BackendKind::Sparse);
}

#[test]
fn factored_samples_above_the_dense_cap() {
    let mut c = Circuit::new(OVERSIZE_QUBITS, OVERSIZE_QUBITS);
    for pair in 0..OVERSIZE_QUBITS / 2 {
        c.add_gate(Gate::H, &[2 * pair]);
        c.add_gate(Gate::Cx, &[2 * pair, 2 * pair + 1]);
    }
    for q in 0..OVERSIZE_QUBITS {
        c.add_measure(q, q);
    }

    let shots = 256;
    let result = simulate(&c)
        .backend(BackendKind::Factored)
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.shots.len(), shots);
    for shot in &result.shots {
        for pair in 0..OVERSIZE_QUBITS / 2 {
            assert_eq!(
                shot[2 * pair],
                shot[2 * pair + 1],
                "factored 40q: bell pair {pair} came back uncorrelated"
            );
        }
    }
}

/// The headline of the expectation-value half: exact observables on a state the
/// dense route cannot hold. GHZ gives `<Z_0 Z_k> = 1` for every `k` and
/// `<Z_0> = 0`.
#[test]
fn mps_expectation_values_above_the_dense_cap() {
    let unitary = ghz(OVERSIZE_QUBITS);
    let values = simulate(&unitary)
        .backend(BackendKind::Mps {
            max_bond_dim: 1 << 4,
        })
        .seed(SEED)
        .expectation_values(&[
            vec![
                prism_q::PauliTerm::z(0),
                prism_q::PauliTerm::z(OVERSIZE_QUBITS - 1),
            ],
            vec![prism_q::PauliTerm::z(0)],
            vec![prism_q::PauliTerm::x(0)],
            vec![],
        ])
        .unwrap();

    let want = [1.0, 0.0, 0.0, 1.0];
    for (i, (&got, &expected)) in values.iter().zip(&want).enumerate() {
        assert!(
            (got - expected).abs() < 1e-9,
            "observable {i}: got {got}, want {expected}"
        );
    }
}
