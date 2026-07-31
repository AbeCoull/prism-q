//! Native shot sampling on Sparse, Factored, MPS, and ProductState, which draw
//! outcomes from their own representation instead of a dense `2^n` probability
//! vector. Below the dense cap the statevector distribution is the reference;
//! the 40-qubit oversize cases use GHZ and Bell-pair states, whose support is
//! known in closed form, and open by pinning that the dense route rejects the
//! same circuit. The product cases run past 1000 qubits, where no dense vector
//! and no merged block distribution exist at all.

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

/// Counts must agree with the shot histogram at the same seed; the counts
/// entry point routes through the same native sampler.
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

// Long-range gates route through SWAP chains, so the site holding a logical
// qubit is no longer its index. A sampler that reads the site index straight
// through returns permuted bitstrings, which the support check rejects.
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

// ===== product state =====

/// Single-qubit rotations only, so every qubit stays independent. The `Rz`
/// layer detunes the per-qubit weights away from a fair coin, so a sampler
/// that ignored the amplitudes would still fail the frequency band.
fn product_layers(n: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(Gate::Ry(0.3 + 0.17 * q as f64), &[q]);
        c.add_gate(Gate::Rz(0.2 * q as f64), &[q]);
    }
    c
}

#[test]
fn product_samples_rotation_layers_distribution() {
    check_sampling(
        "product 10q",
        BackendKind::ProductState,
        &product_layers(10),
    );
}

// Auto routes a circuit with no entangling gates to the product state, so the
// same sampler has to serve it without the caller naming the backend.
#[test]
fn product_auto_route_samples_the_same_distribution() {
    check_sampling("product auto 10q", BackendKind::Auto, &product_layers(10));
}

#[test]
fn product_counts_match_shots() {
    check_counts_match_shots(
        "product 10q",
        BackendKind::ProductState,
        &product_layers(10),
    );
}

// ===== above the dense cap =====

/// Qubit count for the oversize cases. A `2^40` amplitude vector is eight
/// terabytes, so no machine's memory-derived cap admits it and the dense route
/// is unreachable by construction rather than by an environment override.
const OVERSIZE_QUBITS: usize = 40;

fn oversize_ghz() -> Circuit {
    measure_all(&ghz(OVERSIZE_QUBITS))
}

// Pins the premise the oversize cases rest on: the dense route cannot serve
// this circuit, so anything that does is not going through it.
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

// ===== wide product circuits =====

/// Width for the product cases. Well past 64 qubits, where the merged block
/// distribution the decomposed route builds stops existing, and past any dense
/// vector by a margin no memory cap changes.
const WIDE_QUBITS: usize = 1024;

/// Qubit `q` ends in `|1>`, `|+>`, or `|0>` by `q % 3`, so every bit is either
/// pinned or a fair coin and the shots are checkable without a reference
/// vector. `Rz(0.4)` keeps the circuit non-Clifford, which is what stops the
/// compiled Clifford sampler from answering instead.
fn wide_product_circuit() -> Circuit {
    let mut c = Circuit::new(WIDE_QUBITS, 0);
    for q in 0..WIDE_QUBITS {
        match q % 3 {
            0 => c.add_gate(Gate::X, &[q]),
            1 => c.add_gate(Gate::H, &[q]),
            _ => c.add_gate(Gate::Rz(0.4), &[q]),
        }
    }
    c
}

fn assert_wide_product_shots(label: &str, kind: BackendKind) {
    let circuit = measure_all(&wide_product_circuit());
    let shots = 512;

    let result = simulate(&circuit)
        .backend(kind.clone())
        .seed(SEED)
        .shots(shots)
        .unwrap();
    assert_eq!(result.shots.len(), shots, "{label}: wrong shot count");

    let fair: Vec<usize> = (0..WIDE_QUBITS).filter(|q| q % 3 == 1).collect();
    let mut ones = vec![0usize; WIDE_QUBITS];
    for shot in &result.shots {
        for (q, &bit) in shot.iter().enumerate() {
            match q % 3 {
                0 => assert!(bit, "{label}: qubit {q} is |1> and came back 0"),
                2 => assert!(!bit, "{label}: qubit {q} is |0> and came back 1"),
                _ => ones[q] += usize::from(bit),
            }
        }
        assert!(
            fair.iter().any(|&q| shot[q] != shot[fair[0]]),
            "{label}: every fair qubit agreed in one shot, so they share a draw"
        );
    }

    let band = frequency_band(0.5) * (SHOTS as f64 / shots as f64).sqrt();
    for &q in &fair {
        let fraction = ones[q] as f64 / shots as f64;
        assert!(
            (fraction - 0.5).abs() < band,
            "{label}: qubit {q} is |+> and came back at {fraction:.4}"
        );
    }

    let counts = simulate(&circuit)
        .backend(kind.clone())
        .seed(SEED)
        .sample_counts(shots)
        .unwrap();
    assert_eq!(
        result.counts(),
        counts.into_counts(),
        "{label}: counts disagree with the shot histogram at the same seed"
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

// Pins the premise: nothing else can serve this width, so the shots above came
// from the per-qubit sampler.
#[test]
fn wide_product_dense_route_is_unavailable() {
    let err = simulate(&measure_all(&wide_product_circuit()))
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
        "expected the statevector route to reject {WIDE_QUBITS} qubits, got {err:?}"
    );
}

#[test]
fn product_samples_a_thousand_qubits() {
    assert_wide_product_shots("product 1024q", BackendKind::ProductState);
}

#[test]
fn product_auto_route_samples_a_thousand_qubits() {
    assert_wide_product_shots("product auto 1024q", BackendKind::Auto);
}

// Closed-form observables on a product state no dense route can hold: `|1>`
// gives `<Z> = -1` and `<X> = 0`, `|+>` gives `<X> = 1` and `<Z> = 0`, and
// `Rz(0.4)|0>` is `|0>` up to a phase, so `<Z> = 1` and `<X> = 0`. A joint
// string is the product of its factors.
#[test]
fn product_expectation_values_above_the_dense_cap() {
    let circuit = wide_product_circuit();
    let last_one = WIDE_QUBITS - 1 - (WIDE_QUBITS - 1) % 3;
    let values = simulate(&circuit)
        .backend(BackendKind::ProductState)
        .seed(SEED)
        .expectation_values(&[
            vec![prism_q::PauliTerm::z(0)],
            vec![prism_q::PauliTerm::x(0)],
            vec![prism_q::PauliTerm::x(1)],
            vec![prism_q::PauliTerm::z(1)],
            vec![prism_q::PauliTerm::z(2)],
            vec![prism_q::PauliTerm::z(0), prism_q::PauliTerm::z(last_one)],
            vec![
                prism_q::PauliTerm::z(0),
                prism_q::PauliTerm::x(1),
                prism_q::PauliTerm::z(2),
            ],
            vec![prism_q::PauliTerm::y(1)],
            vec![],
        ])
        .unwrap();

    let want = [-1.0, 0.0, 1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0];
    for (i, (&got, &expected)) in values.iter().zip(&want).enumerate() {
        assert!(
            (got - expected).abs() < 1e-12,
            "observable {i}: got {got}, want {expected}"
        );
    }
}

// Exact observables on a state the dense route cannot hold: GHZ gives
// `<Z_0 Z_k> = 1` for every `k` and `<Z_0> = 0`.
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
