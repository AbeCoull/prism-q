//! Batched multi-circuit execution: same answers as a loop, one backend fewer.

use prism_q::sim::run_batch;
use prism_q::{BackendKind, Circuit, Gate, simulate};

const SEED: u64 = 42;
const EPS: f64 = 1e-12;

fn unitary(n: usize, layers: usize, seed: usize) -> Circuit {
    let mut c = Circuit::new(n, n);
    for d in 0..layers {
        for q in 0..n {
            c.add_gate(Gate::Rx(0.11 * (seed + d * n + q + 1) as f64), &[q]);
        }
        for q in 0..n.saturating_sub(1) {
            c.add_gate(Gate::Cx, &[q, q + 1]);
        }
    }
    c
}

fn measured(n: usize, seed: usize) -> Circuit {
    let mut c = unitary(n, 2, seed);
    for q in 0..n {
        c.add_measure(q, q);
    }
    c
}

fn assert_same(batch: &prism_q::RunOutcome, solo: &prism_q::RunOutcome, what: &str) {
    assert_eq!(batch.classical_bits, solo.classical_bits, "{what}: bits");
    assert_eq!(
        batch.metadata.backend, solo.metadata.backend,
        "{what}: backend"
    );
    match (&batch.probabilities, &solo.probabilities) {
        (Some(a), Some(b)) => {
            let (a, b) = (a.to_vec(), b.to_vec());
            assert_eq!(a.len(), b.len(), "{what}: length");
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!((x - y).abs() < EPS, "{what}: state {i}: {x} vs {y}");
            }
        }
        (None, None) => {}
        (a, b) => panic!("{what}: one side has a distribution and the other does not: {a:?} {b:?}"),
    }
}

#[test]
fn a_batch_of_unitary_circuits_matches_running_each_alone() {
    let circuits: Vec<Circuit> = (0..12).map(|i| unitary(6, 3, i)).collect();
    let batch = run_batch(&circuits, BackendKind::Statevector, SEED).unwrap();
    assert_eq!(batch.len(), circuits.len());
    for (i, circuit) in circuits.iter().enumerate() {
        let solo = simulate(circuit)
            .backend(BackendKind::Statevector)
            .seed(SEED)
            .run()
            .unwrap();
        assert_same(&batch[i], &solo, &format!("circuit {i}"));
    }
}

// A held backend would carry its RNG into the next circuit, so a measured
// circuit has to get its own. This is the case that would silently diverge.
#[test]
fn a_batch_of_measured_circuits_matches_running_each_alone() {
    let circuits: Vec<Circuit> = (0..12).map(|i| measured(5, i)).collect();
    let batch = run_batch(&circuits, BackendKind::Statevector, SEED).unwrap();
    for (i, circuit) in circuits.iter().enumerate() {
        let solo = simulate(circuit)
            .backend(BackendKind::Statevector)
            .seed(SEED)
            .run()
            .unwrap();
        assert_same(&batch[i], &solo, &format!("circuit {i}"));
    }
}

#[test]
fn a_batch_mixing_widths_and_measurement_matches_running_each_alone() {
    let circuits = vec![
        unitary(4, 2, 0),
        unitary(7, 2, 1),
        measured(4, 2),
        unitary(4, 3, 3),
        unitary(7, 1, 4),
        measured(7, 5),
    ];
    let batch = run_batch(&circuits, BackendKind::Auto, SEED).unwrap();
    for (i, circuit) in circuits.iter().enumerate() {
        let solo = simulate(circuit).seed(SEED).run().unwrap();
        assert_same(&batch[i], &solo, &format!("circuit {i}"));
    }
}

#[test]
fn a_batch_routed_to_the_stabilizer_matches_running_each_alone() {
    let circuits: Vec<Circuit> = (0..6)
        .map(|i| {
            let mut c = Circuit::new(5, 5);
            c.add_gate(Gate::H, &[i % 5]);
            for q in 0..4 {
                c.add_gate(Gate::Cx, &[q, q + 1]);
            }
            c
        })
        .collect();
    let batch = run_batch(&circuits, BackendKind::Auto, SEED).unwrap();
    for (i, circuit) in circuits.iter().enumerate() {
        let solo = simulate(circuit).seed(SEED).run().unwrap();
        assert_same(&batch[i], &solo, &format!("circuit {i}"));
    }
}

#[test]
fn an_empty_batch_returns_no_results() {
    assert!(run_batch(&[], BackendKind::Auto, SEED).unwrap().is_empty());
}

#[test]
fn a_failing_circuit_ends_the_batch() {
    let mut bad = Circuit::new(3, 0);
    bad.add_gate(Gate::T, &[0]);
    let circuits = vec![unitary(3, 1, 0), bad];
    assert!(run_batch(&circuits, BackendKind::Stabilizer, SEED).is_err());
}
