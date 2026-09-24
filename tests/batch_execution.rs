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

// A 17-qubit circuit keeps the whole batch on one thread, so both paths are covered.
#[test]
fn a_batch_past_the_split_width_matches_running_each_alone() {
    let circuits = vec![unitary(6, 2, 0), unitary(17, 2, 1), unitary(6, 2, 2)];
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

#[test]
fn a_batch_reports_the_first_failure_in_order() {
    let mut clifford = Circuit::new(3, 0);
    clifford.add_gate(Gate::H, &[0]);
    clifford.add_gate(Gate::Cx, &[0, 1]);
    let mut first_bad = Circuit::new(2, 0);
    first_bad.add_gate(Gate::T, &[1]);
    let mut second_bad = Circuit::new(5, 0);
    second_bad.add_gate(Gate::Rx(0.3), &[4]);
    let alone = run_batch(
        std::slice::from_ref(&first_bad),
        BackendKind::Stabilizer,
        SEED,
    )
    .unwrap_err()
    .to_string();
    let mut circuits = vec![clifford; 8];
    circuits.push(first_bad);
    circuits.extend(std::iter::repeat_n(second_bad, 8));
    let err = run_batch(&circuits, BackendKind::Stabilizer, SEED).unwrap_err();
    assert_eq!(err.to_string(), alone);
}

#[cfg(feature = "parallel")]
fn on_four_workers<T: Send>(op: impl FnOnce() -> T + Send) -> T {
    prism_q::ThreadPool::with_threads(4).unwrap().install(op)
}

#[cfg(not(feature = "parallel"))]
fn on_four_workers<T>(op: impl FnOnce() -> T) -> T {
    op()
}

fn assert_bitwise(batch: &prism_q::RunOutcome, solo: &prism_q::RunOutcome, what: &str) {
    assert_eq!(batch.classical_bits, solo.classical_bits, "{what}: bits");
    assert_eq!(
        batch.metadata.backend, solo.metadata.backend,
        "{what}: backend"
    );
    assert_eq!(
        batch.probabilities.as_ref().map(|p| p.to_vec()),
        solo.probabilities.as_ref().map(|p| p.to_vec()),
        "{what}: probabilities"
    );
}

// Each worker holds a backend across the circuits its split covers, so which
// widths a held backend meets depends on how the batch splits. Twenty-four
// circuits on four workers exercise it; the answers must not move by one bit.
#[test]
fn a_batch_longer_than_the_pool_matches_a_sequential_loop_bitwise() {
    let circuits: Vec<Circuit> = (0..24)
        .map(|i| match i % 4 {
            0 => unitary(4 + i % 7, 2, i),
            1 => measured(3 + i % 5, i),
            2 => unitary(9, 3, i),
            _ => unitary(6, 1, i),
        })
        .collect();
    for kind in [BackendKind::Auto, BackendKind::Statevector] {
        let batch = on_four_workers(|| run_batch(&circuits, kind.clone(), SEED)).unwrap();
        for (i, circuit) in circuits.iter().enumerate() {
            let solo = simulate(circuit)
                .backend(kind.clone())
                .seed(SEED)
                .run()
                .unwrap();
            assert_bitwise(&batch[i], &solo, &format!("{kind:?} circuit {i}"));
        }
    }
}

// A circuit an explicit backend rejects must fail with the error `simulate` gives,
// not with the backend's own complaint once the run is under way.
#[test]
fn a_failing_circuit_in_a_split_batch_matches_the_sequential_loop() {
    let mut bad = Circuit::new(4, 0);
    bad.add_gate(Gate::T, &[2]);
    let mut circuits: Vec<Circuit> = (0..12)
        .map(|i| {
            let mut c = Circuit::new(3 + i % 3, 0);
            c.add_gate(Gate::H, &[0]);
            c.add_gate(Gate::Cx, &[0, 1]);
            c
        })
        .collect();
    circuits.insert(7, bad);
    let sequential = circuits
        .iter()
        .map(|c| {
            simulate(c)
                .backend(BackendKind::Stabilizer)
                .seed(SEED)
                .run()
        })
        .collect::<prism_q::Result<Vec<_>>>()
        .unwrap_err()
        .to_string();
    let err = on_four_workers(|| run_batch(&circuits, BackendKind::Stabilizer, SEED)).unwrap_err();
    assert_eq!(err.to_string(), sequential);
}

// From 14 to 16 qubits a batch splits across workers whose kernels parallelize
// too, so the nested joins must leave every result as a solo run gives it.
#[cfg(feature = "parallel")]
#[test]
#[cfg_attr(miri, ignore)]
fn a_split_batch_at_15_and_16_qubits_matches_a_sequential_loop_bitwise() {
    let circuits: Vec<Circuit> = (0..6)
        .map(|i| {
            let mut c =
                prism_q::circuits::hardware_efficient_ansatz(15 + i % 2, 2, SEED + i as u64);
            if i == 3 {
                c.num_classical_bits = c.num_qubits;
                for q in 0..c.num_qubits {
                    c.add_measure(q, q);
                }
            }
            c
        })
        .collect();
    let batch = prism_q::ThreadPool::with_threads(4)
        .unwrap()
        .install(|| run_batch(&circuits, BackendKind::Statevector, SEED))
        .unwrap();
    for (i, circuit) in circuits.iter().enumerate() {
        let solo = simulate(circuit)
            .backend(BackendKind::Statevector)
            .seed(SEED)
            .run()
            .unwrap();
        assert_eq!(batch[i].classical_bits, solo.classical_bits, "circuit {i}");
        assert_eq!(
            batch[i].probabilities.as_ref().map(|p| p.to_vec()),
            solo.probabilities.as_ref().map(|p| p.to_vec()),
            "circuit {i}"
        );
    }
}
