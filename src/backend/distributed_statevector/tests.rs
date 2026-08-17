use super::DistributedStatevectorBackend;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Circuit;
use crate::circuit::builder::CircuitBuilder;
use crate::distributed::DistributedContext;
use crate::distributed::loopback::{run_ranks, run_ranks_max_gather};
use crate::sim::run_on;
use crate::sim::unified_pauli::PauliTerm;
use num_complex::Complex64;

const SEED: u64 = 42;
const TOL: f64 = 1e-10;

/// Run `circuit` across simulated ranks with the given backend configuration
/// and return rank 0's probabilities.
fn loopback_probs_with(circuit: &Circuit, size: usize, chunk: usize, relabel: bool) -> Vec<f64> {
    // Every rank holds the same gathered vector.
    run_ranks(size, |ctx| {
        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
        backend.set_exchange_chunk(chunk);
        backend.set_relabel(relabel);
        run_on(&mut backend, circuit)
            .expect("distributed run")
            .probabilities
            .expect("probabilities")
            .to_vec()
    })
    .swap_remove(0)
}

fn assert_loopback_matches(circuit: &Circuit, sizes: &[usize]) {
    let expected = reference_probs(circuit);
    for &relabel in &[true, false] {
        for &size in sizes {
            let actual = loopback_probs_with(circuit, size, usize::MAX, relabel);
            assert_eq!(
                expected.len(),
                actual.len(),
                "length mismatch at size {size} relabel {relabel}"
            );
            for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (e - a).abs() < TOL,
                    "size {size} relabel {relabel}: prob[{i}] expected {e}, got {a}"
                );
            }
        }
    }
}

/// Run `circuit` across simulated ranks and return rank 0's probabilities and
/// classical bits.
fn loopback_run_with(circuit: &Circuit, size: usize, relabel: bool) -> (Vec<f64>, Vec<bool>) {
    run_ranks(size, |ctx| {
        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
        backend.set_relabel(relabel);
        let out = run_on(&mut backend, circuit).expect("distributed run");
        let probs = out.probabilities.expect("probabilities").to_vec();
        (probs, out.classical_bits)
    })
    .swap_remove(0)
}

fn loopback_run(circuit: &Circuit, size: usize) -> (Vec<f64>, Vec<bool>) {
    loopback_run_with(circuit, size, true)
}

/// Assert that probabilities and classical bits are identical across all rank
/// counts. This checks measurement determinism across rank counts.
fn assert_loopback_deterministic(circuit: &Circuit, sizes: &[usize]) {
    for &relabel in &[true, false] {
        let (ref_probs, ref_bits) = loopback_run_with(circuit, sizes[0], relabel);
        for &size in &sizes[1..] {
            let (probs, bits) = loopback_run_with(circuit, size, relabel);
            assert_eq!(
                ref_bits, bits,
                "classical bits differ at size {size} relabel {relabel}"
            );
            assert_eq!(
                ref_probs.len(),
                probs.len(),
                "length differs at size {size} relabel {relabel}"
            );
            for (i, (e, a)) in ref_probs.iter().zip(probs.iter()).enumerate() {
                assert!(
                    (e - a).abs() < TOL,
                    "size {size} relabel {relabel}: prob[{i}] {e} vs {a} differ across ranks"
                );
            }
        }
    }
}

fn reference_probs(circuit: &Circuit) -> Vec<f64> {
    let mut backend = StatevectorBackend::new(SEED);
    run_on(&mut backend, circuit)
        .expect("statevector run")
        .probabilities
        .expect("probabilities")
        .to_vec()
}

fn distributed_serial_probs(circuit: &Circuit) -> Vec<f64> {
    let ctx = DistributedContext::serial();
    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
    run_on(&mut backend, circuit)
        .expect("distributed run")
        .probabilities
        .expect("probabilities")
        .to_vec()
}

fn assert_probs_match(circuit: &Circuit) {
    let expected = reference_probs(circuit);
    let actual = distributed_serial_probs(circuit);
    assert_eq!(expected.len(), actual.len(), "length mismatch");
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            (e - a).abs() < TOL,
            "prob[{i}] mismatch: expected {e}, got {a}"
        );
    }
}

#[test]
fn serial_matches_statevector_bell() {
    let mut b = CircuitBuilder::new(2);
    b.h(0).cx(0, 1);
    assert_probs_match(&b.build());
}

#[test]
fn serial_matches_statevector_rotations_and_entanglers() {
    let mut b = CircuitBuilder::new(4);
    b.h(0)
        .rx(0.37, 1)
        .ry(1.1, 2)
        .rz(-0.6, 3)
        .cx(0, 1)
        .cz(1, 2)
        .swap(2, 3)
        .t(0)
        .s(1);
    assert_probs_match(&b.build());
}

#[test]
fn serial_matches_statevector_ghz() {
    let n = 6;
    let mut b = CircuitBuilder::new(n);
    b.h(0);
    for q in 0..n - 1 {
        b.cx(q, q + 1);
    }
    assert_probs_match(&b.build());
}

#[test]
fn serial_export_statevector_matches() {
    let mut b = CircuitBuilder::new(3);
    b.h(0).cx(0, 1).ry(0.9, 2).cz(0, 2);
    let circuit = b.build();

    let mut sv = StatevectorBackend::new(SEED);
    sv.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let expected = sv.export_statevector().unwrap();

    let ctx = DistributedContext::serial();
    let mut dist = DistributedStatevectorBackend::new(ctx, SEED);
    dist.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    dist.apply_instructions(&circuit.instructions).unwrap();
    let actual = dist.export_statevector().unwrap();

    assert_eq!(expected.len(), actual.len());
    for (e, a) in expected.iter().zip(actual.iter()) {
        assert!((e - a).norm() < TOL);
    }
}

/// Relax the local qubit floor before constructing any distributed backend.
fn relax_min_local_qubits() {
    // SAFETY: set_var is unsafe because it can race concurrent getenv calls.
    // The supported runner (cargo nextest) executes each test in its own
    // process, so no other thread exists when this runs. Under plain
    // `cargo test` the call still precedes any distributed backend
    // construction in the same test, and the pre-existing getenv race with
    // unrelated tests is unchanged from the pre-edition-2024 code.
    unsafe { std::env::set_var("PRISM_DIST_MIN_LOCAL_QUBITS", "1") };
    assert_eq!(crate::distributed::min_local_qubits(), 1);
}

#[test]
fn loopback_global_hadamard_wall() {
    relax_min_local_qubits();
    let mut b = CircuitBuilder::new(4);
    for q in 0..4 {
        b.h(q);
    }
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_global_rotations_and_diagonals() {
    relax_min_local_qubits();
    let mut b = CircuitBuilder::new(4);
    b.h(0)
        .rx(0.7, 3)
        .ry(1.3, 2)
        .rz(-0.4, 3)
        .t(2)
        .s(3)
        .x(1)
        .h(2)
        .h(3);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_local_only_matches_across_ranks() {
    relax_min_local_qubits();
    let mut b = CircuitBuilder::new(5);
    b.h(0).cx(0, 1).cx(1, 2);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

// With 4 qubits across 4 ranks, qubits 0,1 are local and 2,3 are global, so a
// two qubit gate can place operands in every local and global combination.

fn spread_4q() -> CircuitBuilder {
    let mut b = CircuitBuilder::new(4);
    b.h(0).h(1).h(2).h(3);
    b
}

#[test]
fn loopback_cx_all_qubit_splits() {
    relax_min_local_qubits();
    for &(c, t) in &[(0usize, 1usize), (1, 2), (2, 0), (2, 3)] {
        let mut b = spread_4q();
        b.cx(c, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_cz_all_qubit_splits() {
    relax_min_local_qubits();
    for &(a, t) in &[(0usize, 1usize), (1, 3), (3, 0), (2, 3)] {
        let mut b = spread_4q();
        b.cz(a, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_swap_all_qubit_splits() {
    relax_min_local_qubits();
    for &(a, t) in &[(0usize, 1usize), (1, 2), (3, 0), (2, 3)] {
        let mut b = spread_4q();
        b.swap(a, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_swap_asymmetric_state() {
    relax_min_local_qubits();
    // Distinct rotation per qubit, so every marginal differs and a wrong
    // readout permutation cannot hide behind symmetric probabilities. A
    // uniform H wall would mask map bugs.
    let mut b = CircuitBuilder::new(4);
    b.rx(0.3, 0).rx(0.7, 1).rx(1.1, 2).rx(1.5, 3);
    b.swap(0, 3).swap(1, 2).swap(0, 1);
    b.ry(0.4, 3).cx(3, 0);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_gates_after_relabel_use_moved_qubits() {
    relax_min_local_qubits();
    // Repeated non-diagonal gates on global qubits force relabels and
    // evictions; later gates must follow the moved qubits through the map.
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.rx(0.2, 0).rx(0.5, 1).rx(0.9, 2);
    b.h(3).h(4);
    b.cx(3, 0).cz(4, 1).rzz(0.6, 2, 3);
    b.h(0).h(1);
    b.swap(2, 4).ry(0.8, 4).cx(4, 2);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_export_statevector_after_swap() {
    relax_min_local_qubits();
    // Export must reorder the gathered amplitudes back to circuit qubit order
    // after SWAPs leave the map permuted.
    let n = 4;
    let mut b = CircuitBuilder::new(n);
    b.rx(0.3, 0).ry(0.8, 1).rx(1.2, 2).t(3).h(3);
    b.swap(0, 3).swap(1, 2);
    b.rz(0.5, 3).ry(0.2, 0);
    let circuit = b.build();

    let mut sv = StatevectorBackend::new(SEED);
    sv.init(circuit.num_qubits, circuit.num_classical_bits)
        .unwrap();
    sv.apply_instructions(&circuit.instructions).unwrap();
    let expected = sv.export_statevector().unwrap();

    let results = run_ranks(4, |ctx| {
        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
        backend
            .init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        backend.apply_instructions(&circuit.instructions).unwrap();
        backend.export_statevector().unwrap()
    });
    for actual in &results {
        assert_eq!(expected.len(), actual.len());
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!((e - a).norm() < TOL, "amp[{i}] expected {e}, got {a}");
        }
    }
}

#[test]
fn loopback_measure_after_swap_reads_moved_qubit() {
    relax_min_local_qubits();
    // |0001> swapped across the boundary becomes |1000>: the top qubit must
    // read 1 and the bottom 0 through the permuted map.
    let n = 4;
    let mut b = CircuitBuilder::new_with_classical(n, 2);
    b.x(0).swap(0, n - 1);
    b.measure(n - 1, 0).measure(0, 1);
    let circuit = b.build();
    for &size in &[1usize, 2, 4] {
        let (probs, bits) = loopback_run(&circuit, size);
        assert!(bits[0], "size {size}: moved excitation must read 1");
        assert!(!bits[1], "size {size}: vacated qubit must read 0");
        let expected = 1usize << (n - 1);
        assert!(
            (probs[expected] - 1.0).abs() < TOL,
            "size {size}: state must be |1000>, got p[{expected}]={}",
            probs[expected]
        );
    }
}

#[test]
fn loopback_reset_after_swap_clears_moved_qubit() {
    relax_min_local_qubits();
    let n = 4;
    let mut b = CircuitBuilder::new(n);
    b.rx(0.4, 0).rx(0.9, 1).h(3);
    b.swap(0, 3);
    let mut circuit = b.build();
    circuit.add_reset(3);
    assert_loopback_matches(&circuit, &[1, 2, 4]);
}

#[test]
fn loopback_relabel_victim_starvation_falls_back() {
    relax_min_local_qubits();
    // At 4 ranks of a 4 qubit register only two positions are local. An Mcu
    // referencing every qubit leaves no eviction victim, so the direct global
    // exchange path must apply the gate.
    let x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let mut b = spread_4q();
    b.mcu(x, &[0, 1, 2], 3);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_rzz_all_qubit_splits() {
    relax_min_local_qubits();
    for &(a, t) in &[(0usize, 1usize), (1, 2), (2, 0), (2, 3)] {
        let mut b = spread_4q();
        b.rzz(0.85, a, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_cphase_all_qubit_splits() {
    relax_min_local_qubits();
    for &(c, t) in &[(0usize, 1usize), (1, 2), (2, 0), (2, 3)] {
        let mut b = spread_4q();
        b.cphase(0.6, c, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_controlled_unitary_global_target() {
    relax_min_local_qubits();
    let ry = |theta: f64| {
        let (s, c) = (theta / 2.0).sin_cos();
        [
            [Complex64::new(c, 0.0), Complex64::new(-s, 0.0)],
            [Complex64::new(s, 0.0), Complex64::new(c, 0.0)],
        ]
    };
    for &(c, t) in &[(1usize, 2usize), (2, 0), (2, 3)] {
        let mut b = spread_4q();
        b.cu(ry(0.9), c, t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_toffoli_mixed_splits() {
    relax_min_local_qubits();
    let x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    for &(c0, c1, t) in &[(0usize, 1usize, 2usize), (0, 2, 3), (2, 3, 0), (1, 2, 3)] {
        let mut b = spread_4q();
        b.mcu(x, &[c0, c1], t);
        assert_loopback_matches(&b.build(), &[1, 2, 4]);
    }
}

#[test]
fn loopback_mixed_circuit_qft_like() {
    relax_min_local_qubits();
    // H walls interleaved with controlled phases spanning all splits, plus an
    // entangling tail.
    let mut b = CircuitBuilder::new(4);
    b.h(0).h(1).h(2).h(3);
    b.cphase(0.5, 0, 2)
        .cphase(0.25, 1, 3)
        .cx(2, 1)
        .rzz(0.4, 0, 3)
        .swap(1, 2)
        .cz(0, 3);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

// These circuits are large enough to trigger the fusion pipeline: they meet
// `MIN_QUBITS_FOR_FUSION` (1q), `MIN_QUBITS_FOR_2Q_FUSION`,
// `MIN_QUBITS_FOR_MULTI_FUSION` (multi 1q), and `MIN_QUBITS_FOR_DIAG_BATCH`
// in `circuit::fusion`. The reference statevector fuses identically, so a
// match confirms the distributed backend decomposes each fused or batched
// variant correctly.

// Eight ranks put three qubits in the rank id, so the four-rank group
// `apply_2q_two_global` gathers is a strict subset of the world for the first
// time, and a gate can name more rank bits than that group holds.

#[test]
fn loopback_eight_ranks_fused_2q_on_two_rank_bits() {
    relax_min_local_qubits();
    let n = 12;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    b.ry(0.6, n - 2)
        .rz(0.3, n - 1)
        .cx(n - 2, n - 1)
        .ry(-0.4, n - 1);
    assert_loopback_matches(&b.build(), &[8]);
}

#[test]
fn loopback_eight_ranks_mcu_spans_three_rank_bits() {
    relax_min_local_qubits();
    let x = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let n = 6;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    b.rx(0.4, n - 1).mcu(x, &[n - 3, n - 2], n - 1);
    assert_loopback_matches(&b.build(), &[8]);
}

#[test]
fn serial_measured_circuit_matches_statevector_bit_for_bit() {
    // One rank shares the dense backend's measurement stream, so the outcomes
    // must agree exactly, not just in distribution.
    let n = 5;
    let mut b = CircuitBuilder::new_with_classical(n, n);
    b.h(0).ry(0.7, 1).h(2).rx(1.1, 3).h(4);
    for q in 0..n - 1 {
        b.cx(q, q + 1);
    }
    for q in 0..n {
        b.measure(q, q);
    }
    let mut circuit = b.build();
    circuit.add_reset(2);

    let mut sv = StatevectorBackend::new(SEED);
    let expected = run_on(&mut sv, &circuit).expect("statevector run");

    let mut dist = DistributedStatevectorBackend::new(DistributedContext::serial(), SEED);
    let actual = run_on(&mut dist, &circuit).expect("distributed run");

    assert_eq!(
        expected.classical_bits, actual.classical_bits,
        "single rank must reproduce the dense measurement outcomes"
    );
    let expected = expected.probabilities.expect("probabilities").to_vec();
    let actual = actual.probabilities.expect("probabilities").to_vec();
    assert_eq!(expected.len(), actual.len());
    for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!((e - a).abs() < 1e-12, "prob[{i}] expected {e}, got {a}");
    }
}

#[test]
fn loopback_fused_multifused_and_2q() {
    relax_min_local_qubits();
    // HEA pattern: rotation layers fuse into MultiFused, CX ladders into
    // Fused2q and Multi2q, reaching the global qubits at the top.
    let n = 14;
    let mut b = CircuitBuilder::new(n);
    for layer in 0..3 {
        for q in 0..n {
            b.ry(0.3 + 0.01 * (layer * n + q) as f64, q);
            b.rz(-0.2 + 0.02 * q as f64, q);
        }
        for q in 0..n - 1 {
            b.cx(q, q + 1);
        }
    }
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_fused_batchphase_qft() {
    relax_min_local_qubits();
    // Textbook QFT produces H walls plus controlled phase batches (BatchPhase)
    // and trailing swaps, spanning the full register including global qubits.
    let n = 12;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
        for (j, target) in (q + 1..n).enumerate() {
            let angle = std::f64::consts::PI / (1u64 << (j + 1)) as f64;
            b.cphase(angle, target, q);
        }
    }
    for q in 0..n / 2 {
        b.swap(q, n - 1 - q);
    }
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_fused_batchrzz_qaoa() {
    relax_min_local_qubits();
    // QAOA pattern: Rzz on every edge (fuse into BatchRzz at >= 16q) plus Rx
    // mixers (MultiFused), spanning global qubits.
    let n = 16;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    for round in 0..2 {
        for q in 0..n - 1 {
            b.rzz(0.7 + 0.01 * round as f64, q, q + 1);
        }
        for q in 0..n {
            b.rx(0.4, q);
        }
    }
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_fused_diagonal_batch() {
    relax_min_local_qubits();
    // Mixed diagonal run (cphase + rzz + diagonal 1q) at >= 16 qubits triggers
    // DiagonalBatch fusion; ensure the entries decompose across the boundary.
    let n = 16;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    for q in 0..n - 1 {
        b.cphase(0.3, q, q + 1);
        b.rzz(0.5, q, q + 1);
    }
    for q in 0..n {
        b.t(q);
        b.rz(0.15, q);
    }
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_fused_2q_both_global() {
    relax_min_local_qubits();
    // Adjacent 1q gates and a CX on the top pair fuse into a Fused2q spanning
    // both global qubits at 4 ranks, exercising the four rank gather path.
    let n = 12;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    b.ry(0.6, n - 2)
        .rz(0.3, n - 1)
        .cx(n - 2, n - 1)
        .ry(-0.4, n - 1);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_measure_deterministic_basis_state() {
    relax_min_local_qubits();
    // X on every qubit yields |1...1>, so every measurement must read 1.
    let n = 5;
    let mut b = CircuitBuilder::new_with_classical(n, n);
    for q in 0..n {
        b.x(q);
    }
    for q in 0..n {
        b.measure(q, q);
    }
    let circuit = b.build();
    for &size in &[1usize, 2, 4] {
        let (_probs, bits) = loopback_run(&circuit, size);
        assert_eq!(bits, vec![true; n], "size {size}: expected all one readout");
    }
}

#[test]
fn loopback_measure_determinism_across_ranks() {
    relax_min_local_qubits();
    // With a fixed seed the outcomes and post measurement probabilities must be
    // identical for every rank count (lockstep meas_rng plus Allreduce).
    let n = 6;
    let mut b = CircuitBuilder::new_with_classical(n, n);
    b.h(0).h(3).h(5);
    for q in 0..n - 1 {
        b.cx(q, q + 1);
    }
    b.measure(0, 0).measure(3, 3).measure(5, 5);
    assert_loopback_deterministic(&b.build(), &[1, 2, 4]);
}

#[test]
fn loopback_ghz_measure_correlated() {
    relax_min_local_qubits();
    // A GHZ state collapses to all zeros or all ones, so the measured qubits
    // must agree at every rank count.
    let n = 5;
    let mut b = CircuitBuilder::new_with_classical(n, 2);
    b.h(0);
    for q in 0..n - 1 {
        b.cx(q, q + 1);
    }
    b.measure(0, 0).measure(n - 1, 1);
    let circuit = b.build();
    for &size in &[1usize, 2, 4] {
        let (_probs, bits) = loopback_run(&circuit, size);
        assert_eq!(bits[0], bits[1], "size {size}: GHZ qubits must correlate");
    }
}

#[test]
fn loopback_reset_clears_global_qubit() {
    relax_min_local_qubits();
    // Resetting every qubit of a full superposition must yield |0...0>.
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    let mut circuit = b.build();
    for q in 0..n {
        circuit.add_reset(q);
    }
    for &size in &[1usize, 2, 4] {
        let (probs, _bits) = loopback_run(&circuit, size);
        assert!(
            (probs[0] - 1.0).abs() < TOL,
            "size {size}: reset should yield |0...0>, got p[0]={}",
            probs[0]
        );
    }
}

#[test]
fn loopback_reset_of_entangled_partner_matches_statevector() {
    relax_min_local_qubits();
    // Reset samples a branch, so every rank has to draw the same outcome from
    // the replicated measurement stream. A disagreement leaves ranks holding
    // halves of different branches, which the statevector comparison catches.
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.h(0);
    for q in 1..n {
        b.cx(0, q);
    }
    let mut circuit = b.build();
    circuit.add_reset(n - 1);

    let expected = reference_probs(&circuit);
    assert!(
        (expected[0] + expected[(1 << (n - 1)) - 1] - 1.0).abs() < TOL,
        "reset of a GHZ partner must land wholly on one branch, got {expected:?}"
    );
    assert_loopback_matches(&circuit, &[1, 2, 4]);
}

#[test]
fn loopback_reset_empty_zero_branch_matches_statevector() {
    relax_min_local_qubits();
    // Resetting a qubit that holds |1> has no |0> branch to sample: the
    // outcome is forced to 1, so the reset collapses there and flips it,
    // leaving every other qubit untouched.
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.x(0).x(n - 1);
    let mut circuit = b.build();
    circuit.add_reset(n - 1);

    let expected = reference_probs(&circuit);
    assert!(
        (expected[1] - 1.0).abs() < TOL,
        "reset from |1> must clear only the reset qubit and leave qubit 0 set"
    );
    assert_loopback_matches(&circuit, &[1, 2, 4]);
}

#[test]
fn loopback_conditional_on_global_measurement() {
    relax_min_local_qubits();
    // Measure a |1> qubit into a bit, then conditionally X another qubit. The
    // conditional must fire identically on every rank.
    let n = 4;
    let mut b = CircuitBuilder::new_with_classical(n, 1);
    b.x(n - 1);
    b.measure(n - 1, 0);
    b.conditional(
        crate::circuit::ClassicalCondition::BitIsOne(0),
        crate::gates::Gate::X,
        &[0],
    );
    let circuit = b.build();
    for &size in &[1usize, 2, 4] {
        let (probs, bits) = loopback_run(&circuit, size);
        assert!(bits[0], "size {size}: measured bit should be 1");
        let expected = 1usize | (1usize << (n - 1));
        assert!(
            (probs[expected] - 1.0).abs() < TOL,
            "size {size}: conditional X should set qubit 0"
        );
    }
}

#[test]
fn loopback_guarded_region_takes_the_same_branch_on_every_rank() {
    relax_min_local_qubits();
    // The region body measures and entangles, so a rank that disagreed on the
    // branch would diverge in both the state and the classical bits.
    let n = 4;
    let mut b = CircuitBuilder::new_with_classical(n, 2);
    b.x(n - 1);
    b.measure(n - 1, 0);
    b.guarded(crate::circuit::ClassicalCondition::BitIsOne(0), |body| {
        body.x(0).cx(0, 1).measure(1, 1);
    });
    let circuit = b.build();
    for &size in &[1usize, 2, 4, 8] {
        let (probs, bits) = loopback_run(&circuit, size);
        assert!(bits[0] && bits[1], "size {size}: both bits should be 1");
        let expected = 0b11usize | (1usize << (n - 1));
        assert!(
            (probs[expected] - 1.0).abs() < TOL,
            "size {size}: the region body should have run"
        );
    }
}

#[test]
fn loopback_tiled_exchange_matches_full() {
    relax_min_local_qubits();
    // Every chunk size must match the statevector reference, so tiling the
    // exchange preserves correctness.
    let n = 6;
    let mut b = CircuitBuilder::new(n);
    for q in 0..n {
        b.h(q);
    }
    b.rx(0.5, n - 1).ry(0.8, n - 2).h(n - 1).h(n - 2);
    let circuit = b.build();
    let expected = reference_probs(&circuit);

    // The local slice at 4 ranks is 16 amplitudes; these chunks span the tiling
    // boundaries from one element up to a single whole slice message. Both the
    // direct exchange and the relabel exchange honor the chunk size.
    for &relabel in &[true, false] {
        for &size in &[1usize, 2, 4] {
            for &chunk in &[1usize, 3, 16, 1 << 20] {
                let actual = loopback_probs_with(&circuit, size, chunk, relabel);
                assert_eq!(expected.len(), actual.len());
                for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                    assert!(
                        (e - a).abs() < TOL,
                        "size {size} chunk {chunk} relabel {relabel}: prob[{i}] expected {e}, got {a}"
                    );
                }
            }
        }
    }
}

#[test]
fn exchange_counters_track_communication() {
    relax_min_local_qubits();
    // A single rank keeps every qubit local, so no gate exchanges.
    let messages = run_ranks(1, |ctx| {
        let mut dist = DistributedStatevectorBackend::new(ctx, SEED);
        dist.init(4, 0).unwrap();
        dist.apply(&inst_h(3)).unwrap();
        dist.exchange_messages()
    });
    assert_eq!(messages[0], 0, "single rank never exchanges");

    // At 2 ranks qubit 3 is global and the local slice is 8 amplitudes. The
    // diagonal Z and Rz are free. Direct mode exchanges the full slice per H.
    // Relabel mode pays one half-slice exchange for the first H, after which
    // qubit 3 is local and the second H is free.
    let circuit = {
        let mut b = CircuitBuilder::new(4);
        b.z(3).rz(0.3, 3).h(3).h(3);
        b.build()
    };
    let direct = loopback_exchange_stats(&circuit, 2, usize::MAX, false);
    assert_eq!(
        direct,
        (2, 16),
        "two global H gates exchange the slice twice"
    );
    let relabeled = loopback_exchange_stats(&circuit, 2, usize::MAX, true);
    assert_eq!(relabeled, (1, 4), "one relabel moves half the slice once");
}

#[test]
fn relabel_makes_global_swap_free() {
    relax_min_local_qubits();
    let circuit = {
        let mut b = CircuitBuilder::new(5);
        b.x(0).swap(0, 4);
        b.build()
    };
    let direct = loopback_exchange_stats(&circuit, 2, usize::MAX, false);
    let relabeled = loopback_exchange_stats(&circuit, 2, usize::MAX, true);
    assert_eq!(direct.0, 1, "direct boundary swap exchanges once");
    assert_eq!(relabeled, (0, 0), "relabel swap is a map update");
}

#[test]
fn relabel_reduces_phased_exchange_volume() {
    relax_min_local_qubits();
    // Activity shifts from the bottom qubits to the top qubits, as after
    // fusion passes that concentrate work. Direct exchange pays for every
    // layer touching the global qubits; relabeling pays two half-slice moves
    // when the working set shifts and the remaining layers run locally.
    let n = 6;
    let mut b = CircuitBuilder::new(n);
    for q in 0..4 {
        b.ry(0.3 + 0.01 * q as f64, q);
    }
    b.cx(0, 1).cx(2, 3);
    for layer in 0..3 {
        for q in 2..n {
            b.ry(0.2 + 0.01 * (layer * n + q) as f64, q);
        }
        b.cx(2, 3).cx(3, 4).cx(4, 5);
    }
    let circuit = b.build();
    let direct = loopback_exchange_stats(&circuit, 4, usize::MAX, false);
    let relabeled = loopback_exchange_stats(&circuit, 4, usize::MAX, true);
    assert_eq!(
        relabeled,
        (2, 16),
        "two half-slice relabels cover every layer"
    );
    assert!(
        relabeled.1 < direct.1 / 4,
        "relabel volume {} should be far below direct volume {}",
        relabeled.1,
        direct.1
    );
}

#[test]
fn tiled_exchange_splits_messages_not_volume() {
    relax_min_local_qubits();
    let circuit = {
        let mut b = CircuitBuilder::new(5);
        b.h(4);
        b.build()
    };
    let full = loopback_exchange_stats(&circuit, 2, 1 << 20, false);
    let tiled = loopback_exchange_stats(&circuit, 2, 4, false);
    assert_eq!(full.0, 1, "full slice is one message");
    assert_eq!(tiled.0, 4, "16 amplitudes in chunks of 4 is four messages");
    assert_eq!(full.1, tiled.1, "total amplitudes exchanged is unchanged");

    let relabel_full = loopback_exchange_stats(&circuit, 2, 1 << 20, true);
    let relabel_tiled = loopback_exchange_stats(&circuit, 2, 4, true);
    assert_eq!(
        relabel_full,
        (1, 8),
        "relabel moves the half slice in one message"
    );
    assert_eq!(
        relabel_tiled,
        (2, 8),
        "8 amplitudes in chunks of 4 is two messages"
    );
}

#[test]
fn boundary_swap_direct_exchanges_half_the_slice() {
    relax_min_local_qubits();
    let circuit = {
        let mut b = CircuitBuilder::new(5);
        b.x(0).swap(0, 4);
        b.build()
    };
    let full = loopback_exchange_stats(&circuit, 2, usize::MAX, false);
    assert_eq!(
        full,
        (1, 8),
        "only the moving half of the 16-amplitude slice crosses"
    );
    let tiled = loopback_exchange_stats(&circuit, 2, 4, false);
    assert_eq!(tiled, (2, 8), "8 moving amplitudes in chunks of 4");
    assert_loopback_matches(&circuit, &[2, 4]);
}

#[test]
fn local_controls_cut_the_controlled_exchange_volume() {
    relax_min_local_qubits();
    let x_mat = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    // One local control selects half the slice.
    let single = {
        let mut b = CircuitBuilder::new(5);
        b.h(0).cx(0, 4);
        b.build()
    };
    let stats = loopback_exchange_stats(&single, 2, usize::MAX, false);
    assert_eq!(stats, (1, 8), "one control selects len/2 of the slice");

    // Two local controls select a quarter.
    let double = {
        let mut b = CircuitBuilder::new(5);
        b.h(0).h(1).mcu(x_mat, &[0, 1], 4);
        b.build()
    };
    let stats = loopback_exchange_stats(&double, 2, usize::MAX, false);
    assert_eq!(stats, (1, 4), "two controls select len/4 of the slice");
    let tiled = loopback_exchange_stats(&double, 2, 2, false);
    assert_eq!(tiled, (2, 4), "4 sublattice amplitudes in chunks of 2");

    assert_loopback_matches(&single, &[2, 4]);
    assert_loopback_matches(&double, &[2, 4]);
}

#[test]
fn diagonal_cu_with_global_target_is_communication_free() {
    relax_min_local_qubits();
    let diag = [
        [Complex64::cis(0.3), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::cis(0.7)],
    ];
    // The prep pays for x(4); the trailing h(0) makes the phase visible in
    // probabilities. The diagonal cu itself must add no exchange, so the
    // prep-only circuit and the full circuit report identical counters.
    let prep_only = {
        let mut b = CircuitBuilder::new(5);
        b.x(4).h(0).h(0);
        b.build()
    };
    let with_cu = {
        let mut b = CircuitBuilder::new(5);
        b.x(4).h(0).cu(diag, 0, 4).h(0);
        b.build()
    };
    for relabel in [false, true] {
        let base = loopback_exchange_stats(&prep_only, 2, usize::MAX, relabel);
        let full = loopback_exchange_stats(&with_cu, 2, usize::MAX, relabel);
        assert_eq!(
            base, full,
            "diagonal cu on a global target adds no exchange (relabel {relabel})"
        );
    }
    assert_loopback_matches(&with_cu, &[2, 4]);
}

#[test]
fn multi2q_star_shares_one_exchange_per_run() {
    relax_min_local_qubits();
    // Fan-out onto one global qubit, the syndrome-extraction shape: three
    // same-partner entries share a single exchange.
    let c = |re: f64| Complex64::new(re, 0.0);
    let z = c(0.0);
    let cx_mat = [
        [c(1.0), z, z, z],
        [z, c(1.0), z, z],
        [z, z, z, c(1.0)],
        [z, z, c(1.0), z],
    ];
    let cry_mat = |theta: f64| {
        let (sin, cos) = (theta / 2.0).sin_cos();
        [
            [c(1.0), z, z, z],
            [z, c(1.0), z, z],
            [z, z, c(cos), c(-sin)],
            [z, z, c(sin), c(cos)],
        ]
    };
    let mut circuit = Circuit::new(5, 0);
    {
        let gates = vec![(0, 4, cx_mat), (1, 4, cry_mat(0.3)), (2, 4, cry_mat(0.8))];
        for q in 0..3 {
            circuit.add_gate(crate::gates::Gate::H, &[q]);
        }
        circuit.add_gate(
            crate::gates::Gate::Multi2q(Box::new(crate::gates::Multi2qData { gates })),
            &[0, 1, 2, 4],
        );
    }
    let stats = loopback_exchange_stats(&circuit, 2, usize::MAX, false);
    assert_eq!(
        stats,
        (1, 16),
        "the three-entry run exchanges the slice once, not per entry"
    );
    assert_loopback_matches(&circuit, &[2, 4]);
}

fn inst_h(q: usize) -> crate::circuit::Instruction {
    crate::circuit::Instruction::Gate {
        gate: crate::gates::Gate::H,
        targets: crate::circuit::smallvec![q],
    }
}

/// Run `circuit` across `size` ranks with the given exchange chunk and relabel
/// setting; return rank 0's `(message_count, amplitude_count)`.
fn loopback_exchange_stats(
    circuit: &Circuit,
    size: usize,
    chunk: usize,
    relabel: bool,
) -> (u64, u64) {
    run_ranks(size, |ctx| {
        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
        backend.set_exchange_chunk(chunk);
        backend.set_relabel(relabel);
        backend
            .init(circuit.num_qubits, circuit.num_classical_bits)
            .unwrap();
        backend.apply_instructions(&circuit.instructions).unwrap();
        (backend.exchange_messages(), backend.exchange_amplitudes())
    })
    .swap_remove(0)
}

#[test]
fn reports_global_qubit_count_for_single_rank() {
    let ctx = DistributedContext::serial();
    let mut dist = DistributedStatevectorBackend::new(ctx, SEED);
    dist.init(5, 0).unwrap();
    assert_eq!(dist.num_qubits(), 5);
    assert!(dist.supports_fused_gates());
}

/// Run distributed multi-shot sampling across simulated ranks. Returns the
/// shots for every rank and the largest allgather block sent by any rank.
fn loopback_shots(
    circuit: &Circuit,
    size: usize,
    num_shots: usize,
) -> (Vec<Vec<Vec<bool>>>, usize) {
    run_ranks_max_gather(size, |ctx| {
        let kind = crate::sim::BackendKind::StatevectorDistributed { context: ctx };
        crate::sim::run_shots_with(kind, circuit, num_shots, SEED)
            .expect("distributed shots")
            .shots
    })
}

/// Sample shots through the production dense sampler, so the comparison
/// tests prove distributed sampling reproduces the real dense path.
fn dense_reference_shots(circuit: &Circuit, num_shots: usize) -> Vec<Vec<bool>> {
    let stripped = circuit.without_measurements();
    crate::sim::shots::sample_shots(
        &crate::sim::Probabilities::Dense(reference_probs(&stripped)),
        &circuit.measurement_map(),
        circuit.num_classical_bits,
        num_shots,
        SEED,
    )
}

#[test]
fn shots_terminal_uniform_match_dense_across_rank_counts() {
    relax_min_local_qubits();
    let mut circuit = Circuit::new(4, 4);
    for q in 0..4 {
        circuit.add_gate(crate::gates::Gate::H, &[q]);
    }
    for q in 0..4 {
        circuit.add_measure(q, q);
    }
    let expected = dense_reference_shots(&circuit, 64);
    for size in [1usize, 2, 4] {
        let (per_rank, _) = loopback_shots(&circuit, size, 64);
        for shots in &per_rank {
            assert_eq!(shots, &expected, "size {size}");
        }
    }
}

#[test]
fn shots_terminal_ghz_match_dense_across_rank_counts() {
    relax_min_local_qubits();
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    for q in 0..3 {
        circuit.add_gate(crate::gates::Gate::Cx, &[q, q + 1]);
    }
    for q in 0..4 {
        circuit.add_measure(q, q);
    }
    let expected = dense_reference_shots(&circuit, 100);
    let mut saw = [false, false];
    for shot in &expected {
        assert!(
            shot.iter().all(|&b| b == shot[0]),
            "GHZ shot must be uniform"
        );
        saw[shot[0] as usize] = true;
    }
    assert!(saw[0] && saw[1], "100 GHZ shots should hit both outcomes");
    for size in [1usize, 2, 4] {
        let (per_rank, _) = loopback_shots(&circuit, size, 100);
        for shots in &per_rank {
            assert_eq!(shots, &expected, "size {size}");
        }
    }
}

#[test]
fn shots_restore_relabeled_qubits_before_sampling() {
    relax_min_local_qubits();
    // The swap leaves q3 = 1. At 2 and 4 ranks, q3 is a rank bit, so relabeling
    // turns the swap into a map update and sampling must first restore the
    // relabeled qubits to their circuit positions.
    let mut circuit = Circuit::new(4, 4);
    circuit.add_gate(crate::gates::Gate::X, &[0]);
    circuit.add_gate(crate::gates::Gate::Swap, &[0, 3]);
    for q in 0..4 {
        circuit.add_measure(q, q);
    }
    for size in [1usize, 2, 4] {
        let (per_rank, _) = loopback_shots(&circuit, size, 8);
        for shots in &per_rank {
            for shot in shots {
                assert_eq!(shot, &vec![false, false, false, true], "size {size}");
            }
        }
    }
}

#[test]
fn shots_sample_without_dense_gather() {
    relax_min_local_qubits();
    let mut circuit = Circuit::new(6, 6);
    for q in 0..6 {
        circuit.add_gate(crate::gates::Gate::H, &[q]);
    }
    for q in 0..6 {
        circuit.add_measure(q, q);
    }
    let (per_rank, max_gather) = loopback_shots(&circuit, 4, 32);
    for shots in &per_rank {
        assert_eq!(shots, &per_rank[0], "shots must be identical on every rank");
    }
    assert_eq!(
        max_gather, 1,
        "terminal sampling must only gather one mass value per rank"
    );
}

#[test]
fn shots_mid_circuit_match_across_rank_counts() {
    relax_min_local_qubits();
    let mut circuit = Circuit::new(4, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_measure(0, 0);
    circuit.add_gate(crate::gates::Gate::Cx, &[0, 1]);
    circuit.add_measure(1, 1);
    let (reference, _) = loopback_shots(&circuit, 1, 20);
    let expected = &reference[0];
    let mut saw = [false, false];
    for shot in expected {
        assert_eq!(shot[0], shot[1], "copied bit must match the measured bit");
        saw[shot[0] as usize] = true;
    }
    assert!(
        saw[0] && saw[1],
        "20 fair coin shots should hit both outcomes"
    );
    for size in [2usize, 4] {
        let (per_rank, _) = loopback_shots(&circuit, size, 20);
        for shots in &per_rank {
            assert_eq!(shots, expected, "size {size}");
        }
    }
}

#[test]
fn noisy_shots_rejected_on_distributed_kind() {
    let mut circuit = Circuit::new(2, 2);
    circuit.add_gate(crate::gates::Gate::H, &[0]);
    circuit.add_measure(0, 0);
    let noise = crate::sim::noise::NoiseModel::uniform_depolarizing(&circuit, 0.01);
    let kind = crate::sim::BackendKind::StatevectorDistributed {
        context: DistributedContext::serial(),
    };
    let err = crate::sim::run_shots_with_noise(kind, &circuit, &noise, 10, SEED).unwrap_err();
    assert!(matches!(
        err,
        crate::error::PrismError::IncompatibleBackend { .. }
    ));
}

#[test]
fn shots_without_measurements_still_validate_configuration() {
    relax_min_local_qubits();
    // Three ranks is not a power of two; the error must surface even though a
    // circuit without measurements needs no execution to produce all false
    // shots. Every rank reaches the same rejection, so none blocks.
    let circuit = Circuit::new(4, 0);
    let failed = run_ranks(3, |ctx| {
        let kind = crate::sim::BackendKind::StatevectorDistributed { context: ctx };
        crate::sim::run_shots_with(kind, &circuit, 8, SEED).is_err()
    });
    assert!(
        failed.iter().all(|&f| f),
        "non power of two rank count must error"
    );
}

// A caller holding a `dyn Backend` must reach the rank-local sampler. Before
// the override it saw `supports_native_sampling() == false` and fell back to
// the dense probability vector.
#[test]
fn trait_sampling_draws_natively_across_rank_counts() {
    relax_min_local_qubits();
    let n = 4;
    let shots = 32;
    let mut circuit = Circuit::new(n, n);
    for q in 0..n {
        circuit.add_gate(crate::gates::Gate::H, &[q]);
    }
    circuit.add_gate(crate::gates::Gate::Cx, &[0, n - 1]);
    for q in 0..n {
        circuit.add_measure(q, q);
    }
    let stripped = circuit.without_measurements();
    let expected = dense_reference_shots(&circuit, shots);

    for size in [1usize, 2, 4] {
        let (per_rank, max_gather) = run_ranks_max_gather(size, |ctx| {
            let mut backend: Box<dyn Backend> =
                Box::new(DistributedStatevectorBackend::new(ctx, SEED));
            assert!(
                backend.supports_native_sampling(),
                "the trait must advertise the native sampler"
            );
            backend
                .init(stripped.num_qubits, stripped.num_classical_bits)
                .expect("init");
            backend
                .apply_instructions(&stripped.instructions)
                .expect("apply");
            let samples = backend
                .sample_basis_states(shots, SEED)
                .expect("native sampling");
            (0..shots)
                .map(|shot| (0..n).map(|q| samples.bit(shot, q)).collect::<Vec<bool>>())
                .collect::<Vec<Vec<bool>>>()
        });
        for shots in &per_rank {
            assert_eq!(shots, &expected, "size {size}");
        }
        assert!(
            max_gather <= 1,
            "size {size}: the trait sampler must not gather a dense block"
        );
    }
}

// Observables that put X or Y on a rank bit are the case needing an exchange;
// Z on a rank bit is a per-rank sign. The SWAP leaves the qubit map permuted,
// so the masks must be read through it.
#[test]
fn pauli_expectations_match_the_dense_route_across_rank_counts() {
    relax_min_local_qubits();
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.h(0).ry(0.7, 1).rx(1.1, 2).h(3).ry(-0.5, 4);
    b.cx(0, 4).cz(1, 3).swap(2, 4).cx(3, 2);
    b.rz(0.3, 4).ry(0.9, 0);
    let circuit = b.build();

    let observables = vec![
        vec![PauliTerm::z(0)],
        vec![PauliTerm::z(n - 1)],
        vec![PauliTerm::z(0), PauliTerm::z(n - 1)],
        vec![PauliTerm::x(n - 1)],
        vec![PauliTerm::y(n - 1)],
        vec![PauliTerm::x(n - 2), PauliTerm::y(n - 1)],
        vec![PauliTerm::x(0), PauliTerm::y(2), PauliTerm::z(n - 1)],
        vec![PauliTerm::y(1), PauliTerm::x(n - 1)],
    ];
    let expected =
        crate::sim::run_expectation_values(&circuit, &observables, SEED).expect("dense reference");

    for &relabel in &[true, false] {
        for size in [1usize, 2, 4] {
            let (per_rank, max_gather) = run_ranks_max_gather(size, |ctx| {
                let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                backend.set_relabel(relabel);
                backend
                    .init(circuit.num_qubits, circuit.num_classical_bits)
                    .expect("init");
                backend
                    .apply_instructions(&circuit.instructions)
                    .expect("apply");
                backend
                    .pauli_expectations(&observables)
                    .expect("distributed expectations")
            });
            for values in &per_rank {
                assert_eq!(values.len(), expected.len());
                for (i, (e, a)) in expected.iter().zip(values.iter()).enumerate() {
                    assert!(
                        (e - a).abs() < 1e-12,
                        "size {size} relabel {relabel}: observable {i} expected {e}, got {a}"
                    );
                }
            }
            assert!(
                max_gather <= 1,
                "size {size} relabel {relabel}: expectations must not gather a dense block"
            );
        }
    }

    // The public terminal rejected this backend outright before the override.
    for values in run_ranks(4, |ctx| {
        crate::simulate(&circuit)
            .distributed(ctx)
            .seed(SEED)
            .expectation_values(&observables)
            .expect("distributed expectation values")
    }) {
        for (i, (e, a)) in expected.iter().zip(values.iter()).enumerate() {
            assert!(
                (e - a).abs() < 1e-12,
                "observable {i} expected {e}, got {a}"
            );
        }
    }
}

// Marginals come from per-qubit Z expectations, so no rank builds the 2^n
// vector the dense route allgathers.
#[test]
fn loopback_marginals_match_the_dense_route_without_gathering() {
    relax_min_local_qubits();
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.h(0).ry(0.4, 1).rx(0.9, 2).h(3).ry(1.3, 4);
    b.cx(0, 1).cz(2, 4).swap(1, 4).cx(3, 0);
    let circuit = b.build();

    let expected = crate::sim::Probabilities::Dense(reference_probs(&circuit)).marginals();

    for size in [1usize, 2, 4] {
        let (per_rank, max_gather) = run_ranks_max_gather(size, |ctx| {
            crate::simulate(&circuit)
                .distributed(ctx)
                .seed(SEED)
                .marginals()
                .expect("distributed marginals")
                .into_vec()
        });
        for actual in &per_rank {
            assert_eq!(actual.len(), expected.len());
            for (q, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (e.0 - a.0).abs() < 1e-12 && (e.1 - a.1).abs() < 1e-12,
                    "size {size}: qubit {q} expected {e:?}, got {a:?}"
                );
            }
        }
        assert!(
            max_gather <= 1,
            "size {size}: marginals must not gather a dense block"
        );
    }
}

// A rank whose configuration differs from the others desynchronizes the
// collective sequence, which surfaces as a hang or as diverging measurement
// branches. `init` has to name the mismatch instead.
#[test]
fn mismatched_rank_configuration_is_rejected_at_init() {
    relax_min_local_qubits();
    let mut b = CircuitBuilder::new_with_classical(4, 1);
    b.h(0).cx(0, 3).measure(3, 0);
    let circuit = b.build();

    let odd_seed = run_ranks(2, |ctx| {
        let seed = if ctx.rank() == 1 { SEED + 1 } else { SEED };
        let mut backend = DistributedStatevectorBackend::new(ctx, seed);
        run_on(&mut backend, &circuit).map(|_| ()).unwrap_err()
    });
    let odd_relabel = run_ranks(2, |ctx| {
        let mut backend = DistributedStatevectorBackend::new(ctx.clone(), SEED);
        backend.set_relabel(ctx.rank() != 1);
        run_on(&mut backend, &circuit).map(|_| ()).unwrap_err()
    });

    for err in odd_seed.iter().chain(odd_relabel.iter()) {
        match err {
            crate::error::PrismError::BackendUnsupported { operation, .. } => assert!(
                operation.contains("must be identical on every rank"),
                "expected a configuration mismatch, got {operation}"
            ),
            other => panic!("expected a configuration mismatch, got {other:?}"),
        }
    }
}

// The override has to reproduce the route `apply_gate` takes for a one-qubit gate
// at every position of the local/global split, including the diagonal shortcut.
// The third matrix is a non-unitary jump branch, which is what the trajectory
// engine actually hands this method.
#[test]
fn loopback_apply_1q_matrix_all_qubit_splits() {
    relax_min_local_qubits();
    let n = 4;

    let dense = [
        [Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)],
        [Complex64::new(0.8, 0.0), Complex64::new(-0.6, 0.0)],
    ];
    let diagonal = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::from_polar(1.0, 0.7)],
    ];
    let jump = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.3, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ];

    let mut b = CircuitBuilder::new(n);
    b.rx(0.3, 0).ry(0.8, 1).rx(1.2, 2).t(3).h(3).cx(0, 2);
    let prep = b.build();

    for (label, matrix) in [("dense", dense), ("diagonal", diagonal), ("jump", jump)] {
        for target in 0..n {
            let mut sv = StatevectorBackend::new(SEED);
            sv.init(n, 0).unwrap();
            sv.apply_instructions(&prep.instructions).unwrap();
            sv.apply_1q_matrix(target, &matrix).unwrap();
            let expected = sv.export_statevector().unwrap();

            for &relabel in &[true, false] {
                for &size in &[1usize, 2, 4] {
                    let results = run_ranks(size, |ctx| {
                        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                        backend.set_relabel(relabel);
                        backend.init(n, 0).unwrap();
                        backend.apply_instructions(&prep.instructions).unwrap();
                        backend.apply_1q_matrix(target, &matrix).unwrap();
                        backend.export_statevector().unwrap()
                    });

                    for actual in results {
                        assert_eq!(expected.len(), actual.len());
                        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
                            assert!(
                                (e - a).norm() < TOL,
                                "{label} on q{target}, size {size}, relabel {relabel}: \
                                 amp[{i}] expected {e}, got {a}"
                            );
                        }
                    }
                }
            }
        }
    }
}
