use super::DistributedStatevectorBackend;
use crate::backend::Backend;
use crate::backend::statevector::StatevectorBackend;
use crate::circuit::Circuit;
use crate::circuit::builder::CircuitBuilder;
use crate::distributed::{DistributedContext, RankComm};
use crate::sim::run_on;
use num_complex::Complex64;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

const SEED: u64 = 42;
const TOL: f64 = 1e-10;

/// Test transport that simulates ranks with threads.
///
/// This covers global gate exchange without an MPI launcher. Each rank runs the
/// same circuit and reaches collectives in the same order.
struct LoopbackShared {
    size: usize,
    state: Mutex<LoopbackState>,
    cv: Condvar,
}

struct LoopbackState {
    generation: u64,
    arrived: usize,
    cslots: Vec<Vec<Complex64>>,
    fslots: Vec<f64>,
    reduce: Vec<f64>,
    // Largest block sent by a rank to allgather. Shot sampling tests
    // assert this stays at one element, proving no dense gather happened.
    max_gather_block: usize,
    // Mailboxes indexed by `sender * size + receiver`. FIFO order matches MPI
    // sendrecv order and stays separate from collective barriers.
    mailbox: Vec<VecDeque<Vec<Complex64>>>,
}

impl LoopbackShared {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            size,
            state: Mutex::new(LoopbackState {
                generation: 0,
                arrived: 0,
                cslots: vec![Vec::new(); size],
                fslots: vec![0.0; size],
                reduce: Vec::new(),
                max_gather_block: 0,
                mailbox: (0..size * size).map(|_| VecDeque::new()).collect(),
            }),
            cv: Condvar::new(),
        })
    }

    fn barrier(&self) {
        let mut st = self.state.lock().unwrap();
        let arrival_generation = st.generation;
        st.arrived += 1;
        if st.arrived == self.size {
            st.arrived = 0;
            st.generation = st.generation.wrapping_add(1);
            self.cv.notify_all();
        } else {
            while st.generation == arrival_generation {
                st = self.cv.wait(st).unwrap();
            }
        }
    }
}

#[derive(Clone)]
struct LoopbackComm {
    shared: Arc<LoopbackShared>,
    rank: usize,
}

impl std::fmt::Debug for LoopbackComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopbackComm")
            .field("rank", &self.rank)
            .field("size", &self.shared.size)
            .finish()
    }
}

impl RankComm for LoopbackComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn size(&self) -> usize {
        self.shared.size
    }

    fn allgather_c64(&self, local: &[Complex64]) -> Vec<Complex64> {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.max_gather_block = st.max_gather_block.max(local.len());
            st.cslots[self.rank] = local.to_vec();
        }
        self.shared.barrier();
        let out = {
            let st = self.shared.state.lock().unwrap();
            st.cslots.iter().flat_map(|s| s.iter().copied()).collect()
        };
        self.shared.barrier();
        out
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<f64> {
        let as_c: Vec<Complex64> = local.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        self.allgather_c64(&as_c).iter().map(|c| c.re).collect()
    }

    fn allreduce_sum_f64(&self, value: f64) -> f64 {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.fslots[self.rank] = value;
        }
        self.shared.barrier();
        let sum = {
            let st = self.shared.state.lock().unwrap();
            st.fslots.iter().sum()
        };
        self.shared.barrier();
        sum
    }

    fn allreduce_sum_f64_slice(&self, values: &mut [f64]) {
        {
            let mut st = self.shared.state.lock().unwrap();
            if st.reduce.len() != values.len() {
                st.reduce = vec![0.0; values.len()];
            }
            for (acc, &v) in st.reduce.iter_mut().zip(values.iter()) {
                *acc += v;
            }
        }
        self.shared.barrier();
        {
            let st = self.shared.state.lock().unwrap();
            values.copy_from_slice(&st.reduce);
        }
        self.shared.barrier();
        if self.rank == 0 {
            let mut st = self.shared.state.lock().unwrap();
            st.reduce.clear();
        }
        self.shared.barrier();
    }

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        debug_assert_eq!(send.len(), recv.len());
        let size = self.shared.size;
        let mut st = self.shared.state.lock().unwrap();
        // Send to partner, then wait for partner to send back. Ranks that skip
        // an exchange do not block because their partner skips it too.
        st.mailbox[self.rank * size + partner].push_back(send.to_vec());
        self.shared.cv.notify_all();
        let inbox = partner * size + self.rank;
        loop {
            if let Some(msg) = st.mailbox[inbox].pop_front() {
                recv.copy_from_slice(&msg);
                return;
            }
            st = self.shared.cv.wait(st).unwrap();
        }
    }

    fn barrier(&self) {
        self.shared.barrier();
    }
}

/// Run `circuit` across simulated ranks with the given backend configuration
/// and return rank 0's probabilities.
fn loopback_probs_with(circuit: &Circuit, size: usize, chunk: usize, relabel: bool) -> Vec<f64> {
    let shared = LoopbackShared::new(size);
    let handles: Vec<_> = (0..size)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let ctx = DistributedContext::from_comm(Arc::new(comm));
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend.set_exchange_chunk(chunk);
                    backend.set_relabel(relabel);
                    run_on(&mut backend, &circuit)
                        .expect("distributed run")
                        .probabilities
                        .expect("probabilities")
                        .to_vec()
                })
                .expect("spawn rank thread")
        })
        .collect();
    let mut results: Vec<Vec<f64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    // Every rank holds the same gathered vector.
    let first = results.swap_remove(0);
    results.clear();
    first
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
    let shared = LoopbackShared::new(size);
    let handles: Vec<_> = (0..size)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let ctx = DistributedContext::from_comm(Arc::new(comm));
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend.set_relabel(relabel);
                    let out = run_on(&mut backend, &circuit).expect("distributed run");
                    let probs = out.probabilities.expect("probabilities").to_vec();
                    (probs, out.classical_bits)
                })
                .expect("spawn rank thread")
        })
        .collect();
    let mut results: Vec<(Vec<f64>, Vec<bool>)> =
        handles.into_iter().map(|h| h.join().unwrap()).collect();
    let first = results.swap_remove(0);
    results.clear();
    first
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

    let size = 4;
    let shared = LoopbackShared::new(size);
    let handles: Vec<_> = (0..size)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let ctx = DistributedContext::from_comm(Arc::new(comm));
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend
                        .init(circuit.num_qubits, circuit.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circuit.instructions).unwrap();
                    backend.export_statevector().unwrap()
                })
                .expect("spawn rank thread")
        })
        .collect();
    let results: Vec<Vec<Complex64>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
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

// These circuits are large enough to trigger the fusion pipeline (>= 10 qubits
// for 1q fusion, >= 12 for 2q, >= 14 for multi 1q, >= 16 for diagonal batch).
// The reference statevector fuses identically, so a match confirms the
// distributed backend decomposes each fused or batched variant correctly.

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
fn loopback_reset_empty_zero_branch_matches_statevector() {
    relax_min_local_qubits();
    // Statevector reset projects onto the |0> branch. If that branch is empty,
    // it reinitializes to |0...0>; distributed reset must not preserve other
    // qubits by flipping the |1> branch.
    let n = 5;
    let mut b = CircuitBuilder::new(n);
    b.x(0).x(n - 1);
    let mut circuit = b.build();
    circuit.add_reset(n - 1);

    let expected = reference_probs(&circuit);
    assert!(
        (expected[0] - 1.0).abs() < TOL,
        "statevector reset should reinitialize empty zero branch"
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
    let ctx = DistributedContext::from_comm(Arc::new(LoopbackComm {
        shared: LoopbackShared::new(1),
        rank: 0,
    }));
    let mut dist = DistributedStatevectorBackend::new(ctx, SEED);
    dist.init(4, 0).unwrap();
    dist.apply(&inst_h(3)).unwrap();
    assert_eq!(dist.exchange_messages(), 0, "single rank never exchanges");

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
    let shared = LoopbackShared::new(size);
    let handles: Vec<_> = (0..size)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let ctx = DistributedContext::from_comm(Arc::new(comm));
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend.set_exchange_chunk(chunk);
                    backend.set_relabel(relabel);
                    backend
                        .init(circuit.num_qubits, circuit.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circuit.instructions).unwrap();
                    (backend.exchange_messages(), backend.exchange_amplitudes())
                })
                .expect("spawn rank thread")
        })
        .collect();
    let mut results: Vec<(u64, u64)> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let first = results.swap_remove(0);
    results.clear();
    first
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
    let shared = LoopbackShared::new(size);
    let handles: Vec<_> = (0..size)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let ctx = DistributedContext::from_comm(Arc::new(comm));
                    let kind = crate::sim::BackendKind::StatevectorDistributed { context: ctx };
                    crate::sim::run_shots_with(kind, &circuit, num_shots, SEED)
                        .expect("distributed shots")
                        .shots
                })
                .expect("spawn rank thread")
        })
        .collect();
    let results: Vec<Vec<Vec<bool>>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let max_gather = shared.state.lock().unwrap().max_gather_block;
    (results, max_gather)
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
    // circuit without measurements needs no execution to produce all false shots.
    // The init failure happens before any collective, so ranks do not block.
    let circuit = Circuit::new(4, 0);
    let shared = LoopbackShared::new(3);
    let handles: Vec<_> = (0..3)
        .map(|rank| {
            let comm = LoopbackComm {
                shared: shared.clone(),
                rank,
            };
            let circuit = circuit.clone();
            std::thread::spawn(move || {
                let ctx = DistributedContext::from_comm(Arc::new(comm));
                let kind = crate::sim::BackendKind::StatevectorDistributed { context: ctx };
                crate::sim::run_shots_with(kind, &circuit, 8, SEED).is_err()
            })
        })
        .collect();
    for handle in handles {
        assert!(
            handle.join().unwrap(),
            "non power of two rank count must error"
        );
    }
}
