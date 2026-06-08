use super::DistributedStatevectorBackend;
use crate::backend::statevector::StatevectorBackend;
use crate::backend::Backend;
use crate::circuit::builder::CircuitBuilder;
use crate::circuit::Circuit;
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
    // Pairwise mailboxes indexed by `sender * size + receiver`. Each holds a
    // FIFO queue of messages so repeated exchanges between the same pair stay
    // ordered. This models MPI point-to-point, which is independent of the
    // global barrier used by the collectives.
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
                mailbox: (0..size * size).map(|_| VecDeque::new()).collect(),
            }),
            cv: Condvar::new(),
        })
    }

    fn barrier(&self) {
        let mut st = self.state.lock().unwrap();
        let gen = st.generation;
        st.arrived += 1;
        if st.arrived == self.size {
            st.arrived = 0;
            st.generation = st.generation.wrapping_add(1);
            self.cv.notify_all();
        } else {
            while st.generation == gen {
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

    fn sendrecv_c64(&self, partner: usize, send: &[Complex64], recv: &mut [Complex64]) {
        debug_assert_eq!(send.len(), recv.len());
        let size = self.shared.size;
        let mut st = self.shared.state.lock().unwrap();
        // Deposit into self -> partner, then wait for partner -> self. Pairwise
        // and barrier-independent, so ranks that skip an exchange (a 0 control)
        // never block a partner: their partner skips the same exchange.
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

/// Run `circuit` across simulated ranks and return rank 0's probabilities.
fn loopback_probs(circuit: &Circuit, size: usize) -> Vec<f64> {
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
    for &size in sizes {
        let actual = loopback_probs(circuit, size);
        assert_eq!(
            expected.len(),
            actual.len(),
            "length mismatch at size {size}"
        );
        for (i, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (e - a).abs() < TOL,
                "size {size}: prob[{i}] expected {e}, got {a}"
            );
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
    std::env::set_var("PRISM_DIST_MIN_LOCAL_QUBITS", "1");
    assert_eq!(crate::distributed::min_local_qubits(), 1);
}

#[test]
fn loopback_global_hadamard_wall() {
    relax_min_local_qubits();
    // H on every qubit covers global exchange and combine.
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
    // Entangling chain confined to local qubits.
    let mut b = CircuitBuilder::new(5);
    b.h(0).cx(0, 1).cx(1, 2);
    assert_loopback_matches(&b.build(), &[1, 2, 4]);
}

// --- P2: two-qubit / controlled gates spanning global qubits ---
//
// With 4 qubits across 4 ranks, qubits 0,1 are local and 2,3 are global, so a
// two-qubit gate can place its operands in every local/global combination. A
// leading Hadamard wall spreads amplitude so every branch carries weight.

fn spread_4q() -> CircuitBuilder {
    let mut b = CircuitBuilder::new(4);
    b.h(0).h(1).h(2).h(3);
    b
}

#[test]
fn loopback_cx_all_qubit_splits() {
    relax_min_local_qubits();
    // (local,local), (local,global), (global,local), (global,global).
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
    // A non-phase controlled unitary (Ry(0.9) on the target) with control and
    // target in mixed local/global positions.
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
    // Two controls + target across local/global boundaries.
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
    // entangling tail. Exercises the full P2 path end to end.
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

#[test]
fn reports_global_qubit_count_for_single_rank() {
    let ctx = DistributedContext::serial();
    let mut dist = DistributedStatevectorBackend::new(ctx, SEED);
    dist.init(5, 0).unwrap();
    assert_eq!(dist.num_qubits(), 5);
    // Single rank: every qubit is local.
    assert!(dist.supports_fused_gates());
}
