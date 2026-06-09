//! MPI correctness check for the distributed state vector backend.
//!
//! Run under an MPI launcher; rank 0 compares the gathered distributed result
//! against a single process [`StatevectorBackend`] reference and exits nonzero on
//! mismatch so a test script can gate on the process exit code:
//!
//! ```text
//! . scripts\mpi-env.ps1
//! cargo build --example dist_mpi_check --features distributed-mpi
//! mpiexec -n 4 target\debug\examples\dist_mpi_check.exe
//! ```
//!
//! Requires the `distributed-mpi` feature. Without it the binary prints a note
//! and exits 0 so a plain `cargo build --examples` stays green.
//!
//! The worker thread uses a larger stack because debug SIMD kernels can exceed
//! the default MSVC main thread stack.

#[cfg(not(feature = "distributed-mpi"))]
fn main() {
    eprintln!("dist_mpi_check requires --features distributed-mpi; nothing to do.");
}

#[cfg(feature = "distributed-mpi")]
fn main() {
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(run)
        .expect("spawn worker thread");
    handle.join().expect("worker thread panicked");
}

#[cfg(feature = "distributed-mpi")]
fn run() {
    use prism_q::backend::distributed_statevector::DistributedStatevectorBackend;
    use prism_q::backend::statevector::StatevectorBackend;
    use prism_q::circuit::builder::CircuitBuilder;
    use prism_q::circuit::Circuit;
    use prism_q::distributed::DistributedContext;
    use prism_q::sim::run_on;

    const SEED: u64 = 42;
    const TOL: f64 = 1e-10;

    // Cover local entanglement, global one qubit gates, and two-qubit and
    // controlled gates whose operands span local and global qubits.
    fn build_circuit(num_qubits: usize, _local_qubits: usize) -> Circuit {
        let n = num_qubits;
        let hi = n - 1; // global on multi-rank runs
        let mid = n - 2; // global on 4+ ranks
        let mut b = CircuitBuilder::new(n);
        // Local entanglement on the bottom qubits.
        b.h(0).cx(0, 1).cx(1, 2);
        // One qubit gates on every qubit, including global targets.
        for q in 0..n {
            b.h(q);
        }
        b.rx(0.37, hi).t(hi).rz(-0.6, mid);
        // Two-qubit and controlled gates spanning the local/global boundary.
        b.cx(0, hi) // control local, target global
            .cx(hi, 0) // control global, target local
            .cz(1, mid) // diagonal across the boundary
            .rzz(0.4, 2, hi) // parity diagonal across the boundary
            .swap(0, hi) // swap across the boundary
            .cphase(0.5, mid, hi); // controlled phase, both global on 4 ranks
                                   // Fusion-triggering tail: rotation layers + a CX ladder fuse into
                                   // MultiFused / Fused2q, and the cphase ladder into BatchPhase, all
                                   // reaching the global qubits at the top of the register.
        for q in 0..n {
            b.ry(0.2 + 0.01 * q as f64, q).rz(0.1, q);
        }
        for q in 0..n - 1 {
            b.cx(q, q + 1);
        }
        for q in 0..n - 1 {
            b.cphase(0.3, q, n - 1);
        }
        b.build()
    }

    let ctx = DistributedContext::world().expect("MPI world init");
    let rank = ctx.rank();
    let size = ctx.size();
    assert!(size.is_power_of_two(), "rank count must be a power of two");
    let p = size.trailing_zeros() as usize;

    // 16 qubits crosses every fusion threshold (1q/2q/multi/diagonal-batch), so
    // the run exercises fused and batched gates spanning the global qubits.
    let num_qubits = 16;
    let local_qubits = num_qubits - p;
    let circuit = build_circuit(num_qubits, local_qubits);

    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
    let probs = run_on(&mut backend, &circuit)
        .expect("distributed run")
        .probabilities
        .expect("probabilities")
        .to_vec();

    if rank == 0 {
        let mut reference = StatevectorBackend::new(SEED);
        let expected = run_on(&mut reference, &circuit)
            .expect("reference run")
            .probabilities
            .expect("probabilities")
            .to_vec();

        assert_eq!(expected.len(), probs.len(), "length mismatch");
        let mut max_err = 0.0_f64;
        for (e, a) in expected.iter().zip(probs.iter()) {
            max_err = max_err.max((e - a).abs());
        }
        if max_err < TOL {
            println!("OK: {size} ranks, {num_qubits} qubits, max_err={max_err:.2e}");
        } else {
            eprintln!("FAIL: max_err={max_err:.3e} exceeds tol {TOL:.1e}");
            std::process::exit(1);
        }
    }
}
