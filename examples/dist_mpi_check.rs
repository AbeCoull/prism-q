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

    // Cover local entanglement and global one qubit gates.
    fn build_circuit(num_qubits: usize, _local_qubits: usize) -> Circuit {
        let mut b = CircuitBuilder::new(num_qubits);
        // Local entanglement on the bottom qubits.
        b.h(0).cx(0, 1).cx(1, 2);
        // Include global targets.
        for q in 0..num_qubits {
            b.h(q);
        }
        b.rx(0.37, num_qubits - 1)
            .t(num_qubits - 1)
            .rz(-0.6, num_qubits - 2);
        b.build()
    }

    let ctx = DistributedContext::world().expect("MPI world init");
    let rank = ctx.rank();
    let size = ctx.size();
    assert!(size.is_power_of_two(), "rank count must be a power of two");
    let p = size.trailing_zeros() as usize;

    let num_qubits = 6;
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
