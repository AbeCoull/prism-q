//! Pool-ownership contract for [`prism_q::ThreadPool`]: a scoped run serves the
//! work from the caller's pool and leaves the process-wide pool as it found it.
//!
//! One test function, in order: the global-pool assertion needs the global pool
//! still unbuilt, and a second case in this binary would build it first.

#![cfg(feature = "parallel")]

mod common;

use common::SEED;
use num_complex::Complex64;
use prism_q::circuits::qft_circuit;
use prism_q::{StatevectorBackend, ThreadPool, run_on};

const QUBITS: usize = 16;
const SCOPED_THREADS: usize = 2;
const OUTER_THREADS: usize = 3;

fn amplitudes() -> Vec<(u64, u64)> {
    let mut backend = StatevectorBackend::new(SEED);
    run_on(&mut backend, &qft_circuit(QUBITS)).expect("qft run");
    backend
        .state_vector()
        .iter()
        .map(|a: &Complex64| (a.re.to_bits(), a.im.to_bits()))
        .collect()
}

#[test]
fn scoped_pool_serves_the_work_and_leaves_the_global_pool_alone() {
    let pool = ThreadPool::with_threads(SCOPED_THREADS).expect("scoped pool");
    assert_eq!(pool.num_threads(), SCOPED_THREADS);

    let (width, scoped) = pool.install(|| (rayon::current_num_threads(), amplitudes()));
    assert_eq!(width, SCOPED_THREADS);

    rayon::ThreadPoolBuilder::new()
        .num_threads(OUTER_THREADS)
        .build_global()
        .expect("the scoped run left the global pool unbuilt");
    let outer = rayon::current_num_threads();
    assert_eq!(outer, OUTER_THREADS);

    let global = amplitudes();
    let rescoped = pool.install(amplitudes);
    assert_eq!(rayon::current_num_threads(), outer);

    // Dense unitary evolution is bitwise at any width, so pool width is the only
    // difference between the three runs and none of them may show it.
    assert_eq!(scoped, global);
    assert_eq!(scoped, rescoped);
}
