//! Instruction-count rows under callgrind: one kernel per row on a single thread.
//!
//! Runs on Linux through `iai-callgrind-runner`; elsewhere the target only builds.
//! `benches/README.md` describes the CI gate these rows feed.

use iai_callgrind::{LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main};
use prism_q::BackendKind;
use prism_q::circuit::Circuit;
use prism_q::circuit::fusion::fuse_circuit;
use prism_q::circuits;
use std::hint::black_box;

mod common;
use common::{SEED, run_shots_with, run_with};

fn measured(mut circuit: Circuit) -> Circuit {
    circuit.num_classical_bits = circuit.num_qubits;
    for q in 0..circuit.num_qubits {
        circuit.add_measure(q, q);
    }
    circuit
}

#[library_benchmark]
#[bench::q12(circuits::random_circuit(12, 10, SEED))]
fn statevector_random_d10(circuit: Circuit) {
    black_box(run_with(BackendKind::Statevector, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::q14(circuits::qft_circuit(14))]
fn statevector_qft_textbook(circuit: Circuit) {
    black_box(run_with(BackendKind::Statevector, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::q12(circuits::quantum_volume_circuit(12, 12, SEED))]
fn statevector_qv(circuit: Circuit) {
    black_box(run_with(BackendKind::Statevector, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::q32_1k(measured(circuits::clifford_heavy_circuit(32, 10, SEED)))]
fn stabilizer_clifford_shots(circuit: Circuit) {
    black_box(run_shots_with(BackendKind::Stabilizer, &circuit, 1_000, SEED).unwrap());
}

#[library_benchmark]
#[bench::q100_1k(measured(circuits::clifford_heavy_circuit(100, 10, SEED)))]
fn compiled_sampler(circuit: Circuit) {
    black_box(prism_q::run_shots_compiled(&circuit, 1_000, SEED).unwrap());
}

#[library_benchmark]
#[bench::q8(circuits::random_circuit(8, 10, SEED))]
fn density_matrix_random_d10(circuit: Circuit) {
    black_box(run_with(BackendKind::DensityMatrix, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::q12(circuits::random_circuit(12, 10, SEED))]
fn mps_random_d10(circuit: Circuit) {
    black_box(run_with(BackendKind::Mps { max_bond_dim: 64 }, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::qaoa_q16(circuits::qaoa_circuit(16, 3, SEED))]
fn fuse_circuit_qaoa_l3(circuit: Circuit) {
    black_box(fuse_circuit(&circuit, true).into_owned());
}

library_benchmark_group!(
    name = kernels;
    benchmarks =
        statevector_random_d10,
        statevector_qft_textbook,
        statevector_qv,
        stabilizer_clifford_shots,
        compiled_sampler,
        density_matrix_random_d10,
        mps_random_d10,
        fuse_circuit_qaoa_l3
);

// One Rayon thread keeps the count a function of the code alone, and the tile
// budget is pinned to the development host's so fusion emits the same circuit on
// every runner.
main!(
    config = LibraryBenchmarkConfig::default()
        .env("RAYON_NUM_THREADS", "1")
        .env("PRISM_TILE_KB", "256");
    library_benchmark_groups = kernels
);
