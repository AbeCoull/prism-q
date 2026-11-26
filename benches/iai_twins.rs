//! Callgrind twins of wall-clock gate rows, counted to corroborate a timing regression.
//!
//! Each row replays one `circuits` row's input and timed body; `scripts/bench_gate_rows.txt`
//! pairs them. The counts carry no limit of their own: `scripts/bench_gate_verdict.py` reads
//! them to tell a change in executed work from a layout or host effect on the timing.

use iai_callgrind::{LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main};
use prism_q::circuit::Circuit;
use prism_q::sim::{self, run_expectation_values};
use prism_q::{BackendKind, NoiseModel, PauliTerm, circuits};
use std::hint::black_box;

mod common;
use common::{
    PAULI_PATH_EPSILON, PAULI_PATH_MAX_TERMS, PAULI_PATH_NOISE, SEED, mid_measured_chain,
    non_clifford_noise_circuit, run_shots_with, run_shots_with_noise, run_with,
    with_terminal_measurements,
};

fn measured_clifford(n: usize) -> Circuit {
    with_terminal_measurements(circuits::clifford_heavy_circuit(n, 10, SEED))
}

fn depolarized(circuit: Circuit, p: f64) -> (Circuit, NoiseModel) {
    let noise = NoiseModel::uniform_depolarizing(&circuit, p);
    (circuit, noise)
}

fn zz_chain(n: usize) -> (Circuit, Vec<Vec<PauliTerm>>) {
    let observables = (0..n - 1)
        .map(|q| vec![PauliTerm::z(q), PauliTerm::z(q + 1)])
        .collect();
    (circuits::hardware_efficient_ansatz(n, 2, SEED), observables)
}

fn brickwork_pairs(n: usize) -> (Circuit, Vec<Vec<PauliTerm>>) {
    let observables = (0..8)
        .map(|k| {
            let a = (k * n / 8) % n;
            let b = (a + n / 3 + 1) % n;
            vec![PauliTerm::z(a), PauliTerm::z(b)]
        })
        .collect();
    (circuits::brickwork_circuit(n, 4, SEED), observables)
}

#[library_benchmark]
#[bench::q32_20k(with_terminal_measurements(circuits::ghz_circuit(32)))]
fn sparse_sampling(circuit: Circuit) {
    black_box(run_shots_with(BackendKind::Sparse, &circuit, 20_000, SEED).unwrap());
}

#[library_benchmark]
#[bench::q18(zz_chain(18))]
fn expectation_pauli_sum((circuit, observables): (Circuit, Vec<Vec<PauliTerm>>)) {
    black_box(run_expectation_values(&circuit, &observables, 42).unwrap());
}

#[library_benchmark]
#[bench::brickwork_d4_q20(brickwork_pairs(20))]
fn auto_expectation((circuit, observables): (Circuit, Vec<Vec<PauliTerm>>)) {
    black_box(
        sim::simulate(&circuit)
            .backend(BackendKind::Auto)
            .seed(42)
            .expectation_values(&observables)
            .unwrap(),
    );
}

#[library_benchmark]
#[bench::q20(mid_measured_chain(20, 4))]
fn tn_midmeasure_chain(circuit: Circuit) {
    black_box(run_with(BackendKind::TensorNetwork, &circuit, 42).unwrap());
}

#[library_benchmark]
#[bench::q500_10k(measured_clifford(500))]
fn compiled_sampler_noiseless(circuit: Circuit) {
    black_box(prism_q::run_shots_compiled(&circuit, 10_000, SEED).unwrap());
}

#[library_benchmark]
#[bench::q500_10k(depolarized(measured_clifford(500), 0.001))]
fn compiled_sampler_noisy((circuit, noise): (Circuit, NoiseModel)) {
    black_box(prism_q::run_shots_noisy(&circuit, &noise, 10_000, SEED).unwrap());
}

#[library_benchmark]
#[bench::non_clifford_q12_512(depolarized(non_clifford_noise_circuit(12, 4), 0.001))]
fn trajectory_pauli((circuit, noise): (Circuit, NoiseModel)) {
    black_box(run_shots_with_noise(BackendKind::Auto, &circuit, &noise, 512, SEED).unwrap());
}

#[library_benchmark]
#[bench::q50(depolarized(circuits::hardware_efficient_ansatz(50, 2, SEED), PAULI_PATH_NOISE))]
fn pauli_path_noisy_expectation((circuit, noise): (Circuit, NoiseModel)) {
    let backend = BackendKind::PauliPath {
        epsilon: PAULI_PATH_EPSILON,
        max_terms: PAULI_PATH_MAX_TERMS,
    };
    black_box(
        sim::simulate(&circuit)
            .backend(backend)
            .noise(&noise)
            .seed(SEED)
            .expectation_values(&[vec![PauliTerm::z(0)]])
            .unwrap(),
    );
}

library_benchmark_group!(
    name = twins;
    benchmarks =
        sparse_sampling,
        expectation_pauli_sum,
        auto_expectation,
        tn_midmeasure_chain,
        compiled_sampler_noiseless,
        compiled_sampler_noisy,
        trajectory_pauli,
        pauli_path_noisy_expectation
);

// The same pins as `benches/iai.rs`: one Rayon thread and the development host's tile
// budget, so a count depends on the code and not on the runner.
main!(
    config = LibraryBenchmarkConfig::default()
        .env("RAYON_NUM_THREADS", "1")
        .env("PRISM_TILE_KB", "256");
    library_benchmark_groups = twins
);
