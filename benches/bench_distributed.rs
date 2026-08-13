//! Loopback distributed benchmarks: thread-transport rank pairs measuring
//! exchange-path and steady-state dispatch costs without an MPI environment.
//! Loopback exchanges move at memcpy speed, so these rows resolve packing,
//! copying, and per-gate dispatch cost rather than network latency.
//!
//! Runs with `cargo bench --bench bench_distributed --features "parallel
//! distributed bench-internal"`.

#![cfg(all(feature = "distributed", feature = "bench-internal"))]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use num_complex::Complex64;
use prism_q::DistributedStatevectorBackend;
use prism_q::backend::Backend;
use prism_q::circuit::{Circuit, fusion::fuse_circuit};
use prism_q::circuits;
use prism_q::distributed::loopback::run_ranks;
use prism_q::gates::{Gate, Multi2qData};

mod common;
use common::{SEED, configure_group, is_fast};

fn steady_sizes() -> &'static [usize] {
    if is_fast() { &[12] } else { &[16] }
}

fn direct_sizes() -> &'static [usize] {
    if is_fast() { &[14] } else { &[18] }
}

/// Fused QAOA behind one SWAP, so the qubit map is non-identity for every
/// batched payload: the steady state the permuted-map dispatch pays for.
fn bench_steady_state_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/steady_state_batched");
    configure_group(&mut group);

    for &n in steady_sizes() {
        let fused = fuse_circuit(&circuits::qaoa_circuit(n, 3, SEED), true).into_owned();
        let mut prefix = Circuit::new(n, 0);
        prefix.add_gate(Gate::Swap, &[0, n - 1]);
        group.bench_with_input(BenchmarkId::from_parameter(n), &fused, |b, fused| {
            b.iter(|| {
                let messages = run_ranks(2, |ctx| {
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend
                        .init(fused.num_qubits, fused.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&prefix.instructions).unwrap();
                    backend.apply_instructions(&fused.instructions).unwrap();
                    backend.exchange_messages()
                });
                std::hint::black_box(messages);
            });
        });
    }

    group.finish();
}

/// Boundary SWAPs in direct-exchange mode: the half-slice pack path.
fn bench_boundary_swap_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/boundary_swap_direct");
    configure_group(&mut group);

    for &n in direct_sizes() {
        let mut circuit = Circuit::new(n, 0);
        circuit.add_gate(Gate::X, &[0]);
        for _ in 0..8 {
            circuit.add_gate(Gate::Swap, &[0, n - 1]);
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                let amplitudes = run_ranks(2, |ctx| {
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend.set_relabel(false);
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    backend.exchange_amplitudes()
                });
                std::hint::black_box(amplitudes);
            });
        });
    }

    group.finish();
}

/// Controlled gates and a Multi2q star onto one global qubit in
/// direct-exchange mode: the sublattice pack and shared-run exchange paths.
fn bench_controlled_star_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/controlled_star_direct");
    configure_group(&mut group);

    let c64 = |re: f64| Complex64::new(re, 0.0);
    let z = c64(0.0);
    let x_mat = [[z, c64(1.0)], [c64(1.0), z]];
    let cx_mat = [
        [c64(1.0), z, z, z],
        [z, c64(1.0), z, z],
        [z, z, z, c64(1.0)],
        [z, z, c64(1.0), z],
    ];
    let cry_mat = |theta: f64| {
        let (sin, cos) = (theta / 2.0).sin_cos();
        [
            [c64(1.0), z, z, z],
            [z, c64(1.0), z, z],
            [z, z, c64(cos), c64(-sin)],
            [z, z, c64(sin), c64(cos)],
        ]
    };

    for &n in direct_sizes() {
        let top = n - 1;
        let mut circuit = Circuit::new(n, 0);
        for q in 0..3 {
            circuit.add_gate(Gate::H, &[q]);
        }
        circuit.add_gate(Gate::Cx, &[0, top]);
        circuit.add_gate(Gate::mcu(x_mat, 2), &[0, 1, top]);
        let gates = vec![
            (0, top, cx_mat),
            (1, top, cry_mat(0.3)),
            (2, top, cry_mat(0.8)),
        ];
        circuit.add_gate(
            Gate::Multi2q(Box::new(Multi2qData { gates })),
            &[0, 1, 2, top],
        );
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                let amplitudes = run_ranks(2, |ctx| {
                    let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                    backend.set_relabel(false);
                    backend
                        .init(circ.num_qubits, circ.num_classical_bits)
                        .unwrap();
                    backend.apply_instructions(&circ.instructions).unwrap();
                    backend.exchange_amplitudes()
                });
                std::hint::black_box(amplitudes);
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets =
        bench_steady_state_batched,
    bench_boundary_swap_direct,
    bench_controlled_star_direct
}
criterion_main!(benches);
