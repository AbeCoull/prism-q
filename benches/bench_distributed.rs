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
    if is_fast() { &[12] } else { &[16, 20] }
}

fn direct_sizes() -> &'static [usize] {
    if is_fast() { &[14] } else { &[18, 20] }
}

fn sample_sizes() -> &'static [usize] {
    if is_fast() { &[12] } else { &[20] }
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

/// A wall of non-diagonal one qubit gates on the global qubit in
/// direct-exchange mode: every gate exchanges the full slice and combines the
/// received half with its own.
fn bench_global_1q_wall(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/global_1q_wall");
    configure_group(&mut group);

    for &n in direct_sizes() {
        let top = n - 1;
        let mut circuit = Circuit::new(n, 0);
        for _ in 0..8 {
            circuit.add_gate(Gate::H, &[top]);
            circuit.add_gate(Gate::Rx(0.3), &[top]);
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

/// A wall of dense Fused2q gates on both rank bits at four ranks in
/// direct-exchange mode: the two step butterfly exchange for a two qubit gate
/// over two global qubits.
fn bench_fused_2q_two_global(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/fused_2q_two_global");
    configure_group(&mut group);

    let ry = |theta: f64| {
        let (sin, cos) = (theta / 2.0).sin_cos();
        [
            [Complex64::new(cos, 0.0), Complex64::new(-sin, 0.0)],
            [Complex64::new(sin, 0.0), Complex64::new(cos, 0.0)],
        ]
    };
    let rx = |theta: f64| {
        let (sin, cos) = (theta / 2.0).sin_cos();
        [
            [Complex64::new(cos, 0.0), Complex64::new(0.0, -sin)],
            [Complex64::new(0.0, -sin), Complex64::new(cos, 0.0)],
        ]
    };
    // Kronecker product of two rotations: every entry of the 4x4 is nonzero.
    let (a, b) = (ry(0.3), rx(0.7));
    let mut mat = [[Complex64::new(0.0, 0.0); 4]; 4];
    for (r, row) in mat.iter_mut().enumerate() {
        for (c, entry) in row.iter_mut().enumerate() {
            *entry = a[r >> 1][c >> 1] * b[r & 1][c & 1];
        }
    }

    for &n in direct_sizes() {
        let mut circuit = Circuit::new(n, 0);
        for _ in 0..8 {
            circuit.add_gate(Gate::Fused2q(Box::new(mat)), &[n - 2, n - 1]);
        }
        group.bench_with_input(BenchmarkId::from_parameter(n), &circuit, |b, circ| {
            b.iter(|| {
                let amplitudes = run_ranks(4, |ctx| {
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

/// Terminal shot sampling of a uniform state at two and four ranks: the
/// per-rank cumulative distribution, the owner draws, and the gather that
/// carries the sampled indices back to every rank.
fn bench_sample_indices(c: &mut Criterion) {
    let mut group = c.benchmark_group("distributed/sample_indices");
    configure_group(&mut group);

    const SHOTS: usize = 4096;
    for &n in sample_sizes() {
        let mut circuit = Circuit::new(n, 0);
        for q in 0..n {
            circuit.add_gate(Gate::H, &[q]);
        }
        for ranks in [2usize, 4] {
            let id = BenchmarkId::new(n.to_string(), format!("{ranks}ranks"));
            group.bench_with_input(id, &circuit, |b, circ| {
                b.iter(|| {
                    let indices = run_ranks(ranks, |ctx| {
                        let mut backend = DistributedStatevectorBackend::new(ctx, SEED);
                        backend.set_relabel(false);
                        backend
                            .init(circ.num_qubits, circ.num_classical_bits)
                            .unwrap();
                        backend.apply_instructions(&circ.instructions).unwrap();
                        backend.sample_state_indices(SHOTS, SEED).unwrap()
                    });
                    std::hint::black_box(indices);
                });
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_config();
    targets =
        bench_steady_state_batched,
    bench_boundary_swap_direct,
    bench_global_1q_wall,
    bench_controlled_star_direct,
    bench_fused_2q_two_global,
    bench_sample_indices
}
criterion_main!(benches);
